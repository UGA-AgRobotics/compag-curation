"""Controlled lazy dispatch to source-backed domain handlers.

Importing this module is inert. Handler modules are resolved only after the
typed authorization boundary succeeds. Public QA inspects the registry
without crossing that boundary.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Callable

from .config import DomainConfig
from .contracts import (
    ContractError,
    ExecutionNotAuthorized,
    HandlerServices,
    directory_identity,
    directory_tree_digest,
    regular_file_snapshot,
)
from .plan import build_plan, select_variant
from .registry import resolve


def _authorize(config: DomainConfig) -> None:
    if os.environ.get("COMPAG_CONTROLLED_EXECUTION") != "YES":
        raise ExecutionNotAuthorized("controlled execution environment approval is absent")
    if config.execution.authorized is not True:
        raise ExecutionNotAuthorized("typed execution authorization is absent")


def resolve_handler(
    command: str,
    config: DomainConfig,
    variant: str | None = None,
) -> Callable[..., Any]:
    """Resolve one real handler after authorization, never during planning."""

    _authorize(config)
    selected = select_variant(command, config, variant)
    target = resolve(command, selected).handler_target
    module_name, symbol = target.split(":", 1)
    module = importlib.import_module(module_name)
    handler = getattr(module, symbol, None)
    if not callable(handler):
        raise RuntimeError("the resolved implementation handler is unavailable")
    return handler


def _notification_services(config: DomainConfig, services: object) -> HandlerServices | None:
    """Validate an enabled delivery boundary before a handler can do work."""

    if not config.notifications.enabled:
        return None
    common = getattr(services, "common", None)
    if not isinstance(common, HandlerServices):
        raise ContractError("enabled notifications require typed handler services")
    common.validate_call_shapes()
    if common.notification_sender is not None and not callable(
        getattr(common.notification_sender, "send", None)
    ):
        raise ContractError("enabled notifications require a callable notification sender")
    if common.notifier is not None and not callable(common.notifier):
        raise ContractError("enabled notifications require a callable notifier")
    if common.notification_sender is None and common.notifier is None:
        raise ContractError("enabled notifications require an injected notifier")
    return common


def _deliver_completion_notification(
    command: str,
    variant: str,
    config: DomainConfig,
    common: HandlerServices,
) -> None:
    """Deliver one generic completion only after a non-feature handler succeeds."""

    from .domain.rounds import build_round_notification, deliver_round_notification

    delivered = deliver_round_notification(
        build_round_notification(
            "COMPAG Curation",
            f"{command}/{variant} completed",
            finished_at=common.utc_now(),
        ),
        config.notifications,
        common,
    )
    if not delivered:
        raise ContractError("enabled completion notification was not delivered")


def _completion_output_snapshots(
    command: str,
    variant: str,
    config: DomainConfig,
) -> tuple[tuple[str, object], ...]:
    snapshots: list[tuple[str, object]] = []
    for artifact in build_plan(command, config, variant).outputs:
        path = Path(artifact.path)
        if artifact.kind == "file":
            snapshot: object = regular_file_snapshot(path)
        elif artifact.kind == "directory":
            snapshot = (directory_identity(path), directory_tree_digest(path))
        else:
            raise ContractError("completion output has an unsupported artifact kind")
        snapshots.append((artifact.name, snapshot))
    return tuple(snapshots)


def execute(
    command: str,
    config: DomainConfig,
    services: object,
    variant: str | None = None,
) -> object:
    """Invoke the selected typed handler after the external authorization gate."""

    selected = select_variant(command, config, variant)
    handler = resolve_handler(command, config, selected)
    notification_services = _notification_services(config, services)
    result = handler(config, services)
    if notification_services is not None and command != "extract-features":
        completion_outputs = _completion_output_snapshots(
            command,
            selected,
            config,
        )
        try:
            _deliver_completion_notification(
                command,
                selected,
                config,
                notification_services,
            )
        finally:
            if (
                _completion_output_snapshots(command, selected, config)
                != completion_outputs
            ):
                raise ContractError("completion notification changed a declared output")
    return result
