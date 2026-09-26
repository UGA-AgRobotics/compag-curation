"""Construction helpers for inert plans and typed implementation handlers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..config import DomainConfig
from ..contracts import ArtifactRef, WorkflowPlan, WorkflowStep


def artifact(
    value: Mapping[str, Any],
    name: str,
    default_role: str,
    *,
    output: bool = False,
) -> ArtifactRef:
    required = value.get("required", not output)
    if not isinstance(required, bool):
        raise TypeError("artifact required flag must be boolean")
    return ArtifactRef(
        name=name,
        role=str(value.get("role", default_role)),
        path=str(value["path"]),
        sha256=value.get("sha256"),
        required=required,
        kind=str(value.get("kind", "file")),
    )


def make_plan(
    config: DomainConfig,
    command: str,
    variant: str,
    entry_point: str,
    inputs: Iterable[ArtifactRef],
    outputs: Iterable[ArtifactRef],
    steps: Iterable[WorkflowStep],
    sealed_defaults: Mapping[str, Any] | None = None,
) -> WorkflowPlan:
    defaults: dict[str, Any] = {"decision_threshold": 0.5}
    if sealed_defaults:
        defaults.update(sealed_defaults)
    return WorkflowPlan(
        command=command,
        variant=variant,
        entry_point=entry_point,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        steps=tuple(steps),
        policy=config.execution,
        notification=config.notifications,
        sealed_defaults=defaults,
    )
