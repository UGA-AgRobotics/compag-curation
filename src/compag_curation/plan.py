"""Resolve CLI variants to explicit domain APIs."""

from __future__ import annotations

import importlib
from typing import Callable

from .config import DomainConfig
from .contracts import WorkflowPlan
from .registry import DomainEntry, registry_snapshot, resolve, variants


def select_variant(command: str, config: DomainConfig, variant: str | None = None) -> str:
    configured = str(config.workflow(command)["variant"])
    selected = variant or configured
    if selected not in variants(command):
        raise ValueError("an explicit valid workflow variant is required")
    if selected != configured:
        raise ValueError("the requested variant must match the validated workflow configuration")
    return selected


def _load_plan_target(entry: DomainEntry) -> Callable[[DomainConfig, str | None], WorkflowPlan]:
    module_name, symbol = entry.plan_target.split(":", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, symbol, None)
    if not callable(target):
        raise RuntimeError("domain plan target is unavailable")
    return target


def build_plan(command: str, config: DomainConfig, variant: str | None = None) -> WorkflowPlan:
    selected = select_variant(command, config, variant)
    entry = resolve(command, selected)
    plan = _load_plan_target(entry)(config, selected)
    if plan.entry_point != entry.plan_target or plan.command != command or plan.variant != selected:
        raise RuntimeError("domain plan does not match its registry contract")
    configured = config.workflow(command)
    expected_inputs = entry.input_contract + tuple(
        name for name in entry.optional_input_contract if name in configured
    )
    if tuple(item.name for item in plan.inputs) != expected_inputs:
        raise RuntimeError("domain plan input contract differs from the registry")
    if tuple(item.name for item in plan.outputs) != entry.output_contract:
        raise RuntimeError("domain plan output contract differs from the registry")
    return plan


def registry_document() -> dict[str, object]:
    return {
        "schema": "compag-curation-domain-registry/v2",
        "commands": {
            command: {variant: entry.to_dict() for variant, entry in entries.items()}
            for command, entries in registry_snapshot().items()
        },
        "cell_wrapper_targets": 0,
        "plan_only_operational_targets": 0,
        "real_handler_targets": sum(len(entries) for entries in registry_snapshot().values()),
        "scientific_execution": False,
    }
