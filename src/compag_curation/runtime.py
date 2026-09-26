"""Controlled execution boundary retained for API compatibility.

Runtime consumers use typed plans, explicit artifact contracts, and injected
services. Notebook-state emulation is not part of the public package.
"""

from .contracts import (
    ContractError,
    ExecutionNotAuthorized,
    ExecutionPolicy,
    ExecutionResult,
    NotificationConfig,
    WorkflowPlan,
    execute_plan,
)

__all__ = [
    "ContractError",
    "ExecutionNotAuthorized",
    "ExecutionPolicy",
    "ExecutionResult",
    "NotificationConfig",
    "WorkflowPlan",
    "execute_plan",
]
