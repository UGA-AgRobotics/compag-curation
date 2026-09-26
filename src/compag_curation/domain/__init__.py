"""Domain API namespace with science-free import behavior."""

from __future__ import annotations

from typing import Any

__all__ = ["SafeSMOTE"]


def __getattr__(name: str) -> Any:
    """Resolve persisted estimators only after an explicit attribute request."""

    if name == "SafeSMOTE":
        from compag_curation.training.safe_smote import SafeSMOTE

        return SafeSMOTE
    raise AttributeError(name)
