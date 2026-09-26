"""Stable imports for the explicit round and source-state implementations.

Legacy callers imported this module, so it remains as a facade.  Execution
state now flows through typed requests/results in the owning domain modules.
"""

from __future__ import annotations

from compag_curation.domain.rounds import (
    RoundBootstrapRequest,
    RoundContext,
    RoundNotification,
    bootstrap_round,
    build_round_notification,
    deliver_round_notification,
)
from compag_curation.features.extraction import (
    ReviewMergeRequest,
    ReviewMergeResult,
    SplitRequest,
    SplitResult,
    freeze_or_create_splits,
    merge_review_labels_into_coco,
)


__all__ = [
    "ReviewMergeRequest",
    "ReviewMergeResult",
    "RoundBootstrapRequest",
    "RoundContext",
    "RoundNotification",
    "SplitRequest",
    "SplitResult",
    "bootstrap_round",
    "build_round_notification",
    "deliver_round_notification",
    "freeze_or_create_splits",
    "merge_review_labels_into_coco",
]
