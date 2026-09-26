"""Compatibility imports for border/tiny evaluation algorithms."""

from __future__ import annotations

from compag_curation.domain.evaluation.border_tiny import (
    BorderTinyResult,
    BorderTinySettings,
    SliceMetrics,
    classify_border_tiny,
    compute_slice_metrics,
    evaluate_border_tiny_records,
    execute_border_tiny_evaluation,
    prepare_evaluation_rows,
    weighted_average_precision,
    weighted_precision_recall,
)

__all__ = [
    "BorderTinyResult",
    "BorderTinySettings",
    "SliceMetrics",
    "classify_border_tiny",
    "compute_slice_metrics",
    "evaluate_border_tiny_records",
    "execute_border_tiny_evaluation",
    "prepare_evaluation_rows",
    "weighted_average_precision",
    "weighted_precision_recall",
]
