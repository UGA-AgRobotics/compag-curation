"""Compatibility imports for typed dataset accounting."""

from __future__ import annotations

from compag_curation.domain.reporting.dataset import (
    DatasetSummary,
    DatasetSummaryRequest,
    build_dataset_summary,
)

__all__ = ["DatasetSummary", "DatasetSummaryRequest", "build_dataset_summary"]
