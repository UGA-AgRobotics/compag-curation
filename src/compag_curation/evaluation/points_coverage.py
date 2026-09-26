"""Compatibility imports for exact coverage-v2/v3 orchestration."""

from __future__ import annotations

from compag_curation.domain.evaluation.coverage import (
    CoverageInputArtifact,
    CoverageRequest,
    CoverageResult,
    build_coverage_argv,
    coverage_environment,
    execute_coverage,
    write_coverage_environment,
)

__all__ = [
    "CoverageInputArtifact",
    "CoverageRequest",
    "CoverageResult",
    "build_coverage_argv",
    "coverage_environment",
    "execute_coverage",
    "write_coverage_environment",
]
