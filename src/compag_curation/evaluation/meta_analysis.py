"""Compatibility imports for record-oriented meta-analysis algorithms."""

from __future__ import annotations

from compag_curation.domain.evaluation.meta_analysis import (
    MetaAnalysisResult,
    MetaAnalysisServices,
    SparseLogisticModel,
    StdlibMetaAnalysisServices,
    build_override_snapshot,
    build_review_event_history,
    collect_review_logs,
    decisive_disagreements,
    evaluate_context_rules,
    execute_meta_analysis,
    fit_sparse_meta_model,
    match_features_by_click,
    merge_features_by_identity,
    normalize_binary_label,
    rank_disagreement_features,
    refine_click_matches,
    sweep_zone_features,
)

__all__ = [
    "MetaAnalysisResult",
    "MetaAnalysisServices",
    "SparseLogisticModel",
    "StdlibMetaAnalysisServices",
    "build_override_snapshot",
    "build_review_event_history",
    "collect_review_logs",
    "decisive_disagreements",
    "evaluate_context_rules",
    "execute_meta_analysis",
    "fit_sparse_meta_model",
    "match_features_by_click",
    "merge_features_by_identity",
    "normalize_binary_label",
    "rank_disagreement_features",
    "refine_click_matches",
    "sweep_zone_features",
]
