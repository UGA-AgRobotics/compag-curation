"""Compatibility imports for report computations and artifact rendering."""

from __future__ import annotations

from compag_curation.domain.reporting.analysis import (
    best_pca_pair,
    build_learning_curve,
    build_paper_tables,
    build_reviewed_testset,
    candidate_panel_statistics,
    classification_metrics,
    confusion_matrix_report,
    evaluate_model_scores,
    generate_local_sam_masks,
    hybrid_predictions,
    mask_iou,
    match_selected_masks_to_detections,
    merge_multiscale_masks,
    select_masks,
    similarity_probability_points,
    simulate_active_learning,
    touches_border,
    where_active_learning_looks,
)

__all__ = [
    "best_pca_pair",
    "build_learning_curve",
    "build_paper_tables",
    "build_reviewed_testset",
    "candidate_panel_statistics",
    "classification_metrics",
    "confusion_matrix_report",
    "evaluate_model_scores",
    "generate_local_sam_masks",
    "hybrid_predictions",
    "mask_iou",
    "match_selected_masks_to_detections",
    "merge_multiscale_masks",
    "select_masks",
    "similarity_probability_points",
    "simulate_active_learning",
    "touches_border",
    "where_active_learning_looks",
]
