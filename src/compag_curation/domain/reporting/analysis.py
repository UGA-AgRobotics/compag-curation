"""Pure report computations extracted from the paper-figure source cells."""

from __future__ import annotations

from dataclasses import dataclass
from json import loads
from math import ceil, exp, inf, isfinite, nan, sqrt
from pathlib import Path
from random import Random
from re import match, search
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .mask_publication import (
    C0016MaskPublicationSettings,
    C0017MaskPublicationSettings,
    MaskPublicationResult,
    PrecomputedBooleanMask,
    publish_c0016_masks,
    publish_c0017_masks,
)


class ReportingContractError(ValueError):
    contract_name = "report-analysis"


@dataclass(frozen=True)
class ClassificationMetrics:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    average_precision: float
    roc_auc: float


@dataclass(frozen=True)
class PrecomputedModelScoreColumn:
    """One model score vector stored in the sealed feature table."""

    model_rel: str
    column: str


@dataclass(frozen=True)
class PrecomputedModelMetricRow:
    """One exact model-metrics row stored in the sealed intermediate manifest."""

    model_rel: str
    round: int | None
    pr_auc: float
    fixed_threshold: float | None
    best_threshold: float | None


@dataclass(frozen=True)
class CurveBandPoint:
    x: float
    best: float
    mean: float
    sigma: float


@dataclass(frozen=True)
class ModelCurveBand:
    kind: str
    best_model: str
    baseline: float
    points: tuple[CurveBandPoint, ...]


@dataclass(frozen=True)
class ModelScoreEvaluation:
    """C3 metrics for one explicitly declared precomputed score column."""

    model_rel: str
    score_column: str
    round: int | None
    candidate_metrics: ClassificationMetrics
    best_threshold_metrics: ClassificationMetrics
    group_column: str | None
    group_rows: int
    positive_groups: int
    group_average_precision: float | None
    group_roc_auc: float | None


@dataclass(frozen=True)
class PrecomputedModelCurveEvaluation:
    """Complete C3 continuation over caller-verified table rows."""

    label_column: str
    group_column: str | None
    fixed_threshold: float
    grid_steps: int
    maximum_models: int
    top_k_to_plot: int
    figure_inches: tuple[float, float]
    save_dpi: int
    paper_style: bool
    export_svg: bool
    plot_bars: bool
    row_count: int
    positive_rows: int
    model_metrics: tuple[ModelScoreEvaluation, ...]
    top_models: tuple[str, ...]
    precision_recall_band: ModelCurveBand | None
    roc_band: ModelCurveBand | None


@dataclass(frozen=True)
class PrecomputedHybridScoreColumns:
    """Columns for the sealed round-specific XGB/YOLO score continuation."""

    target_round: int = 92
    model_rel: str = ""
    xgb_probability: str = "prob"
    yolo_confidence: str = "yolo_conf"
    yolo_iou: str = "yolo_iou"
    group_precedence: tuple[str, ...] = ("root", "img_folder")


@dataclass(frozen=True)
class HybridPolicy:
    """Exact source defaults for basic and smart hybrid decisions."""

    xgb_threshold: float = 0.50
    yolo_threshold: float = 0.20
    yolo_iou_threshold: float = 0.60
    detector_threshold: float = 0.50
    missing_policy: str = "ignore"
    yolo_bias: float = 0.60
    smart_rules_enabled: bool = True
    rule_negative_xgb_maximum: float = 0.20
    rule_negative_yolo_maximum: float = 0.70
    rule_positive_yolo_minimum: float = 0.85
    rule_positive_xgb_minimum: float = 0.30
    rule_yolo_solo_minimum: float = 0.97
    xgb_force_enabled: bool = True
    xgb_force_positive_threshold: float | None = None
    xgb_force_negative_threshold: float = 0.10
    xgb_force_negative_yolo_minimum: float = 0.85


@dataclass(frozen=True)
class HybridPrediction:
    prediction: int
    fused_probability: float
    rule: str


@dataclass(frozen=True)
class HybridMetrics:
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float


@dataclass(frozen=True)
class HybridEvaluation:
    level: str
    variant: str
    group_column: str | None
    positive_predictions: int
    row_percentages: tuple[tuple[float, float], tuple[float, float]]
    metrics: HybridMetrics


@dataclass(frozen=True)
class PrecomputedConfusionMetrics:
    """Binary metrics shared by the static C11 and C12 continuations."""

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float


@dataclass(frozen=True)
class PrecomputedModelConfusionEvaluation:
    """Typed, process-free confusion result over one sealed score column."""

    source_cell: str
    target_round: int
    selected_round: int | None
    model_rel: str
    score_column: str
    selection_average_precision: float
    threshold_mode: str
    threshold: float
    threshold_source: str
    requested_level: str
    effective_level: str
    group_column: str | None
    row_count: int
    positive_rows: int
    counts: tuple[tuple[int, int], tuple[int, int]]
    row_normalized: tuple[tuple[float, float], tuple[float, float]]
    display_values: tuple[tuple[float, float], tuple[float, float]]
    display_annotations: tuple[tuple[str, str], tuple[str, str]]
    normalized_display: bool
    count_and_percent_annotations: bool
    reported_metric_names: tuple[str, ...]
    figure_inches: tuple[float, float]
    save_dpi: int
    paper_style: bool
    metrics: PrecomputedConfusionMetrics


@dataclass(frozen=True)
class ReviewedFolderStatistics:
    img_folder: str
    candidates: int
    reviewed: int
    percentage_reviewed: float
    positives: int
    positive_rate: float


@dataclass(frozen=True)
class ReviewedTestsetStatistics:
    candidates_total: int
    reviewed_labeled: int
    percentage_reviewed: float
    positives: int
    positive_rate: float
    border: int
    tiny: int
    border_or_tiny: int
    reviewed_border_or_tiny: int
    mode: str
    folders: tuple[str, ...]
    has_yolo_columns: bool


@dataclass(frozen=True)
class ReviewedTestsetBatch:
    pool_rows: tuple[Mapping[str, Any], ...]
    labeled_rows: tuple[Mapping[str, Any], ...]
    statistics: ReviewedTestsetStatistics
    folder_statistics: tuple[ReviewedFolderStatistics, ...]


@dataclass(frozen=True)
class LearningPoint:
    run: str
    round: int
    pr_auc: float
    smoothed_pr_auc: float
    selected: bool


@dataclass(frozen=True)
class ActiveLearningFocusRow:
    image: str
    candidate_id: int
    probability: float
    uncertain: bool
    conflict: bool
    selected: bool


@dataclass(frozen=True)
class ActiveLearningFocusEvaluation:
    threshold: float
    margin: float
    disagreement_delta: float
    disagreement_iou_minimum: float
    histogram_bins: int
    figure_inches: tuple[float, float]
    save_dpi: int
    pool_rows: int
    uncertain_rows: int
    uncertain_fraction: float
    conflict_rows: int
    conflict_fraction: float
    selected_rows: int
    selected_fraction: float
    rows: tuple[ActiveLearningFocusRow, ...]


@dataclass(frozen=True)
class PrecomputedActiveLearningScoreColumns:
    """Explicit C7 columns from one caller-verified score table."""

    model_rel: str
    probability: str
    image: str = "image"
    candidate_id: str = "id"
    yolo_confidence: str | None = None
    yolo_iou: str | None = None


@dataclass(frozen=True)
class PrecomputedActiveLearningFocusEvaluation:
    """C7 shortlist result without model loading or column inference."""

    columns: PrecomputedActiveLearningScoreColumns
    disagreement_enabled: bool
    paper_style: bool
    focus: ActiveLearningFocusEvaluation


@dataclass(frozen=True)
class ActiveLearningCurvePoint:
    strategy: str
    repeat: int
    budget: int
    pr_auc: float


@dataclass(frozen=True)
class ActiveLearningAggregatePoint:
    strategy: str
    budget: int
    mean_ap: float
    standard_deviation_ap: float
    repeats: int


@dataclass(frozen=True)
class ActiveLearningSimulationEvaluation:
    initial_size: int
    batch_size: int
    maximum_budget: int
    repeats: int
    holdout_fraction: float
    random_seed: int
    strategies: tuple[str, ...]
    figure_inches: tuple[float, float]
    save_dpi: int
    curve: tuple[ActiveLearningCurvePoint, ...]
    aggregate: tuple[ActiveLearningAggregatePoint, ...]


@dataclass(frozen=True)
class MaskRecord:
    segmentation: tuple[tuple[bool, ...], ...]
    area: int
    predicted_iou: float = nan
    stability_score: float = nan


@dataclass(frozen=True)
class PrecomputedMaskCandidate:
    """One mask candidate derived from the sealed multiscale result table."""

    candidate_id: int
    segmentation: tuple[tuple[bool, ...], ...]
    area: int
    area_fraction: float
    predicted_iou: float
    stability_score: float
    score: float
    source_scale: float
    bbox_xyxy: tuple[int, int, int, int]
    touching_border: bool


@dataclass(frozen=True)
class PrecomputedMaskBatch:
    """Typed C16/C17 continuation after SHA-bound mask generation."""

    width: int
    height: int
    candidates: tuple[PrecomputedMaskCandidate, ...]
    visualization_candidates: tuple[PrecomputedMaskCandidate, ...]
    score_grid: tuple[PrecomputedMaskCandidate, ...]
    iou_grid: tuple[PrecomputedMaskCandidate, ...]
    overlay_alpha: float
    overlay_seed: int
    score_grid_columns: int
    score_figure_inches: tuple[float, float]
    iou_grid_columns: int
    iou_figure_inches: tuple[float, float]


@dataclass(frozen=True)
class InsectMaskSelectionEvaluation:
    """C18 mask choice and publication-panel geometry from sealed masks."""

    minimum_area: int
    maximum_area_fraction: float
    border_margin: int
    visualization_maximum_area_fraction: float
    visualization_hide_border: bool
    crop_padding: float
    inset_padding: float
    all_masks_alpha: float
    chosen_mask_alpha: float
    figure_inches: tuple[float, float]
    save_dpi: int
    selected_candidate: PrecomputedMaskCandidate
    visualization_candidate_ids: tuple[int, ...]
    crop_bbox_xyxy: tuple[int, int, int, int]
    inset_bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class SamMaskRequest:
    image: Path
    checkpoint: Path
    repository: Path
    config: str
    points_per_side: int = 64
    points_per_batch: int = 512
    pred_iou_threshold: float = 0.80
    stability_threshold: float = 0.88
    crop_layers: int = 0
    crop_downscale: int = 2
    crop_overlap: float = 0.4
    maximum_masks: int | None = 150
    output_mode: str = "binary_mask"
    apply_postprocessing: bool = False
    multiscale: tuple[float, ...] = (0.67, 0.80, 1.00, 1.25)
    merge_iou: float = 0.75


@dataclass(frozen=True)
class CandidatePanelSettings:
    zoom_padding: float = 0.20
    ring_pixels: int = 5
    crop_padding: float = 0.10
    subsample_count: int = 2500
    predicted_iou_threshold: float = 0.80
    stability_threshold: float = 0.88
    minimum_l: float = 80.0
    minimum_delta_a: float = 0.0
    maximum_delta_b: float = 20.0
    minimum_elongation: float = 0.8
    maximum_elongation: float = 12.0
    minimum_solidity: float = 0.50
    light_low_threshold: float = 5.0
    light_high_threshold: float = 250.0
    maximum_low_clip_fraction: float = 0.10
    maximum_high_clip_fraction: float = 0.08
    maximum_l_standard_deviation: float = 40.0
    maximum_background_delta_l: float = 25.0
    pca_print_components: int = 4
    pca_scatter_max_masks: int = 200
    pca_scatter_alpha: float = 0.35


@dataclass(frozen=True)
class PrecomputedCandidatePanel:
    """One validated scientific panel intermediate from the sealed batch."""

    folder: str
    candidate_id: int
    zoom_width: int
    zoom_height: int
    bbox: tuple[int, int, int, int]
    mask_area: int
    ring_area: int
    predicted_iou: float
    stability_score: float
    mean_l: float
    mean_a: float
    mean_b: float
    background_a: float
    background_b: float
    delta_l_median: float
    low_clip_fraction: float
    high_clip_fraction: float
    l_standard_deviation: float
    area_norm: float
    extent: float
    elongation: float
    solidity: float
    aspect_ratio: float
    circularity: float
    touching_border: bool
    mask_a: tuple[float, ...]
    mask_b: tuple[float, ...]
    ring_a: tuple[float, ...]
    ring_b: tuple[float, ...]
    gradient_magnitude: tuple[float, ...]
    pca_components: tuple[float, ...]


@dataclass(frozen=True)
class PrecomputedPcaPoint:
    candidate_id: int
    pc1: float
    pc2: float
    selected: bool


@dataclass(frozen=True)
class SimilarityPcaPoint:
    candidate_id: int
    human_label: int | None
    probability: float
    similarity: float
    default_x: float
    default_y: float
    best_x: float
    best_y: float
    selected_ordinal: int | None
    selected_iou: float | None


@dataclass(frozen=True)
class SimilarityPcaEvaluation:
    probability_column: str
    similarity_column: str
    default_columns: tuple[str, str]
    best_columns: tuple[str, str]
    best_score: float
    searched_components: int
    probability_threshold: float
    negative_rows: int
    positive_rows: int
    unlabeled_rows: int
    points: tuple[SimilarityPcaPoint, ...]


@dataclass(frozen=True)
class CandidateMaskStatistics:
    area: int
    bbox_xyxy: tuple[int, int, int, int]
    centroid: tuple[float, float]
    ring_area: int
    zoom_width: int
    zoom_height: int
    predicted_iou: float
    stability_score: float
    quality_pass: bool
    touching_border: bool


@dataclass(frozen=True)
class CandidatePanelEvaluation:
    panel: PrecomputedCandidatePanel
    delta_a: float
    delta_b: float
    quality_pass: bool
    light_pass: bool
    light_failure_reasons: tuple[str, ...]
    color_pass: bool
    color_failure_reasons: tuple[str, ...]
    shape_pass: bool
    shape_failure_reasons: tuple[str, ...]
    pca_projection_available: bool
    selected_in_pca_scatter: bool
    gradient_mean: float
    crop_padding: float
    displayed_pca_components: tuple[float, ...]


class SamMaskGenerator(Protocol):
    """Injected local-only SAM2 service; no implicit asset acquisition is allowed."""

    def generate(self, request: SamMaskRequest) -> Sequence[Mapping[str, Any]]: ...


class ProbabilityModel(Protocol):
    def fit(self, features: Sequence[Sequence[float]], labels: Sequence[int]) -> None: ...

    def predict_probability(self, features: Sequence[Sequence[float]]) -> Sequence[float]: ...


class CandidatePanelFeatureService(Protocol):
    """Injected feature service for the source's gate, ring and PCA panels."""

    def compute(
        self,
        mask: MaskRecord,
        settings: CandidatePanelSettings,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PrecomputedReportAnalysisRequest:
    """Typed table inputs for the report-only analysis continuation."""

    feature_rows: tuple[Mapping[str, str], ...]
    detection_rows: tuple[Mapping[str, str], ...]
    decision_threshold: float = 0.5
    active_learning_seed: int = 0
    candidate_panels: tuple[PrecomputedCandidatePanel, ...] = ()
    pca_scatter: tuple[PrecomputedPcaPoint, ...] = ()
    model_score_columns: tuple[PrecomputedModelScoreColumn, ...] = ()
    model_metric_rows: tuple[PrecomputedModelMetricRow, ...] = ()
    hybrid_score_columns: PrecomputedHybridScoreColumns = (
        PrecomputedHybridScoreColumns()
    )
    mode: str = "xgb_recall"


@dataclass(frozen=True)
class ReportOperationResult:
    operation: str
    output_role: str
    value: int | float | str | None


@dataclass(frozen=True)
class PrecomputedReportAnalysisResult:
    reviewed_rows: int
    labeled_rows: int
    evaluated_models: int
    active_learning_rows: int
    active_learning_focus: ActiveLearningFocusEvaluation
    active_learning_simulation: ActiveLearningSimulationEvaluation
    hybrid_positive_rows: int
    mask_rows: int
    selected_mask_rows: int
    selected_detection_rows: int
    similarity_rows: int
    precision_recall_area: float | None
    best_pca_columns: tuple[str, str] | None
    similarity_pca_evaluation: SimilarityPcaEvaluation
    reviewed_testset: ReviewedTestsetBatch
    candidate_panel_settings: CandidatePanelSettings
    candidate_panels: tuple[CandidatePanelEvaluation, ...]
    pca_scatter: tuple[PrecomputedPcaPoint, ...]
    precision_recall_band: ModelCurveBand | None
    roc_band: ModelCurveBand | None
    model_curve_evaluation: PrecomputedModelCurveEvaluation
    active_learning_score_evaluation: PrecomputedActiveLearningFocusEvaluation
    source_confusion_evaluations: tuple[PrecomputedModelConfusionEvaluation, ...]
    model_confusion_evaluations: tuple[HybridEvaluation, ...]
    hybrid_evaluations: tuple[HybridEvaluation, ...]
    mask_candidate_batch: PrecomputedMaskBatch
    c0016_mask_publication: MaskPublicationResult
    c0017_mask_publication: MaskPublicationResult
    insect_mask_selection: InsectMaskSelectionEvaluation
    executed_operations: tuple[str, ...]
    operation_results: tuple[ReportOperationResult, ...]


ReportParameter = str | int | float | bool | tuple[float, ...] | tuple[str, ...]


@dataclass(frozen=True)
class ReportOperation:
    """One executable, typed node in the report workflow program."""

    operation: str
    source_cells: tuple[str, ...]
    input_roles: tuple[str, ...]
    output_role: str
    parameters: tuple[tuple[str, ReportParameter], ...] = ()


def build_report_program(
    request: PrecomputedReportAnalysisRequest,
) -> tuple[ReportOperation, ...]:
    """Build the ordered domain program executed over sealed report tables."""

    threshold = request.decision_threshold
    seed = request.active_learning_seed
    panel = CandidatePanelSettings()
    return (
        ReportOperation(
            "build-reviewed-testset",
            ("NB-LIVE-0012-C0002", "NB-LIVE-0012-C0014"),
            ("sealed_labeled_feature_rows",),
            "reviewed_rows",
            (
                ("mode", request.mode),
                ("tile_size", 512),
                ("border_pixels", 4),
                ("tiny_minimum_dimension", 32),
                ("tiny_maximum_area", 1500),
            ),
        ),
        ReportOperation(
            "evaluate-precomputed-model-scores",
            ("NB-LIVE-0012-C0000", "NB-LIVE-0012-C0003"),
            ("reviewed_rows", "precomputed_model_score_columns"),
            "model_metrics",
            (
                ("fixed_threshold", threshold),
                ("curve_grid_steps", 401),
                ("population_max_models", 250),
                ("top_k_to_plot", 8),
                ("figure_inches", (3.35, 3.00)),
                ("save_dpi", 600),
                ("paper_style", True),
                ("export_svg", False),
                ("plot_bars", True),
            ),
        ),
        ReportOperation(
            "build-learning-curves",
            ("NB-LIVE-0012-C0005", "NB-LIVE-0012-C0006"),
            ("feature_rows", "detection_rows"),
            "learning_curve",
            (("start_round", 60), ("smoothing_window", 11)),
        ),
        ReportOperation(
            "select-active-learning-focus",
            ("NB-LIVE-0012-C0007",),
            ("feature_rows", "detection_rows"),
            "active_learning_focus",
            (
                ("decision_threshold", threshold),
                ("margin", 0.20),
                ("disagreement_delta", 0.60),
                ("disagreement_iou_minimum", 0.60),
                ("histogram_bins", 40),
                ("figure_inches", (4.6, 3.1)),
                ("save_dpi", 600),
            ),
        ),
        ReportOperation(
            "simulate-active-learning",
            ("NB-LIVE-0012-C0008",),
            ("reviewed_rows", "sha_bound_precomputed_probability"),
            "active_learning_curve",
            (
                ("strategies", ("uncertainty", "random")),
                ("seed", seed),
                ("initial_size", 100),
                ("batch_size", 50),
                ("maximum_budget", 2000),
                ("repeats", 10),
                ("holdout_fraction", 0.25),
                ("figure_inches", (4.8, 3.2)),
                ("save_dpi", 600),
            ),
        ),
        ReportOperation(
            "build-paper-tables",
            ("NB-LIVE-0012-C0009", "NB-LIVE-0012-C0010"),
            ("model_metrics",),
            "paper_tables",
            (("mode", "precomputed"),),
        ),
        ReportOperation(
            "compute-confusion-matrices",
            (
                "NB-LIVE-0012-C0011",
                "NB-LIVE-0012-C0012",
                "NB-LIVE-0012-C0015",
            ),
            (
                "reviewed_rows",
                "precomputed_model_score_columns",
                "precomputed_model_metric_rows",
                "precomputed_hybrid_score_columns",
            ),
            "confusion_metrics",
            (
                ("target_round", 92),
                ("threshold_mode", "fixed"),
                ("decision_threshold", threshold),
                ("level", "candidate"),
                ("normalize", False),
            ),
        ),
        ReportOperation(
            "integrate-precision-recall",
            ("NB-LIVE-0012-C0000",),
            ("model_metrics",),
            "precision_recall_area",
            (("threshold_steps", 101),),
        ),
        ReportOperation(
            "compute-hybrid-predictions",
            ("NB-LIVE-0012-C0015",),
            ("feature_rows", "precomputed_hybrid_score_columns"),
            "hybrid_confusion_metrics",
            (
                ("target_round", request.hybrid_score_columns.target_round),
                ("xgb_threshold", threshold),
                ("yolo_threshold", 0.20),
                ("yolo_iou_threshold", 0.60),
                ("detector_threshold", 0.50),
                ("missing_policy", "ignore"),
                ("yolo_bias", 0.60),
                ("smart_rules_enabled", True),
                ("rule_negative_xgb_maximum", 0.20),
                ("rule_negative_yolo_maximum", 0.70),
                ("rule_positive_yolo_minimum", 0.85),
                ("rule_positive_xgb_minimum", 0.30),
                ("rule_yolo_solo_minimum", 0.97),
                ("xgb_force_enabled", True),
                ("xgb_force_positive_threshold", 0.50),
                ("xgb_force_negative_threshold", 0.10),
                ("xgb_force_negative_yolo_minimum", 0.85),
                ("candidate_level", True),
                ("image_level", True),
                ("group_precedence", request.hybrid_score_columns.group_precedence),
            ),
        ),
        ReportOperation(
            "consume-precomputed-mask-batches",
            (
                "NB-LIVE-0012-C0016",
                "NB-LIVE-0012-C0017",
                "NB-LIVE-0012-C0018",
                "NB-LIVE-0012-C0019",
                "NB-LIVE-0012-C0020",
                "NB-LIVE-0012-C0021",
            ),
            ("detection_rows",),
            "mask_records",
            (
                ("predicted_iou_threshold", 0.80),
                ("stability_threshold", 0.88),
                ("multiscale", (0.67, 0.80, 1.00, 1.25)),
                ("merge_iou", 0.75),
                ("maximum_masks", 150),
                ("border_margin", 2),
                ("crop_padding", 0.10),
                ("maximum_area_fraction_for_visualization", 0.35),
                ("score_grid_count", 24),
                ("score_grid_columns", 6),
                ("score_figure_inches", (18.0, 10.0)),
                ("iou_grid_count", 6),
                ("iou_grid_columns", 3),
                ("iou_figure_inches", (14.0, 8.0)),
                ("overlay_alpha", 0.45),
                ("overlay_seed", 3),
                ("insect_minimum_area", 80),
                ("insect_maximum_area_fraction", 0.15),
                ("insect_border_margin", 2),
                ("publication_visualization_maximum_area_fraction", 0.25),
                ("publication_visualization_hide_border", True),
                ("publication_crop_padding", 0.10),
                ("publication_inset_padding", 0.15),
                ("publication_all_masks_alpha", 0.45),
                ("publication_chosen_mask_alpha", 0.45),
                ("publication_figure_inches", (10.0, 10.0)),
                ("publication_save_dpi", 200),
            ),
        ),
        ReportOperation(
            "compute-candidate-panels",
            ("NB-LIVE-0012-C0022", "NB-LIVE-0012-C0023"),
            ("precomputed_candidate_panels", "precomputed_pca_scatter"),
            "candidate_panel_statistics",
            (
                ("zoom_padding", panel.zoom_padding),
                ("ring_pixels", panel.ring_pixels),
                ("crop_padding", panel.crop_padding),
                ("subsample_count", panel.subsample_count),
                ("predicted_iou_threshold", panel.predicted_iou_threshold),
                ("stability_threshold", panel.stability_threshold),
                ("minimum_l", panel.minimum_l),
                ("minimum_delta_a", panel.minimum_delta_a),
                ("maximum_delta_b", panel.maximum_delta_b),
                ("minimum_elongation", panel.minimum_elongation),
                ("maximum_elongation", panel.maximum_elongation),
                ("minimum_solidity", panel.minimum_solidity),
                ("light_low_threshold", panel.light_low_threshold),
                ("light_high_threshold", panel.light_high_threshold),
                ("maximum_low_clip_fraction", panel.maximum_low_clip_fraction),
                ("maximum_high_clip_fraction", panel.maximum_high_clip_fraction),
                (
                    "maximum_l_standard_deviation",
                    panel.maximum_l_standard_deviation,
                ),
                ("maximum_background_delta_l", panel.maximum_background_delta_l),
                ("pca_print_components", panel.pca_print_components),
                ("pca_scatter_max_masks", panel.pca_scatter_max_masks),
                ("pca_scatter_alpha", panel.pca_scatter_alpha),
                ("selected_candidate_count", 4),
            ),
        ),
        ReportOperation(
            "select-detection-rows",
            ("NB-LIVE-0012-C0024", "NB-LIVE-0012-C0025"),
            ("detection_rows",),
            "selected_detections",
        ),
        ReportOperation(
            "match-masks-to-detections",
            ("NB-LIVE-0012-C0024", "NB-LIVE-0012-C0026"),
            ("mask_records", "selected_detections"),
            "mask_detection_matches",
        ),
        ReportOperation(
            "build-similarity-points",
            ("NB-LIVE-0012-C0027", "NB-LIVE-0012-C0028"),
            ("feature_rows", "mask_detection_matches"),
            "similarity_points",
            (
                ("similarity_column", "embed_sim"),
                ("probability_candidates", ("prob", "xgb_p")),
                ("pca_prefix", "xf_embed_pca_"),
                ("maximum_components", 12),
                ("separation_epsilon", 1e-3),
                ("probability_threshold", 0.50),
                ("selected_count", 4),
            ),
        ),
        ReportOperation(
            "select-best-pca-pair",
            ("NB-LIVE-0012-C0023", "NB-LIVE-0012-C0028"),
            ("reviewed_rows",),
            "best_pca_pair",
            (("maximum_components", 12),),
        ),
    )


# SOURCE_CELL: NB-LIVE-0012-C0000
# SOURCE_CELL: NB-LIVE-0012-C0003
# SOURCE_CELL: NB-LIVE-0012-C0005
# SOURCE_CELL: NB-LIVE-0012-C0006
# SOURCE_CELL: NB-LIVE-0012-C0007
# SOURCE_CELL: NB-LIVE-0012-C0008
# SOURCE_CELL: NB-LIVE-0012-C0011
# SOURCE_CELL: NB-LIVE-0012-C0012
# SOURCE_CELL: NB-LIVE-0012-C0015
# SOURCE_CELL: NB-LIVE-0012-C0022
# SOURCE_CELL: NB-LIVE-0012-C0023
# SOURCE_STATEMENT_MAP: structural-pass -> _explicit_noop
def _explicit_noop() -> None:
    """Represent an intentional source no-op as an explicit typed operation."""

    return None


class PrecomputedProbabilityModel:
    """Probability service backed only by a sealed score column."""

    def fit(self, features: Sequence[Sequence[float]], labels: Sequence[int]) -> None:
        if len(features) != len(labels):
            raise ReportingContractError("precomputed model fit arrays must align")

    def predict_probability(
        self,
        features: Sequence[Sequence[float]],
    ) -> Sequence[float]:
        if any(len(row) != 1 for row in features):
            raise ReportingContractError("precomputed model expects one sealed score per row")
        return tuple(min(1.0, max(0.0, float(row[0]))) for row in features)


def _float(value: Any, default: float = nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if isfinite(parsed) else default


def _binary(value: Any) -> int | None:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y", "positive", "accept"}:
        return 1
    if text in {"0", "0.0", "false", "no", "n", "negative", "reject"}:
        return 0
    return None


# SOURCE_CELL: NB-LIVE-0012-C0000
# SOURCE_STATEMENT_MAP: r-prefix-parser -> round_number
def round_number(name: str) -> int:
    parsed = match(r"r(\d+)", name)
    return int(parsed.group(1)) if parsed else -1


# SOURCE_CELL: NB-LIVE-0012-C0000
# SOURCE_STATEMENT_MAP: sorted-recall-trapezoid -> integrate_precision_recall
def integrate_precision_recall(
    recall: Sequence[float],
    precision: Sequence[float],
) -> float:
    points = sorted(
        (r, p)
        for r, p in zip(recall, precision)
        if isfinite(r) and isfinite(p)
    )
    if len(points) < 2:
        return nan
    return sum(
        (right_r - left_r) * (left_p + right_p) / 2.0
        for (left_r, left_p), (right_r, right_p) in zip(points, points[1:])
    )


def _rolling_median(values: Sequence[float], window: int) -> list[float]:
    radius = max(0, window // 2)
    return [
        float(median(values[max(0, index - radius) : min(len(values), index + radius + 1)]))
        for index in range(len(values))
    ]


# SOURCE_CELL: NB-LIVE-0012-C0000
# SOURCE_CELL: NB-LIVE-0012-C0005
# SOURCE_CELL: NB-LIVE-0012-C0006
# SOURCE_STATEMENT_MAP: run-filter-round-count-rolling-median-selected-annotation -> build_learning_curve
def build_learning_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_run: str,
    start_round: int | None = 60,
    smoothing_window: int = 11,
) -> tuple[LearningPoint, ...]:
    parsed: list[tuple[str, int, float]] = []
    for row in rows:
        run = str(row.get("run", row.get("model_rel", ""))).split("/", 1)[0]
        iteration = int(_float(row.get("r", row.get("round", round_number(run))), -1.0))
        value = _float(
            row.get(
                "pr_auc",
                row.get("ap", row.get("average_precision", row.get("test_pr_auc"))),
            )
        )
        if iteration >= 0 and isfinite(value) and (start_round is None or iteration >= start_round):
            parsed.append((run, iteration, value))
    parsed.sort(key=lambda item: item[1])
    if not parsed:
        raise ReportingContractError("no PR-AUC rows remain after round filtering")
    if selected_run and not any(run == selected_run for run, _, _ in parsed):
        raise ReportingContractError(f"selected run is not present: {selected_run}")
    smoothed = _rolling_median([value for _, _, value in parsed], smoothing_window)
    return tuple(
        LearningPoint(run, iteration, value, smooth, run == selected_run)
        for (run, iteration, value), smooth in zip(parsed, smoothed)
    )


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return nan
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    seen_positive = 0
    seen_rows = 0
    previous_recall = 0.0
    total = 0.0
    cursor = 0
    while cursor < len(ordered):
        score = ordered[cursor][0]
        group_end = cursor
        while group_end < len(ordered) and ordered[group_end][0] == score:
            seen_positive += ordered[group_end][1]
            seen_rows += 1
            group_end += 1
        recall = seen_positive / positives
        precision = seen_positive / seen_rows
        total += (recall - previous_recall) * precision
        previous_recall = recall
        cursor = group_end
    return total


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positive = [score for label, score in zip(labels, scores) if label == 1]
    negative = [score for label, score in zip(labels, scores) if label == 0]
    if not positive or not negative:
        return nan
    wins = 0.0
    for pvalue in positive:
        for nvalue in negative:
            wins += 1.0 if pvalue > nvalue else 0.5 if pvalue == nvalue else 0.0
    return wins / (len(positive) * len(negative))


def classification_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = 0.5,
) -> ClassificationMetrics:
    if not labels or len(labels) != len(scores):
        raise ReportingContractError("classification labels and scores must align")
    if any(label not in {0, 1} for label in labels):
        raise ReportingContractError("classification labels must be binary")
    if any(not isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
        raise ReportingContractError("classification scores must be finite probabilities")
    if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ReportingContractError("classification threshold must be a probability")
    predictions = [int(score >= threshold) for score in scores]
    true_positive = sum(p == 1 and y == 1 for p, y in zip(predictions, labels))
    false_positive = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    true_negative = sum(p == 0 and y == 0 for p, y in zip(predictions, labels))
    false_negative = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (true_positive + true_negative) / max(1, len(labels))
    return ClassificationMetrics(
        threshold,
        true_positive,
        false_positive,
        true_negative,
        false_negative,
        precision,
        recall,
        f1,
        accuracy,
        _average_precision(labels, scores),
        _roc_auc(labels, scores),
    )


def _best_f1_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
) -> ClassificationMetrics:
    best_threshold: float | None = None
    best_f1 = -1.0
    for threshold in sorted(set(scores)):
        predictions = [score >= threshold for score in scores]
        true_positive = sum(
            prediction and label == 1
            for prediction, label in zip(predictions, labels)
        )
        false_positive = sum(
            prediction and label == 0
            for prediction, label in zip(predictions, labels)
        )
        false_negative = sum(
            not prediction and label == 1
            for prediction, label in zip(predictions, labels)
        )
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        if f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1
    if best_threshold is None:
        raise ReportingContractError("best-F1 search requires precomputed scores")
    return classification_metrics(labels, scores, best_threshold)


# SOURCE_CELL: NB-LIVE-0012-C0002
# SOURCE_CELL: NB-LIVE-0012-C0014
# SOURCE_STATEMENT_MAP: latest-effective-review-weight-root-border-tiny -> build_reviewed_testset
def build_reviewed_testset(
    feature_rows: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    *,
    tile_size: int = 512,
    border_pixels: int = 4,
    tiny_minimum_dimension: int = 32,
    tiny_maximum_area: int = 1500,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    weights = {"accept": 1.0, "flip": 1.0, "sus_accept": 0.4, "sus_flip": 0.4, "skip": 0.0}
    effective: dict[tuple[str, str], tuple[tuple[int, float | str], dict[str, Any]]] = {}
    for ordinal, source in enumerate(review_rows):
        image = Path(str(source.get("image", ""))).name
        identity = str(source.get("id", ""))
        label = _binary(source.get("human_label"))
        if not image or not identity or label is None:
            continue
        timestamp = _float(source.get("timestamp"))
        order = (1, timestamp) if isfinite(timestamp) else (0, str(source.get("timestamp", ordinal)))
        action = str(source.get("action", "")).strip().lower()
        weight = _float(source.get("review_weight"))
        if not isfinite(weight):
            weight = weights.get(action, 1.0)
        row = dict(source)
        row.update({"image": image, "id": identity, "human_label": label, "review_weight": weight})
        key = (image, identity)
        if key not in effective or order >= effective[key][0]:
            effective[key] = (order, row)

    pool: list[dict[str, Any]] = []
    labeled: list[dict[str, Any]] = []
    for source in feature_rows:
        row = dict(source)
        image = Path(str(row.get("image", row.get("file_name", "")))).name
        identity = str(row.get("id", row.get("ann_id", "")))
        x = _float(row.get("bbox_x"), 0.0)
        y = _float(row.get("bbox_y"), 0.0)
        width = _float(row.get("bbox_w"), 0.0)
        height = _float(row.get("bbox_h"), 0.0)
        touching = _float(row.get("touching_border"), 0.0) == 1.0
        border = (
            touching
            or x <= border_pixels
            or y <= border_pixels
            or x + width >= tile_size - border_pixels
            or y + height >= tile_size - border_pixels
        )
        tiny = min(width, height) < tiny_minimum_dimension or width * height < tiny_maximum_area
        stem = Path(image).stem
        root = stem.split("_y", 1)[0] if "_y" in stem else stem
        row.update(
            {
                "image": image,
                "id": identity,
                "root": root,
                "is_border": border,
                "is_tiny": tiny,
                "is_border_or_tiny": border or tiny,
            }
        )
        review = effective.get((image, identity))
        if review is not None:
            row.update(review[1])
            labeled.append(dict(row))
        pool.append(row)
    return pool, labeled


# SOURCE_CELL: NB-LIVE-0012-C0014
# SOURCE_STATEMENT_MAP: sealed-pool-labeled-summary-folder-tables -> build_precomputed_reviewed_testset
def build_precomputed_reviewed_testset(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    tile_size: int = 512,
    border_pixels: int = 4,
    tiny_minimum_dimension: int = 32,
    tiny_maximum_area: int = 1500,
) -> ReviewedTestsetBatch:
    if not mode or any(character in mode for character in "\r\n/\\"):
        raise ReportingContractError("reviewed testset mode is invalid")
    pool, labeled = build_reviewed_testset(
        feature_rows,
        feature_rows,
        tile_size=tile_size,
        border_pixels=border_pixels,
        tiny_minimum_dimension=tiny_minimum_dimension,
        tiny_maximum_area=tiny_maximum_area,
    )
    if not pool:
        raise ReportingContractError("reviewed testset requires sealed feature rows")
    folder_names = tuple(sorted({str(row.get("img_folder", "")) for row in pool}))
    if any(not folder for folder in folder_names):
        raise ReportingContractError("reviewed testset rows require img_folder")
    reviewed_identity_count = len(
        {
            (str(row["image"]), str(row["id"]), int(row["human_label"]))
            for row in labeled
        }
    )
    positive_count = sum(int(row["human_label"]) for row in labeled)
    statistics = ReviewedTestsetStatistics(
        len(pool),
        reviewed_identity_count,
        reviewed_identity_count / len(pool),
        positive_count,
        positive_count / max(1, reviewed_identity_count),
        sum(bool(row["is_border"]) for row in pool),
        sum(bool(row["is_tiny"]) for row in pool),
        sum(bool(row["is_border_or_tiny"]) for row in pool),
        sum(bool(row["is_border_or_tiny"]) for row in labeled),
        mode,
        folder_names,
        all("yolo_conf" in row and "yolo_iou" in row for row in pool),
    )
    folder_statistics: list[ReviewedFolderStatistics] = []
    for folder in folder_names:
        folder_pool = [row for row in pool if str(row["img_folder"]) == folder]
        folder_labeled = [row for row in labeled if str(row["img_folder"]) == folder]
        positives = sum(int(row["human_label"]) for row in folder_labeled)
        folder_statistics.append(
            ReviewedFolderStatistics(
                folder,
                len(folder_pool),
                len(folder_labeled),
                len(folder_labeled) / len(folder_pool),
                positives,
                positives / max(1, len(folder_labeled)),
            )
        )
    return ReviewedTestsetBatch(
        tuple(dict(row) for row in pool),
        tuple(dict(row) for row in labeled),
        statistics,
        tuple(folder_statistics),
    )


# SOURCE_CELL: NB-LIVE-0012-C0003
# SOURCE_STATEMENT_MAP: model-score-normalization-fixed-and-best-threshold -> evaluate_model_scores
def evaluate_model_scores(
    labels: Sequence[int],
    scores_by_model: Mapping[str, Sequence[float]],
    *,
    fixed_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model, scores in sorted(scores_by_model.items()):
        if len(scores) != len(labels):
            raise ReportingContractError(f"score length differs for {model}")
        fixed = classification_metrics(labels, scores, fixed_threshold)
        best = _best_f1_metrics(labels, scores)
        round_match = search(r"_r(?P<round>\d+)", Path(model).name)
        output.append(
            {
                "model_rel": model,
                "round": (
                    int(round_match.group("round")) if round_match is not None else None
                ),
                "pr_auc": fixed.average_precision,
                "roc_auc": fixed.roc_auc,
                "fixed_thr": fixed_threshold,
                "f1_fixed": fixed.f1,
                "prec_fixed": fixed.precision,
                "rec_fixed": fixed.recall,
                "thr_best": best.threshold,
                "f1_best": best.f1,
                "prec_best": best.precision,
                "rec_best": best.recall,
            }
        )
    return sorted(
        output,
        key=lambda row: (-float(row["pr_auc"]), str(row["model_rel"])),
    )


# SOURCE_CELL: NB-LIVE-0012-C0003
# SOURCE_STATEMENT_MAP: ordered-curve-dedupe-grid-population-sigma -> build_model_curve_bands
def _dedupe_x_take_max_y(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(x_values) != len(y_values):
        raise ReportingContractError("curve coordinates must have equal length")
    maximum_by_x: dict[float, float] = {}
    for x_value, y_value in zip(x_values, y_values):
        if isfinite(x_value) and isfinite(y_value):
            maximum_by_x[x_value] = max(y_value, maximum_by_x.get(x_value, -inf))
    ordered = sorted(maximum_by_x.items())
    return (
        tuple(item[0] for item in ordered),
        tuple(item[1] for item in ordered),
    )


def _interp_curve_to_grid(
    x_values: Sequence[float],
    y_values: Sequence[float],
    grid: Sequence[float],
) -> tuple[float, ...]:
    xs, ys = _dedupe_x_take_max_y(x_values, y_values)
    if len(xs) < 2:
        raise ReportingContractError("curve requires at least two finite x coordinates")
    output: list[float] = []
    right = 1
    for value in grid:
        while right < len(xs) and xs[right] < value:
            right += 1
        if right >= len(xs):
            interpolated = ys[-1]
        elif value <= xs[0]:
            interpolated = ys[0]
        else:
            left = right - 1
            width = xs[right] - xs[left]
            fraction = 0.0 if width == 0.0 else (value - xs[left]) / width
            interpolated = ys[left] + fraction * (ys[right] - ys[left])
        output.append(min(1.0, max(0.0, interpolated)))
    return tuple(output)


def _binary_curve_points(
    labels: Sequence[int],
    scores: Sequence[float],
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[tuple[float, float], ...],
]:
    if len(labels) != len(scores) or not labels:
        raise ReportingContractError("curve labels and scores must be nonempty and aligned")
    if any(label not in {0, 1} for label in labels) or any(
        not isfinite(score) for score in scores
    ):
        raise ReportingContractError("curve labels and scores violate the binary contract")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ReportingContractError("PR/ROC curves require both binary classes")
    precision_recall: list[tuple[float, float]] = [(0.0, 1.0)]
    roc: list[tuple[float, float]] = [(0.0, 0.0)]
    for threshold in sorted(set(scores), reverse=True):
        predictions = [score >= threshold for score in scores]
        true_positive = sum(prediction and label == 1 for prediction, label in zip(predictions, labels))
        false_positive = sum(prediction and label == 0 for prediction, label in zip(predictions, labels))
        recall = true_positive / positives
        precision = true_positive / max(1, true_positive + false_positive)
        precision_recall.append((recall, precision))
        roc.append((false_positive / negatives, recall))
    return tuple(precision_recall), tuple(roc)


def _mean_sigma(curves: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(curves) < 2 or not curves or not curves[0]:
        raise ReportingContractError("population curve band requires at least two models")
    width = len(curves[0])
    if any(len(curve) != width for curve in curves):
        raise ReportingContractError("population curves must use one shared grid")
    means: list[float] = []
    sigmas: list[float] = []
    for ordinal in range(width):
        values = [curve[ordinal] for curve in curves]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means.append(min(1.0, max(0.0, mean)))
        sigmas.append(min(1.0, max(0.0, sqrt(variance))))
    return tuple(means), tuple(sigmas)


def build_model_curve_bands(
    labels: Sequence[int],
    scores_by_model: Mapping[str, Sequence[float]],
    *,
    grid_steps: int = 401,
    maximum_models: int = 250,
) -> tuple[ModelCurveBand, ModelCurveBand]:
    if grid_steps < 2 or maximum_models < 2:
        raise ReportingContractError("curve grid and population limits are invalid")
    if len(scores_by_model) < 2:
        raise ReportingContractError("curve bands require at least two sealed model scores")
    ranked = sorted(
        scores_by_model.items(),
        key=lambda item: (-_average_precision(labels, item[1]), item[0]),
    )[:maximum_models]
    grid = tuple(index / (grid_steps - 1) for index in range(grid_steps))
    pr_curves: list[tuple[float, ...]] = []
    roc_curves: list[tuple[float, ...]] = []
    for _model, scores in ranked:
        precision_recall, roc = _binary_curve_points(labels, scores)
        pr_curves.append(
            _interp_curve_to_grid(
                [point[0] for point in precision_recall],
                [point[1] for point in precision_recall],
                grid,
            )
        )
        roc_curves.append(
            _interp_curve_to_grid(
                [point[0] for point in roc],
                [point[1] for point in roc],
                grid,
            )
        )
    pr_mean, pr_sigma = _mean_sigma(pr_curves)
    roc_mean, roc_sigma = _mean_sigma(roc_curves)
    best_model = ranked[0][0]
    base_rate = sum(labels) / len(labels)
    return (
        ModelCurveBand(
            "precision-recall",
            best_model,
            base_rate,
            tuple(
                CurveBandPoint(x, best, mean, sigma)
                for x, best, mean, sigma in zip(
                    grid,
                    pr_curves[0],
                    pr_mean,
                    pr_sigma,
                )
            ),
        ),
        ModelCurveBand(
            "roc",
            best_model,
            0.0,
            tuple(
                CurveBandPoint(x, best, mean, sigma)
                for x, best, mean, sigma in zip(
                    grid,
                    roc_curves[0],
                    roc_mean,
                    roc_sigma,
                )
            ),
        ),
    )


def _required_precomputed_probability(
    value: object,
    *,
    column: str,
    row_index: int,
) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportingContractError(
            f"precomputed row {row_index} has a nonnumeric {column} value"
        ) from exc
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ReportingContractError(
            f"precomputed row {row_index} has an invalid {column} probability"
        )
    return probability


def _required_precomputed_label(
    value: object,
    *,
    column: str,
    row_index: int,
) -> int:
    try:
        label = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportingContractError(
            f"precomputed row {row_index} has a nonnumeric {column} value"
        ) from exc
    if not isfinite(label) or label not in {0.0, 1.0}:
        raise ReportingContractError(
            f"precomputed row {row_index} has a nonbinary {column} value"
        )
    return int(label)


def _precomputed_group_column(
    rows: Sequence[Mapping[str, object]],
    precedence: Sequence[str],
) -> str | None:
    seen: set[str] = set()
    for column in precedence:
        if not column or column in seen:
            raise ReportingContractError("model group precedence is invalid")
        seen.add(column)
        present = [column in row for row in rows]
        if any(present) and not all(present):
            raise ReportingContractError(
                f"precomputed model group column is incomplete: {column}"
            )
        if all(present):
            return column
    return None


# SOURCE_CELL: NB-LIVE-0012-C0003
# SOURCE_STATEMENT_MAP: declared-score-columns-candidate-image-metrics-bands -> evaluate_precomputed_model_curves
def evaluate_precomputed_model_curves(
    rows: Sequence[Mapping[str, object]],
    score_columns: Sequence[PrecomputedModelScoreColumn],
    *,
    label_column: str = "human_label",
    group_precedence: tuple[str, ...] = ("root", "img_folder"),
    fixed_threshold: float = 0.50,
    grid_steps: int = 401,
    maximum_models: int = 250,
    top_k_to_plot: int = 8,
    figure_inches: tuple[float, float] = (3.35, 3.00),
    save_dpi: int = 600,
    paper_style: bool = True,
    export_svg: bool = False,
    plot_bars: bool = True,
) -> PrecomputedModelCurveEvaluation:
    """Evaluate only explicitly declared score columns in verified C3 rows."""

    if not rows or not label_column:
        raise ReportingContractError("precomputed model rows and label column are required")
    if (
        not 0.0 <= fixed_threshold <= 1.0
        or grid_steps < 2
        or maximum_models < 2
        or top_k_to_plot <= 0
        or len(figure_inches) != 2
        or any(not isfinite(value) or value <= 0.0 for value in figure_inches)
        or save_dpi <= 0
    ):
        raise ReportingContractError("precomputed model curve settings are invalid")
    if not score_columns:
        raise ReportingContractError("precomputed model score columns are required")
    if not group_precedence:
        raise ReportingContractError("model group precedence is required")

    declared_models: set[str] = set()
    declared_columns: set[str] = set()
    for declaration in score_columns:
        if not declaration.model_rel or not declaration.column:
            raise ReportingContractError("precomputed model declarations must be named")
        if declaration.model_rel in declared_models:
            raise ReportingContractError(
                f"duplicate precomputed model declaration: {declaration.model_rel}"
            )
        if declaration.column in declared_columns:
            raise ReportingContractError(
                f"duplicate precomputed score column: {declaration.column}"
            )
        if declaration.column == label_column:
            raise ReportingContractError("model score column cannot be the label column")
        declared_models.add(declaration.model_rel)
        declared_columns.add(declaration.column)

    required_columns = (label_column, *(item.column for item in score_columns))
    for row_index, row in enumerate(rows):
        missing = [column for column in required_columns if column not in row]
        if missing:
            raise ReportingContractError(
                f"precomputed row {row_index} is missing declared columns: {missing}"
            )

    labels = tuple(
        _required_precomputed_label(
            row[label_column],
            column=label_column,
            row_index=row_index,
        )
        for row_index, row in enumerate(rows)
    )
    if len(set(labels)) != 2:
        raise ReportingContractError("precomputed model evaluation requires both classes")
    group_column = _precomputed_group_column(rows, group_precedence)
    evaluations: list[ModelScoreEvaluation] = []
    scores_by_model: dict[str, tuple[float, ...]] = {}

    for declaration in sorted(score_columns, key=lambda item: item.model_rel):
        scores = tuple(
            _required_precomputed_probability(
                row[declaration.column],
                column=declaration.column,
                row_index=row_index,
            )
            for row_index, row in enumerate(rows)
        )
        scores_by_model[declaration.model_rel] = scores
        fixed = classification_metrics(labels, scores, fixed_threshold)
        best = _best_f1_metrics(labels, scores)

        grouped: dict[str, tuple[int, float]] = {}
        if group_column is not None:
            for row_index, (row, label, score) in enumerate(
                zip(rows, labels, scores)
            ):
                group = str(row[group_column]).strip()
                if not group:
                    raise ReportingContractError(
                        f"precomputed row {row_index} has an empty {group_column}"
                    )
                previous_label, previous_score = grouped.get(group, (0, 0.0))
                grouped[group] = (max(previous_label, label), max(previous_score, score))
        group_labels = tuple(value[0] for _, value in sorted(grouped.items()))
        group_scores = tuple(value[1] for _, value in sorted(grouped.items()))
        group_has_both_classes = len(set(group_labels)) == 2
        round_match = search(r"_r(?P<round>\d+)", Path(declaration.model_rel).name)
        evaluations.append(
            ModelScoreEvaluation(
                declaration.model_rel,
                declaration.column,
                (
                    int(round_match.group("round"))
                    if round_match is not None
                    else None
                ),
                fixed,
                best,
                group_column,
                len(grouped),
                sum(group_labels),
                (
                    _average_precision(group_labels, group_scores)
                    if group_has_both_classes
                    else None
                ),
                (
                    _roc_auc(group_labels, group_scores)
                    if group_has_both_classes
                    else None
                ),
            )
        )

    evaluations.sort(
        key=lambda item: (-item.candidate_metrics.average_precision, item.model_rel)
    )
    precision_recall_band: ModelCurveBand | None = None
    roc_band: ModelCurveBand | None = None
    model_curve_evaluation: PrecomputedModelCurveEvaluation | None = None
    active_learning_score_evaluation: PrecomputedActiveLearningFocusEvaluation | None = None
    source_confusion_evaluations: tuple[PrecomputedModelConfusionEvaluation, ...] = ()
    if len(scores_by_model) >= 2:
        precision_recall_band, roc_band = build_model_curve_bands(
            labels,
            scores_by_model,
            grid_steps=grid_steps,
            maximum_models=maximum_models,
        )
    return PrecomputedModelCurveEvaluation(
        label_column,
        group_column,
        fixed_threshold,
        grid_steps,
        maximum_models,
        top_k_to_plot,
        figure_inches,
        save_dpi,
        paper_style,
        export_svg,
        plot_bars,
        len(rows),
        sum(labels),
        tuple(evaluations),
        tuple(item.model_rel for item in evaluations[:top_k_to_plot]),
        precision_recall_band,
        roc_band,
    )


# SOURCE_CELL: NB-LIVE-0012-C0007
# SOURCE_STATEMENT_MAP: uncertainty-margin-disagreement-row -> _active_learning_focus_row
def _active_learning_focus_row(
    source: Mapping[str, Any],
    *,
    threshold: float,
    margin: float,
    disagreement_delta: float,
    disagreement_iou_minimum: float,
) -> ActiveLearningFocusRow:
    probability = _float(
        source.get("prob", source.get("xgb_prob", source.get("xgb_prob_f")))
    )
    yolo = _float(source.get("yolo_conf", source.get("yolo_conf_f")))
    iou = _float(source.get("yolo_iou", source.get("yolo_iou_f")))
    if not isfinite(probability):
        raise ReportingContractError("active-learning score row has no finite probability")
    uncertain = abs(probability - threshold) <= margin
    conflict = (
        isfinite(yolo)
        and isfinite(iou)
        and yolo > 0.0
        and abs(probability - yolo) >= disagreement_delta
        and iou >= disagreement_iou_minimum
    )
    return ActiveLearningFocusRow(
        str(source.get("image", "")),
        int(_float(source.get("id"), -1.0)),
        probability,
        uncertain,
        conflict,
        uncertain or conflict,
    )


# SOURCE_STATEMENT_MAP: uncertainty-margin-disagreement-spatial-summary -> where_active_learning_looks
def where_active_learning_looks(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.50,
    margin: float = 0.20,
    disagreement_delta: float = 0.60,
    disagreement_iou_minimum: float = 0.60,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for source in rows:
        focus = _active_learning_focus_row(
            source,
            threshold=threshold,
            margin=margin,
            disagreement_delta=disagreement_delta,
            disagreement_iou_minimum=disagreement_iou_minimum,
        )
        if focus.selected:
            row = dict(source)
            row.update(
                {
                    "p_model": focus.probability,
                    "sel_unc": int(focus.uncertain),
                    "sel_conflict": int(focus.conflict),
                    "selected": 1,
                }
            )
            selected.append(row)
    return selected


# SOURCE_STATEMENT_MAP: shortlist-stats-and-histogram-contract -> evaluate_active_learning_focus
def evaluate_active_learning_focus(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.50,
    margin: float = 0.20,
    disagreement_delta: float = 0.60,
    disagreement_iou_minimum: float = 0.60,
    histogram_bins: int = 40,
    figure_inches: tuple[float, float] = (4.6, 3.1),
    save_dpi: int = 600,
) -> ActiveLearningFocusEvaluation:
    if (
        not rows
        or not 0.0 <= threshold <= 1.0
        or not 0.0 <= margin <= 1.0
        or not 0.0 <= disagreement_delta <= 1.0
        or not 0.0 <= disagreement_iou_minimum <= 1.0
        or histogram_bins <= 0
        or len(figure_inches) != 2
        or any(not isfinite(value) or value <= 0.0 for value in figure_inches)
        or save_dpi <= 0
    ):
        raise ReportingContractError("active-learning focus settings are invalid")
    focus_rows = tuple(
        _active_learning_focus_row(
            row,
            threshold=threshold,
            margin=margin,
            disagreement_delta=disagreement_delta,
            disagreement_iou_minimum=disagreement_iou_minimum,
        )
        for row in rows
    )
    total = len(focus_rows)
    uncertain = sum(row.uncertain for row in focus_rows)
    conflict = sum(row.conflict for row in focus_rows)
    selected = sum(row.selected for row in focus_rows)
    return ActiveLearningFocusEvaluation(
        threshold,
        margin,
        disagreement_delta,
        disagreement_iou_minimum,
        histogram_bins,
        figure_inches,
        save_dpi,
        total,
        uncertain,
        uncertain / total,
        conflict,
        conflict / total,
        selected,
        selected / total,
        focus_rows,
    )


def _required_precomputed_candidate_id(
    value: object,
    *,
    column: str,
    row_index: int,
) -> int:
    try:
        candidate = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportingContractError(
            f"precomputed row {row_index} has a nonnumeric {column} value"
        ) from exc
    if not isfinite(candidate) or not candidate.is_integer():
        raise ReportingContractError(
            f"precomputed row {row_index} has a nonintegral {column} value"
        )
    return int(candidate)


# SOURCE_CELL: NB-LIVE-0012-C0007
# SOURCE_STATEMENT_MAP: declared-probability-uncertainty-disagreement-focus -> evaluate_precomputed_active_learning_focus
def evaluate_precomputed_active_learning_focus(
    rows: Sequence[Mapping[str, object]],
    columns: PrecomputedActiveLearningScoreColumns,
    *,
    threshold: float = 0.50,
    margin: float = 0.20,
    disagreement_delta: float = 0.60,
    disagreement_iou_minimum: float = 0.60,
    histogram_bins: int = 40,
    figure_inches: tuple[float, float] = (4.6, 3.1),
    save_dpi: int = 600,
    paper_style: bool = True,
) -> PrecomputedActiveLearningFocusEvaluation:
    """Select C7 focus rows from an explicitly declared precomputed score."""

    if not rows:
        raise ReportingContractError("precomputed active-learning rows are required")
    required_names = (
        columns.model_rel,
        columns.probability,
        columns.image,
        columns.candidate_id,
    )
    if any(not value for value in required_names):
        raise ReportingContractError(
            "precomputed active-learning model and columns must be named"
        )
    yolo_pair = (columns.yolo_confidence, columns.yolo_iou)
    if (yolo_pair[0] is None) != (yolo_pair[1] is None):
        raise ReportingContractError(
            "YOLO confidence and IoU columns must be declared together"
        )
    if yolo_pair[0] is not None and (not yolo_pair[0] or not yolo_pair[1]):
        raise ReportingContractError("YOLO column declarations must be named")
    disagreement_enabled = False
    declared_columns = [
        columns.probability,
        columns.image,
        columns.candidate_id,
    ]
    if yolo_pair[0] is not None:
        confidence_column = str(yolo_pair[0])
        iou_column = str(yolo_pair[1])
        confidence_presence = [confidence_column in row for row in rows]
        iou_presence = [iou_column in row for row in rows]
        if (any(confidence_presence) and not all(confidence_presence)) or (
            any(iou_presence) and not all(iou_presence)
        ):
            raise ReportingContractError(
                "precomputed YOLO columns are incomplete across rows"
            )
        disagreement_enabled = all(confidence_presence) and all(iou_presence)
        if disagreement_enabled:
            declared_columns.extend((confidence_column, iou_column))
    if len(set(declared_columns)) != len(declared_columns):
        raise ReportingContractError(
            "precomputed active-learning column declarations must be distinct"
        )

    normalized_rows: list[Mapping[str, object]] = []
    for row_index, row in enumerate(rows):
        missing = [column for column in declared_columns if column not in row]
        if missing:
            raise ReportingContractError(
                f"precomputed row {row_index} is missing declared columns: {missing}"
            )
        probability = _required_precomputed_probability(
            row[columns.probability],
            column=columns.probability,
            row_index=row_index,
        )
        normalized: dict[str, object] = {
            "image": str(row[columns.image]),
            "id": _required_precomputed_candidate_id(
                row[columns.candidate_id],
                column=columns.candidate_id,
                row_index=row_index,
            ),
            "prob": probability,
        }
        if disagreement_enabled:
            confidence_column = str(yolo_pair[0])
            iou_column = str(yolo_pair[1])
            confidence = _float(row[confidence_column], 0.0)
            iou = _float(row[iou_column], 0.0)
            normalized["yolo_conf"] = confidence
            normalized["yolo_iou"] = iou
        normalized_rows.append(normalized)

    focus = evaluate_active_learning_focus(
        normalized_rows,
        threshold=threshold,
        margin=margin,
        disagreement_delta=disagreement_delta,
        disagreement_iou_minimum=disagreement_iou_minimum,
        histogram_bins=histogram_bins,
        figure_inches=figure_inches,
        save_dpi=save_dpi,
    )
    return PrecomputedActiveLearningFocusEvaluation(
        columns,
        disagreement_enabled,
        paper_style,
        focus,
    )


# SOURCE_CELL: NB-LIVE-0012-C0008
# SOURCE_STATEMENT_MAP: group-holdout-uncertainty-vs-random-budget-loop -> simulate_active_learning
def simulate_active_learning(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    groups: Sequence[str],
    model_factory: Callable[[], ProbabilityModel],
    *,
    strategy: str,
    seed: int,
    repeat: int = 0,
    split_seed: int = 0,
    holdout_fraction: float = 0.25,
    initial_size: int = 100,
    batch_size: int = 50,
    maximum_budget: int = 2000,
) -> tuple[ActiveLearningCurvePoint, ...]:
    if not (len(features) == len(labels) == len(groups)):
        raise ReportingContractError("AL arrays must have equal length")
    if (
        len(features) < 4
        or any(label not in {0, 1} for label in labels)
        or strategy not in {"uncertainty", "random"}
        or repeat < 0
        or not 0.0 < holdout_fraction < 1.0
        or initial_size <= 0
        or batch_size <= 0
        or maximum_budget <= 0
    ):
        raise ReportingContractError("active-learning simulation settings are invalid")
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ReportingContractError("active-learning simulation requires two groups")
    Random(split_seed).shuffle(unique_groups)
    holdout_count = min(
        len(unique_groups) - 1,
        max(1, ceil(len(unique_groups) * holdout_fraction)),
    )
    holdout_groups = set(unique_groups[:holdout_count])
    pool = [index for index, group in enumerate(groups) if group not in holdout_groups]
    holdout = [index for index, group in enumerate(groups) if group in holdout_groups]
    rng = Random(seed)
    rng.shuffle(pool)
    maximum_budget = min(maximum_budget, len(pool))
    labeled_indices = pool[: min(initial_size, maximum_budget)]
    unlabeled = pool[len(labeled_indices) :]
    curve: list[ActiveLearningCurvePoint] = []
    while True:
        model = model_factory()
        model.fit([features[index] for index in labeled_indices], [labels[index] for index in labeled_indices])
        probabilities = list(model.predict_probability([features[index] for index in holdout]))
        curve.append(
            ActiveLearningCurvePoint(
                strategy,
                repeat,
                len(labeled_indices),
                _average_precision(
                    [labels[index] for index in holdout], probabilities
                ),
            )
        )
        if len(labeled_indices) >= maximum_budget or not unlabeled:
            break
        count = min(batch_size, maximum_budget - len(labeled_indices), len(unlabeled))
        if strategy == "random":
            rng.shuffle(unlabeled)
            chosen = unlabeled[:count]
        elif strategy == "uncertainty":
            pool_probability = list(model.predict_probability([features[index] for index in unlabeled]))
            order = sorted(range(len(unlabeled)), key=lambda index: abs(pool_probability[index] - 0.5))
            chosen = [unlabeled[index] for index in order[:count]]
        chosen_set = set(chosen)
        unlabeled = [index for index in unlabeled if index not in chosen_set]
        labeled_indices.extend(chosen)
    return tuple(curve)


# SOURCE_CELL: NB-LIVE-0012-C0008
# SOURCE_STATEMENT_MAP: exact-two-strategy-ten-repeat-aggregate -> evaluate_precomputed_active_learning
def evaluate_precomputed_active_learning(
    scores: Sequence[float],
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    initial_size: int = 100,
    batch_size: int = 50,
    maximum_budget: int = 2000,
    repeats: int = 10,
    holdout_fraction: float = 0.25,
    random_seed: int = 0,
    strategies: tuple[str, ...] = ("uncertainty", "random"),
    figure_inches: tuple[float, float] = (4.8, 3.2),
    save_dpi: int = 600,
) -> ActiveLearningSimulationEvaluation:
    if (
        len(scores) != len(labels)
        or len(scores) != len(groups)
        or any(not isfinite(score) or not 0.0 <= score <= 1.0 for score in scores)
        or repeats <= 0
        or strategies != ("uncertainty", "random")
        or any(value <= 0.0 for value in figure_inches)
        or save_dpi <= 0
    ):
        raise ReportingContractError("precomputed active-learning inputs are invalid")
    features = tuple((float(score),) for score in scores)
    curve: list[ActiveLearningCurvePoint] = []
    for strategy in strategies:
        for repeat in range(repeats):
            strategy_seed = random_seed + repeat + (
                0 if strategy == "uncertainty" else 10_000
            )
            curve.extend(
                simulate_active_learning(
                    features,
                    labels,
                    groups,
                    PrecomputedProbabilityModel,
                    strategy=strategy,
                    seed=strategy_seed,
                    repeat=repeat,
                    split_seed=random_seed,
                    holdout_fraction=holdout_fraction,
                    initial_size=initial_size,
                    batch_size=batch_size,
                    maximum_budget=maximum_budget,
                )
            )
    buckets: dict[tuple[str, int], list[float]] = {}
    for point in curve:
        buckets.setdefault((point.strategy, point.budget), []).append(point.pr_auc)
    aggregate: list[ActiveLearningAggregatePoint] = []
    for (strategy, budget), values in sorted(buckets.items()):
        mean = sum(values) / len(values)
        standard_deviation = (
            sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
            if len(values) > 1
            else nan
        )
        aggregate.append(
            ActiveLearningAggregatePoint(
                strategy,
                budget,
                mean,
                standard_deviation,
                len(values),
            )
        )
    return ActiveLearningSimulationEvaluation(
        initial_size,
        batch_size,
        maximum_budget,
        repeats,
        holdout_fraction,
        random_seed,
        strategies,
        figure_inches,
        save_dpi,
        tuple(curve),
        tuple(aggregate),
    )


# SOURCE_CELL: NB-LIVE-0012-C0009
# SOURCE_CELL: NB-LIVE-0012-C0010
# SOURCE_STATEMENT_MAP: dataset-performance-topk-best-table-selection -> build_paper_tables
def build_paper_tables(
    statistics: Mapping[str, Any],
    model_metrics: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    top_k: int = 10,
    training_size: int | None = None,
    decimals: int = 3,
) -> Mapping[str, list[dict[str, Any]]]:
    dataset = [
        {
            "N_candidates_total": int(_float(statistics.get("N_candidates_total"), 0.0)),
            "N_test_reviewed": int(_float(statistics.get("N_reviewed_labeled"), 0.0)),
            "N_train_external": training_size,
            "pct_reviewed": round(_float(statistics.get("pct_reviewed")), decimals),
            "pos_rate_test": round(_float(statistics.get("pos_rate")), decimals),
            "N_border_or_tiny_all": int(_float(statistics.get("N_border_or_tiny"), 0.0)),
            "N_border_or_tiny_in_test": int(
                _float(statistics.get("N_reviewed_border_or_tiny"), 0.0)
            ),
            "mode": str(statistics.get("mode", mode)),
        }
    ]
    ranked = sorted(model_metrics, key=lambda row: _float(row.get("pr_auc"), -inf), reverse=True)
    keep = (
        "model_rel",
        "round",
        "pr_auc",
        "roc_auc",
        "fixed_thr",
        "f1_fixed",
        "prec_fixed",
        "rec_fixed",
        "thr_best",
        "f1_best",
        "pr_auc_img",
        "roc_auc_img",
    )
    performance: list[dict[str, Any]] = []
    for source in ranked[:top_k]:
        row: dict[str, Any] = {}
        for key in keep:
            if key not in source:
                continue
            value = source[key]
            row[key] = round(value, decimals) if isinstance(value, float) else value
        if "model_rel" in row:
            value = str(row.pop("model_rel"))
            row = {"model_short": value.removesuffix(".pkl").replace("/", " | ", 1), **row}
        performance.append(row)
    return {"dataset": dataset, "performance": performance, "best": performance[:1]}


# SOURCE_CELL: NB-LIVE-0012-C0011
# SOURCE_CELL: NB-LIVE-0012-C0012
# SOURCE_STATEMENT_MAP: declared-metric-round-score-selection -> _select_precomputed_confusion_score
def _select_precomputed_confusion_score(
    rows: Sequence[Mapping[str, object]],
    score_columns: Sequence[PrecomputedModelScoreColumn],
    metric_rows: Sequence[PrecomputedModelMetricRow],
    *,
    target_round: int,
    source_cell: str,
) -> tuple[
    PrecomputedModelScoreColumn,
    PrecomputedModelMetricRow,
    tuple[int, ...],
    tuple[float, ...],
]:
    if not rows:
        raise ReportingContractError("precomputed confusion rows are required")
    if type(target_round) is not int or target_round < 0:
        raise ReportingContractError("precomputed confusion target round is invalid")
    if not score_columns:
        raise ReportingContractError(
            "precomputed confusion score declarations are required"
        )
    if not metric_rows:
        raise ReportingContractError(
            "precomputed confusion metric declarations are required"
        )
    if source_cell not in {"NB-LIVE-0012-C0011", "NB-LIVE-0012-C0012"}:
        raise ReportingContractError("precomputed confusion source cell is invalid")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ReportingContractError("precomputed confusion rows must be mappings")

    declared_models: set[str] = set()
    declared_columns: set[str] = set()
    for declaration in score_columns:
        if not isinstance(declaration, PrecomputedModelScoreColumn):
            raise ReportingContractError(
                "precomputed confusion requires typed score declarations"
            )
        if not declaration.model_rel or not declaration.column:
            raise ReportingContractError(
                "precomputed confusion score declarations must be named"
            )
        if declaration.model_rel in declared_models:
            raise ReportingContractError(
                f"duplicate precomputed confusion model: {declaration.model_rel}"
            )
        if declaration.column in declared_columns:
            raise ReportingContractError(
                f"duplicate precomputed confusion score column: {declaration.column}"
            )
        if declaration.column == "human_label":
            raise ReportingContractError(
                "precomputed confusion score column cannot be human_label"
            )
        declared_models.add(declaration.model_rel)
        declared_columns.add(declaration.column)

    metrics_by_model: dict[str, PrecomputedModelMetricRow] = {}
    for metric in metric_rows:
        if not isinstance(metric, PrecomputedModelMetricRow):
            raise ReportingContractError(
                "precomputed confusion requires typed metric declarations"
            )
        if (
            not metric.model_rel
            or metric.model_rel in metrics_by_model
            or (metric.round is not None and type(metric.round) is not int)
            or type(metric.pr_auc) not in {int, float}
            or not isfinite(float(metric.pr_auc))
            or not 0.0 <= float(metric.pr_auc) <= 1.0
        ):
            raise ReportingContractError(
                "precomputed confusion metric declarations are invalid"
            )
        for value in (metric.fixed_threshold, metric.best_threshold):
            if value is not None and (
                type(value) not in {int, float}
                or not isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ReportingContractError(
                    "precomputed confusion metric thresholds are invalid"
                )
        metrics_by_model[metric.model_rel] = metric
    if set(metrics_by_model) != declared_models:
        raise ReportingContractError(
            "precomputed confusion metrics must exactly cover score declarations"
        )

    required_columns = ("human_label", *(item.column for item in score_columns))
    for row_index, row in enumerate(rows):
        missing = [column for column in required_columns if column not in row]
        if missing:
            raise ReportingContractError(
                f"precomputed confusion row {row_index} is missing columns: {missing}"
            )
    labels = tuple(
        _required_precomputed_label(
            row["human_label"],
            column="human_label",
            row_index=row_index,
        )
        for row_index, row in enumerate(rows)
    )

    candidates: list[
        tuple[
            PrecomputedModelScoreColumn,
            PrecomputedModelMetricRow,
            tuple[float, ...],
        ]
    ] = []
    for declaration in score_columns:
        scores = tuple(
            _required_precomputed_probability(
                row[declaration.column],
                column=declaration.column,
                row_index=row_index,
            )
            for row_index, row in enumerate(rows)
        )
        candidates.append(
            (
                declaration,
                metrics_by_model[declaration.model_rel],
                scores,
            )
        )

    target_candidates = [
        item for item in candidates if item[1].round == target_round
    ]
    if target_candidates:
        selection_pool = target_candidates
    elif source_cell == "NB-LIVE-0012-C0012" and any(
        item[1].round is not None for item in candidates
    ):
        # C12 drops rows with an invalid round before its overall fallback.
        selection_pool = [item for item in candidates if item[1].round is not None]
    else:
        # C11 falls back over the original metrics rows, including missing rounds.
        selection_pool = candidates
    selected = selection_pool[0]
    selected_rank = float(selected[1].pr_auc)
    for candidate in selection_pool[1:]:
        candidate_rank = float(candidate[1].pr_auc)
        if candidate_rank > selected_rank:
            selected = candidate
            selected_rank = candidate_rank
    return selected[0], selected[1], labels, selected[2]


def _resolve_precomputed_confusion_threshold(
    *,
    threshold_mode: str,
    threshold_override: float | None,
    fixed_threshold: float | None,
    best_threshold: float | None,
) -> tuple[float, str]:
    if threshold_mode not in {"fixed", "best"}:
        raise ReportingContractError(
            "precomputed confusion threshold mode must be fixed or best"
        )

    def optional_probability(value: float | None, role: str) -> float | None:
        if value is None:
            return None
        if type(value) not in {int, float}:
            raise ReportingContractError(
                f"precomputed confusion {role} must be a numeric probability"
            )
        parsed = float(value)
        if not isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise ReportingContractError(
                f"precomputed confusion {role} must be in [0, 1]"
            )
        return parsed

    override = optional_probability(threshold_override, "threshold override")
    fixed = optional_probability(fixed_threshold, "fixed threshold")
    best = optional_probability(best_threshold, "best threshold")
    if override is not None:
        return override, "override"
    if threshold_mode == "best" and best is not None:
        return best, "thr_best"
    if fixed is not None:
        return fixed, "fixed_thr"
    return 0.50, "fallback_0.5"


def _precomputed_confusion_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
) -> PrecomputedConfusionMetrics:
    if not labels or len(labels) != len(predictions):
        raise ReportingContractError(
            "precomputed confusion labels and predictions must align"
        )
    true_negative = sum(
        label == 0 and prediction == 0
        for label, prediction in zip(labels, predictions)
    )
    false_positive = sum(
        label == 0 and prediction == 1
        for label, prediction in zip(labels, predictions)
    )
    false_negative = sum(
        label == 1 and prediction == 0
        for label, prediction in zip(labels, predictions)
    )
    true_positive = sum(
        label == 1 and prediction == 1
        for label, prediction in zip(labels, predictions)
    )
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    accuracy = (true_positive + true_negative) / len(labels)
    specificity = true_negative / max(1, true_negative + false_positive)
    false_positive_rate = false_positive / max(
        1, false_positive + true_negative
    )
    false_negative_rate = false_negative / max(
        1, false_negative + true_positive
    )
    return PrecomputedConfusionMetrics(
        true_negative,
        false_positive,
        false_negative,
        true_positive,
        precision,
        recall,
        f1,
        accuracy,
        specificity,
        false_positive_rate,
        false_negative_rate,
    )


def _evaluate_precomputed_source_confusion(
    rows: Sequence[Mapping[str, object]],
    score_columns: Sequence[PrecomputedModelScoreColumn],
    metric_rows: Sequence[PrecomputedModelMetricRow],
    *,
    source_cell: str,
    target_round: int,
    threshold_mode: str,
    threshold_override: float | None,
    fixed_threshold: float | None,
    best_threshold: float | None,
    level: str,
    group_precedence: tuple[str, ...],
    normalized_display: bool,
    count_and_percent_annotations: bool,
    figure_inches: tuple[float, float],
    save_dpi: int,
    paper_style: bool,
) -> PrecomputedModelConfusionEvaluation:
    if level not in {"candidate", "image"}:
        raise ReportingContractError(
            "precomputed confusion level must be candidate or image"
        )
    if group_precedence != ("root", "img_folder"):
        raise ReportingContractError(
            "precomputed confusion group precedence must be root then img_folder"
        )
    if type(normalized_display) is not bool:
        raise ReportingContractError(
            "precomputed confusion normalized-display flag must be boolean"
        )
    if type(count_and_percent_annotations) is not bool:
        raise ReportingContractError(
            "precomputed confusion annotation flag must be boolean"
        )
    if (
        len(figure_inches) != 2
        or any(not isfinite(value) or value <= 0.0 for value in figure_inches)
        or type(save_dpi) is not int
        or save_dpi <= 0
        or type(paper_style) is not bool
    ):
        raise ReportingContractError(
            "precomputed confusion presentation settings are invalid"
        )
    declaration, selected_metric, labels, scores = (
        _select_precomputed_confusion_score(
            rows,
            score_columns,
            metric_rows,
            target_round=target_round,
            source_cell=source_cell,
        )
    )
    threshold, threshold_source = _resolve_precomputed_confusion_threshold(
        threshold_mode=threshold_mode,
        threshold_override=threshold_override,
        fixed_threshold=(
            fixed_threshold
            if fixed_threshold is not None
            else selected_metric.fixed_threshold
        ),
        best_threshold=(
            best_threshold
            if best_threshold is not None
            else selected_metric.best_threshold
        ),
    )

    effective_level = "candidate"
    group_column: str | None = None
    evaluated_labels = labels
    evaluated_scores = scores
    if level == "image":
        group_column = next(
            (
                column
                for column in group_precedence
                if all(column in row for row in rows)
            ),
            None,
        )
        if group_column is not None:
            grouped: dict[str, tuple[int, float]] = {}
            for row, label, score in zip(rows, labels, scores):
                group = str(row[group_column])
                previous_label, previous_score = grouped.get(group, (0, 0.0))
                grouped[group] = (
                    max(previous_label, label),
                    max(previous_score, score),
                )
            grouped_values = tuple(grouped[key] for key in sorted(grouped))
            evaluated_labels = tuple(item[0] for item in grouped_values)
            evaluated_scores = tuple(item[1] for item in grouped_values)
            effective_level = "image"

    predictions = tuple(int(score >= threshold) for score in evaluated_scores)
    metrics = _precomputed_confusion_metrics(evaluated_labels, predictions)
    counts = (
        (metrics.true_negative, metrics.false_positive),
        (metrics.false_negative, metrics.true_positive),
    )
    row_normalized = tuple(
        tuple(value / max(1, sum(row)) for value in row)
        for row in counts
    )
    display_values = (
        row_normalized
        if normalized_display
        else tuple(tuple(float(value) for value in row) for row in counts)
    )
    if count_and_percent_annotations:
        display_annotations = tuple(
            tuple(
                f"{count}\n({100.0 * percentage:.1f}%)"
                for count, percentage in zip(count_row, percentage_row)
            )
            for count_row, percentage_row in zip(counts, row_normalized)
        )
    elif normalized_display:
        display_annotations = tuple(
            tuple(f"{100.0 * value:.1f}%" for value in row)
            for row in row_normalized
        )
    else:
        display_annotations = tuple(
            tuple(str(value) for value in row) for row in counts
        )
    reported_metric_names = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
    )
    if source_cell == "NB-LIVE-0012-C0012":
        reported_metric_names += ("false_positive_rate", "false_negative_rate")
    return PrecomputedModelConfusionEvaluation(
        source_cell,
        target_round,
        selected_metric.round,
        declaration.model_rel,
        declaration.column,
        selected_metric.pr_auc,
        threshold_mode,
        threshold,
        threshold_source,
        level,
        effective_level,
        group_column,
        len(evaluated_labels),
        sum(evaluated_labels),
        counts,
        row_normalized,
        display_values,
        display_annotations,
        normalized_display,
        count_and_percent_annotations,
        reported_metric_names,
        (float(figure_inches[0]), float(figure_inches[1])),
        save_dpi,
        paper_style,
        metrics,
    )


# SOURCE_CELL: NB-LIVE-0012-C0011
# SOURCE_STATEMENT_MAP: round-92-count-or-normalized-confusion -> evaluate_precomputed_c0011_confusion
def evaluate_precomputed_c0011_confusion(
    rows: Sequence[Mapping[str, object]],
    score_columns: Sequence[PrecomputedModelScoreColumn],
    metric_rows: Sequence[PrecomputedModelMetricRow],
    *,
    target_round: int = 92,
    threshold_mode: str = "fixed",
    threshold_override: float | None = None,
    fixed_threshold: float | None = None,
    best_threshold: float | None = None,
    level: str = "candidate",
    normalize: bool = False,
    group_precedence: tuple[str, ...] = ("root", "img_folder"),
    figure_inches: tuple[float, float] = (3.2, 3.0),
    save_dpi: int = 600,
    paper_style: bool = True,
) -> PrecomputedModelConfusionEvaluation:
    """Evaluate the C11 count/optional-normalized matrix without loading a model."""

    return _evaluate_precomputed_source_confusion(
        rows,
        score_columns,
        metric_rows,
        source_cell="NB-LIVE-0012-C0011",
        target_round=target_round,
        threshold_mode=threshold_mode,
        threshold_override=threshold_override,
        fixed_threshold=fixed_threshold,
        best_threshold=best_threshold,
        level=level,
        group_precedence=group_precedence,
        normalized_display=normalize,
        count_and_percent_annotations=False,
        figure_inches=figure_inches,
        save_dpi=save_dpi,
        paper_style=paper_style,
    )


# SOURCE_CELL: NB-LIVE-0012-C0012
# SOURCE_STATEMENT_MAP: round-92-row-normalized-count-percent-confusion -> evaluate_precomputed_c0012_confusion
def evaluate_precomputed_c0012_confusion(
    rows: Sequence[Mapping[str, object]],
    score_columns: Sequence[PrecomputedModelScoreColumn],
    metric_rows: Sequence[PrecomputedModelMetricRow],
    *,
    target_round: int = 92,
    threshold_mode: str = "fixed",
    threshold_override: float | None = None,
    fixed_threshold: float | None = None,
    best_threshold: float | None = None,
    level: str = "candidate",
    group_precedence: tuple[str, ...] = ("root", "img_folder"),
    figure_inches: tuple[float, float] = (3.6, 3.1),
    save_dpi: int = 600,
    paper_style: bool = True,
) -> PrecomputedModelConfusionEvaluation:
    """Evaluate C12's row-normalized matrix with count/percent annotations."""

    return _evaluate_precomputed_source_confusion(
        rows,
        score_columns,
        metric_rows,
        source_cell="NB-LIVE-0012-C0012",
        target_round=target_round,
        threshold_mode=threshold_mode,
        threshold_override=threshold_override,
        fixed_threshold=fixed_threshold,
        best_threshold=best_threshold,
        level=level,
        group_precedence=group_precedence,
        normalized_display=True,
        count_and_percent_annotations=True,
        figure_inches=figure_inches,
        save_dpi=save_dpi,
        paper_style=paper_style,
    )


# SOURCE_CELL: NB-LIVE-0012-C0011
# SOURCE_CELL: NB-LIVE-0012-C0012
# SOURCE_STATEMENT_MAP: model-threshold-confusion-count-percent -> confusion_matrix_report
def confusion_matrix_report(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
) -> Mapping[str, Any]:
    metrics = classification_metrics(labels, scores, threshold)
    total = max(1, len(labels))
    return {
        "threshold": threshold,
        "counts": (
            (metrics.true_negative, metrics.false_positive),
            (metrics.false_negative, metrics.true_positive),
        ),
        "percentages": (
            (metrics.true_negative / total, metrics.false_positive / total),
            (metrics.false_negative / total, metrics.true_positive / total),
        ),
        "metrics": metrics,
    }


# SOURCE_CELL: NB-LIVE-0012-C0011
# SOURCE_CELL: NB-LIVE-0012-C0012
# SOURCE_STATEMENT_MAP: round-score-fixed-threshold-candidate-image-confusion -> evaluate_precomputed_model_confusion
def evaluate_precomputed_model_confusion(
    rows: Sequence[Mapping[str, Any]],
    columns: PrecomputedHybridScoreColumns,
    *,
    target_round: int = 92,
    threshold: float = 0.50,
    level: str = "candidate",
) -> HybridEvaluation:
    if columns.target_round != target_round:
        raise ReportingContractError("model confusion target round changed")
    if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ReportingContractError("model confusion threshold must be in [0, 1]")
    if level not in {"candidate", "image"}:
        raise ReportingContractError("model confusion level must be candidate or image")
    selected = [row for row in rows if _binary(row.get("human_label")) is not None]
    if not selected:
        raise ReportingContractError("model confusion requires labeled rows")
    labels = [int(_binary(row.get("human_label")) or 0) for row in selected]
    scores = [_float(row.get(columns.xgb_probability)) for row in selected]
    if any(not isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
        raise ReportingContractError("model confusion scores must be probabilities")
    predictions = [int(score >= threshold) for score in scores]
    group_column: str | None = None
    if level == "image":
        group_column = next(
            (
                name
                for name in columns.group_precedence
                if all(str(row.get(name, "")).strip() for row in selected)
            ),
            None,
        )
        if group_column is None:
            raise ReportingContractError("image confusion requires a declared group column")
        grouped: dict[str, tuple[int, int]] = {}
        for row, label, prediction in zip(selected, labels, predictions):
            key = str(row[group_column])
            previous = grouped.get(key, (0, 0))
            grouped[key] = (
                max(previous[0], label),
                max(previous[1], prediction),
            )
        grouped_values = [grouped[key] for key in sorted(grouped)]
        labels = [item[0] for item in grouped_values]
        predictions = [item[1] for item in grouped_values]
    return _hybrid_evaluation(
        labels,
        predictions,
        level=f"{level}-level",
        variant=f"xgb_round_{target_round}",
        group_column=group_column,
    )


# SOURCE_CELL: NB-LIVE-0012-C0015
# SOURCE_STATEMENT_MAP: thresholded-xgb-yolo-hybrid-policy-smart-rules -> hybrid_predictions
def _validate_hybrid_policy(policy: HybridPolicy) -> None:
    probabilities = (
        policy.xgb_threshold,
        policy.yolo_threshold,
        policy.yolo_iou_threshold,
        policy.detector_threshold,
        policy.yolo_bias,
        policy.rule_negative_xgb_maximum,
        policy.rule_negative_yolo_maximum,
        policy.rule_positive_yolo_minimum,
        policy.rule_positive_xgb_minimum,
        policy.rule_yolo_solo_minimum,
        policy.xgb_force_negative_threshold,
        policy.xgb_force_negative_yolo_minimum,
    )
    if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ReportingContractError("hybrid probability thresholds must be in [0, 1]")
    if policy.xgb_force_positive_threshold is not None and (
        not isfinite(policy.xgb_force_positive_threshold)
        or not 0.0 <= policy.xgb_force_positive_threshold <= 1.0
    ):
        raise ReportingContractError("hybrid force-positive threshold must be in [0, 1]")
    if policy.missing_policy not in {"reject", "ignore"}:
        raise ReportingContractError("hybrid missing policy must be reject or ignore")


def _hybrid_prediction_record(
    row: Mapping[str, Any],
    columns: PrecomputedHybridScoreColumns,
    policy: HybridPolicy,
    *,
    smart: bool,
) -> HybridPrediction:
    xgb = _float(row.get(columns.xgb_probability))
    yolo_confidence = _float(row.get(columns.yolo_confidence))
    yolo_iou = _float(row.get(columns.yolo_iou))
    xgb_available = isfinite(xgb)
    xgb_positive = xgb_available and xgb >= policy.xgb_threshold
    yolo_available = isfinite(yolo_confidence) and yolo_confidence > 0.0
    yolo_accepted = (
        yolo_available
        and yolo_confidence >= policy.yolo_threshold
        and isfinite(yolo_iou)
        and yolo_iou >= policy.yolo_iou_threshold
    )
    if not yolo_accepted:
        fused = xgb if xgb_available else 0.0
        if policy.missing_policy == "reject":
            return HybridPrediction(
                0,
                fused,
                "xgb_only_reject_missing_yolo" if xgb_available else "reject_missing_both",
            )
        return HybridPrediction(
            int(xgb_positive),
            fused,
            "xgb_only_ignore_missing_yolo" if xgb_available else "no_xgb_no_yolo",
        )

    prediction = int(yolo_confidence >= policy.detector_threshold)
    fused = yolo_confidence
    rule = "yolo_only"
    if not xgb_available:
        return HybridPrediction(prediction, fused, rule)

    if smart and policy.xgb_force_enabled:
        positive_threshold = (
            policy.detector_threshold
            if policy.xgb_force_positive_threshold is None
            else policy.xgb_force_positive_threshold
        )
        if xgb >= positive_threshold:
            return HybridPrediction(1, xgb, "xgb_force_pos")
        yolo_requirement_met = (
            policy.xgb_force_negative_yolo_minimum <= 0.0
            or yolo_confidence >= policy.xgb_force_negative_yolo_minimum
        )
        if xgb <= policy.xgb_force_negative_threshold and yolo_requirement_met:
            return HybridPrediction(0, xgb, "xgb_force_neg")

    if smart and policy.smart_rules_enabled:
        if (
            xgb <= policy.rule_negative_xgb_maximum
            and yolo_confidence <= policy.rule_negative_yolo_maximum
        ):
            return HybridPrediction(0, xgb, "rule_noncj")
        if (
            yolo_confidence >= policy.rule_positive_yolo_minimum
            and xgb >= policy.rule_positive_xgb_minimum
        ):
            return HybridPrediction(
                1,
                0.8 * yolo_confidence + 0.2 * xgb,
                "rule_cj",
            )
        if (
            yolo_confidence >= policy.rule_yolo_solo_minimum
            and xgb < policy.rule_positive_xgb_minimum
        ):
            return HybridPrediction(1, yolo_confidence, "rule_cj_yolo_solo")

    denominator = max(yolo_confidence + xgb, 1e-12)
    yolo_weight = yolo_confidence / denominator
    if xgb >= policy.detector_threshold and yolo_confidence < policy.detector_threshold:
        yolo_weight = max(policy.yolo_bias, yolo_weight)
    fused = yolo_weight * yolo_confidence + (1.0 - yolo_weight) * xgb
    if xgb >= policy.detector_threshold and yolo_confidence >= policy.detector_threshold:
        prediction, rule = 1, "hybrid_both_pos"
    elif xgb < policy.detector_threshold and yolo_confidence < policy.detector_threshold:
        prediction, rule = 0, "hybrid_both_neg"
    else:
        prediction, rule = int(fused >= policy.detector_threshold), "hybrid_mix"
    return HybridPrediction(prediction, fused, rule)


def hybrid_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: PrecomputedHybridScoreColumns = PrecomputedHybridScoreColumns(),
    policy: HybridPolicy = HybridPolicy(),
    smart: bool = True,
) -> list[int]:
    _validate_hybrid_policy(policy)
    return [
        _hybrid_prediction_record(row, columns, policy, smart=smart).prediction
        for row in rows
    ]


def _hybrid_metrics(labels: Sequence[int], predictions: Sequence[int]) -> HybridMetrics:
    if len(labels) != len(predictions) or not labels:
        raise ReportingContractError("hybrid labels and predictions must align")
    if any(value not in {0, 1} for value in (*labels, *predictions)):
        raise ReportingContractError("hybrid labels and predictions must be binary")
    true_positive = sum(label == 1 and value == 1 for label, value in zip(labels, predictions))
    false_positive = sum(label == 0 and value == 1 for label, value in zip(labels, predictions))
    false_negative = sum(label == 1 and value == 0 for label, value in zip(labels, predictions))
    true_negative = sum(label == 0 and value == 0 for label, value in zip(labels, predictions))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return HybridMetrics(
        true_negative,
        false_positive,
        false_negative,
        true_positive,
        (true_positive + true_negative) / len(labels),
        precision,
        recall,
        2.0 * precision * recall / max(1e-12, precision + recall),
        true_negative / max(1, true_negative + false_positive),
        false_positive / max(1, false_positive + true_negative),
        false_negative / max(1, false_negative + true_positive),
    )


def _hybrid_evaluation(
    labels: Sequence[int],
    predictions: Sequence[int],
    *,
    level: str,
    variant: str,
    group_column: str | None,
) -> HybridEvaluation:
    metrics = _hybrid_metrics(labels, predictions)
    canonical = confusion_matrix_report(labels, predictions, threshold=0.50)
    canonical_counts = tuple(tuple(int(value) for value in row) for row in canonical["counts"])
    if canonical_counts != (
        (metrics.true_negative, metrics.false_positive),
        (metrics.false_negative, metrics.true_positive),
    ):
        raise ReportingContractError("hybrid confusion helpers disagree")
    negative_total = max(1, metrics.true_negative + metrics.false_positive)
    positive_total = max(1, metrics.false_negative + metrics.true_positive)
    return HybridEvaluation(
        level,
        variant,
        group_column,
        sum(predictions),
        (
            (
                metrics.true_negative / negative_total,
                metrics.false_positive / negative_total,
            ),
            (
                metrics.false_negative / positive_total,
                metrics.true_positive / positive_total,
            ),
        ),
        metrics,
    )


# SOURCE_STATEMENT_MAP: candidate-image-basic-smart-confusion-order -> evaluate_hybrid_policy
def evaluate_hybrid_policy(
    rows: Sequence[Mapping[str, Any]],
    columns: PrecomputedHybridScoreColumns,
    policy: HybridPolicy = HybridPolicy(),
) -> tuple[HybridEvaluation, ...]:
    _validate_hybrid_policy(policy)
    labeled_rows = [row for row in rows if _binary(row.get("human_label")) is not None]
    if not labeled_rows:
        raise ReportingContractError("hybrid evaluation requires labeled score rows")
    labels = [int(_binary(row.get("human_label")) or 0) for row in labeled_rows]
    basic = hybrid_predictions(labeled_rows, columns=columns, policy=policy, smart=False)
    smart = hybrid_predictions(labeled_rows, columns=columns, policy=policy, smart=True)
    output = [
        _hybrid_evaluation(
            labels,
            basic,
            level="candidate-level",
            variant="hybrid_basic",
            group_column=None,
        ),
        _hybrid_evaluation(
            labels,
            smart,
            level="candidate-level",
            variant="hybrid_smart",
            group_column=None,
        ),
    ]
    group_column = next(
        (
            name
            for name in columns.group_precedence
            if all(str(row.get(name, "")).strip() for row in labeled_rows)
        ),
        None,
    )
    if group_column is not None:
        grouped: dict[str, tuple[int, int, int]] = {}
        for row, label, basic_value, smart_value in zip(
            labeled_rows,
            labels,
            basic,
            smart,
        ):
            key = str(row[group_column])
            previous = grouped.get(key, (0, 0, 0))
            grouped[key] = (
                max(previous[0], label),
                max(previous[1], basic_value),
                max(previous[2], smart_value),
            )
        grouped_values = [grouped[key] for key in sorted(grouped)]
        image_labels = [item[0] for item in grouped_values]
        image_basic = [item[1] for item in grouped_values]
        image_smart = [item[2] for item in grouped_values]
        output.extend(
            (
                _hybrid_evaluation(
                    image_labels,
                    image_basic,
                    level="image-level",
                    variant="hybrid_basic",
                    group_column=group_column,
                ),
                _hybrid_evaluation(
                    image_labels,
                    image_smart,
                    level="image-level",
                    variant="hybrid_smart",
                    group_column=group_column,
                ),
            )
        )
    return tuple(output)


def _normalize_mask(mask: Sequence[Sequence[Any]]) -> tuple[tuple[bool, ...], ...]:
    rows = tuple(tuple(bool(value) for value in row) for row in mask)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ReportingContractError("mask must be a non-empty rectangle")
    return rows


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_CELL: NB-LIVE-0012-C0019
# SOURCE_STATEMENT_MAP: uint8-mask-iou-border-sort-multiscale-dedup -> mask_iou,touches_border,merge_multiscale_masks
def mask_iou(left: Sequence[Sequence[Any]], right: Sequence[Sequence[Any]]) -> float:
    a = _normalize_mask(left)
    b = _normalize_mask(right)
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ReportingContractError("mask shapes differ")
    intersection = sum(x and y for ar, br in zip(a, b) for x, y in zip(ar, br))
    union = sum(x or y for ar, br in zip(a, b) for x, y in zip(ar, br))
    return float(intersection) / float(union + 1e-9)


def touches_border(mask: Sequence[Sequence[Any]], margin: int = 0) -> bool:
    value = _normalize_mask(mask)
    height, width = len(value), len(value[0])
    if margin <= 0:
        return any(
            value[y][x]
            for y in range(height)
            for x in range(width)
            if y == 0 or x == 0 or y == height - 1 or x == width - 1
        )
    return any(
        value[y][x]
        for y in range(height)
        for x in range(width)
        if y < margin or x < margin or y >= height - margin or x >= width - margin
    )


def bbox_from_mask(mask: Sequence[Sequence[Any]]) -> tuple[int, int, int, int] | None:
    value = _normalize_mask(mask)
    points = [(x, y) for y, row in enumerate(value) for x, bit in enumerate(row) if bit]
    if not points:
        return None
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


def centroid_from_mask(mask: Sequence[Sequence[Any]]) -> tuple[float, float] | None:
    value = _normalize_mask(mask)
    points = [(x, y) for y, row in enumerate(value) for x, bit in enumerate(row) if bit]
    if not points:
        return None
    return sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points)


def merge_multiscale_masks(
    masks: Sequence[MaskRecord],
    *,
    merge_iou: float = 0.95,
) -> tuple[MaskRecord, ...]:
    ordered = sorted(masks, key=lambda item: item.area, reverse=True)
    kept: list[MaskRecord] = []
    for candidate in ordered:
        if all(mask_iou(candidate.segmentation, existing.segmentation) < merge_iou for existing in kept):
            kept.append(candidate)
    return tuple(kept)


def generate_local_sam_masks(
    request: SamMaskRequest,
    generator: SamMaskGenerator,
) -> tuple[MaskRecord, ...]:
    for path, role in (
        (request.image, "tile image"),
        (request.checkpoint, "SAM2 checkpoint"),
    ):
        if not path.is_file():
            raise ReportingContractError(f"missing explicit local {role}: {path}")
    if not request.repository.is_dir():
        raise ReportingContractError(f"missing explicit local SAM2 repository: {request.repository}")
    generated = generator.generate(request)
    masks: list[MaskRecord] = []
    for item in generated:
        segmentation = _normalize_mask(item["segmentation"])
        masks.append(
            MaskRecord(
                segmentation,
                int(item.get("area", sum(bit for row in segmentation for bit in row))),
                _float(item.get("predicted_iou")),
                _float(item.get("stability_score")),
            )
        )
    return merge_multiscale_masks(masks, merge_iou=request.merge_iou)


# SOURCE_CELL: NB-LIVE-0012-C0018
# SOURCE_STATEMENT_MAP: insect-like-area-border-filter-quality-fallback -> pick_insect_like_mask
def pick_insect_like_mask(
    masks: Sequence[MaskRecord],
    *,
    image_width: int,
    image_height: int,
    minimum_area: int = 50,
    maximum_area_fraction: float = 0.20,
    border_margin: int = 2,
) -> MaskRecord:
    if image_width <= 0 or image_height <= 0:
        raise ReportingContractError("image dimensions must be positive")
    if not masks:
        raise ReportingContractError("at least one mask is required")
    maximum_area = maximum_area_fraction * image_width * image_height
    candidates = tuple(
        mask
        for mask in masks
        if minimum_area <= mask.area <= maximum_area
        and not touches_border(mask.segmentation, margin=border_margin)
    )
    if candidates:
        return max(
            candidates,
            key=lambda mask: _float(mask.predicted_iou, 0.0)
            * _float(mask.stability_score, 1.0),
        )
    non_border = tuple(
        mask
        for mask in masks
        if not touches_border(mask.segmentation, margin=border_margin)
    )
    return max(
        non_border or tuple(masks),
        key=lambda mask: _float(mask.predicted_iou, 0.0),
    )


def _padded_mask_bbox(
    mask: Sequence[Sequence[Any]],
    padding_fraction: float,
) -> tuple[int, int, int, int]:
    normalized = _normalize_mask(mask)
    bbox = bbox_from_mask(normalized)
    if bbox is None:
        raise ReportingContractError("selected mask has no foreground pixels")
    x, y, width, height = bbox
    padding = int(round(padding_fraction * max(width, height)))
    return (
        max(0, x - padding),
        max(0, y - padding),
        min(len(normalized[0]) - 1, x + width - 1 + padding),
        min(len(normalized) - 1, y + height - 1 + padding),
    )


# SOURCE_CELL: NB-LIVE-0012-C0018
# SOURCE_STATEMENT_MAP: exact-insect-choice-viz-filter-crop-inset-settings -> evaluate_precomputed_insect_mask
def evaluate_precomputed_insect_mask(
    batch: PrecomputedMaskBatch,
    *,
    minimum_area: int = 80,
    maximum_area_fraction: float = 0.15,
    border_margin: int = 2,
    visualization_maximum_area_fraction: float = 0.25,
    visualization_hide_border: bool = True,
    crop_padding: float = 0.10,
    inset_padding: float = 0.15,
    all_masks_alpha: float = 0.45,
    chosen_mask_alpha: float = 0.45,
    figure_inches: tuple[float, float] = (10.0, 10.0),
    save_dpi: int = 200,
) -> InsectMaskSelectionEvaluation:
    """Apply C18's exact selection and publication settings to sealed masks."""

    if (
        minimum_area < 0
        or not 0.0 < maximum_area_fraction <= 1.0
        or border_margin < 0
        or not 0.0 < visualization_maximum_area_fraction <= 1.0
        or type(visualization_hide_border) is not bool
        or crop_padding < 0.0
        or inset_padding < 0.0
        or not 0.0 < all_masks_alpha <= 1.0
        or not 0.0 < chosen_mask_alpha <= 1.0
        or len(figure_inches) != 2
        or any(not isfinite(value) or value <= 0.0 for value in figure_inches)
        or save_dpi <= 0
    ):
        raise ReportingContractError("insect mask selection settings are invalid")
    if not batch.candidates or batch.width <= 0 or batch.height <= 0:
        raise ReportingContractError("insect mask selection requires sealed candidates")

    records = tuple(
        MaskRecord(
            candidate.segmentation,
            candidate.area,
            candidate.predicted_iou,
            candidate.stability_score,
        )
        for candidate in batch.candidates
    )
    selected_record = pick_insect_like_mask(
        records,
        image_width=batch.width,
        image_height=batch.height,
        minimum_area=minimum_area,
        maximum_area_fraction=maximum_area_fraction,
        border_margin=border_margin,
    )
    selected_index = next(
        index for index, record in enumerate(records) if record is selected_record
    )
    selected_candidate = batch.candidates[selected_index]
    visualization_candidates = tuple(
        candidate
        for candidate in sorted(
            batch.candidates,
            key=lambda item: item.area,
            reverse=True,
        )
        if candidate.area
        <= visualization_maximum_area_fraction * batch.width * batch.height
        and (
            not visualization_hide_border
            or not touches_border(candidate.segmentation, margin=border_margin)
        )
    )
    return InsectMaskSelectionEvaluation(
        minimum_area,
        maximum_area_fraction,
        border_margin,
        visualization_maximum_area_fraction,
        visualization_hide_border,
        crop_padding,
        inset_padding,
        all_masks_alpha,
        chosen_mask_alpha,
        figure_inches,
        save_dpi,
        selected_candidate,
        tuple(candidate.candidate_id for candidate in visualization_candidates),
        _padded_mask_bbox(selected_candidate.segmentation, crop_padding),
        _padded_mask_bbox(selected_candidate.segmentation, inset_padding),
    )


# SOURCE_CELL: NB-LIVE-0012-C0021
# SOURCE_STATEMENT_MAP: explicit-selected-index-validation-and-metadata -> select_masks
def select_masks(masks: Sequence[MaskRecord], selected_indices: Sequence[int]) -> tuple[MaskRecord, ...]:
    if len(set(selected_indices)) != len(selected_indices):
        raise ReportingContractError("selected mask indices must be unique")
    if any(index < 0 or index >= len(masks) for index in selected_indices):
        raise ReportingContractError("selected mask index is outside the generated mask list")
    return tuple(masks[index] for index in selected_indices)


# SOURCE_CELL: NB-LIVE-0012-C0020
# SOURCE_STATEMENT_MAP: locked-four-mask-selection-order -> select_source_masks
def select_source_masks(
    masks: Sequence[MaskRecord],
    selected_indices: Sequence[int] = (4, 8, 19, 11),
) -> tuple[MaskRecord, ...]:
    return select_masks(masks, selected_indices)


def _ring(mask: Sequence[Sequence[Any]], pixels: int) -> tuple[tuple[bool, ...], ...]:
    value = _normalize_mask(mask)
    height, width = len(value), len(value[0])
    if pixels < 0:
        raise ReportingContractError("ring width cannot be negative")
    if pixels == 0:
        return tuple(tuple(False for _ in range(width)) for _ in range(height))
    offsets: list[tuple[int, int]] = []
    for dy in range(-pixels, pixels + 1):
        dx = round(pixels * sqrt(max(0.0, 1.0 - (dy * dy) / (pixels * pixels))))
        offsets.extend((offset_x, dy) for offset_x in range(-dx, dx + 1))
    expanded = [[False] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            if not value[y][x]:
                continue
            for dx, dy in offsets:
                xx, yy = x + dx, y + dy
                if 0 <= xx < width and 0 <= yy < height:
                    expanded[yy][xx] = True
    return tuple(tuple(expanded[y][x] and not value[y][x] for x in range(width)) for y in range(height))


# SOURCE_CELL: NB-LIVE-0012-C0022
# SOURCE_CELL: NB-LIVE-0012-C0023
# SOURCE_STATEMENT_MAP: bbox-ring-subsample-quality-pca-panel-data -> candidate_panel_statistics
def candidate_panel_statistics(
    mask: Sequence[Sequence[Any]],
    *,
    predicted_iou: float,
    stability: float,
    ring_pixels: int = 5,
    zoom_padding: float = 0.20,
) -> CandidateMaskStatistics:
    normalized = _normalize_mask(mask)
    area = sum(bit for row in normalized for bit in row)
    bbox_xywh = bbox_from_mask(normalized)
    if bbox_xywh is None:
        raise ReportingContractError("candidate mask has no foreground pixels")
    x, y, width, height = bbox_xywh
    x1, y1 = x + width - 1, y + height - 1
    padding = int(round(zoom_padding * max(width, height)))
    xx0, yy0 = max(0, x - padding), max(0, y - padding)
    xx1 = min(len(normalized[0]) - 1, x1 + padding)
    yy1 = min(len(normalized) - 1, y1 + padding)
    zoom_mask = tuple(
        tuple(row[xx0 : xx1 + 1]) for row in normalized[yy0 : yy1 + 1]
    )
    ring = _ring(zoom_mask, ring_pixels)
    ring_area = sum(bit for row in ring for bit in row)
    centroid = centroid_from_mask(normalized)
    if centroid is None:
        raise ReportingContractError("candidate mask has no centroid")
    return CandidateMaskStatistics(
        area,
        (x, y, x1, y1),
        centroid,
        ring_area,
        xx1 - xx0 + 1,
        yy1 - yy0 + 1,
        predicted_iou,
        stability,
        predicted_iou >= 0.80 and stability >= 0.88,
        touches_border(normalized),
    )


# SOURCE_CELL: NB-LIVE-0012-C0022
# SOURCE_CELL: NB-LIVE-0012-C0023
# SOURCE_STATEMENT_MAP: sealed-panel-features-light-gate-pca-consumer -> evaluate_precomputed_candidate_panels
def evaluate_precomputed_candidate_panels(
    panels: Sequence[PrecomputedCandidatePanel],
    pca_scatter: Sequence[PrecomputedPcaPoint],
    settings: CandidatePanelSettings = CandidatePanelSettings(),
    *,
    mask_candidates: Sequence[PrecomputedMaskCandidate] = (),
) -> tuple[CandidatePanelEvaluation, ...]:
    """Consume sealed image/model intermediates using the source panel rules."""

    if len(panels) != 4:
        raise ReportingContractError("candidate panel contract requires exactly four selections")
    if len({panel.candidate_id for panel in panels}) != len(panels):
        raise ReportingContractError("candidate panel identifiers must be unique")
    if not 0.0 <= settings.zoom_padding <= 1.0:
        raise ReportingContractError("candidate zoom padding is outside [0, 1]")
    if not 0.0 <= settings.crop_padding <= 1.0:
        raise ReportingContractError("candidate crop padding is outside [0, 1]")
    if settings.ring_pixels <= 0 or settings.subsample_count <= 0:
        raise ReportingContractError("candidate ring and subsample settings must be positive")
    if not 0.0 <= settings.light_low_threshold < settings.light_high_threshold <= 255.0:
        raise ReportingContractError("candidate light thresholds are invalid")
    numeric_settings = (
        settings.minimum_l,
        settings.minimum_delta_a,
        settings.maximum_delta_b,
        settings.minimum_elongation,
        settings.maximum_elongation,
        settings.minimum_solidity,
        settings.maximum_low_clip_fraction,
        settings.maximum_high_clip_fraction,
        settings.maximum_l_standard_deviation,
        settings.maximum_background_delta_l,
    )
    if (
        any(not isfinite(value) for value in numeric_settings)
        or not 0.0 <= settings.predicted_iou_threshold <= 1.0
        or not 0.0 <= settings.stability_threshold <= 1.0
        or settings.minimum_elongation > settings.maximum_elongation
        or not 0.0 <= settings.minimum_solidity <= 1.0
        or not 0.0 <= settings.maximum_low_clip_fraction <= 1.0
        or not 0.0 <= settings.maximum_high_clip_fraction <= 1.0
        or settings.maximum_l_standard_deviation < 0.0
        or settings.maximum_background_delta_l < 0.0
    ):
        raise ReportingContractError("candidate feature gate settings are invalid")
    if settings.pca_print_components <= 0 or settings.pca_scatter_max_masks <= 0:
        raise ReportingContractError("candidate PCA limits must be positive")
    if sum(not point.selected for point in pca_scatter) > settings.pca_scatter_max_masks:
        raise ReportingContractError("precomputed PCA scatter exceeds the sealed mask limit")
    if not 0.0 < settings.pca_scatter_alpha <= 1.0:
        raise ReportingContractError("candidate PCA scatter alpha is outside (0, 1]")
    if len({point.candidate_id for point in pca_scatter}) != len(pca_scatter):
        raise ReportingContractError("candidate PCA point identifiers must be unique")
    if any(
        not isfinite(value)
        for point in pca_scatter
        for value in (point.pc1, point.pc2)
    ):
        raise ReportingContractError("candidate PCA coordinates must be finite")
    selected_ids = {panel.candidate_id for panel in panels}
    if any(point.selected and point.candidate_id not in selected_ids for point in pca_scatter):
        raise ReportingContractError("PCA selected points do not match candidate panels")
    masks_by_id = {candidate.candidate_id: candidate for candidate in mask_candidates}
    if len(masks_by_id) != len(mask_candidates):
        raise ReportingContractError("candidate mask identifiers must be unique")
    if mask_candidates and any(identifier not in masks_by_id for identifier in selected_ids):
        raise ReportingContractError("candidate panels do not match sealed masks")

    output: list[CandidatePanelEvaluation] = []
    for panel in panels:
        x0, y0, x1, y1 = panel.bbox
        if x0 < 0 or y0 < 0 or x1 < x0 or y1 < y0:
            raise ReportingContractError("candidate bounding box is invalid")
        width, height = x1 - x0 + 1, y1 - y0 + 1
        padding = int(round(settings.zoom_padding * max(width, height)))
        if not (
            width <= panel.zoom_width <= width + 2 * padding
            and height <= panel.zoom_height <= height + 2 * padding
        ):
            raise ReportingContractError("candidate zoom geometry violates the sealed padding")
        if panel.mask_area <= 0 or panel.ring_area < 0:
            raise ReportingContractError("candidate mask and ring areas are invalid")
        bounded_values = (
            panel.predicted_iou,
            panel.stability_score,
            panel.low_clip_fraction,
            panel.high_clip_fraction,
            panel.area_norm,
            panel.extent,
            panel.solidity,
        )
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in bounded_values):
            raise ReportingContractError("candidate bounded intermediates are invalid")
        scalar_values = (
            panel.mean_l,
            panel.mean_a,
            panel.mean_b,
            panel.background_a,
            panel.background_b,
            panel.delta_l_median,
            panel.l_standard_deviation,
            panel.elongation,
            panel.aspect_ratio,
            panel.circularity,
        )
        if any(not isfinite(value) for value in scalar_values):
            raise ReportingContractError("candidate scalar intermediates must be finite")
        paired_samples = (
            (panel.mask_a, panel.mask_b),
            (panel.ring_a, panel.ring_b),
        )
        if any(len(left) != len(right) for left, right in paired_samples):
            raise ReportingContractError("candidate chroma sample arrays must align")
        if any(
            len(values) > settings.subsample_count
            for pair in paired_samples
            for values in pair
        ):
            raise ReportingContractError("candidate chroma samples exceed the sealed limit")
        if not panel.gradient_magnitude:
            raise ReportingContractError("candidate gradient intermediate must be nonempty")
        if any(
            not isfinite(value)
            for values in (
                panel.mask_a,
                panel.mask_b,
                panel.ring_a,
                panel.ring_b,
                panel.gradient_magnitude,
                panel.pca_components,
            )
            for value in values
        ) or any(value < 0.0 for value in panel.gradient_magnitude):
            raise ReportingContractError("candidate array intermediates are invalid")
        if mask_candidates:
            candidate = masks_by_id[panel.candidate_id]
            statistics = candidate_panel_statistics(
                candidate.segmentation,
                predicted_iou=candidate.predicted_iou,
                stability=candidate.stability_score,
                ring_pixels=settings.ring_pixels,
                zoom_padding=settings.zoom_padding,
            )
            if (
                statistics.bbox_xyxy != panel.bbox
                or statistics.area != panel.mask_area
                or statistics.ring_area != panel.ring_area
                or statistics.zoom_width != panel.zoom_width
                or statistics.zoom_height != panel.zoom_height
                or statistics.predicted_iou != panel.predicted_iou
                or statistics.stability_score != panel.stability_score
            ):
                raise ReportingContractError(
                    "candidate panel geometry or quality does not match sealed masks"
                )

        delta_a = panel.background_a - panel.mean_a
        delta_b = panel.mean_b - panel.background_b
        light_failures: list[str] = []
        if panel.low_clip_fraction > settings.maximum_low_clip_fraction:
            light_failures.append("low_clip")
        if panel.high_clip_fraction > settings.maximum_high_clip_fraction:
            light_failures.append("high_clip")
        if panel.l_standard_deviation > settings.maximum_l_standard_deviation:
            light_failures.append("stdL")
        if abs(panel.delta_l_median) > settings.maximum_background_delta_l:
            light_failures.append("deltaL")
        color_failures: list[str] = []
        if delta_a < settings.minimum_delta_a:
            color_failures.append("delta_a")
        if delta_b > settings.maximum_delta_b:
            color_failures.append("delta_b")
        shape_failures: list[str] = []
        if not settings.minimum_elongation <= panel.elongation <= settings.maximum_elongation:
            shape_failures.append("elongation")
        if panel.solidity < settings.minimum_solidity:
            shape_failures.append("solidity")

        quality = (
            panel.predicted_iou >= settings.predicted_iou_threshold
            and panel.stability_score >= settings.stability_threshold
        )
        gradient_mean = sum(panel.gradient_magnitude) / len(panel.gradient_magnitude)
        pca_available = len(panel.pca_components) >= 2
        selected_in_scatter = any(
            point.candidate_id == panel.candidate_id and point.selected
            for point in pca_scatter
        )
        output.append(
            CandidatePanelEvaluation(
                panel,
                delta_a,
                delta_b,
                quality,
                not light_failures,
                tuple(light_failures),
                not color_failures,
                tuple(color_failures),
                not shape_failures,
                tuple(shape_failures),
                pca_available,
                selected_in_scatter,
                gradient_mean,
                settings.crop_padding,
                (
                    panel.pca_components[: settings.pca_print_components]
                    if pca_available
                    else ()
                ),
            )
        )
    return tuple(output)


def polygon_to_mask(
    points: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> tuple[tuple[bool, ...], ...]:
    if len(points) < 3:
        return tuple(tuple(False for _ in range(width)) for _ in range(height))
    output: list[tuple[bool, ...]] = []
    for y in range(height):
        row: list[bool] = []
        for x in range(width):
            inside = False
            j = len(points) - 1
            for i in range(len(points)):
                xi, yi = float(points[i][0]), float(points[i][1])
                xj, yj = float(points[j][0]), float(points[j][1])
                crosses = (yi > y) != (yj > y)
                if crosses and x < (xj - xi) * (y - yi) / max(1e-12, yj - yi) + xi:
                    inside = not inside
                j = i
            row.append(inside)
        output.append(tuple(row))
    return tuple(output)


# SOURCE_CELL: NB-LIVE-0012-C0024
# SOURCE_CELL: NB-LIVE-0012-C0026
# SOURCE_STATEMENT_MAP: polygon-rasterization-best-iou-selected-id -> match_selected_masks_to_detections
def match_selected_masks_to_detections(
    selected_masks: Sequence[MaskRecord],
    detections: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
) -> tuple[tuple[Any, float], ...]:
    detection_masks: list[tuple[Any, tuple[tuple[bool, ...], ...]]] = []
    for row in detections:
        segmentation = row.get("segmentation", row.get("polygon"))
        if isinstance(segmentation, str):
            try:
                segmentation = loads(segmentation)
            except ValueError:
                continue
        if not isinstance(segmentation, list):
            continue
        if (
            len(segmentation) == height
            and all(isinstance(line, list) and len(line) == width for line in segmentation)
            and all(type(bit) is bool for line in segmentation for bit in line)
        ):
            detection_mask = _normalize_mask(segmentation)
        else:
            points = (
                segmentation[0]
                if segmentation
                and isinstance(segmentation[0], list)
                and segmentation[0]
                and isinstance(segmentation[0][0], list)
                else segmentation
            )
            detection_mask = polygon_to_mask(points, width, height)
        detection_masks.append((row.get("id"), detection_mask))
    output: list[tuple[Any, float]] = []
    for selected in selected_masks:
        best_identity: Any = None
        best_iou = -1.0
        for identity, detection_mask in detection_masks:
            score = mask_iou(selected.segmentation, detection_mask)
            if score > best_iou:
                best_identity, best_iou = identity, score
        output.append((best_identity, max(0.0, best_iou)))
    return tuple(output)


# SOURCE_CELL: NB-LIVE-0012-C0027
# SOURCE_STATEMENT_MAP: similarity-vs-probability -> similarity_probability_points
def similarity_probability_points(
    rows: Sequence[Mapping[str, Any]],
    selected_detection_ids: Iterable[Any],
) -> list[dict[str, Any]]:
    selected = {str(value) for value in selected_detection_ids}
    similarity_names = ("xf_embed_sim01", "embed_sim01", "xf_embed_sim", "embed_sim")
    output: list[dict[str, Any]] = []
    for row in rows:
        name = next((candidate for candidate in similarity_names if candidate in row), None)
        if name is None:
            continue
        similarity = _float(row.get(name))
        if "01" not in name and isfinite(similarity):
            similarity = (similarity + 1.0) / 2.0
        output.append(
            {
                "id": row.get("id"),
                "similarity01": similarity,
                "probability": _float(row.get("prob")),
                "human_label": _binary(row.get("human_label")),
                "selected": str(row.get("id")) in selected,
            }
        )
    return output


# SOURCE_CELL: NB-LIVE-0012-C0025
# SOURCE_STATEMENT_MAP: selected-detection-id-projection -> select_detection_rows
def select_detection_rows(
    rows: Sequence[Mapping[str, Any]],
    detection_ids: Sequence[int] = (5, 17, 19, 9),
) -> tuple[Mapping[str, Any], ...]:
    selected = set(int(value) for value in detection_ids)
    return tuple(
        row
        for row in rows
        if int(_float(row.get("id"), -1.0)) in selected
    )


def _covariance2(points: Sequence[tuple[float, float]]) -> tuple[float, float, float]:
    if len(points) < 2:
        return 0.0, 0.0, 0.0
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    return (
        sum((x - mean_x) ** 2 for x, _ in points) / (len(points) - 1),
        sum((x - mean_x) * (y - mean_y) for x, y in points) / (len(points) - 1),
        sum((y - mean_y) ** 2 for _, y in points) / (len(points) - 1),
    )


def _inverse_covariance_score(
    positive: Sequence[tuple[float, float]],
    negative: Sequence[tuple[float, float]],
) -> float:
    if len(positive) < 2 or len(negative) < 2:
        return -inf
    mean_positive = (
        sum(x for x, _ in positive) / len(positive),
        sum(y for _, y in positive) / len(positive),
    )
    mean_negative = (
        sum(x for x, _ in negative) / len(negative),
        sum(y for _, y in negative) / len(negative),
    )
    pxx, pxy, pyy = _covariance2(positive)
    nxx, nxy, nyy = _covariance2(negative)
    xx = (pxx + nxx) / 2.0 + 1e-6
    xy = (pxy + nxy) / 2.0
    yy = (pyy + nyy) / 2.0 + 1e-6
    determinant = xx * yy - xy * xy
    if determinant <= 0:
        return -inf
    dx = mean_positive[0] - mean_negative[0]
    dy = mean_positive[1] - mean_negative[1]
    return (yy * dx * dx - 2 * xy * dx * dy + xx * dy * dy) / determinant


def _pca_class_covariance(
    points: Sequence[tuple[float, float]],
) -> tuple[float, float, float]:
    if len(points) < 2:
        return 1.0, 0.0, 1.0
    return _covariance2(points)


def _pca_separation_score(
    positive: Sequence[tuple[float, float]],
    negative: Sequence[tuple[float, float]],
    *,
    epsilon: float = 1e-3,
) -> float:
    if not positive or not negative or epsilon <= 0.0:
        return -inf
    mean_positive = (
        sum(x for x, _ in positive) / len(positive),
        sum(y for _, y in positive) / len(positive),
    )
    mean_negative = (
        sum(x for x, _ in negative) / len(negative),
        sum(y for _, y in negative) / len(negative),
    )
    pxx, pxy, pyy = _pca_class_covariance(positive)
    nxx, nxy, nyy = _pca_class_covariance(negative)
    xx = pxx + nxx + epsilon
    xy = pxy + nxy
    yy = pyy + nyy + epsilon
    determinant = xx * yy - xy * xy
    if determinant <= 0.0:
        raise ReportingContractError("regularized PCA covariance is not invertible")
    dx = mean_positive[0] - mean_negative[0]
    dy = mean_positive[1] - mean_negative[1]
    return (yy * dx * dx - 2.0 * xy * dx * dy + xx * dy * dy) / determinant


# SOURCE_CELL: NB-LIVE-0012-C0028
# SOURCE_STATEMENT_MAP: merge-labels-pca-bestpair-sim-prob-selected -> evaluate_similarity_pca
def evaluate_similarity_pca(
    rows: Sequence[Mapping[str, Any]],
    selected_matches: Sequence[tuple[Any, float]],
    *,
    similarity_column: str = "embed_sim",
    probability_candidates: tuple[str, ...] = ("prob", "xgb_p"),
    pca_prefix: str = "xf_embed_pca_",
    maximum_components: int = 12,
    separation_epsilon: float = 1e-3,
    probability_threshold: float = 0.50,
    selected_count: int = 4,
) -> SimilarityPcaEvaluation:
    if (
        not rows
        or not similarity_column
        or not probability_candidates
        or not pca_prefix
        or maximum_components < 2
        or separation_epsilon <= 0.0
        or not 0.0 <= probability_threshold <= 1.0
        or selected_count <= 0
    ):
        raise ReportingContractError("similarity/PCA settings are invalid")
    if len(selected_matches) != selected_count:
        raise ReportingContractError("similarity/PCA requires exactly four mask matches")
    selected: dict[str, tuple[int, float]] = {}
    for ordinal, (identity, score) in enumerate(selected_matches, 1):
        if identity is None or not isfinite(float(score)):
            raise ReportingContractError("selected mask match is incomplete")
        key = str(identity)
        if key in selected:
            raise ReportingContractError("selected mask matches must be unique")
        selected[key] = ordinal, float(score)

    probability_column = next(
        (
            column
            for column in probability_candidates
            if all(column in row and isfinite(_float(row.get(column))) for row in rows)
        ),
        None,
    )
    if probability_column is None:
        raise ReportingContractError("no declared probability column is complete")
    if any(
        similarity_column not in row
        or not isfinite(_float(row.get(similarity_column)))
        for row in rows
    ):
        raise ReportingContractError("declared similarity column is incomplete")

    available_columns = {
        str(column)
        for row in rows
        for column in row
        if str(column).startswith(pca_prefix)
    }

    def pca_index(column: str) -> int:
        suffix = column[len(pca_prefix) :]
        return int(suffix) if suffix.isdigit() else 10**9

    pca_columns = tuple(
        column
        for column in sorted(available_columns, key=lambda item: (pca_index(item), item))
        if pca_index(column) < maximum_components
        and all(isfinite(_float(row.get(column))) for row in rows)
    )
    if len(pca_columns) < 2:
        raise ReportingContractError("similarity/PCA requires at least two PCA columns")
    labels = tuple(_binary(row.get("human_label")) for row in rows)

    def labeled_points(first: str, second: str, label: int) -> list[tuple[float, float]]:
        return [
            (_float(row[first]), _float(row[second]))
            for row, value in zip(rows, labels)
            if value == label
        ]

    best_indices = (0, 1)
    best_score = _pca_separation_score(
        labeled_points(pca_columns[0], pca_columns[1], 1),
        labeled_points(pca_columns[0], pca_columns[1], 0),
        epsilon=separation_epsilon,
    )
    for first in range(len(pca_columns)):
        for second in range(first + 1, len(pca_columns)):
            score = _pca_separation_score(
                labeled_points(pca_columns[first], pca_columns[second], 1),
                labeled_points(pca_columns[first], pca_columns[second], 0),
                epsilon=separation_epsilon,
            )
            if score > best_score:
                best_indices = first, second
                best_score = score
    default_columns = pca_columns[0], pca_columns[1]
    best_columns = pca_columns[best_indices[0]], pca_columns[best_indices[1]]
    points: list[SimilarityPcaPoint] = []
    for row, label in zip(rows, labels):
        selected_value = selected.get(str(row.get("id")))
        probability = _float(row[probability_column])
        similarity = _float(row[similarity_column])
        points.append(
            SimilarityPcaPoint(
                int(_float(row.get("id"), -1.0)),
                label,
                probability,
                similarity,
                _float(row[default_columns[0]]),
                _float(row[default_columns[1]]),
                _float(row[best_columns[0]]),
                _float(row[best_columns[1]]),
                selected_value[0] if selected_value is not None else None,
                selected_value[1] if selected_value is not None else None,
            )
        )
    if {str(point.candidate_id) for point in points if point.selected_ordinal} != set(selected):
        raise ReportingContractError("selected matches are absent from the feature table")
    return SimilarityPcaEvaluation(
        probability_column,
        similarity_column,
        default_columns,
        best_columns,
        best_score,
        len(pca_columns),
        probability_threshold,
        sum(label == 0 for label in labels),
        sum(label == 1 for label in labels),
        sum(label is None for label in labels),
        tuple(points),
    )


# SOURCE_CELL: NB-LIVE-0012-C0028
# SOURCE_STATEMENT_MAP: pca-column-search-pooled-covariance-separation -> best_pca_pair
def best_pca_pair(
    rows: Sequence[Mapping[str, Any]],
    *,
    prefix: str = "xf_embed_pca_",
    maximum_components: int = 12,
) -> tuple[str, str, float]:
    columns = [
        f"{prefix}{index}"
        for index in range(maximum_components)
        if any(f"{prefix}{index}" in row for row in rows)
    ]
    best = ("", "", -inf)
    for first, second in ((a, b) for index, a in enumerate(columns) for b in columns[index + 1 :]):
        positive = [
            (_float(row.get(first)), _float(row.get(second)))
            for row in rows
            if _binary(row.get("human_label")) == 1
            and isfinite(_float(row.get(first)))
            and isfinite(_float(row.get(second)))
        ]
        negative = [
            (_float(row.get(first)), _float(row.get(second)))
            for row in rows
            if _binary(row.get("human_label")) == 0
            and isfinite(_float(row.get(first)))
            and isfinite(_float(row.get(second)))
        ]
        score = _inverse_covariance_score(positive, negative)
        if score > best[2]:
            best = (first, second, score)
    if not best[0]:
        raise ReportingContractError("no usable labeled PCA pair")
    return best


def _precomputed_mask_records(
    rows: Sequence[Mapping[str, str]],
) -> tuple[MaskRecord, ...]:
    masks: list[MaskRecord] = []
    for row in rows:
        encoded = row.get("segmentation", row.get("mask", ""))
        if not encoded:
            continue
        try:
            decoded = loads(encoded)
        except ValueError as exc:
            raise ReportingContractError("precomputed mask is not valid JSON") from exc
        if not isinstance(decoded, list) or any(not isinstance(item, list) for item in decoded):
            raise ReportingContractError("precomputed mask must be a rectangular JSON matrix")
        segmentation = _normalize_mask(decoded)
        masks.append(
            MaskRecord(
                segmentation,
                int(_float(row.get("area"), sum(bit for line in segmentation for bit in line))),
                _float(row.get("predicted_iou")),
                _float(row.get("stability_score")),
            )
        )
    return tuple(masks)


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: multiscale-nms-score-area-border-crop-orders -> build_precomputed_mask_candidate_batch
def build_precomputed_mask_candidate_batch(
    rows: Sequence[Mapping[str, str]],
    *,
    merge_iou: float = 0.75,
    maximum_masks: int = 150,
    border_margin: int = 2,
    crop_padding: float = 0.10,
    maximum_area_fraction_for_visualization: float = 0.35,
    score_grid_count: int = 24,
    score_grid_columns: int = 6,
    score_figure_inches: tuple[float, float] = (18.0, 10.0),
    iou_grid_count: int = 6,
    iou_grid_columns: int = 3,
    iou_figure_inches: tuple[float, float] = (14.0, 8.0),
    overlay_alpha: float = 0.45,
    overlay_seed: int = 3,
) -> PrecomputedMaskBatch:
    if (
        not 0.0 <= merge_iou <= 1.0
        or maximum_masks <= 0
        or border_margin < 0
        or crop_padding < 0.0
        or not 0.0 < maximum_area_fraction_for_visualization <= 1.0
        or score_grid_count <= 0
        or score_grid_columns <= 0
        or iou_grid_count <= 0
        or iou_grid_columns <= 0
        or any(value <= 0.0 for value in (*score_figure_inches, *iou_figure_inches))
        or not 0.0 < overlay_alpha <= 1.0
        or overlay_seed < 0
    ):
        raise ReportingContractError("precomputed mask batch settings are invalid")
    records = _precomputed_mask_records(rows)
    if len(records) != len(rows) or not records:
        raise ReportingContractError("precomputed mask rows must all contain masks")
    height = len(records[0].segmentation)
    width = len(records[0].segmentation[0])
    if any(
        len(record.segmentation) != height
        or len(record.segmentation[0]) != width
        for record in records
    ):
        raise ReportingContractError("precomputed masks must share one tile shape")

    ordered_rows = sorted(
        zip(rows, records),
        key=lambda item: item[1].predicted_iou,
        reverse=True,
    )
    kept: list[tuple[Mapping[str, str], MaskRecord]] = []
    for row, record in ordered_rows:
        measured_area = sum(bit for line in record.segmentation for bit in line)
        if measured_area != record.area:
            raise ReportingContractError("precomputed mask area does not match segmentation")
        if any(
            mask_iou(record.segmentation, existing.segmentation) >= merge_iou
            for _existing_row, existing in kept
        ):
            continue
        kept.append((row, record))
        if len(kept) == maximum_masks:
            break

    candidates: list[PrecomputedMaskCandidate] = []
    for row, record in kept:
        bbox = bbox_from_mask(record.segmentation)
        if bbox is None:
            raise ReportingContractError("precomputed mask has no foreground pixels")
        x, y, box_width, box_height = bbox
        pad = int(round(crop_padding * max(box_width, box_height)))
        bbox_xyxy = (
            max(0, x - pad),
            max(0, y - pad),
            min(width - 1, x + box_width - 1 + pad),
            min(height - 1, y + box_height - 1 + pad),
        )
        try:
            candidate_id = int(str(row["id"]))
            source_scale = float(str(row["source_scale"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ReportingContractError(
                "precomputed mask identity and source scale must be numeric"
            ) from exc
        candidates.append(
            PrecomputedMaskCandidate(
                candidate_id,
                record.segmentation,
                record.area,
                record.area / float(width * height),
                record.predicted_iou,
                record.stability_score,
                record.predicted_iou * record.stability_score,
                source_scale,
                bbox_xyxy,
                touches_border(record.segmentation, margin=border_margin),
            )
        )

    by_score = tuple(
        sorted(candidates, key=lambda item: item.score, reverse=True)
    )
    by_iou = tuple(
        sorted(candidates, key=lambda item: item.predicted_iou, reverse=True)
    )
    visualization = tuple(
        item
        for item in by_iou
        if item.area_fraction <= maximum_area_fraction_for_visualization
    )
    return PrecomputedMaskBatch(
        width,
        height,
        by_iou,
        visualization,
        by_score[:score_grid_count],
        by_iou[:iou_grid_count],
        overlay_alpha,
        overlay_seed,
        score_grid_columns,
        score_figure_inches,
        iou_grid_columns,
        iou_figure_inches,
    )


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: precomputed-mask-batch-to-publication-input -> _publication_mask_inputs
def _publication_mask_inputs(
    batch: PrecomputedMaskBatch,
) -> tuple[PrecomputedBooleanMask, ...]:
    if not isinstance(batch, PrecomputedMaskBatch) or not batch.candidates:
        raise ReportingContractError("mask publication requires a typed mask batch")
    return tuple(
        PrecomputedBooleanMask(
            candidate.candidate_id,
            candidate.segmentation,
            candidate.predicted_iou,
            candidate.stability_score,
            candidate.source_scale,
        )
        for candidate in batch.candidates
    )


# SOURCE_CELL: NB-LIVE-0012-C0002
# SOURCE_CELL: NB-LIVE-0012-C0003
# SOURCE_CELL: NB-LIVE-0012-C0007
# SOURCE_CELL: NB-LIVE-0012-C0008
# SOURCE_CELL: NB-LIVE-0012-C0009
# SOURCE_CELL: NB-LIVE-0012-C0010
# SOURCE_CELL: NB-LIVE-0012-C0011
# SOURCE_CELL: NB-LIVE-0012-C0012
# SOURCE_CELL: NB-LIVE-0012-C0014
# SOURCE_CELL: NB-LIVE-0012-C0015
# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_CELL: NB-LIVE-0012-C0018
# SOURCE_CELL: NB-LIVE-0012-C0019
# SOURCE_CELL: NB-LIVE-0012-C0020
# SOURCE_CELL: NB-LIVE-0012-C0021
# SOURCE_CELL: NB-LIVE-0012-C0022
# SOURCE_CELL: NB-LIVE-0012-C0023
# SOURCE_CELL: NB-LIVE-0012-C0024
# SOURCE_CELL: NB-LIVE-0012-C0025
# SOURCE_CELL: NB-LIVE-0012-C0026
# SOURCE_CELL: NB-LIVE-0012-C0027
# SOURCE_CELL: NB-LIVE-0012-C0028
# SOURCE_STATEMENT_MAP: precomputed-table-continuation -> analyze_precomputed_report
def analyze_precomputed_report(
    request: PrecomputedReportAnalysisRequest,
) -> PrecomputedReportAnalysisResult:
    """Execute the ordered report program over sealed table intermediates."""

    features = [dict(row) for row in request.feature_rows]
    detections = [dict(row) for row in request.detection_rows]
    combined = [*features, *detections]
    reviewed: list[dict[str, Any]] = []
    labeled_rows: list[dict[str, Any]] = []
    reviewed_testset: ReviewedTestsetBatch | None = None
    label_score_rows: list[dict[str, Any]] = []
    labels: list[int] = []
    scores: list[float] = []
    evaluated: list[dict[str, Any]] = []
    active_learning: tuple[ActiveLearningCurvePoint, ...] = ()
    active_learning_focus: ActiveLearningFocusEvaluation | None = None
    active_learning_simulation: ActiveLearningSimulationEvaluation | None = None
    hybrid_evaluations: tuple[HybridEvaluation, ...] = ()
    merged_masks: tuple[MaskRecord, ...] = ()
    selected_masks: tuple[MaskRecord, ...] = ()
    selected_detections: list[dict[str, Any]] = []
    matches: tuple[tuple[Any, float], ...] = ()
    similarity: list[dict[str, Any]] = []
    similarity_pca_evaluation: SimilarityPcaEvaluation | None = None
    precision_recall_area: float | None = None
    precision_recall_band: ModelCurveBand | None = None
    roc_band: ModelCurveBand | None = None
    model_curve_evaluation: PrecomputedModelCurveEvaluation | None = None
    active_learning_score_evaluation: (
        PrecomputedActiveLearningFocusEvaluation | None
    ) = None
    source_confusion_evaluations: tuple[
        PrecomputedModelConfusionEvaluation, ...
    ] = ()
    model_confusion_evaluations: tuple[HybridEvaluation, ...] = ()
    mask_candidate_batch: PrecomputedMaskBatch | None = None
    c0016_mask_publication: MaskPublicationResult | None = None
    c0017_mask_publication: MaskPublicationResult | None = None
    insect_mask_selection: InsectMaskSelectionEvaluation | None = None
    best_columns: tuple[str, str] | None = None
    panel_settings = CandidatePanelSettings()
    candidate_panels: tuple[CandidatePanelEvaluation, ...] = ()
    executed: list[str] = []
    operation_results: list[ReportOperationResult] = []

    for operation in build_report_program(request):
        parameters = dict(operation.parameters)
        operation_value: int | float | str | None = None
        if operation.operation == "build-reviewed-testset":
            reviewed_testset = build_precomputed_reviewed_testset(
                features,
                mode=str(parameters["mode"]),
                tile_size=int(parameters["tile_size"]),
                border_pixels=int(parameters["border_pixels"]),
                tiny_minimum_dimension=int(parameters["tiny_minimum_dimension"]),
                tiny_maximum_area=int(parameters["tiny_maximum_area"]),
            )
            reviewed = [dict(row) for row in reviewed_testset.pool_rows]
            labeled_rows = [dict(row) for row in reviewed_testset.labeled_rows]
            operation_value = len(reviewed)
        elif operation.operation == "evaluate-precomputed-model-scores":
            label_score_rows = [
                row
                for row in combined
                if _binary(row.get("human_label", row.get("label"))) is not None
                and isfinite(
                    _float(row.get("prob", row.get("xgb_prob", row.get("score"))))
                )
            ]
            labels = [
                int(_binary(row.get("human_label", row.get("label"))) or 0)
                for row in label_score_rows
            ]
            scores = [
                _float(row.get("prob", row.get("xgb_prob", row.get("score"))), 0.0)
                for row in label_score_rows
            ]
            model_curve_evaluation = evaluate_precomputed_model_curves(
                features,
                request.model_score_columns,
                fixed_threshold=float(parameters["fixed_threshold"]),
                grid_steps=int(parameters["curve_grid_steps"]),
                maximum_models=int(parameters["population_max_models"]),
                top_k_to_plot=int(parameters["top_k_to_plot"]),
                figure_inches=tuple(
                    float(value) for value in parameters["figure_inches"]
                ),
                save_dpi=int(parameters["save_dpi"]),
                paper_style=bool(parameters["paper_style"]),
                export_svg=bool(parameters["export_svg"]),
                plot_bars=bool(parameters["plot_bars"]),
            )
            evaluated = [
                {
                    "model_rel": item.model_rel,
                    "round": item.round,
                    "pr_auc": item.candidate_metrics.average_precision,
                    "roc_auc": item.candidate_metrics.roc_auc,
                    "thr_best": item.best_threshold_metrics.threshold,
                    "f1_best": item.best_threshold_metrics.f1,
                    "f1_fixed": item.candidate_metrics.f1,
                }
                for item in model_curve_evaluation.model_metrics
            ]
            precision_recall_band = model_curve_evaluation.precision_recall_band
            roc_band = model_curve_evaluation.roc_band
            operation_value = len(evaluated)
        elif operation.operation == "select-active-learning-focus":
            active_learning_score_evaluation = evaluate_precomputed_active_learning_focus(
                features,
                PrecomputedActiveLearningScoreColumns(
                    request.hybrid_score_columns.model_rel,
                    request.hybrid_score_columns.xgb_probability,
                    yolo_confidence=request.hybrid_score_columns.yolo_confidence,
                    yolo_iou=request.hybrid_score_columns.yolo_iou,
                ),
                threshold=float(parameters["decision_threshold"]),
                margin=float(parameters["margin"]),
                disagreement_delta=float(parameters["disagreement_delta"]),
                disagreement_iou_minimum=float(
                    parameters["disagreement_iou_minimum"]
                ),
                histogram_bins=int(parameters["histogram_bins"]),
                figure_inches=tuple(
                    float(value) for value in parameters["figure_inches"]
                ),
                save_dpi=int(parameters["save_dpi"]),
            )
            active_learning_focus = active_learning_score_evaluation.focus
            operation_value = active_learning_focus.selected_rows
        elif operation.operation == "build-learning-curves":
            metric_rows = [
                row for row in combined if isfinite(_float(row.get("pr_auc")))
            ]
            if metric_rows:
                selected_run = str(
                    metric_rows[-1].get("run", metric_rows[-1].get("model_rel", ""))
                ).split("/", 1)[0]
                learning_curve = build_learning_curve(
                    metric_rows,
                    selected_run=selected_run,
                    start_round=int(parameters["start_round"]),
                    smoothing_window=int(parameters["smoothing_window"]),
                )
                operation_value = len(learning_curve)
            else:
                operation_value = 0
        elif operation.operation == "simulate-active-learning":
            if len(label_score_rows) >= 4:
                groups = [
                    str(row.get("root", row.get("img_folder", row.get("image", index))))
                    for index, row in enumerate(label_score_rows)
                ]
                active_learning_simulation = evaluate_precomputed_active_learning(
                    scores,
                    labels,
                    groups,
                    strategies=tuple(str(value) for value in parameters["strategies"]),
                    random_seed=int(parameters["seed"]),
                    initial_size=int(parameters["initial_size"]),
                    batch_size=int(parameters["batch_size"]),
                    maximum_budget=int(parameters["maximum_budget"]),
                    repeats=int(parameters["repeats"]),
                    holdout_fraction=float(parameters["holdout_fraction"]),
                    figure_inches=tuple(
                        float(value) for value in parameters["figure_inches"]
                    ),
                    save_dpi=int(parameters["save_dpi"]),
                )
                active_learning = active_learning_simulation.curve
            operation_value = len(active_learning)
        elif operation.operation == "build-paper-tables":
            if evaluated:
                paper_tables = build_paper_tables(
                    {
                        "N_candidates_total": len(combined),
                        "N_reviewed_labeled": len(labels),
                        "pct_reviewed": len(labels) / max(1, len(combined)),
                        "pos_rate": sum(labels) / max(1, len(labels)),
                        "N_border_or_tiny": sum(
                            bool(row.get("is_border_or_tiny")) for row in combined
                        ),
                        "N_reviewed_border_or_tiny": sum(
                            bool(row.get("is_border_or_tiny")) for row in label_score_rows
                        ),
                        "mode": str(parameters["mode"]),
                    },
                    evaluated,
                    mode=str(parameters["mode"]),
                )
                operation_value = sum(len(rows) for rows in paper_tables.values())
            else:
                operation_value = 0
        elif operation.operation == "compute-confusion-matrices":
            source_confusion_evaluations = (
                evaluate_precomputed_c0011_confusion(
                    features,
                    request.model_score_columns,
                    request.model_metric_rows,
                    target_round=int(parameters["target_round"]),
                    threshold_mode=str(parameters["threshold_mode"]),
                    level=str(parameters["level"]),
                    normalize=bool(parameters["normalize"]),
                ),
                evaluate_precomputed_c0012_confusion(
                    features,
                    request.model_score_columns,
                    request.model_metric_rows,
                    target_round=int(parameters["target_round"]),
                    threshold_mode=str(parameters["threshold_mode"]),
                    level=str(parameters["level"]),
                ),
            )
            model_confusion_evaluations = (
                evaluate_precomputed_model_confusion(
                    features,
                    request.hybrid_score_columns,
                    target_round=int(parameters["target_round"]),
                    threshold=float(parameters["decision_threshold"]),
                    level=str(parameters["level"]),
                ),
            )
            confusion_metrics = model_confusion_evaluations[0].metrics
            operation_value = sum(
                (
                    confusion_metrics.true_negative,
                    confusion_metrics.false_positive,
                    confusion_metrics.false_negative,
                    confusion_metrics.true_positive,
                )
            )
        elif operation.operation == "integrate-precision-recall":
            if labels:
                threshold_steps = int(parameters["threshold_steps"])
                curve = [
                    classification_metrics(labels, scores, step / (threshold_steps - 1))
                    for step in range(threshold_steps)
                ]
                precision_recall_area = integrate_precision_recall(
                    [point.recall for point in curve],
                    [point.precision for point in curve],
                )
            operation_value = precision_recall_area
        elif operation.operation == "compute-hybrid-predictions":
            if request.hybrid_score_columns.target_round != int(
                parameters["target_round"]
            ):
                raise ReportingContractError("hybrid target round changed")
            policy = HybridPolicy(
                xgb_threshold=float(parameters["xgb_threshold"]),
                yolo_threshold=float(parameters["yolo_threshold"]),
                yolo_iou_threshold=float(parameters["yolo_iou_threshold"]),
                detector_threshold=float(parameters["detector_threshold"]),
                missing_policy=str(parameters["missing_policy"]),
                yolo_bias=float(parameters["yolo_bias"]),
                smart_rules_enabled=bool(parameters["smart_rules_enabled"]),
                rule_negative_xgb_maximum=float(
                    parameters["rule_negative_xgb_maximum"]
                ),
                rule_negative_yolo_maximum=float(
                    parameters["rule_negative_yolo_maximum"]
                ),
                rule_positive_yolo_minimum=float(
                    parameters["rule_positive_yolo_minimum"]
                ),
                rule_positive_xgb_minimum=float(
                    parameters["rule_positive_xgb_minimum"]
                ),
                rule_yolo_solo_minimum=float(parameters["rule_yolo_solo_minimum"]),
                xgb_force_enabled=bool(parameters["xgb_force_enabled"]),
                xgb_force_positive_threshold=float(
                    parameters["xgb_force_positive_threshold"]
                ),
                xgb_force_negative_threshold=float(
                    parameters["xgb_force_negative_threshold"]
                ),
                xgb_force_negative_yolo_minimum=float(
                    parameters["xgb_force_negative_yolo_minimum"]
                ),
            )
            hybrid_evaluations = evaluate_hybrid_policy(
                features,
                request.hybrid_score_columns,
                policy,
            )
            operation_value = next(
                item.positive_predictions
                for item in hybrid_evaluations
                if item.level == "candidate-level"
                and item.variant == "hybrid_smart"
            )
        elif operation.operation == "consume-precomputed-mask-batches":
            scales = tuple(float(value) for value in parameters["multiscale"])
            mask_rows = [
                row
                for row in detections
                if _float(row.get("source_scale")) in scales
                and _float(row.get("predicted_iou"))
                >= float(parameters["predicted_iou_threshold"])
                and _float(row.get("stability_score"))
                >= float(parameters["stability_threshold"])
            ]
            mask_candidate_batch = build_precomputed_mask_candidate_batch(
                mask_rows,
                merge_iou=float(parameters["merge_iou"]),
                maximum_masks=int(parameters["maximum_masks"]),
                border_margin=int(parameters["border_margin"]),
                crop_padding=float(parameters["crop_padding"]),
                maximum_area_fraction_for_visualization=float(
                    parameters["maximum_area_fraction_for_visualization"]
                ),
                score_grid_count=int(parameters["score_grid_count"]),
                score_grid_columns=int(parameters["score_grid_columns"]),
                score_figure_inches=tuple(
                    float(value) for value in parameters["score_figure_inches"]
                ),
                iou_grid_count=int(parameters["iou_grid_count"]),
                iou_grid_columns=int(parameters["iou_grid_columns"]),
                iou_figure_inches=tuple(
                    float(value) for value in parameters["iou_figure_inches"]
                ),
                overlay_alpha=float(parameters["overlay_alpha"]),
                overlay_seed=int(parameters["overlay_seed"]),
            )
            publication_masks = _publication_mask_inputs(mask_candidate_batch)
            c0016_mask_publication = publish_c0016_masks(
                publication_masks,
                C0016MaskPublicationSettings(),
            )
            c0017_mask_publication = publish_c0017_masks(
                publication_masks,
                C0017MaskPublicationSettings(),
            )
            insect_mask_selection = evaluate_precomputed_insect_mask(
                mask_candidate_batch,
                minimum_area=int(parameters["insect_minimum_area"]),
                maximum_area_fraction=float(
                    parameters["insect_maximum_area_fraction"]
                ),
                border_margin=int(parameters["insect_border_margin"]),
                visualization_maximum_area_fraction=float(
                    parameters[
                        "publication_visualization_maximum_area_fraction"
                    ]
                ),
                visualization_hide_border=bool(
                    parameters["publication_visualization_hide_border"]
                ),
                crop_padding=float(parameters["publication_crop_padding"]),
                inset_padding=float(parameters["publication_inset_padding"]),
                all_masks_alpha=float(
                    parameters["publication_all_masks_alpha"]
                ),
                chosen_mask_alpha=float(
                    parameters["publication_chosen_mask_alpha"]
                ),
                figure_inches=tuple(
                    float(value)
                    for value in parameters["publication_figure_inches"]
                ),
                save_dpi=int(parameters["publication_save_dpi"]),
            )
            merged_masks = tuple(
                MaskRecord(
                    item.segmentation,
                    item.area,
                    item.predicted_iou,
                    item.stability_score,
                )
                for item in mask_candidate_batch.candidates
            )
            if merged_masks:
                selected_masks = select_masks(
                    merged_masks,
                    tuple(range(min(4, len(merged_masks)))),
                )
                if len(merged_masks) >= 20:
                    select_source_masks(merged_masks)
            operation_value = len(merged_masks)
        elif operation.operation == "compute-candidate-panels":
            panel_settings = CandidatePanelSettings(
                zoom_padding=float(parameters["zoom_padding"]),
                ring_pixels=int(parameters["ring_pixels"]),
                crop_padding=float(parameters["crop_padding"]),
                subsample_count=int(parameters["subsample_count"]),
                predicted_iou_threshold=float(parameters["predicted_iou_threshold"]),
                stability_threshold=float(parameters["stability_threshold"]),
                minimum_l=float(parameters["minimum_l"]),
                minimum_delta_a=float(parameters["minimum_delta_a"]),
                maximum_delta_b=float(parameters["maximum_delta_b"]),
                minimum_elongation=float(parameters["minimum_elongation"]),
                maximum_elongation=float(parameters["maximum_elongation"]),
                minimum_solidity=float(parameters["minimum_solidity"]),
                light_low_threshold=float(parameters["light_low_threshold"]),
                light_high_threshold=float(parameters["light_high_threshold"]),
                maximum_low_clip_fraction=float(
                    parameters["maximum_low_clip_fraction"]
                ),
                maximum_high_clip_fraction=float(
                    parameters["maximum_high_clip_fraction"]
                ),
                maximum_l_standard_deviation=float(
                    parameters["maximum_l_standard_deviation"]
                ),
                maximum_background_delta_l=float(
                    parameters["maximum_background_delta_l"]
                ),
                pca_print_components=int(parameters["pca_print_components"]),
                pca_scatter_max_masks=int(parameters["pca_scatter_max_masks"]),
                pca_scatter_alpha=float(parameters["pca_scatter_alpha"]),
            )
            if int(parameters["selected_candidate_count"]) != 4:
                raise ReportingContractError("candidate selection count contract changed")
            candidate_panels = evaluate_precomputed_candidate_panels(
                request.candidate_panels,
                request.pca_scatter,
                panel_settings,
                mask_candidates=(
                    mask_candidate_batch.candidates
                    if mask_candidate_batch is not None
                    else ()
                ),
            )
            detection_areas = {
                int(_float(row.get("id"), -1.0)): int(_float(row.get("area"), -1.0))
                for row in detections
            }
            if any(
                detection_areas.get(item.panel.candidate_id) != item.panel.mask_area
                for item in candidate_panels
            ):
                raise ReportingContractError(
                    "candidate panel areas do not match sealed detections"
                )
            operation_value = sum(panel.panel.mask_area for panel in candidate_panels)
        elif operation.operation == "select-detection-rows":
            selected_detections = select_detection_rows(detections)
            operation_value = len(selected_detections)
        elif operation.operation == "match-masks-to-detections":
            if selected_masks:
                width = len(selected_masks[0].segmentation[0])
                height = len(selected_masks[0].segmentation)
                matches = match_selected_masks_to_detections(
                    selected_masks,
                    detections,
                    width=width,
                    height=height,
                )
            operation_value = len(matches)
        elif operation.operation == "build-similarity-points":
            similarity_pca_evaluation = evaluate_similarity_pca(
                features,
                matches,
                similarity_column=str(parameters["similarity_column"]),
                probability_candidates=tuple(
                    str(value) for value in parameters["probability_candidates"]
                ),
                pca_prefix=str(parameters["pca_prefix"]),
                maximum_components=int(parameters["maximum_components"]),
                separation_epsilon=float(parameters["separation_epsilon"]),
                probability_threshold=float(parameters["probability_threshold"]),
                selected_count=int(parameters["selected_count"]),
            )
            similarity = [
                {
                    "id": point.candidate_id,
                    "similarity": point.similarity,
                    "probability": point.probability,
                    "human_label": point.human_label,
                    "selected": point.selected_ordinal is not None,
                }
                for point in similarity_pca_evaluation.points
            ]
            operation_value = len(similarity)
        elif operation.operation == "select-best-pca-pair":
            if similarity_pca_evaluation is None:
                raise ReportingContractError(
                    "PCA selection ran before similarity/PCA evaluation"
                )
            if similarity_pca_evaluation.searched_components > int(
                parameters["maximum_components"]
            ):
                raise ReportingContractError("PCA component search exceeded its contract")
            best_columns = similarity_pca_evaluation.best_columns
            operation_value = similarity_pca_evaluation.best_score
        else:
            raise ReportingContractError(
                f"unsupported report program operation: {operation.operation}"
            )
        _explicit_noop()
        executed.append(operation.operation)
        operation_results.append(
            ReportOperationResult(
                operation.operation,
                operation.output_role,
                operation_value,
            )
        )

    if reviewed_testset is None:
        raise ReportingContractError("report program did not build the reviewed testset")
    if mask_candidate_batch is None:
        raise ReportingContractError("report program did not consume the mask batch")
    if insect_mask_selection is None:
        raise ReportingContractError("report program did not evaluate the insect mask")
    if c0016_mask_publication is None or c0017_mask_publication is None:
        raise ReportingContractError("report program did not publish the mask batch")
    if similarity_pca_evaluation is None:
        raise ReportingContractError("report program did not evaluate similarity/PCA")
    if active_learning_focus is None:
        raise ReportingContractError("report program did not evaluate AL focus")
    if active_learning_score_evaluation is None:
        raise ReportingContractError("report program did not evaluate declared AL scores")
    if model_curve_evaluation is None:
        raise ReportingContractError("report program did not evaluate declared model scores")
    if len(source_confusion_evaluations) != 2:
        raise ReportingContractError("report program did not evaluate both confusion contracts")
    if active_learning_simulation is None:
        raise ReportingContractError("report program did not simulate active learning")
    return PrecomputedReportAnalysisResult(
        reviewed_rows=len(reviewed),
        labeled_rows=len(labeled_rows),
        evaluated_models=len(evaluated),
        active_learning_rows=len(active_learning),
        active_learning_focus=active_learning_focus,
        active_learning_simulation=active_learning_simulation,
        hybrid_positive_rows=(
            next(
                item.positive_predictions
                for item in hybrid_evaluations
                if item.level == "candidate-level"
                and item.variant == "hybrid_smart"
            )
            if hybrid_evaluations
            else 0
        ),
        mask_rows=len(merged_masks),
        selected_mask_rows=len(selected_masks),
        selected_detection_rows=len(selected_detections),
        similarity_rows=len(similarity),
        precision_recall_area=precision_recall_area,
        best_pca_columns=best_columns,
        similarity_pca_evaluation=similarity_pca_evaluation,
        reviewed_testset=reviewed_testset,
        candidate_panel_settings=panel_settings,
        candidate_panels=candidate_panels,
        pca_scatter=request.pca_scatter,
        precision_recall_band=precision_recall_band,
        roc_band=roc_band,
        model_curve_evaluation=model_curve_evaluation,
        active_learning_score_evaluation=active_learning_score_evaluation,
        source_confusion_evaluations=source_confusion_evaluations,
        model_confusion_evaluations=model_confusion_evaluations,
        hybrid_evaluations=hybrid_evaluations,
        mask_candidate_batch=mask_candidate_batch,
        c0016_mask_publication=c0016_mask_publication,
        c0017_mask_publication=c0017_mask_publication,
        insect_mask_selection=insect_mask_selection,
        executed_operations=tuple(executed),
        operation_results=tuple(operation_results),
    )
