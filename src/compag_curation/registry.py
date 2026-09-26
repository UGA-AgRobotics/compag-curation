"""Closed registry for inert plans and real, lazily resolved domain handlers."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DomainEntry:
    command: str
    variant: str
    plan_target: str
    handler_target: str
    input_contract: tuple[str, ...]
    optional_input_contract: tuple[str, ...]
    output_contract: tuple[str, ...]
    purpose: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_ENTRY_ROWS = (
    ("propose", "sam2-proposals", "compag_curation.domain.proposals:plan_proposals", "compag_curation.domain.proposals:generate_sam2_proposals", ("images", "sam2_package_root", "sam2_checkpoint", "sam2_config"), (), ("output",), "Generate SAM2 proposals from explicit local assets."),
    ("propose", "yolo-dataset-preparation", "compag_curation.domain.proposals:plan_proposals", "compag_curation.domain.proposals:prepare_yolo_dataset", ("coco_annotations", "tiles", "split_manifest"), (), ("yolo_dataset_output",), "Prepare a YOLO dataset; this is distinct from proposal generation."),
    ("extract-features", "coco-features", "compag_curation.domain.features:plan_feature_extraction", "compag_curation.domain.features:run_feature_extraction_workflow", ("original_coco", "coco_annotations", "detections", "review_labels", "tiles", "sam2_checkpoint", "model_root", "previous_split_manifest"), (), ("merged_original_coco", "merged_tiled_coco", "round_new_ids", "split_manifest", "prototype", "pca", "used_ids", "positive_embeddings", "all_embeddings", "pack", "output"), "Run the reviewed round merge, split, fold-safe pack, and feature extraction pipeline."),
    ("review", "al-review-ui", "compag_curation.domain.review:plan_review", "compag_curation.domain.review:run_al_review", ("image_root", "detections", "images"), (), ("review_output",), "Advance and review active-learning candidates without source mutation."),
    ("review", "full-image-review-ui", "compag_curation.domain.review:plan_review", "compag_curation.domain.review:run_full_image_review", ("image_root", "detections", "images"), (), ("review_output",), "Advance and review detections in full-image context without source mutation."),
    ("train", "coco-xgb", "compag_curation.domain.training:plan_training", "compag_curation.domain.training:run_coco_xgb_training", ("feature_table", "split_manifest", "previous_tile_predictions", "previous_threshold"), ("previous_best_parameters",), ("model_output",), "Run the COCO-derived XGBoost implementation."),
    ("train", "xgb", "compag_curation.domain.training:plan_training", "compag_curation.domain.training:train_xgb_classic", ("feature_table", "split_manifest"), (), ("model_output",), "Run the historical XGBoost implementation."),
    ("train", "xgb-main", "compag_curation.domain.training:plan_training", "compag_curation.domain.training:train_xgb_main", ("feature_table", "split_manifest", "hybrid_previous_tile_predictions", "hybrid_previous_threshold", "augmented_previous_tile_predictions", "augmented_previous_threshold"), (), ("model_output",), "Run the current main XGBoost implementation."),
    ("train", "yolo", "compag_curation.domain.training:plan_training", "compag_curation.domain.training:train_yolo_local", ("data_yaml", "dataset_root", "local_initial_weights"), (), ("model_output",), "Run optional YOLO training using an exact local dataset and explicit local weights only."),
    ("infer", "sam2-xgb", "compag_curation.domain.inference:plan_inference", "compag_curation.domain.inference:run_single_image_inference", ("images", "sam2_package_root", "sam2_checkpoint", "sam2_config", "prototype", "padded_prototype", "pca", "xgb_model", "embed_backbone_weights"), (), ("output",), "Run the SAM2 and XGBoost inference implementation."),
    ("infer", "sam2-xgb-yolo", "compag_curation.domain.inference:plan_inference", "compag_curation.domain.inference:run_dual_backend_inference", ("images", "sam2_package_root", "sam2_checkpoint", "sam2_config", "prototype", "padded_prototype", "pca", "xgb_model", "embed_backbone_weights", "yolo_weights"), (), ("output",), "Run the explicit local-YOLO-assisted inference implementation."),
    ("evaluate", "coverage-basic", "compag_curation.domain.evaluation:plan_evaluation", "compag_curation.domain.evaluation:run_coverage_evaluation", ("predictions", "annotations", "evaluator_script", "tile_index"), (), ("output",), "Evaluate point coverage."),
    ("evaluate", "coverage-points-overlay", "compag_curation.domain.evaluation:plan_evaluation", "compag_curation.domain.evaluation:run_coverage_evaluation", ("predictions", "annotations", "images", "evaluator_script", "tile_index"), (), ("output",), "Render point coverage overlays."),
    ("evaluate", "coverage-predictions-overlay", "compag_curation.domain.evaluation:plan_evaluation", "compag_curation.domain.evaluation:run_coverage_evaluation", ("predictions", "annotations", "images", "evaluator_script", "tile_index"), (), ("output",), "Render prediction coverage overlays."),
    ("evaluate", "coverage-reviewability", "compag_curation.domain.evaluation:plan_evaluation", "compag_curation.domain.evaluation:run_coverage_evaluation", ("predictions", "annotations", "images", "review_labels", "evaluator_script", "tile_index"), (), ("output",), "Evaluate reviewability coverage."),
    ("evaluate", "meta-analysis", "compag_curation.domain.evaluation:plan_evaluation", "compag_curation.domain.evaluation:run_meta_analysis", ("review_root",), (), ("output",), "Build review-event and disagreement meta-analysis tables."),
    ("evaluate", "border-tiny", "compag_curation.domain.evaluation:plan_evaluation", "compag_curation.domain.evaluation:run_border_tiny_evaluation", ("predictions", "annotations"), (), ("output",), "Evaluate border and tiny-object strata."),
    ("transfer", "result-tree", "compag_curation.domain.transfer:plan_transfer", "compag_curation.domain.transfer:transfer_result_tree", ("source",), (), ("destination",), "Copy and verify a result tree."),
    ("transfer", "csv-and-text-probe", "compag_curation.domain.transfer:plan_transfer", "compag_curation.domain.transfer:transfer_csv_text_probe", ("source",), (), ("destination", "probe_workspace"), "Copy and verify approved CSV/text artifacts."),
    ("report", "dataset-summary", "compag_curation.domain.reporting:plan_reporting", "compag_curation.domain.reporting:summarize_dataset", ("source_tables",), (), ("output",), "Create a dataset summary."),
    ("report", "plots", "compag_curation.domain.reporting:plan_reporting", "compag_curation.domain.reporting:build_plots_report", ("source_tables",), (), ("output",), "Create documented plots from approved aggregate tables."),
)


def registry_snapshot() -> dict[str, dict[str, DomainEntry]]:
    registry: dict[str, dict[str, DomainEntry]] = {}
    for row in _ENTRY_ROWS:
        entry = DomainEntry(*row)
        variants_for_command = registry.setdefault(entry.command, {})
        if entry.variant in variants_for_command:
            raise RuntimeError("duplicate domain registry entry")
        variants_for_command[entry.variant] = entry
    return registry


def commands() -> tuple[str, ...]:
    return tuple(registry_snapshot())


def variants(command: str) -> tuple[str, ...]:
    registry = registry_snapshot()
    if command not in registry:
        raise ValueError("unknown workflow command")
    return tuple(registry[command])


def resolve(command: str, variant: str) -> DomainEntry:
    try:
        return registry_snapshot()[command][variant]
    except KeyError as exc:
        raise ValueError("unknown command or variant") from exc
