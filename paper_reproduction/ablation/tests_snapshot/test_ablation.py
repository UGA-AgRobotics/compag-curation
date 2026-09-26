from __future__ import annotations

import csv
import gzip
import hashlib
import importlib
import io
import json
import os
import shutil
import sys
import time
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


LAYOUT_ROOT = Path(__file__).resolve().parents[1]
if (LAYOUT_ROOT / "code").is_dir():
    STUDY = LAYOUT_ROOT
    CODE = STUDY / "code"
elif (LAYOUT_ROOT / "code_snapshot").is_dir():
    STUDY = LAYOUT_ROOT.parent
    CODE = LAYOUT_ROOT / "code_snapshot"
else:
    raise RuntimeError(f"Cannot locate reviewer code beside {__file__}")
os.environ.setdefault("ABLATION_STUDY_ROOT", str(STUDY))
sys.path.insert(0, str(CODE))

import ablation_core as legacy  # noqa: E402
import amendment_core as core  # noqa: E402
import amendment_inventory as inventory  # noqa: E402
import build_amended_preflight as builder  # noqa: E402
import evidence_gates as gates  # noqa: E402
import run_study  # noqa: E402


REAL_ACTUAL_BOOSTER_CONFIGURATION = core._actual_booster_configuration
REAL_ACTIVE_RELOADED_BOOSTER_CUDA_PROBE = core._active_reloaded_booster_cuda_probe


@lru_cache(maxsize=1)
def _historical_split_fixture():
    frame = pd.read_csv(core.HISTORICAL_TRAINING_CSV, usecols=["file_name", "label"], low_memory=False)
    groups = frame.file_name.map(core.normalize_card_id)
    train, heldout, info = core.historical_group_split(groups.to_numpy())
    return frame, train, heldout, info


@lru_cache(maxsize=None)
def _valid_test_booster_artifacts(feature_count: int) -> tuple[bytes, dict]:
    rows = np.arange(8 * feature_count, dtype=np.float32).reshape(8, feature_count)
    matrix = core.xgb.DMatrix(rows, label=np.array([0, 0, 0, 0, 1, 1, 1, 1]))
    parameters = dict(
        core.r92_model_parameter_lock()["smoke_classifier_parameters"]
    )
    for key in ("n_estimators", "early_stopping_rounds", "device"):
        parameters.pop(key)
    parameters["device"] = "cpu"
    booster = core.xgb.train(parameters, matrix, num_boost_round=1)
    return (
        bytes(booster.save_raw(raw_format="ubj")),
        json.loads(booster.save_config()),
    )


def _valid_test_booster_bytes(feature_count: int) -> bytes:
    return _valid_test_booster_artifacts(feature_count)[0]


def _gate_complete_training_rows(feature_names: list[str]) -> pd.DataFrame:
    rows = 4
    values = {
        feature: np.full(rows, 0.1, dtype=float)
        for feature in feature_names
    }
    values.update({
        "file_name": [
            "train.jpg", "validation_neg.jpg", "validation_pos.jpg",
            "excluded.jpg",
        ],
        "ann_id": [1, 2, 3, 4],
        "scale": [1.0, 1.0, 1.0, 1.0],
        "label": [0, 0, 1, 0],
        "components_count": [1, 1, 1, 1],
        "mean_L": [100.0, 110.0, 120.0, 130.0],
        "std_L": [10.0, 11.0, 12.0, 13.0],
        "delta_a": [3.0, 4.0, 5.0, 6.0],
        "delta_b": [4.0, 5.0, 6.0, 7.0],
        "circularity": [0.7, 0.6, 0.8, 0.5],
        "solidity": [0.8, 0.7, 0.9, 0.6],
        "eccentricity": [0.2, 0.3, 0.1, 0.4],
        "extent": [0.5, 0.6, 0.4, 0.45],
        "grad_p90": [50.0, 60.0, 70.0, 80.0],
        "touching_border": [0, 0, 0, 1],
        "embed_sim": [0.6, 0.7, 0.8, 0.9],
    })
    frame = pd.DataFrame(values)
    reconstructed = core.reconstruct_gates(frame, prototype_available=True)
    for column in reconstructed:
        frame[column] = reconstructed[column]
    return frame


def _blocked_no_pca_evidence(
    source: Path,
    audit_source: Path,
    raw_cache: Path,
    audit_rows: int,
) -> tuple[dict, pd.DataFrame]:
    scale_evidence = inventory.extract_notebook_scale_evidence(core.NOTEBOOK)
    raw_scales = scale_evidence["raw_embedding_scales"]["values"]
    sidecars = inventory.discover_stable_key_sidecars(
        raw_cache, search_roots=[raw_cache.parent],
    )
    mapping = inventory.raw_cache_table_diagnostics(
        raw_cache,
        source,
        raw_scale_sequence=raw_scales,
        expected_dimension_count=2048,
        sidecar_diagnostic=sidecars,
        population="training",
        embedding_semantics_manifest_path=None,
        embedding_semantics_manifest_sha256=None,
    )
    raw_columns = mapping["ordered_raw_feature_columns"]
    raw_column_hash = mapping["ordered_raw_feature_columns_sha256"]
    audit_mapping = inventory.audit_detection_diagnostics(audit_source, [])
    audit_mapping.update({
        "polygon_treatment": {
            "classification": "VERIFIED",
            "eligible_as_exact_mask": False,
            "reason": (
                "The inference source saves a rounded perspective-transformed "
                "approxPolyDP external contour, not the full pre-transform mask "
                "used for embedding crops."
            ),
            "source_path": str(core.PIPELINE_SOURCE),
            "source_sha256": core.sha256_file(core.PIPELINE_SOURCE),
            "source_symbols": ["cv2.approxPolyDP", "crop_masked_patch"],
        },
        "aligned_raw_layer_reference": {
            "classification": "BLOCKED",
            "status": "BLOCKED_NO_MATERIALIZED_AUDIT_RAW_EMBEDDING_LAYER",
            "reason": (
                "Crop/mask availability is diagnostic input evidence, not a "
                "persisted 2048D audit embedding layer with stable-key alignment."
            ),
        },
    })
    diagnostic = {
        "classification": "BLOCKED",
        "status": "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        "required_raw_dimensions": 2048,
        "substitute_minus_pca_components_used": False,
        "notebook_scale_source": scale_evidence,
        "historical_training_mapping": mapping,
        "current_training_mapping": mapping,
        "audit_mapping": audit_mapping,
        "training_crop_mask_availability": {
            "classification": "VERIFIED",
            "tiled_coco_sha256": core.sha256_file(source),
            "feature_base_candidate_count": 4,
            "referenced_tile_images_existing": 0,
        },
        "raw_layer_contract": None,
        "encoder": {
            "available_weights_artifact": {
                "classification": "VERIFIED",
                "path": str(raw_cache),
                "sha256": core.sha256_file(raw_cache),
            },
            "historical_encoder_identity": {
                "classification": "UNRESOLVED",
                "status": "UNRESOLVED_NO_PERSISTED_CACHE_TO_WEIGHT_LINKAGE",
                "persisted_linkage_found": False,
                "reason": (
                    "The source permits an environment-selected ResNet backbone "
                    "and a weights fallback; no raw-cache sidecar records the exact "
                    "selected backbone/weight artifact."
                ),
            },
            "source_supported_default": (
                "torchvision ResNet50 IMAGENET1K_V2 with DEFAULT fallback, "
                "fc=Identity, L2-normalized"
            ),
            "preprocessing": (
                "RGB, Resize(224), CenterCrop(224), ImageNet normalization, "
                "L2 normalization"
            ),
            "source_path": str(core.GATE_SOURCE),
            "source_sha256": core.sha256_file(core.GATE_SOURCE),
            "source_symbols": [
                "JASSID_EMBED_BACKBONE", "build_embed_model", "embed_patch",
            ],
        },
        "exact_mapping_prerequisites": {
            "historical_training": False,
            "external_audit": False,
            "current_training_comparison_only": False,
            "same_encoder_preprocessing_semantics": False,
            "all_training_validation_audit_layers_materialized": False,
        },
        "exact_ordered_raw_feature_columns": None,
        "exact_ordered_raw_feature_columns_sha256": None,
        "proposed_unmapped_raw_feature_columns": raw_columns,
        "proposed_unmapped_raw_feature_columns_sha256": raw_column_hash,
        "ordered_raw_feature_column_status": "BLOCKED_UNMAPPED_DESIGN_ONLY",
    }
    rows = []
    for population, candidate_rows, candidate_mapping in (
        ("historical_training", 4, mapping),
        ("current_training", 4, mapping),
    ):
        actual = candidate_mapping["actual_stable_key_join"]
        rows.append({
            "population": population,
            "candidate_rows": candidate_rows,
            "base_candidates": candidate_mapping["candidate_table"]["base_candidate_count"],
            "raw_cache_rows": candidate_mapping["raw_cache"]["row_count"],
            "implicit_raw_candidates": candidate_mapping["raw_cache"]["implicit_candidate_count"],
            "candidate_deficit": candidate_mapping["candidate_deficit"],
            "stable_key_duplicate_rows": candidate_mapping["candidate_table"]["stable_key_duplicate_rows"],
            "verified_exact_raw_embedding_rows": int(actual["mapped_candidate_rows"]),
            "exact_join_rate": actual["exact_join_rate"],
            "join_rate_classification": "BLOCKED",
            "join_rate_reason": "No parseable stable-key sidecar permits an actual numeric join",
            "status": candidate_mapping["status"],
            "classification": candidate_mapping["evidence_classification"],
        })
    rows.append({
        "population": "external_audit",
        "candidate_rows": audit_rows,
        "base_candidates": audit_rows,
        "raw_cache_rows": 0,
        "implicit_raw_candidates": None,
        "candidate_deficit": audit_rows,
        "stable_key_duplicate_rows": 0,
        "verified_exact_raw_embedding_rows": 0,
        "exact_join_rate": None,
        "join_rate_classification": "BLOCKED",
        "join_rate_reason": (
            "No materialized audit raw-embedding layer and stable-key alignment; "
            "crop/mask diagnostics are not a raw-layer join"
        ),
        "status": audit_mapping["raw_embedding_input_status"],
        "classification": audit_mapping["evidence_classification"],
    })
    join = pd.DataFrame(rows)
    diagnostic["diagnostic_completeness"] = builder._validate_no_pca_diagnostic(
        diagnostic, join,
    )
    assert diagnostic["diagnostic_completeness"]["status"] == "PASS"
    return diagnostic, join


@pytest.fixture
def runtime_case(request):
    path = STUDY / ".runtime" / "pytest" / request.node.name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _mark_publication_owned(
    path: Path,
    ownership: dict[Path, core.PublicationOwnershipToken],
) -> core.PublicationOwnershipToken:
    token = core._capture_regular_publication_token(
        path, expected_sha256=core.sha256_file(path),
    )
    core._register_owned_publication(path, token, ownership)
    return token


def _valid_preflight_semantic_fixture(
    runtime_case: Path,
    monkeypatch,
    *,
    run_name: str = "valid_preflight",
):
    """Create compact but machine-recomputable blocked Preflight evidence."""

    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    project_root = runtime_case / "project"
    audit_detection_root = project_root / "Shared/maskout_tile/test_set"
    audit_detection_root.mkdir(parents=True)
    training_tile_root = runtime_case / "source/tiles"
    training_tile_root.mkdir(parents=True)
    monkeypatch.setattr(core, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(core, "TRAINING_TILE_ROOT", training_tile_root)
    results = runtime_case / "results"
    results.mkdir(exist_ok=True)
    run = results / run_name
    state_name = "BLOCKED_PREFLIGHT_COMPLETE"
    required = core.PACKAGE_REQUIRED_FILES[("AMENDED_PREFLIGHT", state_name)]
    for relative in required:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n")

    variants = core.build_variant_feature_sets()
    (run / "config/variant_feature_sets.json").write_text(json.dumps(variants))
    (run / "config/feature_dependency_graph.json").write_text(
        json.dumps(core.feature_dependency_graph_payload())
    )
    feature_names = core.get_r92_feature_order()
    core.feature_manifest_frame(feature_names).to_csv(
        run / "config/feature_manifest.csv", index=False,
    )

    source = runtime_case / "source" / "historical.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source_rows = _gate_complete_training_rows(feature_names)
    source_rows.to_csv(source, index=False)
    source_training = core.load_training(source)
    prediction_source = runtime_case / "source" / "tile_preds.csv"
    prediction_rows = source_training.loc[
        [1, 2], ["file_name", "ann_id", "scale", "label"]
    ].copy()
    prediction_rows["proba"] = ["0.1", "0.9"]
    prediction_rows.to_csv(prediction_source, index=False)
    audit_source = runtime_case / "source" / "audit.csv"
    audit_row = {
        "img_folder": "IMG_0001", "image": "audit.jpg", "id": 1,
        "human_label": 0, "review_weight": 1.0,
    }
    for column in feature_names:
        audit_row[column] = source_rows.loc[0, column]
    for column in core.GATE_REQUIRED_INPUTS:
        audit_row[column] = source_rows.loc[0, column]
    pd.DataFrame([audit_row]).to_csv(audit_source, index=False)
    raw_cache = runtime_case / "source" / "raw_embeddings.npy"
    np.save(raw_cache, np.arange(12 * 2048, dtype=np.float32).reshape(12, 2048))
    gate_source = runtime_case / "source" / "gate_core.py"
    gate_source.write_text("# fixture gate source\n")
    tiled_coco = runtime_case / "source" / "CJ_NOCJ_tiles_512.json"
    tiled_coco.write_text("{}\n")
    task_spec = runtime_case / "source" / "PROTOCOL_AMENDMENT_01_TASK_SPEC.md"
    task_spec.write_text("fixture amendment task specification\n")
    (run / "docs/PROTOCOL_AMENDMENT_01_TASK_SPEC.md").write_bytes(
        task_spec.read_bytes()
    )
    monkeypatch.setattr(core, "HISTORICAL_TRAINING_CSV", source)
    monkeypatch.setattr(core, "CURRENT_TRAINING_CSV", source)
    monkeypatch.setattr(core, "AUDIT_CSV", audit_source)
    monkeypatch.setattr(core, "R92_PREDICTIONS", prediction_source)
    monkeypatch.setattr(core, "RAW_EMBEDDINGS", raw_cache)
    monkeypatch.setattr(core, "POST_R92_RAW", raw_cache)
    monkeypatch.setattr(core, "HISTORICAL_TILED_COCO", tiled_coco)
    monkeypatch.setattr(core, "ENCODER_WEIGHTS", raw_cache)
    monkeypatch.setattr(core, "GATE_SOURCE", gate_source)
    monkeypatch.setattr(core, "PIPELINE_SOURCE", gate_source)
    monkeypatch.setattr(core, "TASK_SPEC", task_spec)
    monkeypatch.setattr(core, "EXPECTED_HASHES", {
        str(source): core.sha256_file(source),
        str(audit_source): core.sha256_file(audit_source),
        str(raw_cache): core.sha256_file(raw_cache),
        str(gate_source): core.sha256_file(gate_source),
        str(tiled_coco): core.sha256_file(tiled_coco),
        str(task_spec): core.sha256_file(task_spec),
    })
    monkeypatch.setattr(core, "historical_metric_sidecar_evidence", lambda: {
        "metrics_path": str(source), "metrics_sha256": core.sha256_file(source),
        "threshold_path": str(audit_source),
        "threshold_sha256": core.sha256_file(audit_source),
        "checkpoint_path": str(raw_cache),
        "checkpoint_sha256": core.sha256_file(raw_cache),
        "average_precision": 1.0, "roc_auc": 1.0,
        "confusion_matrix": [[1, 0], [0, 1]], "threshold": 0.5,
        "sidecar_used_trees": 1, "checkpoint_used_trees": 1,
    })

    split = source_training[[
        "stable_candidate_id", "file_name", "ann_id", "scale", "card_id", "label",
    ]].copy()
    split["split"] = [
        "training", "validation", "validation",
        "historical_heldout_ineligible_border",
    ]
    split["eligible"] = [True, True, True, False]
    split["split_order"] = [0, 0, 1, -1]
    split.to_csv(run / "splits/row_split_manifest.csv.gz", index=False)
    split_hash = core.split_assignment_hash(
        split.stable_candidate_id, split.split, split.label,
        split.eligible, split.split_order,
    )
    split_hashes = {
        "candidate_split_manifest_sha256": split_hash,
        "locked_for_scientific_use": False,
        "lock_status": "NOT_LOCKED_DATA_SOURCE_DECISION_UNRESOLVED",
    }
    (run / "splits/split_hashes.json").write_text(json.dumps(split_hashes))
    cards = split.groupby(["card_id", "split"], as_index=False).agg(
        rows=("stable_candidate_id", "size"),
        positive=("label", "sum"),
        eligible=("eligible", "sum"),
    )
    cards["negative"] = cards.rows - cards.positive
    cards.to_csv(run / "splits/card_split_manifest.csv", index=False)
    summary = split.groupby("split", as_index=False).agg(
        rows=("stable_candidate_id", "size"),
        positive=("label", "sum"),
        cards=("card_id", "nunique"),
        eligible=("eligible", "sum"),
    )
    summary["negative"] = summary.rows - summary.positive
    summary.to_csv(run / "splits/split_summary.csv", index=False)
    pd.DataFrame([
        {"comparison": "training_vs_validation", "overlap_count": 0,
         "status": "PASS"},
        {"comparison": "development_vs_external_audit", "overlap_count": 0,
         "status": "PASS"},
    ]).to_csv(run / "splits/overlap_checks.csv", index=False)

    replay_rows = []
    for split_index, probability in ((1, 0.1), (2, 0.9)):
        probability32 = np.float32(probability)
        direct_error = abs(probability - float(probability32))
        replay_rows.append({
            "stable_candidate_id": split.loc[split_index, "stable_candidate_id"],
            "file_name": split.loc[split_index, "file_name"],
            "ann_id": split.loc[split_index, "ann_id"],
            "scale": split.loc[split_index, "scale"],
            "label": split.loc[split_index, "label"],
            "historical_saved_probability_literal": str(probability),
            "historical_saved_probability_decimal": probability,
            "historical_saved_probability_float64_hex": float(probability).hex(),
            "historical_saved_probability_float32": float(probability32),
            "historical_saved_probability_float32_bits_hex": (
                f"0x{probability32.view(np.uint32):08x}"
            ),
            "replayed_probability_float32": float(probability32),
            "replayed_probability_float32_bits_hex": (
                f"0x{probability32.view(np.uint32):08x}"
            ),
            "absolute_difference_direct_decimal": direct_error,
            "absolute_difference_direct_decimal_float64_hex": direct_error.hex(),
            "absolute_difference_source_dtype_float32": 0.0,
        })
    replay = pd.DataFrame(replay_rows)
    replay_path = run / "provenance/historical_prediction_replay.csv.gz"
    replay.to_csv(replay_path, index=False, float_format="%.17g")
    replay_summary, replay_valid = core._historical_replay_reopen_summary(replay_path)
    assert replay_valid
    replay_verification = {
        "status": "PASS",
        "artifact_sha256": core.sha256_file(replay_path),
        "summary": replay_summary,
        "summary_matches_in_memory_calculation": True,
    }
    (run / "provenance/historical_prediction_replay_schema.json").write_text(
        json.dumps({
            "source_prediction_path": str(prediction_source),
            "source_prediction_sha256": core.sha256_file(prediction_source),
            "comparison_sha256": core.sha256_file(replay_path),
            "serialized_reopen_verification": replay_verification,
        })
    )

    source_sha = core.sha256_file(source)
    source_roles = {
        source: "historical/current training feature table",
        prediction_source: "historical replay source",
        audit_source: "canonical external audit table",
        raw_cache: "raw-cache embedding source",
        gate_source: "feature source",
        tiled_coco: "training crop/mask provenance",
        task_spec: "task specification",
    }
    source_frame = pd.DataFrame([
        {"path": str(path), "role": role,
         "size_bytes": path.stat().st_size, "sha256": core.sha256_file(path)}
        for path, role in source_roles.items()
    ])
    source_frame.to_csv(
        run / "provenance/source_input_hashes_pre.tsv", sep="\t", index=False,
    )
    source_frame.to_csv(
        run / "provenance/source_input_hashes_post.tsv", sep="\t", index=False,
    )

    fixture_pca_inventory = inventory.inventory_frame([
        inventory.inventory_record(
            raw_cache,
            logical_name="fixture_pca_inventory_anchor",
            role="timestamped PCA/prototype lineage",
        )
    ])
    fixture_pca_inventory.to_csv(
        run / "provenance/pca_artifact_inventory.tsv", sep="\t", index=False,
    )
    fixture_pca = {
        "classification": "VERIFIED",
        "status": "VERIFIED_SINGLE_COMPATIBLE_TRANSFORM",
        "compatible_single_basis": True,
        "compatibility_conclusion": (
            "A single frozen PCA/prototype basis is proven for every row."
        ),
        "dedicated_artifact_inventory": {
            "classification": "VERIFIED", "status": "PASS",
            "row_count": 1, "lineage_row_count": 1, "prefix_row_count": 0,
            "checks": {"fixture_inventory": True},
        },
    }
    monkeypatch.setattr(
        core, "canonical_pca_artifact_inventory",
        lambda: fixture_pca_inventory.copy(),
    )
    monkeypatch.setattr(
        core, "pca_lineage_evidence", lambda frame: dict(fixture_pca),
    )

    historical = {
        "path": str(source), "sha256": source_sha,
        "rows": len(source_training),
        "cards": int(source_training.card_id.nunique()),
        "negative_rows": int((source_training.label == 0).sum()),
        "positive_rows": int((source_training.label == 1).sum()),
        "feature_count": len(feature_names),
        "stable_candidate_id_unique": True,
        "stable_key_null_rows": 0, "stable_key_duplicate_rows": 0,
        "external_audit_card_overlap": 0,
        "all_features_present": True,
        "ordered_candidate_identity_and_label_match": True,
        "ordered_rows_compared": len(replay),
        "saved_prediction_source_path": str(prediction_source),
        "saved_prediction_source_sha256": core.sha256_file(prediction_source),
        "metrics": {
            "average_precision": 1.0,
            "roc_auc": 1.0,
            "tn": 1, "fp": 0, "fn": 0, "tp": 1,
            "threshold": 0.5,
            "sidecar_average_precision": 1.0,
            "sidecar_roc_auc": 1.0,
            "used_trees": 1,
            "sidecar_used_trees": 1,
            "metric_and_tree_parity": True,
        },
        "direct_decimal_probability_max_absolute_difference": replay_summary[
            "direct_decimal_max_absolute_difference"
        ],
        "direct_decimal_rows_exceeding_1e_12": replay_summary[
            "direct_decimal_mismatch_count_gt_1e_12"
        ],
        "literal_probability_tolerance_status": "FAIL",
        "source_dtype_roundtrip_status": "SUPPLEMENTAL_VERIFIED_BIT_EXACT",
        "source_dtype_float32_probability_max_absolute_difference": (
            replay_summary["source_dtype_float32_max_absolute_difference"]
        ),
        "serialized_replay_artifact": {
            "reopen_verification": replay_verification,
        },
        "pca_prototype_compatibility": fixture_pca,
    }
    (run / "provenance/historical_snapshot_validation.json").write_text(
        json.dumps(historical)
    )
    current = {
        "path": str(source), "sha256": source_sha,
        "rows": len(source_training),
        "cards": int(source_training.card_id.nunique()),
        "negative_rows": int((source_training.label == 0).sum()),
        "positive_rows": int((source_training.label == 1).sum()),
        "feature_count": len(feature_names),
        "stable_candidate_id_unique": True,
        "stable_key_null_rows": 0,
        "stable_key_duplicate_rows": 0,
        "all_features_present": True,
    }
    (run / "provenance/current_snapshot_validation.json").write_text(
        json.dumps(current)
    )
    cleaned_audit = core._clean_canonical_audit_frame(
        audit_source, core.sha256_file(audit_source),
    )
    embed_semantics, harmonized = core.feature_semantics_evidence(
        source_training, cleaned_audit, feature_names,
    )
    harmonized_layer = pd.concat([
        cleaned_audit[["stable_candidate_id", "card_id"]], harmonized,
    ], axis=1)
    harmonized_path = run / "provenance/harmonized_audit_feature_layer.csv.gz"
    harmonized_layer.to_csv(harmonized_path, index=False)
    (run / "provenance/embed_sim_semantics.json").write_text(
        json.dumps(embed_semantics)
    )
    (run / "provenance/harmonized_feature_layer_manifest.json").write_text(
        json.dumps(core.harmonized_layer_manifest(
            fixture_pca,
            harmonized,
            feature_names,
            relative_path="provenance/harmonized_audit_feature_layer.csv.gz",
            layer_sha256=core.sha256_file(harmonized_path),
        ))
    )
    raw_status = "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE"
    raw_diagnostic, raw_join = _blocked_no_pca_evidence(
        source, audit_source, raw_cache, audit_rows=1,
    )
    raw_diagnostic["encoder"].update({
        "source_path": str(gate_source),
        "source_sha256": core.sha256_file(gate_source),
    })
    raw_diagnostic["training_crop_mask_availability"].update({
        "tiled_coco_path": str(tiled_coco),
        "tiled_coco_sha256": core.sha256_file(tiled_coco),
        "feature_base_candidates_joining_coco_annotation": 0,
        "feature_base_candidates_with_segmentation": 0,
        "feature_base_candidates_without_exact_coco_join": 4,
        "referenced_tile_image_count": 4,
        "referenced_tile_images_existing": 0,
        "tile_root": str(training_tile_root.resolve()),
        "tile_content_hash_inventory": (
            "NOT_CAPTURED; availability claim is path-existence only"
        ),
    })
    (run / "provenance/raw_embedding_mapping_diagnostic.json").write_text(
        json.dumps(raw_diagnostic)
    )
    raw_join.to_csv(
        run / "provenance/raw_embedding_join_diagnostic.csv", index=False,
    )
    source_frame[source_frame.role.str.contains(
        r"raw-cache|raw cache|audit detection|embedding|training crop|ResNet50",
        case=False, regex=True,
    )].to_csv(
        run / "provenance/raw_embedding_source_inventory.tsv",
        sep="\t", index=False,
    )
    fixture_weights = {
        "historical_legacy_geometry": {
            "post_augmentation_mean_parity": True,
            "post_augmentation_mean_weight": 0.7384929060935974,
            "expected_r92_metadata_mean_weight": 0.7384929060935974,
            "post_augmentation_rows": 1309608,
        },
        "augmentation_counts": {
            "post_jitter": {"0": 971633, "1": 337975},
            "final_fit_rows": 1457449,
        },
        "historical_safe_smote": {
            "synthetic_positive_rows": 147841,
            "final_rows": 1457449,
        },
        "shuffle_evidence": {
            "training_order_changed": True,
            "heldout_order_changed": True,
        },
    }
    (run / "provenance/review_weight_validation.json").write_text(
        json.dumps(fixture_weights)
    )
    cuda = {"status": "PASS", "use_cuda_compiled": True}
    (run / "provenance/cuda_capability_probe.json").write_text(json.dumps(cuda))
    gate_formula = builder._gate_formula_manifest()
    (run / "provenance/gate_formula_manifest.json").write_text(
        json.dumps(gate_formula)
    )
    pd.DataFrame(
        core.gate_validation_rows(source_training, "historical_363563")
        + core.gate_validation_rows(source_training, "current_511082")
    ).to_csv(
        run / "provenance/gate_reconstruction_validation.csv", index=False,
    )
    canonical_audit = core._clean_canonical_audit_frame(
        audit_source, core.sha256_file(audit_source),
    )
    pd.DataFrame(core.required_numeric_coverage(
        canonical_audit, core.GATE_REQUIRED_INPUTS,
    )).to_csv(
        run / "provenance/audit_gate_upstream_coverage.csv", index=False,
    )
    r92 = {
        "status": "PASS", "max_probability_difference": 0.0,
        "ordered_population_match": True, "population_metadata_match": True,
        "selected_score_source_hash_match": True,
        "expected_metrics_match": True,
    }
    (run / "metrics/r92_control.json").write_text(json.dumps(r92))
    monkeypatch.setattr(
        core, "score_r92_control",
        lambda audit: (dict(r92), np.zeros(len(audit), dtype=float)),
    )
    monkeypatch.setattr(
        core, "recompute_review_weight_evidence",
        lambda stored, historical_frame, split_frame, ordered_features: (
            stored == fixture_weights,
            stored == fixture_weights,
            stored == fixture_weights,
        ),
    )
    prior = {
        "prior": {"manifest_status": "PASS", "zip": {"status": "PASS"}},
        "selected_v3": {"status": "PASS"},
    }
    for name in (
        "prior_immutable_verification_pre.json",
        "prior_immutable_verification_post.json",
    ):
        (run / "provenance" / name).write_text(json.dumps(prior))
    monkeypatch.setattr(core, "xgboost_cuda_capability_probe", lambda: dict(cuda))
    monkeypatch.setattr(
        core, "verify_prior_run_inventory", lambda: dict(prior["prior"]),
    )
    monkeypatch.setattr(
        core, "verify_zip", lambda *args, **kwargs: dict(prior["selected_v3"]),
    )
    command_payload = {
        "command": ["fixture-check"], "cwd": str(runtime_case),
        "returncode": 0, "stdout": "", "stderr": "", "wall_seconds": 0.0,
    }
    command_log = (
        "COMMAND=fixture-check\n"
        f"CWD={runtime_case}\nEXIT_CODE=0\nWALL_SECONDS=0.000000000\n"
        "--- STDOUT ---\n--- STDERR ---\n"
    )
    for relative in (
        "logs/syntax_compile.log", "logs/pytest.log",
        "logs/reviewer_snapshot_syntax_compile.log",
        "logs/reviewer_snapshot_pytest.log",
    ):
        (run / relative).write_text(command_log)

    source_code = Path(core.__file__).resolve().parent
    source_tests = Path(__file__).resolve().parent
    code_tree = {}
    for source_root, target_root, patterns in (
        (source_code, run / "code", ("*.py", "*.sh")),
        (source_tests, run / "tests", ("*.py",)),
    ):
        for pattern in patterns:
            for source_path in source_root.glob(pattern):
                target = target_root / source_path.name
                shutil.copy2(source_path, target)
                relative = target.relative_to(run).as_posix()
                code_tree[relative] = core.sha256_file(target)
                snapshot_root = (
                    run / "provenance"
                    / ("code_snapshot" if relative.startswith("code/") else "tests_snapshot")
                )
                shutil.copy2(source_path, snapshot_root / source_path.name)

    identity_seed = {
        "protocol": "AMENDMENT_01",
        "run_kind": "AMENDED_PREFLIGHT",
        "command_line": ["fixture", run_name],
        "working_directory": str(runtime_case),
        "task_spec_sha256": core.sha256_file(task_spec),
        "source_hashes": dict(zip(source_frame.path, source_frame.sha256)),
        "code_tree": code_tree,
    }
    suffix = core.sha256_bytes(core.canonical_json(identity_seed))[:8]
    final_run = run.parent / (
        f"run_20260812_000000_{suffix}_amended_preflight"
    )
    run.rename(final_run)
    run = final_run

    ledger = {
        "classification": "VERIFIED", "status": "PASS",
        "checks": {"no_execution": True},
        "official_smoke_variants_completed": 0,
        "exploratory_smoke_variants_completed": 0,
        "smoke_report_status": "ABSENT",
        "full_scientific_run_executed": False,
        "full_variants_completed": 0,
        "stability_executed": False,
        "multiseed_executed": False,
    }
    (run / "provenance/execution_ledger.json").write_text(json.dumps(ledger))

    gate_inputs = {
        specification.key: {
            "classification": "VERIFIED", "passed": True,
            "evidence_refs": list(core.PREFLIGHT_GATE_EVIDENCE_REFS[
                specification.key
            ]),
            "detail": "machine-derived fixture evidence",
        }
        for specification in gates.DEFAULT_GATE_SPECS
    }
    gate_inputs["historical_prediction_replay"]["passed"] = False
    for key in ("cuda_smoke_all_runnable", "model_save_reload"):
        gate_inputs[key] = {
            "classification": "BLOCKED", "passed": None,
            "evidence_refs": ["provenance/PREFLIGHT_REPORT.json"],
            "detail": "not run before globally authorized Smoke",
        }
    gate_inputs["output_manifest_zip_verification"] = {
        "classification": "BLOCKED", "passed": None,
        "evidence_refs": ["provenance/package_staging_verification.json"],
        "detail": "pending first-pass package verification",
    }
    facts = gates.WorkflowFacts(
        phase="AMENDED_PREFLIGHT",
        no_pca_variant_status=raw_status,
        no_pca_variant_runnable=False,
    )
    registry = gates.build_gate_registry(gate_inputs)
    state = gates.derive_workflow_state(registry, facts)
    assert state.run_state == state_name
    registry_payload = {
        "allowed_classifications": [
            value.value for value in gates.EvidenceClassification
        ],
        "gates": {
            key: {
                "classification": gate.classification.value,
                "passed": gate.passed,
                "accepted": gate.accepted,
                "status": gate.status_token,
                "evidence_refs": list(gate.evidence_refs),
                "detail": gate.detail,
            }
            for key, gate in registry.evidence.items()
        },
        "derived_workflow": core._preflight_derived_payload(state),
    }
    (run / "provenance/evidence_gate_registry.json").write_text(
        json.dumps(registry_payload)
    )
    report = {
        "protocol": "AMENDMENT_01", "run_kind": state.run_kind,
        "elapsed_seconds": 0.0,
        "status": state.amended_preflight_status,
        "data_source_decision": state.data_source_decision,
        "global_blockers": core._expected_blocker_details(state),
        "source_input_integrity": "PASS",
        "prior_run_immutability": "PASS",
        "training_table_path": str(source),
        "training_table_sha256": source_sha,
        "candidate_split_hash": split_hashes,
        "historical_replay": {
            "literal_probability_tolerance_status": "FAIL",
            "direct_max_difference": historical[
                "direct_decimal_probability_max_absolute_difference"
            ],
            "source_dtype_roundtrip_status": historical[
                "source_dtype_roundtrip_status"
            ],
            "source_dtype_max_difference": historical[
                "source_dtype_float32_probability_max_absolute_difference"
            ],
            "serialized_reopen_status": replay_verification["status"],
            "serialized_reopen_summary_matches": replay_verification[
                "summary_matches_in_memory_calculation"
            ],
        },
        "feature_dependency_closure": state.gate_statuses[
            "feature_dependency_closure"
        ],
        "feature_semantics": state.gate_statuses["feature_semantics"],
        "gate_reconstruction": state.gate_statuses["gate_reconstruction"],
        "review_weight_parity": state.gate_statuses["review_weight_parity"],
        "no_pca": raw_status, "cuda_capability": cuda,
        "cuda_smoke": state.cuda_smoke_status,
        "ready_for_full_awaiting_external_review": False,
        "output_manifest_zip_verification": "BLOCKED",
        "r92_control": r92, "execution_ledger": ledger,
        "syntax_compile": command_payload,
        "tests": command_payload,
        "reviewer_snapshot_syntax_compile": command_payload,
        "reviewer_snapshot_tests": command_payload,
        "full_scientific_run_executed": False,
        "full_variants_completed": "0/10",
        "not_applicable_outputs": {
            "tables/SMOKE_REPORT.csv": "No Smoke because global gates failed",
            "metrics/SMOKE_REPORT.json": "No Smoke because global gates failed",
            "smoke_models_bundle": "No Smoke models were created in this Preflight",
        },
    }
    (run / "provenance/PREFLIGHT_REPORT.json").write_text(json.dumps(report))
    identity = {
        **identity_seed,
        "run_id": run.name, "run_kind": state.run_kind,
        "state": state.run_state,
        "data_source_decision": state.data_source_decision,
        "candidate_split_hash": split_hash,
        "split_locked": state.split_locked,
        "permitted_transitions": ["PACKAGE_REVIEW_EVIDENCE"],
        "full_authorized": False, "stability_authorized": False,
    }
    (run / "config/run_identity.lock.json").write_text(json.dumps(identity))
    (run / "RUN_STATUS.txt").write_text(gates.render_status_lines(
        state,
        extra_lines={
            "TRAINING_TABLE_PATH": str(source),
            "TRAINING_TABLE_SHA256": source_sha,
            "HISTORICAL_LITERAL_REPLAY_TOLERANCE_STATUS": "FAIL",
        },
    ))
    blocker_lines = [
        f"- `{code}` ({value['classification']}/{value['status']}): "
        f"{value['reason']} Evidence: {', '.join(value['evidence_refs'])}."
        for code, value in core._expected_blocker_details(state).items()
    ] or ["- None."]
    (run / "BLOCKERS.md").write_text(
        "# Global Blockers\n\n" + "\n".join(blocker_lines)
        + f"\n\n## Variant-Specific Limitation\n\n- `no_pca`: `{raw_status}`. "
        "This does not replace the global gate registry.\n"
    )
    return run, identity


def test_reviewer_portable_relative_code_resolution():
    assert CODE == Path(core.__file__).resolve().parent
    assert STUDY == core.STUDY_ROOT.resolve()


def test_task_spec_is_verbatim_protocol_amendment():
    assert core.TASK_SPEC.exists()
    assert core.sha256_file(core.TASK_SPEC) == "2a5bdae1a78eca765cbd69aa061f090541ff7c0aaf9eded9059396831d9d0eaf"


def test_conditional_scientific_blocker():
    core.assert_scientific_training_unblocked({})
    core.assert_scientific_training_unblocked([])
    with pytest.raises(core.ScientificBlocker, match="gate"):
        core.assert_scientific_training_unblocked({"gate": "failed"})
    legacy.assert_scientific_training_unblocked({})
    with pytest.raises(legacy.ScientificBlocker):
        legacy.assert_scientific_training_unblocked({"gate": "failed"})


def test_dependency_graph_contains_both_cross_family_composites():
    assert core.FEATURE_DEPENDENCIES["g_quality"] == set(core.FAMILIES)
    assert core.FEATURE_DEPENDENCIES["g_robust"] == {
        "shape_morphology", "texture", "spatial_context",
    }
    manifest = core.feature_manifest_frame()
    composites = manifest[manifest.is_cross_family_composite].feature_name.tolist()
    assert composites == ["g_quality", "g_robust"]
    composite_rows = manifest[manifest.is_cross_family_composite]
    assert set(composite_rows.primary_group) == {"cross_family_composite"}
    assert set(composite_rows.legacy_storage_bucket) == {"other_manual"}
    assert set(composite_rows.primary_group_status) == {
        "VERIFIED_CROSS_FAMILY_COMPOSITE"
    }
    assert set(manifest.verification_status) == {"VERIFIED"}


def test_dependency_closure_exact_variant_counts():
    variants = core.build_variant_feature_sets()
    expected = {
        "full_new_reference": 93, "manual_only": 57, "deep_only": 35,
        "no_color": 65, "no_shape": 78, "no_texture": 86,
        "no_embed_sim": 90, "no_review_aware_training_weights": 93,
        "no_safe_smote": 93, "no_deep_pca_features": 59,
    }
    assert {name: variants[name]["feature_count"] for name in expected} == expected
    assert variants["no_pca"]["expected_total_feature_count_if_mapped"] == 2107
    assert variants["no_pca"]["status"] == "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE"
    assert variants["no_pca"]["dependency_closure_linter"] == (
        "NOT_APPLICABLE_BLOCKED_RAW_MAPPING"
    )


def _materialized_raw_layer_contract(tmp_path: Path, *, nonfinite: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = [f"raw_embed_{index:04d}" for index in range(2048)]
    table = pd.DataFrame({
        "file_name": ["IMG_0001_a.jpg", "IMG_0001_b.jpg", "IMG_0002_c.jpg"],
        "ann_id": [10, 20, 30],
        "scale": [1.0, 1.0, 1.0],
    })
    table_path = tmp_path / "candidates.csv"
    table.to_csv(table_path, index=False)
    audit_table = pd.DataFrame({
        "img_folder": ["IMG_0001", "IMG_0001", "IMG_0002"],
        "image": ["a.jpg", "b.jpg", "c.jpg"],
        "id": [10, 20, 30],
    })
    audit_path = tmp_path / "audit.csv"
    audit_table.to_csv(audit_path, index=False)
    cache = np.arange(3 * 2048, dtype=np.float32).reshape(3, 2048)
    if nonfinite:
        cache[1, 100] = np.nan
    cache_path = tmp_path / "raw.npy"
    np.save(cache_path, cache)
    alignment = np.asarray([2, 0, 1], dtype="<i8")
    alignment_path = tmp_path / "alignment.npz"
    np.savez_compressed(alignment_path, cache_row_index=alignment)
    training_sidecar = table.copy()
    training_sidecar["raw_row_index"] = alignment
    training_sidecar_path = tmp_path / "training_keys.csv"
    training_sidecar.to_csv(training_sidecar_path, index=False)
    audit_sidecar = audit_table.copy()
    audit_sidecar["raw_row_index"] = alignment
    audit_sidecar_path = tmp_path / "audit_keys.csv"
    audit_sidecar.to_csv(audit_sidecar_path, index=False)
    encoder_weights = tmp_path / "encoder_weights.bin"
    encoder_weights.write_bytes(b"synthetic-encoder-weights")
    preprocessing_source = tmp_path / "preprocessing.py"
    preprocessing_source.write_text("RGB_RESIZE224_CENTER_CROP224_IMAGENET_NORM_L2\n")
    crop_mask_source = tmp_path / "crop_mask.py"
    crop_mask_source.write_text("IDENTICAL_MASKED_CROP_CONTRACT\n")
    key_columns = ["file_name", "ann_id", "scale"]
    key_hash = core.inventory_tools.canonical_key_digest(table, key_columns)
    raw_hash = core.sha256_bytes(("\n".join(raw) + "\n").encode())
    reference = {
        "status": "VERIFIED_ALIGNED_RAW_LAYER_REFERENCE",
        "population": "training",
        "raw_cache_path": str(cache_path),
        "raw_cache_sha256": core.sha256_file(cache_path),
        "candidate_table_path": str(table_path),
        "candidate_table_sha256": core.sha256_file(table_path),
        "alignment_index_path": str(alignment_path),
        "alignment_index_sha256": core.sha256_file(alignment_path),
        "alignment_mapping_sha256_int64_le": core.sha256_array(alignment),
        "sidecar_path": str(training_sidecar_path.resolve()),
        "sidecar_sha256": core.sha256_file(training_sidecar_path),
        "measured_exact_join_rate": 1.0,
        "measured_mapped_candidate_rows": len(table),
        "measured_unmatched_candidate_rows": 0,
        "measured_orphan_sidecar_rows": 0,
        "measured_cache_collision_rows": 0,
        "candidate_key_columns": key_columns,
        "candidate_key_sha256": key_hash,
        "candidate_row_order_key_sha256": key_hash,
        "row_count": len(table),
        "dimension_count": 2048,
        "dtype": "float32",
        "ordered_raw_feature_columns": raw,
        "ordered_raw_feature_columns_sha256": raw_hash,
    }
    audit_mapping_hash = core.sha256_array(alignment)
    population_links = {
        "training": {
            "raw_cache_path": str(cache_path.resolve()),
            "raw_cache_sha256": core.sha256_file(cache_path),
            "sidecar_path": str(training_sidecar_path.resolve()),
            "sidecar_sha256": core.sha256_file(training_sidecar_path),
            "alignment_mapping_sha256_int64_le": core.sha256_array(alignment),
        },
        "validation": {
            "raw_cache_path": str(cache_path.resolve()),
            "raw_cache_sha256": core.sha256_file(cache_path),
            "sidecar_path": str(training_sidecar_path.resolve()),
            "sidecar_sha256": core.sha256_file(training_sidecar_path),
            "alignment_mapping_sha256_int64_le": core.sha256_array(alignment),
        },
        "audit": {
            "raw_cache_path": str(cache_path.resolve()),
            "raw_cache_sha256": core.sha256_file(cache_path),
            "sidecar_path": str(audit_sidecar_path.resolve()),
            "sidecar_sha256": core.sha256_file(audit_sidecar_path),
            "alignment_mapping_sha256_int64_le": audit_mapping_hash,
        },
    }
    semantics_manifest = {
        "version": 1,
        "status": "VERIFIED_EXACT_EMBEDDING_SEMANTICS",
        "encoder": {
            "backbone": "synthetic-resnet50-v2",
            "weights_path": str(encoder_weights.resolve()),
            "weights_sha256": core.sha256_file(encoder_weights),
        },
        "preprocessing": {
            "ordered_steps": ["RGB", "Resize224", "CenterCrop224", "ImageNetNormalize", "L2Normalize"],
            "source_path": str(preprocessing_source.resolve()),
            "source_sha256": core.sha256_file(preprocessing_source),
        },
        "crop_mask": {
            "status": "VERIFIED_IDENTICAL_ACROSS_POPULATIONS",
            "source_path": str(crop_mask_source.resolve()),
            "source_sha256": core.sha256_file(crop_mask_source),
        },
        "population_cache_links": population_links,
    }
    semantics_path = tmp_path / "embedding_semantics.json"
    semantics_path.write_text(json.dumps(semantics_manifest, sort_keys=True))
    semantics_hash = core.sha256_file(semantics_path)
    reference.update({
        "embedding_semantics_manifest_path": str(semantics_path.resolve()),
        "embedding_semantics_manifest_sha256": semantics_hash,
    })
    contract = {
        "version": 1,
        "status": "VERIFIED_EXACT_RAW_LAYER_CONTRACT",
        "ordered_raw_feature_columns": raw,
        "ordered_raw_feature_columns_sha256": raw_hash,
        "embedding_semantics_manifest_path": str(semantics_path.resolve()),
        "embedding_semantics_manifest_sha256": semantics_hash,
        "populations": {
            "training": dict(reference),
            "validation": {**reference, "population": "validation"},
            "audit": {
                **reference,
                "population": "audit",
                "candidate_table_path": str(audit_path),
                "candidate_table_sha256": core.sha256_file(audit_path),
                "candidate_key_columns": ["img_folder", "image", "id"],
                "candidate_key_sha256": core.inventory_tools.canonical_key_digest(
                    audit_table, ["img_folder", "image", "id"]
                ),
                "candidate_row_order_key_sha256": core.inventory_tools.canonical_key_digest(
                    audit_table, ["img_folder", "image", "id"]
                ),
                "sidecar_path": str(audit_sidecar_path.resolve()),
                "sidecar_sha256": core.sha256_file(audit_sidecar_path),
            },
        },
    }
    return raw, table, contract, cache, alignment, audit_path


def test_mapping_boolean_alone_cannot_make_true_no_pca_runnable():
    raw = [f"raw_embed_{index:04d}" for index in range(2048)]
    variants = core.build_variant_feature_sets(
        raw_embedding_features=raw,
        raw_mapping_verified=True,
    )
    assert variants["no_pca"]["status"] == "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE"
    assert variants["no_pca"]["raw_layer_contract_validation"]["status"] == "FAIL"


def test_training_loader_hashes_the_same_open_bytes_before_and_after_parse(
    tmp_path, monkeypatch,
):
    path = tmp_path / "training.csv"
    pd.DataFrame([{
        "file_name": "IMG_1.jpg", "ann_id": 1, "scale": 1.0, "label": 1,
    }]).to_csv(path, index=False)
    expected = core.sha256_file(path)
    loaded = core.load_training_verified(path, expected)
    assert loaded.label.tolist() == [1]

    with pytest.raises(core.IntegrityError, match="reviewed source hash"):
        core.load_training_verified(path, "0" * 64)

    original_read_csv = core.pd.read_csv

    def mutate_after_parse(*args, **kwargs):
        frame = original_read_csv(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b"\n")
        return frame

    monkeypatch.setattr(core.pd, "read_csv", mutate_after_parse)
    with pytest.raises(core.IntegrityError, match="changed while"):
        core.load_training_verified(path, expected)


def test_historical_prediction_loader_preserves_literals_and_same_fd_hash(
    tmp_path, monkeypatch,
):
    path = tmp_path / "tile_preds.csv"
    path.write_text(
        "file_name,ann_id,scale,label,proba\n"
        "validation.jpg,7,1.0,1,0.10000000010000000\n"
    )
    expected = core.sha256_file(path)
    loaded = core.load_historical_predictions_verified(path, expected)
    assert loaded.proba.tolist() == ["0.10000000010000000"]

    original_read_csv = core.pd.read_csv

    def mutate_after_parse(*args, **kwargs):
        frame = original_read_csv(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b"\n")
        return frame

    monkeypatch.setattr(core.pd, "read_csv", mutate_after_parse)
    with pytest.raises(core.IntegrityError, match="changed while"):
        core.load_historical_predictions_verified(path, expected)


def test_historical_replay_source_binding_rejects_coordinated_literal_tamper(
    tmp_path,
):
    path = tmp_path / "tile_preds.csv"
    path.write_text(
        "file_name,ann_id,scale,label,proba\n"
        "validation_neg.jpg,2,1.0,0,0.1\n"
        "validation_pos.jpg,3,1.0,1,0.9\n"
    )
    source = core.load_historical_predictions_verified(
        path, core.sha256_file(path),
    )
    source_float32 = source.proba.map(float).to_numpy(np.float32)
    replay = source.rename(columns={
        "proba": "historical_saved_probability_literal",
    }).copy()
    replay["historical_saved_probability_float32_bits_hex"] = [
        f"0x{int(value):08x}" for value in source_float32.view(np.uint32)
    ]
    assert core._historical_prediction_source_matches_replay(source, replay)

    replay.loc[0, "historical_saved_probability_literal"] = "0.1000000001"
    assert np.float32(0.1000000001).view(np.uint32) == (
        source_float32[0].view(np.uint32)
    )
    assert not core._historical_prediction_source_matches_replay(source, replay)


def test_training_labels_are_bound_to_reviewed_split(tmp_path):
    path = tmp_path / "training.csv"
    pd.DataFrame([{
        "file_name": "IMG_1.jpg", "ann_id": 1, "scale": 1.0, "label": 1,
    }]).to_csv(path, index=False)
    training = core.load_training(path)
    split = training[["stable_candidate_id", "label"]].copy()
    core.assert_training_matches_reviewed_split(training, split)
    split.loc[0, "label"] = 0
    with pytest.raises(core.IntegrityError, match="labels differ"):
        core.assert_training_matches_reviewed_split(training, split)


def test_true_no_pca_requires_and_loads_verified_materialized_layers(tmp_path, monkeypatch):
    raw, table, contract, cache, alignment, audit_path = _materialized_raw_layer_contract(tmp_path)
    monkeypatch.setattr(core, "AUDIT_CSV", audit_path)
    variants = core.build_variant_feature_sets(
        raw_embedding_features=raw,
        raw_mapping_verified=True,
        raw_layer_contract=contract,
    )
    no_pca = variants["no_pca"]
    assert no_pca["status"] == "RUNNABLE"
    assert no_pca["feature_count"] == 2107
    assert no_pca["retained_features"][-2048:] == raw
    assert no_pca["dependency_closure_linter"] == "PASS"
    assert len(core.runnable_official_variants(variants)) == 10
    training_raw, validation_raw, evidence = core.load_no_pca_smoke_raw_layers(
        contract,
        table,
        train_indices=[0, 2],
        validation_indices=[1],
        raw_embedding_features=raw,
    )
    assert np.array_equal(training_raw, cache[alignment[[0, 2]]])
    assert np.array_equal(validation_raw, cache[alignment[[1]]])
    assert evidence["contract_validation"]["status"] == "PASS"
    assert evidence["audit_reference_validated"] is True
    smoke_raw_features = list(contract["ordered_raw_feature_columns"])
    assert smoke_raw_features == raw
    assert core.validate_no_pca_raw_layer_contract(
        contract, smoke_raw_features,
    )["status"] == "PASS"


def test_no_pca_contract_requires_top_level_ordered_raw_feature_list(
    tmp_path, monkeypatch,
):
    raw, _, contract, _, _, audit_path = _materialized_raw_layer_contract(
        tmp_path / "missing-list",
    )
    monkeypatch.setattr(core, "AUDIT_CSV", audit_path)
    contract.pop("ordered_raw_feature_columns")
    validation = core.validate_no_pca_raw_layer_contract(contract, raw)
    assert validation["status"] == "FAIL"
    assert "raw_feature_list" in validation["failed_checks"]


def test_no_pca_smoke_layer_loader_rejects_key_order_change_and_nonfinite_values(tmp_path, monkeypatch):
    raw, table, contract, _, _, audit_path = _materialized_raw_layer_contract(tmp_path / "finite")
    monkeypatch.setattr(core, "AUDIT_CSV", audit_path)
    tampered = table.iloc[[1, 0, 2]].reset_index(drop=True)
    with pytest.raises(core.ScientificBlocker, match="keys/order differ"):
        core.load_no_pca_smoke_raw_layers(
            contract, tampered, [0], [1], raw,
        )

    bad_raw, _, bad_contract, _, _, bad_audit_path = _materialized_raw_layer_contract(
        tmp_path / "nonfinite", nonfinite=True,
    )
    monkeypatch.setattr(core, "AUDIT_CSV", bad_audit_path)
    validation = core.validate_no_pca_raw_layer_contract(bad_contract, bad_raw)
    assert validation["status"] == "FAIL"
    assert "all_aligned_raw_values_finite" in validation["populations"]["training"]["failed_checks"]
    with pytest.raises(core.ScientificBlocker, match="valid materialized raw-layer contract"):
        core.load_no_pca_smoke_raw_layers(
            bad_contract, pd.read_csv(bad_contract["populations"]["training"]["candidate_table_path"]),
            [0], [1], bad_raw,
        )


def test_no_pca_contract_rejects_semantics_or_sidecar_tampering(tmp_path, monkeypatch):
    raw, _, contract, _, _, audit_path = _materialized_raw_layer_contract(
        tmp_path / "semantics",
    )
    monkeypatch.setattr(core, "AUDIT_CSV", audit_path)
    semantics_path = Path(contract["embedding_semantics_manifest_path"])
    semantics_path.write_text("{}")
    validation = core.validate_no_pca_raw_layer_contract(contract, raw)
    assert validation["status"] == "FAIL"
    assert "semantics_manifest_hash" in validation["failed_checks"]

    raw, _, contract, _, _, audit_path = _materialized_raw_layer_contract(
        tmp_path / "sidecar",
    )
    monkeypatch.setattr(core, "AUDIT_CSV", audit_path)
    sidecar_path = Path(contract["populations"]["training"]["sidecar_path"])
    sidecar = pd.read_csv(sidecar_path)
    sidecar.loc[0, "raw_row_index"] = 0
    sidecar.to_csv(sidecar_path, index=False)
    validation = core.validate_no_pca_raw_layer_contract(contract, raw)
    assert validation["status"] == "FAIL"
    assert "sidecar_hash" in validation["populations"]["training"]["failed_checks"]


def test_no_pca_contract_rejects_unkeyed_trailing_cache_rows(tmp_path, monkeypatch):
    raw, _, contract, cache, _, audit_path = _materialized_raw_layer_contract(
        tmp_path / "trailing-cache",
    )
    monkeypatch.setattr(core, "AUDIT_CSV", audit_path)
    cache_path = Path(contract["populations"]["training"]["raw_cache_path"])
    enlarged = np.vstack([cache, np.zeros((1, 2048), dtype=np.float32)])
    np.save(cache_path, enlarged)
    enlarged_hash = core.sha256_file(cache_path)
    for reference in contract["populations"].values():
        reference["raw_cache_sha256"] = enlarged_hash
    semantics_path = Path(contract["embedding_semantics_manifest_path"])
    semantics = json.loads(semantics_path.read_text())
    for link in semantics["population_cache_links"].values():
        link["raw_cache_sha256"] = enlarged_hash
    semantics_path.write_text(json.dumps(semantics, sort_keys=True))
    semantics_hash = core.sha256_file(semantics_path)
    contract["embedding_semantics_manifest_sha256"] = semantics_hash
    for reference in contract["populations"].values():
        reference["embedding_semantics_manifest_sha256"] = semantics_hash
    validation = core.validate_no_pca_raw_layer_contract(contract, raw)
    assert validation["status"] == "FAIL"
    assert "raw_cache_row_count_exact" in validation["populations"]["training"]["failed_checks"]


def test_dependency_closure_prevents_composite_contamination():
    variants = core.build_variant_feature_sets()
    assert "g_quality" not in variants["manual_only"]["retained_features"]
    assert "g_robust" in variants["manual_only"]["retained_features"]
    assert "g_quality" not in variants["no_color"]["retained_features"]
    assert "g_quality" not in variants["no_shape"]["retained_features"]
    assert "g_robust" not in variants["no_shape"]["retained_features"]
    assert "g_quality" not in variants["no_texture"]["retained_features"]
    assert "g_robust" not in variants["no_texture"]["retained_features"]
    assert {"embed_sim", "g_embed", "g_quality"}.isdisjoint(
        variants["no_embed_sim"]["retained_features"]
    )
    assert all(
        value["dependency_closure_linter"] == "PASS"
        for value in variants.values()
        if value["status"] == "RUNNABLE"
    )


def test_feature_list_hash_is_order_sensitive():
    variants = core.build_variant_feature_sets()
    retained = variants["full_new_reference"]["retained_features"]
    reversed_hash = core.sha256_bytes(("\n".join(reversed(retained)) + "\n").encode())
    assert reversed_hash != variants["full_new_reference"]["feature_list_sha256"]


def _synthetic_gate_frame() -> pd.DataFrame:
    frame = pd.DataFrame({
        "mean_L": [80.0, 160.0], "std_L": [10.0, 30.0],
        "delta_a": [3.0, 20.0], "delta_b": [4.0, 10.0],
        "circularity": [0.6, 0.8], "solidity": [0.7, 0.9],
        "eccentricity": [0.2, 0.5], "components_count": [1, 2],
        "extent": [0.5, 0.25], "grad_p90": [50.0, 150.0],
        "touching_border": [0, 1], "embed_sim": [0.7, 0.8],
    })
    for index in range(32):
        frame[f"embed_pca_{index}"] = np.array([index / 100.0, -index / 200.0], np.float32)
    gates = core.reconstruct_gates(frame, prototype_available=True)
    for column in gates:
        frame[column] = gates[column]
    return frame


def test_gate_reconstruction_parity_and_coverage():
    frame = _synthetic_gate_frame()
    rows = core.gate_validation_rows(frame, "synthetic")
    assert len(rows) == 8
    assert all(row["status"] == "PASS" for row in rows)
    assert all(row["coverage"] == 1.0 for row in rows)
    assert max(row["max_absolute_error"] for row in rows) == 0.0


def test_gate_reconstruction_rejects_missing_or_nonfinite_inputs():
    frame = _synthetic_gate_frame()
    with pytest.raises(core.ScientificBlocker, match="mean_L"):
        core.reconstruct_gates(frame.drop(columns="mean_L"), prototype_available=True)
    frame.loc[0, "grad_p90"] = np.nan
    with pytest.raises(core.ScientificBlocker, match="grad_p90"):
        core.reconstruct_gates(frame, prototype_available=True)


def test_embed_similarity_canonical_affine_transform():
    raw = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    assert np.array_equal(
        core.canonical_embed_similarity(raw),
        np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.0]),
    )


def test_training_style_uniform_grid_reconstruction():
    frame = pd.DataFrame({"cx": [0.0, 170.0, 340.0], "cy": [0.0, 171.0, 341.0], "scale": [1, 1, 1]})
    grid = core.reconstruct_uniform_grid(frame, audit=False)
    assert grid.grid_c.tolist() == [0, 0, 1]
    assert grid.grid_r.tolist() == [0, 1, 1]
    assert (grid[["grid_c", "grid_r"]].to_numpy() <= 2).all()


def _geometry(rows):
    return pd.DataFrame(rows, columns=["bbox_x", "bbox_y", "bbox_w", "bbox_h", "touching_border"])


def test_border_boundary_inclusive_and_touching_branch():
    frame = _geometry([
        [3.999, 100, 50, 50, 0], [4.0, 100, 50, 50, 0], [4.001, 100, 50, 50, 0],
        [100, 3.999, 50, 50, 0], [100, 4.0, 50, 50, 0], [100, 4.001, 50, 50, 0],
        [458.001, 100, 50, 50, 0], [458.0, 100, 50, 50, 0], [457.999, 100, 50, 50, 0],
        [100, 458.001, 50, 50, 0], [100, 458.0, 50, 50, 0], [100, 457.999, 50, 50, 0],
        [100, 100, 50, 50, 1], [100, 100, 50, 50, 0],
    ])
    border, _ = core.corrected_border_tiny_masks(frame)
    assert border.tolist() == [True, True, False, True, True, False, True, True, False, True, True, False, True, False]
    legacy_border, _ = legacy.border_and_tiny_masks(frame)
    assert np.array_equal(border, legacy_border)


def test_tiny_dimension_and_area_boundaries_are_strict():
    frame = _geometry([
        [100, 100, 31.999, 60, 0], [100, 100, 32.0, 60, 0], [100, 100, 32.001, 60, 0],
        [100, 100, 40.0, 37.499975, 0], [100, 100, 40.0, 37.5, 0], [100, 100, 40.0, 37.500025, 0],
    ])
    _, tiny = core.corrected_border_tiny_masks(frame)
    assert tiny.tolist() == [True, False, False, True, False, False]


def test_r91_join_cardinality_source_behavior():
    _, info = core.r91_probability_mapping()
    assert info["source_behavior"] == "groupby(file_name,sort=False).proba.max"
    assert info["source_rows"] == 55842
    assert info["unique_file_names"] == 1026
    assert info["duplicate_excess_rows"] == 54816
    assert info["duplicate_file_name_keys"] == 1026
    assert info["source_explicitly_justifies_maximum"] is True


def test_historical_review_weight_join_cardinality_classes():
    historical = core.load_training(core.HISTORICAL_TRAINING_CSV)
    train, _, _ = core.historical_group_split(historical.card_id.to_numpy())
    _, info = core.review_weights(historical.iloc[train], historical_geometry=True)
    assert info["shared_unique_file_names"] == 817
    assert info["shared_key_cardinality_cases"] == {
        "one_to_one": 0, "one_to_many": 0, "many_to_one": 0, "many_to_many": 817,
    }
    assert info["mapped_rows"] == 59294
    assert info["missing_rows"] == 231730
    assert info["unaggregated_many_to_many_join_rows"] == 4148392


def test_deterministic_shuffle_repeats_and_changes_with_seed():
    first = core.deterministic_permutation(100, 42)
    second = core.deterministic_permutation(100, 42)
    third = core.deterministic_permutation(100, 43)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)
    assert sorted(first.tolist()) == list(range(100))


def test_no_safe_smote_disables_only_smote():
    semantics = core.variant_training_semantics("no_safe_smote")
    assert semantics["augmentation_enabled"]
    assert semantics["mixup_enabled"] and semantics["dropout_enabled"] and semantics["jitter_enabled"]
    assert semantics["raw_split_shuffle_enabled"]
    assert semantics["review_weights_enabled"]
    assert not semantics["safe_smote_enabled"]
    assert legacy.variant_augmentation_enabled("no_safe_smote", smoke=True)
    assert not legacy.variant_smote_enabled("no_safe_smote")


def test_full_and_no_weight_resampling_plan_equality():
    full = core.variant_training_semantics("full_new_reference")
    no_weight = core.variant_training_semantics("no_review_aware_training_weights")
    for key in ("augmentation_enabled", "mixup_enabled", "dropout_enabled", "jitter_enabled", "raw_split_shuffle_enabled", "safe_smote_enabled", "seed"):
        assert full[key] == no_weight[key]
    assert full["review_weights_enabled"] and not no_weight["review_weights_enabled"]


def test_full_and_no_smote_pre_smote_plan_equality():
    full = core.variant_training_semantics("full_new_reference")
    no_smote = core.variant_training_semantics("no_safe_smote")
    for key in ("augmentation_enabled", "mixup_enabled", "dropout_enabled", "jitter_enabled", "raw_split_shuffle_enabled", "review_weights_enabled", "seed"):
        assert full[key] == no_smote[key]
    assert full["safe_smote_enabled"] and not no_smote["safe_smote_enabled"]


def _small_augmentation_inputs():
    frame = pd.DataFrame({
        "continuous_a": np.linspace(0, 1, 40),
        "continuous_b": np.linspace(1, 3, 40) ** 2,
        "discrete": np.tile([0, 1], 20),
    })
    labels = np.array([0] * 32 + [1] * 8, dtype=np.int8)
    weights = np.linspace(0.4, 1.0, 40, dtype=np.float32)
    ids = [f"row_{index}" for index in range(40)]
    return frame, labels, weights, ids


def test_actual_full_vs_no_weight_xy_and_lineage_parity():
    frame, labels, weights, ids = _small_augmentation_inputs()
    full = core.augment_with_lineage(frame, labels, weights, ids, smote_enabled=True)
    no_weight = core.augment_with_lineage(frame, labels, np.ones(40, np.float32), ids, smote_enabled=True)
    assert np.array_equal(full[0], no_weight[0])
    assert np.array_equal(full[1], no_weight[1])
    assert np.array_equal(full[3], no_weight[3])
    assert not np.array_equal(full[2], no_weight[2])


def test_augmentation_consumes_locked_input_order_without_second_shuffle():
    frame, labels, weights, ids = _small_augmentation_inputs()
    result = core.augment_with_lineage(frame, labels, weights, ids, smote_enabled=False)
    assert result[-1][0]["stage"] == "raw_locked_post_split_shuffle"
    assert result[-1][-2]["stage"] == "post_shuffle"
    assert result[-1][-2]["shuffle_applied"] is False
    assert np.array_equal(result[0][:40], frame.to_numpy(dtype=np.float32))
    assert result[3][:40].tolist() == ids


def test_actual_no_smote_pre_smote_parity_and_zero_synthetic_rows():
    frame, labels, weights, ids = _small_augmentation_inputs()
    full = core.augment_with_lineage(frame, labels, weights, ids, smote_enabled=True)
    no_smote = core.augment_with_lineage(frame, labels, weights, ids, smote_enabled=False)
    assert full[-1][-2] == no_smote[-1][-2]
    assert no_smote[-1][-1]["safe_smote_synthetic_rows"] == 0
    assert no_smote[-1][-1]["rows"] == no_smote[-1][-2]["rows"]


def test_augmentation_counts_match_historical_label_counts():
    frame, train, _, _ = _historical_split_fixture()
    labels = frame.iloc[train].label.to_numpy(np.int8)
    historical = core.load_training(core.HISTORICAL_TRAINING_CSV).iloc[train]
    continuous_count = int((historical[core.get_r92_feature_order()].nunique(dropna=False) > 10).sum())
    assert continuous_count == 88
    counts = core.augmentation_plan_counts(labels, continuous_feature_count=continuous_count)
    assert counts["post_mixup"] == {"0": 323982, "1": 112554}
    assert counts["post_dropout"] == {"0": 647964, "1": 225108}
    assert counts["post_jitter"] == {"0": 971633, "1": 337975}
    assert sum(counts["post_jitter"].values()) == 1309608
    assert counts["safe_smote_synthetic_rows"] == 147841
    assert counts["final_fit_rows"] == 1457449


def test_safe_smote_disabled_has_zero_synthetic_rows():
    matrix = np.arange(40, dtype=np.float32).reshape(20, 2)
    labels = np.array([0] * 16 + [1] * 4)
    weights = np.ones(20, dtype=np.float32)
    x_result, y_result, w_result, info = legacy.safe_smote_resample(matrix, labels, weights, seed=42, enabled=False)
    assert info["synthetic_rows"] == 0
    assert np.array_equal(x_result, matrix)
    assert np.array_equal(y_result, labels)
    assert np.array_equal(w_result, weights)


def test_split_hash_includes_assignments_not_only_ids():
    ids = ["a", "b", "c"]
    first = core.split_assignment_hash(ids, ["training", "training", "validation"], [0, 1, 0], [1, 1, 1])
    second = core.split_assignment_hash(ids, ["validation", "training", "training"], [0, 1, 0], [1, 1, 1])
    assert first != second


def test_split_hash_includes_locked_within_split_order():
    ids = ["a", "b", "c"]
    first = core.split_assignment_hash(ids, ["training"] * 3, [0, 1, 0], [1, 1, 1], [0, 1, 2])
    second = core.split_assignment_hash(ids, ["training"] * 3, [0, 1, 0], [1, 1, 1], [1, 0, 2])
    assert first != second


def test_split_card_leakage_detection():
    passing = core.validate_card_split(["A", "A", "B", "B"], ["training", "training", "validation", "validation"])
    failing = core.validate_card_split(["A", "A", "B"], ["training", "validation", "validation"])
    assert passing["status"] == "PASS"
    assert failing == {"status": "FAIL", "leaking_cards": ["A"], "leaking_card_count": 1}


def test_historical_split_reconstruction_counts_from_real_source():
    _, train, heldout, info = _historical_split_fixture()
    assert (len(train), len(heldout)) == (291024, 72539)
    assert len(info["train_cards"]) == 66
    assert len(info["heldout_cards"]) == 18
    assert set(info["train_cards"]).isdisjoint(info["heldout_cards"])


def test_r92_probability_gate_boundary_and_source_hash():
    common = dict(ordered_population_match=True, expected_metrics_match=True, selected_score_hash_match=True)
    assert core.r92_control_gate(maximum_probability_difference=1e-12, **common)
    assert not core.r92_control_gate(maximum_probability_difference=np.nextafter(1e-12, np.inf), **common)
    assert not core.r92_control_gate(maximum_probability_difference=0.0, **{**common, "selected_score_hash_match": False})
    assert not core.r92_control_gate(maximum_probability_difference=0.0, **{**common, "ordered_population_match": False})


def test_historical_replay_csv_reopens_with_literal_and_bitwise_parity(runtime_case):
    saved_literal = np.array(["0.9999994", "1.422173e-08"], dtype=object)
    saved_decimal = np.array([float(value) for value in saved_literal], np.float64)
    replayed = saved_decimal.astype(np.float32)
    saved_float32 = saved_decimal.astype(np.float32)
    direct_error = np.abs(saved_decimal - replayed.astype(np.float64))
    frame = pd.DataFrame({
        "historical_saved_probability_literal": saved_literal,
        "historical_saved_probability_decimal": saved_decimal,
        "historical_saved_probability_float64_hex": builder._float64_hex(saved_decimal),
        "historical_saved_probability_float32": saved_float32,
        "historical_saved_probability_float32_bits_hex": builder._float32_bits_hex(saved_float32),
        "replayed_probability_float32": replayed,
        "replayed_probability_float32_bits_hex": builder._float32_bits_hex(replayed),
        "absolute_difference_direct_decimal": direct_error,
        "absolute_difference_direct_decimal_float64_hex": builder._float64_hex(direct_error),
        "absolute_difference_source_dtype_float32": np.abs(saved_float32 - replayed),
    })
    path = runtime_case / "historical_prediction_replay.csv.gz"
    core.atomic_write_frame(frame, path, float_format="%.17g")
    verified = builder._verify_historical_replay_artifact(path)
    assert verified["status"] == "PASS"
    assert verified["summary"]["row_count"] == 2
    assert verified["summary"]["direct_decimal_max_absolute_difference"] == float(
        direct_error.max()
    )
    assert verified["summary"]["saved_vs_replayed_float32_bit_mismatch_count"] == 0


def test_gate_formula_manifest_has_all_eight_unique_source_definitions():
    manifest = builder._gate_formula_manifest()
    assert manifest["status"] == "PASS"
    assert manifest["checks"]["source_formula_equivalence_probes_pass"] is True
    assert manifest["checks"]["all_helper_definitions_uniquely_located"] is True
    assert manifest["checks"]["all_casts_and_controls_uniquely_located"] is True
    assert manifest["source_formula_equivalence_probes"]["status"] == "PASS"
    assert len(manifest["source_formula_equivalence_probes"]["rows"]) == 8
    assert manifest["source_formula_equivalence_probes"]["control_dependencies"]["missing_prototype_fail_closed"] is True
    assert set(manifest["formulas"]) == {
        "g_border", "g_color", "g_embed", "g_light",
        "g_maha", "g_quality", "g_robust", "g_shape",
    }
    assert all(
        formula["source_evidence_verified"]
        for formula in manifest["formulas"].values()
    )
    assert set(manifest["shared_source_controls"]["helpers"]) == {"_clamp", "_safe_div"}
    assert set(manifest["shared_source_controls"]["casts_and_controls"]) == {
        "have_mu", "have_pca", "PCA_NAMES", "touch_int_cast",
        "components_int_cast", "numeric_input_cast_block", "pca_branch",
    }


def test_gate_reconstruction_matches_source_out_of_domain_and_have_mu_fallback():
    frame = _synthetic_gate_frame().iloc[[0]].copy()
    frame["mean_L"] = 300.0
    frame["std_L"] = -64.0
    frame["embed_sim"] = 1.25
    frame["touching_border"] = -1
    with_mu = core.reconstruct_gates(frame, prototype_available=True)
    with pytest.raises(core.ScientificBlocker, match="verified prototype"):
        core.reconstruct_gates(frame, prototype_available=False)
    assert with_mu.g_light.iloc[0] == pytest.approx(300.0 / (2.0 * 255.0))
    assert with_mu.g_embed.iloc[0] == 1.25
    assert with_mu.g_border.iloc[0] == 1.0


def test_execution_ledger_derives_no_full_status_from_argv_and_artifacts(
    runtime_case, monkeypatch,
):
    monkeypatch.setattr(sys, "argv", ["run_study.py", "--preflight"])
    command = core.run_command([sys.executable, "-c", "raise SystemExit(0)"], cwd=runtime_case)
    passing = builder._execution_ledger(runtime_case, [command])
    assert passing["status"] == "PASS"
    assert passing["full_scientific_run_executed"] is False
    (runtime_case / "models").mkdir()
    (runtime_case / "models/full.ubj").write_bytes(b"not-a-model")
    blocked = builder._execution_ledger(runtime_case, [command])
    assert blocked["status"] == "FAIL"
    assert blocked["full_scientific_run_executed"] is True
    assert blocked["full_variants_completed"] == 0


def test_execution_ledger_derives_typed_execution_counts(runtime_case, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_study.py", "--preflight"])
    command = core.run_command([sys.executable, "-c", "raise SystemExit(0)"], cwd=runtime_case)
    (runtime_case / "tables").mkdir()
    pd.DataFrame([
        {"variant": "full_new_reference", "status": "PASS"},
        {"variant": "no_color", "status": "DONE"},
        {"variant": "no_shape", "status": "FAIL"},
    ]).to_csv(runtime_case / "tables/SMOKE_REPORT.csv", index=False)
    smoke = builder._execution_ledger(runtime_case, [command])
    assert smoke["smoke_report_status"] == "PARSED_VALID"
    assert smoke["smoke_variants_completed"] == 2
    assert smoke["smoke_variants_completed_ids"] == ["full_new_reference", "no_color"]
    assert smoke["official_smoke_variants_completed"] == 2
    assert smoke["exploratory_smoke_variants_completed"] == 0
    assert smoke["status"] == "FAIL"

    payload = runtime_case / "metrics/full_new_reference.json"
    payload.parent.mkdir(exist_ok=True)
    payload.write_text("{}")
    model = runtime_case / "models/full_new_reference"
    model.mkdir(parents=True)
    (model / "STATUS.txt").write_text("DONE\n")
    (model / "completion_manifest.json").write_text(json.dumps({
        "variant": "full_new_reference",
        "files": {"metrics/full_new_reference.json": core.sha256_file(payload)},
    }))
    (runtime_case / "statistics/stability").mkdir(parents=True)
    (runtime_case / "statistics/stability/result.json").write_text("{}")
    full = builder._execution_ledger(runtime_case, [command])
    assert full["full_variants_completed"] == 1
    assert full["validated_full_variant_completions"] == ["full_new_reference"]
    assert full["stability_executed"] is True
    assert full["full_scientific_run_executed"] is True


def test_execution_ledger_rejects_unknown_duplicate_or_conflicting_smoke_rows(
    runtime_case, monkeypatch,
):
    monkeypatch.setattr(sys, "argv", ["run_study.py", "--preflight"])
    (runtime_case / "tables").mkdir()
    pd.DataFrame([
        {"variant": "full_new_reference", "status": "PASS"},
        {"variant": "full_new_reference", "status": "FAIL"},
        {"variant": "invented_variant", "status": "PASS"},
    ]).to_csv(runtime_case / "tables/SMOKE_REPORT.csv", index=False)
    ledger = builder._execution_ledger(runtime_case, [])
    assert ledger["status"] == "FAIL"
    assert ledger["smoke_report_status"] == "INVALID_CONTENT"
    assert ledger["smoke_duplicate_variants"] == ["full_new_reference"]
    assert ledger["smoke_conflicting_variants"] == ["full_new_reference"]
    assert ledger["smoke_unknown_variants"] == ["invented_variant"]
    assert ledger["official_smoke_variants_completed"] == 0


def test_workflow_facts_receive_execution_ledger_smoke_counts(runtime_case, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_study.py", "--preflight"])
    (runtime_case / "tables").mkdir()
    pd.DataFrame([
        {"variant": "full_new_reference", "status": "PASS"},
        {"variant": core.EXPLORATORY_VARIANT, "status": "PASS"},
    ]).to_csv(runtime_case / "tables/SMOKE_REPORT.csv", index=False)
    ledger = builder._execution_ledger(runtime_case, [])
    facts = gates.WorkflowFacts(
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
        official_smoke_variants_completed=ledger["official_smoke_variants_completed"],
        exploratory_smoke_variants_completed=ledger[
            "exploratory_smoke_variants_completed"
        ],
    )
    state = gates.derive_workflow_state(
        gates.build_gate_registry({
            spec.key: {
                "classification": "VERIFIED", "passed": True,
                "evidence_refs": [f"provenance/{spec.key}.json"],
            }
            for spec in gates.DEFAULT_GATE_SPECS
        }),
        facts,
    )
    rendered = gates.render_status_lines(state)
    assert "OFFICIAL_SMOKE_VARIANTS_COMPLETED=1/9" in rendered
    assert "EXPLORATORY_SMOKE_VARIANTS_COMPLETED=1/1" in rendered


def test_smoke_execution_ledger_derives_nine_official_and_exploratory(runtime_case):
    runnable = [name for name in core.OFFICIAL_VARIANTS if name != "no_pca"]
    (runtime_case / "tables").mkdir()
    pd.DataFrame([
        *({"variant": name, "status": "PASS"} for name in runnable),
        {"variant": core.EXPLORATORY_VARIANT, "status": "PASS"},
    ]).to_csv(runtime_case / "tables/SMOKE_REPORT.csv", index=False)

    ledger = core.smoke_execution_ledger(
        runtime_case,
        command_line="run_study.py --smoke --preflight-run /reviewed/preflight",
        runnable_official=runnable,
    )

    assert ledger["status"] == "PASS"
    assert ledger["official_smoke_variants_completed"] == 9
    assert ledger["exploratory_smoke_variants_completed"] == 1
    assert ledger["full_scientific_run_executed"] is False
    assert ledger["full_variants_completed"] == 0


def test_smoke_execution_ledger_rejects_malformed_rows_and_full_artifacts(runtime_case):
    runnable = [name for name in core.OFFICIAL_VARIANTS if name != "no_pca"]
    (runtime_case / "tables").mkdir()
    pd.DataFrame([
        {"variant": runnable[0], "status": "PASS"},
        {"variant": runnable[0], "status": "FAIL"},
        {"variant": "invented", "status": "PASS"},
    ]).to_csv(runtime_case / "tables/SMOKE_REPORT.csv", index=False)
    (runtime_case / "statistics").mkdir()
    (runtime_case / "statistics/full.json").write_text("{}")

    ledger = core.smoke_execution_ledger(
        runtime_case,
        command_line="run_study.py --smoke --full",
        runnable_official=runnable,
    )

    assert ledger["status"] == "FAIL"
    assert ledger["smoke_report_status"] == "INVALID_CONTENT"
    assert ledger["smoke_duplicate_variants"] == [runnable[0]]
    assert ledger["smoke_conflicting_variants"] == [runnable[0]]
    assert ledger["smoke_unknown_variants"] == ["invented"]
    assert ledger["forbidden_arguments_detected"] == ["--full"]
    assert ledger["full_scientific_run_executed"] is True


def test_smoke_staging_verification_promotes_package_gate_and_ready_state():
    gate_inputs = {
        spec.key: {
            "classification": "VERIFIED", "passed": True,
            "evidence_refs": [f"provenance/{spec.key}.json"],
        }
        for spec in gates.DEFAULT_GATE_SPECS
    }
    gate_inputs["output_manifest_zip_verification"] = {
        "classification": "BLOCKED", "passed": None,
        "evidence_refs": ["provenance/package_staging_verification.json"],
    }
    facts = gates.WorkflowFacts(
        phase="CUDA_SMOKE",
        official_smoke_variants_completed=9,
        exploratory_smoke_variants_completed=1,
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
    )
    before = gates.derive_workflow_state(
        gates.build_gate_registry(gate_inputs), facts,
    )
    assert before.run_state == "SMOKE_COMPLETE"
    assert before.cuda_smoke_status == "PASS"
    assert before.ready_for_full_awaiting_external_review is False
    assert before.gate_statuses["output_manifest_zip_verification"] == "BLOCKED"

    promoted, registry, after = core.promote_smoke_package_gate(
        gate_inputs,
        facts,
        {
            "crc_status": "PASS",
            "independent_reopen_member_verification": "PASS",
            "verification_failures": [],
            "package_completeness": {
                "required_files_status": "PASS",
                "status_identity_match": "PASS",
                "smoke_semantic_completeness": {
                    "status": "PASS", "phase": "STAGING",
                },
            },
        },
    )

    assert promoted["output_manifest_zip_verification"]["passed"] is True
    assert registry.accepted("output_manifest_zip_verification")
    assert after.ready_for_full_awaiting_external_review is True
    assert after.gate_statuses["output_manifest_zip_verification"] == "PASS"


def test_smoke_staging_verification_fails_closed_on_incomplete_reopen_evidence():
    gate_inputs = {
        spec.key: {
            "classification": "VERIFIED", "passed": True,
            "evidence_refs": [f"provenance/{spec.key}.json"],
        }
        for spec in gates.DEFAULT_GATE_SPECS
    }
    facts = gates.WorkflowFacts(
        phase="CUDA_SMOKE",
        official_smoke_variants_completed=9,
        exploratory_smoke_variants_completed=1,
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
    )
    _, registry, state = core.promote_smoke_package_gate(
        gate_inputs,
        facts,
        {
            "crc_status": "PASS",
            "independent_reopen_member_verification": "FAIL",
            "verification_failures": ["member"],
            "package_completeness": {
                "required_files_status": "PASS",
                "status_identity_match": "PASS",
            },
        },
    )
    assert not registry.accepted("output_manifest_zip_verification")
    assert not state.ready_for_full_awaiting_external_review
    assert "OUTPUT_MANIFEST_ZIP_FAILURE" in state.blocker_codes


def test_smoke_staging_promotion_requires_semantic_completeness_pass():
    gate_inputs = {
        spec.key: {
            "classification": "VERIFIED", "passed": True,
            "evidence_refs": [f"provenance/{spec.key}.json"],
        }
        for spec in gates.DEFAULT_GATE_SPECS
    }
    facts = gates.WorkflowFacts(
        phase="CUDA_SMOKE",
        official_smoke_variants_completed=9,
        exploratory_smoke_variants_completed=1,
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
    )
    _, registry, state = core.promote_smoke_package_gate(
        gate_inputs,
        facts,
        {
            "crc_status": "PASS",
            "independent_reopen_member_verification": "PASS",
            "verification_failures": [],
            "package_completeness": {
                "required_files_status": "PASS",
                "status_identity_match": "PASS",
            },
        },
    )
    assert not registry.accepted("output_manifest_zip_verification")
    assert not state.ready_for_full_awaiting_external_review


def test_smoke_model_bundle_has_exact_internal_manifest_and_reopen_hashes(
    runtime_case, monkeypatch,
):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    (runtime_case / "results").mkdir()
    model_root = runtime_case / ".runtime/models/run_smoke"
    model_root.mkdir(parents=True)
    (model_root / "full_new_reference.ubj").write_bytes(
        _valid_test_booster_bytes(2)
    )
    (model_root / "manual_only.ubj").write_bytes(
        _valid_test_booster_bytes(1)
    )

    result = core._package_smoke_models(model_root, "run_smoke")

    assert result["crc_status"] == "PASS"
    assert result["reopen_member_hash_status"] == "PASS"
    assert result["member_count"] == 3
    with zipfile.ZipFile(result["path"]) as archive:
        assert set(archive.namelist()) == {
            "run_smoke_smoke_models/BUNDLE_MANIFEST.tsv",
            "run_smoke_smoke_models/full_new_reference.ubj",
            "run_smoke_smoke_models/manual_only.ubj",
        }
        assert archive.testzip() is None


def test_smoke_model_bundle_rejects_empty_artifact_set(runtime_case, monkeypatch):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    (runtime_case / "results").mkdir()
    empty = runtime_case / ".runtime/models/empty"
    empty.mkdir(parents=True)
    with pytest.raises(core.IntegrityError, match="nonempty regular"):
        core._package_smoke_models(empty, "empty")


def test_smoke_model_bundle_rejects_arbitrary_ubj_bytes(runtime_case, monkeypatch):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    (runtime_case / "results").mkdir()
    model_root = runtime_case / ".runtime/models/fabricated"
    model_root.mkdir(parents=True)
    (model_root / "full_new_reference.ubj").write_bytes(b"not-an-xgboost-model")
    with pytest.raises(core.IntegrityError, match="not a loadable XGBoost model"):
        core._package_smoke_models(model_root, "fabricated")


def test_smoke_model_bundle_failure_never_publishes_canonical_zip(
    runtime_case, monkeypatch,
):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    (runtime_case / "results").mkdir()
    model_root = runtime_case / ".runtime/models/failing"
    model_root.mkdir(parents=True)
    (model_root / "full_new_reference.ubj").write_bytes(
        _valid_test_booster_bytes(1)
    )
    monkeypatch.setattr(core.zipfile.ZipFile, "testzip", lambda self: "bad/member")
    with pytest.raises(core.IntegrityError, match="verification failed"):
        core._package_smoke_models(model_root, "failing")
    destination = (
        runtime_case / "results/failing_NON_SCIENTIFIC_smoke_models.zip"
    )
    assert not destination.exists()
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


@pytest.mark.parametrize(
    ("collision_kind", "expected_type", "expected_link_target"),
    [
        ("regular", "REGULAR_FILE", None),
        ("directory", "DIRECTORY", None),
        ("dangling_symlink", "SYMLINK", "missing-target"),
        ("fifo", "FIFO", None),
    ],
)
def test_model_collision_evidence_uses_lstat_without_reading_target(
    runtime_case, monkeypatch, collision_kind, expected_type,
    expected_link_target,
):
    path = runtime_case / "collision"
    if collision_kind == "regular":
        path.write_bytes(b"unowned")
    elif collision_kind == "directory":
        path.mkdir()
    elif collision_kind == "dangling_symlink":
        path.symlink_to("missing-target")
    else:
        os.mkfifo(path)
    observed_stat = os.lstat(path)
    monkeypatch.setattr(
        core,
        "sha256_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("collision target must not be read")
        ),
    )

    evidence = core._path_collision_evidence(path)

    assert evidence == {
        "status": "PREEXISTING_UNOWNED_COLLISION",
        "path": str(path),
        "lstat_type": expected_type,
        "st_dev": observed_stat.st_dev,
        "st_ino": observed_stat.st_ino,
        "link_target": expected_link_target,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0), (0.0, 0), ("0", 0), (1, 1), (1.0, 1), ("1", 1)],
)
def test_audit_label_canonicalization_accepts_equivalent_binary_representations(
    value, expected,
):
    labels = core.canonicalize_audit_labels(
        pd.DataFrame({"human_label": [value]})
    )

    assert labels.tolist() == [expected]
    assert labels.dtype == np.dtype("int8")


@pytest.mark.parametrize("value", [None, np.nan, "", "invalid", -1, 2, 0.5])
def test_audit_label_canonicalization_rejects_invalid_or_missing_values(value):
    with pytest.raises(core.IntegrityError, match="non-null binary values"):
        core.canonicalize_audit_labels(pd.DataFrame({"human_label": [value]}))


def test_audit_label_canonicalization_rejects_missing_column():
    with pytest.raises(core.IntegrityError, match="missing required human_label"):
        core.canonicalize_audit_labels(pd.DataFrame({"other": [0]}))


def test_audit_label_canonicalization_regression_builder_validator_ordered_identity(
    runtime_case, monkeypatch,
):
    raw = pd.DataFrame({
        "img_folder": ["IMG_0001", "IMG_0001"],
        "image": ["a.jpg", "b.jpg"],
        "id": [1, 2],
        "human_label": [0.0, 1.0],
        "review_weight": [1.0, 1.0],
    })
    integer_identity = raw.copy()
    integer_identity["human_label"] = integer_identity.human_label.astype(np.int8)
    identity = ["img_folder", "image", "id", "human_label"]
    assert not raw[identity].astype(str).equals(integer_identity[identity].astype(str))

    audit_path = runtime_case / "audit.csv"
    raw.to_csv(audit_path, index=False)
    monkeypatch.setattr(core, "AUDIT_CSV", audit_path)
    expected_sha256 = core.sha256_file(audit_path)

    builder_frame, _ = builder._clean_audit(expected_sha256)
    validator_frame = core._clean_canonical_audit_frame(
        audit_path, expected_sha256,
    )

    pd.testing.assert_frame_equal(
        builder_frame[identity].reset_index(drop=True),
        validator_frame[identity].reset_index(drop=True),
        check_dtype=True,
        check_exact=True,
    )
    assert builder_frame.human_label.dtype == np.dtype("int8")
    assert validator_frame.human_label.dtype == np.dtype("int8")


def test_selected_v3_score_source_hash_is_locked():
    member = core.immutable_zip_member_evidence(core.SELECTED_V3_ZIP, "scored_candidates_r92.csv")
    assert core.sha256_file(core.SELECTED_V3_SCORES) == member["member_sha256"]
    assert core.SELECTED_V3_SCORES.stat().st_size == member["member_size_bytes"]
    assert core.sha256_file(core.SELECTED_V3_ZIP) == core.EXPECTED_HASHES[str(core.SELECTED_V3_ZIP)]


def test_r92_expected_metrics_are_derived_from_hashed_v3_artifact():
    audit, _ = builder._clean_audit(core.sha256_file(core.AUDIT_CSV))
    result, _ = core.score_r92_control(audit)
    assert result["status"] == "PASS"
    assert result["expected_metrics_match"]
    assert result["expected_metrics_source"]["path"] == str(core.SELECTED_V3_SCORES)
    assert result["expected_metrics_source"]["sha256"] == core.sha256_file(core.SELECTED_V3_SCORES)
    assert result["expected_population_source"]["path"] == str(core.SELECTED_V3_POPULATION)


def test_real_r92_population_canonicalization_diagnostic_binds_both_cleaners():
    source_sha256 = core.sha256_file(core.AUDIT_CSV)
    builder_audit, _ = builder._clean_audit(source_sha256)
    validator_audit = core._clean_canonical_audit_frame(
        core.AUDIT_CSV, source_sha256,
    )
    r92_control, _ = core.score_r92_control(builder_audit)

    diagnostic = builder._r92_population_canonicalization_diagnostic(
        builder_audit, validator_audit, r92_control,
    )

    expected_rows = len(pd.read_csv(core.SELECTED_V3_SCORES, low_memory=False))
    assert diagnostic["builder_population_row_count"] == expected_rows
    assert diagnostic["independent_validator_population_row_count"] == expected_rows
    assert diagnostic["builder_population_sha256"] == diagnostic[
        "independent_validator_population_sha256"
    ]
    assert diagnostic["first_mismatch"] is None
    assert diagnostic["ordered_population_match"] is True
    assert diagnostic["score_population_binding_result"] is True
    assert diagnostic["historical_r92_prediction_source_sha256"] == (
        core.sha256_file(core.R92_PREDICTIONS)
    )


def test_evidence_derived_ready_status():
    assert core.derive_ready_status({"data": True, "cuda": True}) == {
        "ready_for_full_awaiting_external_review": True, "status": "READY", "failed_gates": [],
    }
    blocked = core.derive_ready_status({"data": False, "cuda": True, "semantics": False})
    assert blocked["status"] == "BLOCKED"
    assert blocked["failed_gates"] == ["data", "semantics"]


def test_builder_status_renderer_uses_actual_evidence_state():
    inputs = {
        spec.key: {
            "classification": "VERIFIED", "passed": True,
            "evidence_refs": [f"provenance/{spec.key}.json"],
        }
        for spec in gates.DEFAULT_GATE_SPECS
    }
    facts = gates.WorkflowFacts(
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
    )
    ready_state = gates.derive_workflow_state(gates.build_gate_registry(inputs), facts)
    report = {
        "training_table_path": str(core.HISTORICAL_TRAINING_CSV),
        "training_table_sha256": core.sha256_file(core.HISTORICAL_TRAINING_CSV),
        "candidate_split_hash": {"candidate_split_manifest_sha256": "abc"},
        "r92_control": {"max_probability_difference": 0.0},
        "historical_replay": {"literal_probability_tolerance_status": "PASS"},
    }
    ready_text = builder._status_text(STUDY / "results/example", ready_state, report)
    assert "AMENDED_PREFLIGHT_STATUS=PASS_READY_FOR_CUDA_SMOKE" in ready_text
    assert "DATA_SOURCE_DECISION=HISTORICAL_R92_SNAPSHOT_VERIFIED_PRIMARY_PAIRED_ABLATION" in ready_text
    assert "CUDA_SMOKE_RUN=NOT_RUN_AWAITING_EXECUTION" in ready_text

    inputs["cuda_capability"] = {
        "classification": "VERIFIED", "passed": False,
        "evidence_refs": ["provenance/cuda_capability_probe.json"],
    }
    blocked_state = gates.derive_workflow_state(gates.build_gate_registry(inputs), facts)
    blocked_text = builder._status_text(STUDY / "results/example", blocked_state, report)
    assert "AMENDED_PREFLIGHT_STATUS=BLOCKED_BEFORE_CUDA_SMOKE" in blocked_text
    assert "CUDA_SMOKE_RUN=NOT_RUN_GLOBAL_GATES_FAILED" in blocked_text
    assert "CUDA_CAPABILITY_FAILURE" in blocked_text


def test_command_result_preserves_actual_exit_code(runtime_case):
    result = core.run_command([sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)"], cwd=runtime_case)
    assert result.returncode == 7
    assert result.stdout.strip() == "out"
    assert result.stderr.strip() == "err"
    assert result.wall_seconds >= 0


def test_code_snapshot_identity_key_convention():
    keys = {
        f"code/{path.name}" for path in CODE.glob("*.py")
    } | {f"code/{path.name}" for path in CODE.glob("*.sh")} | {
        f"tests/{path.name}" for path in (STUDY / "tests").glob("*.py")
    }
    assert "code/amendment_core.py" in keys
    assert "tests/test_ablation.py" in keys


def test_cli_preflight_dispatches_amended_builder(monkeypatch):
    called = []

    class Stub:
        @staticmethod
        def build():
            called.append("amended")
            return STUDY / "results" / "stub"

    original = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda name: Stub if name == "build_amended_preflight" else original(name))
    monkeypatch.setattr(sys, "argv", ["run_study.py", "--preflight"])
    assert run_study.main() == 0
    assert called == ["amended"]


def test_cli_full_resume_and_stability_are_forbidden(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_study.py", "--full"])
    assert run_study.main() == 3
    monkeypatch.setattr(sys, "argv", ["run_study.py", "--stability"])
    assert run_study.main() == 3
    monkeypatch.setattr(sys, "argv", ["run_study.py", "--resume", str(STUDY / "results/example")])
    assert run_study.main() == 3


def test_cli_smoke_requires_preflight_reference(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_study.py", "--smoke"])
    assert run_study.main() == 3


def test_cli_catches_amendment_scientific_blocker(monkeypatch, runtime_case):
    class Stub:
        @staticmethod
        def run_cuda_smoke(**kwargs):
            raise core.ScientificBlocker("calculated gate blocked Smoke")

    original = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: Stub if name == "amendment_core" else original(name),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_study.py", "--smoke", "--preflight-run", str(runtime_case)],
    )
    assert run_study.main() == 3


def test_run_state_transition_validation():
    identity = {"state": "PREFLIGHT_COMPLETE", "permitted_transitions": ["CUDA_SMOKE_NON_SCIENTIFIC"]}
    core.validate_run_transition(identity, "CUDA_SMOKE_NON_SCIENTIFIC")
    with pytest.raises(core.IntegrityError):
        core.validate_run_transition(identity, "PRIMARY_SCIENTIFIC")


def _synthetic_authorized_preflight(runtime_case: Path):
    preflight = runtime_case / "preflight"
    (preflight / "provenance").mkdir(parents=True)
    (preflight / "config").mkdir()
    evidence = preflight / "provenance/calculated_evidence.json"
    evidence.write_text("{}")
    inputs = {
        spec.key: {
            "classification": "VERIFIED", "passed": True,
            "evidence_refs": ["provenance/calculated_evidence.json"],
            "detail": "synthetic calculated fixture",
        }
        for spec in gates.DEFAULT_GATE_SPECS
    }
    facts = gates.WorkflowFacts(
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
    )
    registry = gates.build_gate_registry(inputs)
    state = gates.derive_workflow_state(registry, facts)
    (preflight / "provenance/evidence_gate_registry.json").write_text(
        json.dumps(builder._registry_payload(registry, state))
    )
    (preflight / "provenance/execution_ledger.json").write_text(json.dumps({
        "official_smoke_variants_completed": 0,
        "exploratory_smoke_variants_completed": 0,
        "full_scientific_run_executed": False,
        "full_variants_completed": 0,
    }))
    identity = {
        "run_id": "preflight", "run_kind": "AMENDED_PREFLIGHT",
        "state": "PREFLIGHT_COMPLETE", "split_locked": True,
        "permitted_transitions": ["CUDA_SMOKE_NON_SCIENTIFIC"],
        "code_tree": {
            path.relative_to(core.STUDY_ROOT).as_posix(): core.sha256_file(path)
            for path in [
                *sorted((core.STUDY_ROOT / "code").glob("*.py")),
                *sorted((core.STUDY_ROOT / "code").glob("*.sh")),
                *sorted((core.STUDY_ROOT / "tests").glob("*.py")),
            ]
            if path.is_file()
        },
    }
    report = {
        "global_blockers": {},
        "status": "PASS_READY_FOR_CUDA_SMOKE",
    }
    features = {
        "no_pca": {"status": "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE"},
    }
    return preflight, identity, report, features


def _synthetic_authorization_evidence(preflight: Path) -> dict:
    return {
        "registry_payload": json.loads(
            (preflight / "provenance/evidence_gate_registry.json").read_text()
        ),
        "execution_ledger": json.loads(
            (preflight / "provenance/execution_ledger.json").read_text()
        ),
        "reviewed_member_paths": {
            path.relative_to(preflight).as_posix()
            for path in preflight.rglob("*") if path.is_file()
        },
    }


def test_smoke_authorization_rederives_before_model_execution(runtime_case):
    preflight, identity, report, features = _synthetic_authorized_preflight(
        runtime_case,
    )
    state = core.authorize_cuda_smoke_reference(
        preflight_run=preflight,
        validated_identity=identity,
        preflight_report=report,
        features_by_variant=features,
        **_synthetic_authorization_evidence(preflight),
    )
    assert state.smoke_eligible

    features["no_pca"]["status"] = "INCOHERENT_STATUS"
    with pytest.raises(
        (core.IntegrityError, core.ScientificBlocker),
        match="workflow differs|NOT_AUTHORIZED",
    ):
        core.authorize_cuda_smoke_reference(
            preflight_run=preflight,
            validated_identity=identity,
            preflight_report=report,
            features_by_variant=features,
            **_synthetic_authorization_evidence(preflight),
        )


def test_smoke_authorization_requires_explicit_transition(runtime_case):
    preflight, identity, report, features = _synthetic_authorized_preflight(
        runtime_case,
    )
    identity["permitted_transitions"] = []
    with pytest.raises(core.IntegrityError, match="not permitted"):
        core.authorize_cuda_smoke_reference(
            preflight_run=preflight,
            validated_identity=identity,
            preflight_report=report,
            features_by_variant=features,
            **_synthetic_authorization_evidence(preflight),
        )


def test_write_guard_denies_prior_immutable_artifacts():
    with pytest.raises(core.IntegrityError):
        core.ensure_write_path(core.PRIOR_RUN / "RUN_STATUS.txt")
    with pytest.raises(core.IntegrityError):
        core.ensure_write_path(core.PRIOR_ZIP)


def test_packager_rejects_wrong_run_kind(runtime_case):
    run = runtime_case / "wrong_kind"
    (run / "config").mkdir(parents=True)
    (run / "config/run_identity.lock.json").write_text(json.dumps({"run_kind": "PRIMARY_SCIENTIFIC", "state": "COMPLETE"}))
    with pytest.raises(core.IntegrityError, match="Unsupported package Run kind"):
        core.package_amended_run(run)


def test_packager_rejects_correct_kind_incomplete_run_before_manifest(runtime_case):
    run = runtime_case / "incomplete_amended_preflight"
    (run / "config").mkdir(parents=True)
    identity = {
        "run_id": run.name,
        "run_kind": "AMENDED_PREFLIGHT",
        "state": "BLOCKED_PREFLIGHT_COMPLETE",
    }
    (run / "config/run_identity.lock.json").write_text(json.dumps(identity))
    (run / "RUN_STATUS.txt").write_text(
        "RUN_KIND=AMENDED_PREFLIGHT\nRUN_STATE=BLOCKED_PREFLIGHT_COMPLETE\n"
    )

    with pytest.raises(core.IntegrityError, match="Run completeness validation failed") as caught:
        core.package_amended_run(run)

    assert "docs/PROTOCOL_AMENDMENT_01_TASK_SPEC.md" in str(caught.value)
    assert not (run / "OUTPUT_MANIFEST_FINAL.tsv").exists()
    assert not (run / "BUNDLE_MANIFEST.tsv").exists()


def test_packager_existing_destination_causes_no_manifest_mutation(runtime_case, monkeypatch):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    run = runtime_case / "results/already_sealed"
    run.mkdir(parents=True)
    destination = runtime_case / "results/already_sealed_review_bundle.zip"
    destination.write_bytes(b"immutable")
    with pytest.raises(core.IntegrityError, match="Refusing to overwrite"):
        core.package_amended_run(run)
    assert destination.read_bytes() == b"immutable"
    assert not (run / "OUTPUT_MANIFEST_FINAL.tsv").exists()
    assert not (run / "BUNDLE_MANIFEST.tsv").exists()


def test_package_completeness_contract_accepts_semantic_preflight(
    runtime_case, monkeypatch,
):
    run, identity = _valid_preflight_semantic_fixture(
        runtime_case, monkeypatch,
    )
    required = core.PACKAGE_REQUIRED_FILES[(identity["run_kind"], identity["state"])]

    result = core.validate_package_completeness(run, identity)

    assert result["required_file_count"] == len(required)
    assert result["required_files_status"] == "PASS"
    assert result["status_identity_match"] == "PASS"
    assert result["preflight_semantic_completeness"]["status"] == "PASS"


def test_preflight_package_contract_rejects_nonempty_placeholder_evidence(
    runtime_case,
):
    run = runtime_case / "fabricated_preflight"
    identity = {
        "run_id": run.name,
        "run_kind": "AMENDED_PREFLIGHT",
        "state": "PREFLIGHT_COMPLETE",
    }
    required = core.PACKAGE_REQUIRED_FILES[(identity["run_kind"], identity["state"])]
    for relative in required:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n")
    (run / "config/run_identity.lock.json").write_text(json.dumps(identity))
    (run / "RUN_STATUS.txt").write_text(
        "RUN_KIND=AMENDED_PREFLIGHT\nRUN_STATE=PREFLIGHT_COMPLETE\n"
    )

    with pytest.raises(core.IntegrityError, match="semantic evidence is unreadable"):
        core.validate_package_completeness(run, identity)


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("gate", "gate_registry_schema_and_calculated_tokens"),
        ("gate_ref", "gate_registry_schema_and_calculated_tokens"),
        ("blocker_detail", "report_matches_derived_workflow"),
        ("report", "report_matches_derived_workflow"),
        ("report_ledger", "report_duplicate_evidence_payloads_are_bound"),
        ("split", "split_manifest_recomputed_and_bound"),
        ("split_card", "split_manifest_recomputed_and_bound"),
        ("source", "source_inventory_pre_post_and_identity_bound"),
        ("source_coordinated", "source_inventory_pre_post_and_identity_bound"),
        ("current_path", "source_tables_reopened_and_split_rows_bound"),
        ("feature_manifest", "feature_manifest_matches_verified_ontology_exactly"),
        ("dependency_graph", "feature_dependency_graph_and_variants_rederived_exactly"),
        ("variant_definition", "feature_dependency_graph_and_variants_rederived_exactly"),
        ("gate_formula_path", "gate_reconstruction_machine_evidence_bound"),
        ("gate_reconstruction", "gate_reconstruction_machine_evidence_bound"),
        ("audit_gate_coverage", "gate_reconstruction_machine_evidence_bound"),
        ("raw_diagnostic", "no_pca_diagnostic_and_variant_bound"),
        ("raw_cache_path", "no_pca_diagnostic_and_variant_bound"),
        ("raw_dimensions", "no_pca_diagnostic_and_variant_bound"),
        ("raw_encoder", "no_pca_diagnostic_and_variant_bound"),
        ("raw_crop", "no_pca_diagnostic_and_variant_bound"),
        ("raw_tile_root", "no_pca_diagnostic_and_variant_bound"),
        ("raw_tiled_coco", "no_pca_diagnostic_and_variant_bound"),
        ("raw_audit", "no_pca_diagnostic_and_variant_bound"),
        ("raw_detection_inventory", "no_pca_diagnostic_and_variant_bound"),
        ("raw_prerequisite", "no_pca_diagnostic_and_variant_bound"),
        ("raw_join", "no_pca_diagnostic_and_variant_bound"),
        ("raw_inventory", "no_pca_diagnostic_and_variant_bound"),
        ("pca_inventory", "historical_snapshot_and_pca_gate_bound"),
        ("pca_payload", "historical_snapshot_and_pca_gate_bound"),
        ("embed_semantics", "feature_semantics_gate_bound"),
        ("harmonized_layer", "feature_semantics_gate_bound"),
        ("review_weights", "review_weight_and_augmentation_parity_bound"),
        ("r92_control", "r92_external_control_bound"),
        ("cuda_probe", "cuda_capability_gate_bound"),
        ("prior_inventory", "prior_immutable_evidence_bound"),
        ("replay_identity", "historical_replay_reopened_and_bound"),
        ("replay_probability_source", "historical_replay_reopened_and_bound"),
        ("replay_metric", "historical_split_and_metric_gates_bound"),
    ],
)
def test_preflight_semantic_contract_rejects_tampered_evidence(
    runtime_case, monkeypatch, mutation, failed_check,
):
    run, identity = _valid_preflight_semantic_fixture(
        runtime_case, monkeypatch, run_name=f"tampered_{mutation}",
    )
    if mutation == "gate":
        path = run / "provenance/evidence_gate_registry.json"
        payload = json.loads(path.read_text())
        payload["gates"]["cuda_capability"]["passed"] = False
        path.write_text(json.dumps(payload))
    elif mutation == "gate_ref":
        path = run / "provenance/evidence_gate_registry.json"
        payload = json.loads(path.read_text())
        payload["gates"]["cuda_capability"]["evidence_refs"] = [
            "provenance/PREFLIGHT_REPORT.json"
        ]
        path.write_text(json.dumps(payload))
    elif mutation == "blocker_detail":
        path = run / "provenance/evidence_gate_registry.json"
        payload = json.loads(path.read_text())
        payload["gates"]["historical_prediction_replay"]["detail"] = (
            "tampered blocker reason"
        )
        gate_inputs = {
            key: {
                "classification": value["classification"],
                "passed": value["passed"],
                "evidence_refs": value["evidence_refs"],
                "detail": value["detail"],
            }
            for key, value in payload["gates"].items()
        }
        registry = gates.build_gate_registry(gate_inputs)
        state = gates.derive_workflow_state(registry, gates.WorkflowFacts(
            phase="AMENDED_PREFLIGHT",
            no_pca_variant_status=(
                "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE"
            ),
            no_pca_variant_runnable=False,
        ))
        payload["derived_workflow"] = core._preflight_derived_payload(state)
        path.write_text(json.dumps(payload))
    elif mutation == "report":
        path = run / "provenance/PREFLIGHT_REPORT.json"
        payload = json.loads(path.read_text())
        payload["status"] = "PASS_READY_FOR_CUDA_SMOKE"
        path.write_text(json.dumps(payload))
    elif mutation == "report_ledger":
        path = run / "provenance/PREFLIGHT_REPORT.json"
        payload = json.loads(path.read_text())
        payload["execution_ledger"]["status"] = "FAIL"
        path.write_text(json.dumps(payload))
    elif mutation == "split":
        path = run / "splits/row_split_manifest.csv.gz"
        frame = pd.read_csv(path)
        frame.loc[0, "split_order"] = 9
        frame.to_csv(path, index=False)
    elif mutation == "split_card":
        path = run / "splits/row_split_manifest.csv.gz"
        frame = pd.read_csv(path)
        frame.loc[0, "card_id"] = "tampered_card"
        frame.to_csv(path, index=False)
    elif mutation == "source":
        path = run / "provenance/source_input_hashes_post.tsv"
        frame = pd.read_csv(path, sep="\t")
        frame.loc[0, "sha256"] = "0" * 64
        frame.to_csv(path, sep="\t", index=False)
    elif mutation == "current_path":
        path = run / "provenance/current_snapshot_validation.json"
        payload = json.loads(path.read_text())
        payload["path"] = str(runtime_case / "source/unreviewed_current.csv")
        path.write_text(json.dumps(payload))
    elif mutation == "feature_manifest":
        path = run / "config/feature_manifest.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "primary_group"] = "tampered_group"
        frame.to_csv(path, index=False)
    elif mutation == "dependency_graph":
        path = run / "config/feature_dependency_graph.json"
        payload = json.loads(path.read_text())
        payload["dependencies"]["g_quality"] = ["color"]
        path.write_text(json.dumps(payload))
    elif mutation == "variant_definition":
        path = run / "config/variant_feature_sets.json"
        payload = json.loads(path.read_text())
        payload["manual_only"]["feature_count"] += 1
        path.write_text(json.dumps(payload))
    elif mutation == "gate_formula_path":
        path = run / "provenance/gate_formula_manifest.json"
        payload = json.loads(path.read_text())
        payload["implementation_path"] = "/tmp/unreviewed.py"
        path.write_text(json.dumps(payload))
    elif mutation == "gate_reconstruction":
        path = run / "provenance/gate_reconstruction_validation.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "max_absolute_error"] = 0.25
        frame.to_csv(path, index=False)
    elif mutation == "audit_gate_coverage":
        path = run / "provenance/audit_gate_upstream_coverage.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "finite_rows"] = 0
        frame.to_csv(path, index=False)
    elif mutation == "raw_diagnostic":
        path = run / "provenance/raw_embedding_mapping_diagnostic.json"
        payload = json.loads(path.read_text())
        payload["historical_training_mapping"]["raw_cache"]["sha256"] = "0" * 64
        path.write_text(json.dumps(payload))
    elif mutation == "raw_cache_path":
        path = run / "provenance/raw_embedding_mapping_diagnostic.json"
        payload = json.loads(path.read_text())
        payload["historical_training_mapping"]["raw_cache"]["path"] = str(
            core.GATE_SOURCE
        )
        path.write_text(json.dumps(payload))
    elif mutation == "raw_dimensions":
        path = run / "provenance/raw_embedding_mapping_diagnostic.json"
        payload = json.loads(path.read_text())
        payload["required_raw_dimensions"] = 1
        path.write_text(json.dumps(payload))
    elif mutation == "raw_encoder":
        path = run / "provenance/raw_embedding_mapping_diagnostic.json"
        payload = json.loads(path.read_text())
        payload["encoder"]["available_weights_artifact"]["sha256"] = "0" * 64
        path.write_text(json.dumps(payload))
    elif mutation == "raw_crop":
        path = run / "provenance/raw_embedding_mapping_diagnostic.json"
        payload = json.loads(path.read_text())
        payload["training_crop_mask_availability"][
            "feature_base_candidates_with_segmentation"
        ] = 3
        path.write_text(json.dumps(payload))
    elif mutation == "raw_tile_root":
        path = run / "provenance/raw_embedding_mapping_diagnostic.json"
        payload = json.loads(path.read_text())
        payload["training_crop_mask_availability"]["tile_root"] = str(
            runtime_case / "unreviewed_tiles"
        )
        path.write_text(json.dumps(payload))
    elif mutation == "raw_tiled_coco":
        path = run / "provenance/raw_embedding_mapping_diagnostic.json"
        payload = json.loads(path.read_text())
        payload["training_crop_mask_availability"]["tiled_coco_path"] = str(
            core.TASK_SPEC
        )
        path.write_text(json.dumps(payload))
    elif mutation == "raw_audit":
        path = run / "provenance/raw_embedding_mapping_diagnostic.json"
        payload = json.loads(path.read_text())
        payload["audit_mapping"]["detection_rows"] += 1
        path.write_text(json.dumps(payload))
    elif mutation == "raw_detection_inventory":
        detection = (
            core.PROJECT_ROOT
            / "Shared/maskout_tile/test_set/IMG_0001/new/detections.csv"
        )
        detection.parent.mkdir(parents=True)
        pd.DataFrame([{
            "image": "audit.jpg", "id": 1, "rle": "", "poly": "[]",
        }]).to_csv(detection, index=False)
    elif mutation == "raw_prerequisite":
        path = run / "provenance/raw_embedding_mapping_diagnostic.json"
        payload = json.loads(path.read_text())
        payload["exact_mapping_prerequisites"]["historical_training"] = True
        path.write_text(json.dumps(payload))
    elif mutation == "raw_join":
        path = run / "provenance/raw_embedding_join_diagnostic.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "candidate_rows"] += 1
        frame.to_csv(path, index=False)
    elif mutation == "raw_inventory":
        path = run / "provenance/raw_embedding_source_inventory.tsv"
        frame = pd.read_csv(path, sep="\t").iloc[0:0]
        frame.to_csv(path, sep="\t", index=False)
    elif mutation == "pca_inventory":
        path = run / "provenance/pca_artifact_inventory.tsv"
        frame = pd.read_csv(path, sep="\t")
        frame.loc[0, "sha256"] = "0" * 64
        frame.to_csv(path, sep="\t", index=False)
    elif mutation == "pca_payload":
        path = run / "provenance/historical_snapshot_validation.json"
        payload = json.loads(path.read_text())
        payload["pca_prototype_compatibility"]["compatible_single_basis"] = False
        path.write_text(json.dumps(payload))
    elif mutation == "embed_semantics":
        path = run / "provenance/embed_sim_semantics.json"
        payload = json.loads(path.read_text())
        payload["audit_transformation"] = "unreviewed transformation"
        path.write_text(json.dumps(payload))
    elif mutation == "harmonized_layer":
        path = run / "provenance/harmonized_audit_feature_layer.csv.gz"
        frame = pd.read_csv(path)
        frame.loc[0, "embed_sim"] = 0.123
        frame.to_csv(path, index=False)
    elif mutation == "review_weights":
        path = run / "provenance/review_weight_validation.json"
        payload = json.loads(path.read_text())
        payload["historical_legacy_geometry"]["post_augmentation_rows"] += 1
        path.write_text(json.dumps(payload))
    elif mutation == "r92_control":
        path = run / "metrics/r92_control.json"
        payload = json.loads(path.read_text())
        payload["max_probability_difference"] = 0.5
        path.write_text(json.dumps(payload))
    elif mutation == "cuda_probe":
        path = run / "provenance/cuda_capability_probe.json"
        payload = json.loads(path.read_text())
        payload["use_cuda_compiled"] = False
        path.write_text(json.dumps(payload))
    elif mutation == "prior_inventory":
        path = run / "provenance/prior_immutable_verification_post.json"
        payload = json.loads(path.read_text())
        payload["prior"]["manifest_status"] = "FAIL"
        path.write_text(json.dumps(payload))
    elif mutation == "replay_identity":
        path = run / "provenance/historical_prediction_replay.csv.gz"
        original_id = pd.read_csv(
            run / "splits/row_split_manifest.csv.gz"
        ).query("split == 'validation'").sort_values("split_order").iloc[0][
            "stable_candidate_id"
        ]
        text = gzip.decompress(path.read_bytes()).decode()
        assert original_id in text
        path.write_bytes(gzip.compress(
            text.replace(original_id, "train_" + "0" * 64, 1).encode(), mtime=0,
        ))
        schema_path = run / "provenance/historical_prediction_replay_schema.json"
        schema = json.loads(schema_path.read_text())
        schema["comparison_sha256"] = core.sha256_file(path)
        summary, valid = core._historical_replay_reopen_summary(path)
        assert valid
        schema["serialized_reopen_verification"].update({
            "artifact_sha256": core.sha256_file(path),
            "summary": summary,
        })
        schema_path.write_text(json.dumps(schema))
        historical_path = run / "provenance/historical_snapshot_validation.json"
        historical_payload = json.loads(historical_path.read_text())
        historical_payload["serialized_replay_artifact"]["reopen_verification"] = (
            schema["serialized_reopen_verification"]
        )
        historical_path.write_text(json.dumps(historical_payload))
    elif mutation == "replay_probability_source":
        path = run / "provenance/historical_prediction_replay.csv.gz"
        frame = core._historical_replay_reopen_frame(path)
        tampered_literal = "0.1000000001"
        tampered_decimal = float(tampered_literal)
        replayed_float32 = np.asarray(
            [
                int(
                    str(frame.loc[0, "replayed_probability_float32_bits_hex"]),
                    16,
                )
            ],
            dtype=np.uint32,
        ).view(np.float32)[0]
        direct_error = abs(tampered_decimal - float(replayed_float32))
        assert direct_error > 1e-12
        assert np.float32(tampered_decimal).view(np.uint32) == (
            replayed_float32.view(np.uint32)
        )
        frame.loc[0, "historical_saved_probability_literal"] = tampered_literal
        frame.loc[0, "historical_saved_probability_decimal"] = tampered_decimal
        frame.loc[0, "historical_saved_probability_float64_hex"] = (
            tampered_decimal.hex()
        )
        frame.loc[0, "absolute_difference_direct_decimal"] = direct_error
        frame.loc[0, "absolute_difference_direct_decimal_float64_hex"] = (
            direct_error.hex()
        )
        frame.to_csv(path, index=False, float_format="%.17g")

        summary, valid = core._historical_replay_reopen_summary(path)
        assert valid
        schema_path = run / "provenance/historical_prediction_replay_schema.json"
        schema = json.loads(schema_path.read_text())
        schema["comparison_sha256"] = core.sha256_file(path)
        schema["serialized_reopen_verification"].update({
            "artifact_sha256": core.sha256_file(path),
            "summary": summary,
        })
        schema_path.write_text(json.dumps(schema))

        historical_path = run / "provenance/historical_snapshot_validation.json"
        historical_payload = json.loads(historical_path.read_text())
        historical_payload.update({
            "direct_decimal_probability_max_absolute_difference": summary[
                "direct_decimal_max_absolute_difference"
            ],
            "direct_decimal_rows_exceeding_1e_12": summary[
                "direct_decimal_mismatch_count_gt_1e_12"
            ],
            "source_dtype_float32_probability_max_absolute_difference": summary[
                "source_dtype_float32_max_absolute_difference"
            ],
        })
        historical_payload["serialized_replay_artifact"]["reopen_verification"] = (
            schema["serialized_reopen_verification"]
        )
        historical_path.write_text(json.dumps(historical_payload))

        report_path = run / "provenance/PREFLIGHT_REPORT.json"
        report = json.loads(report_path.read_text())
        report["historical_replay"].update({
            "direct_max_difference": summary[
                "direct_decimal_max_absolute_difference"
            ],
            "source_dtype_max_difference": summary[
                "source_dtype_float32_max_absolute_difference"
            ],
        })
        report_path.write_text(json.dumps(report))
    elif mutation == "replay_metric":
        path = run / "provenance/historical_snapshot_validation.json"
        payload = json.loads(path.read_text())
        payload["metrics"]["average_precision"] = 0.5
        payload["metrics"]["sidecar_average_precision"] = 0.5
        path.write_text(json.dumps(payload))
    elif mutation == "source_coordinated":
        report_path = run / "provenance/PREFLIGHT_REPORT.json"
        report = json.loads(report_path.read_text())
        source_path = Path(report["training_table_path"])
        source_path.write_bytes(source_path.read_bytes() + b"tamper\n")
        replacement_sha = core.sha256_file(source_path)
        for relative in (
            "provenance/source_input_hashes_pre.tsv",
            "provenance/source_input_hashes_post.tsv",
        ):
            path = run / relative
            frame = pd.read_csv(path, sep="\t")
            mask = frame.path.astype(str) == str(source_path)
            frame.loc[mask, "size_bytes"] = source_path.stat().st_size
            frame.loc[mask, "sha256"] = replacement_sha
            frame.to_csv(path, sep="\t", index=False)
        identity_path = run / "config/run_identity.lock.json"
        changed_identity = json.loads(identity_path.read_text())
        changed_identity["source_hashes"][str(source_path)] = replacement_sha
        identity_path.write_text(json.dumps(changed_identity))
        report["training_table_sha256"] = replacement_sha
        report_path.write_text(json.dumps(report))
        for relative in (
            "provenance/historical_snapshot_validation.json",
            "provenance/current_snapshot_validation.json",
        ):
            path = run / relative
            payload = json.loads(path.read_text())
            payload["sha256"] = replacement_sha
            path.write_text(json.dumps(payload))
    else:
        raise AssertionError(f"Unhandled mutation: {mutation}")

    with pytest.raises(core.IntegrityError, match=failed_check):
        core.validate_package_completeness(run, identity)


def test_preflight_semantic_contract_rejects_self_asserted_staging_record(
    runtime_case, monkeypatch,
):
    run, identity = _valid_preflight_semantic_fixture(
        runtime_case, monkeypatch, run_name="fabricated_staging",
    )
    registry_path = run / "provenance/evidence_gate_registry.json"
    payload = json.loads(registry_path.read_text())
    payload["gates"]["output_manifest_zip_verification"].update({
        "classification": "VERIFIED", "passed": True,
        "accepted": True, "status": "PASS",
    })
    gate_inputs = {
        key: {
            "classification": value["classification"],
            "passed": value["passed"],
            "evidence_refs": value["evidence_refs"],
            "detail": value.get("detail", ""),
        }
        for key, value in payload["gates"].items()
    }
    registry = gates.build_gate_registry(gate_inputs)
    facts = gates.WorkflowFacts(
        phase="AMENDED_PREFLIGHT",
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
    )
    state = gates.derive_workflow_state(registry, facts)
    payload["derived_workflow"] = core._preflight_derived_payload(state)
    registry_path.write_text(json.dumps(payload))
    (run / "provenance/package_staging_verification.json").write_text(json.dumps({
        "crc_status": "PASS",
        "independent_reopen_member_verification": "PASS",
        "verification_failures": [],
        "staging_bundle_lifecycle": (
            "SUPERSEDED_BY_FINAL_BUNDLE_AND_DELETED_AFTER_FINAL_REOPEN"
        ),
        "publication_method": "VERIFIED_TEMPORARY_THEN_ATOMIC_NO_CLOBBER_LINK",
        "bundle_sha256": "1" * 64,
        "bundle_size_bytes": 1,
        "member_count": 1,
        "uncompressed_size_bytes": 1,
        "staging_bundle_temporary_path_after_rename": str(
            run.parent / f"{run.name}_review_bundle_staging_verified.zip"
        ),
        "package_completeness": {
            "preflight_semantic_completeness": {
                "status": "PASS", "phase": "STAGING",
            },
        },
    }))
    report_path = run / "provenance/PREFLIGHT_REPORT.json"
    report = json.loads(report_path.read_text())
    report["output_manifest_zip_verification"] = "PASS"
    report_path.write_text(json.dumps(report))
    identity["state"] = state.run_state
    (run / "config/run_identity.lock.json").write_text(json.dumps(identity))
    (run / "RUN_STATUS.txt").write_text(gates.render_status_lines(
        state,
        extra_lines={
            "TRAINING_TABLE_PATH": report["training_table_path"],
            "TRAINING_TABLE_SHA256": report["training_table_sha256"],
            "HISTORICAL_LITERAL_REPLAY_TOLERANCE_STATUS": "FAIL",
        },
    ))

    with pytest.raises(
        core.IntegrityError, match="package_gate_bound_to_staging_verification",
    ):
        core.validate_package_completeness(run, identity)


def test_package_completeness_rejects_incompatible_kind_state(runtime_case):
    identity = {"run_kind": "AMENDED_PREFLIGHT", "state": "SMOKE_COMPLETE"}
    with pytest.raises(core.IntegrityError, match="Unsupported package Run kind/state contract"):
        core.validate_package_completeness(runtime_case, identity)


def test_package_completeness_rejects_identity_run_id_directory_mismatch(
    runtime_case,
):
    run = runtime_case / "renamed_run"
    identity = {
        "run_id": "different_run_id",
        "run_kind": "AMENDED_PREFLIGHT",
        "state": "PREFLIGHT_COMPLETE",
    }
    with pytest.raises(core.IntegrityError, match="identity/name mismatch"):
        core.validate_package_completeness(run, identity)


def test_smoke_package_contract_rejects_nonempty_placeholder_evidence(runtime_case):
    run = runtime_case / "fabricated_smoke"
    identity = {
        "run_id": run.name,
        "run_kind": "CUDA_SMOKE_NON_SCIENTIFIC",
        "state": "SMOKE_COMPLETE",
        "preflight_run_id": "fake",
        "candidate_split_hash": "0" * 64,
        "full_authorized": False,
        "stability_authorized": False,
    }
    required = core.PACKAGE_REQUIRED_FILES[(identity["run_kind"], identity["state"])]
    for relative in required:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n")
    (run / "config/run_identity.lock.json").write_text(json.dumps(identity))
    (run / "RUN_STATUS.txt").write_text(
        "RUN_KIND=CUDA_SMOKE_NON_SCIENTIFIC\nRUN_STATE=SMOKE_COMPLETE\n"
    )
    with pytest.raises(core.IntegrityError, match="semantic evidence is unreadable"):
        core.validate_package_completeness(run, identity)


def _valid_smoke_semantic_fixture(runtime_case: Path, monkeypatch):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    (runtime_case / "results").mkdir(exist_ok=True)
    run = runtime_case / "results/valid_smoke"
    identity = {
        "run_id": "valid_smoke",
        "run_kind": "CUDA_SMOKE_NON_SCIENTIFIC",
        "state": "SMOKE_COMPLETE",
        "preflight_run_id": "synthetic_preflight",
        "candidate_split_hash": "1" * 64,
        "permitted_transitions": ["PACKAGE_REVIEW_EVIDENCE"],
        "full_authorized": False,
        "stability_authorized": False,
    }
    required = core.PACKAGE_REQUIRED_FILES[(identity["run_kind"], identity["state"])]
    for relative in required:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence\n")

    variants = core.build_variant_feature_sets()
    (run / "config/variant_feature_sets.json").write_text(json.dumps(variants))
    model_parameter_lock = core.r92_model_parameter_lock()
    (run / "config/training_configuration.lock.json").write_text(json.dumps({
        "model": model_parameter_lock,
    }))
    runnable = core.runnable_official_variants(variants)
    completed = [*runnable, core.EXPLORATORY_VARIANT]
    split = pd.DataFrame({
        "stable_candidate_id": [f"training:{index}" for index in range(8)]
        + [f"validation:{index}" for index in range(4)],
        "split": ["training"] * 8 + ["validation"] * 4,
        "eligible": [True] * 12,
        "split_order": list(range(8)) + list(range(4)),
        "card_id": ["train_card"] * 8 + ["validation_card"] * 4,
        "label": [0, 1] * 4 + [0, 1] * 2,
    })
    split.to_csv(run / "splits/row_split_manifest.csv.gz", index=False)
    _, _, subset = core.deterministic_smoke_subset(
        split, train_cap=8, validation_cap=4,
    )
    (run / "config/smoke_subset.json").write_text(json.dumps(subset))
    model_root = runtime_case / ".runtime/models/valid_smoke"
    model_root.mkdir(parents=True)
    details = {}
    table_rows = []
    for index, name in enumerate(completed):
        specification = variants[name]
        feature_count = len(specification["retained_features"])
        booster_bytes, configuration = _valid_test_booster_artifacts(feature_count)
        configuration = json.loads(json.dumps(configuration))
        booster = core.xgb.Booster()
        booster.load_model(bytearray(booster_bytes))
        configuration["learner"]["generic_param"]["device"] = "cuda:0"
        gpu_identity = {
            "requested_device": "cuda",
            "cuda_visible_devices": "0",
            "nvidia_visible_devices": "0",
            "nvidia_smi_query_returncode": 0,
            "nvidia_smi_query_stderr": "",
            "visible_gpu_count_reported_by_nvidia_smi": 1,
            "devices": [{
                "physical_index": "0",
                "gpu_name": "Synthetic test GPU",
                "driver_version": "test-driver",
            }],
            "xgboost_cuda_compiled": True,
        }
        embedded_model_attributes = {
            "amendment_classification": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
            "amendment_device": "cuda",
            "amendment_variant": name,
            "amendment_run_id": run.name,
            "booster_configuration_sha256": core.sha256_bytes(
                core.canonical_json(configuration)
            ),
            "feature_list_sha256": specification["feature_list_sha256"],
            "gpu_identity_sha256": core.sha256_bytes(
                core.canonical_json(gpu_identity)
            ),
            "model_parameter_lock_sha256": model_parameter_lock[
                "smoke_classifier_parameters_sha256"
            ],
        }
        booster.set_attr(**embedded_model_attributes)
        model_path = model_root / f"{name}.ubj"
        model_path.write_bytes(bytes(booster.save_raw(raw_format="ubj")))
        average_precision = 0.7 + index / 1000
        roc_auc = 0.8 + index / 1000
        stage_rows = [8, 12, 24, 36, 36, 36 if name == "no_safe_smote" else 40]
        stages = []
        for stage_name, rows_at_stage in zip(core._SMOKE_STAGE_SEQUENCE, stage_rows):
            stage = {
                "stage": stage_name,
                "rows": rows_at_stage,
                "features": feature_count,
                "negative": rows_at_stage // 2,
                "positive": rows_at_stage - rows_at_stage // 2,
            }
            for hash_name in (
                "X_sha256", "y_sha256", "weight_sha256",
                "source_lineage_sha256", "row_order_sha256",
            ):
                stage[hash_name] = hashlib.sha256(
                    f"{name}:{stage_name}:{hash_name}".encode()
                ).hexdigest()
            stages.append(stage)
        stages[-1]["safe_smote_enabled"] = name != "no_safe_smote"
        stages[-1]["safe_smote_synthetic_rows"] = stage_rows[-1] - stage_rows[-2]
        stages[-2].update({
            "shuffle_applied": False,
            "rule": "NO_POST_AUGMENTATION_SHUFFLE",
            "reason": "Verified historical pre-augmentation shuffle only.",
        })
        for key in (
            "rows", "features", "negative", "positive", "X_sha256",
            "y_sha256", "weight_sha256", "source_lineage_sha256",
            "row_order_sha256",
        ):
            stages[-2][key] = stages[-3][key]
        details[name] = {
            "classification": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
            "feature_count": feature_count,
            "feature_list_sha256": specification["feature_list_sha256"],
            "stages": stages,
            "metrics_development_only": {
                "average_precision": average_precision,
                "roc_auc": roc_auc,
            },
            "booster_configuration": configuration,
            "requested_model_parameters": model_parameter_lock[
                "smoke_classifier_parameters"
            ],
            "model_parameter_lock_sha256": model_parameter_lock[
                "smoke_classifier_parameters_sha256"
            ],
            "warnings": [],
            "resource_measurement_scope": "END_TO_END_VARIANT",
            "wall_seconds": 0.1,
            "cpu_seconds": 0.05,
            "fit_wall_seconds": 0.05,
            "fit_cpu_seconds": 0.025,
            "peak_ram_bytes": 1024,
            "gpu_identity_and_visibility": gpu_identity,
            "baseline_process_gpu_memory_mib": 0.0,
            "peak_gpu_memory_mib": 1.0,
            "peak_gpu_memory_delta_mib": 1.0,
            "embedded_model_attributes": embedded_model_attributes,
            "model_sha256": core.sha256_file(model_path),
            "save_reload_max_probability_difference": 0.0,
            "external_audit_scored": False,
        }
        table_rows.append({
            "variant": name,
            "status": "PASS",
            "feature_count": feature_count,
            "average_precision": average_precision,
            "roc_auc": roc_auc,
            "wall_seconds": 0.1,
            "cpu_seconds": 0.05,
            "peak_ram_bytes": 1024,
            "baseline_process_gpu_memory_mib": 0.0,
            "peak_gpu_memory_mib": 1.0,
            "peak_gpu_memory_delta_mib": 1.0,
        })
    def stage_for(variant, stage_name):
        return next(
            stage for stage in details[variant]["stages"]
            if stage["stage"] == stage_name
        )

    full_post = stage_for("full_new_reference", "post_safe_smote")
    no_weight_post = stage_for(
        "no_review_aware_training_weights", "post_safe_smote",
    )
    for key in (
        "X_sha256", "y_sha256", "source_lineage_sha256", "row_order_sha256",
    ):
        no_weight_post[key] = full_post[key]
    full_pre = stage_for("full_new_reference", "post_shuffle")
    no_smote_pre = stage_for("no_safe_smote", "post_shuffle")
    no_smote_jitter = stage_for(
        "no_safe_smote", "post_jitter_pre_smote",
    )
    for key in (
        "X_sha256", "y_sha256", "weight_sha256", "source_lineage_sha256",
        "row_order_sha256",
    ):
        no_smote_pre[key] = full_pre[key]
        no_smote_jitter[key] = full_pre[key]
    pd.DataFrame(table_rows).to_csv(run / "tables/SMOKE_REPORT.csv", index=False)
    model = core._package_smoke_models(model_root, "valid_smoke")
    (run / "provenance/smoke_model_bundle_verification.json").write_text(
        json.dumps(model)
    )
    command_line = "run_study.py --smoke --preflight-run synthetic_preflight"
    ledger = core.smoke_execution_ledger(
        run, command_line=command_line, runnable_official=runnable,
    )
    (run / "provenance/execution_ledger.json").write_text(json.dumps(ledger))

    no_pca_status = variants["no_pca"]["status"]
    facts = gates.WorkflowFacts(
        phase="CUDA_SMOKE",
        official_smoke_variants_completed=len(runnable),
        exploratory_smoke_variants_completed=1,
        no_pca_variant_status=no_pca_status,
        no_pca_variant_runnable=False,
    )
    gate_inputs = {
        spec.key: {
            "classification": "VERIFIED", "passed": True,
            "evidence_refs": ["config/preflight_reference.json"],
            "detail": "synthetic semantic fixture",
        }
        for spec in gates.DEFAULT_GATE_SPECS
    }
    gate_inputs["output_manifest_zip_verification"] = {
        "classification": "BLOCKED", "passed": None,
        "evidence_refs": ["provenance/package_staging_verification.json"],
        "detail": "pending staging",
    }
    registry = gates.build_gate_registry(gate_inputs)
    state = gates.derive_workflow_state(registry, facts)
    (run / "provenance/evidence_gate_registry.json").write_text(json.dumps(
        core._smoke_registry_payload(registry, state)
    ))
    (run / "metrics/SMOKE_REPORT.json").write_text(json.dumps({
        "classification": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
        "state": state.run_state,
        "official_variants_completed": len(runnable),
        "official_variants_expected": len(runnable),
        "exploratory_variants_completed": 1,
        "true_no_pca_status": no_pca_status,
        "details": details,
        "parity": core.smoke_resampling_parity(details),
        "smoke_models_bundle": model,
        "external_audit_scored": False,
    }))
    (run / "config/run_identity.lock.json").write_text(json.dumps(identity))
    (run / "RUN_STATUS.txt").write_text(
        gates.render_status_lines(state, extra_lines={
            "SMOKE_MODELS_BUNDLE_PATH": model["portable_relative_to_run"],
            "SMOKE_MODELS_BUNDLE_SHA256": model["sha256"],
            "SMOKE_MODELS_BUNDLE_RUN_ID": model["run_id"],
            "CLASSIFICATION": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
            "EXTERNAL_AUDIT_SCORED": "NO",
        })
    )
    core._write_smoke_blockers(run, state)
    (run / "NON_SCIENTIFIC_DIAGNOSTIC_ONLY.txt").write_text(
        "Non-scientific diagnostic evidence only.\n"
    )

    copied = {
        destination: core.sha256_file(run / destination)
        for destination in core.smoke_copied_evidence_mapping().values()
        if (run / destination).is_file()
    }
    preflight_zip = runtime_case / "results/synthetic_preflight_review_bundle.zip"
    mapping = core.smoke_copied_evidence_mapping()
    source_payloads = {
        source: (run / destination).read_bytes()
        for source, destination in mapping.items()
        if source not in {
            "__PREFLIGHT_REVIEW_BUNDLE__",
            "OUTPUT_MANIFEST_FINAL.tsv",
            "BUNDLE_MANIFEST.tsv",
        }
    }
    output_manifest = pd.DataFrame([{
        "relative_path": source,
        "size_bytes": len(data),
        "sha256": core.sha256_bytes(data),
        "artifact_role": source.split("/", 1)[0],
        "include_in_review_bundle": True,
        "include_in_models_bundle": False,
    } for source, data in sorted(source_payloads.items())])
    output_bytes = core._frame_bytes(output_manifest, sep="\t")
    bundle_manifest = pd.concat([
        output_manifest[["relative_path", "size_bytes", "sha256"]],
        pd.DataFrame([{
            "relative_path": "OUTPUT_MANIFEST_FINAL.tsv",
            "size_bytes": len(output_bytes),
            "sha256": core.sha256_bytes(output_bytes),
        }]),
    ], ignore_index=True)
    bundle_bytes = core._frame_bytes(bundle_manifest, sep="\t")
    source_payloads["OUTPUT_MANIFEST_FINAL.tsv"] = output_bytes
    source_payloads["BUNDLE_MANIFEST.tsv"] = bundle_bytes
    for source in ("OUTPUT_MANIFEST_FINAL.tsv", "BUNDLE_MANIFEST.tsv"):
        (run / mapping[source]).write_bytes(source_payloads[source])
        copied[mapping[source]] = core.sha256_bytes(source_payloads[source])
    with zipfile.ZipFile(preflight_zip, "w") as archive:
        for source, data in sorted(source_payloads.items()):
            archive.writestr(f"synthetic_preflight/{source}", data)
    packaged_preflight_zip = (
        run / "provenance/preflight_snapshot/preflight_review_bundle.zip"
    )
    packaged_preflight_zip.write_bytes(preflight_zip.read_bytes())
    copied[
        "provenance/preflight_snapshot/preflight_review_bundle.zip"
    ] = core.sha256_file(packaged_preflight_zip)
    reference = {
        "classification": "VERIFIED",
        "command_line": command_line,
        "preflight_run_id": "synthetic_preflight",
        "split_hash": identity["candidate_split_hash"],
        "reference_validation": {
            "required_files_status": "PASS", "status_identity_match": "PASS",
            "live_file_set_status": "PASS",
            "bundle_manifest_status": "PASS",
            "review_bundle_live_byte_equality_status": "PASS",
            "review_bundle_sha256": core.sha256_file(preflight_zip),
            "split_hash_status": "PASS",
        },
        "preflight_identity_sha256": copied[
            "provenance/preflight_snapshot/run_identity.lock.json"
        ],
        "preflight_registry_sha256": copied[
            "provenance/preflight_snapshot/evidence_gate_registry.json"
        ],
        "preflight_output_manifest_sha256": copied[
            "provenance/preflight_snapshot/OUTPUT_MANIFEST_FINAL.tsv"
        ],
        "preflight_bundle_manifest_sha256": copied[
            "provenance/preflight_snapshot/BUNDLE_MANIFEST.tsv"
        ],
        "preflight_review_bundle_path": str(preflight_zip),
        "packaged_preflight_review_bundle_path": (
            "provenance/preflight_snapshot/preflight_review_bundle.zip"
        ),
        "preflight_review_bundle_sha256": core.sha256_file(preflight_zip),
        "preflight_review_bundle_size_bytes": preflight_zip.stat().st_size,
        "preflight_review_bundle_verification": core.verify_zip(
            preflight_zip, embedded_run_manifest=True,
        ),
        "copied_evidence_sha256": copied,
    }
    (run / "config/preflight_reference.json").write_text(json.dumps(reference))
    monkeypatch.setattr(
        core,
        "_actual_booster_configuration",
        lambda booster: details[booster.attr("amendment_variant")][
            "booster_configuration"
        ],
    )
    monkeypatch.setattr(
        core,
        "_active_reloaded_booster_cuda_probe",
        lambda booster, feature_count: {"status": "PASS"},
    )
    return run, identity


def test_smoke_semantic_completeness_accepts_cross_checked_staging_fixture(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    result = core.validate_package_completeness(run, identity)
    assert result["smoke_semantic_completeness"]["status"] == "PASS"
    assert result["smoke_semantic_completeness"]["phase"] == "STAGING"


def test_smoke_review_zip_contains_every_required_self_contained_snapshot(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    result = core.package_amended_run(run)
    embedded = core.verify_zip(Path(result["bundle_path"]), embedded_run_manifest=True)
    assert embedded["status"] == "PASS"
    assert embedded["embedded_manifest_status"] == "PASS"
    with zipfile.ZipFile(result["bundle_path"]) as archive:
        members = {
            name.split("/", 1)[1] for name in archive.namelist()
        }
        required = core.PACKAGE_REQUIRED_FILES[
            (identity["run_kind"], identity["state"])
        ]
        assert required <= members
        assert (
            "provenance/preflight_snapshot/OUTPUT_MANIFEST_FINAL.tsv" in members
        )
        assert "provenance/preflight_snapshot/BUNDLE_MANIFEST.tsv" in members
        assert (
            "provenance/preflight_snapshot/preflight_review_bundle.zip" in members
        )
        assert archive.testzip() is None


def test_smoke_semantics_requires_fixed_copied_preflight_zip_binding(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    reference_path = run / "config/preflight_reference.json"
    reference = json.loads(reference_path.read_text())
    reference["packaged_preflight_review_bundle_path"] = (
        "provenance/preflight_snapshot/another.zip"
    )
    reference_path.write_text(json.dumps(reference))

    with pytest.raises(core.IntegrityError, match="packaged_preflight_review_zip"):
        core.validate_package_completeness(run, identity)


def _promote_smoke_semantic_fixture(run: Path):
    ownership: dict[Path, core.PublicationOwnershipToken] = {}
    payload = json.loads((run / "provenance/evidence_gate_registry.json").read_text())
    gate_inputs = {
        key: {
            "classification": value["classification"],
            "passed": value["passed"],
            "evidence_refs": value["evidence_refs"],
            "detail": value.get("detail", ""),
        }
        for key, value in payload["gates"].items()
    }
    facts = gates.WorkflowFacts(
        phase="CUDA_SMOKE",
        official_smoke_variants_completed=9,
        exploratory_smoke_variants_completed=1,
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
    )
    verification = core.package_amended_run(run, ownership=ownership)
    staging_path = Path(verification["bundle_path"])
    staging_destination = staging_path.with_name(
        staging_path.stem + "_staging_verified.zip"
    )
    core.move_owned_publication_no_clobber(
        staging_path, staging_destination, ownership=ownership,
    )
    verification["staging_bundle_original_path_before_rename"] = (
        verification.pop("bundle_path")
    )
    verification["staging_bundle_temporary_path_after_rename"] = str(
        staging_destination
    )
    verification["staging_bundle_lifecycle"] = (
        "SUPERSEDED_BY_FINAL_BUNDLE_AND_DELETED_AFTER_FINAL_REOPEN"
    )
    verification["purpose"] = (
        "Synthetic first-pass package verification used to exercise the same "
        "state promotion as the production Smoke transaction."
    )
    _, registry, state = core.promote_smoke_package_gate(
        gate_inputs, facts, verification,
    )
    (run / "provenance/package_staging_verification.json").write_text(
        json.dumps(verification)
    )
    (run / "provenance/evidence_gate_registry.json").write_text(json.dumps(
        core._smoke_registry_payload(registry, state)
    ))
    metrics_path = run / "metrics/SMOKE_REPORT.json"
    metrics = json.loads(metrics_path.read_text())
    metrics.update({
        "state": state.run_state,
        "cuda_smoke_status": state.cuda_smoke_status,
        "ready_for_full_awaiting_external_review": True,
        "blockers": [],
        "output_manifest_zip_verification": "PASS",
    })
    metrics_path.write_text(json.dumps(metrics))
    model = json.loads(
        (run / "provenance/smoke_model_bundle_verification.json").read_text()
    )
    (run / "RUN_STATUS.txt").write_text(
        gates.render_status_lines(state, extra_lines={
            "SMOKE_MODELS_BUNDLE_PATH": model["portable_relative_to_run"],
            "SMOKE_MODELS_BUNDLE_SHA256": model["sha256"],
            "SMOKE_MODELS_BUNDLE_RUN_ID": model["run_id"],
            "CLASSIFICATION": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
            "EXTERNAL_AUDIT_SCORED": "NO",
        })
    )
    core._write_smoke_blockers(run, state)
    return verification, state


def test_smoke_semantic_completeness_accepts_promoted_final_fixture(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    _promote_smoke_semantic_fixture(run)

    result = core.validate_package_completeness(run, identity)
    assert result["smoke_semantic_completeness"]["status"] == "PASS"
    assert result["smoke_semantic_completeness"]["phase"] == "FINAL"
    assert result["smoke_semantic_completeness"][
        "ready_for_full_awaiting_external_review"
    ] is True


def test_smoke_final_semantics_rejects_staging_bundle_drift(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    verification, _ = _promote_smoke_semantic_fixture(run)
    verification["bundle_sha256"] = "0" * 64
    (run / "provenance/package_staging_verification.json").write_text(
        json.dumps(verification)
    )

    with pytest.raises(
        core.IntegrityError, match="staging_verification_matches_output_gate",
    ):
        core.validate_package_completeness(run, identity)


def test_smoke_final_semantics_revalidates_after_staging_bundle_is_deleted(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    verification, _ = _promote_smoke_semantic_fixture(run)
    final_verification = core.package_amended_run(run)
    Path(verification["staging_bundle_temporary_path_after_rename"]).unlink()

    result = core.validate_package_completeness(run, identity)
    assert result["smoke_semantic_completeness"]["status"] == "PASS"
    assert Path(final_verification["bundle_path"]).is_file()


def test_smoke_semantics_rejects_direct_copy_drift_even_when_map_is_updated(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    destination = "docs/PROTOCOL_AMENDMENT_01.md"
    (run / destination).write_text("coordinated but unreviewed replacement\n")
    reference_path = run / "config/preflight_reference.json"
    reference = json.loads(reference_path.read_text())
    reference["copied_evidence_sha256"][destination] = core.sha256_file(
        run / destination
    )
    reference_path.write_text(json.dumps(reference))

    with pytest.raises(
        core.IntegrityError,
        match="direct_copies_equal_packaged_preflight_members",
    ):
        core.validate_package_completeness(run, identity)


def test_smoke_semantic_completeness_rejects_unrelated_model_bundle(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    unrelated = runtime_case / "results/unrelated_models.zip"
    payload = b"not-a-completed-variant"
    manifest = pd.DataFrame([{
        "relative_path": "invented.ubj",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }]).to_csv(sep="\t", index=False).encode()
    with zipfile.ZipFile(unrelated, "w") as archive:
        archive.writestr("unrelated/invented.ubj", payload)
        archive.writestr("unrelated/BUNDLE_MANIFEST.tsv", manifest)
    verification = {
        "path": str(unrelated),
        "sha256": core.sha256_file(unrelated),
        "member_count": 2,
        "crc_status": "PASS",
        "reopen_member_hash_status": "PASS",
        "classification": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
    }
    (run / "provenance/smoke_model_bundle_verification.json").write_text(
        json.dumps(verification)
    )
    with pytest.raises(core.IntegrityError, match="model_bundle_verified"):
        core.validate_package_completeness(run, identity)


def test_smoke_semantic_completeness_rejects_repacked_model_bytes(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    model = json.loads(
        (run / "provenance/smoke_model_bundle_verification.json").read_text()
    )
    model_path = Path(model["path"])
    metrics_path = run / "metrics/SMOKE_REPORT.json"
    metrics = json.loads(metrics_path.read_text())
    completed = sorted(metrics["details"])
    replacement_root = runtime_case / ".runtime/models/replacement"
    replacement_root.mkdir(parents=True)
    for variant in completed:
        feature_count = metrics["details"][variant]["feature_count"]
        replacement_path = replacement_root / f"{variant}.ubj"
        replacement_path.write_bytes(_valid_test_booster_bytes(feature_count))
        metrics["details"][variant]["model_sha256"] = core.sha256_file(
            replacement_path
        )
    model_path.unlink()
    replacement = core._package_smoke_models(replacement_root, "valid_smoke")
    assert replacement["path"] == str(model_path)
    (run / "provenance/smoke_model_bundle_verification.json").write_text(
        json.dumps(replacement)
    )
    metrics["smoke_models_bundle"] = replacement
    metrics_path.write_text(json.dumps(metrics))

    with pytest.raises(core.IntegrityError, match="model_bundle_verified"):
        core.validate_package_completeness(run, identity)


def test_smoke_model_bundle_is_bound_to_run_identity(runtime_case, monkeypatch):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    verification_path = run / "provenance/smoke_model_bundle_verification.json"
    verification = json.loads(verification_path.read_text())
    verification["run_id"] = "different_smoke_run"
    verification_path.write_text(json.dumps(verification))
    metrics_path = run / "metrics/SMOKE_REPORT.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["smoke_models_bundle"] = verification
    metrics_path.write_text(json.dumps(metrics))

    with pytest.raises(core.IntegrityError, match="model_bundle_verified"):
        core.validate_package_completeness(run, identity)


def test_smoke_model_bundle_reported_size_is_verified(runtime_case, monkeypatch):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    verification_path = run / "provenance/smoke_model_bundle_verification.json"
    verification = json.loads(verification_path.read_text())
    verification["size_bytes"] += 1
    verification_path.write_text(json.dumps(verification))
    metrics_path = run / "metrics/SMOKE_REPORT.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["smoke_models_bundle"] = verification
    metrics_path.write_text(json.dumps(metrics))

    with pytest.raises(core.IntegrityError, match="model_bundle_verified"):
        core.validate_package_completeness(run, identity)


def test_smoke_status_is_bound_to_model_bundle(runtime_case, monkeypatch):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    status_path = run / "RUN_STATUS.txt"
    status_path.write_text(
        status_path.read_text().replace(
            "SMOKE_MODELS_BUNDLE_RUN_ID=valid_smoke",
            "SMOKE_MODELS_BUNDLE_RUN_ID=different_smoke_run",
        )
    )

    with pytest.raises(
        core.IntegrityError, match="run_status_model_bundle_reference",
    ):
        core.validate_package_completeness(run, identity)


def test_smoke_semantic_completeness_requires_runtime_cuda_and_stage_schema(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    metrics_path = run / "metrics/SMOKE_REPORT.json"
    metrics = json.loads(metrics_path.read_text())
    first_variant = next(iter(metrics["details"]))
    del metrics["details"][first_variant]["cpu_seconds"]
    metrics_path.write_text(json.dumps(metrics))

    with pytest.raises(core.IntegrityError, match="model_bundle_verified"):
        core.validate_package_completeness(run, identity)


def test_smoke_semantics_rejects_failed_active_cuda_probe_with_fabricated_config(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    monkeypatch.setattr(
        core,
        "_actual_booster_configuration",
        REAL_ACTUAL_BOOSTER_CONFIGURATION,
    )
    monkeypatch.setattr(
        core,
        "_active_reloaded_booster_cuda_probe",
        lambda booster, feature_count: {
            "status": "FAIL",
            "device": "cpu",
            "warnings": ["active CUDA inference was not observed"],
        },
    )
    with pytest.raises(core.IntegrityError, match="model_bundle_verified"):
        core.validate_package_completeness(run, identity)


def test_existing_cuda_smoke_outputs_enforces_single_run(runtime_case, monkeypatch):
    results = runtime_case / "results"
    results.mkdir()
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    assert core.existing_cuda_smoke_outputs() == []
    (results / "unrelated_review_bundle.zip").write_bytes(b"evidence")
    assert core.existing_cuda_smoke_outputs() == []
    prior = results / "run_20260813_000000_deadbeef_cuda_smoke"
    prior.mkdir()
    assert core.existing_cuda_smoke_outputs() == [prior.name]


def test_gpu_memory_usage_uses_wsl_global_fallback(monkeypatch):
    def fake_run(command, *, check=True, cwd=None):
        if "--query-compute-apps=pid,used_memory" in command:
            return f"{os.getpid()}, [N/A]\n"
        if "--query-gpu=memory.used" in command:
            return "555\n"
        raise AssertionError(command)

    monkeypatch.setattr(legacy, "run_command", fake_run)
    assert legacy.gpu_memory_usage_mib(os.getpid()) == (
        555.0, "GLOBAL_DEVICE_WSL_FALLBACK",
    )


def test_xgboost_ubj_preserves_stable_configuration_projection(runtime_case):
    booster_bytes, pre_save_configuration = _valid_test_booster_artifacts(7)
    parameter_lock = core.r92_model_parameter_lock()
    assert core.booster_matches_parameter_lock(
        pre_save_configuration,
        parameter_lock["smoke_classifier_parameters"],
    )
    booster = core.xgb.Booster()
    booster.load_model(bytearray(booster_bytes))
    before = core._actual_booster_configuration(booster)
    path = runtime_case / "roundtrip.ubj"
    booster.save_model(path)
    reloaded = core.xgb.Booster()
    reloaded.load_model(path)
    after = core._actual_booster_configuration(reloaded)
    assert core._stable_booster_configuration(before) == (
        core._stable_booster_configuration(after)
    )


def test_r92_model_parameter_lock_matches_checkpoint_and_limits():
    parameter_lock = core.r92_model_parameter_lock()
    historical = parameter_lock["historical_classifier_parameters"]
    assert parameter_lock["classification"] == "VERIFIED"
    assert parameter_lock["checkpoint_classifier_parameters"] == historical
    assert parameter_lock["full_classifier_parameters"] == {
        **historical,
        "n_estimators": 2000,
        "early_stopping_rounds": 30,
        "device": parameter_lock["full_classifier_parameters"]["device"],
    }
    assert parameter_lock["smoke_classifier_parameters"] == {
        **historical,
        "n_estimators": 16,
        "early_stopping_rounds": 4,
        "device": "cuda",
    }
    assert core.sha256_file(core.R92_MODEL) == parameter_lock["checkpoint_sha256"]


def test_smoke_semantics_rejects_parameter_lock_mutation(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    metrics_path = run / "metrics/SMOKE_REPORT.json"
    metrics = json.loads(metrics_path.read_text())
    first_variant = next(iter(metrics["details"]))
    metrics["details"][first_variant]["requested_model_parameters"][
        "max_depth"
    ] = 4
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(core.IntegrityError, match="model_bundle_verified"):
        core.validate_package_completeness(run, identity)


def test_smoke_semantics_rejects_resource_table_detail_drift(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    table_path = run / "tables/SMOKE_REPORT.csv"
    table = pd.read_csv(table_path)
    table.loc[0, "wall_seconds"] += 1.0
    table.to_csv(table_path, index=False)
    with pytest.raises(
        core.IntegrityError, match="smoke_table_matches_variant_details",
    ):
        core.validate_package_completeness(run, identity)


def test_smoke_semantics_recomputes_execution_ledger(runtime_case, monkeypatch):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    ledger_path = run / "provenance/execution_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["invocation_argv"] = ["run_study.py", "--smoke", "--full"]
    ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(
        core.IntegrityError, match="execution_ledger_recomputed_exactly",
    ):
        core.validate_package_completeness(run, identity)


def test_smoke_semantics_recomputes_deterministic_subset(runtime_case, monkeypatch):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    subset_path = run / "config/smoke_subset.json"
    subset = json.loads(subset_path.read_text())
    subset["training_stable_candidate_id_sha256"] = "0" * 64
    subset_path.write_text(json.dumps(subset))
    with pytest.raises(
        core.IntegrityError, match="deterministic_subset_recomputed_exactly",
    ):
        core.validate_package_completeness(run, identity)


def test_smoke_semantics_recomputes_resampling_parity(runtime_case, monkeypatch):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    metrics_path = run / "metrics/SMOKE_REPORT.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["parity"]["full_vs_no_weight"]["X_sha256"] = False
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(
        core.IntegrityError, match="resampling_parity_recomputed_exactly",
    ):
        core.validate_package_completeness(run, identity)


def test_smoke_semantics_rejects_coordinated_scientific_relabeling(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    metrics_path = run / "metrics/SMOKE_REPORT.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["classification"] = "SCIENTIFIC_RESULT"
    metrics["external_audit_scored"] = True
    metrics_path.write_text(json.dumps(metrics))
    status_path = run / "RUN_STATUS.txt"
    status_path.write_text(
        status_path.read_text()
        .replace(
            "CLASSIFICATION=NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
            "CLASSIFICATION=SCIENTIFIC_RESULT",
        )
        .replace("EXTERNAL_AUDIT_SCORED=NO", "EXTERNAL_AUDIT_SCORED=YES")
    )
    with pytest.raises(
        core.IntegrityError, match="non_scientific_global_typing",
    ):
        core.validate_package_completeness(run, identity)


def test_smoke_semantics_rejects_inherited_gate_status_tamper(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    status_path = run / "RUN_STATUS.txt"
    status_path.write_text(
        status_path.read_text().replace(
            "FEATURE_SEMANTICS_STATUS=PASS",
            "FEATURE_SEMANTICS_STATUS=FAIL",
        )
    )
    with pytest.raises(
        core.IntegrityError, match="run_status_matches_derived_state",
    ):
        core.validate_package_completeness(run, identity)


def test_smoke_semantics_rejects_extra_blocker_claim(runtime_case, monkeypatch):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    blockers_path = run / "BLOCKERS.md"
    blockers_path.write_text(
        blockers_path.read_text()
        + "- `INVENTED_BLOCKER` (BLOCKED/BLOCKED): fabricated.\n"
    )
    with pytest.raises(
        core.IntegrityError, match="blocker_document_matches_derived_state",
    ):
        core.validate_package_completeness(run, identity)


def test_incomplete_smoke_verifies_every_completed_model_bundle_member(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    metrics_path = run / "metrics/SMOKE_REPORT.json"
    metrics = json.loads(metrics_path.read_text())
    partial_ids = list(metrics["details"])[:2]
    old_model = json.loads(
        (run / "provenance/smoke_model_bundle_verification.json").read_text()
    )
    Path(old_model["path"]).unlink()
    partial_model = core._package_smoke_models(
        runtime_case / ".runtime/models/valid_smoke",
        "valid_smoke",
        completed_variant_ids=partial_ids,
    )
    (run / "provenance/smoke_model_bundle_verification.json").write_text(
        json.dumps(partial_model)
    )
    table_path = run / "tables/SMOKE_REPORT.csv"
    table = pd.read_csv(table_path)
    table[table.variant.isin(partial_ids)].to_csv(table_path, index=False)
    ledger_path = run / "provenance/execution_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger.update({
        "official_smoke_variants_completed": len(partial_ids),
        "exploratory_smoke_variants_completed": 0,
        "smoke_variants_completed_ids": sorted(partial_ids),
    })
    ledger_path.write_text(json.dumps(ledger))
    metrics.update({
        "state": "SMOKE_INCOMPLETE",
        "official_variants_completed": len(partial_ids),
        "exploratory_variants_completed": 0,
        "details": {
            variant: metrics["details"][variant] for variant in partial_ids
        },
        "smoke_models_bundle": partial_model,
    })
    metrics["parity"] = core.smoke_resampling_parity(metrics["details"])
    registry_payload = json.loads(
        (run / "provenance/evidence_gate_registry.json").read_text()
    )
    gate_inputs = {
        key: {
            "classification": value["classification"],
            "passed": value["passed"],
            "evidence_refs": value["evidence_refs"],
            "detail": value.get("detail", ""),
        }
        for key, value in registry_payload["gates"].items()
    }
    for key in ("cuda_smoke_all_runnable", "model_save_reload"):
        gate_inputs[key]["classification"] = "VERIFIED"
        gate_inputs[key]["passed"] = False
    facts = gates.WorkflowFacts(
        phase="CUDA_SMOKE",
        official_smoke_variants_completed=len(partial_ids),
        exploratory_smoke_variants_completed=0,
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
    )
    registry = gates.build_gate_registry(gate_inputs)
    state = gates.derive_workflow_state(registry, facts)
    assert state.run_state == "SMOKE_INCOMPLETE"
    identity["state"] = state.run_state
    (run / "config/run_identity.lock.json").write_text(json.dumps(identity))
    (run / "provenance/evidence_gate_registry.json").write_text(json.dumps(
        core._smoke_registry_payload(registry, state)
    ))
    metrics["cuda_smoke_status"] = state.cuda_smoke_status
    metrics_path.write_text(json.dumps(metrics))
    (run / "RUN_STATUS.txt").write_text(
        gates.render_status_lines(state, extra_lines={
            "SMOKE_MODELS_BUNDLE_PATH": partial_model["portable_relative_to_run"],
            "SMOKE_MODELS_BUNDLE_SHA256": partial_model["sha256"],
            "SMOKE_MODELS_BUNDLE_RUN_ID": partial_model["run_id"],
            "CLASSIFICATION": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
            "EXTERNAL_AUDIT_SCORED": "NO",
        })
    )
    core._write_smoke_blockers(run, state)
    (run / "logs/smoke_failure.txt").write_text("injected partial failure\n")

    result = core.validate_package_completeness(run, identity)
    assert result["smoke_semantic_completeness"]["status"] == "PASS"
    Path(partial_model["path"]).write_bytes(b"corrupted")
    with pytest.raises(core.IntegrityError, match="model_bundle_verified"):
        core.validate_package_completeness(run, identity)


@pytest.mark.parametrize(
    "collision_kind", ["regular", "directory", "dangling_symlink", "fifo"],
)
def test_incomplete_smoke_preserves_completed_evidence_when_model_bundle_fails(
    runtime_case, monkeypatch, collision_kind,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    metrics_path = run / "metrics/SMOKE_REPORT.json"
    metrics = json.loads(metrics_path.read_text())
    completed_ids = list(metrics["details"])[:2]

    verification_path = run / "provenance/smoke_model_bundle_verification.json"
    model = json.loads(verification_path.read_text())
    Path(model["path"]).unlink()
    verification_path.unlink()

    table_path = run / "tables/SMOKE_REPORT.csv"
    table = pd.read_csv(table_path)
    table[table.variant.isin(completed_ids)].to_csv(table_path, index=False)
    reference = json.loads((run / "config/preflight_reference.json").read_text())
    variants = json.loads((run / "config/variant_feature_sets.json").read_text())
    runnable = core.runnable_official_variants(variants)
    ledger = core.smoke_execution_ledger(
        run,
        command_line=reference["command_line"],
        runnable_official=runnable,
    )
    (run / "provenance/execution_ledger.json").write_text(json.dumps(ledger))

    failure = core.failed_smoke_model_bundle_record(
        run.name, completed_ids, core.IntegrityError("injected model bundle failure"),
    )
    (run / "provenance/smoke_model_bundle_failure.json").write_text(
        json.dumps(failure)
    )
    metrics.update({
        "state": "SMOKE_INCOMPLETE",
        "official_variants_completed": ledger["official_smoke_variants_completed"],
        "exploratory_variants_completed": ledger[
            "exploratory_smoke_variants_completed"
        ],
        "details": {
            variant: metrics["details"][variant] for variant in completed_ids
        },
        "smoke_models_bundle": failure,
    })
    metrics["parity"] = core.smoke_resampling_parity(metrics["details"])

    stored_registry = json.loads(
        (run / "provenance/evidence_gate_registry.json").read_text()
    )
    gate_inputs = {
        key: {
            "classification": value["classification"],
            "passed": value["passed"],
            "evidence_refs": value["evidence_refs"],
            "detail": value.get("detail", ""),
        }
        for key, value in stored_registry["gates"].items()
    }
    for key in ("cuda_smoke_all_runnable", "model_save_reload"):
        gate_inputs[key] = {
            "classification": "VERIFIED",
            "passed": False,
            "evidence_refs": [
                "metrics/SMOKE_REPORT.json",
                "provenance/smoke_model_bundle_failure.json",
            ],
            "detail": "Injected model bundle failure was preserved fail-closed.",
        }
    facts = gates.WorkflowFacts(
        phase="CUDA_SMOKE",
        official_smoke_variants_completed=ledger[
            "official_smoke_variants_completed"
        ],
        exploratory_smoke_variants_completed=ledger[
            "exploratory_smoke_variants_completed"
        ],
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
    )
    registry = gates.build_gate_registry(gate_inputs)
    state = gates.derive_workflow_state(registry, facts)
    assert state.run_state == "SMOKE_INCOMPLETE"
    assert state.cuda_smoke_status != "PASS"

    identity["state"] = state.run_state
    (run / "config/run_identity.lock.json").write_text(json.dumps(identity))
    (run / "provenance/evidence_gate_registry.json").write_text(json.dumps(
        core._smoke_registry_payload(registry, state)
    ))
    metrics["state"] = state.run_state
    metrics["cuda_smoke_status"] = state.cuda_smoke_status
    metrics_path.write_text(json.dumps(metrics))
    (run / "RUN_STATUS.txt").write_text(
        gates.render_status_lines(state, extra_lines={
            "SMOKE_MODELS_BUNDLE_PATH": failure["portable_relative_to_run"],
            "SMOKE_MODELS_BUNDLE_SHA256": failure["sha256"],
            "SMOKE_MODELS_BUNDLE_RUN_ID": failure["run_id"],
            "CLASSIFICATION": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
            "EXTERNAL_AUDIT_SCORED": "NO",
        })
    )
    core._write_smoke_blockers(run, state)
    (run / "logs/smoke_failure.txt").write_text(
        "injected model bundle failure\n"
    )

    result = core.validate_package_completeness(run, identity)
    assert result["smoke_semantic_completeness"]["status"] == "PASS"
    assert result["smoke_semantic_completeness"]["ready_for_full_awaiting_external_review"] is False

    collision_path = (
        run.parent / f"{run.name}_NON_SCIENTIFIC_smoke_models.zip"
    )
    if collision_kind == "regular":
        collision_path.write_bytes(b"orphan")
    elif collision_kind == "directory":
        collision_path.mkdir()
    elif collision_kind == "dangling_symlink":
        collision_path.symlink_to("missing-unowned-target")
    else:
        os.mkfifo(collision_path)
    with pytest.raises(core.IntegrityError, match="model_bundle_verified"):
        core.validate_package_completeness(run, identity)

    recorded = core.failed_smoke_model_bundle_record(
        run.name, completed_ids, core.IntegrityError("collision"),
    )
    (run / "provenance/smoke_model_bundle_failure.json").write_text(
        json.dumps(recorded)
    )
    metrics["smoke_models_bundle"] = recorded
    metrics_path.write_text(json.dumps(metrics))
    (run / "RUN_STATUS.txt").write_text(
        gates.render_status_lines(state, extra_lines={
            "SMOKE_MODELS_BUNDLE_PATH": recorded["portable_relative_to_run"],
            "SMOKE_MODELS_BUNDLE_SHA256": recorded["sha256"],
            "SMOKE_MODELS_BUNDLE_RUN_ID": recorded["run_id"],
            "CLASSIFICATION": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
            "EXTERNAL_AUDIT_SCORED": "NO",
        })
    )
    result = core.validate_package_completeness(run, identity)
    assert result["smoke_semantic_completeness"]["status"] == "PASS"

    recorded["preexisting_unowned_collision"]["st_ino"] += 1
    (run / "provenance/smoke_model_bundle_failure.json").write_text(
        json.dumps(recorded)
    )
    metrics["smoke_models_bundle"] = recorded
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(core.IntegrityError, match="model_bundle_verified"):
        core.validate_package_completeness(run, identity)


def test_smoke_model_bundle_reference_is_relocatable_with_run(
    runtime_case, monkeypatch,
):
    run, identity = _valid_smoke_semantic_fixture(runtime_case, monkeypatch)
    relocated_parent = runtime_case / "relocated"
    relocated_run = relocated_parent / run.name
    shutil.copytree(run, relocated_run)
    model = json.loads(
        (run / "provenance/smoke_model_bundle_verification.json").read_text()
    )
    relocated_parent.mkdir(exist_ok=True)
    shutil.copy2(Path(model["path"]), relocated_parent / model["filename"])

    result = core.validate_package_completeness(relocated_run, identity)
    assert result["smoke_semantic_completeness"]["status"] == "PASS"


@pytest.mark.parametrize(
    "replacement_kind",
    ["same_inode_mutation", "same_bytes_new_inode", "directory", "symlink"],
)
def test_owned_publication_cleanup_preserves_replacements(
    runtime_case, monkeypatch, replacement_kind,
):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    target = runtime_case / "results/owned.zip"
    target.parent.mkdir()
    ownership: dict[Path, core.PublicationOwnershipToken] = {}
    core.publish_bytes_no_clobber(
        target, b"owned-by-transaction", ownership=ownership,
    )

    if replacement_kind == "same_inode_mutation":
        target.write_bytes(b"foreign-mutation")
    elif replacement_kind == "same_bytes_new_inode":
        replacement = target.with_name("replacement")
        replacement.write_bytes(b"owned-by-transaction")
        os.replace(replacement, target)
    elif replacement_kind == "directory":
        target.unlink()
        target.mkdir()
    else:
        target.unlink()
        target.symlink_to("missing-foreign-target")

    before = core._path_collision_evidence(target)
    assert core.remove_owned_publication(target, ownership=ownership) is False
    assert core._path_collision_evidence(target) == before


def test_owned_cleanup_preserves_replacement_created_after_atomic_quarantine(
    runtime_case, monkeypatch,
):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    target = runtime_case / "results/owned.zip"
    target.parent.mkdir()
    ownership: dict[Path, core.PublicationOwnershipToken] = {}
    core.publish_bytes_no_clobber(target, b"owned", ownership=ownership)
    owned_inode = os.lstat(target).st_ino
    real_match = core._publication_matches_token
    injected = False

    def recreate_after_quarantine(path, token):
        nonlocal injected
        if Path(path) != target and not injected:
            injected = True
            target.write_bytes(b"foreign-after-quarantine")
        return real_match(path, token)

    monkeypatch.setattr(core, "_publication_matches_token", recreate_after_quarantine)

    assert core.remove_owned_publication(target, ownership=ownership) is True
    assert target.read_bytes() == b"foreign-after-quarantine"
    assert os.lstat(target).st_ino != owned_inode
    assert ownership == {}
    assert not list(target.parent.glob(f".{target.name}.rollback-*"))


def test_snapshot_restore_is_no_clobber_when_replacement_arrives(
    runtime_case, monkeypatch,
):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    target = runtime_case / "results/OUTPUT_MANIFEST_FINAL.tsv"
    target.parent.mkdir()
    target.write_bytes(b"previous")
    snapshot = {target: target.read_bytes()}
    ownership: dict[Path, core.PublicationOwnershipToken] = {}
    core.atomic_write_bytes(
        target, b"generated", track_publication=True, ownership=ownership,
    )
    real_publish = core.publish_bytes_no_clobber

    def collide_before_restore(path, data, **kwargs):
        if Path(path) == target and not target.exists():
            target.write_bytes(b"foreign")
        return real_publish(path, data, **kwargs)

    monkeypatch.setattr(core, "publish_bytes_no_clobber", collide_before_restore)

    with pytest.raises(core.IntegrityError, match="snapshot restoration"):
        core._restore_file_snapshot(
            snapshot,
            ownership=ownership,
            previous_ownership={target: None},
        )
    assert target.read_bytes() == b"foreign"
    assert core._publication_key(target) not in ownership


def test_required_staging_cleanup_rejects_ownership_mismatch(
    runtime_case, monkeypatch,
):
    target = runtime_case / "staging.zip"
    monkeypatch.setattr(
        core, "remove_owned_publication", lambda *args, **kwargs: False,
    )
    with pytest.raises(
        core.IntegrityError, match="staging bundle publication changed",
    ):
        core.remove_owned_publication_required(
            target, ownership={}, purpose="staging bundle",
        )


def test_packager_rejects_mutation_between_semantic_validation_and_snapshot(
    runtime_case, monkeypatch,
):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    run = runtime_case / "results/semantic_snapshot_drift"
    (run / "config").mkdir(parents=True)
    evidence = run / "evidence.txt"
    evidence.write_bytes(b"reviewed\n")
    identity = {
        "run_id": run.name,
        "run_kind": "AMENDED_PREFLIGHT",
        "state": "PREFLIGHT_COMPLETE",
    }
    (run / "config/run_identity.lock.json").write_text(json.dumps(identity))

    def validate_then_mutate(run_dir, observed_identity):
        assert observed_identity == identity
        evidence.write_bytes(b"tampered\n")
        return {"required_files_status": "PASS", "status_identity_match": "PASS"}

    monkeypatch.setattr(core, "validate_package_completeness", validate_then_mutate)

    with pytest.raises(core.IntegrityError, match="changed during semantic"):
        core.package_amended_run(run)

    destination = runtime_case / "results/semantic_snapshot_drift_review_bundle.zip"
    assert not destination.exists()
    assert not (run / "OUTPUT_MANIFEST_FINAL.tsv").exists()
    assert not (run / "BUNDLE_MANIFEST.tsv").exists()
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_packager_never_publishes_unverified_temporary_zip(
    runtime_case, monkeypatch,
):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    monkeypatch.setattr(
        core, "validate_preflight_semantic_completeness",
        lambda run_dir, identity: {"status": "PASS", "checks": {}},
    )
    run = runtime_case / "results/atomic_failure"
    identity = {
        "run_id": run.name,
        "run_kind": "AMENDED_PREFLIGHT",
        "state": "PREFLIGHT_COMPLETE",
    }
    required = core.PACKAGE_REQUIRED_FILES[(identity["run_kind"], identity["state"])]
    for relative in required:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence\n")
    (run / "config/run_identity.lock.json").write_text(json.dumps(identity))
    (run / "RUN_STATUS.txt").write_text(
        "RUN_KIND=AMENDED_PREFLIGHT\nRUN_STATE=PREFLIGHT_COMPLETE\n"
    )
    monkeypatch.setattr(core.zipfile.ZipFile, "testzip", lambda self: "bad/member")

    with pytest.raises(core.IntegrityError, match="Bundle verification failed"):
        core.package_amended_run(run)

    destination = runtime_case / "results/atomic_failure_review_bundle.zip"
    assert not destination.exists()
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))
    assert not (run / "OUTPUT_MANIFEST_FINAL.tsv").exists()
    assert not (run / "BUNDLE_MANIFEST.tsv").exists()


def test_packager_restores_previous_manifests_after_verification_failure(
    runtime_case, monkeypatch,
):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    monkeypatch.setattr(
        core, "validate_preflight_semantic_completeness",
        lambda run_dir, identity: {"status": "PASS", "checks": {}},
    )
    run = runtime_case / "results/manifest_restore"
    identity = {
        "run_id": run.name,
        "run_kind": "AMENDED_PREFLIGHT",
        "state": "PREFLIGHT_COMPLETE",
    }
    required = core.PACKAGE_REQUIRED_FILES[(identity["run_kind"], identity["state"])]
    for relative in required:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence\n")
    (run / "config/run_identity.lock.json").write_text(json.dumps(identity))
    (run / "RUN_STATUS.txt").write_text(
        "RUN_KIND=AMENDED_PREFLIGHT\nRUN_STATE=PREFLIGHT_COMPLETE\n"
    )
    output_before = b"previous-output-manifest\n"
    bundle_before = b"previous-bundle-manifest\n"
    (run / "OUTPUT_MANIFEST_FINAL.tsv").write_bytes(output_before)
    (run / "BUNDLE_MANIFEST.tsv").write_bytes(bundle_before)
    monkeypatch.setattr(core.zipfile.ZipFile, "testzip", lambda self: "bad/member")

    with pytest.raises(core.IntegrityError, match="Bundle verification failed"):
        core.package_amended_run(run)

    assert (run / "OUTPUT_MANIFEST_FINAL.tsv").read_bytes() == output_before
    assert (run / "BUNDLE_MANIFEST.tsv").read_bytes() == bundle_before
    assert not (runtime_case / "results/manifest_restore_review_bundle.zip").exists()


def test_packager_restores_first_manifest_when_second_manifest_write_fails(
    runtime_case, monkeypatch,
):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    monkeypatch.setattr(
        core, "validate_preflight_semantic_completeness",
        lambda run_dir, identity: {"status": "PASS", "checks": {}},
    )
    run = runtime_case / "results/second_manifest_failure"
    identity = {
        "run_id": run.name,
        "run_kind": "AMENDED_PREFLIGHT",
        "state": "PREFLIGHT_COMPLETE",
    }
    required = core.PACKAGE_REQUIRED_FILES[(identity["run_kind"], identity["state"])]
    for relative in required:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence\n")
    (run / "config/run_identity.lock.json").write_text(json.dumps(identity))
    (run / "RUN_STATUS.txt").write_text(
        "RUN_KIND=AMENDED_PREFLIGHT\nRUN_STATE=PREFLIGHT_COMPLETE\n"
    )
    output_before = b"old-output\n"
    bundle_before = b"old-bundle\n"
    (run / "OUTPUT_MANIFEST_FINAL.tsv").write_bytes(output_before)
    (run / "BUNDLE_MANIFEST.tsv").write_bytes(bundle_before)
    original_write = core.atomic_write_bytes
    injected = {"raised": False}

    def fail_second_manifest(path, data, **kwargs):
        if Path(path).name == "BUNDLE_MANIFEST.tsv" and not injected["raised"]:
            injected["raised"] = True
            raise OSError("injected second-manifest failure")
        return original_write(path, data, **kwargs)

    monkeypatch.setattr(core, "atomic_write_bytes", fail_second_manifest)
    with pytest.raises(OSError, match="second-manifest failure"):
        core.package_amended_run(run)

    assert (run / "OUTPUT_MANIFEST_FINAL.tsv").read_bytes() == output_before
    assert (run / "BUNDLE_MANIFEST.tsv").read_bytes() == bundle_before
    assert not (
        runtime_case / "results/second_manifest_failure_review_bundle.zip"
    ).exists()


def test_manifest_rows_rejects_optional_symlink(runtime_case):
    run = runtime_case / "run_with_symlink"
    run.mkdir()
    outside = runtime_case / "outside.txt"
    outside.write_text("must not be bundled")
    (run / "regular.txt").write_text("evidence")
    (run / "optional-link.txt").symlink_to(outside)
    with pytest.raises(core.IntegrityError, match="contains symlinks"):
        core.manifest_rows(run)


def test_manifest_rows_rejects_unrelated_zip(runtime_case):
    run = runtime_case / "run_with_unrelated_zip"
    run.mkdir()
    (run / "evidence.txt").write_text("review evidence")
    (run / "old_run.zip").write_bytes(b"unrelated")
    with pytest.raises(core.IntegrityError, match="unrelated ZIP"):
        core.manifest_rows(run)


def test_preflight_manifest_rejects_zip_masquerading_at_smoke_allowlisted_path(
    runtime_case,
):
    run = runtime_case / "preflight_with_nested_zip"
    nested = run / "provenance/preflight_snapshot/preflight_review_bundle.zip"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"unrelated")
    with pytest.raises(core.IntegrityError, match="unrelated ZIP"):
        core.manifest_rows(run, run_kind="AMENDED_PREFLIGHT")


def _sealed_preflight_reference_fixture(runtime_case, monkeypatch):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    monkeypatch.setattr(
        core, "validate_preflight_semantic_completeness",
        lambda run_dir, identity: {"status": "PASS", "checks": {}},
    )
    results = runtime_case / "results"
    results.mkdir(exist_ok=True)
    run = results / "sealed_preflight_reference"
    identity = {
        "run_id": run.name,
        "run_kind": "AMENDED_PREFLIGHT",
        "state": "PREFLIGHT_COMPLETE",
        "permitted_transitions": ["CUDA_SMOKE_NON_SCIENTIFIC"],
        "full_authorized": False,
        "stability_authorized": False,
    }
    required = core.PACKAGE_REQUIRED_FILES[(identity["run_kind"], identity["state"])]
    required = required | {
        source
        for source in core.smoke_copied_evidence_mapping()
        if source != "__PREFLIGHT_REVIEW_BUNDLE__"
    }
    for relative in required:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n")

    source = runtime_case / "source-evidence.txt"
    source.write_text("immutable source\n")
    pd.DataFrame([{
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": core.sha256_file(source),
    }]).to_csv(
        run / "provenance/source_input_hashes_post.tsv", sep="\t", index=False,
    )
    split = pd.DataFrame({
        "stable_candidate_id": ["train:a", "validation:b"],
        "split": ["train", "validation"],
        "label": [0, 1],
        "eligible": [True, True],
        "split_order": [0, 0],
    })
    split.to_csv(run / "splits/row_split_manifest.csv.gz", index=False)
    identity["candidate_split_hash"] = core.split_assignment_hash(
        split.stable_candidate_id,
        split.split,
        split.label,
        split.eligible,
        split.split_order,
    )
    (run / "config/run_identity.lock.json").write_text(json.dumps(identity))
    (run / "RUN_STATUS.txt").write_text(
        "RUN_KIND=AMENDED_PREFLIGHT\nRUN_STATE=PREFLIGHT_COMPLETE\n"
    )
    ownership: dict[Path, core.PublicationOwnershipToken] = {}
    verification = core.package_amended_run(run, ownership=ownership)
    adjacent = results / f"{run.name}_package_verification.json"
    core.publish_json_no_clobber(
        adjacent, verification, ownership=ownership,
    )
    return run, identity


def test_sealed_preflight_reference_validates_live_bundle_equality(
    runtime_case, monkeypatch,
):
    run, identity = _sealed_preflight_reference_fixture(runtime_case, monkeypatch)
    validated = core.validate_amended_run_reference(
        run,
        expected_kind="AMENDED_PREFLIGHT",
        allowed_states={"PREFLIGHT_COMPLETE"},
    )
    assert validated["run_id"] == identity["run_id"]
    assert validated["reference_validation"][
        "review_bundle_live_byte_equality_status"
    ] == "PASS"


def test_sealed_preflight_reference_rejects_self_consistent_live_drift(
    runtime_case, monkeypatch,
):
    run, _ = _sealed_preflight_reference_fixture(runtime_case, monkeypatch)
    (run / "docs/DATA_SOURCE_DECISION.md").write_text("changed after review\n")
    manifest = core.manifest_rows(run, run_kind="AMENDED_PREFLIGHT")
    output_bytes = core._frame_bytes(manifest, sep="\t")
    bundle = pd.concat([
        manifest[["relative_path", "size_bytes", "sha256"]],
        pd.DataFrame([{
            "relative_path": "OUTPUT_MANIFEST_FINAL.tsv",
            "size_bytes": len(output_bytes),
            "sha256": core.sha256_bytes(output_bytes),
        }]),
    ], ignore_index=True)
    (run / "OUTPUT_MANIFEST_FINAL.tsv").write_bytes(output_bytes)
    (run / "BUNDLE_MANIFEST.tsv").write_bytes(
        core._frame_bytes(bundle, sep="\t")
    )

    with pytest.raises(core.IntegrityError, match="Run reference manifest"):
        core.validate_amended_run_reference(
            run,
            expected_kind="AMENDED_PREFLIGHT",
            allowed_states={"PREFLIGHT_COMPLETE"},
        )


def test_smoke_snapshot_copies_only_sealed_preflight_zip_bytes(
    runtime_case, monkeypatch,
):
    run, _ = _sealed_preflight_reference_fixture(runtime_case, monkeypatch)
    validated = core.validate_amended_run_reference(
        run,
        expected_kind="AMENDED_PREFLIGHT",
        allowed_states={"PREFLIGHT_COMPLETE"},
    )
    source_relative = "docs/PROTOCOL_AMENDMENT_01.md"
    sealed_bytes, _ = core.read_reviewed_preflight_members(
        run,
        [source_relative],
        expected_bundle_sha256=validated["reference_validation"][
            "review_bundle_sha256"
        ],
    )
    (run / source_relative).write_text("concurrent unreviewed drift\n")
    smoke = runtime_case / "results/smoke_snapshot"
    smoke.mkdir()
    core.snapshot_smoke_review_evidence(
        preflight_run=run,
        run_dir=smoke,
        validated_identity=validated,
        command_line="run_study.py --smoke",
        split_hash=validated["candidate_split_hash"],
    )
    assert (smoke / source_relative).read_bytes() == sealed_bytes[source_relative]


def test_reviewed_gzip_split_bytes_are_read_with_explicit_compression(
    runtime_case,
):
    split = pd.DataFrame({"stable_candidate_id": ["a"], "label": [1]})
    path = runtime_case / "split.csv.gz"
    split.to_csv(path, index=False, compression="gzip")
    reopened = pd.read_csv(io.BytesIO(path.read_bytes()), compression="gzip")
    pd.testing.assert_frame_equal(reopened, split)


def test_smoke_inherits_registry_from_reviewed_zip_not_live_tree(
    runtime_case, monkeypatch,
):
    run, _ = _sealed_preflight_reference_fixture(runtime_case, monkeypatch)
    validated = core.validate_amended_run_reference(
        run,
        expected_kind="AMENDED_PREFLIGHT",
        allowed_states={"PREFLIGHT_COMPLETE"},
    )
    reviewed, _ = core.read_reviewed_preflight_members(
        run,
        ["provenance/evidence_gate_registry.json"],
        expected_bundle_sha256=validated["reference_validation"][
            "review_bundle_sha256"
        ],
    )
    live_registry = run / "provenance/evidence_gate_registry.json"
    live_registry.write_text('{"unreviewed": true}\n')
    assert reviewed["provenance/evidence_gate_registry.json"] != live_registry.read_bytes()


def test_final_smoke_seal_failure_rolls_back_ready_and_canonical_artifacts(
    runtime_case, monkeypatch,
):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    run = runtime_case / "results/failing_final_smoke"
    for relative in ("provenance", "metrics", "logs"):
        (run / relative).mkdir(parents=True, exist_ok=True)
    (run / "OUTPUT_MANIFEST_FINAL.tsv").write_text("stale")
    (run / "BUNDLE_MANIFEST.tsv").write_text("stale")
    canonical = runtime_case / "results/failing_final_smoke_review_bundle.zip"
    adjacent = runtime_case / "results/failing_final_smoke_package_verification.json"
    canonical.write_bytes(b"partial")
    adjacent.write_text("false pass")
    ownership: dict[Path, core.PublicationOwnershipToken] = {}
    for path in (
        run / "OUTPUT_MANIFEST_FINAL.tsv", run / "BUNDLE_MANIFEST.tsv",
        canonical, adjacent,
    ):
        _mark_publication_owned(path, ownership)
    gate_inputs = {
        spec.key: {
            "classification": "VERIFIED", "passed": True,
            "evidence_refs": ["provenance/package_staging_verification.json"],
        }
        for spec in gates.DEFAULT_GATE_SPECS
    }
    facts = gates.WorkflowFacts(
        phase="CUDA_SMOKE",
        official_smoke_variants_completed=9,
        exploratory_smoke_variants_completed=1,
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
    )
    failed = core.rollback_smoke_final_seal_failure(
        run_dir=run,
        preflight_run=runtime_case / "results/preflight",
        preflight_report={
            "training_table_path": "/source/train.csv",
            "training_table_sha256": "2" * 64,
            "r92_control": {"max_probability_difference": 0.0},
        },
        split_hash="3" * 64,
        model_bundle=None,
        gate_inputs=gate_inputs,
        facts=facts,
        smoke_metrics={},
        error=core.IntegrityError("injected second-seal failure"),
        ownership=ownership,
    )

    status = (run / "RUN_STATUS.txt").read_text()
    assert failed.ready_for_full_awaiting_external_review is False
    assert "READY_FOR_FULL_AWAITING_EXTERNAL_REVIEW=NO" in status
    assert "OUTPUT_MANIFEST_AND_ZIP_VERIFICATION=FAIL" in status
    assert "REVIEW_BUNDLE_PATH=FINAL_SEAL_FAILED_NO_CANONICAL_BUNDLE" in status
    assert not canonical.exists()
    assert not adjacent.exists()
    assert not (run / "OUTPUT_MANIFEST_FINAL.tsv").exists()
    assert not (run / "BUNDLE_MANIFEST.tsv").exists()
    registry = json.loads((run / "provenance/evidence_gate_registry.json").read_text())
    assert registry["gates"]["output_manifest_zip_verification"]["status"] == "FAIL"


def test_smoke_rollback_preserves_manifest_recreated_after_quarantine(
    runtime_case, monkeypatch,
):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    run = runtime_case / "results/manifest_cleanup_race"
    for relative in ("provenance", "metrics", "logs"):
        (run / relative).mkdir(parents=True, exist_ok=True)
    output_manifest = run / "OUTPUT_MANIFEST_FINAL.tsv"
    bundle_manifest = run / "BUNDLE_MANIFEST.tsv"
    canonical = runtime_case / "results/manifest_cleanup_race_review_bundle.zip"
    adjacent = runtime_case / "results/manifest_cleanup_race_package_verification.json"
    for path in (output_manifest, bundle_manifest, canonical, adjacent):
        path.write_bytes(b"owned")
    ownership: dict[Path, core.PublicationOwnershipToken] = {}
    output_token = _mark_publication_owned(output_manifest, ownership)
    for path in (bundle_manifest, canonical, adjacent):
        _mark_publication_owned(path, ownership)
    real_match = core._publication_matches_token
    injected = False

    def recreate_manifest(path, token):
        nonlocal injected
        if token == output_token and Path(path) != output_manifest and not injected:
            injected = True
            output_manifest.write_bytes(b"foreign-manifest")
        return real_match(path, token)

    monkeypatch.setattr(core, "_publication_matches_token", recreate_manifest)
    facts = gates.WorkflowFacts(
        phase="CUDA_SMOKE",
        official_smoke_variants_completed=9,
        exploratory_smoke_variants_completed=1,
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
    )
    core.rollback_smoke_final_seal_failure(
        run_dir=run,
        preflight_run=runtime_case / "results/preflight",
        preflight_report={
            "training_table_path": "/source/train.csv",
            "training_table_sha256": "2" * 64,
            "r92_control": {"max_probability_difference": 0.0},
        },
        split_hash="3" * 64,
        model_bundle=None,
        gate_inputs={
            spec.key: {
                "classification": "VERIFIED", "passed": True,
                "evidence_refs": ["provenance/package_staging_verification.json"],
            }
            for spec in gates.DEFAULT_GATE_SPECS
        },
        facts=facts,
        smoke_metrics={},
        error=core.IntegrityError("injected final seal failure"),
        ownership=ownership,
    )

    assert output_manifest.read_bytes() == b"foreign-manifest"
    assert not bundle_manifest.exists()
    assert not canonical.exists()
    assert not adjacent.exists()


def test_smoke_rollback_preserves_unrecognized_preexisting_canonical_files(
    runtime_case, monkeypatch,
):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    run = runtime_case / "results/collision_smoke"
    for relative in ("provenance", "metrics", "logs"):
        (run / relative).mkdir(parents=True, exist_ok=True)
    canonical = runtime_case / "results/collision_smoke_review_bundle.zip"
    adjacent = runtime_case / "results/collision_smoke_package_verification.json"
    canonical.write_bytes(b"preexisting-owner")
    adjacent.write_text("preexisting-owner")
    gate_inputs = {
        spec.key: {
            "classification": "VERIFIED", "passed": True,
            "evidence_refs": ["provenance/package_staging_verification.json"],
        }
        for spec in gates.DEFAULT_GATE_SPECS
    }
    facts = gates.WorkflowFacts(
        phase="CUDA_SMOKE",
        official_smoke_variants_completed=9,
        exploratory_smoke_variants_completed=1,
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
    )
    core.rollback_smoke_final_seal_failure(
        run_dir=run,
        preflight_run=runtime_case / "results/preflight",
        preflight_report={
            "training_table_path": "/source/train.csv",
            "training_table_sha256": "2" * 64,
            "r92_control": {"max_probability_difference": 0.0},
        },
        split_hash="3" * 64,
        model_bundle=None,
        gate_inputs=gate_inputs,
        facts=facts,
        smoke_metrics={},
        error=core.IntegrityError("collision"),
        ownership={},
    )
    assert canonical.read_bytes() == b"preexisting-owner"
    assert adjacent.read_text() == "preexisting-owner"


def test_smoke_rollback_cleans_owned_publications_before_report_write_failure(
    runtime_case, monkeypatch,
):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    run = runtime_case / "results/rollback_write_failure"
    for relative in ("provenance", "metrics", "logs"):
        (run / relative).mkdir(parents=True, exist_ok=True)
    for name in ("OUTPUT_MANIFEST_FINAL.tsv", "BUNDLE_MANIFEST.tsv"):
        (run / name).write_text("stale\n")
    canonical = runtime_case / "results/rollback_write_failure_review_bundle.zip"
    adjacent = runtime_case / "results/rollback_write_failure_package_verification.json"
    canonical.write_bytes(b"owned")
    adjacent.write_text("owned")
    ownership: dict[Path, core.PublicationOwnershipToken] = {}
    for path in (
        run / "OUTPUT_MANIFEST_FINAL.tsv", run / "BUNDLE_MANIFEST.tsv",
        canonical, adjacent,
    ):
        _mark_publication_owned(path, ownership)
    monkeypatch.setattr(
        core, "atomic_write_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk failure")),
    )
    facts = gates.WorkflowFacts(
        phase="CUDA_SMOKE",
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
    )
    with pytest.raises(OSError, match="disk failure"):
        core.rollback_smoke_final_seal_failure(
            run_dir=run,
            preflight_run=runtime_case / "results/preflight",
            preflight_report={
                "training_table_path": "/source/train.csv",
                "training_table_sha256": "2" * 64,
                "r92_control": {"max_probability_difference": 0.0},
            },
            split_hash="3" * 64,
            model_bundle=None,
            gate_inputs={
                spec.key: {
                    "classification": "VERIFIED", "passed": True,
                    "evidence_refs": ["evidence.txt"],
                }
                for spec in gates.DEFAULT_GATE_SPECS
            },
            facts=facts,
            smoke_metrics={},
            error=core.IntegrityError("seal failure"),
            ownership=ownership,
        )
    assert not canonical.exists()
    assert not adjacent.exists()
    assert not (run / "OUTPUT_MANIFEST_FINAL.tsv").exists()
    assert not (run / "BUNDLE_MANIFEST.tsv").exists()


def test_preflight_final_seal_failure_rolls_back_package_claims(
    runtime_case, monkeypatch,
):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    run = runtime_case / "results/failing_preflight"
    for relative in ("provenance", "logs"):
        (run / relative).mkdir(parents=True, exist_ok=True)
    (run / "config").mkdir(parents=True, exist_ok=True)
    (run / "config/run_identity.lock.json").write_text(json.dumps({
        "run_id": run.name,
        "run_kind": "AMENDED_PREFLIGHT",
        "state": "PREFLIGHT_COMPLETE",
        "permitted_transitions": [
            "CUDA_SMOKE_NON_SCIENTIFIC", "PACKAGE_REVIEW_EVIDENCE",
        ],
        "full_authorized": False,
        "stability_authorized": False,
    }))
    (run / "OUTPUT_MANIFEST_FINAL.tsv").write_text("stale")
    (run / "BUNDLE_MANIFEST.tsv").write_text("stale")
    canonical = runtime_case / "results/failing_preflight_review_bundle.zip"
    adjacent = runtime_case / "results/failing_preflight_package_verification.json"
    canonical.write_bytes(b"corrupted-after-publication")
    adjacent.write_text("corrupted-after-publication")
    ownership: dict[Path, core.PublicationOwnershipToken] = {}
    for path in (
        run / "OUTPUT_MANIFEST_FINAL.tsv", run / "BUNDLE_MANIFEST.tsv",
        canonical, adjacent,
    ):
        _mark_publication_owned(path, ownership)
    gate_inputs = {
        spec.key: {
            "classification": "VERIFIED", "passed": True,
            "evidence_refs": ["provenance/package_staging_verification.json"],
        }
        for spec in gates.DEFAULT_GATE_SPECS
    }
    facts = gates.WorkflowFacts(
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
    )
    report = {
        "training_table_path": "/source/train.csv",
        "training_table_sha256": "2" * 64,
        "candidate_split_hash": {"candidate_split_manifest_sha256": "3" * 64},
        "r92_control": {"max_probability_difference": 0.0},
        "historical_replay": {"literal_probability_tolerance_status": "PASS"},
    }
    state = builder._rollback_preflight_final_seal_failure(
        run_dir=run,
        gate_inputs=gate_inputs,
        facts=facts,
        report=report,
        error=core.IntegrityError("injected final seal failure"),
        ownership=ownership,
    )
    status = (run / "RUN_STATUS.txt").read_text()
    assert state.gate_statuses["output_manifest_zip_verification"] == "FAIL"
    assert state.run_state == "BLOCKED_PREFLIGHT_COMPLETE"
    assert state.smoke_eligible is False
    assert "CUDA_SMOKE_ELIGIBLE=NO" in status
    assert "AMENDED_PREFLIGHT_STATUS=BLOCKED_BEFORE_CUDA_SMOKE" in status
    assert "OUTPUT_MANIFEST_AND_ZIP_VERIFICATION=FAIL" in status
    assert "READY_FOR_FULL_AWAITING_EXTERNAL_REVIEW=NO" in status
    assert "REVIEW_BUNDLE_PATH=FINAL_SEAL_FAILED_NO_CANONICAL_BUNDLE" in status
    rolled_identity = json.loads(
        (run / "config/run_identity.lock.json").read_text()
    )
    assert rolled_identity["state"] == "BLOCKED_PREFLIGHT_COMPLETE"
    assert rolled_identity["permitted_transitions"] == [
        "PACKAGE_REVIEW_EVIDENCE",
    ]
    assert not (run / "OUTPUT_MANIFEST_FINAL.tsv").exists()
    assert not (run / "BUNDLE_MANIFEST.tsv").exists()
    assert not canonical.exists()
    assert not adjacent.exists()


def test_preflight_state_refresh_replaces_provisional_package_blocker(
    runtime_case, monkeypatch,
):
    run = runtime_case / "promoted_preflight"
    for relative in ("provenance", "docs"):
        (run / relative).mkdir(parents=True, exist_ok=True)
    gate_inputs = {
        spec.key: {
            "classification": "VERIFIED", "passed": True,
            "evidence_refs": ["evidence.txt"],
        }
        for spec in gates.DEFAULT_GATE_SPECS
    }
    gate_inputs["output_manifest_zip_verification"] = {
        "classification": "BLOCKED", "passed": None,
        "evidence_refs": ["provenance/package_staging_verification.json"],
    }
    for key in ("cuda_smoke_all_runnable", "model_save_reload"):
        gate_inputs[key] = {
            "classification": "BLOCKED", "passed": None,
            "evidence_refs": ["provenance/PREFLIGHT_REPORT.json"],
        }
    facts = gates.WorkflowFacts(
        phase="AMENDED_PREFLIGHT",
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
    )
    provisional = gates.derive_workflow_state(
        gates.build_gate_registry(gate_inputs), facts,
    )
    assert provisional.smoke_eligible is False
    gate_inputs["output_manifest_zip_verification"] = {
        "classification": "VERIFIED", "passed": True,
        "evidence_refs": ["provenance/package_staging_verification.json"],
    }
    promoted = gates.derive_workflow_state(
        gates.build_gate_registry(gate_inputs), facts,
    )
    assert promoted.smoke_eligible is True
    observed = {}
    monkeypatch.setattr(
        builder,
        "_render_docs",
        lambda *args: observed.update(state=args[-1]),
    )
    report = {
        "global_blockers": builder._blocker_details(provisional),
    }
    builder._refresh_preflight_state_artifacts(
        run_dir=run,
        report=report,
        state=promoted,
        historical_validation={},
        current_validation={},
        pca={},
        gate_frame=pd.DataFrame(),
        no_pca={"status": facts.no_pca_variant_status},
        variants={},
    )
    assert report["global_blockers"] == {}
    assert report["cuda_smoke"] == "NOT_RUN_AWAITING_EXECUTION"
    assert observed["state"] is promoted
    assert "- None." in (run / "BLOCKERS.md").read_text()


def test_preflight_rollback_cleans_owned_publications_before_report_write_failure(
    runtime_case, monkeypatch,
):
    monkeypatch.setattr(core, "STUDY_ROOT", runtime_case)
    run = runtime_case / "results/preflight_rollback_write_failure"
    for relative in ("provenance", "logs"):
        (run / relative).mkdir(parents=True, exist_ok=True)
    for name in ("OUTPUT_MANIFEST_FINAL.tsv", "BUNDLE_MANIFEST.tsv"):
        (run / name).write_text("stale\n")
    canonical = (
        runtime_case / "results/preflight_rollback_write_failure_review_bundle.zip"
    )
    adjacent = (
        runtime_case
        / "results/preflight_rollback_write_failure_package_verification.json"
    )
    canonical.write_bytes(b"owned")
    adjacent.write_text("owned")
    ownership: dict[Path, core.PublicationOwnershipToken] = {}
    for path in (
        run / "OUTPUT_MANIFEST_FINAL.tsv", run / "BUNDLE_MANIFEST.tsv",
        canonical, adjacent,
    ):
        _mark_publication_owned(path, ownership)
    monkeypatch.setattr(
        core, "atomic_write_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk failure")),
    )
    facts = gates.WorkflowFacts(
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
    )
    with pytest.raises(OSError, match="disk failure"):
        builder._rollback_preflight_final_seal_failure(
            run_dir=run,
            gate_inputs={
                spec.key: {
                    "classification": "VERIFIED", "passed": True,
                    "evidence_refs": ["evidence.txt"],
                }
                for spec in gates.DEFAULT_GATE_SPECS
            },
            facts=facts,
            report={
                "training_table_path": "/source/train.csv",
                "training_table_sha256": "2" * 64,
                "candidate_split_hash": {
                    "candidate_split_manifest_sha256": "3" * 64,
                },
                "r92_control": {"max_probability_difference": 0.0},
                "historical_replay": {
                    "literal_probability_tolerance_status": "PASS",
                },
            },
            error=core.IntegrityError("seal failure"),
            ownership=ownership,
        )
    assert not canonical.exists()
    assert not adjacent.exists()
    assert not (run / "OUTPUT_MANIFEST_FINAL.tsv").exists()
    assert not (run / "BUNDLE_MANIFEST.tsv").exists()


def test_immutable_run_directory_refuses_collision(runtime_case):
    target = runtime_case / "results/collision"
    core.create_immutable_run_directory(target)
    marker = target / "user_evidence.txt"
    marker.write_text("unchanged")
    with pytest.raises(core.IntegrityError, match="already exists"):
        core.create_immutable_run_directory(target)
    assert marker.read_text() == "unchanged"


def test_zip_embedded_manifest_hash_and_crc_verification(runtime_case):
    archive_path = runtime_case / "bundle.zip"
    payload = b"evidence\n"
    payload_hash = hashlib.sha256(payload).hexdigest()
    manifest = (
        "relative_path\tsize_bytes\tsha256\tartifact_role\tinclude_in_review_bundle\tinclude_in_models_bundle\n"
        f"evidence.txt\t{len(payload)}\t{payload_hash}\tprovenance\tTrue\tFalse\n"
    ).encode()
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("run/evidence.txt", payload)
        archive.writestr("run/OUTPUT_MANIFEST_FINAL.tsv", manifest)
    result = core.verify_zip(archive_path, embedded_run_manifest=True)
    assert result["status"] == "PASS"
    assert result["crc_status"] == "PASS"
    assert result["embedded_manifest_rows_checked"] == 1


def test_zip_headerless_three_column_manifest_verification(runtime_case):
    archive_path = runtime_case / "legacy_bundle.zip"
    payload = b"legacy evidence\n"
    manifest = f"evidence.txt\t{len(payload)}\t{hashlib.sha256(payload).hexdigest()}\n"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("run/evidence.txt", payload)
        archive.writestr("run/OUTPUT_MANIFEST_FINAL.tsv", manifest)
    result = core.verify_zip(archive_path, embedded_run_manifest=True)
    assert result["status"] == "PASS"
    assert result["embedded_manifest_schema"] == "HEADERLESS_3_COLUMNS"


def test_zip_manifest_tamper_detection(runtime_case):
    archive_path = runtime_case / "tampered.zip"
    manifest = (
        "relative_path\tsize_bytes\tsha256\tartifact_role\tinclude_in_review_bundle\tinclude_in_models_bundle\n"
        f"evidence.txt\t4\t{'0' * 64}\tprovenance\tTrue\tFalse\n"
    )
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("run/evidence.txt", b"real")
        archive.writestr("run/OUTPUT_MANIFEST_FINAL.tsv", manifest)
    result = core.verify_zip(archive_path, embedded_run_manifest=True)
    assert result["status"] == "FAIL"
    assert result["embedded_manifest_failures"]


def test_zip_duplicate_member_detection(runtime_case):
    archive_path = runtime_case / "duplicate.zip"
    with pytest.warns(UserWarning):
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("duplicate.txt", b"one")
            archive.writestr("duplicate.txt", b"two")
    result = core.verify_zip(archive_path)
    assert result["status"] == "FAIL"
    assert result["duplicate_member_count"] == 1


def test_no_pca_raw_cache_shape_and_status():
    array = np.load(core.RAW_EMBEDDINGS, mmap_mode="r")
    assert array.shape == (378753, 2048)
    assert str(array.dtype) == "float32"
    variants = core.build_variant_feature_sets()
    assert variants["no_pca"]["status"] == "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE"


def test_cuda_capability_is_not_mislabeled_pass():
    probe = core.xgboost_cuda_capability_probe()
    if not probe["use_cuda_compiled"]:
        assert probe["status"] == "FAIL"
        assert "USE_CUDA=false" in probe["failure_reason"]


def test_gpu_identity_metadata_records_name_driver_and_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "GPU-deadbeef")
    monkeypatch.setattr(core, "run_command", lambda *args, **kwargs: core.CommandResult(
        command=["nvidia-smi"], cwd=str(STUDY), returncode=0,
        stdout="0, NVIDIA GeForce RTX 4090, 595.95\n", stderr="", wall_seconds=0.01,
    ))
    monkeypatch.setattr(core.xgb, "build_info", lambda: {"USE_CUDA": True})

    metadata = core.gpu_identity_metadata()

    assert metadata["cuda_visible_devices"] == "0"
    assert metadata["nvidia_visible_devices"] == "GPU-deadbeef"
    assert metadata["devices"] == [{
        "physical_index": "0",
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "driver_version": "595.95",
    }]
    assert metadata["xgboost_cuda_compiled"] is True
