from __future__ import annotations

import json
import os
import sys
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

import amendment_inventory as inventory  # noqa: E402


def _write(path: Path, data: bytes = b"evidence\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _timestamped(root: Path, card: str, timestamp: str = "2025-12-28_00-00-00") -> Path:
    path = root / f"{card} ({timestamp})"
    path.mkdir(parents=True)
    return path


def _raw_cache_and_table(root: Path) -> tuple[Path, Path]:
    cache = root / "stage1_embeddings/train_foldA/embeddings_train_all.npy"
    cache.parent.mkdir(parents=True)
    np.save(cache, np.arange(24, dtype=np.float32).reshape(6, 4))
    rows = []
    for ann_id in (10, 20):
        for scale in (0.85, 1.0, 1.15):
            rows.append({"file_name": "IMG_0001_tile.jpg", "ann_id": ann_id, "scale": scale})
    table = root / "features_train.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    return cache, table


def _write_complete_sidecar(table: Path, path: Path, *, shuffle: bool = False) -> Path:
    frame = pd.read_csv(table)
    frame["cache_row_index"] = np.arange(len(frame), dtype=np.int64)
    if shuffle:
        frame = frame.iloc[[3, 0, 5, 2, 1, 4]].reset_index(drop=True)
    frame.to_csv(path, index=False)
    return path


def test_evidence_classification_is_exact_enum(tmp_path):
    record = inventory.inventory_record(
        _write(tmp_path / "source.txt"), logical_name="source", role="test",
    )
    assert record.evidence_classification == "VERIFIED"
    assert inventory.EVIDENCE_CLASSIFICATIONS == {"VERIFIED", "INFERRED", "UNRESOLVED", "BLOCKED"}
    with pytest.raises(ValueError, match="Invalid evidence classification"):
        inventory.InventoryRecord("bad", "bad", "test", False, None, None, "VERIFIED_FROM_SOURCE")


def test_timestamped_transform_and_prefix_inventory_is_bounded(tmp_path):
    before = _timestamped(tmp_path, "IMG_9428")
    boundary = _timestamped(tmp_path, "IMG_9431")
    after = _timestamped(tmp_path, "IMG_9432")
    for directory in (before, boundary, after):
        for name in inventory.TRANSFORM_ARTIFACT_NAMES:
            _write(directory / name, f"{directory.name}/{name}\n".encode())
        _write(directory / "features_train.csv", f"{directory.name}\n".encode())

    records = inventory.enumerate_transform_artifacts(tmp_path, through_card="IMG_9431")
    assert len(records) == 6
    assert {json.loads(record.to_dict()["metadata"])["card_id"] for record in records} == {
        "IMG_9428", "IMG_9431",
    }
    assert all("IMG_9432" not in record.path for record in records)
    prefixes = inventory.enumerate_prefix_csv_artifacts(tmp_path)
    assert len(prefixes) == 2
    assert all(record.sha256 == inventory.sha256_file(Path(record.path)) for record in records + prefixes)


def test_prefix_diagnostics_are_computed_from_bytes(tmp_path):
    previous = _write(tmp_path / "previous.csv", b"a,b\n1,2\n")
    anchor = _write(tmp_path / "anchor.csv", previous.read_bytes() + b"3,4\n")
    following = _write(tmp_path / "following.csv", anchor.read_bytes() + b"5,6\n")
    result = inventory.prefix_csv_diagnostics(previous, anchor, following)
    assert result["status"] == "PASS"
    assert result["previous_is_anchor_prefix"] and result["anchor_is_next_prefix"]
    following.write_bytes(b"different\n")
    assert inventory.prefix_csv_diagnostics(previous, anchor, following)["status"] == "FAIL"


def test_notebook_scale_extraction_uses_literal_source(tmp_path):
    notebook = {
        "cells": [
            {"cell_type": "code", "source": [
                "def make_cache():\n",
                "    SCALES_TRAIN = [0.7, 1.0, 1.3]\n",
                "    return SCALES_TRAIN\n",
            ]},
            {"cell_type": "code", "source": "MULTISCALE_SCALES = (0.5, 1, 1.5, 2)\n"},
            {"cell_type": "markdown", "source": "SCALES_TRAIN = [99]"},
        ]
    }
    path = tmp_path / "source.ipynb"
    path.write_text(json.dumps(notebook))
    result = inventory.extract_notebook_scale_evidence(path)
    assert result["raw_embedding_scales"]["values"] == [0.7, 1.0, 1.3]
    assert result["feature_table_scales"]["values"] == [0.5, 1.0, 1.5, 2.0]
    assert result["evidence_classification"] == "VERIFIED"
    assert all(row["evidence_classification"] == "VERIFIED" for row in result["assignments"])


def test_conflicting_notebook_assignments_are_unresolved(tmp_path):
    notebook = {"cells": [{"cell_type": "code", "source": [
        "SCALES_TRAIN = [0.8, 1.0]\n",
        "SCALES_TRAIN = [0.9, 1.0]\n",
        "MULTISCALE_SCALES = [1.0]\n",
    ]}]}
    path = tmp_path / "source.ipynb"
    path.write_text(json.dumps(notebook))
    result = inventory.extract_notebook_scale_evidence(path)
    assert result["raw_embedding_scales"]["values"] is None
    assert result["raw_embedding_scales"]["evidence_classification"] == "UNRESOLVED"


def test_one_row_schema_only_sidecar_cannot_qualify_six_row_cache(tmp_path):
    cache, _ = _raw_cache_and_table(tmp_path)
    keys = cache.parents[2] / "row_keys.csv"
    pd.DataFrame({
        "file_name": ["a"], "ann_id": [1], "scale": [1.0], "offset": [0],
    }).to_csv(keys, index=False)
    (cache.parents[2] / "used_img_ids.json").write_text(json.dumps(["IMG_0001"]))
    result = inventory.discover_stable_key_sidecars(
        cache, search_roots=[cache.parents[2]],
    )
    assert result["status"] != "VERIFIED_STABLE_KEY_SIDECAR_FOUND"
    assert result["qualifying_sidecar_count"] == 0
    assert len(result["inventory_records"]) == 2
    inspected = next(row for row in result["candidates"] if row["path"] == str(keys))
    assert inspected["sha256"] == inventory.sha256_file(keys)
    assert inspected["row_count"] == 1
    assert inspected["raw_cache_row_count"] == 6
    assert inspected["row_count_matches_raw_cache"] is False
    assert inspected["qualifies_as_stable_key_sidecar"] is False


def test_raw_cache_table_diagnostic_derives_shape_counts_and_columns(tmp_path):
    cache, table = _raw_cache_and_table(tmp_path)
    keys = _write_complete_sidecar(table, tmp_path / "row_keys.csv", shuffle=True)
    sidecar = inventory.discover_stable_key_sidecars(cache, search_roots=[tmp_path])
    alignment = tmp_path / "aligned_rows.npz"
    semantics = tmp_path / "embedding_semantics.json"
    semantics.write_text(json.dumps({"version": 1, "status": "TEST_MACHINE_EVIDENCE"}))
    result = inventory.raw_cache_table_diagnostics(
        cache,
        table,
        raw_scale_sequence=[0.85, 1.0, 1.15],
        expected_dimension_count=4,
        sidecar_diagnostic=sidecar,
        alignment_index_output=alignment,
        population="training",
        embedding_semantics_manifest_path=semantics,
        embedding_semantics_manifest_sha256=inventory.sha256_file(semantics),
    )
    assert result["status"] == "VERIFIED_EXACT_RAW_MAPPING_PREREQUISITES"
    assert result["candidate_deficit"] == 0
    assert result["raw_cache"]["shape"] == [6, 4]
    assert result["candidate_table"]["stable_key_duplicate_rows"] == 0
    assert result["actual_stable_key_join"]["mapped_candidate_rows"] == 6
    assert result["actual_stable_key_join"]["exact_join_rate"] == 1.0
    assert result["actual_stable_key_join"]["orphan_sidecar_rows"] == 0
    assert set(result["checks"]) == inventory.RAW_MAPPING_CHECK_KEYS
    assert result["checks"]["verified_embedding_semantics_manifest"] is True
    assert result["aligned_raw_layer_reference"]["status"] == "VERIFIED_ALIGNED_RAW_LAYER_REFERENCE"
    with np.load(alignment, allow_pickle=False) as payload:
        assert payload["cache_row_index"].tolist() == list(range(6))
    assert result["ordered_raw_feature_columns"] == [
        "raw_embed_0000", "raw_embed_0001", "raw_embed_0002", "raw_embed_0003",
    ]


def test_actual_sidecar_join_reports_partial_rate_and_blocks_wrong_key(tmp_path):
    cache, table = _raw_cache_and_table(tmp_path)
    keys = pd.read_csv(table)
    keys["cache_row_index"] = np.arange(len(keys), dtype=np.int64)
    keys.loc[0, "ann_id"] = 999
    keys.to_csv(tmp_path / "row_keys.csv", index=False)
    sidecar = inventory.discover_stable_key_sidecars(cache, search_roots=[tmp_path])
    blocked_alignment = tmp_path / "must_not_exist.npz"
    semantics = tmp_path / "embedding_semantics.json"
    semantics.write_text(json.dumps({"version": 1, "status": "TEST_MACHINE_EVIDENCE"}))
    result = inventory.raw_cache_table_diagnostics(
        cache,
        table,
        raw_scale_sequence=[0.85, 1.0, 1.15],
        expected_dimension_count=4,
        sidecar_diagnostic=sidecar,
        alignment_index_output=blocked_alignment,
        population="training",
        embedding_semantics_manifest_path=semantics,
        embedding_semantics_manifest_sha256=inventory.sha256_file(semantics),
    )
    actual = result["actual_stable_key_join"]
    assert result["status"] == "BLOCKED_RAW_MAPPING_PREREQUISITES"
    assert actual["mapped_candidate_rows"] == 5
    assert actual["unmatched_candidate_rows"] == 1
    assert actual["orphan_sidecar_rows"] == 1
    assert actual["exact_join_rate"] == pytest.approx(5 / 6)
    assert {"actual_table_join_complete", "actual_cache_key_coverage_complete"} <= set(
        result["failed_checks"]
    )
    assert not blocked_alignment.exists()
    assert result["aligned_raw_layer_reference"] is None


def test_raw_cache_table_diagnostic_blocks_real_mismatches(tmp_path):
    cache, table = _raw_cache_and_table(tmp_path)
    frame = pd.read_csv(table)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_csv(table, index=False)
    result = inventory.raw_cache_table_diagnostics(
        cache,
        table,
        raw_scale_sequence=[0.9, 1.0, 1.1],
        expected_dimension_count=8,
    )
    assert result["status"] == "BLOCKED_RAW_MAPPING_PREREQUISITES"
    assert {"expected_dimension_count", "scale_sequence_equal", "stable_table_key_unique", "qualifying_stable_key_sidecar"} <= set(result["failed_checks"])
    assert result["candidate_table"]["stable_key_duplicate_groups"] == 1


def _audit_fixture(root: Path, *, complete_masks: bool) -> tuple[Path, list[Path]]:
    root.mkdir(parents=True, exist_ok=True)
    audit = root / "audit.csv"
    pd.DataFrame([
        {"img_folder": "IMG_0001", "image": "shared.jpg", "id": 1},
        {"img_folder": "IMG_0002", "image": "shared.jpg", "id": 1},
    ]).to_csv(audit, index=False)
    paths = []
    for card in ("IMG_0001", "IMG_0002"):
        card_root = root / "revised_test" / card
        _write(card_root / "tiles/shared.jpg", b"image")
        path = card_root / "run_xgb_recall/detections.csv"
        row = {"image": "shared.jpg", "id": 1, "poly": "[[0,0],[1,0],[1,1]]", "rle": ""}
        if card == "IMG_0001":
            row["rle"] = json.dumps({"size": [2, 2], "counts": "04"})
        elif complete_masks:
            _write(card_root / "masks/shared_1.png", b"mask")
            row["mask_path"] = "masks/shared_1.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(path, index=False)
        paths.append(path)
    return audit, paths


def test_audit_detection_join_is_card_aware_and_reports_missing_masks(tmp_path):
    audit, paths = _audit_fixture(tmp_path, complete_masks=False)
    result = inventory.audit_detection_diagnostics(audit, paths)
    assert result["audit_keys_matched_card_aware"] == 2
    assert result["card_omission_collision_groups"] == 1
    assert result["audit_keys_with_source_tile_image"] == 2
    assert result["audit_keys_with_parseable_rle"] == 1
    assert result["audit_keys_with_reconstructible_crop_mask_input"] == 1
    assert result["raw_embedding_input_status"] == "BLOCKED_INCOMPLETE_RECONSTRUCTIBLE_INPUT"
    assert result["evidence_classification"] == "BLOCKED"


def test_audit_detection_explicit_mask_can_complete_inputs(tmp_path):
    audit, paths = _audit_fixture(tmp_path, complete_masks=True)
    result = inventory.audit_detection_diagnostics(audit, paths)
    assert result["audit_keys_with_existing_explicit_mask"] == 1
    assert result["audit_keys_with_reconstructible_crop_mask_input"] == 2
    assert result["raw_embedding_input_status"] == "VERIFIED_COMPLETE_RECONSTRUCTIBLE_INPUT"
    assert result["evidence_classification"] == "VERIFIED"


def test_detection_inventory_filters_by_audit_card_and_hashes_every_source(tmp_path):
    audit, paths = _audit_fixture(tmp_path, complete_masks=False)
    extra = tmp_path / "revised_test/IMG_0003/run_xgb_recall/detections.csv"
    extra.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"image": "extra.jpg", "id": 1}]).to_csv(extra, index=False)
    records = inventory.enumerate_audit_detection_artifacts(tmp_path, audit_csv=audit)
    assert {Path(record.path) for record in records} == {path.resolve() for path in paths}
    assert all(record.sha256 == inventory.sha256_file(Path(record.path)) for record in records)


def test_extended_inventory_and_pre_post_comparison(tmp_path):
    directory = _timestamped(tmp_path / "maskout", "IMG_9431")
    _write(directory / "embed_pca_32.npz")
    _write(directory / "features_train.csv")
    audit, _ = _audit_fixture(tmp_path / "audit_root", complete_masks=False)
    cache, _ = _raw_cache_and_table(tmp_path / "raw")
    records = inventory.build_extended_source_inventory(
        maskout_root=tmp_path / "maskout",
        audit_csv=audit,
        audit_detection_root=tmp_path / "audit_root",
        raw_cache_paths=[cache],
        sidecar_search_roots=[tmp_path / "raw"],
    )
    frame = inventory.inventory_frame(records)
    comparison = inventory.compare_inventory_frames(frame, frame.copy())
    assert comparison["status"] == "PASS"
    assert set(frame.evidence_classification) <= inventory.EVIDENCE_CLASSIFICATIONS
    changed = frame.copy()
    changed.loc[0, "sha256"] = "0" * 64
    assert inventory.compare_inventory_frames(frame, changed)["status"] == "FAIL"
