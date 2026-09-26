"""Read-only verification of a completed fixed-r92 public round."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .r92_images import _sha
from .r92_review import _full_inference_binding, _registry


def verify_round(project: Path, inference: Path, exported: Path, images: Path | None = None) -> dict:
    project = Path(project).resolve(strict=True)
    inference = Path(inference).resolve(strict=True)
    exported = Path(exported).resolve(strict=True)
    prepared = json.loads((project / "PROJECT_RECEIPT.json").read_text())
    manifest = json.loads((project / "PROJECT_MANIFEST.json").read_text())
    if (prepared.get("schema") != "compag-r92-image-project/v2"
            or prepared.get("status") != "PREPARED_ALL_TILES"
            or prepared.get("project_manifest_sha256") != _sha(project / "PROJECT_MANIFEST.json")
            or manifest.get("schema") != "compag-r92-full-project-manifest/v1"):
        raise ValueError("project receipt or manifest is incomplete")
    hashes = manifest.get("sha256_by_relative_path")
    if not isinstance(hashes, dict):
        raise ValueError("project file hash ledger is missing")
    expected_files = {"original_coco.json", "tiled_coco.json"}
    expected_files |= {f"originals/{r['original']}" for r in prepared["images"]}
    expected_files |= {f"tiles/{t['name']}" for r in prepared["images"] for t in r["tiles"]}
    if set(hashes) != expected_files:
        raise ValueError("project file hash ledger is incomplete")
    if ({f"originals/{p.name}" for p in (project / "originals").iterdir()}
            | {f"tiles/{p.name}" for p in (project / "tiles").iterdir()}) != expected_files - {"original_coco.json", "tiled_coco.json"}:
        raise ValueError("project image/tile inventory changed")
    for relative, digest in hashes.items():
        path = project / relative
        if not path.is_file() or path.is_symlink() or _sha(path) != digest:
            raise ValueError(f"project file changed: {relative}")
    if images is not None:
        images = Path(images).resolve(strict=True)
        if not images.is_dir():
            raise ValueError("source images path is not a directory")
        for row in prepared["images"]:
            source = images / row["original"]
            if not source.is_file() or source.is_symlink() or _sha(source) != row["original_sha256"]:
                raise ValueError(f"original source changed: {row['original']}")
    scored = inference / "scores" / "detections.csv"
    binding = _full_inference_binding(scored, project / "tiles")
    if binding is None or binding["path"] != str((inference / "FULL_INFERENCE_RECEIPT.json").resolve()):
        raise ValueError("full inference receipt is absent or unbound")
    full = json.loads((inference / "FULL_INFERENCE_RECEIPT.json").read_text())
    if _sha(inference / "features.csv") != full.get("feature_csv_sha256"):
        raise ValueError("full inference features changed")
    scores = json.loads((inference / "scores" / "SCORE_RECEIPT.json").read_text())
    if scores != full.get("score_receipt") or scores.get("row_count") != full.get("candidate_count"):
        raise ValueError("full inference score receipt differs")
    with scored.open(newline="", encoding="utf-8") as stream:
        scored_rows = list(csv.DictReader(stream))
    if len(scored_rows) != full["candidate_count"]:
        raise ValueError("scored candidate count differs")
    receipt = json.loads((exported / "EXPORT_RECEIPT.json").read_text())
    export_manifest = json.loads((exported / "EXPORT_MANIFEST.json").read_text())
    if (receipt.get("schema") != "compag-r92-review-export/v3" or receipt.get("status") != "PASS"
            or receipt.get("full_prepared_set") is not True
            or receipt.get("expected_tile_count") != prepared["tile_count"]
            or receipt.get("candidate_count") != full["candidate_count"]
            or receipt.get("scored_sha256") != _sha(scored)
            or receipt.get("full_inference_receipt_sha256") != binding["sha256"]
            or export_manifest.get("schema") != "compag-r92-export-manifest/v1"):
        raise ValueError("export does not match the completed full inference")
    export_hashes = export_manifest.get("sha256_by_file")
    if not isinstance(export_hashes, dict) or set(export_hashes) != {"original_coco.json", "tiled_coco.json", "new_orig_ids.json", "EXPORT_RECEIPT.json"}:
        raise ValueError("export hash ledger is incomplete")
    for name, digest in export_hashes.items():
        if _sha(exported / name) != digest:
            raise ValueError(f"export file changed: {name}")
    out_orig = _registry(exported / "original_coco.json")
    out_tiled = _registry(exported / "tiled_coco.json")
    if (out_orig["annotations"] != _registry(project / "original_coco.json")["annotations"]
            or len(out_tiled["images"]) != prepared["tile_count"]
            or len(out_tiled["annotations"]) != full["candidate_count"]):
        raise ValueError("export COCO accounting differs")
    return {"schema": "compag-r92-round-verification/v1", "status": "PASS",
            "image_count": prepared["image_count"], "prepared_tile_count": prepared["tile_count"],
            "processed_tile_count": full["processed_tile_count"], "candidate_count": full["candidate_count"],
            "reviewed_export_count": receipt["candidate_count"],
            "originals_preserved": True, "human_label_quality_assessed": False,
            "training_included": False}
