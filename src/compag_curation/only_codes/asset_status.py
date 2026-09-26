"""Read-only, stage-specific input inventory for an imported Only_codes project.

This is a preflight, not a scientific acceptance check.  It never imports a
pickle or creates a substitute for an absent fitted asset.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .project import LAYOUT, Project, SAM2_CKPT_REL, RESNET50_REL
from .state import sha256_file

_IMAGE = re.compile(r"IMG_[0-9]+\Z")
_REFERENCE_HASHES = {
    "sam2_checkpoint": "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
    "resnet50_weights": "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
}


def _item(role: str, path: Path, *, directory: bool = False,
          expected_reference_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path)
    result: dict[str, Any] = {"role": role, "path": str(path), "kind": "directory" if directory else "file"}
    if directory:
        if path.is_dir():
            result["file_count"] = sum(1 for p in path.iterdir() if p.is_file())
            result["status"] = "PRESENT" if result["file_count"] else "EMPTY"
        else:
            result["status"] = "MISSING"
        return result
    if not path.is_file() or path.is_symlink():
        result["status"] = "MISSING"
        return result
    result["size_bytes"] = path.stat().st_size
    result["sha256"] = sha256_file(path)
    result["status"] = "PRESENT" if result["size_bytes"] else "EMPTY"
    if expected_reference_sha256:
        result["reference_sha256"] = expected_reference_sha256
        result["reference_match"] = result["sha256"] == expected_reference_sha256
    return result


def _bundle(path: Path) -> dict[str, Any]:
    result = {"role": "verified_model_bundle", "path": str(path), "kind": "bundle", "status": "MISSING"}
    manifest_path = path / "LEGACY_MODEL_BUNDLE.json"
    if not manifest_path.is_file():
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "compag-only-codes-legacy-model-bundle/v1":
            raise ValueError("unexpected bundle schema")
        if manifest.get("verification", {}).get("status") != "PASS_EXACT":
            raise ValueError("bundle has no PASS_EXACT conversion receipt")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("bundle has no member hashes")
        if not {"classifier.ubj", "booster_config.json", "schema.json"}.issubset(files):
            raise ValueError("bundle is missing a classifier, configuration or ordered schema")
        for name, expected in files.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or len(expected) != 64:
                raise ValueError("unsafe bundle member entry")
            member = path / relative
            if not member.is_file() or member.is_symlink() or sha256_file(member) != expected:
                raise ValueError(f"bundle member absent or hash mismatch: {name}")
        schema = json.loads((path / "schema.json").read_text(encoding="utf-8"))
        features = schema.get("features")
        if (not isinstance(features, list) or not features
                or not all(isinstance(f, str) and f for f in features)
                or len(features) != len(set(features))):
            raise ValueError("ordered feature schema is missing or ambiguous")
        n_model = manifest.get("classifier", {}).get("n_features_in")
        if n_model is not None and int(n_model) != len(features):
            raise ValueError("classifier and feature schema dimensions differ")
        result.update(status="PRESENT_VERIFIED", feature_count=len(features),
                      feature_order_sha256=hashlib.sha256(json.dumps(features, separators=(",", ":")).encode()).hexdigest(),
                      member_count=len(files))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        result.update(status="INVALID", error=str(exc))
    return result


def inspect_project_assets(project: Project, *, image: str | None = None,
                           model_round_dir: str | None = None) -> dict[str, Any]:
    """Report prerequisites by stage without importing scientific libraries."""
    image = image or project.manifest.get("current_image")
    if image is not None and not _IMAGE.fullmatch(image):
        raise ValueError("image must have the form IMG_1234")
    state = project.state
    get = lambda key: state.path(LAYOUT[key])  # noqa: E731
    base = f"{LAYOUT['maskout_tile_root']}/{image}" if image else None
    detections = None
    if base:
        detections = next((state.path(f"{base}/run_{mode}/detections.csv")
                           for mode in ("xgb_recall", "xgb", "gate")
                           if state.path(f"{base}/run_{mode}/detections.csv").is_file()),
                          state.path(f"{base}/run_xgb_recall/detections.csv"))
    model_bundles = project.manifest.get("model_bundles", {})
    chosen = model_round_dir or (next(iter(model_bundles)) if len(model_bundles) == 1 else None)
    bundle_path = (project.root / model_bundles[chosen]["bundle_dir"]
                   if chosen in model_bundles else project.root / "bundles" / "SELECT_A_VERIFIED_MODEL")
    sam2_root = Path(project.manifest["roots"]["sam2_repo_root"])
    torch_home = Path(project.manifest["roots"]["torch_home"])
    config_candidates = [sam2_root / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml",
                         sam2_root / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"]
    sam2_config = next((p for p in config_candidates if p.is_file()), config_candidates[0])
    infer_proto = get("infer_proto_padded")
    if not infer_proto.is_file():
        infer_proto = get("infer_proto")
    stages: dict[str, list[dict[str, Any]]] = {
        "infer": [
            _item("prepared_image_tiles", state.path(f"{base}/tiles") if base else project.root / "SELECT_IMAGE", directory=True),
            _bundle(bundle_path),
            _item("inference_prototype", infer_proto),
            _item("inference_pca", get("infer_pca")),
            _item("sam2_checkpoint", sam2_root / SAM2_CKPT_REL,
                  expected_reference_sha256=_REFERENCE_HASHES["sam2_checkpoint"]),
            _item("sam2_config", sam2_config),
            _item("resnet50_weights", torch_home / RESNET50_REL,
                  expected_reference_sha256=_REFERENCE_HASHES["resnet50_weights"]),
            _item("yolo_weights", get("yolo_weights")),
        ],
        "review": [
            _item("detections", detections or project.root / "SELECT_IMAGE_DETECTIONS"),
            _item("prepared_image_tiles", state.path(f"{base}/tiles") if base else project.root / "SELECT_IMAGE", directory=True),
        ],
        "merge": [
            _item("original_coco", get("orig_coco")),
            _item("tiled_coco", get("tiled_coco")),
            _item("detections", detections or project.root / "SELECT_IMAGE_DETECTIONS"),
            _item("review_labels", state.path(f"{base}/review_labels.csv") if base else project.root / "SELECT_IMAGE_REVIEW"),
        ],
        "splits": [
            _item("original_coco", get("orig_coco")),
            _item("train_split_ids", state.path(f"{LAYOUT['split_dir']}/orig_train_ids.json")),
        ],
        "pack": [
            _item("tiled_coco", get("tiled_coco")),
            _item("training_tiles", get("tiles_dir"), directory=True),
            _item("train_split_ids", state.path(f"{LAYOUT['split_dir']}/orig_train_ids.json")),
            _item("resnet50_weights", torch_home / RESNET50_REL,
                  expected_reference_sha256=_REFERENCE_HASHES["resnet50_weights"]),
        ],
        "features": [
            _item("tiled_coco", get("tiled_coco")),
            _item("training_tiles", get("tiles_dir"), directory=True),
            _item("training_pack", state.path(f"{LAYOUT['art_root']}/stage2_proto/{project.manifest['fold_tag']}/foldsafe_pack.joblib")),
            _item("pack_used_image_ids", state.path(f"{LAYOUT['art_root']}/stage2_proto/{project.manifest['fold_tag']}/used_img_ids.json")),
        ],
        "train": [_item("training_feature_table", get("features_csv"))],
    }
    summary = {name: {"status": "READY_FOR_RUNTIME_PREFLIGHT" if all(
        x["status"] in ("PRESENT", "PRESENT_VERIFIED") for x in items) else "BLOCKED_INPUTS",
        "items": items} for name, items in stages.items()}
    return {"schema": "compag-only-codes-asset-status/v1", "project": str(project.root),
            "image": image, "model_round_dir": chosen, "stages": summary,
            "note": "File checks only; runtime compatibility, review completeness and numerical parity remain separate."}
