"""Fixed-r92 image inference using the existing canonical Full producer."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import sys
from pathlib import Path

from .r92_sample import _payloads, load_model, score_csv


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_progress(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def _mask_polygon(mask: object) -> list[int] | None:
    """Return display geometry, or defer a degenerate contour to the bbox bridge.

    The canonical mask, proposal and numerical features are independent of this
    approximation. A nonempty mask can occupy pixels yet have zero contour
    area (for example a one-pixel-wide component); it must stay in the pool.
    """
    import cv2
    import numpy as np

    binary = np.asarray(mask, dtype=np.uint8)
    if binary.shape != (512, 512):
        raise ValueError("SAM2 mask is not 512x512")
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("SAM2 mask is empty")
    contour = max(contours, key=cv2.contourArea)
    approximation = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
    if len(approximation) < 3 or cv2.contourArea(approximation) <= 0:
        return None
    return [int(v) for v in approximation.reshape(-1)]


def run_full(project: Path, model: Path, asset_root: Path, output: Path,
             *, project_model: Path | None = None, yolo_mode: str = "off",
             yolo_checkpoint: Path | None = None, yolo_sha256: str | None = None,
             yolo_boxes: Path | None = None, checkpoint_dir: Path | None = None) -> dict:
    """Process every registered canonical tile of every original image."""
    try:
        return _run(project, model, asset_root, output, full=True, project_model=project_model,
                    yolo_mode=yolo_mode, yolo_checkpoint=yolo_checkpoint, yolo_sha256=yolo_sha256,
                    yolo_boxes=yolo_boxes, checkpoint_dir=checkpoint_dir)
    except BaseException as exc:
        if checkpoint_dir is not None:
            progress_path = Path(checkpoint_dir).resolve() / "PROGRESS.json"
            if progress_path.is_file():
                progress = json.loads(progress_path.read_text())
                if progress.get("status") == "RUNNING" and progress.get("run_output") == str(Path(output).resolve()):
                    failure = {"tile": progress.get("active_tile"), "reason_type": type(exc).__name__,
                               "reason": str(exc)[:500], "completed_tile_count": progress.get("completed_tile_count")}
                    progress["status"] = "FAILED"
                    progress["failure_history"].append(failure)
                    _write_progress(progress_path, progress)
        raise


def run_subset(project: Path, model: Path, asset_root: Path, output: Path) -> dict:
    """Run real SAM2/ResNet/r92 on the one canonical tile per selected image.

    This is a declared Full-profile subset; it does not process every tile of
    either card and does not create human decisions or ground-truth labels.
    """
    return _run(project, model, asset_root, output, full=False, project_model=None,
                yolo_mode="off", yolo_checkpoint=None, yolo_sha256=None, yolo_boxes=None,
                checkpoint_dir=None)


def _run(project: Path, model: Path, asset_root: Path, output: Path, *, full: bool,
         project_model: Path | None, yolo_mode: str, yolo_checkpoint: Path | None,
         yolo_sha256: str | None, yolo_boxes: Path | None,
         checkpoint_dir: Path | None) -> dict:
    import cv2
    import numpy as np
    import torch
    from .assets import asset_registry
    from .canonical.features import CanonicalFeatureState, finalize_canonical_feature
    from .canonical.service import CanonicalRuntimeAssets, _generate_tile_records, _load_inference_runtime
    from .canonical.spec import CANONICAL_GPU_PROFILE
    from .r92_review import _check_registries, _preflight_rows, _registry
    from .canonical.preprocessing import canonical_axis_positions, canonical_tile_name

    project = Path(project).resolve(strict=True)
    model = Path(model).resolve(strict=True)
    asset_root = Path(asset_root).resolve(strict=True)
    output = Path(output).resolve()
    checkpoint_dir = Path(checkpoint_dir).resolve() if checkpoint_dir is not None else None
    if not torch.cuda.is_available():
        raise RuntimeError("Full-profile r92 image inference requires CUDA")
    if output.exists() or any(output == p or p in output.parents for p in (project, asset_root)):
        raise ValueError("output must be new and outside the project and asset cache")
    if checkpoint_dir is not None and (not full or checkpoint_dir == output or
            checkpoint_dir in output.parents or output in checkpoint_dir.parents or
            any(checkpoint_dir == p or p in checkpoint_dir.parents for p in (project, asset_root))):
        raise ValueError("checkpoint directory must be separate from output, project and assets")
    if yolo_mode not in {"off", "prompt", "fusion", "both"} or (not full and yolo_mode != "off"):
        raise ValueError("unsupported YOLO mode for this inference route")
    if yolo_mode == "off" and (yolo_checkpoint is not None or yolo_sha256 is not None or yolo_boxes is not None):
        raise ValueError("YOLO checkpoint supplied while mode is OFF")
    if yolo_mode != "off" and (yolo_checkpoint is None or yolo_sha256 is None):
        raise ValueError("requested YOLO mode requires an explicit checkpoint and SHA-256")
    inventory = json.loads((project / "PROJECT_RECEIPT.json").read_text())
    expected_schema = "compag-r92-image-project/v2" if full else "compag-r92-image-project/v1"
    if inventory.get("schema") != expected_schema or inventory.get("model_archive_sha256") != _sha(model):
        raise ValueError("image project and native model do not match")
    orig = _registry(project / "original_coco.json")
    tiled = _registry(project / "tiled_coco.json")
    if full:
        manifest_path = project / "PROJECT_MANIFEST.json"
        if _sha(manifest_path) != inventory.get("project_manifest_sha256"):
            raise ValueError("project manifest changed")
        manifest = json.loads(manifest_path.read_text())
        hashes = manifest.get("sha256_by_relative_path")
        if manifest.get("schema") != "compag-r92-full-project-manifest/v1" or not isinstance(hashes, dict):
            raise ValueError("full project manifest is malformed")
        registered = {"original_coco.json", "tiled_coco.json"}
        expected = {}
        if len(orig["images"]) != len(inventory["images"]) or len(tiled["images"]) != inventory.get("tile_count"):
            raise ValueError("full project registry counts differ")
        original_by_name = {im["file_name"]: im for im in orig["images"]}
        if len(original_by_name) != len(orig["images"]):
            raise ValueError("original names are duplicated")
        for row in inventory["images"]:
            name = row["original"]
            oi = original_by_name.get(name)
            if oi is None or (oi["width"], oi["height"]) != (row["width"], row["height"]):
                raise ValueError("original image metadata differs from receipt")
            registered.add(f"originals/{name}")
            expected_grid = {canonical_tile_name(name, x, y)
                             for y in canonical_axis_positions(row["warped_height"])
                             for x in canonical_axis_positions(row["warped_width"])}
            if len(row["tiles"]) != row["tile_count"] or {t["name"] for t in row["tiles"]} != expected_grid:
                raise ValueError("full project is missing canonical tiles")
            for tile in row["tiles"]:
                tile_name = tile["name"]
                if tile_name in expected:
                    raise ValueError("duplicate canonical tile")
                expected[tile_name] = tile
                registered.add(f"tiles/{tile_name}")
        if set(expected) != {im["file_name"] for im in tiled["images"]} or set(hashes) != registered:
            raise ValueError("full project tile registry or manifest is incomplete")
        actual_files = {f"originals/{p.name}" for p in (project / "originals").iterdir()}
        actual_files |= {f"tiles/{p.name}" for p in (project / "tiles").iterdir()}
        if actual_files != registered - {"original_coco.json", "tiled_coco.json"}:
            raise ValueError("project image or tile directory differs from manifest")
        for relative, digest in hashes.items():
            path = project / relative
            if not path.is_file() or path.is_symlink() or path.resolve().parent not in {
                    project, project / "originals", project / "tiles"} or _sha(path) != digest:
                raise ValueError(f"project file changed or unsafe: {relative}")
        for im in tiled["images"]:
            tile = expected[im["file_name"]]
            if (im["meta"]["offset_x"], im["meta"]["offset_y"]) != (tile["offset_x"], tile["offset_y"]):
                raise ValueError("tile registry offset differs from manifest")
        _check_registries([{"image": name} for name in sorted(expected)], orig, tiled, project / "original_coco.json")
    else:
        if not (len(orig["images"]) == len(tiled["images"]) == len(inventory["images"])):
            raise ValueError("image project registry counts differ")
        expected = {r["tile"]: r for r in inventory["images"]}
        if set(expected) != {im["file_name"] for im in tiled["images"]}:
            raise ValueError("project receipt does not cover every tile")
        for name, row in expected.items():
            if _sha(project / "tiles" / name) != row["tile_sha256"] or _sha(project / "originals" / row["original"]) != row["original_sha256"]:
                raise ValueError("project source or tile bytes changed")
    required = ("sam2.1-hiera-large-checkpoint", "sam2.1-hiera-large-config", "resnet50-imagenet1k-v2-weights",
                "sam2-apache-license", "torchvision-bsd-license")
    registry = asset_registry()
    asset_rows = []
    for asset_id in required:
        spec = registry[asset_id]
        path = asset_root / spec.filename
        if not path.is_file() or path.is_symlink() or path.stat().st_size != spec.size_bytes or _sha(path) != spec.sha256:
            raise ValueError(f"missing or changed official asset: {asset_id}")
        asset_rows.append({"asset_id": asset_id, "filename": spec.filename, "sha256": spec.sha256,
                           "size_bytes": spec.size_bytes, "url": spec.url, "source_commit": spec.source_commit})
    payloads = _payloads(model)
    if project_model is None:
        predictor = load_model(model)
    else:
        if not full:
            raise ValueError("project model requires the complete tiled route")
        from .r92_project_model import load_project_model
        predictor = load_project_model(project_model)
    state = CanonicalFeatureState(
        prototype=np.load(io.BytesIO(payloads["prototype.npy"]), allow_pickle=False),
        pca_mean=np.load(io.BytesIO(payloads["pca32_mean.npy"]), allow_pickle=False),
        pca_components=np.load(io.BytesIO(payloads["pca32_components.npy"]), allow_pickle=False),
        training_row_count=32, positive_row_count=1)
    assets = CanonicalRuntimeAssets(
        sam2_config=asset_root / registry["sam2.1-hiera-large-config"].filename,
        sam2_checkpoint=asset_root / registry["sam2.1-hiera-large-checkpoint"].filename,
        sam2_config_locator="configs/sam2.1/sam2.1_hiera_l",
        sam2_config_sha256=registry["sam2.1-hiera-large-config"].sha256,
        sam2_checkpoint_sha256=registry["sam2.1-hiera-large-checkpoint"].sha256,
        resnet50_weights=asset_root / registry["resnet50-imagenet1k-v2-weights"].filename,
        resnet50_weights_sha256=registry["resnet50-imagenet1k-v2-weights"].sha256,
        profile=CANONICAL_GPU_PROFILE)
    boxes_by_tile: dict[str, list[dict]] = {}
    boxes_attestation_sha256 = None
    if yolo_mode != "off":
        checkpoint = Path(yolo_checkpoint)
        if checkpoint.is_symlink() or not checkpoint.is_file() or _sha(checkpoint) != yolo_sha256:
            raise ValueError("requested YOLO checkpoint missing or changed")
        if yolo_boxes is not None:
            from .r92_yolo import load_precomputed_boxes
            boxes_by_tile = load_precomputed_boxes(yolo_boxes,project,yolo_sha256)
            boxes_attestation_sha256 = _sha(yolo_boxes)
        else:
            import importlib.util
            if importlib.util.find_spec("ultralytics") is None:
                raise ValueError("YOLO boxes receipt required in the science-GPU environment without Ultralytics")
            from .r92_yolo import predict_tile_boxes, verify_checkpoint
            detector = verify_checkpoint(yolo_checkpoint, yolo_sha256)
            for tile_image in tiled["images"]:
                name = tile_image["file_name"]
                boxes_by_tile[name] = predict_tile_boxes(detector, project / "tiles" / name)
            del detector
    checkpoint_identity = None
    checkpoint_reused_count = 0
    progress_path = None
    progress = None
    if checkpoint_dir is not None:
        checkpoint_identity = {
            "schema": "compag-r92-full-tile-checkpoint/v1",
            "project_receipt_sha256": _sha(project / "PROJECT_RECEIPT.json"),
            "project_manifest_sha256": _sha(project / "PROJECT_MANIFEST.json"),
            "native_model_sha256": _sha(model),
            "project_model_manifest_sha256": (None if project_model is None else
                _sha(Path(project_model) / "PROJECT_MODEL_MANIFEST.json")),
            "asset_sha256": {row["asset_id"]: row["sha256"] for row in asset_rows},
            "yolo_mode": yolo_mode, "yolo_checkpoint_sha256": yolo_sha256,
            "yolo_boxes_sha256": hashlib.sha256(json.dumps(
                boxes_by_tile, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "implementation_sha256": _sha(Path(__file__)),
        }
        identity_path = checkpoint_dir / "CHECKPOINT_IDENTITY.json"
        if checkpoint_dir.exists():
            if (not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink() or
                    not identity_path.is_file() or identity_path.is_symlink() or
                    json.loads(identity_path.read_text()) != checkpoint_identity):
                raise ValueError("checkpoint identity missing or changed")
        else:
            checkpoint_dir.mkdir(parents=True)
            identity_path.write_text(json.dumps(checkpoint_identity, indent=2) + "\n")
        progress_path = checkpoint_dir / "PROGRESS.json"
        if progress_path.is_symlink():
            raise ValueError("unsafe tile progress record")
        previous = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
        if previous and (previous.get("schema") != "compag-r92-tile-progress/v1" or
                not isinstance(previous.get("failure_history"), list)):
            raise ValueError("tile progress record changed")
        progress = {"schema": "compag-r92-tile-progress/v1", "status": "RUNNING",
                    "run_output": str(output), "expected_tile_count": len(tiled["images"]),
                    "completed_tile_count": 0, "active_tile": None,
                    "failure_history": list(previous.get("failure_history", []))}
        _write_progress(progress_path, progress)
    runtime = None
    fields = list(dict.fromkeys(["img_folder", "image", "id", "mask_sha256", "proposal_source", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "poly", *predictor.order]))
    feature_rows = []
    proposal_rows = []
    per_image = []
    for tile_index, tile_image in enumerate(tiled["images"], 1):
        name = tile_image["file_name"]
        print(f"r92 {'infer-full' if full else 'live-subset'}: tile {tile_index}/{len(tiled['images'])} {name}",
              file=sys.stderr, flush=True)
        if progress is not None:
            progress.update({"active_tile": name, "completed_tile_count": len(per_image)})
            _write_progress(progress_path, progress)
        cache_path = checkpoint_dir / f"tile_{tile_index:05d}.json" if checkpoint_dir is not None else None
        if cache_path is not None and cache_path.exists():
            if not cache_path.is_file() or cache_path.is_symlink():
                raise ValueError("unsafe tile checkpoint")
            cached = json.loads(cache_path.read_text())
            payload = cached.get("payload")
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
            if (cached.get("schema") != "compag-r92-tile-result/v1" or
                    cached.get("tile_index") != tile_index or cached.get("tile_name") != name or
                    cached.get("payload_sha256") != digest or not isinstance(payload, dict) or
                    not isinstance(payload.get("features"), list) or
                    not isinstance(payload.get("proposals"), list) or
                    not isinstance(payload.get("ledger"), dict) or
                    payload["ledger"].get("tile") != name or
                    payload["ledger"].get("proposal_rows") != len(payload["features"])):
                raise ValueError(f"tile checkpoint changed: {name}")
            feature_rows.extend(payload["features"])
            proposal_rows.extend(payload["proposals"])
            per_image.append(payload["ledger"])
            checkpoint_reused_count += 1
            if progress is not None:
                progress.update({"active_tile": None, "completed_tile_count": len(per_image)})
                _write_progress(progress_path, progress)
            continue
        if runtime is None:
            runtime = _load_inference_runtime(assets, "cuda", None)
        meta = tile_image["meta"]
        source = next(im for im in orig["images"] if im["id"] == meta["orig_image_id"])
        tile = project / "tiles" / name
        bgr = cv2.imread(str(tile), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (512, 512, 3):
            raise ValueError("project tile pixels changed")
        tile_row = {"image_id": str(source["id"]), "image_sha256": _sha(project / "originals" / source["file_name"]),
                    "image_name": source["file_name"], "group_id": source["file_name"], "tile_name": name,
                    "tile_sha256": _sha(tile), "x": str(meta["offset_x"]), "y": str(meta["offset_y"]),
                    "crop_w": "512", "crop_h": "512", "orig_w": str(source["width"]), "orig_h": str(source["height"])}
        image_record = {"row_lines": meta["row_lines"], "column_lines": meta["column_lines"],
                        "inverse_warp": meta["inverse_matrix"], "source_height": source["height"],
                        "source_width": source["width"], "warped_width": meta["warped_width"],
                        "warped_height": meta["warped_height"]}
        extras = []
        if yolo_mode in {"prompt", "both"}:
            from .r92_yolo import prompt_sam2
            extras = prompt_sam2(runtime, bgr, boxes_by_tile[name])
        generated = _generate_tile_records(runtime, bgr, tile_row, image_record,
                                           extra_annotations=extras)
        if extras:
            from .canonical.proposals import canonical_mask_sha256
            prompted_hashes = {canonical_mask_sha256(item["segmentation"]) for item in extras}
        else:
            prompted_hashes = set()
        amg_count = generated.selection.input_count - len(extras)
        provenance = {proposal.mask_sha256:
            ("YOLO_PROMPT" if proposal.source_index >= amg_count else
             "AMG_WITH_MATCHING_PROMPT" if proposal.mask_sha256 in prompted_hashes else "AMG")
            for proposal in generated.selection.proposals}
        tile_proposals = ([{**r, "proposal_source": provenance[r["mask_sha256"]]}
                           for r in generated.proposal_rows] if full else [])
        proposal_rows.extend(tile_proposals)
        by_id = {proposal.proposal_index: proposal for proposal in generated.selection.proposals}
        rows_before = len(feature_rows)
        polygon_bbox_fallback_count = 0
        for raw in generated.raw_features:
            if raw.scale != 1.0:
                continue
            proposal = by_id[raw.proposal_index]
            polygon = _mask_polygon(proposal.mask)
            x, y, w, h = proposal.bbox
            if polygon is None:
                if not all(isinstance(v, (int, np.integer)) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                    raise ValueError("degenerate SAM2 contour has no valid mask bbox fallback")
                polygon_bbox_fallback_count += 1
            features = finalize_canonical_feature(raw, state)
            if tuple(features) != predictor.order:
                raise ValueError("canonical feature order differs from r92")
            feature_rows.append({"img_folder": source["file_name"], "image": name,
                                 "id": str(proposal.proposal_index), "mask_sha256": proposal.mask_sha256,
                                 "proposal_source": provenance[proposal.mask_sha256],
                                 "bbox_x": str(x), "bbox_y": str(y),
                                 "bbox_w": str(w), "bbox_h": str(h),
                                 "poly": "" if polygon is None else json.dumps(polygon),
                                 **{key: repr(value) for key, value in features.items()}})
        ledger = {"original": source["file_name"], "tile": name, "proposal_rows": len(feature_rows)-rows_before,
                          "polygon_bbox_fallback_count": polygon_bbox_fallback_count,
                          "sam2_input_count": generated.selection.input_count,
                          "sam2_empty_count": generated.selection.empty_count,
                          "sam2_duplicate_count": generated.selection.duplicate_count,
                          "sam2_capped_count": generated.selection.capped_count}
        per_image.append(ledger)
        if cache_path is not None:
            payload = {"features": feature_rows[rows_before:], "proposals": tile_proposals, "ledger": ledger}
            cached = {"schema": "compag-r92-tile-result/v1", "tile_index": tile_index,
                      "tile_name": name, "payload": payload,
                      "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True,
                          separators=(",", ":")).encode()).hexdigest()}
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(cached, separators=(",", ":")) + "\n")
            temporary.replace(cache_path)
            progress.update({"active_tile": None, "completed_tile_count": len(per_image)})
            _write_progress(progress_path, progress)
    if len(per_image) != len(tiled["images"]) or {r["tile"] for r in per_image} != set(expected):
        raise RuntimeError("image inference did not process every registered tile")
    if yolo_boxes is not None and _sha(yolo_boxes) != boxes_attestation_sha256:
        raise ValueError("precomputed YOLO boxes changed during inference")
    if not feature_rows:
        raise ValueError("SAM2 produced no eligible candidates in selected tiles")
    output.mkdir(parents=True)
    try:
        features_path = output / "features.csv"
        with features_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(feature_rows)
        if full:
            proposals_path = output / "proposals.csv"
            with proposals_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(proposal_rows[0]))
                writer.writeheader()
                writer.writerows(proposal_rows)
        if yolo_mode != "off":
            (output / "YOLO_BOXES.json").write_text(json.dumps(boxes_by_tile, separators=(",", ":")) + "\n")
        scored = output / "scores"
        if project_model is None:
            score_receipt = score_csv(model, features_path, scored)
        else:
            from .r92_project_model import score_project_csv
            score_receipt = score_project_csv(project_model, features_path, scored)
        if yolo_mode in {"fusion", "both"}:
            from .r92_yolo import apply_fusion
            score_receipt = apply_fusion(scored / "detections.csv", scored / "SCORE_RECEIPT.json",
                                         boxes_by_tile, yolo_mode)
        _preflight_rows(scored / "detections.csv", project / "tiles")
        # The bridge's COCO registry checks are exercised before claiming this
        # output can enter review/export.
        _check_registries(list(csv.DictReader((scored/"detections.csv").open(newline="",encoding="utf-8"))),
                          orig, tiled, project/"original_coco.json")
        receipt = {"schema": "compag-r92-infer-full/v1" if full else "compag-r92-live-subset/v1",
                   "status": "PASS",
                   "scope": ("all canonical Full-profile 512x512 tiles of every warped card" if full else
                             "one canonical Full-profile 512x512 tile per selected image after warp"),
                   "full_prepared_set": full, "full_card": full,
                   "historical_bitwise_parity_claim": False,
                   "proposal_csv_sha256": _sha(output / "proposals.csv") if full else None,
                   "r92_model_archive_sha256": _sha(model), "feature_csv_sha256": _sha(features_path),
                   "project_path": str(project),
                   "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
                   "checkpoint_reused_count": checkpoint_reused_count,
                   "failed_tile_history": progress["failure_history"] if progress is not None else [],
                   "failed_tile_count": 0, "excluded_tile_count": 0,
                   "yolo_mode": yolo_mode, "yolo_checkpoint_sha256": yolo_sha256,
                   "yolo_execution": "DISABLED" if yolo_mode == "off" else "COMPLETED_ALL_TILES_CPU",
                   "yolo_boxes_per_tile": {k: len(v) for k, v in boxes_by_tile.items()},
                   "yolo_boxes_sha256": _sha(output / "YOLO_BOXES.json") if yolo_mode != "off" else None,
                   "yolo_boxes_attestation_sha256": boxes_attestation_sha256,
                   "scoring_model_kind": "NATIVE_R92" if project_model is None else "PROJECT_XGB",
                   "project_model_manifest_sha256": (None if project_model is None else
                       _sha(Path(project_model) / "PROJECT_MODEL_MANIFEST.json")),
                   "score_receipt": score_receipt, "candidate_count": len(feature_rows),
                   "polygon_bbox_fallback_count": sum(r["polygon_bbox_fallback_count"] for r in per_image),
                   "image_count": len(orig["images"]), "expected_tile_count": len(tiled["images"]),
                   "processed_tile_count": len(per_image), "tiles": per_image, "assets": asset_rows,
                   "project_receipt_sha256": _sha(project / "PROJECT_RECEIPT.json"),
                   "original_coco_sha256": _sha(project / "original_coco.json"),
                   "tiled_coco_sha256": _sha(project / "tiled_coco.json"),
                   "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "mask_geometry": "largest_external_contour_approx_1px; empty poly uses mask bbox for display only; machine proposal, not human boundary",
                   "human_labels": 0}
        (output / ("FULL_INFERENCE_RECEIPT.json" if full else "LIVE_SUBSET_RECEIPT.json")).write_text(json.dumps(receipt, indent=2) + "\n")
        if progress is not None:
            progress.update({"status": "COMPLETE", "active_tile": None,
                             "completed_tile_count": len(per_image)})
            _write_progress(progress_path, progress)
        return receipt
    except Exception:
        shutil.rmtree(output)
        raise
