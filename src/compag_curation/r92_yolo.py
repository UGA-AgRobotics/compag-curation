"""Optional, completeness-certified one-class CJ detector for project rounds."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path


SEGMENT_ULTRALYTICS_VERSION = "8.4.26"
PAPER_HYBRID_POLICY = "paper_hybrid_reject_missing_v1"
LEGACY_FUSION_POLICY = "retained_Only_codes_review_hybrid_ignore_missing_v1"
SEGMENT_CONFIG = {
    "agnostic_nms": False,
    "class_mapping": {"0": "CJ"},
    "confidence": 0.15,
    "half": False,
    "image_size": 512,
    "iou": 0.7,
    "max_detections": 1000,
    "rect": True,
    "retina_masks": True,
    "task": "segment",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def prepare_yolo_dataset(project: Path, qa: Path, output: Path) -> dict:
    """Import exhaustive per-tile target QA; selected AL labels are insufficient."""
    import cv2

    project, qa, output = map(Path, (project, qa, output))
    if output.exists():
        raise FileExistsError("YOLO dataset output exists")
    receipt = _json(project / "PROJECT_RECEIPT.json")
    if receipt.get("schema") != "compag-r92-image-project/v2":
        raise ValueError("YOLO QA requires a complete prepared image project")
    project_manifest_path = project / "PROJECT_MANIFEST.json"
    if _sha(project_manifest_path) != receipt.get("project_manifest_sha256"):
        raise ValueError("prepared image project manifest changed")
    project_manifest = _json(project_manifest_path)
    if project_manifest.get("schema") != "compag-r92-full-project-manifest/v1":
        raise ValueError("prepared project manifest schema differs")
    registry = _json(project / "tiled_coco.json")
    tile_map = {im["file_name"]: im for im in registry["images"]}
    if len(tile_map) != receipt["tile_count"]:
        raise ValueError("prepared tile inventory is incomplete")
    if {"tiles/" + name for name in tile_map} != {
            name for name in project_manifest.get("sha256_by_relative_path", {}) if name.startswith("tiles/")}:
        raise ValueError("prepared tile manifest inventory differs")
    source = _json(qa)
    if source.get("schema") != "compag-r92-yolo-complete-tile-qa/v1" or source.get("target_class") != "CJ":
        raise ValueError("QA schema or target class differs")
    if source.get("project_receipt_sha256") != _sha(project / "PROJECT_RECEIPT.json"):
        raise ValueError("YOLO QA belongs to a different prepared project")
    if source.get("annotation_origin") not in {"human", "test_fixture"}:
        raise ValueError("annotation origin must be explicit")
    units = source.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("YOLO QA must list complete image/tile units")
    seen, groups, standardized = set(), {}, []
    for unit in units:
        name = unit.get("tile")
        if name not in tile_map or name in seen:
            raise ValueError("YOLO QA references absent or duplicate tile")
        seen.add(name)
        image = tile_map[name]
        group = image["meta"]["orig_file_name"]
        role = unit.get("split_role")
        if role not in {"train", "val", "test"}:
            raise ValueError("YOLO QA requires an explicit split role")
        if group in groups and groups[group] != role:
            raise ValueError("all tiles from a source card must share one split")
        groups[group] = role
        if unit.get("complete_for") != "all_CJ_targets_in_512x512_tile" or not unit.get("certified_by"):
            raise ValueError("tile target annotation is not explicitly complete")
        tile_path = project / "tiles" / name
        if tile_path.is_symlink() or not tile_path.is_file() or _sha(tile_path) != unit.get("tile_sha256"):
            raise ValueError("YOLO QA tile bytes changed")
        if unit["tile_sha256"] != project_manifest["sha256_by_relative_path"]["tiles/" + name]:
            raise ValueError("YOLO QA tile differs from prepared manifest")
        pixels = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
        if pixels is None or pixels.shape[:2] != (512, 512):
            raise ValueError("YOLO tile image dimensions differ")
        targets = unit.get("targets")
        if not isinstance(targets, list):
            raise ValueError("target list must be explicit, including empty lists")
        ids, labels = set(), []
        for target in targets:
            object_id = target.get("physical_object_id")
            box = target.get("bbox_xyxy")
            if not isinstance(object_id, str) or not object_id or object_id in ids:
                raise ValueError("missing/duplicate physical object ID within a tile")
            ids.add(object_id)
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("CJ target requires an xyxy box")
            x1, y1, x2, y2 = map(float, box)
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or not (0 <= x1 < x2 <= 512 and 0 <= y1 < y2 <= 512):
                raise ValueError("CJ target box is degenerate or out of bounds")
            for prior in labels:
                px1, py1, px2, py2 = prior
                overlap = max(0.0, min(x2, px2)-max(x1, px1))*max(0.0, min(y2, py2)-max(y1, py1))
                union = (x2-x1)*(y2-y1)+(px2-px1)*(py2-py1)-overlap
                if union and overlap/union >= 0.8:
                    raise ValueError("duplicated physical target boxes need QA correction")
            labels.append((x1, y1, x2, y2))
        standardized.append((name, role, group, labels))
    if "train" not in groups.values() or "val" not in groups.values():
        raise ValueError("YOLO fitting requires independent train and validation cards")
    output.mkdir(parents=True)
    try:
        for role in {"train", "val", "test"}:
            (output / "images" / role).mkdir(parents=True)
            (output / "labels" / role).mkdir(parents=True)
        for name, role, _, labels in standardized:
            shutil.copyfile(project / "tiles" / name, output / "images" / role / name)
            lines = []
            for x1, y1, x2, y2 in labels:
                lines.append(f"0 {(x1+x2)/1024:.8f} {(y1+y2)/1024:.8f} {(x2-x1)/512:.8f} {(y2-y1)/512:.8f}")
            (output / "labels" / role / (Path(name).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        yaml = f"path: {output.resolve()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: {{0: CJ}}\n"
        if "test" in groups.values():
            yaml += "test: images/test\n"
        (output / "data.yaml").write_text(yaml)
        files = {str(p.relative_to(output)): _sha(p) for p in output.rglob("*") if p.is_file()}
        manifest = {"schema": "compag-r92-yolo-dataset/v1", "status": "PASS",
            "task": "detect", "class_map": {"XGB": {"0": "non-CJ", "1": "CJ"},
            "COCO": {"1": "CJ", "2": "non-CJ"}, "YOLO": {"0": "CJ"}},
            "annotation_origin": source["annotation_origin"], "qa_sha256": _sha(qa),
            "project_receipt_sha256": _sha(project / "PROJECT_RECEIPT.json"),
            "split_by_original_card": groups, "complete_tile_count": len(units),
            "positive_box_count": sum(len(r[3]) for r in standardized),
            "empty_target_tile_count": sum(not r[3] for r in standardized),
            "files": files}
        (output / "DATASET_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return {k: manifest[k] for k in ("schema", "status", "complete_tile_count", "positive_box_count", "empty_target_tile_count")}
    except Exception:
        shutil.rmtree(output)
        raise


def verify_yolo_dataset(root: Path) -> dict:
    root = Path(root).resolve(strict=True)
    manifest = _json(root / "DATASET_MANIFEST.json")
    if manifest.get("schema") != "compag-r92-yolo-dataset/v1" or manifest.get("status") != "PASS":
        raise ValueError("YOLO dataset manifest differs")
    files = manifest["files"]
    if set(files) != {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "DATASET_MANIFEST.json"}:
        raise ValueError("YOLO dataset inventory changed")
    for relative, digest in files.items():
        path = root / relative
        if path.is_symlink() or _sha(path) != digest:
            raise ValueError("YOLO dataset file changed")
    return manifest


def train_yolo(dataset: Path, initial_weights: Path, output: Path, *, epochs: int,
               batch: int, device: str | int, imgsz: int = 512) -> dict:
    """Perform a real local optimization through the retained trainer."""
    from .training.yolo import YoloTrainingRequest, train_yolo_local
    from .contracts import HandlerServices, _hash_tree
    import ultralytics
    import platform
    import torch
    import cv2
    import numpy as np

    if ultralytics.__version__ != "8.3.221":
        raise ValueError("optional YOLO profile requires Ultralytics 8.3.221")
    dataset, initial_weights, output = map(Path, (dataset, initial_weights, output))
    manifest = verify_yolo_dataset(dataset)
    if output.exists():
        raise FileExistsError("YOLO training output exists")
    stage = output.with_name(output.name + ".input-stage")
    if stage.exists() or dataset == stage or dataset in stage.parents:
        raise FileExistsError("YOLO private training stage already exists or overlaps its source")
    if not initial_weights.is_file() or initial_weights.is_symlink():
        raise ValueError("explicit local YOLO initialization weights required")
    from ultralytics import YOLO
    initial = YOLO(str(initial_weights))
    random_init = initial_weights.suffix.lower() in {".yaml", ".yml"}
    if initial.task != "detect" or len(initial.names) != 1 or (not random_init and str(initial.names[0]).lower() != "cj"):
        raise ValueError("YOLO initialization must be one-class CJ detection or explicit one-class YAML")
    # Ultralytics writes labels/*.cache during optimization. Preserve the
    # complete-QA snapshot by training on a private, explicit copy.
    shutil.copytree(dataset,stage)
    original_yaml=(dataset/"data.yaml").read_text()
    if not original_yaml.startswith(f"path: {dataset.resolve()}\n"):
        raise ValueError("sealed YOLO data YAML does not name its dataset root")
    (stage/"data.yaml").write_text(original_yaml.replace(
        f"path: {dataset.resolve()}\n",f"path: {stage.resolve()}\n",1))
    stage_initial_sha256=_hash_tree(stage)
    request = YoloTrainingRequest(data_yaml=stage / "data.yaml", dataset_root=stage,
        dataset_root_sha256=stage_initial_sha256, initial_weights=initial_weights,
        initial_weights_sha256=_sha(initial_weights), output_root=output,
        run_name="compag_cj", device=device, imgsz=imgsz, epochs=epochs, batch=batch)
    # HandlerServices only reports completion; all scientific output comes from the real backend.
    service = HandlerServices(report=lambda message: None)
    result = train_yolo_local(request, service)
    checkpoint = result.save_directory / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise RuntimeError("YOLO trainer did not save best.pt")
    verified = YOLO(str(checkpoint))
    if verified.task != "detect" or len(verified.names) != 1 or str(verified.names[0]).lower() != "cj":
        raise RuntimeError("trained YOLO checkpoint task/class map differs")
    val_images = sorted((dataset / "images" / "val").iterdir())
    if not val_images:
        raise RuntimeError("no validation image for trained-checkpoint prediction")
    predictions = verified.predict(source=str(val_images[0]), device=device, imgsz=imgsz, verbose=False)
    if len(predictions) != 1:
        raise RuntimeError("trained YOLO checkpoint cannot predict")
    verify_yolo_dataset(dataset)
    backend_args = result.save_directory / "args.yaml"
    if not backend_args.is_file():
        raise RuntimeError("YOLO trainer did not save backend execution arguments")
    record = {"schema": "compag-r92-yolo-project-model/v1", "status": "PASS",
        "task": "detect", "class_map": {"0": "CJ"}, "training_kind": "RANDOM_INIT" if random_init else "FINE_TUNE_NEW_ROUND",
        "dataset_manifest_sha256": _sha(dataset / "DATASET_MANIFEST.json"),
        "private_training_stage":str(stage.resolve()),
        "private_training_stage_initial_sha256":stage_initial_sha256,
        "initial_weights_sha256": _sha(initial_weights), "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha(checkpoint), "epochs": epochs, "batch": batch,
        "imgsz": imgsz, "device": device, "seed": request.seed,
        "ultralytics_version": ultralytics.__version__,
        "prediction_probe_image_sha256": _sha(val_images[0]),
        "prediction_probe_box_count": len(predictions[0].boxes),
        "split_by_original_card": manifest["split_by_original_card"],
        "optimizer_policy": "ultralytics_auto",
        "backend_args_sha256": _sha(backend_args),
        "augmentation": {"close_mosaic":request.close_mosaic,"hsv_h":request.hsv_h,
            "hsv_s":request.hsv_s,"hsv_v":request.hsv_v,"translate":request.translate,
            "scale":request.scale,"shear":request.shear,"perspective":request.perspective},
        "checkpoint_selection_rule": "Ultralytics best.pt from validation after requested epochs",
        "dependency_versions": {"ultralytics":ultralytics.__version__,"torch":torch.__version__,
            "numpy":np.__version__,"opencv":cv2.__version__},
        "hardware": {"platform":platform.platform(),"cpu":platform.processor(),
            "cuda_available":torch.cuda.is_available(),
            "cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "annotation_origin": manifest["annotation_origin"],
        "scientific_performance_claim": False}
    (output / "YOLO_PROJECT_MODEL.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _segment_package_contract(package: Path, checkpoint: Path, sha256: str) -> dict:
    """Bind the supplied segmentation weight to its reviewed package settings."""
    if Path(package).is_symlink():
        raise ValueError("YOLO model package must not be a link")
    package = Path(package).resolve(strict=True)
    if not package.is_dir():
        raise ValueError("YOLO model package must be a real directory")
    expected = package / "weights" / "best.pt"
    if checkpoint.resolve(strict=True) != expected.resolve(strict=True):
        raise ValueError("segmentation checkpoint is outside its model package")
    manifest_path = package / "MODEL_MANIFEST.json"
    config_path = package / "inference" / "inference_config.json"
    if any(p.is_symlink() or not p.is_file() for p in (expected, manifest_path, config_path)):
        raise ValueError("YOLO model package contains a missing or linked required file")
    manifest, config = _json(manifest_path), _json(config_path)
    if (manifest.get("checkpoint") != "weights/best.pt"
            or manifest.get("checkpoint_sha256") != sha256
            or manifest.get("task") != "instance segmentation"
            or manifest.get("class_mapping") != {"0": "CJ"}
            or manifest.get("verified_runtime", {}).get("ultralytics") != SEGMENT_ULTRALYTICS_VERSION
            or config != SEGMENT_CONFIG):
        raise ValueError("segmentation package manifest or inference settings differ")
    return {"model_package_root": str(package),
            "model_manifest_sha256": _sha(manifest_path),
            "inference_config_sha256": _sha(config_path),
            "inference_config": config}


def verify_checkpoint(path: Path, sha256: str, *, model_package: Path | None = None) -> object:
    path = Path(path)
    if not path.is_file() or path.is_symlink() or _sha(path) != sha256:
        raise ValueError("requested YOLO checkpoint missing or changed")
    import ultralytics
    from ultralytics import YOLO
    if model_package is not None:
        _segment_package_contract(model_package, path, sha256)
        expected_version = SEGMENT_ULTRALYTICS_VERSION
        expected_task = "segment"
    else:
        expected_version = "8.3.221"
        expected_task = "detect"
    if ultralytics.__version__ != expected_version:
        raise ValueError(f"YOLO checkpoint verification requires Ultralytics {expected_version}; "
                         "pass --model-package for the external segmentation checkpoint")
    model = YOLO(str(path))
    if model.task != expected_task or {int(k): str(v) for k, v in model.names.items()} != {0: "CJ"}:
        raise ValueError(f"requested YOLO checkpoint is not one-class CJ {expected_task}")
    return model


def verify_checkpoint_record(checkpoint: Path, sha256: str, output: Path,
                             *, model_package: Path | None = None) -> dict:
    """Run in the optional detector environment before retaining old weights."""
    checkpoint, output = Path(checkpoint).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("YOLO checkpoint attestation output exists")
    model = verify_checkpoint(checkpoint, sha256, model_package=model_package)
    record = {"schema":"compag-r92-yolo-checkpoint-attestation/v2" if model_package else
                       "compag-r92-yolo-checkpoint-attestation/v1","status":"PASS",
              "checkpoint":str(checkpoint),"checkpoint_sha256":sha256,
              "task":model.task,"class_map":{"0":"CJ"},
              "ultralytics_version":SEGMENT_ULTRALYTICS_VERSION if model_package else "8.3.221"}
    if model_package:
        record.update(_segment_package_contract(model_package, checkpoint, sha256))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(record,indent=2)+"\n")
    return record


def verify_segment_attestation(record: dict, checkpoint: Path, sha256: str) -> None:
    """Check an external segment attestation without importing Ultralytics."""
    if (record.get("schema") != "compag-r92-yolo-checkpoint-attestation/v2"
            or record.get("status") != "PASS"
            or record.get("checkpoint_sha256") != sha256
            or record.get("checkpoint") != str(Path(checkpoint).resolve(strict=True))
            or record.get("task") != "segment"
            or record.get("class_map") != {"0": "CJ"}
            or record.get("ultralytics_version") != SEGMENT_ULTRALYTICS_VERSION):
        raise ValueError("external segmentation attestation differs")
    package = record.get("model_package_root")
    if not isinstance(package, str) or not package:
        raise ValueError("external segmentation package is absent")
    contract = _segment_package_contract(Path(package), checkpoint, sha256)
    if any(record.get(key) != value for key, value in contract.items()):
        raise ValueError("external segmentation package changed")


def _project_tile_inventory(project: Path) -> tuple[dict, dict[str,dict]]:
    receipt = _json(project/"PROJECT_RECEIPT.json")
    manifest_path = project/"PROJECT_MANIFEST.json"
    if receipt.get("schema") != "compag-r92-image-project/v2" or _sha(manifest_path) != receipt.get("project_manifest_sha256"):
        raise ValueError("prepared full-card project changed")
    manifest = _json(manifest_path)
    tiled = _json(project/"tiled_coco.json")
    tiles = {im["file_name"]: im for im in tiled["images"]}
    if len(tiles) != receipt["tile_count"]:
        raise ValueError("prepared tile inventory differs")
    for name in tiles:
        path=project/"tiles"/name
        if path.is_symlink() or not path.is_file() or _sha(path) != manifest["sha256_by_relative_path"]["tiles/"+name]:
            raise ValueError("prepared tile changed before YOLO prediction")
    return receipt, tiles


def precompute_boxes(project: Path, checkpoint: Path, sha256: str, output: Path,
                     *, model_package: Path | None = None) -> dict:
    """CPU detector stage in optional environment; no SAM2 GPU required."""
    project, output = Path(project).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("YOLO box output exists")
    receipt, tiles = _project_tile_inventory(project)
    model = verify_checkpoint(checkpoint, sha256, model_package=model_package)
    config = _segment_package_contract(model_package, checkpoint, sha256) if model_package else None
    by_tile = {name: predict_tile_boxes(model,project/"tiles"/name,
               config=config["inference_config"] if config else None) for name in sorted(tiles)}
    record = {"schema":"compag-r92-yolo-boxes/v2" if config else "compag-r92-yolo-boxes/v1",
        "status":"PASS",
        "project_receipt_sha256":_sha(project/"PROJECT_RECEIPT.json"),
        "checkpoint_sha256":sha256,
        "ultralytics_version":SEGMENT_ULTRALYTICS_VERSION if config else "8.3.221",
        "task":model.task,"class_map":{"0":"CJ"},
        "tile_sha256":{name:_sha(project/"tiles"/name) for name in sorted(tiles)},
        "boxes_by_tile":by_tile,"tile_count":receipt["tile_count"],
        "box_count":sum(map(len,by_tile.values()))}
    if config:
        import torch
        record.update(config)
        record.update({"execution_device": "cpu", "torch_version": torch.__version__,
                       "segmentation_masks_consumed": False, "masks_verified_per_box": True})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(record,separators=(",",":"),allow_nan=False)+"\n")
    return {k:record[k] for k in ("schema","status","checkpoint_sha256","tile_count","box_count")}


def load_precomputed_boxes(path: Path, project: Path, sha256: str) -> dict[str,list[dict]]:
    """Verify all detector work before the GPU stage without importing YOLO."""
    project = Path(project).resolve(strict=True)
    receipt, tiles = _project_tile_inventory(project)
    record = _json(path)
    segment = record.get("schema") == "compag-r92-yolo-boxes/v2"
    if record.get("schema") not in {"compag-r92-yolo-boxes/v1", "compag-r92-yolo-boxes/v2"} \
            or record.get("status") != "PASS" \
            or record.get("project_receipt_sha256") != _sha(project/"PROJECT_RECEIPT.json") \
            or record.get("checkpoint_sha256") != sha256 \
            or record.get("ultralytics_version") != (SEGMENT_ULTRALYTICS_VERSION if segment else "8.3.221") \
            or record.get("class_map") != {"0":"CJ"} \
            or record.get("task") != ("segment" if segment else "detect") \
            or record.get("tile_count") != receipt["tile_count"]:
        raise ValueError("precomputed YOLO boxes belong to another model or project")
    if segment:
        if (record.get("execution_device") != "cpu"
                or record.get("segmentation_masks_consumed") is not False
                or record.get("masks_verified_per_box") is not True
                or not isinstance(record.get("torch_version"), str)):
            raise ValueError("segmentation execution provenance differs")
        package = record.get("model_package_root")
        if not isinstance(package, str) or not package:
            raise ValueError("segmentation model package is absent")
        contract = _segment_package_contract(Path(package), Path(package)/"weights"/"best.pt", sha256)
        if any(record.get(key) != value for key, value in contract.items()):
            raise ValueError("segmentation package or inference configuration changed")
    boxes = record.get("boxes_by_tile")
    if not isinstance(boxes,dict) or set(boxes) != set(tiles) or set(record.get("tile_sha256",{})) != set(tiles):
        raise ValueError("YOLO box coverage is incomplete")
    for name, rows in boxes.items():
        if record["tile_sha256"][name] != _sha(project/"tiles"/name) or not isinstance(rows,list):
            raise ValueError("YOLO source tile or box rows changed")
        for row in rows:
            if not isinstance(row,dict) or set(row) != {"bbox_xyxy","confidence"}:
                raise ValueError("YOLO box row schema differs")
            x1,y1,x2,y2 = row["bbox_xyxy"]
            confidence=row["confidence"]
            if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (x1,y1,x2,y2,confidence)) \
                    or not (0 <= x1 < x2 <= 512 and 0 <= y1 < y2 <= 512 and 0 <= confidence <= 1):
                raise ValueError("YOLO box coordinates or score are invalid")
    if record.get("box_count") != sum(map(len,boxes.values())):
        raise ValueError("YOLO box accounting differs")
    return boxes


def predict_tile_boxes(model, image_path: Path, *, imgsz: int = 512,
                       config: dict | None = None) -> list[dict]:
    """Read CJ boxes from detection or segmentation on a prepared tile."""
    options = ({"imgsz": config["image_size"], "conf": config["confidence"],
                "iou": config["iou"], "max_det": config["max_detections"],
                "half": config["half"], "rect": config["rect"],
                "retina_masks": config["retina_masks"],
                "agnostic_nms": config["agnostic_nms"]} if config else
               {"imgsz": imgsz, "conf": 0.01})
    result = model.predict(source=str(image_path), device="cpu", verbose=False, **options)
    if len(result) != 1:
        raise RuntimeError("YOLO did not return exactly one tile prediction")
    boxes = result[0].boxes
    if boxes is None:
        raise RuntimeError("YOLO did not return a box collection")
    if config and len(boxes) and (result[0].masks is None or len(result[0].masks) != len(boxes)):
        raise RuntimeError("segmentation checkpoint returned boxes without matching masks")
    found = []
    for box in boxes:
        klass = int(box.cls[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        confidence = float(box.conf[0])
        if klass != 0 or not all(math.isfinite(v) for v in (x1,y1,x2,y2,confidence)) \
                or not (0 <= x1 < x2 <= 512 and 0 <= y1 < y2 <= 512 and 0 <= confidence <= 1):
            raise ValueError("YOLO returned incompatible class, score, or tile box")
        found.append({"bbox_xyxy": [x1,y1,x2,y2], "confidence": confidence})
    return found


def prompt_sam2(runtime, tile_bgr, boxes: list[dict]) -> list[dict]:
    """Use source-matched SAM2 box prompting on the same prepared tile."""
    import cv2
    import numpy as np
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if len(boxes) > 500:
        raise ValueError("YOLO prompt count exceeds explicit 500-box limit")
    if not boxes:
        return []
    predictor = SAM2ImagePredictor(runtime.amg.predictor.model)
    predictor.set_image(cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB))
    output = []
    for box in boxes:
        x1, y1, x2, y2 = box["bbox_xyxy"]
        dx, dy = .02*(x2-x1), .02*(y2-y1)
        padded = np.asarray([[max(0.,x1-dx), max(0.,y1-dy), min(511.,x2+dx), min(511.,y2+dy)]], dtype=np.float32)
        masks, scores, _ = predictor.predict(point_coords=None, point_labels=None,
                                             box=padded, multimask_output=True)
        if len(scores) == 0:
            raise RuntimeError("SAM2 returned no mask for a valid YOLO box")
        selected = int(np.argmax(scores))
        mask = np.asarray(masks[selected] > 0, dtype=np.uint8)
        if mask.shape != (512,512):
            raise RuntimeError("SAM2 prompted mask has wrong tile shape")
        output.append({"segmentation": mask, "predicted_iou": float(scores[selected]),
                       "stability_score": float(scores[selected]), "src": "YOLO_PROMPT"})
    return output


def matched_box(candidate_bbox: tuple[float,float,float,float], boxes: list[dict],
                *, min_iou: float = .6, min_conf: float = 0.0) -> dict | None:
    x, y, w, h = candidate_bbox
    if not all(math.isfinite(v) for v in candidate_bbox) or w <= 0 or h <= 0:
        raise ValueError("candidate bbox is invalid for YOLO matching")
    best = None
    for box in boxes:
        if box["confidence"] < min_conf:
            continue
        x1,y1,x2,y2 = box["bbox_xyxy"]
        intersection = max(0.,min(x+w,x2)-max(x,x1))*max(0.,min(y+h,y2)-max(y,y1))
        union = w*h+(x2-x1)*(y2-y1)-intersection
        iou = intersection/union if union else 0.
        if iou >= min_iou and (best is None or iou > best["iou"]):
            best = {**box, "iou": iou}
    return best


def apply_fusion(scored: Path, score_receipt_path: Path, boxes_by_tile: dict[str,list[dict]], mode: str) -> dict:
    """Apply the paper's optional hybrid policy; retain raw and review scores."""
    import csv
    from .only_codes.review import GuiParams, compute_final_pred_and_ui

    if mode not in {"fusion", "both"}:
        raise ValueError("fusion is not enabled")
    scored, score_receipt_path = Path(scored), Path(score_receipt_path)
    with scored.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    if not rows or any(float(row["threshold"]) != .5 for row in rows):
        raise ValueError("paper hybrid requires XGBoost threshold tau_xgb=0.50")
    params = GuiParams(use_xgb=True, use_yolo=True, det_policy="hybrid",
                       det_missing="reject", thr_xgb=.50,
                       thr_yolo=.20, thr_iou=.60, det_thr=.50, hybrid_yolo_bias=.80)
    for row in rows:
        boxes = boxes_by_tile.get(row["image"])
        if boxes is None:
            raise ValueError("YOLO execution missing a prepared tile")
        candidate_bbox = tuple(float(row[n]) for n in ("bbox_x","bbox_y","bbox_w","bbox_h"))
        valid_match = matched_box(candidate_bbox, boxes, min_iou=params.thr_iou,
                                  min_conf=params.thr_yolo)
        match = valid_match or matched_box(candidate_bbox, boxes, min_iou=params.thr_iou)
        row["yolo_match"] = "1" if match else "0"
        row["yolo_conf"] = repr(match["confidence"]) if match else "0.0"
        row["yolo_iou"] = repr(match["iou"]) if match else "0.0"
        valid_yolo = valid_match is not None
        row["yolo_valid"] = "1" if valid_yolo else "0"
        predicted, review_score, threshold = compute_final_pred_and_ui(row, params)
        row["final_pred"] = str(predicted)
        row["p_ui"] = repr(review_score)
        row["fused_p"] = repr(review_score) if valid_yolo else ""
        row["p_fused"] = row["fused_p"]
        row["final_threshold"] = repr(threshold)
    with scored.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*fields,"yolo_match","yolo_conf","yolo_iou",
                                                      "yolo_valid","final_pred","p_ui","fused_p",
                                                      "p_fused","final_threshold"])
        writer.writeheader()
        writer.writerows(rows)
    receipt = _json(score_receipt_path)
    receipt.update({"detections_sha256": _sha(scored), "yolo_mode": mode,
        "fusion_policy": PAPER_HYBRID_POLICY, "det_policy": "hybrid",
        "det_missing": "reject", "tau_xgb": .50, "tau_det": .50,
        "tau_yolo": .20, "tau_iou": .60, "hybrid_yolo_bias": .80,
        "yolo_match_count": sum(r["yolo_match"] == "1" for r in rows),
        "yolo_valid_count": sum(r["yolo_valid"] == "1" for r in rows)})
    score_receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
