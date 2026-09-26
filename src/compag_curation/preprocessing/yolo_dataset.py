"""COCO-to-YOLO dataset preparation with explicit artifact paths."""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from compag_curation.contracts import (
    ContractError,
    HandlerServices,
    create_output_directory,
    directory_identity,
    directory_tree_digest,
)


@dataclass(frozen=True)
class YoloDatasetRequest:
    coco_annotations: Path
    tiles: Path
    split_manifest: Path
    output: Path
    positive_category: str = "cj"
    fallback_validation_fraction: float = 0.1
    random_state: int = 42
    allow_fallback_validation: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.fallback_validation_fraction < 1.0:
            raise ContractError("fallback validation fraction must be between zero and one")
        if not self.positive_category.strip():
            raise ContractError("positive category must be nonempty")
        if not isinstance(self.allow_fallback_validation, bool):
            raise ContractError("validation fallback policy must be boolean")


@dataclass(frozen=True)
class YoloSplitCount:
    images: int
    positive_images: int
    negative_images: int


@dataclass(frozen=True)
class YoloDatasetResult:
    data_yaml: Path
    train: YoloSplitCount
    validation: YoloSplitCount


@dataclass(frozen=True)
class _YoloEmissionRecord:
    source: Path
    image_name: str
    label_name: str
    label_text: str


def _load_split_manifest(path: Path) -> dict[str, set[int]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("split manifest cannot be read") from exc
    if not isinstance(value, Mapping):
        raise ContractError("split manifest must be a JSON object")
    allowed = {
        "train",
        "validation",
        "val",
        "test",
        "random_state",
        "freeze_existing",
    }
    if set(value) - allowed or "train" not in value:
        raise ContractError("split manifest fields are invalid")

    def normalize(name: str) -> set[int]:
        raw = value.get(name, [])
        if not isinstance(raw, list) or any(isinstance(item, bool) for item in raw):
            raise ContractError(f"split {name} must be an integer list")
        try:
            normalized = {int(item) for item in raw}
        except (TypeError, ValueError) as exc:
            raise ContractError(f"split {name} must be an integer list") from exc
        if any(item < 0 for item in normalized):
            raise ContractError(f"split {name} identifiers must be nonnegative")
        return normalized

    train = normalize("train")
    validation = normalize("validation") | normalize("val")
    test = normalize("test")
    if train & validation or train & test or validation & test:
        raise ContractError("split manifest contains overlapping original-image IDs")
    return {"train": train, "validation": validation, "test": test}


def _bbox_from_segmentation(segmentation: Any) -> list[float] | None:
    if not isinstance(segmentation, list) or not segmentation:
        return None
    polygon = segmentation[0]
    if not isinstance(polygon, list) or len(polygon) < 4:
        return None
    xs = polygon[0::2]
    ys = polygon[1::2]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def _yolo_line(bbox: list[float], width: int, height: int) -> str:
    x, y, box_width, box_height = bbox
    center_x = (x + box_width / 2.0) / max(width, 1)
    center_y = (y + box_height / 2.0) / max(height, 1)
    normalized_width = box_width / max(width, 1)
    normalized_height = box_height / max(height, 1)
    return (
        f"0 {center_x:.6f} {center_y:.6f} "
        f"{normalized_width:.6f} {normalized_height:.6f}"
    )


def _validated_bbox(annotation: Mapping[str, Any]) -> list[float] | None:
    raw_bbox = annotation.get("bbox")
    if raw_bbox:
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            raise ContractError("YOLO annotation bbox must contain four numbers")
        try:
            bbox = [float(item) for item in raw_bbox]
        except (TypeError, ValueError) as exc:
            raise ContractError("YOLO annotation bbox must contain four numbers") from exc
    else:
        try:
            bbox = _bbox_from_segmentation(annotation.get("segmentation"))
        except (TypeError, ValueError) as exc:
            raise ContractError("YOLO annotation segmentation is invalid") from exc
    if bbox is None or bbox[2] <= 0.0 or bbox[3] <= 0.0:
        return None
    return bbox


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink() or not source.is_file():
        raise ContractError("YOLO source image must be a regular non-symlink file")
    if destination.exists() or destination.is_symlink():
        raise ContractError("YOLO dataset contains a duplicate destination image")
    shutil.copy2(source, destination)


# SOURCE_CELL: NB-LIVE-0008-C0000
# SOURCE_STATEMENT_MAP: reset-yolo-subfolders -> _reset_directory
def _reset_directory(path: Path) -> None:
    if path.exists():
        for child in path.glob("*"):
            try:
                child.unlink()
            except IsADirectoryError:
                shutil.rmtree(child, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _plan_split(
    *,
    images: list[Mapping[str, Any]],
    annotations_by_image: Mapping[int, list[Mapping[str, Any]]],
    original_ids: set[int],
    positive_category_id: int,
    tiles: Path,
) -> tuple[tuple[_YoloEmissionRecord, ...], YoloSplitCount]:
    records: list[_YoloEmissionRecord] = []
    positive_count = negative_count = 0
    for image in images:
        metadata = image.get("meta") or {}
        if not isinstance(metadata, Mapping):
            raise ContractError("YOLO image metadata must be an object")
        try:
            original_id = int(metadata.get("orig_image_id", -1))
        except (TypeError, ValueError) as exc:
            raise ContractError("YOLO image original identifier must be an integer") from exc
        if original_id not in original_ids:
            continue
        try:
            image_id = int(image["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("YOLO selected image identifier must be an integer") from exc
        filename = Path(str(image.get("file_name", ""))).name
        if not filename:
            raise ContractError("YOLO selected image filename must be nonempty")
        source = tiles / filename
        if source.is_symlink() or not source.is_file():
            raise ContractError("YOLO split references an unavailable source image")
        try:
            width = int(image.get("width", 0)) or 512
            height = int(image.get("height", 0)) or 512
        except (TypeError, ValueError) as exc:
            raise ContractError("YOLO selected image dimensions must be integers") from exc
        if width < 1 or height < 1:
            raise ContractError("YOLO selected image dimensions must be positive")
        label_lines: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            try:
                category_id = int(annotation.get("category_id", -1))
            except (TypeError, ValueError) as exc:
                raise ContractError("YOLO annotation category must be an integer") from exc
            if category_id != positive_category_id:
                continue
            bbox = _validated_bbox(annotation)
            if bbox is not None:
                label_lines.append(_yolo_line(bbox, width, height))
        label_text = "\n".join(label_lines)
        records.append(
            _YoloEmissionRecord(
                source=source,
                image_name=filename,
                label_name=f"{Path(filename).stem}.txt",
                label_text=label_text,
            )
        )
        if label_lines:
            positive_count += 1
        else:
            negative_count += 1
    count = YoloSplitCount(len(records), positive_count, negative_count)
    return tuple(records), count


def _emit_split(
    records: tuple[_YoloEmissionRecord, ...],
    *,
    image_output: Path,
    label_output: Path,
) -> None:
    for record in records:
        _link_or_copy(record.source, image_output / record.image_name)
        (label_output / record.label_name).write_text(
            record.label_text,
            encoding="utf-8",
        )


def _preflight_selected_tiles(
    images: list[Mapping[str, Any]],
    original_ids: set[int],
    tiles: Path,
) -> None:
    resolved_original_ids: set[int] = set()
    image_names: set[str] = set()
    label_names: set[str] = set()
    for image in images:
        metadata = image.get("meta") or {}
        if not isinstance(metadata, Mapping):
            raise ContractError("YOLO image metadata must be an object")
        try:
            original_id = int(metadata.get("orig_image_id", -1))
        except (TypeError, ValueError) as exc:
            raise ContractError("YOLO image original identifier must be an integer") from exc
        if original_id not in original_ids:
            continue
        filename = Path(str(image.get("file_name", ""))).name
        label_name = f"{Path(filename).stem}.txt"
        if (
            not filename
            or filename in image_names
            or label_name in label_names
        ):
            raise ContractError("YOLO selected tiles contain a destination-name collision")
        source = tiles / filename
        if source.is_symlink() or not source.is_file():
            raise ContractError("YOLO split references an unavailable source image")
        image_names.add(filename)
        label_names.add(label_name)
        resolved_original_ids.add(original_id)
    if resolved_original_ids != original_ids:
        raise ContractError("YOLO split identifiers do not resolve to the sealed tiled COCO")


# SOURCE_CELL: NB-LIVE-0008-C0000
# SOURCE_STATEMENT_MAP: top-level:000-033 -> prepare_yolo_dataset/full-algorithm
def prepare_yolo_dataset(
    request: YoloDatasetRequest,
    services: HandlerServices,
) -> YoloDatasetResult:
    """Build the train/validation tree and write the reviewed one-class YAML."""

    try:
        coco = json.loads(request.coco_annotations.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("COCO annotations cannot be read") from exc
    if not isinstance(coco, Mapping):
        raise ContractError("COCO annotations must be a JSON object")
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    if (
        not isinstance(images, list)
        or not isinstance(annotations, list)
        or not isinstance(categories, list)
        or any(not isinstance(image, Mapping) for image in images)
        or any(not isinstance(annotation, Mapping) for annotation in annotations)
        or any(not isinstance(category, Mapping) for category in categories)
    ):
        raise ContractError("COCO images, annotations, and categories must be object lists")

    positive_category_id = 1
    target = request.positive_category.casefold()
    for category in categories:
        if target in str(category.get("name", "")).casefold():
            try:
                positive_category_id = int(category["id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ContractError("YOLO positive category identifier is invalid") from exc
            break

    split = _load_split_manifest(request.split_manifest)
    train_ids = split["train"]
    validation_ids = split["validation"]
    if not validation_ids and train_ids and request.allow_fallback_validation:
        generator = random.Random(request.random_state)
        ordered = sorted(train_ids)
        count = max(1, int(request.fallback_validation_fraction * len(ordered)))
        validation_ids = set(generator.sample(ordered, count))
        train_ids -= validation_ids
    if not train_ids or not validation_ids:
        raise ContractError("YOLO dataset requires disjoint nonempty train and validation IDs")
    _preflight_selected_tiles(images, train_ids | validation_ids, request.tiles)

    annotations_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        try:
            image_id = int(annotation["image_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("YOLO annotation image identifier is invalid") from exc
        annotations_by_image[image_id].append(annotation)

    train_records, train_count = _plan_split(
        images=images,
        annotations_by_image=annotations_by_image,
        original_ids=train_ids,
        positive_category_id=positive_category_id,
        tiles=request.tiles,
    )
    validation_records, validation_count = _plan_split(
        images=images,
        annotations_by_image=annotations_by_image,
        original_ids=validation_ids,
        positive_category_id=positive_category_id,
        tiles=request.tiles,
    )
    if train_count.images == 0 or validation_count.images == 0:
        raise ContractError("YOLO split did not resolve to nonempty train and validation images")

    create_output_directory(request.output)
    output_identity = directory_identity(request.output)
    train_images = request.output / "images" / "train"
    validation_images = request.output / "images" / "val"
    train_labels = request.output / "labels" / "train"
    validation_labels = request.output / "labels" / "val"
    for directory in (train_images, validation_images, train_labels, validation_labels):
        directory.mkdir(parents=True, exist_ok=False)

    _emit_split(
        train_records,
        image_output=train_images,
        label_output=train_labels,
    )
    _emit_split(
        validation_records,
        image_output=validation_images,
        label_output=validation_labels,
    )
    data_yaml = request.output / "data.yaml"
    data_yaml.write_text(
        "# CJ YOLO (full dataset from tiled COCO)\n"
        f"path: {request.output}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: [ cj ]\n",
        encoding="utf-8",
    )
    if directory_identity(request.output) != output_identity:
        raise ContractError("YOLO dataset output directory identity changed")
    output_digest = directory_tree_digest(request.output)
    try:
        services.report(
            f"YOLO dataset ready: train={train_count.images} "
            f"validation={validation_count.images}"
        )
    finally:
        if (
            directory_identity(request.output) != output_identity
            or directory_tree_digest(request.output) != output_digest
        ):
            raise ContractError("YOLO dataset output directory identity changed")
    return YoloDatasetResult(data_yaml, train_count, validation_count)


__all__ = [
    "YoloDatasetRequest",
    "YoloDatasetResult",
    "YoloSplitCount",
    "_reset_directory",
    "prepare_yolo_dataset",
]
