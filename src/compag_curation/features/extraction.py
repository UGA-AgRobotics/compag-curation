"""Source-backed feature preprocessing and extraction algorithms.

Scientific libraries are imported only inside authorized calls.  Paths,
models, transforms, and report sinks enter through typed parameters instead
of process environment or Notebook state.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Protocol, Sequence

from compag_curation.contracts import ContractError, HandlerServices, atomic_write_output


REVIEW_WEIGHTS = {
    "accept": 1.0,
    "flip": 1.0,
    "sus_accept": 0.4,
    "sus_flip": 0.4,
    "skip": 0.0,
}
PACK_MULTISCALE_SCALES = (0.85, 1.0, 1.15)
MULTISCALE_SCALES = (0.67, 0.8, 1.0, 1.25)
FEATURE_BASE_COLUMNS = (
    "file_name",
    "ann_id",
    "scale",
    "area_px",
    "area_norm",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "cx",
    "cy",
    "mean_L",
    "mean_a",
    "mean_b",
    "std_L",
    "std_a",
    "std_b",
    "median_L",
    "median_a",
    "median_b",
    "delta_a",
    "delta_b",
    "elongation",
    "eccentricity",
    "solidity",
    "aspect_ratio",
    "circularity",
    "extent",
    "perimeter",
    "bbox_diag_frac",
    "perim_over_sqrt_area",
    "scale_diag",
    "hu1",
    "hu2",
    "hu4",
    "components_count",
    "holes_ratio",
    "touching_border",
    "LBP_u5",
    "GLCM_contrast",
    "GLCM_homogeneity",
    "grad_mean",
    "grad_p90",
    "grid_r",
    "grid_c",
    "grid_r_norm",
    "grid_c_norm",
    "pred_iou",
    "stability",
    "embed_sim",
    "g_quality",
    "g_light",
    "g_color",
    "g_shape",
    "g_embed",
    "g_robust",
    "g_maha",
    "g_border",
)
FEATURE_RICH_COLUMNS = (
    "delta_b_med",
    "delta_a_med",
    "median_b_in",
    "median_b_ring",
    "median_a_in",
    "median_a_ring",
    "histb_q1",
    "histb_q2",
    "histb_q3",
    "histb_q4",
    "histab_q11",
    "histab_q12",
    "histab_q21",
    "histab_q22",
)
FEATURE_TRAILING_COLUMNS = (
    "reviewed",
    "review_tag",
    "class_id",
    "class_name",
    "label",
)
FEATURE_SCHEMA_BACKUP_POLICY = "PRESERVE_BACKUPS"
LEGACY_FEATURE_BACKUP_SUFFIXES = {
    "scale": ".pre_scale.bak.csv",
    "review_tag": ".pre_reviewtag.bak.csv",
}


@dataclass(frozen=True)
class SegmentationPolicy:
    polygon_shape_width: int
    polygon_inferred_dimension: int
    polygon_coordinate_dtype: str
    polygon_raster_dtype: str
    rle_rank: int
    rle_rank_comparison: str
    rle_collapse_axis: int
    mask_dtype: str
    foreground_value: int
    foreground_application: str
    require_positive_bbox_extent: bool
    empty_mask_on_invalid_bbox: bool


class FeatureBackend(Protocol):
    """Injected image/mask/embedding backend for the reviewed algorithm."""

    def read_image(self, path: Path) -> Any: ...

    def segmentation_mask(
        self,
        annotation: Mapping[str, Any],
        height: int,
        width: int,
        *,
        mode: str,
        policy: SegmentationPolicy,
        clipped_bbox: tuple[int, int, int, int],
    ) -> Any: ...

    def compute_features(
        self,
        image: Any,
        mask: Any,
        *,
        mode: str,
        scale: float,
        prototype: Path,
    ) -> dict[str, float]: ...

    def embedding(
        self,
        image: Any,
        mask: Any,
        pad_fraction: float,
        *,
        scale: float = 1.0,
    ) -> Any: ...

    def project_embedding(self, embedding: Any, pca_path: Path) -> Sequence[float]: ...

    def background_ab(self, image: Any, *, scale: float) -> tuple[float, float]: ...

    def scaled_shape(self, image: Any, *, scale: float) -> tuple[int, int]: ...

    def hu_moments(
        self,
        mask: Any,
        *,
        scale: float,
        binary_image: bool,
    ) -> Sequence[float]: ...

    def prototype_dot_product(self, embedding: Any, prototype: Path) -> float: ...

    def pca_component_count(self, pca_path: Path) -> int: ...

    def load_feature_pack(self, path: Path) -> Mapping[str, Any]: ...

    def write_feature_pack(self, handle: BinaryIO, value: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True)
class ReviewMergeRequest:
    original_coco: Path
    tiled_coco: Path
    merged_original_coco: Path
    merged_tiled_coco: Path
    detections_csv: Path
    review_labels_csv: Path
    round_new_ids_output: Path
    mode: str = "xgb_recall"
    backup_directory: Path | None = None


@dataclass(frozen=True)
class ReviewMergeResult:
    added: int
    updated: int
    skipped: int
    new_original_ids: tuple[int, ...]


def _read_json_object(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{role} cannot be read") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{role} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2).encode("utf-8")
    atomic_write_output(path, lambda handle: handle.write(payload))


def _next_id(records: Iterable[Mapping[str, Any]]) -> int:
    current = 0
    for record in records:
        try:
            current = max(current, int(record.get("id", 0)))
        except (TypeError, ValueError):
            pass
    return current + 1


def _rectangle_polygon(row: Mapping[str, str]) -> tuple[list[float], list[float]] | None:
    aliases = (
        ("bbox_x", "bbox_y", "bbox_w", "bbox_h"),
        ("x", "y", "w", "h"),
    )
    for names in aliases:
        if all(str(row.get(name, "")).strip() for name in names):
            x, y, width, height = (float(row[name]) for name in names)
            if width <= 0 or height <= 0:
                return None
            bbox = [x, y, width, height]
            polygon = [x, y, x + width, y, x + width, y + height, x, y + height]
            return bbox, polygon
    return None


# SOURCE_CELL: NB-LIVE-0001-C0001
# SOURCE_STATEMENT_MAP: tile-root-parser -> _parse_tile_root
def _parse_tile_root(tile_name: str) -> str:
    import re

    match = re.match(
        r"^(?P<root>.+?)_y\d{1,8}x\d{1,8}\.(jpg|jpeg|png)$",
        Path(tile_name).name,
        flags=re.IGNORECASE,
    )
    return match.group("root") if match else Path(tile_name).stem.split("_y")[0]


# SOURCE_CELL: NB-LIVE-0001-C0001
# SOURCE_STATEMENT_MAP: ensure-coco-categories -> _ensure_coco_categories
def _ensure_coco_categories(document: dict[str, Any]) -> None:
    document.setdefault("images", [])
    document.setdefault("annotations", [])
    categories = document.setdefault("categories", [])
    if not categories:
        document["categories"] = [
            {"id": 1, "name": "CJ", "supercategory": ""},
            {"id": 2, "name": "non-CJ", "supercategory": ""},
        ]


# SOURCE_CELL: NB-LIVE-0001-C0001
# SOURCE_STATEMENT_MAP: polygon-bbox -> _bbox_from_polygon
def _bbox_from_polygon(polygon: Sequence[float]) -> list[int]:
    xs = polygon[0::2]
    ys = polygon[1::2]
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(xs), max(ys)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


# SOURCE_CELL: NB-LIVE-0001-C0001
# SOURCE_STATEMENT_MAP: detection-polygon-fallbacks -> _detection_polygon
def _detection_polygon(row: Mapping[str, str]) -> list[float] | None:
    polygon: Any = None
    raw_polygon = row.get("poly", "")
    try:
        polygon = json.loads(raw_polygon) if isinstance(raw_polygon, str) else raw_polygon
    except (TypeError, json.JSONDecodeError):
        polygon = None
    if isinstance(polygon, list) and len(polygon) >= 6:
        return [float(value) for value in polygon]
    try:
        if all(name in row for name in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")):
            x = float(row["bbox_x"])
            y = float(row["bbox_y"])
            width = float(row["bbox_w"])
            height = float(row["bbox_h"])
            if width > 0 and height > 0:
                return [
                    x,
                    y,
                    x + width,
                    y,
                    x + width,
                    y + height,
                    x,
                    y + height,
                ]
    except (KeyError, TypeError, ValueError):
        pass
    try:
        raw_bbox = row.get("bbox")
        if isinstance(raw_bbox, str) and "," in raw_bbox:
            x, y, width, height = [float(value) for value in raw_bbox.split(",")]
            if width > 0 and height > 0:
                return [
                    x,
                    y,
                    x + width,
                    y,
                    x + width,
                    y + height,
                    x,
                    y + height,
                ]
    except (TypeError, ValueError):
        pass
    try:
        if all(name in row for name in ("minx", "miny", "maxx", "maxy")):
            x0 = float(row["minx"])
            y0 = float(row["miny"])
            x1 = float(row["maxx"])
            y1 = float(row["maxy"])
            if x1 > x0 and y1 > y0:
                return [x0, y0, x1, y0, x1, y1, x0, y1]
    except (KeyError, TypeError, ValueError):
        pass
    return None


# SOURCE_CELL: NB-LIVE-0001-C0001
# SOURCE_STATEMENT_MAP: review-merge-backup -> _backup_review_merge_inputs
def _backup_review_merge_inputs(
    request: ReviewMergeRequest,
) -> tuple[Path, ...]:
    if request.backup_directory is None:
        return ()
    request.backup_directory.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in (
        request.original_coco,
        request.tiled_coco,
        request.round_new_ids_output,
        request.review_labels_csv,
        request.detections_csv,
    ):
        if source.exists() and source.is_file() and not source.is_symlink():
            destination = request.backup_directory / source.name
            shutil.copy2(source, destination)
            copied.append(destination)
    return tuple(copied)


# SOURCE_CELL: NB-LIVE-0001-C0001
# SOURCE_STATEMENT_MAP: top-level:000-058 -> merge_review_labels_into_coco/full-algorithm
def merge_review_labels_into_coco(
    request: ReviewMergeRequest,
    services: HandlerServices,
) -> ReviewMergeResult:
    """Apply reviewed detections to original and tiled COCO masters."""

    if request.mode not in {"gate", "xgb", "xgb_recall"}:
        raise ContractError("review merge mode is invalid")
    original = _read_json_object(request.original_coco, "original COCO")
    tiled = _read_json_object(request.tiled_coco, "tiled COCO")
    _ensure_coco_categories(original)
    _ensure_coco_categories(tiled)

    root_to_original_id = {
        Path(str(image.get("file_name", ""))).stem: int(image["id"])
        for image in original.get("images", [])
    }
    detection_polygons: dict[tuple[str, str], list[float]] = {}
    with request.detections_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image = row.get("image")
            identifier = row.get("id")
            if image is None or identifier is None:
                continue
            polygon = _detection_polygon(row)
            if polygon is not None:
                detection_polygons[(image, str(identifier))] = polygon

    reviewed: dict[tuple[str, str], tuple[int, int]] = {}
    roots_seen: set[str] = set()
    with request.review_labels_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image = row.get("image")
            identifier = row.get("id")
            if image is None or identifier is None:
                continue
            key = (image, str(identifier))
            if key not in detection_polygons:
                continue
            try:
                human_label = int(row.get("human_label", 0))
            except (TypeError, ValueError):
                human_label = 0
            category_id = 1 if human_label == 1 else 2
            try:
                timestamp = int(row.get("timestamp", 0))
            except (TypeError, ValueError):
                timestamp = 0
            previous = reviewed.get(key)
            if previous is None or timestamp >= previous[1]:
                reviewed[key] = (category_id, timestamp)
            roots_seen.add(_parse_tile_root(image))

    new_original_ids: list[int] = []
    for root in sorted(roots_seen):
        if root not in root_to_original_id:
            original_id = _next_id(original["images"])
            original["images"].append(
                {"id": int(original_id), "file_name": f"{root}.jpg"}
            )
            root_to_original_id[root] = int(original_id)
            new_original_ids.append(int(original_id))

    tiled_name_to_id = {
        Path(str(image["file_name"])).name: int(image["id"])
        for image in tiled.get("images", [])
    }
    for image, _ in reviewed:
        basename = Path(image).name
        if basename not in tiled_name_to_id:
            tiled_image_id = _next_id(tiled["images"])
            root = _parse_tile_root(basename)
            tiled["images"].append(
                {
                    "id": int(tiled_image_id),
                    "width": 512,
                    "height": 512,
                    "file_name": basename,
                    "meta": {
                        "orig_file_name": f"{root}.jpg",
                        "orig_image_id": int(root_to_original_id[root]),
                    },
                }
            )
            tiled_name_to_id[basename] = int(tiled_image_id)

    annotation_by_review_key: dict[tuple[int, int], int] = {}
    for index, annotation in enumerate(tiled.get("annotations", [])):
        metadata = annotation.get("meta", {}) or {}
        review_id = metadata.get("review_id")
        if review_id is None:
            continue
        annotation_by_review_key[
            (int(annotation.get("image_id", -1)), int(review_id))
        ] = index

    next_annotation = _next_id(tiled["annotations"])
    added = updated = skipped = 0
    for (image, review_identifier), (category_id, _) in reviewed.items():
        basename = Path(image).name
        image_id = tiled_name_to_id[basename]
        polygon = detection_polygons[(image, review_identifier)]
        segmentation = [[float(value) for value in polygon]]
        bbox = _bbox_from_polygon(polygon)
        review_id = int(review_identifier)
        key = (int(image_id), review_id)
        if key in annotation_by_review_key:
            annotation = tiled["annotations"][annotation_by_review_key[key]]
            annotation["category_id"] = int(category_id)
            annotation["segmentation"] = segmentation
            annotation["bbox"] = bbox
            annotation["area"] = float(bbox[2] * bbox[3])
            metadata = annotation.get("meta", {}) or {}
            metadata["review_id"] = review_id
            annotation["meta"] = metadata
            updated += 1
        else:
            tiled["annotations"].append(
                {
                    "id": int(next_annotation),
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": bbox,
                    "segmentation": segmentation,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                    "meta": {"review_id": review_id},
                }
            )
            annotation_by_review_key[key] = len(tiled["annotations"]) - 1
            next_annotation += 1
            added += 1

    _write_json_atomic(request.merged_original_coco, original)
    _write_json_atomic(request.merged_tiled_coco, tiled)
    ordered_ids = tuple(sorted(map(int, new_original_ids)))
    _write_json_atomic(request.round_new_ids_output, ordered_ids)
    services.report(
        f"COCO review merge: added={added} updated={updated} skipped={skipped}"
    )
    return ReviewMergeResult(added, updated, skipped, ordered_ids)


@dataclass(frozen=True)
class SplitRequest:
    original_coco: Path
    existing_split_manifest: Path | None
    new_original_ids: Path | None
    output_manifest: Path
    random_state: int = 42
    test_fraction: float = 0.0
    validation_fraction: float = 0.0
    freeze_existing: bool = True
    backup_directory: Path | None = None

    def __post_init__(self) -> None:
        if self.random_state != 42:
            raise ContractError("sealed split seed must remain 42")
        if self.test_fraction != 0.0 or self.validation_fraction != 0.0:
            raise ContractError("source split policy keeps validation and test empty initially")


@dataclass(frozen=True)
class SplitResult:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    frozen_existing: bool


def _integer_ids(value: Any, role: str) -> set[int]:
    if (
        not isinstance(value, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or len(set(value)) != len(value)
    ):
        raise ContractError(f"{role} must be a JSON integer list")
    return set(value)


# SOURCE_CELL: NB-LIVE-0001-C0002
# SOURCE_STATEMENT_MAP: consume-new-id-artifact -> _consume_new_ids
def _consume_new_ids(path: Path) -> None:
    applied = path.with_suffix(".applied.json")
    try:
        path.replace(applied)
    except OSError:
        try:
            path.write_text("[]", encoding="utf-8")
        except OSError:
            pass


# SOURCE_CELL: NB-LIVE-0001-C0002
# SOURCE_STATEMENT_MAP: load-existing-splits -> _load_existing_splits
def _load_existing_splits(path: Path | None) -> tuple[set[int], set[int], set[int]] | None:
    if path is None:
        return None
    if path.is_dir():
        train_path = path / "orig_train_ids.json"
        validation_path = path / "orig_val_ids.json"
        test_path = path / "orig_test_ids.json"
        if train_path.exists() and validation_path.exists() and test_path.exists():
            return (
                set(map(int, json.loads(train_path.read_text(encoding="utf-8")))),
                set(map(int, json.loads(validation_path.read_text(encoding="utf-8")))),
                set(map(int, json.loads(test_path.read_text(encoding="utf-8")))),
            )
        return None
    if path.is_file():
        document = _read_json_object(path, "split manifest")
        return (
            _integer_ids(document.get("train", []), "train split"),
            _integer_ids(
                document.get("validation", document.get("val", [])),
                "validation split",
            ),
            _integer_ids(document.get("test", []), "test split"),
        )
    return None


# SOURCE_CELL: NB-LIVE-0001-C0002
# SOURCE_STATEMENT_MAP: read-new-ids -> _read_new_ids
def _read_new_ids(path: Path | None) -> set[int]:
    if path is None:
        return set()
    try:
        info = path.lstat()
    except OSError as exc:
        raise ContractError("new original-ID artifact is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ContractError("new original-ID artifact must be a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("new original-ID artifact cannot be read as UTF-8 JSON") from exc
    return _integer_ids(value, "new original-ID artifact")


# SOURCE_CELL: NB-LIVE-0001-C0002
# SOURCE_STATEMENT_MAP: save-split-lists -> _save_split_lists
def _save_split_lists(
    output: Path,
    train: Iterable[int],
    validation: Iterable[int],
    test: Iterable[int],
    *,
    random_state: int,
    freeze_existing: bool,
) -> None:
    train_list = sorted(int(value) for value in train)
    validation_list = sorted(int(value) for value in validation)
    test_list = sorted(int(value) for value in test)
    if output.exists() and output.is_dir():
        (output / "orig_train_ids.json").write_text(
            json.dumps(train_list, indent=2), encoding="utf-8"
        )
        (output / "orig_val_ids.json").write_text(
            json.dumps(validation_list, indent=2), encoding="utf-8"
        )
        (output / "orig_test_ids.json").write_text(
            json.dumps(test_list, indent=2), encoding="utf-8"
        )
        return
    _write_json_atomic(
        output,
        {
            "train": train_list,
            "validation": validation_list,
            "test": test_list,
            "random_state": random_state,
            "freeze_existing": freeze_existing,
        },
    )


# SOURCE_CELL: NB-LIVE-0001-C0002
# SOURCE_STATEMENT_MAP: backup-current-split-state -> _backup_current_split_state
def _backup_current_split_state(
    request: SplitRequest,
    old_train: set[int],
    old_validation: set[int],
    old_test: set[int],
    new_ids: set[int],
) -> tuple[Path, ...]:
    if request.backup_directory is None:
        return ()
    request.backup_directory.mkdir(parents=True, exist_ok=True)
    sources: list[Path] = []
    if request.existing_split_manifest is not None:
        if request.existing_split_manifest.is_dir():
            sources.extend(
                request.existing_split_manifest / name
                for name in (
                    "orig_train_ids.json",
                    "orig_val_ids.json",
                    "orig_test_ids.json",
                )
            )
        else:
            sources.append(request.existing_split_manifest)
    if request.new_original_ids is not None:
        sources.append(request.new_original_ids)
    copied: list[Path] = []
    for source in sources:
        if source.is_file() and not source.is_symlink():
            destination = request.backup_directory / source.name
            shutil.copy2(source, destination)
            copied.append(destination)
    (request.backup_directory / "backup_meta_cell0.json").write_text(
        json.dumps(
            {
                "orig_coco_json": str(request.original_coco),
                "split_dir": str(request.existing_split_manifest),
                "counts_before": {
                    "train": len(old_train),
                    "val": len(old_validation),
                    "test": len(old_test),
                },
                "new_ids": sorted(map(int, new_ids)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return tuple(copied)


# SOURCE_CELL: NB-LIVE-0001-C0002
# SOURCE_STATEMENT_MAP: top-level:000-027 -> freeze_or_create_splits/full-algorithm
def freeze_or_create_splits(
    request: SplitRequest,
    services: HandlerServices,
) -> SplitResult:
    """Freeze old group membership and assign only newly introduced IDs."""

    coco = _read_json_object(request.original_coco, "original COCO")
    all_ids = {int(image["id"]) for image in coco.get("images", [])}
    existing_sets = _load_existing_splits(request.existing_split_manifest)
    existing: dict[str, set[int]] | None = None
    if existing_sets is not None:
        existing = {
            "train": existing_sets[0],
            "validation": existing_sets[1],
            "test": existing_sets[2],
        }
        if (
            existing["train"] & existing["validation"]
            or existing["train"] & existing["test"]
            or existing["validation"] & existing["test"]
        ):
            raise ContractError("existing split manifest overlaps")

    if request.freeze_existing and existing is not None:
        missing_existing = (
            existing["train"] | existing["validation"] | existing["test"]
        ) - all_ids
        if missing_existing:
            raise ContractError("existing split IDs are absent from original COCO")
        train = set(existing["train"])
        validation = set(existing["validation"])
        test = set(existing["test"])
        new_ids = _read_new_ids(request.new_original_ids)
        if new_ids:
            missing = new_ids - all_ids
            if missing:
                raise ContractError("new original IDs are absent from original COCO")
            validation -= new_ids
            test -= new_ids
            train |= new_ids
        frozen = True
    else:
        new_ids = _read_new_ids(request.new_original_ids)
        if new_ids:
            missing = new_ids - all_ids
            if missing:
                raise ContractError("new original IDs are absent from original COCO")
        train = all_ids | new_ids
        validation = set()
        test = set()
        frozen = False

    result = SplitResult(
        tuple(sorted(train)),
        tuple(sorted(validation)),
        tuple(sorted(test)),
        frozen,
    )
    _save_split_lists(
        request.output_manifest,
        result.train,
        result.validation,
        result.test,
        random_state=request.random_state,
        freeze_existing=request.freeze_existing,
    )
    services.report(
        f"split ready: train={len(result.train)} validation={len(result.validation)} "
        f"test={len(result.test)} frozen={result.frozen_existing}"
    )
    return result


@dataclass(frozen=True)
class FeaturePackRequest:
    coco_annotations: Path
    images: Path
    allowed_original_ids: tuple[int, ...]
    prototype_output: Path
    pca_output: Path
    used_ids_output: Path
    positive_embeddings_output: Path
    all_embeddings_output: Path
    pack_output: Path
    pca_dimensions: int = 32
    pad_fraction: float = 0.10
    threshold_method: str = "fixed"
    fixed_threshold: float = 0.70
    percentile: float = 5.0
    positive_category: str = "cj"
    allowed_tiled_ids: tuple[int, ...] = ()
    train_scales: tuple[float, ...] = PACK_MULTISCALE_SCALES
    sample_limit: int | None = None
    run_mode: str = "all"

    def __post_init__(self) -> None:
        if self.pca_dimensions != 32:
            raise ContractError("sealed PCA dimensionality must remain 32")
        if self.threshold_method not in {"fixed", "percentile"}:
            raise ContractError("embedding threshold method is invalid")
        if tuple(float(scale) for scale in self.train_scales) != PACK_MULTISCALE_SCALES:
            raise ContractError("sealed training embedding scales must remain unchanged")
        if self.run_mode not in {"all", "stage1", "stage2", "stage3", "skip"}:
            raise ContractError("feature-pack run mode is invalid")
        if self.sample_limit is not None and self.sample_limit <= 0:
            raise ContractError("feature-pack sample limit must be positive")


@dataclass(frozen=True)
class FeaturePackResult:
    positive_embeddings: int
    all_embeddings: int
    threshold: float
    used_image_ids: tuple[int, ...]
    delta_image_ids: tuple[int, ...] = ()
    prototype_written: bool = False
    pca_written: bool = False
    pack_output: Path | None = None


# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_STATEMENT_MAP: l2-epsilon -> _l2_normalize/full-expression
def _l2_normalize(
    vector: Any,
    axis: int = -1,
    eps: float = 1e-12,
) -> Any:
    import numpy as np

    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array, axis=axis, keepdims=True) + eps
    return array / norm


# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_CELL: NB-LIVE-0001-C0004
# SOURCE_STATEMENT_MAP: unit-scale-resize-tolerance -> _effective_scale/full-expression
def _effective_scale(scale: float) -> float:
    value = float(scale)
    return 1.0 if abs(value - 1.0) < 1e-4 else value


# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_STATEMENT_MAP: original-to-tiled-id-map -> _original_to_tiled_ids/full-control
def _original_to_tiled_ids(
    images: Iterable[Mapping[str, Any]], original_ids: Iterable[int]
) -> set[int]:
    selected = set(int(item) for item in original_ids)
    return {
        int(image["id"])
        for image in images
        if image.get("meta", {}).get("orig_image_id") is not None
        and int(image["meta"]["orig_image_id"]) in selected
    }


# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_CELL: NB-LIVE-0001-C0004
# SOURCE_STATEMENT_MAP: segmentation-polygon-rle-bbox -> _segmentation_mask/full-control
def _call_segmentation_backend(
    backend: FeatureBackend,
    annotation: Mapping[str, Any],
    height: int,
    width: int,
    *,
    mode: str,
    policy: SegmentationPolicy,
    clipped_bbox: tuple[int, int, int, int],
) -> Any:
    if mode == "bbox" and policy.require_positive_bbox_extent and not (
        clipped_bbox[2] > clipped_bbox[0]
        and clipped_bbox[3] > clipped_bbox[1]
    ):
        if not policy.empty_mask_on_invalid_bbox:
            raise ContractError("invalid bbox requires an explicit empty-mask policy")
        return backend.segmentation_mask(
            annotation,
            height,
            width,
            mode=mode,
            policy=policy,
            clipped_bbox=clipped_bbox,
        )
    return backend.segmentation_mask(
        annotation,
        height,
        width,
        mode=mode,
        policy=policy,
        clipped_bbox=clipped_bbox,
    )


def _segmentation_mask(
    backend: FeatureBackend,
    annotation: Mapping[str, Any],
    height: int,
    width: int,
) -> Any:
    segmentation = annotation.get("segmentation", None)
    if isinstance(segmentation, list) and segmentation:
        mode = "polygon"
    elif isinstance(segmentation, Mapping) and "counts" in segmentation:
        mode = "rle"
    else:
        mode = "bbox"
    raw_bbox = annotation.get("bbox", (0, 0, 0, 0))
    if not isinstance(raw_bbox, Sequence) or len(raw_bbox) != 4:
        raw_bbox = (0, 0, 0, 0)
    x, y, box_width, box_height = (float(value) for value in raw_bbox)
    clipped_bbox = (
        max(0, int(x)),
        max(0, int(y)),
        min(width - 1, int(x + box_width)),
        min(height - 1, int(y + box_height)),
    )
    policy = SegmentationPolicy(
        polygon_shape_width=2,
        polygon_inferred_dimension=-1,
        polygon_coordinate_dtype="float32",
        polygon_raster_dtype="int32",
        rle_rank=3,
        rle_rank_comparison="eq",
        rle_collapse_axis=2,
        mask_dtype="uint8",
        foreground_value=255,
        foreground_application="multiply",
        require_positive_bbox_extent=True,
        empty_mask_on_invalid_bbox=True,
    )
    try:
        return _call_segmentation_backend(
            backend,
            annotation,
            height,
            width,
            mode=mode,
            policy=policy,
            clipped_bbox=clipped_bbox,
        )
    except Exception:
        if mode != "rle":
            raise
        return _call_segmentation_backend(
            backend,
            annotation,
            height,
            width,
            mode="bbox",
            policy=policy,
            clipped_bbox=clipped_bbox,
        )


# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_STATEMENT_MAP: incremental-array-append -> _append_embedding_matrix/full-control
def _append_embedding_matrix(existing: Any, delta: Any) -> Any:
    import numpy as np

    if existing.size == 0:
        return delta.astype(np.float32)
    if delta.size == 0:
        return existing.astype(np.float32)
    if existing.ndim != 2 or delta.ndim != 2 or existing.shape[1] != delta.shape[1]:
        raise ContractError("incremental embedding dimensions do not match")
    return np.vstack((existing, delta)).astype(np.float32)


# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_STATEMENT_MAP: numpy-artifact-write -> _write_numpy_array_atomic/artifact-io
def _write_numpy_array_atomic(path: Path, value: Any, *, archive: bool = False) -> None:
    import numpy as np

    def write_array(handle: Any) -> None:
        if archive:
            components, mean = value
            np.savez(handle, components=components, mean=mean)
        else:
            np.save(handle, value)

    atomic_write_output(path, write_array)


# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_STATEMENT_MAP: prototype-single-row-schema -> _write_prototype_atomic/artifact-io
def _write_prototype_atomic(path: Path, mean: Any, threshold: float) -> None:
    columns = [f"mu_embed_{index}" for index in range(len(mean))] + [
        "embed_thresh"
    ]
    values = [float(value) for value in mean] + [float(threshold)]

    def write_prototype(handle: Any) -> None:
        text = io.TextIOWrapper(handle, encoding="utf-8", newline="", write_through=True)
        try:
            writer = csv.DictWriter(text, fieldnames=columns)
            writer.writeheader()
            writer.writerow(dict(zip(columns, values)))
            text.flush()
        finally:
            text.detach()

    atomic_write_output(path, write_prototype)


# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_STATEMENT_MAP: pack-artifact-reload -> _read_prototype_artifact/artifact-io
def _read_prototype_artifact(
    path: Path,
    default_threshold: float,
) -> tuple[list[float], float] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        return None
    names = sorted(
        (name for name in row if name.startswith("mu_embed_")),
        key=lambda name: int(name.removeprefix("mu_embed_")),
    )
    if not names:
        names = [name for name in row if name.startswith("mu_")]
    if not names:
        return None
    return (
        [float(row[name]) for name in names],
        float(row.get("embed_thresh") or default_threshold),
    )


# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_STATEMENT_MAP: ensure-parent-directory -> _ensure_parent_directory/full-expression
def _ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_STATEMENT_MAP: relative-image-path -> _join_image_path/full-control
def _join_image_path(images_root: Path, file_name_value: str) -> Path:
    file_name = Path(str(file_name_value))
    if (
        file_name.is_absolute()
        or not file_name.parts
        or any(part in {"", ".", ".."} for part in file_name.parts)
    ):
        raise ContractError("COCO image member must be one confined relative path")
    root = Path(os.path.abspath(os.fspath(images_root)))
    candidate = Path(os.path.abspath(os.fspath(root / file_name)))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ContractError("COCO image member escapes the declared tiles root") from exc
    cursor = root
    for part in file_name.parts[:-1]:
        cursor /= part
        try:
            ancestor = cursor.lstat()
        except OSError as exc:
            raise ContractError("COCO image member ancestor is unavailable") from exc
        if stat.S_ISLNK(ancestor.st_mode) or not stat.S_ISDIR(ancestor.st_mode):
            raise ContractError("COCO image member ancestor must be a real directory")
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise ContractError("COCO image member is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ContractError("COCO image member must be a regular non-symlink file")
    return candidate


# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_STATEMENT_MAP: positive-prototype-threshold -> _make_unified_prototype/full-algorithm
def _make_unified_prototype(
    positive_embeddings: Path,
    prototype_csv: Path,
    threshold_method: str = "fixed",
    fixed_threshold: float = 0.70,
    percentile: float = 5.0,
) -> tuple[Path | None, float | None, int | None]:
    import numpy as np

    embeddings = np.load(positive_embeddings).astype(np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] < 1:
        return None, None, None
    embeddings = _l2_normalize(embeddings, axis=1)
    mean = _l2_normalize(embeddings.mean(axis=0, keepdims=False))[...]
    similarities = embeddings @ mean
    threshold = (
        float(np.percentile(similarities, float(percentile)))
        if threshold_method == "percentile"
        else float(fixed_threshold)
    )
    _write_prototype_atomic(prototype_csv, mean, threshold)
    return prototype_csv, float(threshold), int(mean.shape[0])


# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_STATEMENT_MAP: all-embedding-pca -> _fit_pca_from_embeddings/full-algorithm
def _fit_pca_from_embeddings(
    all_embeddings: Path,
    output_npz: Path,
    dimensions: int = 32,
) -> tuple[Path | None, tuple[int, int] | None, int | None]:
    import numpy as np

    embeddings = np.load(all_embeddings).astype(np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return None, None, None
    mean_matrix = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mean_matrix
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    components = right[:dimensions].astype(np.float32)
    mean = mean_matrix.squeeze(0).astype(np.float32)
    _write_numpy_array_atomic(output_npz, (components, mean), archive=True)
    return (
        output_npz,
        (int(components.shape[0]), int(components.shape[1])),
        int(mean.shape[0]),
    )


# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_STATEMENT_MAP: resolve-allowed-image-ids -> _resolve_allowed_image_ids
def _resolve_allowed_image_ids(
    images: Iterable[Mapping[str, Any]],
    allowed_tiled_ids: Iterable[int],
    allowed_original_ids: Iterable[int],
) -> set[int]:
    direct = {int(value) for value in allowed_tiled_ids}
    if direct:
        return direct
    return _original_to_tiled_ids(images, allowed_original_ids)


# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_STATEMENT_MAP: explicit-predecessor-pack -> _load_previous_feature_pack
def _load_previous_feature_pack(
    backend: FeatureBackend,
    pack_path: Path | None,
    *,
    incremental: bool,
) -> Mapping[str, Any]:
    if not incremental or pack_path is None:
        return {}
    return backend.load_feature_pack(pack_path)


# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_CELL: NB-LIVE-0001-C0004
# SOURCE_STATEMENT_MAP: backup-existing-feature-outputs -> _backup_feature_outputs
def _backup_feature_outputs(
    artifacts: Iterable[Path],
    backup_directory: Path | None,
    metadata_name: str,
    metadata: Mapping[str, Any],
) -> tuple[Path, ...]:
    if backup_directory is None:
        return ()
    existing = [path for path in artifacts if path.is_file() and not path.is_symlink()]
    if not existing:
        return ()
    backup_directory.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in existing:
        destination = backup_directory / source.name
        if _safe_copy(source, destination):
            copied.append(destination)
    _write_json_atomic(
        backup_directory / metadata_name,
        {
            **dict(metadata),
            "backed_up_files": [str(path) for path in existing],
        },
    )
    return tuple(copied)


# SOURCE_CELL: NB-LIVE-0001-C0001
# SOURCE_CELL: NB-LIVE-0001-C0002
# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_CELL: NB-LIVE-0001-C0004
# SOURCE_STATEMENT_MAP: safe-backup-copy -> _safe_copy
def _safe_copy(source: Path, destination: Path) -> bool:
    try:
        if source.exists():
            shutil.copy2(str(source), str(destination))
            return True
    except Exception:
        pass
    return False


# SOURCE_CELL: NB-LIVE-0001-C0004
# SOURCE_STATEMENT_MAP: exact-safe-division -> _safe_divide/full-expression
def _safe_divide(left: Any, right: Any) -> float:
    return float(left) / float(right) if float(right) != 0.0 else 0.0


# SOURCE_CELL: NB-LIVE-0001-C0004
# SOURCE_STATEMENT_MAP: exact-unit-clamp -> _clamp/full-expression
def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


# SOURCE_CELL: NB-LIVE-0001-C0004
# SOURCE_STATEMENT_MAP: image-positive-index -> _build_image_positive_index/full-control
def _build_image_positive_index(
    coco_path: Path,
    positive_names: set[str],
) -> tuple[dict[int, bool], dict[int, str], set[int]]:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    category_names = {
        int(category["id"]): (category.get("name", "") or "")
        for category in data.get("categories", [])
    }
    normalized_names = {name.casefold() for name in positive_names}
    positive_ids = {
        category_id
        for category_id, name in category_names.items()
        if (name or "").casefold() in normalized_names
    }
    image_has_positive = {
        int(image["id"]): False for image in data.get("images", [])
    }
    for annotation in data.get("annotations", []):
        if int(annotation.get("category_id", -1)) in positive_ids:
            image_has_positive[int(annotation.get("image_id", -1))] = True
    return image_has_positive, category_names, positive_ids


# SOURCE_CELL: NB-LIVE-0001-C0004
# SOURCE_STATEMENT_MAP: mixed-subset-validation -> _validate_mixed_subset/full-control
def _validate_mixed_subset(
    allowed_ids: set[int],
    image_has_positive: Mapping[int, bool],
) -> tuple[bool, int, int]:
    import numpy as np

    flags = [bool(image_has_positive.get(image_id, False)) for image_id in allowed_ids]
    positive = int(np.sum(flags))
    negative = int(len(flags) - positive)
    return (positive > 0 and negative > 0), positive, negative


# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_STATEMENT_MAP: top-level:038-056 -> build_fold_safe_feature_pack/full-algorithm
def build_fold_safe_feature_pack(
    request: FeaturePackRequest,
    backend: FeatureBackend,
    services: HandlerServices,
) -> FeaturePackResult:
    """Build incremental train-only embeddings, prototype, PCA, and pack."""

    for label, path in (
        ("prototype", request.prototype_output),
        ("PCA", request.pca_output),
        ("used-ID", request.used_ids_output),
        ("positive-embedding", request.positive_embeddings_output),
        ("all-embedding", request.all_embeddings_output),
        ("pack", request.pack_output),
    ):
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ContractError(f"feature {label} output cannot be inspected") from exc
        else:
            raise ContractError(f"feature {label} output must be absent")
    import numpy as np

    if request.run_mode == "skip":
        services.report("feature pack skipped by explicit typed run mode")
        return FeaturePackResult(0, 0, request.fixed_threshold, ())

    coco = _read_json_object(request.coco_annotations, "tiled COCO")
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    image_by_id = {int(image["id"]): image for image in images}
    allowed = _resolve_allowed_image_ids(
        images,
        request.allowed_tiled_ids,
        request.allowed_original_ids,
    )
    if not allowed:
        raise ContractError("feature-pack construction requires an explicit train subset")
    category_ids = {
        int(category["id"])
        for category in coco.get("categories", [])
        if str(category.get("name", "")).casefold()
        == request.positive_category.casefold()
    }
    positive_path = request.positive_embeddings_output
    all_path = request.all_embeddings_output
    pack_path = request.pack_output
    delta_ids = set(allowed)
    positive_vectors: list[Any] = []
    all_vectors: list[Any] = []
    built_vectors = 0
    if request.run_mode in {"all", "stage1"}:
        ids_to_build = allowed
        for annotation in annotations:
            image_id = int(annotation.get("image_id", -1))
            if image_id not in ids_to_build:
                continue
            image_info = image_by_id.get(image_id)
            if image_info is None:
                continue
            image_path = _join_image_path(
                request.images,
                str(image_info.get("file_name", "")),
            )
            image = backend.read_image(image_path)
            if image is None:
                continue
            height, width = image.shape[:2]
            mask = _segmentation_mask(backend, annotation, height, width)
            for scale in request.train_scales:
                effective_scale = _effective_scale(scale)
                try:
                    vector = _l2_normalize(
                        backend.embedding(
                            image,
                            mask,
                            request.pad_fraction,
                            scale=effective_scale,
                        )
                    ).reshape(-1)
                except Exception:
                    continue
                all_vectors.append(vector)
                if int(annotation.get("category_id", -1)) in category_ids:
                    positive_vectors.append(vector)
                built_vectors += 1
                if request.sample_limit and built_vectors >= request.sample_limit:
                    break
            if request.sample_limit and built_vectors >= request.sample_limit:
                break

        if ids_to_build:
            dimension = next(
                (int(vector.shape[-1]) for vector in all_vectors),
                2048,
            )
            positive_delta = (
                np.stack(positive_vectors).astype(np.float32)
                if positive_vectors
                else np.zeros((0, dimension), dtype=np.float32)
            )
            all_delta = (
                np.stack(all_vectors).astype(np.float32)
                if all_vectors
                else np.zeros((0, dimension), dtype=np.float32)
            )
            if positive_delta.shape[0] < 1 or all_delta.shape[0] < 2:
                raise ContractError(
                    "feature-pack construction produced insufficient fresh embeddings"
                )
            _write_numpy_array_atomic(positive_path, positive_delta)
            _write_numpy_array_atomic(all_path, all_delta)

    positive = (
        np.load(positive_path).astype(np.float32)
        if positive_path.is_file()
        else np.zeros((0, 2048), dtype=np.float32)
    )
    all_embeddings = (
        np.load(all_path).astype(np.float32)
        if all_path.is_file()
        else np.zeros((0, 2048), dtype=np.float32)
    )

    threshold = float(request.fixed_threshold)
    mean = None
    prototype_written = False
    if request.run_mode in {"all", "stage2"}:
        prototype_result = _make_unified_prototype(
            positive_path,
            request.prototype_output,
            request.threshold_method,
            request.fixed_threshold,
            request.percentile,
        )
        if prototype_result[0] is not None:
            threshold = float(prototype_result[1])
            prototype_written = True

    components = None
    pca_mean = None
    pca_written = False
    if request.run_mode in {"all", "stage3"}:
        pca_result = _fit_pca_from_embeddings(
            all_path,
            request.pca_output,
            request.pca_dimensions,
        )
        pca_written = pca_result[0] is not None

    prototype_artifact = _read_prototype_artifact(
        request.prototype_output,
        request.fixed_threshold,
    )
    if prototype_artifact is not None:
        mean_values, threshold = prototype_artifact
        mean = np.asarray(mean_values, dtype=np.float32)
    if request.pca_output.is_file():
        with np.load(request.pca_output) as pca_artifact:
            components = np.asarray(pca_artifact["components"], dtype=np.float32)
            pca_mean = np.asarray(pca_artifact["mean"], dtype=np.float32)

    ordered_used = tuple(sorted(allowed))
    pack: dict[str, Any] = {"used_img_ids": ordered_used}
    if mean is not None:
        pack["mu"] = _l2_normalize(mean).reshape(-1).astype(np.float32)
        pack["embed_thresh"] = threshold
    if components is not None and pca_mean is not None:
        pack["pca_components"] = np.asarray(components, dtype=np.float32)
        pack["pca_mean"] = np.asarray(pca_mean, dtype=np.float32)
    atomic_write_output(
        pack_path,
        lambda handle: backend.write_feature_pack(handle, pack),
    )
    _write_json_atomic(request.used_ids_output, ordered_used)
    services.report(
        f"feature pack ready: positive={len(positive)} all={len(all_embeddings)} "
        f"new_train_images={len(delta_ids)}"
    )
    return FeaturePackResult(
        positive_embeddings=len(positive),
        all_embeddings=len(all_embeddings),
        threshold=threshold,
        used_image_ids=ordered_used,
        delta_image_ids=tuple(sorted(delta_ids)),
        prototype_written=prototype_written,
        pca_written=pca_written,
        pack_output=pack_path,
    )


@dataclass(frozen=True)
class FeatureExtractionRequest:
    coco_annotations: Path
    images: Path
    prototype: Path
    pca: Path
    output_csv: Path
    allowed_image_ids: tuple[int, ...]
    review_labels: Path | None = None
    feature_mode: str = "ultra"
    multiscale_scales: tuple[float, ...] = MULTISCALE_SCALES
    pad_fraction: float = 0.10
    positive_category: str = "cj"

    def __post_init__(self) -> None:
        if self.feature_mode not in {"balanced", "rich", "ultra"}:
            raise ContractError("feature mode is invalid")
        if tuple(float(scale) for scale in self.multiscale_scales) != MULTISCALE_SCALES:
            raise ContractError("sealed extraction scales must remain unchanged")


@dataclass(frozen=True)
class FeatureExtractionResult:
    output_csv: Path
    row_count: int
    feature_count: int
    appended_rows: int = 0
    skipped_existing: int = 0
    migrations: tuple[str, ...] = ()


# SOURCE_CELL: NB-LIVE-0001-C0004
# SOURCE_STATEMENT_MAP: review-column-autodetect -> _review_tags/full-control
def _review_tags(path: Path | None) -> dict[tuple[str, str], str]:
    if path is None or not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or [])
    best_field: str | None = None
    best_rate = 0.0
    for field in fields:
        matches = sum(
            str(row.get(field, "")).strip().casefold() in REVIEW_WEIGHTS
            for row in rows
        )
        rate = float(matches) / float(len(rows)) if rows else 0.0
        if rate > best_rate:
            best_rate = rate
            best_field = field
    tag_field = best_field if best_rate >= 0.01 else None
    result: dict[tuple[str, str], str] = {}
    for index, row in enumerate(rows):
        image = Path(
            str(
                row.get("image")
                or row.get("file_name")
                or row.get("file")
                or row.get("filename")
                or ""
            )
        ).name
        identifier = str(row.get("id", index))
        if image:
            result[(image, identifier)] = (
                str(row.get(tag_field, "")).strip() if tag_field else ""
            )
    return result


# SOURCE_CELL: NB-LIVE-0001-C0004
# SOURCE_STATEMENT_MAP: finite-float-default -> _safe_float/full-control
def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


# SOURCE_CELL: NB-LIVE-0001-C0004
# SOURCE_STATEMENT_MAP: safe-div-clamp-quality-gates -> _feature_gate_values/full-algorithm
def _feature_gate_values(
    features: Mapping[str, Any],
    pca_values: Sequence[float],
    *,
    has_prototype: bool,
) -> dict[str, float]:
    mean_l = _safe_float(features.get("mean_L"))
    std_l = _safe_float(features.get("std_L"))
    delta_a = _safe_float(features.get("delta_a"))
    delta_b = _safe_float(features.get("delta_b"))
    eccentricity = _safe_float(features.get("eccentricity"))
    circularity = _safe_float(features.get("circularity"))
    solidity = _safe_float(features.get("solidity"))
    touching_border = int(_safe_float(features.get("touching_border")))
    component_count = int(_safe_float(features.get("components_count"), 1.0))
    extent = _safe_float(features.get("extent"))
    gradient_p90 = _safe_float(features.get("grad_p90"))
    embed_similarity = _safe_float(features.get("embed_sim"), 0.0)

    light = _clamp(
        0.5 * _safe_divide(mean_l, 255.0) + 0.5 * _clamp(std_l / 64.0)
    )
    color = _clamp(math.hypot(delta_a, delta_b) / 200.0)
    shape = _clamp(
        0.4 * circularity
        + 0.4 * solidity
        + 0.2 * (1.0 - _clamp(eccentricity))
    )
    embed = embed_similarity if has_prototype else 0.5
    robust = _clamp(
        0.6 * (1.0 if component_count == 1 else 0.6)
        + 0.2 * (1.0 - abs(extent - 0.5) * 2.0)
        + 0.2 * _clamp(gradient_p90 / 200.0)
    )
    robust *= 0.7 if touching_border else 1.0
    if pca_values:
        distance = math.sqrt(sum(float(value) ** 2 for value in pca_values))
        mahalanobis = _clamp(1.0 - distance / (distance + 5.0))
    else:
        mahalanobis = 0.5
    border = 1.0 if touching_border else 0.0
    quality_parts = (
        light,
        color,
        shape,
        embed,
        robust,
        mahalanobis,
        1.0 - 0.5 * border,
    )
    return {
        "g_quality": _clamp(sum(quality_parts) / len(quality_parts)),
        "g_light": light,
        "g_color": color,
        "g_shape": shape,
        "g_embed": embed,
        "g_robust": robust,
        "g_maha": mahalanobis,
        "g_border": border,
    }


# SOURCE_CELL: NB-LIVE-0001-C0004
# SOURCE_STATEMENT_MAP: append-resume-triples -> _load_done_triples/artifact-io
def _load_done_triples(path: Path) -> set[tuple[str, str, str]]:
    if not path.is_file():
        return set()
    completed: set[tuple[str, str, str]] = set()
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or "file_name" not in header or "ann_id" not in header:
            return set()
        file_index = header.index("file_name")
        annotation_index = header.index("ann_id")
        scale_index = header.index("scale") if "scale" in header else -1
        for row in reader:
            if max(file_index, annotation_index) >= len(row):
                continue
            scale = row[scale_index] if scale_index >= 0 and scale_index < len(row) else "*"
            completed.add((row[file_index], row[annotation_index], scale.strip() or "*"))
    return completed


# SOURCE_CELL: NB-LIVE-0001-C0004
# SOURCE_STATEMENT_MAP: scale-review-tag-schema-migration -> _migrate_feature_column/full-control
def _migrate_feature_column(
    path: Path,
    column: str,
    *,
    after: str | None,
    before: str | None,
    fallback_index: int | None,
    backup_suffix: str,
) -> bool:
    if LEGACY_FEATURE_BACKUP_SUFFIXES.get(column) != backup_suffix:
        raise ContractError("feature CSV migration requires the registered backup suffix")
    if fallback_index is not None and (
        isinstance(fallback_index, bool)
        or not isinstance(fallback_index, int)
        or fallback_index < 0
    ):
        raise ContractError("feature CSV fallback index must be a nonnegative integer")
    try:
        source_node = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ContractError("feature CSV migration source cannot be inspected") from exc
    if stat.S_ISLNK(source_node.st_mode) or not stat.S_ISREG(source_node.st_mode):
        raise ContractError("feature CSV migration source must be a regular non-symlink file")
    source_identity = (
        source_node.st_dev,
        source_node.st_ino,
        source_node.st_mode,
        source_node.st_size,
        source_node.st_mtime_ns,
        source_node.st_ctime_ns,
    )
    backup = path.with_suffix(backup_suffix)
    try:
        backup.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ContractError("feature CSV migration artifact cannot be inspected") from exc
    else:
        raise ContractError("feature CSV migration artifact already exists")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_source = os.fstat(descriptor)
        opened_identity = (
            opened_source.st_dev,
            opened_source.st_ino,
            opened_source.st_mode,
            opened_source.st_size,
            opened_source.st_mtime_ns,
            opened_source.st_ctime_ns,
        )
        if not stat.S_ISREG(opened_source.st_mode) or opened_identity != source_identity:
            raise ContractError("feature CSV migration source identity changed")
        source = os.fdopen(
            descriptor,
            "r",
            newline="",
            encoding="utf-8",
            errors="replace",
        )
        descriptor = -1
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        raise ContractError("feature CSV migration source cannot be opened safely") from exc
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        raise
    try:
        with source:
            reader = csv.reader(source)
            old_header = next(reader, None)
            if not old_header or column in old_header:
                return False
            if after and after in old_header:
                insertion = old_header.index(after) + 1
            elif before and before in old_header:
                insertion = old_header.index(before)
            elif fallback_index is not None:
                insertion = fallback_index
            else:
                insertion = len(old_header)
            new_header = old_header[:insertion] + [column] + old_header[insertion:]

            def write_migration(handle: Any) -> None:
                destination = io.TextIOWrapper(
                    handle,
                    encoding="utf-8",
                    newline="",
                    write_through=True,
                )
                try:
                    writer = csv.writer(destination)
                    writer.writerow(new_header)
                    old_width = len(old_header)
                    for row in reader:
                        if len(row) < old_width:
                            row += [""] * (old_width - len(row))
                        elif len(row) > old_width:
                            row = row[: old_width - 1] + [",".join(row[old_width - 1 :])]
                        writer.writerow(row[:insertion] + [""] + row[insertion:])
                    destination.flush()
                finally:
                    destination.detach()

            atomic_write_output(
                path,
                write_migration,
                backup_path=backup,
                expected_source_identity=source_identity,
            )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return True


def migrate_legacy_feature_csv(path: Path, column: str) -> bool:
    """Migrate one legacy feature column and retain the exact source as a backup."""

    if column == "scale":
        return _migrate_feature_column(
            path,
            column,
            after="ann_id",
            before=None,
            fallback_index=2,
            backup_suffix=LEGACY_FEATURE_BACKUP_SUFFIXES[column],
        )
    if column == "review_tag":
        return _migrate_feature_column(
            path,
            column,
            after="reviewed",
            before="class_id",
            fallback_index=None,
            backup_suffix=LEGACY_FEATURE_BACKUP_SUFFIXES[column],
        )
    raise ContractError("legacy feature column must be scale or review_tag")


# SOURCE_CELL: NB-LIVE-0001-C0004
# SOURCE_STATEMENT_MAP: output-schema-read -> _csv_header_and_rows/artifact-io
def _csv_header_and_rows(path: Path) -> tuple[list[str], int]:
    if not path.is_file():
        return [], 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        return header, sum(1 for _ in reader)


# SOURCE_CELL: NB-LIVE-0001-C0004
# SOURCE_STATEMENT_MAP: top-level:036-061 -> extract_features_fold_safe/full-algorithm
def extract_features_fold_safe(
    request: FeatureExtractionRequest,
    backend: FeatureBackend,
    services: HandlerServices,
) -> FeatureExtractionResult:
    """Extract the ordered source feature schema once per annotation and scale."""

    try:
        request.output_csv.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ContractError("feature output cannot be inspected") from exc
    else:
        raise ContractError("feature output must be absent under fail-on-collision")
    image_has_positive, category_names, positive_ids = _build_image_positive_index(
        request.coco_annotations,
        {request.positive_category},
    )
    coco = _read_json_object(request.coco_annotations, "tiled COCO")
    image_by_id = {
        int(image["id"]): image for image in coco.get("images", [])
    }
    allowed = {int(value) for value in request.allowed_image_ids}
    if not allowed:
        raise ContractError("feature extraction requires explicit allowed image IDs")
    mixed, positive_images, negative_images = _validate_mixed_subset(
        allowed,
        image_has_positive,
    )
    if not mixed:
        raise ContractError("allowed feature subset must contain positive and negative images")

    review = _review_tags(request.review_labels)
    rows: list[dict[str, Any]] = []
    skipped_existing = 0
    image_cache: dict[int, Any] = {}
    pca_component_count = (
        backend.pca_component_count(request.pca)
        if request.feature_mode == "ultra" and request.pca.is_file()
        else 0
    )
    if pca_component_count < 0:
        raise ContractError("PCA component count cannot be negative")
    selected_annotations = [
        annotation
        for annotation in coco.get("annotations", [])
        if int(annotation.get("image_id", -1)) in allowed
    ]
    for annotation_index, annotation in enumerate(selected_annotations):
        image_id = int(annotation.get("image_id", -1))
        image_info = image_by_id.get(image_id)
        if image_info is None:
            continue
        image_member = str(image_info.get("file_name", ""))
        image_name = Path(image_member).name
        identifier = str(annotation.get("id", annotation_index))
        if image_id not in image_cache:
            image_cache[image_id] = backend.read_image(
                _join_image_path(request.images, image_member)
            )
        image = image_cache[image_id]
        if image is None:
            continue
        height, width = image.shape[:2]
        mask = _segmentation_mask(backend, annotation, height, width)
        review_id = str(annotation.get("meta", {}).get("review_id", -1))
        review_key = (image_name, review_id)
        reviewed = int(review_key in review)
        review_tag = review.get(review_key, "").strip() if reviewed else ""
        category_id = int(annotation.get("category_id", -1))
        for scale in request.multiscale_scales:
            effective_scale = _effective_scale(scale)
            scale_key = round(effective_scale, 4)
            scale_text = f"{scale_key:.4f}"
            features = dict(
                backend.compute_features(
                    image,
                    mask,
                    mode=request.feature_mode,
                    scale=effective_scale,
                    prototype=request.prototype,
                )
            )
            if not features:
                continue
            background_a, background_b = backend.background_ab(
                image,
                scale=effective_scale,
            )
            features["delta_a"] = float(background_a) - _safe_float(
                features.get("mean_a")
            )
            features["delta_b"] = _safe_float(features.get("mean_b")) - float(
                background_b
            )
            scaled_height, scaled_width = backend.scaled_shape(
                image,
                scale=effective_scale,
            )
            scaled_height = max(1.0, float(scaled_height))
            scaled_width = max(1.0, float(scaled_width))
            features["grid_r_norm"] = _safe_float(features.get("cy")) / scaled_height
            features["grid_c_norm"] = _safe_float(features.get("cx")) / scaled_width
            features["grid_r"] = max(
                0,
                min(2, int(features["grid_r_norm"] * 3)),
            )
            features["grid_c"] = max(
                0,
                min(2, int(features["grid_c_norm"] * 3)),
            )
            if any(name not in features for name in ("hu1", "hu2", "hu4")):
                raw_hu = list(
                    backend.hu_moments(
                        mask,
                        scale=effective_scale,
                        binary_image=True,
                    )
                )
                if len(raw_hu) < 4:
                    raise ContractError("feature backend returned fewer than four Hu moments")
                transformed_hu = [
                    0.0
                    if float(value) == 0.0
                    else math.copysign(math.log1p(abs(float(value))), float(value))
                    for value in raw_hu
                ]
                features["hu1"] = transformed_hu[0]
                features["hu2"] = transformed_hu[1]
                features["hu4"] = transformed_hu[3]

            embedding = _l2_normalize(
                backend.embedding(
                    image,
                    mask,
                    request.pad_fraction,
                    scale=effective_scale,
                )
            ).reshape(-1)
            has_prototype = request.prototype.is_file()
            if has_prototype:
                cosine_similarity = _safe_float(
                    backend.prototype_dot_product(embedding, request.prototype)
                )
                features["embed_sim"] = max(
                    0.0,
                    min(1.0, (cosine_similarity + 1.0) * 0.5),
                )
            else:
                features["embed_sim"] = 0.0
            projection: list[float] = []
            if request.feature_mode == "ultra" and pca_component_count:
                try:
                    projection = [
                        _safe_float(value)
                        for value in backend.project_embedding(embedding, request.pca)
                    ]
                except Exception:
                    projection = [0.0] * pca_component_count
                if len(projection) != pca_component_count:
                    raise ContractError(
                        "feature backend returned an inconsistent PCA dimension"
                    )
            for index, value in enumerate(projection):
                features[f"embed_pca_{index}"] = value
            features["pred_iou"] = 1.0
            features["stability"] = 1.0
            features.update(
                _feature_gate_values(
                    features,
                    projection,
                    has_prototype=has_prototype,
                )
            )
            row: dict[str, Any] = {
                "file_name": image_name,
                "ann_id": identifier,
                "scale": scale_text,
                "reviewed": reviewed,
                "review_tag": review_tag,
                "class_id": category_id,
                "class_name": category_names.get(category_id, "").strip(),
                "label": int(category_id in positive_ids),
            }
            integer_columns = {
                "area_px",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "components_count",
                "touching_border",
                "grid_r",
                "grid_c",
            }
            selected_features = FEATURE_BASE_COLUMNS[3:] + (
                FEATURE_RICH_COLUMNS
                if request.feature_mode in {"rich", "ultra"}
                else ()
            )
            for name in selected_features:
                if name in integer_columns:
                    row[name] = int(
                        _safe_float(
                            features.get(name),
                            1.0 if name == "components_count" else 0.0,
                        )
                    )
                else:
                    row[name] = _safe_float(features.get(name))
            for index, value in enumerate(projection):
                row[f"embed_pca_{index}"] = value
            rows.append(row)

    pca_counts = {
        len([name for name in row if name.startswith("embed_pca_")])
        for row in rows
    }
    if len(pca_counts) > 1:
        raise ContractError("feature backend returned inconsistent PCA dimensions")
    pca_count = pca_component_count if request.feature_mode == "ultra" else 0
    if rows and next(iter(pca_counts), 0) != pca_count:
        raise ContractError("feature row PCA width differs from the sealed PCA width")
    pca_columns = tuple(f"embed_pca_{index}" for index in range(pca_count))
    columns = list(FEATURE_BASE_COLUMNS)
    if request.feature_mode in {"rich", "ultra"}:
        columns.extend(FEATURE_RICH_COLUMNS)
    columns.extend(pca_columns)
    columns.extend(FEATURE_TRAILING_COLUMNS)
    def write_feature_table(handle: Any) -> None:
        destination = io.TextIOWrapper(
            handle,
            encoding="utf-8",
            newline="",
            write_through=True,
        )
        try:
            writer = csv.DictWriter(
                destination,
                fieldnames=columns,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
            destination.flush()
        finally:
            destination.detach()

    atomic_write_output(request.output_csv, write_feature_table)
    services.report(
        f"feature table written: appended={len(rows)} skipped={skipped_existing} "
        f"features={len(columns) - len(FEATURE_TRAILING_COLUMNS)}"
    )
    return FeatureExtractionResult(
        output_csv=request.output_csv,
        row_count=len(rows),
        feature_count=len(columns) - len(FEATURE_TRAILING_COLUMNS),
        appended_rows=len(rows),
        skipped_existing=skipped_existing,
        migrations=(),
    )


__all__ = [
    "FEATURE_SCHEMA_BACKUP_POLICY",
    "LEGACY_FEATURE_BACKUP_SUFFIXES",
    "FeatureBackend",
    "FeatureExtractionRequest",
    "FeatureExtractionResult",
    "FeaturePackRequest",
    "FeaturePackResult",
    "ReviewMergeRequest",
    "ReviewMergeResult",
    "SegmentationPolicy",
    "SplitRequest",
    "SplitResult",
    "build_fold_safe_feature_pack",
    "extract_features_fold_safe",
    "freeze_or_create_splits",
    "merge_review_labels_into_coco",
    "migrate_legacy_feature_csv",
]
