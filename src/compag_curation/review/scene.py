"""Read-only, full-image scene backends shared by review frontends.

The scene layer deliberately owns no review state.  It verifies sealed source
artifacts, exposes mask-derived outline geometry separately from bounding
boxes, and renders deterministic PNG bytes without writing a cache into the
scientific run or the append-only human-decision journal.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from compag_curation.public_config import MAX_IMAGE_FILE_BYTES
from compag_curation.public_io import (
    PublicIOError,
    canonical_json_value,
    compact_json_sha256,
    hash_file_snapshot,
    portable_basename,
    stable_file,
)


SCENE_SCHEMA = "compag-curation-full-image-review-scene/v1"
STAGE20_SCENE_KIND = "SEALED_STAGE10_FULL_IMAGE_MOSAIC"
FULL_IMAGE_STAGE20_SCENE_KIND = "SEALED_STAGE10_FULL_WARPED_FRAME"
ROUND_SCENE_KIND = "VERIFIED_STAGE60_ORIGINAL_FULL_IMAGE"
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_TABLE_BYTES = 2 * 1024 * 1024 * 1024
MAX_TILE_BYTES = 32 * 1024 * 1024
MAX_SCENE_PNG_BYTES = 512 * 1024 * 1024
MAX_SCENE_DIMENSION = 32_768
MAX_SCENE_PIXELS = 256_000_000
_SHA256 = re.compile(r"[0-9a-f]{64}")

_TILE_COLUMNS = (
    "tile_name",
    "image_id",
    "image_sha256",
    "image_name",
    "group_id",
    "tile_sha256",
    "x",
    "y",
    "crop_w",
    "crop_h",
    "orig_w",
    "orig_h",
)

_FULL_IMAGE_PREPARE_SCHEMA = "compag-curation-full-image-prepare/v1"
_FULL_IMAGE_PROFILE = "full-image-multiscale-xgb-recall-gpu-v1"
_FULL_IMAGE_SPATIAL_MODE = "full-image-multiscale"
_FULL_IMAGE_UNIT_KIND = "full_warped_frame"
_FULL_IMAGE_UNIT_COLUMNS = (
    "unit_name",
    "image_id",
    "image_sha256",
    "image_name",
    "group_id",
    "unit_sha256",
    "x",
    "y",
    "width",
    "height",
    "orig_w",
    "orig_h",
    "processing_unit_kind",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


def _strict_int(
    value: object,
    role: str,
    *,
    minimum: int = 0,
    maximum: int = 1_000_000_000,
) -> int:
    text = str(value)
    _require(
        re.fullmatch(r"0|[1-9][0-9]*", text) is not None,
        f"{role} is not a canonical integer",
    )
    result = int(text)
    _require(minimum <= result <= maximum, f"{role} is outside its supported range")
    return result


def _strict_float(value: object, role: str) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise PublicIOError(f"{role} is not numeric") from exc
    _require(math.isfinite(result), f"{role} is not finite")
    return result


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@dataclass(frozen=True)
class BoundFile:
    path: Path
    sha256: str
    size_bytes: int
    identity: tuple[int, ...]

    @classmethod
    def capture(
        cls,
        path: Path,
        role: str,
        *,
        max_bytes: int,
        expected_sha256: str | None = None,
    ) -> tuple["BoundFile", bytes]:
        payload, info = stable_file(path, max_bytes=max_bytes)
        _require(
            stat.S_ISREG(info.st_mode)
            and not path.is_symlink()
            and info.st_nlink == 1,
            f"{role} must be a single-link regular file",
        )
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None:
            _require(
                _SHA256.fullmatch(expected_sha256) is not None
                and digest == expected_sha256,
                f"{role} differs from its sealed SHA-256",
            )
        return cls(path, digest, len(payload), _file_identity(info)), payload

    @classmethod
    def capture_streaming(
        cls,
        path: Path,
        role: str,
        *,
        max_bytes: int,
        expected_sha256: str | None = None,
    ) -> "BoundFile":
        if expected_sha256 is not None:
            _require(
                _SHA256.fullmatch(expected_sha256) is not None,
                f"{role} has an invalid sealed SHA-256",
            )
        try:
            initial = path.lstat()
        except OSError as exc:
            raise PublicIOError(f"{role} is unavailable") from exc
        _require(
            stat.S_ISREG(initial.st_mode)
            and not path.is_symlink()
            and initial.st_nlink == 1
            and 0 <= initial.st_size <= max_bytes,
            f"{role} must be a bounded single-link regular file",
        )
        digest, info = hash_file_snapshot(
            path,
            require_single_link=True,
            max_bytes=max_bytes,
        )
        _require(
            _file_identity(info) == _file_identity(initial),
            f"{role} changed before scene indexing",
        )
        if expected_sha256 is not None:
            _require(digest == expected_sha256, f"{role} differs from its sealed SHA-256")
        return cls(path, digest, info.st_size, _file_identity(info))

    def check_metadata(self, role: str) -> None:
        try:
            info = self.path.lstat()
        except OSError as exc:
            raise PublicIOError(f"{role} disappeared after scene indexing") from exc
        _require(
            stat.S_ISREG(info.st_mode)
            and not self.path.is_symlink()
            and _file_identity(info) == self.identity,
            f"{role} changed after scene indexing",
        )

    def read_verified(self, role: str, *, max_bytes: int) -> bytes:
        self.check_metadata(role)
        payload, info = stable_file(self.path, max_bytes=max_bytes)
        _require(
            _file_identity(info) == self.identity
            and len(payload) == self.size_bytes
            and hashlib.sha256(payload).hexdigest() == self.sha256,
            f"{role} changed after scene indexing",
        )
        return payload


@dataclass(frozen=True)
class SceneTile:
    name: str
    source: BoundFile
    x: int
    y: int
    crop_w: int
    crop_h: int


@dataclass(frozen=True)
class SceneImage:
    scene_id: str
    index: int
    image_id: str
    image_sha256: str
    image_name: str
    group_id: str
    width: int
    height: int
    tiles: tuple[SceneTile, ...] = ()
    source: BoundFile | None = None


@dataclass(frozen=True)
class SceneProposal:
    proposal_id: str
    image_id: str
    label_prefix: str
    bbox: tuple[int, int, int, int]
    polygon: tuple[int, ...]
    model: Mapping[str, object] | None = None


def _parse_csv(payload: bytes, columns: tuple[str, ...], role: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise PublicIOError(f"{role} is not UTF-8") from exc
    _require(not text.startswith("\ufeff"), f"{role} contains a byte-order mark")
    try:
        with io.StringIO(text, newline="") as handle:
            reader = csv.DictReader(handle)
            _require(tuple(reader.fieldnames or ()) == columns, f"{role} header mismatch")
            rows: list[dict[str, str]] = []
            for row in reader:
                _require(
                    None not in row
                    and tuple(row) == columns
                    and all(isinstance(value, str) for value in row.values()),
                    f"{role} row has the wrong field count",
                )
                rows.append(row)
    except csv.Error as exc:
        raise PublicIOError(f"{role} contains invalid CSV framing") from exc
    _require(bool(rows), f"{role} is empty")
    return rows


def _invert_3x3(matrix: Sequence[Sequence[object]]) -> tuple[tuple[float, ...], ...]:
    _require(
        len(matrix) == 3 and all(len(row) == 3 for row in matrix),
        "scene inverse warp is not a 3x3 matrix",
    )
    a, b, c = (_strict_float(value, "scene inverse warp") for value in matrix[0])
    d, e, f = (_strict_float(value, "scene inverse warp") for value in matrix[1])
    g, h, i = (_strict_float(value, "scene inverse warp") for value in matrix[2])
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    _require(math.isfinite(determinant) and abs(determinant) > 1e-12, "scene inverse warp is singular")
    inverse = (
        ((e * i - f * h) / determinant, (c * h - b * i) / determinant, (b * f - c * e) / determinant),
        ((f * g - d * i) / determinant, (a * i - c * g) / determinant, (c * d - a * f) / determinant),
        ((d * h - e * g) / determinant, (b * g - a * h) / determinant, (a * e - b * d) / determinant),
    )
    _require(all(math.isfinite(value) for row in inverse for value in row), "scene forward warp is nonfinite")
    return inverse


def _transform_polygon(
    polygon: Sequence[int],
    forward: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> tuple[int, ...]:
    _require(len(polygon) % 2 == 0, "scene proposal polygon has odd length")
    result: list[int] = []
    prior: tuple[int, int] | None = None
    for source_x, source_y in zip(polygon[0::2], polygon[1::2]):
        denominator = forward[2][0] * source_x + forward[2][1] * source_y + forward[2][2]
        _require(math.isfinite(denominator) and abs(denominator) > 1e-12, "scene polygon crosses an invalid homography")
        x = (forward[0][0] * source_x + forward[0][1] * source_y + forward[0][2]) / denominator
        y = (forward[1][0] * source_x + forward[1][1] * source_y + forward[1][2]) / denominator
        point = (
            max(0, min(width - 1, int(round(x)))),
            max(0, min(height - 1, int(round(y)))),
        )
        if point != prior:
            result.extend(point)
            prior = point
    if len(result) >= 4 and result[:2] == result[-2:]:
        del result[-2:]
    return tuple(result) if len(result) >= 6 else ()


def _covered(width: int, height: int, tiles: Sequence[SceneTile]) -> bool:
    """Prove rectangular tile crops cover the scene without a pixel bitmap."""

    boundaries = sorted({0, height, *(tile.y for tile in tiles), *(tile.y + tile.crop_h for tile in tiles)})
    if boundaries[0] != 0 or boundaries[-1] != height:
        return False
    for top, bottom in zip(boundaries, boundaries[1:]):
        if top == bottom:
            continue
        intervals = sorted(
            (tile.x, tile.x + tile.crop_w)
            for tile in tiles
            if tile.y <= top and tile.y + tile.crop_h >= bottom
        )
        if not intervals or intervals[0][0] != 0:
            return False
        right = 0
        for left, candidate_right in intervals:
            if left > right:
                return False
            right = max(right, candidate_right)
        if right != width:
            return False
    return True


def _png_bytes(image: Any) -> bytes:
    encoded = io.BytesIO()
    image.save(encoded, format="PNG", optimize=False, compress_level=6)
    payload = encoded.getvalue()
    _require(
        payload.startswith(b"\x89PNG\r\n\x1a\n")
        and len(payload) <= MAX_SCENE_PNG_BYTES,
        "full-image scene PNG is invalid or too large",
    )
    return payload


class Stage20SceneCatalog:
    """Verified full-card mosaics and geometry for one paused Stage-20 run."""

    def __init__(
        self,
        *,
        bindings: tuple[tuple[BoundFile, str], ...],
        tile_directory: Path,
        tile_directory_identity: tuple[int, ...],
        images: Sequence[SceneImage],
        proposals: Mapping[str, SceneProposal],
        order_by_image: Mapping[str, tuple[str, ...]],
        direct_full_image_png: bool = False,
    ) -> None:
        self._bindings = bindings
        self._tile_directory = tile_directory
        self._tile_directory_identity = tile_directory_identity
        self.images = tuple(images)
        self.images_by_id = {item.image_id: item for item in images}
        self.images_by_scene = {item.scene_id: item for item in images}
        self.proposals = dict(proposals)
        self.order_by_image = dict(order_by_image)
        self._direct_full_image_png = direct_full_image_png

    @classmethod
    def open(
        cls,
        run_root: Path,
        *,
        stage10_receipt_sha256: str,
        stage20_receipt_sha256: str,
        proposals_sha256: str,
        overlay_manifest_sha256: str,
        proposal_views: Mapping[str, object],
        proposal_order: Sequence[str],
        source_polygons: Mapping[str, Sequence[int]],
    ) -> "Stage20SceneCatalog":
        stage10 = run_root / "stages" / "10_prepare"
        stage20 = run_root / "stages" / "20_proposals_features"
        stage10_receipt = BoundFile.capture_streaming(
            stage10 / "_SUCCESS.json",
            "Stage-10 receipt",
            max_bytes=MAX_JSON_BYTES,
            expected_sha256=stage10_receipt_sha256,
        )
        stage20_receipt = BoundFile.capture_streaming(
            stage20 / "_SUCCESS.json",
            "Stage-20 receipt",
            max_bytes=MAX_JSON_BYTES,
            expected_sha256=stage20_receipt_sha256,
        )
        proposals_file = BoundFile.capture_streaming(
            stage20 / "proposals.csv",
            "Stage-20 proposal table",
            max_bytes=MAX_TABLE_BYTES,
            expected_sha256=proposals_sha256,
        )
        overlay_file = BoundFile.capture_streaming(
            stage20 / "overlay_manifest.json",
            "Stage-20 overlay manifest",
            max_bytes=MAX_JSON_BYTES,
            expected_sha256=overlay_manifest_sha256,
        )
        summary_file, summary_payload = BoundFile.capture(
            stage10 / "prepare_summary.json",
            "Stage-10 prepare summary",
            max_bytes=MAX_JSON_BYTES,
        )
        summary = canonical_json_value(summary_payload, "Stage-10 prepare summary")
        if (
            isinstance(summary, dict)
            and summary.get("schema") == _FULL_IMAGE_PREPARE_SCHEMA
        ):
            return cls._open_full_image(
                stage10=stage10,
                stage10_receipt=stage10_receipt,
                stage20_receipt=stage20_receipt,
                proposals_file=proposals_file,
                overlay_file=overlay_file,
                summary_file=summary_file,
                summary=summary,
                stage10_receipt_sha256=stage10_receipt_sha256,
                proposal_views=proposal_views,
                proposal_order=proposal_order,
                source_polygons=source_polygons,
            )
        summary_fields = {
            "schema",
            "status",
            "profile",
            "image_count",
            "group_count",
            "tile_count",
            "tile_size",
            "tile_stride",
            "far_edge_alignment",
            "padding",
            "tile_format",
            "tiles_index_sha256",
            "images",
        }
        _require(
            isinstance(summary, dict)
            and set(summary) == summary_fields
            and summary.get("schema") == "compag-curation-canonical-prepare/v1"
            and summary.get("status") == "PASS"
            and summary.get("tile_size") == 512
            and summary.get("tile_stride") == 512
            and summary.get("far_edge_alignment") is True
            and summary.get("padding") == "bottom-right-edge-value"
            and summary.get("tile_format") == "jpg",
            "Stage-10 prepare summary is not the canonical mosaic contract",
        )
        expected_tile_index_sha = str(summary.get("tiles_index_sha256"))
        tile_index_file, tile_index_payload = BoundFile.capture(
            stage10 / "tiles_index.csv",
            "Stage-10 tile index",
            max_bytes=MAX_TABLE_BYTES,
            expected_sha256=expected_tile_index_sha,
        )
        tile_rows = _parse_csv(tile_index_payload, _TILE_COLUMNS, "Stage-10 tile index")

        raw_images = summary.get("images")
        _require(isinstance(raw_images, list) and bool(raw_images), "Stage-10 image records are missing")
        image_records: dict[str, dict[str, object]] = {}
        forward_by_id: dict[str, tuple[tuple[float, ...], ...]] = {}
        image_record_fields = {
            "image_name",
            "image_id",
            "image_sha256",
            "group_id",
            "source_width",
            "source_height",
            "warped_width",
            "warped_height",
            "warp_mode",
            "inverse_warp",
            "row_lines",
            "column_lines",
            "tile_count",
            "preprocessing_evidence_sha256",
        }
        for raw in raw_images:
            _require(isinstance(raw, dict) and set(raw) == image_record_fields, "Stage-10 image record contract changed")
            image_id = str(raw["image_id"])
            image_sha = str(raw["image_sha256"])
            _require(
                _SHA256.fullmatch(image_id) is not None
                and _SHA256.fullmatch(image_sha) is not None
                and image_id not in image_records,
                "Stage-10 image identity is invalid or duplicated",
            )
            portable_basename(str(raw["image_name"]), "Stage-10 image name")
            width = _strict_int(raw["warped_width"], "Stage-10 warped width", minimum=1, maximum=MAX_SCENE_DIMENSION)
            height = _strict_int(raw["warped_height"], "Stage-10 warped height", minimum=1, maximum=MAX_SCENE_DIMENSION)
            _require(width * height <= MAX_SCENE_PIXELS, "Stage-10 scene exceeds its pixel bound")
            _strict_int(raw["source_width"], "Stage-10 source width", minimum=1, maximum=MAX_SCENE_DIMENSION)
            _strict_int(raw["source_height"], "Stage-10 source height", minimum=1, maximum=MAX_SCENE_DIMENSION)
            _strict_int(raw["tile_count"], "Stage-10 image tile count", minimum=1)
            forward_by_id[image_id] = _invert_3x3(raw["inverse_warp"])
            image_records[image_id] = raw
        _require(
            summary.get("image_count") == len(image_records)
            and summary.get("tile_count") == len(tile_rows)
            and summary.get("group_count") == len({str(item["group_id"]) for item in image_records.values()}),
            "Stage-10 scene counts changed",
        )

        tile_directory = stage10 / "tiles"
        directory_info = tile_directory.lstat()
        _require(
            stat.S_ISDIR(directory_info.st_mode) and not tile_directory.is_symlink(),
            "Stage-10 tile directory is unsafe",
        )
        tiles_by_image: dict[str, list[SceneTile]] = {image_id: [] for image_id in image_records}
        expected_names: set[str] = set()
        tile_bindings: list[tuple[BoundFile, str]] = []
        for row in tile_rows:
            tile_name = portable_basename(row["tile_name"], "Stage-10 tile name")
            image_id = row["image_id"]
            record = image_records.get(image_id)
            _require(record is not None and tile_name not in expected_names, "Stage-10 tile identity is invalid or duplicated")
            _require(
                row["image_sha256"] == record["image_sha256"]
                and row["image_name"] == record["image_name"]
                and row["group_id"] == record["group_id"],
                "Stage-10 tile/image identity changed",
            )
            x = _strict_int(row["x"], "Stage-10 tile x")
            y = _strict_int(row["y"], "Stage-10 tile y")
            crop_w = _strict_int(row["crop_w"], "Stage-10 tile crop width", minimum=1, maximum=512)
            crop_h = _strict_int(row["crop_h"], "Stage-10 tile crop height", minimum=1, maximum=512)
            width = int(record["warped_width"])
            height = int(record["warped_height"])
            _require(x + crop_w <= width and y + crop_h <= height, "Stage-10 tile exceeds its warped image")
            _require(
                _strict_int(row["orig_w"], "Stage-10 original width", minimum=1) == int(record["source_width"])
                and _strict_int(row["orig_h"], "Stage-10 original height", minimum=1) == int(record["source_height"]),
                "Stage-10 tile source dimensions changed",
            )
            bound = BoundFile.capture_streaming(
                tile_directory / tile_name,
                f"Stage-10 tile {tile_name}",
                max_bytes=MAX_TILE_BYTES,
                expected_sha256=row["tile_sha256"],
            )
            expected_names.add(tile_name)
            tile_bindings.append((bound, f"Stage-10 tile {tile_name}"))
            tiles_by_image[image_id].append(SceneTile(tile_name, bound, x, y, crop_w, crop_h))
        observed_names = {path.name for path in tile_directory.iterdir()}
        _require(observed_names == expected_names, "Stage-10 tile directory closure changed")

        images: list[SceneImage] = []
        for index, record in enumerate(
            sorted(image_records.values(), key=lambda item: (str(item["image_name"]), str(item["image_id"]))),
        ):
            image_id = str(record["image_id"])
            tiles = tuple(sorted(tiles_by_image[image_id], key=lambda item: (item.y, item.x, item.name)))
            _require(len(tiles) == int(record["tile_count"]), "Stage-10 per-image tile count changed")
            width, height = int(record["warped_width"]), int(record["warped_height"])
            _require(_covered(width, height, tiles), "Stage-10 tile crops do not cover the full scene")
            scene_id = compact_json_sha256(
                {
                    "domain": "compag-curation-stage20-full-image-scene/v1",
                    "stage10_receipt_sha256": stage10_receipt_sha256,
                    "image_id": image_id,
                }
            )
            images.append(
                SceneImage(
                    scene_id=scene_id,
                    index=index,
                    image_id=image_id,
                    image_sha256=str(record["image_sha256"]),
                    image_name=str(record["image_name"]),
                    group_id=str(record["group_id"]),
                    width=width,
                    height=height,
                    tiles=tiles,
                )
            )

        geometry: dict[str, SceneProposal] = {}
        order_by_image: dict[str, list[str]] = {item.image_id: [] for item in images}
        for proposal_id in proposal_order:
            view = proposal_views.get(proposal_id)
            _require(view is not None and proposal_id not in geometry, "scene proposal order is invalid")
            image_id = str(getattr(view, "image_id"))
            image = next((item for item in images if item.image_id == image_id), None)
            _require(
                image is not None
                and str(getattr(view, "image_name")) == image.image_name
                and str(getattr(view, "group_id")) == image.group_id,
                "scene proposal/image identity changed",
            )
            tile_x = _strict_int(getattr(view, "tile_x"), "scene proposal tile x")
            tile_y = _strict_int(getattr(view, "tile_y"), "scene proposal tile y")
            bbox_x = tile_x + _strict_int(getattr(view, "bbox_x"), "scene proposal bbox x")
            bbox_y = tile_y + _strict_int(getattr(view, "bbox_y"), "scene proposal bbox y")
            bbox_w = _strict_int(getattr(view, "bbox_w"), "scene proposal bbox width", minimum=1)
            bbox_h = _strict_int(getattr(view, "bbox_h"), "scene proposal bbox height", minimum=1)
            left = max(0, min(image.width - 1, bbox_x))
            top = max(0, min(image.height - 1, bbox_y))
            right = max(left + 1, min(image.width, bbox_x + bbox_w))
            bottom = max(top + 1, min(image.height, bbox_y + bbox_h))
            source_polygon = tuple(int(value) for value in source_polygons.get(proposal_id, ()))
            polygon = _transform_polygon(source_polygon, forward_by_id[image_id], image.width, image.height)
            geometry[proposal_id] = SceneProposal(
                proposal_id=proposal_id,
                image_id=image_id,
                label_prefix=str(getattr(view, "label_prefix")),
                bbox=(left, top, right - left, bottom - top),
                polygon=polygon,
            )
            order_by_image[image_id].append(proposal_id)
        _require(set(geometry) == set(proposal_views), "scene proposal geometry is incomplete")

        bindings = (
            (stage10_receipt, "Stage-10 receipt"),
            (stage20_receipt, "Stage-20 receipt"),
            (summary_file, "Stage-10 prepare summary"),
            (tile_index_file, "Stage-10 tile index"),
            (proposals_file, "Stage-20 proposal table"),
            (overlay_file, "Stage-20 overlay manifest"),
            *tile_bindings,
        )
        return cls(
            bindings=bindings,
            tile_directory=tile_directory,
            tile_directory_identity=_file_identity(directory_info),
            images=images,
            proposals=geometry,
            order_by_image={key: tuple(value) for key, value in order_by_image.items()},
        )

    @classmethod
    def _open_full_image(
        cls,
        *,
        stage10: Path,
        stage10_receipt: BoundFile,
        stage20_receipt: BoundFile,
        proposals_file: BoundFile,
        overlay_file: BoundFile,
        summary_file: BoundFile,
        summary: Mapping[str, object],
        stage10_receipt_sha256: str,
        proposal_views: Mapping[str, object],
        proposal_order: Sequence[str],
        source_polygons: Mapping[str, Sequence[int]],
    ) -> "Stage20SceneCatalog":
        """Open the profile-explicit one-PNG-per-image Stage-10 contract."""

        summary_fields = {
            "schema",
            "status",
            "profile",
            "spatial_mode",
            "image_count",
            "group_count",
            "processing_unit_count",
            "processing_unit_kind",
            "external_tiling",
            "unit_format",
            "unit_lossless",
            "tiles_index_sha256",
            "images",
        }
        _require(
            set(summary) == summary_fields
            and summary.get("status") == "PASS"
            and summary.get("profile") == _FULL_IMAGE_PROFILE
            and summary.get("spatial_mode") == _FULL_IMAGE_SPATIAL_MODE
            and summary.get("processing_unit_kind") == _FULL_IMAGE_UNIT_KIND
            and summary.get("external_tiling") is False
            and summary.get("unit_format") == "png"
            and summary.get("unit_lossless") is True,
            "Stage-10 prepare summary is not the full-image unit contract",
        )
        unit_index_file, unit_index_payload = BoundFile.capture(
            stage10 / "tiles_index.csv",
            "Stage-10 full-image unit index",
            max_bytes=MAX_TABLE_BYTES,
            expected_sha256=str(summary.get("tiles_index_sha256")),
        )
        unit_rows = _parse_csv(
            unit_index_payload,
            _FULL_IMAGE_UNIT_COLUMNS,
            "Stage-10 full-image unit index",
        )

        raw_images = summary.get("images")
        _require(
            isinstance(raw_images, list) and bool(raw_images),
            "Stage-10 full-image records are missing",
        )
        image_record_fields = {
            "image_name",
            "image_id",
            "image_sha256",
            "group_id",
            "source_width",
            "source_height",
            "warped_width",
            "warped_height",
            "warp_mode",
            "inverse_warp",
            "row_lines",
            "column_lines",
            "processing_unit_count",
            "preprocessing_evidence_sha256",
        }
        image_records: dict[str, Mapping[str, object]] = {}
        forward_by_id: dict[str, tuple[tuple[float, ...], ...]] = {}
        for raw in raw_images:
            _require(
                isinstance(raw, dict) and set(raw) == image_record_fields,
                "Stage-10 full-image record contract changed",
            )
            image_id = str(raw["image_id"])
            image_sha = str(raw["image_sha256"])
            evidence_sha = str(raw["preprocessing_evidence_sha256"])
            _require(
                _SHA256.fullmatch(image_id) is not None
                and _SHA256.fullmatch(image_sha) is not None
                and _SHA256.fullmatch(evidence_sha) is not None
                and image_id not in image_records,
                "Stage-10 full-image identity is invalid or duplicated",
            )
            portable_basename(str(raw["image_name"]), "Stage-10 image name")
            _require(
                isinstance(raw["group_id"], str)
                and bool(raw["group_id"])
                and isinstance(raw["warp_mode"], str)
                and bool(raw["warp_mode"]),
                "Stage-10 full-image metadata is invalid",
            )
            source_width = _strict_int(
                raw["source_width"],
                "Stage-10 source width",
                minimum=1,
                maximum=MAX_SCENE_DIMENSION,
            )
            source_height = _strict_int(
                raw["source_height"],
                "Stage-10 source height",
                minimum=1,
                maximum=MAX_SCENE_DIMENSION,
            )
            width = _strict_int(
                raw["warped_width"],
                "Stage-10 warped width",
                minimum=1,
                maximum=MAX_SCENE_DIMENSION,
            )
            height = _strict_int(
                raw["warped_height"],
                "Stage-10 warped height",
                minimum=1,
                maximum=MAX_SCENE_DIMENSION,
            )
            _require(
                source_width * source_height <= MAX_SCENE_PIXELS
                and width * height <= MAX_SCENE_PIXELS,
                "Stage-10 full-image scene exceeds its pixel bound",
            )
            _require(
                _strict_int(
                    raw["processing_unit_count"],
                    "Stage-10 per-image processing-unit count",
                    minimum=1,
                )
                == 1,
                "Stage-10 full-image record must bind exactly one unit",
            )
            for name, bound in (("row_lines", height), ("column_lines", width)):
                values = raw[name]
                _require(
                    isinstance(values, list)
                    and all(
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and 0 <= value < bound
                        for value in values
                    )
                    and values == sorted(set(values)),
                    f"Stage-10 full-image {name} are invalid",
                )
            forward_by_id[image_id] = _invert_3x3(raw["inverse_warp"])
            image_records[image_id] = raw

        _require(
            _strict_int(summary.get("image_count"), "Stage-10 image count", minimum=1)
            == len(image_records)
            and _strict_int(
                summary.get("processing_unit_count"),
                "Stage-10 processing-unit count",
                minimum=1,
            )
            == len(unit_rows)
            == len(image_records)
            and _strict_int(summary.get("group_count"), "Stage-10 group count", minimum=1)
            == len({str(item["group_id"]) for item in image_records.values()}),
            "Stage-10 full-image scene counts changed",
        )

        unit_directory = stage10 / "processing_units"
        try:
            directory_info = unit_directory.lstat()
        except OSError as exc:
            raise PublicIOError("Stage-10 full-image unit directory is unavailable") from exc
        _require(
            stat.S_ISDIR(directory_info.st_mode) and not unit_directory.is_symlink(),
            "Stage-10 full-image unit directory is unsafe",
        )
        unit_by_image: dict[str, SceneTile] = {}
        expected_names: set[str] = set()
        unit_bindings: list[tuple[BoundFile, str]] = []
        try:
            from PIL import Image
        except ImportError as exc:
            raise PublicIOError("full-image scene validation requires Pillow") from exc
        for row in unit_rows:
            unit_name = portable_basename(
                row["unit_name"],
                "Stage-10 full-image unit name",
            )
            image_id = row["image_id"]
            record = image_records.get(image_id)
            _require(
                record is not None
                and image_id not in unit_by_image
                and unit_name not in expected_names
                and Path(unit_name).suffix.lower() == ".png",
                "Stage-10 full-image unit identity is invalid or duplicated",
            )
            _require(
                row["image_sha256"] == record["image_sha256"]
                and row["image_name"] == record["image_name"]
                and row["group_id"] == record["group_id"]
                and row["processing_unit_kind"] == _FULL_IMAGE_UNIT_KIND
                and row["x"] == "0"
                and row["y"] == "0",
                "Stage-10 full-image unit/image identity changed",
            )
            width = _strict_int(
                row["width"],
                "Stage-10 full-image unit width",
                minimum=1,
                maximum=MAX_SCENE_DIMENSION,
            )
            height = _strict_int(
                row["height"],
                "Stage-10 full-image unit height",
                minimum=1,
                maximum=MAX_SCENE_DIMENSION,
            )
            _require(
                width == int(record["warped_width"])
                and height == int(record["warped_height"])
                and _strict_int(row["orig_w"], "Stage-10 original width", minimum=1)
                == int(record["source_width"])
                and _strict_int(row["orig_h"], "Stage-10 original height", minimum=1)
                == int(record["source_height"]),
                "Stage-10 full-image unit dimensions changed",
            )
            bound, png_payload = BoundFile.capture(
                unit_directory / unit_name,
                f"Stage-10 full-image unit {unit_name}",
                max_bytes=MAX_SCENE_PNG_BYTES,
                expected_sha256=row["unit_sha256"],
            )
            _require(
                png_payload.startswith(b"\x89PNG\r\n\x1a\n"),
                f"Stage-10 full-image unit is not PNG: {unit_name}",
            )
            try:
                with Image.open(io.BytesIO(png_payload)) as opened:
                    opened.load()
                    decoded_size = opened.size
                    decoded_format = opened.format
                    decoded_frames = int(getattr(opened, "n_frames", 1))
            except Exception as exc:
                raise PublicIOError(
                    f"Stage-10 full-image unit cannot be decoded: {unit_name}"
                ) from exc
            _require(
                decoded_format == "PNG"
                and decoded_frames == 1
                and decoded_size == (width, height),
                f"Stage-10 full-image unit dimensions changed: {unit_name}",
            )
            expected_names.add(unit_name)
            unit_bindings.append(
                (bound, f"Stage-10 full-image unit {unit_name}")
            )
            unit_by_image[image_id] = SceneTile(
                unit_name,
                bound,
                0,
                0,
                width,
                height,
            )
        observed_names = {path.name for path in unit_directory.iterdir()}
        _require(
            observed_names == expected_names,
            "Stage-10 full-image unit directory closure changed",
        )
        _require(
            set(unit_by_image) == set(image_records),
            "Stage-10 full-image unit/image closure changed",
        )

        images: list[SceneImage] = []
        for index, record in enumerate(
            sorted(
                image_records.values(),
                key=lambda item: (str(item["image_name"]), str(item["image_id"])),
            )
        ):
            image_id = str(record["image_id"])
            unit = unit_by_image[image_id]
            scene_id = compact_json_sha256(
                {
                    "domain": "compag-curation-stage20-direct-full-image-scene/v1",
                    "stage10_receipt_sha256": stage10_receipt_sha256,
                    "image_id": image_id,
                    "unit_sha256": unit.source.sha256,
                }
            )
            images.append(
                SceneImage(
                    scene_id=scene_id,
                    index=index,
                    image_id=image_id,
                    image_sha256=str(record["image_sha256"]),
                    image_name=str(record["image_name"]),
                    group_id=str(record["group_id"]),
                    width=int(record["warped_width"]),
                    height=int(record["warped_height"]),
                    tiles=(unit,),
                    source=unit.source,
                )
            )

        images_by_id = {item.image_id: item for item in images}
        geometry: dict[str, SceneProposal] = {}
        order_by_image: dict[str, list[str]] = {item.image_id: [] for item in images}
        for proposal_id in proposal_order:
            view = proposal_views.get(proposal_id)
            _require(
                view is not None and proposal_id not in geometry,
                "scene proposal order is invalid",
            )
            image_id = str(getattr(view, "image_id"))
            image = images_by_id.get(image_id)
            unit = unit_by_image.get(image_id)
            _require(
                image is not None
                and unit is not None
                and str(getattr(view, "image_name")) == image.image_name
                and str(getattr(view, "group_id")) == image.group_id
                and str(getattr(view, "tile_name")) == unit.name
                and _strict_int(getattr(view, "tile_x"), "scene proposal unit x") == 0
                and _strict_int(getattr(view, "tile_y"), "scene proposal unit y") == 0,
                "scene proposal/full-image unit identity changed",
            )
            bbox_x = _strict_int(
                getattr(view, "bbox_x"),
                "scene proposal bbox x",
                maximum=image.width - 1,
            )
            bbox_y = _strict_int(
                getattr(view, "bbox_y"),
                "scene proposal bbox y",
                maximum=image.height - 1,
            )
            bbox_w = _strict_int(
                getattr(view, "bbox_w"),
                "scene proposal bbox width",
                minimum=1,
                maximum=image.width,
            )
            bbox_h = _strict_int(
                getattr(view, "bbox_h"),
                "scene proposal bbox height",
                minimum=1,
                maximum=image.height,
            )
            _require(
                bbox_x + bbox_w <= image.width
                and bbox_y + bbox_h <= image.height,
                "scene proposal bbox exceeds its full-image canvas",
            )
            source_polygon = tuple(
                int(value) for value in source_polygons.get(proposal_id, ())
            )
            polygon = _transform_polygon(
                source_polygon,
                forward_by_id[image_id],
                image.width,
                image.height,
            )
            geometry[proposal_id] = SceneProposal(
                proposal_id=proposal_id,
                image_id=image_id,
                label_prefix=str(getattr(view, "label_prefix")),
                bbox=(bbox_x, bbox_y, bbox_w, bbox_h),
                polygon=polygon,
            )
            order_by_image[image_id].append(proposal_id)
        _require(
            set(geometry) == set(proposal_views),
            "scene proposal geometry is incomplete",
        )

        bindings = (
            (stage10_receipt, "Stage-10 receipt"),
            (stage20_receipt, "Stage-20 receipt"),
            (summary_file, "Stage-10 prepare summary"),
            (unit_index_file, "Stage-10 full-image unit index"),
            (proposals_file, "Stage-20 proposal table"),
            (overlay_file, "Stage-20 overlay manifest"),
            *unit_bindings,
        )
        return cls(
            bindings=bindings,
            tile_directory=unit_directory,
            tile_directory_identity=_file_identity(directory_info),
            images=images,
            proposals=geometry,
            order_by_image={key: tuple(value) for key, value in order_by_image.items()},
            direct_full_image_png=True,
        )

    def _revalidate(self) -> None:
        for binding, role in self._bindings:
            binding.check_metadata(role)
        directory_info = self._tile_directory.lstat()
        _require(
            stat.S_ISDIR(directory_info.st_mode)
            and not self._tile_directory.is_symlink()
            and _file_identity(directory_info) == self._tile_directory_identity,
            (
                "Stage-10 full-image unit directory changed after scene indexing"
                if self._direct_full_image_png
                else "Stage-10 tile directory changed after scene indexing"
            ),
        )

    def payload(
        self,
        proposal_id: str,
        choices: Mapping[str, str | None],
        models: Mapping[str, Mapping[str, object]] | None = None,
    ) -> dict[str, object]:
        self._revalidate()
        selected = self.proposals.get(proposal_id)
        _require(selected is not None, "unknown full-image scene proposal")
        image = self.images_by_id[selected.image_id]
        proposals = []
        for item_id in self.order_by_image[image.image_id]:
            item = self.proposals[item_id]
            row: dict[str, object] = {
                    "proposal_id": item.proposal_id,
                    "label_prefix": item.label_prefix,
                    "polygon": list(item.polygon),
                    "bbox": list(item.bbox),
                    "choice": choices.get(item.proposal_id),
                    "reviewable": True,
                    "selected": item.proposal_id == proposal_id,
                }
            if models is not None:
                model = models.get(item.proposal_id)
                _require(
                    isinstance(model, Mapping),
                    "published-model scene context is incomplete",
                )
                row["model"] = dict(model)
            proposals.append(row)
        payload: dict[str, object] = {
            "schema": SCENE_SCHEMA,
            "scene_id": image.scene_id,
            "index": image.index,
            "kind": (
                FULL_IMAGE_STAGE20_SCENE_KIND
                if self._direct_full_image_png
                else STAGE20_SCENE_KIND
            ),
            "image_id": image.image_id,
            "image_name": image.image_name,
            "group_id": image.group_id,
            "width": image.width,
            "height": image.height,
            "mime_type": "image/png",
            "selected_proposal_id": proposal_id,
            "proposal_count": len(proposals),
            "proposals": proposals,
            "overlay_defaults": {"mask_outline": True, "bbox": False},
            "outline_source": "SEALED_MASK_DERIVED_POLYGON",
            "stitch_policy": (
                "DIRECT_VERIFIED_SINGLE_LOSSLESS_PNG_NO_STITCH"
                if self._direct_full_image_png
                else "SORTED_Y_X_NAME_LAST_TILE_WINS_OVERLAP"
            ),
        }
        if models is not None:
            _require(
                set(models) == set(self.proposals),
                "published-model scene context has the wrong proposal closure",
            )
            payload["model"] = {
                "mode": "PUBLISHED_TRANSFER_ASSIST",
                "model_suggestions_are_human_context": True,
            }
        return payload

    def bytes(self, scene_id: str) -> bytes:
        self._revalidate()
        image = self.images_by_scene.get(scene_id)
        _require(image is not None, "unknown full-image scene")
        try:
            from PIL import Image
        except ImportError as exc:
            raise PublicIOError("full-image scene rendering requires Pillow") from exc
        if self._direct_full_image_png:
            _require(
                image.source is not None and len(image.tiles) == 1,
                "full-image scene does not bind exactly one source unit",
            )
            payload = image.source.read_verified(
                f"Stage-10 full-image unit {image.tiles[0].name}",
                max_bytes=MAX_SCENE_PNG_BYTES,
            )
            _require(
                payload.startswith(b"\x89PNG\r\n\x1a\n"),
                "Stage-10 full-image scene unit is not PNG",
            )
            try:
                with Image.open(io.BytesIO(payload)) as opened:
                    opened.load()
                    decoded_size = opened.size
                    decoded_format = opened.format
                    decoded_frames = int(getattr(opened, "n_frames", 1))
            except Exception as exc:
                raise PublicIOError(
                    "Stage-10 full-image scene unit cannot be decoded"
                ) from exc
            _require(
                decoded_format == "PNG"
                and decoded_frames == 1
                and decoded_size == (image.width, image.height),
                "Stage-10 full-image scene unit dimensions changed",
            )
            return payload
        canvas = Image.new("RGB", (image.width, image.height), (32, 32, 32))
        for tile in image.tiles:
            payload = tile.source.read_verified(f"Stage-10 tile {tile.name}", max_bytes=MAX_TILE_BYTES)
            _require(payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9"), "Stage-10 scene tile is not JPEG")
            try:
                with Image.open(io.BytesIO(payload)) as opened:
                    opened.load()
                    decoded = opened.convert("RGB")
            except Exception as exc:
                raise PublicIOError(f"Stage-10 tile cannot be decoded: {tile.name}") from exc
            _require(decoded.size == (512, 512), f"Stage-10 tile dimensions changed: {tile.name}")
            canvas.paste(decoded.crop((0, 0, tile.crop_w, tile.crop_h)), (tile.x, tile.y))
        return _png_bytes(canvas)


def round_scene_proposals(
    predictions: Mapping[str, Mapping[str, str]],
) -> Mapping[str, SceneProposal]:
    """Project already-validated Stage-60 predictions into scene records."""

    result: dict[str, SceneProposal] = {}
    for proposal_id, row in predictions.items():
        _require(_SHA256.fullmatch(proposal_id) is not None and row.get("proposal_id") == proposal_id, "Stage-60 scene proposal identity is invalid")
        image_id = str(row.get("image_id"))
        _require(_SHA256.fullmatch(image_id) is not None, "Stage-60 scene image identity is invalid")
        image_name = portable_basename(str(row.get("full_image")), "Stage-60 full-image name")
        width = _strict_int(row.get("orig_w"), "Stage-60 scene width", minimum=1, maximum=MAX_SCENE_DIMENSION)
        height = _strict_int(row.get("orig_h"), "Stage-60 scene height", minimum=1, maximum=MAX_SCENE_DIMENSION)
        _require(width * height <= MAX_SCENE_PIXELS, "Stage-60 scene exceeds its pixel bound")
        x = _strict_int(row.get("x"), "Stage-60 scene bbox x")
        y = _strict_int(row.get("y"), "Stage-60 scene bbox y")
        bbox_w = _strict_int(row.get("w"), "Stage-60 scene bbox width", minimum=1)
        bbox_h = _strict_int(row.get("h"), "Stage-60 scene bbox height", minimum=1)
        _require(x + bbox_w <= width and y + bbox_h <= height, "Stage-60 scene bbox exceeds its image")
        try:
            raw_polygon = json.loads(str(row.get("poly")))
        except (json.JSONDecodeError, RecursionError) as exc:
            raise PublicIOError("Stage-60 scene polygon is invalid JSON") from exc
        _require(
            isinstance(raw_polygon, list)
            and len(raw_polygon) % 2 == 0
            and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in raw_polygon)
            and all(value <= width for value in raw_polygon[0::2])
            and all(value <= height for value in raw_polygon[1::2]),
            "Stage-60 scene polygon is invalid",
        )
        xgb_p = _strict_float(row.get("xgb_p"), "Stage-60 scene probability")
        prediction = _strict_int(row.get("prediction"), "Stage-60 scene prediction", maximum=1)
        kept = _strict_int(row.get("kept"), "Stage-60 scene kept flag", maximum=1)
        result[proposal_id] = SceneProposal(
            proposal_id=proposal_id,
            image_id=image_id,
            label_prefix=proposal_id[:12],
            bbox=(x, y, bbox_w, bbox_h),
            polygon=tuple(raw_polygon),
            model={
                "xgb_p": xgb_p,
                "prediction": prediction,
                "kept": kept,
                "image_name": image_name,
                "image_sha256": str(row.get("image_sha256")),
                "group_id": str(row.get("group_id")),
                "width": width,
                "height": height,
            },
        )
    return result


class RoundSceneCatalog:
    """Verified original-image scenes containing every Stage-60 proposal."""

    def __init__(
        self,
        *,
        bindings: tuple[tuple[BoundFile, str], ...],
        images: Sequence[SceneImage],
        proposals: Mapping[str, SceneProposal],
        order_by_image: Mapping[str, tuple[str, ...]],
        reviewable: frozenset[str],
        profile: str,
        bundle_sha256: str,
    ) -> None:
        self._bindings = bindings
        self.images = tuple(images)
        self.images_by_id = {item.image_id: item for item in images}
        self.images_by_scene = {item.scene_id: item for item in images}
        self.proposals = dict(proposals)
        self.order_by_image = dict(order_by_image)
        self.reviewable = reviewable
        self.profile = profile
        self.bundle_sha256 = bundle_sha256

    @classmethod
    def open(
        cls,
        stage60_root: Path,
        *,
        predictions_sha256: str,
        inference_result_sha256: str,
        images: Sequence[object],
        proposals: Mapping[str, SceneProposal],
        reviewable_order: Sequence[str],
        profile: str,
        bundle_sha256: str,
    ) -> "RoundSceneCatalog":
        predictions_file = BoundFile.capture_streaming(
            stage60_root / "predictions.csv",
            "Stage-60 predictions",
            max_bytes=MAX_TABLE_BYTES,
            expected_sha256=predictions_sha256,
        )
        inference_file = BoundFile.capture_streaming(
            stage60_root / "inference_result.json",
            "Stage-60 inference result",
            max_bytes=MAX_JSON_BYTES,
            expected_sha256=inference_result_sha256,
        )
        source_bindings: list[tuple[BoundFile, str]] = []
        scene_images: list[SceneImage] = []
        image_values = sorted(images, key=lambda item: (str(getattr(item, "name")), str(getattr(item, "image_id"))))
        for index, source in enumerate(image_values):
            image_id = str(getattr(source, "image_id"))
            image_sha = str(getattr(source, "image_sha256"))
            _require(_SHA256.fullmatch(image_id) is not None and _SHA256.fullmatch(image_sha) is not None, "round scene image identity is invalid")
            name = portable_basename(str(getattr(source, "name")), "round scene image name")
            width = _strict_int(getattr(source, "width"), "round scene width", minimum=1, maximum=MAX_SCENE_DIMENSION)
            height = _strict_int(getattr(source, "height"), "round scene height", minimum=1, maximum=MAX_SCENE_DIMENSION)
            _require(width * height <= MAX_SCENE_PIXELS, "round scene exceeds its pixel bound")
            bound = BoundFile.capture_streaming(
                Path(getattr(source, "path")),
                f"round scene source image {name}",
                max_bytes=MAX_IMAGE_FILE_BYTES,
                expected_sha256=image_sha,
            )
            _require(bound.size_bytes == int(getattr(source, "size_bytes")), "round scene source image size changed")
            source_bindings.append((bound, f"round scene source image {name}"))
            scene_id = compact_json_sha256(
                {
                    "domain": "compag-curation-round-full-image-scene/v1",
                    "predictions_sha256": predictions_sha256,
                    "image_id": image_id,
                    "image_sha256": image_sha,
                }
            )
            scene_images.append(
                SceneImage(
                    scene_id=scene_id,
                    index=index,
                    image_id=image_id,
                    image_sha256=image_sha,
                    image_name=name,
                    group_id=str(getattr(source, "group_id")),
                    width=width,
                    height=height,
                    source=bound,
                )
            )
        images_by_id = {item.image_id: item for item in scene_images}
        order_by_image: dict[str, list[str]] = {item.image_id: [] for item in scene_images}
        for proposal_id, proposal in sorted(proposals.items(), key=lambda item: (item[1].image_id, item[0])):
            image = images_by_id.get(proposal.image_id)
            model = proposal.model or {}
            _require(
                image is not None
                and model.get("image_name") == image.image_name
                and model.get("image_sha256") == image.image_sha256
                and model.get("group_id") == image.group_id
                and model.get("width") == image.width
                and model.get("height") == image.height,
                "Stage-60 scene proposal differs from original image inventory",
            )
            order_by_image[proposal.image_id].append(proposal_id)
        _require(set(proposals) == {item for values in order_by_image.values() for item in values}, "round scene proposal closure changed")
        reviewable = frozenset(reviewable_order)
        _require(reviewable <= set(proposals), "round scene review shortlist is outside Stage-60 proposals")
        return cls(
            bindings=((predictions_file, "Stage-60 predictions"), (inference_file, "Stage-60 inference result"), *source_bindings),
            images=scene_images,
            proposals=proposals,
            order_by_image={key: tuple(value) for key, value in order_by_image.items()},
            reviewable=reviewable,
            profile=profile,
            bundle_sha256=bundle_sha256,
        )

    def _revalidate(self) -> None:
        for binding, role in self._bindings:
            binding.check_metadata(role)

    def payload(
        self,
        proposal_id: str,
        choices: Mapping[str, str | None],
    ) -> dict[str, object]:
        self._revalidate()
        selected = self.proposals.get(proposal_id)
        _require(selected is not None and proposal_id in self.reviewable, "unknown round-review scene proposal")
        image = self.images_by_id[selected.image_id]
        proposals = []
        for item_id in self.order_by_image[image.image_id]:
            item = self.proposals[item_id]
            model = dict(item.model or {})
            for internal in ("image_name", "image_sha256", "group_id", "width", "height"):
                model.pop(internal, None)
            proposals.append(
                {
                    "proposal_id": item.proposal_id,
                    "label_prefix": item.label_prefix,
                    "polygon": list(item.polygon),
                    "bbox": list(item.bbox),
                    "choice": choices.get(item.proposal_id),
                    "reviewable": item.proposal_id in self.reviewable,
                    "selected": item.proposal_id == proposal_id,
                    "model": model,
                }
            )
        return {
            "schema": SCENE_SCHEMA,
            "scene_id": image.scene_id,
            "index": image.index,
            "kind": ROUND_SCENE_KIND,
            "image_id": image.image_id,
            "image_name": image.image_name,
            "group_id": image.group_id,
            "width": image.width,
            "height": image.height,
            "mime_type": "image/png",
            "selected_proposal_id": proposal_id,
            "proposal_count": len(proposals),
            "reviewable_proposal_count": sum(bool(item["reviewable"]) for item in proposals),
            "proposals": proposals,
            "overlay_defaults": {"mask_outline": True, "bbox": False},
            "outline_source": "SEALED_STAGE60_MASK_DERIVED_POLYGON",
            "model": {"profile": self.profile, "bundle_sha256": self.bundle_sha256},
        }

    def bytes(self, scene_id: str) -> bytes:
        self._revalidate()
        image = self.images_by_scene.get(scene_id)
        _require(image is not None and image.source is not None, "unknown round-review full-image scene")
        payload = image.source.read_verified(
            f"round scene source image {image.image_name}",
            max_bytes=MAX_IMAGE_FILE_BYTES,
        )
        try:
            from PIL import Image
        except ImportError as exc:
            raise PublicIOError("round full-image scene rendering requires Pillow") from exc
        try:
            with Image.open(io.BytesIO(payload)) as opened:
                opened.load()
                rendered = opened.convert("RGB")
        except Exception as exc:
            raise PublicIOError("round full-image scene could not be decoded") from exc
        _require(rendered.size == (image.width, image.height), "round full-image scene dimensions changed")
        return _png_bytes(rendered)


__all__ = [
    "FULL_IMAGE_STAGE20_SCENE_KIND",
    "ROUND_SCENE_KIND",
    "SCENE_SCHEMA",
    "STAGE20_SCENE_KIND",
    "RoundSceneCatalog",
    "SceneImage",
    "SceneProposal",
    "Stage20SceneCatalog",
    "round_scene_proposals",
]
