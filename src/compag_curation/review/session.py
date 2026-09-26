"""Durable human-review sessions for sealed public COMPAG runs.

The reviewer never writes inside a scientific run.  Every human action is an
immutable, hash-chained event in a private external directory; the reviewed
CSV is published once, only after the existing import and split contracts
accept it.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import os
import re
import stat
import threading
import uuid
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from compag_curation.public_io import (
    PublicIOError,
    bounded_csv_field_limit,
    canonical_json_bytes,
    canonical_json_value,
    compact_json_sha256,
    fsync_directory,
    portable_basename,
    publish_file_noreplace,
    publish_directory_noreplace,
    rename_noreplace,
    safe_relative,
    sha256_file,
    stable_file,
    write_new_bytes,
    write_new_json,
)
from compag_curation.review.exchange import (
    CANONICAL_REVIEW_COLUMNS,
    REVIEW_ACTION_WEIGHTS,
    REVIEW_COLUMNS,
    read_review_export,
    validate_review_table,
)
from compag_curation.run_state import ReviewRunSnapshot, open_review_snapshot


SESSION_SCHEMA = "compag-curation-human-review-session/v1"
MODEL_ASSISTED_SESSION_SCHEMA = (
    "compag-curation-published-model-review-session/v1"
)
EVENT_SCHEMA = "compag-curation-human-review-event/v1"
EVENT_HEAD_SCHEMA = "compag-curation-human-review-event-head/v1"
FINAL_SCHEMA = "compag-curation-human-review-final/v1"
MODEL_ASSISTED_FINAL_SCHEMA = (
    "compag-curation-published-model-review-final/v1"
)
HUMAN_DECISION_ORIGIN = "HUMAN_EXPLICIT"
MODEL_DECISION_ORIGIN = "PUBLISHED_MODEL_BULK_CONFIRMED"
OVERLAY_SCHEMA = "compag-curation-canonical-overlay-manifest/v1"
MAX_OVERLAY_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_OVERLAY_BYTES = 32 * 1024 * 1024
MAX_PROPOSALS_BYTES = 2 * 1024 * 1024 * 1024
MAX_EVENT_BYTES = 32 * 1024
MAX_JSON_BODY_BYTES = 16 * 1024
# A reviewer may revisit a proposal many times, but an unbounded event
# directory is neither a useful human workflow nor safe to replay.  Sixteen
# mutations per proposal gives a normal 11k-proposal review more than 180k
# durable actions; tiny fixtures/sessions retain a practical floor.  The hard
# cap keeps both directory enumeration and replay finite for every project.
EVENTS_PER_PROPOSAL = 16
MIN_EVENT_COUNT = 64
MAX_EVENT_COUNT = 250_000
_SHA256 = re.compile(r"[0-9a-f]{64}")
_EVENT_NAME = re.compile(r"([0-9]{8})\.json")
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{7,127}")
_FULL_IMAGE_PREPARE_SCHEMA = "compag-curation-full-image-prepare/v1"
_FULL_IMAGE_PROFILE = "full-image-multiscale-xgb-recall-gpu-v1"
_FULL_IMAGE_SPATIAL_MODE = "full-image-multiscale"
_FULL_IMAGE_UNIT_KIND = "full_warped_frame"
_FULL_IMAGE_MASK_ENCODING = "base64-packbits-little-column-major-bbox-v1"
_MAX_FULL_IMAGE_DIMENSION = 32_768
_MAX_FULL_IMAGE_PIXELS = 256_000_000
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


@dataclass(frozen=True)
class Decision:
    choice: str
    label: int
    review_action: str | None
    review_weight: float | None


DECISIONS: dict[str, Decision] = {
    "target": Decision("target", 1, "accept", 1.0),
    "other": Decision("other", 0, "flip", 1.0),
    "target_uncertain": Decision("target_uncertain", 1, "sus_accept", 0.4),
    "other_uncertain": Decision("other_uncertain", 0, "sus_flip", 0.4),
    # The binary label is a required placeholder.  Its zero weight excludes it
    # from all effective training/split calculations.
    "skip": Decision("skip", 0, "skip", 0.0),
}


@dataclass(frozen=True)
class ProposalView:
    proposal_id: str
    image_id: str
    image_name: str
    group_id: str
    tile_name: str
    tile_index: int
    proposal_index: int
    label_prefix: str
    tile_x: int
    tile_y: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    predicted_iou: float
    stability_score: float


@dataclass(frozen=True)
class TileView:
    index: int
    tile_name: str
    overlay_path: Path
    overlay_sha256: str
    proposal_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReviewIndex:
    request_rows: tuple[dict[str, str], ...]
    request_columns: tuple[str, ...]
    order: tuple[str, ...]
    proposals: Mapping[str, ProposalView]
    tiles: tuple[TileView, ...]
    groups: tuple[str, ...]
    overlay_manifest_sha256: str
    proposals_sha256: str
    source_polygons: Mapping[str, tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class _FullImageUnitBounds:
    image_id: str
    image_sha256: str
    image_name: str
    group_id: str
    unit_sha256: str
    width: int
    height: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_utc_timestamp(value: object, role: str) -> None:
    _require(isinstance(value, str), f"{role} is not a UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise PublicIOError(f"{role} is not a canonical UTC timestamp") from exc
    _require(parsed.strftime("%Y-%m-%dT%H:%M:%SZ") == value, f"{role} is not canonical")


def _strict_int(value: object, role: str, *, minimum: int = 0, maximum: int = 1_000_000_000) -> int:
    text = str(value)
    _require(re.fullmatch(r"0|[1-9][0-9]*", text) is not None, f"{role} is not a canonical integer")
    result = int(text)
    _require(minimum <= result <= maximum, f"{role} is outside its supported range")
    return result


def _strict_float(value: object, role: str) -> float:
    try:
        result = float(str(value))
    except ValueError as exc:
        raise PublicIOError(f"{role} is not numeric") from exc
    _require(result == result and result not in {float("inf"), float("-inf")}, f"{role} is not finite")
    return result


def _external_path(
    path: Path,
    run_root: Path,
    role: str,
    *,
    derived_basename: str,
) -> Path:
    absolute = path.absolute()
    parent = absolute.parent.resolve(strict=True)
    canonical = parent / absolute.name
    _require(absolute == canonical and absolute.name not in {"", ".", ".."}, f"{role} has a symlinked or unsafe component")
    portable_basename(absolute.name, role)
    _require(
        len(os.fsencode(absolute.name)) <= 255
        and len(os.fsencode(derived_basename)) <= 255,
        f"{role} name is too long for atomic publication",
    )
    _require(run_root != absolute and run_root not in absolute.parents, f"{role} must be outside the immutable run")
    return absolute


def _directory_is_private(path: Path, role: str) -> os.stat_result:
    info = path.lstat()
    _require(
        stat.S_ISDIR(info.st_mode)
        and not path.is_symlink()
        and info.st_uid == os.getuid()
        and info.st_gid == os.getgid()
        and stat.S_IMODE(info.st_mode) == 0o700,
        f"{role} must be an owned mode-0700 directory",
    )
    return info


def _regular_private(path: Path, role: str, *, max_bytes: int) -> tuple[bytes, os.stat_result]:
    info = path.lstat()
    _require(
        stat.S_ISREG(info.st_mode)
        and not path.is_symlink()
        and info.st_nlink == 1
        and info.st_uid == os.getuid()
        and info.st_gid == os.getgid()
        and stat.S_IMODE(info.st_mode) == 0o600,
        f"{role} must be an owned single-link mode-0600 file",
    )
    return stable_file(path, max_bytes=max_bytes)


def _manifest(snapshot: ReviewRunSnapshot) -> tuple[dict[str, object], str]:
    path = snapshot.root / "stages" / "20_proposals_features" / "overlay_manifest.json"
    payload, _info = stable_file(path, max_bytes=MAX_OVERLAY_MANIFEST_BYTES)
    value = canonical_json_value(payload, "review overlay manifest")
    _require(isinstance(value, dict), "review overlay manifest must be an object")
    _require(
        set(value)
        == {
            "schema",
            "status",
            "label_format",
            "tile_count",
            "proposal_count",
            "proposal_ids_sha256",
            "tiles",
            "tiles_sha256",
        }
        and value.get("schema") == OVERLAY_SCHEMA
        and value.get("status") == "PASS"
        and value.get("label_format") == "unique-proposal-id-prefix",
        "review overlay manifest contract mismatch",
    )
    return value, hashlib.sha256(payload).hexdigest()


def _overlay_index(snapshot: ReviewRunSnapshot) -> tuple[
    tuple[TileView, ...], dict[str, dict[str, object]], str
]:
    value, digest = _manifest(snapshot)
    raw_tiles = value.get("tiles")
    _require(isinstance(raw_tiles, list), "review overlay tiles must be a list")
    _require(value.get("tile_count") == len(raw_tiles), "review overlay tile count mismatch")
    _require(value.get("tiles_sha256") == compact_json_sha256(raw_tiles), "review overlay tile hash mismatch")
    stage20 = snapshot.root / "stages" / "20_proposals_features"
    tiles: list[TileView] = []
    proposal_map: dict[str, dict[str, object]] = {}
    ordered_ids: list[str] = []
    for tile_index, raw in enumerate(raw_tiles):
        _require(isinstance(raw, dict), "review overlay tile entry must be an object")
        _require(
            set(raw)
            == {
                "tile_name",
                "tile_sha256",
                "overlay",
                "overlay_sha256",
                "proposal_count",
                "proposal_id_prefix_length",
                "proposal_labels",
                "proposal_labels_sha256",
            },
            "review overlay tile entry contract mismatch",
        )
        tile_name = portable_basename(str(raw["tile_name"]), "review tile name")
        overlay_relative = safe_relative(str(raw["overlay"]), "review overlay path")
        _require(len(overlay_relative.parts) == 2 and overlay_relative.parts[0] == "overlays", "review overlay path is outside its sealed directory")
        overlay_path = stage20.joinpath(*overlay_relative.parts)
        overlay_sha256 = str(raw["overlay_sha256"])
        _require(_SHA256.fullmatch(overlay_sha256) is not None, "review overlay SHA-256 is invalid")
        labels = raw["proposal_labels"]
        _require(isinstance(labels, list), "review proposal labels must be a list")
        _require(raw["proposal_count"] == len(labels), "review overlay proposal count mismatch")
        _require(raw["proposal_labels_sha256"] == compact_json_sha256(labels), "review proposal-label hash mismatch")
        prefix_length = _strict_int(raw["proposal_id_prefix_length"], "proposal prefix length", minimum=8, maximum=64)
        tile_ids: list[str] = []
        for label in labels:
            _require(
                isinstance(label, dict)
                and set(label) == {"label", "proposal_id", "proposal_index", "mask_sha256"},
                "review proposal label contract mismatch",
            )
            proposal_id = str(label["proposal_id"])
            mask_sha256 = str(label["mask_sha256"])
            prefix = str(label["label"])
            _require(
                _SHA256.fullmatch(proposal_id) is not None
                and _SHA256.fullmatch(mask_sha256) is not None
                and prefix == proposal_id[:prefix_length],
                "review proposal label identity mismatch",
            )
            _require(proposal_id not in proposal_map, "review overlay contains a duplicate proposal ID")
            proposal_map[proposal_id] = {
                "tile_index": tile_index,
                "tile_name": tile_name,
                "proposal_index": _strict_int(label["proposal_index"], "proposal index", minimum=1),
                "label_prefix": prefix,
                "mask_sha256": mask_sha256,
            }
            tile_ids.append(proposal_id)
            ordered_ids.append(proposal_id)
        tiles.append(TileView(tile_index, tile_name, overlay_path, overlay_sha256, tuple(tile_ids)))
    _require(value.get("proposal_count") == len(ordered_ids), "review overlay global proposal count mismatch")
    _require(value.get("proposal_ids_sha256") == compact_json_sha256(ordered_ids), "review overlay proposal-ID hash mismatch")
    return tuple(tiles), proposal_map, digest


def _full_image_unit_bounds(
    snapshot: ReviewRunSnapshot,
) -> Mapping[str, _FullImageUnitBounds] | None:
    """Return the sealed full-image unit canvas map, or ``None`` for tiled v1."""

    stage10 = snapshot.root / "stages" / "10_prepare"
    summary_payload, _summary_info = stable_file(
        stage10 / "prepare_summary.json",
        max_bytes=MAX_OVERLAY_MANIFEST_BYTES,
    )
    summary = canonical_json_value(summary_payload, "Stage-10 prepare summary")
    if not isinstance(summary, dict):
        return None
    if summary.get("schema") != _FULL_IMAGE_PREPARE_SCHEMA:
        _require(
            summary.get("profile") != _FULL_IMAGE_PROFILE
            and summary.get("spatial_mode") != _FULL_IMAGE_SPATIAL_MODE,
            "full-image review requires the full-image Stage-10 schema",
        )
        return None

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
        "full-image review Stage-10 summary contract changed",
    )
    raw_images = summary.get("images")
    image_fields = {
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
    _require(
        isinstance(raw_images, list) and bool(raw_images),
        "full-image review Stage-10 image records are missing",
    )
    image_records: dict[str, Mapping[str, object]] = {}
    for raw in raw_images:
        _require(
            isinstance(raw, dict) and set(raw) == image_fields,
            "full-image review Stage-10 image record contract changed",
        )
        image_id = str(raw["image_id"])
        _require(
            _SHA256.fullmatch(image_id) is not None
            and _SHA256.fullmatch(str(raw["image_sha256"])) is not None
            and _SHA256.fullmatch(str(raw["preprocessing_evidence_sha256"]))
            is not None
            and image_id not in image_records,
            "full-image review Stage-10 image identity is invalid or duplicated",
        )
        portable_basename(str(raw["image_name"]), "full-image review image name")
        source_width = _strict_int(
            raw["source_width"],
            "full-image review source width",
            minimum=1,
            maximum=_MAX_FULL_IMAGE_DIMENSION,
        )
        source_height = _strict_int(
            raw["source_height"],
            "full-image review source height",
            minimum=1,
            maximum=_MAX_FULL_IMAGE_DIMENSION,
        )
        width = _strict_int(
            raw["warped_width"],
            "full-image review canvas width",
            minimum=1,
            maximum=_MAX_FULL_IMAGE_DIMENSION,
        )
        height = _strict_int(
            raw["warped_height"],
            "full-image review canvas height",
            minimum=1,
            maximum=_MAX_FULL_IMAGE_DIMENSION,
        )
        _require(
            source_width * source_height <= _MAX_FULL_IMAGE_PIXELS
            and width * height <= _MAX_FULL_IMAGE_PIXELS
            and _strict_int(
                raw["processing_unit_count"],
                "full-image review per-image unit count",
                minimum=1,
            )
            == 1,
            "full-image review Stage-10 image bounds are invalid",
        )
        image_records[image_id] = raw

    index_payload, _index_info = stable_file(
        stage10 / "tiles_index.csv",
        max_bytes=MAX_PROPOSALS_BYTES,
    )
    _require(
        hashlib.sha256(index_payload).hexdigest()
        == str(summary.get("tiles_index_sha256")),
        "full-image review Stage-10 unit index hash changed",
    )
    try:
        index_text = index_payload.decode("utf-8")
    except UnicodeError as exc:
        raise PublicIOError("full-image review Stage-10 unit index is not UTF-8") from exc
    _require(
        not index_text.startswith("\ufeff"),
        "full-image review Stage-10 unit index contains a byte-order mark",
    )
    try:
        with io.StringIO(index_text, newline="") as handle:
            reader = csv.DictReader(handle)
            _require(
                tuple(reader.fieldnames or ()) == _FULL_IMAGE_UNIT_COLUMNS,
                "full-image review Stage-10 unit index header mismatch",
            )
            unit_rows = list(reader)
    except csv.Error as exc:
        raise PublicIOError(
            "full-image review Stage-10 unit index contains invalid CSV framing"
        ) from exc
    _require(bool(unit_rows), "full-image review Stage-10 unit index is empty")

    bounds: dict[str, _FullImageUnitBounds] = {}
    observed_images: set[str] = set()
    for row in unit_rows:
        _require(
            None not in row
            and tuple(row) == _FULL_IMAGE_UNIT_COLUMNS
            and all(isinstance(value, str) for value in row.values()),
            "full-image review Stage-10 unit row has the wrong field count",
        )
        unit_name = portable_basename(
            row["unit_name"],
            "full-image review Stage-10 unit name",
        )
        image_id = row["image_id"]
        record = image_records.get(image_id)
        _require(
            record is not None
            and unit_name not in bounds
            and image_id not in observed_images
            and Path(unit_name).suffix.lower() == ".png"
            and _SHA256.fullmatch(row["unit_sha256"]) is not None,
            "full-image review Stage-10 unit identity is invalid or duplicated",
        )
        width = _strict_int(
            row["width"],
            "full-image review Stage-10 unit width",
            minimum=1,
            maximum=_MAX_FULL_IMAGE_DIMENSION,
        )
        height = _strict_int(
            row["height"],
            "full-image review Stage-10 unit height",
            minimum=1,
            maximum=_MAX_FULL_IMAGE_DIMENSION,
        )
        _require(
            row["image_sha256"] == record["image_sha256"]
            and row["image_name"] == record["image_name"]
            and row["group_id"] == record["group_id"]
            and row["x"] == "0"
            and row["y"] == "0"
            and row["processing_unit_kind"] == _FULL_IMAGE_UNIT_KIND
            and width == int(record["warped_width"])
            and height == int(record["warped_height"])
            and _strict_int(row["orig_w"], "full-image review original width", minimum=1)
            == int(record["source_width"])
            and _strict_int(row["orig_h"], "full-image review original height", minimum=1)
            == int(record["source_height"]),
            "full-image review Stage-10 unit/image binding changed",
        )
        bounds[unit_name] = _FullImageUnitBounds(
            image_id=image_id,
            image_sha256=row["image_sha256"],
            image_name=row["image_name"],
            group_id=row["group_id"],
            unit_sha256=row["unit_sha256"],
            width=width,
            height=height,
        )
        observed_images.add(image_id)
    _require(
        set(image_records) == observed_images
        and _strict_int(summary.get("image_count"), "full-image review image count", minimum=1)
        == len(image_records)
        and _strict_int(
            summary.get("processing_unit_count"),
            "full-image review unit count",
            minimum=1,
        )
        == len(bounds)
        == len(image_records)
        and _strict_int(summary.get("group_count"), "full-image review group count", minimum=1)
        == len({str(record["group_id"]) for record in image_records.values()}),
        "full-image review Stage-10 unit closure changed",
    )
    return bounds


def _stream_proposals(
    snapshot: ReviewRunSnapshot,
    overlay_map: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, ProposalView], dict[str, tuple[int, ...]], str]:
    full_image_units = _full_image_unit_bounds(snapshot)
    path = snapshot.root / "stages" / "20_proposals_features" / "proposals.csv"
    path_info = path.lstat()
    _require(
        stat.S_ISREG(path_info.st_mode)
        and not path.is_symlink()
        and path_info.st_nlink == 1
        and 0 < path_info.st_size <= MAX_PROPOSALS_BYTES,
        "review proposal table is unsafe or too large",
    )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    proposals: dict[str, ProposalView] = {}
    source_polygons: dict[str, tuple[int, ...]] = {}
    resources = ExitStack()
    try:
        resources.callback(os.close, descriptor)
        resources.enter_context(bounded_csv_field_limit())
        opened = os.fstat(descriptor)
        _require(
            (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
            == (path_info.st_dev, path_info.st_ino, path_info.st_mode, path_info.st_nlink, path_info.st_size, path_info.st_mtime_ns, path_info.st_ctime_ns),
            "review proposal table changed while opening",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as binary:
            class _DigestReader(io.RawIOBase):
                def readable(self) -> bool:
                    return True

                def readinto(self, buffer: bytearray) -> int:
                    chunk = binary.read(len(buffer))
                    if not chunk:
                        return 0
                    digest.update(chunk)
                    buffer[: len(chunk)] = chunk
                    return len(chunk)

            buffered = io.BufferedReader(_DigestReader(), buffer_size=1024 * 1024)
            with io.TextIOWrapper(buffered, encoding="utf-8", newline="") as text:
                reader = csv.DictReader(text)
                columns = tuple(reader.fieldnames or ())
                required = {
                    "proposal_id",
                    "proposal_sha256",
                    "image_id",
                    "image_name",
                    "group_id",
                    "tile_name",
                    "tile_x",
                    "tile_y",
                    "proposal_index",
                    "mask_sha256",
                    "tile_bbox_x",
                    "tile_bbox_y",
                    "tile_bbox_w",
                    "tile_bbox_h",
                    "poly",
                    "predicted_iou",
                    "stability_score",
                }
                if full_image_units is not None:
                    required |= {
                        "image_sha256",
                        "tile_sha256",
                        "mask_area",
                        "mask_encoding",
                        "mask_height",
                        "mask_width",
                        "mask_packbits_base64",
                    }
                _require(len(columns) == len(set(columns)) and required <= set(columns), "review proposal table header mismatch")
                for row_number, row in enumerate(reader, start=2):
                    _require(None not in row and tuple(row) == columns, f"review proposal row {row_number} has the wrong field count")
                    proposal_id = row["proposal_id"]
                    _require(
                        _SHA256.fullmatch(proposal_id) is not None
                        and row["proposal_sha256"] == proposal_id
                        and proposal_id in overlay_map
                        and proposal_id not in proposals,
                        f"review proposal identity mismatch at row {row_number}",
                    )
                    overlay = overlay_map[proposal_id]
                    _require(
                        row["tile_name"] == overlay["tile_name"]
                        and row["mask_sha256"] == overlay["mask_sha256"]
                        and _strict_int(row["proposal_index"], "proposal index", minimum=1) == overlay["proposal_index"],
                        f"review proposal/overlay mapping mismatch at row {row_number}",
                    )
                    if full_image_units is None:
                        bbox_x = _strict_int(row["tile_bbox_x"], "tile bbox x", maximum=511)
                        bbox_y = _strict_int(row["tile_bbox_y"], "tile bbox y", maximum=511)
                        bbox_w = _strict_int(row["tile_bbox_w"], "tile bbox width", minimum=1, maximum=512)
                        bbox_h = _strict_int(row["tile_bbox_h"], "tile bbox height", minimum=1, maximum=512)
                        _require(bbox_x + bbox_w <= 512 and bbox_y + bbox_h <= 512, "review proposal tile bbox exceeds 512x512")
                        tile_x = _strict_int(row["tile_x"], "tile x")
                        tile_y = _strict_int(row["tile_y"], "tile y")
                    else:
                        unit = full_image_units.get(row["tile_name"])
                        _require(
                            unit is not None
                            and row["image_id"] == unit.image_id
                            and row["image_sha256"] == unit.image_sha256
                            and row["image_name"] == unit.image_name
                            and row["group_id"] == unit.group_id
                            and row["tile_sha256"] == unit.unit_sha256,
                            f"review proposal/full-image unit identity mismatch at row {row_number}",
                        )
                        tile_x = _strict_int(row["tile_x"], "full-image unit x", maximum=0)
                        tile_y = _strict_int(row["tile_y"], "full-image unit y", maximum=0)
                        bbox_x = _strict_int(
                            row["tile_bbox_x"],
                            "full-image bbox x",
                            maximum=unit.width - 1,
                        )
                        bbox_y = _strict_int(
                            row["tile_bbox_y"],
                            "full-image bbox y",
                            maximum=unit.height - 1,
                        )
                        bbox_w = _strict_int(
                            row["tile_bbox_w"],
                            "full-image bbox width",
                            minimum=1,
                            maximum=unit.width,
                        )
                        bbox_h = _strict_int(
                            row["tile_bbox_h"],
                            "full-image bbox height",
                            minimum=1,
                            maximum=unit.height,
                        )
                        canvas_height = _strict_int(
                            row["mask_height"],
                            "full-image mask canvas height",
                            minimum=1,
                            maximum=_MAX_FULL_IMAGE_DIMENSION,
                        )
                        canvas_width = _strict_int(
                            row["mask_width"],
                            "full-image mask canvas width",
                            minimum=1,
                            maximum=_MAX_FULL_IMAGE_DIMENSION,
                        )
                        area = _strict_int(
                            row["mask_area"],
                            "full-image mask area",
                            minimum=1,
                            maximum=bbox_w * bbox_h,
                        )
                        _require(
                            bbox_x + bbox_w <= unit.width
                            and bbox_y + bbox_h <= unit.height
                            and canvas_height == unit.height
                            and canvas_width == unit.width
                            and row["mask_encoding"] == _FULL_IMAGE_MASK_ENCODING,
                            "review proposal full-image mask contract changed",
                        )
                        try:
                            import base64
                            import binascii

                            packed = base64.b64decode(
                                row["mask_packbits_base64"].encode("ascii"),
                                validate=True,
                            )
                        except (UnicodeError, ValueError, binascii.Error) as exc:
                            raise PublicIOError(
                                f"review proposal full-image mask payload is invalid at row {row_number}"
                            ) from exc
                        try:
                            from compag_curation.canonical.full_image import PackedMask

                            PackedMask(
                                canvas_height=canvas_height,
                                canvas_width=canvas_width,
                                origin_x=bbox_x,
                                origin_y=bbox_y,
                                width=bbox_w,
                                height=bbox_h,
                                area=area,
                                column_stride=(bbox_h + 7) // 8,
                                packed=packed,
                                mask_sha256=row["mask_sha256"],
                            )
                        except (TypeError, ValueError) as exc:
                            raise PublicIOError(
                                f"review proposal full-image mask binding is invalid at row {row_number}"
                            ) from exc
                    try:
                        raw_polygon = json.loads(row["poly"])
                    except (json.JSONDecodeError, RecursionError) as exc:
                        raise PublicIOError(
                            f"review proposal polygon is invalid at row {row_number}"
                        ) from exc
                    _require(
                        isinstance(raw_polygon, list)
                        and len(raw_polygon) % 2 == 0
                        and all(
                            isinstance(value, int)
                            and not isinstance(value, bool)
                            and value >= 0
                            for value in raw_polygon
                        ),
                        f"review proposal polygon is invalid at row {row_number}",
                    )
                    proposals[proposal_id] = ProposalView(
                        proposal_id=proposal_id,
                        image_id=row["image_id"],
                        image_name=portable_basename(row["image_name"], "review image name"),
                        group_id=row["group_id"],
                        tile_name=row["tile_name"],
                        tile_index=int(overlay["tile_index"]),
                        proposal_index=int(overlay["proposal_index"]),
                        label_prefix=str(overlay["label_prefix"]),
                        tile_x=tile_x,
                        tile_y=tile_y,
                        bbox_x=bbox_x,
                        bbox_y=bbox_y,
                        bbox_w=bbox_w,
                        bbox_h=bbox_h,
                        predicted_iou=_strict_float(row["predicted_iou"], "predicted IoU"),
                        stability_score=_strict_float(row["stability_score"], "stability score"),
                    )
                    source_polygons[proposal_id] = tuple(raw_polygon)
        completed = os.fstat(descriptor)
        after = path.lstat()
        _require(
            (completed.st_dev, completed.st_ino, completed.st_mode, completed.st_nlink, completed.st_size, completed.st_mtime_ns, completed.st_ctime_ns)
            == (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_mode, after.st_nlink, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "review proposal table changed while indexing",
        )
    except (csv.Error, UnicodeError) as exc:
        raise PublicIOError("review proposal table cannot be parsed") from exc
    finally:
        resources.close()
    _require(set(proposals) == set(overlay_map), "review proposal/overlay ID sets differ")
    _require(set(source_polygons) == set(proposals), "review proposal polygon set differs")
    return proposals, source_polygons, digest.hexdigest()


def build_review_index(snapshot: ReviewRunSnapshot) -> ReviewIndex:
    request_rows, request_meta = read_review_export(snapshot.review_request)
    columns = tuple(str(value) for value in request_meta["columns"])
    _require(columns in {REVIEW_COLUMNS, CANONICAL_REVIEW_COLUMNS}, "review request schema is unsupported")
    tiles, overlay_map, overlay_digest = _overlay_index(snapshot)
    proposals, source_polygons, proposals_digest = _stream_proposals(snapshot, overlay_map)
    request_ids = [row["proposal_id"] for row in request_rows]
    _require(set(request_ids) == set(proposals), "review request/proposal ID sets differ")
    for row in request_rows:
        view = proposals[row["proposal_id"]]
        _require(
            row["image_id"] == view.image_id and row["group_id"] == view.group_id,
            "review request/proposal identity differs",
        )
    visual_order = tuple(
        sorted(
            proposals,
            key=lambda proposal_id: (
                proposals[proposal_id].image_name,
                proposals[proposal_id].tile_y,
                proposals[proposal_id].tile_x,
                proposals[proposal_id].proposal_index,
                proposal_id,
            ),
        )
    )
    return ReviewIndex(
        request_rows=request_rows,
        request_columns=columns,
        order=visual_order,
        proposals=proposals,
        tiles=tiles,
        groups=tuple(sorted({view.group_id for view in proposals.values()})),
        overlay_manifest_sha256=overlay_digest,
        proposals_sha256=proposals_digest,
        source_polygons=source_polygons,
    )


def _decision_payload(decision: Decision | None) -> dict[str, object] | None:
    return None if decision is None else asdict(decision)


def _parse_decision(value: object, *, canonical: bool) -> Decision | None:
    if value is None:
        return None
    _require(isinstance(value, dict) and set(value) == set(Decision.__dataclass_fields__), "review event decision is malformed")
    choice = value.get("choice")
    _require(isinstance(choice, str) and choice in DECISIONS, "review event choice is invalid")
    expected = DECISIONS[choice]
    if not canonical:
        expected = Decision(expected.choice, expected.label, None, None)
    _require(value == asdict(expected), "review event decision contract changed")
    return expected


class ReviewSession:
    """One exclusive durable review session over a shared-locked run snapshot."""

    def __init__(
        self,
        snapshot: ReviewRunSnapshot,
        index: ReviewIndex,
        state_root: Path,
        output: Path,
        lock_fd: int,
        session_sha256: str,
        model_context: Any | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.index = index
        self.state_root = state_root
        self.output = output
        self.model_context = model_context
        self._lock_fd = lock_fd
        _require(
            _SHA256.fullmatch(session_sha256) is not None,
            "review session receipt SHA-256 is invalid",
        )
        self._session_sha256 = session_sha256
        self._mutex = threading.RLock()
        self._decisions: dict[str, Decision] = {}
        self._decision_origins: dict[str, str] = {}
        self._undo_stack: list[dict[str, object]] = []
        self._requests: dict[str, tuple[object, ...]] = {}
        self._last_sequence = 0
        self._last_event_sha256 = "0" * 64
        self._final: dict[str, object] | None = None
        self._scene_catalog: object | None = None
        self._load_events()
        self._load_final()

    @property
    def canonical(self) -> bool:
        return self.index.request_columns == CANONICAL_REVIEW_COLUMNS

    @classmethod
    def open(
        cls,
        run_root: Path,
        output: Path,
        state_root: Path | None = None,
        *,
        model_preset: str | None = None,
    ) -> "ReviewSession":
        snapshot = open_review_snapshot(run_root)
        lock_fd = -1
        try:
            output = _external_path(
                output,
                snapshot.root,
                "review output",
                derived_basename=(
                    f".{output.name}.compag-review.{uuid.uuid4()}.tmp"
                ),
            )
            requested_state = state_root or output.with_name(
                f"{output.name}.review-session"
            )
            state_root = _external_path(
                requested_state,
                snapshot.root,
                "review session directory",
                derived_basename=f".{requested_state.name}.init.{uuid.uuid4()}",
            )
            _require(output != state_root and state_root not in output.parents and output not in state_root.parents, "review output and session paths overlap")
            index = build_review_index(snapshot)
            model_context = None
            if model_preset is not None:
                from compag_curation.review.published_model import (
                    load_published_model_context,
                    score_published_model,
                )

                model_context = (
                    load_published_model_context(
                        snapshot,
                        index,
                        state_root,
                        model_preset,
                    )
                    if state_root.exists() and not state_root.is_symlink()
                    else score_published_model(
                        snapshot,
                        index,
                        model_preset,
                    )
                )
                index = replace(index, order=model_context.order)
            if state_root.exists() or state_root.is_symlink():
                _directory_is_private(state_root, "review session directory")
            else:
                _require(not output.exists() and not output.is_symlink(), "review output already exists")
                cls._create_state(
                    snapshot,
                    index,
                    state_root,
                    output,
                    model_context=model_context,
                )
            lock_path = state_root / "SESSION.LOCK"
            lock_fd = os.open(lock_path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PublicIOError("REVIEW_SESSION_ALREADY_ACTIVE") from exc
            lock_info = os.fstat(lock_fd)
            _require(
                stat.S_ISREG(lock_info.st_mode)
                and lock_info.st_nlink == 1
                and stat.S_IMODE(lock_info.st_mode) == 0o600
                and lock_info.st_uid == os.getuid(),
                "review session lock is unsafe",
            )
            session_sha256 = cls._validate_receipt(
                snapshot,
                index,
                state_root,
                output,
                model_context=model_context,
            )
            return cls(
                snapshot,
                index,
                state_root,
                output,
                lock_fd,
                session_sha256,
                model_context,
            )
        except BaseException:
            if lock_fd >= 0:
                os.close(lock_fd)
            snapshot.close()
            raise

    @staticmethod
    def _receipt(
        snapshot: ReviewRunSnapshot,
        index: ReviewIndex,
        output: Path,
        *,
        created_at_utc: str,
        model_context: Any | None = None,
    ) -> dict[str, object]:
        receipt = {
            "schema": (
                SESSION_SCHEMA
                if model_context is None
                else MODEL_ASSISTED_SESSION_SCHEMA
            ),
            "created_at_utc": created_at_utc,
            "run_id": snapshot.run_id,
            "run_root": str(snapshot.root),
            "stage10_receipt_sha256": snapshot.stage10_receipt_sha256,
            "stage20_receipt_sha256": snapshot.stage20_receipt_sha256,
            "review_request_sha256": snapshot.review_request_sha256,
            "overlay_manifest_sha256": index.overlay_manifest_sha256,
            "proposals_sha256": index.proposals_sha256,
            "proposal_count": len(index.order),
            "visual_order_sha256": compact_json_sha256(list(index.order)),
            "review_columns": list(index.request_columns),
            "output": str(output),
            "decision_contract": {
                choice: asdict(
                    decision
                    if index.request_columns == CANONICAL_REVIEW_COLUMNS
                    else Decision(choice, decision.label, None, None)
                )
                for choice, decision in DECISIONS.items()
            },
            "state_policy": "APPEND_ONLY_HASH_CHAIN_OUTSIDE_RUN",
            "output_policy": "COMPLETE_VALIDATED_NO_CLOBBER",
        }
        if model_context is not None:
            receipt["published_model_assist"] = model_context.receipt_record()
        return receipt

    @staticmethod
    def _head_record(
        snapshot: ReviewRunSnapshot,
        index: ReviewIndex,
        session_sha256: str,
        sequence: int,
        event_sha256: str,
    ) -> dict[str, object]:
        return {
            "schema": EVENT_HEAD_SCHEMA,
            "run_id": snapshot.run_id,
            "review_request_sha256": snapshot.review_request_sha256,
            "visual_order_sha256": compact_json_sha256(list(index.order)),
            "session_sha256": session_sha256,
            "sequence": sequence,
            "event_sha256": event_sha256,
        }

    @classmethod
    def _create_state(
        cls,
        snapshot: ReviewRunSnapshot,
        index: ReviewIndex,
        state_root: Path,
        output: Path,
        *,
        model_context: Any | None = None,
    ) -> None:
        _require(not state_root.exists() and not state_root.is_symlink(), "review session path already exists")
        staging = state_root.parent / f".{state_root.name}.init.{uuid.uuid4()}"
        staging.mkdir(mode=0o700)
        try:
            (staging / "events").mkdir(mode=0o700)
            (staging / ".staging").mkdir(mode=0o700)
            write_new_bytes(staging / "SESSION.LOCK", b"COMPAG human review session lock\n", mode=0o600)
            if model_context is not None:
                assistance_path = staging / "MODEL_ASSISTANCE.json"
                write_new_json(
                    assistance_path,
                    model_context.assistance_record,
                    mode=0o600,
                )
                _require(
                    sha256_file(assistance_path)
                    == model_context.assistance_file_sha256,
                    "published-model assistance cache did not serialize deterministically",
                )
            receipt = cls._receipt(
                snapshot,
                index,
                output,
                created_at_utc=_now(),
                model_context=model_context,
            )
            write_new_json(
                staging / "SESSION.json",
                receipt,
                mode=0o600,
            )
            session_sha256 = hashlib.sha256(
                canonical_json_bytes(receipt)
            ).hexdigest()
            write_new_json(
                staging / "EVENT_HEAD.json",
                cls._head_record(
                    snapshot,
                    index,
                    session_sha256,
                    0,
                    "0" * 64,
                ),
                mode=0o600,
            )
            fsync_directory(staging / "events")
            fsync_directory(staging / ".staging")
            fsync_directory(staging)
            publish_directory_noreplace(staging, state_root)
        except BaseException:
            # A failed unpublished initializer is private scratch owned by this
            # call.  Leave it in place for forensic inspection rather than
            # risking broad cleanup.
            raise

    @classmethod
    def _validate_receipt(
        cls,
        snapshot: ReviewRunSnapshot,
        index: ReviewIndex,
        state_root: Path,
        output: Path,
        *,
        model_context: Any | None = None,
    ) -> str:
        _directory_is_private(state_root, "review session directory")
        _directory_is_private(state_root / "events", "review event directory")
        _directory_is_private(state_root / ".staging", "review staging directory")
        payload, _info = _regular_private(state_root / "SESSION.json", "review session receipt", max_bytes=256 * 1024)
        value = canonical_json_value(payload, "review session receipt")
        _require(
            isinstance(value, dict)
            and value.get("schema")
            == (
                SESSION_SCHEMA
                if model_context is None
                else MODEL_ASSISTED_SESSION_SCHEMA
            ),
            "review session receipt schema mismatch",
        )
        _require_utc_timestamp(value.get("created_at_utc"), "review session creation time")
        expected = cls._receipt(
            snapshot,
            index,
            output,
            created_at_utc=str(value["created_at_utc"]),
            model_context=model_context,
        )
        _require(
            set(value) == set(expected)
            and all(
                value.get(key) == expected_value
                for key, expected_value in expected.items()
            ),
            "review session binding changed",
        )
        session_sha256 = hashlib.sha256(payload).hexdigest()
        head = canonical_json_value(
            _regular_private(
                state_root / "EVENT_HEAD.json",
                "review event head",
                max_bytes=256 * 1024,
            )[0],
            "review event head",
        )
        expected_head = cls._head_record(
            snapshot,
            index,
            session_sha256,
            0,
            "0" * 64,
        )
        _require(
            isinstance(head, dict)
            and set(head) == set(expected_head)
            and head.get("schema") == EVENT_HEAD_SCHEMA
            and head.get("run_id") == snapshot.run_id
            and head.get("review_request_sha256")
            == snapshot.review_request_sha256
            and head.get("visual_order_sha256")
            == compact_json_sha256(list(index.order))
            and head.get("session_sha256") == session_sha256,
            "review event head binding changed",
        )
        return session_sha256

    def close(self) -> None:
        with self._mutex:
            if self._lock_fd >= 0:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
                self._lock_fd = -1
            self.snapshot.close()

    def __enter__(self) -> "ReviewSession":
        return self

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        self.close()

    def _event_limit(self) -> int:
        return min(
            MAX_EVENT_COUNT,
            max(MIN_EVENT_COUNT, len(self.index.order) * EVENTS_PER_PROPOSAL),
        )

    def _read_head(self) -> dict[str, object]:
        receipt_payload, _receipt_info = _regular_private(
            self.state_root / "SESSION.json",
            "review session receipt",
            max_bytes=256 * 1024,
        )
        _require(
            hashlib.sha256(receipt_payload).hexdigest() == self._session_sha256,
            "review session receipt changed after validation",
        )
        payload, _head_info = _regular_private(
            self.state_root / "EVENT_HEAD.json",
            "review event head",
            max_bytes=256 * 1024,
        )
        value = canonical_json_value(payload, "review event head")
        expected_fields = set(
            self._head_record(
                self.snapshot,
                self.index,
                self._session_sha256,
                0,
                "0" * 64,
            )
        )
        _require(
            isinstance(value, dict)
            and set(value) == expected_fields
            and value.get("schema") == EVENT_HEAD_SCHEMA
            and value.get("run_id") == self.snapshot.run_id
            and value.get("review_request_sha256")
            == self.snapshot.review_request_sha256
            and value.get("visual_order_sha256")
            == compact_json_sha256(list(self.index.order))
            and value.get("session_sha256") == self._session_sha256,
            "review event head binding changed",
        )
        sequence = value.get("sequence")
        event_sha256 = value.get("event_sha256")
        _require(
            type(sequence) is int
            and 0 <= sequence <= self._event_limit()
            and isinstance(event_sha256, str)
            and _SHA256.fullmatch(event_sha256) is not None
            and ((sequence == 0) == (event_sha256 == "0" * 64)),
            "review event head value is invalid",
        )
        return value

    def _replace_head(
        self,
        expected: Mapping[str, object],
        sequence: int,
        event_sha256: str,
    ) -> None:
        _require(
            self._read_head() == dict(expected),
            "review event head changed while advancing",
        )
        replacement = self._head_record(
            self.snapshot,
            self.index,
            self._session_sha256,
            sequence,
            event_sha256,
        )
        staged = (
            self.state_root
            / ".staging"
            / f"event-head.{sequence:08d}.{uuid.uuid4()}.json"
        )
        write_new_json(staged, replacement, mode=0o600)
        try:
            _require(
                self._read_head() == dict(expected),
                "review event head changed during advancement",
            )
            os.replace(staged, self.state_root / "EVENT_HEAD.json")
            fsync_directory(self.state_root)
            fsync_directory(staged.parent)
            _require(
                self._read_head() == replacement,
                "review event head replacement was not durable",
            )
        except BaseException:
            if staged.exists() and not staged.is_symlink():
                staged.unlink()
                fsync_directory(staged.parent)
            raise

    def _event_names(self) -> list[str]:
        names: list[str] = []
        with os.scandir(self.state_root / "events") as entries:
            for entry in entries:
                names.append(entry.name)
                _require(
                    len(names) <= self._event_limit(),
                    "review event count exceeds its supported bound",
                )
        _require(
            all(_EVENT_NAME.fullmatch(name) is not None for name in names),
            "review event directory contains an unexpected member",
        )
        return sorted(names)

    def _replay_events(
        self,
        *,
        recover_ahead: bool,
    ) -> tuple[
        dict[str, Decision],
        list[dict[str, object]],
        dict[str, tuple[object, ...]],
        int,
        str,
        dict[str, str],
    ]:
        events = self.state_root / "events"
        names = self._event_names()
        decisions: dict[str, Decision] = {}
        decision_origins: dict[str, str] = {}
        undo_stack: list[dict[str, object]] = []
        requests: dict[str, tuple[object, ...]] = {}
        last_sequence = 0
        last_event_sha256 = "0" * 64
        event_hashes: list[str] = []
        canonical = self.canonical
        for expected_sequence, name in enumerate(names, start=1):
            match = _EVENT_NAME.fullmatch(name)
            _require(
                match is not None and int(match.group(1)) == expected_sequence,
                "review event sequence has a gap",
            )
            payload, _info = _regular_private(
                events / name,
                "review event",
                max_bytes=MAX_EVENT_BYTES,
            )
            event = canonical_json_value(payload, "review event")
            _require(
                isinstance(event, dict) and event.get("schema") == EVENT_SCHEMA,
                "review event schema mismatch",
            )
            supplied_hash = event.get("event_sha256")
            without_hash = {
                key: value for key, value in event.items() if key != "event_sha256"
            }
            _require(
                event.get("sequence") == expected_sequence
                and event.get("previous_event_sha256") == last_event_sha256
                and supplied_hash == compact_json_sha256(without_hash),
                "review event hash chain is invalid",
            )
            _require_utc_timestamp(
                event.get("timestamp_utc"), "review event timestamp"
            )
            operation = event.get("operation")
            proposal_id = event.get("proposal_id")
            request_id = event.get("request_id")
            _require(
                proposal_id in self.index.proposals,
                "review event references an unknown proposal",
            )
            _require(
                isinstance(request_id, str)
                and _REQUEST_ID.fullmatch(request_id) is not None
                and request_id not in requests,
                "review event request identity is invalid or duplicated",
            )
            if operation == "SET":
                _require(
                    set(event)
                    == {
                        "schema",
                        "sequence",
                        "previous_event_sha256",
                        "timestamp_utc",
                        "operation",
                        "request_id",
                        "proposal_id",
                        "previous_decision",
                        "decision",
                        "event_sha256",
                    },
                    "review SET event fields are invalid",
                )
                previous = _parse_decision(
                    event.get("previous_decision"), canonical=canonical
                )
                decision = _parse_decision(
                    event.get("decision"), canonical=canonical
                )
                _require(
                    decision is not None
                    and decisions.get(proposal_id) == previous,
                    "review SET event previous state mismatch",
                )
                previous_origin = decision_origins.get(str(proposal_id))
                _require(
                    (previous is None) == (previous_origin is None),
                    "review SET event previous origin mismatch",
                )
                decisions[str(proposal_id)] = decision
                decision_origins[str(proposal_id)] = HUMAN_DECISION_ORIGIN
                undo_stack.append(
                    {
                        "sequence": expected_sequence,
                        "proposal_id": proposal_id,
                        "previous": previous,
                        "previous_origin": previous_origin,
                    }
                )
                requests[request_id] = (operation, proposal_id, decision.choice)
            elif operation == "UNDO":
                _require(
                    set(event)
                    == {
                        "schema",
                        "sequence",
                        "previous_event_sha256",
                        "timestamp_utc",
                        "operation",
                        "request_id",
                        "proposal_id",
                        "target_sequence",
                        "decision",
                        "event_sha256",
                    },
                    "review UNDO event fields are invalid",
                )
                _require(undo_stack, "review UNDO event has no target")
                target = undo_stack.pop()
                _require(
                    "bulk_ids" not in target,
                    "published-model confirmation requires its bulk undo event",
                )
                restored = _parse_decision(
                    event.get("decision"), canonical=canonical
                )
                previous_origin = target.get("previous_origin")
                _require(
                    event.get("target_sequence") == target["sequence"]
                    and proposal_id == target["proposal_id"]
                    and restored == target["previous"],
                    "review UNDO event target mismatch",
                )
                if restored is None:
                    decisions.pop(str(proposal_id), None)
                    decision_origins.pop(str(proposal_id), None)
                else:
                    _require(
                        previous_origin
                        in {HUMAN_DECISION_ORIGIN, MODEL_DECISION_ORIGIN},
                        "review UNDO origin is invalid",
                    )
                    decisions[str(proposal_id)] = restored
                    decision_origins[str(proposal_id)] = str(previous_origin)
                requests[request_id] = (
                    operation,
                    proposal_id,
                    event.get("target_sequence"),
                )
            elif operation == "CONFIRM_MODEL_REMAINDER":
                _require(
                    self.model_context is not None and canonical,
                    "published-model confirmation requires a canonical assisted session",
                )
                _require(
                    set(event)
                    == {
                        "schema",
                        "sequence",
                        "previous_event_sha256",
                        "timestamp_utc",
                        "operation",
                        "request_id",
                        "proposal_id",
                        "model_preset",
                        "proposal_count",
                        "proposal_ids_sha256",
                        "event_sha256",
                    },
                    "published-model confirmation event fields are invalid",
                )
                pending = tuple(
                    item_id
                    for item_id in self.index.order
                    if item_id not in decisions
                )
                _require(
                    bool(pending)
                    and proposal_id == pending[0]
                    and event.get("model_preset")
                    == self.model_context.preset_id
                    and event.get("proposal_count") == len(pending)
                    and event.get("proposal_ids_sha256")
                    == compact_json_sha256(list(pending)),
                    "published-model confirmation set changed during replay",
                )
                for item_id in pending:
                    decisions[item_id] = self._model_remainder_decision(item_id)
                    decision_origins[item_id] = MODEL_DECISION_ORIGIN
                undo_stack.append(
                    {
                        "sequence": expected_sequence,
                        "proposal_id": proposal_id,
                        "previous": None,
                        "previous_origin": None,
                        "bulk_ids": pending,
                    }
                )
                requests[request_id] = (
                    operation,
                    proposal_id,
                    len(pending),
                    compact_json_sha256(list(pending)),
                )
            elif operation == "UNDO_MODEL_REMAINDER":
                _require(
                    self.model_context is not None and canonical,
                    "published-model undo requires a canonical assisted session",
                )
                _require(
                    set(event)
                    == {
                        "schema",
                        "sequence",
                        "previous_event_sha256",
                        "timestamp_utc",
                        "operation",
                        "request_id",
                        "proposal_id",
                        "target_sequence",
                        "proposal_count",
                        "proposal_ids_sha256",
                        "event_sha256",
                    },
                    "published-model undo event fields are invalid",
                )
                _require(bool(undo_stack), "published-model undo has no target")
                target = undo_stack.pop()
                bulk_ids = target.get("bulk_ids")
                _require(
                    isinstance(bulk_ids, tuple)
                    and bool(bulk_ids)
                    and event.get("target_sequence") == target["sequence"]
                    and proposal_id == target["proposal_id"] == bulk_ids[0]
                    and event.get("proposal_count") == len(bulk_ids)
                    and event.get("proposal_ids_sha256")
                    == compact_json_sha256(list(bulk_ids))
                    and all(
                        decisions.get(item_id)
                        == self._model_remainder_decision(item_id)
                        for item_id in bulk_ids
                    ),
                    "published-model undo target changed during replay",
                )
                for item_id in bulk_ids:
                    decisions.pop(item_id, None)
                    decision_origins.pop(item_id, None)
                requests[request_id] = (
                    operation,
                    proposal_id,
                    event.get("target_sequence"),
                )
            else:
                raise PublicIOError("review event operation is invalid")
            last_sequence = expected_sequence
            last_event_sha256 = str(supplied_hash)
            event_hashes.append(last_event_sha256)

        head = self._read_head()
        head_sequence = int(head["sequence"])
        _require(
            head_sequence <= last_sequence,
            "review event head is ahead of the durable event directory",
        )
        expected_head_hash = (
            "0" * 64
            if head_sequence == 0
            else event_hashes[head_sequence - 1]
        )
        _require(
            head.get("event_sha256") == expected_head_hash,
            "review event head hash does not match the durable chain",
        )
        _require(
            self._event_names() == names and self._read_head() == head,
            "review event state changed during replay",
        )
        if head_sequence < last_sequence:
            # The event file is published and fsynced before its head.  A
            # complete valid suffix with an older head is therefore the one
            # recoverable crash state; advance only after replaying all of it.
            _require(
                recover_ahead,
                "review event head is behind the durable event directory",
            )
            self._replace_head(head, last_sequence, last_event_sha256)
            _require(
                self._event_names() == names,
                "review event directory changed during head recovery",
            )
        return (
            decisions,
            undo_stack,
            requests,
            last_sequence,
            last_event_sha256,
            decision_origins,
        )

    def _load_events(self) -> None:
        (
            self._decisions,
            self._undo_stack,
            self._requests,
            self._last_sequence,
            self._last_event_sha256,
            self._decision_origins,
        ) = self._replay_events(recover_ahead=True)

    def _verify_durable_events(self) -> dict[str, Decision]:
        replayed = self._replay_events(recover_ahead=False)
        _require(
            replayed[0] == self._decisions
            and replayed[1] == self._undo_stack
            and replayed[2] == self._requests
            and replayed[3] == self._last_sequence
            and replayed[4] == self._last_event_sha256
            and replayed[5] == self._decision_origins,
            "durable review events differ from the live session",
        )
        return replayed[0]

    def _load_final(self) -> None:
        path = self.state_root / "FINAL.json"
        if not path.exists() and not path.is_symlink():
            if not self.output.exists() and not self.output.is_symlink():
                return
            _require(
                len(self._decisions) == len(self.index.order),
                "review output exists without a final receipt while the session is incomplete",
            )
            output_payload, _output_info = _regular_private(
                self.output,
                "review output",
                max_bytes=128 * 1024 * 1024,
            )
            _require(
                output_payload == self._render_csv(),
                "review output exists without a receipt and differs from durable decisions",
            )
            rows, validation = validate_review_table(self.snapshot.review_request, self.output)
            feasibility = self._split_feasibility(rows)
            self._verify_durable_events()
            final = self._final_record(validation, feasibility, recovery=True)
            write_new_json(path, final, mode=0o600)
            self._final = final
            return
        payload, _info = _regular_private(path, "review final receipt", max_bytes=256 * 1024)
        value = canonical_json_value(payload, "review final receipt")
        final_fields = {
            "schema",
            "status",
            "finalized_at_utc",
            "recovered_after_output_publication",
            "run_id",
            "review_request_sha256",
            "reviewed_sha256",
            "rows",
            "positive_rows",
            "negative_rows",
            "action_counts",
            "effective_weight",
            "split_feasibility",
            "event_sequence",
            "event_sha256",
            "output",
            "input_policy",
            "publication_policy",
        }
        if self.model_context is not None:
            final_fields.update(
                {
                    "source_kind",
                    "published_model_assist",
                    "decision_origins",
                }
            )
        _require(
            isinstance(value, dict)
            and set(value) == final_fields
            and value.get("schema")
            == (
                FINAL_SCHEMA
                if self.model_context is None
                else MODEL_ASSISTED_FINAL_SCHEMA
            )
            and value.get("status") == "PASS"
            and value.get("run_id") == self.snapshot.run_id
            and value.get("review_request_sha256") == self.snapshot.review_request_sha256
            and value.get("output") == str(self.output)
            and value.get("event_sequence") == self._last_sequence
            and value.get("event_sha256") == self._last_event_sha256,
            "review final receipt binding mismatch",
        )
        _require_utc_timestamp(value.get("finalized_at_utc"), "review finalization time")
        _require(
            type(value.get("recovered_after_output_publication")) is bool,
            "review final recovery marker is invalid",
        )
        _require(self.output.exists() and not self.output.is_symlink(), "final reviewed CSV is missing")
        output_payload, _output_info = _regular_private(
            self.output,
            "review output",
            max_bytes=128 * 1024 * 1024,
        )
        _require(
            output_payload == self._render_csv(),
            "final reviewed CSV differs from durable decisions",
        )
        rows, result = validate_review_table(self.snapshot.review_request, self.output)
        feasibility = self._split_feasibility(rows)
        expected = {
            "reviewed_sha256": result["reviewed_sha256"],
            "rows": result["rows"],
            "positive_rows": result["positive_rows"],
            "negative_rows": result["negative_rows"],
            "action_counts": result["action_counts"],
            "effective_weight": result["effective_weight"],
            "split_feasibility": feasibility,
            "input_policy": "IMMUTABLE_RUN_UNCHANGED",
            "publication_policy": "VALIDATED_NO_CLOBBER",
        }
        if self.model_context is not None:
            expected.update(
                {
                    "source_kind": "STAGE20_PUBLISHED_MODEL_ASSIST",
                    "published_model_assist": self.model_context.receipt_record(),
                    "decision_origins": self._decision_origin_record(),
                }
            )
        _require(
            hashlib.sha256(output_payload).hexdigest() == result["reviewed_sha256"]
            and all(value.get(key) == expected_value for key, expected_value in expected.items()),
            "final reviewed CSV differs from its receipt",
        )
        self._final = value

    def _require_mutable(self) -> None:
        _require(self._lock_fd >= 0, "review session is closed")
        _require(self._final is None, "finalized review session cannot be changed")
        final_path = self.state_root / "FINAL.json"
        _require(
            not self.output.exists()
            and not self.output.is_symlink()
            and not final_path.exists()
            and not final_path.is_symlink(),
            "review publication exists or is uncertain; close and reopen the session",
        )

    def _split_feasibility(self, rows: Sequence[object]) -> dict[str, object]:
        if self.canonical:
            from compag_curation.canonical.service import canonical_review_split_feasibility

            return canonical_review_split_feasibility(rows)
        from compag_curation.public_backend import review_split_feasibility

        return review_split_feasibility(
            [(getattr(row, "group_id"), getattr(row, "label")) for row in rows]
        )

    def _decision_origin_record(self) -> dict[str, object]:
        _require(
            set(self._decision_origins) == set(self._decisions),
            "review decision-origin closure is inconsistent",
        )
        human = [
            proposal_id
            for proposal_id in self.index.order
            if self._decision_origins.get(proposal_id) == HUMAN_DECISION_ORIGIN
        ]
        model = [
            proposal_id
            for proposal_id in self.index.order
            if self._decision_origins.get(proposal_id) == MODEL_DECISION_ORIGIN
        ]
        _require(
            len(human) + len(model) == len(self._decisions),
            "review decision origin is unsupported",
        )
        return {
            "schema": "compag-curation-review-decision-origins/v1",
            "counts": {
                HUMAN_DECISION_ORIGIN: len(human),
                MODEL_DECISION_ORIGIN: len(model),
            },
            "human_explicit_proposal_ids_sha256": compact_json_sha256(human),
            "model_confirmed_proposal_ids_sha256": compact_json_sha256(model),
        }

    def _final_record(
        self,
        validation: Mapping[str, object],
        feasibility: Mapping[str, object],
        *,
        recovery: bool,
    ) -> dict[str, object]:
        result = {
            "schema": (
                FINAL_SCHEMA
                if self.model_context is None
                else MODEL_ASSISTED_FINAL_SCHEMA
            ),
            "status": "PASS",
            "finalized_at_utc": _now(),
            "recovered_after_output_publication": recovery,
            "run_id": self.snapshot.run_id,
            "review_request_sha256": self.snapshot.review_request_sha256,
            "reviewed_sha256": sha256_file(self.output),
            "rows": validation["rows"],
            "positive_rows": validation["positive_rows"],
            "negative_rows": validation["negative_rows"],
            "action_counts": validation["action_counts"],
            "effective_weight": validation["effective_weight"],
            "split_feasibility": feasibility,
            "event_sequence": self._last_sequence,
            "event_sha256": self._last_event_sha256,
            "output": str(self.output),
            "input_policy": "IMMUTABLE_RUN_UNCHANGED",
            "publication_policy": "VALIDATED_NO_CLOBBER",
        }
        if self.model_context is not None:
            result.update(
                {
                    "source_kind": "STAGE20_PUBLISHED_MODEL_ASSIST",
                    "published_model_assist": self.model_context.receipt_record(),
                    "decision_origins": self._decision_origin_record(),
                }
            )
        return result

    def _append(self, event: dict[str, object]) -> dict[str, object]:
        sequence = self._last_sequence + 1
        _require(
            sequence <= self._event_limit(),
            "review event mutation ceiling reached",
        )
        current_head = self._read_head()
        _require(
            current_head.get("sequence") == self._last_sequence
            and current_head.get("event_sha256") == self._last_event_sha256,
            "review event head differs from the live session",
        )
        base = {
            "schema": EVENT_SCHEMA,
            "sequence": sequence,
            "previous_event_sha256": self._last_event_sha256,
            "timestamp_utc": _now(),
            **event,
        }
        base["event_sha256"] = compact_json_sha256(base)
        payload = canonical_json_bytes(base)
        _require(len(payload) <= MAX_EVENT_BYTES, "review event exceeds its supported size")
        staged = self.state_root / ".staging" / f"event.{sequence:08d}.{uuid.uuid4()}.json"
        target = self.state_root / "events" / f"{sequence:08d}.json"
        write_new_bytes(staged, payload, mode=0o600)
        rename_noreplace(staged, target)
        self._replace_head(
            current_head,
            sequence,
            str(base["event_sha256"]),
        )
        self._last_sequence = sequence
        self._last_event_sha256 = str(base["event_sha256"])
        return base

    def _model_remainder_decision(self, proposal_id: str) -> Decision:
        _require(
            self.model_context is not None,
            "published-model assistance is not active",
        )
        score = self.model_context.score(proposal_id)
        return DECISIONS[
            "target_uncertain" if score.prediction == 1 else "other_uncertain"
        ]

    def confirm_model_remainder(
        self,
        *,
        request_id: str | None = None,
    ) -> dict[str, object]:
        """Explicitly project pending model suggestions as reduced-weight labels.

        This is a single durable, reversible user action.  It never claims
        that the remaining rows were inspected one by one: their existing
        ``sus_*`` actions and 0.4 weights keep them distinct from confident
        manual corrections.
        """

        with self._mutex:
            self._require_mutable()
            _require(
                self.model_context is not None and self.canonical,
                "published-model assistance is not active for this session",
            )
            request_id = request_id or str(uuid.uuid4())
            _require(
                _REQUEST_ID.fullmatch(request_id) is not None,
                "review request identity is invalid",
            )
            if request_id in self._requests:
                recorded = self._requests[request_id]
                _require(
                    recorded[0] == "CONFIRM_MODEL_REMAINDER",
                    "review request identity was reused for a different action",
                )
                return self.proposal_payload(str(recorded[1]))
            pending = tuple(
                proposal_id
                for proposal_id in self.index.order
                if proposal_id not in self._decisions
            )
            _require(
                bool(pending),
                "there are no pending model suggestions to confirm",
            )
            projected = {
                proposal_id: self._model_remainder_decision(proposal_id)
                for proposal_id in pending
            }
            _require(
                len(projected) == len(pending),
                "published-model projected decision set is incomplete",
            )
            pending_sha256 = compact_json_sha256(list(pending))
            event = self._append(
                {
                    "operation": "CONFIRM_MODEL_REMAINDER",
                    "request_id": request_id,
                    "proposal_id": pending[0],
                    "model_preset": self.model_context.preset_id,
                    "proposal_count": len(pending),
                    "proposal_ids_sha256": pending_sha256,
                }
            )
            self._decisions.update(projected)
            self._decision_origins.update(
                {proposal_id: MODEL_DECISION_ORIGIN for proposal_id in pending}
            )
            self._undo_stack.append(
                {
                    "sequence": event["sequence"],
                    "proposal_id": pending[0],
                    "previous": None,
                    "previous_origin": None,
                    "bulk_ids": pending,
                }
            )
            self._requests[request_id] = (
                "CONFIRM_MODEL_REMAINDER",
                pending[0],
                len(pending),
                pending_sha256,
            )
            return self.proposal_payload(pending[0])

    def set_decision(
        self,
        proposal_id: str,
        choice: str,
        *,
        request_id: str | None = None,
    ) -> dict[str, object]:
        with self._mutex:
            self._require_mutable()
            _require(proposal_id in self.index.proposals, "unknown proposal ID")
            _require(choice in DECISIONS, "unknown human-review choice")
            request_id = request_id or str(uuid.uuid4())
            _require(_REQUEST_ID.fullmatch(request_id) is not None, "review request identity is invalid")
            if request_id in self._requests:
                _require(
                    self._requests[request_id] == ("SET", proposal_id, choice),
                    "review request identity was reused for a different action",
                )
                return self.proposal_payload(proposal_id)
            decision = DECISIONS[choice]
            if not self.canonical:
                decision = Decision(choice, decision.label, None, None)
            previous = self._decisions.get(proposal_id)
            previous_origin = self._decision_origins.get(proposal_id)
            _require(
                (previous is None) == (previous_origin is None),
                "live review decision origin is inconsistent",
            )
            event = self._append(
                {
                    "operation": "SET",
                    "request_id": request_id,
                    "proposal_id": proposal_id,
                    "previous_decision": _decision_payload(previous),
                    "decision": _decision_payload(decision),
                }
            )
            self._decisions[proposal_id] = decision
            self._decision_origins[proposal_id] = HUMAN_DECISION_ORIGIN
            self._undo_stack.append(
                {
                    "sequence": event["sequence"],
                    "proposal_id": proposal_id,
                    "previous": previous,
                    "previous_origin": previous_origin,
                }
            )
            self._requests[request_id] = ("SET", proposal_id, choice)
            return self.proposal_payload(proposal_id)

    def undo(self, *, request_id: str | None = None) -> dict[str, object]:
        with self._mutex:
            self._require_mutable()
            request_id = request_id or str(uuid.uuid4())
            _require(_REQUEST_ID.fullmatch(request_id) is not None, "review request identity is invalid")
            if request_id in self._requests:
                recorded = self._requests[request_id]
                _require(
                    recorded[0] in {"UNDO", "UNDO_MODEL_REMAINDER"},
                    "review request identity was reused for a different action",
                )
                return self.proposal_payload(str(recorded[1]))
            _require(bool(self._undo_stack), "there is no review action to undo")
            target = self._undo_stack[-1]
            proposal_id = str(target["proposal_id"])
            bulk_ids = target.get("bulk_ids")
            if isinstance(bulk_ids, tuple):
                _require(
                    bool(bulk_ids)
                    and proposal_id == bulk_ids[0]
                    and all(
                        self._decisions.get(item_id)
                        == self._model_remainder_decision(item_id)
                        for item_id in bulk_ids
                    ),
                    "published-model confirmation changed before undo",
                )
                bulk_sha256 = compact_json_sha256(list(bulk_ids))
                self._append(
                    {
                        "operation": "UNDO_MODEL_REMAINDER",
                        "request_id": request_id,
                        "proposal_id": proposal_id,
                        "target_sequence": target["sequence"],
                        "proposal_count": len(bulk_ids),
                        "proposal_ids_sha256": bulk_sha256,
                    }
                )
                self._undo_stack.pop()
                for item_id in bulk_ids:
                    self._decisions.pop(item_id, None)
                    self._decision_origins.pop(item_id, None)
                self._requests[request_id] = (
                    "UNDO_MODEL_REMAINDER",
                    proposal_id,
                    target["sequence"],
                )
                return self.proposal_payload(proposal_id)
            restored = target["previous"]
            previous_origin = target.get("previous_origin")
            self._append(
                {
                    "operation": "UNDO",
                    "request_id": request_id,
                    "proposal_id": proposal_id,
                    "target_sequence": target["sequence"],
                    "decision": _decision_payload(restored if isinstance(restored, Decision) else None),
                }
            )
            self._undo_stack.pop()
            if restored is None:
                self._decisions.pop(proposal_id, None)
                self._decision_origins.pop(proposal_id, None)
            else:
                _require(isinstance(restored, Decision), "undo restoration is invalid")
                _require(
                    previous_origin
                    in {HUMAN_DECISION_ORIGIN, MODEL_DECISION_ORIGIN},
                    "undo restoration origin is invalid",
                )
                self._decisions[proposal_id] = restored
                self._decision_origins[proposal_id] = str(previous_origin)
            self._requests[request_id] = ("UNDO", proposal_id, target["sequence"])
            return self.proposal_payload(proposal_id)

    def _counts(self) -> dict[str, int]:
        counts = {choice: 0 for choice in DECISIONS}
        for decision in self._decisions.values():
            counts[decision.choice] += 1
        return counts

    def summary(self) -> dict[str, object]:
        with self._mutex:
            total = len(self.index.order)
            reviewed = len(self._decisions)
            first_pending = next((proposal_id for proposal_id in self.index.order if proposal_id not in self._decisions), None)
            result = {
                "schema": "compag-curation-human-review-status/v1",
                "source_kind": (
                    "STAGE20_INITIAL"
                    if self.model_context is None
                    else "STAGE20_PUBLISHED_MODEL_ASSIST"
                ),
                "round_number": None,
                "status": "FINALIZED" if self._final is not None else ("READY_TO_FINALIZE" if reviewed == total else "IN_PROGRESS"),
                "run_id": self.snapshot.run_id,
                "total": total,
                "reviewed": reviewed,
                "remaining": total - reviewed,
                "progress_percent": round(100.0 * reviewed / total, 2),
                "counts": self._counts(),
                "groups": list(self.index.groups),
                "first_pending": first_pending,
                "event_sequence": self._last_sequence,
                "output": str(self.output),
                "state_root": str(self.state_root),
                "canonical_actions": self.canonical,
                "final": self._final,
            }
            if self.model_context is not None:
                result["published_model_assist"] = (
                    self.model_context.summary_record()
                )
                result["decision_origins"] = self._decision_origin_record()
                result["can_confirm_model_remainder"] = (
                    self._final is None and reviewed < total
                )
            return result

    def _tile_proposals(self, tile: TileView) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for proposal_id in tile.proposal_ids:
            view = self.index.proposals[proposal_id]
            decision = self._decisions.get(proposal_id)
            result.append(
                {
                    "proposal_id": proposal_id,
                    "label_prefix": view.label_prefix,
                    "bbox": [view.bbox_x, view.bbox_y, view.bbox_w, view.bbox_h],
                    "choice": None if decision is None else decision.choice,
                }
            )
        return result

    def proposal_payload(self, proposal_id: str) -> dict[str, object]:
        with self._mutex:
            _require(proposal_id in self.index.proposals, "unknown proposal ID")
            view = self.index.proposals[proposal_id]
            tile = self.index.tiles[view.tile_index]
            decision = self._decisions.get(proposal_id)
            proposal = {
                    **asdict(view),
                    "decision": _decision_payload(decision),
                    "position": self.index.order.index(proposal_id) + 1,
                    "total": len(self.index.order),
                }
            if self.model_context is not None:
                score = self.model_context.score(proposal_id)
                proposal.update(
                    {
                        "xgb_p": score.probability,
                        "prediction": score.prediction,
                        "uncertainty": score.uncertainty,
                        "selection_rank": self.index.order.index(proposal_id) + 1,
                        "model_preset": self.model_context.preset_id,
                    }
                )
            return {
                "proposal": proposal,
                "tile": {
                    "index": tile.index,
                    "tile_name": tile.tile_name,
                    "proposals": self._tile_proposals(tile),
                },
                "summary": self.summary(),
            }

    def navigate(
        self,
        current: str | None,
        direction: int,
        *,
        state_filter: str = "all",
        group: str | None = None,
    ) -> dict[str, object]:
        with self._mutex:
            _require(direction in {-1, 1}, "navigation direction is invalid")
            _require(state_filter in {"all", "pending", "reviewed"}, "navigation filter is invalid")
            if group is not None:
                _require(group in self.index.groups, "navigation group is invalid")
            candidates = [
                proposal_id
                for proposal_id in self.index.order
                if (group is None or self.index.proposals[proposal_id].group_id == group)
                and (state_filter == "all" or (proposal_id in self._decisions) == (state_filter == "reviewed"))
            ]
            _require(bool(candidates), "no proposals match the current filter")
            if current not in candidates:
                selected = candidates[0] if direction > 0 else candidates[-1]
            else:
                selected = candidates[(candidates.index(str(current)) + direction) % len(candidates)]
            return self.proposal_payload(selected)

    def search(self, prefix: str) -> dict[str, object]:
        value = prefix.strip().lower()
        _require(re.fullmatch(r"[0-9a-f]{4,64}", value) is not None, "search requires 4-64 hexadecimal proposal-ID characters")
        matches = [proposal_id for proposal_id in self.index.order if proposal_id.startswith(value)]
        _require(len(matches) == 1, "proposal-ID search must match exactly one proposal")
        return self.proposal_payload(matches[0])

    def overlay_bytes(self, tile_index: int) -> bytes:
        with self._mutex:
            _require(0 <= tile_index < len(self.index.tiles), "unknown review tile")
            tile = self.index.tiles[tile_index]
            payload, _info = stable_file(tile.overlay_path, max_bytes=MAX_OVERLAY_BYTES)
            _require(hashlib.sha256(payload).hexdigest() == tile.overlay_sha256, "sealed review overlay changed")
            _require(payload.startswith(b"\x89PNG\r\n\x1a\n"), "review overlay is not PNG")
            return payload

    def _full_image_scene_catalog(self) -> Any:
        if self._scene_catalog is None:
            from compag_curation.review.scene import Stage20SceneCatalog

            self._scene_catalog = Stage20SceneCatalog.open(
                self.snapshot.root,
                stage10_receipt_sha256=self.snapshot.stage10_receipt_sha256,
                stage20_receipt_sha256=self.snapshot.stage20_receipt_sha256,
                proposals_sha256=self.index.proposals_sha256,
                overlay_manifest_sha256=self.index.overlay_manifest_sha256,
                proposal_views=self.index.proposals,
                proposal_order=self.index.order,
                source_polygons=self.index.source_polygons,
            )
        return self._scene_catalog

    def scene_payload(self, proposal_id: str) -> dict[str, object]:
        """Return one selected proposal in its verified full-card scene."""

        with self._mutex:
            payload = self.proposal_payload(proposal_id)
            choices = {
                item_id: (
                    None
                    if self._decisions.get(item_id) is None
                    else self._decisions[item_id].choice
                )
                for item_id in self.index.order
            }
            payload["scene"] = self._full_image_scene_catalog().payload(
                proposal_id,
                choices,
                None
                if self.model_context is None
                else self.model_context.scene_records(),
            )
            return payload

    def scene_bytes(self, scene_id: str) -> bytes:
        """Render one verified Stage-10 full-card mosaic as deterministic PNG."""

        with self._mutex:
            return self._full_image_scene_catalog().bytes(scene_id)

    def _render_csv(
        self,
        decisions: Mapping[str, Decision] | None = None,
    ) -> bytes:
        durable_decisions = self._decisions if decisions is None else decisions
        _require(
            len(durable_decisions) == len(self.index.request_rows),
            "review is incomplete",
        )
        with io.StringIO(newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.index.request_columns, lineterminator="\n")
            writer.writeheader()
            for source in self.index.request_rows:
                decision = durable_decisions[source["proposal_id"]]
                row = {name: source[name] for name in self.index.request_columns}
                row["label"] = str(decision.label)
                row["review_status"] = "reviewed"
                if self.canonical:
                    _require(
                        decision.review_action in REVIEW_ACTION_WEIGHTS
                        and decision.review_weight == REVIEW_ACTION_WEIGHTS[decision.review_action],
                        "review decision action/weight changed",
                    )
                    row["review_action"] = str(decision.review_action)
                    row["review_weight"] = format(float(decision.review_weight), ".1f")
                writer.writerow(row)
            return handle.getvalue().encode("utf-8")

    def finalize(self) -> dict[str, object]:
        with self._mutex:
            _require(self._lock_fd >= 0, "review session is closed")
            if self._final is not None:
                return dict(self._final)
            durable_decisions = self._verify_durable_events()
            final_path = self.state_root / "FINAL.json"
            if (
                self.output.exists()
                or self.output.is_symlink()
                or final_path.exists()
                or final_path.is_symlink()
            ):
                self._load_final()
                _require(self._final is not None, "review publication recovery failed")
                return dict(self._final)
            _require(
                len(durable_decisions) == len(self.index.order),
                "review cannot be finalized while proposals remain pending",
            )
            payload = self._render_csv(durable_decisions)
            staged = self.output.parent / f".{self.output.name}.compag-review.{uuid.uuid4()}.tmp"
            write_new_bytes(staged, payload, mode=0o600)
            try:
                rows, validation = validate_review_table(self.snapshot.review_request, staged)
                feasibility = self._split_feasibility(rows)
                # Validation may be expensive.  Re-read the independent
                # durable journal immediately before publishing its CSV
                # projection rather than trusting the earlier in-memory copy.
                self._verify_durable_events()
                publish_file_noreplace(staged, self.output)
            except BaseException:
                if staged.exists() and not staged.is_symlink():
                    staged.unlink()
                    fsync_directory(staged.parent)
                raise
            final = self._final_record(validation, feasibility, recovery=False)
            try:
                write_new_json(final_path, final, mode=0o600)
            except BaseException:
                # The no-clobber write may have succeeded before a durability
                # confirmation failed.  Mutations are fail-closed by the
                # presence of either publication; a reopen performs recovery.
                try:
                    self._load_final()
                except BaseException:
                    pass
                raise
            self._final = final
            return dict(final)


__all__ = [
    "DECISIONS",
    "Decision",
    "ProposalView",
    "ReviewIndex",
    "ReviewSession",
    "TileView",
    "build_review_index",
]
