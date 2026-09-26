"""Strict, inert CVAT point-annotation inspection shared by validation and evaluation."""

from __future__ import annotations

import hashlib
import math
import re
import stat
import xml.etree.ElementTree as element_tree
from pathlib import Path
from typing import Mapping

from .public_config import MAX_IMAGE_DIMENSION
from .public_io import PublicIOError, canonical_json_value, portable_basename, stable_file


MAX_ANNOTATION_BYTES = 64 * 1024 * 1024
MAX_ANNOTATION_ROWS = 250_000
CANONICAL_SCHEMA = "compag-curation-canonical-point-annotations/v1"
_CANONICAL_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


def _canonical_basename(value: str, role: str) -> str:
    portable_basename(value, role)
    _require(
        _CANONICAL_BASENAME.fullmatch(value) is not None,
        f"{role} does not match the canonical annotation name alphabet",
    )
    return value


def inspect_cvat_points(
    path: Path,
    *,
    label: str = "cj",
    expected_images: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    path = path.absolute()
    _require(path == path.resolve(strict=True), "CVAT annotation path contains a symlink component")
    _canonical_basename(path.name, "CVAT annotation source filename")
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), "CVAT annotation input is missing or unsafe")
    payload, snapshot = stable_file(path, max_bytes=MAX_ANNOTATION_BYTES)
    _require(snapshot.st_size == info.st_size and snapshot.st_ino == info.st_ino, "CVAT annotation changed before inspection")
    _require(not payload.startswith((b"\xff\xfe", b"\xfe\xff")), "CVAT annotation XML must be UTF-8")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise PublicIOError("CVAT annotation XML must be UTF-8") from exc
    _require("\x00" not in text, "CVAT annotation XML contains a NUL byte")
    lowered = text.casefold()
    _require("<!doctype" not in lowered and "<!entity" not in lowered, "CVAT DTD/entity declarations are forbidden")
    declaration = re.match(r"\s*<\?xml\s+[^?]*encoding\s*=\s*['\"]([^'\"]+)['\"]", text, flags=re.IGNORECASE)
    _require(declaration is None or declaration.group(1).casefold() in {"utf-8", "utf8"}, "CVAT XML declaration must specify UTF-8")
    try:
        root = element_tree.fromstring(text)
    except element_tree.ParseError as exc:
        raise PublicIOError("CVAT annotation XML is invalid") from exc
    _require(root.tag == "annotations", "CVAT root element must be annotations")
    rows: list[dict[str, object]] = []
    declared_images: set[str] = set()
    for image in root.findall(".//image"):
        image_name = image.attrib.get("name", "")
        _canonical_basename(image_name, "CVAT image name")
        _require(image_name not in declared_images, f"duplicate CVAT image declaration: {image_name}")
        declared_images.add(image_name)
        try:
            width = int(image.attrib.get("width", ""))
            height = int(image.attrib.get("height", ""))
        except ValueError as exc:
            raise PublicIOError(f"CVAT image dimensions are invalid: {image_name}") from exc
        _require(0 < width <= MAX_IMAGE_DIMENSION and 0 < height <= MAX_IMAGE_DIMENSION, f"CVAT image dimensions are out of bounds: {image_name}")
        if expected_images is not None and image_name in expected_images:
            _require(expected_images[image_name] == (width, height), f"CVAT dimensions differ from the configured image: {image_name}")
        for point in image.findall("point") + image.findall("points"):
            if point.attrib.get("label") != label:
                continue
            rendered = point.attrib.get("points", "")
            coordinate_rows = rendered.split(";")
            _require(coordinate_rows and all(value != "" for value in coordinate_rows), f"CVAT point coordinate is invalid: {image_name}")
            for coordinate_row in coordinate_rows:
                coordinates = coordinate_row.split(",")
                _require(len(coordinates) == 2, f"CVAT point coordinate is invalid: {image_name}")
                try:
                    x, y = float(coordinates[0]), float(coordinates[1])
                except ValueError as exc:
                    raise PublicIOError(f"CVAT point coordinate is not numeric: {image_name}") from exc
                _require(math.isfinite(x) and math.isfinite(y), f"CVAT point coordinate is nonfinite: {image_name}")
                _require(0.0 <= x < width and 0.0 <= y < height, f"CVAT point lies outside the declared image: {image_name}")
                _require(len(rows) < MAX_ANNOTATION_ROWS, "CVAT annotations exceed the supported point-row bound")
                rows.append({"image": image_name, "label": label, "x": x, "y": y, "width": width, "height": height})
    _require(rows, f"CVAT annotation XML contains no {label} point rows")
    rows.sort(key=lambda row: (str(row["image"]), float(row["y"]), float(row["x"])))
    return rows, {
        "schema": "compag-curation-cvat-point-inspection/v1",
        "status": "PASS",
        "source_filename": path.name,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "label": label,
        "declared_images": sorted(declared_images),
        "point_rows": len(rows),
    }


def inspect_point_annotations(
    path: Path,
    *,
    label: str = "cj",
    expected_images: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Inspect either CVAT XML or the canonical JSON emitted by the converter."""

    if path.suffix.lower() == ".xml":
        return inspect_cvat_points(path, label=label, expected_images=expected_images)
    _require(path.suffix.lower() == ".json", "point annotations must be CVAT XML or canonical JSON")
    path = path.absolute()
    _require(path == path.resolve(strict=True), "canonical point annotation path contains a symlink component")
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), "canonical point annotation input is missing or unsafe")
    payload, snapshot = stable_file(path, max_bytes=MAX_ANNOTATION_BYTES)
    _require(snapshot.st_size == info.st_size and snapshot.st_ino == info.st_ino, "canonical point annotations changed before inspection")
    try:
        value = canonical_json_value(payload, "canonical point annotation JSON")
    except PublicIOError as exc:
        raise PublicIOError("canonical point annotation JSON is invalid") from exc
    _require(isinstance(value, dict), "canonical point annotations must be an object")
    _require(set(value) == {"schema", "status", "source_filename", "source_sha256", "rows"}, "canonical point annotation fields mismatch")
    _require(value.get("schema") == CANONICAL_SCHEMA and value.get("status") == "PASS", "canonical point annotation schema/status mismatch")
    source_filename = value.get("source_filename")
    source_sha256 = value.get("source_sha256")
    _require(isinstance(source_filename, str), "canonical annotation source name must be text")
    _require(isinstance(source_sha256, str), "canonical annotation source hash must be text")
    _canonical_basename(source_filename, "canonical annotation source name")
    _require(re.fullmatch(r"[0-9a-f]{64}", source_sha256) is not None, "canonical annotation source hash is invalid")
    raw_rows = value.get("rows")
    _require(
        isinstance(raw_rows, list) and raw_rows and len(raw_rows) <= MAX_ANNOTATION_ROWS,
        "canonical point annotation rows are empty or exceed the supported bound",
    )
    rows: list[dict[str, object]] = []
    declared_images: set[str] = set()
    for index, raw in enumerate(raw_rows, start=1):
        _require(isinstance(raw, dict) and set(raw) == {"image", "label", "x", "y", "width", "height"}, f"canonical point row {index} fields mismatch")
        _require(isinstance(raw["image"], str), f"canonical point row {index} image must be text")
        image_name = raw["image"]
        _canonical_basename(image_name, "canonical point image name")
        _require(raw["label"] == label, f"canonical point row {index} has an unexpected label")
        width = raw["width"]
        height = raw["height"]
        _require(isinstance(width, int) and not isinstance(width, bool) and 0 < width <= MAX_IMAGE_DIMENSION, f"canonical point row {index} width is invalid")
        _require(isinstance(height, int) and not isinstance(height, bool) and 0 < height <= MAX_IMAGE_DIMENSION, f"canonical point row {index} height is invalid")
        x = raw["x"]
        y = raw["y"]
        _require(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)), f"canonical point row {index} x is invalid")
        _require(isinstance(y, (int, float)) and not isinstance(y, bool) and math.isfinite(float(y)), f"canonical point row {index} y is invalid")
        _require(0.0 <= float(x) < width and 0.0 <= float(y) < height, f"canonical point row {index} lies outside its image")
        if expected_images is not None and image_name in expected_images:
            _require(expected_images[image_name] == (width, height), f"canonical point dimensions differ from the configured image: {image_name}")
        declared_images.add(image_name)
        rows.append({"image": image_name, "label": label, "x": float(x), "y": float(y), "width": width, "height": height})
    rows.sort(key=lambda row: (str(row["image"]), float(row["y"]), float(row["x"])))
    return rows, {
        "schema": "compag-curation-canonical-point-inspection/v1",
        "status": "PASS",
        "source_filename": path.name,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "label": label,
        "declared_images": sorted(declared_images),
        "point_rows": len(rows),
        "canonical_source_sha256": value["source_sha256"],
    }
