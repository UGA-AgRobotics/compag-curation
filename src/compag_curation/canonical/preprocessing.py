"""Canonical card warp, dashed-grid localisation, and far-edge tiling."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from compag_curation.public_io import PublicIOError, publish_directory_noreplace, write_new_bytes

from .spec import (
    CANONICAL_JPEG_QUALITY,
    CANONICAL_TILE_EXTENSION,
    CANONICAL_TILE_SIZE,
    CANONICAL_TILE_STRIDE,
)


_PORTABLE_STEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class CanonicalTile:
    name: str
    x: int
    y: int
    crop_width: int
    crop_height: int
    pixels: Any


@dataclass(frozen=True)
class CanonicalPreparedImage:
    warped_bgr: Any
    inverse_warp: Any
    warp_mode: str
    row_lines: tuple[int, ...]
    column_lines: tuple[int, ...]
    tiles: tuple[CanonicalTile, ...]
    evidence: dict[str, Any]


FULL_IMAGE_UNIT_KIND = "full_warped_frame"
FULL_IMAGE_UNIT_EXTENSION = ".png"


def full_image_unit_name(image_name: str) -> str:
    """Return the portable name of a lossless, non-tiled analysis unit."""

    if Path(image_name).name != image_name:
        raise ValueError("full-image source name must be a basename")
    stem = Path(image_name).stem
    if not _PORTABLE_STEM.fullmatch(stem):
        raise ValueError("full-image source stem is not portable ASCII")
    return f"{stem}__full{FULL_IMAGE_UNIT_EXTENSION}"


def canonical_axis_positions(
    length: int,
    *,
    tile_size: int = CANONICAL_TILE_SIZE,
    stride: int = CANONICAL_TILE_STRIDE,
) -> tuple[int, ...]:
    """Return main-grid positions plus a final far-edge-aligned position."""

    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("image-axis length must be a positive integer")
    if tile_size != 512 or stride != 512:
        raise ValueError("canonical tiling is fixed at tile=512 and stride=512")
    far_edge = max(0, length - tile_size)
    positions = list(range(0, far_edge + 1, stride))
    if not positions:
        positions = [0]
    if positions[-1] != far_edge:
        positions.append(far_edge)
    return tuple(positions)


def canonical_tile_name(image_name: str, x: int, y: int) -> str:
    if Path(image_name).name != image_name:
        raise ValueError("canonical image name must be a basename")
    stem = Path(image_name).stem
    if not _PORTABLE_STEM.fullmatch(stem):
        raise ValueError("canonical image stem is not portable ASCII")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (x, y)):
        raise ValueError("tile coordinates must be nonnegative integers")
    if x > 99999 or y > 99999:
        raise ValueError("tile coordinates exceed the five-digit naming contract")
    return f"{stem}_y{y:05d}x{x:05d}{CANONICAL_TILE_EXTENSION}"


def iter_canonical_tiles(warped_bgr: Any, image_name: str) -> tuple[CanonicalTile, ...]:
    """Tile a warped image without discarding pixels or changing its input."""

    import numpy as np

    image = np.asarray(warped_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("canonical tiling requires a nonempty uint8 BGR image")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("canonical tiling requires a nonempty image")
    tiles: list[CanonicalTile] = []
    for y in canonical_axis_positions(height):
        for x in canonical_axis_positions(width):
            crop = image[y : min(y + 512, height), x : min(x + 512, width)].copy()
            crop_height, crop_width = crop.shape[:2]
            if crop_height < 512 or crop_width < 512:
                crop = np.pad(
                    crop,
                    ((0, 512 - crop_height), (0, 512 - crop_width), (0, 0)),
                    mode="edge",
                )
            if crop.shape != (512, 512, 3):
                raise RuntimeError("canonical tiler produced an invalid tile shape")
            tiles.append(
                CanonicalTile(
                    name=canonical_tile_name(image_name, x, y),
                    x=x,
                    y=y,
                    crop_width=crop_width,
                    crop_height=crop_height,
                    pixels=crop,
                )
            )
    return tuple(tiles)


def canonical_prepare_images(
    image_bgr: Any,
    image_name: str,
    *,
    locate_grid: bool = True,
) -> CanonicalPreparedImage:
    """Apply the recovered yellow-card warp, grid localisation, then tiling.

    ``gate_core.warp_card_to_rect`` owns the HSV bounds, polygon epsilon, and
    identity fallback.  This wrapper records which path was observed and does
    not reimplement the scientific image operations.
    """

    import numpy as np
    from compag_curation.proposals.sam2_pipeline.gate_core import (
        detect_grid_lines,
        warp_card_to_rect,
    )

    source = np.asarray(image_bgr)
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ValueError("canonical preparation requires a uint8 BGR image")
    warped, inverse = warp_card_to_rect(source)
    inverse_array = np.asarray(inverse, dtype=np.float32)
    if inverse_array.shape != (3, 3):
        raise RuntimeError("card warp returned an invalid inverse transform")
    identity = np.eye(3, dtype=np.float32)
    warp_mode = (
        "identity_fallback"
        if warped.shape == source.shape and np.allclose(inverse_array, identity, rtol=0.0, atol=1e-6)
        else "yellow_card_quadrilateral"
    )
    if locate_grid:
        rows, columns = detect_grid_lines(warped)
    else:
        rows, columns = [], []
    row_lines = tuple(sorted(int(value) for value in rows))
    column_lines = tuple(sorted(int(value) for value in columns))
    tiles = iter_canonical_tiles(warped, image_name)
    evidence = {
        "schema": "compag-curation-canonical-preprocessing/v1",
        "warp": {
            "mode": warp_mode,
            "hsv_lower": [15, 40, 60],
            "hsv_upper": [40, 255, 255],
            "approx_poly_dp_epsilon_perimeter_fraction": 0.02,
            "fallback": "identity",
            "input_shape": [int(value) for value in source.shape],
            "output_shape": [int(value) for value in warped.shape],
            "inverse_matrix": [
                [float(value) for value in row] for row in inverse_array.tolist()
            ],
        },
        "grid": {
            "algorithm": "spacing_constrained_peak_search",
            "expected_horizontal_lines": 5,
            "expected_vertical_lines": 3,
            "row_lines": list(row_lines),
            "column_lines": list(column_lines),
            "requested": bool(locate_grid),
        },
        "tiling": {
            "tile_size": 512,
            "stride": 512,
            "far_edge_alignment": True,
            "small_crop_padding": "edge_bottom_right",
            "format": "jpeg",
            "jpeg_quality": CANONICAL_JPEG_QUALITY,
            "tile_count": len(tiles),
        },
    }
    return CanonicalPreparedImage(
        warped_bgr=warped,
        inverse_warp=inverse_array,
        warp_mode=warp_mode,
        row_lines=row_lines,
        column_lines=column_lines,
        tiles=tiles,
        evidence=evidence,
    )


def full_image_prepare(
    image_bgr: Any,
    image_name: str,
    *,
    locate_grid: bool = True,
) -> CanonicalPreparedImage:
    """Warp one source image and expose exactly one non-tiled analysis unit.

    The returned object deliberately reuses :class:`CanonicalPreparedImage` so
    the orchestration boundary remains small.  Its sole ``tiles`` member is an
    internal compatibility container, not a spatial tile; the evidence and
    Stage-10 profile contract identify it as ``full_warped_frame``.
    """

    import numpy as np
    from compag_curation.proposals.sam2_pipeline.gate_core import (
        detect_grid_lines,
        warp_card_to_rect,
    )

    source = np.asarray(image_bgr)
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ValueError("full-image preparation requires a uint8 BGR image")
    warped, inverse = warp_card_to_rect(source)
    warped = np.ascontiguousarray(warped, dtype=np.uint8)
    inverse_array = np.asarray(inverse, dtype=np.float32)
    if inverse_array.shape != (3, 3):
        raise RuntimeError("card warp returned an invalid inverse transform")
    identity = np.eye(3, dtype=np.float32)
    warp_mode = (
        "identity_fallback"
        if warped.shape == source.shape
        and np.allclose(inverse_array, identity, rtol=0.0, atol=1e-6)
        else "yellow_card_quadrilateral"
    )
    if locate_grid:
        rows, columns = detect_grid_lines(warped)
    else:
        rows, columns = [], []
    row_lines = tuple(sorted(int(value) for value in rows))
    column_lines = tuple(sorted(int(value) for value in columns))
    height, width = (int(value) for value in warped.shape[:2])
    if height <= 0 or width <= 0:
        raise ValueError("full-image preparation produced an empty warp")
    unit = CanonicalTile(
        name=full_image_unit_name(image_name),
        x=0,
        y=0,
        crop_width=width,
        crop_height=height,
        pixels=warped,
    )
    evidence = {
        "schema": "compag-curation-full-image-preprocessing/v1",
        "spatial_mode": "full-image-multiscale",
        "warp": {
            "mode": warp_mode,
            "hsv_lower": [15, 40, 60],
            "hsv_upper": [40, 255, 255],
            "approx_poly_dp_epsilon_perimeter_fraction": 0.02,
            "fallback": "identity",
            "input_shape": [int(value) for value in source.shape],
            "output_shape": [int(value) for value in warped.shape],
            "inverse_matrix": [
                [float(value) for value in row] for row in inverse_array.tolist()
            ],
        },
        "grid": {
            "algorithm": "spacing_constrained_peak_search",
            "expected_horizontal_lines": 5,
            "expected_vertical_lines": 3,
            "row_lines": list(row_lines),
            "column_lines": list(column_lines),
            "requested": bool(locate_grid),
        },
        "analysis_units": {
            "kind": FULL_IMAGE_UNIT_KIND,
            "count": 1,
            "external_tiling": False,
            "format": "png",
            "lossless": True,
            "width": width,
            "height": height,
        },
    }
    return CanonicalPreparedImage(
        warped_bgr=warped,
        inverse_warp=inverse_array,
        warp_mode=warp_mode,
        row_lines=row_lines,
        column_lines=column_lines,
        tiles=(unit,),
        evidence=evidence,
    )


def encode_canonical_jpeg(tile: CanonicalTile) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(
        CANONICAL_TILE_EXTENSION,
        tile.pixels,
        [cv2.IMWRITE_JPEG_QUALITY, CANONICAL_JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError(f"JPEG encoding failed for canonical tile: {tile.name}")
    return bytes(encoded)


def encode_full_image_png(unit: CanonicalTile) -> bytes:
    """Encode the sole full-image analysis unit losslessly."""

    import cv2

    ok, encoded = cv2.imencode(
        FULL_IMAGE_UNIT_EXTENSION,
        unit.pixels,
        [cv2.IMWRITE_PNG_COMPRESSION, 6],
    )
    if not ok:
        raise RuntimeError(f"PNG encoding failed for full-image unit: {unit.name}")
    return bytes(encoded)


def write_canonical_tiles(
    tiles: Iterable[CanonicalTile],
    destination: Path,
) -> tuple[dict[str, Any], ...]:
    """Atomically publish a fresh directory of deterministic JPEG tiles."""

    import os
    import shutil
    import tempfile

    destination = destination.resolve(strict=False)
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink():
        raise PublicIOError("canonical tile output parent is unsafe")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
    records: list[dict[str, Any]] = []
    try:
        for tile in tiles:
            payload = encode_canonical_jpeg(tile)
            write_new_bytes(temporary / tile.name, payload)
            records.append(
                {
                    "tile_name": tile.name,
                    "x": tile.x,
                    "y": tile.y,
                    "crop_width": tile.crop_width,
                    "crop_height": tile.crop_height,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            )
        os.chmod(temporary, 0o755)
        publish_directory_noreplace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return tuple(records)


__all__ = [
    "CanonicalPreparedImage",
    "CanonicalTile",
    "FULL_IMAGE_UNIT_EXTENSION",
    "FULL_IMAGE_UNIT_KIND",
    "canonical_axis_positions",
    "canonical_prepare_images",
    "canonical_tile_name",
    "encode_canonical_jpeg",
    "encode_full_image_png",
    "full_image_prepare",
    "full_image_unit_name",
    "iter_canonical_tiles",
    "write_canonical_tiles",
]
