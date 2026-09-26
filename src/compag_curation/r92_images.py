"""Prepare every canonical tile for a fixed-r92 full public round."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .r92_sample import _payloads


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def init_images(images: Path, model: Path, output: Path) -> dict:
    """Preserve all selected originals and all far-edge-aligned warped-card tiles.

    This stage creates honest empty-label registries. It does not run SAM2,
    extract features, score candidates or invent human labels.
    """
    import cv2
    import numpy as np
    from .canonical.preprocessing import canonical_prepare_images, encode_canonical_jpeg

    images = Path(images).resolve(strict=True)
    model = Path(model).resolve(strict=True)
    output = Path(output).absolute()
    if not images.is_dir():
        raise ValueError("images must be a directory")
    if images == output or images in output.parents:
        raise ValueError("project output must be outside the image source")
    if output.exists():
        raise FileExistsError("image project output already exists")
    _payloads(model)
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files or len(files) > 20 or any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError("select 1–20 regular JPG/PNG images")
    stems = [p.stem for p in files]
    if len(set(stems)) != len(stems):
        raise ValueError("image stems must be unique")
    categories = [{"id": 1, "name": "CJ", "supercategory": ""},
                  {"id": 2, "name": "non-CJ", "supercategory": ""}]
    original = {"images": [], "annotations": [], "categories": categories}
    tiled = {"images": [], "annotations": [], "categories": categories}
    inventory = []
    hashes: dict[str, str] = {}
    output.mkdir(parents=True)
    try:
        (output / "originals").mkdir()
        (output / "tiles").mkdir()
        for original_id, src in enumerate(files, 1):
            raw = src.read_bytes()
            pixels = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if pixels is None:
                raise ValueError(f"unreadable original image: {src.name}")
            height, width = pixels.shape[:2]
            if min(height, width) < 512:
                raise ValueError(f"image is smaller than 512x512: {src.name}")
            prepared = canonical_prepare_images(pixels, src.name)
            warped_height, warped_width = prepared.warped_bgr.shape[:2]
            if min(warped_height, warped_width) < 512:
                raise ValueError(f"canonical warped card is smaller than 512x512: {src.name}")
            dst = output / "originals" / src.name
            dst.write_bytes(raw)
            original_sha = _sha(src)
            if _sha(dst) != original_sha:
                raise RuntimeError("original byte preservation failed")
            hashes[f"originals/{src.name}"] = original_sha
            original["images"].append({"id": original_id, "file_name": src.name,
                                       "width": width, "height": height})
            tile_inventory = []
            for prepared_tile in prepared.tiles:
                tile_path = output / "tiles" / prepared_tile.name
                tile_path.write_bytes(encode_canonical_jpeg(prepared_tile))
                decoded = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
                if decoded is None or decoded.shape != (512, 512, 3):
                    raise ValueError(f"canonical tile cannot be decoded: {prepared_tile.name}")
                tile_sha = _sha(tile_path)
                hashes[f"tiles/{prepared_tile.name}"] = tile_sha
                tiled["images"].append({
                    "id": len(tiled["images"]) + 1, "file_name": prepared_tile.name,
                    "width": 512, "height": 512,
                    "meta": {"orig_file_name": src.name, "orig_image_id": original_id,
                             "offset_x": prepared_tile.x, "offset_y": prepared_tile.y,
                             "warped_width": warped_width, "warped_height": warped_height,
                             "warp_mode": prepared.warp_mode,
                             "row_lines": list(prepared.row_lines),
                             "column_lines": list(prepared.column_lines),
                             "inverse_matrix": prepared.evidence["warp"]["inverse_matrix"]}})
                tile_inventory.append({"name": prepared_tile.name, "sha256": tile_sha,
                                       "offset_x": prepared_tile.x, "offset_y": prepared_tile.y})
            inventory.append({"original": src.name, "original_bytes": len(raw),
                              "original_sha256": original_sha, "width": width, "height": height,
                              "warp_mode": prepared.warp_mode, "warped_width": warped_width,
                              "warped_height": warped_height,
                              "tile_count": len(tile_inventory), "tiles": tile_inventory})
        for name, data in (("original_coco.json", original), ("tiled_coco.json", tiled)):
            path = output / name
            path.write_text(json.dumps(data, indent=2) + "\n")
            hashes[name] = _sha(path)
        manifest = {"schema": "compag-r92-full-project-manifest/v1",
                    "sha256_by_relative_path": dict(sorted(hashes.items())),
                    "image_count": len(inventory), "tile_count": len(tiled["images"])}
        manifest_path = output / "PROJECT_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        receipt = {"schema": "compag-r92-image-project/v2", "status": "PREPARED_ALL_TILES",
                   "model_archive_sha256": _sha(model),
                   "scope": "every canonical 512x512 far-edge tile of every warped card",
                   "images": inventory, "image_count": len(inventory),
                   "tile_count": len(tiled["images"]),
                   "project_manifest_sha256": _sha(manifest_path),
                   "annotations": 0, "masks": 0, "feature_rows": 0,
                   "scores": 0, "live_inference": "NOT_RUN"}
        (output / "PROJECT_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    except Exception:
        shutil.rmtree(output)
        raise
