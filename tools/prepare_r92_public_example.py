#!/usr/bin/env python3
"""Verify public r92 Release assets and stage all ten original cards."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image

MODEL_NAME = "COMPAG_CJ_R92_INFERENCE_SAMPLE_20260924T200000Z.zip"
MODEL_SHA256 = "fffef7a85192b99dcca3bcc5750215150d2ca9659121363198041244bbfe57c1"
PHOTO_ROOT = "COMPAG_R92_TEN_PHOTO_INPUTS_RC6_R3_20260924"
PHOTO_NAME = PHOTO_ROOT + ".zip"
PHOTO_SHA256 = "344c8d767a1719403a3713e0f51587971f4ab9ea18e1d1a44034ed72c9297914"
MANIFEST = Path(__file__).resolve().parent.parent / "PUBLIC_PHOTO_MANIFEST.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_file(path: Path, name: str, digest: str) -> bytes:
    if not path.is_file() or path.is_symlink() or path.name != name:
        raise ValueError(f"expected regular Release asset named {name}")
    data = path.read_bytes()
    if sha256(data) != digest:
        raise ValueError(f"Release asset hash mismatch: {name}")
    return data


def _safe_members(archive: zipfile.ZipFile, expected: set[str]) -> None:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("photo ZIP member set or duplicate identity mismatch")
    for info in infos:
        parts = PurePosixPath(info.filename).parts
        if info.is_dir() or info.filename.startswith("/") or ".." in parts or "\\" in info.filename:
            raise ValueError("unsafe photo ZIP path")
        mode = info.external_attr >> 16
        if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
            raise ValueError("non-file member in photo ZIP")
    if archive.testzip() is not None:
        raise ValueError("photo ZIP CRC failure")


def prepare(model_zip: Path, photo_zip: Path, output: Path) -> dict[str, object]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    model_bytes = require_file(model_zip, MODEL_NAME, MODEL_SHA256)
    photo_bytes = require_file(photo_zip, PHOTO_NAME, PHOTO_SHA256)
    with zipfile.ZipFile(io.BytesIO(model_bytes)) as archive:
        if archive.testzip() is not None:
            raise ValueError("model ZIP CRC failure")
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    rows = manifest["photos"]
    if manifest["photo_count"] != 10 or len(rows) != 10:
        raise ValueError("expected exactly ten public photos")
    names = [row["original_filename"] for row in rows]
    digests = [row["sha256"] for row in rows]
    if len(set(names)) != 10 or len(set(digests)) != 10:
        raise ValueError("duplicate photo name or bytes in manifest")
    expected = {f"{PHOTO_ROOT}/{row['relative_public_path']}" for row in rows}
    expected |= {f"{PHOTO_ROOT}/PUBLIC_PHOTO_MANIFEST.json", f"{PHOTO_ROOT}/README_TEN_PHOTO_EN.md"}
    photos: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(photo_bytes)) as archive:
        _safe_members(archive, expected)
        if archive.read(f"{PHOTO_ROOT}/PUBLIC_PHOTO_MANIFEST.json") != manifest_bytes:
            raise ValueError("photo ZIP manifest differs from public source")
        for row in rows:
            name = row["original_filename"]
            if row["relative_public_path"] != f"originals/{name}" or name != f"{row['source_card_identity']}.jpg":
                raise ValueError(f"photo path or card identity mismatch: {name}")
            data = archive.read(f"{PHOTO_ROOT}/originals/{name}")
            if sha256(data) != row["sha256"] or len(data) != row["byte_size"]:
                raise ValueError(f"photo hash or size mismatch: {name}")
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                if image.format != "JPEG" or image.size != (row["decoded_width"], row["decoded_height"]) or image.getexif().get(274, 1) != 1:
                    raise ValueError(f"photo decode or orientation mismatch: {name}")
            photos[name] = data
    output.mkdir(parents=True)
    (output / "model").mkdir()
    (output / "cards").mkdir()
    (output / "model" / MODEL_NAME).write_bytes(model_bytes)
    card_rows = []
    for row in rows:
        name = row["original_filename"]
        card_dir = output / "cards" / row["source_card_identity"]
        card_dir.mkdir()
        (card_dir / name).write_bytes(photos[name])
        card_rows.append({"card_id": row["source_card_identity"], "relative_directory": f"cards/{row['source_card_identity']}", "filename": name, "sha256": row["sha256"]})
    receipt: dict[str, object] = {
        "schema": "compag-r92-public-example-inputs/v2",
        "model": str(output / "model" / MODEL_NAME),
        "model_sha256": MODEL_SHA256,
        "photo_archive_sha256": PHOTO_SHA256,
        "photo_count": len(card_rows),
        "cards": card_rows,
    }
    (output / "EXAMPLE_INPUTS.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-zip", required=True, type=Path)
    parser.add_argument("--photos-zip", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_zip, args.photos_zip, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
