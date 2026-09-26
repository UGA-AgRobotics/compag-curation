#!/usr/bin/env python3
"""Generate the deterministic, public-safe onboarding fixture without overwrite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("short fixture write")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)


def _ppm(seed: int) -> bytes:
    width = height = 64
    rows = [f"P3\n{width} {height}\n255\n"]
    for y in range(height):
        values = []
        for x in range(width):
            grid = 35 if x % 16 == 0 or y % 16 == 0 else 0
            spot = 1 if (x - (16 + seed * 3) % 40) ** 2 + (y - (20 + seed * 5) % 36) ** 2 <= 25 else 0
            values.extend((str(220 - grid - 120 * spot), str(205 - grid - 145 * spot), str(45 - min(35, grid) + 25 * spot)))
        rows.append(" ".join(values) + "\n")
    return "".join(rows).encode("ascii")


def _review_csv(*, reviewed: bool) -> bytes:
    lines = ["proposal_id,proposal_sha256,image_id,image_sha256,group_id,label,review_status\n"]
    for index in range(1, 9):
        label = str((index + 1) % 2) if reviewed else ""
        status = "reviewed" if reviewed else "pending"
        proposal_id = f"{index:064x}"
        lines.append(
            f"{proposal_id},{proposal_id},"
            f"{0x2000000000000000000000000000000000000000000000000000000000000000 + index:064x},"
            f"{0x3000000000000000000000000000000000000000000000000000000000000000 + index:064x},"
            f"synthetic-group-{(index + 1) // 2:02d},{label},{status}\n"
        )
    return "".join(lines).encode("ascii")


def _canonical_review_csv(*, reviewed: bool) -> bytes:
    lines = [
        "proposal_id,proposal_sha256,image_id,image_sha256,group_id,label,"
        "review_action,review_weight,review_status\n"
    ]
    actions = (("accept", "1.0"), ("sus_accept", "0.4"))
    for index, (action, weight) in enumerate(actions, start=1):
        proposal_id = f"{index:064x}"
        label = str(index - 1) if reviewed else ""
        status = "reviewed" if reviewed else "pending"
        selected_action = action if reviewed else ""
        selected_weight = weight if reviewed else ""
        lines.append(
            f"{proposal_id},{proposal_id},"
            f"{0x2000000000000000000000000000000000000000000000000000000000000000 + index:064x},"
            f"{0x3000000000000000000000000000000000000000000000000000000000000000 + index:064x},"
            f"synthetic-group-01,{label},{selected_action},{selected_weight},{status}\n"
        )
    return "".join(lines).encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise SystemExit("fixture output already exists")
    output.mkdir(mode=0o755)
    members = []
    for index in range(6):
        relative = Path("data/images") / f"synthetic-group-{index + 1:02d}__train.ppm"
        payload = _ppm(index)
        _write_new(output / relative, payload)
        members.append({"path": relative.as_posix(), "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)})
    heldout = Path("data/inference_images/synthetic-heldout__card.ppm")
    payload = _ppm(19)
    _write_new(output / heldout, payload)
    members.append({"path": heldout.as_posix(), "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)})
    xml = b'''<?xml version="1.0" encoding="UTF-8"?>\n<annotations>\n  <image id="0" name="synthetic-heldout__card.ppm" width="64" height="64">\n    <points label="cj" points="17,23"/>\n    <points label="cj" points="45,44"/>\n  </image>\n</annotations>\n'''
    annotation = Path("annotations/points.xml")
    _write_new(output / annotation, xml)
    members.append({"path": annotation.as_posix(), "sha256": hashlib.sha256(xml).hexdigest(), "size_bytes": len(xml)})
    for relative, reviewed in (
        (Path("review_request_example.csv"), False),
        (Path("reviewed_example.csv"), True),
    ):
        payload = _review_csv(reviewed=reviewed)
        _write_new(output / relative, payload)
        members.append({"path": relative.as_posix(), "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)})
    for relative, reviewed in (
        (Path("canonical_review_request_example.csv"), False),
        (Path("canonical_reviewed_example.csv"), True),
    ):
        payload = _canonical_review_csv(reviewed=reviewed)
        _write_new(output / relative, payload)
        members.append({"path": relative.as_posix(), "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)})
    members.sort(key=lambda row: row["path"])
    manifest = {
        "schema": "compag-curation-public-example-fixture/v1",
        "status": "PASS",
        "source": "DETERMINISTIC_FROM_SCRATCH_SYNTHETIC",
        "research_data_derivative": False,
        "license": "CC-BY-4.0",
        "members": members,
    }
    rendered = (json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
    _write_new(output / "FIXTURE_MANIFEST.json", rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
