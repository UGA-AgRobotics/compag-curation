"""Port of notebook cell 3 "CELL 0 — Train-only splits (FREEZE+APPEND)".

New original-image ids produced by the merge are appended to TRAIN only;
existing VAL/TEST lists are frozen (the new ids are removed from them).  The
append file is marked consumed by renaming it to ``*.applied.json`` -- in the
application this happens on the project copy, never on the original.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _save_lists(split_dir: Path, train_ids, val_ids, test_ids) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "orig_train_ids.json").write_text(json.dumps(sorted([int(x) for x in train_ids]), indent=2), encoding="utf-8")
    (split_dir / "orig_val_ids.json").write_text(json.dumps(sorted([int(x) for x in val_ids]), indent=2), encoding="utf-8")
    (split_dir / "orig_test_ids.json").write_text(json.dumps(sorted([int(x) for x in test_ids]), indent=2), encoding="utf-8")


def _load_existing(split_dir: Path):
    t = split_dir / "orig_train_ids.json"
    v = split_dir / "orig_val_ids.json"
    s = split_dir / "orig_test_ids.json"
    if t.exists() and v.exists() and s.exists():
        return (set(map(int, json.loads(t.read_text()))), set(map(int, json.loads(v.read_text()))),
                set(map(int, json.loads(s.read_text()))))
    return None


def _read_new_ids(append_file: Path) -> set[int]:
    if append_file is not None and append_file.exists():
        try:
            return set(map(int, json.loads(append_file.read_text(encoding="utf-8"))))
        except Exception:
            return set()
    return set()


def _consume(append_file: Path) -> str:
    try:
        append_file.rename(append_file.with_suffix(".applied.json"))
        return "renamed_to_applied"
    except Exception:
        try:
            append_file.write_text("[]", encoding="utf-8")
            return "emptied"
        except Exception:
            return "unchanged"


def update_splits(*, orig_coco_json: Path, split_dir: Path, append_new_orig_ids: Path,
                  freeze_old_splits: bool = True) -> dict[str, Any]:
    """Operate on the project's split directory (a copy of the historical one)."""

    data = json.loads(Path(orig_coco_json).read_text(encoding="utf-8"))
    images = data.get("images", [])
    orig_ids = np.array([int(im["id"]) for im in images], dtype=int)
    orig_id_set = set(map(int, orig_ids.tolist()))
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    append_file = Path(append_new_orig_ids)

    existing = _load_existing(split_dir)
    if freeze_old_splits and existing is not None:
        old_train, old_val, old_test = existing
        new_ids = _read_new_ids(append_file)
        if new_ids:
            diff = new_ids - orig_id_set
            if diff:
                raise ValueError(f"These IDs are not in ORIG_COCO_JSON: {sorted(diff)}")
            old_val = old_val - new_ids
            old_test = old_test - new_ids
            train_ids = sorted(old_train | new_ids)
            _save_lists(split_dir, train_ids, sorted(old_val), sorted(old_test))
            consumed = _consume(append_file)
            return {"mode": "APPEND_TRAIN_ONLY", "new_ids": sorted(new_ids), "train": len(train_ids),
                    "val": len(old_val), "test": len(old_test), "append_file": consumed}
        return {"mode": "SAFE_NO_NEW_IDS", "train": len(old_train), "val": len(old_val), "test": len(old_test)}

    new_ids = _read_new_ids(append_file)
    if new_ids:
        diff = new_ids - orig_id_set
        if diff:
            raise ValueError(f"These IDs are not in ORIG_COCO_JSON: {sorted(diff)}")
    train_ids = sorted(set(orig_ids) | new_ids)
    _save_lists(split_dir, train_ids, [], [])
    consumed = "none"
    if new_ids and append_file.exists():
        consumed = _consume(append_file)
    return {"mode": "FRESH_TRAIN_ONLY", "train": len(train_ids), "val": 0, "test": 0, "append_file": consumed}
