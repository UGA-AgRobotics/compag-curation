"""Copy-on-write view of an imported Only_codes workspace.

The original workspace (``<legacy_root>`` = the old ``Clean`` directory) is a
read-only input.  Every notebook cell that used to rewrite files in place
(COCO masters, splits, embeddings, pack, feature table, round models) writes
into the project overlay instead, under the *same relative path*.  Reads
resolve to the newest version: overlay first, then the original.  Absolute
paths recorded by the old code are mapped with explicit prefix rules.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class PathMapper:
    """Explicit prefix mapping for absolute paths written by the old code."""

    rules: list[tuple[str, str]] = field(default_factory=list)  # (old_prefix, new_prefix)

    def map(self, value: str) -> str:
        s = str(value)
        best = None
        for old, new in self.rules:
            o = old.rstrip("/")
            if s == o or s.startswith(o + "/"):
                if best is None or len(o) > len(best[0]):
                    best = (o, new.rstrip("/"))
        if best is None:
            return s
        return best[1] + s[len(best[0]):]

    def to_dict(self) -> list[dict[str, str]]:
        return [{"old_prefix": o, "new_prefix": n} for o, n in self.rules]


class LegacyState:
    def __init__(self, original_root: Path | None, overlay_root: Path):
        self.original_root = None if original_root is None else Path(original_root)
        self.overlay_root = Path(overlay_root)

    @staticmethod
    def _rel(rel: str | PurePosixPath) -> PurePosixPath:
        p = PurePosixPath(str(rel))
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"state paths must be relative: {rel}")
        return p

    def overlay(self, rel) -> Path:
        return self.overlay_root / self._rel(rel)

    def original(self, rel) -> Path | None:
        return None if self.original_root is None else self.original_root / self._rel(rel)

    def path(self, rel) -> Path:
        ov = self.overlay(rel)
        if ov.exists():
            return ov
        orig = self.original(rel)
        if orig is not None and orig.exists():
            return orig
        return ov

    def exists(self, rel) -> bool:
        return self.path(rel).exists()

    def write_path(self, rel) -> Path:
        p = self.overlay(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def ensure_overlay_copy(self, rel) -> Path:
        """Copy the current version into the overlay for append-style updates."""

        ov = self.overlay(rel)
        if ov.exists():
            return ov
        src = self.path(rel)
        ov.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            if src.is_dir():
                shutil.copytree(src, ov)
            else:
                shutil.copy2(src, ov)
        return ov

    def source_of(self, rel) -> str:
        if self.overlay(rel).exists():
            return "PROJECT_OVERLAY"
        orig = self.original(rel)
        if orig is not None and orig.exists():
            return "ORIGINAL_READ_ONLY"
        return "ABSENT"
