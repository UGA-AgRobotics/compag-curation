"""Complete self-excluding public-file manifest verification."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import stat
import unicodedata
from pathlib import Path, PurePosixPath


MANIFEST_FIELDS = (
    "relative_path",
    "size_bytes",
    "sha256",
    "mode_octal",
    "license_class",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
MODE_PATTERN = re.compile(r"[0-7]{4}\Z")


class ManifestError(ValueError):
    """Raised when a public manifest is incomplete, unsafe, or inconsistent."""


def _hash(path: Path) -> str:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ManifestError("manifest payload is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ManifestError("manifest payload cannot be opened safely") from exc
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mode) != (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mode,
        ):
            raise ManifestError("manifest payload changed before hashing")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        completed = os.fstat(descriptor)
        if (
            completed.st_dev,
            completed.st_ino,
            completed.st_size,
            completed.st_mode,
            completed.st_mtime_ns,
            completed.st_ctime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mode,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ):
            raise ManifestError("manifest payload changed while hashing")
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mode,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) != (
        completed.st_dev,
        completed.st_ino,
        completed.st_size,
        completed.st_mode,
        completed.st_mtime_ns,
        completed.st_ctime_ns,
    ):
        raise ManifestError("manifest payload changed after hashing")
    return digest.hexdigest()


def _safe_relative(value: str) -> str:
    if not value or "\\" in value or unicodedata.normalize("NFC", value) != value:
        raise ManifestError("manifest path is empty, non-NFC, or uses a backslash")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or value != pure.as_posix():
        raise ManifestError("manifest path is absolute, traversing, or noncanonical")
    if pure.parts and pure.parts[0] == ".git":
        raise ManifestError("manifest cannot include Git metadata")
    return value


def _files(root: Path) -> dict[str, os.stat_result]:
    output: dict[str, os.stat_result] = {}
    folded: set[str] = set()
    normalized: set[str] = set()
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(name for name in directories if name != ".git")
        for name in directories:
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ManifestError("symlink or special directory is prohibited")
        for name in sorted(files):
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ManifestError("symlink or special file is prohibited")
            relative = _safe_relative(path.relative_to(root).as_posix())
            fold_key = relative.casefold()
            nfc_key = unicodedata.normalize("NFC", relative)
            if fold_key in folded or nfc_key in normalized:
                raise ManifestError("casefold or NFC path collision")
            folded.add(fold_key)
            normalized.add(nfc_key)
            output[relative] = info
    return output


def _read_rows(manifest: Path) -> list[dict[str, str]]:
    info = manifest.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ManifestError("manifest must be a regular non-symlink file")
    try:
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
                raise ManifestError("manifest schema mismatch")
            rows = list(reader)
    except (csv.Error, UnicodeError, OSError) as exc:
        raise ManifestError("manifest cannot be parsed") from exc
    if not rows:
        raise ManifestError("manifest has no payload rows")
    return rows


def _verify_manifest_impl(
    root: Path,
    manifest: Path,
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    """Verify one canonical ``license_class`` manifest and its full tree."""

    root = root.resolve(strict=True)
    root_info = root.lstat()
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ManifestError("manifest root must be a real directory")
    manifest = manifest.absolute()
    try:
        manifest_relative = manifest.relative_to(root).as_posix()
    except ValueError as exc:
        raise ManifestError("manifest is outside the requested root") from exc
    try:
        if manifest.resolve(strict=True) != manifest:
            raise ManifestError("manifest path resolves through a symlink")
    except OSError as exc:
        raise ManifestError("manifest is missing or unavailable") from exc
    rows = _read_rows(manifest)
    live = _files(root)

    by_path: dict[str, dict[str, str]] = {}
    folded: set[str] = set()
    normalized: set[str] = set()
    for row in rows:
        if set(row) != set(MANIFEST_FIELDS) or any(value is None for value in row.values()):
            raise ManifestError("manifest row schema mismatch")
        relative = _safe_relative(row["relative_path"])
        if relative == manifest_relative:
            raise ManifestError("self-excluding manifest contains its own row")
        fold_key = relative.casefold()
        nfc_key = unicodedata.normalize("NFC", relative)
        if relative in by_path:
            raise ManifestError("duplicate manifest row")
        if fold_key in folded or nfc_key in normalized:
            raise ManifestError("manifest casefold or NFC collision")
        folded.add(fold_key)
        normalized.add(nfc_key)
        if not SHA256_PATTERN.fullmatch(row["sha256"]):
            raise ManifestError("manifest SHA-256 field is malformed")
        if not MODE_PATTERN.fullmatch(row["mode_octal"]):
            raise ManifestError("manifest mode field is malformed")
        if not row["license_class"].strip():
            raise ManifestError("manifest license class is empty")
        try:
            size = int(row["size_bytes"], 10)
        except ValueError as exc:
            raise ManifestError("manifest size field is malformed") from exc
        if size < 0 or str(size) != row["size_bytes"]:
            raise ManifestError("manifest size field is noncanonical")
        by_path[relative] = row

    expected_paths = set(by_path) | {manifest_relative}
    if require_complete and expected_paths != set(live):
        raise ManifestError("manifest does not cover the complete regular-file set")
    for relative, row in by_path.items():
        candidate = root.joinpath(*PurePosixPath(relative).parts)
        try:
            if candidate.resolve(strict=True) != candidate:
                raise ManifestError("manifest payload resolves through a symlink")
        except OSError as exc:
            raise ManifestError("manifest payload is missing or unavailable") from exc
        info = live.get(relative)
        if info is None:
            raise ManifestError("manifest payload is missing")
        if int(row["size_bytes"]) != info.st_size or row["sha256"] != _hash(candidate):
            raise ManifestError("manifest byte binding mismatch")
        if row["mode_octal"] != f"{stat.S_IMODE(info.st_mode):04o}":
            raise ManifestError("manifest mode binding mismatch")
    return {
        "schema": "compag-public-file-inventory/v1",
        "status": "PASS",
        "verified_rows": len(rows),
        "complete": require_complete,
        "manifest_field": "license_class",
    }


def verify_manifest(
    root: Path,
    manifest: Path,
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    """Normalize every filesystem/parser contract failure to ``ManifestError``."""

    try:
        return _verify_manifest_impl(root, manifest, require_complete=require_complete)
    except ManifestError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise ManifestError("manifest verification could not complete") from exc
