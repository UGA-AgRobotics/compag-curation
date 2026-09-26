#!/usr/bin/env python3
"""Normalize a setuptools source distribution for reproducible release evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import stat
import tarfile
import unicodedata
from pathlib import Path, PurePosixPath


class SdistNormalizationError(RuntimeError):
    """Raised when an input sdist is unsafe or cannot be normalized exactly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name: str) -> str:
    if not name or "\\" in name or unicodedata.normalize("NFC", name) != name:
        raise SdistNormalizationError("sdist member name is empty, non-NFC, or uses a backslash")
    pure = PurePosixPath(name.rstrip("/"))
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != name.rstrip("/"):
        raise SdistNormalizationError("sdist member name is absolute, traversing, or noncanonical")
    if not pure.parts or pure.parts[0] in {".", ".git"}:
        raise SdistNormalizationError("sdist member has a prohibited top-level path")
    return pure.as_posix()


def _signature(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_uid,
        info.st_gid,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


def _read_members(source: Path) -> tuple[list[tuple[str, bool, bytes]], str]:
    try:
        before = source.lstat()
    except OSError as exc:
        raise SdistNormalizationError("input sdist is missing or unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise SdistNormalizationError("input sdist must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise SdistNormalizationError("input sdist could not be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _signature(opened) != _signature(before):
            raise SdistNormalizationError("input sdist changed before reading")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        completed = os.fstat(descriptor)
        if _signature(completed) != _signature(opened):
            raise SdistNormalizationError("input sdist changed while reading")
    finally:
        os.close(descriptor)
    try:
        after = source.lstat()
    except OSError as exc:
        raise SdistNormalizationError("input sdist changed after reading") from exc
    if _signature(after) != _signature(completed):
        raise SdistNormalizationError("input sdist changed after reading")
    source_bytes = b"".join(chunks)
    rows: list[tuple[str, bool, bytes]] = []
    folded: set[str] = set()
    normalized: set[str] = set()
    roots: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(source_bytes), mode="r:gz") as archive:
            for member in archive.getmembers():
                name = _safe_name(member.name)
                if not (member.isdir() or member.isreg()):
                    raise SdistNormalizationError("sdist contains a non-regular member")
                fold_key = name.casefold()
                nfc_key = unicodedata.normalize("NFC", name)
                if fold_key in folded or nfc_key in normalized:
                    raise SdistNormalizationError("sdist contains a casefold or NFC collision")
                folded.add(fold_key)
                normalized.add(nfc_key)
                roots.add(PurePosixPath(name).parts[0])
                if member.isdir():
                    payload = b""
                else:
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise SdistNormalizationError("regular sdist member has no payload")
                    payload = extracted.read()
                    if len(payload) != member.size:
                        raise SdistNormalizationError("sdist member size changed while reading")
                rows.append((name, member.isdir(), payload))
    except (OSError, tarfile.TarError) as exc:
        raise SdistNormalizationError("input sdist could not be read") from exc
    if len(roots) != 1 or not rows:
        raise SdistNormalizationError("sdist must contain one non-empty top-level source directory")
    return sorted(rows, key=lambda row: (row[0], not row[1])), hashlib.sha256(source_bytes).hexdigest()


def _unlink_owned(path: Path, owner: tuple[int, ...] | None) -> bool:
    if owner is None:
        return True
    try:
        current = path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if _signature(current) != owner:
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


def normalize_sdist(source: Path, output: Path, source_date_epoch: int) -> dict[str, object]:
    if source_date_epoch < 0:
        raise SdistNormalizationError("SOURCE_DATE_EPOCH must be non-negative")
    source = source.absolute()
    try:
        if source.resolve(strict=True) != source:
            raise SdistNormalizationError("input sdist resolves through a symlink")
    except OSError as exc:
        raise SdistNormalizationError("input sdist is missing or unavailable") from exc
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent_status = output.parent.lstat()
        parent_resolved = output.parent.resolve(strict=True)
    except OSError as exc:
        raise SdistNormalizationError("normalized output parent is unavailable") from exc
    if (
        stat.S_ISLNK(parent_status.st_mode)
        or not stat.S_ISDIR(parent_status.st_mode)
        or parent_resolved != output.parent
    ):
        raise SdistNormalizationError("normalized output parent must be a real directory")
    rows, input_sha256 = _read_members(source)
    temporary = output.with_name(f".{output.name}.normalize-{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    parent_fd = os.open(
        output.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        descriptor = os.open(temporary, flags, 0o600)
    except BaseException:
        os.close(parent_fd)
        raise
    temporary_owner: tuple[int, ...] | None = _signature(os.fstat(descriptor))
    output_owner: tuple[int, ...] | None = None
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=source_date_epoch, compresslevel=9) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for name, is_directory, payload in rows:
                        member_name = f"{name}/" if is_directory else name
                        info = tarfile.TarInfo(member_name)
                        info.type = tarfile.DIRTYPE if is_directory else tarfile.REGTYPE
                        info.mode = 0o755 if is_directory else 0o644
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = source_date_epoch
                        info.size = 0 if is_directory else len(payload)
                        archive.addfile(info, None if is_directory else io.BytesIO(payload))
            raw.flush()
            os.fsync(raw.fileno())
        os.chmod(temporary, 0o644)
        temporary_owner = _signature(temporary.lstat())
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as exc:
            raise SdistNormalizationError("normalized output already exists") from exc
        temporary_owner = _signature(temporary.lstat())
        output_owner = _signature(output.lstat())
        if output_owner != temporary_owner:
            raise SdistNormalizationError("normalized output ownership changed during installation")
        if not _unlink_owned(temporary, temporary_owner):
            raise SdistNormalizationError("normalized staging file could not be released")
        temporary_owner = None
        output_owner = _signature(output.lstat())
        os.fsync(parent_fd)
        if _signature(output.lstat()) != output_owner:
            raise SdistNormalizationError("normalized output changed after installation")
    except BaseException as exc:
        cleanup_ok = _unlink_owned(output, output_owner) and _unlink_owned(temporary, temporary_owner)
        try:
            os.fsync(parent_fd)
        except OSError:
            cleanup_ok = False
        if not cleanup_ok:
            raise SdistNormalizationError("normalized output rollback was incomplete") from exc
        raise
    finally:
        os.close(parent_fd)
    return {
        "schema": "compag-normalized-sdist/v1",
        "status": "PASS",
        "input_sha256": input_sha256,
        "output_sha256": sha256_file(output),
        "output_size": output.stat().st_size,
        "members": len(rows),
        "source_date_epoch": source_date_epoch,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    args = parser.parse_args()
    result = normalize_sdist(args.input, args.output, args.source_date_epoch)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
