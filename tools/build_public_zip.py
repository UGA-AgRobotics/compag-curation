#!/usr/bin/env python3
"""Build and verify the deterministic, allowlist-closed public source ZIP."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import sys
import time
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable


DEFAULT_TOP_LEVEL = "COMPAG_D_26_02120_PUBLIC_RUNNABLE_GITHUB_v1.9.3"
ALLOWLIST_RELATIVE = "manifests/public_release_allowlist.json"
ALLOWLIST_FIELDS = {
    "schema",
    "policy",
    "release_version",
    "path_encoding",
    "path_count",
    "paths_sha256",
    "paths",
}
MAX_ALLOWLIST_BYTES = 4 * 1024 * 1024
MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 384 * 1024 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 65_536
MAX_PATH_BYTES = 512
PROHIBITED_COMPONENTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".state",
    ".tox",
    ".venv",
    "__pycache__",
    "evidence_prompts",
    "manuscript_authority",
    "manuscript_working",
    "private_records",
    "qa_tmp",
    "response_working",
    "toolchain",
}
PROHIBITED_NAME_TERMS = (
    "_run_now",
    "response_to_reviewers",
    "reviewer_comments",
)
WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class PublicZipError(RuntimeError):
    """Raised when a source snapshot or output cannot be proven safe."""


def _signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _inode(value: os.stat_result | tuple[int, ...]) -> tuple[int, int]:
    if isinstance(value, os.stat_result):
        return value.st_dev, value.st_ino
    return value[0], value[1]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublicZipError(f"duplicate JSON key in public allowlist: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PublicZipError(f"non-finite JSON constant in public allowlist: {value}")


def _canonical_relative(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or not value.isprintable()
        or len(value.encode("ascii")) > MAX_PATH_BYTES
    ):
        raise PublicZipError("allowlist path must be bounded nonempty printable ASCII")
    pure = PurePosixPath(value)
    if (
        value.startswith(("/", "\\"))
        or "\\" in value
        or "\x00" in value
        or re.match(r"^[A-Za-z]:", value)
        or pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
        or (pure.parts and pure.parts[0] == ".git")
        or unicodedata.normalize("NFC", value) != value
        or any(
            part.endswith((".", " "))
            or ":" in part
            or part.split(".", 1)[0].casefold() in WINDOWS_RESERVED
            for part in pure.parts
        )
    ):
        raise PublicZipError("allowlist contains a noncanonical or unsafe path")
    components = {part.casefold() for part in pure.parts}
    if (
        components & {item.casefold() for item in PROHIBITED_COMPONENTS}
        or any(part.endswith("_evidence_prompts") for part in components)
    ):
        raise PublicZipError("allowlist contains a prohibited private or cache component")
    if any(term in pure.name.casefold() for term in PROHIBITED_NAME_TERMS):
        raise PublicZipError("allowlist contains a prohibited prompt or manuscript name")
    return value


def _derived_directories(paths: Iterable[str]) -> set[str]:
    result: set[str] = set()
    for relative in paths:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            result.add(parent.as_posix())
            parent = parent.parent
    return result


def _reject_collisions(files: Iterable[str], directories: Iterable[str]) -> None:
    folded: dict[str, str] = {}
    normalized: dict[str, str] = {}
    for kind, value in sorted(
        [("file", item) for item in files] + [("directory", item) for item in directories]
    ):
        marker = f"{kind}:{value}"
        for table, key, label in (
            (folded, value.casefold(), "casefold"),
            (normalized, unicodedata.normalize("NFC", value), "NFC"),
        ):
            previous = table.get(key)
            if previous is not None and previous != marker:
                raise PublicZipError(f"public paths contain a {label} collision")
            table[key] = marker


def _lstat(path: Path | str, *, directory_fd: int | None = None) -> os.stat_result:
    if directory_fd is None:
        return Path(path).lstat()
    return os.stat(path, dir_fd=directory_fd, follow_symlinks=False)


def _stable_read(
    path: Path | str,
    *,
    max_bytes: int,
    directory_fd: int | None = None,
) -> tuple[bytes, tuple[int, ...]]:
    try:
        before = _lstat(path, directory_fd=directory_fd)
    except OSError as exc:
        raise PublicZipError(f"public member is missing or unavailable: {Path(path).name}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > max_bytes
    ):
        raise PublicZipError("public member must be a bounded single-link regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise PublicZipError("public member could not be opened safely") from exc
    opened_signature: tuple[int, ...] | None = None
    completed_signature: tuple[int, ...] | None = None
    try:
        opened = os.fstat(descriptor)
        opened_signature = _signature(opened)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or opened_signature != _signature(before):
            raise PublicZipError("public member changed before reading")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise PublicZipError("public member exceeds its explicit size bound")
        completed = os.fstat(descriptor)
        completed_signature = _signature(completed)
        if completed_signature != opened_signature:
            raise PublicZipError("public member changed while reading")
    finally:
        os.close(descriptor)
    try:
        after_signature = _signature(_lstat(path, directory_fd=directory_fd))
    except OSError as exc:
        raise PublicZipError("public member changed after reading") from exc
    if completed_signature is None or after_signature != completed_signature:
        raise PublicZipError("public member changed after reading")
    return b"".join(chunks), completed_signature


def _load_allowlist(root: Path) -> tuple[tuple[str, ...], bytes, tuple[int, ...]]:
    path = root / ALLOWLIST_RELATIVE
    payload, signature = _stable_read(path, max_bytes=MAX_ALLOWLIST_BYTES)
    if payload.startswith(b"\xef\xbb\xbf"):
        raise PublicZipError("public allowlist must not contain a UTF-8 BOM")
    try:
        value = json.loads(
            payload.decode("utf-8", "strict"),
            object_pairs_hook=_duplicate_key,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PublicZipError("public allowlist is malformed") from exc
    if not isinstance(value, dict) or set(value) != ALLOWLIST_FIELDS:
        raise PublicZipError("public allowlist has the wrong fields")
    if (
        value.get("schema") != "compag-curation-public-release-allowlist/v1"
        or value.get("policy") != "DENY_BY_DEFAULT_EXACT_REGULAR_FILES"
        or value.get("release_version") != "1.9.3"
        or value.get("path_encoding") != "UTF8_NFC_POSIX_RELATIVE_SORTED_LF_TERMINATED"
        or type(value.get("path_count")) is not int
        or not isinstance(value.get("paths_sha256"), str)
        or not isinstance(value.get("paths"), list)
    ):
        raise PublicZipError("public allowlist contract mismatch")
    paths = tuple(_canonical_relative(item) for item in value["paths"])
    if not paths or list(paths) != sorted(paths) or len(paths) != len(set(paths)):
        raise PublicZipError("public allowlist paths must be nonempty, sorted, and unique")
    digest = _sha256("".join(f"{item}\n" for item in paths).encode("ascii"))
    if value["path_count"] != len(paths) or value["paths_sha256"] != digest:
        raise PublicZipError("public allowlist count or digest mismatch")
    if ALLOWLIST_RELATIVE not in paths:
        raise PublicZipError("public allowlist must include itself")
    directories = _derived_directories(paths)
    _reject_collisions(paths, directories)
    return paths, payload, signature


def _validate_root(root: Path) -> Path:
    requested = root.absolute()
    try:
        node = requested.lstat()
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise PublicZipError("public root is missing or unavailable") from exc
    if (
        stat.S_ISLNK(node.st_mode)
        or not stat.S_ISDIR(node.st_mode)
        or resolved != requested
        or stat.S_IMODE(node.st_mode) != 0o755
    ):
        raise PublicZipError("public root must be a real mode-0755 non-symlink directory")
    return requested


def _validate_tree(root: Path, expected_paths: Iterable[str]) -> list[Path]:
    expected = set(expected_paths)
    expected_directories = _derived_directories(expected)
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for current, directories, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        if current_path == root:
            directories[:] = sorted(name for name in directories if name != ".git")
        else:
            directories[:] = sorted(directories)
        for name in directories:
            path = current_path / name
            relative = _canonical_relative(path.relative_to(root).as_posix())
            node = path.lstat()
            if (
                stat.S_ISLNK(node.st_mode)
                or not stat.S_ISDIR(node.st_mode)
                or stat.S_IMODE(node.st_mode) != 0o755
            ):
                raise PublicZipError("public tree contains an unsafe or non-normalized directory")
            observed_directories.add(relative)
        for name in sorted(names):
            path = current_path / name
            relative = _canonical_relative(path.relative_to(root).as_posix())
            node = path.lstat()
            if (
                stat.S_ISLNK(node.st_mode)
                or not stat.S_ISREG(node.st_mode)
                or node.st_nlink != 1
                or stat.S_IMODE(node.st_mode) != 0o644
            ):
                raise PublicZipError("public tree contains a symlink, special file, hardlink, or non-normalized file")
            observed_files.add(relative)
    if observed_files != expected:
        missing = sorted(expected - observed_files)
        extra = sorted(observed_files - expected)
        raise PublicZipError(f"public allowlist closure mismatch: missing={missing[:3]} extra={extra[:3]}")
    if observed_directories != expected_directories:
        missing = sorted(expected_directories - observed_directories)
        extra = sorted(observed_directories - expected_directories)
        raise PublicZipError(f"public directory closure mismatch: missing={missing[:3]} extra={extra[:3]}")
    _reject_collisions(observed_files, observed_directories)
    return [root / relative for relative in sorted(observed_files)]


def _assert_source_snapshot(root: Path, signatures: dict[str, tuple[int, ...]]) -> None:
    for relative, signature in signatures.items():
        try:
            current = _signature((root / relative).lstat())
        except OSError as exc:
            raise PublicZipError("public tree changed after snapshot capture") from exc
        if current != signature:
            raise PublicZipError("public tree changed after snapshot capture")


def _top_level(value: str) -> str:
    canonical = _canonical_relative(value)
    if len(PurePosixPath(canonical).parts) != 1 or canonical in {".", ".git"}:
        raise PublicZipError("ZIP top-level name must be exactly one safe directory component")
    return canonical


def _zip_datetime(source_date_epoch: int) -> tuple[tuple[int, int, int, int, int, int], int]:
    if type(source_date_epoch) is not int or source_date_epoch < 0:
        raise PublicZipError("SOURCE_DATE_EPOCH must be a non-negative integer")
    normalized = source_date_epoch - source_date_epoch % 2
    try:
        value = time.gmtime(normalized)
    except (OverflowError, OSError, ValueError) as exc:
        raise PublicZipError("SOURCE_DATE_EPOCH is outside the supported ZIP timestamp range") from exc
    date_time = value[:6]
    if not 1980 <= date_time[0] <= 2107:
        raise PublicZipError("SOURCE_DATE_EPOCH is outside the 1980-2107 ZIP timestamp range")
    return date_time, normalized


def _assert_output_parent_binding(parent: Path, descriptor: int, owner: tuple[int, int]) -> None:
    try:
        held = os.fstat(descriptor)
        before = parent.lstat()
        resolved = parent.resolve(strict=True)
        after = parent.lstat()
    except (OSError, RuntimeError) as exc:
        raise PublicZipError("public ZIP output parent binding changed") from exc
    if (
        not stat.S_ISDIR(held.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or stat.S_ISLNK(after.st_mode)
        or not stat.S_ISDIR(after.st_mode)
        or resolved != parent
        or _inode(held) != owner
        or _inode(before) != owner
        or _inode(after) != owner
    ):
        raise PublicZipError("public ZIP output parent binding changed")


def _output_target(root: Path, output: Path) -> tuple[Path, int, tuple[int, int]]:
    requested = output.absolute()
    if not requested.name or requested.name in {".", ".."} or requested.suffix.casefold() != ".zip":
        raise PublicZipError("public ZIP output must have a nonempty .zip filename")
    parent_fd: int | None = None
    try:
        parent_node = requested.parent.lstat()
    except OSError as exc:
        raise PublicZipError("public ZIP output parent must already exist") from exc
    if stat.S_ISLNK(parent_node.st_mode) or not stat.S_ISDIR(parent_node.st_mode):
        raise PublicZipError("public ZIP output parent must be a real non-symlink directory")
    try:
        parent_fd = os.open(
            requested.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        held = os.fstat(parent_fd)
        parent = requested.parent.resolve(strict=True)
        parent_after = requested.parent.lstat()
    except (OSError, RuntimeError) as exc:
        if parent_fd is not None:
            os.close(parent_fd)
        raise PublicZipError("public ZIP output parent must be a real non-symlink directory") from exc
    owner = _inode(held)
    if (
        stat.S_ISLNK(parent_node.st_mode)
        or not stat.S_ISDIR(parent_node.st_mode)
        or not stat.S_ISDIR(held.st_mode)
        or stat.S_ISLNK(parent_after.st_mode)
        or not stat.S_ISDIR(parent_after.st_mode)
        or parent != requested.parent
        or _inode(parent_node) != owner
        or _inode(parent_after) != owner
    ):
        os.close(parent_fd)
        raise PublicZipError("public ZIP output parent must be a real non-symlink directory")
    target = parent / requested.name
    if target == root or target.is_relative_to(root):
        os.close(parent_fd)
        raise PublicZipError("public ZIP output must be outside the source tree")
    try:
        _lstat(target.name, directory_fd=parent_fd)
    except FileNotFoundError:
        pass
    except OSError as exc:
        os.close(parent_fd)
        raise PublicZipError("public ZIP output target is unavailable") from exc
    else:
        os.close(parent_fd)
        raise PublicZipError("public ZIP output already exists; no-clobber policy refused it")
    try:
        _assert_output_parent_binding(parent, parent_fd, owner)
    except BaseException:
        os.close(parent_fd)
        raise
    return target, parent_fd, owner


def _zip_rows(
    top_level: str,
    paths: Iterable[str],
    payloads: dict[str, bytes],
) -> list[tuple[str, bool, bytes]]:
    directories = _derived_directories(paths)
    rows = [(f"{top_level}/", True, b"")]
    rows.extend((f"{top_level}/{name}/", True, b"") for name in directories)
    rows.extend((f"{top_level}/{name}", False, payloads[name]) for name in paths)
    rows.sort(key=lambda row: row[0])
    if len(rows) > MAX_MEMBERS:
        raise PublicZipError("public ZIP exceeds its explicit member-count bound")
    return rows


def _write_zip_contents(
    handle: BinaryIO,
    rows: list[tuple[str, bool, bytes]],
    date_time: tuple[int, ...],
) -> None:
    with zipfile.ZipFile(handle, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.comment = b""
        for name, is_directory, payload in rows:
            info = zipfile.ZipInfo(name, date_time=date_time)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = (stat.S_IFDIR | 0o755) if is_directory else (stat.S_IFREG | 0o644)
            info.external_attr = (mode << 16) | (0x10 if is_directory else 0)
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _write_zip(handle: BinaryIO, rows: list[tuple[str, bool, bytes]], date_time: tuple[int, ...]) -> None:
    _write_zip_contents(handle, rows, date_time)


def _render_canonical_zip(
    rows: list[tuple[str, bool, bytes]],
    date_time: tuple[int, ...],
) -> bytes:
    handle = io.BytesIO()
    _write_zip_contents(handle, rows, date_time)
    return handle.getvalue()


def _verify_zip(
    archive_bytes: bytes,
    rows: list[tuple[str, bool, bytes]],
    date_time: tuple[int, ...],
    *,
    canonical_bytes: bytes | None = None,
) -> None:
    expected_bytes = canonical_bytes
    if expected_bytes is None:
        expected_bytes = _render_canonical_zip(rows, date_time)
    if archive_bytes != expected_bytes:
        raise PublicZipError("completed public ZIP canonical byte stream mismatch")
    expected_names = [row[0] for row in rows]
    if expected_names != sorted(expected_names) or len(expected_names) != len(set(expected_names)):
        raise PublicZipError("internal public ZIP row order or uniqueness invariant failed")
    expected = {name: (is_directory, payload) for name, is_directory, payload in rows}
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
            infos = archive.infolist()
            if archive.comment or [info.filename for info in infos] != expected_names:
                raise PublicZipError("completed public ZIP member order or closure mismatch")
            for info in infos:
                is_directory, payload = expected[info.filename]
                expected_mode = (stat.S_IFDIR | 0o755) if is_directory else (stat.S_IFREG | 0o644)
                if (
                    info.date_time != date_time
                    or info.create_system != 3
                    or info.compress_type != zipfile.ZIP_DEFLATED
                    or info.extra
                    or info.comment
                    or info.flag_bits & 1
                    or info.is_dir() != is_directory
                    or info.external_attr >> 16 != expected_mode
                ):
                    raise PublicZipError("completed public ZIP metadata verification failed")
                extracted = archive.read(info)
                if extracted != payload or info.file_size != len(payload):
                    raise PublicZipError("completed public ZIP payload verification failed")
            if archive.testzip() is not None:
                raise PublicZipError("completed public ZIP CRC verification failed")
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        if isinstance(exc, PublicZipError):
            raise
        raise PublicZipError("completed public ZIP could not be verified") from exc


def _unlink_owned(
    path: Path | str | None,
    owner: tuple[int, int] | None,
    *,
    directory_fd: int | None = None,
) -> bool:
    if owner is None:
        return True
    if path is None:
        return False
    try:
        current = _lstat(path, directory_fd=directory_fd)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if _inode(current) != owner:
        return False
    try:
        if directory_fd is None:
            Path(path).unlink()
        else:
            os.unlink(path, dir_fd=directory_fd)
    except OSError:
        return False
    return True


def build_public_zip(
    root: Path,
    output: Path,
    source_date_epoch: int,
    *,
    top_level: str = DEFAULT_TOP_LEVEL,
) -> dict[str, object]:
    """Create an immutable deterministic public ZIP at an absent external path."""

    public_root = _validate_root(root)
    root_name = _top_level(top_level)
    date_time, normalized_epoch = _zip_datetime(source_date_epoch)
    paths, allowlist_payload, allowlist_signature = _load_allowlist(public_root)
    source_paths = _validate_tree(public_root, paths)
    payloads: dict[str, bytes] = {ALLOWLIST_RELATIVE: allowlist_payload}
    signatures: dict[str, tuple[int, ...]] = {ALLOWLIST_RELATIVE: allowlist_signature}
    total_bytes = len(allowlist_payload)
    for path in source_paths:
        relative = path.relative_to(public_root).as_posix()
        if relative == ALLOWLIST_RELATIVE:
            continue
        payload, signature = _stable_read(path, max_bytes=MAX_MEMBER_BYTES)
        total_bytes += len(payload)
        if total_bytes > MAX_TOTAL_BYTES:
            raise PublicZipError("public tree exceeds its explicit aggregate size bound")
        payloads[relative] = payload
        signatures[relative] = signature
    _validate_tree(public_root, paths)
    _assert_source_snapshot(public_root, signatures)
    rows = _zip_rows(root_name, paths, payloads)
    canonical_bytes = _render_canonical_zip(rows, date_time)
    if len(canonical_bytes) > MAX_ARCHIVE_BYTES:
        raise PublicZipError("public ZIP exceeds its explicit archive-size bound")
    _verify_zip(canonical_bytes, rows, date_time, canonical_bytes=canonical_bytes)

    target, parent_fd, parent_owner = _output_target(public_root, output)
    temporary: str | None = None
    descriptor: int | None = None
    temporary_owner: tuple[int, int] | None = None
    output_owner: tuple[int, int] | None = None
    ownership_descriptor: int | None = None
    try:
        _assert_output_parent_binding(target.parent, parent_fd, parent_owner)

        for index in range(100):
            candidate = f".{target.name}.public-zip-{os.getpid()}-{index}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                continue
            except OSError as exc:
                raise PublicZipError("public ZIP staging file could not be created safely") from exc
            temporary = candidate
            temporary_owner = _inode(os.fstat(descriptor))
            break
        if temporary is None or descriptor is None or temporary_owner is None:
            raise PublicZipError("could not reserve an exclusive public ZIP staging file")

        handle = os.fdopen(descriptor, "wb", closefd=True)
        descriptor = None
        with handle:
            _write_zip(handle, rows, date_time)
            os.fchmod(handle.fileno(), 0o644)
            handle.flush()
            os.fsync(handle.fileno())
        _assert_output_parent_binding(target.parent, parent_fd, parent_owner)
        temporary_bytes, temporary_signature = _stable_read(
            temporary,
            max_bytes=MAX_ARCHIVE_BYTES,
            directory_fd=parent_fd,
        )
        if _inode(temporary_signature) != temporary_owner or stat.S_IMODE(temporary_signature[2]) != 0o644:
            raise PublicZipError("public ZIP staging identity or normalized mode changed")
        _verify_zip(temporary_bytes, rows, date_time, canonical_bytes=canonical_bytes)
        _assert_source_snapshot(public_root, signatures)
        _assert_output_parent_binding(target.parent, parent_fd, parent_owner)
        try:
            ownership_descriptor = os.open(
                temporary,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise PublicZipError("public ZIP staging ownership hold could not be opened") from exc
        held = os.fstat(ownership_descriptor)
        if _inode(held) != temporary_owner or stat.S_IMODE(held.st_mode) != 0o644:
            raise PublicZipError("public ZIP staging ownership hold could not be established")
        try:
            os.link(
                temporary,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise PublicZipError("public ZIP output appeared; no-clobber policy refused it") from exc
        except OSError as exc:
            raise PublicZipError("public ZIP output could not be installed atomically") from exc
        linked_temporary = _lstat(temporary, directory_fd=parent_fd)
        linked_output = _lstat(target.name, directory_fd=parent_fd)
        if (
            _inode(linked_temporary) != temporary_owner
            or _inode(linked_output) != temporary_owner
            or _signature(linked_output) != _signature(linked_temporary)
            or stat.S_IMODE(linked_output.st_mode) != 0o644
        ):
            raise PublicZipError("public ZIP output identity changed during atomic publication")
        output_owner = temporary_owner
        if not _unlink_owned(temporary, temporary_owner, directory_fd=parent_fd):
            raise PublicZipError("public ZIP staging file could not be released")
        temporary_owner = None
        installed = _lstat(target.name, directory_fd=parent_fd)
        if (
            _inode(installed) != output_owner
            or installed.st_nlink != 1
            or stat.S_IMODE(installed.st_mode) != 0o644
        ):
            raise PublicZipError("public ZIP output identity or normalized mode changed after installation")
        _assert_output_parent_binding(target.parent, parent_fd, parent_owner)
        os.fsync(parent_fd)
        completed_bytes, completed_signature = _stable_read(
            target.name,
            max_bytes=MAX_ARCHIVE_BYTES,
            directory_fd=parent_fd,
        )
        if _inode(completed_signature) != output_owner or stat.S_IMODE(completed_signature[2]) != 0o644:
            raise PublicZipError("public ZIP output changed after atomic publication")
        _verify_zip(completed_bytes, rows, date_time, canonical_bytes=canonical_bytes)
        _assert_source_snapshot(public_root, signatures)
        _assert_output_parent_binding(target.parent, parent_fd, parent_owner)
    except BaseException as exc:
        if descriptor is not None:
            os.close(descriptor)
        output_cleanup_ok = _unlink_owned(target.name, output_owner, directory_fd=parent_fd)
        temporary_cleanup_ok = _unlink_owned(temporary, temporary_owner, directory_fd=parent_fd)
        cleanup_ok = output_cleanup_ok and temporary_cleanup_ok
        try:
            os.fsync(parent_fd)
        except OSError:
            cleanup_ok = False
        if not cleanup_ok:
            raise PublicZipError("public ZIP rollback was incomplete") from exc
        if isinstance(exc, OSError):
            raise PublicZipError("public ZIP filesystem operation failed") from exc
        raise
    finally:
        if ownership_descriptor is not None:
            os.close(ownership_descriptor)
        os.close(parent_fd)

    return {
        "schema": "compag-deterministic-public-source-zip/v1",
        "status": "PASS",
        "sha256": _sha256(completed_bytes),
        "size_bytes": len(completed_bytes),
        "member_count": len(rows),
        "file_count": len(paths),
        "top_level": root_name,
        "source_date_epoch": source_date_epoch,
        "zip_timestamp_epoch_utc": normalized_epoch,
        "zip_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(normalized_epoch)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-level", default=DEFAULT_TOP_LEVEL)
    parser.add_argument("--source-date-epoch", type=int)
    args = parser.parse_args()
    epoch = args.source_date_epoch
    if epoch is None:
        raw_epoch = os.environ.get("SOURCE_DATE_EPOCH")
        if raw_epoch is None or not raw_epoch.isascii() or not raw_epoch.isdecimal():
            parser.error("--source-date-epoch or decimal SOURCE_DATE_EPOCH is required")
        epoch = int(raw_epoch)
    try:
        receipt = build_public_zip(args.root, args.output, epoch, top_level=args.top_level)
    except PublicZipError as exc:
        print(f"public ZIP build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
