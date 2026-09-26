"""Small fail-closed filesystem primitives for the public execution facade."""

from __future__ import annotations

import ctypes
import csv
import errno
import hashlib
import contextlib
import json
import os
import stat
import threading
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping


class PublicIOError(RuntimeError):
    """Raised when a public path or immutable-file contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


MAX_CSV_FIELD_BYTES = 64 * 1024 * 1024
_CSV_FIELD_LIMIT_LOCK = threading.RLock()


@contextlib.contextmanager
def bounded_csv_field_limit(maximum: int = MAX_CSV_FIELD_BYTES):
    """Temporarily raise Python's process-global CSV field bound safely."""

    require(
        isinstance(maximum, int)
        and not isinstance(maximum, bool)
        and 1 <= maximum <= MAX_CSV_FIELD_BYTES,
        "CSV field-size bound is invalid",
    )
    with _CSV_FIELD_LIMIT_LOCK:
        previous = csv.field_size_limit()
        csv.field_size_limit(maximum)
        try:
            yield
        finally:
            csv.field_size_limit(previous)


def canonical_json_bytes(value: object) -> bytes:
    try:
        rendered = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise PublicIOError("value cannot be represented as canonical JSON") from exc
    return (rendered + "\n").encode("ascii")


def compact_json_sha256(value: object) -> str:
    try:
        payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    except (TypeError, ValueError, RecursionError) as exc:
        raise PublicIOError("value cannot be represented as compact canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def strict_json_bytes(payload: bytes, role: str) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant rejected: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key rejected: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise PublicIOError(f"{role} is not strict JSON") from exc


def canonical_json_value(payload: bytes, role: str) -> object:
    value = strict_json_bytes(payload, role)
    require(canonical_json_bytes(value) == payload, f"{role} is not canonical JSON")
    return value


def safe_relative(value: str, role: str) -> PurePosixPath:
    require(value != "", f"{role} must not be blank")
    require("\\" not in value, f"{role} must use POSIX separators")
    require(unicodedata.normalize("NFC", value) == value, f"{role} must be NFC-normalized")
    path = PurePosixPath(value)
    require(not path.is_absolute(), f"{role} must be relative")
    require(path.as_posix() == value, f"{role} is not canonical")
    require(all(part not in {"", ".", ".."} for part in path.parts), f"{role} contains traversal")
    return path


def portable_basename(value: str, role: str) -> str:
    require(value != "" and Path(value).name == value, f"{role} must be a basename")
    require(unicodedata.normalize("NFC", value) == value, f"{role} must be NFC-normalized")
    require(all(32 <= ord(character) < 127 for character in value), f"{role} must use printable ASCII")
    require(not any(character in '<>:"/\\|?*' for character in value), f"{role} contains a non-portable character")
    require(not value.startswith(".") and not value.endswith((".", " ")), f"{role} has a non-portable leading or trailing character")
    require(len(value.encode("ascii")) <= 255, f"{role} exceeds NAME_MAX")
    stem = value.split(".", 1)[0].casefold()
    reserved = {"con", "prn", "aux", "nul", *(f"com{index}" for index in range(1, 10)), *(f"lpt{index}" for index in range(1, 10))}
    require(stem not in reserved, f"{role} is a reserved device name")
    return value


def _open_read(path: Path) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if hasattr(os, "O_NOATIME"):
        flags |= os.O_NOATIME
    try:
        return os.open(path, flags)
    except PermissionError as exc:
        raise PublicIOError(
            f"cannot open without changing access time: {path}; copy the input into an owned project directory"
        ) from exc


def _stat_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _directory_anchor_identity(info: os.stat_result) -> tuple[int, ...]:
    """Return directory identity fields that child cleanup cannot change."""

    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
    )


def stable_file(path: Path, *, max_bytes: int | None = None) -> tuple[bytes, os.stat_result]:
    path_before = path.lstat()
    require(stat.S_ISREG(path_before.st_mode) and not stat.S_ISLNK(path_before.st_mode), f"not a regular file: {path}")
    fd = _open_read(path)
    try:
        before = os.fstat(fd)
        require(_stat_identity(before) == _stat_identity(path_before), f"file changed while opening: {path}")
        if max_bytes is not None:
            require(0 <= before.st_size <= max_bytes, f"file exceeds the supported read bound: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None:
                require(total <= max_bytes, f"file grew beyond the supported read bound: {path}")
            chunks.append(chunk)
        after = os.fstat(fd)
        require(_stat_identity(before) == _stat_identity(after), f"file changed while reading: {path}")
        path_after = path.lstat()
        require(_stat_identity(after) == _stat_identity(path_after), f"file path changed while reading: {path}")
        return b"".join(chunks), after
    finally:
        os.close(fd)


def hash_file_snapshot(
    path: Path,
    *,
    require_single_link: bool = False,
    max_bytes: int | None = None,
) -> tuple[str, os.stat_result]:
    path_before = path.lstat()
    require(stat.S_ISREG(path_before.st_mode) and not stat.S_ISLNK(path_before.st_mode), f"not a regular file: {path}")
    if require_single_link:
        require(path_before.st_nlink == 1, f"file is hardlinked: {path}")
    if max_bytes is not None:
        require(0 <= path_before.st_size <= max_bytes, f"file exceeds the supported hash bound: {path}")
    fd = _open_read(path)
    try:
        before = os.fstat(fd)
        require(_stat_identity(before) == _stat_identity(path_before), f"file changed while opening: {path}")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None:
                require(total <= max_bytes, f"file grew beyond the supported hash bound: {path}")
            digest.update(chunk)
        require(total == before.st_size, f"file size changed while hashing: {path}")
        after = os.fstat(fd)
        require(_stat_identity(before) == _stat_identity(after), f"file changed while hashing: {path}")
        path_after = path.lstat()
        require(_stat_identity(after) == _stat_identity(path_after), f"file path changed while hashing: {path}")
        return digest.hexdigest(), after
    finally:
        os.close(fd)


def scan_file_for_forbidden_bytes(
    path: Path,
    forbidden_needles: Iterable[bytes],
) -> tuple[str, os.stat_result]:
    """Hash one stable file while rejecting case-insensitive byte tokens."""
    needles = tuple(bytes(value).lower() for value in forbidden_needles)
    require(needles and all(needle for needle in needles), "forbidden byte token set is invalid")
    overlap = max(len(needle) for needle in needles) - 1
    path_before = path.lstat()
    require(
        stat.S_ISREG(path_before.st_mode)
        and not stat.S_ISLNK(path_before.st_mode)
        and path_before.st_nlink == 1,
        f"not a single-link regular file: {path}",
    )
    fd = _open_read(path)
    try:
        before = os.fstat(fd)
        require(_stat_identity(before) == _stat_identity(path_before), f"file changed while opening: {path}")
        digest = hashlib.sha256()
        carry = b""
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            lowered = (carry + chunk).lower()
            require(not any(needle in lowered for needle in needles), f"private locator found in output member: {path.name}")
            carry = lowered[-overlap:] if overlap else b""
        after = os.fstat(fd)
        require(_stat_identity(after) == _stat_identity(before), f"file changed while scanning: {path}")
        path_after = path.lstat()
        require(_stat_identity(path_after) == _stat_identity(after), f"file path changed while scanning: {path}")
        return digest.hexdigest(), after
    finally:
        os.close(fd)


def sha256_file(path: Path) -> str:
    digest, _snapshot = hash_file_snapshot(path)
    return digest


@contextlib.contextmanager
def verified_file_path(
    path: Path,
    expected_sha256: str,
    *,
    max_bytes: int | None = None,
    require_single_link: bool = True,
):
    """Yield a proc-fd path pinned to the exact verified inode."""
    path_before = path.lstat()
    require(stat.S_ISREG(path_before.st_mode) and not stat.S_ISLNK(path_before.st_mode), f"not a regular file: {path}")
    if require_single_link:
        require(path_before.st_nlink == 1, f"file is hardlinked: {path}")
    if max_bytes is not None:
        require(path_before.st_size <= max_bytes, f"file exceeds its supported size bound: {path}")
    fd = _open_read(path)
    try:
        before = os.fstat(fd)
        require(_stat_identity(before) == _stat_identity(path_before), f"file changed while opening: {path}")

        def hash_fd() -> str:
            os.lseek(fd, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if max_bytes is not None:
                    require(total <= max_bytes, f"file exceeds its supported size bound: {path}")
                digest.update(chunk)
            require(total == before.st_size, f"file size changed while hashing: {path}")
            return digest.hexdigest()

        require(hash_fd() == expected_sha256, f"file SHA-256 mismatch: {path}")
        proc_path = Path(f"/proc/self/fd/{fd}")
        require(proc_path.exists(), "verified file descriptor path is unavailable")
        yield proc_path, before
        after = os.fstat(fd)
        require(_stat_identity(after) == _stat_identity(before), f"verified file changed while in use: {path}")
        require(hash_fd() == expected_sha256, f"verified file content changed while in use: {path}")
        path_after = path.lstat()
        require(_stat_identity(path_after) == _stat_identity(before), f"verified file path changed while in use: {path}")
    finally:
        os.close(fd)


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new_bytes(path: Path, payload: bytes, mode: int = 0o644) -> None:
    require(path.is_absolute(), "output path must be absolute")
    require(path.parent.exists() and path.parent.is_dir() and not path.parent.is_symlink(), "output parent is unsafe")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, mode)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            require(written > 0, f"short write: {path}")
            offset += written
        os.fchmod(fd, mode)
        os.fsync(fd)
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, f"unsafe output file: {path}")
        require(info.st_size == len(payload), f"output size mismatch: {path}")
    finally:
        os.close(fd)
    fsync_directory(path.parent)


def write_new_json(path: Path, value: object, mode: int = 0o644) -> None:
    write_new_bytes(path, canonical_json_bytes(value), mode)


def copy_new(source: Path, destination: Path, mode: int = 0o644) -> dict[str, object]:
    source_fd = _open_read(source)
    destination_fd = -1
    try:
        before = os.fstat(source_fd)
        require(stat.S_ISREG(before.st_mode), f"copy source is not regular: {source}")
        destination_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, mode)
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            offset = 0
            while offset < len(chunk):
                offset += os.write(destination_fd, chunk[offset:])
        os.fchmod(destination_fd, mode)
        os.fsync(destination_fd)
        source_after = os.fstat(source_fd)
        target = os.fstat(destination_fd)
        require(
            (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (source_after.st_dev, source_after.st_ino, source_after.st_mode, source_after.st_size, source_after.st_mtime_ns, source_after.st_ctime_ns),
            f"copy source changed: {source}",
        )
        require(total == before.st_size == target.st_size, f"copy size mismatch: {destination}")
        require((before.st_dev, before.st_ino) != (target.st_dev, target.st_ino), "copy shares source inode")
        require(target.st_nlink == 1 and stat.S_ISREG(target.st_mode), "copy target is unsafe")
    finally:
        os.close(source_fd)
        if destination_fd >= 0:
            os.close(destination_fd)
    require(sha256_file(destination) == digest.hexdigest(), f"destination byte verification failed: {destination}")
    fsync_directory(destination.parent)
    return {"sha256": digest.hexdigest(), "size_bytes": total, "mode_octal": f"{mode:04o}"}


def regular_files(
    root: Path,
    *,
    excluded: Iterable[str] = (),
    allowed_empty_directories: Iterable[str] = (),
) -> list[Path]:
    require(root.exists() and root.is_dir() and not root.is_symlink(), f"unsafe tree root: {root}")
    exclusions = set(excluded)
    allowed_empty = {safe_relative(value, "allowed empty directory").as_posix() for value in allowed_empty_directories}
    files: list[Path] = []
    folded: dict[str, str] = {}
    normalized: dict[str, str] = {}
    observed_empty: set[str] = set()

    def register(relative: str) -> None:
        safe_relative(relative, "tree member")
        fold = relative.casefold()
        nfc = unicodedata.normalize("NFC", relative)
        require(fold not in folded, f"tree contains a case-fold path collision: {relative} and {folded.get(fold)}")
        require(nfc not in normalized, f"tree contains a normalization path collision: {relative} and {normalized.get(nfc)}")
        folded[fold] = relative
        normalized[nfc] = relative

    for current, directories, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories.sort()
        names.sort()
        if current_path != root and not directories and not names:
            relative = current_path.relative_to(root).as_posix()
            require(relative in allowed_empty, f"tree contains an undeclared empty directory: {relative}")
            observed_empty.add(relative)
        for name in directories:
            path = current_path / name
            node = path.lstat()
            require(stat.S_ISDIR(node.st_mode) and not stat.S_ISLNK(node.st_mode), "tree contains unsafe directory")
            register(path.relative_to(root).as_posix())
        for name in names:
            path = current_path / name
            node = path.lstat()
            require(stat.S_ISREG(node.st_mode) and not stat.S_ISLNK(node.st_mode) and node.st_nlink == 1, "tree contains symlink, hardlink, or special file")
            relative = path.relative_to(root).as_posix()
            register(relative)
            if relative not in exclusions:
                files.append(path)
    require(observed_empty <= allowed_empty, "empty-directory closure mismatch")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def manifest_rows(
    root: Path,
    *,
    excluded: Iterable[str] = (),
    allowed_empty_directories: Iterable[str] = (),
    include_identity: bool = False,
    include_directories: bool = False,
    allowed_files: Iterable[str] | None = None,
    file_size_limits: Mapping[str, int] | None = None,
    max_total_bytes: int | None = None,
) -> list[dict[str, object]]:
    require(root.exists() and root.is_dir() and not root.is_symlink(), f"unsafe tree root: {root}")
    exclusions = set(excluded)
    allowed_empty = {safe_relative(value, "allowed empty directory").as_posix() for value in allowed_empty_directories}
    folded: dict[str, str] = {}
    normalized: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    allowed_file_set = None if allowed_files is None else {safe_relative(value, "allowed file").as_posix() for value in allowed_files}
    size_limits = {} if file_size_limits is None else {
        safe_relative(path, "file size limit").as_posix(): int(limit)
        for path, limit in file_size_limits.items()
    }
    require(all(limit >= 0 for limit in size_limits.values()), "file size limit is invalid")
    total_bytes = 0

    def register(relative: str) -> None:
        safe_relative(relative, "tree member")
        fold = relative.casefold()
        nfc = unicodedata.normalize("NFC", relative)
        require(fold not in folded, f"tree contains a case-fold path collision: {relative} and {folded.get(fold)}")
        require(nfc not in normalized, f"tree contains a normalization path collision: {relative} and {normalized.get(nfc)}")
        folded[fold] = relative
        normalized[nfc] = relative

    def open_directory_at(parent_fd: int, name: str, expected: os.stat_result, relative: str) -> int:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        fd = os.open(name, flags, dir_fd=parent_fd)
        try:
            opened = os.fstat(fd)
            require(_stat_identity(opened) == _stat_identity(expected), f"tree directory changed while opening: {relative}")
            return fd
        except BaseException:
            os.close(fd)
            raise

    def hash_at(parent_fd: int, name: str, expected: os.stat_result, relative: str) -> tuple[str, os.stat_result]:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        if hasattr(os, "O_NOATIME"):
            flags |= os.O_NOATIME
        try:
            fd = os.open(name, flags, dir_fd=parent_fd)
        except PermissionError as exc:
            raise PublicIOError(f"cannot hash tree member without changing access time: {relative}") from exc
        try:
            before = os.fstat(fd)
            require(_stat_identity(before) == _stat_identity(expected), f"tree member changed while opening: {relative}")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(fd)
            require(_stat_identity(after) == _stat_identity(before), f"tree member changed while hashing: {relative}")
            path_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            require(_stat_identity(path_after) == _stat_identity(after), f"tree member path changed while hashing: {relative}")
            return digest.hexdigest(), after
        finally:
            os.close(fd)

    def visit(directory_fd: int, relative_parent: PurePosixPath | None) -> None:
        nonlocal total_bytes
        names = sorted(os.listdir(directory_fd))
        if relative_parent is not None and not names:
            relative = relative_parent.as_posix()
            require(relative in allowed_empty, f"tree contains an undeclared empty directory: {relative}")
        for name in names:
            require(isinstance(name, str), "tree member name is not text")
            relative_path = PurePosixPath(name) if relative_parent is None else relative_parent / name
            relative = relative_path.as_posix()
            register(relative)
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(before.st_mode):
                child_fd = open_directory_at(directory_fd, name, before, relative)
                try:
                    visit(child_fd, relative_path)
                    after_fd = os.fstat(child_fd)
                    after_path = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    require(
                        _stat_identity(after_fd) == _stat_identity(before) == _stat_identity(after_path),
                        f"tree directory changed during traversal: {relative}",
                    )
                    if include_directories:
                        rows.append(
                            {
                                "path": relative,
                                "type": "directory",
                                "mode_octal": f"{stat.S_IMODE(after_fd.st_mode):04o}",
                                "device": after_fd.st_dev,
                                "inode": after_fd.st_ino,
                                "uid": after_fd.st_uid,
                                "gid": after_fd.st_gid,
                                "nlink": after_fd.st_nlink,
                                "size_bytes": after_fd.st_size,
                                "mtime_ns": after_fd.st_mtime_ns,
                                "ctime_ns": after_fd.st_ctime_ns,
                            }
                        )
                finally:
                    os.close(child_fd)
            else:
                require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "tree contains symlink, hardlink, or special file")
                if allowed_file_set is not None:
                    require(relative in allowed_file_set, f"tree contains an unexpected file: {relative}")
                if relative in size_limits:
                    require(before.st_size <= size_limits[relative], f"tree member exceeds its size bound: {relative}")
                total_bytes += before.st_size
                if max_total_bytes is not None:
                    require(total_bytes <= max_total_bytes, "tree exceeds its total byte bound")
                digest, snapshot = hash_at(directory_fd, name, before, relative)
                if relative not in exclusions:
                    row: dict[str, object] = {
                        "path": relative,
                        "sha256": digest,
                        "size_bytes": snapshot.st_size,
                        "mode_octal": f"{stat.S_IMODE(snapshot.st_mode):04o}",
                    }
                    if include_identity:
                        row.update(
                            {
                                "type": "file",
                                "device": snapshot.st_dev,
                                "inode": snapshot.st_ino,
                                "uid": snapshot.st_uid,
                                "gid": snapshot.st_gid,
                                "nlink": snapshot.st_nlink,
                                "mtime_ns": snapshot.st_mtime_ns,
                                "ctime_ns": snapshot.st_ctime_ns,
                            }
                        )
                    rows.append(row)

    root_before = root.lstat()
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        require(_stat_identity(os.fstat(root_fd)) == _stat_identity(root_before), "tree root changed while opening")
        visit(root_fd, None)
        root_after_fd = os.fstat(root_fd)
        root_after_path = root.lstat()
        require(
            _stat_identity(root_after_fd) == _stat_identity(root_before) == _stat_identity(root_after_path),
            "tree root changed during traversal",
        )
        if include_directories:
            rows.append(
                {
                    "path": ".",
                    "type": "directory",
                    "mode_octal": f"{stat.S_IMODE(root_after_fd.st_mode):04o}",
                    "device": root_after_fd.st_dev,
                    "inode": root_after_fd.st_ino,
                    "uid": root_after_fd.st_uid,
                    "gid": root_after_fd.st_gid,
                    "nlink": root_after_fd.st_nlink,
                    "size_bytes": root_after_fd.st_size,
                    "mtime_ns": root_after_fd.st_mtime_ns,
                    "ctime_ns": root_after_fd.st_ctime_ns,
                }
            )
    finally:
        os.close(root_fd)
    return sorted(rows, key=lambda row: str(row["path"]))


def tree_identity_rows(
    root: Path,
    *,
    allowed_empty_directories: Iterable[str] = (),
    allowed_files: Iterable[str] | None = None,
    file_size_limits: Mapping[str, int] | None = None,
    max_total_bytes: int | None = None,
) -> list[dict[str, object]]:
    return manifest_rows(
        root,
        allowed_empty_directories=allowed_empty_directories,
        include_identity=True,
        include_directories=True,
        allowed_files=allowed_files,
        file_size_limits=file_size_limits,
        max_total_bytes=max_total_bytes,
    )


def owned_manifest_contract(
    root: Path,
    *,
    excluded: Iterable[str] = (),
    allowed_empty_directories: Iterable[str] = (),
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Return portable file and directory rows for an output owned by this uid/gid."""

    observed = manifest_rows(
        root,
        excluded=excluded,
        allowed_empty_directories=allowed_empty_directories,
        include_identity=True,
        include_directories=True,
    )
    files: list[dict[str, object]] = []
    directories: list[dict[str, str]] = []
    for row in observed:
        mode = int(str(row["mode_octal"]), 8)
        require(row.get("uid") == os.getuid() and row.get("gid") == os.getgid(), "output tree ownership changed")
        require(mode & 0o022 == 0, "output tree contains a group/world-writable node")
        if row.get("type") == "directory":
            directories.append({"path": str(row["path"]), "mode_octal": str(row["mode_octal"])})
            continue
        require(row.get("type") == "file", "output tree contains an unsupported node")
        files.append(
            {
                "path": str(row["path"]),
                "sha256": str(row["sha256"]),
                "size_bytes": int(row["size_bytes"]),
                "mode_octal": str(row["mode_octal"]),
            }
        )
    return (
        sorted(files, key=lambda row: str(row["path"])),
        sorted(directories, key=lambda row: str(row["path"])),
    )


def prune_owned_empty_directories(root: Path) -> tuple[str, ...]:
    """Remove only empty descendants from a closed, run-owned runtime tree."""

    root_info = root.lstat()
    require(
        stat.S_ISDIR(root_info.st_mode)
        and not root.is_symlink()
        and root_info.st_uid == os.getuid()
        and root_info.st_gid == os.getgid()
        and root_info.st_mode & 0o022 == 0,
        "runtime cache root is unsafe",
    )
    removed: list[str] = []
    for current, directories, names in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        for name in sorted(names):
            info = (current_path / name).lstat()
            require(
                stat.S_ISREG(info.st_mode)
                and not stat.S_ISLNK(info.st_mode)
                and info.st_nlink == 1
                and info.st_uid == root_info.st_uid
                and info.st_gid == root_info.st_gid
                and info.st_mode & 0o022 == 0,
                "runtime cache contains an unsafe file",
            )
        for name in sorted(directories):
            child = current_path / name
            info = child.lstat()
            require(
                stat.S_ISDIR(info.st_mode)
                and not stat.S_ISLNK(info.st_mode)
                and info.st_uid == root_info.st_uid
                and info.st_gid == root_info.st_gid
                and info.st_mode & 0o022 == 0,
                "runtime cache contains an unsafe directory",
            )
            try:
                child.rmdir()
            except OSError as exc:
                if exc.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                    raise PublicIOError("runtime cache changed during empty-directory pruning") from exc
            else:
                removed.append(child.relative_to(root).as_posix())
                fsync_directory(current_path)
    after = root.lstat()
    require(
        (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_gid,
        )
        == (
            root_info.st_dev,
            root_info.st_ino,
            root_info.st_mode,
            root_info.st_uid,
            root_info.st_gid,
        ),
        "runtime cache root changed during empty-directory pruning",
    )
    return tuple(sorted(removed))


def purge_owned_runtime_cache_files(
    root: Path,
    *,
    retained_files: Iterable[str],
) -> tuple[str, ...]:
    """Delete generated cache files from a closed run-owned runtime tree.

    The caller must wait for every process using ``root`` before calling this
    function.  Evidence files named by ``retained_files`` are preserved.  The
    complete tree is validated before the first unlink so a symlink, hard link,
    special node, ownership change, or unsafe permission fails closed.
    """

    root_info = root.lstat()
    require(
        stat.S_ISDIR(root_info.st_mode)
        and not root.is_symlink()
        and root_info.st_uid == os.getuid()
        and root_info.st_gid == os.getgid()
        and root_info.st_mode & 0o022 == 0,
        "runtime cache root is unsafe",
    )
    retained = tuple(
        safe_relative(str(relative), "retained runtime evidence").as_posix()
        for relative in retained_files
    )
    require(len(retained) == len(set(retained)), "retained runtime evidence is duplicated")
    retained_set = set(retained)
    observed: dict[str, tuple[Path, tuple[int, ...]]] = {}
    directory_identities: dict[str, tuple[Path, tuple[int, ...]]] = {}
    for current, directories, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        current_info = current_path.lstat()
        require(
            stat.S_ISDIR(current_info.st_mode)
            and not stat.S_ISLNK(current_info.st_mode)
            and current_info.st_uid == root_info.st_uid
            and current_info.st_gid == root_info.st_gid
            and current_info.st_mode & 0o022 == 0,
            "runtime cache contains an unsafe directory",
        )
        current_relative = (
            "."
            if current_path == root
            else current_path.relative_to(root).as_posix()
        )
        directory_identities[current_relative] = (
            current_path,
            _directory_anchor_identity(current_info),
        )
        for name in sorted(directories):
            info = (current_path / name).lstat()
            require(
                stat.S_ISDIR(info.st_mode)
                and not stat.S_ISLNK(info.st_mode)
                and info.st_uid == root_info.st_uid
                and info.st_gid == root_info.st_gid
                and info.st_mode & 0o022 == 0,
                "runtime cache contains an unsafe directory",
            )
        for name in sorted(names):
            path = current_path / name
            info = path.lstat()
            require(
                stat.S_ISREG(info.st_mode)
                and not stat.S_ISLNK(info.st_mode)
                and info.st_nlink == 1
                and info.st_uid == root_info.st_uid
                and info.st_gid == root_info.st_gid
                and info.st_mode & 0o022 == 0,
                "runtime cache contains an unsafe file",
            )
            relative = path.relative_to(root).as_posix()
            observed[relative] = (path, _stat_identity(info))
    require(
        retained_set <= set(observed),
        "retained runtime evidence is missing",
    )
    retained_identities: dict[str, tuple[tuple[int, ...], str]] = {}
    for relative in retained:
        path, identity = observed[relative]
        payload, snapshot = stable_file(path, max_bytes=16 * 1024 * 1024)
        require(
            _stat_identity(snapshot) == identity,
            "retained runtime evidence changed before cleanup",
        )
        retained_identities[relative] = (
            identity,
            hashlib.sha256(payload).hexdigest(),
        )
    removed: list[str] = []
    for relative in sorted(set(observed) - retained_set):
        path, identity = observed[relative]
        parent_relative = (
            "."
            if path.parent == root
            else path.parent.relative_to(root).as_posix()
        )
        parent_path, parent_identity = directory_identities[parent_relative]
        descriptor = -1
        try:
            descriptor = os.open(
                parent_path,
                os.O_RDONLY
                | os.O_CLOEXEC
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            require(
                _directory_anchor_identity(os.fstat(descriptor))
                == parent_identity,
                "runtime cache directory changed before cleanup",
            )
            current = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
            require(
                _stat_identity(current) == identity,
                "runtime cache changed before cleanup",
            )
            os.unlink(path.name, dir_fd=descriptor)
            os.fsync(descriptor)
        except OSError as exc:
            raise PublicIOError("runtime cache file could not be removed") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        removed.append(relative)
    for relative in sorted(
        (value for value in directory_identities if value != "."),
        key=lambda value: (len(PurePosixPath(value).parts), value),
        reverse=True,
    ):
        path, identity = directory_identities[relative]
        parent_relative = (
            "."
            if path.parent == root
            else path.parent.relative_to(root).as_posix()
        )
        parent_path, parent_identity = directory_identities[parent_relative]
        descriptor = -1
        try:
            descriptor = os.open(
                parent_path,
                os.O_RDONLY
                | os.O_CLOEXEC
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            require(
                _directory_anchor_identity(os.fstat(descriptor))
                == parent_identity,
                "runtime cache directory changed before cleanup",
            )
            current = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
            require(
                _directory_anchor_identity(current) == identity,
                "runtime cache directory changed before cleanup",
            )
            try:
                os.rmdir(path.name, dir_fd=descriptor)
            except OSError as exc:
                if exc.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                    raise
            else:
                os.fsync(descriptor)
        except OSError as exc:
            raise PublicIOError("runtime cache directory could not be removed") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    remaining, _directories = owned_manifest_contract(root)
    require(
        {str(row["path"]) for row in remaining} == retained_set,
        "runtime cache cleanup did not reach its evidence-only closure",
    )
    for relative, (identity, digest) in retained_identities.items():
        payload, snapshot = stable_file(
            root / relative,
            max_bytes=16 * 1024 * 1024,
        )
        require(
            _stat_identity(snapshot) == identity
            and hashlib.sha256(payload).hexdigest() == digest,
            "retained runtime evidence changed during cleanup",
        )
    after = root.lstat()
    require(
        (after.st_dev, after.st_ino, after.st_mode, after.st_uid, after.st_gid)
        == (
            root_info.st_dev,
            root_info.st_ino,
            root_info.st_mode,
            root_info.st_uid,
            root_info.st_gid,
        ),
        "runtime cache root changed during cleanup",
    )
    return tuple(removed)


def rename_noreplace(source: Path, destination: Path) -> None:
    source_parent = source.parent
    destination_parent = destination.parent
    require(source_parent.stat().st_dev == destination_parent.stat().st_dev, "atomic publish requires one filesystem")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise PublicIOError(f"publish target already exists: {destination}")
        raise OSError(error, os.strerror(error), str(destination))
    fsync_directory(destination_parent)
    if source_parent != destination_parent:
        fsync_directory(source_parent)


def publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Publish a sealed directory and recover a completed rename from fsync errors."""
    try:
        rename_noreplace(source, destination)
    except BaseException as exc:
        if source.exists() or source.is_symlink() or not destination.is_dir() or destination.is_symlink():
            raise
        try:
            fsync_directory(destination)
            fsync_directory(destination.parent)
            if source.parent != destination.parent:
                fsync_directory(source.parent)
        except BaseException as durability_exc:
            raise PublicIOError(
                "directory was published but durability confirmation failed; verify the exact output before reuse"
            ) from durability_exc


def publish_file_noreplace(source: Path, destination: Path) -> None:
    """Publish a sealed file and recover a completed rename from fsync errors."""
    try:
        rename_noreplace(source, destination)
    except BaseException as exc:
        if source.exists() or source.is_symlink():
            raise
        try:
            info = destination.lstat()
        except OSError:
            raise exc
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
            raise exc
        try:
            fd = os.open(destination, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                opened = os.fstat(fd)
                require(_stat_identity(opened) == _stat_identity(info), "published file changed while confirming durability")
                os.fsync(fd)
            finally:
                os.close(fd)
            fsync_directory(destination.parent)
            if source.parent != destination.parent:
                fsync_directory(source.parent)
        except BaseException as durability_exc:
            raise PublicIOError(
                "file was published but durability confirmation failed; verify the exact output before reuse"
            ) from durability_exc
