#!/usr/bin/env python3
"""Verify release identity without modifying source, artifacts, commits, or tags.

The verifier compares the normalized tracked-file manifest of an already
created Git version tag with both the clean local worktree and an independently
extracted public source tree.  Its sole write is a deterministic,
no-clobber ``RELEASE_PROVENANCE.json`` outside both input trees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import tomllib
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence
from urllib.parse import quote, urlsplit


sys.dont_write_bytecode = True


REPORT_SCHEMA = "compag-release-provenance/v1"
MANIFEST_SCHEMA = "compag-normalized-git-tracked-files/v1"
PROVENANCE_FILENAME = "RELEASE_PROVENANCE.json"
TRUSTED_GIT_EXECUTABLE = Path("/usr/bin/git")
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TREE_BYTES = 2 * 1024 * 1024 * 1024
MAX_FILES = 100_000
FORBIDDEN_TREE_COMPONENTS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "venv",
}
PLACEHOLDER_SLUGS = {
    "example",
    "example-org",
    "example-user",
    "organization",
    "org",
    "owner",
    "project",
    "repo",
    "repository",
    "sample",
    "username",
    "your-org",
    "your-repo",
    "your-username",
}
WINDOWS_RESERVED = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class ProvenanceError(RuntimeError):
    """A fail-closed release-provenance verification error."""


@dataclass(frozen=True)
class ReleaseIdentity:
    repository_url: str
    owner: str
    repository: str
    version: str
    tag: str
    commit: str

    @property
    def release_url(self) -> str:
        return f"{self.repository_url}/releases/tag/{quote(self.tag, safe='')}"

    @property
    def commit_url(self) -> str:
        return f"{self.repository_url}/commit/{self.commit}"


@dataclass(frozen=True)
class ManifestRecord:
    relative_path: str
    git_mode: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class GitIndexRecord:
    relative_path: str
    git_mode: str
    object_id: str


@dataclass(frozen=True)
class OutputTarget:
    path: Path
    directory_descriptor: int
    directory_device: int
    directory_inode: int


GitRunner = Callable[[str, Path, Sequence[str]], bytes]


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii")


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _safe_relative(relative: str) -> None:
    if not relative or len(relative.encode("utf-8", "strict")) > 4096:
        raise ProvenanceError("manifest contains an empty or overlong path")
    if unicodedata.normalize("NFC", relative) != relative:
        raise ProvenanceError(f"manifest path is not NFC-normalized: {relative!r}")
    if "\\" in relative or "\x00" in relative or any(ord(value) < 32 or ord(value) == 127 for value in relative):
        raise ProvenanceError(f"manifest contains a non-portable path: {relative!r}")
    parsed = PurePosixPath(relative)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != relative
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ProvenanceError(f"manifest contains an unsafe path: {relative!r}")
    for part in parsed.parts:
        folded = part.casefold()
        if folded in FORBIDDEN_TREE_COMPONENTS or folded.endswith(".egg-info"):
            raise ProvenanceError(f"manifest contains excluded release state: {relative!r}")
        if part.endswith((".", " ")) or ":" in part or part.split(".", 1)[0].casefold() in WINDOWS_RESERVED:
            raise ProvenanceError(f"manifest contains a non-portable component: {relative!r}")


def _validate_path_set(paths: Sequence[str]) -> None:
    seen: set[str] = set()
    folded: dict[str, str] = {}
    normalized: dict[str, str] = {}
    for relative in paths:
        _safe_relative(relative)
        if relative in seen:
            raise ProvenanceError(f"manifest contains a duplicate path: {relative!r}")
        seen.add(relative)
        folded_key = relative.casefold()
        normalized_key = unicodedata.normalize("NFC", relative)
        if folded_key in folded and folded[folded_key] != relative:
            raise ProvenanceError(f"manifest contains a case-fold collision: {folded[folded_key]!r}, {relative!r}")
        if normalized_key in normalized and normalized[normalized_key] != relative:
            raise ProvenanceError(
                f"manifest contains a Unicode-normalization collision: {normalized[normalized_key]!r}, {relative!r}"
            )
        folded[folded_key] = relative
        normalized[normalized_key] = relative


def _stable_read(path: Path) -> tuple[bytes, os.stat_result]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ProvenanceError(f"release member is not a single-link regular file: {path}")
    if before.st_size > MAX_FILE_BYTES:
        raise ProvenanceError(f"release member exceeds the bounded file limit: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)

        def signature(node: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
            return (
                node.st_dev,
                node.st_ino,
                node.st_mode,
                node.st_nlink,
                node.st_size,
                node.st_mtime_ns,
                node.st_ctime_ns,
            )

        if signature(opened) != signature(before):
            raise ProvenanceError(f"release member changed while opening: {path}")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ProvenanceError(f"release member was truncated while reading: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ProvenanceError(f"release member grew while reading: {path}")
        if signature(os.fstat(descriptor)) != signature(opened):
            raise ProvenanceError(f"release member changed while reading: {path}")
        return b"".join(chunks), before
    finally:
        os.close(descriptor)


def _filesystem_mode(node: os.stat_result) -> str:
    return "100755" if stat.S_IMODE(node.st_mode) & 0o111 else "100644"


def _manifest_digest(records: Sequence[ManifestRecord]) -> str:
    payload = json.dumps(
        [asdict(record) for record in records],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _run_git(git: str, repository_root: Path, arguments: Sequence[str]) -> bytes:
    environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C.UTF-8",
        "LC_ALL": "C",
        "PATH": os.defpath,
    }
    try:
        completed = subprocess.run(
            [git, "-c", "core.fsmonitor=false", "-c", "gc.auto=0", "-C", os.fspath(repository_root), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProvenanceError(f"local Git command could not complete: {' '.join(arguments)}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise ProvenanceError(f"local Git command failed: {' '.join(arguments)}: {detail}")
    return completed.stdout


def _validate_git_executable(git: str) -> str:
    candidate = Path(git)
    if not candidate.is_absolute():
        raise ProvenanceError("--git must be an absolute path to a trusted Git executable")
    raw = candidate.absolute()
    node = raw.lstat()
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode) or not os.access(raw, os.X_OK):
        raise ProvenanceError("--git must be an executable, non-symlink regular file")
    resolved = raw.resolve(strict=True)
    trusted = TRUSTED_GIT_EXECUTABLE.resolve(strict=True)
    if resolved != trusted:
        raise ProvenanceError("--git must resolve exactly to the trusted system executable /usr/bin/git")
    for path in (trusted, *trusted.parents):
        path_node = path.lstat()
        if path_node.st_uid != 0 or stat.S_IMODE(path_node.st_mode) & 0o022:
            raise ProvenanceError("trusted Git executable ancestry must be root-owned and not group/world-writable")
    return os.fspath(trusted)


def _resolve_repository_root(repository_root: Path, git: str, runner: GitRunner) -> Path:
    raw = repository_root.absolute()
    node = raw.lstat()
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
        raise ProvenanceError("repository root must be a real directory")
    requested = raw.resolve(strict=True)
    discovered_raw = runner(git, requested, ["rev-parse", "--show-toplevel"])
    try:
        discovered = Path(discovered_raw.decode("utf-8", "strict").strip()).resolve(strict=True)
    except (UnicodeError, OSError) as error:
        raise ProvenanceError("Git returned an invalid repository root") from error
    if discovered != requested:
        raise ProvenanceError("--repository-root must be the exact Git worktree root")
    if runner(git, requested, ["rev-parse", "--is-bare-repository"]).strip() != b"false":
        raise ProvenanceError("repository root must be a non-bare worktree")
    return requested


def _project_version(payload: bytes) -> str:
    try:
        parsed = tomllib.loads(payload.decode("utf-8", "strict"))
        version = parsed["project"]["version"]
    except (KeyError, TypeError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ProvenanceError("pyproject.toml does not declare a valid project version") from error
    if not isinstance(version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ProvenanceError("project version must be a three-part numeric release")
    return version


def validate_identity(repository_url: str, version: str, tag: str, commit: str) -> ReleaseIdentity:
    parsed = urlsplit(repository_url)
    try:
        port = parsed.port
    except ValueError as error:
        raise ProvenanceError("repository URL has an invalid port") from error
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or repository_url.endswith("/")
    ):
        raise ProvenanceError("repository URL must be a canonical https://github.com/OWNER/REPOSITORY URL")
    parts = parsed.path.split("/")
    if len(parts) != 3 or not parts[1] or not parts[2] or parts[2].endswith(".git"):
        raise ProvenanceError("repository URL must identify exactly one GitHub owner and repository")
    owner, repository = parts[1], parts[2]
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", owner):
        raise ProvenanceError("GitHub owner is not a valid canonical slug")
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,99})", repository):
        raise ProvenanceError("GitHub repository is not a valid canonical slug")
    if owner.casefold() in PLACEHOLDER_SLUGS or repository.casefold() in PLACEHOLDER_SLUGS:
        raise ProvenanceError("repository URL contains a placeholder-looking owner or repository")
    if repository_url != f"https://github.com/{owner}/{repository}":
        raise ProvenanceError("repository URL is not in canonical form")
    if tag not in {version, f"v{version}"}:
        raise ProvenanceError("tag must exactly match the project version, optionally prefixed with v")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or len(set(commit)) == 1:
        raise ProvenanceError("commit must be a non-placeholder, lowercase, full 40-hex object ID")
    return ReleaseIdentity(repository_url, owner, repository, version, tag, commit)


def _git_records(
    git: str,
    repository_root: Path,
    commit: str,
    runner: GitRunner,
) -> tuple[list[ManifestRecord], list[GitIndexRecord]]:
    output = runner(git, repository_root, ["ls-tree", "-r", "-z", "-l", "--full-tree", commit])
    records: list[ManifestRecord] = []
    git_records: list[GitIndexRecord] = []
    total_bytes = 0
    for raw_row in output.split(b"\0"):
        if not raw_row:
            continue
        try:
            metadata, raw_path = raw_row.split(b"\t", 1)
            raw_mode, raw_type, raw_object, raw_size = metadata.split(None, 3)
            relative = raw_path.decode("utf-8", "strict")
            mode = raw_mode.decode("ascii", "strict")
            object_type = raw_type.decode("ascii", "strict")
            object_id = raw_object.decode("ascii", "strict")
            declared_size = int(raw_size.decode("ascii", "strict"))
        except (ValueError, UnicodeError) as error:
            raise ProvenanceError("Git tree contains an undecodable entry") from error
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise ProvenanceError(f"Git tree contains a link, gitlink, or unsupported mode: {relative!r}")
        if not re.fullmatch(r"[0-9a-f]{40}", object_id):
            raise ProvenanceError(f"Git tree contains an invalid blob object ID: {relative!r}")
        if declared_size < 0 or declared_size > MAX_FILE_BYTES:
            raise ProvenanceError(f"Git blob exceeds the bounded file limit: {relative!r}")
        data = runner(git, repository_root, ["cat-file", "blob", object_id])
        if len(data) != declared_size:
            raise ProvenanceError(f"Git blob size changed while reading: {relative!r}")
        total_bytes += len(data)
        if total_bytes > MAX_TREE_BYTES or len(records) + 1 > MAX_FILES:
            raise ProvenanceError("Git tree exceeds bounded release limits")
        records.append(ManifestRecord(relative, mode, len(data), hashlib.sha256(data).hexdigest()))
        git_records.append(GitIndexRecord(relative, mode, object_id))
    records.sort(key=lambda item: item.relative_path.encode("utf-8"))
    git_records.sort(key=lambda item: item.relative_path.encode("utf-8"))
    _validate_path_set([record.relative_path for record in records])
    if not records:
        raise ProvenanceError("Git commit contains no tracked release files")
    return records, git_records


def _scan_extracted_tree(root: Path) -> list[ManifestRecord]:
    requested = root.resolve(strict=True)
    node = requested.lstat()
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
        raise ProvenanceError("extracted source root must be a real directory")
    records: list[ManifestRecord] = []
    total_bytes = 0
    for path in sorted(requested.rglob("*"), key=lambda item: item.relative_to(requested).as_posix().encode("utf-8")):
        relative = path.relative_to(requested).as_posix()
        _safe_relative(relative)
        member = path.lstat()
        if stat.S_ISLNK(member.st_mode):
            raise ProvenanceError(f"extracted source contains a symbolic link: {relative!r}")
        if stat.S_ISDIR(member.st_mode):
            continue
        data, stable = _stable_read(path)
        total_bytes += len(data)
        if total_bytes > MAX_TREE_BYTES or len(records) + 1 > MAX_FILES:
            raise ProvenanceError("extracted source exceeds bounded release limits")
        records.append(ManifestRecord(relative, _filesystem_mode(stable), len(data), hashlib.sha256(data).hexdigest()))
    _validate_path_set([record.relative_path for record in records])
    return records


def _scan_worktree_tracked(repository_root: Path, expected: Sequence[ManifestRecord]) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for item in expected:
        path = repository_root / item.relative_path
        try:
            data, node = _stable_read(path)
        except FileNotFoundError as error:
            raise ProvenanceError(f"tracked worktree file is missing: {item.relative_path!r}") from error
        records.append(ManifestRecord(item.relative_path, _filesystem_mode(node), len(data), hashlib.sha256(data).hexdigest()))
    return records


def _compare_manifest(expected: Sequence[ManifestRecord], observed: Sequence[ManifestRecord], label: str) -> None:
    if list(expected) == list(observed):
        return
    expected_map = {item.relative_path: item for item in expected}
    observed_map = {item.relative_path: item for item in observed}
    missing = sorted(set(expected_map) - set(observed_map))
    extra = sorted(set(observed_map) - set(expected_map))
    changed = sorted(
        relative
        for relative in set(expected_map) & set(observed_map)
        if expected_map[relative] != observed_map[relative]
    )
    raise ProvenanceError(
        f"{label} does not match the tagged commit manifest: "
        f"missing={missing[:10]!r}, extra={extra[:10]!r}, changed={changed[:10]!r}"
    )


def _verify_index_matches_commit(
    git: str,
    repository_root: Path,
    expected: Sequence[GitIndexRecord],
    runner: GitRunner,
) -> None:
    output = runner(git, repository_root, ["ls-files", "--stage", "-z"])
    observed: list[GitIndexRecord] = []
    for raw_row in output.split(b"\0"):
        if not raw_row:
            continue
        try:
            metadata, raw_path = raw_row.split(b"\t", 1)
            raw_mode, raw_object, raw_stage = metadata.split(b" ", 2)
            relative = raw_path.decode("utf-8", "strict")
            mode = raw_mode.decode("ascii", "strict")
            object_id = raw_object.decode("ascii", "strict")
            stage = raw_stage.decode("ascii", "strict")
        except (ValueError, UnicodeError) as error:
            raise ProvenanceError("Git index contains an undecodable entry") from error
        _safe_relative(relative)
        if mode not in {"100644", "100755"} or not re.fullmatch(r"[0-9a-f]{40}", object_id) or stage != "0":
            raise ProvenanceError(f"Git index contains an unsupported or unresolved entry: {relative!r}")
        observed.append(GitIndexRecord(relative, mode, object_id))
    observed.sort(key=lambda item: item.relative_path.encode("utf-8"))
    _validate_path_set([item.relative_path for item in observed])
    if list(expected) != observed:
        raise ProvenanceError("Git index does not exactly match the asserted commit tree")


def _verify_git_identity(
    git: str,
    repository_root: Path,
    identity: ReleaseIdentity,
    remote: str,
    runner: GitRunner,
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", remote):
        raise ProvenanceError("remote name is invalid")
    head = runner(git, repository_root, ["rev-parse", "--verify", "HEAD"]).decode("ascii", "strict").strip()
    if head != identity.commit:
        raise ProvenanceError("HEAD does not equal the asserted full commit")
    tagged = runner(
        git,
        repository_root,
        ["rev-parse", "--verify", f"refs/tags/{identity.tag}^{{commit}}"],
    ).decode("ascii", "strict").strip()
    if tagged != identity.commit:
        raise ProvenanceError("version tag does not resolve to the asserted full commit")
    remote_url = runner(git, repository_root, ["remote", "get-url", remote]).decode("utf-8", "strict").strip()
    if remote_url != identity.repository_url:
        raise ProvenanceError("configured Git remote does not equal the canonical repository URL")


def _resolve_output(output: Path, repository_root: Path, extracted_source_root: Path) -> Path:
    raw = output.absolute()
    if raw.name != PROVENANCE_FILENAME:
        raise ProvenanceError(f"output filename must be exactly {PROVENANCE_FILENAME}")
    parent = raw.parent.resolve(strict=True)
    parent_node = raw.parent.lstat()
    if stat.S_ISLNK(parent_node.st_mode) or not stat.S_ISDIR(parent_node.st_mode):
        raise ProvenanceError("output parent must be a real existing directory")
    resolved = parent / raw.name
    if _lexists(resolved):
        raise ProvenanceError("RELEASE_PROVENANCE.json already exists; no-clobber policy refused it")
    for label, root in (("repository", repository_root), ("extracted source", extracted_source_root)):
        try:
            resolved.relative_to(root)
        except ValueError:
            pass
        else:
            raise ProvenanceError(f"RELEASE_PROVENANCE.json must be outside the {label} tree")
    return resolved


def _open_directory_without_symlinks(path: Path) -> int:
    if not path.is_absolute():
        raise ProvenanceError("output parent must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(Path(path.anchor), flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_output_target(path: Path) -> OutputTarget:
    try:
        directory_descriptor = _open_directory_without_symlinks(path.parent)
    except OSError as error:
        raise ProvenanceError("output parent could not be opened without following links") from error
    try:
        opened = os.fstat(directory_descriptor)
        observed = path.parent.lstat()
        if (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino):
            raise ProvenanceError("output parent changed while it was pinned")
        try:
            os.stat(path.name, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ProvenanceError("RELEASE_PROVENANCE.json already exists; no-clobber policy refused it")
        return OutputTarget(path, directory_descriptor, opened.st_dev, opened.st_ino)
    except Exception:
        os.close(directory_descriptor)
        raise


def _parent_still_pinned(target: OutputTarget) -> bool:
    try:
        observed = target.path.parent.lstat()
    except OSError:
        return False
    return (observed.st_dev, observed.st_ino) == (target.directory_device, target.directory_inode)


def _publish_owned_temp(target: OutputTarget, payload: bytes) -> None:
    temporary_name = f".{PROVENANCE_FILENAME}.owned-{os.getpid()}-{secrets.token_hex(16)}"
    temporary_descriptor: int | None = None
    temporary_identity: tuple[int, int] | None = None
    published = False
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            temporary_descriptor = os.open(temporary_name, flags, 0o600, dir_fd=target.directory_descriptor)
        except FileExistsError as error:
            raise ProvenanceError("owned provenance temporary name collided unexpectedly") from error
        except OSError as error:
            raise ProvenanceError("owned provenance temporary file could not be created") from error
        temporary_node = os.fstat(temporary_descriptor)
        temporary_identity = (temporary_node.st_dev, temporary_node.st_ino)
        view = memoryview(payload)
        while view:
            written = os.write(temporary_descriptor, view)
            if written <= 0:
                raise ProvenanceError("could not write complete provenance payload")
            view = view[written:]
        os.fchmod(temporary_descriptor, 0o644)
        os.fsync(temporary_descriptor)
        if not _parent_still_pinned(target):
            raise ProvenanceError("output parent changed before provenance publication")
        try:
            os.link(
                temporary_name,
                target.path.name,
                src_dir_fd=target.directory_descriptor,
                dst_dir_fd=target.directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise ProvenanceError("RELEASE_PROVENANCE.json appeared during verification; no-clobber policy refused it") from error
        published = True
        os.fsync(target.directory_descriptor)
        final_node = os.stat(target.path.name, dir_fd=target.directory_descriptor, follow_symlinks=False)
        if (final_node.st_dev, final_node.st_ino) != temporary_identity or not _parent_still_pinned(target):
            raise ProvenanceError("published provenance identity or parent changed before signoff")
        os.unlink(temporary_name, dir_fd=target.directory_descriptor)
        temporary_name = ""
        os.fsync(target.directory_descriptor)
    except Exception as error:
        if published and temporary_identity is not None:
            try:
                final_node = os.stat(target.path.name, dir_fd=target.directory_descriptor, follow_symlinks=False)
                if (final_node.st_dev, final_node.st_ino) == temporary_identity:
                    os.unlink(target.path.name, dir_fd=target.directory_descriptor)
                    published = False
            except OSError:
                pass
        if temporary_name and temporary_identity is not None:
            try:
                temporary_node = os.stat(temporary_name, dir_fd=target.directory_descriptor, follow_symlinks=False)
                if (temporary_node.st_dev, temporary_node.st_ino) == temporary_identity:
                    os.unlink(temporary_name, dir_fd=target.directory_descriptor)
                    temporary_name = ""
            except OSError:
                pass
        try:
            os.fsync(target.directory_descriptor)
        except OSError:
            pass
        if isinstance(error, ProvenanceError):
            raise
        raise ProvenanceError(f"provenance publication failed safely: {type(error).__name__}") from error
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)


def _verify_with_pinned_output(
    repository: Path,
    extracted: Path,
    target: OutputTarget,
    identity: ReleaseIdentity,
    git: str,
    remote: str,
    git_runner: GitRunner,
) -> dict[str, Any]:
    _verify_git_identity(git, repository, identity, remote, git_runner)
    committed, committed_git_index = _git_records(git, repository, identity.commit, git_runner)
    _verify_index_matches_commit(git, repository, committed_git_index, git_runner)
    _compare_manifest(committed, _scan_worktree_tracked(repository, committed), "tracked Git worktree")
    _compare_manifest(committed, _scan_extracted_tree(extracted), "extracted public source")

    # Repeat every mutable observation immediately before point-in-time publication.
    _verify_git_identity(git, repository, identity, remote, git_runner)
    _verify_index_matches_commit(git, repository, committed_git_index, git_runner)
    _compare_manifest(committed, _scan_worktree_tracked(repository, committed), "rechecked tracked Git worktree")
    _compare_manifest(committed, _scan_extracted_tree(extracted), "rechecked extracted public source")

    digest = _manifest_digest(committed)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "PASS_POINT_IN_TIME_LOCAL_TAGGED_COMMIT_AND_EXTRACTED_SOURCE_IDENTICAL",
        "release_identity": {
            "commit": identity.commit,
            "commit_url": identity.commit_url,
            "repository": f"{identity.owner}/{identity.repository}",
            "repository_url": identity.repository_url,
            "release_url": identity.release_url,
            "tag": identity.tag,
            "version": identity.version,
        },
        "verification": {
            "concurrent_mutation_assumption": "NO_CONCURRENT_MUTATION_DURING_POINT_IN_TIME_VERIFICATION",
            "configured_remote_matched": remote,
            "extracted_source_exact_regular_file_path_closure": True,
            "git_head_equals_commit": True,
            "github_remote_publication": "NOT_QUERIED_LOCAL_READ_ONLY_VERIFIER",
            "observation_model": "REPEATED_POINT_IN_TIME_LOCAL_OBSERVATION",
            "tag_peels_to_commit": True,
            "trusted_git_executable": git,
            "trusted_git_policy": "CANONICAL_ROOT_OWNED_NON_WRITABLE_SYSTEM_GIT",
            "git_index_matches_commit": True,
            "tracked_worktree_matches_commit": True,
        },
        "normalized_tracked_file_manifest": {
            "file_count": len(committed),
            "identity_sha256": digest,
            "records": [asdict(record) for record in committed],
            "schema": MANIFEST_SCHEMA,
        },
        "side_effect_policy": {
            "created_or_moved_tag": False,
            "modified_extracted_source": False,
            "modified_tracked_source": False,
            "pushed_commit": False,
            "uploaded_artifact": False,
            "wrote_only_external_provenance_file": True,
        },
    }
    _publish_owned_temp(target, _canonical_json(report))
    return {
        "manifest_identity_sha256": digest,
        "output": str(target.path),
        "status": report["status"],
    }


def verify_release_provenance(
    repository_root: Path,
    extracted_source_root: Path,
    output: Path,
    repository_url: str,
    tag: str,
    commit: str,
    *,
    git: str = "/usr/bin/git",
    remote: str = "origin",
    git_runner: GitRunner = _run_git,
) -> dict[str, Any]:
    """Verify tagged Git/source identity and emit one external provenance file."""

    if git_runner is _run_git:
        git = _validate_git_executable(git)
    repository = _resolve_repository_root(repository_root, git, git_runner)
    extracted_raw = extracted_source_root.absolute()
    extracted_node = extracted_raw.lstat()
    if stat.S_ISLNK(extracted_node.st_mode) or not stat.S_ISDIR(extracted_node.st_mode):
        raise ProvenanceError("extracted source root must be a real directory")
    extracted = extracted_raw.resolve(strict=True)
    if extracted == repository:
        raise ProvenanceError("repository and extracted source roots must be distinct")
    for outer, inner in ((repository, extracted), (extracted, repository)):
        try:
            inner.relative_to(outer)
        except ValueError:
            pass
        else:
            raise ProvenanceError("repository and extracted source trees must not overlap")

    # Read version from the asserted commit, not mutable operator text.
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or len(set(commit)) == 1:
        raise ProvenanceError("commit must be a non-placeholder, lowercase, full 40-hex object ID")
    commit_pyproject = git_runner(git, repository, ["show", f"{commit}:pyproject.toml"])
    version = _project_version(commit_pyproject)
    identity = validate_identity(repository_url, version, tag, commit)
    destination = _resolve_output(output, repository, extracted)
    target = _open_output_target(destination)
    try:
        return _verify_with_pinned_output(repository, extracted, target, identity, git, remote, git_runner)
    finally:
        os.close(target.directory_descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only verification that a local Git version tag, clean tracked worktree, and extracted public source "
            "are byte-identical; emits external RELEASE_PROVENANCE.json without changing release inputs."
        ),
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--extracted-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-url", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--git", default="/usr/bin/git", help="Absolute path to a trusted, non-symlink Git executable.")
    parser.add_argument("--remote", default="origin")
    args = parser.parse_args(argv)
    try:
        result = verify_release_provenance(
            args.repository_root,
            args.extracted_source_root,
            args.output,
            args.repository_url,
            args.tag,
            args.commit,
            git=args.git,
            remote=args.remote,
        )
    except (ProvenanceError, OSError, UnicodeError) as error:
        print(json.dumps({"error": str(error), "status": "FAIL"}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
