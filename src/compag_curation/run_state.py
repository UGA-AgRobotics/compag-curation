"""Owned, locked, no-clobber public run and atomic stage state."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .public_io import (
    PublicIOError,
    canonical_json_value,
    canonical_json_bytes,
    compact_json_sha256,
    fsync_directory,
    manifest_rows,
    owned_manifest_contract,
    portable_basename,
    publish_directory_noreplace,
    publish_file_noreplace,
    rename_noreplace,
    sha256_file,
    stable_file,
    strict_json_bytes,
    write_new_bytes,
    write_new_json,
)


PAUSED_FOR_REVIEW_EXIT_CODE = 3
RUN_SCHEMA = "compag-curation-public-run/v1"
STAGE_SCHEMA = "compag-curation-public-stage/v1"
MAX_STAGE_RECEIPT_BYTES = 128 * 1024 * 1024
STAGE_NAMES = (
    "00_input_inventory",
    "10_prepare",
    "20_proposals_features",
    "30_review_import",
    "40_group_split",
    "50_train_bundle",
    "60_evaluate",
    "70_report",
)

STAGE_REQUIRED_FILES: dict[str, tuple[str, ...]] = {
    "00_input_inventory": (
        "config.toml",
        "normalized_config.json",
        "command_record.json",
        "input_inventory.json",
        "asset_inventory.json",
        "dependency_inventory.json",
        "disk_preflight.json",
    ),
    "10_prepare": ("tiles_index.csv", "prepare_summary.json"),
    "20_proposals_features": (
        "proposals.csv",
        "features.csv",
        "proposal_config.json",
        "proposal_summary.json",
        "review_request.csv",
    ),
    "30_review_import": ("reviewed_supplied.csv", "reviewed.csv", "review_import.json"),
    "40_group_split": ("split_manifest.json",),
    "50_train_bundle": (
        "test_predictions.csv",
        "training_result.json",
        "model_bundle/bundle.json",
        "model_bundle/classifier.ubj",
    ),
    "60_evaluate": ("predictions.csv", "inference_result.json", "evaluation_status.json"),
    "70_report": ("report.json", "report.md"),
}
CANONICAL_STAGE60_REQUIRED_FILES = (
    *STAGE_REQUIRED_FILES["60_evaluate"],
    "raw_features.csv",
)
_STAGE_REQUIRED_FILE_VARIANTS = {
    name: (
        (required, CANONICAL_STAGE60_REQUIRED_FILES)
        if name == "60_evaluate"
        else (required,)
    )
    for name, required in STAGE_REQUIRED_FILES.items()
}

_UUID_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
_ATTEMPT_PATTERN = re.compile(rf"({'|'.join(re.escape(name) for name in STAGE_NAMES)})\.({_UUID_PATTERN})")
_POSTPUBLISH_PATTERN = re.compile(
    rf"({'|'.join(re.escape(name) for name in STAGE_NAMES)})\.postpublish-failure\.({_UUID_PATTERN})"
)
_EVENT_TEMP_PATTERN = re.compile(rf"event\.(\d{{8}})\.({_UUID_PATTERN})\.json")
_EVENT_CLASSIFICATION_PATTERN = re.compile(rf"event\.(\d{{8}})\.({_UUID_PATTERN})\.json\.classification\.json")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _event_payload(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise PublicIOError("event cannot be represented as strict JSON") from exc
    return (rendered + "\n").encode("ascii")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


@dataclass(frozen=True)
class RunBindings:
    config_sha256: str
    input_manifest_sha256: str
    dependency_sha256: str
    asset_sha256: str

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise PublicIOError("run binding must be lowercase SHA-256")


@dataclass(frozen=True)
class StageWorkspace:
    run_id: str
    name: str
    root: Path
    target: Path


class RunState:
    def __init__(self, root: Path, run_id: str, bindings: RunBindings, lock_fd: int, lock_path: Path) -> None:
        self.root = root
        self.run_id = run_id
        self.bindings = bindings
        self._lock_fd = lock_fd
        self.lock_path = lock_path

    @property
    def stages(self) -> Path:
        return self.root / "stages"

    def close(self) -> None:
        if self._lock_fd >= 0:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = -1

    def __enter__(self) -> "RunState":
        return self

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        self.close()

    def append_event(self, event: str, **fields: object) -> None:
        self.require_active_mutation()
        reserved = {"sequence", "previous_event_sha256", "timestamp_utc", "run_id", "event", "event_sha256"}
        _require(not (reserved & set(fields)), "event fields attempt to override reserved keys")
        event_root = self.root / "events"
        info = event_root.lstat()
        _require(
            stat.S_ISDIR(info.st_mode) and not event_root.is_symlink() and stat.S_IMODE(info.st_mode) == 0o755,
            "run event directory is unsafe",
        )
        sequence, previous = _event_state(event_root, self.run_id)
        row = {
            "sequence": sequence,
            "previous_event_sha256": previous,
            "timestamp_utc": _now(),
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        row["event_sha256"] = compact_json_sha256(row)
        payload = _event_payload(row)
        staged = self.root / ".staging" / f"event.{sequence:08d}.{uuid.uuid4()}.json"
        write_new_bytes(staged, payload)
        rename_noreplace(staged, event_root / f"{sequence:08d}.json")

    def require_active_mutation(self) -> None:
        _require(self._lock_fd >= 0, "run state is closed and cannot be mutated")
        _require(
            not (self.root / "FINAL").exists() and not (self.root / "FINAL").is_symlink(),
            "finalized run cannot be mutated",
        )


@dataclass
class ReviewRunSnapshot:
    """Read-only, shared-locked view of a run paused at the review boundary."""

    root: Path
    run_id: str
    bindings: RunBindings
    review_request: Path
    review_request_sha256: str
    stage10_receipt_sha256: str
    stage20_receipt_sha256: str
    _lock_fd: int

    def close(self) -> None:
        if self._lock_fd >= 0:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = -1

    def __enter__(self) -> "ReviewRunSnapshot":
        return self

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        self.close()


def _canonical_output(path: Path, *, must_exist: bool) -> Path:
    absolute = path.absolute()
    parent = absolute.parent.resolve(strict=True)
    canonical = parent / absolute.name
    _require(absolute == canonical and absolute.name not in {"", ".", ".."}, "output path has a symlinked or unsafe component")
    portable_basename(absolute.name, "run output name")
    _require(len(f".{absolute.name}.run-init.{uuid.uuid4()}".encode("ascii")) <= 255, "run output name is too long for atomic initialization")
    if must_exist:
        info = canonical.lstat()
        _require(
            stat.S_ISDIR(info.st_mode)
            and not canonical.is_symlink()
            and stat.S_IMODE(info.st_mode) == 0o755
            and info.st_uid == os.getuid()
            and info.st_gid == os.getgid(),
            "run output is unsafe",
        )
    return canonical


def _lock_path(output: Path) -> Path:
    return output / ".COMPAG_RUN_LOCK"


def validate_new_run_target(path: Path) -> Path:
    output = _canonical_output(path, must_exist=False)
    _require(not output.exists() and not output.is_symlink(), f"output already exists: {output}")
    return output


def _open_lock(output: Path, *, create: bool) -> tuple[int, Path]:
    lock = _lock_path(output)
    if create:
        fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.fsync(fd)
            fsync_directory(lock.parent)
        except BaseException:
            os.close(fd)
            raise
        return fd, lock
    info = lock.lstat()
    _require(
        stat.S_ISREG(info.st_mode) and not lock.is_symlink() and info.st_nlink == 1 and stat.S_IMODE(info.st_mode) == 0o600,
        "run lock is unsafe",
    )
    fd = os.open(lock, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        _require(
            (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_uid, opened.st_gid, opened.st_nlink, opened.st_size)
            == (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink, info.st_size),
            "run lock changed while opening",
        )
    except BaseException:
        os.close(fd)
        raise
    return fd, lock


def _write_lock_marker(fd: int, value: object) -> None:
    payload = canonical_json_bytes(value)
    _require(os.lseek(fd, 0, os.SEEK_END) == 0, "new run lock is not empty")
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        _require(written > 0, "short write to run lock")
        offset += written
    os.fsync(fd)


def _read_lock_marker(fd: int) -> dict[str, object]:
    os.lseek(fd, 0, os.SEEK_SET)
    payload = b""
    while True:
        chunk = os.read(fd, 64 * 1024)
        if not chunk:
            break
        payload += chunk
        _require(len(payload) <= 1024 * 1024, "run lock marker is too large")
    try:
        value = canonical_json_value(payload, "run lock marker")
    except PublicIOError as exc:
        raise PublicIOError("run lock marker cannot be parsed") from exc
    _require(
        isinstance(value, dict)
        and set(value) == {
            "schema", "run_id", "output_name", "output_device", "output_inode", "lock_device", "lock_inode",
        }
        and value.get("schema") == "compag-curation-public-run-lock/v1",
        "run lock marker schema mismatch",
    )
    return value


def _event_records(path: Path, run_id: str) -> list[dict[str, object]]:
    info = path.lstat()
    _require(stat.S_ISDIR(info.st_mode) and not path.is_symlink(), "run event directory is unsafe")
    entries = sorted(path.iterdir(), key=lambda item: item.name)
    previous: str | None = None
    records: list[dict[str, object]] = []
    for sequence, event_path in enumerate(entries):
        _require(event_path.name == f"{sequence:08d}.json", "run event sequence filename changed")
        event_info = event_path.lstat()
        _require(
            stat.S_ISREG(event_info.st_mode) and not event_path.is_symlink() and event_info.st_nlink == 1 and stat.S_IMODE(event_info.st_mode) == 0o644,
            "run event record is unsafe",
        )
        payload, _snapshot = stable_file(event_path, max_bytes=1024 * 1024)
        _require(payload.endswith(b"\n") and payload.count(b"\n") == 1, "run event record framing is invalid")
        try:
            row = strict_json_bytes(payload, "run event record")
        except PublicIOError as exc:
            raise PublicIOError("run event record is malformed") from exc
        _require(isinstance(row, dict), "run event row must be an object")
        _require(payload == _event_payload(row), "run event record is not canonical compact JSON")
        claimed = row.get("event_sha256")
        unhashed = {key: value for key, value in row.items() if key != "event_sha256"}
        _require(unhashed.get("sequence") == sequence, "run event sequence changed")
        _require(unhashed.get("previous_event_sha256") == previous, "run event chain changed")
        _require(unhashed.get("run_id") == run_id, "run event belongs to another run")
        actual = compact_json_sha256(unhashed)
        _require(claimed == actual, "run event hash changed")
        previous = actual
        records.append(dict(row))
    return records


def _event_state(path: Path, run_id: str) -> tuple[int, str | None]:
    records = _event_records(path, run_id)
    previous = None if not records else str(records[-1]["event_sha256"])
    return len(records), previous


def _validate_genesis_event(
    records: Sequence[dict[str, object]],
    run_id: str,
    bindings: RunBindings,
) -> None:
    _require(bool(records), "run event ledger is missing its genesis record")
    genesis = records[0]
    _require(
        set(genesis)
        == {
            "sequence", "previous_event_sha256", "timestamp_utc", "run_id", "event",
            "bindings", "event_sha256",
        }
        and genesis.get("sequence") == 0
        and genesis.get("previous_event_sha256") is None
        and genesis.get("run_id") == run_id
        and genesis.get("event") == "RUN_CREATED"
        and genesis.get("bindings") == _binding_payload(bindings)
        and isinstance(genesis.get("timestamp_utc"), str)
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", str(genesis["timestamp_utc"])) is not None,
        "run event genesis record changed",
    )


def _acquire(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise PublicIOError("RUN_ALREADY_ACTIVE: another process owns this run") from exc


def _classify_staged_events(root: Path, run_id: str) -> None:
    staging = root / ".staging"
    pattern = re.compile(r"event\.(\d{8})\.[0-9a-f-]{36}\.json")
    records = _event_records(root / "events", run_id)
    next_sequence = len(records)
    previous = None if not records else str(records[-1]["event_sha256"])
    for path in sorted(staging.iterdir(), key=lambda item: item.name):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        info = path.lstat()
        _require(
            stat.S_ISREG(info.st_mode) and not path.is_symlink() and info.st_nlink == 1 and stat.S_IMODE(info.st_mode) == 0o644,
            "staged event diagnostic is unsafe",
        )
        payload, snapshot = stable_file(path, max_bytes=1024 * 1024)
        classification = "PARTIAL_OR_INVALID_EVENT"
        row: object = None
        try:
            row = (
                strict_json_bytes(payload, "staged event diagnostic")
                if payload.endswith(b"\n") and payload.count(b"\n") == 1
                else None
            )
            if isinstance(row, dict) and payload == _event_payload(row):
                claimed = row.get("event_sha256")
                unhashed = {key: value for key, value in row.items() if key != "event_sha256"}
                if (
                    row.get("sequence") == int(match.group(1))
                    and row.get("sequence") == next_sequence
                    and row.get("previous_event_sha256") == previous
                    and row.get("run_id") == run_id
                    and claimed == compact_json_sha256(unhashed)
                ):
                    classification = "UNPUBLISHED_COMPLETE_EVENT"
        except (PublicIOError, TypeError, ValueError):
            pass
        record = {
            "schema": "compag-curation-staged-event-diagnostic/v1",
            "status": "PRESERVED_AND_CLASSIFIED",
            "run_id": run_id,
            "staged_name": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": snapshot.st_size,
            "ledger_sequence_at_classification": next_sequence,
            "ledger_tip_sha256_at_classification": previous,
            "classification": classification,
        }
        sidecar = staging / f"{path.name}.classification.json"
        if sidecar.exists() or sidecar.is_symlink():
            sidecar_payload, _sidecar_snapshot = stable_file(sidecar, max_bytes=1024 * 1024)
            try:
                observed = canonical_json_value(sidecar_payload, "staged event classification")
            except PublicIOError as exc:
                raise PublicIOError("staged event classification is malformed") from exc
            _require(
                isinstance(observed, dict)
                and set(observed)
                == {
                    "schema", "status", "run_id", "staged_name", "sha256", "size_bytes",
                    "ledger_sequence_at_classification", "ledger_tip_sha256_at_classification", "classification",
                }
                and observed.get("schema") == "compag-curation-staged-event-diagnostic/v1"
                and observed.get("status") == "PRESERVED_AND_CLASSIFIED"
                and observed.get("run_id") == run_id
                and observed.get("staged_name") == path.name
                and observed.get("sha256") == hashlib.sha256(payload).hexdigest()
                and observed.get("size_bytes") == snapshot.st_size
                and isinstance(observed.get("ledger_sequence_at_classification"), int)
                and not isinstance(observed.get("ledger_sequence_at_classification"), bool)
                and int(observed["ledger_sequence_at_classification"]) >= 0
                and (
                    observed.get("ledger_tip_sha256_at_classification") is None
                    or (
                        isinstance(observed.get("ledger_tip_sha256_at_classification"), str)
                        and re.fullmatch(r"[0-9a-f]{64}", str(observed["ledger_tip_sha256_at_classification"])) is not None
                    )
                )
                and observed.get("classification") in {"PARTIAL_OR_INVALID_EVENT", "UNPUBLISHED_COMPLETE_EVENT"},
                "staged event classification changed",
            )
            if observed.get("classification") == "UNPUBLISHED_COMPLETE_EVENT":
                _require(isinstance(row, dict) and payload == _event_payload(row), "staged complete-event evidence changed")
                claimed = row.get("event_sha256")
                unhashed = {key: value for key, value in row.items() if key != "event_sha256"}
                _require(
                    row.get("sequence") == observed.get("ledger_sequence_at_classification")
                    and row.get("sequence") == int(match.group(1))
                    and row.get("previous_event_sha256") == observed.get("ledger_tip_sha256_at_classification")
                    and row.get("run_id") == run_id
                    and claimed == compact_json_sha256(unhashed),
                    "staged complete-event historical basis changed",
                )
            continue
        temporary = root.parent / f".compag-event-classification-init.{run_id}.{uuid.uuid4()}.json"
        write_new_json(temporary, record)
        publish_file_noreplace(temporary, sidecar)


def _read_diagnostic_json(path: Path, role: str) -> dict[str, object]:
    payload, _snapshot = stable_file(path, max_bytes=1024 * 1024)
    try:
        value = canonical_json_value(payload, role)
    except PublicIOError as exc:
        raise PublicIOError(f"{role} is malformed") from exc
    _require(isinstance(value, dict), f"{role} must be an object")
    return value


def _validate_failure_record(path: Path, run_id: str, stage: str) -> None:
    info = path.lstat()
    _require(
        stat.S_ISREG(info.st_mode)
        and not path.is_symlink()
        and info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == 0o644,
        "stage failure diagnostic is unsafe",
    )
    record = _read_diagnostic_json(path, "stage failure diagnostic")
    _require(
        set(record)
        == {
            "schema",
            "status",
            "run_id",
            "stage",
            "failed_at_utc",
            "error_type",
            "error_code",
            "remediation",
        }
        and record.get("schema") == "compag-curation-public-stage-failure/v1"
        and record.get("status") == "FAILED_DIAGNOSTIC_PRESERVED"
        and record.get("run_id") == run_id
        and record.get("stage") == stage
        and record.get("error_code") == "STAGE_EXECUTION_FAILED"
        and isinstance(record.get("failed_at_utc"), str)
        and isinstance(record.get("error_type"), str)
        and isinstance(record.get("remediation"), str),
        "stage failure diagnostic identity changed",
    )


def _validate_attempt_tree(path: Path, run_id: str, stage: str, *, postpublish: bool) -> set[str]:
    root_info = path.lstat()
    _require(
        stat.S_ISDIR(root_info.st_mode)
        and not path.is_symlink()
        and stat.S_IMODE(root_info.st_mode) == 0o700
        and root_info.st_uid == os.getuid(),
        "stage diagnostic attempt root is unsafe",
    )
    attempt = path / "_ATTEMPT.json"
    failure = path / "_FAILURE.json"
    if postpublish:
        _require(not attempt.exists() and not attempt.is_symlink(), "post-publish failure contains an attempt marker")
        _require(failure.exists() and not failure.is_symlink(), "post-publish failure record is missing")
    else:
        info = attempt.lstat()
        _require(
            stat.S_ISREG(info.st_mode)
            and not attempt.is_symlink()
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) == 0o644,
            "stage attempt marker is unsafe",
        )
        record = _read_diagnostic_json(attempt, "stage attempt marker")
        _require(
            set(record) == {"schema", "status", "run_id", "stage", "started_at_utc"}
            and record.get("schema") == "compag-curation-public-stage-attempt/v1"
            and record.get("status") == "WRITING"
            and record.get("run_id") == run_id
            and record.get("stage") == stage
            and isinstance(record.get("started_at_utc"), str),
            "stage attempt marker identity changed",
        )
    if failure.exists() or failure.is_symlink():
        _validate_failure_record(failure, run_id, stage)

    empty_directories: set[str] = set()
    for current, directories, names in os.walk(path, topdown=True, followlinks=False):
        current_path = Path(current)
        directories.sort()
        names.sort()
        current_info = current_path.lstat()
        _require(
            stat.S_ISDIR(current_info.st_mode)
            and not current_path.is_symlink()
            and current_info.st_uid == root_info.st_uid
            and stat.S_IMODE(current_info.st_mode) & 0o022 == 0,
            "stage diagnostic contains an unsafe directory",
        )
        if current_path != path and not directories and not names:
            empty_directories.add(current_path.relative_to(path.parent.parent).as_posix())
        for name in directories:
            child = current_path / name
            child_info = child.lstat()
            _require(
                stat.S_ISDIR(child_info.st_mode)
                and not child.is_symlink()
                and child_info.st_uid == root_info.st_uid
                and stat.S_IMODE(child_info.st_mode) & 0o022 == 0,
                "stage diagnostic contains an unsafe directory",
            )
        for name in names:
            child = current_path / name
            child_info = child.lstat()
            _require(
                stat.S_ISREG(child_info.st_mode)
                and not child.is_symlink()
                and child_info.st_nlink == 1
                and child_info.st_uid == root_info.st_uid
                and stat.S_IMODE(child_info.st_mode) & 0o022 == 0,
                "stage diagnostic contains an unsafe file",
            )
    return empty_directories


def validate_run_namespace(root: Path, run_id: str, *, allow_final: bool) -> set[str]:
    """Validate the complete owned run namespace and return declared empty directories."""
    root_info = root.lstat()
    _require(
        stat.S_ISDIR(root_info.st_mode)
        and not root.is_symlink()
        and stat.S_IMODE(root_info.st_mode) == 0o755
        and root_info.st_uid == os.getuid()
        and root_info.st_gid == os.getgid(),
        "run root metadata changed",
    )
    required = {".COMPAG_RUN_LOCK", "RUN_OWNER.json", ".staging", "events", "stages"}
    allowed = set(required)
    if allow_final:
        allowed.add("FINAL")
    observed = {path.name for path in root.iterdir()}
    _require(required <= observed and observed <= allowed, "run top-level namespace closure changed")

    for relative, mode in (("stages", 0o755), (".staging", 0o700), ("events", 0o755)):
        directory = root / relative
        info = directory.lstat()
        _require(
            stat.S_ISDIR(info.st_mode)
            and not directory.is_symlink()
            and stat.S_IMODE(info.st_mode) == mode
            and info.st_uid == os.getuid()
            and info.st_gid == os.getgid(),
            f"run {relative} directory is unsafe",
        )
    if allow_final and "FINAL" in observed:
        info = (root / "FINAL").lstat()
        _require(
            stat.S_ISDIR(info.st_mode)
            and not (root / "FINAL").is_symlink()
            and stat.S_IMODE(info.st_mode) == 0o755
            and info.st_uid == os.getuid()
            and info.st_gid == os.getgid(),
            "completed-run evidence directory metadata changed",
        )

    stages = root / "stages"
    stage_entries = sorted(stages.iterdir(), key=lambda item: item.name)
    for path in stage_entries:
        _require(path.name in STAGE_NAMES, "run contains an unknown completed stage")
        info = path.lstat()
        _require(
            stat.S_ISDIR(info.st_mode)
            and not path.is_symlink()
            and stat.S_IMODE(info.st_mode) == 0o700
            and info.st_uid == os.getuid()
            and info.st_gid == os.getgid(),
            "completed stage node is unsafe",
        )

    staging = root / ".staging"
    entries = sorted(staging.iterdir(), key=lambda item: item.name)
    empty_directories: set[str] = set()
    if not stage_entries:
        empty_directories.add("stages")
    if not entries:
        empty_directories.add(".staging")
    names = {path.name for path in entries}
    for path in entries:
        info = path.lstat()
        attempt = _ATTEMPT_PATTERN.fullmatch(path.name)
        postpublish = _POSTPUBLISH_PATTERN.fullmatch(path.name)
        event_temp = _EVENT_TEMP_PATTERN.fullmatch(path.name)
        classification = _EVENT_CLASSIFICATION_PATTERN.fullmatch(path.name)
        if attempt is not None or postpublish is not None:
            _require(stat.S_ISDIR(info.st_mode) and not path.is_symlink(), "stage diagnostic node is unsafe")
            match = attempt if attempt is not None else postpublish
            assert match is not None
            empty_directories.update(
                _validate_attempt_tree(path, run_id, match.group(1), postpublish=postpublish is not None)
            )
            continue
        _require(event_temp is not None or classification is not None, "run staging namespace contains an unknown node")
        _require(
            stat.S_ISREG(info.st_mode)
            and not path.is_symlink()
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) == 0o644,
            "staged event diagnostic node is unsafe",
        )
        if classification is not None:
            source_name = path.name.removesuffix(".classification.json")
            _require(source_name in names, "staged event classification has no source event")
    return empty_directories


def _validate_locked_path(fd: int, path: Path) -> os.stat_result:
    path_info = path.lstat()
    opened = os.fstat(fd)
    _require(
        stat.S_ISREG(path_info.st_mode)
        and not path.is_symlink()
        and path_info.st_nlink == 1
        and stat.S_IMODE(path_info.st_mode) == 0o600
        and path_info.st_uid == os.getuid()
        and path_info.st_gid == os.getgid()
        and (
            path_info.st_dev, path_info.st_ino, path_info.st_mode, path_info.st_uid,
            path_info.st_gid, path_info.st_nlink, path_info.st_size,
        ) == (
            opened.st_dev, opened.st_ino, opened.st_mode, opened.st_uid,
            opened.st_gid, opened.st_nlink, opened.st_size,
        ),
        "run lock path changed before or during acquisition",
    )
    return opened


def _owner(root: Path) -> dict[str, object]:
    path = root / "RUN_OWNER.json"
    info = path.lstat()
    _require(
        stat.S_ISREG(info.st_mode)
        and not path.is_symlink()
        and info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == 0o644
        and info.st_uid == os.getuid()
        and info.st_gid == os.getgid(),
        "run owner marker is unsafe",
    )
    try:
        payload, snapshot = stable_file(path, max_bytes=1024 * 1024)
        _require(
            (snapshot.st_dev, snapshot.st_ino, snapshot.st_mode, snapshot.st_uid, snapshot.st_gid, snapshot.st_nlink, snapshot.st_size)
            == (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink, info.st_size),
            "run owner marker changed while opening",
        )
        value = canonical_json_value(payload, "run owner marker")
    except (OSError, PublicIOError) as exc:
        raise PublicIOError("run owner marker cannot be parsed") from exc
    _require(
        isinstance(value, dict)
        and set(value) == {
            "schema", "status", "run_id", "created_at_utc", "output_name", "output_device",
            "output_inode", "lock_name", "lock_device", "lock_inode", "bindings", "stage_order",
            "output_policy", "input_policy",
        }
        and value.get("schema") == RUN_SCHEMA
        and value.get("status") == "OWNED"
        and value.get("output_policy") == "NEW_OR_VERIFIED_RESUME_ONLY"
        and value.get("input_policy") == "READ_ONLY_NO_IN_PLACE_CHANGES"
        and isinstance(value.get("created_at_utc"), str)
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", str(value["created_at_utc"])) is not None,
        "run owner schema mismatch",
    )
    return value


def _binding_payload(bindings: RunBindings) -> dict[str, str]:
    return {key: str(value) for key, value in asdict(bindings).items()}


def claim_run(output: Path, bindings: RunBindings) -> RunState:
    output = validate_new_run_target(output)
    _require(output.parent.exists() and output.parent.is_dir() and not output.parent.is_symlink(), "output parent is unsafe")
    run_id = str(uuid.uuid4())
    initialization = output.parent / f".{output.name}.run-init.{run_id}"
    initialization.mkdir(mode=0o700)
    fsync_directory(output.parent)
    lock_fd = -1
    try:
        (initialization / "stages").mkdir(mode=0o755)
        (initialization / ".staging").mkdir(mode=0o700)
        (initialization / "events").mkdir(mode=0o755)
        os.chmod(initialization / "stages", 0o755, follow_symlinks=False)
        os.chmod(initialization / ".staging", 0o700, follow_symlinks=False)
        os.chmod(initialization / "events", 0o755, follow_symlinks=False)
        lock_fd, initialization_lock = _open_lock(initialization, create=True)
        _acquire(lock_fd)
        lock_info = _validate_locked_path(lock_fd, initialization_lock)
        output_info = initialization.lstat()
        _require(stat.S_ISDIR(output_info.st_mode) and not initialization.is_symlink(), "new run initialization is unsafe")
        _write_lock_marker(
            lock_fd,
            {
                "schema": "compag-curation-public-run-lock/v1",
                "run_id": run_id,
                "output_name": output.name,
                "output_device": output_info.st_dev,
                "output_inode": output_info.st_ino,
                "lock_device": lock_info.st_dev,
                "lock_inode": lock_info.st_ino,
            },
        )
        write_new_json(
            initialization / "RUN_OWNER.json",
            {
                "schema": RUN_SCHEMA,
                "status": "OWNED",
                "run_id": run_id,
                "created_at_utc": _now(),
                "output_name": output.name,
                "output_device": output_info.st_dev,
                "output_inode": output_info.st_ino,
                "lock_name": initialization_lock.name,
                "lock_device": lock_info.st_dev,
                "lock_inode": lock_info.st_ino,
                "bindings": _binding_payload(bindings),
                "stage_order": list(STAGE_NAMES),
                "output_policy": "NEW_OR_VERIFIED_RESUME_ONLY",
                "input_policy": "READ_ONLY_NO_IN_PLACE_CHANGES",
            },
        )
        state = RunState(initialization, run_id, bindings, lock_fd, initialization_lock)
        state.append_event("RUN_CREATED", bindings=_binding_payload(bindings))
        os.chmod(initialization, 0o755, follow_symlinks=False)
        fsync_directory(initialization)
        publish_directory_noreplace(initialization, output)
        state.root = output
        state.lock_path = _lock_path(output)
        return state
    except BaseException:
        if lock_fd >= 0:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        raise


def _validate_stage(path: Path, expected_name: str, expected_run_id: str) -> dict[str, object]:
    _require(path.exists() and path.is_dir() and not path.is_symlink(), f"completed stage is unavailable: {expected_name}")
    success = path / "_SUCCESS.json"
    info = success.lstat()
    _require(
        stat.S_ISREG(info.st_mode) and not success.is_symlink() and info.st_nlink == 1 and stat.S_IMODE(info.st_mode) == 0o644,
        f"stage receipt is unsafe: {expected_name}",
    )
    try:
        payload, snapshot = stable_file(success, max_bytes=MAX_STAGE_RECEIPT_BYTES)
        _require(
            (snapshot.st_dev, snapshot.st_ino, snapshot.st_mode, snapshot.st_uid, snapshot.st_gid, snapshot.st_nlink, snapshot.st_size)
            == (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink, info.st_size),
            f"stage receipt changed while opening: {expected_name}",
        )
        record = canonical_json_value(payload, f"stage receipt {expected_name}")
    except (OSError, PublicIOError) as exc:
        raise PublicIOError(f"stage receipt cannot be parsed: {expected_name}") from exc
    _require(
        isinstance(record, dict)
        and set(record)
        == {
            "schema", "status", "run_id", "stage", "completed_at_utc", "required",
            "files", "files_sha256", "directories", "directories_sha256",
        }
        and record.get("schema") == STAGE_SCHEMA
        and record.get("status") == "PASS",
        f"stage receipt status mismatch: {expected_name}",
    )
    _require(record.get("stage") == expected_name, f"stage receipt identity mismatch: {expected_name}")
    _require(record.get("run_id") == expected_run_id, f"stage receipt belongs to another run: {expected_name}")
    declared_required = record.get("required")
    _require(
        isinstance(declared_required, list)
        and all(isinstance(value, str) for value in declared_required)
        and tuple(declared_required)
        in _STAGE_REQUIRED_FILE_VARIANTS[expected_name],
        f"stage required-output contract changed: {expected_name}",
    )
    observed, directories = owned_manifest_contract(path, excluded={"_SUCCESS.json"})
    _require(record.get("files") == observed, f"completed stage drift detected: {expected_name}")
    _require(record.get("files_sha256") == compact_json_sha256(observed), f"stage manifest hash mismatch: {expected_name}")
    _require(record.get("directories") == directories, f"completed stage directory metadata changed: {expected_name}")
    _require(
        record.get("directories_sha256") == compact_json_sha256(directories),
        f"stage directory manifest hash mismatch: {expected_name}",
    )
    observed_by_path = {str(row["path"]): row for row in observed}
    for required in declared_required:
        _require(
            required in observed_by_path and int(observed_by_path[required]["size_bytes"]) > 0,
            f"required stage output is missing or empty: {expected_name}/{required}",
        )
    return record


def _stage_receipt_anchors(records: Sequence[dict[str, object]]) -> dict[str, str]:
    anchors: dict[str, str] = {}
    for record in records:
        if record.get("event") not in {"STAGE_PUBLISHED", "STAGE_PUBLICATION_RECOVERED"}:
            continue
        stage = record.get("stage")
        receipt = record.get("receipt_sha256")
        _require(
            isinstance(stage, str)
            and stage in STAGE_NAMES
            and isinstance(receipt, str)
            and re.fullmatch(r"[0-9a-f]{64}", receipt) is not None,
            "stage publication event is malformed",
        )
        _require(stage not in anchors, f"stage has multiple publication anchors: {stage}")
        anchors[stage] = receipt
    return anchors


def validate_complete_stage_chain(state: RunState) -> None:
    observed_names = {path.name for path in state.stages.iterdir()}
    _require(observed_names == set(STAGE_NAMES), "run does not contain the exact completed stage set")
    records = _event_records(state.root / "events", state.run_id)
    _validate_genesis_event(records, state.run_id, state.bindings)
    anchors = _stage_receipt_anchors(records)
    _require(set(anchors) == set(STAGE_NAMES), "run stage publication ledger is incomplete")
    for stage in STAGE_NAMES:
        path = state.stages / stage
        _validate_stage(path, stage, state.run_id)
        _require(
            sha256_file(path / "_SUCCESS.json") == anchors[stage],
            f"stage receipt ledger anchor changed: {stage}",
        )


def open_resume(root: Path, bindings: RunBindings, *, record_event: bool = False) -> RunState:
    root = _canonical_output(root, must_exist=True)
    lock_fd, lock_path = _open_lock(root, create=False)
    try:
        _acquire(lock_fd)
        lock_info = _validate_locked_path(lock_fd, lock_path)
        owner = _owner(root)
        _require(owner.get("bindings") == _binding_payload(bindings), "RESUME_DRIFT: configuration, input, dependency, or asset identity changed")
        run_id = owner.get("run_id")
        _require(isinstance(run_id, str) and str(uuid.UUID(run_id)) == run_id, "run UUID is invalid")
        output_info = root.lstat()
        expected_identity = (
            root.name, output_info.st_dev, output_info.st_ino, lock_path.name, lock_info.st_dev, lock_info.st_ino,
        )
        _require(
            (
                owner.get("output_name"), owner.get("output_device"), owner.get("output_inode"), owner.get("lock_name"),
                owner.get("lock_device"), owner.get("lock_inode"),
            ) == expected_identity,
            "run owner output identity changed",
        )
        marker = _read_lock_marker(lock_fd)
        _require(
            (
                marker.get("run_id"), marker.get("output_name"), marker.get("output_device"), marker.get("output_inode"),
                marker.get("lock_device"), marker.get("lock_inode"),
            ) == (run_id, root.name, output_info.st_dev, output_info.st_ino, lock_info.st_dev, lock_info.st_ino),
            "run lock identity changed",
        )
        _require(owner.get("stage_order") == list(STAGE_NAMES), "run stage order changed")
        for relative, mode in (("stages", 0o755), (".staging", 0o700), ("events", 0o755)):
            directory = root / relative
            info = directory.lstat()
            _require(
                stat.S_ISDIR(info.st_mode) and not directory.is_symlink() and stat.S_IMODE(info.st_mode) == mode,
                f"run {relative} directory is unsafe",
            )
        validate_run_namespace(root, run_id, allow_final=True)
        event_records = _event_records(root / "events", run_id)
        _validate_genesis_event(event_records, run_id, bindings)
        _classify_staged_events(root, run_id)
        validate_run_namespace(root, run_id, allow_final=True)
        anchors = _stage_receipt_anchors(event_records)
        recoveries: list[tuple[str, str]] = []
        gap = False
        for stage in STAGE_NAMES:
            path = root / "stages" / stage
            if path.exists() or path.is_symlink():
                _require(not gap, "completed stages are not a contiguous prefix")
                _validate_stage(path, stage, run_id)
                receipt_sha256 = sha256_file(path / "_SUCCESS.json")
                if stage in anchors:
                    _require(anchors[stage] == receipt_sha256, f"stage receipt ledger anchor changed: {stage}")
                else:
                    _require(
                        not (root / "FINAL").exists() and not (root / "FINAL").is_symlink(),
                        f"completed stage lacks a publication anchor: {stage}",
                    )
                    recoveries.append((stage, receipt_sha256))
            else:
                gap = True
        state = RunState(root, run_id, bindings, lock_fd, lock_path)
        for stage, receipt_sha256 in recoveries:
            state.append_event(
                "STAGE_PUBLICATION_RECOVERED",
                stage=stage,
                receipt_sha256=receipt_sha256,
                recovery="ATOMIC_STAGE_PRESENT_AND_RECEIPT_VALIDATED",
            )
        if recoveries:
            validate_run_namespace(root, run_id, allow_final=True)
        if record_event:
            state.append_event("RUN_RESUMED", bindings=_binding_payload(bindings))
        return state
    except BaseException:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        raise


def open_review_snapshot(root: Path) -> ReviewRunSnapshot:
    """Open a verified run paused after Stage 20 without mutating its ledger.

    The shared lock is intentionally held for the lifetime of the returned
    object.  A scientific resume therefore cannot race a human-review session,
    while any number of read-only inspections may coexist.
    """

    root = _canonical_output(root, must_exist=True)
    lock_fd, lock_path = _open_lock(root, create=False)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PublicIOError(
                "RUN_ALREADY_ACTIVE: the run is being written or resumed"
            ) from exc
        lock_info = _validate_locked_path(lock_fd, lock_path)
        owner = _owner(root)
        run_id = owner.get("run_id")
        _require(
            isinstance(run_id, str) and str(uuid.UUID(run_id)) == run_id,
            "run UUID is invalid",
        )
        bindings_value = owner.get("bindings")
        _require(
            isinstance(bindings_value, dict)
            and set(bindings_value) == set(RunBindings.__dataclass_fields__),
            "run binding marker is malformed",
        )
        bindings = RunBindings(
            **{name: str(bindings_value[name]) for name in RunBindings.__dataclass_fields__}
        )
        _require(
            owner.get("bindings") == _binding_payload(bindings),
            "run binding marker changed",
        )
        output_info = root.lstat()
        expected_identity = (
            root.name,
            output_info.st_dev,
            output_info.st_ino,
            lock_path.name,
            lock_info.st_dev,
            lock_info.st_ino,
        )
        _require(
            (
                owner.get("output_name"),
                owner.get("output_device"),
                owner.get("output_inode"),
                owner.get("lock_name"),
                owner.get("lock_device"),
                owner.get("lock_inode"),
            )
            == expected_identity,
            "run owner output identity changed",
        )
        marker = _read_lock_marker(lock_fd)
        _require(
            (
                marker.get("run_id"),
                marker.get("output_name"),
                marker.get("output_device"),
                marker.get("output_inode"),
                marker.get("lock_device"),
                marker.get("lock_inode"),
            )
            == (
                run_id,
                root.name,
                output_info.st_dev,
                output_info.st_ino,
                lock_info.st_dev,
                lock_info.st_ino,
            ),
            "run lock identity changed",
        )
        _require(owner.get("stage_order") == list(STAGE_NAMES), "run stage order changed")
        validate_run_namespace(root, run_id, allow_final=False)

        observed_stages = {path.name for path in (root / "stages").iterdir()}
        expected_stages = set(STAGE_NAMES[:3])
        _require(
            observed_stages == expected_stages,
            "review UI requires a run paused after exactly Stage 20",
        )
        records = _event_records(root / "events", run_id)
        _validate_genesis_event(records, run_id, bindings)
        anchors = _stage_receipt_anchors(records)
        _require(
            set(anchors) == expected_stages,
            "paused run stage publication ledger is incomplete",
        )
        stage10_receipt_sha256 = ""
        stage20_receipt_sha256 = ""
        for stage in STAGE_NAMES[:3]:
            stage_path = root / "stages" / stage
            _validate_stage(stage_path, stage, run_id)
            receipt_sha256 = sha256_file(stage_path / "_SUCCESS.json")
            _require(
                anchors[stage] == receipt_sha256,
                f"stage receipt ledger anchor changed: {stage}",
            )
            if stage == "10_prepare":
                stage10_receipt_sha256 = receipt_sha256
            elif stage == "20_proposals_features":
                stage20_receipt_sha256 = receipt_sha256

        _require(bool(records), "paused run event ledger is empty")
        terminal = records[-1]
        _require(
            terminal.get("event") == "PAUSED_FOR_REVIEW"
            and terminal.get("run_id") == run_id,
            "run is not at the human-review pause boundary",
        )
        review_request = root / "stages" / "20_proposals_features" / "review_request.csv"
        review_request_sha256 = sha256_file(review_request)
        _require(
            terminal.get("review_request_sha256") == review_request_sha256,
            "review request differs from its pause event",
        )
        return ReviewRunSnapshot(
            root=root,
            run_id=run_id,
            bindings=bindings,
            review_request=review_request,
            review_request_sha256=review_request_sha256,
            stage10_receipt_sha256=stage10_receipt_sha256,
            stage20_receipt_sha256=stage20_receipt_sha256,
            _lock_fd=lock_fd,
        )
    except BaseException:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        raise


def begin_stage(state: RunState, name: str) -> StageWorkspace:
    state.require_active_mutation()
    _require(name in STAGE_NAMES, f"unknown stage: {name}")
    index = STAGE_NAMES.index(name)
    for predecessor in STAGE_NAMES[:index]:
        _validate_stage(state.stages / predecessor, predecessor, state.run_id)
    for successor in STAGE_NAMES[index + 1 :]:
        _require(not (state.stages / successor).exists() and not (state.stages / successor).is_symlink(), "later stage exists before its predecessor")
    target = state.stages / name
    _require(not target.exists() and not target.is_symlink(), f"stage already complete: {name}")
    attempt = state.root / ".staging" / f"{name}.{uuid.uuid4()}"
    initialization = state.root.parent / f".compag-stage-init.{state.run_id}.{STAGE_NAMES.index(name):02d}.{uuid.uuid4()}"
    initialization.mkdir(mode=0o700)
    fsync_directory(initialization.parent)
    write_new_json(
        initialization / "_ATTEMPT.json",
        {"schema": "compag-curation-public-stage-attempt/v1", "status": "WRITING", "run_id": state.run_id, "stage": name, "started_at_utc": _now()},
    )
    fsync_directory(initialization)
    rename_noreplace(initialization, attempt)
    state.append_event("STAGE_STARTED", stage=name, staging=attempt.name)
    return StageWorkspace(state.run_id, name, attempt, target)


def publish_stage(state: RunState, workspace: StageWorkspace, required: Sequence[str]) -> dict[str, object]:
    state.require_active_mutation()
    _require(workspace.run_id == state.run_id, "stage workspace belongs to another run")
    _require(workspace.name in STAGE_NAMES, "stage workspace name is unknown")
    _require(workspace.root.parent == state.root / ".staging", "stage workspace escaped the run")
    _require(workspace.target == state.stages / workspace.name, "stage workspace target is not canonical")
    _require(
        tuple(required) in _STAGE_REQUIRED_FILE_VARIANTS[workspace.name],
        f"required stage output contract mismatch: {workspace.name}",
    )
    for value in required:
        _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value) is not None and ".." not in Path(value).parts, f"invalid required stage path: {value}")
        path = workspace.root / value
        info = path.lstat()
        _require(stat.S_ISREG(info.st_mode) and not path.is_symlink() and info.st_size > 0, f"required stage output is missing or empty: {value}")
    rows, directories = owned_manifest_contract(workspace.root, excluded={"_SUCCESS.json"})
    receipt = {
        "schema": STAGE_SCHEMA,
        "status": "PASS",
        "run_id": state.run_id,
        "stage": workspace.name,
        "completed_at_utc": _now(),
        "required": list(required),
        "files": rows,
        "files_sha256": compact_json_sha256(rows),
        "directories": directories,
        "directories_sha256": compact_json_sha256(directories),
    }
    write_new_json(workspace.root / "_SUCCESS.json", receipt)
    fsync_directory(workspace.root)
    rename_noreplace(workspace.root, workspace.target)
    validated = _validate_stage(workspace.target, workspace.name, state.run_id)
    state.append_event("STAGE_PUBLISHED", stage=workspace.name, receipt_sha256=sha256_file(workspace.target / "_SUCCESS.json"))
    return validated


def completed_stage(state: RunState, name: str) -> dict[str, object] | None:
    path = state.stages / name
    if not path.exists() and not path.is_symlink():
        return None
    return _validate_stage(path, name, state.run_id)


def record_stage_failure(state: RunState, workspace: StageWorkspace, error: BaseException) -> dict[str, object]:
    state.require_active_mutation()
    payload = {
        "schema": "compag-curation-public-stage-failure/v1",
        "status": "FAILED_DIAGNOSTIC_PRESERVED",
        "run_id": state.run_id,
        "stage": workspace.name,
        "failed_at_utc": _now(),
        "error_type": type(error).__name__,
        "error_code": "STAGE_EXECUTION_FAILED",
        "remediation": "Inspect the private process stderr, correct the reported input or environment defect, and resume the owned run.",
    }
    diagnostic_root = workspace.root
    if not diagnostic_root.exists() and not diagnostic_root.is_symlink():
        diagnostic_root = state.root / ".staging" / f"{workspace.name}.postpublish-failure.{uuid.uuid4()}"
        initialization = state.root.parent / f".compag-stage-failure-init.{state.run_id}.{uuid.uuid4()}"
        try:
            initialization.mkdir(mode=0o700)
            fsync_directory(initialization.parent)
            write_new_json(initialization / "_FAILURE.json", payload)
            fsync_directory(initialization)
            publish_directory_noreplace(initialization, diagnostic_root)
        except BaseException:
            pass
    else:
        try:
            failure = diagnostic_root / "_FAILURE.json"
            if not failure.exists() and not failure.is_symlink():
                initialization = state.root.parent / f".compag-stage-failure-file-init.{state.run_id}.{uuid.uuid4()}.json"
                write_new_json(initialization, payload)
                publish_file_noreplace(initialization, failure)
        except BaseException:
            pass
    try:
        state.append_event(
            "STAGE_FAILED",
            stage=workspace.name,
            staging=diagnostic_root.name,
            output_already_published=workspace.target.exists(),
            error_type=type(error).__name__,
            error_code="STAGE_EXECUTION_FAILED",
        )
    except BaseException:
        pass
    return payload
