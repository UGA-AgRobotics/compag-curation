"""Stable proposal-review export/import boundary for public runs."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from compag_curation.public_io import PublicIOError, sha256_file, stable_file, write_new_bytes


REVIEW_COLUMNS = (
    "proposal_id",
    "proposal_sha256",
    "image_id",
    "image_sha256",
    "group_id",
    "label",
    "review_status",
)
CANONICAL_REVIEW_COLUMNS = (
    "proposal_id",
    "proposal_sha256",
    "image_id",
    "image_sha256",
    "group_id",
    "label",
    "review_action",
    "review_weight",
    "review_status",
)
REVIEW_ACTION_WEIGHTS = {
    "accept": 1.0,
    "flip": 1.0,
    "sus_accept": 0.4,
    "sus_flip": 0.4,
    "skip": 0.0,
}
_SHA = re.compile(r"[0-9a-f]{64}")
MAX_REVIEW_BYTES = 128 * 1024 * 1024
MAX_REVIEW_ROWS = 250_000


@dataclass(frozen=True)
class ReviewRow:
    proposal_id: str
    proposal_sha256: str
    image_id: str
    image_sha256: str
    group_id: str
    label: int
    review_status: str
    review_action: str | None = None
    review_weight: float | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


def _read(
    path: Path,
    *,
    expected_columns: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, str]], str, tuple[str, ...]]:
    path = path.absolute()
    _require(path == path.resolve(strict=True), "review table path contains a symlink component")
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and not path.is_symlink() and info.st_nlink == 1, f"review table must be a single-link regular file: {path.name}")
    _require(0 < info.st_size <= MAX_REVIEW_BYTES, f"review table exceeds the supported size bound: {path.name}")
    payload, snapshot = stable_file(path, max_bytes=MAX_REVIEW_BYTES)
    _require(
        (snapshot.st_dev, snapshot.st_ino, snapshot.st_mode, snapshot.st_uid, snapshot.st_gid, snapshot.st_nlink, snapshot.st_size, snapshot.st_mtime_ns, snapshot.st_ctime_ns)
        == (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns),
        f"review table changed before reading: {path.name}",
    )
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeError as exc:
        raise PublicIOError(f"review table is not UTF-8: {path}") from exc
    try:
        with io.StringIO(text, newline="") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            _require(
                columns in {REVIEW_COLUMNS, CANONICAL_REVIEW_COLUMNS},
                "review table header mismatch",
            )
            if expected_columns is not None:
                _require(columns == expected_columns, "reviewed table schema differs from its export")
            rows = []
            for row in reader:
                _require(len(rows) < MAX_REVIEW_ROWS, "review table exceeds the supported row bound")
                rows.append(row)
    except csv.Error as exc:
        raise PublicIOError("review table contains invalid CSV framing") from exc
    for index, row in enumerate(rows, start=2):
        _require(None not in row and tuple(row) == columns, f"review row {index} has the wrong field count")
        _require(all(isinstance(value, str) for value in row.values()), f"review row {index} contains a missing value")
    _require(rows, "review table is empty")
    return rows, hashlib.sha256(payload).hexdigest(), columns


def _identity(row: dict[str, str], index: int) -> tuple[str, str, str, str, str]:
    proposal = row["proposal_id"]
    _require(_SHA.fullmatch(proposal) is not None, f"row {index} proposal_id is not SHA-256")
    _require(row["proposal_sha256"] == proposal, f"row {index} proposal hash differs from ID")
    _require(_SHA.fullmatch(row["image_id"]) is not None, f"row {index} image_id is not SHA-256")
    _require(_SHA.fullmatch(row["image_sha256"]) is not None, f"row {index} image hash is invalid")
    _require(re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", row["group_id"]) is not None, f"row {index} group identity is not canonical")
    return proposal, row["proposal_sha256"], row["image_id"], row["image_sha256"], row["group_id"]


def write_review_export(
    proposal_rows: list[dict[str, str]],
    output: Path,
    *,
    canonical_actions: bool = False,
) -> dict[str, object]:
    _require(proposal_rows, "cannot export an empty proposal set")
    _require(len(proposal_rows) <= MAX_REVIEW_ROWS, "review export exceeds the supported row bound")
    output = output.absolute()
    _require(not output.exists() and not output.is_symlink(), "review export already exists")
    identities: set[str] = set()
    ordered = sorted(proposal_rows, key=lambda row: row["proposal_id"])
    try:
        with io.StringIO(newline="") as handle:
            columns = CANONICAL_REVIEW_COLUMNS if canonical_actions else REVIEW_COLUMNS
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for index, row in enumerate(ordered, start=1):
                identity = _identity(row, index)
                _require(identity[0] not in identities, f"duplicate proposal ID: {identity[0]}")
                identities.add(identity[0])
                row = {
                        "proposal_id": identity[0],
                        "proposal_sha256": identity[1],
                        "image_id": identity[2],
                        "image_sha256": identity[3],
                        "group_id": identity[4],
                        "label": "",
                        "review_status": "pending",
                }
                if canonical_actions:
                    row.update({"review_action": "", "review_weight": ""})
                writer.writerow(row)
            payload = handle.getvalue().encode("utf-8")
    except csv.Error as exc:
        raise PublicIOError("review export could not be encoded as CSV") from exc
    _require(len(payload) <= MAX_REVIEW_BYTES, "review export exceeds the supported size bound")
    write_new_bytes(output, payload)
    return {
        "status": "PAUSED_FOR_REVIEW",
        "rows": len(ordered),
        "sha256": sha256_file(output),
        "path": output.name,
        "review_contract": "CANONICAL_ACTION_WEIGHTED_V2" if canonical_actions else "BINARY_V1",
    }


def read_review_export(
    path: Path,
) -> tuple[tuple[dict[str, str], ...], dict[str, object]]:
    """Read and validate one immutable pending review export.

    This is the supported read boundary used by interactive review clients.  It
    deliberately accepts only the same two schemas as :func:`validate_review_table`
    and requires every mutable field to still be in its original pending state.
    """

    rows, digest, columns = _read(path)
    canonical_actions = columns == CANONICAL_REVIEW_COLUMNS
    seen: set[str] = set()
    normalized: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        identity = _identity(row, index)
        _require(identity[0] not in seen, "exported review table contains duplicate IDs")
        seen.add(identity[0])
        _require(
            row["label"] == "" and row["review_status"] == "pending",
            "exported review state is not pending",
        )
        if canonical_actions:
            _require(
                row["review_action"] == "" and row["review_weight"] == "",
                "exported canonical review action is not pending",
            )
        normalized.append({name: row[name] for name in columns})
    return tuple(normalized), {
        "status": "PASS",
        "rows": len(normalized),
        "sha256": digest,
        "columns": list(columns),
        "review_contract": (
            "CANONICAL_ACTION_WEIGHTED_V2" if canonical_actions else "BINARY_V1"
        ),
    }


def validate_review_table(exported: Path, reviewed: Path) -> tuple[tuple[ReviewRow, ...], dict[str, object]]:
    _require(exported.resolve(strict=True) != reviewed.resolve(strict=True), "reviewed table must be a separate file")
    pending, pending_sha, columns = _read(exported)
    supplied, supplied_sha, supplied_columns = _read(reviewed, expected_columns=columns)
    _require(supplied_columns == columns, "review table schema mismatch")
    canonical_actions = columns == CANONICAL_REVIEW_COLUMNS
    pending_by_id: dict[str, tuple[str, str, str, str, str]] = {}
    for index, row in enumerate(pending, start=1):
        identity = _identity(row, index)
        _require(row["label"] == "" and row["review_status"] == "pending", "exported review state is not pending")
        if canonical_actions:
            _require(row["review_action"] == "" and row["review_weight"] == "", "exported canonical review action is not pending")
        _require(identity[0] not in pending_by_id, "exported review table contains duplicate IDs")
        pending_by_id[identity[0]] = identity
    reviewed_rows: list[ReviewRow] = []
    seen: set[str] = set()
    for index, row in enumerate(supplied, start=1):
        identity = _identity(row, index)
        proposal = identity[0]
        _require(proposal not in seen, f"reviewed table contains duplicate ID: {proposal}")
        seen.add(proposal)
        _require(proposal in pending_by_id, f"reviewed table contains unknown ID: {proposal}")
        _require(identity == pending_by_id[proposal], f"reviewed table contains stale identity: {proposal}")
        _require(row["label"] in {"0", "1"}, f"reviewed label must be 0 or 1: {proposal}")
        _require(row["review_status"] == "reviewed", f"review status must be reviewed: {proposal}")
        action: str | None = None
        weight: float | None = None
        if canonical_actions:
            action = row["review_action"]
            _require(action in REVIEW_ACTION_WEIGHTS, f"review action is invalid: {proposal}")
            expected_weight = REVIEW_ACTION_WEIGHTS[action]
            _require(
                row["review_weight"] == format(expected_weight, ".1f"),
                f"review weight differs from the canonical action weight: {proposal}",
            )
            weight = expected_weight
        reviewed_rows.append(
            ReviewRow(*identity, int(row["label"]), row["review_status"], action, weight)
        )
    missing = set(pending_by_id) - seen
    _require(not missing, f"reviewed table is missing {len(missing)} proposal IDs")
    action_counts = {
        action: sum(row.review_action == action for row in reviewed_rows)
        for action in REVIEW_ACTION_WEIGHTS
    } if canonical_actions else None
    return tuple(sorted(reviewed_rows, key=lambda row: row.proposal_id)), {
        "status": "PASS",
        "rows": len(reviewed_rows),
        "pending_sha256": pending_sha,
        "reviewed_sha256": supplied_sha,
        "positive_rows": sum(row.label == 1 for row in reviewed_rows),
        "negative_rows": sum(row.label == 0 for row in reviewed_rows),
        "review_contract": "CANONICAL_ACTION_WEIGHTED_V2" if canonical_actions else "BINARY_V1",
        "action_counts": action_counts,
        "effective_weight": (
            sum(float(row.review_weight or 0.0) for row in reviewed_rows)
            if canonical_actions else None
        ),
    }


def write_review_import(rows: tuple[ReviewRow, ...], output: Path) -> dict[str, object]:
    _require(rows, "cannot import an empty reviewed table")
    _require(len(rows) <= MAX_REVIEW_ROWS, "review import exceeds the supported row bound")
    output = output.absolute()
    _require(not output.exists() and not output.is_symlink(), "review import output already exists")
    try:
        with io.StringIO(newline="") as handle:
            canonical_actions = all(
                row.review_action is not None and row.review_weight is not None for row in rows
            )
            _require(
                canonical_actions or all(
                    row.review_action is None and row.review_weight is None for row in rows
                ),
                "review rows mix binary and canonical action contracts",
            )
            columns = CANONICAL_REVIEW_COLUMNS if canonical_actions else REVIEW_COLUMNS
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                value = {
                        "proposal_id": row.proposal_id,
                        "proposal_sha256": row.proposal_sha256,
                        "image_id": row.image_id,
                        "image_sha256": row.image_sha256,
                        "group_id": row.group_id,
                        "label": row.label,
                        "review_status": row.review_status,
                }
                if canonical_actions:
                    _require(row.review_action in REVIEW_ACTION_WEIGHTS, "canonical review action is invalid")
                    _require(
                        row.review_weight == REVIEW_ACTION_WEIGHTS[row.review_action],
                        "canonical review action weight changed",
                    )
                    value.update({
                        "review_action": row.review_action,
                        "review_weight": format(float(row.review_weight), ".1f"),
                    })
                writer.writerow(value)
            payload = handle.getvalue().encode("utf-8")
    except csv.Error as exc:
        raise PublicIOError("review import could not be encoded as CSV") from exc
    _require(len(payload) <= MAX_REVIEW_BYTES, "review import exceeds the supported size bound")
    write_new_bytes(output, payload)
    return {
        "status": "PASS",
        "rows": len(rows),
        "sha256": sha256_file(output),
        "path": output.name,
        "review_contract": "CANONICAL_ACTION_WEIGHTED_V2" if canonical_actions else "BINARY_V1",
    }
