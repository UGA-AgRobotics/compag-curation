"""Dynamic source inventory and raw-embedding diagnostics for Amendment 01.

This module is intentionally independent of the preflight builder.  It performs
read-only discovery and returns ordinary dictionaries/data frames that can be
snapshotted in a Run and compared before and after a workflow.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class EvidenceClassification(str, Enum):
    """The only evidence classifications allowed by Amendment 01."""

    VERIFIED = "VERIFIED"
    INFERRED = "INFERRED"
    UNRESOLVED = "UNRESOLVED"
    BLOCKED = "BLOCKED"


EVIDENCE_CLASSIFICATIONS = frozenset(item.value for item in EvidenceClassification)
TIMESTAMPED_CARD_RE = re.compile(r"^(IMG_(\d+))\s+\(([^)]+)\)$")
CARD_RE = re.compile(r"^IMG_(\d+)$")
TRANSFORM_ARTIFACT_NAMES = (
    "embed_pca_32.npz",
    "foldsafe_pack.joblib",
    "jassid_unified_proto.csv",
)
SIDECAR_SUFFIXES = frozenset({".csv", ".tsv", ".json", ".jsonl", ".txt", ".parquet", ".npy", ".npz"})
SIDECAR_NAME_RE = re.compile(r"(?:^|[_-])(key|keys|id|ids|index|indices|order|rows?|manifest|metadata)(?:[_-]|$)", re.I)
DEFAULT_KEY_SETS = (
    frozenset({"file_name", "ann_id", "scale"}),
    frozenset({"card_id", "image", "id", "scale"}),
    frozenset({"img_folder", "image", "id", "scale"}),
)
ROW_INDEX_COLUMNS = (
    "raw_row_index", "cache_row_index", "embedding_row_index", "row_index",
    "array_index", "offset",
)
RAW_MAPPING_CHECK_KEYS = frozenset({
    "two_dimensional", "expected_dtype", "expected_dimension_count",
    "cache_rows_divisible_by_raw_scale_count", "candidate_count_equal",
    "scale_sequence_equal", "stable_table_key_nonnull",
    "stable_table_key_unique", "qualifying_stable_key_sidecar",
    "actual_table_join_complete", "actual_cache_key_coverage_complete",
    "actual_mapping_has_no_cache_collisions",
    "unambiguous_stable_key_mapping",
    "verified_embedding_semantics_manifest",
})


def _classification(value: EvidenceClassification | str) -> str:
    normalized = value.value if isinstance(value, EvidenceClassification) else str(value)
    if normalized not in EVIDENCE_CLASSIFICATIONS:
        raise ValueError(f"Invalid evidence classification: {value!r}")
    return normalized


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class InventoryRecord:
    logical_name: str
    path: str
    role: str
    exists: bool
    size_bytes: int | None
    sha256: str | None
    evidence_classification: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_classification", _classification(self.evidence_classification))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["metadata"] = json.dumps(result["metadata"], sort_keys=True, separators=(",", ":"))
        return result


def inventory_record(
    path: Path,
    *,
    logical_name: str,
    role: str,
    metadata: Mapping[str, Any] | None = None,
) -> InventoryRecord:
    path = Path(path)
    exists = path.is_file()
    return InventoryRecord(
        logical_name=logical_name,
        path=str(path.resolve(strict=False)),
        role=role,
        exists=exists,
        size_bytes=path.stat().st_size if exists else None,
        sha256=sha256_file(path) if exists else None,
        evidence_classification=(
            EvidenceClassification.VERIFIED.value if exists else EvidenceClassification.BLOCKED.value
        ),
        metadata=dict(metadata or {}),
    )


def inventory_frame(records: Iterable[InventoryRecord | Mapping[str, Any]]) -> pd.DataFrame:
    rows = [record.to_dict() if isinstance(record, InventoryRecord) else dict(record) for record in records]
    columns = [
        "logical_name", "path", "role", "exists", "size_bytes", "sha256",
        "evidence_classification", "metadata",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    invalid = sorted(set(frame.evidence_classification.astype(str)) - EVIDENCE_CLASSIFICATIONS)
    if invalid:
        raise ValueError(f"Inventory contains invalid evidence classifications: {invalid}")
    return frame.sort_values(["logical_name", "path"], kind="stable").reset_index(drop=True)[columns]


def compare_inventory_frames(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, Any]:
    """Compare the immutable fields of two source inventories."""

    columns = ["logical_name", "path", "exists", "size_bytes", "sha256"]
    left = before[columns].sort_values(["logical_name", "path"], kind="stable").reset_index(drop=True)
    right = after[columns].sort_values(["logical_name", "path"], kind="stable").reset_index(drop=True)
    equal = left.equals(right)
    return {
        "status": "PASS" if equal else "FAIL",
        "equal": equal,
        "before_rows": len(left),
        "after_rows": len(right),
        "before_sha256": sha256_bytes(left.to_csv(index=False, lineterminator="\n").encode()),
        "after_sha256": sha256_bytes(right.to_csv(index=False, lineterminator="\n").encode()),
        "evidence_classification": EvidenceClassification.VERIFIED.value,
    }


def _card_number(value: str) -> int:
    match = CARD_RE.fullmatch(str(value))
    if not match:
        raise ValueError(f"Not a canonical card ID: {value!r}")
    return int(match.group(1))


def timestamped_card_directories(maskout_root: Path, through_card: str = "IMG_9431") -> list[dict[str, Any]]:
    root = Path(maskout_root)
    maximum = _card_number(through_card)
    rows = []
    if not root.is_dir():
        return rows
    for path in root.iterdir():
        match = TIMESTAMPED_CARD_RE.fullmatch(path.name)
        if path.is_dir() and match and int(match.group(2)) <= maximum:
            rows.append({
                "path": path,
                "card_id": match.group(1),
                "card_number": int(match.group(2)),
                "timestamp": match.group(3),
            })
    return sorted(rows, key=lambda row: (row["card_number"], row["timestamp"], row["path"].name))


def enumerate_transform_artifacts(
    maskout_root: Path,
    *,
    through_card: str = "IMG_9431",
    artifact_names: Sequence[str] = TRANSFORM_ARTIFACT_NAMES,
) -> list[InventoryRecord]:
    """Hash every existing timestamped transform artifact through a card boundary."""

    records: list[InventoryRecord] = []
    for directory in timestamped_card_directories(maskout_root, through_card):
        for artifact_name in artifact_names:
            path = directory["path"] / artifact_name
            if not path.is_file():
                continue
            stem = Path(artifact_name).stem
            records.append(inventory_record(
                path,
                logical_name=f"timestamped_transform_{directory['card_id']}_{directory['timestamp']}_{stem}",
                role="timestamped PCA/prototype lineage",
                metadata={
                    "card_id": directory["card_id"],
                    "timestamp": directory["timestamp"],
                    "artifact_name": artifact_name,
                },
            ))
    return records


def enumerate_prefix_csv_artifacts(
    maskout_root: Path,
    *,
    cards: Sequence[str] = ("IMG_9428", "IMG_9431"),
) -> list[InventoryRecord]:
    requested = set(map(str, cards))
    records: list[InventoryRecord] = []
    for directory in timestamped_card_directories(maskout_root, max(requested, key=_card_number)):
        if directory["card_id"] not in requested:
            continue
        path = directory["path"] / "features_train.csv"
        if path.is_file():
            records.append(inventory_record(
                path,
                logical_name=f"prefix_csv_{directory['card_id']}_{directory['timestamp']}",
                role="historical append-prefix verification",
                metadata={"card_id": directory["card_id"], "timestamp": directory["timestamp"]},
            ))
    return records


def file_is_byte_prefix(prefix_path: Path, complete_path: Path, chunk_size: int = 8 << 20) -> bool:
    prefix_path, complete_path = Path(prefix_path), Path(complete_path)
    if prefix_path.stat().st_size > complete_path.stat().st_size:
        return False
    remaining = prefix_path.stat().st_size
    with prefix_path.open("rb") as prefix, complete_path.open("rb") as complete:
        while remaining:
            count = min(chunk_size, remaining)
            if prefix.read(count) != complete.read(count):
                return False
            remaining -= count
    return True


def prefix_csv_diagnostics(previous_csv: Path, anchor_csv: Path, next_csv: Path) -> dict[str, Any]:
    paths = [Path(previous_csv), Path(anchor_csv), Path(next_csv)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return {
            "status": "BLOCKED_MISSING_PREFIX_INPUT",
            "missing_paths": missing,
            "evidence_classification": EvidenceClassification.BLOCKED.value,
        }
    previous_prefix = file_is_byte_prefix(paths[0], paths[1])
    next_prefix = file_is_byte_prefix(paths[1], paths[2])
    return {
        "status": "PASS" if previous_prefix and next_prefix else "FAIL",
        "previous_is_anchor_prefix": previous_prefix,
        "anchor_is_next_prefix": next_prefix,
        "paths": [str(path.resolve()) for path in paths],
        "sha256": [sha256_file(path) for path in paths],
        "evidence_classification": EvidenceClassification.VERIFIED.value,
    }


def _infer_card_from_path(path: Path, allowed_cards: set[str] | None = None) -> str | None:
    matches = [part for part in path.parts if CARD_RE.fullmatch(part)]
    if allowed_cards is not None:
        matches = [value for value in matches if value in allowed_cards]
    return matches[-1] if matches else None


def _audit_cards(audit_csv: Path, card_column: str | None = None) -> list[str]:
    header = pd.read_csv(audit_csv, nrows=0).columns.tolist()
    selected = card_column or next((name for name in ("img_folder", "card_id", "folder") if name in header), None)
    if selected is None:
        raise ValueError("Audit CSV has no card column")
    values = pd.read_csv(audit_csv, usecols=[selected], keep_default_na=False)[selected]
    return sorted({str(value).strip() for value in values if CARD_RE.fullmatch(str(value).strip())}, key=_card_number)


def enumerate_audit_detection_artifacts(
    search_root: Path,
    *,
    audit_csv: Path | None = None,
    audit_cards: Sequence[str] | None = None,
) -> list[InventoryRecord]:
    """Hash every detections.csv whose path is attributable to an audit card."""

    if audit_cards is None:
        if audit_csv is None:
            raise ValueError("audit_csv or audit_cards is required")
        audit_cards = _audit_cards(Path(audit_csv))
    allowed = set(map(str, audit_cards))
    root = Path(search_root)
    records: list[InventoryRecord] = []
    if not root.is_dir():
        return records
    for path in sorted(root.rglob("detections.csv")):
        card_id = _infer_card_from_path(path, allowed)
        if card_id is None:
            continue
        relative = path.relative_to(root).as_posix()
        records.append(inventory_record(
            path,
            logical_name=f"audit_detection_{card_id}_{sha256_bytes(relative.encode())[:12]}",
            role="card-aware audit detection source",
            metadata={"card_id": card_id, "relative_path": relative},
        ))
    return records


def _literal_numeric_sequence(value: ast.AST) -> list[float] | None:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError):
        return None
    if not isinstance(parsed, (list, tuple)) or not parsed:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in parsed):
        return None
    return [float(item) for item in parsed]


def extract_notebook_scale_evidence(
    notebook_path: Path,
    *,
    raw_scale_symbol: str = "SCALES_TRAIN",
    feature_scale_symbol: str = "MULTISCALE_SCALES",
) -> dict[str, Any]:
    """Extract literal scale assignments from executable notebook source via AST."""

    path = Path(notebook_path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assignments: list[dict[str, Any]] = []
    wanted = {raw_scale_symbol, feature_scale_symbol}
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else str(source)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            targets: list[str] = []
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
                value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets, value = [node.target.id], node.value
            if value is None:
                continue
            sequence = _literal_numeric_sequence(value)
            for symbol in targets:
                if symbol in wanted and sequence is not None:
                    expression = ast.get_source_segment(source, value) or repr(sequence)
                    assignments.append({
                        "symbol": symbol,
                        "values": sequence,
                        "cell_index": cell_index,
                        "cell_line": int(getattr(node, "lineno", 0)),
                        "literal_expression": expression,
                        "literal_expression_sha256": sha256_bytes(expression.encode()),
                        "evidence_classification": EvidenceClassification.VERIFIED.value,
                    })

    def resolve(symbol: str) -> dict[str, Any]:
        matches = [item for item in assignments if item["symbol"] == symbol]
        unique = {tuple(item["values"]) for item in matches}
        if len(unique) == 1:
            return {
                "symbol": symbol,
                "values": list(next(iter(unique))),
                "assignment_count": len(matches),
                "status": "VERIFIED_LITERAL_ASSIGNMENT",
                "evidence_classification": EvidenceClassification.VERIFIED.value,
            }
        return {
            "symbol": symbol,
            "values": None,
            "assignment_count": len(matches),
            "distinct_literal_values": [list(value) for value in sorted(unique)],
            "status": "UNRESOLVED_MISSING_OR_CONFLICTING_LITERAL_ASSIGNMENT",
            "evidence_classification": EvidenceClassification.UNRESOLVED.value,
        }

    raw, feature = resolve(raw_scale_symbol), resolve(feature_scale_symbol)
    overall = (
        EvidenceClassification.VERIFIED.value
        if raw["values"] is not None and feature["values"] is not None
        else EvidenceClassification.UNRESOLVED.value
    )
    return {
        "notebook_path": str(path.resolve()),
        "notebook_sha256": sha256_file(path),
        "assignments": assignments,
        "raw_embedding_scales": raw,
        "feature_table_scales": feature,
        "evidence_classification": overall,
    }


def _schema_keys(path: Path, maximum_parse_bytes: int) -> tuple[list[str], str | None]:
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".tsv", ".txt"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            return list(pd.read_csv(path, sep=delimiter, nrows=0).columns.astype(str)), None
        if suffix == ".parquet":
            return list(pd.read_parquet(path).columns.astype(str)), None
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                return list(payload.files), None
        if suffix == ".npy":
            payload = np.load(path, mmap_mode="r", allow_pickle=False)
            return list(payload.dtype.names or ()), None
        if suffix in {".json", ".jsonl"}:
            if path.stat().st_size > maximum_parse_bytes:
                return [], f"JSON exceeds parse limit ({maximum_parse_bytes} bytes)"
            if suffix == ".jsonl":
                with path.open(encoding="utf-8") as handle:
                    value = json.loads(handle.readline())
            else:
                value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, list) and value:
                value = value[0]
            return list(map(str, value.keys())) if isinstance(value, dict) else [], None
    except Exception as exc:  # A candidate may be corrupt or use a nonstandard schema.
        return [], f"{type(exc).__name__}: {exc}"
    return [], None


def _read_sidecar_frame(path: Path, maximum_parse_bytes: int) -> tuple[pd.DataFrame | None, str | None]:
    """Read a candidate row map without allowing object-bearing NumPy payloads."""

    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".tsv", ".txt"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            return pd.read_csv(path, sep=delimiter, keep_default_na=False, low_memory=False), None
        if suffix == ".parquet":
            return pd.read_parquet(path), None
        if suffix in {".json", ".jsonl"}:
            if path.stat().st_size > maximum_parse_bytes:
                return None, f"JSON exceeds parse limit ({maximum_parse_bytes} bytes)"
            if suffix == ".jsonl":
                return pd.read_json(path, lines=True), None
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, list) and (not value or isinstance(value[0], dict)):
                return pd.DataFrame(value), None
            if isinstance(value, dict) and value and all(
                isinstance(item, list) for item in value.values()
            ):
                return pd.DataFrame(value), None
            return None, "JSON is not a row-record or column-array table"
        if suffix == ".npy":
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if not value.dtype.names:
                return None, "NPY is not a structured row table"
            return pd.DataFrame({name: value[name] for name in value.dtype.names}), None
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                arrays = {name: np.asarray(payload[name]) for name in payload.files}
            if not arrays or any(array.ndim != 1 for array in arrays.values()):
                return None, "NPZ does not contain only one-dimensional row arrays"
            lengths = {len(array) for array in arrays.values()}
            if len(lengths) != 1:
                return None, "NPZ row arrays have inconsistent lengths"
            return pd.DataFrame(arrays), None
    except Exception as exc:  # Candidate evidence may be corrupt or non-tabular.
        return None, f"{type(exc).__name__}: {exc}"
    return None, f"Unsupported sidecar format: {suffix}"


def canonical_key_frame(frame: pd.DataFrame, key_columns: Sequence[str]) -> pd.DataFrame:
    """Canonicalize stable-key values without changing their row order."""

    lookup = {str(column).lower(): str(column) for column in frame.columns}
    missing = [str(column) for column in key_columns if str(column).lower() not in lookup]
    if missing:
        raise KeyError(f"Stable-key columns missing: {missing}")
    result: dict[str, pd.Series] = {}
    for requested in map(str, key_columns):
        source = frame[lookup[requested.lower()]]
        if requested.lower() in {"file_name", "image"}:
            result[requested] = source.map(
                lambda value: "" if _is_missing_value(value) else Path(str(value)).name
            )
        elif requested.lower() == "scale":
            result[requested] = source.map(_canonical_scale)
        else:
            result[requested] = source.map(_canonical_identifier)
    return pd.DataFrame(result, index=frame.index)


def canonical_key_digest(frame: pd.DataFrame, key_columns: Sequence[str]) -> str:
    canonical = canonical_key_frame(frame, key_columns)
    rows = (
        json.dumps(list(row), ensure_ascii=True, separators=(",", ":"))
        for row in canonical.itertuples(index=False, name=None)
    )
    return sha256_bytes(("\n".join(rows) + "\n").encode())


def _explicit_row_index(
    frame: pd.DataFrame,
) -> tuple[str | None, np.ndarray | None, dict[str, bool]]:
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    matches = [lookup[name] for name in ROW_INDEX_COLUMNS if name in lookup]
    checks = {
        "explicit_row_index_present": len(matches) == 1,
        "explicit_row_index_integer": False,
        "explicit_row_index_complete_permutation": False,
    }
    if len(matches) != 1:
        return None, None, checks
    column = matches[0]
    numeric = pd.to_numeric(frame[column], errors="coerce")
    integral = bool(numeric.notna().all() and np.equal(numeric, np.floor(numeric)).all())
    checks["explicit_row_index_integer"] = integral
    if not integral:
        return column, None, checks
    values = numeric.to_numpy(np.int64)
    checks["explicit_row_index_complete_permutation"] = bool(
        len(values) == 0 or np.array_equal(np.sort(values), np.arange(len(values), dtype=np.int64))
    )
    return column, values, checks


def discover_stable_key_sidecars(
    raw_cache: Path,
    *,
    search_roots: Sequence[Path] | None = None,
    required_key_sets: Sequence[frozenset[str]] = DEFAULT_KEY_SETS,
    maximum_parse_bytes: int = 64 << 20,
) -> dict[str, Any]:
    """Search nearby files for a persisted row-key mapping; never infer one by order."""

    raw_cache = Path(raw_cache).resolve(strict=False)
    if search_roots is None:
        parent = raw_cache.parent
        broader = parent.parents[1] if len(parent.parents) > 1 else parent
        search_roots = (parent, broader)
    roots = sorted({Path(root).resolve(strict=False) for root in search_roots}, key=str)
    candidates: set[Path] = set()
    files_seen: set[Path] = set()
    eligible_suffix_files: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                files_seen.add(path.resolve())
                if path.suffix.lower() in SIDECAR_SUFFIXES:
                    eligible_suffix_files.add(path.resolve())
            if (
                path.is_file()
                and path.resolve(strict=False) != raw_cache
                and path.suffix.lower() in SIDECAR_SUFFIXES
                and SIDECAR_NAME_RE.search(path.stem)
            ):
                candidates.add(path.resolve())
    cache = inspect_raw_cache(raw_cache)
    cache_rows = cache.get("row_count")
    rows = []
    records = []
    qualifying = 0
    unresolved = 0
    normalized_requirements = [set(map(str.lower, item)) for item in required_key_sets]
    for index, path in enumerate(sorted(candidates, key=str)):
        keys, schema_error = _schema_keys(path, maximum_parse_bytes)
        lower = set(map(str.lower, keys))
        matched = [sorted(required) for required in normalized_requirements if required <= lower]
        frame, read_error = _read_sidecar_frame(path, maximum_parse_bytes)
        error = schema_error or read_error
        row_count = len(frame) if frame is not None else None
        row_index_column = None
        row_indices = None
        order_checks = {
            "explicit_row_index_present": False,
            "explicit_row_index_integer": False,
            "explicit_row_index_complete_permutation": False,
        }
        verified_key_sets: list[list[str]] = []
        key_diagnostics: dict[str, Any] = {}
        if frame is not None:
            row_index_column, row_indices, order_checks = _explicit_row_index(frame)
            for required in matched:
                canonical = canonical_key_frame(frame, required)
                null_rows = int(canonical.eq("").any(axis=1).sum())
                duplicate_rows = int(canonical.duplicated(keep=False).sum())
                key_diagnostics["|".join(required)] = {
                    "stable_key_null_rows": null_rows,
                    "stable_key_duplicate_rows": duplicate_rows,
                    "ordered_key_sha256": canonical_key_digest(frame, required),
                }
                if null_rows == 0 and duplicate_rows == 0:
                    verified_key_sets.append(required)
        row_count_matches = bool(cache_rows is not None and row_count == cache_rows)
        is_qualifying = bool(
            verified_key_sets
            and row_count_matches
            and all(order_checks.values())
            and row_indices is not None
        )
        qualifying += int(is_qualifying)
        unresolved += int(error is not None)
        classification = EvidenceClassification.UNRESOLVED.value if error else EvidenceClassification.VERIFIED.value
        row = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "schema_keys": keys,
            "matching_stable_key_sets": matched,
            "verified_stable_key_sets": verified_key_sets,
            "row_count": row_count,
            "raw_cache_row_count": cache_rows,
            "row_count_matches_raw_cache": row_count_matches,
            "explicit_row_index_column": row_index_column,
            "row_order_checks": order_checks,
            "stable_key_diagnostics": key_diagnostics,
            "qualifies_as_stable_key_sidecar": is_qualifying,
            "inspection_error": error,
            "evidence_classification": classification,
        }
        rows.append(row)
        records.append(inventory_record(
            path,
            logical_name=f"raw_key_sidecar_candidate_{index:04d}_{path.stem}",
            role="raw-cache stable-key sidecar candidate",
            metadata={
                "schema_keys": keys,
                "qualifies_as_stable_key_sidecar": is_qualifying,
                "row_count": row_count,
                "raw_cache_row_count": cache_rows,
                "explicit_row_index_column": row_index_column,
                "inspection_error": error,
            },
        ))
    if qualifying:
        status, classification = "VERIFIED_STABLE_KEY_SIDECAR_FOUND", EvidenceClassification.VERIFIED.value
    elif unresolved:
        status, classification = "UNRESOLVED_CANDIDATE_SIDECAR_INSPECTION", EvidenceClassification.UNRESOLVED.value
    else:
        status, classification = "BLOCKED_NO_QUALIFYING_STABLE_KEY_SIDECAR", EvidenceClassification.BLOCKED.value
    return {
        "raw_cache": str(raw_cache),
        "raw_cache_inspection": cache,
        "search_roots": list(map(str, roots)),
        "search_roots_existing": [str(root) for root in roots if root.is_dir()],
        "files_seen": len(files_seen),
        "eligible_suffix_files_seen": len(eligible_suffix_files),
        "candidate_count": len(rows),
        "qualifying_sidecar_count": qualifying,
        "status": status,
        "candidates": rows,
        "inventory_records": records,
        "evidence_classification": classification,
    }


def _canonical_scale(value: Any) -> str:
    if _is_missing_value(value):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if not np.isfinite(numeric):
        return str(value).strip()
    return format(numeric, ".12g")


def _canonical_identifier(value: Any) -> str:
    if _is_missing_value(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def _is_missing_value(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null", "<na>"}


def inspect_candidate_table(
    table_csv: Path,
    *,
    key_columns: Sequence[str] = ("file_name", "ann_id", "scale"),
) -> dict[str, Any]:
    path = Path(table_csv)
    columns = list(key_columns)
    frame = pd.read_csv(path, usecols=columns, keep_default_na=False, low_memory=False)
    file_column, candidate_column, scale_column = columns
    frame[file_column] = frame[file_column].map(
        lambda value: "" if _is_missing_value(value) else Path(str(value)).name
    )
    frame[candidate_column] = frame[candidate_column].map(_canonical_identifier)
    frame[scale_column] = frame[scale_column].map(_canonical_scale)
    null_mask = frame[columns].eq("").any(axis=1)
    duplicated = frame.duplicated(columns, keep=False)
    duplicate_groups = frame.loc[duplicated, columns].drop_duplicates()
    base_columns = [file_column, candidate_column]
    base_count = int(frame[base_columns].drop_duplicates().shape[0])
    scale_sequence = list(dict.fromkeys(frame[scale_column].tolist()))
    expected_scale_set = frozenset(scale_sequence)
    scale_sets = frame.groupby(base_columns, sort=False)[scale_column].agg(lambda values: frozenset(values))
    incomplete = int((scale_sets != expected_scale_set).sum())
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "stable_key_columns": columns,
        "stable_key_null_rows": int(null_mask.sum()),
        "stable_key_duplicate_rows": int(duplicated.sum()),
        "stable_key_duplicate_groups": len(duplicate_groups),
        "base_candidate_columns": base_columns,
        "base_candidate_count": base_count,
        "scale_sequence_first_appearance": scale_sequence,
        "scale_count": len(scale_sequence),
        "base_candidates_with_incomplete_scale_set": incomplete,
        "rows_minus_complete_cartesian_rows": len(frame) - base_count * len(scale_sequence),
        "evidence_classification": EvidenceClassification.VERIFIED.value,
    }


def inspect_raw_cache(raw_cache: Path, *, scale_sequence: Sequence[float] | None = None) -> dict[str, Any]:
    path = Path(raw_cache)
    try:
        payload = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        return {
            "path": str(path.resolve(strict=False)),
            "status": "UNRESOLVED_UNREADABLE_RAW_CACHE",
            "error": f"{type(exc).__name__}: {exc}",
            "evidence_classification": EvidenceClassification.UNRESOLVED.value,
        }
    shape = list(map(int, payload.shape))
    rows = shape[0] if shape else 0
    scale_count = len(scale_sequence) if scale_sequence is not None else None
    divisible = bool(scale_count and rows % scale_count == 0)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "shape": shape,
        "ndim": int(payload.ndim),
        "dtype": str(payload.dtype),
        "row_count": rows,
        "dimension_count": shape[1] if payload.ndim == 2 else None,
        "scale_sequence": list(map(float, scale_sequence)) if scale_sequence is not None else None,
        "scale_count": scale_count,
        "row_count_divisible_by_scale_count": divisible if scale_count else None,
        "implicit_candidate_count": rows // scale_count if divisible and scale_count else None,
        "status": "VERIFIED_READABLE_RAW_CACHE",
        "evidence_classification": EvidenceClassification.VERIFIED.value,
    }


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame, names=list(frame.columns))


def _actual_sidecar_join(
    sidecar_diagnostic: Mapping[str, Any] | None,
    candidate_table_csv: Path,
    *,
    key_columns: Sequence[str],
    cache_row_count: int,
    maximum_parse_bytes: int = 64 << 20,
) -> dict[str, Any]:
    table_source = pd.read_csv(
        candidate_table_csv,
        usecols=list(key_columns),
        keep_default_na=False,
        low_memory=False,
    )
    table_keys = canonical_key_frame(table_source, key_columns)
    table_index = _key_index(table_keys)
    expected = sorted(str(column).lower() for column in key_columns)
    attempts: list[dict[str, Any]] = []
    for candidate in (sidecar_diagnostic or {}).get("candidates", []):
        matching = [sorted(map(str.lower, value)) for value in candidate.get("matching_stable_key_sets", [])]
        if expected not in matching:
            continue
        path = Path(str(candidate.get("path", "")))
        if not path.is_file():
            attempts.append({
                "sidecar_path": str(path),
                "inspection_error": "Sidecar disappeared before actual join",
                "mapped_candidate_rows": 0,
                "exact_join_rate": 0.0 if len(table_keys) else 1.0,
            })
            continue
        current_hash = sha256_file(path)
        if current_hash != candidate.get("sha256"):
            attempts.append({
                "sidecar_path": str(path),
                "inspection_error": "Sidecar hash changed between discovery and actual join",
                "mapped_candidate_rows": 0,
                "exact_join_rate": 0.0 if len(table_keys) else 1.0,
            })
            continue
        frame, error = _read_sidecar_frame(path, maximum_parse_bytes)
        if frame is None:
            attempts.append({
                "sidecar_path": str(path), "inspection_error": error,
                "mapped_candidate_rows": 0,
                "exact_join_rate": 0.0 if len(table_keys) else 1.0,
            })
            continue
        try:
            sidecar_keys = canonical_key_frame(frame, key_columns)
        except KeyError as exc:
            attempts.append({
                "sidecar_path": str(path), "inspection_error": str(exc),
                "mapped_candidate_rows": 0,
                "exact_join_rate": 0.0 if len(table_keys) else 1.0,
            })
            continue
        row_index_column, row_indices, order_checks = _explicit_row_index(frame)
        sidecar_index = _key_index(sidecar_keys)
        stable_null_rows = int(sidecar_keys.eq("").any(axis=1).sum())
        stable_duplicate_rows = int(sidecar_keys.duplicated(keep=False).sum())
        row_index_in_cache_range = bool(
            row_indices is not None
            and ((row_indices >= 0) & (row_indices < cache_row_count)).all()
        )
        usable = bool(
            row_indices is not None
            and order_checks["explicit_row_index_integer"]
            and stable_null_rows == 0
            and stable_duplicate_rows == 0
            and row_index_in_cache_range
        )
        alignment = np.full(len(table_index), -1, dtype=np.int64)
        if usable:
            row_map = pd.Series(row_indices, index=sidecar_index)
            mapped = row_map.reindex(table_index)
            matched_mask = mapped.notna().to_numpy()
            alignment[matched_mask] = mapped.loc[matched_mask].to_numpy(np.int64)
        else:
            matched_mask = np.zeros(len(table_index), dtype=bool)
        mapped_rows = int(matched_mask.sum())
        orphan_rows = int((~sidecar_index.isin(table_index)).sum()) if usable else len(sidecar_index)
        mapped_cache_collision_rows = int(
            pd.Series(alignment[alignment >= 0]).duplicated(keep=False).sum()
        )
        attempts.append({
            "sidecar_path": str(path),
            "sidecar_sha256": current_hash,
            "sidecar_row_count": len(frame),
            "candidate_is_cache_qualifying": bool(candidate.get("qualifies_as_stable_key_sidecar")),
            "explicit_row_index_column": row_index_column,
            "row_order_checks": order_checks,
            "row_index_in_cache_range": row_index_in_cache_range,
            "stable_key_null_rows": stable_null_rows,
            "stable_key_duplicate_rows": stable_duplicate_rows,
            "mapped_candidate_rows": mapped_rows,
            "unmatched_candidate_rows": len(table_index) - mapped_rows,
            "orphan_sidecar_rows": orphan_rows,
            "mapped_cache_collision_rows": mapped_cache_collision_rows,
            "exact_join_rate": mapped_rows / len(table_index) if len(table_index) else 1.0,
            "mapping_sha256_int64_le": sha256_bytes(alignment.astype("<i8", copy=False).tobytes()),
            "inspection_error": None,
            "_alignment": alignment,
        })
    selected = max(
        attempts,
        key=lambda row: (
            int(row.get("mapped_candidate_rows", 0)),
            -int(row.get("orphan_sidecar_rows", cache_row_count)),
            str(row.get("sidecar_path", "")),
        ),
        default=None,
    )
    complete_hashes = {
        str(row["mapping_sha256_int64_le"])
        for row in attempts
        if row.get("candidate_is_cache_qualifying")
        and row.get("mapped_candidate_rows") == len(table_index)
        and row.get("orphan_sidecar_rows") == 0
        and row.get("mapped_cache_collision_rows") == 0
    }
    public_attempts = [
        {key: value for key, value in row.items() if key != "_alignment"}
        for row in attempts
    ]
    return {
        "actual_join_attempted": bool(attempts),
        "candidate_table_rows": len(table_index),
        "candidate_table_ordered_key_sha256": canonical_key_digest(table_source, key_columns),
        "candidate_table_duplicate_key_rows": int(table_keys.duplicated(keep=False).sum()),
        "sidecars_evaluated": len(attempts),
        "attempts": public_attempts,
        "selected_sidecar_path": selected.get("sidecar_path") if selected else None,
        "selected_sidecar_sha256": selected.get("sidecar_sha256") if selected else None,
        "mapped_candidate_rows": int(selected.get("mapped_candidate_rows", 0)) if selected else 0,
        "unmatched_candidate_rows": int(selected.get("unmatched_candidate_rows", len(table_index))) if selected else len(table_index),
        "orphan_sidecar_rows": int(selected.get("orphan_sidecar_rows", 0)) if selected else None,
        "mapped_cache_collision_rows": int(selected.get("mapped_cache_collision_rows", 0)) if selected else None,
        "exact_join_rate": float(selected["exact_join_rate"]) if selected and selected.get("inspection_error") is None else None,
        "selected_mapping_sha256_int64_le": selected.get("mapping_sha256_int64_le") if selected else None,
        "complete_mapping_hash_count": len(complete_hashes),
        "unambiguous_complete_mapping": len(complete_hashes) == 1,
        "selected_cache_row_index": selected.get("_alignment") if selected else None,
    }


def _write_alignment_index(path: Path, cache_row_index: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, cache_row_index=np.asarray(cache_row_index, dtype="<i8"))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def raw_cache_table_diagnostics(
    raw_cache: Path,
    candidate_table_csv: Path,
    *,
    raw_scale_sequence: Sequence[float] | None,
    expected_dimension_count: int,
    expected_dtype: str = "float32",
    sidecar_diagnostic: Mapping[str, Any] | None = None,
    key_columns: Sequence[str] = ("file_name", "ann_id", "scale"),
    alignment_index_output: Path | None = None,
    population: str | None = None,
    embedding_semantics_manifest_path: Path | None = None,
    embedding_semantics_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Compute mapping constraints from actual cache/table files."""

    cache = inspect_raw_cache(raw_cache, scale_sequence=raw_scale_sequence)
    table = inspect_candidate_table(candidate_table_csv, key_columns=key_columns)
    if cache.get("evidence_classification") != EvidenceClassification.VERIFIED.value:
        return {
            "status": "UNRESOLVED_RAW_CACHE",
            "raw_cache": cache,
            "candidate_table": table,
            "evidence_classification": EvidenceClassification.UNRESOLVED.value,
        }
    inferred_count = cache.get("implicit_candidate_count")
    deficit = table["base_candidate_count"] - inferred_count if inferred_count is not None else None
    raw_scales = [_canonical_scale(value) for value in raw_scale_sequence] if raw_scale_sequence is not None else None
    scale_match = raw_scales == table["scale_sequence_first_appearance"] if raw_scales is not None else None
    actual_join = _actual_sidecar_join(
        sidecar_diagnostic,
        candidate_table_csv,
        key_columns=key_columns,
        cache_row_count=int(cache["row_count"]),
    )
    sidecar_verified = bool(
        sidecar_diagnostic
        and sidecar_diagnostic.get("qualifying_sidecar_count", 0) > 0
        and actual_join["actual_join_attempted"]
    )
    checks = {
        "two_dimensional": cache["ndim"] == 2,
        "expected_dimension_count": cache["dimension_count"] == int(expected_dimension_count),
        "expected_dtype": cache["dtype"] == str(np.dtype(expected_dtype)),
        "cache_rows_divisible_by_raw_scale_count": cache["row_count_divisible_by_scale_count"] is True,
        "candidate_count_equal": deficit == 0,
        "scale_sequence_equal": scale_match is True,
        "stable_table_key_nonnull": table["stable_key_null_rows"] == 0,
        "stable_table_key_unique": table["stable_key_duplicate_rows"] == 0,
        "qualifying_stable_key_sidecar": sidecar_verified,
        "actual_table_join_complete": actual_join["mapped_candidate_rows"] == table["rows"],
        "actual_cache_key_coverage_complete": actual_join["orphan_sidecar_rows"] == 0,
        "actual_mapping_has_no_cache_collisions": actual_join["mapped_cache_collision_rows"] == 0,
        "unambiguous_stable_key_mapping": actual_join["unambiguous_complete_mapping"],
    }
    raw_columns = [f"raw_embed_{index:04d}" for index in range(int(expected_dimension_count))]
    semantics_path = (
        Path(embedding_semantics_manifest_path).resolve(strict=False)
        if embedding_semantics_manifest_path is not None else None
    )
    semantics_verified = bool(
        population in {"training", "validation", "audit"}
        and semantics_path is not None
        and semantics_path.is_file()
        and isinstance(embedding_semantics_manifest_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", embedding_semantics_manifest_sha256)
        and sha256_file(semantics_path) == embedding_semantics_manifest_sha256
    )
    checks["verified_embedding_semantics_manifest"] = semantics_verified
    if set(checks) != RAW_MAPPING_CHECK_KEYS:
        raise RuntimeError(
            "Raw-mapping check schema drifted from RAW_MAPPING_CHECK_KEYS: "
            f"actual={sorted(checks)}, expected={sorted(RAW_MAPPING_CHECK_KEYS)}"
        )
    failed = sorted(name for name, passed in checks.items() if not passed)
    alignment_reference = None
    if not failed and alignment_index_output is not None and semantics_verified:
        alignment = actual_join["selected_cache_row_index"]
        if alignment is None or (alignment < 0).any():
            raise RuntimeError("Verified mapping did not retain a complete alignment index")
        _write_alignment_index(Path(alignment_index_output), alignment)
        alignment_reference = {
            "status": "VERIFIED_ALIGNED_RAW_LAYER_REFERENCE",
            "population": population,
            "raw_cache_path": str(Path(raw_cache).resolve()),
            "raw_cache_sha256": cache["sha256"],
            "candidate_table_path": str(Path(candidate_table_csv).resolve()),
            "candidate_table_sha256": table["sha256"],
            "sidecar_path": actual_join["selected_sidecar_path"],
            "sidecar_sha256": actual_join["selected_sidecar_sha256"],
            "measured_exact_join_rate": actual_join["exact_join_rate"],
            "measured_mapped_candidate_rows": actual_join["mapped_candidate_rows"],
            "measured_unmatched_candidate_rows": actual_join["unmatched_candidate_rows"],
            "measured_orphan_sidecar_rows": actual_join["orphan_sidecar_rows"],
            "measured_cache_collision_rows": actual_join["mapped_cache_collision_rows"],
            "alignment_index_path": str(Path(alignment_index_output).resolve()),
            "alignment_index_sha256": sha256_file(Path(alignment_index_output)),
            "alignment_mapping_sha256_int64_le": actual_join["selected_mapping_sha256_int64_le"],
            "candidate_key_columns": list(map(str, key_columns)),
            "candidate_key_sha256": actual_join["candidate_table_ordered_key_sha256"],
            "candidate_row_order_key_sha256": actual_join["candidate_table_ordered_key_sha256"],
            "row_count": table["rows"],
            "dimension_count": int(expected_dimension_count),
            "dtype": str(np.dtype(expected_dtype)),
            "embedding_semantics_manifest_path": str(semantics_path),
            "embedding_semantics_manifest_sha256": embedding_semantics_manifest_sha256,
            "ordered_raw_feature_columns": raw_columns,
            "ordered_raw_feature_columns_sha256": sha256_bytes(("\n".join(raw_columns) + "\n").encode()),
            "evidence_classification": EvidenceClassification.VERIFIED.value,
        }
    actual_join_public = {
        key: value for key, value in actual_join.items()
        if key != "selected_cache_row_index"
    }
    return {
        "status": "VERIFIED_EXACT_RAW_MAPPING_PREREQUISITES" if not failed else "BLOCKED_RAW_MAPPING_PREREQUISITES",
        "raw_cache": cache,
        "candidate_table": table,
        "candidate_deficit": deficit,
        "raw_and_feature_scale_sequence_match": scale_match,
        "sidecar_status": sidecar_diagnostic.get("status") if sidecar_diagnostic else "NOT_INSPECTED",
        "actual_stable_key_join": actual_join_public,
        "aligned_raw_layer_reference": alignment_reference,
        "alignment_materialization_status": (
            "VERIFIED_MATERIALIZED"
            if alignment_reference is not None
            else "NOT_MATERIALIZED_MAPPING_OR_ENCODER_SEMANTICS_BLOCKED"
        ),
        "row_order_provenance": {
            "status": (
                "VERIFIED_PERSISTED_STABLE_KEY_SIDECAR" if sidecar_verified
                else "BLOCKED_NO_PERSISTED_STABLE_KEY_SIDECAR"
            ),
            "qualifying_sidecar_paths": [
                row["path"] for row in (sidecar_diagnostic or {}).get("candidates", [])
                if row.get("qualifies_as_stable_key_sidecar")
            ],
            "selected_sidecar_path": actual_join["selected_sidecar_path"],
            "selected_mapping_sha256_int64_le": actual_join["selected_mapping_sha256_int64_le"],
            "evidence_classification": (
                EvidenceClassification.VERIFIED.value if sidecar_verified
                else EvidenceClassification.BLOCKED.value
            ),
        },
        "checks": checks,
        "failed_checks": failed,
        "ordered_raw_feature_columns": raw_columns,
        "ordered_raw_feature_columns_sha256": sha256_bytes(("\n".join(raw_columns) + "\n").encode()),
        "evidence_classification": (
            EvidenceClassification.VERIFIED.value if not failed else EvidenceClassification.BLOCKED.value
        ),
    }


def _nonblank(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    return ~normalized.isin({"", "nan", "none", "null", "<na>"})


def _parseable_rle(value: Any) -> bool:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return False
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return False
    return isinstance(parsed, dict) and "counts" in parsed and "size" in parsed


def _candidate_asset_path(value: Any, detection_path: Path, card_root: Path) -> Path | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    path = Path(text).expanduser()
    candidates = [path] if path.is_absolute() else [detection_path.parent / path, card_root / path]
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def _card_root(detection_path: Path, card_id: str) -> Path:
    for parent in detection_path.parents:
        if parent.name == card_id:
            return parent
    return detection_path.parent


def audit_detection_diagnostics(
    audit_csv: Path,
    detection_inputs: Sequence[Path | InventoryRecord],
    *,
    card_column: str | None = None,
) -> dict[str, Any]:
    """Measure detection coverage with the card included in every join key."""

    audit_path = Path(audit_csv)
    audit_header = pd.read_csv(audit_path, nrows=0).columns.tolist()
    card_column = card_column or next((name for name in ("img_folder", "card_id", "folder") if name in audit_header), None)
    if card_column is None or not {"image", "id"}.issubset(audit_header):
        raise ValueError("Audit CSV requires a card column plus image and id")
    audit = pd.read_csv(audit_path, usecols=[card_column, "image", "id"], keep_default_na=False, low_memory=False)
    audit_keys = pd.DataFrame({
        "card_id": audit[card_column].astype(str).str.strip(),
        "image": audit.image.astype(str).map(lambda value: Path(value).name),
        "id": audit.id.map(_canonical_identifier),
    })
    audit_null = audit_keys.eq("").any(axis=1)
    audit_duplicate = audit_keys.duplicated(["card_id", "image", "id"], keep=False)
    allowed_cards = set(audit_keys.card_id)

    paths = []
    for value in detection_inputs:
        path = Path(value.path if isinstance(value, InventoryRecord) else value).resolve(strict=False)
        if path not in paths:
            paths.append(path)
    paths.sort(key=str)
    source_rows: list[pd.DataFrame] = []
    file_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for source_index, path in enumerate(paths):
        card_id = _infer_card_from_path(path, allowed_cards)
        if card_id is None:
            errors.append({"path": str(path), "error": "No audit card ID in source path"})
            continue
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
            if not {"image", "id"}.issubset(header):
                raise ValueError("detections.csv lacks image/id")
            path_columns = [
                name for name in header
                if ("mask" in name.lower() or "crop" in name.lower())
                and ("path" in name.lower() or "file" in name.lower())
            ]
            selected = [name for name in ("image", "id", "rle", "poly", *path_columns) if name in header]
            frame = pd.read_csv(path, usecols=selected, keep_default_na=False, low_memory=False)
        except Exception as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        card_root = _card_root(path, card_id)
        keys = pd.DataFrame({
            "card_id": card_id,
            "image": frame.image.astype(str).map(lambda value: Path(value).name),
            "id": frame.id.map(_canonical_identifier),
        })
        key_null = keys.eq("").any(axis=1)
        duplicate = keys.duplicated(["card_id", "image", "id"], keep=False)
        rle_nonblank = _nonblank(frame.rle) if "rle" in frame else pd.Series(False, index=frame.index)
        rle_parseable = frame.rle.map(_parseable_rle) if "rle" in frame else pd.Series(False, index=frame.index)
        poly_nonblank = _nonblank(frame.poly) if "poly" in frame else pd.Series(False, index=frame.index)
        tile_available = frame.image.astype(str).map(
            lambda value: (card_root / "tiles" / Path(value).name).is_file()
        )
        declared_mask = pd.Series(False, index=frame.index)
        existing_mask = pd.Series(False, index=frame.index)
        declared_crop = pd.Series(False, index=frame.index)
        existing_crop = pd.Series(False, index=frame.index)
        for column in path_columns:
            present = _nonblank(frame[column])
            exists = frame[column].map(lambda value: _candidate_asset_path(value, path, card_root) is not None)
            if "mask" in column.lower():
                declared_mask |= present
                existing_mask |= exists
            if "crop" in column.lower():
                declared_crop |= present
                existing_crop |= exists
        row_frame = keys.copy()
        row_frame["source_index"] = source_index
        row_frame["source_path"] = str(path)
        row_frame["key_null"] = key_null.to_numpy()
        row_frame["rle_nonblank"] = rle_nonblank.to_numpy()
        row_frame["rle_parseable"] = rle_parseable.to_numpy()
        row_frame["poly_nonblank"] = poly_nonblank.to_numpy()
        row_frame["tile_image_available"] = tile_available.to_numpy()
        row_frame["explicit_mask_available"] = existing_mask.to_numpy()
        row_frame["explicit_crop_available"] = existing_crop.to_numpy()
        row_frame["reconstructible_input"] = (
            tile_available.to_numpy() & (rle_parseable.to_numpy() | existing_mask.to_numpy())
        )
        payload = pd.DataFrame({
            "rle": frame.rle.astype(str) if "rle" in frame else "",
            "poly": frame.poly.astype(str) if "poly" in frame else "",
        })
        row_frame["mask_payload_sha256"] = [
            sha256_bytes(f"{rle}\0{poly}".encode()) for rle, poly in payload.itertuples(index=False, name=None)
        ]
        source_rows.append(row_frame)
        file_rows.append({
            "path": str(path),
            "card_id": card_id,
            "rows": len(frame),
            "key_null_rows": int(key_null.sum()),
            "duplicate_key_rows": int(duplicate.sum()),
            "duplicate_key_groups": int(keys.loc[duplicate].drop_duplicates().shape[0]),
            "nonblank_rle_rows": int(rle_nonblank.sum()),
            "parseable_rle_rows": int(rle_parseable.sum()),
            "nonblank_poly_rows": int(poly_nonblank.sum()),
            "source_tile_image_available_rows": int(tile_available.sum()),
            "declared_mask_path_rows": int(declared_mask.sum()),
            "existing_mask_path_rows": int(existing_mask.sum()),
            "declared_crop_path_rows": int(declared_crop.sum()),
            "existing_crop_path_rows": int(existing_crop.sum()),
            "evidence_classification": EvidenceClassification.VERIFIED.value,
        })

    source = pd.concat(source_rows, ignore_index=True) if source_rows else pd.DataFrame(columns=[
        "card_id", "image", "id", "source_index", "source_path", "key_null", "rle_nonblank",
        "rle_parseable", "poly_nonblank", "tile_image_available", "explicit_mask_available",
        "explicit_crop_available", "reconstructible_input", "mask_payload_sha256",
    ])
    key_columns = ["card_id", "image", "id"]
    source_duplicate = source.duplicated(key_columns, keep=False)
    unique_source_keys = source[key_columns].drop_duplicates()
    unique_audit_keys = audit_keys[key_columns].drop_duplicates()
    matched = unique_audit_keys.merge(unique_source_keys, on=key_columns, how="inner")
    per_key = source.groupby(key_columns, sort=False).agg(
        source_file_count=("source_index", "nunique"),
        payload_count=("mask_payload_sha256", "nunique"),
        any_tile=("tile_image_available", "any"),
        any_rle=("rle_parseable", "any"),
        any_mask=("explicit_mask_available", "any"),
        any_reconstructible=("reconstructible_input", "any"),
    ).reset_index() if len(source) else pd.DataFrame(columns=key_columns + [
        "source_file_count", "payload_count", "any_tile", "any_rle", "any_mask", "any_reconstructible",
    ])
    audit_coverage = unique_audit_keys.merge(per_key, on=key_columns, how="left")
    for column in ("any_tile", "any_rle", "any_mask", "any_reconstructible"):
        audit_coverage[column] = (
            audit_coverage[column].astype("boolean").fillna(False).astype(bool)
        )
    cardless_collisions = source.groupby(["image", "id"], sort=False).card_id.nunique() if len(source) else pd.Series(dtype=int)
    cross_file = per_key.source_file_count.fillna(0).astype(int) > 1 if len(per_key) else pd.Series(dtype=bool)
    payload_collision = per_key.payload_count.fillna(0).astype(int) > 1 if len(per_key) else pd.Series(dtype=bool)
    source_null_rows = int(source.key_null.sum()) if len(source) else 0
    within_file_duplicate_rows = sum(row["duplicate_key_rows"] for row in file_rows)
    within_file_duplicate_groups = sum(row["duplicate_key_groups"] for row in file_rows)
    full_reconstruction = (
        len(audit_coverage) == len(unique_audit_keys)
        and int(audit_coverage.any_reconstructible.sum()) == len(unique_audit_keys)
        and int(audit_null.sum()) == 0
        and int(audit_duplicate.sum()) == 0
        and source_null_rows == 0
        and int(cross_file.sum()) == 0
        and int(payload_collision.sum()) == 0
        and not errors
    )
    return {
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": sha256_file(audit_path),
        "audit_rows": len(audit_keys),
        "audit_unique_card_aware_keys": len(unique_audit_keys),
        "audit_null_key_rows": int(audit_null.sum()),
        "audit_duplicate_key_rows": int(audit_duplicate.sum()),
        "audit_duplicate_key_groups": int(audit_keys.loc[audit_duplicate].drop_duplicates().shape[0]),
        "detection_file_count_requested": len(paths),
        "detection_file_count_read": len(file_rows),
        "detection_read_errors": errors,
        "per_file": file_rows,
        "detection_rows": len(source),
        "detection_null_key_rows": source_null_rows,
        "detection_within_file_duplicate_key_rows": within_file_duplicate_rows,
        "detection_within_file_duplicate_key_groups": within_file_duplicate_groups,
        "detection_unique_card_aware_keys": len(unique_source_keys),
        "detection_duplicate_key_rows_across_all_sources": int(source_duplicate.sum()),
        "detection_duplicate_key_groups_across_all_sources": int(source.loc[source_duplicate, key_columns].drop_duplicates().shape[0]),
        "cross_file_collision_key_groups": int(cross_file.sum()),
        "conflicting_mask_payload_key_groups": int(payload_collision.sum()),
        "card_omission_collision_groups": int((cardless_collisions > 1).sum()),
        "detection_nonblank_rle_rows": int(source.rle_nonblank.sum()) if len(source) else 0,
        "detection_parseable_rle_rows": int(source.rle_parseable.sum()) if len(source) else 0,
        "detection_nonblank_poly_rows": int(source.poly_nonblank.sum()) if len(source) else 0,
        "detection_source_tile_image_available_rows": int(source.tile_image_available.sum()) if len(source) else 0,
        "detection_explicit_mask_available_rows": int(source.explicit_mask_available.sum()) if len(source) else 0,
        "detection_explicit_crop_available_rows": int(source.explicit_crop_available.sum()) if len(source) else 0,
        "audit_keys_matched_card_aware": len(matched),
        "audit_keys_unmatched_card_aware": len(unique_audit_keys) - len(matched),
        "audit_keys_with_source_tile_image": int(audit_coverage.any_tile.sum()),
        "audit_keys_with_parseable_rle": int(audit_coverage.any_rle.sum()),
        "audit_keys_with_existing_explicit_mask": int(audit_coverage.any_mask.sum()),
        "audit_keys_with_reconstructible_crop_mask_input": int(audit_coverage.any_reconstructible.sum()),
        "raw_embedding_input_status": (
            "VERIFIED_COMPLETE_RECONSTRUCTIBLE_INPUT" if full_reconstruction
            else "BLOCKED_INCOMPLETE_RECONSTRUCTIBLE_INPUT"
        ),
        "evidence_classification": (
            EvidenceClassification.VERIFIED.value if full_reconstruction
            else EvidenceClassification.BLOCKED.value
        ),
    }


def build_extended_source_inventory(
    *,
    maskout_root: Path,
    audit_csv: Path,
    audit_detection_root: Path,
    raw_cache_paths: Sequence[Path] = (),
    sidecar_search_roots: Sequence[Path] | None = None,
    through_card: str = "IMG_9431",
    prefix_cards: Sequence[str] = ("IMG_9428", "IMG_9431"),
) -> list[InventoryRecord]:
    """Return all amendment-discovered inputs for source pre/post integrity."""

    records = enumerate_transform_artifacts(maskout_root, through_card=through_card)
    records.extend(enumerate_prefix_csv_artifacts(maskout_root, cards=prefix_cards))
    records.extend(enumerate_audit_detection_artifacts(audit_detection_root, audit_csv=audit_csv))
    seen_sidecars: set[str] = set()
    for raw_cache in raw_cache_paths:
        diagnostic = discover_stable_key_sidecars(raw_cache, search_roots=sidecar_search_roots)
        for record in diagnostic["inventory_records"]:
            if record.path not in seen_sidecars:
                cache_key = sha256_bytes(str(Path(raw_cache).resolve(strict=False)).encode())[:8]
                records.append(InventoryRecord(
                    logical_name=f"{record.logical_name}_{cache_key}",
                    path=record.path,
                    role=record.role,
                    exists=record.exists,
                    size_bytes=record.size_bytes,
                    sha256=record.sha256,
                    evidence_classification=record.evidence_classification,
                    metadata=record.metadata,
                ))
                seen_sidecars.add(record.path)
    return sorted(records, key=lambda record: (record.logical_name, record.path))
