"""Review-label meta-analysis implemented without Notebook state.

The implementation uses rows (mappings) as its public data contract.  CSV I/O is
stdlib-only; persisted scientific models are never imported while this module is
imported.  Source constants, ordering, alignment buckets, rule thresholds and the
grouped sparse-logistic role are retained.
"""

from __future__ import annotations

from csv import DictReader, DictWriter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from io import StringIO
from itertools import combinations
from json import dumps, loads
from math import exp, isfinite, isnan, nan, sqrt
import os
from pathlib import Path
from random import Random
import stat
from statistics import median
from typing import Any, Iterable, Mapping, MutableMapping, Protocol, Sequence

from ...config import DomainConfig
from ...contracts import artifact_path, create_output_directory
from .meta_event_statistics import (
    MetaEventColumnDtypes,
    MetaEventInputRow,
    MetaEventRecord,
    MetaEventStatisticsResult,
    MetaLatestOverride,
    evaluate_meta_events_c0003,
    evaluate_meta_events_c0004,
)


class MetaAnalysisContractError(ValueError):
    contract_name = "meta-analysis"
    pass


Row = dict[str, Any]


class MetaAnalysisServices(Protocol):
    def read_rows(self, path: Path) -> list[Row]: ...

    def write_rows(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> None: ...


class StdlibMetaAnalysisServices:
    def read_rows(self, path: Path) -> list[Row]:
        if not path.is_file():
            raise MetaAnalysisContractError(f"CSV is missing: {path}")
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            return [dict(row) for row in DictReader(handle)]

    def write_rows(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
        if not fields:
            fields = ["status"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


@dataclass(frozen=True)
class KeyPairStatistic:
    first: str
    second: str
    coverage: float
    rows: int
    unique_pairs: int
    duplicate_rows: int


@dataclass(frozen=True)
class FeatureRank:
    feature: str
    auc: float
    strength: float
    direction: str


@dataclass(frozen=True)
class RuleSweep:
    feature: str
    direction: str
    threshold: float
    precision: float
    lift: float
    coverage: float
    recall: float
    count: int
    base_rate: float


@dataclass(frozen=True)
class BinaryMetrics:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    accuracy: float


@dataclass(frozen=True)
class SparseLogisticModel:
    features: tuple[str, ...]
    imputer_median: tuple[float, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    regularization_c: float
    validation_auc: float
    threshold_choose_yolo: float


@dataclass(frozen=True)
class MetaAnalysisResult:
    review_files: int
    raw_rows: int
    latest_rows: int
    flip_rows: int
    decisive_disagreements: int
    output: Path
    artifacts: tuple[Path, ...]
    event_contract_manifest: Path
    event_contract_manifest_sha256: str
    event_table: Path
    event_table_sha256: str


@dataclass(frozen=True)
class MetaFeatureBatch:
    """One direct, SHA-bound feature table consumed by meta-analysis."""

    manifest: Path
    manifest_sha256: str
    features: Path
    features_sha256: str
    rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class MetaEventBatch:
    """One direct-child, SHA-bound event table with explicit source dtypes."""

    manifest: Path
    manifest_sha256: str
    events: Path
    events_sha256: str
    columns: tuple[str, ...]
    column_dtypes: MetaEventColumnDtypes
    rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class MetaEventEvaluation:
    """Ordered C0003 then C0004 evaluation over one sealed event batch."""

    batch: MetaEventBatch
    c0003: MetaEventStatisticsResult
    c0004: MetaEventStatisticsResult


META_EVENT_MANIFEST_NAME = "meta_event_statistics_manifest.json"
META_EVENT_TABLE_NAME = "meta_event_statistics.csv"
META_EVENT_SCHEMA = "compag-curation-meta-events/v1"
META_EVENT_TEXT_GRAMMAR = "extended-iso8601-nanoseconds"
META_EVENT_COLUMNS = (
    "image",
    "id",
    "timestamp",
    "source_mtime",
    "_row_in_file",
    "human_label",
    "xgb_prob",
    "yolo_conf",
    "yolo_iou",
    "thr_xgb",
    "thr_yolo",
    "thr_iou",
    "det_thr",
    "xgb_pred",
    "final_pred",
    "kept",
    "det_policy",
)
META_EVENT_REQUIRED_COLUMNS = frozenset(META_EVENT_COLUMNS) - {"final_pred"}
META_EVENT_NULL_TOKENS = (
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "NaT",
    "None",
    "n/a",
    "nan",
    "null",
)
META_EVENT_OUTPUT_NAMES = (
    "review_events_all__key_image_id.csv",
    "review_label_flips__key_image_id.csv",
    "review_final_by_item__key_image_id.csv",
    "review_overrides_latest__key_image_id.csv",
)


MetaParameter = str | int | float | bool


@dataclass(frozen=True)
class MetaAnalysisOperation:
    """One executable node in the sealed review meta-analysis program."""

    operation: str
    source_cells: tuple[str, ...]
    input_roles: tuple[str, ...]
    output_role: str
    parameters: tuple[tuple[str, MetaParameter], ...] = ()


def build_meta_analysis_program() -> tuple[MetaAnalysisOperation, ...]:
    """Return the exact operation order executed by ``execute_meta_analysis``."""

    return (
        MetaAnalysisOperation(
            "collect-review-logs",
            ("NB-LIVE-0010-C0000",),
            ("review_root",),
            "raw_review_rows",
        ),
        MetaAnalysisOperation(
            "audit-candidate-keys",
            ("NB-LIVE-0010-C0001",),
            ("raw_review_rows",),
            "candidate_key_statistics",
        ),
        MetaAnalysisOperation(
            "build-review-event-history",
            ("NB-LIVE-0010-C0002",),
            ("raw_review_rows",),
            "review_events",
            (("include_source_folder", False),),
        ),
        MetaAnalysisOperation(
            "validate-sealed-meta-events",
            ("NB-LIVE-0010-C0003", "NB-LIVE-0010-C0004"),
            ("review_root",),
            "sealed_meta_events",
            (("implicit_dtype_inference", False),),
        ),
        MetaAnalysisOperation(
            "build-effective-overrides",
            ("NB-LIVE-0010-C0003", "NB-LIVE-0010-C0004"),
            ("sealed_meta_events",),
            "effective_overrides",
        ),
        MetaAnalysisOperation(
            "select-decisive-disagreements",
            ("NB-LIVE-0010-C0005", "NB-LIVE-0010-C0014"),
            ("effective_overrides",),
            "decisive_disagreements",
        ),
        MetaAnalysisOperation(
            "validate-sealed-review-features",
            ("NB-LIVE-0010-C0006",),
            ("review_root",),
            "feature_rows",
            (("implicit_feature_discovery", False),),
        ),
        MetaAnalysisOperation(
            "merge-features-by-identity",
            ("NB-LIVE-0010-C0007",),
            ("effective_overrides", "feature_rows"),
            "merged_feature_rows",
        ),
        MetaAnalysisOperation(
            "match-features-by-click",
            ("NB-LIVE-0010-C0010",),
            ("effective_overrides", "feature_rows"),
            "click_matched_rows",
        ),
        MetaAnalysisOperation(
            "refine-click-matches",
            ("NB-LIVE-0010-C0012",),
            ("click_matched_rows",),
            "matched_feature_rows",
        ),
        MetaAnalysisOperation(
            "rank-disagreement-features",
            ("NB-LIVE-0010-C0008", "NB-LIVE-0010-C0011", "NB-LIVE-0010-C0013"),
            ("matched_feature_rows",),
            "feature_ranks",
        ),
        MetaAnalysisOperation(
            "audit-identity-ranges",
            ("NB-LIVE-0010-C0009",),
            ("matched_feature_rows",),
            "identity_audit",
        ),
        MetaAnalysisOperation(
            "sweep-zone-features",
            ("NB-LIVE-0010-C0015",),
            ("decisive_disagreements",),
            "rule_sweeps",
            (("minimum_count", 80),),
        ),
        MetaAnalysisOperation(
            "combine-zone-rules",
            ("NB-LIVE-0010-C0016", "NB-LIVE-0010-C0017"),
            ("decisive_disagreements", "rule_sweeps"),
            "combined_rules",
            (("minimum_count", 80),),
        ),
        MetaAnalysisOperation(
            "evaluate-context-rules",
            ("NB-LIVE-0010-C0018", "NB-LIVE-0010-C0019"),
            ("matched_feature_rows",),
            "context_metrics",
        ),
        MetaAnalysisOperation(
            "fit-sparse-meta-model",
            ("NB-LIVE-0010-C0020",),
            ("matched_feature_rows",),
            "sparse_meta_model",
        ),
        MetaAnalysisOperation(
            "write-declared-meta-artifacts",
            tuple(f"NB-LIVE-0010-C{index:04d}" for index in range(21)),
            (
                "raw_review_rows",
                "sealed_meta_events",
                "review_events",
                "effective_overrides",
                "analysis_results",
            ),
            "output",
        ),
    )


# SOURCE_CELL: NB-LIVE-0010-C0008
# SOURCE_CELL: NB-LIVE-0010-C0011
# SOURCE_CELL: NB-LIVE-0010-C0013
# SOURCE_STATEMENT_MAP: structural-pass -> _explicit_noop
def _explicit_noop() -> None:
    """Represent an intentional source no-op as an explicit typed operation."""

    return None


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_sealed_meta_child(
    root: Path,
    name: str,
    role: str,
) -> tuple[bytes, tuple[int, int, int, int, int, int]]:
    """Read one exact direct child through a non-following stable descriptor."""

    if Path(name).name != name or name in {"", ".", ".."}:
        raise MetaAnalysisContractError(f"meta-analysis {role} name is not portable")
    path = root / name
    try:
        before = path.lstat()
    except OSError as exc:
        raise MetaAnalysisContractError(f"meta-analysis {role} is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise MetaAnalysisContractError(f"meta-analysis {role} must be a regular file")
    if path.resolve(strict=True).parent != root:
        raise MetaAnalysisContractError(f"meta-analysis {role} escaped review root")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise MetaAnalysisContractError(f"meta-analysis {role} cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _stat_identity(opened) != _stat_identity(before):
            raise MetaAnalysisContractError(f"meta-analysis {role} changed before reading")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        completed = os.fstat(descriptor)
        if _stat_identity(completed) != _stat_identity(opened):
            raise MetaAnalysisContractError(f"meta-analysis {role} changed while reading")
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise MetaAnalysisContractError(f"meta-analysis {role} changed after reading") from exc
    identity = _stat_identity(after)
    if identity != _stat_identity(completed):
        raise MetaAnalysisContractError(f"meta-analysis {role} changed after reading")
    return b"".join(chunks), identity


def _require_meta_sha256(value: object, role: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MetaAnalysisContractError(f"meta-analysis {role} SHA-256 is invalid")
    return value


# SOURCE_CELL: NB-LIVE-0010-C0003
# SOURCE_CELL: NB-LIVE-0010-C0004
# SOURCE_STATEMENT_MAP: declared-dataframe-dtypes-and-raw-csv -> validate_meta_event_batch
def validate_meta_event_batch(
    review_root: Path,
) -> MetaEventBatch:
    """Validate the exact event table, source dtypes, grammar, and SHA contract."""

    if review_root.is_symlink() or not review_root.is_dir():
        raise MetaAnalysisContractError("review root must be a regular directory")
    root = review_root.resolve(strict=True)
    manifest_bytes, _ = _read_sealed_meta_child(
        root,
        META_EVENT_MANIFEST_NAME,
        "event manifest",
    )
    try:
        contract = loads(manifest_bytes.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise MetaAnalysisContractError("meta-analysis event manifest is invalid JSON") from exc
    if not isinstance(contract, dict) or set(contract) != {
        "schema",
        "events",
        "columns",
        "column_dtypes",
        "null_tokens",
    }:
        raise MetaAnalysisContractError("meta-analysis event manifest fields changed")
    if contract["schema"] != META_EVENT_SCHEMA:
        raise MetaAnalysisContractError("meta-analysis event manifest schema changed")
    artifact = contract["events"]
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise MetaAnalysisContractError("meta-analysis event artifact fields changed")
    if artifact["path"] != META_EVENT_TABLE_NAME:
        raise MetaAnalysisContractError("meta-analysis event table name changed")
    expected_sha256 = _require_meta_sha256(artifact["sha256"], "event table")

    declared_columns = contract["columns"]
    if (
        not isinstance(declared_columns, list)
        or any(type(name) is not str for name in declared_columns)
        or len(declared_columns) != len(set(declared_columns))
    ):
        raise MetaAnalysisContractError("meta-analysis event columns are invalid")
    columns = tuple(declared_columns)
    expected_columns = tuple(name for name in META_EVENT_COLUMNS if name in columns)
    if (
        columns != expected_columns
        or not META_EVENT_REQUIRED_COLUMNS.issubset(columns)
        or any(name not in META_EVENT_COLUMNS for name in columns)
    ):
        raise MetaAnalysisContractError("meta-analysis event columns changed")

    dtype_contract = contract["column_dtypes"]
    if not isinstance(dtype_contract, dict) or set(dtype_contract) != {
        "timestamp",
        "source_mtime",
        "text_timestamp_grammar",
    }:
        raise MetaAnalysisContractError("meta-analysis event dtype fields changed")
    column_dtypes = MetaEventColumnDtypes(
        timestamp=dtype_contract["timestamp"],
        source_mtime=dtype_contract["source_mtime"],
        text_timestamp_grammar=dtype_contract["text_timestamp_grammar"],
    )
    if contract["null_tokens"] != list(META_EVENT_NULL_TOKENS):
        raise MetaAnalysisContractError("meta-analysis event null-token contract changed")

    event_bytes, before_identity = _read_sealed_meta_child(
        root,
        META_EVENT_TABLE_NAME,
        "event table",
    )
    observed_sha256 = sha256(event_bytes).hexdigest()
    if observed_sha256 != expected_sha256:
        raise MetaAnalysisContractError("meta-analysis event table SHA-256 mismatch")
    event_path = root / META_EVENT_TABLE_NAME
    try:
        reader = DictReader(StringIO(event_bytes.decode("utf-8-sig"), newline=""))
        if tuple(reader.fieldnames or ()) != columns:
            raise MetaAnalysisContractError("meta-analysis event table schema changed")
        rows = [dict(row) for row in reader]
    except UnicodeError as exc:
        raise MetaAnalysisContractError(
            "meta-analysis event table is not valid UTF-8 CSV"
        ) from exc
    if not rows:
        raise MetaAnalysisContractError("meta-analysis event table must contain rows")
    if any(tuple(row) != columns for row in rows):
        raise MetaAnalysisContractError("meta-analysis event table schema changed")
    after_bytes, after_identity = _read_sealed_meta_child(
        root,
        META_EVENT_TABLE_NAME,
        "event table",
    )
    if after_identity != before_identity or after_bytes != event_bytes:
        raise MetaAnalysisContractError("meta-analysis event table changed during decoding")
    return MetaEventBatch(
        root / META_EVENT_MANIFEST_NAME,
        sha256(manifest_bytes).hexdigest(),
        event_path,
        observed_sha256,
        columns,
        column_dtypes,
        tuple(dict(row) for row in rows),
    )


def _canonical_meta_scalar(value: object, role: str) -> str | int | float | bool | None:
    """Map source missing values to None without coercing other scalar types."""

    if value is None:
        return None
    if type(value) is float and isnan(value):
        return None
    if type(value) is str:
        return None if value in META_EVENT_NULL_TOKENS else value
    if type(value) in {int, float, bool}:
        return value
    raise MetaAnalysisContractError(f"meta-analysis {role} has an invalid scalar type")


def _canonical_meta_timestamp(
    value: object,
    storage: str,
    role: str,
) -> str | int | float | None:
    canonical = _canonical_meta_scalar(value, role)
    if canonical is None:
        return None
    if type(canonical) is bool:
        raise MetaAnalysisContractError(f"meta-analysis {role} cannot be boolean")
    if storage == "text" and type(canonical) is not str:
        raise MetaAnalysisContractError(
            f"meta-analysis {role} contradicts its declared text dtype"
        )
    if storage == "numeric":
        if type(canonical) in {int, float}:
            return canonical
        if type(canonical) is not str:
            raise MetaAnalysisContractError(
                f"meta-analysis {role} contradicts its declared numeric dtype"
            )
        text = canonical.strip()
        try:
            Decimal(text)
        except InvalidOperation as exc:
            raise MetaAnalysisContractError(
                f"meta-analysis {role} contradicts its declared numeric dtype"
            ) from exc
        signed_digits = text.removeprefix("+").removeprefix("-")
        return int(text) if signed_digits.isascii() and signed_digits.isdigit() else float(text)
    return canonical


def _canonical_row_ordinal(value: object) -> int:
    canonical = _canonical_meta_scalar(value, "row ordinal")
    if type(canonical) is int:
        ordinal = canonical
    elif type(canonical) is str and canonical.isascii() and canonical.isdigit():
        ordinal = int(canonical)
    else:
        raise MetaAnalysisContractError("meta-analysis row ordinal is invalid")
    if ordinal < 0:
        raise MetaAnalysisContractError("meta-analysis row ordinal is invalid")
    return ordinal


def _adapt_meta_event_rows(batch: MetaEventBatch) -> tuple[MetaEventInputRow, ...]:
    """Adapt sealed table rows without dtype inference or local-state capture."""

    xgb_pred_present = "xgb_pred" in batch.columns
    final_pred_present = "final_pred" in batch.columns
    adapted: list[MetaEventInputRow] = []
    for row in batch.rows:
        image = _canonical_meta_scalar(row["image"], "image")
        identifier = _canonical_meta_scalar(row["id"], "identifier")
        if type(image) is not str or type(identifier) is not str:
            raise MetaAnalysisContractError(
                "meta-analysis event image and identifier must be stored as strings"
            )
        adapted.append(
            MetaEventInputRow(
                image=image,
                identifier=identifier,
                timestamp=_canonical_meta_timestamp(
                    row["timestamp"],
                    batch.column_dtypes.timestamp,
                    "timestamp",
                ),
                source_mtime=_canonical_meta_timestamp(
                    row["source_mtime"],
                    batch.column_dtypes.source_mtime,
                    "source mtime",
                ),
                row_in_file=_canonical_row_ordinal(row["_row_in_file"]),
                human_label=_canonical_meta_scalar(row["human_label"], "human label"),
                xgb_prob=_canonical_meta_scalar(row["xgb_prob"], "XGB probability"),
                yolo_conf=_canonical_meta_scalar(row["yolo_conf"], "YOLO confidence"),
                yolo_iou=_canonical_meta_scalar(row["yolo_iou"], "YOLO IoU"),
                thr_xgb=_canonical_meta_scalar(row["thr_xgb"], "XGB threshold"),
                thr_yolo=_canonical_meta_scalar(row["thr_yolo"], "YOLO threshold"),
                thr_iou=_canonical_meta_scalar(row["thr_iou"], "IoU threshold"),
                detector_threshold=_canonical_meta_scalar(
                    row["det_thr"],
                    "detector threshold",
                ),
                xgb_pred=(
                    _canonical_meta_scalar(row["xgb_pred"], "XGB prediction")
                    if xgb_pred_present
                    else None
                ),
                xgb_pred_present=xgb_pred_present,
                final_pred=(
                    _canonical_meta_scalar(row["final_pred"], "final prediction")
                    if final_pred_present
                    else None
                ),
                final_pred_present=final_pred_present,
                kept=_canonical_meta_scalar(row["kept"], "kept prediction"),
                detector_policy=(
                    None
                    if _canonical_meta_scalar(row["det_policy"], "detector policy") is None
                    else str(_canonical_meta_scalar(row["det_policy"], "detector policy"))
                ),
            )
        )
    return tuple(adapted)


# SOURCE_CELL: NB-LIVE-0010-C0003
# SOURCE_CELL: NB-LIVE-0010-C0004
# SOURCE_STATEMENT_MAP: event-formulas-and-cell-order -> evaluate_meta_event_batch
def evaluate_meta_event_batch(batch: MetaEventBatch) -> MetaEventEvaluation:
    """Run the exact C0003 then C0004 typed event algorithms."""

    rows = _adapt_meta_event_rows(batch)
    c0003 = evaluate_meta_events_c0003(rows, batch.column_dtypes)
    c0004 = evaluate_meta_events_c0004(rows, batch.column_dtypes)
    return MetaEventEvaluation(batch, c0003, c0004)


def _render_event_time(epoch_nanoseconds: int | None) -> str | None:
    if epoch_nanoseconds is None:
        return None
    seconds, remainder = divmod(epoch_nanoseconds, 1_000_000_000)
    value = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if remainder:
        value += "." + f"{remainder:09d}".rstrip("0")
    return value + "+00:00"


def _output_meta_scalar(value: object) -> object:
    return None if type(value) is float and isnan(value) else value


def _meta_event_record_row(
    event: MetaEventRecord,
    columns: tuple[str, ...],
    variant: str,
) -> Row:
    source = event.source
    source_values: dict[str, object] = {
        "image": source.image,
        "id": source.identifier,
        "timestamp": source.timestamp,
        "source_mtime": source.source_mtime,
        "_row_in_file": source.row_in_file,
        "human_label": source.human_label,
        "xgb_prob": source.xgb_prob,
        "yolo_conf": source.yolo_conf,
        "yolo_iou": source.yolo_iou,
        "thr_xgb": source.thr_xgb,
        "thr_yolo": source.thr_yolo,
        "thr_iou": source.thr_iou,
        "det_thr": source.detector_threshold,
        "xgb_pred": source.xgb_pred,
        "final_pred": source.final_pred,
        "kept": source.kept,
        "det_policy": source.detector_policy,
    }
    row: Row = {name: _output_meta_scalar(source_values[name]) for name in columns}
    row.update(
        {
            "item_key": event.item_key,
            "event_time": _render_event_time(event.event_time_nanoseconds),
            "source_mtime_dt": _render_event_time(event.source_mtime_nanoseconds),
            "human_label_n": _output_meta_scalar(event.human_label_n),
            "human_last": _output_meta_scalar(event.human_last),
            "human_prev": _output_meta_scalar(event.human_prev),
            "label_changed": event.label_changed,
            "xgb_prob_f": _output_meta_scalar(event.xgb_prob_f),
            "yolo_conf_f": _output_meta_scalar(event.yolo_conf_f),
            "yolo_iou_f": _output_meta_scalar(event.yolo_iou_f),
            "xgb_pred_eff": _output_meta_scalar(event.xgb_pred_eff),
            "yolo_pred_eff": event.yolo_pred_eff,
            "final_pred_eff": _output_meta_scalar(event.final_pred_eff),
        }
    )
    if variant == "c0004":
        row["kept_eff"] = _output_meta_scalar(event.kept_eff)
    return row


def _meta_event_tables(
    result: MetaEventStatisticsResult,
    columns: tuple[str, ...],
) -> tuple[tuple[Row, ...], tuple[Row, ...], tuple[Row, ...], tuple[Row, ...]]:
    events = tuple(
        _meta_event_record_row(event, columns, result.source_variant)
        for event in result.events
    )
    flips = tuple(
        _meta_event_record_row(event, columns, result.source_variant)
        for event in result.flips
    )
    latest = tuple(
        _meta_event_record_row(event, columns, result.source_variant)
        for event in result.latest
    )
    override_rows: list[Row] = []
    for override in result.overrides:
        row = _meta_event_record_row(
            override.event,
            columns,
            result.source_variant,
        )
        if result.source_variant == "c0003":
            row["det_policy_s"] = override.detector_policy
        row.update(
            {
                "human_ok": override.human_ok,
                "override_final": override.override_final,
                "agree_yolo": override.agree_yolo,
                "agree_xgb": override.agree_xgb,
                "align_bucket": override.align_bucket,
            }
        )
        override_rows.append(row)
    return events, flips, latest, tuple(override_rows)


def _require_meta_output_root(
    output: Path,
    expected: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    """Bind every writer phase to the uniquely created output directory."""

    try:
        info = output.lstat()
        resolved = output.resolve(strict=True)
    except OSError as exc:
        raise MetaAnalysisContractError(
            "meta-analysis output root is unavailable"
        ) from exc
    if output.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise MetaAnalysisContractError(
            "meta-analysis output root must remain a regular non-symlink directory"
        )
    if resolved != Path(os.path.abspath(os.fspath(output))):
        raise MetaAnalysisContractError("meta-analysis output root locator changed")
    current = (info.st_dev, info.st_ino, info.st_mode)
    if expected is not None and current != expected:
        raise MetaAnalysisContractError("meta-analysis output root identity changed")
    return current


# SOURCE_CELL: NB-LIVE-0010-C0003
# SOURCE_CELL: NB-LIVE-0010-C0004
# SOURCE_STATEMENT_MAP: four-declared-tables-c0003-then-c0004 -> write_meta_event_outputs
def write_meta_event_outputs(
    output: Path,
    services: MetaAnalysisServices,
    evaluation: MetaEventEvaluation,
    *,
    output_identity: tuple[int, int, int] | None = None,
) -> tuple[Path, ...]:
    """Write both source-cell results in order; C0004 is the final state."""

    bound_output_identity = _require_meta_output_root(output, output_identity)
    paths = tuple(output / name for name in META_EVENT_OUTPUT_NAMES)
    for result in (evaluation.c0003, evaluation.c0004):
        for path, rows in zip(
            paths,
            _meta_event_tables(result, evaluation.batch.columns),
            strict=True,
        ):
            services.write_rows(path, rows)
            _require_meta_output_root(output, bound_output_identity)
    return paths


POSITIVE_LABELS = {
    "1",
    "1.0",
    "true",
    "t",
    "yes",
    "y",
    "pos",
    "positive",
    "keep",
    "kept",
    "accept",
    "accepted",
}
NEGATIVE_LABELS = {
    "0",
    "0.0",
    "false",
    "f",
    "no",
    "n",
    "neg",
    "negative",
    "reject",
    "rejected",
    "drop",
    "remove",
}


def _float(value: Any, default: float = nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if isfinite(parsed) else default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _stem(value: Any) -> str:
    return Path(str(value)).stem


# SOURCE_CELL: NB-LIVE-0010-C0002
# SOURCE_CELL: NB-LIVE-0010-C0004
# SOURCE_STATEMENT_MAP: numeric-and-string-01-normalizer -> normalize_binary_label
def normalize_binary_label(value: Any) -> int | None:
    text = str(value).strip().lower()
    if text in POSITIVE_LABELS:
        return 1
    if text in NEGATIVE_LABELS:
        return 0
    numeric = _float(value)
    if numeric in (0.0, 1.0):
        return int(numeric)
    return None


def _timestamp_rank(value: Any, fallback: float) -> tuple[int, float | str]:
    numeric = _float(value)
    if isfinite(numeric):
        return 2, numeric
    text = str(value).strip()
    if text:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return 2, parsed.timestamp()
        except ValueError:
            return 1, text
    return 0, fallback


# SOURCE_CELL: NB-LIVE-0010-C0000
# SOURCE_STATEMENT_MAP: recursive-discovery-provenance-concat-latest -> collect_review_logs
def collect_review_logs(
    review_root: Path,
    services: MetaAnalysisServices,
) -> tuple[list[Row], list[Row], list[Row]]:
    """Collect all non-empty review logs and deduplicate by source key candidates."""

    if not review_root.is_dir():
        raise MetaAnalysisContractError(f"review root is not a directory: {review_root}")
    raw_rows: list[Row] = []
    manifest: list[Row] = []
    for path in sorted(review_root.rglob("review_labels.csv")):
        try:
            rows = services.read_rows(path)
        except (OSError, MetaAnalysisContractError, UnicodeError) as exc:
            manifest.append(
                {"path": str(path), "rows": 0, "status": f"failed:{type(exc).__name__}"}
            )
            continue
        if not rows:
            manifest.append({"path": str(path), "rows": 0, "status": "empty"})
            continue
        relative = path.relative_to(review_root)
        modified = float(path.stat().st_mtime)
        for ordinal, row in enumerate(rows):
            normalized = {str(key).strip(): value for key, value in row.items()}
            raw_rows.append(
                {
                    "source_path": str(path),
                    "source_rel": str(relative.parent),
                    "source_mtime": modified,
                    "_row_in_file": ordinal,
                    **normalized,
                }
            )
        manifest.append({"path": str(path), "rows": len(rows), "status": "ok"})
    if not raw_rows:
        raise MetaAnalysisContractError("no readable non-empty review_labels.csv files found")

    key_options = (("file_name", "review_id"), ("file_name", "ann_id"))
    key_columns = next(
        (
            option
            for option in key_options
            if all(all(column in row for column in option) for row in raw_rows)
        ),
        None,
    )
    if key_columns is None:
        latest = [dict(row) for row in raw_rows]
    else:
        selected: dict[tuple[str, str], tuple[tuple[Any, ...], Row]] = {}
        timestamp_columns = ("timestamp", "ts", "updated_at", "created_at", "time")
        for row in raw_rows:
            timestamp_name = next((name for name in timestamp_columns if name in row), None)
            rank = _timestamp_rank(
                row.get(timestamp_name) if timestamp_name else None,
                _float(row.get("source_mtime"), -1.0),
            )
            order = (
                rank,
                _float(row.get("source_mtime"), -1.0),
                int(_float(row.get("_row_in_file"), -1.0)),
            )
            key = tuple(str(row[column]) for column in key_columns)
            if key not in selected or order >= selected[key][0]:
                selected[key] = (order, row)
        latest = []
        for _, row in selected.values():
            latest.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"_row_in_file", "_ts_parsed", "_ts_rank"}
                }
            )
    return raw_rows, latest, manifest


# SOURCE_CELL: NB-LIVE-0010-C0001
# SOURCE_STATEMENT_MAP: id-like-column-scan-pair-coverage -> candidate_key_statistics
def candidate_key_statistics(rows: Sequence[Mapping[str, Any]]) -> tuple[KeyPairStatistic, ...]:
    if not rows:
        return ()
    columns = list(dict.fromkeys(key for row in rows for key in row))
    terms = (
        "file",
        "name",
        "path",
        "tile",
        "img",
        "image",
        "review",
        "rid",
        "uuid",
        "guid",
        "ann",
        "annot",
        "id",
    )
    excluded = {"source_path", "source_rel", "source_mtime", "_row_in_file"}
    candidates = [
        column
        for column in columns
        if column not in excluded and any(term in column.lower() for term in terms)
    ][:14]
    stats: list[KeyPairStatistic] = []
    for first, second in combinations(candidates, 2):
        present = [
            (str(row[first]), str(row[second]))
            for row in rows
            if row.get(first) not in (None, "") and row.get(second) not in (None, "")
        ]
        if not present:
            continue
        unique = len(set(present))
        stats.append(
            KeyPairStatistic(
                first,
                second,
                len(present) / len(rows),
                len(present),
                unique,
                len(present) - unique,
            )
        )
    return tuple(sorted(stats, key=lambda item: (item.coverage, item.duplicate_rows), reverse=True))


def _event_order(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _timestamp_rank(row.get("timestamp"), _float(row.get("source_mtime"), -1.0)),
        _float(row.get("source_mtime"), -1.0),
        int(_float(row.get("_row_in_file"), -1.0)),
    )


# SOURCE_CELL: NB-LIVE-0010-C0002
# SOURCE_STATEMENT_MAP: item-key-sort-forward-label-flip-final -> build_review_event_history
def build_review_event_history(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_source_folder: bool,
) -> tuple[list[Row], list[Row], list[Row]]:
    events: list[Row] = []
    for row in rows:
        if "image" not in row or "id" not in row:
            raise MetaAnalysisContractError("review events require image and id")
        folder = str(row.get("source_rel", "")).lstrip("./").split("/", 1)[0]
        key_parts = [str(row["image"]), str(row["id"])]
        if include_source_folder:
            key_parts.insert(0, folder)
        normalized = dict(row)
        normalized["img_folder"] = folder
        normalized["item_key"] = "||".join(key_parts)
        normalized["human_label_n"] = normalize_binary_label(row.get("human_label"))
        events.append(normalized)
    events.sort(key=lambda row: (str(row["item_key"]), _event_order(row)))

    last_label: dict[str, int | None] = {}
    flips: list[Row] = []
    for event in events:
        key = str(event["item_key"])
        current = event["human_label_n"]
        previous = last_label.get(key)
        event["human_prev"] = previous
        if current is None:
            event["human_last"] = previous
        else:
            event["human_last"] = current
            last_label[key] = current
        event["label_changed"] = (
            event["human_last"] is not None
            and previous is not None
            and event["human_last"] != previous
        )
        if event["label_changed"]:
            flips.append(dict(event))
    latest_by_key: dict[str, Row] = {}
    for event in events:
        latest_by_key[str(event["item_key"])] = dict(event)
    return events, flips, list(latest_by_key.values())


def _effective_threshold(row: Mapping[str, Any], primary: str, fallback: str, default: float) -> float:
    value = _float(row.get(primary))
    if isfinite(value):
        return value
    value = _float(row.get(fallback))
    return value if isfinite(value) else default


# SOURCE_CELL: NB-LIVE-0010-C0003
# SOURCE_CELL: NB-LIVE-0010-C0004
# SOURCE_STATEMENT_MAP: effective-model-preds-latest-alignment -> build_override_snapshot
def build_override_snapshot(events: Sequence[Mapping[str, Any]]) -> list[Row]:
    latest: dict[str, Row] = {}
    for source in events:
        row = dict(source)
        xgb_probability = _float(row.get("xgb_prob"))
        yolo_confidence = _float(row.get("yolo_conf"))
        yolo_iou = _float(row.get("yolo_iou"))
        xgb_pred = normalize_binary_label(row.get("xgb_pred"))
        if xgb_pred is None and isfinite(xgb_probability):
            threshold = _effective_threshold(row, "thr_xgb", "det_thr", 0.5)
            xgb_pred = int(xgb_probability >= threshold)
        yolo_threshold = _effective_threshold(row, "thr_yolo", "det_thr", 0.5)
        iou_threshold = _effective_threshold(row, "thr_iou", "", 0.70)
        yolo_pred = (
            int(yolo_confidence >= yolo_threshold and yolo_iou >= iou_threshold)
            if isfinite(yolo_confidence) and isfinite(yolo_iou)
            else None
        )
        final_pred = normalize_binary_label(row.get("final_pred"))
        if final_pred is None:
            final_pred = normalize_binary_label(row.get("kept"))
        row.update(
            {
                "xgb_prob_f": xgb_probability,
                "yolo_conf_f": yolo_confidence,
                "yolo_iou_f": yolo_iou,
                "xgb_pred_eff": xgb_pred,
                "yolo_pred_eff": yolo_pred,
                "final_pred_eff": final_pred,
            }
        )
        latest[str(row["item_key"])] = row

    output: list[Row] = []
    for row in latest.values():
        human = normalize_binary_label(row.get("human_last"))
        xgb = normalize_binary_label(row.get("xgb_pred_eff"))
        yolo = normalize_binary_label(row.get("yolo_pred_eff"))
        final = normalize_binary_label(row.get("final_pred_eff"))
        human_ok = human is not None
        agree_yolo = human_ok and yolo is not None and human == yolo
        agree_xgb = human_ok and xgb is not None and human == xgb
        if agree_yolo and not agree_xgb:
            bucket = "human->yolo"
        elif agree_xgb and not agree_yolo:
            bucket = "human->xgb"
        elif agree_xgb and agree_yolo:
            bucket = "human->both"
        elif human_ok:
            bucket = "human->neither"
        else:
            bucket = "no_human"
        row.update(
            {
                "human_ok": human_ok,
                "override_final": human_ok and final is not None and human != final,
                "agree_yolo": agree_yolo,
                "agree_xgb": agree_xgb,
                "align_bucket": bucket,
            }
        )
        output.append(row)
    return output


# SOURCE_CELL: NB-LIVE-0010-C0005
# SOURCE_CELL: NB-LIVE-0010-C0014
# SOURCE_STATEMENT_MAP: disagreement-and-override-zone -> decisive_disagreements
def decisive_disagreements(rows: Sequence[Mapping[str, Any]]) -> list[Row]:
    output: list[Row] = []
    for source in rows:
        xgb = normalize_binary_label(source.get("xgb_pred_eff"))
        yolo = normalize_binary_label(source.get("yolo_pred_eff"))
        if xgb is None or yolo is None or xgb == yolo:
            continue
        if source.get("align_bucket") not in {"human->xgb", "human->yolo"}:
            continue
        row = dict(source)
        yolo_conf = _float(row.get("yolo_conf_f", row.get("yolo_conf")))
        yolo_iou = _float(row.get("yolo_iou_f", row.get("yolo_iou")))
        xgb_prob = _float(row.get("xgb_prob_f", row.get("xgb_prob")))
        difference = abs(yolo_conf - xgb_prob)
        row["diff_p"] = difference
        row["zone"] = (
            isfinite(yolo_conf)
            and isfinite(yolo_iou)
            and isfinite(xgb_prob)
            and yolo_conf >= 0.65
            and yolo_iou >= 0.70
            and xgb_prob <= 0.35
            and difference >= 0.25
        )
        output.append(row)
    return output


# SOURCE_CELL: NB-LIVE-0010-C0006
# SOURCE_STATEMENT_MAP: implicit-discovery -> validate_meta_feature_batch
def validate_meta_feature_batch(
    review_root: Path,
    services: MetaAnalysisServices,
) -> MetaFeatureBatch:
    """Load the one explicitly named feature table after exact SHA validation."""

    if review_root.is_symlink() or not review_root.is_dir():
        raise MetaAnalysisContractError("review root must be a regular directory")
    root = review_root.resolve(strict=True)
    manifest = review_root / "meta_analysis_features_manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise MetaAnalysisContractError("meta-analysis feature manifest is missing")
    if manifest.resolve(strict=True).parent != root:
        raise MetaAnalysisContractError("meta-analysis feature manifest escaped review root")
    try:
        contract = loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MetaAnalysisContractError("meta-analysis feature manifest is invalid JSON") from exc
    if not isinstance(contract, dict) or set(contract) != {"schema", "features"}:
        raise MetaAnalysisContractError("meta-analysis feature manifest fields changed")
    if contract["schema"] != "compag-curation-meta-features/v1":
        raise MetaAnalysisContractError("meta-analysis feature manifest schema changed")
    artifact = contract["features"]
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise MetaAnalysisContractError("meta-analysis feature artifact fields changed")
    relative = Path(str(artifact["path"]))
    expected_sha256 = str(artifact["sha256"])
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name != "meta_analysis_features.csv"
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise MetaAnalysisContractError("meta-analysis feature artifact contract is invalid")
    features = review_root / relative
    if features.is_symlink() or not features.is_file():
        raise MetaAnalysisContractError("meta-analysis features must be a regular file")
    if features.resolve(strict=True).parent != root:
        raise MetaAnalysisContractError("meta-analysis features escaped review root")
    actual_sha256 = sha256(features.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise MetaAnalysisContractError("meta-analysis feature SHA-256 mismatch")
    rows = services.read_rows(features)
    required = {
        "file_name",
        "ann_id",
        "scale",
        "cx",
        "cy",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    }
    if not rows or any(not required.issubset(row) for row in rows):
        raise MetaAnalysisContractError("meta-analysis feature table schema changed")
    if any(
        not str(row["file_name"]).strip()
        or not all(
            isfinite(_float(row[name]))
            for name in required - {"file_name"}
        )
        for row in rows
    ):
        raise MetaAnalysisContractError("meta-analysis feature table values are invalid")
    return MetaFeatureBatch(
        manifest,
        sha256(manifest.read_bytes()).hexdigest(),
        features,
        actual_sha256,
        tuple(rows),
    )


# Provenance-only disposition: the runtime never calls this auto-discovery helper.
# SOURCE_CELL: NB-LIVE-0010-C0006
# SOURCE_STATEMENT_MAP: recursive-feature-discovery-newest -> validate_meta_feature_batch
def discover_feature_table(search_roots: Iterable[Path]) -> Path:
    candidates: set[Path] = set()
    for root in search_roots:
        if root.exists():
            candidates.update(root.rglob("features_train.csv"))
            candidates.update(root.rglob("features_train*.csv"))
    ordered = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    if not ordered:
        raise MetaAnalysisContractError("no features_train*.csv found")
    return ordered[0]


def _best_scale_rows(features: Sequence[Mapping[str, Any]]) -> list[Row]:
    selected: dict[tuple[str, int], tuple[float, Row]] = {}
    for source in features:
        image_name = source.get("file_name", source.get("image"))
        identity = _float(
            source.get(
                "review_id",
                source.get("id", source.get("ann_id", source.get("annotation_id"))),
            )
        )
        if image_name is None or not isfinite(identity):
            continue
        row = dict(source)
        key = (_stem(image_name), int(identity))
        scale = _float(row.get("scale"), 1.0)
        distance = abs(scale - 1.0) if isfinite(scale) else 1e30
        if key not in selected or distance < selected[key][0]:
            row["image_key"] = key[0]
            row["ann_id_key"] = key[1]
            selected[key] = (distance, row)
    return [item[1] for item in selected.values()]


# SOURCE_CELL: NB-LIVE-0010-C0007
# SOURCE_STATEMENT_MAP: normalized-image-annid-scale-nearest-merge -> merge_features_by_identity
def merge_features_by_identity(
    overrides: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
) -> list[Row]:
    feature_index = {
        (str(row["image_key"]), int(row["ann_id_key"])): row
        for row in _best_scale_rows(features)
    }
    merged: list[Row] = []
    for source in overrides:
        row = dict(source)
        identity = _float(row.get("id"))
        key = (_stem(row.get("image", "")), int(identity)) if isfinite(identity) else None
        feature = feature_index.get(key) if key is not None else None
        row["image_key"] = key[0] if key else _stem(row.get("image", ""))
        row["ann_id_key"] = key[1] if key else None
        row["_merge"] = "both" if feature is not None else "left_only"
        if feature is not None:
            for name, value in feature.items():
                target = name if name not in row else f"{name}_feat"
                row[target] = value
        merged.append(row)
    return merged


def _auc(labels: Sequence[int], values: Sequence[float]) -> float:
    pairs = [(value, label) for value, label in zip(values, labels) if isfinite(value)]
    positives = sum(label == 1 for _, label in pairs)
    negatives = sum(label == 0 for _, label in pairs)
    if positives == 0 or negatives == 0:
        return nan
    ordered = sorted(pairs, key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(label == 1 for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


LEAKAGE_EXACT = {
    "image",
    "id",
    "item_key",
    "image_key",
    "id_key",
    "ann_id",
    "file_name",
    "scale",
    "xgb_pred_eff",
    "yolo_pred_eff",
    "final_pred_eff",
    "xgb_prob_f",
    "yolo_conf_f",
    "yolo_iou_f",
    "xgb_prob",
    "yolo_conf",
    "yolo_iou",
    "prob",
    "label",
    "class_id",
}
LEAKAGE_SUBSTRINGS = (
    "agree_",
    "align_",
    "human_",
    "override",
    "final_",
    "kept",
    "review",
    "timestamp",
    "source_",
    "_row_",
    "_match",
    "det_",
    "thr_",
    "policy",
)


# SOURCE_CELL: NB-LIVE-0010-C0008
# SOURCE_CELL: NB-LIVE-0010-C0011
# SOURCE_CELL: NB-LIVE-0010-C0013
# SOURCE_STATEMENT_MAP: decisive-filter-coverage-auc-leakage-ban -> rank_disagreement_features
def rank_disagreement_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    pure_features_only: bool = True,
) -> tuple[FeatureRank, ...]:
    decisive = [
        row
        for row in rows
        if row.get("align_bucket") in {"human->xgb", "human->yolo"}
    ]
    labels = [int(row.get("align_bucket") == "human->yolo") for row in decisive]
    columns = list(dict.fromkeys(key for row in decisive for key in row))
    ranks: list[FeatureRank] = []
    for column in columns:
        lowered = column.lower()
        if pure_features_only and (
            column in LEAKAGE_EXACT
            or any(term in lowered for term in LEAKAGE_SUBSTRINGS)
            or lowered.startswith(("xgb_", "yolo_"))
        ):
            continue
        values = [_float(row.get(column)) for row in decisive]
        coverage = sum(isfinite(value) for value in values) / len(values) if values else 0.0
        if coverage < 0.95:
            continue
        auc = _auc(labels, values)
        if not isfinite(auc):
            continue
        ranks.append(
            FeatureRank(
                column,
                auc,
                abs(auc - 0.5),
                "higher->yolo" if auc > 0.5 else "lower->yolo",
            )
        )
    return tuple(sorted(ranks, key=lambda item: item.strength, reverse=True))


# SOURCE_CELL: NB-LIVE-0010-C0009
# SOURCE_STATEMENT_MAP: id-range-and-feature-name-audit -> audit_identity_ranges
def audit_identity_ranges(
    overrides: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    override_ids = [_float(row.get("id")) for row in overrides]
    feature_ids = [_float(row.get("ann_id")) for row in features]
    override_ids = [value for value in override_ids if isfinite(value)]
    feature_ids = [value for value in feature_ids if isfinite(value)]
    return {
        "override_unique": len(set(override_ids)),
        "override_min": min(override_ids, default=nan),
        "override_max": max(override_ids, default=nan),
        "feature_unique": len(set(feature_ids)),
        "feature_min": min(feature_ids, default=nan),
        "feature_max": max(feature_ids, default=nan),
        "feature_name_examples": [
            str(row.get("file_name")) for row in features[:10] if row.get("file_name")
        ],
    }


# SOURCE_CELL: NB-LIVE-0010-C0010
# SOURCE_STATEMENT_MAP: five-nearest-prefer-containing-bbox-distance15 -> match_features_by_click
def match_features_by_click(
    overrides: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
) -> list[Row]:
    grouped: dict[str, list[Row]] = {}
    for row in _best_scale_rows(features):
        center_x = _float(row.get("cx"))
        center_y = _float(row.get("cy"))
        if not all(
            isfinite(value)
            for value in (
                center_x,
                center_y,
                _float(row.get("bbox_x")),
                _float(row.get("bbox_y")),
                _float(row.get("bbox_w")),
                _float(row.get("bbox_h")),
            )
        ):
            continue
        grouped.setdefault(str(row["image_key"]), []).append(row)
    output: list[Row] = []
    for source in overrides:
        row = dict(source)
        image_key = _stem(row.get("image", ""))
        click_x = _float(row.get("click_x"))
        click_y = _float(row.get("click_y"))
        candidates = grouped.get(image_key, [])
        if not candidates or not isfinite(click_x) or not isfinite(click_y):
            row.update({"_match_found": False, "_match_dist": nan, "_match_idx": -1})
            output.append(row)
            continue
        nearest = sorted(
            enumerate(candidates),
            key=lambda pair: sqrt(
                (_float(pair[1].get("cx")) - click_x) ** 2
                + (_float(pair[1].get("cy")) - click_y) ** 2
            ),
        )[:5]
        choice = nearest[0]
        for candidate in nearest:
            feature = candidate[1]
            x = _float(feature.get("bbox_x"))
            y = _float(feature.get("bbox_y"))
            width = _float(feature.get("bbox_w"))
            height = _float(feature.get("bbox_h"))
            if x <= click_x <= x + width and y <= click_y <= y + height:
                choice = candidate
                break
        index, feature = choice
        distance = sqrt(
            (_float(feature.get("cx")) - click_x) ** 2
            + (_float(feature.get("cy")) - click_y) ** 2
        )
        row.update(feature)
        row.update(
            {
                "_match_found": True,
                "_match_dist": distance,
                "_match_idx": index,
                "match_ok": distance <= 15.0,
            }
        )
        output.append(row)
    return output


# SOURCE_CELL: NB-LIVE-0010-C0012
# SOURCE_STATEMENT_MAP: click-inside-or-distance15 -> refine_click_matches
def refine_click_matches(rows: Sequence[Mapping[str, Any]]) -> list[Row]:
    output: list[Row] = []
    for source in rows:
        row = dict(source)
        click_x = _float(row.get("click_x"))
        click_y = _float(row.get("click_y"))
        x = _float(row.get("bbox_x"))
        y = _float(row.get("bbox_y"))
        width = _float(row.get("bbox_w"))
        height = _float(row.get("bbox_h"))
        inside = (
            all(isfinite(value) for value in (click_x, click_y, x, y, width, height))
            and x <= click_x <= x + width
            and y <= click_y <= y + height
        )
        distance = _float(row.get("_match_dist"))
        match_ok = isfinite(distance) and distance <= 15.0
        row.update(
            {
                "click_inside_bbox": inside,
                "match_ok": match_ok,
                "match_ok2": inside or match_ok,
            }
        )
        output.append(row)
    return output


RULE_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("mean_a", "lower"),
    ("median_a", "lower"),
    ("median_a_in", "lower"),
    ("median_a_ring", "lower"),
    ("median_b", "lower"),
    ("median_b_in", "lower"),
    ("mean_b", "lower"),
    ("histb_q2", "lower"),
    ("histb_q3", "lower"),
    ("histb_q4", "lower"),
    ("std_L", "lower"),
    ("grad_mean", "lower"),
    ("grad_p90", "lower"),
    ("LBP_u5", "lower"),
    ("embed_pca_0", "lower"),
    ("embed_pca_14", "lower"),
    ("embed_pca_11", "higher"),
    ("delta_a", "higher"),
    ("solidity", "higher"),
    ("area_norm", "higher"),
    ("area_px", "higher"),
    ("median_L", "higher"),
    ("mean_L", "higher"),
    ("embed_pca_6", "higher"),
    ("embed_pca_7", "higher"),
    ("embed_pca_8", "higher"),
)


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return nan
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


# SOURCE_CELL: NB-LIVE-0010-C0015
# SOURCE_STATEMENT_MAP: quantile-threshold-precision-lift-coverage-recall -> sweep_zone_features
def sweep_zone_features(
    disagreements: Sequence[Mapping[str, Any]],
    *,
    minimum_count: int = 80,
) -> tuple[RuleSweep, ...]:
    zone = [row for row in disagreements if _bool(row.get("zone"))]
    labels = [int(row.get("align_bucket") == "human->yolo") for row in zone]
    if not zone:
        return ()
    base = sum(labels) / len(labels)
    output: list[RuleSweep] = []
    quantiles = [0.05 + 0.025 * index for index in range(37)]
    for feature, direction in RULE_CANDIDATES:
        pairs = [
            (_float(row.get(feature)), label)
            for row, label in zip(zone, labels)
            if isfinite(_float(row.get(feature)))
        ]
        if len(pairs) < max(minimum_count, 200):
            continue
        values = [value for value, _ in pairs]
        best: RuleSweep | None = None
        for threshold in sorted({_quantile(values, value) for value in quantiles}):
            selected = [
                label
                for value, label in pairs
                if (value <= threshold if direction == "lower" else value >= threshold)
            ]
            if len(selected) < minimum_count:
                continue
            precision = sum(selected) / len(selected)
            recall = sum(selected) / max(1, sum(labels))
            candidate = RuleSweep(
                feature,
                direction,
                threshold,
                precision,
                precision / base if base > 0 else nan,
                len(selected) / len(pairs),
                recall,
                len(selected),
                base,
            )
            if best is None or candidate.lift > best.lift:
                best = candidate
        if best is not None:
            output.append(best)
    return tuple(
        sorted(output, key=lambda item: (item.lift, item.precision, item.coverage), reverse=True)
    )


def _rule_matches(row: Mapping[str, Any], rule: RuleSweep) -> bool:
    value = _float(row.get(rule.feature))
    if not isfinite(value):
        return False
    return value <= rule.threshold if rule.direction == "lower" else value >= rule.threshold


# SOURCE_CELL: NB-LIVE-0010-C0016
# SOURCE_CELL: NB-LIVE-0010-C0017
# SOURCE_STATEMENT_MAP: single-rules-or-and-combination-search -> combine_zone_rules
def combine_zone_rules(
    disagreements: Sequence[Mapping[str, Any]],
    rules: Sequence[RuleSweep],
    *,
    minimum_count: int = 80,
) -> list[Row]:
    zone = [row for row in disagreements if _bool(row.get("zone"))]
    labels = [int(row.get("align_bucket") == "human->yolo") for row in zone]
    candidates = [rule for rule in rules if rule.feature not in LEAKAGE_EXACT][:12]
    output: list[Row] = []
    for count in (1, 2, 3):
        for selected_rules in combinations(candidates, count):
            mask = [any(_rule_matches(row, rule) for rule in selected_rules) for row in zone]
            selected_labels = [label for label, keep in zip(labels, mask) if keep]
            if len(selected_labels) < minimum_count:
                continue
            output.append(
                {
                    "k": count,
                    "feats": "|".join(rule.feature for rule in selected_rules),
                    "n": len(selected_labels),
                    "prec": sum(selected_labels) / len(selected_labels),
                    "cov": len(selected_labels) / len(zone) if zone else nan,
                    "recall": sum(selected_labels) / max(1, sum(labels)),
                }
            )
    return sorted(
        output,
        key=lambda row: (float(row["prec"]), float(row["recall"]), float(row["cov"])),
        reverse=True,
    )


def binary_metrics(predictions: Sequence[int], labels: Sequence[int]) -> BinaryMetrics:
    true_positive = sum(p == 1 and y == 1 for p, y in zip(predictions, labels))
    false_positive = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    true_negative = sum(p == 0 and y == 0 for p, y in zip(predictions, labels))
    false_negative = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (true_positive + true_negative) / max(
        1, true_positive + true_negative + false_positive + false_negative
    )
    return BinaryMetrics(
        true_positive,
        false_positive,
        true_negative,
        false_negative,
        precision,
        recall,
        f1,
        accuracy,
    )


CONTEXT_RULES = {
    "ctx_v1_median_a_ring<=122": lambda row: _float(row.get("median_a_ring")) <= 122.0,
    "ctx_v2_or_ring_or_pca_or_sol": lambda row: (
        _float(row.get("median_a_ring")) <= 122.0
        or _float(row.get("embed_pca_0")) <= -0.353897
        or _float(row.get("solidity")) >= 0.97868
    ),
    "ctx_v3_chroma_texture_pack": lambda row: (
        _float(row.get("median_a_ring")) <= 122.0
        or _float(row.get("histb_q2")) <= 142.0
        or _float(row.get("grad_mean")) <= 16.152241
    ),
}


# SOURCE_CELL: NB-LIVE-0010-C0018
# SOURCE_CELL: NB-LIVE-0010-C0019
# SOURCE_STATEMENT_MAP: zone-context-override-confusion-delta-slices -> evaluate_context_rules
def evaluate_context_rules(rows: Sequence[Mapping[str, Any]]) -> list[Row]:
    labeled = [
        row
        for row in rows
        if _bool(row.get("human_ok")) and normalize_binary_label(row.get("human_last")) is not None
    ]
    labels = [int(normalize_binary_label(row.get("human_last")) or 0) for row in labeled]
    baseline = [int(normalize_binary_label(row.get("final_pred_eff")) or 0) for row in labeled]
    yolo = [int(normalize_binary_label(row.get("yolo_pred_eff")) or 0) for row in labeled]
    zone = [
        _float(row.get("yolo_conf_f")) >= 0.65
        and _float(row.get("yolo_iou_f")) >= 0.70
        and _float(row.get("xgb_prob_f")) <= 0.35
        and abs(_float(row.get("yolo_conf_f")) - _float(row.get("xgb_prob_f"))) >= 0.25
        for row in labeled
    ]
    slices: dict[str, list[bool]] = {"ALL": [True] * len(labeled), "ZONE": zone}
    for column in ("match_ok", "match_ok2", "inside_bbox"):
        if any(column in row for row in labeled):
            slices[column.upper()] = [_bool(row.get(column)) for row in labeled]
    output: list[Row] = []
    for name, predicate in CONTEXT_RULES.items():
        predictions = list(baseline)
        for index, row in enumerate(labeled):
            if zone[index] and predicate(row):
                predictions[index] = yolo[index]
        for slice_name, mask in slices.items():
            base_values = [value for value, keep in zip(baseline, mask) if keep]
            new_values = [value for value, keep in zip(predictions, mask) if keep]
            slice_labels = [value for value, keep in zip(labels, mask) if keep]
            old = binary_metrics(base_values, slice_labels)
            new = binary_metrics(new_values, slice_labels)
            output.append(
                {
                    "rule": name,
                    "slice": slice_name,
                    "flip": sum(a != b for a, b in zip(base_values, new_values)),
                    "prec_base": old.precision,
                    "rec_base": old.recall,
                    "f1_base": old.f1,
                    "prec_new": new.precision,
                    "rec_new": new.recall,
                    "f1_new": new.f1,
                    "tp_gain": new.true_positive - old.true_positive,
                    "fp_gain": new.false_positive - old.false_positive,
                    "fn_gain": new.false_negative - old.false_negative,
                }
            )
    return sorted(output, key=lambda row: (str(row["slice"]), -float(row["f1_new"])))


META_FEATURES = (
    "diff_yolo_xgb",
    "yolo_conf_f",
    "yolo_iou_f",
    "xgb_prob_f",
    "median_a_ring",
    "mean_a",
    "median_a",
    "median_a_in",
    "histb_q2",
    "histb_q3",
    "histb_q4",
    "grad_mean",
    "grad_p90",
    "embed_pca_0",
    "embed_pca_11",
    "embed_pca_6",
    "solidity",
    "area_norm",
    "std_L",
    "g_quality",
    "g_light",
)


def _sigmoid(value: float) -> float:
    clipped = max(-50.0, min(50.0, value))
    return 1.0 / (1.0 + exp(-clipped))


def _group_split(groups: Sequence[str]) -> tuple[list[int], list[int]]:
    unique = sorted(set(groups))
    Random(42).shuffle(unique)
    validation_count = max(1, round(len(unique) * 0.25))
    validation_groups = set(unique[:validation_count])
    validation = [index for index, group in enumerate(groups) if group in validation_groups]
    training = [index for index, group in enumerate(groups) if group not in validation_groups]
    if not training or not validation:
        split = max(1, len(groups) * 3 // 4)
        training = list(range(split))
        validation = list(range(split, len(groups)))
    return training, validation


def _fit_logistic(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[int],
    training: Sequence[int],
    regularization_c: float,
) -> tuple[float, list[float], list[float], list[float], list[float]]:
    columns = len(matrix[0])
    medians: list[float] = []
    means: list[float] = []
    scales: list[float] = []
    for column in range(columns):
        values = [matrix[index][column] for index in training if isfinite(matrix[index][column])]
        medians.append(float(median(values)) if values else 0.0)
    imputed = [
        [value if isfinite(value) else medians[column] for column, value in enumerate(row)]
        for row in matrix
    ]
    for column in range(columns):
        values = [imputed[index][column] for index in training]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means.append(mean)
        scales.append(sqrt(variance) if variance > 0 else 1.0)
    standardized = [
        [(value - means[column]) / scales[column] for column, value in enumerate(row)]
        for row in imputed
    ]
    weights = [0.0] * columns
    intercept = 0.0
    positives = max(1, sum(labels[index] == 1 for index in training))
    negatives = max(1, sum(labels[index] == 0 for index in training))
    class_weight = {0: len(training) / (2 * negatives), 1: len(training) / (2 * positives)}
    learning_rate = 0.08
    l1 = 1.0 / max(regularization_c, 1e-12)
    for step in range(700):
        grad_intercept = 0.0
        gradient = [0.0] * columns
        for index in training:
            probability = _sigmoid(
                intercept
                + sum(weight * value for weight, value in zip(weights, standardized[index]))
            )
            error = (probability - labels[index]) * class_weight[labels[index]]
            grad_intercept += error
            for column, value in enumerate(standardized[index]):
                gradient[column] += error * value
        rate = learning_rate / sqrt(1.0 + step / 40.0)
        intercept -= rate * grad_intercept / len(training)
        threshold = rate * l1 / len(training)
        for column in range(columns):
            raw = weights[column] - rate * gradient[column] / len(training)
            weights[column] = max(0.0, raw - threshold) - max(0.0, -raw - threshold)
    probabilities = [
        _sigmoid(intercept + sum(weight * value for weight, value in zip(weights, row)))
        for row in standardized
    ]
    return intercept, weights, medians, means, scales, probabilities


# SOURCE_CELL: NB-LIVE-0010-C0020
# SOURCE_STATEMENT_MAP: grouped-split-impute-scale-l1-grid-auc-threshold-export -> fit_sparse_meta_model
def fit_sparse_meta_model(rows: Sequence[Mapping[str, Any]]) -> SparseLogisticModel:
    decisive: list[Row] = []
    labels: list[int] = []
    for source in rows:
        human = normalize_binary_label(source.get("human_last"))
        yolo = normalize_binary_label(source.get("yolo_pred_eff"))
        xgb = normalize_binary_label(source.get("xgb_pred_eff"))
        if human is None or yolo is None or xgb is None or yolo == xgb:
            continue
        agrees_yolo = human == yolo
        agrees_xgb = human == xgb
        if agrees_yolo == agrees_xgb:
            continue
        row = dict(source)
        row["diff_yolo_xgb"] = abs(
            _float(row.get("yolo_conf_f")) - _float(row.get("xgb_prob_f"))
        )
        decisive.append(row)
        labels.append(int(agrees_yolo))
    if len(decisive) < 8 or len(set(labels)) < 2:
        raise MetaAnalysisContractError("sparse meta model requires decisive rows from both classes")
    features = tuple(
        feature
        for feature in META_FEATURES
        if sum(isfinite(_float(row.get(feature))) for row in decisive) / len(decisive) >= 0.95
    )
    if len(features) < 4:
        raise MetaAnalysisContractError(
            f"too few usable meta features with >=0.95 coverage: {list(features)}"
        )
    matrix = [[_float(row.get(feature)) for feature in features] for row in decisive]
    groups = [str(row.get("image", row.get("image_key", ""))) for row in decisive]
    training, validation = _group_split(groups)
    best: tuple[float, float, float, list[float], list[float], list[float], list[float]] | None = None
    for regularization_c in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
        intercept, weights, medians, means, scales, probabilities = _fit_logistic(
            matrix, labels, training, regularization_c
        )
        auc = _auc([labels[index] for index in validation], [probabilities[index] for index in validation])
        if best is None or auc > best[0]:
            best = (auc, regularization_c, intercept, weights, medians, means, scales)
    assert best is not None
    auc, regularization_c, intercept, weights, medians, means, scales = best
    normalized = [
        [
            ((value if isfinite(value) else medians[column]) - means[column]) / scales[column]
            for column, value in enumerate(row)
        ]
        for row in matrix
    ]
    probabilities = [
        _sigmoid(intercept + sum(weight * value for weight, value in zip(weights, row)))
        for row in normalized
    ]
    best_threshold = 0.5
    best_f1 = -1.0
    for step in range(37):
        threshold = 0.05 + step * 0.025
        final_predictions: list[int] = []
        final_labels: list[int] = []
        for index in validation:
            chosen = (
                normalize_binary_label(decisive[index].get("yolo_pred_eff"))
                if probabilities[index] >= threshold
                else normalize_binary_label(decisive[index].get("xgb_pred_eff"))
            )
            final_predictions.append(int(chosen or 0))
            final_labels.append(
                int(normalize_binary_label(decisive[index].get("human_last")) or 0)
            )
        score = binary_metrics(final_predictions, final_labels).f1
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return SparseLogisticModel(
        features,
        tuple(medians),
        tuple(means),
        tuple(scales),
        intercept,
        tuple(weights),
        regularization_c,
        auc,
        best_threshold,
    )


def _model_dict(model: SparseLogisticModel) -> Mapping[str, Any]:
    return {
        "features": model.features,
        "imputer_median": model.imputer_median,
        "scaler_mean": model.scaler_mean,
        "scaler_scale": model.scaler_scale,
        "intercept": model.intercept,
        "coef": model.coefficients,
        "C": model.regularization_c,
        "val_auc": model.validation_auc,
        "threshold_choose_yolo": model.threshold_choose_yolo,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_meta_outputs(
    output: Path,
    services: MetaAnalysisServices,
    raw: Sequence[Mapping[str, Any]],
    latest: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
    event_evaluation: MetaEventEvaluation,
    merged: Sequence[Mapping[str, Any]],
    matched: Sequence[Mapping[str, Any]],
    ranks: Sequence[FeatureRank],
    sweeps: Sequence[RuleSweep],
    combinations_: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
    model: SparseLogisticModel | None,
) -> tuple[Path, ...]:
    create_output_directory(output)
    output_identity = _require_meta_output_root(output)
    artifacts: list[Path] = []
    source_table_values = (
        ("review_labels_all_raw.csv", raw),
        ("review_labels_all.csv", latest),
        ("review_labels_manifest.csv", manifest),
    )
    for name, rows in source_table_values:
        path = output / name
        services.write_rows(path, rows)
        _require_meta_output_root(output, output_identity)
        artifacts.append(path)
    artifacts.extend(
        write_meta_event_outputs(
            output,
            services,
            event_evaluation,
            output_identity=output_identity,
        )
    )
    _require_meta_output_root(output, output_identity)
    analysis_table_values = (
        ("overrides_plus_features__key_image_annid.csv", merged),
        ("overrides_plus_features__match_click.csv", matched),
        ("feature_auc_ranking.csv", [asdict(item) for item in ranks]),
        ("rule_threshold_sweep__zone.csv", [asdict(item) for item in sweeps]),
        ("best_or_rules__zone_top.csv", combinations_),
        ("context_rule_metrics.csv", contexts),
    )
    for name, rows in analysis_table_values:
        path = output / name
        services.write_rows(path, rows)
        _require_meta_output_root(output, output_identity)
        artifacts.append(path)
    if model is not None:
        model_path = output / "meta" / "meta_logreg_coef.json"
        _write_json(model_path, _model_dict(model))
        _require_meta_output_root(output, output_identity)
        artifacts.append(model_path)
        coefficient_rows = sorted(
            (
                {"feature": feature, "coef": coefficient}
                for feature, coefficient in zip(model.features, model.coefficients)
                if abs(coefficient) > 1e-9
            ),
            key=lambda row: float(row["coef"]),
        )
        coefficient_path = output / "meta" / "meta_top_coefs.csv"
        services.write_rows(coefficient_path, coefficient_rows)
        _require_meta_output_root(output, output_identity)
        artifacts.append(coefficient_path)
    _require_meta_output_root(output, output_identity)
    output_root = output.resolve(strict=True)
    if len(artifacts) != len(set(artifacts)):
        raise MetaAnalysisContractError("meta-analysis output artifact names collided")
    for path in artifacts:
        if path.is_symlink() or not path.is_file():
            raise MetaAnalysisContractError(
                "meta-analysis output artifact must be a regular non-symlink file"
            )
        try:
            path.resolve(strict=True).relative_to(output_root)
        except (OSError, ValueError) as exc:
            raise MetaAnalysisContractError(
                "meta-analysis output artifact escaped the declared output"
            ) from exc
    return tuple(artifacts)


# SOURCE_CELL: NB-LIVE-0010-C0000
# SOURCE_CELL: NB-LIVE-0010-C0001
# SOURCE_CELL: NB-LIVE-0010-C0002
# SOURCE_CELL: NB-LIVE-0010-C0003
# SOURCE_CELL: NB-LIVE-0010-C0004
# SOURCE_CELL: NB-LIVE-0010-C0005
# SOURCE_CELL: NB-LIVE-0010-C0006
# SOURCE_CELL: NB-LIVE-0010-C0007
# SOURCE_CELL: NB-LIVE-0010-C0008
# SOURCE_CELL: NB-LIVE-0010-C0009
# SOURCE_CELL: NB-LIVE-0010-C0010
# SOURCE_CELL: NB-LIVE-0010-C0011
# SOURCE_CELL: NB-LIVE-0010-C0012
# SOURCE_CELL: NB-LIVE-0010-C0013
# SOURCE_CELL: NB-LIVE-0010-C0014
# SOURCE_CELL: NB-LIVE-0010-C0015
# SOURCE_CELL: NB-LIVE-0010-C0016
# SOURCE_CELL: NB-LIVE-0010-C0017
# SOURCE_CELL: NB-LIVE-0010-C0018
# SOURCE_CELL: NB-LIVE-0010-C0019
# SOURCE_CELL: NB-LIVE-0010-C0020
# SOURCE_STATEMENT_MAP: explicit-sequential-domain-pipeline -> run_meta_analysis
def execute_meta_analysis(
    config: DomainConfig,
    services: MetaAnalysisServices | None = None,
) -> MetaAnalysisResult:
    """Registry handler for the complete source-backed review meta-analysis."""

    values = config.workflow("evaluate")
    review_root = artifact_path(values, "review_root")
    output = artifact_path(values, "output")
    service = services or StdlibMetaAnalysisServices()
    raw: list[Row] = []
    latest: list[Row] = []
    manifest: list[Row] = []
    events: list[Row] = []
    flips: list[Row] = []
    overrides: list[Row] = []
    event_latest: list[Row] = []
    event_batch: MetaEventBatch | None = None
    event_evaluation: MetaEventEvaluation | None = None
    disagreements: list[Row] = []
    feature_batch: MetaFeatureBatch | None = None
    features: list[Row] = []
    merged: list[Row] = []
    click_matched: list[Row] = []
    matched: list[Row] = []
    decisive_matched: list[Row] = []
    ranks: tuple[FeatureRank, ...] = ()
    sweeps: tuple[RuleSweep, ...] = ()
    combined: list[Row] = []
    contexts: list[Row] = []
    model: SparseLogisticModel | None = None
    artifacts: tuple[Path, ...] = ()

    for operation in build_meta_analysis_program():
        parameters = dict(operation.parameters)
        if operation.operation == "collect-review-logs":
            raw, latest, manifest = collect_review_logs(review_root, service)
        elif operation.operation == "audit-candidate-keys":
            candidate_key_statistics(raw)
        elif operation.operation == "build-review-event-history":
            events, flips, _ = build_review_event_history(
                raw,
                include_source_folder=bool(parameters["include_source_folder"]),
            )
        elif operation.operation == "validate-sealed-meta-events":
            if parameters["implicit_dtype_inference"] is not False:
                raise MetaAnalysisContractError(
                    "sealed meta events prohibit implicit dtype inference"
                )
            event_batch = validate_meta_event_batch(review_root)
        elif operation.operation == "build-effective-overrides":
            if event_batch is None:
                raise MetaAnalysisContractError("meta event batch was not validated")
            event_evaluation = evaluate_meta_event_batch(event_batch)
            event_tables = _meta_event_tables(
                event_evaluation.c0004,
                event_evaluation.batch.columns,
            )
            events = list(event_tables[0])
            flips = list(event_tables[1])
            event_latest = list(event_tables[2])
            overrides = list(event_tables[3])
        elif operation.operation == "select-decisive-disagreements":
            disagreements = decisive_disagreements(overrides)
        elif operation.operation == "validate-sealed-review-features":
            if parameters["implicit_feature_discovery"] is not False:
                raise MetaAnalysisContractError(
                    "sealed review features prohibit implicit discovery"
                )
            feature_batch = validate_meta_feature_batch(review_root, service)
            features = [dict(row) for row in feature_batch.rows]
        elif operation.operation == "merge-features-by-identity":
            if feature_batch is None:
                raise MetaAnalysisContractError("feature batch was not validated")
            merged = merge_features_by_identity(overrides, features)
        elif operation.operation == "match-features-by-click":
            if feature_batch is None:
                raise MetaAnalysisContractError("feature batch was not validated")
            click_matched = match_features_by_click(overrides, features)
        elif operation.operation == "refine-click-matches":
            matched = refine_click_matches(click_matched)
            decisive_matched = decisive_disagreements(matched)
        elif operation.operation == "rank-disagreement-features":
            ranks = rank_disagreement_features(decisive_matched)
        elif operation.operation == "audit-identity-ranges":
            audit_identity_ranges(overrides, features)
        elif operation.operation == "sweep-zone-features":
            sweeps = sweep_zone_features(
                decisive_matched,
                minimum_count=int(parameters["minimum_count"]),
            )
        elif operation.operation == "combine-zone-rules":
            combined = combine_zone_rules(
                decisive_matched,
                sweeps,
                minimum_count=int(parameters["minimum_count"]),
            )
        elif operation.operation == "evaluate-context-rules":
            contexts = evaluate_context_rules(matched)
        elif operation.operation == "fit-sparse-meta-model":
            try:
                model = fit_sparse_meta_model(matched)
            except MetaAnalysisContractError:
                model = None
        elif operation.operation == "write-declared-meta-artifacts":
            if event_evaluation is None:
                raise MetaAnalysisContractError("meta event evaluation was not completed")
            artifacts = _write_meta_outputs(
                output,
                service,
                raw,
                latest,
                manifest,
                event_evaluation,
                merged,
                matched,
                ranks,
                sweeps,
                combined,
                contexts,
                model,
            )
        else:
            raise MetaAnalysisContractError(
                f"unsupported meta-analysis program operation: {operation.operation}"
            )
        _explicit_noop()
    if event_evaluation is None:
        raise MetaAnalysisContractError("meta event evaluation was not completed")
    return MetaAnalysisResult(
        len(manifest),
        len(raw),
        len(event_latest),
        len(flips),
        len(disagreements),
        output,
        artifacts,
        event_evaluation.batch.manifest,
        event_evaluation.batch.manifest_sha256,
        event_evaluation.batch.events,
        event_evaluation.batch.events_sha256,
    )
