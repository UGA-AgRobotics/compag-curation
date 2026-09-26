"""Typed review-event statistics for meta-analysis cells C0003 and C0004.

The source cells expressed this workflow through pandas columns.  This module
keeps the event ordering, forward-filled label history, effective prediction
rules, latest-row selection, and summary statistics in immutable stdlib-only
records.  File discovery and table serialization remain owned by the registered
meta-analysis handler.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from math import floor, isfinite, isnan
from re import fullmatch
from struct import pack, unpack
from typing import Literal, TypeAlias


MetaScalar: TypeAlias = str | int | float | bool | None
TimestampScalar: TypeAlias = str | int | float | None
MetaVariant: TypeAlias = Literal["c0003", "c0004"]
TimestampStorage: TypeAlias = Literal["numeric", "text"]
TimestampUnit: TypeAlias = Literal["s", "ms", "us", "ns"]

_MIN_DATETIME64_NS = -9_223_372_036_854_775_807
_MAX_DATETIME64_NS = 9_223_372_036_854_775_807


class MetaEventStatisticsContractError(ValueError):
    """Raised when a typed review-event contract cannot be evaluated."""


@dataclass(frozen=True)
class MetaEventInputRow:
    """One source row with explicit column-presence and ordering state."""

    image: str
    identifier: str
    timestamp: TimestampScalar
    source_mtime: TimestampScalar
    row_in_file: int
    human_label: MetaScalar
    xgb_prob: MetaScalar = None
    yolo_conf: MetaScalar = None
    yolo_iou: MetaScalar = None
    thr_xgb: MetaScalar = None
    thr_yolo: MetaScalar = None
    thr_iou: MetaScalar = None
    detector_threshold: MetaScalar = None
    xgb_pred: MetaScalar = None
    xgb_pred_present: bool = True
    final_pred: MetaScalar = None
    final_pred_present: bool = True
    kept: MetaScalar = None
    detector_policy: str | None = None

    def __post_init__(self) -> None:
        if type(self.image) is not str or type(self.identifier) is not str:
            raise MetaEventStatisticsContractError(
                "meta event image and identifier must be strings"
            )
        if type(self.row_in_file) is not int or self.row_in_file < 0:
            raise MetaEventStatisticsContractError(
                "meta event row ordinal must be a nonnegative integer"
            )
        if type(self.xgb_pred_present) is not bool or type(self.final_pred_present) is not bool:
            raise MetaEventStatisticsContractError(
                "meta event column-presence flags must be exact booleans"
            )
        if self.detector_policy is not None and type(self.detector_policy) is not str:
            raise MetaEventStatisticsContractError(
                "meta event detector policy must be a string or null"
            )
        for label, value in (
            ("human label", self.human_label),
            ("XGB probability", self.xgb_prob),
            ("YOLO confidence", self.yolo_conf),
            ("YOLO IoU", self.yolo_iou),
            ("XGB threshold", self.thr_xgb),
            ("YOLO threshold", self.thr_yolo),
            ("IoU threshold", self.thr_iou),
            ("detector threshold", self.detector_threshold),
            ("XGB prediction", self.xgb_pred),
            ("final prediction", self.final_pred),
            ("kept prediction", self.kept),
        ):
            if value is not None and type(value) not in {str, int, float, bool}:
                raise MetaEventStatisticsContractError(
                    f"meta event {label} has an invalid scalar type"
                )
        for label, value in (
            ("timestamp", self.timestamp),
            ("source mtime", self.source_mtime),
        ):
            if value is not None and (
                type(value) not in {str, int, float} or type(value) is bool
            ):
                raise MetaEventStatisticsContractError(
                    f"meta event {label} has an invalid scalar type"
                )


@dataclass(frozen=True)
class MetaEventSettings:
    """Sealed source defaults shared by the two event-statistics cells."""

    iou_fallback: float = 0.70
    xgb_sparse_fraction: float = 0.05
    event_quantiles: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95, 0.99)

    def __post_init__(self) -> None:
        if type(self.iou_fallback) is not float or self.iou_fallback != 0.70:
            raise MetaEventStatisticsContractError(
                "meta event IoU fallback must remain 0.70"
            )
        if (
            type(self.xgb_sparse_fraction) is not float
            or self.xgb_sparse_fraction != 0.05
        ):
            raise MetaEventStatisticsContractError(
                "meta event sparse-XGB fraction must remain 0.05"
            )
        if (
            type(self.event_quantiles) is not tuple
            or any(type(value) is not float for value in self.event_quantiles)
            or self.event_quantiles != (0.50, 0.75, 0.90, 0.95, 0.99)
        ):
            raise MetaEventStatisticsContractError(
                "meta event quantiles must remain sealed"
            )


@dataclass(frozen=True)
class MetaEventColumnDtypes:
    """Explicit source dataframe dtypes and the sealed text-time grammar."""

    timestamp: TimestampStorage
    source_mtime: TimestampStorage
    text_timestamp_grammar: Literal["extended-iso8601-nanoseconds"] = (
        "extended-iso8601-nanoseconds"
    )

    def __post_init__(self) -> None:
        if type(self.timestamp) is not str or self.timestamp not in {"numeric", "text"}:
            raise MetaEventStatisticsContractError(
                "meta event timestamp dtype must be numeric or text"
            )
        if (
            type(self.source_mtime) is not str
            or self.source_mtime not in {"numeric", "text"}
        ):
            raise MetaEventStatisticsContractError(
                "meta event source-mtime dtype must be numeric or text"
            )
        if (
            type(self.text_timestamp_grammar) is not str
            or self.text_timestamp_grammar != "extended-iso8601-nanoseconds"
        ):
            raise MetaEventStatisticsContractError(
                "meta event text timestamp grammar must remain the sealed "
                "extended ISO-8601 nanosecond grammar"
            )


@dataclass(frozen=True)
class MetaEventRecord:
    """One normalized event in stable item/time/source/row order."""

    source: MetaEventInputRow
    source_ordinal: int
    item_key: str
    event_time: datetime | None
    event_time_nanoseconds: int | None
    source_mtime_time: datetime | None
    source_mtime_nanoseconds: int | None
    human_label_n: float | None
    human_last: float | None
    human_prev: float | None
    label_changed: bool
    xgb_prob_f: float | None
    yolo_conf_f: float | None
    yolo_iou_f: float | None
    effective_xgb_threshold: float | None
    effective_yolo_threshold: float | None
    effective_iou_threshold: float
    xgb_pred_eff: float | None
    kept_eff: float | None
    yolo_pred_eff: float
    final_pred_eff: float | None


@dataclass(frozen=True)
class MetaLatestOverride:
    """Latest event for an item plus human/model agreement disposition."""

    event: MetaEventRecord
    detector_policy: str
    human_ok: bool
    override_final: bool
    agree_yolo: bool
    agree_xgb: bool
    align_bucket: Literal[
        "human->yolo",
        "human->xgb",
        "human->both",
        "human->neither",
        "no_human",
    ]


@dataclass(frozen=True)
class MetaEventStatisticsResult:
    """Exact typed row sets and aggregate statistics for one source cell."""

    source_variant: MetaVariant
    events: tuple[MetaEventRecord, ...]
    flips: tuple[MetaEventRecord, ...]
    latest: tuple[MetaEventRecord, ...]
    overrides: tuple[MetaLatestOverride, ...]
    row_count: int
    item_count: int
    events_per_item_quantiles: tuple[tuple[float, float], ...]
    flip_row_count: int
    items_with_flip_count: int
    human_ok_count: int
    override_final_rate: float | None
    align_bucket_counts: tuple[tuple[str, int], ...]
    xgb_sparse_fallback_applied: bool


@dataclass(frozen=True)
class MetaParsedTimestamp:
    value: datetime
    epoch_nanoseconds: int


def _numeric_timestamp_unit(value: float | Decimal) -> Literal["s", "ms", "us"]:
    if value > 1e14:
        return "us"
    if value > 1e11:
        return "ms"
    return "s"


def _numeric_decimal(value: TimestampScalar) -> Decimal | None:
    if value is None or type(value) is bool:
        return None
    if type(value) is int:
        return Decimal(value)
    if type(value) is float:
        if isnan(value):
            return None
        return Decimal.from_float(value)
    if type(value) is not str:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        numeric = Decimal(text)
    except InvalidOperation:
        return None
    return None if numeric.is_nan() else numeric


def _datetime_from_nanoseconds(epoch_nanoseconds: int) -> datetime | None:
    seconds, remainder = divmod(epoch_nanoseconds, 1_000_000_000)
    try:
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
            seconds=seconds,
            microseconds=remainder // 1_000,
        )
    except (OverflowError, ValueError):
        return None


def _numeric_timestamp_value(
    value: TimestampScalar,
    *,
    unit: TimestampUnit,
) -> MetaParsedTimestamp | None:
    numeric = _numeric_decimal(value)
    if numeric is None or not numeric.is_finite():
        return None
    multiplier = {
        "s": 1_000_000_000,
        "ms": 1_000_000,
        "us": 1_000,
        "ns": 1,
    }[unit]
    epoch_nanoseconds = int(numeric * multiplier)
    if not _MIN_DATETIME64_NS <= epoch_nanoseconds <= _MAX_DATETIME64_NS:
        return None
    parsed = _datetime_from_nanoseconds(epoch_nanoseconds)
    if parsed is None:
        return None
    return MetaParsedTimestamp(parsed, epoch_nanoseconds)


def _text_timestamp_value(value: TimestampScalar) -> MetaParsedTimestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "nat", "none"}:
        return None
    grammar_match = fullmatch(
        r"\d{4}-\d{2}-\d{2}"
        r"(?:(?:T| )\d{2}:\d{2}:\d{2}"
        r"(?:[.,](\d+))?"
        r"(?:Z|z|[+-]\d{2}(?::?\d{2})?)?"
        r")?",
        text,
    )
    if grammar_match is None:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(normalized.replace(",", "."))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    delta = parsed - datetime(1970, 1, 1, tzinfo=timezone.utc)
    epoch_nanoseconds = (
        (delta.days * 86_400 + delta.seconds) * 1_000_000_000
        + delta.microseconds * 1_000
    )
    fraction_digits = grammar_match.group(1)
    if fraction_digits is not None:
        fraction = fraction_digits[:9].ljust(9, "0")
        epoch_nanoseconds += int(fraction[6:9])
    if not _MIN_DATETIME64_NS <= epoch_nanoseconds <= _MAX_DATETIME64_NS:
        return None
    return MetaParsedTimestamp(parsed, epoch_nanoseconds)


# SOURCE_CELL: NB-LIVE-0010-C0003
# SOURCE_CELL: NB-LIVE-0010-C0004
# SOURCE_STATEMENT_MAP: numeric-s-ms-us-and-text-utc-time -> parse_event_timestamp
def parse_event_timestamp(
    value: TimestampScalar,
    *,
    storage: TimestampStorage,
    numeric_unit: TimestampUnit | None = None,
) -> MetaParsedTimestamp | None:
    """Parse one value under an explicit dataframe dtype and UTC unit contract."""

    if type(storage) is not str or storage not in {"numeric", "text"}:
        raise MetaEventStatisticsContractError(
            "meta event timestamp storage is invalid"
        )
    if storage == "numeric" and numeric_unit not in {"s", "ms", "us", "ns"}:
        raise MetaEventStatisticsContractError(
            "numeric meta event timestamps require an explicit unit"
        )
    if storage == "text" and numeric_unit is not None:
        raise MetaEventStatisticsContractError(
            "text meta event timestamps cannot carry a numeric unit"
        )
    if value is None:
        return None
    if storage == "numeric":
        assert numeric_unit is not None
        if type(value) not in {int, float} or (
            type(value) is float and isnan(value)
        ):
            raise MetaEventStatisticsContractError(
                "numeric meta event timestamp values must be typed numbers or null"
            )
        return _numeric_timestamp_value(value, unit=numeric_unit)
    if type(value) is not str:
        raise MetaEventStatisticsContractError(
            "text meta event timestamp values must be strings or null"
        )
    parsed = _text_timestamp_value(value)
    if parsed is None:
        raise MetaEventStatisticsContractError(
            "text meta event timestamp values must use the sealed extended "
            "ISO-8601 nanosecond grammar or null"
        )
    return parsed


# SOURCE_CELL: NB-LIVE-0010-C0003
# SOURCE_CELL: NB-LIVE-0010-C0004
# SOURCE_STATEMENT_MAP: robust-bool-int-and-token-label-normalization -> normalize_binary_label
def normalize_binary_label(
    value: MetaScalar,
    *,
    variant: MetaVariant,
) -> int | None:
    """Apply the exact C0003 or C0004 label parser to one scalar."""

    if variant not in {"c0003", "c0004"}:
        raise MetaEventStatisticsContractError("unknown meta event parser variant")
    if value is None:
        return None
    if type(value) is bool:
        return 1 if value else 0
    if variant == "c0003" and type(value) is int:
        return value
    if variant == "c0004" and type(value) in {int, float}:
        numeric = float(value)
        return int(numeric) if isfinite(numeric) and numeric in {0.0, 1.0} else None
    if type(value) is float and not isfinite(value):
        return None
    text = str(value).strip().lower()
    if variant == "c0004":
        try:
            numeric_text = float(text)
        except ValueError:
            numeric_text = None
        if (
            numeric_text is not None
            and isfinite(numeric_text)
            and numeric_text in {0.0, 1.0}
        ):
            return int(numeric_text)
    positive = {
        "1",
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
    negative = {
        "0",
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
    if variant == "c0004":
        positive.add("1.0")
        negative.add("0.0")
    if text in positive:
        return 1
    if text in negative:
        return 0
    return None


def _to_float(value: MetaScalar) -> float | None:
    if value is None:
        return None
    if type(value) is bool:
        return float(int(value))
    if type(value) in {int, float}:
        result = float(value)
    elif type(value) is str:
        try:
            result = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return None if isnan(result) else result


def _as_float32(value: int | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
        return unpack(">f", pack(">f", numeric))[0]
    except OverflowError:
        return float("-inf") if value < 0 else float("inf")


def _effective_threshold(primary: MetaScalar, fallback: MetaScalar) -> float | None:
    value = _to_float(primary)
    return value if value is not None else _to_float(fallback)


def _meets(value: float | None, threshold: float | None) -> float:
    return float(value is not None and threshold is not None and value >= threshold)


def _time_key(value: int | None) -> tuple[int, int]:
    if value is None:
        return 1, 0
    return 0, value


def _latest_time_key(value: int | None) -> tuple[int, int]:
    if value is None:
        return 0, 0
    return 1, value


def _is_missing_scalar(value: TimestampScalar) -> bool:
    return value is None


def _validate_column_dtypes(
    rows: tuple[MetaEventInputRow, ...],
    column_dtypes: MetaEventColumnDtypes,
) -> None:
    for field_name, storage in (
        ("timestamp", column_dtypes.timestamp),
        ("source_mtime", column_dtypes.source_mtime),
    ):
        for row in rows:
            value = getattr(row, field_name)
            if value is None:
                continue
            if storage == "numeric":
                if type(value) not in {int, float} or (
                    type(value) is float and isnan(value)
                ):
                    raise MetaEventStatisticsContractError(
                        f"numeric meta event {field_name} values must be typed numbers or null"
                    )
                continue
            if type(value) is not str or _text_timestamp_value(value) is None:
                raise MetaEventStatisticsContractError(
                    f"text meta event {field_name} values must use the sealed "
                    "extended ISO-8601 nanosecond grammar or null"
                )


def _raw_source_time_key(
    source: MetaEventInputRow,
    *,
    numeric_column: bool,
) -> tuple[int, Decimal, str]:
    value = source.source_mtime
    if _is_missing_scalar(value):
        return 1, Decimal(0), ""
    if numeric_column:
        assert value is not None
        numeric = _numeric_decimal(value)
        assert numeric is not None
        return 0, numeric, ""
    return 0, Decimal(0), str(value)


def _latest_raw_source_time_key(
    source: MetaEventInputRow,
    *,
    numeric_column: bool,
) -> tuple[int, Decimal, str]:
    value = source.source_mtime
    if _is_missing_scalar(value):
        return 0, Decimal(0), ""
    if numeric_column:
        assert value is not None
        numeric = _numeric_decimal(value)
        assert numeric is not None
        return 1, numeric, ""
    return 1, Decimal(0), str(value)


def _quantile(values: tuple[int, ...], fraction: float) -> float:
    if not values:
        raise MetaEventStatisticsContractError(
            "meta event quantiles require at least one item"
        )
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _event_numeric_unit(rows: tuple[MetaEventInputRow, ...]) -> Literal["s", "ms", "us"] | None:
    values = tuple(row.timestamp for row in rows)
    numeric_values = tuple(
        numeric
        for value in values
        if not _is_missing_scalar(value)
        and (numeric := _numeric_decimal(value)) is not None
    )
    return _numeric_timestamp_unit(max(numeric_values)) if numeric_values else "s"


def _initial_record(
    row: MetaEventInputRow,
    source_ordinal: int,
    *,
    variant: MetaVariant,
    event_numeric_unit: Literal["s", "ms", "us"] | None,
    column_dtypes: MetaEventColumnDtypes,
    sparse_xgb_fallback: bool,
) -> MetaEventRecord:
    if column_dtypes.timestamp == "numeric":
        timestamp_value = parse_event_timestamp(
            row.timestamp,
            storage="numeric",
            numeric_unit=event_numeric_unit if variant == "c0003" else "ns",
        )
    else:
        timestamp_value = parse_event_timestamp(row.timestamp, storage="text")
    source_mtime_value = parse_event_timestamp(
        row.source_mtime,
        storage=column_dtypes.source_mtime,
        numeric_unit="ns" if column_dtypes.source_mtime == "numeric" else None,
    )
    event_time_value = timestamp_value or source_mtime_value
    xgb_probability = _to_float(row.xgb_prob)
    yolo_confidence = _to_float(row.yolo_conf)
    yolo_iou = _to_float(row.yolo_iou)
    xgb_threshold = _effective_threshold(row.thr_xgb, row.detector_threshold)
    yolo_threshold = _effective_threshold(row.thr_yolo, row.detector_threshold)
    iou_threshold = _to_float(row.thr_iou)
    if iou_threshold is None:
        iou_threshold = 0.70

    if variant == "c0003":
        kept_prediction = None
        xgb_prediction = (
            _to_float(row.xgb_pred)
            if row.xgb_pred_present
            else _meets(xgb_probability, xgb_threshold)
        )
        final_prediction = (
            _to_float(row.final_pred)
            if row.final_pred_present
            else _to_float(row.kept)
        )
    else:
        kept_prediction = _to_float(
            normalize_binary_label(row.kept, variant="c0004")
        )
        xgb_prediction = (
            _meets(xgb_probability, xgb_threshold)
            if sparse_xgb_fallback
            else _to_float(normalize_binary_label(row.xgb_pred, variant="c0004"))
        )
        final_prediction = (
            _to_float(normalize_binary_label(row.final_pred, variant="c0004"))
            if row.final_pred_present
            else kept_prediction
        )

    return MetaEventRecord(
        source=row,
        source_ordinal=source_ordinal,
        item_key=f"{row.image}||{row.identifier}",
        event_time=None if event_time_value is None else event_time_value.value,
        event_time_nanoseconds=(
            None if event_time_value is None else event_time_value.epoch_nanoseconds
        ),
        source_mtime_time=(
            None if source_mtime_value is None else source_mtime_value.value
        ),
        source_mtime_nanoseconds=(
            None
            if source_mtime_value is None
            else source_mtime_value.epoch_nanoseconds
        ),
        human_label_n=_as_float32(
            normalize_binary_label(row.human_label, variant=variant)
        ),
        human_last=None,
        human_prev=None,
        label_changed=False,
        xgb_prob_f=xgb_probability,
        yolo_conf_f=yolo_confidence,
        yolo_iou_f=yolo_iou,
        effective_xgb_threshold=xgb_threshold,
        effective_yolo_threshold=yolo_threshold,
        effective_iou_threshold=float(iou_threshold),
        xgb_pred_eff=xgb_prediction,
        kept_eff=kept_prediction,
        yolo_pred_eff=(
            1.0
            if _meets(yolo_confidence, yolo_threshold)
            and _meets(yolo_iou, iou_threshold)
            else 0.0
        ),
        final_pred_eff=final_prediction,
    )


def _evaluate_meta_events(
    rows: tuple[MetaEventInputRow, ...],
    column_dtypes: MetaEventColumnDtypes,
    settings: MetaEventSettings,
    *,
    variant: MetaVariant,
) -> MetaEventStatisticsResult:
    if type(settings) is not MetaEventSettings:
        raise MetaEventStatisticsContractError(
            "meta event settings must use the typed contract"
        )
    if type(column_dtypes) is not MetaEventColumnDtypes:
        raise MetaEventStatisticsContractError(
            "meta event column dtypes must use the typed contract"
        )
    if type(rows) is not tuple:
        raise MetaEventStatisticsContractError("meta event rows must use an exact tuple")
    if any(type(row) is not MetaEventInputRow for row in rows):
        raise MetaEventStatisticsContractError(
            "meta event rows must use MetaEventInputRow"
        )
    _validate_column_dtypes(rows, column_dtypes)
    if variant == "c0003" and column_dtypes.timestamp == "numeric" and not rows:
        raise MetaEventStatisticsContractError(
            "C0003 numeric timestamps require at least one row"
        )
    xgb_presence = {row.xgb_pred_present for row in rows}
    final_presence = {row.final_pred_present for row in rows}
    if rows and (len(xgb_presence) != 1 or len(final_presence) != 1):
        raise MetaEventStatisticsContractError(
            "meta event source-column presence must be consistent"
        )
    if rows and variant == "c0004" and xgb_presence != {True}:
        raise MetaEventStatisticsContractError(
            "C0004 requires the source XGB prediction column"
        )

    sparse_xgb_fallback = False
    if variant == "c0004":
        valid_xgb = sum(
            normalize_binary_label(row.xgb_pred, variant="c0004") is not None
            for row in rows
        )
        sparse_xgb_fallback = bool(rows) and (
            valid_xgb / len(rows) < settings.xgb_sparse_fraction
        )

    event_unit = (
        _event_numeric_unit(rows)
        if variant == "c0003" and column_dtypes.timestamp == "numeric"
        else None
    )
    source_mtime_numeric = column_dtypes.source_mtime == "numeric"
    initial = tuple(
        _initial_record(
            row,
            source_ordinal,
            variant=variant,
            event_numeric_unit=event_unit,
            column_dtypes=column_dtypes,
            sparse_xgb_fallback=sparse_xgb_fallback,
        )
        for source_ordinal, row in enumerate(rows)
    )
    ordered = sorted(
        initial,
        key=lambda event: (
            event.item_key,
            _time_key(event.event_time_nanoseconds),
            _raw_source_time_key(
                event.source,
                numeric_column=source_mtime_numeric,
            ),
            event.source.row_in_file,
            event.source_ordinal,
        ),
    )

    last_human: dict[str, float | None] = {}
    events: list[MetaEventRecord] = []
    for event in ordered:
        previous = last_human.get(event.item_key)
        current = event.human_label_n
        human_last = previous if current is None else current
        changed = human_last is not None and previous is not None and human_last != previous
        normalized = replace(
            event,
            human_last=human_last,
            human_prev=previous,
            label_changed=changed,
        )
        events.append(normalized)
        last_human[event.item_key] = human_last

    latest_by_key: dict[str, MetaEventRecord] = {}
    for event in events:
        current = latest_by_key.get(event.item_key)
        selection_key = (
            _latest_time_key(event.event_time_nanoseconds),
            _latest_raw_source_time_key(
                event.source,
                numeric_column=source_mtime_numeric,
            ),
            event.source.row_in_file,
            -event.source_ordinal,
        )
        if current is None:
            latest_by_key[event.item_key] = event
            continue
        current_key = (
            _latest_time_key(current.event_time_nanoseconds),
            _latest_raw_source_time_key(
                current.source,
                numeric_column=source_mtime_numeric,
            ),
            current.source.row_in_file,
            -current.source_ordinal,
        )
        if selection_key > current_key:
            latest_by_key[event.item_key] = event
    latest = tuple(latest_by_key[key] for key in sorted(latest_by_key))

    overrides: list[MetaLatestOverride] = []
    for event in latest:
        human_ok = event.human_last is not None
        agree_yolo = (
            human_ok
            and event.yolo_pred_eff is not None
            and float(event.human_last) == event.yolo_pred_eff
        )
        agree_xgb = (
            human_ok
            and event.xgb_pred_eff is not None
            and float(event.human_last) == event.xgb_pred_eff
        )
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
        override_final = (
            human_ok
            and (variant == "c0003" or event.final_pred_eff is not None)
            and float(event.human_last) != event.final_pred_eff
        )
        overrides.append(
            MetaLatestOverride(
                event,
                event.source.detector_policy or "nan",
                human_ok,
                override_final,
                agree_yolo,
                agree_xgb,
                bucket,
            )
        )

    item_counts: dict[str, int] = {}
    for event in events:
        item_counts[event.item_key] = item_counts.get(event.item_key, 0) + 1
    flips = tuple(event for event in events if event.label_changed)
    human_overrides = tuple(item for item in overrides if item.human_ok)
    override_rate = (
        sum(item.override_final for item in human_overrides) / len(human_overrides)
        if human_overrides
        else (float("nan") if variant == "c0003" else None)
    )
    observed_bucket_counts: dict[str, int] = {}
    for item in human_overrides:
        observed_bucket_counts[item.align_bucket] = (
            observed_bucket_counts.get(item.align_bucket, 0) + 1
        )
    bucket_counts = tuple(
        sorted(
            observed_bucket_counts.items(),
            key=lambda item: -item[1],
        )
    )
    return MetaEventStatisticsResult(
        variant,
        tuple(events),
        flips,
        latest,
        tuple(overrides),
        len(events),
        len(item_counts),
        (
            tuple(
                (fraction, _quantile(tuple(item_counts.values()), fraction))
                for fraction in settings.event_quantiles
            )
            if item_counts
            else tuple((fraction, float("nan")) for fraction in settings.event_quantiles)
        ),
        len(flips),
        len({event.item_key for event in flips}),
        len(human_overrides),
        override_rate,
        bucket_counts,
        sparse_xgb_fallback,
    )


# SOURCE_CELL: NB-LIVE-0010-C0003
# SOURCE_STATEMENT_MAP: event-history-effective-preds-latest-overrides-statistics -> evaluate_meta_events_c0003
def evaluate_meta_events_c0003(
    rows: tuple[MetaEventInputRow, ...],
    column_dtypes: MetaEventColumnDtypes,
    settings: MetaEventSettings = MetaEventSettings(),
) -> MetaEventStatisticsResult:
    """Evaluate the C0003 parser and direct XGB-column fallback policy."""

    return _evaluate_meta_events(rows, column_dtypes, settings, variant="c0003")


# SOURCE_CELL: NB-LIVE-0010-C0004
# SOURCE_STATEMENT_MAP: robust-binary-sparse-xgb-latest-overrides-statistics -> evaluate_meta_events_c0004
def evaluate_meta_events_c0004(
    rows: tuple[MetaEventInputRow, ...],
    column_dtypes: MetaEventColumnDtypes,
    settings: MetaEventSettings = MetaEventSettings(),
) -> MetaEventStatisticsResult:
    """Evaluate C0004 robust parsing and the less-than-five-percent XGB fallback."""

    return _evaluate_meta_events(rows, column_dtypes, settings, variant="c0004")


__all__ = [
    "MetaEventColumnDtypes",
    "MetaEventInputRow",
    "MetaEventRecord",
    "MetaEventSettings",
    "MetaEventStatisticsContractError",
    "MetaEventStatisticsResult",
    "MetaLatestOverride",
    "MetaParsedTimestamp",
    "evaluate_meta_events_c0003",
    "evaluate_meta_events_c0004",
    "normalize_binary_label",
    "parse_event_timestamp",
]
