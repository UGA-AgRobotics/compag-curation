"""Candidate-level border/tiny evaluation using source-locked definitions."""

from __future__ import annotations

from csv import DictReader, DictWriter
from dataclasses import dataclass, field
from json import dumps
from math import inf, isfinite, nan
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ...config import DomainConfig
from ...contracts import artifact_path, create_output_directory


SCORE_COLUMNS = ("xgb_p", "xgb_proba", "proba", "prob", "p_fused", "score")
LABEL_COLUMNS = ("human_label", "label", "y", "gt")
REVIEW_WEIGHTS = {
    "accept": 1.0,
    "flip": 1.0,
    "sus_accept": 0.4,
    "sus_flip": 0.4,
    "skip": 0.0,
    "auto_accept": 0.0,
}


class BorderTinyContractError(ValueError):
    pass


@dataclass(frozen=True)
class BorderTinySettings:
    threshold: float = 0.5
    tile_size: int = 512
    border_pixels: int = 4
    tiny_minimum_dimension: int = 32
    tiny_maximum_area: int = 1500
    review_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(REVIEW_WEIGHTS)
    )
    drop_zero_weight: bool = True


@dataclass(frozen=True)
class SliceMetrics:
    name: str
    count: int
    positives: int
    positive_rate: float
    average_precision: float
    precision: float
    recall: float


@dataclass(frozen=True)
class BorderTinyResult:
    total_candidates: int
    labeled_candidates: int
    border_or_tiny_candidates: int
    coverage: float
    slices: tuple[SliceMetrics, ...]
    area_slices: tuple[SliceMetrics, ...]
    output: Path


BorderTinyParameter = str | int | float | bool


@dataclass(frozen=True)
class BorderTinyOperation:
    """One executable node in the border/tiny evaluation program."""

    operation: str
    source_cells: tuple[str, ...]
    input_roles: tuple[str, ...]
    output_role: str
    parameters: tuple[tuple[str, BorderTinyParameter], ...] = ()


def build_border_tiny_program(settings: BorderTinySettings) -> tuple[BorderTinyOperation, ...]:
    return (
        BorderTinyOperation(
            "read-sealed-candidate-tables",
            ("NB-LIVE-0011-C0000", "NB-LIVE-0011-C0001"),
            ("predictions", "annotations"),
            "candidate_rows",
        ),
        BorderTinyOperation(
            "standardize-border-tiny-flags",
            ("NB-LIVE-0011-C0002", "NB-LIVE-0011-C0003"),
            ("candidate_rows",),
            "evaluated_rows",
            (
                ("threshold", settings.threshold),
                ("tile_size", settings.tile_size),
                ("border_pixels", settings.border_pixels),
                ("tiny_minimum_dimension", settings.tiny_minimum_dimension),
                ("tiny_maximum_area", settings.tiny_maximum_area),
                ("drop_zero_weight", settings.drop_zero_weight),
            ),
        ),
        BorderTinyOperation(
            "compute-weighted-slice-metrics",
            ("NB-LIVE-0011-C0004", "NB-LIVE-0011-C0005"),
            ("evaluated_rows",),
            "slice_metrics",
            (("threshold", settings.threshold),),
        ),
        BorderTinyOperation(
            "compute-area-strata",
            ("NB-LIVE-0011-C0006",),
            ("evaluated_rows",),
            "area_slice_metrics",
        ),
        BorderTinyOperation(
            "write-declared-evaluation-artifacts",
            tuple(f"NB-LIVE-0011-C{index:04d}" for index in range(7)),
            ("slice_metrics", "area_slice_metrics"),
            "output",
        ),
    )


def _float(value: Any, default: float = nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if isfinite(parsed) else default


def _integer_label(value: Any) -> int | None:
    parsed = _float(value)
    if parsed in (0.0, 1.0):
        return int(parsed)
    return None


def candidate_key(image: Any, detection_id: Any) -> str:
    return f"{str(image)}||{str(detection_id)}"


def _first_column(row: Mapping[str, Any], candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in row:
            return name
    raise BorderTinyContractError(f"none of these columns exist: {list(candidates)}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise BorderTinyContractError(f"CSV is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        return [dict(row) for row in DictReader(handle)]


# SOURCE_CELL: NB-LIVE-0011-C0002
# SOURCE_STATEMENT_MAP: bbox-standardization-and-paper-thresholds -> classify_border_tiny
def classify_border_tiny(
    row: Mapping[str, Any],
    settings: BorderTinySettings,
) -> tuple[bool, bool]:
    """Return source-defined ``(is_border, is_tiny)`` for one candidate."""

    if all(name in row for name in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")):
        x = _float(row.get("bbox_x"), 0.0)
        y = _float(row.get("bbox_y"), 0.0)
        width = _float(row.get("bbox_w"), 0.0)
        height = _float(row.get("bbox_h"), 0.0)
    elif all(name in row for name in ("x", "y", "w", "h")):
        x = _float(row.get("x"), 0.0)
        y = _float(row.get("y"), 0.0)
        width = _float(row.get("w"), 0.0)
        height = _float(row.get("h"), 0.0)
    elif "cx" in row and "cy" in row:
        x = _float(row.get("cx"), 0.0) - 15.0
        y = _float(row.get("cy"), 0.0) - 15.0
        width = height = 30.0
    else:
        raise BorderTinyContractError(
            "bbox requires bbox_x/y/w/h, x/y/w/h, or centroid cx/cy"
        )
    tile_width = max(1.0, _float(row.get("tile_w"), float(settings.tile_size)))
    tile_height = max(1.0, _float(row.get("tile_h"), float(settings.tile_size)))
    touching = _float(row.get("touching_border"), 0.0) == 1.0
    margin = float(settings.border_pixels)
    is_border = (
        touching
        or x <= margin
        or y <= margin
        or x + width >= tile_width - margin
        or y + height >= tile_height - margin
    )
    is_tiny = (
        min(width, height) < float(settings.tiny_minimum_dimension)
        or width * height < float(settings.tiny_maximum_area)
    )
    return is_border, is_tiny


# SOURCE_CELL: NB-LIVE-0011-C0002
# SOURCE_STATEMENT_MAP: weighted-tp-fp-fn -> weighted_precision_recall
def weighted_precision_recall(
    labels: Sequence[int],
    scores: Sequence[float],
    weights: Sequence[float],
    threshold: float,
) -> tuple[float, float]:
    if not (len(labels) == len(scores) == len(weights)):
        raise BorderTinyContractError("metric vectors must have equal length")
    true_positive = sum(
        weight
        for label, score, weight in zip(labels, scores, weights)
        if label == 1 and score >= threshold
    )
    false_positive = sum(
        weight
        for label, score, weight in zip(labels, scores, weights)
        if label == 0 and score >= threshold
    )
    false_negative = sum(
        weight
        for label, score, weight in zip(labels, scores, weights)
        if label == 1 and score < threshold
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else nan
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else nan
    )
    return precision, recall


def weighted_average_precision(
    labels: Sequence[int],
    scores: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Weighted non-interpolated AP, matching the source sklearn call role."""

    ordered = sorted(
        zip(labels, scores, weights),
        key=lambda item: item[1],
        reverse=True,
    )
    total_positive = sum(weight for label, _, weight in ordered if label == 1)
    if total_positive <= 0:
        return nan
    accumulated_positive = 0.0
    accumulated_weight = 0.0
    weighted_precision_sum = 0.0
    for label, _, weight in ordered:
        accumulated_weight += weight
        if label == 1:
            accumulated_positive += weight
            if accumulated_weight > 0:
                weighted_precision_sum += weight * (
                    accumulated_positive / accumulated_weight
                )
    return weighted_precision_sum / total_positive


def compute_slice_metrics(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
) -> SliceMetrics:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    weights = [float(row["weight"]) for row in rows]
    precision, recall = weighted_precision_recall(labels, scores, weights, threshold)
    positives = sum(label == 1 for label in labels)
    return SliceMetrics(
        name=name,
        count=len(rows),
        positives=positives,
        positive_rate=positives / len(rows) if rows else nan,
        average_precision=weighted_average_precision(labels, scores, weights),
        precision=precision,
        recall=recall,
    )


# SOURCE_CELL: NB-LIVE-0011-C0003
# SOURCE_STATEMENT_MAP: append-log-last-label-merge-sanitize -> prepare_evaluation_rows
def prepare_evaluation_rows(
    detections: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    settings: BorderTinySettings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge effective review labels onto the complete candidate pool."""

    effective: dict[str, tuple[tuple[int, float | str], dict[str, Any]]] = {}
    for ordinal, raw in enumerate(review_rows):
        if "image" not in raw or "id" not in raw:
            raise BorderTinyContractError("review rows require image and id")
        label = _integer_label(raw.get("human_label", raw.get("label")))
        if label is None:
            continue
        timestamp_value = raw.get("timestamp", ordinal)
        timestamp_float = _float(timestamp_value)
        order: tuple[int, float | str] = (
            (1, timestamp_float) if isfinite(timestamp_float) else (0, str(timestamp_value))
        )
        action = str(raw.get("action", "")).strip().lower()
        weight = _float(raw.get("review_weight"))
        if not isfinite(weight):
            weight = float(settings.review_weights.get(action, 1.0))
        normalized = {"label": label, "weight": weight, "action": action}
        key = candidate_key(raw["image"], raw["id"])
        if key not in effective or order >= effective[key][0]:
            effective[key] = (order, normalized)

    all_rows: list[dict[str, Any]] = []
    evaluated: list[dict[str, Any]] = []
    for raw in detections:
        if "image" not in raw or "id" not in raw:
            raise BorderTinyContractError("detections require image and id")
        score_name = _first_column(raw, SCORE_COLUMNS)
        score = _float(raw.get(score_name))
        is_border, is_tiny = classify_border_tiny(raw, settings)
        item = dict(raw)
        item.update(
            {
                "key": candidate_key(raw["image"], raw["id"]),
                "score": score,
                "is_border": is_border,
                "is_tiny": is_tiny,
                "border_or_tiny": is_border or is_tiny,
            }
        )
        label = effective.get(item["key"])
        if label is not None:
            item.update(label[1])
        all_rows.append(item)
        if (
            label is not None
            and isfinite(score)
            and (not settings.drop_zero_weight or float(item["weight"]) > 0)
        ):
            evaluated.append(item)
    return all_rows, evaluated


def _area_bucket(row: Mapping[str, Any]) -> str:
    width = _float(row.get("bbox_w", row.get("w")), 0.0)
    height = _float(row.get("bbox_h", row.get("h")), 0.0)
    area = width * height
    for upper, label in (
        (1500.0, "<1.5k"),
        (4000.0, "1.5-4k"),
        (9000.0, "4-9k"),
        (20000.0, "9-20k"),
        (inf, ">=20k"),
    ):
        if area < upper:
            return label
    return ">=20k"


# SOURCE_CELL: NB-LIVE-0011-C0004
# SOURCE_CELL: NB-LIVE-0011-C0005
# SOURCE_CELL: NB-LIVE-0011-C0006
# SOURCE_STATEMENT_MAP: overall-coverage-strata-area-bins -> evaluate_border_tiny_records
def evaluate_border_tiny_records(
    detections: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    settings: BorderTinySettings = BorderTinySettings(),
) -> tuple[int, int, int, float, tuple[SliceMetrics, ...], tuple[SliceMetrics, ...]]:
    all_rows, evaluated = prepare_evaluation_rows(detections, review_rows, settings)
    border = [row for row in evaluated if bool(row["border_or_tiny"])]
    non_border = [row for row in evaluated if not bool(row["border_or_tiny"])]
    slices = (
        compute_slice_metrics("border-or-tiny", border, settings.threshold),
        compute_slice_metrics("non-border/tiny", non_border, settings.threshold),
        compute_slice_metrics("OVERALL", evaluated, settings.threshold),
    )
    area_slices: list[SliceMetrics] = []
    for label in ("<1.5k", "1.5-4k", "4-9k", "9-20k", ">=20k"):
        selected = [row for row in evaluated if _area_bucket(row) == label]
        if selected:
            area_slices.append(
                compute_slice_metrics(f"area {label}", selected, settings.threshold)
            )
    total = len(all_rows)
    coverage = len(evaluated) / total if total else nan
    border_pool = sum(bool(row["border_or_tiny"]) for row in all_rows)
    return total, len(evaluated), border_pool, coverage, slices, tuple(area_slices)


def _metrics_dict(metrics: SliceMetrics) -> dict[str, Any]:
    return {
        "slice": metrics.name,
        "n": metrics.count,
        "pos": metrics.positives,
        "pos_rate": metrics.positive_rate,
        "pr_auc": metrics.average_precision,
        "precision@tau": metrics.precision,
        "recall@tau": metrics.recall,
    }


def _write_result(
    output: Path,
    total: int,
    labeled: int,
    border_pool: int,
    coverage: float,
    slices: Sequence[SliceMetrics],
    area_slices: Sequence[SliceMetrics],
) -> Path:
    create_output_directory(output)
    slice_path = output / "supp_stratified_metrics__border_tiny.csv"
    rows = [_metrics_dict(metric) for metric in slices]
    with slice_path.open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    area_path = output / "supp_stratified_metrics__size_bins.csv"
    area_rows = [_metrics_dict(metric) for metric in area_slices]
    with area_path.open("w", newline="", encoding="utf-8") as handle:
        fields = tuple(area_rows[0]) if area_rows else tuple(rows[0])
        writer = DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(area_rows)
    (output / "border_tiny_summary.json").write_text(
        dumps(
            {
                "total_candidates": total,
                "labeled_candidates": labeled,
                "border_or_tiny_candidates": border_pool,
                "coverage": coverage,
            },
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    output_root = output.resolve(strict=True)
    for artifact in (
        slice_path,
        area_path,
        output / "border_tiny_summary.json",
    ):
        if artifact.is_symlink() or not artifact.is_file():
            raise BorderTinyContractError(
                "border/tiny output artifact must be a regular non-symlink file"
            )
        try:
            artifact.resolve(strict=True).relative_to(output_root)
        except (OSError, ValueError) as exc:
            raise BorderTinyContractError(
                "border/tiny output artifact escaped the declared output"
            ) from exc
    return output


# SOURCE_CELL: NB-LIVE-0011-C0000
# SOURCE_CELL: NB-LIVE-0011-C0001
# SOURCE_STATEMENT_MAP: imports-and-locked-config -> run_border_tiny_evaluation
def execute_border_tiny_evaluation(config: DomainConfig) -> BorderTinyResult:
    """Registry handler for source-backed border/tiny evaluation."""

    values = config.workflow("evaluate")
    detections_path = artifact_path(values, "predictions")
    review_path = artifact_path(
        values, "review_labels" if "review_labels" in values else "annotations"
    )
    output = artifact_path(values, "output")
    settings = BorderTinySettings(
        threshold=float(values.get("decision_threshold", 0.5)),
        tile_size=int(values.get("tile_size", 512)),
        border_pixels=int(values.get("border_pixels", 4)),
        tiny_minimum_dimension=int(values.get("tiny_minimum_dimension", 32)),
        tiny_maximum_area=int(values.get("tiny_maximum_area", 1500)),
        drop_zero_weight=bool(values.get("drop_zero_weight", True)),
    )
    detections: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []
    all_rows: list[dict[str, Any]] = []
    evaluated: list[dict[str, Any]] = []
    total = 0
    labeled = 0
    border_pool = 0
    coverage = nan
    slices: tuple[SliceMetrics, ...] = ()
    area_slices: tuple[SliceMetrics, ...] = ()
    destination = output

    for operation in build_border_tiny_program(settings):
        parameters = dict(operation.parameters)
        if operation.operation == "read-sealed-candidate-tables":
            detections = _read_csv(detections_path)
            review_rows = _read_csv(review_path)
        elif operation.operation == "standardize-border-tiny-flags":
            program_settings = BorderTinySettings(
                threshold=float(parameters["threshold"]),
                tile_size=int(parameters["tile_size"]),
                border_pixels=int(parameters["border_pixels"]),
                tiny_minimum_dimension=int(parameters["tiny_minimum_dimension"]),
                tiny_maximum_area=int(parameters["tiny_maximum_area"]),
                drop_zero_weight=bool(parameters["drop_zero_weight"]),
                review_weights=settings.review_weights,
            )
            all_rows, evaluated = prepare_evaluation_rows(
                detections,
                review_rows,
                program_settings,
            )
            total = len(all_rows)
            labeled = len(evaluated)
            border_pool = sum(bool(row["border_or_tiny"]) for row in all_rows)
            coverage = labeled / total if total else nan
        elif operation.operation == "compute-weighted-slice-metrics":
            threshold = float(parameters["threshold"])
            border = [row for row in evaluated if bool(row["border_or_tiny"])]
            non_border = [row for row in evaluated if not bool(row["border_or_tiny"])]
            slices = (
                compute_slice_metrics("border-or-tiny", border, threshold),
                compute_slice_metrics("non-border/tiny", non_border, threshold),
                compute_slice_metrics("OVERALL", evaluated, threshold),
            )
        elif operation.operation == "compute-area-strata":
            area_values: list[SliceMetrics] = []
            for label in ("<1.5k", "1.5-4k", "4-9k", "9-20k", ">=20k"):
                selected = [row for row in evaluated if _area_bucket(row) == label]
                if selected:
                    area_values.append(
                        compute_slice_metrics(
                            f"area {label}",
                            selected,
                            settings.threshold,
                        )
                    )
            area_slices = tuple(area_values)
        elif operation.operation == "write-declared-evaluation-artifacts":
            destination = _write_result(
                output,
                total,
                labeled,
                border_pool,
                coverage,
                slices,
                area_slices,
            )
        else:
            raise BorderTinyContractError(
                f"unsupported border/tiny program operation: {operation.operation}"
            )
    return BorderTinyResult(
        total, labeled, border_pool, coverage, slices, area_slices, destination
    )
