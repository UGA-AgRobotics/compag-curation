"""Pure scientific reporting helpers for Amendment 06.

This module consumes already-frozen prediction/metric evidence. It contains no
training, imputation, augmentation, resampling, or model-selection operation.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASELINE_VARIANT = "full_new_reference"
OFFICIAL_RUNNABLE_VARIANTS = (
    "full_new_reference",
    "manual_only",
    "deep_only",
    "no_color",
    "no_shape",
    "no_texture",
    "no_embed_sim",
    "no_review_aware_training_weights",
    "no_safe_smote",
)
EXPLORATORY_VARIANT = "no_deep_pca_features"
BLOCKED_VARIANT = "no_pca"
DECISION_THRESHOLD = 0.5
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2_026_081_201
PRIMARY_METRICS = (
    "average_precision",
    "roc_auc",
    "precision",
    "recall",
    "f1",
)
BOOTSTRAP_METRICS = ("average_precision", "f1", "recall")
PREDICTION_COLUMNS = (
    "stable_candidate_id",
    "card_id",
    "true_label",
    "probability",
    "prediction_at_0_5",
    "review_weight",
    "variant_id",
    "split_population",
    "row_order",
)


class Amendment06ReportingError(RuntimeError):
    """A frozen reporting input or generated artifact is invalid."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite_array(values: Sequence[float], *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise Amendment06ReportingError(f"{label} must be a finite 1D vector")
    return array


def _binary_labels(values: Sequence[int], *, label: str) -> np.ndarray:
    numeric = np.asarray(values)
    if numeric.ndim != 1:
        raise Amendment06ReportingError(f"{label} must be a 1D vector")
    try:
        array = numeric.astype(np.int8)
    except (TypeError, ValueError) as exc:
        raise Amendment06ReportingError(f"{label} is not binary numeric") from exc
    if not np.array_equal(numeric.astype(np.float64), array.astype(np.float64)) or not np.isin(array, [0, 1]).all():
        raise Amendment06ReportingError(f"{label} must contain only 0 and 1")
    return array


def binary_metrics(
    y: Sequence[int],
    probability: Sequence[float],
    *,
    threshold: float = DECISION_THRESHOLD,
    weights: Sequence[float] | None = None,
) -> dict[str, float | int]:
    """Compute the finite fixed-threshold Amendment 06 metric contract."""

    labels = _binary_labels(y, label="labels")
    scores = _finite_array(probability, label="probabilities")
    if len(labels) != len(scores) or len(labels) == 0 or set(labels.tolist()) != {0, 1}:
        raise Amendment06ReportingError(
            "Required aggregate metrics need aligned nonempty two-class vectors"
        )
    if not math.isfinite(float(threshold)) or float(threshold) != DECISION_THRESHOLD:
        raise Amendment06ReportingError("Decision threshold must be exactly 0.5")
    sample_weight: np.ndarray | None = None
    if weights is not None:
        sample_weight = _finite_array(weights, label="sample weights")
        if len(sample_weight) != len(labels) or np.any(sample_weight < 0) or sample_weight.sum() <= 0:
            raise Amendment06ReportingError(
                "Sample weights must be aligned, nonnegative, and have positive mass"
            )
    predicted = (scores >= float(threshold)).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predicted,
        labels=[0, 1],
        sample_weight=sample_weight,
    ).ravel()
    result: dict[str, float | int] = {
        "average_precision": float(
            average_precision_score(labels, scores, sample_weight=sample_weight)
        ),
        "roc_auc": float(roc_auc_score(labels, scores, sample_weight=sample_weight)),
        "precision": float(
            precision_score(
                labels, predicted, sample_weight=sample_weight, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                labels, predicted, sample_weight=sample_weight, zero_division=0
            )
        ),
        "f1": float(
            f1_score(labels, predicted, sample_weight=sample_weight, zero_division=0)
        ),
        "tn": int(tn) if sample_weight is None else float(tn),
        "fp": int(fp) if sample_weight is None else float(fp),
        "fn": int(fn) if sample_weight is None else float(fn),
        "tp": int(tp) if sample_weight is None else float(tp),
        "threshold": float(threshold),
        "rows": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
    }
    required = [float(result[key]) for key in (*PRIMARY_METRICS, "tn", "fp", "fn", "tp")]
    if not np.isfinite(required).all():
        raise Amendment06ReportingError("A required aggregate metric is non-finite")
    return result


def _validate_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise Amendment06ReportingError(
            f"Prediction columns differ: {tuple(frame.columns)}"
        )
    if len(frame) == 0 or frame.stable_candidate_id.astype(str).duplicated().any():
        raise Amendment06ReportingError("Prediction identity is empty or duplicated")
    expected_order = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(pd.to_numeric(frame.row_order).to_numpy(np.int64), expected_order):
        raise Amendment06ReportingError("Prediction row_order is not exact contiguous order")
    labels = _binary_labels(frame.true_label, label="prediction labels")
    predicted = _binary_labels(frame.prediction_at_0_5, label="threshold predictions")
    probability = _finite_array(frame.probability, label="prediction probabilities")
    weights = _finite_array(frame.review_weight, label="prediction review weights")
    if (
        np.any(weights <= 0)
        or np.any(probability < 0)
        or np.any(probability > 1)
        or not np.array_equal(
        predicted, (probability >= DECISION_THRESHOLD).astype(np.int8)
        )
    ):
        raise Amendment06ReportingError("Prediction threshold/weight contract differs")
    if (
        frame.card_id.astype(str).eq("").any()
        or frame.split_population.astype(str).eq("").any()
        or frame.variant_id.astype(str).nunique() != 1
    ):
        raise Amendment06ReportingError("Prediction card/Variant identity is invalid")
    normalized = frame.copy()
    # A float32 scientific value must be promoted exactly before pandas text
    # formatting; otherwise e.g. float32(0.1) may be emitted as the shorter
    # decimal 0.1 and cannot reopen to the authoritative promoted bit pattern.
    normalized["probability"] = probability
    normalized["review_weight"] = weights
    normalized["true_label"] = labels
    normalized["prediction_at_0_5"] = predicted
    normalized["row_order"] = expected_order
    return normalized


def exact_prediction_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize and immediately prove exact C-engine round-trip equality."""

    source = _validate_prediction_frame(frame)
    data = source.to_csv(index=False, lineterminator="\n").encode("utf-8")
    reopened = pd.read_csv(
        io.BytesIO(data), engine="c", float_precision="round_trip",
        keep_default_na=False,
        dtype={
            "stable_candidate_id": "string", "card_id": "string",
            "variant_id": "string", "split_population": "string",
        },
    )
    validate_prediction_csv_roundtrip(data, source)
    if tuple(reopened.columns) != PREDICTION_COLUMNS:
        raise Amendment06ReportingError("Round-trip prediction schema changed")
    return data


def deterministic_gzip_bytes(data: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as stream:
        stream.write(data)
    return output.getvalue()


def validate_prediction_csv_roundtrip(
    path_or_bytes: Path | bytes,
    expected: pd.DataFrame,
    *,
    metrics: Mapping[str, Any] | None = None,
    compressed: bool | None = None,
) -> dict[str, Any]:
    if isinstance(path_or_bytes, Path):
        data = path_or_bytes.read_bytes()
        is_compressed = path_or_bytes.suffix == ".gz" if compressed is None else compressed
    else:
        data = bytes(path_or_bytes)
        is_compressed = False if compressed is None else compressed
    raw = gzip.decompress(data) if is_compressed else data
    reopened = pd.read_csv(
        io.BytesIO(raw), engine="c", float_precision="round_trip",
        keep_default_na=False,
        dtype={
            "stable_candidate_id": "string", "card_id": "string",
            "variant_id": "string", "split_population": "string",
        },
    )
    expected = _validate_prediction_frame(expected)
    if tuple(reopened.columns) != PREDICTION_COLUMNS or len(reopened) != len(expected):
        raise Amendment06ReportingError("Prediction CSV round-trip shape/schema differs")
    string_columns = (
        "stable_candidate_id", "card_id", "variant_id", "split_population",
    )
    integer_columns = ("true_label", "prediction_at_0_5", "row_order")
    float_columns = ("probability", "review_weight")
    for column in string_columns:
        if not np.array_equal(
            reopened[column].astype(str).to_numpy(),
            expected[column].astype(str).to_numpy(),
        ):
            raise Amendment06ReportingError(f"Prediction CSV changed {column}")
    for column in integer_columns:
        if not np.array_equal(
            pd.to_numeric(reopened[column]).to_numpy(np.int64),
            pd.to_numeric(expected[column]).to_numpy(np.int64),
        ):
            raise Amendment06ReportingError(f"Prediction CSV changed {column}")
    for column in float_columns:
        if not np.array_equal(
            pd.to_numeric(reopened[column]).to_numpy(np.float64),
            pd.to_numeric(expected[column]).to_numpy(np.float64),
        ):
            raise Amendment06ReportingError(f"Prediction CSV changed {column}")
    recomputed = binary_metrics(
        reopened.true_label,
        reopened.probability,
        threshold=DECISION_THRESHOLD,
    )
    if metrics is not None:
        for key in (*PRIMARY_METRICS, "tn", "fp", "fn", "tp"):
            if key not in metrics or type(metrics[key]) not in {int, float}:
                raise Amendment06ReportingError(
                    f"Saved prediction metric is missing or nonnumeric: {key}"
                )
            if float(metrics[key]) != float(recomputed[key]):
                raise Amendment06ReportingError(
                    f"Saved prediction metric differs exactly after reopen: {key}"
                )
    return {
        "status": "PASS",
        "rows": len(expected),
        "csv_sha256": _sha256_bytes(data),
        "float_precision": "round_trip",
        "engine": "c",
        "numeric_equality": "EXACT",
        "metric_recomputation": "PASS" if metrics is not None else "NOT_REQUESTED",
    }


prediction_csv_bytes = exact_prediction_csv_bytes


def per_card_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _validate_prediction_frame(frame)
    rows: list[dict[str, Any]] = []
    variant = str(frame.variant_id.iloc[0])
    for card, group in frame.groupby("card_id", sort=True):
        labels = group.true_label.to_numpy(np.int8)
        scores = group.probability.to_numpy(np.float64)
        base = binary_metrics(labels, scores) if set(labels.tolist()) == {0, 1} else None
        if base is None:
            predicted = (scores >= DECISION_THRESHOLD).astype(np.int8)
            tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
            precision = precision_score(labels, predicted, zero_division=0)
            recall = recall_score(labels, predicted, zero_division=0)
            f1 = f1_score(labels, predicted, zero_division=0)
            base = {
                "average_precision": None,
                "roc_auc": None,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            }
            status = "UNDEFINED_AP_AUC_SINGLE_CLASS"
        else:
            status = "PASS"
        rows.append({
            "variant": variant,
            "card_id": str(card),
            "rows": len(group),
            "positive": int(labels.sum()),
            "negative": int(len(labels) - labels.sum()),
            "status": status,
            **{key: base[key] for key in (*PRIMARY_METRICS, "tn", "fp", "fn", "tp")},
        })
    return pd.DataFrame(rows)


def _ordered_population_hash(frame: pd.DataFrame) -> str:
    payload = "".join(
        f"{sid}\t{card}\t{int(label)}\t{int(order)}\n"
        for sid, card, label, order in frame[
            ["stable_candidate_id", "card_id", "true_label", "row_order"]
        ].itertuples(index=False, name=None)
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _weighted_ap_from_card_counts(
    frame: pd.DataFrame,
    counts: np.ndarray,
    cards: Sequence[str],
) -> np.ndarray:
    probability = frame.probability.to_numpy(np.float64)
    labels = frame.true_label.to_numpy(np.int8)
    card_codes = pd.Categorical(frame.card_id.astype(str), categories=list(cards)).codes
    if np.any(card_codes < 0):
        raise Amendment06ReportingError("Prediction contains an unexpected Audit card")
    order = np.argsort(-probability, kind="mergesort")
    probability = probability[order]
    labels = labels[order]
    card_codes = card_codes[order]
    boundaries = np.r_[np.flatnonzero(np.diff(probability) != 0), len(probability) - 1]
    starts = np.r_[0, boundaries[:-1] + 1]
    total_by_group_card = np.zeros((len(boundaries), len(cards)), np.float64)
    positive_by_group_card = np.zeros_like(total_by_group_card)
    for group_index, (start, stop) in enumerate(zip(starts, boundaries + 1)):
        code_slice = card_codes[start:stop]
        total_by_group_card[group_index] = np.bincount(
            code_slice, minlength=len(cards)
        )
        positive_by_group_card[group_index] = np.bincount(
            code_slice,
            weights=labels[start:stop],
            minlength=len(cards),
        )
    result = np.empty(len(counts), np.float64)
    for start in range(0, len(counts), 256):
        multiplicity = counts[start : start + 256].astype(np.float64)
        group_total = multiplicity @ total_by_group_card.T
        group_positive = multiplicity @ positive_by_group_card.T
        cumulative_total = np.cumsum(group_total, axis=1)
        cumulative_positive = np.cumsum(group_positive, axis=1)
        total_positive = cumulative_positive[:, -1]
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        result[start : start + len(multiplicity)] = np.divide(
            (group_positive * precision).sum(axis=1),
            total_positive,
            out=np.full(len(multiplicity), np.nan),
            where=total_positive > 0,
        )
    if not np.isfinite(result).all():
        raise Amendment06ReportingError("Bootstrap produced an undefined AP replicate")
    return result


def _bootstrap_metrics(
    frame: pd.DataFrame,
    counts: np.ndarray,
    cards: Sequence[str],
) -> dict[str, np.ndarray]:
    ap = _weighted_ap_from_card_counts(frame, counts, cards)
    card_confusion: list[list[int]] = []
    for card in cards:
        group = frame.loc[frame.card_id.astype(str) == str(card)]
        labels = group.true_label.to_numpy(np.int8)
        predicted = group.prediction_at_0_5.to_numpy(np.int8)
        tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
        card_confusion.append([int(tn), int(fp), int(fn), int(tp)])
    aggregate = counts @ np.asarray(card_confusion, dtype=np.int64)
    _, fp, fn, tp = aggregate.T
    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros(len(counts), dtype=np.float64),
        where=(tp + fn) > 0,
    )
    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros(len(counts), dtype=np.float64),
        where=(tp + fp) > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(counts), dtype=np.float64),
        where=(precision + recall) > 0,
    )
    return {"average_precision": ap, "f1": f1, "recall": recall}


def paired_card_bootstrap(
    predictions_by_variant: Mapping[str, pd.DataFrame],
    cards: Sequence[str],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run one common paired card-cluster bootstrap without any model fit."""

    expected_variants = {*OFFICIAL_RUNNABLE_VARIANTS, EXPLORATORY_VARIANT}
    if set(predictions_by_variant) != expected_variants:
        raise Amendment06ReportingError("Bootstrap Variant set is not exact 9+1")
    card_order = tuple(map(str, cards))
    if len(card_order) != 10 or len(set(card_order)) != 10:
        raise Amendment06ReportingError("Bootstrap requires the exact ten Audit cards")
    if int(replicates) != BOOTSTRAP_REPLICATES or int(seed) != BOOTSTRAP_SEED:
        raise Amendment06ReportingError("Bootstrap replicate/seed contract differs")
    validated = {
        variant: _validate_prediction_frame(frame)
        for variant, frame in predictions_by_variant.items()
    }
    baseline_hash = _ordered_population_hash(validated[BASELINE_VARIANT])
    for variant, frame in validated.items():
        if _ordered_population_hash(frame) != baseline_hash:
            raise Amendment06ReportingError(
                f"Bootstrap population/order differs for {variant}"
            )
        if set(frame.card_id.astype(str)) != set(card_order):
            raise Amendment06ReportingError(f"Bootstrap card set differs for {variant}")
    rng = np.random.default_rng(int(seed))
    counts = rng.multinomial(
        len(card_order),
        np.full(len(card_order), 1.0 / len(card_order)),
        size=int(replicates),
    )
    baseline = _bootstrap_metrics(validated[BASELINE_VARIANT], counts, card_order)
    rows: list[dict[str, Any]] = []
    for variant in (*OFFICIAL_RUNNABLE_VARIANTS, EXPLORATORY_VARIANT):
        current = _bootstrap_metrics(validated[variant], counts, card_order)
        for metric in BOOTSTRAP_METRICS:
            delta = current[metric] - baseline[metric]
            lower, median, upper = np.percentile(delta, [2.5, 50.0, 97.5])
            if not np.isfinite([lower, median, upper]).all():
                raise Amendment06ReportingError("Bootstrap interval is non-finite")
            rows.append({
                "variant": variant,
                "classification": (
                    "EXPLORATORY_PCA_DERIVED_FEATURE_BLOCK_REMOVAL"
                    if variant == EXPLORATORY_VARIANT else "OFFICIAL_RUNNABLE"
                ),
                "metric": metric,
                "delta_definition": "variant_minus_full",
                "delta_point_estimate": float(
                    binary_metrics(
                        validated[variant].true_label,
                        validated[variant].probability,
                    )[metric]
                    - binary_metrics(
                        validated[BASELINE_VARIANT].true_label,
                        validated[BASELINE_VARIANT].probability,
                    )[metric]
                ),
                "percentile_2_5": float(lower),
                "percentile_50": float(median),
                "percentile_97_5": float(upper),
            })
    return {
        "status": "PASS",
        "classification": "PREDICTION_ONLY_PAIRED_CARD_CLUSTER_BOOTSTRAP",
        "retraining": False,
        "model_retraining": False,
        "replicates": int(replicates),
        "seed": int(seed),
        "cards": list(card_order),
        "card_count": len(card_order),
        "common_card_multiplicity_vectors": True,
        "multiplicity_sha256_int16_le": _sha256_bytes(
            np.ascontiguousarray(counts, dtype="<i2").tobytes()
        ),
        "ordered_population_sha256": baseline_hash,
        "interval": "PERCENTILE_95",
        "p_values_computed": False,
        "rows": rows,
    }


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a report table and prove an exact C/round-trip reopen."""

    data = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    reopened = pd.read_csv(
        io.BytesIO(data), engine="c", float_precision="round_trip",
    )
    if tuple(reopened.columns) != tuple(frame.columns) or len(reopened) != len(frame):
        raise Amendment06ReportingError("Report CSV round-trip shape/schema differs")
    for column in frame.columns:
        expected = frame[column]
        observed = reopened[column]
        expected_missing = pd.isna(expected).to_numpy()
        observed_missing = pd.isna(observed).to_numpy()
        if not np.array_equal(expected_missing, observed_missing):
            raise Amendment06ReportingError(
                f"Report CSV changed the missingness mask for {column}"
            )
        finite = ~expected_missing
        if pd.api.types.is_numeric_dtype(expected.dtype):
            expected_values = pd.to_numeric(expected[finite]).to_numpy(np.float64)
            observed_values = pd.to_numeric(observed[finite]).to_numpy(np.float64)
            if not np.array_equal(expected_values, observed_values):
                raise Amendment06ReportingError(
                    f"Report CSV changed a numeric value for {column}"
                )
        elif not np.array_equal(
            expected[finite].astype(str).to_numpy(),
            observed[finite].astype(str).to_numpy(),
        ):
            raise Amendment06ReportingError(
                f"Report CSV changed a text value for {column}"
            )
    return data


def _latex_bytes(frame: pd.DataFrame, caption: str, label: str) -> bytes:
    return (
        frame.to_latex(index=False, escape=True, caption=caption, label=label)
        .replace("NaN", "--")
        .encode("utf-8")
    )


def _figure_bytes(
    frame: pd.DataFrame,
    *,
    value: str,
    title: str,
    ylabel: str,
) -> bytes:
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    colors = ["#176B87", "#DAA520", "#5B6770", "#25855A", "#9B4F5D"]
    axis.bar(
        frame.variant.astype(str),
        frame[value].astype(float),
        color=[colors[index % len(colors)] for index in range(len(frame))],
    )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Variant")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    output = io.BytesIO()
    figure.savefig(output, format="png", dpi=160, metadata={"Software": "Amendment06"})
    plt.close(figure)
    return output.getvalue()


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def build_scientific_reports(
    run_dir: Path,
    *,
    metrics_by_variant: Mapping[str, Mapping[str, Any]],
    per_card_by_variant: Mapping[str, pd.DataFrame],
    bootstrap: Mapping[str, Any],
    stage_table: pd.DataFrame,
    timing_table: pd.DataFrame,
    parity_table: pd.DataFrame,
    r92_control: Mapping[str, Any],
) -> dict[str, Any]:
    """Build all table/figure/draft bytes in an empty attempt staging Run."""

    root = Path(run_dir)
    variants = (*OFFICIAL_RUNNABLE_VARIANTS, EXPLORATORY_VARIANT)
    if set(metrics_by_variant) != set(variants) or set(per_card_by_variant) != set(variants):
        raise Amendment06ReportingError("Reporting Variant set is not exact 9+1")
    if not (
        bootstrap.get("status") == "PASS"
        and bootstrap.get("classification")
        == "PREDICTION_ONLY_PAIRED_CARD_CLUSTER_BOOTSTRAP"
        and bootstrap.get("replicates") == BOOTSTRAP_REPLICATES
        and bootstrap.get("seed") == BOOTSTRAP_SEED
        and bootstrap.get("model_retraining") is False
        and bootstrap.get("common_card_multiplicity_vectors") is True
        and bootstrap.get("card_count") == 10
        and bootstrap.get("p_values_computed") is False
    ):
        raise Amendment06ReportingError("Bootstrap reporting contract differs")
    raw_rows: list[dict[str, Any]] = []
    weighted_rows: list[dict[str, Any]] = []
    for variant in variants:
        payload = metrics_by_variant[variant]
        raw = dict(payload.get("external_raw", {}))
        weighted = dict(payload.get("external_weighted_sensitivity", {}))
        for label, metric_payload in (("raw", raw), ("weighted", weighted)):
            missing = [name for name in PRIMARY_METRICS if name not in metric_payload]
            values = [metric_payload.get(name) for name in PRIMARY_METRICS]
            if missing or not np.isfinite(np.asarray(values, dtype=float)).all():
                raise Amendment06ReportingError(
                    f"Required {label} metrics are missing/non-finite for {variant}: {missing}"
                )
        classification = (
            "EXPLORATORY_PCA_DERIVED_FEATURE_BLOCK_REMOVAL"
            if variant == EXPLORATORY_VARIANT else "OFFICIAL_RUNNABLE"
        )
        common = {
            "variant": variant,
            "classification": classification,
            "feature_count": int(payload["feature_count"]),
            "best_iteration": int(payload["best_iteration"]),
        }
        raw_rows.append({**common, **raw})
        weighted_rows.append({**common, **weighted})
    raw_frame = pd.DataFrame(raw_rows)
    weighted_frame = pd.DataFrame(weighted_rows)
    full = raw_frame.set_index("variant").loc[BASELINE_VARIANT]
    delta_rows: list[dict[str, Any]] = []
    for row in raw_frame.to_dict("records"):
        delta_rows.append({
            "variant": row["variant"],
            "classification": row["classification"],
            "delta_definition": "variant_minus_full",
            **{
                f"delta_{metric}": float(row[metric]) - float(full[metric])
                for metric in PRIMARY_METRICS
            },
        })
    delta_frame = pd.DataFrame(delta_rows)
    bootstrap_frame = pd.DataFrame(list(bootstrap.get("rows", [])))
    if len(bootstrap_frame) != len(variants) * len(BOOTSTRAP_METRICS):
        raise Amendment06ReportingError("Bootstrap reporting rows are incomplete")
    expected_bootstrap_pairs = [
        (variant, metric)
        for variant in variants
        for metric in BOOTSTRAP_METRICS
    ]
    if list(zip(bootstrap_frame.variant, bootstrap_frame.metric)) != expected_bootstrap_pairs:
        raise Amendment06ReportingError("Bootstrap reporting row order/set differs")
    bootstrap_numeric = bootstrap_frame.loc[:, [
        "delta_point_estimate", "percentile_2_5", "percentile_50",
        "percentile_97_5",
    ]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if not np.isfinite(bootstrap_numeric).all():
        raise Amendment06ReportingError("Bootstrap reporting values are non-finite")
    cards = pd.concat(
        [per_card_by_variant[variant].copy() for variant in variants],
        ignore_index=True,
    )
    blocked = pd.DataFrame([{
        "variant": BLOCKED_VARIANT,
        "classification": "OFFICIAL_PRESPECIFIED_BLOCKED",
        "status": "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        "model_created": False,
        "metric_created": False,
    }])
    official_mask = raw_frame.variant.isin(OFFICIAL_RUNNABLE_VARIANTS)
    exploratory_mask = raw_frame.variant.eq(EXPLORATORY_VARIANT)
    bootstrap_official = bootstrap_frame.loc[
        bootstrap_frame.variant.isin(OFFICIAL_RUNNABLE_VARIANTS)
    ].reset_index(drop=True)
    bootstrap_exploratory = bootstrap_frame.loc[
        bootstrap_frame.variant.eq(EXPLORATORY_VARIANT)
    ].reset_index(drop=True)

    artifacts: dict[str, bytes] = {
        "tables/primary_metrics_official.csv": _csv_bytes(raw_frame.loc[official_mask]),
        "tables/primary_metrics_exploratory.csv": _csv_bytes(raw_frame.loc[exploratory_mask]),
        "tables/review_weighted_sensitivity_official.csv": _csv_bytes(
            weighted_frame.loc[weighted_frame.variant.isin(OFFICIAL_RUNNABLE_VARIANTS)]
        ),
        "tables/review_weighted_sensitivity_exploratory.csv": _csv_bytes(
            weighted_frame.loc[weighted_frame.variant.eq(EXPLORATORY_VARIANT)]
        ),
        "tables/variant_minus_full_official.csv": _csv_bytes(
            delta_frame.loc[delta_frame.variant.isin(OFFICIAL_RUNNABLE_VARIANTS)]
        ),
        "tables/variant_minus_full_exploratory.csv": _csv_bytes(
            delta_frame.loc[delta_frame.variant.eq(EXPLORATORY_VARIANT)]
        ),
        "tables/paired_bootstrap_ci_official.csv": _csv_bytes(bootstrap_official),
        "tables/paired_bootstrap_ci_exploratory.csv": _csv_bytes(bootstrap_exploratory),
        "tables/per_card_metrics_official.csv": _csv_bytes(
            cards.loc[cards.variant.isin(OFFICIAL_RUNNABLE_VARIANTS)]
        ),
        "tables/per_card_metrics_exploratory.csv": _csv_bytes(
            cards.loc[cards.variant.eq(EXPLORATORY_VARIANT)]
        ),
        "tables/official_blocked_variants.csv": _csv_bytes(blocked),
        "tables/class_balance_and_augmentation_stages.csv": _csv_bytes(stage_table),
        "tables/training_resource_timing.csv": _csv_bytes(timing_table),
        "tables/feature_and_resampling_parity.csv": _csv_bytes(parity_table),
        "tables/r92_historical_control.csv": _csv_bytes(pd.DataFrame([dict(r92_control)])),
        "tables/primary_metrics_official.tex": _latex_bytes(
            raw_frame.loc[official_mask, ["variant", *PRIMARY_METRICS]],
            "Official Amendment 06 external-Audit metrics.",
            "tab:a06-primary",
        ),
        "tables/paired_bootstrap_ci_official.tex": _latex_bytes(
            bootstrap_official,
            "Paired card-cluster bootstrap intervals for variant-minus-full deltas.",
            "tab:a06-bootstrap",
        ),
        "figures/official_average_precision.png": _figure_bytes(
            raw_frame.loc[official_mask],
            value="average_precision",
            title="Official runnable Variants: external-Audit Average Precision",
            ylabel="Average Precision",
        ),
        "figures/official_delta_average_precision.png": _figure_bytes(
            delta_frame.loc[delta_frame.variant.isin(OFFICIAL_RUNNABLE_VARIANTS)],
            value="delta_average_precision",
            title="Official variant-minus-full AP deltas",
            ylabel="Delta Average Precision",
        ),
        "figures/exploratory_average_precision.png": _figure_bytes(
            raw_frame.loc[exploratory_mask],
            value="average_precision",
            title="Exploratory PCA-derived feature-block removal",
            ylabel="Average Precision",
        ),
    }
    best = raw_frame.loc[official_mask].sort_values(
        "average_precision", ascending=False, kind="stable"
    ).iloc[0]
    common_disclosure = (
        "This retrospective ablation uses the 363,563-row historical fixed feature "
        "table, the locked 291,024-row training and 56,843-row validation Split, "
        "one shared 93D paired-resampling realization, and a ten-card external Audit. "
        "True no-PCA is blocked because aligned raw embeddings are unavailable. "
        "PCA/prototype lineage remains an accepted unresolved fixed-table limitation."
    )
    artifacts.update({
        "docs/METHODS_DRAFT.md": (
            "# Methods Draft\n\n" + common_disclosure
            + "\n\nAll models used XGBoost 2.1.1 on CUDA, Seed 42, early stopping "
            "only on the locked validation population, and a fixed threshold of 0.5. "
            "External-Audit labels were opened only after all ten models were frozen. "
            "Uncertainty uses 10,000 paired card-cluster bootstrap replicates with "
            "Seed 2026081201 and no refitting.\n"
        ).encode("utf-8"),
        "docs/RESULTS_DRAFT.md": (
            "# Results Draft\n\n" + common_disclosure
            + f"\n\nThe highest observed official candidate-level AP was {float(best.average_precision)!r} "
            f"for `{best.variant}`. Deltas are always variant-minus-full. Exploratory "
            "`no_deep_pca_features` results are reported separately and are not true no-PCA. "
            "Intervals are descriptive; no p-values or statistical-significance claims are made.\n"
        ).encode("utf-8"),
        "docs/REVIEWER_RESPONSE_DRAFT.md": (
            "# Reviewer Response Draft\n\n" + common_disclosure
            + "\n\nWe report nine official runnable ablations, one prespecified blocked "
            "true no-PCA outcome, and one separately labeled exploratory PCA-derived "
            "feature-block removal. Models, reload parity, exact prediction tables, and "
            "paired resampling evidence are supplied for external review.\n"
        ).encode("utf-8"),
        "docs/REPRODUCE.md": (
            "# Reproduce\n\n" + common_disclosure
            + "\n\nUse the sealed Run identity, configuration, Split, source hashes, "
            "scientific code manifest, frozen resampling manifest, and completion manifests. "
            "The accepted CUDA environment is XGBoost 2.1.1 with Seed 42. Package-only "
            "retry must use the same frozen Run and must not call scientific APIs.\n"
        ).encode("utf-8"),
        "docs/WARNINGS_AND_LIMITATIONS.md": (
            "# Warnings And Limitations\n\n" + common_disclosure
            + "\n\nOnly ten external cards are available. Bootstrap intervals are descriptive, "
            "not significance tests. This design does not establish fold-safe PCA/prototype "
            "transforms and does not compare raw embeddings against PCA features.\n"
        ).encode("utf-8"),
    })
    written: list[dict[str, Any]] = []
    for relative, data in sorted(artifacts.items()):
        path = root / relative
        _write_new(path, data)
        written.append({
            "relative_path": relative,
            "size_bytes": len(data),
            "sha256": _sha256_bytes(data),
        })
    return {
        "status": "PASS",
        "artifact_count": len(written),
        "artifacts": written,
        "official_runnable_count": len(OFFICIAL_RUNNABLE_VARIANTS),
        "official_blocked_true_no_pca_count": 1,
        "exploratory_count": 1,
        "delta_definition": "variant_minus_full",
        "strong_inference_claimed": False,
        "csv_roundtrip": {
            "status": "PASS",
            "engine": "c",
            "float_precision": "round_trip",
            "numeric_equality": "EXACT",
            "validated_csv_count": sum(
                1 for relative in artifacts if relative.endswith(".csv")
            ),
        },
    }


build_all_reports = build_scientific_reports


__all__ = [
    "Amendment06ReportingError",
    "BLOCKED_VARIANT",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "DECISION_THRESHOLD",
    "EXPLORATORY_VARIANT",
    "OFFICIAL_RUNNABLE_VARIANTS",
    "PREDICTION_COLUMNS",
    "binary_metrics",
    "build_scientific_reports",
    "build_all_reports",
    "deterministic_gzip_bytes",
    "exact_prediction_csv_bytes",
    "paired_card_bootstrap",
    "prediction_csv_bytes",
    "per_card_metrics",
    "validate_prediction_csv_roundtrip",
]
