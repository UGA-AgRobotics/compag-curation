from __future__ import annotations

import gzip
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


STUDY = Path(__file__).resolve().parents[1]
CODE = STUDY / "code"
sys.path.insert(0, str(CODE))
os.environ.setdefault("ABLATION_STUDY_ROOT", str(STUDY))

import amendment06_reporting as reporting  # noqa: E402


def _predictions(variant: str, *, shift: float = 0.0) -> pd.DataFrame:
    rows = []
    order = 0
    for card_index in range(10):
        for offset, label in enumerate((0, 1, 0, 1)):
            probability = (0.16 + 0.62 * label + 0.005 * card_index + shift)
            probability = float(min(0.999, max(0.001, probability)))
            rows.append({
                "stable_candidate_id": f"card-{card_index:02d}-row-{offset}",
                "card_id": f"IMG_{9500 + card_index}",
                "true_label": label,
                "probability": probability,
                "prediction_at_0_5": int(probability >= 0.5),
                "review_weight": float(0.7 + 0.1 * (offset % 2)),
                "variant_id": variant,
                "split_population": "external_audit",
                "row_order": order,
            })
            order += 1
    return pd.DataFrame(rows, columns=reporting.PREDICTION_COLUMNS)


def _all_predictions() -> dict[str, pd.DataFrame]:
    variants = (*reporting.OFFICIAL_RUNNABLE_VARIANTS, reporting.EXPLORATORY_VARIANT)
    return {
        variant: _predictions(variant, shift=-0.002 * index)
        for index, variant in enumerate(variants)
    }


def test_binary_metrics_are_finite_and_fixed_threshold():
    frame = _predictions("full_new_reference")
    result = reporting.binary_metrics(
        frame.true_label,
        frame.probability,
        threshold=0.5,
    )
    assert result["threshold"] == 0.5
    assert result["rows"] == 40
    assert result["tn"] == 20 and result["tp"] == 20
    assert all(np.isfinite(float(result[key])) for key in reporting.PRIMARY_METRICS)


@pytest.mark.parametrize(
    "labels,probability,weights",
    [
        ([0, 1], [0.1, float("nan")], None),
        ([0, 1], [0.1, 0.9], [1.0, float("inf")]),
        ([0, 2], [0.1, 0.9], None),
        ([1, 1], [0.1, 0.9], None),
    ],
)
def test_required_metric_inputs_fail_closed(labels, probability, weights):
    with pytest.raises(reporting.Amendment06ReportingError):
        reporting.binary_metrics(labels, probability, weights=weights)


def test_prediction_csv_is_deterministic_and_exact_round_trip(tmp_path):
    frame = _predictions("full_new_reference")
    frame.loc[0, "probability"] = np.nextafter(0.12345678901234568, 1.0)
    raw = reporting.prediction_csv_bytes(frame)
    assert raw == reporting.prediction_csv_bytes(frame)
    compressed = reporting.deterministic_gzip_bytes(raw)
    assert compressed == reporting.deterministic_gzip_bytes(raw)
    path = tmp_path / "prediction.csv.gz"
    path.write_bytes(compressed)
    metrics = reporting.binary_metrics(frame.true_label, frame.probability)
    evidence = reporting.validate_prediction_csv_roundtrip(
        path,
        frame,
        metrics=metrics,
    )
    assert evidence["status"] == "PASS"
    assert evidence["numeric_equality"] == "EXACT"
    assert gzip.decompress(compressed) == raw


def test_prediction_csv_promotes_float32_exactly_without_mutating_source():
    frame = _predictions("full_new_reference")
    frame["probability"] = frame.probability.astype(np.float32)
    frame["review_weight"] = frame.review_weight.astype(np.float32)
    original_probability = frame.probability.copy()
    original_dtypes = frame.dtypes.copy()
    raw = reporting.prediction_csv_bytes(frame)
    evidence = reporting.validate_prediction_csv_roundtrip(raw, frame)
    assert evidence["numeric_equality"] == "EXACT"
    assert frame.dtypes.equals(original_dtypes)
    assert frame.probability.equals(original_probability)


def test_prediction_csv_one_ulp_tamper_fails():
    frame = _predictions("full_new_reference")
    raw = reporting.prediction_csv_bytes(frame)
    tampered = raw.replace(b"0.16,0,", b"0.16000000000000003,0,", 1)
    assert tampered != raw
    with pytest.raises(reporting.Amendment06ReportingError, match="probability"):
        reporting.validate_prediction_csv_roundtrip(tampered, frame)


def test_prediction_schema_duplicate_and_order_tamper_fail():
    frame = _predictions("full_new_reference")
    with pytest.raises(reporting.Amendment06ReportingError, match="row_order"):
        reporting.prediction_csv_bytes(frame.iloc[::-1].reset_index(drop=True))
    duplicate = frame.copy()
    duplicate.loc[1, "stable_candidate_id"] = duplicate.loc[0, "stable_candidate_id"]
    with pytest.raises(reporting.Amendment06ReportingError, match="duplicated"):
        reporting.prediction_csv_bytes(duplicate)


@pytest.mark.parametrize("column,value", [("probability", 1.01), ("review_weight", 0.0)])
def test_prediction_probability_and_weight_domains_fail(column, value):
    frame = _predictions("full_new_reference")
    frame.loc[0, column] = value
    with pytest.raises(reporting.Amendment06ReportingError, match="threshold/weight"):
        reporting.prediction_csv_bytes(frame)


def test_paired_bootstrap_is_deterministic_common_and_separated():
    predictions = _all_predictions()
    cards = sorted(predictions[reporting.BASELINE_VARIANT].card_id.unique())
    first = reporting.paired_card_bootstrap(
        predictions,
        cards,
        replicates=reporting.BOOTSTRAP_REPLICATES,
        seed=reporting.BOOTSTRAP_SEED,
    )
    second = reporting.paired_card_bootstrap(
        predictions,
        cards,
        replicates=reporting.BOOTSTRAP_REPLICATES,
        seed=reporting.BOOTSTRAP_SEED,
    )
    assert first == second
    assert first["retraining"] is False
    assert first["model_retraining"] is False
    assert first["common_card_multiplicity_vectors"] is True
    assert len(first["rows"]) == 30
    assert [
        (row["variant"], row["metric"]) for row in first["rows"]
    ] == [
        (variant, metric)
        for variant in (*reporting.OFFICIAL_RUNNABLE_VARIANTS, reporting.EXPLORATORY_VARIANT)
        for metric in ("average_precision", "f1", "recall")
    ]
    official = [row for row in first["rows"] if row["classification"] == "OFFICIAL_RUNNABLE"]
    exploratory = [row for row in first["rows"] if row["classification"].startswith("EXPLORATORY")]
    assert len(official) == 27 and len(exploratory) == 3
    assert all(row["delta_definition"] == "variant_minus_full" for row in first["rows"])


def test_paired_bootstrap_rejects_population_or_card_drift():
    predictions = _all_predictions()
    cards = sorted(predictions[reporting.BASELINE_VARIANT].card_id.unique())
    predictions["no_color"] = predictions["no_color"].iloc[:-1].copy()
    with pytest.raises(reporting.Amendment06ReportingError, match="population/order"):
        reporting.paired_card_bootstrap(
            predictions,
            cards,
            replicates=reporting.BOOTSTRAP_REPLICATES,
            seed=reporting.BOOTSTRAP_SEED,
        )


def test_build_scientific_reports_is_complete_and_separates_exploratory(tmp_path):
    predictions = _all_predictions()
    cards = sorted(predictions[reporting.BASELINE_VARIANT].card_id.unique())
    bootstrap = reporting.paired_card_bootstrap(
        predictions,
        cards,
        replicates=reporting.BOOTSTRAP_REPLICATES,
        seed=reporting.BOOTSTRAP_SEED,
    )
    metrics = {}
    per_card = {}
    for index, (variant, frame) in enumerate(predictions.items()):
        metrics[variant] = {
            "external_raw": reporting.binary_metrics(frame.true_label, frame.probability),
            "external_weighted_sensitivity": reporting.binary_metrics(
                frame.true_label,
                frame.probability,
                weights=frame.review_weight,
            ),
            "feature_count": 93 - index,
            "best_iteration": 12 + index,
        }
        per_card[variant] = reporting.per_card_metrics(frame)
    result = reporting.build_all_reports(
        tmp_path,
        metrics_by_variant=metrics,
        per_card_by_variant=per_card,
        bootstrap=bootstrap,
        stage_table=pd.DataFrame([{"stage": "post_safe_smote", "rows": 1457449}]),
        timing_table=pd.DataFrame([{"variant": "full_new_reference", "wall_seconds": 1.0}]),
        parity_table=pd.DataFrame([{"variant": "full_new_reference", "status": "PASS"}]),
        r92_control={"status": "PASS", "classification": "HISTORICAL_CONTROL"},
    )
    assert result["status"] == "PASS"
    assert result["official_runnable_count"] == 9
    assert result["exploratory_count"] == 1
    assert result["csv_roundtrip"]["status"] == "PASS"
    assert result["csv_roundtrip"]["numeric_equality"] == "EXACT"
    expected = {
        "tables/primary_metrics_official.csv",
        "tables/primary_metrics_exploratory.csv",
        "tables/official_blocked_variants.csv",
        "tables/primary_metrics_official.tex",
        "figures/official_average_precision.png",
        "figures/exploratory_average_precision.png",
        "docs/METHODS_DRAFT.md",
        "docs/RESULTS_DRAFT.md",
        "docs/REVIEWER_RESPONSE_DRAFT.md",
        "docs/REPRODUCE.md",
        "docs/WARNINGS_AND_LIMITATIONS.md",
    }
    assert expected <= {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    official = pd.read_csv(tmp_path / "tables/primary_metrics_official.csv")
    exploratory = pd.read_csv(tmp_path / "tables/primary_metrics_exploratory.csv")
    blocked = pd.read_csv(tmp_path / "tables/official_blocked_variants.csv")
    assert official.variant.tolist() == list(reporting.OFFICIAL_RUNNABLE_VARIANTS)
    assert exploratory.variant.tolist() == [reporting.EXPLORATORY_VARIANT]
    assert blocked.to_dict("records")[0]["status"] == "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE"
    docs = "\n".join(path.read_text() for path in (tmp_path / "docs").glob("*.md"))
    assert "363,563-row" in docs
    assert "291,024-row" in docs and "56,843-row" in docs
    assert "ten-card external Audit" in docs
    assert "511,082" not in docs and "70:30" not in docs
    for path in (tmp_path / "docs").glob("*.md"):
        text = path.read_text()
        assert "363,563-row" in text
        assert "291,024-row" in text and "56,843-row" in text
        assert "ten-card external Audit" in text
        assert "True no-PCA is blocked" in text
        assert "PCA/prototype" in text
    assert all(path.stat().st_size > 1_000 for path in (tmp_path / "figures").glob("*.png"))
