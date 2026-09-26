from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "code"))

import amendment02_resampling as paired  # noqa: E402
import amendment_core as core  # noqa: E402


def _text_sequence_hash(values) -> str:
    payload = ("\n".join(map(str, values)) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _full_space_inputs(rows: int = 40):
    features = core.get_r92_feature_order()
    base = np.linspace(-2.0, 3.0, rows, dtype=np.float32)
    frame = pd.DataFrame({
        feature: base * np.float32(1.0 + index / 100.0) + np.float32(index)
        for index, feature in enumerate(features)
    })
    frame.loc[3, features[0]] = np.nan
    labels = np.asarray([0] * 32 + [1] * 8, dtype=np.int8)
    weights = np.linspace(0.4, 1.0, rows, dtype=np.float32)
    source_ids = [f"locked_row_{index:03d}" for index in range(rows)]
    return frame, labels, weights, source_ids, features


def _frozen_population():
    frame, labels, weights, source_ids, features = _full_space_inputs()
    return paired.build_frozen_resampling_population(
        frame,
        labels,
        weights,
        source_ids,
        full_features=features,
    )


def test_frozen_generation_calls_augmentation_and_smote_once():
    frame, labels, weights, source_ids, features = _full_space_inputs()
    calls = {"augmentation": 0, "smote": 0}
    real_augmentation = core.augment_with_lineage

    def counted_augmentation(*args, **kwargs):
        calls["augmentation"] += 1
        return real_augmentation(*args, **kwargs)

    class CountedSafeSMOTE:
        def __init__(self, **kwargs):
            self.sampler = core.legacy.SafeSMOTE(**kwargs)

        def fit_resample(self, X, y):
            calls["smote"] += 1
            return self.sampler.fit_resample(X, y)

    frozen = paired.build_frozen_resampling_population(
        frame,
        labels,
        weights,
        source_ids,
        full_features=features,
        augment_fn=counted_augmentation,
        smote_factory=CountedSafeSMOTE,
    )
    projections = paired.project_all_variants(frozen)

    assert calls == {"augmentation": 1, "smote": 1}
    assert len(projections) == 10
    assert frozen.augmentation_generation_count == 1
    assert frozen.safe_smote_generation_count == 1


def test_module_import_does_not_eagerly_import_amendment_core(tmp_path):
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(STUDY / 'code')!r}); "
        "assert 'amendment_core' not in sys.modules; "
        "import amendment02_resampling; "
        "assert 'amendment_core' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_feature_variants_project_one_post_smote_population():
    frozen = _frozen_population()
    projections = paired.project_all_variants(frozen)

    for variant, projection in projections.items():
        if variant == "no_safe_smote":
            continue
        assert np.array_equal(projection.y, frozen.post_smote_y), variant
        assert np.array_equal(projection.lineage, frozen.post_smote_lineage), variant
        assert np.array_equal(
            projection.X,
            frozen.post_smote_X[:, projection.feature_indices],
        ), variant
        assert projection.evidence["projection_only"] is True
        assert projection.evidence["source_stage"] == "post_safe_smote"


def test_full_and_no_weight_are_identical_except_sample_weights():
    frozen = _frozen_population()
    full = paired.project_variant_population(frozen, "full_new_reference")
    no_weight = paired.project_variant_population(
        frozen, "no_review_aware_training_weights",
    )

    assert np.array_equal(full.X, no_weight.X)
    assert np.array_equal(full.y, no_weight.y)
    assert np.array_equal(full.lineage, no_weight.lineage)
    assert np.array_equal(full.weights, frozen.post_smote_weights)
    assert np.array_equal(no_weight.weights, np.ones(len(no_weight.y), np.float32))
    assert not np.array_equal(full.weights, no_weight.weights)


def test_no_safe_smote_uses_frozen_pre_smote_and_retains_augmentation():
    frozen = _frozen_population()
    no_smote = paired.project_variant_population(frozen, "no_safe_smote")
    by_stage = {stage["stage"]: stage for stage in frozen.stages}

    assert np.array_equal(no_smote.X, frozen.pre_smote_X)
    assert np.array_equal(no_smote.y, frozen.pre_smote_y)
    assert np.array_equal(no_smote.weights, frozen.pre_smote_weights)
    assert np.array_equal(no_smote.lineage, frozen.pre_smote_lineage)
    assert no_smote.evidence["safe_smote_synthetic_rows"] == 0
    assert by_stage["post_mixup"]["rows"] > by_stage[
        "raw_locked_post_split_shuffle"
    ]["rows"]
    assert by_stage["post_dropout"]["rows"] > by_stage["post_mixup"]["rows"]
    assert by_stage["post_jitter_pre_smote"]["rows"] > by_stage[
        "post_dropout"
    ]["rows"]


def test_full_space_imputer_fits_once_then_validation_is_projected(monkeypatch):
    frame, labels, weights, source_ids, features = _full_space_inputs()
    calls = {"fit_transform": 0}

    class TrackingImputer(SimpleImputer):
        def fit_transform(self, X, y=None, **fit_params):
            calls["fit_transform"] += 1
            return super().fit_transform(X, y, **fit_params)

    monkeypatch.setattr(core, "SimpleImputer", TrackingImputer)
    frozen = paired.build_frozen_resampling_population(
        frame, labels, weights, source_ids, full_features=features,
    )
    validation = paired.transform_validation_full_space(frozen, frame.iloc[:7])
    projections = paired.project_all_variants(frozen)

    for projection in projections.values():
        projected = paired.project_validation_population(validation, projection)
        assert np.array_equal(
            projected, validation.X[:, projection.feature_indices],
        )
    assert calls["fit_transform"] == 1


def test_parity_evidence_contains_all_required_hashes_and_comparisons():
    frozen = _frozen_population()
    projections = paired.project_all_variants(frozen)
    evidence = paired.frozen_resampling_parity(frozen, projections)

    assert evidence["status"] == "PASS"
    assert evidence["generation"] == {
        "scope": "ONE_FULL_SPACE_POPULATION",
        "seed": 42,
        "augmentation_generation_count": 1,
        "safe_smote_generation_count": 1,
        "full_feature_count": 93,
        "full_feature_order_sha256": frozen.feature_order_sha256,
    }
    for population in (evidence["pre_smote"], evidence["post_smote"]):
        for key in (
            "X_sha256", "y_sha256", "weight_sha256",
            "source_lineage_sha256", "row_order_sha256",
            "feature_order_sha256",
        ):
            assert len(population[key]) == 64

    expected_populations = (
        (
            evidence["pre_smote"], frozen.pre_smote_X, frozen.pre_smote_y,
            frozen.pre_smote_weights, frozen.pre_smote_lineage,
        ),
        (
            evidence["post_smote"], frozen.post_smote_X, frozen.post_smote_y,
            frozen.post_smote_weights, frozen.post_smote_lineage,
        ),
    )
    for observed, X, y, weights, lineage in expected_populations:
        assert observed["X_sha256"] == core.sha256_array(
            np.asarray(X, dtype="<f4")
        )
        assert observed["y_sha256"] == core.sha256_array(
            np.asarray(y, dtype=np.int8)
        )
        assert observed["weight_sha256"] == core.sha256_array(
            np.asarray(weights, dtype="<f4")
        )
        assert observed["source_lineage_sha256"] == _text_sequence_hash(lineage)
        assert observed["row_order_sha256"] == _text_sequence_hash(lineage)
        assert observed["feature_order_sha256"] == _text_sequence_hash(
            frozen.feature_names
        )

    variant_evidence = evidence["variants"]
    full = variant_evidence["full_new_reference"]
    no_weight = variant_evidence["no_review_aware_training_weights"]
    for key in ("X_sha256", "y_sha256", "source_lineage_sha256", "row_order_sha256"):
        assert full[key] == no_weight[key]
    assert full["weight_sha256"] != no_weight["weight_sha256"]
    no_smote = variant_evidence["no_safe_smote"]
    for key in (
        "X_sha256", "y_sha256", "weight_sha256",
        "source_lineage_sha256", "row_order_sha256",
    ):
        assert no_smote[key] == evidence["pre_smote"][key]
    assert all(evidence["comparisons"]["full_vs_no_weight"].values())
    assert all(
        evidence["comparisons"]["full_vs_no_safe_smote_pre_smote"].values()
    )
    expected_post = set(projections) - {"no_safe_smote"}
    observed_post = set(
        evidence["comparisons"]["post_smote_variant_projections"]
    )
    assert observed_post == expected_post
    assert all(
        value
        for checks in evidence["comparisons"][
            "post_smote_variant_projections"
        ].values()
        for value in checks.values()
    )


def test_frozen_arrays_are_read_only():
    frozen = _frozen_population()
    projection = paired.project_variant_population(frozen, "no_color")

    for array in (
        frozen.pre_smote_X,
        frozen.pre_smote_y,
        frozen.pre_smote_weights,
        frozen.pre_smote_lineage,
        frozen.post_smote_X,
        frozen.post_smote_y,
        frozen.post_smote_weights,
        frozen.post_smote_lineage,
        projection.X,
    ):
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        projection.X[0, 0] = 0.0


def test_projection_rejects_blocked_no_pca_and_feature_order_drift():
    frozen = _frozen_population()
    no_color = list(paired.canonical_variant_features("no_color"))

    with pytest.raises(paired.PairedResamplingError, match="not projectable"):
        paired.project_variant_population(frozen, "no_pca")
    with pytest.raises(paired.PairedResamplingError, match="feature order differs"):
        paired.project_variant_population(
            frozen, "no_color", list(reversed(no_color)),
        )


def test_dependency_closure_and_exploratory_pca_status_remain_locked():
    variants = core.build_variant_feature_sets()

    assert variants["no_pca"]["status"] == (
        "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE"
    )
    assert variants[core.EXPLORATORY_VARIANT]["kind"] == (
        "EXPLORATORY_PCA_DERIVED_FEATURE_BLOCK_REMOVAL"
    )
    assert variants[core.EXPLORATORY_VARIANT]["status"] == "RUNNABLE"
    assert "g_quality" not in variants["no_color"]["retained_features"]
    assert "g_quality" not in variants["no_shape"]["retained_features"]
    assert "g_robust" not in variants["no_shape"]["retained_features"]
    assert "g_robust" not in variants["no_texture"]["retained_features"]
