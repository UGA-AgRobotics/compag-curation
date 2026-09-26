"""Paired full-space resampling for Protocol Amendment 02.

This module is intentionally independent of the current Smoke runner.  It builds
one immutable augmented population in the complete r92 feature space and exposes
only column projections for the prespecified Variants.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


DEFAULT_SEED = 42
EXPLORATORY_VARIANT = "no_deep_pca_features"
EXPECTED_AUGMENTATION_STAGES = (
    "raw_locked_post_split_shuffle",
    "post_mixup",
    "post_dropout",
    "post_jitter_pre_smote",
    "post_shuffle",
    "post_safe_smote",
)


def _load_backend() -> Any:
    """Resolve the legacy helpers only when a public operation needs them."""

    return importlib.import_module("amendment_core")


class PairedResamplingError(RuntimeError):
    """The frozen paired-resampling contract is invalid."""


def _freeze_array(values: np.ndarray, dtype: Any | None = None) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _feature_order_hash(features: Sequence[str]) -> str:
    payload = ("\n".join(map(str, features)) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _lineage_hash(values: np.ndarray) -> str:
    payload = ("\n".join(map(str, values.tolist())) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _population_evidence(
    *,
    stage: str,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    lineage: np.ndarray,
    features: Sequence[str],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "rows": int(len(y)),
        "features": int(X.shape[1]),
        "negative": int((y == 0).sum()),
        "positive": int((y == 1).sum()),
        "feature_order_sha256": _feature_order_hash(features),
        "X_sha256": _array_hash(np.asarray(X, dtype="<f4")),
        "y_sha256": _array_hash(np.asarray(y, dtype=np.int8)),
        "weight_sha256": _array_hash(np.asarray(weights, dtype="<f4")),
        "source_lineage_sha256": _lineage_hash(lineage),
        "row_order_sha256": _lineage_hash(lineage),
    }


def _validated_labels(values: Sequence[int], rows: int) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if len(numeric) != rows or numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise PairedResamplingError(
            "Labels must contain one non-null binary value per training row"
        )
    return numeric.to_numpy(dtype=np.int8)


def _validated_weights(values: Sequence[float], rows: int) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    if (
        len(numeric) != rows
        or not np.isfinite(numeric).all()
        or np.any(numeric <= 0)
    ):
        raise PairedResamplingError(
            "Review weights must contain one finite positive value per training row"
        )
    return numeric.astype(np.float32)


def _validated_source_ids(values: Sequence[str], rows: int) -> np.ndarray:
    lineage = np.asarray(list(map(str, values)), dtype=object)
    if (
        len(lineage) != rows
        or any(not value for value in lineage)
        or len(set(lineage.tolist())) != rows
    ):
        raise PairedResamplingError(
            "Source lineage must contain one unique nonempty ID per training row"
        )
    return lineage


@dataclass(frozen=True)
class FrozenResamplingPopulation:
    """One full-space augmentation/SMOTE realization shared by all Variants."""

    feature_names: tuple[str, ...]
    feature_order_sha256: str
    seed: int
    imputer: SimpleImputer
    pre_smote_X: np.ndarray
    pre_smote_y: np.ndarray
    pre_smote_weights: np.ndarray
    pre_smote_lineage: np.ndarray
    post_smote_X: np.ndarray
    post_smote_y: np.ndarray
    post_smote_weights: np.ndarray
    post_smote_lineage: np.ndarray
    stages: tuple[Mapping[str, Any], ...]
    augmentation_generation_count: int = 1
    safe_smote_generation_count: int = 1

    @property
    def synthetic_smote_rows(self) -> int:
        return int(len(self.post_smote_y) - len(self.pre_smote_y))


@dataclass(frozen=True)
class VariantProjection:
    """A read-only Variant view derived from a frozen full-space population."""

    variant: str
    source_stage: str
    feature_names: tuple[str, ...]
    feature_indices: tuple[int, ...]
    X: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    lineage: np.ndarray
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenValidationPopulation:
    """One full-space imputer transform reusable for every validation Variant."""

    feature_names: tuple[str, ...]
    feature_order_sha256: str
    X: np.ndarray


def build_frozen_resampling_population(
    training_frame: pd.DataFrame,
    labels: Sequence[int],
    review_weights: Sequence[float],
    source_ids: Sequence[str],
    *,
    full_features: Sequence[str] | None = None,
    seed: int = DEFAULT_SEED,
    backend: Any | None = None,
    augment_fn: Callable[..., Any] | None = None,
    smote_factory: Callable[..., Any] | None = None,
) -> FrozenResamplingPopulation:
    """Generate augmentation and Safe-SMOTE exactly once in the full 93 features."""

    backend = backend or _load_backend()
    augment_fn = augment_fn or backend.augment_with_lineage
    smote_factory = smote_factory or backend.legacy.SafeSMOTE
    expected = tuple(backend.get_r92_feature_order())
    feature_names = tuple(full_features or expected)
    if feature_names != expected or len(feature_names) != 93:
        raise PairedResamplingError(
            "Frozen resampling requires the exact ordered 93-feature r92 representation"
        )
    missing = [feature for feature in feature_names if feature not in training_frame]
    if missing:
        raise PairedResamplingError(f"Full-space training frame is missing: {missing}")

    rows = len(training_frame)
    y = _validated_labels(labels, rows)
    weights = _validated_weights(review_weights, rows)
    lineage = _validated_source_ids(source_ids, rows)
    full_frame = training_frame.loc[:, feature_names]

    (
        pre_X,
        pre_y,
        pre_weights,
        pre_lineage,
        imputer,
        augmentation_stages,
    ) = augment_fn(
        full_frame,
        y,
        weights,
        lineage,
        seed=seed,
        smote_enabled=False,
    )
    if [stage["stage"] for stage in augmentation_stages] != list(
        EXPECTED_AUGMENTATION_STAGES
    ):
        raise PairedResamplingError("Unexpected augmentation stage sequence")

    sampler = smote_factory(
        sampling_strategy=0.5,
        k_neighbors=3,
        random_state=seed,
    )
    post_X, post_y = sampler.fit_resample(pre_X, pre_y)
    post_X = np.asarray(post_X, dtype=np.float32)
    post_y = np.asarray(post_y, dtype=np.int8)
    synthetic_rows = int(len(post_y) - len(pre_y))
    if synthetic_rows < 0 or post_X.shape != (len(post_y), len(feature_names)):
        raise PairedResamplingError("Safe-SMOTE returned an invalid full-space population")

    post_weights = np.asarray(pre_weights, dtype=np.float32)
    post_lineage = np.asarray(pre_lineage, dtype=object)
    if synthetic_rows:
        synthetic_labels = post_y[len(pre_y):]
        synthetic_weights = np.asarray(
            [pre_weights[pre_y == label].mean() for label in synthetic_labels],
            dtype=np.float32,
        )
        post_weights = np.concatenate([post_weights, synthetic_weights])
        synthetic_ids = np.asarray(
            [
                "smote_"
                + hashlib.sha256(
                    f"{seed}|{offset}|{int(label)}".encode()
                ).hexdigest()
                for offset, label in enumerate(synthetic_labels)
            ],
            dtype=object,
        )
        post_lineage = np.concatenate([post_lineage, synthetic_ids])
    if not (len(post_y) == len(post_weights) == len(post_lineage)):
        raise PairedResamplingError("Frozen post-SMOTE arrays have inconsistent lengths")

    pre_X = _freeze_array(pre_X, np.float32)
    pre_y = _freeze_array(pre_y, np.int8)
    pre_weights = _freeze_array(pre_weights, np.float32)
    pre_lineage = _freeze_array(pre_lineage, object)
    post_X = _freeze_array(post_X, np.float32)
    post_y = _freeze_array(post_y, np.int8)
    post_weights = _freeze_array(post_weights, np.float32)
    post_lineage = _freeze_array(post_lineage, object)

    stages = [dict(stage) for stage in augmentation_stages[:-1]]
    for stage in stages:
        stage["feature_order_sha256"] = _feature_order_hash(feature_names)
    post_evidence = _population_evidence(
        stage="post_safe_smote",
        X=post_X,
        y=post_y,
        weights=post_weights,
        lineage=post_lineage,
        features=feature_names,
    )
    post_evidence.update({
        "safe_smote_enabled": True,
        "safe_smote_synthetic_rows": synthetic_rows,
        "generation_scope": "ONCE_IN_FULL_93_FEATURE_REPRESENTATION",
    })
    stages.append(post_evidence)

    return FrozenResamplingPopulation(
        feature_names=feature_names,
        feature_order_sha256=_feature_order_hash(feature_names),
        seed=int(seed),
        imputer=imputer,
        pre_smote_X=pre_X,
        pre_smote_y=pre_y,
        pre_smote_weights=pre_weights,
        pre_smote_lineage=pre_lineage,
        post_smote_X=post_X,
        post_smote_y=post_y,
        post_smote_weights=post_weights,
        post_smote_lineage=post_lineage,
        stages=tuple(stages),
    )


def canonical_variant_features(
    variant: str, *, backend: Any | None = None,
) -> tuple[str, ...]:
    backend = backend or _load_backend()
    specifications = backend.build_variant_feature_sets()
    if variant not in specifications:
        raise PairedResamplingError(f"Unknown Variant: {variant}")
    specification = specifications[variant]
    if variant == "no_pca" or specification.get("status") != "RUNNABLE":
        raise PairedResamplingError(
            f"Variant {variant} is not projectable: {specification.get('status')}"
        )
    retained = specification.get("retained_features")
    if not isinstance(retained, list) or not retained:
        raise PairedResamplingError(f"Variant {variant} has no retained feature list")
    return tuple(map(str, retained))


def project_variant_population(
    frozen: FrozenResamplingPopulation,
    variant: str,
    retained_features: Sequence[str] | None = None,
    *,
    backend: Any | None = None,
) -> VariantProjection:
    """Project one prespecified Variant without refitting or resampling."""

    backend = backend or _load_backend()
    expected = canonical_variant_features(variant, backend=backend)
    selected = tuple(retained_features or expected)
    if selected != expected:
        raise PairedResamplingError(
            f"Variant {variant} feature order differs from its reviewed definition"
        )
    index_by_name = {name: index for index, name in enumerate(frozen.feature_names)}
    if len(index_by_name) != len(frozen.feature_names) or any(
        feature not in index_by_name for feature in selected
    ):
        raise PairedResamplingError("Variant features are not a valid full-space subset")
    indices = tuple(index_by_name[feature] for feature in selected)
    if tuple(sorted(indices)) != indices:
        raise PairedResamplingError("Variant features must preserve full-space column order")

    use_pre_smote = variant == "no_safe_smote"
    source_stage = "post_jitter_pre_smote" if use_pre_smote else "post_safe_smote"
    source_X = frozen.pre_smote_X if use_pre_smote else frozen.post_smote_X
    source_y = frozen.pre_smote_y if use_pre_smote else frozen.post_smote_y
    source_weights = (
        frozen.pre_smote_weights if use_pre_smote else frozen.post_smote_weights
    )
    source_lineage = (
        frozen.pre_smote_lineage if use_pre_smote else frozen.post_smote_lineage
    )
    X = _freeze_array(source_X[:, indices], np.float32)
    y = source_y
    lineage = source_lineage
    weights = (
        _freeze_array(np.ones(len(source_y), dtype=np.float32))
        if variant == "no_review_aware_training_weights"
        else source_weights
    )
    evidence = _population_evidence(
        stage=source_stage,
        X=X,
        y=y,
        weights=weights,
        lineage=lineage,
        features=selected,
    )
    evidence.update({
        "variant": variant,
        "source_stage": source_stage,
        "projection_only": True,
        "full_space_source_X_sha256": _array_hash(
            np.asarray(source_X, dtype="<f4")
        ),
        "safe_smote_enabled": not use_pre_smote,
        "safe_smote_synthetic_rows": (
            0 if use_pre_smote else frozen.synthetic_smote_rows
        ),
        "review_weights_enabled": variant != "no_review_aware_training_weights",
    })
    return VariantProjection(
        variant=variant,
        source_stage=source_stage,
        feature_names=selected,
        feature_indices=indices,
        X=X,
        y=y,
        weights=weights,
        lineage=lineage,
        evidence=evidence,
    )


def project_all_variants(
    frozen: FrozenResamplingPopulation,
    *,
    backend: Any | None = None,
) -> dict[str, VariantProjection]:
    backend = backend or _load_backend()
    specifications = backend.build_variant_feature_sets()
    variants = [
        *backend.runnable_official_variants(specifications),
        EXPLORATORY_VARIANT,
    ]
    return {
        variant: project_variant_population(frozen, variant, backend=backend)
        for variant in variants
    }


def transform_validation_full_space(
    frozen: FrozenResamplingPopulation,
    validation_frame: pd.DataFrame,
) -> FrozenValidationPopulation:
    missing = [name for name in frozen.feature_names if name not in validation_frame]
    if missing:
        raise PairedResamplingError(f"Validation frame is missing: {missing}")
    X = _freeze_array(
        frozen.imputer.transform(validation_frame.loc[:, frozen.feature_names]),
        np.float32,
    )
    return FrozenValidationPopulation(
        feature_names=frozen.feature_names,
        feature_order_sha256=frozen.feature_order_sha256,
        X=X,
    )


def project_validation_population(
    validation: FrozenValidationPopulation,
    projection: VariantProjection,
    *,
    backend: Any | None = None,
) -> np.ndarray:
    backend = backend or _load_backend()
    if validation.feature_names != tuple(backend.get_r92_feature_order()):
        raise PairedResamplingError("Validation population is not in canonical full space")
    return _freeze_array(validation.X[:, projection.feature_indices], np.float32)


def frozen_resampling_parity(
    frozen: FrozenResamplingPopulation,
    projections: Mapping[str, VariantProjection],
    *,
    backend: Any | None = None,
) -> dict[str, Any]:
    """Return machine-readable, independently recomputable parity evidence."""

    backend = backend or _load_backend()
    required = {
        *backend.runnable_official_variants(backend.build_variant_feature_sets()),
        EXPLORATORY_VARIANT,
    }
    missing = sorted(required - set(projections))
    extra = sorted(set(projections) - required)
    if missing or extra:
        raise PairedResamplingError(
            f"Projection set differs from runnable Variant set: missing={missing}, extra={extra}"
        )
    full = projections["full_new_reference"]
    no_weight = projections["no_review_aware_training_weights"]
    no_smote = projections["no_safe_smote"]

    full_vs_no_weight = {
        "X_exact": bool(np.array_equal(full.X, no_weight.X)),
        "y_exact": bool(np.array_equal(full.y, no_weight.y)),
        "lineage_exact": bool(np.array_equal(full.lineage, no_weight.lineage)),
        "row_order_exact": full.evidence["row_order_sha256"]
        == no_weight.evidence["row_order_sha256"],
        "only_weight_policy_changes": bool(
            np.array_equal(no_weight.weights, np.ones(len(no_weight.y), np.float32))
            and np.array_equal(full.weights, frozen.post_smote_weights)
        ),
    }
    full_vs_no_smote_pre = {
        "X_exact": bool(np.array_equal(no_smote.X, frozen.pre_smote_X)),
        "y_exact": bool(np.array_equal(no_smote.y, frozen.pre_smote_y)),
        "weight_exact": bool(
            np.array_equal(no_smote.weights, frozen.pre_smote_weights)
        ),
        "lineage_exact": bool(
            np.array_equal(no_smote.lineage, frozen.pre_smote_lineage)
        ),
        "row_order_exact": no_smote.evidence["row_order_sha256"]
        == _lineage_hash(frozen.pre_smote_lineage),
        "zero_safe_smote_rows": no_smote.evidence["safe_smote_synthetic_rows"]
        == 0,
    }

    post_smote_variant_checks: dict[str, dict[str, bool]] = {}
    for variant, projection in projections.items():
        if variant == "no_safe_smote":
            continue
        expected_X = frozen.post_smote_X[:, projection.feature_indices]
        post_smote_variant_checks[variant] = {
            "X_is_exact_column_projection": bool(
                np.array_equal(projection.X, expected_X)
            ),
            "y_exact": bool(np.array_equal(projection.y, frozen.post_smote_y)),
            "lineage_exact": bool(
                np.array_equal(projection.lineage, frozen.post_smote_lineage)
            ),
            "row_order_exact": projection.evidence["row_order_sha256"]
            == _lineage_hash(frozen.post_smote_lineage),
            "feature_order_hash_exact": projection.evidence[
                "feature_order_sha256"
            ] == _feature_order_hash(projection.feature_names),
        }

    stage_rows = {
        str(stage["stage"]): {
            "rows": int(stage["rows"]),
            "negative": int(stage["negative"]),
            "positive": int(stage["positive"]),
        }
        for stage in frozen.stages
    }
    augmentation_retained = bool(
        stage_rows["post_mixup"]["rows"] > stage_rows[
            "raw_locked_post_split_shuffle"
        ]["rows"]
        and stage_rows["post_dropout"]["rows"] > stage_rows["post_mixup"]["rows"]
        and stage_rows["post_jitter_pre_smote"]["rows"]
        > stage_rows["post_dropout"]["rows"]
    )
    all_checks = [
        frozen.augmentation_generation_count == 1,
        frozen.safe_smote_generation_count == 1,
        augmentation_retained,
        *full_vs_no_weight.values(),
        *full_vs_no_smote_pre.values(),
        *(
            value
            for checks in post_smote_variant_checks.values()
            for value in checks.values()
        ),
    ]
    return {
        "classification": "VERIFIED",
        "status": "PASS" if all(all_checks) else "FAIL",
        "generation": {
            "scope": "ONE_FULL_SPACE_POPULATION",
            "seed": frozen.seed,
            "augmentation_generation_count": frozen.augmentation_generation_count,
            "safe_smote_generation_count": frozen.safe_smote_generation_count,
            "full_feature_count": len(frozen.feature_names),
            "full_feature_order_sha256": frozen.feature_order_sha256,
        },
        "stage_counts": stage_rows,
        "stages": [dict(stage) for stage in frozen.stages],
        "pre_smote": _population_evidence(
            stage="post_jitter_pre_smote",
            X=frozen.pre_smote_X,
            y=frozen.pre_smote_y,
            weights=frozen.pre_smote_weights,
            lineage=frozen.pre_smote_lineage,
            features=frozen.feature_names,
        ),
        "post_smote": _population_evidence(
            stage="post_safe_smote",
            X=frozen.post_smote_X,
            y=frozen.post_smote_y,
            weights=frozen.post_smote_weights,
            lineage=frozen.post_smote_lineage,
            features=frozen.feature_names,
        ),
        "variants": {
            variant: dict(projection.evidence)
            for variant, projection in projections.items()
        },
        "comparisons": {
            "full_vs_no_weight": full_vs_no_weight,
            "full_vs_no_safe_smote_pre_smote": full_vs_no_smote_pre,
            "post_smote_variant_projections": post_smote_variant_checks,
        },
        "no_safe_smote_retains_mixup_dropout_jitter": augmentation_retained,
        "true_no_pca_status": backend.build_variant_feature_sets()["no_pca"]["status"],
        "exploratory_variant": EXPLORATORY_VARIANT,
    }
