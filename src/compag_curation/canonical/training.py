"""Group-safe canonical XGBoost search, refit, and fixed-point evaluation."""

from __future__ import annotations

import gc
import json
import math
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .spec import (
    CANONICAL_DECISION_THRESHOLD,
    CANONICAL_EARLY_STOPPING_ROUNDS,
    CANONICAL_EARLY_STOP_VALIDATION_FRACTION,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_GROUP_FOLDS,
    CANONICAL_RANDOM_STATE,
    CANONICAL_REFIT_ESTIMATORS,
    CANONICAL_SEARCH_ITERATIONS,
    CANONICAL_TEST_FRACTION,
)


CANONICAL_XGB_GPU_MAX_HOST_THREADS = 1


@dataclass(frozen=True)
class CanonicalTrainingConfig:
    random_state: int = CANONICAL_RANDOM_STATE
    test_fraction: float = CANONICAL_TEST_FRACTION
    group_folds: int = CANONICAL_GROUP_FOLDS
    search_iterations: int = CANONICAL_SEARCH_ITERATIONS
    refit_estimators: int = CANONICAL_REFIT_ESTIMATORS
    early_stopping_rounds: int = CANONICAL_EARLY_STOPPING_ROUNDS
    early_stop_validation_fraction: float = (
        CANONICAL_EARLY_STOP_VALIDATION_FRACTION
    )
    decision_threshold: float = CANONICAL_DECISION_THRESHOLD
    scale_pos_weight: float = 1.0
    device: str = "cuda"
    thread_count: int = 1

    def __post_init__(self) -> None:
        sealed_scientific_settings = (
            (self.random_state, 42),
            (self.test_fraction, 0.20),
            (self.group_folds, 5),
            (self.search_iterations, 30),
            (self.refit_estimators, 2000),
            (self.early_stopping_rounds, 30),
            (self.early_stop_validation_fraction, 0.30),
            (self.decision_threshold, 0.50),
            (self.scale_pos_weight, 1.0),
        )
        if any(
            observed != sealed
            for observed, sealed in sealed_scientific_settings
        ):
            raise ValueError("canonical XGBoost training settings are immutable")
        normalized_device = str(self.device).strip().lower()
        if normalized_device != "cuda":
            raise ValueError(
                "canonical XGBoost training is CUDA-only; CPU execution is unsupported"
            )
        object.__setattr__(self, "device", normalized_device)
        if isinstance(self.thread_count, bool) or not isinstance(self.thread_count, int):
            raise ValueError("canonical XGBoost host thread count must be an integer")
        if self.thread_count != 1:
            raise ValueError("canonical CUDA XGBoost host thread count is sealed at 1")


@dataclass(frozen=True)
class CanonicalGroupSplit:
    train_groups: frozenset[str]
    test_groups: frozenset[str]
    target_test_rows: int
    observed_test_rows: int


@dataclass(frozen=True)
class CanonicalSearchResult:
    best_parameters: Mapping[str, Any]
    best_average_precision: float
    sampled_configuration_count: int
    fold_count: int
    score_ledger: tuple[float, ...]


@dataclass(frozen=True)
class CanonicalTrainingResult:
    classifier: Any
    imputer_statistics: tuple[float, ...]
    feature_order: tuple[str, ...]
    split: CanonicalGroupSplit
    search: CanonicalSearchResult
    used_trees: int
    fixed_threshold: float
    test_metrics: Mapping[str, Any]
    retrospective_test_best_f1_threshold: float | None
    test_probabilities: tuple[float, ...]
    test_labels: tuple[int, ...]
    test_groups: tuple[str, ...]


ClassifierFactory = Callable[[Mapping[str, Any]], Any]
ResampleFunction = Callable[[Any, Any, Any, float, int, int], tuple[Any, Any, Any]]


def canonical_tile_balanced_split(groups: Sequence[Any]) -> CanonicalGroupSplit:
    """Create the recovered whole-card, tile-balanced 80:20 split."""

    import numpy as np
    from compag_curation.training.xgb import pick_groups_tile_balanced

    normalized = np.asarray([str(value) for value in groups])
    if normalized.ndim != 1 or len(normalized) < 2 or len(np.unique(normalized)) < 2:
        raise ValueError("canonical split requires at least two observed groups")
    train, test, target, observed = pick_groups_tile_balanced(
        normalized,
        test_fraction=CANONICAL_TEST_FRACTION,
        random_state=CANONICAL_RANDOM_STATE,
    )
    train_groups = frozenset(str(value) for value in train)
    test_groups = frozenset(str(value) for value in test)
    if not train_groups or not test_groups or train_groups & test_groups:
        raise RuntimeError("canonical tile-balanced split is not group-pure")
    if train_groups | test_groups != frozenset(normalized.tolist()):
        raise RuntimeError("canonical tile-balanced split omitted observed groups")
    return CanonicalGroupSplit(train_groups, test_groups, target, observed)


def validate_canonical_group_split(
    groups: Sequence[Any],
    split: CanonicalGroupSplit,
) -> None:
    observed = frozenset(str(value) for value in groups)
    if not split.train_groups or not split.test_groups:
        raise ValueError("canonical split contains an empty partition")
    if split.train_groups & split.test_groups:
        raise ValueError("canonical split contains group leakage")
    if split.train_groups | split.test_groups != observed:
        raise ValueError("canonical split does not close the observed groups")


def validate_canonical_review_weights(
    review_actions: Sequence[Any],
    supplied_weights: Sequence[Any],
) -> Any:
    """Validate supplied review weights without inferring or overwriting them."""

    import numpy as np
    from compag_curation.training.xgb import REVIEW_TAG_WEIGHTS

    if len(review_actions) != len(supplied_weights):
        raise ValueError("canonical review actions and weights do not align")
    values: list[float] = []
    for raw_action, raw_weight in zip(
        review_actions, supplied_weights, strict=True
    ):
        action = str(raw_action or "").strip().casefold()
        if action not in REVIEW_TAG_WEIGHTS:
            raise ValueError(f"unsupported canonical review action: {action}")
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("canonical review weight is missing or nonnumeric") from exc
        expected = float(REVIEW_TAG_WEIGHTS[action])
        if not math.isfinite(weight) or not math.isclose(
            weight, expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"canonical review action/weight mismatch: {action} requires {expected}"
            )
        values.append(weight)
    return np.asarray(values, dtype=np.float32)


def canonical_parameter_samples(
    config: CanonicalTrainingConfig = CanonicalTrainingConfig(),
) -> tuple[dict[str, Any], ...]:
    """Draw the exact 30 deterministic recovered random-search settings."""

    from scipy.stats import loguniform, randint, uniform
    from sklearn.model_selection import ParameterSampler

    distributions = {
        "clf__max_depth": randint(3, 10),
        "clf__min_child_weight": loguniform(0.5, 10.0),
        "clf__subsample": uniform(0.7, 0.3),
        "clf__colsample_bytree": uniform(0.7, 0.3),
        "clf__reg_lambda": loguniform(0.1, 10.0),
        "clf__reg_alpha": loguniform(1e-3, 1.0),
        "clf__gamma": loguniform(1e-3, 1.0),
        "clf__learning_rate": loguniform(0.02, 0.2),
        "clf__max_bin": [256],
        "smote__sampling_strategy": [0.4, 0.5, 0.6],
        "smote__k_neighbors": randint(2, 6),
    }
    samples = ParameterSampler(
        distributions,
        n_iter=config.search_iterations,
        random_state=config.random_state,
    )
    return tuple({name: _plain_scalar(value) for name, value in row.items()} for row in samples)


def _plain_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _base_classifier_parameters(config: CanonicalTrainingConfig) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
        "random_state": config.random_state,
        "n_jobs": config.thread_count,
        "verbosity": 0,
        "tree_method": "hist",
        "device": config.device,
        "fail_on_invalid_gpu_id": True,
        "max_bin": 256,
    }


def _require_cuda_execution_device(device: Any, role: str) -> str:
    normalized = str(device).strip().lower()
    if normalized != "cuda":
        raise ValueError(f"{role} is CUDA-only; CPU execution is unsupported")
    return normalized


def _require_xgboost_cuda(xgb: Any | None = None) -> Any:
    """Return CuPy only when both XGBoost and the runtime provide CUDA."""

    if xgb is None:
        import xgboost as xgb

    build_info = getattr(xgb, "build_info", None)
    if not callable(build_info) or not bool(build_info().get("USE_CUDA", False)):
        raise RuntimeError(
            "canonical XGBoost CUDA was requested, but this XGBoost build has no CUDA support"
        )
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "canonical XGBoost CUDA was requested, but CuPy is not installed"
        ) from exc
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError(
            "canonical XGBoost CUDA was requested, but the CUDA runtime is unavailable"
        ) from exc
    if device_count < 1:
        raise RuntimeError(
            "canonical XGBoost CUDA was requested, but no CUDA device is available"
        )
    return cp


def _cupy_cuda_ordinal(cp: Any) -> int:
    try:
        ordinal = int(cp.cuda.runtime.getDevice())
    except Exception as exc:
        raise RuntimeError(
            "canonical XGBoost CUDA device ordinal cannot be established"
        ) from exc
    if ordinal < 0:
        raise RuntimeError("canonical XGBoost CUDA device ordinal is invalid")
    return ordinal


def _release_cuda_execution_memory(device: Any, cp: Any | None = None) -> None:
    """Synchronize and release unreferenced CuPy CUDA allocation caches."""

    _require_cuda_execution_device(
        device,
        "canonical XGBoost CUDA memory release",
    )
    selected = _require_xgboost_cuda() if cp is None else cp
    try:
        device_count = int(selected.cuda.runtime.getDeviceCount())
        ordinal = _cupy_cuda_ordinal(selected)
        synchronize = selected.cuda.runtime.deviceSynchronize
        default_pool_factory = selected.get_default_memory_pool
        pinned_pool_factory = selected.get_default_pinned_memory_pool
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "canonical XGBoost CUDA memory-release interface is unavailable"
        ) from exc
    if device_count < 1 or ordinal >= device_count:
        raise RuntimeError(
            "canonical XGBoost CUDA memory-release device is unavailable"
        )
    if not all(
        callable(value)
        for value in (synchronize, default_pool_factory, pinned_pool_factory)
    ):
        raise RuntimeError(
            "canonical XGBoost CUDA memory-release interface is unavailable"
        )
    try:
        gc.collect()
        synchronize()
        default_pool = default_pool_factory()
        pinned_pool = pinned_pool_factory()
        default_release = getattr(default_pool, "free_all_blocks", None)
        pinned_release = getattr(pinned_pool, "free_all_blocks", None)
        if not callable(default_release) or not callable(pinned_release):
            raise RuntimeError(
                "canonical XGBoost CUDA memory pools cannot be released"
            )
        default_release()
        pinned_release()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "canonical XGBoost CUDA memory release failed"
        ) from exc


@contextmanager
def _reject_xgboost_device_warnings() -> Any:
    """Turn XGBoost device fallback/mismatch warnings into hard failures."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for warning in caught:
        message = str(warning.message)
        lowered = message.casefold()
        if (
            "mismatched devices" in lowered
            or "falling back" in lowered
            or "setting device to cpu" in lowered
            or "not compiled with cuda" in lowered
            or ("fallback" in lowered and "device" in lowered)
        ):
            raise RuntimeError(
                f"canonical XGBoost rejected a CUDA fallback warning: {message}"
            )


def _xgboost_booster(model_or_booster: Any) -> Any:
    get_booster = getattr(model_or_booster, "get_booster", None)
    return get_booster() if callable(get_booster) else model_or_booster


def _attest_xgboost_cuda_booster(
    model_or_booster: Any,
    cp: Any,
) -> Any:
    """Attest XGBoost 2.1.1's effective device after fit/load/set_param."""

    booster = _xgboost_booster(model_or_booster)
    save_config = getattr(booster, "save_config", None)
    if not callable(save_config):
        raise RuntimeError("canonical XGBoost booster cannot expose effective config")
    try:
        with _reject_xgboost_device_warnings():
            rendered_config = save_config()
        parsed = json.loads(rendered_config)
        generic = parsed["learner"]["generic_param"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("canonical XGBoost effective config is malformed") from exc
    expected_device = f"cuda:{_cupy_cuda_ordinal(cp)}"
    try:
        nthread = int(generic.get("nthread", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("canonical XGBoost effective host thread count is invalid") from exc
    if (
        generic.get("device") != expected_device
        or nthread != CANONICAL_XGB_GPU_MAX_HOST_THREADS
        or str(generic.get("fail_on_invalid_gpu_id", "0")) != "1"
    ):
        raise RuntimeError(
            "canonical XGBoost effective CUDA device/thread/fallback policy changed"
        )
    return booster


def _execution_arrays(device: str, *values: Any) -> tuple[Any, ...]:
    if device == "cpu":
        return values
    cp = _require_xgboost_cuda()
    return tuple(cp.asarray(value) for value in values)


def _xgb_factory(parameters: Mapping[str, Any]) -> Any:
    _require_cuda_execution_device(
        parameters.get("device"),
        "canonical XGBoost classifier construction",
    )
    import xgboost as xgb

    _require_xgboost_cuda(xgb)

    with _reject_xgboost_device_warnings():
        return xgb.XGBClassifier(**dict(parameters))


def _classifier_parameters(
    config: CanonicalTrainingConfig,
    sampled: Mapping[str, Any],
    *,
    estimators: int,
    early_stopping_rounds: int | None,
) -> dict[str, Any]:
    parameters = _base_classifier_parameters(config)
    parameters.update(
        {
            name.removeprefix("clf__"): value
            for name, value in sampled.items()
            if name.startswith("clf__")
        }
    )
    parameters["n_estimators"] = int(estimators)
    parameters["early_stopping_rounds"] = early_stopping_rounds
    return parameters


def _safe_resample(
    features: Any,
    labels: Any,
    weights: Any,
    sampling_strategy: float,
    neighbors: int,
    random_state: int,
) -> tuple[Any, Any, Any]:
    from compag_curation.training.safe_smote import build_smote_safe
    from compag_curation.training.xgb import make_weights_after_smote

    sampler = build_smote_safe(
        labels,
        sampling_strategy=float(sampling_strategy),
        k_neighbors=int(neighbors),
        random_state=int(random_state),
    )
    if sampler is None:
        return features, labels, weights
    resampled_features, resampled_labels = sampler.fit_resample(features, labels)
    resampled_weights = make_weights_after_smote(
        weights,
        labels,
        resampled_labels,
    )
    return resampled_features, resampled_labels, resampled_weights


def _positive_probabilities(
    model: Any,
    features: Any,
    *,
    device: str = "cuda",
) -> Any:
    import numpy as np

    if str(device).strip().lower() == "cpu":
        # Private dependency-injected unit harnesses may use a fake CPU model.
        probabilities = np.asarray(model.predict_proba(features), dtype=np.float32)
        if probabilities.ndim != 2 or probabilities.shape != (len(features), 2):
            raise RuntimeError("canonical classifier returned invalid probabilities")
        positive = probabilities[:, 1]
    else:
        _require_cuda_execution_device(device, "canonical XGBoost prediction")
        cp = _require_xgboost_cuda()
        booster = prediction_features = raw_probabilities = None
        prediction_complete = False
        try:
            booster = _attest_xgboost_cuda_booster(model, cp)
            prediction_features = cp.asarray(features)
            with _reject_xgboost_device_warnings():
                raw_probabilities = booster.inplace_predict(
                    prediction_features,
                    validate_features=False,
                )
            _attest_xgboost_cuda_booster(booster, cp)
            if not hasattr(raw_probabilities, "__cuda_array_interface__"):
                raise RuntimeError(
                    "canonical XGBoost CUDA prediction returned a host array"
                )
            positive = np.asarray(
                cp.asnumpy(raw_probabilities),
                dtype=np.float32,
            ).reshape(-1)
            if positive.shape != (len(features),):
                raise RuntimeError(
                    "canonical classifier returned invalid probabilities"
                )
            prediction_complete = True
        finally:
            del booster, raw_probabilities, prediction_features
            if prediction_complete:
                _release_cuda_execution_memory(device, cp)
    if np.any(~np.isfinite(positive)) or np.any((positive < 0.0) | (positive > 1.0)):
        raise RuntimeError("canonical classifier returned non-probabilities")
    return positive


def _weighted_parameter_search(
    features: Any,
    labels: Any,
    groups: Any,
    weights: Any,
    cv_pairs: Sequence[tuple[Any, Any]],
    parameter_samples: Sequence[Mapping[str, Any]],
    *,
    config: CanonicalTrainingConfig,
    canonical_validation_mask: Any | None = None,
    classifier_factory: ClassifierFactory = _xgb_factory,
    resample: ResampleFunction = _safe_resample,
) -> CanonicalSearchResult:
    import numpy as np
    from sklearn.metrics import average_precision_score

    if len(parameter_samples) != config.search_iterations:
        raise ValueError("canonical random search must evaluate exactly 30 configurations")
    if len(cv_pairs) != config.group_folds:
        raise ValueError("canonical random search requires exactly five valid group folds")
    if canonical_validation_mask is None:
        canonical_validation_mask = np.ones(len(features), dtype=bool)
    else:
        canonical_validation_mask = np.asarray(canonical_validation_mask, dtype=bool)
    if canonical_validation_mask.shape != (len(features),):
        raise ValueError("canonical validation-scale mask does not align")
    best_score = -math.inf
    best_parameters: dict[str, Any] | None = None
    ledger: list[float] = []
    for sampled in parameter_samples:
        fold_scores: list[float] = []
        for train_indices, validation_indices in cv_pairs:
            validation_indices = np.asarray(validation_indices, dtype=int)
            validation_indices = validation_indices[
                canonical_validation_mask[validation_indices]
            ]
            if (
                not len(validation_indices)
                or len(np.unique(labels[validation_indices])) != 2
            ):
                raise ValueError(
                    "canonical outer validation fold lacks both classes at scale 1.0"
                )
            model, imputer, _used_trees = _fit_early_stop_then_full(
                features[train_indices],
                labels[train_indices],
                groups[train_indices],
                weights[train_indices],
                sampled,
                config=config,
                canonical_validation_mask=canonical_validation_mask[train_indices],
                classifier_factory=classifier_factory,
                resample=resample,
            )
            validation_features = probabilities = fold_booster = None
            production_cuda = classifier_factory is _xgb_factory
            try:
                validation_features = imputer.transform(features[validation_indices])
                probabilities = _positive_probabilities(
                    model,
                    validation_features,
                    device=config.device,
                )
                score = float(
                    average_precision_score(
                        labels[validation_indices],
                        probabilities,
                    )
                )
                if not math.isfinite(score):
                    raise RuntimeError(
                        "canonical search produced a nonfinite AP score"
                    )
            finally:
                if production_cuda:
                    fold_booster = _xgboost_booster(model)
                del fold_booster, model, imputer
                del probabilities, validation_features, _used_trees
                if production_cuda:
                    _release_cuda_execution_memory(config.device)
            fold_scores.append(score)
        mean_score = float(np.mean(fold_scores))
        ledger.append(mean_score)
        if mean_score > best_score:
            best_score = mean_score
            best_parameters = dict(sampled)
    if best_parameters is None:
        raise RuntimeError("canonical random search did not produce a valid setting")
    return CanonicalSearchResult(
        best_parameters=best_parameters,
        best_average_precision=best_score,
        sampled_configuration_count=len(parameter_samples),
        fold_count=len(cv_pairs),
        score_ledger=tuple(ledger),
    )


def _group_safe_inner_split(
    labels: Any,
    groups: Any,
    config: CanonicalTrainingConfig,
    *,
    canonical_validation_mask: Any | None = None,
) -> tuple[Any, Any]:
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit

    if len(np.unique(groups)) < 2:
        raise ValueError("canonical early stopping requires at least two train groups")
    if canonical_validation_mask is None:
        canonical_validation_mask = np.ones(len(labels), dtype=bool)
    else:
        canonical_validation_mask = np.asarray(canonical_validation_mask, dtype=bool)
    if canonical_validation_mask.shape != (len(labels),):
        raise ValueError("canonical inner validation-scale mask does not align")
    splitter = GroupShuffleSplit(
        n_splits=64,
        test_size=config.early_stop_validation_fraction,
        random_state=config.random_state + 123,
    )
    for train_indices, validation_indices in splitter.split(
        np.zeros(len(labels)), labels, groups
    ):
        validation_indices = validation_indices[
            canonical_validation_mask[validation_indices]
        ]
        if (
            len(validation_indices)
            and
            len(np.unique(labels[train_indices])) == 2
            and len(np.unique(labels[validation_indices])) == 2
            and not set(groups[train_indices]) & set(groups[validation_indices])
        ):
            return train_indices, validation_indices
    raise ValueError("canonical group-safe 0.30 early-stop split lacks both classes")


def _fit_early_stop_then_full(
    features: Any,
    labels: Any,
    groups: Any,
    weights: Any,
    sampled: Mapping[str, Any],
    *,
    config: CanonicalTrainingConfig,
    canonical_validation_mask: Any | None = None,
    classifier_factory: ClassifierFactory = _xgb_factory,
    resample: ResampleFunction = _safe_resample,
) -> tuple[Any, Any, int]:
    from sklearn.impute import SimpleImputer

    inner_train, inner_validation = _group_safe_inner_split(
        labels,
        groups,
        config,
        canonical_validation_mask=canonical_validation_mask,
    )
    early_imputer = SimpleImputer(strategy="median")
    early_train = early_imputer.fit_transform(features[inner_train])
    early_validation = early_imputer.transform(features[inner_validation])
    fit_features, fit_labels, fit_weights = resample(
        early_train,
        labels[inner_train],
        weights[inner_train],
        float(sampled["smote__sampling_strategy"]),
        int(sampled["smote__k_neighbors"]),
        config.random_state,
    )
    early_model = classifier_factory(
        _classifier_parameters(
            config,
            sampled,
            estimators=config.refit_estimators,
            early_stopping_rounds=config.early_stopping_rounds,
        )
    )
    (
        execution_features,
        execution_labels,
        execution_weights,
        execution_validation,
        execution_validation_labels,
    ) = _execution_arrays(
        config.device,
        fit_features,
        fit_labels,
        fit_weights,
        early_validation,
        labels[inner_validation],
    )
    production_cuda = classifier_factory is _xgb_factory
    early_cp = _require_xgboost_cuda() if production_cuda else None
    early_booster = None
    try:
        if production_cuda:
            with _reject_xgboost_device_warnings():
                early_model.fit(
                    execution_features,
                    execution_labels,
                    sample_weight=execution_weights,
                    eval_set=[
                        (execution_validation, execution_validation_labels)
                    ],
                    verbose=False,
                )
            early_booster = _attest_xgboost_cuda_booster(
                early_model,
                early_cp,
            )
        else:
            early_model.fit(
                execution_features,
                execution_labels,
                sample_weight=execution_weights,
                eval_set=[(execution_validation, execution_validation_labels)],
                verbose=False,
            )
        best_iteration = getattr(early_model, "best_iteration", None)
        used_trees = (
            int(best_iteration) + 1
            if best_iteration is not None
            else config.refit_estimators
        )
        if not 1 <= used_trees <= config.refit_estimators:
            raise RuntimeError(
                "canonical early stopping returned an invalid tree count"
            )
    finally:
        del early_booster, early_model
        del execution_features, execution_labels, execution_weights
        del execution_validation, execution_validation_labels
        if production_cuda:
            _release_cuda_execution_memory(config.device, early_cp)

    final_imputer = SimpleImputer(strategy="median")
    full_train = final_imputer.fit_transform(features)
    full_features, full_labels, full_weights = resample(
        full_train,
        labels,
        weights,
        float(sampled["smote__sampling_strategy"]),
        int(sampled["smote__k_neighbors"]),
        config.random_state,
    )
    final_model = classifier_factory(
        _classifier_parameters(
            config,
            sampled,
            estimators=used_trees,
            early_stopping_rounds=None,
        )
    )
    execution_features, execution_labels, execution_weights = _execution_arrays(
        config.device,
        full_features,
        full_labels,
        full_weights,
    )
    final_cp = _require_xgboost_cuda() if production_cuda else None
    final_booster = None
    try:
        if production_cuda:
            with _reject_xgboost_device_warnings():
                final_model.fit(
                    execution_features,
                    execution_labels,
                    sample_weight=execution_weights,
                    verbose=False,
                )
            final_booster = _attest_xgboost_cuda_booster(
                final_model,
                final_cp,
            )
        else:
            final_model.fit(
                execution_features,
                execution_labels,
                sample_weight=execution_weights,
                verbose=False,
            )
    except BaseException:
        del final_model
        raise
    finally:
        del final_booster
        del execution_features, execution_labels, execution_weights
        if production_cuda:
            _release_cuda_execution_memory(config.device, final_cp)
    return final_model, final_imputer, used_trees


def train_canonical_xgb(
    feature_frame: Any,
    labels: Sequence[int],
    groups: Sequence[Any],
    review_actions: Sequence[Any],
    review_weights: Sequence[Any],
    *,
    scales: Sequence[Any] | None = None,
    split: CanonicalGroupSplit | None = None,
    config: CanonicalTrainingConfig = CanonicalTrainingConfig(),
) -> CanonicalTrainingResult:
    """Run exact-order canonical training without selecting on held-out test."""

    _require_cuda_execution_device(config.device, "canonical XGBoost training")
    import numpy as np
    from compag_curation.training.xgb import (
        _metrics,
        build_cv_pairs,
    )

    _require_xgboost_cuda()
    feature_rows = list(feature_frame)
    if not feature_rows or any(
        not isinstance(row, Mapping) or tuple(row) != CANONICAL_FEATURE_ORDER
        for row in feature_rows
    ):
        raise ValueError("canonical training rows must have the exact ordered feature schema")
    try:
        matrix = np.asarray(
            [
                [float(row[name]) for name in CANONICAL_FEATURE_ORDER]
                for row in feature_rows
            ],
            dtype=np.float32,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("canonical training features must be numeric") from exc
    normalized_labels = np.asarray(labels, dtype=int)
    normalized_groups = np.asarray([str(value) for value in groups])
    weights = validate_canonical_review_weights(review_actions, review_weights)
    count = len(matrix)
    if not (
        len(normalized_labels) == len(normalized_groups) == len(weights) == count
    ):
        raise ValueError("canonical training inputs do not align")
    normalized_scales = (
        np.ones(count, dtype=np.float32)
        if scales is None
        else np.asarray(scales, dtype=np.float32)
    )
    if normalized_scales.shape != (count,) or np.any(
        ~np.isin(normalized_scales, np.asarray((0.67, 0.80, 1.00, 1.25), dtype=np.float32))
    ):
        raise ValueError("canonical training scales must use the exact export scale set")
    if count == 0 or np.any(~np.isfinite(matrix)):
        raise ValueError("canonical training features must be present and finite")
    if set(np.unique(normalized_labels).tolist()) - {0, 1}:
        raise ValueError("canonical training labels must be binary")

    # A skip action is excluded before splitting, Safe-SMOTE, and fit.
    keep = weights > 0.0
    matrix = matrix[keep]
    normalized_labels = normalized_labels[keep]
    normalized_groups = normalized_groups[keep]
    normalized_scales = normalized_scales[keep]
    weights = weights[keep]
    if split is None:
        split = canonical_tile_balanced_split(normalized_groups)
    validate_canonical_group_split(normalized_groups, split)
    train_indices = np.flatnonzero(
        np.isin(normalized_groups, tuple(split.train_groups))
    )
    test_indices = np.flatnonzero(
        np.isin(normalized_groups, tuple(split.test_groups))
        & (normalized_scales == np.float32(1.0))
    )
    if set(normalized_groups[train_indices]) & set(normalized_groups[test_indices]):
        raise RuntimeError("canonical train/test groups overlap")
    if (
        len(np.unique(normalized_labels[train_indices])) != 2
        or len(np.unique(normalized_labels[test_indices])) != 2
    ):
        raise ValueError("canonical train and untouched test must each contain both classes")

    # The frozen canonical profile uses exported crop scales plus train-fold
    # Safe-SMOTE. Mixup, dropout, and jitter belong to a later notebook path.
    augmented_features = matrix[train_indices]
    augmented_labels = normalized_labels[train_indices]
    augmented_groups = normalized_groups[train_indices]
    augmented_weights = weights[train_indices]
    augmented_scales = normalized_scales[train_indices]
    cv_pairs = build_cv_pairs(
        augmented_groups,
        augmented_labels,
        config.group_folds,
    )
    search = _weighted_parameter_search(
        augmented_features,
        augmented_labels,
        augmented_groups,
        augmented_weights,
        cv_pairs,
        canonical_parameter_samples(config),
        config=config,
        canonical_validation_mask=(augmented_scales == np.float32(1.0)),
    )
    final_model, final_imputer, used_trees = _fit_early_stop_then_full(
        augmented_features,
        augmented_labels,
        augmented_groups,
        augmented_weights,
        search.best_parameters,
        config=config,
        canonical_validation_mask=(augmented_scales == np.float32(1.0)),
    )
    test_features = final_imputer.transform(matrix[test_indices])
    probabilities = _positive_probabilities(
        final_model,
        test_features,
        device=config.device,
    )
    test_labels = normalized_labels[test_indices]
    metrics = _metrics(test_labels, probabilities, CANONICAL_DECISION_THRESHOLD)
    statistics = tuple(float(value) for value in final_imputer.statistics_)
    if len(statistics) != len(CANONICAL_FEATURE_ORDER) or any(
        not math.isfinite(value) for value in statistics
    ):
        raise RuntimeError("canonical median imputer state is invalid")
    return CanonicalTrainingResult(
        classifier=final_model,
        imputer_statistics=statistics,
        feature_order=CANONICAL_FEATURE_ORDER,
        split=split,
        search=search,
        used_trees=used_trees,
        fixed_threshold=CANONICAL_DECISION_THRESHOLD,
        test_metrics=metrics,
        retrospective_test_best_f1_threshold=None,
        test_probabilities=tuple(float(value) for value in probabilities),
        test_labels=tuple(int(value) for value in test_labels),
        test_groups=tuple(str(value) for value in normalized_groups[test_indices]),
    )


__all__ = [
    "CANONICAL_XGB_GPU_MAX_HOST_THREADS",
    "CanonicalGroupSplit",
    "CanonicalSearchResult",
    "CanonicalTrainingConfig",
    "CanonicalTrainingResult",
    "canonical_parameter_samples",
    "canonical_tile_balanced_split",
    "train_canonical_xgb",
    "validate_canonical_review_weights",
    "validate_canonical_group_split",
]
