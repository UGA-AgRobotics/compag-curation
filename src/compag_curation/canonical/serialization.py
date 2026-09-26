"""Portable UBJ/NPY helpers for canonical model state and inference."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .features import CanonicalFeatureState, validate_canonical_feature_state
from .spec import (
    CANONICAL_DECISION_THRESHOLD,
    CANONICAL_EMBEDDING_DIMENSIONS,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_PCA_DIMENSIONS,
    CANONICAL_PROFILE,
)
from .training import (
    CANONICAL_XGB_GPU_MAX_HOST_THREADS,
    _attest_xgboost_cuda_booster,
    _reject_xgboost_device_warnings,
    _require_cuda_execution_device,
    _require_xgboost_cuda,
)


@dataclass(frozen=True)
class CanonicalFeatureStatePayloads:
    prototype_npy: bytes
    pca_components_npy: bytes
    pca_mean_npy: bytes


@dataclass(frozen=True)
class PortableCanonicalPredictor:
    booster: Any
    imputer_statistics: Any
    feature_order: tuple[str, ...]
    fixed_threshold: float = CANONICAL_DECISION_THRESHOLD
    device: str = "cuda"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "device",
            _require_cuda_execution_device(
                self.device,
                "canonical portable XGBoost predictor",
            ),
        )


def portable_classifier_bytes(classifier: Any) -> bytes:
    """Serialize an XGBoost estimator as raw UBJ, never as Python objects."""

    booster = (
        classifier.get_booster()
        if callable(getattr(classifier, "get_booster", None))
        else classifier
    )
    save_raw = getattr(booster, "save_raw", None)
    if not callable(save_raw):
        raise TypeError("canonical classifier does not expose XGBoost save_raw")
    if tuple(getattr(booster, "feature_names", None) or ()) != CANONICAL_FEATURE_ORDER:
        raise ValueError("canonical classifier feature names do not match the sealed 93-feature order")
    payload = bytes(save_raw(raw_format="ubj"))
    if not payload or payload.startswith(b"\x80"):
        raise RuntimeError("canonical classifier UBJ payload is empty or object-like")
    return payload


def _float32_npy_bytes(value: Any, expected_shape: tuple[int, ...], role: str) -> bytes:
    import numpy as np

    array = np.asarray(value, dtype="<f4")
    if array.shape != expected_shape or np.any(~np.isfinite(array)):
        raise ValueError(f"canonical {role} has invalid shape or values")
    handle = io.BytesIO()
    np.save(handle, array, allow_pickle=False)
    payload = handle.getvalue()
    if not payload.startswith(b"\x93NUMPY"):
        raise RuntimeError(f"canonical {role} did not serialize as NPY")
    return payload


def serialize_canonical_feature_state(
    state: CanonicalFeatureState,
) -> CanonicalFeatureStatePayloads:
    validate_canonical_feature_state(state)
    return CanonicalFeatureStatePayloads(
        prototype_npy=_float32_npy_bytes(
            state.prototype,
            (CANONICAL_EMBEDDING_DIMENSIONS,),
            "prototype",
        ),
        pca_components_npy=_float32_npy_bytes(
            state.pca_components,
            (CANONICAL_PCA_DIMENSIONS, CANONICAL_EMBEDDING_DIMENSIONS),
            "PCA components",
        ),
        pca_mean_npy=_float32_npy_bytes(
            state.pca_mean,
            (CANONICAL_EMBEDDING_DIMENSIONS,),
            "PCA mean",
        ),
    )


def _load_float32_npy(
    payload: bytes,
    expected_shape: tuple[int, ...],
    role: str,
) -> Any:
    import numpy as np

    if not isinstance(payload, bytes) or not payload.startswith(b"\x93NUMPY"):
        raise ValueError(f"canonical {role} is not an NPY payload")
    handle = io.BytesIO(payload)
    try:
        array = np.load(handle, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"canonical {role} NPY cannot be decoded") from exc
    if handle.tell() != len(payload):
        raise ValueError(f"canonical {role} NPY has trailing bytes")
    array = np.asarray(array)
    if (
        array.dtype.kind != "f"
        or array.dtype.itemsize != 4
        or array.shape != expected_shape
        or np.any(~np.isfinite(array))
    ):
        raise ValueError(f"canonical {role} NPY has invalid dtype, shape, or values")
    output = np.ascontiguousarray(array, dtype=np.float32)
    output.setflags(write=False)
    return output


def deserialize_canonical_feature_state(
    payloads: CanonicalFeatureStatePayloads,
    *,
    training_row_count: int,
    positive_row_count: int,
) -> CanonicalFeatureState:
    if training_row_count < 32 or positive_row_count < 1:
        raise ValueError("canonical feature-state row counts are invalid")
    state = CanonicalFeatureState(
        prototype=_load_float32_npy(
            payloads.prototype_npy,
            (CANONICAL_EMBEDDING_DIMENSIONS,),
            "prototype",
        ),
        pca_components=_load_float32_npy(
            payloads.pca_components_npy,
            (CANONICAL_PCA_DIMENSIONS, CANONICAL_EMBEDDING_DIMENSIONS),
            "PCA components",
        ),
        pca_mean=_load_float32_npy(
            payloads.pca_mean_npy,
            (CANONICAL_EMBEDDING_DIMENSIONS,),
            "PCA mean",
        ),
        training_row_count=training_row_count,
        positive_row_count=positive_row_count,
    )
    validate_canonical_feature_state(state)
    return state


def canonical_imputer_record(statistics: Sequence[Any]) -> dict[str, Any]:
    if len(statistics) != len(CANONICAL_FEATURE_ORDER):
        raise ValueError("canonical imputer statistics do not match the 93 features")
    values: dict[str, float] = {}
    for name, raw in zip(CANONICAL_FEATURE_ORDER, statistics, strict=True):
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"canonical imputer statistic is nonnumeric: {name}") from exc
        if not math.isfinite(value):
            raise ValueError(f"canonical imputer statistic is nonfinite: {name}")
        values[name] = value
    return {
        "schema": "compag-curation-canonical-imputer/v1",
        "profile": CANONICAL_PROFILE,
        "strategy": "median_train_only",
        "feature_order": list(CANONICAL_FEATURE_ORDER),
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        "statistics": values,
    }


def _normalize_device(device: str) -> str:
    return _require_cuda_execution_device(
        device,
        "canonical portable XGBoost prediction",
    )


def _device_dmatrix(
    xgb: Any,
    matrix: Any,
    feature_order: Sequence[str],
    device: str,
) -> tuple[Any, Any | None]:
    _normalize_device(device)
    cp = _require_xgboost_cuda(xgb)
    cuda_matrix = cp.asarray(matrix)
    return xgb.DMatrix(cuda_matrix, feature_names=list(feature_order)), cp


def _host_float32(value: Any, cp: Any | None) -> Any:
    import numpy as np

    if cp is not None and hasattr(value, "__cuda_array_interface__"):
        value = cp.asnumpy(value)
    return np.asarray(value, dtype=np.float32)


def load_portable_predictor(
    classifier_ubj: bytes,
    imputer_statistics: Sequence[Any],
    *,
    device: str = "cuda",
) -> PortableCanonicalPredictor:
    resolved_device = _normalize_device(device)

    import numpy as np
    import xgboost as xgb

    cp = _require_xgboost_cuda(xgb)
    if not isinstance(classifier_ubj, bytes) or not classifier_ubj:
        raise ValueError("canonical classifier payload must be nonempty UBJ bytes")
    record = canonical_imputer_record(imputer_statistics)
    booster = xgb.Booster()
    try:
        with _reject_xgboost_device_warnings():
            booster.load_model(bytearray(classifier_ubj))
            booster.set_param(
                {
                    "device": resolved_device,
                    "nthread": CANONICAL_XGB_GPU_MAX_HOST_THREADS,
                    "fail_on_invalid_gpu_id": True,
                }
            )
    except (TypeError, ValueError, xgb.core.XGBoostError) as exc:
        raise ValueError("canonical classifier UBJ cannot be loaded") from exc
    _attest_xgboost_cuda_booster(booster, cp)
    if (
        tuple(booster.feature_names or ()) != CANONICAL_FEATURE_ORDER
        or int(booster.num_features()) != len(CANONICAL_FEATURE_ORDER)
    ):
        raise ValueError("canonical classifier UBJ feature schema changed")
    medians = np.asarray(
        [record["statistics"][name] for name in CANONICAL_FEATURE_ORDER],
        dtype=np.float32,
    )
    medians.setflags(write=False)
    return PortableCanonicalPredictor(
        booster=booster,
        imputer_statistics=medians,
        feature_order=CANONICAL_FEATURE_ORDER,
        device=resolved_device,
    )


def _ordered_matrix(feature_rows: Any) -> Any:
    import numpy as np

    if hasattr(feature_rows, "columns"):
        missing = [
            name for name in CANONICAL_FEATURE_ORDER if name not in feature_rows.columns
        ]
        if missing:
            raise ValueError(
                "canonical inference table is missing predictors: "
                + ", ".join(missing)
            )
        matrix = feature_rows.loc[:, list(CANONICAL_FEATURE_ORDER)].to_numpy(
            dtype=np.float32
        )
    elif isinstance(feature_rows, Sequence) and feature_rows and isinstance(
        feature_rows[0], Mapping
    ):
        matrix = np.asarray(
            [
                [row[name] for name in CANONICAL_FEATURE_ORDER]
                for row in feature_rows
            ],
            dtype=np.float32,
        )
    else:
        matrix = np.asarray(feature_rows, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(CANONICAL_FEATURE_ORDER):
        raise ValueError("canonical inference matrix must have exactly 93 columns")
    if np.any(np.isinf(matrix)):
        raise ValueError("canonical inference matrix contains infinite values")
    return matrix


def _impute_matrix(matrix: Any, statistics: Any) -> Any:
    import numpy as np

    output = np.asarray(matrix, dtype=np.float32).copy()
    rows, columns = np.where(np.isnan(output))
    if len(rows):
        output[rows, columns] = statistics[columns]
    if np.any(~np.isfinite(output)):
        raise ValueError("canonical median imputation did not produce finite values")
    return output


def predict_portable_probabilities(
    predictor: PortableCanonicalPredictor,
    feature_rows: Any,
) -> Any:
    resolved_device = _normalize_device(predictor.device)

    import numpy as np
    import xgboost as xgb

    if predictor.feature_order != CANONICAL_FEATURE_ORDER:
        raise ValueError("portable predictor feature order changed")
    cp = _require_xgboost_cuda(xgb)
    with _reject_xgboost_device_warnings():
        predictor.booster.set_param(
            {
                "device": resolved_device,
                "nthread": CANONICAL_XGB_GPU_MAX_HOST_THREADS,
                "fail_on_invalid_gpu_id": True,
            }
        )
    _attest_xgboost_cuda_booster(predictor.booster, cp)
    matrix = _impute_matrix(
        _ordered_matrix(feature_rows), predictor.imputer_statistics
    )
    cuda_matrix = cp.asarray(matrix)
    with _reject_xgboost_device_warnings():
        raw_probabilities = predictor.booster.inplace_predict(
            cuda_matrix,
            validate_features=False,
        )
    _attest_xgboost_cuda_booster(predictor.booster, cp)
    if not hasattr(raw_probabilities, "__cuda_array_interface__"):
        raise RuntimeError("portable XGBoost CUDA prediction returned a host array")
    probabilities = np.asarray(
        cp.asnumpy(raw_probabilities),
        dtype=np.float32,
    ).reshape(-1)
    if len(probabilities) != len(matrix) or np.any(~np.isfinite(probabilities)):
        raise RuntimeError("portable predictor returned invalid probabilities")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise RuntimeError("portable predictor returned values outside [0,1]")
    return probabilities


def verify_probability_parity(
    classifier: Any,
    imputer_statistics: Sequence[Any],
    feature_rows: Any,
    *,
    classifier_ubj: bytes | None = None,
    absolute_tolerance: float = 1e-7,
    device: str = "cuda",
) -> dict[str, Any]:
    """Prove raw-UBJ predictions match the fitted in-memory classifier."""

    resolved_device = _normalize_device(device)

    import numpy as np
    import xgboost as xgb

    if absolute_tolerance != 1e-7:
        raise ValueError("canonical probability parity tolerance is sealed at 1e-7")
    cp = _require_xgboost_cuda(xgb)
    record = canonical_imputer_record(imputer_statistics)
    medians = np.asarray(
        [record["statistics"][name] for name in CANONICAL_FEATURE_ORDER],
        dtype=np.float32,
    )
    matrix = _impute_matrix(_ordered_matrix(feature_rows), medians)
    booster = (
        classifier.get_booster()
        if callable(getattr(classifier, "get_booster", None))
        else classifier
    )
    if tuple(getattr(booster, "feature_names", None) or ()) != CANONICAL_FEATURE_ORDER:
        raise ValueError("in-memory canonical classifier feature schema changed")
    with _reject_xgboost_device_warnings():
        booster.set_param(
            {
                "device": resolved_device,
                "nthread": CANONICAL_XGB_GPU_MAX_HOST_THREADS,
                "fail_on_invalid_gpu_id": True,
            }
        )
    _attest_xgboost_cuda_booster(booster, cp)
    cuda_matrix = cp.asarray(matrix)
    with _reject_xgboost_device_warnings():
        raw_expected = booster.inplace_predict(
            cuda_matrix,
            validate_features=False,
        )
    _attest_xgboost_cuda_booster(booster, cp)
    if not hasattr(raw_expected, "__cuda_array_interface__"):
        raise RuntimeError("in-memory XGBoost CUDA prediction returned a host array")
    expected = np.asarray(cp.asnumpy(raw_expected), dtype=np.float32).reshape(-1)
    if len(expected) != len(matrix) or np.any(~np.isfinite(expected)):
        raise RuntimeError("in-memory classifier returned invalid probabilities")
    payload = classifier_ubj or portable_classifier_bytes(classifier)
    loaded = load_portable_predictor(
        payload,
        imputer_statistics,
        device=resolved_device,
    )
    observed = predict_portable_probabilities(loaded, matrix)
    difference = np.abs(expected - observed)
    maximum = float(difference.max()) if len(difference) else 0.0
    if maximum > absolute_tolerance:
        raise RuntimeError(
            f"canonical UBJ probability parity failed: max_abs={maximum:.9g}"
        )
    return {
        "schema": "compag-curation-canonical-probability-parity/v1",
        "status": "PASS",
        "row_count": len(matrix),
        "absolute_tolerance": absolute_tolerance,
        "maximum_absolute_difference": maximum,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        "decision_threshold": CANONICAL_DECISION_THRESHOLD,
        "device": resolved_device,
    }


__all__ = [
    "CanonicalFeatureStatePayloads",
    "PortableCanonicalPredictor",
    "canonical_imputer_record",
    "deserialize_canonical_feature_state",
    "load_portable_predictor",
    "portable_classifier_bytes",
    "predict_portable_probabilities",
    "serialize_canonical_feature_state",
    "verify_probability_parity",
]
