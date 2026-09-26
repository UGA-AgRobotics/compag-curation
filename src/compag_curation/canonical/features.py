"""Canonical Ultra feature extraction with train-fitted deep state."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from compag_curation.public_io import verified_file_path

from .spec import (
    CANONICAL_EMBEDDING_BACKBONE,
    CANONICAL_EMBEDDING_DIMENSIONS,
    CANONICAL_EMBEDDING_PAD_FRACTION,
    CANONICAL_FEATURE_CROP_SCALES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_PCA_DIMENSIONS,
)


# Batching changes only execution shape, not the transform or feature math.
CANONICAL_RESNET50_CPU_BATCH_SIZE = 16
CANONICAL_RESNET50_GPU_BATCH_SIZE = 64
_MAX_CANONICAL_RESNET50_WEIGHTS_BYTES = 256 * 1024 * 1024
_GATE_FEATURES = (
    "g_quality",
    "g_light",
    "g_color",
    "g_shape",
    "g_embed",
    "g_robust",
    "g_maha",
    "g_border",
)
_PCA_FEATURES = tuple(f"embed_pca_{index}" for index in range(32))
_DERIVED_FEATURES = frozenset((*_GATE_FEATURES, *_PCA_FEATURES, "embed_sim"))
_RAW_REQUIRED_FEATURES = tuple(
    name for name in CANONICAL_FEATURE_ORDER if name not in _DERIVED_FEATURES
)
CANONICAL_RAW_FEATURE_ORDER = _RAW_REQUIRED_FEATURES
CANONICAL_RAW_FEATURE_COLUMNS = (
    "proposal_index",
    "scale",
    "predicted_iou",
    "stability_score",
    *CANONICAL_RAW_FEATURE_ORDER,
    "embedding_encoding",
    "embedding_dimensions",
    "embedding_f32le_base64",
)


class BatchEmbedder(Protocol):
    def embed_many(self, patches_bgr: Sequence[Any]) -> Any: ...


@dataclass(frozen=True)
class CanonicalGridContext:
    row_lines: tuple[int, ...]
    column_lines: tuple[int, ...]
    tile_x: int = 0
    tile_y: int = 0

    def __post_init__(self) -> None:
        if tuple(sorted(self.row_lines)) != self.row_lines:
            raise ValueError("canonical row lines must be sorted")
        if tuple(sorted(self.column_lines)) != self.column_lines:
            raise ValueError("canonical column lines must be sorted")
        if self.tile_x < 0 or self.tile_y < 0:
            raise ValueError("canonical tile offsets must be nonnegative")


@dataclass(frozen=True)
class CanonicalRawFeature:
    proposal_index: int
    scale: float
    predicted_iou: float
    stability_score: float
    values: Mapping[str, float]
    embedding: Any


@dataclass(frozen=True)
class CanonicalFeatureState:
    prototype: Any
    pca_components: Any
    pca_mean: Any
    training_row_count: int
    positive_row_count: int


class CanonicalResNet50Embedder:
    """Exact masked-crop transform with fail-closed CUDA execution."""

    def __init__(
        self,
        model: Any,
        transform: Callable[[Any], Any],
        *,
        device: str = "cuda",
        batch_size: int | None = None,
    ) -> None:
        normalized_device = str(device).strip().lower()
        if normalized_device != "cuda":
            raise ValueError(
                "canonical ResNet50 execution is CUDA-only; CPU execution is unsupported"
            )

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "canonical ResNet50 CUDA was requested, but PyTorch cannot access a CUDA device"
            )
        cuda_ordinal = int(torch.cuda.current_device())
        if batch_size is None:
            batch_size = CANONICAL_RESNET50_GPU_BATCH_SIZE
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("canonical ResNet50 batch size must be a positive integer")
        to_float = getattr(model, "float", None)
        if callable(to_float):
            model = to_float()
        explicit_device = f"cuda:{cuda_ordinal}"
        self._model = model.to(explicit_device).eval()
        self._transform = transform
        self.device = normalized_device
        self._cuda_ordinal = cuda_ordinal
        self.batch_size = batch_size
        _attest_cuda_module(
            self._model,
            torch,
            cuda_ordinal=cuda_ordinal,
            role="canonical ResNet50",
        )

    @classmethod
    def from_local_weights(
        cls,
        weights_path: Path,
        expected_sha256: str,
        *,
        device: str = "cuda",
        batch_size: int | None = None,
    ) -> "CanonicalResNet50Embedder":
        normalized_device = str(device).strip().lower()
        if normalized_device != "cuda":
            raise ValueError(
                "canonical ResNet50 execution is CUDA-only; CPU execution is unsupported"
            )
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError("ResNet50 weights require a lowercase SHA-256")
        from compag_curation.proposals.sam2_pipeline.gate_core import build_embed_model

        with verified_file_path(
            weights_path,
            expected_sha256,
            max_bytes=_MAX_CANONICAL_RESNET50_WEIGHTS_BYTES,
            require_single_link=True,
        ) as (weights_fd_path, _weights_snapshot):
            with weights_fd_path.open("rb") as weights_file:
                model, transform = build_embed_model(
                    CANONICAL_EMBEDDING_BACKBONE, weights_file
                )
        return cls(model, transform, device=device, batch_size=batch_size)

    def embed_many(self, patches_bgr: Sequence[Any]) -> Any:
        self._attest_cuda_state()
        import cv2
        import numpy as np
        import torch

        if not patches_bgr:
            return np.empty((0, CANONICAL_EMBEDDING_DIMENSIONS), dtype=np.float32)
        outputs: list[Any] = []
        with torch.inference_mode():
            for start in range(0, len(patches_bgr), self.batch_size):
                batch = patches_bgr[start : start + self.batch_size]
                tensors = []
                for patch in batch:
                    array = np.asarray(patch)
                    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
                        raise ValueError("masked embedding patch must be uint8 BGR")
                    rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                    tensors.append(self._transform(rgb))
                inputs = torch.stack(tensors, dim=0).to(
                    device=f"cuda:{self._cuda_ordinal}",
                    dtype=torch.float32,
                    non_blocking=True,
                )
                _attest_cuda_float32_tensor(
                    inputs,
                    torch,
                    cuda_ordinal=self._cuda_ordinal,
                    role="canonical ResNet50 input batch",
                )
                raw_vectors = self._model(inputs)
                _attest_cuda_float32_tensor(
                    raw_vectors,
                    torch,
                    cuda_ordinal=self._cuda_ordinal,
                    role="canonical ResNet50 output batch",
                )
                vectors = raw_vectors.detach().cpu().numpy()
                vectors = np.asarray(vectors, dtype=np.float32).reshape(len(batch), -1)
                if vectors.shape[1] != CANONICAL_EMBEDDING_DIMENSIONS:
                    raise RuntimeError("ResNet50 returned an unexpected embedding width")
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
                    raise RuntimeError("ResNet50 returned an invalid embedding")
                outputs.append((vectors / (norms + 1e-12)).astype(np.float32))
        result = np.vstack(outputs).astype(np.float32)
        self._attest_cuda_state()
        return result

    def embed_one_scalar(self, patch_bgr: Any) -> Any:
        vector = self.embed_many((patch_bgr,))[0]
        if len(vector) != CANONICAL_EMBEDDING_DIMENSIONS:
            raise RuntimeError("ResNet50 returned an unexpected embedding width")
        return vector

    def _attest_cuda_state(self) -> None:
        """Re-attest the model immediately before a production stage executes."""

        import torch

        _attest_cuda_module(
            self._model,
            torch,
            cuda_ordinal=self._cuda_ordinal,
            role="canonical ResNet50",
        )


def _cuda_device_matches(value: Any, cuda_ordinal: int) -> bool:
    device = getattr(value, "device", value)
    return (
        getattr(device, "type", None) == "cuda"
        and getattr(device, "index", None) == cuda_ordinal
    )


def _attest_cuda_float32_tensor(
    tensor: Any,
    torch: Any,
    *,
    cuda_ordinal: int,
    role: str,
) -> None:
    if not _cuda_device_matches(tensor, cuda_ordinal):
        raise RuntimeError(f"{role} is not on cuda:{cuda_ordinal}")
    is_floating = getattr(tensor, "is_floating_point", None)
    if not callable(is_floating) or not bool(is_floating()):
        raise RuntimeError(f"{role} is not floating point")
    if getattr(tensor, "dtype", None) != torch.float32:
        raise RuntimeError(f"{role} is not FP32")


def _attest_cuda_module(
    model: Any,
    torch: Any,
    *,
    cuda_ordinal: int,
    role: str,
) -> None:
    named_parameters = getattr(model, "named_parameters", None)
    named_buffers = getattr(model, "named_buffers", None)
    modules = getattr(model, "modules", None)
    if not all(callable(value) for value in (named_parameters, named_buffers, modules)):
        raise RuntimeError(f"{role} does not expose a complete Torch module state")
    parameters = tuple(named_parameters(recurse=True))
    if not parameters:
        raise RuntimeError(f"{role} has no parameters to attest")
    for name, tensor in (*parameters, *tuple(named_buffers(recurse=True))):
        if not _cuda_device_matches(tensor, cuda_ordinal):
            raise RuntimeError(f"{role} tensor is not on cuda:{cuda_ordinal}: {name}")
        is_floating = getattr(tensor, "is_floating_point", None)
        if not callable(is_floating):
            raise RuntimeError(f"{role} state is not a Torch tensor: {name}")
        if bool(is_floating()) and getattr(tensor, "dtype", None) != torch.float32:
            raise RuntimeError(f"{role} floating tensor is not FP32: {name}")
    if any(bool(getattr(module, "training", True)) for module in modules()):
        raise RuntimeError(f"{role} contains a submodule that is not in eval mode")


def _finite(value: Any, role: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"canonical feature is nonnumeric: {role}") from exc
    if not math.isfinite(number):
        raise ValueError(f"canonical feature is nonfinite: {role}")
    return number


def _scaled_context(image_bgr: Any, scale: float) -> tuple[Any, Any, float, float, Any]:
    import cv2
    import numpy as np
    from compag_curation.proposals.sam2_pipeline.gate_core import yellow_bg_stats

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("canonical features require a uint8 BGR tile")
    value = float(scale)
    if value not in CANONICAL_FEATURE_CROP_SCALES:
        raise ValueError("canonical export scale is not supported")
    scaled = (
        image.copy()
        if abs(value - 1.0) < 1e-4
        else cv2.resize(image, None, fx=value, fy=value, interpolation=cv2.INTER_LINEAR)
    )
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    background_a, background_b, background_bgr = yellow_bg_stats(scaled)
    return scaled, lab, float(background_a), float(background_b), background_bgr


def _raw_values_and_patch(
    scaled_bgr: Any,
    scaled_lab: Any,
    source_mask: Any,
    *,
    scale: float,
    background_a: float,
    background_b: float,
    background_bgr: Any,
    grid: CanonicalGridContext,
) -> tuple[dict[str, float], Any]:
    import cv2
    import numpy as np
    from compag_curation.proposals.sam2_pipeline.gate_core import (
        cell_of_point,
        compute_features,
        crop_masked_patch,
    )

    mask = np.asarray(source_mask)
    if mask.ndim != 2 or mask.size == 0 or not np.any(mask > 0):
        raise ValueError("canonical proposal mask must be nonempty and rank 2")
    scaled_mask = (
        (mask > 0).astype(np.uint8)
        if abs(scale - 1.0) < 1e-4
        else (
            cv2.resize(
                (mask > 0).astype(np.uint8),
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        ).astype(np.uint8)
    )
    if scaled_mask.shape != scaled_bgr.shape[:2]:
        raise ValueError("proposal mask dimensions do not match its tile")
    values = dict(compute_features(scaled_mask * 255, scaled_lab, mode="ultra"))
    if not values:
        raise ValueError("canonical Ultra feature extraction returned no values")
    values["delta_a"] = background_a - _finite(values["mean_a"], "mean_a")
    values["delta_b"] = _finite(values["mean_b"], "mean_b") - background_b

    # Cell lookup is performed in the warped-card coordinate system.  Feature
    # geometry itself remains tile-local, matching the historical table.
    global_x = _finite(values["cx"], "cx") / scale + grid.tile_x
    global_y = _finite(values["cy"], "cy") / scale + grid.tile_y
    grid_r, grid_c = cell_of_point(
        global_x,
        global_y,
        list(grid.row_lines),
        list(grid.column_lines),
    )
    values["grid_r"] = float(grid_r)
    values["grid_c"] = float(grid_c)
    values["grid_r_norm"] = float(grid_r) / max(1, len(grid.row_lines))
    values["grid_c_norm"] = float(grid_c) / max(1, len(grid.column_lines))
    missing = [name for name in _RAW_REQUIRED_FEATURES if name not in values]
    if missing:
        raise ValueError(
            "canonical Ultra extraction omitted required raw features: "
            + ", ".join(missing)
        )
    normalized = {name: _finite(values[name], name) for name in _RAW_REQUIRED_FEATURES}
    patch = crop_masked_patch(
        scaled_bgr,
        scaled_mask,
        pad_frac=CANONICAL_EMBEDDING_PAD_FRACTION,
        bg_bgr=background_bgr,
    )
    if patch is None:
        raise ValueError("canonical masked crop is empty")
    return normalized, patch


def extract_canonical_raw_features(
    image_bgr: Any,
    proposals: Sequence[Any],
    embedder: BatchEmbedder,
    *,
    grid: CanonicalGridContext,
) -> tuple[CanonicalRawFeature, ...]:
    """Extract all proposal/scale rows and embed their crops in device batches."""

    contexts = {
        scale: _scaled_context(image_bgr, scale)
        for scale in CANONICAL_FEATURE_CROP_SCALES
    }
    records: list[tuple[int, float, float, float, dict[str, float]]] = []
    patches: list[Any] = []
    for ordinal, proposal in enumerate(proposals, start=1):
        mask = getattr(proposal, "mask", None)
        if mask is None:
            raise ValueError("canonical proposal is missing its full mask")
        proposal_index = int(getattr(proposal, "proposal_index", ordinal))
        predicted_iou = _finite(
            getattr(proposal, "predicted_iou", None), "predicted_iou"
        )
        stability_score = _finite(
            getattr(proposal, "stability_score", None), "stability_score"
        )
        if not 0.0 <= predicted_iou <= 1.0 or not 0.0 <= stability_score <= 1.0:
            raise ValueError("canonical SAM2 confidence is outside [0,1]")
        for scale in CANONICAL_FEATURE_CROP_SCALES:
            scaled, lab, bg_a, bg_b, bg_bgr = contexts[scale]
            values, patch = _raw_values_and_patch(
                scaled,
                lab,
                mask,
                scale=scale,
                background_a=bg_a,
                background_b=bg_b,
                background_bgr=bg_bgr,
                grid=grid,
            )
            records.append(
                (proposal_index, scale, predicted_iou, stability_score, values)
            )
            patches.append(patch)
    embeddings = embedder.embed_many(patches)
    if getattr(embeddings, "shape", None) != (
        len(records),
        CANONICAL_EMBEDDING_DIMENSIONS,
    ):
        raise RuntimeError("batched ResNet50 output does not align with feature rows")
    output: list[CanonicalRawFeature] = []
    for record, embedding in zip(records, embeddings, strict=True):
        proposal_index, scale, predicted_iou, stability_score, values = record
        output.append(
            CanonicalRawFeature(
                proposal_index=proposal_index,
                scale=scale,
                predicted_iou=predicted_iou,
                stability_score=stability_score,
                values=values,
                embedding=embedding,
            )
        )
    return tuple(output)


def fit_canonical_feature_state(
    raw_features: Sequence[CanonicalRawFeature],
    labels: Sequence[int],
    training_indices: Sequence[int],
) -> CanonicalFeatureState:
    """Fit prototype from train positives and PCA32 from train rows only."""

    import numpy as np
    from compag_curation.features.extraction import _l2_normalize
    from compag_curation.proposals.sam2_pipeline.pca_utils import _compute_pca

    if len(raw_features) != len(labels):
        raise ValueError("raw feature rows and labels do not align")
    indices = np.asarray(training_indices, dtype=int)
    if indices.ndim != 1 or len(indices) < CANONICAL_PCA_DIMENSIONS:
        raise ValueError("canonical PCA32 requires at least 32 training rows")
    if np.any(indices < 0) or np.any(indices >= len(raw_features)):
        raise ValueError("canonical feature-state training indices are invalid")
    embeddings = np.vstack(
        [np.asarray(raw_features[index].embedding, dtype=np.float32) for index in indices]
    )
    if embeddings.shape[1] != CANONICAL_EMBEDDING_DIMENSIONS:
        raise ValueError("canonical feature state requires 2048-wide ResNet50 embeddings")
    if np.any(~np.isfinite(embeddings)):
        raise ValueError("canonical training embeddings contain nonfinite values")
    embeddings = _l2_normalize(embeddings, axis=1)
    train_labels = np.asarray(labels, dtype=int)[indices]
    if set(np.unique(train_labels).tolist()) - {0, 1}:
        raise ValueError("canonical feature-state labels must be binary")
    positives = embeddings[train_labels == 1]
    if len(positives) < 1:
        raise ValueError("canonical prototype requires training-positive embeddings")
    prototype = _l2_normalize(positives.mean(axis=0)).astype(np.float32)
    components, mean = _compute_pca(embeddings, CANONICAL_PCA_DIMENSIONS)
    if components.shape != (
        CANONICAL_PCA_DIMENSIONS,
        CANONICAL_EMBEDDING_DIMENSIONS,
    ):
        raise ValueError("canonical training rows cannot support exact PCA32")
    prototype = np.ascontiguousarray(prototype, dtype=np.float32)
    components = np.ascontiguousarray(components, dtype=np.float32)
    mean = np.ascontiguousarray(mean, dtype=np.float32)
    for value in (prototype, components, mean):
        value.setflags(write=False)
    return CanonicalFeatureState(
        prototype=prototype,
        pca_components=components,
        pca_mean=mean,
        training_row_count=len(indices),
        positive_row_count=len(positives),
    )


def validate_canonical_feature_state(state: CanonicalFeatureState) -> None:
    import numpy as np

    prototype = np.asarray(state.prototype, dtype=np.float32)
    components = np.asarray(state.pca_components, dtype=np.float32)
    mean = np.asarray(state.pca_mean, dtype=np.float32)
    if prototype.shape != (CANONICAL_EMBEDDING_DIMENSIONS,):
        raise ValueError("canonical prototype has an invalid shape")
    if components.shape != (
        CANONICAL_PCA_DIMENSIONS,
        CANONICAL_EMBEDDING_DIMENSIONS,
    ):
        raise ValueError("canonical PCA components have an invalid shape")
    if mean.shape != (CANONICAL_EMBEDDING_DIMENSIONS,):
        raise ValueError("canonical PCA mean has an invalid shape")
    if any(np.any(~np.isfinite(value)) for value in (prototype, components, mean)):
        raise ValueError("canonical feature state contains nonfinite values")
    norm = float(np.linalg.norm(prototype))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("canonical prototype must be L2-normalized")


def finalize_canonical_feature(
    raw: CanonicalRawFeature,
    state: CanonicalFeatureState,
) -> dict[str, float]:
    """Produce one complete exact-order 93-feature record without zero-fill."""

    import numpy as np
    from compag_curation.features.extraction import _feature_gate_values, _l2_normalize
    from compag_curation.proposals.sam2_pipeline.gate_core import project_embed_pca

    validate_canonical_feature_state(state)
    missing_raw = [name for name in _RAW_REQUIRED_FEATURES if name not in raw.values]
    if missing_raw:
        raise ValueError(
            "canonical raw feature record is incomplete: " + ", ".join(missing_raw)
        )
    values = {name: _finite(raw.values[name], name) for name in _RAW_REQUIRED_FEATURES}
    embedding = _l2_normalize(np.asarray(raw.embedding, dtype=np.float32)).reshape(-1)
    if embedding.shape != (CANONICAL_EMBEDDING_DIMENSIONS,):
        raise ValueError("canonical raw embedding has an invalid shape")
    cosine = float(embedding @ np.asarray(state.prototype, dtype=np.float32))
    values["embed_sim"] = max(0.0, min(1.0, (cosine + 1.0) * 0.5))
    projection = project_embed_pca(
        embedding,
        {
            "components": state.pca_components,
            "mean": state.pca_mean,
        },
    )
    if getattr(projection, "shape", None) != (CANONICAL_PCA_DIMENSIONS,):
        raise RuntimeError("canonical PCA projection is not 32-dimensional")
    for index, value in enumerate(projection):
        values[f"embed_pca_{index}"] = _finite(value, f"embed_pca_{index}")
    values.update(
        _feature_gate_values(values, projection.tolist(), has_prototype=True)
    )
    missing = [name for name in CANONICAL_FEATURE_ORDER if name not in values]
    if missing:
        raise ValueError(
            "canonical feature generation omitted required predictors: "
            + ", ".join(missing)
        )
    ordered = {name: _finite(values[name], name) for name in CANONICAL_FEATURE_ORDER}
    if tuple(ordered) != CANONICAL_FEATURE_ORDER or len(ordered) != 93:
        raise RuntimeError("canonical feature order changed during finalization")
    return ordered


def canonical_feature_matrix(
    raw_features: Sequence[CanonicalRawFeature],
    state: CanonicalFeatureState,
) -> Any:
    import numpy as np

    rows = [finalize_canonical_feature(raw, state) for raw in raw_features]
    if not rows:
        return np.empty((0, len(CANONICAL_FEATURE_ORDER)), dtype=np.float32)
    return np.asarray(
        [[row[name] for name in CANONICAL_FEATURE_ORDER] for row in rows],
        dtype=np.float32,
    )


def encode_canonical_embedding(embedding: Any) -> str:
    """Encode one normalized ResNet vector losslessly for a portable CSV cell."""

    import numpy as np

    vector = np.asarray(embedding, dtype="<f4").reshape(-1)
    if vector.shape != (CANONICAL_EMBEDDING_DIMENSIONS,) or np.any(
        ~np.isfinite(vector)
    ):
        raise ValueError("canonical CSV embedding must contain 2048 finite float32 values")
    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("canonical CSV embedding must be L2-normalized")
    return base64.b64encode(vector.tobytes(order="C")).decode("ascii")


def decode_canonical_embedding(encoded: str) -> Any:
    import binascii
    import numpy as np

    if not isinstance(encoded, str) or not encoded:
        raise ValueError("canonical CSV embedding is missing")
    try:
        payload = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeError, binascii.Error) as exc:
        raise ValueError("canonical CSV embedding is not strict base64") from exc
    if len(payload) != CANONICAL_EMBEDDING_DIMENSIONS * 4:
        raise ValueError("canonical CSV embedding byte length is invalid")
    vector = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
    if np.any(~np.isfinite(vector)):
        raise ValueError("canonical CSV embedding contains nonfinite values")
    norm = float(np.linalg.norm(vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("canonical CSV embedding is not L2-normalized")
    vector.setflags(write=False)
    return vector


def canonical_raw_feature_csv_row(raw: CanonicalRawFeature) -> dict[str, Any]:
    missing = [name for name in CANONICAL_RAW_FEATURE_ORDER if name not in raw.values]
    if missing:
        raise ValueError(
            "canonical raw CSV row is incomplete: " + ", ".join(missing)
        )
    row: dict[str, Any] = {
        "proposal_index": int(raw.proposal_index),
        "scale": _finite(raw.scale, "scale"),
        "predicted_iou": _finite(raw.predicted_iou, "predicted_iou"),
        "stability_score": _finite(raw.stability_score, "stability_score"),
    }
    row.update(
        {name: _finite(raw.values[name], name) for name in CANONICAL_RAW_FEATURE_ORDER}
    )
    row.update(
        {
            "embedding_encoding": "base64-float32-little-endian-v1",
            "embedding_dimensions": CANONICAL_EMBEDDING_DIMENSIONS,
            "embedding_f32le_base64": encode_canonical_embedding(raw.embedding),
        }
    )
    if tuple(row) != CANONICAL_RAW_FEATURE_COLUMNS:
        raise RuntimeError("canonical raw CSV column order changed")
    return row


__all__ = [
    "CANONICAL_RESNET50_CPU_BATCH_SIZE",
    "CANONICAL_RESNET50_GPU_BATCH_SIZE",
    "CANONICAL_RAW_FEATURE_COLUMNS",
    "CANONICAL_RAW_FEATURE_ORDER",
    "BatchEmbedder",
    "CanonicalFeatureState",
    "CanonicalGridContext",
    "CanonicalRawFeature",
    "CanonicalResNet50Embedder",
    "canonical_feature_matrix",
    "canonical_raw_feature_csv_row",
    "decode_canonical_embedding",
    "encode_canonical_embedding",
    "extract_canonical_raw_features",
    "finalize_canonical_feature",
    "fit_canonical_feature_state",
    "validate_canonical_feature_state",
]
