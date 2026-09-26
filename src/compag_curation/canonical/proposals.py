"""Full-mask proposal handling for the canonical AMG-only profile."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .spec import (
    CANONICAL_ARCHIVED_SAM_SCORE_DECIMALS,
    CANONICAL_GPU_PROFILE,
    CanonicalAMGSettings,
    amg_settings_for_profile,
    inference_amg_execution_points_per_batch,
    stage20_amg_execution_points_per_batch,
)


CANONICAL_MAX_PROPOSALS_PER_TILE = 500


@dataclass(frozen=True)
class CanonicalProposal:
    source_index: int
    proposal_index: int
    mask: Any
    mask_sha256: str
    area: int
    bbox: tuple[int, int, int, int]
    predicted_iou: float
    stability_score: float


@dataclass(frozen=True)
class CanonicalProposalSelection:
    proposals: tuple[CanonicalProposal, ...]
    input_count: int
    empty_count: int
    duplicate_count: int
    capped_count: int


def canonical_mask_sha256(mask: Any) -> str:
    import numpy as np

    binary = np.asarray(mask) > 0
    if binary.ndim != 2 or binary.size == 0:
        raise ValueError("SAM2 proposal mask must be a nonempty rank-2 array")
    height, width = binary.shape
    packed = np.packbits(binary.reshape(-1), bitorder="little").tobytes()
    payload = (
        b"compag-canonical-full-mask-v1\0"
        + int(height).to_bytes(4, "big")
        + int(width).to_bytes(4, "big")
        + packed
    )
    return hashlib.sha256(payload).hexdigest()


def _full_mask_bbox(binary: Any) -> tuple[int, int, int, int]:
    import numpy as np

    ys, xs = np.nonzero(binary)
    if not len(xs):
        raise ValueError("SAM2 proposal mask is empty")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _sam_score(value: Any, role: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SAM2 returned a nonnumeric {role}") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"SAM2 returned {role} outside [0,1]")
    return number


def _archived_sam_score(value: float) -> float:
    """Canonicalize selected confidence evidence without changing selection."""

    rounded = round(value, CANONICAL_ARCHIVED_SAM_SCORE_DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def _mask_iou(left: Any, right: Any) -> float:
    import numpy as np

    left_binary = np.asarray(left, dtype=np.uint8) > 0
    right_binary = np.asarray(right, dtype=np.uint8) > 0
    if left_binary.shape != right_binary.shape:
        raise ValueError("SAM2 proposal masks have inconsistent shapes")
    intersection = int(np.logical_and(left_binary, right_binary).sum())
    union = int(np.logical_or(left_binary, right_binary).sum())
    return float(intersection) / float(union + 1e-6)


def select_canonical_proposals(
    annotations: Iterable[Mapping[str, Any]],
) -> CanonicalProposalSelection:
    """Apply the frozen single-scale AMG score/IoU merge and 500-mask cap."""

    import numpy as np

    settings = CanonicalAMGSettings()
    candidates: list[CanonicalProposal] = []
    input_count = empty_count = duplicate_count = capped_count = 0
    for source_index, annotation in enumerate(annotations):
        input_count += 1
        if not isinstance(annotation, Mapping) or "segmentation" not in annotation:
            raise ValueError("SAM2 returned a malformed proposal")
        mask = np.asarray(annotation["segmentation"])
        if mask.ndim != 2 or mask.size == 0:
            raise ValueError("SAM2 proposal mask must be a nonempty rank-2 array")
        binary = (mask > 0).astype(np.uint8, copy=False)
        area = int(binary.sum())
        if area == 0:
            empty_count += 1
            continue
        digest = canonical_mask_sha256(binary)
        preserved = np.ascontiguousarray(binary.copy())
        preserved.setflags(write=False)
        candidates.append(
            CanonicalProposal(
                source_index=source_index,
                proposal_index=0,
                mask=preserved,
                mask_sha256=digest,
                area=area,
                bbox=_full_mask_bbox(preserved),
                predicted_iou=_sam_score(
                    annotation.get("predicted_iou", 1.0), "predicted IoU"
                ),
                stability_score=_sam_score(
                    annotation.get("stability_score", 1.0), "stability score"
                ),
            )
        )
    candidates.sort(
        key=lambda proposal: (proposal.predicted_iou, proposal.stability_score),
        reverse=True,
    )
    selected_candidates: list[CanonicalProposal] = []
    for candidate in candidates:
        if any(
            _mask_iou(candidate.mask, kept.mask) >= settings.merge_iou_threshold
            for kept in selected_candidates
        ):
            duplicate_count += 1
            continue
        if len(selected_candidates) >= settings.max_masks_per_tile:
            capped_count += 1
            continue
        selected_candidates.append(candidate)
    selected = tuple(
        CanonicalProposal(
            source_index=candidate.source_index,
            proposal_index=proposal_index,
            mask=candidate.mask,
            mask_sha256=candidate.mask_sha256,
            area=candidate.area,
            bbox=candidate.bbox,
            predicted_iou=_archived_sam_score(candidate.predicted_iou),
            stability_score=_archived_sam_score(candidate.stability_score),
        )
        for proposal_index, candidate in enumerate(selected_candidates, start=1)
    )
    return CanonicalProposalSelection(
        proposals=selected,
        input_count=input_count,
        empty_count=empty_count,
        duplicate_count=duplicate_count,
        capped_count=capped_count,
    )


def build_canonical_amg(
    model: Any,
    *,
    profile: str = CANONICAL_GPU_PROFILE,
    execution_points_per_batch: int | None = None,
) -> Any:
    """Construct the profile-bound SAM2 AMG with a sealed microbatch."""

    from .model_loading import _attest_canonical_sam2_cuda_model

    _attest_canonical_sam2_cuda_model(model)
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    settings = amg_settings_for_profile(profile)
    if execution_points_per_batch is None:
        points_per_batch = settings.points_per_batch
    else:
        allowed = {
            settings.points_per_batch,
            inference_amg_execution_points_per_batch(profile),
            stage20_amg_execution_points_per_batch(profile),
        }
        if (
            type(execution_points_per_batch) is not int
            or execution_points_per_batch not in allowed
        ):
            raise ValueError("profile AMG execution microbatch is unsupported")
        points_per_batch = execution_points_per_batch
    generator_kwargs = settings.generator_kwargs()
    generator_kwargs["points_per_batch"] = points_per_batch
    generator_type = _memory_bounded_sam2_amg_type(SAM2AutomaticMaskGenerator)
    generator = generator_type(
        model,
        **generator_kwargs,
        output_mode="binary_mask",
    )
    _attest_canonical_amg_cuda(generator, expected_model=model)
    return generator


def _memory_bounded_sam2_amg_type(upstream_type: type[Any]) -> type[Any]:
    """Lazily specialize SAM2 AMG without importing the optional dependency.

    Pinned SAM2 keeps ``low_res_masks`` in every returned non-M2M batch even
    though no later operation reads them.  Removing only that dead tensor
    before upstream ``MaskData.cat`` prevents quadratic CUDA accumulation while
    leaving the canonical 512-prompt method and all selected-mask data intact.
    """

    class _CanonicalMemoryBoundedSAM2AMG(upstream_type):
        def _process_batch(self, *args: Any, **kwargs: Any) -> Any:
            data = super()._process_batch(*args, **kwargs)
            if getattr(self, "use_m2m", None) is False:
                keys = {key for key, _value in data.items()}
                if "low_res_masks" in keys:
                    del data["low_res_masks"]
            return data

    _CanonicalMemoryBoundedSAM2AMG.__name__ = (
        "_CanonicalMemoryBoundedSAM2AutomaticMaskGenerator"
    )
    return _CanonicalMemoryBoundedSAM2AMG


def _attest_canonical_amg_cuda(
    generator: Any,
    *,
    expected_model: Any | None = None,
) -> int:
    """Prove AMG retains the fully-attested CUDA SAM2 model and device."""

    from .model_loading import (
        _attest_canonical_sam2_cuda_model,
        _cuda_device_matches,
    )

    predictor = getattr(generator, "predictor", None)
    model = getattr(predictor, "model", None)
    if predictor is None or model is None:
        raise RuntimeError("canonical AMG does not expose its SAM2 predictor model")
    if expected_model is not None and model is not expected_model:
        raise RuntimeError("canonical AMG replaced the attested SAM2 model")
    cuda_ordinal = _attest_canonical_sam2_cuda_model(model)
    if not _cuda_device_matches(getattr(predictor, "device", None), cuda_ordinal):
        raise RuntimeError(
            f"canonical AMG predictor.device is not cuda:{cuda_ordinal}"
        )
    return cuda_ordinal


__all__ = [
    "CANONICAL_MAX_PROPOSALS_PER_TILE",
    "CanonicalProposal",
    "CanonicalProposalSelection",
    "build_canonical_amg",
    "canonical_mask_sha256",
    "select_canonical_proposals",
]
