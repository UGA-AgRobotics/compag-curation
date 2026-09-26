"""ROI-first Ultra features for the non-tiled full-image profile.

This module intentionally has a different scientific identity from the
historical 512x512 feature extractor.  It never materializes an HxW mask per
proposal and never scans the complete image once per proposal.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .features import (
    CANONICAL_RAW_FEATURE_ORDER,
    BatchEmbedder,
    CanonicalGridContext,
    CanonicalRawFeature,
)
from .spec import (
    CANONICAL_EMBEDDING_DIMENSIONS,
    CANONICAL_EMBEDDING_PAD_FRACTION,
    CANONICAL_FEATURE_CROP_SCALES,
)


FULL_IMAGE_FEATURE_SEMANTICS_ID = (
    "compag-full-image-roi-ultra-features-v1"
)
FULL_IMAGE_FEATURE_PROPOSAL_BATCH = 64
FULL_IMAGE_CONTEXT_FRACTION = 0.15
FULL_IMAGE_CONTEXT_MINIMUM_PIXELS = 8


def _finite(value: Any, role: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"full-image feature is nonnumeric: {role}") from exc
    if not math.isfinite(number):
        raise ValueError(f"full-image feature is nonfinite: {role}")
    return number


def _proposal_geometry(proposal: Any, image_shape: tuple[int, int]) -> tuple[int, int, int, int, Any]:
    import numpy as np

    raw_bbox = getattr(proposal, "bbox", None)
    if (
        not isinstance(raw_bbox, tuple)
        or len(raw_bbox) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_bbox)
    ):
        raise ValueError("full-image proposal has an invalid source-frame bbox")
    x, y, width, height = raw_bbox
    image_height, image_width = image_shape
    if (
        x < 0
        or y < 0
        or width < 1
        or height < 1
        or x + width > image_width
        or y + height > image_height
    ):
        raise ValueError("full-image proposal bbox exceeds its analysis frame")
    mask = np.asarray(getattr(proposal, "mask_crop", None))
    if mask.shape != (height, width) or mask.size == 0:
        raise ValueError("full-image proposal mask crop/bbox closure is invalid")
    binary = np.ascontiguousarray(mask > 0, dtype=np.uint8)
    if not bool(binary.any()):
        raise ValueError("full-image proposal mask crop is empty")
    return x, y, width, height, binary


def _roi_values_and_patch(
    image_bgr: Any,
    proposal: Any,
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

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("full-image features require a uint8 BGR frame")
    image_height, image_width = (int(value) for value in image.shape[:2])
    x, y, width, height, mask_crop = _proposal_geometry(
        proposal, (image_height, image_width)
    )
    padding = max(
        FULL_IMAGE_CONTEXT_MINIMUM_PIXELS,
        int(math.ceil(max(width, height) * FULL_IMAGE_CONTEXT_FRACTION)),
    )
    roi_x0 = max(0, x - padding)
    roi_y0 = max(0, y - padding)
    roi_x1 = min(image_width, x + width + padding)
    roi_y1 = min(image_height, y + height + padding)
    roi_bgr = np.ascontiguousarray(image[roi_y0:roi_y1, roi_x0:roi_x1])
    roi_mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    local_x = x - roi_x0
    local_y = y - roi_y0
    roi_mask[local_y : local_y + height, local_x : local_x + width] = mask_crop

    value = float(scale)
    if value not in CANONICAL_FEATURE_CROP_SCALES:
        raise ValueError("full-image export scale is unsupported")
    if abs(value - 1.0) < 1e-4:
        scaled_bgr = roi_bgr
        scaled_mask = roi_mask
    else:
        scaled_bgr = cv2.resize(
            roi_bgr, None, fx=value, fy=value, interpolation=cv2.INTER_LINEAR
        )
        scaled_mask = (
            cv2.resize(
                roi_mask, None, fx=value, fy=value, interpolation=cv2.INTER_NEAREST
            )
            > 0
        ).astype(np.uint8)
    scaled_lab = cv2.cvtColor(scaled_bgr, cv2.COLOR_BGR2LAB)
    values = dict(compute_features(scaled_mask * 255, scaled_lab, mode="ultra"))
    if not values:
        raise ValueError("full-image Ultra extraction returned no values")

    scaled_full_width = max(1, int(round(image_width * value)))
    scaled_full_height = max(1, int(round(image_height * value)))
    scaled_roi_x0 = int(round(roi_x0 * value))
    scaled_roi_y0 = int(round(roi_y0 * value))
    global_x = _finite(values["bbox_x"], "bbox_x") + scaled_roi_x0
    global_y = _finite(values["bbox_y"], "bbox_y") + scaled_roi_y0
    global_cx = _finite(values["cx"], "cx") + scaled_roi_x0
    global_cy = _finite(values["cy"], "cy") + scaled_roi_y0
    bbox_width = _finite(values["bbox_w"], "bbox_w")
    bbox_height = _finite(values["bbox_h"], "bbox_h")
    area = _finite(values.get("area_px"), "area_px")
    full_diag = math.hypot(scaled_full_width, scaled_full_height)
    values.update(
        {
            "area_norm": area / float(scaled_full_width * scaled_full_height),
            "bbox_x": global_x,
            "bbox_y": global_y,
            "bbox_w": bbox_width,
            "bbox_h": bbox_height,
            "cx": global_cx,
            "cy": global_cy,
            "bbox_diag_frac": math.hypot(bbox_width, bbox_height)
            / (full_diag + 1e-6),
            "scale_diag": full_diag,
            "touching_border": float(
                global_x <= 0
                or global_y <= 0
                or global_x + bbox_width >= scaled_full_width - 1
                or global_y + bbox_height >= scaled_full_height - 1
            ),
            "delta_a": background_a - _finite(values["mean_a"], "mean_a"),
            "delta_b": _finite(values["mean_b"], "mean_b") - background_b,
        }
    )
    native_cx = global_cx / value
    native_cy = global_cy / value
    grid_r, grid_c = cell_of_point(
        native_cx,
        native_cy,
        list(grid.row_lines),
        list(grid.column_lines),
    )
    values["grid_r"] = float(grid_r)
    values["grid_c"] = float(grid_c)
    values["grid_r_norm"] = float(grid_r) / max(1, len(grid.row_lines))
    values["grid_c_norm"] = float(grid_c) / max(1, len(grid.column_lines))

    missing = [name for name in CANONICAL_RAW_FEATURE_ORDER if name not in values]
    if missing:
        raise ValueError(
            "full-image Ultra extraction omitted required raw features: "
            + ", ".join(missing)
        )
    normalized = {
        name: _finite(values[name], name) for name in CANONICAL_RAW_FEATURE_ORDER
    }
    patch = crop_masked_patch(
        scaled_bgr,
        scaled_mask,
        pad_frac=CANONICAL_EMBEDDING_PAD_FRACTION,
        bg_bgr=background_bgr,
    )
    if patch is None:
        raise ValueError("full-image masked crop is empty")
    return normalized, patch


def extract_full_image_raw_features(
    image_bgr: Any,
    proposals: Sequence[Any],
    embedder: BatchEmbedder,
    *,
    grid: CanonicalGridContext,
) -> tuple[CanonicalRawFeature, ...]:
    """Extract full-image features in bounded proposal batches."""

    import numpy as np
    from compag_curation.proposals.sam2_pipeline.gate_core import yellow_bg_stats

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("full-image features require a uint8 BGR frame")
    background_a, background_b, background_bgr = yellow_bg_stats(image)
    output: list[CanonicalRawFeature] = []
    for start in range(0, len(proposals), FULL_IMAGE_FEATURE_PROPOSAL_BATCH):
        batch = proposals[start : start + FULL_IMAGE_FEATURE_PROPOSAL_BATCH]
        records: list[tuple[int, float, float, float, dict[str, float]]] = []
        patches: list[Any] = []
        for ordinal, proposal in enumerate(batch, start=start + 1):
            proposal_index = int(getattr(proposal, "proposal_index", ordinal))
            predicted_iou = _finite(
                getattr(proposal, "predicted_iou", None), "predicted_iou"
            )
            stability_score = _finite(
                getattr(proposal, "stability_score", None), "stability_score"
            )
            if not 0.0 <= predicted_iou <= 1.0 or not 0.0 <= stability_score <= 1.0:
                raise ValueError("full-image SAM2 confidence is outside [0,1]")
            for scale in CANONICAL_FEATURE_CROP_SCALES:
                values, patch = _roi_values_and_patch(
                    image,
                    proposal,
                    scale=scale,
                    background_a=float(background_a),
                    background_b=float(background_b),
                    background_bgr=background_bgr,
                    grid=grid,
                )
                records.append(
                    (
                        proposal_index,
                        scale,
                        predicted_iou,
                        stability_score,
                        values,
                    )
                )
                patches.append(patch)
        embeddings = np.asarray(embedder.embed_many(patches), dtype=np.float32)
        if embeddings.shape != (len(records), CANONICAL_EMBEDDING_DIMENSIONS):
            raise RuntimeError(
                "full-image batched ResNet50 output does not align with feature rows"
            )
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


__all__ = [
    "FULL_IMAGE_CONTEXT_FRACTION",
    "FULL_IMAGE_CONTEXT_MINIMUM_PIXELS",
    "FULL_IMAGE_FEATURE_PROPOSAL_BATCH",
    "FULL_IMAGE_FEATURE_SEMANTICS_ID",
    "extract_full_image_raw_features",
]
