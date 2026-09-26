"""Process-free mask publication algorithms for report cells C16 and C17."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


MaskGrid = tuple[tuple[bool, ...], ...]


class MaskPublicationContractError(ValueError):
    """Raised when a precomputed mask violates the publication contract."""


@dataclass(frozen=True)
class PrecomputedBooleanMask:
    """One explicitly identified, already-computed boolean segmentation mask."""

    candidate_id: int
    segmentation: MaskGrid
    predicted_iou: float
    stability_score: float
    source_scale: float = 1.0


@dataclass(frozen=True)
class PrecomputedMaskGenerationSettings:
    """Source AMG settings retained as validation and provenance, not execution."""

    points_per_side: int = 64
    points_per_batch: int = 512
    predicted_iou_threshold: float = 0.80
    stability_score_threshold: float = 0.88
    crop_layers: int = 0
    crop_points_downscale_factor: int = 2
    crop_overlap_ratio: float = 0.40
    maximum_masks: int = 150
    output_mode: str = "binary_mask"
    apply_postprocessing: bool = False
    use_multiscale: bool = True
    multiscale_scales: tuple[float, ...] = (0.67, 0.80, 1.00, 1.25)
    multiscale_merge_iou: float = 0.75


@dataclass(frozen=True)
class C0016MaskPublicationSettings:
    """Exact C16 publication defaults over precomputed mask grids."""

    generation: PrecomputedMaskGenerationSettings = (
        PrecomputedMaskGenerationSettings()
    )
    overlay_alpha: float = 0.45
    crop_padding_fraction: float = 0.10
    display_top_k: int = 24
    sort_by: str = "score"
    border_margin: int = 2
    iou_epsilon: float = 1e-9
    display_columns: int = 6
    display_figure_inches: tuple[float, float] = (18.0, 10.0)


@dataclass(frozen=True)
class C0017MaskPublicationSettings:
    """Exact C17 publication defaults over precomputed mask grids."""

    generation: PrecomputedMaskGenerationSettings = (
        PrecomputedMaskGenerationSettings()
    )
    save_per_mask: bool = True
    maximum_visualization_area_fraction: float = 0.35
    single_overlay_alpha: float = 0.45
    all_masks_overlay_alpha: float = 0.50
    crop_padding_fraction: float = 0.10
    preview_top_k: int = 6
    sort_by: str = "predicted_iou"
    border_margin: int = 2
    iou_epsilon: float = 1e-9
    all_masks_seed: int = 3
    all_masks_borders: bool = True
    preview_columns: int = 3
    preview_figure_inches: tuple[float, float] = (14.0, 8.0)


@dataclass(frozen=True)
class MaskCropGeometry:
    """Inclusive source bounds and the exact padded crop rectangle."""

    bbox_xyxy: tuple[int, int, int, int]
    bbox_xywh: tuple[int, int, int, int]
    crop_bbox_xyxy: tuple[int, int, int, int]
    crop_width: int
    crop_height: int


@dataclass(frozen=True)
class PrecomputedMaskSvgRendererBoundary:
    """Explicit boundary for SVGs rendered without source-image pixels."""

    canvas_width: int
    canvas_height: int
    renderer_scope: str = "precomputed-mask-grid-only"
    includes_source_image_pixels: bool = False
    background_rgb: tuple[int, int, int] = (255, 255, 255)
    individual_fill_rgb: tuple[int, int, int] = (0, 255, 0)
    individual_outline_rgb: tuple[int, int, int] = (255, 0, 0)
    all_masks_outline_rgb: tuple[int, int, int] = (0, 0, 255)
    all_masks_palette: str = "lcg-rgb-v1"


@dataclass(frozen=True)
class PublishedMaskCandidate:
    """Computed metadata and deterministic individual rendering for one mask."""

    publication_index: int
    mask: PrecomputedBooleanMask
    area: int
    area_fraction: float
    score: float
    geometry: MaskCropGeometry
    touches_border: bool
    visualization_eligible: bool
    individual_svg: str | None


@dataclass(frozen=True)
class MaskPublicationResult:
    """Renderer-ready text and metadata; this type performs no file writes."""

    source_variant: str
    settings: C0016MaskPublicationSettings | C0017MaskPublicationSettings
    renderer_boundary: PrecomputedMaskSvgRendererBoundary
    input_mask_count: int
    quality_mask_count: int
    deduplicated_mask_count: int
    overlap_suppressed_count: int
    maximum_mask_truncated_count: int
    candidates: tuple[PublishedMaskCandidate, ...]
    display_candidate_ids: tuple[int, ...]
    visualization_candidate_ids: tuple[int, ...]
    all_masks_svg: str | None


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: exact-amg-default-contract -> _validate_generation_settings
def _validate_generation_settings(
    settings: PrecomputedMaskGenerationSettings,
) -> None:
    if not isinstance(settings, PrecomputedMaskGenerationSettings):
        raise MaskPublicationContractError(
            "mask generation settings must use the typed contract"
        )
    integer_settings = (
        settings.points_per_side,
        settings.points_per_batch,
        settings.crop_layers,
        settings.crop_points_downscale_factor,
        settings.maximum_masks,
    )
    if (
        any(type(value) is not int for value in integer_settings)
        or settings.points_per_side <= 0
        or settings.points_per_batch <= 0
        or settings.crop_layers < 0
        or settings.crop_points_downscale_factor <= 0
        or settings.maximum_masks <= 0
    ):
        raise MaskPublicationContractError(
            "mask generation integer settings are invalid"
        )
    probabilities = (
        settings.predicted_iou_threshold,
        settings.stability_score_threshold,
        settings.crop_overlap_ratio,
        settings.multiscale_merge_iou,
    )
    if any(
        type(value) not in {int, float}
        or not isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for value in probabilities
    ):
        raise MaskPublicationContractError(
            "mask generation probability settings are invalid"
        )
    if (
        settings.output_mode != "binary_mask"
        or type(settings.apply_postprocessing) is not bool
        or type(settings.use_multiscale) is not bool
        or type(settings.multiscale_scales) is not tuple
        or not settings.multiscale_scales
        or any(
            type(scale) not in {int, float}
            or not isfinite(float(scale))
            or float(scale) <= 0.0
            for scale in settings.multiscale_scales
        )
        or len(set(float(scale) for scale in settings.multiscale_scales))
        != len(settings.multiscale_scales)
    ):
        raise MaskPublicationContractError(
            "mask generation output or multiscale settings are invalid"
        )


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: boolean-rectangular-mask-contract -> _validate_mask_grid
def _validate_mask_grid(segmentation: MaskGrid) -> tuple[int, int, int]:
    if (
        type(segmentation) is not tuple
        or not segmentation
        or any(type(row) is not tuple or not row for row in segmentation)
    ):
        raise MaskPublicationContractError(
            "segmentation must be a nonempty tuple rectangle"
        )
    width = len(segmentation[0])
    if any(len(row) != width for row in segmentation):
        raise MaskPublicationContractError("segmentation rows have inconsistent widths")
    if any(type(bit) is not bool for row in segmentation for bit in row):
        raise MaskPublicationContractError("segmentation values must be exact booleans")
    area = sum(bit for row in segmentation for bit in row)
    if area <= 0:
        raise MaskPublicationContractError(
            "segmentation must contain at least one foreground pixel"
        )
    return width, len(segmentation), area


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: typed-precomputed-mask-validation -> _validate_precomputed_masks
def _validate_precomputed_masks(
    masks: Sequence[PrecomputedBooleanMask],
    settings: PrecomputedMaskGenerationSettings,
) -> tuple[PrecomputedBooleanMask, ...]:
    _validate_generation_settings(settings)
    if not masks:
        raise MaskPublicationContractError("precomputed masks are required")
    validated = tuple(masks)
    if any(type(mask) is not PrecomputedBooleanMask for mask in validated):
        raise MaskPublicationContractError(
            "precomputed masks must use PrecomputedBooleanMask"
        )
    identifiers: set[int] = set()
    expected_shape: tuple[int, int] | None = None
    allowed_scales = tuple(float(scale) for scale in settings.multiscale_scales)
    for mask in validated:
        if type(mask.candidate_id) is not int or mask.candidate_id < 0:
            raise MaskPublicationContractError("mask candidate identifiers are invalid")
        if mask.candidate_id in identifiers:
            raise MaskPublicationContractError(
                f"duplicate mask candidate identifier: {mask.candidate_id}"
            )
        identifiers.add(mask.candidate_id)
        width, height, _area = _validate_mask_grid(mask.segmentation)
        if expected_shape is None:
            expected_shape = (width, height)
        elif expected_shape != (width, height):
            raise MaskPublicationContractError(
                "precomputed masks must share one tile shape"
            )
        bounded_values = (mask.predicted_iou, mask.stability_score)
        if any(
            type(value) not in {int, float}
            or not isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in bounded_values
        ):
            raise MaskPublicationContractError(
                "mask quality values must be probabilities"
            )
        if (
            type(mask.source_scale) not in {int, float}
            or not isfinite(float(mask.source_scale))
            or not any(
                abs(float(mask.source_scale) - scale) < 1e-9
                for scale in allowed_scales
            )
        ):
            raise MaskPublicationContractError(
                "mask source scale is not declared by the generation contract"
            )
        if not settings.use_multiscale and abs(float(mask.source_scale) - 1.0) >= 1e-9:
            raise MaskPublicationContractError(
                "single-scale publication requires source scale 1.0"
            )
    return validated


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: logical-mask-intersection-union-epsilon -> mask_iou
def mask_iou(
    left: MaskGrid,
    right: MaskGrid,
    *,
    epsilon: float = 1e-9,
) -> float:
    left_width, left_height, _left_area = _validate_mask_grid(left)
    right_width, right_height, _right_area = _validate_mask_grid(right)
    if (left_width, left_height) != (right_width, right_height):
        raise MaskPublicationContractError("mask shapes differ")
    if (
        type(epsilon) not in {int, float}
        or not isfinite(float(epsilon))
        or float(epsilon) <= 0.0
    ):
        raise MaskPublicationContractError("mask IoU epsilon must be positive")
    intersection = sum(
        left_bit and right_bit
        for left_row, right_row in zip(left, right)
        for left_bit, right_bit in zip(left_row, right_row)
    )
    union = sum(
        left_bit or right_bit
        for left_row, right_row in zip(left, right)
        for left_bit, right_bit in zip(left_row, right_row)
    )
    return float(intersection) / float(union + float(epsilon))


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: mask-border-margin-two -> touches_border
def touches_border(mask: MaskGrid, *, margin: int = 2) -> bool:
    width, height, _area = _validate_mask_grid(mask)
    if type(margin) is not int or margin < 0:
        raise MaskPublicationContractError("mask border margin is invalid")
    if margin <= 0:
        return any(
            mask[y][x]
            for y in range(height)
            for x in range(width)
            if y == 0 or x == 0 or y == height - 1 or x == width - 1
        )
    return any(
        mask[y][x]
        for y in range(height)
        for x in range(width)
        if y < margin or x < margin or y >= height - margin or x >= width - margin
    )


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: exact-mask-bbox-and-padded-crop -> padded_crop_geometry
def padded_crop_geometry(
    mask: MaskGrid,
    *,
    padding_fraction: float = 0.10,
) -> MaskCropGeometry:
    width, height, _area = _validate_mask_grid(mask)
    if (
        type(padding_fraction) not in {int, float}
        or not isfinite(float(padding_fraction))
        or float(padding_fraction) < 0.0
    ):
        raise MaskPublicationContractError("mask crop padding is invalid")
    points = tuple(
        (x, y)
        for y, row in enumerate(mask)
        for x, bit in enumerate(row)
        if bit
    )
    xs = tuple(point[0] for point in points)
    ys = tuple(point[1] for point in points)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    box_width = max(1, x1 - x0 + 1)
    box_height = max(1, y1 - y0 + 1)
    padding = int(round(float(padding_fraction) * max(box_width, box_height)))
    crop_x0 = max(0, x0 - padding)
    crop_y0 = max(0, y0 - padding)
    crop_x1 = min(width - 1, x1 + padding)
    crop_y1 = min(height - 1, y1 + padding)
    return MaskCropGeometry(
        (x0, y0, x1, y1),
        (x0, y0, box_width, box_height),
        (crop_x0, crop_y0, crop_x1, crop_y1),
        crop_x1 - crop_x0 + 1,
        crop_y1 - crop_y0 + 1,
    )


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: quality-threshold-filter -> filter_precomputed_masks
def filter_precomputed_masks(
    masks: Sequence[PrecomputedBooleanMask],
    settings: PrecomputedMaskGenerationSettings = PrecomputedMaskGenerationSettings(),
) -> tuple[PrecomputedBooleanMask, ...]:
    validated = _validate_precomputed_masks(masks, settings)
    return tuple(
        mask
        for mask in validated
        if float(mask.predicted_iou) >= settings.predicted_iou_threshold
        and float(mask.stability_score) >= settings.stability_score_threshold
    )


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: predicted-iou-ordered-multiscale-nms -> merge_multiscale_masks
def merge_multiscale_masks(
    masks: Sequence[PrecomputedBooleanMask],
    *,
    merge_iou: float = 0.75,
    epsilon: float = 1e-9,
) -> tuple[PrecomputedBooleanMask, ...]:
    if not masks:
        raise MaskPublicationContractError("multiscale merge requires masks")
    if (
        type(merge_iou) not in {int, float}
        or not isfinite(float(merge_iou))
        or not 0.0 <= float(merge_iou) <= 1.0
    ):
        raise MaskPublicationContractError("multiscale merge IoU is invalid")
    validated = _validate_precomputed_masks(
        masks,
        PrecomputedMaskGenerationSettings(),
    )
    ordered = sorted(
        validated,
        key=lambda mask: float(mask.predicted_iou),
        reverse=True,
    )
    kept: list[PrecomputedBooleanMask] = []
    for candidate in ordered:
        if any(
            mask_iou(
                candidate.segmentation,
                existing.segmentation,
                epsilon=epsilon,
            )
            >= float(merge_iou)
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return tuple(kept)


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_STATEMENT_MAP: score-predicted-iou-area-sort -> sort_precomputed_masks
def sort_precomputed_masks(
    masks: Sequence[PrecomputedBooleanMask],
    *,
    sort_by: str = "score",
) -> tuple[PrecomputedBooleanMask, ...]:
    if sort_by not in {"score", "predicted_iou", "area"}:
        raise MaskPublicationContractError(
            "mask sort must be score, predicted_iou, or area"
        )
    validated = _validate_precomputed_masks(
        masks,
        PrecomputedMaskGenerationSettings(),
    )
    if sort_by == "predicted_iou":
        key = lambda mask: float(mask.predicted_iou)
    elif sort_by == "area":
        key = lambda mask: sum(bit for row in mask.segmentation for bit in row)
    else:
        key = lambda mask: float(mask.predicted_iou) * float(mask.stability_score)
    return tuple(sorted(validated, key=key, reverse=True))


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: exact-mask-grid-svg-fill-path -> _mask_fill_path
def _mask_fill_path(mask: MaskGrid) -> str:
    _validate_mask_grid(mask)
    commands: list[str] = []
    for y, row in enumerate(mask):
        x = 0
        while x < len(row):
            if not row[x]:
                x += 1
                continue
            start = x
            while x < len(row) and row[x]:
                x += 1
            length = x - start
            commands.append(f"M{start} {y}h{length}v1h-{length}z")
    return "".join(commands)


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: exact-mask-grid-svg-outline -> _mask_outline_path
def _mask_outline_path(mask: MaskGrid) -> str:
    width, height, _area = _validate_mask_grid(mask)
    commands: list[str] = []
    for y, row in enumerate(mask):
        for x, bit in enumerate(row):
            if not bit:
                continue
            if y == 0 or not mask[y - 1][x]:
                commands.append(f"M{x} {y}h1")
            if x == width - 1 or not mask[y][x + 1]:
                commands.append(f"M{x + 1} {y}v1")
            if y == height - 1 or not mask[y + 1][x]:
                commands.append(f"M{x + 1} {y + 1}h-1")
            if x == 0 or not mask[y][x - 1]:
                commands.append(f"M{x} {y + 1}v-1")
    return "".join(commands)


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: mask-grid-only-renderer-boundary -> _svg_opening
def _svg_opening(boundary: PrecomputedMaskSvgRendererBoundary) -> str:
    colors = (
        boundary.background_rgb,
        boundary.individual_fill_rgb,
        boundary.individual_outline_rgb,
        boundary.all_masks_outline_rgb,
    ) if type(boundary) is PrecomputedMaskSvgRendererBoundary else ()
    if (
        type(boundary) is not PrecomputedMaskSvgRendererBoundary
        or type(boundary.canvas_width) is not int
        or type(boundary.canvas_height) is not int
        or boundary.canvas_width <= 0
        or boundary.canvas_height <= 0
        or boundary.renderer_scope != "precomputed-mask-grid-only"
        or boundary.includes_source_image_pixels
        or boundary.all_masks_palette != "lcg-rgb-v1"
        or any(
            type(color) is not tuple
            or len(color) != 3
            or any(
                type(component) is not int or not 0 <= component <= 255
                for component in color
            )
            for color in colors
        )
    ):
        raise MaskPublicationContractError("mask SVG renderer boundary is invalid")
    red, green, blue = boundary.background_rgb
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{boundary.canvas_width}" height="{boundary.canvas_height}" '
        f'viewBox="0 0 {boundary.canvas_width} {boundary.canvas_height}" '
        f'shape-rendering="crispEdges" data-render-scope="{boundary.renderer_scope}">'
        f'<rect width="{boundary.canvas_width}" height="{boundary.canvas_height}" '
        f'fill="#{red:02x}{green:02x}{blue:02x}" data-role="no-source-image-pixels"/>'
    )


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: green-alpha-red-outline-individual-svg -> render_individual_mask_svg
def render_individual_mask_svg(
    mask: PrecomputedBooleanMask,
    boundary: PrecomputedMaskSvgRendererBoundary,
    *,
    alpha: float = 0.45,
) -> str:
    if type(mask) is not PrecomputedBooleanMask:
        raise MaskPublicationContractError(
            "individual SVG requires a typed precomputed mask"
        )
    _validate_precomputed_masks((mask,), PrecomputedMaskGenerationSettings())
    width, height, _area = _validate_mask_grid(mask.segmentation)
    if (width, height) != (boundary.canvas_width, boundary.canvas_height):
        raise MaskPublicationContractError(
            "individual mask and SVG boundary shapes differ"
        )
    if (
        type(alpha) not in {int, float}
        or not isfinite(float(alpha))
        or not 0.0 < float(alpha) <= 1.0
    ):
        raise MaskPublicationContractError("individual overlay alpha is invalid")
    fill = boundary.individual_fill_rgb
    outline = boundary.individual_outline_rgb
    return (
        _svg_opening(boundary)
        + f'<path d="{_mask_fill_path(mask.segmentation)}" '
        + f'fill="#{fill[0]:02x}{fill[1]:02x}{fill[2]:02x}" '
        + f'fill-opacity="{float(alpha):.6g}" data-candidate-id="{mask.candidate_id}"/>'
        + f'<path d="{_mask_outline_path(mask.segmentation)}" fill="none" '
        + f'stroke="#{outline[0]:02x}{outline[1]:02x}{outline[2]:02x}" '
        + 'stroke-width="2" vector-effect="non-scaling-stroke"/>'
        + "</svg>"
    )


# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: deterministic-seed-three-mask-palette -> _all_mask_color
def _all_mask_color(seed: int, ordinal: int) -> tuple[int, int, int]:
    if type(seed) is not int or seed < 0 or type(ordinal) is not int or ordinal < 0:
        raise MaskPublicationContractError("all-mask palette inputs are invalid")
    state = ((seed + 1) * 1103515245 + (ordinal + 1) * 12345) & 0xFFFFFFFF
    red = (state >> 16) & 0xFF
    green = ((state * 1664525 + 1013904223) >> 16) & 0xFF
    blue = ((state * 22695477 + 1) >> 16) & 0xFF
    return red, green, blue


# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: area-descending-all-mask-overlay-svg -> render_all_masks_svg
def render_all_masks_svg(
    masks: Sequence[PrecomputedBooleanMask],
    boundary: PrecomputedMaskSvgRendererBoundary,
    *,
    seed: int = 3,
    alpha: float = 0.50,
    borders: bool = True,
) -> str:
    if type(seed) is not int or seed < 0:
        raise MaskPublicationContractError("all-mask overlay seed is invalid")
    if (
        type(alpha) not in {int, float}
        or not isfinite(float(alpha))
        or not 0.0 < float(alpha) <= 1.0
        or type(borders) is not bool
    ):
        raise MaskPublicationContractError("all-mask overlay settings are invalid")
    validated = (
        _validate_precomputed_masks(masks, PrecomputedMaskGenerationSettings())
        if masks
        else ()
    )
    ordered = sorted(
        validated,
        key=lambda mask: sum(bit for row in mask.segmentation for bit in row),
        reverse=True,
    )
    parts = [_svg_opening(boundary)]
    outline = boundary.all_masks_outline_rgb
    for ordinal, mask in enumerate(ordered):
        width, height, _area = _validate_mask_grid(mask.segmentation)
        if (width, height) != (boundary.canvas_width, boundary.canvas_height):
            raise MaskPublicationContractError(
                "all-mask overlay contains inconsistent shapes"
            )
        color = _all_mask_color(seed, ordinal)
        parts.append(
            f'<path d="{_mask_fill_path(mask.segmentation)}" '
            f'fill="#{color[0]:02x}{color[1]:02x}{color[2]:02x}" '
            f'fill-opacity="{float(alpha):.6g}" '
            f'data-candidate-id="{mask.candidate_id}"/>'
        )
        if borders:
            parts.append(
                f'<path d="{_mask_outline_path(mask.segmentation)}" fill="none" '
                f'stroke="#{outline[0]:02x}{outline[1]:02x}{outline[2]:02x}" '
                'stroke-width="1" vector-effect="non-scaling-stroke"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: filter-nms-maximum-mask-order -> _prepare_publication_masks
def _prepare_publication_masks(
    masks: Sequence[PrecomputedBooleanMask],
    generation: PrecomputedMaskGenerationSettings,
    *,
    epsilon: float,
) -> tuple[
    tuple[PrecomputedBooleanMask, ...],
    tuple[PrecomputedBooleanMask, ...],
    tuple[PrecomputedBooleanMask, ...],
]:
    validated = _validate_precomputed_masks(masks, generation)
    quality = tuple(
        mask
        for mask in validated
        if float(mask.predicted_iou) >= generation.predicted_iou_threshold
        and float(mask.stability_score) >= generation.stability_score_threshold
    )
    if not quality:
        raise MaskPublicationContractError(
            "no precomputed masks pass the source quality thresholds"
        )
    deduplicated = (
        merge_multiscale_masks(
            quality,
            merge_iou=generation.multiscale_merge_iou,
            epsilon=epsilon,
        )
        if generation.use_multiscale
        else quality
    )
    published = deduplicated[: generation.maximum_masks]
    if not published:
        raise MaskPublicationContractError("mask publication produced no candidates")
    return quality, deduplicated, published


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: bbox-area-score-border-svg-metadata -> _published_candidate
def _published_candidate(
    mask: PrecomputedBooleanMask,
    *,
    publication_index: int,
    width: int,
    height: int,
    padding_fraction: float,
    border_margin: int,
    visualization_eligible: bool,
    individual_alpha: float,
    render_individual: bool,
    boundary: PrecomputedMaskSvgRendererBoundary,
) -> PublishedMaskCandidate:
    measured_width, measured_height, area = _validate_mask_grid(mask.segmentation)
    if (measured_width, measured_height) != (width, height):
        raise MaskPublicationContractError("publication candidate shape changed")
    geometry = padded_crop_geometry(
        mask.segmentation,
        padding_fraction=padding_fraction,
    )
    return PublishedMaskCandidate(
        publication_index,
        mask,
        area,
        area / float(width * height),
        float(mask.predicted_iou) * float(mask.stability_score),
        geometry,
        touches_border(mask.segmentation, margin=border_margin),
        visualization_eligible,
        (
            render_individual_mask_svg(mask, boundary, alpha=individual_alpha)
            if render_individual
            else None
        ),
    )


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_STATEMENT_MAP: score-ordered-top-24-mask-publication -> publish_c0016_masks
def publish_c0016_masks(
    masks: Sequence[PrecomputedBooleanMask],
    settings: C0016MaskPublicationSettings = C0016MaskPublicationSettings(),
) -> MaskPublicationResult:
    if type(settings) is not C0016MaskPublicationSettings:
        raise MaskPublicationContractError("C16 settings must use the typed contract")
    if (
        type(settings.overlay_alpha) not in {int, float}
        or not isfinite(float(settings.overlay_alpha))
        or not 0.0 < float(settings.overlay_alpha) <= 1.0
        or type(settings.crop_padding_fraction) not in {int, float}
        or not isfinite(float(settings.crop_padding_fraction))
        or float(settings.crop_padding_fraction) < 0.0
        or type(settings.display_top_k) is not int
        or settings.display_top_k <= 0
        or settings.sort_by not in {"score", "predicted_iou", "area"}
        or type(settings.border_margin) is not int
        or settings.border_margin < 0
        or type(settings.iou_epsilon) not in {int, float}
        or not isfinite(float(settings.iou_epsilon))
        or float(settings.iou_epsilon) <= 0.0
        or type(settings.display_columns) is not int
        or settings.display_columns <= 0
        or type(settings.display_figure_inches) is not tuple
        or len(settings.display_figure_inches) != 2
        or any(
            type(value) not in {int, float}
            or not isfinite(float(value))
            or float(value) <= 0.0
            for value in settings.display_figure_inches
        )
    ):
        raise MaskPublicationContractError("C16 publication settings are invalid")
    quality, deduplicated, published = _prepare_publication_masks(
        masks,
        settings.generation,
        epsilon=float(settings.iou_epsilon),
    )
    ordered = sort_precomputed_masks(published, sort_by=settings.sort_by)
    width, height, _area = _validate_mask_grid(ordered[0].segmentation)
    boundary = PrecomputedMaskSvgRendererBoundary(width, height)
    candidates = tuple(
        _published_candidate(
            mask,
            publication_index=index,
            width=width,
            height=height,
            padding_fraction=float(settings.crop_padding_fraction),
            border_margin=settings.border_margin,
            visualization_eligible=True,
            individual_alpha=float(settings.overlay_alpha),
            render_individual=True,
            boundary=boundary,
        )
        for index, mask in enumerate(ordered)
    )
    return MaskPublicationResult(
        "c0016",
        settings,
        boundary,
        len(masks),
        len(quality),
        len(deduplicated),
        len(quality) - len(deduplicated),
        len(deduplicated) - len(published),
        candidates,
        tuple(item.mask.candidate_id for item in candidates[: settings.display_top_k]),
        tuple(item.mask.candidate_id for item in candidates),
        None,
    )


# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: iou-order-area-viz-all-and-individual-svg -> publish_c0017_masks
def publish_c0017_masks(
    masks: Sequence[PrecomputedBooleanMask],
    settings: C0017MaskPublicationSettings = C0017MaskPublicationSettings(),
) -> MaskPublicationResult:
    if type(settings) is not C0017MaskPublicationSettings:
        raise MaskPublicationContractError("C17 settings must use the typed contract")
    if (
        type(settings.save_per_mask) is not bool
        or type(settings.maximum_visualization_area_fraction) not in {int, float}
        or not isfinite(float(settings.maximum_visualization_area_fraction))
        or not 0.0 < float(settings.maximum_visualization_area_fraction) <= 1.0
        or type(settings.single_overlay_alpha) not in {int, float}
        or not isfinite(float(settings.single_overlay_alpha))
        or not 0.0 < float(settings.single_overlay_alpha) <= 1.0
        or type(settings.all_masks_overlay_alpha) not in {int, float}
        or not isfinite(float(settings.all_masks_overlay_alpha))
        or not 0.0 < float(settings.all_masks_overlay_alpha) <= 1.0
        or type(settings.crop_padding_fraction) not in {int, float}
        or not isfinite(float(settings.crop_padding_fraction))
        or float(settings.crop_padding_fraction) < 0.0
        or type(settings.preview_top_k) is not int
        or settings.preview_top_k <= 0
        or settings.sort_by != "predicted_iou"
        or type(settings.border_margin) is not int
        or settings.border_margin < 0
        or type(settings.iou_epsilon) not in {int, float}
        or not isfinite(float(settings.iou_epsilon))
        or float(settings.iou_epsilon) <= 0.0
        or type(settings.all_masks_seed) is not int
        or settings.all_masks_seed < 0
        or type(settings.all_masks_borders) is not bool
        or type(settings.preview_columns) is not int
        or settings.preview_columns <= 0
        or type(settings.preview_figure_inches) is not tuple
        or len(settings.preview_figure_inches) != 2
        or any(
            type(value) not in {int, float}
            or not isfinite(float(value))
            or float(value) <= 0.0
            for value in settings.preview_figure_inches
        )
    ):
        raise MaskPublicationContractError("C17 publication settings are invalid")
    quality, deduplicated, published = _prepare_publication_masks(
        masks,
        settings.generation,
        epsilon=float(settings.iou_epsilon),
    )
    ordered = sort_precomputed_masks(published, sort_by=settings.sort_by)
    width, height, _area = _validate_mask_grid(ordered[0].segmentation)
    boundary = PrecomputedMaskSvgRendererBoundary(width, height)
    maximum_visualization_area = int(
        float(settings.maximum_visualization_area_fraction) * width * height
    )
    candidates = tuple(
        _published_candidate(
            mask,
            publication_index=index,
            width=width,
            height=height,
            padding_fraction=float(settings.crop_padding_fraction),
            border_margin=settings.border_margin,
            visualization_eligible=(
                sum(bit for row in mask.segmentation for bit in row)
                <= maximum_visualization_area
            ),
            individual_alpha=float(settings.single_overlay_alpha),
            render_individual=settings.save_per_mask,
            boundary=boundary,
        )
        for index, mask in enumerate(ordered)
    )
    visualization_masks = tuple(
        candidate.mask for candidate in candidates if candidate.visualization_eligible
    )
    all_masks_svg = render_all_masks_svg(
        visualization_masks,
        boundary,
        seed=settings.all_masks_seed,
        alpha=float(settings.all_masks_overlay_alpha),
        borders=settings.all_masks_borders,
    )
    return MaskPublicationResult(
        "c0017",
        settings,
        boundary,
        len(masks),
        len(quality),
        len(deduplicated),
        len(quality) - len(deduplicated),
        len(deduplicated) - len(published),
        candidates,
        tuple(item.mask.candidate_id for item in candidates[: settings.preview_top_k]),
        tuple(item.mask.candidate_id for item in candidates if item.visualization_eligible),
        all_masks_svg,
    )
