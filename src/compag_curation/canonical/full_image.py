"""Memory-bounded SAM2 proposals for a native full-image profile.

The released SAM2 automatic mask generator necessarily creates dense masks
while a prompt batch is decoded.  This module bounds that transient CUDA work
with a small execution microbatch and requests uncompressed RLE output so no
full-resolution binary mask survives the generator call.  Each returned RLE is
then converted directly to a bbox-cropped, column-packed representation.

The module deliberately has no dependency on the canonical service, project
configuration, or model-bundle code.  A generator can be injected for tests or
constructed lazily from the optional SAM2 dependency.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Iterable, Mapping, Protocol


FULL_IMAGE_MASK_HASH_DOMAIN = b"compag-full-image-compact-mask-v1\0"
FULL_IMAGE_PROPOSAL_HASH_DOMAIN = b"compag-full-image-proposal-v1\0"
FULL_IMAGE_RUNTIME_MAX_POINTS_PER_BATCH = 8
FULL_IMAGE_MASK_ENCODING = "base64-packbits-little-column-major-bbox-v1"
FULL_IMAGE_DURABLE_MASK_ENCODING = "bbox-cropped-column-packbits-little-v1"
FULL_IMAGE_COMPONENT_POLICY = "largest-8-connected-pixel-area-top-left-tie-v1"


class AutomaticMaskGenerator(Protocol):
    """Small dependency-injection surface used by :class:`FullImageBackend`."""

    def generate(self, image: Any) -> Iterable[Mapping[str, Any]]:
        """Return SAM-style annotation mappings for one image."""


GeneratorFactory = Callable[..., AutomaticMaskGenerator]


@dataclass(frozen=True, slots=True)
class FullImageSettings:
    """SAM2 and selection settings for the full-image proposal profile.

    ``points_per_batch`` is an execution-only safety control.  Eight prompts
    keeps the large float32 mask-logit tensor bounded for multi-megapixel
    canvases; changing it does not change the 64-by-64 prompt lattice.
    """

    points_per_side: int = 64
    points_per_batch: int = FULL_IMAGE_RUNTIME_MAX_POINTS_PER_BATCH
    pred_iou_thresh: float = 0.80
    stability_score_thresh: float = 0.88
    crop_n_layers: int = 2
    crop_n_points_downscale_factor: int = 1
    crop_overlap_ratio: float = 512 / 1500
    box_nms_thresh: float = 0.70
    crop_nms_thresh: float = 0.70
    merge_iou_threshold: float = 0.75
    max_proposals: int = 1000

    def __post_init__(self) -> None:
        _positive_int(self.points_per_side, "points_per_side")
        _positive_int(self.points_per_batch, "points_per_batch")
        if self.points_per_batch > FULL_IMAGE_RUNTIME_MAX_POINTS_PER_BATCH:
            raise ValueError(
                "full-image points_per_batch exceeds the memory-safe runtime policy"
            )
        _unit_score(self.pred_iou_thresh, "pred_iou_thresh")
        _unit_score(self.stability_score_thresh, "stability_score_thresh")
        _positive_int(self.crop_n_layers, "crop_n_layers")
        _positive_int(
            self.crop_n_points_downscale_factor,
            "crop_n_points_downscale_factor",
        )
        _open_unit_score(self.crop_overlap_ratio, "crop_overlap_ratio")
        _open_unit_score(self.box_nms_thresh, "box_nms_thresh")
        _open_unit_score(self.crop_nms_thresh, "crop_nms_thresh")
        _open_unit_score(self.merge_iou_threshold, "merge_iou_threshold")
        _positive_int(self.max_proposals, "max_proposals")

    def generator_kwargs(self) -> dict[str, int | float | bool | str]:
        """Return the native SAM2 crop-pyramid arguments for this profile."""

        return {
            "points_per_side": self.points_per_side,
            "points_per_batch": self.points_per_batch,
            "pred_iou_thresh": self.pred_iou_thresh,
            "stability_score_thresh": self.stability_score_thresh,
            "stability_score_offset": 1.0,
            "mask_threshold": 0.0,
            "crop_n_layers": self.crop_n_layers,
            "crop_n_points_downscale_factor": self.crop_n_points_downscale_factor,
            "crop_overlap_ratio": self.crop_overlap_ratio,
            "box_nms_thresh": self.box_nms_thresh,
            "crop_nms_thresh": self.crop_nms_thresh,
            "min_mask_region_area": 0,
            "output_mode": "uncompressed_rle",
            "use_m2m": False,
            "multimask_output": True,
        }


@dataclass(frozen=True, slots=True)
class PackedMask:
    """A nonempty mask packed inside its minimal bbox.

    Bits are little-endian and column-major.  Every local x-coordinate owns a
    separately padded byte column.  This matches COCO RLE's column traversal,
    permits direct RLE packing, and lets intersection counts operate one column
    at a time without reconstructing an ``H x W`` canvas.
    """

    canvas_height: int
    canvas_width: int
    origin_x: int
    origin_y: int
    width: int
    height: int
    area: int
    column_stride: int
    packed: bytes
    mask_sha256: str

    def __post_init__(self) -> None:
        for value, role in (
            (self.canvas_height, "canvas_height"),
            (self.canvas_width, "canvas_width"),
            (self.width, "width"),
            (self.height, "height"),
            (self.area, "area"),
            (self.column_stride, "column_stride"),
        ):
            _positive_int(value, role)
        for value, role in (
            (self.origin_x, "origin_x"),
            (self.origin_y, "origin_y"),
        ):
            _nonnegative_int(value, role)
        if self.origin_x + self.width > self.canvas_width:
            raise ValueError("packed mask exceeds the canvas width")
        if self.origin_y + self.height > self.canvas_height:
            raise ValueError("packed mask exceeds the canvas height")
        if self.area > self.width * self.height:
            raise ValueError("packed mask area exceeds its bbox")
        expected_stride = (self.height + 7) // 8
        if self.column_stride != expected_stride:
            raise ValueError("packed mask column stride is not canonical")
        if not isinstance(self.packed, bytes):
            raise TypeError("packed mask payload must be immutable bytes")
        if len(self.packed) != self.width * self.column_stride:
            raise ValueError("packed mask payload length is invalid")
        if not _is_lower_sha256(self.mask_sha256):
            raise ValueError("packed mask SHA256 is invalid")
        if self.mask_sha256 != _packed_mask_sha256(
            canvas_height=self.canvas_height,
            canvas_width=self.canvas_width,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            width=self.width,
            height=self.height,
            area=self.area,
            column_stride=self.column_stride,
            packed=self.packed,
        ):
            raise ValueError("packed mask SHA256 does not match its payload")
        _validate_padding_bits(self)
        if sum(byte.bit_count() for byte in self.packed) != self.area:
            raise ValueError("packed mask area does not match its payload")
        first_column = self._column_bits(self.origin_x)
        last_column = self._column_bits(self.origin_x + self.width - 1)
        if (
            first_column == 0
            or last_column == 0
            or not any(
                self._column_bits(self.origin_x + local_x) & 1
                for local_x in range(self.width)
            )
            or not any(
                (self._column_bits(self.origin_x + local_x) >> (self.height - 1))
                & 1
                for local_x in range(self.width)
            )
        ):
            raise ValueError("packed mask bbox is not minimal")

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return ``(x, y, width, height)`` in canvas coordinates."""

        return self.origin_x, self.origin_y, self.width, self.height

    @property
    def retained_bytes(self) -> int:
        """Return the exact retained bitmap payload size."""

        return len(self.packed)

    def intersection_area(self, other: "PackedMask") -> int:
        """Count intersecting pixels without creating either full mask."""

        if (
            self.canvas_height != other.canvas_height
            or self.canvas_width != other.canvas_width
        ):
            raise ValueError("packed masks use different canvases")
        overlap = _bbox_intersection(self.bbox, other.bbox)
        if overlap is None:
            return 0
        x0, y0, width, height = overlap
        bit_mask = (1 << height) - 1
        intersection = 0
        for canvas_x in range(x0, x0 + width):
            left = self._column_bits(canvas_x)
            right = other._column_bits(canvas_x)
            left >>= y0 - self.origin_y
            right >>= y0 - other.origin_y
            intersection += ((left & right) & bit_mask).bit_count()
        return intersection

    def iou(self, other: "PackedMask") -> float:
        """Return exact mask IoU from compact payloads."""

        intersection = self.intersection_area(other)
        if intersection == 0:
            return 0.0
        return intersection / (self.area + other.area - intersection)

    def crop_array(self) -> Any:
        """Decode a read-only uint8 bbox crop, never the full canvas."""

        import numpy as np

        columns = np.frombuffer(self.packed, dtype=np.uint8).reshape(
            self.width, self.column_stride
        )
        unpacked = np.unpackbits(
            columns,
            axis=1,
            count=self.height,
            bitorder="little",
        )
        crop = np.ascontiguousarray(unpacked.T.astype(np.uint8, copy=False))
        crop.setflags(write=False)
        return crop

    def _column_bits(self, canvas_x: int) -> int:
        local_x = canvas_x - self.origin_x
        if not 0 <= local_x < self.width:
            return 0
        start = local_x * self.column_stride
        end = start + self.column_stride
        return int.from_bytes(self.packed[start:end], "little")


@dataclass(frozen=True, slots=True)
class FullImageProposal:
    """One selected proposal with geometry-only identity."""

    source_index: int
    proposal_index: int
    proposal_sha256: str
    packed_mask: PackedMask
    predicted_iou: float
    stability_score: float

    def __post_init__(self) -> None:
        _nonnegative_int(self.source_index, "source_index")
        _positive_int(self.proposal_index, "proposal_index")
        if not _is_lower_sha256(self.proposal_sha256):
            raise ValueError("full-image proposal SHA256 is invalid")
        if self.proposal_sha256 != full_image_proposal_sha256(self.packed_mask):
            raise ValueError("full-image proposal SHA256 does not match its mask")
        _unit_score(self.predicted_iou, "predicted_iou")
        _unit_score(self.stability_score, "stability_score")

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.packed_mask.bbox

    @property
    def area(self) -> int:
        return self.packed_mask.area

    @property
    def mask_sha256(self) -> str:
        return self.packed_mask.mask_sha256

    @property
    def mask_crop(self) -> Any:
        """Decode the read-only uint8 bbox crop expected by feature extractors."""

        return self.packed_mask.crop_array()


@dataclass(frozen=True, slots=True)
class FullImageProposalSelection:
    proposals: tuple[FullImageProposal, ...]
    input_count: int
    empty_count: int
    duplicate_count: int
    capped_count: int
    bbox_prefilter_count: int
    mask_comparison_count: int
    retained_mask_bytes: int


class FullImageBackend:
    """Run an injected RLE-mode generator and compact its proposals."""

    def __init__(
        self,
        generator: AutomaticMaskGenerator,
        *,
        settings: FullImageSettings | None = None,
    ) -> None:
        if not callable(getattr(generator, "generate", None)):
            raise TypeError("full-image generator must expose generate(image)")
        self._generator = generator
        self.settings = settings or FullImageSettings()

    @property
    def generator(self) -> AutomaticMaskGenerator:
        """Expose the retained generator for integration-time attestation."""

        return self._generator

    @property
    def predictor(self) -> Any:
        """Expose the generator predictor without retaining another model copy."""

        return getattr(self._generator, "predictor", None)

    def generate(self, image: Any) -> FullImageProposalSelection:
        canvas_shape = _image_canvas(image)
        annotations = self._generator.generate(image)
        if annotations is None:
            raise ValueError("SAM2 full-image generator returned no iterable")
        return select_full_image_proposals(
            annotations,
            canvas_shape=canvas_shape,
            merge_iou_threshold=self.settings.merge_iou_threshold,
            max_proposals=self.settings.max_proposals,
        )


def build_full_image_amg(
    model: Any,
    execution_points_per_batch: int = FULL_IMAGE_RUNTIME_MAX_POINTS_PER_BATCH,
    *,
    settings: FullImageSettings | None = None,
    generator_factory: GeneratorFactory | None = None,
) -> AutomaticMaskGenerator:
    """Build a native-crop, uncompressed-RLE SAM2 mask generator.

    The execution microbatch may be reduced below eight for unusually large
    canvases.  Values above eight are rejected by the full-image safety policy.
    """

    resolved = settings or FullImageSettings()
    batch = _positive_int(execution_points_per_batch, "execution_points_per_batch")
    if batch > FULL_IMAGE_RUNTIME_MAX_POINTS_PER_BATCH:
        raise ValueError(
            "full-image execution_points_per_batch exceeds the memory-safe policy"
        )
    kwargs = resolved.generator_kwargs()
    kwargs["points_per_batch"] = batch
    _attest_full_image_sam2_cuda_model(model)
    if generator_factory is None:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        generator_factory = _memory_bounded_sam2_amg_type(SAM2AutomaticMaskGenerator)
    generator = generator_factory(model, **kwargs)
    _attest_full_image_amg_cuda(generator, expected_model=model)
    return generator


def build_full_image_backend(
    model: Any,
    *,
    settings: FullImageSettings | None = None,
    generator_factory: GeneratorFactory | None = None,
) -> FullImageBackend:
    """Construct the standalone backend from SAM2 or an injected factory."""

    resolved = settings or FullImageSettings()
    generator = build_full_image_amg(
        model,
        execution_points_per_batch=resolved.points_per_batch,
        settings=resolved,
        generator_factory=generator_factory,
    )
    return FullImageBackend(generator, settings=resolved)


def pack_uncompressed_rle(
    rle: Mapping[str, Any],
    *,
    canvas_height: int | None = None,
    canvas_width: int | None = None,
) -> PackedMask | None:
    """Convert uncompressed COCO RLE directly to a minimal packed bbox.

    ``None`` represents a valid empty RLE.  Compressed COCO byte/string counts
    are intentionally rejected because the backend requests uncompressed RLE
    from SAM2 and should not import pycocotools merely to normalize its output.
    """

    if not isinstance(rle, Mapping):
        raise ValueError("SAM2 segmentation must be an uncompressed RLE mapping")
    size = rle.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("SAM2 RLE size must be [height, width]")
    height = _positive_int(size[0], "RLE height")
    width = _positive_int(size[1], "RLE width")
    if canvas_height is not None and height != _positive_int(
        canvas_height, "canvas_height"
    ):
        raise ValueError("SAM2 RLE height differs from the image canvas")
    if canvas_width is not None and width != _positive_int(
        canvas_width, "canvas_width"
    ):
        raise ValueError("SAM2 RLE width differs from the image canvas")
    raw_counts = rle.get("counts")
    if isinstance(raw_counts, (str, bytes, bytearray)) or not isinstance(
        raw_counts, (list, tuple)
    ):
        raise ValueError("SAM2 RLE counts must be an uncompressed sequence")
    counts = tuple(
        _nonnegative_int(value, f"RLE count {index}")
        for index, value in enumerate(raw_counts)
    )
    if not counts:
        raise ValueError("SAM2 RLE counts must not be empty")
    if sum(counts) != height * width:
        raise ValueError("SAM2 RLE counts do not cover the image canvas")

    runs = _largest_8_connected_component(
        tuple(_foreground_column_runs(counts, height))
    )
    if not runs:
        return None

    area = sum(run_length for _x, _y, run_length in runs)
    min_x = min(x for x, _y, _run_length in runs)
    max_x = max(x for x, _y, _run_length in runs)
    min_y = min(y for _x, y, _run_length in runs)
    max_y = max(y + run_length - 1 for _x, y, run_length in runs)

    crop_width = max_x - min_x + 1
    crop_height = max_y - min_y + 1
    column_stride = (crop_height + 7) // 8
    packed = bytearray(crop_width * column_stride)
    for x, y, run_length in runs:
        _set_column_bit_range(
            packed,
            column_stride=column_stride,
            local_x=x - min_x,
            start=y - min_y,
            stop=y - min_y + run_length,
        )
    payload = bytes(packed)
    digest = _packed_mask_sha256(
        canvas_height=height,
        canvas_width=width,
        origin_x=min_x,
        origin_y=min_y,
        width=crop_width,
        height=crop_height,
        area=area,
        column_stride=column_stride,
        packed=payload,
    )
    return PackedMask(
        canvas_height=height,
        canvas_width=width,
        origin_x=min_x,
        origin_y=min_y,
        width=crop_width,
        height=crop_height,
        area=area,
        column_stride=column_stride,
        packed=payload,
        mask_sha256=digest,
    )


def full_image_proposal_sha256(mask: PackedMask) -> str:
    """Return the independent geometry-only full-image proposal identity."""

    return hashlib.sha256(
        FULL_IMAGE_PROPOSAL_HASH_DOMAIN + bytes.fromhex(mask.mask_sha256)
    ).hexdigest()


def select_full_image_proposals(
    annotations: Iterable[Mapping[str, Any]],
    *,
    canvas_shape: tuple[int, ...],
    merge_iou_threshold: float = 0.75,
    max_proposals: int = 1000,
) -> FullImageProposalSelection:
    """Pack, deterministically deduplicate, and cap RLE proposals.

    Candidate ordering is independent of generator iteration order.  Before an
    exact compact-mask comparison, bbox overlap and a safe area-based upper
    bound prove whether the requested IoU is attainable.
    """

    height, width = _canvas_shape(canvas_shape)
    threshold = _open_unit_score(merge_iou_threshold, "merge_iou_threshold")
    cap = _positive_int(max_proposals, "max_proposals")
    candidates: list[tuple[int, PackedMask, float, float, str]] = []
    input_count = 0
    empty_count = 0
    for annotation in annotations:
        input_count += 1
        if not isinstance(annotation, Mapping) or "segmentation" not in annotation:
            raise ValueError("SAM2 returned a malformed full-image proposal")
        mask = pack_uncompressed_rle(
            annotation["segmentation"],
            canvas_height=height,
            canvas_width=width,
        )
        if mask is None:
            empty_count += 1
            continue
        predicted_iou = _unit_score(
            annotation.get("predicted_iou", 1.0), "predicted_iou"
        )
        stability_score = _unit_score(
            annotation.get("stability_score", 1.0), "stability_score"
        )
        proposal_sha256 = full_image_proposal_sha256(mask)
        candidates.append(
            (
                input_count - 1,
                mask,
                predicted_iou,
                stability_score,
                proposal_sha256,
            )
        )

    candidates.sort(key=lambda row: (-row[2], -row[3], -row[1].area, row[4]))
    selected: list[tuple[int, PackedMask, float, float, str]] = []
    dedup_representatives: list[
        tuple[int, PackedMask, float, float, str]
    ] = []
    duplicate_count = 0
    capped_count = 0
    bbox_prefilter_count = 0
    mask_comparison_count = 0
    for candidate in candidates:
        (
            _source_index,
            mask,
            _predicted_iou,
            _stability_score,
            proposal_sha256,
        ) = candidate
        duplicate = False
        for (
            _kept_source,
            kept_mask,
            _kept_iou,
            _kept_stability,
            kept_sha256,
        ) in dedup_representatives:
            if proposal_sha256 == kept_sha256:
                duplicate = True
                break
            if _mask_iou_upper_bound(mask, kept_mask) < threshold:
                bbox_prefilter_count += 1
                continue
            mask_comparison_count += 1
            if mask.iou(kept_mask) >= threshold:
                duplicate = True
                break
        if duplicate:
            duplicate_count += 1
        else:
            dedup_representatives.append(candidate)
            if len(selected) >= cap:
                capped_count += 1
            else:
                selected.append(candidate)

    proposals = tuple(
        FullImageProposal(
            # The source ordinal is canonicalized after score/hash sorting so
            # semantically identical AMG output is byte-stable even if the
            # upstream iterable arrives in a different order.
            source_index=index - 1,
            proposal_index=index,
            proposal_sha256=proposal_sha256,
            packed_mask=mask,
            predicted_iou=predicted_iou,
            stability_score=stability_score,
        )
        for index, (
            source_index,
            mask,
            predicted_iou,
            stability_score,
            proposal_sha256,
        ) in enumerate(selected, start=1)
    )
    return FullImageProposalSelection(
        proposals=proposals,
        input_count=input_count,
        empty_count=empty_count,
        duplicate_count=duplicate_count,
        capped_count=capped_count,
        bbox_prefilter_count=bbox_prefilter_count,
        mask_comparison_count=mask_comparison_count,
        retained_mask_bytes=sum(row.packed_mask.retained_bytes for row in proposals),
    )


def _foreground_column_runs(
    counts: tuple[int, ...], canvas_height: int
) -> Iterable[tuple[int, int, int]]:
    cursor = 0
    foreground = False
    for count in counts:
        if foreground:
            remaining = count
            position = cursor
            while remaining:
                x, y = divmod(position, canvas_height)
                run_length = min(remaining, canvas_height - y)
                yield x, y, run_length
                position += run_length
                remaining -= run_length
        cursor += count
        foreground = not foreground


def _largest_8_connected_component(
    raw_runs: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    """Return one deterministic component without allocating an image canvas.

    RLE foreground intervals are merged within columns and unioned across
    adjacent columns when their closed y-ranges touch (8-connectivity).  The
    largest pixel-area component wins; exact ties choose the top-most, then
    left-most component and finally its smallest extent.
    """

    if not raw_runs:
        return ()
    merged: list[tuple[int, int, int]] = []
    for x, y, length in raw_runs:
        if (
            merged
            and merged[-1][0] == x
            and y <= merged[-1][1] + merged[-1][2]
        ):
            previous_x, previous_y, previous_length = merged[-1]
            stop = max(previous_y + previous_length, y + length)
            merged[-1] = (previous_x, previous_y, stop - previous_y)
        else:
            merged.append((x, y, length))

    parent = list(range(len(merged)))
    component_area = [length for _x, _y, length in merged]
    min_x = [x for x, _y, _length in merged]
    max_x = min_x.copy()
    min_y = [y for _x, y, _length in merged]
    max_y = [y + length - 1 for _x, y, length in merged]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        component_area[left_root] += component_area[right_root]
        min_x[left_root] = min(min_x[left_root], min_x[right_root])
        max_x[left_root] = max(max_x[left_root], max_x[right_root])
        min_y[left_root] = min(min_y[left_root], min_y[right_root])
        max_y[left_root] = max(max_y[left_root], max_y[right_root])

    previous_x: int | None = None
    previous_indices: list[int] = []
    cursor = 0
    while cursor < len(merged):
        x = merged[cursor][0]
        stop = cursor
        while stop < len(merged) and merged[stop][0] == x:
            stop += 1
        current_indices = list(range(cursor, stop))
        if previous_x is not None and x == previous_x + 1:
            for current in current_indices:
                _current_x, current_y, current_length = merged[current]
                current_stop = current_y + current_length
                for previous in previous_indices:
                    _previous_x, previous_y, previous_length = merged[previous]
                    previous_stop = previous_y + previous_length
                    if current_y <= previous_stop and previous_y <= current_stop:
                        union(current, previous)
        previous_x = x
        previous_indices = current_indices
        cursor = stop

    roots = {find(index) for index in range(len(merged))}
    winner = min(
        roots,
        key=lambda root: (
            -component_area[root],
            min_y[root],
            min_x[root],
            max_y[root],
            max_x[root],
            root,
        ),
    )
    return tuple(
        run for index, run in enumerate(merged) if find(index) == winner
    )


def _set_column_bit_range(
    payload: bytearray,
    *,
    column_stride: int,
    local_x: int,
    start: int,
    stop: int,
) -> None:
    if start >= stop:
        return
    base = local_x * column_stride
    first_byte, first_bit = divmod(start, 8)
    last_byte, last_bit = divmod(stop - 1, 8)
    if first_byte == last_byte:
        width = stop - start
        payload[base + first_byte] |= ((1 << width) - 1) << first_bit
        return
    payload[base + first_byte] |= (0xFF << first_bit) & 0xFF
    for byte_index in range(first_byte + 1, last_byte):
        payload[base + byte_index] = 0xFF
    payload[base + last_byte] |= (1 << (last_bit + 1)) - 1


def _memory_bounded_sam2_amg_type(upstream_type: type[Any]) -> type[Any]:
    """Drop SAM2's dead non-M2M logits before crop-level aggregation.

    The pinned SAM2 implementation returns ``low_res_masks`` from every
    prompt microbatch even though native automatic-mask generation does not
    read them again when ``use_m2m`` is false.  Upstream otherwise concatenates
    those CUDA tensors across all 64-by-64 prompts and native crop layers.
    """

    class _FullImageMemoryBoundedSAM2AMG(upstream_type):
        def _process_batch(self, *args: Any, **kwargs: Any) -> Any:
            data = super()._process_batch(*args, **kwargs)
            if getattr(self, "use_m2m", None) is False:
                keys = {key for key, _value in data.items()}
                if "low_res_masks" in keys:
                    del data["low_res_masks"]
            return data

    _FullImageMemoryBoundedSAM2AMG.__name__ = (
        "_FullImageMemoryBoundedSAM2AutomaticMaskGenerator"
    )
    return _FullImageMemoryBoundedSAM2AMG


def _attest_full_image_sam2_cuda_model(model: Any) -> int:
    """Use the canonical fail-closed model/device attestation."""

    from .model_loading import _attest_canonical_sam2_cuda_model

    return _attest_canonical_sam2_cuda_model(model)


def _attest_full_image_amg_cuda(
    generator: Any,
    *,
    expected_model: Any,
) -> int:
    """Prove the AMG retained the attested CUDA model and predictor device."""

    from .model_loading import _cuda_device_matches

    predictor = getattr(generator, "predictor", None)
    model = getattr(predictor, "model", None)
    if predictor is None or model is None:
        raise RuntimeError("full-image AMG does not expose its SAM2 predictor model")
    if model is not expected_model:
        raise RuntimeError("full-image AMG replaced the attested SAM2 model")
    cuda_ordinal = _attest_full_image_sam2_cuda_model(model)
    if not _cuda_device_matches(getattr(predictor, "device", None), cuda_ordinal):
        raise RuntimeError(
            f"full-image AMG predictor.device is not cuda:{cuda_ordinal}"
        )
    return cuda_ordinal


def _packed_mask_sha256(
    *,
    canvas_height: int,
    canvas_width: int,
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
    area: int,
    column_stride: int,
    packed: bytes,
) -> str:
    header = struct.pack(
        ">8Q",
        canvas_height,
        canvas_width,
        origin_x,
        origin_y,
        width,
        height,
        area,
        column_stride,
    )
    return hashlib.sha256(FULL_IMAGE_MASK_HASH_DOMAIN + header + packed).hexdigest()


def _validate_padding_bits(mask: PackedMask) -> None:
    remainder = mask.height % 8
    if remainder == 0:
        return
    allowed = (1 << remainder) - 1
    for local_x in range(mask.width):
        last_byte = mask.packed[(local_x + 1) * mask.column_stride - 1]
        if last_byte & ~allowed:
            raise ValueError("packed mask has nonzero column padding")


def _mask_iou_upper_bound(left: PackedMask, right: PackedMask) -> float:
    overlap = _bbox_intersection(left.bbox, right.bbox)
    if overlap is None:
        return 0.0
    _x, _y, width, height = overlap
    maximum_intersection = min(left.area, right.area, width * height)
    return maximum_intersection / (left.area + right.area - maximum_intersection)


def _bbox_intersection(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    left_x, left_y, left_width, left_height = left
    right_x, right_y, right_width, right_height = right
    x0 = max(left_x, right_x)
    y0 = max(left_y, right_y)
    x1 = min(left_x + left_width, right_x + right_width)
    y1 = min(left_y + left_height, right_y + right_height)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1 - x0, y1 - y0


def _image_canvas(image: Any) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if not isinstance(shape, tuple) or len(shape) not in (2, 3):
        raise ValueError("full-image input must expose an HxW or HxWxC shape")
    height = _positive_int(shape[0], "image height")
    width = _positive_int(shape[1], "image width")
    if len(shape) == 3 and _positive_int(shape[2], "image channels") not in (3, 4):
        raise ValueError("full-image input must have three or four channels")
    return height, width


def _canvas_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if not isinstance(shape, tuple) or len(shape) not in (2, 3):
        raise ValueError(
            "canvas_shape must be (height, width) or (height, width, channels)"
        )
    height = _positive_int(shape[0], "canvas height")
    width = _positive_int(shape[1], "canvas width")
    if len(shape) == 3:
        _positive_int(shape[2], "canvas channels")
    return height, width


def _positive_int(value: Any, role: str) -> int:
    number = _nonnegative_int(value, role)
    if number == 0:
        raise ValueError(f"{role} must be positive")
    return number


def _nonnegative_int(value: Any, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{role} must be an integer")
    number = int(value)
    if number < 0:
        raise ValueError(f"{role} must be nonnegative")
    if number >= 1 << 64:
        raise ValueError(f"{role} exceeds the supported bound")
    return number


def _unit_score(value: Any, role: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{role} must be numeric") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{role} must be in [0,1]")
    return number


def _open_unit_score(value: Any, role: str) -> float:
    number = _unit_score(value, role)
    if number == 0.0:
        raise ValueError(f"{role} must be greater than zero")
    return number


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "FULL_IMAGE_COMPONENT_POLICY",
    "FULL_IMAGE_DURABLE_MASK_ENCODING",
    "FULL_IMAGE_MASK_ENCODING",
    "FULL_IMAGE_MASK_HASH_DOMAIN",
    "FULL_IMAGE_PROPOSAL_HASH_DOMAIN",
    "FULL_IMAGE_RUNTIME_MAX_POINTS_PER_BATCH",
    "FullImageBackend",
    "FullImageProposal",
    "FullImageProposalSelection",
    "FullImageSettings",
    "PackedMask",
    "build_full_image_amg",
    "build_full_image_backend",
    "full_image_proposal_sha256",
    "pack_uncompressed_rle",
    "select_full_image_proposals",
]
