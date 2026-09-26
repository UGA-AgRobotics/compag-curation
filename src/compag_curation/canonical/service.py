"""Stage orchestration for the canonical public XGBoost profile.

The functions in this module are the narrow boundary used by the public run
facade.  Scientific operations remain owned by the canonical backend modules;
this module binds their identities, schemas, and no-clobber filesystem I/O.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from compag_curation.public_config import (
    CANONICAL_CPU_PROFILE,
    CANONICAL_GPU_PROFILE,
    CANONICAL_PROFILE,
    EFFICIENT_GPU_PROFILE,
    FULL_IMAGE_GPU_PROFILE,
    MAX_DECODED_BYTES,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_FILE_BYTES,
    MAX_IMAGE_PIXELS,
    RESNET50_WEIGHTS_SHA256,
    SUPPORTED_SUFFIXES,
    PublicProjectConfig,
    group_id_from_name,
    image_files,
)
from compag_curation.public_io import (
    PublicIOError,
    bounded_csv_field_limit,
    canonical_json_value,
    compact_json_sha256,
    fsync_directory,
    portable_basename,
    sha256_file,
    stable_file,
    write_new_bytes,
    write_new_json,
)
from compag_curation.review.exchange import (
    CANONICAL_REVIEW_COLUMNS,
    MAX_REVIEW_BYTES,
    MAX_REVIEW_ROWS,
    REVIEW_ACTION_WEIGHTS,
)

from .features import (
    CANONICAL_RAW_FEATURE_COLUMNS,
    CANONICAL_RAW_FEATURE_ORDER,
    CanonicalFeatureState,
    CanonicalGridContext,
    CanonicalRawFeature,
    CanonicalResNet50Embedder,
    canonical_raw_feature_csv_row,
    decode_canonical_embedding,
    extract_canonical_raw_features,
    finalize_canonical_feature,
    fit_canonical_feature_state,
)
from .full_image import (
    FULL_IMAGE_COMPONENT_POLICY,
    FULL_IMAGE_DURABLE_MASK_ENCODING,
    FULL_IMAGE_MASK_ENCODING,
    FullImageBackend,
    FullImageProposal,
    FullImageProposalSelection,
    FullImageSettings,
    PackedMask,
    build_full_image_backend,
)
from .full_image_features import (
    FULL_IMAGE_CONTEXT_FRACTION,
    FULL_IMAGE_CONTEXT_MINIMUM_PIXELS,
    FULL_IMAGE_FEATURE_PROPOSAL_BATCH,
    FULL_IMAGE_FEATURE_SEMANTICS_ID,
    extract_full_image_raw_features,
)
from .model_loading import load_canonical_sam2_model
from .preprocessing import (
    canonical_prepare_images,
    encode_canonical_jpeg,
    encode_full_image_png,
    full_image_prepare,
    full_image_unit_name,
)
from .proposals import (
    CanonicalProposal,
    CanonicalProposalSelection,
    _attest_canonical_amg_cuda,
    build_canonical_amg,
    canonical_mask_sha256,
    select_canonical_proposals,
)
from .serialization import (
    load_portable_predictor,
    portable_classifier_bytes,
    predict_portable_probabilities,
    serialize_canonical_feature_state,
    verify_probability_parity,
)
from .spec import (
    CANONICAL_DECISION_THRESHOLD,
    CANONICAL_EMBEDDING_DIMENSIONS,
    CANONICAL_FEATURE_CROP_SCALES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_FULL_IMAGE_NMS_IOU,
    CANONICAL_GROUP_FOLDS,
    CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
    CANONICAL_PCA_DIMENSIONS,
    CANONICAL_RANDOM_STATE,
    CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256,
    CANONICAL_SAM2_HIERA_L_CONFIG_SHA256,
    CANONICAL_STAGE20_AMG_POINTS_PER_BATCH,
    EFFICIENT_SAM2_HIERA_T_CHECKPOINT_SHA256,
    EFFICIENT_SAM2_HIERA_T_CONFIG_SHA256,
    GPU_EXECUTION_PROFILES,
    V2_PIPELINE_PROFILES,
    amg_settings_for_profile,
    inference_amg_execution_points_per_batch,
    proposal_identity_for_profile,
    stage20_amg_execution_points_per_batch,
)
from .training import (
    CanonicalGroupSplit,
    CanonicalTrainingConfig,
    canonical_tile_balanced_split,
    train_canonical_xgb,
    validate_canonical_review_weights,
)


MAX_CANONICAL_TABLE_BYTES = 2 * 1024 * 1024 * 1024
MAX_CANONICAL_TABLE_ROWS = 2_000_000
MASK_ENCODING = "base64-packbits-little-row-major-v1"

# Stage 10 retains the established public facade contract exactly.
CANONICAL_TILE_COLUMNS = (
    "tile_name",
    "image_id",
    "image_sha256",
    "image_name",
    "group_id",
    "tile_sha256",
    "x",
    "y",
    "crop_w",
    "crop_h",
    "orig_w",
    "orig_h",
)

# The historical filename ``tiles_index.csv`` remains part of the public run
# facade, but a full-image profile writes this explicit one-unit schema and no
# ``tiles/`` directory.  Compatibility names are introduced only later in the
# proposal table, whose stable columns are consumed by review/AL code.
FULL_IMAGE_UNIT_COLUMNS = (
    "unit_name",
    "image_id",
    "image_sha256",
    "image_name",
    "group_id",
    "unit_sha256",
    "x",
    "y",
    "width",
    "height",
    "orig_w",
    "orig_h",
    "processing_unit_kind",
)

CANONICAL_PROPOSAL_COLUMNS = (
    "proposal_id",
    "proposal_sha256",
    "image_id",
    "image_sha256",
    "image_name",
    "group_id",
    "tile_name",
    "tile_sha256",
    "tile_x",
    "tile_y",
    "source_index",
    "proposal_index",
    "mask_sha256",
    "mask_area",
    "mask_encoding",
    "mask_height",
    "mask_width",
    "mask_packbits_base64",
    "tile_bbox_x",
    "tile_bbox_y",
    "tile_bbox_w",
    "tile_bbox_h",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "poly",
    "predicted_iou",
    "stability_score",
)

_RAW_IDENTITY_COLUMNS = (
    "proposal_id",
    "proposal_sha256",
    "image_id",
    "image_sha256",
    "image_name",
    "group_id",
    "tile_name",
    "tile_sha256",
    "tile_x",
    "tile_y",
    "mask_sha256",
)
CANONICAL_RAW_FEATURE_TABLE_COLUMNS = (
    *_RAW_IDENTITY_COLUMNS,
    *CANONICAL_RAW_FEATURE_COLUMNS,
)

_FINAL_IDENTITY_COLUMNS = (
    *_RAW_IDENTITY_COLUMNS,
    "scale",
    "label",
    "review_action",
    "review_weight",
    "partition",
)
CANONICAL_FINAL_FEATURE_COLUMNS = (
    *_FINAL_IDENTITY_COLUMNS,
    *CANONICAL_FEATURE_ORDER,
)

CANONICAL_TEST_PREDICTION_COLUMNS = (
    "proposal_id",
    "image_id",
    "image_name",
    "group_id",
    "scale",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "poly",
    "label",
    "review_action",
    "review_weight",
    "xgb_p",
    "prediction",
)

CANONICAL_INFERENCE_COLUMNS = (
    "id",
    "detection_id",
    "proposal_id",
    "proposal_sha256",
    "image",
    "full_image",
    "image_id",
    "image_sha256",
    "group_id",
    "tile_name",
    "tile_sha256",
    "scale",
    "mask_sha256",
    "x",
    "y",
    "w",
    "h",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "orig_w",
    "orig_h",
    "poly",
    "xgb_p",
    "probability",
    "prediction",
    "kept",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CanonicalRuntimeAssets:
    sam2_config: Path
    sam2_checkpoint: Path
    sam2_config_locator: str
    sam2_config_sha256: str
    sam2_checkpoint_sha256: str
    resnet50_weights: Path
    resnet50_weights_sha256: str
    # Defaults to Full so existing dependency-injection tests and callers that
    # predate the dual-profile release retain their exact behavior.
    profile: str = CANONICAL_GPU_PROFILE


@dataclass(frozen=True)
class CanonicalStageRuntime:
    amg: Any
    embedder: Any
    profile: str = CANONICAL_GPU_PROFILE


CanonicalRuntimeFactory = Callable[[CanonicalRuntimeAssets], CanonicalStageRuntime]


@dataclass(frozen=True)
class _GeneratedTile:
    proposal_rows: tuple[dict[str, object], ...]
    raw_rows: tuple[dict[str, object], ...]
    raw_features: tuple[CanonicalRawFeature, ...]
    selection: CanonicalProposalSelection | FullImageProposalSelection


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


def _require_cuda_service_device(device: Any, role: str) -> str:
    normalized = str(device).strip().lower()
    _require(
        normalized == "cuda",
        f"canonical {role} is CUDA-only; CPU execution is unsupported",
    )
    return normalized


def _plain(value: Any) -> Any:
    """Convert numpy-style scalars recursively before canonical JSON writes."""

    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item_value) for key, item_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item_value) for item_value in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _strict_float(value: object, role: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PublicIOError(f"{role} is not numeric") from exc
    _require(math.isfinite(number), f"{role} is not finite")
    return number


def _strict_int(value: object, role: str, *, minimum: int = 0) -> int:
    text = str(value)
    _require(re.fullmatch(r"0|[1-9][0-9]*", text) is not None, f"{role} is not a canonical integer")
    number = int(text)
    _require(number >= minimum, f"{role} is below its minimum")
    return number


def _format_float(value: object) -> str:
    number = _strict_float(value, "floating-point output")
    return format(0.0 if number == 0.0 else number, ".17g")


def _safe_output_directory(output: Path) -> Path:
    absolute = output.absolute()
    _require(absolute == output.resolve(strict=True), "stage output contains a symlinked path component")
    info = absolute.lstat()
    _require(
        stat.S_ISDIR(info.st_mode) and not absolute.is_symlink(),
        "stage output is not a safe directory",
    )
    return absolute


def _write_csv_exact(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, object]],
    *,
    require_rows: bool = True,
) -> int:
    """Stream one exact-schema UTF-8 CSV to a fresh 0644 file."""

    path = path.absolute()
    _require(path.parent.is_dir() and not path.parent.is_symlink(), "CSV parent is unsafe")
    _require(not path.exists() and not path.is_symlink(), f"CSV output already exists: {path.name}")
    fields = tuple(columns)
    _require(fields and len(set(fields)) == len(fields), "CSV schema is invalid")
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o644,
    )
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="", closefd=False) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row in rows:
                _require(tuple(row) == fields, "CSV row schema or order changed")
                _require(count < MAX_CANONICAL_TABLE_ROWS, "CSV row bound exceeded")
                writer.writerow(row)
                count += 1
            handle.flush()
            os.fchmod(fd, 0o644)
            os.fsync(fd)
            info = os.fstat(fd)
            _require(
                stat.S_ISREG(info.st_mode)
                and info.st_nlink == 1
                and stat.S_IMODE(info.st_mode) == 0o644
                and info.st_size <= MAX_CANONICAL_TABLE_BYTES,
                "CSV output metadata or size is invalid",
            )
    finally:
        os.close(fd)
    _require(count > 0 or not require_rows, f"CSV output contains no rows: {path.name}")
    fsync_directory(path.parent)
    return count


def _parse_csv_exact(
    payload: bytes,
    name: str,
    columns: Sequence[str],
    *,
    max_rows: int = MAX_CANONICAL_TABLE_ROWS,
    require_rows: bool = True,
) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise PublicIOError(f"CSV input is not UTF-8: {name}") from exc
    try:
        with bounded_csv_field_limit():
            with io.StringIO(text, newline="") as handle:
                reader = csv.DictReader(handle)
                expected = tuple(columns)
                _require(tuple(reader.fieldnames or ()) == expected, f"CSV header mismatch: {name}")
                rows: list[dict[str, str]] = []
                for row in reader:
                    _require(len(rows) < max_rows, f"CSV row bound exceeded: {name}")
                    _require(None not in row and tuple(row) == expected, f"CSV row width mismatch: {name}")
                    _require(all(isinstance(item, str) for item in row.values()), f"CSV row is incomplete: {name}")
                    rows.append(row)
    except csv.Error as exc:
        raise PublicIOError(f"CSV framing is invalid: {name}") from exc
    _require(rows or not require_rows, f"CSV input contains no rows: {name}")
    return rows


def _read_csv_exact(
    path: Path,
    columns: Sequence[str],
    *,
    max_bytes: int = MAX_CANONICAL_TABLE_BYTES,
    max_rows: int = MAX_CANONICAL_TABLE_ROWS,
    require_rows: bool = True,
) -> list[dict[str, str]]:
    payload, _snapshot = stable_file(path, max_bytes=max_bytes)
    return _parse_csv_exact(
        payload,
        path.name,
        columns,
        max_rows=max_rows,
        require_rows=require_rows,
    )


def _read_canonical_json(path: Path, role: str, *, max_bytes: int = 16 * 1024 * 1024) -> Mapping[str, object]:
    payload, _snapshot = stable_file(path, max_bytes=max_bytes)
    value = canonical_json_value(payload, role)
    _require(isinstance(value, dict), f"{role} must be a JSON object")
    return value


def _canonical_device(profile: str) -> str:
    _require(
        profile in V2_PIPELINE_PROFILES,
        "v2 service requires a supported pipeline profile",
    )
    return {
        CANONICAL_CPU_PROFILE: "cpu",
        CANONICAL_GPU_PROFILE: "cuda",
        EFFICIENT_GPU_PROFILE: "cuda",
        FULL_IMAGE_GPU_PROFILE: "cuda",
    }[profile]


def _stage20_execution_batch(profile: str) -> int:
    """Preserve historical CPU evidence while sealing live GPU profiles."""

    if profile == CANONICAL_CPU_PROFILE:
        return CANONICAL_STAGE20_AMG_POINTS_PER_BATCH
    return stage20_amg_execution_points_per_batch(profile)


def _inference_execution_batch(profile: str) -> int:
    """Preserve historical CPU evidence while sealing live GPU profiles."""

    if profile == CANONICAL_CPU_PROFILE:
        return CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH
    return inference_amg_execution_points_per_batch(profile)


def _canonical_config(config: PublicProjectConfig) -> None:
    _require(
        config.profile in V2_PIPELINE_PROFILES,
        "v2 service requires a supported pipeline profile",
    )
    expected_device = _canonical_device(config.profile)
    _require(
        config.device == expected_device and config.seed == CANONICAL_RANDOM_STATE,
        "canonical profile/device/seed identity changed",
    )
    _require(
        config.tile_size == 512
        and config.tile_stride == 512
        and config.tile_overlap == 0,
        "v2 compatibility spatial settings changed",
    )
    settings = amg_settings_for_profile(config.profile)
    expected_checkpoint_sha256 = (
        EFFICIENT_SAM2_HIERA_T_CHECKPOINT_SHA256
        if config.profile == EFFICIENT_GPU_PROFILE
        else CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256
    )
    expected_config_sha256 = (
        EFFICIENT_SAM2_HIERA_T_CONFIG_SHA256
        if config.profile == EFFICIENT_GPU_PROFILE
        else CANONICAL_SAM2_HIERA_L_CONFIG_SHA256
    )
    _require(
        config.checkpoint_sha256 == expected_checkpoint_sha256,
        "v2 checkpoint identity changed",
    )
    _require(
        config.sam2_config_sha256 == expected_config_sha256
        and config.sam2_config_locator == settings.config_locator,
        "v2 SAM2 configuration identity changed",
    )
    _require(config.embedding_weights is not None, "canonical ResNet50 weights are missing")
    _require(config.embedding_weights_sha256 == RESNET50_WEIGHTS_SHA256, "canonical ResNet50 identity changed")
    _require(config.yolo_enabled is False and tuple(config.proposal_scales) == (1.0,), "canonical proposal route changed")
    _require(tuple(config.feature_crop_scales) == CANONICAL_FEATURE_CROP_SCALES, "canonical feature scales changed")
    _require(config.inference_threshold == 0.5 and config.nms_iou_threshold == 0.5, "canonical inference policy changed")
    if config.profile == FULL_IMAGE_GPU_PROFILE:
        _require(
            config.spatial_mode == "full-image-multiscale"
            and config.max_proposals_per_image == settings.max_masks_per_image
            and config.execution_points_per_batch == 8
            and settings.crop_n_layers == 2
            and settings.crop_n_points_downscale_factor == 1
            and settings.crop_overlap_ratio == 512 / 1500,
            "full-image multiscale method identity changed",
        )
    else:
        _require(
            getattr(config, "spatial_mode", "tiled") == "tiled"
            and getattr(config, "max_proposals_per_image", None) is None
            and getattr(config, "execution_points_per_batch", None) is None,
            "tiled profile spatial identity changed",
        )


def _image_id(filename: str, group_id: str, image_sha256: str) -> str:
    return hashlib.sha256(
        b"compag-image-v1\0"
        + filename.encode("utf-8")
        + b"\0"
        + group_id.encode("ascii")
        + b"\0"
        + image_sha256.encode("ascii")
    ).hexdigest()


def _proposal_id(
    image_id: str,
    tile_name: str,
    tile_sha256: str,
    tile_x: int,
    tile_y: int,
    proposal: CanonicalProposal | FullImageProposal,
    *,
    profile: str = CANONICAL_GPU_PROFILE,
) -> str:
    payload = (
        proposal_identity_for_profile(profile).encode("ascii") + b"\0"
        + image_id.encode("ascii")
        + b"\0"
        + tile_name.encode("utf-8")
        + b"\0"
        + tile_sha256.encode("ascii")
        + b"\0"
        + str(tile_x).encode("ascii")
        + b"\0"
        + str(tile_y).encode("ascii")
        + b"\0"
        + proposal.mask_sha256.encode("ascii")
        + b"\0"
        + (
            proposal.proposal_sha256.encode("ascii")
            if profile == FULL_IMAGE_GPU_PROFILE
            else str(proposal.source_index).encode("ascii")
        )
    )
    return hashlib.sha256(payload).hexdigest()


def _decode_image(payload: bytes, role: str) -> Any:
    import cv2
    import numpy as np

    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    _require(
        decoded is not None
        and decoded.ndim == 3
        and decoded.shape[2] == 3
        and decoded.dtype == np.uint8,
        f"image decoder rejected {role}",
    )
    _require(decoded.shape[0] >= 1 and decoded.shape[1] >= 1, f"decoded image is empty: {role}")
    return decoded


def _encode_mask(mask: Any) -> tuple[int, int, str]:
    import numpy as np

    binary = np.asarray(mask) > 0
    _require(binary.ndim == 2 and binary.size > 0, "canonical mask is invalid")
    height, width = (int(value) for value in binary.shape)
    packed = np.packbits(binary.reshape(-1), bitorder="little").tobytes()
    return height, width, base64.b64encode(packed).decode("ascii")


def decode_lossless_mask(height: object, width: object, encoded: str) -> Any:
    """Decode and validate the lossless mask representation in proposals.csv."""

    import binascii
    import numpy as np

    h = _strict_int(height, "mask height", minimum=1)
    w = _strict_int(width, "mask width", minimum=1)
    _require(h <= 512 and w <= 512, "canonical mask shape exceeds a tile")
    try:
        payload = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeError, binascii.Error) as exc:
        raise PublicIOError("canonical mask is not strict base64") from exc
    _require(len(payload) == (h * w + 7) // 8, "canonical mask payload length is invalid")
    unpacked = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    if len(unpacked) > h * w:
        _require(not unpacked[h * w :].any(), "canonical mask padding bits are nonzero")
    mask = np.ascontiguousarray(unpacked[: h * w].reshape(h, w), dtype=np.uint8)
    mask.setflags(write=False)
    return mask


def _encode_full_image_mask(mask: PackedMask) -> tuple[int, int, str]:
    return (
        mask.canvas_height,
        mask.canvas_width,
        base64.b64encode(mask.packed).decode("ascii"),
    )


def _decode_full_image_packed_mask(row: Mapping[str, str]) -> PackedMask:
    """Reconstruct and fully verify a compact full-image proposal mask."""

    import binascii

    canvas_height = _strict_int(row["mask_height"], "mask canvas height", minimum=1)
    canvas_width = _strict_int(row["mask_width"], "mask canvas width", minimum=1)
    _require(
        canvas_height <= MAX_IMAGE_DIMENSION
        and canvas_width <= MAX_IMAGE_DIMENSION
        and canvas_height * canvas_width <= MAX_IMAGE_PIXELS,
        "full-image mask canvas exceeds supported bounds",
    )
    origin_x = _strict_int(row["tile_bbox_x"], "mask crop x")
    origin_y = _strict_int(row["tile_bbox_y"], "mask crop y")
    width = _strict_int(row["tile_bbox_w"], "mask crop width", minimum=1)
    height = _strict_int(row["tile_bbox_h"], "mask crop height", minimum=1)
    area = _strict_int(row["mask_area"], "mask area", minimum=1)
    try:
        packed = base64.b64decode(
            row["mask_packbits_base64"].encode("ascii"),
            validate=True,
        )
    except (UnicodeError, binascii.Error) as exc:
        raise PublicIOError("full-image compact mask is not strict base64") from exc
    _require(
        len(packed) == width * ((height + 7) // 8),
        "full-image compact mask payload length is invalid",
    )
    try:
        return PackedMask(
            canvas_height=canvas_height,
            canvas_width=canvas_width,
            origin_x=origin_x,
            origin_y=origin_y,
            width=width,
            height=height,
            area=area,
            column_stride=(height + 7) // 8,
            packed=packed,
            mask_sha256=row["mask_sha256"],
        )
    except (TypeError, ValueError) as exc:
        raise PublicIOError("full-image compact mask failed validation") from exc


def load_canonical_runtime(
    assets: CanonicalRuntimeAssets,
    *,
    device: str = "cuda",
    amg_execution_points_per_batch: int | None = None,
) -> CanonicalStageRuntime:
    """Load the profile-bound SAM2 AMG and ResNet50 extractor once."""

    resolved_device = _require_cuda_service_device(device, "runtime loading")
    settings = amg_settings_for_profile(assets.profile)
    allowed_execution_batches = {
        settings.points_per_batch,
        inference_amg_execution_points_per_batch(assets.profile),
        stage20_amg_execution_points_per_batch(assets.profile),
    }
    if amg_execution_points_per_batch is not None:
        _require(
            type(amg_execution_points_per_batch) is int
            and amg_execution_points_per_batch in allowed_execution_batches,
            "profile AMG execution microbatch is unsupported",
        )
    model = load_canonical_sam2_model(
        assets.sam2_config,
        assets.sam2_checkpoint,
        assets.sam2_config_locator,
        assets.sam2_config_sha256,
        assets.sam2_checkpoint_sha256,
        device=resolved_device,
        profile=assets.profile,
    )
    if assets.profile == FULL_IMAGE_GPU_PROFILE:
        execution_batch = (
            int(amg_execution_points_per_batch)
            if amg_execution_points_per_batch is not None
            else int(getattr(settings, "execution_points_per_batch", 8))
        )
        amg = build_full_image_backend(
            model,
            settings=FullImageSettings(
                points_per_side=settings.points_per_side,
                points_per_batch=execution_batch,
                pred_iou_thresh=settings.pred_iou_thresh,
                stability_score_thresh=settings.stability_score_thresh,
                crop_n_layers=settings.crop_n_layers,
                crop_n_points_downscale_factor=(
                    settings.crop_n_points_downscale_factor
                ),
                crop_overlap_ratio=settings.crop_overlap_ratio,
                merge_iou_threshold=settings.merge_iou_threshold,
                max_proposals=settings.max_masks_per_image,
            ),
        )
    elif amg_execution_points_per_batch is None:
        amg = build_canonical_amg(model, profile=assets.profile)
    else:
        amg = build_canonical_amg(
            model,
            profile=assets.profile,
            execution_points_per_batch=amg_execution_points_per_batch,
        )
    embedder = CanonicalResNet50Embedder.from_local_weights(
        assets.resnet50_weights,
        assets.resnet50_weights_sha256,
        device=resolved_device,
    )
    runtime = CanonicalStageRuntime(
        amg=amg,
        embedder=embedder,
        profile=assets.profile,
    )
    _attest_canonical_stage_runtime_cuda(runtime)
    return runtime


def _attest_canonical_stage_runtime_cuda(runtime: CanonicalStageRuntime) -> None:
    amg = runtime.amg
    if runtime.profile == FULL_IMAGE_GPU_PROFILE:
        _require(
            isinstance(amg, FullImageBackend),
            "full-image runtime does not use the bounded backend",
        )
        amg = getattr(amg, "generator", getattr(amg, "_generator", None))
    _attest_canonical_amg_cuda(amg)
    attest_embedder = getattr(runtime.embedder, "_attest_cuda_state", None)
    _require(
        callable(attest_embedder),
        "canonical runtime embedder cannot attest CUDA state",
    )
    attest_embedder()


def _load_stage_runtime(
    assets: CanonicalRuntimeAssets,
    device: str,
    runtime_factory: CanonicalRuntimeFactory | None,
    *,
    amg_execution_points_per_batch: int | None = None,
) -> CanonicalStageRuntime:
    """Route production runtimes to the requested device, preserving test factories."""

    if runtime_factory is not None:
        _require(
            str(device).strip().lower() in {"cpu", "cuda"},
            "canonical runtime device is unsupported",
        )
        runtime = runtime_factory(assets)
        if assets.profile in GPU_EXECUTION_PROFILES:
            _require(
                getattr(runtime, "profile", None) == assets.profile,
                "injected runtime profile differs from its profile-bound assets",
            )
        return runtime
    resolved_device = _require_cuda_service_device(device, "runtime loading")
    if amg_execution_points_per_batch is None:
        return load_canonical_runtime(assets, device=resolved_device)
    return load_canonical_runtime(
        assets,
        device=resolved_device,
        amg_execution_points_per_batch=amg_execution_points_per_batch,
    )


def _load_inference_runtime(
    assets: CanonicalRuntimeAssets,
    device: str,
    runtime_factory: CanonicalRuntimeFactory | None,
) -> CanonicalStageRuntime:
    """Load inference with the sealed, VRAM-safe CUDA prompt microbatch."""

    profile = getattr(assets, "profile", CANONICAL_GPU_PROFILE)
    # CPU-profile bundles are historical read/test fixtures; production
    # execution rejects CPU before reaching this loader.  Their v2 AMG method
    # used the same sealed batch as Full.
    return _load_stage_runtime(
        assets,
        device,
        runtime_factory,
        amg_execution_points_per_batch=_inference_execution_batch(profile),
    )


def _load_stage20_runtime(
    assets: CanonicalRuntimeAssets,
    device: str,
    runtime_factory: CanonicalRuntimeFactory | None,
) -> CanonicalStageRuntime:
    """Load Stage 20 with its sealed execution-only CUDA prompt microbatch."""

    profile = getattr(assets, "profile", CANONICAL_GPU_PROFILE)
    return _load_stage_runtime(
        assets,
        device,
        runtime_factory,
        amg_execution_points_per_batch=_stage20_execution_batch(profile),
    )


def _load_bundle_predictor(
    classifier: bytes,
    imputer: Sequence[Any],
    device: str,
    predictor_loader: Callable[[bytes, Sequence[Any]], Any] | None,
) -> Any:
    """Load portable XGBoost on the explicit device without implicit fallback."""

    if predictor_loader is not None:
        _require(
            str(device).strip().lower() in {"cpu", "cuda"},
            "canonical inference device is unsupported",
        )
        return predictor_loader(classifier, imputer)
    resolved_device = _require_cuda_service_device(device, "predictor loading")
    return load_portable_predictor(classifier, imputer, device=resolved_device)


def _full_image_analysis_unit(
    prepared: Any,
    role: str,
    *,
    image_name: str,
) -> Any:
    """Validate and return the sole native analysis unit for a full-image path."""

    import numpy as np

    units = getattr(prepared, "tiles", None)
    warped = np.asarray(getattr(prepared, "warped_bgr", None))
    _require(
        isinstance(units, tuple)
        and len(units) == 1
        and warped.ndim == 3
        and warped.shape[2] == 3
        and warped.dtype == np.uint8
        and warped.size > 0,
        f"{role} must expose exactly one full-image analysis unit",
    )
    unit = units[0]
    pixels = np.asarray(getattr(unit, "pixels", None))
    evidence = getattr(prepared, "evidence", None)
    analysis_units = (
        evidence.get("analysis_units") if isinstance(evidence, dict) else None
    )
    _require(
        getattr(unit, "name", None) == full_image_unit_name(image_name)
        and int(unit.x) == 0
        and int(unit.y) == 0
        and int(unit.crop_width) == int(warped.shape[1])
        and int(unit.crop_height) == int(warped.shape[0])
        and pixels.shape == warped.shape
        and pixels.dtype == np.uint8
        and bool(np.array_equal(pixels, warped))
        and isinstance(analysis_units, dict)
        and analysis_units
        == {
            "kind": "full_warped_frame",
            "count": 1,
            "external_tiling": False,
            "format": "png",
            "lossless": True,
            "width": int(warped.shape[1]),
            "height": int(warped.shape[0]),
        },
        f"{role} full-image analysis-unit geometry changed",
    )
    return unit


def prepare_canonical_stage(
    config: PublicProjectConfig,
    output: Path,
    *,
    prepare_image: Callable[..., Any] = canonical_prepare_images,
    jpeg_encoder: Callable[[Any], bytes] = encode_canonical_jpeg,
    full_image_preparer: Callable[..., Any] = full_image_prepare,
    png_encoder: Callable[[Any], bytes] = encode_full_image_png,
) -> dict[str, object]:
    """Run the profile-bound warp/grid/spatial preparation stage."""

    _canonical_config(config)
    output = _safe_output_directory(output)
    full_image = config.profile == FULL_IMAGE_GPU_PROFILE
    units_root = output / ("processing_units" if full_image else "tiles")
    _require(
        not units_root.exists() and not units_root.is_symlink(),
        "profile processing-unit directory already exists",
    )
    units_root.mkdir(mode=0o755)
    fsync_directory(output)

    unit_rows: list[dict[str, object]] = []
    image_records: list[dict[str, object]] = []
    unit_names: set[str] = set()
    for source in image_files(config):
        payload, _snapshot = stable_file(source, max_bytes=MAX_IMAGE_FILE_BYTES)
        image_sha256 = hashlib.sha256(payload).hexdigest()
        group_id = group_id_from_name(source.name)
        image_id = _image_id(source.name, group_id, image_sha256)
        decoded = _decode_image(payload, source.name)
        source_height, source_width = (int(value) for value in decoded.shape[:2])
        prepared = (
            full_image_preparer(decoded, source.name, locate_grid=True)
            if full_image
            else prepare_image(decoded, source.name, locate_grid=True)
        )
        warped_height, warped_width = (int(value) for value in prepared.warped_bgr.shape[:2])
        evidence = _plain(prepared.evidence)
        _require(
            isinstance(evidence, dict)
            and evidence.get("schema")
            == (
                "compag-curation-full-image-preprocessing/v1"
                if full_image
                else "compag-curation-canonical-preprocessing/v1"
            ),
            "profile preparation evidence is invalid",
        )
        if full_image:
            _full_image_analysis_unit(
                prepared,
                "full-image preparation",
                image_name=source.name,
            )
        for unit in prepared.tiles:
            portable_basename(unit.name, "profile processing-unit name")
            _require(
                unit.name not in unit_names,
                f"profile processing-unit name collision: {unit.name}",
            )
            unit_names.add(unit.name)
            encoded = bytes(png_encoder(unit) if full_image else jpeg_encoder(unit))
            if full_image:
                _require(
                    encoded.startswith(b"\x89PNG\r\n\x1a\n"),
                    "full-image processing unit is not PNG",
                )
            else:
                _require(
                    encoded.startswith(b"\xff\xd8") and encoded.endswith(b"\xff\xd9"),
                    "canonical tile is not JPEG",
                )
            target = units_root / unit.name
            write_new_bytes(target, encoded)
            digest = hashlib.sha256(encoded).hexdigest()
            if full_image:
                unit_rows.append(
                    {
                        "unit_name": unit.name,
                        "image_id": image_id,
                        "image_sha256": image_sha256,
                        "image_name": source.name,
                        "group_id": group_id,
                        "unit_sha256": digest,
                        "x": 0,
                        "y": 0,
                        "width": warped_width,
                        "height": warped_height,
                        "orig_w": source_width,
                        "orig_h": source_height,
                        "processing_unit_kind": "full_warped_frame",
                    }
                )
            else:
                unit_rows.append(
                    {
                    "tile_name": unit.name,
                    "image_id": image_id,
                    "image_sha256": image_sha256,
                    "image_name": source.name,
                    "group_id": group_id,
                    "tile_sha256": digest,
                    "x": int(unit.x),
                    "y": int(unit.y),
                    "crop_w": int(unit.crop_width),
                    "crop_h": int(unit.crop_height),
                    "orig_w": source_width,
                    "orig_h": source_height,
                }
                )
        image_record: dict[str, object] = {
                "image_name": source.name,
                "image_id": image_id,
                "image_sha256": image_sha256,
                "group_id": group_id,
                "source_width": source_width,
                "source_height": source_height,
                "warped_width": warped_width,
                "warped_height": warped_height,
                "warp_mode": prepared.warp_mode,
                "inverse_warp": evidence["warp"]["inverse_matrix"],
                "row_lines": list(prepared.row_lines),
                "column_lines": list(prepared.column_lines),
                "preprocessing_evidence_sha256": compact_json_sha256(evidence),
            }
        image_record[
            "processing_unit_count" if full_image else "tile_count"
        ] = len(prepared.tiles)
        image_records.append(image_record)

    unit_rows.sort(
        key=lambda row: (
            str(row["image_id"]),
            int(row["y"]),
            int(row["x"]),
            str(row["unit_name"] if full_image else row["tile_name"]),
        )
    )
    image_records.sort(key=lambda row: (str(row["image_id"]), str(row["image_name"])))
    _require(unit_rows and image_records, "profile preparation produced no units")
    _write_csv_exact(
        output / "tiles_index.csv",
        FULL_IMAGE_UNIT_COLUMNS if full_image else CANONICAL_TILE_COLUMNS,
        unit_rows,
    )
    if full_image:
        summary = {
            "schema": "compag-curation-full-image-prepare/v1",
            "status": "PASS",
            "profile": config.profile,
            "spatial_mode": "full-image-multiscale",
            "image_count": len(image_records),
            "group_count": len({str(row["group_id"]) for row in image_records}),
            "processing_unit_count": len(unit_rows),
            "processing_unit_kind": "full_warped_frame",
            "external_tiling": False,
            "unit_format": "png",
            "unit_lossless": True,
            "tiles_index_sha256": sha256_file(output / "tiles_index.csv"),
            "images": image_records,
        }
    else:
        summary = {
            "schema": "compag-curation-canonical-prepare/v1",
            "status": "PASS",
            "profile": config.profile,
            "image_count": len(image_records),
            "group_count": len({str(row["group_id"]) for row in image_records}),
            "tile_count": len(unit_rows),
            "tile_size": 512,
            "tile_stride": 512,
            "far_edge_alignment": True,
            "padding": "bottom-right-edge-value",
            "tile_format": "jpg",
            "tiles_index_sha256": sha256_file(output / "tiles_index.csv"),
            "images": image_records,
        }
    write_new_json(output / "prepare_summary.json", summary)
    return summary


def _prepared_image_records(
    prepared: Path,
    *,
    profile: str = CANONICAL_PROFILE,
) -> dict[str, Mapping[str, object]]:
    _canonical_device(profile)
    summary = _read_canonical_json(
        prepared / "prepare_summary.json",
        "canonical prepare summary",
        max_bytes=64 * 1024 * 1024,
    )
    full_image = profile == FULL_IMAGE_GPU_PROFILE
    _require(
        summary.get("schema")
        == (
            "compag-curation-full-image-prepare/v1"
            if full_image
            else "compag-curation-canonical-prepare/v1"
        )
        and summary.get("status") == "PASS"
        and summary.get("profile") == profile,
        "profile prepare summary identity changed",
    )
    if full_image:
        _require(
            set(summary)
            == {
                "schema",
                "status",
                "profile",
                "spatial_mode",
                "image_count",
                "group_count",
                "processing_unit_count",
                "processing_unit_kind",
                "external_tiling",
                "unit_format",
                "unit_lossless",
                "tiles_index_sha256",
                "images",
            }
            and summary.get("spatial_mode") == "full-image-multiscale"
            and summary.get("processing_unit_kind") == "full_warped_frame"
            and summary.get("external_tiling") is False
            and summary.get("unit_format") == "png"
            and summary.get("unit_lossless") is True,
            "full-image prepare summary contract changed",
        )
    _require(summary.get("tiles_index_sha256") == sha256_file(prepared / "tiles_index.csv"), "canonical tile index identity changed")
    raw_images = summary.get("images")
    _require(isinstance(raw_images, list) and raw_images, "canonical prepare image records are missing")
    records: dict[str, Mapping[str, object]] = {}
    for value in raw_images:
        _require(isinstance(value, dict), "canonical prepare image record is invalid")
        identity = str(value.get("image_id", ""))
        _require(_SHA256.fullmatch(identity) is not None and identity not in records, "canonical prepare image identity is invalid")
        inverse = value.get("inverse_warp")
        _require(
            isinstance(inverse, list)
            and len(inverse) == 3
            and all(isinstance(row, list) and len(row) == 3 for row in inverse),
            "canonical inverse warp is invalid",
        )
        for number in (item for row in inverse for item in row):
            _strict_float(number, "canonical inverse warp element")
        for name in ("source_width", "source_height", "warped_width", "warped_height"):
            _strict_int(value.get(name), f"canonical prepare {name}", minimum=1)
        count_name = "processing_unit_count" if full_image else "tile_count"
        _require(
            _strict_int(value.get(count_name), f"prepare {count_name}", minimum=1)
            == (1 if full_image else int(value[count_name])),
            "profile prepare unit count is invalid",
        )
        records[identity] = value
    if full_image:
        _require(
            summary.get("image_count") == len(records)
            and summary.get("processing_unit_count") == len(records),
            "full-image prepare count closure changed",
        )
    return records


def _prepared_processing_rows(
    prepared: Path,
    *,
    profile: str,
) -> tuple[list[dict[str, str]], Path]:
    """Read Stage-10 rows and adapt full units to stable proposal aliases."""

    if profile != FULL_IMAGE_GPU_PROFILE:
        return (
            _read_csv_exact(prepared / "tiles_index.csv", CANONICAL_TILE_COLUMNS),
            prepared / "tiles",
        )
    rows = _read_csv_exact(prepared / "tiles_index.csv", FULL_IMAGE_UNIT_COLUMNS)
    adapted: list[dict[str, str]] = []
    for row in rows:
        width = _strict_int(row["width"], "full-image unit width", minimum=1)
        height = _strict_int(row["height"], "full-image unit height", minimum=1)
        _require(
            row["processing_unit_kind"] == "full_warped_frame"
            and row["x"] == "0"
            and row["y"] == "0",
            "full-image processing-unit contract changed",
        )
        adapted.append(
            {
                "tile_name": row["unit_name"],
                "image_id": row["image_id"],
                "image_sha256": row["image_sha256"],
                "image_name": row["image_name"],
                "group_id": row["group_id"],
                "tile_sha256": row["unit_sha256"],
                "x": "0",
                "y": "0",
                "crop_w": str(width),
                "crop_h": str(height),
                "orig_w": row["orig_w"],
                "orig_h": row["orig_h"],
            }
        )
    _require(
        len({row["image_id"] for row in adapted}) == len(adapted),
        "full-image Stage-10 must contain exactly one unit per image",
    )
    return adapted, prepared / "processing_units"


def _runtime_assets_from_config(config: PublicProjectConfig) -> CanonicalRuntimeAssets:
    _canonical_config(config)
    _require(config.embedding_weights is not None and config.embedding_weights_sha256 is not None, "canonical embedding asset is missing")
    return CanonicalRuntimeAssets(
        sam2_config=config.sam2_config,
        sam2_checkpoint=config.checkpoint,
        sam2_config_locator=config.sam2_config_locator,
        sam2_config_sha256=config.sam2_config_sha256,
        sam2_checkpoint_sha256=config.checkpoint_sha256,
        resnet50_weights=config.embedding_weights,
        resnet50_weights_sha256=config.embedding_weights_sha256,
        profile=config.profile,
    )


def _mask_geometry(
    proposal: CanonicalProposal | FullImageProposal,
    tile_row: Mapping[str, object],
    image_record: Mapping[str, object],
) -> tuple[list[int], tuple[int, int, int, int]]:
    """Map a tile mask through the recorded inverse warp using source math."""

    import numpy as np
    from compag_curation.proposals.sam2_pipeline.gate_core import (
        mask_to_poly_and_bbox_in_original,
    )

    if isinstance(proposal, FullImageProposal):
        tile_x, tile_y, crop_width, crop_height = proposal.bbox
        valid_mask = np.ascontiguousarray(proposal.mask_crop, dtype=np.uint8)
        _require(
            valid_mask.shape == (crop_height, crop_width),
            "full-image proposal mask/crop geometry is invalid",
        )
    else:
        crop_width = _strict_int(tile_row["crop_w"], "tile crop width", minimum=1)
        crop_height = _strict_int(tile_row["crop_h"], "tile crop height", minimum=1)
        tile_x = _strict_int(tile_row["x"], "tile x")
        tile_y = _strict_int(tile_row["y"], "tile y")
        mask = np.asarray(proposal.mask, dtype=np.uint8)
        _require(mask.ndim == 2 and crop_height <= mask.shape[0] and crop_width <= mask.shape[1], "proposal mask/crop geometry is invalid")
        valid_mask = np.ascontiguousarray(mask[:crop_height, :crop_width], dtype=np.uint8)
    inverse = np.asarray(image_record["inverse_warp"], dtype=np.float32)
    translation = np.asarray(
        [[1.0, 0.0, float(tile_x)], [0.0, 1.0, float(tile_y)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    local_to_original = np.asarray(inverse @ translation, dtype=np.float32)
    source_height = _strict_int(image_record["source_height"], "source height", minimum=1)
    source_width = _strict_int(image_record["source_width"], "source width", minimum=1)
    polygon, bbox = mask_to_poly_and_bbox_in_original(
        valid_mask,
        local_to_original,
        (source_height, source_width, 3),
    )
    _require(
        isinstance(polygon, list)
        and len(polygon) % 2 == 0
        and isinstance(bbox, tuple)
        and len(bbox) == 4,
        "source polygon conversion returned invalid geometry",
    )
    return [int(value) for value in polygon], tuple(int(value) for value in bbox)


def _generate_tile_records(
    runtime: CanonicalStageRuntime,
    tile_bgr: Any,
    tile_row: Mapping[str, object],
    image_record: Mapping[str, object],
    *,
    attest_cuda: bool = True,
    extra_annotations: Sequence[Mapping[str, Any]] = (),
) -> _GeneratedTile:
    import cv2

    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    if attest_cuda:
        _attest_canonical_amg_cuda(runtime.amg)
    annotations = list(runtime.amg.generate(rgb) or [])
    if extra_annotations:
        annotations.extend(extra_annotations)
    if attest_cuda:
        _attest_canonical_amg_cuda(runtime.amg)
    selection = select_canonical_proposals(annotations)
    grid = CanonicalGridContext(
        row_lines=tuple(int(value) for value in image_record["row_lines"]),
        column_lines=tuple(int(value) for value in image_record["column_lines"]),
        tile_x=_strict_int(tile_row["x"], "tile x"),
        tile_y=_strict_int(tile_row["y"], "tile y"),
    )
    raw_features = extract_canonical_raw_features(
        tile_bgr,
        selection.proposals,
        runtime.embedder,
        grid=grid,
    )
    _require(
        len(raw_features) == len(selection.proposals) * len(CANONICAL_FEATURE_CROP_SCALES),
        "canonical raw feature row count changed",
    )

    proposal_rows: list[dict[str, object]] = []
    identity_by_index: dict[int, dict[str, object]] = {}
    for proposal in selection.proposals:
        proposal_id = _proposal_id(
            str(tile_row["image_id"]),
            str(tile_row["tile_name"]),
            str(tile_row["tile_sha256"]),
            _strict_int(tile_row["x"], "tile x"),
            _strict_int(tile_row["y"], "tile y"),
            proposal,
            profile=runtime.profile,
        )
        height, width, encoded_mask = _encode_mask(proposal.mask)
        polygon, (bbox_x, bbox_y, bbox_w, bbox_h) = _mask_geometry(
            proposal,
            tile_row,
            image_record,
        )
        tile_bbox_x, tile_bbox_y, tile_bbox_w, tile_bbox_h = proposal.bbox
        row: dict[str, object] = {
            "proposal_id": proposal_id,
            "proposal_sha256": proposal_id,
            "image_id": str(tile_row["image_id"]),
            "image_sha256": str(tile_row["image_sha256"]),
            "image_name": str(tile_row["image_name"]),
            "group_id": str(tile_row["group_id"]),
            "tile_name": str(tile_row["tile_name"]),
            "tile_sha256": str(tile_row["tile_sha256"]),
            "tile_x": str(tile_row["x"]),
            "tile_y": str(tile_row["y"]),
            "source_index": str(proposal.source_index),
            "proposal_index": str(proposal.proposal_index),
            "mask_sha256": proposal.mask_sha256,
            "mask_area": str(proposal.area),
            "mask_encoding": MASK_ENCODING,
            "mask_height": str(height),
            "mask_width": str(width),
            "mask_packbits_base64": encoded_mask,
            "tile_bbox_x": str(tile_bbox_x),
            "tile_bbox_y": str(tile_bbox_y),
            "tile_bbox_w": str(tile_bbox_w),
            "tile_bbox_h": str(tile_bbox_h),
            "bbox_x": str(bbox_x),
            "bbox_y": str(bbox_y),
            "bbox_w": str(bbox_w),
            "bbox_h": str(bbox_h),
            "poly": json.dumps(polygon, ensure_ascii=True, separators=(",", ":")),
            "predicted_iou": _format_float(proposal.predicted_iou),
            "stability_score": _format_float(proposal.stability_score),
        }
        _require(tuple(row) == CANONICAL_PROPOSAL_COLUMNS, "canonical proposal schema changed")
        proposal_rows.append(row)
        identity_by_index[proposal.proposal_index] = {
            name: row[name] for name in _RAW_IDENTITY_COLUMNS
        }

    raw_rows: list[dict[str, object]] = []
    for raw in raw_features:
        _require(raw.proposal_index in identity_by_index, "raw feature references an unknown proposal")
        identity = identity_by_index[raw.proposal_index]
        scientific = canonical_raw_feature_csv_row(raw)
        row = {**identity, **scientific}
        _require(tuple(row) == CANONICAL_RAW_FEATURE_TABLE_COLUMNS, "canonical raw feature table schema changed")
        raw_rows.append(row)
    return _GeneratedTile(
        proposal_rows=tuple(proposal_rows),
        raw_rows=tuple(raw_rows),
        raw_features=tuple(raw_features),
        selection=selection,
    )


def _generate_full_image_records(
    runtime: CanonicalStageRuntime,
    image_bgr: Any,
    unit_row: Mapping[str, object],
    image_record: Mapping[str, object],
    *,
    attest_cuda: bool = True,
) -> _GeneratedTile:
    """Generate compact proposals/features for one full warped frame."""

    import cv2

    _require(
        runtime.profile == FULL_IMAGE_GPU_PROFILE,
        "full-image record generation requires its profile-bound runtime",
    )
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if attest_cuda:
        _attest_canonical_stage_runtime_cuda(runtime)
    selection = runtime.amg.generate(rgb)
    if attest_cuda:
        _attest_canonical_stage_runtime_cuda(runtime)
    _require(
        isinstance(selection, FullImageProposalSelection),
        "full-image backend returned an invalid proposal selection",
    )
    grid = CanonicalGridContext(
        row_lines=tuple(int(value) for value in image_record["row_lines"]),
        column_lines=tuple(int(value) for value in image_record["column_lines"]),
        tile_x=0,
        tile_y=0,
    )
    raw_features = extract_full_image_raw_features(
        image_bgr,
        selection.proposals,
        runtime.embedder,
        grid=grid,
    )
    _require(
        len(raw_features)
        == len(selection.proposals) * len(CANONICAL_FEATURE_CROP_SCALES),
        "full-image raw feature row count changed",
    )

    proposal_rows: list[dict[str, object]] = []
    identity_by_index: dict[int, dict[str, object]] = {}
    for proposal in selection.proposals:
        proposal_id = _proposal_id(
            str(unit_row["image_id"]),
            str(unit_row["tile_name"]),
            str(unit_row["tile_sha256"]),
            0,
            0,
            proposal,
            profile=runtime.profile,
        )
        canvas_height, canvas_width, encoded_mask = _encode_full_image_mask(
            proposal.packed_mask
        )
        polygon, (bbox_x, bbox_y, bbox_w, bbox_h) = _mask_geometry(
            proposal,
            unit_row,
            image_record,
        )
        crop_x, crop_y, crop_w, crop_h = proposal.bbox
        row: dict[str, object] = {
            "proposal_id": proposal_id,
            "proposal_sha256": proposal_id,
            "image_id": str(unit_row["image_id"]),
            "image_sha256": str(unit_row["image_sha256"]),
            "image_name": str(unit_row["image_name"]),
            "group_id": str(unit_row["group_id"]),
            # Stable aliases consumed by the review/active-learning exchange.
            "tile_name": str(unit_row["tile_name"]),
            "tile_sha256": str(unit_row["tile_sha256"]),
            "tile_x": "0",
            "tile_y": "0",
            "source_index": str(proposal.source_index),
            "proposal_index": str(proposal.proposal_index),
            "mask_sha256": proposal.mask_sha256,
            "mask_area": str(proposal.area),
            "mask_encoding": FULL_IMAGE_MASK_ENCODING,
            "mask_height": str(canvas_height),
            "mask_width": str(canvas_width),
            "mask_packbits_base64": encoded_mask,
            "tile_bbox_x": str(crop_x),
            "tile_bbox_y": str(crop_y),
            "tile_bbox_w": str(crop_w),
            "tile_bbox_h": str(crop_h),
            "bbox_x": str(bbox_x),
            "bbox_y": str(bbox_y),
            "bbox_w": str(bbox_w),
            "bbox_h": str(bbox_h),
            "poly": json.dumps(polygon, ensure_ascii=True, separators=(",", ":")),
            "predicted_iou": _format_float(proposal.predicted_iou),
            "stability_score": _format_float(proposal.stability_score),
        }
        _require(
            tuple(row) == CANONICAL_PROPOSAL_COLUMNS,
            "full-image proposal schema changed",
        )
        proposal_rows.append(row)
        identity_by_index[proposal.proposal_index] = {
            name: row[name] for name in _RAW_IDENTITY_COLUMNS
        }

    raw_rows: list[dict[str, object]] = []
    for raw in raw_features:
        _require(
            raw.proposal_index in identity_by_index,
            "full-image raw feature references an unknown proposal",
        )
        row = {
            **identity_by_index[raw.proposal_index],
            **canonical_raw_feature_csv_row(raw),
        }
        _require(
            tuple(row) == CANONICAL_RAW_FEATURE_TABLE_COLUMNS,
            "full-image raw feature table schema changed",
        )
        raw_rows.append(row)
    return _GeneratedTile(
        proposal_rows=tuple(proposal_rows),
        raw_rows=tuple(raw_rows),
        raw_features=tuple(raw_features),
        selection=selection,
    )


def _overlay_proposal_labels(
    proposals: Sequence[CanonicalProposal | FullImageProposal],
    proposal_rows: Sequence[Mapping[str, object]],
) -> tuple[int, list[dict[str, object]]]:
    _require(
        len(proposals) == len(proposal_rows),
        "canonical overlay proposal mapping changed",
    )
    proposal_ids: list[str] = []
    for proposal, row in zip(proposals, proposal_rows, strict=True):
        proposal_id = str(row.get("proposal_id", ""))
        _require(
            _SHA256.fullmatch(proposal_id) is not None
            and row.get("proposal_sha256") == proposal_id
            and row.get("mask_sha256") == proposal.mask_sha256
            and _strict_int(row.get("proposal_index"), "overlay proposal index")
            == proposal.proposal_index,
            "canonical overlay proposal identity changed",
        )
        proposal_ids.append(proposal_id)
    _require(
        len(proposal_ids) == len(set(proposal_ids)),
        "canonical overlay contains duplicate proposal identities",
    )

    prefix_length = 8
    while (
        prefix_length < 64
        and len({proposal_id[:prefix_length] for proposal_id in proposal_ids})
        != len(proposal_ids)
    ):
        prefix_length += 1
    _require(
        len({proposal_id[:prefix_length] for proposal_id in proposal_ids})
        == len(proposal_ids),
        "canonical overlay proposal labels are ambiguous",
    )
    return prefix_length, [
        {
            "label": proposal_id[:prefix_length],
            "proposal_id": proposal_id,
            "proposal_index": proposal.proposal_index,
            "mask_sha256": proposal.mask_sha256,
        }
        for proposal, proposal_id in zip(proposals, proposal_ids, strict=True)
    ]


def _write_overlay(
    path: Path,
    tile_bgr: Any,
    proposals: Sequence[CanonicalProposal | FullImageProposal],
    proposal_rows: Sequence[Mapping[str, object]],
) -> tuple[int, list[dict[str, object]]]:
    import cv2
    import numpy as np

    canvas = np.asarray(tile_bgr, dtype=np.uint8).copy()
    _require(
        canvas.ndim == 3 and canvas.shape[2] == 3 and canvas.size > 0,
        "canonical overlay tile is invalid",
    )
    prefix_length, labels = _overlay_proposal_labels(proposals, proposal_rows)
    canvas_height, canvas_width = canvas.shape[:2]
    for proposal, label_row in zip(proposals, labels, strict=True):
        if isinstance(proposal, FullImageProposal):
            contour_mask = proposal.mask_crop
            contour_offset = (proposal.bbox[0], proposal.bbox[1])
        else:
            contour_mask = proposal.mask
            contour_offset = (0, 0)
        contours, _hierarchy = cv2.findContours(
            (np.asarray(contour_mask) > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        color = (
            64 + int(proposal.mask_sha256[0:2], 16) % 160,
            64 + int(proposal.mask_sha256[2:4], 16) % 160,
            64 + int(proposal.mask_sha256[4:6], 16) % 160,
        )
        cv2.drawContours(
            canvas,
            contours,
            -1,
            color,
            1,
            lineType=cv2.LINE_8,
            offset=contour_offset,
        )
        label = str(label_row["label"])
        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            1,
        )
        bbox_x, bbox_y, _bbox_width, _bbox_height = proposal.bbox
        label_x = min(max(0, bbox_x), max(0, canvas_width - label_width - 1))
        label_y = min(
            max(label_height + baseline + 1, bbox_y),
            max(label_height + baseline + 1, canvas_height - baseline - 1),
        )
        cv2.putText(
            canvas,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            3,
            lineType=cv2.LINE_8,
        )
        cv2.putText(
            canvas,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            lineType=cv2.LINE_8,
        )
    ok, encoded = cv2.imencode(".png", canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    _require(bool(ok), "canonical overlay encoding failed")
    write_new_bytes(path, bytes(encoded))
    return prefix_length, labels


def _stage20_proposal_config_record(
    config: PublicProjectConfig,
    *,
    proposals_sha256: str,
    raw_features_sha256: str,
) -> dict[str, object]:
    """Return the exact method/data binding persisted beside Stage-20 tables."""

    _canonical_config(config)
    _require(
        _SHA256.fullmatch(proposals_sha256) is not None
        and _SHA256.fullmatch(raw_features_sha256) is not None,
        "Stage-20 table identity is invalid",
    )
    settings = amg_settings_for_profile(config.profile)
    record: dict[str, object] = {
        "schema": "compag-curation-canonical-proposal-config/v1",
        "status": "PASS",
        **settings.provenance_record(profile=config.profile),
        "sam2_execution_points_per_batch": _stage20_execution_batch(
            config.profile
        ),
        "sam2_checkpoint_sha256": config.checkpoint_sha256,
        "sam2_config_sha256": config.sam2_config_sha256,
        "sam2_config_locator": config.sam2_config_locator,
        "resnet50_weights_sha256": config.embedding_weights_sha256,
        "embedding_backbone": "resnet50-imagenet1k-v2",
        "embedding_dimensions": CANONICAL_EMBEDDING_DIMENSIONS,
        "masked_crop_padding": 0.10,
        "pca_fit": "DEFERRED_UNTIL_REVIEWED_TRAIN_PARTITION",
        "prototype_fit": "DEFERRED_UNTIL_REVIEWED_TRAIN_POSITIVES",
        "raw_feature_columns": list(CANONICAL_RAW_FEATURE_TABLE_COLUMNS),
        "final_feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
    }
    if config.profile == FULL_IMAGE_GPU_PROFILE:
        record.update(
            {
                "external_tiling": False,
                "analysis_unit_kind": "full_warped_frame",
                "sam2_output_mode": "uncompressed_rle",
                "durable_mask_encoding": FULL_IMAGE_DURABLE_MASK_ENCODING,
                "csv_mask_encoding": FULL_IMAGE_MASK_ENCODING,
                "connected_component_policy": FULL_IMAGE_COMPONENT_POLICY,
                "feature_semantics_id": FULL_IMAGE_FEATURE_SEMANTICS_ID,
                "feature_geometry": "full-analysis-frame",
                "feature_texture_context": "bounded-roi-context",
                "proposals_sha256": proposals_sha256,
                "raw_features_sha256": raw_features_sha256,
            }
        )
    return record


def _generate_canonical_stage_with_dependencies(
    config: PublicProjectConfig,
    prepared: Path,
    output: Path,
    *,
    runtime_factory: CanonicalRuntimeFactory | None = None,
    write_overlays: bool = False,
) -> dict[str, object]:
    """Dependency-controlled stage-20 harness used by production and tests."""

    _canonical_config(config)
    output = _safe_output_directory(output)
    prepared = prepared.resolve(strict=True)
    _require(prepared.is_dir() and not prepared.is_symlink(), "canonical prepared stage is unsafe")
    full_image = config.profile == FULL_IMAGE_GPU_PROFILE
    tile_rows, processing_root = _prepared_processing_rows(
        prepared,
        profile=config.profile,
    )
    image_records = _prepared_image_records(prepared, profile=config.profile)
    _require(
        {row["image_id"] for row in tile_rows} == set(image_records),
        "canonical tile/image record closure changed",
    )
    assets = _runtime_assets_from_config(config)
    runtime = _load_stage20_runtime(assets, config.device, runtime_factory)
    if runtime_factory is None:
        _attest_canonical_stage_runtime_cuda(runtime)
    _require(callable(getattr(runtime.amg, "generate", None)), "canonical runtime AMG is invalid")
    _require(callable(getattr(runtime.embedder, "embed_many", None)), "canonical runtime embedder is invalid")
    overlays = output / "overlays"
    if write_overlays:
        _require(not overlays.exists() and not overlays.is_symlink(), "canonical overlay directory already exists")
        overlays.mkdir(mode=0o755)

    proposal_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    overlay_rows: list[dict[str, object]] = []
    proposal_ids: set[str] = set()
    counters = {
        "input": 0,
        "empty": 0,
        "duplicate": 0,
        "capped": 0,
        "bbox_prefilter": 0,
        "mask_comparison": 0,
        "retained_mask_bytes": 0,
    }
    for tile_row in tile_rows:
        tile_name = portable_basename(tile_row["tile_name"], "canonical tile name")
        tile_path = processing_root / tile_name
        payload, _snapshot = stable_file(
            tile_path,
            max_bytes=MAX_IMAGE_FILE_BYTES if full_image else 16 * 1024 * 1024,
        )
        _require(hashlib.sha256(payload).hexdigest() == tile_row["tile_sha256"], f"canonical tile bytes changed: {tile_name}")
        tile_bgr = _decode_image(payload, tile_name)
        expected_shape = (
            (
                _strict_int(tile_row["crop_h"], "processing unit height", minimum=1),
                _strict_int(tile_row["crop_w"], "processing unit width", minimum=1),
                3,
            )
            if full_image
            else (512, 512, 3)
        )
        _require(
            tuple(tile_bgr.shape) == expected_shape,
            f"profile processing-unit dimensions changed: {tile_name}",
        )
        generator = (
            _generate_full_image_records if full_image else _generate_tile_records
        )
        generated = generator(
            runtime,
            tile_bgr,
            tile_row,
            image_records[tile_row["image_id"]],
            attest_cuda=runtime_factory is None,
        )
        for row in generated.proposal_rows:
            proposal_id = str(row["proposal_id"])
            _require(proposal_id not in proposal_ids, "canonical proposal identity collision")
            proposal_ids.add(proposal_id)
            proposal_rows.append(row)
        raw_rows.extend(generated.raw_rows)
        counters["input"] += generated.selection.input_count
        counters["empty"] += generated.selection.empty_count
        counters["duplicate"] += generated.selection.duplicate_count
        counters["capped"] += generated.selection.capped_count
        counters["bbox_prefilter"] += int(
            getattr(generated.selection, "bbox_prefilter_count", 0)
        )
        counters["mask_comparison"] += int(
            getattr(generated.selection, "mask_comparison_count", 0)
        )
        counters["retained_mask_bytes"] += int(
            getattr(generated.selection, "retained_mask_bytes", 0)
        )
        if write_overlays:
            overlay_path = overlays / f"{Path(tile_name).stem}.png"
            prefix_length, labels = _write_overlay(
                overlay_path,
                tile_bgr,
                generated.selection.proposals,
                generated.proposal_rows,
            )
            overlay_rows.append(
                {
                    "tile_name": tile_name,
                    "tile_sha256": str(tile_row["tile_sha256"]),
                    "overlay": f"overlays/{overlay_path.name}",
                    "overlay_sha256": sha256_file(overlay_path),
                    "proposal_count": len(labels),
                    "proposal_id_prefix_length": prefix_length,
                    "proposal_labels": labels,
                    "proposal_labels_sha256": compact_json_sha256(labels),
                }
            )

    _require(proposal_rows, "canonical AMG produced no proposals")
    _require(
        {str(row["image_id"]) for row in proposal_rows} == set(image_records),
        "canonical Stage-20 requires at least one proposal for every genesis image",
    )
    _require(
        len(raw_rows) == len(proposal_rows) * len(CANONICAL_FEATURE_CROP_SCALES),
        "canonical proposal/raw feature closure changed",
    )
    _write_csv_exact(output / "proposals.csv", CANONICAL_PROPOSAL_COLUMNS, proposal_rows)
    _write_csv_exact(output / "features.csv", CANONICAL_RAW_FEATURE_TABLE_COLUMNS, raw_rows)
    overlay_manifest: dict[str, object] | None = None
    if write_overlays:
        ordered_proposal_ids = [str(row["proposal_id"]) for row in proposal_rows]
        mapped_proposal_ids = [
            str(label["proposal_id"])
            for overlay_row in overlay_rows
            for label in overlay_row["proposal_labels"]
        ]
        _require(
            len(overlay_rows) == len(tile_rows)
            and mapped_proposal_ids == ordered_proposal_ids,
            "canonical overlay/proposal closure changed",
        )
        overlay_manifest = {
            "schema": "compag-curation-canonical-overlay-manifest/v1",
            "status": "PASS",
            "label_format": "unique-proposal-id-prefix",
            "tile_count": len(overlay_rows),
            "proposal_count": len(mapped_proposal_ids),
            "proposal_ids_sha256": compact_json_sha256(mapped_proposal_ids),
            "tiles": overlay_rows,
            "tiles_sha256": compact_json_sha256(overlay_rows),
        }
        write_new_json(output / "overlay_manifest.json", overlay_manifest)
    stage20_microbatch = _stage20_execution_batch(config.profile)
    proposal_config = _stage20_proposal_config_record(
        config,
        proposals_sha256=sha256_file(output / "proposals.csv"),
        raw_features_sha256=sha256_file(output / "features.csv"),
    )
    write_new_json(output / "proposal_config.json", proposal_config)
    summary = {
        "schema": "compag-curation-canonical-proposals/v1",
        "status": "PASS",
        "profile": config.profile,
        "sam2_execution_points_per_batch": stage20_microbatch,
        "tile_count": len(tile_rows),
        "proposal_count": len(proposal_rows),
        "raw_feature_row_count": len(raw_rows),
        "raw_rows_per_proposal": len(CANONICAL_FEATURE_CROP_SCALES),
        "amg_input_count": counters["input"],
        "empty_mask_count": counters["empty"],
        "merge_iou_suppressed_count": counters["duplicate"],
        "merge_iou_threshold": 0.75,
        "post_dedup_cap_count": counters["capped"],
        "full_masks_retained_losslessly": True,
        "mask_encoding": MASK_ENCODING,
        "proposals_sha256": sha256_file(output / "proposals.csv"),
        "raw_features_sha256": sha256_file(output / "features.csv"),
        "proposal_config_sha256": sha256_file(output / "proposal_config.json"),
        "overlays": "overlays" if write_overlays else None,
    }
    if full_image:
        summary.update(
            {
                "spatial_mode": "full-image-multiscale",
                "processing_unit_kind": "full_warped_frame",
                "external_tiling": False,
                "tile_count_compatibility_alias": True,
                "mask_encoding": FULL_IMAGE_MASK_ENCODING,
                "bbox_prefilter_count": counters["bbox_prefilter"],
                "exact_mask_comparison_count": counters["mask_comparison"],
                "retained_mask_bytes": counters["retained_mask_bytes"],
                "feature_semantics_id": FULL_IMAGE_FEATURE_SEMANTICS_ID,
                "canonical_tiled_method_equivalence": "NOT_CLAIMED",
            }
        )
    if overlay_manifest is not None:
        summary.update(
            {
                "overlay_manifest": "overlay_manifest.json",
                "overlay_manifest_sha256": sha256_file(output / "overlay_manifest.json"),
                "overlay_tile_count": overlay_manifest["tile_count"],
                "overlay_proposal_count": overlay_manifest["proposal_count"],
                "overlay_proposal_ids_sha256": overlay_manifest[
                    "proposal_ids_sha256"
                ],
            }
        )
    write_new_json(output / "proposal_summary.json", summary)
    return summary


def generate_canonical_stage(
    config: PublicProjectConfig,
    prepared: Path,
    output: Path,
    *,
    write_overlays: bool = False,
) -> dict[str, object]:
    """Run the profile-bound CUDA SAM2/ResNet50 proposal stage."""

    _canonical_config(config)
    _require_cuda_service_device(config.device, "proposal generation")
    return _generate_canonical_stage_with_dependencies(
        config,
        prepared,
        output,
        write_overlays=write_overlays,
    )


def _canonical_review_rows(path: Path, *, payload: bytes | None = None) -> list[dict[str, str]]:
    rows = (
        _read_csv_exact(
            path,
            CANONICAL_REVIEW_COLUMNS,
            max_bytes=MAX_REVIEW_BYTES,
            max_rows=MAX_REVIEW_ROWS,
        )
        if payload is None
        else _parse_csv_exact(
            payload,
            path.name,
            CANONICAL_REVIEW_COLUMNS,
            max_rows=MAX_REVIEW_ROWS,
        )
    )
    seen: set[str] = set()
    actions: list[str] = []
    weights: list[str] = []
    for row in rows:
        proposal_id = row["proposal_id"]
        _require(_SHA256.fullmatch(proposal_id) is not None, "review proposal identity is invalid")
        _require(row["proposal_sha256"] == proposal_id, "review proposal hash differs from its identity")
        _require(proposal_id not in seen, "review table contains duplicate proposal identities")
        seen.add(proposal_id)
        _require(_SHA256.fullmatch(row["image_id"]) is not None, "review image identity is invalid")
        _require(_SHA256.fullmatch(row["image_sha256"]) is not None, "review image hash is invalid")
        _require(re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", row["group_id"]) is not None, "review group identity is invalid")
        _require(row["label"] in {"0", "1"}, "canonical review label must be 0 or 1")
        _require(row["review_status"] == "reviewed", "canonical review row is not reviewed")
        action = row["review_action"]
        _require(action in REVIEW_ACTION_WEIGHTS, "canonical review action is invalid")
        expected = format(REVIEW_ACTION_WEIGHTS[action], ".1f")
        _require(row["review_weight"] == expected, "canonical review action/weight mismatch")
        actions.append(action)
        weights.append(row["review_weight"])
    validate_canonical_review_weights(actions, weights)
    return rows


def _review_value(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        return row[name]
    return getattr(row, name)


def canonical_review_split_feasibility(review_rows: Sequence[Any]) -> dict[str, object]:
    """Validate effective canonical reviews and derive the fixed 80:20 split."""

    import numpy as np
    from compag_curation.training.xgb import build_cv_pairs
    from .training import _group_safe_inner_split

    _require(review_rows, "canonical review table is empty")
    groups: list[str] = []
    labels: list[int] = []
    actions: list[str] = []
    weights: list[float] = []
    for row in review_rows:
        group = str(_review_value(row, "group_id"))
        label = int(_review_value(row, "label"))
        action = str(_review_value(row, "review_action"))
        weight = float(_review_value(row, "review_weight"))
        _require(label in {0, 1}, "canonical review label is not binary")
        groups.append(group)
        labels.append(label)
        actions.append(action)
        weights.append(weight)
    validated_weights = validate_canonical_review_weights(actions, weights)
    keep = validated_weights > 0.0
    effective_groups = np.asarray(groups, dtype=str)[keep]
    effective_labels = np.asarray(labels, dtype=int)[keep]
    _require(len(effective_groups) >= 2, "canonical review has fewer than two effective rows")
    split = canonical_tile_balanced_split(effective_groups)
    train_mask = np.isin(effective_groups, tuple(split.train_groups))
    test_mask = np.isin(effective_groups, tuple(split.test_groups))
    train_groups = effective_groups[train_mask]
    train_labels = effective_labels[train_mask]
    test_labels = effective_labels[test_mask]
    _require(len(split.train_groups) >= CANONICAL_GROUP_FOLDS, "canonical training requires at least five train groups")
    _require(set(train_labels.tolist()) == {0, 1}, "canonical train partition lacks both labels")
    _require(set(test_labels.tolist()) == {0, 1}, "canonical untouched test partition lacks both labels")
    cv_pairs = build_cv_pairs(train_groups, train_labels, CANONICAL_GROUP_FOLDS)
    _require(len(cv_pairs) == CANONICAL_GROUP_FOLDS, "canonical GroupKFold cannot form five class-valid folds")
    inner_train, inner_validation = _group_safe_inner_split(
        train_labels,
        train_groups,
        CanonicalTrainingConfig(),
    )
    _require(
        not set(train_groups[inner_train]) & set(train_groups[inner_validation]),
        "canonical early-stop validation leaks groups",
    )

    def counts(mask: Any) -> dict[str, int]:
        selected = effective_labels[mask]
        return {
            "rows": int(len(selected)),
            "negative": int((selected == 0).sum()),
            "positive": int((selected == 1).sum()),
        }

    return {
        "schema": "compag-curation-canonical-review-split-feasibility/v1",
        "status": "PASS",
        "seed": CANONICAL_RANDOM_STATE,
        "method": "SOURCE_TILE_BALANCED_GROUP_PURE_NEAREST_80_20",
        "review_rows": len(review_rows),
        "effective_rows": int(keep.sum()),
        "skipped_rows": int((~keep).sum()),
        "train_groups": sorted(split.train_groups),
        "test_groups": sorted(split.test_groups),
        "target_test_rows": split.target_test_rows,
        "observed_test_rows": split.observed_test_rows,
        "group_cv_folds": len(cv_pairs),
        "inner_validation_fraction": 0.30,
        "coverage": {
            "train": counts(train_mask),
            "test": counts(test_mask),
        },
    }


def build_canonical_group_split(
    reviewed: Path,
    output: Path,
    *,
    profile: str = CANONICAL_PROFILE,
) -> dict[str, object]:
    """Write the canonical effective-review, group-pure 80:20 manifest."""

    _canonical_device(profile)
    output = _safe_output_directory(output)
    reviewed_payload, _reviewed_snapshot = stable_file(reviewed, max_bytes=MAX_REVIEW_BYTES)
    rows = _canonical_review_rows(reviewed, payload=reviewed_payload)
    feasibility = canonical_review_split_feasibility(rows)
    manifest = {
        "schema": "compag-curation-canonical-group-split/v1",
        "status": "PASS",
        "profile": profile,
        "seed": CANONICAL_RANDOM_STATE,
        "method": feasibility["method"],
        "train_groups": feasibility["train_groups"],
        "test_groups": feasibility["test_groups"],
        "target_test_rows": feasibility["target_test_rows"],
        "observed_test_rows": feasibility["observed_test_rows"],
        "group_cv_folds": feasibility["group_cv_folds"],
        "coverage": feasibility["coverage"],
        "effective_rows": feasibility["effective_rows"],
        "skipped_rows": feasibility["skipped_rows"],
        "reviewed_sha256": hashlib.sha256(reviewed_payload).hexdigest(),
    }
    write_new_json(output / "split_manifest.json", manifest)
    return manifest


def _canonical_proposal_rows(
    path: Path,
    *,
    profile: str = CANONICAL_PROFILE,
) -> list[dict[str, str]]:
    import numpy as np

    rows = _read_csv_exact(path, CANONICAL_PROPOSAL_COLUMNS)
    seen: set[str] = set()
    for row in rows:
        proposal_id = row["proposal_id"]
        _require(
            _SHA256.fullmatch(proposal_id) is not None
            and row["proposal_sha256"] == proposal_id
            and proposal_id not in seen,
            "canonical proposal identity is invalid or duplicated",
        )
        seen.add(proposal_id)
        for name in ("image_id", "image_sha256", "tile_sha256", "mask_sha256"):
            _require(_SHA256.fullmatch(row[name]) is not None, f"canonical proposal {name} is invalid")
        full_image = profile == FULL_IMAGE_GPU_PROFILE
        _require(
            row["mask_encoding"]
            == (FULL_IMAGE_MASK_ENCODING if full_image else MASK_ENCODING),
            "profile proposal mask encoding changed",
        )
        if full_image:
            mask = _decode_full_image_packed_mask(row)
            _require(
                mask.area == _strict_int(row["mask_area"], "mask area", minimum=1),
                "full-image proposal mask area mismatch",
            )
        else:
            mask = decode_lossless_mask(
                row["mask_height"],
                row["mask_width"],
                row["mask_packbits_base64"],
            )
            _require(canonical_mask_sha256(mask) == row["mask_sha256"], "canonical proposal mask hash mismatch")
            _require(int(np.asarray(mask).sum()) == _strict_int(row["mask_area"], "mask area", minimum=1), "canonical proposal mask area mismatch")
        for name in (
            "tile_x",
            "tile_y",
            "source_index",
            "proposal_index",
            "tile_bbox_x",
            "tile_bbox_y",
            "tile_bbox_w",
            "tile_bbox_h",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
        ):
            _strict_int(row[name], f"canonical proposal {name}")
        try:
            polygon = json.loads(row["poly"])
        except (json.JSONDecodeError, RecursionError) as exc:
            raise PublicIOError("canonical proposal polygon is invalid JSON") from exc
        _require(
            isinstance(polygon, list)
            and len(polygon) % 2 == 0
            and all(isinstance(value, int) and not isinstance(value, bool) for value in polygon),
            "canonical proposal polygon is invalid",
        )
        for name in ("predicted_iou", "stability_score"):
            value = _strict_float(row[name], f"canonical proposal {name}")
            _require(0.0 <= value <= 1.0, f"canonical proposal {name} is outside [0,1]")
    return rows


def _canonical_raw_records(
    path: Path,
    proposals: Mapping[str, Mapping[str, str]],
) -> tuple[list[CanonicalRawFeature], list[dict[str, str]]]:
    rows = _read_csv_exact(path, CANONICAL_RAW_FEATURE_TABLE_COLUMNS)
    raw_features: list[CanonicalRawFeature] = []
    observed: set[tuple[str, float]] = set()
    counts: dict[str, int] = {proposal_id: 0 for proposal_id in proposals}
    for row in rows:
        proposal_id = row["proposal_id"]
        _require(proposal_id in proposals, "canonical raw feature references an unknown proposal")
        proposal = proposals[proposal_id]
        for name in _RAW_IDENTITY_COLUMNS:
            _require(row[name] == proposal[name], f"canonical raw feature identity mismatch: {name}")
        scale = _strict_float(row["scale"], "canonical raw feature scale")
        _require(scale in CANONICAL_FEATURE_CROP_SCALES, "canonical raw feature scale changed")
        key = (proposal_id, scale)
        _require(key not in observed, "canonical raw feature row is duplicated")
        observed.add(key)
        counts[proposal_id] += 1
        proposal_index = _strict_int(row["proposal_index"], "canonical raw proposal index", minimum=1)
        _require(proposal_index == int(proposal["proposal_index"]), "canonical raw proposal index mismatch")
        predicted_iou = _strict_float(row["predicted_iou"], "canonical raw predicted IoU")
        stability_score = _strict_float(row["stability_score"], "canonical raw stability score")
        _require(
            math.isclose(predicted_iou, float(proposal["predicted_iou"]), rel_tol=0.0, abs_tol=1e-15)
            and math.isclose(stability_score, float(proposal["stability_score"]), rel_tol=0.0, abs_tol=1e-15),
            "canonical raw SAM2 scores differ from proposal evidence",
        )
        _require(row["embedding_encoding"] == "base64-float32-little-endian-v1", "canonical embedding encoding changed")
        _require(int(row["embedding_dimensions"]) == CANONICAL_EMBEDDING_DIMENSIONS, "canonical embedding width changed")
        values = {
            name: _strict_float(row[name], f"canonical raw feature {name}")
            for name in CANONICAL_RAW_FEATURE_ORDER
        }
        raw_features.append(
            CanonicalRawFeature(
                proposal_index=proposal_index,
                scale=scale,
                predicted_iou=predicted_iou,
                stability_score=stability_score,
                values=values,
                embedding=decode_canonical_embedding(row["embedding_f32le_base64"]),
            )
        )
    expected_count = len(CANONICAL_FEATURE_CROP_SCALES)
    _require(
        all(count == expected_count for count in counts.values())
        and len(rows) == len(proposals) * expected_count,
        "canonical raw feature/proposal closure changed",
    )
    return raw_features, rows


def _canonical_split(
    path: Path,
    reviewed_sha256: str,
    review_rows: Sequence[Mapping[str, str]],
    *,
    profile: str = CANONICAL_PROFILE,
) -> CanonicalGroupSplit:
    _canonical_device(profile)
    value = _read_canonical_json(path, "canonical split manifest")
    required = {
        "schema",
        "status",
        "profile",
        "seed",
        "method",
        "train_groups",
        "test_groups",
        "target_test_rows",
        "observed_test_rows",
        "group_cv_folds",
        "coverage",
        "effective_rows",
        "skipped_rows",
        "reviewed_sha256",
    }
    _require(set(value) == required, "canonical split manifest fields changed")
    feasibility = canonical_review_split_feasibility(review_rows)
    numeric_fields = (
        "seed",
        "target_test_rows",
        "observed_test_rows",
        "group_cv_folds",
        "effective_rows",
        "skipped_rows",
    )
    _require(
        value["schema"] == "compag-curation-canonical-group-split/v1"
        and value["status"] == "PASS"
        and value["profile"] == profile
        and all(type(value[name]) is int and int(value[name]) >= 0 for name in numeric_fields)
        and value["seed"] == CANONICAL_RANDOM_STATE
        and value["method"] == "SOURCE_TILE_BALANCED_GROUP_PURE_NEAREST_80_20"
        and value["method"] == feasibility["method"]
        and value["group_cv_folds"] == CANONICAL_GROUP_FOLDS
        and value["reviewed_sha256"] == reviewed_sha256,
        "canonical split manifest identity changed",
    )
    train = value["train_groups"]
    test = value["test_groups"]
    _require(
        isinstance(train, list)
        and isinstance(test, list)
        and all(
            isinstance(group, str)
            and re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", group) is not None
            for group in (*train, *test)
        )
        and train == sorted(train)
        and test == sorted(test)
        and len(set(train)) == len(train)
        and len(set(test)) == len(test)
        and not set(train) & set(test),
        "canonical split group partitions are invalid",
    )
    coverage = value["coverage"]
    _require(
        isinstance(coverage, dict)
        and set(coverage) == {"train", "test"}
        and all(
            isinstance(coverage[name], dict)
            and set(coverage[name]) == {"rows", "negative", "positive"}
            and all(type(coverage[name][field]) is int and coverage[name][field] >= 0 for field in coverage[name])
            for name in ("train", "test")
        ),
        "canonical split coverage schema is invalid",
    )
    _require(
        train == feasibility["train_groups"]
        and test == feasibility["test_groups"]
        and value["target_test_rows"] == feasibility["target_test_rows"]
        and value["observed_test_rows"] == feasibility["observed_test_rows"]
        and value["effective_rows"] == feasibility["effective_rows"]
        and value["skipped_rows"] == feasibility["skipped_rows"]
        and value["coverage"] == feasibility["coverage"],
        "canonical split manifest differs from the reviewed table",
    )
    return CanonicalGroupSplit(
        train_groups=frozenset(train),
        test_groups=frozenset(test),
        target_test_rows=int(value["target_test_rows"]),
        observed_test_rows=int(value["observed_test_rows"]),
    )


def _pca_explained_variance(
    raw_features: Sequence[CanonicalRawFeature],
    training_indices: Sequence[int],
    state: CanonicalFeatureState,
) -> tuple[float, ...]:
    import numpy as np

    embeddings = np.vstack(
        [np.asarray(raw_features[index].embedding, dtype=np.float32) for index in training_indices]
    )
    _require(len(embeddings) >= CANONICAL_PCA_DIMENSIONS, "canonical PCA training row count changed")
    centered = embeddings - np.asarray(state.pca_mean, dtype=np.float32)
    projections = centered @ np.asarray(state.pca_components, dtype=np.float32).T
    values = np.var(projections, axis=0, ddof=1, dtype=np.float64)
    _require(
        values.shape == (CANONICAL_PCA_DIMENSIONS,)
        and bool(np.all(np.isfinite(values)))
        and bool(np.all(values >= 0.0)),
        "canonical PCA explained variance is invalid",
    )
    return tuple(float(value) for value in values)


def _bundle_config_maps(config: PublicProjectConfig) -> tuple[dict[str, object], ...]:
    _canonical_config(config)
    if config.profile == FULL_IMAGE_GPU_PROFILE:
        from compag_curation.model_bundle import (
            BUNDLE_V2_ASSET_PROFILES,
            _canonical_config_contracts,
        )

        contracts = _canonical_config_contracts(
            profile=config.profile,
            asset_profile=BUNDLE_V2_ASSET_PROFILES[config.profile],
            feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
        )
        return tuple(
            dict(contracts[name])
            for name in (
                "normalized",
                "preprocessing",
                "proposal",
                "features",
                "training",
            )
        )
    settings = amg_settings_for_profile(config.profile)
    normalized = {
        "profile": config.profile,
        "device": config.device,
        "seed": CANONICAL_RANDOM_STATE,
        "tile_size": 512,
        "tile_stride": 512,
        "sam2_checkpoint_sha256": config.checkpoint_sha256,
        "sam2_config_sha256": config.sam2_config_sha256,
        "sam2_config_locator": config.sam2_config_locator,
        "embedding_weights_sha256": config.embedding_weights_sha256,
        "embedding_dimensions": CANONICAL_EMBEDDING_DIMENSIONS,
        "pca_components": CANONICAL_PCA_DIMENSIONS,
        "proposal_scales": [1.0],
        "yolo_enabled": False,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        "inference_threshold": CANONICAL_DECISION_THRESHOLD,
        "nms_iou_threshold": CANONICAL_FULL_IMAGE_NMS_IOU,
        "threshold_method": "FIXED_CANONICAL_METHOD",
    }
    preprocessing = {
        "profile": config.profile,
        "tile_size": 512,
        "tile_stride": 512,
        "far_edge_alignment": True,
        "padding": "bottom-right-edge-value",
        "tile_format": "jpg",
        "jpeg_quality": 95,
        "yellow_card_hsv_lower": [15, 40, 60],
        "yellow_card_hsv_upper": [40, 255, 255],
        "yellow_card_quad_epsilon_fraction": 0.02,
        "yellow_card_fallback": "identity",
    }
    proposal = {
        **settings.provenance_record(profile=config.profile),
        "sam2_checkpoint_sha256": config.checkpoint_sha256,
        "sam2_config_sha256": config.sam2_config_sha256,
        "sam2_config_locator": config.sam2_config_locator,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
    }
    features = {
        "profile": config.profile,
        "mode": "ultra",
        "feature_crop_scales": list(CANONICAL_FEATURE_CROP_SCALES),
        "masked_crop_padding": 0.10,
        "embedding_backbone": "resnet50-imagenet1k-v2",
        "embedding_weights_sha256": config.embedding_weights_sha256,
        "embedding_dimensions": CANONICAL_EMBEDDING_DIMENSIONS,
        "prototype": "train_positive_mean_l2_normalized",
        "pca_components": CANONICAL_PCA_DIMENSIONS,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        "missing_feature_policy": "reject",
        "zero_fill": False,
    }
    training = {
        "profile": config.profile,
        "group_test_fraction": 0.20,
        "group_cv_folds": 5,
        "search_iterations": 30,
        "inner_validation_fraction": 0.30,
        "validation_scale": 1.0,
        "early_stopping_rounds": 30,
        "review_action_weights": dict(REVIEW_ACTION_WEIGHTS),
        "safe_smote_train_fold_only": True,
        "tabular_augmentation_enabled": False,
        "scale_pos_weight": 1.0,
        "xgboost_host_thread_count": 1,
        "inference_threshold": CANONICAL_DECISION_THRESHOLD,
        "threshold_method": "FIXED_CANONICAL_METHOD",
        "full_image_nms_iou": CANONICAL_FULL_IMAGE_NMS_IOU,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
    }
    return normalized, preprocessing, proposal, features, training


def _canonical_training_config(config: PublicProjectConfig) -> CanonicalTrainingConfig:
    _canonical_config(config)
    return CanonicalTrainingConfig(
        device=config.device,
        thread_count=1,
    )


def train_canonical_stage(
    config: PublicProjectConfig,
    proposals_path: Path,
    reviewed_path: Path,
    split_path: Path,
    output: Path,
    *,
    features_path: Path | None = None,
    bundle_writer: Callable[[Any], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Fit train-only feature state and the canonical classifier, then seal v2."""

    from compag_curation.model_bundle import BundleV2WriteRequest, write_model_bundle_v2

    _canonical_config(config)
    _require_cuda_service_device(config.device, "XGBoost training")
    output = _safe_output_directory(output)
    proposals_path = proposals_path.resolve(strict=True)
    features_path = (
        proposals_path.parent / "features.csv"
        if features_path is None
        else features_path.resolve(strict=True)
    )
    reviewed_path = reviewed_path.resolve(strict=True)
    split_path = split_path.resolve(strict=True)

    stage20_proposal_config_sha256: str | None = None
    if config.profile == FULL_IMAGE_GPU_PROFILE:
        stage20_proposal_config_path = proposals_path.parent / "proposal_config.json"
        _require(
            stage20_proposal_config_path.is_file()
            and not stage20_proposal_config_path.is_symlink(),
            "full-image training requires its sealed Stage-20 proposal config",
        )
        observed_stage20_config = _read_canonical_json(
            stage20_proposal_config_path,
            "full-image Stage-20 proposal config",
            max_bytes=4 * 1024 * 1024,
        )
        expected_stage20_config = _stage20_proposal_config_record(
            config,
            proposals_sha256=sha256_file(proposals_path),
            raw_features_sha256=sha256_file(features_path),
        )
        _require(
            observed_stage20_config == expected_stage20_config,
            "full-image Stage-20 proposal config/data binding changed",
        )
        stage20_proposal_config_sha256 = sha256_file(
            stage20_proposal_config_path
        )

    proposal_rows = _canonical_proposal_rows(
        proposals_path,
        profile=config.profile,
    )
    proposals = {row["proposal_id"]: row for row in proposal_rows}
    raw_features, raw_rows = _canonical_raw_records(features_path, proposals)
    reviewed_payload, _reviewed_snapshot = stable_file(
        reviewed_path,
        max_bytes=MAX_REVIEW_BYTES,
    )
    reviews = _canonical_review_rows(reviewed_path, payload=reviewed_payload)
    review_by_id = {row["proposal_id"]: row for row in reviews}
    _require(set(review_by_id) == set(proposals), "canonical proposal/review identity closure changed")
    reviewed_sha256 = hashlib.sha256(reviewed_payload).hexdigest()
    split = _canonical_split(
        split_path,
        reviewed_sha256,
        reviews,
        profile=config.profile,
    )

    labels: list[int] = []
    groups: list[str] = []
    actions: list[str] = []
    weights: list[float] = []
    for raw_row in raw_rows:
        review = review_by_id[raw_row["proposal_id"]]
        _require(
            all(review[name] == raw_row[name] for name in ("proposal_id", "proposal_sha256", "image_id", "image_sha256", "group_id")),
            "canonical raw/review immutable identity changed",
        )
        labels.append(int(review["label"]))
        groups.append(review["group_id"])
        actions.append(review["review_action"])
        weights.append(float(review["review_weight"]))
    validated_weights = validate_canonical_review_weights(actions, weights)
    training_indices = [
        index
        for index, group in enumerate(groups)
        if group in split.train_groups and float(validated_weights[index]) > 0.0
    ]
    _require(
        len(training_indices) >= CANONICAL_PCA_DIMENSIONS,
        "canonical reviewed train partition has fewer than 32 raw rows",
    )
    state = fit_canonical_feature_state(raw_features, labels, training_indices)
    finalized = [finalize_canonical_feature(raw, state) for raw in raw_features]
    _require(
        all(tuple(row) == CANONICAL_FEATURE_ORDER for row in finalized),
        "canonical finalized feature schema changed",
    )

    final_rows: list[dict[str, object]] = []
    partitions: list[str] = []
    for raw_row, feature_values, label, action, weight, group in zip(
        raw_rows,
        finalized,
        labels,
        actions,
        weights,
        groups,
        strict=True,
    ):
        if weight == 0.0:
            partition = "excluded_skip"
        elif group in split.train_groups:
            partition = "train"
        elif group in split.test_groups:
            partition = "test"
        else:
            raise PublicIOError("effective canonical row is absent from the split")
        partitions.append(partition)
        identity = {name: raw_row[name] for name in _RAW_IDENTITY_COLUMNS}
        row: dict[str, object] = {
            **identity,
            "scale": _format_float(raw_row["scale"]),
            "label": str(label),
            "review_action": action,
            "review_weight": format(weight, ".1f"),
            "partition": partition,
            **{name: _format_float(feature_values[name]) for name in CANONICAL_FEATURE_ORDER},
        }
        _require(tuple(row) == CANONICAL_FINAL_FEATURE_COLUMNS, "canonical finalized CSV schema changed")
        final_rows.append(row)
    _write_csv_exact(
        output / "finalized_features.csv",
        CANONICAL_FINAL_FEATURE_COLUMNS,
        final_rows,
    )

    training_arguments: dict[str, object] = {
        "scales": [float(row["scale"]) for row in raw_rows],
        "split": split,
        "config": _canonical_training_config(config),
    }
    training = train_canonical_xgb(
        finalized,
        labels,
        groups,
        actions,
        weights,
        **training_arguments,
    )
    _require(
        training.feature_order == CANONICAL_FEATURE_ORDER
        and training.fixed_threshold == CANONICAL_DECISION_THRESHOLD,
        "canonical trainer returned a changed schema or threshold",
    )
    effective_test_indices = [
        index
        for index, (partition, weight) in enumerate(zip(partitions, weights, strict=True))
        if partition == "test"
        and weight > 0.0
        and float(raw_rows[index]["scale"]) == 1.0
    ]
    _require(
        len(effective_test_indices)
        == len({raw_rows[index]["proposal_id"] for index in effective_test_indices}),
        "canonical test evaluation must contain one scale-1 row per proposal",
    )
    _require(
        len(effective_test_indices) == len(training.test_probabilities)
        and tuple(labels[index] for index in effective_test_indices) == training.test_labels
        and tuple(groups[index] for index in effective_test_indices) == training.test_groups,
        "canonical trainer test prediction alignment changed",
    )
    test_rows: list[dict[str, object]] = []
    for index, probability in zip(
        effective_test_indices,
        training.test_probabilities,
        strict=True,
    ):
        raw_row = raw_rows[index]
        proposal = proposals[raw_row["proposal_id"]]
        probability_value = _strict_float(probability, "canonical test probability")
        _require(0.0 <= probability_value <= 1.0, "canonical test probability is outside [0,1]")
        row: dict[str, object] = {
            "proposal_id": raw_row["proposal_id"],
            "image_id": raw_row["image_id"],
            "image_name": raw_row["image_name"],
            "group_id": raw_row["group_id"],
            "scale": _format_float(raw_row["scale"]),
            "bbox_x": proposal["bbox_x"],
            "bbox_y": proposal["bbox_y"],
            "bbox_w": proposal["bbox_w"],
            "bbox_h": proposal["bbox_h"],
            "poly": proposal["poly"],
            "label": str(labels[index]),
            "review_action": actions[index],
            "review_weight": format(weights[index], ".1f"),
            "xgb_p": _format_float(probability_value),
            "prediction": str(int(probability_value >= CANONICAL_DECISION_THRESHOLD)),
        }
        _require(tuple(row) == CANONICAL_TEST_PREDICTION_COLUMNS, "canonical test-prediction schema changed")
        test_rows.append(row)
    _write_csv_exact(
        output / "test_predictions.csv",
        CANONICAL_TEST_PREDICTION_COLUMNS,
        test_rows,
    )

    booster = training.classifier.get_booster()
    booster.feature_names = list(CANONICAL_FEATURE_ORDER)
    classifier_ubj = portable_classifier_bytes(training.classifier)
    parity_rows = [finalized[index] for index in effective_test_indices]
    parity = verify_probability_parity(
        training.classifier,
        training.imputer_statistics,
        parity_rows,
        classifier_ubj=classifier_ubj,
        device=config.device,
    )
    _require(parity.get("status") == "PASS", "canonical UBJ probability parity did not pass")
    state_payloads = serialize_canonical_feature_state(state)
    pca_explained_variance = _pca_explained_variance(
        raw_features,
        training_indices,
        state,
    )
    normalized, preprocessing, proposal_config, feature_config, training_config = _bundle_config_maps(config)
    compatibility = {
        "classifier_format": "XGBOOST_UBJ",
        "no_pickle_or_joblib": True,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        "fixed_threshold": CANONICAL_DECISION_THRESHOLD,
    }
    provenance = {
        "profile": config.profile,
        "device": config.device,
        "reviewed_sha256": reviewed_sha256,
        "split_manifest_sha256": sha256_file(split_path),
        "raw_features_sha256": sha256_file(features_path),
        "training_row_count": state.training_row_count,
        "positive_training_row_count": state.positive_row_count,
        "feature_state_fit_group_sha256": compact_json_sha256(
            sorted({groups[index] for index in training_indices})
        ),
        "feature_state_fit_rows_sha256": compact_json_sha256(
            [
                {
                    "proposal_id": raw_rows[index]["proposal_id"],
                    "scale": _format_float(raw_rows[index]["scale"]),
                    "group_id": groups[index],
                    "label": labels[index],
                }
                for index in training_indices
            ]
        ),
        "feature_state_fit_scope": "OUTER_TRAIN_ROWS_BEFORE_CV",
        "validation_feature_scale": 1.0,
        "cv_score_interpretation": "CONDITIONAL_ON_OUTER_TRAIN_PREFIT_PROTOTYPE_AND_PCA",
        "fixed_threshold": CANONICAL_DECISION_THRESHOLD,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        "ubj_parity_status": "PASS",
        "paper_result_reproduction": "NOT_CLAIMED",
    }
    sam2_license = config.asset_root / "SAM2-APACHE-2.0.txt"
    resnet50_license = config.asset_root / "TORCHVISION-BSD-3-CLAUSE.txt"
    _require(config.embedding_weights is not None, "canonical ResNet50 asset path is missing")
    request = BundleV2WriteRequest(
        output=output / "model_bundle",
        profile=config.profile,
        classifier_ubj=classifier_ubj,
        feature_order=CANONICAL_FEATURE_ORDER,
        feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
        imputer_statistics={
            name: float(value)
            for name, value in zip(
                CANONICAL_FEATURE_ORDER,
                training.imputer_statistics,
                strict=True,
            )
        },
        prototype_npy=state_payloads.prototype_npy,
        pca_mean_npy=state_payloads.pca_mean_npy,
        pca_components_npy=state_payloads.pca_components_npy,
        pca_explained_variance=pca_explained_variance,
        normalized_config=normalized,
        preprocessing=preprocessing,
        proposal_config=proposal_config,
        feature_config=feature_config,
        training_config=training_config,
        compatibility=compatibility,
        provenance=provenance,
        sam2_config=config.sam2_config,
        sam2_checkpoint=config.checkpoint,
        sam2_license=sam2_license,
        resnet50_weights=config.embedding_weights,
        resnet50_license=resnet50_license,
    )
    bundle_result = dict((bundle_writer or write_model_bundle_v2)(request))
    persisted_bundle = {
        key: _plain(value)
        for key, value in bundle_result.items()
        if key != "root"
    }
    persisted_bundle["path"] = "model_bundle"
    result = {
        "schema": "compag-curation-canonical-training-result/v1",
        "status": "PASS",
        "profile": config.profile,
        "device": config.device,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        "raw_feature_rows": len(raw_rows),
        "finalized_feature_rows": len(final_rows),
        "feature_state_training_rows": state.training_row_count,
        "feature_state_positive_rows": state.positive_row_count,
        "threshold_selected_on": "FIXED_CANONICAL_METHOD",
        "threshold": CANONICAL_DECISION_THRESHOLD,
        "full_image_nms_iou": CANONICAL_FULL_IMAGE_NMS_IOU,
        "used_trees": int(training.used_trees),
        "search": {
            "sampled_configuration_count": training.search.sampled_configuration_count,
            "fold_count": training.search.fold_count,
            "best_average_precision": training.search.best_average_precision,
            "best_parameters": _plain(training.search.best_parameters),
            "score_ledger": list(training.search.score_ledger),
        },
        "test_metrics": _plain(training.test_metrics),
        "retrospective_test_best_f1_threshold": training.retrospective_test_best_f1_threshold,
        "ubj_parity": _plain(parity),
        "reviewed_sha256": reviewed_sha256,
        "split_manifest_sha256": sha256_file(split_path),
        "proposals_sha256": sha256_file(proposals_path),
        "raw_features_sha256": sha256_file(features_path),
        "finalized_features_sha256": sha256_file(output / "finalized_features.csv"),
        "test_predictions_sha256": sha256_file(output / "test_predictions.csv"),
        "bundle": persisted_bundle,
        "paper_result_reproduction": "NOT_CLAIMED",
    }
    if config.profile == FULL_IMAGE_GPU_PROFILE:
        result["stage20_proposal_config_sha256"] = (
            stage20_proposal_config_sha256
        )
    write_new_json(output / "training_result.json", _plain(result))
    return result


def _inference_image_paths(root: Path) -> list[Path]:
    root = root.resolve(strict=True)
    _require(root.is_dir() and not root.is_symlink(), "canonical inference image directory is unsafe")
    paths = sorted(
        (path for path in root.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: path.name,
    )
    _require(paths, "canonical inference image directory is empty")
    names: set[str] = set()
    stems: set[str] = set()
    for path in paths:
        portable_basename(path.name, "canonical inference image name")
        info = path.lstat()
        _require(
            stat.S_ISREG(info.st_mode)
            and not path.is_symlink()
            and 0 < info.st_size <= MAX_IMAGE_FILE_BYTES,
            f"canonical inference image is unsafe: {path.name}",
        )
        folded_name = path.name.casefold()
        folded_stem = path.stem.casefold()
        _require(folded_name not in names and folded_stem not in stems, "canonical inference image name collision")
        names.add(folded_name)
        stems.add(folded_stem)
    return paths


def _runtime_assets_from_bundle(bundle: Any) -> CanonicalRuntimeAssets:
    _require(bundle.resnet50_weights is not None, "canonical bundle omits ResNet50 weights")
    settings = amg_settings_for_profile(bundle.profile)
    checkpoint_sha256 = (
        EFFICIENT_SAM2_HIERA_T_CHECKPOINT_SHA256
        if bundle.profile == EFFICIENT_GPU_PROFILE
        else CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256
    )
    config_sha256 = (
        EFFICIENT_SAM2_HIERA_T_CONFIG_SHA256
        if bundle.profile == EFFICIENT_GPU_PROFILE
        else CANONICAL_SAM2_HIERA_L_CONFIG_SHA256
    )
    return CanonicalRuntimeAssets(
        sam2_config=bundle.sam2_config,
        sam2_checkpoint=bundle.checkpoint,
        sam2_config_locator=settings.config_locator,
        sam2_config_sha256=config_sha256,
        sam2_checkpoint_sha256=checkpoint_sha256,
        resnet50_weights=bundle.resnet50_weights,
        resnet50_weights_sha256=RESNET50_WEIGHTS_SHA256,
        profile=bundle.profile,
    )


def _feature_state_from_bundle(bundle: Any) -> CanonicalFeatureState:
    import numpy as np

    _require(
        bundle.prototype is not None
        and bundle.pca_mean is not None
        and bundle.pca_components is not None,
        "canonical bundle omits feature-state arrays",
    )
    details = bundle.provenance.get("details", {}) if isinstance(bundle.provenance, Mapping) else {}
    training_count = int(details.get("training_row_count", CANONICAL_PCA_DIMENSIONS))
    positive_count = int(details.get("positive_training_row_count", 1))
    return CanonicalFeatureState(
        prototype=np.asarray(bundle.prototype, dtype=np.float32),
        pca_components=np.asarray(bundle.pca_components, dtype=np.float32),
        pca_mean=np.asarray(bundle.pca_mean, dtype=np.float32),
        training_row_count=training_count,
        positive_row_count=positive_count,
    )


def _default_nms(boxes: Any, scores: Any, threshold: float) -> Any:
    from compag_curation.evaluation.geometry import nms_xyxy

    return nms_xyxy(boxes, scores, threshold)


def _canonical_nms_indices(
    boxes: Any,
    scores: Any,
    proposal_ids: Sequence[str],
    threshold: float,
    *,
    nms_function: Callable[[Any, Any, float], Any] | None = None,
) -> Any:
    """Run NMS with an explicit full-proposal-ID score-tie order."""

    box_rows = [tuple(float(value) for value in row) for row in boxes]
    score_rows = [float(value) for value in scores]
    _require(
        all(len(row) == 4 and all(math.isfinite(value) for value in row) for row in box_rows)
        and len(box_rows) == len(score_rows)
        and len(proposal_ids) == len(score_rows)
        and len(set(proposal_ids)) == len(proposal_ids)
        and all(_SHA256.fullmatch(proposal_id) is not None for proposal_id in proposal_ids)
        and all(math.isfinite(value) for value in score_rows),
        "canonical NMS inputs are invalid",
    )
    order = sorted(
        range(len(score_rows)),
        key=lambda index: (-score_rows[index], proposal_ids[index]),
    )
    ordered_boxes = [box_rows[index] for index in order]
    # NMS uses scores only to establish greedy order. Unique rank scores make
    # the probability/proposal-ID ordering independent of backend sort stability.
    rank_scores = list(range(len(order), 0, -1))
    raw_kept = list((nms_function or _default_nms)(ordered_boxes, rank_scores, threshold))
    kept_ordered = [int(value) for value in raw_kept]
    _require(
        all(not isinstance(value, bool) and int(value) == value for value in raw_kept)
        and all(0 <= value < len(order) for value in kept_ordered)
        and len(set(kept_ordered)) == len(kept_ordered),
        "canonical NMS returned invalid indices",
    )
    return [order[index] for index in kept_ordered]


def _infer_canonical_bundle_with_dependencies(
    images: Path,
    bundle_root: Path,
    output: Path,
    *,
    device: str = "cuda",
    runtime_factory: CanonicalRuntimeFactory | None = None,
    bundle_verifier: Callable[[Path], Any] | None = None,
    predictor_loader: Callable[[bytes, Sequence[Any]], Any] | None = None,
    probability_predictor: Callable[[Any, Any], Any] | None = None,
    nms_function: Callable[[Any, Any, float], Any] | None = None,
) -> dict[str, object]:
    """Internal dependency-controlled inference harness.

    ``output`` is an existing owned stage directory.  Other evidence members
    may already be present; ``predictions.csv``, ``raw_features.csv``, and
    ``inference_result.json`` are created here with no-clobber writes.
    """

    normalized_device = str(device).strip().lower()
    _require(
        normalized_device in {"cpu", "cuda"},
        "canonical inference device is unsupported",
    )
    if normalized_device == "cpu":
        _require(
            runtime_factory is not None
            and bundle_verifier is not None
            and predictor_loader is not None
            and probability_predictor is not None,
            "canonical CPU inference is restricted to the fully dependency-injected private test harness",
        )

    import numpy as np
    from compag_curation.model_bundle import BUNDLE_SCHEMA_V2, verify_model_bundle

    device = normalized_device
    output = _safe_output_directory(output)
    image_root = images.absolute()
    model_root = bundle_root.absolute()
    try:
        resolved_images = image_root.resolve(strict=True)
        resolved_bundle = model_root.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError("canonical inference input root cannot be resolved") from exc
    _require(
        image_root == resolved_images and model_root == resolved_bundle,
        "canonical inference input root contains a symlinked component",
    )
    for protected, role in (
        (resolved_images, "image input"),
        (resolved_bundle, "model bundle"),
    ):
        _require(
            protected != output
            and protected not in output.parents
            and output not in protected.parents,
            f"canonical inference output overlaps the {role}",
        )
    _require(
        all(
            not (output / name).exists() and not (output / name).is_symlink()
            for name in (
                "predictions.csv",
                "raw_features.csv",
                "inference_result.json",
            )
        ),
        "canonical inference output already contains result artifacts",
    )
    verify = bundle_verifier or verify_model_bundle
    bundle = verify(resolved_bundle)
    bundle_device = _canonical_device(bundle.profile)
    full_image = bundle.profile == FULL_IMAGE_GPU_PROFILE
    _require(
        bundle.schema == BUNDLE_SCHEMA_V2
        and bundle.profile in V2_PIPELINE_PROFILES
        and device == bundle_device
        and tuple(bundle.feature_order) == CANONICAL_FEATURE_ORDER
        and bundle.feature_order_sha256 == CANONICAL_FEATURE_ORDER_SHA256
        and bundle.threshold == CANONICAL_DECISION_THRESHOLD,
        "canonical model bundle method identity changed",
    )
    state = _feature_state_from_bundle(bundle)
    predictor = _load_bundle_predictor(
        bundle.classifier,
        bundle.imputer,
        device,
        predictor_loader,
    )
    predict = probability_predictor or predict_portable_probabilities
    assets = _runtime_assets_from_bundle(bundle)
    runtime = _load_inference_runtime(assets, device, runtime_factory)
    if runtime_factory is None:
        _attest_canonical_stage_runtime_cuda(runtime)
    nms = nms_function or _default_nms

    candidates: list[dict[str, object]] = []
    raw_feature_rows: list[dict[str, object]] = []
    input_inventory: list[dict[str, object]] = []
    tile_count = proposal_count = 0
    for source in _inference_image_paths(images):
        payload, snapshot = stable_file(source, max_bytes=MAX_IMAGE_FILE_BYTES)
        image_sha256 = hashlib.sha256(payload).hexdigest()
        group_id = group_id_from_name(source.name)
        image_id = _image_id(source.name, group_id, image_sha256)
        decoded = _decode_image(payload, source.name)
        source_height, source_width = (int(value) for value in decoded.shape[:2])
        _require(
            source_height <= MAX_IMAGE_DIMENSION
            and source_width <= MAX_IMAGE_DIMENSION
            and source_height * source_width <= MAX_IMAGE_PIXELS
            and int(decoded.nbytes) <= MAX_DECODED_BYTES,
            f"canonical inference image exceeds decoded bounds: {source.name}",
        )
        prepared = (
            full_image_prepare(decoded, source.name, locate_grid=True)
            if full_image
            else canonical_prepare_images(decoded, source.name, locate_grid=True)
        )
        if full_image:
            _full_image_analysis_unit(
                prepared,
                "full-image inference",
                image_name=source.name,
            )
        evidence = _plain(prepared.evidence)
        _require(
            isinstance(evidence, dict)
            and evidence.get("schema")
            == (
                "compag-curation-full-image-preprocessing/v1"
                if full_image
                else "compag-curation-canonical-preprocessing/v1"
            ),
            "inference preprocessing evidence is invalid",
        )
        image_record: dict[str, object] = {
            "image_name": source.name,
            "image_id": image_id,
            "image_sha256": image_sha256,
            "group_id": group_id,
            "source_width": source_width,
            "source_height": source_height,
            "warped_width": int(prepared.warped_bgr.shape[1]),
            "warped_height": int(prepared.warped_bgr.shape[0]),
            "inverse_warp": evidence["warp"]["inverse_matrix"],
            "row_lines": list(prepared.row_lines),
            "column_lines": list(prepared.column_lines),
        }
        image_candidate_start = len(candidates)
        for tile in prepared.tiles:
            encoded = (
                encode_full_image_png(tile)
                if full_image
                else encode_canonical_jpeg(tile)
            )
            tile_sha256 = hashlib.sha256(encoded).hexdigest()
            tile_bgr = _decode_image(encoded, tile.name)
            tile_row: dict[str, object] = {
                "tile_name": tile.name,
                "image_id": image_id,
                "image_sha256": image_sha256,
                "image_name": source.name,
                "group_id": group_id,
                "tile_sha256": tile_sha256,
                "x": tile.x,
                "y": tile.y,
                "crop_w": tile.crop_width,
                "crop_h": tile.crop_height,
                "orig_w": source_width,
                "orig_h": source_height,
            }
            generator = (
                _generate_full_image_records if full_image else _generate_tile_records
            )
            generated = generator(
                runtime,
                tile_bgr,
                tile_row,
                image_record,
                attest_cuda=runtime_factory is None,
            )
            raw_feature_rows.extend(generated.raw_rows)
            _require(
                len(raw_feature_rows) <= MAX_CANONICAL_TABLE_ROWS,
                "canonical inference raw-feature row bound exceeded",
            )
            proposal_by_id = {
                str(row["proposal_id"]): row for row in generated.proposal_rows
            }
            inference_pairs = [
                (raw, row)
                for raw, row in zip(
                    generated.raw_features,
                    generated.raw_rows,
                    strict=True,
                )
                if raw.scale == 1.0
            ]
            _require(
                len(inference_pairs) == len(generated.proposal_rows),
                "canonical inference requires exactly one scale-1.0 row per proposal",
            )
            finalized = [
                finalize_canonical_feature(raw, state)
                for raw, _row in inference_pairs
            ]
            probabilities = (
                np.asarray(predict(predictor, finalized), dtype=np.float32).reshape(-1)
                if finalized
                else np.empty(0, dtype=np.float32)
            )
            _require(
                len(probabilities) == len(inference_pairs)
                and bool(np.all(np.isfinite(probabilities)))
                and bool(np.all((probabilities >= 0.0) & (probabilities <= 1.0))),
                "canonical portable predictor returned invalid probabilities",
            )
            for (_raw, raw_row), probability in zip(
                inference_pairs,
                probabilities,
                strict=True,
            ):
                proposal = proposal_by_id[str(raw_row["proposal_id"])]
                scale = float(raw_row["scale"])
                detection_id = hashlib.sha256(
                    b"compag-canonical-detection-v1\0"
                    + str(raw_row["proposal_id"]).encode("ascii")
                    + b"\0"
                    + _format_float(scale).encode("ascii")
                ).hexdigest()
                candidates.append(
                    {
                        "detection_id": detection_id,
                        "proposal": proposal,
                        "scale": scale,
                        "probability": float(probability),
                        "kept": 0,
                    }
                )
            tile_count += 1
            proposal_count += len(generated.proposal_rows)
            _require(len(candidates) <= MAX_CANONICAL_TABLE_ROWS, "canonical inference row bound exceeded")

        image_candidates = sorted(
            candidates[image_candidate_start:],
            key=lambda candidate: str(candidate["proposal"]["proposal_id"]),
        )
        candidates[image_candidate_start:] = image_candidates
        positive_indices = [
            index
            for index, candidate in enumerate(image_candidates)
            if float(candidate["probability"]) >= CANONICAL_DECISION_THRESHOLD
            and int(candidate["proposal"]["bbox_w"]) > 0
            and int(candidate["proposal"]["bbox_h"]) > 0
        ]
        if positive_indices:
            boxes = np.asarray(
                [
                    [
                        float(image_candidates[index]["proposal"]["bbox_x"]),
                        float(image_candidates[index]["proposal"]["bbox_y"]),
                        float(image_candidates[index]["proposal"]["bbox_x"])
                        + float(image_candidates[index]["proposal"]["bbox_w"]),
                        float(image_candidates[index]["proposal"]["bbox_y"])
                        + float(image_candidates[index]["proposal"]["bbox_h"]),
                    ]
                    for index in positive_indices
                ],
                dtype=np.float32,
            )
            scores = np.asarray(
                [float(image_candidates[index]["probability"]) for index in positive_indices],
                dtype=np.float32,
            )
            kept_relative = _canonical_nms_indices(
                boxes,
                scores,
                [
                    str(image_candidates[index]["proposal"]["proposal_id"])
                    for index in positive_indices
                ],
                CANONICAL_FULL_IMAGE_NMS_IOU,
                nms_function=nms,
            )
            for relative_index in kept_relative:
                image_candidates[positive_indices[int(relative_index)]]["kept"] = 1
        input_inventory.append(
            {
                "image_name": source.name,
                "image_id": image_id,
                "image_sha256": image_sha256,
                "group_id": group_id,
                "size_bytes": snapshot.st_size,
                "width": source_width,
                "height": source_height,
                "tile_count": len(prepared.tiles),
                "candidate_rows": len(image_candidates),
            }
        )

    candidates.sort(
        key=lambda candidate: (
            str(candidate["proposal"]["image_id"]),
            str(candidate["proposal"]["proposal_id"]),
            CANONICAL_FEATURE_CROP_SCALES.index(float(candidate["scale"])),
        )
    )
    prediction_rows: list[dict[str, object]] = []
    for ordinal, candidate in enumerate(candidates, start=1):
        proposal = candidate["proposal"]
        probability = float(candidate["probability"])
        x = int(proposal["bbox_x"])
        y = int(proposal["bbox_y"])
        width = int(proposal["bbox_w"])
        height = int(proposal["bbox_h"])
        row: dict[str, object] = {
            "id": str(ordinal),
            "detection_id": candidate["detection_id"],
            "proposal_id": proposal["proposal_id"],
            "proposal_sha256": proposal["proposal_sha256"],
            "image": proposal["image_name"],
            "full_image": proposal["image_name"],
            "image_id": proposal["image_id"],
            "image_sha256": proposal["image_sha256"],
            "group_id": proposal["group_id"],
            "tile_name": proposal["tile_name"],
            "tile_sha256": proposal["tile_sha256"],
            "scale": _format_float(candidate["scale"]),
            "mask_sha256": proposal["mask_sha256"],
            "x": str(x),
            "y": str(y),
            "w": str(width),
            "h": str(height),
            "bbox_x1": str(x),
            "bbox_y1": str(y),
            "bbox_x2": str(x + width),
            "bbox_y2": str(y + height),
            "orig_w": str(next(item["width"] for item in input_inventory if item["image_id"] == proposal["image_id"])),
            "orig_h": str(next(item["height"] for item in input_inventory if item["image_id"] == proposal["image_id"])),
            "poly": proposal["poly"],
            "xgb_p": _format_float(probability),
            "probability": _format_float(probability),
            "prediction": str(int(probability >= CANONICAL_DECISION_THRESHOLD)),
            "kept": str(int(candidate["kept"])),
        }
        _require(tuple(row) == CANONICAL_INFERENCE_COLUMNS, "canonical inference CSV schema changed")
        prediction_rows.append(row)
    _write_csv_exact(
        output / "predictions.csv",
        CANONICAL_INFERENCE_COLUMNS,
        prediction_rows,
        require_rows=False,
    )
    from .active_learning_service import (
        write_canonical_active_learning_raw_feature_archive,
    )

    raw_feature_archive = write_canonical_active_learning_raw_feature_archive(
        raw_feature_rows,
        output / "predictions.csv",
        output / "raw_features.csv",
        bundle_sha256=bundle.bundle_sha256,
        profile=bundle.profile,
    )
    bundle_after = verify(bundle.root)
    _require(
        bundle_after.bundle_sha256 == bundle.bundle_sha256
        and tuple(bundle_after.tree_identity) == tuple(bundle.tree_identity),
        "canonical model bundle changed during inference",
    )
    result = {
        "schema": "compag-curation-canonical-inference/v1",
        "status": "PASS",
        "profile": bundle.profile,
        "device": device,
        "bundle_sha256": bundle.bundle_sha256,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        "threshold": CANONICAL_DECISION_THRESHOLD,
        "threshold_method": "FIXED_CANONICAL_METHOD",
        "full_image_nms_iou": CANONICAL_FULL_IMAGE_NMS_IOU,
        "sam2_execution_points_per_batch": _inference_execution_batch(
            bundle.profile
        ),
        "image_count": len(input_inventory),
        "tile_count": tile_count,
        "proposal_count": proposal_count,
        "prediction_rows": len(prediction_rows),
        "positive_rows": sum(int(row["prediction"]) for row in prediction_rows),
        "kept_rows": sum(int(row["kept"]) for row in prediction_rows),
        "predictions_sha256": sha256_file(output / "predictions.csv"),
        "raw_feature_archive": raw_feature_archive,
        "input_inventory": input_inventory,
        "paper_result_reproduction": "NOT_CLAIMED",
    }
    write_new_json(output / "inference_result.json", result)
    return result


def infer_canonical_bundle(
    images: Path,
    bundle_root: Path,
    output: Path,
    *,
    device: str = "cuda",
) -> dict[str, object]:
    """Run seam-free verified v2-bundle canonical inference."""

    resolved_device = _require_cuda_service_device(device, "bundle inference")
    return _infer_canonical_bundle_with_dependencies(
        images,
        bundle_root,
        output,
        device=resolved_device,
    )


__all__ = [
    "CANONICAL_FINAL_FEATURE_COLUMNS",
    "CANONICAL_INFERENCE_COLUMNS",
    "CANONICAL_PROPOSAL_COLUMNS",
    "CANONICAL_RAW_FEATURE_TABLE_COLUMNS",
    "CANONICAL_TEST_PREDICTION_COLUMNS",
    "CANONICAL_TILE_COLUMNS",
    "FULL_IMAGE_MASK_ENCODING",
    "FULL_IMAGE_UNIT_COLUMNS",
    "CanonicalRuntimeAssets",
    "CanonicalStageRuntime",
    "build_canonical_group_split",
    "canonical_review_split_feasibility",
    "decode_lossless_mask",
    "generate_canonical_stage",
    "infer_canonical_bundle",
    "load_canonical_runtime",
    "prepare_canonical_stage",
    "train_canonical_stage",
]
