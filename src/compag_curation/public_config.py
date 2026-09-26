"""Versioned user-project configuration for the runnable public pipeline."""

from __future__ import annotations

import hashlib
import math
import re
import stat
import struct
import tomllib
import uuid
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .canonical.spec import (
    CANONICAL_CPU_PROFILE,
    CANONICAL_GPU_PROFILE,
    CANONICAL_PROFILE,
    CANONICAL_PROFILES,
    EFFICIENT_GPU_PROFILE,
    FULL_IMAGE_GPU_PROFILE,
    GPU_EXECUTION_PROFILES,
    V2_PIPELINE_PROFILES,
)
from .public_io import (
    PublicIOError,
    fsync_directory,
    portable_basename,
    publish_directory_noreplace,
    safe_relative,
    sha256_file,
    stable_file,
    write_new_bytes,
    write_new_json,
)


LEGACY_SCHEMA = "compag-curation-public-project/v1"
SCHEMA = "compag-curation-public-project/v2"
BALANCED_PROFILE = "public-safe-balanced-v1"
TILED_SPATIAL_MODE = "tiled"
FULL_IMAGE_SPATIAL_MODE = "full-image-multiscale"
LEGACY_SUPPORTED_PROFILES = (
    BALANCED_PROFILE,
    CANONICAL_CPU_PROFILE,
    CANONICAL_GPU_PROFILE,
    EFFICIENT_GPU_PROFILE,
)
SUPPORTED_PROFILES = (BALANCED_PROFILE, *V2_PIPELINE_PROFILES)
# Historical CPU and balanced identities remain structurally readable, while
# new projects may select one of the explicit CUDA execution profiles.
PUBLIC_INITIALIZATION_PROFILES = GPU_EXECUTION_PROFILES
PROFILE = CANONICAL_GPU_PROFILE
SAM2_TINY_CHECKPOINT_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
SAM2_TINY_CONFIG_SHA256 = "f932eac1c6241e910031b2f000a81cd9f8a8d4896e2277ab5ffb721f378b188d"
SAM2_LARGE_CHECKPOINT_SHA256 = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
SAM2_LARGE_CONFIG_SHA256 = "1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107"
RESNET50_WEIGHTS_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".ppm")
MAX_IMAGE_FILE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_DIMENSION = 16_384
MAX_IMAGE_PIXELS = 100_000_000
MAX_DECODED_BYTES = 400_000_000
MAX_CONFIG_BYTES = 1024 * 1024
PUBLIC_FEATURE_ORDER = (
    "delta_b", "delta_a", "area_norm", "elongation", "eccentricity",
    "solidity", "aspect_ratio", "circularity", "extent", "bbox_diag_frac",
    "perim_over_sqrt_area", "LBP_u5", "GLCM_contrast", "GLCM_homogeneity",
    "grad_mean", "grad_p90", "mean_L", "std_L", "std_a", "std_b",
    "grid_r_norm", "grid_c_norm", "scale_diag", "pred_iou", "stability",
)
FEATURE_ORDER_SHA256 = hashlib.sha256("\n".join(PUBLIC_FEATURE_ORDER).encode("ascii")).hexdigest()
if FEATURE_ORDER_SHA256 != "8f92995489dba4061f102cc523c7209e61dafe7b1425a02a1669e5de1e3b44f9":
    raise RuntimeError("public feature-order identity changed")
CANONICAL_FEATURE_ORDER_SHA256 = "a5478eb2e571809d80fbdf0e132d39f301d936b0b711ceb08a326ade0b24ed47"


class PublicConfigurationError(RuntimeError):
    """Raised for an invalid public project or unsafe local path."""


@dataclass(frozen=True)
class PublicProjectConfig:
    config_path: Path
    project_root: Path
    project_name: str
    images: Path
    inference_images: Path
    annotations: Path | None
    asset_root: Path
    device: str
    seed: int
    tile_size: int
    tile_overlap: int
    points_per_side: int
    points_per_batch: int
    pred_iou_threshold: float
    stability_threshold: float
    max_proposals_per_tile: int
    min_mask_area: int
    checkpoint: Path
    checkpoint_sha256: str
    sam2_config: Path
    sam2_config_sha256: str
    sam2_config_locator: str
    config_sha256: str
    profile: str = PROFILE
    spatial_mode: str = TILED_SPATIAL_MODE
    max_proposals_per_image: int | None = None
    execution_points_per_batch: int | None = None
    tile_stride: int = 448
    tile_edge_alignment: str = "retain-partial-edge"
    tile_padding: str = "none"
    tile_format: str = "png"
    proposal_backend: str = "sam2-amg"
    proposal_scales: tuple[float, ...] = (1.0,)
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 1
    crop_overlap_ratio: float = 512 / 1500
    exclude_largest_mask: bool = False
    feature_mode: str = "balanced"
    feature_crop_scales: tuple[float, ...] = (1.0,)
    embedding_backbone: str | None = None
    embedding_dimensions: int = 0
    masked_crop_padding: float = 0.0
    pca_components: int = 0
    group_test_fraction: float = 0.2
    group_cv_splits: int = 0
    hyperparameter_search_iterations: int = 0
    early_stopping_rounds: int = 0
    inner_validation_fraction: float = 0.2
    review_accept_weight: float = 1.0
    review_flip_weight: float = 1.0
    review_suspect_weight: float = 1.0
    review_skip_weight: float = 0.0
    safe_smote: bool = True
    scale_pos_weight: float = 1.0
    inference_threshold: float | None = None
    nms_iou_threshold: float = 0.5
    yolo_enabled: bool = False
    embedding_weights: Path | None = None
    embedding_weights_sha256: str | None = None

    def normalized(self) -> dict[str, object]:
        value = asdict(self)
        for key in (
            "config_path", "project_root", "images", "inference_images", "annotations",
            "asset_root", "checkpoint", "sam2_config", "embedding_weights",
        ):
            item = value[key]
            value[key] = None if item is None else str(Path(item))
        return value

    @property
    def is_canonical(self) -> bool:
        return self.profile in CANONICAL_PROFILES

    @property
    def uses_v2_pipeline(self) -> bool:
        return self.profile in V2_PIPELINE_PROFILES

    @property
    def is_gpu_execution_profile(self) -> bool:
        return self.profile in GPU_EXECUTION_PROFILES

    @property
    def uses_full_image_multiscale(self) -> bool:
        return self.spatial_mode == FULL_IMAGE_SPATIAL_MODE


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicConfigurationError(message)


def _mapping(value: object, role: str) -> Mapping[str, object]:
    _require(isinstance(value, dict), f"{role} must be a table")
    return value


def _exact(mapping: Mapping[str, object], expected: set[str], role: str) -> None:
    _require(set(mapping) == expected, f"{role} fields must be exactly {sorted(expected)}")


def _text(mapping: Mapping[str, object], key: str, role: str) -> str:
    value = mapping.get(key)
    _require(isinstance(value, str) and value.strip() == value and value != "", f"{role}.{key} must be nonblank text")
    return value


def _number(mapping: Mapping[str, object], key: str, role: str, kind: type | tuple[type, ...]) -> int | float:
    value = mapping.get(key)
    _require(isinstance(value, kind) and not isinstance(value, bool), f"{role}.{key} has the wrong type")
    return value


def _boolean(mapping: Mapping[str, object], key: str, role: str) -> bool:
    value = mapping.get(key)
    _require(isinstance(value, bool), f"{role}.{key} must be boolean")
    return value


def _float_sequence(mapping: Mapping[str, object], key: str, role: str) -> tuple[float, ...]:
    value = mapping.get(key)
    _require(isinstance(value, list) and value, f"{role}.{key} must be a nonempty array")
    _require(
        all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)) for item in value),
        f"{role}.{key} must contain only finite numbers",
    )
    return tuple(float(item) for item in value)


def _hash(mapping: Mapping[str, object], key: str, role: str) -> str:
    value = _text(mapping, key, role)
    _require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"{role}.{key} must be lowercase SHA-256")
    return value


def _resolve(root: Path, value: str, role: str) -> Path:
    relative = safe_relative(value, role)
    _require(
        all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", part) is not None for part in relative.parts),
        f"{role} contains a non-portable path segment",
    )
    candidate = root.joinpath(*relative.parts)
    _require(candidate == root / relative.as_posix(), f"{role} is not confined")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        _require(not stat.S_ISLNK(info.st_mode), f"{role} contains a symlink component")
    return candidate


_BALANCED_EXECUTION_FIELDS = {
    "profile", "device", "seed", "tile_size", "tile_overlap", "points_per_side",
    "points_per_batch", "pred_iou_threshold", "stability_threshold",
    "max_proposals_per_tile", "min_mask_area",
}
_CANONICAL_EXECUTION_FIELDS = {
    "profile", "device", "seed", "tile_size", "tile_overlap", "tile_stride",
    "tile_edge_alignment", "tile_padding", "tile_format", "proposal_backend",
    "proposal_scales", "points_per_side", "points_per_batch", "pred_iou_threshold",
    "stability_threshold", "crop_n_layers", "crop_n_points_downscale_factor",
    "crop_overlap_ratio", "exclude_largest_mask", "max_proposals_per_tile",
    "min_mask_area", "feature_mode", "feature_crop_scales", "embedding_backbone",
    "embedding_dimensions", "masked_crop_padding", "pca_components",
    "group_test_fraction", "group_cv_splits", "hyperparameter_search_iterations",
    "early_stopping_rounds", "inner_validation_fraction", "review_accept_weight",
    "review_flip_weight", "review_suspect_weight", "review_skip_weight", "safe_smote", "scale_pos_weight",
    "inference_threshold", "nms_iou_threshold", "yolo_enabled",
}
_V2_BALANCED_EXECUTION_FIELDS = {*_BALANCED_EXECUTION_FIELDS, "spatial_mode"}
_V2_TILED_EXECUTION_FIELDS = {*_CANONICAL_EXECUTION_FIELDS, "spatial_mode"}
_FULL_IMAGE_EXECUTION_FIELDS = {
    *_V2_TILED_EXECUTION_FIELDS,
    "execution_points_per_batch",
    "max_proposals_per_image",
}
_SAM2_ASSET_FIELDS = {
    "checkpoint", "checkpoint_sha256", "sam2_config", "sam2_config_sha256",
    "sam2_config_locator",
}
_CANONICAL_ASSET_FIELDS = {
    *_SAM2_ASSET_FIELDS, "embedding_weights", "embedding_weights_sha256",
}


def validate_public_mapping(value: object, *, config_path: Path, config_sha256: str) -> PublicProjectConfig:
    top = _mapping(value, "configuration")
    _exact(top, {"schema", "project", "paths", "execution", "assets"}, "configuration")
    source_schema = top.get("schema")
    _require(
        isinstance(source_schema, str)
        and source_schema in (LEGACY_SCHEMA, SCHEMA),
        f"schema must be one of {(LEGACY_SCHEMA, SCHEMA)}",
    )
    is_legacy = source_schema == LEGACY_SCHEMA
    project = _mapping(top["project"], "project")
    paths = _mapping(top["paths"], "paths")
    execution = _mapping(top["execution"], "execution")
    assets = _mapping(top["assets"], "assets")
    _exact(project, {"name"}, "project")
    _exact(paths, {"images", "inference_images", "annotations", "asset_root"}, "paths")
    profile = _text(execution, "profile", "execution")
    allowed_profiles = LEGACY_SUPPORTED_PROFILES if is_legacy else SUPPORTED_PROFILES
    _require(profile in allowed_profiles, f"execution.profile must be one of {allowed_profiles}")
    if profile == BALANCED_PROFILE:
        _exact(
            execution,
            _BALANCED_EXECUTION_FIELDS if is_legacy else _V2_BALANCED_EXECUTION_FIELDS,
            "execution",
        )
        _exact(assets, _SAM2_ASSET_FIELDS, "assets")
    elif profile == FULL_IMAGE_GPU_PROFILE:
        _require(not is_legacy, f"{FULL_IMAGE_GPU_PROFILE} requires schema {SCHEMA}")
        _exact(execution, _FULL_IMAGE_EXECUTION_FIELDS, "execution")
        _exact(assets, _CANONICAL_ASSET_FIELDS, "assets")
    else:
        _exact(
            execution,
            _CANONICAL_EXECUTION_FIELDS if is_legacy else _V2_TILED_EXECUTION_FIELDS,
            "execution",
        )
        _exact(assets, _CANONICAL_ASSET_FIELDS, "assets")
    root = config_path.parent.resolve(strict=True)
    name = _text(project, "name", "project")
    _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name) is not None, "project.name is not portable")
    annotations_text = paths.get("annotations")
    _require(isinstance(annotations_text, str), "paths.annotations must be text")
    device = _text(execution, "device", "execution")
    required_device = "cuda" if profile in GPU_EXECUTION_PROFILES else "cpu"
    _require(
        device == required_device,
        f"{profile} requires execution.device={required_device}",
    )
    seed = int(_number(execution, "seed", "execution", int))
    _require(seed == 42, "execution.seed must remain 42")
    spatial_mode = (
        TILED_SPATIAL_MODE
        if is_legacy
        else _text(execution, "spatial_mode", "execution")
    )
    required_spatial_mode = (
        FULL_IMAGE_SPATIAL_MODE
        if profile == FULL_IMAGE_GPU_PROFILE
        else TILED_SPATIAL_MODE
    )
    _require(
        spatial_mode == required_spatial_mode,
        f"{profile} requires execution.spatial_mode={required_spatial_mode}",
    )
    tile_size = int(_number(execution, "tile_size", "execution", int))
    tile_overlap = int(_number(execution, "tile_overlap", "execution", int))
    points_per_side = int(_number(execution, "points_per_side", "execution", int))
    points_per_batch = int(_number(execution, "points_per_batch", "execution", int))
    pred_iou = float(_number(execution, "pred_iou_threshold", "execution", (int, float)))
    stability = float(_number(execution, "stability_threshold", "execution", (int, float)))
    max_proposals = int(_number(execution, "max_proposals_per_tile", "execution", int))
    max_proposals_per_image = (
        int(_number(execution, "max_proposals_per_image", "execution", int))
        if profile == FULL_IMAGE_GPU_PROFILE
        else None
    )
    execution_points_per_batch = (
        int(_number(execution, "execution_points_per_batch", "execution", int))
        if profile == FULL_IMAGE_GPU_PROFILE
        else None
    )
    min_area = int(_number(execution, "min_mask_area", "execution", int))
    _require(2 <= points_per_side <= 64 and 1 <= points_per_batch <= 512, "SAM2 sampling values are out of range")
    _require(math.isfinite(pred_iou) and math.isfinite(stability) and 0.0 <= pred_iou <= 1.0 and 0.0 <= stability <= 1.0, "SAM2 thresholds must be finite probabilities")
    _require(2 <= max_proposals <= 500 and 0 <= min_area <= tile_size * tile_size, "proposal limits are invalid")
    if profile == FULL_IMAGE_GPU_PROFILE:
        _require(
            max_proposals_per_image == 1000,
            "full-image multiscale proposal cap must remain 1000 per image",
        )
        _require(
            execution_points_per_batch == 8,
            "full-image execution prompt microbatch must remain 8",
        )
    locator = _text(assets, "sam2_config_locator", "assets")
    checkpoint_sha256 = _hash(assets, "checkpoint_sha256", "assets")
    sam2_config_sha256 = _hash(assets, "sam2_config_sha256", "assets")
    canonical: dict[str, object] = {}
    embedding_weights: Path | None = None
    embedding_weights_sha256: str | None = None
    if profile == BALANCED_PROFILE:
        _require(tile_size == 512 and tile_overlap == 64, "public-safe-balanced-v1 tiling contract is fixed at 512/64")
        _require(locator == "configs/sam2.1/sam2.1_hiera_t", "public-safe-balanced-v1 requires the registered SAM2.1 Hiera Tiny configuration")
        _require(checkpoint_sha256 == SAM2_TINY_CHECKPOINT_SHA256, "public-safe-balanced-v1 checkpoint SHA-256 differs from the registered asset")
        _require(sam2_config_sha256 == SAM2_TINY_CONFIG_SHA256, "public-safe-balanced-v1 SAM2 configuration SHA-256 differs from the registered asset")
    else:
        canonical = {
            "tile_stride": int(_number(execution, "tile_stride", "execution", int)),
            "tile_edge_alignment": _text(execution, "tile_edge_alignment", "execution"),
            "tile_padding": _text(execution, "tile_padding", "execution"),
            "tile_format": _text(execution, "tile_format", "execution"),
            "proposal_backend": _text(execution, "proposal_backend", "execution"),
            "proposal_scales": _float_sequence(execution, "proposal_scales", "execution"),
            "crop_n_layers": int(_number(execution, "crop_n_layers", "execution", int)),
            "crop_n_points_downscale_factor": int(_number(execution, "crop_n_points_downscale_factor", "execution", int)),
            "crop_overlap_ratio": float(_number(execution, "crop_overlap_ratio", "execution", (int, float))),
            "exclude_largest_mask": _boolean(execution, "exclude_largest_mask", "execution"),
            "feature_mode": _text(execution, "feature_mode", "execution"),
            "feature_crop_scales": _float_sequence(execution, "feature_crop_scales", "execution"),
            "embedding_backbone": _text(execution, "embedding_backbone", "execution"),
            "embedding_dimensions": int(_number(execution, "embedding_dimensions", "execution", int)),
            "masked_crop_padding": float(_number(execution, "masked_crop_padding", "execution", (int, float))),
            "pca_components": int(_number(execution, "pca_components", "execution", int)),
            "group_test_fraction": float(_number(execution, "group_test_fraction", "execution", (int, float))),
            "group_cv_splits": int(_number(execution, "group_cv_splits", "execution", int)),
            "hyperparameter_search_iterations": int(_number(execution, "hyperparameter_search_iterations", "execution", int)),
            "early_stopping_rounds": int(_number(execution, "early_stopping_rounds", "execution", int)),
            "inner_validation_fraction": float(_number(execution, "inner_validation_fraction", "execution", (int, float))),
            "review_accept_weight": float(_number(execution, "review_accept_weight", "execution", (int, float))),
            "review_flip_weight": float(_number(execution, "review_flip_weight", "execution", (int, float))),
            "review_suspect_weight": float(_number(execution, "review_suspect_weight", "execution", (int, float))),
            "review_skip_weight": float(_number(execution, "review_skip_weight", "execution", (int, float))),
            "safe_smote": _boolean(execution, "safe_smote", "execution"),
            "scale_pos_weight": float(_number(execution, "scale_pos_weight", "execution", (int, float))),
            "inference_threshold": float(_number(execution, "inference_threshold", "execution", (int, float))),
            "nms_iou_threshold": float(_number(execution, "nms_iou_threshold", "execution", (int, float))),
            "yolo_enabled": _boolean(execution, "yolo_enabled", "execution"),
        }
        required_pipeline = {
            "tile_stride": 512,
            "tile_edge_alignment": "far-edge",
            "tile_padding": "bottom-right-edge-value",
            "tile_format": "jpg",
            "proposal_backend": "sam2-amg",
            "proposal_scales": (1.0,),
            "crop_n_layers": 2 if profile == FULL_IMAGE_GPU_PROFILE else 0,
            "crop_n_points_downscale_factor": (
                1 if profile == FULL_IMAGE_GPU_PROFILE else 2
            ),
            "crop_overlap_ratio": (
                512 / 1500 if profile == FULL_IMAGE_GPU_PROFILE else 0.4
            ),
            "exclude_largest_mask": False,
            "feature_mode": "ultra",
            "feature_crop_scales": (0.67, 0.8, 1.0, 1.25),
            "embedding_backbone": "resnet50-imagenet1k-v2",
            "embedding_dimensions": 2048,
            "masked_crop_padding": 0.1,
            "pca_components": 32,
            "group_test_fraction": 0.2,
            "group_cv_splits": 5,
            "hyperparameter_search_iterations": 30,
            "early_stopping_rounds": 30,
            "inner_validation_fraction": 0.3,
            "review_accept_weight": 1.0,
            "review_flip_weight": 1.0,
            "review_suspect_weight": 0.4,
            "review_skip_weight": 0.0,
            "safe_smote": True,
            "scale_pos_weight": 1.0,
            "inference_threshold": 0.5,
            "nms_iou_threshold": 0.5,
            "yolo_enabled": False,
        }
        _require(
            tile_size == 512 and tile_overlap == 0,
            (
                "full-image compatibility fields must remain 512/0 and are not used to tile images"
                if profile == FULL_IMAGE_GPU_PROFILE
                else "v2 tiling requires 512-pixel tiles and zero regular-grid overlap"
            ),
        )
        _require(points_per_side == 64 and points_per_batch == 512, "v2 SAM2 sampling must remain 64/512")
        _require(pred_iou == 0.8 and stability == 0.88, "v2 SAM2 thresholds must remain 0.80/0.88")
        _require(max_proposals == 500 and min_area == 0, "v2 proposal cap/minimum area must remain 500/0")
        _require(canonical == required_pipeline, f"{profile} execution defaults are immutable")
        expected_sam2 = (
            (
                "configs/sam2.1/sam2.1_hiera_t",
                SAM2_TINY_CHECKPOINT_SHA256,
                SAM2_TINY_CONFIG_SHA256,
                "efficient profile requires the registered SAM2.1 Hiera Tiny assets",
            )
            if profile == EFFICIENT_GPU_PROFILE
            else (
                "configs/sam2.1/sam2.1_hiera_l",
                SAM2_LARGE_CHECKPOINT_SHA256,
                SAM2_LARGE_CONFIG_SHA256,
                (
                    "full-image profile requires the registered SAM2.1 Hiera Large assets"
                    if profile == FULL_IMAGE_GPU_PROFILE
                    else "canonical profile requires the registered SAM2.1 Hiera Large assets"
                ),
            )
        )
        expected_locator, expected_checkpoint, expected_config, asset_error = expected_sam2
        _require(
            (locator, checkpoint_sha256, sam2_config_sha256)
            == (expected_locator, expected_checkpoint, expected_config),
            asset_error,
        )
        embedding_weights_sha256 = _hash(assets, "embedding_weights_sha256", "assets")
        _require(embedding_weights_sha256 == RESNET50_WEIGHTS_SHA256, "v2 ResNet50 weights SHA-256 differs from the registered asset")
        embedding_weights = _resolve(root, _text(assets, "embedding_weights", "assets"), "assets.embedding_weights")
    return PublicProjectConfig(
        config_path=config_path.resolve(strict=True),
        project_root=root,
        project_name=name,
        images=_resolve(root, _text(paths, "images", "paths"), "paths.images"),
        inference_images=_resolve(root, _text(paths, "inference_images", "paths"), "paths.inference_images"),
        annotations=None if annotations_text == "" else _resolve(root, annotations_text, "paths.annotations"),
        asset_root=_resolve(root, _text(paths, "asset_root", "paths"), "paths.asset_root"),
        device=device,
        seed=seed,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_threshold=pred_iou,
        stability_threshold=stability,
        max_proposals_per_tile=max_proposals,
        min_mask_area=min_area,
        checkpoint=_resolve(root, _text(assets, "checkpoint", "assets"), "assets.checkpoint"),
        checkpoint_sha256=checkpoint_sha256,
        sam2_config=_resolve(root, _text(assets, "sam2_config", "assets"), "assets.sam2_config"),
        sam2_config_sha256=sam2_config_sha256,
        sam2_config_locator=locator,
        config_sha256=config_sha256,
        profile=profile,
        spatial_mode=spatial_mode,
        max_proposals_per_image=max_proposals_per_image,
        execution_points_per_batch=execution_points_per_batch,
        embedding_weights=embedding_weights,
        embedding_weights_sha256=embedding_weights_sha256,
        **canonical,
    )


def _jpeg_info(payload: bytes) -> tuple[int, int, int | None]:
    _require(payload.startswith(b"\xff\xd8"), "JPEG signature is invalid")
    offset = 2
    width = height = 0
    orientation: int | None = None
    saw_scan = False
    while offset + 4 <= len(payload):
        if payload[offset] != 0xFF:
            offset += 1
            continue
        marker = payload[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        _require(offset + 2 <= len(payload), "JPEG segment is truncated")
        length = int.from_bytes(payload[offset : offset + 2], "big")
        _require(length >= 2 and offset + length <= len(payload), "JPEG segment length is invalid")
        segment = payload[offset + 2 : offset + length]
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF} and len(segment) >= 5:
            height = int.from_bytes(segment[1:3], "big")
            width = int.from_bytes(segment[3:5], "big")
        if marker == 0xE1 and segment.startswith(b"Exif\x00\x00"):
            tiff = segment[6:]
            if len(tiff) >= 8 and tiff[:2] in {b"II", b"MM"}:
                order = "little" if tiff[:2] == b"II" else "big"
                ifd = int.from_bytes(tiff[4:8], order)
                if ifd + 2 <= len(tiff):
                    count = int.from_bytes(tiff[ifd : ifd + 2], order)
                    for index in range(count):
                        start = ifd + 2 + index * 12
                        if start + 12 > len(tiff):
                            break
                        if int.from_bytes(tiff[start : start + 2], order) == 0x0112:
                            orientation = int.from_bytes(tiff[start + 8 : start + 10], order)
                            break
        if marker == 0xDA:
            _require(segment, "JPEG scan header is empty")
            component_count = segment[0]
            _require(1 <= component_count <= 4, "JPEG scan component count is invalid")
            _require(len(segment) == 1 + 2 * component_count + 3, "JPEG scan header length is invalid")
            _require(offset + length < len(payload) - 2, "JPEG entropy-coded scan is empty")
            saw_scan = True
            break
        offset += length
    _require(width > 0 and height > 0 and saw_scan and payload.endswith(b"\xff\xd9"), "JPEG structure is incomplete")
    return width, height, orientation


def _image_info(path: Path, payload: bytes) -> tuple[int, int, int | None]:
    _require(0 < len(payload) <= MAX_IMAGE_FILE_BYTES, "image file size is outside the supported bound")
    suffix = path.suffix.lower()
    if suffix == ".png":
        _require(payload.startswith(b"\x89PNG\r\n\x1a\n") and len(payload) >= 33, "PNG signature is invalid")
        offset = 8
        chunks: list[tuple[bytes, bytes]] = []
        while offset + 12 <= len(payload):
            length = int.from_bytes(payload[offset : offset + 4], "big")
            kind = payload[offset + 4 : offset + 8]
            end = offset + 12 + length
            _require(end <= len(payload), "PNG chunk is truncated")
            data = payload[offset + 8 : offset + 8 + length]
            expected_crc = int.from_bytes(payload[offset + 8 + length : end], "big")
            _require(zlib.crc32(kind + data) & 0xFFFFFFFF == expected_crc, "PNG chunk CRC mismatch")
            chunks.append((kind, data))
            offset = end
            if kind == b"IEND":
                break
        _require(offset == len(payload) and chunks and chunks[0][0] == b"IHDR" and chunks[-1] == (b"IEND", b""), "PNG chunk closure is invalid")
        ihdr = chunks[0][1]
        _require(len(ihdr) == 13, "PNG IHDR length is invalid")
        width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", ihdr)
        _require(width > 0 and height > 0, "PNG dimensions are invalid")
        _require(width <= MAX_IMAGE_DIMENSION and height <= MAX_IMAGE_DIMENSION and width * height <= MAX_IMAGE_PIXELS, "PNG dimensions exceed the supported bound")
        _require(bit_depth == 8 and color_type in {0, 2, 4, 6}, "PNG must use 8-bit grayscale, RGB, grayscale-alpha, or RGBA")
        _require(compression == 0 and filtering == 0 and interlace == 0, "PNG compression/filter/interlace mode is unsupported")
        _require(not any(kind == b"eXIf" for kind, _data in chunks), "PNG EXIF metadata is unsupported; normalize a copy first")
        compressed = b"".join(data for kind, data in chunks if kind == b"IDAT")
        _require(compressed, "PNG contains no IDAT payload")
        channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
        stride = width * channels + 1
        expected = stride * height
        _require(expected <= MAX_DECODED_BYTES, "PNG decoded size exceeds the supported bound")
        try:
            decompressor = zlib.decompressobj()
            raster = decompressor.decompress(compressed, expected + 1)
        except zlib.error as exc:
            raise PublicConfigurationError("PNG raster stream is corrupt") from exc
        _require(
            len(raster) == expected and decompressor.eof and not decompressor.unused_data and not decompressor.unconsumed_tail,
            "PNG raster length or stream closure is invalid",
        )
        _require(all(raster[row * stride] <= 4 for row in range(height)), "PNG row filter is invalid")
        return width, height, None
    if suffix in {".jpg", ".jpeg"}:
        width, height, orientation = _jpeg_info(payload)
        _require(width <= MAX_IMAGE_DIMENSION and height <= MAX_IMAGE_DIMENSION and width * height <= MAX_IMAGE_PIXELS, "JPEG dimensions exceed the supported bound")
        return width, height, orientation
    if suffix == ".ppm":
        def tokens():
            offset = 0
            while offset < len(payload):
                while offset < len(payload) and payload[offset] in b" \t\r\n\v\f":
                    offset += 1
                if offset < len(payload) and payload[offset] == 0x23:
                    newline = payload.find(b"\n", offset + 1)
                    _require(newline >= 0, "PPM comment is not line terminated")
                    offset = newline + 1
                    continue
                if offset >= len(payload):
                    return
                start = offset
                while offset < len(payload) and payload[offset] not in b" \t\r\n\v\f#":
                    offset += 1
                _require(offset > start, "PPM token is empty")
                yield payload[start:offset], offset

        iterator = iter(tokens())
        header: list[tuple[bytes, int]] = []
        for _index in range(4):
            try:
                header.append(next(iterator))
            except StopIteration as exc:
                raise PublicConfigurationError("PPM header is incomplete") from exc
        _require(header[0][0] in {b"P3", b"P6"}, "PPM header is invalid")
        try:
            width, height, maximum = (int(header[index][0]) for index in (1, 2, 3))
        except ValueError as exc:
            raise PublicConfigurationError("PPM numeric header is invalid") from exc
        _require(width > 0 and height > 0 and maximum == 255, "PPM dimensions/maxval are invalid")
        _require(width <= MAX_IMAGE_DIMENSION and height <= MAX_IMAGE_DIMENSION and width * height <= MAX_IMAGE_PIXELS, "PPM dimensions exceed the supported bound")
        if header[0][0] == b"P3":
            expected_samples = width * height * 3
            sample_count = 0
            for value, _end in iterator:
                sample_count += 1
                _require(sample_count <= expected_samples, "P3 sample count is invalid")
                try:
                    sample = int(value)
                except ValueError as exc:
                    raise PublicConfigurationError("P3 sample is invalid") from exc
                _require(0 <= sample <= 255, "P3 sample is outside [0,255]")
            _require(sample_count == expected_samples, "P3 sample count is invalid")
        else:
            raster_offset = header[-1][1]
            _require(raster_offset < len(payload) and payload[raster_offset] in b" \t\r\n\v\f", "P6 header delimiter is missing")
            if payload[raster_offset : raster_offset + 2] == b"\r\n":
                raster_offset += 2
            else:
                raster_offset += 1
            _require(len(payload) - raster_offset == width * height * 3, "P6 raster length is invalid")
        return width, height, None
    raise PublicConfigurationError(f"unsupported image format: {path.suffix}")


def image_files(config: PublicProjectConfig, *, inference: bool = False, allow_empty: bool = False) -> list[Path]:
    root = config.inference_images if inference else config.images
    _require(root.exists() and root.is_dir() and not root.is_symlink(), "image directory is missing or unsafe")
    files = sorted((path for path in root.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES), key=lambda item: item.name)
    _require(files or allow_empty, "image directory contains no supported images")
    folded: set[str] = set()
    stems: set[str] = set()
    for path in files:
        portable_basename(path.name, "image name")
        _require("," not in path.name, f"image name contains a reserved separator: {path.name}")
        key = path.name.casefold()
        _require(key not in folded, f"image name collision: {path.name}")
        folded.add(key)
        stem_key = path.stem.casefold()
        _require(stem_key not in stems, f"image stems collide after suffix removal: {path.name}")
        stems.add(stem_key)
        if not bool(getattr(config, "uses_full_image_multiscale", False)):
            tile_extension = f".{config.tile_format}"
            _require(
                len(f"{path.stem}_y00000x00000{tile_extension}".encode("utf-8")) <= 255,
                f"derived {tile_extension} tile name exceeds NAME_MAX: {path.name}",
            )
        info = path.lstat()
        _require(not path.is_symlink() and stat.S_ISREG(info.st_mode), f"image is not regular: {path.name}")
        _require(0 < info.st_size <= MAX_IMAGE_FILE_BYTES, f"image file size is outside the supported bound: {path.name}")
    return files


def group_id_from_name(name: str) -> str:
    stem = Path(name).stem
    if "__" in stem:
        group = stem.split("__", 1)[0]
    else:
        group = re.sub(r"_y\d+x\d+$", "", stem, flags=re.IGNORECASE)
    _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", group) is not None, f"invalid group identifier derived from {name}")
    return group.casefold()


def inspect_public_data(
    config: PublicProjectConfig,
    *,
    decode_headers: bool = True,
    require_inputs: bool = False,
) -> dict[str, object]:
    def inspect(files: list[Path]) -> list[dict[str, object]]:
        rows = []
        for path in files:
            payload, info = stable_file(path, max_bytes=MAX_IMAGE_FILE_BYTES)
            width = height = 0
            orientation = None
            if decode_headers:
                width, height, orientation = _image_info(path, payload)
                _require(width >= 64 and height >= 64, f"image is smaller than 64x64: {path.name}")
                _require(orientation in {None, 1}, f"image EXIF orientation must be absent or 1: {path.name}")
            rows.append(
                {
                    "filename": path.name,
                    "group_id": group_id_from_name(path.name),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": info.st_size,
                    "width": width,
                    "height": height,
                    "exif_orientation": orientation,
                }
            )
        return rows

    rows = inspect(image_files(config, allow_empty=not require_inputs))
    # The caller reports the inference-specific prerequisite so validation errors
    # remain role-specific rather than surfacing a generic directory message.
    inference_rows = inspect(image_files(config, inference=True, allow_empty=True))
    overlap = {str(row["sha256"]) for row in rows} & {str(row["sha256"]) for row in inference_rows}
    _require(not overlap, "training and inference image sets overlap by content hash")
    group_overlap = {str(row["group_id"]) for row in rows} & {str(row["group_id"]) for row in inference_rows}
    _require(not group_overlap, "training and inference image sets overlap by group identity")
    _require(len({str(row["sha256"]) for row in rows}) == len(rows), "training images contain duplicate content")
    _require(len({str(row["sha256"]) for row in inference_rows}) == len(inference_rows), "inference images contain duplicate content")
    return {
        "schema": "compag-curation-data-inspection/v1",
        "status": "PASS",
        "image_count": len(rows),
        "group_count": len({row["group_id"] for row in rows}),
        "images": rows,
        "inference_image_count": len(inference_rows),
        "inference_group_count": len({row["group_id"] for row in inference_rows}),
        "inference_images": inference_rows,
    }


def check_local_assets(config: PublicProjectConfig) -> dict[str, object]:
    from .assets import asset_registry, verify_asset

    registry = asset_registry()
    checks = []
    if config.uses_v2_pipeline:
        _require(config.embedding_weights is not None, "v2 ResNet50 weights path is missing")
        sam2_checkpoint_id, sam2_config_id = (
            ("sam2.1-hiera-tiny-checkpoint", "sam2.1-hiera-tiny-config")
            if config.profile == EFFICIENT_GPU_PROFILE
            else ("sam2.1-hiera-large-checkpoint", "sam2.1-hiera-large-config")
        )
        rows = (
            ("SAM2 checkpoint", sam2_checkpoint_id, config.checkpoint, config.checkpoint_sha256),
            ("SAM2 configuration", sam2_config_id, config.sam2_config, config.sam2_config_sha256),
            ("ResNet50 weights", "resnet50-imagenet1k-v2-weights", config.embedding_weights, config.embedding_weights_sha256),
            ("SAM2 license", "sam2-apache-license", config.asset_root / "SAM2-APACHE-2.0.txt", "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
            ("torchvision license", "torchvision-bsd-license", config.asset_root / "TORCHVISION-BSD-3-CLAUSE.txt", "6502f676851cfe25f8af75531dfb32375b7325b73c37e7b43741fa422893e71d"),
        )
    else:
        rows = (
            ("SAM2 checkpoint", "sam2.1-hiera-tiny-checkpoint", config.checkpoint, config.checkpoint_sha256),
            ("SAM2 configuration", "sam2.1-hiera-tiny-config", config.sam2_config, config.sam2_config_sha256),
            ("SAM2 license", "sam2-apache-license", config.asset_root / "SAM2-APACHE-2.0.txt", "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
        )
    for role, asset_id, path, expected in rows:
        spec = registry[asset_id]
        _require(path == config.asset_root / spec.filename, f"{role} must use the registered asset filename")
        _require(expected == spec.sha256, f"{role} configuration hash differs from the registered asset")
        try:
            verified = verify_asset(config.asset_root, asset_id)
        except PublicIOError as exc:
            raise PublicIOError(f"{role}: {exc}") from exc
        checks.append({
            "role": role,
            "asset_id": asset_id,
            "filename": verified["filename"],
            "sha256": verified["sha256"],
            "size_bytes": verified["size_bytes"],
        })
    return {"status": "PASS", "assets": checks}


def load_public_config(path: Path, *, check_local: bool = False) -> PublicProjectConfig:
    try:
        node = path.lstat()
    except OSError as exc:
        raise PublicConfigurationError(f"configuration is unavailable: {path}") from exc
    _require(stat.S_ISREG(node.st_mode) and not path.is_symlink(), "configuration must be a regular non-symlink file")
    _require(path.absolute() == path.resolve(strict=True), "configuration path contains a symlink component")
    try:
        payload, _snapshot = stable_file(path, max_bytes=MAX_CONFIG_BYTES)
        raw = tomllib.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise PublicConfigurationError(f"configuration cannot be parsed: {path}") from exc
    try:
        config = validate_public_mapping(raw, config_path=path, config_sha256=hashlib.sha256(payload).hexdigest())
        if check_local:
            inspect_public_data(config, require_inputs=True)
            check_local_assets(config)
        return config
    except PublicIOError as exc:
        raise PublicConfigurationError(str(exc)) from exc


def build_public_plan(config: PublicProjectConfig, *, check_local: bool = False) -> dict[str, object]:
    inspection = inspect_public_data(config, require_inputs=True) if check_local else None
    return {
        "schema": "compag-curation-public-plan/v1",
        "status": "VALID",
        "scientific_execution": False,
        "profile": config.profile,
        "device": config.device,
        "stages": ["00_input_inventory", "10_prepare", "20_proposals_features", "30_review_import", "40_group_split", "50_train_bundle", "60_evaluate", "70_report"],
        "feature_order_sha256": (
            CANONICAL_FEATURE_ORDER_SHA256 if config.uses_v2_pipeline else FEATURE_ORDER_SHA256
        ),
        "local_inspection": inspection,
    }


BALANCED_PROJECT_TOML = f'''schema = "{SCHEMA}"

[project]
name = "sticky-card-project"

[paths]
images = "data/images"
inference_images = "data/inference_images"
annotations = ""
asset_root = "assets"

[execution]
profile = "{BALANCED_PROFILE}"
device = "cpu"
spatial_mode = "{TILED_SPATIAL_MODE}"
seed = 42
tile_size = 512
tile_overlap = 64
points_per_side = 8
points_per_batch = 16
pred_iou_threshold = 0.0
stability_threshold = 0.0
max_proposals_per_tile = 24
min_mask_area = 16

[assets]
checkpoint = "assets/sam2.1_hiera_tiny.pt"
checkpoint_sha256 = "{SAM2_TINY_CHECKPOINT_SHA256}"
sam2_config = "assets/sam2.1_hiera_t.yaml"
sam2_config_sha256 = "{SAM2_TINY_CONFIG_SHA256}"
sam2_config_locator = "configs/sam2.1/sam2.1_hiera_t"
'''


CANONICAL_CPU_PROJECT_TOML = f'''schema = "{SCHEMA}"

[project]
name = "sticky-card-project"

[paths]
images = "data/images"
inference_images = "data/inference_images"
annotations = ""
asset_root = "assets"

[execution]
profile = "{CANONICAL_CPU_PROFILE}"
device = "cpu"
spatial_mode = "{TILED_SPATIAL_MODE}"
seed = 42
tile_size = 512
tile_overlap = 0
tile_stride = 512
tile_edge_alignment = "far-edge"
tile_padding = "bottom-right-edge-value"
tile_format = "jpg"
proposal_backend = "sam2-amg"
proposal_scales = [1.0]
points_per_side = 64
points_per_batch = 512
pred_iou_threshold = 0.80
stability_threshold = 0.88
crop_n_layers = 0
crop_n_points_downscale_factor = 2
crop_overlap_ratio = 0.40
exclude_largest_mask = false
max_proposals_per_tile = 500
min_mask_area = 0
feature_mode = "ultra"
feature_crop_scales = [0.67, 0.80, 1.00, 1.25]
embedding_backbone = "resnet50-imagenet1k-v2"
embedding_dimensions = 2048
masked_crop_padding = 0.10
pca_components = 32
group_test_fraction = 0.20
group_cv_splits = 5
hyperparameter_search_iterations = 30
early_stopping_rounds = 30
inner_validation_fraction = 0.30
review_accept_weight = 1.0
review_flip_weight = 1.0
review_suspect_weight = 0.4
review_skip_weight = 0.0
safe_smote = true
scale_pos_weight = 1.0
inference_threshold = 0.50
nms_iou_threshold = 0.50
yolo_enabled = false

[assets]
checkpoint = "assets/sam2.1_hiera_large.pt"
checkpoint_sha256 = "{SAM2_LARGE_CHECKPOINT_SHA256}"
sam2_config = "assets/sam2.1_hiera_l.yaml"
sam2_config_sha256 = "{SAM2_LARGE_CONFIG_SHA256}"
sam2_config_locator = "configs/sam2.1/sam2.1_hiera_l"
embedding_weights = "assets/resnet50-11ad3fa6.pth"
embedding_weights_sha256 = "{RESNET50_WEIGHTS_SHA256}"
'''

# Retain the historical template name as an exact alias of the v1.2 CPU
# profile.  The GPU template changes only the execution identity and device;
# every scientific parameter and registered asset remains identical.
CANONICAL_PROJECT_TOML = CANONICAL_CPU_PROJECT_TOML
CANONICAL_GPU_PROJECT_TOML = CANONICAL_CPU_PROJECT_TOML.replace(
    f'profile = "{CANONICAL_CPU_PROFILE}"\ndevice = "cpu"',
    f'profile = "{CANONICAL_GPU_PROFILE}"\ndevice = "cuda"',
    1,
)
EFFICIENT_GPU_PROJECT_TOML = (
    CANONICAL_GPU_PROJECT_TOML.replace(
        f'profile = "{CANONICAL_GPU_PROFILE}"',
        f'profile = "{EFFICIENT_GPU_PROFILE}"',
        1,
    )
    .replace('checkpoint = "assets/sam2.1_hiera_large.pt"', 'checkpoint = "assets/sam2.1_hiera_tiny.pt"', 1)
    .replace(SAM2_LARGE_CHECKPOINT_SHA256, SAM2_TINY_CHECKPOINT_SHA256, 1)
    .replace('sam2_config = "assets/sam2.1_hiera_l.yaml"', 'sam2_config = "assets/sam2.1_hiera_t.yaml"', 1)
    .replace(SAM2_LARGE_CONFIG_SHA256, SAM2_TINY_CONFIG_SHA256, 1)
    .replace(
        'sam2_config_locator = "configs/sam2.1/sam2.1_hiera_l"',
        'sam2_config_locator = "configs/sam2.1/sam2.1_hiera_t"',
        1,
    )
)
FULL_IMAGE_GPU_PROJECT_TOML = (
    CANONICAL_GPU_PROJECT_TOML.replace(
        f'profile = "{CANONICAL_GPU_PROFILE}"',
        f'profile = "{FULL_IMAGE_GPU_PROFILE}"',
        1,
    )
    .replace(
        f'spatial_mode = "{TILED_SPATIAL_MODE}"',
        f'spatial_mode = "{FULL_IMAGE_SPATIAL_MODE}"',
        1,
    )
    .replace(
        "crop_n_layers = 0\ncrop_n_points_downscale_factor = 2\ncrop_overlap_ratio = 0.40",
        "crop_n_layers = 2\ncrop_n_points_downscale_factor = 1\n"
        "crop_overlap_ratio = 0.3413333333333333",
        1,
    )
    .replace(
        "points_per_batch = 512\npred_iou_threshold = 0.80",
        "points_per_batch = 512\nexecution_points_per_batch = 8\n"
        "pred_iou_threshold = 0.80",
        1,
    )
    .replace(
        "max_proposals_per_tile = 500\nmin_mask_area = 0",
        "max_proposals_per_tile = 500\nmax_proposals_per_image = 1000\nmin_mask_area = 0",
        1,
    )
)

# Full tiled remains the public API default; Full, Lite, and experimental
# Full-image CUDA profiles are available for project initialization.
PROJECT_TOML = CANONICAL_GPU_PROJECT_TOML


def initialize_project(output: Path, *, profile: str = PROFILE) -> dict[str, object]:
    _require(
        profile in PUBLIC_INITIALIZATION_PROFILES,
        f"project initialization profile must be one of {PUBLIC_INITIALIZATION_PROFILES}; "
        "balanced and canonical CPU projects are historical read/inspect/verify-only",
    )
    absolute = output.absolute()
    parent = absolute.parent.resolve(strict=True)
    output = parent / absolute.name
    _require(output == absolute and output.name not in {"", ".", ".."}, "project output contains a symlinked or unsafe component")
    portable_basename(output.name, "project output name")
    _require(len(f".{output.name}.project-init.{uuid.uuid4()}".encode("ascii")) <= 255, "project output name is too long for staging")
    _require(not output.exists() and not output.is_symlink(), f"project output already exists: {output}")
    _require(output.parent.exists() and output.parent.is_dir() and not output.parent.is_symlink(), "project output parent is unsafe")
    staging = parent / f".{output.name}.project-init.{uuid.uuid4()}"
    staging.mkdir(mode=0o700)
    fsync_directory(parent)
    for relative in ("data/images", "data/inference_images", "annotations", "assets", "review"):
        staging.joinpath(*relative.split("/")).mkdir(parents=True, exist_ok=False)
    for directory in sorted((path for path in staging.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
        directory.chmod(0o755)
        fsync_directory(directory)
    project_toml = {
        CANONICAL_GPU_PROFILE: CANONICAL_GPU_PROJECT_TOML,
        EFFICIENT_GPU_PROFILE: EFFICIENT_GPU_PROJECT_TOML,
        FULL_IMAGE_GPU_PROFILE: FULL_IMAGE_GPU_PROJECT_TOML,
    }[profile]
    next_command = "python -I -B -m compag_curation inspect-data --config config.toml"
    write_new_bytes(staging / "config.toml", project_toml.encode("ascii"))
    write_new_json(
        staging / "PROJECT_MANIFEST.json",
        {
            "schema": "compag-curation-project-template/v1",
            "status": "READY_FOR_USER_DATA",
            "config": "config.toml",
            "profile": profile,
            "input_policy": "READ_ONLY_NO_IN_PLACE_CHANGES",
            "next_command": next_command,
        },
    )
    staging.chmod(0o755)
    fsync_directory(staging)
    publish_directory_noreplace(staging, output)
    return {"status": "PASS", "output_name": output.name, "config": "config.toml", "profile": profile}
