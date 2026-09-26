"""Frozen scientific constants for the canonical ``xgb_recall`` profile.

This module is deliberately import-inert.  Numerical and imaging libraries
are imported only by the execution helpers that need them.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any


CANONICAL_CPU_PROFILE = "canonical-xgb-recall-cpu-v1"
CANONICAL_GPU_PROFILE = "canonical-xgb-recall-gpu-v1"
EFFICIENT_GPU_PROFILE = "efficient-xgb-recall-gpu-v1"
FULL_IMAGE_GPU_PROFILE = "full-image-multiscale-xgb-recall-gpu-v1"
CANONICAL_PROFILES = (CANONICAL_CPU_PROFILE, CANONICAL_GPU_PROFILE)
# The efficient and full-image profiles reuse the v2 feature/training shape,
# but remain scientifically distinct variants.  Neither may be added to
# ``CANONICAL_PROFILES``.
V2_PIPELINE_PROFILES = (
    *CANONICAL_PROFILES,
    EFFICIENT_GPU_PROFILE,
    FULL_IMAGE_GPU_PROFILE,
)
GPU_EXECUTION_PROFILES = (
    CANONICAL_GPU_PROFILE,
    EFFICIENT_GPU_PROFILE,
    FULL_IMAGE_GPU_PROFILE,
)
# Backward-compatible name retained for v1.2 callers and serialized CPU
# artifacts.  New code that accepts either execution backend should use
# ``CANONICAL_PROFILES`` instead.
CANONICAL_PROFILE = CANONICAL_CPU_PROFILE
CANONICAL_TILE_SIZE = 512
CANONICAL_TILE_STRIDE = 512
CANONICAL_TILE_EXTENSION = ".jpg"
CANONICAL_JPEG_QUALITY = 95
CANONICAL_FEATURE_CROP_SCALES = (0.67, 0.80, 1.00, 1.25)
CANONICAL_PROPOSAL_SCALES = (1.0,)
CANONICAL_PROPOSAL_IDENTITY = "compag-canonical-proposal-v2"
EFFICIENT_PROPOSAL_IDENTITY = "compag-efficient-sam2-tiny-proposal-v1"
FULL_IMAGE_PROPOSAL_IDENTITY = "compag-full-image-multiscale-proposal-v1"
CANONICAL_EMBEDDING_BACKBONE = "resnet50"
CANONICAL_EMBEDDING_DIMENSIONS = 2048
CANONICAL_EMBEDDING_PAD_FRACTION = 0.10
CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256 = (
    "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
)
CANONICAL_SAM2_HIERA_L_CONFIG_SHA256 = (
    "1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107"
)
EFFICIENT_SAM2_HIERA_T_CHECKPOINT_SHA256 = (
    "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
)
EFFICIENT_SAM2_HIERA_T_CONFIG_SHA256 = (
    "f932eac1c6241e910031b2f000a81cd9f8a8d4896e2277ab5ffb721f378b188d"
)
CANONICAL_PCA_DIMENSIONS = 32
CANONICAL_TEST_FRACTION = 0.20
CANONICAL_RANDOM_STATE = 42
CANONICAL_GROUP_FOLDS = 5
CANONICAL_SEARCH_ITERATIONS = 30
CANONICAL_REFIT_ESTIMATORS = 2000
CANONICAL_EARLY_STOPPING_ROUNDS = 30
CANONICAL_EARLY_STOP_VALIDATION_FRACTION = 0.30
CANONICAL_DECISION_THRESHOLD = 0.50
CANONICAL_FULL_IMAGE_NMS_IOU = 0.50
# Execution-only SAM2 prompt microbatch used by Stage 20 proposal generation.
# The canonical method remains the 64-point lattice with points_per_batch=512;
# this value only bounds the number of prompt samples resident on CUDA at once.
CANONICAL_STAGE20_AMG_POINTS_PER_BATCH = 32
# Execution-only SAM2 prompt microbatch used by fresh bundle inference.  The
# canonical 64-point lattice and its Stage-20 method identity remain unchanged;
# only the number of prompt samples resident on CUDA at once is reduced.
CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH = 32
# The efficient profile evaluates the same configured 64-point lattice with a
# smaller execution-only prompt microbatch.  Its Hiera Tiny backbone means
# scientific equivalence to the canonical profile is not claimed.
EFFICIENT_STAGE20_AMG_POINTS_PER_BATCH = 16
EFFICIENT_INFERENCE_AMG_POINTS_PER_BATCH = 16
# The full-image Hiera-L operating point keeps the configured 512 prompt batch
# as method metadata while retaining a separately bounded execution microbatch.
# This initial bound matches the reviewed Hiera-L CUDA execution limit; it does
# not imply canonical-method equivalence for the full-image profile.
FULL_IMAGE_STAGE20_AMG_POINTS_PER_BATCH = 8
FULL_IMAGE_INFERENCE_AMG_POINTS_PER_BATCH = 8
# SAM2 confidence values are archival evidence rather than predictor features.
# Store them at a stable decimal precision after raw-score selection completes.
CANONICAL_ARCHIVED_SAM_SCORE_DECIMALS = 6


_PCA_FEATURE_ORDER = tuple(sorted(f"embed_pca_{index}" for index in range(32)))

# Exact fitted r92 predictor order recovered from the preserved feature
# manifest.  The PCA names intentionally retain lexicographic order.
CANONICAL_FEATURE_ORDER = (
    "GLCM_contrast",
    "GLCM_homogeneity",
    "LBP_u5",
    "area_norm",
    "aspect_ratio",
    "bbox_diag_frac",
    "bbox_h",
    "bbox_w",
    "bbox_x",
    "bbox_y",
    "circularity",
    "cx",
    "cy",
    "delta_a",
    "delta_a_med",
    "delta_b",
    "delta_b_med",
    "eccentricity",
    "elongation",
    *_PCA_FEATURE_ORDER,
    "embed_sim",
    "extent",
    "g_border",
    "g_color",
    "g_embed",
    "g_light",
    "g_maha",
    "g_quality",
    "g_robust",
    "g_shape",
    "grad_mean",
    "grad_p90",
    "grid_c",
    "grid_c_norm",
    "grid_r",
    "grid_r_norm",
    "histab_q11",
    "histab_q12",
    "histab_q21",
    "histab_q22",
    "histb_q1",
    "histb_q2",
    "histb_q3",
    "histb_q4",
    "mean_L",
    "mean_a",
    "mean_b",
    "median_L",
    "median_a",
    "median_a_in",
    "median_a_ring",
    "median_b",
    "median_b_in",
    "median_b_ring",
    "perim_over_sqrt_area",
    "perimeter",
    "scale_diag",
    "solidity",
    "std_L",
    "std_a",
    "std_b",
    "touching_border",
)

# Digest of the canonical one-name-per-line representation, including its
# terminal LF.  This matches ``resources/canonical_feature_order.txt``.
CANONICAL_FEATURE_ORDER_SHA256 = (
    "a5478eb2e571809d80fbdf0e132d39f301d936b0b711ceb08a326ade0b24ed47"
)


def canonical_feature_order_bytes() -> bytes:
    return ("\n".join(CANONICAL_FEATURE_ORDER) + "\n").encode("ascii")


def stage20_amg_execution_points_per_batch(profile: str) -> int:
    """Return the sealed Stage-20 CUDA microbatch for an executable profile."""

    try:
        return {
            CANONICAL_GPU_PROFILE: CANONICAL_STAGE20_AMG_POINTS_PER_BATCH,
            EFFICIENT_GPU_PROFILE: EFFICIENT_STAGE20_AMG_POINTS_PER_BATCH,
            FULL_IMAGE_GPU_PROFILE: FULL_IMAGE_STAGE20_AMG_POINTS_PER_BATCH,
        }[profile]
    except KeyError as exc:
        raise ValueError("unsupported GPU execution profile") from exc


def inference_amg_execution_points_per_batch(profile: str) -> int:
    """Return the sealed inference CUDA microbatch for an executable profile."""

    try:
        return {
            CANONICAL_GPU_PROFILE: CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
            EFFICIENT_GPU_PROFILE: EFFICIENT_INFERENCE_AMG_POINTS_PER_BATCH,
            FULL_IMAGE_GPU_PROFILE: FULL_IMAGE_INFERENCE_AMG_POINTS_PER_BATCH,
        }[profile]
    except KeyError as exc:
        raise ValueError("unsupported GPU execution profile") from exc


def proposal_identity_for_profile(profile: str) -> str:
    """Return the hash-domain identity sealed to a v2 proposal profile."""

    if profile in CANONICAL_PROFILES:
        return CANONICAL_PROPOSAL_IDENTITY
    if profile == EFFICIENT_GPU_PROFILE:
        return EFFICIENT_PROPOSAL_IDENTITY
    if profile == FULL_IMAGE_GPU_PROFILE:
        return FULL_IMAGE_PROPOSAL_IDENTITY
    raise ValueError("unsupported v2 proposal profile")


if len(CANONICAL_FEATURE_ORDER) != 93:
    raise RuntimeError("canonical predictor schema must contain exactly 93 names")
if hashlib.sha256(canonical_feature_order_bytes()).hexdigest() != CANONICAL_FEATURE_ORDER_SHA256:
    raise RuntimeError("canonical predictor schema identity changed")


@dataclass(frozen=True)
class CanonicalAMGSettings:
    """Exact recovered single-scale Hiera-L automatic-mask settings."""

    architecture: str = "sam2.1_hiera_large"
    config_locator: str = "configs/sam2.1/sam2.1_hiera_l"
    points_per_side: int = 64
    points_per_batch: int = 512
    pred_iou_thresh: float = 0.80
    stability_score_thresh: float = 0.88
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 2
    crop_overlap_ratio: float = 0.40
    exclude_largest: bool = False
    proposal_scales: tuple[float, ...] = CANONICAL_PROPOSAL_SCALES
    merge_iou_threshold: float = 0.75
    max_masks_per_tile: int = 500
    yolo_enabled: bool = False

    def __post_init__(self) -> None:
        expected: dict[str, Any] = {
            "architecture": "sam2.1_hiera_large",
            "config_locator": "configs/sam2.1/sam2.1_hiera_l",
            "points_per_side": 64,
            "points_per_batch": 512,
            "pred_iou_thresh": 0.80,
            "stability_score_thresh": 0.88,
            "crop_n_layers": 0,
            "crop_n_points_downscale_factor": 2,
            "crop_overlap_ratio": 0.40,
            "exclude_largest": False,
            "proposal_scales": (1.0,),
            "merge_iou_threshold": 0.75,
            "max_masks_per_tile": 500,
            "yolo_enabled": False,
        }
        observed = asdict(self)
        if any(observed[name] != value for name, value in expected.items()):
            raise ValueError("canonical AMG settings are immutable")

    def generator_kwargs(self) -> dict[str, int | float]:
        """Return only arguments accepted by SAM2AutomaticMaskGenerator."""

        return {
            "points_per_side": self.points_per_side,
            "points_per_batch": self.points_per_batch,
            "pred_iou_thresh": self.pred_iou_thresh,
            "stability_score_thresh": self.stability_score_thresh,
            "crop_n_layers": self.crop_n_layers,
            "crop_n_points_downscale_factor": self.crop_n_points_downscale_factor,
            "crop_overlap_ratio": self.crop_overlap_ratio,
        }

    def provenance_record(
        self,
        *,
        profile: str = CANONICAL_PROFILE,
    ) -> dict[str, Any]:
        if profile not in CANONICAL_PROFILES:
            raise ValueError("canonical AMG provenance profile is unsupported")
        return {
            "profile": profile,
            **asdict(self),
            "proposal_scales": list(self.proposal_scales),
            "proposal_policy": "amg_only",
            "proposal_identity": CANONICAL_PROPOSAL_IDENTITY,
            "feature_mode": "ultra",
            "feature_crop_scales": list(CANONICAL_FEATURE_CROP_SCALES),
            "decision_threshold": CANONICAL_DECISION_THRESHOLD,
            "full_image_nms_iou": CANONICAL_FULL_IMAGE_NMS_IOU,
        }


@dataclass(frozen=True)
class EfficientAMGSettings(CanonicalAMGSettings):
    """Noncanonical low-memory SAM2 Tiny variant of the v2 AMG contract."""

    architecture: str = "sam2.1_hiera_tiny"
    config_locator: str = "configs/sam2.1/sam2.1_hiera_t"

    def __post_init__(self) -> None:
        expected: dict[str, Any] = {
            "architecture": "sam2.1_hiera_tiny",
            "config_locator": "configs/sam2.1/sam2.1_hiera_t",
            "points_per_side": 64,
            "points_per_batch": 512,
            "pred_iou_thresh": 0.80,
            "stability_score_thresh": 0.88,
            "crop_n_layers": 0,
            "crop_n_points_downscale_factor": 2,
            "crop_overlap_ratio": 0.40,
            "exclude_largest": False,
            "proposal_scales": (1.0,),
            "merge_iou_threshold": 0.75,
            "max_masks_per_tile": 500,
            "yolo_enabled": False,
        }
        observed = asdict(self)
        if any(observed[name] != value for name, value in expected.items()):
            raise ValueError("efficient AMG settings are immutable")

    def provenance_record(
        self,
        *,
        profile: str = EFFICIENT_GPU_PROFILE,
    ) -> dict[str, Any]:
        if profile != EFFICIENT_GPU_PROFILE:
            raise ValueError("efficient AMG provenance profile is unsupported")
        return {
            "profile": profile,
            **asdict(self),
            "proposal_scales": list(self.proposal_scales),
            "proposal_policy": "amg_only",
            "proposal_identity": EFFICIENT_PROPOSAL_IDENTITY,
            "method_classification": "EFFICIENT_SAM2_TINY_NON_EQUIVALENT_VARIANT",
            "canonical_method_equivalence": "NOT_CLAIMED",
            "feature_mode": "ultra",
            "feature_crop_scales": list(CANONICAL_FEATURE_CROP_SCALES),
            "decision_threshold": CANONICAL_DECISION_THRESHOLD,
            "full_image_nms_iou": CANONICAL_FULL_IMAGE_NMS_IOU,
        }


@dataclass(frozen=True)
class FullImageMultiscaleAMGSettings:
    """Initial noncanonical full-image, multiscale Hiera-L operating point."""

    architecture: str = "sam2.1_hiera_large"
    config_locator: str = "configs/sam2.1/sam2.1_hiera_l"
    points_per_side: int = 64
    points_per_batch: int = 512
    execution_points_per_batch: int = FULL_IMAGE_STAGE20_AMG_POINTS_PER_BATCH
    pred_iou_thresh: float = 0.80
    stability_score_thresh: float = 0.88
    crop_n_layers: int = 2
    crop_n_points_downscale_factor: int = 1
    crop_overlap_ratio: float = 512 / 1500
    exclude_largest: bool = False
    proposal_scales: tuple[float, ...] = (1.0,)
    merge_iou_threshold: float = 0.75
    max_masks_per_image: int = 1000
    yolo_enabled: bool = False

    def __post_init__(self) -> None:
        expected: dict[str, Any] = {
            "architecture": "sam2.1_hiera_large",
            "config_locator": "configs/sam2.1/sam2.1_hiera_l",
            "points_per_side": 64,
            "points_per_batch": 512,
            "execution_points_per_batch": 8,
            "pred_iou_thresh": 0.80,
            "stability_score_thresh": 0.88,
            "crop_n_layers": 2,
            "crop_n_points_downscale_factor": 1,
            "crop_overlap_ratio": 512 / 1500,
            "exclude_largest": False,
            "proposal_scales": (1.0,),
            "merge_iou_threshold": 0.75,
            "max_masks_per_image": 1000,
            "yolo_enabled": False,
        }
        observed = asdict(self)
        if any(observed[name] != value for name, value in expected.items()):
            raise ValueError("full-image multiscale AMG settings are immutable")

    @property
    def max_masks_per_tile(self) -> int:
        """Compatibility accessor while callers migrate to the image cap."""

        return self.max_masks_per_image

    def generator_kwargs(self) -> dict[str, int | float]:
        """Return only arguments accepted by SAM2AutomaticMaskGenerator."""

        return {
            "points_per_side": self.points_per_side,
            "points_per_batch": self.points_per_batch,
            "pred_iou_thresh": self.pred_iou_thresh,
            "stability_score_thresh": self.stability_score_thresh,
            "crop_n_layers": self.crop_n_layers,
            "crop_n_points_downscale_factor": self.crop_n_points_downscale_factor,
            "crop_overlap_ratio": self.crop_overlap_ratio,
        }

    def provenance_record(
        self,
        *,
        profile: str = FULL_IMAGE_GPU_PROFILE,
    ) -> dict[str, Any]:
        if profile != FULL_IMAGE_GPU_PROFILE:
            raise ValueError("full-image multiscale AMG provenance profile is unsupported")
        return {
            "profile": profile,
            **asdict(self),
            "proposal_scales": list(self.proposal_scales),
            "proposal_policy": "full_image_multiscale_amg_only",
            "proposal_identity": FULL_IMAGE_PROPOSAL_IDENTITY,
            "spatial_mode": "full-image-multiscale",
            "stability_score_offset": 1.0,
            "mask_threshold": 0.0,
            "box_nms_thresh": 0.7,
            "crop_nms_thresh": 0.7,
            "min_mask_region_area": 0,
            "sam2_output_mode": "uncompressed_rle",
            "use_m2m": False,
            "multimask_output": True,
            "connected_component_policy": "largest-8-connected-pixel-area-top-left-tie-v1",
            "method_classification": "FULL_IMAGE_MULTISCALE_HIERA_L_EXPERIMENTAL_VARIANT",
            "canonical_method_equivalence": "NOT_CLAIMED",
            "feature_mode": "ultra",
            "feature_crop_scales": list(CANONICAL_FEATURE_CROP_SCALES),
            "decision_threshold": CANONICAL_DECISION_THRESHOLD,
            "full_image_nms_iou": CANONICAL_FULL_IMAGE_NMS_IOU,
        }


def amg_settings_for_profile(
    profile: str,
) -> CanonicalAMGSettings | EfficientAMGSettings | FullImageMultiscaleAMGSettings:
    """Return the immutable backbone/settings contract for one v2 profile."""

    if profile in CANONICAL_PROFILES:
        return CanonicalAMGSettings()
    if profile == EFFICIENT_GPU_PROFILE:
        return EfficientAMGSettings()
    if profile == FULL_IMAGE_GPU_PROFILE:
        return FullImageMultiscaleAMGSettings()
    raise ValueError("unsupported v2 pipeline profile")


__all__ = [
    "CANONICAL_ARCHIVED_SAM_SCORE_DECIMALS",
    "CANONICAL_DECISION_THRESHOLD",
    "CANONICAL_EARLY_STOPPING_ROUNDS",
    "CANONICAL_EARLY_STOP_VALIDATION_FRACTION",
    "CANONICAL_EMBEDDING_BACKBONE",
    "CANONICAL_EMBEDDING_DIMENSIONS",
    "CANONICAL_EMBEDDING_PAD_FRACTION",
    "CANONICAL_FEATURE_CROP_SCALES",
    "CANONICAL_FEATURE_ORDER",
    "CANONICAL_FEATURE_ORDER_SHA256",
    "CANONICAL_FULL_IMAGE_NMS_IOU",
    "CANONICAL_GROUP_FOLDS",
    "CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH",
    "CANONICAL_JPEG_QUALITY",
    "CANONICAL_PCA_DIMENSIONS",
    "CANONICAL_CPU_PROFILE",
    "CANONICAL_GPU_PROFILE",
    "CANONICAL_PROFILE",
    "CANONICAL_PROFILES",
    "CANONICAL_PROPOSAL_SCALES",
    "CANONICAL_PROPOSAL_IDENTITY",
    "CANONICAL_RANDOM_STATE",
    "CANONICAL_REFIT_ESTIMATORS",
    "CANONICAL_SEARCH_ITERATIONS",
    "CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256",
    "CANONICAL_SAM2_HIERA_L_CONFIG_SHA256",
    "CANONICAL_STAGE20_AMG_POINTS_PER_BATCH",
    "CANONICAL_TEST_FRACTION",
    "CANONICAL_TILE_EXTENSION",
    "CANONICAL_TILE_SIZE",
    "CANONICAL_TILE_STRIDE",
    "EFFICIENT_GPU_PROFILE",
    "EFFICIENT_INFERENCE_AMG_POINTS_PER_BATCH",
    "EFFICIENT_PROPOSAL_IDENTITY",
    "EFFICIENT_SAM2_HIERA_T_CHECKPOINT_SHA256",
    "EFFICIENT_SAM2_HIERA_T_CONFIG_SHA256",
    "EFFICIENT_STAGE20_AMG_POINTS_PER_BATCH",
    "FULL_IMAGE_GPU_PROFILE",
    "FULL_IMAGE_INFERENCE_AMG_POINTS_PER_BATCH",
    "FULL_IMAGE_PROPOSAL_IDENTITY",
    "FULL_IMAGE_STAGE20_AMG_POINTS_PER_BATCH",
    "GPU_EXECUTION_PROFILES",
    "V2_PIPELINE_PROFILES",
    "CanonicalAMGSettings",
    "EfficientAMGSettings",
    "FullImageMultiscaleAMGSettings",
    "amg_settings_for_profile",
    "canonical_feature_order_bytes",
    "inference_amg_execution_points_per_batch",
    "proposal_identity_for_profile",
    "stage20_amg_execution_points_per_batch",
]
