"""Portable, hash-closed model bundle for fresh-process inference."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import uuid
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping, Sequence

from .canonical.spec import (
    CANONICAL_CPU_PROFILE,
    CANONICAL_DECISION_THRESHOLD,
    CANONICAL_EMBEDDING_DIMENSIONS,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_GPU_PROFILE,
    CANONICAL_PCA_DIMENSIONS,
    EFFICIENT_GPU_PROFILE,
    FULL_IMAGE_GPU_PROFILE,
    GPU_EXECUTION_PROFILES,
    V2_PIPELINE_PROFILES,
    amg_settings_for_profile,
)
from .canonical.full_image_features import (
    FULL_IMAGE_CONTEXT_FRACTION,
    FULL_IMAGE_CONTEXT_MINIMUM_PIXELS,
    FULL_IMAGE_FEATURE_PROPOSAL_BATCH,
    FULL_IMAGE_FEATURE_SEMANTICS_ID,
)
from .canonical.full_image import (
    FULL_IMAGE_COMPONENT_POLICY,
    FULL_IMAGE_DURABLE_MASK_ENCODING,
)
from .canonical.preprocessing import FULL_IMAGE_UNIT_KIND
from .public_config import (
    FEATURE_ORDER_SHA256,
    FULL_IMAGE_SPATIAL_MODE,
    PUBLIC_FEATURE_ORDER,
)
from .public_config import (
    RESNET50_WEIGHTS_SHA256,
    SAM2_LARGE_CHECKPOINT_SHA256,
    SAM2_LARGE_CONFIG_SHA256,
    SAM2_TINY_CHECKPOINT_SHA256,
    SAM2_TINY_CONFIG_SHA256,
)
from .public_io import (
    PublicIOError,
    canonical_json_value,
    canonical_json_bytes,
    copy_new,
    fsync_directory,
    hash_file_snapshot,
    manifest_rows,
    portable_basename,
    publish_directory_noreplace,
    sha256_file,
    stable_file,
    strict_json_bytes,
    tree_identity_rows,
    write_new_bytes,
    write_new_json,
)
from .runtime_lock import (
    SCIENCE_CPU_LOCK_SHA256,
    SCIENCE_CPU_PYTHON,
    SCIENCE_CPU_VERSIONS,
    SCIENCE_GPU_LOCK_SHA256,
    SCIENCE_GPU_PYTHON,
    SCIENCE_GPU_VERSIONS,
    package_implementation_identity,
    science_cpu_lock_identity,
    science_cpu_runtime_policy,
    science_gpu_lock_identity,
    science_gpu_runtime_policy,
)


BUNDLE_SCHEMA = "compag-curation-model-bundle/v1"
BUNDLE_SCHEMA_V2 = "compag-curation-model-bundle/v2"
MAX_BUNDLE_JSON_BYTES = 8 * 1024 * 1024
MAX_CLASSIFIER_BYTES = 512 * 1024 * 1024
# The textual XGBoost projection is used only to inspect decoded strings.  Keep
# its parse bounded independently: JSON numeric arrays can be larger than the
# UBJ typed arrays from which XGBoost renders them.
MAX_CLASSIFIER_JSON_BYTES = MAX_CLASSIFIER_BYTES
MAX_SAM_CHECKPOINT_BYTES = 1024 * 1024 * 1024
MAX_RESNET50_WEIGHTS_BYTES = 256 * 1024 * 1024
MAX_NPY_STATE_BYTES = 2 * 1024 * 1024
MAX_NPY_HEADER_BYTES = 4096
MAX_FEATURE_COUNT = 4096
SAM2_LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
RESNET50_LICENSE_SHA256 = "6502f676851cfe25f8af75531dfb32375b7325b73c37e7b43741fa422893e71d"
FIXED_CANONICAL_METHOD = "FIXED_CANONICAL_METHOD"
FEATURE_ORDER_HASH_ENCODING_V2 = "ascii-lines-terminal-lf/v1"
PUBLIC_BUNDLE_PROFILE = "public-safe-balanced-v1"
POST_R92_TRANSFER_WORKFLOW = "SINGLE_NEW_IMAGE_POST_R92_TRANSFER_V1"
POST_R92_TRANSFER_LINEAGE_POLICY = "POST_R92_REVIEWED_TRANSFER_BASELINE"
POST_R92_TRANSFER_REPRODUCTION_CLAIM = "NOT_R92_REPRODUCTION"
POST_R92_TRANSFER_FEATURE_STATE_SCOPE = "FROZEN_PUBLISHED_R92_TRANSFER_STATE"
POST_R92_TRANSFER_FROZEN_STATE_POLICY = (
    "FROZEN_PUBLISHED_R92_PCA_AND_PROTOTYPE"
)
POST_R92_TRANSFER_CV_INTERPRETATION = (
    "CONDITIONAL_ON_FROZEN_R92_TRANSFER_PROTOTYPE_AND_PCA"
)
POST_R92_TRANSFER_PCA_VARIANCE_POLICY = (
    "OBSERVED_TRANSFER_TRAIN_PROJECTION_VARIANCE_NOT_R92_FIT_VARIANCE"
)
HISTORICAL_V1_2_PACKAGE_IMPLEMENTATION_IDENTITY = {
    "schema": "compag-curation-package-implementation-identity/v1",
    "sha256": "e67ea22b13d2089e819b90a70387c529c3097d5d09171e7d6a2f520f6ec9f3cb",
    "file_count": 101,
    "size_bytes": 2_432_081,
}


def _current_package_implementation_identity(
    *,
    expected_distribution_version: str = SCIENCE_CPU_VERSIONS["compag-curation"],
) -> dict[str, object]:
    try:
        return package_implementation_identity(
            expected_distribution_version=expected_distribution_version,
        )
    except RuntimeError as exc:
        raise PublicIOError("package implementation identity could not be established") from exc


def _expected_package_implementation_identity(
    profile: str,
    *,
    expected_distribution_version: str,
) -> dict[str, object]:
    if profile == CANONICAL_CPU_PROFILE:
        # CPU v2 bundles are historical v1.2 evidence.  Their verifier must
        # compare against the frozen published package closure, not whatever
        # current source tree happens to perform the inspection.
        return dict(HISTORICAL_V1_2_PACKAGE_IMPLEMENTATION_IDENTITY)
    return _current_package_implementation_identity(
        expected_distribution_version=expected_distribution_version,
    )


BUNDLE_ASSET_PROFILES = {
    PUBLIC_BUNDLE_PROFILE: {
        "locator": "configs/sam2.1/sam2.1_hiera_t",
        "checkpoint_sha256": SAM2_TINY_CHECKPOINT_SHA256,
        "config_sha256": SAM2_TINY_CONFIG_SHA256,
    },
}
COMPATIBILITY_PACKAGES = frozenset(SCIENCE_CPU_VERSIONS)
BUNDLE_FILES = (
    "classifier.ubj",
    "imputer.json",
    "feature_schema.json",
    "threshold.json",
    "class_map.json",
    "normalized_config.json",
    "preprocessing.json",
    "proposal_config.json",
    "compatibility.json",
    "provenance.json",
    "sam2/configs/sam2.1/model.yaml",
    "sam2/checkpoints/model.pt",
    "licenses/SAM2-APACHE-2.0.txt",
)
BUNDLE_SIZE_LIMITS = {
    **{path: MAX_BUNDLE_JSON_BYTES for path in BUNDLE_FILES},
    "bundle.json": MAX_BUNDLE_JSON_BYTES,
    "classifier.ubj": MAX_CLASSIFIER_BYTES,
    "sam2/checkpoints/model.pt": MAX_SAM_CHECKPOINT_BYTES,
    "sam2/configs/sam2.1/model.yaml": 1024 * 1024,
    "licenses/SAM2-APACHE-2.0.txt": 1024 * 1024,
}
BUNDLE_MAX_TOTAL_BYTES = sum(BUNDLE_SIZE_LIMITS.values())

_CANONICAL_V2_ASSET_PROFILE = {
    "sam2_architecture": "sam2.1_hiera_large",
    "sam2_config_locator": "configs/sam2.1/sam2.1_hiera_l",
    "sam2_config_bundle_path": "sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2_checkpoint_bundle_path": "sam2/checkpoints/sam2.1_hiera_large.pt",
    "sam2_checkpoint_sha256": SAM2_LARGE_CHECKPOINT_SHA256,
    "sam2_config_sha256": SAM2_LARGE_CONFIG_SHA256,
    "sam2_license_sha256": SAM2_LICENSE_SHA256,
    "resnet50_architecture": "torchvision.models.resnet50",
    "resnet50_weights_identity": "ResNet50_Weights.IMAGENET1K_V2",
    "resnet50_weights_sha256": RESNET50_WEIGHTS_SHA256,
    "resnet50_license_sha256": RESNET50_LICENSE_SHA256,
}
_EFFICIENT_V2_ASSET_PROFILE = {
    **_CANONICAL_V2_ASSET_PROFILE,
    "sam2_architecture": "sam2.1_hiera_tiny",
    "sam2_config_locator": "configs/sam2.1/sam2.1_hiera_t",
    "sam2_config_bundle_path": "sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2_checkpoint_bundle_path": "sam2/checkpoints/sam2.1_hiera_tiny.pt",
    "sam2_checkpoint_sha256": SAM2_TINY_CHECKPOINT_SHA256,
    "sam2_config_sha256": SAM2_TINY_CONFIG_SHA256,
}
BUNDLE_V2_ASSET_PROFILES = {
    CANONICAL_CPU_PROFILE: dict(_CANONICAL_V2_ASSET_PROFILE),
    CANONICAL_GPU_PROFILE: dict(_CANONICAL_V2_ASSET_PROFILE),
    EFFICIENT_GPU_PROFILE: dict(_EFFICIENT_V2_ASSET_PROFILE),
    FULL_IMAGE_GPU_PROFILE: dict(_CANONICAL_V2_ASSET_PROFILE),
}
# Frozen per-profile SAM2 member names.  These are selected from the sealed
# profile identity, never inferred from a caller-supplied filename.
_BUNDLE_V2_SAM_PATHS = {
    CANONICAL_CPU_PROFILE: (
        "sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2/checkpoints/sam2.1_hiera_large.pt",
    ),
    CANONICAL_GPU_PROFILE: (
        "sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2/checkpoints/sam2.1_hiera_large.pt",
    ),
    FULL_IMAGE_GPU_PROFILE: (
        "sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2/checkpoints/sam2.1_hiera_large.pt",
    ),
    EFFICIENT_GPU_PROFILE: (
        "sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2/checkpoints/sam2.1_hiera_tiny.pt",
    ),
}
BUNDLE_V2_FILES = (
    "classifier.ubj",
    "imputer.json",
    "feature_schema.json",
    "prototype.npy",
    "pca32_mean.npy",
    "pca32_components.npy",
    "pca32_explained_variance.npy",
    "threshold.json",
    "class_map.json",
    "normalized_config.json",
    "preprocessing.json",
    "proposal_config.json",
    "feature_config.json",
    "training_config.json",
    "compatibility.json",
    "provenance.json",
    "sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2/checkpoints/sam2.1_hiera_large.pt",
    "resnet50/resnet50-imagenet1k-v2.pth",
    "licenses/SAM2-APACHE-2.0.txt",
    "licenses/TORCHVISION-BSD-3-CLAUSE.txt",
)
# ``BUNDLE_V2_FILES`` remains the historical Full/CPU closure for callers that
# imported it before the efficient profile existed.  New validation always
# uses the exact closure selected by the bundle profile.
_BUNDLE_V2_COMMON_FILES = tuple(
    relative
    for relative in BUNDLE_V2_FILES
    if relative
    not in {
        "sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2/checkpoints/sam2.1_hiera_large.pt",
    }
)
BUNDLE_V2_FILES_BY_PROFILE = {
    CANONICAL_CPU_PROFILE: BUNDLE_V2_FILES,
    CANONICAL_GPU_PROFILE: BUNDLE_V2_FILES,
    FULL_IMAGE_GPU_PROFILE: BUNDLE_V2_FILES,
    EFFICIENT_GPU_PROFILE: (
        *_BUNDLE_V2_COMMON_FILES,
        "sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2/checkpoints/sam2.1_hiera_tiny.pt",
    ),
}
BUNDLE_V2_SIZE_LIMITS = {
    **{path: MAX_BUNDLE_JSON_BYTES for path in BUNDLE_V2_FILES},
    "bundle.json": MAX_BUNDLE_JSON_BYTES,
    "classifier.ubj": MAX_CLASSIFIER_BYTES,
    "prototype.npy": MAX_NPY_STATE_BYTES,
    "pca32_mean.npy": MAX_NPY_STATE_BYTES,
    "pca32_components.npy": MAX_NPY_STATE_BYTES,
    "pca32_explained_variance.npy": MAX_NPY_STATE_BYTES,
    "sam2/checkpoints/sam2.1_hiera_large.pt": MAX_SAM_CHECKPOINT_BYTES,
    "sam2/configs/sam2.1/sam2.1_hiera_l.yaml": 1024 * 1024,
    "resnet50/resnet50-imagenet1k-v2.pth": MAX_RESNET50_WEIGHTS_BYTES,
    "licenses/SAM2-APACHE-2.0.txt": 1024 * 1024,
    "licenses/TORCHVISION-BSD-3-CLAUSE.txt": 1024 * 1024,
}
BUNDLE_V2_MAX_TOTAL_BYTES = sum(BUNDLE_V2_SIZE_LIMITS.values())
BUNDLE_V2_SIZE_LIMITS_BY_PROFILE = {
    CANONICAL_CPU_PROFILE: BUNDLE_V2_SIZE_LIMITS,
    CANONICAL_GPU_PROFILE: BUNDLE_V2_SIZE_LIMITS,
    FULL_IMAGE_GPU_PROFILE: BUNDLE_V2_SIZE_LIMITS,
    EFFICIENT_GPU_PROFILE: {
        **{
            path: BUNDLE_V2_SIZE_LIMITS[path]
            for path in _BUNDLE_V2_COMMON_FILES
        },
        "bundle.json": MAX_BUNDLE_JSON_BYTES,
        "sam2/checkpoints/sam2.1_hiera_tiny.pt": MAX_SAM_CHECKPOINT_BYTES,
        "sam2/configs/sam2.1/sam2.1_hiera_t.yaml": 1024 * 1024,
    },
}
BUNDLE_V2_MAX_TOTAL_BYTES_BY_PROFILE = {
    profile: sum(limits.values())
    for profile, limits in BUNDLE_V2_SIZE_LIMITS_BY_PROFILE.items()
}

_BUNDLE_V2_ASSET_FIELDS = frozenset(
    {
        "sam2_architecture",
        "sam2_config_locator",
        "sam2_checkpoint_sha256",
        "sam2_config_sha256",
        "sam2_license_sha256",
        "resnet50_architecture",
        "resnet50_weights_identity",
        "resnet50_weights_sha256",
        "resnet50_license_sha256",
    }
)
_BUNDLE_V2_ASSET_PATH_FIELDS = frozenset(
    {"sam2_config_bundle_path", "sam2_checkpoint_bundle_path"}
)


@dataclass(frozen=True)
class _CanonicalRuntimeContract:
    python: str
    versions: Mapping[str, str]
    lock_sha256: str
    policy_field: str
    policy_sha256: str
    distribution_version: str


@dataclass(frozen=True)
class BundleWriteRequest:
    output: Path
    classifier_ubj: bytes
    imputer_statistics: Mapping[str, float]
    threshold: float
    normalized_config: Mapping[str, object]
    preprocessing: Mapping[str, object]
    proposal_config: Mapping[str, object]
    compatibility: Mapping[str, object]
    provenance: Mapping[str, object]
    sam2_config: Path
    sam2_checkpoint: Path
    sam2_license: Path


@dataclass(frozen=True)
class BundleV2WriteRequest:
    output: Path
    profile: str
    classifier_ubj: bytes
    feature_order: Sequence[str]
    feature_order_sha256: str
    imputer_statistics: Mapping[str, float]
    prototype_npy: bytes
    pca_mean_npy: bytes
    pca_components_npy: bytes
    pca_explained_variance: Sequence[float]
    normalized_config: Mapping[str, object]
    preprocessing: Mapping[str, object]
    proposal_config: Mapping[str, object]
    feature_config: Mapping[str, object]
    training_config: Mapping[str, object]
    compatibility: Mapping[str, object]
    provenance: Mapping[str, object]
    sam2_config: Path
    sam2_checkpoint: Path
    sam2_license: Path
    resnet50_weights: Path
    resnet50_license: Path


@dataclass(frozen=True)
class VerifiedBundle:
    root: Path
    bundle_sha256: str
    classifier: bytes
    imputer: tuple[float, ...]
    threshold: float
    normalized_config: Mapping[str, object]
    preprocessing: Mapping[str, object]
    proposal_config: Mapping[str, object]
    compatibility: Mapping[str, object]
    provenance: Mapping[str, object]
    checkpoint: Path
    sam2_config: Path
    manifest: Mapping[str, object]
    tree_identity: tuple[Mapping[str, object], ...]
    schema: str = BUNDLE_SCHEMA
    profile: str = PUBLIC_BUNDLE_PROFILE
    feature_order: tuple[str, ...] = PUBLIC_FEATURE_ORDER
    feature_order_sha256: str = FEATURE_ORDER_SHA256
    prototype: tuple[float, ...] | None = None
    pca_mean: tuple[float, ...] | None = None
    pca_components: tuple[tuple[float, ...], ...] | None = None
    pca_explained_variance: tuple[float, ...] | None = None
    prototype_npy: bytes | None = None
    pca_mean_npy: bytes | None = None
    pca_components_npy: bytes | None = None
    pca_explained_variance_npy: bytes | None = None
    resnet50_weights: Path | None = None
    feature_config: Mapping[str, object] | None = None
    training_config: Mapping[str, object] | None = None
    assets: Mapping[str, object] | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


def _bundle_v2_profile_contract(
    profile: str,
) -> tuple[Mapping[str, object], tuple[str, ...], Mapping[str, int], int, str, str]:
    """Return the exact v2 asset and filesystem closure for ``profile``."""

    _require(profile in BUNDLE_V2_ASSET_PROFILES, "model-bundle v2 profile is unsupported")
    _require(profile in BUNDLE_V2_FILES_BY_PROFILE, "model-bundle v2 file profile is unsupported")
    asset_profile = BUNDLE_V2_ASSET_PROFILES[profile]
    observed_fields = frozenset(asset_profile)
    _require(
        observed_fields in {
            _BUNDLE_V2_ASSET_FIELDS,
            _BUNDLE_V2_ASSET_FIELDS | _BUNDLE_V2_ASSET_PATH_FIELDS,
        },
        "model-bundle v2 asset profile is invalid",
    )
    config_path, checkpoint_path = _BUNDLE_V2_SAM_PATHS[profile]
    if _BUNDLE_V2_ASSET_PATH_FIELDS <= observed_fields:
        _require(
            asset_profile.get("sam2_config_bundle_path") == config_path
            and asset_profile.get("sam2_checkpoint_bundle_path") == checkpoint_path,
            "model-bundle v2 SAM2 member paths differ from the selected profile",
        )
    bundle_files = BUNDLE_V2_FILES_BY_PROFILE[profile]
    size_limits = BUNDLE_V2_SIZE_LIMITS_BY_PROFILE[profile]
    max_total_bytes = BUNDLE_V2_MAX_TOTAL_BYTES_BY_PROFILE[profile]
    _require(
        set(size_limits) == {"bundle.json", *bundle_files},
        "model-bundle v2 size-limit closure is invalid",
    )
    return (
        asset_profile,
        bundle_files,
        size_limits,
        max_total_bytes,
        config_path,
        checkpoint_path,
    )


def _canonical_profile_device(profile: str) -> str:
    _require(
        profile in V2_PIPELINE_PROFILES,
        "v2 model-bundle profile is unsupported",
    )
    return {
        CANONICAL_CPU_PROFILE: "cpu",
        CANONICAL_GPU_PROFILE: "cuda",
        EFFICIENT_GPU_PROFILE: "cuda",
        FULL_IMAGE_GPU_PROFILE: "cuda",
    }[profile]


def _canonical_runtime_contract(profile: str) -> _CanonicalRuntimeContract:
    """Return and validate the exact runtime closure for a v2 profile."""

    _canonical_profile_device(profile)
    if profile == CANONICAL_CPU_PROFILE:
        python = SCIENCE_CPU_PYTHON
        versions = SCIENCE_CPU_VERSIONS
        lock_sha256 = SCIENCE_CPU_LOCK_SHA256
        policy_field = "cpu_runtime_policy_sha256"
        policy = science_cpu_runtime_policy()
        lock_loader = science_cpu_lock_identity
    else:
        python = SCIENCE_GPU_PYTHON
        versions = SCIENCE_GPU_VERSIONS
        lock_sha256 = SCIENCE_GPU_LOCK_SHA256
        policy_field = "gpu_runtime_policy_sha256"
        policy = science_gpu_runtime_policy()
        lock_loader = science_gpu_lock_identity
    distribution_version = versions.get("compag-curation")
    _require(
        isinstance(distribution_version, str)
        and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", distribution_version) is not None,
        "canonical runtime package version is invalid",
    )
    try:
        runtime_lock = lock_loader()
    except RuntimeError as exc:
        raise PublicIOError("canonical runtime lock identity could not be established") from exc
    _require(
        runtime_lock.get("sha256") == lock_sha256,
        "canonical runtime lock differs from the selected profile",
    )
    policy_sha256 = policy.get("sha256")
    _require(
        isinstance(policy_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", policy_sha256) is not None,
        "canonical runtime policy identity is invalid",
    )
    return _CanonicalRuntimeContract(
        python=python,
        versions=versions,
        lock_sha256=lock_sha256,
        policy_field=policy_field,
        policy_sha256=policy_sha256,
        distribution_version=distribution_version,
    )


def _json(path: Path) -> object:
    payload, _info = stable_file(path, max_bytes=MAX_BUNDLE_JSON_BYTES)
    return canonical_json_value(payload, f"bundle JSON {path.name}")


def _finite(value: object, role: str) -> float:
    _require(isinstance(value, Real) and not isinstance(value, bool), f"{role} must be numeric")
    number = float(value)
    _require(math.isfinite(number), f"{role} must be finite")
    return number


def _feature_order_v2(value: Sequence[str]) -> tuple[str, ...]:
    _require(not isinstance(value, (str, bytes, bytearray)), "feature order must be a sequence of names")
    try:
        order = tuple(value)
    except TypeError as exc:
        raise PublicIOError("feature order must be a sequence of names") from exc
    _require(1 <= len(order) <= MAX_FEATURE_COUNT, "feature order length is outside the supported bound")
    _require(len(set(order)) == len(order), "feature order contains duplicate names")
    for name in order:
        _require(
            isinstance(name, str)
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}", name) is not None,
            "feature order contains an invalid name",
        )
    return order


def compute_feature_order_sha256(feature_order: Sequence[str]) -> str:
    """Return the v2 LF-terminated, one-feature-per-line identity."""

    order = _feature_order_v2(feature_order)
    return hashlib.sha256(("\n".join(order) + "\n").encode("ascii")).hexdigest()


def _finite_vector(
    values: Sequence[float],
    *,
    length: int,
    role: str,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    _require(not isinstance(values, (str, bytes, bytearray)), f"{role} must be a numeric array")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise PublicIOError(f"{role} must be a numeric array") from exc
    _require(len(raw) == length, f"{role} shape mismatch")
    result = tuple(_finite(value, f"{role}[{index}]") for index, value in enumerate(raw))
    if nonnegative:
        _require(all(value >= 0.0 for value in result), f"{role} must be nonnegative")
    return result


def _checked_npy_shape(shape: Sequence[int], role: str) -> tuple[int, ...]:
    _require(not isinstance(shape, (str, bytes, bytearray)), f"{role} NPY shape is invalid")
    normalized = tuple(shape)
    _require(
        normalized
        and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in normalized),
        f"{role} NPY shape is invalid",
    )
    _require(math.prod(normalized) * 4 <= MAX_NPY_STATE_BYTES, f"{role} NPY shape exceeds its bound")
    return normalized


def encode_float32_npy(
    values: Sequence[float],
    *,
    shape: Sequence[int],
    role: str = "numeric state",
) -> bytes:
    """Encode a flat finite vector as deterministic little-endian float32 NPY v1."""

    normalized_shape = _checked_npy_shape(shape, role)
    vector = _finite_vector(values, length=math.prod(normalized_shape), role=role)
    try:
        data = struct.pack(f"<{len(vector)}f", *vector)
    except (OverflowError, struct.error) as exc:
        raise PublicIOError(f"{role} cannot be represented as finite float32") from exc
    _require(
        all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", data)),
        f"{role} cannot be represented as finite float32",
    )
    header_text = (
        "{'descr': '<f4', 'fortran_order': False, "
        f"'shape': {normalized_shape!r}, }}"
    ).encode("ascii")
    padding = (-(10 + len(header_text) + 1)) % 64
    header = header_text + (b" " * padding) + b"\n"
    _require(len(header) <= MAX_NPY_HEADER_BYTES, f"{role} NPY header exceeds its bound")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + data


def _decode_float32_npy(
    payload: bytes,
    *,
    shape: Sequence[int],
    role: str,
) -> tuple[float, ...]:
    expected_shape = _checked_npy_shape(shape, role)
    _require(
        isinstance(payload, bytes)
        and 10 <= len(payload) <= MAX_NPY_STATE_BYTES
        and payload.startswith(b"\x93NUMPY\x01\x00"),
        f"{role} must be a bounded NPY v1 payload",
    )
    header_length = struct.unpack("<H", payload[8:10])[0]
    _require(
        1 <= header_length <= MAX_NPY_HEADER_BYTES
        and 10 + header_length <= len(payload)
        and (10 + header_length) % 64 == 0,
        f"{role} NPY header is invalid",
    )
    header = payload[10 : 10 + header_length]
    _require(header.endswith(b"\n"), f"{role} NPY header is invalid")
    try:
        header_value = ast.literal_eval(header.decode("ascii").strip())
    except (SyntaxError, ValueError, UnicodeError, RecursionError) as exc:
        raise PublicIOError(f"{role} NPY header is invalid") from exc
    _require(
        isinstance(header_value, dict)
        and set(header_value) == {"descr", "fortran_order", "shape"}
        and header_value.get("descr") == "<f4"
        and header_value.get("fortran_order") is False
        and header_value.get("shape") == expected_shape,
        f"{role} NPY dtype, order, or shape mismatch",
    )
    data = payload[10 + header_length :]
    _require(
        len(data) == math.prod(expected_shape) * 4,
        f"{role} NPY payload length or trailing bytes mismatch",
    )
    values = tuple(value[0] for value in struct.iter_unpack("<f", data))
    _require(all(math.isfinite(value) for value in values), f"{role} NPY contains nonfinite values")
    return values


def _canonical_float32_npy(
    payload: bytes,
    *,
    shape: Sequence[int],
    role: str,
) -> tuple[bytes, tuple[float, ...]]:
    values = _decode_float32_npy(payload, shape=shape, role=role)
    return encode_float32_npy(values, shape=shape, role=role), values


def _metadata_mapping(value: Mapping[str, object], role: str) -> dict[str, object]:
    _require(isinstance(value, Mapping), f"{role} must be an object")
    result = dict(value)
    encoded = canonical_json_bytes(result)
    _require(len(encoded) <= MAX_BUNDLE_JSON_BYTES, f"{role} exceeds its JSON size bound")
    _require(
        re.search(
            r"(?:/home/|/root(?:/|\b)|/Users/|/mnt/[a-z]/Users/|fi" + r"le://|[A-Za-z]:\\|\\\\|wsl\.localhost)",
            encoded.decode("ascii"),
            flags=re.IGNORECASE,
        )
        is None,
        f"{role} contains a private locator",
    )
    pending: list[object] = [result]
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            for key, item in current.items():
                if isinstance(key, str) and (key == "sha256" or key.endswith("_sha256")):
                    _require(
                        isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item) is not None,
                        f"{role} contains an invalid declared SHA-256",
                    )
                pending.append(item)
        elif isinstance(current, (list, tuple)):
            pending.extend(current)
    return result


def _v2_metadata_details(
    value: Mapping[str, object],
    role: str,
    *,
    feature_order_sha256: str,
) -> dict[str, object]:
    result = _metadata_mapping(value, role)
    if "feature_order_sha256" in result:
        _require(
            result["feature_order_sha256"] == feature_order_sha256,
            f"{role} feature hash mismatch",
        )
    if "fixed_threshold" in result:
        _require(
            _finite(result["fixed_threshold"], f"{role}.fixed_threshold")
            == CANONICAL_DECISION_THRESHOLD,
            f"{role} fixed threshold mismatch",
        )
    if "ubj_parity_status" in result:
        _require(result["ubj_parity_status"] == "PASS", f"{role} UBJ parity status mismatch")
    return result


def _config_wrapper(
    role: str,
    value: Mapping[str, object],
    *,
    profile: str,
    feature_order_sha256: str,
) -> dict[str, object]:
    config = _metadata_mapping(value, f"bundle {role} config")
    if "profile" in config:
        _require(config["profile"] == profile, f"bundle {role} config profile mismatch")
    if "feature_order_sha256" in config:
        _require(
            config["feature_order_sha256"] == feature_order_sha256,
            f"bundle {role} config feature hash mismatch",
        )
    return {
        "schema": "compag-curation-bundle-config/v2",
        "role": role,
        "profile": profile,
        "feature_order_sha256": feature_order_sha256,
        "config": config,
    }


def _asset_record(path: str, record: Mapping[str, object]) -> dict[str, object]:
    _require(set(record) == {"sha256", "size_bytes", "mode_octal"}, "copied asset record is invalid")
    return {"path": path, **dict(record)}


def _exact_json_object(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    role: str,
) -> None:
    _require(
        set(observed) == set(expected)
        and canonical_json_bytes(dict(observed)) == canonical_json_bytes(dict(expected)),
        f"{role} fields, types, or canonical values mismatch",
    )


def _full_image_config_contracts(
    *,
    profile: str,
    asset_profile: Mapping[str, object],
    feature_order_sha256: str,
) -> dict[str, dict[str, object]]:
    """Return the independent full-image extractor/classifier closure.

    The 93 exported column names remain stable, but their spatial semantics are
    not compatible with the historical 512x512 tiled extractor.  Both the
    spatial mode and the feature-semantics identity therefore travel through
    every contract that can create or consume predictor state.
    """

    _require(
        profile == FULL_IMAGE_GPU_PROFILE,
        "full-image model-bundle config profile mismatch",
    )
    proposal_contract = amg_settings_for_profile(profile).provenance_record(
        profile=profile
    )
    _require(
        proposal_contract.get("spatial_mode") == FULL_IMAGE_SPATIAL_MODE,
        "full-image proposal spatial identity mismatch",
    )
    return {
        "normalized": {
            "profile": profile,
            "spatial_mode": FULL_IMAGE_SPATIAL_MODE,
            "feature_semantics_id": FULL_IMAGE_FEATURE_SEMANTICS_ID,
            "device": "cuda",
            "seed": 42,
            "analysis_unit_kind": FULL_IMAGE_UNIT_KIND,
            "external_tiling": False,
            "max_proposals_per_image": 1000,
            "sam2_checkpoint_sha256": asset_profile["sam2_checkpoint_sha256"],
            "sam2_config_sha256": asset_profile["sam2_config_sha256"],
            "sam2_config_locator": asset_profile["sam2_config_locator"],
            "embedding_weights_sha256": asset_profile["resnet50_weights_sha256"],
            "embedding_dimensions": CANONICAL_EMBEDDING_DIMENSIONS,
            "pca_components": CANONICAL_PCA_DIMENSIONS,
            "proposal_scales": [1.0],
            "yolo_enabled": False,
            "feature_order_sha256": feature_order_sha256,
            "inference_threshold": CANONICAL_DECISION_THRESHOLD,
            "nms_iou_threshold": 0.5,
            "threshold_method": FIXED_CANONICAL_METHOD,
        },
        "preprocessing": {
            "profile": profile,
            "spatial_mode": FULL_IMAGE_SPATIAL_MODE,
            "analysis_unit_kind": FULL_IMAGE_UNIT_KIND,
            "analysis_units_per_image": 1,
            "external_tiling": False,
            "unit_format": "png",
            "unit_lossless": True,
            "png_compression": 6,
            "yellow_card_hsv_lower": [15, 40, 60],
            "yellow_card_hsv_upper": [40, 255, 255],
            "yellow_card_quad_epsilon_fraction": 0.02,
            "yellow_card_fallback": "identity",
        },
        "proposal": {
            **proposal_contract,
            "external_tiling": False,
            "sam2_output_mode": "uncompressed_rle",
            "durable_mask_encoding": FULL_IMAGE_DURABLE_MASK_ENCODING,
            "connected_component_policy": FULL_IMAGE_COMPONENT_POLICY,
            "feature_semantics_id": FULL_IMAGE_FEATURE_SEMANTICS_ID,
            "sam2_checkpoint_sha256": asset_profile["sam2_checkpoint_sha256"],
            "sam2_config_sha256": asset_profile["sam2_config_sha256"],
            "sam2_config_locator": asset_profile["sam2_config_locator"],
            "feature_order_sha256": feature_order_sha256,
        },
        "features": {
            "profile": profile,
            "spatial_mode": FULL_IMAGE_SPATIAL_MODE,
            "feature_semantics_id": FULL_IMAGE_FEATURE_SEMANTICS_ID,
            "mode": "ultra",
            "geometry_reference": "full-analysis-frame",
            "texture_reference": "bounded-roi-context",
            "roi_context_fraction": FULL_IMAGE_CONTEXT_FRACTION,
            "roi_context_minimum_pixels": FULL_IMAGE_CONTEXT_MINIMUM_PIXELS,
            "proposal_batch_size": FULL_IMAGE_FEATURE_PROPOSAL_BATCH,
            "feature_crop_scales": [0.67, 0.8, 1.0, 1.25],
            "masked_crop_padding": 0.1,
            "embedding_backbone": "resnet50-imagenet1k-v2",
            "embedding_weights_sha256": asset_profile["resnet50_weights_sha256"],
            "embedding_dimensions": CANONICAL_EMBEDDING_DIMENSIONS,
            "prototype": "train_positive_mean_l2_normalized",
            "pca_components": CANONICAL_PCA_DIMENSIONS,
            "feature_order_sha256": feature_order_sha256,
            "missing_feature_policy": "reject",
            "zero_fill": False,
        },
        "training": {
            "profile": profile,
            "spatial_mode": FULL_IMAGE_SPATIAL_MODE,
            "feature_semantics_id": FULL_IMAGE_FEATURE_SEMANTICS_ID,
            "group_test_fraction": 0.2,
            "group_cv_folds": 5,
            "search_iterations": 30,
            "inner_validation_fraction": 0.3,
            "validation_scale": 1.0,
            "early_stopping_rounds": 30,
            "review_action_weights": {
                "accept": 1.0,
                "flip": 1.0,
                "sus_accept": 0.4,
                "sus_flip": 0.4,
                "skip": 0.0,
            },
            "tabular_augmentation_enabled": False,
            "safe_smote_train_fold_only": True,
            "scale_pos_weight": 1.0,
            "xgboost_host_thread_count": 1,
            "inference_threshold": CANONICAL_DECISION_THRESHOLD,
            "threshold_method": FIXED_CANONICAL_METHOD,
            "full_image_nms_iou": 0.5,
            "feature_order_sha256": feature_order_sha256,
        },
    }


def _canonical_config_contracts(
    *,
    profile: str,
    asset_profile: Mapping[str, object],
    feature_order_sha256: str,
) -> dict[str, dict[str, object]]:
    if profile == FULL_IMAGE_GPU_PROFILE:
        return _full_image_config_contracts(
            profile=profile,
            asset_profile=asset_profile,
            feature_order_sha256=feature_order_sha256,
        )
    device = _canonical_profile_device(profile)
    proposal_contract = amg_settings_for_profile(profile).provenance_record(
        profile=profile
    )
    return {
        "normalized": {
            "profile": profile,
            "device": device,
            "seed": 42,
            "tile_size": 512,
            "tile_stride": 512,
            "sam2_checkpoint_sha256": asset_profile["sam2_checkpoint_sha256"],
            "sam2_config_sha256": asset_profile["sam2_config_sha256"],
            "sam2_config_locator": asset_profile["sam2_config_locator"],
            "embedding_weights_sha256": asset_profile["resnet50_weights_sha256"],
            "embedding_dimensions": CANONICAL_EMBEDDING_DIMENSIONS,
            "pca_components": CANONICAL_PCA_DIMENSIONS,
            "proposal_scales": [1.0],
            "yolo_enabled": False,
            "feature_order_sha256": feature_order_sha256,
            "inference_threshold": CANONICAL_DECISION_THRESHOLD,
            "nms_iou_threshold": 0.5,
            "threshold_method": FIXED_CANONICAL_METHOD,
        },
        "preprocessing": {
            "profile": profile,
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
        },
        "proposal": {
            **proposal_contract,
            "sam2_checkpoint_sha256": asset_profile["sam2_checkpoint_sha256"],
            "sam2_config_sha256": asset_profile["sam2_config_sha256"],
            "sam2_config_locator": asset_profile["sam2_config_locator"],
            "feature_order_sha256": feature_order_sha256,
        },
        "features": {
            "profile": profile,
            "mode": "ultra",
            "feature_crop_scales": [0.67, 0.8, 1.0, 1.25],
            "masked_crop_padding": 0.1,
            "embedding_backbone": "resnet50-imagenet1k-v2",
            "embedding_weights_sha256": asset_profile["resnet50_weights_sha256"],
            "embedding_dimensions": CANONICAL_EMBEDDING_DIMENSIONS,
            "prototype": "train_positive_mean_l2_normalized",
            "pca_components": CANONICAL_PCA_DIMENSIONS,
            "feature_order_sha256": feature_order_sha256,
            "missing_feature_policy": "reject",
            "zero_fill": False,
        },
        "training": {
            "profile": profile,
            "group_test_fraction": 0.2,
            "group_cv_folds": 5,
            "search_iterations": 30,
            "inner_validation_fraction": 0.3,
            "validation_scale": 1.0,
            "early_stopping_rounds": 30,
            "review_action_weights": {
                "accept": 1.0,
                "flip": 1.0,
                "sus_accept": 0.4,
                "sus_flip": 0.4,
                "skip": 0.0,
            },
            "tabular_augmentation_enabled": False,
            "safe_smote_train_fold_only": True,
            "scale_pos_weight": 1.0,
            "xgboost_host_thread_count": 1,
            "inference_threshold": CANONICAL_DECISION_THRESHOLD,
            "threshold_method": FIXED_CANONICAL_METHOD,
            "full_image_nms_iou": 0.5,
            "feature_order_sha256": feature_order_sha256,
        },
    }


def _validate_v2_details(
    compatibility: Mapping[str, object],
    provenance: Mapping[str, object],
    *,
    profile: str,
    feature_order_sha256: str,
) -> tuple[int, int]:
    expected_device = _canonical_profile_device(profile)
    _exact_json_object(
        compatibility,
        {
            "classifier_format": "XGBOOST_UBJ",
            "no_pickle_or_joblib": True,
            "feature_order_sha256": feature_order_sha256,
            "fixed_threshold": CANONICAL_DECISION_THRESHOLD,
        },
        "bundle compatibility details",
    )
    legacy_fields = {
        "reviewed_sha256",
        "split_manifest_sha256",
        "raw_features_sha256",
        "feature_state_fit_group_sha256",
        "feature_state_fit_rows_sha256",
        "feature_state_fit_scope",
        "validation_feature_scale",
        "cv_score_interpretation",
        "training_row_count",
        "positive_training_row_count",
        "fixed_threshold",
        "feature_order_sha256",
        "ubj_parity_status",
        "paper_result_reproduction",
    }
    profiled_fields = legacy_fields | {"profile", "device"}
    sequential_image_fields = {
        "active_learning_workflow",
        "model_label",
        "active_learning_round_number",
        "parent_bundle_sha256",
        "genesis_stage20_features_sha256",
        "genesis_split_sha256",
        "round_image_id",
        "round_image_sha256",
        "round_group_id",
        "cumulative_al_feature_archive_sha256",
        "effective_train_groups_sha256",
        "frozen_test_groups_sha256",
    }
    transfer_image_fields = {
        "active_learning_workflow",
        "lineage_policy",
        "reproduction_claim",
        "model_label",
        "active_learning_round_number",
        "parent_model_identity_sha256",
        "transfer_baseline_sha256",
        "transfer_split_sha256",
        "r92_resource_manifest_sha256",
        "r92_classifier_sha256",
        "r92_feature_state_sha256",
        "frozen_feature_state_policy",
        "pca_explained_variance_policy",
        "round_image_id",
        "round_image_sha256",
        "round_group_id",
        "cumulative_al_feature_archive_sha256",
        "effective_train_groups_sha256",
        "frozen_test_groups_sha256",
    }
    observed_fields = set(provenance)
    is_transfer_image_lineage = observed_fields == profiled_fields | transfer_image_fields
    _require(
        observed_fields == profiled_fields
        or observed_fields == profiled_fields | sequential_image_fields
        or is_transfer_image_lineage
        or (profile == CANONICAL_CPU_PROFILE and observed_fields == legacy_fields),
        "bundle provenance details field closure mismatch",
    )
    if profiled_fields <= observed_fields:
        _require(
            provenance.get("profile") == profile
            and provenance.get("device") == expected_device,
            "bundle provenance profile/device pairing mismatch",
        )
    for key in (
        "reviewed_sha256",
        "split_manifest_sha256",
        "raw_features_sha256",
        "feature_state_fit_group_sha256",
        "feature_state_fit_rows_sha256",
    ):
        _require(
            isinstance(provenance.get(key), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(provenance[key])) is not None,
            f"bundle provenance details {key} is invalid",
        )
    training_rows = provenance.get("training_row_count")
    positive_rows = provenance.get("positive_training_row_count")
    _require(
        isinstance(training_rows, int)
        and not isinstance(training_rows, bool)
        and training_rows >= CANONICAL_PCA_DIMENSIONS,
        "bundle provenance training_row_count must be an integer >= 32",
    )
    _require(
        isinstance(positive_rows, int)
        and not isinstance(positive_rows, bool)
        and 1 <= positive_rows <= training_rows,
        "bundle provenance positive_training_row_count is invalid",
    )
    _require(
        isinstance(provenance.get("fixed_threshold"), float)
        and provenance.get("fixed_threshold") == CANONICAL_DECISION_THRESHOLD
        and provenance.get("feature_order_sha256") == feature_order_sha256
        and provenance.get("ubj_parity_status") == "PASS"
        and provenance.get("paper_result_reproduction") == "NOT_CLAIMED",
        "bundle provenance details fixed canonical values mismatch",
    )
    expected_feature_state_scope = (
        POST_R92_TRANSFER_FEATURE_STATE_SCOPE
        if is_transfer_image_lineage
        else "OUTER_TRAIN_ROWS_BEFORE_CV"
    )
    expected_cv_interpretation = (
        POST_R92_TRANSFER_CV_INTERPRETATION
        if is_transfer_image_lineage
        else "CONDITIONAL_ON_OUTER_TRAIN_PREFIT_PROTOTYPE_AND_PCA"
    )
    _require(
        provenance.get("feature_state_fit_scope") == expected_feature_state_scope
        and isinstance(provenance.get("validation_feature_scale"), float)
        and provenance.get("validation_feature_scale") == 1.0
        and provenance.get("cv_score_interpretation")
        == expected_cv_interpretation,
        "bundle provenance feature-state fit scope mismatch",
    )
    if sequential_image_fields <= observed_fields:
        _require(
            provenance.get("active_learning_workflow")
            == "SINGLE_NEW_IMAGE_UNCERTAINTY_BATCH_FULL_RETRAIN_V1"
            and isinstance(provenance.get("model_label"), str)
            and re.fullmatch(
                r"project-r[1-9][0-9]*",
                str(provenance["model_label"]),
            )
            is not None
            and isinstance(provenance.get("active_learning_round_number"), int)
            and not isinstance(provenance.get("active_learning_round_number"), bool)
            and int(provenance["active_learning_round_number"]) >= 1
            and provenance.get("model_label")
            == f"project-r{provenance['active_learning_round_number']}"
            and isinstance(provenance.get("round_group_id"), str)
            and re.fullmatch(
                r"[a-z0-9][a-z0-9._-]{0,63}",
                str(provenance["round_group_id"]),
            )
            is not None,
            "bundle sequential-image active-learning identity is invalid",
        )
        for key in (
            "parent_bundle_sha256",
            "genesis_stage20_features_sha256",
            "genesis_split_sha256",
            "round_image_id",
            "round_image_sha256",
            "cumulative_al_feature_archive_sha256",
            "effective_train_groups_sha256",
            "frozen_test_groups_sha256",
        ):
            _require(
                isinstance(provenance.get(key), str)
                and re.fullmatch(r"[0-9a-f]{64}", str(provenance[key]))
                is not None,
                f"bundle sequential-image lineage {key} is invalid",
            )
    if is_transfer_image_lineage:
        _require(
            profile == CANONICAL_GPU_PROFILE
            and provenance.get("active_learning_workflow")
            == POST_R92_TRANSFER_WORKFLOW
            and provenance.get("lineage_policy")
            == POST_R92_TRANSFER_LINEAGE_POLICY
            and provenance.get("reproduction_claim")
            == POST_R92_TRANSFER_REPRODUCTION_CLAIM
            and provenance.get("frozen_feature_state_policy")
            == POST_R92_TRANSFER_FROZEN_STATE_POLICY
            and provenance.get("pca_explained_variance_policy")
            == POST_R92_TRANSFER_PCA_VARIANCE_POLICY
            and isinstance(provenance.get("model_label"), str)
            and re.fullmatch(
                r"project-r[1-9][0-9]*",
                str(provenance["model_label"]),
            )
            is not None
            and isinstance(provenance.get("active_learning_round_number"), int)
            and not isinstance(provenance.get("active_learning_round_number"), bool)
            and int(provenance["active_learning_round_number"]) >= 1
            and provenance.get("model_label")
            == f"project-r{provenance['active_learning_round_number']}"
            and isinstance(provenance.get("round_group_id"), str)
            and re.fullmatch(
                r"[a-z0-9][a-z0-9._-]{0,63}",
                str(provenance["round_group_id"]),
            )
            is not None,
            "bundle post-r92 transfer active-learning identity is invalid",
        )
        for key in (
            "parent_model_identity_sha256",
            "transfer_baseline_sha256",
            "transfer_split_sha256",
            "r92_resource_manifest_sha256",
            "r92_classifier_sha256",
            "r92_feature_state_sha256",
            "round_image_id",
            "round_image_sha256",
            "cumulative_al_feature_archive_sha256",
            "effective_train_groups_sha256",
            "frozen_test_groups_sha256",
        ):
            _require(
                isinstance(provenance.get(key), str)
                and re.fullmatch(r"[0-9a-f]{64}", str(provenance[key]))
                is not None,
                f"bundle post-r92 transfer lineage {key} is invalid",
            )
    return training_rows, positive_rows


def _validate_canonical_ubj(payload: bytes) -> object:
    _require(
        isinstance(payload, bytes)
        and 0 < len(payload) <= MAX_CLASSIFIER_BYTES
        and payload.startswith(b"{L"),
        "classifier payload must be bounded canonical XGBoost UBJ bytes",
    )
    try:
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(bytearray(payload))
        roundtrip = bytes(booster.save_raw(raw_format="ubj"))
        decoded_model_payload = bytes(booster.save_raw(raw_format="json"))
        config_payload = booster.save_config().encode("utf-8")
        attributes = booster.attributes()
        reported_feature_count = booster.num_features()
    except (Exception, RecursionError) as exc:
        raise PublicIOError("classifier.ubj is not a loadable canonical XGBoost UBJ model") from exc
    _require(roundtrip == payload, "classifier UBJ is noncanonical or has trailing bytes")
    _require(
        0 < len(decoded_model_payload) <= MAX_CLASSIFIER_JSON_BYTES,
        "classifier decoded JSON model exceeds its size bound",
    )
    decoded_model = strict_json_bytes(decoded_model_payload, "classifier decoded JSON model")
    config = strict_json_bytes(config_payload, "classifier configuration")
    _require(
        isinstance(decoded_model, Mapping) and isinstance(config, Mapping),
        "classifier decoded JSON metadata must be objects",
    )
    # UBJ typed arrays contain arbitrary IEEE-754 and integer bytes.  Searching
    # the serialized payload itself therefore treats an accidental byte pair
    # such as 0x5c,0x5c in a learned weight as a UNC path.  Instead, ask
    # XGBoost for its complete JSON model projection, parse it strictly, and
    # inspect only decoded strings.  Configuration and Python metadata views
    # are included as independent surfaces as well.
    metadata_values: list[object] = [
        decoded_model,
        config,
        attributes,
        tuple(booster.feature_names or ()),
        tuple(booster.feature_types or ()),
    ]
    private_locator = re.compile(
        r"(?:/home/|/root(?:/|\b)|/Users/|/mnt/[a-z]/Users/|fi" + r"le://|[A-Za-z]:\\|\\\\|wsl\.localhost)",
        flags=re.IGNORECASE,
    )
    pending = metadata_values
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            candidates = (*current.keys(), *current.values())
        elif isinstance(current, (list, tuple)):
            candidates = current
        elif isinstance(current, str):
            _require(
                private_locator.search(current) is None,
                "classifier UBJ contains private locator metadata",
            )
            continue
        else:
            continue
        # Large tree arrays contain only numbers.  Do not duplicate all those
        # leaves into the traversal stack; retain only nested containers and
        # textual values that can actually carry a locator.
        pending.extend(
            candidate
            for candidate in candidates
            if isinstance(candidate, (Mapping, list, tuple, str))
        )
    _require(attributes == {}, "classifier UBJ contains unexpected Booster attributes")
    _require(tuple(booster.feature_names or ()) == CANONICAL_FEATURE_ORDER, "classifier UBJ feature names differ from the canonical feature order")
    _require(
        isinstance(reported_feature_count, Integral)
        and not isinstance(reported_feature_count, bool)
        and int(reported_feature_count) == len(CANONICAL_FEATURE_ORDER) == 93,
        "classifier UBJ feature count mismatch",
    )
    _require(not (booster.feature_types or []), "classifier UBJ contains unexpected feature types")
    learner = config.get("learner") if isinstance(config, dict) else None
    model_param = learner.get("learner_model_param") if isinstance(learner, dict) else None
    train_param = learner.get("learner_train_param") if isinstance(learner, dict) else None
    objective = learner.get("objective") if isinstance(learner, dict) else None
    _require(
        isinstance(model_param, dict)
        and model_param.get("num_feature") == "93"
        and model_param.get("num_target") == "1"
        and model_param.get("num_class") == "0"
        and isinstance(train_param, dict)
        and train_param.get("objective") == "binary:logistic"
        and train_param.get("multi_strategy") == "one_output_per_tree"
        and isinstance(objective, dict)
        and objective.get("name") == "binary:logistic",
        "classifier UBJ objective or single-output semantics mismatch",
    )
    return booster


def _copy_verified_asset(
    source: Path,
    destination: Path,
    *,
    expected_sha256: object,
    max_bytes: int,
    role: str,
) -> dict[str, object]:
    _require(
        isinstance(expected_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
        f"{role} expected hash is invalid",
    )
    before_path = source.lstat()
    _require(
        stat.S_ISREG(before_path.st_mode)
        and not stat.S_ISLNK(before_path.st_mode)
        and before_path.st_nlink == 1
        and 0 <= before_path.st_size <= max_bytes,
        f"{role} source is unsafe or exceeds its size bound",
    )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if hasattr(os, "O_NOATIME"):
        flags |= os.O_NOATIME
    source_fd = os.open(source, flags)
    destination_fd = -1
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    try:
        before = os.fstat(source_fd)
        _require(identity(before) == identity(before_path), f"{role} source changed while opening")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o644,
        )
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(source_fd, min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            _require(total <= max_bytes, f"{role} source exceeds its size bound")
            digest.update(chunk)
            offset = 0
            while offset < len(chunk):
                offset += os.write(destination_fd, chunk[offset:])
        observed_sha256 = digest.hexdigest()
        _require(observed_sha256 == expected_sha256, f"{role} hash mismatch")
        os.fchmod(destination_fd, 0o644)
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        target = os.fstat(destination_fd)
        _require(identity(after) == identity(before), f"{role} source changed while copying")
        _require(identity(source.lstat()) == identity(before), f"{role} source path changed while copying")
        _require(
            total == before.st_size == target.st_size
            and target.st_nlink == 1
            and stat.S_ISREG(target.st_mode)
            and (target.st_dev, target.st_ino) != (before.st_dev, before.st_ino),
            f"{role} copied file is invalid",
        )
    finally:
        os.close(source_fd)
        if destination_fd >= 0:
            os.close(destination_fd)
    fsync_directory(destination.parent)
    return {"sha256": expected_sha256, "size_bytes": total, "mode_octal": "0644"}


def _remove_staging(path: Path, parent: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    fsync_directory(parent)


def write_model_bundle(request: BundleWriteRequest) -> dict[str, object]:
    _require(
        False,
        "balanced v1 model-bundle writing is historical read/inspect/verify-only in v1.7",
    )
    absolute = request.output.absolute()
    parent = absolute.parent.resolve(strict=True)
    final_output = parent / absolute.name
    _require(final_output == absolute, "model-bundle output contains a symlinked path component")
    portable_basename(final_output.name, "model-bundle output name")
    _require(not final_output.exists() and not final_output.is_symlink(), "model-bundle output already exists")
    _require(parent.is_dir() and not parent.is_symlink(), "model-bundle parent is unsafe")
    _require(len(f".{final_output.name}.bundle.{uuid.uuid4()}".encode("ascii")) <= 255, "model-bundle output name is too long for staging")
    _require(
        request.classifier_ubj
        and len(request.classifier_ubj) <= MAX_CLASSIFIER_BYTES
        and not request.classifier_ubj.startswith(b"\x80"),
        "classifier UBJ payload is empty, oversized, or pickle-like",
    )
    _require(set(request.imputer_statistics) == set(PUBLIC_FEATURE_ORDER), "imputer feature set mismatch")
    imputer = {name: _finite(request.imputer_statistics[name], f"imputer.{name}") for name in PUBLIC_FEATURE_ORDER}
    threshold = _finite(request.threshold, "threshold")
    _require(0.0 <= threshold <= 1.0, "threshold must be a probability")
    profile = request.normalized_config.get("profile")
    _require(isinstance(profile, str) and profile in BUNDLE_ASSET_PROFILES, "model-bundle profile is unsupported")
    output = parent / f".{final_output.name}.bundle.{uuid.uuid4()}"
    output.mkdir(mode=0o755)
    fsync_directory(parent)
    output.chmod(0o755)
    for relative in ("sam2/configs/sam2.1", "sam2/checkpoints", "licenses"):
        directory = output.joinpath(*relative.split("/"))
        directory.mkdir(parents=True, exist_ok=False)
        current = directory
        while current != output:
            current.chmod(0o755)
            current = current.parent
    write_new_bytes(output / "classifier.ubj", request.classifier_ubj)
    write_new_json(output / "imputer.json", {"schema": "compag-curation-imputer/v1", "strategy": "median", "feature_order": list(PUBLIC_FEATURE_ORDER), "statistics": imputer})
    write_new_json(output / "feature_schema.json", {"schema": "compag-curation-feature-schema/v1", "profile": profile, "feature_order": list(PUBLIC_FEATURE_ORDER), "feature_order_sha256": FEATURE_ORDER_SHA256, "scaler": None, "pca": None, "embedding": None})
    write_new_json(output / "threshold.json", {"schema": "compag-curation-threshold/v1", "selection_partition": "validation", "value": threshold})
    write_new_json(output / "class_map.json", {"schema": "compag-curation-class-map/v1", "negative": 0, "positive": 1})
    write_new_json(output / "normalized_config.json", dict(request.normalized_config))
    write_new_json(output / "preprocessing.json", dict(request.preprocessing))
    write_new_json(output / "proposal_config.json", dict(request.proposal_config))
    write_new_json(output / "compatibility.json", dict(request.compatibility))
    write_new_json(output / "provenance.json", dict(request.provenance))
    config_copy = copy_new(request.sam2_config, output / "sam2/configs/sam2.1/model.yaml")
    checkpoint_copy = copy_new(request.sam2_checkpoint, output / "sam2/checkpoints/model.pt")
    license_copy = copy_new(request.sam2_license, output / "licenses/SAM2-APACHE-2.0.txt")
    _require(license_copy["sha256"] == SAM2_LICENSE_SHA256, "SAM2 license text differs from the reviewed Apache-2.0 authority")
    rows = manifest_rows(output, excluded={"bundle.json"})
    _require({str(row["path"]) for row in rows} == set(BUNDLE_FILES), "bundle payload closure mismatch before sealing")
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "status": "PASS",
        "profile": profile,
        "feature_order_sha256": FEATURE_ORDER_SHA256,
        "classifier_format": "XGBOOST_UBJ",
        "checkpoint": checkpoint_copy,
        "sam2_config": config_copy,
        "sam2_license": license_copy,
        "members": rows,
    }
    write_new_json(output / "bundle.json", manifest)
    verified = verify_model_bundle(output)
    fsync_directory(output)
    publish_directory_noreplace(output, final_output)
    verified = verify_model_bundle(final_output)
    return {
        "status": "PASS",
        "root": str(final_output),
        "bundle_sha256": verified.bundle_sha256,
        "members": len(rows) + 1,
        "checkpoint_sha256": checkpoint_copy["sha256"],
        "config_sha256": config_copy["sha256"],
    }


def write_model_bundle_v2(request: BundleV2WriteRequest) -> dict[str, object]:
    """Write a sealed v2 extractor/classifier closure without pickle state."""

    _require(
        request.profile in GPU_EXECUTION_PROFILES,
        "v2 model-bundle writing requires a supported GPU execution profile; canonical CPU bundles are historical read/inspect/verify-only",
    )
    absolute = request.output.absolute()
    parent = absolute.parent.resolve(strict=True)
    final_output = parent / absolute.name
    _require(final_output == absolute, "model-bundle output contains a symlinked path component")
    portable_basename(final_output.name, "model-bundle output name")
    _require(not final_output.exists() and not final_output.is_symlink(), "model-bundle output already exists")
    _require(parent.is_dir() and not parent.is_symlink(), "model-bundle parent is unsafe")
    _require(
        len(f".{final_output.name}.bundle.{uuid.uuid4()}".encode("ascii")) <= 255,
        "model-bundle output name is too long for staging",
    )
    _validate_canonical_ubj(request.classifier_ubj)
    (
        asset_profile,
        bundle_files,
        _size_limits,
        _max_total_bytes,
        sam2_config_bundle_path,
        sam2_checkpoint_bundle_path,
    ) = _bundle_v2_profile_contract(request.profile)

    feature_order = _feature_order_v2(request.feature_order)
    feature_order_sha256 = compute_feature_order_sha256(feature_order)
    _require(
        feature_order == CANONICAL_FEATURE_ORDER
        and len(feature_order) == 93
        and isinstance(request.feature_order_sha256, str)
        and request.feature_order_sha256 == feature_order_sha256 == CANONICAL_FEATURE_ORDER_SHA256,
        "model-bundle v2 canonical feature-order identity mismatch",
    )
    _require(
        set(request.imputer_statistics) == set(feature_order),
        "model-bundle v2 imputer feature set mismatch",
    )
    imputer = {
        name: _finite(request.imputer_statistics[name], f"imputer.{name}")
        for name in feature_order
    }
    _require(isinstance(request.prototype_npy, bytes), "prototype must be NPY bytes")
    _require(isinstance(request.pca_mean_npy, bytes), "PCA mean must be NPY bytes")
    _require(isinstance(request.pca_components_npy, bytes), "PCA components must be NPY bytes")
    prototype_npy, prototype = _canonical_float32_npy(
        request.prototype_npy,
        shape=(CANONICAL_EMBEDDING_DIMENSIONS,),
        role="prototype",
    )
    _require(
        math.isclose(math.sqrt(sum(value * value for value in prototype)), 1.0, rel_tol=0.0, abs_tol=1e-5),
        "canonical prototype must be unit L2 normalized",
    )
    pca_mean_npy, _pca_mean = _canonical_float32_npy(
        request.pca_mean_npy,
        shape=(CANONICAL_EMBEDDING_DIMENSIONS,),
        role="PCA mean",
    )
    pca_components_npy, _pca_components = _canonical_float32_npy(
        request.pca_components_npy,
        shape=(CANONICAL_PCA_DIMENSIONS, CANONICAL_EMBEDDING_DIMENSIONS),
        role="PCA components",
    )
    pca_explained_variance = _finite_vector(
        request.pca_explained_variance,
        length=CANONICAL_PCA_DIMENSIONS,
        role="pca.explained_variance",
        nonnegative=True,
    )
    pca_explained_variance_npy = encode_float32_npy(
        pca_explained_variance,
        shape=(CANONICAL_PCA_DIMENSIONS,),
        role="PCA explained variance",
    )

    configs = {
        role: _config_wrapper(
            role,
            value,
            profile=request.profile,
            feature_order_sha256=feature_order_sha256,
        )
        for role, value in (
            ("normalized", request.normalized_config),
            ("preprocessing", request.preprocessing),
            ("proposal", request.proposal_config),
            ("features", request.feature_config),
            ("training", request.training_config),
        )
    }
    expected_configs = _canonical_config_contracts(
        profile=request.profile,
        asset_profile=asset_profile,
        feature_order_sha256=feature_order_sha256,
    )
    for role, expected in expected_configs.items():
        config = configs[role].get("config")
        _require(isinstance(config, dict), f"bundle {role} config wrapper is invalid")
        _exact_json_object(config, expected, f"bundle {role} config")

    compatibility_details = _v2_metadata_details(
        request.compatibility,
        "bundle compatibility details",
        feature_order_sha256=feature_order_sha256,
    )
    provenance_details = _v2_metadata_details(
        request.provenance,
        "bundle provenance details",
        feature_order_sha256=feature_order_sha256,
    )
    _validate_v2_details(
        compatibility_details,
        provenance_details,
        profile=request.profile,
        feature_order_sha256=feature_order_sha256,
    )

    feature_schema = {
        "schema": "compag-curation-feature-schema/v2",
        "profile": request.profile,
        "feature_order": list(feature_order),
        "feature_order_sha256": feature_order_sha256,
        "feature_order_hash_encoding": FEATURE_ORDER_HASH_ENCODING_V2,
        "imputer": "imputer.json",
        "prototype": "prototype.npy",
        "pca": {
            "mean": "pca32_mean.npy",
            "components": "pca32_components.npy",
            "explained_variance": "pca32_explained_variance.npy",
        },
        "scaler": None,
        "embedding": {
            "architecture": asset_profile["resnet50_architecture"],
            "weights_identity": asset_profile["resnet50_weights_identity"],
            "weights": "resnet50/resnet50-imagenet1k-v2.pth",
            "dimensions": CANONICAL_EMBEDDING_DIMENSIONS,
        },
    }
    imputer_payload = {
        "schema": "compag-curation-imputer/v2",
        "strategy": "median",
        "feature_order": list(feature_order),
        "feature_order_sha256": feature_order_sha256,
        "statistics": imputer,
    }
    runtime_contract = _canonical_runtime_contract(request.profile)
    implementation_identity = _expected_package_implementation_identity(
        request.profile,
        expected_distribution_version=runtime_contract.distribution_version,
    )
    compatibility = {
        "schema": "compag-curation-model-compatibility/v3",
        "profile": request.profile,
        "feature_order_sha256": feature_order_sha256,
        "classifier_format": "XGBOOST_UBJ",
        "threshold_method": FIXED_CANONICAL_METHOD,
        "python": runtime_contract.python,
        "versions": dict(runtime_contract.versions),
        "lock_sha256": runtime_contract.lock_sha256,
        runtime_contract.policy_field: runtime_contract.policy_sha256,
        "implementation_identity": implementation_identity,
        "sam2_architecture": asset_profile["sam2_architecture"],
        "resnet50_weights_identity": asset_profile["resnet50_weights_identity"],
        "embedding_dimensions": CANONICAL_EMBEDDING_DIMENSIONS,
        "pca_components": CANONICAL_PCA_DIMENSIONS,
        "details": compatibility_details,
    }
    provenance = {
        "schema": "compag-curation-model-provenance/v2",
        "profile": request.profile,
        "feature_order_sha256": feature_order_sha256,
        "threshold_method": FIXED_CANONICAL_METHOD,
        "details": provenance_details,
    }
    json_payloads = {
        "imputer.json": imputer_payload,
        "feature_schema.json": feature_schema,
        "threshold.json": {
            "schema": "compag-curation-threshold/v2",
            "method": FIXED_CANONICAL_METHOD,
            "value": CANONICAL_DECISION_THRESHOLD,
        },
        "class_map.json": {
            "schema": "compag-curation-class-map/v1",
            "negative": 0,
            "positive": 1,
        },
        "normalized_config.json": configs["normalized"],
        "preprocessing.json": configs["preprocessing"],
        "proposal_config.json": configs["proposal"],
        "feature_config.json": configs["features"],
        "training_config.json": configs["training"],
        "compatibility.json": compatibility,
        "provenance.json": provenance,
    }
    _require(
        all(len(canonical_json_bytes(value)) <= MAX_BUNDLE_JSON_BYTES for value in json_payloads.values()),
        "model-bundle v2 JSON member exceeds its size bound",
    )

    output = parent / f".{final_output.name}.bundle.{uuid.uuid4()}"
    try:
        return _write_model_bundle_v2_staging(
            request=request,
            output=output,
            final_output=final_output,
            parent=parent,
            asset_profile=asset_profile,
            bundle_files=bundle_files,
            sam2_config_bundle_path=sam2_config_bundle_path,
            sam2_checkpoint_bundle_path=sam2_checkpoint_bundle_path,
            feature_order_sha256=feature_order_sha256,
            prototype_npy=prototype_npy,
            pca_mean_npy=pca_mean_npy,
            pca_components_npy=pca_components_npy,
            pca_explained_variance_npy=pca_explained_variance_npy,
            json_payloads=json_payloads,
        )
    except BaseException:
        _remove_staging(output, parent)
        raise


def _write_model_bundle_v2_staging(
    *,
    request: BundleV2WriteRequest,
    output: Path,
    final_output: Path,
    parent: Path,
    asset_profile: Mapping[str, object],
    bundle_files: Sequence[str],
    sam2_config_bundle_path: str,
    sam2_checkpoint_bundle_path: str,
    feature_order_sha256: str,
    prototype_npy: bytes,
    pca_mean_npy: bytes,
    pca_components_npy: bytes,
    pca_explained_variance_npy: bytes,
    json_payloads: Mapping[str, object],
) -> dict[str, object]:
    output.mkdir(mode=0o755)
    fsync_directory(parent)
    output.chmod(0o755)
    for relative in ("sam2/configs/sam2.1", "sam2/checkpoints", "resnet50", "licenses"):
        directory = output.joinpath(*relative.split("/"))
        directory.mkdir(parents=True, exist_ok=False)
        current = directory
        while current != output:
            current.chmod(0o755)
            current = current.parent
    write_new_bytes(output / "classifier.ubj", request.classifier_ubj)
    write_new_bytes(output / "prototype.npy", prototype_npy)
    write_new_bytes(output / "pca32_mean.npy", pca_mean_npy)
    write_new_bytes(output / "pca32_components.npy", pca_components_npy)
    write_new_bytes(output / "pca32_explained_variance.npy", pca_explained_variance_npy)
    for relative, value in json_payloads.items():
        write_new_json(output / relative, value)
    sam2_config = _copy_verified_asset(
        request.sam2_config,
        output / sam2_config_bundle_path,
        expected_sha256=asset_profile["sam2_config_sha256"],
        max_bytes=1024 * 1024,
        role="SAM2 config",
    )
    sam2_checkpoint = _copy_verified_asset(
        request.sam2_checkpoint,
        output / sam2_checkpoint_bundle_path,
        expected_sha256=asset_profile["sam2_checkpoint_sha256"],
        max_bytes=MAX_SAM_CHECKPOINT_BYTES,
        role="SAM2 checkpoint",
    )
    sam2_license = _copy_verified_asset(
        request.sam2_license,
        output / "licenses/SAM2-APACHE-2.0.txt",
        expected_sha256=asset_profile["sam2_license_sha256"],
        max_bytes=1024 * 1024,
        role="SAM2 license",
    )
    resnet50_weights = _copy_verified_asset(
        request.resnet50_weights,
        output / "resnet50/resnet50-imagenet1k-v2.pth",
        expected_sha256=asset_profile["resnet50_weights_sha256"],
        max_bytes=MAX_RESNET50_WEIGHTS_BYTES,
        role="ResNet50 weights",
    )
    resnet50_license = _copy_verified_asset(
        request.resnet50_license,
        output / "licenses/TORCHVISION-BSD-3-CLAUSE.txt",
        expected_sha256=asset_profile["resnet50_license_sha256"],
        max_bytes=1024 * 1024,
        role="ResNet50 license",
    )

    rows = manifest_rows(output, excluded={"bundle.json"})
    _require(
        {str(row["path"]) for row in rows} == set(bundle_files),
        "bundle v2 payload closure mismatch before sealing",
    )
    rows_by_path = {str(row["path"]): row for row in rows}

    def generated_record(relative: str) -> dict[str, object]:
        row = rows_by_path[relative]
        return _asset_record(relative, {key: row[key] for key in ("sha256", "size_bytes", "mode_octal")})

    assets = {
        "classifier": generated_record("classifier.ubj"),
        "transforms": {
            "prototype": generated_record("prototype.npy"),
            "pca32_mean": generated_record("pca32_mean.npy"),
            "pca32_components": generated_record("pca32_components.npy"),
            "pca32_explained_variance": generated_record("pca32_explained_variance.npy"),
        },
        "sam2": {
            "architecture": asset_profile["sam2_architecture"],
            "config_locator": asset_profile["sam2_config_locator"],
            "config": _asset_record(sam2_config_bundle_path, sam2_config),
            "checkpoint": _asset_record(sam2_checkpoint_bundle_path, sam2_checkpoint),
            "license": _asset_record("licenses/SAM2-APACHE-2.0.txt", sam2_license),
        },
        "resnet50": {
            "architecture": asset_profile["resnet50_architecture"],
            "weights_identity": asset_profile["resnet50_weights_identity"],
            "embedding_dimensions": CANONICAL_EMBEDDING_DIMENSIONS,
            "weights": _asset_record("resnet50/resnet50-imagenet1k-v2.pth", resnet50_weights),
            "license": _asset_record("licenses/TORCHVISION-BSD-3-CLAUSE.txt", resnet50_license),
        },
    }
    manifest = {
        "schema": BUNDLE_SCHEMA_V2,
        "status": "PASS",
        "profile": request.profile,
        "feature_order_sha256": feature_order_sha256,
        "feature_order_hash_encoding": FEATURE_ORDER_HASH_ENCODING_V2,
        "classifier_format": "XGBOOST_UBJ",
        "threshold_method": FIXED_CANONICAL_METHOD,
        "assets": assets,
        "members": rows,
    }
    write_new_json(output / "bundle.json", manifest)
    verified = verify_model_bundle(output)
    fsync_directory(output)
    publish_directory_noreplace(output, final_output)
    verified = verify_model_bundle(final_output)
    return {
        "status": "PASS",
        "root": str(final_output),
        "bundle_sha256": verified.bundle_sha256,
        "schema": BUNDLE_SCHEMA_V2,
        "profile": request.profile,
        "feature_order_sha256": feature_order_sha256,
        "members": len(rows) + 1,
        "checkpoint_sha256": sam2_checkpoint["sha256"],
        "config_sha256": sam2_config["sha256"],
        "resnet50_weights_sha256": resnet50_weights["sha256"],
    }


def verify_model_bundle(root: Path) -> VerifiedBundle:
    absolute = root.absolute()
    info_root = absolute.lstat()
    _require(
        stat.S_ISDIR(info_root.st_mode) and not absolute.is_symlink() and absolute == absolute.resolve(strict=True),
        "model-bundle root is unsafe or contains a symlink component",
    )
    root = absolute
    bundle_path = root / "bundle.json"
    info = bundle_path.lstat()
    _require(stat.S_ISREG(info.st_mode) and not bundle_path.is_symlink(), "bundle manifest is unsafe")
    bundle_payload, bundle_snapshot = stable_file(bundle_path, max_bytes=MAX_BUNDLE_JSON_BYTES)
    _require(
        bundle_snapshot.st_dev == info.st_dev
        and bundle_snapshot.st_ino == info.st_ino
        and bundle_snapshot.st_mode == info.st_mode
        and bundle_snapshot.st_nlink == info.st_nlink
        and bundle_snapshot.st_size == info.st_size,
        "bundle manifest changed while opening",
    )
    manifest = canonical_json_value(bundle_payload, "bundle manifest")
    bundle_sha256 = hashlib.sha256(bundle_payload).hexdigest()
    _require(isinstance(manifest, dict), "bundle manifest must be an object")
    if manifest.get("schema") == BUNDLE_SCHEMA_V2:
        return _verify_model_bundle_v2(root)
    runtime_lock = science_cpu_lock_identity()
    _require(manifest.get("schema") == BUNDLE_SCHEMA and manifest.get("status") == "PASS", "bundle manifest schema/status mismatch")
    _require(
        set(manifest) == {"schema", "status", "profile", "feature_order_sha256", "classifier_format", "checkpoint", "sam2_config", "sam2_license", "members"}
        and manifest.get("classifier_format") == "XGBOOST_UBJ",
        "bundle manifest fields or classifier format mismatch",
    )
    profile = manifest.get("profile")
    _require(isinstance(profile, str) and profile in BUNDLE_ASSET_PROFILES, "bundle profile mismatch")
    asset_profile = BUNDLE_ASSET_PROFILES[profile]
    _require(manifest.get("feature_order_sha256") == FEATURE_ORDER_SHA256, "bundle feature identity mismatch")
    identity_before = tree_identity_rows(
        root,
        allowed_files={"bundle.json", *BUNDLE_FILES},
        file_size_limits=BUNDLE_SIZE_LIMITS,
        max_total_bytes=BUNDLE_MAX_TOTAL_BYTES,
    )
    observed = [
        {key: row[key] for key in ("path", "sha256", "size_bytes", "mode_octal")}
        for row in identity_before
        if row.get("type") == "file" and row.get("path") != "bundle.json"
    ]
    _require(manifest.get("members") == observed, "bundle member bytes or closure changed")
    _require({str(row["path"]) for row in observed} == set(BUNDLE_FILES), "bundle payload set mismatch")
    observed_by_path = {str(row["path"]): row for row in observed}

    def member_bytes(relative: str, *, max_bytes: int) -> bytes:
        record = observed_by_path.get(relative)
        _require(record is not None, f"bundle member is absent from manifest: {relative}")
        payload, snapshot = stable_file(root / relative, max_bytes=max_bytes)
        _require(
            len(payload) == record["size_bytes"]
            and f"{stat.S_IMODE(snapshot.st_mode):04o}" == record["mode_octal"]
            and hashlib.sha256(payload).hexdigest() == record["sha256"],
            f"bundle member differs from manifest snapshot: {relative}",
        )
        return payload

    def member_json(relative: str) -> object:
        payload = member_bytes(relative, max_bytes=MAX_BUNDLE_JSON_BYTES)
        return canonical_json_value(payload, f"bundle JSON {Path(relative).name}")

    def member_digest(relative: str, *, max_bytes: int) -> str:
        record = observed_by_path.get(relative)
        _require(record is not None, f"bundle member is absent from manifest: {relative}")
        _require(isinstance(record["size_bytes"], int) and record["size_bytes"] <= max_bytes, f"bundle member exceeds its size bound: {relative}")
        digest, snapshot = hash_file_snapshot(root / relative, require_single_link=True)
        _require(
            snapshot.st_size == record["size_bytes"]
            and f"{stat.S_IMODE(snapshot.st_mode):04o}" == record["mode_octal"]
            and digest == record["sha256"],
            f"bundle member differs from manifest snapshot: {relative}",
        )
        return digest

    classifier = member_bytes("classifier.ubj", max_bytes=MAX_CLASSIFIER_BYTES)
    _require(classifier and not classifier.startswith(b"\x80"), "classifier payload is empty or pickle-like")
    feature = member_json("feature_schema.json")
    _require(
        isinstance(feature, dict)
        and set(feature) == {"schema", "profile", "feature_order", "feature_order_sha256", "scaler", "pca", "embedding"}
        and feature.get("schema") == "compag-curation-feature-schema/v1"
        and feature.get("profile") == profile
        and feature.get("feature_order") == list(PUBLIC_FEATURE_ORDER),
        "bundle feature order or schema mismatch",
    )
    _require(feature.get("feature_order_sha256") == FEATURE_ORDER_SHA256, "bundle feature hash mismatch")
    _require(feature.get("scaler") is None and feature.get("pca") is None and feature.get("embedding") is None, "unsupported preprocessing state is present")
    imputer_raw = member_json("imputer.json")
    _require(
        isinstance(imputer_raw, dict)
        and set(imputer_raw) == {"schema", "strategy", "feature_order", "statistics"}
        and imputer_raw.get("schema") == "compag-curation-imputer/v1"
        and imputer_raw.get("strategy") == "median",
        "bundle imputer schema mismatch",
    )
    _require(imputer_raw.get("feature_order") == list(PUBLIC_FEATURE_ORDER), "bundle imputer feature order mismatch")
    statistics = imputer_raw.get("statistics")
    _require(isinstance(statistics, dict) and set(statistics) == set(PUBLIC_FEATURE_ORDER), "bundle imputer fields mismatch")
    imputer = tuple(_finite(statistics[name], f"imputer.{name}") for name in PUBLIC_FEATURE_ORDER)
    threshold_raw = member_json("threshold.json")
    _require(
        isinstance(threshold_raw, dict)
        and set(threshold_raw) == {"schema", "selection_partition", "value"}
        and threshold_raw.get("schema") == "compag-curation-threshold/v1"
        and threshold_raw.get("selection_partition") == "validation",
        "bundle threshold provenance mismatch",
    )
    threshold = _finite(threshold_raw.get("value"), "threshold")
    _require(0.0 <= threshold <= 1.0, "bundle threshold is outside [0,1]")
    class_map = member_json("class_map.json")
    _require(class_map == {"schema": "compag-curation-class-map/v1", "negative": 0, "positive": 1}, "bundle class map mismatch")
    normalized = member_json("normalized_config.json")
    preprocessing = member_json("preprocessing.json")
    proposal = member_json("proposal_config.json")
    compatibility = member_json("compatibility.json")
    provenance = member_json("provenance.json")
    for role, value in (("normalized config", normalized), ("preprocessing", preprocessing), ("proposal config", proposal), ("compatibility", compatibility), ("provenance", provenance)):
        _require(isinstance(value, dict), f"bundle {role} must be an object")
        encoded = canonical_json_bytes(value).decode("ascii")
        _require(
            re.search(r"(?:/home/|/root(?:/|\b)|/Users/|/mnt/[a-z]/Users/|fi" + r"le://|[A-Za-z]:\\|\\\\|wsl\.localhost)", encoded, flags=re.IGNORECASE) is None,
            f"bundle {role} contains a private locator",
        )
    _require(
        set(normalized) == {"schema", "profile", "device", "seed", "checkpoint_sha256", "sam2_config_sha256", "feature_order_sha256"}
        and normalized.get("schema") == "compag-curation-normalized-public-config/v1"
        and normalized.get("profile") == profile
        and normalized.get("device") == "cpu"
        and normalized.get("seed") == 42
        and normalized.get("feature_order_sha256") == FEATURE_ORDER_SHA256,
        "bundle normalized configuration contract mismatch",
    )
    _require(
        preprocessing == {
            "schema": "compag-curation-preprocessing/v1",
            "tile_size": 512,
            "tile_overlap": 64,
            "image_color": "OpenCV BGR decoded then SAM2 RGB and OpenCV uint8 LAB features",
            "exif_orientation": "absent_or_1",
        },
        "bundle preprocessing contract mismatch",
    )
    proposal_fields = {
        "schema", "profile", "sam2_config_locator", "sam2_checkpoint_sha256", "sam2_config_sha256",
        "source_commit", "device", "points_per_side", "points_per_batch", "pred_iou_threshold",
        "stability_threshold", "stability_score_offset", "box_nms_threshold", "crop_layers",
        "crop_nms_threshold", "crop_overlap_ratio", "crop_points_downscale", "min_mask_region_area",
        "multiscale", "multiscale_iou", "sam_score_decimal_places", "proposal_identity",
        "cpu_runtime_policy", "max_proposals_per_tile", "min_mask_area", "feature_order_sha256",
    }
    _require(set(proposal) == proposal_fields, "bundle proposal configuration fields mismatch")
    _require(
        proposal.get("schema") == "compag-curation-proposal-config/v2"
        and proposal.get("profile") == profile
        and proposal.get("sam2_config_locator") == asset_profile["locator"]
        and proposal.get("source_commit") == "2b90b9f5ceec907a1c18123530e92e794ad901a4"
        and proposal.get("device") == "cpu"
        and proposal.get("stability_score_offset") == 1.0
        and proposal.get("box_nms_threshold") == 0.7
        and proposal.get("crop_layers") == 0
        and proposal.get("crop_nms_threshold") == 0.7
        and proposal.get("crop_overlap_ratio") == 512 / 1500
        and proposal.get("crop_points_downscale") == 1
        and proposal.get("min_mask_region_area") == 0
        and proposal.get("multiscale") == [1.0]
        and proposal.get("multiscale_iou") == 0.75
        and proposal.get("sam_score_decimal_places") == 6
        and proposal.get("proposal_identity") == "geometry-mask-v2"
        and proposal.get("cpu_runtime_policy") == science_cpu_runtime_policy()
        and proposal.get("feature_order_sha256") == FEATURE_ORDER_SHA256,
        "bundle fixed proposal configuration mismatch",
    )
    for key, lower, upper in (
        ("points_per_side", 2, 64), ("points_per_batch", 1, 512),
        ("max_proposals_per_tile", 2, 500), ("min_mask_area", 0, 512 * 512),
    ):
        value = proposal.get(key)
        _require(isinstance(value, int) and not isinstance(value, bool) and lower <= value <= upper, f"bundle {key} is invalid")
    for key in ("pred_iou_threshold", "stability_threshold"):
        _require(0.0 <= _finite(proposal.get(key), key) <= 1.0, f"bundle {key} is invalid")
    versions = compatibility.get("versions")
    _require(
        set(compatibility) == {
            "schema", "python", "versions", "lock_sha256", "cpu_runtime_policy_sha256",
            "feature_math", "classifier",
        }
        and compatibility.get("schema") == "compag-curation-model-compatibility/v2"
        and compatibility.get("python") == SCIENCE_CPU_PYTHON
        and isinstance(versions, dict)
        and set(versions) == COMPATIBILITY_PACKAGES
        and versions == SCIENCE_CPU_VERSIONS
        and compatibility.get("lock_sha256") == SCIENCE_CPU_LOCK_SHA256
        and compatibility.get("cpu_runtime_policy_sha256") == science_cpu_runtime_policy()["sha256"]
        and runtime_lock.get("sha256") == SCIENCE_CPU_LOCK_SHA256
        and compatibility.get("feature_math") == "gate-core-balanced-v1"
        and compatibility.get("classifier") == "xgboost-ubj",
        "bundle dependency compatibility contract mismatch",
    )
    provenance_fields = {
        "schema", "profile", "seed", "reviewed_table_sha256", "proposal_table_sha256",
        "split_manifest_sha256", "feature_order_sha256", "training_rows_before_smote",
        "training_rows_after_smote", "smote_applied", "validation_rows", "test_rows",
        "paper_result_reproduction",
    }
    _require(
        set(provenance) == provenance_fields
        and provenance.get("schema") == "compag-curation-model-provenance/v1"
        and provenance.get("profile") == profile
        and provenance.get("seed") == 42
        and provenance.get("feature_order_sha256") == FEATURE_ORDER_SHA256
        and provenance.get("paper_result_reproduction") == "NOT_CLAIMED"
        and isinstance(provenance.get("smote_applied"), bool),
        "bundle provenance contract mismatch",
    )
    for key in ("reviewed_table_sha256", "proposal_table_sha256", "split_manifest_sha256"):
        _require(re.fullmatch(r"[0-9a-f]{64}", str(provenance.get(key))) is not None, f"bundle provenance {key} is invalid")
    for key in ("training_rows_before_smote", "training_rows_after_smote", "validation_rows", "test_rows"):
        value = provenance.get(key)
        _require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"bundle provenance {key} is invalid")
    _require(
        int(provenance["training_rows_after_smote"]) >= int(provenance["training_rows_before_smote"]),
        "bundle SMOTE row counts are inconsistent",
    )
    checkpoint = root / "sam2/checkpoints/model.pt"
    config = root / "sam2/configs/sam2.1/model.yaml"
    license_path = root / "licenses/SAM2-APACHE-2.0.txt"
    checkpoint_record = manifest.get("checkpoint")
    config_record = manifest.get("sam2_config")
    license_record = manifest.get("sam2_license")
    _require(
        all(isinstance(value, dict) and set(value) == {"sha256", "size_bytes", "mode_octal"} for value in (checkpoint_record, config_record, license_record)),
        "bundle asset records are invalid",
    )
    portable_by_path = {
        path: {key: row[key] for key in ("sha256", "size_bytes", "mode_octal")}
        for path, row in observed_by_path.items()
    }
    _require(checkpoint_record == portable_by_path["sam2/checkpoints/model.pt"], "bundle checkpoint record differs from the member manifest")
    _require(config_record == portable_by_path["sam2/configs/sam2.1/model.yaml"], "bundle configuration record differs from the member manifest")
    _require(license_record == portable_by_path["licenses/SAM2-APACHE-2.0.txt"], "bundle license record differs from the member manifest")
    manifest_encoded = canonical_json_bytes(manifest).decode("ascii")
    _require(
        re.search(r"(?:/home/|/root(?:/|\b)|/Users/|/mnt/[a-z]/Users/|fi" + r"le://|[A-Za-z]:\\|\\\\|wsl\.localhost)", manifest_encoded, flags=re.IGNORECASE) is None,
        "bundle manifest contains a private locator",
    )
    _require(member_digest("sam2/checkpoints/model.pt", max_bytes=MAX_SAM_CHECKPOINT_BYTES) == checkpoint_record.get("sha256"), "bundle checkpoint identity mismatch")
    _require(hashlib.sha256(member_bytes("sam2/configs/sam2.1/model.yaml", max_bytes=MAX_BUNDLE_JSON_BYTES)).hexdigest() == config_record.get("sha256"), "bundle SAM2 configuration identity mismatch")
    _require(license_record.get("sha256") == SAM2_LICENSE_SHA256, "bundle SAM2 license identity mismatch")
    _require(hashlib.sha256(member_bytes("licenses/SAM2-APACHE-2.0.txt", max_bytes=MAX_BUNDLE_JSON_BYTES)).hexdigest() == SAM2_LICENSE_SHA256, "bundle SAM2 license payload mismatch")
    _require(proposal.get("sam2_checkpoint_sha256") == checkpoint_record.get("sha256"), "proposal configuration checkpoint hash mismatch")
    _require(proposal.get("sam2_config_sha256") == config_record.get("sha256"), "proposal configuration SAM2 config hash mismatch")
    _require(checkpoint_record.get("sha256") == asset_profile["checkpoint_sha256"], "bundle checkpoint differs from its registered profile asset")
    _require(config_record.get("sha256") == asset_profile["config_sha256"], "bundle configuration differs from its registered profile asset")
    _require(normalized.get("checkpoint_sha256") == checkpoint_record.get("sha256"), "normalized checkpoint identity mismatch")
    _require(normalized.get("sam2_config_sha256") == config_record.get("sha256"), "normalized SAM2 config identity mismatch")
    bundle_payload_after, bundle_snapshot_after = stable_file(bundle_path, max_bytes=MAX_BUNDLE_JSON_BYTES)
    _require(
        bundle_payload_after == bundle_payload
        and bundle_snapshot_after.st_dev == bundle_snapshot.st_dev
        and bundle_snapshot_after.st_ino == bundle_snapshot.st_ino
        and bundle_snapshot_after.st_mode == bundle_snapshot.st_mode
        and bundle_snapshot_after.st_nlink == bundle_snapshot.st_nlink
        and bundle_snapshot_after.st_size == bundle_snapshot.st_size
        and tree_identity_rows(
            root,
            allowed_files={"bundle.json", *BUNDLE_FILES},
            file_size_limits=BUNDLE_SIZE_LIMITS,
            max_total_bytes=BUNDLE_MAX_TOTAL_BYTES,
        ) == identity_before,
        "model bundle changed during semantic verification",
    )
    # Parse the native model format only after every byte, schema, compatibility,
    # provenance, and registered-asset check has passed.
    try:
        import xgboost as xgb

        parsed_classifier = xgb.Booster()
        parsed_classifier.load_model(bytearray(classifier))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PublicIOError("classifier.ubj is not a loadable XGBoost UBJ model") from exc
    _require(tuple(parsed_classifier.feature_names or ()) == PUBLIC_FEATURE_ORDER, "classifier UBJ feature names differ from the public feature order")
    _require(
        tree_identity_rows(
            root,
            allowed_files={"bundle.json", *BUNDLE_FILES},
            file_size_limits=BUNDLE_SIZE_LIMITS,
            max_total_bytes=BUNDLE_MAX_TOTAL_BYTES,
        ) == identity_before,
        "model bundle changed while parsing the classifier",
    )
    return VerifiedBundle(
        root=root,
        bundle_sha256=bundle_sha256,
        classifier=classifier,
        imputer=imputer,
        threshold=threshold,
        normalized_config=normalized,
        preprocessing=preprocessing,
        proposal_config=proposal,
        compatibility=compatibility,
        provenance=provenance,
        checkpoint=checkpoint,
        sam2_config=config,
        manifest=manifest,
        tree_identity=tuple(identity_before),
    )


def _verify_model_bundle_v2(root: Path) -> VerifiedBundle:
    absolute = root.absolute()
    info_root = absolute.lstat()
    _require(
        stat.S_ISDIR(info_root.st_mode)
        and not absolute.is_symlink()
        and absolute == absolute.resolve(strict=True),
        "model-bundle root is unsafe or contains a symlink component",
    )
    root = absolute
    bundle_path = root / "bundle.json"
    info = bundle_path.lstat()
    _require(stat.S_ISREG(info.st_mode) and not bundle_path.is_symlink(), "bundle manifest is unsafe")
    bundle_payload, bundle_snapshot = stable_file(bundle_path, max_bytes=MAX_BUNDLE_JSON_BYTES)
    _require(
        bundle_snapshot.st_dev == info.st_dev
        and bundle_snapshot.st_ino == info.st_ino
        and bundle_snapshot.st_mode == info.st_mode
        and bundle_snapshot.st_nlink == info.st_nlink
        and bundle_snapshot.st_size == info.st_size,
        "bundle manifest changed while opening",
    )
    manifest = canonical_json_value(bundle_payload, "bundle manifest")
    bundle_sha256 = hashlib.sha256(bundle_payload).hexdigest()
    manifest_fields = {
        "schema",
        "status",
        "profile",
        "feature_order_sha256",
        "feature_order_hash_encoding",
        "classifier_format",
        "threshold_method",
        "assets",
        "members",
    }
    _require(
        isinstance(manifest, dict)
        and set(manifest) == manifest_fields
        and manifest.get("schema") == BUNDLE_SCHEMA_V2
        and manifest.get("status") == "PASS"
        and manifest.get("classifier_format") == "XGBOOST_UBJ"
        and manifest.get("threshold_method") == FIXED_CANONICAL_METHOD
        and manifest.get("feature_order_hash_encoding") == FEATURE_ORDER_HASH_ENCODING_V2,
        "bundle v2 manifest fields or fixed methods mismatch",
    )
    profile = manifest.get("profile")
    _require(isinstance(profile, str) and profile in BUNDLE_V2_ASSET_PROFILES, "bundle v2 profile mismatch")
    runtime_contract = _canonical_runtime_contract(profile)
    (
        asset_profile,
        bundle_files,
        size_limits,
        max_total_bytes,
        sam2_config_bundle_path,
        sam2_checkpoint_bundle_path,
    ) = _bundle_v2_profile_contract(profile)
    feature_order_sha256 = manifest.get("feature_order_sha256")
    _require(
        feature_order_sha256 == CANONICAL_FEATURE_ORDER_SHA256,
        "bundle v2 canonical feature-order hash is invalid",
    )

    identity_before = tree_identity_rows(
        root,
        allowed_files={"bundle.json", *bundle_files},
        file_size_limits=size_limits,
        max_total_bytes=max_total_bytes,
    )
    observed = [
        {key: row[key] for key in ("path", "sha256", "size_bytes", "mode_octal")}
        for row in identity_before
        if row.get("type") == "file" and row.get("path") != "bundle.json"
    ]
    _require(manifest.get("members") == observed, "bundle v2 member bytes or closure changed")
    _require(
        {str(row["path"]) for row in observed} == set(bundle_files),
        "bundle v2 payload set mismatch",
    )
    observed_by_path = {str(row["path"]): row for row in observed}

    def member_bytes(relative: str, *, max_bytes: int) -> bytes:
        record = observed_by_path.get(relative)
        _require(record is not None, f"bundle v2 member is absent from manifest: {relative}")
        payload, snapshot = stable_file(root / relative, max_bytes=max_bytes)
        _require(
            len(payload) == record["size_bytes"]
            and f"{stat.S_IMODE(snapshot.st_mode):04o}" == record["mode_octal"]
            and hashlib.sha256(payload).hexdigest() == record["sha256"],
            f"bundle v2 member differs from manifest snapshot: {relative}",
        )
        return payload

    def member_json(relative: str) -> object:
        return canonical_json_value(
            member_bytes(relative, max_bytes=MAX_BUNDLE_JSON_BYTES),
            f"bundle JSON {Path(relative).name}",
        )

    def member_digest(relative: str, *, max_bytes: int) -> str:
        record = observed_by_path.get(relative)
        _require(record is not None, f"bundle v2 member is absent from manifest: {relative}")
        _require(
            isinstance(record["size_bytes"], int) and record["size_bytes"] <= max_bytes,
            f"bundle v2 member exceeds its size bound: {relative}",
        )
        digest, snapshot = hash_file_snapshot(root / relative, require_single_link=True)
        _require(
            snapshot.st_size == record["size_bytes"]
            and f"{stat.S_IMODE(snapshot.st_mode):04o}" == record["mode_octal"]
            and digest == record["sha256"],
            f"bundle v2 member differs from manifest snapshot: {relative}",
        )
        return digest

    classifier = member_bytes("classifier.ubj", max_bytes=MAX_CLASSIFIER_BYTES)
    _require(classifier and not classifier.startswith(b"\x80"), "classifier payload is empty or pickle-like")
    feature = member_json("feature_schema.json")
    feature_fields = {
        "schema",
        "profile",
        "feature_order",
        "feature_order_sha256",
        "feature_order_hash_encoding",
        "imputer",
        "prototype",
        "pca",
        "scaler",
        "embedding",
    }
    _require(
        isinstance(feature, dict)
        and set(feature) == feature_fields
        and feature.get("schema") == "compag-curation-feature-schema/v2"
        and feature.get("profile") == profile
        and feature.get("feature_order_hash_encoding") == FEATURE_ORDER_HASH_ENCODING_V2
        and feature.get("imputer") == "imputer.json"
        and feature.get("prototype") == "prototype.npy"
        and feature.get("pca")
        == {
            "mean": "pca32_mean.npy",
            "components": "pca32_components.npy",
            "explained_variance": "pca32_explained_variance.npy",
        }
        and feature.get("scaler") is None,
        "bundle v2 feature schema mismatch",
    )
    feature_order_raw = feature.get("feature_order")
    _require(isinstance(feature_order_raw, list), "bundle v2 feature order must be an array")
    feature_order = _feature_order_v2(feature_order_raw)
    computed_feature_hash = compute_feature_order_sha256(feature_order)
    _require(
        feature_order == CANONICAL_FEATURE_ORDER
        and len(feature_order) == 93
        and computed_feature_hash
        == feature_order_sha256
        == feature.get("feature_order_sha256")
        == CANONICAL_FEATURE_ORDER_SHA256,
        "bundle v2 canonical 93-feature identity mismatch",
    )
    embedding = feature.get("embedding")
    _require(
        isinstance(embedding, dict)
        and set(embedding) == {"architecture", "weights_identity", "weights", "dimensions"}
        and embedding.get("architecture") == asset_profile["resnet50_architecture"]
        and embedding.get("weights_identity") == asset_profile["resnet50_weights_identity"]
        and embedding.get("weights") == "resnet50/resnet50-imagenet1k-v2.pth"
        and embedding.get("dimensions") == CANONICAL_EMBEDDING_DIMENSIONS,
        "bundle v2 ResNet50 feature identity mismatch",
    )

    imputer_raw = member_json("imputer.json")
    _require(
        isinstance(imputer_raw, dict)
        and set(imputer_raw) == {"schema", "strategy", "feature_order", "feature_order_sha256", "statistics"}
        and imputer_raw.get("schema") == "compag-curation-imputer/v2"
        and imputer_raw.get("strategy") == "median"
        and imputer_raw.get("feature_order") == list(feature_order)
        and imputer_raw.get("feature_order_sha256") == feature_order_sha256,
        "bundle v2 imputer schema or feature identity mismatch",
    )
    statistics = imputer_raw.get("statistics")
    _require(isinstance(statistics, dict) and set(statistics) == set(feature_order), "bundle v2 imputer fields mismatch")
    imputer = tuple(_finite(statistics[name], f"imputer.{name}") for name in feature_order)

    prototype_npy = member_bytes("prototype.npy", max_bytes=MAX_NPY_STATE_BYTES)
    canonical_prototype_npy, prototype = _canonical_float32_npy(
        prototype_npy,
        shape=(CANONICAL_EMBEDDING_DIMENSIONS,),
        role="prototype",
    )
    _require(prototype_npy == canonical_prototype_npy, "prototype NPY bytes are not canonical")
    _require(
        math.isclose(math.sqrt(sum(value * value for value in prototype)), 1.0, rel_tol=0.0, abs_tol=1e-5),
        "canonical prototype must be unit L2 normalized",
    )
    pca_mean_npy = member_bytes("pca32_mean.npy", max_bytes=MAX_NPY_STATE_BYTES)
    canonical_pca_mean_npy, pca_mean = _canonical_float32_npy(
        pca_mean_npy,
        shape=(CANONICAL_EMBEDDING_DIMENSIONS,),
        role="pca.mean",
    )
    _require(pca_mean_npy == canonical_pca_mean_npy, "PCA mean NPY bytes are not canonical")
    pca_components_npy = member_bytes("pca32_components.npy", max_bytes=MAX_NPY_STATE_BYTES)
    canonical_pca_components_npy, pca_components_flat = _canonical_float32_npy(
        pca_components_npy,
        shape=(CANONICAL_PCA_DIMENSIONS, CANONICAL_EMBEDDING_DIMENSIONS),
        role="pca.components",
    )
    _require(
        pca_components_npy == canonical_pca_components_npy,
        "PCA components NPY bytes are not canonical",
    )
    pca_components = tuple(
        pca_components_flat[
            index * CANONICAL_EMBEDDING_DIMENSIONS : (index + 1) * CANONICAL_EMBEDDING_DIMENSIONS
        ]
        for index in range(CANONICAL_PCA_DIMENSIONS)
    )
    pca_explained_variance_npy = member_bytes(
        "pca32_explained_variance.npy",
        max_bytes=MAX_NPY_STATE_BYTES,
    )
    canonical_pca_explained_variance_npy, pca_explained_variance = _canonical_float32_npy(
        pca_explained_variance_npy,
        shape=(CANONICAL_PCA_DIMENSIONS,),
        role="pca.explained_variance",
    )
    _require(
        pca_explained_variance_npy == canonical_pca_explained_variance_npy,
        "PCA explained variance NPY bytes are not canonical",
    )
    _require(all(value >= 0.0 for value in pca_explained_variance), "PCA explained variance must be nonnegative")
    threshold_raw = member_json("threshold.json")
    _require(
        threshold_raw
        == {
            "schema": "compag-curation-threshold/v2",
            "method": FIXED_CANONICAL_METHOD,
            "value": CANONICAL_DECISION_THRESHOLD,
        },
        "bundle v2 fixed canonical threshold mismatch",
    )
    class_map = member_json("class_map.json")
    _require(
        class_map == {"schema": "compag-curation-class-map/v1", "negative": 0, "positive": 1},
        "bundle v2 class map mismatch",
    )

    def config_member(relative: str, role: str) -> dict[str, object]:
        raw = member_json(relative)
        _require(
            isinstance(raw, dict)
            and set(raw) == {"schema", "role", "profile", "feature_order_sha256", "config"}
            and raw.get("schema") == "compag-curation-bundle-config/v2"
            and raw.get("role") == role
            and raw.get("profile") == profile
            and raw.get("feature_order_sha256") == feature_order_sha256
            and isinstance(raw.get("config"), dict),
            f"bundle v2 {role} config wrapper mismatch",
        )
        return _metadata_mapping(raw["config"], f"bundle {role} config")

    normalized = config_member("normalized_config.json", "normalized")
    preprocessing = config_member("preprocessing.json", "preprocessing")
    proposal = config_member("proposal_config.json", "proposal")
    feature_config = config_member("feature_config.json", "features")
    training_config = config_member("training_config.json", "training")
    expected_configs = _canonical_config_contracts(
        profile=profile,
        asset_profile=asset_profile,
        feature_order_sha256=feature_order_sha256,
    )
    for role, observed in (
        ("normalized", normalized),
        ("preprocessing", preprocessing),
        ("proposal", proposal),
        ("features", feature_config),
        ("training", training_config),
    ):
        _exact_json_object(observed, expected_configs[role], f"bundle {role} config")

    compatibility = member_json("compatibility.json")
    runtime_policy_field = runtime_contract.policy_field
    compatibility_fields = {
        "schema",
        "profile",
        "feature_order_sha256",
        "classifier_format",
        "threshold_method",
        "python",
        "versions",
        "lock_sha256",
        runtime_policy_field,
        "implementation_identity",
        "sam2_architecture",
        "resnet50_weights_identity",
        "embedding_dimensions",
        "pca_components",
        "details",
    }
    _require(
        isinstance(compatibility, dict)
        and set(compatibility) == compatibility_fields
        and compatibility.get("schema") == "compag-curation-model-compatibility/v3"
        and compatibility.get("profile") == profile
        and compatibility.get("feature_order_sha256") == feature_order_sha256
        and compatibility.get("classifier_format") == "XGBOOST_UBJ"
        and compatibility.get("threshold_method") == FIXED_CANONICAL_METHOD
        and compatibility.get("python") == runtime_contract.python
        and compatibility.get("versions") == runtime_contract.versions
        and compatibility.get("lock_sha256") == runtime_contract.lock_sha256
        and compatibility.get(runtime_policy_field) == runtime_contract.policy_sha256
        and compatibility.get("implementation_identity")
        == _expected_package_implementation_identity(
            profile,
            expected_distribution_version=runtime_contract.distribution_version,
        )
        and compatibility.get("sam2_architecture") == asset_profile["sam2_architecture"]
        and compatibility.get("resnet50_weights_identity") == asset_profile["resnet50_weights_identity"]
        and compatibility.get("embedding_dimensions") == CANONICAL_EMBEDDING_DIMENSIONS
        and compatibility.get("pca_components") == CANONICAL_PCA_DIMENSIONS
        and isinstance(compatibility.get("details"), dict),
        "bundle v2 dependency compatibility contract mismatch",
    )
    compatibility_details = _v2_metadata_details(
        compatibility["details"],
        "bundle compatibility details",
        feature_order_sha256=feature_order_sha256,
    )
    provenance = member_json("provenance.json")
    _require(
        isinstance(provenance, dict)
        and set(provenance) == {"schema", "profile", "feature_order_sha256", "threshold_method", "details"}
        and provenance.get("schema") == "compag-curation-model-provenance/v2"
        and provenance.get("profile") == profile
        and provenance.get("feature_order_sha256") == feature_order_sha256
        and provenance.get("threshold_method") == FIXED_CANONICAL_METHOD
        and isinstance(provenance.get("details"), dict),
        "bundle v2 provenance contract mismatch",
    )
    provenance_details = _v2_metadata_details(
        provenance["details"],
        "bundle provenance details",
        feature_order_sha256=feature_order_sha256,
    )
    _validate_v2_details(
        compatibility_details,
        provenance_details,
        profile=profile,
        feature_order_sha256=feature_order_sha256,
    )

    assets = manifest.get("assets")
    _require(isinstance(assets, dict) and set(assets) == {"classifier", "transforms", "sam2", "resnet50"}, "bundle v2 asset index mismatch")

    def require_asset_record(value: object, relative: str, role: str) -> Mapping[str, object]:
        expected = {"path": relative, **observed_by_path[relative]}
        _require(value == expected, f"bundle v2 {role} record differs from the member manifest")
        _require(isinstance(value, dict), f"bundle v2 {role} record is invalid")
        return value

    require_asset_record(assets.get("classifier"), "classifier.ubj", "classifier")
    transforms = assets.get("transforms")
    _require(
        isinstance(transforms, dict)
        and set(transforms)
        == {"prototype", "pca32_mean", "pca32_components", "pca32_explained_variance"},
        "bundle v2 transform asset index mismatch",
    )
    require_asset_record(transforms.get("prototype"), "prototype.npy", "prototype")
    require_asset_record(transforms.get("pca32_mean"), "pca32_mean.npy", "PCA32 mean")
    require_asset_record(
        transforms.get("pca32_components"),
        "pca32_components.npy",
        "PCA32 components",
    )
    require_asset_record(
        transforms.get("pca32_explained_variance"),
        "pca32_explained_variance.npy",
        "PCA32 explained variance",
    )
    sam2 = assets.get("sam2")
    _require(
        isinstance(sam2, dict)
        and set(sam2) == {"architecture", "config_locator", "config", "checkpoint", "license"}
        and sam2.get("architecture") == asset_profile["sam2_architecture"]
        and sam2.get("config_locator") == asset_profile["sam2_config_locator"],
        "bundle v2 SAM2 asset identity mismatch",
    )
    sam2_config_record = require_asset_record(
        sam2.get("config"),
        sam2_config_bundle_path,
        "SAM2 config",
    )
    sam2_checkpoint_record = require_asset_record(
        sam2.get("checkpoint"),
        sam2_checkpoint_bundle_path,
        "SAM2 checkpoint",
    )
    sam2_license_record = require_asset_record(
        sam2.get("license"),
        "licenses/SAM2-APACHE-2.0.txt",
        "SAM2 license",
    )
    resnet50 = assets.get("resnet50")
    _require(
        isinstance(resnet50, dict)
        and set(resnet50) == {"architecture", "weights_identity", "embedding_dimensions", "weights", "license"}
        and resnet50.get("architecture") == asset_profile["resnet50_architecture"]
        and resnet50.get("weights_identity") == asset_profile["resnet50_weights_identity"]
        and resnet50.get("embedding_dimensions") == CANONICAL_EMBEDDING_DIMENSIONS,
        "bundle v2 ResNet50 asset identity mismatch",
    )
    resnet50_weights_record = require_asset_record(
        resnet50.get("weights"),
        "resnet50/resnet50-imagenet1k-v2.pth",
        "ResNet50 weights",
    )
    resnet50_license_record = require_asset_record(
        resnet50.get("license"),
        "licenses/TORCHVISION-BSD-3-CLAUSE.txt",
        "ResNet50 license",
    )
    _require(sam2_config_record.get("sha256") == asset_profile["sam2_config_sha256"], "SAM2 config identity mismatch")
    _require(sam2_checkpoint_record.get("sha256") == asset_profile["sam2_checkpoint_sha256"], "SAM2 checkpoint identity mismatch")
    _require(sam2_license_record.get("sha256") == asset_profile["sam2_license_sha256"], "SAM2 license identity mismatch")
    _require(resnet50_weights_record.get("sha256") == asset_profile["resnet50_weights_sha256"], "ResNet50 weights identity mismatch")
    _require(resnet50_license_record.get("sha256") == asset_profile["resnet50_license_sha256"], "ResNet50 license identity mismatch")
    manifest_encoded = canonical_json_bytes(manifest).decode("ascii")
    _require(
        re.search(
            r"(?:/home/|/root(?:/|\b)|/Users/|/mnt/[a-z]/Users/|fi" + r"le://|[A-Za-z]:\\|\\\\|wsl\.localhost)",
            manifest_encoded,
            flags=re.IGNORECASE,
        )
        is None,
        "bundle v2 manifest contains a private locator",
    )
    _require(
        member_digest(sam2_checkpoint_bundle_path, max_bytes=MAX_SAM_CHECKPOINT_BYTES)
        == asset_profile["sam2_checkpoint_sha256"],
        "bundle v2 SAM2 checkpoint hash mismatch",
    )
    _require(
        hashlib.sha256(
            member_bytes(sam2_config_bundle_path, max_bytes=MAX_BUNDLE_JSON_BYTES)
        ).hexdigest()
        == asset_profile["sam2_config_sha256"],
        "bundle v2 SAM2 config hash mismatch",
    )
    _require(
        member_digest("resnet50/resnet50-imagenet1k-v2.pth", max_bytes=MAX_RESNET50_WEIGHTS_BYTES)
        == asset_profile["resnet50_weights_sha256"],
        "bundle v2 ResNet50 weights hash mismatch",
    )
    _require(
        hashlib.sha256(
            member_bytes("licenses/SAM2-APACHE-2.0.txt", max_bytes=MAX_BUNDLE_JSON_BYTES)
        ).hexdigest()
        == asset_profile["sam2_license_sha256"],
        "bundle v2 SAM2 license payload mismatch",
    )
    _require(
        hashlib.sha256(
            member_bytes("licenses/TORCHVISION-BSD-3-CLAUSE.txt", max_bytes=MAX_BUNDLE_JSON_BYTES)
        ).hexdigest()
        == asset_profile["resnet50_license_sha256"],
        "bundle v2 ResNet50 license payload mismatch",
    )
    bundle_payload_after, bundle_snapshot_after = stable_file(bundle_path, max_bytes=MAX_BUNDLE_JSON_BYTES)
    _require(
        bundle_payload_after == bundle_payload
        and bundle_snapshot_after.st_dev == bundle_snapshot.st_dev
        and bundle_snapshot_after.st_ino == bundle_snapshot.st_ino
        and bundle_snapshot_after.st_mode == bundle_snapshot.st_mode
        and bundle_snapshot_after.st_nlink == bundle_snapshot.st_nlink
        and bundle_snapshot_after.st_size == bundle_snapshot.st_size
        and tree_identity_rows(
            root,
            allowed_files={"bundle.json", *bundle_files},
            file_size_limits=size_limits,
            max_total_bytes=max_total_bytes,
        )
        == identity_before,
        "model bundle v2 changed during semantic verification",
    )
    _validate_canonical_ubj(classifier)
    _require(
        tree_identity_rows(
            root,
            allowed_files={"bundle.json", *bundle_files},
            file_size_limits=size_limits,
            max_total_bytes=max_total_bytes,
        )
        == identity_before,
        "model bundle v2 changed while parsing the classifier",
    )
    return VerifiedBundle(
        root=root,
        bundle_sha256=bundle_sha256,
        classifier=classifier,
        imputer=imputer,
        threshold=CANONICAL_DECISION_THRESHOLD,
        normalized_config=normalized,
        preprocessing=preprocessing,
        proposal_config=proposal,
        compatibility=compatibility,
        provenance=provenance,
        checkpoint=root / sam2_checkpoint_bundle_path,
        sam2_config=root / sam2_config_bundle_path,
        manifest=manifest,
        tree_identity=tuple(identity_before),
        schema=BUNDLE_SCHEMA_V2,
        profile=profile,
        feature_order=feature_order,
        feature_order_sha256=feature_order_sha256,
        prototype=prototype,
        pca_mean=pca_mean,
        pca_components=pca_components,
        pca_explained_variance=pca_explained_variance,
        prototype_npy=prototype_npy,
        pca_mean_npy=pca_mean_npy,
        pca_components_npy=pca_components_npy,
        pca_explained_variance_npy=pca_explained_variance_npy,
        resnet50_weights=root / "resnet50/resnet50-imagenet1k-v2.pth",
        feature_config=feature_config,
        training_config=training_config,
        assets=assets,
    )


def feature_matrix(
    rows: Sequence[Mapping[str, object]],
    imputer: Sequence[float],
    *,
    feature_order: Sequence[str] = PUBLIC_FEATURE_ORDER,
) -> list[list[float]]:
    order = _feature_order_v2(feature_order)
    _require(len(imputer) == len(order), "imputer vector length mismatch")
    matrix: list[list[float]] = []
    for index, row in enumerate(rows):
        _require(set(order) <= set(row), f"feature row {index} is missing required columns")
        values = []
        for position, name in enumerate(order):
            raw = row[name]
            if raw is None or (isinstance(raw, str) and raw == ""):
                number = float(imputer[position])
            elif isinstance(raw, str):
                _require(raw.strip() == raw, f"row {index}.{name} contains surrounding whitespace")
                try:
                    number = float(raw)
                except ValueError as exc:
                    raise PublicIOError(f"row {index}.{name} is not numeric") from exc
                _require(math.isfinite(number), f"row {index}.{name} must be finite")
            else:
                number = _finite(raw, f"row {index}.{name}")
            values.append(number)
        matrix.append(values)
    _require(matrix, "feature matrix is empty")
    return matrix
