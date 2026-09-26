"""One-image proposal, feature, and r92 scoring for transfer-baseline AL.

The published r92 preset is intentionally not promoted to a canonical v2
parent bundle.  This module supplies only the adapter needed to execute the
existing canonical Full proposal/feature/inference engine for exactly one
image, then seals an explicit transfer-scorer receipt beside the ordinary
Stage-60-compatible files.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from compag_curation.assets import asset_ids_for_profile, asset_registry
from compag_curation.model_bundle import BUNDLE_SCHEMA_V2
from compag_curation.public_backend import (
    require_science_dependencies,
    revalidate_science_gpu_dependencies,
)
from compag_curation.public_config import (
    CANONICAL_GPU_PROFILE,
    PublicProjectConfig,
    check_local_assets,
    load_public_config,
)
from compag_curation.public_io import (
    PublicIOError,
    compact_json_sha256,
    fsync_directory,
    publish_directory_noreplace,
    sha256_file,
    write_new_json,
)
from compag_curation.review.published_model import (
    PRESET_ID,
    PublishedModelTransferAssets,
    load_published_model_transfer_assets,
)

from .service import _infer_canonical_bundle_with_dependencies
from .spec import (
    CANONICAL_DECISION_THRESHOLD,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
)


TRANSFER_IMAGE_INFERENCE_SCHEMA = (
    "compag-curation-r92-transfer-image-inference/v1"
)
TRANSFER_WORKFLOW = "SINGLE_NEW_IMAGE_POST_R92_TRANSFER_V1"
TRANSFER_LINEAGE_POLICY = "POST_R92_REVIEWED_TRANSFER_BASELINE"
TRANSFER_REPRODUCTION_CLAIM = "NOT_R92_REPRODUCTION"
_TRANSFER_SCORER_IDENTITY_SCHEMA = (
    "compag-curation-r92-transfer-scorer-identity/v1"
)
_ASSET_FIELDS = {"role", "asset_id", "filename", "sha256", "size_bytes"}
_ASSET_ROLES = {
    "sam2.1-hiera-large-checkpoint": "SAM2 checkpoint",
    "sam2.1-hiera-large-config": "SAM2 configuration",
    "resnet50-imagenet1k-v2-weights": "ResNet50 weights",
    "sam2-apache-license": "SAM2 license",
    "torchvision-bsd-license": "torchvision license",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


@dataclass(frozen=True)
class _TransferScorerAdapter:
    root: Path
    bundle_sha256: str
    classifier: bytes
    imputer: tuple[float, ...]
    threshold: float
    checkpoint: Path
    sam2_config: Path
    resnet50_weights: Path
    prototype: tuple[float, ...]
    pca_mean: tuple[float, ...]
    pca_components: tuple[tuple[float, ...], ...]
    provenance: Mapping[str, object]
    tree_identity: tuple[Mapping[str, object], ...]
    schema: str = BUNDLE_SCHEMA_V2
    profile: str = CANONICAL_GPU_PROFILE
    feature_order: tuple[str, ...] = CANONICAL_FEATURE_ORDER
    feature_order_sha256: str = CANONICAL_FEATURE_ORDER_SHA256


def _validated_asset_inventory(
    value: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Return the exact registered Full-profile assets in canonical order."""

    _require(
        isinstance(value, (list, tuple)) and len(value) == 5,
        "r92 transfer scorer requires exactly five asset inventory rows",
    )
    registry = asset_registry()
    expected_ids = set(asset_ids_for_profile(CANONICAL_GPU_PROFILE))
    rows: list[dict[str, object]] = []
    for raw in value:
        _require(
            isinstance(raw, Mapping) and set(raw) == _ASSET_FIELDS,
            "r92 transfer scorer asset inventory fields changed",
        )
        asset_id = raw.get("asset_id")
        _require(
            isinstance(asset_id, str) and asset_id in expected_ids,
            "r92 transfer scorer names an unexpected public asset",
        )
        spec = registry[asset_id]
        row = {
            "role": raw.get("role"),
            "asset_id": asset_id,
            "filename": raw.get("filename"),
            "sha256": raw.get("sha256"),
            "size_bytes": raw.get("size_bytes"),
        }
        _require(
            row["role"] == _ASSET_ROLES[asset_id]
            and row["filename"] == spec.filename
            and row["sha256"] == spec.sha256
            and type(row["size_bytes"]) is int
            and row["size_bytes"] == spec.size_bytes,
            "r92 transfer scorer asset inventory differs from the public registry",
        )
        rows.append(row)
    ordered = tuple(sorted(rows, key=lambda row: str(row["asset_id"])))
    _require(
        {str(row["asset_id"]) for row in ordered} == expected_ids
        and list(value) == list(ordered),
        "r92 transfer scorer asset inventory is incomplete or not canonical",
    )
    return ordered


def transfer_scorer_identity_sha256(
    *,
    preset_id: str,
    resource_manifest_sha256: str,
    classifier_sha256: str,
    feature_state_sha256: str,
    profile: str,
    threshold: float,
    feature_order_sha256: str,
    asset_inventory: Sequence[Mapping[str, object]],
) -> str:
    """Compute the public, path-free identity of an r92 transfer scorer."""

    assets = _validated_asset_inventory(asset_inventory)
    _require(
        preset_id == PRESET_ID
        and profile == CANONICAL_GPU_PROFILE
        and threshold == CANONICAL_DECISION_THRESHOLD
        and feature_order_sha256 == CANONICAL_FEATURE_ORDER_SHA256,
        "r92 transfer scorer identity metadata is not canonical",
    )
    for value, role in (
        (resource_manifest_sha256, "r92 resource manifest"),
        (classifier_sha256, "r92 classifier"),
        (feature_state_sha256, "r92 feature state"),
    ):
        _require(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"{role} identity is not a lowercase SHA-256",
        )
    return compact_json_sha256(
        {
            "schema": _TRANSFER_SCORER_IDENTITY_SCHEMA,
            "workflow": TRANSFER_WORKFLOW,
            "preset_id": preset_id,
            "resource_manifest_sha256": resource_manifest_sha256,
            "classifier_sha256": classifier_sha256,
            "feature_state_sha256": feature_state_sha256,
            "profile": profile,
            "threshold": threshold,
            "feature_order_sha256": feature_order_sha256,
            "assets": list(assets),
        }
    )


def _asset_inventory(config: PublicProjectConfig) -> tuple[dict[str, object], ...]:
    checked = check_local_assets(config)
    rows = tuple(
        {
            "role": str(row["role"]),
            "asset_id": str(row["asset_id"]),
            "filename": str(row["filename"]),
            "sha256": str(row["sha256"]),
            "size_bytes": int(row["size_bytes"]),
        }
        for row in checked["assets"]
    )
    return _validated_asset_inventory(
        tuple(sorted(rows, key=lambda row: str(row["asset_id"])))
    )


def _adapter(
    config: PublicProjectConfig,
    preset: PublishedModelTransferAssets,
    assets: tuple[dict[str, object], ...],
) -> _TransferScorerAdapter:
    import numpy as np

    _require(
        config.profile == CANONICAL_GPU_PROFILE
        and config.device == "cuda"
        and preset.preset_id == PRESET_ID
        and preset.profile == CANONICAL_GPU_PROFILE
        and preset.threshold == CANONICAL_DECISION_THRESHOLD,
        "r92 transfer inference requires the canonical Full CUDA profile",
    )
    _require(
        config.embedding_weights is not None,
        "r92 transfer inference is missing the ResNet50 asset",
    )
    identity = {
        "schema": _TRANSFER_SCORER_IDENTITY_SCHEMA,
        "workflow": TRANSFER_WORKFLOW,
        "preset_id": preset.preset_id,
        "resource_manifest_sha256": preset.resource_manifest_sha256,
        "classifier_sha256": preset.classifier_sha256,
        "feature_state_sha256": preset.feature_state_sha256,
        "profile": preset.profile,
        "threshold": preset.threshold,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        "assets": list(assets),
    }
    scorer_sha256 = transfer_scorer_identity_sha256(
        preset_id=preset.preset_id,
        resource_manifest_sha256=preset.resource_manifest_sha256,
        classifier_sha256=preset.classifier_sha256,
        feature_state_sha256=preset.feature_state_sha256,
        profile=preset.profile,
        threshold=preset.threshold,
        feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
        asset_inventory=assets,
    )
    state = preset.feature_state
    return _TransferScorerAdapter(
        root=config.asset_root,
        bundle_sha256=scorer_sha256,
        classifier=preset.classifier_ubj,
        imputer=preset.imputer_statistics,
        threshold=preset.threshold,
        checkpoint=config.checkpoint,
        sam2_config=config.sam2_config,
        resnet50_weights=config.embedding_weights,
        prototype=tuple(float(value) for value in np.asarray(state.prototype).reshape(-1)),
        pca_mean=tuple(float(value) for value in np.asarray(state.pca_mean).reshape(-1)),
        pca_components=tuple(
            tuple(float(value) for value in row)
            for row in np.asarray(state.pca_components)
        ),
        provenance={
            "details": {
                # These counts are intentionally not presented as historical
                # r92 fit counts.  The canonical inference adapter needs only
                # positive validation bounds when reconstructing frozen state.
                "training_row_count": 32,
                "positive_training_row_count": 1,
                "lineage_policy": TRANSFER_LINEAGE_POLICY,
                "reproduction_claim": TRANSFER_REPRODUCTION_CLAIM,
            }
        },
        tree_identity=tuple([identity, *assets]),
    )


def _safe_output(
    output: Path,
    *,
    protected_inputs: Sequence[Path],
) -> tuple[Path, Path]:
    absolute = output.absolute()
    _require(
        absolute.name not in {"", ".", ".."}
        and len(absolute.name.encode("utf-8")) <= 180,
        "transfer inference output name is unsafe or too long",
    )
    try:
        parent = absolute.parent.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError("transfer inference output parent is unavailable") from exc
    info = absolute.parent.lstat()
    _require(
        parent == absolute.parent
        and stat.S_ISDIR(info.st_mode)
        and not absolute.parent.is_symlink()
        and info.st_uid == os.geteuid()
        and not absolute.exists()
        and not absolute.is_symlink(),
        "transfer inference output is unsafe or already exists",
    )
    for raw_input in protected_inputs:
        try:
            source = raw_input.absolute().resolve(strict=True)
        except OSError as exc:
            raise PublicIOError(
                "transfer inference protected input is unavailable"
            ) from exc
        _require(
            absolute != source
            and absolute not in source.parents
            and source not in absolute.parents,
            "transfer inference output overlaps its image, config, or asset input",
        )
    staging = parent / f".{absolute.name}.r92-transfer.{uuid.uuid4()}"
    _require(len(staging.name.encode("utf-8")) <= 255, "transfer staging name is too long")
    return absolute, staging


def infer_r92_transfer_image(
    config_path: Path,
    images: Path,
    output: Path,
    *,
    preset_id: str = PRESET_ID,
) -> tuple[dict[str, object], int]:
    """Run proposal→feature→r92 inference for exactly one new image."""

    from compag_curation.public_pipeline import _input_images_identity

    config = load_public_config(config_path, check_local=False)
    _require(
        config.profile == CANONICAL_GPU_PROFILE and config.device == "cuda",
        "r92 transfer inference requires a canonical Full CUDA project config",
    )
    before_images = _input_images_identity(images)
    _require(
        len(before_images) == 1,
        "one-image transfer inference requires exactly one supported image",
    )
    dependencies = require_science_dependencies("cuda")
    preset = load_published_model_transfer_assets(preset_id)
    assets = _asset_inventory(config)
    initial_adapter = _adapter(config, preset, assets)
    final, staging = _safe_output(
        output,
        protected_inputs=(images, config.config_path, config.asset_root),
    )
    staging.mkdir(mode=0o700)
    fsync_directory(staging.parent)

    def verify_adapter(root: Path) -> _TransferScorerAdapter:
        _require(
            root.absolute().resolve(strict=True) == config.asset_root,
            "r92 transfer scorer adapter root changed",
        )
        current_config = load_public_config(config_path, check_local=False)
        _require(
            current_config.config_sha256 == config.config_sha256,
            "project config changed during r92 transfer inference",
        )
        return _adapter(
            current_config,
            load_published_model_transfer_assets(preset_id),
            _asset_inventory(current_config),
        )

    try:
        result = _infer_canonical_bundle_with_dependencies(
            images,
            config.asset_root,
            staging,
            device="cuda",
            bundle_verifier=verify_adapter,
        )
        after_images = _input_images_identity(images)
        _require(
            after_images == before_images,
            "one-image input changed during r92 transfer inference",
        )
        final_adapter = verify_adapter(config.asset_root)
        _require(
            final_adapter.bundle_sha256 == initial_adapter.bundle_sha256
            and final_adapter.tree_identity == initial_adapter.tree_identity,
            "r92 transfer scorer or public assets changed during inference",
        )
        revalidated = revalidate_science_gpu_dependencies("cuda", dependencies)
        _require(
            dict(revalidated) == dict(dependencies),
            "science-gpu dependencies changed during r92 transfer inference",
        )
        inventory = result.get("input_inventory")
        _require(
            isinstance(inventory, list)
            and len(inventory) == 1
            and isinstance(inventory[0], dict),
            "r92 transfer inference did not close exactly one image",
        )
        image = dict(inventory[0])
        expected_image_fields = {
            "image_name",
            "image_id",
            "image_sha256",
            "group_id",
            "size_bytes",
            "width",
            "height",
            "tile_count",
            "candidate_rows",
        }
        _require(
            set(image) == expected_image_fields,
            "r92 transfer inference image receipt fields changed",
        )
        receipt = {
            "schema": TRANSFER_IMAGE_INFERENCE_SCHEMA,
            "status": "PASS",
            "workflow": TRANSFER_WORKFLOW,
            "lineage_policy": TRANSFER_LINEAGE_POLICY,
            "reproduction_claim": TRANSFER_REPRODUCTION_CLAIM,
            "preset_id": preset.preset_id,
            "profile": preset.profile,
            "device": "cuda",
            "threshold": preset.threshold,
            "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
            "resource_manifest_sha256": preset.resource_manifest_sha256,
            "classifier_sha256": preset.classifier_sha256,
            "feature_state_sha256": preset.feature_state_sha256,
            "scorer_identity_sha256": initial_adapter.bundle_sha256,
            "config_sha256": config.config_sha256,
            "asset_inventory": list(assets),
            "asset_inventory_sha256": compact_json_sha256(list(assets)),
            "image_inventory_sha256": compact_json_sha256([image]),
            "image": image,
            "predictions_sha256": sha256_file(staging / "predictions.csv"),
            "raw_features_sha256": sha256_file(staging / "raw_features.csv"),
            "inference_result_sha256": sha256_file(staging / "inference_result.json"),
        }
        _require(
            result.get("bundle_sha256") == receipt["scorer_identity_sha256"]
            and result.get("predictions_sha256") == receipt["predictions_sha256"]
            and isinstance(result.get("raw_feature_archive"), dict)
            and result["raw_feature_archive"].get("sha256")
            == receipt["raw_features_sha256"],
            "r92 transfer scorer receipt differs from canonical inference output",
        )
        write_new_json(staging / "transfer_scorer.json", receipt)
        os.chmod(staging, 0o755, follow_symlinks=False)
        fsync_directory(staging)
        publish_directory_noreplace(staging, final)
        return receipt, 0
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
            fsync_directory(staging.parent)
        raise


__all__ = [
    "TRANSFER_IMAGE_INFERENCE_SCHEMA",
    "TRANSFER_LINEAGE_POLICY",
    "TRANSFER_REPRODUCTION_CLAIM",
    "TRANSFER_WORKFLOW",
    "infer_r92_transfer_image",
    "transfer_scorer_identity_sha256",
]
