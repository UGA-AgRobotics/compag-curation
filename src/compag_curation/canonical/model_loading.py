"""Verified local model loading for the canonical Hiera-L proposal stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from compag_curation.public_io import verified_file_path

from .spec import (
    CANONICAL_GPU_PROFILE,
    CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256,
    CANONICAL_SAM2_HIERA_L_CONFIG_SHA256,
    EFFICIENT_GPU_PROFILE,
    EFFICIENT_SAM2_HIERA_T_CHECKPOINT_SHA256,
    EFFICIENT_SAM2_HIERA_T_CONFIG_SHA256,
    amg_settings_for_profile,
)


def _resolve_torch_device(device: str) -> str:
    """Validate a canonical Torch execution device without falling back."""

    normalized = str(device).strip().lower()
    if normalized != "cuda":
        raise ValueError(
            "canonical SAM2 execution is CUDA-only; CPU execution is unsupported"
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "canonical SAM2 CUDA was requested, but PyTorch cannot access a CUDA device"
        )
    return normalized


def _cuda_device_matches(value: Any, cuda_ordinal: int) -> bool:
    device = getattr(value, "device", value)
    return (
        getattr(device, "type", None) == "cuda"
        and getattr(device, "index", None) == cuda_ordinal
    )


def _attest_canonical_sam2_cuda_model(model: Any) -> int:
    """Prove the complete loaded SAM2 module is CUDA/FP32/eval."""

    import torch

    cuda_ordinal = int(torch.cuda.current_device())
    named_parameters = getattr(model, "named_parameters", None)
    named_buffers = getattr(model, "named_buffers", None)
    modules = getattr(model, "modules", None)
    if not all(callable(value) for value in (named_parameters, named_buffers, modules)):
        raise RuntimeError("canonical SAM2 does not expose a complete Torch module state")
    parameters = tuple(named_parameters(recurse=True))
    if not parameters:
        raise RuntimeError("canonical SAM2 has no parameters to attest")
    for name, tensor in (*parameters, *tuple(named_buffers(recurse=True))):
        if not _cuda_device_matches(tensor, cuda_ordinal):
            raise RuntimeError(
                f"canonical SAM2 tensor is not on cuda:{cuda_ordinal}: {name}"
            )
        is_floating = getattr(tensor, "is_floating_point", None)
        if not callable(is_floating):
            raise RuntimeError(f"canonical SAM2 state is not a Torch tensor: {name}")
        if bool(is_floating()) and getattr(tensor, "dtype", None) != torch.float32:
            raise RuntimeError(f"canonical SAM2 floating tensor is not FP32: {name}")
    if any(bool(getattr(module, "training", True)) for module in modules()):
        raise RuntimeError("canonical SAM2 contains a submodule that is not in eval mode")
    if not _cuda_device_matches(getattr(model, "device", None), cuda_ordinal):
        raise RuntimeError(f"canonical SAM2 model.device is not cuda:{cuda_ordinal}")
    return cuda_ordinal


def _lower_sha256(value: str, role: str) -> str:
    rendered = str(value).strip()
    if len(rendered) != 64 or any(
        character not in "0123456789abcdef" for character in rendered
    ):
        raise ValueError(f"{role} requires a lowercase SHA-256")
    return rendered


def _register_verified_sam2_config(config_fd_path: Path, config_hash: str) -> str:
    """Register parsed, descriptor-pinned YAML for the upstream SAM2 builder."""

    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf

    config = OmegaConf.load(str(config_fd_path))
    if "model" not in config:
        raise ValueError("canonical SAM2 configuration has no model definition")
    backbone = (
        "hiera_t"
        if config_hash == EFFICIENT_SAM2_HIERA_T_CONFIG_SHA256
        else "hiera_l"
    )
    config_name = f"compag_verified_sam2_{backbone}_{config_hash}"
    ConfigStore.instance().store(
        name=config_name,
        node=config,
        package="_global_",
        provider="compag-curation",
    )
    return config_name


def load_canonical_sam2_model(
    config_path: Path,
    checkpoint_path: Path,
    config_locator: str,
    expected_config_sha256: str,
    expected_checkpoint_sha256: str,
    *,
    device: str = "cuda",
    profile: str = CANONICAL_GPU_PROFILE,
) -> Any:
    """Load only the official assets sealed to the selected v2 profile."""

    settings = amg_settings_for_profile(profile)
    if config_locator != settings.config_locator:
        if profile != EFFICIENT_GPU_PROFILE:
            raise ValueError("canonical SAM2 configuration locator is not Hiera-L")
        raise ValueError("efficient SAM2 configuration locator is not Hiera-T")
    resolved_device = _resolve_torch_device(device)
    config_hash = _lower_sha256(expected_config_sha256, "SAM2 configuration")
    checkpoint_hash = _lower_sha256(
        expected_checkpoint_sha256, "SAM2 checkpoint"
    )
    expected_config_hash = (
        EFFICIENT_SAM2_HIERA_T_CONFIG_SHA256
        if profile == EFFICIENT_GPU_PROFILE
        else CANONICAL_SAM2_HIERA_L_CONFIG_SHA256
    )
    expected_checkpoint_hash = (
        EFFICIENT_SAM2_HIERA_T_CHECKPOINT_SHA256
        if profile == EFFICIENT_GPU_PROFILE
        else CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256
    )
    if config_hash != expected_config_hash:
        raise ValueError("SAM2 configuration identity does not match the profile")
    if checkpoint_hash != expected_checkpoint_hash:
        raise ValueError("SAM2 checkpoint identity does not match the profile")
    if not config_path.is_absolute() or not checkpoint_path.is_absolute():
        raise ValueError("canonical SAM2 asset paths must be absolute")

    import sam2
    from sam2.build_sam import build_sam2

    package_root = Path(sam2.__file__).resolve(strict=True).parent
    installed_config = package_root / f"{config_locator}.yaml"
    with (
        verified_file_path(
            config_path,
            config_hash,
            max_bytes=64 * 1024,
            require_single_link=True,
        ) as (config_fd_path, _config_snapshot),
        verified_file_path(
            installed_config,
            config_hash,
            max_bytes=64 * 1024,
            require_single_link=True,
        ),
        verified_file_path(
            checkpoint_path,
            checkpoint_hash,
            max_bytes=1024 * 1024 * 1024,
            require_single_link=True,
        ) as (checkpoint_fd_path, _checkpoint_snapshot),
    ):
        verified_config_name = _register_verified_sam2_config(
            config_fd_path, config_hash
        )
        model = build_sam2(
            verified_config_name,
            str(checkpoint_fd_path),
            device=resolved_device,
            mode="eval",
            apply_postprocessing=False,
        )
    _attest_canonical_sam2_cuda_model(model)
    return model


__all__ = ["load_canonical_sam2_model"]
