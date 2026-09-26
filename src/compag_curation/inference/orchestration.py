"""Compatibility imports for the typed inference domain API."""

from __future__ import annotations

from compag_curation.domain.inference import (
    InferenceAssets,
    InferenceOptions,
    InferenceRequest,
    InferenceResult,
    InferenceWorkflowServices,
    build_inference_argv,
    execute_inference,
    normalize_sam2_config,
    run_dual_backend_inference,
    run_single_image_inference,
)

__all__ = [
    "InferenceAssets",
    "InferenceOptions",
    "InferenceRequest",
    "InferenceResult",
    "InferenceWorkflowServices",
    "build_inference_argv",
    "execute_inference",
    "normalize_sam2_config",
    "run_dual_backend_inference",
    "run_single_image_inference",
]
