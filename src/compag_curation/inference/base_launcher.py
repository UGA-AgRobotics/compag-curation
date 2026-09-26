"""Compatibility facade for the typed, explicit inference launcher.

The former source-derived script embedded an environment-specific interpreter
and scientific Python bodies.  Execution now uses ``InferenceOptions`` (whose
default is the current interpreter) and an injected ``CommandRunner``.
"""

from __future__ import annotations

from compag_curation.domain.inference import (
    CommandRunner,
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
    "CommandRunner",
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
