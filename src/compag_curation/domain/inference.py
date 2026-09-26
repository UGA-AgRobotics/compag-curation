"""Source-backed inference orchestration with explicit local assets.

This module is deliberately import-safe: importing it does not import a scientific
package, inspect a model, access the network, or start a process.  The execution
boundary is the injected :class:`CommandRunner`.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
import os
from pathlib import Path
from subprocess import CompletedProcess, run
from sys import executable as current_python
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence

from ..config import DomainConfig
from ..contracts import (
    ContractError,
    HandlerServices,
    WorkflowPlan,
    WorkflowStep,
    artifact_path,
    create_output_directory,
    directory_identity,
    directory_tree_digest,
    require_callable_shape,
    require_authorized,
    resolve_python_executable,
    validate_artifact_preconditions,
)
from ._shared import artifact, make_plan


InferenceMode = Literal["gate", "xgb", "xgb_recall"]
EmbeddingBackbone = Literal["resnet18", "resnet34", "resnet50", "resnet101"]

class InferenceContractError(ContractError):
    """Raised before a scientific process when the local contract is incomplete."""


@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    environment: Mapping[str, str] = field(default_factory=dict)
    cwd: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(closed_inference_environment(self.environment)),
        )


class CommandRunner(Protocol):
    """Narrow subprocess boundary used by inference orchestration."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None,
        env: Mapping[str, str],
        acceptable_exit_codes: frozenset[int] = frozenset({0}),
        capture_output: bool = False,
    ) -> ProcessResult: ...


# SOURCE_CELL: NB-LIVE-0001-C0008
# SOURCE_STATEMENT_MAP: explicit-workflow-environment -> closed_inference_environment
def closed_inference_environment(injected: Mapping[str, str]) -> dict[str, str]:
    """Copy only values explicitly supplied by the authorized caller."""

    if not isinstance(injected, MappingABC):
        raise InferenceContractError("inference environment must be a string mapping")
    if any(type(key) is not str or type(value) is not str for key, value in injected.items()):
        raise InferenceContractError("inference environment keys and values must be strings")
    return dict(injected)


class LocalCommandRunner:
    """Execute one argv vector without shell interpolation."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None,
        env: Mapping[str, str],
        acceptable_exit_codes: frozenset[int] = frozenset({0}),
        capture_output: bool = False,
    ) -> ProcessResult:
        completed: CompletedProcess[str] = run(
            [str(part) for part in argv],
            cwd=None if cwd is None else str(cwd),
            env=closed_inference_environment(env),
            check=False,
            text=True,
            capture_output=capture_output,
        )
        result = ProcessResult(
            tuple(str(part) for part in argv),
            completed.returncode,
            completed.stdout or "",
            completed.stderr or "",
            closed_inference_environment(env),
            cwd,
        )
        return result


@dataclass(frozen=True)
class InferenceAssets:
    images: Path
    output: Path
    sam2_package_root: Path
    sam2_package_root_sha256: str
    sam2_checkpoint: Path
    sam2_checkpoint_sha256: str
    sam2_config: Path
    sam2_config_sha256: str
    prototype: Path
    prototype_sha256: str
    padded_prototype: Path
    padded_prototype_sha256: str
    pca: Path
    pca_sha256: str
    xgb_model: Path
    xgb_model_sha256: str
    embed_backbone: EmbeddingBackbone
    embed_backbone_weights: Path
    embed_backbone_weights_sha256: str
    yolo_weights: Path | None = None
    yolo_weights_sha256: str | None = None


@dataclass(frozen=True)
class InferenceOptions:
    mode: InferenceMode = "xgb_recall"
    python_executable: Path = field(default_factory=lambda: Path(current_python))
    entry_point: tuple[str, ...] = (
        "-c",
        "from compag_curation.proposals.sam2_pipeline.args import make_argparser; "
        "from compag_curation.proposals.sam2_pipeline.pipeline import run_pipeline; "
        "ap=make_argparser(); args=ap.parse_args(); run_pipeline(args)",
    )
    device: Literal["cpu", "cuda"] = "cpu"
    sam2_policy: str = "both"
    save_all: bool = True
    xgb_threshold: float | int = 0.5
    yolo_confidence: float | int = 0.20
    yolo_iou: float | int = 0.60
    yolo_device: str = "0"
    detector_policy: str = "hybrid"
    detector_missing: str = "ignore"
    detector_threshold: float | int = 0.5
    hybrid_yolo_bias: float | int = 0.8
    force_xgb_probability: bool = False
    allow_download: bool = False


@dataclass(frozen=True)
class InferenceRequest:
    assets: InferenceAssets
    options: InferenceOptions = field(default_factory=InferenceOptions)
    cwd: Path | None = None
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(closed_inference_environment(self.environment)),
        )


@dataclass(frozen=True)
class InferenceResult:
    output: Path
    command: tuple[str, ...]
    process: ProcessResult
    backend: Literal["sam2-xgb", "sam2-xgb-yolo"]


@dataclass(frozen=True)
class InferenceWorkflowServices:
    common: HandlerServices
    runner: CommandRunner
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.common, HandlerServices):
            raise ContractError("inference execution requires typed handler services")
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(closed_inference_environment(self.environment)),
        )
        self.validate_call_shapes()

    def validate_call_shapes(self) -> None:
        """Validate the runner and common services without invoking either."""

        if not isinstance(self.common, HandlerServices):
            raise ContractError("inference execution requires typed handler services")
        self.common.validate_call_shapes()
        closed_inference_environment(self.environment)
        require_callable_shape(
            getattr(self.runner, "run", None),
            "inference runner",
            (),
            cwd=None,
            env={},
            acceptable_exit_codes=frozenset({0}),
            capture_output=False,
        )


def _require_inference_process_result(
    process: ProcessResult,
    argv: Sequence[str],
    environment: Mapping[str, str],
    cwd: Path | None,
    acceptable_exit_codes: frozenset[int],
) -> ProcessResult:
    """Enforce the injected runner result contract at the domain boundary."""

    if not isinstance(process, ProcessResult):
        raise InferenceContractError("inference runner returned an invalid result")
    if process.argv != tuple(str(part) for part in argv):
        raise InferenceContractError("inference runner changed the configured argv")
    if dict(process.environment) != closed_inference_environment(environment):
        raise InferenceContractError("inference runner did not bind the closed environment")
    if process.cwd != cwd:
        raise InferenceContractError("inference runner changed the configured working directory")
    if process.returncode not in acceptable_exit_codes:
        raise InferenceContractError(
            f"inference process returned {process.returncode}; "
            f"accepted={sorted(acceptable_exit_codes)}"
        )
    return process


def normalize_sam2_config(config_path: Path, *, package_root: Path) -> str:
    """Return the verified package-relative SAM2 configuration locator."""

    if not config_path.is_absolute() or not package_root.is_absolute():
        raise InferenceContractError("SAM2 configuration paths must be absolute")
    absolute_config = Path(os.path.abspath(os.fspath(config_path)))
    absolute_root = Path(os.path.abspath(os.fspath(package_root)))
    try:
        resolved_config = config_path.resolve(strict=True)
        resolved_root = package_root.resolve(strict=True)
    except OSError as exc:
        raise InferenceContractError("SAM2 configuration path is unavailable") from exc
    if (
        resolved_config != absolute_config
        or resolved_root != absolute_root
        or config_path.is_symlink()
        or package_root.is_symlink()
        or not resolved_config.is_file()
        or not resolved_root.is_dir()
    ):
        raise InferenceContractError("SAM2 configuration path is not a real package file")
    try:
        relative = resolved_config.relative_to(resolved_root)
    except ValueError as exc:
        raise InferenceContractError(
            "SAM2 configuration must reside inside the loaded package"
        ) from exc
    if not relative.parts or relative.parts[0] != "configs":
        raise InferenceContractError("SAM2 configuration must reside below package configs")
    if relative.suffix != ".yaml":
        raise InferenceContractError("SAM2 configuration must be a YAML package resource")
    return relative.with_suffix("").as_posix()


def _require_file(path: Path, role: str) -> Path:
    if not path.is_file():
        raise InferenceContractError(f"missing explicit local {role}: {path}")
    return path


def _require_directory(path: Path, role: str) -> Path:
    if not path.is_dir():
        raise InferenceContractError(f"missing explicit local {role}: {path}")
    return path


def _require_sha256(value: str | None, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise InferenceContractError(f"missing or invalid SHA-256 for {role}")
    return value


# SOURCE_CELL: NB-LIVE-0001-C0008
# SOURCE_CELL: NB-LIVE-0012-C0001
# SOURCE_STATEMENT_MAP: ordered-shell-arrays -> common,yolo,decision,mode-specific argv
def build_inference_argv(request: InferenceRequest) -> tuple[str, ...]:
    """Build the source-stable, shell-free inference command vector."""

    assets = request.assets
    options = request.options
    if options.mode not in {"gate", "xgb", "xgb_recall"}:
        raise InferenceContractError("mode must be gate, xgb, or xgb_recall")
    if options.allow_download:
        raise InferenceContractError("inference assets must be explicit and local")

    _require_directory(assets.images, "images directory")
    _require_directory(assets.sam2_package_root, "SAM2 package root")
    _require_file(assets.sam2_checkpoint, "SAM2 checkpoint")
    _require_file(assets.sam2_config, "SAM2 configuration")
    _require_file(assets.prototype, "prototype")
    _require_file(assets.padded_prototype, "padded prototype")
    _require_file(assets.pca, "PCA transform")
    _require_file(assets.embed_backbone_weights, "embedding backbone weights")
    if assets.embed_backbone not in {"resnet18", "resnet34", "resnet50", "resnet101"}:
        raise InferenceContractError("embedding backbone architecture is invalid")
    _require_sha256(assets.sam2_package_root_sha256, "SAM2 package root")
    _require_sha256(assets.sam2_checkpoint_sha256, "SAM2 checkpoint")
    _require_sha256(assets.sam2_config_sha256, "SAM2 configuration")
    _require_sha256(assets.prototype_sha256, "prototype")
    _require_sha256(assets.padded_prototype_sha256, "padded prototype")
    _require_sha256(assets.pca_sha256, "PCA transform")
    _require_sha256(assets.embed_backbone_weights_sha256, "embedding backbone weights")
    if options.mode in {"xgb", "xgb_recall"}:
        _require_file(assets.xgb_model, "XGBoost model")
        _require_sha256(assets.xgb_model_sha256, "XGBoost model")
    try:
        python_executable = Path(resolve_python_executable(str(options.python_executable)))
    except ContractError as exc:
        raise InferenceContractError("configured Python interpreter is unavailable") from exc
    try:
        prototype_identity = assets.prototype.resolve(strict=True)
        padded_identity = assets.padded_prototype.resolve(strict=True)
    except OSError as exc:
        raise InferenceContractError("inference prototype assets are unavailable") from exc
    if prototype_identity == padded_identity:
        raise InferenceContractError(
            "padded prototype must be an independent SHA-bound artifact"
        )
    sam2_config_locator = normalize_sam2_config(
        assets.sam2_config,
        package_root=assets.sam2_package_root,
    )

    common = [
        str(python_executable),
        *options.entry_point,
        "--images",
        str(assets.images),
        "--out",
        str(assets.output),
        "--sam2-package-root",
        str(assets.sam2_package_root),
        "--sam2-package-root-sha256",
        assets.sam2_package_root_sha256,
        "--sam2-config",
        str(assets.sam2_config),
        "--sam2-config-locator",
        sam2_config_locator,
        "--sam2-config-sha256",
        assets.sam2_config_sha256,
        "--sam2-ckpt",
        str(assets.sam2_checkpoint),
        "--sam2-ckpt-sha256",
        assets.sam2_checkpoint_sha256,
        "--sam2-policy",
        options.sam2_policy,
        "--feature-mode",
        "ultra",
        "--device",
        options.device,
        "--unified-proto-csv",
        str(assets.prototype),
        "--unified-proto-csv-sha256",
        assets.prototype_sha256,
        "--embed-csv",
        str(assets.padded_prototype),
        "--embed-csv-sha256",
        assets.padded_prototype_sha256,
        "--embed-pca",
        str(assets.pca),
        "--embed-pca-sha256",
        assets.pca_sha256,
        "--embed-backbone",
        assets.embed_backbone,
        "--embed-backbone-weights",
        str(assets.embed_backbone_weights),
        "--embed-backbone-weights-sha256",
        assets.embed_backbone_weights_sha256,
        "--ms-scales",
        "0.67,0.80,1.00,1.25",
    ]

    yolo = ["--yolo-hint", "off"]
    if assets.yolo_weights is not None and not options.force_xgb_probability:
        _require_file(assets.yolo_weights, "YOLO weights")
        yolo_sha256 = _require_sha256(assets.yolo_weights_sha256, "YOLO weights")
        yolo = [
            "--yolo-weights",
            str(assets.yolo_weights),
            "--yolo-weights-sha256",
            yolo_sha256,
            "--yolo-conf",
            str(options.yolo_confidence),
            "--yolo-iou",
            str(options.yolo_iou),
            "--yolo-device",
            options.yolo_device,
            "--yolo-imgsz",
            "512",
            "--yolo-max-det",
            "300",
            "--debug-yolo-dump",
            "--yolo-hint",
            "off",
        ]

    decision = [
        "--use-xgb-inference",
        "--xgb-model",
        str(assets.xgb_model),
        "--xgb-model-sha256",
        assets.xgb_model_sha256,
        "--xgb-features",
        "auto",
        "--xgb-policy",
        "replace",
        "--xgb-threshold",
        str(options.xgb_threshold),
        "--det-policy",
        options.detector_policy,
        "--det-missing",
        options.detector_missing,
        "--det-thr",
        str(options.detector_threshold),
        "--hybrid-yolo-bias",
        str(options.hybrid_yolo_bias),
        "--thr_yolo_raw",
        str(options.yolo_confidence),
        "--thr_yolo_iou",
        str(options.yolo_iou),
    ]

    if options.mode == "gate":
        mode_args = [
            "--points-per-side",
            "64",
            "--pred-iou",
            "0.88",
            "--stability",
            "0.92",
        ]
    elif options.mode == "xgb":
        mode_args = [
            "--points-per-side",
            "64",
            "--pred-iou",
            "0.88",
            "--stability",
            "0.92",
            *decision,
        ]
    else:
        export = ["--export-features"] if options.save_all else []
        mode_args = [
            "--points-per-side",
            "64",
            "--points-per-batch",
            "512",
            "--crop-n-layers",
            "0",
            "--crop-n-points-downscale-factor",
            "2",
            "--crop-overlap-ratio",
            "0.4",
            "--pred-iou",
            "0.80",
            "--stability",
            "0.88",
            "--no-exclude-largest",
            "--min-true-gates",
            "3",
            *export,
            *decision,
        ]
    return tuple(common + yolo + mode_args)


# SOURCE_CELL: NB-LIVE-0001-C0008
# SOURCE_STATEMENT_MAP: mkdir-env-process-result -> execute_inference
def execute_inference(
    request: InferenceRequest,
    *,
    runner: CommandRunner,
) -> InferenceResult:
    """Validate assets, claim the output root, and run the injected child."""

    if runner is None:
        raise InferenceContractError("inference execution requires an explicit runner")
    argv = build_inference_argv(request)
    environment = {
        **closed_inference_environment(request.environment),
        "CJ_SAVE_ALL": "1" if request.options.save_all else "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    acceptable_exit_codes = frozenset({0})
    output = create_output_directory(request.assets.output)
    output_identity = output.lstat()
    argv = (
        *argv,
        "--precreated-out-device",
        str(output_identity.st_dev),
        "--precreated-out-inode",
        str(output_identity.st_ino),
    )
    process = runner.run(
        argv,
        cwd=request.cwd,
        env=environment,
        acceptable_exit_codes=acceptable_exit_codes,
        capture_output=False,
    )
    process = _require_inference_process_result(
        process,
        argv,
        environment,
        request.cwd,
        acceptable_exit_codes,
    )
    try:
        completed_output_identity = output.lstat()
    except OSError as exc:
        raise InferenceContractError(
            "inference runner removed the declared output directory"
        ) from exc
    if output.is_symlink() or not output.is_dir():
        raise InferenceContractError(
            "inference runner replaced the declared output directory"
        )
    if (
        completed_output_identity.st_dev != output_identity.st_dev
        or completed_output_identity.st_ino != output_identity.st_ino
    ):
        raise InferenceContractError(
            "inference runner changed the declared output identity"
        )
    if output.resolve(strict=True) != Path(os.path.abspath(os.fspath(output))):
        raise InferenceContractError(
            "inference runner changed the declared output locator"
        )
    backend: Literal["sam2-xgb", "sam2-xgb-yolo"] = (
        "sam2-xgb-yolo" if request.assets.yolo_weights is not None else "sam2-xgb"
    )
    return InferenceResult(output, argv, process, backend)


# SOURCE_CELL: NB-LIVE-0001-C0008
# SOURCE_STATEMENT_MAP: explicit-domain-config -> execute_inference
def run_inference(
    config: DomainConfig,
    variant: str | None = None,
    *,
    runner: CommandRunner,
    environment: Mapping[str, str] = MappingProxyType({}),
) -> InferenceResult:
    """Registry handler for the canonical integrated dual-backend implementation."""

    environment = MappingProxyType(closed_inference_environment(environment))
    if runner is None:
        raise InferenceContractError("configured inference requires an explicit runner")
    values = config.workflow("infer")
    selected = variant or str(values.get("variant", "sam2-xgb"))
    if selected not in {"sam2-xgb", "sam2-xgb-yolo"}:
        raise InferenceContractError(f"unknown inference variant: {selected}")
    execution = getattr(config, "execution", None)
    python_value = getattr(execution, "python_executable", current_python)
    options = InferenceOptions(
        mode=values["mode"],  # type: ignore[arg-type]
        python_executable=Path(str(python_value)),
        sam2_policy=values["sam2_policy"],
        save_all=values["save_all"],
        xgb_threshold=values["decision_threshold"],
        yolo_confidence=values["yolo_confidence"],
        yolo_iou=values["yolo_iou"],
        detector_policy=values["detector_policy"],
        detector_missing=values["detector_missing"],
        detector_threshold=values["detector_threshold"],
        hybrid_yolo_bias=values["hybrid_yolo_bias"],
        yolo_device="0" if values["device"] == "cuda" else "cpu",
        allow_download=bool(getattr(execution, "allow_download", False)),
        device=str(values["device"]),  # type: ignore[arg-type]
    )
    config_path = artifact_path(values, "sam2_config")
    prototype_path = artifact_path(values, "prototype")

    def configured_sha256(name: str) -> str:
        value = values.get(name)
        if not isinstance(value, dict):
            raise InferenceContractError(f"configured inference artifact is invalid: {name}")
        return _require_sha256(value.get("sha256"), name)

    assets = InferenceAssets(
        images=artifact_path(values, "images"),
        output=artifact_path(values, "output"),
        sam2_package_root=artifact_path(values, "sam2_package_root"),
        sam2_package_root_sha256=configured_sha256("sam2_package_root"),
        sam2_checkpoint=artifact_path(values, "sam2_checkpoint"),
        sam2_checkpoint_sha256=configured_sha256("sam2_checkpoint"),
        sam2_config=config_path,
        sam2_config_sha256=configured_sha256("sam2_config"),
        prototype=prototype_path,
        prototype_sha256=configured_sha256("prototype"),
        padded_prototype=artifact_path(values, "padded_prototype"),
        padded_prototype_sha256=configured_sha256("padded_prototype"),
        pca=artifact_path(values, "pca"),
        pca_sha256=configured_sha256("pca"),
        xgb_model=artifact_path(values, "xgb_model"),
        xgb_model_sha256=configured_sha256("xgb_model"),
        embed_backbone=values["embed_backbone"],  # type: ignore[arg-type]
        embed_backbone_weights=artifact_path(values, "embed_backbone_weights"),
        embed_backbone_weights_sha256=configured_sha256("embed_backbone_weights"),
        yolo_weights=(
            artifact_path(values, "yolo_weights")
            if selected == "sam2-xgb-yolo" and values.get("yolo_weights")
            else None
        ),
        yolo_weights_sha256=(
            configured_sha256("yolo_weights")
            if selected == "sam2-xgb-yolo" and values.get("yolo_weights")
            else None
        ),
    )
    return execute_inference(
        InferenceRequest(assets, options, environment=environment),
        runner=runner,
    )


def _run_inference_variant(
    config: DomainConfig,
    services: InferenceWorkflowServices,
    expected_variant: str,
) -> InferenceResult:
    if not isinstance(services, InferenceWorkflowServices):
        raise ContractError("inference handler requires InferenceWorkflowServices")
    services.validate_call_shapes()
    if config.workflow("infer")["variant"] != expected_variant:
        raise ContractError(f"inference handler requires {expected_variant} variant")
    require_authorized(config.execution)
    validate_artifact_preconditions(plan_inference(config, expected_variant))
    result = run_inference(
        config,
        expected_variant,
        runner=services.runner,
        environment=services.environment,
    )
    output_identity = directory_identity(result.output)
    output_digest = directory_tree_digest(result.output)
    if directory_identity(result.output) != output_identity:
        raise InferenceContractError("inference output changed before completion report")
    try:
        services.common.report(f"inference output written: {result.output}")
    finally:
        if (
            directory_identity(result.output) != output_identity
            or directory_tree_digest(result.output) != output_digest
        ):
            raise InferenceContractError("inference output changed during completion report")
    return result


def run_single_image_inference(
    config: DomainConfig,
    services: InferenceWorkflowServices,
) -> InferenceResult:
    """Execute SAM2/XGBoost inference from explicit local artifacts."""

    return _run_inference_variant(config, services, "sam2-xgb")


def run_dual_backend_inference(
    config: DomainConfig,
    services: InferenceWorkflowServices,
) -> InferenceResult:
    """Canonical replacement integrating optional local YOLO evidence."""

    return _run_inference_variant(config, services, "sam2-xgb-yolo")


def plan_inference(config: DomainConfig, variant: str | None = None) -> WorkflowPlan:
    values = config.workflow("infer")
    selected = variant or str(values["variant"])
    if selected not in {"sam2-xgb", "sam2-xgb-yolo"}:
        raise ContractError("unknown inference variant")
    if config.execution.allow_download:
        raise ContractError("inference is restricted to explicit local assets")
    names = (
        "images",
        "sam2_package_root",
        "sam2_checkpoint",
        "sam2_config",
        "prototype",
        "padded_prototype",
        "pca",
        "xgb_model",
        "embed_backbone_weights",
    )
    roles = {
        "images": "image-directory",
        "sam2_package_root": "sealed-package-root",
        "sam2_checkpoint": "sealed-model",
        "sam2_config": "sealed-model-config",
        "prototype": "sealed-feature-prototype",
        "padded_prototype": "sealed-embedding-prototype",
        "pca": "sealed-transform",
        "xgb_model": "sealed-model",
        "embed_backbone_weights": "sealed-model",
        "yolo_weights": "local-model-asset",
    }
    inputs = tuple(artifact(values[name], name, roles[name]) for name in names)
    if selected == "sam2-xgb-yolo":
        inputs += (artifact(values["yolo_weights"], "yolo_weights", roles["yolo_weights"]),)
    outputs = (artifact(values["output"], "output", "prediction-output", output=True),)
    inference_parameters = {
        "mode": values["mode"],
        "save_all": values["save_all"],
        "decision_threshold": values["decision_threshold"],
        "sam2_policy": values["sam2_policy"],
        "yolo_confidence": values["yolo_confidence"],
        "yolo_iou": values["yolo_iou"],
        "detector_policy": values["detector_policy"],
        "detector_missing": values["detector_missing"],
        "detector_threshold": values["detector_threshold"],
        "hybrid_yolo_bias": values["hybrid_yolo_bias"],
    }
    steps = (
        WorkflowStep(
            "inference-preflight",
            "validate-local-inference-assets",
            tuple(item.name for item in inputs),
            source_cell_ids=("NB-LIVE-0001-C0008",),
        ),
        WorkflowStep(
            "inference-run",
            f"orchestrate-{selected}",
            tuple(item.name for item in inputs),
            ("output",),
            {
                **inference_parameters,
                "python_executable": config.execution.python_executable,
                "embed_backbone": values["embed_backbone"],
                "device": values["device"],
                "allow_download": False,
            },
            ("NB-LIVE-0001-C0008",),
        ),
    )
    return make_plan(
        config,
        "infer",
        selected,
        "compag_curation.domain.inference:plan_inference",
        inputs,
        outputs,
        steps,
        inference_parameters,
    )


__all__ = [
    "InferenceWorkflowServices",
    "normalize_sam2_config",
    "plan_inference",
    "run_dual_backend_inference",
    "run_single_image_inference",
]
