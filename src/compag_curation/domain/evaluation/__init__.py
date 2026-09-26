"""Execution handlers for candidate, coverage, border/tiny and meta-analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from sys import executable as current_python
from typing import Any, Mapping

from ...config import ConfigurationError, DomainConfig, validate_image_identifier
from ...contracts import (
    ContractError,
    HandlerServices,
    WorkflowPlan,
    WorkflowStep,
    artifact_path,
    directory_identity,
    directory_tree_digest,
    require_callable_shape,
    require_authorized,
    validate_artifact_preconditions,
)
from .._shared import artifact, make_plan
from .border_tiny import BorderTinyResult, execute_border_tiny_evaluation
from .coverage import (
    CoverageInputArtifact,
    CoverageRequest,
    CoverageResult,
    CoverageRunner,
    execute_coverage,
)
from .meta_analysis import (
    MetaAnalysisResult,
    MetaAnalysisServices,
    execute_meta_analysis,
)
VARIANTS = (
    "coverage-basic",
    "coverage-points-overlay",
    "coverage-predictions-overlay",
    "coverage-reviewability",
    "meta-analysis",
    "border-tiny",
)


class EvaluationContractError(ValueError):
    pass


@dataclass(frozen=True)
class EvaluationServices:
    common: HandlerServices
    coverage_runner: CoverageRunner | None = None
    meta_analysis: MetaAnalysisServices | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.common, HandlerServices):
            raise ContractError("evaluation execution requires typed handler services")
        self.validate_call_shapes()

    def validate_call_shapes(self, variant: str | None = None) -> None:
        """Validate selected injected boundaries without invoking them."""

        if not isinstance(self.common, HandlerServices):
            raise ContractError("evaluation execution requires typed handler services")
        self.common.validate_call_shapes()
        if self.coverage_runner is not None:
            require_callable_shape(
                getattr(self.coverage_runner, "run", None),
                "coverage runner",
                (),
                cwd=None,
                env={},
                acceptable_exit_codes=frozenset({0}),
            )
        if self.meta_analysis is not None:
            require_callable_shape(
                getattr(self.meta_analysis, "read_rows", None),
                "meta-analysis table reader",
                Path("input.csv"),
            )
            require_callable_shape(
                getattr(self.meta_analysis, "write_rows", None),
                "meta-analysis table writer",
                Path("output.csv"),
                (),
            )
        if variant in VARIANTS[:4] and self.coverage_runner is None:
            raise ContractError("coverage execution requires an explicit runner")


def _configured_coverage_image_id(values: Mapping[str, Any]) -> str:
    try:
        return validate_image_identifier(values.get("image_id"), "coverage image_id")
    except ConfigurationError as exc:
        raise EvaluationContractError(str(exc)) from exc


def _coverage_request(
    config: DomainConfig,
    selected: str,
) -> CoverageRequest:
    values = config.workflow("evaluate")
    execution = getattr(config, "execution", None)
    python_value = getattr(execution, "python_executable", current_python)
    predictions = artifact_path(values, "predictions")
    mode = str(values.get("mode", "xgb_recall"))
    images_value = artifact_path(values, "images") if "images" in values else None
    review_value = artifact_path(values, "review_labels") if "review_labels" in values else None
    image_id = _configured_coverage_image_id(values)

    def configured_input(name: str) -> CoverageInputArtifact:
        value = values.get(name)
        if not isinstance(value, dict) or not isinstance(value.get("sha256"), str):
            raise EvaluationContractError(f"coverage {name} requires an explicit SHA-256")
        return CoverageInputArtifact(artifact_path(values, name), value["sha256"])

    return CoverageRequest(
        workflow=selected,  # type: ignore[arg-type]
        image_id=image_id,
        predictions=predictions,
        annotations=artifact_path(values, "annotations"),
        tile_index=configured_input("tile_index"),
        output=artifact_path(values, "output"),
        script=configured_input("evaluator_script"),
        images_root=images_value,
        review_candidates=review_value,
        python_executable=Path(str(python_value)),
        mode=mode,
        xgb_threshold=float(values.get("decision_threshold", 0.5)),
        active_learning_margin=0.20,
        nms_iou=0.50,
        max_area_fraction=0.05,
        cwd=None,
    )


# SOURCE_CELL: NB-LIVE-0004-C0000
# SOURCE_CELL: NB-LIVE-0004-C0001
# SOURCE_CELL: NB-LIVE-0004-C0003
# SOURCE_CELL: NB-LIVE-0004-C0004
# SOURCE_CELL: NB-LIVE-0004-C0005
# SOURCE_CELL: NB-LIVE-0004-C0006
# SOURCE_STATEMENT_MAP: variant-to-v2-v3-contract -> run_coverage_evaluation
def run_coverage_evaluation(
    config: DomainConfig,
    services: EvaluationServices,
) -> CoverageResult:
    if not isinstance(services, EvaluationServices):
        raise ContractError("evaluation handler requires EvaluationServices")
    services.validate_call_shapes("coverage-basic")
    values = config.workflow("evaluate")
    selected = str(values.get("variant", "coverage-basic"))
    if selected not in VARIANTS[:4]:
        raise EvaluationContractError(f"not a coverage variant: {selected}")
    require_authorized(config.execution)
    validate_artifact_preconditions(plan_evaluation(config, selected))
    request = _coverage_request(config, selected)
    return execute_coverage(
        request,
        runner=services.coverage_runner,
    )


def run_meta_analysis(
    config: DomainConfig,
    services: EvaluationServices,
) -> MetaAnalysisResult:
    if not isinstance(services, EvaluationServices):
        raise ContractError("evaluation handler requires EvaluationServices")
    services.validate_call_shapes("meta-analysis")
    if config.workflow("evaluate")["variant"] != "meta-analysis":
        raise ContractError("meta-analysis handler requires meta-analysis variant")
    require_authorized(config.execution)
    validate_artifact_preconditions(plan_evaluation(config, "meta-analysis"))
    result = execute_meta_analysis(config, services.meta_analysis)
    output_identity = directory_identity(result.output)
    output_digest = directory_tree_digest(result.output)
    if directory_identity(result.output) != output_identity:
        raise EvaluationContractError("meta-analysis output changed before completion report")
    try:
        services.common.report(f"meta-analysis output written: {result.output}")
    finally:
        if (
            directory_identity(result.output) != output_identity
            or directory_tree_digest(result.output) != output_digest
        ):
            raise EvaluationContractError("meta-analysis output changed during completion report")
    return result


def run_border_tiny_evaluation(
    config: DomainConfig,
    services: EvaluationServices,
) -> BorderTinyResult:
    if not isinstance(services, EvaluationServices):
        raise ContractError("evaluation handler requires EvaluationServices")
    services.validate_call_shapes("border-tiny")
    if config.workflow("evaluate")["variant"] != "border-tiny":
        raise ContractError("border/tiny handler requires border-tiny variant")
    require_authorized(config.execution)
    validate_artifact_preconditions(plan_evaluation(config, "border-tiny"))
    result = execute_border_tiny_evaluation(config)
    output_identity = directory_identity(result.output)
    output_digest = directory_tree_digest(result.output)
    if directory_identity(result.output) != output_identity:
        raise EvaluationContractError("border/tiny output changed before completion report")
    try:
        services.common.report(f"border/tiny output written: {result.output}")
    finally:
        if (
            directory_identity(result.output) != output_identity
            or directory_tree_digest(result.output) != output_digest
        ):
            raise EvaluationContractError("border/tiny output changed during completion report")
    return result


def run_evaluation(
    config: DomainConfig,
    services: EvaluationServices,
) -> (
    CoverageResult
    | MetaAnalysisResult
    | BorderTinyResult
):
    """Common execution facade; variant-specific handlers remain registry targets."""

    if not isinstance(services, EvaluationServices):
        raise ContractError("evaluation handler requires EvaluationServices")
    services.validate_call_shapes()
    values = config.workflow("evaluate")
    selected = str(values.get("variant", "coverage-basic"))
    if selected in VARIANTS[:4]:
        return run_coverage_evaluation(config, services)
    if selected == "meta-analysis":
        return run_meta_analysis(
            config,
            services,
        )
    if selected == "border-tiny":
        return run_border_tiny_evaluation(config, services)
    raise EvaluationContractError(f"unknown evaluation variant: {selected}")


def plan_evaluation(config: DomainConfig, variant: str | None = None) -> WorkflowPlan:
    values = config.workflow("evaluate")
    selected = variant or values.get("variant", "coverage-basic")
    if selected not in VARIANTS:
        raise EvaluationContractError(f"unknown evaluation variant: {selected}")
    required = {
        "coverage-basic": ("predictions", "annotations", "evaluator_script", "tile_index"),
        "coverage-points-overlay": ("predictions", "annotations", "images", "evaluator_script", "tile_index"),
        "coverage-predictions-overlay": ("predictions", "annotations", "images", "evaluator_script", "tile_index"),
        "coverage-reviewability": ("predictions", "annotations", "images", "review_labels", "evaluator_script", "tile_index"),
        "meta-analysis": ("review_root",),
        "border-tiny": ("predictions", "annotations"),
    }[selected]
    roles = {
        "predictions": "prediction-table",
        "annotations": "sealed-annotation-contract",
        "images": "image-directory",
        "review_labels": "review-input",
        "review_root": "sealed-review-root",
        "evaluator_script": "sealed-evaluator-script",
        "tile_index": "sealed-tile-index",
    }
    inputs = tuple(artifact(values[name], name, roles[name]) for name in required)
    outputs = (artifact(values["output"], "output", "evaluation-output", output=True),)
    source_cells = {
        "coverage-basic": ("NB-LIVE-0004-C0000", "NB-LIVE-0004-C0001"),
        "coverage-points-overlay": ("NB-LIVE-0004-C0003",),
        "coverage-predictions-overlay": ("NB-LIVE-0004-C0004",),
        "coverage-reviewability": ("NB-LIVE-0004-C0005", "NB-LIVE-0004-C0006"),
        "meta-analysis": tuple(f"NB-LIVE-0010-C{index:04d}" for index in range(21)),
        "border-tiny": tuple(f"NB-LIVE-0011-C{index:04d}" for index in range(7)),
    }[selected]
    execution_parameters: dict[str, Any] = {
        "decision_threshold": values.get("decision_threshold", 0.5),
    }
    if selected in VARIANTS[:4]:
        execution_parameters["image_id"] = _configured_coverage_image_id(values)
    steps = (
        WorkflowStep(
            "evaluation-preflight",
            "validate-evaluation-contract",
            tuple(item.name for item in inputs),
            source_cell_ids=source_cells,
        ),
        WorkflowStep(
            "evaluation-run",
            f"evaluate-{selected}",
            tuple(item.name for item in inputs),
            ("output",),
            execution_parameters,
            source_cells,
        ),
    )
    return make_plan(
        config,
        "evaluate",
        selected,
        "compag_curation.domain.evaluation:plan_evaluation",
        inputs,
        outputs,
        steps,
        {"decision_threshold": values.get("decision_threshold", 0.5)},
    )


__all__ = [
    "EvaluationServices",
    "plan_evaluation",
    "run_border_tiny_evaluation",
    "run_coverage_evaluation",
    "run_evaluation",
    "run_meta_analysis",
]
