"""Explicit review UI plans and concrete launch handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..config import DomainConfig
from ..contracts import (
    ContractError,
    FileIdentity,
    HandlerServices,
    WorkflowPlan,
    WorkflowStep,
    artifact_path,
    create_output_directory,
    directory_identity,
    directory_tree_digest,
    prepare_output_parent,
    require_callable_shape,
    require_authorized,
    validate_artifact_preconditions,
)
from ..review.adapters import (
    ActiveLearningReviewConfig,
    FullImageReviewConfig,
    ReviewTableStore,
    launch_active_learning_review,
    launch_full_image_review,
)
from ..review.workflow import ReviewAdvanceRequest, advance_review_paths
from ._shared import artifact, make_plan


VARIANTS = ("al-review-ui", "full-image-review-ui")


@dataclass(frozen=True)
class ReviewWorkflowServices:
    common: HandlerServices
    launch_al: Callable[[ActiveLearningReviewConfig], FileIdentity] = launch_active_learning_review
    launch_full: Callable[[FullImageReviewConfig], None] = launch_full_image_review

    def __post_init__(self) -> None:
        if not isinstance(self.common, HandlerServices):
            raise ContractError("review execution requires typed handler services")
        self.validate_call_shapes()

    def validate_call_shapes(self) -> None:
        """Validate launcher arity without starting either interface."""

        if not isinstance(self.common, HandlerServices):
            raise ContractError("review execution requires typed handler services")
        self.common.validate_call_shapes()
        require_callable_shape(
            self.launch_al,
            "active-learning review launcher",
            object(),
        )
        require_callable_shape(
            self.launch_full,
            "full-image review launcher",
            object(),
        )


def plan_review(config: DomainConfig, variant: str | None = None) -> WorkflowPlan:
    values = config.workflow("review")
    selected = variant or str(values["variant"])
    if selected not in VARIANTS:
        raise ContractError("unknown review variant")
    inputs = (
        artifact(values["image_root"], "image_root", "sealed-review-image-root"),
        artifact(values["detections"], "detections", "review-candidate-table"),
        artifact(values["images"], "images", "image-directory"),
    )
    outputs = (artifact(values["review_output"], "review_output", "review-output", output=True),)
    parameters = {
        "current_image": values["current_image"],
        "review_mode": values["review_mode"],
        "host": values.get("host", "127.0.0.1"),
        "port": values.get("port"),
    }
    steps = (
        WorkflowStep(
            "review-preflight",
            "validate-review-contract",
            ("image_root", "detections", "images"),
        ),
        WorkflowStep(
            "review-advance",
            "resolve-next-image-without-source-mutation",
            ("image_root", "detections", "images"),
            (),
            {
                "current_image": values["current_image"],
                "review_mode": values["review_mode"],
            },
            ("NB-LIVE-0001-C0009",),
        ),
        WorkflowStep("review-launch", selected, ("detections", "images"), ("review_output",), parameters),
    )
    return make_plan(config, "review", selected, "compag_curation.domain.review:plan_review", inputs, outputs, steps)


# SOURCE_CELL: NB-LIVE-0001-C0009
# SOURCE_STATEMENT_MAP: review-path-contract -> run_al_review/typed-adapter
def run_al_review(
    config: DomainConfig,
    services: ReviewWorkflowServices,
) -> None:
    """Launch the two-stage active-learning reviewer with explicit locators."""

    if not isinstance(services, ReviewWorkflowServices):
        raise ContractError("review handler requires ReviewWorkflowServices")
    services.validate_call_shapes()
    values = config.workflow("review")
    if values["variant"] != "al-review-ui":
        raise ContractError("active-learning review handler requires al-review-ui")
    require_authorized(config.execution)
    validate_artifact_preconditions(plan_review(config, "al-review-ui"))
    paths = advance_review_paths(
        ReviewAdvanceRequest(
            image_root=artifact_path(values, "image_root"),
            current_image=str(values["current_image"]),
            mode=str(values["review_mode"]),
            detections=artifact_path(values, "detections"),
            images=artifact_path(values, "images"),
        )
    )
    output = artifact_path(values, "review_output")
    prepare_output_parent(output)
    table_store = ReviewTableStore()
    initial_table = table_store.create(output)
    request = ActiveLearningReviewConfig(
        shortlist=paths.detections,
        detections=paths.detections,
        images=paths.tiles,
        output=output,
        output_identity=initial_table.identity,
        table_store=table_store,
    )
    try:
        services.common.report(
            f"launching active-learning review: {paths.current_image} -> {paths.next_image}; "
            f"mode={paths.mode}"
        )
    finally:
        table_store.read(output, initial_table.identity)
    terminal_identity = services.launch_al(request)
    if not isinstance(terminal_identity, FileIdentity):
        raise ContractError("active-learning launcher did not return a file identity")
    table_store.read(output, terminal_identity)
    return None


# SOURCE_CELL: NB-LIVE-0001-C0009
# SOURCE_STATEMENT_MAP: review-path-contract -> run_full_image_review/typed-adapter
def run_full_image_review(
    config: DomainConfig,
    services: ReviewWorkflowServices,
) -> None:
    """Launch the full-image reviewer with explicit output/cache paths."""

    if not isinstance(services, ReviewWorkflowServices):
        raise ContractError("review handler requires ReviewWorkflowServices")
    services.validate_call_shapes()
    values = config.workflow("review")
    if values["variant"] != "full-image-review-ui":
        raise ContractError("full review handler requires full-image-review-ui")
    require_authorized(config.execution)
    validate_artifact_preconditions(plan_review(config, "full-image-review-ui"))
    paths = advance_review_paths(
        ReviewAdvanceRequest(
            image_root=artifact_path(values, "image_root"),
            current_image=str(values["current_image"]),
            mode=str(values["review_mode"]),
            detections=artifact_path(values, "detections"),
            images=artifact_path(values, "images"),
        )
    )
    output_root = artifact_path(values, "review_output")
    cache = output_root / "_dash_cache"
    request = FullImageReviewConfig(
        detections=paths.detections,
        images=paths.tiles,
        cache=cache,
        output=output_root / "review_labels.csv",
        seen=cache / "seen.json",
        host=str(values.get("host", "127.0.0.1")),
        port=int(values.get("port", 8050)),
    )
    create_output_directory(output_root)
    output_identity = directory_identity(output_root)
    initial_digest = directory_tree_digest(output_root)
    try:
        services.common.report(
            f"launching full-image review: {paths.current_image} -> {paths.next_image}; "
            f"mode={paths.mode}"
        )
    finally:
        if (
            directory_identity(output_root) != output_identity
            or directory_tree_digest(output_root) != initial_digest
        ):
            raise ContractError("full-image review output directory identity changed")
    services.launch_full(request)
    if directory_identity(output_root) != output_identity:
        raise ContractError("full-image review output directory identity changed")
    return None


__all__ = [
    "ReviewWorkflowServices",
    "plan_review",
    "run_al_review",
    "run_full_image_review",
]
