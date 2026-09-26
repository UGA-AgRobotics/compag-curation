"""Proposal plans and concrete source-backed execution handlers."""

from __future__ import annotations

from ..config import DomainConfig
from dataclasses import dataclass
from pathlib import Path

from ..contracts import (
    ContractError,
    HandlerServices,
    WorkflowPlan,
    WorkflowStep,
    artifact_path,
    create_output_directory,
    directory_identity,
    directory_tree_digest,
    require_authorized,
    validate_artifact_preconditions,
)
from ._shared import artifact, make_plan
from .inference import normalize_sam2_config


VARIANTS = ("sam2-proposals", "yolo-dataset-preparation")


@dataclass(frozen=True)
class ProposalResult:
    variant: str
    output: Path


@dataclass(frozen=True)
class ProposalWorkflowServices:
    common: HandlerServices

    def __post_init__(self) -> None:
        self.validate_call_shapes()

    def validate_call_shapes(self) -> None:
        """Revalidate the common service before a proposal handler does work."""

        if not isinstance(self.common, HandlerServices):
            raise ContractError("proposal execution requires typed handler services")
        self.common.validate_call_shapes()


def plan_proposals(config: DomainConfig, variant: str | None = None) -> WorkflowPlan:
    values = config.workflow("propose")
    selected = variant or str(values["variant"])
    if selected not in VARIANTS:
        raise ContractError("propose variant must distinguish proposal generation from dataset preparation")
    if selected == "sam2-proposals":
        inputs = (
            artifact(values["images"], "images", "proposal-images"),
            artifact(
                values["sam2_package_root"],
                "sam2_package_root",
                "sealed-package-root",
            ),
            artifact(values["sam2_checkpoint"], "sam2_checkpoint", "sealed-model"),
            artifact(values["sam2_config"], "sam2_config", "sealed-model-config"),
        )
        outputs = (artifact(values["output"], "output", "proposal-output", output=True),)
        steps = (
            WorkflowStep("proposal-preflight", "validate-local-sam2-assets", ("images", "sam2_package_root", "sam2_checkpoint", "sam2_config")),
            WorkflowStep(
                "proposal-generate",
                "sam2-proposal-generation",
                ("images", "sam2_package_root", "sam2_checkpoint", "sam2_config"),
                ("output",),
                {
                    "feature_mode": values.get("feature_mode", "balanced"),
                    "sam2_policy": values.get("sam2_policy", "both"),
                    "save_all_outputs": values.get("save_all_outputs", True),
                    "device": values["device"],
                },
            ),
        )
    else:
        required = ("coco_annotations", "tiles", "split_manifest", "yolo_dataset_output")
        if any(name not in values for name in required):
            raise ContractError("yolo-dataset-preparation requires COCO, tiles, split, and output")
        inputs = (
            artifact(values["coco_annotations"], "coco_annotations", "sealed-annotation-contract"),
            artifact(values["tiles"], "tiles", "image-directory"),
            artifact(values["split_manifest"], "split_manifest", "sealed-split"),
        )
        outputs = (
            artifact(values["yolo_dataset_output"], "yolo_dataset_output", "prepared-dataset", output=True),
        )
        steps = (
            WorkflowStep("dataset-preflight", "validate-coco-and-locked-split", ("coco_annotations", "tiles", "split_manifest")),
            WorkflowStep("dataset-prepare", "prepare-yolo-dataset", ("coco_annotations", "tiles", "split_manifest"), ("yolo_dataset_output",)),
        )
    return make_plan(config, "propose", selected, "compag_curation.domain.proposals:plan_proposals", inputs, outputs, steps)


def _argument_vector(values: dict[str, object]) -> list[str]:
    package_root = values["sam2_package_root"]
    checkpoint = values["sam2_checkpoint"]
    configuration = values["sam2_config"]
    if not isinstance(package_root, dict) or not package_root.get("sha256"):
        raise ContractError("SAM2 package root requires an explicit SHA-256")
    if not isinstance(checkpoint, dict) or not checkpoint.get("sha256"):
        raise ContractError("SAM2 checkpoint requires an explicit SHA-256")
    if not isinstance(configuration, dict) or not configuration.get("sha256"):
        raise ContractError("SAM2 configuration requires an explicit SHA-256")
    package_root_path = artifact_path(values, "sam2_package_root")
    config_path = artifact_path(values, "sam2_config")
    config_locator = normalize_sam2_config(
        config_path,
        package_root=package_root_path,
    )
    arguments = [
        "--images",
        str(artifact_path(values, "images")),
        "--out",
        str(artifact_path(values, "output")),
        "--sam2-package-root",
        str(package_root_path),
        "--sam2-package-root-sha256",
        str(package_root["sha256"]),
        "--sam2-config",
        str(config_path),
        "--sam2-config-locator",
        config_locator,
        "--sam2-config-sha256",
        str(configuration["sha256"]),
        "--sam2-ckpt",
        str(artifact_path(values, "sam2_checkpoint")),
        "--sam2-ckpt-sha256",
        str(checkpoint["sha256"]),
        "--device",
        str(values["device"]),
        "--feature-mode",
        str(values.get("feature_mode", "balanced")),
        "--sam2-policy",
        str(values.get("sam2_policy", "both")),
    ]
    arguments.append(
        "--save-all-outputs"
        if values.get("save_all_outputs", True)
        else "--no-save-all-outputs"
    )
    return arguments


# BASE_SOURCE: src/compag_curation/proposals/sam2_pipeline/pipeline.py:run_pipeline
# SOURCE_STATEMENT_MAP: retained-base-source -> generate_sam2_proposals/pipeline-closure
def generate_sam2_proposals(
    config: DomainConfig,
    services: ProposalWorkflowServices,
) -> ProposalResult:
    """Run the reviewed SAM2 proposal pipeline after controlled preflight."""

    if not isinstance(services, ProposalWorkflowServices):
        raise ContractError("proposal handler requires ProposalWorkflowServices")
    services.validate_call_shapes()
    values = config.workflow("propose")
    if values["variant"] != "sam2-proposals":
        raise ContractError("SAM2 proposal handler requires sam2-proposals variant")
    require_authorized(config.execution)
    plan = plan_proposals(config, "sam2-proposals")
    validate_artifact_preconditions(plan)
    from ..proposals.sam2_pipeline.args import make_argparser
    from ..proposals.sam2_pipeline.pipeline import run_pipeline

    arguments = make_argparser().parse_args(_argument_vector(values))
    output = artifact_path(values, "output")
    output = create_output_directory(output)
    output_identity = directory_identity(output)
    arguments.precreated_out_device = output_identity.device
    arguments.precreated_out_inode = output_identity.inode
    run_pipeline(arguments)
    if directory_identity(output) != output_identity:
        raise ContractError("SAM2 proposal output directory identity changed")
    output_digest = directory_tree_digest(output)
    try:
        services.common.report(f"SAM2 proposals completed: {output}")
    finally:
        if (
            directory_identity(output) != output_identity
            or directory_tree_digest(output) != output_digest
        ):
            raise ContractError("SAM2 proposal output directory identity changed")
    return ProposalResult("sam2-proposals", output)


# SOURCE_CELL: NB-LIVE-0008-C0000
# SOURCE_STATEMENT_MAP: top-level:000-033 -> prepare_yolo_dataset/coco-to-yolo-closure
def prepare_yolo_dataset(
    config: DomainConfig,
    services: ProposalWorkflowServices,
) -> ProposalResult:
    """Execute the distinct optional COCO-to-YOLO preparation variant."""

    if not isinstance(services, ProposalWorkflowServices):
        raise ContractError("proposal handler requires ProposalWorkflowServices")
    services.validate_call_shapes()
    values = config.workflow("propose")
    if values["variant"] != "yolo-dataset-preparation":
        raise ContractError(
            "YOLO dataset handler requires yolo-dataset-preparation variant"
        )
    require_authorized(config.execution)
    plan = plan_proposals(config, "yolo-dataset-preparation")
    validate_artifact_preconditions(plan)
    from ..preprocessing.yolo_dataset import (
        YoloDatasetRequest,
        prepare_yolo_dataset as build_dataset,
    )

    output = artifact_path(values, "yolo_dataset_output")
    build_dataset(
        YoloDatasetRequest(
            coco_annotations=artifact_path(values, "coco_annotations"),
            tiles=artifact_path(values, "tiles"),
            split_manifest=artifact_path(values, "split_manifest"),
            output=output,
        ),
        services.common,
    )
    return ProposalResult("yolo-dataset-preparation", output)


__all__ = [
    "ProposalResult",
    "ProposalWorkflowServices",
    "generate_sam2_proposals",
    "plan_proposals",
    "prepare_yolo_dataset",
]
