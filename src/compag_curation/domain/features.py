"""Feature-extraction plan and concrete execution service."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Protocol

from ..config import DomainConfig
from ..contracts import (
    ContractError,
    HandlerServices,
    WorkflowPlan,
    WorkflowStep,
    artifact_path,
    prepare_output_parent,
    regular_file_snapshot,
    require_callable_shape,
    require_authorized,
    _stat_identity,
    validate_artifact_preconditions,
)
from ..features.extraction import (
    FeatureBackend,
    FeatureExtractionRequest,
    FeatureExtractionResult,
    FeaturePackRequest,
    FeaturePackResult,
    ReviewMergeRequest,
    SplitRequest,
    build_fold_safe_feature_pack,
    extract_features_fold_safe,
    freeze_or_create_splits,
    merge_review_labels_into_coco,
)
from .rounds import (
    RoundBackupHelperBinding,
    RoundBootstrapRequest,
    RoundContext,
    RoundNotification,
    bootstrap_round,
    bind_round_backup_helper,
    build_round_notification,
    deliver_round_notification,
)
from ._shared import artifact, make_plan


VARIANTS = ("coco-features",)


@dataclass(frozen=True)
class FeatureWorkflowServices:
    common: HandlerServices
    backend_factory: "FeatureBackendFactory"
    optional_sound: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.common, HandlerServices):
            raise ContractError("feature execution requires typed handler services")
        self.validate_call_shapes()

    def validate_call_shapes(self) -> None:
        """Validate injected call signatures without invoking a service."""

        if not isinstance(self.common, HandlerServices):
            raise ContractError("feature execution requires typed handler services")
        self.common.validate_call_shapes()
        require_callable_shape(
            self.backend_factory,
            "feature backend factory",
            Path("checkpoint"),
            "0" * 64,
            device="cpu",
        )
        if self.optional_sound is not None:
            require_callable_shape(
                self.optional_sound,
                "feature optional sound service",
            )

    def emit_optional_sound(self) -> bool:
        """Invoke an explicitly injected completion sound without failing the workflow."""

        if self.optional_sound is None:
            return False
        try:
            self.optional_sound()
        except Exception:
            return False
        return True


class FeatureBackendFactory(Protocol):
    def __call__(
        self,
        checkpoint: Path,
        checkpoint_sha256: str,
        *,
        device: str,
    ) -> FeatureBackend: ...


def _read_training_image_ids(path: Path) -> tuple[int, ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("split manifest cannot be read as UTF-8 JSON") from exc
    if not isinstance(value, dict) or not isinstance(value.get("train"), list):
        raise ContractError("split manifest must contain an explicit train ID list")
    raw_ids = value["train"]
    if (
        not raw_ids
        or any(not isinstance(item, int) or isinstance(item, bool) for item in raw_ids)
        or len(set(raw_ids)) != len(raw_ids)
    ):
        raise ContractError("split manifest train IDs must be nonempty unique integers")
    return tuple(sorted(raw_ids))


def _require_feature_pack_outputs(
    request: FeaturePackRequest,
    result: FeaturePackResult,
) -> None:
    if result.pack_output != request.pack_output:
        raise ContractError("feature-pack result does not bind the declared pack output")
    for label, path in (
        ("pack", request.pack_output),
        ("prototype", request.prototype_output),
        ("PCA", request.pca_output),
        ("used-ID", request.used_ids_output),
        ("positive-embedding", request.positive_embeddings_output),
        ("all-embedding", request.all_embeddings_output),
    ):
        try:
            status = path.lstat()
        except OSError as exc:
            raise ContractError(f"feature {label} output was not produced") from exc
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise ContractError(f"feature {label} output must be a regular non-symlink file")


def _require_preprocessing_output(path: Path, label: str) -> None:
    try:
        status = path.lstat()
    except OSError as exc:
        raise ContractError(f"feature {label} output was not produced") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ContractError(f"feature {label} output must be a regular non-symlink file")


def _require_feature_extraction_output(
    expected: Path,
    result: FeatureExtractionResult,
) -> None:
    if result.output_csv != expected:
        raise ContractError("feature result does not bind the declared CSV output")
    try:
        status = expected.lstat()
    except OSError as exc:
        raise ContractError("feature CSV output was not produced") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ContractError("feature CSV output must be a regular non-symlink file")


@dataclass(frozen=True)
class FeaturePreprocessingSequence:
    checkpoint: Path
    checkpoint_sha256: str
    device: str
    round: RoundBootstrapRequest
    review_merge: ReviewMergeRequest
    split: SplitRequest
    feature_pack: FeaturePackRequest
    notification_title: str
    notification_text: str

    def __post_init__(self) -> None:
        if len(self.checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.checkpoint_sha256
        ):
            raise ContractError("feature preprocessing checkpoint SHA-256 is invalid")
        if self.device != "cpu":
            raise ContractError("feature preprocessing is sealed to CPU")


@dataclass(frozen=True)
class FeaturePreprocessingResult:
    notification: RoundNotification
    allowed_image_ids: tuple[int, ...]
    backend: FeatureBackend
    round_context: RoundContext
    backup_helper: RoundBackupHelperBinding


@dataclass(frozen=True)
class FeatureWorkflowResult:
    """Explicit workflow state returned instead of ambient notebook exports."""

    extraction: FeatureExtractionResult
    round_context: RoundContext
    backup_helper: RoundBackupHelperBinding

    @property
    def output_csv(self) -> Path:
        return self.extraction.output_csv

    @property
    def row_count(self) -> int:
        return self.extraction.row_count

    @property
    def feature_count(self) -> int:
        return self.extraction.feature_count

    @property
    def appended_rows(self) -> int:
        return self.extraction.appended_rows

    @property
    def skipped_existing(self) -> int:
        return self.extraction.skipped_existing

    @property
    def migrations(self) -> tuple[str, ...]:
        return self.extraction.migrations


def _run_feature_preprocessing_sequence(
    sequence: FeaturePreprocessingSequence,
    services: FeatureWorkflowServices,
) -> FeaturePreprocessingResult:
    """Bind the five source stages through explicit typed results."""

    if sequence.device != "cpu":
        raise ContractError("feature preprocessing is sealed to CPU")
    round_context = bootstrap_round(sequence.round, services.common)
    backup_helper = bind_round_backup_helper()
    if (
        backup_helper.runtime_files_written != 0
        or backup_helper.search_path_mutations != 0
        or backup_helper.ambient_environment_allowed is not False
    ):
        raise ContractError("feature workflow requires the inert checked-in backup helper")
    merge_request = replace(
        sequence.review_merge,
        backup_directory=(
            sequence.review_merge.backup_directory
            or round_context.backup_directory
        ),
    )
    merge_review_labels_into_coco(merge_request, services.common)
    for label, path in (
        ("merged original COCO", merge_request.merged_original_coco),
        ("merged tiled COCO", merge_request.merged_tiled_coco),
        ("round new-ID", merge_request.round_new_ids_output),
    ):
        _require_preprocessing_output(path, label)
    split_request = replace(
        sequence.split,
        backup_directory=(
            sequence.split.backup_directory
            or round_context.backup_directory
        ),
    )
    split = freeze_or_create_splits(split_request, services.common)
    _require_preprocessing_output(split_request.output_manifest, "split manifest")
    if _read_training_image_ids(split_request.output_manifest) != split.train:
        raise ContractError("split result does not bind the declared train manifest")
    pack_request = replace(
        sequence.feature_pack,
        allowed_original_ids=split.train,
    )
    backend = services.backend_factory(
        sequence.checkpoint,
        sequence.checkpoint_sha256,
        device=sequence.device,
    )
    pack = build_fold_safe_feature_pack(
        pack_request,
        backend,
        services.common,
    )
    _require_feature_pack_outputs(pack_request, pack)
    if not pack.used_image_ids:
        raise ContractError("feature pack produced no allowed tiled image identifiers")
    return FeaturePreprocessingResult(
        notification=build_round_notification(
            sequence.notification_title,
            sequence.notification_text,
            finished_at=services.common.utc_now(),
        ),
        allowed_image_ids=pack.used_image_ids,
        backend=backend,
        round_context=round_context,
        backup_helper=backup_helper,
    )


def _artifact_sha256(values: Mapping[str, object], name: str) -> str:
    value = values.get(name)
    digest = value.get("sha256") if isinstance(value, Mapping) else None
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ContractError(f"feature {name} requires an explicit SHA-256")
    return digest


def _copy_sealed_master(source: Path, destination: Path, expected_sha256: str) -> None:
    """Descriptor-copy one sealed master into a distinct declared output."""

    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ContractError("feature master SHA-256 is invalid")

    try:
        source_status = source.lstat()
    except OSError as exc:
        raise ContractError("feature master input is unavailable") from exc
    if stat.S_ISLNK(source_status.st_mode) or not stat.S_ISREG(source_status.st_mode):
        raise ContractError("feature master input must be a regular non-symlink file")
    prepare_output_parent(destination)

    source_descriptor = -1
    destination_descriptor = -1
    destination_inode: tuple[int, int] | None = None
    try:
        read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        write_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        source_descriptor = os.open(source, read_flags)
        opened_source = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(opened_source.st_mode)
            or _stat_identity(opened_source) != _stat_identity(source_status)
        ):
            raise ContractError("feature master changed before copy")

        destination_descriptor = os.open(destination, write_flags, 0o600)
        os.fchmod(destination_descriptor, 0o600)
        opened_destination = os.fstat(destination_descriptor)
        destination_inode = (opened_destination.st_dev, opened_destination.st_ino)
        if not stat.S_ISREG(opened_destination.st_mode) or opened_destination.st_nlink != 1:
            raise ContractError("feature mutable master output is not a new regular file")

        digest = hashlib.sha256()
        while chunk := os.read(source_descriptor, 1024 * 1024):
            digest.update(chunk)
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(destination_descriptor, remaining)
                if written <= 0:
                    raise ContractError("feature mutable master output write did not advance")
                remaining = remaining[written:]
        os.fsync(destination_descriptor)

        completed_source = os.fstat(source_descriptor)
        completed_destination = os.fstat(destination_descriptor)
        if _stat_identity(completed_source) != _stat_identity(opened_source):
            raise ContractError("feature master changed during copy")
        if (
            not stat.S_ISREG(completed_destination.st_mode)
            or completed_destination.st_nlink != 1
            or completed_destination.st_size != completed_source.st_size
        ):
            raise ContractError("feature mutable master output is incomplete")
        if digest.hexdigest() != expected_sha256:
            raise ContractError("feature master SHA-256 changed before copy")

        source_after = source.lstat()
        destination_after = destination.lstat()
        if _stat_identity(source_after) != _stat_identity(completed_source):
            raise ContractError("feature master path changed during copy")
        if _stat_identity(destination_after) != _stat_identity(completed_destination):
            raise ContractError("feature mutable master path changed during copy")
    except (OSError, ContractError) as exc:
        if destination_inode is not None:
            try:
                current = destination.lstat()
                if (current.st_dev, current.st_ino) == destination_inode:
                    destination.unlink()
            except OSError:
                pass
        if isinstance(exc, ContractError):
            raise
        raise ContractError("feature mutable master output could not be created") from exc
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)


def plan_feature_extraction(config: DomainConfig, variant: str | None = None) -> WorkflowPlan:
    values = config.workflow("extract-features")
    selected = variant or str(values["variant"])
    if selected not in VARIANTS:
        raise ContractError("unknown feature-extraction variant")
    threshold = values.get("decision_threshold", 0.5)
    device = str(values["device"])
    inputs = (
        artifact(values["original_coco"], "original_coco", "sealed-original-annotation-contract"),
        artifact(values["coco_annotations"], "coco_annotations", "sealed-tiled-annotation-contract"),
        artifact(values["detections"], "detections", "sealed-detection-table"),
        artifact(values["review_labels"], "review_labels", "sealed-review-labels"),
        artifact(values["tiles"], "tiles", "image-directory"),
        artifact(values["sam2_checkpoint"], "sam2_checkpoint", "sealed-model"),
        artifact(values["model_root"], "model_root", "sealed-existing-model-directory"),
        artifact(
            values["previous_split_manifest"],
            "previous_split_manifest",
            "sealed-predecessor-split",
        ),
    )
    output_roles = (
        ("merged_original_coco", "merged-original-annotation-contract"),
        ("merged_tiled_coco", "merged-tiled-annotation-contract"),
        ("round_new_ids", "round-new-original-identifiers"),
        ("split_manifest", "frozen-split-output"),
        ("prototype", "feature-prototype-output"),
        ("pca", "pca-output"),
        ("used_ids", "used-image-identifiers-output"),
        ("positive_embeddings", "positive-embeddings-output"),
        ("all_embeddings", "all-embeddings-output"),
        ("pack", "fold-safe-pack-output"),
        ("output", "feature-output"),
    )
    outputs = tuple(
        artifact(values[name], name, role, output=True)
        for name, role in output_roles
    )
    steps = (
        WorkflowStep("feature-preflight", "validate-feature-assets", tuple(item.name for item in inputs)),
        WorkflowStep(
            "feature-bootstrap",
            "build-explicit-round-context",
            ("model_root",),
            (),
            {"image_name": values["image_name"], "mode": values["mode"]},
            ("NB-LIVE-0001-C0000",),
        ),
        WorkflowStep(
            "feature-review-merge",
            "merge-reviewed-detections-into-declared-coco-outputs",
            ("original_coco", "coco_annotations", "detections", "review_labels"),
            ("merged_original_coco", "merged_tiled_coco", "round_new_ids"),
            {},
            ("NB-LIVE-0001-C0001",),
        ),
        WorkflowStep(
            "feature-split",
            "freeze-and-append-training-split",
            ("merged_original_coco", "previous_split_manifest", "round_new_ids"),
            ("split_manifest",),
            {"seed": 42, "validation_fraction": 0.0, "test_fraction": 0.0},
            ("NB-LIVE-0001-C0002",),
        ),
        WorkflowStep(
            "feature-pack",
            "build-train-only-prototype-pca-pack",
            (
                "merged_tiled_coco",
                "tiles",
                "sam2_checkpoint",
                "split_manifest",
            ),
            ("prototype", "pca", "used_ids", "positive_embeddings", "all_embeddings", "pack"),
            {"pca_dimensions": 32, "fixed_threshold": 0.70, "device": device},
            ("NB-LIVE-0001-C0003",),
        ),
        WorkflowStep(
            "feature-extract",
            "extract-coco-tile-features",
            ("merged_tiled_coco", "tiles", "prototype", "pca", "split_manifest", "review_labels"),
            ("output",),
            {"decision_threshold": threshold},
            ("NB-LIVE-0001-C0004",),
        ),
    )
    return make_plan(
        config,
        "extract-features",
        selected,
        "compag_curation.domain.features:plan_feature_extraction",
        inputs,
        outputs,
        steps,
        {"decision_threshold": threshold, "device": device},
    )


# SOURCE_CELL: NB-LIVE-0001-C0000
# SOURCE_CELL: NB-LIVE-0001-C0001
# SOURCE_CELL: NB-LIVE-0001-C0002
# SOURCE_CELL: NB-LIVE-0001-C0003
# SOURCE_CELL: NB-LIVE-0001-C0004
# SOURCE_STATEMENT_MAP: workflow-closure -> run_feature_extraction_workflow/typed-orchestrator
def run_feature_extraction_workflow(
    config: DomainConfig,
    services: FeatureWorkflowServices,
) -> FeatureWorkflowResult:
    """Execute the reviewed fold-safe extraction implementation."""

    if not isinstance(services, FeatureWorkflowServices):
        raise ContractError("feature handler requires FeatureWorkflowServices")
    services.validate_call_shapes()
    values = config.workflow("extract-features")
    if values["variant"] != "coco-features":
        raise ContractError("feature handler requires coco-features variant")
    require_authorized(config.execution)
    plan = plan_feature_extraction(config, "coco-features")
    validate_artifact_preconditions(plan)
    checkpoint_sha256 = _artifact_sha256(values, "sam2_checkpoint")
    merged_original = artifact_path(values, "merged_original_coco")
    merged_tiled = artifact_path(values, "merged_tiled_coco")
    sequence = FeaturePreprocessingSequence(
        checkpoint=artifact_path(values, "sam2_checkpoint"),
        checkpoint_sha256=checkpoint_sha256,
        device=str(values["device"]),
        round=RoundBootstrapRequest(
            image_name=str(values["image_name"]),
            mode=str(values["mode"]),
            output_root=merged_original.parent,
            artifact_root=artifact_path(values, "pack").parent,
            split_root=artifact_path(values, "split_manifest").parent,
            feature_output=artifact_path(values, "output"),
            model_root=artifact_path(values, "model_root"),
            requested_model_name=str(
                values.get(
                    "requested_model_name",
                    "cj_ultra_tilesafe_xgb_r17_hybrid.pkl",
                )
            ),
            backup_existing=False,
        ),
        review_merge=ReviewMergeRequest(
            original_coco=artifact_path(values, "original_coco"),
            tiled_coco=artifact_path(values, "coco_annotations"),
            merged_original_coco=merged_original,
            merged_tiled_coco=merged_tiled,
            detections_csv=artifact_path(values, "detections"),
            review_labels_csv=artifact_path(values, "review_labels"),
            round_new_ids_output=artifact_path(values, "round_new_ids"),
            mode=str(values["mode"]),
        ),
        split=SplitRequest(
            original_coco=merged_original,
            existing_split_manifest=artifact_path(values, "previous_split_manifest"),
            new_original_ids=artifact_path(values, "round_new_ids"),
            output_manifest=artifact_path(values, "split_manifest"),
        ),
        feature_pack=FeaturePackRequest(
            coco_annotations=merged_tiled,
            images=artifact_path(values, "tiles"),
            allowed_original_ids=(),
            prototype_output=artifact_path(values, "prototype"),
            pca_output=artifact_path(values, "pca"),
            used_ids_output=artifact_path(values, "used_ids"),
            positive_embeddings_output=artifact_path(values, "positive_embeddings"),
            all_embeddings_output=artifact_path(values, "all_embeddings"),
            pack_output=artifact_path(values, "pack"),
        ),
        notification_title="COMPAG Curation",
        notification_text="Feature workflow completed",
    )
    for output_name in (
        "round_new_ids",
        "split_manifest",
        "prototype",
        "pca",
        "used_ids",
        "positive_embeddings",
        "all_embeddings",
        "pack",
        "output",
    ):
        prepare_output_parent(artifact_path(values, output_name))
    preprocessing = _run_feature_preprocessing_sequence(sequence, services)
    allowed_image_ids = tuple(preprocessing.allowed_image_ids)
    result = extract_features_fold_safe(
        FeatureExtractionRequest(
            coco_annotations=merged_tiled,
            images=artifact_path(values, "tiles"),
            prototype=artifact_path(values, "prototype"),
            pca=artifact_path(values, "pca"),
            output_csv=artifact_path(values, "output"),
            allowed_image_ids=allowed_image_ids,
            review_labels=artifact_path(values, "review_labels"),
        ),
        preprocessing.backend,
        services.common,
    )
    _require_feature_extraction_output(artifact_path(values, "output"), result)
    output_snapshots = {
        output_name: regular_file_snapshot(artifact_path(values, output_name))
        for output_name in (
            "merged_original_coco",
            "merged_tiled_coco",
            "round_new_ids",
            "split_manifest",
            "prototype",
            "pca",
            "used_ids",
            "positive_embeddings",
            "all_embeddings",
            "pack",
            "output",
        )
    }
    try:
        services.emit_optional_sound()
        deliver_round_notification(
            preprocessing.notification,
            config.notifications,
            services.common,
        )
    finally:
        if any(
            regular_file_snapshot(artifact_path(values, output_name)) != snapshot
            for output_name, snapshot in output_snapshots.items()
        ):
            raise ContractError("feature workflow output changed during completion delivery")
    return FeatureWorkflowResult(
        result,
        preprocessing.round_context,
        preprocessing.backup_helper,
    )


__all__ = [
    "FeatureWorkflowServices",
    "FeaturePreprocessingSequence",
    "FeaturePreprocessingResult",
    "FeatureWorkflowResult",
    "plan_feature_extraction",
    "run_feature_extraction_workflow",
]
