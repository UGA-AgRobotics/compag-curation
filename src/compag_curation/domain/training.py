"""Training plans and real XGBoost/YOLO execution handlers."""

from __future__ import annotations

import stat
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..config import DomainConfig, YOLO_DEFAULTS
from ..contracts import (
    ContractError,
    DirectoryIdentity,
    HandlerServices,
    WorkflowPlan,
    WorkflowStep,
    artifact_path,
    create_output_directory,
    directory_identity,
    directory_tree_digest,
    regular_file_snapshot,
    require_callable_shape,
    require_authorized,
    validate_artifact_preconditions,
)
from ..training.yolo import (
    YoloModelLoader,
    YoloTrainingRequest,
    YoloTrainingResult,
    load_local_yolo_model,
    train_yolo_local as execute_yolo_training,
)
from ._shared import artifact, make_plan

if TYPE_CHECKING:
    from ..training.xgb_nb0001 import (
        SourceAlgorithmBatchResult as Nb0001BatchResult,
        SourceAlgorithmDependencies as Nb0001Dependencies,
    )
    from ..training.xgb_nb6 import (
        SourceAlgorithmBatchResult as Nb6BatchResult,
        SourceAlgorithmDependencies as Nb6Dependencies,
    )
    from ..training.xgb_nb7 import (
        SourceAlgorithmBatchResult as Nb7BatchResult,
        SourceAlgorithmDependencies as Nb7Dependencies,
    )


VARIANTS = ("coco-xgb", "xgb", "xgb-main", "yolo")

_XGB_DEPENDENCY_SURFACES = {
    "coco-xgb": frozenset(
        {
            "GroupKFold", "GroupShuffleSplit", "ImbPipeline", "ParameterSampler",
            "ParserError", "SMOTE", "SimpleImputer", "TrainingCallback",
            "XGBClassifier", "average_precision_score", "confusion_matrix", "cupy",
            "joblib", "loguniform", "np", "pd", "plt",
            "precision_recall_curve", "randint", "roc_auc_score", "roc_curve",
            "tqdm", "uniform", "version", "xgb",
        }
    ),
    "xgb": frozenset(
        {
            "DummyClassifier", "GridSearchCV", "GroupKFold", "GroupShuffleSplit",
            "ImbPipeline", "ParameterGrid", "Pipeline", "RandomizedSearchCV",
            "SMOTE", "SimpleImputer", "StratifiedKFold", "StratifiedShuffleSplit",
            "TrainingCallback", "XGBClassifier", "average_precision_score", "clone",
            "confusion_matrix", "cupy", "joblib", "loguniform", "make_pipeline",
            "np", "pd", "plt", "precision_recall_curve", "randint",
            "roc_auc_score", "roc_curve", "sk_shuffle", "tqdm", "uniform",
            "version", "xgb",
        }
    ),
    "xgb-main": frozenset(
        {
            "DummyClassifier", "GroupKFold", "GroupShuffleSplit", "ImbPipeline",
            "Pipeline", "RandomizedSearchCV", "SMOTE", "SimpleImputer",
            "StratifiedKFold", "StratifiedShuffleSplit", "TrainingCallback",
            "XGBClassifier", "average_precision_score", "confusion_matrix", "cupy",
            "joblib", "loguniform", "np", "pd", "plt",
            "precision_recall_curve", "randint", "roc_auc_score", "roc_curve",
            "tqdm", "uniform", "version", "xgb",
        }
    ),
}

_XGB_MODULE_DEPENDENCY_PROBES = {
    "np": ("asarray",),
    "pd": ("read_csv",),
    "joblib": ("load", "dump"),
    "plt": ("subplots", "close"),
    "version": ("parse",),
}


def _require_xgb_dependency_surface(dependencies: object, variant: str) -> None:
    """Reject an incomplete lazy scientific bundle before artifact effects."""

    expected = _XGB_DEPENDENCY_SURFACES.get(variant)
    if expected is None:
        raise ContractError("unknown XGBoost dependency surface")
    missing = tuple(sorted(name for name in expected if not hasattr(dependencies, name)))
    if missing:
        raise ContractError(
            f"{variant} scientific dependencies are incomplete: {', '.join(missing)}"
        )
    module_names = frozenset(_XGB_MODULE_DEPENDENCY_PROBES) | {"cupy", "xgb"}
    for name in sorted(expected - module_names):
        if not callable(getattr(dependencies, name)):
            raise ContractError(f"{variant} scientific dependency is not callable: {name}")
    for name, members in _XGB_MODULE_DEPENDENCY_PROBES.items():
        component = getattr(dependencies, name)
        if any(not callable(getattr(component, member, None)) for member in members):
            raise ContractError(f"{variant} scientific module surface is invalid: {name}")
    xgb = getattr(dependencies, "xgb")
    callback = getattr(xgb, "callback", None)
    if (
        not isinstance(getattr(xgb, "__version__", None), str)
        or not callable(getattr(callback, "EvaluationMonitor", None))
    ):
        raise ContractError(f"{variant} XGBoost module surface is invalid")
    cupy = getattr(dependencies, "cupy")
    if cupy is not None and not callable(getattr(cupy, "asarray", None)):
        raise ContractError(f"{variant} CuPy dependency surface is invalid")


@dataclass(frozen=True)
class TrainingWorkflowServices:
    common: HandlerServices
    nb0001_dependencies: Nb0001Dependencies | None = None
    nb6_dependencies: Nb6Dependencies | None = None
    nb7_dependencies: Nb7Dependencies | None = None
    thread_environment_setter: Callable[[str, str], None] | None = None
    nb6_fit_with_maybe_callbacks: Callable[..., object] | None = None
    yolo_model_loader: YoloModelLoader = load_local_yolo_model

    def __post_init__(self) -> None:
        if not isinstance(self.common, HandlerServices):
            raise ContractError("training execution requires typed handler services")
        self.common.validate_call_shapes()
        for label, callback in (
            ("thread environment setter", self.thread_environment_setter),
            ("fit callback", self.nb6_fit_with_maybe_callbacks),
        ):
            if callback is not None and not callable(callback):
                raise ContractError(f"training {label} must be callable")
        if not callable(self.yolo_model_loader):
            raise ContractError("training YOLO model loader must be callable")

    def validate_call_shapes(self, variant: str) -> None:
        """Validate the selected training DI surface before artifact preflight."""

        if not isinstance(self.common, HandlerServices):
            raise ContractError("training execution requires typed handler services")
        self.common.validate_call_shapes()
        if variant == "yolo":
            require_callable_shape(
                self.yolo_model_loader,
                "training YOLO model loader",
                Path("weights.pt"),
            )
            return
        require_callable_shape(
            self.thread_environment_setter,
            "training thread environment setter",
            "OMP_NUM_THREADS",
            "1",
        )
        dependencies = {
            "coco-xgb": self.nb0001_dependencies,
            "xgb": self.nb6_dependencies,
            "xgb-main": self.nb7_dependencies,
        }.get(variant)
        if dependencies is None:
            raise ContractError(f"{variant} execution requires injected scientific dependencies")
        _require_xgb_dependency_surface(dependencies, variant)
        if variant == "xgb":
            require_callable_shape(
                self.nb6_fit_with_maybe_callbacks,
                "training fit callback",
                object(),
                object(),
                object(),
                eval_set=[],
                verbose=False,
                callbacks=[],
            )


@dataclass(frozen=True)
class LocalYoloResult:
    model_output: Path
    best_weights: Path
    run: YoloTrainingResult


def plan_training(config: DomainConfig, variant: str | None = None) -> WorkflowPlan:
    values = config.workflow("train")
    selected = variant or str(values["variant"])
    if selected not in VARIANTS:
        raise ContractError("unknown training variant")
    if selected == "yolo":
        if config.execution.allow_download:
            raise ContractError("YOLO training is restricted to explicit local weights")
        inputs = (
            artifact(values["data_yaml"], "data_yaml", "sealed-yolo-data-contract"),
            artifact(values["dataset_root"], "dataset_root", "sealed-yolo-dataset-root"),
            artifact(
                values["local_initial_weights"],
                "local_initial_weights",
                "local-model-asset",
            ),
        )
        parameters = {name: values[name] for name in YOLO_DEFAULTS}
        parameters.update(
            {
                "local_weights_only": True,
                "allow_download": False,
            }
        )
        defaults = dict(YOLO_DEFAULTS)
    else:
        common_inputs = (
            artifact(values["feature_table"], "feature_table", "sealed-feature-table"),
            artifact(values["split_manifest"], "split_manifest", "sealed-split"),
        )
        if selected == "coco-xgb":
            inputs = common_inputs + (
                artifact(
                    values["previous_tile_predictions"],
                    "previous_tile_predictions",
                    "sealed-previous-tile-predictions",
                ),
                artifact(
                    values["previous_threshold"],
                    "previous_threshold",
                    "sealed-previous-threshold",
                ),
            )
            if "previous_best_parameters" in values:
                inputs += (
                    artifact(
                        values["previous_best_parameters"],
                        "previous_best_parameters",
                        "sealed-previous-best-parameters",
                    ),
                )
        elif selected == "xgb-main":
            inputs = common_inputs + tuple(
                artifact(values[name], name, f"sealed-{name.replace('_', '-')}")
                for name in (
                    "hybrid_previous_tile_predictions",
                    "hybrid_previous_threshold",
                    "augmented_previous_tile_predictions",
                    "augmented_previous_threshold",
                )
            )
        else:
            inputs = common_inputs
        threshold = values.get("decision_threshold", 0.5)
        parameters = {
            "seed": 42,
            "device": values.get("device", "cpu"),
            "thread_count": values.get("thread_count", 1),
            "decision_threshold": threshold,
            "feature_order_sha256": values.get("feature_order_sha256"),
            "persisted_estimator": "compag_curation.training.safe_smote.SafeSMOTE",
            "source_algorithm_dispatch": selected,
        }
        defaults = {
            "decision_threshold": threshold,
            "seed": 42,
            "device": values.get("device", "cpu"),
            "thread_count": values.get("thread_count", 1),
        }
    outputs = (
        artifact(values["model_output"], "model_output", "training-output-directory", output=True),
    )
    steps = (
        WorkflowStep("training-preflight", "validate-training-inputs", tuple(item.name for item in inputs)),
        WorkflowStep(
            "training-orchestrate",
            f"orchestrate-{selected}",
            tuple(item.name for item in inputs),
            ("model_output",),
            parameters,
        ),
    )
    return make_plan(
        config,
        "train",
        selected,
        "compag_curation.domain.training:plan_training",
        inputs,
        outputs,
        steps,
        defaults,
    )


def _validated_xgb_context(
    config: DomainConfig,
    services: TrainingWorkflowServices,
    expected_variant: str,
) -> tuple[dict[str, object], Path, Callable[[str, str], None]]:
    values = config.workflow("train")
    if values["variant"] != expected_variant:
        raise ContractError(f"training handler requires {expected_variant} variant")
    require_authorized(config.execution)
    validate_artifact_preconditions(plan_training(config, expected_variant))
    setter = services.thread_environment_setter
    if setter is None or not callable(setter):
        raise ContractError("XGBoost execution requires an injected thread-environment setter")
    return values, artifact_path(values, "model_output"), setter


def _source_reporter(messages: list[str]) -> Callable[..., None]:
    def report(*items: object, **options: object) -> None:
        unknown = set(options) - {"flush"}
        if unknown or (
            "flush" in options
            and (not isinstance(options["flush"], bool) or options["flush"] is not True)
        ):
            raise ContractError("source reporter accepts only the exact flush=True option")
        messages.append(" ".join(str(item) for item in items))

    return report


def _deliver_source_reports(
    common: HandlerServices,
    messages: list[str],
    output_root: Path,
    expected_root_identity: DirectoryIdentity,
) -> None:
    output_digest = directory_tree_digest(output_root)
    try:
        for message in messages:
            common.report(message)
    finally:
        if (
            directory_identity(output_root) != expected_root_identity
            or directory_tree_digest(output_root) != output_digest
        ):
            raise ContractError("XGBoost reporting changed a declared output")


def _source_input(values: dict[str, object], name: str, constructor: type[object]) -> object:
    try:
        value = values[name]
        digest = value["sha256"]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise ContractError(f"XGBoost source artifact is missing: {name}") from exc
    if not isinstance(digest, str):
        raise ContractError(f"XGBoost source artifact SHA-256 is missing: {name}")
    return constructor(artifact_path(values, name), digest)


def _unique_source_inputs(*artifacts: object | None) -> tuple[object, ...]:
    unique: list[object] = []
    for artifact_value in artifacts:
        if artifact_value is not None and artifact_value not in unique:
            unique.append(artifact_value)
    return tuple(unique)


def _validate_source_training_artifacts(
    output_root: Path,
    expected_root_identity: DirectoryIdentity,
    artifact_sets: tuple[object, ...],
) -> None:
    if not isinstance(expected_root_identity, DirectoryIdentity):
        raise ContractError("XGBoost output directory identity must be typed")
    if directory_identity(output_root) != expected_root_identity:
        raise ContractError("XGBoost output directory identity changed")
    try:
        root_status = output_root.lstat()
        resolved_root = output_root.resolve(strict=True)
    except OSError as exc:
        raise ContractError("XGBoost output directory is unavailable") from exc
    if (
        stat.S_ISLNK(root_status.st_mode)
        or not stat.S_ISDIR(root_status.st_mode)
        or resolved_root != output_root
    ):
        raise ContractError("XGBoost output directory is not a real directory")
    if not artifact_sets:
        raise ContractError("XGBoost execution returned no artifact sets")
    for artifact_set in artifact_sets:
        if not is_dataclass(artifact_set):
            raise ContractError("XGBoost execution returned an untyped artifact set")
        for field in fields(artifact_set):
            candidate = getattr(artifact_set, field.name)
            if not isinstance(candidate, Path):
                raise ContractError("XGBoost artifact result contains a non-path field")
            if not candidate.is_absolute() or candidate == output_root:
                raise ContractError("XGBoost artifact result is outside its declared layout")
            try:
                candidate.relative_to(output_root)
                status = candidate.lstat()
                resolved = candidate.resolve(strict=True)
            except (OSError, ValueError) as exc:
                raise ContractError("XGBoost artifact result is unavailable") from exc
            expected_directory = field.name == "output_directory"
            if (
                stat.S_ISLNK(status.st_mode)
                or resolved != candidate
                or (expected_directory and not stat.S_ISDIR(status.st_mode))
                or (not expected_directory and not stat.S_ISREG(status.st_mode))
            ):
                raise ContractError("XGBoost artifact result has the wrong filesystem type")
    if directory_identity(output_root) != expected_root_identity:
        raise ContractError("XGBoost output directory identity changed")


# SOURCE_CELL: NB-LIVE-0001-C0005
# SOURCE_CELL: NB-LIVE-0001-C0006
# SOURCE_STATEMENT_MAP: paired-cell-closure -> run_coco_xgb_training/train_xgb
def run_coco_xgb_training(
    config: DomainConfig,
    services: TrainingWorkflowServices,
) -> Nb0001BatchResult:
    if not isinstance(services, TrainingWorkflowServices):
        raise ContractError("training handler requires TrainingWorkflowServices")
    services.validate_call_shapes("coco-xgb")
    values, model_output, setter = _validated_xgb_context(
        config, services, "coco-xgb"
    )
    if services.nb0001_dependencies is None:
        raise ContractError("coco-xgb execution requires injected scientific dependencies")
    from ..training.xgb_nb0001 import (
        SourceAlgorithmBatch,
        SourceAlgorithmServices,
        SourceInputArtifact,
        SourceInputs_0001_0005,
        XgbDevicePolicy,
        execute_xgb_nb0001_source_algorithms,
    )

    feature_table = _source_input(values, "feature_table", SourceInputArtifact)
    split_manifest = _source_input(values, "split_manifest", SourceInputArtifact)
    previous_tiles = _source_input(values, "previous_tile_predictions", SourceInputArtifact)
    previous_threshold = _source_input(values, "previous_threshold", SourceInputArtifact)
    previous_best = (
        _source_input(values, "previous_best_parameters", SourceInputArtifact)
        if "previous_best_parameters" in values
        else None
    )
    source_messages: list[str] = []
    source_services = SourceAlgorithmServices(
        dependencies=services.nb0001_dependencies,
        report=_source_reporter(source_messages),
        approved_input_artifacts=_unique_source_inputs(
            feature_table,
            split_manifest,
            previous_tiles,
            previous_threshold,
            previous_best,
        ),
        cpu_count=int(values["thread_count"]),
        thread_environment_setter=setter,
        device_policy=XgbDevicePolicy(str(values["device"])),
    )
    create_output_directory(model_output)
    output_identity = directory_identity(model_output)
    result = execute_xgb_nb0001_source_algorithms(
        SourceAlgorithmBatch(
            SourceInputs_0001_0005(
                model_output,
                feature_table,
                split_manifest,
                previous_tiles,
                previous_threshold,
            ),
            previous_best,
        ),
        source_services,
        config.execution,
    )
    _validate_source_training_artifacts(
        model_output,
        output_identity,
        (result.fit.artifacts,),
    )
    _deliver_source_reports(
        services.common,
        source_messages,
        model_output,
        output_identity,
    )
    return result


# SOURCE_CELL: NB-LIVE-0006-C0000
# SOURCE_CELL: NB-LIVE-0006-C0001
# SOURCE_CELL: NB-LIVE-0006-C0002
# SOURCE_STATEMENT_MAP: classic-cell-family -> train_xgb_classic/train_xgb
def train_xgb_classic(
    config: DomainConfig,
    services: TrainingWorkflowServices,
) -> Nb6BatchResult:
    if not isinstance(services, TrainingWorkflowServices):
        raise ContractError("training handler requires TrainingWorkflowServices")
    services.validate_call_shapes("xgb")
    values, model_output, setter = _validated_xgb_context(config, services, "xgb")
    if services.nb6_dependencies is None:
        raise ContractError("xgb execution requires injected scientific dependencies")
    if services.nb6_fit_with_maybe_callbacks is None or not callable(
        services.nb6_fit_with_maybe_callbacks
    ):
        raise ContractError("xgb execution requires an injected fit callback")
    from ..training.xgb_nb6 import (
        SourceAlgorithmBatch,
        SourceAlgorithmServices,
        SourceInputArtifact,
        SourceInputs_0006_0000,
        SourceInputs_0006_0001,
        SourceInputs_0006_0002,
        XgbDevicePolicy,
        execute_xgb_nb6_source_algorithms,
    )

    feature_table = _source_input(values, "feature_table", SourceInputArtifact)
    split_manifest = _source_input(values, "split_manifest", SourceInputArtifact)
    source_messages: list[str] = []
    source_services = SourceAlgorithmServices(
        dependencies=services.nb6_dependencies,
        report=_source_reporter(source_messages),
        approved_input_artifacts=_unique_source_inputs(feature_table, split_manifest),
        cpu_count=int(values["thread_count"]),
        thread_environment_setter=setter,
        device_policy=XgbDevicePolicy(str(values["device"])),
    )
    create_output_directory(model_output)
    output_identity = directory_identity(model_output)
    result = execute_xgb_nb6_source_algorithms(
        SourceAlgorithmBatch(
            SourceInputs_0006_0000(
                model_output / "round_8",
                feature_table,
                split_manifest,
                services.nb6_fit_with_maybe_callbacks,
            ),
            SourceInputs_0006_0001(model_output / "round_9", feature_table, split_manifest),
            SourceInputs_0006_0002(
                model_output / "round_10_smote", feature_table, split_manifest
            ),
        ),
        source_services,
        config.execution,
    )
    _validate_source_training_artifacts(
        model_output,
        output_identity,
        tuple(item.artifacts for item in result.results),
    )
    _deliver_source_reports(
        services.common,
        source_messages,
        model_output,
        output_identity,
    )
    return result


# SOURCE_CELL: NB-LIVE-0007-C0000
# SOURCE_CELL: NB-LIVE-0007-C0001
# SOURCE_CELL: NB-LIVE-0007-C0006
# SOURCE_CELL: NB-LIVE-0007-C0007
# SOURCE_STATEMENT_MAP: main-cell-family -> train_xgb_main/train_xgb
def train_xgb_main(
    config: DomainConfig,
    services: TrainingWorkflowServices,
) -> Nb7BatchResult:
    if not isinstance(services, TrainingWorkflowServices):
        raise ContractError("training handler requires TrainingWorkflowServices")
    services.validate_call_shapes("xgb-main")
    values, model_output, setter = _validated_xgb_context(config, services, "xgb-main")
    if services.nb7_dependencies is None:
        raise ContractError("xgb-main execution requires injected scientific dependencies")
    from ..training.xgb_nb7 import (
        SourceAlgorithmBatch,
        SourceAlgorithmServices,
        SourceInputArtifact,
        SourceInputs_0007_0000,
        SourceInputs_0007_0006,
        XgbDevicePolicy,
        execute_xgb_nb7_source_algorithms,
    )

    feature_table = _source_input(values, "feature_table", SourceInputArtifact)
    split_manifest = _source_input(values, "split_manifest", SourceInputArtifact)
    hybrid_tiles = _source_input(
        values, "hybrid_previous_tile_predictions", SourceInputArtifact
    )
    hybrid_threshold = _source_input(values, "hybrid_previous_threshold", SourceInputArtifact)
    augmented_tiles = _source_input(
        values, "augmented_previous_tile_predictions", SourceInputArtifact
    )
    augmented_threshold = _source_input(
        values, "augmented_previous_threshold", SourceInputArtifact
    )
    source_messages: list[str] = []
    source_services = SourceAlgorithmServices(
        dependencies=services.nb7_dependencies,
        report=_source_reporter(source_messages),
        approved_input_artifacts=_unique_source_inputs(
            feature_table,
            split_manifest,
            hybrid_tiles,
            hybrid_threshold,
            augmented_tiles,
            augmented_threshold,
        ),
        cpu_count=int(values["thread_count"]),
        thread_environment_setter=setter,
        device_policy=XgbDevicePolicy(str(values["device"])),
    )
    create_output_directory(model_output)
    output_identity = directory_identity(model_output)
    result = execute_xgb_nb7_source_algorithms(
        SourceAlgorithmBatch(
            SourceInputs_0007_0000(
                model_output / "hybrid",
                feature_table,
                split_manifest,
                hybrid_tiles,
                hybrid_threshold,
            ),
            SourceInputs_0007_0006(
                model_output / "augmented",
                feature_table,
                split_manifest,
                augmented_tiles,
                augmented_threshold,
            ),
        ),
        source_services,
        config.execution,
    )
    _validate_source_training_artifacts(
        model_output,
        output_identity,
        tuple(item.artifacts for item in result.results),
    )
    _deliver_source_reports(
        services.common,
        source_messages,
        model_output,
        output_identity,
    )
    return result


# SOURCE_CELL: NB-LIVE-0008-C0002
# SOURCE_STATEMENT_MAP: top-level:000-017 -> train_yolo_local/local-only-closure
def train_yolo_local(
    config: DomainConfig,
    services: TrainingWorkflowServices,
) -> LocalYoloResult:
    if not isinstance(services, TrainingWorkflowServices):
        raise ContractError("training handler requires TrainingWorkflowServices")
    services.validate_call_shapes("yolo")
    values = config.workflow("train")
    if values["variant"] != "yolo":
        raise ContractError("YOLO training handler requires yolo variant")
    if config.execution.allow_download:
        raise ContractError("YOLO execution never permits implicit downloads")
    require_authorized(config.execution)
    validate_artifact_preconditions(plan_training(config, "yolo"))
    model_output = artifact_path(values, "model_output")
    configured_device = values.get("device", "cpu")
    if not (
        configured_device == "cpu"
        or (
            isinstance(configured_device, int)
            and not isinstance(configured_device, bool)
            and configured_device >= 0
        )
    ):
        raise ContractError("YOLO device must be an explicit CPU or GPU index")
    run = execute_yolo_training(
        YoloTrainingRequest(
            data_yaml=artifact_path(values, "data_yaml"),
            dataset_root=artifact_path(values, "dataset_root"),
            dataset_root_sha256=str(values["dataset_root"]["sha256"]),
            initial_weights=artifact_path(values, "local_initial_weights"),
            initial_weights_sha256=str(values["local_initial_weights"]["sha256"]),
            output_root=model_output,
            device=configured_device,
            seed=int(values["seed"]),
            deterministic=bool(values["deterministic"]),
            imgsz=int(values["imgsz"]),
            epochs=int(values["epochs"]),
            batch=int(values["batch"]),
            patience=int(values["patience"]),
            lr0=float(values["lr0"]),
            weight_decay=float(values["weight_decay"]),
            close_mosaic=int(values["close_mosaic"]),
        ),
        services.common,
        model_loader=services.yolo_model_loader,
    )
    best_weights = run.save_directory / "weights" / "best.pt"
    if not best_weights.is_file() or best_weights.is_symlink():
        raise ContractError("YOLO run did not produce local best weights")
    best_weights_snapshot = regular_file_snapshot(best_weights)
    if directory_identity(model_output) != run.output_identity:
        raise ContractError("YOLO output directory identity changed")
    output_digest = directory_tree_digest(model_output)
    try:
        services.common.report(f"YOLO model directory written: {model_output}")
    finally:
        if (
            directory_identity(model_output) != run.output_identity
            or directory_tree_digest(model_output) != output_digest
        ):
            raise ContractError("YOLO output directory identity changed")
    if regular_file_snapshot(best_weights) != best_weights_snapshot:
        raise ContractError("YOLO best weights changed during completion reporting")
    return LocalYoloResult(model_output, best_weights, run)


__all__ = [
    "LocalYoloResult",
    "TrainingWorkflowServices",
    "plan_training",
    "run_coco_xgb_training",
    "train_xgb_classic",
    "train_xgb_main",
    "train_yolo_local",
]
