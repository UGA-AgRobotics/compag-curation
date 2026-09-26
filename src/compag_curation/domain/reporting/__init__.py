"""Static-source-backed report execution handlers."""

from __future__ import annotations

from csv import DictReader, DictWriter
from dataclasses import asdict, dataclass
from hashlib import sha256
from html import escape
from io import StringIO
from json import dumps, loads
from math import isfinite
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Protocol, Sequence

from ...config import DomainConfig
from ...contracts import (
    ContractError,
    DirectoryIdentity,
    FileIdentity,
    HandlerServices,
    WorkflowPlan,
    WorkflowStep,
    artifact_path,
    create_output_directory,
    directory_identity,
    directory_tree_digest,
    require_callable_shape,
    require_authorized,
    validate_artifact_preconditions,
)
from .._shared import artifact, make_plan

from .analysis import (
    ActiveLearningSimulationEvaluation,
    ActiveLearningFocusEvaluation,
    CandidatePanelEvaluation,
    CandidatePanelSettings,
    HybridEvaluation,
    InsectMaskSelectionEvaluation,
    LearningPoint,
    ModelCurveBand,
    PrecomputedActiveLearningFocusEvaluation,
    PrecomputedActiveLearningScoreColumns,
    PrecomputedCandidatePanel,
    PrecomputedHybridScoreColumns,
    PrecomputedMaskBatch,
    PrecomputedModelConfusionEvaluation,
    PrecomputedModelCurveEvaluation,
    PrecomputedModelMetricRow,
    PrecomputedModelScoreColumn,
    PrecomputedPcaPoint,
    PrecomputedReportAnalysisRequest,
    PrecomputedReportAnalysisResult,
    ReportOperation,
    SimilarityPcaEvaluation,
    analyze_precomputed_report,
    best_pca_pair,
    build_report_program,
    build_learning_curve,
    build_paper_tables,
    classification_metrics,
    confusion_matrix_report,
    evaluate_hybrid_policy,
    evaluate_precomputed_active_learning_focus,
    evaluate_precomputed_candidate_panels,
    evaluate_precomputed_c0011_confusion,
    evaluate_precomputed_c0012_confusion,
    evaluate_precomputed_insect_mask,
    evaluate_precomputed_model_curves,
    similarity_probability_points,
    where_active_learning_looks,
)
from .dataset import DatasetSummary, DatasetSummaryRequest, build_dataset_summary
VARIANTS = ("dataset-summary", "plots")


class ReportingTableIO(Protocol):
    def read_rows(self, path: Path) -> list[dict[str, str]]: ...

    def write_rows(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> None: ...


class StdlibReportingTableIO:
    def read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            return [dict(row) for row in DictReader(handle)]

    def write_rows(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


@dataclass(frozen=True)
class ReportingServices:
    common: HandlerServices
    table_io: ReportingTableIO | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.common, HandlerServices):
            raise ContractError("reporting execution requires typed handler services")
        self.validate_call_shapes()

    def validate_call_shapes(self) -> None:
        """Validate injected report services without reading or writing a table."""

        if not isinstance(self.common, HandlerServices):
            raise ContractError("reporting execution requires typed handler services")
        self.common.validate_call_shapes()
        if self.table_io is not None:
            require_callable_shape(
                getattr(self.table_io, "read_rows", None),
                "reporting table reader",
                Path("input.csv"),
            )
            require_callable_shape(
                getattr(self.table_io, "write_rows", None),
                "reporting table writer",
                Path("output.csv"),
                (),
            )


@dataclass(frozen=True)
class DatasetReportResult:
    summary: DatasetSummary
    output: Path
    json_path: Path
    latex_path: Path


@dataclass(frozen=True)
class PrecomputedBatchProductionContract:
    """Sealed production settings attested by the precomputed report manifest."""

    mode: str
    folders: tuple[str, ...]
    tiles_subdirectory: str
    environment: tuple[tuple[str, str], ...]
    asset_sha256: tuple[tuple[str, str], ...]
    pipeline_entrypoint: str
    sam2_config: str
    sam2_policy: str
    feature_mode: str
    multiscale: tuple[float, ...]
    points_per_side: int
    points_per_batch: int
    crop_n_layers: int
    crop_n_points_downscale_factor: int
    crop_overlap_ratio: float
    predicted_iou_threshold: float
    stability_threshold: float
    exclude_largest: bool
    minimum_true_gates: int
    export_features: bool
    xgb_features: str
    xgb_policy: str
    xgb_threshold: float
    yolo_enabled: bool
    yolo_confidence: float
    yolo_iou: float
    yolo_device: str
    yolo_image_size: int
    yolo_maximum_detections: int
    yolo_hint: str
    detector_policy: str
    detector_missing: str
    detector_threshold: float
    hybrid_yolo_bias: float
    per_folder_run_template: str
    log_template: str
    features_filename: str
    detections_filename: str
    merged_features_template: str
    merged_detections_template: str
    shell_strict: bool
    accepted_process_codes: tuple[int, ...]
    invalid_mode_code: int
    missing_output_code: int
    aggregate_folder_failures: bool
    merge_requires_both_tables: bool


@dataclass(frozen=True)
class PrecomputedTestsetBatch:
    """Hash-bound batch-inference outputs consumed by the report handler."""

    mode: str
    contract_manifest: Path
    contract_manifest_sha256: str
    intermediates_manifest: Path
    intermediates_manifest_sha256: str
    features: Path
    detections: Path
    operations: tuple[ReportOperation, ...]
    candidate_panels: tuple[PrecomputedCandidatePanel, ...]
    pca_scatter: tuple[PrecomputedPcaPoint, ...]
    model_score_columns: tuple[PrecomputedModelScoreColumn, ...]
    model_metric_rows: tuple[PrecomputedModelMetricRow, ...]
    hybrid_score_columns: PrecomputedHybridScoreColumns
    folders: tuple[str, ...]
    feature_rows: int
    detection_rows: int
    production: PrecomputedBatchProductionContract
    source_identity: DirectoryIdentity
    contract_manifest_identity: FileIdentity
    intermediates_manifest_identity: FileIdentity
    features_identity: FileIdentity
    detections_identity: FileIdentity
    feature_table: tuple[Mapping[str, str], ...]
    detection_table: tuple[Mapping[str, str], ...]


@dataclass(frozen=True)
class PlotsReportResult:
    output: Path
    manifest: Path
    analysis: Path
    plots: tuple[Path, ...]
    tables: tuple[Path, ...]
    precomputed_batch: PrecomputedTestsetBatch


@dataclass(frozen=True)
class _SealedReportingFile:
    path: Path
    data: bytes
    identity: FileIdentity
    sha256: str


def _read_sealed_reporting_file(
    root: Path,
    path: Path,
    role: str,
    *,
    expected_sha256: str | None = None,
) -> _SealedReportingFile:
    """Capture one confined input through a stable, non-following descriptor."""

    root_absolute = root.resolve(strict=True)
    try:
        path.relative_to(root_absolute)
        before = path.lstat()
    except (OSError, ValueError) as exc:
        raise ContractError(f"reporting {role} is unavailable or unconfined") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ContractError(f"reporting {role} must be a regular non-symlink file")
    if root_absolute not in path.resolve(strict=True).parents:
        raise ContractError(f"reporting {role} escaped source_tables")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContractError(f"reporting {role} cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or FileIdentity.from_stat(opened) != FileIdentity.from_stat(before)
        ):
            raise ContractError(f"reporting {role} changed before reading")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        completed = os.fstat(descriptor)
        if FileIdentity.from_stat(completed) != FileIdentity.from_stat(opened):
            raise ContractError(f"reporting {role} changed while reading")
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise ContractError(f"reporting {role} changed after reading") from exc
    identity = FileIdentity.from_stat(after)
    if identity != FileIdentity.from_stat(completed):
        raise ContractError(f"reporting {role} changed after reading")
    data = b"".join(chunks)
    digest = sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ContractError(f"reporting {role} SHA-256 mismatch")
    return _SealedReportingFile(path, data, identity, digest)


def _decode_sealed_reporting_csv(
    sealed: _SealedReportingFile,
    role: str,
) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    """Decode exactly the bytes captured and hashed by the reporting boundary."""

    try:
        reader = DictReader(StringIO(sealed.data.decode("utf-8-sig"), newline=""))
        fields = tuple(reader.fieldnames or ())
        rows = tuple(dict(row) for row in reader)
    except (UnicodeError, ValueError) as exc:
        raise ContractError(f"reporting {role} is not valid UTF-8 CSV") from exc
    if (
        not fields
        or any(not field for field in fields)
        or len(fields) != len(set(fields))
        or any(tuple(row) != fields or None in row for row in rows)
    ):
        raise ContractError(f"reporting {role} CSV schema changed")
    return fields, rows


def _require_reporting_file_identity(
    sealed: _SealedReportingFile,
    role: str,
) -> None:
    try:
        current = FileIdentity.from_stat(sealed.path.lstat())
    except (OSError, ContractError) as exc:
        raise ContractError(f"reporting {role} identity changed") from exc
    if current != sealed.identity or sealed.path.is_symlink():
        raise ContractError(f"reporting {role} identity changed")


def _require_reporting_output_root(
    output: Path,
    expected: DirectoryIdentity | None = None,
) -> DirectoryIdentity:
    current = directory_identity(output)
    if expected is not None and current != expected:
        raise ContractError("report output root identity changed")
    return current


def _require_reporting_output_artifact(
    output: Path,
    output_identity: DirectoryIdentity,
    path: Path,
) -> _SealedReportingFile:
    _require_reporting_output_root(output, output_identity)
    if path.parent != output:
        raise ContractError("report output artifact must be a direct child")
    sealed = _read_sealed_reporting_file(output, path, "output artifact")
    _require_reporting_output_root(output, output_identity)
    return sealed


def _csv_output_cell(value: Any) -> str:
    return "" if value is None else str(value)


def _write_reporting_rows(
    io: ReportingTableIO,
    output: Path,
    output_identity: DirectoryIdentity,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Bind an injected serializer to the claimed root and exact row contract."""

    expected_rows = tuple(dict(row) for row in rows)
    fields = tuple(
        dict.fromkeys(key for row in expected_rows for key in row)
    ) or ("status",)
    if any(type(field) is not str or not field for field in fields):
        raise ContractError("report output columns must be nonempty strings")
    io.write_rows(path, expected_rows)
    _require_reporting_output_root(output, output_identity)
    sealed = _require_reporting_output_artifact(output, output_identity, path)
    observed_fields, observed_rows = _decode_sealed_reporting_csv(
        sealed,
        "output table",
    )
    normalized = tuple(
        {
            field: _csv_output_cell(row.get(field))
            for field in fields
        }
        for row in expected_rows
    )
    if observed_fields != fields or observed_rows != normalized:
        raise ContractError("report output table does not match declared rows")
    return path


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(source: Path) -> tuple[Path, ...]:
    if source.is_file():
        return (source,)
    if not source.is_dir():
        raise ValueError(f"report source_tables is missing: {source}")
    return tuple(
        sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in {".csv", ".json"}
        )
    )


def _operation_contract_json(operation: ReportOperation) -> dict[str, Any]:
    """Render one executable report operation in its sealed portable schema."""

    return {
        "operation": operation.operation,
        "source_cells": list(operation.source_cells),
        "input_roles": list(operation.input_roles),
        "output_role": operation.output_role,
        "parameters": [
            {
                "name": name,
                "value": list(value) if isinstance(value, tuple) else value,
            }
            for name, value in operation.parameters
        ],
    }


def _sealed_number(value: object, field: str) -> float:
    if type(value) not in {int, float}:
        raise ContractError(f"candidate panel {field} must be a JSON number")
    parsed = float(value)
    if not isfinite(parsed):
        raise ContractError(f"candidate panel {field} must be finite")
    return parsed


def _sealed_integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ContractError(f"candidate panel {field} must be a JSON integer")
    return int(value)


def _sealed_number_array(value: object, field: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ContractError(f"candidate panel {field} must be a JSON array")
    return tuple(_sealed_number(item, field) for item in value)


# SOURCE_CELL: NB-LIVE-0012-C0022
# SOURCE_CELL: NB-LIVE-0012-C0023
# SOURCE_STATEMENT_MAP: exact-candidate-panel-intermediate-schema -> _candidate_panel_from_json
def _candidate_panel_from_json(value: object) -> PrecomputedCandidatePanel:
    if not isinstance(value, dict) or set(value) != {
        "folder",
        "candidate_id",
        "zoom_width",
        "zoom_height",
        "bbox",
        "mask_area",
        "ring_area",
        "predicted_iou",
        "stability_score",
        "mean_l",
        "mean_a",
        "mean_b",
        "background_a",
        "background_b",
        "delta_l_median",
        "low_clip_fraction",
        "high_clip_fraction",
        "l_standard_deviation",
        "area_norm",
        "extent",
        "elongation",
        "solidity",
        "aspect_ratio",
        "circularity",
        "touching_border",
        "mask_a",
        "mask_b",
        "ring_a",
        "ring_b",
        "gradient_magnitude",
        "pca_components",
    }:
        raise ContractError("candidate panel intermediate fields changed")
    folder = value["folder"]
    touching_border = value["touching_border"]
    bbox_value = value["bbox"]
    if not isinstance(folder, str) or not folder.strip():
        raise ContractError("candidate panel folder must be a nonempty string")
    if type(touching_border) is not bool:
        raise ContractError("candidate panel touching_border must be boolean")
    if not isinstance(bbox_value, list) or len(bbox_value) != 4:
        raise ContractError("candidate panel bbox must contain four integers")
    bbox = tuple(_sealed_integer(item, "bbox") for item in bbox_value)
    panel = PrecomputedCandidatePanel(
        folder.strip(),
        _sealed_integer(value["candidate_id"], "candidate_id"),
        _sealed_integer(value["zoom_width"], "zoom_width"),
        _sealed_integer(value["zoom_height"], "zoom_height"),
        (bbox[0], bbox[1], bbox[2], bbox[3]),
        _sealed_integer(value["mask_area"], "mask_area"),
        _sealed_integer(value["ring_area"], "ring_area"),
        _sealed_number(value["predicted_iou"], "predicted_iou"),
        _sealed_number(value["stability_score"], "stability_score"),
        _sealed_number(value["mean_l"], "mean_l"),
        _sealed_number(value["mean_a"], "mean_a"),
        _sealed_number(value["mean_b"], "mean_b"),
        _sealed_number(value["background_a"], "background_a"),
        _sealed_number(value["background_b"], "background_b"),
        _sealed_number(value["delta_l_median"], "delta_l_median"),
        _sealed_number(value["low_clip_fraction"], "low_clip_fraction"),
        _sealed_number(value["high_clip_fraction"], "high_clip_fraction"),
        _sealed_number(value["l_standard_deviation"], "l_standard_deviation"),
        _sealed_number(value["area_norm"], "area_norm"),
        _sealed_number(value["extent"], "extent"),
        _sealed_number(value["elongation"], "elongation"),
        _sealed_number(value["solidity"], "solidity"),
        _sealed_number(value["aspect_ratio"], "aspect_ratio"),
        _sealed_number(value["circularity"], "circularity"),
        touching_border,
        _sealed_number_array(value["mask_a"], "mask_a"),
        _sealed_number_array(value["mask_b"], "mask_b"),
        _sealed_number_array(value["ring_a"], "ring_a"),
        _sealed_number_array(value["ring_b"], "ring_b"),
        _sealed_number_array(value["gradient_magnitude"], "gradient_magnitude"),
        _sealed_number_array(value["pca_components"], "pca_components"),
    )
    if (
        panel.zoom_width <= 0
        or panel.zoom_height <= 0
        or not 0.0 <= panel.predicted_iou <= 1.0
        or not 0.0 <= panel.stability_score <= 1.0
        or not 0.0 <= panel.low_clip_fraction <= 1.0
        or not 0.0 <= panel.high_clip_fraction <= 1.0
        or not 0.0 <= panel.area_norm <= 1.0
        or not 0.0 <= panel.extent <= 1.0
        or not 0.0 <= panel.solidity <= 1.0
        or len(panel.pca_components) > 4
    ):
        raise ContractError("candidate panel intermediate values violate the sealed schema")
    return panel


def _pca_point_from_json(value: object) -> PrecomputedPcaPoint:
    if not isinstance(value, dict) or set(value) != {
        "candidate_id",
        "pc1",
        "pc2",
        "selected",
    }:
        raise ContractError("candidate PCA point fields changed")
    if type(value["selected"]) is not bool:
        raise ContractError("candidate PCA selected flag must be boolean")
    return PrecomputedPcaPoint(
        _sealed_integer(value["candidate_id"], "candidate_id"),
        _sealed_number(value["pc1"], "pc1"),
        _sealed_number(value["pc2"], "pc2"),
        value["selected"],
    )


# SOURCE_CELL: NB-LIVE-0012-C0003
# SOURCE_STATEMENT_MAP: exact-sealed-model-score-column-contract -> _model_score_column_from_json
def _model_score_column_from_json(value: object) -> PrecomputedModelScoreColumn:
    if not isinstance(value, dict) or set(value) != {"model_rel", "column"}:
        raise ContractError("precomputed model score column fields changed")
    model_rel = value["model_rel"]
    column = value["column"]
    if (
        not isinstance(model_rel, str)
        or not model_rel.strip()
        or not isinstance(column, str)
        or not column.strip()
        or any(character in column for character in "\r\n,/")
    ):
        raise ContractError("precomputed model score column values are invalid")
    return PrecomputedModelScoreColumn(model_rel.strip(), column.strip())


# SOURCE_CELL: NB-LIVE-0012-C0011
# SOURCE_CELL: NB-LIVE-0012-C0012
# SOURCE_STATEMENT_MAP: exact-round-pr-auc-threshold-metric-row -> _model_metric_row_from_json
def _model_metric_row_from_json(value: object) -> PrecomputedModelMetricRow:
    if not isinstance(value, dict) or set(value) != {
        "model_rel",
        "round",
        "pr_auc",
        "fixed_thr",
        "thr_best",
    }:
        raise ContractError("precomputed model metric row fields changed")
    model_rel = value["model_rel"]
    round_value = value["round"]
    if (
        type(model_rel) is not str
        or not model_rel.strip()
        or (round_value is not None and type(round_value) is not int)
    ):
        raise ContractError("precomputed model metric identity is invalid")

    def optional_probability(item: object, field: str) -> float | None:
        if item is None:
            return None
        parsed = _sealed_number(item, field)
        if not 0.0 <= parsed <= 1.0:
            raise ContractError(f"precomputed model metric {field} is invalid")
        return parsed

    pr_auc = optional_probability(value["pr_auc"], "pr_auc")
    if pr_auc is None:
        raise ContractError("precomputed model metric pr_auc is required")
    return PrecomputedModelMetricRow(
        model_rel.strip(),
        round_value,
        pr_auc,
        optional_probability(value["fixed_thr"], "fixed_thr"),
        optional_probability(value["thr_best"], "thr_best"),
    )


# SOURCE_CELL: NB-LIVE-0012-C0015
# SOURCE_STATEMENT_MAP: exact-round-model-yolo-group-columns -> _hybrid_score_columns_from_json
def _hybrid_score_columns_from_json(value: object) -> PrecomputedHybridScoreColumns:
    fields = {
        "target_round",
        "model_rel",
        "xgb_probability",
        "yolo_confidence",
        "yolo_iou",
        "group_precedence",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ContractError("precomputed hybrid score contract fields changed")
    if type(value["target_round"]) is not int or value["target_round"] != 92:
        raise ContractError("precomputed hybrid target round changed")
    model_rel = value["model_rel"]
    group_precedence = value["group_precedence"]
    column_values = (
        value["xgb_probability"],
        value["yolo_confidence"],
        value["yolo_iou"],
    )
    if (
        not isinstance(model_rel, str)
        or not model_rel.strip()
        or model_rel.startswith(("/", "\\"))
        or ".." in model_rel.replace("\\", "/").split("/")
        or not isinstance(group_precedence, list)
        or group_precedence != ["root", "img_folder"]
        or any(
            not isinstance(column, str)
            or not column.strip()
            or any(character in column for character in "\r\n,/")
            for column in column_values
        )
        or len(set(column_values)) != len(column_values)
    ):
        raise ContractError("precomputed hybrid score contract values are invalid")
    return PrecomputedHybridScoreColumns(
        92,
        model_rel.strip(),
        *(str(column).strip() for column in column_values),
        ("root", "img_folder"),
    )


# SOURCE_CELL: NB-LIVE-0012-C0001
# SOURCE_STATEMENT_MAP: sealed-production-argv-env-assets-outputs -> _batch_production_from_json
def _batch_production_from_json(
    value: object,
    *,
    mode: str,
) -> PrecomputedBatchProductionContract:
    fields = {
        "schema",
        "folders",
        "tiles_subdirectory",
        "environment",
        "asset_sha256",
        "pipeline_entrypoint",
        "options",
        "outputs",
        "exit_policy",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ContractError("precomputed batch production fields changed")
    if value["schema"] != "compag-curation-report-batch-production/v1":
        raise ContractError("precomputed batch production schema changed")
    if value["tiles_subdirectory"] != "tiles":
        raise ContractError("precomputed batch production tiles subdirectory changed")

    folder_values = value["folders"]
    if (
        not isinstance(folder_values, list)
        or not folder_values
        or any(
            type(folder) is not str
            or not folder
            or folder in {".", ".."}
            or "/" in folder
            or "\\" in folder
            or any(character in folder for character in "\r\n")
            for folder in folder_values
        )
        or len(set(folder_values)) != len(folder_values)
    ):
        raise ContractError("precomputed batch production folders are invalid")

    environment_value = value["environment"]
    expected_environment = (
        ("CJ_SAVE_ALL", "1"),
        ("PYTHONUNBUFFERED", "1"),
    )
    if (
        not isinstance(environment_value, list)
        or any(
            not isinstance(item, dict) or set(item) != {"name", "value"}
            for item in environment_value
        )
        or tuple(
            (item["name"], item["value"])
            for item in environment_value
        )
        != expected_environment
    ):
        raise ContractError("precomputed batch production environment changed")

    asset_value = value["asset_sha256"]
    asset_names = (
        "test_root",
        "sam2_repository",
        "sam2_checkpoint",
        "unified_prototype",
        "padded_prototype",
        "pca",
        "xgb_model",
        "yolo_weights",
    )
    if (
        not isinstance(asset_value, list)
        or len(asset_value) != len(asset_names)
        or any(
            not isinstance(item, dict) or set(item) != {"name", "sha256"}
            for item in asset_value
        )
    ):
        raise ContractError("precomputed batch production asset fields changed")
    asset_sha256 = tuple(
        (item["name"], item["sha256"])
        for item in asset_value
    )
    if tuple(name for name, _ in asset_sha256) != asset_names or any(
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for _, digest in asset_sha256
    ):
        raise ContractError("precomputed batch production asset SHA-256 changed")

    options = value["options"]
    expected_options: tuple[tuple[str, object], ...] = (
        ("mode", mode),
        ("sam2_config", "configs/sam2.1/sam2.1_hiera_l"),
        ("sam2_policy", "both"),
        ("feature_mode", "ultra"),
        ("multiscale", [0.67, 0.80, 1.00, 1.25]),
        ("points_per_side", 64),
        ("points_per_batch", 512),
        ("crop_n_layers", 0),
        ("crop_n_points_downscale_factor", 2),
        ("crop_overlap_ratio", 0.4),
        ("predicted_iou_threshold", 0.80),
        ("stability_threshold", 0.88),
        ("exclude_largest", False),
        ("minimum_true_gates", 3),
        ("export_features", True),
        ("xgb_features", "auto"),
        ("xgb_policy", "replace"),
        ("xgb_threshold", 0.5),
        ("yolo_enabled", True),
        ("yolo_confidence", 0.20),
        ("yolo_iou", 0.60),
        ("yolo_device", "0"),
        ("yolo_image_size", 512),
        ("yolo_maximum_detections", 300),
        ("yolo_hint", "off"),
        ("detector_policy", "hybrid"),
        ("detector_missing", "reject"),
        ("detector_threshold", 0.5),
        ("hybrid_yolo_bias", 0.8),
    )
    if not isinstance(options, dict) or set(options) != {
        name for name, _ in expected_options
    }:
        raise ContractError("precomputed batch production option fields changed")
    if any(
        type(options[name]) is not type(expected)
        or options[name] != expected
        for name, expected in expected_options
    ):
        raise ContractError("precomputed batch production option values changed")

    outputs = value["outputs"]
    expected_outputs: tuple[tuple[str, object], ...] = (
        ("per_folder_run_template", "run_{mode}"),
        ("log_template", "run_{mode}/logs/run_{mode}.log"),
        ("features_filename", "features_pool.csv"),
        ("detections_filename", "detections.csv"),
        ("merged_features_template", "test_features_pool__{mode}.csv"),
        ("merged_detections_template", "test_detections__{mode}.csv"),
    )
    if (
        not isinstance(outputs, dict)
        or set(outputs) != {name for name, _ in expected_outputs}
        or any(
            type(outputs[name]) is not type(expected)
            or outputs[name] != expected
            for name, expected in expected_outputs
        )
    ):
        raise ContractError("precomputed batch production output contract changed")

    exit_policy = value["exit_policy"]
    expected_exit_policy: tuple[tuple[str, object], ...] = (
        ("shell_strict", True),
        ("accepted_process_codes", [0]),
        ("invalid_mode_code", 2),
        ("missing_output_code", 3),
        ("aggregate_folder_failures", True),
        ("merge_requires_both_tables", True),
    )
    if (
        not isinstance(exit_policy, dict)
        or set(exit_policy) != {name for name, _ in expected_exit_policy}
        or any(
            type(exit_policy[name]) is not type(expected)
            or exit_policy[name] != expected
            for name, expected in expected_exit_policy
        )
    ):
        raise ContractError("precomputed batch production exit policy changed")

    pipeline_entrypoint = value["pipeline_entrypoint"]
    if pipeline_entrypoint != "sam2_pipeline.pipeline:run_pipeline":
        raise ContractError("precomputed batch production entrypoint changed")
    return PrecomputedBatchProductionContract(
        mode=mode,
        folders=tuple(folder_values),
        tiles_subdirectory=value["tiles_subdirectory"],
        environment=expected_environment,
        asset_sha256=asset_sha256,
        pipeline_entrypoint=pipeline_entrypoint,
        sam2_config=options["sam2_config"],
        sam2_policy=options["sam2_policy"],
        feature_mode=options["feature_mode"],
        multiscale=tuple(options["multiscale"]),
        points_per_side=options["points_per_side"],
        points_per_batch=options["points_per_batch"],
        crop_n_layers=options["crop_n_layers"],
        crop_n_points_downscale_factor=options[
            "crop_n_points_downscale_factor"
        ],
        crop_overlap_ratio=options["crop_overlap_ratio"],
        predicted_iou_threshold=options["predicted_iou_threshold"],
        stability_threshold=options["stability_threshold"],
        exclude_largest=options["exclude_largest"],
        minimum_true_gates=options["minimum_true_gates"],
        export_features=options["export_features"],
        xgb_features=options["xgb_features"],
        xgb_policy=options["xgb_policy"],
        xgb_threshold=options["xgb_threshold"],
        yolo_enabled=options["yolo_enabled"],
        yolo_confidence=options["yolo_confidence"],
        yolo_iou=options["yolo_iou"],
        yolo_device=options["yolo_device"],
        yolo_image_size=options["yolo_image_size"],
        yolo_maximum_detections=options["yolo_maximum_detections"],
        yolo_hint=options["yolo_hint"],
        detector_policy=options["detector_policy"],
        detector_missing=options["detector_missing"],
        detector_threshold=options["detector_threshold"],
        hybrid_yolo_bias=options["hybrid_yolo_bias"],
        per_folder_run_template=outputs["per_folder_run_template"],
        log_template=outputs["log_template"],
        features_filename=outputs["features_filename"],
        detections_filename=outputs["detections_filename"],
        merged_features_template=outputs["merged_features_template"],
        merged_detections_template=outputs["merged_detections_template"],
        shell_strict=exit_policy["shell_strict"],
        accepted_process_codes=tuple(exit_policy["accepted_process_codes"]),
        invalid_mode_code=exit_policy["invalid_mode_code"],
        missing_output_code=exit_policy["missing_output_code"],
        aggregate_folder_failures=exit_policy["aggregate_folder_failures"],
        merge_requires_both_tables=exit_policy["merge_requires_both_tables"],
    )


# SOURCE_CELL: NB-LIVE-0012-C0001
# SOURCE_STATEMENT_MAP: sealed-batch-inference-output-contract -> _validate_precomputed_testset_batch
def _validate_precomputed_testset_batch(
    source: Path,
    io: ReportingTableIO,
    *,
    decision_threshold: float = 0.5,
    active_learning_seed: int = 0,
) -> PrecomputedTestsetBatch:
    """Validate and load the sealed batch result without running a process."""

    if source.is_symlink() or not source.is_dir():
        raise ContractError("plots source_tables must be a regular directory")
    source_absolute = source.resolve(strict=True)
    source_identity = directory_identity(source_absolute)
    contract_manifest_file = _read_sealed_reporting_file(
        source_absolute,
        source_absolute / "report_precomputed_manifest.json",
        "precomputed manifest",
    )
    contract_manifest = contract_manifest_file.path
    try:
        manifest = loads(contract_manifest_file.data.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ContractError("plots precomputed manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema",
        "mode",
        "features",
        "detections",
        "intermediates",
        "batch_production",
    }:
        raise ContractError("plots precomputed manifest fields changed")
    if manifest["schema"] != "compag-curation-report-precomputed/v3":
        raise ContractError("plots precomputed manifest schema changed")
    mode_value = manifest["mode"]
    if type(mode_value) is not str or not mode_value.strip():
        raise ContractError("plots precomputed manifest mode must be nonempty")
    mode = mode_value.strip()
    production = _batch_production_from_json(
        manifest["batch_production"],
        mode=mode,
    )
    if production.tiles_subdirectory != "tiles":
        raise ContractError("precomputed batch production tiles subdirectory changed")

    def manifest_artifact(field: str, expected_name: str) -> _SealedReportingFile:
        value = manifest[field]
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise ContractError(f"plots manifest {field} artifact fields changed")
        path_value = value["path"]
        expected = value["sha256"]
        if (
            type(path_value) is not str
            or type(expected) is not str
            or not path_value
        ):
            raise ContractError(f"plots manifest {field} artifact types changed")
        relative = Path(path_value)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name != expected_name
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ContractError(f"plots manifest {field} artifact contract is invalid")
        return _read_sealed_reporting_file(
            source_absolute,
            source_absolute / relative,
            f"precomputed {field}",
            expected_sha256=expected,
        )

    feature_prefix = "test_features_pool__"
    detection_prefix = "test_detections__"
    feature_file = manifest_artifact("features", f"{feature_prefix}{mode}.csv")
    detection_file = manifest_artifact("detections", f"{detection_prefix}{mode}.csv")
    intermediate_file = manifest_artifact(
        "intermediates",
        "report_precomputed_intermediates.json",
    )
    features = feature_file.path
    detections = detection_file.path
    intermediates = intermediate_file.path
    feature_mode = features.name[len(feature_prefix) : -len(".csv")]
    detection_mode = detections.name[len(detection_prefix) : -len(".csv")]
    if not feature_mode or feature_mode != detection_mode or feature_mode != mode:
        raise ContractError("precomputed batch feature/detection modes do not match")
    _feature_fields, feature_rows = _decode_sealed_reporting_csv(
        feature_file,
        "precomputed features",
    )
    _detection_fields, detection_rows = _decode_sealed_reporting_csv(
        detection_file,
        "precomputed detections",
    )
    if not feature_rows or not detection_rows:
        raise ContractError("precomputed batch tables must contain data rows")

    required_feature_columns = {
        "img_folder",
        "image",
        "id",
        "human_label",
        "action",
        "review_weight",
        "timestamp",
        "prob",
        "embed_sim",
        "root",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "touching_border",
        "xf_embed_pca_0",
        "xf_embed_pca_1",
    }
    required_detection_columns = {
        "img_folder",
        "id",
        "segmentation",
        "area",
        "predicted_iou",
        "stability_score",
        "source_scale",
    }
    if any(not required_feature_columns.issubset(row) for row in feature_rows):
        raise ContractError("precomputed feature table schema changed")
    if any(not required_detection_columns.issubset(row) for row in detection_rows):
        raise ContractError("precomputed detection table schema changed")
    allowed_scales = {0.67, 0.80, 1.00, 1.25}
    for row in detection_rows:
        try:
            segmentation = loads(str(row["segmentation"]))
            area = int(str(row["area"]))
            predicted_iou = float(str(row["predicted_iou"]))
            stability = float(str(row["stability_score"]))
            source_scale = float(str(row["source_scale"]))
        except (TypeError, ValueError) as exc:
            raise ContractError(
                "precomputed detection values must have exact numeric/mask types"
            ) from exc
        if (
            not isinstance(segmentation, list)
            or not segmentation
            or any(not isinstance(line, list) or not line for line in segmentation)
            or len({len(line) for line in segmentation}) != 1
            or any(type(bit) is not bool for line in segmentation for bit in line)
            or area <= 0
            or not 0.0 <= predicted_iou <= 1.0
            or not 0.0 <= stability <= 1.0
            or source_scale not in allowed_scales
        ):
            raise ContractError("precomputed detection values violate the sealed schema")

    request = PrecomputedReportAnalysisRequest(
        tuple(feature_rows),
        tuple(detection_rows),
        decision_threshold,
        active_learning_seed,
        mode=mode,
    )
    operations = build_report_program(request)
    try:
        intermediate_value = loads(intermediate_file.data.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ContractError("report precomputed intermediates are not valid JSON") from exc
    if not isinstance(intermediate_value, dict) or set(intermediate_value) != {
        "schema",
        "mode",
        "operations",
        "model_score_columns",
        "model_metric_rows",
        "hybrid_score_columns",
        "candidate_panels",
        "pca_scatter",
    }:
        raise ContractError("report precomputed intermediate fields changed")
    if (
        intermediate_value["schema"] != "compag-curation-report-intermediates/v2"
        or intermediate_value["mode"] != mode
        or intermediate_value["operations"]
        != [_operation_contract_json(item) for item in operations]
    ):
        raise ContractError(
            "report precomputed operation order, fields, values, or types changed"
        )
    candidate_values = intermediate_value["candidate_panels"]
    pca_values = intermediate_value["pca_scatter"]
    model_score_values = intermediate_value["model_score_columns"]
    model_metric_values = intermediate_value["model_metric_rows"]
    hybrid_score_value = intermediate_value["hybrid_score_columns"]
    if (
        not isinstance(candidate_values, list)
        or not isinstance(pca_values, list)
        or not isinstance(model_score_values, list)
        or not isinstance(model_metric_values, list)
    ):
        raise ContractError("report candidate intermediates must be JSON arrays")
    candidate_panels = tuple(
        _candidate_panel_from_json(item) for item in candidate_values
    )
    pca_scatter = tuple(_pca_point_from_json(item) for item in pca_values)
    model_score_columns = tuple(
        _model_score_column_from_json(item) for item in model_score_values
    )
    model_metric_rows = tuple(
        _model_metric_row_from_json(item) for item in model_metric_values
    )
    hybrid_score_columns = _hybrid_score_columns_from_json(hybrid_score_value)
    if (
        len(model_score_columns) < 2
        or len({item.model_rel for item in model_score_columns})
        != len(model_score_columns)
        or len({item.column for item in model_score_columns})
        != len(model_score_columns)
        or len(model_metric_rows) != len(model_score_columns)
        or len({item.model_rel for item in model_metric_rows})
        != len(model_metric_rows)
        or {item.model_rel for item in model_metric_rows}
        != {item.model_rel for item in model_score_columns}
    ):
        raise ContractError("precomputed model score columns must name two unique models")
    if not any(
        item.model_rel == hybrid_score_columns.model_rel
        and item.column == hybrid_score_columns.xgb_probability
        for item in model_score_columns
    ):
        raise ContractError(
            "precomputed hybrid model/column must select one declared model score"
        )
    for row in feature_rows:
        for item in model_score_columns:
            if item.column not in row or _float(row[item.column]) is None:
                raise ContractError(
                    "precomputed model score columns must contain finite feature scores"
                )
        hybrid_values = (
            _float(row.get(hybrid_score_columns.xgb_probability)),
            _float(row.get(hybrid_score_columns.yolo_confidence)),
            _float(row.get(hybrid_score_columns.yolo_iou)),
        )
        if any(value is None or not 0.0 <= value <= 1.0 for value in hybrid_values):
            raise ContractError(
                "precomputed hybrid score columns must contain probabilities in [0, 1]"
            )

    def folders(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
        if any("img_folder" not in row for row in rows):
            raise ContractError("precomputed batch rows require img_folder")
        values = tuple(str(row["img_folder"]).strip() for row in rows)
        if any(not value for value in values):
            raise ContractError("precomputed batch folder identifiers must be nonempty")
        return tuple(sorted(set(values)))

    feature_folders = folders(feature_rows)
    detection_folders = folders(detection_rows)
    if feature_folders != detection_folders:
        raise ContractError("precomputed batch tables cover different folder sets")
    if any(folder not in production.folders for folder in feature_folders):
        raise ContractError("precomputed batch tables contain an undeclared folder")
    if tuple(sorted({item.folder for item in candidate_panels})) != feature_folders:
        raise ContractError("candidate panels cover a different folder set")
    detection_ids = {int(str(row["id"])) for row in detection_rows}
    if any(item.candidate_id not in detection_ids for item in candidate_panels):
        raise ContractError("candidate panels reference undeclared detections")
    if directory_identity(source_absolute) != source_identity:
        raise ContractError("plots source_tables identity changed")
    for sealed, role in (
        (contract_manifest_file, "precomputed manifest"),
        (feature_file, "precomputed features"),
        (detection_file, "precomputed detections"),
        (intermediate_file, "precomputed intermediates"),
    ):
        _require_reporting_file_identity(sealed, role)
    return PrecomputedTestsetBatch(
        feature_mode,
        contract_manifest,
        contract_manifest_file.sha256,
        intermediates,
        intermediate_file.sha256,
        features,
        detections,
        operations,
        candidate_panels,
        pca_scatter,
        model_score_columns,
        model_metric_rows,
        hybrid_score_columns,
        feature_folders,
        len(feature_rows),
        len(detection_rows),
        production,
        source_identity,
        contract_manifest_file.identity,
        intermediate_file.identity,
        feature_file.identity,
        detection_file.identity,
        feature_rows,
        detection_rows,
    )


def _require_precomputed_testset_batch_current(
    batch: PrecomputedTestsetBatch,
) -> None:
    root = batch.contract_manifest.parent
    if directory_identity(root) != batch.source_identity:
        raise ContractError("plots source_tables identity changed")
    for path, expected, role in (
        (batch.contract_manifest, batch.contract_manifest_identity, "precomputed manifest"),
        (
            batch.intermediates_manifest,
            batch.intermediates_manifest_identity,
            "precomputed intermediates",
        ),
        (batch.features, batch.features_identity, "precomputed features"),
        (batch.detections, batch.detections_identity, "precomputed detections"),
    ):
        try:
            current = FileIdentity.from_stat(path.lstat())
        except (OSError, ContractError) as exc:
            raise ContractError(f"reporting {role} identity changed") from exc
        if path.is_symlink() or current != expected:
            raise ContractError(f"reporting {role} identity changed")


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: typed-mask-publication-svg-artifacts -> _render_mask_publication_artifacts
def _render_mask_publication_artifacts(
    analysis: PrecomputedReportAnalysisResult,
    output: Path,
) -> tuple[Path, ...]:
    publications = (
        analysis.c0016_mask_publication,
        analysis.c0017_mask_publication,
    )
    if tuple(item.source_variant for item in publications) != ("c0016", "c0017"):
        raise ContractError("mask publication renderer received invalid variants")
    paths: list[Path] = []
    for publication in publications:
        if publication.all_masks_svg is not None:
            all_masks_path = output / f"{publication.source_variant}__all_masks.svg"
            all_masks_path.write_text(
                publication.all_masks_svg,
                encoding="utf-8",
                newline="\n",
            )
            paths.append(all_masks_path)
        for candidate in publication.candidates:
            if candidate.individual_svg is None:
                continue
            path = output / (
                f"{publication.source_variant}__mask_"
                f"{candidate.mask.candidate_id:06d}.svg"
            )
            path.write_text(
                candidate.individual_svg,
                encoding="utf-8",
                newline="\n",
            )
            paths.append(path)
    return tuple(paths)


# SOURCE_CELL: NB-LIVE-0012-C0007
# SOURCE_STATEMENT_MAP: forty-bin-shortlist-density-render -> _render_active_learning_focus_svg
def _render_active_learning_focus_svg(
    evaluation: ActiveLearningFocusEvaluation,
    path: Path,
) -> Path:
    if evaluation.pool_rows <= 0 or evaluation.histogram_bins <= 0:
        raise ContractError("active-learning focus renderer requires pool rows")
    pixel_width = round(evaluation.figure_inches[0] * evaluation.save_dpi)
    pixel_height = round(evaluation.figure_inches[1] * evaluation.save_dpi)
    view_width, view_height = 736, 496
    left, right, top, bottom = 72.0, 20.0, 24.0, 62.0
    plot_width = view_width - left - right
    plot_height = view_height - top - bottom
    bins = evaluation.histogram_bins
    bin_width = 1.0 / bins
    all_counts = [0] * bins
    selected_counts = [0] * bins
    for row in evaluation.rows:
        index = min(bins - 1, max(0, int(row.probability * bins)))
        all_counts[index] += 1
        if row.selected:
            selected_counts[index] += 1
    all_density = [count / (evaluation.pool_rows * bin_width) for count in all_counts]
    selected_density = [
        count / (max(1, evaluation.selected_rows) * bin_width)
        for count in selected_counts
    ]
    maximum = max((*all_density, *selected_density, 1.0))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{pixel_width}" '
        f'height="{pixel_height}" viewBox="0 0 {view_width} {view_height}">',
        f'<rect width="{view_width}" height="{view_height}" fill="white"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        'stroke="#333"/>',
    ]
    rendered_bin_width = plot_width / bins
    for index, (all_value, selected_value) in enumerate(
        zip(all_density, selected_density)
    ):
        x = left + index * rendered_bin_width
        all_height = all_value / maximum * plot_height
        selected_height = selected_value / maximum * plot_height
        elements.extend(
            (
                f'<rect x="{x:.3f}" y="{top + plot_height - all_height:.3f}" '
                f'width="{rendered_bin_width:.3f}" height="{all_height:.3f}" '
                'fill="#78909c" fill-opacity="0.55"/>',
                f'<rect x="{x:.3f}" y="{top + plot_height - selected_height:.3f}" '
                f'width="{rendered_bin_width:.3f}" height="{selected_height:.3f}" '
                'fill="#1769aa" fill-opacity="0.55"/>',
            )
        )
    for value, dash, width in (
        (evaluation.threshold, "6 6", 1.2),
        (evaluation.threshold - evaluation.margin, "2 4", 1.1),
        (evaluation.threshold + evaluation.margin, "2 4", 1.1),
    ):
        x = left + min(1.0, max(0.0, value)) * plot_width
        elements.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
            f'y2="{top + plot_height}" stroke="#333" stroke-width="{width:.1f}" '
            f'stroke-dasharray="{dash}" opacity="0.7"/>'
        )
    elements.extend(
        (
            f'<text x="{left + plot_width / 2:.2f}" y="{view_height - 18}" '
            'text-anchor="middle" font-family="sans-serif" font-size="13">'
            'Predicted probability p</text>',
            f'<text x="18" y="{top + plot_height / 2:.2f}" '
            f'transform="rotate(-90 18 {top + plot_height / 2:.2f})" '
            'text-anchor="middle" font-family="sans-serif" font-size="13">Density</text>',
            f'<text x="{left + 12:.2f}" y="{top + 18:.2f}" '
            f'font-family="sans-serif" font-size="11">All pool (N={evaluation.pool_rows})</text>',
            f'<text x="{left + 12:.2f}" y="{top + 36:.2f}" '
            f'font-family="sans-serif" font-size="11">AL shortlist '
            f'(N={evaluation.selected_rows})</text>',
            "</svg>",
        )
    )
    path.write_text("\n".join(elements) + "\n", encoding="utf-8", newline="\n")
    return path


# SOURCE_CELL: NB-LIVE-0012-C0008
# SOURCE_STATEMENT_MAP: mean-standard-deviation-budget-curves -> _render_active_learning_simulation_svg
def _render_active_learning_simulation_svg(
    evaluation: ActiveLearningSimulationEvaluation,
    path: Path,
) -> Path:
    if (
        not evaluation.aggregate
        or evaluation.strategies != ("uncertainty", "random")
        or evaluation.repeats != 10
    ):
        raise ContractError("active-learning simulation renderer contract changed")
    pixel_width = round(evaluation.figure_inches[0] * evaluation.save_dpi)
    pixel_height = round(evaluation.figure_inches[1] * evaluation.save_dpi)
    view_width, view_height = 768.0, 512.0
    left, right, top, bottom = 76.0, 24.0, 30.0, 66.0
    plot_width = view_width - left - right
    plot_height = view_height - top - bottom
    budgets = [point.budget for point in evaluation.aggregate]
    minimum_budget = min(budgets)
    maximum_budget = max(budgets)
    if minimum_budget == maximum_budget:
        maximum_budget += 1

    def coordinate(budget: int, score: float) -> tuple[float, float]:
        return (
            left
            + (budget - minimum_budget)
            / (maximum_budget - minimum_budget)
            * plot_width,
            top + (1.0 - min(1.0, max(0.0, score))) * plot_height,
        )

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{pixel_width}" '
        f'height="{pixel_height}" viewBox="0 0 {view_width:.0f} {view_height:.0f}">',
        f'<rect width="{view_width:.0f}" height="{view_height:.0f}" fill="white"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        'stroke="#333"/>',
    ]
    for strategy, color in (("uncertainty", "#1769aa"), ("random", "#b3261e")):
        points = [
            point for point in evaluation.aggregate if point.strategy == strategy
        ]
        if not points or any(point.repeats != evaluation.repeats for point in points):
            raise ContractError("active-learning aggregate coverage changed")
        mean_points = [coordinate(point.budget, point.mean_ap) for point in points]
        upper = [
            coordinate(
                point.budget,
                point.mean_ap
                + (
                    point.standard_deviation_ap
                    if isfinite(point.standard_deviation_ap)
                    else 0.0
                ),
            )
            for point in points
        ]
        lower = [
            coordinate(
                point.budget,
                point.mean_ap
                - (
                    point.standard_deviation_ap
                    if isfinite(point.standard_deviation_ap)
                    else 0.0
                ),
            )
            for point in reversed(points)
        ]
        band = " ".join(f"{x:.2f},{y:.2f}" for x, y in (*upper, *lower))
        line = " ".join(f"{x:.2f},{y:.2f}" for x, y in mean_points)
        elements.extend(
            (
                f'<polygon points="{band}" fill="{color}" fill-opacity="0.20"/>',
                f'<polyline points="{line}" fill="none" stroke="{color}" '
                'stroke-width="2.4"/>',
                *(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{color}"/>'
                    for x, y in mean_points
                ),
            )
        )
    elements.extend(
        (
            f'<text x="{left + plot_width / 2:.2f}" y="{view_height - 18:.2f}" '
            'text-anchor="middle" font-family="sans-serif" font-size="13">'
            '# labeled (budget)</text>',
            f'<text x="18" y="{top + plot_height / 2:.2f}" '
            f'transform="rotate(-90 18 {top + plot_height / 2:.2f})" '
            'text-anchor="middle" font-family="sans-serif" font-size="13">'
            'PR-AUC on holdout</text>',
            f'<text x="{left + 12:.2f}" y="{top + 18:.2f}" '
            'font-family="sans-serif" font-size="11" fill="#1769aa">'
            'uncertainty</text>',
            f'<text x="{left + 12:.2f}" y="{top + 36:.2f}" '
            'font-family="sans-serif" font-size="11" fill="#b3261e">random</text>',
            "</svg>",
        )
    )
    path.write_text("\n".join(elements) + "\n", encoding="utf-8", newline="\n")
    return path


def _render_learning_svg(points: Sequence[LearningPoint], path: Path) -> Path:
    width, height = 864, 420
    left, right, top, bottom = 70, 24, 28, 58
    x_min = min(point.round for point in points)
    x_max = max(point.round for point in points)
    y_min = min(min(point.pr_auc, point.smoothed_pr_auc) for point in points)
    y_max = max(max(point.pr_auc, point.smoothed_pr_auc) for point in points)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 0.01

    def xy(round_value: int, score: float) -> tuple[float, float]:
        x = left + (round_value - x_min) / (x_max - x_min) * (width - left - right)
        y = top + (y_max - score) / (y_max - y_min) * (height - top - bottom)
        return x, y

    raw_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (xy(p.round, p.pr_auc) for p in points))
    smooth_points = " ".join(
        f"{x:.2f},{y:.2f}" for x, y in (xy(p.round, p.smoothed_pr_auc) for p in points)
    )
    selected = [point for point in points if point.selected]
    selected_svg = "".join(
        f'<circle cx="{xy(point.round, point.pr_auc)[0]:.2f}" '
        f'cy="{xy(point.round, point.pr_auc)[1]:.2f}" r="6" fill="#b3261e"/>'
        for point in selected
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>\n'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>\n'
        f'<polyline points="{raw_points}" fill="none" stroke="#78909c" stroke-width="1.2" opacity="0.65"/>\n'
        f'<polyline points="{smooth_points}" fill="none" stroke="#1769aa" stroke-width="2.6"/>\n'
        f'{selected_svg}\n'
        f'<text x="{width/2:.1f}" y="{height-14}" text-anchor="middle" font-family="sans-serif" font-size="13">Active-learning iteration (round)</text>\n'
        f'<text x="18" y="{height/2:.1f}" transform="rotate(-90 18 {height/2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="13">PR-AUC</text>\n'
        '</svg>\n'
    )
    path.write_text(svg, encoding="utf-8", newline="\n")
    return path


# SOURCE_CELL: NB-LIVE-0012-C0003
# SOURCE_STATEMENT_MAP: best-mean-one-sigma-baseline-highlight-render -> _render_model_curve_band_svg
def _render_model_curve_band_svg(band: ModelCurveBand, path: Path) -> Path:
    if band.kind not in {"precision-recall", "roc"} or len(band.points) != 401:
        raise ContractError("model curve renderer received an invalid sealed curve band")
    width, height = 640, 460
    left, right, top, bottom = 72, 24, 34, 62

    def coordinate(x_value: float, y_value: float) -> tuple[float, float]:
        x = left + min(1.0, max(0.0, x_value)) * (width - left - right)
        y = top + (1.0 - min(1.0, max(0.0, y_value))) * (height - top - bottom)
        return x, y

    best = " ".join(
        f"{x:.2f},{y:.2f}"
        for x, y in (coordinate(point.x, point.best) for point in band.points)
    )
    mean = " ".join(
        f"{x:.2f},{y:.2f}"
        for x, y in (coordinate(point.x, point.mean) for point in band.points)
    )
    upper = [
        coordinate(point.x, min(1.0, point.mean + point.sigma))
        for point in band.points
    ]
    lower = [
        coordinate(point.x, max(0.0, point.mean - point.sigma))
        for point in reversed(band.points)
    ]
    sigma = " ".join(f"{x:.2f},{y:.2f}" for x, y in (*upper, *lower))
    highlight = "".join(
        (
            f'<circle cx="{coordinate(point.x, point.best)[0]:.2f}" '
            f'cy="{coordinate(point.x, point.best)[1]:.2f}" r="1.2" fill="#b3261e"/>'
        )
        for point in band.points
        if point.best > point.mean + point.sigma
    )
    baseline = (
        f'<line x1="{left}" y1="{coordinate(0.0, band.baseline)[1]:.2f}" '
        f'x2="{width-right}" y2="{coordinate(1.0, band.baseline)[1]:.2f}" '
        'stroke="#555" stroke-dasharray="5 5" opacity="0.55"/>'
        if band.kind == "precision-recall"
        else (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{top}" '
            'stroke="#555" stroke-dasharray="5 5" opacity="0.55"/>'
        )
    )
    x_label = "Recall" if band.kind == "precision-recall" else "False Positive Rate"
    y_label = "Precision" if band.kind == "precision-recall" else "True Positive Rate"
    title = "Precision-Recall curve" if band.kind == "precision-recall" else "ROC curve"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>\n'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>\n'
        f'<polygon points="{sigma}" fill="#78909c" fill-opacity="0.25"/>\n'
        f'{baseline}\n'
        f'<polyline points="{mean}" fill="none" stroke="#546e7a" stroke-width="1.2" stroke-dasharray="6 4"/>\n'
        f'{highlight}\n'
        f'<polyline points="{best}" fill="none" stroke="#1769aa" stroke-width="2.2"/>\n'
        f'<text x="{width/2:.1f}" y="22" text-anchor="middle" font-family="sans-serif" font-size="13">{title}</text>\n'
        f'<text x="{width/2:.1f}" y="{height-15}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_label}</text>\n'
        f'<text x="18" y="{height/2:.1f}" transform="rotate(-90 18 {height/2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="12">{y_label}</text>\n'
        f'<text x="{left+8}" y="{top+18}" font-family="sans-serif" font-size="10">best={escape(band.best_model)}</text>\n'
        '</svg>\n'
    )
    path.write_text(svg, encoding="utf-8", newline="\n")
    return path


# SOURCE_CELL: NB-LIVE-0012-C0015
# SOURCE_STATEMENT_MAP: row-normalized-count-percent-metric-render -> _render_hybrid_confusion_svg
def _render_hybrid_confusion_svg(
    evaluation: HybridEvaluation,
    path: Path,
    *,
    width_inches: float = 3.8,
    height_inches: float = 3.2,
    save_dpi: int = 600,
    show_percentages: bool = True,
    show_footer: bool = True,
) -> Path:
    if (
        width_inches <= 0.0
        or height_inches <= 0.0
        or save_dpi <= 0
        or evaluation.level not in {"candidate-level", "image-level"}
    ):
        raise ContractError("confusion renderer layout contract is invalid")
    pixel_width = int(width_inches * save_dpi)
    pixel_height = int(height_inches * save_dpi)
    view_width, view_height = 760, 640
    left, top, cell_width, cell_height = 190, 112, 230, 180
    metrics = evaluation.metrics
    counts = (
        (metrics.true_negative, metrics.false_positive),
        (metrics.false_negative, metrics.true_positive),
    )
    cells: list[str] = []
    for row_index in range(2):
        for column_index in range(2):
            x = left + column_index * cell_width
            y = top + row_index * cell_height
            fraction = evaluation.row_percentages[row_index][column_index]
            opacity = 0.12 + 0.68 * fraction
            annotation = (
                f"{counts[row_index][column_index]} ({100.0 * fraction:.1f}%)"
                if show_percentages
                else str(counts[row_index][column_index])
            )
            cells.append(
                f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" '
                f'fill="#1769aa" fill-opacity="{opacity:.4f}" stroke="#333"/>'
                f'<text x="{x + cell_width / 2:.1f}" y="{y + 78}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="28">{annotation}</text>'
            )
    footer = (
        f"Precision={metrics.precision:.3f}  Recall={metrics.recall:.3f}  "
        f"F1={metrics.f1:.3f}  FPR={metrics.false_positive_rate:.3f}  "
        f"FNR={metrics.false_negative_rate:.3f}"
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{pixel_width}" '
        f'height="{pixel_height}" viewBox="0 0 {view_width} {view_height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f'<text x="{view_width / 2:.1f}" y="42" text-anchor="middle" '
        f'font-family="sans-serif" font-size="24">Confusion matrix (row-norm) | '
        f'{escape(evaluation.variant)} | DET_THR=0.50 | {escape(evaluation.level)}</text>\n'
        f'{"".join(cells)}\n'
        f'<text x="{left + cell_width}" y="{top + 2 * cell_height + 58}" '
        'text-anchor="middle" font-family="sans-serif" font-size="24">Predicted</text>\n'
        f'<text x="54" y="{top + cell_height}" '
        f'transform="rotate(-90 54 {top + cell_height})" text-anchor="middle" '
        'font-family="sans-serif" font-size="24">True</text>\n'
        f'<text x="{left + cell_width / 2:.1f}" y="96" text-anchor="middle" '
        'font-family="sans-serif" font-size="20">0</text>\n'
        f'<text x="{left + 1.5 * cell_width:.1f}" y="96" text-anchor="middle" '
        'font-family="sans-serif" font-size="20">1</text>\n'
        f'<text x="{left - 28}" y="{top + cell_height / 2:.1f}" text-anchor="middle" '
        'font-family="sans-serif" font-size="20">0</text>\n'
        f'<text x="{left - 28}" y="{top + 1.5 * cell_height:.1f}" text-anchor="middle" '
        'font-family="sans-serif" font-size="20">1</text>\n'
        + (
            f'<text x="{view_width / 2:.1f}" y="{view_height - 24}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="15">{footer}</text>\n'
            if show_footer
            else ""
        )
        + '</svg>\n'
    )
    path.write_text(svg, encoding="utf-8", newline="\n")
    return path


# SOURCE_CELL: NB-LIVE-0012-C0022
# SOURCE_CELL: NB-LIVE-0012-C0023
# SOURCE_STATEMENT_MAP: six-by-four-candidate-panel-render -> _render_candidate_panel_svg
def _render_candidate_panel_svg(
    panels: Sequence[CandidatePanelEvaluation],
    path: Path,
) -> Path:
    if len(panels) != 4:
        raise ContractError("candidate renderer requires exactly four panels")
    ncols = len(panels)
    source_dpi = 240
    width = round(3.8 * ncols * 96)
    height = 16 * 96
    column_width = width / ncols
    row_height = height / 6
    row_titles = (
        "(a) mask (zoom)",
        "(b) mask + dilated ring (for contrasts)",
        "(c) L* (illumination diagnostics)",
        "(d) Lab chroma: mask vs ring",
        "(e) Gradient magnitude (texture proxy)",
        "(f) Masked crop (embedding input)",
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" data-source-dpi="{source_dpi}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<text x="50%" y="24" text-anchor="middle" font-family="sans-serif" '
        'font-size="13">Feature extraction overview for 4 selected SAM2 candidates (+ PCA)</text>',
    ]
    for column, evaluation in enumerate(panels):
        panel = evaluation.panel
        x0 = column * column_width
        elements.append(
            f'<text x="{x0 + column_width / 2:.2f}" y="48" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11">cand {column + 1} '
            f'idx={panel.candidate_id}</text>'
        )
        for row, title in enumerate(row_titles):
            y0 = 58 + row * row_height
            elements.append(
                f'<rect x="{x0 + 8:.2f}" y="{y0:.2f}" '
                f'width="{column_width - 16:.2f}" height="{row_height - 10:.2f}" '
                'fill="none" stroke="#777" stroke-width="1.2"/>'
            )
            elements.append(
                f'<text x="{x0 + 16:.2f}" y="{y0 + 18:.2f}" '
                f'font-family="sans-serif" font-size="9">{escape(title)}</text>'
            )
        quality = "PASS" if evaluation.quality_pass else "FAIL"
        light = "PASS" if evaluation.light_pass else "FAIL"
        color = "PASS" if evaluation.color_pass else "FAIL"
        shape = "PASS" if evaluation.shape_pass else "FAIL"
        light_reason = (
            f" ({'+'.join(evaluation.light_failure_reasons)})"
            if evaluation.light_failure_reasons
            else ""
        )
        elements.extend(
            (
                f'<text x="{x0 + 16:.2f}" y="102" font-family="sans-serif" '
                f'font-size="8">SAM2: p_iou={panel.predicted_iou:.2f}  '
                f'stab={panel.stability_score:.2f}  q={quality}</text>',
                f'<rect x="{x0 + 32:.2f}" y="{row_height + 92:.2f}" '
                f'width="{max(1.0, panel.zoom_width * 0.35):.2f}" height="18" '
                'fill="#00a05a" fill-opacity="0.30"/>',
                f'<rect x="{x0 + 44:.2f}" y="{row_height + 98:.2f}" '
                f'width="{max(1.0, panel.ring_area * 0.20):.2f}" height="8" '
                'fill="#f4d03f" fill-opacity="0.20"/>',
                f'<text x="{x0 + 16:.2f}" y="{2 * row_height + 104:.2f}" '
                f'font-family="sans-serif" font-size="8">low_clip={panel.low_clip_fraction:.3f}  '
                f'high_clip={panel.high_clip_fraction:.3f}</text>',
                f'<text x="{x0 + 16:.2f}" y="{2 * row_height + 120:.2f}" '
                f'font-family="sans-serif" font-size="8">stdL={panel.l_standard_deviation:.1f}  '
                f'dL_med={panel.delta_l_median:.1f}  light_gate={light}'
                f'{escape(light_reason)}</text>',
            )
        )
        chroma_y = 3 * row_height + 86
        for a_value, b_value in zip(panel.ring_a, panel.ring_b):
            elements.append(
                f'<circle cx="{x0 + column_width / 2 + a_value:.2f}" '
                f'cy="{chroma_y - b_value:.2f}" r="2" fill="#f4d03f" '
                'fill-opacity="0.35"/>'
            )
        for a_value, b_value in zip(panel.mask_a, panel.mask_b):
            elements.append(
                f'<circle cx="{x0 + column_width / 2 + a_value:.2f}" '
                f'cy="{chroma_y - b_value:.2f}" r="2" fill="#00a05a" '
                'fill-opacity="0.35"/>'
            )
        elements.append(
            f'<text x="{x0 + 16:.2f}" y="{3 * row_height + 118:.2f}" '
            f'font-family="sans-serif" font-size="8">delta_a={evaluation.delta_a:.1f} '
            f'delta_b={evaluation.delta_b:.1f} color={color}</text>'
        )
        elements.append(
            f'<rect x="{x0 + 16:.2f}" y="{4 * row_height + 92:.2f}" '
            f'width="{min(column_width - 32, evaluation.gradient_mean):.2f}" '
            'height="24" fill="#78909c"/>'
        )
        pca_text = "  ".join(
            f"PC{index + 1}={value:.2f}"
            for index, value in enumerate(evaluation.displayed_pca_components)
        )
        elements.extend(
            (
                f'<text x="{x0 + 16:.2f}" y="{5 * row_height + 92:.2f}" '
                f'font-family="sans-serif" font-size="8">area_norm={panel.area_norm:.4f} '
                f'extent={panel.extent:.2f}</text>',
                f'<text x="{x0 + 16:.2f}" y="{5 * row_height + 108:.2f}" '
                f'font-family="sans-serif" font-size="8">elong={panel.elongation:.2f} '
                f'sol={panel.solidity:.2f} aspect={panel.aspect_ratio:.2f} '
                f'circ={panel.circularity:.2f} shape={shape}</text>',
                f'<text x="{x0 + 16:.2f}" y="{5 * row_height + 124:.2f}" '
                f'font-family="sans-serif" font-size="8">touch_border='
                f'{int(panel.touching_border)} {escape(pca_text)}</text>',
            )
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8", newline="\n")
    return path


# SOURCE_CELL: NB-LIVE-0012-C0023
# SOURCE_STATEMENT_MAP: sealed-pca-scatter-render -> _render_precomputed_pca_svg
def _render_precomputed_pca_svg(
    points: Sequence[PrecomputedPcaPoint],
    path: Path,
    *,
    alpha: float = 0.35,
) -> Path | None:
    if not points:
        return None
    width = round(6.5 * 96)
    height = round(5.5 * 96)
    xs = [point.pc1 for point in points]
    ys = [point.pc2 for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max += 1.0
    if y_min == y_max:
        y_max += 1.0

    def coordinate(point: PrecomputedPcaPoint) -> tuple[float, float]:
        x = 54 + (point.pc1 - x_min) / (x_max - x_min) * (width - 82)
        y = 42 + (y_max - point.pc2) / (y_max - y_min) * (height - 90)
        return x, y

    circles = []
    for point in points:
        x, y = coordinate(point)
        circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" '
            f'r="{60 if point.selected else 10}" fill="'
            f'{"#b3261e" if point.selected else "#1769aa"}" '
            f'fill-opacity="{0.95 if point.selected else alpha:.2f}"/>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" data-source-dpi="240">\n'
        f'<rect width="{width}" height="{height}" fill="white"/>\n'
        '<text x="50%" y="24" text-anchor="middle" font-family="sans-serif" '
        'font-size="13">Embedding PCA space (tile candidates)</text>\n'
        + "\n".join(circles)
        + f'\n<text x="{width / 2:.2f}" y="{height - 12}" text-anchor="middle" '
        'font-family="sans-serif" font-size="11">PC1</text>\n'
        f'<text x="14" y="{height / 2:.2f}" font-family="sans-serif" '
        'font-size="11">PC2</text>\n</svg>\n'
    )
    path.write_text(svg, encoding="utf-8", newline="\n")
    return path


# SOURCE_CELL: NB-LIVE-0012-C0028
# SOURCE_STATEMENT_MAP: three-panel-pca-bestpair-sim-prob-render -> _render_similarity_pca_svg
def _render_similarity_pca_svg(
    evaluation: SimilarityPcaEvaluation,
    path: Path,
) -> Path:
    if not evaluation.points or sum(
        point.selected_ordinal is not None for point in evaluation.points
    ) != 4:
        raise ContractError("similarity/PCA renderer requires four selected points")
    canvas_width = round(16.5 * 96)
    canvas_height = round(5.6 * 96)
    gap = 22.0
    top = 64.0
    bottom = 56.0
    left_margin = 52.0
    panel_width = (canvas_width - 2.0 * left_margin - 2.0 * gap) / 3.0
    panel_height = canvas_height - top - bottom
    panels = (
        (
            "PCA (PC0/PC1) | colored by human label; circled=selected",
            evaluation.default_columns[0],
            evaluation.default_columns[1],
            tuple((point.default_x, point.default_y) for point in evaluation.points),
        ),
        (
            f"PCA best pair in first {evaluation.searched_components} PCs",
            evaluation.best_columns[0],
            evaluation.best_columns[1],
            tuple((point.best_x, point.best_y) for point in evaluation.points),
        ),
        (
            "Similarity vs probability | high sim = more CJ-like embedding",
            f"{evaluation.probability_column} (XGB probability)",
            f"{evaluation.similarity_column} (cosine sim to prototype mu)",
            tuple((point.probability, point.similarity) for point in evaluation.points),
        ),
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" '
        f'height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}" '
        'data-source-dpi="220">',
        f'<rect width="{canvas_width}" height="{canvas_height}" fill="white"/>',
        f'<text x="{canvas_width / 2:.2f}" y="28" text-anchor="middle" '
        'font-family="sans-serif" font-size="15">Tile-level embedding PCA &amp; '
        'prototype similarity</text>',
        f'<text x="{canvas_width / 2:.2f}" y="48" text-anchor="middle" '
        f'font-family="sans-serif" font-size="11">labels: neg={evaluation.negative_rows}, '
        f'pos={evaluation.positive_rows}, unlabeled={evaluation.unlabeled_rows}; '
        f'best score={evaluation.best_score:.3f}</text>',
    ]
    for panel_index, (title, x_label, y_label, coordinates) in enumerate(panels):
        x0 = left_margin + panel_index * (panel_width + gap)
        xs = [value[0] for value in coordinates]
        ys = [value[1] for value in coordinates]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if x_min == x_max:
            x_max += 1.0
        if y_min == y_max:
            y_max += 1.0

        def coordinate(x_value: float, y_value: float) -> tuple[float, float]:
            return (
                x0 + (x_value - x_min) / (x_max - x_min) * panel_width,
                top + (y_max - y_value) / (y_max - y_min) * panel_height,
            )

        elements.extend(
            (
                f'<rect x="{x0:.2f}" y="{top:.2f}" width="{panel_width:.2f}" '
                f'height="{panel_height:.2f}" fill="none" stroke="#555"/>',
                f'<text x="{x0 + panel_width / 2:.2f}" y="{top - 12:.2f}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="10">'
                f'{escape(title)}</text>',
                f'<text x="{x0 + panel_width / 2:.2f}" y="{canvas_height - 12:.2f}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="10">'
                f'{escape(x_label)}</text>',
                f'<text x="{x0 - 34:.2f}" y="{top + panel_height / 2:.2f}" '
                f'transform="rotate(-90 {x0 - 34:.2f} {top + panel_height / 2:.2f})" '
                f'text-anchor="middle" font-family="sans-serif" font-size="10">'
                f'{escape(y_label)}</text>',
            )
        )
        if panel_index == 2:
            guide_x, _guide_y = coordinate(evaluation.probability_threshold, y_min)
            elements.append(
                f'<line x1="{guide_x:.2f}" y1="{top:.2f}" x2="{guide_x:.2f}" '
                f'y2="{top + panel_height:.2f}" stroke="#555" '
                'stroke-width="1.2" opacity="0.35"/>'
            )
        for point, (x_value, y_value) in zip(evaluation.points, coordinates):
            x, y = coordinate(x_value, y_value)
            color = (
                "#b3261e"
                if point.human_label == 1
                else "#1769aa"
                if point.human_label == 0
                else "#78909c"
            )
            radius = 5.0 if point.human_label == 1 else 3.5
            opacity = 0.95 if point.human_label == 1 else 0.35
            elements.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
                f'fill="{color}" fill-opacity="{opacity:.2f}"/>'
            )
            if point.selected_ordinal is not None:
                elements.extend(
                    (
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="11" fill="none" '
                        'stroke="black" stroke-width="2.2"/>',
                        f'<text x="{x:.2f}" y="{y + 3.5:.2f}" text-anchor="middle" '
                        f'font-family="sans-serif" font-size="9" font-weight="bold">'
                        f'{point.selected_ordinal}</text>',
                        f'<text x="{x:.2f}" y="{y - 15:.2f}" text-anchor="middle" '
                        f'font-family="sans-serif" font-size="7">id={point.candidate_id} '
                        f'p={point.probability:.2f} sim={point.similarity:.2f}</text>',
                    )
                )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8", newline="\n")
    return path


# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_STATEMENT_MAP: boolean-mask-row-runs -> _mask_svg_path
def _mask_svg_path(
    mask: Sequence[Sequence[bool]],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    row_count = len(mask)
    column_count = len(mask[0])
    commands: list[str] = []
    for row_index, row in enumerate(mask):
        start: int | None = None
        for column_index in range(column_count + 1):
            active = column_index < column_count and bool(row[column_index])
            if active and start is None:
                start = column_index
            elif not active and start is not None:
                left = x + start * width / column_count
                top = y + row_index * height / row_count
                run_width = (column_index - start) * width / column_count
                run_height = height / row_count
                commands.append(
                    f"M{left:.3f},{top:.3f}h{run_width:.3f}v{run_height:.3f}"
                    f"h{-run_width:.3f}z"
                )
                start = None
    return "".join(commands)


def _render_mask_candidate_grid_svg(
    batch: PrecomputedMaskBatch,
    path: Path,
    *,
    order: str,
) -> Path:
    if order == "score":
        candidates = batch.score_grid
        columns = batch.score_grid_columns
        figure_inches = batch.score_figure_inches
        title = "SAM2 candidates ordered by predicted IoU x stability"
    elif order == "predicted_iou":
        candidates = batch.iou_grid
        columns = batch.iou_grid_columns
        figure_inches = batch.iou_figure_inches
        title = "SAM2 candidates ordered by predicted IoU"
    else:
        raise ContractError("mask candidate renderer order is invalid")
    if not candidates or columns <= 0:
        raise ContractError("mask candidate renderer requires candidates")
    canvas_width = round(figure_inches[0] * 96)
    canvas_height = round(figure_inches[1] * 96)
    rows = (len(candidates) + columns - 1) // columns
    gap = 12.0
    top = 48.0
    cell_width = (canvas_width - gap * (columns + 1)) / columns
    cell_height = (canvas_height - top - gap * (rows + 1)) / rows
    mask_height = max(1.0, cell_height - 34.0)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" '
        f'height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}">',
        f'<rect width="{canvas_width}" height="{canvas_height}" fill="white"/>',
        f'<text x="{canvas_width / 2:.2f}" y="28" text-anchor="middle" '
        f'font-family="sans-serif" font-size="16">{escape(title)}</text>',
    ]
    for index, candidate in enumerate(candidates):
        row_index, column_index = divmod(index, columns)
        left = gap + column_index * (cell_width + gap)
        cell_top = top + gap + row_index * (cell_height + gap)
        path_data = _mask_svg_path(
            candidate.segmentation,
            x=left,
            y=cell_top,
            width=cell_width,
            height=mask_height,
        )
        elements.extend(
            (
                f'<rect x="{left:.2f}" y="{cell_top:.2f}" width="{cell_width:.2f}" '
                f'height="{mask_height:.2f}" fill="#f5f5f5" stroke="#444"/>',
                f'<path d="{path_data}" fill="#2e7d32" '
                f'fill-opacity="{batch.overlay_alpha:.3f}"/>',
                f'<text x="{left + 4:.2f}" y="{cell_top + mask_height + 16:.2f}" '
                f'font-family="sans-serif" font-size="10">#{index:03d} '
                f'id={candidate.candidate_id} A={candidate.area} '
                f's={candidate.score:.3f}</text>',
            )
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8", newline="\n")
    return path


# SOURCE_STATEMENT_MAP: top24-six-column-score-grid -> _render_precomputed_mask_score_grid_svg
def _render_precomputed_mask_score_grid_svg(
    batch: PrecomputedMaskBatch,
    path: Path,
) -> Path:
    return _render_mask_candidate_grid_svg(batch, path, order="score")


# SOURCE_STATEMENT_MAP: top6-three-column-iou-grid -> _render_precomputed_mask_iou_grid_svg
def _render_precomputed_mask_iou_grid_svg(
    batch: PrecomputedMaskBatch,
    path: Path,
) -> Path:
    return _render_mask_candidate_grid_svg(batch, path, order="predicted_iou")


# SOURCE_STATEMENT_MAP: seeded-area-filtered-all-mask-overlay -> _render_precomputed_mask_overlay_svg
def _render_precomputed_mask_overlay_svg(
    batch: PrecomputedMaskBatch,
    path: Path,
) -> Path:
    if not batch.visualization_candidates:
        raise ContractError("mask overlay renderer requires visualization candidates")
    canvas_width = round(14.0 * 96)
    canvas_height = round(6.0 * 96)
    panel_width = (canvas_width - 84.0) / 2.0
    panel_height = canvas_height - 92.0
    palette = ("#1565c0", "#c62828", "#2e7d32", "#6a1b9a", "#ef6c00", "#00838f")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" '
        f'height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}">',
        f'<rect width="{canvas_width}" height="{canvas_height}" fill="white"/>',
        '<text x="25%" y="30" text-anchor="middle" font-family="sans-serif" '
        'font-size="15">Sealed mask field</text>',
        '<text x="75%" y="30" text-anchor="middle" font-family="sans-serif" '
        'font-size="15">All masks overlay (viz-filtered)</text>',
    ]
    union = tuple(
        tuple(
            any(
                candidate.segmentation[row][column]
                for candidate in batch.visualization_candidates
            )
            for column in range(batch.width)
        )
        for row in range(batch.height)
    )
    elements.append(
        f'<path d="{_mask_svg_path(union, x=24.0, y=52.0, width=panel_width, height=panel_height)}" '
        'fill="#455a64" fill-opacity="0.75"/>'
    )
    right = 60.0 + panel_width
    for index, candidate in enumerate(batch.visualization_candidates):
        color = palette[(index + batch.overlay_seed) % len(palette)]
        elements.append(
            f'<path d="{_mask_svg_path(candidate.segmentation, x=right, y=52.0, width=panel_width, height=panel_height)}" '
            f'fill="{color}" fill-opacity="0.500"/>'
        )
    elements.extend(
        (
            f'<rect x="24" y="52" width="{panel_width:.2f}" height="{panel_height:.2f}" '
            'fill="none" stroke="#333"/>',
            f'<rect x="{right:.2f}" y="52" width="{panel_width:.2f}" '
            f'height="{panel_height:.2f}" fill="none" stroke="#333"/>',
            "</svg>",
        )
    )
    path.write_text("\n".join(elements) + "\n", encoding="utf-8", newline="\n")
    return path


# SOURCE_CELL: NB-LIVE-0012-C0000
# SOURCE_CELL: NB-LIVE-0012-C0002
# SOURCE_CELL: NB-LIVE-0012-C0003
# SOURCE_CELL: NB-LIVE-0012-C0005
# SOURCE_CELL: NB-LIVE-0012-C0006
# SOURCE_CELL: NB-LIVE-0012-C0007
# SOURCE_CELL: NB-LIVE-0012-C0008
# SOURCE_CELL: NB-LIVE-0012-C0009
# SOURCE_CELL: NB-LIVE-0012-C0010
# SOURCE_CELL: NB-LIVE-0012-C0011
# SOURCE_CELL: NB-LIVE-0012-C0012
# SOURCE_CELL: NB-LIVE-0012-C0014
# SOURCE_CELL: NB-LIVE-0012-C0015
# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_CELL: NB-LIVE-0012-C0018
# SOURCE_CELL: NB-LIVE-0012-C0019
# SOURCE_CELL: NB-LIVE-0012-C0020
# SOURCE_CELL: NB-LIVE-0012-C0021
# SOURCE_CELL: NB-LIVE-0012-C0022
# SOURCE_CELL: NB-LIVE-0012-C0023
# SOURCE_CELL: NB-LIVE-0012-C0024
# SOURCE_CELL: NB-LIVE-0012-C0025
# SOURCE_CELL: NB-LIVE-0012-C0026
# SOURCE_CELL: NB-LIVE-0012-C0027
# SOURCE_CELL: NB-LIVE-0012-C0028
# SOURCE_STATEMENT_MAP: plotting-backend -> _render_precomputed_operation_svgs
def _render_precomputed_operation_svgs(
    analysis: PrecomputedReportAnalysisResult,
    output: Path,
) -> tuple[Path, ...]:
    """Render one deterministic SVG from each executed typed operation result."""

    if output.is_symlink() or not output.is_dir():
        raise ContractError("report renderer output must be a regular directory")
    paths: list[Path] = []
    paths.append(
        _render_active_learning_focus_svg(
            analysis.active_learning_focus,
            output / "fig_where_al_looks__precomputed.svg",
        )
    )
    paths.append(
        _render_active_learning_simulation_svg(
            analysis.active_learning_simulation,
            output / "fig_al_vs_random__precomputed.svg",
        )
    )
    if analysis.precision_recall_band is not None:
        paths.append(
            _render_model_curve_band_svg(
                analysis.precision_recall_band,
                output / "fig_pr_curve__precomputed.svg",
            )
        )
    if analysis.roc_band is not None:
        paths.append(
            _render_model_curve_band_svg(
                analysis.roc_band,
                output / "fig_roc_curve__precomputed.svg",
            )
        )
    for evaluation in analysis.model_confusion_evaluations:
        level = evaluation.level.removesuffix("-level")
        if level not in {"candidate", "image"} or evaluation.variant != "xgb_round_92":
            raise ContractError("model confusion renderer received an invalid result")
        paths.append(
            _render_hybrid_confusion_svg(
                evaluation,
                output
                / f"fig_confusion_matrix__precomputed__r92__{level}.svg",
                width_inches=3.2,
                height_inches=3.0,
                save_dpi=600,
                show_percentages=False,
                show_footer=False,
            )
        )
        paths.append(
            _render_hybrid_confusion_svg(
                evaluation,
                output
                / (
                    "fig_confusion_matrix_norm_counts__precomputed__r92__"
                    f"{level}.svg"
                ),
                width_inches=3.6,
                height_inches=3.1,
                save_dpi=600,
                show_percentages=True,
                show_footer=True,
            )
        )
    for evaluation in analysis.hybrid_evaluations:
        level = evaluation.level.removesuffix("-level")
        if level not in {"candidate", "image"} or evaluation.variant not in {
            "hybrid_basic",
            "hybrid_smart",
        }:
            raise ContractError("hybrid confusion renderer received an invalid result")
        paths.append(
            _render_hybrid_confusion_svg(
                evaluation,
                output
                / (
                    "fig_confusion_matrix__precomputed__r92__"
                    f"{level}__{evaluation.variant}.svg"
                ),
            )
        )
    if analysis.candidate_panels:
        paths.append(
            _render_candidate_panel_svg(
                analysis.candidate_panels,
                output / "feature_extraction_demo__selected4__6x4.svg",
            )
        )
    pca_path = _render_precomputed_pca_svg(
        analysis.pca_scatter,
        output / "embedding_pca_space__tile_candidates.svg",
        alpha=analysis.candidate_panel_settings.pca_scatter_alpha,
    )
    if pca_path is not None:
        paths.append(pca_path)
    paths.append(
        _render_similarity_pca_svg(
            analysis.similarity_pca_evaluation,
            output / "precomputed__pca_bestpair__human__and__sim_vs_prob.svg",
        )
    )
    paths.extend(
        (
            _render_precomputed_mask_overlay_svg(
                analysis.mask_candidate_batch,
                output / "sam2_precomputed__overlay_all_masks.svg",
            ),
            _render_precomputed_mask_score_grid_svg(
                analysis.mask_candidate_batch,
                output / "sam2_precomputed__score_top24.svg",
            ),
            _render_precomputed_mask_iou_grid_svg(
                analysis.mask_candidate_batch,
                output / "sam2_precomputed__iou_top6.svg",
            ),
        )
    )
    paths.extend(_render_mask_publication_artifacts(analysis, output))
    for ordinal, result in enumerate(analysis.operation_results, 1):
        if result.operation not in analysis.executed_operations:
            raise ContractError("report renderer received an unexecuted operation")
        rendered_value = "NA" if result.value is None else str(result.value)
        numeric = _float(result.value)
        bar_width = 0.0 if numeric is None else min(700.0, max(0.0, abs(numeric)))
        name = result.operation.replace("-", "_")
        if not name.replace("_", "").isalnum():
            raise ContractError("report operation name is not path-safe")
        path = output / f"report_operation__{ordinal:02d}__{name}.svg"
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="864" height="180" '
            'viewBox="0 0 864 180">\n'
            '<rect width="864" height="180" fill="white"/>\n'
            f'<text x="32" y="42" font-family="sans-serif" font-size="18">'
            f'{escape(result.operation)}</text>\n'
            f'<text x="32" y="72" font-family="sans-serif" font-size="13">'
            f'{escape(result.output_role)} = {escape(rendered_value)}</text>\n'
            f'<rect x="32" y="102" width="{bar_width:.2f}" height="28" '
            'fill="#1769aa"/>\n'
            '</svg>\n'
        )
        path.write_text(svg, encoding="utf-8", newline="\n")
        paths.append(path)
    return tuple(paths)


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        dumps(value, indent=2, sort_keys=True, allow_nan=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


# SOURCE_CELL: NB-LIVE-0005-C0003
# SOURCE_STATEMENT_MAP: explicit-artifacts-domain-summary-export -> summarize_dataset
def summarize_dataset(
    config: DomainConfig,
    services: ReportingServices,
) -> DatasetReportResult:
    if not isinstance(services, ReportingServices):
        raise ContractError("reporting handler requires ReportingServices")
    services.validate_call_shapes()
    values = config.workflow("report")
    if values["variant"] != "dataset-summary":
        raise ContractError("reporting handler requires dataset-summary variant")
    require_authorized(config.execution)
    validate_artifact_preconditions(plan_reporting(config, "dataset-summary"))
    source = artifact_path(values, "source_tables")
    output = artifact_path(values, "output")
    request = DatasetSummaryRequest(
        tiled_coco=Path(values.get("tiled_coco", source / "tiled_coco.json")),
        original_coco=Path(values.get("original_coco", source / "original_coco.json")),
        tiles_directory=Path(values.get("tiles", source / "tiles")),
        split_directory=Path(values.get("splits", source / "splits")),
        tile_size=int(values.get("tile_size", 512)),
    )
    summary = build_dataset_summary(request)
    create_output_directory(output)
    json_path = _write_json(output / "dataset_summary.json", asdict(summary))
    latex_path = output / "dataset_summary.tex"
    latex_path.write_text(summary.latex + "\n", encoding="utf-8", newline="\n")
    result = DatasetReportResult(summary, output, json_path, latex_path)
    output_identity = directory_identity(output)
    output_digest = directory_tree_digest(output)
    if directory_identity(output) != output_identity:
        raise ContractError("dataset report output changed before completion report")
    try:
        services.common.report(f"dataset summary written: {output}")
    finally:
        if (
            directory_identity(output) != output_identity
            or directory_tree_digest(output) != output_digest
        ):
            raise ContractError("dataset report output changed during completion report")
    return result


# SOURCE_CELL: NB-LIVE-0012-C0000
# SOURCE_CELL: NB-LIVE-0012-C0001
# SOURCE_CELL: NB-LIVE-0012-C0002
# SOURCE_CELL: NB-LIVE-0012-C0003
# SOURCE_CELL: NB-LIVE-0012-C0005
# SOURCE_CELL: NB-LIVE-0012-C0006
# SOURCE_CELL: NB-LIVE-0012-C0007
# SOURCE_CELL: NB-LIVE-0012-C0008
# SOURCE_CELL: NB-LIVE-0012-C0009
# SOURCE_CELL: NB-LIVE-0012-C0010
# SOURCE_CELL: NB-LIVE-0012-C0011
# SOURCE_CELL: NB-LIVE-0012-C0012
# SOURCE_CELL: NB-LIVE-0012-C0014
# SOURCE_CELL: NB-LIVE-0012-C0015
# SOURCE_CELL: NB-LIVE-0012-C0016
# SOURCE_CELL: NB-LIVE-0012-C0017
# SOURCE_CELL: NB-LIVE-0012-C0018
# SOURCE_CELL: NB-LIVE-0012-C0019
# SOURCE_CELL: NB-LIVE-0012-C0020
# SOURCE_CELL: NB-LIVE-0012-C0021
# SOURCE_CELL: NB-LIVE-0012-C0022
# SOURCE_CELL: NB-LIVE-0012-C0023
# SOURCE_CELL: NB-LIVE-0012-C0024
# SOURCE_CELL: NB-LIVE-0012-C0025
# SOURCE_CELL: NB-LIVE-0012-C0026
# SOURCE_CELL: NB-LIVE-0012-C0027
# SOURCE_CELL: NB-LIVE-0012-C0028
# SOURCE_STATEMENT_MAP: sealed-table-analysis-real-svg-and-table-artifacts -> build_plots_report
def build_plots_report(
    config: DomainConfig,
    services: ReportingServices,
) -> PlotsReportResult:
    """Registry handler that materializes report tables, JSON and a real SVG plot."""

    if not isinstance(services, ReportingServices):
        raise ContractError("reporting handler requires ReportingServices")
    services.validate_call_shapes()
    values = config.workflow("report")
    if values["variant"] != "plots":
        raise ContractError("reporting handler requires plots variant")
    require_authorized(config.execution)
    plan = plan_reporting(config, "plots")
    validate_artifact_preconditions(plan)
    source = artifact_path(values, "source_tables")
    output = artifact_path(values, "output")
    io = services.table_io or StdlibReportingTableIO()
    decision_threshold = float(values.get("decision_threshold", 0.5))
    precomputed_batch = _validate_precomputed_testset_batch(
        source,
        io,
        decision_threshold=decision_threshold,
        active_learning_seed=0,
    )
    precomputed_analysis = analyze_precomputed_report(
        PrecomputedReportAnalysisRequest(
            precomputed_batch.feature_table,
            precomputed_batch.detection_table,
            decision_threshold,
            0,
            precomputed_batch.candidate_panels,
            precomputed_batch.pca_scatter,
            precomputed_batch.model_score_columns,
            precomputed_batch.model_metric_rows,
            precomputed_batch.hybrid_score_columns,
            precomputed_batch.mode,
        )
    )
    files = _source_files(source)
    if not files:
        raise ValueError(f"no CSV/JSON source tables found under: {source}")
    manifest_rows: list[dict[str, Any]] = []
    csv_tables: dict[Path, list[dict[str, str]]] = {}
    sealed_sources: list[_SealedReportingFile] = []
    for path in files:
        sealed = _read_sealed_reporting_file(source, path, "source table")
        sealed_sources.append(sealed)
        rows = (
            list(_decode_sealed_reporting_csv(sealed, "source table")[1])
            if path.suffix.lower() == ".csv"
            else []
        )
        if rows:
            csv_tables[path] = rows
        manifest_rows.append(
            {
                "path": str(path.relative_to(source) if source.is_dir() else path.name),
                "sha256": sealed.sha256,
                "bytes": sealed.identity.size,
                "rows": len(rows),
                "columns": len(rows[0]) if rows else 0,
            }
        )
    validate_artifact_preconditions(plan)
    _require_precomputed_testset_batch_current(precomputed_batch)
    for sealed in sealed_sources:
        _require_reporting_file_identity(sealed, "source table")
    create_output_directory(output)
    output_identity = _require_reporting_output_root(output)
    manifest_path = output / "report_source_manifest.csv"
    _write_reporting_rows(
        io,
        output,
        output_identity,
        manifest_path,
        manifest_rows,
    )

    all_rows = [row for rows in csv_tables.values() for row in rows]
    analysis: dict[str, Any] = {
        "source_files": len(files),
        "source_csv_rows": len(all_rows),
        "decision_threshold": float(values.get("decision_threshold", 0.5)),
        "precomputed_batch": {
            "mode": precomputed_batch.mode,
            "contract_manifest": precomputed_batch.contract_manifest.name,
            "contract_manifest_sha256": precomputed_batch.contract_manifest_sha256,
            "intermediates_manifest": precomputed_batch.intermediates_manifest.name,
            "intermediates_manifest_sha256": (
                precomputed_batch.intermediates_manifest_sha256
            ),
            "operation_count": len(precomputed_batch.operations),
            "folders": list(precomputed_batch.folders),
            "feature_rows": precomputed_batch.feature_rows,
            "detection_rows": precomputed_batch.detection_rows,
            "production": asdict(precomputed_batch.production),
        },
        "precomputed_analysis": asdict(precomputed_analysis),
    }
    plots: list[Path] = list(
        _render_precomputed_operation_svgs(precomputed_analysis, output)
    )
    _require_reporting_output_root(output, output_identity)
    tables: list[Path] = []
    reviewed_testset = precomputed_analysis.reviewed_testset
    reviewed_statistics = reviewed_testset.statistics
    reviewed_table_rows: tuple[tuple[str, Sequence[Mapping[str, Any]]], ...] = (
        (
            f"testset_pool__{precomputed_batch.mode}.csv",
            reviewed_testset.pool_rows,
        ),
        (
            f"testset_labeled__{precomputed_batch.mode}.csv",
            reviewed_testset.labeled_rows,
        ),
        (
            f"testset_stats__{precomputed_batch.mode}.csv",
            (
                {
                    "N_candidates_total": reviewed_statistics.candidates_total,
                    "N_reviewed_labeled": reviewed_statistics.reviewed_labeled,
                    "pct_reviewed": reviewed_statistics.percentage_reviewed,
                    "positives": reviewed_statistics.positives,
                    "pos_rate": reviewed_statistics.positive_rate,
                    "N_border": reviewed_statistics.border,
                    "N_tiny": reviewed_statistics.tiny,
                    "N_border_or_tiny": reviewed_statistics.border_or_tiny,
                    "N_reviewed_border_or_tiny": (
                        reviewed_statistics.reviewed_border_or_tiny
                    ),
                    "mode": reviewed_statistics.mode,
                    "folders": ",".join(reviewed_statistics.folders),
                    "has_yolo_cols": int(reviewed_statistics.has_yolo_columns),
                },
            ),
        ),
        (
            f"testset_per_folder__{precomputed_batch.mode}.csv",
            tuple(
                {
                    "img_folder": item.img_folder,
                    "N_candidates": item.candidates,
                    "N_reviewed": item.reviewed,
                    "pct_reviewed": item.percentage_reviewed,
                    "positives": item.positives,
                    "pos_rate": item.positive_rate,
                }
                for item in reviewed_testset.folder_statistics
            ),
        ),
    )
    for name, rows in reviewed_table_rows:
        path = output / name
        _write_reporting_rows(io, output, output_identity, path, rows)
        tables.append(path)
    mask_candidate_table = output / "sam2_precomputed_candidates.csv"
    _write_reporting_rows(
        io,
        output,
        output_identity,
        mask_candidate_table,
        tuple(
            {
                "idx": index,
                "id": item.candidate_id,
                "area": item.area,
                "area_frac": item.area_fraction,
                "predicted_iou": item.predicted_iou,
                "stability_score": item.stability_score,
                "score": item.score,
                "source_scale": item.source_scale,
                "bbox_xyxy": dumps(list(item.bbox_xyxy)),
                "touches_border": int(item.touching_border),
            }
            for index, item in enumerate(
                precomputed_analysis.mask_candidate_batch.candidates
            )
        ),
    )
    tables.append(mask_candidate_table)
    similarity_pca_table = output / "similarity_pca_points.csv"
    _write_reporting_rows(
        io,
        output,
        output_identity,
        similarity_pca_table,
        tuple(
            {
                "id": point.candidate_id,
                "human_label": point.human_label,
                "probability": point.probability,
                "similarity": point.similarity,
                "default_x": point.default_x,
                "default_y": point.default_y,
                "best_x": point.best_x,
                "best_y": point.best_y,
                "selected_ordinal": point.selected_ordinal,
                "selected_iou": point.selected_iou,
            }
            for point in precomputed_analysis.similarity_pca_evaluation.points
        ),
    )
    tables.append(similarity_pca_table)
    focus = precomputed_analysis.active_learning_focus
    focus_stats_table = output / f"where_al_looks_stats__{precomputed_batch.mode}.csv"
    _write_reporting_rows(
        io,
        output,
        output_identity,
        focus_stats_table,
        (
            {
                "N_pool": focus.pool_rows,
                "thr": focus.threshold,
                "margin": focus.margin,
                "N_uncertain": focus.uncertain_rows,
                "pct_uncertain": focus.uncertain_fraction,
                "N_conflict": focus.conflict_rows,
                "pct_conflict": focus.conflict_fraction,
                "N_selected": focus.selected_rows,
                "pct_selected": focus.selected_fraction,
            },
        ),
    )
    tables.append(focus_stats_table)
    focus_rows_table = output / f"where_al_looks_rows__{precomputed_batch.mode}.csv"
    _write_reporting_rows(
        io,
        output,
        output_identity,
        focus_rows_table,
        tuple(
            {
                "image": row.image,
                "id": row.candidate_id,
                "p_model": row.probability,
                "sel_unc": int(row.uncertain),
                "sel_conflict": int(row.conflict),
                "selected": int(row.selected),
            }
            for row in focus.rows
        ),
    )
    tables.append(focus_rows_table)
    active_learning = precomputed_analysis.active_learning_simulation
    active_learning_curve_table = (
        output / f"al_vs_random_curve__{precomputed_batch.mode}.csv"
    )
    _write_reporting_rows(
        io,
        output,
        output_identity,
        active_learning_curve_table,
        tuple(
            {
                "strategy": point.strategy,
                "rep": point.repeat,
                "budget": point.budget,
                "pr_auc": point.pr_auc,
            }
            for point in active_learning.curve
        ),
    )
    tables.append(active_learning_curve_table)
    active_learning_aggregate_table = (
        output / f"al_vs_random_aggregate__{precomputed_batch.mode}.csv"
    )
    _write_reporting_rows(
        io,
        output,
        output_identity,
        active_learning_aggregate_table,
        tuple(
            {
                "strategy": point.strategy,
                "budget": point.budget,
                "mean_ap": point.mean_ap,
                "std_ap": point.standard_deviation_ap,
                "n": point.repeats,
            }
            for point in active_learning.aggregate
        ),
    )
    tables.append(active_learning_aggregate_table)
    metric_rows = [row for row in all_rows if _float(row.get("pr_auc")) is not None]
    if metric_rows:
        selected_run = str(values.get("selected_run", ""))
        if not selected_run:
            selected_run = str(metric_rows[-1].get("run", metric_rows[-1].get("model_rel", ""))).split("/", 1)[0]
        curve = build_learning_curve(
            metric_rows,
            selected_run=selected_run,
            start_round=(int(values["start_round"]) if values.get("start_round") is not None else None),
            smoothing_window=int(values.get("smoothing_window", 11)),
        )
        curve_rows = [asdict(point) for point in curve]
        curve_table = output / "learning_curve.csv"
        _write_reporting_rows(
            io,
            output,
            output_identity,
            curve_table,
            curve_rows,
        )
        tables.append(curve_table)
        plot_path = _render_learning_svg(curve, output / "fig_learning_curve_selected.svg")
        plots.append(plot_path)
        analysis["learning_curve"] = {
            "points": len(curve),
            "selected_run": selected_run,
            "round_min": min(point.round for point in curve),
            "round_max": max(point.round for point in curve),
        }

    label_rows = [
        row
        for row in all_rows
        if _float(row.get("human_label", row.get("label"))) in (0.0, 1.0)
        and _float(row.get("prob", row.get("xgb_prob", row.get("score")))) is not None
    ]
    if label_rows:
        labels = [int(float(row.get("human_label", row.get("label", 0)))) for row in label_rows]
        scores = [
            float(row.get("prob", row.get("xgb_prob", row.get("score", 0.0))))
            for row in label_rows
        ]
        threshold = float(values.get("decision_threshold", 0.5))
        analysis["classification_metrics"] = asdict(
            classification_metrics(labels, scores, threshold)
        )
        analysis["confusion_matrix"] = confusion_matrix_report(
            labels, scores, threshold=threshold
        )
        analysis["active_learning_focus_count"] = len(
            where_active_learning_looks(label_rows, threshold=threshold)
        )
        analysis["similarity_probability_points"] = len(
            similarity_probability_points(label_rows, ())
        )
        try:
            analysis["best_pca_pair"] = best_pca_pair(label_rows)
        except ValueError:
            analysis["best_pca_pair"] = None

    stats_row = next(
        (row for row in all_rows if "N_candidates_total" in row),
        None,
    )
    if stats_row is not None and metric_rows:
        paper = build_paper_tables(
            stats_row,
            metric_rows,
            mode=str(values.get("mode", "xgb_recall")),
            top_k=int(values.get("top_k", 10)),
            training_size=(int(values["training_size"]) if values.get("training_size") else None),
            decimals=int(values.get("decimals", 3)),
        )
        for name, rows in paper.items():
            path = output / f"table_{name}.csv"
            _write_reporting_rows(io, output, output_identity, path, rows)
            tables.append(path)
        analysis["paper_tables"] = {name: len(rows) for name, rows in paper.items()}

    analysis_path = _write_json(output / "report_analysis.json", analysis)
    _require_reporting_output_root(output, output_identity)
    result_artifacts = (
        manifest_path,
        analysis_path,
        *plots,
        *tables,
    )
    for path in result_artifacts:
        _require_reporting_output_artifact(output, output_identity, path)
    result = PlotsReportResult(
        output,
        manifest_path,
        analysis_path,
        tuple(plots),
        tuple(tables),
        precomputed_batch,
    )
    output_digest = directory_tree_digest(output)
    _require_reporting_output_root(output, output_identity)
    try:
        services.common.report(f"plots report written: {output}")
    finally:
        _require_reporting_output_root(output, output_identity)
        if directory_tree_digest(output) != output_digest:
            raise ContractError("plots report output changed during completion report")
    for path in result_artifacts:
        _require_reporting_output_artifact(output, output_identity, path)
    return result


def run_reporting(
    config: DomainConfig,
    services: ReportingServices,
) -> (
    DatasetReportResult
    | PlotsReportResult
):
    selected = str(config.workflow("report").get("variant", "plots"))
    if selected == "dataset-summary":
        return summarize_dataset(config, services)
    if selected == "plots":
        return build_plots_report(config, services)
    raise ValueError(f"unknown reporting variant: {selected}")


def plan_reporting(config: DomainConfig, variant: str | None = None) -> WorkflowPlan:
    values = config.workflow("report")
    selected = variant or values.get("variant", "plots")
    if selected not in VARIANTS:
        raise ValueError(f"unknown reporting variant: {selected}")
    required = ("source_tables",)
    roles = {
        "source_tables": "sealed-aggregate-tables",
        "images": "image-directory",
        "model": "sealed-model",
        "sam2_checkpoint": "sealed-model",
        "sam2_config": "sealed-model-config",
        "prototype": "sealed-feature-prototype",
        "pca": "sealed-transform",
    }
    inputs = tuple(artifact(values[name], name, roles[name]) for name in required)
    outputs = (artifact(values["output"], "output", "report-output", output=True),)
    source_cells = (
        ("NB-LIVE-0005-C0003",)
        if selected == "dataset-summary"
        else tuple(
            f"NB-LIVE-0012-C{index:04d}"
            for index in range(29)
            if index not in {4, 13}
        )
    )
    steps = (
        WorkflowStep(
            "report-preflight",
            "validate-report-input-contract",
            tuple(item.name for item in inputs),
            source_cell_ids=source_cells,
        ),
        WorkflowStep(
            "report-render",
            f"render-{selected}",
            tuple(item.name for item in inputs),
            ("output",),
            {"decision_threshold": values.get("decision_threshold", 0.5)},
            source_cells,
        ),
    )
    return make_plan(
        config,
        "report",
        selected,
        "compag_curation.domain.reporting:plan_reporting",
        inputs,
        outputs,
        steps,
        {"decision_threshold": values.get("decision_threshold", 0.5)},
    )


__all__ = [
    "ReportingServices",
    "build_plots_report",
    "plan_reporting",
    "run_reporting",
    "summarize_dataset",
]
