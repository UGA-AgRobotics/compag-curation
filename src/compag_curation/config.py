"""Typed, side-effect-free configuration for the domain API."""

from __future__ import annotations

import json
import math
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    ContractError,
    ExecutionPolicy,
    NotificationConfig,
    _deep_freeze,
    _deep_thaw,
)


SCHEMA = "compag-curation-domain-config/v2"
COMMANDS = (
    "propose",
    "extract-features",
    "review",
    "train",
    "infer",
    "evaluate",
    "transfer",
    "report",
)
ENV_PATTERN = r"\$\{([A-Z][A-Z0-9_]*)\}"
ALLOWED_ENVIRONMENT_NAMES = frozenset({
    "COMPAG_ASSET_ROOT",
    "COMPAG_DATA_ROOT",
    "COMPAG_OUTPUT_ROOT",
    "COMPAG_PYTHON_EXECUTABLE",
    "COMPAG_SOURCE_ROOT",
})
SECRET_ENVIRONMENT_FRAGMENTS = (
    "ACCESS_KEY",
    "API_KEY",
    "AUTH",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "TOKEN",
)


class ConfigurationError(ContractError):
    pass


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be an object")
    return dict(value)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} must be a nonempty string")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def validate_image_identifier(value: Any, label: str = "image identifier") -> str:
    """Validate one portable image selector without interpreting it as a path."""

    identifier = _string(value, label)
    if (
        identifier != identifier.strip()
        or identifier in {".", ".."}
        or any(separator in identifier for separator in ("/", "\\", ","))
        or any(ord(character) < 32 or ord(character) == 127 for character in identifier)
    ):
        raise ConfigurationError(
            f"{label} must be one portable non-path identity"
        )
    return identifier


def _choice(value: Any, label: str, choices: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ConfigurationError(f"{label} is invalid")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{label} must be at least {minimum}")
    return value


def _number(value: Any, label: str, *, minimum: float | None = None) -> float | int:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ConfigurationError(f"{label} must be a finite number")
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{label} must be at least {minimum}")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ConfigurationError(f"{label} must be lowercase hexadecimal SHA-256")
    return value


def _literal_notification_string(value: Any, label: str) -> str:
    rendered = _string(value, label)
    if "$" in rendered:
        raise ConfigurationError(f"{label} must be a literal value, not an environment token")
    return rendered


def interpolate(
    value: Any,
    environment: Mapping[str, str],
    *,
    allowed_names: frozenset[str] | None = None,
) -> Any:
    """Expand only ${UPPER_CASE_NAME} using an explicit environment mapping."""

    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if any(fragment in name for fragment in SECRET_ENVIRONMENT_FRAGMENTS):
                raise ConfigurationError("secret-like environment interpolation is prohibited")
            if allowed_names is not None and name not in allowed_names:
                raise ConfigurationError(f"environment variable is not in the closed allowlist: {name}")
            if name not in environment:
                raise ConfigurationError(f"missing configured environment variable: {name}")
            return _string(environment[name], f"environment.{name}")

        rendered = re.sub(ENV_PATTERN, replace, value)
        if "$" in rendered:
            raise ConfigurationError("unsupported environment interpolation syntax")
        return rendered
    if isinstance(value, list):
        return [interpolate(item, environment, allowed_names=allowed_names) for item in value]
    if isinstance(value, dict):
        return {
            key: interpolate(item, environment, allowed_names=allowed_names)
            for key, item in value.items()
        }
    return value


@dataclass(frozen=True)
class DomainConfig:
    schema: str
    execution: ExecutionPolicy
    notifications: NotificationConfig
    workflows: Mapping[str, Mapping[str, Any]]
    fixture_classification: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflows", _deep_freeze(self.workflows))

    def workflow(self, command: str) -> dict[str, Any]:
        if command not in COMMANDS:
            raise ConfigurationError("unknown workflow command")
        value = self.workflows.get(command)
        if value is None:
            raise ConfigurationError(f"workflow configuration is missing: {command}")
        return _deep_thaw(value)


def _validate_artifact(value: Any, label: str, *, output: bool = False) -> dict[str, Any]:
    item = _object(value, label)
    allowed = {"path", "sha256", "role", "kind", "required"}
    if set(item) - allowed or "path" not in item:
        raise ConfigurationError(f"{label} fields are invalid")
    _string(item["path"], f"{label}.path")
    if "role" in item:
        _string(item["role"], f"{label}.role")
    _choice(item.get("kind", "file"), f"{label}.kind", frozenset({"file", "directory"}))
    if "required" in item and not isinstance(item["required"], bool):
        raise ConfigurationError(f"{label}.required must be boolean")
    if output:
        if item.get("required", False) is not False:
            raise ConfigurationError(f"{label} is a future output and cannot be required")
        if "sha256" in item:
            raise ConfigurationError(f"{label} is a future output and cannot declare an input SHA-256")
    elif item.get("required") is False:
        raise ConfigurationError(f"{label} is a semantic input and cannot disable preflight")
    if item.get("sha256") is not None:
        _sha256(item["sha256"], f"{label}.sha256")
    return item


VARIANT_FIELDS: dict[tuple[str, str], tuple[set[str], set[str]]] = {
    ("propose", "sam2-proposals"): (
        {
            "variant", "images", "output", "sam2_package_root",
            "sam2_checkpoint", "sam2_config", "device",
        },
        {"feature_mode", "sam2_policy", "save_all_outputs"},
    ),
    ("propose", "yolo-dataset-preparation"): (
        {"variant", "coco_annotations", "tiles", "split_manifest", "yolo_dataset_output"},
        set(),
    ),
    ("extract-features", "coco-features"): (
        {
            "variant",
            "image_name",
            "mode",
            "device",
            "original_coco",
            "coco_annotations",
            "detections",
            "review_labels",
            "tiles",
            "sam2_checkpoint",
            "model_root",
            "previous_split_manifest",
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
        },
        {"requested_model_name", "decision_threshold"},
    ),
    ("review", "al-review-ui"): (
        {
            "variant", "image_root", "current_image", "review_mode",
            "detections", "images", "review_output",
        },
        {"host", "port"},
    ),
    ("review", "full-image-review-ui"): (
        {
            "variant", "image_root", "current_image", "review_mode",
            "detections", "images", "review_output",
        },
        {"host", "port"},
    ),
    ("train", "coco-xgb"): (
        {
            "variant",
            "feature_table",
            "split_manifest",
            "previous_tile_predictions",
            "previous_threshold",
            "model_output",
        },
        {
            "previous_best_parameters",
            "seed",
            "decision_threshold",
            "device",
            "thread_count",
        },
    ),
    ("train", "xgb"): (
        {"variant", "feature_table", "split_manifest", "model_output"},
        {"seed", "decision_threshold", "device", "thread_count"},
    ),
    ("train", "xgb-main"): (
        {
            "variant",
            "feature_table",
            "split_manifest",
            "hybrid_previous_tile_predictions",
            "hybrid_previous_threshold",
            "augmented_previous_tile_predictions",
            "augmented_previous_threshold",
            "model_output",
        },
        {"seed", "decision_threshold", "device", "thread_count"},
    ),
    ("train", "yolo"): (
        {"variant", "data_yaml", "dataset_root", "local_initial_weights", "model_output"},
        {
            "imgsz", "epochs", "batch", "patience", "lr0", "weight_decay",
            "close_mosaic", "seed", "deterministic", "device",
        },
    ),
    ("infer", "sam2-xgb"): (
        {
            "variant",
            "images",
            "output",
            "sam2_package_root",
            "sam2_checkpoint",
            "sam2_config",
            "prototype",
            "padded_prototype",
            "pca",
            "xgb_model",
            "embed_backbone",
            "embed_backbone_weights",
            "device",
        },
        {
            "decision_threshold",
            "mode",
            "save_all",
            "sam2_policy",
            "yolo_confidence",
            "yolo_iou",
            "detector_policy",
            "detector_missing",
            "detector_threshold",
            "hybrid_yolo_bias",
        },
    ),
    ("infer", "sam2-xgb-yolo"): (
        {
            "variant",
            "images",
            "output",
            "sam2_package_root",
            "sam2_checkpoint",
            "sam2_config",
            "prototype",
            "padded_prototype",
            "pca",
            "xgb_model",
            "yolo_weights",
            "embed_backbone",
            "embed_backbone_weights",
            "device",
        },
        {
            "decision_threshold",
            "mode",
            "save_all",
            "sam2_policy",
            "yolo_confidence",
            "yolo_iou",
            "detector_policy",
            "detector_missing",
            "detector_threshold",
            "hybrid_yolo_bias",
        },
    ),
    ("evaluate", "coverage-basic"): (
        {"variant", "image_id", "predictions", "annotations", "evaluator_script", "tile_index", "output"},
        {"decision_threshold"},
    ),
    ("evaluate", "coverage-points-overlay"): (
        {"variant", "image_id", "predictions", "annotations", "images", "evaluator_script", "tile_index", "output"},
        {"decision_threshold"},
    ),
    ("evaluate", "coverage-predictions-overlay"): (
        {"variant", "image_id", "predictions", "annotations", "images", "evaluator_script", "tile_index", "output"},
        {"decision_threshold"},
    ),
    ("evaluate", "coverage-reviewability"): (
        {"variant", "image_id", "predictions", "annotations", "images", "review_labels", "evaluator_script", "tile_index", "output"},
        {"decision_threshold"},
    ),
    ("evaluate", "meta-analysis"): (
        {"variant", "review_root", "output"},
        {"decision_threshold"},
    ),
    ("evaluate", "border-tiny"): (
        {"variant", "predictions", "annotations", "output"},
        {"decision_threshold"},
    ),
    ("transfer", "result-tree"): (
        {"variant", "source", "destination"},
        {"expected_sha256", "include_patterns"},
    ),
    ("transfer", "csv-and-text-probe"): (
        {"variant", "source", "destination", "probe_workspace"},
        {"expected_sha256", "include_patterns"},
    ),
    ("report", "dataset-summary"): (
        {"variant", "source_tables", "output"},
        {"decision_threshold"},
    ),
    ("report", "plots"): (
        {"variant", "source_tables", "output"},
        {"decision_threshold"},
    ),
}
VARIANTS_BY_COMMAND = {
    command: tuple(variant for candidate, variant in VARIANT_FIELDS if candidate == command)
    for command in COMMANDS
}


ARTIFACT_FIELDS = {
    "images",
    "image_root",
    "output",
    "sam2_package_root",
    "sam2_checkpoint",
    "model_root",
    "sam2_config",
    "coco_annotations",
    "original_coco",
    "previous_split_manifest",
    "merged_original_coco",
    "merged_tiled_coco",
    "round_new_ids",
    "used_ids",
    "positive_embeddings",
    "all_embeddings",
    "pack",
    "tiles",
    "split_manifest",
    "yolo_dataset_output",
    "prototype",
    "padded_prototype",
    "pca",
    "review_labels",
    "detections",
    "review_output",
    "feature_table",
    "previous_tile_predictions",
    "previous_threshold",
    "previous_best_parameters",
    "hybrid_previous_tile_predictions",
    "hybrid_previous_threshold",
    "augmented_previous_tile_predictions",
    "augmented_previous_threshold",
    "model_output",
    "local_initial_weights",
    "data_yaml",
    "dataset_root",
    "xgb_model",
    "yolo_weights",
    "embed_backbone_weights",
    "predictions",
    "annotations",
    "source",
    "destination",
    "probe_workspace",
    "source_tables",
    "review_root",
    "model",
    "evaluator_script",
    "tile_index",
}

VARIANT_ARTIFACT_KINDS: dict[tuple[str, str], dict[str, str]] = {
    ("propose", "sam2-proposals"): {
        "images": "directory", "output": "directory", "sam2_package_root": "directory",
        "sam2_checkpoint": "file", "sam2_config": "file",
    },
    ("propose", "yolo-dataset-preparation"): {
        "coco_annotations": "file", "tiles": "directory", "split_manifest": "file", "yolo_dataset_output": "directory",
    },
    ("extract-features", "coco-features"): {
        "original_coco": "file", "coco_annotations": "file", "detections": "file",
        "review_labels": "file", "tiles": "directory", "sam2_checkpoint": "file",
        "model_root": "directory",
        "previous_split_manifest": "file",
        "merged_original_coco": "file",
        "merged_tiled_coco": "file", "round_new_ids": "file", "split_manifest": "file",
        "prototype": "file", "pca": "file", "used_ids": "file",
        "positive_embeddings": "file", "all_embeddings": "file", "pack": "file",
        "output": "file",
    },
    ("review", "al-review-ui"): {
        "image_root": "directory", "detections": "file", "images": "directory",
        "review_output": "file",
    },
    ("review", "full-image-review-ui"): {
        "image_root": "directory", "detections": "file", "images": "directory",
        "review_output": "directory",
    },
    ("train", "coco-xgb"): {
        "feature_table": "file", "split_manifest": "file",
        "previous_tile_predictions": "file", "previous_threshold": "file",
        "previous_best_parameters": "file", "model_output": "directory",
    },
    ("train", "xgb"): {
        "feature_table": "file", "split_manifest": "file", "model_output": "directory",
    },
    ("train", "xgb-main"): {
        "feature_table": "file", "split_manifest": "file",
        "hybrid_previous_tile_predictions": "file", "hybrid_previous_threshold": "file",
        "augmented_previous_tile_predictions": "file", "augmented_previous_threshold": "file",
        "model_output": "directory",
    },
    ("train", "yolo"): {
        "data_yaml": "file", "dataset_root": "directory", "local_initial_weights": "file",
        "model_output": "directory",
    },
    ("infer", "sam2-xgb"): {
        "images": "directory", "output": "directory", "sam2_package_root": "directory",
        "sam2_checkpoint": "file", "sam2_config": "file", "prototype": "file",
        "padded_prototype": "file", "pca": "file", "xgb_model": "file",
        "embed_backbone_weights": "file",
    },
    ("infer", "sam2-xgb-yolo"): {
        "images": "directory", "output": "directory", "sam2_package_root": "directory",
        "sam2_checkpoint": "file", "sam2_config": "file", "prototype": "file",
        "padded_prototype": "file", "pca": "file", "xgb_model": "file",
        "yolo_weights": "file", "embed_backbone_weights": "file",
    },
    ("evaluate", "coverage-basic"): {"predictions": "file", "annotations": "file", "evaluator_script": "file", "tile_index": "file", "output": "directory"},
    ("evaluate", "coverage-points-overlay"): {"predictions": "file", "annotations": "file", "images": "directory", "evaluator_script": "file", "tile_index": "file", "output": "directory"},
    ("evaluate", "coverage-predictions-overlay"): {"predictions": "file", "annotations": "file", "images": "directory", "evaluator_script": "file", "tile_index": "file", "output": "directory"},
    ("evaluate", "coverage-reviewability"): {"predictions": "file", "annotations": "file", "images": "directory", "review_labels": "file", "evaluator_script": "file", "tile_index": "file", "output": "directory"},
    ("evaluate", "meta-analysis"): {"review_root": "directory", "output": "directory"},
    ("evaluate", "border-tiny"): {"predictions": "file", "annotations": "file", "output": "directory"},
    ("transfer", "result-tree"): {"source": "directory", "destination": "directory"},
    ("transfer", "csv-and-text-probe"): {
        "source": "directory", "destination": "directory", "probe_workspace": "directory",
    },
    ("report", "dataset-summary"): {"source_tables": "directory", "output": "directory"},
    ("report", "plots"): {"source_tables": "directory", "output": "directory"},
}
VARIANT_OUTPUT_ARTIFACT_FIELDS: dict[tuple[str, str], frozenset[str]] = {
    ("propose", "sam2-proposals"): frozenset({"output"}),
    ("propose", "yolo-dataset-preparation"): frozenset({"yolo_dataset_output"}),
    ("extract-features", "coco-features"): frozenset({
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
    }),
    ("review", "al-review-ui"): frozenset({"review_output"}),
    ("review", "full-image-review-ui"): frozenset({"review_output"}),
    ("train", "coco-xgb"): frozenset({"model_output"}),
    ("train", "xgb"): frozenset({"model_output"}),
    ("train", "xgb-main"): frozenset({"model_output"}),
    ("train", "yolo"): frozenset({"model_output"}),
    ("infer", "sam2-xgb"): frozenset({"output"}),
    ("infer", "sam2-xgb-yolo"): frozenset({"output"}),
    ("evaluate", "coverage-basic"): frozenset({"output"}),
    ("evaluate", "coverage-points-overlay"): frozenset({"output"}),
    ("evaluate", "coverage-predictions-overlay"): frozenset({"output"}),
    ("evaluate", "coverage-reviewability"): frozenset({"output"}),
    ("evaluate", "meta-analysis"): frozenset({"output"}),
    ("evaluate", "border-tiny"): frozenset({"output"}),
    ("transfer", "result-tree"): frozenset({"destination"}),
    ("transfer", "csv-and-text-probe"): frozenset({"destination", "probe_workspace"}),
    ("report", "dataset-summary"): frozenset({"output"}),
    ("report", "plots"): frozenset({"output"}),
}
OUTPUT_ARTIFACT_FIELDS = frozenset().union(*VARIANT_OUTPUT_ARTIFACT_FIELDS.values())

YOLO_DEFAULTS: dict[str, int | float | bool | str] = {
    "imgsz": 512,
    "epochs": 60,
    "batch": 16,
    "patience": 20,
    "lr0": 0.001,
    "weight_decay": 0.0005,
    "close_mosaic": 10,
    "seed": 0,
    "deterministic": True,
    "device": "cpu",
}


def validate_mapping(
    value: Any,
    *,
    fixture: bool = False,
    environment: Mapping[str, str] | None = None,
) -> DomainConfig:
    root = _object(value, "configuration")
    allowed = {"schema", "execution", "notifications", "workflows", "fixture_classification"}
    required_root = {"schema", "execution", "notifications", "workflows"}
    if set(root) - allowed or not required_root <= set(root) or root.get("schema") != SCHEMA:
        raise ConfigurationError("configuration fields or schema are invalid")
    classification = root.get("fixture_classification")
    if classification is not None and classification != "SYNTHETIC_NON_SCIENTIFIC_TEST_FIXTURE":
        raise ConfigurationError("fixture classification is invalid")
    if fixture and classification != "SYNTHETIC_NON_SCIENTIFIC_TEST_FIXTURE":
        raise ConfigurationError("synthetic fixture classification is required")
    is_fixture = classification == "SYNTHETIC_NON_SCIENTIFIC_TEST_FIXTURE"
    env = environment if environment is not None else os.environ
    notification_source = _object(root.get("notifications", {}), "notifications")
    if set(notification_source) - {"enabled", "topic", "endpoint"}:
        raise ConfigurationError("notification fields are invalid")
    notification_enabled = notification_source.get("enabled", False)
    if not isinstance(notification_enabled, bool):
        raise ConfigurationError("notifications.enabled must be boolean")
    if is_fixture and notification_enabled:
        raise ConfigurationError("synthetic fixture notifications must be disabled")
    expandable = dict(root)
    expandable["notifications"] = {}
    expanded = interpolate(expandable, env, allowed_names=ALLOWED_ENVIRONMENT_NAMES)

    execution_raw = _object(expanded.get("execution", {}), "execution")
    if set(execution_raw) - {"authorized", "allow_download", "output_collision", "interpreter"}:
        raise ConfigurationError("execution fields are invalid")
    authorized = execution_raw.get("authorized", False)
    allow_download = execution_raw.get("allow_download", False)
    if not isinstance(authorized, bool) or not isinstance(allow_download, bool):
        raise ConfigurationError("execution booleans are invalid")
    output_collision = execution_raw.get("output_collision", "fail")
    if output_collision != "fail":
        raise ConfigurationError("execution.output_collision must be fail")
    if is_fixture and (authorized or allow_download):
        raise ConfigurationError("synthetic fixture cannot authorize execution or download")
    execution = ExecutionPolicy(
        authorized=authorized,
        allow_download=allow_download,
        output_collision=output_collision,
        interpreter=_optional_string(execution_raw.get("interpreter"), "execution.interpreter"),
    )

    if notification_enabled:
        topic = _literal_notification_string(notification_source.get("topic"), "notifications.topic")
        endpoint = _literal_notification_string(
            notification_source.get("endpoint"), "notifications.endpoint"
        )
    else:
        topic = endpoint = None
    notifications = NotificationConfig(enabled=notification_enabled, topic=topic, endpoint=endpoint)

    workflows_raw = _object(expanded.get("workflows"), "workflows")
    if set(workflows_raw) != set(COMMANDS):
        raise ConfigurationError("configuration must define exactly all public workflow commands")
    workflows: dict[str, Mapping[str, Any]] = {}
    for command in COMMANDS:
        item = _object(workflows_raw[command], f"workflows.{command}")
        variant = _string(item.get("variant"), f"workflows.{command}.variant")
        if variant not in VARIANTS_BY_COMMAND[command]:
            raise ConfigurationError(f"workflows.{command}.variant is invalid")
        required, optional = VARIANT_FIELDS[(command, variant)]
        if not required <= set(item) or set(item) - required - optional:
            raise ConfigurationError(f"workflows.{command} fields are invalid")
        variant_outputs = VARIANT_OUTPUT_ARTIFACT_FIELDS[(command, variant)]
        for name in set(item) & ARTIFACT_FIELDS:
            item[name] = _validate_artifact(
                item[name], f"workflows.{command}.{name}", output=name in variant_outputs
            )
            expected_kind = VARIANT_ARTIFACT_KINDS[(command, variant)][name]
            if item[name].get("kind", "file") != expected_kind:
                raise ConfigurationError(
                    f"workflows.{command}.{name} must be a {expected_kind} artifact for {variant}"
                )
        if not is_fixture:
            for name in (set(item) & ARTIFACT_FIELDS) - variant_outputs:
                if item[name].get("sha256") is None:
                    raise ConfigurationError(f"workflows.{command}.{name} requires an explicit SHA-256")
        if "decision_threshold" in item:
            threshold = item["decision_threshold"]
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold != 0.5:
                raise ConfigurationError("sealed decision_threshold must remain 0.5")
        if command == "train" and variant != "yolo":
            item.setdefault("seed", 42)
            if _integer(item["seed"], "training seed") != 42:
                raise ConfigurationError("sealed XGBoost training seed must remain 42")
            item.setdefault("device", "cpu")
            if item["device"] != "cpu":
                raise ConfigurationError("registered XGBoost training is sealed to CPU")
            item.setdefault("thread_count", 1)
            if _integer(item["thread_count"], "XGBoost thread count", minimum=1) != 1:
                raise ConfigurationError("sealed XGBoost thread count must remain 1")
        if command == "extract-features":
            image_name = _string(item["image_name"], "feature image name")
            if re.fullmatch(r"IMG_[0-9]+", image_name) is None:
                raise ConfigurationError("feature image_name must use IMG_<integer> form")
            _choice(
                item["mode"],
                "feature round mode",
                frozenset({"gate", "xgb", "xgb_recall"}),
            )
            if item["device"] != "cpu":
                raise ConfigurationError("registered feature extraction is sealed to CPU")
            model_name = _string(
                item.get(
                    "requested_model_name",
                    "cj_ultra_tilesafe_xgb_r17_hybrid.pkl",
                ),
                "feature requested model name",
            )
            if Path(model_name).name != model_name or not model_name.endswith(".pkl"):
                raise ConfigurationError(
                    "feature requested_model_name must be one portable .pkl filename"
                )
        if command == "review":
            current_image = _string(item["current_image"], "current review image")
            if re.fullmatch(r"IMG_[0-9]+", current_image) is None:
                raise ConfigurationError("review current_image must use IMG_<integer> form")
            _choice(
                item["review_mode"],
                "review mode",
                frozenset({"gate", "xgb", "xgb_recall"}),
            )
            expected_review_kind = (
                "directory" if variant == "full-image-review-ui" else "file"
            )
            if item["review_output"].get("kind", "file") != expected_review_kind:
                raise ConfigurationError(
                    f"{variant} review_output must be a {expected_review_kind} artifact"
                )
            if "host" in item:
                _choice(
                    item["host"],
                    "review host",
                    frozenset({"127.0.0.1", "localhost"}),
                )
            if "port" in item and (
                _integer(item["port"], "review port", minimum=1) > 65535
            ):
                raise ConfigurationError("review port is invalid")
        if command == "evaluate" and variant.startswith("coverage-"):
            validate_image_identifier(
                item["image_id"],
                "coverage image_id",
            )
        for name in ("expected_sha256",):
            if name in item:
                _sha256(item[name], f"workflows.{command}.{name}")
        if "include_patterns" in item and (
            not isinstance(item["include_patterns"], list)
            or not item["include_patterns"]
            or any(not isinstance(pattern, str) or not pattern.strip() for pattern in item["include_patterns"])
            or len(set(item["include_patterns"])) != len(item["include_patterns"])
        ):
            raise ConfigurationError("transfer include_patterns must be a nonempty unique string list")
        if command == "transfer":
            sealed_patterns = (
                ["*.*"] if variant == "result-tree" else ["*.csv", "*.txt"]
            )
            item.setdefault("include_patterns", sealed_patterns)
            if item["include_patterns"] != sealed_patterns:
                raise ConfigurationError(
                    f"transfer {variant} include_patterns must remain sealed"
                )
        if command == "propose" and variant == "sam2-proposals":
            _choice(item["device"], "proposal device", frozenset({"cpu", "cuda"}))
            _choice(
                item.get("feature_mode", "balanced"),
                "feature mode",
                frozenset({"balanced", "rich", "ultra"}),
            )
            _choice(
                item.get("sam2_policy", "both"),
                "SAM2 policy",
                frozenset({"both", "auto", "prompt"}),
            )
            if "save_all_outputs" in item and not isinstance(item["save_all_outputs"], bool):
                raise ConfigurationError("save_all_outputs must be boolean")
        if command == "infer":
            item.setdefault("decision_threshold", 0.5)
            item.setdefault("mode", "xgb_recall")
            item.setdefault("save_all", True)
            if type(item["save_all"]) is not bool:
                raise ConfigurationError("inference save_all must be an exact boolean")
            item.setdefault("sam2_policy", "both")
            _choice(
                item["sam2_policy"],
                "inference SAM2 policy",
                frozenset({"both", "auto", "prompt"}),
            )
            item.setdefault("yolo_confidence", 0.20)
            item.setdefault("yolo_iou", 0.60)
            item.setdefault("detector_policy", "hybrid")
            item.setdefault(
                "detector_missing",
                "ignore" if variant == "sam2-xgb" else "reject",
            )
            item.setdefault("detector_threshold", 0.5)
            item.setdefault("hybrid_yolo_bias", 0.8)
            for name in (
                "yolo_confidence",
                "yolo_iou",
                "detector_threshold",
                "hybrid_yolo_bias",
            ):
                number = _number(item[name], f"inference {name}", minimum=0.0)
                if number > 1.0:
                    raise ConfigurationError(f"inference {name} must not exceed 1.0")
            _choice(
                item["detector_policy"],
                "inference detector policy",
                frozenset({"and", "or", "xgb", "yolo", "hybrid"}),
            )
            _choice(
                item["detector_missing"],
                "inference missing-detector policy",
                frozenset({"reject", "ignore"}),
            )
            _choice(
                item["embed_backbone"],
                "embedding backbone",
                frozenset({"resnet18", "resnet34", "resnet50", "resnet101"}),
            )
            _choice(
                item["device"],
                "inference device",
                frozenset({"cpu", "cuda"}),
            )
            _choice(
                item.get("mode", "xgb_recall"),
                "inference mode",
                frozenset({"gate", "xgb", "xgb_recall"}),
            )
        if command == "train" and variant == "yolo":
            if item["model_output"].get("kind") != "directory":
                raise ConfigurationError(
                    "YOLO model_output must be an explicit directory artifact"
                )
            for name, default in YOLO_DEFAULTS.items():
                if name in {"imgsz", "epochs", "batch"}:
                    if name in item:
                        _integer(item[name], f"YOLO {name}", minimum=1)
                elif name in {"patience", "close_mosaic"}:
                    if name in item:
                        _integer(item[name], f"YOLO {name}", minimum=0)
                elif name == "seed":
                    if name in item and _integer(item[name], "YOLO seed", minimum=0) != 0:
                        raise ConfigurationError("sealed YOLO seed must remain 0")
                elif name == "deterministic":
                    if name in item and item[name] is not True:
                        raise ConfigurationError("sealed YOLO deterministic policy must remain enabled")
                elif name == "device":
                    if name in item and item[name] != "cpu":
                        raise ConfigurationError("registered YOLO training is sealed to CPU")
                elif name == "lr0":
                    if name in item:
                        _number(item[name], f"YOLO {name}", minimum=0.0)
                        if item[name] == 0:
                            raise ConfigurationError("YOLO lr0 must be greater than zero")
                elif name in item:
                    _number(item[name], f"YOLO {name}", minimum=0.0)
                item.setdefault(name, default)
        workflows[command] = item
    return DomainConfig(
        schema=SCHEMA,
        execution=execution,
        notifications=notifications,
        workflows=workflows,
        fixture_classification=classification,
    )


def load_config(
    path: Path,
    *,
    fixture: bool = False,
    environment: Mapping[str, str] | None = None,
) -> DomainConfig:
    raw = path.read_bytes()
    if len(raw) > 1024 * 1024:
        raise ConfigurationError("configuration exceeds the static size limit")
    try:
        if path.suffix.casefold() == ".json":
            value = json.loads(raw.decode("utf-8"))
        elif path.suffix.casefold() == ".toml":
            value = tomllib.loads(raw.decode("utf-8"))
        else:
            raise ConfigurationError("configuration must be JSON or TOML")
    except (UnicodeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError("configuration syntax is invalid") from exc
    return validate_mapping(value, fixture=fixture, environment=environment)
