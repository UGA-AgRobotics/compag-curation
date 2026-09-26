"""Runnable public-project facade with review pause, verified resume, and portable inference."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import stat
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .assets import asset_registry, fetch_asset, verify_asset
from .public_config import (
    CANONICAL_CPU_PROFILE,
    CANONICAL_GPU_PROFILE,
    EFFICIENT_GPU_PROFILE,
    FULL_IMAGE_GPU_PROFILE,
    GPU_EXECUTION_PROFILES,
    MAX_IMAGE_FILE_BYTES,
    PublicProjectConfig,
    V2_PIPELINE_PROFILES,
    _image_info,
    build_public_plan,
    check_local_assets,
    group_id_from_name,
    initialize_project,
    inspect_public_data,
    load_public_config,
)
from .public_io import (
    PublicIOError,
    bounded_csv_field_limit,
    canonical_json_value,
    canonical_json_bytes,
    compact_json_sha256,
    copy_new,
    fsync_directory,
    hash_file_snapshot,
    manifest_rows,
    owned_manifest_contract,
    portable_basename,
    purge_owned_runtime_cache_files,
    publish_directory_noreplace,
    publish_file_noreplace,
    rename_noreplace,
    scan_file_for_forbidden_bytes,
    sha256_file,
    stable_file,
    strict_json_bytes,
    write_new_bytes,
    write_new_json,
)
from .review.exchange import validate_review_table, write_review_export, write_review_import
from .run_state import (
    CANONICAL_STAGE60_REQUIRED_FILES,
    PAUSED_FOR_REVIEW_EXIT_CODE,
    STAGE_NAMES,
    STAGE_REQUIRED_FILES,
    RunBindings,
    RunState,
    _classify_staged_events,
    begin_stage,
    claim_run,
    completed_stage,
    open_resume,
    publish_stage,
    record_stage_failure,
    validate_complete_stage_chain,
    validate_run_namespace,
    validate_new_run_target,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


def _uses_v2_pipeline(config: object) -> bool:
    """Recognize v2 profiles while retaining compatibility with older adapters."""

    return bool(
        getattr(
            config,
            "uses_v2_pipeline",
            getattr(config, "is_canonical", False),
        )
    )


def _write_fd_all(fd: int, payload: bytes, role: str) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        _require(written > 0, f"short write to {role}")
        offset += written


def _record_inference_failure(staging: Path, run_id: str, error: BaseException) -> None:
    if not staging.exists() or staging.is_symlink() or not staging.is_dir():
        return
    payload = {
        "schema": "compag-curation-inference-failure/v1",
        "status": "FAILED_DIAGNOSTIC_PRESERVED",
        "run_id": run_id,
        "error_type": type(error).__name__,
        "error_code": "INFERENCE_EXECUTION_FAILED",
        "remediation": "Use the structured stderr error to correct the input, dependency, bundle, device, or permission defect and choose a new output name.",
    }
    try:
        if not (staging / "_FAILURE.json").exists() and not (staging / "_FAILURE.json").is_symlink():
            write_new_json(staging / "_FAILURE.json", payload)
        if not (staging / "FINAL_STATUS.json").exists() and not (staging / "FINAL_STATUS.json").is_symlink():
            write_new_json(staging / "FINAL_STATUS.json", payload)
    except BaseException:
        pass


def _validated_invocation(value: Mapping[str, object] | None) -> dict[str, object]:
    record = dict(value or {"source": "PYTHON_API", "call": "run_project"})
    source = record.get("source")
    if source == "CLI":
        _require(set(record) == {"source", "command", "mode", "exact_argv_sha256"}, "CLI invocation fields mismatch")
        _require(record.get("command") == "run" and record.get("mode") in {"NEW", "RESUME", "DRY_RUN"}, "CLI invocation mode mismatch")
        _require(re.fullmatch(r"[0-9a-f]{64}", str(record.get("exact_argv_sha256", ""))) is not None, "CLI argv hash is invalid")
    elif source == "PYTHON_API":
        _require(record == {"source": "PYTHON_API", "call": "run_project"}, "Python API invocation fields mismatch")
    elif source == "SYNTHETIC_REAL_BACKEND_DEMO":
        _require(
            set(record) == {"source", "operation"}
            and record.get("operation") in {"INITIAL_TO_REVIEW_PAUSE", "REVIEW_IMPORT_AND_RESUME"},
            "synthetic demo invocation fields mismatch",
        )
    else:
        raise PublicIOError("run invocation source is invalid")
    return record


def _json(path: Path) -> Mapping[str, object]:
    payload, _snapshot = stable_file(path, max_bytes=128 * 1024 * 1024)
    value = canonical_json_value(payload, f"JSON evidence {path.name}")
    _require(isinstance(value, dict), f"JSON evidence must be an object: {path.name}")
    return value


def _annotation_identity(config: PublicProjectConfig) -> dict[str, object] | None:
    if config.annotations is None:
        return None
    from .point_annotations import inspect_point_annotations

    inference_rows = inspect_public_data(config, require_inputs=True)["inference_images"]
    expected = {str(row["filename"]): (int(row["width"]), int(row["height"])) for row in inference_rows}
    rows, summary = inspect_point_annotations(config.annotations, expected_images=expected)
    inference_names = set(expected)
    evaluated_names = {str(row["image"]) for row in rows}
    _require(bool(inference_names & evaluated_names), "CVAT points do not reference any configured inference image")
    return {**summary, "configured_inference_matches": sorted(inference_names & evaluated_names)}


def _preflight_inputs(config: PublicProjectConfig) -> dict[str, object]:
    _require(sha256_file(config.config_path) == config.config_sha256, "RUN_INPUT_DRIFT: configuration bytes changed after parsing")
    inspection = inspect_public_data(config, require_inputs=True)
    if _uses_v2_pipeline(config):
        _require(
            int(inspection["group_count"]) >= 6,
            "at least six independent training groups are required for a v2 pipeline profile",
        )
    else:
        _require(
            int(inspection["group_count"]) >= 4,
            "at least four independent training groups are required",
        )
    _require(int(inspection["inference_image_count"]) >= 1, "at least one independent inference image is required")
    annotation = _annotation_identity(config)
    checked_assets = check_local_assets(config)
    assets = [
        {
            "role": str(row["role"]),
            "filename": str(row["filename"]),
            "sha256": str(row["sha256"]),
            "size_bytes": int(row["size_bytes"]),
        }
        for row in checked_assets["assets"]
    ]
    assets.sort(key=lambda row: str(row["role"]))
    input_identity = {
        "training_images": inspection["images"],
        "inference_images": inspection["inference_images"],
        "annotations": annotation,
    }
    return {
        "inspection": inspection,
        "annotations": annotation,
        "assets": assets,
        "input_identity": input_identity,
    }


def _preflight_from_inputs(
    config: PublicProjectConfig,
    inputs: Mapping[str, object],
    dependencies: Mapping[str, object],
) -> tuple[RunBindings, dict[str, object]]:
    from .public_backend import validate_decodable_images

    inspection = inputs["inspection"]
    _require(isinstance(inspection, Mapping), "preflight inspection evidence is invalid")
    assets = inputs["assets"]
    _require(isinstance(assets, list), "preflight asset evidence is invalid")
    input_identity = inputs["input_identity"]
    _require(isinstance(input_identity, Mapping), "preflight input identity is invalid")
    decode = validate_decodable_images(config, inspection)
    bindings = RunBindings(
        config_sha256=config.config_sha256,
        input_manifest_sha256=compact_json_sha256(input_identity),
        dependency_sha256=str(dependencies["sha256"]),
        asset_sha256=compact_json_sha256(assets),
    )
    return bindings, {
        "schema": "compag-curation-public-preflight/v1",
        "status": "PASS",
        "profile": config.profile,
        "device": config.device,
        "bindings": asdict(bindings),
        "inspection": inspection,
        "annotations": inputs["annotations"],
        "assets": assets,
        "dependencies": dict(dependencies),
        "decode_validation": decode,
    }


def _preflight(config: PublicProjectConfig) -> tuple[RunBindings, dict[str, object]]:
    from .public_backend import require_science_dependencies

    inputs = _preflight_inputs(config)
    dependencies = require_science_dependencies(config.device)
    return _preflight_from_inputs(config, inputs, dependencies)


def _require_gpu_executable_profile(config: PublicProjectConfig) -> None:
    _require(
        config.profile in GPU_EXECUTION_PROFILES and config.device == "cuda",
        "executable runtime requires a supported Lite or Full GPU profile with device=cuda; balanced and canonical CPU profiles are historical verification-only",
    )


def _disk_preflight(
    config: PublicProjectConfig,
    output: Path,
    *,
    completed_stages: Sequence[str] = (),
) -> dict[str, object]:
    target = _future_path(output)
    if target.exists() or target.is_symlink():
        _require(target.is_dir() and not target.is_symlink(), "capacity target is an unsafe existing node")
    inspection = inspect_public_data(config, require_inputs=True)
    training_pixels = sum(int(row["width"]) * int(row["height"]) for row in inspection["images"])
    inference_pixels = sum(int(row["width"]) * int(row["height"]) for row in inspection["inference_images"])
    checkpoint_bytes = config.checkpoint.lstat().st_size
    embedding_weights = getattr(config, "embedding_weights", None)
    embedding_bytes = (
        embedding_weights.lstat().st_size
        if _uses_v2_pipeline(config) and embedding_weights is not None
        else 0
    )
    completed = set(completed_stages)
    _require(completed <= set(STAGE_NAMES), "capacity preflight received an unknown completed stage")
    components = {
        "checkpoint_bundle_and_staging": (
            0
            if "50_train_bundle" in completed
            else (checkpoint_bytes + embedding_bytes) * 2
        ),
        "training_tiles_overlays_and_tables": 0 if "20_proposals_features" in completed else training_pixels * 3 * 6,
        "inference_tiles_overlays_and_tables": 0 if "60_evaluate" in completed else inference_pixels * 3 * 6,
        "classifier_reports_and_diagnostics": 64 * 1024 * 1024 if "50_train_bundle" in completed else 512 * 1024 * 1024,
    }
    estimated = sum(components.values())
    required = (estimated * 120 + 99) // 100
    stats = os.statvfs(target if target.exists() else target.parent)
    available = stats.f_bavail * stats.f_frsize
    _require(available >= required, f"insufficient disk space: require {required} bytes including reserve, have {available}")
    return {
        "schema": "compag-curation-run-disk-preflight/v1",
        "status": "PASS",
        "target_filesystem_device": (target if target.exists() else target.parent).stat().st_dev,
        "estimate_components": components,
        "estimated_peak_bytes": estimated,
        "reserve_percent": 20,
        "required_free_bytes": required,
        "available_bytes": available,
        "completed_stages": sorted(completed),
    }


def _assert_preflight(
    config: PublicProjectConfig,
    expected: RunBindings,
    captured_dependencies: Mapping[str, object],
) -> None:
    if config.device == "cuda":
        from .public_backend import revalidate_science_gpu_dependencies

        inputs = _preflight_inputs(config)
        dependencies = revalidate_science_gpu_dependencies(
            config.device,
            captured_dependencies,
        )
        observed, _payload = _preflight_from_inputs(config, inputs, dependencies)
    else:
        observed, _payload = _preflight(config)
    _require(observed == expected, "RUN_INPUT_DRIFT: config, data, annotations, assets, or dependencies changed")


def _future_path(path: Path) -> Path:
    absolute = path.absolute()
    parent = absolute.parent.resolve(strict=True)
    candidate = parent / absolute.name
    _require(candidate == absolute and candidate.name not in {"", ".", ".."}, "output path contains a symlinked or unsafe component")
    portable_basename(candidate.name, "output name")
    return candidate


def _assert_output_disjoint(output: Path, *, directories: Sequence[Path], files: Sequence[Path]) -> Path:
    candidate = _future_path(output)
    for directory in directories:
        root = directory.absolute()
        _require(root == root.resolve(strict=True), f"input root contains a symlink component: {directory}")
        _require(
            candidate != root and root not in candidate.parents and candidate not in root.parents,
            f"output overlaps protected input root: {directory}",
        )
    for path in files:
        resolved = path.absolute()
        _require(resolved == resolved.resolve(strict=True), f"input path contains a symlink component: {path}")
        _require(candidate != resolved and candidate not in resolved.parents, f"output aliases or contains protected input: {path}")
    return candidate


_RUN_PRIVATE_LOCATOR_TOKENS = (
    b"/home/",
    b"/root/",
    b"/users/",
    b"file" + b"://",
    b"wsl.localhost",
    b"\\\\wsl",
    b":\\users\\",
    b":\\\\users\\\\",
    b"<home>" + b"/clean",
    b"clean" + b"/revision/",
)

_RUN_CLASSIFIER_RELATIVE = "stages/50_train_bundle/model_bundle/classifier.ubj"


def _validated_run_member_identity(
    path: Path,
    relative: str,
) -> tuple[str, os.stat_result]:
    if relative != _RUN_CLASSIFIER_RELATIVE:
        return scan_file_for_forbidden_bytes(path, _RUN_PRIVATE_LOCATOR_TOKENS)

    # The legacy balanced bundle has a different feature schema and retains the
    # historical byte-token scan.  Canonical v2 UBJ typed arrays are binary
    # numeric data: locator-like byte sequences there are not metadata.  Let
    # the v2 validator inspect XGBoost's complete decoded JSON projection while
    # this layer preserves the stable-file and hash contract.
    from .model_bundle import (
        BUNDLE_SCHEMA_V2,
        MAX_BUNDLE_JSON_BYTES,
        MAX_CLASSIFIER_BYTES,
        _validate_canonical_ubj,
    )

    bundle_payload, _bundle_snapshot = stable_file(
        path.with_name("bundle.json"),
        max_bytes=MAX_BUNDLE_JSON_BYTES,
    )
    try:
        bundle = canonical_json_value(bundle_payload, "run model bundle descriptor")
    except PublicIOError:
        return scan_file_for_forbidden_bytes(path, _RUN_PRIVATE_LOCATOR_TOKENS)
    if not isinstance(bundle, Mapping) or bundle.get("schema") != BUNDLE_SCHEMA_V2:
        return scan_file_for_forbidden_bytes(path, _RUN_PRIVATE_LOCATOR_TOKENS)

    classifier, before = stable_file(path, max_bytes=MAX_CLASSIFIER_BYTES)
    digest = hashlib.sha256(classifier).hexdigest()
    _validate_canonical_ubj(classifier)
    digest_after, after = hash_file_snapshot(
        path,
        require_single_link=True,
        max_bytes=MAX_CLASSIFIER_BYTES,
    )
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_gid,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    _require(
        digest_after == digest and after_identity == before_identity,
        "run classifier changed during decoded-metadata privacy validation",
    )
    return digest_after, after


def _validated_run_rows(
    state: RunState,
    *,
    allow_final: bool,
) -> tuple[list[dict[str, object]], list[str], list[dict[str, str]]]:
    allowed_empty = validate_run_namespace(state.root, state.run_id, allow_final=allow_final)
    for relative in allowed_empty:
        lowered_name = relative.encode("utf-8").lower()
        _require(
            not any(token in lowered_name for token in _RUN_PRIVATE_LOCATOR_TOKENS),
            f"private locator found in empty output directory name: {Path(relative).name}",
        )
    before, before_directories = owned_manifest_contract(
        state.root,
        allowed_empty_directories=allowed_empty,
    )
    for row in before_directories:
        lowered_name = str(row["path"]).encode("utf-8").lower()
        _require(
            not any(token in lowered_name for token in _RUN_PRIVATE_LOCATOR_TOKENS),
            f"private locator found in output directory name: {Path(str(row['path'])).name}",
        )
    for row in before:
        relative = str(row["path"])
        lowered_name = relative.encode("utf-8").lower()
        _require(
            not any(token in lowered_name for token in _RUN_PRIVATE_LOCATOR_TOKENS),
            f"private locator found in output member name: {Path(relative).name}",
        )
        digest, snapshot = _validated_run_member_identity(
            state.root / relative,
            relative,
        )
        _require(
            digest == row["sha256"]
            and snapshot.st_size == row["size_bytes"]
            and f"{stat.S_IMODE(snapshot.st_mode):04o}" == row["mode_octal"],
            f"output member changed during privacy validation: {Path(relative).name}",
        )
    after, after_directories = owned_manifest_contract(
        state.root,
        allowed_empty_directories=allowed_empty,
    )
    _require(after == before and after_directories == before_directories, "run tree changed during privacy validation")
    return after, sorted(allowed_empty), after_directories


def _existing_completion(state: RunState) -> dict[str, object] | None:
    all_rows, empty_directories, directories = _validated_run_rows(state, allow_final=True)
    final = state.root / "FINAL"
    if not final.exists() and not final.is_symlink():
        return None
    validate_complete_stage_chain(state)
    info = final.lstat()
    _require(stat.S_ISDIR(info.st_mode) and not final.is_symlink(), "completed-run evidence is unsafe")
    complete_path = final / "RUN_COMPLETE.json"
    manifest_path = final / "OUTPUT_MANIFEST.json"
    _require({path.name for path in final.iterdir()} == {"RUN_COMPLETE.json", "OUTPUT_MANIFEST.json"}, "completed-run evidence closure is invalid")
    for path in (complete_path, manifest_path):
        evidence_info = path.lstat()
        _require(
            stat.S_ISREG(evidence_info.st_mode)
            and not path.is_symlink()
            and evidence_info.st_nlink == 1
            and stat.S_IMODE(evidence_info.st_mode) == 0o644
            and evidence_info.st_uid == os.getuid()
            and evidence_info.st_gid == os.getgid(),
            f"completed-run evidence file metadata changed: {path.name}",
        )
    complete = dict(_json(complete_path))
    manifest = _json(manifest_path)
    _require(
        set(complete) == {"schema", "status", "run_id", "stages", "report", "model_bundle", "fresh_inference"}
        and complete == {
            "schema": "compag-curation-public-run-completion/v1",
            "status": "PASS",
            "run_id": state.run_id,
            "stages": 8,
            "report": "stages/70_report/report.json",
            "model_bundle": "stages/50_train_bundle/model_bundle",
            "fresh_inference": "stages/60_evaluate/predictions.csv",
        },
        "run completion record is invalid",
    )
    _require(
        set(manifest)
        == {
            "schema",
            "status",
            "run_id",
            "completion_sha256",
            "members",
            "members_sha256",
            "empty_directories",
            "directories",
            "directories_sha256",
        }
        and manifest.get("schema") == "compag-curation-public-run-output-manifest/v1"
        and manifest.get("status") == "PASS"
        and manifest.get("run_id") == state.run_id,
        "run output manifest is invalid",
    )
    _require(manifest.get("completion_sha256") == sha256_file(complete_path), "run completion identity changed")
    rows = [row for row in all_rows if row["path"] != "FINAL/OUTPUT_MANIFEST.json"]
    _require(
        manifest.get("members") == rows
        and manifest.get("members_sha256") == compact_json_sha256(rows)
        and manifest.get("empty_directories") == empty_directories
        and manifest.get("directories") == directories
        and manifest.get("directories_sha256") == compact_json_sha256(directories),
        "completed run output changed",
    )
    return complete


def _produce_stage(
    state: RunState,
    name: str,
    required: Sequence[str],
    producer: Callable[[Path], None],
    *,
    dependency_guard: Callable[[], None] | None = None,
) -> Mapping[str, object]:
    existing = completed_stage(state, name)
    if existing is not None:
        if dependency_guard is not None:
            dependency_guard()
        _require(
            existing.get("required") == list(required),
            f"completed stage required-output variant changed: {name}",
        )
        return existing
    workspace = begin_stage(state, name)
    try:
        if dependency_guard is not None:
            dependency_guard()
        producer(workspace.root)
        if dependency_guard is not None:
            dependency_guard()
        return publish_stage(state, workspace, required)
    except BaseException as exc:
        record_stage_failure(state, workspace, exc)
        raise


def _gpu_stage_preflight_guard(
    config: PublicProjectConfig,
    expected_bindings: RunBindings,
    captured_dependencies: Mapping[str, object],
) -> Callable[[], None] | None:
    if config.device != "cuda":
        return None
    captured = dict(captured_dependencies)

    def guard() -> None:
        _assert_preflight(config, expected_bindings, captured)

    return guard


def _proposal_rows(path: Path, *, canonical: bool = False) -> list[dict[str, str]]:
    if canonical:
        from .canonical.service import CANONICAL_PROPOSAL_COLUMNS as proposal_columns
    else:
        from .public_backend import PROPOSAL_COLUMNS as proposal_columns
    from .review.exchange import MAX_REVIEW_ROWS

    payload, _snapshot = stable_file(path, max_bytes=512 * 1024 * 1024)
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise PublicIOError("proposal table is not UTF-8") from exc
    with bounded_csv_field_limit():
        with io.StringIO(text, newline="") as handle:
            reader = csv.DictReader(handle)
            _require(tuple(reader.fieldnames or ()) == proposal_columns, "proposal table header mismatch")
            rows = []
            try:
                for row in reader:
                    _require(len(rows) < MAX_REVIEW_ROWS, "proposal table exceeds the supported review-row bound")
                    rows.append(row)
            except csv.Error as exc:
                raise PublicIOError("proposal table contains invalid CSV framing") from exc
    for index, row in enumerate(rows, start=2):
        _require(None not in row and tuple(row) == proposal_columns, f"proposal row {index} has the wrong field count")
        _require(all(isinstance(value, str) for value in row.values()), f"proposal row {index} contains a missing value")
    _require(rows, "proposal table is empty")
    return rows


def _stage_zero(
    config: PublicProjectConfig,
    preflight: Mapping[str, object],
    invocation: Mapping[str, object],
    output: Path,
) -> None:
    copied = copy_new(config.config_path, output / "config.toml")
    _require(copied["sha256"] == preflight["bindings"]["config_sha256"], "configuration copy differs from preflight")
    write_new_json(output / "input_inventory.json", {
        "schema": "compag-curation-run-input-inventory/v1",
        "status": "PASS",
        "inspection": preflight["inspection"],
        "annotations": preflight["annotations"],
        "input_manifest_sha256": preflight["bindings"]["input_manifest_sha256"],
    })
    write_new_json(output / "asset_inventory.json", {
        "schema": "compag-curation-run-assets/v1",
        "status": "PASS",
        "assets": preflight["assets"],
        "asset_sha256": preflight["bindings"]["asset_sha256"],
    })
    write_new_json(output / "dependency_inventory.json", preflight["dependencies"])
    write_new_json(output / "disk_preflight.json", preflight["disk_preflight"])
    normalized_config = {
        "schema": "compag-curation-normalized-run-config/v1",
        "project_name": config.project_name,
        "profile": config.profile,
        "device": config.device,
        "seed": config.seed,
        "paths": {
            "images": config.images.relative_to(config.project_root).as_posix(),
            "inference_images": config.inference_images.relative_to(config.project_root).as_posix(),
            "annotations": None if config.annotations is None else config.annotations.relative_to(config.project_root).as_posix(),
            "asset_root": config.asset_root.relative_to(config.project_root).as_posix(),
        },
        "execution": {
            "tile_size": config.tile_size,
            "tile_overlap": config.tile_overlap,
            "points_per_side": config.points_per_side,
            "points_per_batch": config.points_per_batch,
            "pred_iou_threshold": config.pred_iou_threshold,
            "stability_threshold": config.stability_threshold,
            "max_proposals_per_tile": config.max_proposals_per_tile,
            "min_mask_area": config.min_mask_area,
        },
        "assets": {
            "checkpoint_sha256": config.checkpoint_sha256,
            "sam2_config_sha256": config.sam2_config_sha256,
            "sam2_config_locator": config.sam2_config_locator,
        },
        "config_sha256": config.config_sha256,
    }
    if _uses_v2_pipeline(config):
        normalized_config["execution"] = {
            **normalized_config["execution"],
            "tile_stride": config.tile_stride,
            "tile_edge_alignment": config.tile_edge_alignment,
            "tile_padding": config.tile_padding,
            "tile_format": config.tile_format,
            "proposal_backend": config.proposal_backend,
            "proposal_scales": list(config.proposal_scales),
            "crop_n_layers": config.crop_n_layers,
            "crop_n_points_downscale_factor": config.crop_n_points_downscale_factor,
            "crop_overlap_ratio": config.crop_overlap_ratio,
            "exclude_largest_mask": config.exclude_largest_mask,
            "feature_mode": config.feature_mode,
            "feature_crop_scales": list(config.feature_crop_scales),
            "embedding_backbone": config.embedding_backbone,
            "embedding_dimensions": config.embedding_dimensions,
            "masked_crop_padding": config.masked_crop_padding,
            "pca_components": config.pca_components,
            "group_test_fraction": config.group_test_fraction,
            "group_cv_splits": config.group_cv_splits,
            "hyperparameter_search_iterations": config.hyperparameter_search_iterations,
            "early_stopping_rounds": config.early_stopping_rounds,
            "inner_validation_fraction": config.inner_validation_fraction,
            "review_action_weights": {
                "accept": config.review_accept_weight,
                "flip": config.review_flip_weight,
                "sus_accept": config.review_suspect_weight,
                "sus_flip": config.review_suspect_weight,
                "skip": config.review_skip_weight,
            },
            "safe_smote": config.safe_smote,
            "scale_pos_weight": config.scale_pos_weight,
            "inference_threshold": config.inference_threshold,
            "nms_iou_threshold": config.nms_iou_threshold,
            "yolo_enabled": config.yolo_enabled,
        }
        normalized_config["assets"] = {
            **normalized_config["assets"],
            "embedding_weights_sha256": config.embedding_weights_sha256,
        }
    if config.profile == FULL_IMAGE_GPU_PROFILE:
        normalized_config["execution"] = {
            **normalized_config["execution"],
            "spatial_mode": config.spatial_mode,
            "execution_points_per_batch": config.execution_points_per_batch,
            "max_proposals_per_image": config.max_proposals_per_image,
        }
    write_new_json(output / "normalized_config.json", normalized_config)
    write_new_json(output / "command_record.json", {
        "schema": "compag-curation-run-command/v1",
        "service": "compag_curation.public_pipeline.run_project",
        "mode": "NEW_OR_VERIFIED_RESUME",
        "config_filename": config.config_path.name,
        "scientific_execution": True,
        "invocation": dict(invocation),
    })


def _stage_twenty(config: PublicProjectConfig, prepared: Path, output: Path) -> None:
    if _uses_v2_pipeline(config):
        from .canonical.service import generate_canonical_stage

        generate_canonical_stage(config, prepared, output, write_overlays=True)
    else:
        from .public_backend import generate_proposals

        generate_proposals(config, prepared, output)
    write_review_export(
        _proposal_rows(output / "proposals.csv", canonical=_uses_v2_pipeline(config)),
        output / "review_request.csv",
        canonical_actions=_uses_v2_pipeline(config),
    )


def _review_split_feasibility(rows: Sequence[object]) -> Mapping[str, object]:
    canonical = bool(rows) and getattr(rows[0], "review_action", None) is not None
    _require(
        all((getattr(row, "review_action", None) is not None) == canonical for row in rows),
        "review rows mix binary and canonical contracts",
    )
    if canonical:
        from .canonical.service import canonical_review_split_feasibility

        return canonical_review_split_feasibility(rows)
    from .public_backend import review_split_feasibility

    return review_split_feasibility(
        [(getattr(row, "group_id"), getattr(row, "label")) for row in rows]
    )


def _validated_review_source(exported: Path, supplied: Path) -> tuple[str, int]:
    rows, result = validate_review_table(exported, supplied)
    _review_split_feasibility(rows)
    return str(result["reviewed_sha256"]), len(rows)


def _stage_review(exported: Path, supplied: Path, expected_sha256: str, output: Path) -> None:
    copied = copy_new(supplied, output / "reviewed_supplied.csv")
    _require(copied["sha256"] == expected_sha256, "reviewed table changed after validation")
    rows, result = validate_review_table(exported, output / "reviewed_supplied.csv")
    _require(result["reviewed_sha256"] == expected_sha256, "copied review table differs from validated input")
    split_feasibility = _review_split_feasibility(rows)
    normalized = write_review_import(rows, output / "reviewed.csv")
    write_new_json(output / "review_import.json", {
        "schema": "compag-curation-review-import/v1",
        "status": "PASS",
        "source_sha256": expected_sha256,
        "normalized_sha256": normalized["sha256"],
        "rows": result["rows"],
        "positive_rows": result["positive_rows"],
        "negative_rows": result["negative_rows"],
        "split_feasibility": split_feasibility,
        "immutable_identity_fields": ["proposal_id", "proposal_sha256", "image_id", "image_sha256", "group_id"],
    })


def _validated_canonical_inference_result(
    value: Mapping[str, object],
    *,
    bundle_sha256: str,
    expected_inputs: Sequence[Mapping[str, object]],
    expected_profile: str = CANONICAL_CPU_PROFILE,
    expected_device: str = "cpu",
) -> dict[str, object]:
    """Validate the complete canonical result before trusting child evidence."""

    from .canonical.spec import (
        CANONICAL_DECISION_THRESHOLD,
        CANONICAL_FEATURE_CROP_SCALES,
        CANONICAL_FEATURE_ORDER_SHA256,
        CANONICAL_FULL_IMAGE_NMS_IOU,
        CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
        inference_amg_execution_points_per_batch,
    )
    from .canonical.active_learning_service import (
        CANONICAL_AL_RAW_FEATURE_ARCHIVE_SCHEMA,
    )
    from .canonical.service import CANONICAL_RAW_FEATURE_TABLE_COLUMNS

    _require(
        expected_profile in V2_PIPELINE_PROFILES,
        "canonical inference expected profile/device pairing is invalid",
    )
    expected_execution_device = (
        "cpu" if expected_profile == CANONICAL_CPU_PROFILE else "cuda"
    )
    expected_execution_batch = (
        CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH
        if expected_profile == CANONICAL_CPU_PROFILE
        else inference_amg_execution_points_per_batch(expected_profile)
    )
    _require(
        expected_device == expected_execution_device,
        "canonical inference expected profile/device pairing is invalid",
    )

    fields = {
        "schema", "status", "profile", "device", "bundle_sha256",
        "feature_order_sha256", "threshold", "threshold_method",
        "full_image_nms_iou", "sam2_execution_points_per_batch",
        "image_count", "tile_count", "proposal_count",
        "prediction_rows", "positive_rows", "kept_rows", "predictions_sha256",
        "raw_feature_archive", "input_inventory", "paper_result_reproduction",
    }
    _require(
        set(value) == fields
        and value.get("schema") == "compag-curation-canonical-inference/v1"
        and value.get("status") == "PASS"
        and value.get("profile") == expected_profile
        and value.get("device") == expected_device
        and value.get("bundle_sha256") == bundle_sha256
        and value.get("feature_order_sha256") == CANONICAL_FEATURE_ORDER_SHA256
        and value.get("threshold") == CANONICAL_DECISION_THRESHOLD
        and value.get("threshold_method") == "FIXED_CANONICAL_METHOD"
        and value.get("full_image_nms_iou") == CANONICAL_FULL_IMAGE_NMS_IOU
        and value.get("sam2_execution_points_per_batch")
        == expected_execution_batch
        and value.get("paper_result_reproduction") == "NOT_CLAIMED"
        and isinstance(value.get("predictions_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(value["predictions_sha256"])) is not None,
        "canonical inference result method or field closure is invalid",
    )
    count_fields = (
        "image_count", "tile_count", "proposal_count", "prediction_rows",
        "positive_rows", "kept_rows",
    )
    _require(
        all(type(value.get(name)) is int and int(value[name]) >= 0 for name in count_fields),
        "canonical inference result counts are invalid",
    )
    inventory = value.get("input_inventory")
    _require(isinstance(inventory, list), "canonical inference input inventory is invalid")
    expected_projection = [
        {
            key: row[key]
            for key in ("filename", "sha256", "size_bytes", "width", "height", "exif_orientation")
        }
        for row in expected_inputs
    ]
    _require(
        value["image_count"] == len(expected_projection) == len(inventory),
        "canonical inference image count changed",
    )
    tile_total = 0
    candidate_total = 0
    for row, expected in zip(inventory, expected_projection, strict=True):
        expected_group = group_id_from_name(str(expected["filename"]))
        expected_image_id = hashlib.sha256(
            b"compag-image-v1\0"
            + str(expected["filename"]).encode("utf-8")
            + b"\0"
            + expected_group.encode("ascii")
            + b"\0"
            + str(expected["sha256"]).encode("ascii")
        ).hexdigest()
        _require(
            isinstance(row, dict)
            and set(row)
            == {
                "image_name", "image_id", "image_sha256", "group_id",
                "size_bytes", "width", "height", "tile_count", "candidate_rows",
            }
            and row.get("image_name") == expected["filename"]
            and row.get("image_sha256") == expected["sha256"]
            and row.get("size_bytes") == expected["size_bytes"]
            and row.get("width") == expected["width"]
            and row.get("height") == expected["height"]
            and row.get("image_id") == expected_image_id
            and row.get("group_id") == expected_group
            and type(row.get("tile_count")) is int
            and int(row["tile_count"]) > 0
            and (
                expected_profile != FULL_IMAGE_GPU_PROFILE
                or int(row["tile_count"]) == 1
            )
            and type(row.get("candidate_rows")) is int
            and int(row["candidate_rows"]) >= 0,
            "canonical inference input-inventory row is invalid",
        )
        tile_total += int(row["tile_count"])
        candidate_total += int(row["candidate_rows"])
    _require(
        value["tile_count"] == tile_total
        and value["proposal_count"] == candidate_total
        and value["prediction_rows"] == candidate_total
        and (
            expected_profile != FULL_IMAGE_GPU_PROFILE
            or value["tile_count"] == value["image_count"]
        )
        and 0 <= int(value["kept_rows"]) <= int(value["positive_rows"]) <= candidate_total,
        "canonical inference result count closure changed",
    )
    archive = value.get("raw_feature_archive")
    archive_fields = {
        "schema", "status", "profile", "basename", "columns_sha256",
        "crop_scales", "rows_per_proposal", "proposal_count", "row_count",
        "bundle_sha256", "predictions_sha256", "sha256", "size_bytes",
    }
    _require(
        isinstance(archive, dict)
        and set(archive) == archive_fields
        and archive.get("schema") == CANONICAL_AL_RAW_FEATURE_ARCHIVE_SCHEMA
        and archive.get("status") == "PASS"
        and archive.get("profile") == expected_profile
        and archive.get("basename") == "raw_features.csv"
        and archive.get("columns_sha256")
        == compact_json_sha256(list(CANONICAL_RAW_FEATURE_TABLE_COLUMNS))
        and archive.get("crop_scales") == list(CANONICAL_FEATURE_CROP_SCALES)
        and archive.get("rows_per_proposal") == len(CANONICAL_FEATURE_CROP_SCALES)
        and archive.get("proposal_count") == candidate_total
        and archive.get("row_count")
        == candidate_total * len(CANONICAL_FEATURE_CROP_SCALES)
        and archive.get("bundle_sha256") == bundle_sha256
        and archive.get("predictions_sha256") == value["predictions_sha256"]
        and isinstance(archive.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(archive["sha256"])) is not None
        and type(archive.get("size_bytes")) is int
        and int(archive["size_bytes"]) > 0,
        "canonical inference raw-feature archive receipt is invalid",
    )
    return dict(value)


def _validated_canonical_raw_feature_archive(
    root: Path,
    result: Mapping[str, object],
) -> dict[str, object]:
    """Reparse and bind the four-scale archive to its result and predictions."""

    from .canonical.active_learning_service import (
        read_canonical_active_learning_inference,
    )

    archive = read_canonical_active_learning_inference(root)
    receipt = result.get("raw_feature_archive")
    _require(
        isinstance(receipt, dict)
        and archive.bundle_sha256 == result.get("bundle_sha256")
        and archive.predictions_sha256 == result.get("predictions_sha256")
        and archive.raw_features_sha256 == receipt.get("sha256")
        and archive.inference_result_sha256 == sha256_file(root / "inference_result.json")
        and len(archive.pool_rows) == result.get("prediction_rows")
        and len(archive.feature_rows) == receipt.get("row_count"),
        "canonical inference raw-feature archive differs from its result",
    )
    return {
        "schema": "compag-curation-canonical-raw-feature-validation/v1",
        "status": "PASS",
        "proposal_count": len(archive.pool_rows),
        "row_count": len(archive.feature_rows),
        "sha256": archive.raw_features_sha256,
    }


def _validated_canonical_predictions(
    path: Path,
    result: Mapping[str, object],
) -> dict[str, object]:
    """Parse and bind the canonical prediction table before publication."""

    from .canonical.service import (
        CANONICAL_INFERENCE_COLUMNS,
        _canonical_nms_indices,
        _read_csv_exact,
    )
    from .canonical.spec import (
        CANONICAL_DECISION_THRESHOLD,
        CANONICAL_FULL_IMAGE_NMS_IOU,
    )
    rows = _read_csv_exact(
        path,
        CANONICAL_INFERENCE_COLUMNS,
        require_rows=False,
    )
    inventory_value = result.get("input_inventory")
    _require(isinstance(inventory_value, list), "canonical prediction inventory is invalid")
    inventory: dict[str, Mapping[str, object]] = {}
    for value in inventory_value:
        _require(isinstance(value, dict), "canonical prediction inventory row is invalid")
        image_name = value.get("image_name")
        _require(
            isinstance(image_name, str) and image_name not in inventory,
            "canonical prediction inventory image names are invalid",
        )
        inventory[image_name] = value

    def integer(text: str, role: str, *, minimum: int = 0) -> int:
        _require(
            re.fullmatch(r"0|[1-9][0-9]*", text) is not None,
            f"canonical prediction {role} is not a canonical integer",
        )
        value = int(text)
        _require(value >= minimum, f"canonical prediction {role} is below its minimum")
        return value

    def floating(text: str, role: str) -> float:
        try:
            value = float(text)
        except ValueError as exc:
            raise PublicIOError(f"canonical prediction {role} is not numeric") from exc
        normalized = format(0.0 if value == 0.0 else value, ".17g")
        _require(
            math.isfinite(value) and text == normalized,
            f"canonical prediction {role} is not a canonical finite float",
        )
        return value

    proposal_ids: set[str] = set()
    detection_ids: set[str] = set()
    per_image = {name: 0 for name in inventory}
    positive_rows = 0
    kept_rows = 0
    row_order: list[tuple[str, str, float]] = []
    nms_rows: dict[
        str,
        list[tuple[tuple[int, int, int, int], float, str, int, int]],
    ] = {
        name: [] for name in inventory
    }
    for ordinal, row in enumerate(rows, start=1):
        _require(tuple(row) == CANONICAL_INFERENCE_COLUMNS, "canonical prediction row schema changed")
        image_name = row["image"]
        expected = inventory.get(image_name)
        _require(expected is not None, "canonical prediction references an unknown input image")
        proposal_id = row["proposal_id"]
        detection_id = row["detection_id"]
        scale = floating(row["scale"], "feature scale")
        _require(
            re.fullmatch(r"[0-9a-f]{64}", proposal_id) is not None,
            "canonical prediction proposal identity is invalid",
        )
        expected_detection = hashlib.sha256(
            b"compag-canonical-detection-v1\0"
            + proposal_id.encode("ascii")
            + b"\0"
            + row["scale"].encode("ascii", errors="strict")
        ).hexdigest()
        _require(
            row["id"] == str(ordinal)
            and row["proposal_sha256"] == proposal_id
            and proposal_id not in proposal_ids
            and re.fullmatch(r"[0-9a-f]{64}", detection_id) is not None
            and detection_id == expected_detection
            and detection_id not in detection_ids
            and row["full_image"] == image_name
            and row["image_id"] == expected.get("image_id")
            and row["image_sha256"] == expected.get("image_sha256")
            and row["group_id"] == expected.get("group_id")
            and re.fullmatch(r"[0-9a-f]{64}", row["tile_sha256"]) is not None
            and re.fullmatch(r"[0-9a-f]{64}", row["mask_sha256"]) is not None
            and scale == 1.0,
            "canonical prediction identity or input binding is invalid",
        )
        portable_basename(row["tile_name"], "canonical prediction tile name")
        x = integer(row["x"], "x")
        y = integer(row["y"], "y")
        width = integer(row["w"], "width", minimum=1)
        height = integer(row["h"], "height", minimum=1)
        orig_w = integer(row["orig_w"], "original width", minimum=1)
        orig_h = integer(row["orig_h"], "original height", minimum=1)
        _require(
            integer(row["bbox_x1"], "bbox x1") == x
            and integer(row["bbox_y1"], "bbox y1") == y
            and integer(row["bbox_x2"], "bbox x2", minimum=1) == x + width
            and integer(row["bbox_y2"], "bbox y2", minimum=1) == y + height
            and orig_w == expected.get("width")
            and orig_h == expected.get("height")
            and x + width <= orig_w
            and y + height <= orig_h,
            "canonical prediction geometry is invalid",
        )
        try:
            polygon = json.loads(row["poly"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise PublicIOError("canonical prediction polygon is invalid") from exc
        _require(
            isinstance(polygon, list)
            and len(polygon) % 2 == 0
            and all(type(value) is int and value >= 0 for value in polygon)
            and all(int(value) <= orig_w for value in polygon[0::2])
            and all(int(value) <= orig_h for value in polygon[1::2]),
            "canonical prediction polygon is invalid",
        )
        probability = floating(row["probability"], "probability")
        _require(
            0.0 <= probability <= 1.0
            and row["xgb_p"] == row["probability"]
            and row["prediction"] in {"0", "1"}
            and int(row["prediction"]) == int(probability >= CANONICAL_DECISION_THRESHOLD)
            and row["kept"] in {"0", "1"}
            and int(row["kept"]) <= int(row["prediction"]),
            "canonical prediction score or decision is invalid",
        )
        proposal_ids.add(proposal_id)
        detection_ids.add(detection_id)
        per_image[image_name] += 1
        positive_rows += int(row["prediction"])
        kept_rows += int(row["kept"])
        row_order.append((str(row["image_id"]), proposal_id, scale))
        nms_rows[image_name].append(
            (
                (x, y, x + width, y + height),
                probability,
                proposal_id,
                int(row["prediction"]),
                int(row["kept"]),
            )
        )

    _require(row_order == sorted(row_order), "canonical prediction row order changed")
    for image_name, image_rows in nms_rows.items():
        positive = [row for row in image_rows if row[3] == 1]
        expected_kept: set[int] = set()
        if positive:
            indices = _canonical_nms_indices(
                [row[0] for row in positive],
                [row[1] for row in positive],
                [row[2] for row in positive],
                CANONICAL_FULL_IMAGE_NMS_IOU,
            )
            _require(
                all(0 <= index < len(positive) for index in indices)
                and len(set(indices)) == len(indices),
                "canonical prediction NMS returned invalid indices",
            )
            expected_kept = set(indices)
        _require(
            all(row[4] == int(index in expected_kept) for index, row in enumerate(positive))
            and all(row[4] == 0 for row in image_rows if row[3] == 0),
            f"canonical prediction kept decisions differ from full-image NMS: {image_name}",
        )

    _require(
        len(rows) == result.get("prediction_rows") == result.get("proposal_count")
        and positive_rows == result.get("positive_rows")
        and kept_rows == result.get("kept_rows")
        and all(per_image[name] == row.get("candidate_rows") for name, row in inventory.items())
        and sha256_file(path) == result.get("predictions_sha256"),
        "canonical prediction table differs from the inference result",
    )
    return {
        "rows": len(rows),
        "positive_rows": positive_rows,
        "kept_rows": kept_rows,
        "sha256": str(result["predictions_sha256"]),
    }


def _captured_inference_child_environment(
    device: str,
    captured_dependencies: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object] | None, dict[str, str]]:
    """Validate a parent receipt and preserve its CUDA visibility for a child."""

    dependencies = dict(captured_dependencies)
    digest = dependencies.get("sha256")
    _require(
        isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
        and digest
        == compact_json_sha256(
            {key: value for key, value in dependencies.items() if key != "sha256"}
        ),
        "captured inference dependency receipt is invalid",
    )
    if device != "cuda":
        _require(device == "cpu", "fresh inference device is unsupported")
        return dependencies, None, {}
    visibility = dependencies.get("cuda_visibility")
    _require(
        dependencies.get("schema") == "compag-curation-science-gpu-dependencies/v2"
        and dependencies.get("requested_device") == device
        and isinstance(visibility, Mapping),
        "captured science-gpu dependency receipt is invalid",
    )
    from .runtime_lock import preserved_cuda_visibility_environment

    try:
        child_environment = preserved_cuda_visibility_environment(visibility)
    except RuntimeError as exc:
        raise PublicIOError(str(exc)) from exc
    return dependencies, dict(visibility), child_environment


def _validated_fresh_inference_environment(
    child_output: Path,
    expected_dependencies: Mapping[str, object],
) -> dict[str, object]:
    """Bind a child inference environment receipt to its completed publication."""

    environment = _json(child_output / "environment.json")
    status = _json(child_output / "FINAL_STATUS.json")
    dependencies = environment.get("dependencies")
    _require(
        set(environment) == {"schema", "run_id", "dependencies"}
        and environment.get("schema") == "compag-curation-inference-environment/v1"
        and environment.get("run_id") == status.get("run_id")
        and isinstance(dependencies, Mapping)
        and dict(dependencies) == dict(expected_dependencies)
        and dependencies.get("sha256") == expected_dependencies.get("sha256"),
        "fresh-process inference dependency receipt differs from run preflight",
    )
    return dict(environment)


def _fresh_canonical_inference(
    config: PublicProjectConfig,
    bundle: Path,
    output: Path,
    captured_dependencies: Mapping[str, object],
) -> Mapping[str, object]:
    """Execute canonical bundle inference in a new, explicitly closed process."""

    from .domain.inference import LocalCommandRunner, _require_inference_process_result
    from .runtime_lock import confined_accelerator_cache_environment

    dependencies, cuda_visibility, cuda_environment = (
        _captured_inference_child_environment(config.device, captured_dependencies)
    )
    expected_input_rows = _input_images_identity(config.inference_images)
    expected_inputs = [
        {
            key: row[key]
            for key in ("filename", "sha256", "size_bytes", "width", "height", "exif_orientation")
        }
        for row in expected_input_rows
    ]
    bundle_sha256 = sha256_file(bundle / "bundle.json")
    runtime = output / "fresh_process_runtime"
    runtime.mkdir(mode=0o700)
    write_new_json(runtime / "ENVIRONMENT_POLICY.json", {
        "schema": "compag-curation-fresh-process-environment/v1",
        "status": "CLOSED_EXPLICIT_ENVIRONMENT",
        "python_isolated": True,
        "python_no_user_site": True,
        "bytecode_disabled": True,
        "cpu_threads": 1,
        **(
            {
                "device": "cuda",
                "cuda_visibility": cuda_visibility,
                "cublas_workspace_config": ":4096:8",
            }
            if config.device == "cuda"
            else {}
        ),
    })
    child_output = output / "fresh_process_inference"
    argv = (
        sys.executable,
        "-I",
        "-B",
        "-m",
        "compag_curation",
        "infer",
        "--execute",
        "--images",
        str(config.inference_images),
        "--bundle",
        str(bundle),
        "--output",
        str(child_output),
        "--device",
        config.device,
    )
    environment = {
        "HOME": str(runtime),
        "TMPDIR": str(runtime),
        "XDG_CACHE_HOME": str(runtime / "xdg"),
        "TORCH_HOME": str(runtime / "torch"),
        "HF_HOME": str(runtime / "huggingface"),
        "MPLCONFIGDIR": str(runtime),
        "NUMBA_CACHE_DIR": str(runtime / "numba"),
        "JOBLIB_TEMP_FOLDER": str(runtime / "joblib"),
        **confined_accelerator_cache_environment(runtime),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": "",
        "PIP_CONFIG_FILE": "/dev/null",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        **cuda_environment,
        **(
            {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
            }
            if config.device == "cuda"
            else {}
        ),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
    }
    memory_release: Mapping[str, object] | None = None
    if config.device == "cuda":
        from .public_backend import release_science_gpu_allocator_caches

        memory_release = release_science_gpu_allocator_caches(config.device)
        write_new_json(
            runtime / "GPU_PARENT_ALLOCATOR_CACHE_RELEASE.json",
            memory_release,
        )
    try:
        process = LocalCommandRunner().run(
            argv,
            cwd=output,
            env=environment,
            acceptable_exit_codes=frozenset({0}),
            capture_output=True,
        )
    finally:
        retained_runtime_evidence = ["ENVIRONMENT_POLICY.json"]
        if memory_release is not None:
            retained_runtime_evidence.append(
                "GPU_PARENT_ALLOCATOR_CACHE_RELEASE.json"
            )
        purge_owned_runtime_cache_files(
            runtime,
            retained_files=retained_runtime_evidence,
        )
    try:
        process = _require_inference_process_result(
            process,
            argv,
            environment,
            output,
            frozenset({0}),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise PublicIOError(
            "canonical fresh-process inference failed; "
            f"return_code={getattr(process, 'returncode', 'INVALID')}; "
            f"stderr_sha256={hashlib.sha256(str(getattr(process, 'stderr', '')).encode('utf-8')).hexdigest()}"
        ) from exc
    _require(process.stderr == "", "canonical fresh-process inference emitted stderr")
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    _require(len(lines) == 1, "canonical fresh-process inference must emit one JSON record")
    child_record = strict_json_bytes(lines[0].encode("utf-8"), "canonical fresh-process stdout")
    _require(
        isinstance(child_record, dict) and child_record.get("status") == "PASS",
        "canonical fresh-process inference did not report PASS",
    )
    child_result = _json(child_output / "inference_result.json")
    validated_result = _validated_canonical_inference_result(
        child_result,
        bundle_sha256=bundle_sha256,
        expected_inputs=expected_inputs,
        expected_profile=config.profile,
        expected_device=config.device,
    )
    validated_predictions = _validated_canonical_predictions(
        child_output / "predictions.csv",
        validated_result,
    )
    validated_raw_features = _validated_canonical_raw_feature_archive(
        child_output,
        validated_result,
    )
    from .quick_demo import _verify_fresh_inference_output

    verified_output = _verify_fresh_inference_output(
        child_output,
        bundle_sha256=bundle_sha256,
        expected_result=validated_result,
        expected_inputs=expected_inputs,
        process_payload=child_record,
    )
    child_environment = _validated_fresh_inference_environment(
        child_output,
        dependencies,
    )
    copy_new(child_output / "predictions.csv", output / "predictions.csv")
    copy_new(child_output / "raw_features.csv", output / "raw_features.csv")
    copy_new(child_output / "inference_result.json", output / "inference_result.json")
    _require(
        _json(output / "inference_result.json") == validated_result
        and sha256_file(output / "predictions.csv") == child_record["predictions_sha256"],
        "canonical stage inference copy differs from the fresh-process output",
    )
    _require(
        _validated_canonical_raw_feature_archive(output, validated_result)
        == validated_raw_features,
        "canonical stage raw-feature archive differs from the fresh-process output",
    )
    return {
        "schema": "compag-curation-canonical-fresh-process/v1",
        "status": "PASS",
        "process_isolation": "NEW_PYTHON_INTERPRETER_SAME_OS_NAMESPACE",
        "output_validation": "FULL_COMPLETION_AND_MANIFEST_CLOSURE",
        "prediction_table_validation": validated_predictions,
        "raw_feature_archive_validation": validated_raw_features,
        "environment_policy": "fresh_process_runtime/ENVIRONMENT_POLICY.json",
        **(
            {
                "parent_gpu_allocator_cache_release": {
                    **memory_release,
                    "evidence": "fresh_process_runtime/GPU_PARENT_ALLOCATOR_CACHE_RELEASE.json",
                    "evidence_sha256": sha256_file(
                        runtime / "GPU_PARENT_ALLOCATOR_CACHE_RELEASE.json"
                    ),
                }
            }
            if memory_release is not None
            else {}
        ),
        "output": "fresh_process_inference",
        "dependency_sha256": dependencies["sha256"],
        "child_environment_sha256": sha256_file(
            child_output / "environment.json"
        ),
        "child_run_id": child_environment["run_id"],
        "output_manifest_sha256": verified_output["output_manifest_sha256"],
        "predictions_sha256": str(child_record["predictions_sha256"]),
        "stdout_sha256": hashlib.sha256(process.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(process.stderr.encode("utf-8")).hexdigest(),
        "return_code": process.returncode,
    }


def _stage_sixty(
    config: PublicProjectConfig,
    bundle: Path,
    output: Path,
    captured_dependencies: Mapping[str, object],
) -> None:
    if _uses_v2_pipeline(config):
        from .canonical.evaluation import evaluate_canonical_points

        fresh_process = _fresh_canonical_inference(
            config,
            bundle,
            output,
            captured_dependencies,
        )
        inference = _json(output / "inference_result.json")
    else:
        from .public_backend import infer_from_bundle

        inference = infer_from_bundle(config.inference_images, bundle, output, device=config.device)
        fresh_process = {
            "schema": "compag-curation-balanced-stage-inference/v1",
            "status": "IN_PROCESS",
        }
    point_result: Mapping[str, object]
    if config.annotations is None:
        point_result = {
            "schema": "compag-curation-point-coverage/v1",
            "status": "NOT_PROVIDED",
            "reason": "No CVAT point-annotation file was configured.",
        }
    else:
        evaluation = output / "point_evaluation"
        if _uses_v2_pipeline(config):
            point_result = evaluate_canonical_points(
                config.annotations,
                (output / "predictions.csv",),
                evaluation,
                profile=config.profile,
                images=tuple(
                    str(row["filename"])
                    for row in inspect_public_data(config, require_inputs=True)["inference_images"]
                ),
            )
        else:
            from .public_backend import evaluate_cvat_points

            evaluation.mkdir(mode=0o755)
            point_result = evaluate_cvat_points(
                config.annotations,
                output / "20_proposals_features/proposals.csv",
                output / "predictions.csv",
                evaluation,
                expected_images={
                    str(row["filename"]): (int(row["width"]), int(row["height"]))
                    for row in inspect_public_data(config, require_inputs=True)["inference_images"]
                },
            )
    write_new_json(output / "evaluation_status.json", {
        "schema": "compag-curation-inference-evaluation/v1",
        "status": "PASS",
        "inference": inference,
        "fresh_process": fresh_process,
        "point_coverage": point_result,
        "interpretation": "Point coverage is a recall-style coverage analysis and is not instance precision.",
    })


def _validated_canonical_published_stage_sixty(
    config: PublicProjectConfig,
    state: RunState,
) -> dict[str, object]:
    """Revalidate a published canonical stage 60 before reuse or finalization."""

    _require(
        _uses_v2_pipeline(config),
        "v2 stage-60 validation requires a v2 pipeline profile",
    )
    receipt = completed_stage(state, "60_evaluate")
    _require(
        receipt is not None
        and receipt.get("required") == list(CANONICAL_STAGE60_REQUIRED_FILES),
        "canonical stage 60 did not declare its raw-feature archive",
    )
    _validated_stage_sixty_runtime_evidence(receipt, device=config.device)
    stage = state.stages / "60_evaluate"
    expected_inputs = _input_images_identity(config.inference_images)
    result = _validated_canonical_inference_result(
        _json(stage / "inference_result.json"),
        bundle_sha256=sha256_file(
            state.stages / "50_train_bundle/model_bundle/bundle.json"
        ),
        expected_inputs=expected_inputs,
        expected_profile=config.profile,
        expected_device=config.device,
    )
    predictions = _validated_canonical_predictions(
        stage / "predictions.csv",
        result,
    )
    raw_features = _validated_canonical_raw_feature_archive(stage, result)
    return {
        "schema": "compag-curation-canonical-published-stage60-validation/v1",
        "status": "PASS",
        "receipt_sha256": sha256_file(stage / "_SUCCESS.json"),
        "predictions": predictions,
        "raw_features": raw_features,
    }


def _validated_stage_sixty_runtime_evidence(
    receipt: Mapping[str, object],
    *,
    device: str,
) -> dict[str, tuple[str, ...]]:
    """Require a published Stage 60 runtime to contain evidence, never caches."""

    _require(device in {"cpu", "cuda"}, "stage 60 runtime device is invalid")
    files = receipt.get("files")
    directories = receipt.get("directories")
    _require(
        isinstance(files, list) and isinstance(directories, list),
        "canonical stage 60 runtime receipt is invalid",
    )
    runtime_files = tuple(
        sorted(
            str(row["path"])
            for row in files
            if isinstance(row, Mapping)
            and isinstance(row.get("path"), str)
            and str(row["path"]).startswith("fresh_process_runtime/")
        )
    )
    runtime_directories = tuple(
        sorted(
            str(row["path"])
            for row in directories
            if isinstance(row, Mapping)
            and isinstance(row.get("path"), str)
            and (
                str(row["path"]) == "fresh_process_runtime"
                or str(row["path"]).startswith("fresh_process_runtime/")
            )
        )
    )
    expected_files = (
        "fresh_process_runtime/ENVIRONMENT_POLICY.json",
        *(
            ("fresh_process_runtime/GPU_PARENT_ALLOCATOR_CACHE_RELEASE.json",)
            if device == "cuda"
            else ()
        ),
    )
    _require(
        runtime_files == tuple(sorted(expected_files))
        and runtime_directories == ("fresh_process_runtime",),
        "canonical stage 60 runtime contains generated cache artifacts",
    )
    return {
        "files": runtime_files,
        "directories": runtime_directories,
    }


def _stage_sixty_required(config: PublicProjectConfig) -> tuple[str, ...]:
    return (
        CANONICAL_STAGE60_REQUIRED_FILES
        if _uses_v2_pipeline(config)
        else STAGE_REQUIRED_FILES["60_evaluate"]
    )


def _validate_published_stage_sixty_for_profile(
    config: PublicProjectConfig,
    state: RunState,
) -> Mapping[str, object]:
    receipt = completed_stage(state, "60_evaluate")
    _require(
        receipt is not None
        and receipt.get("required") == list(_stage_sixty_required(config)),
        "completed stage 60 does not match the configured profile",
    )
    if _uses_v2_pipeline(config):
        _validated_canonical_published_stage_sixty(config, state)
    return receipt


def _stage_report(config: PublicProjectConfig, state: RunState, output: Path) -> None:
    training = _json(state.stages / "50_train_bundle/training_result.json")
    inference = _json(state.stages / "60_evaluate/inference_result.json")
    evaluation = _json(state.stages / "60_evaluate/evaluation_status.json")
    receipts = {
        name: sha256_file(state.stages / name / "_SUCCESS.json")
        for name in (
            "00_input_inventory", "10_prepare", "20_proposals_features", "30_review_import",
            "40_group_split", "50_train_bundle", "60_evaluate",
        )
    }
    if config.is_canonical:
        profile_deviations: list[str] = []
    elif config.profile == EFFICIENT_GPU_PROFILE:
        profile_deviations = [
            "SAM2.1 Hiera Tiny replaces the canonical Hiera Large backbone; canonical-method and numerical equivalence are not claimed."
        ]
    elif config.profile == FULL_IMAGE_GPU_PROFILE:
        profile_deviations = [
            "Each image is processed as one lossless full warped frame without externally persisted tiles; SAM2.1 Hiera Large uses a two-layer internal crop pyramid for proposal coverage.",
            "Full-image ROI features and fitted bundles have an independent sealed spatial contract; equivalence to tiled profiles and reuse of tiled or published-r92 bundles are not claimed.",
        ]
    else:
        profile_deviations = [
            "Overlapping partial-edge PNG tiles replace canonical stride-512 padded JPEG tiles.",
            "SAM2.1 Hiera Tiny replaces the canonical Hiera Large checkpoint.",
            "The fixed 25-feature balanced profile excludes canonical ResNet50, prototype, PCA32, and searched XGBoost training.",
        ]
    report = {
        "schema": "compag-curation-public-report/v1",
        "status": "PASS",
        "profile": config.profile,
        "model_bundle_sha256": training["bundle"]["bundle_sha256"],
        "threshold_selected_on": training["threshold_selected_on"],
        "test_metrics": training["test_metrics"],
        "independent_inference": inference,
        "point_coverage": evaluation["point_coverage"],
        "stage_receipt_sha256": receipts,
        "original_research_data_and_models": "NOT_BUNDLED",
        "exact_paper_result_reproduction": "NOT_CLAIMED",
        "profile_deviations": profile_deviations,
    }
    write_new_json(output / "report.json", report)
    threshold_text = (
        "The decision threshold is the canonical fixed value 0.5."
        if _uses_v2_pipeline(config)
        else "The threshold was selected on the validation partition."
    )
    markdown = f"""# COMPAG curation run report

Status: PASS

Profile: `{config.profile}`

{threshold_text} Classification metrics in `report.json` are from the untouched
test partition. Fresh bundle inference used the separate inference-image directory.

This runnable profile does not bundle the original research data or fitted study
models, and exact paper-result reproduction is not claimed. Point coverage, when
provided, is a recall-style coverage analysis and is not instance precision.
"""
    write_new_bytes(output / "report.md", markdown.encode("ascii"))


def _finalize_run(state: RunState) -> dict[str, object]:
    _require(_existing_completion(state) is None, "run is already finalized")
    validate_complete_stage_chain(state)
    _classify_staged_events(state.root, state.run_id)
    validate_run_namespace(state.root, state.run_id, allow_final=False)
    state.append_event("FINALIZATION_STARTED", stage_count=8)
    complete = {
        "schema": "compag-curation-public-run-completion/v1",
        "status": "PASS",
        "run_id": state.run_id,
        "stages": 8,
        "report": "stages/70_report/report.json",
        "model_bundle": "stages/50_train_bundle/model_bundle",
        "fresh_inference": "stages/60_evaluate/predictions.csv",
    }
    final_staging = state.root.parent / f".compag-final-init.{state.run_id}.{uuid.uuid4()}"
    try:
        rows, empty_directories, directories = _validated_run_rows(state, allow_final=False)
        directories = sorted(
            [*directories, {"path": "FINAL", "mode_octal": "0755"}],
            key=lambda row: str(row["path"]),
        )
        final_staging.mkdir(mode=0o700)
        write_new_json(final_staging / "RUN_COMPLETE.json", complete)
        complete_info = (final_staging / "RUN_COMPLETE.json").lstat()
        rows.append({
            "path": "FINAL/RUN_COMPLETE.json",
            "sha256": sha256_file(final_staging / "RUN_COMPLETE.json"),
            "size_bytes": complete_info.st_size,
            "mode_octal": f"{stat.S_IMODE(complete_info.st_mode):04o}",
        })
        rows.sort(key=lambda row: str(row["path"]))
        write_new_json(final_staging / "OUTPUT_MANIFEST.json", {
            "schema": "compag-curation-public-run-output-manifest/v1",
            "status": "PASS",
            "run_id": state.run_id,
            "completion_sha256": sha256_file(final_staging / "RUN_COMPLETE.json"),
            "members": rows,
            "members_sha256": compact_json_sha256(rows),
            "empty_directories": empty_directories,
            "directories": directories,
            "directories_sha256": compact_json_sha256(directories),
        })
        os.chmod(final_staging, 0o755, follow_symlinks=False)
        final_info = final_staging.lstat()
        _require(
            stat.S_ISDIR(final_info.st_mode)
            and not final_staging.is_symlink()
            and stat.S_IMODE(final_info.st_mode) == 0o755
            and final_info.st_uid == os.getuid()
            and final_info.st_gid == os.getgid(),
            "FINAL staging directory metadata is unsafe",
        )
        fsync_directory(final_staging)
        rename_noreplace(final_staging, state.root / "FINAL")
    except BaseException as exc:
        if not final_staging.exists() and not final_staging.is_symlink() and (state.root / "FINAL").is_dir():
            try:
                _existing_completion(state)
            except BaseException:
                pass
            raise PublicIOError(
                "FINAL was atomically published but directory durability confirmation failed; rerun verified resume"
            ) from exc
        if not final_staging.exists() and not final_staging.is_symlink():
            try:
                final_staging.mkdir(mode=0o700)
            except BaseException:
                pass
        try:
            failure_path = final_staging / "_FAILURE.json"
            if final_staging.is_dir() and not final_staging.is_symlink() and not failure_path.exists() and not failure_path.is_symlink():
                write_new_json(failure_path, {
                    "schema": "compag-curation-finalization-failure/v1",
                    "status": "FAILED_DIAGNOSTIC_PRESERVED",
                    "run_id": state.run_id,
                    "error_type": type(exc).__name__,
                    "error_code": "ATOMIC_FINALIZATION_FAILED",
                    "remediation": "Inspect private stderr, correct the filesystem or integrity defect, and resume the owned run.",
                })
        except BaseException:
            pass
        try:
            state.append_event("FINALIZATION_FAILED", error_type=type(exc).__name__, staging=final_staging.name)
        except BaseException:
            pass
        raise
    verified = _existing_completion(state)
    _require(verified is not None, "published FINAL evidence is unavailable")
    return verified


def run_project(
    config_path: Path,
    *,
    output: Path | None = None,
    resume: Path | None = None,
    reviewed_labels: Path | None = None,
    dry_run: bool = False,
    invocation: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], int]:
    invocation_record = _validated_invocation(invocation)
    config = load_public_config(config_path, check_local=False)
    _require_gpu_executable_profile(config)
    bindings, preflight = _preflight(config)
    if dry_run:
        _require(output is not None and resume is None and reviewed_labels is None, "dry-run requires a new --output target and no resume/review path")
        candidate = _assert_output_disjoint(
            output,
            directories=(config.images, config.inference_images, config.asset_root),
            files=tuple(path for path in (config.config_path, config.annotations) if path is not None),
        )
        validate_new_run_target(candidate)
        disk = _disk_preflight(config, candidate)
        return {
            "schema": "compag-curation-public-dry-run/v1",
            "status": "PASS",
            "scientific_execution": False,
            "plan": build_public_plan(config, check_local=True),
            "bindings": asdict(bindings),
            "dependencies": preflight["dependencies"],
            "disk_preflight": disk,
        }, 0
    _require((output is None) != (resume is None), "run requires exactly one new output or resume root")
    _require(resume is not None or reviewed_labels is None, "review labels are accepted only on resume")
    selected_output = output if output is not None else resume
    _require(selected_output is not None, "run output is missing")
    _assert_output_disjoint(
        selected_output,
        directories=(config.images, config.inference_images, config.asset_root),
        files=tuple(path for path in (config.config_path, config.annotations, reviewed_labels) if path is not None),
    )
    if output is not None:
        preflight["disk_preflight"] = _disk_preflight(config, selected_output)
        state = claim_run(output, bindings)
    else:
        state = open_resume(resume, bindings, record_event=False)
    stage_dependency_guard = _gpu_stage_preflight_guard(
        config,
        bindings,
        preflight["dependencies"],
    )
    with state:
        stage_sixty_prevalidated = False
        if resume is not None:
            if completed_stage(state, "60_evaluate") is not None:
                _validate_published_stage_sixty_for_profile(config, state)
                stage_sixty_prevalidated = True
            existing_completion = _existing_completion(state)
            if existing_completion is not None:
                _require(reviewed_labels is None, "completed run does not accept another review table")
                return existing_completion, 0
            completed_names = tuple(name for name in STAGE_NAMES if completed_stage(state, name) is not None)
            preflight["disk_preflight"] = _disk_preflight(config, selected_output, completed_stages=completed_names)
            stage_twenty = completed_stage(state, "20_proposals_features")
            stage_thirty = completed_stage(state, "30_review_import")
            if stage_twenty is not None and stage_thirty is None and reviewed_labels is not None:
                expected_review_sha, _review_rows_count = _validated_review_source(
                    state.stages / "20_proposals_features/review_request.csv",
                    reviewed_labels,
                )
            elif stage_twenty is not None and stage_thirty is None:
                expected_review_sha = None
            else:
                _require(reviewed_labels is None, "review labels are not accepted after review import or before proposal publication")
                expected_review_sha = None
            state.append_event(
                "RUN_RESUMED",
                bindings=asdict(bindings),
                invocation=invocation_record,
                disk_preflight_sha256=compact_json_sha256(preflight["disk_preflight"]),
            )
        else:
            expected_review_sha = None
            state.append_event("RUN_INVOKED", invocation=invocation_record)

        _produce_stage(
            state,
            "00_input_inventory",
            (
                "config.toml",
                "normalized_config.json",
                "command_record.json",
                "input_inventory.json",
                "asset_inventory.json",
                "dependency_inventory.json",
                "disk_preflight.json",
            ),
            lambda target: _stage_zero(config, preflight, invocation_record, target),
            dependency_guard=stage_dependency_guard,
        )
        if _uses_v2_pipeline(config):
            from .canonical.service import (
                build_canonical_group_split as build_group_split,
                prepare_canonical_stage as prepare_images,
                train_canonical_stage as train_classifier,
            )
        else:
            from .public_backend import build_group_split, prepare_images, train_classifier

        _produce_stage(
            state,
            "10_prepare",
            ("tiles_index.csv", "prepare_summary.json"),
            lambda target: prepare_images(config, target),
            dependency_guard=stage_dependency_guard,
        )
        _produce_stage(
            state,
            "20_proposals_features",
            ("proposals.csv", "features.csv", "proposal_config.json", "proposal_summary.json", "review_request.csv"),
            lambda target: _stage_twenty(config, state.stages / "10_prepare", target),
            dependency_guard=stage_dependency_guard,
        )
        _assert_preflight(config, bindings, preflight["dependencies"])

        if completed_stage(state, "30_review_import") is None and reviewed_labels is None:
            review = state.stages / "20_proposals_features/review_request.csv"
            state.append_event("PAUSED_FOR_REVIEW", review_request_sha256=sha256_file(review))
            return {
                "schema": "compag-curation-public-run-pause/v1",
                "status": "PAUSED_FOR_REVIEW",
                "exit_code": PAUSED_FOR_REVIEW_EXIT_CODE,
                "run_id": state.run_id,
                "review_request": "stages/20_proposals_features/review_request.csv",
                "review_request_sha256": sha256_file(review),
                "resume_command": "python -I -B -m compag_curation run --config CONFIG --resume RUN_OUTPUT --review-labels REVIEWED_CSV",
            }, PAUSED_FOR_REVIEW_EXIT_CODE

        if completed_stage(state, "30_review_import") is None:
            _require(reviewed_labels is not None and expected_review_sha is not None, "review input was not validated")
            _produce_stage(
                state,
                "30_review_import",
                ("reviewed_supplied.csv", "reviewed.csv", "review_import.json"),
                lambda target: _stage_review(
                    state.stages / "20_proposals_features/review_request.csv",
                    reviewed_labels,
                    expected_review_sha,
                    target,
                ),
                dependency_guard=stage_dependency_guard,
            )
        _produce_stage(
            state,
            "40_group_split",
            ("split_manifest.json",),
            lambda target: (
                build_group_split(
                    state.stages / "30_review_import/reviewed.csv",
                    target,
                    profile=config.profile,
                )
                if _uses_v2_pipeline(config)
                else build_group_split(
                    state.stages / "30_review_import/reviewed.csv",
                    target,
                )
            ),
            dependency_guard=stage_dependency_guard,
        )
        _produce_stage(
            state,
            "50_train_bundle",
            ("test_predictions.csv", "training_result.json", "model_bundle/bundle.json", "model_bundle/classifier.ubj"),
            lambda target: (
                train_classifier(
                    config,
                    state.stages / "20_proposals_features/proposals.csv",
                    state.stages / "30_review_import/reviewed.csv",
                    state.stages / "40_group_split/split_manifest.json",
                    target,
                    features_path=state.stages / "20_proposals_features/features.csv",
                )
                if _uses_v2_pipeline(config)
                else train_classifier(
                    config,
                    state.stages / "20_proposals_features/proposals.csv",
                    state.stages / "30_review_import/reviewed.csv",
                    state.stages / "40_group_split/split_manifest.json",
                    target,
                )
            ),
            dependency_guard=stage_dependency_guard,
        )
        _produce_stage(
            state,
            "60_evaluate",
            _stage_sixty_required(config),
            lambda target: _stage_sixty(
                config,
                state.stages / "50_train_bundle/model_bundle",
                target,
                preflight["dependencies"],
            ),
            dependency_guard=stage_dependency_guard,
        )
        if not stage_sixty_prevalidated:
            _validate_published_stage_sixty_for_profile(config, state)
        _produce_stage(
            state,
            "70_report",
            ("report.json", "report.md"),
            lambda target: _stage_report(config, state, target),
            dependency_guard=stage_dependency_guard,
        )
        _assert_preflight(config, bindings, preflight["dependencies"])
        result = _finalize_run(state)
        return result, 0


def doctor(profile: str = "base") -> tuple[dict[str, object], int]:
    _require(profile in {"base", "quick", "science-cpu", "science-gpu"}, "unknown doctor profile")
    if profile == "science-cpu":
        return {
            "schema": "compag-curation-doctor/v1",
            "status": "UNSUPPORTED",
            "profile": profile,
            "reason": "the executable runtime is GPU-only; science-cpu is historical verification-only",
        }, 1
    from .public_backend import (
        EXPECTED_SCIENCE_GPU_VERSIONS,
        require_base_dependencies,
        require_quick_dependencies,
        require_science_dependencies,
    )

    checker = {
        "base": require_base_dependencies,
        "quick": require_quick_dependencies,
        "science-gpu": lambda: require_science_dependencies("cuda"),
    }[profile]
    try:
        checks = checker()
    except (ImportError, OSError, RuntimeError, UnicodeError, ValueError) as exc:
        required = {
            "base": ("compag-curation",),
            "quick": ("compag-curation", "numpy", "opencv-python"),
            "science-gpu": tuple(EXPECTED_SCIENCE_GPU_VERSIONS),
        }[profile]
        missing = []
        for name in required:
            try:
                importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                missing.append(name)
        return {
            "schema": "compag-curation-doctor/v1",
            "status": "FAIL",
            "profile": profile,
            "checks": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "missing": missing,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        }, 1
    return {"schema": "compag-curation-doctor/v1", "status": "PASS", "profile": profile, "checks": checks}, 0


def init_project(output: Path) -> tuple[dict[str, object], int]:
    return initialize_project(output), 0


def inspect_project(config_path: Path) -> tuple[dict[str, object], int]:
    config = load_public_config(config_path, check_local=False)
    return inspect_public_data(config), 0


def validate_project(config_path: Path) -> tuple[dict[str, object], int]:
    config = load_public_config(config_path, check_local=False)
    _require_gpu_executable_profile(config)
    bindings, preflight = _preflight(config)
    return {
        "schema": "compag-curation-public-validation/v1",
        "status": "PASS",
        "profile": config.profile,
        "bindings": asdict(bindings),
        "inspection": preflight["inspection"],
        "annotations": preflight["annotations"],
        "assets": preflight["assets"],
        "dependencies": preflight["dependencies"],
        "decode_validation": preflight["decode_validation"],
        "scientific_execution": False,
    }, 0


def plan_project(config_path: Path) -> tuple[dict[str, object], int]:
    config = load_public_config(config_path, check_local=False)
    return build_public_plan(config, check_local=False), 0


def fetch_assets(asset_root: Path, asset_ids: Sequence[str] | None = None) -> tuple[dict[str, object], int]:
    selected = tuple(asset_ids or sorted(asset_registry()))
    _require(len(selected) == len(set(selected)), "duplicate asset request")
    absolute = asset_root.absolute()
    parent = absolute.parent.resolve(strict=True)
    asset_root = parent / absolute.name
    _require(asset_root == absolute and asset_root.name not in {"", ".", ".."}, "asset root contains a symlinked or unsafe component")
    if not asset_root.exists() and not asset_root.is_symlink():
        asset_root.mkdir(mode=0o755)
        os.chmod(asset_root, 0o755, follow_symlinks=False)
        fsync_directory(parent)
    else:
        info = asset_root.lstat()
        _require(stat.S_ISDIR(info.st_mode) and not asset_root.is_symlink(), "asset root is unsafe")
    results = [fetch_asset(asset_root, asset_id) for asset_id in selected]
    return {"schema": "compag-curation-asset-acquisition/v1", "status": "PASS", "assets": results}, 0


def verify_assets(asset_root: Path, asset_ids: Sequence[str] | None = None) -> tuple[dict[str, object], int]:
    selected = tuple(asset_ids or sorted(asset_registry()))
    _require(len(selected) == len(set(selected)), "duplicate asset request")
    results = [verify_asset(asset_root, asset_id) for asset_id in selected]
    return {"schema": "compag-curation-asset-verification/v1", "status": "PASS", "assets": results}, 0


def convert_annotations(source: Path, output: Path) -> tuple[dict[str, object], int]:
    from .point_annotations import MAX_ANNOTATION_BYTES, CANONICAL_SCHEMA, inspect_cvat_points, inspect_point_annotations

    source = source.absolute()
    output = _future_path(output)
    _require(source.resolve(strict=True) != output, "annotation conversion output must differ from the input")
    _require(output.suffix == ".json", "annotation conversion output must use the .json suffix")
    portable_basename(source.name, "annotation source filename")
    rows, inspection = inspect_cvat_points(source)
    result = {
        "schema": CANONICAL_SCHEMA,
        "status": "PASS",
        "source_filename": source.name,
        "source_sha256": inspection["source_sha256"],
        "rows": rows,
    }
    payload = canonical_json_bytes(result)
    _require(len(payload) <= MAX_ANNOTATION_BYTES, "canonical annotation output exceeds the supported size bound")
    _require(not output.exists() and not output.is_symlink(), "annotation conversion output already exists")
    staging_name = f".{output.stem}.annotation-conversion.{uuid.uuid4()}.json"
    _require(len(staging_name.encode("ascii")) <= 255, "annotation output name is too long for staging")
    staging = output.parent / staging_name
    write_new_bytes(staging, payload)
    roundtrip_rows, roundtrip_summary = inspect_point_annotations(staging)
    _require(
        roundtrip_rows == rows and roundtrip_summary.get("canonical_source_sha256") == inspection.get("source_sha256"),
        "canonical annotation round trip failed",
    )
    publish_file_noreplace(staging, output)
    published_rows, published_summary = inspect_point_annotations(output)
    _require(
        published_rows == rows and published_summary.get("canonical_source_sha256") == inspection.get("source_sha256"),
        "published canonical annotation round trip failed",
    )
    return {"status": "PASS", "output_name": output.name, "sha256": sha256_file(output), "rows": len(rows)}, 0


def _input_images_identity(root: Path) -> list[dict[str, object]]:
    import cv2
    import numpy as np

    absolute = root.absolute()
    root_info = absolute.lstat()
    _require(
        stat.S_ISDIR(root_info.st_mode) and not absolute.is_symlink() and absolute == absolute.resolve(strict=True),
        "inference image root is unsafe or contains a symlink component",
    )
    root = absolute
    rows = []
    names: set[str] = set()
    stems: set[str] = set()
    digests: set[str] = set()
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".ppm"}:
            continue
        info = path.lstat()
        portable_basename(path.name, "inference image name")
        _require(stat.S_ISREG(info.st_mode) and not path.is_symlink() and info.st_nlink == 1, f"inference image is unsafe: {path.name}")
        _require(0 < info.st_size <= MAX_IMAGE_FILE_BYTES, f"inference image size is outside the supported bound: {path.name}")
        name_key = path.name.casefold()
        stem_key = path.stem.casefold()
        _require(name_key not in names and stem_key not in stems, f"inference image name/stem collision: {path.name}")
        names.add(name_key)
        stems.add(stem_key)
        _require(len(f"{path.stem}_y00000x00000.png".encode("utf-8")) <= 255, f"derived inference tile name exceeds NAME_MAX: {path.name}")
        payload, snapshot = stable_file(path, max_bytes=MAX_IMAGE_FILE_BYTES)
        _require(
            (snapshot.st_dev, snapshot.st_ino, snapshot.st_mode, snapshot.st_uid, snapshot.st_gid, snapshot.st_nlink, snapshot.st_size, snapshot.st_mtime_ns, snapshot.st_ctime_ns)
            == (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns),
            f"inference image changed while opening: {path.name}",
        )
        digest = hashlib.sha256(payload).hexdigest()
        _require(digest not in digests, f"duplicate inference image content: {path.name}")
        digests.add(digest)
        width, height, orientation = _image_info(path, payload)
        _require(width >= 64 and height >= 64 and orientation in {None, 1}, f"inference image dimensions/orientation are unsupported: {path.name}")
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        _require(decoded is not None and decoded.ndim == 3 and decoded.shape[2] == 3, f"image decoder rejected inference input: {path.name}")
        _require(int(decoded.shape[0]) >= 64 and int(decoded.shape[1]) >= 64, f"decoded inference image is smaller than 64x64: {path.name}")
        _require(
            (int(decoded.shape[1]), int(decoded.shape[0])) == (width, height),
            f"decoded inference dimensions differ from structural inspection: {path.name}",
        )
        rows.append({
            "filename": path.name,
            "sha256": digest,
            "size_bytes": snapshot.st_size,
            "width": width,
            "height": height,
            "exif_orientation": orientation,
            "mode_octal": f"{stat.S_IMODE(info.st_mode):04o}",
            "uid": info.st_uid,
            "gid": info.st_gid,
            "nlink": info.st_nlink,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
        })
    _require(rows, "inference image directory contains no supported images")
    root_after = root.lstat()
    _require(
        (root_after.st_dev, root_after.st_ino, root_after.st_mode, root_after.st_uid, root_after.st_gid, root_after.st_mtime_ns, root_after.st_ctime_ns)
        == (root_info.st_dev, root_info.st_ino, root_info.st_mode, root_info.st_uid, root_info.st_gid, root_info.st_mtime_ns, root_info.st_ctime_ns),
        "inference image directory changed during inspection",
    )
    return rows


def _revalidate_inference_dependencies(
    device: str,
    captured: Mapping[str, object],
) -> dict[str, object]:
    """Close a long-running inference against its captured dependency receipt."""

    if device == "cuda":
        from .public_backend import revalidate_science_gpu_dependencies

        observed = revalidate_science_gpu_dependencies(device, captured)
        _require(
            observed == dict(captured),
            "science-gpu inference dependencies changed before publication",
        )
        return observed
    _require(device == "cpu", "inference dependency revalidation device is unsupported")
    return dict(captured)


def infer_bundle(
    images: Path,
    model_bundle: Path,
    output: Path,
    *,
    device: str = "cuda",
    invocation: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], int]:
    from .model_bundle import BUNDLE_SCHEMA_V2, verify_model_bundle
    from .public_backend import require_science_dependencies

    invocation_record = dict(invocation or {"source": "PYTHON_API", "call": "infer_bundle"})
    if invocation_record.get("source") == "CLI":
        _require(
            set(invocation_record) == {"source", "command", "exact_argv_sha256"}
            and invocation_record.get("command") == "infer"
            and re.fullmatch(r"[0-9a-f]{64}", str(invocation_record.get("exact_argv_sha256", ""))) is not None,
            "inference CLI invocation fields mismatch",
        )
    else:
        _require(invocation_record == {"source": "PYTHON_API", "call": "infer_bundle"}, "inference Python invocation fields mismatch")

    output = _assert_output_disjoint(output, directories=(images, model_bundle), files=())
    parent = output.parent
    _require(not output.exists() and not output.is_symlink(), "inference output is not a fresh canonical path")
    lock = parent / f".{output.name}.compag-curation-infer.lock"
    _require(len(lock.name.encode("ascii")) <= 255, "inference output name is too long for its ownership lock")
    _require(len(f".{output.name}.inference.{uuid.uuid4()}".encode("ascii")) <= 255, "inference output name is too long for staging")
    _require(not lock.exists() and not lock.is_symlink(), "inference ownership lock already exists")
    verified = verify_model_bundle(model_bundle)
    _require(
        verified.schema == BUNDLE_SCHEMA_V2
        and verified.profile in GPU_EXECUTION_PROFILES
        and device == "cuda",
        "executable inference requires a supported Lite or Full GPU v2 bundle and device=cuda; historical CPU/balanced bundles are verification-only",
    )
    dependencies = require_science_dependencies(device)
    before_images = _input_images_identity(images)
    bundle_sha = verified.bundle_sha256
    lock_fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    run_id = str(uuid.uuid4())
    staging = parent / f".{output.name}.inference.{run_id}"
    acquired = False
    try:
        fsync_directory(parent)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        os.fchmod(lock_fd, 0o600)
        marker = (json.dumps({
            "schema": "compag-curation-inference-lock/v1",
            "run_id": run_id,
            "output_name": output.name,
            "parent_device": parent.stat().st_dev,
            "bundle_sha256": bundle_sha,
        }, sort_keys=True) + "\n").encode("ascii")
        _write_fd_all(lock_fd, marker, "inference ownership lock")
        os.fsync(lock_fd)
        staging.mkdir(mode=0o700)
        fsync_directory(parent)
        write_new_json(staging / "_ATTEMPT.json", {"schema": "compag-curation-inference-attempt/v1", "status": "WRITING", "run_id": run_id})
        write_new_json(staging / "command_record.json", {
            "schema": "compag-curation-inference-command/v1",
            "run_id": run_id,
            "command": "compag-curation infer --execute",
            "invocation": invocation_record,
            "device": device,
            "images_role": "READ_ONLY_INFERENCE_IMAGES",
            "bundle_role": "VERIFIED_MODEL_BUNDLE",
            "output_policy": "FRESH_NO_CLOBBER",
        })
        write_new_json(staging / "input_manifest.json", {
            "schema": "compag-curation-inference-inputs/v1",
            "run_id": run_id,
            "images": before_images,
            "images_sha256": compact_json_sha256(before_images),
        })
        write_new_json(staging / "environment.json", {
            "schema": "compag-curation-inference-environment/v1",
            "run_id": run_id,
            "dependencies": dependencies,
        })
        write_new_json(staging / "model_identity.json", {
            "schema": "compag-curation-inference-model/v1",
            "run_id": run_id,
            "bundle_sha256": bundle_sha,
            "profile": verified.normalized_config["profile"],
            "feature_order_sha256": verified.manifest["feature_order_sha256"],
            "compatibility": verified.compatibility,
        })
        if verified.schema == BUNDLE_SCHEMA_V2:
            from .canonical.service import infer_canonical_bundle

            result = infer_canonical_bundle(
                images,
                model_bundle,
                staging,
                device=device,
            )
            on_disk_result = _json(staging / "inference_result.json")
            expected_inputs = [
                {
                    key: row[key]
                    for key in ("filename", "sha256", "size_bytes", "width", "height", "exif_orientation")
                }
                for row in before_images
            ]
            validated_result = _validated_canonical_inference_result(
                on_disk_result,
                bundle_sha256=bundle_sha,
                expected_inputs=expected_inputs,
                expected_profile=verified.profile,
                expected_device=device,
            )
            _require(
                dict(result) == validated_result,
                "canonical inference return value differs from its result file",
            )
            _validated_canonical_predictions(
                staging / "predictions.csv",
                validated_result,
            )
            _validated_canonical_raw_feature_archive(staging, validated_result)
            result = validated_result
        else:
            from .public_backend import infer_from_bundle

            result = infer_from_bundle(images, model_bundle, staging, device=device)
        _require(_input_images_identity(images) == before_images, "inference image inputs changed during execution")
        verified_after = verify_model_bundle(model_bundle)
        _require(
            verified_after.bundle_sha256 == bundle_sha and verified_after.tree_identity == verified.tree_identity,
            "model bundle changed during inference",
        )
        _revalidate_inference_dependencies(device, dependencies)
        expected_status = {
            "schema": "compag-curation-inference-completion/v1",
            "status": "PASS",
            "run_id": run_id,
            "output_name": output.name,
            "publication_condition": "VALID_ONLY_AT_DECLARED_OUTPUT_NAME_AFTER_ATOMIC_RENAME",
            "bundle_sha256": bundle_sha,
            "inference_result_sha256": sha256_file(staging / "inference_result.json"),
        }
        write_new_json(staging / "FINAL_STATUS.json", expected_status)
        rows = manifest_rows(staging, excluded={"OUTPUT_MANIFEST.json"})
        expected_manifest = {
            "schema": "compag-curation-inference-output-manifest/v1",
            "status": "PASS",
            "run_id": run_id,
            "bundle_sha256": bundle_sha,
            "input_images": before_images,
            "members": rows,
            "members_sha256": compact_json_sha256(rows),
        }
        write_new_json(staging / "OUTPUT_MANIFEST.json", expected_manifest)
        output_manifest_sha256 = sha256_file(staging / "OUTPUT_MANIFEST.json")
        publish_directory_noreplace(staging, output)
        final_status = _json(output / "FINAL_STATUS.json")
        final_manifest = _json(output / "OUTPUT_MANIFEST.json")
        _require(
            final_status == expected_status,
            "published inference completion condition failed",
        )
        observed = manifest_rows(output, excluded={"OUTPUT_MANIFEST.json"})
        _require(
            observed == rows
            and final_manifest == expected_manifest
            and sha256_file(output / "OUTPUT_MANIFEST.json") == output_manifest_sha256
            and sha256_file(output / "inference_result.json") == expected_status["inference_result_sha256"],
            "published inference manifest verification failed",
        )
        return {**result, "run_id": run_id, "output_name": output.name, "output_manifest_sha256": output_manifest_sha256}, 0
    except BaseException as exc:
        _record_inference_failure(staging, run_id, exc)
        raise
    finally:
        if acquired:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def run_demo(
    profile: str,
    output: Path,
    asset_root: Path | None = None,
    *,
    execution_profile: str = CANONICAL_GPU_PROFILE,
    invocation: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], int]:
    from .quick_demo import run_quick_demo, run_real_demo

    if profile == "quick":
        return run_quick_demo(output, invocation=invocation), 0
    _require(profile == "real" and asset_root is not None, "real demo requires --asset-root")
    return run_real_demo(
        output,
        asset_root,
        execution_profile=execution_profile,
        invocation=invocation,
    ), 0
