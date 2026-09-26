"""No-clobber operational facade for canonical active learning."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import stat
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from compag_curation.model_bundle import (
    BUNDLE_SCHEMA_V2,
    BUNDLE_V2_ASSET_PROFILES,
    POST_R92_TRANSFER_FROZEN_STATE_POLICY,
    POST_R92_TRANSFER_LINEAGE_POLICY,
    POST_R92_TRANSFER_REPRODUCTION_CLAIM,
    POST_R92_TRANSFER_WORKFLOW,
    verify_model_bundle,
)
from compag_curation.public_backend import (
    require_science_dependencies,
    revalidate_science_gpu_dependencies,
)
from compag_curation.public_io import (
    PublicIOError,
    canonical_json_bytes,
    canonical_json_value,
    compact_json_sha256,
    fsync_directory,
    manifest_rows,
    portable_basename,
    publish_directory_noreplace,
    stable_file,
    write_new_bytes,
    write_new_json,
)
from compag_curation.review import validate_review_table
from compag_curation.runtime_lock import package_implementation_identity
from compag_curation.run_state import STAGE_REQUIRED_FILES, _validate_stage

from .active_learning import (
    CANONICAL_AL_DECISION_LOG_COLUMNS,
    CanonicalTransferBaselineReference,
    CanonicalTransferScorerReference,
    TransferRetrainFunction,
    _canonical_device_for_profile,
    begin_canonical_active_learning_round,
    begin_canonical_transfer_image_round,
    complete_initial_labeling_export_all,
    read_canonical_accumulated_feature_archive,
    resume_canonical_active_learning_round,
    resume_canonical_transfer_image_round,
    write_canonical_active_learning_pool,
    write_initial_labeling_export_all,
)
from .active_learning_service import (
    read_canonical_active_learning_inference,
    read_canonical_r92_transfer_inference,
    read_canonical_stage20_raw_features,
    retrain_canonical_active_learning_bundle,
    retrain_canonical_transfer_bundle,
    write_initial_canonical_active_learning_pool,
)
from .serialization import (
    CanonicalFeatureStatePayloads,
    deserialize_canonical_feature_state,
    serialize_canonical_feature_state,
)
from .spec import (
    CANONICAL_EMBEDDING_DIMENSIONS,
    CANONICAL_FEATURE_CROP_SCALES,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_GPU_PROFILE,
    FULL_IMAGE_GPU_PROFILE,
    GPU_EXECUTION_PROFILES,
    V2_PIPELINE_PROFILES,
    amg_settings_for_profile,
    stage20_amg_execution_points_per_batch,
)
from .transfer_baseline import (
    EXPECTED_R92_TRANSFER_SOURCE_SHA256,
    EXPECTED_R92_TRANSFER_SOURCE_SIZE_BYTES,
    verify_transfer_baseline,
)


_COMMAND_SCHEMA = "compag-curation-canonical-active-learning-command/v1"
_INPUT_SCHEMA = "compag-curation-canonical-active-learning-inputs/v1"
_ENVIRONMENT_SCHEMA = "compag-curation-canonical-active-learning-environment/v1"
_ATTEMPT_SCHEMA = "compag-curation-canonical-active-learning-attempt/v1"
_STATUS_SCHEMA = "compag-curation-canonical-active-learning-operation/v1"
_MANIFEST_SCHEMA = "compag-curation-canonical-active-learning-output-manifest/v1"
_RESULT_SCHEMA = "compag-curation-canonical-active-learning-result/v1"
_FAILURE_SCHEMA = "compag-curation-canonical-active-learning-failure/v1"
_PUBLICATION_CONDITION = "VALID_ONLY_AT_DECLARED_OUTPUT_NAME_AFTER_ATOMIC_RENAME"
_UTC = re.compile(
    r"(?:19|20)[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_TREE_BYTES = 16 * 1024 * 1024 * 1024


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


def _strict_utc(value: object, role: str) -> str:
    _require(isinstance(value, str) and _UTC.fullmatch(value) is not None, f"{role} is not strict UTC")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise PublicIOError(f"{role} is not a valid UTC timestamp") from exc
    return value


def _utc_now() -> str:
    return _strict_utc(
        datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "operation time",
    )


def _round_number(value: object) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 1,
        "active-learning round number must be a positive integer",
    )
    return value


def _safe_directory(path: Path, role: str) -> Path:
    absolute = path.absolute()
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError(f"{role} cannot be resolved") from exc
    info = absolute.lstat()
    _require(
        absolute == resolved
        and stat.S_ISDIR(info.st_mode)
        and not absolute.is_symlink(),
        f"{role} is not a safe directory",
    )
    return absolute


def _safe_file(path: Path, role: str, *, max_bytes: int = _MAX_FILE_BYTES) -> tuple[Path, bytes]:
    absolute = path.absolute()
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError(f"{role} cannot be resolved") from exc
    _require(absolute == resolved, f"{role} contains a symlinked path component")
    payload, info = stable_file(absolute, max_bytes=max_bytes)
    _require(
        stat.S_ISREG(info.st_mode)
        and not absolute.is_symlink()
        and info.st_nlink == 1
        and 0 < info.st_size <= max_bytes,
        f"{role} must be a bounded single-link regular file",
    )
    return absolute, payload


def _file_record(path: Path, role: str, *, max_bytes: int = _MAX_FILE_BYTES) -> dict[str, object]:
    absolute, payload = _safe_file(path, role, max_bytes=max_bytes)
    return {
        "role": role,
        "basename": portable_basename(absolute.name, f"{role} basename"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _directory_record(path: Path, role: str) -> dict[str, object]:
    absolute = _safe_directory(path, role)
    portable_basename(absolute.name, f"{role} basename")
    rows = manifest_rows(absolute, max_total_bytes=_MAX_TREE_BYTES)
    _require(bool(rows), f"{role} is empty")
    return {
        "role": role,
        "basename": absolute.name,
        "sha256": compact_json_sha256(rows),
        "size_bytes": sum(int(row["size_bytes"]) for row in rows),
    }


def _validate_record(value: object, role: str | None = None) -> dict[str, object]:
    _require(
        isinstance(value, dict)
        and set(value) == {"role", "basename", "sha256", "size_bytes"}
        and isinstance(value.get("role"), str)
        and (role is None or value.get("role") == role)
        and isinstance(value.get("sha256"), str)
        and _SHA256.fullmatch(str(value.get("sha256"))) is not None
        and isinstance(value.get("size_bytes"), int)
        and not isinstance(value.get("size_bytes"), bool)
        and int(value["size_bytes"]) > 0,
        "operation artifact record is invalid",
    )
    portable_basename(str(value["basename"]), "operation artifact basename")
    return dict(value)


@dataclass(frozen=True)
class _InputSpec:
    role: str
    path: Path
    directory: bool
    max_bytes: int = _MAX_FILE_BYTES


def _capture_inputs(specs: Sequence[_InputSpec]) -> tuple[dict[str, object], ...]:
    _require(len({spec.role for spec in specs}) == len(specs), "operation input roles are duplicated")
    return tuple(
        _directory_record(spec.path, spec.role)
        if spec.directory
        else _file_record(spec.path, spec.role, max_bytes=spec.max_bytes)
        for spec in specs
    )


def _environment_artifacts(
    science_dependencies: Mapping[str, object] | None = None,
    *,
    science_mode: str = "cpu",
) -> tuple[dict[str, object], ...]:
    identity = package_implementation_identity(require_installed=False)
    artifacts: tuple[dict[str, object], ...] = (
        {
            "role": "package_implementation",
            "basename": "compag_curation",
            "sha256": identity["sha256"],
            "size_bytes": identity["size_bytes"],
        },
    )
    if science_dependencies is None:
        return artifacts
    _require(
        science_mode in {"cpu", "gpu"},
        "science dependency mode is invalid",
    )
    dependency_sha256 = science_dependencies.get("sha256")
    _require(
        isinstance(dependency_sha256, str)
        and _SHA256.fullmatch(dependency_sha256) is not None,
        "science dependency identity is invalid",
    )
    dependency_payload = canonical_json_bytes(dict(science_dependencies))
    return (
        *artifacts,
        {
            "role": "science_dependency_identity",
            "basename": f"science-{science_mode}",
            "sha256": dependency_sha256,
            "size_bytes": len(dependency_payload),
        },
    )


def _science_device_for_mode(science_mode: str) -> str:
    _require(science_mode in {"cpu", "gpu"}, "science dependency mode is invalid")
    return "cuda" if science_mode == "gpu" else "cpu"


def _safe_output(output: Path) -> Path:
    absolute = output.absolute()
    portable_basename(absolute.name, "active-learning operation output basename")
    try:
        parent = absolute.parent.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError("active-learning operation output parent cannot be resolved") from exc
    info = absolute.parent.lstat()
    _require(
        parent == absolute.parent
        and stat.S_ISDIR(info.st_mode)
        and not absolute.parent.is_symlink()
        and info.st_uid == os.geteuid()
        and not absolute.exists()
        and not absolute.is_symlink(),
        "active-learning operation output is unsafe or already exists",
    )
    return absolute


def _assert_disjoint(output: Path, specs: Sequence[_InputSpec]) -> None:
    for spec in specs:
        source = spec.path.absolute().resolve(strict=True)
        if spec.directory:
            _require(
                not output.is_relative_to(source) and not source.is_relative_to(output),
                f"active-learning output overlaps {spec.role}",
            )
        else:
            _require(
                source != output and not source.is_relative_to(output),
                f"active-learning output overlaps {spec.role}",
            )


@dataclass(frozen=True)
class _Attempt:
    operation: str
    operation_id: str
    output: Path
    staging: Path
    started_at_utc: str
    specs: tuple[_InputSpec, ...]
    inputs: tuple[dict[str, object], ...]
    environment: tuple[dict[str, object], ...]
    science_dependencies: Mapping[str, object] | None
    science_mode: str | None


def _dependencies_for_sealing(
    attempt: _Attempt,
) -> Mapping[str, object] | None:
    if attempt.science_dependencies is None:
        return None
    _require(
        attempt.science_mode == "gpu",
        "v1.7 active-learning compute operations require science-gpu",
    )
    device = _science_device_for_mode(attempt.science_mode)
    current = revalidate_science_gpu_dependencies(
        device,
        attempt.science_dependencies,
    )
    _require(
        dict(current) == dict(attempt.science_dependencies),
        "science-gpu dependency identity changed during operation sealing",
    )
    return current


def _record_failure(
    attempt: _Attempt,
    error: BaseException,
    *,
    allow_unmarked: bool = False,
) -> None:
    staging = attempt.staging
    try:
        created = False
        if not staging.exists() and not staging.is_symlink():
            staging.mkdir(mode=0o700)
            fsync_directory(staging.parent)
            created = True
        if not staging.is_dir() or staging.is_symlink():
            return
        marker = staging / "_ATTEMPT.json"
        if not allow_unmarked and not created:
            if not marker.is_file() or marker.is_symlink():
                return
            marker_value = _json_object(marker, "active-learning attempt marker")
            if marker_value.get("operation_id") != attempt.operation_id:
                return
        failure = staging / "_FAILURE.json"
        if not failure.exists() and not failure.is_symlink():
            write_new_json(
                failure,
                {
                    "schema": _FAILURE_SCHEMA,
                    "status": "FAILED",
                    "operation": attempt.operation,
                    "operation_id": attempt.operation_id,
                    "failed_at_utc": _utc_now(),
                    "error_type": type(error).__name__,
                },
            )
    except BaseException:
        return


def _start_operation(
    operation: str,
    output: Path,
    specs: Sequence[_InputSpec],
    *,
    science_dependencies: Mapping[str, object] | None = None,
    science_mode: str | None = None,
) -> _Attempt:
    _require(
        operation
        in {
            "initial_export",
            "initial_resume",
            "begin_round",
            "resume_round",
            "begin_image_round",
            "resume_image_round",
            "begin_transfer_image_round",
            "resume_transfer_image_round",
        },
        "operation is invalid",
    )
    final = _safe_output(output)
    fixed_specs = tuple(specs)
    _assert_disjoint(final, fixed_specs)
    inputs = _capture_inputs(fixed_specs)
    resolved_science_mode = (
        "cpu" if science_dependencies is not None and science_mode is None else science_mode
    )
    _require(
        (science_dependencies is None and resolved_science_mode is None)
        or (
            science_dependencies is not None
            and resolved_science_mode in {"cpu", "gpu"}
        ),
        "science dependency identity and mode must be supplied together",
    )
    environment = _environment_artifacts(
        science_dependencies,
        science_mode=resolved_science_mode or "cpu",
    )
    operation_id = str(uuid.uuid4())
    started = _utc_now()
    name = f".{final.name}.canonical-active-learning.{operation_id}"
    _require(len(name.encode("ascii")) <= 255, "active-learning output name is too long for staging")
    staging = final.parent / name
    attempt = _Attempt(
        operation,
        operation_id,
        final,
        staging,
        started,
        fixed_specs,
        inputs,
        environment,
        science_dependencies,
        resolved_science_mode,
    )
    claimed = False
    try:
        staging.mkdir(mode=0o700)
        claimed = True
        os.chmod(staging, 0o700, follow_symlinks=False)
        fsync_directory(staging.parent)
        write_new_json(
            staging / "_ATTEMPT.json",
            {"schema": _ATTEMPT_SCHEMA, "status": "WRITING", "operation": operation, "operation_id": operation_id},
        )
        write_new_json(
            staging / "command_record.json",
            {
                "schema": _COMMAND_SCHEMA,
                "operation": operation,
                "operation_id": operation_id,
                "started_at_utc": started,
                "output_basename": final.name,
                "output_policy": "FRESH_SIBLING_STAGING_ATOMIC_NOREPLACE",
                "inputs": list(inputs),
            },
        )
        write_new_json(
            staging / "input_manifest.json",
            {"schema": _INPUT_SCHEMA, "operation_id": operation_id, "inputs": list(inputs)},
        )
        write_new_json(
            staging / "environment.json",
            {"schema": _ENVIRONMENT_SCHEMA, "operation_id": operation_id, "artifacts": list(environment)},
        )
        return attempt
    except BaseException as exc:
        if claimed:
            _record_failure(attempt, exc, allow_unmarked=True)
        raise


def _json_object(path: Path, role: str) -> Mapping[str, object]:
    payload, _info = stable_file(path, max_bytes=128 * 1024 * 1024)
    value = canonical_json_value(payload, role)
    _require(isinstance(value, dict), f"{role} is not an object")
    return value


def _verify_published_operation(
    root: Path,
    *,
    operation: str | None = None,
    status: str | None = None,
    exit_code: int | None = None,
    round_number: int | None | object = ...,
    expected_environment: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    root = _safe_directory(root, "active-learning operation root")
    portable_basename(root.name, "active-learning operation root basename")
    final_status = _json_object(root / "FINAL_STATUS.json", "active-learning final status")
    _require(
        set(final_status)
        == {
            "schema", "status", "operation", "operation_id", "round_number",
            "started_at_utc", "completed_at_utc", "exit_code", "output_name",
            "payload", "publication_condition",
        }
        and final_status.get("schema") == _STATUS_SCHEMA
        and final_status.get("status") in {"PAUSED_FOR_REVIEW", "PASS"}
        and final_status.get("operation")
        in {
            "initial_export",
            "initial_resume",
            "begin_round",
            "resume_round",
            "begin_image_round",
            "resume_image_round",
            "begin_transfer_image_round",
            "resume_transfer_image_round",
        }
        and isinstance(final_status.get("operation_id"), str)
        and final_status.get("output_name") == root.name
        and final_status.get("publication_condition") == _PUBLICATION_CONDITION,
        "active-learning final status identity changed",
    )
    uuid.UUID(str(final_status["operation_id"]))
    started = _strict_utc(final_status["started_at_utc"], "operation start")
    completed = _strict_utc(final_status["completed_at_utc"], "operation completion")
    _require(completed >= started, "operation completion precedes its start")
    _require(
        (final_status["status"] == "PAUSED_FOR_REVIEW" and final_status["exit_code"] == 3)
        or (final_status["status"] == "PASS" and final_status["exit_code"] == 0),
        "active-learning status and exit code disagree",
    )
    payload_record = _directory_record(root / "payload", "operation_payload")
    _require(final_status.get("payload") == payload_record, "active-learning payload receipt changed")

    manifest_payload, _info = stable_file(
        root / "OUTPUT_MANIFEST.json",
        max_bytes=128 * 1024 * 1024,
    )
    manifest = canonical_json_value(manifest_payload, "active-learning output manifest")
    rows = manifest_rows(root, excluded={"OUTPUT_MANIFEST.json"}, max_total_bytes=_MAX_TREE_BYTES)
    _require(
        isinstance(manifest, dict)
        and set(manifest)
        == {
            "schema", "status", "operation", "operation_id", "exit_code",
            "members", "members_sha256",
        }
        and manifest.get("schema") == _MANIFEST_SCHEMA
        and manifest.get("status") == final_status["status"]
        and manifest.get("operation") == final_status["operation"]
        and manifest.get("operation_id") == final_status["operation_id"]
        and manifest.get("exit_code") == final_status["exit_code"]
        and manifest.get("members") == rows
        and manifest.get("members_sha256") == compact_json_sha256(rows),
        "active-learning published manifest closure changed",
    )
    command = _json_object(root / "command_record.json", "active-learning command record")
    inputs = _json_object(root / "input_manifest.json", "active-learning input manifest")
    environment = _json_object(root / "environment.json", "active-learning environment record")
    expected_inputs = command.get("inputs")
    _require(
        set(command)
        == {
            "schema", "operation", "operation_id", "started_at_utc",
            "output_basename", "output_policy", "inputs",
        }
        and command.get("schema") == _COMMAND_SCHEMA
        and command.get("operation") == final_status["operation"]
        and command.get("operation_id") == final_status["operation_id"]
        and command.get("started_at_utc") == started
        and command.get("output_basename") == root.name
        and command.get("output_policy") == "FRESH_SIBLING_STAGING_ATOMIC_NOREPLACE"
        and isinstance(expected_inputs, list)
        and set(inputs) == {"schema", "operation_id", "inputs"}
        and inputs.get("schema") == _INPUT_SCHEMA
        and inputs.get("operation_id") == final_status["operation_id"]
        and inputs.get("inputs") == expected_inputs
        and set(environment) == {"schema", "operation_id", "artifacts"}
        and environment.get("schema") == _ENVIRONMENT_SCHEMA
        and environment.get("operation_id") == final_status["operation_id"]
        and isinstance(environment.get("artifacts"), list),
        "active-learning command, input, or environment record changed",
    )
    parsed_inputs = tuple(_validate_record(item) for item in expected_inputs)
    parsed_environment = tuple(_validate_record(item) for item in environment["artifacts"])
    environment_roles = tuple(str(item["role"]) for item in parsed_environment)
    if expected_environment is None:
        if environment_roles == ("package_implementation",):
            current_environment = _environment_artifacts()
        elif environment_roles == (
            "package_implementation",
            "science_dependency_identity",
        ):
            dependency_basename = str(parsed_environment[-1]["basename"])
            _require(
                dependency_basename in {"science-cpu", "science-gpu"},
                "active-learning science dependency basename is invalid",
            )
            science_mode = dependency_basename.removeprefix("science-")
            if science_mode == "gpu":
                # A sealed GPU operation records the preflight identity that
                # preceded CUDA initialization.  Re-running that preflight
                # after a context exists is intentionally forbidden; the next
                # compute operation performs its own fresh preflight.
                current_environment = (
                    *_environment_artifacts(),
                    parsed_environment[-1],
                )
            else:
                # Historical CPU operations are immutable evidence, not a
                # request to recreate the retired science-cpu execution
                # environment.  Validate their sealed record structurally
                # without invoking a live CPU preflight or comparing it with
                # the current package closure.
                current_environment = parsed_environment
        else:
            raise PublicIOError("active-learning environment roles are invalid")
    else:
        current_environment = tuple(
            _validate_record(dict(item)) for item in expected_environment
        )
    _require(
        len({str(item["role"]) for item in parsed_inputs}) == len(parsed_inputs)
        and parsed_environment == current_environment,
        "active-learning evidence roles are invalid",
    )
    if operation is not None:
        _require(final_status["operation"] == operation, "active-learning operation kind mismatch")
    if status is not None:
        _require(final_status["status"] == status, "active-learning operation status mismatch")
    if exit_code is not None:
        _require(final_status["exit_code"] == exit_code, "active-learning operation exit code mismatch")
    if round_number is not ...:
        _require(final_status["round_number"] == round_number, "active-learning operation round mismatch")
    return {
        "root": root,
        "status": dict(final_status),
        "inputs": parsed_inputs,
        "output_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }


def _finish_operation(
    attempt: _Attempt,
    *,
    status: str,
    exit_code: int,
    round_number: int | None,
) -> tuple[dict[str, object], int]:
    _require(
        (status, exit_code) in {("PAUSED_FOR_REVIEW", 3), ("PASS", 0)},
        "operation completion status is invalid",
    )
    payload = _safe_directory(attempt.staging / "payload", "operation payload")
    _require(
        _capture_inputs(attempt.specs) == attempt.inputs,
        "active-learning input changed during operation",
    )
    current_science_dependencies = _dependencies_for_sealing(attempt)
    _require(
        _environment_artifacts(
            current_science_dependencies,
            science_mode=attempt.science_mode or "cpu",
        )
        == attempt.environment,
        "active-learning implementation or dependency identity changed during operation",
    )
    payload_record = _directory_record(payload, "operation_payload")
    final_status = {
        "schema": _STATUS_SCHEMA,
        "status": status,
        "operation": attempt.operation,
        "operation_id": attempt.operation_id,
        "round_number": round_number,
        "started_at_utc": attempt.started_at_utc,
        "completed_at_utc": _utc_now(),
        "exit_code": exit_code,
        "output_name": attempt.output.name,
        "payload": payload_record,
        "publication_condition": _PUBLICATION_CONDITION,
    }
    status_payload = canonical_json_bytes(final_status)
    rows = manifest_rows(attempt.staging, excluded={"OUTPUT_MANIFEST.json"}, max_total_bytes=_MAX_TREE_BYTES)
    _require(all(row["path"] != "FINAL_STATUS.json" for row in rows), "final status exists before operation sealing")
    rows.append(
        {
            "path": "FINAL_STATUS.json",
            "sha256": hashlib.sha256(status_payload).hexdigest(),
            "size_bytes": len(status_payload),
            "mode_octal": "0644",
        }
    )
    rows.sort(key=lambda row: str(row["path"]))
    manifest = {
        "schema": _MANIFEST_SCHEMA,
        "status": status,
        "operation": attempt.operation,
        "operation_id": attempt.operation_id,
        "exit_code": exit_code,
        "members": rows,
        "members_sha256": compact_json_sha256(rows),
    }
    # The manifest is written first so a PASS marker can never precede all outputs.
    write_new_json(attempt.staging / "OUTPUT_MANIFEST.json", manifest)
    write_new_bytes(attempt.staging / "FINAL_STATUS.json", status_payload)
    observed = manifest_rows(attempt.staging, excluded={"OUTPUT_MANIFEST.json"}, max_total_bytes=_MAX_TREE_BYTES)
    _require(observed == rows, "active-learning staging closure changed before publication")
    manifest_sha256 = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    fsync_directory(attempt.staging)
    publish_directory_noreplace(attempt.staging, attempt.output)
    verified = _verify_published_operation(
        attempt.output,
        operation=attempt.operation,
        status=status,
        exit_code=exit_code,
        round_number=round_number,
        expected_environment=attempt.environment,
    )
    _require(
        verified["output_manifest_sha256"] == manifest_sha256,
        "active-learning manifest hash changed after publication",
    )
    return (
        {
            "schema": _RESULT_SCHEMA,
            "status": status,
            "operation": attempt.operation,
            "operation_id": attempt.operation_id,
            "round_number": round_number,
            "exit_code": exit_code,
            "output_name": attempt.output.name,
            "payload_sha256": payload_record["sha256"],
            "output_manifest_sha256": manifest_sha256,
        },
        exit_code,
    )


def _payload_root(attempt: _Attempt) -> Path:
    payload = attempt.staging / "payload"
    payload.mkdir(mode=0o755)
    fsync_directory(attempt.staging)
    return payload


def _input_by_role(operation: Mapping[str, object], role: str) -> dict[str, object]:
    matches = [item for item in operation["inputs"] if item["role"] == role]
    _require(len(matches) == 1, f"sealed operation omits the {role} input")
    return dict(matches[0])


def _ancestry_root(root: Path, number: int) -> tuple[dict[str, object], Path, Path]:
    if number == 1:
        operation = _verify_published_operation(
            root,
            operation="initial_resume",
            status="PASS",
            exit_code=0,
            round_number=None,
        )
        completion = Path(operation["root"]) / "payload/initial_completion/initial_completion.json"
        decision_log = Path(operation["root"]) / "payload/initial_completion/decision_log.csv"
    else:
        operation = _verify_published_operation(
            root,
            operation="resume_round",
            status="PASS",
            exit_code=0,
            round_number=number - 1,
        )
        completion = Path(operation["root"]) / "payload/round_completion/round_completion.json"
        decision_log = Path(operation["root"]) / "payload/round_completion/decision_log.csv"
    _safe_file(completion, "active-learning ancestry completion", max_bytes=16 * 1024 * 1024)
    _safe_file(decision_log, "active-learning ancestry decision log")
    return operation, completion, decision_log


def _image_ancestry_root(
    root: Path,
    number: int,
) -> tuple[dict[str, object], Path, Path]:
    if number == 1:
        return _ancestry_root(root, number)
    operation = _verify_published_operation(
        root,
        operation="resume_image_round",
        status="PASS",
        exit_code=0,
        round_number=number - 1,
    )
    operation_root = Path(operation["root"])
    completion = (
        operation_root
        / "payload/round_completion/round_completion.json"
    )
    decision_log = operation_root / "payload/round_completion/decision_log.csv"
    _safe_file(
        completion,
        "sequential image active-learning ancestry completion",
        max_bytes=16 * 1024 * 1024,
    )
    _safe_file(
        decision_log,
        "sequential image active-learning ancestry decision log",
    )
    return operation, completion, decision_log


def _transfer_baseline_reference(
    root: Path,
) -> tuple[object, CanonicalTransferBaselineReference]:
    """Verify the one recovered baseline authorized for the transfer workflow."""

    baseline = verify_transfer_baseline(root)
    feature_contract = baseline.manifest.get("feature_contract")
    claims = baseline.manifest.get("claims")
    _require(
        baseline.source_sha256 == EXPECTED_R92_TRANSFER_SOURCE_SHA256
        and baseline.source_size_bytes == EXPECTED_R92_TRANSFER_SOURCE_SIZE_BYTES
        and isinstance(feature_contract, Mapping)
        and feature_contract.get("feature_order_sha256")
        == CANONICAL_FEATURE_ORDER_SHA256
        and isinstance(claims, Mapping)
        and claims.get("recovered_r92_source_sha256_and_size_match") is True,
        "transfer rounds require the strict recovered post-r92 feature snapshot",
    )
    archive = _directory_record(baseline.root, "transfer_baseline")
    split = _file_record(
        baseline.split_path,
        "transfer_baseline_split",
        max_bytes=16 * 1024 * 1024,
    )
    reference = CanonicalTransferBaselineReference(
        archive_sha256=str(archive["sha256"]),
        source_sha256=baseline.source_sha256,
        source_size_bytes=baseline.source_size_bytes,
        split_sha256=str(split["sha256"]),
        source_groups_sha256=baseline.source_groups_sha256,
        feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
        row_count=baseline.row_count,
        proposal_count=baseline.proposal_count,
        source_groups=baseline.source_groups,
        train_groups=baseline.train_groups,
        test_groups=baseline.test_groups,
    )
    return baseline, reference


def _transfer_ancestry_root(
    root: Path,
    number: int,
) -> tuple[dict[str, object], Path, Path, Path]:
    _require(number > 1, "transfer ancestry is only defined after round 1")
    operation = _verify_published_operation(
        root,
        operation="resume_transfer_image_round",
        status="PASS",
        exit_code=0,
        round_number=number - 1,
    )
    operation_root = Path(operation["root"])
    completion_root = operation_root / "payload/round_completion"
    completion = completion_root / "round_completion.json"
    decision_log = completion_root / "decision_log.csv"
    model_bundle = completion_root / "model_bundle"
    _safe_file(
        completion,
        "transfer active-learning ancestry completion",
        max_bytes=16 * 1024 * 1024,
    )
    _safe_file(
        decision_log,
        "transfer active-learning ancestry decision log",
    )
    _safe_directory(model_bundle, "transfer active-learning ancestry bundle")
    return operation, completion, decision_log, model_bundle


def _transfer_round_context(
    stage60_root: Path,
    baseline: CanonicalTransferBaselineReference,
    number: int,
    *,
    source_bundle: Path | None,
    decision_log: Path | None,
    prior_completion: Path | None,
) -> tuple[object, CanonicalTransferScorerReference, object, dict[str, object] | None, Path | None, Path | None]:
    """Resolve the published-r92 first scorer or a verified project-rN scorer."""

    if number == 1:
        _require(
            source_bundle is None
            and decision_log is None
            and prior_completion is None,
            "transfer round 1 cannot name a project bundle or prior ancestry",
        )
        transfer = read_canonical_r92_transfer_inference(stage60_root)
        _verify_single_image_inference_inventory(stage60_root, transfer.inference)
        return (
            transfer.inference,
            transfer.scorer,
            transfer.feature_state,
            None,
            None,
            None,
        )

    _require(
        source_bundle is not None
        and decision_log is not None
        and prior_completion is not None,
        "later transfer rounds require source bundle, decision log, and prior completion",
    )
    ancestry, completion_path, ancestry_log, ancestry_bundle = (
        _transfer_ancestry_root(prior_completion, number)
    )
    _require(
        _file_record(ancestry_log, "decision_log")
        == _file_record(decision_log, "decision_log"),
        "supplied transfer decision log differs from prior ancestry",
    )
    archive = read_canonical_active_learning_inference(stage60_root)
    _verify_single_image_inference_inventory(stage60_root, archive)
    bundle = _verified_gpu_execution_bundle(source_bundle)
    ancestry_verified_bundle = _verified_gpu_execution_bundle(ancestry_bundle)
    provenance = getattr(bundle, "provenance", None)
    details = provenance.get("details") if isinstance(provenance, Mapping) else None
    completion = _json_object(
        completion_path,
        "prior transfer active-learning completion",
    )
    outputs = completion.get("outputs")
    from compag_curation.review.published_model import (
        load_published_model_transfer_assets,
    )

    assets = load_published_model_transfer_assets()
    published_payloads = serialize_canonical_feature_state(assets.feature_state)
    prototype_sha256 = hashlib.sha256(published_payloads.prototype_npy).hexdigest()
    pca_components_sha256 = hashlib.sha256(
        published_payloads.pca_components_npy
    ).hexdigest()
    pca_mean_sha256 = hashlib.sha256(published_payloads.pca_mean_npy).hexdigest()
    _require(
        archive.profile == CANONICAL_GPU_PROFILE
        and archive.device == "cuda"
        and bundle.profile == CANONICAL_GPU_PROFILE
        and archive.bundle_sha256 == bundle.bundle_sha256
        and ancestry_verified_bundle.bundle_sha256 == bundle.bundle_sha256
        and isinstance(details, Mapping)
        and details.get("active_learning_workflow") == POST_R92_TRANSFER_WORKFLOW
        and details.get("lineage_policy") == POST_R92_TRANSFER_LINEAGE_POLICY
        and details.get("reproduction_claim")
        == POST_R92_TRANSFER_REPRODUCTION_CLAIM
        and details.get("frozen_feature_state_policy")
        == POST_R92_TRANSFER_FROZEN_STATE_POLICY
        and details.get("active_learning_round_number") == number - 1
        and details.get("model_label") == f"project-r{number - 1}"
        and details.get("transfer_baseline_sha256") == baseline.archive_sha256
        and details.get("transfer_split_sha256") == baseline.split_sha256
        and details.get("r92_resource_manifest_sha256")
        == assets.resource_manifest_sha256
        and details.get("r92_classifier_sha256") == assets.classifier_sha256
        and details.get("r92_feature_state_sha256")
        == assets.feature_state_sha256
        and isinstance(outputs, Mapping)
        and outputs.get("new_bundle_sha256") == bundle.bundle_sha256
        and hashlib.sha256(bundle.prototype_npy).hexdigest() == prototype_sha256
        and hashlib.sha256(bundle.pca_components_npy).hexdigest()
        == pca_components_sha256
        and hashlib.sha256(bundle.pca_mean_npy).hexdigest() == pca_mean_sha256,
        "later transfer scorer is not the frozen-state project parent",
    )
    scorer = CanonicalTransferScorerReference(
        kind="PROJECT_TRANSFER_BUNDLE",
        scorer_sha256=bundle.bundle_sha256,
        scores_sha256=archive.predictions_sha256,
        # Rebound to the derived AL-pool CSV identity by begin/resume below.
        pool_scores_sha256=archive.predictions_sha256,
        feature_state_sha256=assets.feature_state_sha256,
        prototype_sha256=prototype_sha256,
        pca_components_sha256=pca_components_sha256,
        pca_mean_sha256=pca_mean_sha256,
        r92_classifier_sha256=assets.classifier_sha256,
        r92_resource_manifest_sha256=assets.resource_manifest_sha256,
        preset_id=None,
    )
    return (
        archive,
        scorer,
        assets.feature_state,
        ancestry,
        completion_path,
        ancestry_log,
    )


def _verify_transfer_project_config(
    config_path: Path,
    stage60_root: Path,
    number: int,
    *,
    source_bundle: Path | None,
) -> object:
    from compag_curation.canonical.service import (
        _bundle_config_maps,
        _canonical_config,
    )
    from compag_curation.public_config import (
        check_local_assets,
        load_public_config,
    )

    config = load_public_config(config_path, check_local=False)
    _canonical_config(config)
    assets = check_local_assets(config)
    _require(
        config.profile == CANONICAL_GPU_PROFILE
        and config.device == "cuda"
        and assets.get("status") == "PASS"
        and isinstance(assets.get("assets"), list)
        and len(assets["assets"]) == 5,
        "transfer resume requires a verified canonical Full CUDA project config",
    )
    if number == 1:
        receipt = _json_object(
            stage60_root.absolute() / "transfer_scorer.json",
            "r92 transfer scorer receipt",
        )
        _require(
            receipt.get("config_sha256") == config.config_sha256,
            "transfer resume config differs from the r92-scored image config",
        )
    else:
        _require(source_bundle is not None, "later transfer config lacks a parent bundle")
        parent = _verified_gpu_execution_bundle(source_bundle)
        normalized, preprocessing, proposal, features, training = (
            _bundle_config_maps(config)
        )
        _require(
            parent.profile == CANONICAL_GPU_PROFILE
            and parent.normalized_config == normalized
            and parent.preprocessing == preprocessing
            and parent.proposal_config == proposal
            and parent.feature_config == features
            and parent.training_config == training,
            "transfer config maps or public assets differ from the project parent",
        )
    return config


def _verified_canonical_bundle(root: Path) -> object:
    bundle = verify_model_bundle(root)
    _canonical_device_for_profile(getattr(bundle, "profile", None))
    _require(
        bundle.schema == BUNDLE_SCHEMA_V2
        and bundle.profile in V2_PIPELINE_PROFILES,
        "active-learning facade requires a verified v2 pipeline bundle",
    )
    return bundle


def _verified_gpu_execution_bundle(root: Path) -> object:
    bundle = _verified_canonical_bundle(root)
    _require(
        bundle.profile in GPU_EXECUTION_PROFILES
        and _canonical_device_for_profile(bundle.profile) == "cuda",
        "active-learning execution requires a canonical GPU/CUDA bundle, an efficient Lite GPU/CUDA bundle, or the full-image GPU/CUDA bundle",
    )
    return bundle


def _feature_state(bundle: object) -> object:
    provenance = getattr(bundle, "provenance", None)
    details = provenance.get("details") if isinstance(provenance, Mapping) else None
    _require(isinstance(details, Mapping), "canonical bundle feature-state provenance is missing")
    training_rows = details.get("training_row_count")
    positive_rows = details.get("positive_training_row_count")
    _require(
        isinstance(training_rows, int)
        and not isinstance(training_rows, bool)
        and training_rows >= 32
        and isinstance(positive_rows, int)
        and not isinstance(positive_rows, bool)
        and 1 <= positive_rows <= training_rows
        and isinstance(getattr(bundle, "prototype_npy", None), bytes)
        and isinstance(getattr(bundle, "pca_components_npy", None), bytes)
        and isinstance(getattr(bundle, "pca_mean_npy", None), bytes),
        "canonical bundle feature state is incomplete",
    )
    return deserialize_canonical_feature_state(
        CanonicalFeatureStatePayloads(
            prototype_npy=bundle.prototype_npy,
            pca_components_npy=bundle.pca_components_npy,
            pca_mean_npy=bundle.pca_mean_npy,
        ),
        training_row_count=training_rows,
        positive_row_count=positive_rows,
    )


def _verify_sequential_image_parent(
    bundle: object,
    *,
    round_number: int,
    ancestry_completion: Path,
    stage20_features: Path,
    genesis_split: Path,
) -> None:
    provenance = getattr(bundle, "provenance", None)
    details = provenance.get("details") if isinstance(provenance, Mapping) else None
    _require(
        isinstance(details, Mapping),
        "sequential image parent bundle provenance is missing",
    )
    stage20_sha256 = _file_record(
        stage20_features,
        "stage20_features",
    )["sha256"]
    split_sha256 = _file_record(
        genesis_split,
        "genesis_split",
        max_bytes=16 * 1024 * 1024,
    )["sha256"]
    if round_number == 1:
        initial = _json_object(
            ancestry_completion,
            "initial active-learning completion",
        )
        stage20_image_sha256s = frozenset(
            row.image_sha256
            for row in read_canonical_stage20_raw_features(stage20_features)
        )
        _initial_ids, initial_image_sha256s = _decision_identity_sets(
            ancestry_completion.absolute().parent / "decision_log.csv"
        )
        _require(
            details.get("reviewed_sha256")
            == initial.get("reviewed_batch_sha256")
            and details.get("split_manifest_sha256") == split_sha256
            and details.get("raw_features_sha256") == stage20_sha256
            and initial_image_sha256s == stage20_image_sha256s,
            "round-1 parent bundle is not the model trained from the bound genesis review, features, and split",
        )
        # The canonical Stage-20 producer now fails closed unless every
        # prepared genesis image has at least one proposal.  These feature-row
        # image hashes therefore close the genesis identities for the official
        # workflow.  The Stage-20 schema still has no independent Stage-10 (or
        # Stage-00) inventory hash, so a deliberately re-sealed standalone
        # Stage-20 artifact remains outside that ancestry guarantee.
        return
    prior = _json_object(
        ancestry_completion,
        "prior sequential image completion",
    )
    outputs = prior.get("outputs")
    _require(
        prior.get("schema")
        == "compag-curation-canonical-active-learning-completion/v3"
        and prior.get("workflow")
        == "SINGLE_NEW_IMAGE_UNCERTAINTY_BATCH_FULL_RETRAIN_V1"
        and isinstance(outputs, dict)
        and outputs.get("new_bundle_sha256")
        == getattr(bundle, "bundle_sha256", None)
        and details.get("active_learning_workflow")
        == "SINGLE_NEW_IMAGE_UNCERTAINTY_BATCH_FULL_RETRAIN_V1"
        and details.get("active_learning_round_number") == round_number - 1
        and details.get("model_label") == f"project-r{round_number - 1}"
        and details.get("genesis_stage20_features_sha256") == stage20_sha256
        and details.get("genesis_split_sha256") == split_sha256,
        "sequential image parent bundle or project-round lineage changed",
    )


def _verify_single_image_inference_inventory(
    stage60_root: Path,
    archive: object,
) -> None:
    """Close an image round against the full Stage-60 input inventory."""

    result = _json_object(
        stage60_root.absolute() / "inference_result.json",
        "single-image canonical inference result",
    )
    inventory = result.get("input_inventory")
    pool_rows = tuple(getattr(archive, "pool_rows", ()))
    identities = {
        (row.image_id, row.image_sha256, row.group_id) for row in pool_rows
    }
    _require(
        len(pool_rows) > 0
        and len(identities) == 1
        and result.get("image_count") == 1
        and result.get("proposal_count") == len(pool_rows)
        and result.get("prediction_rows") == len(pool_rows)
        and isinstance(inventory, list)
        and len(inventory) == 1
        and isinstance(inventory[0], dict),
        "sequential image active learning requires exactly one Stage-60 input image",
    )
    image_id, image_sha256, group_id = next(iter(identities))
    image = inventory[0]
    _require(
        image.get("image_id") == image_id
        and image.get("image_sha256") == image_sha256
        and image.get("group_id") == group_id
        and image.get("candidate_rows") == len(pool_rows),
        "sequential image Stage-60 inventory differs from its proposal pool",
    )


def _decision_identity_sets(
    path: Path,
) -> tuple[frozenset[str], frozenset[str]]:
    _absolute, payload = _safe_file(path, "active-learning decision log")
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
        _require(tuple(reader.fieldnames or ()) == CANONICAL_AL_DECISION_LOG_COLUMNS, "decision log header changed")
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise PublicIOError("decision log is malformed") from exc
    ids = [row.get("proposal_id", "") for row in rows]
    image_sha256s = [row.get("image_sha256", "") for row in rows]
    _require(
        bool(ids)
        and len(ids) == len(set(ids))
        and all(_SHA256.fullmatch(value) is not None for value in ids)
        and all(
            isinstance(value, str) and _SHA256.fullmatch(value) is not None
            for value in image_sha256s
        ),
        "decision log proposal identities are invalid",
    )
    return frozenset(ids), frozenset(image_sha256s)


def _decision_ids(path: Path) -> frozenset[str]:
    return _decision_identity_sets(path)[0]


def _validated_stage20_root(stage20_root: Path) -> tuple[Path, str]:
    """Verify one published Stage-20 closure and return its bound GPU profile."""

    from .service import CANONICAL_RAW_FEATURE_TABLE_COLUMNS

    root = _safe_directory(stage20_root, "canonical Stage-20 root")
    receipt_value = _json_object(
        root / "_SUCCESS.json",
        "canonical Stage-20 receipt",
    )
    run_id = receipt_value.get("run_id")
    try:
        parsed_run_id = uuid.UUID(str(run_id))
    except (AttributeError, ValueError) as exc:
        raise PublicIOError("canonical Stage-20 receipt run ID is invalid") from exc
    _require(
        isinstance(run_id, str) and str(parsed_run_id) == run_id,
        "canonical Stage-20 receipt run ID is invalid",
    )
    receipt = _validate_stage(root, "20_proposals_features", run_id)
    _require(
        receipt.get("required")
        == list(STAGE_REQUIRED_FILES["20_proposals_features"]),
        "canonical Stage-20 required-output contract changed",
    )

    proposal_config = _json_object(
        root / "proposal_config.json",
        "canonical Stage-20 proposal configuration",
    )
    summary = _json_object(
        root / "proposal_summary.json",
        "canonical Stage-20 proposal summary",
    )
    profile = proposal_config.get("profile")
    _require(
        isinstance(profile, str) and profile in GPU_EXECUTION_PROFILES,
        "canonical Stage-20 profile must be Full or Lite GPU execution, or full-image GPU execution",
    )
    settings = amg_settings_for_profile(profile)
    assets = BUNDLE_V2_ASSET_PROFILES[profile]
    expected_microbatch = stage20_amg_execution_points_per_batch(profile)
    expected_config = {
        "schema": "compag-curation-canonical-proposal-config/v1",
        "status": "PASS",
        **settings.provenance_record(profile=profile),
        "sam2_execution_points_per_batch": expected_microbatch,
        "sam2_checkpoint_sha256": assets["sam2_checkpoint_sha256"],
        "sam2_config_sha256": assets["sam2_config_sha256"],
        "sam2_config_locator": assets["sam2_config_locator"],
        "resnet50_weights_sha256": assets["resnet50_weights_sha256"],
        "embedding_backbone": "resnet50-imagenet1k-v2",
        "embedding_dimensions": CANONICAL_EMBEDDING_DIMENSIONS,
        "masked_crop_padding": 0.10,
        "pca_fit": "DEFERRED_UNTIL_REVIEWED_TRAIN_PARTITION",
        "prototype_fit": "DEFERRED_UNTIL_REVIEWED_TRAIN_POSITIVES",
        "raw_feature_columns": list(CANONICAL_RAW_FEATURE_TABLE_COLUMNS),
        "final_feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
    }
    _require(
        dict(proposal_config) == expected_config,
        "canonical Stage-20 proposal configuration differs from its profile contract",
    )

    file_hashes = {
        name: _file_record(root / name, f"canonical Stage-20 {name}")["sha256"]
        for name in ("proposals.csv", "features.csv", "proposal_config.json")
    }
    proposal_count = summary.get("proposal_count")
    raw_feature_row_count = summary.get("raw_feature_row_count")
    _require(
        summary.get("schema") == "compag-curation-canonical-proposals/v1"
        and summary.get("status") == "PASS"
        and summary.get("profile") == profile
        and summary.get("sam2_execution_points_per_batch") == expected_microbatch
        and isinstance(proposal_count, int)
        and not isinstance(proposal_count, bool)
        and proposal_count > 0
        and isinstance(raw_feature_row_count, int)
        and not isinstance(raw_feature_row_count, bool)
        and raw_feature_row_count
        == proposal_count * len(CANONICAL_FEATURE_CROP_SCALES)
        and summary.get("raw_rows_per_proposal")
        == len(CANONICAL_FEATURE_CROP_SCALES)
        and summary.get("proposals_sha256") == file_hashes["proposals.csv"]
        and summary.get("raw_features_sha256") == file_hashes["features.csv"]
        and summary.get("proposal_config_sha256")
        == file_hashes["proposal_config.json"],
        "canonical Stage-20 summary, hashes, or microbatch are invalid",
    )
    if profile == FULL_IMAGE_GPU_PROFILE:
        _require(
            summary.get("tile_count") == summary.get("amg_input_count"),
            "full-image Stage-20 tile_count compatibility counter changed",
        )
    parsed_features = read_canonical_stage20_raw_features(root / "features.csv")
    _require(
        len(parsed_features) == raw_feature_row_count,
        "canonical Stage-20 summary differs from its raw feature table",
    )
    return root, profile


def initial_export(stage20_root: Path, output: Path) -> tuple[dict[str, object], int]:
    """Publish an export-all cold-start operation and pause for review."""

    specs = (_InputSpec("stage20_root", stage20_root, True),)
    attempt = _start_operation("initial_export", output, specs)
    try:
        stage20, profile = _validated_stage20_root(stage20_root)
        _require(
            _directory_record(stage20, "stage20_root") == attempt.inputs[0],
            "canonical Stage-20 root changed before validation",
        )
        payload = _payload_root(attempt)
        pool = payload / "initial_pool"
        pool_result = write_initial_canonical_active_learning_pool(
            stage20 / "features.csv",
            pool,
            profile=profile,
        )
        _require(pool_result.get("status") == "PASS", "initial pool did not complete")
        export = payload / "initial_export"
        result = write_initial_labeling_export_all(pool, export, created_at_utc=_utc_now())
        _require(
            result.get("status") == "PAUSED_FOR_REVIEW"
            and (export / "review_request.csv").is_file()
            and (export / "initial_manifest.json").is_file(),
            "initial export did not close its review payload",
        )
        return _finish_operation(attempt, status="PAUSED_FOR_REVIEW", exit_code=3, round_number=None)
    except BaseException as exc:
        _record_failure(attempt, exc)
        raise


def initial_resume(
    initial_operation_root: Path,
    reviewed: Path,
    output: Path,
) -> tuple[dict[str, object], int]:
    """Validate the export-all review batch and publish its immutable log."""

    initial = _verify_published_operation(
        initial_operation_root,
        operation="initial_export",
        status="PAUSED_FOR_REVIEW",
        exit_code=3,
        round_number=None,
    )
    initial_root = Path(initial["root"])
    pool = initial_root / "payload/initial_pool"
    export = initial_root / "payload/initial_export"
    validate_review_table(export / "review_request.csv", reviewed)
    specs = (
        _InputSpec("initial_operation_root", initial_root, True),
        _InputSpec("reviewed", reviewed, False, 128 * 1024 * 1024),
    )
    attempt = _start_operation("initial_resume", output, specs)
    try:
        payload = _payload_root(attempt)
        completion_root = payload / "initial_completion"
        result = complete_initial_labeling_export_all(
            export,
            pool,
            reviewed,
            completion_root,
            decision_timestamp_utc=_utc_now(),
        )
        _require(
            result.get("status") == "PASS"
            and (completion_root / "decision_log.csv").is_file()
            and (completion_root / "initial_completion.json").is_file(),
            "initial resume did not close its completion payload",
        )
        return _finish_operation(attempt, status="PASS", exit_code=0, round_number=None)
    except BaseException as exc:
        _record_failure(attempt, exc)
        raise


def begin_round(
    stage60_root: Path,
    bundle: Path,
    split: Path,
    decision_log: Path,
    round_number: int,
    initial_completion_or_prior_completion: Path,
    output: Path,
) -> tuple[dict[str, object], int]:
    """Select one model-driven canonical round and pause for review."""

    number = _round_number(round_number)
    ancestry, ancestry_completion, ancestry_log = _ancestry_root(
        initial_completion_or_prior_completion,
        number,
    )
    _require(
        _file_record(ancestry_log, "decision_log") == _file_record(decision_log, "decision_log"),
        "supplied decision log differs from the ancestry operation",
    )
    archive = read_canonical_active_learning_inference(stage60_root)
    verified = _verified_gpu_execution_bundle(bundle)
    _require(
        archive.profile == verified.profile
        and archive.device == _canonical_device_for_profile(verified.profile)
        and archive.bundle_sha256 == verified.bundle_sha256,
        "canonical inference archive was not scored by the supplied bundle",
    )
    specs = (
        _InputSpec("canonical_inference_root", stage60_root, True),
        _InputSpec("source_bundle", bundle, True),
        _InputSpec("split", split, False, 16 * 1024 * 1024),
        _InputSpec("decision_log", decision_log, False),
        _InputSpec("ancestry_operation_root", Path(ancestry["root"]), True),
    )
    attempt = _start_operation("begin_round", output, specs)
    try:
        payload = _payload_root(attempt)
        pool = payload / "round_pool"
        pool_result = write_canonical_active_learning_pool(
            archive.pool_rows,
            pool,
            bundle_sha256=archive.bundle_sha256,
            source_sha256=archive.raw_features_sha256,
            feature_archive_sha256=archive.feature_archive_sha256,
            profile=archive.profile,
        )
        _require(pool_result.get("status") == "PASS", "round pool did not complete")
        selection = payload / "selection"
        kwargs = (
            {"initial_completion_path": ancestry_completion}
            if number == 1
            else {"prior_round_completion_path": ancestry_completion}
        )
        result = begin_canonical_active_learning_round(
            pool,
            bundle,
            split,
            decision_log,
            selection,
            round_number=number,
            created_at_utc=_utc_now(),
            **kwargs,
        )
        _require(
            result.get("status") == "PAUSED_FOR_REVIEW"
            and result.get("round_number") == number
            and (selection / "review_request.csv").is_file()
            and (selection / "round_manifest.json").is_file(),
            "active-learning selection did not close its review payload",
        )
        return _finish_operation(attempt, status="PAUSED_FOR_REVIEW", exit_code=3, round_number=number)
    except BaseException as exc:
        _record_failure(attempt, exc)
        raise


def resume_round(
    selection_operation_root: Path,
    stage60_root: Path,
    stage20_features: Path,
    source_bundle: Path,
    split: Path,
    decision_log: Path,
    reviewed: Path,
    round_number: int,
    initial_completion_or_prior_completion: Path,
    output: Path,
) -> tuple[dict[str, object], int]:
    """Retrain, verify, rescore, and publish one completed canonical round."""

    number = _round_number(round_number)
    selection_operation = _verify_published_operation(
        selection_operation_root,
        operation="begin_round",
        status="PAUSED_FOR_REVIEW",
        exit_code=3,
        round_number=number,
    )
    selection_root = Path(selection_operation["root"])
    ancestry, ancestry_completion, ancestry_log = _ancestry_root(
        initial_completion_or_prior_completion,
        number,
    )
    _require(
        _file_record(ancestry_log, "decision_log") == _file_record(decision_log, "decision_log"),
        "supplied decision log differs from the ancestry operation",
    )
    current_bindings = {
        "canonical_inference_root": _directory_record(
            stage60_root,
            "canonical_inference_root",
        ),
        "source_bundle": _directory_record(source_bundle, "source_bundle"),
        "split": _file_record(split, "split", max_bytes=16 * 1024 * 1024),
        "decision_log": _file_record(decision_log, "decision_log"),
        "ancestry_operation_root": _directory_record(Path(ancestry["root"]), "ancestry_operation_root"),
    }
    for role, record in current_bindings.items():
        _require(
            _input_by_role(selection_operation, role) == record,
            f"resume {role} differs from the sealed selection operation",
        )
    selection_payload = selection_root / "payload/selection"
    pool = selection_root / "payload/round_pool"
    reviewed_rows, _review_result = validate_review_table(
        selection_payload / "review_request.csv",
        reviewed,
    )
    _safe_output(output)
    archive = read_canonical_active_learning_inference(stage60_root)
    verified = _verified_gpu_execution_bundle(source_bundle)
    device = _canonical_device_for_profile(verified.profile)
    _require(
        archive.profile == verified.profile
        and archive.device == device
        and archive.bundle_sha256 == verified.bundle_sha256,
        "canonical inference archive was not scored by the source bundle",
    )
    science_mode = "gpu"
    science_dependencies = require_science_dependencies(device)
    previous_state = _feature_state(verified)
    stage20_rows = read_canonical_stage20_raw_features(stage20_features)
    stage20_ids = frozenset(row.proposal_id for row in stage20_rows)
    desired_archive_ids = (
        _decision_ids(decision_log)
        | frozenset(row.proposal_id for row in reviewed_rows)
    ) - stage20_ids
    accumulated_archive_rows = tuple(
        row for row in archive.feature_rows
        if row.proposal_id in desired_archive_ids
    )
    _require(
        {row.proposal_id for row in accumulated_archive_rows} == set(desired_archive_ids),
        "canonical inference raw archive omits accumulated AL decisions",
    )
    training_rows = tuple(stage20_rows) + accumulated_archive_rows
    specs = (
        _InputSpec("selection_operation_root", selection_root, True),
        _InputSpec("canonical_inference_root", stage60_root, True),
        _InputSpec("stage20_features", stage20_features, False),
        _InputSpec("source_bundle", source_bundle, True),
        _InputSpec("split", split, False, 16 * 1024 * 1024),
        _InputSpec("decision_log", decision_log, False),
        _InputSpec("reviewed", reviewed, False, 128 * 1024 * 1024),
        _InputSpec("ancestry_operation_root", Path(ancestry["root"]), True),
    )
    attempt = _start_operation(
        "resume_round",
        output,
        specs,
        science_dependencies=science_dependencies,
        science_mode=science_mode,
    )
    try:
        payload = _payload_root(attempt)
        completion_root = payload / "round_completion"
        kwargs = (
            {"initial_completion_path": ancestry_completion}
            if number == 1
            else {"prior_round_completion_path": ancestry_completion}
        )

        def concrete_retrain(request: object, destination: Path) -> None:
            retrain_canonical_active_learning_bundle(
                request,
                destination,
                source_bundle_root=source_bundle,
            )

        result = resume_canonical_active_learning_round(
            selection_payload,
            pool,
            source_bundle,
            split,
            decision_log,
            reviewed,
            previous_state,
            training_rows,
            archive.feature_rows,
            completion_root,
            completed_at_utc=_utc_now(),
            retrain=concrete_retrain,
            **kwargs,
        )
        _require(
            result.get("status") == "PASS"
            and result.get("round_number") == number
            and (completion_root / "decision_log.csv").is_file()
            and (completion_root / "retrain_request.json").is_file()
            and (completion_root / "model_bundle/bundle.json").is_file()
            and (completion_root / "rescored_pool/pool_result.json").is_file()
            and (completion_root / "round_completion.json").is_file(),
            "active-learning resume did not close every required output",
        )
        return _finish_operation(attempt, status="PASS", exit_code=0, round_number=number)
    except BaseException as exc:
        _record_failure(attempt, exc)
        raise


def begin_image_round(
    stage60_root: Path,
    bundle: Path,
    stage20_features: Path,
    genesis_split: Path,
    decision_log: Path,
    round_number: int,
    initial_completion_or_prior_completion: Path,
    output: Path,
) -> tuple[dict[str, object], int]:
    """Select one uncertainty batch from exactly one previously unseen image."""

    number = _round_number(round_number)
    ancestry, ancestry_completion, ancestry_log = _image_ancestry_root(
        initial_completion_or_prior_completion,
        number,
    )
    _require(
        _file_record(ancestry_log, "decision_log")
        == _file_record(decision_log, "decision_log"),
        "supplied decision log differs from the sequential-image ancestry operation",
    )
    archive = read_canonical_active_learning_inference(stage60_root)
    _verify_single_image_inference_inventory(stage60_root, archive)
    verified = _verified_gpu_execution_bundle(bundle)
    _require(
        archive.profile == verified.profile
        and archive.device == _canonical_device_for_profile(verified.profile)
        and archive.bundle_sha256 == verified.bundle_sha256,
        "single-image inference was not scored by the supplied parent bundle",
    )
    _verify_sequential_image_parent(
        verified,
        round_number=number,
        ancestry_completion=ancestry_completion,
        stage20_features=stage20_features,
        genesis_split=genesis_split,
    )
    specs = (
        _InputSpec("canonical_inference_root", stage60_root, True),
        _InputSpec("source_bundle", bundle, True),
        _InputSpec("stage20_features", stage20_features, False),
        _InputSpec("genesis_split", genesis_split, False, 16 * 1024 * 1024),
        _InputSpec("decision_log", decision_log, False),
        _InputSpec("ancestry_operation_root", Path(ancestry["root"]), True),
    )
    attempt = _start_operation(
        "begin_image_round",
        output,
        specs,
    )
    try:
        payload = _payload_root(attempt)
        pool = payload / "round_pool"
        pool_result = write_canonical_active_learning_pool(
            archive.pool_rows,
            pool,
            bundle_sha256=archive.bundle_sha256,
            source_sha256=archive.raw_features_sha256,
            feature_archive_sha256=archive.feature_archive_sha256,
            profile=archive.profile,
        )
        _require(
            pool_result.get("status") == "PASS",
            "single-image round pool did not complete",
        )
        selection = payload / "selection"
        kwargs = (
            {"initial_completion_path": ancestry_completion}
            if number == 1
            else {"prior_round_completion_path": ancestry_completion}
        )
        result = begin_canonical_active_learning_round(
            pool,
            bundle,
            genesis_split,
            decision_log,
            selection,
            round_number=number,
            created_at_utc=_utc_now(),
            image_scoped=True,
            **kwargs,
        )
        _require(
            result.get("status") == "PAUSED_FOR_REVIEW"
            and result.get("schema")
            == "compag-curation-canonical-active-learning-selection/v3"
            and result.get("round_number") == number
            and (selection / "review_request.csv").is_file()
            and (selection / "round_manifest.json").is_file(),
            "sequential image selection did not close its review payload",
        )
        return _finish_operation(
            attempt,
            status="PAUSED_FOR_REVIEW",
            exit_code=3,
            round_number=number,
        )
    except BaseException as exc:
        _record_failure(attempt, exc)
        raise


def resume_image_round(
    selection_operation_root: Path,
    stage60_root: Path,
    stage20_features: Path,
    source_bundle: Path,
    genesis_split: Path,
    decision_log: Path,
    reviewed: Path,
    round_number: int,
    initial_completion_or_prior_completion: Path,
    output: Path,
) -> tuple[dict[str, object], int]:
    """Full-retrain one model from genesis plus cumulative one-image AL labels."""

    number = _round_number(round_number)
    selection_operation = _verify_published_operation(
        selection_operation_root,
        operation="begin_image_round",
        status="PAUSED_FOR_REVIEW",
        exit_code=3,
        round_number=number,
    )
    selection_root = Path(selection_operation["root"])
    ancestry, ancestry_completion, ancestry_log = _image_ancestry_root(
        initial_completion_or_prior_completion,
        number,
    )
    _require(
        _file_record(ancestry_log, "decision_log")
        == _file_record(decision_log, "decision_log"),
        "supplied decision log differs from the sequential-image ancestry operation",
    )
    current_bindings = {
        "canonical_inference_root": _directory_record(
            stage60_root,
            "canonical_inference_root",
        ),
        "source_bundle": _directory_record(source_bundle, "source_bundle"),
        "stage20_features": _file_record(
            stage20_features,
            "stage20_features",
        ),
        "genesis_split": _file_record(
            genesis_split,
            "genesis_split",
            max_bytes=16 * 1024 * 1024,
        ),
        "decision_log": _file_record(decision_log, "decision_log"),
        "ancestry_operation_root": _directory_record(
            Path(ancestry["root"]),
            "ancestry_operation_root",
        ),
    }
    for role, record in current_bindings.items():
        _require(
            _input_by_role(selection_operation, role) == record,
            f"sequential image resume {role} differs from the sealed selection operation",
        )
    selection_payload = selection_root / "payload/selection"
    pool = selection_root / "payload/round_pool"
    reviewed_rows, _review_result = validate_review_table(
        selection_payload / "review_request.csv",
        reviewed,
    )
    _safe_output(output)
    archive = read_canonical_active_learning_inference(stage60_root)
    _verify_single_image_inference_inventory(stage60_root, archive)
    verified = _verified_gpu_execution_bundle(source_bundle)
    _require(
        archive.profile == verified.profile
        and archive.device == _canonical_device_for_profile(verified.profile)
        and archive.bundle_sha256 == verified.bundle_sha256,
        "single-image inference was not scored by the supplied parent bundle",
    )
    _verify_sequential_image_parent(
        verified,
        round_number=number,
        ancestry_completion=ancestry_completion,
        stage20_features=stage20_features,
        genesis_split=genesis_split,
    )
    science_dependencies = require_science_dependencies(
        _canonical_device_for_profile(verified.profile)
    )
    previous_state = _feature_state(verified)
    stage20_rows = read_canonical_stage20_raw_features(stage20_features)
    prior_al_rows: tuple[object, ...] = ()
    if number > 1:
        prior_al_rows, _prior_archive_receipt = (
            read_canonical_accumulated_feature_archive(
                ancestry_completion.absolute().parent
                / "accumulated_al_features.csv"
            )
        )
    reviewed_ids = {row.proposal_id for row in reviewed_rows}
    current_rows = tuple(
        row for row in archive.feature_rows if row.proposal_id in reviewed_ids
    )
    _require(
        {row.proposal_id for row in current_rows} == reviewed_ids,
        "single-image raw feature archive omits a reviewed proposal",
    )
    training_rows = tuple(stage20_rows) + tuple(prior_al_rows) + current_rows
    specs = (
        _InputSpec("selection_operation_root", selection_root, True),
        _InputSpec("canonical_inference_root", stage60_root, True),
        _InputSpec("stage20_features", stage20_features, False),
        _InputSpec("source_bundle", source_bundle, True),
        _InputSpec("genesis_split", genesis_split, False, 16 * 1024 * 1024),
        _InputSpec("decision_log", decision_log, False),
        _InputSpec("reviewed", reviewed, False, 128 * 1024 * 1024),
        _InputSpec("ancestry_operation_root", Path(ancestry["root"]), True),
    )
    attempt = _start_operation(
        "resume_image_round",
        output,
        specs,
        science_dependencies=science_dependencies,
        science_mode="gpu",
    )
    try:
        payload = _payload_root(attempt)
        completion_root = payload / "round_completion"
        kwargs = (
            {"initial_completion_path": ancestry_completion}
            if number == 1
            else {"prior_round_completion_path": ancestry_completion}
        )

        def concrete_retrain(request: object, destination: Path) -> None:
            retrain_canonical_active_learning_bundle(
                request,
                destination,
                source_bundle_root=source_bundle,
            )

        result = resume_canonical_active_learning_round(
            selection_payload,
            pool,
            source_bundle,
            genesis_split,
            decision_log,
            reviewed,
            previous_state,
            training_rows,
            archive.feature_rows,
            completion_root,
            completed_at_utc=_utc_now(),
            retrain=concrete_retrain,
            image_scoped=True,
            **kwargs,
        )
        _require(
            result.get("status") == "PASS"
            and result.get("schema")
            == "compag-curation-canonical-active-learning-completion/v3"
            and result.get("round_number") == number
            and (completion_root / "decision_log.csv").is_file()
            and (completion_root / "retrain_request.json").is_file()
            and (completion_root / "accumulated_al_features.csv").is_file()
            and (completion_root / "model_lineage.json").is_file()
            and (completion_root / "model_bundle/bundle.json").is_file()
            and (completion_root / "rescored_pool/pool_result.json").is_file()
            and (completion_root / "round_completion.json").is_file(),
            "sequential image resume did not close every required output",
        )
        return _finish_operation(
            attempt,
            status="PASS",
            exit_code=0,
            round_number=number,
        )
    except BaseException as exc:
        _record_failure(attempt, exc)
        raise


def begin_transfer_image_round(
    stage60_root: Path,
    transfer_baseline: Path,
    round_number: int,
    output: Path,
    *,
    source_bundle: Path | None = None,
    decision_log: Path | None = None,
    prior_completion: Path | None = None,
) -> tuple[dict[str, object], int]:
    """Pause one strict one-image post-r92 transfer round for explicit review."""

    number = _round_number(round_number)
    _safe_output(output)
    _baseline, baseline_reference = _transfer_baseline_reference(
        transfer_baseline
    )
    (
        archive,
        scorer,
        _feature_state_value,
        ancestry,
        ancestry_completion,
        ancestry_log,
    ) = _transfer_round_context(
        stage60_root,
        baseline_reference,
        number,
        source_bundle=source_bundle,
        decision_log=decision_log,
        prior_completion=prior_completion,
    )
    specs: list[_InputSpec] = [
        _InputSpec("canonical_inference_root", stage60_root, True),
        _InputSpec("transfer_baseline", transfer_baseline, True),
    ]
    if number > 1:
        _require(
            source_bundle is not None
            and decision_log is not None
            and ancestry is not None
            and ancestry_completion is not None
            and ancestry_log is not None,
            "later transfer begin did not resolve its ancestry",
        )
        specs.extend(
            (
                _InputSpec("source_bundle", source_bundle, True),
                _InputSpec("decision_log", decision_log, False),
                _InputSpec(
                    "prior_completion_operation_root",
                    Path(ancestry["root"]),
                    True,
                ),
            )
        )
    attempt = _start_operation(
        "begin_transfer_image_round",
        output,
        tuple(specs),
    )
    try:
        payload = _payload_root(attempt)
        pool = payload / "round_pool"
        pool_result = write_canonical_active_learning_pool(
            archive.pool_rows,
            pool,
            bundle_sha256=scorer.scorer_sha256,
            source_sha256=archive.raw_features_sha256,
            feature_archive_sha256=archive.feature_archive_sha256,
            profile=CANONICAL_GPU_PROFILE,
        )
        _require(
            pool_result.get("status") == "PASS",
            "transfer round pool did not complete",
        )
        scorer = replace(
            scorer,
            pool_scores_sha256=str(pool_result["predictions_sha256"]),
        )
        selection = payload / "selection"
        result = begin_canonical_transfer_image_round(
            pool,
            baseline_reference,
            scorer,
            selection,
            round_number=number,
            created_at_utc=_utc_now(),
            prior_round_completion_path=ancestry_completion,
            decision_log_path=(decision_log if number > 1 else None),
        )
        _require(
            result.get("status") == "PAUSED_FOR_REVIEW"
            and result.get("schema")
            == "compag-curation-canonical-transfer-active-learning-selection/v1"
            and result.get("round_number") == number
            and result.get("workflow") == POST_R92_TRANSFER_WORKFLOW
            and (selection / "review_request.csv").is_file()
            and (selection / "round_manifest.json").is_file(),
            "transfer selection did not close its explicit review payload",
        )
        return _finish_operation(
            attempt,
            status="PAUSED_FOR_REVIEW",
            exit_code=3,
            round_number=number,
        )
    except BaseException as exc:
        _record_failure(attempt, exc)
        raise


def resume_transfer_image_round(
    selection_operation_root: Path,
    stage60_root: Path,
    transfer_baseline: Path,
    reviewed: Path,
    round_number: int,
    output: Path,
    *,
    config: Path,
    source_bundle: Path | None = None,
    decision_log: Path | None = None,
    prior_completion: Path | None = None,
    retrain: TransferRetrainFunction | None = None,
) -> tuple[dict[str, object], int]:
    """Fresh-fit project-rN after one complete, explicit image review."""

    number = _round_number(round_number)
    _require(
        retrain is None or callable(retrain),
        "transfer retrain override must be callable",
    )
    selection_operation = _verify_published_operation(
        selection_operation_root,
        operation="begin_transfer_image_round",
        status="PAUSED_FOR_REVIEW",
        exit_code=3,
        round_number=number,
    )
    selection_root = Path(selection_operation["root"])
    _baseline, baseline_reference = _transfer_baseline_reference(
        transfer_baseline
    )
    (
        archive,
        scorer,
        feature_state,
        ancestry,
        ancestry_completion,
        ancestry_log,
    ) = _transfer_round_context(
        stage60_root,
        baseline_reference,
        number,
        source_bundle=source_bundle,
        decision_log=decision_log,
        prior_completion=prior_completion,
    )
    project_config = _verify_transfer_project_config(
        config,
        stage60_root,
        number,
        source_bundle=source_bundle,
    )
    current_bindings = {
        "canonical_inference_root": _directory_record(
            stage60_root,
            "canonical_inference_root",
        ),
        "transfer_baseline": _directory_record(
            transfer_baseline,
            "transfer_baseline",
        ),
    }
    if number > 1:
        _require(
            source_bundle is not None
            and decision_log is not None
            and ancestry is not None
            and ancestry_completion is not None
            and ancestry_log is not None,
            "later transfer resume did not resolve its ancestry",
        )
        current_bindings.update(
            {
                "source_bundle": _directory_record(
                    source_bundle,
                    "source_bundle",
                ),
                "decision_log": _file_record(
                    decision_log,
                    "decision_log",
                ),
                "prior_completion_operation_root": _directory_record(
                    Path(ancestry["root"]),
                    "prior_completion_operation_root",
                ),
            }
        )
    for role, record in current_bindings.items():
        _require(
            _input_by_role(selection_operation, role) == record,
            f"transfer resume {role} differs from the sealed selection operation",
        )
    selection_payload = selection_root / "payload/selection"
    pool = selection_root / "payload/round_pool"
    sealed_pool_result = _json_object(
        pool / "pool_result.json",
        "transfer round pool result",
    )
    _require(
        isinstance(sealed_pool_result.get("predictions_sha256"), str),
        "transfer selection pool score identity is missing",
    )
    scorer = replace(
        scorer,
        pool_scores_sha256=str(sealed_pool_result["predictions_sha256"]),
    )
    reviewed_rows, _review_result = validate_review_table(
        selection_payload / "review_request.csv",
        reviewed,
    )
    _safe_output(output)
    prior_rows: tuple[object, ...] = ()
    if number > 1:
        _require(
            ancestry_completion is not None,
            "later transfer cumulative archive ancestry is missing",
        )
        prior_rows, _prior_receipt = read_canonical_accumulated_feature_archive(
            ancestry_completion.absolute().parent / "accumulated_al_features.csv"
        )
    reviewed_ids = {row.proposal_id for row in reviewed_rows}
    current_rows = tuple(
        row for row in archive.feature_rows if row.proposal_id in reviewed_ids
    )
    _require(
        {row.proposal_id for row in current_rows} == reviewed_ids,
        "transfer inference raw archive omits a reviewed proposal",
    )
    cumulative_rows = tuple(prior_rows) + current_rows
    specs: list[_InputSpec] = [
        _InputSpec("selection_operation_root", selection_root, True),
        _InputSpec("canonical_inference_root", stage60_root, True),
        _InputSpec("transfer_baseline", transfer_baseline, True),
        _InputSpec("config", config, False, 1024 * 1024),
        _InputSpec("reviewed", reviewed, False, 128 * 1024 * 1024),
    ]
    if number > 1:
        _require(
            source_bundle is not None
            and decision_log is not None
            and ancestry is not None,
            "later transfer resume did not retain its verified ancestry",
        )
        specs.extend(
            (
                _InputSpec("source_bundle", source_bundle, True),
                _InputSpec("decision_log", decision_log, False),
                _InputSpec(
                    "prior_completion_operation_root",
                    Path(ancestry["root"]),
                    True,
                ),
            )
        )
    science_dependencies = require_science_dependencies("cuda")
    attempt = _start_operation(
        "resume_transfer_image_round",
        output,
        tuple(specs),
        science_dependencies=science_dependencies,
        science_mode="gpu",
    )
    try:
        payload = _payload_root(attempt)
        completion_root = payload / "round_completion"

        def concrete_retrain(request: object, destination: Path) -> None:
            retrain_canonical_transfer_bundle(
                request,
                destination,
                config_path=config,
                expected_config_sha256=project_config.config_sha256,
                source_bundle_root=source_bundle,
            )

        result = resume_canonical_transfer_image_round(
            selection_payload,
            pool,
            transfer_baseline,
            baseline_reference,
            scorer,
            reviewed,
            feature_state,
            cumulative_rows,
            archive.feature_rows,
            completion_root,
            completed_at_utc=_utc_now(),
            retrain=(retrain if retrain is not None else concrete_retrain),
            prior_round_completion_path=ancestry_completion,
            decision_log_path=(decision_log if number > 1 else None),
        )
        _require(
            result.get("status") == "PASS"
            and result.get("schema")
            == "compag-curation-canonical-transfer-active-learning-completion/v1"
            and result.get("round_number") == number
            and result.get("workflow") == POST_R92_TRANSFER_WORKFLOW
            and (completion_root / "decision_log.csv").is_file()
            and (completion_root / "retrain_request.json").is_file()
            and (completion_root / "accumulated_al_features.csv").is_file()
            and (completion_root / "model_lineage.json").is_file()
            and (completion_root / "model_bundle/bundle.json").is_file()
            and (completion_root / "round_completion.json").is_file(),
            "transfer resume did not close every required output",
        )
        return _finish_operation(
            attempt,
            status="PASS",
            exit_code=0,
            round_number=number,
        )
    except BaseException as exc:
        _record_failure(attempt, exc)
        raise


__all__ = [
    "initial_export",
    "initial_resume",
    "begin_round",
    "resume_round",
    "begin_image_round",
    "resume_image_round",
    "begin_transfer_image_round",
    "resume_transfer_image_round",
]
