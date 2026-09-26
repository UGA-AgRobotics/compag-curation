"""Generated-data quick and real demonstrations for the public execution facade."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import uuid
from pathlib import Path
from typing import Iterable, Mapping

from .assets import verify_asset
from .public_config import (
    CANONICAL_GPU_PROFILE,
    EFFICIENT_GPU_PROFILE,
    GPU_EXECUTION_PROFILES,
    FEATURE_ORDER_SHA256,
    PUBLIC_FEATURE_ORDER,
    initialize_project,
)
from .public_io import (
    PublicIOError,
    canonical_json_value,
    canonical_json_bytes,
    compact_json_sha256,
    copy_new,
    fsync_directory,
    manifest_rows,
    portable_basename,
    purge_owned_runtime_cache_files,
    publish_directory_noreplace,
    sha256_file,
    stable_file,
    strict_json_bytes,
    write_new_bytes,
    write_new_json,
)
from .review.exchange import (
    CANONICAL_REVIEW_COLUMNS,
    REVIEW_ACTION_WEIGHTS,
    REVIEW_COLUMNS,
    validate_review_table,
    write_review_export,
    write_review_import,
)


_CANONICAL_REAL_DEMO_ASSETS = (
    ("resnet50-imagenet1k-v2-weights", "resnet50-11ad3fa6.pth"),
    ("sam2-apache-license", "SAM2-APACHE-2.0.txt"),
    ("sam2.1-hiera-large-checkpoint", "sam2.1_hiera_large.pt"),
    ("sam2.1-hiera-large-config", "sam2.1_hiera_l.yaml"),
    ("torchvision-bsd-license", "TORCHVISION-BSD-3-CLAUSE.txt"),
)
_EFFICIENT_REAL_DEMO_ASSETS = (
    ("resnet50-imagenet1k-v2-weights", "resnet50-11ad3fa6.pth"),
    ("sam2-apache-license", "SAM2-APACHE-2.0.txt"),
    ("sam2.1-hiera-tiny-checkpoint", "sam2.1_hiera_tiny.pt"),
    ("sam2.1-hiera-tiny-config", "sam2.1_hiera_t.yaml"),
    ("torchvision-bsd-license", "TORCHVISION-BSD-3-CLAUSE.txt"),
)
_REAL_DEMO_ASSETS_BY_PROFILE = {
    CANONICAL_GPU_PROFILE: _CANONICAL_REAL_DEMO_ASSETS,
    EFFICIENT_GPU_PROFILE: _EFFICIENT_REAL_DEMO_ASSETS,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


def _write_fd_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        _require(written > 0, "short write to demo ownership lock")
        offset += written


def _write_csv_file(output: Path, fields: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> int:
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o644)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="", closefd=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                count += 1
            handle.flush()
            os.fchmod(fd, 0o644)
            output_info = os.fstat(fd)
            _require(
                stat.S_ISREG(output_info.st_mode)
                and output_info.st_nlink == 1
                and stat.S_IMODE(output_info.st_mode) == 0o644
                and output_info.st_size > 0,
                f"demo CSV output metadata is unsafe: {output.name}",
            )
            os.fsync(fd)
    finally:
        os.close(fd)
    _require(count > 0, f"demo CSV output contains no rows: {output.name}")
    return count


def _record_demo_failure(staging: Path, run_id: str, kind: str, error: BaseException) -> None:
    if not staging.exists() or staging.is_symlink() or not staging.is_dir():
        return
    payload = {
        "schema": "compag-curation-demo-failure/v1",
        "status": "FAILED_DIAGNOSTIC_PRESERVED",
        "run_id": run_id,
        "profile": kind,
        "error_type": type(error).__name__,
        "error_code": "DEMO_EXECUTION_FAILED",
        "remediation": "Use the structured stderr error to correct the input, dependency, asset, or permission defect and choose a new output name.",
    }
    try:
        if not (staging / "_FAILURE.json").exists() and not (staging / "_FAILURE.json").is_symlink():
            write_new_json(staging / "_FAILURE.json", payload)
        if not (staging / "FINAL_STATUS.json").exists() and not (staging / "FINAL_STATUS.json").is_symlink():
            write_new_json(staging / "FINAL_STATUS.json", payload)
    except BaseException:
        pass


def _ppm(width: int, height: int, group_index: int) -> bytes:
    pixels = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            grid = x % 64 < 2 or y % 64 < 2
            insect_a = (x - (80 + 17 * group_index)) ** 2 + (y - (110 + 11 * group_index)) ** 2 <= 18 ** 2
            insect_b = ((x - (310 - 9 * group_index)) / 24) ** 2 + ((y - (300 + 7 * group_index)) / 13) ** 2 <= 1
            if insect_a:
                rgb = (35, 38, 30)
            elif insect_b:
                rgb = (82, 34, 28)
            elif grid:
                rgb = (108, 105, 70)
            else:
                rgb = (236, 207, 52)
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes(rgb)
    return f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(pixels)


def _demo_output(
    output: Path,
    kind: str,
    invocation: Mapping[str, object] | None,
) -> tuple[Path, Path, int, str]:
    absolute = output.absolute()
    parent = absolute.parent.resolve(strict=True)
    output = parent / absolute.name
    _require(output == absolute and not output.exists() and not output.is_symlink(), "demo output must be a fresh canonical path")
    portable_basename(output.name, "demo output name")
    lock = parent / f".{output.name}.compag-curation-{kind}-demo.lock"
    _require(len(lock.name.encode("ascii")) <= 255, "demo output name is too long for its ownership lock")
    _require(len(f".{output.name}.{kind}-demo.{uuid.uuid4()}".encode("ascii")) <= 255, "demo output name is too long for staging")
    _require(not lock.exists() and not lock.is_symlink(), "demo ownership lock already exists")
    invocation_record = dict(invocation or {"source": "PYTHON_API", "call": f"run_{kind}_demo"})
    if invocation_record.get("source") == "CLI":
        _require(
            set(invocation_record) == {"source", "command", "exact_argv_sha256"}
            and invocation_record.get("command") == "demo"
            and len(str(invocation_record.get("exact_argv_sha256", ""))) == 64
            and all(character in "0123456789abcdef" for character in str(invocation_record.get("exact_argv_sha256", ""))),
            "demo CLI invocation fields mismatch",
        )
    else:
        _require(
            invocation_record == {"source": "PYTHON_API", "call": f"run_{kind}_demo"},
            "demo Python invocation fields mismatch",
        )
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    run_id = str(uuid.uuid4())
    staging = parent / f".{output.name}.{kind}-demo.{uuid.uuid4()}"
    try:
        fsync_directory(parent)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.fchmod(fd, 0o600)
        marker = canonical_json_bytes({
            "schema": "compag-curation-demo-lock/v1",
            "run_id": run_id,
            "kind": kind,
            "output_name": output.name,
            "parent_device": parent.stat().st_dev,
        })
        _write_fd_all(fd, marker)
        os.fsync(fd)
        staging.mkdir(mode=0o700)
        fsync_directory(parent)
        write_new_json(staging / "_ATTEMPT.json", {"schema": "compag-curation-demo-attempt/v1", "status": "WRITING", "run_id": run_id, "kind": kind})
        write_new_json(staging / "command_record.json", {
            "schema": "compag-curation-demo-command/v1",
            "run_id": run_id,
            "command": "compag-curation demo",
            "invocation": invocation_record,
            "profile": kind,
            "output_policy": "FRESH_NO_CLOBBER",
        })
        return output, staging, fd, run_id
    except BaseException as exc:
        _record_demo_failure(staging, run_id, kind, exc)
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        raise


def _seal_demo(output: Path, staging: Path, kind: str, run_id: str) -> dict[str, object]:
    _require(kind in {"quick", "real"}, "demo profile is invalid")
    allowed_empty = (
        {"project/annotations", "project/review", "run/.staging"}
        if kind == "real"
        else set()
    )
    expected_status = {
        "schema": "compag-curation-demo-completion/v1",
        "status": "PASS",
        "run_id": run_id,
        "profile": kind,
        "output_name": output.name,
        "publication_condition": "VALID_ONLY_AT_DECLARED_OUTPUT_NAME_AFTER_ATOMIC_RENAME",
    }
    write_new_json(staging / "FINAL_STATUS.json", expected_status)
    rows = manifest_rows(
        staging,
        excluded={"OUTPUT_MANIFEST.json"},
        allowed_empty_directories=allowed_empty,
    )
    expected_manifest = {
        "schema": "compag-curation-demo-output-manifest/v1",
        "status": "PASS",
        "run_id": run_id,
        "kind": kind,
        "members": rows,
        "members_sha256": compact_json_sha256(rows),
    }
    write_new_json(staging / "OUTPUT_MANIFEST.json", expected_manifest)
    output_manifest_sha256 = sha256_file(staging / "OUTPUT_MANIFEST.json")
    publish_directory_noreplace(staging, output)
    status_payload, _status_snapshot = stable_file(output / "FINAL_STATUS.json", max_bytes=1024 * 1024)
    manifest_payload, _manifest_snapshot = stable_file(output / "OUTPUT_MANIFEST.json", max_bytes=128 * 1024 * 1024)
    try:
        final_status = canonical_json_value(status_payload, "published demo completion evidence")
        final_manifest = canonical_json_value(manifest_payload, "published demo output manifest")
    except PublicIOError as exc:
        raise PublicIOError("published demo completion evidence is invalid") from exc
    _require(
        final_status == expected_status,
        "published demo completion condition failed",
    )
    observed = manifest_rows(
        output,
        excluded={"OUTPUT_MANIFEST.json"},
        allowed_empty_directories=allowed_empty,
    )
    _require(
        observed == rows
        and final_manifest == expected_manifest
        and hashlib.sha256(manifest_payload).hexdigest() == output_manifest_sha256,
        "published demo manifest verification failed",
    )
    return {
        "schema": "compag-curation-demo-result/v1",
        "status": "PASS",
        "run_id": run_id,
        "profile": kind,
        "output_name": output.name,
        "output_manifest_sha256": output_manifest_sha256,
    }


def _reviewed_copy(
    request: Path,
    output: Path,
    *,
    positives_per_group: int = 1,
    canonical_actions: bool = False,
) -> None:
    payload, _snapshot = stable_file(request)
    with io.StringIO(payload.decode("utf-8"), newline="") as source:
        reader = csv.DictReader(source)
        columns = CANONICAL_REVIEW_COLUMNS if canonical_actions else REVIEW_COLUMNS
        _require(tuple(reader.fieldnames or ()) == columns, "demo review request header mismatch")
        rows = list(reader)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        _require(None not in row and all(isinstance(value, str) for value in row.values()), "demo review request row is malformed")
        grouped.setdefault(row["group_id"], []).append(row)
    _require(len(grouped) >= 4, "real demo requires at least four independent groups")
    for group, values in grouped.items():
        _require(len(values) >= positives_per_group + 1, f"demo group has too few proposals: {group}")
        values.sort(key=lambda row: row["proposal_id"])
        for index, row in enumerate(values):
            row["label"] = "1" if index < positives_per_group else "0"
            row["review_status"] = "reviewed"
            if canonical_actions:
                row["review_action"] = "accept"
                row["review_weight"] = format(REVIEW_ACTION_WEIGHTS["accept"], ".1f")
    ordered = sorted((row for values in grouped.values() for row in values), key=lambda row: row["proposal_id"])
    _write_csv_file(output, columns, ordered)


def _json_object(path: Path) -> dict[str, object]:
    payload, _snapshot = stable_file(path, max_bytes=128 * 1024 * 1024)
    try:
        value = canonical_json_value(payload, f"fresh-process evidence {path.name}")
    except PublicIOError as exc:
        raise PublicIOError(f"fresh-process evidence is malformed: {path.name}") from exc
    _require(isinstance(value, dict), f"fresh-process evidence must be an object: {path.name}")
    return value


def _demo_cuda_visibility_identity() -> dict[str, object]:
    from .runtime_lock import cuda_visibility_identity

    try:
        return cuda_visibility_identity()
    except RuntimeError as exc:
        raise PublicIOError(str(exc)) from exc


def _expected_demo_gpu_archive_provenance() -> dict[str, object]:
    """Derive the compact archive receipt from the reviewed packaged lock."""

    from .runtime_lock import SCIENCE_GPU_ARCHIVE_COUNT, science_gpu_lock_identity

    archives = science_gpu_lock_identity().get("archives")
    _require(
        isinstance(archives, dict) and len(archives) == SCIENCE_GPU_ARCHIVE_COUNT,
        "real-demo GPU archive lock is invalid",
    )
    names = sorted(str(name) for name in archives)
    closure = {
        "schema": "compag-curation-science-gpu-archive-set/v1",
        "archives": archives,
    }
    encoded = json.dumps(
        closure,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return {
        "schema": "compag-curation-science-gpu-archive-provenance/v1",
        "count": len(names),
        "names": names,
        "sha256": hashlib.sha256(
            b"compag-curation-science-gpu-archive-provenance-v1\0" + encoded
        ).hexdigest(),
    }


def _validated_demo_gpu_dependencies(
    path: Path,
    *,
    expected_cuda_visibility: Mapping[str, object] | None = None,
) -> dict[str, object]:
    from .public_backend import SCIENCE_GPU_IMPORTS
    from .runtime_lock import (
        SCIENCE_GPU_LOCK_SHA256,
        SCIENCE_GPU_PYTHON,
        SCIENCE_GPU_SAM2_COMMIT,
        SCIENCE_GPU_SAM2_URL,
        SCIENCE_GPU_VERSIONS,
        science_gpu_runtime_policy,
    )

    value = _json_object(path)
    runtime_policy = value.get("gpu_runtime_policy")
    cuda = value.get("cuda")
    xgboost_build = value.get("xgboost_build")
    sam2_extension = value.get("sam2_cuda_extension")
    versions = value.get("versions")
    direct = value.get("sam2_direct_url")
    vcs = direct.get("vcs_info") if isinstance(direct, dict) else None
    implementation = value.get("package_implementation")
    archive_provenance = value.get("archive_provenance")
    digest = value.get("sha256")
    visibility = value.get("cuda_visibility")
    expected_visibility = (
        _demo_cuda_visibility_identity()
        if expected_cuda_visibility is None
        else dict(expected_cuda_visibility)
    )
    _require(
        set(value)
        == {
            "schema",
            "python",
            "implementation",
            "platform",
            "versions",
            "missing",
            "sam2_direct_url",
            "archive_provenance",
            "lock_sha256",
            "lock_resource",
            "package_implementation",
            "sha256",
            "imports",
            "package_installation",
            "gpu_runtime_policy",
            "requested_device",
            "resolved_device",
            "cuda_visibility",
            "cuda",
            "xgboost_build",
            "sam2_cuda_extension",
        }
        and value.get("schema") == "compag-curation-science-gpu-dependencies/v2"
        and value.get("python") == SCIENCE_GPU_PYTHON
        and value.get("implementation") == "CPython"
        and isinstance(value.get("platform"), str)
        and bool(str(value.get("platform", "")).strip())
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        and digest
        == compact_json_sha256(
            {key: item for key, item in value.items() if key != "sha256"}
        )
        and value.get("requested_device") == "cuda"
        and value.get("resolved_device") == "cuda:0"
        and isinstance(visibility, dict)
        and visibility == expected_visibility
        and runtime_policy == science_gpu_runtime_policy()
        and isinstance(cuda, dict)
        and set(cuda)
        == {
            "device_index",
            "device_name",
            "compute_capability",
            "total_memory_bytes",
            "torch_cuda_runtime",
            "cudnn_version",
            "driver_api_version",
        }
        and cuda.get("device_index") == 0
        and isinstance(cuda.get("device_name"), str)
        and bool(str(cuda.get("device_name", "")).strip())
        and isinstance(cuda.get("driver_api_version"), int)
        and not isinstance(cuda.get("driver_api_version"), bool)
        and int(cuda["driver_api_version"]) > 0
        and isinstance(xgboost_build, dict)
        and xgboost_build.get("USE_CUDA") is True
        and isinstance(sam2_extension, dict)
        and sam2_extension.get("schema")
        == "compag-curation-sam2-cuda-extension-attestation/v1"
        and sam2_extension.get("distribution") == "sam-2"
        and sam2_extension.get("distribution_version")
        == SCIENCE_GPU_VERSIONS["sam-2"]
        and sam2_extension.get("module") == "sam2._C"
        and sam2_extension.get("distribution_path") == "sam2/_C.so"
        and isinstance(sam2_extension.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(sam2_extension["sha256"]))
        is not None
        and isinstance(sam2_extension.get("size_bytes"), int)
        and not isinstance(sam2_extension.get("size_bytes"), bool)
        and int(sam2_extension["size_bytes"]) > 0
        and isinstance(sam2_extension.get("probe"), dict)
        and sam2_extension["probe"].get("function")
        == "get_connected_componnets"
        and sam2_extension["probe"].get("cpu_fallback") is False
        and versions == SCIENCE_GPU_VERSIONS
        and value.get("missing") == []
        and value.get("lock_sha256") == SCIENCE_GPU_LOCK_SHA256
        and value.get("lock_resource") == "resources/science_gpu_lock.json"
        and isinstance(direct, dict)
        and direct.get("url") == SCIENCE_GPU_SAM2_URL
        and isinstance(vcs, dict)
        and vcs.get("vcs") == "git"
        and vcs.get("commit_id") == SCIENCE_GPU_SAM2_COMMIT
        and archive_provenance == _expected_demo_gpu_archive_provenance()
        and isinstance(implementation, dict)
        and set(implementation) == {"schema", "sha256", "file_count", "size_bytes"}
        and value.get("imports") == list(SCIENCE_GPU_IMPORTS)
        and value.get("package_installation")
        == "SYS_PREFIX_DISTRIBUTION_NON_EDITABLE_ISOLATED_NO_BYTECODE",
        "real-demo GPU dependency evidence is invalid",
    )
    return value


def _fresh_demo_gpu_environment(
    runtime: Path,
    cuda_visibility: Mapping[str, object],
) -> dict[str, str]:
    from .runtime_lock import (
        confined_accelerator_cache_environment,
        preserved_cuda_visibility_environment,
    )

    try:
        visibility_environment = preserved_cuda_visibility_environment(
            cuda_visibility
        )
    except RuntimeError as exc:
        raise PublicIOError(str(exc)) from exc
    return {
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
        **visibility_environment,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
    }


def _run_fresh_demo_project_operation(
    project: Path,
    run_root: Path,
    staging: Path,
    runner: object | None,
    *,
    operation: str,
    reviewed_labels: Path | None = None,
    cuda_visibility: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Run one GPU project phase in a fresh interpreter and seal its receipt."""

    from .domain.inference import LocalCommandRunner, _require_inference_process_result

    _require(operation in {"pause", "resume"}, "real-demo project operation is invalid")
    if operation == "pause":
        _require(reviewed_labels is None, "pause operation does not accept reviewed labels")
        _require(not run_root.exists() and not run_root.is_symlink(), "real-demo run output already exists")
    else:
        _require(
            reviewed_labels is not None
            and reviewed_labels.is_file()
            and not reviewed_labels.is_symlink()
            and run_root.is_dir()
            and not run_root.is_symlink(),
            "resume operation requires a safe run root and reviewed table",
        )
    selected_runner = LocalCommandRunner() if runner is None else runner
    selected_visibility = (
        _demo_cuda_visibility_identity()
        if cuda_visibility is None
        else dict(cuda_visibility)
    )
    runtime = staging / f"{operation}_process_runtime"
    runtime.mkdir(mode=0o700)
    write_new_json(
        runtime / "ENVIRONMENT_POLICY.json",
        {
            "schema": "compag-curation-real-demo-process-environment/v1",
            "status": "CLOSED_EXPLICIT_ENVIRONMENT",
            "operation": operation,
            "device": "cuda",
            "python_isolated": True,
            "python_no_user_site": True,
            "bytecode_disabled": True,
            "cuda_visibility": selected_visibility,
            "cublas_workspace_config": ":4096:8",
            "cpu_threads": 1,
        },
    )
    argv_parts = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "compag_curation",
        "run",
        "--config",
        str(project / "config.toml"),
    ]
    acceptable_exit_codes: frozenset[int]
    if operation == "pause":
        argv_parts.extend(("--output", str(run_root)))
        acceptable_exit_codes = frozenset({3})
    else:
        assert reviewed_labels is not None
        argv_parts.extend(
            (
                "--resume",
                str(run_root),
                "--review-labels",
                str(reviewed_labels),
            )
        )
        acceptable_exit_codes = frozenset({0})
    argv = tuple(argv_parts)
    environment = _fresh_demo_gpu_environment(runtime, selected_visibility)
    try:
        process = selected_runner.run(
            argv,
            cwd=staging,
            env=environment,
            acceptable_exit_codes=acceptable_exit_codes,
            capture_output=True,
        )
    finally:
        purge_owned_runtime_cache_files(
            runtime,
            retained_files=("ENVIRONMENT_POLICY.json",),
        )
    try:
        process = _require_inference_process_result(
            process,
            argv,
            environment,
            staging,
            acceptable_exit_codes,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        stderr = str(getattr(process, "stderr", ""))
        child_error_type = "UNAVAILABLE"
        stderr_lines = [line for line in stderr.splitlines() if line.strip()]
        if stderr_lines:
            try:
                error_record = strict_json_bytes(
                    stderr_lines[-1].encode("utf-8"),
                    f"real-demo {operation} stderr",
                )
                if isinstance(error_record, dict) and isinstance(
                    error_record.get("error_type"), str
                ):
                    child_error_type = str(error_record["error_type"])
            except PublicIOError:
                pass
        raise PublicIOError(
            f"real-demo {operation} process contract failed: "
            f"return_code={getattr(process, 'returncode', 'INVALID')}; "
            f"child_error_type={child_error_type}; "
            f"stderr_sha256={hashlib.sha256(stderr.encode('utf-8')).hexdigest()}"
        ) from exc
    _require(process.stderr == "", f"real-demo {operation} process emitted stderr")
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    _require(
        len(lines) == 1,
        f"real-demo {operation} process must emit exactly one JSON record",
    )
    try:
        payload = strict_json_bytes(
            lines[0].encode("utf-8"),
            f"real-demo {operation} stdout",
        )
    except PublicIOError as exc:
        raise PublicIOError(
            f"real-demo {operation} process stdout is not JSON"
        ) from exc
    _require(isinstance(payload, dict), f"real-demo {operation} stdout must be an object")
    run_id = payload.get("run_id")
    try:
        canonical_run_id = str(uuid.UUID(run_id)) if isinstance(run_id, str) else None
    except ValueError:
        canonical_run_id = None
    if operation == "pause":
        review_request = run_root / "stages/20_proposals_features/review_request.csv"
        _require(
            set(payload)
            == {
                "schema",
                "status",
                "exit_code",
                "run_id",
                "review_request",
                "review_request_sha256",
                "resume_command",
            }
            and payload.get("schema") == "compag-curation-public-run-pause/v1"
            and payload.get("status") == "PAUSED_FOR_REVIEW"
            and payload.get("exit_code") == 3
            and canonical_run_id == run_id
            and payload.get("review_request")
            == "stages/20_proposals_features/review_request.csv"
            and payload.get("review_request_sha256") == sha256_file(review_request)
            and payload.get("resume_command")
            == "python -I -B -m compag_curation run --config CONFIG --resume RUN_OUTPUT --review-labels REVIEWED_CSV",
            "real-demo pause process evidence is invalid",
        )
    else:
        completion = _json_object(run_root / "FINAL/RUN_COMPLETE.json")
        _require(
            set(payload)
            == {
                "schema",
                "status",
                "run_id",
                "stages",
                "report",
                "model_bundle",
                "fresh_inference",
            }
            and payload.get("schema") == "compag-curation-public-run-completion/v1"
            and payload.get("status") == "PASS"
            and canonical_run_id == run_id
            and payload.get("stages") == 8
            and payload.get("report") == "stages/70_report/report.json"
            and payload.get("model_bundle") == "stages/50_train_bundle/model_bundle"
            and payload.get("fresh_inference") == "stages/60_evaluate/predictions.csv"
            and completion == payload,
            "real-demo resume process evidence is invalid",
        )
    dependencies = _validated_demo_gpu_dependencies(
        run_root / "stages/00_input_inventory/dependency_inventory.json",
        expected_cuda_visibility=selected_visibility,
    )
    receipt = {
        "schema": "compag-curation-real-demo-project-process/v1",
        "status": "PASS",
        "operation": operation,
        "process_isolation": "NEW_PYTHON_INTERPRETER_SAME_OS_NAMESPACE",
        "environment_policy": f"{operation}_process_runtime/ENVIRONMENT_POLICY.json",
        "run_id": run_id,
        "dependency_sha256": dependencies["sha256"],
        "cuda_visibility_sha256": selected_visibility["sha256"],
        "return_code": process.returncode,
        "argv_sha256": compact_json_sha256(list(argv)),
        "stdout_sha256": hashlib.sha256(process.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(process.stderr.encode("utf-8")).hexdigest(),
    }
    return payload, receipt, dependencies


def _verify_fresh_inference_output(
    output: Path,
    *,
    bundle_sha256: str,
    expected_result: Mapping[str, object],
    expected_inputs: list[dict[str, object]],
    process_payload: Mapping[str, object],
) -> dict[str, object]:
    status = _json_object(output / "FINAL_STATUS.json")
    manifest = _json_object(output / "OUTPUT_MANIFEST.json")
    result = _json_object(output / "inference_result.json")
    status_run_id = status.get("run_id")
    try:
        canonical_status_run_id = str(uuid.UUID(status_run_id)) if isinstance(status_run_id, str) else None
    except ValueError:
        canonical_status_run_id = None
    _require(
        set(status)
        == {
            "schema",
            "status",
            "run_id",
            "output_name",
            "publication_condition",
            "bundle_sha256",
            "inference_result_sha256",
        }
        and status.get("schema") == "compag-curation-inference-completion/v1"
        and status.get("status") == "PASS"
        and canonical_status_run_id == status_run_id
        and status.get("output_name") == output.name
        and status.get("publication_condition") == "VALID_ONLY_AT_DECLARED_OUTPUT_NAME_AFTER_ATOMIC_RENAME"
        and status.get("bundle_sha256") == bundle_sha256,
        "fresh-process completion record is invalid",
    )
    _require(
        set(manifest)
        == {"schema", "status", "run_id", "bundle_sha256", "input_images", "members", "members_sha256"}
        and manifest.get("schema") == "compag-curation-inference-output-manifest/v1"
        and manifest.get("status") == "PASS"
        and manifest.get("run_id") == status.get("run_id")
        and manifest.get("bundle_sha256") == bundle_sha256,
        "fresh-process output manifest is invalid",
    )
    observed = manifest_rows(output, excluded={"OUTPUT_MANIFEST.json"})
    output_manifest_sha256 = sha256_file(output / "OUTPUT_MANIFEST.json")
    predictions_sha256 = sha256_file(output / "predictions.csv")
    manifest_inputs = manifest.get("input_images")
    _require(isinstance(manifest_inputs, list), "fresh-process input manifest is invalid")
    input_fields = {
        "filename", "sha256", "size_bytes", "width", "height", "exif_orientation",
        "mode_octal", "uid", "gid", "nlink", "mtime_ns", "ctime_ns",
    }
    _require(
        all(isinstance(row, dict) and set(row) == input_fields for row in manifest_inputs),
        "fresh-process input manifest row closure is invalid",
    )
    input_projection = [
        {
            key: row[key]
            for key in ("filename", "sha256", "size_bytes", "width", "height", "exif_orientation")
        }
        for row in manifest_inputs
    ]
    expected_stdout = {
        **expected_result,
        "run_id": status["run_id"],
        "output_name": output.name,
        "output_manifest_sha256": output_manifest_sha256,
    }
    _require(
        manifest.get("members") == observed
        and manifest.get("members_sha256") == compact_json_sha256(observed)
        and status.get("inference_result_sha256") == sha256_file(output / "inference_result.json")
        and set(result) == set(expected_result)
        and result == expected_result
        and result.get("status") == "PASS"
        and result.get("bundle_sha256") == bundle_sha256
        and result.get("predictions_sha256") == predictions_sha256
        and input_projection == expected_inputs
        and process_payload == expected_stdout,
        "fresh-process inference differs from the completed run",
    )
    return {
        "schema": "compag-curation-real-demo-fresh-process/v1",
        "status": "PASS",
        "process_isolation": "NEW_PYTHON_INTERPRETER_SAME_OS_NAMESPACE",
        "os_input_unavailability_proof": "NOT_CLAIMED_BY_DEMO_ACCEPTANCE_HARNESS_REQUIRED",
        "bundle_sha256": bundle_sha256,
        "predictions_sha256": predictions_sha256,
        "matches_stage60_predictions": True,
        "output_manifest_sha256": output_manifest_sha256,
    }


def _run_fresh_demo_inference(
    project: Path,
    run_root: Path,
    staging: Path,
    runner: object | None,
    *,
    device: str = "cpu",
    cuda_visibility: Mapping[str, object] | None = None,
    captured_dependencies: Mapping[str, object] | None = None,
) -> dict[str, object]:
    from .domain.inference import LocalCommandRunner, _require_inference_process_result
    from .runtime_lock import confined_accelerator_cache_environment

    _require(device in {"cpu", "cuda"}, "fresh demo inference device is unsupported")
    selected_visibility: dict[str, object] | None = None
    visibility_environment: dict[str, str] = {}
    if device == "cuda":
        _require(
            isinstance(captured_dependencies, Mapping),
            "fresh demo GPU inference requires captured run dependencies",
        )
        selected_visibility = (
            _demo_cuda_visibility_identity()
            if cuda_visibility is None
            else dict(cuda_visibility)
        )
        visibility_environment = _fresh_demo_gpu_environment(
            staging / "fresh_process_runtime",
            selected_visibility,
        )
    selected_runner = LocalCommandRunner() if runner is None else runner
    runtime = staging / "fresh_process_runtime"
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
                "cuda_visibility": selected_visibility,
                "cublas_workspace_config": ":4096:8",
            }
            if device == "cuda"
            else {}
        ),
    })
    output = staging / "fresh_process_inference"
    bundle = run_root / "stages/50_train_bundle/model_bundle"
    images = project / "data/inference_images"
    argv = (
        sys.executable,
        "-I",
        "-B",
        "-m",
        "compag_curation",
        "infer",
        "--execute",
        "--images",
        str(images),
        "--bundle",
        str(bundle),
        "--output",
        str(output),
        "--device",
        device,
    )
    environment = {
        "HOME": str(runtime),
        "TMPDIR": str(runtime),
        "XDG_CACHE_HOME": str(runtime),
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
        **visibility_environment,
        **(
            {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
            }
            if device == "cuda"
            else {}
        ),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
    }
    try:
        process = selected_runner.run(
            argv,
            cwd=staging,
            env=environment,
            acceptable_exit_codes=frozenset({0}),
            capture_output=True,
        )
    finally:
        purge_owned_runtime_cache_files(
            runtime,
            retained_files=("ENVIRONMENT_POLICY.json",),
        )
    try:
        process = _require_inference_process_result(process, argv, environment, staging, frozenset({0}))
    except (RuntimeError, TypeError, ValueError) as exc:
        stderr = str(getattr(process, "stderr", ""))
        child_error_type = "UNAVAILABLE"
        stderr_lines = [line for line in stderr.splitlines() if line.strip()]
        if stderr_lines:
            try:
                error_record = strict_json_bytes(stderr_lines[-1].encode("utf-8"), "fresh-process stderr")
                if isinstance(error_record, dict) and isinstance(error_record.get("error_type"), str):
                    child_error_type = str(error_record["error_type"])
            except PublicIOError:
                pass
        raise PublicIOError(
            "fresh-process inference contract failed: "
            f"return_code={getattr(process, 'returncode', 'INVALID')}; "
            f"child_error_type={child_error_type}; "
            f"stderr_sha256={hashlib.sha256(stderr.encode('utf-8')).hexdigest()}"
        ) from exc
    lines = [line for line in str(process.stdout).splitlines() if line.strip()]
    _require(len(lines) == 1, "fresh-process inference must return exactly one structured stdout record")
    try:
        process_payload = strict_json_bytes(lines[-1].encode("utf-8"), "fresh-process stdout")
    except PublicIOError as exc:
        raise PublicIOError("fresh-process inference stdout is not JSON") from exc
    _require(isinstance(process_payload, dict), "fresh-process inference stdout must be an object")
    bundle_sha256 = sha256_file(bundle / "bundle.json")
    expected_result = _json_object(run_root / "stages/60_evaluate/inference_result.json")
    stage_zero = _json_object(run_root / "stages/00_input_inventory/input_inventory.json")
    inspection = stage_zero.get("inspection")
    _require(isinstance(inspection, dict), "completed run input inventory is invalid")
    source_inputs = inspection.get("inference_images")
    _require(isinstance(source_inputs, list) and source_inputs, "completed run has no independent inference inputs")
    expected_inputs = []
    for row in source_inputs:
        _require(isinstance(row, dict), "completed run inference input row is invalid")
        expected_inputs.append(
            {
                key: row[key]
                for key in ("filename", "sha256", "size_bytes", "width", "height", "exif_orientation")
            }
        )
    if expected_result.get("schema") == "compag-curation-canonical-inference/v1":
        from .public_pipeline import (
            _validated_canonical_inference_result,
            _validated_canonical_predictions,
            _validated_canonical_raw_feature_archive,
        )

        expected_result = _validated_canonical_inference_result(
            expected_result,
            bundle_sha256=bundle_sha256,
            expected_inputs=expected_inputs,
            expected_profile=str(expected_result["profile"]),
            expected_device=device,
        )
        _validated_canonical_predictions(
            output / "predictions.csv",
            expected_result,
        )
        _validated_canonical_raw_feature_archive(output, expected_result)
    receipt = _verify_fresh_inference_output(
        output,
        bundle_sha256=bundle_sha256,
        expected_result=expected_result,
        expected_inputs=expected_inputs,
        process_payload=process_payload,
    )
    dependency_receipt: dict[str, object] = {}
    if device == "cuda":
        child_environment = _json_object(output / "environment.json")
        _require(
            set(child_environment) == {"schema", "run_id", "dependencies"}
            and child_environment.get("schema")
            == "compag-curation-inference-environment/v1"
            and child_environment.get("run_id")
            == _json_object(output / "FINAL_STATUS.json").get("run_id")
            and isinstance(child_environment.get("dependencies"), dict),
            "fresh-process inference dependency evidence is invalid",
        )
        dependencies = child_environment["dependencies"]
        assert isinstance(dependencies, dict)
        dependency_evidence = staging / "fresh_process_inference_dependencies.json"
        write_new_json(dependency_evidence, dependencies)
        assert selected_visibility is not None
        validated_dependencies = _validated_demo_gpu_dependencies(
            dependency_evidence,
            expected_cuda_visibility=selected_visibility,
        )
        _require(
            validated_dependencies == dict(captured_dependencies or {}),
            "fresh-process inference dependency receipt differs from completed run",
        )
        dependency_receipt["dependency_sha256"] = validated_dependencies["sha256"]
    return {
        **receipt,
        "return_code": int(process.returncode),
        **dependency_receipt,
        "argv_sha256": compact_json_sha256(list(argv)),
        "stdout_sha256": hashlib.sha256(str(process.stdout).encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(str(process.stderr).encode("utf-8")).hexdigest(),
    }


def run_quick_demo(output: Path, *, invocation: Mapping[str, object] | None = None) -> dict[str, object]:
    from .public_backend import require_quick_dependencies

    dependencies = require_quick_dependencies()
    final, staging, lock_fd, run_id = _demo_output(output, "quick", invocation)
    try:
        import cv2
        import numpy as np

        from .proposals.sam2_pipeline.gate_core import cell_of_point, compute_features, detect_grid_lines, yellow_bg_stats
        write_new_json(staging / "environment.json", {
            "schema": "compag-curation-demo-environment/v1",
            "run_id": run_id,
            "dependencies": dependencies,
        })

        fixtures = staging / "generated_fixtures"
        fixtures.mkdir(mode=0o755)
        proposal_rows: list[dict[str, str]] = []
        feature_rows: list[dict[str, object]] = []
        for group_index in range(6):
            group = f"synthetic-group-{group_index + 1:02d}"
            filename = f"{group}__card.ppm"
            image_payload = _ppm(512, 512, group_index)
            image_path = fixtures / filename
            write_new_bytes(image_path, image_payload)
            image_sha = hashlib.sha256(image_payload).hexdigest()
            image_id = hashlib.sha256(b"compag-quick-image-v1\0" + image_sha.encode("ascii")).hexdigest()
            bgr = cv2.imdecode(np.frombuffer(image_payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            _require(bgr is not None, "quick-demo PPM decode failed")
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            bg_a, bg_b, _sample = yellow_bg_stats(bgr)
            row_lines, col_lines = detect_grid_lines(bgr)
            yy, xx = np.ogrid[:512, :512]
            masks = (
                ((xx - (80 + 17 * group_index)) ** 2 + (yy - (110 + 11 * group_index)) ** 2 <= 20 ** 2),
                ((xx - 420) ** 2 + (yy - 80) ** 2 <= 16 ** 2),
            )
            for proposal_index, mask in enumerate(masks):
                features = compute_features(mask.astype(np.uint8) * 255, lab, mode="balanced")
                _require(bool(features), "quick-demo feature extraction returned no values")
                features["delta_a"] = float(bg_a - float(features["mean_a"]))
                features["delta_b"] = float(float(features["mean_b"]) - bg_b)
                grid_r, grid_c = cell_of_point(float(features["cx"]), float(features["cy"]), row_lines, col_lines)
                features["grid_r_norm"] = float(grid_r / max(1, len(row_lines) - 1))
                features["grid_c_norm"] = float(grid_c / max(1, len(col_lines) - 1))
                features["pred_iou"] = 0.95 - proposal_index * 0.1
                features["stability"] = 0.96 - proposal_index * 0.1
                selected = {name: float(features[name]) for name in PUBLIC_FEATURE_ORDER}
                _require(all(math.isfinite(value) for value in selected.values()), "quick-demo feature is nonfinite")
                identity = hashlib.sha256(canonical_json_bytes({
                    "schema": "compag-curation-quick-proposal/v1",
                    "image_sha256": image_sha,
                    "group_id": group,
                    "proposal_index": proposal_index,
                    "features": {name: float.hex(value) for name, value in selected.items()},
                })).hexdigest()
                proposal_rows.append({
                    "proposal_id": identity,
                    "proposal_sha256": identity,
                    "image_id": image_id,
                    "image_sha256": image_sha,
                    "group_id": group,
                })
                feature_rows.append({"proposal_id": identity, "group_id": group, **selected})
        write_review_export(proposal_rows, staging / "review_request.csv")
        _reviewed_copy(staging / "review_request.csv", staging / "reviewed_fixture.csv")
        reviewed, review_result = validate_review_table(staging / "review_request.csv", staging / "reviewed_fixture.csv")
        normalized = write_review_import(reviewed, staging / "reviewed.csv")
        feature_fields = ("proposal_id", "group_id", *PUBLIC_FEATURE_ORDER)
        _write_csv_file(
            staging / "features.csv",
            feature_fields,
            ({name: row[name] for name in feature_fields} for row in sorted(feature_rows, key=lambda item: str(item["proposal_id"]))),
        )
        write_new_json(staging / "report.json", {
            "schema": "compag-curation-quick-demo/v1",
            "status": "PASS",
            "profile": "quick",
            "generated_data_only": True,
            "groups": 6,
            "proposals": len(proposal_rows),
            "feature_order": list(PUBLIC_FEATURE_ORDER),
            "feature_order_sha256": FEATURE_ORDER_SHA256,
            "review_import": review_result,
            "normalized_review_sha256": normalized["sha256"],
            "sam2_executed": False,
            "xgboost_executed": False,
            "paper_result_reproduction": "NOT_CLAIMED",
        })
        return _seal_demo(final, staging, "quick", run_id)
    except BaseException as exc:
        _record_demo_failure(staging, run_id, "quick", exc)
        raise
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def run_real_demo(
    output: Path,
    asset_root: Path,
    *,
    execution_profile: str = CANONICAL_GPU_PROFILE,
    invocation: Mapping[str, object] | None = None,
    runner: object | None = None,
) -> dict[str, object]:
    _require(
        execution_profile in GPU_EXECUTION_PROFILES,
        "real demo execution profile must be Full or Lite",
    )
    selected_assets = _REAL_DEMO_ASSETS_BY_PROFILE[execution_profile]
    asset_root = asset_root.absolute()
    asset_info = asset_root.lstat()
    _require(
        stat.S_ISDIR(asset_info.st_mode) and not asset_root.is_symlink() and asset_root == asset_root.resolve(strict=True),
        "real-demo asset root is unsafe or contains a symlink component",
    )
    candidate = output.absolute()
    candidate = candidate.parent.resolve(strict=True) / candidate.name
    _require(
        candidate != asset_root and asset_root not in candidate.parents and candidate not in asset_root.parents,
        "real-demo output overlaps the read-only asset root",
    )
    assets = {}
    for asset_id, filename in selected_assets:
        verified = verify_asset(asset_root, asset_id)
        _require(verified.get("filename") == filename, "real-demo asset filename differs from the canonical registry")
        assets[asset_id] = verified
    final, staging, lock_fd, run_id = _demo_output(output, "real", invocation)
    try:
        cuda_visibility = _demo_cuda_visibility_identity()
        write_new_json(staging / "asset_inventory.json", {
            "schema": "compag-curation-demo-assets/v1",
            "run_id": run_id,
            "assets": {asset_id: {"sha256": value["sha256"], "size_bytes": value["size_bytes"]} for asset_id, value in sorted(assets.items())},
        })
        project = staging / "project"
        initialize_project(project, profile=execution_profile)
        generated_training_groups = 8
        for group_index in range(generated_training_groups):
            write_new_bytes(project / "data/images" / f"synthetic-group-{group_index + 1:02d}__train.ppm", _ppm(512, 512, group_index))
        write_new_bytes(project / "data/inference_images/synthetic-heldout__card.ppm", _ppm(512, 512, 19))
        for asset_id, destination in selected_assets:
            copy_new(asset_root / str(assets[asset_id]["filename"]), project / "assets" / destination)
        run_root = staging / "run"
        pause, pause_process, dependencies = _run_fresh_demo_project_operation(
            project,
            run_root,
            staging,
            runner,
            operation="pause",
            cuda_visibility=cuda_visibility,
        )
        pause_rc = int(pause_process["return_code"])
        _require(
            pause_rc == 3 and pause.get("status") == "PAUSED_FOR_REVIEW",
            "real demo did not reach the review boundary",
        )
        reviewed_path = staging / "reviewed_fixture.csv"
        _reviewed_copy(
            run_root / "stages/20_proposals_features/review_request.csv",
            reviewed_path,
            canonical_actions=True,
        )
        complete, resume_process, resumed_dependencies = _run_fresh_demo_project_operation(
            project,
            run_root,
            staging,
            runner,
            operation="resume",
            reviewed_labels=reviewed_path,
            cuda_visibility=cuda_visibility,
        )
        complete_rc = int(resume_process["return_code"])
        _require(complete_rc == 0 and complete.get("status") == "PASS", "real demo did not complete")
        _require(
            resumed_dependencies == dependencies,
            "real-demo dependency identity changed between pause and resume",
        )
        fresh_process = _run_fresh_demo_inference(
            project,
            run_root,
            staging,
            runner,
            device="cuda",
            cuda_visibility=cuda_visibility,
            captured_dependencies=dependencies,
        )
        _require(
            fresh_process.get("dependency_sha256") == dependencies["sha256"],
            "real-demo dependency identity changed for fresh inference",
        )
        write_new_json(staging / "environment.json", {
            "schema": "compag-curation-demo-environment/v2",
            "run_id": run_id,
            "dependencies": dependencies,
            "gpu_processes": [pause_process, resume_process, fresh_process],
        })
        write_new_json(staging / "REAL_DEMO_RESULT.json", {
            "schema": "compag-curation-real-demo/v1",
            "status": "PASS",
            "profile": execution_profile,
            "demo_classification": "SYNTHETIC_REAL_BACKEND_DEMO",
            "generated_data_only": True,
            "review_fixture": "DETERMINISTIC_SYNTHETIC_PRE_REVIEWED_NON_HUMAN",
            "review_contract": "CANONICAL_ACTION_WEIGHTED_V2",
            "generated_training_groups": generated_training_groups,
            "pause_exit_code": pause_rc,
            "run_id": complete["run_id"],
            "gpu_project_processes": [pause_process, resume_process],
            "asset_sha256": {asset_id: value["sha256"] for asset_id, value in sorted(assets.items())},
            "fresh_process_inference": fresh_process,
            "paper_result_reproduction": "NOT_CLAIMED",
        })
        return _seal_demo(final, staging, "real", run_id)
    except BaseException as exc:
        _record_demo_failure(staging, run_id, "real", exc)
        raise
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
