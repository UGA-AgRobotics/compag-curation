"""Strict scientific backend used only by explicit public execution commands."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.metadata
import importlib
import io
import json
import math
import os
import platform
import re
import stat
import sys
import sysconfig
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .model_bundle import BundleWriteRequest, VerifiedBundle, feature_matrix, verify_model_bundle, write_model_bundle
from .public_config import FEATURE_ORDER_SHA256, MAX_IMAGE_FILE_BYTES, PUBLIC_FEATURE_ORDER, PublicProjectConfig, group_id_from_name, image_files
from .public_io import PublicIOError, canonical_json_bytes, compact_json_sha256, hash_file_snapshot, sha256_file, stable_file, verified_file_path, write_new_bytes, write_new_json
from .runtime_lock import (
    SCIENCE_CPU_LOCK_SHA256,
    SCIENCE_CPU_PYTHON,
    SCIENCE_CPU_SAM2_COMMIT,
    SCIENCE_CPU_SAM2_URL,
    SCIENCE_CPU_VERSIONS,
    SCIENCE_GPU_ARCHIVE_COUNT,
    SCIENCE_GPU_CUDA_RUNTIME,
    SCIENCE_GPU_LOCK_SHA256,
    SCIENCE_GPU_PYTHON,
    SCIENCE_GPU_SAM2_COMMIT,
    SCIENCE_GPU_SAM2_URL,
    SCIENCE_GPU_VERSIONS,
    cuda_visibility_identity,
    package_implementation_identity,
    requested_cuda_device,
    science_cpu_lock_identity,
    science_cpu_runtime_policy,
    science_gpu_lock_identity,
    science_gpu_runtime_policy,
)


PROPOSAL_COLUMNS = (
    "proposal_id", "proposal_sha256", "image_id", "image_sha256", "group_id",
    "tile_name", "tile_sha256", "tile_x", "tile_y", "mask_sha256",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h", "poly",
    *PUBLIC_FEATURE_ORDER,
)
TILE_COLUMNS = (
    "tile_name", "image_id", "image_sha256", "image_name", "group_id", "tile_sha256",
    "x", "y", "crop_w", "crop_h", "orig_w", "orig_h",
)
PREDICTION_COLUMNS = (
    "proposal_id", "image_id", "group_id", "x", "y", "w", "h",
    "probability", "prediction", "kept",
)
SAM2_SOURCE_COMMIT = SCIENCE_CPU_SAM2_COMMIT
SAM_SCORE_DECIMAL_PLACES = 6
SAM_SCORE_QUANTUM = Decimal("0.000001")
EXPECTED_SCIENCE_VERSIONS = SCIENCE_CPU_VERSIONS
EXPECTED_SCIENCE_GPU_VERSIONS = SCIENCE_GPU_VERSIONS
SCIENCE_IMPORTS = (
    "numpy", "cv2", "torch", "torchvision", "sam2", "hydra", "omegaconf",
    "antlr4", "iopath", "portalocker", "PIL", "yaml", "tqdm", "scipy",
    "sklearn", "imblearn", "xgboost", "filelock", "fsspec", "jinja2", "joblib",
    "markupsafe", "mpmath", "networkx", "packaging", "setuptools", "sympy",
    "threadpoolctl", "typing_extensions", "contourpy", "cycler", "fontTools",
    "kiwisolver", "matplotlib", "pandas", "pyparsing", "dateutil", "pytz", "six", "tabulate",
    "tzdata",
)
SCIENCE_GPU_IMPORTS = (*SCIENCE_IMPORTS, "cupy")
EXPECTED_QUICK_VERSIONS = {
    "compag-curation": "1.9.4rc10",
    "numpy": "2.0.2",
    "opencv-python": "4.12.0.88",
}
_SCIENCE_THREADPOOL_LIMITER: object | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON field: {key}")
        parsed[key] = value
    return parsed


def _filesystem_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _science_gpu_distribution_roots() -> tuple[Path, ...]:
    """Return stable canonical install roots for the active locked interpreter."""

    try:
        prefix_path = Path(sys.prefix)
        _require(prefix_path.is_absolute(), "science-gpu installation prefix is invalid")
        prefix_before = prefix_path.lstat()
        _require(
            stat.S_ISDIR(prefix_before.st_mode),
            "science-gpu installation prefix is not a directory",
        )
        prefix = prefix_path.resolve(strict=True)
        _require(
            prefix == prefix_path.absolute(),
            "science-gpu installation prefix has a symlinked path component",
        )
        configured = (
            sysconfig.get_path("purelib"),
            sysconfig.get_path("platlib"),
        )
        roots: list[Path] = []
        identities: set[tuple[str, int, int]] = set()
        for raw_root in configured:
            _require(
                isinstance(raw_root, str) and bool(raw_root),
                "science-gpu installation root is invalid",
            )
            root_path = Path(raw_root)
            _require(
                root_path.is_absolute(),
                "science-gpu installation root is not absolute",
            )
            before = root_path.lstat()
            _require(
                stat.S_ISDIR(before.st_mode),
                "science-gpu installation root is not a directory",
            )
            root = root_path.resolve(strict=True)
            _require(
                root == root_path.absolute()
                and root != prefix
                and root.is_relative_to(prefix),
                "science-gpu installation root is outside the active environment",
            )
            after = root_path.lstat()
            _require(
                _filesystem_identity(before) == _filesystem_identity(after),
                "science-gpu installation root changed during validation",
            )
            identity = (str(root), before.st_dev, before.st_ino)
            if identity not in identities:
                identities.add(identity)
                roots.append(root)
        prefix_after = prefix_path.lstat()
        _require(
            _filesystem_identity(prefix_before) == _filesystem_identity(prefix_after),
            "science-gpu installation prefix changed during validation",
        )
    except PublicIOError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PublicIOError(
            "science-gpu installation roots could not be attested"
        ) from exc
    _require(bool(roots), "science-gpu installation roots are missing")
    return tuple(roots)


def _science_gpu_archive_installation_rows(
    expected_names: Sequence[str],
) -> dict[str, list[dict[str, str]]]:
    """Read top-level archive distributions from attested environment roots."""

    roots = _science_gpu_distribution_roots()
    installed: dict[str, dict[tuple[str, int, int], dict[str, str]]] = {
        name: {} for name in expected_names
    }
    try:
        root_identities = {
            root: _filesystem_identity(root.lstat())
            for root in roots
        }
        distributions = list(
            importlib.metadata.distributions(path=[str(root) for root in roots])
        )
        for distribution in distributions:
            raw_name = distribution.metadata.get("Name")
            if not isinstance(raw_name, str) or not raw_name:
                continue
            canonical = re.sub(r"[-_.]+", "-", raw_name).lower()
            if canonical not in installed:
                continue
            raw_metadata_path = getattr(distribution, "_path", None)
            _require(
                isinstance(raw_metadata_path, (str, os.PathLike)),
                f"science-gpu archive metadata path is invalid for {canonical}",
            )
            metadata_path = Path(raw_metadata_path)
            _require(
                metadata_path.is_absolute(),
                f"science-gpu archive metadata path is not absolute for {canonical}",
            )
            before = metadata_path.lstat()
            _require(
                stat.S_ISDIR(before.st_mode),
                f"science-gpu archive metadata path is unsafe for {canonical}",
            )
            resolved = metadata_path.resolve(strict=True)
            _require(
                resolved == metadata_path.absolute()
                and resolved.parent in roots
                and resolved.name.endswith(".dist-info"),
                f"science-gpu archive metadata is outside an installation root for {canonical}",
            )
            installed_version = distribution.version
            direct_text = distribution.read_text("direct_url.json")
            repeated_name = distribution.metadata.get("Name")
            after = metadata_path.lstat()
            _require(
                isinstance(repeated_name, str)
                and re.sub(r"[-_.]+", "-", repeated_name).lower() == canonical
                and _filesystem_identity(before) == _filesystem_identity(after),
                f"science-gpu archive metadata changed during validation for {canonical}",
            )
            _require(
                isinstance(installed_version, str),
                f"science-gpu archive distribution version is invalid for {canonical}",
            )
            _require(
                direct_text is None or isinstance(direct_text, str),
                f"science-gpu archive direct_url.json is invalid for {canonical}",
            )
            physical_identity = (str(resolved), before.st_dev, before.st_ino)
            row = {
                "version": installed_version,
                "direct_url": direct_text or "",
            }
            previous = installed[canonical].get(physical_identity)
            _require(
                previous is None or previous == row,
                f"science-gpu archive metadata is inconsistent for {canonical}",
            )
            installed[canonical][physical_identity] = row
        _require(
            all(
                _filesystem_identity(root.lstat()) == identity
                for root, identity in root_identities.items()
            ),
            "science-gpu installation roots changed during archive enumeration",
        )
    except PublicIOError:
        raise
    except (AttributeError, OSError, TypeError, UnicodeError, ValueError) as exc:
        raise PublicIOError(
            "science-gpu archive distribution metadata could not be enumerated"
        ) from exc
    return {
        name: list(installed[name].values())
        for name in expected_names
    }


def _science_gpu_archive_provenance_identity(
    archives: object,
) -> dict[str, object]:
    """Validate exact PEP 610 archive origins without importing science code."""

    _require(
        isinstance(archives, Mapping) and len(archives) == SCIENCE_GPU_ARCHIVE_COUNT,
        "science-gpu archive provenance lock count is invalid",
    )
    expected_names = sorted(str(name) for name in archives)
    _require(
        len(expected_names) == SCIENCE_GPU_ARCHIVE_COUNT
        and all(re.sub(r"[-_.]+", "-", name).lower() == name for name in expected_names),
        "science-gpu archive provenance lock names are invalid",
    )
    expected_rows: dict[str, dict[str, str]] = {}
    for name in expected_names:
        expected = archives.get(name)
        _require(
            isinstance(expected, dict)
            and set(expected) == {"sha256", "url", "version"}
            and isinstance(expected.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(expected["sha256"])) is not None
            and isinstance(expected.get("url"), str)
            and isinstance(expected.get("version"), str),
            f"science-gpu archive provenance lock row is invalid for {name}",
        )
        expected_rows[name] = {
            "sha256": str(expected["sha256"]),
            "url": str(expected["url"]),
            "version": str(expected["version"]),
        }

    installed = _science_gpu_archive_installation_rows(expected_names)

    verified: dict[str, dict[str, str]] = {}
    for name in expected_names:
        candidates = installed[name]
        _require(
            len(candidates) == 1,
            f"science-gpu archive distribution is missing or duplicated for {name}",
        )
        distribution = candidates[0]
        expected = expected_rows[name]
        installed_version = distribution["version"]
        direct_text = distribution["direct_url"]
        _require(
            isinstance(installed_version, str)
            and installed_version == expected["version"],
            f"science-gpu archive distribution version differs from the lock for {name}",
        )
        _require(
            isinstance(direct_text, str) and bool(direct_text),
            f"science-gpu archive direct_url.json is missing for {name}",
        )
        try:
            direct = json.loads(
                direct_text,
                object_pairs_hook=_json_object_without_duplicate_keys,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise PublicIOError(
                f"science-gpu archive direct_url.json is malformed for {name}"
            ) from exc
        _require(
            isinstance(direct, dict)
            and set(direct) == {"archive_info", "url"}
            and isinstance(direct.get("archive_info"), dict),
            f"science-gpu provenance is not an archive direct URL for {name}",
        )
        _require(
            direct.get("url") == expected["url"],
            f"science-gpu archive URL differs from the lock for {name}",
        )
        archive_info = direct["archive_info"]
        assert isinstance(archive_info, dict)
        _require(
            bool(archive_info)
            and set(archive_info).issubset({"hash", "hashes"}),
            f"science-gpu archive hash metadata is malformed for {name}",
        )
        modern_present = "hashes" in archive_info
        legacy_present = "hash" in archive_info
        _require(
            modern_present or legacy_present,
            f"science-gpu archive SHA256 is missing for {name}",
        )
        if modern_present:
            hashes = archive_info["hashes"]
            _require(
                isinstance(hashes, dict)
                and set(hashes) == {"sha256"}
                and hashes.get("sha256") == expected["sha256"],
                f"science-gpu archive SHA256 differs from the lock for {name}",
            )
        if legacy_present:
            _require(
                archive_info["hash"] == f"sha256={expected['sha256']}",
                f"science-gpu archive SHA256 differs from the lock for {name}",
            )
        verified[name] = dict(expected)

    closure = {
        "schema": "compag-curation-science-gpu-archive-set/v1",
        "archives": verified,
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
        "count": len(expected_names),
        "names": expected_names,
        "sha256": hashlib.sha256(
            b"compag-curation-science-gpu-archive-provenance-v1\0" + encoded
        ).hexdigest(),
    }


def _canonical_sam_score(value: object, role: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PublicIOError(f"SAM2 returned a nonnumeric {role}") from exc
    _require(math.isfinite(number), f"SAM2 returned a nonfinite {role}")
    _require(0.0 <= number <= 1.0, f"SAM2 returned {role} outside [0,1]")
    canonical = float(Decimal(str(number)).quantize(SAM_SCORE_QUANTUM, rounding=ROUND_HALF_EVEN))
    return 0.0 if canonical == 0.0 else canonical


def _format_feature_value(name: str, value: float) -> str:
    if name in {"pred_iou", "stability"}:
        return format(value, f".{SAM_SCORE_DECIMAL_PLACES}f")
    return format(value, ".17g")


def _require_supported_python() -> None:
    _require(platform.python_implementation() == "CPython", "supported runtime requires CPython")
    _require(platform.python_version_tuple()[:2] == ("3", "12"), "supported runtime requires CPython 3.12")
    _require(platform.system() == "Linux" and platform.machine() in {"x86_64", "AMD64"}, "supported runtime requires Linux x86-64")


def _platform_identity() -> str:
    value = os.uname()
    return f"{value.sysname}-{value.release}-{value.machine}"


def require_base_dependencies() -> dict[str, object]:
    _require_supported_python()
    try:
        version = importlib.metadata.version("compag-curation")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PublicIOError("compag-curation distribution metadata is missing") from exc
    _require(version == "1.9.4rc10", "installed compag-curation version differs from 1.9.4rc10")
    return {
        "schema": "compag-curation-base-dependencies/v1",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": _platform_identity(),
        "versions": {"compag-curation": version},
    }


def require_quick_dependencies() -> dict[str, object]:
    _require_supported_python()
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in EXPECTED_QUICK_VERSIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    _require(not missing, f"quick dependencies are missing: {', '.join(sorted(missing))}")
    _require(versions == EXPECTED_QUICK_VERSIONS, "installed quick dependency versions differ from the supported lock")
    for name in ("numpy", "cv2"):
        importlib.import_module(name)
    payload = {
        "schema": "compag-curation-quick-dependencies/v1",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": _platform_identity(),
        "versions": versions,
        "imports": ["numpy", "cv2"],
    }
    payload["sha256"] = compact_json_sha256(payload)
    return payload


def dependency_identity() -> dict[str, object]:
    lock = science_cpu_lock_identity()
    implementation_identity = package_implementation_identity()
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in EXPECTED_SCIENCE_VERSIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    sam2_direct: dict[str, object] | None = None
    if "sam-2" not in missing:
        direct_text = importlib.metadata.distribution("sam-2").read_text("direct_url.json")
        if direct_text is not None:
            try:
                parsed = json.loads(direct_text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                sam2_direct = parsed
    payload = {
        "schema": "compag-curation-science-cpu-dependencies/v1",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": _platform_identity(),
        "versions": versions,
        "missing": missing,
        "sam2_direct_url": sam2_direct,
        "lock_sha256": lock["sha256"],
        "lock_resource": lock["resource"],
        "package_implementation": implementation_identity,
    }
    payload["sha256"] = compact_json_sha256(payload)
    return payload


def gpu_dependency_identity() -> dict[str, object]:
    """Return a hash-bound identity for the v1.9.3 shared Lite/Full science-gpu closure."""

    lock = science_gpu_lock_identity()
    archive_provenance = _science_gpu_archive_provenance_identity(
        lock.get("archives")
    )
    implementation_identity = package_implementation_identity()
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in EXPECTED_SCIENCE_GPU_VERSIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    sam2_direct: dict[str, object] | None = None
    if "sam-2" not in missing:
        direct_text = importlib.metadata.distribution("sam-2").read_text("direct_url.json")
        if direct_text is not None:
            try:
                parsed = json.loads(direct_text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                sam2_direct = parsed
    payload = {
        "schema": "compag-curation-science-gpu-dependencies/v1",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": _platform_identity(),
        "versions": versions,
        "missing": missing,
        "sam2_direct_url": sam2_direct,
        "archive_provenance": archive_provenance,
        "lock_sha256": lock["sha256"],
        "lock_resource": lock["resource"],
        "package_implementation": implementation_identity,
    }
    payload["sha256"] = compact_json_sha256(payload)
    return payload


def _enforce_science_cpu_runtime(
    policy: Mapping[str, object],
    *,
    torch_module: object,
    cv2_module: object,
    threadpoolctl_module: object,
) -> dict[str, object]:
    global _SCIENCE_THREADPOOL_LIMITER

    try:
        torch_module.set_num_threads(1)
        if torch_module.get_num_interop_threads() != 1:
            torch_module.set_num_interop_threads(1)
        torch_module.use_deterministic_algorithms(True)
        cv2_module.setNumThreads(1)
        cv2_module.ocl.setUseOpenCL(False)
        _SCIENCE_THREADPOOL_LIMITER = threadpoolctl_module.threadpool_limits(limits=1)
        native_pools = threadpoolctl_module.threadpool_info()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise PublicIOError("science-cpu numerical runtime policy could not be applied") from exc
    _require(torch_module.get_num_threads() == 1, "science-cpu Torch intra-op thread limit was not applied")
    _require(torch_module.get_num_interop_threads() == 1, "science-cpu Torch inter-op thread limit was not applied")
    _require(
        torch_module.are_deterministic_algorithms_enabled(),
        "science-cpu deterministic Torch algorithms were not enabled",
    )
    _require(cv2_module.getNumThreads() == 1, "science-cpu OpenCV thread limit was not applied")
    _require(not cv2_module.ocl.useOpenCL(), "science-cpu OpenCV OpenCL disablement was not applied")
    _require(isinstance(native_pools, list) and native_pools, "science-cpu native thread pools were not observable")
    _require(
        all(
            isinstance(row, dict)
            and isinstance(row.get("num_threads"), int)
            and not isinstance(row.get("num_threads"), bool)
            and row["num_threads"] == 1
            for row in native_pools
        ),
        "science-cpu native thread-pool limit was not applied",
    )
    return dict(policy)


def _set_science_cpu_environment(policy: Mapping[str, object]) -> None:
    environment = policy.get("environment")
    _require(isinstance(environment, dict), "science-cpu runtime environment policy is invalid")
    for name, value in environment.items():
        _require(isinstance(name, str) and isinstance(value, str), "science-cpu runtime environment policy is invalid")
        os.environ[name] = value


def _set_science_gpu_environment(policy: Mapping[str, object]) -> None:
    """Set deterministic CUDA variables before Torch creates a CUDA context."""

    try:
        cuda_visibility_identity()
    except RuntimeError as exc:
        raise PublicIOError(str(exc)) from exc
    loaded_torch = sys.modules.get("torch")
    if loaded_torch is not None:
        try:
            initialized = bool(loaded_torch.cuda.is_initialized())
        except AttributeError as exc:
            raise PublicIOError("loaded Torch module cannot report CUDA initialization state") from exc
        _require(
            not initialized,
            "science-gpu preflight must run before Torch initializes a CUDA context",
        )
    environment = policy.get("environment")
    _require(isinstance(environment, dict), "science-gpu runtime environment policy is invalid")
    for name, value in environment.items():
        _require(isinstance(name, str) and isinstance(value, str), "science-gpu runtime environment policy is invalid")
        os.environ[name] = value


def _xgboost_cuda_build_identity(xgboost_module: object) -> dict[str, object]:
    try:
        raw = xgboost_module.build_info()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise PublicIOError("science-gpu could not inspect the XGBoost build") from exc
    _require(isinstance(raw, dict), "science-gpu XGBoost build information is invalid")
    _require(raw.get("USE_CUDA") is True, "science-gpu requires an XGBoost build with USE_CUDA=true")
    recorded: dict[str, object] = {}
    for name in (
        "CUDA_VERSION",
        "USE_CUDA",
        "USE_DLOPEN_NCCL",
        "USE_NCCL",
        "USE_OPENMP",
        "USE_RMM",
    ):
        value = raw.get(name)
        if isinstance(value, (bool, int, str)) and not isinstance(value, float):
            recorded[name] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(part, (bool, int, str)) and not isinstance(part, float)
            for part in value
        ):
            recorded[name] = list(value)
    return recorded


def _sam2_cuda_extension_attestation(
    *,
    torch_module: object,
    extension_module: object,
    distribution: object,
    resolved_device: str,
    device_index: int,
) -> dict[str, object]:
    """Hash and execute the installed SAM2 CUDA extension without fallback."""

    try:
        version = distribution.version
        files = distribution.files
    except AttributeError as exc:
        raise PublicIOError("science-gpu SAM2 distribution metadata is invalid") from exc
    _require(
        version == SCIENCE_GPU_VERSIONS["sam-2"] and files is not None,
        "science-gpu SAM2 extension distribution identity is invalid",
    )
    candidates: list[object] = []
    for entry in files:
        relative = Path(str(entry))
        if not relative.is_absolute() and relative.parts == ("sam2", "_C.so"):
            candidates.append(entry)
    _require(
        len(candidates) == 1,
        "science-gpu requires exactly one installed sam2._C extension",
    )
    relative_entry = str(candidates[0])
    try:
        located = Path(distribution.locate_file(candidates[0])).absolute()
        module_file = Path(extension_module.__file__).absolute()
        located_info = located.lstat()
    except (AttributeError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise PublicIOError("science-gpu SAM2 extension path is invalid") from exc
    _require(
        located == module_file
        and located == located.resolve(strict=True)
        and module_file == module_file.resolve(strict=True)
        and located.resolve(strict=True) == module_file.resolve(strict=True)
        and stat.S_ISREG(located_info.st_mode)
        and not stat.S_ISLNK(located_info.st_mode)
        and located_info.st_nlink == 1
        and located_info.st_size > 0,
        "science-gpu SAM2 extension is not a regular single-link installed distribution file",
    )
    _require(
        callable(getattr(extension_module, "get_connected_componnets", None)),
        "science-gpu SAM2 extension lacks callable get_connected_componnets",
    )
    try:
        extension_sha256, extension_before = hash_file_snapshot(located)
        probe_input = torch_module.zeros(
            (1, 1, 4, 4),
            dtype=torch_module.uint8,
            device=resolved_device,
        )
        _require(
            bool(probe_input.is_cuda)
            and probe_input.device.type == "cuda"
            and probe_input.device.index == device_index
            and probe_input.dtype == torch_module.uint8,
            "science-gpu SAM2 extension probe input did not stay on CUDA uint8",
        )
        outputs = extension_module.get_connected_componnets(probe_input)
        _require(
            isinstance(outputs, (list, tuple)) and len(outputs) == 2,
            "science-gpu SAM2 extension probe returned an invalid result",
        )
        output_rows = []
        for output in outputs:
            shape = tuple(output.shape)
            _require(
                bool(output.is_cuda)
                and output.device.type == "cuda"
                and output.device.index == device_index
                and output.dtype == torch_module.int32
                and shape == (1, 1, 4, 4)
                and int(torch_module.count_nonzero(output).item()) == 0,
                "science-gpu SAM2 extension probe did not return CUDA int32 outputs",
            )
            output_rows.append(
                {
                    "shape": list(shape),
                    "dtype": "int32",
                    "device": resolved_device,
                    "nonzero": 0,
                }
            )
        torch_module.cuda.synchronize(device_index)
        extension_after_sha256, extension_after = hash_file_snapshot(located)
    except PublicIOError:
        raise
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PublicIOError(
            "science-gpu SAM2 compiled CUDA extension probe failed; CPU fallback is disabled"
        ) from exc
    _require(
        extension_after_sha256 == extension_sha256
        and extension_after.st_size == extension_before.st_size
        and extension_after.st_ino == extension_before.st_ino
        and extension_after.st_dev == extension_before.st_dev,
        "science-gpu SAM2 extension changed during attestation",
    )
    return {
        "schema": "compag-curation-sam2-cuda-extension-attestation/v1",
        "distribution": "sam-2",
        "distribution_version": version,
        "module": "sam2._C",
        "distribution_path": relative_entry,
        "sha256": extension_sha256,
        "size_bytes": extension_before.st_size,
        "probe": {
            "function": "get_connected_componnets",
            "input": {
                "shape": [1, 1, 4, 4],
                "dtype": "uint8",
                "device": resolved_device,
                "nonzero": 0,
            },
            "outputs": output_rows,
            "cpu_fallback": False,
        },
    }


def _enforce_science_gpu_runtime(
    policy: Mapping[str, object],
    *,
    requested_device: str,
    torch_module: object,
    xgboost_module: object,
    cupy_module: object,
    sam2_extension_module: object,
    sam2_distribution: object,
    cv2_module: object,
    threadpoolctl_module: object,
    apply_policy: bool = True,
) -> dict[str, object]:
    """Apply or revalidate the CUDA FP32 policy and prove GPU execution."""

    global _SCIENCE_THREADPOOL_LIMITER

    try:
        resolved_device, device_index = requested_cuda_device(requested_device)
    except RuntimeError as exc:
        raise PublicIOError(str(exc)) from exc
    try:
        cuda_runtime = torch_module.version.cuda
        cuda = torch_module.cuda
        backends = torch_module.backends
        _require(cuda_runtime == SCIENCE_GPU_CUDA_RUNTIME, "science-gpu Torch CUDA runtime differs from 13.2")
        _require(cuda.is_available(), "science-gpu requires an available CUDA device; CPU fallback is disabled")
        count = cuda.device_count()
        _require(
            isinstance(count, int) and not isinstance(count, bool) and device_index < count,
            f"science-gpu requested unavailable CUDA device index {device_index}",
        )
        _require(backends.cudnn.is_available(), "science-gpu requires cuDNN")
        if apply_policy:
            torch_module.set_num_threads(1)
            if torch_module.get_num_interop_threads() != 1:
                torch_module.set_num_interop_threads(1)
            torch_module.use_deterministic_algorithms(True)
            torch_module.set_float32_matmul_precision("highest")
            torch_module.set_default_dtype(torch_module.float32)
            cv2_module.setNumThreads(1)
            cv2_module.ocl.setUseOpenCL(False)
            _SCIENCE_THREADPOOL_LIMITER = threadpoolctl_module.threadpool_limits(
                limits=1
            )
            backends.cudnn.benchmark = False
            backends.cudnn.deterministic = True
            backends.cudnn.allow_tf32 = False
            backends.cuda.matmul.allow_tf32 = False
            cuda.set_device(device_index)
        native_pools = threadpoolctl_module.threadpool_info()
        _require(
            torch_module.get_num_threads() == 1,
            "science-gpu Torch intra-op thread limit was not applied",
        )
        _require(
            torch_module.get_num_interop_threads() == 1,
            "science-gpu Torch inter-op thread limit was not applied",
        )
        _require(
            cv2_module.getNumThreads() == 1,
            "science-gpu OpenCV thread limit was not applied",
        )
        _require(
            not cv2_module.ocl.useOpenCL(),
            "science-gpu OpenCV OpenCL disablement was not applied",
        )
        _require(
            isinstance(native_pools, list)
            and bool(native_pools)
            and all(
                isinstance(row, dict)
                and isinstance(row.get("num_threads"), int)
                and not isinstance(row.get("num_threads"), bool)
                and row["num_threads"] == 1
                for row in native_pools
            ),
            "science-gpu native thread-pool limit was not applied",
        )
        _require(
            torch_module.are_deterministic_algorithms_enabled(),
            "science-gpu deterministic Torch algorithms were not enabled",
        )
        _require(
            torch_module.get_float32_matmul_precision() == "highest",
            "science-gpu FP32 matmul precision was not applied",
        )
        _require(
            torch_module.get_default_dtype() == torch_module.float32,
            "science-gpu Torch default dtype is not float32",
        )
        _require(
            os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "0",
            "science-gpu inherited Torch TF32 cuBLAS override is not disabled",
        )
        autocast_enabled = torch_module.is_autocast_enabled("cuda")
        _require(
            autocast_enabled is False,
            "science-gpu CUDA autocast must be disabled; implicit precision changes are forbidden",
        )
        _require(backends.cudnn.benchmark is False, "science-gpu cuDNN benchmark disablement was not applied")
        _require(backends.cudnn.deterministic is True, "science-gpu deterministic cuDNN policy was not applied")
        _require(backends.cudnn.allow_tf32 is False, "science-gpu cuDNN TF32 disablement was not applied")
        _require(backends.cuda.matmul.allow_tf32 is False, "science-gpu CUDA matmul TF32 disablement was not applied")
        _require(cuda.current_device() == device_index, "science-gpu resolved CUDA device is not current")
        properties = cuda.get_device_properties(device_index)
        capability = cuda.get_device_capability(device_index)
        driver_api_version = cupy_module.cuda.runtime.driverGetVersion()
        _require(
            isinstance(driver_api_version, int)
            and not isinstance(driver_api_version, bool)
            and driver_api_version > 0,
            "science-gpu CuPy CUDA driver API version is invalid",
        )
        probe = torch_module.ones((2, 2), dtype=torch_module.float32, device=resolved_device)
        probe_result = float((probe @ probe).sum().item())
        _require(probe_result == 8.0, "science-gpu CUDA execution probe returned an unexpected result")
        _require(bool(probe.is_cuda), "science-gpu CUDA execution probe did not run on CUDA")
        _require(probe.device.type == "cuda" and probe.device.index == device_index, "science-gpu CUDA probe used the wrong device")
        cuda.synchronize(device_index)
        del probe
    except PublicIOError:
        raise
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise PublicIOError("science-gpu numerical runtime policy could not be applied") from exc
    _require(
        isinstance(capability, tuple)
        and len(capability) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in capability),
        "science-gpu CUDA capability is invalid",
    )
    name = str(properties.name).strip()
    total_memory = properties.total_memory
    _require(name != "", "science-gpu CUDA device name is empty")
    _require(isinstance(total_memory, int) and not isinstance(total_memory, bool) and total_memory > 0, "science-gpu CUDA memory identity is invalid")
    xgboost_build = _xgboost_cuda_build_identity(xgboost_module)
    sam2_cuda_extension = _sam2_cuda_extension_attestation(
        torch_module=torch_module,
        extension_module=sam2_extension_module,
        distribution=sam2_distribution,
        resolved_device=resolved_device,
        device_index=device_index,
    )
    return {
        "policy": dict(policy),
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "cuda": {
            "device_index": device_index,
            "device_name": name,
            "compute_capability": list(capability),
            "total_memory_bytes": total_memory,
            "torch_cuda_runtime": cuda_runtime,
            "cudnn_version": backends.cudnn.version(),
            "driver_api_version": driver_api_version,
        },
        "xgboost_build": xgboost_build,
        "sam2_cuda_extension": sam2_cuda_extension,
    }


def _science_gpu_static_identity(
    device: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Recompute the installed GPU closure without touching CUDA policy state."""

    _require_supported_python()
    _require(
        platform.python_version() == SCIENCE_GPU_PYTHON,
        "science-gpu requires the locked CPython 3.12.7 runtime",
    )
    try:
        requested_cuda_device(device)
    except RuntimeError as exc:
        raise PublicIOError(str(exc)) from exc
    try:
        installed_implementation = package_implementation_identity(
            require_installed=True,
            expected_distribution_version=SCIENCE_GPU_VERSIONS["compag-curation"],
        )
    except RuntimeError as exc:
        raise PublicIOError(
            "science-gpu requires a non-editable, bytecode-free install in an "
            "active virtual or attested Conda/Mamba environment; "
            f"installation attestation failed: {exc}"
        ) from exc
    identity = gpu_dependency_identity()
    expected_implementation = {
        key: value
        for key, value in installed_implementation.items()
        if key != "installation"
    }
    _require(
        identity.get("package_implementation") == expected_implementation,
        "installed package implementation changed during science-gpu validation",
    )
    missing = set(EXPECTED_SCIENCE_GPU_VERSIONS) & set(identity["missing"])
    _require(
        not missing,
        f"science-gpu dependencies are missing: {', '.join(sorted(missing))}; install the documented science-gpu lock",
    )
    _require(
        identity["versions"] == EXPECTED_SCIENCE_GPU_VERSIONS,
        "installed science-gpu versions differ from the supported lock",
    )
    direct = identity.get("sam2_direct_url")
    vcs = direct.get("vcs_info") if isinstance(direct, dict) else None
    _require(
        isinstance(direct, dict)
        and direct.get("url") == SCIENCE_GPU_SAM2_URL
        and isinstance(vcs, dict)
        and vcs.get("vcs") == "git"
        and vcs.get("commit_id") == SCIENCE_GPU_SAM2_COMMIT,
        "SAM-2 installation is not bound to the reviewed official Git commit",
    )
    _require(
        identity.get("lock_sha256") == SCIENCE_GPU_LOCK_SHA256,
        "science-gpu lock resource identity mismatch",
    )
    return identity, installed_implementation


def _science_gpu_dependency_record(
    identity: Mapping[str, object],
    installed_implementation: Mapping[str, object],
    imported: list[str],
    enforced: Mapping[str, object],
) -> dict[str, object]:
    record = dict(identity)
    record["schema"] = "compag-curation-science-gpu-dependencies/v2"
    record["imports"] = list(imported)
    record["package_installation"] = installed_implementation["installation"]
    record["gpu_runtime_policy"] = enforced["policy"]
    record["requested_device"] = enforced["requested_device"]
    record["resolved_device"] = enforced["resolved_device"]
    record["cuda_visibility"] = cuda_visibility_identity()
    record["cuda"] = enforced["cuda"]
    record["xgboost_build"] = enforced["xgboost_build"]
    record["sam2_cuda_extension"] = enforced["sam2_cuda_extension"]
    record["sha256"] = compact_json_sha256(
        {key: value for key, value in record.items() if key != "sha256"}
    )
    return record


def require_science_gpu_dependencies(device: str) -> dict[str, object]:
    """Fail closed unless the exact science-gpu closure and CUDA runtime pass."""

    runtime_policy = science_gpu_runtime_policy()
    _set_science_gpu_environment(runtime_policy)
    identity, installed_implementation = _science_gpu_static_identity(device)
    controlled_modules = {
        name: importlib.import_module(name)
        for name in (
            "torch",
            "xgboost",
            "cupy",
            "sam2._C",
            "cv2",
            "threadpoolctl",
        )
    }
    try:
        sam2_distribution = importlib.metadata.distribution("sam-2")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PublicIOError("science-gpu SAM2 distribution metadata is missing") from exc
    enforced = _enforce_science_gpu_runtime(
        runtime_policy,
        requested_device=device,
        torch_module=controlled_modules["torch"],
        xgboost_module=controlled_modules["xgboost"],
        cupy_module=controlled_modules["cupy"],
        sam2_extension_module=controlled_modules["sam2._C"],
        sam2_distribution=sam2_distribution,
        cv2_module=controlled_modules["cv2"],
        threadpoolctl_module=controlled_modules["threadpoolctl"],
    )
    imported = []
    for name in SCIENCE_GPU_IMPORTS:
        importlib.import_module(name)
        imported.append(name)
    enforced = _enforce_science_gpu_runtime(
        runtime_policy,
        requested_device=device,
        torch_module=controlled_modules["torch"],
        xgboost_module=controlled_modules["xgboost"],
        cupy_module=controlled_modules["cupy"],
        sam2_extension_module=controlled_modules["sam2._C"],
        sam2_distribution=sam2_distribution,
        cv2_module=controlled_modules["cv2"],
        threadpoolctl_module=controlled_modules["threadpoolctl"],
    )
    return _science_gpu_dependency_record(
        identity,
        installed_implementation,
        imported,
        enforced,
    )


def revalidate_science_gpu_dependencies(
    device: str,
    expected: Mapping[str, object],
) -> dict[str, object]:
    """Revalidate a captured GPU preflight after its CUDA context exists."""

    _require(isinstance(expected, Mapping), "science-gpu preflight receipt is invalid")
    captured = dict(expected)
    _require(
        captured.get("schema") == "compag-curation-science-gpu-dependencies/v2"
        and captured.get("requested_device") == device
        and isinstance(captured.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(captured["sha256"])) is not None
        and captured["sha256"]
        == compact_json_sha256(
            {key: value for key, value in captured.items() if key != "sha256"}
        ),
        "science-gpu preflight receipt is invalid",
    )
    runtime_policy = science_gpu_runtime_policy()
    environment = runtime_policy.get("environment")
    _require(
        isinstance(environment, dict)
        and all(os.environ.get(str(name)) == value for name, value in environment.items()),
        "science-gpu runtime environment changed after preflight",
    )
    identity, installed_implementation = _science_gpu_static_identity(device)
    controlled_modules = {
        name: importlib.import_module(name)
        for name in (
            "torch",
            "xgboost",
            "cupy",
            "sam2._C",
            "cv2",
            "threadpoolctl",
        )
    }
    try:
        sam2_distribution = importlib.metadata.distribution("sam-2")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PublicIOError("science-gpu SAM2 distribution metadata is missing") from exc
    enforced = _enforce_science_gpu_runtime(
        runtime_policy,
        requested_device=device,
        torch_module=controlled_modules["torch"],
        xgboost_module=controlled_modules["xgboost"],
        cupy_module=controlled_modules["cupy"],
        sam2_extension_module=controlled_modules["sam2._C"],
        sam2_distribution=sam2_distribution,
        cv2_module=controlled_modules["cv2"],
        threadpoolctl_module=controlled_modules["threadpoolctl"],
        apply_policy=False,
    )
    imported = []
    for name in SCIENCE_GPU_IMPORTS:
        importlib.import_module(name)
        imported.append(name)
    current = _science_gpu_dependency_record(
        identity,
        installed_implementation,
        imported,
        enforced,
    )
    _require(
        current == captured,
        "science-gpu dependencies or runtime identity changed after preflight",
    )
    return current


def release_science_gpu_allocator_caches(device: str) -> dict[str, object]:
    """Release observable Torch and CuPy allocator caches before child inference."""

    try:
        resolved_device, device_index = requested_cuda_device(device)
    except RuntimeError as exc:
        raise PublicIOError(str(exc)) from exc
    try:
        torch_module = importlib.import_module("torch")
        cupy_module = importlib.import_module("cupy")
        cuda = torch_module.cuda
        _require(
            cuda.is_available() and cuda.is_initialized(),
            "science-gpu memory release requires an initialized CUDA runtime",
        )
        _require(
            cuda.current_device() == device_index,
            "science-gpu memory release found the wrong current CUDA device",
        )
        cupy_device = cupy_module.cuda.Device(device_index)
        torch_before = {
            "allocated_bytes": int(cuda.memory_allocated(device_index)),
            "reserved_bytes": int(cuda.memory_reserved(device_index)),
        }
        memory_pool = cupy_module.get_default_memory_pool()
        pinned_pool = cupy_module.get_default_pinned_memory_pool()
        cupy_before = {
            "used_bytes": int(memory_pool.used_bytes()),
            "total_bytes": int(memory_pool.total_bytes()),
            "pinned_free_blocks": int(pinned_pool.n_free_blocks()),
        }
        clear_cublas_workspaces = getattr(
            getattr(torch_module, "_C", None),
            "_cuda_clearCublasWorkspaces",
            None,
        )
        _require(
            callable(clear_cublas_workspaces),
            "science-gpu Torch cuBLAS workspace release API is unavailable",
        )
        cuda.synchronize(device_index)
        cupy_device.synchronize()
        clear_cublas_workspaces()
        cuda.synchronize(device_index)
        first_collection = int(gc.collect())
        memory_pool.free_all_blocks()
        pinned_pool.free_all_blocks()
        cuda.empty_cache()
        second_collection = int(gc.collect())
        cuda.synchronize(device_index)
        cupy_device.synchronize()
        torch_after = {
            "allocated_bytes": int(cuda.memory_allocated(device_index)),
            "reserved_bytes": int(cuda.memory_reserved(device_index)),
        }
        cupy_after = {
            "used_bytes": int(memory_pool.used_bytes()),
            "total_bytes": int(memory_pool.total_bytes()),
            "pinned_free_blocks": int(pinned_pool.n_free_blocks()),
        }
    except PublicIOError:
        raise
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise PublicIOError(
            "science-gpu parent allocator caches could not be released before child inference"
        ) from exc
    _require(
        torch_after["allocated_bytes"] == 0
        and torch_after["reserved_bytes"] == 0
        and cupy_after["used_bytes"] == 0
        and cupy_after["total_bytes"] == 0,
        "science-gpu parent Torch/CuPy allocator counters remain nonzero before child inference",
    )
    return {
        "schema": "compag-curation-science-gpu-parent-allocator-cache-release/v1",
        "status": "PASS",
        "requested_device": device,
        "resolved_device": resolved_device,
        "device_index": device_index,
        "validation_scope": "OBSERVED_TORCH_AND_CUPY_ALLOCATOR_COUNTERS_ONLY",
        "native_cuda_context_xgboost_vram_zero_proof": "NOT_CLAIMED",
        "actions": [
            "CUDA_SYNCHRONIZE",
            "CUPY_DEVICE_SYNCHRONIZE",
            "TORCH_CUDA_CLEAR_CUBLAS_WORKSPACES",
            "POST_CUBLAS_WORKSPACE_RELEASE_SYNCHRONIZE",
            "PYTHON_GC_COLLECT",
            "CUPY_DEFAULT_POOL_FREE_ALL_BLOCKS",
            "CUPY_PINNED_POOL_FREE_ALL_BLOCKS",
            "TORCH_CUDA_EMPTY_CACHE",
            "POST_RELEASE_SYNCHRONIZE",
        ],
        "gc_collected": first_collection + second_collection,
        "torch_before": torch_before,
        "torch_after": torch_after,
        "cupy_before": cupy_before,
        "cupy_after": cupy_after,
        "cpu_fallback": False,
    }


def require_science_dependencies(device: str) -> dict[str, object]:
    if device == "cuda" or re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device):
        return require_science_gpu_dependencies(device)
    raise PublicIOError(
        "v1.7 executable science requires device=cuda; science-cpu is "
        "historical verification-only and no CPU fallback is available"
    )


def validate_decodable_images(config: PublicProjectConfig, inspection: Mapping[str, object]) -> dict[str, object]:
    """Decode every configured image from the exact bytes bound by inspection."""

    import cv2
    import numpy as np

    expected = {
        (role, str(row["filename"])): (str(row["sha256"]), int(row["width"]), int(row["height"]))
        for role in ("images", "inference_images")
        for row in inspection[role]
    }
    checked: list[dict[str, object]] = []
    for root, inference in ((config.images, False), (config.inference_images, True)):
        role = "inference_images" if inference else "images"
        for path in image_files(config, inference=inference):
            payload, snapshot = stable_file(path, max_bytes=MAX_IMAGE_FILE_BYTES)
            digest = hashlib.sha256(payload).hexdigest()
            expected_row = expected.get((role, path.name))
            _require(expected_row is not None and expected_row[0] == digest, f"decoded image bytes differ from inspected input: {path.name}")
            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            _require(decoded is not None and decoded.ndim == 3 and decoded.shape[2] == 3, f"image decoder rejected input: {path.name}")
            _require(int(decoded.shape[0]) >= 64 and int(decoded.shape[1]) >= 64, f"decoded image is smaller than 64x64: {path.name}")
            _require(
                (int(decoded.shape[1]), int(decoded.shape[0])) == expected_row[1:],
                f"decoded image dimensions differ from structural inspection: {path.name}",
            )
            checked.append({"role": role, "filename": path.name, "sha256": digest, "size_bytes": snapshot.st_size, "width": int(decoded.shape[1]), "height": int(decoded.shape[0])})
    _require(len(checked) == len(expected), "decoded image closure mismatch")
    return {"schema": "compag-curation-image-decode-validation/v1", "status": "PASS", "images": checked}


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> int:
    path = path.absolute()
    _require(not path.exists() and not path.is_symlink(), f"CSV output already exists: {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o644)
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
                f"CSV output metadata is unsafe: {path.name}",
            )
            os.fsync(fd)
    finally:
        os.close(fd)
    _require(count > 0, f"CSV output contains no rows: {path.name}")
    return count


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), f"CSV input is unsafe: {path}")
    payload, snapshot = stable_file(path)
    _require(
        (snapshot.st_dev, snapshot.st_ino, snapshot.st_size, snapshot.st_mtime_ns, snapshot.st_ctime_ns)
        == (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns),
        f"CSV input changed before reading: {path}",
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise PublicIOError(f"CSV input is not UTF-8: {path}") from exc
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()) == tuple(fields), f"CSV header mismatch: {path.name}")
        rows = list(reader)
    for index, row in enumerate(rows, start=2):
        _require(None not in row and tuple(row) == tuple(fields), f"CSV row {index} has the wrong field count: {path.name}")
        _require(all(isinstance(value, str) for value in row.values()), f"CSV row {index} contains a missing value: {path.name}")
    _require(rows, f"CSV input is empty: {path.name}")
    return rows


def _image_id(filename: str, group_id: str, digest: str) -> str:
    return hashlib.sha256(
        b"compag-image-v1\0" + filename.encode("utf-8") + b"\0" + group_id.encode("ascii") + b"\0" + digest.encode("ascii")
    ).hexdigest()


def _tile_positions(length: int, tile_size: int, step: int) -> list[int]:
    positions = []
    for value in range(0, length, step):
        positions.append(value)
        if value + tile_size >= length:
            break
    return positions


def prepare_images(config: PublicProjectConfig, output: Path) -> dict[str, object]:
    import cv2
    import numpy as np

    tiles = output / "tiles"
    tiles.mkdir(mode=0o755)
    rows: list[dict[str, object]] = []
    step = config.tile_size - config.tile_overlap
    _require(step > 0, "tile overlap leaves a nonpositive step")
    for source in image_files(config):
        source_payload, _source_snapshot = stable_file(source, max_bytes=MAX_IMAGE_FILE_BYTES)
        image_sha = hashlib.sha256(source_payload).hexdigest()
        group = group_id_from_name(source.name)
        identity = _image_id(source.name, group, image_sha)
        bgr = cv2.imdecode(np.frombuffer(source_payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        _require(bgr is not None and len(bgr.shape) == 3 and bgr.shape[2] == 3, f"image decode failed: {source.name}")
        height, width = bgr.shape[:2]
        _require(width >= 64 and height >= 64, f"image is smaller than 64x64: {source.name}")
        for y in _tile_positions(height, config.tile_size, step):
            for x in _tile_positions(width, config.tile_size, step):
                patch = bgr[y : min(y + config.tile_size, height), x : min(x + config.tile_size, width)]
                _require(patch.size > 0, f"empty tile generated for {source.name}")
                tile_name = f"{source.stem}_y{y:05d}x{x:05d}.png"
                target = tiles / tile_name
                _require(not target.exists(), f"tile name collision: {tile_name}")
                ok, encoded = cv2.imencode(".png", patch, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                _require(bool(ok), f"tile encoding failed: {tile_name}")
                write_new_bytes(target, bytes(encoded))
                rows.append(
                    {
                        "tile_name": tile_name,
                        "image_id": identity,
                        "image_sha256": image_sha,
                        "image_name": source.name,
                        "group_id": group,
                        "tile_sha256": sha256_file(target),
                        "x": x,
                        "y": y,
                        "crop_w": int(patch.shape[1]),
                        "crop_h": int(patch.shape[0]),
                        "orig_w": width,
                        "orig_h": height,
                    }
                )
    rows.sort(key=lambda row: (str(row["image_id"]), int(row["y"]), int(row["x"]), str(row["tile_name"])))
    _write_csv(output / "tiles_index.csv", TILE_COLUMNS, rows)
    summary = {
        "schema": "compag-curation-prepare/v1",
        "status": "PASS",
        "profile": config.profile,
        "image_count": len({str(row["image_id"]) for row in rows}),
        "group_count": len({str(row["group_id"]) for row in rows}),
        "tile_count": len(rows),
        "tile_size": config.tile_size,
        "tile_overlap": config.tile_overlap,
        "tiles_index_sha256": sha256_file(output / "tiles_index.csv"),
    }
    write_new_json(output / "prepare_summary.json", summary)
    return summary


def _mask_sha256(mask: object) -> str:
    import numpy as np

    binary = np.asarray(mask, dtype=np.uint8) > 0
    _require(binary.ndim == 2 and binary.size > 0, "SAM2 returned an invalid mask")
    height, width = binary.shape
    packed = np.packbits(binary.reshape(-1), bitorder="little").tobytes()
    return hashlib.sha256(b"compag-mask-v1\0" + height.to_bytes(4, "big") + width.to_bytes(4, "big") + packed).hexdigest()


def _largest_component(mask: object):
    import cv2
    import numpy as np

    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    _require(binary.ndim == 2 and binary.size > 0, "SAM2 returned an invalid mask")
    count, labels, statistics, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count < 2:
        return None
    component = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    selected = (labels == component).astype(np.uint8)
    _require(int(selected.sum()) > 0, "SAM2 proposal component is empty")
    return selected


def _polygon(mask: object) -> tuple[list[int], tuple[int, int, int, int]]:
    import cv2
    import numpy as np

    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _require(bool(contours), "proposal mask has no external contour")
    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    epsilon = max(0.5, 0.002 * cv2.arcLength(contour, True))
    simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    _require(len(simplified) >= 3, "proposal polygon has fewer than three vertices")
    return [int(value) for value in simplified.reshape(-1)], (int(x), int(y), int(width), int(height))


def _proposal_id(
    *,
    image_sha256: str,
    tile_sha256: str,
    tile_x: int,
    tile_y: int,
    mask_sha256: str,
    bbox: tuple[int, int, int, int],
    polygon: Sequence[int],
    proposal_settings_sha256: str,
) -> str:
    identity = {
        "schema": "compag-proposal-geometry/v2",
        "image_sha256": image_sha256,
        "tile_sha256": tile_sha256,
        "tile_x": tile_x,
        "tile_y": tile_y,
        "mask_sha256": mask_sha256,
        "bbox": list(bbox),
        "poly": list(polygon),
        "proposal_settings_sha256": proposal_settings_sha256,
    }
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def _proposal_settings(config: PublicProjectConfig) -> dict[str, object]:
    return {
        "schema": "compag-curation-proposal-config/v2",
        "profile": config.profile,
        "sam2_config_locator": config.sam2_config_locator,
        "sam2_checkpoint_sha256": config.checkpoint_sha256,
        "sam2_config_sha256": config.sam2_config_sha256,
        "source_commit": SAM2_SOURCE_COMMIT,
        "device": config.device,
        "points_per_side": config.points_per_side,
        "points_per_batch": config.points_per_batch,
        "pred_iou_threshold": config.pred_iou_threshold,
        "stability_threshold": config.stability_threshold,
        "stability_score_offset": 1.0,
        "box_nms_threshold": 0.7,
        "crop_layers": 0,
        "crop_nms_threshold": 0.7,
        "crop_overlap_ratio": 512 / 1500,
        "crop_points_downscale": 1,
        "min_mask_region_area": 0,
        "multiscale": [1.0],
        "multiscale_iou": 0.75,
        "sam_score_decimal_places": SAM_SCORE_DECIMAL_PLACES,
        "proposal_identity": "geometry-mask-v2",
        "cpu_runtime_policy": science_cpu_runtime_policy(),
        "max_proposals_per_tile": config.max_proposals_per_tile,
        "min_mask_area": config.min_mask_area,
        "feature_order_sha256": FEATURE_ORDER_SHA256,
    }


def _smote_plan(negative_count: int, positive_count: int) -> tuple[bool, int, int]:
    _require(negative_count > 0 and positive_count > 0, "training partition lacks one class")
    minority_class = 0 if negative_count <= positive_count else 1
    minority = min(negative_count, positive_count)
    majority = max(negative_count, positive_count)
    target = math.ceil(majority * 0.5)
    if target <= minority:
        return False, minority_class, minority
    _require(minority >= 2, "training partition has too few minority rows for deterministic SMOTE")
    return True, minority_class, target


def _load_sam2(config: PublicProjectConfig):
    require_science_dependencies(config.device)
    import sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    package_root = Path(sam2.__file__).resolve(strict=True).parent
    installed_config = package_root / f"{config.sam2_config_locator}.yaml"
    _require(installed_config.exists() and not installed_config.is_symlink(), "installed SAM2 configuration is unavailable or unsafe")
    _require(sha256_file(config.sam2_config) == config.sam2_config_sha256, "configured SAM2 configuration hash mismatch")
    installed_digest, installed_before = hash_file_snapshot(installed_config, require_single_link=True)
    _require(installed_digest == config.sam2_config_sha256, "installed SAM2 configuration differs from the verified asset")
    installed_parent_before = installed_config.parent.lstat()
    with verified_file_path(
        config.checkpoint,
        config.checkpoint_sha256,
        max_bytes=1024 * 1024 * 1024,
        require_single_link=True,
    ) as (checkpoint_path, _checkpoint_snapshot):
        model = build_sam2(
            config.sam2_config_locator,
            str(checkpoint_path),
            device=config.device,
            mode="eval",
            apply_postprocessing=False,
        )
    installed_digest_after, installed_after = hash_file_snapshot(installed_config, require_single_link=True)
    installed_parent_after = installed_config.parent.lstat()
    _require(
        installed_digest_after == installed_digest
        and (
            installed_after.st_dev, installed_after.st_ino, installed_after.st_mode, installed_after.st_uid,
            installed_after.st_gid, installed_after.st_nlink, installed_after.st_size,
            installed_after.st_mtime_ns, installed_after.st_ctime_ns,
        ) == (
            installed_before.st_dev, installed_before.st_ino, installed_before.st_mode, installed_before.st_uid,
            installed_before.st_gid, installed_before.st_nlink, installed_before.st_size,
            installed_before.st_mtime_ns, installed_before.st_ctime_ns,
        )
        and (
            installed_parent_after.st_dev, installed_parent_after.st_ino, installed_parent_after.st_mode,
            installed_parent_after.st_uid, installed_parent_after.st_gid, installed_parent_after.st_nlink,
            installed_parent_after.st_mtime_ns, installed_parent_after.st_ctime_ns,
        ) == (
            installed_parent_before.st_dev, installed_parent_before.st_ino, installed_parent_before.st_mode,
            installed_parent_before.st_uid, installed_parent_before.st_gid, installed_parent_before.st_nlink,
            installed_parent_before.st_mtime_ns, installed_parent_before.st_ctime_ns,
        ),
        "installed SAM2 configuration changed while constructing the model",
    )
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=config.points_per_side,
        points_per_batch=config.points_per_batch,
        pred_iou_thresh=config.pred_iou_threshold,
        stability_score_thresh=config.stability_threshold,
        stability_score_offset=1.0,
        mask_threshold=0.0,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        crop_nms_thresh=0.7,
        crop_overlap_ratio=512 / 1500,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=0,
        output_mode="binary_mask",
        use_m2m=False,
        multimask_output=True,
    )
    return model, generator


def generate_proposals(config: PublicProjectConfig, prepared: Path, output: Path) -> dict[str, object]:
    import cv2
    import numpy as np
    from compag_curation.proposals.sam2_pipeline.gate_core import cell_of_point, compute_features, detect_grid_lines, yellow_bg_stats
    tile_rows = _read_csv(prepared / "tiles_index.csv", TILE_COLUMNS)
    _model, generator = _load_sam2(config)
    proposal_settings = _proposal_settings(config)
    proposal_settings_sha = compact_json_sha256(proposal_settings)
    proposals: list[dict[str, object]] = []
    overlays = output / "review_overlays"
    overlays.mkdir(mode=0o755)
    counts: dict[str, int] = {}
    rejected_annotations = {"empty_mask": 0, "below_minimum_area": 0, "duplicate_mask": 0}
    for tile in tile_rows:
        tile_path = prepared / "tiles" / tile["tile_name"]
        tile_payload, _tile_snapshot = stable_file(tile_path, max_bytes=MAX_IMAGE_FILE_BYTES)
        _require(hashlib.sha256(tile_payload).hexdigest() == tile["tile_sha256"], f"prepared tile drift: {tile['tile_name']}")
        bgr = cv2.imdecode(np.frombuffer(tile_payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        _require(bgr is not None, f"prepared tile decode failed: {tile['tile_name']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        anns = generator.generate(rgb) or []
        _require(anns, f"SAM2 generated zero proposals: {tile['tile_name']}")
        candidates: list[tuple[float, float, int, str, object]] = []
        seen_masks: set[str] = set()
        for ann in anns:
            _require(isinstance(ann, dict) and "segmentation" in ann, "SAM2 returned a malformed proposal")
            mask = _largest_component(ann["segmentation"])
            if mask is None:
                rejected_annotations["empty_mask"] += 1
                continue
            area = int(mask.sum())
            if area < config.min_mask_area:
                rejected_annotations["below_minimum_area"] += 1
                continue
            mask_sha = _mask_sha256(mask)
            if mask_sha in seen_masks:
                rejected_annotations["duplicate_mask"] += 1
                continue
            seen_masks.add(mask_sha)
            pred_iou = _canonical_sam_score(ann.get("predicted_iou", 0.0), "predicted IoU")
            stability = _canonical_sam_score(ann.get("stability_score", 0.0), "stability score")
            candidates.append((pred_iou, stability, area, mask_sha, mask))
        candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        candidates = candidates[: config.max_proposals_per_tile]
        _require(len(candidates) >= 2, f"fewer than two usable SAM2 proposals: {tile['tile_name']}")
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        bg_a, bg_b, _bg = yellow_bg_stats(bgr)
        row_lines, col_lines = detect_grid_lines(bgr)
        overlay = bgr.copy()
        for pred_iou, stability, _area, mask_sha, mask in candidates:
            features = compute_features(mask * 255, lab, mode="balanced")
            _require(bool(features), f"feature extraction returned no values: {tile['tile_name']}")
            features["delta_a"] = float(bg_a - float(features["mean_a"]))
            features["delta_b"] = float(float(features["mean_b"]) - bg_b)
            grid_r, grid_c = cell_of_point(float(features["cx"]), float(features["cy"]), row_lines, col_lines)
            features["grid_r_norm"] = float(grid_r / max(1, len(row_lines) - 1))
            features["grid_c_norm"] = float(grid_c / max(1, len(col_lines) - 1))
            features["pred_iou"] = pred_iou
            features["stability"] = stability
            selected = {name: float(features[name]) for name in PUBLIC_FEATURE_ORDER}
            _require(all(math.isfinite(value) for value in selected.values()), "nonfinite feature value rejected")
            poly, bbox = _polygon(mask)
            proposal_id = _proposal_id(
                image_sha256=str(tile["image_sha256"]),
                tile_sha256=str(tile["tile_sha256"]),
                tile_x=int(tile["x"]),
                tile_y=int(tile["y"]),
                mask_sha256=mask_sha,
                bbox=bbox,
                polygon=poly,
                proposal_settings_sha256=proposal_settings_sha,
            )
            row: dict[str, object] = {
                "proposal_id": proposal_id,
                "proposal_sha256": proposal_id,
                "image_id": tile["image_id"],
                "image_sha256": tile["image_sha256"],
                "group_id": tile["group_id"],
                "tile_name": tile["tile_name"],
                "tile_sha256": tile["tile_sha256"],
                "tile_x": int(tile["x"]),
                "tile_y": int(tile["y"]),
                "mask_sha256": mask_sha,
                "bbox_x": bbox[0],
                "bbox_y": bbox[1],
                "bbox_w": bbox[2],
                "bbox_h": bbox[3],
                "poly": json.dumps(poly, separators=(",", ":")),
                **{name: _format_feature_value(name, selected[name]) for name in PUBLIC_FEATURE_ORDER},
            }
            proposals.append(row)
            contour = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 1)
            cv2.putText(overlay, proposal_id[:6], (bbox[0], max(10, bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        counts[tile["group_id"]] = counts.get(tile["group_id"], 0) + len(candidates)
        ok, encoded = cv2.imencode(".png", overlay, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        _require(bool(ok), f"review overlay encoding failed: {tile['tile_name']}")
        write_new_bytes(overlays / tile["tile_name"], bytes(encoded))
    proposals.sort(key=lambda row: (str(row["image_id"]), int(row["tile_y"]), int(row["tile_x"]), str(row["proposal_id"])))
    ids = [str(row["proposal_id"]) for row in proposals]
    _require(len(ids) == len(set(ids)), "duplicate stable proposal IDs")
    _write_csv(output / "proposals.csv", PROPOSAL_COLUMNS, proposals)
    feature_fields = ("proposal_id", "image_id", "group_id", *PUBLIC_FEATURE_ORDER)
    feature_rows = [{field: row[field] for field in feature_fields} for row in proposals]
    _write_csv(output / "features.csv", feature_fields, feature_rows)
    write_new_json(output / "proposal_config.json", proposal_settings)
    summary = {
        "schema": "compag-curation-proposals-features/v1",
        "status": "PASS",
        "profile": config.profile,
        "proposal_count": len(proposals),
        "image_count": len({str(row["image_id"]) for row in proposals}),
        "group_counts": dict(sorted(counts.items())),
        "rejected_annotations": rejected_annotations,
        "feature_order": list(PUBLIC_FEATURE_ORDER),
        "feature_order_sha256": FEATURE_ORDER_SHA256,
        "proposal_settings_sha256": proposal_settings_sha,
        "proposals_sha256": sha256_file(output / "proposals.csv"),
        "features_sha256": sha256_file(output / "features.csv"),
    }
    write_new_json(output / "proposal_summary.json", summary)
    return summary


def _review_rows(path: Path) -> list[dict[str, str]]:
    from compag_curation.review.exchange import REVIEW_COLUMNS

    rows = _read_csv(path, REVIEW_COLUMNS)
    for row in rows:
        _require(row["proposal_id"] == row["proposal_sha256"], "review proposal identity mismatch")
        _require(row["label"] in {"0", "1"} and row["review_status"] == "reviewed", "review row is not complete")
    _require(len({row["proposal_id"] for row in rows}) == len(rows), "review rows contain duplicate IDs")
    return rows


def review_split_feasibility(group_labels: Sequence[tuple[str, int]]) -> dict[str, object]:
    _require(bool(group_labels), "reviewed label set is empty")
    _require(
        all(
            isinstance(group, str)
            and re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", group) is not None
            and type(label) is int
            and label in {0, 1}
            for group, label in group_labels
        ),
        "reviewed group/label input is invalid",
    )
    groups = sorted({group for group, _label in group_labels})
    _require(len(groups) >= 4, "at least four independent group IDs are required")
    ordered = sorted(groups, key=lambda group: (hashlib.sha256(f"42:{group}".encode("utf-8")).hexdigest(), group))
    validation_count = max(1, round(len(ordered) * 0.2))
    test_count = max(1, round(len(ordered) * 0.2))
    _require(len(ordered) - validation_count - test_count >= 2, "group split leaves fewer than two training groups")
    partitions = {
        "train": sorted(ordered[: -(test_count + validation_count)]),
        "validation": sorted(ordered[-(test_count + validation_count) : -test_count]),
        "test": sorted(ordered[-test_count:]),
    }
    union = set().union(*(set(value) for value in partitions.values()))
    _require(union == set(groups), "group split does not cover every observed group")
    _require(sum(len(value) for value in partitions.values()) == len(union), "group split partitions overlap")
    coverage: dict[str, dict[str, object]] = {}
    class_counts: dict[str, dict[str, int]] = {}
    for name, selected in partitions.items():
        labels = [label for group, label in group_labels if group in selected]
        _require(set(labels) == {0, 1}, f"{name} partition does not contain both labels")
        negative_count = sum(label == 0 for label in labels)
        positive_count = sum(label == 1 for label in labels)
        coverage[name] = {"groups": len(selected), "rows": len(labels), "labels": [0, 1]}
        class_counts[name] = {"negative": negative_count, "positive": positive_count}
    smote_applied, minority_class, target_minority = _smote_plan(
        class_counts["train"]["negative"],
        class_counts["train"]["positive"],
    )
    return {
        "schema": "compag-curation-review-split-feasibility/v1",
        "status": "PASS",
        "seed": 42,
        "method": "SHA256_SEEDED_GROUP_ORDER_NEAREST_60_20_20",
        "train_groups": partitions["train"],
        "validation_groups": partitions["validation"],
        "test_groups": partitions["test"],
        "coverage": coverage,
        "class_counts": class_counts,
        "smote": {
            "applied": smote_applied,
            "minority_class": minority_class,
            "target_minority_rows": target_minority,
        },
    }


def build_group_split(reviewed: Path, output: Path) -> dict[str, object]:
    rows = _review_rows(reviewed)
    feasibility = review_split_feasibility([(row["group_id"], int(row["label"])) for row in rows])
    manifest = {
        "schema": "compag-curation-group-split/v1",
        "status": "PASS",
        "seed": 42,
        "method": "SHA256_SEEDED_GROUP_ORDER_NEAREST_60_20_20",
        "train_groups": feasibility["train_groups"],
        "validation_groups": feasibility["validation_groups"],
        "test_groups": feasibility["test_groups"],
        "coverage": feasibility["coverage"],
        "reviewed_sha256": sha256_file(reviewed),
    }
    write_new_json(output / "split_manifest.json", manifest)
    return manifest


def _select_threshold(labels: object, probabilities: object) -> float:
    import numpy as np

    y = np.asarray(labels, dtype=np.int32)
    p = np.asarray(probabilities, dtype=np.float64)
    candidates = sorted({0.0, 0.5, 1.0, *(float(value) for value in p.tolist())})
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        predicted = (p >= threshold).astype(np.int32)
        tp = int(((predicted == 1) & (y == 1)).sum())
        fp = int(((predicted == 1) & (y == 0)).sum())
        fn = int(((predicted == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, recall, -threshold)
        if best is None or candidate > best:
            best = candidate
    _require(best is not None, "validation threshold selection failed")
    return float(-best[2])


def _metrics(labels: object, probabilities: object, threshold: float) -> dict[str, object]:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(labels, dtype=np.int32)
    p = np.asarray(probabilities, dtype=np.float64)
    predicted = (p >= threshold).astype(np.int32)
    tn = int(((predicted == 0) & (y == 0)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    tp = int(((predicted == 1) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": int(len(y)),
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }


def train_classifier(
    config: PublicProjectConfig,
    proposals_path: Path,
    reviewed_path: Path,
    split_path: Path,
    output: Path,
) -> dict[str, object]:
    import numpy as np
    import xgboost as xgb
    from imblearn.over_sampling import SMOTE

    proposals = _read_csv(proposals_path, PROPOSAL_COLUMNS)
    reviewed = _review_rows(reviewed_path)
    proposal_by_id = {row["proposal_id"]: row for row in proposals}
    _require(len(proposal_by_id) == len(proposals), "proposal table contains duplicate IDs")
    _require(set(proposal_by_id) == {row["proposal_id"] for row in reviewed}, "review/proposal ID closure mismatch")
    split_payload, _split_snapshot = stable_file(split_path)
    try:
        split_raw = json.loads(split_payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PublicIOError("split manifest is not valid UTF-8 JSON") from exc
    _require(isinstance(split_raw, dict), "split manifest must be an object")
    _require(
        set(split_raw) == {
            "schema", "status", "seed", "method", "train_groups", "validation_groups",
            "test_groups", "coverage", "reviewed_sha256",
        }
        and split_raw.get("schema") == "compag-curation-group-split/v1"
        and split_raw.get("status") == "PASS"
        and split_raw.get("seed") == 42
        and split_raw.get("method") == "SHA256_SEEDED_GROUP_ORDER_NEAREST_60_20_20",
        "split manifest is invalid",
    )
    _require(split_raw.get("reviewed_sha256") == sha256_file(reviewed_path), "split manifest belongs to another reviewed table")
    for key in ("train_groups", "validation_groups", "test_groups"):
        values = split_raw.get(key)
        _require(
            isinstance(values, list)
            and values
            and all(isinstance(value, str) and value for value in values)
            and len(values) == len(set(values)),
            f"split manifest {key} is invalid",
        )
    partitions = {
        "train": set(split_raw["train_groups"]),
        "validation": set(split_raw["validation_groups"]),
        "test": set(split_raw["test_groups"]),
    }
    _require(
        not (partitions["train"] & partitions["validation"])
        and not (partitions["train"] & partitions["test"])
        and not (partitions["validation"] & partitions["test"]),
        "split partitions overlap",
    )
    observed_groups = {row["group_id"] for row in reviewed}
    _require(set().union(*partitions.values()) == observed_groups, "split partition union differs from reviewed groups")
    expected_coverage: dict[str, dict[str, object]] = {}
    for name, groups in partitions.items():
        selected_review = [row for row in reviewed if row["group_id"] in groups]
        labels = sorted({int(row["label"]) for row in selected_review})
        _require(labels == [0, 1], f"{name} partition does not contain both labels")
        expected_coverage[name] = {"groups": len(groups), "rows": len(selected_review), "labels": labels}
    _require(split_raw.get("coverage") == expected_coverage, "split coverage evidence differs from the reviewed table")
    joined = []
    for review in reviewed:
        proposal = proposal_by_id[review["proposal_id"]]
        _require(
            (review["image_id"], review["image_sha256"], review["group_id"], review["proposal_sha256"])
            == (proposal["image_id"], proposal["image_sha256"], proposal["group_id"], proposal["proposal_sha256"]),
            f"review identity differs from proposal: {review['proposal_id']}",
        )
        joined.append((proposal, int(review["label"])))

    def selected(groups: set[str]):
        chosen = [(row, label) for row, label in joined if row["group_id"] in groups]
        _require(chosen, "empty training partition")
        matrix = np.asarray([[float(row[name]) for name in PUBLIC_FEATURE_ORDER] for row, _label in chosen], dtype=np.float64)
        labels = np.asarray([label for _row, label in chosen], dtype=np.int32)
        _require(np.isfinite(matrix).all(), "nonfinite training feature rejected")
        _require(set(labels.tolist()) == {0, 1}, "training partition lacks one class")
        return chosen, matrix, labels

    train_rows, train_x, train_y = selected(partitions["train"])
    validation_rows, validation_x, validation_y = selected(partitions["validation"])
    test_rows, test_x, test_y = selected(partitions["test"])
    medians = np.median(train_x, axis=0)
    _require(np.isfinite(medians).all(), "median imputer produced nonfinite statistics")
    train_x = np.where(np.isfinite(train_x), train_x, medians)
    validation_x = np.where(np.isfinite(validation_x), validation_x, medians)
    test_x = np.where(np.isfinite(test_x), test_x, medians)
    negative_count = int((train_y == 0).sum())
    positive_count = int((train_y == 1).sum())
    smote_applied, minority_class, target_minority = _smote_plan(negative_count, positive_count)
    if smote_applied:
        minority = min(negative_count, positive_count)
        neighbors = min(5, minority - 1)
        train_x, train_y = SMOTE(
            sampling_strategy={minority_class: target_minority},
            k_neighbors=neighbors,
            random_state=42,
        ).fit_resample(train_x, train_y)
    train_matrix = xgb.DMatrix(train_x, label=train_y, feature_names=list(PUBLIC_FEATURE_ORDER))
    validation_matrix = xgb.DMatrix(validation_x, label=validation_y, feature_names=list(PUBLIC_FEATURE_ORDER))
    test_matrix = xgb.DMatrix(test_x, label=test_y, feature_names=list(PUBLIC_FEATURE_ORDER))
    parameters = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1.0,
        "lambda": 1.0,
        "seed": 42,
        "tree_method": "hist",
        "nthread": 1,
    }
    booster = xgb.train(parameters, train_matrix, num_boost_round=300, evals=[(validation_matrix, "validation")], verbose_eval=False)
    validation_probability = booster.predict(validation_matrix, validate_features=True, strict_shape=True).reshape(-1)
    threshold = _select_threshold(validation_y, validation_probability)
    test_probability = booster.predict(test_matrix, validate_features=True, strict_shape=True).reshape(-1)
    metrics = _metrics(test_y, test_probability, threshold)
    classifier = bytes(booster.save_raw(raw_format="ubj"))
    _require(classifier and not classifier.startswith(b"\x80"), "XGBoost portable serialization failed")
    prediction_rows = []
    for (row, label), probability in zip(test_rows, test_probability.tolist(), strict=True):
        prediction_rows.append(
            {
                "proposal_id": row["proposal_id"],
                "group_id": row["group_id"],
                "label": label,
                "probability": format(float(probability), ".17g"),
                "prediction": int(float(probability) >= threshold),
            }
        )
    _write_csv(output / "test_predictions.csv", ("proposal_id", "group_id", "label", "probability", "prediction"), prediction_rows)
    dependencies = require_science_dependencies(config.device)
    compatibility = {
        "schema": "compag-curation-model-compatibility/v2",
        "python": dependencies["python"],
        "versions": dependencies["versions"],
        "lock_sha256": dependencies["lock_sha256"],
        "cpu_runtime_policy_sha256": dependencies["cpu_runtime_policy"]["sha256"],
        "feature_math": "gate-core-balanced-v1",
        "classifier": "xgboost-ubj",
    }
    provenance = {
        "schema": "compag-curation-model-provenance/v1",
        "profile": config.profile,
        "seed": 42,
        "reviewed_table_sha256": sha256_file(reviewed_path),
        "proposal_table_sha256": sha256_file(proposals_path),
        "split_manifest_sha256": sha256_file(split_path),
        "feature_order_sha256": FEATURE_ORDER_SHA256,
        "training_rows_before_smote": len(train_rows),
        "training_rows_after_smote": int(len(train_y)),
        "smote_applied": smote_applied,
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "paper_result_reproduction": "NOT_CLAIMED",
    }
    normalized = {
        "schema": "compag-curation-normalized-public-config/v1",
        "profile": config.profile,
        "device": config.device,
        "seed": config.seed,
        "checkpoint_sha256": config.checkpoint_sha256,
        "sam2_config_sha256": config.sam2_config_sha256,
        "feature_order_sha256": FEATURE_ORDER_SHA256,
    }
    preprocessing = {
        "schema": "compag-curation-preprocessing/v1",
        "tile_size": config.tile_size,
        "tile_overlap": config.tile_overlap,
        "image_color": "OpenCV BGR decoded then SAM2 RGB and OpenCV uint8 LAB features",
        "exif_orientation": "absent_or_1",
    }
    proposal_config = _proposal_settings(config)
    proposal_config["device"] = "cpu"
    license_path = config.asset_root / "SAM2-APACHE-2.0.txt"
    _require(license_path.exists() and sha256_file(license_path) == "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4", "verified SAM2 license asset is missing")
    bundle_result = write_model_bundle(
        BundleWriteRequest(
            output=output / "model_bundle",
            classifier_ubj=classifier,
            imputer_statistics={name: float(medians[index]) for index, name in enumerate(PUBLIC_FEATURE_ORDER)},
            threshold=threshold,
            normalized_config=normalized,
            preprocessing=preprocessing,
            proposal_config=proposal_config,
            compatibility=compatibility,
            provenance=provenance,
            sam2_config=config.sam2_config,
            sam2_checkpoint=config.checkpoint,
            sam2_license=license_path,
        )
    )
    persisted_bundle = {key: value for key, value in bundle_result.items() if key != "root"}
    persisted_bundle["path"] = "model_bundle"
    result = {
        "schema": "compag-curation-training-result/v1",
        "status": "PASS",
        "profile": config.profile,
        "feature_order_sha256": FEATURE_ORDER_SHA256,
        "threshold_selected_on": "validation",
        "threshold": threshold,
        "test_metrics": metrics,
        "split_manifest_sha256": sha256_file(split_path),
        "bundle": persisted_bundle,
        "test_predictions_sha256": sha256_file(output / "test_predictions.csv"),
    }
    write_new_json(output / "training_result.json", result)
    return result


def _bundle_config(images: Path, bundle: VerifiedBundle, device: str) -> PublicProjectConfig:
    proposal = bundle.proposal_config
    required = {
        "sam2_config_locator", "sam2_checkpoint_sha256", "sam2_config_sha256",
        "points_per_side", "points_per_batch", "pred_iou_threshold", "stability_threshold",
        "max_proposals_per_tile", "min_mask_area",
    }
    _require(required <= set(proposal), "bundle proposal configuration is incomplete")
    return PublicProjectConfig(
        config_path=bundle.root / "bundle.json",
        project_root=images.parent,
        project_name="bundle-inference",
        images=images.resolve(strict=True),
        inference_images=images.resolve(strict=True),
        annotations=None,
        asset_root=bundle.root / "sam2",
        device=device,
        seed=42,
        tile_size=int(bundle.preprocessing["tile_size"]),
        tile_overlap=int(bundle.preprocessing["tile_overlap"]),
        points_per_side=int(proposal["points_per_side"]),
        points_per_batch=int(proposal["points_per_batch"]),
        pred_iou_threshold=float(proposal["pred_iou_threshold"]),
        stability_threshold=float(proposal["stability_threshold"]),
        max_proposals_per_tile=int(proposal["max_proposals_per_tile"]),
        min_mask_area=int(proposal["min_mask_area"]),
        checkpoint=bundle.checkpoint,
        checkpoint_sha256=str(proposal["sam2_checkpoint_sha256"]),
        sam2_config=bundle.sam2_config,
        sam2_config_sha256=str(proposal["sam2_config_sha256"]),
        sam2_config_locator=str(proposal["sam2_config_locator"]),
        config_sha256=bundle.bundle_sha256,
        profile=str(bundle.normalized_config["profile"]),
    )


def _box_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    inter_w = max(0, min(lx2, rx2) - max(lx1, rx1))
    inter_h = max(0, min(ly2, ry2) - max(ly1, ry1))
    intersection = inter_w * inter_h
    union = lw * lh + rw * rh - intersection
    return intersection / union if union else 0.0


def infer_from_bundle(images: Path, bundle_root: Path, output: Path, *, device: str = "cuda") -> dict[str, object]:
    """Reject execution of the historical balanced-v1 bundle format.

    The arguments remain accepted so callers can receive a stable fail-closed
    error after separately verifying historical bundle structure.
    """

    del images, bundle_root, output, device
    raise PublicIOError(
        "balanced-v1 bundle inference is historical verification-only in v1.7; "
        "use a Full or Lite GPU v2 bundle with device=cuda"
    )


def evaluate_cvat_points(
    annotations: Path,
    proposals_path: Path,
    predictions_path: Path,
    output: Path,
    *,
    expected_images: Mapping[str, tuple[int, int]],
) -> dict[str, object]:
    from .point_annotations import inspect_point_annotations

    proposals = {row["proposal_id"]: row for row in _read_csv(proposals_path, PROPOSAL_COLUMNS)}
    predictions = _read_csv(predictions_path, PREDICTION_COLUMNS)
    positive = [row for row in predictions if row["kept"] == "1"]
    expected_by_stem = {Path(name).stem: name for name in expected_images}
    _require(len(expected_by_stem) == len(expected_images), "configured inference image stems collide")
    available_stems = {re.sub(r"_y\d{5}x\d{5}$", "", Path(row["tile_name"]).stem) for row in proposals.values()}
    _require(available_stems <= set(expected_by_stem), "proposal table contains an unknown inference image")
    available_names = {expected_by_stem[stem] for stem in available_stems}
    annotation_rows, annotation_summary = inspect_point_annotations(annotations, expected_images=expected_images)
    points: list[tuple[str, float, float]] = []
    ignored_images: set[str] = set()
    for row in annotation_rows:
        name = str(row["image"])
        if name not in available_names:
            ignored_images.add(name)
            continue
        points.append((name, float(row["x"]), float(row["y"])))
    _require(points, "annotation file contains no cj points")

    def inside(poly: list[int], x: float, y: float) -> bool:
        vertices = list(zip(poly[0::2], poly[1::2], strict=True))
        result = False
        j = len(vertices) - 1
        for i, (xi, yi) in enumerate(vertices):
            xj, yj = vertices[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
                result = not result
            j = i
        return result

    details = []
    for image_name, point_x, point_y in points:
        hits = []
        for prediction in positive:
            proposal = proposals[prediction["proposal_id"]]
            proposal_name = expected_by_stem[re.sub(r"_y\d{5}x\d{5}$", "", Path(proposal["tile_name"]).stem)]
            if proposal_name != image_name:
                continue
            poly_local = [int(value) for value in json.loads(proposal["poly"])]
            poly = []
            for index, value in enumerate(poly_local):
                poly.append(value + (int(proposal["tile_x"]) if index % 2 == 0 else int(proposal["tile_y"])))
            if inside(poly, point_x, point_y):
                hits.append(prediction["proposal_id"])
        details.append({"image": image_name, "x": point_x, "y": point_y, "covered": int(bool(hits)), "matching_proposal_ids": "|".join(sorted(hits))})
    _write_csv(output / "point_detail.csv", ("image", "x", "y", "covered", "matching_proposal_ids"), details)
    summary = {
        "schema": "compag-curation-point-coverage/v1",
        "status": "PASS",
        "definition": "a point is covered when it lies inside at least one kept predicted polygon; this is not instance precision",
        "points": len(details),
        "covered": sum(int(row["covered"]) for row in details),
        "coverage": sum(int(row["covered"]) for row in details) / len(details),
        "evaluated_images": sorted(available_names),
        "ignored_annotation_images": sorted(ignored_images),
        "annotations_sha256": annotation_summary["source_sha256"],
        "point_detail_sha256": sha256_file(output / "point_detail.csv"),
    }
    write_new_json(output / "point_summary.json", summary)
    return summary
