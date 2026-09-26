"""Hash-bound runtime closures for supported Linux science profiles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from .version import __version__


SCIENCE_CPU_LOCK_SHA256 = "5ca95d4e64c7865307b3e35dad07b1678aba87d0ca18291360609d07c449b7a3"
SCIENCE_CPU_PYTHON = "3.12.7"
SCIENCE_CPU_SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SCIENCE_CPU_SAM2_URL = "https://github.com/facebookresearch/sam2.git"
SCIENCE_CPU_VERSIONS = {
    "antlr4-python3-runtime": "4.9.3",
    "compag-curation": "1.2.0",
    "contourpy": "1.3.3",
    "cycler": "0.12.1",
    "filelock": "3.32.4",
    "fonttools": "4.63.0",
    "fsspec": "2026.7.0",
    "hydra-core": "1.3.2",
    "imbalanced-learn": "0.14.0",
    "iopath": "0.1.10",
    "jinja2": "3.1.6",
    "joblib": "1.5.3",
    "kiwisolver": "1.5.0",
    "markupsafe": "3.0.3",
    "matplotlib": "3.10.7",
    "mpmath": "1.3.0",
    "networkx": "3.6.1",
    "numpy": "2.0.2",
    "omegaconf": "2.3.0",
    "opencv-python": "4.12.0.88",
    "packaging": "26.3",
    "pandas": "2.3.3",
    "pillow": "11.3.0",
    "portalocker": "3.2.0",
    "pyyaml": "6.0.3",
    "pyparsing": "3.3.2",
    "python-dateutil": "2.9.0.post0",
    "pytz": "2026.3.post1",
    "sam-2": "1.0",
    "scikit-learn": "1.7.2",
    "scipy": "1.16.2",
    "six": "1.17.0",
    "setuptools": "80.9.0",
    "sympy": "1.14.0",
    "tabulate": "0.9.0",
    "threadpoolctl": "3.6.0",
    "torch": "2.9.0+cpu",
    "torchvision": "0.24.0+cpu",
    "tqdm": "4.67.1",
    "tzdata": "2026.3",
    "typing-extensions": "4.16.0",
    "xgboost-cpu": "2.1.1",
}

SCIENCE_GPU_LOCK_SHA256 = "54163e0a58988e332202406d072913133cfa184447606969ac5b14b2d77d786d"
SCIENCE_GPU_ARCHIVE_COUNT = 62
SCIENCE_GPU_ARCHIVES_SHA256 = "c8dd857738553e22293296f06cca35a594e2ebfd946f2abf7420856a5453caf4"
SCIENCE_GPU_PYTHON = "3.12.7"
SCIENCE_GPU_SAM2_COMMIT = SCIENCE_CPU_SAM2_COMMIT
SCIENCE_GPU_SAM2_URL = SCIENCE_CPU_SAM2_URL
SCIENCE_GPU_CUDA_RUNTIME = "13.2"
SCIENCE_GPU_SAM2_NVCC_VERSION = "13.2.86"
SCIENCE_GPU_VERSIONS = {
    "antlr4-python3-runtime": "4.9.3",
    "compag-curation": "1.9.4rc10",
    "contourpy": "1.3.3",
    "cupy-cuda13x": "14.2.0",
    "cuda-bindings": "13.3.1",
    "cuda-pathfinder": "1.8.0",
    "cuda-toolkit": "13.2.1",
    "cycler": "0.12.1",
    "filelock": "3.32.4",
    "fonttools": "4.63.0",
    "fsspec": "2026.7.0",
    "hydra-core": "1.3.5",
    "imbalanced-learn": "0.14.0",
    "iopath": "0.1.10",
    "jinja2": "3.1.6",
    "joblib": "1.5.3",
    "kiwisolver": "1.5.0",
    "markupsafe": "3.0.3",
    "matplotlib": "3.10.7",
    "mpmath": "1.3.0",
    "networkx": "3.6.1",
    "numpy": "2.0.2",
    "nvidia-cublas": "13.4.0.1",
    "nvidia-cuda-cupti": "13.2.75",
    "nvidia-cuda-nvrtc": "13.2.78",
    "nvidia-cuda-runtime": "13.2.75",
    "nvidia-cudnn-cu13": "9.20.0.48",
    "nvidia-cufft": "12.2.0.46",
    "nvidia-cufile": "1.17.1.22",
    "nvidia-curand": "10.4.2.55",
    "nvidia-cusolver": "12.2.0.1",
    "nvidia-cusparse": "12.7.10.1",
    "nvidia-cusparselt-cu13": "0.8.1",
    "nvidia-nccl-cu12": "2.31.2",
    "nvidia-nccl-cu13": "2.29.7",
    "nvidia-nvjitlink": "13.3.33",
    "nvidia-nvshmem-cu13": "3.4.5",
    "nvidia-nvtx": "13.2.75",
    "omegaconf": "2.3.0",
    "opencv-python": "4.12.0.88",
    "packaging": "26.3",
    "pandas": "2.3.3",
    "pillow": "12.3.0",
    "portalocker": "3.2.0",
    "pyyaml": "6.0.3",
    "pyparsing": "3.3.2",
    "python-dateutil": "2.9.0.post0",
    "pytz": "2026.3.post1",
    "sam-2": "1.0",
    "scikit-learn": "1.7.2",
    "scipy": "1.16.2",
    "setuptools": "84.0.0",
    "six": "1.17.0",
    "sympy": "1.14.0",
    "tabulate": "0.9.0",
    "threadpoolctl": "3.6.0",
    "torch": "2.13.0+cu132",
    "torchvision": "0.28.0+cu132",
    "tqdm": "4.67.1",
    "triton": "3.7.1",
    "typing-extensions": "4.16.0",
    "tzdata": "2026.3",
    "wheel": "0.48.0",
    "xgboost": "2.1.1",
}

MAX_PACKAGE_IMPLEMENTATION_FILES = 1024
MAX_PACKAGE_IMPLEMENTATION_FILE_BYTES = 16 * 1024 * 1024
MAX_PACKAGE_IMPLEMENTATION_TOTAL_BYTES = 128 * 1024 * 1024
MAX_CONDA_PREFIX_HISTORY_BYTES = 16 * 1024 * 1024


def _science_directory_metadata_safe(info: os.stat_result) -> bool:
    required_owner_access = stat.S_IRUSR | stat.S_IXUSR
    return (
        stat.S_ISDIR(info.st_mode)
        and info.st_mode & required_owner_access == required_owner_access
        and not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    )


def _package_root() -> Path:
    return Path(__file__).absolute().parent


def _package_lstat(path: Path) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as exc:
        raise RuntimeError("package implementation metadata is unsafe") from exc


def _package_paths(root: Path, *, reject_bytecode: bool) -> tuple[tuple[str, ...], tuple[str, ...]]:
    files: list[str] = []
    bytecode: list[str] = []
    root_info = _package_lstat(root)
    if not stat.S_ISDIR(root_info.st_mode):
        raise RuntimeError("package implementation contains an unsafe directory")
    if not _science_directory_metadata_safe(root_info):
        raise RuntimeError("package implementation directory permissions are unsafe")

    def reject_walk_error(error: OSError) -> None:
        raise RuntimeError("package implementation directory traversal is unsafe") from error

    for directory, names, filenames in os.walk(
        root,
        topdown=True,
        onerror=reject_walk_error,
        followlinks=False,
    ):
        current = Path(directory)
        directory_info = _package_lstat(current)
        if not stat.S_ISDIR(directory_info.st_mode):
            raise RuntimeError("package implementation contains an unsafe directory")
        if not _science_directory_metadata_safe(directory_info):
            raise RuntimeError("package implementation directory permissions are unsafe")
        retained: list[str] = []
        for name in sorted(names):
            child = current / name
            info = _package_lstat(child)
            if not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("package implementation contains a symlink or special directory")
            if not _science_directory_metadata_safe(info):
                raise RuntimeError("package implementation directory permissions are unsafe")
            if name == "__pycache__":
                if reject_bytecode:
                    raise RuntimeError("installed package contains a bytecode cache directory")
                continue
            retained.append(name)
        names[:] = retained
        for name in sorted(filenames):
            child = current / name
            info = _package_lstat(child)
            relative = child.relative_to(root).as_posix()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise RuntimeError("package implementation contains a symlink, special file, or hard link")
            if child.suffix in {".pyc", ".pyo"}:
                bytecode.append(relative)
                continue
            if not info.st_mode & stat.S_IRUSR or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
                raise RuntimeError("package implementation file permissions are unsafe")
            files.append(relative)
    if reject_bytecode and bytecode:
        raise RuntimeError("installed package contains bytecode outside the bound source manifest")
    return tuple(sorted(files)), tuple(sorted(bytecode))


def _package_file_row(
    root: Path,
    relative: str,
) -> tuple[dict[str, object], tuple[int, ...]]:
    path = root / relative
    before_path = _package_lstat(path)
    if (
        not stat.S_ISREG(before_path.st_mode)
        or before_path.st_nlink != 1
        or not 0 < before_path.st_size <= MAX_PACKAGE_IMPLEMENTATION_FILE_BYTES
    ):
        raise RuntimeError("package implementation file metadata is unsafe")
    fd = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | getattr(os, "O_NOATIME", 0),
    )
    try:
        before = os.fstat(fd)
        if _metadata_tuple(before) != _metadata_tuple(before_path):
            raise RuntimeError("package implementation file changed while opening")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("package implementation file was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise RuntimeError("package implementation file grew while reading")
        after = os.fstat(fd)
    finally:
        os.close(fd)
    after_path = _package_lstat(path)
    if not (_metadata_tuple(before) == _metadata_tuple(after) == _metadata_tuple(after_path)):
        raise RuntimeError("package implementation file changed while reading")
    payload = b"".join(chunks)
    return (
        {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        },
        _metadata_tuple(after),
    )


def _attest_installation_directory_chain(prefix: Path, root: Path) -> None:
    """Pin and recheck every runtime directory from prefix through package root."""

    error = "science execution requires a safely permissioned runtime directory chain"
    try:
        relative = root.relative_to(prefix)
    except ValueError as exc:
        raise RuntimeError(error) from exc
    directory_flags = (
        os.O_RDONLY
        | os.O_CLOEXEC
        | os.O_NOFOLLOW
        | getattr(os, "O_DIRECTORY", 0)
    )
    records: list[tuple[int, int | None, str | None, tuple[int, ...]]] = []
    descriptors: list[int] = []
    try:
        prefix_path_before = prefix.lstat()
        prefix_fd = os.open(prefix, directory_flags)
        descriptors.append(prefix_fd)
        prefix_opened = os.fstat(prefix_fd)
        records.append((prefix_fd, None, None, _metadata_tuple(prefix_opened)))
        if (
            _metadata_tuple(prefix_opened) != _metadata_tuple(prefix_path_before)
            or not _science_directory_metadata_safe(prefix_opened)
        ):
            raise RuntimeError(error)

        parent_fd = prefix_fd
        for name in relative.parts:
            path_before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            child_fd = os.open(name, directory_flags, dir_fd=parent_fd)
            descriptors.append(child_fd)
            opened = os.fstat(child_fd)
            records.append((child_fd, parent_fd, name, _metadata_tuple(opened)))
            if (
                _metadata_tuple(opened) != _metadata_tuple(path_before)
                or not _science_directory_metadata_safe(opened)
            ):
                raise RuntimeError(error)
            parent_fd = child_fd

        for descriptor, parent_descriptor, name, expected in records:
            opened_after = os.fstat(descriptor)
            path_after = (
                prefix.lstat()
                if parent_descriptor is None
                else os.stat(
                    name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            )
            if not (
                _metadata_tuple(opened_after) == expected == _metadata_tuple(path_after)
                and _science_directory_metadata_safe(opened_after)
            ):
                raise RuntimeError(error)
    except OSError as exc:
        raise RuntimeError(error) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _attest_conda_prefix(prefix: Path) -> None:
    """Require a race-checked Conda/Mamba history below an opened prefix."""

    error = (
        "science execution requires an active virtual environment or an "
        "attested Conda/Mamba prefix"
    )
    conda_meta = prefix / "conda-meta"
    history = conda_meta / "history"
    prefix_fd: int | None = None
    conda_meta_fd: int | None = None
    history_fd: int | None = None
    try:
        prefix_path_before = prefix.lstat()
        conda_meta_path_before = conda_meta.lstat()
        history_path_before = history.lstat()
        directory_flags = (
            os.O_RDONLY
            | os.O_CLOEXEC
            | os.O_NOFOLLOW
            | getattr(os, "O_DIRECTORY", 0)
        )
        prefix_fd = os.open(prefix, directory_flags)
        prefix_opened_before = os.fstat(prefix_fd)
        if _metadata_tuple(prefix_opened_before) != _metadata_tuple(prefix_path_before):
            raise RuntimeError(error)
        conda_meta_fd = os.open("conda-meta", directory_flags, dir_fd=prefix_fd)
        conda_meta_opened_before = os.fstat(conda_meta_fd)
        if _metadata_tuple(conda_meta_opened_before) != _metadata_tuple(conda_meta_path_before):
            raise RuntimeError(error)
        history_fd = os.open(
            "history",
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=conda_meta_fd,
        )
        history_opened_before = os.fstat(history_fd)
        if _metadata_tuple(history_opened_before) != _metadata_tuple(history_path_before):
            raise RuntimeError(error)
        if (
            not _science_directory_metadata_safe(prefix_opened_before)
            or not _science_directory_metadata_safe(conda_meta_opened_before)
            or not stat.S_ISREG(history_opened_before.st_mode)
            or history_opened_before.st_nlink != 1
            or not 0 < history_opened_before.st_size <= MAX_CONDA_PREFIX_HISTORY_BYTES
            or not history_opened_before.st_mode & stat.S_IRUSR
            or history_opened_before.st_mode
            & (stat.S_IWGRP | stat.S_IWOTH | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        ):
            raise RuntimeError(error)
        chunks: list[bytes] = []
        remaining = history_opened_before.st_size
        while remaining:
            chunk = os.read(history_fd, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(error)
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(history_fd, 1):
            raise RuntimeError(error)
        history_opened_after = os.fstat(history_fd)
        conda_meta_opened_after = os.fstat(conda_meta_fd)
        prefix_opened_after = os.fstat(prefix_fd)
        history_path_after = history.lstat()
        conda_meta_path_after = conda_meta.lstat()
        prefix_path_after = prefix.lstat()
    except OSError as exc:
        raise RuntimeError(error) from exc
    finally:
        if history_fd is not None:
            os.close(history_fd)
        if conda_meta_fd is not None:
            os.close(conda_meta_fd)
        if prefix_fd is not None:
            os.close(prefix_fd)
    if not (
        _metadata_tuple(prefix_opened_before)
        == _metadata_tuple(prefix_opened_after)
        == _metadata_tuple(prefix_path_after)
        and _metadata_tuple(conda_meta_opened_before)
        == _metadata_tuple(conda_meta_opened_after)
        == _metadata_tuple(conda_meta_path_after)
        and _metadata_tuple(history_opened_before)
        == _metadata_tuple(history_opened_after)
        == _metadata_tuple(history_path_after)
    ):
        raise RuntimeError(error)
    payload = b"".join(chunks)
    if not payload.startswith(b"==> ") or b"\n# cmd: " not in payload:
        raise RuntimeError(error)


def package_implementation_identity(
    *,
    require_installed: bool = False,
    expected_distribution_version: str = __version__,
) -> dict[str, object]:
    """Hash the loaded package bytes without embedding an installation path."""

    root = _package_root()
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("package implementation root is unavailable or unsafe") from exc
    if root != resolved_root:
        raise RuntimeError("package implementation root contains a symlinked component")
    paths_before, bytecode_before = _package_paths(root, reject_bytecode=require_installed)
    if not paths_before or len(paths_before) > MAX_PACKAGE_IMPLEMENTATION_FILES:
        raise RuntimeError("package implementation file count is outside its bound")
    captured = [_package_file_row(root, relative) for relative in paths_before]
    rows = [row for row, _metadata in captured]
    total_bytes = sum(int(row["size_bytes"]) for row in rows)
    if total_bytes > MAX_PACKAGE_IMPLEMENTATION_TOTAL_BYTES:
        raise RuntimeError("package implementation exceeds its total byte bound")
    paths_after, bytecode_after = _package_paths(root, reject_bytecode=require_installed)
    if paths_before != paths_after or bytecode_before != bytecode_after:
        raise RuntimeError("package implementation namespace changed while hashing")
    closing = [_package_file_row(root, relative) for relative in paths_after]
    closing_rows = [row for row, _metadata in closing]
    if rows != closing_rows:
        raise RuntimeError("package implementation file changed after hashing")
    paths_final, bytecode_final = _package_paths(root, reject_bytecode=require_installed)
    if paths_after != paths_final or bytecode_after != bytecode_final:
        raise RuntimeError("package implementation namespace changed after hashing")
    for relative, (_row, metadata) in zip(paths_after, closing, strict=True):
        if _metadata_tuple(_package_lstat(root / relative)) != metadata:
            raise RuntimeError("package implementation file changed after closing hash")
    closure = {
        "schema": "compag-curation-package-implementation-manifest/v1",
        "rows": rows,
    }
    encoded = json.dumps(
        closure,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    identity = {
        "schema": "compag-curation-package-implementation-identity/v1",
        "sha256": hashlib.sha256(
            b"compag-curation-package-implementation-v1\0" + encoded
        ).hexdigest(),
        "file_count": len(rows),
        "size_bytes": total_bytes,
    }
    if not require_installed:
        return identity

    try:
        prefix = Path(sys.prefix).absolute()
        base_prefix = Path(sys.base_prefix).absolute()
        distribution = importlib.metadata.distribution("compag-curation")
        distribution_root = Path(distribution.locate_file("compag_curation")).resolve(strict=True)
    except (OSError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError("compag-curation is not an installed distribution in this runtime") from exc
    if not root.is_relative_to(prefix) or distribution_root != root:
        raise RuntimeError(
            "science execution requires the imported distribution to match the "
            "active runtime prefix"
        )
    _attest_installation_directory_chain(prefix, root)
    if prefix == base_prefix:
        _attest_conda_prefix(prefix)
    if distribution.version != expected_distribution_version:
        raise RuntimeError("installed compag-curation version differs from the runtime lock")
    direct_text = distribution.read_text("direct_url.json")
    if direct_text is not None:
        try:
            direct = json.loads(direct_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("compag-curation direct_url metadata is invalid") from exc
        directory = direct.get("dir_info") if isinstance(direct, dict) else None
        if isinstance(directory, dict) and directory.get("editable") is True:
            raise RuntimeError("editable compag-curation installs are unsupported for science execution")
    if not sys.dont_write_bytecode:
        raise RuntimeError("science execution requires bytecode generation to be disabled")
    if sys.flags.isolated != 1:
        raise RuntimeError("science execution requires Python isolated mode")
    if sys.pycache_prefix is not None:
        raise RuntimeError("science execution rejects an external bytecode-cache prefix")
    return {
        **identity,
        "installation": "SYS_PREFIX_DISTRIBUTION_NON_EDITABLE_ISOLATED_NO_BYTECODE",
    }


def science_cpu_runtime_policy() -> dict[str, object]:
    """Return the exact numerical-execution policy for the CPU profile."""

    policy: dict[str, object] = {
        "schema": "compag-curation-science-cpu-runtime-policy/v1",
        "device": "cpu",
        "environment": {
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "native_threadpool_limit": 1,
        "opencv_opencl": False,
        "opencv_threads": 1,
        "torch_deterministic_algorithms": True,
        "torch_interop_threads": 1,
        "torch_intraop_threads": 1,
    }
    encoded = json.dumps(
        policy,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return {**policy, "sha256": hashlib.sha256(encoded).hexdigest()}


def science_gpu_runtime_policy() -> dict[str, object]:
    """Return the deterministic FP32 numerical policy for science-gpu."""

    policy: dict[str, object] = {
        "schema": "compag-curation-science-gpu-runtime-policy/v1",
        "device": "cuda",
        "allow_cpu_fallback": False,
        "precision": "float32",
        "autocast": False,
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        },
        "native_threadpool_limit": 1,
        "opencv_opencl": False,
        "opencv_threads": 1,
        "torch_deterministic_algorithms": True,
        "torch_interop_threads": 1,
        "torch_intraop_threads": 1,
        "torch_float32_matmul_precision": "highest",
        "torch_default_dtype": "float32",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_allow_tf32": False,
        "cuda_matmul_allow_tf32": False,
    }
    encoded = json.dumps(
        policy,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return {**policy, "sha256": hashlib.sha256(encoded).hexdigest()}


def requested_cuda_device(device: str) -> tuple[str, int]:
    """Resolve an explicit CUDA request without permitting CPU fallback."""

    match = re.fullmatch(r"cuda(?::(0|[1-9][0-9]*))?", device)
    if match is None:
        raise RuntimeError("science-gpu requires device=cuda or device=cuda:<index>")
    index = int(match.group(1) or "0")
    return f"cuda:{index}", index


def cuda_visibility_identity(
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Capture a canonical CUDA visibility mapping without changing it."""

    selected = os.environ if environment is None else environment
    raw = selected.get("CUDA_VISIBLE_DEVICES")
    device_order = selected.get("CUDA_DEVICE_ORDER")
    if device_order not in {None, "FASTEST_FIRST", "PCI_BUS_ID"}:
        raise RuntimeError(
            "CUDA_DEVICE_ORDER must be unset, FASTEST_FIRST, or PCI_BUS_ID"
        )
    if raw is None:
        payload: dict[str, object] = {
            "schema": "compag-curation-cuda-visibility/v1",
            "mode": "UNSET_ALL_VISIBLE",
            "selector_count": None,
            "selector_sha256": None,
            "cuda_device_order": device_order,
        }
    else:
        if not isinstance(raw, str) or raw == "" or raw.strip() != raw:
            raise RuntimeError("CUDA_VISIBLE_DEVICES must be unset or a canonical nonempty mapping")
        selectors = raw.split(",")
        ordinal = re.compile(r"0|[1-9][0-9]*")
        gpu_uuid = re.compile(
            r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
            r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
        )
        if all(ordinal.fullmatch(value) is not None for value in selectors):
            mode = "EXPLICIT_ORDINALS"
        elif all(gpu_uuid.fullmatch(value) is not None for value in selectors):
            mode = "EXPLICIT_GPU_UUIDS"
        else:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES must contain only canonical ordinals or GPU UUIDs"
            )
        if len(selectors) != len(set(selectors)):
            raise RuntimeError("CUDA_VISIBLE_DEVICES contains duplicate selectors")
        payload = {
            "schema": "compag-curation-cuda-visibility/v1",
            "mode": mode,
            "selector_count": len(selectors),
            "selector_sha256": hashlib.sha256(raw.encode("ascii")).hexdigest(),
            "cuda_device_order": device_order,
        }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def preserved_cuda_visibility_environment(
    expected: Mapping[str, object],
) -> dict[str, str]:
    """Return the captured CUDA visibility binding or reject parent drift."""

    snapshot = {
        name: value
        for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
        if (value := os.environ.get(name)) is not None
    }
    current = cuda_visibility_identity(snapshot)
    if dict(expected) != current:
        raise RuntimeError("CUDA visibility mapping changed after dependency preflight")
    return snapshot


def confined_accelerator_cache_environment(runtime: Path) -> dict[str, str]:
    """Confine accelerator caches to an already nonempty, run-owned directory."""

    if not runtime.is_absolute():
        raise RuntimeError("science cache runtime path must be absolute")
    root = os.fspath(runtime)
    return {
        name: root
        for name in (
            "CUDA_CACHE_PATH",
            "CUPY_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
        )
    }


def _metadata_tuple(info: os.stat_result) -> tuple[int, ...]:
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


def science_cpu_lock_identity() -> Mapping[str, object]:
    resource = (Path(__file__).absolute().parent / "resources/science_cpu_lock.json").absolute()
    if resource != resource.resolve(strict=True):
        raise RuntimeError("science-cpu lock resource has a symlinked path component")
    path_before = resource.lstat()
    if not stat.S_ISREG(path_before.st_mode) or path_before.st_nlink != 1 or not 0 < path_before.st_size <= 64 * 1024:
        raise RuntimeError("science-cpu lock resource metadata is unsafe")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | getattr(os, "O_NOATIME", 0)
    fd = os.open(resource, flags)
    try:
        opened_before = os.fstat(fd)
        if _metadata_tuple(opened_before) != _metadata_tuple(path_before):
            raise RuntimeError("science-cpu lock resource changed while opening")
        chunks: list[bytes] = []
        remaining = opened_before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("science-cpu lock resource was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise RuntimeError("science-cpu lock resource grew while reading")
        opened_after = os.fstat(fd)
    finally:
        os.close(fd)
    path_after = resource.lstat()
    if not (_metadata_tuple(opened_before) == _metadata_tuple(opened_after) == _metadata_tuple(path_after)):
        raise RuntimeError("science-cpu lock resource changed while reading")
    payload = b"".join(chunks)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SCIENCE_CPU_LOCK_SHA256:
        raise RuntimeError("science-cpu lock resource hash differs from the reviewed lock")
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("science-cpu lock resource is invalid JSON") from exc
    expected = {
        "implementation": "CPython",
        "platform": "Linux x86_64",
        "python": SCIENCE_CPU_PYTHON,
        "sam2": {"commit_id": SCIENCE_CPU_SAM2_COMMIT, "url": SCIENCE_CPU_SAM2_URL, "vcs": "git"},
        "schema": "compag-curation-science-cpu-lock/v1",
        "versions": SCIENCE_CPU_VERSIONS,
    }
    if parsed != expected:
        raise RuntimeError("science-cpu lock resource content differs from the executable lock")
    return {**expected, "sha256": digest, "resource": "resources/science_cpu_lock.json"}


def science_gpu_lock_identity() -> Mapping[str, object]:
    """Validate and return the packaged rc4 science-gpu lock."""

    resource = (Path(__file__).absolute().parent / "resources/science_gpu_lock.json").absolute()
    if resource != resource.resolve(strict=True):
        raise RuntimeError("science-gpu lock resource has a symlinked path component")
    path_before = resource.lstat()
    if not stat.S_ISREG(path_before.st_mode) or path_before.st_nlink != 1 or not 0 < path_before.st_size <= 64 * 1024:
        raise RuntimeError("science-gpu lock resource metadata is unsafe")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | getattr(os, "O_NOATIME", 0)
    fd = os.open(resource, flags)
    try:
        opened_before = os.fstat(fd)
        if _metadata_tuple(opened_before) != _metadata_tuple(path_before):
            raise RuntimeError("science-gpu lock resource changed while opening")
        chunks: list[bytes] = []
        remaining = opened_before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("science-gpu lock resource was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise RuntimeError("science-gpu lock resource grew while reading")
        opened_after = os.fstat(fd)
    finally:
        os.close(fd)
    path_after = resource.lstat()
    if not (_metadata_tuple(opened_before) == _metadata_tuple(opened_after) == _metadata_tuple(path_after)):
        raise RuntimeError("science-gpu lock resource changed while reading")
    payload = b"".join(chunks)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SCIENCE_GPU_LOCK_SHA256:
        raise RuntimeError("science-gpu lock resource hash differs from the reviewed lock")
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("science-gpu lock resource is invalid JSON") from exc
    archives = parsed.get("archives") if isinstance(parsed, dict) else None
    expected_archive_names = tuple(
        sorted(set(SCIENCE_GPU_VERSIONS) - {"compag-curation", "sam-2"})
    )
    if (
        not isinstance(archives, dict)
        or len(archives) != SCIENCE_GPU_ARCHIVE_COUNT
        or tuple(archives) != expected_archive_names
    ):
        raise RuntimeError("science-gpu archive lock names or count are invalid")
    for name, row in archives.items():
        if (
            not isinstance(name, str)
            or re.sub(r"[-_.]+", "-", name).lower() != name
            or not isinstance(row, dict)
            or set(row) != {"sha256", "url", "version"}
            or row.get("version") != SCIENCE_GPU_VERSIONS[name]
        ):
            raise RuntimeError("science-gpu archive lock row is invalid")
        url = row.get("url")
        sha256 = row.get("sha256")
        if not isinstance(url, str) or not isinstance(sha256, str):
            raise RuntimeError("science-gpu archive lock URL or hash is invalid")
        split = urlsplit(url)
        if (
            split.scheme != "https"
            or not split.netloc
            or split.username is not None
            or split.password is not None
            or split.fragment
        ):
            raise RuntimeError("science-gpu archive lock URL is not canonical HTTPS")
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise RuntimeError("science-gpu archive lock SHA256 is not canonical")
    archives_encoded = json.dumps(
        archives,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    archives_digest = hashlib.sha256(
        b"compag-curation-science-gpu-archives-v1\0" + archives_encoded
    ).hexdigest()
    if archives_digest != SCIENCE_GPU_ARCHIVES_SHA256:
        raise RuntimeError("science-gpu archive lock mapping differs from the reviewed closure")
    expected = {
        "archives": archives,
        "cuda": {
            "pip_cuda_toolkit": "13.2.1",
            "pytorch_build": "cu132",
            "runtime": SCIENCE_GPU_CUDA_RUNTIME,
            "sam2_build_cuda": True,
            "sam2_build_toolchain": {
                "compiler": "conda cuda-nvcc",
                "release": "CUDA 13.2 Update 2",
                "version": SCIENCE_GPU_SAM2_NVCC_VERSION,
            },
        },
        "dependency_owners": {
            "nvidia-nccl-cu12": "xgboost 2.1.1 wheel dependency",
            "nvidia-nccl-cu13": "torch 2.13.0+cu132 wheel dependency",
        },
        "implementation": "CPython",
        "note": (
            "COMPAG Curation v1.9.4rc10 reuses the v1.9.3 shared Lite/Full/Full-image science-gpu dependency closure; GPU execution is "
            "required and CPU fallback is not permitted."
        ),
        "platform": "Linux x86_64",
        "profile": "science-gpu",
        "python": SCIENCE_GPU_PYTHON,
        "release": "v1.9.4rc10",
        "sam2": {
            "commit_id": SCIENCE_GPU_SAM2_COMMIT,
            "url": SCIENCE_GPU_SAM2_URL,
            "vcs": "git",
        },
        "schema": "compag-curation-science-gpu-lock/v2",
        "versions": SCIENCE_GPU_VERSIONS,
    }
    if parsed != expected:
        raise RuntimeError("science-gpu lock resource content differs from the executable lock")
    return {**expected, "sha256": digest, "resource": "resources/science_gpu_lock.json"}
