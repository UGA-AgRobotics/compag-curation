"""Coverage-v2/v3 shell semantics expressed as typed Python orchestration."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from hashlib import sha256
import os
from pathlib import Path
import stat
from subprocess import CompletedProcess, run
from sys import executable as current_python
from types import MappingProxyType
from typing import Literal, Mapping, Protocol, Sequence

from ...contracts import FileIdentity, create_output_directory, resolve_python_executable


CoverageWorkflow = Literal[
    "coverage-basic",
    "coverage-points-overlay",
    "coverage-predictions-overlay",
    "coverage-reviewability",
]

class CoverageContractError(ValueError):
    pass


@dataclass(frozen=True)
class CoverageProcessResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    environment: Mapping[str, str] = field(default_factory=dict)
    cwd: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(closed_coverage_environment(self.environment)),
        )


class CoverageRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None,
        env: Mapping[str, str],
        acceptable_exit_codes: frozenset[int] = frozenset({0}),
    ) -> CoverageProcessResult: ...


# SOURCE_CELL: NB-LIVE-0004-C0001
# SOURCE_CELL: NB-LIVE-0004-C0003
# SOURCE_CELL: NB-LIVE-0004-C0004
# SOURCE_CELL: NB-LIVE-0004-C0006
# SOURCE_STATEMENT_MAP: explicit-workflow-environment -> closed_coverage_environment
def closed_coverage_environment(injected: Mapping[str, str]) -> dict[str, str]:
    """Copy only values explicitly supplied by the authorized caller."""

    if not isinstance(injected, MappingABC):
        raise CoverageContractError("coverage environment must be a string mapping")
    if any(type(key) is not str or type(value) is not str for key, value in injected.items()):
        raise CoverageContractError("coverage environment keys and values must be strings")
    return dict(injected)


class LocalCoverageRunner:
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None,
        env: Mapping[str, str],
        acceptable_exit_codes: frozenset[int] = frozenset({0}),
    ) -> CoverageProcessResult:
        completed: CompletedProcess[str] = run(
            [str(part) for part in argv],
            cwd=None if cwd is None else str(cwd),
            env=closed_coverage_environment(env),
            check=False,
            text=True,
            capture_output=False,
        )
        result = CoverageProcessResult(
            tuple(str(part) for part in argv),
            completed.returncode,
            completed.stdout or "",
            completed.stderr or "",
            closed_coverage_environment(env),
            cwd,
        )
        if result.returncode not in acceptable_exit_codes:
            raise RuntimeError(
                f"coverage process returned {result.returncode}; "
                f"accepted={sorted(acceptable_exit_codes)}"
            )
        return result


@dataclass(frozen=True)
class CoverageInputArtifact:
    """One exact executable/data input admitted to the coverage process."""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise CoverageContractError("coverage input path must be absolute")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise CoverageContractError(
                "coverage input SHA-256 must be lowercase hexadecimal"
            )


@dataclass(frozen=True)
class CoverageRequest:
    workflow: CoverageWorkflow
    image_id: str
    predictions: Path
    annotations: Path
    tile_index: CoverageInputArtifact
    output: Path
    script: CoverageInputArtifact
    images_root: Path | None = None
    review_candidates: Path | None = None
    python_executable: Path = field(default_factory=lambda: Path(current_python))
    mode: str = "xgb_recall"
    xgb_threshold: float = 0.50
    active_learning_margin: float = 0.20
    nms_iou: float = 0.50
    max_area_fraction: float = 0.05
    cwd: Path | None = None


@dataclass(frozen=True)
class CoverageResult:
    output: Path
    environment_file: Path
    argv: tuple[str, ...]
    process: CoverageProcessResult


@dataclass(frozen=True)
class CoverageEnvironmentBinding:
    path: Path
    identity: FileIdentity
    sha256: str


def _write_new_coverage_file(path: Path, content: bytes) -> FileIdentity:
    """Create one no-clobber file and bind its completed descriptor identity."""

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise CoverageContractError(
            "coverage environment artifact could not be created atomically"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise CoverageContractError(
                "coverage environment artifact is not a private regular file"
            )
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise CoverageContractError(
                    "coverage environment artifact write did not complete"
                )
            offset += written
        os.fsync(descriptor)
        completed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        installed = path.lstat()
    except OSError as exc:
        raise CoverageContractError(
            "coverage environment artifact disappeared after writing"
        ) from exc
    identity = FileIdentity.from_stat(completed)
    if (
        stat.S_ISLNK(installed.st_mode)
        or not stat.S_ISREG(installed.st_mode)
        or FileIdentity.from_stat(installed) != identity
    ):
        raise CoverageContractError(
            "coverage environment artifact identity changed after writing"
        )
    return identity


def _require_coverage_process_result(
    process: CoverageProcessResult,
    argv: Sequence[str],
    environment: Mapping[str, str],
    cwd: Path | None,
    acceptable_exit_codes: frozenset[int],
) -> CoverageProcessResult:
    """Enforce the injected runner result contract at the domain boundary."""

    if not isinstance(process, CoverageProcessResult):
        raise CoverageContractError("coverage runner returned an invalid result")
    if process.argv != tuple(str(part) for part in argv):
        raise CoverageContractError("coverage runner changed the configured argv")
    if dict(process.environment) != closed_coverage_environment(environment):
        raise CoverageContractError("coverage runner did not bind the closed environment")
    if process.cwd != cwd:
        raise CoverageContractError("coverage runner changed the configured working directory")
    if process.returncode not in acceptable_exit_codes:
        raise CoverageContractError(
            f"coverage process returned {process.returncode}; "
            f"accepted={sorted(acceptable_exit_codes)}"
        )
    return process


def _require_file(path: Path, role: str) -> Path:
    if not path.is_file():
        raise CoverageContractError(f"missing {role}: {path}")
    return path


def _verify_input_artifact(artifact: CoverageInputArtifact, role: str) -> Path:
    path = artifact.path
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        resolved = path.resolve(strict=True)
        before = path.lstat()
    except OSError as exc:
        raise CoverageContractError(f"missing {role}") from exc
    if resolved != absolute or stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CoverageContractError(f"{role} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CoverageContractError(f"cannot open {role} safely") from exc
    try:
        opened = os.fstat(descriptor)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        expected_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if identity != expected_identity or not stat.S_ISREG(opened.st_mode):
            raise CoverageContractError(f"{role} changed before hashing")
        digest = sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        completed = os.fstat(descriptor)
        completed_identity = (
            completed.st_dev,
            completed.st_ino,
            completed.st_mode,
            completed.st_size,
            completed.st_mtime_ns,
            completed.st_ctime_ns,
        )
        if completed_identity != identity:
            raise CoverageContractError(f"{role} changed while hashing")
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise CoverageContractError(f"{role} changed after hashing") from exc
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if after_identity != completed_identity:
        raise CoverageContractError(f"{role} changed after hashing")
    if digest.hexdigest() != artifact.sha256:
        raise CoverageContractError(f"{role} SHA-256 mismatch")
    return path


def _require_directory(path: Path, role: str) -> Path:
    if not path.is_dir():
        raise CoverageContractError(f"missing {role}: {path}")
    return path


def _shell_double_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
    return f'"{escaped}"'


def coverage_environment(request: CoverageRequest) -> tuple[tuple[str, str], ...]:
    """Return the source heredoc variables in their original order."""

    values: list[tuple[str, str]] = [
        ("TEST_ROOT", str(request.output)),
        ("MODE", request.mode),
        ("PYTHON_BIN", str(request.python_executable)),
        ("SCRIPT", str(request.script.path)),
        ("CVAT_XML", str(request.annotations)),
        ("THR_XGB", f"{request.xgb_threshold:.2f}"),
    ]
    if request.workflow == "coverage-reviewability":
        values.append(("AL_MARGIN", f"{request.active_learning_margin:.2f}"))
    values.append(("NMS_IOU", f"{request.nms_iou:.2f}"))
    if request.workflow != "coverage-reviewability":
        values.append(("MAX_AREA_FRAC", f"{request.max_area_fraction:.2f}"))
    if request.images_root is not None:
        values.append(("IMAGES_ROOT", str(request.images_root)))
    return tuple(values)


# SOURCE_CELL: NB-LIVE-0004-C0000
# SOURCE_CELL: NB-LIVE-0004-C0005
# SOURCE_STATEMENT_MAP: set-euo-heredoc-export-order -> write_coverage_environment
def _write_coverage_environment_binding(
    request: CoverageRequest,
) -> CoverageEnvironmentBinding:
    """Atomically write and bind the source-compatible environment artifact."""

    create_output_directory(request.output)
    version = "v3" if request.workflow == "coverage-reviewability" else "v2"
    path = request.output / f"eval_env__coverage_{version}__{request.mode}.sh"
    body = "".join(
        f"export {key}={_shell_double_quote(value)}\n"
        for key, value in coverage_environment(request)
    )
    encoded = body.encode("utf-8")
    identity = _write_new_coverage_file(path, encoded)
    return CoverageEnvironmentBinding(path, identity, sha256(encoded).hexdigest())


def write_coverage_environment(request: CoverageRequest) -> Path:
    """Write the source-compatible export file without evaluating shell text."""

    return _write_coverage_environment_binding(request).path


# SOURCE_CELL: NB-LIVE-0004-C0001
# SOURCE_CELL: NB-LIVE-0004-C0003
# SOURCE_CELL: NB-LIVE-0004-C0004
# SOURCE_CELL: NB-LIVE-0004-C0006
# SOURCE_STATEMENT_MAP: workflow-case-validation-argv-order -> build_coverage_argv
def build_coverage_argv(request: CoverageRequest) -> tuple[str, ...]:
    """Build a no-shell argv vector preserving the v2/v3 option order."""

    if request.workflow not in {
        "coverage-basic",
        "coverage-points-overlay",
        "coverage-predictions-overlay",
        "coverage-reviewability",
    }:
        raise CoverageContractError(f"unknown coverage workflow: {request.workflow}")
    try:
        python_executable = Path(resolve_python_executable(str(request.python_executable)))
    except ValueError as exc:
        raise CoverageContractError("configured Python interpreter is unavailable") from exc
    script = _verify_input_artifact(request.script, "coverage evaluator")
    _require_file(request.annotations, "CVAT XML")
    _require_file(request.predictions, "prediction CSV")
    tile_index = _verify_input_artifact(request.tile_index, "tile index CSV")
    if request.workflow in {
        "coverage-points-overlay",
        "coverage-predictions-overlay",
        "coverage-reviewability",
    }:
        if request.images_root is None:
            raise CoverageContractError("overlay coverage requires images_root")
        _require_directory(request.images_root, "full-card image directory")
    if request.workflow == "coverage-reviewability":
        if request.review_candidates is None:
            raise CoverageContractError("reviewability coverage requires review_candidates")
        _require_file(request.review_candidates, "active-learning candidates CSV")

    argv = [
        str(python_executable),
        str(script),
        "--cvat_xml",
        str(request.annotations),
        "--pred_csv",
        str(request.predictions),
        "--tile_index",
        str(tile_index),
    ]
    if request.workflow == "coverage-reviewability":
        argv.extend(["--al_candidates_csv", str(request.review_candidates)])
    argv.extend(
        [
            "--images",
            request.image_id,
            "--stage",
            "xgb",
            "--thr_xgb",
            f"{request.xgb_threshold:.2f}",
        ]
    )
    if request.workflow == "coverage-reviewability":
        argv.extend(["--al_margin", f"{request.active_learning_margin:.2f}"])
    argv.extend(["--nms_iou", f"{request.nms_iou:.2f}"])
    if request.workflow != "coverage-reviewability":
        argv.extend(["--max_area_frac", f"{request.max_area_fraction:.2f}"])
    if request.workflow in {
        "coverage-points-overlay",
        "coverage-predictions-overlay",
        "coverage-reviewability",
    }:
        argv.extend(["--images_root", str(request.images_root)])
    if request.workflow == "coverage-predictions-overlay":
        argv.extend(
            ["--overlay_preds", "--overlay_only_matched_preds", "--overlay_max_preds", "150"]
        )
    argv.extend(["--out_dir", str(request.output)])
    return tuple(argv)


# SOURCE_CELL: NB-LIVE-0004-C0001
# SOURCE_CELL: NB-LIVE-0004-C0003
# SOURCE_CELL: NB-LIVE-0004-C0004
# SOURCE_CELL: NB-LIVE-0004-C0006
# SOURCE_STATEMENT_MAP: source-env-process-exit -> execute_coverage
def execute_coverage(
    request: CoverageRequest,
    *,
    runner: CoverageRunner,
) -> CoverageResult:
    if runner is None:
        raise CoverageContractError("coverage execution requires an explicit runner")
    argv = build_coverage_argv(request)
    environment_binding = _write_coverage_environment_binding(request)
    environment_file = environment_binding.path
    output_identity = request.output.lstat()
    env = {key: value for key, value in coverage_environment(request)}
    acceptable_exit_codes = frozenset({0})
    process = runner.run(
        argv,
        cwd=request.cwd,
        env=env,
        acceptable_exit_codes=acceptable_exit_codes,
    )
    process = _require_coverage_process_result(
        process,
        argv,
        env,
        request.cwd,
        acceptable_exit_codes,
    )
    try:
        completed_output_identity = request.output.lstat()
        completed_environment_identity = environment_file.lstat()
    except OSError as exc:
        raise CoverageContractError(
            "coverage process removed a declared output artifact"
        ) from exc
    if (
        stat.S_ISLNK(completed_output_identity.st_mode)
        or not stat.S_ISDIR(completed_output_identity.st_mode)
        or completed_output_identity.st_dev != output_identity.st_dev
        or completed_output_identity.st_ino != output_identity.st_ino
    ):
        raise CoverageContractError("coverage process changed the declared output root")
    if (
        stat.S_ISLNK(completed_environment_identity.st_mode)
        or not stat.S_ISREG(completed_environment_identity.st_mode)
        or (
            completed_environment_identity.st_dev,
            completed_environment_identity.st_ino,
            completed_environment_identity.st_mode,
            completed_environment_identity.st_size,
            completed_environment_identity.st_mtime_ns,
            completed_environment_identity.st_ctime_ns,
        )
        != environment_binding.identity.as_tuple()
        or environment_file.resolve(strict=True).parent
        != request.output.resolve(strict=True)
    ):
        raise CoverageContractError(
            "coverage environment artifact changed after process execution"
        )
    _verify_input_artifact(
        CoverageInputArtifact(environment_file, environment_binding.sha256),
        "coverage environment artifact",
    )
    return CoverageResult(request.output, environment_file, argv, process)
