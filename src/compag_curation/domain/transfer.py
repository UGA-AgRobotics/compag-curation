"""Source-backed WSL-to-Windows transfer adapters and artifact verification."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatchcase
from hashlib import sha256
import os
from pathlib import Path
from shutil import copyfile
import stat
from subprocess import CompletedProcess, run
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from ..config import DomainConfig
from ..contracts import (
    ContractError,
    DirectoryIdentity,
    FileIdentity,
    HandlerServices,
    WorkflowPlan,
    WorkflowStep,
    artifact_path,
    directory_identity,
    directory_tree_digest,
    require_callable_shape,
    require_authorized,
    validate_artifact_preconditions,
)
from ._shared import artifact, make_plan


class TransferContractError(ContractError):
    """Raised before copy when a transfer boundary is not valid."""


@dataclass(frozen=True)
class TransferProcessResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(closed_transfer_environment(self.environment)),
        )


class TransferRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        acceptable_exit_codes: frozenset[int],
        capture_output: bool = False,
    ) -> TransferProcessResult: ...


# SOURCE_CELL: NB-LIVE-0005-C0000
# SOURCE_CELL: NB-LIVE-0005-C0002
# SOURCE_STATEMENT_MAP: explicit-workflow-environment -> closed_transfer_environment
def closed_transfer_environment(injected: Mapping[str, str]) -> dict[str, str]:
    """Copy only values explicitly supplied by the authorized caller."""

    if not isinstance(injected, MappingABC):
        raise TransferContractError("transfer environment must be a string mapping")
    if any(type(key) is not str or type(value) is not str for key, value in injected.items()):
        raise TransferContractError("transfer environment keys and values must be strings")
    return dict(injected)


@dataclass(frozen=True)
class LocalTransferRunner:
    """Run transfer argv with a closed, explicitly injectable environment."""

    def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        acceptable_exit_codes: frozenset[int],
        capture_output: bool = False,
    ) -> TransferProcessResult:
        completed: CompletedProcess[str] = run(
            [str(part) for part in argv],
            check=False,
            text=True,
            capture_output=capture_output,
            env=closed_transfer_environment(env),
        )
        result = TransferProcessResult(
            tuple(str(part) for part in argv),
            completed.returncode,
            completed.stdout or "",
            completed.stderr or "",
            closed_transfer_environment(env),
        )
        if result.returncode not in acceptable_exit_codes:
            raise RuntimeError(
                f"transfer command returned {result.returncode}; "
                f"accepted={sorted(acceptable_exit_codes)}"
            )
        return result


@dataclass(frozen=True)
class TransferRequest:
    source: Path
    destination: Path
    variant: Literal["result-tree", "csv-and-text-probe"] = "result-tree"
    include_patterns: tuple[str, ...] = ("*.*",)
    probe_workspace: Path | None = None
    path_converter: str = "wslpath"
    robocopy_executable: str = "robocopy.exe"
    powershell_executable: str = "powershell.exe"
    command_executable: str = "cmd.exe"
    expected_source_sha256: Mapping[str, str] = field(default_factory=dict)
    expected_tree_sha256: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(closed_transfer_environment(self.environment)),
        )


@dataclass(frozen=True)
class TransferResult:
    destination: Path
    source_windows: str
    destination_windows: str
    return_codes: tuple[int, ...]
    final_status: int
    verified_source_sha256: Mapping[str, str]
    verified_destination_sha256: Mapping[str, str]
    local_outputs: tuple[Path, ...] = ()
    destination_identity: DirectoryIdentity | None = None
    workspace_identity: DirectoryIdentity | None = None
    verified_workspace_sha256: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verified_source_sha256",
            MappingProxyType(dict(self.verified_source_sha256)),
        )
        object.__setattr__(
            self,
            "verified_destination_sha256",
            MappingProxyType(dict(self.verified_destination_sha256)),
        )
        object.__setattr__(
            self,
            "verified_workspace_sha256",
            MappingProxyType(dict(self.verified_workspace_sha256)),
        )


@dataclass(frozen=True)
class TransferWorkflowServices:
    common: HandlerServices
    runner: TransferRunner
    clock: Callable[[], datetime] | None = None
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.common, HandlerServices):
            raise ContractError("transfer execution requires typed handler services")
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(closed_transfer_environment(self.environment)),
        )
        self.validate_call_shapes()

    def validate_call_shapes(self) -> None:
        """Validate process and clock injection without invoking either."""

        if not isinstance(self.common, HandlerServices):
            raise ContractError("transfer execution requires typed handler services")
        self.common.validate_call_shapes()
        closed_transfer_environment(self.environment)
        require_callable_shape(
            getattr(self.runner, "run", None),
            "transfer runner",
            (),
            env={},
            acceptable_exit_codes=frozenset({0}),
            capture_output=False,
        )
        if self.clock is not None:
            require_callable_shape(self.clock, "transfer clock")


def _ps_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _cmd_literal(value: str) -> str:
    if any(character in value for character in '\r\n"&|<>^%!'):
        raise TransferContractError("converted Windows path contains shell metacharacters")
    return f'"{value}"'


def _sha256_file(path: Path) -> str:
    try:
        before = path.lstat()
    except OSError as exc:
        raise TransferContractError("transfer file is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TransferContractError("transfer file must be regular and non-symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TransferContractError("transfer file cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if FileIdentity.from_stat(opened) != FileIdentity.from_stat(before):
            raise TransferContractError("transfer file changed before hashing")
        digest = sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        completed = os.fstat(descriptor)
        if FileIdentity.from_stat(completed) != FileIdentity.from_stat(opened):
            raise TransferContractError("transfer file changed while hashing")
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise TransferContractError("transfer file changed after hashing") from exc
    if path.is_symlink() or FileIdentity.from_stat(after) != FileIdentity.from_stat(completed):
        raise TransferContractError("transfer file changed after hashing")
    return digest.hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_safe_directory(path: Path, role: str) -> Path:
    if not path.is_absolute():
        raise TransferContractError(f"{role} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError as exc:
        raise TransferContractError(f"{role} is unavailable") from exc
    if resolved != _absolute(path) or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise TransferContractError(f"{role} must be a non-symlink directory")
    return path


def _prepare_output_path(
    path: Path,
    role: str,
    *,
    direct_parent: Path | None = None,
) -> Path:
    if not path.is_absolute():
        raise TransferContractError(f"{role} must be an absolute path")
    if path.exists() or path.is_symlink():
        raise TransferContractError(f"{role} already exists")
    parent = path.parent
    cursor = parent
    while not cursor.exists() and cursor != cursor.parent:
        cursor = cursor.parent
    try:
        cursor_info = cursor.lstat()
        cursor_resolved = cursor.resolve(strict=True)
    except OSError as exc:
        raise TransferContractError(f"{role} parent is unavailable") from exc
    if (
        cursor_resolved != _absolute(cursor)
        or stat.S_ISLNK(cursor_info.st_mode)
        or not stat.S_ISDIR(cursor_info.st_mode)
    ):
        raise TransferContractError(f"{role} parent is not symlink-safe")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.resolve(strict=True) != _absolute(parent) or parent.is_symlink():
        raise TransferContractError(f"{role} parent is not symlink-safe")
    if direct_parent is not None and parent != direct_parent:
        raise TransferContractError(f"{role} must be a direct child of the transfer source")
    return path


def _prepare_output_directory(path: Path, role: str) -> Path:
    _prepare_output_path(path, role)
    path.mkdir(parents=False, exist_ok=False)
    return path


def verify_source_artifacts(
    source: Path,
    expected: Mapping[str, str],
) -> Mapping[str, str]:
    """Verify explicitly named source artifacts before crossing the boundary."""

    verified: dict[str, str] = {}
    for relative, expected_digest in sorted(expected.items()):
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or any(part in {".", ".."} for part in relative_path.parts)
        ):
            raise TransferContractError(
                "expected transfer artifact must be a normalized relative path"
            )
        candidate = source / relative_path
        try:
            source_root = source.resolve(strict=True)
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise TransferContractError(
                f"expected transfer artifact is missing: {candidate}"
            ) from exc
        if (
            not candidate.is_absolute()
            or not resolved.is_relative_to(source_root)
            or resolved != _absolute(candidate)
            or candidate.is_symlink()
            or not candidate.is_file()
        ):
            raise TransferContractError(f"expected transfer artifact is missing: {candidate}")
        actual = _sha256_file(candidate)
        if actual.lower() != str(expected_digest).lower():
            raise TransferContractError(
                f"sha256 mismatch for {relative}: expected {expected_digest}, got {actual}"
            )
        verified[relative] = actual
    return verified


def _tree_sha256(root: Path) -> str:
    records: list[bytes] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise TransferContractError("transfer source tree contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            raise TransferContractError("transfer source tree contains a non-regular node")
        relative = path.relative_to(root).as_posix()
        records.append(
            f"{relative}\0{path.stat().st_size}\0{_sha256_file(path)}\n".encode("utf-8")
        )
    return sha256(b"".join(records)).hexdigest()


def _matches_pattern(name: str, pattern: str) -> bool:
    if pattern == "*.*":
        return True
    return fnmatchcase(name.lower(), pattern.lower())


def _selected_files(
    root: Path,
    patterns: Sequence[str],
    excludes: Sequence[str] = (),
) -> dict[str, str]:
    selected: dict[str, str] = {}
    paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    initial_inventory = tuple(path.relative_to(root).as_posix() for path in paths)
    for path in paths:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise TransferContractError("transfer tree contains a symbolic link")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise TransferContractError("transfer tree contains a non-regular node")
        if not any(_matches_pattern(path.name, pattern) for pattern in patterns):
            continue
        if any(_matches_pattern(path.name, pattern) for pattern in excludes):
            continue
        selected[path.relative_to(root).as_posix()] = _sha256_file(path)
    final_inventory = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    )
    if final_inventory != initial_inventory:
        raise TransferContractError("transfer tree changed while it was inspected")
    return selected


def _require_transfer_directory_identity(
    path: Path,
    expected: DirectoryIdentity | None,
    role: str,
) -> DirectoryIdentity:
    try:
        current = directory_identity(path)
    except ContractError as exc:
        raise TransferContractError(f"{role} identity is unavailable") from exc
    if expected is not None and current != expected:
        raise TransferContractError(f"{role} identity changed")
    return current


def _capture_transfer_tree(
    root: Path,
    role: str,
) -> tuple[DirectoryIdentity, Mapping[str, str]]:
    identity = _require_transfer_directory_identity(root, None, role)
    files = _selected_files(root, ("*.*",))
    _require_transfer_directory_identity(root, identity, role)
    return identity, MappingProxyType(files)


def _require_transfer_tree_snapshot(
    root: Path,
    expected_identity: DirectoryIdentity,
    expected_files: Mapping[str, str],
    role: str,
) -> None:
    _require_transfer_directory_identity(root, expected_identity, role)
    observed = _selected_files(root, ("*.*",))
    _require_transfer_directory_identity(root, expected_identity, role)
    if observed != dict(expected_files):
        raise TransferContractError(f"{role} content changed")


def _require_transfer_result_current(result: TransferResult) -> None:
    if result.destination_identity is None:
        raise TransferContractError("transfer destination identity is missing")
    _require_transfer_tree_snapshot(
        result.destination,
        result.destination_identity,
        result.verified_destination_sha256,
        "transfer destination",
    )
    if result.workspace_identity is not None:
        if not result.local_outputs:
            raise TransferContractError("transfer workspace outputs are missing")
        workspace = result.local_outputs[0].parent
        if any(path.parent != workspace for path in result.local_outputs):
            raise TransferContractError("transfer workspace outputs changed location")
        _require_transfer_tree_snapshot(
            workspace,
            result.workspace_identity,
            result.verified_workspace_sha256,
            "CSV/text probe workspace",
        )
    elif result.verified_workspace_sha256:
        raise TransferContractError("transfer workspace identity is missing")


def _verify_destination_copy(
    source: Path,
    destination: Path,
    patterns: Sequence[str],
    excludes: Sequence[str] = (),
) -> Mapping[str, str]:
    expected = _selected_files(source, patterns, excludes)
    observed = _selected_files(
        _require_safe_directory(destination, "transfer destination"),
        ("*.*",),
    )
    if observed != expected:
        raise TransferContractError("transfer destination does not match selected source bytes")
    return observed


def _verify_tree_if_requested(request: TransferRequest) -> None:
    if request.expected_tree_sha256 is None:
        return
    observed = _tree_sha256(request.source)
    if observed != request.expected_tree_sha256:
        raise TransferContractError("transfer source tree SHA-256 does not match")


def _run_transfer_command(
    request: TransferRequest,
    runner: TransferRunner,
    argv: Sequence[str],
    *,
    acceptable_exit_codes: frozenset[int],
    capture_output: bool = False,
) -> TransferProcessResult:
    """Run one exact argv with a closed environment and verify its result."""

    normalized_argv = tuple(str(part) for part in argv)
    process = runner.run(
        normalized_argv,
        env=closed_transfer_environment(request.environment),
        acceptable_exit_codes=acceptable_exit_codes,
        capture_output=capture_output,
    )
    if not isinstance(process, TransferProcessResult):
        raise TransferContractError("transfer runner returned an invalid result")
    if process.argv != normalized_argv:
        raise TransferContractError("transfer runner changed the configured argv")
    if dict(process.environment) != closed_transfer_environment(request.environment):
        raise TransferContractError("transfer runner did not bind the closed environment")
    if process.returncode not in acceptable_exit_codes:
        raise TransferContractError(
            f"transfer command returned {process.returncode}; "
            f"accepted={sorted(acceptable_exit_codes)}"
        )
    return process


def _convert_path(
    request: TransferRequest,
    path: Path,
    runner: TransferRunner,
) -> str:
    converted = _run_transfer_command(
        request,
        runner,
        (request.path_converter, "-w", str(path)),
        acceptable_exit_codes=frozenset({0}),
        capture_output=True,
    )
    rendered = converted.stdout.strip()
    if not rendered:
        raise TransferContractError("path converter returned an empty Windows path")
    return rendered


def _require_pending_destination(path: Path) -> Path:
    _prepare_output_path(path, "transfer destination")
    return path


# SOURCE_CELL: NB-LIVE-0005-C0000
# SOURCE_STATEMENT_MAP: probe-create-robocopy-list -> transfer_result_tree
def execute_result_tree_transfer(
    request: TransferRequest,
    *,
    runner: TransferRunner,
) -> TransferResult:
    """Execute the exact result-tree probe and robocopy contract."""

    if runner is None:
        raise TransferContractError("result-tree transfer requires an explicit runner")
    _require_safe_directory(request.source, "transfer source")
    destination_path = _require_pending_destination(request.destination)
    if request.include_patterns != ("*.*",):
        raise TransferContractError("result-tree transfer requires the sealed *.* pattern")
    service = runner
    _verify_tree_if_requested(request)
    verified = verify_source_artifacts(request.source, request.expected_source_sha256)
    source_windows = _convert_path(request, request.source, service)
    destination_parent_windows = _convert_path(request, destination_path.parent, service)
    destination_windows = _convert_path(request, destination_path, service)

    checked = _run_transfer_command(
        request,
        service,
        (
            request.powershell_executable,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"[bool](Test-Path {_ps_literal(destination_parent_windows)})",
        ),
        acceptable_exit_codes=frozenset(range(256)),
        capture_output=True,
    )
    if checked.stdout.strip() != "True":
        raise TransferContractError("configured RDP share is not reachable")
    _run_transfer_command(
        request,
        service,
        (
            request.powershell_executable,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"[void](New-Item -ItemType Directory -Path {_ps_literal(destination_windows)})",
        ),
        acceptable_exit_codes=frozenset({0}),
    )
    copied = _run_transfer_command(
        request,
        service,
        (
            request.robocopy_executable,
            source_windows,
            destination_windows,
            "*.*",
            "/E",
            "/R:1",
            "/W:1",
            "/NFL",
            "/NDL",
            "/NP",
            "/NJH",
            "/NJS",
            "/XF",
            "*.jpg",
        ),
        acceptable_exit_codes=frozenset(range(8)),
    )
    destination_created_identity = _require_transfer_directory_identity(
        destination_path,
        None,
        "transfer destination",
    )
    _run_transfer_command(
        request,
        service,
        (
            request.powershell_executable,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"Get-ChildItem -Force -LiteralPath {_ps_literal(destination_windows)} | "
            "Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize",
        ),
        acceptable_exit_codes=frozenset(range(256)),
    )
    _require_transfer_directory_identity(
        destination_path,
        destination_created_identity,
        "transfer destination",
    )
    destination_verified = _verify_destination_copy(
        request.source,
        destination_path,
        request.include_patterns,
        ("*.jpg",),
    )
    destination_identity, destination_snapshot = _capture_transfer_tree(
        destination_path,
        "transfer destination",
    )
    if destination_identity != destination_created_identity:
        raise TransferContractError("transfer destination identity changed")
    if dict(destination_snapshot) != dict(destination_verified):
        raise TransferContractError("transfer destination snapshot changed")
    return TransferResult(
        destination=destination_path,
        source_windows=source_windows,
        destination_windows=destination_windows,
        return_codes=(copied.returncode,),
        final_status=0,
        verified_source_sha256=verified,
        verified_destination_sha256=destination_verified,
        local_outputs=(),
        destination_identity=destination_identity,
    )


# SOURCE_CELL: NB-LIVE-0005-C0002
# SOURCE_STATEMENT_MAP: cmd-probe-fixtures-filtered-robocopy-list -> transfer_csv_and_text_probe
def execute_csv_text_probe_transfer(
    request: TransferRequest,
    *,
    runner: TransferRunner,
    clock: Callable[[], datetime] | None = None,
) -> TransferResult:
    """Create the source fixtures and copy CSV/TXT with visible robocopy status."""

    if runner is None:
        raise TransferContractError("CSV/text transfer requires an explicit runner")
    source = _require_safe_directory(request.source, "transfer source")
    destination_path = _require_pending_destination(request.destination)
    if request.include_patterns != ("*.csv", "*.txt"):
        raise TransferContractError(
            "CSV/text transfer requires the sealed *.csv then *.txt patterns"
        )
    if request.probe_workspace is None:
        raise TransferContractError(
            "CSV/text transfer requires an explicit probe workspace"
        )
    _verify_tree_if_requested(request)
    verified = verify_source_artifacts(request.source, request.expected_source_sha256)
    workspace = _prepare_output_path(
        request.probe_workspace,
        "CSV/text probe workspace",
    )
    if workspace == destination_path or workspace in destination_path.parents or destination_path in workspace.parents:
        raise TransferContractError("transfer outputs must not overlap")
    workspace.mkdir(parents=False, exist_ok=False)
    workspace_identity = _require_transfer_directory_identity(
        workspace,
        None,
        "CSV/text probe workspace",
    )
    for relative in _selected_files(source, request.include_patterns):
        source_path = source / relative
        staged_path = workspace / relative
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source_path, staged_path)
    _require_transfer_directory_identity(
        workspace,
        workspace_identity,
        "CSV/text probe workspace",
    )
    now = (clock or (lambda: datetime.now().astimezone()))()
    _require_transfer_directory_identity(
        workspace,
        workspace_identity,
        "CSV/text probe workspace",
    )
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    probe_text = _prepare_output_path(
        workspace / f"___probe_{timestamp}.txt",
        "probe text output",
        direct_parent=workspace,
    )
    probe_csv = _prepare_output_path(
        workspace / f"___probe_{timestamp}.csv",
        "probe CSV output",
        direct_parent=workspace,
    )
    csv_log = _prepare_output_path(workspace / "robocopy_csv.log", "CSV transfer log")
    text_log = _prepare_output_path(workspace / "robocopy_txt.log", "text transfer log")
    if len({probe_text, probe_csv, csv_log, text_log}) != 4:
        raise TransferContractError("transfer output paths must be unique")
    _require_transfer_directory_identity(
        workspace,
        workspace_identity,
        "CSV/text probe workspace",
    )
    service = runner
    source_windows = _convert_path(request, workspace, service)
    _require_transfer_directory_identity(
        workspace,
        workspace_identity,
        "CSV/text probe workspace",
    )
    destination_parent_windows = _convert_path(request, destination_path.parent, service)
    _require_transfer_directory_identity(
        workspace,
        workspace_identity,
        "CSV/text probe workspace",
    )
    destination_windows = _convert_path(request, destination_path, service)
    _require_transfer_directory_identity(
        workspace,
        workspace_identity,
        "CSV/text probe workspace",
    )

    probe = _run_transfer_command(
        request,
        service,
        (
            request.command_executable,
            "/c",
            f"dir {_cmd_literal(destination_parent_windows)} 1>nul 2>nul",
        ),
        acceptable_exit_codes=frozenset(range(256)),
    )
    if probe.returncode != 0:
        raise TransferContractError("configured RDP share is not reachable")
    _require_transfer_directory_identity(
        workspace,
        workspace_identity,
        "CSV/text probe workspace",
    )
    _run_transfer_command(
        request,
        service,
        (
            request.command_executable,
            "/c",
            f"mkdir {_cmd_literal(destination_windows)} 1>nul 2>nul",
        ),
        acceptable_exit_codes=frozenset({0}),
    )
    _require_transfer_directory_identity(
        workspace,
        workspace_identity,
        "CSV/text probe workspace",
    )

    probe_text.write_text(
        f"probe from WSL {now.astimezone().isoformat()}\n",
        encoding="utf-8",
        newline="\n",
    )
    _require_transfer_directory_identity(
        workspace,
        workspace_identity,
        "CSV/text probe workspace",
    )
    probe_csv.write_text(
        "a,b\n1,2\n",
        encoding="utf-8",
        newline="\n",
    )
    _require_transfer_directory_identity(
        workspace,
        workspace_identity,
        "CSV/text probe workspace",
    )

    return_codes: list[int] = []
    destination_created_identity: DirectoryIdentity | None = None
    for pattern, log_path in (
        ("*.csv", csv_log),
        ("*.txt", text_log),
    ):
        log_windows = _convert_path(request, log_path, service)
        _require_transfer_directory_identity(
            workspace,
            workspace_identity,
            "CSV/text probe workspace",
        )
        copied = _run_transfer_command(
            request,
            service,
            (
                request.robocopy_executable,
                source_windows,
                destination_windows,
                pattern,
                "/E",
                "/R:1",
                "/W:1",
                "/NFL",
                "/NDL",
                "/NP",
                "/NJH",
                "/NJS",
                "/MT:8",
                f"/LOG:{log_windows}",
            ),
            acceptable_exit_codes=frozenset(range(256)),
        )
        _require_transfer_directory_identity(
            workspace,
            workspace_identity,
            "CSV/text probe workspace",
        )
        if destination_created_identity is None:
            destination_created_identity = _require_transfer_directory_identity(
                destination_path,
                None,
                "transfer destination",
            )
        else:
            _require_transfer_directory_identity(
                destination_path,
                destination_created_identity,
                "transfer destination",
            )
        return_codes.append(copied.returncode)
    _run_transfer_command(
        request,
        service,
        (
            request.command_executable,
            "/c",
            f"dir /s /b {_cmd_literal(destination_windows + '\\\\*.csv')} "
            f"{_cmd_literal(destination_windows + '\\\\*.txt')} 2>nul | more +0",
        ),
        acceptable_exit_codes=frozenset(range(256)),
    )
    _require_transfer_directory_identity(
        workspace,
        workspace_identity,
        "CSV/text probe workspace",
    )
    if destination_created_identity is None:
        raise TransferContractError("transfer destination was not created")
    _require_transfer_directory_identity(
        destination_path,
        destination_created_identity,
        "transfer destination",
    )
    final_status = 8 if any(code >= 8 for code in return_codes) else 0
    for path, role in ((csv_log, "CSV transfer log"), (text_log, "text transfer log")):
        if not path.is_file() or path.is_symlink():
            raise TransferContractError(f"{role} was not created as a regular file")
    destination_verified = _verify_destination_copy(
        workspace,
        destination_path,
        request.include_patterns,
    )
    destination_identity, destination_snapshot = _capture_transfer_tree(
        destination_path,
        "transfer destination",
    )
    if destination_identity != destination_created_identity:
        raise TransferContractError("transfer destination identity changed")
    if dict(destination_snapshot) != dict(destination_verified):
        raise TransferContractError("transfer destination snapshot changed")
    captured_workspace_identity, workspace_snapshot = _capture_transfer_tree(
        workspace,
        "CSV/text probe workspace",
    )
    if captured_workspace_identity != workspace_identity:
        raise TransferContractError("CSV/text probe workspace identity changed")
    return TransferResult(
        destination=destination_path,
        source_windows=source_windows,
        destination_windows=destination_windows,
        return_codes=tuple(return_codes),
        final_status=final_status,
        verified_source_sha256=verified,
        verified_destination_sha256=destination_verified,
        local_outputs=(probe_text, probe_csv, csv_log, text_log),
        destination_identity=destination_identity,
        workspace_identity=workspace_identity,
        verified_workspace_sha256=workspace_snapshot,
    )


def _transfer_request(
    config: DomainConfig,
    expected_variant: str,
    environment: Mapping[str, str],
) -> TransferRequest:
    values = config.workflow("transfer")
    return TransferRequest(
        source=artifact_path(values, "source"),
        destination=artifact_path(values, "destination"),
        variant=expected_variant,  # type: ignore[arg-type]
        include_patterns=tuple(values["include_patterns"]),
        probe_workspace=(
            artifact_path(values, "probe_workspace")
            if expected_variant == "csv-and-text-probe"
            else None
        ),
        expected_tree_sha256=values.get("expected_sha256"),
        environment=environment,
    )


def transfer_result_tree(
    config: DomainConfig,
    services: TransferWorkflowServices,
) -> TransferResult:
    if not isinstance(services, TransferWorkflowServices):
        raise ContractError("transfer handler requires TransferWorkflowServices")
    services.validate_call_shapes()
    values = config.workflow("transfer")
    if values["variant"] != "result-tree":
        raise ContractError("transfer handler requires result-tree variant")
    require_authorized(config.execution)
    validate_artifact_preconditions(plan_transfer(config, "result-tree"))
    result = execute_result_tree_transfer(
        _transfer_request(config, "result-tree", services.environment),
        runner=services.runner,
    )
    _require_transfer_result_current(result)
    destination_digest = directory_tree_digest(result.destination)
    try:
        services.common.report(f"transfer destination: {result.destination}")
    finally:
        _require_transfer_result_current(result)
        if directory_tree_digest(result.destination) != destination_digest:
            raise TransferContractError("transfer destination changed during completion report")
    return result


def transfer_csv_text_probe(
    config: DomainConfig,
    services: TransferWorkflowServices,
) -> TransferResult:
    if not isinstance(services, TransferWorkflowServices):
        raise ContractError("transfer handler requires TransferWorkflowServices")
    services.validate_call_shapes()
    values = config.workflow("transfer")
    if values["variant"] != "csv-and-text-probe":
        raise ContractError("transfer handler requires csv-and-text-probe variant")
    require_authorized(config.execution)
    validate_artifact_preconditions(plan_transfer(config, "csv-and-text-probe"))
    result = execute_csv_text_probe_transfer(
        _transfer_request(config, "csv-and-text-probe", services.environment),
        runner=services.runner,
        clock=services.clock,
    )
    _require_transfer_result_current(result)
    destination_digest = directory_tree_digest(result.destination)
    if not result.local_outputs:
        raise TransferContractError("transfer workspace outputs are missing")
    workspace = result.local_outputs[0].parent
    workspace_digest = directory_tree_digest(workspace)
    try:
        services.common.report(f"transfer destination: {result.destination}")
    finally:
        _require_transfer_result_current(result)
        if (
            directory_tree_digest(result.destination) != destination_digest
            or directory_tree_digest(workspace) != workspace_digest
        ):
            raise TransferContractError("transfer outputs changed during completion report")
    return result


def plan_transfer(config: DomainConfig, variant: str | None = None) -> WorkflowPlan:
    values = config.workflow("transfer")
    selected = variant or str(values["variant"])
    if selected not in {"result-tree", "csv-and-text-probe"}:
        raise ContractError("unknown transfer variant")
    inputs = (artifact(values["source"], "source", "transfer-source"),)
    outputs = [
        artifact(values["destination"], "destination", "transfer-destination", output=True),
    ]
    if selected == "csv-and-text-probe":
        outputs.append(
            artifact(
                values["probe_workspace"],
                "probe_workspace",
                "transfer-probe-workspace",
                output=True,
            )
        )
    parameters = {
        "expected_sha256": values.get("expected_sha256"),
        "include_patterns": tuple(values.get("include_patterns", ())),
        "verify_after_copy": True,
    }
    cells = (
        ("NB-LIVE-0005-C0000",)
        if selected == "result-tree"
        else ("NB-LIVE-0005-C0002",)
    )
    steps = (
        WorkflowStep(
            "transfer-preflight",
            "validate-transfer-boundary",
            ("source",),
            source_cell_ids=cells,
        ),
        WorkflowStep(
            "transfer-copy",
            "copy-and-verify-artifacts",
            ("source",),
            tuple(item.name for item in outputs),
            parameters,
            cells,
        ),
    )
    return make_plan(
        config,
        "transfer",
        selected,
        "compag_curation.domain.transfer:plan_transfer",
        inputs,
        tuple(outputs),
        steps,
    )


__all__ = [
    "TransferWorkflowServices",
    "plan_transfer",
    "transfer_csv_text_probe",
    "transfer_result_tree",
]
