"""Typed contracts shared by the public domain APIs.

Planning is safe by default. Scientific execution requires a separate,
explicit policy and caller-injected services.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import shutil
import stat
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Callable, Mapping, Protocol, Sequence


SEALED_DECISION_THRESHOLD = 0.5


def _sealed_defaults() -> dict[str, float]:
    return {"decision_threshold": SEALED_DECISION_THRESHOLD}


def _deep_freeze(value: Any) -> Any:
    """Detach and recursively freeze JSON-like contract values."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    """Return detached JSON-compatible containers for serialization/callers."""

    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return [_deep_thaw(item) for item in sorted(value, key=repr)]
    return value


class ContractError(ValueError):
    """Raised before any scientific operation when a contract is incomplete."""


class ExecutionNotAuthorized(RuntimeError):
    """Raised when a caller attempts execution without controlled approval."""


class DomainExecutionError(RuntimeError):
    """Raised when an authorized domain handler cannot complete its contract."""


@dataclass(frozen=True)
class FileIdentity:
    """Stable identity fields for one regular file owned by a workflow."""

    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    def __post_init__(self) -> None:
        values = (
            self.device,
            self.inode,
            self.mode,
            self.size,
            self.mtime_ns,
            self.ctime_ns,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ContractError("file identity fields must be integers")
        if any(value < 0 for value in values):
            raise ContractError("file identity fields must be nonnegative")
        if not stat.S_ISREG(self.mode):
            raise ContractError("file identity must describe a regular file")

    @classmethod
    def from_stat(cls, info: os.stat_result) -> FileIdentity:
        return cls(*_stat_identity(info))

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.device,
            self.inode,
            self.mode,
            self.size,
            self.mtime_ns,
            self.ctime_ns,
        )


@dataclass(frozen=True)
class FileSnapshot:
    """Descriptor-stable identity and content digest for a regular file."""

    identity: FileIdentity
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, FileIdentity):
            raise ContractError("file snapshot identity must be typed")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ContractError("file snapshot SHA-256 must be lowercase hexadecimal")


@dataclass(frozen=True)
class DirectoryIdentity:
    """Stable device, inode and mode for a workflow-owned directory root."""

    device: int
    inode: int
    mode: int

    def __post_init__(self) -> None:
        values = (self.device, self.inode, self.mode)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ContractError("directory identity fields must be integers")
        if any(value < 0 for value in values):
            raise ContractError("directory identity fields must be nonnegative")
        if not stat.S_ISDIR(self.mode):
            raise ContractError("directory identity must describe a directory")

    @classmethod
    def from_stat(cls, info: os.stat_result) -> DirectoryIdentity:
        return cls(info.st_dev, info.st_ino, info.st_mode)


def _nonempty(value: Any, label: str) -> str:
    rendered = str(value).strip()
    if not rendered:
        raise ContractError(f"{label} must be a nonempty string")
    return rendered


@dataclass(frozen=True)
class ArtifactRef:
    """An explicit input or output artifact contract.

    SHA-256 is optional for a future output and required for sealed inputs.
    Paths are never opened while a plan is constructed.
    """

    name: str
    role: str
    path: str
    sha256: str | None = None
    required: bool = True
    kind: str = "file"

    def __post_init__(self) -> None:
        _nonempty(self.name, "artifact name")
        _nonempty(self.role, "artifact role")
        _nonempty(self.path, "artifact path")
        if self.kind not in {"file", "directory"}:
            raise ContractError("artifact kind must be file or directory")
        if not isinstance(self.required, bool):
            raise ContractError("artifact required flag must be boolean")
        if self.sha256 is not None and (
            len(self.sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.sha256)
        ):
            raise ContractError("artifact SHA-256 must be lowercase hexadecimal")


@dataclass(frozen=True)
class ExecutionPolicy:
    authorized: bool = False
    allow_download: bool = False
    output_collision: str = "fail"
    interpreter: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.authorized, bool) or not isinstance(self.allow_download, bool):
            raise ContractError("execution authorization flags must be boolean")
        if self.output_collision != "fail":
            raise ContractError("the public package supports only fail-on-collision")
        if self.interpreter is not None:
            _nonempty(self.interpreter, "interpreter")

    @property
    def python_executable(self) -> str:
        return resolve_python_executable(self.interpreter)


def resolve_python_executable(interpreter: str | None = None) -> str:
    """Resolve and validate a Python process boundary without using a shell."""

    configured = sys.executable if interpreter is None else _nonempty(interpreter, "interpreter")
    candidate = Path(configured)
    try:
        if candidate.is_absolute():
            resolved = candidate.resolve(strict=True)
        else:
            if "\\" in configured or len(candidate.parts) != 1:
                raise ContractError("interpreter must be an absolute path or a bare executable name")
            discovered = shutil.which(configured)
            if discovered is None:
                raise ContractError(f"interpreter executable was not found on PATH: {configured}")
            resolved = Path(discovered).resolve(strict=True)
        info = resolved.stat()
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError(f"interpreter executable is unavailable: {configured}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ContractError(f"interpreter executable is not a regular file: {configured}")
    if not os.access(resolved, os.X_OK):
        raise ContractError(f"interpreter executable is not executable: {configured}")
    return str(resolved)


@dataclass(frozen=True)
class NotificationConfig:
    enabled: bool = False
    topic: str | None = None
    endpoint: str | None = None
    priority: str = "5"
    tags: tuple[str, ...] = ("bell", "heavy_check_mark")
    timeout_seconds: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ContractError("notification enabled flag must be boolean")
        if self.enabled and (
            not isinstance(self.topic, str)
            or not self.topic.strip()
            or not isinstance(self.endpoint, str)
            or not self.endpoint.strip()
        ):
            raise ContractError("enabled notifications require explicit topic and endpoint")
        if not self.enabled and (self.topic is not None or self.endpoint is not None):
            raise ContractError("disabled notifications cannot retain topic or endpoint values")
        if self.priority != "5":
            raise ContractError("sealed notification priority must remain 5")
        if self.tags != ("bell", "heavy_check_mark"):
            raise ContractError("sealed notification tags changed")
        if self.timeout_seconds != 10:
            raise ContractError("sealed notification timeout must remain 10 seconds")


@dataclass(frozen=True)
class WorkflowStep:
    step_id: str
    operation: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    source_cell_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.step_id, "step ID")
        _nonempty(self.operation, "step operation")
        for label, values in (
            ("step input", self.inputs),
            ("step output", self.outputs),
            ("source cell ID", self.source_cell_ids),
        ):
            if not isinstance(values, tuple):
                raise ContractError(f"{label}s must be a tuple")
            normalized = [_nonempty(value, label) for value in values]
            if len(normalized) != len(set(normalized)):
                raise ContractError(f"{label}s must be unique within a workflow step")
        if set(self.inputs) & set(self.outputs):
            raise ContractError("a workflow step cannot overwrite one of its inputs")
        object.__setattr__(self, "parameters", _deep_freeze(self.parameters))


@dataclass(frozen=True)
class WorkflowPlan:
    command: str
    variant: str
    entry_point: str
    inputs: tuple[ArtifactRef, ...]
    outputs: tuple[ArtifactRef, ...]
    steps: tuple[WorkflowStep, ...]
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    sealed_defaults: Mapping[str, Any] = field(default_factory=_sealed_defaults)
    scientific_execution: bool = False
    runtime_behavioral_parity: str = "NOT_RETESTED_BY_DESIGN"
    scientific_result_parity: str = "NOT_RETESTED_BY_DESIGN"

    def __post_init__(self) -> None:
        _nonempty(self.command, "command")
        _nonempty(self.variant, "variant")
        entry_point = _nonempty(self.entry_point, "entry point")
        module_name, separator, symbol_name = entry_point.partition(":")
        if not separator or not module_name or not symbol_name or ":" in symbol_name:
            raise ContractError("entry point must use module:symbol form")
        if not all(part.isidentifier() for part in module_name.split(".")) or not symbol_name.isidentifier():
            raise ContractError("entry point module and symbol must be valid Python identifiers")
        if self.scientific_execution is not False:
            raise ContractError("plan construction cannot claim scientific execution")
        if self.runtime_behavioral_parity != "NOT_RETESTED_BY_DESIGN":
            raise ContractError("runtime behavioral parity cannot be claimed by a static plan")
        if self.scientific_result_parity != "NOT_RETESTED_BY_DESIGN":
            raise ContractError("scientific result parity cannot be claimed by a static plan")
        for label, values, item_type in (
            ("inputs", self.inputs, ArtifactRef),
            ("outputs", self.outputs, ArtifactRef),
            ("steps", self.steps, WorkflowStep),
        ):
            if not isinstance(values, tuple) or any(not isinstance(item, item_type) for item in values):
                raise ContractError(f"workflow {label} must be a typed tuple")
        artifacts = (*self.inputs, *self.outputs)
        names = [item.name for item in artifacts]
        if len(names) != len(set(names)):
            raise ContractError("artifact names must be unique within a workflow plan")
        input_contracts: dict[Path, tuple[str | None, str, bool]] = {}
        input_paths: list[Path] = []
        for item in self.inputs:
            path = _absolute(Path(item.path))
            contract = (item.sha256, item.kind, item.required)
            previous = input_contracts.setdefault(path, contract)
            if previous != contract:
                raise ContractError("input path aliases must have identical hash, kind, and requiredness")
            input_paths.append(path)
        output_paths = [_absolute(Path(item.path)) for item in self.outputs]
        if len(output_paths) != len(set(output_paths)):
            raise ContractError("output artifact paths must be unique within a workflow plan")
        for input_path in input_paths:
            for output_path in output_paths:
                if (
                    input_path == output_path
                    or input_path in output_path.parents
                    or output_path in input_path.parents
                ):
                    raise ContractError("input and output artifact paths must not overlap")
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ContractError("workflow step IDs must be unique")

        input_names = {item.name for item in self.inputs}
        output_names = {item.name for item in self.outputs}
        available = set(input_names)
        produced: set[str] = set()
        consumed: set[str] = set()
        for step in self.steps:
            missing = set(step.inputs) - available
            if missing:
                raise ContractError("workflow step references an unavailable input artifact")
            collisions = set(step.outputs) & available
            if collisions:
                raise ContractError("workflow step attempts to overwrite an available artifact")
            consumed.update(step.inputs)
            produced.update(step.outputs)
            available.update(step.outputs)
        if output_names - produced:
            raise ContractError("every declared output artifact must be produced by a workflow step")
        if input_names - consumed:
            raise ContractError("every declared input artifact must be consumed by a workflow step")
        orphaned = produced - output_names - consumed
        if orphaned:
            raise ContractError("workflow contains an unconsumed intermediate artifact")
        if self.sealed_defaults.get("decision_threshold") != SEALED_DECISION_THRESHOLD:
            raise ContractError("sealed decision threshold changed")
        object.__setattr__(self, "sealed_defaults", _deep_freeze(self.sealed_defaults))

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "variant": self.variant,
            "entry_point": self.entry_point,
            "inputs": [asdict(item) for item in self.inputs],
            "outputs": [asdict(item) for item in self.outputs],
            "steps": [
                {
                    "step_id": item.step_id,
                    "operation": item.operation,
                    "inputs": list(item.inputs),
                    "outputs": list(item.outputs),
                    "parameters": _deep_thaw(item.parameters),
                    "source_cell_ids": list(item.source_cell_ids),
                }
                for item in self.steps
            ],
            "policy": asdict(self.policy),
            "notification": asdict(self.notification),
            "sealed_defaults": _deep_thaw(self.sealed_defaults),
            "scientific_execution": self.scientific_execution,
            "runtime_behavioral_parity": self.runtime_behavioral_parity,
            "scientific_result_parity": self.scientific_result_parity,
            "schema": "compag-curation-domain-plan/v2",
        }


@dataclass(frozen=True)
class ExecutionResult:
    return_code: int
    output_artifacts: tuple[str, ...] = ()
    notification_sent: bool = False


class PlanRunner(Protocol):
    def __call__(self, plan: WorkflowPlan) -> ExecutionResult: ...


Notifier = Callable[[str, str], bool]


@dataclass(frozen=True)
class NotificationDelivery:
    """Complete message supplied to an injected notification transport."""

    topic: str
    endpoint: str
    title: str
    message: str
    priority: str = "5"
    tags: tuple[str, ...] = ("bell", "heavy_check_mark")
    timeout_seconds: int = 10


class NotificationSender(Protocol):
    def send(self, delivery: NotificationDelivery) -> bool: ...


class Reporter(Protocol):
    """Injected status sink used instead of Notebook display state."""

    def __call__(self, message: str) -> None: ...


def require_callable_shape(
    callback: object,
    label: str,
    /,
    *args: object,
    **kwargs: object,
) -> None:
    """Reject an unavailable or incompatible injected call boundary."""

    if not callable(callback):
        raise ContractError(f"{label} must be callable")
    try:
        inspect.signature(callback).bind(*args, **kwargs)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} has an incompatible call signature") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _runtime_identity() -> tuple[str, str]:
    """Resolve the dynamic user/host pair only when a handler needs it."""

    import getpass
    import platform

    return getpass.getuser(), platform.node()


@dataclass(frozen=True)
class HandlerServices:
    """Nonscientific services shared by concrete execution handlers.

    Scientific backends are deliberately injected by the owning domain
    module.  Keeping reporting, notification, and time sources here avoids
    ambient Notebook globals and makes disabled notifications inert.
    """

    report: Reporter
    notifier: Notifier | None = None
    utc_now: Callable[[], str] = _utc_now
    notification_sender: NotificationSender | None = None
    runtime_identity: Callable[[], tuple[str, str]] = _runtime_identity

    def __post_init__(self) -> None:
        self.validate_call_shapes()

    def validate_call_shapes(self) -> None:
        """Revalidate injected service signatures at a handler boundary."""

        require_callable_shape(self.report, "handler reporter", "message")
        require_callable_shape(self.utc_now, "handler UTC clock")
        require_callable_shape(
            self.runtime_identity,
            "handler runtime identity provider",
        )
        if self.notifier is not None:
            require_callable_shape(
                self.notifier,
                "handler notifier",
                "topic",
                "message",
            )
        if self.notification_sender is not None:
            require_callable_shape(
                getattr(self.notification_sender, "send", None),
                "handler notification sender",
                NotificationDelivery("topic", "endpoint", "title", "message"),
            )


def require_authorized(policy: ExecutionPolicy) -> None:
    """Reject a concrete handler call before it can touch an artifact."""

    if not policy.authorized:
        raise ExecutionNotAuthorized("controlled execution is not authorized by configuration")


def artifact_path(values: Mapping[str, Any], name: str) -> Path:
    """Resolve one validated semantic artifact locator without opening it."""

    try:
        value = values[name]
        path = value["path"]
    except (KeyError, TypeError) as exc:
        raise ContractError(f"workflow artifact is missing: {name}") from exc
    return Path(_nonempty(path, f"workflow artifact {name}"))


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _hash_file(path: Path, expected: os.stat_result | None = None) -> str:
    """Hash one regular file through a bound descriptor and reject path drift."""

    before = path.lstat() if expected is None else expected
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ContractError("artifact hash input is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContractError("artifact hash input cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _stat_identity(opened) != _stat_identity(before):
            raise ContractError("artifact changed before hashing completed")
        value = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            value.update(chunk)
        completed = os.fstat(descriptor)
        if _stat_identity(completed) != _stat_identity(opened):
            raise ContractError("artifact changed while it was hashed")
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise ContractError("artifact changed after it was hashed") from exc
    if _stat_identity(after) != _stat_identity(completed):
        raise ContractError("artifact changed after it was hashed")
    return value.hexdigest()


def _hash_tree(root: Path) -> str:
    root_before = root.lstat()
    if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode):
        raise ContractError("artifact tree root is not a directory")
    directory_states: dict[Path, tuple[int, int, int, int, int, int]] = {
        root: _stat_identity(root_before)
    }
    file_states: dict[Path, tuple[int, int, int, int, int, int]] = {}
    file_digests: dict[Path, str] = {}
    rows: list[bytes] = []
    paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    initial_inventory = tuple(path.relative_to(root).as_posix() for path in paths)
    for path in paths:
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            if stat.S_ISDIR(info.st_mode) and not path.is_symlink():
                directory_states[path] = _stat_identity(info)
                continue
            raise ContractError("artifact directory contains a non-regular node")
        relative = path.relative_to(root).as_posix()
        mode_octal = f"{stat.S_IMODE(info.st_mode):04o}"
        digest = _hash_file(path, info)
        rows.append(
            f"{relative}\0{mode_octal}\0{info.st_size}\0{digest}\n".encode("utf-8")
        )
        file_states[path] = _stat_identity(info)
        file_digests[path] = digest
    final_inventory = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    )
    if final_inventory != initial_inventory:
        raise ContractError("artifact tree changed while it was hashed")
    for path, expected_identity, expected_kind in (
        *((path, identity, "directory") for path, identity in directory_states.items()),
        *((path, identity, "file") for path, identity in file_states.items()),
    ):
        try:
            observed = path.lstat()
        except OSError as exc:
            raise ContractError("artifact tree changed while it was hashed") from exc
        valid_kind = (
            stat.S_ISDIR(observed.st_mode)
            if expected_kind == "directory"
            else stat.S_ISREG(observed.st_mode)
        )
        if (
            stat.S_ISLNK(observed.st_mode)
            or not valid_kind
            or _stat_identity(observed) != expected_identity
        ):
            raise ContractError("artifact tree changed while it was hashed")
        if expected_kind == "file" and _hash_file(path, observed) != file_digests[path]:
            raise ContractError("artifact tree changed while it was hashed")
    return hashlib.sha256(b"".join(rows)).hexdigest()


def _tree_inventory_rows(root: Path) -> tuple[bytes, ...]:
    rows: list[bytes] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        try:
            info = path.lstat()
        except OSError as exc:
            raise ContractError("artifact tree inventory changed while it was read") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ContractError("artifact directory contains a symbolic link")
        if stat.S_ISDIR(info.st_mode):
            kind = "directory"
            size = 0
        elif stat.S_ISREG(info.st_mode):
            kind = "file"
            size = info.st_size
        else:
            raise ContractError("artifact directory contains a non-regular node")
        relative = path.relative_to(root).as_posix()
        rows.append(
            f"{kind}\0{relative}\0{stat.S_IMODE(info.st_mode):04o}\0{size}\n".encode(
                "utf-8"
            )
        )
    return tuple(rows)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_absolute_real_ancestors(path: Path, role: str) -> Path:
    if not path.is_absolute():
        raise ContractError(f"configured {role} must use an absolute locator")
    cursor = path.parent
    while True:
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            parent = cursor.parent
            if parent == cursor:
                raise ContractError(f"configured {role} has no existing parent directory")
            cursor = parent
            continue
        except OSError as exc:
            raise ContractError(f"configured {role} parent cannot be inspected") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ContractError(f"configured {role} parent is not a real directory")
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    return path


def directory_identity(path: Path) -> DirectoryIdentity:
    """Read one real directory identity through a no-follow descriptor."""

    path = _require_absolute_real_ancestors(Path(path), "output directory")
    try:
        before = path.lstat()
    except OSError as exc:
        raise ContractError("output directory is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise ContractError("output directory must be a real directory")
    if path.resolve(strict=True) != _absolute(path):
        raise ContractError("output directory locator changed")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContractError("output directory cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if DirectoryIdentity.from_stat(opened) != DirectoryIdentity.from_stat(before):
            raise ContractError("output directory identity changed while opening")
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise ContractError("output directory identity changed after opening") from exc
    identity = DirectoryIdentity.from_stat(after)
    if identity != DirectoryIdentity.from_stat(before):
        raise ContractError("output directory identity changed after opening")
    return identity


def directory_tree_digest(path: Path) -> str:
    """Hash exact file bytes plus the complete stable directory inventory."""

    path = _require_absolute_real_ancestors(Path(path), "output directory")
    initial_identity = directory_identity(path)
    initial_inventory = _tree_inventory_rows(path)
    initial_content = _hash_tree(path)
    final_content = _hash_tree(path)
    final_inventory = _tree_inventory_rows(path)
    final_identity = directory_identity(path)
    if (
        final_identity != initial_identity
        or final_inventory != initial_inventory
        or final_content != initial_content
    ):
        raise ContractError("output directory changed while it was snapshotted")
    digest = hashlib.sha256()
    digest.update(b"A02A_DIRECTORY_TREE_SNAPSHOT_V1\0")
    digest.update(f"{stat.S_IMODE(initial_identity.mode):04o}\n".encode("ascii"))
    digest.update(initial_content.encode("ascii"))
    digest.update(b"\n")
    for row in initial_inventory:
        digest.update(row)
    return digest.hexdigest()


def regular_file_snapshot(path: Path) -> FileSnapshot:
    """Bind one regular file's exact identity and bytes through a stable read."""

    path = _require_absolute_real_ancestors(Path(path), "output file")
    try:
        before = path.lstat()
    except OSError as exc:
        raise ContractError("output file is unavailable") from exc
    identity = FileIdentity.from_stat(before)
    digest = _hash_file(path, before)
    try:
        after = path.lstat()
    except OSError as exc:
        raise ContractError("output file identity changed after hashing") from exc
    if FileIdentity.from_stat(after) != identity:
        raise ContractError("output file identity changed after hashing")
    return FileSnapshot(identity, digest)


def validate_artifact_preconditions(plan: WorkflowPlan) -> None:
    """Validate declared paths immediately before an authorized execution.

    The planner deliberately performs no filesystem reads.  This preflight is
    kept separate so help, import, config, and plan construction remain inert.
    """

    input_entries: list[tuple[ArtifactRef, Path]] = []
    output_paths = [_absolute(Path(artifact.path)) for artifact in plan.outputs]
    for artifact in plan.inputs:
        if not artifact.required:
            continue
        path = _require_absolute_real_ancestors(Path(artifact.path), artifact.role)
        try:
            info = path.lstat()
        except OSError as exc:
            raise ContractError(f"required {artifact.role} is unavailable; locator suppressed") from exc
        valid_type = (
            stat.S_ISREG(info.st_mode) and not path.is_symlink()
            if artifact.kind == "file"
            else stat.S_ISDIR(info.st_mode) and not path.is_symlink()
        )
        if not valid_type:
            raise ContractError(f"required {artifact.role} is unavailable; locator suppressed")
        if artifact.sha256 is not None:
            observed = _hash_file(path) if artifact.kind == "file" else _hash_tree(path)
            if observed != artifact.sha256:
                raise ContractError(f"required {artifact.role} hash does not match; locator suppressed")
        input_entries.append((artifact, _absolute(path)))
    input_paths = [path for _artifact, path in input_entries]
    input_contracts: dict[Path, tuple[str | None, str, bool]] = {}
    for artifact, path in input_entries:
        contract = (artifact.sha256, artifact.kind, artifact.required)
        previous = input_contracts.setdefault(path, contract)
        if previous != contract:
            raise ContractError("input path aliases have conflicting contracts")
    if len(output_paths) != len(set(output_paths)):
        raise ContractError("output artifact paths must be unique")
    for input_path in input_paths:
        for output_path in output_paths:
            if (
                input_path == output_path
                or input_path in output_path.parents
                or output_path in input_path.parents
            ):
                raise ContractError("input and output artifact paths must not overlap")
    for artifact in plan.outputs:
        path = _require_absolute_real_ancestors(Path(artifact.path), artifact.role)
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ContractError("configured output cannot be inspected; locator suppressed") from exc
        else:
            if plan.policy.output_collision == "fail":
                raise ContractError("configured output already exists; locator suppressed")


def prepare_output_parent(path: Path) -> Path:
    """Create and revalidate only the real parent chain for an absent output."""

    path = _require_absolute_real_ancestors(path, "output")
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ContractError("configured output cannot be inspected") from exc
    else:
        raise ContractError("configured output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_absolute_real_ancestors(path, "output")
    if path.parent.resolve(strict=True) != _absolute(path.parent) or path.parent.is_symlink():
        raise ContractError("configured output parent is not a real directory")
    return path.parent


def atomic_write_output(
    path: Path,
    writer: Callable[[BinaryIO], None],
    *,
    backup_path: Path | None = None,
    expected_source_identity: tuple[int, int, int, int, int, int] | None = None,
    discard_backup: bool = False,
) -> FileIdentity:
    """Write privately, then install by a same-directory no-clobber link."""

    if not callable(writer):
        raise ContractError("atomic output writer must be callable")
    if not isinstance(discard_backup, bool):
        raise ContractError("atomic output backup disposition must be boolean")
    if discard_backup and backup_path is None:
        raise ContractError("discarded backup requires a replacement output")
    original: os.stat_result | None = None
    if backup_path is None:
        if expected_source_identity is not None:
            raise ContractError("fresh atomic output cannot bind a replacement source")
        parent = prepare_output_parent(path)
    else:
        path = _require_absolute_real_ancestors(path, "replacement output")
        backup_path = _require_absolute_real_ancestors(backup_path, "replacement backup")
        if backup_path == path or backup_path.parent != path.parent:
            raise ContractError("replacement backup must be a distinct same-parent path")
        try:
            original = path.lstat()
        except OSError as exc:
            raise ContractError("replacement source is unavailable") from exc
        if stat.S_ISLNK(original.st_mode) or not stat.S_ISREG(original.st_mode):
            raise ContractError("replacement source must be a regular non-symlink file")
        if (
            expected_source_identity is not None
            and _stat_identity(original) != expected_source_identity
        ):
            raise ContractError("replacement source does not match the bound reader")
        try:
            backup_path.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ContractError("replacement backup cannot be inspected") from exc
        else:
            raise ContractError("replacement backup already exists")
        parent = path.parent
        if parent.resolve(strict=True) != _absolute(parent) or parent.is_symlink():
            raise ContractError("replacement parent is not a real directory")

    parent_before = parent.lstat()
    if stat.S_ISLNK(parent_before.st_mode) or not stat.S_ISDIR(parent_before.st_mode):
        raise ContractError("atomic output parent is not a real directory")
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        parent_descriptor = os.open(parent, parent_flags)
    except OSError as exc:
        raise ContractError("atomic output parent cannot be opened safely") from exc
    opened_parent = os.fstat(parent_descriptor)
    if (
        not stat.S_ISDIR(opened_parent.st_mode)
        or (opened_parent.st_dev, opened_parent.st_ino, opened_parent.st_mode)
        != (parent_before.st_dev, parent_before.st_ino, parent_before.st_mode)
    ):
        os.close(parent_descriptor)
        raise ContractError("atomic output parent identity changed")

    def stat_at(name: str) -> os.stat_result:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    temporary_name: str | None = None
    temporary_inode: tuple[int, int] | None = None
    installed = False
    backup_installed = False
    original_unlinked = False
    try:
        for index in range(256):
            candidate_name = f".{path.name}.tmp.{index:03d}"
            try:
                descriptor = os.open(
                    candidate_name,
                    flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            except OSError as exc:
                raise ContractError("atomic output temporary file cannot be created") from exc
            temporary_name = candidate_name
            break
        if descriptor < 0 or temporary_name is None:
            raise ContractError("atomic output has no available temporary sibling")
        os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ContractError("atomic output temporary node is not a private regular file")
        temporary_inode = (opened.st_dev, opened.st_ino)

        with os.fdopen(os.dup(descriptor), "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        completed = os.fstat(descriptor)
        try:
            temporary_after = stat_at(temporary_name)
        except OSError as exc:
            raise ContractError("atomic output temporary path changed") from exc
        if (
            not stat.S_ISREG(completed.st_mode)
            or completed.st_nlink != 1
            or _stat_identity(temporary_after) != _stat_identity(completed)
        ):
            raise ContractError("atomic output temporary identity changed")

        parent_after = parent.lstat()
        if (
            parent_after.st_dev,
            parent_after.st_ino,
            parent_after.st_mode,
        ) != (
            opened_parent.st_dev,
            opened_parent.st_ino,
            opened_parent.st_mode,
        ):
            raise ContractError("atomic output parent changed before installation")
        if original is None:
            try:
                stat_at(path.name)
            except FileNotFoundError:
                pass
            else:
                raise ContractError("configured output appeared before installation")
        else:
            current = stat_at(path.name)
            if _stat_identity(current) != _stat_identity(original):
                raise ContractError("replacement source changed before installation")
            assert backup_path is not None
            try:
                stat_at(backup_path.name)
            except FileNotFoundError:
                pass
            else:
                raise ContractError("replacement backup appeared before installation")
            try:
                os.link(
                    path.name,
                    backup_path.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise ContractError("replacement backup appeared before installation") from exc
            except OSError as exc:
                raise ContractError("replacement source could not be linked to backup") from exc
            backup_installed = True
            moved_original = stat_at(backup_path.name)
            current = stat_at(path.name)
            if (
                (
                    moved_original.st_dev,
                    moved_original.st_ino,
                    moved_original.st_mode,
                    moved_original.st_size,
                    moved_original.st_mtime_ns,
                )
                != (
                    original.st_dev,
                    original.st_ino,
                    original.st_mode,
                    original.st_size,
                    original.st_mtime_ns,
                )
                or (
                    current.st_dev,
                    current.st_ino,
                    current.st_mode,
                    current.st_size,
                    current.st_mtime_ns,
                )
                != (
                    original.st_dev,
                    original.st_ino,
                    original.st_mode,
                    original.st_size,
                    original.st_mtime_ns,
                )
            ):
                raise ContractError("replacement backup identity changed")
            os.unlink(path.name, dir_fd=parent_descriptor)
            original_unlinked = True

        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise ContractError("configured output appeared before installation") from exc
        except OSError as exc:
            raise ContractError("atomic output could not be installed") from exc
        installed = True
        linked = stat_at(path.name)
        if (
            stat.S_ISLNK(linked.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or (linked.st_dev, linked.st_ino) != temporary_inode
            or linked.st_size != completed.st_size
        ):
            raise ContractError("atomic output final identity changed")
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        final = stat_at(path.name)
        if (
            stat.S_ISLNK(final.st_mode)
            or not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or (final.st_dev, final.st_ino) != temporary_inode
            or final.st_size != completed.st_size
        ):
            raise ContractError("atomic output final identity changed")
        os.fsync(parent_descriptor)
        parent_final = parent.lstat()
        if (
            parent_final.st_dev,
            parent_final.st_ino,
            parent_final.st_mode,
        ) != (
            opened_parent.st_dev,
            opened_parent.st_ino,
            opened_parent.st_mode,
        ):
            raise ContractError("atomic output parent changed after installation")
        if discard_backup:
            assert backup_path is not None
            assert original is not None
            backup_current = stat_at(backup_path.name)
            if (
                stat.S_ISLNK(backup_current.st_mode)
                or not stat.S_ISREG(backup_current.st_mode)
                or (backup_current.st_dev, backup_current.st_ino)
                != (original.st_dev, original.st_ino)
            ):
                raise ContractError("replacement cleanup identity changed")
            os.unlink(backup_path.name, dir_fd=parent_descriptor)
            try:
                stat_at(backup_path.name)
            except FileNotFoundError:
                pass
            else:
                raise ContractError("replacement cleanup retained a backup")
            backup_installed = False
        return FileIdentity.from_stat(final)
    except Exception:
        if installed and temporary_inode is not None:
            try:
                current = stat_at(path.name)
                if (current.st_dev, current.st_ino) == temporary_inode:
                    os.unlink(path.name, dir_fd=parent_descriptor)
                    installed = False
            except OSError:
                pass
        if temporary_name is not None and temporary_inode is not None:
            try:
                current = stat_at(temporary_name)
                if (current.st_dev, current.st_ino) == temporary_inode:
                    os.unlink(temporary_name, dir_fd=parent_descriptor)
            except OSError:
                pass
        if backup_installed and backup_path is not None and original is not None:
            try:
                backup_current = stat_at(backup_path.name)
            except OSError:
                backup_current = None
            if backup_current is not None and (
                backup_current.st_dev,
                backup_current.st_ino,
            ) == (original.st_dev, original.st_ino):
                if original_unlinked:
                    try:
                        stat_at(path.name)
                    except FileNotFoundError:
                        try:
                            os.link(
                                backup_path.name,
                                path.name,
                                src_dir_fd=parent_descriptor,
                                dst_dir_fd=parent_descriptor,
                                follow_symlinks=False,
                            )
                        except OSError:
                            pass
                        else:
                            try:
                                os.unlink(backup_path.name, dir_fd=parent_descriptor)
                            except OSError:
                                pass
                    else:
                        if discard_backup:
                            try:
                                os.unlink(backup_path.name, dir_fd=parent_descriptor)
                            except OSError:
                                pass
                else:
                    try:
                        os.unlink(backup_path.name, dir_fd=parent_descriptor)
                    except OSError:
                        pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def create_output_directory(path: Path) -> Path:
    """Create one declared directory output and bind it to a real locator."""

    prepare_output_parent(path)
    try:
        path.mkdir(exist_ok=False)
        info = path.lstat()
    except OSError as exc:
        raise ContractError("configured output directory could not be created") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ContractError("configured output directory is not a real directory")
    if path.resolve(strict=True) != _absolute(path):
        raise ContractError("configured output directory locator changed")
    return path


def execute_plan(
    plan: WorkflowPlan,
    runner: PlanRunner,
    notifier: Notifier | None = None,
) -> ExecutionResult:
    """Execute only after explicit controlled authorization.

    Notification values are never inspected while notifications are disabled.
    A success notification is recorded only after both the runner and injected
    notifier report success.
    """

    if not plan.policy.authorized:
        raise ExecutionNotAuthorized("controlled execution is not authorized by configuration")
    require_callable_shape(runner, "plan runner", plan)
    if plan.notification.enabled:
        if notifier is None:
            raise ContractError("enabled notifications require an injected notifier")
        require_callable_shape(
            notifier,
            "plan notifier",
            plan.notification.topic or "",
            plan.notification.endpoint or "",
        )
    validate_artifact_preconditions(plan)
    result = runner(plan)
    if result.return_code != 0:
        return result
    sent = False
    if plan.notification.enabled:
        if notifier is None:
            raise ContractError("enabled notifications require an injected notifier")
        sent = bool(notifier(plan.notification.topic or "", plan.notification.endpoint or ""))
        if not sent:
            raise ContractError("notification callback did not confirm delivery")
    return ExecutionResult(
        return_code=result.return_code,
        output_artifacts=result.output_artifacts,
        notification_sent=sent,
    )
