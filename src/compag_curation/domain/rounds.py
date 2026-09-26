"""Round bootstrap and notification services with explicit state.

The source Notebook exported dozens of process-environment variables and then
relied on later cells to discover them.  This module represents the same
semantic values as immutable data and returns them to the caller.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..contracts import (
    ContractError,
    HandlerServices,
    NotificationConfig,
    NotificationDelivery,
)


ROUND_NAME = re.compile(r"^IMG_(\d+)$")
MODEL_ROUND = re.compile(
    r"^(?P<prefix>.*_xgb_)r(?P<round>\d+)(?P<suffix>_hybrid)\.pkl$"
)
DEFAULT_NOTIFICATION_TITLE = "COMPAG Curation"
TRAINING_COMPLETED_TEXT = "Training completed"
LAST_CELL_EXECUTED_TEXT = "\N{WHITE HEAVY CHECK MARK} Last cell executed"


@dataclass(frozen=True)
class RoundBootstrapRequest:
    image_name: str
    mode: str
    output_root: Path
    artifact_root: Path
    split_root: Path
    feature_output: Path
    model_root: Path
    requested_model_name: str
    decision_threshold: float = 0.5
    fold_tag: str = "train_foldA"
    positive_category: str = "cj"
    pca_dimensions: int = 32
    force_cpu: bool = False
    backup_existing: bool = False
    backup_root: Path | None = None

    def __post_init__(self) -> None:
        if not ROUND_NAME.fullmatch(self.image_name):
            raise ContractError("image_name must use IMG_<integer> form")
        if self.mode not in {"gate", "xgb", "xgb_recall"}:
            raise ContractError("round mode is invalid")
        if self.decision_threshold != 0.5:
            raise ContractError("sealed decision threshold must remain 0.5")
        if self.pca_dimensions != 32:
            raise ContractError("sealed PCA dimensionality must remain 32")
        if not self.requested_model_name.endswith(".pkl"):
            raise ContractError("requested model name must end in .pkl")


@dataclass(frozen=True)
class RoundContext:
    image_name: str
    mode: str
    output_root: Path
    artifact_root: Path
    split_root: Path
    feature_output: Path
    model_root: Path
    model_name: str
    model_directory_tag: str
    decision_threshold: float
    fold_tag: str
    positive_category: str
    pca_dimensions: int
    force_cpu: bool
    backup_directory: Path | None


def _allocate_model_name(model_root: Path, requested: str) -> tuple[str, str]:
    """Allocate the first unused rN model/directory pair."""

    match = MODEL_ROUND.match(requested)
    if match is None:
        rounds: list[int] = []
        for candidate in model_root.iterdir():
            directory_match = re.fullmatch(r"r(\d+)_hybrid", candidate.name)
            if candidate.is_dir() and directory_match is not None:
                rounds.append(int(directory_match.group(1)))
        round_number = max(rounds) + 1 if rounds else 1
        stem = requested.rsplit(".pkl", 1)[0]
        tag = f"r{round_number}_hybrid"
        name = (
            f"{stem}_r{round_number}_hybrid.pkl"
            if "_r" not in stem
            else requested
        )
        return name, tag
    round_number = int(match.group("round"))
    while True:
        tag = f"r{round_number}_hybrid"
        if not (model_root / tag).exists():
            name = (
                f"{match.group('prefix')}r{round_number}"
                f"{match.group('suffix')}.pkl"
            )
            return name, tag
        round_number += 1


def _backup_files(
    sources: Iterable[Path],
    backup_directory: Path,
) -> tuple[Path, ...]:
    copied: list[Path] = []
    for source in sources:
        if source.is_file() and not source.is_symlink():
            destination = backup_directory / source.name
            shutil.copy2(source, destination)
            copied.append(destination)
    return tuple(copied)


@dataclass(frozen=True)
class RoundBackupHelperBinding:
    """Usable checked-in replacement for the Notebook-written helper module."""

    implementation_fqn: str
    backup_files: Callable[[Iterable[Path], Path], tuple[Path, ...]]
    runtime_files_written: int
    search_path_mutations: int
    ambient_environment_allowed: bool

    def __post_init__(self) -> None:
        if self.implementation_fqn != "compag_curation.domain.rounds:_backup_files":
            raise ContractError("round backup helper FQN is not the checked-in API")
        if not callable(self.backup_files):
            raise ContractError("round backup helper must expose a callable")
        if type(self.runtime_files_written) is not int or self.runtime_files_written != 0:
            raise ContractError("round backup helper cannot write a runtime module")
        if type(self.search_path_mutations) is not int or self.search_path_mutations != 0:
            raise ContractError("round backup helper cannot mutate the search path")
        if self.ambient_environment_allowed is not False:
            raise ContractError("round backup helper requires explicit typed inputs")


def bind_round_backup_helper() -> RoundBackupHelperBinding:
    """Expose the stable helper API without writing code or mutating import state."""

    binding = RoundBackupHelperBinding(
        implementation_fqn="compag_curation.domain.rounds:_backup_files",
        backup_files=_backup_files,
        runtime_files_written=0,
        search_path_mutations=0,
        ambient_environment_allowed=False,
    )
    if (
        not callable(binding.backup_files)
        or binding.runtime_files_written != 0
        or binding.search_path_mutations != 0
        or binding.ambient_environment_allowed is not False
    ):
        raise ContractError("round backup helper binding violates the static API policy")
    return binding


# SOURCE_CELL: NB-LIVE-0001-C0000
# SOURCE_STATEMENT_MAP: backup-directory-name -> _round_backup_name
def _round_backup_name(image_name: str, timestamp: str) -> str:
    return f"{image_name} ({timestamp})"


# SOURCE_CELL: NB-LIVE-0001-C0000
# SOURCE_STATEMENT_MAP: backup-directory-reuse-create -> _resolve_round_backup_directory
def _resolve_round_backup_directory(
    output_root: Path,
    backup_root: Path | None,
    timestamp: str,
    *,
    force_new: bool = False,
) -> Path:
    marker = output_root / ".backup_dir"
    if not force_new and marker.exists():
        try:
            previous = Path(marker.read_text(encoding="utf-8").strip())
            if previous.exists() and previous.is_dir():
                return previous
        except OSError:
            pass
    parent = backup_root or output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / _round_backup_name(output_root.name, timestamp)
    serial = 1
    while candidate.exists():
        candidate = parent / f"{_round_backup_name(output_root.name, timestamp)}_{serial}"
        serial += 1
    candidate.mkdir(parents=True, exist_ok=False)
    output_root.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(candidate), encoding="utf-8")
    return candidate


def _round_backup_timestamp(value: str) -> str:
    digits = re.sub(r"[^0-9]", "", value)
    if len(digits) < 14:
        raise ContractError("injected round timestamp is incomplete")
    return (
        f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}_"
        f"{digits[8:10]}-{digits[10:12]}-{digits[12:14]}"
    )


# SOURCE_CELL: NB-LIVE-0001-C0000
# SOURCE_STATEMENT_MAP: top-level:000-055 -> bootstrap_round/configuration-and-backup
def bootstrap_round(
    request: RoundBootstrapRequest,
    services: HandlerServices,
) -> RoundContext:
    """Create explicit round state and preserve the reviewed backup policy."""

    output_root = request.output_root / request.image_name
    model_name, model_tag = _allocate_model_name(
        request.model_root, request.requested_model_name
    )
    backup_directory: Path | None = None
    if request.backup_existing:
        now = services.utc_now()
        backup_directory = _resolve_round_backup_directory(
            output_root,
            request.backup_root,
            _round_backup_timestamp(now),
            force_new=False,
        )
        copied = _backup_files(
            (
                output_root / "review_labels.csv",
                output_root / f"run_{request.mode}" / "detections.csv",
                output_root / "round_k_new_orig_ids.json",
            ),
            backup_directory,
        )
        (backup_directory / "backup_meta_round.json").write_text(
            json.dumps(
                {
                    "time_iso": now,
                    "img": output_root.name,
                    "out_root": str(output_root),
                    "mode": request.mode,
                    "xgb_threshold": request.decision_threshold,
                    "copied_files": [str(path) for path in copied],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    result = RoundContext(
        image_name=request.image_name,
        mode=request.mode,
        output_root=output_root,
        artifact_root=request.artifact_root,
        split_root=request.split_root,
        feature_output=request.feature_output,
        model_root=request.model_root,
        model_name=model_name,
        model_directory_tag=model_tag,
        decision_threshold=request.decision_threshold,
        fold_tag=request.fold_tag,
        positive_category=request.positive_category,
        pca_dimensions=request.pca_dimensions,
        force_cpu=request.force_cpu,
        backup_directory=backup_directory,
    )
    services.report(
        f"round ready: {result.image_name} mode={result.mode} "
        f"threshold={result.decision_threshold:.2f}"
    )
    return result


@dataclass(frozen=True)
class RoundNotification:
    title: str
    text: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    duration_hms: str = ""


def _format_elapsed_hms(elapsed_seconds: float) -> str:
    """Format the Cell 1 duration using its reviewed floor/modulo rules."""

    # SOURCE_CELL: NB-LIVE-0001-C0007
    # SOURCE_STATEMENT_MAP: S:C0007:duration-arithmetic -> elapsed-hms
    dt = max(0.0, float(elapsed_seconds))
    hh = int(dt // 3600)
    mm = int((dt % 3600) // 60)
    ss = int(dt % 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _format_delta_line(
    elapsed_seconds: float,
    started_at: str,
    finished_at: str,
) -> str:
    """Format the Cell 2 delta line without performing notification I/O."""

    # SOURCE_CELL: NB-LIVE-0001-C0010
    # SOURCE_STATEMENT_MAP: S:C0010:delta-arithmetic -> notification-delta-line
    seconds = int(round(float(elapsed_seconds)))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remainder = seconds % 60
    pretty = (
        f"{hours}h {minutes}m {remainder}s"
        if hours
        else (f"{minutes}m {remainder}s" if minutes else f"{remainder}s")
    )
    return (
        f"delta_since_cell1={pretty} "
        f"(cell1={started_at} -> cell2={finished_at})"
    )


# SOURCE_CELL: NB-LIVE-0001-C0010
# SOURCE_STATEMENT_MAP: notification-message-fields -> _format_notification_body
def _format_notification_body(
    message: RoundNotification,
    delta_line: str,
    user: str,
    host: str,
) -> str:
    return (
        f"{message.text}\nuser={user} host={host}\n"
        f"time={message.finished_at or ''}\n{delta_line}"
    )


# SOURCE_CELL: NB-LIVE-0001-C0007
# SOURCE_STATEMENT_MAP: top-level:000-019 -> build_round_notification/no-send
def build_round_notification(
    title: str = DEFAULT_NOTIFICATION_TITLE,
    text: str = TRAINING_COMPLETED_TEXT,
    *,
    started_at: str | None = None,
    finished_at: str | None = None,
    started_at_epoch_seconds: float | None = None,
    finished_at_epoch_seconds: float | None = None,
) -> RoundNotification:
    """Store completion content without resolving or contacting an endpoint."""

    if not title.strip() or not text.strip():
        raise ContractError("notification title and text must be nonempty")
    duration_seconds = None
    duration_hms = ""
    if started_at_epoch_seconds is not None and finished_at_epoch_seconds is not None:
        duration_seconds = max(
            0.0,
            float(finished_at_epoch_seconds) - float(started_at_epoch_seconds),
        )
        duration_hms = _format_elapsed_hms(duration_seconds)
    return RoundNotification(
        title,
        text,
        started_at,
        finished_at,
        duration_seconds,
        duration_hms,
    )


# SOURCE_CELL: NB-LIVE-0001-C0010
# SOURCE_STATEMENT_MAP: top-level:000-015 -> deliver_round_notification/injected-notifier
def deliver_round_notification(
    message: RoundNotification,
    config: NotificationConfig,
    services: HandlerServices,
) -> bool:
    """Deliver only when explicitly enabled and through an injected notifier."""

    if not config.enabled:
        return False
    if not config.topic or not config.endpoint:
        raise ContractError("enabled notifications require explicit topic and endpoint")
    delta_line = "delta_since_cell1=UNKNOWN (run cell1 first)"
    if (
        message.duration_seconds is not None
        and message.started_at
        and message.finished_at
    ):
        delta_line = _format_delta_line(
            message.duration_seconds,
            message.started_at,
            message.finished_at,
        )
    user, host = services.runtime_identity()
    delivery_message = RoundNotification(
        title=message.title,
        text=LAST_CELL_EXECUTED_TEXT,
        started_at=message.started_at,
        finished_at=message.finished_at,
        duration_seconds=message.duration_seconds,
        duration_hms=message.duration_hms,
    )
    body = _format_notification_body(delivery_message, delta_line, user, host)
    if services.notification_sender is not None:
        delivered = bool(
            services.notification_sender.send(
                NotificationDelivery(
                    topic=config.topic,
                    endpoint=config.endpoint,
                    title=message.title,
                    message=body,
                    priority=config.priority,
                    tags=config.tags,
                    timeout_seconds=config.timeout_seconds,
                )
            )
        )
    elif services.notifier is not None:
        delivered = bool(services.notifier(config.topic, config.endpoint))
    else:
        raise ContractError("enabled notifications require an injected notifier")
    if not delivered:
        raise ContractError("notification callback did not confirm delivery")
    services.report(f"notification delivered: {message.title}")
    return True


__all__ = [
    "RoundBootstrapRequest",
    "RoundBackupHelperBinding",
    "RoundContext",
    "RoundNotification",
    "_format_delta_line",
    "_format_elapsed_hms",
    "_format_notification_body",
    "_resolve_round_backup_directory",
    "_round_backup_name",
    "_round_backup_timestamp",
    "bootstrap_round",
    "bind_round_backup_helper",
    "build_round_notification",
    "deliver_round_notification",
]
