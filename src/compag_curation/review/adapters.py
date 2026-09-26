"""Typed adapters for the two reviewed interactive interfaces."""

from __future__ import annotations

import csv
import io
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from compag_curation.contracts import (
    ContractError,
    FileIdentity,
    atomic_write_output,
)


ACTIVE_LEARNING_REVIEW_COLUMNS = (
    "image",
    "id",
    "human_label",
    "action",
    "click_x",
    "click_y",
    "prob",
    "xgb_prob",
    "final_pred",
    "xgb_pred",
    "kept",
    "yolo_conf",
    "yolo_iou",
    "det_policy",
    "det_missing",
    "thr_xgb",
    "thr_yolo",
    "thr_iou",
    "det_thr",
    "hybrid_yolo_bias",
    "use_xgb",
    "use_yolo",
    "timestamp",
)


@dataclass(frozen=True)
class ReviewTableSnapshot:
    identity: FileIdentity
    payload: bytes


@dataclass(frozen=True)
class ReviewTableStore:
    """Descriptor-bound compare-and-swap storage for the interactive CSV."""

    columns: tuple[str, ...] = ACTIVE_LEARNING_REVIEW_COLUMNS

    def __post_init__(self) -> None:
        if self.columns != ACTIVE_LEARNING_REVIEW_COLUMNS:
            raise ContractError("active-learning review columns must use the sealed schema")

    def _validate_path(self, path: Path) -> Path:
        path = Path(path)
        if not path.is_absolute():
            raise ContractError("review table path must be absolute")
        try:
            parent = path.parent.resolve(strict=True)
        except OSError as exc:
            raise ContractError("review table parent is unavailable") from exc
        if parent != path.parent or path.parent.is_symlink() or not path.parent.is_dir():
            raise ContractError("review table parent must be a real directory")
        return path

    def _validate_payload(self, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise ContractError("review table payload must be bytes")
        try:
            text = payload.decode("utf-8")
            rows = csv.reader(io.StringIO(text, newline=""))
            header = tuple(next(rows))
            if header != self.columns:
                raise ContractError("review table header does not match the sealed schema")
            if any(len(row) != len(self.columns) for row in rows):
                raise ContractError("review table row width does not match the sealed schema")
        except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
            raise ContractError("review table payload is not canonical UTF-8 CSV") from exc

    def read(self, path: Path, expected: FileIdentity) -> ReviewTableSnapshot:
        path = self._validate_path(path)
        if not isinstance(expected, FileIdentity):
            raise ContractError("review table read requires a typed file identity")
        try:
            before = path.lstat()
        except OSError as exc:
            raise ContractError("review table is unavailable") from exc
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or FileIdentity.from_stat(before) != expected
        ):
            raise ContractError("review table identity changed before read")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ContractError("review table cannot be opened safely") from exc
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or FileIdentity.from_stat(opened) != expected:
                raise ContractError("review table identity changed while opening")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            completed = os.fstat(descriptor)
            if FileIdentity.from_stat(completed) != expected:
                raise ContractError("review table identity changed during read")
        finally:
            os.close(descriptor)
        try:
            after = path.lstat()
        except OSError as exc:
            raise ContractError("review table changed after read") from exc
        if FileIdentity.from_stat(after) != expected:
            raise ContractError("review table identity changed after read")
        payload = b"".join(chunks)
        self._validate_payload(payload)
        return ReviewTableSnapshot(expected, payload)

    def create(self, path: Path) -> ReviewTableSnapshot:
        path = self._validate_path(path)
        payload = (",".join(self.columns) + "\n").encode("utf-8")
        identity = atomic_write_output(path, lambda handle: handle.write(payload))
        snapshot = self.read(path, identity)
        if snapshot.payload != payload:
            raise ContractError("claimed review table content changed")
        return snapshot

    def replace(
        self,
        path: Path,
        expected: FileIdentity,
        payload: bytes,
    ) -> ReviewTableSnapshot:
        path = self._validate_path(path)
        self._validate_payload(payload)
        self.read(path, expected)
        backup_path: Path | None = None
        for index in range(256):
            candidate = path.with_name(f".{path.name}.previous.{index:03d}")
            try:
                candidate.lstat()
            except FileNotFoundError:
                backup_path = candidate
                break
            except OSError as exc:
                raise ContractError("review table replacement backup cannot be inspected") from exc
        if backup_path is None:
            raise ContractError("review table has no available replacement sibling")
        identity = atomic_write_output(
            path,
            lambda handle: handle.write(payload),
            backup_path=backup_path,
            expected_source_identity=expected.as_tuple(),
            discard_backup=True,
        )
        snapshot = self.read(path, identity)
        if snapshot.payload != payload:
            raise ContractError("replaced review table content changed")
        return snapshot


@dataclass(frozen=True)
class ActiveLearningReviewConfig:
    shortlist: Path
    detections: Path
    images: Path
    output: Path
    output_identity: FileIdentity
    table_store: ReviewTableStore

    def __post_init__(self) -> None:
        if not isinstance(self.output_identity, FileIdentity):
            raise ContractError("active-learning output identity must be typed")
        if not isinstance(self.table_store, ReviewTableStore):
            raise ContractError("active-learning review requires a typed table store")


@dataclass(frozen=True)
class FullImageReviewConfig:
    detections: Path
    images: Path
    cache: Path
    output: Path
    seen: Path
    host: str = "127.0.0.1"
    port: int = 8050

    def __post_init__(self) -> None:
        if self.host not in {"127.0.0.1", "localhost"}:
            raise ContractError("review host must be loopback")
        if isinstance(self.port, bool) or not 1 <= self.port <= 65535:
            raise ContractError("review port must be from 1 through 65535")


def launch_active_learning_review(config: ActiveLearningReviewConfig) -> FileIdentity:
    """Launch the reviewed OpenCV application with an immutable call contract."""

    from compag_curation.review import al_review_gui

    return al_review_gui.main(
        shortlist_path=config.shortlist,
        full_detections_path=config.detections,
        images_root=config.images,
        output_csv=config.output,
        output_identity=config.output_identity,
        table_store=config.table_store,
    )


def launch_full_image_review(config: FullImageReviewConfig) -> None:
    """Launch the reviewed Dash application with an immutable call contract."""

    from compag_curation.review import full_image_review_gui

    return full_image_review_gui.main(
        detections_csv=config.detections,
        tiles_dir=config.images,
        cache_dir=config.cache,
        review_labels_csv=config.output,
        seen_json=config.seen,
        host=config.host,
        port=config.port,
    )


__all__ = [
    "ACTIVE_LEARNING_REVIEW_COLUMNS",
    "ActiveLearningReviewConfig",
    "FullImageReviewConfig",
    "ReviewTableSnapshot",
    "ReviewTableStore",
    "launch_active_learning_review",
    "launch_full_image_review",
]
