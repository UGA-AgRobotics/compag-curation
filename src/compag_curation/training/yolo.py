"""Local-only YOLO training entry point."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from compag_curation.contracts import (
    ContractError,
    DirectoryIdentity,
    HandlerServices,
    _hash_tree,
    create_output_directory,
    directory_identity,
    directory_tree_digest,
)


class YoloModel(Protocol):
    def train(self, **parameters: Any) -> Any: ...


YoloModelLoader = Callable[[Path], YoloModel]


@dataclass(frozen=True)
class YoloTrainingRequest:
    data_yaml: Path
    dataset_root: Path
    dataset_root_sha256: str
    initial_weights: Path
    initial_weights_sha256: str
    output_root: Path
    run_name: str | None = None
    device: str | int = "cpu"
    imgsz: int = 512
    epochs: int = 60
    batch: int = 16
    patience: int = 20
    lr0: float = 0.001
    weight_decay: float = 0.0005
    close_mosaic: int = 10
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    translate: float = 0.10
    scale: float = 0.40
    shear: float = 0.0
    perspective: float = 0.0
    seed: int = 0
    deterministic: bool = True

    def __post_init__(self) -> None:
        for role, digest in (
            ("dataset root", self.dataset_root_sha256),
            ("weights", self.initial_weights_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ContractError(f"YOLO {role} SHA-256 must be lowercase hexadecimal")
        if not (
            self.device == "cpu"
            or (
                isinstance(self.device, int)
                and not isinstance(self.device, bool)
                and self.device >= 0
            )
        ):
            raise ContractError("YOLO device must be an explicit CPU or GPU index")
        for name in ("imgsz", "epochs", "batch"):
            if getattr(self, name) < 1:
                raise ContractError(f"YOLO {name} must be positive")
        for name in ("patience", "close_mosaic"):
            if getattr(self, name) < 0:
                raise ContractError(f"YOLO {name} must be nonnegative")
        if self.lr0 <= 0.0 or self.weight_decay < 0.0:
            raise ContractError("YOLO optimizer defaults are invalid")
        if isinstance(self.seed, bool) or self.seed != 0:
            raise ContractError("sealed YOLO seed must remain 0")
        if self.deterministic is not True:
            raise ContractError("sealed YOLO deterministic policy must remain enabled")
        if self.run_name is not None and not self.run_name.strip():
            raise ContractError("YOLO run name must be nonempty")
        if self.run_name is not None and (
            Path(self.run_name).name != self.run_name
            or "/" in self.run_name
            or "\\" in self.run_name
        ):
            raise ContractError("YOLO run name must be one portable path component")


@dataclass(frozen=True)
class YoloTrainingResult:
    save_directory: Path
    initial_weights: Path
    output_identity: DirectoryIdentity


def load_local_yolo_model(weights: Path) -> YoloModel:
    """Import Ultralytics only after local-asset preflight."""

    if not weights.is_file() or weights.is_symlink():
        raise ContractError("explicit local YOLO weights are unavailable")
    from ultralytics import YOLO

    return YOLO(str(weights))


def _hash_regular_local_file(path: Path) -> str:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ContractError("explicit YOLO weights must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
        ):
            raise ContractError("YOLO weights changed during preflight")
        value = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            value.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = path.lstat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(final):
        raise ContractError("YOLO weights changed during preflight")
    return value.hexdigest()


# SOURCE_CELL: NB-LIVE-0008-C0002
# SOURCE_STATEMENT_MAP: weight-fallback-loop -> _load_first_local_yolo_model
def _load_first_local_yolo_model(
    weights: Path,
    expected_sha256: str,
    model_loader: YoloModelLoader,
) -> tuple[YoloModel, Path]:
    if _hash_regular_local_file(weights) != expected_sha256:
        raise ContractError("explicit YOLO weights SHA-256 mismatch")
    try:
        model = model_loader(weights)
    except Exception as exc:
        raise ContractError("explicit local YOLO weights could not be loaded") from exc
    return model, weights


def _validate_local_yolo_dataset(
    path: Path,
    configured_root: Path,
) -> tuple[Path, ...]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContractError("explicit YOLO data YAML cannot be read") from exc
    if len(raw) > 64 * 1024:
        raise ContractError("explicit YOLO data YAML exceeds the reviewed size limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ContractError("explicit YOLO data YAML must be UTF-8") from exc
    values: dict[str, str] = {}
    allowed = {"path", "train", "val", "test", "nc", "names"}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if raw_line[:1].isspace() or ":" not in line:
            raise ContractError("YOLO data YAML must use the reviewed flat mapping form")
        key, rendered = line.split(":", 1)
        key = key.strip()
        rendered = rendered.strip()
        if key not in allowed or key in values or not rendered:
            raise ContractError("YOLO data YAML fields differ from the reviewed local contract")
        if "://" in rendered or "$" in rendered or "!" in rendered:
            raise ContractError("YOLO data YAML contains a remote or dynamic value")
        values[key] = rendered
    if not {"path", "train", "val", "nc", "names"} <= set(values):
        raise ContractError("YOLO data YAML is missing a reviewed local field")
    dataset_root = Path(values["path"])
    if not dataset_root.is_absolute():
        dataset_root = path.parent / dataset_root
    root_locator_info = dataset_root.lstat()
    if stat.S_ISLNK(root_locator_info.st_mode):
        raise ContractError("YOLO dataset root locator must not be a symlink")
    dataset_root = dataset_root.resolve(strict=True)
    try:
        expected_root = configured_root.resolve(strict=True)
        yaml_path = path.resolve(strict=True)
    except OSError as exc:
        raise ContractError("configured YOLO dataset root is unavailable") from exc
    if configured_root.is_symlink() or dataset_root != expected_root:
        raise ContractError("YOLO YAML path does not match the configured dataset root")
    if yaml_path != dataset_root and dataset_root not in yaml_path.parents:
        raise ContractError("YOLO data YAML must reside inside the configured dataset root")
    root_info = dataset_root.lstat()
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ContractError("YOLO dataset root must be a real local directory")
    resolved_splits: list[Path] = []
    for key in ("train", "val", "test"):
        if key not in values:
            continue
        split = Path(values[key])
        if not split.is_absolute():
            split = dataset_root / split
        split_locator_info = split.lstat()
        if stat.S_ISLNK(split_locator_info.st_mode):
            raise ContractError("YOLO dataset split locator must not be a symlink")
        split = split.resolve(strict=True)
        info = split.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or (split != dataset_root and dataset_root not in split.parents)
        ):
            raise ContractError("YOLO dataset split must be a real directory below the dataset root")
        resolved_splits.append(split)
    return tuple(resolved_splits)


# SOURCE_CELL: NB-LIVE-0008-C0002
# SOURCE_STATEMENT_MAP: top-level:000-017 -> train_yolo_local/local-only-training
def train_yolo_local(
    request: YoloTrainingRequest,
    services: HandlerServices,
    *,
    model_loader: YoloModelLoader = load_local_yolo_model,
) -> YoloTrainingResult:
    """Train with sealed source defaults and an explicit local weight file."""

    if (
        not request.data_yaml.exists()
        or not request.data_yaml.is_file()
        or request.data_yaml.is_symlink()
    ):
        raise ContractError("explicit YOLO data YAML is unavailable")
    _validate_local_yolo_dataset(request.data_yaml, request.dataset_root)
    if _hash_tree(request.dataset_root) != request.dataset_root_sha256:
        raise ContractError("explicit YOLO dataset root SHA-256 mismatch")
    model, chosen_weights = _load_first_local_yolo_model(
        request.initial_weights,
        request.initial_weights_sha256,
        model_loader,
    )
    chosen_text = chosen_weights.name or ""
    run_name = request.run_name or (
        "yolo11n_full" if "11" in chosen_text else "yolov8n_full"
    )
    output_root = create_output_directory(request.output_root)
    output_identity = directory_identity(output_root)
    result = model.train(
        data=str(request.data_yaml),
        imgsz=request.imgsz,
        epochs=request.epochs,
        batch=request.batch,
        device=request.device,
        project=str(output_root),
        name=run_name,
        patience=request.patience,
        lr0=request.lr0,
        weight_decay=request.weight_decay,
        close_mosaic=request.close_mosaic,
        hsv_h=request.hsv_h,
        hsv_s=request.hsv_s,
        hsv_v=request.hsv_v,
        translate=request.translate,
        scale=request.scale,
        shear=request.shear,
        perspective=request.perspective,
        seed=request.seed,
        deterministic=request.deterministic,
    )
    save_directory = Path(result.save_dir)
    observed_save_directory = Path(os.path.abspath(os.fspath(save_directory)))
    if directory_identity(output_root) != output_identity:
        raise ContractError("YOLO output directory identity changed during training")
    try:
        output_info = output_root.lstat()
        save_info = observed_save_directory.lstat()
    except OSError as exc:
        raise ContractError("YOLO run did not return a materialized local directory") from exc
    if (
        stat.S_ISLNK(output_info.st_mode)
        or not stat.S_ISDIR(output_info.st_mode)
        or stat.S_ISLNK(save_info.st_mode)
        or not stat.S_ISDIR(save_info.st_mode)
        or observed_save_directory != output_root / run_name
        or observed_save_directory.resolve(strict=True).parent != output_root.resolve(strict=True)
    ):
        raise ContractError("YOLO run escaped the configured model output directory")
    output_digest = directory_tree_digest(output_root)
    try:
        services.report(f"YOLO training completed: {save_directory}")
    finally:
        if (
            directory_identity(output_root) != output_identity
            or directory_tree_digest(output_root) != output_digest
        ):
            raise ContractError("YOLO output directory identity changed after reporting")
    return YoloTrainingResult(save_directory, chosen_weights, output_identity)


__all__ = [
    "YoloModel",
    "YoloModelLoader",
    "YoloTrainingRequest",
    "YoloTrainingResult",
    "_load_first_local_yolo_model",
    "load_local_yolo_model",
    "train_yolo_local",
]
