"""Non-mutating review-round path advancement."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from compag_curation.contracts import ContractError


IMAGE_DIRECTORY = re.compile(r"^IMG_(\d+)$")
REVIEW_MODES = frozenset({"gate", "xgb", "xgb_recall"})


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_real_directory(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ContractError(f"{label} directory is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ContractError(f"{label} must be a real directory")
    return path


def _require_regular_file(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ContractError(f"{label} file is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ContractError(f"{label} must be a regular non-symlink file")
    return path


@dataclass(frozen=True)
class ReviewAdvanceRequest:
    image_root: Path
    current_image: str
    mode: str
    detections: Path
    images: Path

    def __post_init__(self) -> None:
        if IMAGE_DIRECTORY.fullmatch(self.current_image) is None:
            raise ContractError("current review image must use IMG_<integer> form")
        if self.mode not in REVIEW_MODES:
            raise ContractError("review mode is invalid")


@dataclass(frozen=True)
class ReviewPaths:
    current_image: str
    next_image: str
    root: Path
    mode: str
    base_directory: Path
    run_directory: Path
    detections: Path
    tiles: Path


# SOURCE_CELL: NB-LIVE-0001-C0009
# SOURCE_STATEMENT_MAP: pick-next-img -> pick_next_image/typed-directory-selection
def pick_next_image(root: Path, current: str) -> str:
    match = IMAGE_DIRECTORY.fullmatch(current)
    if match is None:
        raise ContractError("current review image must use IMG_<integer> form")
    _require_real_directory(root, "review image root")
    current_number = int(match.group(1))
    candidates: list[tuple[int, str]] = []
    try:
        children = tuple(root.iterdir())
    except OSError as exc:
        raise ContractError("review image root cannot be enumerated") from exc
    for path in children:
        candidate = IMAGE_DIRECTORY.fullmatch(path.name)
        if candidate is None:
            continue
        try:
            info = path.lstat()
        except OSError as exc:
            raise ContractError("review image candidate cannot be inspected") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ContractError("review image root contains a symlink candidate")
        if not stat.S_ISDIR(info.st_mode):
            continue
        number = int(candidate.group(1))
        if number > current_number:
            candidates.append((number, path.name))
    return min(candidates)[1] if candidates else current


# SOURCE_CELL: NB-LIVE-0001-C0009
# SOURCE_STATEMENT_MAP: compute/patch-gui -> advance_review_paths/typed-state-conversion
def advance_review_paths(request: ReviewAdvanceRequest) -> ReviewPaths:
    """Resolve the next reviewed image without reading or rewriting GUI source."""

    root = _absolute(request.image_root)
    _require_real_directory(root, "review image root")
    next_image = pick_next_image(root, request.current_image)
    base = root / next_image
    _require_real_directory(base, "next review image")
    run = base / f"run_{request.mode}"
    _require_real_directory(run, "review run")
    expected_detections = run / "detections.csv"
    expected_tiles = base / "tiles"
    _require_regular_file(expected_detections, "review detections")
    _require_real_directory(expected_tiles, "review tiles")
    if _absolute(request.detections) != expected_detections:
        raise ContractError("configured detections do not match the typed next-image path")
    if _absolute(request.images) != expected_tiles:
        raise ContractError("configured images do not match the typed next-image path")
    return ReviewPaths(
        current_image=request.current_image,
        next_image=next_image,
        root=root,
        mode=request.mode,
        base_directory=base,
        run_directory=run,
        detections=expected_detections,
        tiles=expected_tiles,
    )


__all__ = [
    "ReviewAdvanceRequest",
    "ReviewPaths",
    "advance_review_paths",
    "pick_next_image",
]
