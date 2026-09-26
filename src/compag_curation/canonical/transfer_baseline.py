"""Sealed import boundary for user-supplied legacy finalized features.

The legacy table is useful as a transfer baseline, but it is not raw canonical
feature evidence and it does not establish the historical fit-time split.  This
module therefore gives the imported rows a separate, deliberately narrow
schema.  It never copies the source table or serializes its local path.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import math
import os
import re
import shutil
import stat
import tempfile
import unicodedata
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from compag_curation.public_io import (
    PublicIOError,
    bounded_csv_field_limit,
    canonical_json_bytes,
    canonical_json_value,
    compact_json_sha256,
    fsync_directory,
    hash_file_snapshot,
    portable_basename,
    publish_directory_noreplace,
    require,
    stable_file,
    write_new_json,
)
from compag_curation.review.exchange import REVIEW_ACTION_WEIGHTS
from compag_curation.review.published_model import (
    PRESET_ID,
    PRESET_RESOURCE_MANIFEST_SHA256,
)

from .spec import (
    CANONICAL_FEATURE_CROP_SCALES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_GROUP_FOLDS,
    CANONICAL_RANDOM_STATE,
    CANONICAL_TEST_FRACTION,
)


TRANSFER_BASELINE_SCHEMA = "compag-curation-transfer-baseline/v1"
TRANSFER_BASELINE_SPLIT_SCHEMA = "compag-curation-transfer-baseline-split/v1"
TRANSFER_BASELINE_SOURCE_GROUPS_SCHEMA = (
    "compag-curation-transfer-baseline-source-groups/v1"
)
TRANSFER_BASELINE_PROPOSAL_IDENTITY = (
    "compag-curation-legacy-transfer-proposal/v1"
)
TRANSFER_BASELINE_SCIENTIFIC_ROLE = (
    "USER_SUPPLIED_LEGACY_FINALIZED_FEATURE_TRANSFER_BASELINE"
)
EXPECTED_R92_TRANSFER_SOURCE_SHA256 = (
    "1836d704d15f27728e3ba09d7cddf0aea9458049bc66700dc98f14e7111ec29d"
)
EXPECTED_R92_TRANSFER_SOURCE_SIZE_BYTES = 582_968_556
EXPECTED_R92_TRANSFER_FEATURES_SHA256 = (
    "7015cd65ecc01e9698bc5782cf094a30fd13b168b3cdd38c5f8e2464fcd8f8a6"
)
EXPECTED_R92_TRANSFER_SOURCE_GROUPS_SHA256 = (
    "63462855d6bd45a1d24278b5b3d46348665b7dd0f4f8e0932de58ed4c887f74c"
)
EXPECTED_R92_TRANSFER_SOURCE_ROWS = 363_563
EXPECTED_R92_TRANSFER_SOURCE_PROPOSALS = 90_892
EXPECTED_R92_TRANSFER_GROUPS = 84
EXPECTED_R92_TRANSFER_INCOMPLETE_PROPOSALS = 5
EXPECTED_R92_TRANSFER_IMPORTED_ROWS = 363_548
EXPECTED_R92_TRANSFER_IMPORTED_PROPOSALS = 90_887
EXPECTED_R92_TRANSFER_ACTION_COUNTS = {
    "accept": 4_265,
    "flip": 43,
    "sus_accept": 1,
    "sus_flip": 10,
    "unreviewed": 86_568,
}

FEATURES_BASENAME = "features.csv"
SPLIT_BASENAME = "split.json"
SOURCE_GROUPS_BASENAME = "source_groups.json"
MANIFEST_BASENAME = "manifest.json"

MAX_LEGACY_SOURCE_BYTES = 2 * 1024 * 1024 * 1024
MAX_LEGACY_ROWS = 2_000_000
MAX_OUTPUT_BYTES = 2 * 1024 * 1024 * 1024

LEGACY_TRANSFER_COLUMNS = (
    "file_name",
    "ann_id",
    "scale",
    "area_px",
    "area_norm",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "cx",
    "cy",
    "mean_L",
    "mean_a",
    "mean_b",
    "std_L",
    "std_a",
    "std_b",
    "median_L",
    "median_a",
    "median_b",
    "delta_a",
    "delta_b",
    "elongation",
    "eccentricity",
    "solidity",
    "aspect_ratio",
    "circularity",
    "extent",
    "perimeter",
    "bbox_diag_frac",
    "perim_over_sqrt_area",
    "scale_diag",
    "hu1",
    "hu2",
    "hu4",
    "components_count",
    "holes_ratio",
    "touching_border",
    "LBP_u5",
    "GLCM_contrast",
    "GLCM_homogeneity",
    "grad_mean",
    "grad_p90",
    "grid_r",
    "grid_c",
    "grid_r_norm",
    "grid_c_norm",
    "pred_iou",
    "stability",
    "embed_sim",
    "g_quality",
    "g_light",
    "g_color",
    "g_shape",
    "g_embed",
    "g_robust",
    "g_maha",
    "g_border",
    "delta_b_med",
    "delta_a_med",
    "median_b_in",
    "median_b_ring",
    "median_a_in",
    "median_a_ring",
    "histb_q1",
    "histb_q2",
    "histb_q3",
    "histb_q4",
    "histab_q11",
    "histab_q12",
    "histab_q21",
    "histab_q22",
    *(f"embed_pca_{index}" for index in range(32)),
    "reviewed",
    "review_tag",
    "class_id",
    "class_name",
    "label",
)

TRANSFER_BASELINE_COLUMNS = (
    "proposal_id",
    "proposal_sha256",
    "group_id",
    "scale",
    *CANONICAL_FEATURE_ORDER,
    "label",
    "reviewed",
    "review_action",
    "review_weight",
)

_FEATURE_INDICES = tuple(
    LEGACY_TRANSFER_COLUMNS.index(name) for name in CANONICAL_FEATURE_ORDER
)
_INDEX = {name: LEGACY_TRANSFER_COLUMNS.index(name) for name in LEGACY_TRANSFER_COLUMNS}
_SCALE_TEXT = {
    Decimal("0.67"): "0.67",
    Decimal("0.80"): "0.8",
    Decimal("1.00"): "1",
    Decimal("1.25"): "1.25",
}
_SCALE_VALUES = tuple(Decimal(str(value)) for value in CANONICAL_FEATURE_CROP_SCALES)
_SCALE_FLOATS = tuple(float(value) for value in CANONICAL_FEATURE_CROP_SCALES)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GROUP_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_ANNOTATION_ID = re.compile(r"\d+")
_TILE_PATTERNS = (
    re.compile(r"^(.*?)(?:_y\d+x\d+)$", re.IGNORECASE),
    re.compile(r"^(.*?)(?:[_-]tile[_-]?\d+)$", re.IGNORECASE),
    re.compile(r"^(.*?)(?:[_-][rc]\d+[_-][rc]\d+)$", re.IGNORECASE),
)
_SOURCE_GROUP_NORMALIZATION = (
    "BASENAME_STEM_TILE_SUFFIX_REMOVAL_NFKC_CASEFOLD_ASCII_SLUG_V1"
)
_SPLIT_METHOD = (
    "DETERMINISTIC_GROUP_PURE_CLASS_VALID_NEAREST_FEASIBLE_80_20_V1"
)

require(len(LEGACY_TRANSFER_COLUMNS) == 109, "legacy transfer schema width changed")
require(len(set(LEGACY_TRANSFER_COLUMNS)) == 109, "legacy transfer schema is duplicated")
require(len(CANONICAL_FEATURE_ORDER) == 93, "canonical feature width changed")
require(
    set(CANONICAL_FEATURE_ORDER).issubset(LEGACY_TRANSFER_COLUMNS),
    "legacy transfer schema omits a canonical feature",
)
require(
    tuple(_SCALE_FLOATS) == tuple(CANONICAL_FEATURE_CROP_SCALES),
    "canonical transfer scales changed",
)


@dataclass(frozen=True)
class TransferBaselineRow:
    """One finalized feature row from a verified transfer baseline."""

    proposal_id: str
    group_id: str
    scale: float
    features: tuple[float, ...]
    label: int
    reviewed: bool
    review_action: str
    review_weight: float


@dataclass(frozen=True)
class TransferBaselineFold:
    """One group-pure, class-valid training/validation fold."""

    index: int
    train_groups: frozenset[str]
    validation_groups: frozenset[str]


@dataclass(frozen=True)
class TransferBaseline:
    """Verified paths and identities for a sealed transfer baseline."""

    root: Path
    manifest: Mapping[str, object]
    features_path: Path
    split_path: Path
    source_groups_path: Path
    source_sha256: str
    source_size_bytes: int
    source_groups: frozenset[str]
    source_groups_sha256: str
    row_count: int
    proposal_count: int
    train_groups: frozenset[str]
    test_groups: frozenset[str]
    folds: tuple[TransferBaselineFold, ...]


@dataclass(frozen=True)
class _LegacyRow:
    scale: Decimal
    feature_text: tuple[str, ...]
    reviewed: bool
    action: str
    class_id: str
    class_name: str
    label: int


@dataclass
class _LegacyBlock:
    proposal_id: str
    group_id: str
    rows: list[_LegacyRow]


@dataclass(frozen=True)
class _ImportSummary:
    source_sha256: str
    source_size_bytes: int
    source_rows: int
    source_proposals: int
    source_incomplete_proposals: int
    source_groups: frozenset[str]
    reviewed_proposals: int
    unreviewed_proposals: int
    skipped_proposals: int
    non_skip_proposals: int
    excluded_incomplete_proposals: int
    excluded_incomplete_rows: int
    excluded_incomplete_reviewed_proposals: int
    excluded_incomplete_unreviewed_proposals: int
    output_rows: int
    output_proposals: int
    output_reviewed_proposals: int
    output_unreviewed_proposals: int
    group_counts: Mapping[str, tuple[int, int]]
    action_counts: Mapping[str, int]


@dataclass(frozen=True)
class _TableSummary:
    row_count: int
    proposal_count: int
    group_counts: Mapping[str, tuple[int, int]]
    action_counts: Mapping[str, int]


def _file_signature(info: os.stat_result) -> tuple[int, ...]:
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


@contextmanager
def _stream_csv(
    path: Path,
    role: str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    maximum_size: int,
) -> Iterator[tuple[csv.reader, Any, os.stat_result]]:
    """Yield a descriptor-pinned CSV reader while hashing the same byte stream."""

    absolute = path.absolute()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError(f"{role} cannot be resolved") from exc
    require(absolute == resolved, f"{role} path contains a symlink component")
    before = absolute.lstat()
    require(
        stat.S_ISREG(before.st_mode)
        and not absolute.is_symlink()
        and before.st_nlink == 1
        and 0 < before.st_size <= maximum_size,
        f"{role} must be a bounded single-link regular file",
    )
    if expected_size is not None:
        require(before.st_size == expected_size, f"{role} size changed")
    descriptor = os.open(absolute, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    total = 0
    resources = ExitStack()
    completed_normally = False
    try:
        resources.callback(os.close, descriptor)
        resources.enter_context(bounded_csv_field_limit(16 * 1024 * 1024))
        opened = os.fstat(descriptor)
        require(
            _file_signature(opened) == _file_signature(before),
            f"{role} changed while opening",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as binary:

            class _DigestReader(io.RawIOBase):
                def readable(self) -> bool:
                    return True

                def readinto(self, buffer: Any) -> int:
                    nonlocal total
                    chunk = binary.read(len(buffer))
                    if not chunk:
                        return 0
                    total += len(chunk)
                    require(total <= maximum_size, f"{role} grew beyond its size bound")
                    digest.update(chunk)
                    buffer[: len(chunk)] = chunk
                    return len(chunk)

            buffered = io.BufferedReader(_DigestReader(), buffer_size=1024 * 1024)
            with io.TextIOWrapper(
                buffered,
                encoding="utf-8-sig",
                newline="",
            ) as text:
                yield csv.reader(text), digest, opened
                completed_normally = True
        if completed_normally:
            after = os.fstat(descriptor)
            path_after = absolute.lstat()
            require(total == opened.st_size, f"{role} was not consumed to EOF")
            require(
                _file_signature(after)
                == _file_signature(opened)
                == _file_signature(path_after),
                f"{role} changed while reading",
            )
            if expected_sha256 is not None:
                require(
                    digest.hexdigest() == expected_sha256,
                    f"{role} SHA-256 changed",
                )
    except (csv.Error, UnicodeError) as exc:
        raise PublicIOError(f"{role} is not valid UTF-8 CSV") from exc
    finally:
        resources.close()


@contextmanager
def _new_csv_writer(path: Path, columns: Sequence[str]) -> Iterator[csv.writer]:
    absolute = path.absolute()
    require(
        absolute.parent.is_dir() and not absolute.parent.is_symlink(),
        "transfer CSV parent is unsafe",
    )
    require(
        not absolute.exists() and not absolute.is_symlink(),
        f"transfer CSV already exists: {absolute.name}",
    )
    fields = tuple(columns)
    require(fields and len(fields) == len(set(fields)), "transfer CSV schema is invalid")
    descriptor = os.open(
        absolute,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o644,
    )
    completed_normally = False
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="",
            closefd=False,
        ) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(fields)
            yield writer
            handle.flush()
            completed_normally = True
        if completed_normally:
            os.fchmod(descriptor, 0o644)
            os.fsync(descriptor)
            info = os.fstat(descriptor)
            require(
                stat.S_ISREG(info.st_mode)
                and info.st_nlink == 1
                and stat.S_IMODE(info.st_mode) == 0o644
                and 0 < info.st_size <= MAX_OUTPUT_BYTES,
                "transfer CSV output metadata or size is invalid",
            )
    finally:
        os.close(descriptor)
    if completed_normally:
        fsync_directory(absolute.parent)


def _safe_output(output: Path) -> Path:
    absolute = output.absolute()
    portable_basename(absolute.name, "transfer baseline output basename")
    try:
        parent = absolute.parent.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError("transfer baseline output parent cannot be resolved") from exc
    info = absolute.parent.lstat()
    require(
        parent == absolute.parent
        and stat.S_ISDIR(info.st_mode)
        and not absolute.parent.is_symlink()
        and info.st_uid == os.geteuid()
        and not absolute.exists()
        and not absolute.is_symlink(),
        "transfer baseline output is unsafe or already exists",
    )
    return absolute


def _safe_root(root: Path) -> Path:
    absolute = root.absolute()
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError("transfer baseline root cannot be resolved") from exc
    require(absolute == resolved, "transfer baseline root contains a symlink component")
    info = absolute.lstat()
    require(
        stat.S_ISDIR(info.st_mode) and not absolute.is_symlink(),
        "transfer baseline root is not a safe directory",
    )
    return absolute


def _finite_text(value: str, role: str) -> tuple[float, str]:
    rendered = value.strip()
    require(_DECIMAL.fullmatch(rendered) is not None, f"{role} is not decimal")
    try:
        number = float(rendered)
    except ValueError as exc:
        raise PublicIOError(f"{role} is not numeric") from exc
    require(math.isfinite(number), f"{role} is not finite")
    canonical = format(0.0 if number == 0.0 else number, ".17g")
    return number, canonical


def _canonical_scale(value: str, role: str) -> Decimal:
    rendered = value.strip()
    require(_DECIMAL.fullmatch(rendered) is not None, f"{role} is not decimal")
    try:
        scale = Decimal(rendered)
    except InvalidOperation as exc:
        raise PublicIOError(f"{role} is not numeric") from exc
    require(scale in _SCALE_TEXT, f"{role} is not a canonical crop scale")
    return scale


def _annotation_id(value: str, role: str) -> str:
    rendered = value.strip()
    require(_ANNOTATION_ID.fullmatch(rendered) is not None, f"{role} is invalid")
    number = int(rendered)
    require(number >= 0, f"{role} is negative")
    return str(number)


def _legacy_group_stem(file_name: str) -> str:
    portable_basename(file_name, "legacy file name")
    stem = Path(file_name).stem
    for pattern in _TILE_PATTERNS:
        match = pattern.fullmatch(stem)
        if match is not None:
            stem = match.group(1)
            break
    require(stem != "", "legacy file name has an empty source group")
    return stem


def _normalize_group(raw_group: str) -> tuple[str, str]:
    semantic = unicodedata.normalize("NFKC", raw_group).casefold()
    slug = re.sub(r"[^a-z0-9._-]+", "-", semantic)
    slug = re.sub(r"-+", "-", slug).strip("._-")
    require(slug != "", "legacy source group cannot be normalized")
    if len(slug) > 64:
        suffix = hashlib.sha256(semantic.encode("utf-8")).hexdigest()[:12]
        prefix = slug[:51].rstrip("._-")
        require(prefix != "", "legacy source group prefix is empty")
        slug = f"{prefix}-{suffix}"
    require(_GROUP_ID.fullmatch(slug) is not None, "normalized source group is invalid")
    return slug, semantic


def _proposal_id(file_name: str, annotation_id: str) -> str:
    return compact_json_sha256(
        {
            "schema": TRANSFER_BASELINE_PROPOSAL_IDENTITY,
            "legacy_file_name": file_name,
            "legacy_annotation_id": annotation_id,
        }
    )


def _legacy_row(row: Sequence[str], row_number: int) -> _LegacyRow:
    role = f"legacy row {row_number}"
    scale = _canonical_scale(row[_INDEX["scale"]], f"{role} scale")
    reviewed_text = row[_INDEX["reviewed"]].strip()
    require(reviewed_text in {"0", "1"}, f"{role} reviewed flag is invalid")
    reviewed = reviewed_text == "1"
    action = row[_INDEX["review_tag"]].strip().casefold()
    if reviewed:
        require(action in REVIEW_ACTION_WEIGHTS, f"{role} review action is invalid")
    else:
        require(action == "", f"{role} has an action without explicit review")
    label_text = row[_INDEX["label"]].strip()
    require(label_text in {"0", "1"}, f"{role} label is not binary")
    label = int(label_text)
    class_id = row[_INDEX["class_id"]].strip()
    class_name = row[_INDEX["class_name"]].strip().casefold()
    expected_class = {0: ("2", "non-cj"), 1: ("1", "cj")}[label]
    require(
        (class_id, class_name) == expected_class,
        f"{role} class fields disagree with its label",
    )
    return _LegacyRow(
        scale=scale,
        feature_text=tuple(row[index] for index in _FEATURE_INDICES),
        reviewed=reviewed,
        action=action,
        class_id=class_id,
        class_name=class_name,
        label=label,
    )


def _increment_group_count(
    mutable: dict[str, list[int]],
    group_id: str,
    label: int,
    amount: int,
) -> None:
    counts = mutable.setdefault(group_id, [0, 0])
    counts[label] += amount


def _import_source(source: Path, features_path: Path) -> _ImportSummary:
    source_rows = 0
    source_proposals = 0
    source_incomplete = 0
    reviewed_proposals = 0
    unreviewed_proposals = 0
    skipped_proposals = 0
    non_skip_proposals = 0
    excluded_incomplete = 0
    excluded_incomplete_rows = 0
    excluded_incomplete_reviewed = 0
    excluded_incomplete_unreviewed = 0
    output_rows = 0
    output_proposals = 0
    output_reviewed = 0
    output_unreviewed = 0
    source_groups: set[str] = set()
    group_semantics: dict[str, str] = {}
    seen_proposals: set[str] = set()
    group_counts: dict[str, list[int]] = {}
    action_counts: dict[str, int] = {
        "unreviewed": 0,
        **{
            action: 0
            for action in REVIEW_ACTION_WEIGHTS
            if action != "skip"
        },
    }
    current: _LegacyBlock | None = None

    def finish(block: _LegacyBlock, writer: csv.writer) -> None:
        nonlocal source_incomplete, reviewed_proposals, unreviewed_proposals
        nonlocal skipped_proposals, non_skip_proposals
        nonlocal excluded_incomplete, excluded_incomplete_rows
        nonlocal excluded_incomplete_reviewed, excluded_incomplete_unreviewed
        nonlocal output_rows, output_proposals, output_reviewed, output_unreviewed
        require(block.rows, "legacy proposal block is empty")
        signatures = {
            (
                row.reviewed,
                row.action,
                row.class_id,
                row.class_name,
                row.label,
            )
            for row in block.rows
        }
        require(
            len(signatures) == 1,
            "legacy proposal has inconsistent review, action, or label fields",
        )
        scale_map: dict[Decimal, _LegacyRow] = {}
        for item in block.rows:
            require(
                item.scale not in scale_map,
                "legacy proposal contains a duplicate crop scale",
            )
            scale_map[item.scale] = item
        complete = tuple(sorted(scale_map)) == tuple(sorted(_SCALE_VALUES))
        if not complete:
            source_incomplete += 1
        first = block.rows[0]
        if first.reviewed:
            reviewed_proposals += 1
        else:
            unreviewed_proposals += 1
        if first.reviewed and first.action == "skip":
            skipped_proposals += 1
            return
        non_skip_proposals += 1
        if not complete:
            excluded_incomplete += 1
            excluded_incomplete_rows += len(block.rows)
            if first.reviewed:
                excluded_incomplete_reviewed += 1
            else:
                excluded_incomplete_unreviewed += 1
            return
        weight = (
            float(REVIEW_ACTION_WEIGHTS[first.action])
            if first.reviewed
            else 1.0
        )
        require(weight > 0.0, "included legacy label has zero weight")
        for scale in _SCALE_VALUES:
            item = scale_map[scale]
            feature_values = tuple(
                _finite_text(value, f"legacy proposal feature {name}")[1]
                for name, value in zip(
                    CANONICAL_FEATURE_ORDER,
                    item.feature_text,
                    strict=True,
                )
            )
            writer.writerow(
                (
                    block.proposal_id,
                    block.proposal_id,
                    block.group_id,
                    _SCALE_TEXT[scale],
                    *feature_values,
                    str(first.label),
                    "1" if first.reviewed else "0",
                    first.action,
                    format(weight, ".1f"),
                )
            )
            output_rows += 1
        output_proposals += 1
        if first.reviewed:
            output_reviewed += 1
            action_counts[first.action] += 1
        else:
            output_unreviewed += 1
            action_counts["unreviewed"] += 1
        _increment_group_count(group_counts, block.group_id, first.label, len(_SCALE_VALUES))

    with _new_csv_writer(features_path, TRANSFER_BASELINE_COLUMNS) as writer:
        with _stream_csv(
            source,
            "legacy transfer source",
            maximum_size=MAX_LEGACY_SOURCE_BYTES,
        ) as (reader, source_digest, source_info):
            try:
                header = tuple(next(reader))
            except StopIteration as exc:
                raise PublicIOError("legacy transfer source is empty") from exc
            require(
                header == LEGACY_TRANSFER_COLUMNS,
                "legacy transfer source must have the exact 109-column schema",
            )
            for raw in reader:
                source_rows += 1
                require(source_rows <= MAX_LEGACY_ROWS, "legacy transfer row bound exceeded")
                row_number = reader.line_num
                require(
                    len(raw) == len(LEGACY_TRANSFER_COLUMNS),
                    f"legacy row {row_number} has the wrong field count",
                )
                file_name = raw[_INDEX["file_name"]]
                raw_group = _legacy_group_stem(file_name)
                group_id, group_semantic = _normalize_group(raw_group)
                prior_semantic = group_semantics.setdefault(group_id, group_semantic)
                require(
                    prior_semantic == group_semantic,
                    "distinct legacy source groups collide after normalization",
                )
                source_groups.add(group_id)
                annotation_id = _annotation_id(
                    raw[_INDEX["ann_id"]],
                    f"legacy row {row_number} annotation identity",
                )
                identity = _proposal_id(file_name, annotation_id)
                if current is None or current.proposal_id != identity:
                    if current is not None:
                        finish(current, writer)
                    require(
                        identity not in seen_proposals,
                        "legacy proposal block is non-contiguous or duplicated",
                    )
                    seen_proposals.add(identity)
                    source_proposals += 1
                    current = _LegacyBlock(identity, group_id, [])
                require(
                    current.group_id == group_id,
                    "legacy proposal changes source group within its scale block",
                )
                current.rows.append(_legacy_row(raw, row_number))
            if current is not None:
                finish(current, writer)
        source_sha256 = source_digest.hexdigest()
        source_size_bytes = int(source_info.st_size)

    require(source_rows > 0 and source_proposals > 0, "legacy transfer source has no rows")
    require(output_proposals > 0 and output_rows > 0, "legacy transfer selection is empty")
    require(
        output_rows == output_proposals * len(CANONICAL_FEATURE_CROP_SCALES),
        "legacy transfer output does not close four scales per proposal",
    )
    require(
        non_skip_proposals == output_proposals + excluded_incomplete,
        "legacy transfer selection accounting does not close",
    )
    require(
        reviewed_proposals + unreviewed_proposals == source_proposals,
        "legacy review-state accounting does not close",
    )
    require(
        output_reviewed + output_unreviewed == output_proposals,
        "legacy output review-state accounting does not close",
    )
    return _ImportSummary(
        source_sha256=source_sha256,
        source_size_bytes=source_size_bytes,
        source_rows=source_rows,
        source_proposals=source_proposals,
        source_incomplete_proposals=source_incomplete,
        source_groups=frozenset(source_groups),
        reviewed_proposals=reviewed_proposals,
        unreviewed_proposals=unreviewed_proposals,
        skipped_proposals=skipped_proposals,
        non_skip_proposals=non_skip_proposals,
        excluded_incomplete_proposals=excluded_incomplete,
        excluded_incomplete_rows=excluded_incomplete_rows,
        excluded_incomplete_reviewed_proposals=excluded_incomplete_reviewed,
        excluded_incomplete_unreviewed_proposals=excluded_incomplete_unreviewed,
        output_rows=output_rows,
        output_proposals=output_proposals,
        output_reviewed_proposals=output_reviewed,
        output_unreviewed_proposals=output_unreviewed,
        group_counts={group: tuple(counts) for group, counts in group_counts.items()},
        action_counts={action: count for action, count in action_counts.items()},
    )


def _coverage(
    groups: Sequence[str] | frozenset[str] | set[str],
    group_counts: Mapping[str, tuple[int, int]],
) -> dict[str, int]:
    selected = tuple(groups)
    negative = sum(group_counts[group][0] for group in selected)
    positive = sum(group_counts[group][1] for group in selected)
    return {
        "groups": len(selected),
        "rows": negative + positive,
        "negative": negative,
        "positive": positive,
    }


def _hash_order(values: Sequence[str], domain: str, attempt: int) -> tuple[str, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: (
                hashlib.sha256(
                    f"{domain}\0{CANONICAL_RANDOM_STATE}\0{attempt}\0{value}".encode(
                        "ascii"
                    )
                ).digest(),
                value,
            ),
        )
    )


def _candidate_folds(
    train_groups: frozenset[str],
    group_counts: Mapping[str, tuple[int, int]],
) -> tuple[TransferBaselineFold, ...] | None:
    values = tuple(sorted(train_groups))
    if len(values) < CANONICAL_GROUP_FOLDS:
        return None
    best: tuple[tuple[object, ...], tuple[TransferBaselineFold, ...]] | None = None
    for attempt in range(4096):
        order = _hash_order(values, "transfer-fold-order-v1", attempt)
        validation: list[list[str]] = [[] for _ in range(CANONICAL_GROUP_FOLDS)]
        for index, group in enumerate(order):
            if index < CANONICAL_GROUP_FOLDS:
                fold_index = index
            else:
                fold_index = int.from_bytes(
                    hashlib.sha256(
                        f"transfer-fold-target-v1\0{CANONICAL_RANDOM_STATE}\0{attempt}\0{group}".encode(
                            "ascii"
                        )
                    ).digest()[:8],
                    "big",
                ) % CANONICAL_GROUP_FOLDS
            validation[fold_index].append(group)
        folds: list[TransferBaselineFold] = []
        valid = True
        for index, raw_validation in enumerate(validation, start=1):
            validation_groups = frozenset(raw_validation)
            fold_train = train_groups - validation_groups
            validation_coverage = _coverage(validation_groups, group_counts)
            train_coverage = _coverage(fold_train, group_counts)
            if (
                not validation_groups
                or not fold_train
                or validation_coverage["negative"] == 0
                or validation_coverage["positive"] == 0
                or train_coverage["negative"] == 0
                or train_coverage["positive"] == 0
            ):
                valid = False
                break
            folds.append(
                TransferBaselineFold(index, fold_train, validation_groups)
            )
        if not valid:
            continue
        row_sizes = [
            _coverage(fold.validation_groups, group_counts)["rows"] for fold in folds
        ]
        class_skew = sum(
            abs(
                _coverage(fold.validation_groups, group_counts)["positive"]
                * _coverage(train_groups, group_counts)["rows"]
                - _coverage(train_groups, group_counts)["positive"]
                * _coverage(fold.validation_groups, group_counts)["rows"]
            )
            for fold in folds
        )
        identity = tuple(tuple(sorted(fold.validation_groups)) for fold in folds)
        rank: tuple[object, ...] = (
            max(row_sizes) - min(row_sizes),
            class_skew,
            identity,
        )
        fixed = tuple(folds)
        if best is None or rank < best[0]:
            best = rank, fixed
    return None if best is None else best[1]


def _candidate_test_sets(
    groups: tuple[str, ...],
    group_counts: Mapping[str, tuple[int, int]],
    target_rows: int,
) -> tuple[frozenset[str], ...]:
    maximum_test_groups = len(groups) - CANONICAL_GROUP_FOLDS
    require(maximum_test_groups >= 1, "transfer split requires a held-out group")
    candidates: set[frozenset[str]] = set()
    if len(groups) <= 16:
        for size in range(1, maximum_test_groups + 1):
            candidates.update(
                frozenset(values) for values in itertools.combinations(groups, size)
            )
    else:
        candidates.update(frozenset({group}) for group in groups)
        for attempt in range(4096):
            order = _hash_order(groups, "transfer-test-order-v1", attempt)
            selected: set[str] = set()
            selected_rows = 0
            for group in order:
                if len(selected) >= maximum_test_groups:
                    break
                group_rows = sum(group_counts[group])
                candidate_rows = selected_rows + group_rows
                if (
                    not selected
                    or abs(candidate_rows - target_rows)
                    <= abs(selected_rows - target_rows)
                ):
                    selected.add(group)
                    selected_rows = candidate_rows
            if selected:
                candidates.add(frozenset(selected))
    return tuple(candidates)


def _build_split(
    group_counts: Mapping[str, tuple[int, int]],
    *,
    features_sha256: str,
    source_groups_sha256: str,
) -> dict[str, object]:
    all_groups = tuple(sorted(group_counts))
    require(
        len(all_groups) >= CANONICAL_GROUP_FOLDS + 1,
        "transfer baseline needs at least six effective groups",
    )
    total_coverage = _coverage(all_groups, group_counts)
    require(
        total_coverage["negative"] > 0 and total_coverage["positive"] > 0,
        "transfer baseline lacks one class",
    )
    target_rows = int(round(total_coverage["rows"] * CANONICAL_TEST_FRACTION))
    ranked: list[tuple[tuple[object, ...], frozenset[str]]] = []
    for test_groups in _candidate_test_sets(all_groups, group_counts, target_rows):
        train_groups = frozenset(all_groups) - test_groups
        train_coverage = _coverage(train_groups, group_counts)
        test_coverage = _coverage(test_groups, group_counts)
        if (
            train_coverage["negative"] == 0
            or train_coverage["positive"] == 0
            or test_coverage["negative"] == 0
            or test_coverage["positive"] == 0
            or sum(group_counts[group][0] > 0 for group in train_groups)
            < CANONICAL_GROUP_FOLDS
            or sum(group_counts[group][1] > 0 for group in train_groups)
            < CANONICAL_GROUP_FOLDS
        ):
            continue
        class_skew = abs(
            test_coverage["positive"] * total_coverage["rows"]
            - total_coverage["positive"] * test_coverage["rows"]
        )
        rank: tuple[object, ...] = (
            abs(test_coverage["rows"] - target_rows),
            class_skew,
            len(test_groups),
            tuple(sorted(test_groups)),
        )
        ranked.append((rank, test_groups))
    ranked.sort(key=lambda item: item[0])
    selected: tuple[frozenset[str], frozenset[str], tuple[TransferBaselineFold, ...]] | None = None
    for _rank, test_groups in ranked:
        train_groups = frozenset(all_groups) - test_groups
        folds = _candidate_folds(train_groups, group_counts)
        if folds is not None:
            selected = train_groups, test_groups, folds
            break
    require(
        selected is not None,
        "transfer baseline cannot form a group-pure 80:20 split with five class-valid folds",
    )
    train_groups, test_groups, folds = selected
    train_coverage = _coverage(train_groups, group_counts)
    test_coverage = _coverage(test_groups, group_counts)
    fold_rows: list[dict[str, object]] = []
    for fold in folds:
        fold_rows.append(
            {
                "fold": fold.index,
                "train_groups": sorted(fold.train_groups),
                "validation_groups": sorted(fold.validation_groups),
                "coverage": {
                    "train": _coverage(fold.train_groups, group_counts),
                    "validation": _coverage(
                        fold.validation_groups,
                        group_counts,
                    ),
                },
            }
        )
    return {
        "schema": TRANSFER_BASELINE_SPLIT_SCHEMA,
        "status": "PASS",
        "seed": CANONICAL_RANDOM_STATE,
        "method": _SPLIT_METHOD,
        "test_fraction": CANONICAL_TEST_FRACTION,
        "features_sha256": features_sha256,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        "source_groups_sha256": source_groups_sha256,
        "effective_groups_sha256": compact_json_sha256(list(all_groups)),
        "train_groups": sorted(train_groups),
        "test_groups": sorted(test_groups),
        "target_test_rows": target_rows,
        "observed_test_rows": test_coverage["rows"],
        "coverage": {
            "all": total_coverage,
            "train": train_coverage,
            "test": test_coverage,
        },
        "group_cv_folds": CANONICAL_GROUP_FOLDS,
        "cv_assignment_role": (
            "IMPORT_TIME_CLASS_VALID_FEASIBILITY_ONLY_NOT_TRAINING_ASSIGNMENTS"
        ),
        "training_cv_policy": (
            "DETERMINISTIC_SKLEARN_GROUP_KFOLD_REBUILT_FROM_"
            "EFFECTIVE_TRAIN_ROWS_EACH_ROUND"
        ),
        "cv_feasibility_folds": fold_rows,
    }


def _source_groups_record(summary: _ImportSummary) -> dict[str, object]:
    groups = sorted(summary.source_groups)
    return {
        "schema": TRANSFER_BASELINE_SOURCE_GROUPS_SCHEMA,
        "source_sha256": summary.source_sha256,
        "source_size_bytes": summary.source_size_bytes,
        "scope": "ALL_SOURCE_PROPOSALS_INCLUDING_UNREVIEWED_AND_SKIP",
        "normalization": _SOURCE_GROUP_NORMALIZATION,
        "group_count": len(groups),
        "groups": groups,
        "groups_sha256": compact_json_sha256(groups),
    }


def _file_record(path: Path) -> dict[str, object]:
    digest, info = hash_file_snapshot(path, require_single_link=True, max_bytes=MAX_OUTPUT_BYTES)
    require(stat.S_IMODE(info.st_mode) == 0o644, f"output mode changed: {path.name}")
    return {
        "path": path.name,
        "sha256": digest,
        "size_bytes": int(info.st_size),
        "mode_octal": "0644",
    }


def _manifest_record(
    summary: _ImportSummary,
    source_groups: Mapping[str, object],
    members: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    action_counts = {
        action: int(summary.action_counts[action])
        for action in sorted(summary.action_counts)
    }
    member_rows = [dict(row) for row in members]
    return {
        "schema": TRANSFER_BASELINE_SCHEMA,
        "status": "PASS",
        "scientific_role": TRANSFER_BASELINE_SCIENTIFIC_ROLE,
        "source": {
            "sha256": summary.source_sha256,
            "size_bytes": summary.source_size_bytes,
            "identity": (
                "RECOVERED_R92_FIT_SNAPSHOT_SHA256_AND_SIZE"
                if (
                    summary.source_sha256 == EXPECTED_R92_TRANSFER_SOURCE_SHA256
                    and summary.source_size_bytes
                    == EXPECTED_R92_TRANSFER_SOURCE_SIZE_BYTES
                )
                else "USER_SUPPLIED_LEGACY_109_COLUMN_TABLE"
            ),
            "column_count": len(LEGACY_TRANSFER_COLUMNS),
            "row_count": summary.source_rows,
            "proposal_count": summary.source_proposals,
            "group_count": len(summary.source_groups),
            "incomplete_scale_proposal_count": summary.source_incomplete_proposals,
        },
        "selection": {
            "policy": "ALL_LABELED_NON_SKIP_COMPLETE_CANONICAL_SCALE_BLOCKS",
            "reviewed_proposal_count": summary.reviewed_proposals,
            "unreviewed_proposal_count": summary.unreviewed_proposals,
            "skipped_proposal_count": summary.skipped_proposals,
            "non_skip_proposal_count": summary.non_skip_proposals,
            "imported_proposal_count": summary.output_proposals,
            "imported_row_count": summary.output_rows,
            "imported_group_count": len(summary.group_counts),
            "imported_reviewed_proposal_count": summary.output_reviewed_proposals,
            "imported_unreviewed_proposal_count": summary.output_unreviewed_proposals,
            "review_action_proposal_counts": action_counts,
            "exclusions": [
                {
                    "reason": "LABELED_NON_SKIP_PROPOSAL_LACKS_EXACT_FOUR_CANONICAL_SCALES",
                    "proposal_count": summary.excluded_incomplete_proposals,
                    "source_row_count": summary.excluded_incomplete_rows,
                    "reviewed_proposal_count": summary.excluded_incomplete_reviewed_proposals,
                    "unreviewed_proposal_count": summary.excluded_incomplete_unreviewed_proposals,
                }
            ],
        },
        "feature_contract": {
            "representation": "LEGACY_FINALIZED_BINARY64_DECIMAL_ROWS",
            "feature_count": len(CANONICAL_FEATURE_ORDER),
            "feature_order": list(CANONICAL_FEATURE_ORDER),
            "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
            "crop_scales": list(CANONICAL_FEATURE_CROP_SCALES),
            "rows_per_proposal": len(CANONICAL_FEATURE_CROP_SCALES),
            "missing_or_nonfinite_policy": "REJECT_IMPORTED_PROPOSAL",
            "sample_weight_policy": (
                "UNREVIEWED_1.0_ACCEPT_FLIP_1.0_SUS_ACCEPT_SUS_FLIP_0.4_SKIP_EXCLUDED"
            ),
        },
        "source_groups_sha256": source_groups["groups_sha256"],
        "published_model_binding": {
            "preset_id": PRESET_ID,
            "resource_manifest_sha256": PRESET_RESOURCE_MANIFEST_SHA256,
            "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
            "feature_state_source": "BUNDLED_PUBLISHED_MODEL_PRESET_NOT_LEGACY_CSV",
            "legacy_csv_feature_state_sha256": None,
            "legacy_csv_prototype_sha256": None,
            "legacy_csv_pca_sha256": None,
        },
        "claims": {
            "private_source_copied": False,
            "absolute_source_path_serialized": False,
            "historical_fit_time_split_established": False,
            "historical_decision_log_established": False,
            "fresh_canonical_feature_recomputation": False,
            "historical_fit_population_receipt_established": False,
            "recovered_r92_source_sha256_and_size_match": (
                summary.source_sha256 == EXPECTED_R92_TRANSFER_SOURCE_SHA256
                and summary.source_size_bytes
                == EXPECTED_R92_TRANSFER_SOURCE_SIZE_BYTES
            ),
            "paper_result_reproduction": "NOT_CLAIMED",
            "model_round_identity": "NOT_CLAIMED",
            "allowed_role": "SEALED_TRANSFER_BASELINE_ONLY",
        },
        "members": member_rows,
        "members_sha256": compact_json_sha256(member_rows),
    }


def import_legacy_transfer_baseline(
    source_csv: Path,
    output_dir: Path,
    *,
    seed: int = CANONICAL_RANDOM_STATE,
    expected_source_sha256: str | None = None,
    expected_source_size_bytes: int | None = None,
) -> TransferBaseline:
    """Stream a legacy 109-column CSV into a fresh sealed transfer baseline.

    The source file remains in place.  Publication uses a sibling staging
    directory and an atomic no-replace rename, so a partial or pre-existing
    destination is never treated as a completed baseline.
    """

    require(
        type(seed) is int and seed == CANONICAL_RANDOM_STATE,
        "transfer baseline split seed is sealed at 42",
    )
    require(
        (expected_source_sha256 is None and expected_source_size_bytes is None)
        or (
            isinstance(expected_source_sha256, str)
            and _SHA256.fullmatch(expected_source_sha256) is not None
            and type(expected_source_size_bytes) is int
            and 0 < expected_source_size_bytes <= MAX_LEGACY_SOURCE_BYTES
        ),
        "expected transfer source SHA-256 and size must be supplied together",
    )
    source = source_csv.absolute()
    try:
        source_resolved = source_csv.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError("legacy transfer source cannot be resolved") from exc
    require(source == source_resolved, "legacy transfer source contains a symlink component")
    output = _safe_output(output_dir)
    require(source != output, "legacy transfer source and output overlap")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.transfer-baseline-",
            dir=output.parent,
        )
    )
    try:
        summary = _import_source(source, staging / FEATURES_BASENAME)
        if expected_source_sha256 is not None:
            require(
                summary.source_sha256 == expected_source_sha256
                and summary.source_size_bytes == expected_source_size_bytes,
                "legacy transfer source does not match its required SHA-256 and size",
            )
        features_record = _file_record(staging / FEATURES_BASENAME)
        groups_record = _source_groups_record(summary)
        write_new_json(staging / SOURCE_GROUPS_BASENAME, groups_record)
        split = _build_split(
            summary.group_counts,
            features_sha256=str(features_record["sha256"]),
            source_groups_sha256=str(groups_record["groups_sha256"]),
        )
        write_new_json(staging / SPLIT_BASENAME, split)
        members = sorted(
            (
                features_record,
                _file_record(staging / SOURCE_GROUPS_BASENAME),
                _file_record(staging / SPLIT_BASENAME),
            ),
            key=lambda row: str(row["path"]),
        )
        manifest = _manifest_record(summary, groups_record, members)
        write_new_json(staging / MANIFEST_BASENAME, manifest)
        fsync_directory(staging)
        os.chmod(staging, 0o755, follow_symlinks=False)
        fsync_directory(staging)
        publish_directory_noreplace(staging, output)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return verify_transfer_baseline(output)


def import_r92_transfer_baseline(
    source_csv: Path,
    output_dir: Path,
) -> TransferBaseline:
    """Import only the recovered r92 fit snapshot pinned by SHA-256 and size."""

    return import_legacy_transfer_baseline(
        source_csv,
        output_dir,
        expected_source_sha256=EXPECTED_R92_TRANSFER_SOURCE_SHA256,
        expected_source_size_bytes=EXPECTED_R92_TRANSFER_SOURCE_SIZE_BYTES,
    )


def _strict_json(path: Path, role: str) -> tuple[Mapping[str, object], bytes]:
    payload, info = stable_file(path, max_bytes=64 * 1024 * 1024)
    require(
        info.st_nlink == 1 and stat.S_IMODE(info.st_mode) == 0o644,
        f"{role} metadata is unsafe",
    )
    value = canonical_json_value(payload, role)
    require(isinstance(value, dict), f"{role} is not an object")
    return value, payload


def _validate_members(
    root: Path,
    manifest: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    raw = manifest.get("members")
    require(isinstance(raw, list), "transfer member manifest is not a list")
    expected_names = {FEATURES_BASENAME, SOURCE_GROUPS_BASENAME, SPLIT_BASENAME}
    records: dict[str, Mapping[str, object]] = {}
    normalized: list[dict[str, object]] = []
    for item in raw:
        require(
            isinstance(item, dict)
            and set(item) == {"path", "sha256", "size_bytes", "mode_octal"}
            and isinstance(item.get("path"), str)
            and item.get("path") in expected_names
            and isinstance(item.get("sha256"), str)
            and _SHA256.fullmatch(str(item["sha256"])) is not None
            and type(item.get("size_bytes")) is int
            and 0 < int(item["size_bytes"]) <= MAX_OUTPUT_BYTES
            and item.get("mode_octal") == "0644"
            and item["path"] not in records,
            "transfer member record is invalid",
        )
        name = str(item["path"])
        path = root / name
        info = path.lstat()
        require(
            stat.S_ISREG(info.st_mode)
            and not path.is_symlink()
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) == 0o644
            and info.st_size == item["size_bytes"],
            f"transfer member metadata changed: {name}",
        )
        record = dict(item)
        records[name] = record
        normalized.append(record)
    normalized.sort(key=lambda row: str(row["path"]))
    require(
        set(records) == expected_names
        and raw == normalized
        and manifest.get("members_sha256") == compact_json_sha256(normalized),
        "transfer member closure changed",
    )
    return records


def _parse_output_row(raw: Sequence[str], row_number: int) -> TransferBaselineRow:
    require(
        len(raw) == len(TRANSFER_BASELINE_COLUMNS),
        f"transfer feature row {row_number} has the wrong field count",
    )
    proposal_id, proposal_sha256, group_id, scale_text = raw[:4]
    require(
        _SHA256.fullmatch(proposal_id) is not None
        and proposal_sha256 == proposal_id,
        f"transfer feature row {row_number} has an invalid proposal identity",
    )
    require(
        _GROUP_ID.fullmatch(group_id) is not None,
        f"transfer feature row {row_number} has an invalid group",
    )
    scale = _canonical_scale(scale_text, f"transfer feature row {row_number} scale")
    require(
        scale_text == _SCALE_TEXT[scale],
        f"transfer feature row {row_number} scale is not canonical text",
    )
    start = 4
    end = start + len(CANONICAL_FEATURE_ORDER)
    features: list[float] = []
    for name, value in zip(
        CANONICAL_FEATURE_ORDER,
        raw[start:end],
        strict=True,
    ):
        number, canonical = _finite_text(
            value,
            f"transfer feature row {row_number} {name}",
        )
        require(
            value == canonical,
            f"transfer feature row {row_number} {name} is not canonical text",
        )
        features.append(number)
    label_text, reviewed_text, action, weight_text = raw[end:]
    require(label_text in {"0", "1"}, "transfer feature label is not binary")
    require(reviewed_text in {"0", "1"}, "transfer feature reviewed flag is invalid")
    reviewed = reviewed_text == "1"
    if reviewed:
        require(
            action in REVIEW_ACTION_WEIGHTS and action != "skip",
            "transfer feature review action is ineffective",
        )
        expected_weight = format(float(REVIEW_ACTION_WEIGHTS[action]), ".1f")
    else:
        require(action == "", "unreviewed transfer feature has a review action")
        expected_weight = "1.0"
    require(weight_text == expected_weight, "transfer feature review action/weight mismatch")
    return TransferBaselineRow(
        proposal_id=proposal_id,
        group_id=group_id,
        scale=float(scale),
        features=tuple(features),
        label=int(label_text),
        reviewed=reviewed,
        review_action=action,
        review_weight=float(weight_text),
    )


def _iter_verified_rows(
    path: Path,
    record: Mapping[str, object],
) -> Iterator[TransferBaselineRow]:
    seen: set[str] = set()
    current_id: str | None = None
    block: list[TransferBaselineRow] = []

    def finish() -> tuple[TransferBaselineRow, ...]:
        require(block, "transfer feature proposal block is empty")
        first = block[0]
        require(
            tuple(row.scale for row in block) == _SCALE_FLOATS,
            "transfer feature proposal lacks the exact four ordered scales",
        )
        require(
            all(
                row.proposal_id == first.proposal_id
                and row.group_id == first.group_id
                and row.label == first.label
                and row.reviewed == first.reviewed
                and row.review_action == first.review_action
                and row.review_weight == first.review_weight
                for row in block
            ),
            "transfer feature proposal fields are inconsistent",
        )
        return tuple(block)

    with _stream_csv(
        path,
        "transfer feature table",
        expected_sha256=str(record["sha256"]),
        expected_size=int(record["size_bytes"]),
        maximum_size=MAX_OUTPUT_BYTES,
    ) as (reader, _digest, _info):
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise PublicIOError("transfer feature table is empty") from exc
        require(header == TRANSFER_BASELINE_COLUMNS, "transfer feature header changed")
        for raw in reader:
            row = _parse_output_row(raw, reader.line_num)
            if current_id is None or row.proposal_id != current_id:
                if current_id is not None:
                    yield from finish()
                require(
                    row.proposal_id not in seen,
                    "transfer feature proposal block is duplicated",
                )
                seen.add(row.proposal_id)
                current_id = row.proposal_id
                block = []
            block.append(row)
        if current_id is not None:
            yield from finish()


def _scan_table(path: Path, record: Mapping[str, object]) -> _TableSummary:
    rows = 0
    proposals: set[str] = set()
    group_counts: dict[str, list[int]] = {}
    action_counts: dict[str, int] = {
        "unreviewed": 0,
        **{
            action: 0
            for action in REVIEW_ACTION_WEIGHTS
            if action != "skip"
        },
    }
    for row in _iter_verified_rows(path, record):
        rows += 1
        require(rows <= MAX_LEGACY_ROWS, "transfer feature row bound exceeded")
        if row.proposal_id not in proposals:
            proposals.add(row.proposal_id)
            action_counts[row.review_action if row.reviewed else "unreviewed"] += 1
        _increment_group_count(group_counts, row.group_id, row.label, 1)
    require(rows > 0 and proposals, "transfer feature table has no rows")
    require(
        rows == len(proposals) * len(CANONICAL_FEATURE_CROP_SCALES),
        "transfer feature proposal/row closure changed",
    )
    return _TableSummary(
        row_count=rows,
        proposal_count=len(proposals),
        group_counts={group: tuple(counts) for group, counts in group_counts.items()},
        action_counts=action_counts,
    )


def _folds_from_split(split: Mapping[str, object]) -> tuple[TransferBaselineFold, ...]:
    rows = split.get("cv_feasibility_folds")
    require(isinstance(rows, list), "transfer split folds are invalid")
    folds: list[TransferBaselineFold] = []
    for expected_index, row in enumerate(rows, start=1):
        require(
            isinstance(row, dict)
            and row.get("fold") == expected_index
            and isinstance(row.get("train_groups"), list)
            and isinstance(row.get("validation_groups"), list),
            "transfer split fold is malformed",
        )
        folds.append(
            TransferBaselineFold(
                index=expected_index,
                train_groups=frozenset(str(value) for value in row["train_groups"]),
                validation_groups=frozenset(
                    str(value) for value in row["validation_groups"]
                ),
            )
        )
    require(len(folds) == CANONICAL_GROUP_FOLDS, "transfer split does not contain five folds")
    return tuple(folds)


def verify_transfer_baseline(root: Path) -> TransferBaseline:
    """Verify a sealed transfer-baseline directory and return its read API."""

    root = _safe_root(root)
    require(
        {path.name for path in root.iterdir()}
        == {MANIFEST_BASENAME, FEATURES_BASENAME, SPLIT_BASENAME, SOURCE_GROUPS_BASENAME},
        "transfer baseline directory closure changed",
    )
    manifest, _manifest_payload = _strict_json(
        root / MANIFEST_BASENAME,
        "transfer baseline manifest",
    )
    expected_manifest_fields = {
        "schema",
        "status",
        "scientific_role",
        "source",
        "selection",
        "feature_contract",
        "source_groups_sha256",
        "published_model_binding",
        "claims",
        "members",
        "members_sha256",
    }
    require(
        set(manifest) == expected_manifest_fields
        and manifest.get("schema") == TRANSFER_BASELINE_SCHEMA
        and manifest.get("status") == "PASS"
        and manifest.get("scientific_role") == TRANSFER_BASELINE_SCIENTIFIC_ROLE,
        "transfer baseline manifest identity changed",
    )
    members = _validate_members(root, manifest)
    groups_record, groups_payload = _strict_json(
        root / SOURCE_GROUPS_BASENAME,
        "transfer source-group ledger",
    )
    require(
        hashlib.sha256(groups_payload).hexdigest()
        == members[SOURCE_GROUPS_BASENAME]["sha256"],
        "transfer source-group ledger SHA-256 changed",
    )
    required_group_fields = {
        "schema",
        "source_sha256",
        "source_size_bytes",
        "scope",
        "normalization",
        "group_count",
        "groups",
        "groups_sha256",
    }
    groups = groups_record.get("groups")
    require(
        set(groups_record) == required_group_fields
        and groups_record.get("schema") == TRANSFER_BASELINE_SOURCE_GROUPS_SCHEMA
        and groups_record.get("scope")
        == "ALL_SOURCE_PROPOSALS_INCLUDING_UNREVIEWED_AND_SKIP"
        and groups_record.get("normalization") == _SOURCE_GROUP_NORMALIZATION
        and isinstance(groups, list)
        and groups == sorted(groups)
        and len(groups) == len(set(groups))
        and all(isinstance(group, str) and _GROUP_ID.fullmatch(group) for group in groups)
        and groups_record.get("group_count") == len(groups)
        and groups_record.get("groups_sha256") == compact_json_sha256(groups),
        "transfer source-group ledger contract changed",
    )
    source = manifest.get("source")
    require(
        isinstance(source, dict)
        and set(source)
        == {
            "sha256",
            "size_bytes",
            "identity",
            "column_count",
            "row_count",
            "proposal_count",
            "group_count",
            "incomplete_scale_proposal_count",
        }
        and isinstance(source.get("sha256"), str)
        and _SHA256.fullmatch(str(source["sha256"])) is not None
        and type(source.get("size_bytes")) is int
        and 0 < int(source["size_bytes"]) <= MAX_LEGACY_SOURCE_BYTES
        and source.get("identity")
        == (
            "RECOVERED_R92_FIT_SNAPSHOT_SHA256_AND_SIZE"
            if (
                source.get("sha256") == EXPECTED_R92_TRANSFER_SOURCE_SHA256
                and source.get("size_bytes")
                == EXPECTED_R92_TRANSFER_SOURCE_SIZE_BYTES
            )
            else "USER_SUPPLIED_LEGACY_109_COLUMN_TABLE"
        )
        and source.get("column_count") == len(LEGACY_TRANSFER_COLUMNS)
        and type(source.get("row_count")) is int
        and int(source["row_count"]) > 0
        and type(source.get("proposal_count")) is int
        and 0 < int(source["proposal_count"]) <= int(source["row_count"])
        and source.get("group_count") == len(groups)
        and type(source.get("incomplete_scale_proposal_count")) is int
        and int(source["incomplete_scale_proposal_count"]) >= 0
        and groups_record.get("source_sha256") == source["sha256"]
        and groups_record.get("source_size_bytes") == source["size_bytes"]
        and manifest.get("source_groups_sha256") == groups_record["groups_sha256"],
        "transfer source receipt is invalid",
    )
    feature_contract = manifest.get("feature_contract")
    require(
        feature_contract
        == {
            "representation": "LEGACY_FINALIZED_BINARY64_DECIMAL_ROWS",
            "feature_count": len(CANONICAL_FEATURE_ORDER),
            "feature_order": list(CANONICAL_FEATURE_ORDER),
            "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
            "crop_scales": list(CANONICAL_FEATURE_CROP_SCALES),
            "rows_per_proposal": len(CANONICAL_FEATURE_CROP_SCALES),
            "missing_or_nonfinite_policy": "REJECT_IMPORTED_PROPOSAL",
            "sample_weight_policy": (
                "UNREVIEWED_1.0_ACCEPT_FLIP_1.0_SUS_ACCEPT_SUS_FLIP_0.4_SKIP_EXCLUDED"
            ),
        },
        "transfer feature contract changed",
    )
    require(
        manifest.get("published_model_binding")
        == {
            "preset_id": PRESET_ID,
            "resource_manifest_sha256": PRESET_RESOURCE_MANIFEST_SHA256,
            "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
            "feature_state_source": "BUNDLED_PUBLISHED_MODEL_PRESET_NOT_LEGACY_CSV",
            "legacy_csv_feature_state_sha256": None,
            "legacy_csv_prototype_sha256": None,
            "legacy_csv_pca_sha256": None,
        },
        "transfer published-model binding changed",
    )
    require(
        manifest.get("claims")
        == {
            "private_source_copied": False,
            "absolute_source_path_serialized": False,
            "historical_fit_time_split_established": False,
            "historical_decision_log_established": False,
            "fresh_canonical_feature_recomputation": False,
            "historical_fit_population_receipt_established": False,
            "recovered_r92_source_sha256_and_size_match": (
                source["sha256"] == EXPECTED_R92_TRANSFER_SOURCE_SHA256
                and source["size_bytes"]
                == EXPECTED_R92_TRANSFER_SOURCE_SIZE_BYTES
            ),
            "paper_result_reproduction": "NOT_CLAIMED",
            "model_round_identity": "NOT_CLAIMED",
            "allowed_role": "SEALED_TRANSFER_BASELINE_ONLY",
        },
        "transfer scientific claims changed",
    )
    table = _scan_table(root / FEATURES_BASENAME, members[FEATURES_BASENAME])
    require(
        set(table.group_counts).issubset(groups),
        "transfer feature group is absent from the all-source ledger",
    )
    selection = manifest.get("selection")
    require(isinstance(selection, dict), "transfer selection receipt is invalid")
    exclusions = selection.get("exclusions")
    require(
        set(selection)
        == {
            "policy",
            "reviewed_proposal_count",
            "unreviewed_proposal_count",
            "skipped_proposal_count",
            "non_skip_proposal_count",
            "imported_proposal_count",
            "imported_row_count",
            "imported_group_count",
            "imported_reviewed_proposal_count",
            "imported_unreviewed_proposal_count",
            "review_action_proposal_counts",
            "exclusions",
        }
        and selection.get("policy")
        == "ALL_LABELED_NON_SKIP_COMPLETE_CANONICAL_SCALE_BLOCKS"
        and type(selection.get("reviewed_proposal_count")) is int
        and type(selection.get("unreviewed_proposal_count")) is int
        and type(selection.get("skipped_proposal_count")) is int
        and type(selection.get("non_skip_proposal_count")) is int
        and selection.get("imported_proposal_count") == table.proposal_count
        and selection.get("imported_row_count") == table.row_count
        and selection.get("imported_group_count") == len(table.group_counts)
        and selection.get("imported_reviewed_proposal_count")
        == sum(
            count
            for action, count in table.action_counts.items()
            if action != "unreviewed"
        )
        and selection.get("imported_unreviewed_proposal_count")
        == table.action_counts["unreviewed"]
        and selection.get("review_action_proposal_counts")
        == {action: table.action_counts[action] for action in sorted(table.action_counts)}
        and isinstance(exclusions, list)
        and len(exclusions) == 1
        and isinstance(exclusions[0], dict)
        and set(exclusions[0])
        == {
            "reason",
            "proposal_count",
            "source_row_count",
            "reviewed_proposal_count",
            "unreviewed_proposal_count",
        }
        and exclusions[0].get("reason")
        == "LABELED_NON_SKIP_PROPOSAL_LACKS_EXACT_FOUR_CANONICAL_SCALES"
        and type(exclusions[0].get("proposal_count")) is int
        and int(exclusions[0]["proposal_count"]) >= 0
        and type(exclusions[0].get("source_row_count")) is int
        and int(exclusions[0]["source_row_count"]) >= 0
        and type(exclusions[0].get("reviewed_proposal_count")) is int
        and int(exclusions[0]["reviewed_proposal_count"]) >= 0
        and type(exclusions[0].get("unreviewed_proposal_count")) is int
        and int(exclusions[0]["unreviewed_proposal_count"]) >= 0
        and exclusions[0]["proposal_count"]
        == exclusions[0]["reviewed_proposal_count"]
        + exclusions[0]["unreviewed_proposal_count"]
        and selection["non_skip_proposal_count"]
        == table.proposal_count + exclusions[0]["proposal_count"]
        and selection["reviewed_proposal_count"]
        + selection["unreviewed_proposal_count"]
        == source["proposal_count"]
        and selection["non_skip_proposal_count"]
        + selection["skipped_proposal_count"]
        == source["proposal_count"]
        and selection["imported_reviewed_proposal_count"]
        + exclusions[0]["reviewed_proposal_count"]
        + selection["skipped_proposal_count"]
        == selection["reviewed_proposal_count"]
        and selection["imported_unreviewed_proposal_count"]
        + exclusions[0]["unreviewed_proposal_count"]
        == selection["unreviewed_proposal_count"],
        "transfer selection accounting changed",
    )
    if selection["skipped_proposal_count"] == 0:
        require(
            source["row_count"]
            == table.row_count + exclusions[0]["source_row_count"]
            and source["incomplete_scale_proposal_count"]
            == exclusions[0]["proposal_count"],
            "transfer source rows do not close the imported and excluded rows",
        )
    recovered_r92 = (
        source["sha256"] == EXPECTED_R92_TRANSFER_SOURCE_SHA256
        and source["size_bytes"] == EXPECTED_R92_TRANSFER_SOURCE_SIZE_BYTES
    )
    if recovered_r92:
        require(
            source
            == {
                "sha256": EXPECTED_R92_TRANSFER_SOURCE_SHA256,
                "size_bytes": EXPECTED_R92_TRANSFER_SOURCE_SIZE_BYTES,
                "identity": "RECOVERED_R92_FIT_SNAPSHOT_SHA256_AND_SIZE",
                "column_count": len(LEGACY_TRANSFER_COLUMNS),
                "row_count": EXPECTED_R92_TRANSFER_SOURCE_ROWS,
                "proposal_count": EXPECTED_R92_TRANSFER_SOURCE_PROPOSALS,
                "group_count": EXPECTED_R92_TRANSFER_GROUPS,
                "incomplete_scale_proposal_count": (
                    EXPECTED_R92_TRANSFER_INCOMPLETE_PROPOSALS
                ),
            }
            and groups_record["groups_sha256"]
            == EXPECTED_R92_TRANSFER_SOURCE_GROUPS_SHA256
            and len(groups) == EXPECTED_R92_TRANSFER_GROUPS
            and set(table.group_counts) == set(groups)
            and members[FEATURES_BASENAME]["sha256"]
            == EXPECTED_R92_TRANSFER_FEATURES_SHA256
            and table.row_count == EXPECTED_R92_TRANSFER_IMPORTED_ROWS
            and table.proposal_count == EXPECTED_R92_TRANSFER_IMPORTED_PROPOSALS
            and table.action_counts == EXPECTED_R92_TRANSFER_ACTION_COUNTS
            and selection["reviewed_proposal_count"] == 4_322
            and selection["unreviewed_proposal_count"] == 86_570
            and selection["skipped_proposal_count"] == 0
            and selection["non_skip_proposal_count"]
            == EXPECTED_R92_TRANSFER_SOURCE_PROPOSALS
            and selection["imported_proposal_count"]
            == EXPECTED_R92_TRANSFER_IMPORTED_PROPOSALS
            and selection["imported_row_count"]
            == EXPECTED_R92_TRANSFER_IMPORTED_ROWS
            and selection["imported_group_count"] == EXPECTED_R92_TRANSFER_GROUPS
            and selection["imported_reviewed_proposal_count"] == 4_319
            and selection["imported_unreviewed_proposal_count"] == 86_568
            and selection["review_action_proposal_counts"]
            == EXPECTED_R92_TRANSFER_ACTION_COUNTS
            and exclusions
            == [
                {
                    "reason": (
                        "LABELED_NON_SKIP_PROPOSAL_LACKS_EXACT_FOUR_"
                        "CANONICAL_SCALES"
                    ),
                    "proposal_count": 5,
                    "source_row_count": 15,
                    "reviewed_proposal_count": 3,
                    "unreviewed_proposal_count": 2,
                }
            ],
            "recovered r92 transfer baseline differs from its audited matrix",
        )
    split, split_payload = _strict_json(
        root / SPLIT_BASENAME,
        "transfer split",
    )
    require(
        hashlib.sha256(split_payload).hexdigest() == members[SPLIT_BASENAME]["sha256"],
        "transfer split SHA-256 changed",
    )
    expected_split = _build_split(
        table.group_counts,
        features_sha256=str(members[FEATURES_BASENAME]["sha256"]),
        source_groups_sha256=str(groups_record["groups_sha256"]),
    )
    require(dict(split) == expected_split, "transfer split differs from deterministic policy")
    folds = _folds_from_split(split)
    train_groups = frozenset(str(value) for value in split["train_groups"])
    test_groups = frozenset(str(value) for value in split["test_groups"])
    return TransferBaseline(
        root=root,
        manifest=dict(manifest),
        features_path=root / FEATURES_BASENAME,
        split_path=root / SPLIT_BASENAME,
        source_groups_path=root / SOURCE_GROUPS_BASENAME,
        source_sha256=str(source["sha256"]),
        source_size_bytes=int(source["size_bytes"]),
        source_groups=frozenset(str(value) for value in groups),
        source_groups_sha256=str(groups_record["groups_sha256"]),
        row_count=table.row_count,
        proposal_count=table.proposal_count,
        train_groups=train_groups,
        test_groups=test_groups,
        folds=folds,
    )


def iter_transfer_baseline_rows(
    baseline: TransferBaseline | Path,
) -> Iterator[TransferBaselineRow]:
    """Yield rows from a freshly reverified sealed transfer baseline."""

    verified = verify_transfer_baseline(
        baseline.root if isinstance(baseline, TransferBaseline) else baseline
    )
    records = verified.manifest["members"]
    require(isinstance(records, list), "transfer member manifest is invalid")
    matches = [row for row in records if row.get("path") == FEATURES_BASENAME]
    require(len(matches) == 1, "transfer feature member is missing")
    yield from _iter_verified_rows(verified.features_path, matches[0])


__all__ = [
    "EXPECTED_R92_TRANSFER_SOURCE_SHA256",
    "EXPECTED_R92_TRANSFER_SOURCE_SIZE_BYTES",
    "LEGACY_TRANSFER_COLUMNS",
    "TRANSFER_BASELINE_COLUMNS",
    "TRANSFER_BASELINE_SCHEMA",
    "TRANSFER_BASELINE_SOURCE_GROUPS_SCHEMA",
    "TRANSFER_BASELINE_SPLIT_SCHEMA",
    "TransferBaseline",
    "TransferBaselineFold",
    "TransferBaselineRow",
    "import_legacy_transfer_baseline",
    "import_r92_transfer_baseline",
    "iter_transfer_baseline_rows",
    "verify_transfer_baseline",
]
