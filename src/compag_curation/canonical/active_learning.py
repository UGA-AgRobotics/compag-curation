"""Fail-closed active-learning rounds for the canonical XGBoost profile.

The cold-start export and model-driven rounds are deliberately separate.  A
model-driven round has two immutable outputs: a paused selection directory and
a completion directory.  The latter is written only after review validation,
retraining, bundle verification, and full-pool rescoring have all succeeded.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from compag_curation.public_io import (
    PublicIOError,
    canonical_json_value,
    compact_json_sha256,
    fsync_directory,
    portable_basename,
    stable_file,
    write_new_bytes,
    write_new_json,
)
from compag_curation.model_bundle import (
    POST_R92_TRANSFER_FROZEN_STATE_POLICY,
    POST_R92_TRANSFER_LINEAGE_POLICY,
    POST_R92_TRANSFER_PCA_VARIANCE_POLICY,
    POST_R92_TRANSFER_REPRODUCTION_CLAIM,
    POST_R92_TRANSFER_WORKFLOW,
)
from compag_curation.review.exchange import (
    CANONICAL_REVIEW_COLUMNS,
    REVIEW_ACTION_WEIGHTS,
    ReviewRow,
    validate_review_table,
    write_review_export,
)

from .features import (
    CANONICAL_RAW_FEATURE_ORDER,
    CanonicalFeatureState,
    CanonicalRawFeature,
    decode_canonical_embedding,
    encode_canonical_embedding,
    validate_canonical_feature_state,
)
from .serialization import serialize_canonical_feature_state
from .spec import (
    CANONICAL_DECISION_THRESHOLD,
    CANONICAL_FEATURE_CROP_SCALES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_GROUP_FOLDS,
    CANONICAL_GPU_PROFILE,
    CANONICAL_PROFILE,
    CANONICAL_RANDOM_STATE,
    GPU_EXECUTION_PROFILES,
    V2_PIPELINE_PROFILES,
)


CANONICAL_AL_MARGIN = 0.20
CANONICAL_AL_TOP_K = 50
CANONICAL_AL_POOL_COLUMNS = (
    "proposal_id",
    "proposal_sha256",
    "image_id",
    "image_sha256",
    "group_id",
    "scale",
    "xgb_p",
)
CANONICAL_AL_SHORTLIST_COLUMNS = (
    *CANONICAL_AL_POOL_COLUMNS,
    "distance_to_threshold",
    "uncertainty",
    "selection_rank",
)
CANONICAL_AL_DECISION_LOG_COLUMNS = (
    "sequence",
    "phase",
    "round_number",
    "decision_timestamp_utc",
    "timestamp_kind",
    "proposal_id",
    "proposal_sha256",
    "image_id",
    "image_sha256",
    "group_id",
    "label",
    "review_action",
    "review_weight",
    "decision_policy",
    "threshold",
    "margin",
    "top_k",
    "source_bundle_sha256",
    "source_predictions_sha256",
    "source_round_manifest_sha256",
    "reviewed_batch_sha256",
)
CANONICAL_AL_ACCUMULATED_FEATURE_COLUMNS = (
    "proposal_id",
    "proposal_sha256",
    "image_id",
    "image_sha256",
    "group_id",
    "source_sha256",
    "proposal_index",
    "scale",
    "predicted_iou",
    "stability_score",
    *CANONICAL_RAW_FEATURE_ORDER,
    "embedding_encoding",
    "embedding_dimensions",
    "embedding_f32le_base64",
)
CANONICAL_TRANSFER_SCORE_COLUMNS = CANONICAL_AL_POOL_COLUMNS
CANONICAL_TRANSFER_DECISION_LOG_COLUMNS = (
    "sequence",
    "phase",
    "round_number",
    "decision_timestamp_utc",
    "timestamp_kind",
    "proposal_id",
    "proposal_sha256",
    "image_id",
    "image_sha256",
    "group_id",
    "label",
    "review_action",
    "review_weight",
    "decision_policy",
    "threshold",
    "margin",
    "top_k",
    "source_scorer_kind",
    "source_scorer_sha256",
    "source_scores_sha256",
    "source_round_manifest_sha256",
    "reviewed_batch_sha256",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GROUP = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_UTC = re.compile(r"(?:19|20)[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z")
_PRIVATE_LOCATOR = re.compile(
    r"(?:/home/|/root(?:/|\b)|/Users/|/mnt/[a-z]/Users/|fi" + r"le://|[A-Za-z]:\\|\\\\|wsl\.localhost)",
    flags=re.IGNORECASE,
)
_MAX_TABLE_BYTES = 256 * 1024 * 1024
_MAX_TABLE_ROWS = 1_000_000
_POOL_SCHEMA = "compag-curation-canonical-active-learning-pool/v2"
_INITIAL_SCHEMA = "compag-curation-canonical-initial-labeling/v1"
_INITIAL_COMPLETION_SCHEMA = "compag-curation-canonical-initial-labeling-completion/v1"
_ROUND_SCHEMA = "compag-curation-canonical-active-learning-selection/v2"
_ROUND_COMPLETION_SCHEMA = "compag-curation-canonical-active-learning-completion/v2"
_IMAGE_ROUND_SCHEMA = "compag-curation-canonical-active-learning-selection/v3"
_IMAGE_ROUND_COMPLETION_SCHEMA = "compag-curation-canonical-active-learning-completion/v3"
_TRANSFER_IMAGE_ROUND_SCHEMA = (
    "compag-curation-canonical-transfer-active-learning-selection/v1"
)
_TRANSFER_IMAGE_ROUND_COMPLETION_SCHEMA = (
    "compag-curation-canonical-transfer-active-learning-completion/v1"
)
_ACCUMULATED_FEATURE_SCHEMA = (
    "compag-curation-canonical-active-learning-accumulated-raw-features/v1"
)
_MODEL_LINEAGE_SCHEMA = "compag-curation-canonical-active-learning-model-lineage/v1"
_IMAGE_WORKFLOW = "SINGLE_NEW_IMAGE_UNCERTAINTY_BATCH_FULL_RETRAIN_V1"
_TRANSFER_MODEL_LINEAGE_SCHEMA = (
    "compag-curation-canonical-transfer-active-learning-model-lineage/v1"
)
_TRANSFER_IMAGE_WORKFLOW = POST_R92_TRANSFER_WORKFLOW
_TRANSFER_CORPUS_ROLE = POST_R92_TRANSFER_LINEAGE_POLICY
_TRANSFER_REPRODUCTION_CLAIM = POST_R92_TRANSFER_REPRODUCTION_CLAIM
_TRANSFER_FEATURE_STATE_POLICY = POST_R92_TRANSFER_FROZEN_STATE_POLICY
_TRANSFER_TRAINING_MODE = (
    "FRESH_FULL_XGBOOST_FROM_TRANSFER_BASELINE_PLUS_CUMULATIVE_REVIEWED_ROWS"
)
TRANSFER_BASELINE_ACTION_ADAPTER_POLICY = (
    "UNREVIEWED_BLANK_ACTION_WEIGHT_1_ADAPTED_TO_ACCEPT_FOR_NUMERIC_WEIGHT_VALIDATION_ONLY"
)
_TRANSFER_R92_SCORER_KIND = "PUBLISHED_R92_TRANSFER_SCORER"
_TRANSFER_PROJECT_SCORER_KIND = "PROJECT_TRANSFER_BUNDLE"
_RETRAIN_REQUEST_SCHEMA_V1 = (
    "compag-curation-canonical-active-learning-retrain-request/v1"
)
_RETRAIN_REQUEST_SCHEMA_V2 = (
    "compag-curation-canonical-active-learning-retrain-request/v2"
)
_TRANSFER_RETRAIN_REQUEST_SCHEMA = (
    "compag-curation-canonical-transfer-active-learning-retrain-request/v1"
)
_SPLIT_SCHEMA = "compag-curation-canonical-group-split/v1"
_SPLIT_METHOD = "SOURCE_TILE_BALANCED_GROUP_PURE_NEAREST_80_20"
_INITIAL_PHASE = "INITIAL_LABELING_EXPORT_ALL"
_ROUND_PHASE = "ACTIVE_LEARNING_ROUND"


@dataclass(frozen=True)
class CanonicalActiveLearningPoolRow:
    proposal_id: str
    proposal_sha256: str
    image_id: str
    image_sha256: str
    group_id: str
    scale: float
    xgb_p: float | None


@dataclass(frozen=True)
class CanonicalActiveLearningDecision:
    sequence: int
    phase: str
    round_number: int
    decision_timestamp_utc: str
    proposal_id: str
    proposal_sha256: str
    image_id: str
    image_sha256: str
    group_id: str
    label: int
    review_action: str
    review_weight: float


@dataclass(frozen=True)
class CanonicalActiveLearningTrainingRow:
    proposal_id: str
    proposal_sha256: str
    image_id: str
    image_sha256: str
    group_id: str
    raw_feature: CanonicalRawFeature
    source_sha256: str


@dataclass(frozen=True)
class CanonicalActiveLearningRetrainRequest:
    round_number: int
    feature_state: CanonicalFeatureState
    effective_decisions: tuple[CanonicalActiveLearningDecision, ...]
    training_rows: tuple[CanonicalActiveLearningTrainingRow, ...]
    train_groups: frozenset[str]
    test_groups: frozenset[str]
    decision_log_sha256: str
    training_rows_sha256: str
    source_bundle_sha256: str
    frozen_split_sha256: str
    prototype_sha256: str
    pca_components_sha256: str
    pca_mean_sha256: str
    expected_bundle_provenance: Mapping[str, object]
    profile: str = CANONICAL_PROFILE
    device: str = "cpu"

    def __post_init__(self) -> None:
        profile, device = _canonical_profile_device(self.profile, self.device)
        carries_identity = (
            "profile" in self.expected_bundle_provenance
            or "device" in self.expected_bundle_provenance
        )
        _require(
            (
                profile not in GPU_EXECUTION_PROFILES
                and not carries_identity
            )
            or (
                self.expected_bundle_provenance.get("profile") == profile
                and self.expected_bundle_provenance.get("device") == device
            ),
            "active-learning retrain provenance profile/device identity changed",
        )


RetrainFunction = Callable[[CanonicalActiveLearningRetrainRequest, Path], None]


@dataclass(frozen=True)
class CanonicalTransferBaselineReference:
    """Path-free identity and split contract for one sealed external baseline."""

    archive_sha256: str
    source_sha256: str
    source_size_bytes: int
    split_sha256: str
    source_groups_sha256: str
    feature_order_sha256: str
    row_count: int
    proposal_count: int
    source_groups: frozenset[str]
    train_groups: frozenset[str]
    test_groups: frozenset[str]

    def __post_init__(self) -> None:
        for value, role in (
            (self.archive_sha256, "transfer baseline archive"),
            (self.source_sha256, "transfer baseline source"),
            (self.split_sha256, "transfer baseline split"),
            (self.source_groups_sha256, "transfer baseline source groups"),
            (self.feature_order_sha256, "transfer baseline feature order"),
        ):
            _sha256(value, role)
        _require(
            self.feature_order_sha256 == CANONICAL_FEATURE_ORDER_SHA256
            and isinstance(self.source_size_bytes, int)
            and not isinstance(self.source_size_bytes, bool)
            and self.source_size_bytes > 0
            and isinstance(self.row_count, int)
            and not isinstance(self.row_count, bool)
            and self.row_count > 0
            and isinstance(self.proposal_count, int)
            and not isinstance(self.proposal_count, bool)
            and self.proposal_count > 0
            and self.row_count
            == self.proposal_count * len(CANONICAL_FEATURE_CROP_SCALES),
            "transfer baseline row/proposal closure is invalid",
        )
        all_groups = frozenset(self.source_groups)
        train = frozenset(self.train_groups)
        test = frozenset(self.test_groups)
        _require(
            bool(all_groups)
            and bool(train)
            and bool(test)
            and not train & test
            and train | test <= all_groups
            and all(_GROUP.fullmatch(value) is not None for value in all_groups)
            and self.source_groups_sha256 == compact_json_sha256(sorted(all_groups)),
            "transfer baseline group identity or split is invalid",
        )

    def receipt(self) -> dict[str, object]:
        return {
            "archive_sha256": self.archive_sha256,
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "split_sha256": self.split_sha256,
            "source_groups_sha256": self.source_groups_sha256,
            "feature_order_sha256": self.feature_order_sha256,
            "row_count": self.row_count,
            "proposal_count": self.proposal_count,
            "train_groups_sha256": compact_json_sha256(sorted(self.train_groups)),
            "test_groups_sha256": compact_json_sha256(sorted(self.test_groups)),
        }


@dataclass(frozen=True)
class CanonicalTransferScorerReference:
    """Explicit first-r92 or later-project scorer identity."""

    kind: str
    scorer_sha256: str
    # ``scores_sha256`` binds the complete Stage-60 predictions emitted by the
    # scorer.  ``pool_scores_sha256`` separately binds the canonical AL pool
    # projection, whose CSV schema (and therefore bytes) is intentionally
    # different from predictions.csv.
    scores_sha256: str
    pool_scores_sha256: str
    feature_state_sha256: str
    prototype_sha256: str
    pca_components_sha256: str
    pca_mean_sha256: str
    r92_classifier_sha256: str
    r92_resource_manifest_sha256: str
    preset_id: str | None = None

    def __post_init__(self) -> None:
        _require(
            self.kind in {_TRANSFER_R92_SCORER_KIND, _TRANSFER_PROJECT_SCORER_KIND},
            "transfer scorer kind is invalid",
        )
        for value, role in (
            (self.scorer_sha256, "transfer scorer"),
            (self.scores_sha256, "transfer scores"),
            (self.pool_scores_sha256, "transfer pool scores"),
            (self.feature_state_sha256, "transfer feature state"),
            (self.prototype_sha256, "transfer prototype"),
            (self.pca_components_sha256, "transfer PCA components"),
            (self.pca_mean_sha256, "transfer PCA mean"),
            (self.r92_classifier_sha256, "published r92 classifier"),
            (
                self.r92_resource_manifest_sha256,
                "published r92 resource manifest",
            ),
        ):
            _sha256(value, role)
        if self.kind == _TRANSFER_R92_SCORER_KIND:
            _require(
                self.preset_id == "compag-cj-r92"
                and self.r92_classifier_sha256 != "",
                "round 1 requires the published compag-cj-r92 scorer",
            )
        else:
            _require(
                self.preset_id is None,
                "later transfer rounds cannot claim a published r92 preset",
            )

    def receipt(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "scorer_sha256": self.scorer_sha256,
            "scores_sha256": self.scores_sha256,
            "pool_scores_sha256": self.pool_scores_sha256,
            "feature_state_sha256": self.feature_state_sha256,
            "prototype_sha256": self.prototype_sha256,
            "pca_components_sha256": self.pca_components_sha256,
            "pca_mean_sha256": self.pca_mean_sha256,
            "r92_classifier_sha256": self.r92_classifier_sha256,
            "r92_resource_manifest_sha256": self.r92_resource_manifest_sha256,
            "preset_id": self.preset_id,
        }


@dataclass(frozen=True)
class CanonicalTransferRetrainRequest:
    """Verified input boundary for a fresh transfer-baseline classifier fit.

    ``baseline_root`` is an in-process locator only.  It must never be copied
    into public provenance; ``baseline`` is the complete path-free identity.
    """

    round_number: int
    baseline_root: Path
    baseline: CanonicalTransferBaselineReference
    feature_state: CanonicalFeatureState
    effective_decisions: tuple[CanonicalActiveLearningDecision, ...]
    cumulative_al_rows: tuple[CanonicalActiveLearningTrainingRow, ...]
    train_groups: frozenset[str]
    test_groups: frozenset[str]
    decision_log_sha256: str
    cumulative_al_rows_sha256: str
    source_scorer_sha256: str
    selection_manifest_sha256: str
    expected_bundle_provenance: Mapping[str, object]
    profile: str
    device: str

    def __post_init__(self) -> None:
        _canonical_profile_device(self.profile, self.device)
        _require(
            isinstance(self.round_number, int)
            and not isinstance(self.round_number, bool)
            and self.round_number >= 1,
            "transfer retrain round number is invalid",
        )
        for value, role in (
            (self.decision_log_sha256, "transfer decision log"),
            (self.cumulative_al_rows_sha256, "transfer cumulative AL rows"),
            (self.source_scorer_sha256, "transfer source scorer"),
            (self.selection_manifest_sha256, "transfer selection manifest"),
        ):
            _sha256(value, role)
        validate_canonical_feature_state(self.feature_state)


TransferRetrainFunction = Callable[[CanonicalTransferRetrainRequest, Path], None]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


def _canonical_device_for_profile(profile: object) -> str:
    """Return the sealed execution device for one v2 pipeline profile."""

    _require(
        isinstance(profile, str) and profile in V2_PIPELINE_PROFILES,
        "active-learning profile is not a supported canonical profile or "
        "separately supported noncanonical v2 pipeline profile",
    )
    return "cuda" if profile in GPU_EXECUTION_PROFILES else "cpu"


def _canonical_profile_device(profile: object, device: object) -> tuple[str, str]:
    resolved = str(profile) if isinstance(profile, str) else ""
    expected = _canonical_device_for_profile(resolved)
    _require(
        isinstance(device, str) and device == expected,
        f"active-learning profile {resolved!r} requires device {expected!r}",
    )
    return resolved, expected


def _sha256(value: object, role: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{role} is not a lowercase SHA-256",
    )
    return value


def _timestamp(value: object) -> str:
    _require(
        isinstance(value, str) and _UTC.fullmatch(value) is not None,
        "decision timestamp must be RFC3339 UTC to whole seconds",
    )
    # The regular expression constrains representation; this rejects invalid
    # calendar dates without accepting offset or fractional variants.
    from datetime import datetime

    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise PublicIOError("decision timestamp is not a valid UTC date") from exc
    return value


def _strict_int(value: object, role: str, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{role} is not an integer >= {minimum}",
    )
    return value


def _strict_float(value: object, role: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{role} is not numeric",
    )
    number = float(value)
    _require(math.isfinite(number), f"{role} is not finite")
    return number


def _format_float(value: float) -> str:
    number = _strict_float(value, "canonical active-learning number")
    return format(number, ".17g")


def _new_directory(path: Path) -> Path:
    path = path.absolute()
    portable_basename(path.name, "active-learning output basename")
    parent = path.parent
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError("active-learning output parent cannot be resolved") from exc
    parent_info = parent.lstat()
    _require(
        parent == resolved_parent
        and stat.S_ISDIR(parent_info.st_mode)
        and not parent.is_symlink()
        and parent_info.st_uid == os.geteuid(),
        "active-learning output parent is unsafe",
    )
    _require(not path.exists() and not path.is_symlink(), "active-learning output already exists")
    path.mkdir(mode=0o755)
    fsync_directory(path.parent)
    return path


def _safe_directory(path: Path, role: str) -> Path:
    absolute = path.absolute()
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError(f"{role} cannot be resolved") from exc
    info = absolute.lstat()
    _require(
        absolute == resolved
        and stat.S_ISDIR(info.st_mode)
        and not absolute.is_symlink(),
        f"{role} is unsafe or contains a symlink component",
    )
    return absolute


def _stable_payload(path: Path, role: str, *, max_bytes: int) -> tuple[bytes, int]:
    absolute = path.absolute()
    _require(
        absolute == absolute.resolve(strict=True),
        f"{role} path contains a symlink component",
    )
    payload, snapshot = stable_file(absolute, max_bytes=max_bytes)
    _require(
        stat.S_ISREG(snapshot.st_mode) and snapshot.st_nlink == 1,
        f"{role} must be a single-link regular file",
    )
    return payload, snapshot.st_size


def _input_record(path: Path, role: str, *, max_bytes: int) -> dict[str, object]:
    payload, size = _stable_payload(path, role, max_bytes=max_bytes)
    return {
        "role": role,
        "basename": portable_basename(path.name, f"{role} basename"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": size,
    }


def _read_json(path: Path, role: str, *, max_bytes: int = 16 * 1024 * 1024) -> tuple[Mapping[str, object], dict[str, object]]:
    payload, size = _stable_payload(path, role, max_bytes=max_bytes)
    value = canonical_json_value(payload, role)
    _require(isinstance(value, dict), f"{role} must be a JSON object")
    return value, {
        "role": role,
        "basename": portable_basename(path.name, f"{role} basename"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": size,
    }


def _retrain_request_manifest(
    path: Path,
) -> tuple[Mapping[str, object], dict[str, object]]:
    """Read v1 CPU legacy or v2 profile/device-bound retrain evidence."""

    value, record = _read_json(path, "active-learning retrain request")
    common_fields = {
        "schema",
        "profile",
        "round_number",
        "source_bundle_sha256",
        "frozen_split_sha256",
        "decision_log_sha256",
        "training_rows_sha256",
        "pool_feature_archive_sha256",
        "effective_proposals",
        "feature_rows",
        "train_groups_sha256",
        "test_groups_sha256",
        "feature_state_training_rows",
        "feature_state_positive_rows",
        "prototype_sha256",
        "pca_components_sha256",
        "pca_mean_sha256",
        "pca_policy",
        "prototype_policy",
    }
    schema = value.get("schema")
    if schema == _RETRAIN_REQUEST_SCHEMA_V1:
        _require(
            set(value) == common_fields
            and value.get("profile") == CANONICAL_PROFILE,
            "legacy active-learning retrain request must be the exact CPU v1 contract",
        )
        _canonical_profile_device(value.get("profile"), "cpu")
    else:
        _require(
            schema == _RETRAIN_REQUEST_SCHEMA_V2
            and set(value) == common_fields | {"device"},
            "active-learning retrain request fields or schema changed",
        )
        _canonical_profile_device(value.get("profile"), value.get("device"))
    for name in (
        "source_bundle_sha256",
        "frozen_split_sha256",
        "decision_log_sha256",
        "training_rows_sha256",
        "pool_feature_archive_sha256",
        "train_groups_sha256",
        "test_groups_sha256",
        "prototype_sha256",
        "pca_components_sha256",
        "pca_mean_sha256",
    ):
        _sha256(value.get(name), f"active-learning retrain request {name}")
    _strict_int(
        value.get("round_number"),
        "active-learning retrain request round",
        minimum=1,
    )
    for name in (
        "effective_proposals",
        "feature_rows",
        "feature_state_training_rows",
        "feature_state_positive_rows",
    ):
        _strict_int(
            value.get(name),
            f"active-learning retrain request {name}",
            minimum=1,
        )
    _require(
        value["feature_state_positive_rows"] <= value["feature_state_training_rows"]
        and value["pca_policy"] == "FROZEN_FROM_PARENT_BUNDLE"
        and value["prototype_policy"]
        == "RECOMPUTED_ACCUMULATED_OUTER_TRAIN_POSITIVES",
        "active-learning retrain request feature-state policy changed",
    )
    return value, record


def _read_csv(path: Path, columns: tuple[str, ...], role: str, *, allow_empty: bool = False) -> tuple[list[dict[str, str]], dict[str, object]]:
    payload, size = _stable_payload(path, role, max_bytes=_MAX_TABLE_BYTES)
    _require(payload.endswith(b"\n"), f"{role} must end in LF")
    try:
        text = payload.decode("ascii")
    except UnicodeError as exc:
        raise PublicIOError(f"{role} must be ASCII") from exc
    try:
        with io.StringIO(text, newline="") as handle:
            reader = csv.DictReader(handle)
            _require(tuple(reader.fieldnames or ()) == columns, f"{role} header mismatch")
            rows: list[dict[str, str]] = []
            for row in reader:
                _require(len(rows) < _MAX_TABLE_ROWS, f"{role} exceeds its row bound")
                _require(
                    None not in row and tuple(row) == columns,
                    f"{role} row has the wrong field count",
                )
                rows.append(row)
    except csv.Error as exc:
        raise PublicIOError(f"{role} contains invalid CSV framing") from exc
    _require(allow_empty or bool(rows), f"{role} is empty")
    return rows, {
        "role": role,
        "basename": portable_basename(path.name, f"{role} basename"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": size,
    }


def _csv_bytes(columns: tuple[str, ...], rows: Sequence[Mapping[str, object]]) -> bytes:
    try:
        with io.StringIO(newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=columns,
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row in rows:
                _require(tuple(row) == columns, "active-learning output row schema changed")
                writer.writerow(row)
            payload = handle.getvalue().encode("ascii")
    except (UnicodeError, csv.Error, ValueError) as exc:
        raise PublicIOError("active-learning CSV cannot be encoded canonically") from exc
    _require(len(payload) <= _MAX_TABLE_BYTES, "active-learning output exceeds its size bound")
    return payload


def _public_evidence(value: object) -> None:
    from compag_curation.public_io import canonical_json_bytes

    encoded = canonical_json_bytes(value).decode("ascii")
    _require(
        _PRIVATE_LOCATOR.search(encoded) is None,
        "active-learning public evidence contains a private locator",
    )


def _directory_closure(
    root: Path,
    files: set[str],
    directories: set[str] | None = None,
) -> None:
    expected_directories = set() if directories is None else directories
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for child in root.iterdir():
        info = child.lstat()
        if stat.S_ISREG(info.st_mode) and not child.is_symlink() and info.st_nlink == 1:
            observed_files.add(child.name)
        elif stat.S_ISDIR(info.st_mode) and not child.is_symlink():
            observed_directories.add(child.name)
        else:
            raise PublicIOError("active-learning output contains an unsafe member")
    _require(
        observed_files == files and observed_directories == expected_directories,
        "active-learning directory closure changed",
    )


def _pool_identity(row: Mapping[str, str], index: int) -> tuple[str, str, str, str, str]:
    proposal = row["proposal_id"]
    _require(_SHA256.fullmatch(proposal) is not None, f"pool row {index} proposal ID is invalid")
    _require(row["proposal_sha256"] == proposal, f"pool row {index} proposal hash differs from ID")
    _require(_SHA256.fullmatch(row["image_id"]) is not None, f"pool row {index} image ID is invalid")
    _require(_SHA256.fullmatch(row["image_sha256"]) is not None, f"pool row {index} image hash is invalid")
    _require(_GROUP.fullmatch(row["group_id"]) is not None, f"pool row {index} group is invalid")
    return (
        proposal,
        row["proposal_sha256"],
        row["image_id"],
        row["image_sha256"],
        row["group_id"],
    )


def _parse_pool_rows(rows: Sequence[Mapping[str, str]], *, require_scores: bool) -> tuple[CanonicalActiveLearningPoolRow, ...]:
    output: list[CanonicalActiveLearningPoolRow] = []
    seen: set[str] = set()
    prior = ""
    for index, row in enumerate(rows, start=1):
        identity = _pool_identity(row, index)
        proposal = identity[0]
        _require(proposal not in seen, "active-learning pool contains duplicate proposal IDs")
        _require(prior < proposal, "active-learning pool rows are not sorted by proposal ID")
        prior = proposal
        seen.add(proposal)
        try:
            scale = float(row["scale"])
        except ValueError as exc:
            raise PublicIOError("active-learning pool scale is nonnumeric") from exc
        _require(
            math.isfinite(scale)
            and scale == 1.0
            and row["scale"] == _format_float(scale),
            "active-learning pool must contain canonical scale-1 rows only",
        )
        probability: float | None
        if row["xgb_p"] == "":
            _require(not require_scores, "model-driven active learning requires xgb_p")
            probability = None
        else:
            try:
                probability = float(row["xgb_p"])
            except ValueError as exc:
                raise PublicIOError("active-learning pool xgb_p is nonnumeric") from exc
            _require(
                math.isfinite(probability)
                and 0.0 <= probability <= 1.0
                and row["xgb_p"] == _format_float(probability),
                "active-learning pool xgb_p is outside [0,1]",
            )
        output.append(CanonicalActiveLearningPoolRow(*identity, scale, probability))
    return tuple(output)


def _pool_row(row: CanonicalActiveLearningPoolRow, *, scored: bool) -> dict[str, object]:
    _require(row.proposal_sha256 == row.proposal_id, "active-learning pool proposal identity changed")
    values = {
        "proposal_id": row.proposal_id,
        "proposal_sha256": row.proposal_sha256,
        "image_id": row.image_id,
        "image_sha256": row.image_sha256,
        "group_id": row.group_id,
        "scale": _format_float(row.scale),
        "xgb_p": _format_float(row.xgb_p) if row.xgb_p is not None else "",
    }
    _parse_pool_rows([values], require_scores=scored)
    return values


def write_canonical_active_learning_pool(
    rows: Sequence[CanonicalActiveLearningPoolRow],
    output: Path,
    *,
    bundle_sha256: str | None,
    source_sha256: str | None = None,
    feature_archive_sha256: str | None = None,
    profile: str = CANONICAL_PROFILE,
) -> dict[str, object]:
    """Write an exact scale-1 pool and its physical/logical source receipts."""

    resolved_profile = str(profile)
    _canonical_device_for_profile(resolved_profile)
    _require(bool(rows), "cannot write an empty active-learning pool")
    scored = bundle_sha256 is not None
    bundle_hash = _sha256(bundle_sha256, "active-learning pool bundle", allow_none=True)
    source_hash = _sha256(
        source_sha256,
        "active-learning pool source",
        allow_none=True,
    )
    feature_archive_hash = _sha256(
        feature_archive_sha256,
        "active-learning pool feature archive",
        allow_none=True,
    )
    _require(
        not scored or (source_hash is not None and feature_archive_hash is not None),
        "a scored active-learning pool requires physical and logical feature-archive hashes",
    )
    _require(len(rows) <= _MAX_TABLE_ROWS, "active-learning pool exceeds its row bound")
    ordered = tuple(sorted(rows, key=lambda row: row.proposal_id))
    _require(
        scored == all(row.xgb_p is not None for row in ordered),
        "active-learning pool scoring state is inconsistent",
    )
    rendered_rows = [_pool_row(row, scored=scored) for row in ordered]
    parsed_rows = _parse_pool_rows(rendered_rows, require_scores=scored)
    _require(
        len(parsed_rows) == len(ordered),
        "active-learning pool normalization changed its row count",
    )
    payload = _csv_bytes(CANONICAL_AL_POOL_COLUMNS, rendered_rows)
    output = _new_directory(output)
    pool_path = output / "pool.csv"
    write_new_bytes(pool_path, payload)
    pool_hash = hashlib.sha256(payload).hexdigest()
    result = {
        "schema": _POOL_SCHEMA,
        "status": "PASS",
        "profile": resolved_profile,
        "scoring": "XGB_ONLY_FIXED_THRESHOLD" if scored else "UNSCORED_INITIAL_LABELING",
        "bundle_sha256": bundle_hash,
        "source_sha256": source_hash,
        "feature_archive_sha256": feature_archive_hash,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256 if scored else None,
        "threshold": CANONICAL_DECISION_THRESHOLD if scored else None,
        "scale": 1.0,
        "prediction_rows": len(ordered),
        "predictions_sha256": pool_hash,
    }
    _public_evidence(result)
    write_new_json(output / "pool_result.json", result)
    _directory_closure(output, {"pool.csv", "pool_result.json"})
    return result


def _read_pool(root: Path, *, require_scores: bool) -> tuple[tuple[CanonicalActiveLearningPoolRow, ...], Mapping[str, object], tuple[dict[str, object], ...]]:
    root = _safe_directory(root, "active-learning pool root")
    _directory_closure(root, {"pool.csv", "pool_result.json"})
    raw_rows, csv_record = _read_csv(root / "pool.csv", CANONICAL_AL_POOL_COLUMNS, "active-learning pool")
    rows = _parse_pool_rows(raw_rows, require_scores=require_scores)
    result, result_record = _read_json(root / "pool_result.json", "active-learning pool result")
    fields = {
        "schema", "status", "profile", "scoring", "bundle_sha256",
        "source_sha256", "feature_archive_sha256",
        "feature_order_sha256", "threshold", "scale", "prediction_rows",
        "predictions_sha256",
    }
    scored = result.get("scoring") == "XGB_ONLY_FIXED_THRESHOLD"
    _require(set(result) == fields, "active-learning pool result fields changed")
    _sha256(result.get("source_sha256"), "active-learning pool source", allow_none=True)
    _sha256(
        result.get("feature_archive_sha256"),
        "active-learning pool feature archive",
        allow_none=True,
    )
    _require(
        result.get("schema") == _POOL_SCHEMA
        and result.get("status") == "PASS"
        and result.get("profile") in V2_PIPELINE_PROFILES
        and isinstance(result.get("scale"), (int, float))
        and not isinstance(result.get("scale"), bool)
        and result.get("scale") == 1.0
        and isinstance(result.get("prediction_rows"), int)
        and not isinstance(result.get("prediction_rows"), bool)
        and result.get("prediction_rows") == len(rows)
        and result.get("predictions_sha256") == csv_record["sha256"],
        "active-learning pool result identity changed",
    )
    if scored:
        _sha256(result.get("bundle_sha256"), "active-learning pool bundle")
        _require(
            result.get("feature_order_sha256") == CANONICAL_FEATURE_ORDER_SHA256
            and isinstance(result.get("threshold"), (int, float))
            and not isinstance(result.get("threshold"), bool)
            and result.get("threshold") == CANONICAL_DECISION_THRESHOLD
            and all(row.xgb_p is not None for row in rows)
            and result.get("source_sha256") is not None
            and result.get("feature_archive_sha256") is not None,
            "active-learning scored-pool method identity changed",
        )
    else:
        _require(
            result.get("scoring") == "UNSCORED_INITIAL_LABELING"
            and result.get("bundle_sha256") is None
            and result.get("feature_order_sha256") is None
            and result.get("threshold") is None
            and (result.get("source_sha256") is None)
            == (result.get("feature_archive_sha256") is None)
            and all(row.xgb_p is None for row in rows),
            "initial-labeling pool scoring state changed",
        )
    _require(scored or not require_scores, "model-driven active learning requires a scored pool")
    return rows, result, (csv_record, result_record)


def _split_manifest(path: Path) -> tuple[frozenset[str], frozenset[str], Mapping[str, object], dict[str, object]]:
    value, record = _read_json(path, "frozen group split")
    fields = {
        "schema", "status", "profile", "seed", "method", "train_groups",
        "test_groups", "target_test_rows", "observed_test_rows",
        "group_cv_folds", "coverage", "effective_rows", "skipped_rows",
        "reviewed_sha256",
    }
    _require(set(value) == fields, "frozen split manifest fields changed")
    train = value.get("train_groups")
    test = value.get("test_groups")
    _require(
        value.get("schema") == _SPLIT_SCHEMA
        and value.get("status") == "PASS"
        and value.get("profile") in V2_PIPELINE_PROFILES
        and value.get("seed") == CANONICAL_RANDOM_STATE
        and value.get("method") == _SPLIT_METHOD
        and value.get("group_cv_folds") == CANONICAL_GROUP_FOLDS,
        "frozen split method identity changed",
    )
    _sha256(value.get("reviewed_sha256"), "frozen split reviewed input")
    _require(
        isinstance(train, list)
        and isinstance(test, list)
        and train == sorted(train)
        and test == sorted(test)
        and len(train) >= CANONICAL_GROUP_FOLDS
        and bool(test)
        and all(isinstance(group, str) and _GROUP.fullmatch(group) is not None for group in (*train, *test))
        and len(set(train)) == len(train)
        and len(set(test)) == len(test)
        and not set(train) & set(test),
        "frozen split group partitions are invalid",
    )
    coverage = value.get("coverage")
    _require(isinstance(coverage, dict) and set(coverage) == {"train", "test"}, "frozen split coverage fields changed")
    coverage_rows = 0
    for partition in ("train", "test"):
        counts = coverage[partition]
        _require(
            isinstance(counts, dict)
            and set(counts) == {"rows", "negative", "positive"}
            and all(isinstance(counts[name], int) and not isinstance(counts[name], bool) and counts[name] >= 0 for name in counts)
            and counts["rows"] == counts["negative"] + counts["positive"],
            f"frozen split {partition} coverage is invalid",
        )
        coverage_rows += counts["rows"]
    _require(
        isinstance(value.get("target_test_rows"), int)
        and not isinstance(value.get("target_test_rows"), bool)
        and value["target_test_rows"] > 0
        and isinstance(value.get("observed_test_rows"), int)
        and not isinstance(value.get("observed_test_rows"), bool)
        and value["observed_test_rows"] == coverage["test"]["rows"]
        and isinstance(value.get("effective_rows"), int)
        and not isinstance(value.get("effective_rows"), bool)
        and value["effective_rows"] == coverage_rows
        and isinstance(value.get("skipped_rows"), int)
        and not isinstance(value.get("skipped_rows"), bool)
        and value["skipped_rows"] >= 0,
        "frozen split row counts are inconsistent",
    )
    return frozenset(train), frozenset(test), value, record


def _decision_csv_row(
    decision: ReviewRow,
    *,
    sequence: int,
    phase: str,
    round_number: int,
    timestamp_utc: str,
    policy: str,
    threshold: str,
    margin: str,
    top_k: str,
    source_bundle_sha256: str,
    source_predictions_sha256: str,
    source_round_manifest_sha256: str,
    reviewed_batch_sha256: str,
) -> dict[str, object]:
    _require(
        decision.review_action in REVIEW_ACTION_WEIGHTS
        and decision.review_weight == REVIEW_ACTION_WEIGHTS[decision.review_action],
        "canonical active-learning decision action/weight changed",
    )
    return {
        "sequence": str(sequence),
        "phase": phase,
        "round_number": str(round_number),
        "decision_timestamp_utc": timestamp_utc,
        "timestamp_kind": "BATCH_IMPORT_UTC",
        "proposal_id": decision.proposal_id,
        "proposal_sha256": decision.proposal_sha256,
        "image_id": decision.image_id,
        "image_sha256": decision.image_sha256,
        "group_id": decision.group_id,
        "label": str(decision.label),
        "review_action": decision.review_action,
        "review_weight": format(float(decision.review_weight), ".1f"),
        "decision_policy": policy,
        "threshold": threshold,
        "margin": margin,
        "top_k": top_k,
        "source_bundle_sha256": source_bundle_sha256,
        "source_predictions_sha256": source_predictions_sha256,
        "source_round_manifest_sha256": source_round_manifest_sha256,
        "reviewed_batch_sha256": reviewed_batch_sha256,
    }


def _parse_decision_rows(rows: Sequence[Mapping[str, str]]) -> tuple[CanonicalActiveLearningDecision, ...]:
    output: list[CanonicalActiveLearningDecision] = []
    identities: dict[str, tuple[str, str, str, str, str]] = {}
    prior_timestamp = ""
    prior_round_number = 0
    round_receipts: dict[int, tuple[str, ...]] = {}
    for index, row in enumerate(rows, start=1):
        try:
            sequence = int(row["sequence"])
            round_number = int(row["round_number"])
            label = int(row["label"])
            weight = float(row["review_weight"])
        except ValueError as exc:
            raise PublicIOError("active-learning decision log contains a nonnumeric field") from exc
        _require(sequence == index, "active-learning decision sequence is not contiguous")
        _require(
            row["sequence"] == str(sequence)
            and row["round_number"] == str(round_number)
            and row["label"] == str(label),
            "active-learning decision numeric representation is not canonical",
        )
        phase = row["phase"]
        _require(
            (phase == _INITIAL_PHASE and round_number == 0)
            or (phase == _ROUND_PHASE and round_number >= 1),
            "active-learning decision phase/round is invalid",
        )
        _require(
            round_number in {prior_round_number, prior_round_number + 1},
            "active-learning decision rounds are not monotone and contiguous",
        )
        prior_round_number = round_number
        timestamp = _timestamp(row["decision_timestamp_utc"])
        _require(
            row["timestamp_kind"] == "BATCH_IMPORT_UTC",
            "active-learning decision timestamp kind changed",
        )
        _require(timestamp >= prior_timestamp, "active-learning decision timestamps are not monotone")
        prior_timestamp = timestamp
        identity = _pool_identity(row, index)
        proposal = identity[0]
        _require(
            proposal not in identities or identities[proposal] == identity,
            "active-learning decision identity changed between rounds",
        )
        identities[proposal] = identity
        action = row["review_action"]
        _require(label in {0, 1}, "active-learning decision label is not binary")
        _require(action in REVIEW_ACTION_WEIGHTS, "active-learning decision action is invalid")
        _require(
            math.isfinite(weight)
            and row["review_weight"] == format(REVIEW_ACTION_WEIGHTS[action], ".1f")
            and weight == REVIEW_ACTION_WEIGHTS[action],
            "active-learning decision action/weight is invalid",
        )
        _sha256(row["source_predictions_sha256"], "decision source predictions")
        _sha256(row["source_round_manifest_sha256"], "decision source manifest")
        _sha256(row["reviewed_batch_sha256"], "decision reviewed batch")
        if phase == _INITIAL_PHASE:
            _require(
                row["decision_policy"] == _INITIAL_PHASE
                and row["threshold"] == ""
                and row["margin"] == ""
                and row["top_k"] == ""
                and row["source_bundle_sha256"] == "",
                "initial-labeling decision policy changed",
            )
        else:
            _sha256(row["source_bundle_sha256"], "decision source bundle")
            _require(
                row["decision_policy"] == "XGB_ONLY_DISTANCE_TO_FIXED_THRESHOLD"
                and row["threshold"] == _format_float(CANONICAL_DECISION_THRESHOLD)
                and row["margin"] == _format_float(CANONICAL_AL_MARGIN)
                and row["top_k"] == str(CANONICAL_AL_TOP_K),
                "active-learning decision policy changed",
            )
        receipt = (
            phase,
            timestamp,
            row["timestamp_kind"],
            row["decision_policy"],
            row["threshold"],
            row["margin"],
            row["top_k"],
            row["source_bundle_sha256"],
            row["source_predictions_sha256"],
            row["source_round_manifest_sha256"],
            row["reviewed_batch_sha256"],
        )
        _require(
            round_number not in round_receipts
            or round_receipts[round_number] == receipt,
            "active-learning decision-policy receipt changed within a round",
        )
        round_receipts[round_number] = receipt
        output.append(
            CanonicalActiveLearningDecision(
                sequence=sequence,
                phase=phase,
                round_number=round_number,
                decision_timestamp_utc=timestamp,
                proposal_id=proposal,
                proposal_sha256=identity[1],
                image_id=identity[2],
                image_sha256=identity[3],
                group_id=identity[4],
                label=label,
                review_action=action,
                review_weight=weight,
            )
        )
    return tuple(output)


def _read_decision_log(path: Path) -> tuple[tuple[CanonicalActiveLearningDecision, ...], list[dict[str, str]], dict[str, object]]:
    rows, record = _read_csv(path, CANONICAL_AL_DECISION_LOG_COLUMNS, "active-learning decision log")
    return _parse_decision_rows(rows), rows, record


def _latest_decisions(decisions: Sequence[CanonicalActiveLearningDecision]) -> dict[str, CanonicalActiveLearningDecision]:
    latest: dict[str, CanonicalActiveLearningDecision] = {}
    for decision in decisions:
        latest[decision.proposal_id] = decision
    return latest


def _verify_bundle(root: Path) -> Any:
    root = _safe_directory(root, "active-learning bundle root")
    from compag_curation.model_bundle import BUNDLE_SCHEMA_V2, verify_model_bundle

    value = verify_model_bundle(root)
    profile = getattr(value, "profile", None)
    _canonical_device_for_profile(profile)
    _require(
        getattr(value, "schema", None) == BUNDLE_SCHEMA_V2
        and tuple(getattr(value, "feature_order", ())) == CANONICAL_FEATURE_ORDER
        and getattr(value, "feature_order_sha256", None) == CANONICAL_FEATURE_ORDER_SHA256
        and getattr(value, "threshold", None) == CANONICAL_DECISION_THRESHOLD,
        "active-learning bundle method identity changed",
    )
    _sha256(getattr(value, "bundle_sha256", None), "active-learning bundle")
    return value


def _review_rows(rows: Sequence[CanonicalActiveLearningPoolRow]) -> list[dict[str, str]]:
    return [
        {
            "proposal_id": row.proposal_id,
            "proposal_sha256": row.proposal_sha256,
            "image_id": row.image_id,
            "image_sha256": row.image_sha256,
            "group_id": row.group_id,
        }
        for row in rows
    ]


def _pending_request_record(
    path: Path,
    pool_rows: Sequence[CanonicalActiveLearningPoolRow],
    role: str,
) -> dict[str, object]:
    rows, record = _read_csv(path, CANONICAL_REVIEW_COLUMNS, role)
    expected = _review_rows(sorted(pool_rows, key=lambda row: row.proposal_id))
    _require(len(rows) == len(expected), f"{role} does not close its candidate set")
    for observed, identity in zip(rows, expected, strict=True):
        _require(
            all(observed[name] == identity[name] for name in identity)
            and observed["label"] == ""
            and observed["review_action"] == ""
            and observed["review_weight"] == ""
            and observed["review_status"] == "pending",
            f"{role} contains a changed pending-review row",
        )
    return record


def _initial_manifest(root: Path, pool_root: Path) -> tuple[Mapping[str, object], dict[str, object], tuple[CanonicalActiveLearningPoolRow, ...], Mapping[str, object]]:
    _directory_closure(root, {"review_request.csv", "initial_manifest.json"})
    manifest, record = _read_json(root / "initial_manifest.json", "initial-labeling manifest")
    fields = {
        "schema", "status", "profile", "kind", "round_number",
        "created_at_utc", "pool", "review_request", "candidate_rows",
        "review_contract", "paper_result_reproduction",
    }
    _require(
        set(manifest) == fields
        and manifest.get("schema") == _INITIAL_SCHEMA
        and manifest.get("status") == "PAUSED_FOR_REVIEW"
        and manifest.get("profile") in V2_PIPELINE_PROFILES
        and manifest.get("kind") == _INITIAL_PHASE
        and manifest.get("round_number") is None
        and manifest.get("review_contract") == "CANONICAL_ACTION_WEIGHTED_V2"
        and manifest.get("paper_result_reproduction") == "NOT_CLAIMED",
        "initial-labeling manifest identity changed",
    )
    _timestamp(manifest.get("created_at_utc"))
    pool_rows, pool_result, pool_records = _read_pool(pool_root, require_scores=False)
    _require(
        manifest.get("profile") == pool_result.get("profile"),
        "initial-labeling profile differs from its pool",
    )
    request_record = _pending_request_record(
        root / "review_request.csv",
        pool_rows,
        "initial review request",
    )
    _require(
        manifest.get("pool") == {"files": list(pool_records), "result": dict(pool_result)}
        and manifest.get("review_request") == request_record
        and manifest.get("candidate_rows") == len(pool_rows),
        "initial-labeling manifest input binding changed",
    )
    return manifest, record, pool_rows, pool_result


def write_initial_labeling_export_all(
    pool_root: Path,
    output: Path,
    *,
    created_at_utc: str,
) -> dict[str, object]:
    """Export every cold-start proposal; this is explicitly not an AL round."""

    timestamp = _timestamp(created_at_utc)
    pool_rows, pool_result, pool_records = _read_pool(pool_root, require_scores=False)
    _require(
        pool_result["scoring"] == "UNSCORED_INITIAL_LABELING",
        "cold-start export requires an unscored initial pool",
    )
    output = _new_directory(output)
    review_result = write_review_export(
        _review_rows(pool_rows),
        output / "review_request.csv",
        canonical_actions=True,
    )
    request_record = _pending_request_record(
        output / "review_request.csv",
        pool_rows,
        "initial review request",
    )
    manifest = {
        "schema": _INITIAL_SCHEMA,
        "status": "PAUSED_FOR_REVIEW",
        "profile": pool_result["profile"],
        "kind": _INITIAL_PHASE,
        "round_number": None,
        "created_at_utc": timestamp,
        "pool": {"files": list(pool_records), "result": dict(pool_result)},
        "review_request": request_record,
        "candidate_rows": len(pool_rows),
        "review_contract": review_result["review_contract"],
        "paper_result_reproduction": "NOT_CLAIMED",
    }
    _public_evidence(manifest)
    write_new_json(output / "initial_manifest.json", manifest)
    _directory_closure(output, {"review_request.csv", "initial_manifest.json"})
    return manifest


def complete_initial_labeling_export_all(
    initial_root: Path,
    pool_root: Path,
    reviewed_path: Path,
    output: Path,
    *,
    decision_timestamp_utc: str,
) -> dict[str, object]:
    """Validate the cold-start batch and create the first immutable log."""

    timestamp = _timestamp(decision_timestamp_utc)
    initial_root = _safe_directory(initial_root, "initial-labeling root")
    manifest, manifest_record, pool_rows, pool_result = _initial_manifest(initial_root, pool_root)
    _require(
        timestamp >= manifest["created_at_utc"],
        "initial review timestamp precedes its export",
    )
    reviewed, review_result = validate_review_table(
        initial_root / "review_request.csv",
        reviewed_path,
    )
    _require(len(reviewed) == len(pool_rows), "initial review does not close the export-all pool")
    output = _new_directory(output)
    log_rows = [
        _decision_csv_row(
            decision,
            sequence=index,
            phase=_INITIAL_PHASE,
            round_number=0,
            timestamp_utc=timestamp,
            policy=_INITIAL_PHASE,
            threshold="",
            margin="",
            top_k="",
            source_bundle_sha256="",
            source_predictions_sha256=str(pool_result["predictions_sha256"]),
            source_round_manifest_sha256=str(manifest_record["sha256"]),
            reviewed_batch_sha256=str(review_result["reviewed_sha256"]),
        )
        for index, decision in enumerate(reviewed, start=1)
    ]
    write_new_bytes(
        output / "decision_log.csv",
        _csv_bytes(CANONICAL_AL_DECISION_LOG_COLUMNS, log_rows),
    )
    log_record = _input_record(output / "decision_log.csv", "active-learning decision log", max_bytes=_MAX_TABLE_BYTES)
    completion = {
        "schema": _INITIAL_COMPLETION_SCHEMA,
        "status": "PASS",
        "profile": manifest["profile"],
        "kind": _INITIAL_PHASE,
        "round_number": None,
        "decision_timestamp_utc": timestamp,
        "initial_manifest_sha256": manifest_record["sha256"],
        "review_request_sha256": manifest["review_request"]["sha256"],
        "reviewed_batch_sha256": review_result["reviewed_sha256"],
        "source_predictions_sha256": pool_result["predictions_sha256"],
        "decision_log": log_record,
        "decision_rows": len(reviewed),
        "paper_result_reproduction": "NOT_CLAIMED",
    }
    _public_evidence(completion)
    write_new_json(output / "initial_completion.json", completion)
    _directory_closure(output, {"decision_log.csv", "initial_completion.json"})
    return completion


def _completion_manifest(path: Path) -> tuple[Mapping[str, object], dict[str, object]]:
    value, record = _read_json(path, "prior active-learning completion")
    common_fields = {
        "schema", "status", "profile", "kind", "round_number",
        "completed_at_utc", "policy", "chain", "counts", "feature_state",
        "outputs", "paper_result_reproduction",
    }
    image_scoped = value.get("schema") == _IMAGE_ROUND_COMPLETION_SCHEMA
    fields = common_fields | ({"workflow", "image"} if image_scoped else set())
    _require(
        set(value) == fields
        and value.get("schema")
        == (_IMAGE_ROUND_COMPLETION_SCHEMA if image_scoped else _ROUND_COMPLETION_SCHEMA)
        and value.get("status") == "PASS"
        and value.get("profile") in V2_PIPELINE_PROFILES
        and value.get("kind") == _ROUND_PHASE
        and (not image_scoped or value.get("workflow") == _IMAGE_WORKFLOW)
        and value.get("paper_result_reproduction") == "NOT_CLAIMED",
        "prior active-learning completion identity changed",
    )
    _strict_int(value.get("round_number"), "prior active-learning round", minimum=1)
    _timestamp(value.get("completed_at_utc"))
    _require(value.get("policy") == _policy(), "prior active-learning policy changed")
    chain = value.get("chain")
    chain_fields = {
        "genesis_initial_completion_sha256",
        "prior_round_completion_sha256",
        "selection_manifest_sha256",
        "prior_review_log_sha256",
        "reviewed_batch_sha256",
        "source_bundle_sha256",
        "source_predictions_sha256",
        "source_pool_source_sha256",
        "source_feature_archive_sha256",
        "frozen_split_sha256",
    }
    if image_scoped:
        chain_fields |= {
            "genesis_stage20_features_sha256",
            "genesis_split_sha256",
        }
    _require(
        isinstance(chain, dict) and set(chain) == chain_fields,
        "prior active-learning chain fields changed",
    )
    _sha256(chain["genesis_initial_completion_sha256"], "initial-labeling genesis")
    _sha256(chain["prior_round_completion_sha256"], "prior round parent", allow_none=True)
    _sha256(chain["source_pool_source_sha256"], "prior active-learning raw archive")
    _sha256(
        chain["source_feature_archive_sha256"],
        "prior active-learning logical feature archive",
    )
    for name in (
        "selection_manifest_sha256",
        "prior_review_log_sha256",
        "reviewed_batch_sha256",
        "source_bundle_sha256",
        "source_predictions_sha256",
        "frozen_split_sha256",
    ):
        _sha256(chain[name], f"prior active-learning {name}")
    if image_scoped:
        _sha256(
            chain["genesis_stage20_features_sha256"],
            "sequential image genesis Stage-20 features",
        )
        _sha256(chain["genesis_split_sha256"], "sequential image genesis split")
        _require(
            chain["genesis_split_sha256"] == chain["frozen_split_sha256"],
            "sequential image genesis split changed",
        )
        image = value.get("image")
        _require(
            isinstance(image, dict)
            and set(image)
            == {"image_id", "image_sha256", "group_id", "proposal_count"}
            and _SHA256.fullmatch(str(image.get("image_id"))) is not None
            and _SHA256.fullmatch(str(image.get("image_sha256"))) is not None
            and _GROUP.fullmatch(str(image.get("group_id"))) is not None
            and isinstance(image.get("proposal_count"), int)
            and not isinstance(image.get("proposal_count"), bool)
            and int(image["proposal_count"]) > 0,
            "sequential image completion identity is invalid",
        )
    counts = value.get("counts")
    _require(
        isinstance(counts, dict)
        and set(counts) == {
            "reviewed_batch_rows", "decision_log_rows", "effective_proposals",
            "feature_rows", "rescored_prediction_rows",
        }
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in counts.values()
        ),
        "prior active-learning counts are invalid",
    )
    feature_state = value.get("feature_state")
    _require(
        isinstance(feature_state, dict)
        and set(feature_state) == {
            "prototype_sha256", "pca_components_sha256", "pca_mean_sha256",
            "training_rows", "positive_rows", "pca_policy", "prototype_policy",
        }
        and feature_state["pca_policy"] == "FROZEN_FROM_PARENT_BUNDLE"
        and feature_state["prototype_policy"] == "RECOMPUTED_ACCUMULATED_OUTER_TRAIN_POSITIVES"
        and isinstance(feature_state["training_rows"], int)
        and not isinstance(feature_state["training_rows"], bool)
        and feature_state["training_rows"] >= 32
        and isinstance(feature_state["positive_rows"], int)
        and not isinstance(feature_state["positive_rows"], bool)
        and feature_state["positive_rows"] > 0,
        "prior active-learning feature-state receipt is invalid",
    )
    for name in ("prototype_sha256", "pca_components_sha256", "pca_mean_sha256"):
        _sha256(feature_state[name], f"prior active-learning {name}")
    outputs = value.get("outputs")
    output_fields = {
        "decision_log", "training_rows_sha256",
        "pool_feature_archive_sha256", "new_bundle_sha256",
        "rescored_predictions_sha256", "rescore_method",
    }
    if image_scoped:
        output_fields |= {"accumulated_al_features", "model_lineage"}
    _require(
        isinstance(outputs, dict)
        and set(outputs) == output_fields
        and isinstance(outputs["decision_log"], dict)
        and set(outputs["decision_log"])
        == {"role", "basename", "sha256", "size_bytes"}
        and outputs["decision_log"]["role"] == "active-learning decision log"
        and isinstance(outputs["decision_log"]["size_bytes"], int)
        and not isinstance(outputs["decision_log"]["size_bytes"], bool)
        and outputs["decision_log"]["size_bytes"] > 0,
        "prior active-learning output receipt is invalid",
    )
    _require(
        outputs["rescore_method"] == "VERIFIED_BUNDLE_XGB_PREDICT_PORTABLE",
        "prior active-learning rescore method changed",
    )
    portable_basename(outputs["decision_log"]["basename"], "prior decision-log basename")
    for candidate in (
        outputs["decision_log"]["sha256"],
        outputs["training_rows_sha256"],
        outputs["pool_feature_archive_sha256"],
        outputs["new_bundle_sha256"],
        outputs["rescored_predictions_sha256"],
    ):
        _sha256(candidate, "prior active-learning output")
    if image_scoped:
        completion_root = path.absolute().parent
        archive_path = completion_root / "accumulated_al_features.csv"
        archive_rows, archive_receipt = read_canonical_accumulated_feature_archive(
            archive_path
        )
        _require(
            outputs["accumulated_al_features"] == archive_receipt,
            "sequential image accumulated feature archive receipt changed",
        )
        lineage, lineage_record = _read_json(
            path.absolute().parent / "model_lineage.json",
            "sequential image model lineage",
        )
        lineage_fields = {
            "schema", "status", "workflow", "model_label", "round_number",
            "training_mode", "parent_bundle_sha256", "new_bundle_sha256",
            "genesis_initial_completion_sha256",
            "genesis_stage20_features_sha256", "genesis_split_sha256",
            "round_image_id", "round_image_sha256", "round_group_id",
            "decision_log_sha256", "training_rows_sha256",
            "cumulative_al_feature_archive_sha256",
            "effective_train_groups_sha256", "frozen_test_groups_sha256",
        }
        _require(
            isinstance(lineage, dict)
            and set(lineage) == lineage_fields
            and lineage.get("schema") == _MODEL_LINEAGE_SCHEMA
            and lineage.get("status") == "PASS"
            and lineage.get("workflow") == _IMAGE_WORKFLOW
            and lineage.get("model_label") == f"project-r{value['round_number']}"
            and lineage.get("round_number") == value["round_number"]
            and lineage.get("training_mode")
            == "FRESH_FULL_XGBOOST_FROM_CUMULATIVE_REVIEWED_ROWS"
            and lineage.get("parent_bundle_sha256") == chain["source_bundle_sha256"]
            and lineage.get("new_bundle_sha256") == outputs["new_bundle_sha256"]
            and lineage.get("genesis_initial_completion_sha256")
            == chain["genesis_initial_completion_sha256"]
            and lineage.get("genesis_stage20_features_sha256")
            == chain["genesis_stage20_features_sha256"]
            and lineage.get("genesis_split_sha256") == chain["genesis_split_sha256"]
            and lineage.get("round_image_id") == value["image"]["image_id"]
            and lineage.get("round_image_sha256") == value["image"]["image_sha256"]
            and lineage.get("round_group_id") == value["image"]["group_id"]
            and lineage.get("decision_log_sha256")
            == outputs["decision_log"]["sha256"]
            and lineage.get("training_rows_sha256")
            == outputs["training_rows_sha256"]
            and lineage.get("cumulative_al_feature_archive_sha256")
            == archive_receipt["logical_sha256"]
            and all(
                _SHA256.fullmatch(str(lineage.get(name))) is not None
                for name in (
                    "effective_train_groups_sha256",
                    "frozen_test_groups_sha256",
                )
            ),
            "sequential image model lineage changed",
        )
        _require(
            isinstance(outputs["model_lineage"], dict)
            and outputs["model_lineage"] == lineage_record,
            "sequential image lineage receipt changed",
        )
        decisions, _decision_rows, decision_record = _read_decision_log(
            completion_root / "decision_log.csv"
        )
        round_decisions = tuple(
            decision for decision in decisions if decision.phase == _ROUND_PHASE
        )
        current_round_decisions = tuple(
            decision
            for decision in round_decisions
            if decision.round_number == value["round_number"]
        )
        round_by_proposal = {
            decision.proposal_id: decision for decision in round_decisions
        }
        archive_by_proposal: dict[
            str, list[CanonicalActiveLearningTrainingRow]
        ] = {}
        for archive_row in archive_rows:
            archive_by_proposal.setdefault(archive_row.proposal_id, []).append(
                archive_row
            )
        expected_rounds = set(range(1, int(value["round_number"]) + 1))
        groups_by_round = {
            round_number: {
                decision.group_id
                for decision in round_decisions
                if decision.round_number == round_number
            }
            for round_number in expected_rounds
        }
        images_by_round = {
            round_number: {
                (decision.image_id, decision.image_sha256)
                for decision in round_decisions
                if decision.round_number == round_number
            }
            for round_number in expected_rounds
        }
        _require(
            decision_record == outputs["decision_log"]
            and len(round_by_proposal) == len(round_decisions)
            and {decision.round_number for decision in round_decisions}
            == expected_rounds
            and all(len(groups_by_round[number]) == 1 for number in expected_rounds)
            and all(len(images_by_round[number]) == 1 for number in expected_rounds)
            and len({decision.group_id for decision in round_decisions})
            == len(expected_rounds)
            and len({decision.image_sha256 for decision in round_decisions})
            == len(expected_rounds)
            and set(archive_by_proposal) == set(round_by_proposal)
            and all(
                all(
                    (
                        row.proposal_sha256,
                        row.image_id,
                        row.image_sha256,
                        row.group_id,
                    )
                    == (
                        round_by_proposal[proposal_id].proposal_sha256,
                        round_by_proposal[proposal_id].image_id,
                        round_by_proposal[proposal_id].image_sha256,
                        round_by_proposal[proposal_id].group_id,
                    )
                    for row in proposal_rows
                )
                for proposal_id, proposal_rows in archive_by_proposal.items()
            )
            and archive_receipt["proposal_count"] == len(round_by_proposal)
            and archive_receipt["row_count"]
            == len(round_by_proposal) * len(CANONICAL_FEATURE_CROP_SCALES)
            and archive_receipt["groups_sha256"]
            == compact_json_sha256(
                sorted({decision.group_id for decision in round_decisions})
            )
            and counts["decision_log_rows"] == len(decisions)
            and counts["effective_proposals"] == len(_latest_decisions(decisions))
            and counts["reviewed_batch_rows"] == len(current_round_decisions)
            and counts["feature_rows"]
            == len(decisions) * len(CANONICAL_FEATURE_CROP_SCALES)
            and counts["rescored_prediction_rows"]
            == value["image"]["proposal_count"],
            "sequential image cumulative archive does not close its decision lineage",
        )
        model_bundle = _verify_bundle(completion_root / "model_bundle")
        provenance = getattr(model_bundle, "provenance", None)
        details = (
            provenance.get("details")
            if isinstance(provenance, Mapping)
            else None
        )
        _require(
            isinstance(details, Mapping),
            "sequential image parent bundle provenance differs from its completion lineage",
        )
        bundle_feature_state = CanonicalFeatureState(
            prototype=model_bundle.prototype,
            pca_components=model_bundle.pca_components,
            pca_mean=model_bundle.pca_mean,
            training_row_count=feature_state["training_rows"],
            positive_row_count=feature_state["positive_rows"],
        )
        validate_canonical_feature_state(bundle_feature_state)
        bundle_feature_state_hashes = _feature_state_hashes(bundle_feature_state)
        _require(
            model_bundle.bundle_sha256 == outputs["new_bundle_sha256"]
            and details.get("reviewed_sha256")
            == outputs["decision_log"]["sha256"]
            and details.get("split_manifest_sha256")
            == chain["frozen_split_sha256"]
            and details.get("raw_features_sha256")
            == outputs["training_rows_sha256"]
            and details.get("active_learning_workflow") == lineage["workflow"]
            and details.get("model_label") == lineage["model_label"]
            and details.get("active_learning_round_number")
            == lineage["round_number"]
            and details.get("parent_bundle_sha256")
            == lineage["parent_bundle_sha256"]
            == chain["source_bundle_sha256"]
            and details.get("genesis_stage20_features_sha256")
            == lineage["genesis_stage20_features_sha256"]
            == chain["genesis_stage20_features_sha256"]
            and details.get("genesis_split_sha256")
            == lineage["genesis_split_sha256"]
            == chain["genesis_split_sha256"]
            and details.get("round_image_id") == lineage["round_image_id"]
            and details.get("round_image_sha256")
            == lineage["round_image_sha256"]
            and details.get("round_group_id") == lineage["round_group_id"]
            and details.get("cumulative_al_feature_archive_sha256")
            == lineage["cumulative_al_feature_archive_sha256"]
            == archive_receipt["logical_sha256"]
            and details.get("effective_train_groups_sha256")
            == lineage["effective_train_groups_sha256"]
            and details.get("frozen_test_groups_sha256")
            == lineage["frozen_test_groups_sha256"]
            and feature_state["training_rows"]
            == details.get("training_row_count")
            and feature_state["positive_rows"]
            == details.get("positive_training_row_count")
            and feature_state["prototype_sha256"]
            == bundle_feature_state_hashes["prototype_sha256"]
            and feature_state["pca_components_sha256"]
            == bundle_feature_state_hashes["pca_components_sha256"]
            and feature_state["pca_mean_sha256"]
            == bundle_feature_state_hashes["pca_mean_sha256"],
            "sequential image parent bundle provenance differs from its completion lineage",
        )
    return value, record


def _initial_completion_manifest(
    path: Path,
) -> tuple[Mapping[str, object], dict[str, object]]:
    value, record = _read_json(path, "initial-labeling completion")
    fields = {
        "schema", "status", "profile", "kind", "round_number",
        "decision_timestamp_utc", "initial_manifest_sha256",
        "review_request_sha256", "reviewed_batch_sha256",
        "source_predictions_sha256", "decision_log", "decision_rows",
        "paper_result_reproduction",
    }
    _require(
        set(value) == fields
        and value.get("schema") == _INITIAL_COMPLETION_SCHEMA
        and value.get("status") == "PASS"
        and value.get("profile") in V2_PIPELINE_PROFILES
        and value.get("kind") == _INITIAL_PHASE
        and value.get("round_number") is None
        and value.get("paper_result_reproduction") == "NOT_CLAIMED",
        "initial-labeling completion identity changed",
    )
    _timestamp(value.get("decision_timestamp_utc"))
    for name in (
        "initial_manifest_sha256", "review_request_sha256",
        "reviewed_batch_sha256", "source_predictions_sha256",
    ):
        _sha256(value.get(name), f"initial-labeling {name}")
    decision_log = value.get("decision_log")
    _require(
        isinstance(decision_log, dict)
        and set(decision_log) == {"role", "basename", "sha256", "size_bytes"}
        and decision_log.get("role") == "active-learning decision log"
        and isinstance(decision_log.get("size_bytes"), int)
        and not isinstance(decision_log.get("size_bytes"), bool)
        and decision_log["size_bytes"] > 0
        and isinstance(value.get("decision_rows"), int)
        and not isinstance(value.get("decision_rows"), bool)
        and value["decision_rows"] > 0,
        "initial-labeling completion decision-log receipt is invalid",
    )
    portable_basename(decision_log["basename"], "initial decision-log basename")
    _sha256(decision_log["sha256"], "initial decision log")
    return value, record


def _validate_split_genesis(
    split: Mapping[str, object],
    decisions: Sequence[CanonicalActiveLearningDecision],
    initial_completion: Mapping[str, object],
) -> None:
    _require(
        bool(decisions)
        and all(decision.phase == _INITIAL_PHASE for decision in decisions)
        and split["reviewed_sha256"] == initial_completion["reviewed_batch_sha256"],
        "frozen split is not bound to the initial reviewed batch",
    )
    train = frozenset(split["train_groups"])
    test = frozenset(split["test_groups"])
    latest = _latest_decisions(decisions)
    _require(
        {decision.group_id for decision in latest.values()} <= train | test,
        "initial decisions contain a group outside the frozen split",
    )
    effective = [decision for decision in latest.values() if decision.review_weight > 0.0]
    skipped = [decision for decision in latest.values() if decision.review_weight == 0.0]

    def counts(groups: frozenset[str]) -> dict[str, int]:
        selected = [decision for decision in effective if decision.group_id in groups]
        return {
            "rows": len(selected),
            "negative": sum(decision.label == 0 for decision in selected),
            "positive": sum(decision.label == 1 for decision in selected),
        }

    _require(
        split["coverage"] == {"train": counts(train), "test": counts(test)}
        and split["effective_rows"] == len(effective)
        and split["skipped_rows"] == len(skipped),
        "frozen split coverage does not close the initial decisions",
    )


def _policy() -> dict[str, object]:
    return {
        "score_source": "xgb_p",
        "classifier_policy": "XGB_ONLY",
        "threshold": CANONICAL_DECISION_THRESHOLD,
        "distance": "abs(xgb_p-threshold)",
        "margin": CANONICAL_AL_MARGIN,
        "only_within_margin": True,
        "top_k": CANONICAL_AL_TOP_K,
        "tie_break": ["distance_to_threshold", "proposal_id"],
        "proposal_scale": 1.0,
        "exclude_reviewed": True,
        "exclude_frozen_test_groups": True,
    }


def _select(
    pool_rows: Sequence[CanonicalActiveLearningPoolRow],
    reviewed_ids: frozenset[str],
    train_groups: frozenset[str],
    test_groups: frozenset[str],
) -> tuple[
    tuple[tuple[CanonicalActiveLearningPoolRow, float, float], ...],
    dict[str, int],
]:
    _require(all(row.xgb_p is not None for row in pool_rows), "active-learning selection requires XGB probabilities")
    observed_groups = {row.group_id for row in pool_rows}
    _require(
        observed_groups <= train_groups | test_groups,
        "active-learning pool contains a group outside the frozen split",
    )
    reviewed_excluded = test_excluded = outside_margin = 0
    eligible: list[tuple[CanonicalActiveLearningPoolRow, float, float]] = []
    for row in pool_rows:
        if row.proposal_id in reviewed_ids:
            reviewed_excluded += 1
            continue
        if row.group_id in test_groups:
            test_excluded += 1
            continue
        _require(row.group_id in train_groups, "active-learning candidate is outside the frozen training groups")
        probability = float(row.xgb_p)
        distance = abs(probability - CANONICAL_DECISION_THRESHOLD)
        if distance > CANONICAL_AL_MARGIN:
            outside_margin += 1
            continue
        uncertainty = max(0.0, 1.0 - min(1.0, distance / CANONICAL_AL_MARGIN))
        eligible.append((row, distance, uncertainty))
    eligible.sort(key=lambda item: (item[1], item[0].proposal_id))
    selected = tuple(eligible[:CANONICAL_AL_TOP_K])
    _require(bool(selected), "no unreviewed training-pool candidates are within the AL margin")
    return selected, {
        "pool_rows": len(pool_rows),
        "reviewed_excluded": reviewed_excluded,
        "frozen_test_group_excluded": test_excluded,
        "outside_margin_excluded": outside_margin,
        "margin_eligible": len(eligible),
        "selected": len(selected),
    }


def _single_image_identity(
    pool_rows: Sequence[CanonicalActiveLearningPoolRow],
) -> dict[str, object]:
    identities = {
        (row.image_id, row.image_sha256, row.group_id)
        for row in pool_rows
    }
    _require(
        len(identities) == 1,
        "sequential image active learning requires exactly one image and one group",
    )
    image_id, image_sha256, group_id = next(iter(identities))
    return {
        "image_id": image_id,
        "image_sha256": image_sha256,
        "group_id": group_id,
        "proposal_count": len(pool_rows),
    }


def _select_single_new_image(
    pool_rows: Sequence[CanonicalActiveLearningPoolRow],
    reviewed_ids: frozenset[str],
    known_groups: frozenset[str],
    known_image_sha256s: frozenset[str],
    train_groups: frozenset[str],
    test_groups: frozenset[str],
) -> tuple[
    tuple[tuple[CanonicalActiveLearningPoolRow, float, float], ...],
    dict[str, int],
    dict[str, object],
]:
    _require(
        all(row.xgb_p is not None for row in pool_rows),
        "sequential image active learning requires XGB probabilities",
    )
    image = _single_image_identity(pool_rows)
    group_id = str(image["group_id"])
    _require(
        group_id not in train_groups
        and group_id not in test_groups
        and group_id not in known_groups
        and str(image["image_sha256"]) not in known_image_sha256s,
        "sequential image active learning requires a previously unseen image group and image identity",
    )
    _require(
        not {row.proposal_id for row in pool_rows} & set(reviewed_ids),
        "sequential image active learning received an already reviewed proposal",
    )
    outside_margin = 0
    eligible: list[tuple[CanonicalActiveLearningPoolRow, float, float]] = []
    for row in pool_rows:
        probability = float(row.xgb_p)
        distance = abs(probability - CANONICAL_DECISION_THRESHOLD)
        if distance > CANONICAL_AL_MARGIN:
            outside_margin += 1
            continue
        uncertainty = max(
            0.0,
            1.0 - min(1.0, distance / CANONICAL_AL_MARGIN),
        )
        eligible.append((row, distance, uncertainty))
    eligible.sort(key=lambda item: (item[1], item[0].proposal_id))
    selected = tuple(eligible[:CANONICAL_AL_TOP_K])
    _require(
        bool(selected),
        "the new image has no candidates inside the canonical AL margin",
    )
    return selected, {
        "pool_rows": len(pool_rows),
        "reviewed_excluded": 0,
        "frozen_test_group_excluded": 0,
        "outside_margin_excluded": outside_margin,
        "margin_eligible": len(eligible),
        "selected": len(selected),
    }, image


def _shortlist_rows(selected: Sequence[tuple[CanonicalActiveLearningPoolRow, float, float]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for rank, (row, distance, uncertainty) in enumerate(selected, start=1):
        output.append(
            {
                **_pool_row(row, scored=True),
                "distance_to_threshold": _format_float(distance),
                "uncertainty": _format_float(uncertainty),
                "selection_rank": str(rank),
            }
        )
    return output


def _candidate_identity(
    row: CanonicalActiveLearningPoolRow | CanonicalActiveLearningDecision,
) -> tuple[str, str, str, str, str]:
    return (
        row.proposal_id,
        row.proposal_sha256,
        row.image_id,
        row.image_sha256,
        row.group_id,
    )


def _validate_cumulative_round_pool(
    prior_round_completion_path: Path,
    prior: Mapping[str, object],
    decisions: Sequence[CanonicalActiveLearningDecision],
    current_pool_rows: Sequence[CanonicalActiveLearningPoolRow],
) -> None:
    prior_pool_rows, prior_pool_result, _records = _read_pool(
        prior_round_completion_path.absolute().parent / "rescored_pool",
        require_scores=True,
    )
    outputs = prior["outputs"]
    chain = prior["chain"]
    _require(
        prior_pool_result["profile"] == prior["profile"]
        and prior_pool_result["bundle_sha256"] == outputs["new_bundle_sha256"]
        and prior_pool_result["predictions_sha256"]
        == outputs["rescored_predictions_sha256"]
        and prior_pool_result["source_sha256"]
        == chain["source_pool_source_sha256"]
        and prior_pool_result["feature_archive_sha256"]
        == outputs["pool_feature_archive_sha256"],
        "prior active-learning rescore closure is broken",
    )
    prior_by_id = {row.proposal_id: row for row in prior_pool_rows}
    current_by_id = {row.proposal_id: row for row in current_pool_rows}
    reviewed = {
        decision.proposal_id: decision
        for decision in decisions
        if decision.phase == _ROUND_PHASE
    }
    _require(bool(reviewed), "later active-learning ancestry has no reviewed round candidates")
    _require(
        all(
            proposal_id in prior_by_id
            and proposal_id in current_by_id
            and _candidate_identity(prior_by_id[proposal_id])
            == _candidate_identity(decision)
            == _candidate_identity(current_by_id[proposal_id])
            for proposal_id, decision in reviewed.items()
        ),
        "cumulative active-learning pool omits or changes a prior reviewed candidate",
    )


def begin_canonical_active_learning_round(
    pool_root: Path,
    bundle_root: Path,
    frozen_split_path: Path,
    decision_log_path: Path,
    output: Path,
    *,
    round_number: int,
    created_at_utc: str,
    initial_completion_path: Path | None = None,
    prior_round_completion_path: Path | None = None,
    image_scoped: bool = False,
) -> dict[str, object]:
    """Select, bind, and pause one canonical model-driven AL round."""

    number = _strict_int(round_number, "active-learning round number", minimum=1)
    timestamp = _timestamp(created_at_utc)
    pool_rows, pool_result, pool_records = _read_pool(pool_root, require_scores=True)
    _sha256(pool_result["source_sha256"], "active-learning scored-pool source")
    _sha256(
        pool_result["feature_archive_sha256"],
        "active-learning scored-pool logical feature archive",
    )
    bundle = _verify_bundle(bundle_root)
    _require(
        bundle.profile in GPU_EXECUTION_PROFILES,
        "active-learning round execution requires a canonical GPU bundle, an efficient Lite GPU bundle, or the full-image GPU bundle",
    )
    _require(
        pool_result["bundle_sha256"] == bundle.bundle_sha256
        and pool_result["profile"] == bundle.profile,
        "active-learning pool was not scored by the supplied profile-bound bundle",
    )
    train_groups, test_groups, split_value, split_record = _split_manifest(frozen_split_path)
    _require(
        split_value["profile"] == bundle.profile,
        "active-learning frozen split profile differs from the supplied bundle",
    )
    decisions, _log_rows, log_record = _read_decision_log(decision_log_path)
    _require(
        timestamp >= decisions[-1].decision_timestamp_utc,
        "active-learning selection timestamp precedes the review log",
    )
    latest = _latest_decisions(decisions)
    prior_record: dict[str, object] | None = None
    genesis_sha256: str
    if number == 1:
        _require(prior_round_completion_path is None, "round 1 cannot name a prior AL round")
        _require(
            all(decision.phase == _INITIAL_PHASE for decision in decisions),
            "round 1 decision log already contains an AL round",
        )
        _require(initial_completion_path is not None, "round 1 requires its initial-labeling completion")
        initial_completion, initial_record = _initial_completion_manifest(initial_completion_path)
        _require(
            initial_completion["profile"] == bundle.profile
            and initial_completion["decision_log"]["sha256"]
            == log_record["sha256"],
            "round-1 initial-labeling profile or decision log differs from the supplied bundle",
        )
        _validate_split_genesis(split_value, decisions, initial_completion)
        _require(
            timestamp >= initial_completion["decision_timestamp_utc"],
            "round-1 selection timestamp precedes initial completion",
        )
        genesis_sha256 = str(initial_record["sha256"])
    else:
        _require(initial_completion_path is None, "later rounds use their prior-round ancestry")
        _require(prior_round_completion_path is not None, "later AL rounds require their prior completion")
        prior, prior_record = _completion_manifest(prior_round_completion_path)
        _require(
            prior["profile"] == bundle.profile
            and prior["round_number"] == number - 1
            and prior["outputs"]["decision_log"]["sha256"] == log_record["sha256"]
            and prior["outputs"]["new_bundle_sha256"] == bundle.bundle_sha256
            and max(decision.round_number for decision in decisions) == number - 1,
            "active-learning prior-round chain is broken",
        )
        if image_scoped:
            _require(
                prior.get("schema") == _IMAGE_ROUND_COMPLETION_SCHEMA
                and prior.get("workflow") == _IMAGE_WORKFLOW
                and prior["chain"]["genesis_split_sha256"]
                == split_record["sha256"],
                "sequential image active-learning ancestry or genesis split changed",
            )
        else:
            _validate_cumulative_round_pool(
                prior_round_completion_path,
                prior,
                decisions,
                pool_rows,
            )
        _require(
            timestamp >= prior["completed_at_utc"],
            "active-learning selection timestamp precedes the prior completion",
        )
        genesis_sha256 = str(prior["chain"]["genesis_initial_completion_sha256"])
    image: dict[str, object] | None = None
    if image_scoped:
        selected, counts, image = _select_single_new_image(
            pool_rows,
            frozenset(latest),
            frozenset(decision.group_id for decision in decisions),
            frozenset(decision.image_sha256 for decision in decisions),
            train_groups,
            test_groups,
        )
    else:
        selected, counts = _select(
            pool_rows,
            frozenset(latest),
            train_groups,
            test_groups,
        )
    output = _new_directory(output)
    shortlist_payload = _csv_bytes(
        CANONICAL_AL_SHORTLIST_COLUMNS,
        _shortlist_rows(selected),
    )
    write_new_bytes(output / "shortlist.csv", shortlist_payload)
    review_result = write_review_export(
        _review_rows([item[0] for item in selected]),
        output / "review_request.csv",
        canonical_actions=True,
    )
    shortlist_record = _input_record(output / "shortlist.csv", "active-learning shortlist", max_bytes=_MAX_TABLE_BYTES)
    review_record = _input_record(output / "review_request.csv", "active-learning review request", max_bytes=_MAX_TABLE_BYTES)
    chain = {
        "genesis_initial_completion_sha256": genesis_sha256,
        "prior_round_completion_sha256": prior_record["sha256"] if prior_record is not None else None,
        "prior_review_log_sha256": log_record["sha256"],
        "source_bundle_sha256": bundle.bundle_sha256,
        "source_predictions_sha256": pool_result["predictions_sha256"],
        "source_pool_source_sha256": pool_result["source_sha256"],
        "source_feature_archive_sha256": pool_result["feature_archive_sha256"],
        "frozen_split_sha256": split_record["sha256"],
    }
    if image_scoped:
        details = bundle.provenance.get("details")
        _require(
            isinstance(details, Mapping),
            "sequential image source bundle provenance is missing",
        )
        genesis_features_sha256 = details.get(
            "genesis_stage20_features_sha256",
            details.get("raw_features_sha256"),
        )
        _sha256(
            genesis_features_sha256,
            "sequential image genesis Stage-20 features",
        )
        chain["genesis_stage20_features_sha256"] = genesis_features_sha256
        chain["genesis_split_sha256"] = split_record["sha256"]
    round_id = compact_json_sha256(
        {
            "domain": (
                "compag-curation-canonical-active-learning-image-round-id/v3"
                if image_scoped
                else "compag-curation-canonical-active-learning-round-id/v2"
            ),
            "round_number": number,
            "policy": _policy(),
            "chain": chain,
            "shortlist_sha256": shortlist_record["sha256"],
            **({"workflow": _IMAGE_WORKFLOW, "image": image} if image_scoped else {}),
        }
    )
    manifest = {
        "schema": _IMAGE_ROUND_SCHEMA if image_scoped else _ROUND_SCHEMA,
        "status": "PAUSED_FOR_REVIEW",
        "profile": bundle.profile,
        "kind": _ROUND_PHASE,
        "round_number": number,
        "round_id": round_id,
        "created_at_utc": timestamp,
        "policy": _policy(),
        "chain": chain,
        "inputs": {
            "pool": {"files": list(pool_records), "result": dict(pool_result)},
            "frozen_split": split_record,
            "decision_log": log_record,
        },
        "counts": counts,
        "shortlist": shortlist_record,
        "review_request": review_record,
        "review_contract": review_result["review_contract"],
        "paper_result_reproduction": "NOT_CLAIMED",
    }
    if image_scoped:
        manifest["workflow"] = _IMAGE_WORKFLOW
        manifest["image"] = image
    _public_evidence(manifest)
    write_new_json(output / "round_manifest.json", manifest)
    _directory_closure(output, {"shortlist.csv", "review_request.csv", "round_manifest.json"})
    return manifest


def _selection_manifest(
    root: Path,
    pool_root: Path,
    bundle_root: Path,
    frozen_split_path: Path,
    decision_log_path: Path,
    initial_completion_path: Path | None,
    prior_round_completion_path: Path | None,
    *,
    image_scoped: bool = False,
) -> tuple[Mapping[str, object], dict[str, object], tuple[CanonicalActiveLearningPoolRow, ...], Any, frozenset[str], frozenset[str], tuple[CanonicalActiveLearningDecision, ...], list[dict[str, str]], dict[str, object]]:
    root = _safe_directory(root, "active-learning selection root")
    _directory_closure(root, {"shortlist.csv", "review_request.csv", "round_manifest.json"})
    manifest, manifest_record = _read_json(root / "round_manifest.json", "active-learning round manifest")
    fields = {
        "schema", "status", "profile", "kind", "round_number", "round_id",
        "created_at_utc", "policy", "chain", "inputs", "counts", "shortlist",
        "review_request", "review_contract", "paper_result_reproduction",
    }
    if image_scoped:
        fields |= {"workflow", "image"}
    _require(
        set(manifest) == fields
        and manifest.get("schema")
        == (_IMAGE_ROUND_SCHEMA if image_scoped else _ROUND_SCHEMA)
        and manifest.get("status") == "PAUSED_FOR_REVIEW"
        and manifest.get("profile") in V2_PIPELINE_PROFILES
        and manifest.get("kind") == _ROUND_PHASE
        and manifest.get("policy") == _policy()
        and (not image_scoped or manifest.get("workflow") == _IMAGE_WORKFLOW)
        and manifest.get("review_contract") == "CANONICAL_ACTION_WEIGHTED_V2"
        and manifest.get("paper_result_reproduction") == "NOT_CLAIMED",
        "active-learning paused manifest identity changed",
    )
    number = _strict_int(manifest.get("round_number"), "active-learning round number", minimum=1)
    _timestamp(manifest.get("created_at_utc"))
    _sha256(manifest.get("round_id"), "active-learning round ID")
    pool_rows, pool_result, pool_records = _read_pool(pool_root, require_scores=True)
    bundle = _verify_bundle(bundle_root)
    train_groups, test_groups, split_value, split_record = _split_manifest(frozen_split_path)
    decisions, raw_log_rows, log_record = _read_decision_log(decision_log_path)
    _require(
        pool_result["bundle_sha256"] == bundle.bundle_sha256
        and pool_result["profile"] == bundle.profile
        and split_value["profile"] == bundle.profile
        and manifest["profile"] == bundle.profile
        and manifest["inputs"] == {
            "pool": {"files": list(pool_records), "result": dict(pool_result)},
            "frozen_split": split_record,
            "decision_log": log_record,
        },
        "active-learning paused inputs changed",
    )
    chain = manifest["chain"]
    chain_fields = {
        "genesis_initial_completion_sha256",
        "prior_round_completion_sha256", "prior_review_log_sha256",
        "source_bundle_sha256", "source_predictions_sha256",
        "source_pool_source_sha256", "source_feature_archive_sha256",
        "frozen_split_sha256",
    }
    if image_scoped:
        chain_fields |= {
            "genesis_stage20_features_sha256",
            "genesis_split_sha256",
        }
    _require(
        isinstance(chain, dict)
        and set(chain) == chain_fields
        and chain["prior_review_log_sha256"] == log_record["sha256"]
        and chain["source_bundle_sha256"] == bundle.bundle_sha256
        and chain["source_predictions_sha256"] == pool_result["predictions_sha256"]
        and chain["source_pool_source_sha256"] == pool_result["source_sha256"]
        and chain["source_feature_archive_sha256"]
        == pool_result["feature_archive_sha256"]
        and chain["frozen_split_sha256"] == split_record["sha256"],
        "active-learning paused hash chain changed",
    )
    if image_scoped:
        _require(
            chain["genesis_split_sha256"] == split_record["sha256"]
            and chain["genesis_stage20_features_sha256"]
            == bundle.provenance["details"].get(
                "genesis_stage20_features_sha256",
                bundle.provenance["details"].get("raw_features_sha256"),
            ),
            "sequential image genesis features or split changed",
        )
        _sha256(
            chain["genesis_stage20_features_sha256"],
            "sequential image genesis Stage-20 features",
        )
    _sha256(chain["genesis_initial_completion_sha256"], "initial-labeling genesis")
    _sha256(
        chain["source_pool_source_sha256"],
        "active-learning raw archive",
    )
    _sha256(
        chain["source_feature_archive_sha256"],
        "active-learning logical feature archive",
    )
    _sha256(
        chain["prior_round_completion_sha256"],
        "prior active-learning completion",
        allow_none=True,
    )
    if number == 1:
        _require(
            initial_completion_path is not None
            and prior_round_completion_path is None
            and chain["prior_round_completion_sha256"] is None
            and all(decision.phase == _INITIAL_PHASE for decision in decisions),
            "active-learning round-1 ancestry changed",
        )
        initial_completion, initial_record = _initial_completion_manifest(initial_completion_path)
        _require(
            initial_record["sha256"] == chain["genesis_initial_completion_sha256"]
            and initial_completion["profile"] == bundle.profile
            and initial_completion["decision_log"]["sha256"] == log_record["sha256"],
            "active-learning initial-labeling ancestry is broken",
        )
        _validate_split_genesis(split_value, decisions, initial_completion)
    else:
        _require(
            initial_completion_path is None and prior_round_completion_path is not None,
            "later active-learning resume requires the prior completion",
        )
        prior, prior_record = _completion_manifest(prior_round_completion_path)
        _require(
            prior["profile"] == bundle.profile
            and prior["round_number"] == number - 1
            and prior_record["sha256"] == chain["prior_round_completion_sha256"]
            and prior["chain"]["genesis_initial_completion_sha256"]
            == chain["genesis_initial_completion_sha256"]
            and prior["outputs"]["decision_log"]["sha256"] == log_record["sha256"]
            and prior["outputs"]["new_bundle_sha256"] == bundle.bundle_sha256
            and max(decision.round_number for decision in decisions) == number - 1,
            "active-learning resume ancestry is broken",
        )
        if image_scoped:
            _require(
                prior.get("schema") == _IMAGE_ROUND_COMPLETION_SCHEMA
                and prior.get("workflow") == _IMAGE_WORKFLOW
                and prior["chain"]["genesis_split_sha256"]
                == split_record["sha256"],
                "sequential image resume ancestry or genesis split changed",
            )
        else:
            _validate_cumulative_round_pool(
                prior_round_completion_path,
                prior,
                decisions,
                pool_rows,
            )
    if image_scoped:
        selected, counts, image = _select_single_new_image(
            pool_rows,
            frozenset(_latest_decisions(decisions)),
            frozenset(decision.group_id for decision in decisions),
            frozenset(decision.image_sha256 for decision in decisions),
            train_groups,
            test_groups,
        )
        _require(
            manifest.get("image") == image,
            "sequential image selection identity changed",
        )
    else:
        selected, counts = _select(
            pool_rows,
            frozenset(_latest_decisions(decisions)),
            train_groups,
            test_groups,
        )
        image = None
    expected_shortlist = _csv_bytes(CANONICAL_AL_SHORTLIST_COLUMNS, _shortlist_rows(selected))
    observed_shortlist, _ = _stable_payload(root / "shortlist.csv", "active-learning shortlist", max_bytes=_MAX_TABLE_BYTES)
    shortlist_record = _input_record(root / "shortlist.csv", "active-learning shortlist", max_bytes=_MAX_TABLE_BYTES)
    review_record = _pending_request_record(
        root / "review_request.csv",
        [item[0] for item in selected],
        "active-learning review request",
    )
    expected_round_id = compact_json_sha256(
        {
            "domain": (
                "compag-curation-canonical-active-learning-image-round-id/v3"
                if image_scoped
                else "compag-curation-canonical-active-learning-round-id/v2"
            ),
            "round_number": number,
            "policy": _policy(),
            "chain": dict(chain),
            "shortlist_sha256": shortlist_record["sha256"],
            **(
                {"workflow": _IMAGE_WORKFLOW, "image": image}
                if image_scoped
                else {}
            ),
        }
    )
    _require(
        observed_shortlist == expected_shortlist
        and manifest["counts"] == counts
        and manifest["shortlist"] == shortlist_record
        and manifest["review_request"] == review_record
        and manifest["round_id"] == expected_round_id,
        "active-learning shortlist or round binding changed",
    )
    return (
        manifest, manifest_record, pool_rows, bundle, train_groups, test_groups,
        decisions, raw_log_rows, log_record,
    )


def _append_decisions(
    prior_rows: Sequence[Mapping[str, str]],
    reviewed: Sequence[ReviewRow],
    *,
    round_number: int,
    timestamp_utc: str,
    bundle_sha256: str,
    predictions_sha256: str,
    round_manifest_sha256: str,
    reviewed_sha256: str,
) -> tuple[list[dict[str, object]], tuple[CanonicalActiveLearningDecision, ...]]:
    rows: list[dict[str, object]] = [dict(row) for row in prior_rows]
    start = len(rows) + 1
    for offset, decision in enumerate(reviewed):
        rows.append(
            _decision_csv_row(
                decision,
                sequence=start + offset,
                phase=_ROUND_PHASE,
                round_number=round_number,
                timestamp_utc=timestamp_utc,
                policy="XGB_ONLY_DISTANCE_TO_FIXED_THRESHOLD",
                threshold=_format_float(CANONICAL_DECISION_THRESHOLD),
                margin=_format_float(CANONICAL_AL_MARGIN),
                top_k=str(CANONICAL_AL_TOP_K),
                source_bundle_sha256=bundle_sha256,
                source_predictions_sha256=predictions_sha256,
                source_round_manifest_sha256=round_manifest_sha256,
                reviewed_batch_sha256=reviewed_sha256,
            )
        )
    parsed = _parse_decision_rows(rows)
    return rows, parsed


def _training_row_hash(
    rows: Sequence[CanonicalActiveLearningTrainingRow],
    latest: Mapping[str, CanonicalActiveLearningDecision],
) -> str:
    import numpy as np

    values: list[dict[str, object]] = []
    for row in rows:
        decision = latest[row.proposal_id]
        embedding = np.asarray(row.raw_feature.embedding, dtype="<f4")
        raw_payload = {
            "proposal_index": row.raw_feature.proposal_index,
            "scale": _format_float(row.raw_feature.scale),
            "predicted_iou": _format_float(row.raw_feature.predicted_iou),
            "stability_score": _format_float(row.raw_feature.stability_score),
            "values": {
                name: _format_float(row.raw_feature.values[name])
                for name in CANONICAL_RAW_FEATURE_ORDER
            },
            "embedding_sha256": hashlib.sha256(
                embedding.tobytes(order="C")
            ).hexdigest(),
        }
        values.append(
            {
                "proposal_id": row.proposal_id,
                "proposal_sha256": row.proposal_sha256,
                "image_id": row.image_id,
                "image_sha256": row.image_sha256,
                "group_id": row.group_id,
                "source_sha256": row.source_sha256,
                "scale": _format_float(row.raw_feature.scale),
                "label": decision.label,
                "review_action": decision.review_action,
                "review_weight": _format_float(decision.review_weight),
                "raw_feature_sha256": compact_json_sha256(raw_payload),
            }
        )
    return compact_json_sha256(values)


def _expected_bundle_provenance(
    rows: Sequence[CanonicalActiveLearningTrainingRow],
    latest: Mapping[str, CanonicalActiveLearningDecision],
    train_groups: frozenset[str],
    *,
    decision_log_sha256: str,
    split_sha256: str,
    training_rows_sha256: str,
    state: CanonicalFeatureState,
    profile: str | None = None,
    device: str | None = None,
    lineage: Mapping[str, object] | None = None,
) -> dict[str, object]:
    effective = [
        row
        for row in rows
        if row.group_id in train_groups and latest[row.proposal_id].review_weight > 0.0
    ]
    result: dict[str, object] = {
        "reviewed_sha256": decision_log_sha256,
        "split_manifest_sha256": split_sha256,
        "raw_features_sha256": training_rows_sha256,
        "feature_state_fit_group_sha256": compact_json_sha256(
            sorted({row.group_id for row in effective})
        ),
        "feature_state_fit_rows_sha256": compact_json_sha256(
            [
                {
                    "proposal_id": row.proposal_id,
                    "scale": _format_float(row.raw_feature.scale),
                    "group_id": row.group_id,
                    "label": latest[row.proposal_id].label,
                }
                for row in effective
            ]
        ),
        "feature_state_fit_scope": "OUTER_TRAIN_ROWS_BEFORE_CV",
        "validation_feature_scale": 1.0,
        "cv_score_interpretation": "CONDITIONAL_ON_OUTER_TRAIN_PREFIT_PROTOTYPE_AND_PCA",
        "training_row_count": state.training_row_count,
        "positive_training_row_count": state.positive_row_count,
        "fixed_threshold": 0.5,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        "ubj_parity_status": "PASS",
        "paper_result_reproduction": "NOT_CLAIMED",
    }
    if profile is not None or device is not None:
        resolved_profile, resolved_device = _canonical_profile_device(
            profile,
            device,
        )
        result.update(profile=resolved_profile, device=resolved_device)
    if lineage is not None:
        expected_lineage_fields = {
            "active_learning_workflow",
            "model_label",
            "active_learning_round_number",
            "parent_bundle_sha256",
            "genesis_stage20_features_sha256",
            "genesis_split_sha256",
            "round_image_id",
            "round_image_sha256",
            "round_group_id",
            "cumulative_al_feature_archive_sha256",
            "effective_train_groups_sha256",
            "frozen_test_groups_sha256",
        }
        _require(
            set(lineage) == expected_lineage_fields
            and lineage.get("active_learning_workflow") == _IMAGE_WORKFLOW
            and isinstance(lineage.get("model_label"), str)
            and re.fullmatch(r"project-r[1-9][0-9]*", str(lineage["model_label"]))
            is not None
            and isinstance(lineage.get("active_learning_round_number"), int)
            and not isinstance(lineage.get("active_learning_round_number"), bool)
            and int(lineage["active_learning_round_number"]) >= 1
            and lineage.get("model_label")
            == f"project-r{lineage['active_learning_round_number']}"
            and _GROUP.fullmatch(str(lineage.get("round_group_id"))) is not None,
            "sequential image model provenance lineage is invalid",
        )
        for name in (
            "parent_bundle_sha256",
            "genesis_stage20_features_sha256",
            "genesis_split_sha256",
            "round_image_id",
            "round_image_sha256",
            "cumulative_al_feature_archive_sha256",
            "effective_train_groups_sha256",
            "frozen_test_groups_sha256",
        ):
            _sha256(lineage.get(name), f"sequential image provenance {name}")
        result.update(lineage)
    return result


def _validate_raw_feature(row: CanonicalActiveLearningTrainingRow) -> None:
    import numpy as np

    raw = row.raw_feature
    _sha256(row.source_sha256, "active-learning raw-feature source")
    _require(
        row.proposal_sha256 == row.proposal_id
        and _SHA256.fullmatch(row.proposal_id) is not None
        and _SHA256.fullmatch(row.image_id) is not None
        and _SHA256.fullmatch(row.image_sha256) is not None
        and _GROUP.fullmatch(row.group_id) is not None,
        "active-learning raw-feature immutable identity is invalid",
    )
    _require(
        isinstance(raw.proposal_index, int)
        and not isinstance(raw.proposal_index, bool)
        and raw.proposal_index >= 1
        and float(raw.scale) in CANONICAL_FEATURE_CROP_SCALES
        and 0.0 <= _strict_float(raw.predicted_iou, "active-learning predicted IoU") <= 1.0
        and 0.0 <= _strict_float(raw.stability_score, "active-learning stability score") <= 1.0
        and tuple(raw.values) == CANONICAL_RAW_FEATURE_ORDER
        and all(math.isfinite(float(raw.values[name])) for name in CANONICAL_RAW_FEATURE_ORDER),
        "active-learning raw feature schema or values changed",
    )
    embedding = np.asarray(raw.embedding, dtype=np.float32)
    _require(
        embedding.shape == (2048,) and np.all(np.isfinite(embedding)),
        "active-learning feature embedding is invalid",
    )


def _raw_feature_content_sha256(row: CanonicalActiveLearningTrainingRow) -> str:
    import numpy as np

    _validate_raw_feature(row)
    raw = row.raw_feature
    return compact_json_sha256(
        {
            "proposal_id": row.proposal_id,
            "proposal_sha256": row.proposal_sha256,
            "image_id": row.image_id,
            "image_sha256": row.image_sha256,
            "group_id": row.group_id,
            "proposal_index": raw.proposal_index,
            "scale": _format_float(raw.scale),
            "predicted_iou": _format_float(raw.predicted_iou),
            "stability_score": _format_float(raw.stability_score),
            "values": {
                name: _format_float(raw.values[name])
                for name in CANONICAL_RAW_FEATURE_ORDER
            },
            "embedding_sha256": hashlib.sha256(
                np.asarray(raw.embedding, dtype="<f4").tobytes(order="C")
            ).hexdigest(),
        }
    )


def _recompute_feature_state(
    previous: CanonicalFeatureState,
    training_rows: Sequence[CanonicalActiveLearningTrainingRow],
    latest: Mapping[str, CanonicalActiveLearningDecision],
    train_groups: frozenset[str],
    test_groups: frozenset[str],
) -> tuple[CanonicalFeatureState, tuple[CanonicalActiveLearningTrainingRow, ...]]:
    import numpy as np

    validate_canonical_feature_state(previous)
    _require(bool(training_rows), "active-learning accumulated feature rows are empty")
    by_proposal: dict[str, list[CanonicalActiveLearningTrainingRow]] = {}
    observed: set[tuple[str, float]] = set()
    ordered = tuple(sorted(training_rows, key=lambda row: (row.proposal_id, row.raw_feature.scale)))
    for row in ordered:
        _validate_raw_feature(row)
        _require(row.proposal_id in latest, "active-learning feature row has no effective decision")
        decision = latest[row.proposal_id]
        _require(
            row.proposal_sha256 == decision.proposal_sha256
            and row.image_id == decision.image_id
            and row.image_sha256 == decision.image_sha256
            and row.group_id == decision.group_id
            and row.group_id in train_groups | test_groups,
            "active-learning feature row group identity changed",
        )
        scale = float(row.raw_feature.scale)
        key = (row.proposal_id, scale)
        _require(key not in observed, "active-learning feature row is duplicated")
        observed.add(key)
        by_proposal.setdefault(row.proposal_id, []).append(row)
    _require(set(by_proposal) == set(latest), "active-learning feature rows do not close effective decisions")
    expected_scales = tuple(CANONICAL_FEATURE_CROP_SCALES)
    _require(
        all(
            tuple(item.raw_feature.scale for item in rows) == expected_scales
            and len({item.source_sha256 for item in rows}) == 1
            for rows in by_proposal.values()
        ),
        "active-learning feature rows do not have exact scale/source closure",
    )
    effective_train = tuple(
        row
        for row in ordered
        if row.group_id in train_groups and latest[row.proposal_id].review_weight > 0.0
    )
    _require(len(effective_train) >= 32, "active-learning accumulated training pool cannot support PCA32")
    positives = [
        np.asarray(row.raw_feature.embedding, dtype=np.float32)
        for row in effective_train
        if latest[row.proposal_id].label == 1
    ]
    _require(bool(positives), "active-learning accumulated training pool has no positive embedding")
    matrix = np.vstack(positives).astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    _require(np.all(norms > 0.0), "active-learning positive embedding has zero norm")
    normalized = matrix / norms
    prototype = normalized.mean(axis=0)
    norm = float(np.linalg.norm(prototype))
    _require(math.isfinite(norm) and norm > 0.0, "active-learning prototype mean has zero norm")
    prototype = np.ascontiguousarray(prototype / norm, dtype=np.float32)
    components = np.asarray(previous.pca_components, dtype=np.float32)
    mean = np.asarray(previous.pca_mean, dtype=np.float32)
    for value in (prototype, components, mean):
        value.setflags(write=False)
    state = CanonicalFeatureState(
        prototype=prototype,
        pca_components=components,
        pca_mean=mean,
        training_row_count=len(effective_train),
        positive_row_count=len(positives),
    )
    validate_canonical_feature_state(state)
    _require(
        np.asarray(state.pca_components, dtype=np.float32).tobytes(order="C")
        == np.asarray(previous.pca_components, dtype=np.float32).tobytes(order="C")
        and np.asarray(state.pca_mean, dtype=np.float32).tobytes(order="C")
        == np.asarray(previous.pca_mean, dtype=np.float32).tobytes(order="C"),
        "active-learning PCA state changed",
    )
    return state, ordered


def _feature_state_hashes(state: CanonicalFeatureState) -> dict[str, str]:
    payloads = serialize_canonical_feature_state(state)
    return {
        "prototype_sha256": hashlib.sha256(payloads.prototype_npy).hexdigest(),
        "pca_components_sha256": hashlib.sha256(payloads.pca_components_npy).hexdigest(),
        "pca_mean_sha256": hashlib.sha256(payloads.pca_mean_npy).hexdigest(),
    }


def _feature_archive_hash(
    rows: Sequence[CanonicalActiveLearningTrainingRow],
) -> str:
    ordered = sorted(rows, key=lambda row: (row.proposal_id, row.raw_feature.scale))
    return compact_json_sha256(
        [
            {
                "proposal_id": row.proposal_id,
                "scale": _format_float(row.raw_feature.scale),
                "source_sha256": row.source_sha256,
                "content_sha256": _raw_feature_content_sha256(row),
            }
            for row in ordered
        ]
    )


def _accumulated_feature_csv_row(
    row: CanonicalActiveLearningTrainingRow,
) -> dict[str, object]:
    """Render one training feature without depending on a Stage-60 tree."""

    _validate_raw_feature(row)
    raw = row.raw_feature
    return {
        "proposal_id": row.proposal_id,
        "proposal_sha256": row.proposal_sha256,
        "image_id": row.image_id,
        "image_sha256": row.image_sha256,
        "group_id": row.group_id,
        "source_sha256": row.source_sha256,
        "proposal_index": str(raw.proposal_index),
        "scale": _format_float(raw.scale),
        "predicted_iou": _format_float(raw.predicted_iou),
        "stability_score": _format_float(raw.stability_score),
        **{
            name: _format_float(raw.values[name])
            for name in CANONICAL_RAW_FEATURE_ORDER
        },
        "embedding_encoding": "base64-float32-little-endian-v1",
        "embedding_dimensions": "2048",
        "embedding_f32le_base64": encode_canonical_embedding(raw.embedding),
    }


def _validate_accumulated_feature_rows(
    rows: Sequence[CanonicalActiveLearningTrainingRow],
) -> tuple[CanonicalActiveLearningTrainingRow, ...]:
    ordered = tuple(
        sorted(rows, key=lambda item: (item.proposal_id, item.raw_feature.scale))
    )
    _require(bool(ordered), "accumulated active-learning feature archive is empty")
    _require(
        len(ordered) <= _MAX_TABLE_ROWS,
        "accumulated active-learning feature archive exceeds its row bound",
    )
    by_proposal: dict[str, list[CanonicalActiveLearningTrainingRow]] = {}
    observed: set[tuple[str, float]] = set()
    for row in ordered:
        _validate_raw_feature(row)
        key = (row.proposal_id, float(row.raw_feature.scale))
        _require(
            key not in observed,
            "accumulated active-learning feature archive contains a duplicate scale row",
        )
        observed.add(key)
        by_proposal.setdefault(row.proposal_id, []).append(row)
    expected_scales = tuple(CANONICAL_FEATURE_CROP_SCALES)
    _require(
        all(
            tuple(item.raw_feature.scale for item in proposal_rows)
            == expected_scales
            and len({item.source_sha256 for item in proposal_rows}) == 1
            and len(
                {
                    (
                        item.proposal_sha256,
                        item.image_id,
                        item.image_sha256,
                        item.group_id,
                        item.raw_feature.proposal_index,
                        item.raw_feature.predicted_iou,
                        item.raw_feature.stability_score,
                    )
                    for item in proposal_rows
                }
            )
            == 1
            for proposal_rows in by_proposal.values()
        ),
        "accumulated active-learning features do not have exact scale/source closure",
    )
    return ordered


def _accumulated_feature_receipt(
    path: Path,
    rows: Sequence[CanonicalActiveLearningTrainingRow],
) -> dict[str, object]:
    record = _input_record(
        path,
        "accumulated active-learning raw features",
        max_bytes=_MAX_TABLE_BYTES,
    )
    proposals = {row.proposal_id for row in rows}
    return {
        "schema": _ACCUMULATED_FEATURE_SCHEMA,
        "status": "PASS",
        **record,
        "columns_sha256": compact_json_sha256(
            list(CANONICAL_AL_ACCUMULATED_FEATURE_COLUMNS)
        ),
        "logical_sha256": _feature_archive_hash(rows),
        "proposal_count": len(proposals),
        "row_count": len(rows),
        "source_count": len({row.source_sha256 for row in rows}),
        "groups_sha256": compact_json_sha256(
            sorted({row.group_id for row in rows})
        ),
    }


def write_canonical_accumulated_feature_archive(
    rows: Sequence[CanonicalActiveLearningTrainingRow],
    output: Path,
) -> dict[str, object]:
    """Persist the cumulative reviewed AL features used by later image rounds."""

    ordered = _validate_accumulated_feature_rows(rows)
    absolute = output.absolute()
    _require(
        portable_basename(absolute.name, "accumulated feature archive basename")
        == "accumulated_al_features.csv",
        "accumulated feature archive basename changed",
    )
    _require(
        not absolute.exists() and not absolute.is_symlink(),
        "accumulated feature archive output already exists",
    )
    payload = _csv_bytes(
        CANONICAL_AL_ACCUMULATED_FEATURE_COLUMNS,
        [_accumulated_feature_csv_row(row) for row in ordered],
    )
    write_new_bytes(absolute, payload)
    parsed, receipt = read_canonical_accumulated_feature_archive(absolute)
    _require(
        _feature_archive_hash(parsed) == _feature_archive_hash(ordered),
        "accumulated feature archive changed while writing",
    )
    return receipt


def read_canonical_accumulated_feature_archive(
    path: Path,
) -> tuple[tuple[CanonicalActiveLearningTrainingRow, ...], dict[str, object]]:
    """Read and canonically verify a persisted cumulative AL feature archive."""

    raw_rows, _record = _read_csv(
        path,
        CANONICAL_AL_ACCUMULATED_FEATURE_COLUMNS,
        "accumulated active-learning raw features",
    )
    parsed: list[CanonicalActiveLearningTrainingRow] = []
    for raw_row in raw_rows:
        try:
            raw = CanonicalRawFeature(
                proposal_index=int(raw_row["proposal_index"]),
                scale=float(raw_row["scale"]),
                predicted_iou=float(raw_row["predicted_iou"]),
                stability_score=float(raw_row["stability_score"]),
                values={
                    name: float(raw_row[name])
                    for name in CANONICAL_RAW_FEATURE_ORDER
                },
                embedding=decode_canonical_embedding(
                    raw_row["embedding_f32le_base64"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise PublicIOError(
                "accumulated active-learning feature archive is malformed"
            ) from exc
        row = CanonicalActiveLearningTrainingRow(
            proposal_id=raw_row["proposal_id"],
            proposal_sha256=raw_row["proposal_sha256"],
            image_id=raw_row["image_id"],
            image_sha256=raw_row["image_sha256"],
            group_id=raw_row["group_id"],
            raw_feature=raw,
            source_sha256=raw_row["source_sha256"],
        )
        _require(
            raw_row["embedding_encoding"]
            == "base64-float32-little-endian-v1"
            and raw_row["embedding_dimensions"] == "2048"
            and _accumulated_feature_csv_row(row)
            == dict(raw_row),
            "accumulated active-learning feature representation is noncanonical",
        )
        parsed.append(row)
    ordered = _validate_accumulated_feature_rows(parsed)
    _require(
        tuple(parsed) == ordered,
        "accumulated active-learning feature rows are not canonically sorted",
    )
    return ordered, _accumulated_feature_receipt(path, ordered)


def _finalized_pool_rows(
    pool_rows: Sequence[CanonicalActiveLearningPoolRow],
    pool_feature_rows: Sequence[CanonicalActiveLearningTrainingRow],
    training_rows: Sequence[CanonicalActiveLearningTrainingRow],
    state: CanonicalFeatureState,
) -> tuple[tuple[Mapping[str, float], ...], str]:
    from .features import finalize_canonical_feature

    pool_by_id = {row.proposal_id: row for row in pool_rows}
    ordered = tuple(
        sorted(pool_feature_rows, key=lambda row: (row.proposal_id, row.raw_feature.scale))
    )
    by_proposal: dict[str, list[CanonicalActiveLearningTrainingRow]] = {}
    observed: set[tuple[str, float]] = set()
    for row in ordered:
        _validate_raw_feature(row)
        _require(row.proposal_id in pool_by_id, "pool feature archive references an unknown proposal")
        identity = pool_by_id[row.proposal_id]
        _require(
            (
                row.proposal_id,
                row.proposal_sha256,
                row.image_id,
                row.image_sha256,
                row.group_id,
            )
            == (
                identity.proposal_id,
                identity.proposal_sha256,
                identity.image_id,
                identity.image_sha256,
                identity.group_id,
            ),
            "pool feature archive immutable identity changed",
        )
        key = (row.proposal_id, float(row.raw_feature.scale))
        _require(key not in observed, "pool feature archive contains a duplicate scale row")
        observed.add(key)
        by_proposal.setdefault(row.proposal_id, []).append(row)
    _require(set(by_proposal) == set(pool_by_id), "pool feature archive does not close the declared pool")
    _require(
        all(
            tuple(item.raw_feature.scale for item in values)
            == tuple(CANONICAL_FEATURE_CROP_SCALES)
            and len({item.source_sha256 for item in values}) == 1
            for values in by_proposal.values()
        ),
        "pool feature archive scale or source closure changed",
    )
    accumulated_by_key = {
        (row.proposal_id, float(row.raw_feature.scale)): row for row in training_rows
    }
    for row in ordered:
        key = (row.proposal_id, float(row.raw_feature.scale))
        accumulated = accumulated_by_key.get(key)
        if accumulated is not None:
            _require(
                _raw_feature_content_sha256(accumulated)
                == _raw_feature_content_sha256(row),
                "pool and accumulated archives disagree for the same proposal scale",
            )
    scale_one = [
        next(
            item
            for item in by_proposal[row.proposal_id]
            if float(item.raw_feature.scale) == 1.0
        )
        for row in pool_rows
    ]
    finalized = tuple(
        finalize_canonical_feature(item.raw_feature, state) for item in scale_one
    )
    _require(
        all(tuple(row) == CANONICAL_FEATURE_ORDER for row in finalized),
        "active-learning rescore feature schema changed",
    )
    return finalized, _feature_archive_hash(ordered)


def _predict_rescored_probabilities(
    bundle: Any,
    pool_rows: tuple[CanonicalActiveLearningPoolRow, ...],
    finalized_rows: tuple[Mapping[str, float], ...],
    *,
    device: str,
) -> tuple[float, ...]:
    from .serialization import (
        load_portable_predictor,
        predict_portable_probabilities,
    )

    _canonical_profile_device(getattr(bundle, "profile", None), device)
    predictor = load_portable_predictor(
        bundle.classifier,
        bundle.imputer,
        device=device,
    )
    raw = predict_portable_probabilities(predictor, finalized_rows)
    try:
        probabilities = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise PublicIOError("active-learning predictor returned nonnumeric probabilities") from exc
    _require(
        len(probabilities) == len(pool_rows)
        and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities),
        "active-learning predictor result does not close the declared pool",
    )
    return probabilities


def _verify_bundle_feature_state(bundle: Any, state: CanonicalFeatureState) -> None:
    import numpy as np

    _require(
        getattr(bundle, "prototype", None) is not None
        and getattr(bundle, "pca_components", None) is not None
        and getattr(bundle, "pca_mean", None) is not None,
        "retrained bundle omits canonical feature state",
    )
    for role, observed, expected in (
        ("prototype", bundle.prototype, state.prototype),
        ("PCA components", bundle.pca_components, state.pca_components),
        ("PCA mean", bundle.pca_mean, state.pca_mean),
    ):
        observed_array = np.asarray(observed, dtype=np.float32)
        expected_array = np.asarray(expected, dtype=np.float32)
        _require(
            observed_array.shape == expected_array.shape
            and observed_array.tobytes(order="C") == expected_array.tobytes(order="C"),
            f"retrained bundle {role} differs from the AL retrain request",
        )


def _verify_bundle_training_provenance(
    bundle: Any,
    expected: Mapping[str, object],
    *,
    profile: str,
) -> None:
    provenance = getattr(bundle, "provenance", None)
    _require(
        isinstance(provenance, Mapping)
        and set(provenance) == {
            "schema", "profile", "feature_order_sha256", "threshold_method",
            "details",
        }
        and provenance.get("schema") == "compag-curation-model-provenance/v2"
        and provenance.get("profile") == profile
        and provenance.get("feature_order_sha256") == CANONICAL_FEATURE_ORDER_SHA256
        and provenance.get("threshold_method") == "FIXED_CANONICAL_METHOD"
        and provenance.get("details") == dict(expected),
        "retrained bundle provenance does not bind the AL retrain request",
    )


def resume_canonical_active_learning_round(
    selection_root: Path,
    pool_root: Path,
    source_bundle_root: Path,
    frozen_split_path: Path,
    decision_log_path: Path,
    reviewed_path: Path,
    previous_feature_state: CanonicalFeatureState,
    training_rows: Sequence[CanonicalActiveLearningTrainingRow],
    pool_feature_rows: Sequence[CanonicalActiveLearningTrainingRow],
    output: Path,
    *,
    completed_at_utc: str,
    retrain: RetrainFunction,
    initial_completion_path: Path | None = None,
    prior_round_completion_path: Path | None = None,
    image_scoped: bool = False,
) -> dict[str, object]:
    """Complete one AL round only after verified retraining and rescoring."""

    timestamp = _timestamp(completed_at_utc)
    (
        selection, selection_record, pool_rows, source_bundle, train_groups,
        test_groups, prior_decisions, raw_log_rows, prior_log_record,
    ) = _selection_manifest(
        selection_root,
        pool_root,
        source_bundle_root,
        frozen_split_path,
        decision_log_path,
        initial_completion_path,
        prior_round_completion_path,
        image_scoped=image_scoped,
    )
    _require(
        source_bundle.profile in GPU_EXECUTION_PROFILES,
        "active-learning round execution requires a canonical GPU bundle, an efficient Lite GPU bundle, or the full-image GPU bundle",
    )
    reviewed, review_result = validate_review_table(
        selection_root.resolve(strict=True) / "review_request.csv",
        reviewed_path,
    )
    _require(
        review_result["pending_sha256"] == selection["review_request"]["sha256"]
        and review_result["review_contract"] == "CANONICAL_ACTION_WEIGHTED_V2",
        "active-learning reviewed batch binding changed",
    )
    _require(
        timestamp >= prior_decisions[-1].decision_timestamp_utc,
        "active-learning completion timestamp precedes the immutable log",
    )
    _require(
        timestamp >= selection["created_at_utc"],
        "active-learning completion timestamp precedes its selection",
    )
    _verify_bundle_feature_state(source_bundle, previous_feature_state)
    profile = str(source_bundle.profile)
    device = _canonical_device_for_profile(profile)
    _require(
        selection["profile"] == profile,
        "active-learning selection profile differs from its source bundle",
    )
    number = int(selection["round_number"])
    pool_raw_archive_sha256 = str(
        selection["chain"]["source_pool_source_sha256"]
    )
    pool_feature_archive_receipt = str(
        selection["chain"]["source_feature_archive_sha256"]
    )
    _require(
        {row.source_sha256 for row in pool_feature_rows}
        == {pool_raw_archive_sha256}
        and _feature_archive_hash(pool_feature_rows)
        == pool_feature_archive_receipt,
        "pool features differ from the bound physical or logical archive",
    )
    appended_rows, all_decisions = _append_decisions(
        raw_log_rows,
        reviewed,
        round_number=number,
        timestamp_utc=timestamp,
        bundle_sha256=source_bundle.bundle_sha256,
        predictions_sha256=selection["chain"]["source_predictions_sha256"],
        round_manifest_sha256=selection_record["sha256"],
        reviewed_sha256=review_result["reviewed_sha256"],
    )
    latest = _latest_decisions(all_decisions)
    effective_train_groups = train_groups
    if image_scoped:
        round_decisions = [
            decision
            for decision in all_decisions
            if decision.phase == _ROUND_PHASE
        ]
        round_groups = {decision.group_id for decision in round_decisions}
        groups_by_round = {
            round_value: {
                decision.group_id
                for decision in round_decisions
                if decision.round_number == round_value
            }
            for round_value in range(1, number + 1)
        }
        images_by_round = {
            round_value: {
                (decision.image_id, decision.image_sha256)
                for decision in round_decisions
                if decision.round_number == round_value
            }
            for round_value in range(1, number + 1)
        }
        _require(
            len(round_groups) == number
            and all(len(groups_by_round[index]) == 1 for index in groups_by_round)
            and all(len(images_by_round[index]) == 1 for index in images_by_round)
            and not round_groups & set(train_groups)
            and not round_groups & set(test_groups),
            "sequential image decisions do not contain one new image group per round",
        )
        effective_train_groups = frozenset(set(train_groups) | round_groups)
    state, ordered_training_rows = _recompute_feature_state(
        previous_feature_state,
        training_rows,
        latest,
        effective_train_groups,
        test_groups,
    )
    finalized_pool_rows, pool_feature_archive_sha256 = _finalized_pool_rows(
        pool_rows,
        pool_feature_rows,
        ordered_training_rows,
        state,
    )
    output = _new_directory(output)
    accumulated_feature_receipt: dict[str, object] | None = None
    accumulated_al_rows: tuple[CanonicalActiveLearningTrainingRow, ...] = ()
    if image_scoped:
        accumulated_al_rows = tuple(
            row
            for row in ordered_training_rows
            if latest[row.proposal_id].phase == _ROUND_PHASE
        )
        accumulated_feature_receipt = write_canonical_accumulated_feature_archive(
            accumulated_al_rows,
            output / "accumulated_al_features.csv",
        )
    decision_payload = _csv_bytes(CANONICAL_AL_DECISION_LOG_COLUMNS, appended_rows)
    write_new_bytes(output / "decision_log.csv", decision_payload)
    decision_record = _input_record(output / "decision_log.csv", "active-learning decision log", max_bytes=_MAX_TABLE_BYTES)
    state_hashes = _feature_state_hashes(state)
    training_hash = _training_row_hash(ordered_training_rows, latest)
    split_record = _input_record(frozen_split_path, "frozen group split", max_bytes=16 * 1024 * 1024)
    lineage_provenance: dict[str, object] | None = None
    if image_scoped:
        image = selection["image"]
        lineage_provenance = {
            "active_learning_workflow": _IMAGE_WORKFLOW,
            "model_label": f"project-r{number}",
            "active_learning_round_number": number,
            "parent_bundle_sha256": source_bundle.bundle_sha256,
            "genesis_stage20_features_sha256": str(
                source_bundle.provenance["details"].get(
                    "genesis_stage20_features_sha256",
                    source_bundle.provenance["details"]["raw_features_sha256"],
                )
            ),
            "genesis_split_sha256": split_record["sha256"],
            "round_image_id": image["image_id"],
            "round_image_sha256": image["image_sha256"],
            "round_group_id": image["group_id"],
            "cumulative_al_feature_archive_sha256": (
                accumulated_feature_receipt["logical_sha256"]
                if accumulated_feature_receipt is not None
                else ""
            ),
            "effective_train_groups_sha256": compact_json_sha256(
                sorted(effective_train_groups)
            ),
            "frozen_test_groups_sha256": compact_json_sha256(
                sorted(test_groups)
            ),
        }
    expected_provenance = _expected_bundle_provenance(
        ordered_training_rows,
        latest,
        effective_train_groups,
        decision_log_sha256=str(decision_record["sha256"]),
        split_sha256=str(split_record["sha256"]),
        training_rows_sha256=training_hash,
        state=state,
        profile=profile,
        device=device,
        lineage=lineage_provenance,
    )
    request = CanonicalActiveLearningRetrainRequest(
        round_number=number,
        feature_state=state,
        effective_decisions=tuple(sorted(latest.values(), key=lambda value: value.proposal_id)),
        training_rows=ordered_training_rows,
        train_groups=effective_train_groups,
        test_groups=test_groups,
        decision_log_sha256=str(decision_record["sha256"]),
        training_rows_sha256=training_hash,
        source_bundle_sha256=source_bundle.bundle_sha256,
        frozen_split_sha256=str(split_record["sha256"]),
        prototype_sha256=state_hashes["prototype_sha256"],
        pca_components_sha256=state_hashes["pca_components_sha256"],
        pca_mean_sha256=state_hashes["pca_mean_sha256"],
        expected_bundle_provenance=MappingProxyType(expected_provenance),
        profile=profile,
        device=device,
    )
    request_record = {
        "schema": _RETRAIN_REQUEST_SCHEMA_V2,
        "profile": profile,
        "device": device,
        "round_number": number,
        "source_bundle_sha256": request.source_bundle_sha256,
        "frozen_split_sha256": request.frozen_split_sha256,
        "decision_log_sha256": request.decision_log_sha256,
        "training_rows_sha256": request.training_rows_sha256,
        "pool_feature_archive_sha256": pool_feature_archive_sha256,
        "effective_proposals": len(request.effective_decisions),
        "feature_rows": len(request.training_rows),
        "train_groups_sha256": compact_json_sha256(sorted(request.train_groups)),
        "test_groups_sha256": compact_json_sha256(sorted(request.test_groups)),
        "feature_state_training_rows": state.training_row_count,
        "feature_state_positive_rows": state.positive_row_count,
        **state_hashes,
        "pca_policy": "FROZEN_FROM_PARENT_BUNDLE",
        "prototype_policy": "RECOMPUTED_ACCUMULATED_OUTER_TRAIN_POSITIVES",
    }
    _public_evidence(request_record)
    write_new_json(output / "retrain_request.json", request_record)
    parsed_retrain_request, retrain_request_file_record = _retrain_request_manifest(
        output / "retrain_request.json",
    )
    _require(
        parsed_retrain_request == request_record,
        "active-learning retrain request changed while writing",
    )

    bundle_output = output / "model_bundle"
    _require(not bundle_output.exists() and not bundle_output.is_symlink(), "retrain output already exists")
    retrain(request, bundle_output)
    _require(bundle_output.exists(), "active-learning retrain callback did not create a bundle")
    new_bundle = _verify_bundle(bundle_output)
    _require(
        new_bundle.profile == profile,
        "retrained bundle profile differs from its source bundle",
    )
    _verify_bundle_feature_state(new_bundle, state)
    _verify_bundle_training_provenance(
        new_bundle,
        expected_provenance,
        profile=profile,
    )

    rescored_output = output / "rescored_pool"
    _require(not rescored_output.exists() and not rescored_output.is_symlink(), "rescore output already exists")
    probabilities = _predict_rescored_probabilities(
        new_bundle,
        tuple(pool_rows),
        finalized_pool_rows,
        device=device,
    )
    rescored_candidates = tuple(
        CanonicalActiveLearningPoolRow(
            proposal_id=row.proposal_id,
            proposal_sha256=row.proposal_sha256,
            image_id=row.image_id,
            image_sha256=row.image_sha256,
            group_id=row.group_id,
            scale=1.0,
            xgb_p=probability,
        )
        for row, probability in zip(pool_rows, probabilities, strict=True)
    )
    write_canonical_active_learning_pool(
        rescored_candidates,
        rescored_output,
        bundle_sha256=new_bundle.bundle_sha256,
        source_sha256=pool_raw_archive_sha256,
        feature_archive_sha256=pool_feature_archive_sha256,
        profile=profile,
    )
    rescored_rows, rescored_result, _rescored_records = _read_pool(
        rescored_output,
        require_scores=True,
    )
    _require(
        rescored_result["profile"] == profile
        and rescored_result["bundle_sha256"] == new_bundle.bundle_sha256,
        "rescored pool does not bind the retrained bundle",
    )
    source_identities = [
        (row.proposal_id, row.proposal_sha256, row.image_id, row.image_sha256, row.group_id, row.scale)
        for row in pool_rows
    ]
    rescored_identities = [
        (row.proposal_id, row.proposal_sha256, row.image_id, row.image_sha256, row.group_id, row.scale)
        for row in rescored_rows
    ]
    _require(
        rescored_identities == source_identities,
        "rescoring did not close the declared immutable candidate pool",
    )
    new_bundle_after = _verify_bundle(bundle_output)
    _require(
        new_bundle_after.bundle_sha256 == new_bundle.bundle_sha256
        and tuple(getattr(new_bundle_after, "tree_identity", ()))
        == tuple(getattr(new_bundle, "tree_identity", ())),
        "retrained bundle changed during rescoring",
    )
    _verify_bundle_feature_state(new_bundle_after, state)
    _verify_bundle_training_provenance(
        new_bundle_after,
        expected_provenance,
        profile=profile,
    )
    source_bundle_after = _verify_bundle(source_bundle_root)
    _require(
        source_bundle_after.profile == profile
        and source_bundle_after.bundle_sha256 == source_bundle.bundle_sha256
        and tuple(getattr(source_bundle_after, "tree_identity", ()))
        == tuple(getattr(source_bundle, "tree_identity", ())),
        "source bundle changed during the active-learning round",
    )
    (
        _selection_after,
        selection_record_after,
        _pool_rows_after,
        _source_bundle_verified_after,
        _train_groups_after,
        _test_groups_after,
        _prior_decisions_after,
        _raw_log_rows_after,
        prior_log_record_after,
    ) = _selection_manifest(
        selection_root,
        pool_root,
        source_bundle_root,
        frozen_split_path,
        decision_log_path,
        initial_completion_path,
        prior_round_completion_path,
        image_scoped=image_scoped,
    )
    reviewed_after = _input_record(
        reviewed_path,
        "active-learning reviewed batch",
        max_bytes=_MAX_TABLE_BYTES,
    )
    decision_record_after = _input_record(
        output / "decision_log.csv",
        "active-learning decision log",
        max_bytes=_MAX_TABLE_BYTES,
    )
    parsed_retrain_request_after, retrain_request_after = _retrain_request_manifest(
        output / "retrain_request.json",
    )
    _require(
        selection_record_after == selection_record
        and prior_log_record_after == prior_log_record
        and reviewed_after["sha256"] == review_result["reviewed_sha256"]
        and decision_record_after == decision_record
        and parsed_retrain_request_after == request_record
        and retrain_request_after == retrain_request_file_record
        and _feature_state_hashes(state) == state_hashes
        and _training_row_hash(ordered_training_rows, latest) == training_hash
        and _feature_archive_hash(pool_feature_rows) == pool_feature_archive_sha256
        and dict(request.expected_bundle_provenance) == expected_provenance,
        "active-learning inputs or retrain request changed during callbacks",
    )
    model_lineage_record: dict[str, object] | None = None
    if image_scoped:
        _require(
            accumulated_feature_receipt is not None
            and lineage_provenance is not None,
            "sequential image feature or lineage receipt is missing",
        )
        lineage = {
            "schema": _MODEL_LINEAGE_SCHEMA,
            "status": "PASS",
            "workflow": _IMAGE_WORKFLOW,
            "model_label": f"project-r{number}",
            "round_number": number,
            "training_mode": "FRESH_FULL_XGBOOST_FROM_CUMULATIVE_REVIEWED_ROWS",
            "parent_bundle_sha256": source_bundle.bundle_sha256,
            "new_bundle_sha256": new_bundle.bundle_sha256,
            "genesis_initial_completion_sha256": selection["chain"][
                "genesis_initial_completion_sha256"
            ],
            "genesis_stage20_features_sha256": lineage_provenance[
                "genesis_stage20_features_sha256"
            ],
            "genesis_split_sha256": split_record["sha256"],
            "round_image_id": selection["image"]["image_id"],
            "round_image_sha256": selection["image"]["image_sha256"],
            "round_group_id": selection["image"]["group_id"],
            "decision_log_sha256": decision_record["sha256"],
            "training_rows_sha256": training_hash,
            "cumulative_al_feature_archive_sha256": accumulated_feature_receipt[
                "logical_sha256"
            ],
            "effective_train_groups_sha256": compact_json_sha256(
                sorted(effective_train_groups)
            ),
            "frozen_test_groups_sha256": compact_json_sha256(
                sorted(test_groups)
            ),
        }
        _public_evidence(lineage)
        write_new_json(output / "model_lineage.json", lineage)
        model_lineage_record = _input_record(
            output / "model_lineage.json",
            "sequential image model lineage",
            max_bytes=16 * 1024 * 1024,
        )
    _directory_closure(
        output,
        {
            "decision_log.csv",
            "retrain_request.json",
            *(
                {"accumulated_al_features.csv", "model_lineage.json"}
                if image_scoped
                else set()
            ),
        },
        {"model_bundle", "rescored_pool"},
    )
    completion = {
        "schema": (
            _IMAGE_ROUND_COMPLETION_SCHEMA
            if image_scoped
            else _ROUND_COMPLETION_SCHEMA
        ),
        "status": "PASS",
        "profile": profile,
        "kind": _ROUND_PHASE,
        "round_number": number,
        "completed_at_utc": timestamp,
        "policy": _policy(),
        "chain": {
            "genesis_initial_completion_sha256": selection["chain"]["genesis_initial_completion_sha256"],
            "prior_round_completion_sha256": selection["chain"]["prior_round_completion_sha256"],
            "selection_manifest_sha256": selection_record["sha256"],
            "prior_review_log_sha256": prior_log_record["sha256"],
            "reviewed_batch_sha256": review_result["reviewed_sha256"],
            "source_bundle_sha256": source_bundle.bundle_sha256,
            "source_predictions_sha256": selection["chain"]["source_predictions_sha256"],
            "source_pool_source_sha256": selection["chain"]["source_pool_source_sha256"],
            "source_feature_archive_sha256": selection["chain"]["source_feature_archive_sha256"],
            "frozen_split_sha256": split_record["sha256"],
        },
        "counts": {
            "reviewed_batch_rows": len(reviewed),
            "decision_log_rows": len(all_decisions),
            "effective_proposals": len(latest),
            "feature_rows": len(ordered_training_rows),
            "rescored_prediction_rows": len(rescored_rows),
        },
        "feature_state": {
            **state_hashes,
            "training_rows": state.training_row_count,
            "positive_rows": state.positive_row_count,
            "pca_policy": "FROZEN_FROM_PARENT_BUNDLE",
            "prototype_policy": "RECOMPUTED_ACCUMULATED_OUTER_TRAIN_POSITIVES",
        },
        "outputs": {
            "decision_log": decision_record,
            "training_rows_sha256": training_hash,
            "pool_feature_archive_sha256": pool_feature_archive_sha256,
            "new_bundle_sha256": new_bundle.bundle_sha256,
            "rescored_predictions_sha256": rescored_result["predictions_sha256"],
            "rescore_method": "VERIFIED_BUNDLE_XGB_PREDICT_PORTABLE",
        },
        "paper_result_reproduction": "NOT_CLAIMED",
    }
    if image_scoped:
        completion["workflow"] = _IMAGE_WORKFLOW
        completion["image"] = dict(selection["image"])
        completion["chain"]["genesis_stage20_features_sha256"] = selection[
            "chain"
        ]["genesis_stage20_features_sha256"]
        completion["chain"]["genesis_split_sha256"] = split_record["sha256"]
        completion["outputs"]["accumulated_al_features"] = (
            accumulated_feature_receipt
        )
        completion["outputs"]["model_lineage"] = model_lineage_record
    _public_evidence(completion)
    write_new_json(output / "round_completion.json", completion)
    _directory_closure(
        output,
        {
            "decision_log.csv",
            "retrain_request.json",
            "round_completion.json",
            *(
                {"accumulated_al_features.csv", "model_lineage.json"}
                if image_scoped
                else set()
            ),
        },
        {"model_bundle", "rescored_pool"},
    )
    return completion


def _transfer_scientific_lineage() -> dict[str, str]:
    return {
        "source_corpus_role": _TRANSFER_CORPUS_ROLE,
        "reproduction_claim": _TRANSFER_REPRODUCTION_CLAIM,
        "feature_state_policy": _TRANSFER_FEATURE_STATE_POLICY,
    }


def _transfer_decision_csv_row(
    decision: ReviewRow,
    *,
    sequence: int,
    round_number: int,
    timestamp_utc: str,
    scorer: CanonicalTransferScorerReference,
    round_manifest_sha256: str,
    reviewed_sha256: str,
) -> dict[str, object]:
    _require(
        decision.review_action in REVIEW_ACTION_WEIGHTS
        and decision.review_weight == REVIEW_ACTION_WEIGHTS[decision.review_action],
        "transfer decision action/weight changed",
    )
    return {
        "sequence": str(sequence),
        "phase": _ROUND_PHASE,
        "round_number": str(round_number),
        "decision_timestamp_utc": timestamp_utc,
        "timestamp_kind": "BATCH_IMPORT_UTC",
        "proposal_id": decision.proposal_id,
        "proposal_sha256": decision.proposal_sha256,
        "image_id": decision.image_id,
        "image_sha256": decision.image_sha256,
        "group_id": decision.group_id,
        "label": str(decision.label),
        "review_action": decision.review_action,
        "review_weight": format(float(decision.review_weight), ".1f"),
        "decision_policy": "XGB_ONLY_DISTANCE_TO_FIXED_THRESHOLD",
        "threshold": _format_float(CANONICAL_DECISION_THRESHOLD),
        "margin": _format_float(CANONICAL_AL_MARGIN),
        "top_k": str(CANONICAL_AL_TOP_K),
        "source_scorer_kind": scorer.kind,
        "source_scorer_sha256": scorer.scorer_sha256,
        "source_scores_sha256": scorer.scores_sha256,
        "source_round_manifest_sha256": round_manifest_sha256,
        "reviewed_batch_sha256": reviewed_sha256,
    }


def _parse_transfer_decision_rows(
    rows: Sequence[Mapping[str, str]],
) -> tuple[CanonicalActiveLearningDecision, ...]:
    parsed: list[CanonicalActiveLearningDecision] = []
    prior_round = 0
    prior_timestamp = ""
    seen_proposals: set[str] = set()
    groups_by_round: dict[int, set[str]] = {}
    images_by_round: dict[int, set[tuple[str, str]]] = {}
    receipts: dict[int, tuple[str, ...]] = {}
    for index, row in enumerate(rows, start=1):
        try:
            sequence = int(row["sequence"])
            number = int(row["round_number"])
            label = int(row["label"])
            weight = float(row["review_weight"])
        except (TypeError, ValueError) as exc:
            raise PublicIOError("transfer decision log has a nonnumeric field") from exc
        _require(
            sequence == index
            and row["sequence"] == str(sequence)
            and number >= 1
            and row["round_number"] == str(number)
            and number in {prior_round, prior_round + 1}
            and row["phase"] == _ROUND_PHASE,
            "transfer decision sequence or round is invalid",
        )
        prior_round = number
        timestamp = _timestamp(row["decision_timestamp_utc"])
        _require(
            timestamp >= prior_timestamp
            and row["timestamp_kind"] == "BATCH_IMPORT_UTC",
            "transfer decision timestamps are not monotone",
        )
        prior_timestamp = timestamp
        identity = _pool_identity(row, index)
        _require(
            identity[0] not in seen_proposals,
            "transfer decisions cannot revise an earlier reviewed proposal",
        )
        seen_proposals.add(identity[0])
        action = row["review_action"]
        _require(
            label in {0, 1}
            and action in REVIEW_ACTION_WEIGHTS
            and math.isfinite(weight)
            and row["label"] == str(label)
            and row["review_weight"] == format(REVIEW_ACTION_WEIGHTS[action], ".1f")
            and weight == REVIEW_ACTION_WEIGHTS[action]
            and row["decision_policy"]
            == "XGB_ONLY_DISTANCE_TO_FIXED_THRESHOLD"
            and row["threshold"] == _format_float(CANONICAL_DECISION_THRESHOLD)
            and row["margin"] == _format_float(CANONICAL_AL_MARGIN)
            and row["top_k"] == str(CANONICAL_AL_TOP_K)
            and row["source_scorer_kind"]
            in {_TRANSFER_R92_SCORER_KIND, _TRANSFER_PROJECT_SCORER_KIND},
            "transfer decision policy or action is invalid",
        )
        if number == 1:
            _require(
                row["source_scorer_kind"] == _TRANSFER_R92_SCORER_KIND,
                "the first transfer round was not scored by published r92",
            )
        else:
            _require(
                row["source_scorer_kind"] == _TRANSFER_PROJECT_SCORER_KIND,
                "published r92 may score only the first transfer round",
            )
        for name in (
            "source_scorer_sha256",
            "source_scores_sha256",
            "source_round_manifest_sha256",
            "reviewed_batch_sha256",
        ):
            _sha256(row[name], f"transfer decision {name}")
        receipt = tuple(
            row[name]
            for name in (
                "decision_timestamp_utc",
                "source_scorer_kind",
                "source_scorer_sha256",
                "source_scores_sha256",
                "source_round_manifest_sha256",
                "reviewed_batch_sha256",
            )
        )
        _require(
            number not in receipts or receipts[number] == receipt,
            "transfer scorer receipt changed within one round",
        )
        receipts[number] = receipt
        groups_by_round.setdefault(number, set()).add(identity[4])
        images_by_round.setdefault(number, set()).add((identity[2], identity[3]))
        parsed.append(
            CanonicalActiveLearningDecision(
                sequence=sequence,
                phase=_ROUND_PHASE,
                round_number=number,
                decision_timestamp_utc=timestamp,
                proposal_id=identity[0],
                proposal_sha256=identity[1],
                image_id=identity[2],
                image_sha256=identity[3],
                group_id=identity[4],
                label=label,
                review_action=action,
                review_weight=weight,
            )
        )
    _require(bool(parsed), "transfer decision log is empty")
    _require(
        set(groups_by_round) == set(range(1, prior_round + 1))
        and all(len(value) == 1 for value in groups_by_round.values())
        and all(len(value) == 1 for value in images_by_round.values())
        and len({next(iter(value)) for value in groups_by_round.values()})
        == prior_round
        and len({next(iter(value))[1] for value in images_by_round.values()})
        == prior_round,
        "transfer decision log does not contain exactly one fresh image group per round",
    )
    return tuple(parsed)


def _read_transfer_decision_log(
    path: Path,
) -> tuple[
    tuple[CanonicalActiveLearningDecision, ...],
    list[dict[str, str]],
    dict[str, object],
]:
    rows, record = _read_csv(
        path,
        CANONICAL_TRANSFER_DECISION_LOG_COLUMNS,
        "transfer active-learning decision log",
    )
    return _parse_transfer_decision_rows(rows), rows, record


def _validate_transfer_al_rows(
    rows: Sequence[CanonicalActiveLearningTrainingRow],
    decisions: Sequence[CanonicalActiveLearningDecision],
    baseline: CanonicalTransferBaselineReference,
) -> tuple[CanonicalActiveLearningTrainingRow, ...]:
    ordered = _validate_accumulated_feature_rows(rows)
    latest = _latest_decisions(decisions)
    by_proposal = {row.proposal_id for row in ordered}
    _require(
        by_proposal == set(latest),
        "cumulative transfer features do not close reviewed round decisions",
    )
    for row in ordered:
        decision = latest[row.proposal_id]
        _require(
            row.proposal_sha256 == decision.proposal_sha256
            and row.image_id == decision.image_id
            and row.image_sha256 == decision.image_sha256
            and row.group_id == decision.group_id
            and row.group_id not in baseline.source_groups
            and row.group_id not in baseline.test_groups,
            "cumulative transfer feature identity leaks into the baseline",
        )
    round_groups = {
        number: {item.group_id for item in decisions if item.round_number == number}
        for number in range(1, max(item.round_number for item in decisions) + 1)
    }
    _require(
        all(len(value) == 1 for value in round_groups.values()),
        "cumulative transfer features contain more than one group per round",
    )
    return ordered


def _transfer_expected_bundle_provenance(
    *,
    number: int,
    baseline: CanonicalTransferBaselineReference,
    scorer: CanonicalTransferScorerReference,
    decision_log_sha256: str,
    cumulative_al_rows_sha256: str,
    selection_manifest_sha256: str,
    state: CanonicalFeatureState,
    decisions: Sequence[CanonicalActiveLearningDecision],
    cumulative_archive_sha256: str,
    profile: str,
    device: str,
) -> dict[str, object]:
    validate_canonical_feature_state(state)
    _canonical_profile_device(profile, device)
    for value, role in (
        (decision_log_sha256, "transfer provenance decision log"),
        (cumulative_al_rows_sha256, "transfer provenance cumulative rows"),
        (selection_manifest_sha256, "transfer provenance selection"),
    ):
        _sha256(value, role)
    current = [item for item in decisions if item.round_number == number]
    _require(bool(current), "transfer provenance has no current-round decisions")
    image_identities = {
        (item.image_id, item.image_sha256, item.group_id) for item in current
    }
    _require(len(image_identities) == 1, "transfer provenance is not one image")
    image_id, image_sha256, group_id = next(iter(image_identities))
    result: dict[str, object] = {
        "active_learning_workflow": _TRANSFER_IMAGE_WORKFLOW,
        "lineage_policy": _TRANSFER_CORPUS_ROLE,
        "reproduction_claim": _TRANSFER_REPRODUCTION_CLAIM,
        "model_label": f"project-r{number}",
        "active_learning_round_number": number,
        "parent_model_identity_sha256": scorer.scorer_sha256,
        "transfer_baseline_sha256": baseline.archive_sha256,
        "transfer_split_sha256": baseline.split_sha256,
        "r92_resource_manifest_sha256": scorer.r92_resource_manifest_sha256,
        "r92_classifier_sha256": scorer.r92_classifier_sha256,
        "r92_feature_state_sha256": scorer.feature_state_sha256,
        "frozen_feature_state_policy": _TRANSFER_FEATURE_STATE_POLICY,
        "pca_explained_variance_policy": POST_R92_TRANSFER_PCA_VARIANCE_POLICY,
        "cumulative_al_feature_archive_sha256": cumulative_archive_sha256,
        "effective_train_groups_sha256": compact_json_sha256(
            sorted(set(baseline.train_groups) | {item.group_id for item in decisions})
        ),
        "frozen_test_groups_sha256": compact_json_sha256(
            sorted(baseline.test_groups)
        ),
        "round_image_id": image_id,
        "round_image_sha256": image_sha256,
        "round_group_id": group_id,
    }
    _public_evidence(result)
    return result


def begin_canonical_transfer_image_round(
    pool_root: Path,
    baseline: CanonicalTransferBaselineReference,
    scorer: CanonicalTransferScorerReference,
    output: Path,
    *,
    round_number: int,
    created_at_utc: str,
    prior_round_completion_path: Path | None = None,
    decision_log_path: Path | None = None,
) -> dict[str, object]:
    """Select one explicit-review image round from the transfer lineage."""

    number = _strict_int(round_number, "transfer round number", minimum=1)
    timestamp = _timestamp(created_at_utc)
    pool_rows, pool_result, pool_records = _read_pool(pool_root, require_scores=True)
    _require(
        pool_result["profile"] == CANONICAL_GPU_PROFILE
        and pool_result["bundle_sha256"] == scorer.scorer_sha256
        and pool_result["predictions_sha256"] == scorer.pool_scores_sha256,
        "transfer pool was not scored by the declared Full-profile scorer",
    )
    prior_record: dict[str, object] | None = None
    prior_log_record: dict[str, object] | None = None
    decisions: tuple[CanonicalActiveLearningDecision, ...] = ()
    if number == 1:
        _require(
            scorer.kind == _TRANSFER_R92_SCORER_KIND
            and prior_round_completion_path is None
            and decision_log_path is None,
            "transfer round 1 must start directly from the published r92 scorer",
        )
    else:
        _require(
            scorer.kind == _TRANSFER_PROJECT_SCORER_KIND
            and prior_round_completion_path is not None
            and decision_log_path is not None,
            "later transfer rounds require their project bundle and prior completion",
        )
        prior, prior_record = _transfer_completion_manifest(
            prior_round_completion_path
        )
        decisions, _raw_rows, prior_log_record = _read_transfer_decision_log(
            decision_log_path
        )
        _require(
            prior["round_number"] == number - 1
            and prior["outputs"]["decision_log"]["sha256"]
            == prior_log_record["sha256"]
            and prior["outputs"]["new_bundle_sha256"] == scorer.scorer_sha256
            and prior["chain"]["transfer_baseline"] == baseline.receipt()
            and max(item.round_number for item in decisions) == number - 1,
            "later transfer round ancestry is broken",
        )
    selected, counts, image = _select_single_new_image(
        pool_rows,
        frozenset(item.proposal_id for item in decisions),
        frozenset(set(baseline.source_groups) | {item.group_id for item in decisions}),
        frozenset(item.image_sha256 for item in decisions),
        baseline.train_groups,
        baseline.test_groups,
    )
    output = _new_directory(output)
    shortlist_payload = _csv_bytes(
        CANONICAL_AL_SHORTLIST_COLUMNS,
        _shortlist_rows(selected),
    )
    write_new_bytes(output / "shortlist.csv", shortlist_payload)
    review_result = write_review_export(
        _review_rows([item[0] for item in selected]),
        output / "review_request.csv",
        canonical_actions=True,
    )
    shortlist_record = _input_record(
        output / "shortlist.csv",
        "transfer active-learning shortlist",
        max_bytes=_MAX_TABLE_BYTES,
    )
    review_record = _input_record(
        output / "review_request.csv",
        "transfer active-learning review request",
        max_bytes=_MAX_TABLE_BYTES,
    )
    chain = {
        "transfer_baseline": baseline.receipt(),
        "prior_round_completion_sha256": (
            prior_record["sha256"] if prior_record is not None else None
        ),
        "prior_review_log_sha256": (
            prior_log_record["sha256"] if prior_log_record is not None else None
        ),
        "source_scorer": scorer.receipt(),
        "source_predictions_sha256": pool_result["predictions_sha256"],
        "source_pool_source_sha256": pool_result["source_sha256"],
        "source_feature_archive_sha256": pool_result["feature_archive_sha256"],
    }
    round_id = compact_json_sha256(
        {
            "domain": "compag-curation-transfer-image-round-id/v1",
            "round_number": number,
            "workflow": _TRANSFER_IMAGE_WORKFLOW,
            "policy": _policy(),
            "chain": chain,
            "image": image,
            "shortlist_sha256": shortlist_record["sha256"],
        }
    )
    manifest = {
        "schema": _TRANSFER_IMAGE_ROUND_SCHEMA,
        "status": "PAUSED_FOR_REVIEW",
        "profile": CANONICAL_GPU_PROFILE,
        "kind": _ROUND_PHASE,
        "workflow": _TRANSFER_IMAGE_WORKFLOW,
        "round_number": number,
        "round_id": round_id,
        "created_at_utc": timestamp,
        "policy": _policy(),
        "scientific_lineage": _transfer_scientific_lineage(),
        "chain": chain,
        "inputs": {
            "pool": {"files": list(pool_records), "result": dict(pool_result)},
            "decision_log": prior_log_record,
        },
        "counts": counts,
        "image": image,
        "shortlist": shortlist_record,
        "review_request": review_record,
        "review_contract": review_result["review_contract"],
        "paper_result_reproduction": "NOT_CLAIMED",
    }
    _public_evidence(manifest)
    write_new_json(output / "round_manifest.json", manifest)
    _directory_closure(
        output,
        {"shortlist.csv", "review_request.csv", "round_manifest.json"},
    )
    return manifest


def _transfer_selection_manifest(
    root: Path,
    pool_root: Path,
    baseline: CanonicalTransferBaselineReference,
    scorer: CanonicalTransferScorerReference,
    prior_round_completion_path: Path | None,
    decision_log_path: Path | None,
) -> tuple[
    Mapping[str, object],
    dict[str, object],
    tuple[CanonicalActiveLearningPoolRow, ...],
    tuple[CanonicalActiveLearningDecision, ...],
    list[dict[str, str]],
    dict[str, object] | None,
]:
    root = _safe_directory(root, "transfer selection root")
    _directory_closure(
        root,
        {"shortlist.csv", "review_request.csv", "round_manifest.json"},
    )
    manifest, manifest_record = _read_json(
        root / "round_manifest.json", "transfer round manifest"
    )
    expected_fields = {
        "schema", "status", "profile", "kind", "workflow", "round_number",
        "round_id", "created_at_utc", "policy", "scientific_lineage", "chain",
        "inputs", "counts", "image", "shortlist", "review_request",
        "review_contract", "paper_result_reproduction",
    }
    _require(
        set(manifest) == expected_fields
        and manifest.get("schema") == _TRANSFER_IMAGE_ROUND_SCHEMA
        and manifest.get("status") == "PAUSED_FOR_REVIEW"
        and manifest.get("profile") == CANONICAL_GPU_PROFILE
        and manifest.get("kind") == _ROUND_PHASE
        and manifest.get("workflow") == _TRANSFER_IMAGE_WORKFLOW
        and manifest.get("policy") == _policy()
        and manifest.get("scientific_lineage") == _transfer_scientific_lineage()
        and manifest.get("review_contract") == "CANONICAL_ACTION_WEIGHTED_V2"
        and manifest.get("paper_result_reproduction") == "NOT_CLAIMED",
        "transfer paused manifest identity changed",
    )
    number = _strict_int(manifest.get("round_number"), "transfer round", minimum=1)
    _timestamp(manifest.get("created_at_utc"))
    pool_rows, pool_result, pool_records = _read_pool(pool_root, require_scores=True)
    decisions: tuple[CanonicalActiveLearningDecision, ...] = ()
    raw_log_rows: list[dict[str, str]] = []
    log_record: dict[str, object] | None = None
    prior_record: dict[str, object] | None = None
    if number == 1:
        _require(
            prior_round_completion_path is None and decision_log_path is None,
            "transfer round-1 resume cannot name prior ancestry",
        )
    else:
        _require(
            prior_round_completion_path is not None and decision_log_path is not None,
            "later transfer resume requires prior ancestry",
        )
        prior, prior_record = _transfer_completion_manifest(
            prior_round_completion_path
        )
        decisions, raw_log_rows, log_record = _read_transfer_decision_log(
            decision_log_path
        )
        _require(
            prior["round_number"] == number - 1
            and prior["outputs"]["decision_log"]["sha256"] == log_record["sha256"]
            and prior["outputs"]["new_bundle_sha256"] == scorer.scorer_sha256,
            "transfer resume prior ancestry changed",
        )
    chain = manifest.get("chain")
    expected_chain = {
        "transfer_baseline": baseline.receipt(),
        "prior_round_completion_sha256": (
            prior_record["sha256"] if prior_record is not None else None
        ),
        "prior_review_log_sha256": (
            log_record["sha256"] if log_record is not None else None
        ),
        "source_scorer": scorer.receipt(),
        "source_predictions_sha256": pool_result["predictions_sha256"],
        "source_pool_source_sha256": pool_result["source_sha256"],
        "source_feature_archive_sha256": pool_result["feature_archive_sha256"],
    }
    _require(
        chain == expected_chain
        and manifest.get("inputs")
        == {
            "pool": {"files": list(pool_records), "result": dict(pool_result)},
            "decision_log": log_record,
        },
        "transfer selection hash chain changed",
    )
    selected, counts, image = _select_single_new_image(
        pool_rows,
        frozenset(item.proposal_id for item in decisions),
        frozenset(set(baseline.source_groups) | {item.group_id for item in decisions}),
        frozenset(item.image_sha256 for item in decisions),
        baseline.train_groups,
        baseline.test_groups,
    )
    shortlist_record = _input_record(
        root / "shortlist.csv", "transfer active-learning shortlist",
        max_bytes=_MAX_TABLE_BYTES,
    )
    review_record = _pending_request_record(
        root / "review_request.csv",
        [item[0] for item in selected],
        "transfer active-learning review request",
    )
    shortlist_payload, _size = _stable_payload(
        root / "shortlist.csv",
        "transfer active-learning shortlist",
        max_bytes=_MAX_TABLE_BYTES,
    )
    expected_round_id = compact_json_sha256(
        {
            "domain": "compag-curation-transfer-image-round-id/v1",
            "round_number": number,
            "workflow": _TRANSFER_IMAGE_WORKFLOW,
            "policy": _policy(),
            "chain": expected_chain,
            "image": image,
            "shortlist_sha256": shortlist_record["sha256"],
        }
    )
    _require(
        shortlist_payload
        == _csv_bytes(CANONICAL_AL_SHORTLIST_COLUMNS, _shortlist_rows(selected))
        and manifest.get("counts") == counts
        and manifest.get("image") == image
        and manifest.get("shortlist") == shortlist_record
        and manifest.get("review_request") == review_record
        and manifest.get("round_id") == expected_round_id,
        "transfer shortlist or round binding changed",
    )
    return (
        manifest,
        manifest_record,
        pool_rows,
        decisions,
        raw_log_rows,
        log_record,
    )


def _verify_transfer_written_bundle(
    root: Path,
    *,
    state: CanonicalFeatureState,
    expected_provenance: Mapping[str, object],
    profile: str,
) -> Any:
    root = _safe_directory(root, "transfer model bundle")
    from compag_curation.model_bundle import verify_model_bundle

    bundle = verify_model_bundle(root)
    _require(
        getattr(bundle, "profile", None) == profile
        and getattr(bundle, "threshold", None) == CANONICAL_DECISION_THRESHOLD
        and tuple(getattr(bundle, "feature_order", ())) == CANONICAL_FEATURE_ORDER,
        "transfer retrain callback wrote an incompatible model bundle",
    )
    _verify_bundle_feature_state(bundle, state)
    details = (
        bundle.provenance.get("details")
        if isinstance(getattr(bundle, "provenance", None), Mapping)
        else None
    )
    _require(
        isinstance(details, Mapping)
        and all(details.get(name) == value for name, value in expected_provenance.items()),
        "transfer bundle provenance does not bind the retrain request",
    )
    _sha256(getattr(bundle, "bundle_sha256", None), "transfer model bundle")
    return bundle


def resume_canonical_transfer_image_round(
    selection_root: Path,
    pool_root: Path,
    baseline_root: Path,
    baseline: CanonicalTransferBaselineReference,
    scorer: CanonicalTransferScorerReference,
    reviewed_path: Path,
    feature_state: CanonicalFeatureState,
    cumulative_al_rows: Sequence[CanonicalActiveLearningTrainingRow],
    pool_feature_rows: Sequence[CanonicalActiveLearningTrainingRow],
    output: Path,
    *,
    completed_at_utc: str,
    retrain: TransferRetrainFunction,
    prior_round_completion_path: Path | None = None,
    decision_log_path: Path | None = None,
) -> dict[str, object]:
    """Append one image review and perform a fresh transfer-baseline fit."""

    timestamp = _timestamp(completed_at_utc)
    (
        selection,
        selection_record,
        pool_rows,
        prior_decisions,
        prior_log_rows,
        _prior_log_record,
    ) = _transfer_selection_manifest(
        selection_root,
        pool_root,
        baseline,
        scorer,
        prior_round_completion_path,
        decision_log_path,
    )
    number = int(selection["round_number"])
    reviewed, review_result = validate_review_table(
        Path(selection_root).resolve(strict=True) / "review_request.csv",
        reviewed_path,
    )
    _require(
        0 < len(reviewed) <= CANONICAL_AL_TOP_K
        and len(reviewed) == selection["counts"]["selected"]
        and review_result["pending_sha256"] == selection["review_request"]["sha256"]
        and review_result["review_contract"] == "CANONICAL_ACTION_WEIGHTED_V2",
        "transfer review is incomplete or exceeds the explicit Top-50 contract",
    )
    _require(
        any(item.review_weight > 0.0 for item in reviewed),
        "transfer round cannot be all skip; at least one effective reviewed "
        "proposal is required for a scientifically meaningful fresh fit",
    )
    _require(
        timestamp >= selection["created_at_utc"]
        and (
            not prior_decisions
            or timestamp >= prior_decisions[-1].decision_timestamp_utc
        ),
        "transfer completion timestamp precedes its ancestry",
    )
    validated_pool_features = _validate_accumulated_feature_rows(
        tuple(pool_feature_rows)
    )
    _require(
        {row.proposal_id for row in validated_pool_features}
        == {row.proposal_id for row in pool_rows}
        and {row.source_sha256 for row in validated_pool_features}
        == {selection["chain"]["source_pool_source_sha256"]}
        and _feature_archive_hash(validated_pool_features)
        == selection["chain"]["source_feature_archive_sha256"],
        "transfer pool features differ from the bound physical or logical archive",
    )
    appended_rows = [dict(row) for row in prior_log_rows]
    for offset, decision in enumerate(reviewed, start=len(appended_rows) + 1):
        appended_rows.append(
            _transfer_decision_csv_row(
                decision,
                sequence=offset,
                round_number=number,
                timestamp_utc=timestamp,
                scorer=scorer,
                round_manifest_sha256=str(selection_record["sha256"]),
                reviewed_sha256=str(review_result["reviewed_sha256"]),
            )
        )
    all_decisions = _parse_transfer_decision_rows(appended_rows)
    reviewed_ids = {item.proposal_id for item in reviewed}
    current_rows = tuple(
        row for row in validated_pool_features if row.proposal_id in reviewed_ids
    )
    _require(
        {row.proposal_id for row in current_rows} == reviewed_ids,
        "transfer raw feature archive omits a reviewed proposal",
    )
    ordered_al_rows = _validate_transfer_al_rows(
        tuple(cumulative_al_rows),
        all_decisions,
        baseline,
    )
    cumulative_current_rows = tuple(
        row for row in ordered_al_rows if row.proposal_id in reviewed_ids
    )
    _require(
        len(cumulative_current_rows) == len(current_rows)
        and {
            (row.proposal_id, float(row.raw_feature.scale)): _raw_feature_content_sha256(row)
            for row in cumulative_current_rows
        }
        == {
            (row.proposal_id, float(row.raw_feature.scale)): _raw_feature_content_sha256(row)
            for row in current_rows
        },
        "cumulative transfer archive does not contain the exact current inference rows",
    )
    if prior_round_completion_path is not None:
        prior_completion, _prior_completion_record = _transfer_completion_manifest(
            prior_round_completion_path
        )
        prior_archive_rows, prior_archive_receipt = (
            read_canonical_accumulated_feature_archive(
                prior_round_completion_path.absolute().parent
                / "accumulated_al_features.csv"
            )
        )
        prior_ids = {item.proposal_id for item in prior_decisions}
        supplied_prior_rows = tuple(
            row for row in ordered_al_rows if row.proposal_id in prior_ids
        )
        _require(
            prior_completion["outputs"]["cumulative_al_features"]
            == prior_archive_receipt
            and len(supplied_prior_rows) == len(prior_archive_rows)
            and _feature_archive_hash(supplied_prior_rows)
            == _feature_archive_hash(prior_archive_rows)
            and {
                (row.proposal_id, float(row.raw_feature.scale)):
                _raw_feature_content_sha256(row)
                for row in supplied_prior_rows
            }
            == {
                (row.proposal_id, float(row.raw_feature.scale)):
                _raw_feature_content_sha256(row)
                for row in prior_archive_rows
            },
            "cumulative transfer rows differ from the sealed prior ancestry",
        )
    validate_canonical_feature_state(feature_state)
    state_hashes = _feature_state_hashes(feature_state)
    _require(
        state_hashes["prototype_sha256"] == scorer.prototype_sha256
        and state_hashes["pca_components_sha256"] == scorer.pca_components_sha256
        and state_hashes["pca_mean_sha256"] == scorer.pca_mean_sha256,
        "transfer feature state differs from the declared frozen scorer state",
    )
    output = _new_directory(output)
    decision_payload = _csv_bytes(
        CANONICAL_TRANSFER_DECISION_LOG_COLUMNS,
        appended_rows,
    )
    write_new_bytes(output / "decision_log.csv", decision_payload)
    decision_record = _input_record(
        output / "decision_log.csv",
        "transfer active-learning decision log",
        max_bytes=_MAX_TABLE_BYTES,
    )
    accumulated_receipt = write_canonical_accumulated_feature_archive(
        ordered_al_rows,
        output / "accumulated_al_features.csv",
    )
    latest = _latest_decisions(all_decisions)
    al_rows_sha256 = _training_row_hash(ordered_al_rows, latest)
    effective_train_groups = frozenset(
        set(baseline.train_groups) | {item.group_id for item in all_decisions}
    )
    expected_provenance = _transfer_expected_bundle_provenance(
        number=number,
        baseline=baseline,
        scorer=scorer,
        decision_log_sha256=str(decision_record["sha256"]),
        cumulative_al_rows_sha256=al_rows_sha256,
        selection_manifest_sha256=str(selection_record["sha256"]),
        state=feature_state,
        decisions=all_decisions,
        cumulative_archive_sha256=str(accumulated_receipt["logical_sha256"]),
        profile=CANONICAL_GPU_PROFILE,
        device="cuda",
    )
    request = CanonicalTransferRetrainRequest(
        round_number=number,
        baseline_root=_safe_directory(baseline_root, "transfer baseline root"),
        baseline=baseline,
        feature_state=feature_state,
        effective_decisions=all_decisions,
        cumulative_al_rows=ordered_al_rows,
        train_groups=effective_train_groups,
        test_groups=baseline.test_groups,
        decision_log_sha256=str(decision_record["sha256"]),
        cumulative_al_rows_sha256=al_rows_sha256,
        source_scorer_sha256=scorer.scorer_sha256,
        selection_manifest_sha256=str(selection_record["sha256"]),
        expected_bundle_provenance=MappingProxyType(expected_provenance),
        profile=CANONICAL_GPU_PROFILE,
        device="cuda",
    )
    request_record = {
        "schema": _TRANSFER_RETRAIN_REQUEST_SCHEMA,
        "profile": request.profile,
        "device": request.device,
        "round_number": number,
        "transfer_baseline": baseline.receipt(),
        "decision_log_sha256": request.decision_log_sha256,
        "cumulative_al_rows_sha256": request.cumulative_al_rows_sha256,
        "source_scorer_sha256": request.source_scorer_sha256,
        "selection_manifest_sha256": request.selection_manifest_sha256,
        "effective_proposals": len(request.effective_decisions),
        "cumulative_al_feature_rows": len(request.cumulative_al_rows),
        "train_groups_sha256": compact_json_sha256(sorted(request.train_groups)),
        "test_groups_sha256": compact_json_sha256(sorted(request.test_groups)),
        **state_hashes,
        "feature_state_policy": _TRANSFER_FEATURE_STATE_POLICY,
        "pca_explained_variance": None,
        "training_mode": _TRANSFER_TRAINING_MODE,
        "baseline_action_adapter_policy": TRANSFER_BASELINE_ACTION_ADAPTER_POLICY,
        "warm_start": False,
    }
    _public_evidence(request_record)
    write_new_json(output / "retrain_request.json", request_record)
    model_output = output / "model_bundle"
    _require(
        not model_output.exists() and not model_output.is_symlink(),
        "transfer model output already exists",
    )
    retrain(request, model_output)
    _require(
        model_output.exists() and model_output.is_dir() and not model_output.is_symlink(),
        "transfer retrain callback did not create a model bundle",
    )
    new_bundle = _verify_transfer_written_bundle(
        model_output,
        state=feature_state,
        expected_provenance=expected_provenance,
        profile=CANONICAL_GPU_PROFILE,
    )
    lineage = {
        "schema": _TRANSFER_MODEL_LINEAGE_SCHEMA,
        "status": "PASS",
        **expected_provenance,
        "new_bundle_sha256": new_bundle.bundle_sha256,
    }
    _public_evidence(lineage)
    write_new_json(output / "model_lineage.json", lineage)
    lineage_record = _input_record(
        output / "model_lineage.json",
        "transfer model lineage",
        max_bytes=16 * 1024 * 1024,
    )
    completion = {
        "schema": _TRANSFER_IMAGE_ROUND_COMPLETION_SCHEMA,
        "status": "PASS",
        "profile": CANONICAL_GPU_PROFILE,
        "kind": _ROUND_PHASE,
        "workflow": _TRANSFER_IMAGE_WORKFLOW,
        "round_number": number,
        "completed_at_utc": timestamp,
        "policy": _policy(),
        "scientific_lineage": _transfer_scientific_lineage(),
        "chain": {
            **dict(selection["chain"]),
            "selection_manifest_sha256": selection_record["sha256"],
            "reviewed_batch_sha256": review_result["reviewed_sha256"],
        },
        "image": dict(selection["image"]),
        "counts": {
            "reviewed_batch_rows": len(reviewed),
            "decision_log_rows": len(all_decisions),
            "effective_al_proposals": len(latest),
            "cumulative_al_feature_rows": len(ordered_al_rows),
            "baseline_feature_rows": baseline.row_count,
        },
        "feature_state": {
            **state_hashes,
            "policy": _TRANSFER_FEATURE_STATE_POLICY,
            "pca_explained_variance": None,
        },
        "outputs": {
            "decision_log": decision_record,
            "cumulative_al_features": accumulated_receipt,
            "cumulative_al_rows_sha256": al_rows_sha256,
            "new_bundle_sha256": new_bundle.bundle_sha256,
            "model_lineage": lineage_record,
        },
        "paper_result_reproduction": "NOT_CLAIMED",
    }
    _public_evidence(completion)
    write_new_json(output / "round_completion.json", completion)
    _directory_closure(
        output,
        {
            "decision_log.csv",
            "accumulated_al_features.csv",
            "retrain_request.json",
            "model_lineage.json",
            "round_completion.json",
        },
        {"model_bundle"},
    )
    return completion


def _transfer_completion_manifest(
    path: Path,
) -> tuple[Mapping[str, object], dict[str, object]]:
    value, record = _read_json(path, "prior transfer completion")
    expected_fields = {
        "schema", "status", "profile", "kind", "workflow", "round_number",
        "completed_at_utc", "policy", "scientific_lineage", "chain", "image",
        "counts", "feature_state", "outputs", "paper_result_reproduction",
    }
    _require(
        set(value) == expected_fields
        and value.get("schema") == _TRANSFER_IMAGE_ROUND_COMPLETION_SCHEMA
        and value.get("status") == "PASS"
        and value.get("profile") == CANONICAL_GPU_PROFILE
        and value.get("kind") == _ROUND_PHASE
        and value.get("workflow") == _TRANSFER_IMAGE_WORKFLOW
        and value.get("policy") == _policy()
        and value.get("scientific_lineage") == _transfer_scientific_lineage()
        and value.get("paper_result_reproduction") == "NOT_CLAIMED",
        "prior transfer completion identity changed",
    )
    number = _strict_int(value.get("round_number"), "prior transfer round", minimum=1)
    _timestamp(value.get("completed_at_utc"))
    chain = value.get("chain")
    _require(
        isinstance(chain, dict)
        and set(chain)
        == {
            "transfer_baseline", "prior_round_completion_sha256",
            "prior_review_log_sha256", "source_scorer",
            "source_predictions_sha256", "source_pool_source_sha256",
            "source_feature_archive_sha256", "selection_manifest_sha256",
            "reviewed_batch_sha256",
        },
        "prior transfer completion chain changed",
    )
    outputs = value.get("outputs")
    _require(
        isinstance(outputs, dict)
        and set(outputs)
        == {
            "decision_log", "cumulative_al_features",
            "cumulative_al_rows_sha256", "new_bundle_sha256", "model_lineage",
        },
        "prior transfer completion outputs changed",
    )
    for digest in (
        outputs.get("cumulative_al_rows_sha256"),
        outputs.get("new_bundle_sha256"),
        chain.get("selection_manifest_sha256"),
        chain.get("reviewed_batch_sha256"),
    ):
        _sha256(digest, "prior transfer completion hash")
    decision_path = path.absolute().parent / "decision_log.csv"
    decisions, _raw, decision_record = _read_transfer_decision_log(decision_path)
    _require(
        max(item.round_number for item in decisions) == number
        and outputs.get("decision_log") == decision_record,
        "prior transfer decision-log receipt changed",
    )
    archive_rows, archive_receipt = read_canonical_accumulated_feature_archive(
        path.absolute().parent / "accumulated_al_features.csv"
    )
    _require(
        outputs.get("cumulative_al_features") == archive_receipt
        and len(archive_rows) == value["counts"]["cumulative_al_feature_rows"],
        "prior transfer cumulative feature archive changed",
    )
    return value, record


__all__ = [
    "CANONICAL_AL_ACCUMULATED_FEATURE_COLUMNS",
    "CANONICAL_AL_DECISION_LOG_COLUMNS",
    "CANONICAL_AL_MARGIN",
    "CANONICAL_AL_POOL_COLUMNS",
    "CANONICAL_AL_SHORTLIST_COLUMNS",
    "CANONICAL_AL_TOP_K",
    "CANONICAL_TRANSFER_DECISION_LOG_COLUMNS",
    "CANONICAL_TRANSFER_SCORE_COLUMNS",
    "TRANSFER_BASELINE_ACTION_ADAPTER_POLICY",
    "CanonicalActiveLearningDecision",
    "CanonicalActiveLearningPoolRow",
    "CanonicalActiveLearningRetrainRequest",
    "CanonicalActiveLearningTrainingRow",
    "CanonicalTransferBaselineReference",
    "CanonicalTransferRetrainRequest",
    "CanonicalTransferScorerReference",
    "begin_canonical_active_learning_round",
    "begin_canonical_transfer_image_round",
    "complete_initial_labeling_export_all",
    "read_canonical_accumulated_feature_archive",
    "resume_canonical_active_learning_round",
    "resume_canonical_transfer_image_round",
    "write_canonical_active_learning_pool",
    "write_canonical_accumulated_feature_archive",
    "write_initial_labeling_export_all",
]
