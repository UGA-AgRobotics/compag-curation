"""Durable visual review for one sealed active-learning ``begin-round``.

This module deliberately stops at publication of a reviewed CSV.  It never
resumes an active-learning operation, trains a model, or infers a label.  The
visual index is accepted only when the published begin-round operation, its
exact stage-60 input, the selection tables, and a caller-supplied directory of
original inference images all close over the same identities.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import stat
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from compag_curation.canonical.active_learning import (
    CANONICAL_AL_SHORTLIST_COLUMNS,
    _directory_closure,
    _format_float,
    _input_record,
    _policy,
    _read_csv,
    _read_json,
    _read_pool,
)
from compag_curation.canonical.active_learning_facade import (
    _directory_record,
    _input_by_role,
    _verify_published_operation,
)
from compag_curation.canonical.active_learning_service import (
    _INFERENCE_RESULT_FIELDS,
    _INFERENCE_SCHEMA,
    _canonical_profile_device,
    _prediction_rows,
    _validate_inventory,
)
from compag_curation.canonical.service import (
    CANONICAL_FULL_IMAGE_NMS_IOU,
    CANONICAL_INFERENCE_COLUMNS,
)
from compag_curation.canonical.spec import (
    CANONICAL_DECISION_THRESHOLD,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
    CANONICAL_PROFILE,
    CANONICAL_GPU_PROFILE,
    V2_PIPELINE_PROFILES,
    inference_amg_execution_points_per_batch,
)
from compag_curation.model_bundle import (
    POST_R92_TRANSFER_FROZEN_STATE_POLICY,
    POST_R92_TRANSFER_LINEAGE_POLICY,
    POST_R92_TRANSFER_REPRODUCTION_CLAIM,
    POST_R92_TRANSFER_WORKFLOW,
)
from compag_curation.public_config import (
    MAX_IMAGE_FILE_BYTES,
    SUPPORTED_SUFFIXES,
    _image_info,
)
from compag_curation.public_io import (
    PublicIOError,
    canonical_json_bytes,
    canonical_json_value,
    compact_json_sha256,
    fsync_directory,
    portable_basename,
    publish_directory_noreplace,
    publish_file_noreplace,
    rename_noreplace,
    sha256_file,
    stable_file,
    write_new_bytes,
    write_new_json,
)
from compag_curation.review.exchange import (
    CANONICAL_REVIEW_COLUMNS,
    REVIEW_ACTION_WEIGHTS,
    read_review_export,
    validate_review_table,
)
from compag_curation.review.session import DECISIONS, Decision
from compag_curation.review.scene import SceneProposal, round_scene_proposals


SESSION_SCHEMA = "compag-curation-active-learning-round-review-session/v1"
EVENT_SCHEMA = "compag-curation-active-learning-round-review-event/v1"
EVENT_HEAD_SCHEMA = "compag-curation-active-learning-round-review-event-head/v1"
FINAL_SCHEMA = "compag-curation-active-learning-round-review-final/v1"
STATUS_SCHEMA = "compag-curation-active-learning-round-review-status/v1"
ROUND_SCHEMA = "compag-curation-canonical-active-learning-selection/v2"
IMAGE_ROUND_SCHEMA = "compag-curation-canonical-active-learning-selection/v3"
IMAGE_ROUND_WORKFLOW = "SINGLE_NEW_IMAGE_UNCERTAINTY_BATCH_FULL_RETRAIN_V1"
TRANSFER_ROUND_SCHEMA = (
    "compag-curation-canonical-transfer-active-learning-selection/v1"
)
TRANSFER_ROUND_WORKFLOW = POST_R92_TRANSFER_WORKFLOW
SOURCE_KIND = "ACTIVE_LEARNING_ROUND"
VISUAL_SIZE = 512
MAX_EVENT_BYTES = 32 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_PREDICTIONS_BYTES = 2 * 1024 * 1024 * 1024
MAX_EVENT_COUNT = 100_000
MAX_IMAGE_DIRECTORY_EXTRA_MEMBERS = 32
_SHA256 = re.compile(r"[0-9a-f]{64}")
_EVENT_NAME = re.compile(r"([0-9]{8})\.json")
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{7,127}")
_FINAL_FIELDS = {
    "schema",
    "status",
    "finalized_at_utc",
    "recovered_after_output_publication",
    "source_kind",
    "operation_id",
    "round_number",
    "round_id",
    "source_binding_sha256",
    "review_request_sha256",
    "reviewed_sha256",
    "rows",
    "positive_rows",
    "negative_rows",
    "action_counts",
    "effective_weight",
    "event_sequence",
    "event_sha256",
    "output",
    "input_policy",
    "publication_policy",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_timestamp(value: object, role: str) -> None:
    _require(isinstance(value, str), f"{role} is not a UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise PublicIOError(f"{role} is not a canonical UTC timestamp") from exc
    _require(parsed.strftime("%Y-%m-%dT%H:%M:%SZ") == value, f"{role} is not canonical")


def _strict_int(value: object, role: str, *, minimum: int = 0, maximum: int = 1_000_000_000) -> int:
    text = str(value)
    _require(re.fullmatch(r"0|[1-9][0-9]*", text) is not None, f"{role} is not a canonical integer")
    result = int(text)
    _require(minimum <= result <= maximum, f"{role} is outside its supported range")
    return result


def _strict_float(value: object, role: str) -> float:
    try:
        result = float(str(value))
    except ValueError as exc:
        raise PublicIOError(f"{role} is not numeric") from exc
    _require(math.isfinite(result), f"{role} is not finite")
    return result


def _file_record(path: Path, role: str, *, max_bytes: int) -> tuple[bytes, dict[str, object]]:
    payload, info = stable_file(path, max_bytes=max_bytes)
    _require(
        stat.S_ISREG(info.st_mode) and not path.is_symlink() and info.st_nlink == 1,
        f"{role} must be a single-link regular file",
    )
    return payload, {
        "role": role,
        "basename": portable_basename(path.name, f"{role} basename"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _safe_directory(path: Path, role: str) -> Path:
    absolute = path.absolute()
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError(f"{role} cannot be resolved") from exc
    info = absolute.lstat()
    _require(
        absolute == resolved and stat.S_ISDIR(info.st_mode) and not absolute.is_symlink(),
        f"{role} is unsafe or contains a symlink component",
    )
    return absolute


def _external_path(path: Path, immutable_roots: Sequence[Path], role: str, *, derived_basename: str) -> Path:
    absolute = path.absolute()
    parent = absolute.parent.resolve(strict=True)
    canonical = parent / absolute.name
    _require(absolute == canonical and absolute.name not in {"", ".", ".."}, f"{role} has an unsafe component")
    portable_basename(absolute.name, role)
    _require(
        len(os.fsencode(absolute.name)) <= 255 and len(os.fsencode(derived_basename)) <= 255,
        f"{role} name is too long for atomic publication",
    )
    for root in immutable_roots:
        _require(root != absolute and root not in absolute.parents, f"{role} must be outside sealed inputs")
    return absolute


def _directory_is_private(path: Path, role: str) -> None:
    info = path.lstat()
    _require(
        stat.S_ISDIR(info.st_mode)
        and not path.is_symlink()
        and info.st_uid == os.getuid()
        and info.st_gid == os.getgid()
        and stat.S_IMODE(info.st_mode) == 0o700,
        f"{role} must be an owned mode-0700 directory",
    )


def _regular_private(path: Path, role: str, *, max_bytes: int) -> bytes:
    info = path.lstat()
    _require(
        stat.S_ISREG(info.st_mode)
        and not path.is_symlink()
        and info.st_nlink == 1
        and info.st_uid == os.getuid()
        and info.st_gid == os.getgid()
        and stat.S_IMODE(info.st_mode) == 0o600,
        f"{role} must be an owned single-link mode-0600 file",
    )
    payload, _snapshot = stable_file(path, max_bytes=max_bytes)
    return payload


@dataclass(frozen=True)
class RoundProposalView:
    proposal_id: str
    proposal_sha256: str
    image_id: str
    image_sha256: str
    image_name: str
    group_id: str
    tile_name: str
    tile_index: int
    proposal_index: int
    label_prefix: str
    selection_rank: int
    xgb_p: float
    distance_to_threshold: float
    uncertainty: float
    prediction: int
    kept: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    original_width: int
    original_height: int
    polygon: tuple[int, ...]


@dataclass(frozen=True)
class RoundVisualView:
    index: int
    tile_name: str
    proposal_ids: tuple[str, ...]


@dataclass(frozen=True)
class RoundReviewIndex:
    request_rows: tuple[dict[str, str], ...]
    request_columns: tuple[str, ...]
    order: tuple[str, ...]
    proposals: Mapping[str, RoundProposalView]
    tiles: tuple[RoundVisualView, ...]
    groups: tuple[str, ...]
    review_request_sha256: str
    shortlist_sha256: str
    predictions_sha256: str
    visual_order_sha256: str
    scene_proposals: Mapping[str, SceneProposal] = field(default_factory=dict)


@dataclass(frozen=True)
class RoundImage:
    name: str
    path: Path
    image_id: str
    image_sha256: str
    group_id: str
    size_bytes: int
    width: int
    height: int

    def receipt(self) -> dict[str, object]:
        return {
            "image_name": self.name,
            "image_id": self.image_id,
            "image_sha256": self.image_sha256,
            "group_id": self.group_id,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class RoundSnapshot:
    operation_root: Path
    stage60_root: Path
    image_root: Path
    operation_id: str
    operation_manifest_sha256: str
    stage60_record: Mapping[str, object]
    round_number: int
    round_id: str
    profile: str
    bundle_sha256: str
    inference_result_sha256: str
    predictions_sha256: str
    review_request: Path
    review_request_sha256: str
    shortlist_sha256: str
    round_manifest_sha256: str
    image_inventory: tuple[RoundImage, ...]
    image_inventory_sha256: str

    @property
    def images_by_name(self) -> Mapping[str, RoundImage]:
        return {item.name: item for item in self.image_inventory}

    @property
    def binding_sha256(self) -> str:
        return compact_json_sha256(
            {
                "source_kind": SOURCE_KIND,
                "operation_id": self.operation_id,
                "operation_manifest_sha256": self.operation_manifest_sha256,
                "stage60": dict(self.stage60_record),
                "round_number": self.round_number,
                "round_id": self.round_id,
                "round_manifest_sha256": self.round_manifest_sha256,
                "review_request_sha256": self.review_request_sha256,
                "shortlist_sha256": self.shortlist_sha256,
                "predictions_sha256": self.predictions_sha256,
                "inference_result_sha256": self.inference_result_sha256,
                "image_inventory_sha256": self.image_inventory_sha256,
            }
        )


def _image_inventory(root: Path, expected: Sequence[Mapping[str, object]]) -> tuple[tuple[RoundImage, ...], str]:
    root = _safe_directory(root, "original inference image directory")
    expected_by_name = {str(item["image_name"]): item for item in expected}
    _require(len(expected_by_name) == len(expected), "inference image inventory contains duplicate names")
    supported: dict[str, Path] = {}
    folded: set[str] = set()
    member_count = 0
    member_limit = len(expected) + MAX_IMAGE_DIRECTORY_EXTRA_MEMBERS
    for child in root.iterdir():
        member_count += 1
        _require(
            member_count <= member_limit,
            "original image directory exceeds its member-count bound",
        )
        info = child.lstat()
        _require(not child.is_symlink(), f"original image directory contains a symlink: {child.name}")
        if child.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, f"original image is not a single-link file: {child.name}")
        name = portable_basename(child.name, "original image name")
        key = name.casefold()
        _require(key not in folded, f"original image name collision: {name}")
        folded.add(key)
        supported[name] = child
    _require(set(supported) == set(expected_by_name), "original image names differ from the sealed inference inventory")
    rows: list[RoundImage] = []
    for name in sorted(supported):
        path = supported[name]
        payload, info = stable_file(path, max_bytes=MAX_IMAGE_FILE_BYTES)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and not path.is_symlink(), f"original image is unsafe: {name}")
        try:
            width, height, _orientation = _image_info(path, payload)
        except Exception as exc:
            if isinstance(exc, PublicIOError):
                raise
            raise PublicIOError(f"original image cannot be validated: {name}") from exc
        expected_row = expected_by_name[name]
        digest = hashlib.sha256(payload).hexdigest()
        _require(
            digest == expected_row["image_sha256"]
            and len(payload) == expected_row["size_bytes"]
            and width == expected_row["width"]
            and height == expected_row["height"],
            f"original image differs from sealed inference input: {name}",
        )
        rows.append(
            RoundImage(
                name=name,
                path=path,
                image_id=str(expected_row["image_id"]),
                image_sha256=digest,
                group_id=str(expected_row["group_id"]),
                size_bytes=len(payload),
                width=width,
                height=height,
            )
        )
    receipts = [item.receipt() for item in rows]
    return tuple(rows), compact_json_sha256(receipts)


def _inference_index(stage60_root: Path) -> tuple[
    Mapping[str, object],
    dict[str, dict[str, str]],
    tuple[Mapping[str, object], ...],
    str,
    str,
]:
    predictions_path = stage60_root / "predictions.csv"
    predictions_payload, _prediction_record = _file_record(
        predictions_path, "canonical inference predictions", max_bytes=MAX_PREDICTIONS_BYTES
    )
    prediction_rows, by_id = _prediction_rows(predictions_payload)
    result_payload, result_record = _file_record(
        stage60_root / "inference_result.json", "canonical inference result", max_bytes=MAX_JSON_BYTES
    )
    value = canonical_json_value(result_payload, "canonical inference result")
    profile = value.get("profile") if isinstance(value, dict) else None
    device = value.get("device") if isinstance(value, dict) else None
    _canonical_profile_device(profile, device)
    _require(
        isinstance(value, dict)
        and set(value) == _INFERENCE_RESULT_FIELDS
        and value.get("schema") == _INFERENCE_SCHEMA
        and value.get("status") == "PASS"
        and value.get("profile") in V2_PIPELINE_PROFILES
        and value.get("feature_order_sha256") == CANONICAL_FEATURE_ORDER_SHA256
        and value.get("threshold") == CANONICAL_DECISION_THRESHOLD
        and value.get("threshold_method") == "FIXED_CANONICAL_METHOD"
        and value.get("full_image_nms_iou") == CANONICAL_FULL_IMAGE_NMS_IOU
        and value.get("sam2_execution_points_per_batch")
        == (
            CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH
            if profile == CANONICAL_PROFILE
            else inference_amg_execution_points_per_batch(str(profile))
        )
        and value.get("paper_result_reproduction") == "NOT_CLAIMED",
        "canonical inference result identity changed",
    )
    predictions_sha256 = hashlib.sha256(predictions_payload).hexdigest()
    _require(value.get("predictions_sha256") == predictions_sha256, "inference result does not bind predictions.csv")
    inventory = tuple(_validate_inventory(value.get("input_inventory")))
    inventory_by_id = {str(item["image_id"]): item for item in inventory}
    counts = {image_id: 0 for image_id in inventory_by_id}
    for row in prediction_rows:
        item = inventory_by_id.get(row["image_id"])
        _require(
            item is not None
            and row["image"] == item["image_name"]
            and row["image_sha256"] == item["image_sha256"]
            and row["group_id"] == item["group_id"]
            and int(row["orig_w"]) == item["width"]
            and int(row["orig_h"]) == item["height"],
            "prediction differs from the inference image inventory",
        )
        counts[row["image_id"]] += 1
    _require(
        all(counts[str(item["image_id"])] == item["candidate_rows"] for item in inventory)
        and value.get("image_count") == len(inventory)
        and value.get("tile_count") == sum(int(item["tile_count"]) for item in inventory)
        and value.get("proposal_count") == value.get("prediction_rows") == len(prediction_rows)
        and value.get("proposal_count") == sum(int(item["candidate_rows"]) for item in inventory)
        and value.get("positive_rows") == sum(row["prediction"] == "1" for row in prediction_rows)
        and value.get("kept_rows") == sum(row["kept"] == "1" for row in prediction_rows),
        "canonical inference counts changed",
    )
    raw = value.get("raw_feature_archive")
    _require(
        isinstance(raw, dict)
        and set(raw)
        == {
            "schema", "status", "profile", "basename", "columns_sha256",
            "crop_scales", "rows_per_proposal", "proposal_count", "row_count",
            "bundle_sha256", "predictions_sha256", "sha256", "size_bytes",
        }
        and raw.get("schema") == "compag-curation-canonical-active-learning-raw-features/v1"
        and raw.get("status") == "PASS"
        and raw.get("profile") == profile
        and raw.get("basename") == "raw_features.csv"
        and raw.get("bundle_sha256") == value.get("bundle_sha256")
        and raw.get("predictions_sha256") == predictions_sha256
        and raw.get("proposal_count") == len(prediction_rows)
        and raw.get("rows_per_proposal") == len(raw.get("crop_scales", ()))
        and raw.get("row_count") == len(prediction_rows) * int(raw.get("rows_per_proposal", 0))
        and isinstance(raw.get("sha256"), str)
        and _SHA256.fullmatch(str(raw.get("sha256"))) is not None
        and isinstance(raw.get("size_bytes"), int)
        and not isinstance(raw.get("size_bytes"), bool)
        and int(raw.get("size_bytes")) > 0,
        "canonical raw-feature receipt binding changed",
    )
    return value, by_id, inventory, predictions_sha256, str(result_record["sha256"])


def _selection_index(
    operation_root: Path,
    inference: Mapping[str, object],
    predictions: Mapping[str, Mapping[str, str]],
    predictions_sha256: str,
    *,
    image_scoped: bool = False,
    transfer_scoped: bool = False,
) -> tuple[Mapping[str, object], RoundReviewIndex, str]:
    _require(
        not (image_scoped and transfer_scoped),
        "round review selection mode is ambiguous",
    )
    single_image_scoped = image_scoped or transfer_scoped
    payload_root = operation_root / "payload"
    selection = payload_root / "selection"
    pool_root = payload_root / "round_pool"
    _directory_closure(selection, {"shortlist.csv", "review_request.csv", "round_manifest.json"})
    pool_rows, pool_result, pool_records = _read_pool(pool_root, require_scores=True)
    shortlist_role = (
        "transfer active-learning shortlist"
        if transfer_scoped
        else "active-learning shortlist"
    )
    review_request_role = (
        "transfer active-learning review request"
        if transfer_scoped
        else "active-learning review request"
    )
    shortlist, shortlist_record = _read_csv(
        selection / "shortlist.csv", CANONICAL_AL_SHORTLIST_COLUMNS, shortlist_role
    )
    request_rows, request_meta = read_review_export(selection / "review_request.csv")
    _require(tuple(request_meta["columns"]) == CANONICAL_REVIEW_COLUMNS, "round review request is not canonical")
    request_record = _input_record(
        selection / "review_request.csv",
        review_request_role,
        max_bytes=128 * 1024 * 1024,
    )
    _require(request_record["sha256"] == request_meta["sha256"], "round review request changed while indexing")
    manifest, manifest_record = _read_json(selection / "round_manifest.json", "active-learning round manifest")
    manifest_fields = {
        "schema", "status", "profile", "kind", "round_number", "round_id",
        "created_at_utc", "policy", "chain", "inputs", "counts", "shortlist",
        "review_request", "review_contract", "paper_result_reproduction",
    }
    if single_image_scoped:
        manifest_fields |= {"workflow", "image"}
    if transfer_scoped:
        manifest_fields |= {"scientific_lineage"}
    expected_schema = (
        TRANSFER_ROUND_SCHEMA
        if transfer_scoped
        else IMAGE_ROUND_SCHEMA
        if image_scoped
        else ROUND_SCHEMA
    )
    expected_workflow = (
        TRANSFER_ROUND_WORKFLOW if transfer_scoped else IMAGE_ROUND_WORKFLOW
    )
    _require(
        set(manifest) == manifest_fields
        and manifest.get("schema") == expected_schema
        and manifest.get("status") == "PAUSED_FOR_REVIEW"
        and manifest.get("kind") == SOURCE_KIND
        and (
            manifest.get("profile") == CANONICAL_GPU_PROFILE
            if transfer_scoped
            else manifest.get("profile") in V2_PIPELINE_PROFILES
        )
        and manifest.get("profile") == inference.get("profile") == pool_result.get("profile")
        and manifest.get("policy") == _policy()
        and (
            not single_image_scoped
            or manifest.get("workflow") == expected_workflow
        )
        and (
            not transfer_scoped
            or manifest.get("scientific_lineage")
            == {
                "source_corpus_role": POST_R92_TRANSFER_LINEAGE_POLICY,
                "reproduction_claim": POST_R92_TRANSFER_REPRODUCTION_CLAIM,
                "feature_state_policy": POST_R92_TRANSFER_FROZEN_STATE_POLICY,
            }
        )
        and manifest.get("review_contract") == "CANONICAL_ACTION_WEIGHTED_V2"
        and manifest.get("paper_result_reproduction") == "NOT_CLAIMED",
        "active-learning round manifest identity changed",
    )
    _require_timestamp(manifest.get("created_at_utc"), "active-learning round creation time")
    round_number = _strict_int(manifest.get("round_number"), "active-learning round number", minimum=1)
    round_id = str(manifest.get("round_id"))
    _require(_SHA256.fullmatch(round_id) is not None, "active-learning round ID is invalid")
    manifest_inputs = manifest.get("inputs")
    expected_input_fields = (
        {"pool", "decision_log"}
        if transfer_scoped
        else {"pool", "frozen_split", "decision_log"}
    )
    _require(
        isinstance(manifest_inputs, dict)
        and set(manifest_inputs) == expected_input_fields
        and manifest_inputs.get("pool") == {"files": list(pool_records), "result": dict(pool_result)}
        and manifest.get("shortlist") == shortlist_record
        and manifest.get("review_request") == request_record,
        "active-learning selection files differ from the round manifest",
    )
    chain = manifest.get("chain")
    chain_fields = (
        {
            "transfer_baseline",
            "prior_round_completion_sha256",
            "prior_review_log_sha256",
            "source_scorer",
            "source_predictions_sha256",
            "source_pool_source_sha256",
            "source_feature_archive_sha256",
        }
        if transfer_scoped
        else {
            "genesis_initial_completion_sha256",
            "prior_round_completion_sha256",
            "prior_review_log_sha256",
            "source_bundle_sha256",
            "source_predictions_sha256",
            "source_pool_source_sha256",
            "source_feature_archive_sha256",
            "frozen_split_sha256",
        }
    )
    if image_scoped:
        chain_fields |= {
            "genesis_stage20_features_sha256",
            "genesis_split_sha256",
        }
    source_scorer = chain.get("source_scorer") if isinstance(chain, dict) else None
    source_identity = (
        source_scorer.get("scorer_sha256")
        if isinstance(source_scorer, dict)
        else chain.get("source_bundle_sha256")
        if isinstance(chain, dict)
        else None
    )
    _require(
        isinstance(chain, dict)
        and set(chain) == chain_fields
        and chain.get("source_predictions_sha256") == pool_result.get("predictions_sha256")
        and source_identity == pool_result.get("bundle_sha256")
        and chain.get("source_pool_source_sha256") == pool_result.get("source_sha256")
        and chain.get("source_feature_archive_sha256") == pool_result.get("feature_archive_sha256")
        and all(
            value is None or (isinstance(value, str) and _SHA256.fullmatch(value) is not None)
            for key, value in chain.items()
            if key not in {"transfer_baseline", "source_scorer"}
        ),
        "active-learning round source chain changed",
    )
    if transfer_scoped:
        baseline = chain.get("transfer_baseline")
        scorer = chain.get("source_scorer")
        baseline_fields = {
            "archive_sha256", "source_sha256", "source_size_bytes",
            "split_sha256", "source_groups_sha256", "feature_order_sha256",
            "row_count", "proposal_count", "train_groups_sha256",
            "test_groups_sha256",
        }
        scorer_fields = {
            "kind", "scorer_sha256", "scores_sha256", "pool_scores_sha256",
            "feature_state_sha256", "prototype_sha256",
            "pca_components_sha256", "pca_mean_sha256",
            "r92_classifier_sha256", "r92_resource_manifest_sha256",
            "preset_id",
        }
        _require(
            isinstance(baseline, dict)
            and set(baseline) == baseline_fields
            and baseline.get("feature_order_sha256")
            == CANONICAL_FEATURE_ORDER_SHA256
            and type(baseline.get("source_size_bytes")) is int
            and int(baseline["source_size_bytes"]) > 0
            and type(baseline.get("row_count")) is int
            and int(baseline["row_count"]) > 0
            and type(baseline.get("proposal_count")) is int
            and int(baseline["proposal_count"]) > 0
            and isinstance(scorer, dict)
            and set(scorer) == scorer_fields
            and scorer.get("kind")
            in {"PUBLISHED_R92_TRANSFER_SCORER", "PROJECT_TRANSFER_BUNDLE"}
            and scorer.get("scores_sha256") == predictions_sha256
            and scorer.get("pool_scores_sha256")
            == pool_result.get("predictions_sha256")
            and (
                (
                    round_number == 1
                    and scorer.get("kind") == "PUBLISHED_R92_TRANSFER_SCORER"
                    and scorer.get("preset_id") == "compag-cj-r92"
                )
                or (
                    round_number > 1
                    and scorer.get("kind") == "PROJECT_TRANSFER_BUNDLE"
                    and scorer.get("preset_id") is None
                )
            )
            and all(
                isinstance(value, str) and _SHA256.fullmatch(value) is not None
                for key, value in {**baseline, **scorer}.items()
                if key.endswith("sha256")
            ),
            "transfer round baseline or scorer receipt changed",
        )
    expected_round_id = compact_json_sha256(
        {
            "domain": (
                "compag-curation-transfer-image-round-id/v1"
                if transfer_scoped
                else
                "compag-curation-canonical-active-learning-image-round-id/v3"
                if image_scoped
                else "compag-curation-canonical-active-learning-round-id/v2"
            ),
            "round_number": round_number,
            "policy": _policy(),
            "chain": chain,
            "shortlist_sha256": shortlist_record["sha256"],
            **(
                {
                    "workflow": expected_workflow,
                    "image": manifest.get("image"),
                }
                if single_image_scoped
                else {}
            ),
        }
    )
    _require(round_id == expected_round_id, "active-learning round ID changed")
    raw_archive = inference.get("raw_feature_archive")
    _require(
        pool_result.get("bundle_sha256") == inference.get("bundle_sha256")
        and isinstance(raw_archive, dict)
        and pool_result.get("source_sha256") == raw_archive.get("sha256"),
        "round pool differs from the bound stage-60 inference",
    )
    pool_by_id = {row.proposal_id: row for row in pool_rows}
    _require(set(pool_by_id) == set(predictions), "round pool and stage-60 prediction ID sets differ")
    if single_image_scoped:
        image_identities = {
            (row.image_id, row.image_sha256, row.group_id)
            for row in pool_rows
        }
        _require(
            len(image_identities) == 1
            and manifest.get("image")
            == {
                "image_id": next(iter(image_identities))[0],
                "image_sha256": next(iter(image_identities))[1],
                "group_id": next(iter(image_identities))[2],
                "proposal_count": len(pool_rows),
            },
            "single-image round does not bind exactly one image",
        )
    for proposal_id, pool in pool_by_id.items():
        prediction = predictions[proposal_id]
        _require(
            pool.proposal_sha256 == prediction["proposal_sha256"]
            and pool.image_id == prediction["image_id"]
            and pool.image_sha256 == prediction["image_sha256"]
            and pool.group_id == prediction["group_id"]
            and pool.scale == 1.0
            and pool.xgb_p is not None
            and _format_float(float(pool.xgb_p)) == prediction["xgb_p"],
            f"round pool differs from stage-60 prediction: {proposal_id}",
        )
    request_by_id = {row["proposal_id"]: row for row in request_rows}
    _require(len(request_by_id) == len(request_rows), "round review request has duplicate proposal IDs")
    views: dict[str, RoundProposalView] = {}
    prior_sort: tuple[float, str] | None = None
    selected_ids: list[str] = []
    policy = _policy()
    for offset, raw in enumerate(shortlist, start=1):
        proposal_id = raw["proposal_id"]
        rank = _strict_int(raw["selection_rank"], "selection rank", minimum=1)
        _require(rank == offset and proposal_id not in views, "selection ranks are not unique and contiguous")
        pool = pool_by_id.get(proposal_id)
        prediction = predictions.get(proposal_id)
        request = request_by_id.get(proposal_id)
        _require(pool is not None and prediction is not None and request is not None, "selected proposal is absent from a bound source")
        identity = (proposal_id, raw["proposal_sha256"], raw["image_id"], raw["image_sha256"], raw["group_id"])
        _require(
            identity
            == (pool.proposal_id, pool.proposal_sha256, pool.image_id, pool.image_sha256, pool.group_id)
            == (
                prediction["proposal_id"], prediction["proposal_sha256"], prediction["image_id"],
                prediction["image_sha256"], prediction["group_id"],
            )
            == tuple(request[name] for name in ("proposal_id", "proposal_sha256", "image_id", "image_sha256", "group_id")),
            "selected proposal identity differs between sealed sources",
        )
        xgb_p = _strict_float(raw["xgb_p"], "selection xgb_p")
        distance = _strict_float(raw["distance_to_threshold"], "selection threshold distance")
        uncertainty = _strict_float(raw["uncertainty"], "selection uncertainty")
        expected_distance = abs(xgb_p - float(policy["threshold"]))
        expected_uncertainty = max(0.0, 1.0 - min(1.0, expected_distance / float(policy["margin"])))
        _require(
            raw["scale"] == "1"
            and raw["xgb_p"] == prediction["xgb_p"]
            and raw["distance_to_threshold"] == _format_float(expected_distance)
            and raw["uncertainty"] == _format_float(expected_uncertainty)
            and distance == expected_distance
            and uncertainty == expected_uncertainty
            and distance <= float(policy["margin"]),
            "selected proposal score or uncertainty changed",
        )
        sort_key = (distance, proposal_id)
        _require(prior_sort is None or prior_sort < sort_key, "selection order differs from canonical tie-breaking")
        prior_sort = sort_key
        try:
            polygon_value = json.loads(prediction["poly"])
        except json.JSONDecodeError as exc:  # Already checked by _prediction_rows; keep typing local.
            raise PublicIOError("selected prediction polygon changed") from exc
        view = RoundProposalView(
            proposal_id=proposal_id,
            proposal_sha256=raw["proposal_sha256"],
            image_id=raw["image_id"],
            image_sha256=raw["image_sha256"],
            image_name=prediction["full_image"],
            group_id=raw["group_id"],
            tile_name=prediction["full_image"],
            tile_index=offset - 1,
            proposal_index=int(prediction["id"]),
            label_prefix=proposal_id[:12],
            selection_rank=rank,
            xgb_p=xgb_p,
            distance_to_threshold=distance,
            uncertainty=uncertainty,
            prediction=int(prediction["prediction"]),
            kept=int(prediction["kept"]),
            bbox_x=int(prediction["x"]),
            bbox_y=int(prediction["y"]),
            bbox_w=int(prediction["w"]),
            bbox_h=int(prediction["h"]),
            original_width=int(prediction["orig_w"]),
            original_height=int(prediction["orig_h"]),
            polygon=tuple(int(item) for item in polygon_value),
        )
        views[proposal_id] = view
        selected_ids.append(proposal_id)
    _require(
        bool(views)
        and len(views) <= int(policy["top_k"])
        and set(request_by_id) == set(views),
        "round shortlist and review request ID sets differ",
    )
    counts = manifest.get("counts")
    count_fields = {
        "pool_rows", "reviewed_excluded", "frozen_test_group_excluded",
        "outside_margin_excluded", "margin_eligible", "selected",
    }
    _require(
        isinstance(counts, dict)
        and set(counts) == count_fields
        and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in counts.values())
        and counts["pool_rows"] == len(pool_rows)
        and counts["selected"] == len(views)
        and counts["reviewed_excluded"] + counts["frozen_test_group_excluded"]
        + counts["outside_margin_excluded"] + counts["margin_eligible"] == len(pool_rows)
        and counts["selected"] <= counts["margin_eligible"],
        "active-learning round selection counts changed",
    )
    order = tuple(selected_ids)
    tiles = tuple(RoundVisualView(index, views[proposal_id].image_name, (proposal_id,)) for index, proposal_id in enumerate(order))
    return manifest, RoundReviewIndex(
        request_rows=tuple(request_rows),
        request_columns=CANONICAL_REVIEW_COLUMNS,
        order=order,
        proposals=views,
        tiles=tiles,
        groups=tuple(sorted({view.group_id for view in views.values()})),
        review_request_sha256=str(request_meta["sha256"]),
        shortlist_sha256=str(shortlist_record["sha256"]),
        predictions_sha256=predictions_sha256,
        visual_order_sha256=compact_json_sha256(list(order)),
        scene_proposals=round_scene_proposals(predictions),
    ), str(manifest_record["sha256"])


def _verify_begin_operation(
    operation_root: Path,
    *,
    round_number: int | None | object = ...,
) -> tuple[Mapping[str, object], str]:
    errors: list[PublicIOError] = []
    for operation_name, mode in (
        ("begin_round", "standard"),
        ("begin_image_round", "image"),
        ("begin_transfer_image_round", "transfer"),
    ):
        try:
            if round_number is ...:
                operation = _verify_published_operation(
                    operation_root,
                    operation=operation_name,
                    status="PAUSED_FOR_REVIEW",
                    exit_code=3,
                )
            else:
                operation = _verify_published_operation(
                    operation_root,
                    operation=operation_name,
                    status="PAUSED_FOR_REVIEW",
                    exit_code=3,
                    round_number=round_number,
                )
            return operation, mode
        except PublicIOError as exc:
            errors.append(exc)
    raise PublicIOError(
        "selection root is not a sealed supported begin-round operation"
    ) from errors[-1]


def build_round_review_index(
    operation_root: Path,
    stage60_root: Path,
    image_root: Path,
) -> tuple[RoundSnapshot, RoundReviewIndex]:
    """Verify and index a sealed begin-round and its original image inventory."""

    operation, mode = _verify_begin_operation(operation_root)
    operation_root = Path(operation["root"])
    stage60_root = _safe_directory(stage60_root, "bound stage-60 root")
    expected_stage60 = _input_by_role(operation, "canonical_inference_root")
    stage60_record = _directory_record(stage60_root, "canonical_inference_root")
    _require(stage60_record == expected_stage60, "supplied stage-60 root differs from begin-round input")
    inference, predictions, expected_images, predictions_sha256, inference_result_sha256 = _inference_index(stage60_root)
    manifest, index, round_manifest_sha256 = _selection_index(
        operation_root,
        inference,
        predictions,
        predictions_sha256,
        image_scoped=mode == "image",
        transfer_scoped=mode == "transfer",
    )
    status = operation["status"]
    round_number = int(manifest["round_number"])
    _require(status.get("round_number") == round_number, "operation and selection round numbers differ")
    image_root = _safe_directory(image_root, "original inference image directory")
    images, image_inventory_sha256 = _image_inventory(image_root, expected_images)
    snapshot = RoundSnapshot(
        operation_root=operation_root,
        stage60_root=stage60_root,
        image_root=image_root,
        operation_id=str(status["operation_id"]),
        operation_manifest_sha256=str(operation["output_manifest_sha256"]),
        stage60_record=stage60_record,
        round_number=round_number,
        round_id=str(manifest["round_id"]),
        profile=str(manifest["profile"]),
        bundle_sha256=str(inference["bundle_sha256"]),
        inference_result_sha256=inference_result_sha256,
        predictions_sha256=predictions_sha256,
        review_request=operation_root / "payload/selection/review_request.csv",
        review_request_sha256=index.review_request_sha256,
        shortlist_sha256=index.shortlist_sha256,
        round_manifest_sha256=round_manifest_sha256,
        image_inventory=images,
        image_inventory_sha256=image_inventory_sha256,
    )
    return snapshot, index


def _decision_payload(decision: Decision | None) -> dict[str, object] | None:
    return None if decision is None else asdict(decision)


def _parse_decision(value: object) -> Decision | None:
    if value is None:
        return None
    _require(isinstance(value, dict) and set(value) == set(Decision.__dataclass_fields__), "round-review decision is malformed")
    choice = value.get("choice")
    _require(isinstance(choice, str) and choice in DECISIONS, "round-review choice is invalid")
    expected = DECISIONS[choice]
    _require(value == asdict(expected), "round-review decision contract changed")
    return expected


class RoundReviewSession:
    """Exclusive durable review state for one sealed active-learning round."""

    def __init__(
        self,
        snapshot: RoundSnapshot,
        index: RoundReviewIndex,
        state_root: Path,
        output: Path,
        lock_fd: int,
        session_sha256: str,
    ) -> None:
        self.snapshot = snapshot
        self.index = index
        self.state_root = state_root
        self.output = output
        self._lock_fd = lock_fd
        self._session_sha256 = session_sha256
        self._mutex = threading.RLock()
        self._decisions: dict[str, Decision] = {}
        self._undo_stack: list[dict[str, object]] = []
        self._requests: dict[str, tuple[object, ...]] = {}
        self._last_sequence = 0
        self._last_event_sha256 = "0" * 64
        self._final: dict[str, object] | None = None
        self._scene_catalog: object | None = None
        self._load_events()
        self._load_final()

    @property
    def canonical(self) -> bool:
        return True

    @classmethod
    def open(
        cls,
        operation_root: Path,
        stage60_root: Path,
        image_root: Path,
        output: Path,
        state_root: Path | None = None,
    ) -> "RoundReviewSession":
        snapshot, index = build_round_review_index(operation_root, stage60_root, image_root)
        roots = (snapshot.operation_root, snapshot.stage60_root, snapshot.image_root)
        output = _external_path(
            output, roots, "round-review output", derived_basename=f".{output.name}.compag-round-review.{uuid.uuid4()}.tmp"
        )
        requested_state = state_root or output.with_name(f"{output.name}.review-session")
        state_root = _external_path(
            requested_state, roots, "round-review state", derived_basename=f".{requested_state.name}.init.{uuid.uuid4()}"
        )
        _require(
            output != state_root and state_root not in output.parents and output not in state_root.parents,
            "round-review output and state paths overlap",
        )
        lock_fd = -1
        try:
            if state_root.exists() or state_root.is_symlink():
                _directory_is_private(state_root, "round-review state directory")
            else:
                _require(not output.exists() and not output.is_symlink(), "round-review output already exists")
                cls._create_state(snapshot, index, state_root, output)
            lock_path = state_root / "SESSION.LOCK"
            lock_fd = os.open(lock_path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PublicIOError("ROUND_REVIEW_SESSION_ALREADY_ACTIVE") from exc
            lock_info = os.fstat(lock_fd)
            _require(
                stat.S_ISREG(lock_info.st_mode)
                and lock_info.st_nlink == 1
                and stat.S_IMODE(lock_info.st_mode) == 0o600
                and lock_info.st_uid == os.getuid(),
                "round-review lock is unsafe",
            )
            session_sha256 = cls._validate_receipt(snapshot, index, state_root, output)
            return cls(snapshot, index, state_root, output, lock_fd, session_sha256)
        except BaseException:
            if lock_fd >= 0:
                os.close(lock_fd)
            raise

    @staticmethod
    def _receipt(
        snapshot: RoundSnapshot,
        index: RoundReviewIndex,
        output: Path,
        *,
        created_at_utc: str,
    ) -> dict[str, object]:
        return {
            "schema": SESSION_SCHEMA,
            "created_at_utc": created_at_utc,
            "source_kind": SOURCE_KIND,
            "operation_root": str(snapshot.operation_root),
            "stage60_root": str(snapshot.stage60_root),
            "image_root": str(snapshot.image_root),
            "operation_id": snapshot.operation_id,
            "round_number": snapshot.round_number,
            "round_id": snapshot.round_id,
            "profile": snapshot.profile,
            "source_binding_sha256": snapshot.binding_sha256,
            "review_request_sha256": snapshot.review_request_sha256,
            "proposal_count": len(index.order),
            "visual_order_sha256": index.visual_order_sha256,
            "output": str(output),
            "decision_contract": {choice: asdict(decision) for choice, decision in DECISIONS.items()},
            "state_policy": "APPEND_ONLY_HASH_CHAIN_OUTSIDE_SEALED_INPUTS",
            "output_policy": "COMPLETE_VALIDATED_NO_CLOBBER_NO_AUTO_RESUME",
        }

    @staticmethod
    def _head_record(
        snapshot: RoundSnapshot,
        index: RoundReviewIndex,
        session_sha256: str,
        sequence: int,
        event_sha256: str,
    ) -> dict[str, object]:
        return {
            "schema": EVENT_HEAD_SCHEMA,
            "source_binding_sha256": snapshot.binding_sha256,
            "visual_order_sha256": index.visual_order_sha256,
            "session_sha256": session_sha256,
            "sequence": sequence,
            "event_sha256": event_sha256,
        }

    @classmethod
    def _create_state(cls, snapshot: RoundSnapshot, index: RoundReviewIndex, state_root: Path, output: Path) -> None:
        _require(not state_root.exists() and not state_root.is_symlink(), "round-review state path exists")
        staging = state_root.parent / f".{state_root.name}.init.{uuid.uuid4()}"
        staging.mkdir(mode=0o700)
        (staging / "events").mkdir(mode=0o700)
        (staging / ".staging").mkdir(mode=0o700)
        write_new_bytes(staging / "SESSION.LOCK", b"COMPAG active-learning round review lock\n", mode=0o600)
        receipt = cls._receipt(snapshot, index, output, created_at_utc=_now())
        write_new_json(staging / "SESSION.json", receipt, mode=0o600)
        session_sha256 = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
        write_new_json(
            staging / "EVENT_HEAD.json",
            cls._head_record(snapshot, index, session_sha256, 0, "0" * 64),
            mode=0o600,
        )
        fsync_directory(staging / "events")
        fsync_directory(staging / ".staging")
        fsync_directory(staging)
        publish_directory_noreplace(staging, state_root)

    @classmethod
    def _validate_receipt(
        cls,
        snapshot: RoundSnapshot,
        index: RoundReviewIndex,
        state_root: Path,
        output: Path,
    ) -> str:
        _directory_is_private(state_root, "round-review state directory")
        _directory_is_private(state_root / "events", "round-review event directory")
        _directory_is_private(state_root / ".staging", "round-review staging directory")
        receipt_payload = _regular_private(
            state_root / "SESSION.json", "round-review session receipt", max_bytes=256 * 1024
        )
        value = canonical_json_value(
            receipt_payload,
            "round-review session receipt",
        )
        _require(isinstance(value, dict), "round-review session receipt schema changed")
        _require_timestamp(value.get("created_at_utc"), "round-review session creation time")
        expected = cls._receipt(
            snapshot,
            index,
            output,
            created_at_utc=str(value["created_at_utc"]),
        )
        _require(isinstance(value, dict) and set(value) == set(expected), "round-review session receipt schema changed")
        _require(all(value.get(key) == item for key, item in expected.items()), "round-review session binding changed")
        session_sha256 = hashlib.sha256(receipt_payload).hexdigest()
        head = canonical_json_value(
            _regular_private(
                state_root / "EVENT_HEAD.json",
                "round-review event head",
                max_bytes=256 * 1024,
            ),
            "round-review event head",
        )
        initial_fields = set(cls._head_record(snapshot, index, session_sha256, 0, "0" * 64))
        _require(
            isinstance(head, dict)
            and set(head) == initial_fields
            and head.get("schema") == EVENT_HEAD_SCHEMA
            and head.get("source_binding_sha256") == snapshot.binding_sha256
            and head.get("visual_order_sha256") == index.visual_order_sha256
            and head.get("session_sha256") == session_sha256,
            "round-review event head binding changed",
        )
        return session_sha256

    def close(self) -> None:
        with self._mutex:
            if self._lock_fd >= 0:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
                self._lock_fd = -1

    def __enter__(self) -> "RoundReviewSession":
        return self

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        self.close()

    def _event_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for path in (self.state_root / "events").iterdir():
            _require(
                len(names) < MAX_EVENT_COUNT,
                "round-review event directory exceeds its member-count bound",
            )
            names.append(path.name)
        names.sort()
        _require(
            all(_EVENT_NAME.fullmatch(name) is not None for name in names),
            "round-review event directory has an unexpected member",
        )
        return tuple(names)

    def _read_event_head(self) -> dict[str, object]:
        value = canonical_json_value(
            _regular_private(
                self.state_root / "EVENT_HEAD.json",
                "round-review event head",
                max_bytes=256 * 1024,
            ),
            "round-review event head",
        )
        expected_fields = set(
            self._head_record(
                self.snapshot,
                self.index,
                self._session_sha256,
                0,
                "0" * 64,
            )
        )
        _require(
            isinstance(value, dict)
            and set(value) == expected_fields
            and value.get("schema") == EVENT_HEAD_SCHEMA
            and value.get("source_binding_sha256") == self.snapshot.binding_sha256
            and value.get("visual_order_sha256") == self.index.visual_order_sha256
            and value.get("session_sha256") == self._session_sha256
            and type(value.get("sequence")) is int
            and 0 <= int(value["sequence"]) <= MAX_EVENT_COUNT
            and isinstance(value.get("event_sha256"), str)
            and _SHA256.fullmatch(str(value["event_sha256"])) is not None,
            "round-review event head is invalid or has changed binding",
        )
        return dict(value)

    def _advance_event_head(
        self,
        expected: Mapping[str, object],
        sequence: int,
        event_sha256: str,
    ) -> None:
        _require(
            self._read_event_head() == dict(expected),
            "round-review event head changed before advancement",
        )
        target = self.state_root / "EVENT_HEAD.json"
        staged = self.state_root / ".staging" / f"event-head.{uuid.uuid4()}.json"
        next_head = self._head_record(
            self.snapshot,
            self.index,
            self._session_sha256,
            sequence,
            event_sha256,
        )
        write_new_json(staged, next_head, mode=0o600)
        try:
            _require(
                self._read_event_head() == dict(expected),
                "round-review event head changed during advancement",
            )
            os.replace(staged, target)
            fsync_directory(self.state_root)
            fsync_directory(staged.parent)
            _require(
                self._read_event_head() == next_head,
                "round-review event head advancement was not durable",
            )
        except BaseException:
            if staged.exists() and not staged.is_symlink():
                staged.unlink()
                fsync_directory(staged.parent)
            raise

    def _replay_events(
        self,
        *,
        recover_ahead: bool,
    ) -> tuple[
        dict[str, Decision],
        list[dict[str, object]],
        dict[str, tuple[object, ...]],
        int,
        str,
    ]:
        names = self._event_names()
        head = self._read_event_head()
        head_sequence = int(head["sequence"])
        _require(
            head_sequence <= len(names),
            "round-review event head is ahead of the durable event directory",
        )
        decisions: dict[str, Decision] = {}
        undo_stack: list[dict[str, object]] = []
        requests: dict[str, tuple[object, ...]] = {}
        last_sequence = 0
        last_event_sha256 = "0" * 64
        hashes = [last_event_sha256]
        for expected_sequence, name in enumerate(names, start=1):
            match = _EVENT_NAME.fullmatch(name)
            _require(match is not None and int(match.group(1)) == expected_sequence, "round-review event sequence has a gap")
            event = canonical_json_value(
                _regular_private(self.state_root / "events" / name, "round-review event", max_bytes=MAX_EVENT_BYTES),
                "round-review event",
            )
            _require(isinstance(event, dict) and event.get("schema") == EVENT_SCHEMA, "round-review event schema changed")
            supplied_hash = event.get("event_sha256")
            without_hash = {key: value for key, value in event.items() if key != "event_sha256"}
            _require(
                event.get("sequence") == expected_sequence
                and event.get("previous_event_sha256") == last_event_sha256
                and supplied_hash == compact_json_sha256(without_hash),
                "round-review event hash chain is invalid",
            )
            proposal_id = event.get("proposal_id")
            request_id = event.get("request_id")
            _require(proposal_id in self.index.proposals, "round-review event references an unknown proposal")
            _require(isinstance(request_id, str) and _REQUEST_ID.fullmatch(request_id) and request_id not in requests, "round-review request ID is invalid")
            operation = event.get("operation")
            if operation == "SET":
                _require(
                    set(event) == {"schema", "sequence", "previous_event_sha256", "timestamp_utc", "operation", "request_id", "proposal_id", "previous_decision", "decision", "event_sha256"},
                    "round-review SET event fields changed",
                )
                previous = _parse_decision(event.get("previous_decision"))
                decision = _parse_decision(event.get("decision"))
                _require(decision is not None and decisions.get(proposal_id) == previous, "round-review SET previous state differs")
                decisions[str(proposal_id)] = decision
                undo_stack.append({"sequence": expected_sequence, "proposal_id": proposal_id, "previous": previous})
                requests[request_id] = ("SET", proposal_id, decision.choice)
            elif operation == "UNDO":
                _require(
                    set(event) == {"schema", "sequence", "previous_event_sha256", "timestamp_utc", "operation", "request_id", "proposal_id", "target_sequence", "decision", "event_sha256"}
                    and bool(undo_stack),
                    "round-review UNDO event fields changed",
                )
                target = undo_stack.pop()
                restored = _parse_decision(event.get("decision"))
                _require(event.get("target_sequence") == target["sequence"] and proposal_id == target["proposal_id"] and restored == target["previous"], "round-review UNDO target differs")
                if restored is None:
                    decisions.pop(str(proposal_id), None)
                else:
                    decisions[str(proposal_id)] = restored
                requests[request_id] = ("UNDO", proposal_id, target["sequence"])
            else:
                raise PublicIOError("round-review event operation is invalid")
            last_sequence = expected_sequence
            last_event_sha256 = str(supplied_hash)
            hashes.append(last_event_sha256)
        _require(
            str(head["event_sha256"]) == hashes[head_sequence],
            "round-review event head hash does not match its durable sequence",
        )
        _require(
            self._event_names() == names and self._read_event_head() == head,
            "round-review event state changed during replay",
        )
        if head_sequence < last_sequence:
            _require(recover_ahead, "round-review event head is behind the durable event directory")
            self._advance_event_head(head, last_sequence, last_event_sha256)
            _require(self._event_names() == names, "round-review event directory changed during recovery")
        return decisions, undo_stack, requests, last_sequence, last_event_sha256

    def _load_events(self) -> None:
        (
            self._decisions,
            self._undo_stack,
            self._requests,
            self._last_sequence,
            self._last_event_sha256,
        ) = self._replay_events(recover_ahead=True)

    def _verify_durable_events(self) -> None:
        decisions, undo_stack, requests, sequence, event_sha256 = self._replay_events(
            recover_ahead=False
        )
        _require(
            decisions == self._decisions
            and undo_stack == self._undo_stack
            and requests == self._requests
            and sequence == self._last_sequence
            and event_sha256 == self._last_event_sha256,
            "durable round-review events differ from in-memory review state",
        )

    def _append(self, event: dict[str, object]) -> dict[str, object]:
        sequence = self._last_sequence + 1
        _require(sequence <= MAX_EVENT_COUNT, "round-review event-count limit reached")
        base = {"schema": EVENT_SCHEMA, "sequence": sequence, "previous_event_sha256": self._last_event_sha256, "timestamp_utc": _now(), **event}
        base["event_sha256"] = compact_json_sha256(base)
        payload = canonical_json_bytes(base)
        _require(len(payload) <= MAX_EVENT_BYTES, "round-review event is too large")
        staged = self.state_root / ".staging" / f"event.{sequence:08d}.{uuid.uuid4()}.json"
        target = self.state_root / "events" / f"{sequence:08d}.json"
        write_new_bytes(staged, payload, mode=0o600)
        rename_noreplace(staged, target)
        head = self._read_event_head()
        _require(
            head["sequence"] == self._last_sequence
            and head["event_sha256"] == self._last_event_sha256,
            "round-review event head differs before event commit",
        )
        self._advance_event_head(head, sequence, str(base["event_sha256"]))
        self._last_sequence = sequence
        self._last_event_sha256 = str(base["event_sha256"])
        return base

    def _require_mutable(self) -> None:
        _require(self._lock_fd >= 0, "round-review session is closed")
        _require(self._final is None, "finalized round-review session cannot change")
        final_path = self.state_root / "FINAL.json"
        _require(not self.output.exists() and not self.output.is_symlink() and not final_path.exists() and not final_path.is_symlink(), "round-review publication exists or is uncertain")

    def set_decision(self, proposal_id: str, choice: str, *, request_id: str | None = None) -> dict[str, object]:
        with self._mutex:
            self._require_mutable()
            _require(proposal_id in self.index.proposals, "unknown proposal ID")
            _require(choice in DECISIONS, "unknown human-review choice")
            request_id = request_id or str(uuid.uuid4())
            _require(_REQUEST_ID.fullmatch(request_id) is not None, "round-review request ID is invalid")
            if request_id in self._requests:
                _require(self._requests[request_id] == ("SET", proposal_id, choice), "round-review request ID was reused")
                return self.proposal_payload(proposal_id)
            decision = DECISIONS[choice]
            previous = self._decisions.get(proposal_id)
            event = self._append({"operation": "SET", "request_id": request_id, "proposal_id": proposal_id, "previous_decision": _decision_payload(previous), "decision": _decision_payload(decision)})
            self._decisions[proposal_id] = decision
            self._undo_stack.append({"sequence": event["sequence"], "proposal_id": proposal_id, "previous": previous})
            self._requests[request_id] = ("SET", proposal_id, choice)
            return self.proposal_payload(proposal_id)

    def undo(self, *, request_id: str | None = None) -> dict[str, object]:
        with self._mutex:
            self._require_mutable()
            request_id = request_id or str(uuid.uuid4())
            _require(_REQUEST_ID.fullmatch(request_id) is not None, "round-review request ID is invalid")
            if request_id in self._requests:
                recorded = self._requests[request_id]
                _require(recorded[0] == "UNDO", "round-review request ID was reused")
                return self.proposal_payload(str(recorded[1]))
            _require(bool(self._undo_stack), "there is no round-review action to undo")
            target = self._undo_stack[-1]
            proposal_id = str(target["proposal_id"])
            restored = target["previous"]
            self._append({"operation": "UNDO", "request_id": request_id, "proposal_id": proposal_id, "target_sequence": target["sequence"], "decision": _decision_payload(restored if isinstance(restored, Decision) else None)})
            self._undo_stack.pop()
            if restored is None:
                self._decisions.pop(proposal_id, None)
            else:
                _require(isinstance(restored, Decision), "round-review undo restoration is invalid")
                self._decisions[proposal_id] = restored
            self._requests[request_id] = ("UNDO", proposal_id, target["sequence"])
            return self.proposal_payload(proposal_id)

    def _counts(self) -> dict[str, int]:
        counts = {choice: 0 for choice in DECISIONS}
        for decision in self._decisions.values():
            counts[decision.choice] += 1
        return counts

    def summary(self) -> dict[str, object]:
        with self._mutex:
            total = len(self.index.order)
            reviewed = len(self._decisions)
            first_pending = next((proposal_id for proposal_id in self.index.order if proposal_id not in self._decisions), None)
            return {
                "schema": STATUS_SCHEMA,
                "source_kind": SOURCE_KIND,
                "status": "FINALIZED" if self._final is not None else ("READY_TO_FINALIZE" if reviewed == total else "IN_PROGRESS"),
                "operation_id": self.snapshot.operation_id,
                "round_number": self.snapshot.round_number,
                "round_id": self.snapshot.round_id,
                "profile": self.snapshot.profile,
                "bundle_sha256": self.snapshot.bundle_sha256,
                "total": total,
                "reviewed": reviewed,
                "remaining": total - reviewed,
                "progress_percent": round(100.0 * reviewed / total, 2),
                "counts": self._counts(),
                "groups": list(self.index.groups),
                "first_pending": first_pending,
                "event_sequence": self._last_sequence,
                "output": str(self.output),
                "state_root": str(self.state_root),
                "canonical_actions": True,
                "final": self._final,
            }

    @staticmethod
    def _crop(view: RoundProposalView) -> tuple[int, int, int, int]:
        context = max(view.bbox_w, view.bbox_h, 32)
        side = max(context * 2, 128)
        crop_w = min(view.original_width, side)
        crop_h = min(view.original_height, side)
        center_x = view.bbox_x + view.bbox_w / 2.0
        center_y = view.bbox_y + view.bbox_h / 2.0
        x0 = max(0, min(view.original_width - crop_w, int(round(center_x - crop_w / 2.0))))
        y0 = max(0, min(view.original_height - crop_h, int(round(center_y - crop_h / 2.0))))
        return x0, y0, crop_w, crop_h

    @classmethod
    def _visual_geometry(cls, view: RoundProposalView) -> dict[str, object]:
        x0, y0, crop_w, crop_h = cls._crop(view)
        sx, sy = VISUAL_SIZE / crop_w, VISUAL_SIZE / crop_h
        bbox = [
            max(0, min(VISUAL_SIZE - 1, int(round((view.bbox_x - x0) * sx)))),
            max(0, min(VISUAL_SIZE - 1, int(round((view.bbox_y - y0) * sy)))),
            max(1, min(VISUAL_SIZE, int(round(view.bbox_w * sx)))),
            max(1, min(VISUAL_SIZE, int(round(view.bbox_h * sy)))),
        ]
        bbox[2] = min(bbox[2], VISUAL_SIZE - bbox[0])
        bbox[3] = min(bbox[3], VISUAL_SIZE - bbox[1])
        polygon: list[int] = []
        for x, y in zip(view.polygon[0::2], view.polygon[1::2]):
            polygon.extend(
                [
                    max(0, min(VISUAL_SIZE - 1, int(round((x - x0) * sx)))),
                    max(0, min(VISUAL_SIZE - 1, int(round((y - y0) * sy)))),
                ]
            )
        return {"width": VISUAL_SIZE, "height": VISUAL_SIZE, "source_crop": [x0, y0, crop_w, crop_h], "bbox": bbox, "polygon": polygon}

    def proposal_payload(self, proposal_id: str) -> dict[str, object]:
        with self._mutex:
            _require(proposal_id in self.index.proposals, "unknown proposal ID")
            view = self.index.proposals[proposal_id]
            decision = self._decisions.get(proposal_id)
            geometry = self._visual_geometry(view)
            visual_proposal = {"proposal_id": proposal_id, "label_prefix": view.label_prefix, "bbox": geometry["bbox"], "polygon": geometry["polygon"], "choice": None if decision is None else decision.choice}
            visual = {
                "index": view.tile_index,
                "kind": "VERIFIED_ORIGINAL_IMAGE_CROP",
                "tile_name": view.image_name,
                "image_name": view.image_name,
                "width": VISUAL_SIZE,
                "height": VISUAL_SIZE,
                "source_crop": geometry["source_crop"],
                "proposals": [visual_proposal],
                "model": {
                    "profile": self.snapshot.profile,
                    "bundle_sha256": self.snapshot.bundle_sha256,
                    "xgb_p": view.xgb_p,
                    "threshold": CANONICAL_DECISION_THRESHOLD,
                    "prediction": view.prediction,
                    "kept": view.kept,
                    "uncertainty": view.uncertainty,
                    "selection_rank": view.selection_rank,
                },
            }
            return {
                "proposal": {**asdict(view), "decision": _decision_payload(decision), "position": self.index.order.index(proposal_id) + 1, "total": len(self.index.order)},
                "visual": visual,
                "tile": visual,
                "summary": self.summary(),
            }

    def navigate(self, current: str | None, direction: int, *, state_filter: str = "all", group: str | None = None) -> dict[str, object]:
        with self._mutex:
            _require(direction in {-1, 1}, "navigation direction is invalid")
            _require(state_filter in {"all", "pending", "reviewed"}, "navigation filter is invalid")
            if group is not None:
                _require(group in self.index.groups, "navigation group is invalid")
            candidates = [
                proposal_id for proposal_id in self.index.order
                if (group is None or self.index.proposals[proposal_id].group_id == group)
                and (state_filter == "all" or (proposal_id in self._decisions) == (state_filter == "reviewed"))
            ]
            _require(bool(candidates), "no proposals match the current filter")
            selected = candidates[0] if current not in candidates and direction > 0 else candidates[-1] if current not in candidates else candidates[(candidates.index(str(current)) + direction) % len(candidates)]
            return self.proposal_payload(selected)

    def search(self, prefix: str) -> dict[str, object]:
        value = prefix.strip().lower()
        _require(re.fullmatch(r"[0-9a-f]{4,64}", value) is not None, "search requires 4-64 hexadecimal proposal-ID characters")
        matches = [proposal_id for proposal_id in self.index.order if proposal_id.startswith(value)]
        _require(len(matches) == 1, "proposal-ID search must match exactly one proposal")
        return self.proposal_payload(matches[0])

    def visual_bytes(self, visual_index: int) -> bytes:
        with self._mutex:
            _require(0 <= visual_index < len(self.index.tiles), "unknown round-review visual")
            proposal_id = self.index.tiles[visual_index].proposal_ids[0]
            view = self.index.proposals[proposal_id]
            source = self.snapshot.images_by_name[view.image_name]
            payload, info = stable_file(source.path, max_bytes=MAX_IMAGE_FILE_BYTES)
            _require(
                stat.S_ISREG(info.st_mode)
                and info.st_nlink == 1
                and not source.path.is_symlink()
                and len(payload) == source.size_bytes
                and hashlib.sha256(payload).hexdigest() == source.image_sha256,
                "original image changed after round-review indexing",
            )
            try:
                from PIL import Image, ImageDraw
            except ImportError as exc:
                raise PublicIOError("round-review visual rendering requires Pillow") from exc
            try:
                with Image.open(io.BytesIO(payload)) as opened:
                    opened.load()
                    image = opened.convert("RGB")
            except Exception as exc:
                raise PublicIOError("original image could not be decoded for round review") from exc
            _require(image.size == (source.width, source.height), "original image decode differs from sealed dimensions")
            x0, y0, crop_w, crop_h = self._crop(view)
            crop = image.crop((x0, y0, x0 + crop_w, y0 + crop_h))
            resampling = Image.Resampling.LANCZOS if max(crop_w, crop_h) > VISUAL_SIZE else Image.Resampling.BILINEAR
            rendered = crop.resize((VISUAL_SIZE, VISUAL_SIZE), resampling)
            draw = ImageDraw.Draw(rendered)
            geometry = self._visual_geometry(view)
            bx, by, bw, bh = geometry["bbox"]
            draw.rectangle(
                (bx, by, min(VISUAL_SIZE - 1, bx + bw - 1), min(VISUAL_SIZE - 1, by + bh - 1)),
                outline=(255, 165, 0),
                width=3,
            )
            polygon = geometry["polygon"]
            if len(polygon) >= 6:
                points = list(zip(polygon[0::2], polygon[1::2]))
                draw.line([*points, points[0]], fill=(64, 220, 64), width=3, joint="curve")
            label = f"rank {view.selection_rank}  p={view.xgb_p:.3f}  pred={view.prediction}  kept={view.kept}"
            draw.rectangle((0, 0, VISUAL_SIZE - 1, 27), fill=(0, 0, 0))
            draw.text((8, 7), label, fill=(255, 255, 255))
            encoded = io.BytesIO()
            rendered.save(encoded, format="PNG", optimize=False, compress_level=3)
            result = encoded.getvalue()
            _require(len(result) <= 8 * 1024 * 1024 and result.startswith(b"\x89PNG\r\n\x1a\n"), "round-review visual output is invalid")
            return result

    def overlay_bytes(self, tile_index: int) -> bytes:
        """Compatibility alias for existing review clients."""

        return self.visual_bytes(tile_index)

    def _full_image_scene_catalog(self) -> Any:
        if self._scene_catalog is None:
            from compag_curation.review.scene import RoundSceneCatalog

            self._scene_catalog = RoundSceneCatalog.open(
                self.snapshot.stage60_root,
                predictions_sha256=self.snapshot.predictions_sha256,
                inference_result_sha256=self.snapshot.inference_result_sha256,
                images=self.snapshot.image_inventory,
                proposals=self.index.scene_proposals,
                reviewable_order=self.index.order,
                profile=self.snapshot.profile,
                bundle_sha256=self.snapshot.bundle_sha256,
            )
        return self._scene_catalog

    def scene_payload(self, proposal_id: str) -> dict[str, object]:
        """Return a shortlisted proposal in its complete Stage-60 image scene."""

        with self._mutex:
            payload = self.proposal_payload(proposal_id)
            choices = {
                item_id: (
                    None
                    if self._decisions.get(item_id) is None
                    else self._decisions[item_id].choice
                )
                for item_id in self.index.order
            }
            payload["scene"] = self._full_image_scene_catalog().payload(
                proposal_id,
                choices,
            )
            return payload

    def scene_bytes(self, scene_id: str) -> bytes:
        """Render the verified original full image as deterministic PNG."""

        with self._mutex:
            return self._full_image_scene_catalog().bytes(scene_id)

    def _render_csv(self) -> bytes:
        _require(len(self._decisions) == len(self.index.request_rows), "round review is incomplete")
        with io.StringIO(newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.index.request_columns, lineterminator="\n")
            writer.writeheader()
            for source in self.index.request_rows:
                decision = self._decisions[source["proposal_id"]]
                _require(decision.review_action in REVIEW_ACTION_WEIGHTS and decision.review_weight == REVIEW_ACTION_WEIGHTS[decision.review_action], "round-review action weight changed")
                row = {name: source[name] for name in self.index.request_columns}
                row.update({"label": str(decision.label), "review_action": str(decision.review_action), "review_weight": format(float(decision.review_weight), ".1f"), "review_status": "reviewed"})
                writer.writerow(row)
            return handle.getvalue().encode("utf-8")

    def _revalidate_inputs(self) -> None:
        operation, _mode = _verify_begin_operation(
            self.snapshot.operation_root,
            round_number=self.snapshot.round_number,
        )
        _require(
            operation["status"]["operation_id"] == self.snapshot.operation_id
            and operation["output_manifest_sha256"] == self.snapshot.operation_manifest_sha256
            and _input_by_role(operation, "canonical_inference_root") == self.snapshot.stage60_record
            and _directory_record(self.snapshot.stage60_root, "canonical_inference_root") == self.snapshot.stage60_record,
            "sealed round or stage-60 input changed before publication",
        )
        images, digest = _image_inventory(
            self.snapshot.image_root,
            [
                {
                    **item.receipt(),
                    "tile_count": 1,
                    "candidate_rows": sum(view.image_id == item.image_id for view in self.index.proposals.values()),
                }
                for item in self.snapshot.image_inventory
            ],
        )
        _require(
            digest == self.snapshot.image_inventory_sha256
            and tuple(item.receipt() for item in images) == tuple(item.receipt() for item in self.snapshot.image_inventory),
            "original image inventory changed before publication",
        )

    def _final_record(self, validation: Mapping[str, object], *, recovery: bool) -> dict[str, object]:
        return {
            "schema": FINAL_SCHEMA,
            "status": "PASS",
            "finalized_at_utc": _now(),
            "recovered_after_output_publication": recovery,
            "source_kind": SOURCE_KIND,
            "operation_id": self.snapshot.operation_id,
            "round_number": self.snapshot.round_number,
            "round_id": self.snapshot.round_id,
            "source_binding_sha256": self.snapshot.binding_sha256,
            "review_request_sha256": self.snapshot.review_request_sha256,
            "reviewed_sha256": sha256_file(self.output),
            "rows": validation["rows"],
            "positive_rows": validation["positive_rows"],
            "negative_rows": validation["negative_rows"],
            "action_counts": validation["action_counts"],
            "effective_weight": validation["effective_weight"],
            "event_sequence": self._last_sequence,
            "event_sha256": self._last_event_sha256,
            "output": str(self.output),
            "input_policy": "SEALED_BEGIN_ROUND_STAGE60_AND_ORIGINAL_IMAGES_REVALIDATED",
            "publication_policy": "VALIDATED_NO_CLOBBER_NO_AUTO_RESUME",
        }

    def _load_final(self) -> None:
        final_path = self.state_root / "FINAL.json"
        if not final_path.exists() and not final_path.is_symlink():
            if not self.output.exists() and not self.output.is_symlink():
                return
            _require(len(self._decisions) == len(self.index.order), "round-review output exists without receipt while incomplete")
            output_payload = _regular_private(self.output, "round-review output", max_bytes=128 * 1024 * 1024)
            _require(output_payload == self._render_csv(), "round-review output differs from durable decisions")
            self._revalidate_inputs()
            self._verify_durable_events()
            _rows, validation = validate_review_table(self.snapshot.review_request, self.output)
            final = self._final_record(validation, recovery=True)
            write_new_json(final_path, final, mode=0o600)
            self._final = final
            return
        value = canonical_json_value(
            _regular_private(final_path, "round-review final receipt", max_bytes=256 * 1024),
            "round-review final receipt",
        )
        _require(
            isinstance(value, dict)
            and set(value) == _FINAL_FIELDS
            and value.get("schema") == FINAL_SCHEMA
            and value.get("status") == "PASS"
            and value.get("source_kind") == SOURCE_KIND
            and value.get("operation_id") == self.snapshot.operation_id
            and value.get("round_number") == self.snapshot.round_number
            and value.get("round_id") == self.snapshot.round_id
            and value.get("source_binding_sha256") == self.snapshot.binding_sha256
            and value.get("review_request_sha256") == self.snapshot.review_request_sha256
            and value.get("event_sequence") == self._last_sequence
            and value.get("event_sha256") == self._last_event_sha256
            and value.get("output") == str(self.output),
            "round-review final receipt binding changed",
        )
        _require_timestamp(value.get("finalized_at_utc"), "round-review finalization time")
        _require(type(value.get("recovered_after_output_publication")) is bool, "round-review recovery marker is invalid")
        output_payload = _regular_private(self.output, "round-review output", max_bytes=128 * 1024 * 1024)
        _require(output_payload == self._render_csv(), "final round-review CSV differs from durable decisions")
        _rows, validation = validate_review_table(self.snapshot.review_request, self.output)
        expected = {
            "reviewed_sha256": validation["reviewed_sha256"],
            "rows": validation["rows"],
            "positive_rows": validation["positive_rows"],
            "negative_rows": validation["negative_rows"],
            "action_counts": validation["action_counts"],
            "effective_weight": validation["effective_weight"],
            "input_policy": "SEALED_BEGIN_ROUND_STAGE60_AND_ORIGINAL_IMAGES_REVALIDATED",
            "publication_policy": "VALIDATED_NO_CLOBBER_NO_AUTO_RESUME",
        }
        _require(hashlib.sha256(output_payload).hexdigest() == validation["reviewed_sha256"] and all(value.get(key) == item for key, item in expected.items()), "round-review final receipt differs from CSV")
        self._final = dict(value)

    def finalize(self) -> dict[str, object]:
        with self._mutex:
            _require(self._lock_fd >= 0, "round-review session is closed")
            if self._final is not None:
                return dict(self._final)
            final_path = self.state_root / "FINAL.json"
            if self.output.exists() or self.output.is_symlink() or final_path.exists() or final_path.is_symlink():
                self._load_final()
                _require(self._final is not None, "round-review publication recovery failed")
                return dict(self._final)
            _require(len(self._decisions) == len(self.index.order), "round review cannot finalize while proposals remain pending")
            staged = self.output.parent / f".{self.output.name}.compag-round-review.{uuid.uuid4()}.tmp"
            write_new_bytes(staged, self._render_csv(), mode=0o600)
            try:
                _rows, validation = validate_review_table(self.snapshot.review_request, staged)
                # This is deliberately the last potentially expensive operation
                # before the no-clobber publication boundary.
                self._revalidate_inputs()
                # Re-read the independent durable journal after all expensive
                # validation and immediately before publishing its projection.
                self._verify_durable_events()
                publish_file_noreplace(staged, self.output)
            except BaseException:
                if staged.exists() and not staged.is_symlink():
                    staged.unlink()
                    fsync_directory(staged.parent)
                raise
            final = self._final_record(validation, recovery=False)
            try:
                write_new_json(final_path, final, mode=0o600)
            except BaseException:
                try:
                    self._load_final()
                except BaseException:
                    pass
                raise
            self._final = final
            return dict(final)


__all__ = [
    "RoundImage",
    "RoundProposalView",
    "RoundReviewIndex",
    "RoundReviewSession",
    "RoundSnapshot",
    "RoundVisualView",
    "build_round_review_index",
]
