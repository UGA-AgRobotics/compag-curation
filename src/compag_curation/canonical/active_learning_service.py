"""Production adapters between canonical inference, AL, and bundle training."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from compag_curation.public_io import (
    PublicIOError,
    canonical_json_value,
    compact_json_sha256,
    manifest_rows,
    portable_basename,
    stable_file,
    write_new_bytes,
)
from compag_curation.public_config import group_id_from_name
from compag_curation.review.exchange import REVIEW_ACTION_WEIGHTS

from .active_learning import (
    CanonicalActiveLearningDecision,
    CanonicalActiveLearningPoolRow,
    CanonicalActiveLearningRetrainRequest,
    CanonicalActiveLearningTrainingRow,
    CanonicalTransferBaselineReference,
    CanonicalTransferRetrainRequest,
    CanonicalTransferScorerReference,
    TRANSFER_BASELINE_ACTION_ADAPTER_POLICY,
    _canonical_device_for_profile,
    _canonical_profile_device,
    _expected_bundle_provenance,
    _feature_archive_hash,
    _training_row_hash,
    _validate_transfer_al_rows,
    _validate_raw_feature,
    write_canonical_active_learning_pool,
)
from .features import (
    CANONICAL_RAW_FEATURE_ORDER,
    CanonicalRawFeature,
    canonical_raw_feature_csv_row,
    decode_canonical_embedding,
    encode_canonical_embedding,
    finalize_canonical_feature,
    validate_canonical_feature_state,
)
from .serialization import (
    portable_classifier_bytes,
    serialize_canonical_feature_state,
    verify_probability_parity,
)
from .spec import (
    CANONICAL_DECISION_THRESHOLD,
    CANONICAL_FEATURE_CROP_SCALES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_FULL_IMAGE_NMS_IOU,
    CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
    CANONICAL_GPU_PROFILE,
    CANONICAL_PROFILE,
    FULL_IMAGE_GPU_PROFILE,
    GPU_EXECUTION_PROFILES,
    V2_PIPELINE_PROFILES,
    inference_amg_execution_points_per_batch,
)
from .training import (
    CanonicalGroupSplit,
    CanonicalTrainingConfig,
    train_canonical_xgb,
)


CANONICAL_AL_RAW_FEATURE_ARCHIVE_SCHEMA = (
    "compag-curation-canonical-active-learning-raw-features/v1"
)
R92_TRANSFER_INFERENCE_SCHEMA = "compag-curation-r92-transfer-image-inference/v1"
R92_TRANSFER_INFERENCE_WORKFLOW = "SINGLE_NEW_IMAGE_POST_R92_TRANSFER_V1"
R92_TRANSFER_LINEAGE_POLICY = "POST_R92_REVIEWED_TRANSFER_BASELINE"
R92_TRANSFER_REPRODUCTION_CLAIM = "NOT_R92_REPRODUCTION"
_INFERENCE_SCHEMA = "compag-curation-canonical-inference/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GROUP = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_TABLE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_TABLE_ROWS = 2_000_000
_RAW_IDENTITY_COLUMNS = (
    "proposal_id",
    "proposal_sha256",
    "image_id",
    "image_sha256",
    "image_name",
    "group_id",
    "tile_name",
    "tile_sha256",
    "tile_x",
    "tile_y",
    "mask_sha256",
)
_FINAL_FEATURE_INDEX = {
    name: index for index, name in enumerate(CANONICAL_FEATURE_ORDER)
}
_INFERENCE_RESULT_FIELDS = {
    "schema",
    "status",
    "profile",
    "device",
    "bundle_sha256",
    "feature_order_sha256",
    "threshold",
    "threshold_method",
    "full_image_nms_iou",
    "sam2_execution_points_per_batch",
    "image_count",
    "tile_count",
    "proposal_count",
    "prediction_rows",
    "positive_rows",
    "kept_rows",
    "predictions_sha256",
    "raw_feature_archive",
    "input_inventory",
    "paper_result_reproduction",
}


@dataclass(frozen=True)
class CanonicalActiveLearningInferenceArchive:
    pool_rows: tuple[CanonicalActiveLearningPoolRow, ...]
    feature_rows: tuple[CanonicalActiveLearningTrainingRow, ...]
    bundle_sha256: str
    predictions_sha256: str
    raw_features_sha256: str
    feature_archive_sha256: str
    inference_result_sha256: str
    profile: str = CANONICAL_PROFILE
    device: str = "cpu"

    def __post_init__(self) -> None:
        _canonical_profile_device(self.profile, self.device)


@dataclass(frozen=True)
class CanonicalR92TransferInference:
    """One ordinary Stage-60 archive bound to the explicit r92 scorer."""

    inference: CanonicalActiveLearningInferenceArchive
    scorer: CanonicalTransferScorerReference
    feature_state: Any
    receipt: Mapping[str, object]


@dataclass(frozen=True)
class CanonicalTransferTrainingResult:
    """Fresh classifier fit plus honestly named projection-variance evidence."""

    training: Any
    observed_train_projection_variance: tuple[float, ...]
    baseline_feature_rows: int
    reviewed_baseline_feature_rows: int
    unreviewed_baseline_feature_rows: int
    cumulative_al_feature_rows: int
    effective_training_feature_rows: int
    positive_training_feature_rows: int
    heldout_parity_rows: tuple[Mapping[str, float], ...]
    baseline_action_adapter_policy: str = TRANSFER_BASELINE_ACTION_ADAPTER_POLICY


class _TupleFeatureMapping(Mapping[str, float]):
    """Low-overhead exact-order Mapping view over one finalized feature tuple."""

    __slots__ = ("_values",)

    def __init__(self, values: Sequence[float]) -> None:
        fixed = tuple(float(value) for value in values)
        if len(fixed) != len(CANONICAL_FEATURE_ORDER) or any(
            not math.isfinite(value) for value in fixed
        ):
            raise PublicIOError("transfer baseline finalized features are invalid")
        self._values = fixed

    def __iter__(self):
        return iter(CANONICAL_FEATURE_ORDER)

    def __len__(self) -> int:
        return len(CANONICAL_FEATURE_ORDER)

    def __getitem__(self, name: str) -> float:
        try:
            index = _FINAL_FEATURE_INDEX[name]
        except KeyError as exc:
            raise KeyError(name) from exc
        return self._values[index]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


def _sha256(value: object, role: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{role} is not a lowercase SHA-256",
    )
    return value


def _canonical_image_identity(name: str, image_sha256: str) -> tuple[str, str]:
    portable_basename(name, "canonical image name")
    image_hash = _sha256(image_sha256, "canonical image hash")
    group_id = group_id_from_name(name)
    image_id = hashlib.sha256(
        b"compag-image-v1\0"
        + name.encode("utf-8")
        + b"\0"
        + group_id.encode("ascii")
        + b"\0"
        + image_hash.encode("ascii")
    ).hexdigest()
    return group_id, image_id


def _strict_int(value: object, role: str, *, minimum: int = 0) -> int:
    text = str(value)
    _require(
        re.fullmatch(r"0|[1-9][0-9]*", text) is not None,
        f"{role} is not a canonical integer",
    )
    number = int(text)
    _require(number >= minimum, f"{role} is below its minimum")
    return number


def _strict_float(value: object, role: str) -> float:
    _require(
        isinstance(value, (str, int, float)) and not isinstance(value, bool),
        f"{role} is not numeric",
    )
    try:
        number = float(value)
    except ValueError as exc:
        raise PublicIOError(f"{role} is not numeric") from exc
    _require(math.isfinite(number), f"{role} is not finite")
    return number


def _format_float(value: object) -> str:
    number = _strict_float(value, "canonical active-learning number")
    return format(0.0 if number == 0.0 else number, ".17g")


def _safe_root(root: Path) -> Path:
    absolute = root.absolute()
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError("canonical inference archive root cannot be resolved") from exc
    info = absolute.lstat()
    _require(
        absolute == resolved
        and stat.S_ISDIR(info.st_mode)
        and not absolute.is_symlink(),
        "canonical inference archive root is unsafe",
    )
    return absolute


def _stable_payload(
    path: Path,
    role: str,
    *,
    max_bytes: int,
) -> tuple[bytes, tuple[int, ...]]:
    absolute = path.absolute()
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError(f"{role} cannot be resolved") from exc
    _require(absolute == resolved, f"{role} path contains a symlink component")
    payload, info = stable_file(absolute, max_bytes=max_bytes)
    _require(
        stat.S_ISREG(info.st_mode)
        and info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == 0o644,
        f"{role} must be a single-link 0644 regular file",
    )
    identity = (
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
    return payload, identity


def _parse_csv(
    payload: bytes,
    columns: tuple[str, ...],
    role: str,
    *,
    allow_empty: bool,
) -> list[dict[str, str]]:
    _require(payload.endswith(b"\n"), f"{role} must end in LF")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise PublicIOError(f"{role} is not UTF-8") from exc
    _require(not text.startswith("\ufeff"), f"{role} contains a byte-order mark")
    try:
        with io.StringIO(text, newline="") as handle:
            reader = csv.DictReader(handle)
            _require(tuple(reader.fieldnames or ()) == columns, f"{role} header mismatch")
            rows: list[dict[str, str]] = []
            for row in reader:
                _require(len(rows) < _MAX_TABLE_ROWS, f"{role} exceeds its row bound")
                _require(
                    None not in row
                    and tuple(row) == columns
                    and all(isinstance(item, str) for item in row.values()),
                    f"{role} row has the wrong field count",
                )
                rows.append(row)
    except csv.Error as exc:
        raise PublicIOError(f"{role} contains invalid CSV framing") from exc
    _require(allow_empty or bool(rows), f"{role} is empty")
    return rows


def _csv_bytes(
    columns: tuple[str, ...],
    rows: Sequence[Mapping[str, object]],
) -> bytes:
    try:
        with io.StringIO(newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=columns,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                _require(tuple(row) == columns, "raw-feature archive row schema changed")
                writer.writerow(row)
            payload = handle.getvalue().encode("utf-8")
    except (UnicodeError, csv.Error, ValueError) as exc:
        raise PublicIOError("raw-feature archive cannot be encoded") from exc
    _require(len(payload) <= _MAX_TABLE_BYTES, "raw-feature archive exceeds its size bound")
    return payload


def _service_columns() -> tuple[tuple[str, ...], tuple[str, ...]]:
    # Imported lazily so canonical.service can call the archive writer after
    # its own module initialization without creating an import cycle.
    from .service import (
        CANONICAL_INFERENCE_COLUMNS,
        CANONICAL_RAW_FEATURE_TABLE_COLUMNS,
    )

    return tuple(CANONICAL_INFERENCE_COLUMNS), tuple(CANONICAL_RAW_FEATURE_TABLE_COLUMNS)


def _prediction_rows(
    payload: bytes,
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    prediction_columns, _feature_columns = _service_columns()
    rows = _parse_csv(
        payload,
        prediction_columns,
        "canonical inference predictions",
        allow_empty=True,
    )
    by_id: dict[str, dict[str, str]] = {}
    prior_key: tuple[str, str] | None = None
    for index, row in enumerate(rows, start=1):
        _require(
            row["id"] == str(index),
            "canonical inference prediction ordinals are not contiguous",
        )
        proposal_id = _sha256(row["proposal_id"], "prediction proposal ID")
        _require(
            row["proposal_sha256"] == proposal_id and proposal_id not in by_id,
            "canonical inference proposal identity is invalid or duplicated",
        )
        for name in (
            "detection_id",
            "image_id",
            "image_sha256",
            "tile_sha256",
            "mask_sha256",
        ):
            _sha256(row[name], f"prediction {name}")
        _require(
            _GROUP.fullmatch(row["group_id"]) is not None,
            "canonical inference prediction group is invalid",
        )
        portable_basename(row["image"], "prediction image name")
        portable_basename(row["full_image"], "prediction full-image name")
        portable_basename(row["tile_name"], "prediction tile name")
        _require(
            row["image"] == row["full_image"],
            "canonical inference image identity changed",
        )
        scale = _strict_float(row["scale"], "prediction scale")
        _require(
            scale == 1.0 and row["scale"] == _format_float(scale),
            "canonical inference predictions must contain scale 1 only",
        )
        expected_detection = hashlib.sha256(
            b"compag-canonical-detection-v1\0"
            + proposal_id.encode("ascii")
            + b"\0"
            + _format_float(scale).encode("ascii")
        ).hexdigest()
        _require(
            row["detection_id"] == expected_detection,
            "canonical inference detection identity changed",
        )
        xgb_p = _strict_float(row["xgb_p"], "prediction xgb_p")
        probability = _strict_float(row["probability"], "prediction probability")
        _require(
            0.0 <= xgb_p <= 1.0
            and xgb_p == probability
            and row["xgb_p"] == _format_float(xgb_p)
            and row["probability"] == _format_float(probability),
            "canonical inference probability fields changed",
        )
        prediction = _strict_int(row["prediction"], "prediction decision")
        kept = _strict_int(row["kept"], "prediction kept flag")
        _require(
            prediction in {0, 1}
            and prediction == int(xgb_p >= CANONICAL_DECISION_THRESHOLD)
            and kept in {0, 1}
            and (kept == 0 or prediction == 1),
            "canonical inference decision fields changed",
        )
        x = _strict_int(row["x"], "prediction x")
        y = _strict_int(row["y"], "prediction y")
        width = _strict_int(row["w"], "prediction width", minimum=1)
        height = _strict_int(row["h"], "prediction height", minimum=1)
        original_width = _strict_int(
            row["orig_w"],
            "prediction original width",
            minimum=1,
        )
        original_height = _strict_int(
            row["orig_h"],
            "prediction original height",
            minimum=1,
        )
        _require(
            _strict_int(row["bbox_x1"], "prediction bbox_x1") == x
            and _strict_int(row["bbox_y1"], "prediction bbox_y1") == y
            and _strict_int(row["bbox_x2"], "prediction bbox_x2") == x + width
            and _strict_int(row["bbox_y2"], "prediction bbox_y2") == y + height
            and original_width >= x + width
            and original_height >= y + height,
            "canonical inference prediction geometry changed",
        )
        try:
            polygon = json.loads(row["poly"])
        except (json.JSONDecodeError, RecursionError) as exc:
            raise PublicIOError("canonical inference polygon is invalid JSON") from exc
        _require(
            isinstance(polygon, list)
            and len(polygon) % 2 == 0
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in polygon
            )
            and all(value <= original_width for value in polygon[0::2])
            and all(value <= original_height for value in polygon[1::2]),
            "canonical inference polygon is invalid",
        )
        key = (row["image_id"], proposal_id)
        _require(
            prior_key is None or prior_key < key,
            "canonical inference predictions are not in canonical order",
        )
        prior_key = key
        by_id[proposal_id] = row
    return rows, by_id


def _normalize_raw_row(
    row: Mapping[str, object],
    prediction: Mapping[str, str],
) -> dict[str, object]:
    _prediction_columns, feature_columns = _service_columns()
    _require(tuple(row) == feature_columns, "raw-feature archive input schema changed")
    identity = {
        "proposal_id": prediction["proposal_id"],
        "proposal_sha256": prediction["proposal_sha256"],
        "image_id": prediction["image_id"],
        "image_sha256": prediction["image_sha256"],
        "image_name": prediction["image"],
        "group_id": prediction["group_id"],
        "tile_name": prediction["tile_name"],
        "tile_sha256": prediction["tile_sha256"],
        "mask_sha256": prediction["mask_sha256"],
    }
    _require(
        identity["proposal_sha256"] == identity["proposal_id"]
        and _SHA256.fullmatch(identity["proposal_id"]) is not None
        and _SHA256.fullmatch(identity["image_id"]) is not None
        and _SHA256.fullmatch(identity["image_sha256"]) is not None
        and _SHA256.fullmatch(identity["tile_sha256"]) is not None
        and _SHA256.fullmatch(identity["mask_sha256"]) is not None
        and _GROUP.fullmatch(identity["group_id"]) is not None,
        "raw-feature archive immutable identity is invalid",
    )
    portable_basename(identity["image_name"], "raw-feature image name")
    portable_basename(identity["tile_name"], "raw-feature tile name")
    expected_group, expected_image_id = _canonical_image_identity(
        str(identity["image_name"]),
        str(identity["image_sha256"]),
    )
    _require(
        all(str(row[name]) == value for name, value in identity.items()),
        "raw-feature archive identity differs from predictions",
    )
    _require(
        identity["group_id"] == expected_group
        and identity["image_id"] == expected_image_id,
        "raw-feature image identity differs from its filename-derived group",
    )
    tile_x = _strict_int(row["tile_x"], "raw-feature tile x")
    tile_y = _strict_int(row["tile_y"], "raw-feature tile y")
    proposal_index = _strict_int(
        row["proposal_index"],
        "raw-feature proposal index",
        minimum=1,
    )
    scale = _strict_float(row["scale"], "raw-feature scale")
    _require(scale in CANONICAL_FEATURE_CROP_SCALES, "raw-feature scale changed")
    predicted_iou = _strict_float(row["predicted_iou"], "raw-feature predicted IoU")
    stability_score = _strict_float(
        row["stability_score"],
        "raw-feature stability score",
    )
    _require(
        0.0 <= predicted_iou <= 1.0 and 0.0 <= stability_score <= 1.0,
        "raw-feature SAM2 score is outside [0,1]",
    )
    values = {
        name: _strict_float(row[name], f"raw-feature {name}")
        for name in CANONICAL_RAW_FEATURE_ORDER
    }
    _require(
        row["embedding_encoding"] == "base64-float32-little-endian-v1"
        and _strict_int(
            row["embedding_dimensions"],
            "raw-feature embedding dimensions",
            minimum=1,
        )
        == 2048,
        "raw-feature embedding contract changed",
    )
    try:
        embedding = decode_canonical_embedding(str(row["embedding_f32le_base64"]))
    except ValueError as exc:
        raise PublicIOError("raw-feature embedding is invalid") from exc
    encoded = encode_canonical_embedding(embedding)
    _require(
        encoded == row["embedding_f32le_base64"],
        "raw-feature embedding encoding is noncanonical",
    )
    normalized: dict[str, object] = {
        **{name: identity[name] for name in _RAW_IDENTITY_COLUMNS if name in identity},
        "tile_x": str(tile_x),
        "tile_y": str(tile_y),
        "mask_sha256": identity["mask_sha256"],
        "proposal_index": str(proposal_index),
        "scale": _format_float(scale),
        "predicted_iou": _format_float(predicted_iou),
        "stability_score": _format_float(stability_score),
        **{name: _format_float(values[name]) for name in CANONICAL_RAW_FEATURE_ORDER},
        "embedding_encoding": "base64-float32-little-endian-v1",
        "embedding_dimensions": "2048",
        "embedding_f32le_base64": encoded,
    }
    normalized = {name: normalized[name] for name in feature_columns}
    return normalized


def _normalized_feature_rows(
    rows: Sequence[Mapping[str, object]],
    predictions: Mapping[str, Mapping[str, str]],
) -> list[dict[str, object]]:
    _require(
        len(rows) <= _MAX_TABLE_ROWS,
        "raw-feature archive exceeds its row bound",
    )
    normalized: list[dict[str, object]] = []
    observed: set[tuple[str, float]] = set()
    for row in rows:
        proposal_id = str(row.get("proposal_id", ""))
        _require(
            proposal_id in predictions,
            "raw-feature archive references an unknown prediction",
        )
        item = _normalize_raw_row(row, predictions[proposal_id])
        key = (proposal_id, float(item["scale"]))
        _require(key not in observed, "raw-feature archive contains a duplicate scale row")
        observed.add(key)
        normalized.append(item)
    scale_index = {
        float(scale): index for index, scale in enumerate(CANONICAL_FEATURE_CROP_SCALES)
    }
    normalized.sort(
        key=lambda row: (str(row["proposal_id"]), scale_index[float(row["scale"])])
    )
    expected = {
        (proposal_id, float(scale))
        for proposal_id in predictions
        for scale in CANONICAL_FEATURE_CROP_SCALES
    }
    _require(
        observed == expected
        and len(normalized) == len(predictions) * len(CANONICAL_FEATURE_CROP_SCALES),
        "raw-feature archive does not contain exactly four scales per prediction",
    )
    proposal_receipts: dict[str, tuple[str, ...]] = {}
    for row in normalized:
        proposal_id = str(row["proposal_id"])
        receipt = tuple(
            str(row[name])
            for name in (
                *_RAW_IDENTITY_COLUMNS,
                "proposal_index",
                "predicted_iou",
                "stability_score",
            )
        )
        _require(
            proposal_id not in proposal_receipts
            or proposal_receipts[proposal_id] == receipt,
            "raw-feature archive proposal identity changed between scales",
        )
        proposal_receipts[proposal_id] = receipt
    return normalized


def _archive_receipt(
    payload: bytes,
    *,
    bundle_sha256: str,
    predictions_sha256: str,
    proposal_count: int,
    row_count: int,
    profile: str = CANONICAL_PROFILE,
) -> dict[str, object]:
    _prediction_columns, feature_columns = _service_columns()
    _canonical_device_for_profile(profile)
    return {
        "schema": CANONICAL_AL_RAW_FEATURE_ARCHIVE_SCHEMA,
        "status": "PASS",
        "profile": profile,
        "basename": "raw_features.csv",
        "columns_sha256": compact_json_sha256(list(feature_columns)),
        "crop_scales": list(CANONICAL_FEATURE_CROP_SCALES),
        "rows_per_proposal": len(CANONICAL_FEATURE_CROP_SCALES),
        "proposal_count": proposal_count,
        "row_count": row_count,
        "bundle_sha256": bundle_sha256,
        "predictions_sha256": predictions_sha256,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def write_canonical_active_learning_raw_feature_archive(
    rows: Sequence[Mapping[str, object]],
    predictions_path: Path,
    output: Path,
    *,
    bundle_sha256: str,
    profile: str = CANONICAL_PROFILE,
) -> dict[str, object]:
    """Write an exact four-scale archive bound to one stage-60 prediction table."""

    bundle_hash = _sha256(bundle_sha256, "raw-feature archive bundle")
    resolved_profile = str(profile)
    _canonical_device_for_profile(resolved_profile)
    predictions_payload, predictions_identity = _stable_payload(
        predictions_path,
        "canonical inference predictions",
        max_bytes=_MAX_TABLE_BYTES,
    )
    _prediction_list, predictions = _prediction_rows(predictions_payload)
    normalized = _normalized_feature_rows(rows, predictions)
    _prediction_columns, feature_columns = _service_columns()
    payload = _csv_bytes(feature_columns, normalized)
    # Parse the rendered bytes before creating the output file.
    parsed = _parse_csv(
        payload,
        feature_columns,
        "canonical raw-feature archive",
        allow_empty=True,
    )
    _require(
        parsed == normalized,
        "raw-feature archive changed during canonical rendering",
    )
    absolute = output.absolute()
    portable_basename(absolute.name, "raw-feature archive basename")
    _require(absolute.name == "raw_features.csv", "raw-feature archive basename changed")
    try:
        parent = absolute.parent.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError("raw-feature archive parent cannot be resolved") from exc
    _require(
        parent == absolute.parent
        and parent.is_dir()
        and not parent.is_symlink()
        and not absolute.exists()
        and not absolute.is_symlink(),
        "raw-feature archive output is unsafe or already exists",
    )
    write_new_bytes(absolute, payload)
    predictions_after, identity_after = _stable_payload(
        predictions_path,
        "canonical inference predictions",
        max_bytes=_MAX_TABLE_BYTES,
    )
    _require(
        predictions_after == predictions_payload
        and identity_after == predictions_identity,
        "canonical inference predictions changed while archiving features",
    )
    return _archive_receipt(
        payload,
        bundle_sha256=bundle_hash,
        predictions_sha256=hashlib.sha256(predictions_payload).hexdigest(),
        proposal_count=len(predictions),
        row_count=len(normalized),
        profile=resolved_profile,
    )


def _self_described_predictions(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, str]]:
    predictions: dict[str, dict[str, str]] = {}
    for row in rows:
        proposal_id = str(row.get("proposal_id", ""))
        if proposal_id in predictions:
            continue
        predictions[proposal_id] = {
            "proposal_id": proposal_id,
            "proposal_sha256": str(row.get("proposal_sha256", "")),
            "image_id": str(row.get("image_id", "")),
            "image_sha256": str(row.get("image_sha256", "")),
            "image": str(row.get("image_name", "")),
            "group_id": str(row.get("group_id", "")),
            "tile_name": str(row.get("tile_name", "")),
            "tile_sha256": str(row.get("tile_sha256", "")),
            "mask_sha256": str(row.get("mask_sha256", "")),
        }
    return predictions


def _training_rows_from_normalized(
    rows: Sequence[Mapping[str, object]],
    *,
    source_sha256: str,
) -> tuple[CanonicalActiveLearningTrainingRow, ...]:
    output: list[CanonicalActiveLearningTrainingRow] = []
    for row in rows:
        values = {
            name: float(row[name]) for name in CANONICAL_RAW_FEATURE_ORDER
        }
        try:
            embedding = decode_canonical_embedding(str(row["embedding_f32le_base64"]))
        except ValueError as exc:
            raise PublicIOError("canonical raw-feature embedding is invalid") from exc
        output.append(
            CanonicalActiveLearningTrainingRow(
                proposal_id=str(row["proposal_id"]),
                proposal_sha256=str(row["proposal_sha256"]),
                image_id=str(row["image_id"]),
                image_sha256=str(row["image_sha256"]),
                group_id=str(row["group_id"]),
                raw_feature=CanonicalRawFeature(
                    proposal_index=int(row["proposal_index"]),
                    scale=float(row["scale"]),
                    predicted_iou=float(row["predicted_iou"]),
                    stability_score=float(row["stability_score"]),
                    values=values,
                    embedding=embedding,
                ),
                source_sha256=source_sha256,
            )
        )
    return tuple(output)


def read_canonical_stage20_raw_features(
    features_path: Path,
) -> tuple[CanonicalActiveLearningTrainingRow, ...]:
    """Descriptor-pin and parse a strict stage-20 four-scale feature table."""

    payload, identity = _stable_payload(
        features_path,
        "canonical stage-20 raw features",
        max_bytes=_MAX_TABLE_BYTES,
    )
    _prediction_columns, feature_columns = _service_columns()
    rows = _parse_csv(
        payload,
        feature_columns,
        "canonical stage-20 raw features",
        allow_empty=False,
    )
    normalized = _normalized_feature_rows(rows, _self_described_predictions(rows))
    normalized_by_key = {
        (str(row["proposal_id"]), float(row["scale"])): row for row in normalized
    }
    for row in rows:
        normalized_row = normalized_by_key.get(
            (str(row["proposal_id"]), float(row["scale"]))
        )
        _require(
            normalized_row is not None,
            "canonical stage-20 raw feature identity changed",
        )
        raw = CanonicalRawFeature(
            proposal_index=int(row["proposal_index"]),
            scale=float(row["scale"]),
            predicted_iou=float(row["predicted_iou"]),
            stability_score=float(row["stability_score"]),
            values={
                name: float(row[name]) for name in CANONICAL_RAW_FEATURE_ORDER
            },
            embedding=decode_canonical_embedding(row["embedding_f32le_base64"]),
        )
        scientific = canonical_raw_feature_csv_row(raw)
        rendered = {
            **{name: str(normalized_row[name]) for name in _RAW_IDENTITY_COLUMNS},
            **{name: str(value) for name, value in scientific.items()},
        }
        rendered = {name: rendered[name] for name in feature_columns}
        _require(
            row == rendered,
            "canonical stage-20 raw feature representation changed",
        )
    block_size = len(CANONICAL_FEATURE_CROP_SCALES)
    observed_proposals: set[str] = set()
    for offset in range(0, len(rows), block_size):
        block = rows[offset : offset + block_size]
        proposal_id = block[0]["proposal_id"]
        _require(
            proposal_id not in observed_proposals
            and all(row["proposal_id"] == proposal_id for row in block)
            and tuple(float(row["scale"]) for row in block)
            == tuple(CANONICAL_FEATURE_CROP_SCALES),
            "canonical stage-20 raw features are not exact proposal-scale blocks",
        )
        observed_proposals.add(proposal_id)
    source_hash = hashlib.sha256(payload).hexdigest()
    _payload_after, identity_after = _stable_payload(
        features_path,
        "canonical stage-20 raw features",
        max_bytes=_MAX_TABLE_BYTES,
    )
    _require(
        identity_after == identity,
        "canonical stage-20 raw features changed while parsing",
    )
    return _training_rows_from_normalized(
        normalized,
        source_sha256=source_hash,
    )


def write_initial_canonical_active_learning_pool(
    features_path: Path,
    output: Path,
    *,
    profile: str = CANONICAL_PROFILE,
) -> dict[str, object]:
    """Write a source-bound unscored AL pool from stage-20 raw features."""

    feature_rows = read_canonical_stage20_raw_features(features_path)
    source_hashes = {row.source_sha256 for row in feature_rows}
    _require(
        len(source_hashes) == 1,
        "canonical initial pool feature-source identity changed",
    )
    scale_one = [
        row for row in feature_rows if float(row.raw_feature.scale) == 1.0
    ]
    _require(
        len(scale_one) * len(CANONICAL_FEATURE_CROP_SCALES) == len(feature_rows),
        "canonical initial pool scale-1 closure changed",
    )
    pool_rows = tuple(
        CanonicalActiveLearningPoolRow(
            proposal_id=row.proposal_id,
            proposal_sha256=row.proposal_sha256,
            image_id=row.image_id,
            image_sha256=row.image_sha256,
            group_id=row.group_id,
            scale=1.0,
            xgb_p=None,
        )
        for row in scale_one
    )
    return write_canonical_active_learning_pool(
        pool_rows,
        output,
        bundle_sha256=None,
        source_sha256=next(iter(source_hashes)),
        feature_archive_sha256=_feature_archive_hash(
            feature_rows
        ),
        profile=profile,
    )


def _validate_inventory(value: object) -> list[Mapping[str, object]]:
    _require(isinstance(value, list), "canonical inference input inventory is invalid")
    rows: list[Mapping[str, object]] = []
    names: set[str] = set()
    identities: set[str] = set()
    prior_name: str | None = None
    for item in value:
        _require(
            isinstance(item, dict)
            and set(item)
            == {
                "image_name",
                "image_id",
                "image_sha256",
                "group_id",
                "size_bytes",
                "width",
                "height",
                "tile_count",
                "candidate_rows",
            },
            "canonical inference inventory fields changed",
        )
        _require(
            isinstance(item["image_name"], str)
            and isinstance(item["image_id"], str)
            and isinstance(item["image_sha256"], str)
            and isinstance(item["group_id"], str),
            "canonical inference inventory identity types changed",
        )
        name = portable_basename(item["image_name"], "inference inventory image")
        image_id = _sha256(item["image_id"], "inference inventory image ID")
        image_sha256 = _sha256(
            item["image_sha256"],
            "inference inventory image hash",
        )
        group_id = item["group_id"]
        expected_group, expected_image_id = _canonical_image_identity(
            name,
            image_sha256,
        )
        _require(
            _GROUP.fullmatch(group_id) is not None
            and group_id == expected_group
            and image_id == expected_image_id
            and name not in names
            and image_id not in identities
            and (prior_name is None or prior_name < name)
            and all(
                isinstance(item[field], int)
                and not isinstance(item[field], bool)
                and item[field] >= minimum
                for field, minimum in (
                    ("size_bytes", 1),
                    ("width", 1),
                    ("height", 1),
                    ("tile_count", 1),
                    ("candidate_rows", 0),
                )
            ),
            "canonical inference inventory row is invalid",
        )
        names.add(name)
        identities.add(image_id)
        prior_name = name
        rows.append(item)
    _require(bool(rows), "canonical inference inventory is empty")
    return rows


def _parse_inference_result(
    payload: bytes,
    *,
    predictions_payload: bytes,
    feature_payload: bytes,
    prediction_rows: Sequence[Mapping[str, str]],
    feature_row_count: int,
) -> Mapping[str, object]:
    value = canonical_json_value(payload, "canonical inference result")
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
        "canonical inference result method identity changed",
    )
    bundle_hash = _sha256(value.get("bundle_sha256"), "inference result bundle")
    predictions_hash = hashlib.sha256(predictions_payload).hexdigest()
    _require(
        value.get("predictions_sha256") == predictions_hash,
        "canonical inference result does not bind predictions.csv",
    )
    inventory = _validate_inventory(value.get("input_inventory"))
    _require(
        profile != FULL_IMAGE_GPU_PROFILE
        or all(int(item["tile_count"]) == 1 for item in inventory),
        "full-image inference tile_count compatibility counter must be one per image",
    )
    inventory_by_id = {str(item["image_id"]): item for item in inventory}
    prediction_counts = {image_id: 0 for image_id in inventory_by_id}
    for row in prediction_rows:
        item = inventory_by_id.get(row["image_id"])
        _require(
            item is not None
            and row["image"] == item["image_name"]
            and row["image_sha256"] == item["image_sha256"]
            and row["group_id"] == item["group_id"]
            and _strict_int(row["orig_w"], "prediction original width", minimum=1)
            == item["width"]
            and _strict_int(row["orig_h"], "prediction original height", minimum=1)
            == item["height"],
            "canonical inference prediction differs from its input inventory",
        )
        prediction_counts[row["image_id"]] += 1
    _require(
        all(
            prediction_counts[str(item["image_id"])] == item["candidate_rows"]
            for item in inventory
        ),
        "canonical inference per-image candidate counts changed",
    )
    counts = {
        name: value.get(name)
        for name in (
            "image_count",
            "tile_count",
            "proposal_count",
            "prediction_rows",
            "positive_rows",
            "kept_rows",
        )
    }
    _require(
        all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in counts.values()
        )
        and counts["image_count"] == len(inventory)
        and counts["tile_count"] == sum(int(item["tile_count"]) for item in inventory)
        and (
            profile != FULL_IMAGE_GPU_PROFILE
            or counts["tile_count"] == counts["image_count"]
        )
        and counts["proposal_count"] == counts["prediction_rows"] == len(prediction_rows)
        and counts["proposal_count"]
        == sum(int(item["candidate_rows"]) for item in inventory)
        and counts["positive_rows"]
        == sum(row["prediction"] == "1" for row in prediction_rows)
        and counts["kept_rows"] == sum(row["kept"] == "1" for row in prediction_rows),
        "canonical inference result counts changed",
    )
    expected_receipt = _archive_receipt(
        feature_payload,
        bundle_sha256=bundle_hash,
        predictions_sha256=predictions_hash,
        proposal_count=len(prediction_rows),
        row_count=feature_row_count,
        profile=str(profile),
    )
    _require(
        value.get("raw_feature_archive") == expected_receipt,
        "canonical inference result does not bind the raw-feature archive",
    )
    return value


def _feature_rows(
    payload: bytes,
    predictions: Mapping[str, Mapping[str, str]],
    *,
    source_sha256: str,
) -> tuple[CanonicalActiveLearningTrainingRow, ...]:
    _prediction_columns, feature_columns = _service_columns()
    rows = _parse_csv(
        payload,
        feature_columns,
        "canonical raw-feature archive",
        allow_empty=True,
    )
    normalized = _normalized_feature_rows(rows, predictions)
    _require(rows == normalized, "canonical raw-feature archive representation changed")
    return _training_rows_from_normalized(rows, source_sha256=source_sha256)


def read_canonical_active_learning_inference(
    stage_root: Path,
) -> CanonicalActiveLearningInferenceArchive:
    """Descriptor-pin and parse one AL-capable canonical inference output."""

    root = _safe_root(stage_root)
    predictions_path = root / "predictions.csv"
    features_path = root / "raw_features.csv"
    result_path = root / "inference_result.json"
    predictions_payload, predictions_identity = _stable_payload(
        predictions_path,
        "canonical inference predictions",
        max_bytes=_MAX_TABLE_BYTES,
    )
    features_payload, features_identity = _stable_payload(
        features_path,
        "canonical raw-feature archive",
        max_bytes=_MAX_TABLE_BYTES,
    )
    result_payload, result_identity = _stable_payload(
        result_path,
        "canonical inference result",
        max_bytes=_MAX_JSON_BYTES,
    )
    prediction_rows, predictions = _prediction_rows(predictions_payload)
    raw_hash = hashlib.sha256(features_payload).hexdigest()
    feature_rows = _feature_rows(
        features_payload,
        predictions,
        source_sha256=raw_hash,
    )
    result = _parse_inference_result(
        result_payload,
        predictions_payload=predictions_payload,
        feature_payload=features_payload,
        prediction_rows=prediction_rows,
        feature_row_count=len(feature_rows),
    )
    pool_rows = tuple(
        sorted(
            (
                CanonicalActiveLearningPoolRow(
                    proposal_id=row["proposal_id"],
                    proposal_sha256=row["proposal_sha256"],
                    image_id=row["image_id"],
                    image_sha256=row["image_sha256"],
                    group_id=row["group_id"],
                    scale=1.0,
                    xgb_p=float(row["xgb_p"]),
                )
                for row in prediction_rows
            ),
            key=lambda row: row.proposal_id,
        )
    )
    for path, role, limit, expected_identity, expected_payload in (
        (
            predictions_path,
            "canonical inference predictions",
            _MAX_TABLE_BYTES,
            predictions_identity,
            predictions_payload,
        ),
        (
            features_path,
            "canonical raw-feature archive",
            _MAX_TABLE_BYTES,
            features_identity,
            features_payload,
        ),
        (
            result_path,
            "canonical inference result",
            _MAX_JSON_BYTES,
            result_identity,
            result_payload,
        ),
    ):
        payload_after, identity_after = _stable_payload(path, role, max_bytes=limit)
        _require(
            payload_after == expected_payload and identity_after == expected_identity,
            "canonical inference input changed while parsing",
        )
    return CanonicalActiveLearningInferenceArchive(
        pool_rows=pool_rows,
        feature_rows=feature_rows,
        bundle_sha256=str(result["bundle_sha256"]),
        predictions_sha256=hashlib.sha256(predictions_payload).hexdigest(),
        raw_features_sha256=raw_hash,
        feature_archive_sha256=_feature_archive_hash(
            feature_rows
        ),
        inference_result_sha256=hashlib.sha256(result_payload).hexdigest(),
        profile=str(result["profile"]),
        device=str(result["device"]),
    )


def read_canonical_r92_transfer_inference(
    stage_root: Path,
) -> CanonicalR92TransferInference:
    """Verify a standard Stage-60 archive and its explicit r92 transfer receipt."""

    root = _safe_root(stage_root)
    inference = read_canonical_active_learning_inference(root)
    receipt_payload, receipt_identity = _stable_payload(
        root / "transfer_scorer.json",
        "r92 transfer scorer receipt",
        max_bytes=_MAX_JSON_BYTES,
    )
    receipt = canonical_json_value(receipt_payload, "r92 transfer scorer receipt")
    fields = {
        "schema", "status", "workflow", "lineage_policy",
        "reproduction_claim", "preset_id", "profile", "device", "threshold",
        "feature_order_sha256", "resource_manifest_sha256", "classifier_sha256",
        "feature_state_sha256", "scorer_identity_sha256", "config_sha256",
        "asset_inventory", "asset_inventory_sha256", "image_inventory_sha256", "image",
        "predictions_sha256", "raw_features_sha256", "inference_result_sha256",
    }
    _require(
        isinstance(receipt, dict)
        and set(receipt) == fields
        and receipt.get("schema") == R92_TRANSFER_INFERENCE_SCHEMA
        and receipt.get("status") == "PASS"
        and receipt.get("workflow") == R92_TRANSFER_INFERENCE_WORKFLOW
        and receipt.get("lineage_policy") == R92_TRANSFER_LINEAGE_POLICY
        and receipt.get("reproduction_claim") == R92_TRANSFER_REPRODUCTION_CLAIM
        and receipt.get("preset_id") == "compag-cj-r92"
        and receipt.get("profile") == CANONICAL_GPU_PROFILE
        and receipt.get("device") == "cuda"
        and receipt.get("threshold") == CANONICAL_DECISION_THRESHOLD
        and receipt.get("feature_order_sha256") == CANONICAL_FEATURE_ORDER_SHA256
        and inference.profile == CANONICAL_GPU_PROFILE
        and inference.device == "cuda"
        and receipt.get("scorer_identity_sha256") == inference.bundle_sha256
        and receipt.get("predictions_sha256") == inference.predictions_sha256
        and receipt.get("raw_features_sha256") == inference.raw_features_sha256
        and receipt.get("inference_result_sha256")
        == inference.inference_result_sha256,
        "r92 transfer scorer receipt does not bind the Stage-60 archive",
    )
    for name in (
        "resource_manifest_sha256", "classifier_sha256", "feature_state_sha256",
        "scorer_identity_sha256", "config_sha256", "asset_inventory_sha256",
        "image_inventory_sha256", "predictions_sha256", "raw_features_sha256",
        "inference_result_sha256",
    ):
        _sha256(receipt.get(name), f"r92 transfer receipt {name}")
    from compag_curation.canonical.transfer_inference import (
        transfer_scorer_identity_sha256,
    )

    asset_inventory = receipt.get("asset_inventory")
    _require(
        isinstance(asset_inventory, list)
        and receipt.get("asset_inventory_sha256")
        == compact_json_sha256(asset_inventory)
        and receipt.get("scorer_identity_sha256")
        == transfer_scorer_identity_sha256(
            preset_id=str(receipt["preset_id"]),
            resource_manifest_sha256=str(receipt["resource_manifest_sha256"]),
            classifier_sha256=str(receipt["classifier_sha256"]),
            feature_state_sha256=str(receipt["feature_state_sha256"]),
            profile=str(receipt["profile"]),
            threshold=float(receipt["threshold"]),
            feature_order_sha256=str(receipt["feature_order_sha256"]),
            asset_inventory=asset_inventory,
        ),
        "r92 transfer scorer identity differs from its preset and public assets",
    )
    result_payload, _result_identity = _stable_payload(
        root / "inference_result.json",
        "r92 transfer inference result",
        max_bytes=_MAX_JSON_BYTES,
    )
    result = canonical_json_value(result_payload, "r92 transfer inference result")
    inventory = result.get("input_inventory") if isinstance(result, dict) else None
    image = receipt.get("image")
    image_fields = {
        "image_name", "image_id", "image_sha256", "group_id", "size_bytes",
        "width", "height", "tile_count", "candidate_rows",
    }
    _require(
        isinstance(inventory, list)
        and len(inventory) == 1
        and isinstance(image, dict)
        and set(image) == image_fields
        and image == inventory[0]
        and receipt.get("image_inventory_sha256")
        == compact_json_sha256(inventory),
        "r92 transfer inference must contain exactly one bound image",
    )
    from compag_curation.review.published_model import (
        load_published_model_transfer_assets,
    )

    assets = load_published_model_transfer_assets()
    state = assets.feature_state
    validate_canonical_feature_state(state)
    payloads = serialize_canonical_feature_state(state)
    prototype_sha256 = hashlib.sha256(payloads.prototype_npy).hexdigest()
    pca_components_sha256 = hashlib.sha256(
        payloads.pca_components_npy
    ).hexdigest()
    pca_mean_sha256 = hashlib.sha256(payloads.pca_mean_npy).hexdigest()
    _require(
        receipt["resource_manifest_sha256"] == assets.resource_manifest_sha256
        and receipt["classifier_sha256"] == assets.classifier_sha256
        and receipt["feature_state_sha256"] == assets.feature_state_sha256,
        "r92 transfer receipt differs from the installed published preset",
    )
    scorer = CanonicalTransferScorerReference(
        kind="PUBLISHED_R92_TRANSFER_SCORER",
        scorer_sha256=str(receipt["scorer_identity_sha256"]),
        scores_sha256=str(receipt["predictions_sha256"]),
        # This is rebound to the derived AL-pool CSV identity by the facade
        # before the scorer receipt enters a round manifest.
        pool_scores_sha256=str(receipt["predictions_sha256"]),
        feature_state_sha256=str(receipt["feature_state_sha256"]),
        prototype_sha256=prototype_sha256,
        pca_components_sha256=pca_components_sha256,
        pca_mean_sha256=pca_mean_sha256,
        r92_classifier_sha256=str(receipt["classifier_sha256"]),
        r92_resource_manifest_sha256=str(receipt["resource_manifest_sha256"]),
        preset_id="compag-cj-r92",
    )
    receipt_after, identity_after = _stable_payload(
        root / "transfer_scorer.json",
        "r92 transfer scorer receipt",
        max_bytes=_MAX_JSON_BYTES,
    )
    _require(
        receipt_after == receipt_payload and identity_after == receipt_identity,
        "r92 transfer scorer receipt changed while reading",
    )
    return CanonicalR92TransferInference(
        inference=inference,
        scorer=scorer,
        feature_state=state,
        receipt=receipt,
    )


def train_canonical_transfer_request(
    request: CanonicalTransferRetrainRequest,
) -> CanonicalTransferTrainingResult:
    """Fresh-fit XGBoost from a sealed finalized baseline plus reviewed rounds.

    This function never accepts an existing classifier and therefore has no
    warm-start route.  The r92 prototype and PCA are used only to finalize new
    raw rows; the imported baseline already contains values in that same
    feature coordinate system.
    """

    import numpy as np
    from compag_curation.canonical.transfer_baseline import (
        iter_transfer_baseline_rows,
        verify_transfer_baseline,
    )

    validate_canonical_feature_state(request.feature_state)
    baseline = verify_transfer_baseline(request.baseline_root)
    archive_rows = manifest_rows(
        request.baseline_root,
        max_total_bytes=4 * 1024 * 1024 * 1024,
    )
    archive_sha256 = compact_json_sha256(archive_rows)
    split_payload, _split_identity = _stable_payload(
        baseline.split_path,
        "transfer baseline split",
        max_bytes=_MAX_JSON_BYTES,
    )
    source_groups = frozenset(str(value) for value in baseline.source_groups)
    _require(
        archive_sha256 == request.baseline.archive_sha256
        and str(baseline.source_sha256) == request.baseline.source_sha256
        and int(baseline.source_size_bytes) == request.baseline.source_size_bytes
        and hashlib.sha256(split_payload).hexdigest() == request.baseline.split_sha256
        and compact_json_sha256(sorted(source_groups))
        == request.baseline.source_groups_sha256
        and source_groups == request.baseline.source_groups
        and frozenset(str(value) for value in baseline.train_groups)
        == request.baseline.train_groups
        and frozenset(str(value) for value in baseline.test_groups)
        == request.baseline.test_groups
        and int(baseline.row_count) == request.baseline.row_count
        and int(baseline.proposal_count) == request.baseline.proposal_count,
        "transfer baseline changed after the retrain request was sealed",
    )
    al_rows = _validate_transfer_al_rows(
        request.cumulative_al_rows,
        request.effective_decisions,
        request.baseline,
    )
    latest = {
        decision.proposal_id: decision
        for decision in request.effective_decisions
    }
    _require(
        request.cumulative_al_rows_sha256 == _training_row_hash(al_rows, latest),
        "transfer cumulative AL row receipt changed",
    )
    feature_rows: list[Mapping[str, float]] = []
    labels: list[int] = []
    groups: list[str] = []
    actions: list[str] = []
    weights: list[float] = []
    scales: list[float] = []
    baseline_proposals: set[str] = set()
    observed_baseline_rows = 0
    reviewed_baseline_rows = 0
    unreviewed_baseline_rows = 0
    pca_names = tuple(f"embed_pca_{index}" for index in range(32))
    pca_indices = tuple(_FINAL_FEATURE_INDEX[name] for name in pca_names)
    variance_count = 0
    variance_mean = np.zeros(32, dtype=np.float64)
    variance_m2 = np.zeros(32, dtype=np.float64)
    effective_training_rows = 0
    positive_training_rows = 0
    heldout_parity_rows: list[Mapping[str, float]] = []

    def observe_variance(values: Sequence[float], group: str, weight: float) -> None:
        nonlocal variance_count
        if group not in request.train_groups or weight <= 0.0:
            return
        vector = np.asarray([values[index] for index in pca_indices], dtype=np.float64)
        variance_count += 1
        delta = vector - variance_mean
        variance_mean[:] = variance_mean + delta / variance_count
        variance_m2[:] = variance_m2 + delta * (vector - variance_mean)

    for row in iter_transfer_baseline_rows(baseline):
        proposal_id = str(row.proposal_id)
        group_id = str(row.group_id)
        action = str(row.review_action)
        reviewed = bool(row.reviewed)
        try:
            label = int(row.label)
            weight = float(row.review_weight)
            scale = float(row.scale)
        except (TypeError, ValueError) as exc:
            raise PublicIOError("transfer baseline training metadata is invalid") from exc
        review_metadata_valid = (
            reviewed
            and action in REVIEW_ACTION_WEIGHTS
            and action != "skip"
            and weight == REVIEW_ACTION_WEIGHTS[action]
            and weight > 0.0
        ) or (
            not reviewed
            and action == ""
            and weight == 1.0
        )
        _require(
            _SHA256.fullmatch(proposal_id) is not None
            and _GROUP.fullmatch(group_id) is not None
            and group_id
            in (request.baseline.train_groups | request.baseline.test_groups)
            and label in {0, 1}
            and review_metadata_valid
            and scale in CANONICAL_FEATURE_CROP_SCALES,
            "transfer baseline row identity, action, or scale is invalid",
        )
        values = tuple(float(value) for value in row.features)
        mapping = _TupleFeatureMapping(values)
        feature_rows.append(mapping)
        labels.append(label)
        groups.append(group_id)
        # ``train_canonical_xgb`` validates weights through action names.  The
        # legacy non-reviewed rows legitimately have no human action, so map
        # only that numeric-validation input to the unit-weight action.  The
        # sealed row and every public receipt continue to say unreviewed/blank.
        actions.append(action if reviewed else "accept")
        weights.append(weight)
        scales.append(scale)
        baseline_proposals.add(proposal_id)
        observed_baseline_rows += 1
        reviewed_baseline_rows += int(reviewed)
        unreviewed_baseline_rows += int(not reviewed)
        if group_id in request.train_groups and weight > 0.0:
            effective_training_rows += 1
            positive_training_rows += int(label == 1)
        if group_id in request.test_groups and weight > 0.0 and scale == 1.0:
            heldout_parity_rows.append(mapping)
        observe_variance(values, group_id, weight)
    _require(
        observed_baseline_rows == request.baseline.row_count
        and len(baseline_proposals) == request.baseline.proposal_count,
        "transfer baseline iterator does not close its sealed counts",
    )
    for row in al_rows:
        _require(
            row.proposal_id not in baseline_proposals,
            "reviewed transfer proposal collides with the baseline",
        )
        decision = latest[row.proposal_id]
        finalized = finalize_canonical_feature(row.raw_feature, request.feature_state)
        _require(
            tuple(finalized) == CANONICAL_FEATURE_ORDER,
            "transfer AL row did not finalize to the exact feature order",
        )
        feature_rows.append(finalized)
        labels.append(decision.label)
        groups.append(row.group_id)
        actions.append(decision.review_action)
        weights.append(decision.review_weight)
        scales.append(float(row.raw_feature.scale))
        if row.group_id in request.train_groups and decision.review_weight > 0.0:
            effective_training_rows += 1
            positive_training_rows += int(decision.label == 1)
        observe_variance(
            tuple(float(finalized[name]) for name in CANONICAL_FEATURE_ORDER),
            row.group_id,
            decision.review_weight,
        )
    observed_groups = frozenset(groups)
    _require(
        observed_groups == request.train_groups | request.test_groups
        and not request.train_groups & request.test_groups,
        "transfer training rows do not close the frozen group split",
    )
    _require(
        effective_training_rows >= 32
        and 0 < positive_training_rows <= effective_training_rows
        and bool(heldout_parity_rows),
        "transfer train/test rows cannot support training or UBJ parity",
    )
    split = CanonicalGroupSplit(
        train_groups=request.train_groups,
        test_groups=request.test_groups,
        target_test_rows=sum(
            group in request.test_groups and scale == 1.0 and weight > 0.0
            for group, scale, weight in zip(groups, scales, weights, strict=True)
        ),
        observed_test_rows=sum(
            group in request.test_groups and scale == 1.0 and weight > 0.0
            for group, scale, weight in zip(groups, scales, weights, strict=True)
        ),
    )
    training = train_canonical_xgb(
        feature_rows,
        labels,
        groups,
        actions,
        weights,
        scales=scales,
        split=split,
        config=CanonicalTrainingConfig(device=request.device, thread_count=1),
    )
    _require(
        training.split == split
        and training.feature_order == CANONICAL_FEATURE_ORDER
        and training.fixed_threshold == CANONICAL_DECISION_THRESHOLD,
        "fresh transfer trainer returned a changed method or split",
    )
    _require(
        variance_count >= 2,
        "transfer training rows cannot estimate observed projection variance",
    )
    observed_variance = variance_m2 / float(variance_count - 1)
    _require(
        np.all(np.isfinite(observed_variance))
        and np.all(observed_variance >= 0.0),
        "observed transfer PCA projection variance is invalid",
    )
    return CanonicalTransferTrainingResult(
        training=training,
        observed_train_projection_variance=tuple(
            float(value) for value in observed_variance
        ),
        baseline_feature_rows=observed_baseline_rows,
        reviewed_baseline_feature_rows=reviewed_baseline_rows,
        unreviewed_baseline_feature_rows=unreviewed_baseline_rows,
        cumulative_al_feature_rows=len(al_rows),
        effective_training_feature_rows=effective_training_rows,
        positive_training_feature_rows=positive_training_rows,
        heldout_parity_rows=tuple(heldout_parity_rows),
        baseline_action_adapter_policy=TRANSFER_BASELINE_ACTION_ADAPTER_POLICY,
    )


def retrain_canonical_transfer_bundle(
    request: CanonicalTransferRetrainRequest,
    output: Path,
    *,
    config_path: Path,
    expected_config_sha256: str,
    source_bundle_root: Path | None = None,
) -> None:
    """Fresh-fit and seal a Full-profile project-rN transfer bundle.

    The project config supplies only verified public extractor assets and the
    canonical config maps.  It never supplies classifier or feature state.
    Those are produced by the request and frozen published-r92 state.
    """

    from compag_curation.canonical.service import (
        _bundle_config_maps,
        _canonical_config,
    )
    from compag_curation.model_bundle import (
        BUNDLE_SCHEMA_V2,
        POST_R92_TRANSFER_CV_INTERPRETATION,
        POST_R92_TRANSFER_FEATURE_STATE_SCOPE,
        POST_R92_TRANSFER_FROZEN_STATE_POLICY,
        POST_R92_TRANSFER_LINEAGE_POLICY,
        POST_R92_TRANSFER_REPRODUCTION_CLAIM,
        POST_R92_TRANSFER_WORKFLOW,
        BundleV2WriteRequest,
        verify_model_bundle,
        write_model_bundle_v2,
    )
    from compag_curation.public_config import (
        check_local_assets,
        load_public_config,
    )

    _sha256(expected_config_sha256, "transfer project config")
    config = load_public_config(config_path, check_local=False)
    _canonical_config(config)
    _require(
        config.profile == CANONICAL_GPU_PROFILE
        and config.device == "cuda"
        and config.config_sha256 == expected_config_sha256,
        "transfer retraining requires the bound canonical Full CUDA config",
    )
    asset_check = check_local_assets(config)
    _require(
        asset_check.get("status") == "PASS"
        and isinstance(asset_check.get("assets"), list)
        and len(asset_check["assets"]) == 5,
        "transfer retraining did not verify all five Full-profile assets",
    )
    output = output.absolute()
    portable_basename(output.name, "transfer model-bundle output basename")
    try:
        output_parent = output.parent.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError("transfer model-bundle output parent is unavailable") from exc
    _require(
        output_parent == output.parent
        and output_parent.is_dir()
        and not output_parent.is_symlink()
        and not output.exists()
        and not output.is_symlink(),
        "transfer model-bundle output is unsafe or already exists",
    )
    maps = _bundle_config_maps(config)
    normalized, preprocessing, proposal, features, training_config = maps
    source = None
    if source_bundle_root is not None:
        source_path = source_bundle_root.absolute().resolve(strict=True)
        _require(
            source_path != output
            and source_path not in output.parents
            and output not in source_path.parents,
            "transfer output cannot overlap its project parent bundle",
        )
        source = verify_model_bundle(source_path)
        _require(
            source.schema == BUNDLE_SCHEMA_V2
            and source.profile == CANONICAL_GPU_PROFILE
            and source.normalized_config == normalized
            and source.preprocessing == preprocessing
            and source.proposal_config == proposal
            and source.feature_config == features
            and source.training_config == training_config,
            "transfer config/assets differ from the verified project parent bundle",
        )

    trained = train_canonical_transfer_request(request)
    _require(
        trained.baseline_action_adapter_policy
        == TRANSFER_BASELINE_ACTION_ADAPTER_POLICY,
        "transfer baseline action adapter policy changed",
    )
    booster = trained.training.classifier.get_booster()
    booster.feature_names = list(CANONICAL_FEATURE_ORDER)
    classifier_ubj = portable_classifier_bytes(trained.training.classifier)
    parity = verify_probability_parity(
        trained.training.classifier,
        trained.training.imputer_statistics,
        trained.heldout_parity_rows,
        classifier_ubj=classifier_ubj,
        device="cuda",
    )
    _require(
        parity.get("status") == "PASS",
        "transfer UBJ probability parity did not pass",
    )
    extras = dict(request.expected_bundle_provenance)
    _require(
        extras.get("active_learning_workflow") == POST_R92_TRANSFER_WORKFLOW
        and extras.get("lineage_policy") == POST_R92_TRANSFER_LINEAGE_POLICY
        and extras.get("reproduction_claim")
        == POST_R92_TRANSFER_REPRODUCTION_CLAIM
        and extras.get("frozen_feature_state_policy")
        == POST_R92_TRANSFER_FROZEN_STATE_POLICY,
        "transfer bundle request lineage policy changed",
    )
    combined_feature_identity = compact_json_sha256(
        {
            "domain": "compag-curation-transfer-combined-feature-input/v1",
            "transfer_baseline_sha256": request.baseline.archive_sha256,
            "cumulative_al_rows_sha256": request.cumulative_al_rows_sha256,
        }
    )
    unavailable_fit_group_identity = compact_json_sha256(
        {
            "domain": "compag-curation-frozen-r92-fit-groups-unavailable/v1",
            "r92_feature_state_sha256": extras["r92_feature_state_sha256"],
        }
    )
    unavailable_fit_row_identity = compact_json_sha256(
        {
            "domain": "compag-curation-frozen-r92-fit-rows-unavailable/v1",
            "r92_feature_state_sha256": extras["r92_feature_state_sha256"],
        }
    )
    provenance = {
        "profile": CANONICAL_GPU_PROFILE,
        "device": "cuda",
        "reviewed_sha256": request.decision_log_sha256,
        "split_manifest_sha256": request.baseline.split_sha256,
        "raw_features_sha256": combined_feature_identity,
        "feature_state_fit_group_sha256": unavailable_fit_group_identity,
        "feature_state_fit_rows_sha256": unavailable_fit_row_identity,
        "feature_state_fit_scope": POST_R92_TRANSFER_FEATURE_STATE_SCOPE,
        "validation_feature_scale": 1.0,
        "cv_score_interpretation": POST_R92_TRANSFER_CV_INTERPRETATION,
        "training_row_count": trained.effective_training_feature_rows,
        "positive_training_row_count": trained.positive_training_feature_rows,
        "fixed_threshold": CANONICAL_DECISION_THRESHOLD,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        "ubj_parity_status": "PASS",
        "paper_result_reproduction": "NOT_CLAIMED",
        **extras,
    }
    state_payloads = serialize_canonical_feature_state(request.feature_state)
    compatibility = {
        "classifier_format": "XGBOOST_UBJ",
        "no_pickle_or_joblib": True,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        "fixed_threshold": CANONICAL_DECISION_THRESHOLD,
    }
    _require(config.embedding_weights is not None, "transfer ResNet50 asset is missing")
    write_model_bundle_v2(
        BundleV2WriteRequest(
            output=output,
            profile=CANONICAL_GPU_PROFILE,
            classifier_ubj=classifier_ubj,
            feature_order=CANONICAL_FEATURE_ORDER,
            feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
            imputer_statistics={
                name: float(value)
                for name, value in zip(
                    CANONICAL_FEATURE_ORDER,
                    trained.training.imputer_statistics,
                    strict=True,
                )
            },
            prototype_npy=state_payloads.prototype_npy,
            pca_mean_npy=state_payloads.pca_mean_npy,
            pca_components_npy=state_payloads.pca_components_npy,
            pca_explained_variance=trained.observed_train_projection_variance,
            normalized_config=normalized,
            preprocessing=preprocessing,
            proposal_config=proposal,
            feature_config=features,
            training_config=training_config,
            compatibility=compatibility,
            provenance=provenance,
            sam2_config=config.sam2_config,
            sam2_checkpoint=config.checkpoint,
            sam2_license=config.asset_root / "SAM2-APACHE-2.0.txt",
            resnet50_weights=config.embedding_weights,
            resnet50_license=config.asset_root
            / "TORCHVISION-BSD-3-CLAUSE.txt",
        )
    )
    written = verify_model_bundle(output)
    config_after = load_public_config(config_path, check_local=False)
    asset_check_after = check_local_assets(config_after)
    _require(
        written.schema == BUNDLE_SCHEMA_V2
        and written.profile == CANONICAL_GPU_PROFILE
        and written.provenance.get("details") == provenance
        and written.prototype_npy == state_payloads.prototype_npy
        and written.pca_mean_npy == state_payloads.pca_mean_npy
        and written.pca_components_npy == state_payloads.pca_components_npy
        and config_after.config_sha256 == config.config_sha256
        and asset_check_after == asset_check,
        "written transfer bundle, config, or public assets changed during retraining",
    )
    if source is not None:
        source_after = verify_model_bundle(source.root)
        _require(
            source_after.bundle_sha256 == source.bundle_sha256
            and source_after.tree_identity == source.tree_identity,
            "project parent bundle changed during transfer retraining",
        )


def read_canonical_active_learning_pool(
    stage_root: Path,
) -> tuple[CanonicalActiveLearningPoolRow, ...]:
    """Return exact scale-1 XGB pool rows from a bound stage-60 output."""

    return read_canonical_active_learning_inference(stage_root).pool_rows


def read_canonical_active_learning_raw_feature_archive(
    stage_root: Path,
) -> tuple[CanonicalActiveLearningTrainingRow, ...]:
    """Return the exact four-scale feature archive from a bound stage-60 output."""

    return read_canonical_active_learning_inference(stage_root).feature_rows


def _request_decisions(
    request: CanonicalActiveLearningRetrainRequest,
) -> dict[str, CanonicalActiveLearningDecision]:
    decisions: dict[str, CanonicalActiveLearningDecision] = {}
    prior_proposal = ""
    for decision in request.effective_decisions:
        _require(
            decision.proposal_id > prior_proposal
            and decision.proposal_id not in decisions
            and decision.proposal_sha256 == decision.proposal_id
            and _SHA256.fullmatch(decision.proposal_id) is not None
            and _SHA256.fullmatch(decision.image_id) is not None
            and _SHA256.fullmatch(decision.image_sha256) is not None
            and _GROUP.fullmatch(decision.group_id) is not None
            and decision.label in {0, 1}
            and decision.review_action in REVIEW_ACTION_WEIGHTS
            and decision.review_weight == REVIEW_ACTION_WEIGHTS[decision.review_action],
            "canonical AL retrain decision closure changed",
        )
        prior_proposal = decision.proposal_id
        decisions[decision.proposal_id] = decision
    _require(bool(decisions), "canonical AL retrain decisions are empty")
    return decisions


def _validated_retrain_rows(
    request: CanonicalActiveLearningRetrainRequest,
) -> tuple[
    tuple[Mapping[str, float], ...],
    tuple[int, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    import numpy as np

    decisions = _request_decisions(request)
    _require(
        bool(request.train_groups)
        and bool(request.test_groups)
        and not request.train_groups & request.test_groups,
        "canonical AL retrain split is invalid",
    )
    rows = tuple(request.training_rows)
    _require(bool(rows), "canonical AL retrain feature rows are empty")
    observed: set[tuple[str, float]] = set()
    prior_key: tuple[str, float] | None = None
    finalized: list[Mapping[str, float]] = []
    labels: list[int] = []
    groups: list[str] = []
    actions: list[str] = []
    weights: list[float] = []
    scales: list[float] = []
    sources_by_proposal: dict[str, set[str]] = {}
    for row in rows:
        _validate_raw_feature(row)
        decision = decisions.get(row.proposal_id)
        _require(decision is not None, "canonical AL retrain feature has no decision")
        _sha256(row.source_sha256, "canonical AL retrain feature source")
        _require(
            row.proposal_sha256 == decision.proposal_sha256
            and row.image_id == decision.image_id
            and row.image_sha256 == decision.image_sha256
            and row.group_id == decision.group_id
            and row.group_id in request.train_groups | request.test_groups,
            "canonical AL retrain feature identity changed",
        )
        scale = float(row.raw_feature.scale)
        key = (row.proposal_id, scale)
        _require(
            scale in CANONICAL_FEATURE_CROP_SCALES
            and key not in observed
            and (prior_key is None or prior_key < key),
            "canonical AL retrain features are duplicated or unordered",
        )
        prior_key = key
        observed.add(key)
        sources_by_proposal.setdefault(row.proposal_id, set()).add(
            row.source_sha256
        )
        finalized_row = finalize_canonical_feature(row.raw_feature, request.feature_state)
        _require(
            tuple(finalized_row) == CANONICAL_FEATURE_ORDER,
            "canonical AL retrain finalized feature schema changed",
        )
        finalized.append(finalized_row)
        labels.append(decision.label)
        groups.append(decision.group_id)
        actions.append(decision.review_action)
        weights.append(decision.review_weight)
        scales.append(scale)
    expected = {
        (proposal_id, float(scale))
        for proposal_id in decisions
        for scale in CANONICAL_FEATURE_CROP_SCALES
    }
    _require(
        observed == expected
        and all(len(sources) == 1 for sources in sources_by_proposal.values())
        and {decision.group_id for decision in decisions.values()}
        <= request.train_groups | request.test_groups,
        "canonical AL retrain feature identity, scale, or source closure changed",
    )
    effective_train = tuple(
        row
        for row in rows
        if row.group_id in request.train_groups
        and decisions[row.proposal_id].review_weight > 0.0
    )
    positive_embeddings = tuple(
        np.asarray(row.raw_feature.embedding, dtype=np.float32)
        for row in effective_train
        if decisions[row.proposal_id].label == 1
    )
    _require(
        len(effective_train) == request.feature_state.training_row_count
        and len(positive_embeddings) == request.feature_state.positive_row_count
        and bool(positive_embeddings),
        "canonical AL retrain feature-state row counts changed",
    )
    matrix = np.vstack(positive_embeddings).astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    _require(bool(np.all(norms > 0.0)), "canonical AL retrain positive embedding has zero norm")
    prototype = (matrix / norms).mean(axis=0)
    prototype_norm = float(np.linalg.norm(prototype))
    _require(
        math.isfinite(prototype_norm) and prototype_norm > 0.0,
        "canonical AL retrain positive prototype has zero norm",
    )
    expected_prototype = np.ascontiguousarray(
        prototype / prototype_norm,
        dtype=np.float32,
    )
    actual_prototype = np.ascontiguousarray(
        request.feature_state.prototype,
        dtype=np.float32,
    )
    _require(
        actual_prototype.tobytes(order="C")
        == expected_prototype.tobytes(order="C"),
        "canonical AL retrain prototype was not recomputed from outer-train positives",
    )
    return (
        tuple(finalized),
        tuple(labels),
        tuple(groups),
        tuple(actions),
        tuple(weights),
        tuple(scales),
    )


def retrain_canonical_active_learning_bundle(
    request: CanonicalActiveLearningRetrainRequest,
    output: Path,
    *,
    source_bundle_root: Path,
) -> None:
    """Train and seal one v2 bundle from a validated canonical AL request."""

    from compag_curation.model_bundle import (
        BUNDLE_SCHEMA_V2,
        BundleV2WriteRequest,
        verify_model_bundle,
        write_model_bundle_v2,
    )

    _require(
        isinstance(request.round_number, int)
        and not isinstance(request.round_number, bool)
        and request.round_number >= 1,
        "canonical AL retrain round number is invalid",
    )
    profile, device = _canonical_profile_device(request.profile, request.device)
    _require(
        profile in GPU_EXECUTION_PROFILES and device == "cuda",
        "active-learning retraining requires the GPU/CUDA profile: canonical Full, efficient Lite, or full-image multiscale",
    )
    output = output.absolute()
    portable_basename(output.name, "canonical AL retrain bundle basename")
    try:
        output_parent = output.parent.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError("canonical AL retrain output parent cannot be resolved") from exc
    _require(
        output_parent == output.parent
        and output_parent.is_dir()
        and not output_parent.is_symlink()
        and not output.exists()
        and not output.is_symlink(),
        "canonical AL retrain output is unsafe or already exists",
    )
    source_path = source_bundle_root.absolute()
    try:
        source_resolved = source_path.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError("canonical AL source bundle cannot be resolved") from exc
    _require(
        source_path == source_resolved
        and source_resolved != output
        and source_resolved not in output.parents
        and output not in source_resolved.parents,
        "canonical AL retrain output overlaps its source bundle",
    )
    validate_canonical_feature_state(request.feature_state)
    source = verify_model_bundle(source_resolved)
    _require(
        source.schema == BUNDLE_SCHEMA_V2
        and source.profile == profile
        and source.normalized_config.get("profile") == profile
        and source.normalized_config.get("device") == device
        and source.bundle_sha256 == request.source_bundle_sha256
        and tuple(source.feature_order) == CANONICAL_FEATURE_ORDER
        and source.feature_order_sha256 == CANONICAL_FEATURE_ORDER_SHA256
        and source.threshold == CANONICAL_DECISION_THRESHOLD,
        "canonical AL source bundle identity changed",
    )
    state_payloads = serialize_canonical_feature_state(request.feature_state)
    state_hashes = {
        "prototype_sha256": hashlib.sha256(state_payloads.prototype_npy).hexdigest(),
        "pca_components_sha256": hashlib.sha256(
            state_payloads.pca_components_npy
        ).hexdigest(),
        "pca_mean_sha256": hashlib.sha256(state_payloads.pca_mean_npy).hexdigest(),
    }
    _require(
        request.prototype_sha256 == state_hashes["prototype_sha256"]
        and request.pca_components_sha256 == state_hashes["pca_components_sha256"]
        and request.pca_mean_sha256 == state_hashes["pca_mean_sha256"]
        and state_payloads.pca_components_npy == source.pca_components_npy
        and state_payloads.pca_mean_npy == source.pca_mean_npy,
        "canonical AL retrain feature-state receipt or frozen PCA changed",
    )
    for value, role in (
        (request.decision_log_sha256, "decision log"),
        (request.training_rows_sha256, "training rows"),
        (request.frozen_split_sha256, "frozen split"),
    ):
        _sha256(value, f"canonical AL retrain {role}")
    finalized, labels, groups, actions, weights, scales = _validated_retrain_rows(
        request
    )
    latest = _request_decisions(request)
    computed_training_hash = _training_row_hash(request.training_rows, latest)
    _require(
        request.training_rows_sha256 == computed_training_hash,
        "canonical AL retrain training-row receipt changed",
    )
    lineage_fields = (
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
    )
    lineage = (
        {
            name: request.expected_bundle_provenance[name]
            for name in lineage_fields
        }
        if "active_learning_workflow" in request.expected_bundle_provenance
        else None
    )
    expected_provenance = _expected_bundle_provenance(
        request.training_rows,
        latest,
        request.train_groups,
        decision_log_sha256=request.decision_log_sha256,
        split_sha256=request.frozen_split_sha256,
        training_rows_sha256=computed_training_hash,
        state=request.feature_state,
        profile=(
            profile
            if "profile" in request.expected_bundle_provenance
            or profile != CANONICAL_PROFILE
            else None
        ),
        device=(
            device
            if "device" in request.expected_bundle_provenance
            or profile != CANONICAL_PROFILE
            else None
        ),
        lineage=lineage,
    )
    _require(
        dict(request.expected_bundle_provenance) == expected_provenance,
        "canonical AL expected provenance differs from validated retrain inputs",
    )
    split = CanonicalGroupSplit(
        train_groups=request.train_groups,
        test_groups=request.test_groups,
        target_test_rows=sum(
            decision.group_id in request.test_groups
            and decision.review_weight > 0.0
            for decision in request.effective_decisions
        ),
        observed_test_rows=sum(
            decision.group_id in request.test_groups
            and decision.review_weight > 0.0
            for decision in request.effective_decisions
        ),
    )
    training = train_canonical_xgb(
        finalized,
        labels,
        groups,
        actions,
        weights,
        scales=scales,
        split=split,
        config=CanonicalTrainingConfig(
            device=device,
            thread_count=1,
        ),
    )
    _require(
        training.feature_order == CANONICAL_FEATURE_ORDER
        and training.fixed_threshold == CANONICAL_DECISION_THRESHOLD
        and training.split == split,
        "canonical AL trainer returned a changed method or split",
    )
    booster = training.classifier.get_booster()
    booster.feature_names = list(CANONICAL_FEATURE_ORDER)
    classifier_ubj = portable_classifier_bytes(training.classifier)
    parity_rows = tuple(
        row
        for row, group, action, scale in zip(
            finalized,
            groups,
            actions,
            scales,
            strict=True,
        )
        if group in request.test_groups
        and REVIEW_ACTION_WEIGHTS[action] > 0.0
        and scale == 1.0
    )
    _require(bool(parity_rows), "canonical AL retrain has no held-out parity rows")
    parity = verify_probability_parity(
        training.classifier,
        training.imputer_statistics,
        parity_rows,
        classifier_ubj=classifier_ubj,
        device=device,
    )
    _require(parity.get("status") == "PASS", "canonical AL UBJ parity did not pass")
    compatibility = {
        "classifier_format": "XGBOOST_UBJ",
        "no_pickle_or_joblib": True,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        "fixed_threshold": CANONICAL_DECISION_THRESHOLD,
    }
    _require(
        source.feature_config is not None
        and source.training_config is not None
        and source.pca_explained_variance is not None
        and source.resnet50_weights is not None,
        "canonical AL source bundle omits v2 training state",
    )
    write_model_bundle_v2(
        BundleV2WriteRequest(
            output=output,
            profile=profile,
            classifier_ubj=classifier_ubj,
            feature_order=CANONICAL_FEATURE_ORDER,
            feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
            imputer_statistics={
                name: float(value)
                for name, value in zip(
                    CANONICAL_FEATURE_ORDER,
                    training.imputer_statistics,
                    strict=True,
                )
            },
            prototype_npy=state_payloads.prototype_npy,
            pca_mean_npy=state_payloads.pca_mean_npy,
            pca_components_npy=state_payloads.pca_components_npy,
            pca_explained_variance=tuple(source.pca_explained_variance),
            normalized_config=source.normalized_config,
            preprocessing=source.preprocessing,
            proposal_config=source.proposal_config,
            feature_config=source.feature_config,
            training_config=source.training_config,
            compatibility=compatibility,
            provenance=expected_provenance,
            sam2_config=source.sam2_config,
            sam2_checkpoint=source.checkpoint,
            sam2_license=source.root / "licenses/SAM2-APACHE-2.0.txt",
            resnet50_weights=source.resnet50_weights,
            resnet50_license=source.root
            / "licenses/TORCHVISION-BSD-3-CLAUSE.txt",
        )
    )
    written = verify_model_bundle(output)
    source_after = verify_model_bundle(source_resolved)
    _require(
        written.schema == BUNDLE_SCHEMA_V2
        and written.profile == profile
        and written.provenance.get("details") == expected_provenance
        and written.prototype_npy == state_payloads.prototype_npy
        and written.pca_mean_npy == state_payloads.pca_mean_npy
        and written.pca_components_npy == state_payloads.pca_components_npy
        and source_after.bundle_sha256 == source.bundle_sha256
        and source_after.tree_identity == source.tree_identity,
        "canonical AL written bundle or source bundle verification failed",
    )


__all__ = [
    "CANONICAL_AL_RAW_FEATURE_ARCHIVE_SCHEMA",
    "CanonicalActiveLearningInferenceArchive",
    "CanonicalR92TransferInference",
    "CanonicalTransferTrainingResult",
    "read_canonical_active_learning_inference",
    "read_canonical_r92_transfer_inference",
    "read_canonical_active_learning_pool",
    "read_canonical_active_learning_raw_feature_archive",
    "read_canonical_stage20_raw_features",
    "retrain_canonical_transfer_bundle",
    "retrain_canonical_active_learning_bundle",
    "train_canonical_transfer_request",
    "write_canonical_active_learning_raw_feature_archive",
    "write_initial_canonical_active_learning_pool",
]
