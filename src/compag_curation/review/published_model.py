"""Hash-closed historical CJ transfer assistance for Stage-20 review.

The public runtime never imports the original joblib/pickle artifact.  A
one-time offline migration converts the manuscript-selected r92 estimator to
XGBoost UBJ and plain JSON/NPY numerical state.  This module verifies that
portable closure, reconstructs the exact 93 predictors from sealed Stage-20
raw evidence, and performs CUDA-only scoring.

The preset is intentionally described as *transfer assistance*.  Its
historical PCA/prototype lineage is preserved but is not claimed numerically
equivalent to a model freshly fitted by the current canonical pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from compag_curation.canonical.features import (
    CANONICAL_RAW_FEATURE_ORDER,
    CanonicalFeatureState,
    CanonicalRawFeature,
    decode_canonical_embedding,
    finalize_canonical_feature,
    validate_canonical_feature_state,
)
from compag_curation.canonical.spec import (
    CANONICAL_FEATURE_CROP_SCALES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_GPU_PROFILE,
)
from compag_curation.public_io import (
    PublicIOError,
    bounded_csv_field_limit,
    canonical_json_bytes,
    canonical_json_value,
    compact_json_sha256,
    stable_file,
)


PRESET_ID = "compag-cj-r92"
PRESET_DISPLAY_NAME = "Project 1 — CJ · manuscript-selected r92 transfer-assist"
PRESET_THRESHOLD = 0.50
PRESET_RESOURCE_MANIFEST_SHA256 = (
    "0fb259e5d4bb366d4368068b32f26d54dd361892ed3811c4824d9cdac4e405df"
)
PRESET_SCHEMA = "compag-curation-published-model-preset/v1"
ASSIST_RECEIPT_SCHEMA = "compag-curation-published-model-assist/v1"
SCORE_SCHEMA = "compag-curation-published-model-scores/v1"
ASSISTANCE_FILE_SCHEMA = "compag-curation-published-model-assistance-cache/v1"
MAX_RESOURCE_FILE_BYTES = 64 * 1024 * 1024
MAX_STAGE20_FEATURE_BYTES = 2 * 1024 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PublishedModelScore:
    probability: float
    prediction: int
    uncertainty: float


@dataclass(frozen=True)
class PublishedModelContext:
    preset_id: str
    display_name: str
    threshold: float
    order: tuple[str, ...]
    scores: Mapping[str, PublishedModelScore]
    scene_models: Mapping[str, Mapping[str, object]]
    resource_manifest_sha256: str
    classifier_sha256: str
    feature_state_sha256: str
    feature_source_sha256: str
    scores_sha256: str
    navigation_order_sha256: str
    assistance_record: Mapping[str, object]
    assistance_file_sha256: str

    def score(self, proposal_id: str) -> PublishedModelScore:
        try:
            return self.scores[proposal_id]
        except KeyError as exc:
            raise PublicIOError(
                "published-model score is missing for one proposal"
            ) from exc

    def scene_records(self) -> Mapping[str, Mapping[str, object]]:
        return self.scene_models

    def receipt_record(self) -> dict[str, object]:
        return {
            "schema": ASSIST_RECEIPT_SCHEMA,
            "preset_id": self.preset_id,
            "display_name": self.display_name,
            "assist_mode": "HISTORICAL_R92_TRANSFER_ASSIST",
            "profile": CANONICAL_GPU_PROFILE,
            "threshold_method": "FIXED_MANUSCRIPT_THRESHOLD",
            "threshold": self.threshold,
            "resource_manifest_sha256": self.resource_manifest_sha256,
            "classifier_sha256": self.classifier_sha256,
            "feature_state_sha256": self.feature_state_sha256,
            "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
            "feature_source_sha256": self.feature_source_sha256,
            "scores_sha256": self.scores_sha256,
            "navigation_order_sha256": self.navigation_order_sha256,
            "assistance_file_sha256": self.assistance_file_sha256,
            "proposal_count": len(self.order),
            "device": "cuda",
            "lineage_policy": "TRANSFER_ASSIST_NOT_FRESH_CANONICAL_EQUIVALENCE",
            "remainder_policy": "EXPLICIT_USER_CONFIRMATION_AS_REDUCED_WEIGHT_0.4",
        }

    def summary_record(self) -> dict[str, object]:
        return {
            "schema": ASSIST_RECEIPT_SCHEMA,
            "preset_id": self.preset_id,
            "display_name": self.display_name,
            "threshold": self.threshold,
            "threshold_method": "FIXED_MANUSCRIPT_THRESHOLD",
            "proposal_count": len(self.order),
            "scores_sha256": self.scores_sha256,
            "navigation_order_sha256": self.navigation_order_sha256,
            "device": "cuda",
            "lineage_policy": "TRANSFER_ASSIST_NOT_FRESH_CANONICAL_EQUIVALENCE",
        }


@dataclass(frozen=True)
class PublishedModelTransferAssets:
    """Verified portable state needed by the explicit r92 transfer workflow.

    This is deliberately narrower than a canonical v2 model bundle.  In
    particular it carries no training corpus, split, decision log, SAM2
    checkpoint, or ResNet50 weights.  Those inputs remain separate and are
    hash-bound by the transfer operation that consumes this object.
    """

    preset_id: str
    profile: str
    threshold: float
    classifier_ubj: bytes
    classifier_sha256: str
    imputer_statistics: tuple[float, ...]
    feature_state: CanonicalFeatureState
    feature_state_sha256: str
    resource_manifest_sha256: str


@dataclass(frozen=True)
class _PresetAssets:
    manifest_sha256: str
    classifier_ubj: bytes
    classifier_sha256: str
    imputer_statistics: tuple[float, ...]
    feature_state: CanonicalFeatureState
    feature_state_sha256: str


def load_published_model_transfer_assets(
    preset_id: str = PRESET_ID,
) -> PublishedModelTransferAssets:
    """Return the hash-verified r92 scorer/state without inventing lineage.

    The returned object is suitable only for the separately labelled
    ``POST_R92_REVIEWED_TRANSFER_BASELINE`` workflow.  Loading it does not
    convert the preset into a canonical v2 parent bundle and does not imply
    that the historical r92 fit can be reproduced from the public package.
    """

    _require(preset_id == PRESET_ID, "published-model preset is unsupported")
    assets = _load_assets()
    return PublishedModelTransferAssets(
        preset_id=PRESET_ID,
        profile=CANONICAL_GPU_PROFILE,
        threshold=PRESET_THRESHOLD,
        classifier_ubj=assets.classifier_ubj,
        classifier_sha256=assets.classifier_sha256,
        imputer_statistics=assets.imputer_statistics,
        feature_state=assets.feature_state,
        feature_state_sha256=assets.feature_state_sha256,
        resource_manifest_sha256=assets.manifest_sha256,
    )


def _score_rows(
    scores: Mapping[str, PublishedModelScore],
) -> list[dict[str, object]]:
    return [
        {
            "proposal_id": proposal_id,
            "probability_hex": score.probability.hex(),
            "prediction": score.prediction,
        }
        for proposal_id, score in sorted(scores.items())
    ]


def _scores_sha256(
    *,
    feature_source_sha256: str,
    scores: Mapping[str, PublishedModelScore],
) -> str:
    return compact_json_sha256(
        {
            "schema": SCORE_SCHEMA,
            "preset_id": PRESET_ID,
            "threshold": PRESET_THRESHOLD,
            "feature_source_sha256": feature_source_sha256,
            "rows": _score_rows(scores),
        }
    )


def _context(
    *,
    snapshot: Any,
    assets: _PresetAssets,
    order: tuple[str, ...],
    scores: Mapping[str, PublishedModelScore],
    feature_source_sha256: str,
) -> PublishedModelContext:
    scores_sha256 = _scores_sha256(
        feature_source_sha256=feature_source_sha256,
        scores=scores,
    )
    navigation_order_sha256 = compact_json_sha256(list(order))
    assistance_record: dict[str, object] = {
        "schema": ASSISTANCE_FILE_SCHEMA,
        "preset_id": PRESET_ID,
        "display_name": PRESET_DISPLAY_NAME,
        "assist_mode": "HISTORICAL_R92_TRANSFER_ASSIST",
        "profile": CANONICAL_GPU_PROFILE,
        "threshold_method": "FIXED_MANUSCRIPT_THRESHOLD",
        "threshold": PRESET_THRESHOLD,
        "run_id": snapshot.run_id,
        "review_request_sha256": snapshot.review_request_sha256,
        "stage20_receipt_sha256": snapshot.stage20_receipt_sha256,
        "feature_source_sha256": feature_source_sha256,
        "resource_manifest_sha256": assets.manifest_sha256,
        "classifier_sha256": assets.classifier_sha256,
        "feature_state_sha256": assets.feature_state_sha256,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        "scores_sha256": scores_sha256,
        "proposal_count": len(order),
        "navigation_order": list(order),
        "navigation_order_sha256": navigation_order_sha256,
        "scores": _score_rows(scores),
        "device": "cuda",
        "lineage_policy": "TRANSFER_ASSIST_NOT_FRESH_CANONICAL_EQUIVALENCE",
    }
    assistance_file_sha256 = hashlib.sha256(
        canonical_json_bytes(assistance_record)
    ).hexdigest()
    scene_models = {
        proposal_id: {
            "preset_id": PRESET_ID,
            "xgb_p": score.probability,
            "prediction": score.prediction,
            "uncertainty": score.uncertainty,
            "threshold": PRESET_THRESHOLD,
        }
        for proposal_id, score in scores.items()
    }
    return PublishedModelContext(
        preset_id=PRESET_ID,
        display_name=PRESET_DISPLAY_NAME,
        threshold=PRESET_THRESHOLD,
        order=order,
        scores=dict(scores),
        scene_models=scene_models,
        resource_manifest_sha256=assets.manifest_sha256,
        classifier_sha256=assets.classifier_sha256,
        feature_state_sha256=assets.feature_state_sha256,
        feature_source_sha256=feature_source_sha256,
        scores_sha256=scores_sha256,
        navigation_order_sha256=navigation_order_sha256,
        assistance_record=assistance_record,
        assistance_file_sha256=assistance_file_sha256,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicIOError(message)


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


def _resource_root() -> Path:
    return Path(__file__).parent.parent / "resources/presets/compag_cj_r92"


def _resource_payload(path: Path, expected: Mapping[str, object]) -> bytes:
    _require(path.parent == _resource_root(), "preset resource escaped its closure")
    info = path.lstat()
    _require(
        stat.S_ISREG(info.st_mode)
        and not path.is_symlink()
        and info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == 0o644
        and expected.get("mode_octal") == "0644"
        and info.st_size == expected.get("size_bytes")
        and 0 < info.st_size <= MAX_RESOURCE_FILE_BYTES,
        f"preset resource metadata is invalid: {path.name}",
    )
    payload, _snapshot = stable_file(path, max_bytes=MAX_RESOURCE_FILE_BYTES)
    _require(
        hashlib.sha256(payload).hexdigest() == expected.get("sha256"),
        f"preset resource hash changed: {path.name}",
    )
    return payload


def _npy(payload: bytes, shape: tuple[int, ...], role: str) -> Any:
    import numpy as np

    _require(payload.startswith(b"\x93NUMPY"), f"{role} is not NPY")
    handle = io.BytesIO(payload)
    try:
        array = np.load(handle, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise PublicIOError(f"{role} NPY cannot be decoded") from exc
    _require(handle.tell() == len(payload), f"{role} NPY has trailing bytes")
    array = np.asarray(array)
    _require(
        array.dtype.kind == "f"
        and array.dtype.itemsize == 4
        and array.shape == shape
        and bool(np.all(np.isfinite(array))),
        f"{role} NPY has invalid dtype, shape, or values",
    )
    result = np.ascontiguousarray(array, dtype=np.float32)
    result.setflags(write=False)
    return result


def _load_assets() -> _PresetAssets:
    root = _resource_root()
    root_info = root.lstat()
    _require(
        stat.S_ISDIR(root_info.st_mode) and not root.is_symlink(),
        "published-model preset directory is unsafe",
    )
    manifest_path = root / "preset.json"
    manifest_payload, _identity = stable_file(
        manifest_path,
        max_bytes=1024 * 1024,
    )
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    _require(
        manifest_sha256 == PRESET_RESOURCE_MANIFEST_SHA256,
        "published-model preset manifest identity changed",
    )
    manifest = canonical_json_value(
        manifest_payload,
        "published-model preset manifest",
    )
    required_manifest_fields = {
        "schema",
        "preset_id",
        "display_name",
        "positive_label",
        "negative_label",
        "scientific_profile",
        "required_stage20_profile",
        "portable_runtime_format",
        "classifier",
        "imputer",
        "feature_schema",
        "feature_state",
        "threshold",
        "class_map",
        "provenance",
        "probability_parity",
        "fixed_threshold",
        "feature_count",
        "feature_order_sha256",
        "payload_tree_sha256",
        "members",
    }
    _require(
        isinstance(manifest, dict)
        and set(manifest) == required_manifest_fields
        and manifest.get("schema") == PRESET_SCHEMA
        and manifest.get("preset_id") == PRESET_ID
        and manifest.get("display_name")
        == "COMPAG CJ — published XGBoost r92"
        and manifest.get("positive_label") == "CJ"
        and manifest.get("negative_label") == "Non-CJ"
        and manifest.get("scientific_profile") == CANONICAL_GPU_PROFILE
        and manifest.get("required_stage20_profile") == CANONICAL_GPU_PROFILE
        and manifest.get("portable_runtime_format") == "safe-ubj-json-npy/v1"
        and manifest.get("classifier") == "classifier.ubj"
        and manifest.get("imputer") == "imputer.json"
        and manifest.get("feature_schema") == "feature_schema.json"
        and manifest.get("feature_state") == "feature_state.json"
        and manifest.get("threshold") == "threshold.json"
        and manifest.get("class_map") == "class_map.json"
        and manifest.get("provenance") == "provenance.json"
        and manifest.get("probability_parity") == "probability_parity.json"
        and manifest.get("fixed_threshold") == PRESET_THRESHOLD
        and manifest.get("feature_count") == len(CANONICAL_FEATURE_ORDER)
        and manifest.get("feature_order_sha256")
        == CANONICAL_FEATURE_ORDER_SHA256,
        "published-model preset contract changed",
    )
    raw_files = manifest.get("members")
    _require(isinstance(raw_files, list), "preset resource list is invalid")
    files: dict[str, Mapping[str, object]] = {}
    for row in raw_files:
        _require(
            isinstance(row, dict)
            and set(row) == {"path", "sha256", "size_bytes", "mode_octal"}
            and isinstance(row.get("path"), str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(row["path"]))
            and isinstance(row.get("sha256"), str)
            and _SHA256.fullmatch(str(row["sha256"])) is not None
            and type(row.get("size_bytes")) is int
            and int(row["size_bytes"]) > 0
            and row.get("mode_octal") == "0644"
            and str(row["path"]) not in files,
            "preset resource manifest row is invalid",
        )
        files[str(row["path"])] = row
    expected_names = {
        "classifier.ubj",
        "imputer.json",
        "feature_schema.json",
        "feature_state.json",
        "prototype.npy",
        "pca32_mean.npy",
        "pca32_components.npy",
        "threshold.json",
        "class_map.json",
        "provenance.json",
        "probability_parity.json",
    }
    _require(set(files) == expected_names, "preset resource closure changed")
    _require(
        {path.name for path in root.iterdir()} == {"preset.json", *expected_names},
        "preset resource directory contains an unexpected member",
    )
    payloads = {
        name: _resource_payload(root / name, files[name])
        for name in sorted(expected_names)
    }
    imputer = canonical_json_value(payloads["imputer.json"], "r92 imputer")
    _require(
        isinstance(imputer, dict)
        and set(imputer)
        == {
            "schema",
            "profile",
            "strategy",
            "feature_order",
            "feature_order_sha256",
            "statistics",
        }
        and imputer.get("schema") == "compag-curation-canonical-imputer/v1"
        and imputer.get("profile") == CANONICAL_GPU_PROFILE
        and imputer.get("strategy") == "median_train_only"
        and imputer.get("feature_order") == list(CANONICAL_FEATURE_ORDER)
        and imputer.get("feature_order_sha256")
        == CANONICAL_FEATURE_ORDER_SHA256
        and isinstance(imputer.get("statistics"), dict)
        and set(imputer["statistics"]) == set(CANONICAL_FEATURE_ORDER),
        "published r92 imputer contract changed",
    )
    try:
        statistics = tuple(
            float(imputer["statistics"][name])
            for name in CANONICAL_FEATURE_ORDER
        )
    except (TypeError, ValueError) as exc:
        raise PublicIOError("published r92 imputer contains nonnumeric values") from exc
    _require(
        all(math.isfinite(value) for value in statistics),
        "published r92 imputer contains nonfinite values",
    )
    feature_schema = canonical_json_value(
        payloads["feature_schema.json"], "r92 feature schema"
    )
    _require(
        isinstance(feature_schema, dict)
        and feature_schema.get("schema")
        == "compag-curation-r92-feature-schema/v1"
        and feature_schema.get("feature_count") == len(CANONICAL_FEATURE_ORDER)
        and feature_schema.get("feature_order") == list(CANONICAL_FEATURE_ORDER)
        and feature_schema.get("feature_order_sha256")
        == CANONICAL_FEATURE_ORDER_SHA256,
        "published r92 feature schema changed",
    )
    feature_state = canonical_json_value(
        payloads["feature_state.json"], "r92 feature state"
    )
    _require(
        isinstance(feature_state, dict)
        and feature_state.get("schema")
        == "compag-curation-r92-feature-state/v1"
        and feature_state.get("embedding_backbone")
        == "torchvision.models.resnet50"
        and feature_state.get("embedding_weights")
        == "ResNet50_Weights.IMAGENET1K_V2"
        and feature_state.get("embedding_dimensions") == 2048
        and isinstance(feature_state.get("prototype"), dict)
        and feature_state["prototype"].get("path") == "prototype.npy"
        and feature_state["prototype"].get("shape") == [2048]
        and isinstance(feature_state.get("pca"), dict)
        and feature_state["pca"].get("components_path")
        == "pca32_components.npy"
        and feature_state["pca"].get("components_shape") == [32, 2048]
        and feature_state["pca"].get("mean_path") == "pca32_mean.npy"
        and feature_state["pca"].get("mean_shape") == [2048]
        and feature_state.get("transform_fit_row_counts")
        == {
            "all_embeddings": None,
            "positive_embeddings": None,
            "status": "not_encoded_in_the_frozen_transform_assets",
        },
        "published r92 feature state manifest changed",
    )
    threshold = canonical_json_value(payloads["threshold.json"], "r92 threshold")
    _require(
        isinstance(threshold, dict)
        and threshold.get("schema") == "compag-curation-r92-threshold/v1"
        and threshold.get("runtime_policy") == "paper_published_fixed_threshold"
        and threshold.get("runtime_threshold") == PRESET_THRESHOLD
        and threshold.get("published_fixed_threshold") == PRESET_THRESHOLD
        and threshold.get("positive_class") == 1
        and threshold.get("positive_label") == "CJ"
        and threshold.get("legacy_training_selected_f1_threshold_runtime_role")
        == "provenance_only",
        "published r92 threshold policy changed",
    )
    class_map = canonical_json_value(payloads["class_map.json"], "r92 class map")
    _require(
        isinstance(class_map, dict)
        and class_map
        == {
            "schema": "compag-curation-r92-class-map/v1",
            "positive_class": 1,
            "classes": [
                {"classifier_value": 0, "display_label": "Non-CJ"},
                {"classifier_value": 1, "display_label": "CJ"},
            ],
        },
        "published r92 class map changed",
    )
    parity = canonical_json_value(
        payloads["probability_parity.json"], "r92 probability parity"
    )
    _require(
        isinstance(parity, dict)
        and parity.get("sample_count") == 64
        and parity.get("exact_probability_match_count") == 64
        and parity.get("maximum_absolute_difference") == 0.0
        and parity.get("published_threshold") == PRESET_THRESHOLD
        and parity.get("published_threshold_decision_match_count") == 64
        and parity.get("feature_order_sha256")
        == CANONICAL_FEATURE_ORDER_SHA256,
        "published r92 parity evidence changed",
    )
    state = CanonicalFeatureState(
        prototype=_npy(payloads["prototype.npy"], (2048,), "r92 prototype"),
        pca_components=_npy(
            payloads["pca32_components.npy"],
            (32, 2048),
            "r92 PCA components",
        ),
        pca_mean=_npy(payloads["pca32_mean.npy"], (2048,), "r92 PCA mean"),
        # The historical files do not encode the exact fitting row counts for
        # this separately frozen transform state.  Counts are unused by this
        # deterministic transform and deliberately remain unknown here.
        training_row_count=0,
        positive_row_count=0,
    )
    try:
        validate_canonical_feature_state(state)
    except ValueError as exc:
        raise PublicIOError("published r92 feature state is invalid") from exc
    feature_state_sha256 = compact_json_sha256(
        {
            name: files[name]["sha256"]
            for name in (
                "prototype.npy",
                "pca32_mean.npy",
                "pca32_components.npy",
            )
        }
    )
    return _PresetAssets(
        manifest_sha256=manifest_sha256,
        classifier_ubj=payloads["classifier.ubj"],
        classifier_sha256=str(files["classifier.ubj"]["sha256"]),
        imputer_statistics=statistics,
        feature_state=state,
        feature_state_sha256=feature_state_sha256,
    )


def _stage20_feature_identity(snapshot: Any) -> tuple[str, int]:
    receipt_path = snapshot.root / "stages/20_proposals_features/_SUCCESS.json"
    payload, _identity = stable_file(receipt_path, max_bytes=128 * 1024 * 1024)
    _require(
        hashlib.sha256(payload).hexdigest() == snapshot.stage20_receipt_sha256,
        "Stage-20 receipt changed before published-model scoring",
    )
    receipt = canonical_json_value(payload, "Stage-20 receipt")
    _require(isinstance(receipt, dict), "Stage-20 receipt is invalid")
    files = receipt.get("files")
    _require(isinstance(files, list), "Stage-20 receipt file list is invalid")
    matches = [
        row
        for row in files
        if isinstance(row, dict) and row.get("path") == "features.csv"
    ]
    _require(len(matches) == 1, "Stage-20 feature receipt is missing")
    row = matches[0]
    digest = row.get("sha256")
    size = row.get("size_bytes")
    _require(
        isinstance(digest, str)
        and _SHA256.fullmatch(digest) is not None
        and type(size) is int
        and 0 < size <= MAX_STAGE20_FEATURE_BYTES,
        "Stage-20 feature receipt identity is invalid",
    )
    return digest, size


def _finite(value: object, role: str) -> float:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise PublicIOError(f"{role} is not numeric") from exc
    _require(math.isfinite(number), f"{role} is not finite")
    return number


def _scale_one_rows(
    snapshot: Any,
    index: Any,
) -> tuple[tuple[str, ...], tuple[CanonicalRawFeature, ...], str]:
    from compag_curation.canonical.service import (
        CANONICAL_RAW_FEATURE_TABLE_COLUMNS,
    )

    expected_sha256, expected_size = _stage20_feature_identity(snapshot)
    path = snapshot.root / "stages/20_proposals_features/features.csv"
    before = path.lstat()
    _require(
        stat.S_ISREG(before.st_mode)
        and not path.is_symlink()
        and before.st_nlink == 1
        and before.st_size == expected_size,
        "Stage-20 feature table metadata is unsafe",
    )
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    selected_ids: list[str] = []
    selected_rows: list[CanonicalRawFeature] = []
    block_id: str | None = None
    block_scales: list[float] = []
    observed: set[str] = set()
    resources = ExitStack()
    try:
        resources.callback(os.close, descriptor)
        resources.enter_context(bounded_csv_field_limit(16 * 1024 * 1024))
        opened = os.fstat(descriptor)
        _require(
            _file_signature(opened) == _file_signature(before),
            "Stage-20 feature table changed while opening",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as binary:
            class _DigestReader(io.RawIOBase):
                def readable(self) -> bool:
                    return True

                def readinto(self, buffer: bytearray) -> int:
                    chunk = binary.read(len(buffer))
                    if not chunk:
                        return 0
                    digest.update(chunk)
                    buffer[: len(chunk)] = chunk
                    return len(chunk)

            buffered = io.BufferedReader(_DigestReader(), buffer_size=1024 * 1024)
            with io.TextIOWrapper(buffered, encoding="utf-8", newline="") as text:
                reader = csv.DictReader(text)
                _require(
                    tuple(reader.fieldnames or ())
                    == tuple(CANONICAL_RAW_FEATURE_TABLE_COLUMNS),
                    "Stage-20 feature header changed",
                )
                for row_number, row in enumerate(reader, start=2):
                    _require(
                        None not in row
                        and tuple(row) == tuple(CANONICAL_RAW_FEATURE_TABLE_COLUMNS),
                        f"Stage-20 feature row {row_number} has the wrong field count",
                    )
                    proposal_id = row["proposal_id"]
                    view = index.proposals.get(proposal_id)
                    _require(
                        view is not None
                        and row["proposal_sha256"] == proposal_id
                        and row["image_id"] == view.image_id
                        and row["image_name"] == view.image_name
                        and row["group_id"] == view.group_id
                        and row["tile_name"] == view.tile_name
                        and int(row["proposal_index"]) == view.proposal_index,
                        f"Stage-20 feature identity mismatch at row {row_number}",
                    )
                    scale = _finite(row["scale"], "Stage-20 feature scale")
                    if block_id != proposal_id:
                        if block_id is not None:
                            _require(
                                tuple(block_scales)
                                == tuple(CANONICAL_FEATURE_CROP_SCALES),
                                "Stage-20 feature scale block changed",
                            )
                        _require(
                            proposal_id not in observed,
                            "Stage-20 feature proposal block is duplicated",
                        )
                        observed.add(proposal_id)
                        block_id = proposal_id
                        block_scales = []
                    block_scales.append(scale)
                    predicted_iou = _finite(
                        row["predicted_iou"],
                        "Stage-20 predicted IoU",
                    )
                    stability = _finite(
                        row["stability_score"],
                        "Stage-20 stability score",
                    )
                    _require(
                        math.isclose(
                            predicted_iou,
                            view.predicted_iou,
                            rel_tol=0.0,
                            abs_tol=1e-15,
                        )
                        and math.isclose(
                            stability,
                            view.stability_score,
                            rel_tol=0.0,
                            abs_tol=1e-15,
                        ),
                        "Stage-20 feature SAM2 scores changed",
                    )
                    if scale == 1.0:
                        _require(
                            row["embedding_encoding"]
                            == "base64-float32-little-endian-v1"
                            and row["embedding_dimensions"] == "2048",
                            "Stage-20 embedding contract changed",
                        )
                        values = {
                            name: _finite(
                                row[name],
                                f"Stage-20 raw predictor {name}",
                            )
                            for name in CANONICAL_RAW_FEATURE_ORDER
                        }
                        try:
                            embedding = decode_canonical_embedding(
                                row["embedding_f32le_base64"]
                            )
                        except ValueError as exc:
                            raise PublicIOError(
                                "Stage-20 embedding cannot be decoded"
                            ) from exc
                        selected_ids.append(proposal_id)
                        selected_rows.append(
                            CanonicalRawFeature(
                                proposal_index=view.proposal_index,
                                scale=scale,
                                predicted_iou=predicted_iou,
                                stability_score=stability,
                                values=values,
                                embedding=embedding,
                            )
                        )
                _require(block_id is not None, "Stage-20 feature table is empty")
                _require(
                    tuple(block_scales) == tuple(CANONICAL_FEATURE_CROP_SCALES),
                    "Stage-20 final feature scale block changed",
                )
        completed = os.fstat(descriptor)
        after = path.lstat()
        _require(
            _file_signature(completed)
            == _file_signature(opened)
            == _file_signature(after),
            "Stage-20 feature table changed while scoring",
        )
    except (csv.Error, UnicodeError, ValueError) as exc:
        if isinstance(exc, PublicIOError):
            raise
        raise PublicIOError("Stage-20 feature table cannot be parsed") from exc
    finally:
        resources.close()
    actual_sha256 = digest.hexdigest()
    _require(
        actual_sha256 == expected_sha256,
        "Stage-20 feature table hash changed",
    )
    _require(
        observed == set(index.proposals)
        and len(selected_ids) == len(index.proposals)
        and len(selected_ids) == len(set(selected_ids)),
        "Stage-20 scale-1 feature closure changed",
    )
    return tuple(selected_ids), tuple(selected_rows), actual_sha256


def score_published_model(
    snapshot: Any,
    index: Any,
    preset_id: str,
    *,
    _predictor_loader: Callable[..., Any] | None = None,
    _probability_predictor: Callable[[Any, Any], Any] | None = None,
) -> PublishedModelContext:
    """Score one sealed canonical Full Stage-20 run with the r92 preset."""

    _require(preset_id == PRESET_ID, "published-model preset is unsupported")
    proposal_config_path = (
        snapshot.root / "stages/20_proposals_features/proposal_config.json"
    )
    config_payload, _identity = stable_file(
        proposal_config_path,
        max_bytes=8 * 1024 * 1024,
    )
    proposal_config = canonical_json_value(
        config_payload,
        "Stage-20 proposal configuration",
    )
    _require(
        isinstance(proposal_config, dict)
        and proposal_config.get("profile") == CANONICAL_GPU_PROFILE
        and proposal_config.get("final_feature_order_sha256")
        == CANONICAL_FEATURE_ORDER_SHA256
        and proposal_config.get("embedding_backbone")
        == "resnet50-imagenet1k-v2",
        "Project 1/CJ transfer assistance requires the canonical Full profile",
    )
    assets = _load_assets()
    proposal_ids, raw_rows, feature_source_sha256 = _scale_one_rows(
        snapshot,
        index,
    )
    finalized = [
        finalize_canonical_feature(row, assets.feature_state)
        for row in raw_rows
    ]
    if _predictor_loader is None or _probability_predictor is None:
        from compag_curation.canonical.serialization import (
            load_portable_predictor,
            predict_portable_probabilities,
        )

        _predictor_loader = load_portable_predictor
        _probability_predictor = predict_portable_probabilities
    predictor = _predictor_loader(
        assets.classifier_ubj,
        assets.imputer_statistics,
        device="cuda",
    )
    probabilities = _probability_predictor(predictor, finalized)
    _require(
        len(probabilities) == len(proposal_ids),
        "published-model predictor returned the wrong row count",
    )
    scores: dict[str, PublishedModelScore] = {}
    for proposal_id, raw_probability in zip(
        proposal_ids,
        probabilities,
        strict=True,
    ):
        probability = float(raw_probability)
        _require(
            math.isfinite(probability) and 0.0 <= probability <= 1.0,
            "published-model probability is outside [0,1]",
        )
        prediction = int(probability >= PRESET_THRESHOLD)
        uncertainty = abs(probability - PRESET_THRESHOLD)
        score = PublishedModelScore(probability, prediction, uncertainty)
        scores[proposal_id] = score
    _require(set(scores) == set(index.proposals), "published scores are incomplete")
    order = tuple(
        sorted(
            scores,
            key=lambda proposal_id: (
                scores[proposal_id].uncertainty,
                proposal_id,
            ),
        )
    )
    return _context(
        snapshot=snapshot,
        assets=assets,
        order=order,
        scores=scores,
        feature_source_sha256=feature_source_sha256,
    )


def load_published_model_context(
    snapshot: Any,
    index: Any,
    state_root: Path,
    preset_id: str,
) -> PublishedModelContext:
    """Load and verify a session-bound score cache without rescoring CUDA."""

    _require(preset_id == PRESET_ID, "published-model preset is unsupported")
    assets = _load_assets()
    path = state_root / "MODEL_ASSISTANCE.json"
    payload, info = stable_file(path, max_bytes=32 * 1024 * 1024)
    _require(
        stat.S_IMODE(info.st_mode) == 0o600
        and info.st_uid == os.getuid()
        and info.st_gid == os.getgid()
        and info.st_nlink == 1,
        "published-model assistance cache must be an owned mode-0600 file",
    )
    value = canonical_json_value(payload, "published-model assistance cache")
    fields = {
        "schema",
        "preset_id",
        "display_name",
        "assist_mode",
        "profile",
        "threshold_method",
        "threshold",
        "run_id",
        "review_request_sha256",
        "stage20_receipt_sha256",
        "feature_source_sha256",
        "resource_manifest_sha256",
        "classifier_sha256",
        "feature_state_sha256",
        "feature_order_sha256",
        "scores_sha256",
        "proposal_count",
        "navigation_order",
        "navigation_order_sha256",
        "scores",
        "device",
        "lineage_policy",
    }
    expected_feature_sha256, _feature_size = _stage20_feature_identity(snapshot)
    _require(
        isinstance(value, dict)
        and set(value) == fields
        and value.get("schema") == ASSISTANCE_FILE_SCHEMA
        and value.get("preset_id") == PRESET_ID
        and value.get("display_name") == PRESET_DISPLAY_NAME
        and value.get("assist_mode") == "HISTORICAL_R92_TRANSFER_ASSIST"
        and value.get("profile") == CANONICAL_GPU_PROFILE
        and value.get("threshold_method") == "FIXED_MANUSCRIPT_THRESHOLD"
        and value.get("threshold") == PRESET_THRESHOLD
        and value.get("run_id") == snapshot.run_id
        and value.get("review_request_sha256") == snapshot.review_request_sha256
        and value.get("stage20_receipt_sha256")
        == snapshot.stage20_receipt_sha256
        and value.get("feature_source_sha256") == expected_feature_sha256
        and value.get("resource_manifest_sha256") == assets.manifest_sha256
        and value.get("classifier_sha256") == assets.classifier_sha256
        and value.get("feature_state_sha256") == assets.feature_state_sha256
        and value.get("feature_order_sha256")
        == CANONICAL_FEATURE_ORDER_SHA256
        and value.get("device") == "cuda"
        and value.get("lineage_policy")
        == "TRANSFER_ASSIST_NOT_FRESH_CANONICAL_EQUIVALENCE"
        and value.get("proposal_count") == len(index.proposals),
        "published-model assistance cache binding changed",
    )
    raw_scores = value.get("scores")
    _require(isinstance(raw_scores, list), "assistance score rows are invalid")
    scores: dict[str, PublishedModelScore] = {}
    for row in raw_scores:
        _require(
            isinstance(row, dict)
            and set(row) == {"proposal_id", "probability_hex", "prediction"}
            and isinstance(row.get("proposal_id"), str)
            and isinstance(row.get("probability_hex"), str)
            and type(row.get("prediction")) is int
            and row["proposal_id"] not in scores,
            "assistance score row is malformed or duplicated",
        )
        try:
            probability = float.fromhex(str(row["probability_hex"]))
        except ValueError as exc:
            raise PublicIOError("assistance probability is invalid") from exc
        prediction = int(row["prediction"])
        _require(
            math.isfinite(probability)
            and 0.0 <= probability <= 1.0
            and prediction in {0, 1}
            and prediction == int(probability >= PRESET_THRESHOLD),
            "assistance probability/prediction contract changed",
        )
        scores[str(row["proposal_id"])] = PublishedModelScore(
            probability=probability,
            prediction=prediction,
            uncertainty=abs(probability - PRESET_THRESHOLD),
        )
    _require(
        set(scores) == set(index.proposals)
        and raw_scores == _score_rows(scores),
        "assistance score proposal closure or order changed",
    )
    scores_sha256 = _scores_sha256(
        feature_source_sha256=expected_feature_sha256,
        scores=scores,
    )
    raw_order = value.get("navigation_order")
    _require(
        isinstance(raw_order, list)
        and all(isinstance(item, str) for item in raw_order),
        "assistance navigation order is invalid",
    )
    order = tuple(raw_order)
    expected_order = tuple(
        sorted(
            scores,
            key=lambda proposal_id: (
                scores[proposal_id].uncertainty,
                proposal_id,
            ),
        )
    )
    _require(
        order == expected_order
        and value.get("navigation_order_sha256")
        == compact_json_sha256(list(order))
        and value.get("scores_sha256") == scores_sha256,
        "assistance score digest or navigation order changed",
    )
    context = _context(
        snapshot=snapshot,
        assets=assets,
        order=order,
        scores=scores,
        feature_source_sha256=expected_feature_sha256,
    )
    _require(
        dict(context.assistance_record) == value
        and context.assistance_file_sha256
        == hashlib.sha256(payload).hexdigest(),
        "assistance cache differs from its reconstructed context",
    )
    return context


__all__ = [
    "PRESET_DISPLAY_NAME",
    "PRESET_ID",
    "PRESET_THRESHOLD",
    "PublishedModelContext",
    "PublishedModelScore",
    "PublishedModelTransferAssets",
    "load_published_model_context",
    "load_published_model_transfer_assets",
    "score_published_model",
]
