"""Narrow Amendment 03 recovery helpers for the replacement CUDA Smoke.

The helpers in this module do not authorize a Run.  They validate and copy the
sealed historical CUDA probe, and they repair only the verified XGBoost 2.1.1
binary-classifier reload defect described by Protocol Amendment 03.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import xgboost as xgb


SEALED_PROBE_RELATIVE_PATH = "provenance/cuda_active_fit_probe.json"
ROOT_PROBE_DESTINATION = SEALED_PROBE_RELATIVE_PATH
NESTED_PROBE_DESTINATION = (
    "provenance/preflight_snapshot/cuda_active_fit_probe.json"
)
WITHDRAWN_RUNTIME_PROBE_SHA256 = (
    "41f4a3c35fd1d8308c791dab155c2a82b5e5ae20236f142645cfa71a652768d5"
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MODEL_METADATA = {
    "amendment_estimator_type": "XGBClassifier",
    "amendment_class_labels_json": "[0,1]",
    "amendment_positive_class": "1",
    "amendment_objective": "binary:logistic",
}
_FALLBACK_WARNING_TOKENS = (
    "fallback",
    "falling back",
    "mismatched devices",
    "not compiled",
    "cpu",
)


class Amendment03CompatibilityError(RuntimeError):
    """A fail-closed Amendment 03 integrity or compatibility failure."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise Amendment03CompatibilityError(f"Invalid {label} SHA-256: {text!r}")
    return text


def _safe_unique_zip_names(archive: zipfile.ZipFile) -> list[str]:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise Amendment03CompatibilityError("Preflight ZIP has duplicate members")
    for name in names:
        pure = PurePosixPath(name)
        if (
            not name
            or name.startswith("/")
            or "\\" in name
            or pure.is_absolute()
            or ".." in pure.parts
            or name.endswith("/")
        ):
            raise Amendment03CompatibilityError(
                f"Preflight ZIP has an unsafe member path: {name!r}"
            )
    bad_member = archive.testzip()
    if bad_member is not None:
        raise Amendment03CompatibilityError(
            f"Preflight ZIP CRC failure: {bad_member}"
        )
    return names


def _manifest_probe_row(data: bytes, manifest_name: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        required = {"relative_path", "size_bytes", "sha256"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("required columns are absent")
        rows = list(reader)
    except Exception as exc:
        raise Amendment03CompatibilityError(
            f"Malformed {manifest_name}: {type(exc).__name__}: {exc}"
        ) from exc
    paths = [str(row["relative_path"]) for row in rows]
    if len(paths) != len(set(paths)):
        raise Amendment03CompatibilityError(
            f"{manifest_name} has duplicate relative paths"
        )
    matches = [
        row for row in rows
        if str(row["relative_path"]) == SEALED_PROBE_RELATIVE_PATH
    ]
    if len(matches) != 1:
        raise Amendment03CompatibilityError(
            f"{manifest_name} does not cover the sealed CUDA probe exactly once"
        )
    row = matches[0]
    try:
        size_bytes = int(row["size_bytes"])
    except (TypeError, ValueError) as exc:
        raise Amendment03CompatibilityError(
            f"{manifest_name} has an invalid probe size"
        ) from exc
    return {
        "relative_path": SEALED_PROBE_RELATIVE_PATH,
        "size_bytes": size_bytes,
        "sha256": _require_sha256(
            row["sha256"], f"{manifest_name} probe member"
        ),
    }


def _validate_probe_payload(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise Amendment03CompatibilityError("Sealed CUDA probe is not a JSON object")
    build_info = payload.get("build_info")
    if not (
        payload.get("classification") == "VERIFIED"
        and payload.get("status") == "PASS"
        and payload.get("xgboost_version") == "2.1.1"
        and isinstance(build_info, Mapping)
        and build_info.get("USE_CUDA") is True
        and payload.get("requested_device") == "cuda"
        and payload.get("tree_method") == "hist"
        and payload.get("training_status") == "PASS"
        and payload.get("save_reload_predict_status") == "PASS"
        and payload.get("cpu_fallback_detected") is False
    ):
        raise Amendment03CompatibilityError(
            "Sealed CUDA probe does not record the required active CUDA PASS"
        )


@dataclass(frozen=True)
class SealedProbeContract:
    """Validated exact bytes and machine-readable binding evidence."""

    data: bytes
    evidence: dict[str, Any]


def validate_sealed_preflight_probe(
    preflight_run: Path,
    *,
    expected_bundle_sha256: str,
    package_verification_path: Path | None = None,
) -> SealedProbeContract:
    """Validate and read the sole authoritative probe from a pinned Review ZIP."""

    preflight_run = Path(preflight_run)
    review_bundle = Path(str(preflight_run) + "_review_bundle.zip")
    expected_bundle_sha256 = _require_sha256(
        expected_bundle_sha256, "accepted Preflight Review ZIP"
    )
    if not review_bundle.is_file() or review_bundle.is_symlink():
        raise Amendment03CompatibilityError(
            f"Accepted Preflight Review ZIP is absent or unsafe: {review_bundle}"
        )
    actual_bundle_sha256 = _sha256_file(review_bundle)
    if actual_bundle_sha256 != expected_bundle_sha256:
        raise Amendment03CompatibilityError(
            "Accepted Preflight Review ZIP differs from its pinned SHA-256"
        )

    verification_path = package_verification_path or preflight_run.parent / (
        preflight_run.name + "_package_verification.json"
    )
    try:
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Amendment03CompatibilityError(
            f"Accepted package verification is unreadable: {type(exc).__name__}: {exc}"
        ) from exc

    with zipfile.ZipFile(review_bundle) as archive:
        names = _safe_unique_zip_names(archive)
        prefix = f"{preflight_run.name}/"
        if any(not name.startswith(prefix) for name in names):
            raise Amendment03CompatibilityError(
                "Accepted Preflight Review ZIP has an unexpected root"
            )
        source_member = prefix + SEALED_PROBE_RELATIVE_PATH
        output_manifest_member = prefix + "OUTPUT_MANIFEST_FINAL.tsv"
        bundle_manifest_member = prefix + "BUNDLE_MANIFEST.tsv"
        required_members = {
            source_member, output_manifest_member, bundle_manifest_member,
        }
        if not required_members <= set(names):
            raise Amendment03CompatibilityError(
                "Accepted Preflight Review ZIP omits the probe or canonical manifests"
            )
        probe_bytes = archive.read(source_member)
        probe_sha256 = _sha256_bytes(probe_bytes)
        output_row = _manifest_probe_row(
            archive.read(output_manifest_member), "OUTPUT_MANIFEST_FINAL.tsv"
        )
        bundle_row = _manifest_probe_row(
            archive.read(bundle_manifest_member), "BUNDLE_MANIFEST.tsv"
        )
        uncompressed_size = sum(info.file_size for info in archive.infolist())

    for manifest_name, row in (
        ("OUTPUT_MANIFEST_FINAL.tsv", output_row),
        ("BUNDLE_MANIFEST.tsv", bundle_row),
    ):
        if row["size_bytes"] != len(probe_bytes) or row["sha256"] != probe_sha256:
            raise Amendment03CompatibilityError(
                f"Sealed CUDA probe disagrees with {manifest_name}"
            )
    if output_row != bundle_row:
        raise Amendment03CompatibilityError(
            "Canonical manifests disagree about the sealed CUDA probe"
        )

    try:
        probe_payload = json.loads(probe_bytes)
    except Exception as exc:
        raise Amendment03CompatibilityError(
            f"Sealed CUDA probe JSON is malformed: {type(exc).__name__}: {exc}"
        ) from exc
    _validate_probe_payload(probe_payload)

    package_completeness = verification.get("package_completeness")
    verification_ok = bool(
        isinstance(verification, Mapping)
        and verification.get("bundle_path") == str(review_bundle)
        and verification.get("bundle_sha256") == actual_bundle_sha256
        and int(verification.get("bundle_size_bytes", -1))
        == review_bundle.stat().st_size
        and verification.get("crc_status") == "PASS"
        and verification.get("independent_reopen_member_verification") == "PASS"
        and int(verification.get("member_count", -1)) == len(names)
        and int(verification.get("uncompressed_size_bytes", -1))
        == uncompressed_size
        and not verification.get("verification_failures")
        and isinstance(package_completeness, Mapping)
        and package_completeness.get("required_files_status") == "PASS"
        and package_completeness.get("status_identity_match") == "PASS"
    )
    if not verification_ok:
        raise Amendment03CompatibilityError(
            "Sealed CUDA probe disagrees with accepted package-verification evidence"
        )

    evidence = {
        "classification": "VERIFIED",
        "status": "PASS",
        "source_zip_path": str(review_bundle),
        "source_zip_sha256": actual_bundle_sha256,
        "source_member": SEALED_PROBE_RELATIVE_PATH,
        "source_archive_member": source_member,
        "member_size_bytes": len(probe_bytes),
        "member_sha256": probe_sha256,
        "manifest_rows": {
            "OUTPUT_MANIFEST_FINAL.tsv": output_row,
            "BUNDLE_MANIFEST.tsv": bundle_row,
        },
        "manifest_match": "PASS",
        "package_verification_path": str(verification_path),
        "package_verification_match": "PASS",
        "probe_active_cuda_validation": "PASS",
        "runtime_probe_used_as_source": False,
        "withdrawn_runtime_probe_sha256_imposed": False,
    }
    return SealedProbeContract(data=probe_bytes, evidence=evidence)


def validate_smoke_probe_copies(
    run_dir: Path,
    contract: SealedProbeContract,
) -> dict[str, Any]:
    """Require both Smoke destinations to contain the exact sealed bytes."""

    run_dir = Path(run_dir)
    expected_sha256 = str(contract.evidence["member_sha256"])
    destinations: dict[str, dict[str, Any]] = {}
    for relative in (ROOT_PROBE_DESTINATION, NESTED_PROBE_DESTINATION):
        path = run_dir / relative
        if not path.is_file() or path.is_symlink():
            raise Amendment03CompatibilityError(
                f"Required sealed probe copy is absent or unsafe: {relative}"
            )
        data = path.read_bytes()
        digest = _sha256_bytes(data)
        if data != contract.data or digest != expected_sha256:
            raise Amendment03CompatibilityError(
                f"Required sealed probe copy differs from source: {relative}"
            )
        destinations[relative] = {
            "size_bytes": len(data),
            "sha256": digest,
            "byte_equal_to_sealed_member": True,
        }
    return {
        **contract.evidence,
        "copy_status": "PASS",
        "destinations": destinations,
        "root_nested_byte_equal": True,
    }


def copy_sealed_probe_to_smoke(
    run_dir: Path,
    contract: SealedProbeContract,
) -> dict[str, Any]:
    """Create both required probe copies without overwrite or runtime fallback."""

    run_dir = Path(run_dir)
    paths = [
        run_dir / ROOT_PROBE_DESTINATION,
        run_dir / NESTED_PROBE_DESTINATION,
    ]
    if any(path.exists() or path.is_symlink() for path in paths):
        raise Amendment03CompatibilityError(
            "Refusing to overwrite an existing Smoke CUDA-probe destination"
        )
    created: list[Path] = []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(contract.data)
                handle.flush()
            created.append(path)
        return validate_smoke_probe_copies(run_dir, contract)
    except Exception:
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def _learner_configuration(booster: xgb.Booster) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        configuration = json.loads(booster.save_config())
        learner = configuration["learner"]
        parameters = learner["learner_model_param"]
    except Exception as exc:
        raise Amendment03CompatibilityError(
            f"Loaded Booster configuration is malformed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(learner, Mapping) or not isinstance(parameters, Mapping):
        raise Amendment03CompatibilityError("Loaded Booster configuration is malformed")
    return dict(learner), dict(parameters)


def _integer_config_value(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Amendment03CompatibilityError(f"Invalid Booster {name}: {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise Amendment03CompatibilityError(
            f"Invalid Booster {name}: {value!r}"
        ) from exc
    if parsed < 0:
        raise Amendment03CompatibilityError(f"Invalid Booster {name}: {parsed}")
    return parsed


def _verified_binary_model_contract(
    booster: xgb.Booster,
    *,
    expected_feature_count: int,
    expected_feature_list_sha256: str,
) -> dict[str, Any]:
    expected_feature_list_sha256 = _require_sha256(
        expected_feature_list_sha256, "feature-list"
    )
    learner, parameters = _learner_configuration(booster)
    try:
        objective = str(learner["objective"]["name"])
    except Exception as exc:
        raise Amendment03CompatibilityError(
            "Loaded Booster objective is missing or malformed"
        ) from exc
    if objective != "binary:logistic":
        raise Amendment03CompatibilityError(
            f"Expected objective binary:logistic, found {objective!r}"
        )
    native_num_class = _integer_config_value(parameters.get("num_class"), "num_class")
    derived_num_classes = 2 if native_num_class < 2 else native_num_class
    if derived_num_classes != 2:
        raise Amendment03CompatibilityError(
            f"Binary model has a conflicting native class count: {native_num_class}"
        )
    configured_features = _integer_config_value(
        parameters.get("num_feature"), "num_feature"
    )
    if (
        int(expected_feature_count) <= 0
        or configured_features != int(expected_feature_count)
        or booster.num_features() != int(expected_feature_count)
    ):
        raise Amendment03CompatibilityError(
            "Loaded Booster feature count differs from the locked Variant contract"
        )
    attributes = booster.attributes()
    if any(attributes.get(key) != value for key, value in _MODEL_METADATA.items()):
        raise Amendment03CompatibilityError(
            "Loaded Booster is missing valid Amendment 03 binary-class metadata"
        )
    try:
        class_labels = json.loads(attributes["amendment_class_labels_json"])
        positive_class = int(attributes["amendment_positive_class"])
    except Exception as exc:
        raise Amendment03CompatibilityError(
            "Loaded Booster class metadata is malformed"
        ) from exc
    if class_labels != [0, 1] or positive_class != 1:
        raise Amendment03CompatibilityError(
            "Loaded Booster class metadata differs from locked [0,1]/positive-1 semantics"
        )
    if attributes.get("feature_list_sha256") != expected_feature_list_sha256:
        raise Amendment03CompatibilityError(
            "Loaded Booster feature-list hash differs from the locked Variant contract"
        )
    return {
        "objective": objective,
        "native_num_class": native_num_class,
        "derived_num_classes": derived_num_classes,
        "feature_count": configured_features,
        "feature_list_sha256": expected_feature_list_sha256,
        "project_model_metadata": {
            **_MODEL_METADATA,
            "feature_list_sha256": expected_feature_list_sha256,
        },
    }


def embed_binary_classifier_metadata(
    classifier: xgb.XGBClassifier,
    *,
    expected_feature_count: int,
    feature_list_sha256: str,
) -> dict[str, str]:
    """Embed the locked, non-predictive binary semantics before ``save_model``."""

    booster = classifier.get_booster()
    learner, parameters = _learner_configuration(booster)
    try:
        objective = str(learner["objective"]["name"])
    except Exception as exc:
        raise Amendment03CompatibilityError("Fitted Booster objective is missing") from exc
    native_num_class = _integer_config_value(parameters.get("num_class"), "num_class")
    derived_num_classes = 2 if native_num_class < 2 else native_num_class
    try:
        fitted_labels = np.asarray(classifier.classes_).tolist()
    except Exception as exc:
        raise Amendment03CompatibilityError(
            "Fitted classifier does not expose verified class labels"
        ) from exc
    if objective != "binary:logistic" or derived_num_classes != 2:
        raise Amendment03CompatibilityError(
            "Only a verified binary:logistic two-class model can be marked"
        )
    if fitted_labels != [0, 1]:
        raise Amendment03CompatibilityError(
            "Fitted classifier class labels differ from locked [0,1] semantics"
        )
    attributes = {
        **_MODEL_METADATA,
        "feature_list_sha256": _require_sha256(
            feature_list_sha256, "feature-list"
        ),
    }
    booster.set_attr(**attributes)
    _verified_binary_model_contract(
        booster,
        expected_feature_count=expected_feature_count,
        expected_feature_list_sha256=feature_list_sha256,
    )
    return attributes


def repair_loaded_binary_classifier(
    classifier: xgb.XGBClassifier,
    *,
    expected_feature_count: int,
    expected_feature_list_sha256: str,
) -> dict[str, Any]:
    """Repair only an absent, config-verified binary ``n_classes_`` attribute."""

    contract = _verified_binary_model_contract(
        classifier.get_booster(),
        expected_feature_count=expected_feature_count,
        expected_feature_list_sha256=expected_feature_list_sha256,
    )
    had_n_classes = hasattr(classifier, "n_classes_")
    previous_n_classes = getattr(classifier, "n_classes_", None)
    if had_n_classes:
        try:
            loaded_num_classes = int(previous_n_classes)
        except (TypeError, ValueError) as exc:
            raise Amendment03CompatibilityError(
                "Loaded classifier has a malformed n_classes_ attribute"
            ) from exc
        if loaded_num_classes != contract["derived_num_classes"]:
            raise Amendment03CompatibilityError(
                "Loaded classifier n_classes_ conflicts with verified Booster metadata"
            )
        repair_applied = False
    else:
        classifier.n_classes_ = contract["derived_num_classes"]
        repair_applied = True
    if classifier.n_classes_ != 2:
        raise Amendment03CompatibilityError(
            "Compatibility loader did not establish exactly two classes"
        )
    return {
        **contract,
        "compatibility_repair_applied": repair_applied,
        "n_classes_present_before_repair": had_n_classes,
        "n_classes_before_repair": previous_n_classes,
        "n_classes_after_repair": int(classifier.n_classes_),
        "repair_scope": "MISSING_N_CLASSES_ONLY",
    }


@dataclass(frozen=True)
class LoadedBinaryClassifier:
    classifier: xgb.XGBClassifier
    evidence: dict[str, Any]


def load_binary_classifier_compat(
    model_path: Path,
    *,
    expected_feature_count: int,
    expected_feature_list_sha256: str,
    expected_model_sha256: str | None = None,
    device: str = "cuda",
) -> LoadedBinaryClassifier:
    """Public-load an XGBClassifier, validate metadata, and apply the narrow repair."""

    model_path = Path(model_path)
    if not model_path.is_file() or model_path.is_symlink():
        raise Amendment03CompatibilityError(
            f"Model is absent or unsafe: {model_path}"
        )
    before_sha256 = _sha256_file(model_path)
    if expected_model_sha256 is not None and before_sha256 != _require_sha256(
        expected_model_sha256, "model"
    ):
        raise Amendment03CompatibilityError("Model differs from its pinned SHA-256")
    classifier = xgb.XGBClassifier()
    classifier.load_model(model_path)
    evidence = repair_loaded_binary_classifier(
        classifier,
        expected_feature_count=expected_feature_count,
        expected_feature_list_sha256=expected_feature_list_sha256,
    )
    classifier.set_params(device=device)
    classifier.get_booster().set_param({"device": device})
    after_sha256 = _sha256_file(model_path)
    if after_sha256 != before_sha256:
        raise Amendment03CompatibilityError(
            "Model bytes changed during compatibility loading"
        )
    evidence.update({
        "loader": "XGBClassifier.load_model",
        "requested_device": device,
        "model_path": str(model_path),
        "model_sha256_before": before_sha256,
        "model_sha256_after": after_sha256,
        "model_bytes_unchanged": True,
    })
    return LoadedBinaryClassifier(classifier=classifier, evidence=evidence)


def classifier_equivalent_iteration_range(booster: xgb.Booster) -> tuple[int, int]:
    """Return the explicit range used by the XGBoost classifier wrapper."""

    best = booster.attr("best_iteration")
    if best is None:
        return (0, 0)
    try:
        best_iteration = int(best)
    except (TypeError, ValueError) as exc:
        raise Amendment03CompatibilityError(
            f"Malformed best_iteration Booster attribute: {best!r}"
        ) from exc
    if best_iteration < 0 or best_iteration >= booster.num_boosted_rounds():
        raise Amendment03CompatibilityError(
            "best_iteration is outside the saved Booster tree range"
        )
    return (0, best_iteration + 1)


def _device_matrix(matrix: np.ndarray, device: str) -> Any:
    if device == "cuda":
        try:
            import cupy as cp
        except Exception as exc:
            raise Amendment03CompatibilityError(
                f"CuPy is required for active CUDA parity: {exc}"
            ) from exc
        return cp.asarray(matrix)
    if device == "cpu":
        return matrix
    raise Amendment03CompatibilityError(f"Unsupported parity device: {device!r}")


def _host_float32(values: Any, *, expected_rows: int, label: str) -> np.ndarray:
    try:
        import cupy as cp

        host = cp.asnumpy(values) if isinstance(values, cp.ndarray) else np.asarray(values)
    except ImportError:
        host = np.asarray(values)
    if (
        host.dtype != np.float32
        or host.ndim != 1
        or len(host) != expected_rows
        or not np.isfinite(host).all()
    ):
        raise Amendment03CompatibilityError(
            f"{label} did not return a finite float32 positive-probability vector"
        )
    return np.ascontiguousarray(host)


def _reject_fallback_warnings(caught: list[warnings.WarningMessage], label: str) -> list[str]:
    text = [str(item.message) for item in caught]
    if any(
        token in message.lower()
        for message in text
        for token in _FALLBACK_WARNING_TOKENS
    ):
        raise Amendment03CompatibilityError(
            f"{label} emitted a CPU/device-fallback warning"
        )
    return text


def _wrapper_positive_probabilities(
    classifier: xgb.XGBClassifier,
    matrix: Any,
    *,
    iteration_range: tuple[int, int],
    expected_rows: int,
    label: str,
) -> tuple[np.ndarray, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probabilities = classifier.predict_proba(
            matrix, iteration_range=iteration_range,
        )[:, 1]
    return (
        _host_float32(probabilities, expected_rows=expected_rows, label=label),
        _reject_fallback_warnings(caught, label),
    )


def _booster_positive_probabilities(
    booster: xgb.Booster,
    matrix: Any,
    *,
    iteration_range: tuple[int, int],
    missing: float | None,
    expected_rows: int,
    label: str,
) -> tuple[np.ndarray, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probabilities = booster.inplace_predict(
            matrix,
            iteration_range=iteration_range,
            predict_type="value",
            missing=missing,
            validate_features=True,
        )
    return (
        _host_float32(probabilities, expected_rows=expected_rows, label=label),
        _reject_fallback_warnings(caught, label),
    )


@dataclass(frozen=True)
class FourWayParityResult:
    fitted_classifier_positive: np.ndarray
    fitted_booster_positive: np.ndarray
    reloaded_booster_positive: np.ndarray
    compatibility_classifier_positive: np.ndarray
    compatibility_classifier: xgb.XGBClassifier
    reloaded_booster: xgb.Booster
    evidence: dict[str, Any]


def four_way_binary_reload_parity(
    fitted_classifier: xgb.XGBClassifier,
    model_path: Path,
    validation_matrix: np.ndarray,
    *,
    expected_feature_count: int,
    expected_feature_list_sha256: str,
    device: str = "cuda",
) -> FourWayParityResult:
    """Require bit-exact wrapper/Booster/save/reload positive probabilities."""

    model_path = Path(model_path)
    before_sha256 = _sha256_file(model_path)
    matrix = np.ascontiguousarray(validation_matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != int(expected_feature_count):
        raise Amendment03CompatibilityError(
            "Parity matrix differs from the locked feature-count contract"
        )
    device_data = _device_matrix(matrix, device)
    fitted_booster = fitted_classifier.get_booster()
    fitted_classifier.set_params(device=device)
    fitted_booster.set_param({"device": device})
    _verified_binary_model_contract(
        fitted_booster,
        expected_feature_count=expected_feature_count,
        expected_feature_list_sha256=expected_feature_list_sha256,
    )
    iteration_range = classifier_equivalent_iteration_range(fitted_booster)
    missing = getattr(fitted_classifier, "missing", None)

    fitted_wrapper, fitted_wrapper_warnings = _wrapper_positive_probabilities(
        fitted_classifier,
        device_data,
        iteration_range=iteration_range,
        expected_rows=len(matrix),
        label="fitted XGBClassifier",
    )
    fitted_raw, fitted_raw_warnings = _booster_positive_probabilities(
        fitted_booster,
        device_data,
        iteration_range=iteration_range,
        missing=missing,
        expected_rows=len(matrix),
        label="fitted raw Booster",
    )

    reloaded_booster = xgb.Booster()
    reloaded_booster.load_model(model_path)
    reloaded_booster.set_param({"device": device})
    _verified_binary_model_contract(
        reloaded_booster,
        expected_feature_count=expected_feature_count,
        expected_feature_list_sha256=expected_feature_list_sha256,
    )
    if classifier_equivalent_iteration_range(reloaded_booster) != iteration_range:
        raise Amendment03CompatibilityError(
            "Reloaded Booster iteration range differs from fitted Booster"
        )
    reloaded_raw, reloaded_raw_warnings = _booster_positive_probabilities(
        reloaded_booster,
        device_data,
        iteration_range=iteration_range,
        missing=missing,
        expected_rows=len(matrix),
        label="reloaded raw Booster",
    )

    loaded = load_binary_classifier_compat(
        model_path,
        expected_feature_count=expected_feature_count,
        expected_feature_list_sha256=expected_feature_list_sha256,
        expected_model_sha256=before_sha256,
        device=device,
    )
    compatibility_wrapper, compatibility_warnings = _wrapper_positive_probabilities(
        loaded.classifier,
        device_data,
        iteration_range=iteration_range,
        expected_rows=len(matrix),
        label="compatibility-loaded XGBClassifier",
    )

    vectors = {
        "fitted_classifier": fitted_wrapper,
        "fitted_booster": fitted_raw,
        "reloaded_booster": reloaded_raw,
        "compatibility_classifier": compatibility_wrapper,
    }
    reference = fitted_wrapper
    maximum_differences = {
        name: float(np.max(np.abs(reference - values)))
        if len(reference) else 0.0
        for name, values in vectors.items()
    }
    bit_exact = {
        name: bool(np.array_equal(reference, values))
        for name, values in vectors.items()
    }
    if not all(bit_exact.values()) or any(
        not math.isfinite(value) or value != 0.0
        for value in maximum_differences.values()
    ):
        raise Amendment03CompatibilityError(
            "Four-way classifier/Booster reload probability parity failed"
        )
    after_sha256 = _sha256_file(model_path)
    if after_sha256 != before_sha256:
        raise Amendment03CompatibilityError(
            "Model bytes changed during four-way reload parity"
        )
    evidence = {
        "classification": "VERIFIED",
        "status": "PASS",
        "device": device,
        "input_dtype": str(matrix.dtype),
        "input_contiguous": bool(matrix.flags.c_contiguous),
        "input_shape": list(matrix.shape),
        "feature_list_sha256": expected_feature_list_sha256,
        "missing_value": missing,
        "iteration_range": list(iteration_range),
        "vector_sha256": {
            name: _sha256_bytes(values.tobytes(order="C"))
            for name, values in vectors.items()
        },
        "bit_exact_against_fitted_classifier": bit_exact,
        "maximum_absolute_difference": maximum_differences,
        "warnings": {
            "fitted_classifier": fitted_wrapper_warnings,
            "fitted_booster": fitted_raw_warnings,
            "reloaded_booster": reloaded_raw_warnings,
            "compatibility_classifier": compatibility_warnings,
        },
        "cpu_fallback_detected": False,
        "compatibility_loader": loaded.evidence,
        "model_sha256_before": before_sha256,
        "model_sha256_after": after_sha256,
        "model_bytes_unchanged": True,
    }
    return FourWayParityResult(
        fitted_classifier_positive=fitted_wrapper,
        fitted_booster_positive=fitted_raw,
        reloaded_booster_positive=reloaded_raw,
        compatibility_classifier_positive=compatibility_wrapper,
        compatibility_classifier=loaded.classifier,
        reloaded_booster=reloaded_booster,
        evidence=evidence,
    )


__all__ = [
    "Amendment03CompatibilityError",
    "FourWayParityResult",
    "LoadedBinaryClassifier",
    "NESTED_PROBE_DESTINATION",
    "ROOT_PROBE_DESTINATION",
    "SEALED_PROBE_RELATIVE_PATH",
    "SealedProbeContract",
    "WITHDRAWN_RUNTIME_PROBE_SHA256",
    "classifier_equivalent_iteration_range",
    "copy_sealed_probe_to_smoke",
    "embed_binary_classifier_metadata",
    "four_way_binary_reload_parity",
    "load_binary_classifier_compat",
    "repair_loaded_binary_classifier",
    "validate_sealed_preflight_probe",
    "validate_smoke_probe_copies",
]
