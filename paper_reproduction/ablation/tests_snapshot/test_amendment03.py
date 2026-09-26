from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb


STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "code"))

import amendment03_compat as compat  # noqa: E402
import amendment_core as core  # noqa: E402


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_probe_payload() -> dict:
    return {
        "classification": "VERIFIED",
        "status": "PASS",
        "xgboost_version": "2.1.1",
        "build_info": {"USE_CUDA": True},
        "requested_device": "cuda",
        "tree_method": "hist",
        "training_status": "PASS",
        "save_reload_predict_status": "PASS",
        "cpu_fallback_detected": False,
        "raw_probe_sha256": compat.WITHDRAWN_RUNTIME_PROBE_SHA256,
    }


def _make_sealed_preflight(
    tmp_path: Path,
    *,
    probe_bytes: bytes | None = None,
    include_probe: bool = True,
    output_probe_path: str = compat.SEALED_PROBE_RELATIVE_PATH,
    output_probe_sha256: str | None = None,
    bundle_probe_sha256: str | None = None,
) -> tuple[Path, str]:
    run = tmp_path / "accepted_preflight"
    run.mkdir()
    probe_bytes = probe_bytes or json.dumps(
        _valid_probe_payload(), sort_keys=True,
    ).encode()
    probe_sha256 = _sha256(probe_bytes)
    output_probe_sha256 = output_probe_sha256 or probe_sha256
    bundle_probe_sha256 = bundle_probe_sha256 or probe_sha256
    output_manifest = (
        "relative_path\tsize_bytes\tsha256\tartifact_role\t"
        "include_in_review_bundle\tinclude_in_models_bundle\n"
        f"{output_probe_path}\t{len(probe_bytes)}\t{output_probe_sha256}\t"
        "provenance\tTrue\tFalse\n"
    ).encode()
    bundle_manifest = (
        "relative_path\tsize_bytes\tsha256\n"
        f"{compat.SEALED_PROBE_RELATIVE_PATH}\t{len(probe_bytes)}\t"
        f"{bundle_probe_sha256}\n"
    ).encode()
    bundle = Path(str(run) + "_review_bundle.zip")
    prefix = f"{run.name}/"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if include_probe:
            archive.writestr(prefix + compat.SEALED_PROBE_RELATIVE_PATH, probe_bytes)
        archive.writestr(prefix + "OUTPUT_MANIFEST_FINAL.tsv", output_manifest)
        archive.writestr(prefix + "BUNDLE_MANIFEST.tsv", bundle_manifest)
    with zipfile.ZipFile(bundle) as archive:
        member_count = len(archive.namelist())
        uncompressed_size = sum(info.file_size for info in archive.infolist())
    verification = {
        "bundle_path": str(bundle),
        "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "bundle_size_bytes": bundle.stat().st_size,
        "crc_status": "PASS",
        "independent_reopen_member_verification": "PASS",
        "member_count": member_count,
        "uncompressed_size_bytes": uncompressed_size,
        "verification_failures": [],
        "package_completeness": {
            "required_files_status": "PASS",
            "status_identity_match": "PASS",
        },
    }
    (tmp_path / f"{run.name}_package_verification.json").write_text(
        json.dumps(verification)
    )
    return run, verification["bundle_sha256"]


def test_sealed_probe_is_validated_from_zip_and_both_manifests(tmp_path):
    run, bundle_sha256 = _make_sealed_preflight(tmp_path)
    contract = compat.validate_sealed_preflight_probe(
        run, expected_bundle_sha256=bundle_sha256,
    )
    assert contract.evidence["status"] == "PASS"
    assert contract.evidence["member_sha256"] == _sha256(contract.data)
    assert contract.evidence["manifest_match"] == "PASS"
    assert contract.evidence["package_verification_match"] == "PASS"


def test_withdrawn_runtime_hash_is_not_imposed_on_sealed_member(tmp_path):
    run, bundle_sha256 = _make_sealed_preflight(tmp_path)
    contract = compat.validate_sealed_preflight_probe(
        run, expected_bundle_sha256=bundle_sha256,
    )
    assert contract.evidence["member_sha256"] != (
        compat.WITHDRAWN_RUNTIME_PROBE_SHA256
    )
    assert contract.evidence["runtime_probe_used_as_source"] is False
    assert contract.evidence["withdrawn_runtime_probe_sha256_imposed"] is False


@pytest.mark.parametrize("failure", ["missing", "malformed", "unmanifested", "mismatch"])
def test_invalid_sealed_probe_contract_fails_closed(tmp_path, failure):
    kwargs = {}
    if failure == "missing":
        kwargs["include_probe"] = False
    elif failure == "malformed":
        kwargs["probe_bytes"] = b"{not-json"
    elif failure == "unmanifested":
        kwargs["output_probe_path"] = "provenance/not_the_probe.json"
    else:
        kwargs["bundle_probe_sha256"] = "0" * 64
    run, bundle_sha256 = _make_sealed_preflight(tmp_path, **kwargs)
    with pytest.raises(compat.Amendment03CompatibilityError):
        compat.validate_sealed_preflight_probe(
            run, expected_bundle_sha256=bundle_sha256,
        )


def test_package_verification_mismatch_fails_closed(tmp_path):
    run, bundle_sha256 = _make_sealed_preflight(tmp_path)
    verification_path = tmp_path / f"{run.name}_package_verification.json"
    verification = json.loads(verification_path.read_text())
    verification["crc_status"] = "FAIL"
    verification_path.write_text(json.dumps(verification))
    with pytest.raises(
        compat.Amendment03CompatibilityError, match="package-verification",
    ):
        compat.validate_sealed_preflight_probe(
            run, expected_bundle_sha256=bundle_sha256,
        )


def test_dual_probe_copies_are_exact_and_exist_before_fit(tmp_path):
    preflight, bundle_sha256 = _make_sealed_preflight(tmp_path)
    contract = compat.validate_sealed_preflight_probe(
        preflight, expected_bundle_sha256=bundle_sha256,
    )
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    binding = compat.copy_sealed_probe_to_smoke(smoke, contract)

    fit_called = False

    def fake_fit():
        nonlocal fit_called
        validated = compat.validate_smoke_probe_copies(smoke, contract)
        assert validated["copy_status"] == "PASS"
        fit_called = True

    fake_fit()
    assert fit_called
    assert binding["root_nested_byte_equal"] is True
    assert set(binding["destinations"]) == {
        compat.ROOT_PROBE_DESTINATION,
        compat.NESTED_PROBE_DESTINATION,
    }
    assert all(
        item["sha256"] == contract.evidence["member_sha256"]
        for item in binding["destinations"].values()
    )


def test_probe_copy_mismatch_and_overwrite_fail_closed(tmp_path):
    preflight, bundle_sha256 = _make_sealed_preflight(tmp_path)
    contract = compat.validate_sealed_preflight_probe(
        preflight, expected_bundle_sha256=bundle_sha256,
    )
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    compat.copy_sealed_probe_to_smoke(smoke, contract)
    with pytest.raises(compat.Amendment03CompatibilityError, match="overwrite"):
        compat.copy_sealed_probe_to_smoke(smoke, contract)
    (smoke / compat.ROOT_PROBE_DESTINATION).write_bytes(b"drift")
    with pytest.raises(compat.Amendment03CompatibilityError, match="differs"):
        compat.validate_smoke_probe_copies(smoke, contract)


def _feature_hash(feature_count: int) -> str:
    names = [f"feature_{index}" for index in range(feature_count)]
    return _sha256(("\n".join(names) + "\n").encode())


def _fitted_binary_model(
    tmp_path: Path,
    *,
    feature_count: int = 4,
) -> tuple[xgb.XGBClassifier, Path, np.ndarray, str]:
    rng = np.random.default_rng(42)
    matrix = np.ascontiguousarray(
        rng.normal(size=(80, feature_count)), dtype=np.float32,
    )
    labels = (matrix[:, 0] + 0.4 * matrix[:, 1] > 0).astype(np.int8)
    classifier = xgb.XGBClassifier(
        n_estimators=8,
        max_depth=3,
        learning_rate=0.2,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        early_stopping_rounds=2,
        random_state=42,
        n_jobs=1,
    )
    classifier.fit(matrix[:64], labels[:64], eval_set=[(matrix[64:], labels[64:])])
    feature_sha256 = _feature_hash(feature_count)
    compat.embed_binary_classifier_metadata(
        classifier,
        expected_feature_count=feature_count,
        feature_list_sha256=feature_sha256,
    )
    path = tmp_path / "binary.ubj"
    classifier.save_model(path)
    return classifier, path, matrix[64:], feature_sha256


def test_compat_loader_repairs_only_verified_missing_binary_attribute(tmp_path):
    _, model_path, _, feature_sha256 = _fitted_binary_model(tmp_path)
    direct = xgb.XGBClassifier()
    direct.load_model(model_path)
    assert not hasattr(direct, "n_classes_")

    before = _sha256(model_path.read_bytes())
    loaded = compat.load_binary_classifier_compat(
        model_path,
        expected_feature_count=4,
        expected_feature_list_sha256=feature_sha256,
        expected_model_sha256=before,
        device="cpu",
    )
    assert loaded.classifier.n_classes_ == 2
    assert loaded.evidence["compatibility_repair_applied"] is True
    assert loaded.evidence["native_num_class"] < 2
    assert loaded.evidence["repair_scope"] == "MISSING_N_CLASSES_ONLY"
    assert _sha256(model_path.read_bytes()) == before


def test_compat_repair_rejects_conflicting_existing_class_count(tmp_path):
    _, model_path, _, feature_sha256 = _fitted_binary_model(tmp_path)
    classifier = xgb.XGBClassifier()
    classifier.load_model(model_path)
    classifier.n_classes_ = 3
    with pytest.raises(compat.Amendment03CompatibilityError, match="conflicts"):
        compat.repair_loaded_binary_classifier(
            classifier,
            expected_feature_count=4,
            expected_feature_list_sha256=feature_sha256,
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("amendment_estimator_type", None),
        ("amendment_estimator_type", "XGBRegressor"),
        ("amendment_class_labels_json", "[1,0]"),
        ("amendment_class_labels_json", "not-json"),
        ("amendment_positive_class", "0"),
        ("amendment_objective", "binary:logitraw"),
        ("feature_list_sha256", "0" * 64),
    ],
)
def test_compat_loader_rejects_missing_or_conflicting_project_metadata(
    tmp_path, key, value,
):
    _, model_path, _, feature_sha256 = _fitted_binary_model(tmp_path)
    booster = xgb.Booster()
    booster.load_model(model_path)
    booster.set_attr(**{key: value})
    invalid_path = tmp_path / f"invalid-{key}.ubj"
    booster.save_model(invalid_path)
    with pytest.raises(compat.Amendment03CompatibilityError):
        compat.load_binary_classifier_compat(
            invalid_path,
            expected_feature_count=4,
            expected_feature_list_sha256=feature_sha256,
            device="cpu",
        )


def test_compat_loader_rejects_wrong_objective(tmp_path):
    matrix = np.arange(40, dtype=np.float32).reshape(10, 4)
    regressor = xgb.XGBRegressor(
        n_estimators=2, objective="reg:squarederror", tree_method="hist", device="cpu",
    )
    regressor.fit(matrix, np.arange(10, dtype=np.float32))
    booster = regressor.get_booster()
    booster.set_attr(
        amendment_estimator_type="XGBClassifier",
        amendment_class_labels_json="[0,1]",
        amendment_positive_class="1",
        amendment_objective="binary:logistic",
        feature_list_sha256=_feature_hash(4),
        scikit_learn=None,
    )
    path = tmp_path / "regression.ubj"
    booster.save_model(path)
    with pytest.raises(compat.Amendment03CompatibilityError, match="objective"):
        compat.load_binary_classifier_compat(
            path,
            expected_feature_count=4,
            expected_feature_list_sha256=_feature_hash(4),
            device="cpu",
        )


def test_compat_loader_rejects_feature_count_and_hash_mismatch(tmp_path):
    _, model_path, _, feature_sha256 = _fitted_binary_model(tmp_path)
    with pytest.raises(compat.Amendment03CompatibilityError, match="feature count"):
        compat.load_binary_classifier_compat(
            model_path,
            expected_feature_count=3,
            expected_feature_list_sha256=feature_sha256,
            device="cpu",
        )
    with pytest.raises(compat.Amendment03CompatibilityError, match="feature-list hash"):
        compat.load_binary_classifier_compat(
            model_path,
            expected_feature_count=4,
            expected_feature_list_sha256="0" * 64,
            device="cpu",
        )


def test_four_way_binary_reload_parity_is_bit_exact_and_model_immutable(tmp_path):
    classifier, model_path, validation, feature_sha256 = _fitted_binary_model(tmp_path)
    before = _sha256(model_path.read_bytes())
    result = compat.four_way_binary_reload_parity(
        classifier,
        model_path,
        validation,
        expected_feature_count=4,
        expected_feature_list_sha256=feature_sha256,
        device="cpu",
    )
    vectors = [
        result.fitted_classifier_positive,
        result.fitted_booster_positive,
        result.reloaded_booster_positive,
        result.compatibility_classifier_positive,
    ]
    assert all(vector.dtype == np.float32 for vector in vectors)
    assert all(np.array_equal(vectors[0], vector) for vector in vectors[1:])
    assert result.evidence["status"] == "PASS"
    assert result.evidence["iteration_range"] == [
        0, int(classifier.get_booster().attr("best_iteration")) + 1,
    ]
    assert set(result.evidence["maximum_absolute_difference"].values()) == {0.0}
    assert result.evidence["cpu_fallback_detected"] is False
    assert _sha256(model_path.read_bytes()) == before


def test_amendment_core_revalidates_corrigendum_model_evidence(tmp_path):
    classifier, model_path, validation, feature_sha256 = _fitted_binary_model(tmp_path)
    result = compat.four_way_binary_reload_parity(
        classifier,
        model_path,
        validation,
        expected_feature_count=4,
        expected_feature_list_sha256=feature_sha256,
        device="cpu",
    )
    parity = json.loads(json.dumps(result.evidence))
    parity["device"] = "cuda"
    parity["compatibility_loader"]["requested_device"] = "cuda"
    maximum = parity["maximum_absolute_difference"]
    bit_exact = parity["bit_exact_against_fitted_classifier"]
    mandatory = {
        "amendment_estimator_type": "XGBClassifier",
        "amendment_class_labels_json": "[0,1]",
        "amendment_positive_class": "1",
        "amendment_objective": "binary:logistic",
        "feature_list_sha256": feature_sha256,
    }
    detail = {
        "feature_count": 4,
        "feature_list_sha256": feature_sha256,
        "model_sha256": _sha256(model_path.read_bytes()),
        "amendment03_model_recovery": {
            "classification": "VERIFIED",
            "status": "PASS",
            "authorization": "AMENDMENT_03_CORRIGENDUM_ONE",
            "compatibility_loader": (
                "PROJECT_OWNED_XGB211_BINARY_CLASSIFIER_LOADER"
            ),
            "four_way_parity_required": True,
        },
        "four_way_reload_parity": parity,
        "raw_booster_parity": {
            "status": "PASS",
            "iteration_range": parity["iteration_range"],
            "fitted_booster_bit_exact": bit_exact["fitted_booster"],
            "reloaded_booster_bit_exact": bit_exact["reloaded_booster"],
            "fitted_booster_max_probability_difference": maximum[
                "fitted_booster"
            ],
            "reloaded_booster_max_probability_difference": maximum[
                "reloaded_booster"
            ],
            "model_bytes_unchanged": True,
        },
        "compatibility_classifier_parity": {
            "status": "PASS",
            "iteration_range": parity["iteration_range"],
            "bit_exact": bit_exact["compatibility_classifier"],
            "max_probability_difference": maximum["compatibility_classifier"],
            "repair_applied": True,
        },
    }
    booster = xgb.Booster()
    booster.load_model(model_path)
    assert core._amendment03_model_detail_failures(
        detail=detail,
        booster=booster,
        expected_attributes=mandatory,
    ) == []

    detail["raw_booster_parity"][
        "reloaded_booster_max_probability_difference"
    ] = 1e-7
    assert "amendment03_raw_booster_parity" in (
        core._amendment03_model_detail_failures(
            detail=detail,
            booster=booster,
            expected_attributes=mandatory,
        )
    )


@pytest.mark.skipif(
    os.environ.get("AMENDMENT03_ACTIVE_CUDA_PROBE") != "1",
    reason="Run only as the explicit post-suite Amendment 03 CUDA gate",
)
def test_amendment03_active_cuda_production_helper(tmp_path):
    assert xgb.__version__ == "2.1.1"
    assert xgb.build_info().get("USE_CUDA") is True

    rng = np.random.default_rng(42)
    matrix = np.ascontiguousarray(
        rng.normal(size=(512, 12)), dtype=np.float32,
    )
    labels = (matrix[:, 0] + 0.4 * matrix[:, 1] > 0).astype(np.int8)
    classifier = xgb.XGBClassifier(
        n_estimators=8,
        max_depth=3,
        learning_rate=0.2,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device="cuda",
        random_state=42,
        n_jobs=8,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier.fit(matrix[:384], labels[:384])
    fit_warnings = [str(item.message) for item in caught]
    forbidden = (
        "fallback", "falling back", "mismatched devices", "not compiled", "cpu",
    )
    assert not any(
        token in message.lower()
        for message in fit_warnings
        for token in forbidden
    )
    configuration = json.loads(classifier.get_booster().save_config())
    assert configuration["learner"]["generic_param"]["device"].startswith("cuda")

    feature_sha256 = _feature_hash(12)
    compat.embed_binary_classifier_metadata(
        classifier,
        expected_feature_count=12,
        feature_list_sha256=feature_sha256,
    )
    model_path = tmp_path / "active-cuda.ubj"
    classifier.save_model(model_path)
    before = _sha256(model_path.read_bytes())
    result = compat.four_way_binary_reload_parity(
        classifier,
        model_path,
        matrix[384:],
        expected_feature_count=12,
        expected_feature_list_sha256=feature_sha256,
        device="cuda",
    )
    assert result.evidence["status"] == "PASS"
    assert result.evidence["cpu_fallback_detected"] is False
    assert all(result.evidence["bit_exact_against_fitted_classifier"].values())
    assert set(result.evidence["maximum_absolute_difference"].values()) == {0.0}
    assert _sha256(model_path.read_bytes()) == before
    print(json.dumps({
        "classification": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
        "status": "PASS",
        "active_cuda_fit": "PASS",
        "cpu_fallback_detected": False,
        "fit_warnings": fit_warnings,
        "model_sha256": before,
        "four_way_reload_parity": result.evidence,
        "xgboost_version": xgb.__version__,
        "xgboost_build_info": xgb.build_info(),
    }, sort_keys=True))
