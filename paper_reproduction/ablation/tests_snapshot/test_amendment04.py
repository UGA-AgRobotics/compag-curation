from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import struct
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn
import xgboost as xgb


STUDY = Path(__file__).resolve().parents[1]
CODE = STUDY / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import amendment04_recovery as recovery  # noqa: E402
import amendment03_compat as compat  # noqa: E402
import amendment_core as core  # noqa: E402
import run_study  # noqa: E402


FAILED_A03_RUN = (
    STUDY
    / "results/run_20260813_125445_373539_bc48422a_amendment03_cuda_smoke"
)
FAILED_A03_CSV = FAILED_A03_RUN / "tables/SMOKE_REPORT.csv"
FAILED_A03_JSON = FAILED_A03_RUN / "metrics/SMOKE_REPORT.json"
ACCEPTED_PREFLIGHT = (
    STUDY
    / "results/run_20260813_050405_9970c566_amendment02_preflight"
)
METRIC_NAMES = ("average_precision", "roc_auc", "precision", "recall", "f1")
EXPECTED_VARIANTS = (
    "full_new_reference",
    "manual_only",
    "deep_only",
    "no_color",
    "no_shape",
    "no_texture",
    "no_embed_sim",
    "no_review_aware_training_weights",
    "no_safe_smote",
    "no_deep_pca_features",
)


def _float64_bits(value: float) -> bytes:
    return struct.pack(">d", float(value))


def _ulp_distance(left: float, right: float) -> int:
    left_bits = struct.unpack(">Q", _float64_bits(left))[0]
    right_bits = struct.unpack(">Q", _float64_bits(right))[0]
    return abs(left_bits - right_bits)


def _failed_metrics() -> dict:
    return json.loads(FAILED_A03_JSON.read_text())


def _concurrent_guard_worker(results, run_name, barrier, queue):
    try:
        barrier.wait(timeout=10)
        evidence = recovery.create_authorized_run_directory(
            Path(results) / run_name,
            Path(results),
        )
        queue.put(("PASS", evidence["authorization"]))
    except Exception as exc:
        queue.put(("ERROR", type(exc).__name__, str(exc)))


def _scientific_metric_mismatches(table: pd.DataFrame) -> list[dict]:
    details = _failed_metrics()["details"]
    mismatches = []
    passing = table.loc[table.status.astype(str).str.upper() == "PASS"]
    for row in passing.itertuples(index=False):
        expected = details[str(row.variant)]["metrics_development_only"]
        for metric in METRIC_NAMES:
            observed_value = float(getattr(row, metric))
            expected_value = float(expected[metric])
            if observed_value != expected_value:
                mismatches.append({
                    "variant": str(row.variant),
                    "metric": metric,
                    "observed_value": observed_value,
                    "expected_value": expected_value,
                    "abs_difference": abs(observed_value - expected_value),
                    "ulp_distance": _ulp_distance(observed_value, expected_value),
                })
    return mismatches


def _synthetic_semantic_fixture(tmp_path: Path, monkeypatch):
    try:
        from test_ablation import _valid_smoke_semantic_fixture
    except ModuleNotFoundError:
        tests_path = STUDY / "tests"
        if str(tests_path) not in sys.path:
            sys.path.insert(0, str(tests_path))
        from test_ablation import _valid_smoke_semantic_fixture

    return _valid_smoke_semantic_fixture(tmp_path, monkeypatch)


def test_failed_amendment03_csv_reproduces_exact_22_to_0_diagnosis():
    default_mismatches = _scientific_metric_mismatches(pd.read_csv(FAILED_A03_CSV))
    roundtrip_mismatches = _scientific_metric_mismatches(
        pd.read_csv(
            FAILED_A03_CSV,
            engine="c",
            float_precision="round_trip",
        )
    )
    diagnostic = recovery.csv_roundtrip_diagnostic(
        FAILED_A03_CSV,
        FAILED_A03_JSON,
    )

    assert len(default_mismatches) == 22
    assert max(row["abs_difference"] for row in default_mismatches) == (
        1.1102230246251565e-16
    )
    assert {row["ulp_distance"] for row in default_mismatches} == {1}
    assert roundtrip_mismatches == []
    assert diagnostic["status"] == "PASS"
    assert tuple(diagnostic["metric_names"]) == METRIC_NAMES
    assert diagnostic["default_read"]["mismatch_count"] == 22
    assert diagnostic["default_read"]["max_abs_difference"] == (
        1.1102230246251565e-16
    )
    assert diagnostic["default_read"]["max_ulp_distance"] == 1
    assert diagnostic["round_trip_read"]["mismatch_count"] == 0
    mismatches = diagnostic["mismatches"]
    assert len(mismatches) == 22
    assert all(row["ulp_distance"] == 1 for row in mismatches)
    assert all(row["metric"] in METRIC_NAMES for row in mismatches)
    assert {row["variant"] for row in mismatches} <= set(EXPECTED_VARIANTS)


def test_roundtrip_reader_is_bit_exact_for_all_five_metric_columns():
    table = recovery.read_smoke_report_roundtrip(FAILED_A03_CSV)
    metrics = _failed_metrics()["details"]

    passing = table.loc[table.status.astype(str).str.upper() == "PASS"]
    assert tuple(passing.variant.astype(str)) == EXPECTED_VARIANTS
    comparisons = 0
    for row in passing.itertuples(index=False):
        expected = metrics[str(row.variant)]["metrics_development_only"]
        for metric in METRIC_NAMES:
            observed_value = float(getattr(row, metric))
            expected_value = float(expected[metric])
            assert observed_value == expected_value
            assert _float64_bits(observed_value) == _float64_bits(expected_value)
            comparisons += 1
    assert comparisons == 50


def test_production_semantic_validator_uses_project_roundtrip_reader(
    monkeypatch,
):
    original_reader = recovery.read_smoke_report_roundtrip
    observed: list[Path] = []

    def recording_reader(path):
        observed.append(Path(path).resolve())
        return original_reader(path)

    stored_ledger = json.loads(
        (FAILED_A03_RUN / "provenance/execution_ledger.json").read_text()
    )
    monkeypatch.setattr(
        core.amendment04_recovery,
        "read_smoke_report_roundtrip",
        recording_reader,
    )
    monkeypatch.setattr(
        core,
        "smoke_execution_ledger",
        lambda *args, **kwargs: stored_ledger,
    )
    identity = json.loads(
        (FAILED_A03_RUN / "config/run_identity.lock.json").read_text()
    )
    try:
        core.validate_smoke_semantic_completeness(FAILED_A03_RUN, identity)
    except core.IntegrityError:
        # Later A03 historical/provenance checks are independent of this call spy.
        pass

    assert observed == [FAILED_A03_CSV.resolve()]


def test_adjacent_binary64_metric_tamper_still_fails_exact_comparison(
    tmp_path,
    monkeypatch,
):
    run, identity = _synthetic_semantic_fixture(tmp_path, monkeypatch)
    table_path = run / "tables/SMOKE_REPORT.csv"
    table = recovery.read_smoke_report_roundtrip(table_path)
    original = float(table.loc[0, "average_precision"])
    adjacent = float(np.nextafter(original, np.inf))
    assert adjacent != original
    assert _ulp_distance(original, adjacent) == 1
    table.loc[0, "average_precision"] = adjacent
    table.to_csv(table_path, index=False)
    reopened = recovery.read_smoke_report_roundtrip(table_path)
    assert float(reopened.loc[0, "average_precision"]) == adjacent

    with pytest.raises(
        core.IntegrityError,
        match="smoke_table_matches_variant_details",
    ):
        core.validate_smoke_semantic_completeness(run, identity)


def test_smoke_metric_evidence_remains_fail_closed(
    tmp_path,
    monkeypatch,
):
    run, identity = _synthetic_semantic_fixture(tmp_path, monkeypatch)
    table_path = run / "tables/SMOKE_REPORT.csv"
    original_bytes = table_path.read_bytes()

    def rejects(mutator, expected_check: str | None = None):
        table_path.write_bytes(original_bytes)
        frame = recovery.read_smoke_report_roundtrip(table_path)
        replacement = mutator(frame)
        if isinstance(replacement, bytes):
            table_path.write_bytes(replacement)
        else:
            replacement.to_csv(table_path, index=False)
        with pytest.raises(core.IntegrityError) as caught:
            core.validate_smoke_semantic_completeness(run, identity)
        if expected_check is not None:
            assert expected_check in str(caught.value)

    rejects(
        lambda frame: b'variant,status,average_precision\n"unterminated',
    )
    rejects(
        lambda frame: frame.assign(
            average_precision=[np.inf, *frame.average_precision.iloc[1:]],
        ),
        "smoke_table_matches_variant_details",
    )
    rejects(
        lambda frame: frame.drop(columns=["average_precision"]),
        "smoke_table_matches_variant_details",
    )
    rejects(
        lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
        "smoke_table_ids_and_counts",
    )
    rejects(
        lambda frame: frame.iloc[::-1].reset_index(drop=True),
        "smoke_table_order",
    )


def test_amendment04_code_drift_is_the_exact_minimal_historical_path_set():
    # Later authorized amendments add files to the live tree.  Treat those
    # later files as the comparison baseline here so this historical test keeps
    # exercising Amendment 04's own four-path drift contract.
    live = recovery._live_code_tree(STUDY)
    baseline = dict(live)
    a03_baseline = recovery._a03_baseline_hashes(STUDY)
    expected = {
        "code/amendment04_recovery.py",
        "code/amendment_core.py",
        "code/run_study.py",
        "tests/test_amendment04.py",
    }
    for relative in expected:
        if relative in a03_baseline:
            baseline[relative] = a03_baseline[relative]
        else:
            baseline.pop(relative, None)
    allowed = recovery.code_drift_allowlist(
        STUDY,
        baseline_executed_hashes=baseline,
    )
    assert set(allowed) == {
        "code/amendment04_recovery.py",
        "code/amendment_core.py",
        "code/run_study.py",
        "tests/test_amendment04.py",
    }
    assert all(len(digest) == 64 for digest in allowed.values())


def test_amendment04_guard_is_distinct_and_starts_zero_of_one(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / recovery.FAILED_AMENDMENT02_RUN_ID).mkdir()
    (results / recovery.FAILED_AMENDMENT03_RUN_ID).mkdir()
    (results / f"{recovery.FAILED_AMENDMENT03_RUN_ID}.zip").write_bytes(
        b"immutable-forensic-placeholder"
    )
    guard = recovery.assert_run_guard(results)
    assert guard["status"] == "PASS"
    authorization = guard.get(
        "authorization",
        guard.get("replacement_authorization"),
    )
    assert authorization == "0_OF_1_BEFORE_RUN_CREATION"
    assert guard.get("new_run_id") in (None, "")


def test_amendment04_guard_consumes_exactly_one_without_reusing_prior_runs(
    tmp_path,
):
    results = tmp_path / "results"
    results.mkdir()
    (results / recovery.FAILED_AMENDMENT02_RUN_ID).mkdir()
    (results / recovery.FAILED_AMENDMENT03_RUN_ID).mkdir()
    forensic = results / f"{recovery.FAILED_AMENDMENT03_RUN_ID}.zip"
    forensic.write_bytes(b"immutable-forensic-placeholder")
    assert recovery.assert_run_guard(results)["authorization"] == (
        "0_OF_1_BEFORE_RUN_CREATION"
    )
    new_run = results / "run_20990101_000000_deadbeef_amendment04_cuda_smoke"
    consumed = recovery.create_authorized_run_directory(new_run, results)
    assert consumed["authorization"] == "1_OF_1_CONSUMED"
    assert (results / recovery.FAILED_AMENDMENT02_RUN_ID).is_dir()
    assert (results / recovery.FAILED_AMENDMENT03_RUN_ID).is_dir()
    assert forensic.read_bytes() == b"immutable-forensic-placeholder"
    staging = results / f"{new_run.name}_review_bundle_staging_verified.zip"
    staging.write_bytes(b"verified-transaction-staging")
    assert recovery.assert_run_guard(
        results, new_run_id=new_run.name,
    )["authorization"] == "1_OF_1_CONSUMED"
    with pytest.raises(recovery.RecoveryIntegrityError):
        recovery.create_authorized_run_directory(
            results / "run_20990101_000001_feedface_amendment04_cuda_smoke",
            results,
        )


def test_amendment04_guard_serializes_concurrent_run_creation(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / recovery.FAILED_AMENDMENT02_RUN_ID).mkdir()
    (results / recovery.FAILED_AMENDMENT03_RUN_ID).mkdir()
    (results / f"{recovery.FAILED_AMENDMENT03_RUN_ID}.zip").write_bytes(
        b"immutable-forensic-placeholder"
    )
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    queue = context.Queue()
    names = (
        "run_20990101_000010_deadbeef_amendment04_cuda_smoke",
        "run_20990101_000011_feedface_amendment04_cuda_smoke",
    )
    processes = [
        context.Process(
            target=_concurrent_guard_worker,
            args=(str(results), name, barrier, queue),
        )
        for name in names
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    outcomes = [queue.get(timeout=2) for _ in processes]
    assert sorted(outcome[0] for outcome in outcomes) == ["ERROR", "PASS"]
    assert sum(
        path.is_dir()
        for path in results.glob("run_*_amendment04_cuda_smoke")
    ) == 1


def test_concurrent_loser_never_mutates_an_unowned_winner_run(
    tmp_path,
    monkeypatch,
):
    study = tmp_path / "study"
    results = study / "results"
    winner = results / "run_20990101_000012_cafebabe_amendment04_cuda_smoke"
    (winner / "config").mkdir(parents=True)
    identity = winner / "config/run_identity.lock.json"
    identity.write_text(json.dumps({"run_id": winner.name, "state": "SMOKE_COMPLETE"}))
    before = identity.read_bytes()
    monkeypatch.setattr(core, "STUDY_ROOT", study)

    def loser_rejected_after_winner_appears(**kwargs):
        assert kwargs["created_run_holder"] == {}
        raise recovery.RecoveryIntegrityError(
            "Amendment 04 authorization was already consumed"
        )

    monkeypatch.setattr(
        core,
        "_run_cuda_smoke_impl",
        loser_rejected_after_winner_appears,
    )
    with pytest.raises(
        recovery.RecoveryIntegrityError,
        match="already consumed",
    ):
        core.run_cuda_smoke(
            preflight_run=ACCEPTED_PREFLIGHT,
            command_line="run_study.py --smoke --amendment-04-csv-roundtrip-one",
            amendment04_csv_roundtrip=True,
        )
    assert identity.read_bytes() == before
    assert not (winner / "logs/smoke_failure.txt").exists()


def test_amendment04_models_publication_is_complete_only():
    variants = list(EXPECTED_VARIANTS)
    official = variants[:-1]
    assert core.smoke_model_package_eligible(
        completed=variants,
        variants=variants,
        runnable_official=official,
        state="SMOKE_COMPLETE",
        amendment03_replacement=False,
        amendment04_csv_roundtrip=True,
    )
    assert not core.smoke_model_package_eligible(
        completed=variants,
        variants=variants,
        runnable_official=official,
        state="SMOKE_INCOMPLETE",
        amendment03_replacement=False,
        amendment04_csv_roundtrip=True,
    )
    assert not core.smoke_model_package_eligible(
        completed=variants[:-1],
        variants=variants,
        runnable_official=official,
        state="SMOKE_COMPLETE",
        amendment03_replacement=False,
        amendment04_csv_roundtrip=True,
    )


def test_canonical_review_publication_is_reopened_against_live_run(
    tmp_path,
    monkeypatch,
):
    run, _ = _synthetic_semantic_fixture(tmp_path, monkeypatch)
    verification = core.package_amended_run(run)
    reopened = core.verify_published_review_bundle(
        run,
        Path(verification["bundle_path"]),
        verification,
    )
    assert reopened["status"] == "PASS"
    assert reopened["crc_status"] == "PASS"
    assert reopened["embedded_manifest_status"] == "PASS"
    assert reopened["unique_safe_member_set"] == "PASS"
    assert reopened["live_member_byte_equality"] == "PASS"


def test_amendment04_rollback_preserves_the_primary_failure_log(
    tmp_path,
    monkeypatch,
):
    study = tmp_path / "study"
    run = study / "results/run_failure_amendment04_cuda_smoke"
    for relative in ("config", "provenance", "metrics", "logs"):
        (run / relative).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(core, "STUDY_ROOT", study)
    primary = RuntimeError("injected variant failure")
    core._write_amendment04_structured_failure(
        run_dir=run,
        phase="CUDA_SMOKE_VARIANTS",
        command_line="run_study.py --smoke --amendment-04-csv-roundtrip-one",
        error=primary,
        active_variant="manual_only",
        active_variant_index=2,
    )
    primary_bytes = (run / "logs/smoke_failure.txt").read_bytes()
    gate_inputs = {
        spec.key: {
            "classification": "VERIFIED",
            "passed": True,
            "evidence_refs": ["provenance/package_staging_verification.json"],
        }
        for spec in core.evidence_workflow.DEFAULT_GATE_SPECS
    }
    facts = core.evidence_workflow.WorkflowFacts(
        phase="CUDA_SMOKE",
        official_smoke_variants_completed=0,
        exploratory_smoke_variants_completed=0,
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
        protocol_amendment="AMENDMENT_02",
    )
    core.rollback_smoke_final_seal_failure(
        run_dir=run,
        preflight_run=study / "results/preflight",
        preflight_report={
            "protocol": "AMENDMENT_02",
            "training_table_path": "/source/train.csv",
            "training_table_sha256": "2" * 64,
            "r92_control": {"max_probability_difference": 0.0},
        },
        split_hash="3" * 64,
        model_bundle=None,
        gate_inputs=gate_inputs,
        facts=facts,
        smoke_metrics={},
        error=core.IntegrityError("downstream review seal failure"),
        ownership={},
        amendment04_csv_roundtrip=True,
        amendment04_recovery_validation={
            "sealed_probe_sha256": "4" * 64,
            "lineage_sha256": "5" * 64,
        },
        command_line="run_study.py --smoke --amendment-04-csv-roundtrip-one",
    )
    assert (run / "logs/smoke_failure.txt").read_bytes() == primary_bytes
    package_failure = (run / "logs/final_package_failure.txt").read_text()
    assert "PHASE=CUDA_SMOKE_REVIEW_PACKAGE" in package_failure
    assert "downstream review seal failure" in package_failure


@pytest.mark.parametrize("remaining", ["models", "adjacent"])
def test_outer_cleanup_removes_a_bound_partial_models_publication(
    tmp_path,
    monkeypatch,
    remaining,
):
    study = tmp_path / "study"
    results = study / "results"
    run = results / "run_partial_amendment04_cuda_smoke"
    (run / "provenance").mkdir(parents=True)
    models = results / f"{run.name}_NON_SCIENTIFIC_smoke_models.zip"
    adjacent = results / (
        f"{run.name}_NON_SCIENTIFIC_smoke_models_package_verification.json"
    )
    if remaining == "models":
        archive_root = f"{run.name}_smoke_models"
        with zipfile.ZipFile(models, "x") as archive:
            archive.writestr(f"{archive_root}/model.ubj", b"model")
            archive.writestr(f"{archive_root}/BUNDLE_MANIFEST.tsv", b"manifest")
    else:
        adjacent.write_text(json.dumps({
            "run_id": run.name,
            "path": str(models),
            "sha256": "0" * 64,
        }))
    monkeypatch.setattr(core, "STUDY_ROOT", study)
    core._cleanup_amendment03_models_after_unhandled_failure(run)
    assert not models.exists()
    assert not adjacent.exists()


@pytest.mark.parametrize(
    "mode_args",
    [
        ["--full"],
        ["--stability"],
        ["--resume", "/tmp/not-a-run"],
        ["--package", "/tmp/not-a-run"],
        ["--preflight"],
        ["--amendment-02-preflight"],
    ],
)
def test_amendment04_flag_cannot_enter_forbidden_modes(
    mode_args,
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_study.py", *mode_args, "--amendment-04-csv-roundtrip-one"],
    )
    assert run_study.main() == 3


def test_amendment04_and_amendment03_flags_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_study.py",
            "--smoke",
            "--preflight-run",
            str(ACCEPTED_PREFLIGHT),
            "--amendment-03-corrigendum-one",
            "--amendment-04-csv-roundtrip-one",
        ],
    )
    assert run_study.main() == 3


def test_amendment04_cli_routes_only_the_distinct_recovery_flag(monkeypatch):
    observed = {}

    def fake_smoke(**kwargs):
        observed.update(kwargs)
        return Path("/tmp/amendment04-dry-run-result")

    monkeypatch.setattr(core, "run_cuda_smoke", fake_smoke)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "code/run_study.py",
            "--smoke",
            "--preflight-run",
            str(ACCEPTED_PREFLIGHT),
            "--amendment-04-csv-roundtrip-one",
        ],
    )
    assert run_study.main() == 0
    assert observed["preflight_run"] == ACCEPTED_PREFLIGHT
    assert observed["amendment03_replacement"] is False
    assert observed["amendment04_csv_roundtrip"] is True


def test_amendment04_exact_invocation_contract(monkeypatch):
    command = (
        "code/run_study.py --smoke --preflight-run "
        f"{ACCEPTED_PREFLIGHT} --amendment-04-csv-roundtrip-one"
    )
    monkeypatch.setattr(sys, "executable", str(recovery.CUDA_PYTHON))
    monkeypatch.setattr(sys, "prefix", str(recovery.CUDA_ENV))
    monkeypatch.setenv("ABLATION_PYTHON", str(recovery.CUDA_PYTHON))
    identity = {"run_id": recovery.ACCEPTED_PREFLIGHT_RUN_ID}
    gate = {
        "accepted_preflight": {
            "run_id": recovery.ACCEPTED_PREFLIGHT_RUN_ID,
            "review_zip_sha256": recovery.ACCEPTED_PREFLIGHT_ZIP_SHA256,
        }
    }
    core._assert_amendment04_invocation(
        preflight_run=ACCEPTED_PREFLIGHT,
        command_line=command,
        identity=identity,
        immutable_gate=gate,
    )
    with pytest.raises(core.IntegrityError):
        core._assert_amendment04_invocation(
            preflight_run=ACCEPTED_PREFLIGHT,
            command_line=command + " --full",
            identity=identity,
            immutable_gate=gate,
        )


@pytest.mark.skipif(
    os.environ.get("AMENDMENT04_ACTIVE_CUDA_PROBE") != "1",
    reason="run only during the explicit Amendment 04 active CUDA gate",
)
def test_amendment04_active_cuda_production_helper(tmp_path):
    assert xgb.__version__ == "2.1.1"
    assert xgb.build_info().get("USE_CUDA") is True
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_name, driver_version = [
        value.strip() for value in gpu_query.stdout.splitlines()[0].split(",", 1)
    ]
    assert gpu_name == recovery.VISIBLE_GPU
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
    feature_names = [f"feature_{index:02d}" for index in range(12)]
    feature_sha256 = core.sha256_bytes(
        ("\n".join(feature_names) + "\n").encode("utf-8")
    )
    compat.embed_binary_classifier_metadata(
        classifier,
        expected_feature_count=12,
        feature_list_sha256=feature_sha256,
    )
    model_path = tmp_path / "amendment04-active-cuda.ubj"
    classifier.save_model(model_path)
    before = recovery.sha256_file(model_path)
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
    assert recovery.sha256_file(model_path) == before
    print(json.dumps({
        "classification": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
        "status": "PASS",
        "sys_executable": sys.executable,
        "active_cuda_fit": "PASS",
        "raw_booster_reload_on_cuda": "PASS",
        "cpu_fallback_detected": False,
        "cpu_fallback": False,
        "visible_gpu": gpu_name,
        "driver_version": driver_version,
        "fit_warnings": fit_warnings,
        "model_sha256": before,
        "four_way_reload_parity": result.evidence,
        "xgboost_version": xgb.__version__,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_use_cuda": xgb.build_info().get("USE_CUDA"),
        "xgboost_build_info": xgb.build_info(),
    }, sort_keys=True))


@pytest.mark.skipif(
    os.environ.get("AMENDMENT04_PACKAGE_REHEARSAL") != "1",
    reason="run only during the explicit Amendment 04 packaging rehearsal",
)
def test_temporary_adversarial_roundtrip_package_builds_and_reopens(
    tmp_path,
    monkeypatch,
):
    run, _ = _synthetic_semantic_fixture(tmp_path, monkeypatch)
    table_path = run / "tables/SMOKE_REPORT.csv"
    metrics_path = run / "metrics/SMOKE_REPORT.json"
    table = recovery.read_smoke_report_roundtrip(table_path)
    metrics = json.loads(metrics_path.read_text())

    adversarial = float.fromhex("0x1.0000000000001p-1")
    variant = str(table.loc[0, "variant"])
    table.loc[0, "average_precision"] = adversarial
    metrics["details"][variant]["metrics_development_only"][
        "average_precision"
    ] = adversarial
    table.to_csv(table_path, index=False)
    metrics_path.write_text(json.dumps(metrics))

    verification = core.package_amended_run(run)
    bundle = Path(verification["bundle_path"])
    assert STUDY.resolve() not in bundle.resolve().parents
    assert verification["crc_status"] == "PASS"
    assert verification["independent_reopen_member_verification"] == "PASS"
    assert verification["package_completeness"][
        "smoke_semantic_completeness"
    ]["status"] == "PASS"
    reopened = core.verify_zip(bundle, embedded_run_manifest=True)
    assert reopened["status"] == "PASS"
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert all(not name.startswith("/") and ".." not in Path(name).parts for name in names)
