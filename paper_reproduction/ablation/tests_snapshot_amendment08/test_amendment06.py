from __future__ import annotations

import ast
import base64
from contextlib import nullcontext
import copy
import io
import json
import os
from pathlib import Path
import shutil
import stat
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest

import amendment06_full as full


def test_smoke_admission_reads_the_production_nested_semantic_result() -> None:
    accepted = {
        "run_id": full.ACCEPTED_SMOKE.name,
        "full_authorized": False,
        "reference_validation": {
            "smoke_semantic_completeness": {
                "status": "PASS",
                "ready_for_full_awaiting_external_review": True,
            },
        },
    }
    assert full._smoke_identity_is_ready_for_full(accepted) is True
    assert full._smoke_identity_is_ready_for_full({
        **accepted,
        "reference_validation": {
            "smoke_semantic_completeness": {
                "status": "PASS",
                "ready_for_full_awaiting_external_review": False,
            },
        },
    }) is False
    # A stale top-level token must not bypass the production nested contract.
    assert full._smoke_identity_is_ready_for_full({
        "run_id": full.ACCEPTED_SMOKE.name,
        "full_authorized": False,
        "ready_for_full_awaiting_external_review": True,
    }) is False


def _identity(run_dir: Path) -> None:
    full.atomic_write_strict_json(
        run_dir / "config/run_identity.lock.json",
        {"run_id": run_dir.name, "run_kind": full.RUN_KIND, "state": "FULL_IN_PROGRESS"},
    )
    full.publish_strict_json_no_clobber(
        full._state_marker_path(run_dir, "FULL_IN_PROGRESS"),
        {
            "run_id": run_dir.name,
            "run_kind": full.RUN_KIND,
            "phase": "FULL_IN_PROGRESS",
            "phase_index": 0,
            "status": "PASS",
            "previous_marker_sha256": None,
            "recorded_at": "2026-08-20T00:00:00+00:00",
        },
    )


def test_strict_json_boundary_is_stable_nonmutating_and_rfc8259() -> None:
    value = {
        "python": [float("nan"), float("inf"), -float("inf")],
        "numpy": np.asarray([np.float32("nan"), np.float64(2.5)]),
        "nested": (np.int64(7), {"finite": 1.25}),
    }
    original = copy.deepcopy(value)
    first = full.strict_full_json_bytes(value)
    second = full.strict_full_json_bytes(value)
    assert first == second
    assert first.endswith(b"\n")
    assert b"NaN" not in first and b"Infinity" not in first
    assert full.strict_full_loads(first)["python"] == [None, None, None]
    assert np.isnan(value["python"][0]) and np.isinf(value["python"][1])
    assert np.array_equal(value["numpy"], original["numpy"], equal_nan=True)
    with pytest.raises(full.StrictJSONError):
        full.strict_full_loads('{"bad":NaN}')
    assert full.strict_full_loads('{"ok":null}') == {"ok": None}


def test_required_metric_nan_is_preserved_and_hard_stops() -> None:
    metrics = {name: 1.0 for name in full.REQUIRED_METRICS}
    metrics["average_precision"] = float("nan")
    with pytest.raises(full.Amendment06IntegrityError):
        full.assert_required_metrics_finite(metrics)
    assert np.isnan(metrics["average_precision"])


def test_publish_no_clobber_never_deletes_preexisting_collision(tmp_path: Path) -> None:
    path = tmp_path / "owned-by-someone-else.txt"
    path.write_bytes(b"preserve-me")
    with pytest.raises(FileExistsError):
        full.publish_bytes_no_clobber(path, b"replacement")
    assert path.read_bytes() == b"preserve-me"


def test_atomic_no_clobber_leaves_no_named_temp_or_external_alias(tmp_path: Path) -> None:
    path = tmp_path / "final.json"
    full.publish_bytes_no_clobber(path, b"complete-bytes")
    assert path.read_bytes() == b"complete-bytes"
    assert path.stat().st_nlink == 1
    assert [member.name for member in tmp_path.iterdir()] == ["final.json"]


def test_directory_publication_is_atomic_no_clobber(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "a").write_text("new")
    (destination / "a").write_text("old")
    with pytest.raises(FileExistsError):
        full._rename_directory_no_clobber(source, destination)
    assert (source / "a").read_text() == "new"
    assert (destination / "a").read_text() == "old"


def test_attempt_publication_moves_without_a_writable_staging_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", runtime)
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    run_dir.mkdir()
    attempt_id = "a" * 32
    staging = full._attempt_root(run_dir, "TEST", "v", attempt_id)
    relative = "metrics/result.json"
    source = staging / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"immutable-result")

    record = full._publish_attempt_file(
        run_dir=run_dir, staging=staging, relative=relative,
        phase="TEST", variant="v", attempt_id=attempt_id,
    )
    destination = run_dir / relative
    assert destination.read_bytes() == b"immutable-result"
    assert destination.stat().st_nlink == 1
    assert not source.exists()
    assert record["sha256"] == full.sha256_file(destination)
    assert [event["event"] for event in full._read_journal(run_dir)] == [
        "ARTIFACT_PUBLICATION_INTENT", "ARTIFACT_PUBLISHED",
    ]


def test_atomic_event_journal_rejects_torn_or_unexpected_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", tmp_path / "runtime")
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    run_dir.mkdir()
    full._journal_event(run_dir, "FIXTURE", status="PASS")
    journal = full._journal_path(run_dir)
    (journal / "00000001.json.partial").write_text('{"torn":')
    with pytest.raises(full.Amendment06IntegrityError, match="unexpected"):
        full._read_journal(run_dir)
    (journal / "00000001.json.partial").unlink()
    (journal / "00000002.json").write_text("{}\n")
    with pytest.raises(full.Amendment06IntegrityError, match="noncontiguous"):
        full._read_journal(run_dir)


def test_exact_target_journal_route_is_production_safe_and_fixture_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulated_production = tmp_path / "simulated_production"
    production_runtime = simulated_production / "runtime"
    production_results = simulated_production / "results"
    monkeypatch.setattr(full, "PRODUCTION_EXPECTED_RUNTIME_ROOT", production_runtime)
    monkeypatch.setattr(full, "PRODUCTION_RESULTS_ROOT", production_results)
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", production_runtime)

    outside_run = tmp_path / "outside" / full.A07_TARGET_RUN_ID
    outside_run.mkdir(parents=True)
    with pytest.raises(full.Amendment06IntegrityError, match="outside canonical"):
        full._journal_event(outside_run, "MUST_NOT_WRITE")
    assert not production_runtime.exists()

    live_run = production_results / full.A07_TARGET_RUN_ID
    live_run.mkdir(parents=True)
    with pytest.raises(full.Amendment06IntegrityError, match="pytest cannot write"):
        full._journal_event(live_run, "MUST_NOT_WRITE")
    assert not production_runtime.exists()

    isolated_root = tmp_path / "isolated_fixture"
    isolated_runtime = isolated_root / "runtime"
    isolated_run = isolated_root / "results" / full.A07_TARGET_RUN_ID
    isolated_run.mkdir(parents=True)
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", isolated_runtime)
    full._journal_event(isolated_run, "ISOLATED_FIXTURE", status="PASS")
    assert [event["event"] for event in full._read_journal(isolated_run)] == [
        "ISOLATED_FIXTURE"
    ]
    assert (
        isolated_runtime / "journals" / full.A07_TARGET_RUN_ID / "00000000.json"
    ).is_file()
    assert not production_runtime.exists()


def test_resume_runtime_defaults_follow_the_isolated_runtime_after_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_a06 = tmp_path / "isolated_a06"
    isolated_execution = isolated_a06 / "runtime"
    isolated_execution.mkdir(parents=True)
    preflight = tmp_path / "accepted_preflight"
    smoke = tmp_path / "accepted_smoke"
    preflight.mkdir()
    smoke.mkdir()

    monkeypatch.setattr(full, "A06_RUNTIME", isolated_a06)
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", isolated_execution)
    monkeypatch.setattr(full, "ACCEPTED_PREFLIGHT", preflight)
    monkeypatch.setattr(full, "ACCEPTED_SMOKE", smoke)
    monkeypatch.setattr(full, "CUDA_PYTHON", Path(full.sys.executable).resolve())
    monkeypatch.setattr(full, "CUDA_ENV", Path(full.sys.prefix).resolve())
    monkeypatch.setattr(full.xgb, "__version__", "2.1.1")
    monkeypatch.setattr(full.sklearn, "__version__", "1.7.2")
    monkeypatch.setattr(full.xgb, "build_info", lambda: {"USE_CUDA": True})
    monkeypatch.setenv("ABLATION_RUNTIME_ROOT", str(isolated_execution))

    with full.amendment06_lock():
        assert (isolated_a06 / "amendment06_full.lock").is_file()
    result = full.assert_exact_invocation(
        preflight_run=preflight, smoke_run=smoke,
    )
    assert result["runtime_root"] == str(isolated_execution.resolve())


def test_one_run_guard_rejects_symlink_or_special_entry(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    unsafe = tmp_path / f"run_unsafe{full.RUN_SUFFIX}"
    unsafe.symlink_to(real, target_is_directory=True)
    with pytest.raises(full.Amendment06IntegrityError, match="Unsafe one-Run guard"):
        full.existing_full_runs(tmp_path)


def test_intent_precedes_directory_and_missing_receipt_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(full, "A06_RUNTIME", runtime)
    monkeypatch.setattr(full, "RESULTS_ROOT", results)
    intent = full._reserve_full_run_id()
    assert full._run_intent_path().is_file()
    run_dir = Path(intent["run_path"])
    run_dir.mkdir()
    _identity(run_dir)
    assert not full._run_receipt_path().exists()
    receipt = full._publish_or_validate_run_receipt(run_dir)
    assert receipt["authorization_consumed"] is True
    assert receipt["run_id"] == intent["run_id"]
    assert full._reserve_full_run_id() == intent


def test_identity_stays_immutable_while_state_markers_advance(tmp_path: Path) -> None:
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    run_dir.mkdir()
    _identity(run_dir)
    identity_before = (run_dir / "config/run_identity.lock.json").read_bytes()
    full._set_run_state(run_dir, "FULL_MODELS_FROZEN")
    assert full._current_run_state(run_dir) == "FULL_MODELS_FROZEN"
    assert (run_dir / "config/run_identity.lock.json").read_bytes() == identity_before
    marker = full.strict_full_load_file(full._state_marker_path(run_dir, "FULL_MODELS_FROZEN"))
    assert marker["previous_marker_sha256"] == full.sha256_file(
        full._state_marker_path(run_dir, "FULL_IN_PROGRESS")
    )


def test_state_marker_gap_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    run_dir.mkdir()
    _identity(run_dir)
    full.publish_strict_json_no_clobber(
        full._state_marker_path(run_dir, "FULL_EXTERNAL_AUDIT_SCORED"),
        {
            "run_id": run_dir.name,
            "run_kind": full.RUN_KIND,
            "phase": "FULL_EXTERNAL_AUDIT_SCORED",
            "phase_index": 2,
            "status": "PASS",
            "previous_marker_sha256": "0" * 64,
            "recorded_at": "2026-08-20T00:00:00+00:00",
        },
    )
    with pytest.raises(full.Amendment06IntegrityError, match="gap"):
        full._current_run_state(run_dir)


def test_final_state_without_output_manifest_resumes_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", tmp_path / "runtime")
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    run_dir.mkdir()
    _identity(run_dir)
    for phase in full.RUN_PHASES[1:]:
        full._set_run_state(run_dir, phase)
    assert full._current_run_state(run_dir) == "FULL_SCIENTIFIC_COMPLETE"
    assert not (run_dir / "OUTPUT_MANIFEST_FINAL.tsv").exists()

    class RecoveryContinued(RuntimeError):
        pass

    monkeypatch.setattr(full, "_validate_run_identity", lambda path: {})
    monkeypatch.setattr(
        full, "validate_final_scientific_run",
        lambda path: pytest.fail("incomplete finalization was treated as sealed"),
    )
    monkeypatch.setattr(
        full, "initialize_run_evidence",
        lambda path: (_ for _ in ()).throw(RecoveryContinued()),
    )
    with pytest.raises(RecoveryContinued):
        full._execute_scientific_full(
            run_dir=run_dir,
            preflight_run=full.ACCEPTED_PREFLIGHT,
            smoke_run=full.ACCEPTED_SMOKE,
        )


def test_owned_incomplete_publication_is_hash_verified_and_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", runtime)
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    relative = "metrics/example.json"
    path = run_dir / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"owned")
    record = full._publication_record(path, relative)
    full._journal_event(
        run_dir, "ARTIFACT_PUBLICATION_INTENT", phase="TEST", variant_id="v",
        attempt_id="a", **record,
    )
    full._recover_unpublished_paths(
        run_dir=run_dir, phase="TEST", variant="v",
        completion_relative="metrics/completion.json",
        possible_relatives=[relative],
    )
    assert not path.exists()
    quarantined = list((runtime / "quarantine").rglob("example.json"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"owned"


def test_quarantine_intent_only_crash_is_reconciled_and_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", runtime)
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    relative = "metrics/example.json"
    source = run_dir / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"owned")
    record = full._publication_record(source, relative)
    attempt_id = "b" * 32
    full._journal_event(
        run_dir, "ARTIFACT_PUBLICATION_INTENT", phase="TEST", variant_id="v",
        attempt_id=attempt_id, **record,
    )
    transaction_id = "c" * 32
    destination = (
        runtime / "quarantine" / run_dir.name / "TEST" / "v"
        / transaction_id / relative
    )
    full._journal_event(
        run_dir, "ARTIFACT_QUARANTINE_INTENT", phase="TEST", variant_id="v",
        attempt_id=attempt_id, transaction_id=transaction_id,
        relative_path=relative, sha256=record["sha256"],
        size_bytes=record["size_bytes"], quarantine_path=str(destination),
    )
    destination.parent.mkdir(parents=True)
    full._rename_directory_no_clobber(source, destination)

    full._recover_unpublished_paths(
        run_dir=run_dir, phase="TEST", variant="v",
        completion_relative="metrics/completion.json",
        possible_relatives=[relative],
    )
    assert not source.exists()
    assert destination.read_bytes() == b"owned"
    committed = [
        event for event in full._read_journal(run_dir)
        if event["event"] == "ARTIFACT_QUARANTINE_COMMITTED"
    ]
    assert len(committed) == 1
    assert committed[0]["transaction_id"] == transaction_id
    assert committed[0]["recovered_after_interruption"] is True


def test_unowned_incomplete_publication_is_never_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", tmp_path / "runtime")
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    path = run_dir / "metrics/example.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"unowned")
    with pytest.raises(full.Amendment06IntegrityError, match="ownership proof"):
        full._recover_unpublished_paths(
            run_dir=run_dir, phase="TEST", variant="v",
            completion_relative="metrics/completion.json",
            possible_relatives=["metrics/example.json"],
        )
    assert path.read_bytes() == b"unowned"


def test_population_cache_rejects_extra_member_before_loading_arrays(tmp_path: Path) -> None:
    destination = tmp_path / "cache"
    destination.mkdir()
    (destination / "unowned-extra.bin").write_bytes(b"extra")
    with pytest.raises(full.Amendment06IntegrityError, match="member set differs"):
        full._validate_population_cache(
            run_dir=tmp_path / f"run_fixture{full.RUN_SUFFIX}",
            destination=destination, locked=None, development=None,
        )


def test_prediction_gzip_exact_roundtrip_and_metrics() -> None:
    frame = pd.DataFrame({
        "stable_candidate_id": ["a", "b", "c", "d"],
        "card_id": ["c1", "c1", "c2", "c2"],
        "true_label": [0, 1, 0, 1],
        "probability": np.asarray([0.1, 0.9, 0.7, 0.3], np.float32).astype(np.float64),
        "prediction_at_0_5": [0, 1, 1, 0],
        "review_weight": np.asarray([1.0, 0.7, 0.5, 1.0], np.float32).astype(np.float64),
        "variant_id": ["fixture"] * 4,
        "split_population": ["validation:fixture"] * 4,
        "row_order": np.arange(4),
    }, columns=full.PREDICTION_COLUMNS)
    first = full.deterministic_csv_gzip_bytes(frame)
    second = full.deterministic_csv_gzip_bytes(frame)
    assert first == second
    reopened, evidence = full.validate_prediction_roundtrip(first, frame)
    assert evidence["status"] == "PASS" and evidence["mismatch_count"] == 0
    assert full._binary_metrics(reopened.true_label, reopened.probability)["average_precision"] == pytest.approx(5 / 6)


def test_exact_table_csv_reopen_preserves_values_and_missingness(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "card_id": ["c1", "c2", "c3"],
        "average_precision": [0.125, np.nan, 0.875],
        "candidate_count": [3, 4, 5],
        "status": ["PASS", "UNDEFINED_SINGLE_CLASS", "PASS"],
    })
    data, evidence = full.exact_table_csv_bytes(frame)
    path = tmp_path / "per_card.csv"
    path.write_bytes(data)
    assert full.validate_exact_table_csv(path, frame) == evidence
    path.write_bytes(data.replace(b"0.125", b"0.126"))
    with pytest.raises(full.Amendment06IntegrityError, match="bytes differ"):
        full.validate_exact_table_csv(path, frame)


def test_variant_status_matches_frozen_packaging_tokens() -> None:
    text = full._variant_complete_status("run", "variant").decode()
    assert "TRAINING_AND_VALIDATION_COMPLETE" in text
    assert "EXTERNAL_AUDIT_SCORING_COMPLETE" in text


def test_models_bundle_contract_has_exact_seven_variant_members() -> None:
    variant = "full_new_reference"
    selected = {
        name for name in (
            "model.ubj", "feature_list.txt", "model_metadata.json",
            "transform_binding.json", "training_completion_manifest.json",
            "external_scoring_manifest.json", "STATUS.txt", "extra.json",
        )
        if full._models_bundle_flag(f"models/{variant}/{name}")
    }
    assert selected == {
        "model.ubj", "feature_list.txt", "model_metadata.json",
        "transform_binding.json", "training_completion_manifest.json",
        "external_scoring_manifest.json", "STATUS.txt",
    }
    assert full._models_bundle_flag("config/feature_manifest.csv")
    assert full._models_bundle_flag("config/feature_dependency_graph.json")


def test_reporting_completion_commit_is_reconstructed_after_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", tmp_path / "runtime")
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    completion = run_dir / "provenance/reporting_completion_manifest.json"
    completion.parent.mkdir(parents=True)
    completion.write_text("{}\n")
    monkeypatch.setattr(
        full, "_validate_reporting_completion",
        lambda path: {"status": "PASS", "run_id": path.name},
    )
    result = full._build_reports_and_bootstrap_impl(
        run_dir=run_dir, locked=None, cache=None, audit=None,
        scoring_results=[],
    )
    assert result["status"] == "PASS"
    events = full._read_journal(run_dir)
    assert sum(event["event"] == "REPORTING_COMPLETION_COMMITTED" for event in events) == 1
    assert sum(event["event"] == "COMPLETED_REPORTING_SKIPPED" for event in events) == 1


@pytest.mark.parametrize("failure", [False, True])
def test_bounded_fit_progress_stops_on_success_and_failure(
    failure: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates: list[tuple] = []
    monkeypatch.setattr(full, "_progress", lambda *args, **kwargs: updates.append((args, kwargs)))
    monitor = full.BoundedFitProgress(
        "run", "variant", interval_seconds=0.01,
        snapshot=lambda: {"rss_bytes": 1, "gpu_memory_mib": 2},
    )
    monitor.__enter__()
    time.sleep(0.035)
    error = RuntimeError("planned") if failure else None
    monitor.__exit__(RuntimeError if failure else None, error, None)
    assert not monitor.running
    assert any(args[:2] == ("FULL_VARIANT_FIT", "ACTIVE") for args, _ in updates)


def test_post_freeze_guards_cover_every_training_and_resampling_primitive() -> None:
    expected = {
        ("XGBClassifier", "fit"),
        ("xgboost", "train"),
        ("xgboost", "cv"),
        ("Booster", "update"),
        ("Booster", "boost"),
        ("SimpleImputer", "fit"),
        ("SimpleImputer", "fit_transform"),
        ("amendment02_resampling", "build_frozen_resampling_population"),
        ("amendment_core", "augment_with_lineage"),
        ("amendment_core", "augmentation_plan_counts"),
        ("amendment_core", "replay_augmentation_weights"),
        ("ablation_core", "apply_training_augmentation"),
        ("ablation_core", "safe_smote_resample"),
        ("SafeSMOTE", "fit_resample"),
    }
    assert full._POST_FREEZE_GUARDS_INSTALLED is False
    try:
        full._install_post_freeze_guards()
        observed = {
            (str(getattr(owner, "__name__", type(owner).__name__)), name)
            for owner, name, _ in full._GUARDED_ORIGINALS
        }
        assert observed == expected
        for owner, name, _ in full._GUARDED_ORIGINALS:
            with pytest.raises(full.PostFreezeScientificCallError):
                getattr(owner, name)()
    finally:
        for owner, name, original in reversed(full._GUARDED_ORIGINALS):
            setattr(owner, name, original)
        full._GUARDED_ORIGINALS.clear()
        full._POST_FREEZE_GUARDS_INSTALLED = False


def test_external_scoring_uses_independent_raw_and_compatibility_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.ubj"
    model_path.write_bytes(b"fixture-model")
    compatibility_booster = object()
    classifier = SimpleNamespace(get_booster=lambda: compatibility_booster)
    calls: list[str] = []

    def compatibility_loader(*args, **kwargs):
        calls.append("compatibility")
        return SimpleNamespace(classifier=classifier, evidence={})

    class RawBooster:
        def load_model(self, path):
            calls.append("raw")

        def set_param(self, values):
            assert values == {"device": "cuda"}

    monkeypatch.setattr(full.model_compat, "load_binary_classifier_compat", compatibility_loader)
    monkeypatch.setattr(full.xgb, "Booster", RawBooster)
    loaded, raw, evidence = full.load_independent_external_predictors(
        model_path, expected_feature_count=1,
        expected_feature_list_sha256="0" * 64,
        expected_model_sha256=full.sha256_file(model_path),
    )
    assert calls == ["compatibility", "raw"]
    assert raw is not loaded.classifier.get_booster()
    assert evidence["independent_objects"] is True


def test_same_run_interruption_resume_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stable synthetic rehearsal node used by the final pre-Run gate."""
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", runtime)
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    run_dir.mkdir()
    completed = "full_new_reference"
    completion = run_dir / full._training_completion_relative(completed)
    completion.parent.mkdir(parents=True)
    completion.write_text("{}\n")
    project_calls: list[str] = []
    fit_constructions: list[str] = []
    active_prediction_calls: list[bool] = []

    def valid_completion(**kwargs):
        active_prediction_calls.append(kwargs["active_prediction"])
        return {"status": "PASS", "variant_id": kwargs["variant"]}

    monkeypatch.setattr(full, "validate_training_completion", valid_completion)
    monkeypatch.setattr(
        full.xgb, "XGBClassifier",
        lambda **kwargs: fit_constructions.append("fit") or pytest.fail("completed Variant retrained"),
    )
    result = full.train_one_variant(
        run_dir=run_dir, locked=None, cache=None, development=None, variant=completed,
    )
    assert result["status"] == "PASS"
    assert fit_constructions == []
    assert active_prediction_calls == [True]

    incomplete = "manual_only"
    partial_relative = full._training_relatives(incomplete)[0]
    partial = run_dir / partial_relative
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"owned-partial")
    record = full._publication_record(partial, partial_relative)
    full._journal_event(
        run_dir, "ARTIFACT_PUBLICATION_INTENT", phase="TRAINING",
        variant_id=incomplete, attempt_id="interrupted", **record,
    )

    class RetryReached(RuntimeError):
        pass

    def retry_projection(cache, locked, variant):
        project_calls.append(variant)
        raise RetryReached("retry callback reached")

    monkeypatch.setattr(full, "project_variant", retry_projection)
    with pytest.raises(RetryReached):
        full.train_one_variant(
            run_dir=run_dir, locked=None, cache=None,
            development=None, variant=incomplete,
        )
    assert project_calls == [incomplete]
    assert not partial.exists()
    quarantined = list((runtime / "quarantine").rglob("model.ubj"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"owned-partial"

    invalid = "deep_only"
    invalid_completion = run_dir / full._training_completion_relative(invalid)
    invalid_completion.parent.mkdir(parents=True, exist_ok=True)
    invalid_completion.write_text("{}\n")

    def invalid_completion_validator(**kwargs):
        raise full.Amendment06IntegrityError("invalid completion")

    monkeypatch.setattr(full, "validate_training_completion", invalid_completion_validator)
    with pytest.raises(full.Amendment06IntegrityError, match="invalid completion"):
        full.train_one_variant(
            run_dir=run_dir, locked=None, cache=None,
            development=None, variant=invalid,
        )
    assert project_calls == [incomplete]
    assert {event["run_id"] for event in full._read_journal(run_dir)} == {run_dir.name}


def _copy_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, list[dict]]:
    runtime = tmp_path / "a06_runtime"
    runtime.mkdir()
    baseline_tsv = runtime / "baseline.tsv"
    shutil.copy2(full.BASELINE_TREE_TSV, baseline_tsv)
    study = tmp_path / "study"
    snapshot = full.A06_RUNTIME / "pre_edit/baseline_snapshot"
    shutil.copytree(snapshot / "code", study / "code")
    shutil.copytree(snapshot / "tests", study / "tests")
    monkeypatch.setattr(full, "A06_RUNTIME", runtime)
    monkeypatch.setattr(full, "BASELINE_TREE_TSV", baseline_tsv)
    genesis = full._load_concatenated_json(full.AUTHORIZED_CHANGE_LEDGER)[:2]
    return runtime, study, genesis


def _complete_edit_event(
    runtime: Path, study: Path, genesis: list[dict],
) -> dict:
    before = full._baseline_tree_mapping()
    added = study / "tests/test_final.py"
    added.write_text("def test_final():\n    assert True\n")
    after = full.live_code_test_mapping(study)
    changed = full._changed_tree_records(before, after)
    log = runtime / "logs/edit.log"
    log.parent.mkdir()
    log.write_text("PASS\n")
    before_sha, before_count, before_bytes = full._tree_summary(before)
    after_sha, after_count, after_bytes = full._tree_summary(after)
    return {
        "event": "AUTHORIZED_EDIT_SET_COMMITTED",
        "event_index": 2,
        "previous_event_index": 1,
        "previous_event_sha256": full.sha256_bytes(
            full.strict_full_canonical_json_bytes(genesis[1])
        ),
        "authorization": full.AUTHORIZATION,
        "prompt_sha256": full.PROMPT_SHA256,
        "purpose": "focused final fixture",
        "scope": [item["relative_path"] for item in changed],
        "before_tree": before,
        "after_tree": after,
        "before_tree_sha256": before_sha,
        "after_tree_sha256": after_sha,
        "before_file_count": before_count,
        "after_file_count": after_count,
        "before_total_bytes": before_bytes,
        "after_total_bytes": after_bytes,
        "changed_paths": changed,
        "diff_sha256": full.sha256_bytes(full.strict_full_canonical_json_bytes(changed)),
        "test_evidence": [{
            "category": "targeted_tests",
            "status": "PASS",
            "command": "pytest fixture",
            "exit_code": 0,
            "log_path": str(log),
            "log_size_bytes": log.stat().st_size,
            "log_sha256": full.sha256_file(log),
        }],
        "final_byte_freeze": True,
    }


def _write_ledger(path: Path, events: list[dict]) -> None:
    path.write_bytes(b"".join(full.strict_full_canonical_json_bytes(item) + b"\n" for item in events))


def test_ledger_rejects_minimal_tree_only_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, study, genesis = _copy_baseline(tmp_path, monkeypatch)
    event = _complete_edit_event(runtime, study, genesis)
    event.pop("diff_sha256")
    ledger = runtime / "ledger.jsonl"
    _write_ledger(ledger, [*genesis, event])
    with pytest.raises(full.Amendment06IntegrityError, match="incomplete"):
        full.validate_authorized_change_chain(study_root=study, ledger_path=ledger)


def test_ledger_accepts_complete_hash_chained_final_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, study, genesis = _copy_baseline(tmp_path, monkeypatch)
    event = _complete_edit_event(runtime, study, genesis)
    ledger = runtime / "ledger.jsonl"
    _write_ledger(ledger, [*genesis, event])
    result = full.validate_authorized_change_chain(study_root=study, ledger_path=ledger)
    assert result["status"] == "PASS"
    assert result["latest_event_index"] == 2
    assert result["canonical_tree_sha256"] == event["after_tree_sha256"]


def test_ledger_preserves_historical_log_hashes_while_latest_log_is_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, study, genesis = _copy_baseline(tmp_path, monkeypatch)
    first = _complete_edit_event(runtime, study, genesis)
    ledger = runtime / "ledger.jsonl"
    _write_ledger(ledger, [*genesis, first])

    shared_log = Path(first["test_evidence"][0]["log_path"])
    shared_log.write_text("PASS FINAL\n")
    (study / "tests/test_final_followup.py").write_text(
        "def test_final_followup():\n    assert True\n"
    )
    latest_evidence = full.make_test_evidence_record(
        category="targeted_tests",
        command="pytest final followup",
        log_path=shared_log,
    )
    second = full.build_authorized_edit_event(
        purpose="final followup fixture",
        test_evidence=[latest_evidence],
        study_root=study,
        ledger_path=ledger,
    )
    _write_ledger(ledger, [*genesis, first, second])

    result = full.validate_authorized_change_chain(
        study_root=study, ledger_path=ledger,
    )
    assert result["latest_event_index"] == 3
    assert first["test_evidence"][0]["log_sha256"] != full.sha256_file(shared_log)

    malformed_values = (
        ("log_sha256", "not-a-digest"),
        ("log_size_bytes", True),
        ("log_path", "relative/historical.log"),
    )
    for key, value in malformed_values:
        malformed_first = copy.deepcopy(first)
        malformed_first["test_evidence"][0][key] = value
        rebound_second = copy.deepcopy(second)
        rebound_second["previous_event_sha256"] = full.sha256_bytes(
            full.strict_full_canonical_json_bytes(malformed_first)
        )
        _write_ledger(ledger, [*genesis, malformed_first, rebound_second])
        with pytest.raises(
            full.Amendment06IntegrityError, match="test log binding differs",
        ):
            full.validate_authorized_change_chain(
                study_root=study, ledger_path=ledger,
            )

    _write_ledger(ledger, [*genesis, first, second])
    shared_log.write_text("TAMPERED\n")
    with pytest.raises(full.Amendment06IntegrityError, match="test log binding differs"):
        full.validate_authorized_change_chain(study_root=study, ledger_path=ledger)


def test_final_prerun_evidence_requires_all_six_bound_rehearsals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, study, genesis = _copy_baseline(tmp_path, monkeypatch)
    event = _complete_edit_event(runtime, study, genesis)
    ledger = runtime / "ledger.jsonl"
    _write_ledger(ledger, [*genesis, event])
    monkeypatch.setattr(full, "AUTHORIZED_CHANGE_LEDGER", ledger)
    chain = full.validate_authorized_change_chain(study_root=study, ledger_path=ledger)
    facts = {
        "syntax_compile": {"all_changed_python_files_compiled": True},
        "targeted_tests": {"amendment06_targeted_tests_passed": True},
        "full_suite_exactly_once": {
            "full_project_suite_passed": True, "suite_invocation_count": 1,
        },
        "active_cuda_probe": {
            "xgboost_version": "2.1.1", "scikit_learn_version": "1.7.2",
            "use_cuda": True, "gpu_name": "NVIDIA GeForce RTX 4090",
            "cpu_fallback_detected": False, "fit_save_reload_predict_status": "PASS",
        },
        "package_synthetic_rehearsal": {
            "synthetic_full_fixture": True, "production_packager_used": True,
            "production_final_validator_used": True,
            "package_retry_without_training": True, "scientific_api_calls": 0,
        },
        "resume_rehearsal": {
            "same_run_id_preserved": True, "completed_variant_skipped": True,
            "unpublished_incomplete_variant_retried": True,
            "completed_variant_retrained": False, "distinct_full_run_ids": 1,
        },
    }
    checks = []
    for category in full.PRERUN_CHECK_CATEGORIES:
        log = runtime / f"logs/{category}.log"
        log.parent.mkdir(exist_ok=True)
        log.write_text("PASS\n")
        checks.append({
            "category": category, "status": "PASS", "command": category,
            "exit_code": 0, "log_path": str(log),
            "log_size_bytes": log.stat().st_size, "log_sha256": full.sha256_file(log),
            "facts": facts[category],
        })
    evidence = {
        "status": "PASS", "authorization": full.AUTHORIZATION,
        "prompt_sha256": full.PROMPT_SHA256,
        "generated_at": "2026-08-20T00:00:00+00:00",
        "live_code_tree_sha256": chain["canonical_tree_sha256"],
        "live_code_tree_file_count": len(chain["mapping"]),
        "live_code_tree_total_bytes": sum(item["size_bytes"] for item in chain["mapping"].values()),
        "authorized_change_ledger_path": str(ledger),
        "authorized_change_ledger_sha256": chain["ledger_sha256"],
        "latest_event_index": chain["latest_event_index"],
        "latest_event_sha256": chain["latest_event_sha256"],
        "full_suite_invocation_count": 1,
        "checks": checks,
    }
    path = runtime / "final.json"
    full.atomic_write_strict_json(path, evidence)
    assert full.validate_final_prerun_evidence(chain, path=path)["status"] == "PASS"
    evidence["checks"] = checks[:-1]
    full.atomic_write_strict_json(path, evidence)
    with pytest.raises(full.Amendment06IntegrityError):
        full.validate_final_prerun_evidence(chain, path=path)


def test_output_manifest_excludes_itself_and_uses_exact_boolean_tokens(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "models/v").mkdir(parents=True)
    (run_dir / "models/v/model.ubj").write_bytes(b"model")
    (run_dir / "metrics").mkdir()
    (run_dir / "metrics/result.json").write_text("{}\n")
    data = full.output_manifest_bytes(run_dir)
    frame = pd.read_csv(io.BytesIO(data), sep="\t", dtype=str, keep_default_na=False)
    assert "OUTPUT_MANIFEST_FINAL.tsv" not in set(frame.relative_path)
    assert set(frame.include_in_review_bundle) == {"True", "False"}
    assert set(frame.include_in_models_bundle) <= {"True", "False"}


def _write_execution_ledger_fixture(
    run_dir: Path, *, unterminated_fit: bool, tamper: str | None = None,
) -> None:
    if unterminated_fit:
        attempt_id = "f" * 32
        full._journal_event(
            run_dir, "TRAINING_ATTEMPT_STARTED", phase="TRAINING",
            variant_id=full.TRAINED_VARIANTS[0], attempt_id=attempt_id,
        )
        full._journal_event(
            run_dir, "SCIENTIFIC_FIT_CALLED", phase="TRAINING",
            variant_id=full.TRAINED_VARIANTS[0], attempt_id=attempt_id,
            seed=full.PRIMARY_MODEL_SEED, stability_fit=False,
        )
    for index, variant in enumerate(full.TRAINED_VARIANTS, start=1):
        attempt_id = f"{index:032x}"
        full._journal_event(
            run_dir, "TRAINING_ATTEMPT_STARTED", phase="TRAINING",
            variant_id=variant, attempt_id=attempt_id,
        )
        full._journal_event(
            run_dir, "SCIENTIFIC_FIT_CALLED", phase="TRAINING",
            variant_id=variant, attempt_id=attempt_id,
            seed=(7 if tamper == "seed" and index == 1 else full.PRIMARY_MODEL_SEED),
            stability_fit=(tamper == "stability" and index == 1),
        )
        completion = run_dir / full._training_completion_relative(variant)
        completion.parent.mkdir(parents=True, exist_ok=True)
        completion.write_text("{}\n")
        full._journal_event(
            run_dir, "TRAINING_COMPLETION_COMMITTED", phase="TRAINING",
            variant_id=variant, attempt_id=attempt_id,
            completion_sha256=full.sha256_file(completion),
        )
    full._journal_event(run_dir, "MODELS_FROZEN", phase="FULL_MODELS_FROZEN")
    full._journal_event(run_dir, "EXTERNAL_AUDIT_OPENED", phase="EXTERNAL_AUDIT")
    if tamper == "post_audit_fit":
        attempt_id = "d" * 32
        full._journal_event(
            run_dir, "TRAINING_ATTEMPT_STARTED", phase="TRAINING",
            variant_id=full.TRAINED_VARIANTS[0], attempt_id=attempt_id,
        )
        full._journal_event(
            run_dir, "SCIENTIFIC_FIT_CALLED", phase="TRAINING",
            variant_id=full.TRAINED_VARIANTS[0], attempt_id=attempt_id,
            seed=full.PRIMARY_MODEL_SEED, stability_fit=False,
        )
        full._journal_event(
            run_dir, "TRAINING_ATTEMPT_FAILED", phase="TRAINING",
            variant_id=full.TRAINED_VARIANTS[0], attempt_id=attempt_id,
        )
    if tamper == "bootstrap_order":
        full._journal_event(
            run_dir, "BOOTSTRAP_RESAMPLING", phase="REPORTING", status="PASS",
            replicates=full.BOOTSTRAP_REPLICATES, seed=full.BOOTSTRAP_SEED,
            model_retraining=False, bootstrap_sha256="0" * 64,
        )
    for index, variant in enumerate(full.TRAINED_VARIANTS, start=1):
        attempt_id = f"{index + 100:032x}"
        full._journal_event(
            run_dir, "AUDIT_SCORING_ATTEMPT_STARTED", phase="AUDIT_SCORING",
            variant_id=variant, attempt_id=attempt_id,
        )
        completion = run_dir / full._scoring_completion_relative(variant)
        completion.parent.mkdir(parents=True, exist_ok=True)
        completion.write_text("{}\n")
        full._journal_event(
            run_dir, "AUDIT_SCORING_COMPLETION_COMMITTED",
            phase="AUDIT_SCORING", variant_id=variant, attempt_id=attempt_id,
            completion_sha256=full.sha256_file(completion),
        )
    if tamper != "bootstrap_order":
        full._journal_event(
            run_dir, "BOOTSTRAP_RESAMPLING", phase="REPORTING", status="PASS",
            replicates=full.BOOTSTRAP_REPLICATES, seed=full.BOOTSTRAP_SEED,
            model_retraining=False, bootstrap_sha256="0" * 64,
        )
    if tamper == "threshold":
        full._journal_event(
            run_dir, "THRESHOLD_TUNING_CALLED", phase="REPORTING",
        )
    reporting_completion = run_dir / "provenance/reporting_completion_manifest.json"
    reporting_completion.parent.mkdir(parents=True, exist_ok=True)
    reporting_completion.write_text("{}\n")
    full._journal_event(
        run_dir, "REPORTING_COMPLETION_COMMITTED", phase="REPORTING",
        variant_id="GLOBAL", attempt_id="e" * 32,
        completion_sha256=full.sha256_file(reporting_completion),
    )


def test_execution_ledger_accounts_for_every_fit_attempt_and_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", tmp_path / "runtime")
    valid = tmp_path / f"run_valid{full.RUN_SUFFIX}"
    valid.mkdir()
    _write_execution_ledger_fixture(valid, unterminated_fit=False)
    result = full.build_execution_ledger(valid)
    assert result["scientific_fit_attempts"] == 10
    assert result["completed_trained_models"] == 10
    assert result["post_audit_training_calls"] == 0
    assert result["bootstrap_resampling_events"] == 1

    invalid = tmp_path / f"run_invalid{full.RUN_SUFFIX}"
    invalid.mkdir()
    _write_execution_ledger_fixture(invalid, unterminated_fit=True)
    with pytest.raises(full.Amendment06IntegrityError, match="lifecycle differs"):
        full.build_execution_ledger(invalid)


@pytest.mark.parametrize(
    "tamper,match",
    [
        ("seed", "fit authorization"),
        ("stability", "fit authorization"),
        ("post_audit_fit", "fit authorization"),
        ("threshold", "forbidden scientific calls"),
        ("bootstrap_order", "bootstrap binding/order"),
    ],
)
def test_execution_ledger_rejects_forbidden_calls_and_phase_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str, match: str,
) -> None:
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", tmp_path / "runtime")
    run_dir = tmp_path / f"run_{tamper}{full.RUN_SUFFIX}"
    run_dir.mkdir()
    _write_execution_ledger_fixture(
        run_dir, unterminated_fit=False, tamper=tamper,
    )
    with pytest.raises(full.Amendment06IntegrityError, match=match):
        full.build_execution_ledger(run_dir)


def test_output_manifest_detects_every_post_seal_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    status = run_dir / "RUN_STATUS.txt"
    status.write_text("RUN_STATE=FULL_SCIENTIFIC_COMPLETE\n")
    monkeypatch.setattr(
        full, "expected_scientific_run_files", lambda: frozenset({"RUN_STATUS.txt"}),
    )
    full.publish_bytes_no_clobber(
        run_dir / "OUTPUT_MANIFEST_FINAL.tsv",
        full.output_manifest_bytes(run_dir, require_exact_contract=True),
    )
    assert full.validate_output_manifest(run_dir)["status"] == "PASS"
    status.write_text("RUN_STATE=TAMPERED\n")
    with pytest.raises(full.Amendment06IntegrityError, match="manifest differs"):
        full.validate_output_manifest(run_dir)
    status.write_text("RUN_STATE=FULL_SCIENTIFIC_COMPLETE\n")
    (run_dir / "late.txt").write_text("forbidden\n")
    with pytest.raises(full.Amendment06IntegrityError, match="member set differs"):
        full.validate_output_manifest(run_dir)


def _report_prediction_fixture(variant: str, shift: float) -> pd.DataFrame:
    rows: list[dict] = []
    for card_index in range(10):
        for offset, (label, base_probability) in enumerate(
            ((0, 0.12), (1, 0.88), (0, 0.28), (1, 0.72))
        ):
            probability = float(base_probability + shift + 0.001 * card_index)
            rows.append({
                "stable_candidate_id": f"card-{card_index:02d}-row-{offset}",
                "card_id": f"IMG_{9500 + card_index}",
                "true_label": label,
                "probability": probability,
                "prediction_at_0_5": int(probability >= 0.5),
                "review_weight": float(0.7 + 0.1 * (offset % 2)),
                "variant_id": variant,
                "split_population": "external_audit:locked_10_card",
                "row_order": len(rows),
            })
    return pd.DataFrame(rows, columns=full.PREDICTION_COLUMNS)


def _write_complete_reporting_fixture(run_dir: Path) -> dict:
    predictions: dict[str, pd.DataFrame] = {}
    metrics_by_variant: dict[str, dict] = {}
    per_card_by_variant: dict[str, pd.DataFrame] = {}
    timing_rows: list[dict] = []
    parity_rows: list[dict] = []
    for index, variant in enumerate(full.TRAINED_VARIANTS):
        prediction = _report_prediction_fixture(variant, -0.001 * index)
        predictions[variant] = prediction
        prediction_path = run_dir / f"predictions/{variant}_external_audit.csv.gz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_bytes(full.deterministic_csv_gzip_bytes(prediction))
        per_card = full.reporting.per_card_metrics(prediction)
        per_card_path = run_dir / f"metrics/{variant}_per_card.csv"
        per_card_path.parent.mkdir(parents=True, exist_ok=True)
        per_card_path.write_bytes(full.exact_table_csv_bytes(per_card)[0])
        per_card_by_variant[variant] = per_card
        raw = full.reporting.binary_metrics(prediction.true_label, prediction.probability)
        weighted = full.reporting.binary_metrics(
            prediction.true_label, prediction.probability,
            weights=prediction.review_weight,
        )
        metrics_by_variant[variant] = {
            "feature_count": 93 - index,
            "best_iteration": 20 + index,
            "external_raw": raw,
            "external_weighted_sensitivity": weighted,
        }
        classification = (
            "EXPLORATORY" if variant == full.EXPLORATORY_VARIANT
            else "OFFICIAL_RUNNABLE"
        )
        full.atomic_write_strict_json(
            run_dir / full._training_completion_relative(variant),
            {
                "feature_count": 93 - index, "best_iteration": 20 + index,
                "variant_classification": classification,
                "feature_list_sha256": f"{index + 1:064x}",
                "frozen_population_sha256": "a" * 64,
                "four_way_parity_max_abs_diff": 0.0,
            },
        )
        full.atomic_write_strict_json(
            run_dir / full._scoring_completion_relative(variant),
            {
                "raw_candidate_metrics": raw,
                "review_weighted_metrics": weighted,
                "raw_booster_parity_max_abs_diff": 0.0,
            },
        )
        timing = {"variant": variant, "wall_seconds": float(index + 1)}
        full.atomic_write_strict_json(run_dir / f"timing/{variant}.json", timing)
        timing_rows.append(timing)
        parity_rows.append({
            "variant": variant, "classification": classification,
            "feature_count": 93 - index,
            "feature_list_sha256": f"{index + 1:064x}",
            "population": (
                "post_jitter_pre_smote" if variant == "no_safe_smote"
                else "post_safe_smote"
            ),
            "frozen_population_sha256": "a" * 64,
            "four_way_parity_max_abs_diff": 0.0,
            "external_raw_booster_parity_max_abs_diff": 0.0,
            "projection_only": True,
        })
    cards = sorted(predictions[full.TRAINED_VARIANTS[0]].card_id.unique())
    bootstrap = full.reporting.paired_card_bootstrap(
        predictions, cards, replicates=full.BOOTSTRAP_REPLICATES,
        seed=full.BOOTSTRAP_SEED,
    )
    full.atomic_write_strict_json(
        run_dir / "statistics/paired_card_bootstrap.json", bootstrap,
    )
    r92 = {"classification": "HISTORICAL_CONTROL", "status": "PASS"}
    full.atomic_write_strict_json(run_dir / "metrics/r92_historical_control.json", r92)
    stage_counts = {
        stage: {"negative": negative, "positive": positive, "rows": rows}
        for stage, (rows, negative, positive) in full.EXPECTED_STAGE_COUNTS.items()
    }
    full.atomic_write_strict_json(
        run_dir / "provenance/frozen_full_resampling_manifest.json",
        {"stage_counts": stage_counts},
    )
    report = full.reporting.build_all_reports(
        run_dir,
        metrics_by_variant=metrics_by_variant,
        per_card_by_variant=per_card_by_variant,
        bootstrap=bootstrap,
        stage_table=pd.DataFrame([
            {"stage": stage, **values} for stage, values in stage_counts.items()
        ]),
        timing_table=pd.DataFrame(timing_rows),
        parity_table=pd.DataFrame(parity_rows),
        r92_control=r92,
    )
    records = [
        full._publication_record(run_dir / relative, relative)
        for relative in sorted(full.REPORTING_REQUIRED_RELATIVES)
    ]
    completion = {
        "run_id": run_dir.name, "run_kind": full.RUN_KIND,
        "phase": "SCIENTIFIC_REPORTING_COMPLETE", "status": "PASS",
        "bootstrap_replicates": full.BOOTSTRAP_REPLICATES,
        "bootstrap_seed": full.BOOTSTRAP_SEED, "model_retraining": False,
        "report_csv_roundtrip": report["csv_roundtrip"],
        "report_artifact_count": 25, "artifact_records": records,
    }
    full.atomic_write_strict_json(
        run_dir / "provenance/reporting_completion_manifest.json", completion,
    )
    return completion


def test_reporting_resume_recomputes_bootstrap_and_rejects_self_consistent_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", tmp_path / "runtime")
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    run_dir.mkdir()
    completion = _write_complete_reporting_fixture(run_dir)
    assert full._validate_reporting_completion(run_dir)["status"] == "PASS"

    bootstrap_path = run_dir / "statistics/paired_card_bootstrap.json"
    bootstrap = full.strict_full_load_file(bootstrap_path)
    numeric_key = next(
        key for key, value in bootstrap["rows"][0].items()
        if isinstance(value, float)
    )
    bootstrap["rows"][0][numeric_key] += 0.001
    full.atomic_write_strict_json(bootstrap_path, bootstrap)
    for record in completion["artifact_records"]:
        if record["relative_path"] == "statistics/paired_card_bootstrap.json":
            record.update({
                "size_bytes": bootstrap_path.stat().st_size,
                "sha256": full.sha256_file(bootstrap_path),
            })
    full.atomic_write_strict_json(
        run_dir / "provenance/reporting_completion_manifest.json", completion,
    )
    with pytest.raises(full.Amendment06IntegrityError, match="exact recomputation"):
        full._validate_reporting_completion(run_dir)


def _valid_amendment07_gpu_identity() -> dict:
    return {
        "devices": [{
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "name": "NVIDIA GeForce RTX 4090",
        }],
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "name": "NVIDIA GeForce RTX 4090",
        "visible_gpu_count_reported_by_nvidia_smi": 1,
    }


def test_amendment07_gpu_name_requires_one_nested_exact_identity() -> None:
    identity = _valid_amendment07_gpu_identity()
    assert full._gpu_name(identity) == "NVIDIA GeForce RTX 4090"
    identity["devices"][0]["gpu_name"] = "  NVIDIA GeForce RTX 4090  "
    with pytest.raises(full.Amendment06IntegrityError, match="not canonical"):
        full._gpu_name(identity)


def test_amendment08_gpu_producer_shape_is_consumed_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"nvidia_smi": 0, "build_info": 0}

    def nvidia_smi(command, *, cwd):
        calls["nvidia_smi"] += 1
        assert command == [
            "nvidia-smi", "--query-gpu=index,name,driver_version",
            "--format=csv,noheader,nounits",
        ]
        assert cwd == full.core.STUDY_ROOT
        return SimpleNamespace(
            returncode=0,
            stdout="0, NVIDIA GeForce RTX 4090, 580.97\n",
            stderr="",
        )

    def build_info() -> dict[str, bool]:
        calls["build_info"] += 1
        return {"USE_CUDA": True}

    monkeypatch.setattr(full.core, "run_command", nvidia_smi)
    monkeypatch.setattr(full.core.xgb, "build_info", build_info)
    identity = full.core.gpu_identity_metadata()
    assert identity["devices"] == [{
        "physical_index": "0",
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "driver_version": "580.97",
    }]
    assert identity["visible_gpu_count_reported_by_nvidia_smi"] == 1
    assert full._gpu_name(identity) == "NVIDIA GeForce RTX 4090"
    assert calls == {"nvidia_smi": 1, "build_info": 1}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"devices": []}),
        lambda value: value.update({"devices": "NVIDIA GeForce RTX 4090"}),
        lambda value: value.update({"devices": [{"gpu_name": None}]}),
        lambda value: value["devices"][0].update({"gpu_name": "   "}),
        lambda value: value["devices"][0].update({"name": " NVIDIA GeForce RTX 4090 "}),
        lambda value: value.update({"gpu_name": " NVIDIA GeForce RTX 4090 "}),
        lambda value: value.update({"name": "NVIDIA GeForce RTX 4080"}),
        lambda value: value.update({"visible_gpu_count_reported_by_nvidia_smi": True}),
        lambda value: value.update({"visible_gpu_count_reported_by_nvidia_smi": 2}),
    ],
)
def test_amendment07_gpu_name_rejects_malformed_or_normalized_duplicates(
    mutate,
) -> None:
    identity = _valid_amendment07_gpu_identity()
    mutate(identity)
    with pytest.raises(full.Amendment06IntegrityError, match="GPU identity"):
        full._gpu_name(identity)


@pytest.mark.parametrize(
    "identity",
    [
        None,
        [],
        {"devices": ["NVIDIA GeForce RTX 4090"]},
        {"devices": [{"gpu_name": 4090}]},
        {
            "devices": [
                {"gpu_name": "NVIDIA GeForce RTX 4090"},
                {"gpu_name": "NVIDIA GeForce RTX 4090"},
            ]
        },
    ],
)
def test_amendment08_gpu_name_rejects_nonmapping_and_noncanonical_device_shapes(
    identity,
) -> None:
    with pytest.raises(full.Amendment06IntegrityError, match="GPU identity"):
        full._gpu_name(identity)


def test_amendment07_malformed_gpu_identity_stops_before_attempt_or_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", tmp_path / "runtime")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls = {"producer": 0}

    def malformed_identity() -> dict:
        calls["producer"] += 1
        return {"gpu_name": "NVIDIA GeForce RTX 4090"}

    monkeypatch.setattr(full.core, "gpu_identity_metadata", malformed_identity)
    monkeypatch.setattr(full, "_record_interrupted_attempts", lambda **kwargs: None)
    monkeypatch.setattr(full, "_recover_unpublished_paths", lambda **kwargs: None)
    monkeypatch.setattr(
        full, "_attempt_root",
        lambda *args, **kwargs: pytest.fail("attempt staging created before GPU parsing"),
    )
    monkeypatch.setattr(
        full.xgb, "XGBClassifier",
        lambda *args, **kwargs: pytest.fail("classifier created before GPU parsing"),
    )
    with pytest.raises(full.Amendment06IntegrityError, match="devices"):
        full._train_one_variant_impl(
            run_dir=run_dir, locked=SimpleNamespace(), cache=SimpleNamespace(),
            development=SimpleNamespace(), variant="full_new_reference",
        )
    assert calls == {"producer": 1}
    assert full._read_journal(run_dir) == []


def test_amendment08_gpu_producer_and_parser_run_once_before_attempt_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "isolated_run"
    run_dir.mkdir()
    order: list[str] = []
    identity = _valid_amendment07_gpu_identity()
    parse_gpu_name = full._gpu_name

    def produce() -> dict:
        order.append("producer")
        return identity

    def parse(value: dict) -> str:
        order.append("parser")
        return parse_gpu_name(value)

    def journal(path: Path, event: str, **fields) -> dict:
        order.append(event)
        return {"event": event, **fields}

    def project(*args, **kwargs):
        order.append("projection")
        return SimpleNamespace()

    def classifier(**kwargs):
        order.append("classifier")
        raise RuntimeError("stop before fit")

    monkeypatch.setattr(full.core, "gpu_identity_metadata", produce)
    monkeypatch.setattr(full, "_gpu_name", parse)
    monkeypatch.setattr(full, "_record_interrupted_attempts", lambda **kwargs: None)
    monkeypatch.setattr(full, "_recover_unpublished_paths", lambda **kwargs: None)
    monkeypatch.setattr(full, "_journal_event", journal)
    monkeypatch.setattr(full, "_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(full, "project_variant", project)
    monkeypatch.setattr(full.xgb, "XGBClassifier", classifier)
    monkeypatch.setattr(
        full, "_attempt_root", lambda *args, **kwargs: tmp_path / "attempt",
    )
    with pytest.raises(RuntimeError, match="before fit"):
        full._train_one_variant_impl(
            run_dir=run_dir,
            locked=SimpleNamespace(model_parameters={}),
            cache=SimpleNamespace(),
            development=SimpleNamespace(),
            variant="full_new_reference",
        )
    assert order == [
        "producer", "parser", "TRAINING_ATTEMPT_STARTED", "projection", "classifier",
    ]


def test_amendment08_training_reuses_the_one_parsed_gpu_name_in_all_outputs() -> None:
    tree = ast.parse(Path(full.__file__).read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_train_one_variant_impl"
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    producer_calls = [
        node for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "core"
        and node.func.attr == "gpu_identity_metadata"
    ]
    parser_calls = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "_gpu_name"
    ]
    metadata_call = next(
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "_model_common_metadata"
    )
    assert len(producer_calls) == 1
    assert len(parser_calls) == 1
    assert any(
        keyword.arg == "gpu_name"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "gpu_name"
        for keyword in metadata_call.keywords
    )
    assert any(
        isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant) and key.value == "gpu_name"
            and isinstance(value, ast.Name) and value.id == "gpu_name"
            for key, value in zip(node.keys, node.values, strict=True)
        )
        for node in ast.walk(function)
    )


def test_amendment07_forensic_bundle_reopens_exact_pinned_inventories() -> None:
    evidence = full._validate_amendment07_forensic_snapshot()
    assert evidence == {
        "status": "PASS",
        "forensic_zip_sha256": full.A07_FORENSIC_ZIP_SHA256,
        "forensic_verification_sha256": full.A07_FORENSIC_VERIFICATION_SHA256,
        "quarantine_receipt_sha256": full.A07_FAILED_ATTEMPT_QUARANTINE_RECEIPT_SHA256,
        "original_prerun_forensic_sha256": full.A07_ORIGINAL_PRERUN_SHA256,
    }


def test_amendment07_cuda_probe_validator_uses_strict_producer_consumer_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.json"
    payload = {
        "schema": "amendment07_cuda_producer_consumer_probe/v1",
        "status": "PASS", "authorization": full.A07_AUTHORIZATION,
        "prompt_sha256": full.A07_PROMPT_SHA256,
        "target_run_id": full.A07_TARGET_RUN_ID,
        "recorded_at": "2026-08-20T00:00:00+00:00",
        "xgboost_version": "2.1.1", "scikit_learn_version": "1.7.2",
        "use_cuda": True, "gpu_identity": _valid_amendment07_gpu_identity(),
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "producer_consumer_status": "PASS",
        "fit_save_reload_predict_status": "PASS",
        "four_way_bit_exact_max_abs_diff": 0.0,
        "model_bytes_unchanged": True, "cpu_fallback_detected": False,
        "fallback_warnings": [],
        "active_reloaded_cuda_probe": {
            "status": "PASS", "device": "cuda:0", "warnings": [],
        },
    }
    full.atomic_write_strict_json(path, payload)
    assert full.validate_amendment07_cuda_producer_consumer_probe(path) == payload
    payload["gpu_identity"]["gpu_name"] = " NVIDIA GeForce RTX 4090 "
    full.atomic_write_strict_json(path, payload)
    with pytest.raises(full.Amendment06IntegrityError, match="conflicts"):
        full.validate_amendment07_cuda_producer_consumer_probe(path)


def test_amendment07_model_metadata_binds_base_effective_and_corrigendum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    (run_dir / "provenance").mkdir(parents=True)
    (run_dir / "provenance/source_input_hashes_pre.tsv").write_bytes(b"source\n")
    model_path = tmp_path / "model.ubj"
    model_path.write_bytes(b"model")
    lineage = _amendment07_overlay_stub()
    monkeypatch.setattr(
        full, "_scientific_lineage_bindings", lambda path: dict(lineage),
    )
    metadata = full._model_common_metadata(
        run_dir=run_dir,
        variant_data=SimpleNamespace(
            variant="full_new_reference", classification="OFFICIAL_RUNNABLE",
            features=("a", "b"),
        ),
        model_path=model_path,
        cache=SimpleNamespace(binding_sha256="b" * 64, imputer_sha256="i" * 64),
        gpu_identity=_valid_amendment07_gpu_identity(),
        gpu_name="NVIDIA GeForce RTX 4090", best_iteration=7,
        iteration_range=(0, 8),
    )
    assert all(metadata[key] == value for key, value in lineage.items())
    assert metadata["scientific_execution_code_manifest_sha256"] == lineage[
        "effective_scientific_execution_code_manifest_sha256"
    ]


def _amendment07_overlay_stub() -> dict[str, str]:
    return {
        "base_scientific_execution_code_manifest_sha256": (
            full.A07_BASE_SCIENTIFIC_MANIFEST_SHA256
        ),
        "effective_scientific_execution_code_manifest_sha256": "e" * 64,
        "amendment07_code_corrigendum_sha256": "c" * 64,
    }


def _write_amendment07_corrected_prefix(
    run_dir: Path, overlay: dict[str, str], *, start_lineage: dict | None = None,
) -> str:
    for name in ("BASE_0", "BASE_1", "BASE_2", "BASE_3"):
        full._journal_event(run_dir, name)
    full._journal_event(
        run_dir, "TRAINING_ATTEMPT_FAILED", phase="TRAINING",
        variant_id="full_new_reference", attempt_id=full.A07_FAILED_ATTEMPT_ID,
        error_type="Amendment06IntegrityError", error="GPU identity has no name",
    )
    full._journal_event(
        run_dir, "AMENDMENT07_CORRIGENDUM_INSTALLED", status="PASS",
        authorization=full.A07_AUTHORIZATION, prompt_sha256=full.A07_PROMPT_SHA256,
        effective_scientific_execution_code_manifest_sha256=overlay[
            "effective_scientific_execution_code_manifest_sha256"
        ],
        amendment07_code_corrigendum_sha256=overlay[
            "amendment07_code_corrigendum_sha256"
        ],
        cache_manifest_sha256=full.A07_CACHE_MANIFEST_SHA256,
    )
    full._journal_event(
        run_dir, "CACHE_LOAD", status="PASS",
        cache_path=str(full.EXPECTED_RUNTIME_ROOT / "full_population" / run_dir.name),
        cache_manifest_sha256=full.A07_CACHE_MANIFEST_SHA256,
        frozen_population_sha256=full.A07_FROZEN_POPULATION_SHA256,
        unique_realizations=1, augmentation_generation_count=1,
        safe_smote_generation_count=1,
        effective_scientific_execution_code_manifest_sha256=overlay[
            "effective_scientific_execution_code_manifest_sha256"
        ],
        amendment07_code_corrigendum_sha256=overlay[
            "amendment07_code_corrigendum_sha256"
        ],
    )
    attempt_id = "7" * 32
    lineage = {
        "amendment07_authorization": full.A07_AUTHORIZATION,
        "amendment07_prompt_sha256": full.A07_PROMPT_SHA256,
        **overlay,
    }
    if start_lineage is not None:
        lineage = dict(start_lineage)
    full._journal_event(
        run_dir, "TRAINING_ATTEMPT_STARTED", phase="TRAINING",
        variant_id="full_new_reference", attempt_id=attempt_id, **lineage,
    )
    return attempt_id


def test_amendment07_event_5_6_7_and_interrupted_attempt_reconcile_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", tmp_path / "runtime")
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    run_dir.mkdir()
    overlay = _amendment07_overlay_stub()
    attempt_id = _write_amendment07_corrected_prefix(run_dir, overlay)
    full._journal_event(
        run_dir, "SCIENTIFIC_FIT_CALLED", phase="TRAINING",
        variant_id="full_new_reference", attempt_id=attempt_id,
        seed=full.PRIMARY_MODEL_SEED, stability_fit=False,
        amendment07_authorization=full.A07_AUTHORIZATION,
        amendment07_prompt_sha256=full.A07_PROMPT_SHA256, **overlay,
    )
    interrupted = full._validate_amendment07_corrected_training_history(
        run_dir=run_dir, overlay=overlay, allow_unterminated=True,
    )
    assert interrupted["unterminated_attempt_ids"] == [attempt_id]
    with pytest.raises(full.Amendment06IntegrityError, match="terminal/order"):
        full._validate_amendment07_corrected_training_history(
            run_dir=run_dir, overlay=overlay, allow_unterminated=False,
        )
    full._record_interrupted_attempts(
        run_dir=run_dir, phase="TRAINING", variant="full_new_reference",
    )
    recovered = full._validate_amendment07_corrected_training_history(
        run_dir=run_dir, overlay=overlay, allow_unterminated=False,
    )
    assert recovered["unterminated_attempt_ids"] == []
    assert recovered["next_variant"] == "full_new_reference"
    assert full._read_journal(run_dir)[7]["event"] == "TRAINING_ATTEMPT_STARTED"


def test_amendment07_corrected_history_rejects_event7_lineage_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", tmp_path / "runtime")
    run_dir = tmp_path / full.A07_TARGET_RUN_ID
    run_dir.mkdir()
    overlay = _amendment07_overlay_stub()
    bad_lineage = {
        "amendment07_authorization": full.A07_AUTHORIZATION,
        "amendment07_prompt_sha256": full.A07_PROMPT_SHA256,
        **overlay,
        "effective_scientific_execution_code_manifest_sha256": "f" * 64,
    }
    _write_amendment07_corrected_prefix(
        run_dir, overlay, start_lineage=bad_lineage,
    )
    with pytest.raises(full.Amendment06IntegrityError, match="start differs"):
        full._validate_amendment07_corrected_training_history(
            run_dir=run_dir, overlay=overlay, allow_unterminated=True,
        )


def test_amendment07_corrected_history_rejects_orphan_or_malformed_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", tmp_path / "runtime")
    overlay = _amendment07_overlay_stub()
    for malformed in ("orphan", "wrong_variant", "extra_claim"):
        run_dir = tmp_path / malformed
        run_dir.mkdir()
        attempt_id = _write_amendment07_corrected_prefix(run_dir, overlay)
        fields = {
            "phase": "TRAINING", "variant_id": "full_new_reference",
            "attempt_id": attempt_id, "failure_phase": (
                "POST_START_THROUGH_COMPLETION_COMMIT"
            ),
            "error_type": "RuntimeError", "error": "synthetic operational failure",
        }
        if malformed == "orphan":
            fields["attempt_id"] = "9" * 32
        elif malformed == "wrong_variant":
            fields["variant_id"] = "no_pca"
        else:
            fields["unbound_claim"] = True
        full._journal_event(run_dir, "TRAINING_ATTEMPT_FAILED", **fields)
        with pytest.raises(
            full.Amendment06IntegrityError, match="failed-attempt lifecycle differs",
        ):
            full._validate_amendment07_corrected_training_history(
                run_dir=run_dir, overlay=overlay, allow_unterminated=False,
            )


def test_execution_ledger_invokes_strict_amendment07_final_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", tmp_path / "runtime")
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    run_dir.mkdir()
    completion = run_dir / "provenance/reporting_completion_manifest.json"
    completion.parent.mkdir(parents=True)
    completion.write_bytes(b"{}\n")
    full._journal_event(
        run_dir, "REPORTING_COMPLETION_COMMITTED", phase="REPORTING",
        variant_id="GLOBAL", attempt_id="a" * 32,
        completion_sha256=full.sha256_file(completion),
    )
    overlay = _amendment07_overlay_stub()
    monkeypatch.setattr(full, "A07_TARGET_RUN", run_dir)
    monkeypatch.setattr(full, "_scientific_lineage_bindings", lambda path: overlay)
    monkeypatch.setattr(full, "validate_amendment07_overlay", lambda path: overlay)

    def reject_final_history(*, run_dir, overlay, allow_unterminated):
        assert allow_unterminated is False
        raise full.Amendment06IntegrityError("strict final history sentinel")

    monkeypatch.setattr(
        full, "_validate_amendment07_corrected_training_history",
        reject_final_history,
    )
    with pytest.raises(full.Amendment06IntegrityError, match="strict final history sentinel"):
        full.build_execution_ledger(run_dir)


def test_amendment07_overlay_mismatch_quarantine_is_permanent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "a07"
    monkeypatch.setattr(full, "A07_RUNTIME", runtime)
    run_dir = tmp_path / "run"
    path = run_dir / "provenance/member.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong")
    with pytest.raises(full.Amendment06IntegrityError, match="quarantined"):
        full._publish_or_validate_amendment07_member(
            run_dir=run_dir, relative="provenance/member.json", data=b"right",
            transaction_id="a" * 32,
        )
    assert not path.exists()
    with pytest.raises(full.Amendment06IntegrityError, match="permanently blocks"):
        full._assert_no_amendment07_overlay_mismatch()


def _amendment07_final_checks(
    runtime: Path,
) -> list[dict]:
    facts = {
        "syntax_compile": {"all_changed_python_files_compiled": True},
        "targeted_tests": {"amendment06_targeted_tests_passed": True},
        "full_suite_exactly_once": {
            "full_project_suite_passed": True, "suite_invocation_count": 1,
        },
        "active_cuda_probe": {
            "xgboost_version": "2.1.1", "scikit_learn_version": "1.7.2",
            "use_cuda": True, "gpu_name": "NVIDIA GeForce RTX 4090",
            "cpu_fallback_detected": False,
            "fit_save_reload_predict_status": "PASS",
        },
        "package_synthetic_rehearsal": {
            "synthetic_full_fixture": True, "production_packager_used": True,
            "production_final_validator_used": True,
            "package_retry_without_training": True, "scientific_api_calls": 0,
        },
        "resume_rehearsal": {
            "same_run_id_preserved": True, "completed_variant_skipped": True,
            "unpublished_incomplete_variant_retried": True,
            "completed_variant_retrained": False, "distinct_full_run_ids": 1,
        },
    }
    records: list[dict] = []
    for category in full.PRERUN_CHECK_CATEGORIES:
        path = runtime / "pre_run" / f"{category}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{category}: PASS\n", encoding="utf-8")
        records.append(full.make_test_evidence_record(
            category=category, command=f"run {category}", log_path=path,
            facts=facts[category],
        ))
    return records


def test_amendment07_ledger_idempotence_requires_same_test_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    a06 = tmp_path / "a06"
    a07 = tmp_path / "a07"
    ledger = a06 / "authorized_change_ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(full, "A06_RUNTIME", a06)
    monkeypatch.setattr(full, "A07_RUNTIME", a07)
    monkeypatch.setattr(full, "AUTHORIZED_CHANGE_LEDGER", ledger)
    checks = _amendment07_final_checks(a06 / "corrigendum07_gpu_identity")
    event = {
        "corrigendum_authorization": full.A07_AUTHORIZATION,
        "test_evidence": checks,
    }
    monkeypatch.setattr(full, "validate_amendment07_precorrection", lambda: {})
    monkeypatch.setattr(full, "_load_concatenated_json", lambda path: [event])
    monkeypatch.setattr(full, "_validate_amendment07_ledger_event", lambda *args, **kwargs: None)
    assert full.append_amendment07_authorized_edit_event(
        test_evidence=checks,
    ) == event
    changed = copy.deepcopy(checks)
    changed[0]["command"] = "different command"
    with pytest.raises(full.Amendment06IntegrityError, match="different test evidence"):
        full.append_amendment07_authorized_edit_event(test_evidence=changed)


def test_amendment07_final_evidence_must_equal_latest_ledger_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    a06 = tmp_path / "a06"
    bound = a06 / "corrigendum07_gpu_identity"
    monkeypatch.setattr(full, "A06_RUNTIME", a06)
    monkeypatch.setattr(full, "A07_A06_BOUND_RUNTIME", bound)
    checks = _amendment07_final_checks(bound)
    event = {
        "corrigendum_authorization": full.A07_AUTHORIZATION,
        "test_evidence": checks,
    }
    monkeypatch.setattr(full, "validate_authorized_change_chain", lambda: {})
    monkeypatch.setattr(full, "_load_concatenated_json", lambda path: [event])
    monkeypatch.setattr(full, "_validate_amendment07_ledger_event", lambda *args, **kwargs: None)
    changed = copy.deepcopy(checks)
    changed[-1]["command"] = "different resume rehearsal"
    with pytest.raises(full.Amendment06IntegrityError, match="differ from the latest ledger"):
        full.replace_amendment07_final_prerun_evidence(checks=changed)


def test_amendment07_effective_manifest_changes_exactly_two_scientific_paths() -> None:
    base = full._validate_amendment07_base_history(full.A07_TARGET_RUN)
    effective = full._parse_code_manifest_bytes(
        full._effective_amendment07_manifest_bytes(full.A07_TARGET_RUN),
        label="test effective Amendment 07 manifest",
    )
    changed = [
        str(after["relative_path"])
        for before, after in zip(base, effective, strict=True)
        if (before["size_bytes"], before["sha256"])
        != (after["size_bytes"], after["sha256"])
    ]
    assert changed == list(full.A07_CHANGED_SCIENTIFIC_RELATIVES)


def test_amendment07_overlay_validation_rechecks_raw_journal_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / full.A07_TARGET_RUN_ID
    run_dir.mkdir()
    monkeypatch.setattr(full, "A07_TARGET_RUN", run_dir)
    monkeypatch.setattr(full, "_assert_no_amendment07_overlay_mismatch", lambda: None)
    monkeypatch.setattr(full, "_validate_amendment07_base_history", lambda path: [])
    monkeypatch.setattr(full, "_validate_amendment07_ledger_prefix", lambda: {})

    def reject_raw_prefix(path):
        assert path == run_dir.resolve()
        raise full.Amendment06IntegrityError("raw Journal prefix sentinel")

    monkeypatch.setattr(full, "_validate_amendment07_journal_prefix", reject_raw_prefix)
    with pytest.raises(full.Amendment06IntegrityError, match="raw Journal prefix sentinel"):
        full.validate_amendment07_overlay(run_dir)


def _write_metric_recomputation_receipt_fixture(
    run_dir: Path, lineage: dict[str, str],
) -> dict:
    reporting_path = run_dir / "provenance/reporting_completion_manifest.json"
    bootstrap_path = run_dir / "statistics/paired_card_bootstrap.json"
    reporting_path.parent.mkdir(parents=True)
    bootstrap_path.parent.mkdir(parents=True)
    reporting_path.write_bytes(b"reporting-completion\n")
    bootstrap_path.write_bytes(b"paired-bootstrap\n")
    records = []
    for variant in full.TRAINED_VARIANTS:
        paths = {
            "training_completion_sha256": (
                run_dir / full._training_completion_relative(variant)
            ),
            "external_scoring_completion_sha256": (
                run_dir / full._scoring_completion_relative(variant)
            ),
            "validation_prediction_sha256": (
                run_dir / f"predictions/{variant}_validation.csv.gz"
            ),
            "external_prediction_sha256": (
                run_dir / f"predictions/{variant}_external_audit.csv.gz"
            ),
            "metrics_sha256": run_dir / f"metrics/{variant}.json",
            "per_card_metrics_sha256": (
                run_dir / f"metrics/{variant}_per_card.csv"
            ),
        }
        for field, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{variant}:{field}\n".encode("ascii"))
        validation_sha = full.sha256_file(paths["validation_prediction_sha256"])
        external_sha = full.sha256_file(paths["external_prediction_sha256"])
        evidence_common = {
            "status": "PASS", "float_precision": "round_trip", "engine": "c",
            "numeric_equality": "EXACT", "metric_recomputation": "PASS",
        }
        records.append({
            "variant_id": variant,
            **{field: full.sha256_file(path) for field, path in paths.items()},
            "validation_metric_recomputation": {
                **evidence_common, "rows": 56_843,
                "csv_sha256": validation_sha,
            },
            "external_metric_recomputation": {
                **evidence_common, "rows": 10_097,
                "csv_sha256": external_sha,
            },
            "raw_metrics_exact": True, "weighted_metrics_exact": True,
        })
    payload = {
        "schema": "amendment06_metric_recomputation/v2",
        "run_id": run_dir.name, "run_kind": full.RUN_KIND,
        "phase": "SCIENTIFIC_METRIC_RECOMPUTATION_COMPLETE", "status": "PASS",
        **lineage,
        "reporting_completion_sha256": full.sha256_file(reporting_path),
        "paired_card_bootstrap_sha256": full.sha256_file(bootstrap_path),
        "external_audit_sha256": full.EXTERNAL_AUDIT_SHA256,
        "external_audit_rows": 10_097,
        "performed_before_full_scientific_complete": True,
        "package_metric_recomputation_required": False,
        "model_retraining": False, "required_metrics_finite": True,
        "numeric_equality": "EXACT_NO_TOLERANCE",
        "csv_engine": "c", "float_precision": "round_trip",
        "variant_count": 10, "records": records,
    }
    full.atomic_write_strict_json(
        run_dir / "provenance/metric_recomputation.json", payload,
    )
    return payload


def test_metric_recomputation_v2_receipt_is_exact_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    run_dir.mkdir()
    lineage = _amendment07_overlay_stub()
    payload = _write_metric_recomputation_receipt_fixture(run_dir, lineage)
    monkeypatch.setattr(full, "_scientific_lineage_bindings", lambda path: lineage)
    monkeypatch.setattr(
        full, "_validate_reporting_completion_receipt", lambda path: {"status": "PASS"},
    )
    assert full._validate_metric_recomputation_receipt(run_dir) == payload

    metric_path = run_dir / f"metrics/{full.TRAINED_VARIANTS[0]}.json"
    metric_path.write_bytes(b"tampered\n")
    with pytest.raises(full.Amendment06IntegrityError, match="Variant binding differs"):
        full._validate_metric_recomputation_receipt(run_dir)


def test_metric_recomputation_v2_receipt_rejects_extra_nested_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    run_dir.mkdir()
    lineage = _amendment07_overlay_stub()
    payload = _write_metric_recomputation_receipt_fixture(run_dir, lineage)
    payload["records"][0]["validation_metric_recomputation"]["unbound_claim"] = True
    full.atomic_write_strict_json(
        run_dir / "provenance/metric_recomputation.json", payload,
    )
    monkeypatch.setattr(full, "_scientific_lineage_bindings", lambda path: lineage)
    with pytest.raises(full.Amendment06IntegrityError, match="round-trip receipt differs"):
        full._validate_metric_recomputation_receipt(run_dir)


def test_completed_scientific_validation_uses_receipts_without_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    (run_dir / "provenance").mkdir(parents=True)
    (run_dir / "RUN_STATUS.txt").write_text(
        "RUN_STATE=FULL_SCIENTIFIC_COMPLETE\n", encoding="ascii",
    )
    (run_dir / "provenance/strict_json_validation.json").write_text(
        "{}\n", encoding="ascii",
    )
    calls: list[str] = []
    monkeypatch.setattr(
        full, "_current_run_state", lambda path: "FULL_SCIENTIFIC_COMPLETE",
    )
    monkeypatch.setattr(
        full, "_validate_reporting_completion_receipt",
        lambda path: calls.append("reporting_receipt") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        full, "_validate_metric_recomputation_receipt",
        lambda path: calls.append("metric_receipt") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        full, "validate_output_manifest",
        lambda path: calls.append("manifest") or {"status": "PASS"},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("post-completion scientific recomputation was called")

    monkeypatch.setattr(full, "_validate_reporting_completion", forbidden)
    monkeypatch.setattr(full, "_recompute_and_validate_reporting_outputs", forbidden)
    monkeypatch.setattr(full, "build_metric_recomputation", forbidden)
    monkeypatch.setattr(full, "_binary_metrics", forbidden)
    assert full.validate_final_scientific_run(run_dir) == {"status": "PASS"}
    assert calls == ["reporting_receipt", "metric_receipt", "manifest"]


def test_completed_scientific_finalization_recovery_is_receipt_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / f"run_fixture{full.RUN_SUFFIX}"
    run_dir.mkdir()
    calls: list[str] = []
    monkeypatch.setattr(
        full, "_current_run_state", lambda path: "FULL_SCIENTIFIC_COMPLETE",
    )
    monkeypatch.setattr(
        full, "_validate_reporting_completion_receipt",
        lambda path: calls.append("reporting_receipt") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        full, "_validate_metric_recomputation_receipt",
        lambda path: calls.append("metric_receipt") or {"status": "PASS"},
    )
    monkeypatch.setattr(full, "_snapshot_source_post", lambda path: None)
    monkeypatch.setattr(full, "build_execution_ledger", lambda path: {})
    monkeypatch.setattr(
        full, "validate_strict_json_tree", lambda path, include: {"status": "PASS"},
    )
    monkeypatch.setattr(
        full, "output_manifest_bytes", lambda path, require_exact_contract: b"manifest\n",
    )
    monkeypatch.setattr(
        full, "validate_output_manifest",
        lambda path: {"status": "PASS", "sha256": "a" * 64},
    )
    monkeypatch.setattr(full, "_progress", lambda *args, **kwargs: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("post-completion scientific recomputation was called")

    monkeypatch.setattr(full, "_validate_reporting_completion", forbidden)
    monkeypatch.setattr(full, "_recompute_and_validate_reporting_outputs", forbidden)
    monkeypatch.setattr(full, "build_metric_recomputation", forbidden)
    monkeypatch.setattr(full, "_binary_metrics", forbidden)
    result = full.finalize_scientific_run(
        run_dir=run_dir, audit=SimpleNamespace(frame=None),
    )
    assert result["status"] == "PASS"
    assert calls == ["reporting_receipt", "metric_receipt"]


def _amendment08_isolated_journal_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    isolated_root = tmp_path / "amendment08_fixture"
    runtime = isolated_root / "runtime"
    run_dir = isolated_root / "results" / full.A08_TARGET_RUN_ID
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", runtime)
    return run_dir, runtime


def _write_amendment08_raw_prefix(run_dir: Path) -> list[dict]:
    journal = full._journal_path(run_dir)
    journal.mkdir(parents=True)
    events = [
        copy.deepcopy(event)
        for event in full._amendment08_expected_raw_prefix_events()
    ]
    for index, event in enumerate(events):
        (journal / f"{index:08d}.json").write_bytes(
            full.strict_full_json_bytes(event)
        )
    return events


def _install_amendment08_semantic_view_fixture(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, dict]:
    raw = full._validate_amendment08_raw_prefix(
        run_dir, require_exact_boundary=True,
    )
    source = _amendment08_source_stub()
    package = _amendment08_package_stub()
    adjudication_path = (
        run_dir / "provenance/amendment08_journal_contamination_adjudication.json"
    )
    adjudication = full._amendment08_adjudication_payload(
        recorded_at="2026-08-20T21:00:01-04:00",
        raw=raw,
    )
    full.atomic_write_strict_json(adjudication_path, adjudication)
    adjudication_sha256 = full.sha256_file(adjudication_path)
    lineage = _amendment08_lineage_stub()
    lineage["amendment08_journal_contamination_adjudication_sha256"] = (
        adjudication_sha256
    )
    event7 = full._amendment08_event7_payload(
        recorded_at="2026-08-20T21:00:00-04:00",
        transaction_id="7" * 32,
        effective_manifest_sha256=lineage[
            "effective_scientific_execution_code_manifest_sha256"
        ],
        corrigendum_sha256=lineage["amendment08_code_corrigendum_sha256"],
        adjudication_sha256=lineage[
            "amendment08_journal_contamination_adjudication_sha256"
        ],
        source=source,
        package=package,
    )
    full._append_or_validate_exact_journal_event(run_dir, event7)
    corrigendum_path = run_dir / "provenance/amendment08_code_corrigendum.json"
    corrigendum = {
        "source_freeze_receipt_sha256": source["source_freeze_receipt_sha256"],
        "current_a06_authorized_change_ledger_sha256": source[
            "current_a06_authorized_change_ledger_sha256"
        ],
        "a08_source_ledger_event_index": source["a08_source_ledger_event_index"],
        "a08_source_ledger_event_canonical_sha256": source[
            "a08_source_ledger_event_canonical_sha256"
        ],
        "current_a06_final_prerun_evidence_sha256": source[
            "current_a06_final_prerun_evidence_sha256"
        ],
        "a08_prelive_package_lineage_sha256": package["lineage_sha256"],
        "a08_prelive_package_receipt_sha256": package["receipt_sha256"],
    }
    full.atomic_write_strict_json(corrigendum_path, corrigendum)
    event7_path = full._journal_event_path(run_dir, 7)
    receipt_path = run_dir / "provenance/amendment08_overlay_install_receipt.json"
    receipt = {
        "schema": "amendment08_overlay_install_receipt/v1",
        "status": "PASS",
        "transaction_id": event7["transaction_id"],
        "base_scientific_execution_code_manifest_sha256": (
            full.A07_BASE_SCIENTIFIC_MANIFEST_SHA256
        ),
        "effective_scientific_execution_code_manifest_sha256": lineage[
            "effective_scientific_execution_code_manifest_sha256"
        ],
        "amendment08_code_corrigendum_sha256": lineage[
            "amendment08_code_corrigendum_sha256"
        ],
        "amendment08_journal_contamination_adjudication_sha256": (
            adjudication_sha256
        ),
        "source_freeze_receipt_sha256": source["source_freeze_receipt_sha256"],
        "current_a06_authorized_change_ledger_sha256": source[
            "current_a06_authorized_change_ledger_sha256"
        ],
        "current_a06_final_prerun_evidence_sha256": source[
            "current_a06_final_prerun_evidence_sha256"
        ],
        "a08_prelive_package_lineage_sha256": package["lineage_sha256"],
        "a08_prelive_package_receipt_sha256": package["receipt_sha256"],
        "cache_manifest_sha256": full.A07_CACHE_MANIFEST_SHA256,
        "frozen_population_sha256": full.A07_FROZEN_POPULATION_SHA256,
        "corrigendum_journal_event_raw_sha256": full.sha256_file(event7_path),
        "corrigendum_journal_head_sha256": full.sha256_bytes(
            full.strict_full_canonical_json_bytes(event7)
        ),
    }
    full.atomic_write_strict_json(receipt_path, receipt)

    def validate_overlay(path: Path) -> dict:
        return {
            "status": "PASS",
            "overlay_install_receipt_sha256": full.sha256_file(receipt_path),
            **lineage,
        }

    monkeypatch.setattr(full, "validate_amendment08_overlay", validate_overlay)
    monkeypatch.setattr(
        full, "_amendment08_frozen_lineage_bindings",
        lambda path: dict(lineage),
    )
    return event7, receipt


def test_amendment08_raw_prefix_preserves_exact_bytes_chain_and_physical_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, _ = _amendment08_isolated_journal_fixture(tmp_path, monkeypatch)
    expected = _write_amendment08_raw_prefix(run_dir)
    validated = full._validate_amendment08_raw_prefix(
        run_dir, require_exact_boundary=True,
    )
    assert validated["status"] == "PASS"
    assert validated["base_event_count"] == 7
    assert validated["raw_event_count"] == 7
    assert validated["events"] == expected
    assert [event["event_index"] for event in validated["events"]] == list(range(7))
    assert [
        record["physical_event_index"]
        for record in validated["classified_records"]
    ] == [5, 6]
    assert [
        record["event"]["event_index"]
        for record in validated["classified_records"]
    ] == [5, 6]
    inventory = full.amendment08_live_journal_inventory_bytes(run_dir)
    assert inventory.endswith(b"\n")
    assert len(inventory.splitlines()) == 7
    assert [line.split(b"\t")[2] for line in inventory.splitlines()] == [
        f"{index:08d}.json".encode("ascii") for index in range(7)
    ]


@pytest.mark.parametrize(
    "tamper,index",
    [
        ("one_bit", 1),
        ("raw", 0),
        ("canonical", 0),
        ("index", 5),
        ("time", 5),
        ("link", 6),
        ("payload", 5),
    ],
)
def test_amendment08_raw_prefix_rejects_every_frozen_dimension_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str, index: int,
) -> None:
    run_dir, _ = _amendment08_isolated_journal_fixture(tmp_path, monkeypatch)
    events = _write_amendment08_raw_prefix(run_dir)
    path = full._journal_event_path(run_dir, index)
    if tamper == "one_bit":
        data = bytearray(path.read_bytes())
        offset = data.index(b"PASS")
        data[offset] ^= 0x01
        path.write_bytes(bytes(data))
    elif tamper == "raw":
        path.write_bytes(path.read_bytes() + b" ")
    else:
        event = copy.deepcopy(events[index])
        if tamper == "canonical":
            event["status"] = "FAIL"
        elif tamper == "index":
            event["event_index"] = 50
        elif tamper == "time":
            event["recorded_at"] = "2026-08-20T00:00:00-04:00"
        elif tamper == "link":
            event["previous_event_sha256"] = "0" * 64
        elif tamper == "payload":
            event["completion_sha256"] = "f" * 64
        path.write_bytes(full.strict_full_json_bytes(event))
    with pytest.raises((full.Amendment06IntegrityError, full.StrictJSONError)):
        full._validate_amendment08_raw_prefix(
            run_dir, require_exact_boundary=True,
        )


def test_amendment08_semantic_view_retains_physical_indices_and_one_real_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, _ = _amendment08_isolated_journal_fixture(tmp_path, monkeypatch)
    _write_amendment08_raw_prefix(run_dir)
    _install_amendment08_semantic_view_fixture(run_dir, monkeypatch)
    initial = full.read_amendment08_journal_views(run_dir)
    assert [event["event_index"] for event in initial["raw_events"]] == list(range(8))
    assert [event["event_index"] for event in initial["semantic_events"]] == [
        0, 1, 2, 3, 4, 7,
    ]
    assert sorted(initial["semantic_events_by_physical_index"]) == [0, 1, 2, 3, 4, 7]
    assert initial["raw_reporting_completion_events"] == 2
    assert initial["semantic_reporting_completion_events"] == 0

    completion = run_dir / "provenance/reporting_completion_manifest.json"
    completion.write_bytes(b"genuine-reporting-completion\n")
    committed = full._ensure_unique_bound_journal_event(
        run_dir=run_dir,
        event_name="REPORTING_COMPLETION_COMMITTED",
        binding_path=completion,
        binding_field="completion_sha256",
        phase="REPORTING",
        variant="GLOBAL",
        recovered=False,
        extra={"attempt_id": "b" * 32},
    )
    assert committed["event_index"] == 8
    final = full.read_amendment08_journal_views(run_dir)
    assert [event["event_index"] for event in final["semantic_events"]] == [
        0, 1, 2, 3, 4, 7, 8,
    ]
    assert final["raw_reporting_completion_events"] == 3
    assert final["semantic_reporting_completion_events"] == 1


@pytest.mark.parametrize(
    "tamper",
    ["third_exclusion", "classified_record", "receipt_event7"],
)
def test_amendment08_semantic_gate_rejects_adjudication_and_receipt_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str,
) -> None:
    run_dir, _ = _amendment08_isolated_journal_fixture(tmp_path, monkeypatch)
    _write_amendment08_raw_prefix(run_dir)
    _, receipt = _install_amendment08_semantic_view_fixture(run_dir, monkeypatch)
    adjudication_path = (
        run_dir / "provenance/amendment08_journal_contamination_adjudication.json"
    )
    receipt_path = run_dir / "provenance/amendment08_overlay_install_receipt.json"
    if tamper in {"third_exclusion", "classified_record"}:
        adjudication = full.strict_full_load_file(adjudication_path)
        if tamper == "third_exclusion":
            adjudication["excluded_physical_event_indices"] = [5, 6, 7]
            adjudication["semantic_exclusion_count"] = 3
        else:
            adjudication["classified_records"][0]["raw_sha256"] = "f" * 64
        full.atomic_write_strict_json(adjudication_path, adjudication)
        receipt[
            "amendment08_journal_contamination_adjudication_sha256"
        ] = full.sha256_file(adjudication_path)
    else:
        receipt["transaction_id"] = "8" * 32
    full.atomic_write_strict_json(receipt_path, receipt)
    with pytest.raises(full.Amendment06IntegrityError):
        full.read_amendment08_journal_views(run_dir)


def test_amendment08_semantic_gate_rejects_a_third_fixture_contaminant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, _ = _amendment08_isolated_journal_fixture(tmp_path, monkeypatch)
    _write_amendment08_raw_prefix(run_dir)
    _install_amendment08_semantic_view_fixture(run_dir, monkeypatch)
    full._journal_event(
        run_dir,
        "REPORTING_COMPLETION_COMMITTED",
        phase="REPORTING",
        variant_id="GLOBAL",
        attempt_id="a" * 32,
        completion_sha256=full.A08_TEST_FIXTURE_SHA256,
    )
    with pytest.raises(full.Amendment06IntegrityError, match="contamin"):
        full.read_amendment08_journal_views(run_dir)


def test_amendment08_interrupted_attempt_recovery_preserves_classified_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, _ = _amendment08_isolated_journal_fixture(tmp_path, monkeypatch)
    _write_amendment08_raw_prefix(run_dir)
    _install_amendment08_semantic_view_fixture(run_dir, monkeypatch)
    attempt_id = "b" * 32
    full._journal_event(
        run_dir,
        "TRAINING_ATTEMPT_STARTED",
        phase="TRAINING",
        variant_id="full_new_reference",
        attempt_id=attempt_id,
        **_amendment08_lineage_stub(),
    )
    full._record_interrupted_attempts(
        run_dir=run_dir,
        phase="TRAINING",
        variant="full_new_reference",
    )
    raw = full._read_journal(run_dir)
    assert [raw[index]["event"] for index in (5, 6)] == [
        "REPORTING_COMPLETION_COMMITTED",
        "REPORTING_COMPLETION_COMMITTED",
    ]
    failures = [
        event for event in raw
        if event.get("event") == "TRAINING_ATTEMPT_FAILED"
        and event.get("attempt_id") == attempt_id
    ]
    assert len(failures) == 1
    assert failures[0]["event_index"] == 9
    assert failures[0]["recovered_after_interruption"] is True


def test_amendment08_cache_missing_fails_before_any_generation_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, _ = _amendment08_isolated_journal_fixture(tmp_path, monkeypatch)
    receipt = run_dir / "provenance/amendment08_overlay_install_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b"installed\n")

    def forbidden(*args, **kwargs):
        pytest.fail("Amendment 08 attempted cache generation/rematerialization")

    monkeypatch.setattr(full, "_build_population_cache", forbidden)
    monkeypatch.setattr(
        full.paired_resampling, "build_frozen_resampling_population", forbidden,
    )
    monkeypatch.setattr(full.core, "augment_with_lineage", forbidden)
    monkeypatch.setattr(full.legacy, "safe_smote_resample", forbidden)
    with pytest.raises(full.Amendment06IntegrityError, match="forbids cache"):
        full.load_or_build_population_cache(
            run_dir=run_dir,
            locked=SimpleNamespace(),
            development=SimpleNamespace(),
        )


def test_amendment08_existing_cache_load_has_zero_generation_or_rematerialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, runtime = _amendment08_isolated_journal_fixture(tmp_path, monkeypatch)
    receipt = run_dir / "provenance/amendment08_overlay_install_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b"installed\n")
    destination = runtime / "full_population" / run_dir.name
    destination.mkdir(parents=True)
    manifest = {"status": "PASS", "cache": "frozen"}
    imputer = {"feature_order": ["feature"]}
    full.atomic_write_strict_json(
        run_dir / "provenance/frozen_full_resampling_manifest.json", manifest,
    )
    full.atomic_write_strict_json(
        run_dir / "provenance/frozen_full_imputer.json", imputer,
    )
    calls = {"validate": 0, "cache_load": 0}

    def validate_cache(**kwargs):
        calls["validate"] += 1
        assert kwargs["destination"] == destination
        return manifest, imputer

    def cache_load(*, run_dir: Path, destination: Path) -> dict:
        calls["cache_load"] += 1
        return {"event": "CACHE_LOAD", "event_index": 8}

    def forbidden(*args, **kwargs):
        pytest.fail("Amendment 08 generated or rematerialized the frozen cache")

    shapes = {
        "pre_smote_X.npy": (full.PRE_SMOTE_ROWS, 93),
        "pre_smote_y.npy": (full.PRE_SMOTE_ROWS,),
        "pre_smote_weights.npy": (full.PRE_SMOTE_ROWS,),
        "post_smote_X.npy": (full.POST_SMOTE_ROWS, 93),
        "post_smote_y.npy": (full.POST_SMOTE_ROWS,),
        "post_smote_weights.npy": (full.POST_SMOTE_ROWS,),
        "validation_X.npy": (56_843, 93),
    }
    monkeypatch.setattr(full, "_scientific_journal_events", lambda path: [])
    monkeypatch.setattr(full, "_validate_population_cache", validate_cache)
    monkeypatch.setattr(full, "_ensure_amendment08_cache_load_event", cache_load)
    monkeypatch.setattr(full, "_build_population_cache", forbidden)
    monkeypatch.setattr(
        full.paired_resampling, "build_frozen_resampling_population", forbidden,
    )
    monkeypatch.setattr(full.core, "augment_with_lineage", forbidden)
    monkeypatch.setattr(full.legacy, "safe_smote_resample", forbidden)
    monkeypatch.setattr(
        full.np,
        "load",
        lambda path, **kwargs: SimpleNamespace(shape=shapes[Path(path).name]),
    )
    cache = full.load_or_build_population_cache(
        run_dir=run_dir,
        locked=SimpleNamespace(),
        development=SimpleNamespace(),
    )
    assert cache.root == destination
    assert calls == {"validate": 1, "cache_load": 1}


def test_amendment08_event7_to_cache8_to_corrected_start9_is_exact_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, runtime = _amendment08_isolated_journal_fixture(tmp_path, monkeypatch)
    _write_amendment08_raw_prefix(run_dir)
    _install_amendment08_semantic_view_fixture(run_dir, monkeypatch)
    expected_lineage = _amendment08_lineage_stub()
    expected_lineage["amendment08_journal_contamination_adjudication_sha256"] = (
        full.sha256_file(
            run_dir
            / "provenance/amendment08_journal_contamination_adjudication.json"
        )
    )
    monkeypatch.setattr(
        full,
        "_validate_preserved_target_cache_files",
        lambda path: {
            "cache_manifest_sha256": full.A07_CACHE_MANIFEST_SHA256,
            "cache_member_count": 10,
            "unique_realizations": 1,
        },
    )
    destination = runtime / "full_population" / run_dir.name
    destination.mkdir(parents=True)
    event8 = full._ensure_amendment08_cache_load_event(
        run_dir=run_dir, destination=destination,
    )
    assert event8["event"] == "CACHE_LOAD"
    assert event8["event_index"] == 8
    assert all(event8[key] == value for key, value in expected_lineage.items())
    assert full._ensure_amendment08_cache_load_event(
        run_dir=run_dir, destination=destination,
    ) == event8
    assert len(full._read_journal(run_dir)) == 9

    attempt_id = "b" * 32
    event9 = full._journal_event(
        run_dir,
        "TRAINING_ATTEMPT_STARTED",
        phase="TRAINING",
        variant_id="full_new_reference",
        attempt_id=attempt_id,
        **expected_lineage,
    )
    assert event9["event_index"] == 9
    assert event9["previous_event_sha256"] == full.sha256_bytes(
        full.strict_full_canonical_json_bytes(event8)
    )
    overlay = full.validate_amendment08_overlay(run_dir)
    history = full._validate_amendment08_corrected_training_history(
        run_dir=run_dir,
        overlay=overlay,
        allow_unterminated=True,
    )
    assert history["unterminated_attempt_ids"] == [attempt_id]
    assert history["next_variant"] == "full_new_reference"
    assert [
        event["event"] for event in full._read_journal(run_dir)[7:10]
    ] == [
        "AMENDMENT08_CORRIGENDUM_INSTALLED",
        "CACHE_LOAD",
        "TRAINING_ATTEMPT_STARTED",
    ]
    event9_bytes = full._journal_event_path(run_dir, 9).read_bytes()
    full._record_interrupted_attempts(
        run_dir=run_dir,
        phase="TRAINING",
        variant="full_new_reference",
    )
    assert full._journal_event_path(run_dir, 9).read_bytes() == event9_bytes
    recovered = full._validate_amendment08_corrected_training_history(
        run_dir=run_dir,
        overlay=overlay,
        allow_unterminated=False,
    )
    assert recovered["unterminated_attempt_ids"] == []
    assert recovered["next_variant"] == "full_new_reference"
    failure = full._read_journal(run_dir)[10]
    assert failure["event"] == "TRAINING_ATTEMPT_FAILED"
    assert failure["attempt_id"] == attempt_id
    assert failure["failure_phase"] == "PROCESS_INTERRUPTION_BEFORE_COMPLETION_COMMIT"
    assert failure["recovered_after_interruption"] is True

    retry_id = "c" * 32
    retry = full._journal_event(
        run_dir,
        "TRAINING_ATTEMPT_STARTED",
        phase="TRAINING",
        variant_id="full_new_reference",
        attempt_id=retry_id,
        **full._amendment08_training_event_fields(
            run_dir=run_dir, variant="full_new_reference",
        ),
    )
    assert retry["event_index"] == 11
    retried = full._validate_amendment08_corrected_training_history(
        run_dir=run_dir,
        overlay=overlay,
        allow_unterminated=True,
    )
    assert retried["unterminated_attempt_ids"] == [retry_id]
    assert retried["corrected_attempt_count"] == 2


def test_amendment08_semantic_consumers_do_not_call_the_raw_reader_directly() -> None:
    tree = ast.parse(Path(full.__file__).read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    semantic_consumers = {
        "_ensure_unique_bound_journal_event",
        "_record_interrupted_attempts",
        "_phase_publication_records",
        "_recover_unpublished_paths",
        "load_or_build_population_cache",
        "_ensure_amendment08_cache_load_event",
        "_validate_amendment08_corrected_training_history",
        "train_one_variant",
        "validate_models_frozen_manifest",
        "freeze_all_models",
        "score_one_external_variant",
        "_build_reports_and_bootstrap_impl",
        "build_reports_and_bootstrap",
        "build_amendment08_raw_journal_cutoff",
        "build_execution_ledger",
        "_amendment08_finalization_handoff_ready",
        "finalize_scientific_run",
        "validate_final_scientific_run",
    }
    assert semantic_consumers <= set(functions)
    bypasses: dict[str, list[int]] = {}
    for name in sorted(semantic_consumers):
        lines = [
            call.lineno
            for call in ast.walk(functions[name])
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_read_journal"
        ]
        if lines:
            bypasses[name] = lines
    assert bypasses == {}
    build_cutoff_calls = {
        call.func.id
        for call in ast.walk(functions["build_amendment08_raw_journal_cutoff"])
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    handoff_calls = {
        call.func.id
        for call in ast.walk(functions["_amendment08_finalization_handoff_ready"])
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "read_amendment08_journal_views" in build_cutoff_calls
    assert "_scientific_journal_events" in handoff_calls


def test_amendment08_former_culprit_twice_writes_only_isolated_journals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulated_production = tmp_path / "simulated_production"
    production_runtime = simulated_production / "runtime"
    production_results = simulated_production / "results"
    production_run = production_results / full.A08_TARGET_RUN_ID
    production_journal = (
        production_runtime / "journals" / full.A08_TARGET_RUN_ID
    )
    production_run.mkdir(parents=True)
    production_journal.mkdir(parents=True)
    for index in range(7):
        (production_journal / f"{index:08d}.json").write_bytes(
            f"simulated-live-event-{index}\n".encode("ascii")
        )
    monkeypatch.setattr(
        full, "PRODUCTION_EXPECTED_RUNTIME_ROOT", production_runtime,
    )
    monkeypatch.setattr(full, "PRODUCTION_RESULTS_ROOT", production_results)
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", production_runtime)
    before = full.amendment08_live_journal_inventory_bytes(production_run)

    for iteration in range(2):
        isolated_root = tmp_path / f"culprit_reproduction_{iteration}"
        isolated_runtime = isolated_root / "runtime"
        run_dir = isolated_root / "results" / full.A08_TARGET_RUN_ID
        run_dir.mkdir(parents=True)
        monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", isolated_runtime)
        completion = run_dir / "provenance/reporting_completion_manifest.json"
        completion.parent.mkdir(parents=True)
        completion.write_bytes(b"{}\n")
        full._journal_event(
            run_dir, "REPORTING_COMPLETION_COMMITTED", phase="REPORTING",
            variant_id="GLOBAL", attempt_id="a" * 32,
            completion_sha256=full.sha256_file(completion),
        )
        assert [event["event"] for event in full._read_journal(run_dir)] == [
            "REPORTING_COMPLETION_COMMITTED"
        ]
        assert not (
            production_runtime / "journals" / f".{full.A08_TARGET_RUN_ID}.lock"
        ).exists()

    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", production_runtime)
    assert full.amendment08_live_journal_inventory_bytes(production_run) == before


@pytest.mark.parametrize(
    "crash_boundary",
    ("after_intent", "after_ledger", "after_evidence", "after_receipt"),
)
def test_amendment08_source_freeze_reconciles_each_publication_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_boundary: str,
) -> None:
    a08_runtime = tmp_path / "a08"
    a06_runtime = tmp_path / "a06"
    ledger = a06_runtime / "authorized_change_ledger.jsonl"
    evidence_path = a06_runtime / "pre_run/final_prerun_evidence.json"
    ledger.parent.mkdir(parents=True)
    evidence_path.parent.mkdir(parents=True)
    old_ledger = b"original-ledger-prefix\n"
    old_evidence = full.strict_full_json_bytes({"state": "original"})
    ledger.write_bytes(old_ledger)
    evidence_path.write_bytes(old_evidence)
    monkeypatch.setattr(full, "A06_RUNTIME", a06_runtime)
    monkeypatch.setattr(full, "A08_RUNTIME", a08_runtime)
    monkeypatch.setattr(
        full, "A08_SOURCE_FREEZE_INTENT",
        a08_runtime / "source_freeze/install_intent.json",
    )
    monkeypatch.setattr(
        full, "A08_SOURCE_FREEZE_RECEIPT",
        a08_runtime / "source_freeze/install_receipt.json",
    )
    monkeypatch.setattr(full, "AUTHORIZED_CHANGE_LEDGER", ledger)
    monkeypatch.setattr(full, "FINAL_PRERUN_EVIDENCE", evidence_path)
    monkeypatch.setattr(full, "A07_ORIGINAL_LEDGER_SHA256", full.sha256_bytes(old_ledger))
    monkeypatch.setattr(full, "A07_ORIGINAL_LEDGER_SIZE_BYTES", len(old_ledger))
    monkeypatch.setattr(
        full, "A07_ORIGINAL_PRERUN_SHA256", full.sha256_bytes(old_evidence),
    )
    bound_runtime = a06_runtime / "corrigendum08_journal_adjudication"
    monkeypatch.setattr(full, "A08_A06_BOUND_RUNTIME", bound_runtime)
    checks = _amendment07_final_checks(bound_runtime)
    desired_ledger = old_ledger + b"amendment08-event\n"
    desired_evidence = full.strict_full_json_bytes({"state": "amendment08"})
    mapping = {
        relative: {"relative_path": relative, "size_bytes": 1, "sha256": f"{index + 1:x}" * 64}
        for index, relative in enumerate(full.A08_CHANGED_ALL_RELATIVES)
    }
    material = {
        "event": {"source_freeze_transaction_id": "b" * 32},
        "event_canonical_sha256": "c" * 64,
        "ledger_bytes": desired_ledger,
        "ledger_sha256": full.sha256_bytes(desired_ledger),
        "evidence": {"state": "amendment08"},
        "evidence_bytes": desired_evidence,
        "evidence_sha256": full.sha256_bytes(desired_evidence),
        "mapping": mapping,
        "checks": checks,
    }
    chain = {
        "mapping": mapping,
        "ledger_sha256": material["ledger_sha256"],
        "latest_event_sha256": material["event_canonical_sha256"],
    }
    monkeypatch.setattr(
        full, "_amendment08_source_freeze_material", lambda **kwargs: material,
    )
    monkeypatch.setattr(
        full, "_amendment08_source_freeze_material_from_intent",
        lambda intent: material,
    )
    monkeypatch.setattr(
        full, "validate_authorized_change_chain", lambda: chain,
    )
    monkeypatch.setattr(
        full, "validate_final_prerun_evidence", lambda value: {"status": "PASS"},
    )
    monkeypatch.setattr(
        full, "validate_amendment08_source_freeze",
        lambda: {"status": "PASS", "transaction_id": "b" * 32},
    )
    monkeypatch.setattr(full.uuid, "uuid4", lambda: SimpleNamespace(hex="b" * 32))

    publish_json = full.publish_strict_json_no_clobber
    json_crash_enabled = True

    def crash_after_json_publication(path: Path, value: dict) -> None:
        nonlocal json_crash_enabled
        publish_json(path, value)
        boundary_path = {
            "after_intent": full.A08_SOURCE_FREEZE_INTENT,
            "after_receipt": full.A08_SOURCE_FREEZE_RECEIPT,
        }.get(crash_boundary)
        if json_crash_enabled and boundary_path is not None and Path(path) == boundary_path:
            json_crash_enabled = False
            raise RuntimeError(f"synthetic crash {crash_boundary}")

    monkeypatch.setattr(
        full, "publish_strict_json_no_clobber", crash_after_json_publication,
    )
    replace = full._replace_exact_file_from_a08_staging
    replacement_calls = 0

    def crash_after_replacement(**kwargs) -> None:
        nonlocal replacement_calls
        replace(**kwargs)
        replacement_calls += 1
        expected_call = 1 if crash_boundary == "after_ledger" else 2
        if (
            crash_boundary in {"after_ledger", "after_evidence"}
            and replacement_calls == expected_call
        ):
            raise RuntimeError(f"synthetic crash {crash_boundary}")

    monkeypatch.setattr(
        full, "_replace_exact_file_from_a08_staging", crash_after_replacement,
    )
    with pytest.raises(RuntimeError, match="synthetic crash"):
        full.install_amendment08_source_freeze(
            checks=checks,
            recorded_at="2026-08-20T20:00:00-04:00",
            evidence_generated_at="2026-08-20T20:00:01-04:00",
        )
    intent_bytes = full.A08_SOURCE_FREEZE_INTENT.read_bytes()
    assert ledger.read_bytes() == (
        old_ledger if crash_boundary == "after_intent" else desired_ledger
    )
    assert evidence_path.read_bytes() == (
        desired_evidence
        if crash_boundary in {"after_evidence", "after_receipt"}
        else old_evidence
    )
    receipt_bytes = (
        full.A08_SOURCE_FREEZE_RECEIPT.read_bytes()
        if crash_boundary == "after_receipt"
        else None
    )
    assert full.A08_SOURCE_FREEZE_RECEIPT.exists() is (
        crash_boundary == "after_receipt"
    )

    monkeypatch.setattr(full, "_replace_exact_file_from_a08_staging", replace)
    result = full.install_amendment08_source_freeze(checks=checks)
    assert result["status"] == "PASS"
    assert full.A08_SOURCE_FREEZE_INTENT.read_bytes() == intent_bytes
    assert ledger.read_bytes() == desired_ledger
    assert evidence_path.read_bytes() == desired_evidence
    receipt = full.strict_full_load_file(full.A08_SOURCE_FREEZE_RECEIPT)
    if receipt_bytes is not None:
        assert full.A08_SOURCE_FREEZE_RECEIPT.read_bytes() == receipt_bytes
    assert receipt["source_ledger_event_index"] == 5
    assert receipt["test_evidence"] == checks
    assert set(receipt["final_source_hashes"]) == set(full.A08_CHANGED_ALL_RELATIVES)


def test_amendment08_effective_manifest_changes_exactly_two_scientific_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_root = tmp_path / "study"
    run_dir = tmp_path / "run"
    manifest = run_dir / "provenance/scientific_execution_code_manifest.tsv"
    manifest.parent.mkdir(parents=True)
    rows = [
        {
            "relative_path": relative,
            "size_bytes": 0,
            "sha256": "0" * 64,
            "role": "SCIENTIFIC_TEST" if relative.startswith("tests/") else "SCIENTIFIC_CODE",
        }
        for relative in (*full.SCIENTIFIC_CODE_RELATIVES, *full.SCIENTIFIC_TEST_RELATIVES)
    ]
    manifest.write_bytes(full._manifest_bytes(rows))
    for relative, data in {
        "code/amendment06_full.py": b"final amendment08 runner\n",
        "tests/test_amendment06.py": b"final amendment08 tests\n",
    }.items():
        path = study_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    monkeypatch.setattr(full, "STUDY_ROOT", study_root)
    monkeypatch.setattr(
        full, "A07_BASE_SCIENTIFIC_MANIFEST_SHA256", full.sha256_file(manifest),
    )
    effective = full._parse_code_manifest_bytes(
        full._effective_amendment08_manifest_bytes(run_dir),
        label="isolated Amendment 08 effective manifest",
    )
    changed = [
        after["relative_path"]
        for before, after in zip(rows, effective, strict=True)
        if (before["size_bytes"], before["sha256"])
        != (after["size_bytes"], after["sha256"])
    ]
    assert changed == list(full.A08_CHANGED_SCIENTIFIC_RELATIVES)
    assert [row["relative_path"] for row in effective] == [
        *full.SCIENTIFIC_CODE_RELATIVES, *full.SCIENTIFIC_TEST_RELATIVES,
    ]


def test_amendment08_runner_whitelist_has_exact_236_plus_manifest_contract() -> None:
    expected = full.expected_scientific_run_files()
    assert len(expected) == 236
    assert "OUTPUT_MANIFEST_FINAL.tsv" not in expected
    assert len(expected | {"OUTPUT_MANIFEST_FINAL.tsv"}) == 237
    assert set(full.A08_OVERLAY_RUN_RELATIVES).issubset(expected)
    assert full.A08_RAW_JOURNAL_CUTOFF_RELATIVE in expected
    assert set(full.A07_OVERLAY_RUN_RELATIVES).isdisjoint(expected)
    selected_a08 = {
        relative
        for relative in expected
        if relative in full.A08_MODEL_BUNDLE_GLOBALS
        and full._models_bundle_flag(relative)
    }
    assert selected_a08 == set(full.A08_MODEL_BUNDLE_GLOBALS)


def _amendment08_lineage_stub() -> dict[str, str]:
    lineage = {
        "base_scientific_execution_code_manifest_sha256": (
            full.A07_BASE_SCIENTIFIC_MANIFEST_SHA256
        ),
        "effective_scientific_execution_code_manifest_sha256": "e" * 64,
        "amendment08_code_corrigendum_sha256": "c" * 64,
        "amendment08_journal_contamination_adjudication_sha256": "d" * 64,
        "amendment08_authorization": full.A08_AUTHORIZATION,
        "amendment08_prompt_sha256": full.A08_PROMPT_SHA256,
    }
    assert tuple(lineage) == full.A08_LINEAGE_FIELDS
    return lineage


def _amendment08_source_stub() -> dict:
    return {
        "source_freeze_receipt_sha256": "1" * 64,
        "current_a06_authorized_change_ledger_sha256": "2" * 64,
        "a08_source_ledger_event_index": 5,
        "a08_source_ledger_event_canonical_sha256": "3" * 64,
        "current_a06_final_prerun_evidence_sha256": "4" * 64,
        "final_source_hashes": {
            relative: f"{index + 5:x}" * 64
            for index, relative in enumerate(full.A08_CHANGED_ALL_RELATIVES)
        },
    }


def _amendment08_package_stub() -> dict[str, str]:
    return {
        "lineage_sha256": "9" * 64,
        "receipt_sha256": "a" * 64,
        "final_code_sha256": "b" * 64,
        "final_test_sha256": "c" * 64,
    }


def _write_amendment08_package_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(full.strict_full_json_bytes(payload))


def _amendment08_prelive_package_lineage_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    study_root = tmp_path / "study"
    runtime_root = tmp_path / "runtime"
    run_dir = tmp_path / "results" / full.A08_TARGET_RUN_ID
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(full, "STUDY_ROOT", study_root)
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", runtime_root)

    final_hashes: dict[str, str] = {}
    for index, relative in enumerate(full.A08_CHANGED_PACKAGE_RELATIVES):
        live = study_root / relative
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_bytes(f"final package fixture {index}\n".encode("ascii"))
        final_hashes[relative] = full.sha256_file(live)

    paths = full._amendment08_package_paths(run_dir)
    root = paths["lineage"].parent
    lock_path = paths["intent"].parent / "install.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")

    targeted_log = (tmp_path / "targeted-tests.log").resolve()
    package_log = (tmp_path / "package-rehearsal.log").resolve()
    targeted_log.write_bytes(b"17 passed, 2 skipped in 0.41s\n")
    package_log.write_bytes(b"STATUS=PASS\n")

    def log_binding(path: Path, command: str) -> dict:
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": full.sha256_file(path),
            "exit_code": 0,
            "command": command,
        }

    targeted_binding = log_binding(targeted_log, "pytest -q tests/test_amendment06.py")
    package_binding = log_binding(package_log, "python -m package_rehearsal")
    initial = dict(full.A08_INITIAL_PACKAGE_HASHES)
    forensic = {
        "code/amendment06_packaging.py": full.A08_PRE_PACKAGE_SHA256,
        "tests/test_amendment06_packaging.py": full.A08_PRE_PACKAGE_TEST_SHA256,
    }
    guarded_names = sorted(full.A08_REQUIRED_ZERO_SCIENCE_API_NAMES)
    rehearsal_path = (tmp_path / "package-rehearsal-evidence.json").resolve()
    rehearsal = {
        "schema": "amendment08_package_rehearsal_evidence/v1",
        "status": "PASS",
        "authorization": full.A08_AUTHORIZATION,
        "prompt_sha256": full.A08_PROMPT_SHA256,
        "target_run_id": run_dir.name,
        "package_set_id": full.A08_PACKAGE_SET_ID,
        "recorded_at": "2026-08-20T22:00:00-04:00",
        "fixture_run_id": run_dir.name,
        "targeted_tests_log": targeted_binding,
        "package_rehearsal_log": package_binding,
        "initial_package_validator_manifest_sha256": (
            full.A08_INITIAL_PACKAGE_MANIFEST_SHA256
        ),
        "initial_hashes": initial,
        "final_hashes": final_hashes,
        "expected_run_manifest_rows": 236,
        "actual_run_regular_files": 237,
        "review_selected_run_rows": 226,
        "review_embedded_manifest_rows": 227,
        "review_zip_file_entry_count": 228,
        "models_selected_run_rows": 101,
        "models_embedded_manifest_rows": 102,
        "models_zip_file_entry_count": 103,
        "receipt_only_validation": "PASS",
        "independent_zip_reopen": "PASS",
        "package_retry_status": "PASS",
        "guarded_scientific_api_calls": 0,
        "guarded_api_names": guarded_names,
        "run_mutation_calls": 0,
        "live_runtime_journal_reads": 0,
    }
    _write_amendment08_package_json(rehearsal_path, rehearsal)
    rehearsal_binding = {
        "path": str(rehearsal_path),
        "size_bytes": rehearsal_path.stat().st_size,
        "sha256": full.sha256_file(rehearsal_path),
    }

    snapshots = [paths["code_snapshot"], paths["test_snapshot"]]
    evidences: list[dict] = []
    for index, (relative, snapshot) in enumerate(zip(
        full.A08_CHANGED_PACKAGE_RELATIVES, snapshots, strict=True,
    )):
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((study_root / relative).read_bytes())
        evidence = {
            "schema": "amendment08_package_rehearsal_guard/v1",
            "status": "PASS",
            "authorization": full.A08_AUTHORIZATION,
            "prompt_sha256": full.A08_PROMPT_SHA256,
            "target_run_id": run_dir.name,
            "package_set_id": full.A08_PACKAGE_SET_ID,
            "sequence": index + 1,
            "target_relative_path": relative,
            "previous_code_sha256": initial[relative],
            "new_code_sha256": final_hashes[relative],
            "source_snapshot_path": str(snapshot.resolve()),
            "source_snapshot_sha256": final_hashes[relative],
            "targeted_tests_log": targeted_binding,
            "package_rehearsal_log": package_binding,
            "package_rehearsal_evidence_path": str(rehearsal_path),
            "package_rehearsal_evidence_sha256": rehearsal_binding["sha256"],
            "guarded_scientific_api_calls": 0,
            "guarded_api_names": guarded_names,
            "run_mutation_calls": 0,
            "live_runtime_journal_reads": 0,
            "receipt_only_validation": "PASS",
        }
        evidences.append(evidence)
        _write_amendment08_package_json(paths[f"evidence{index}"], evidence)

    events: list[dict] = []
    previous_event_sha256 = None
    for index, (relative, snapshot) in enumerate(zip(
        full.A08_CHANGED_PACKAGE_RELATIVES, snapshots, strict=True,
    )):
        evidence_path = paths[f"evidence{index}"]
        event = {
            "schema": "amendment08_package_code_lineage_event/v1",
            "status": "PASS",
            "authorization": full.A08_AUTHORIZATION,
            "prompt_sha256": full.A08_PROMPT_SHA256,
            "target_run_id": run_dir.name,
            "package_set_id": full.A08_PACKAGE_SET_ID,
            "set_event_index": index,
            "sequence": index + 1,
            "target_relative_path": relative,
            "previous_code_sha256": initial[relative],
            "new_code_sha256": final_hashes[relative],
            "previous_event_sha256": previous_event_sha256,
            "initial_package_validator_manifest_sha256": (
                full.A08_INITIAL_PACKAGE_MANIFEST_SHA256
            ),
            "targeted_tests_log_sha256": targeted_binding["sha256"],
            "package_rehearsal_log_sha256": package_binding["sha256"],
            "package_rehearsal_evidence_sha256": rehearsal_binding["sha256"],
            "zero_science_evidence_path": str(evidence_path.resolve()),
            "zero_science_evidence_sha256": full.sha256_file(evidence_path),
            "source_snapshot_path": str(snapshot.resolve()),
            "source_snapshot_sha256": final_hashes[relative],
            "scientific_api_calls": 0,
        }
        events.append(event)
        event_bytes = full.strict_full_canonical_json_bytes(event) + b"\n"
        paths[f"event{index}"].parent.mkdir(parents=True, exist_ok=True)
        paths[f"event{index}"].write_bytes(event_bytes)
        previous_event_sha256 = full.sha256_bytes(
            full.strict_full_canonical_json_bytes(event)
        )
    lineage_bytes = b"".join(
        full.strict_full_canonical_json_bytes(event) + b"\n" for event in events
    )
    paths["lineage"].parent.mkdir(parents=True, exist_ok=True)
    paths["lineage"].write_bytes(lineage_bytes)

    records = []
    for index, (relative, snapshot) in enumerate(zip(
        full.A08_CHANGED_PACKAGE_RELATIVES, snapshots, strict=True,
    )):
        evidence_path = paths[f"evidence{index}"]
        event_path = paths[f"event{index}"]
        records.append({
            "sequence": index + 1,
            "target_relative_path": relative,
            "previous_code_sha256": initial[relative],
            "new_code_sha256": final_hashes[relative],
            "snapshot_relative_path": snapshot.relative_to(root).as_posix(),
            "snapshot_sha256": full.sha256_file(snapshot),
            "evidence_relative_path": evidence_path.relative_to(root).as_posix(),
            "evidence_sha256": full.sha256_file(evidence_path),
            "event_relative_path": event_path.relative_to(root).as_posix(),
            "event_sha256": full.sha256_file(event_path),
            "event_canonical_sha256": full.sha256_bytes(
                full.strict_full_canonical_json_bytes(events[index])
            ),
        })
    intent = {
        "schema": "amendment08_package_code_lineage_install_intent/v1",
        "status": "INTENT",
        "authorization": full.A08_AUTHORIZATION,
        "prompt_sha256": full.A08_PROMPT_SHA256,
        "target_run_id": run_dir.name,
        "package_set_id": full.A08_PACKAGE_SET_ID,
        "set_kind": "PRELIVE",
        "created_at": "2026-08-20T22:00:01-04:00",
        "initial_package_validator_manifest_sha256": (
            full.A08_INITIAL_PACKAGE_MANIFEST_SHA256
        ),
        "initial_hashes": initial,
        "forensic_pre_a08_draft_hashes": forensic,
        "final_hashes": final_hashes,
        "base_lineage_event_count": 0,
        "base_lineage_head_sha256": None,
        "targeted_tests_log": targeted_binding,
        "package_rehearsal_log": package_binding,
        "package_rehearsal_evidence": rehearsal_binding,
        "event_count": 2,
        "records": records,
    }
    _write_amendment08_package_json(paths["intent"], intent)
    event_hashes = [
        full.sha256_bytes(full.strict_full_canonical_json_bytes(event))
        for event in events
    ]
    receipt = {
        "schema": "amendment08_package_code_lineage_install_receipt/v1",
        "status": "PASS",
        "authorization": full.A08_AUTHORIZATION,
        "prompt_sha256": full.A08_PROMPT_SHA256,
        "target_run_id": run_dir.name,
        "package_set_id": full.A08_PACKAGE_SET_ID,
        "set_kind": "PRELIVE",
        "intent_sha256": full.sha256_file(paths["intent"]),
        "initial_package_validator_manifest_sha256": (
            full.A08_INITIAL_PACKAGE_MANIFEST_SHA256
        ),
        "initial_hashes": initial,
        "forensic_pre_a08_draft_hashes": forensic,
        "final_hashes": final_hashes,
        "event_count": 2,
        "event_hashes": event_hashes,
        "lineage_event_count": 2,
        "lineage_sha256": full.sha256_bytes(lineage_bytes),
        "lineage_head_sha256": event_hashes[-1],
        "evidence_hashes": [
            full.sha256_file(paths[f"evidence{index}"]) for index in range(2)
        ],
        "snapshot_hashes": {
            relative: full.sha256_file(snapshot)
            for relative, snapshot in zip(
                full.A08_CHANGED_PACKAGE_RELATIVES, snapshots, strict=True,
            )
        },
        "targeted_tests_log": targeted_binding,
        "package_rehearsal_log": package_binding,
        "package_rehearsal_evidence": rehearsal_binding,
        "scientific_api_calls": 0,
        "installed_at": "2026-08-20T22:00:02-04:00",
    }
    _write_amendment08_package_json(paths["receipt"], receipt)
    return {
        "run_dir": run_dir,
        "paths": paths,
        "targeted_log": targeted_log,
        "rehearsal_path": rehearsal_path,
        "rehearsal": rehearsal,
        "events": events,
        "evidences": evidences,
        "intent": intent,
        "receipt": receipt,
        "snapshots": snapshots,
        "root": root,
    }


def _rebind_amendment08_package_evidence(fixture: dict) -> None:
    paths = fixture["paths"]
    events = fixture["events"]
    evidences = fixture["evidences"]
    intent = fixture["intent"]
    receipt = fixture["receipt"]
    snapshots = fixture["snapshots"]
    root = fixture["root"]
    previous_event_sha256 = None
    event_hashes: list[str] = []
    evidence_hashes: list[str] = []
    records: list[dict] = []
    for index, (relative, snapshot) in enumerate(zip(
        full.A08_CHANGED_PACKAGE_RELATIVES, snapshots, strict=True,
    )):
        evidence_path = paths[f"evidence{index}"]
        _write_amendment08_package_json(evidence_path, evidences[index])
        evidence_sha256 = full.sha256_file(evidence_path)
        event = events[index]
        event["previous_event_sha256"] = previous_event_sha256
        event["zero_science_evidence_sha256"] = evidence_sha256
        event_bytes = full.strict_full_canonical_json_bytes(event) + b"\n"
        event_path = paths[f"event{index}"]
        event_path.write_bytes(event_bytes)
        event_canonical_sha256 = full.sha256_bytes(
            full.strict_full_canonical_json_bytes(event)
        )
        previous_event_sha256 = event_canonical_sha256
        event_hashes.append(event_canonical_sha256)
        evidence_hashes.append(evidence_sha256)
        records.append({
            "sequence": index + 1,
            "target_relative_path": relative,
            "previous_code_sha256": intent["initial_hashes"][relative],
            "new_code_sha256": intent["final_hashes"][relative],
            "snapshot_relative_path": snapshot.relative_to(root).as_posix(),
            "snapshot_sha256": full.sha256_file(snapshot),
            "evidence_relative_path": evidence_path.relative_to(root).as_posix(),
            "evidence_sha256": evidence_sha256,
            "event_relative_path": event_path.relative_to(root).as_posix(),
            "event_sha256": full.sha256_file(event_path),
            "event_canonical_sha256": event_canonical_sha256,
        })
    lineage_bytes = b"".join(
        full.strict_full_canonical_json_bytes(event) + b"\n" for event in events
    )
    paths["lineage"].write_bytes(lineage_bytes)
    intent["records"] = records
    _write_amendment08_package_json(paths["intent"], intent)
    receipt["intent_sha256"] = full.sha256_file(paths["intent"])
    receipt["event_hashes"] = event_hashes
    receipt["lineage_sha256"] = full.sha256_bytes(lineage_bytes)
    receipt["lineage_head_sha256"] = event_hashes[-1]
    receipt["evidence_hashes"] = evidence_hashes
    receipt["snapshot_hashes"] = {
        relative: full.sha256_file(snapshot)
        for relative, snapshot in zip(
            full.A08_CHANGED_PACKAGE_RELATIVES, snapshots, strict=True,
        )
    }
    _write_amendment08_package_json(paths["receipt"], receipt)


def _rebind_amendment08_package_rehearsal(fixture: dict) -> None:
    rehearsal_path = fixture["rehearsal_path"]
    _write_amendment08_package_json(rehearsal_path, fixture["rehearsal"])
    binding = {
        "path": str(rehearsal_path),
        "size_bytes": rehearsal_path.stat().st_size,
        "sha256": full.sha256_file(rehearsal_path),
    }
    fixture["intent"]["package_rehearsal_evidence"] = binding
    fixture["receipt"]["package_rehearsal_evidence"] = binding
    for evidence in fixture["evidences"]:
        evidence["package_rehearsal_evidence_path"] = binding["path"]
        evidence["package_rehearsal_evidence_sha256"] = binding["sha256"]
    for event in fixture["events"]:
        event["package_rehearsal_evidence_sha256"] = binding["sha256"]
    _rebind_amendment08_package_evidence(fixture)


def test_amendment08_prelive_package_lineage_validates_exact_public_schemas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tuple(map(len, (
        full.A08_PACKAGE_LINEAGE_EVENT_KEYS,
        full.A08_PACKAGE_LINEAGE_EVIDENCE_KEYS,
        full.A08_PACKAGE_LINEAGE_RECORD_KEYS,
        full.A08_PACKAGE_LINEAGE_INTENT_KEYS,
        full.A08_PACKAGE_LINEAGE_RECEIPT_KEYS,
        full.A08_PACKAGE_REHEARSAL_EVIDENCE_KEYS,
    ))) == (21, 21, 11, 19, 24, 28)
    assert len(full.A08_REQUIRED_ZERO_SCIENCE_API_NAMES) == 81
    fixture = _amendment08_prelive_package_lineage_fixture(tmp_path, monkeypatch)
    validated = full.validate_amendment08_prelive_package_lineage(
        fixture["run_dir"]
    )
    assert validated["status"] == "PASS"
    assert validated["event_count"] == 2
    assert validated["scientific_api_calls"] == 0


@pytest.mark.parametrize(
    "case",
    (
        "fabricated_evidence",
        "fixture_run_id",
        "undercovered_rehearsal_guard",
        "undercovered_event_guard",
        "log_hash",
        "intent_record",
        "receipt_snapshot",
        "receipt_evidence",
    ),
)
def test_amendment08_prelive_package_lineage_rejects_adversarial_bindings(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _amendment08_prelive_package_lineage_fixture(tmp_path, monkeypatch)
    paths = fixture["paths"]
    if case == "fabricated_evidence":
        fixture["evidences"][0] = {"fabricated": True}
        _rebind_amendment08_package_evidence(fixture)
    elif case == "fixture_run_id":
        fixture["rehearsal"]["fixture_run_id"] = "different_run_id"
        _rebind_amendment08_package_rehearsal(fixture)
    elif case == "undercovered_rehearsal_guard":
        fixture["rehearsal"]["guarded_api_names"] = sorted(
            full.A08_REQUIRED_ZERO_SCIENCE_API_NAMES
        )[:-1]
        _rebind_amendment08_package_rehearsal(fixture)
    elif case == "undercovered_event_guard":
        fixture["evidences"][0]["guarded_api_names"] = sorted(
            full.A08_REQUIRED_ZERO_SCIENCE_API_NAMES
        )[:-1]
        _rebind_amendment08_package_evidence(fixture)
    elif case == "log_hash":
        fixture["targeted_log"].write_bytes(b"18 passed in 0.42s\n")
    elif case == "intent_record":
        fixture["intent"]["records"][0]["snapshot_sha256"] = "0" * 64
        _write_amendment08_package_json(paths["intent"], fixture["intent"])
        fixture["receipt"]["intent_sha256"] = full.sha256_file(paths["intent"])
        _write_amendment08_package_json(paths["receipt"], fixture["receipt"])
    elif case == "receipt_snapshot":
        relative = full.A08_CHANGED_PACKAGE_RELATIVES[0]
        fixture["receipt"]["snapshot_hashes"][relative] = "0" * 64
        _write_amendment08_package_json(paths["receipt"], fixture["receipt"])
    else:
        fixture["receipt"]["evidence_hashes"][0] = "0" * 64
        _write_amendment08_package_json(paths["receipt"], fixture["receipt"])
    with pytest.raises(full.Amendment06IntegrityError):
        full.validate_amendment08_prelive_package_lineage(fixture["run_dir"])


def _append_amendment08_cache8(
    run_dir: Path, runtime: Path, monkeypatch: pytest.MonkeyPatch,
) -> dict:
    monkeypatch.setattr(
        full,
        "_validate_preserved_target_cache_files",
        lambda path: {
            "cache_manifest_sha256": full.A07_CACHE_MANIFEST_SHA256,
            "cache_member_count": 10,
            "unique_realizations": 1,
        },
    )
    destination = runtime / "full_population" / run_dir.name
    destination.mkdir(parents=True)
    return full._ensure_amendment08_cache_load_event(
        run_dir=run_dir, destination=destination,
    )


@pytest.mark.parametrize("event_name", ["CACHE_LOAD", "TRAINING_ATTEMPT_STARTED"])
@pytest.mark.parametrize("field", full.A08_LINEAGE_FIELDS)
def test_amendment08_history_rejects_each_lineage_disagreement_at_events_8_and_9(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    event_name: str,
    field: str,
) -> None:
    run_dir, runtime = _amendment08_isolated_journal_fixture(tmp_path, monkeypatch)
    _write_amendment08_raw_prefix(run_dir)
    _install_amendment08_semantic_view_fixture(run_dir, monkeypatch)
    event8 = _append_amendment08_cache8(run_dir, runtime, monkeypatch)
    if event_name == "CACHE_LOAD":
        target = event8
        index = 8
    else:
        target = full._journal_event(
            run_dir,
            "TRAINING_ATTEMPT_STARTED",
            phase="TRAINING",
            variant_id="full_new_reference",
            attempt_id="b" * 32,
            **_amendment08_lineage_stub(),
        )
        index = 9
    tampered = {**target, field: "0" * 64}
    full._journal_event_path(run_dir, index).write_bytes(
        full.strict_full_json_bytes(tampered)
    )
    with pytest.raises(full.Amendment06IntegrityError):
        full._validate_amendment08_corrected_training_history(
            run_dir=run_dir,
            overlay=full.validate_amendment08_overlay(run_dir),
            allow_unterminated=True,
        )


def test_amendment08_training_start_fields_require_exact_next_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, runtime = _amendment08_isolated_journal_fixture(tmp_path, monkeypatch)
    _write_amendment08_raw_prefix(run_dir)
    _install_amendment08_semantic_view_fixture(run_dir, monkeypatch)
    _append_amendment08_cache8(run_dir, runtime, monkeypatch)
    expected_lineage = _amendment08_lineage_stub()
    expected_lineage["amendment08_journal_contamination_adjudication_sha256"] = (
        full.sha256_file(
            run_dir
            / "provenance/amendment08_journal_contamination_adjudication.json"
        )
    )
    assert full._amendment08_training_event_fields(
        run_dir=run_dir, variant="full_new_reference",
    ) == expected_lineage
    with pytest.raises(full.Amendment06IntegrityError, match="Variant order"):
        full._amendment08_training_event_fields(
            run_dir=run_dir, variant=full.TRAINED_VARIANTS[1],
        )


def test_amendment08_event7_and_precheck_use_reserved_non_circular_bindings() -> None:
    source = _amendment08_source_stub()
    package = _amendment08_package_stub()
    event7 = full._amendment08_event7_payload(
        recorded_at="2026-08-20T20:10:00-04:00",
        transaction_id="7" * 32,
        effective_manifest_sha256="e" * 64,
        corrigendum_sha256="c" * 64,
        adjudication_sha256="d" * 64,
        source=source,
        package=package,
    )
    assert set(event7) == {
        "event", "event_index", "run_id", "previous_event_sha256",
        "recorded_at", "transaction_id", "status", "authorization",
        "prompt_sha256", "base_scientific_execution_code_manifest_sha256",
        "effective_scientific_execution_code_manifest_sha256",
        "amendment08_code_corrigendum_sha256",
        "amendment08_journal_contamination_adjudication_sha256",
        "source_freeze_receipt_sha256",
        "current_a06_authorized_change_ledger_sha256",
        "a08_source_ledger_event_index",
        "a08_source_ledger_event_canonical_sha256",
        "current_a06_final_prerun_evidence_sha256",
        "a08_prelive_package_lineage_sha256",
        "a08_prelive_package_receipt_sha256", "cache_manifest_sha256",
        "contamination_event_indices", "contamination_event_raw_sha256",
        "contamination_event_canonical_sha256", "a07_overlay_installed",
        "a07_continuation_authorization_consumed",
    }
    assert event7["event_index"] == 7
    assert event7["previous_event_sha256"] == full.A08_CANONICAL_JOURNAL_SHA256[6]
    assert event7["recorded_at"] == "2026-08-20T20:10:00-04:00"
    assert event7["transaction_id"] == "7" * 32
    assert event7["contamination_event_indices"] == [5, 6]
    assert "installation_intent_sha256" not in event7
    assert "overlay_install_receipt_sha256" not in event7

    precheck = full._amendment08_precheck_payload(
        recorded_at="2026-08-20T20:10:01-04:00",
        transaction_id="7" * 32,
        event7=event7,
        effective_manifest_sha256="e" * 64,
        corrigendum_sha256="c" * 64,
        adjudication_sha256="d" * 64,
        source=source,
        package=package,
    )
    assert precheck["corrigendum_journal_event_index"] == 7
    assert precheck["raw_journal_event_count"] == 8
    assert precheck["semantic_journal_event_count"] == 6
    assert precheck["classified_test_contamination_event_indices"] == [5, 6]
    assert precheck["transaction_id"] == event7["transaction_id"]
    assert "installation_intent_sha256" not in precheck
    assert "overlay_install_receipt_sha256" not in precheck


def test_amendment08_target_lineage_never_routes_through_amendment07(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / full.A08_TARGET_RUN_ID
    receipt = run_dir / "provenance/amendment08_overlay_install_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b"installed\n")
    lineage = _amendment08_lineage_stub()
    overlay = {
        "status": "PASS",
        "authorization": full.A08_AUTHORIZATION,
        "prompt_sha256": full.A08_PROMPT_SHA256,
        **lineage,
    }
    monkeypatch.setattr(full, "A08_TARGET_RUN", run_dir)
    monkeypatch.setattr(full, "validate_amendment08_overlay", lambda path: overlay)
    monkeypatch.setattr(
        full, "_amendment08_frozen_lineage_bindings", lambda path: lineage,
    )
    monkeypatch.setattr(
        full, "validate_amendment07_overlay",
        lambda path: pytest.fail("Amendment 07 validator was invoked for A08 target"),
    )
    assert full._scientific_lineage_bindings(run_dir) == lineage


def test_amendment08_resume_dispatch_is_branch_specific(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        full, "amendment06_lock", lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(full, "assert_exact_invocation", lambda **kwargs: None)
    monkeypatch.setattr(
        full, "_amendment08_finalization_handoff_ready", lambda path: False,
    )
    monkeypatch.setattr(
        full, "validate_amendment08_pre_resume",
        lambda **kwargs: calls.append("a08_admission") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        full, "validate_amendment07_pre_resume",
        lambda **kwargs: pytest.fail("A08 target reached Amendment 07 admission"),
    )
    monkeypatch.setattr(full, "_validate_run_identity", lambda path: None)
    monkeypatch.setattr(
        full, "_execute_scientific_full",
        lambda **kwargs: calls.append("execute") or kwargs["run_dir"],
    )
    monkeypatch.setattr(
        full, "package_dispatch", lambda *args, **kwargs: {"status": "PASS"},
    )
    preflight = tmp_path / "preflight"
    smoke = tmp_path / "smoke"
    target = tmp_path / "results" / full.A08_TARGET_RUN_ID
    assert full.resume_full(
        target, preflight_run=preflight, smoke_run=smoke,
    ) == target.resolve()
    assert calls == ["a08_admission", "execute"]

    calls.clear()
    monkeypatch.setattr(
        full, "validate_amendment08_pre_resume",
        lambda **kwargs: pytest.fail("legacy Run reached Amendment 08 admission"),
    )
    monkeypatch.setattr(
        full, "validate_amendment07_pre_resume",
        lambda **kwargs: calls.append("a07_admission") or {"status": "PASS"},
    )
    legacy = tmp_path / "results" / "legacy_full_run"
    assert full.resume_full(
        legacy, preflight_run=preflight, smoke_run=smoke,
    ) == legacy.resolve()
    assert calls == ["a07_admission", "execute"]


def test_amendment08_model_metadata_emits_all_six_lineage_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / full.A08_TARGET_RUN_ID
    (run_dir / "provenance").mkdir(parents=True)
    (run_dir / "provenance/source_input_hashes_pre.tsv").write_bytes(b"source\n")
    model_path = run_dir / "model.ubj"
    model_path.write_bytes(b"model\n")
    lineage = _amendment08_lineage_stub()
    monkeypatch.setattr(
        full, "_scientific_lineage_bindings", lambda path: dict(lineage),
    )
    metadata = full._model_common_metadata(
        run_dir=run_dir,
        variant_data=SimpleNamespace(
            variant="full_new_reference",
            classification="OFFICIAL_RUNNABLE",
            features=("feature_a", "feature_b"),
        ),
        model_path=model_path,
        cache=SimpleNamespace(binding_sha256="f" * 64, imputer_sha256="i" * 64),
        gpu_identity=_valid_amendment07_gpu_identity(),
        gpu_name="NVIDIA GeForce RTX 4090",
        best_iteration=7,
        iteration_range=(0, 8),
    )
    assert {key: metadata[key] for key in full.A08_LINEAGE_FIELDS} == lineage
    assert metadata["scientific_execution_code_manifest_sha256"] == lineage[
        "effective_scientific_execution_code_manifest_sha256"
    ]


def test_amendment08_models_freeze_rejects_each_of_six_lineage_disagreements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / full.A08_TARGET_RUN_ID
    manifest_path = run_dir / "provenance/full_models_frozen_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    lineage = _amendment08_lineage_stub()
    model_records = [
        {
            "variant_id": variant,
            "variant_classification": (
                "EXPLORATORY"
                if variant == full.EXPLORATORY_VARIANT
                else "OFFICIAL_RUNNABLE"
            ),
            "model_sha256": f"{index + 1:x}" * 64,
            "training_completion_sha256": f"{index + 1:x}" * 64,
        }
        for index, variant in enumerate(full.TRAINED_VARIANTS)
    ]
    payload = {
        "run_id": run_dir.name,
        "run_kind": full.RUN_KIND,
        "phase": "FULL_MODELS_FROZEN",
        "status": "PASS",
        "frozen_at": "2026-08-20T22:00:00-04:00",
        "trained_model_count": 10,
        "official_runnable_model_count": 9,
        "exploratory_model_count": 1,
        "true_no_pca_model_count": 0,
        "frozen_population_sha256": "f" * 64,
        "imputer_sha256": "i" * 64,
        "full_parameter_lock_sha256": full.FULL_MODEL_PARAMETERS_SHA256,
        "split_assignment_sha256": full.SPLIT_ASSIGNMENT_SHA256,
        **lineage,
        "scientific_training_calls": 10,
        "scientific_fit_attempts": 11,
        "failed_unpublished_training_attempts": 1,
        "external_audit_opened": False,
        "models": model_records,
    }
    events = [
        *({"event": "SCIENTIFIC_FIT_CALLED"} for _ in range(11)),
        {"event": "TRAINING_ATTEMPT_FAILED"},
    ]
    monkeypatch.setattr(
        full, "_scientific_journal_events", lambda path: events,
    )
    monkeypatch.setattr(
        full, "_scientific_lineage_bindings", lambda path: dict(lineage),
    )
    monkeypatch.setattr(
        full,
        "_validate_amendment08_corrected_training_history",
        lambda **kwargs: {
            "completed_variants": list(full.TRAINED_VARIANTS),
            "next_variant": None,
        },
    )
    by_variant = {record["variant_id"]: record for record in model_records}
    monkeypatch.setattr(
        full,
        "validate_training_completion",
        lambda **kwargs: {
            "model_sha256": by_variant[kwargs["variant"]]["model_sha256"],
            "completion_sha256": by_variant[kwargs["variant"]][
                "training_completion_sha256"
            ],
        },
    )
    monkeypatch.setattr(
        full, "_ensure_unique_bound_journal_event", lambda **kwargs: {},
    )
    cache = SimpleNamespace(binding_sha256="f" * 64, imputer_sha256="i" * 64)
    full.atomic_write_strict_json(manifest_path, payload)
    assert full.validate_models_frozen_manifest(
        run_dir=run_dir,
        locked=SimpleNamespace(),
        cache=cache,
        development=SimpleNamespace(),
    ) == payload
    for field in full.A08_LINEAGE_FIELDS:
        tampered = copy.deepcopy(payload)
        tampered[field] = "0" * 64
        full.atomic_write_strict_json(manifest_path, tampered)
        with pytest.raises(full.Amendment06IntegrityError, match="schema/count"):
            full.validate_models_frozen_manifest(
                run_dir=run_dir,
                locked=SimpleNamespace(),
                cache=cache,
                development=SimpleNamespace(),
            )
    full.atomic_write_strict_json(manifest_path, payload)


def _append_amendment08_fixture_event(
    events: list[dict], event_name: str, **fields,
) -> dict:
    index = int(events[-1]["event_index"]) + 1
    event = {
        "event_index": index,
        "event": event_name,
        "run_id": full.A08_TARGET_RUN_ID,
        "previous_event_sha256": full.sha256_bytes(
            full.strict_full_canonical_json_bytes(events[-1])
        ),
        "recorded_at": (
            f"2026-08-20T22:{index // 60:02d}:{index % 60:02d}-04:00"
        ),
        **fields,
    }
    events.append(event)
    return event


def _amendment08_complete_ledger_fixture(
    run_dir: Path,
) -> tuple[list[dict], list[dict], dict]:
    lineage = _amendment08_lineage_stub()
    raw_events = [
        copy.deepcopy(event)
        for event in full._amendment08_expected_raw_prefix_events()
    ]
    event7 = full._amendment08_event7_payload(
        recorded_at="2026-08-20T22:00:07-04:00",
        transaction_id="7" * 32,
        effective_manifest_sha256=lineage[
            "effective_scientific_execution_code_manifest_sha256"
        ],
        corrigendum_sha256=lineage["amendment08_code_corrigendum_sha256"],
        adjudication_sha256=lineage[
            "amendment08_journal_contamination_adjudication_sha256"
        ],
        source=_amendment08_source_stub(),
        package=_amendment08_package_stub(),
    )
    raw_events.append(event7)
    _append_amendment08_fixture_event(
        raw_events,
        "CACHE_LOAD",
        status="PASS",
        cache_path=str(
            full.EXPECTED_RUNTIME_ROOT / "full_population" / run_dir.name
        ),
        cache_manifest_sha256=full.A07_CACHE_MANIFEST_SHA256,
        frozen_population_sha256=full.A07_FROZEN_POPULATION_SHA256,
        unique_realizations=1,
        augmentation_generation_count=1,
        safe_smote_generation_count=1,
        **lineage,
    )
    for sequence, variant in enumerate(full.TRAINED_VARIANTS, start=1):
        attempt_id = f"{sequence:032x}"
        _append_amendment08_fixture_event(
            raw_events,
            "TRAINING_ATTEMPT_STARTED",
            phase="TRAINING",
            variant_id=variant,
            attempt_id=attempt_id,
            **lineage,
        )
        _append_amendment08_fixture_event(
            raw_events,
            "SCIENTIFIC_FIT_CALLED",
            phase="TRAINING",
            variant_id=variant,
            attempt_id=attempt_id,
            seed=full.PRIMARY_MODEL_SEED,
            stability_fit=False,
            **lineage,
        )
        completion = run_dir / full._training_completion_relative(variant)
        completion.parent.mkdir(parents=True, exist_ok=True)
        completion.write_bytes(f"training:{variant}\n".encode("ascii"))
        _append_amendment08_fixture_event(
            raw_events,
            "TRAINING_COMPLETION_COMMITTED",
            phase="TRAINING",
            variant_id=variant,
            attempt_id=attempt_id,
            completion_sha256=full.sha256_file(completion),
            recovered_after_interruption=False,
        )
    frozen_path = run_dir / "provenance/full_models_frozen_manifest.json"
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_bytes(b"frozen-models\n")
    _append_amendment08_fixture_event(
        raw_events,
        "MODELS_FROZEN",
        status="PASS",
        model_count=10,
        manifest_sha256=full.sha256_file(frozen_path),
        recovered_after_interruption=False,
    )
    audit_path = run_dir / "provenance/external_audit_exclusion_and_opening.json"
    audit_path.write_bytes(b"external-audit-opening\n")
    _append_amendment08_fixture_event(
        raw_events,
        "EXTERNAL_AUDIT_OPENED",
        status="PASS",
        evidence_sha256=full.sha256_file(audit_path),
        recovered_after_interruption=False,
    )
    for sequence, variant in enumerate(full.TRAINED_VARIANTS, start=1):
        attempt_id = f"{sequence + 100:032x}"
        _append_amendment08_fixture_event(
            raw_events,
            "AUDIT_SCORING_ATTEMPT_STARTED",
            phase="AUDIT_SCORING",
            variant_id=variant,
            attempt_id=attempt_id,
        )
        completion = run_dir / full._scoring_completion_relative(variant)
        completion.parent.mkdir(parents=True, exist_ok=True)
        completion.write_bytes(f"scoring:{variant}\n".encode("ascii"))
        _append_amendment08_fixture_event(
            raw_events,
            "AUDIT_SCORING_COMPLETION_COMMITTED",
            phase="AUDIT_SCORING",
            variant_id=variant,
            attempt_id=attempt_id,
            completion_sha256=full.sha256_file(completion),
            recovered_after_interruption=False,
        )
    reporting_attempt_id = "f" * 32
    _append_amendment08_fixture_event(
        raw_events,
        "REPORTING_ATTEMPT_STARTED",
        phase="REPORTING",
        variant_id="GLOBAL",
        attempt_id=reporting_attempt_id,
    )
    bootstrap_path = run_dir / "statistics/paired_card_bootstrap.json"
    bootstrap_payload = {"schema": "fixture_bootstrap/v1", "status": "PASS"}
    full.atomic_write_strict_json(bootstrap_path, bootstrap_payload)
    _append_amendment08_fixture_event(
        raw_events,
        "BOOTSTRAP_RESAMPLING",
        phase="REPORTING",
        status="PASS",
        replicates=full.BOOTSTRAP_REPLICATES,
        seed=full.BOOTSTRAP_SEED,
        model_retraining=False,
        bootstrap_sha256=full.sha256_bytes(
            full.strict_full_canonical_json_bytes(bootstrap_payload)
        ),
    )
    reporting_path = run_dir / "provenance/reporting_completion_manifest.json"
    reporting_path.parent.mkdir(parents=True, exist_ok=True)
    reporting_path.write_bytes(b"genuine-reporting-completion\n")
    genuine = _append_amendment08_fixture_event(
        raw_events,
        "REPORTING_COMPLETION_COMMITTED",
        phase="REPORTING",
        variant_id="GLOBAL",
        attempt_id=reporting_attempt_id,
        completion_sha256=full.sha256_file(reporting_path),
        recovered_after_interruption=False,
    )
    semantic_events = [
        event for event in raw_events
        if event["event_index"] not in full.A08_CONTAMINATION_INDICES
    ]
    return raw_events, semantic_events, genuine


def _amendment08_cutoff_payload_fixture(
    run_dir: Path,
    raw_events: list[dict],
    semantic_events: list[dict],
    genuine: dict,
) -> dict:
    raw_lines = b"".join(
        full.strict_full_canonical_json_bytes(event) + b"\n"
        for event in raw_events
    )
    semantic_lines = b"".join(
        full.strict_full_canonical_json_bytes(event) + b"\n"
        for event in semantic_events
    )
    records = []
    for event in raw_events:
        index = int(event["event_index"])
        raw = full.strict_full_json_bytes(event)
        canonical = full.strict_full_canonical_json_bytes(event)
        records.append({
            "physical_event_index": index,
            "relative_path": f"{index:08d}.json",
            "raw_size_bytes": len(raw),
            "raw_sha256": full.sha256_bytes(raw),
            "canonical_sha256": full.sha256_bytes(canonical),
            "event": event,
            "raw_bytes_base64": base64.b64encode(raw).decode("ascii"),
        })
    previous_marker_sha256 = "9" * 64
    marker = {
        "run_id": run_dir.name,
        "run_kind": full.RUN_KIND,
        "phase": "FULL_SCIENTIFIC_COMPLETE",
        "phase_index": full.RUN_PHASES.index("FULL_SCIENTIFIC_COMPLETE"),
        "status": "PASS",
        "previous_phase": "FULL_EXTERNAL_AUDIT_SCORED",
        "previous_marker_sha256": previous_marker_sha256,
        "recorded_at": "2026-08-20T23:00:00-04:00",
    }
    lineage = _amendment08_lineage_stub()
    event7 = raw_events[7]
    genuine_raw = full.strict_full_json_bytes(genuine)
    receipt_path = run_dir / "provenance/amendment08_overlay_install_receipt.json"
    receipt_path.write_bytes(b"overlay-receipt\n")
    payload = {
        "schema": "amendment08_raw_journal_at_reporting_cutoff/v1",
        "status": "PASS",
        "authorization": full.A08_AUTHORIZATION,
        "prompt_sha256": full.A08_PROMPT_SHA256,
        "target_run_id": run_dir.name,
        "created_at": "2026-08-20T23:00:00-04:00",
        "record_count": len(records),
        "records": records,
        "raw_journal_event_count_at_reporting_cutoff": len(raw_events),
        "raw_journal_head_sha256_at_reporting_cutoff": full.sha256_bytes(
            full.strict_full_canonical_json_bytes(genuine)
        ),
        "raw_journal_canonical_json_lines_sha256_at_reporting_cutoff": (
            full.sha256_bytes(raw_lines)
        ),
        "semantic_journal_event_count_at_reporting_cutoff": len(semantic_events),
        "semantic_journal_canonical_json_lines_sha256_at_reporting_cutoff": (
            full.sha256_bytes(semantic_lines)
        ),
        "classified_test_contamination_event_count": 2,
        "classified_test_contamination_event_indices": [5, 6],
        "classified_test_contamination_raw_sha256": list(
            full.A08_RAW_JOURNAL_SHA256[5:7]
        ),
        "classified_test_contamination_canonical_sha256": list(
            full.A08_CANONICAL_JOURNAL_SHA256[5:7]
        ),
        "raw_reporting_completion_events": 3,
        "semantic_reporting_completion_events": 1,
        "corrigendum_journal_event_index": 7,
        "corrigendum_journal_event_raw_sha256": full.sha256_bytes(
            full.strict_full_json_bytes(event7)
        ),
        "corrigendum_journal_head_sha256": full.sha256_bytes(
            full.strict_full_canonical_json_bytes(event7)
        ),
        "amendment08_overlay_install_receipt_sha256": full.sha256_file(receipt_path),
        "genuine_reporting_completion_event_index": genuine["event_index"],
        "genuine_reporting_completion_event_raw_sha256": full.sha256_bytes(genuine_raw),
        "genuine_reporting_completion_event_canonical_sha256": full.sha256_bytes(
            full.strict_full_canonical_json_bytes(genuine)
        ),
        "genuine_reporting_completion_manifest_sha256": genuine[
            "completion_sha256"
        ],
        **{
            key: lineage[key]
            for key in full.A08_LINEAGE_FIELDS[:4]
        },
        "full_models_frozen_manifest_sha256": "1" * 64,
        "external_audit_opening_sha256": "2" * 64,
        "reporting_completion_manifest_sha256": genuine["completion_sha256"],
        "metric_recomputation_sha256": "3" * 64,
        "source_input_hashes_post_sha256": "4" * 64,
        "previous_state_marker_sha256": previous_marker_sha256,
        "planned_full_scientific_complete_marker_recorded_at": marker["recorded_at"],
        "planned_full_scientific_complete_marker": marker,
        "planned_full_scientific_complete_marker_sha256": full.sha256_bytes(
            full.strict_full_json_bytes(marker)
        ),
        "prerequisite_validation_status": "PASS_FROZEN_RECEIPTS_ONLY",
    }
    assert set(payload) == set(full.A08_RAW_JOURNAL_CUTOFF_KEYS)
    assert all(set(record) == set(full.A08_RAW_JOURNAL_RECORD_KEYS) for record in records)
    return payload


def test_amendment08_execution_ledger_binds_cutoff_views_and_six_lineages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", tmp_path / "runtime")
    run_dir = tmp_path / "result" / full.A08_TARGET_RUN_ID
    run_dir.mkdir(parents=True)
    raw_events, semantic_events, genuine = _amendment08_complete_ledger_fixture(
        run_dir
    )
    cutoff = _amendment08_cutoff_payload_fixture(
        run_dir, raw_events, semantic_events, genuine,
    )
    cutoff_path = run_dir / full.A08_RAW_JOURNAL_CUTOFF_RELATIVE
    full.atomic_write_strict_json(cutoff_path, cutoff)
    view = {
        "status": "PASS",
        "raw_events": raw_events,
        "semantic_events": semantic_events,
        "raw_journal_event_count": len(raw_events),
        "raw_journal_head_sha256": cutoff[
            "raw_journal_head_sha256_at_reporting_cutoff"
        ],
        "raw_journal_canonical_json_lines_sha256": cutoff[
            "raw_journal_canonical_json_lines_sha256_at_reporting_cutoff"
        ],
        "semantic_journal_event_count": len(semantic_events),
        "semantic_journal_canonical_json_lines_sha256": cutoff[
            "semantic_journal_canonical_json_lines_sha256_at_reporting_cutoff"
        ],
        "classified_test_contamination_event_count": 2,
        "classified_test_contamination_event_indices": [5, 6],
        "classified_test_contamination_raw_sha256": list(
            full.A08_RAW_JOURNAL_SHA256[5:7]
        ),
        "classified_test_contamination_canonical_sha256": list(
            full.A08_CANONICAL_JOURNAL_SHA256[5:7]
        ),
        "raw_reporting_completion_events": 3,
        "semantic_reporting_completion_events": 1,
    }
    lineage = _amendment08_lineage_stub()
    overlay = {
        "status": "PASS",
        "overlay_install_receipt_sha256": cutoff[
            "amendment08_overlay_install_receipt_sha256"
        ],
        **lineage,
    }
    monkeypatch.setattr(full, "read_amendment08_journal_views", lambda path: view)
    monkeypatch.setattr(full, "validate_amendment08_overlay", lambda path: overlay)
    monkeypatch.setattr(
        full, "_amendment08_frozen_lineage_bindings", lambda path: lineage,
    )
    monkeypatch.setattr(
        full, "validate_amendment08_raw_journal_cutoff", lambda path: cutoff,
    )
    ledger = full.build_execution_ledger(run_dir)
    assert set(full.A08_EXECUTION_LEDGER_ADDED_FIELDS).issubset(ledger)
    assert {field: ledger[field] for field in full.A08_LINEAGE_FIELDS} == lineage
    assert ledger["amendment08_raw_journal_at_reporting_cutoff_sha256"] == (
        full.sha256_file(cutoff_path)
    )
    for field in (
        "raw_journal_event_count_at_reporting_cutoff",
        "raw_journal_head_sha256_at_reporting_cutoff",
        "raw_journal_canonical_json_lines_sha256_at_reporting_cutoff",
        "semantic_journal_event_count_at_reporting_cutoff",
        "semantic_journal_canonical_json_lines_sha256_at_reporting_cutoff",
        "classified_test_contamination_event_count",
        "classified_test_contamination_event_indices",
        "classified_test_contamination_raw_sha256",
        "classified_test_contamination_canonical_sha256",
        "raw_reporting_completion_events",
        "semantic_reporting_completion_events",
    ):
        assert ledger[field] == cutoff[field]
    assert ledger["scientific_fit_attempts"] == 11
    assert ledger["failed_unpublished_training_attempts"] == 1
    assert ledger["cache_generation_events"] == 1
    assert ledger["cache_load_events"] == 1
    for field in full.A08_LINEAGE_FIELDS:
        event = next(
            candidate for candidate in semantic_events
            if candidate["event"] == "SCIENTIFIC_FIT_CALLED"
            and candidate["event_index"] > 8
        )
        original = event[field]
        event[field] = "0" * 64
        with pytest.raises(full.Amendment06IntegrityError):
            full.build_execution_ledger(run_dir)
        event[field] = original


@pytest.mark.parametrize(
    "tamper",
    [
        "bootstrap_hash",
        "bootstrap_before_reporting_start",
        "orphan_reporting_failure",
        "duplicate_reporting_terminal",
        "blank_scoring_failure",
        "freeze_extra_field",
        "audit_recovery_type",
    ],
)
def test_amendment08_cutoff_lifecycle_rejects_reporting_and_scoring_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    run_dir = tmp_path / full.A08_TARGET_RUN_ID
    run_dir.mkdir()
    raw_events, _, _ = _amendment08_complete_ledger_fixture(run_dir)
    lineage = _amendment08_lineage_stub()
    baseline = full._validate_amendment08_cutoff_lifecycle(
        run_dir=run_dir,
        events=raw_events,
        lineage=lineage,
    )
    assert baseline["training_fit_count"] == 11

    bootstrap = next(
        event for event in raw_events if event["event"] == "BOOTSTRAP_RESAMPLING"
    )
    reporting_start = next(
        event for event in raw_events if event["event"] == "REPORTING_ATTEMPT_STARTED"
    )
    reporting_completion = next(
        event
        for event in raw_events
        if event["event"] == "REPORTING_COMPLETION_COMMITTED"
        and event["event_index"] > 7
    )
    if tamper == "bootstrap_hash":
        bootstrap["bootstrap_sha256"] = "0" * 64
    elif tamper == "bootstrap_before_reporting_start":
        bootstrap["event_index"] = int(reporting_start["event_index"]) - 1
    elif tamper in {"orphan_reporting_failure", "duplicate_reporting_terminal"}:
        failure = {
            "event_index": int(reporting_completion["event_index"]),
            "event": "REPORTING_ATTEMPT_FAILED",
            "run_id": run_dir.name,
            "previous_event_sha256": reporting_completion["previous_event_sha256"],
            "recorded_at": reporting_completion["recorded_at"],
            "phase": "REPORTING",
            "variant_id": "GLOBAL",
            "attempt_id": (
                "0" * 32
                if tamper == "orphan_reporting_failure"
                else reporting_start["attempt_id"]
            ),
            "error_type": "RuntimeError",
            "error": "synthetic reporting failure",
        }
        reporting_completion["event_index"] = int(reporting_completion["event_index"]) + 1
        raw_events.insert(raw_events.index(reporting_completion), failure)
    elif tamper == "blank_scoring_failure":
        scoring_completion = next(
            event
            for event in raw_events
            if event["event"] == "AUDIT_SCORING_COMPLETION_COMMITTED"
        )
        scoring_completion.clear()
        scoring_completion.update({
            "event_index": 42,
            "event": "AUDIT_SCORING_ATTEMPT_FAILED",
            "run_id": run_dir.name,
            "previous_event_sha256": "0" * 64,
            "recorded_at": "2026-08-20T22:00:42-04:00",
            "phase": "AUDIT_SCORING",
            "variant_id": full.TRAINED_VARIANTS[0],
            "attempt_id": f"{101:032x}",
            "error_type": "",
            "error": "synthetic scoring failure",
        })
    elif tamper == "freeze_extra_field":
        next(
            event for event in raw_events if event["event"] == "MODELS_FROZEN"
        )["unexpected"] = True
    else:
        next(
            event for event in raw_events if event["event"] == "EXTERNAL_AUDIT_OPENED"
        )["recovered_after_interruption"] = "False"

    with pytest.raises(full.Amendment06IntegrityError):
        full._validate_amendment08_cutoff_lifecycle(
            run_dir=run_dir,
            events=raw_events,
            lineage=lineage,
        )


def test_amendment08_cutoff_build_publish_reopen_and_tamper_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, runtime = _amendment08_isolated_journal_fixture(tmp_path, monkeypatch)
    _write_amendment08_raw_prefix(run_dir)
    _install_amendment08_semantic_view_fixture(run_dir, monkeypatch)
    _append_amendment08_cache8(run_dir, runtime, monkeypatch)
    full._journal_event(
        run_dir,
        "TRAINING_ATTEMPT_STARTED",
        phase="TRAINING",
        variant_id="full_new_reference",
        attempt_id="b" * 32,
        **_amendment08_lineage_stub(),
    )
    reporting_path = run_dir / "provenance/reporting_completion_manifest.json"
    reporting_path.write_bytes(b"genuine-reporting-completion\n")
    genuine = full._ensure_unique_bound_journal_event(
        run_dir=run_dir,
        event_name="REPORTING_COMPLETION_COMMITTED",
        binding_path=reporting_path,
        binding_field="completion_sha256",
        phase="REPORTING",
        variant="GLOBAL",
        recovered=False,
        extra={"attempt_id": "f" * 32},
    )
    previous_marker = full._state_marker_path(
        run_dir, "FULL_EXTERNAL_AUDIT_SCORED",
    )
    previous_marker.parent.mkdir(parents=True, exist_ok=True)
    previous_marker.write_bytes(b"frozen-previous-state\n")
    lineage = _amendment08_lineage_stub()
    prerequisites = {
        "status": "PASS_FROZEN_RECEIPTS_ONLY",
        **lineage,
        "full_models_frozen_manifest_sha256": "1" * 64,
        "external_audit_opening_sha256": "2" * 64,
        "reporting_completion_manifest_sha256": full.sha256_file(reporting_path),
        "metric_recomputation_sha256": "3" * 64,
        "source_input_hashes_post_sha256": "4" * 64,
    }
    monkeypatch.setattr(
        full,
        "_validate_amendment08_frozen_prerequisite_receipts",
        lambda path: prerequisites,
    )
    monkeypatch.setattr(
        full, "_amendment08_frozen_lineage_bindings", lambda path: lineage,
    )
    monkeypatch.setattr(
        full,
        "_validate_amendment08_corrected_training_history",
        lambda **kwargs: {
            "completed_variants": list(full.TRAINED_VARIANTS),
            "next_variant": None,
        },
    )
    monkeypatch.setattr(
        full,
        "_validate_amendment08_cutoff_lifecycle",
        lambda **kwargs: {"status": "PASS"},
    )
    recorded_at = "2026-08-20T23:30:00-04:00"
    built = full.build_amendment08_raw_journal_cutoff(
        run_dir, final_state_recorded_at=recorded_at,
    )
    cutoff_path = run_dir / full.A08_RAW_JOURNAL_CUTOFF_RELATIVE
    assert not cutoff_path.exists()
    assert set(built) == set(full.A08_RAW_JOURNAL_CUTOFF_KEYS)
    assert built["record_count"] == genuine["event_index"] + 1
    assert built["raw_reporting_completion_events"] == 3
    assert built["semantic_reporting_completion_events"] == 1
    assert built["planned_full_scientific_complete_marker_recorded_at"] == recorded_at
    for record in built["records"]:
        assert set(record) == set(full.A08_RAW_JOURNAL_RECORD_KEYS)
        raw = base64.b64decode(record["raw_bytes_base64"], validate=True)
        assert raw == full._journal_event_path(
            run_dir, record["physical_event_index"],
        ).read_bytes()
        assert full.sha256_bytes(raw) == record["raw_sha256"]

    published = full._publish_or_validate_amendment08_raw_journal_cutoff(
        run_dir, final_state_recorded_at=recorded_at,
    )
    assert published == built
    assert full.validate_amendment08_raw_journal_cutoff(run_dir) == built
    original_bytes = cutoff_path.read_bytes()
    event8_path = full._journal_event_path(run_dir, 8)
    event8_bytes = event8_path.read_bytes()
    event8_path.write_bytes(event8_bytes + b" ")
    with pytest.raises(full.Amendment06IntegrityError):
        full.validate_amendment08_raw_journal_cutoff(run_dir)
    event8_path.write_bytes(event8_bytes)
    assert full.validate_amendment08_raw_journal_cutoff(run_dir) == built
    for tamper in (
        "base64", "raw_sha256", "event", "order",
        "semantic_hash", "marker_hash", "receipt_hash",
    ):
        changed = copy.deepcopy(built)
        if tamper == "base64":
            changed["records"][0]["raw_bytes_base64"] = "!invalid-base64!"
        elif tamper == "raw_sha256":
            changed["records"][0]["raw_sha256"] = "0" * 64
        elif tamper == "event":
            changed["records"][-1]["event"]["completion_sha256"] = "0" * 64
        elif tamper == "order":
            changed["records"][8], changed["records"][9] = (
                changed["records"][9], changed["records"][8]
            )
        elif tamper == "semantic_hash":
            changed[
                "semantic_journal_canonical_json_lines_sha256_at_reporting_cutoff"
            ] = "0" * 64
        elif tamper == "marker_hash":
            changed["planned_full_scientific_complete_marker_sha256"] = "0" * 64
        else:
            changed["amendment08_overlay_install_receipt_sha256"] = "0" * 64
        full.atomic_write_strict_json(cutoff_path, changed)
        with pytest.raises(full.Amendment06IntegrityError):
            full.validate_amendment08_raw_journal_cutoff(run_dir)
        cutoff_path.write_bytes(original_bytes)
    assert full.validate_amendment08_raw_journal_cutoff(run_dir) == built

    marker_path = full._state_marker_path(run_dir, "FULL_SCIENTIFIC_COMPLETE")
    marker_bytes = full.strict_full_json_bytes(
        built["planned_full_scientific_complete_marker"]
    )
    marker_path.write_bytes(marker_bytes)
    cutoff_path.unlink()
    reconstructed = full._publish_or_validate_amendment08_raw_journal_cutoff(
        run_dir
    )
    assert reconstructed == built
    assert cutoff_path.read_bytes() == original_bytes
    assert marker_path.read_bytes() == marker_bytes


def test_amendment08_finalization_crashes_reenter_to_identical_durable_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    crash_boundaries = (
        "prerequisites", "cutoff", "state", "ledger", "status", "blockers",
        "strict_json", "output_manifest",
    )
    science_calls: list[str] = []

    def forbidden_science(name: str):
        def fail(*args, **kwargs):
            science_calls.append(name)
            raise AssertionError(f"finalization reentry called scientific API: {name}")

        return fail

    for name in (
        "initialize_run_evidence",
        "load_locked_inputs",
        "load_development_population",
        "load_or_build_population_cache",
        "train_one_variant",
        "freeze_all_models",
        "validate_models_frozen_manifest",
        "open_external_audit_after_freeze",
        "score_one_external_variant",
        "complete_external_audit_phase",
        "build_reports_and_bootstrap",
        "_validate_reporting_completion",
        "build_metric_recomputation",
        "_snapshot_source_post",
        "_recompute_and_validate_reporting_outputs",
        "_binary_metrics",
    ):
        monkeypatch.setattr(full, name, forbidden_science(name))

    final_snapshots: list[dict[str, bytes]] = []
    for crash_index, crash_boundary in enumerate(crash_boundaries):
        monkeypatch.setattr(
            full, "EXPECTED_RUNTIME_ROOT", tmp_path / crash_boundary / "runtime",
        )
        run_dir = (
            tmp_path / crash_boundary / "results" / full.A08_TARGET_RUN_ID
        )
        provenance = run_dir / "provenance"
        provenance.mkdir(parents=True)
        (provenance / "metric_recomputation.json").write_bytes(b"{}\n")
        (provenance / "source_input_hashes_post.tsv").write_bytes(b"source\n")
        previous_marker_path = full._state_marker_path(
            run_dir, "FULL_EXTERNAL_AUDIT_SCORED",
        )
        previous_marker_path.parent.mkdir(parents=True, exist_ok=True)
        previous_marker_path.write_bytes(b"previous-state-marker\n")
        initial_status = full._initial_status(run_dir.name).encode("utf-8")
        (run_dir / "RUN_STATUS.txt").write_bytes(initial_status)

        marker = {
            "run_id": run_dir.name,
            "run_kind": full.RUN_KIND,
            "phase": "FULL_SCIENTIFIC_COMPLETE",
            "phase_index": full.RUN_PHASES.index("FULL_SCIENTIFIC_COMPLETE"),
            "status": "PASS",
            "previous_phase": "FULL_EXTERNAL_AUDIT_SCORED",
            "previous_marker_sha256": full.sha256_file(previous_marker_path),
            "recorded_at": "2026-08-20T23:59:00-04:00",
        }
        cutoff = {
            "schema": "amendment08_raw_journal_at_reporting_cutoff/v1",
            "status": "PASS",
            "planned_full_scientific_complete_marker": marker,
        }
        cutoff_path = run_dir / full.A08_RAW_JOURNAL_CUTOFF_RELATIVE
        marker_path = full._state_marker_path(
            run_dir, "FULL_SCIENTIFIC_COMPLETE",
        )
        ledger_path = provenance / "execution_ledger.json"
        strict_path = provenance / "strict_json_validation.json"
        manifest_path = run_dir / "OUTPUT_MANIFEST_FINAL.tsv"
        cutoff_bytes = full.strict_full_json_bytes(cutoff)
        marker_bytes = full.strict_full_json_bytes(marker)
        cutoff_sha256 = full.sha256_bytes(cutoff_bytes)
        ledger = {
            "schema": "amendment08_execution_ledger_fixture/v1",
            "status": "PASS",
            "amendment08_raw_journal_cutoff_sha256": cutoff_sha256,
            "amendment08_raw_journal_at_reporting_cutoff_sha256": cutoff_sha256,
        }
        ledger_bytes = full.strict_full_json_bytes(ledger)
        strict_evidence = {
            "status": "PASS",
            "validated_json_files": 237,
            "run_id": run_dir.name,
            "self_excluded_by_non_circular_contract": True,
            "new_full_json_nonfinite_constants": 0,
        }
        strict_bytes = full.strict_full_json_bytes(strict_evidence)
        manifest_bytes = b"deterministic-output-manifest\n"
        desired = {
            full.A08_RAW_JOURNAL_CUTOFF_RELATIVE: cutoff_bytes,
            marker_path.relative_to(run_dir).as_posix(): marker_bytes,
            "provenance/execution_ledger.json": ledger_bytes,
            "RUN_STATUS.txt": full._final_status(run_dir).encode("utf-8"),
            "BLOCKERS.md": b"# Blockers\n\nNone.\n",
            "provenance/strict_json_validation.json": strict_bytes,
            "OUTPUT_MANIFEST_FINAL.tsv": manifest_bytes,
        }

        def publish_cutoff(path: Path) -> dict:
            assert Path(path).resolve() == run_dir.resolve()
            if cutoff_path.exists():
                assert cutoff_path.read_bytes() == cutoff_bytes
            else:
                full.publish_bytes_no_clobber(cutoff_path, cutoff_bytes)
            return copy.deepcopy(cutoff)

        def validate_cutoff(path: Path) -> dict:
            assert Path(path).resolve() == run_dir.resolve()
            assert cutoff_path.read_bytes() == cutoff_bytes
            return copy.deepcopy(cutoff)

        def current_state(path: Path) -> str:
            assert Path(path).resolve() == run_dir.resolve()
            return (
                "FULL_SCIENTIFIC_COMPLETE"
                if marker_path.exists()
                else "FULL_EXTERNAL_AUDIT_SCORED"
            )

        def validate_final(path: Path) -> dict:
            assert Path(path).resolve() == run_dir.resolve()
            for relative, expected_bytes in desired.items():
                assert (run_dir / relative).read_bytes() == expected_bytes
            return {
                "status": "PASS",
                "sha256": full.sha256_bytes(manifest_bytes),
                "rows": 236,
            }

        monkeypatch.setattr(
            full, "_validate_amendment08_finalization_identity",
            lambda path: {"status": "PASS"},
        )
        monkeypatch.setattr(full, "_current_run_state", current_state)
        monkeypatch.setattr(
            full, "_validate_metric_recomputation_receipt",
            lambda path: {"status": "PASS"},
        )
        monkeypatch.setattr(
            full, "_validate_amendment08_frozen_prerequisite_receipts",
            lambda path: {"status": "PASS_FROZEN_RECEIPTS_ONLY"},
        )
        monkeypatch.setattr(
            full, "_publish_or_validate_amendment08_raw_journal_cutoff",
            publish_cutoff,
        )
        monkeypatch.setattr(
            full, "validate_amendment08_raw_journal_cutoff", validate_cutoff,
        )
        monkeypatch.setattr(full, "build_execution_ledger", lambda path: ledger)
        monkeypatch.setattr(
            full, "validate_strict_json_tree",
            lambda path, include: {"status": "PASS", "validated_json_files": 237},
        )
        monkeypatch.setattr(
            full, "output_manifest_bytes",
            lambda path, require_exact_contract: manifest_bytes,
        )
        monkeypatch.setattr(full, "validate_final_scientific_run", validate_final)
        monkeypatch.setattr(full, "_progress", lambda *args, **kwargs: None)

        with pytest.raises(
            full.Amendment06ResumeError,
            match=f"CRASH_AFTER_{crash_boundary.upper()}",
        ):
            full.finalize_scientific_run(
                run_dir=run_dir, audit=None, crash_after=crash_boundary,
            )

        published_count = crash_index
        for artifact_index, (relative, expected_bytes) in enumerate(
            desired.items()
        ):
            path = run_dir / relative
            if artifact_index < published_count:
                assert path.read_bytes() == expected_bytes
            elif relative == "RUN_STATUS.txt":
                assert path.read_bytes() == initial_status
            else:
                assert not path.exists()

        assert full.finalize_scientific_run(run_dir=run_dir, audit=None)[
            "status"
        ] == "PASS"
        final_snapshots.append({
            relative: (run_dir / relative).read_bytes()
            for relative in desired
        })

    assert science_calls == []
    assert all(snapshot == final_snapshots[0] for snapshot in final_snapshots[1:])


def test_amendment08_finalization_handoff_dispatches_before_scientific_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "results" / full.A08_TARGET_RUN_ID
    run_dir.mkdir(parents=True)
    calls: list[str] = []
    monkeypatch.setattr(
        full, "_amendment08_finalization_handoff_ready", lambda path: True,
    )
    monkeypatch.setattr(
        full, "_validate_amendment08_finalization_identity",
        lambda path: calls.append("identity") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        full, "finalize_scientific_run",
        lambda **kwargs: calls.append("finalize") or {"status": "PASS"},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("scientific setup ran after durable finalization handoff")

    for name in (
        "_validate_run_identity", "initialize_run_evidence", "load_locked_inputs",
        "load_development_population", "load_or_build_population_cache",
        "train_one_variant", "open_external_audit_after_freeze",
        "score_one_external_variant", "build_reports_and_bootstrap",
    ):
        monkeypatch.setattr(full, name, forbidden)
    assert full._execute_scientific_full(
        run_dir=run_dir,
        preflight_run=tmp_path / "preflight",
        smoke_run=tmp_path / "smoke",
    ) == run_dir.resolve()
    assert calls == ["identity", "finalize"]


def test_amendment08_overlay_reconciles_event7_to_receipt_crash_without_reappend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_overlay = full.validate_amendment08_overlay
    run_dir, _ = _amendment08_isolated_journal_fixture(tmp_path, monkeypatch)
    _write_amendment08_raw_prefix(run_dir)
    a08_runtime = tmp_path / "a08_overlay"
    intent_path = a08_runtime / "authorization/amendment08_overlay_install_intent.json"
    monkeypatch.setattr(full, "A08_INSTALL_INTENT", intent_path)
    source = _amendment08_source_stub()
    package = _amendment08_package_stub()
    monkeypatch.setattr(
        full, "validate_amendment08_source_freeze", lambda: source,
    )
    monkeypatch.setattr(
        full, "validate_amendment08_prelive_package_lineage", lambda path: package,
    )
    monkeypatch.setattr(
        full, "validate_amendment08_precorrection", lambda path: {"status": "PASS"},
    )
    monkeypatch.setattr(
        full, "_validate_preserved_target_cache_files",
        lambda path: {"cache_manifest_sha256": full.A07_CACHE_MANIFEST_SHA256},
    )
    monkeypatch.setattr(
        full, "validate_amendment08_overlay",
        lambda path=run_dir: {"status": "PASS", "run_id": Path(path).name},
    )
    generated_at = "2026-08-20T20:20:00-04:00"
    event7_recorded_at = "2026-08-20T20:20:01-04:00"
    transaction_id = "7" * 32
    times = iter((generated_at, event7_recorded_at))
    monkeypatch.setattr(full, "_now", lambda: next(times))
    monkeypatch.setattr(full.uuid, "uuid4", lambda: SimpleNamespace(hex=transaction_id))
    probe = tmp_path / "probe.json"
    probe.write_bytes(b"probe\n")
    monkeypatch.setattr(
        full, "_amendment08_probe_for_install_retry",
        lambda **kwargs: probe,
    )
    material = {
        relative: f"fixture:{relative}\n".encode("utf-8")
        for relative in full.A08_OVERLAY_RUN_RELATIVES
        if relative != "provenance/amendment08_overlay_install_receipt.json"
    }
    effective_sha256 = full.sha256_bytes(material[
        "provenance/scientific_execution_code_manifest_effective_amendment08.tsv"
    ])
    corrigendum_sha256 = full.sha256_bytes(
        material["provenance/amendment08_code_corrigendum.json"]
    )
    adjudication_sha256 = full.sha256_bytes(
        material["provenance/amendment08_journal_contamination_adjudication.json"]
    )
    event7 = full._amendment08_event7_payload(
        recorded_at=event7_recorded_at,
        transaction_id=transaction_id,
        effective_manifest_sha256=effective_sha256,
        corrigendum_sha256=corrigendum_sha256,
        adjudication_sha256=adjudication_sha256,
        source=source,
        package=package,
    )
    bindings = {
        "effective_manifest_sha256": effective_sha256,
        "corrigendum_sha256": corrigendum_sha256,
        "adjudication_sha256": adjudication_sha256,
        "event7": event7,
        "precheck": {"status": "PASS"},
    }
    monkeypatch.setattr(
        full, "_build_amendment08_overlay_material",
        lambda **kwargs: (material, bindings),
    )
    publish = full._publish_or_validate_amendment08_member
    crash_enabled = True

    def crash_before_run_precheck(**kwargs) -> None:
        nonlocal crash_enabled
        if (
            crash_enabled
            and kwargs.get("source_scope", "RUN") == "RUN"
            and kwargs["relative"] == "provenance/amendment08_resume_precheck.json"
        ):
            crash_enabled = False
            raise RuntimeError("synthetic crash after event 7")
        publish(**kwargs)

    monkeypatch.setattr(
        full, "_publish_or_validate_amendment08_member", crash_before_run_precheck,
    )
    with pytest.raises(RuntimeError, match="after event 7"):
        full.install_amendment08_overlay(probe_path=probe, run_dir=run_dir)
    event7_path = full._journal_event_path(run_dir, 7)
    intent_bytes = intent_path.read_bytes()
    event7_bytes = event7_path.read_bytes()
    assert full._read_journal(run_dir)[7] == event7
    assert not (run_dir / "provenance/amendment08_resume_precheck.json").exists()
    assert not (
        run_dir / "provenance/amendment08_overlay_install_receipt.json"
    ).exists()

    result = full.install_amendment08_overlay(probe_path=probe, run_dir=run_dir)
    assert result["status"] == "PASS"
    assert intent_path.read_bytes() == intent_bytes
    assert event7_path.read_bytes() == event7_bytes
    assert len(full._read_journal(run_dir)) == 8
    assert (run_dir / "provenance/amendment08_resume_precheck.json").is_file()
    assert (
        run_dir / "provenance/amendment08_overlay_install_receipt.json"
    ).is_file()

    monkeypatch.setattr(full, "validate_amendment08_overlay", validate_overlay)
    monkeypatch.setattr(
        full, "_validate_amendment08_base_history", lambda path: {"status": "PASS"},
    )
    monkeypatch.setattr(
        full, "_validate_amendment08_effective_manifest",
        lambda path: {"effective_manifest_sha256": effective_sha256},
    )
    monkeypatch.setattr(
        full, "read_amendment08_journal_views", lambda path: {"status": "PASS"},
    )
    assert full.validate_amendment08_overlay(run_dir)["status"] == "PASS"

    adjudication_path = (
        run_dir / "provenance/amendment08_journal_contamination_adjudication.json"
    )
    adjudication_bytes = adjudication_path.read_bytes()
    adjudication_path.write_bytes(adjudication_bytes + b" ")
    with pytest.raises(full.Amendment06IntegrityError, match="overlay member drifted"):
        full.validate_amendment08_overlay(run_dir)
    adjudication_path.write_bytes(adjudication_bytes)

    receipt_path = run_dir / "provenance/amendment08_overlay_install_receipt.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = full.strict_full_loads(receipt_bytes)
    receipt["installed_at"] = "2026-08-20T20:20:03-04:00"
    receipt_path.write_bytes(full.strict_full_json_bytes(receipt))
    with pytest.raises(full.Amendment06IntegrityError, match="receipt differs"):
        full.validate_amendment08_overlay(run_dir)
    receipt_path.write_bytes(receipt_bytes)

    changed_event7 = {**event7, "recorded_at": "2026-08-20T20:20:02-04:00"}
    with pytest.raises(full.Amendment06IntegrityError, match="event 7 differs"):
        full._append_or_validate_exact_journal_event(run_dir, changed_event7)


def _amendment08_publish_or_validate_rehearsal_file(
    path: Path, data: bytes, *, mode: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.parent.resolve(strict=True) != path.parent:
        raise AssertionError(f"unsafe rehearsal fixture parent: {path.parent}")
    digest = full.sha256_bytes(data)
    if path.exists():
        assert full._safe_regular(path, digest)
        assert path.stat().st_size == len(data)
    else:
        full.publish_bytes_no_clobber(path, data)
    os.chmod(path, mode)
    assert stat.S_IMODE(os.lstat(path).st_mode) == mode


def _amendment08_extract_true_rehearsal_fixture(
    *, run_dir: Path, runtime_root: Path, a06_runtime: Path,
) -> None:
    forensic = full._validate_amendment08_forensic_snapshot()
    assert forensic["status"] == "PASS"
    destinations = {
        "amendment08_pre_mutation_forensic/target_run/": run_dir,
        "amendment08_pre_mutation_forensic/raw_journal/": (
            runtime_root / "journals" / full.A08_TARGET_RUN_ID
        ),
        "amendment08_pre_mutation_forensic/a06_runtime/": a06_runtime,
    }
    expected_counts = {
        "amendment08_pre_mutation_forensic/target_run/": 45,
        "amendment08_pre_mutation_forensic/raw_journal/": 7,
        "amendment08_pre_mutation_forensic/a06_runtime/": 5,
    }
    observed = dict.fromkeys(destinations, 0)
    with zipfile.ZipFile(full.A08_FORENSIC_ZIP, "r") as archive:
        infos = archive.infolist()
        assert archive.testzip() is None
        assert len(infos) == 81
        assert len({info.filename for info in infos}) == len(infos)
        for info in infos:
            matched = next(
                (prefix for prefix in destinations if info.filename.startswith(prefix)),
                None,
            )
            if matched is None:
                continue
            relative = info.filename[len(matched):]
            relative_path = Path(relative)
            raw_mode = (info.external_attr >> 16) & 0xFFFF
            assert (
                relative
                and not relative_path.is_absolute()
                and ".." not in relative_path.parts
                and relative_path.as_posix() == relative
                and not info.is_dir()
                and not (info.flag_bits & 0x1)
                and (not raw_mode or stat.S_ISREG(raw_mode))
            )
            data = archive.read(info)
            assert len(data) == info.file_size
            _amendment08_publish_or_validate_rehearsal_file(
                destinations[matched] / relative_path,
                data,
                mode=stat.S_IMODE(raw_mode) if raw_mode else 0o600,
            )
            observed[matched] += 1
    assert observed == expected_counts


def _amendment08_clone_true_rehearsal_cache(
    *, run_dir: Path, runtime_root: Path,
) -> dict[str, object]:
    source = (
        full.PRODUCTION_EXPECTED_RUNTIME_ROOT
        / "full_population" / full.A08_TARGET_RUN_ID
    )
    destination = runtime_root / "full_population" / run_dir.name
    assert source.is_dir() and not source.is_symlink()
    destination.mkdir(parents=True, exist_ok=True)
    assert destination.is_dir() and not destination.is_symlink()
    manifest_path = source / "cache_manifest.json"
    assert full._safe_regular(manifest_path, full.A07_CACHE_MANIFEST_SHA256)
    manifest = full.strict_full_load_file(manifest_path)
    cache_files = manifest["cache_files"]
    assert isinstance(cache_files, list) and len(cache_files) == 10
    expected: dict[str, tuple[int, str]] = {
        "cache_manifest.json": (
            manifest_path.stat().st_size, full.A07_CACHE_MANIFEST_SHA256,
        ),
    }
    for item in cache_files:
        assert set(item) == {"relative_path", "size_bytes", "sha256"}
        relative = str(item["relative_path"])
        assert relative and len(Path(relative).parts) == 1 and relative not in expected
        expected[relative] = (int(item["size_bytes"]), str(item["sha256"]))
    assert len(expected) == 11
    assert sum(
        size for name, (size, _) in expected.items()
        if name != "cache_manifest.json"
    ) == full.A08_CACHE_MEMBER_BYTES
    assert {path.name for path in source.iterdir()} == set(expected)

    records: list[dict[str, object]] = []
    for relative in sorted(expected, key=lambda value: value.encode("utf-8")):
        size, digest = expected[relative]
        source_path = source / relative
        clone_path = destination / relative
        source_stat = os.lstat(source_path)
        assert stat.S_ISREG(source_stat.st_mode) and not source_path.is_symlink()
        assert source_stat.st_size == size and full.sha256_file(source_path) == digest
        temporary = destination / f".{relative}.copying"
        if not clone_path.exists():
            if temporary.exists():
                assert full._safe_regular(temporary, digest)
                assert temporary.stat().st_size == size
            else:
                shutil.copy2(source_path, temporary, follow_symlinks=False)
                with temporary.open("rb+") as handle:
                    os.fsync(handle.fileno())
                assert full._safe_regular(temporary, digest)
                assert temporary.stat().st_size == size
            os.rename(temporary, clone_path)
            full._fsync_parent(clone_path)
        elif temporary.exists():
            assert full._safe_regular(temporary, digest)
            temporary.unlink()
            full._fsync_parent(clone_path)
        clone_stat = os.lstat(clone_path)
        assert stat.S_ISREG(clone_stat.st_mode) and not clone_path.is_symlink()
        assert clone_stat.st_size == size and full.sha256_file(clone_path) == digest
        assert (source_stat.st_dev, source_stat.st_ino) != (
            clone_stat.st_dev, clone_stat.st_ino,
        )
        records.append({
            "relative_path": relative, "size_bytes": size, "sha256": digest,
            "source_device": int(source_stat.st_dev),
            "source_inode": int(source_stat.st_ino),
            "clone_device": int(clone_stat.st_dev),
            "clone_inode": int(clone_stat.st_ino),
        })
    assert {path.name for path in destination.iterdir()} == set(expected)
    inventory_sha256 = full.sha256_bytes(
        full._amendment08_cache_inventory_bytes(records)
    )
    return {
        "status": "PASS", "records": records,
        "source_inventory_sha256": inventory_sha256,
        "clone_inventory_sha256": inventory_sha256,
        "distinct_inode_count": len(records),
    }


def _amendment08_patch_true_rehearsal_roots(
    *, monkeypatch: pytest.MonkeyPatch, isolated_root: Path,
) -> tuple[Path, Path, Path, Path]:
    root = isolated_root.resolve(strict=True)
    results = root / "results"
    runtime = root / "runtime"
    a06_runtime = root / "a06_runtime"
    a08_runtime = root / "a08_runtime"
    run_dir = results / full.A08_TARGET_RUN_ID
    for directory in (results, runtime, a06_runtime, a08_runtime):
        directory.mkdir(parents=True, exist_ok=True)
        assert directory.resolve(strict=True) == directory and not directory.is_symlink()
    monkeypatch.setenv("ABLATION_RUNTIME_ROOT", str(runtime))
    monkeypatch.setenv("AMENDMENT08_ENABLE_TRUE_REHEARSAL", "1")
    monkeypatch.setattr(full, "RESULTS_ROOT", results)
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", runtime)
    monkeypatch.setattr(full, "A06_RUNTIME", a06_runtime)
    monkeypatch.setattr(full, "A07_TARGET_RUN", run_dir)
    monkeypatch.setattr(full, "A08_TARGET_RUN", run_dir)
    monkeypatch.setattr(full, "A08_RUNTIME", a08_runtime)
    monkeypatch.setattr(
        full, "A08_A06_BOUND_RUNTIME",
        a06_runtime / "corrigendum08_journal_adjudication",
    )
    monkeypatch.setattr(
        full, "A08_INSTALL_INTENT",
        a08_runtime / "authorization/amendment08_overlay_install_intent.json",
    )
    monkeypatch.setattr(
        full, "A08_SOURCE_FREEZE_INTENT",
        a08_runtime / "source_freeze/install_intent.json",
    )
    monkeypatch.setattr(
        full, "A08_SOURCE_FREEZE_RECEIPT",
        a08_runtime / "source_freeze/install_receipt.json",
    )
    monkeypatch.setattr(
        full, "AUTHORIZED_CHANGE_LEDGER",
        a06_runtime / "authorized_change_ledger.jsonl",
    )
    monkeypatch.setattr(
        full, "FINAL_PRERUN_EVIDENCE",
        a06_runtime / "pre_run/final_prerun_evidence.json",
    )
    monkeypatch.setattr(
        full, "BASELINE_TREE_TSV",
        a06_runtime / "pre_edit/baseline_tree_sha_size_rel.tsv",
    )
    package_root = runtime / "package_code_lineage" / full.A08_TARGET_RUN_ID
    monkeypatch.setattr(full, "A08_PACKAGE_LINEAGE_ROOT", package_root)
    monkeypatch.setattr(
        full, "A08_PACKAGE_A08_LINEAGE_ROOT", package_root / "amendment08",
    )
    return run_dir, runtime, a06_runtime, a08_runtime


def test_amendment08_rehearsal_context_rejects_misrouted_a06_before_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_root = tmp_path / "isolated"
    results = isolated_root / "results"
    runtime = isolated_root / "runtime"
    wrong_a06 = isolated_root / "wrong_a06_runtime"
    run_dir = results / full.A08_TARGET_RUN_ID
    for directory in (run_dir, runtime, wrong_a06):
        directory.mkdir(parents=True)
    sentinel = wrong_a06 / "sentinel.txt"
    sentinel.write_bytes(b"must-not-change\n")
    monkeypatch.setenv("AMENDMENT08_ENABLE_TRUE_REHEARSAL", "1")
    monkeypatch.setattr(full, "RESULTS_ROOT", results)
    monkeypatch.setattr(full, "EXPECTED_RUNTIME_ROOT", runtime)
    monkeypatch.setattr(full, "A06_RUNTIME", wrong_a06)
    monkeypatch.setattr(full, "A07_TARGET_RUN", run_dir)
    monkeypatch.setattr(full, "A08_TARGET_RUN", run_dir)
    with pytest.raises(
        full.Amendment06IntegrityError, match="exact isolated pytest Run/Runtime pair",
    ):
        with full.amendment08_isolated_rehearsal_context(
            run_dir=run_dir, runtime_root=runtime,
        ):
            pytest.fail("misrouted rehearsal context became active")
    assert sentinel.read_bytes() == b"must-not-change\n"
    assert not (wrong_a06 / "amendment06_full.lock").exists()


def test_amendment08_true_same_run_resume_rehearsal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.environ.get("AMENDMENT08_RUN_TRUE_REHEARSAL") != "1":
        pytest.skip("requires the dedicated Amendment 08 true CUDA rehearsal")
    handoff_text = os.environ.get("AMENDMENT08_RESUME_REHEARSAL_HANDOFF", "")
    if not handoff_text:
        pytest.fail("AMENDMENT08_RESUME_REHEARSAL_HANDOFF is required")
    handoff_path = Path(handoff_text).expanduser()
    assert handoff_path.is_absolute()
    assert handoff_path.name == "amendment08_resume_rehearsal_handoff.json"
    isolated_root = handoff_path.parent
    isolated_root.mkdir(parents=True, exist_ok=True)
    assert isolated_root.resolve(strict=True) == isolated_root
    assert not isolated_root.is_symlink()

    live_before = full.sha256_bytes(
        full.amendment08_production_live_journal_inventory_bytes()
    )
    assert live_before == full.A08_PRELIVE_JOURNAL_SENTINEL_SHA256
    run_dir = isolated_root / "results" / full.A08_TARGET_RUN_ID
    runtime = isolated_root / "runtime"
    a06_runtime = isolated_root / "a06_runtime"
    _amendment08_extract_true_rehearsal_fixture(
        run_dir=run_dir, runtime_root=runtime, a06_runtime=a06_runtime,
    )
    cache_clone = _amendment08_clone_true_rehearsal_cache(
        run_dir=run_dir, runtime_root=runtime,
    )
    assert cache_clone["distinct_inode_count"] == 11
    patched_run, patched_runtime, _, a08_runtime = (
        _amendment08_patch_true_rehearsal_roots(
            monkeypatch=monkeypatch, isolated_root=isolated_root,
        )
    )
    assert patched_run == run_dir and patched_runtime == runtime

    with full.amendment08_isolated_rehearsal_context(
        run_dir=run_dir, runtime_root=runtime,
    ):
        if handoff_path.exists():
            handoff = full.validate_amendment08_resume_rehearsal_handoff(
                handoff_path, run_dir=run_dir,
            )
            assert handoff["status"] == "PASS"
            assert full.sha256_bytes(
                full.amendment08_production_live_journal_inventory_bytes()
            ) == live_before
            return

        projection = full.install_amendment08_isolated_rehearsal_projection(
            run_dir=run_dir,
            preflight_run=full.ACCEPTED_PREFLIGHT,
            smoke_run=full.ACCEPTED_SMOKE,
        )
        assert projection["status"] == "PASS"
        assert projection["mode"] == full.A08_ISOLATED_REHEARSAL_PROJECTION_MODE
        assert projection["cache_clone_distinct_inodes"] is True
        probe_path = (
            a08_runtime / "pre_run/amendment08_cuda_producer_consumer_probe.json"
        )
        probe = full.run_amendment08_cuda_producer_consumer_probe(probe_path)
        assert probe["status"] == "PASS"
        overlay = full.install_amendment08_overlay(
            probe_path=probe_path, run_dir=run_dir,
        )
        assert overlay["status"] == "PASS"

        package_calls: list[Path] = []

        def defer_package(path: Path, *, command_line: str = "") -> dict[str, object]:
            assert full._safe_regular(handoff_path)
            published_handoff = full.strict_full_load_file(handoff_path)
            assert set(published_handoff) == set(full.A08_REHEARSAL_HANDOFF_KEYS)
            assert published_handoff["status"] == "PASS"
            package_calls.append(Path(path).resolve())
            return {
                "status": "PASS", "package_deferred_to_dedicated_node": True,
                "command_line": command_line,
            }

        monkeypatch.setattr(full, "package_dispatch", defer_package)
        resumed = full.resume_full(
            run_dir,
            preflight_run=full.ACCEPTED_PREFLIGHT,
            smoke_run=full.ACCEPTED_SMOKE,
            command_line="AMENDMENT08_TRUE_SAME_RUN_RESUME_REHEARSAL",
        )
        assert resumed == run_dir
        assert package_calls == [run_dir]
        final = full.validate_final_scientific_run(run_dir)
        assert final["status"] == "PASS" and final["rows"] == 236

        views = full.read_amendment08_journal_views(run_dir)
        raw_events = views["raw_events"]
        semantic_events = views["semantic_events"]
        assert raw_events[7]["event"] == "AMENDMENT08_CORRIGENDUM_INSTALLED"
        assert raw_events[8]["event"] == "CACHE_LOAD"
        assert raw_events[9]["event"] == "TRAINING_ATTEMPT_STARTED"
        fit_events = [
            event for event in semantic_events
            if event.get("event") == "SCIENTIFIC_FIT_CALLED"
        ]
        failed_events = [
            event for event in semantic_events
            if event.get("event") == "TRAINING_ATTEMPT_FAILED"
        ]
        completions = [
            event for event in semantic_events
            if event.get("event") == "TRAINING_COMPLETION_COMMITTED"
        ]
        freezes = [
            event for event in semantic_events if event.get("event") == "MODELS_FROZEN"
        ]
        audit_opens = [
            event for event in semantic_events
            if event.get("event") == "EXTERNAL_AUDIT_OPENED"
        ]
        assert len(fit_events) == 11
        assert len(failed_events) == 1
        assert failed_events[0]["attempt_id"] == full.A07_FAILED_ATTEMPT_ID
        assert len(completions) == 10
        assert [event["variant_id"] for event in completions] == list(
            full.TRAINED_VARIANTS
        )
        assert len(freezes) == len(audit_opens) == 1
        freeze_index = int(freezes[0]["event_index"])
        audit_open_index = int(audit_opens[0]["event_index"])
        assert freeze_index < audit_open_index
        assert all(
            int(event["event_index"]) < freeze_index for event in fit_events
        )
        assert sum(
            event.get("event") == "CACHE_GENERATION_INTENT"
            for event in raw_events
        ) == 1
        assert sum(
            event.get("event") == "CACHE_GENERATION" for event in raw_events
        ) == 1
        assert sum(event.get("event") == "CACHE_LOAD" for event in raw_events) == 1
        assert len(list(run_dir.glob("models/*/model.ubj"))) == 10
        assert len(list(run_dir.glob("models/*/training_completion_manifest.json"))) == 10
        frozen = full.strict_full_load_file(
            run_dir / "provenance/full_models_frozen_manifest.json"
        )
        assert frozen["scientific_fit_attempts"] == 11
        assert frozen["failed_unpublished_training_attempts"] == 1

        live_after = full.sha256_bytes(
            full.amendment08_production_live_journal_inventory_bytes()
        )
        assert live_after == live_before
        log = (
            "STATUS=PASS\n"
            "CACHE_CLONE_DISTINCT_INODES=PASS_11_OF_11\n"
            "SCIENTIFIC_FIT_ATTEMPTS=11\n"
            "FAILED_UNPUBLISHED_FIT_ATTEMPTS=1\n"
            "COMPLETED_MODELS=10\n"
            "FREEZE_BEFORE_AUDIT_OPEN=PASS\n"
            "NEW_RESAMPLING_REALIZATIONS=0\n"
            f"PRODUCTION_JOURNAL_SENTINEL_SHA256={live_after}\n"
        ).encode("utf-8")
        log_path = isolated_root / "logs/amendment08_true_resume_rehearsal.log"
        if log_path.exists():
            assert full._safe_regular(log_path, full.sha256_bytes(log))
        else:
            full.publish_bytes_no_clobber(log_path, log)
        print(
            "AMENDMENT08_TRUE_RESUME_REHEARSAL "
            "STATUS=PASS FITS=11 FAILED=1 MODELS=10 DISTINCT_CACHE_INODES=11"
        )
        handoff = full.publish_amendment08_resume_rehearsal_handoff(
            run_dir=run_dir, handoff_path=handoff_path,
            live_journal_before_sha256=live_before,
            live_journal_after_sha256=live_after,
        )
        assert handoff["status"] == "PASS"
        assert handoff["output_manifest_rows"] == 236
        assert handoff["cache_source_inventory_sha256"] == (
            cache_clone["source_inventory_sha256"]
        )
