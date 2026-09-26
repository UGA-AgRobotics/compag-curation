from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

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
