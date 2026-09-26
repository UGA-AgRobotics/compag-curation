from __future__ import annotations

import csv
import inspect
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "code"))

import amendment03_recovery as recovery  # noqa: E402
import amendment03_compat as compat  # noqa: E402
import amendment_core as core  # noqa: E402
import evidence_gates as gates  # noqa: E402


def _manifest_bytes(rows, *, output=False):
    columns = ["relative_path", "size_bytes", "sha256"]
    if output:
        columns += ["artifact_role", "include_in_review_bundle", "include_in_models_bundle"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        value = dict(row)
        if output:
            value.update({
                "artifact_role": value["relative_path"].split("/", 1)[0],
                "include_in_review_bundle": True,
                "include_in_models_bundle": False,
            })
        writer.writerow(value)
    return stream.getvalue().encode()


def _corrected_gate_fixture(tmp_path, *, corrupt_probe_manifest=False):
    root = tmp_path / "study"
    results = root / "results"
    run_id = recovery.ACCEPTED_PREFLIGHT_RUN_ID
    run = results / run_id
    run.mkdir(parents=True)

    probe = {
        "classification": "VERIFIED",
        "status": "PASS",
        "xgboost_version": "2.1.1",
        "build_info": {"USE_CUDA": True},
        "requested_device": "cuda",
        "tree_method": "hist",
        "training_status": "PASS",
        "save_reload_predict_status": "PASS",
        "cpu_fallback_detected": False,
        "fallback_warnings": [],
    }
    files = {
        "RUN_STATUS.txt": (
            "RUN_STATE=PREFLIGHT_COMPLETE\n"
            "PROTOCOL_AMENDMENT=AMENDMENT_02\n"
            "SMOKE_ELIGIBLE=YES\n"
            "CUDA_SMOKE_ELIGIBLE=YES\n"
            "FULL_AUTHORIZED=NO\n"
            "BLOCKERS=NONE\n"
            f"LOCKED_SPLIT_SHA256={recovery.LOCKED_SPLIT_SHA256}\n"
            "BASE_SEED=42\n"
        ).encode(),
        "config/run_identity.lock.json": (json.dumps({
            "run_id": run_id,
            "state": "PREFLIGHT_COMPLETE",
            "protocol": "AMENDMENT_02",
            "candidate_split_hash": recovery.LOCKED_SPLIT_SHA256,
            "base_seed": 42,
            "full_authorized": False,
            "split_locked": True,
            "code_tree": {"code/a.py": "a" * 64},
        }) + "\n").encode(),
        "config/locked_split_identity.json": (json.dumps({
            "status": "PASS",
            "base_seed": 42,
            "split_assignment_sha256": recovery.LOCKED_SPLIT_SHA256,
        }) + "\n").encode(),
        recovery.SEALED_PROBE_RELATIVE: (json.dumps(probe) + "\n").encode(),
    }
    for relative, data in files.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    output_rows = [{
        "relative_path": relative,
        "size_bytes": len(data),
        "sha256": (
            "0" * 64
            if corrupt_probe_manifest and relative == recovery.SEALED_PROBE_RELATIVE
            else recovery.sha256_bytes(data)
        ),
    } for relative, data in sorted(files.items())]
    output = _manifest_bytes(output_rows, output=True)
    (run / "OUTPUT_MANIFEST_FINAL.tsv").write_bytes(output)
    bundle_rows = [
        {key: row[key] for key in ("relative_path", "size_bytes", "sha256")}
        for row in output_rows
    ] + [{
        "relative_path": "OUTPUT_MANIFEST_FINAL.tsv",
        "size_bytes": len(output),
        "sha256": recovery.sha256_bytes(output),
    }]
    bundle = _manifest_bytes(bundle_rows)
    (run / "BUNDLE_MANIFEST.tsv").write_bytes(bundle)

    archive_path = results / f"{run_id}_review_bundle.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in run.rglob("*") if item.is_file()):
            archive.write(path, f"{run_id}/{path.relative_to(run).as_posix()}")
    archive_sha = recovery.sha256_file(archive_path)
    adjacent = results / f"{run_id}_package_verification.json"
    adjacent.write_text(json.dumps({
        "bundle_sha256": archive_sha,
        "crc_status": "PASS",
        "independent_reopen_member_verification": "PASS",
        "verification_failures": [],
        "package_completeness": {
            "preflight_semantic_completeness": {
                "status": "PASS",
                "checks": {"isolated_cuda_probe_active_pass": True},
            },
        },
    }))

    failure_hashes = {}
    for index, relative in enumerate(recovery.FAILURE_EVIDENCE_RELATIVE_HASHES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"failure-{index}\n")
        failure_hashes[relative] = recovery.sha256_file(path)
    orphan = root / recovery.ORPHAN_MODEL_RELATIVE
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"orphan")
    runtime = root / recovery.RUNTIME_PROBE_RELATIVE
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_bytes(b"runtime diagnostic")
    return {
        "root": root,
        "run": run,
        "zip": archive_path,
        "zip_sha": archive_sha,
        "adjacent": adjacent,
        "failure_hashes": failure_hashes,
        "orphan_sha": recovery.sha256_file(orphan),
        "runtime_sha": recovery.sha256_file(runtime),
        "probe_bytes": files[recovery.SEALED_PROBE_RELATIVE],
    }


def test_corrected_gate_and_sealed_probe_use_manifested_zip_bytes(tmp_path):
    fixture = _corrected_gate_fixture(tmp_path)
    evidence = recovery.validate_corrected_immutable_gate(
        study_root=fixture["root"],
        expected_zip_sha256=fixture["zip_sha"],
        expected_failure_hashes=fixture["failure_hashes"],
        expected_orphan_sha256=fixture["orphan_sha"],
        expected_runtime_probe_sha256=fixture["runtime_sha"],
    )
    assert evidence["status"] == "PASS"
    assert evidence["sealed_probe"]["member_sha256"] == recovery.sha256_bytes(
        fixture["probe_bytes"]
    )
    assert evidence["runtime_probe"]["used_as_source"] is False


def test_unmanifested_sealed_probe_is_rejected(tmp_path):
    fixture = _corrected_gate_fixture(tmp_path, corrupt_probe_manifest=True)
    with pytest.raises(recovery.RecoveryIntegrityError, match="canonical Manifest"):
        recovery.validate_sealed_probe_contract(
            fixture["zip"],
            fixture["adjacent"],
            expected_zip_sha256=fixture["zip_sha"],
        )


def test_dual_probe_copy_is_exact_and_refuses_root_overwrite(tmp_path):
    fixture = _corrected_gate_fixture(tmp_path)
    smoke = fixture["root"] / "results/replacement_cuda_smoke"
    smoke.mkdir()
    binding = recovery.copy_sealed_probe_to_run(
        run_dir=smoke,
        preflight_zip=fixture["zip"],
        package_verification_path=fixture["adjacent"],
        expected_zip_sha256=fixture["zip_sha"],
    )
    for relative in recovery.SEALED_PROBE_DESTINATIONS:
        assert (smoke / relative).read_bytes() == fixture["probe_bytes"]
    assert binding["both_destinations_byte_identical"] is True
    with pytest.raises(recovery.RecoveryIntegrityError, match="overwrite"):
        recovery.copy_sealed_probe_to_run(
            run_dir=smoke,
            preflight_zip=fixture["zip"],
            package_verification_path=fixture["adjacent"],
            expected_zip_sha256=fixture["zip_sha"],
        )


def test_replacement_guard_consumes_exactly_one_slot(tmp_path):
    results = tmp_path / "results"
    (results / recovery.FAILED_SMOKE_RUN_ID).mkdir(parents=True)
    before = recovery.assert_replacement_guard(results_root=results)
    assert before["replacement_authorization"] == "0_OF_1_BEFORE_RUN_CREATION"
    replacement = results / "run_new_amendment03_cuda_smoke"
    after = recovery.create_authorized_replacement_run_directory(
        replacement, results_root=results,
    )
    assert after["replacement_authorization"] == "1_OF_1_CONSUMED"
    with pytest.raises(recovery.RecoveryIntegrityError):
        recovery.create_authorized_replacement_run_directory(
            results / "run_second_amendment03_cuda_smoke", results_root=results,
        )


def test_executed_snapshot_requires_hash_specific_exact_allowlist(tmp_path):
    study = tmp_path / "study"
    code = study / "code/a.py"
    test = study / "tests/test_a.py"
    code.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    code.write_text("before = 1\n")
    test.write_text("def test_a(): pass\n")
    reviewed = {
        "code/a.py": recovery.sha256_file(code),
        "tests/test_a.py": recovery.sha256_file(test),
    }
    code.write_text("after = 2\n")
    allowed = {"code/a.py": recovery.sha256_file(code)}
    run = study / "results/replacement_cuda_smoke"
    run.mkdir(parents=True)
    built = recovery.build_executed_code_provenance(
        run_dir=run,
        reviewed_code_tree=reviewed,
        allowed_after_hashes=allowed,
        study_root=study,
    )
    assert built["changed_paths"] == ["code/a.py"]
    assert recovery.validate_executed_code_provenance(
        run_dir=run,
        reviewed_code_tree=reviewed,
        allowed_after_hashes=allowed,
        study_root=study,
    )["status"] == "PASS"
    test.write_text("def test_a(): assert False\n")
    with pytest.raises(recovery.RecoveryIntegrityError, match="provenance mismatch"):
        recovery.validate_executed_code_provenance(
            run_dir=run,
            reviewed_code_tree=reviewed,
            allowed_after_hashes=allowed,
            study_root=study,
        )


def test_recovery_lineage_revalidates_owned_artifacts(tmp_path):
    study = tmp_path / "study"
    code = study / "code/a.py"
    test = study / "tests/test_a.py"
    code.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    code.write_text("value = 1\n")
    test.write_text("def test_a(): pass\n")
    reviewed = {
        "code/a.py": recovery.sha256_file(code),
        "tests/test_a.py": recovery.sha256_file(test),
    }
    run = study / "results/run_replacement_amendment03_cuda_smoke"
    run.mkdir(parents=True)
    executed = recovery.build_executed_code_provenance(
        run_dir=run,
        reviewed_code_tree=reviewed,
        allowed_after_hashes={},
        study_root=study,
    )
    probe_bytes = b'{"status":"PASS"}\n'
    probe_sha = recovery.sha256_bytes(probe_bytes)
    destinations = {}
    for relative in recovery.SEALED_PROBE_DESTINATIONS:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(probe_bytes)
        destinations[relative] = {
            "size_bytes": len(probe_bytes),
            "sha256": probe_sha,
            "byte_equal_to_sealed_member": True,
        }
    binding = {
        "status": "PASS",
        "source_zip_sha256": recovery.ACCEPTED_PREFLIGHT_ZIP_SHA256,
        "source_member": recovery.SEALED_PROBE_RELATIVE,
        "member_size_bytes": len(probe_bytes),
        "member_sha256": probe_sha,
        "manifest_match": "PASS",
        "manifest_rows": {
            "OUTPUT_MANIFEST_FINAL.tsv": {
                "relative_path": recovery.SEALED_PROBE_RELATIVE,
                "size_bytes": len(probe_bytes),
                "sha256": probe_sha,
            },
            "BUNDLE_MANIFEST.tsv": {
                "relative_path": recovery.SEALED_PROBE_RELATIVE,
                "size_bytes": len(probe_bytes),
                "sha256": probe_sha,
            },
        },
        "runtime_probe_used_as_source": False,
        "root_nested_byte_equal": True,
        "destinations": destinations,
    }
    (run / "provenance/sealed_probe_binding.json").write_text(json.dumps(binding))
    corrected = {
        relative: {"path": relative, "sha256": digest}
        for relative, digest in recovery.FAILURE_EVIDENCE_RELATIVE_HASHES.items()
    }
    immutable = {
        "status": "PASS",
        "accepted_preflight_run_id": recovery.ACCEPTED_PREFLIGHT_RUN_ID,
        "accepted_preflight_run": "accepted",
        "accepted_preflight_zip": "accepted.zip",
        "accepted_preflight_zip_sha256": recovery.ACCEPTED_PREFLIGHT_ZIP_SHA256,
        "sealed_probe": {"member_sha256": probe_sha},
        "corrected_failure_evidence": corrected,
        "orphan_model": {
            "sha256": recovery.ORPHAN_MODEL_SHA256,
            "diagnostic_only": True,
        },
        "runtime_probe": {
            "sha256": recovery.RUNTIME_PROBE_SHA256,
            "used_as_source": False,
        },
    }
    recovery.build_recovery_lineage(
        run_dir=run,
        new_run_id=run.name,
        immutable_gate=immutable,
        executed_code=executed,
    )
    result = recovery.validate_recovery_artifacts(
        run_dir=run,
        reviewed_code_tree=reviewed,
        allowed_after_hashes={},
        study_root=study,
    )
    assert result["status"] == "PASS"
    nested = run / recovery.SEALED_PROBE_DESTINATIONS[1]
    nested.write_bytes(b"drift")
    with pytest.raises(recovery.RecoveryIntegrityError, match="copy"):
        recovery.validate_recovery_artifacts(
            run_dir=run,
            reviewed_code_tree=reviewed,
            allowed_after_hashes={},
            study_root=study,
        )


def test_amendment03_models_zip_requires_complete_nine_plus_one():
    official = [
        "full_new_reference", "manual_only", "deep_only", "no_color",
        "no_shape", "no_texture", "no_embed_sim",
        "no_review_aware_training_weights", "no_safe_smote",
    ]
    variants = [*official, core.EXPLORATORY_VARIANT]
    assert not core.smoke_model_package_eligible(
        completed=official,
        variants=variants,
        runnable_official=official,
        state="SMOKE_INCOMPLETE",
        amendment03_replacement=True,
    )
    assert not core.smoke_model_package_eligible(
        completed=variants[:-1],
        variants=variants,
        runnable_official=official,
        state="SMOKE_COMPLETE",
        amendment03_replacement=True,
    )
    assert core.smoke_model_package_eligible(
        completed=variants,
        variants=variants,
        runnable_official=official,
        state="SMOKE_COMPLETE",
        amendment03_replacement=True,
    )


def test_amendment03_structured_failure_is_explicitly_incomplete(
    tmp_path, monkeypatch,
):
    study = tmp_path / "study"
    run = study / "results/run_test_amendment03_cuda_smoke"
    (run / "logs").mkdir(parents=True)
    (run / "config").mkdir()
    monkeypatch.setattr(core, "STUDY_ROOT", study)
    error = RuntimeError("injected")
    core._write_amendment03_structured_failure(
        run_dir=run,
        phase="CUDA_SMOKE_VARIANTS",
        command_line="run_study.py --smoke --amendment-03-corrigendum-one",
        error=error,
        active_variant="manual_only",
        active_variant_index=2,
    )
    log = (run / "logs/smoke_failure.txt").read_text()
    status = (run / "RUN_STATUS.txt").read_text()
    identity = json.loads((run / "config/run_identity.lock.json").read_text())
    assert "PHASE=CUDA_SMOKE_VARIANTS" in log
    assert "ACTIVE_VARIANT=manual_only" in log
    assert "EXCEPTION_TYPE=RuntimeError" in log
    assert "TRACEBACK_BEGIN" in log and "TRACEBACK_END" in log
    assert "RUN_STATE=SMOKE_INCOMPLETE" in status
    assert "SCIENTIFIC_RESULTS_CLAIMED=NO" in status
    assert identity["recovery_authorization"] == core._AMENDMENT03_AUTHORIZATION
    assert "non-scientific" in core._NON_SCIENTIFIC_SMOKE_MARKER.lower()


def test_amendment03_late_review_failure_removes_owned_models(
    tmp_path, monkeypatch,
):
    study = tmp_path / "study"
    run = study / "results/run_late_amendment03_cuda_smoke"
    for relative in ("config", "provenance", "metrics", "logs"):
        (run / relative).mkdir(parents=True, exist_ok=True)
    (run / "config/run_identity.lock.json").write_text(json.dumps({
        "run_id": run.name,
        "protocol": "AMENDMENT_02",
        "run_kind": "CUDA_SMOKE_NON_SCIENTIFIC",
        "state": "SMOKE_COMPLETE",
    }))
    (run / "provenance/smoke_model_bundle_verification.json").write_text("{}\n")
    monkeypatch.setattr(core, "STUDY_ROOT", study)
    ownership = {}
    models = study / "results" / f"{run.name}_NON_SCIENTIFIC_smoke_models.zip"
    model_adjacent = study / "results" / (
        f"{run.name}_NON_SCIENTIFIC_smoke_models_package_verification.json"
    )
    core.publish_bytes_no_clobber(models, b"models", ownership=ownership)
    core.publish_bytes_no_clobber(model_adjacent, b"{}\n", ownership=ownership)
    model = {
        "path": str(models),
        "portable_relative_to_run": f"../{models.name}",
        "sha256": core.sha256_file(models),
        "run_id": run.name,
    }
    gate_inputs = {
        spec.key: {
            "classification": "VERIFIED",
            "passed": True,
            "evidence_refs": ["provenance/package_staging_verification.json"],
        }
        for spec in gates.DEFAULT_GATE_SPECS
    }
    facts = gates.WorkflowFacts(
        phase="CUDA_SMOKE",
        official_smoke_variants_completed=9,
        exploratory_smoke_variants_completed=1,
        no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        no_pca_variant_runnable=False,
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
        model_bundle=model,
        gate_inputs=gate_inputs,
        facts=facts,
        smoke_metrics={"smoke_models_bundle": model},
        error=core.IntegrityError("late seal failure"),
        ownership=ownership,
        amendment03_replacement=True,
        amendment03_recovery_validation={
            "sealed_probe_sha256": "4" * 64,
            "recovery_lineage_sha256": "5" * 64,
        },
    )
    assert not models.exists()
    assert not model_adjacent.exists()
    assert not (run / "provenance/smoke_model_bundle_verification.json").exists()
    metrics = json.loads((run / "metrics/SMOKE_REPORT.json").read_text())
    assert metrics["state"] == "SMOKE_INCOMPLETE"
    assert metrics["smoke_models_bundle"]["path"] == (
        "NOT_CREATED_NO_SMOKE_MODELS"
    )
    identity = json.loads((run / "config/run_identity.lock.json").read_text())
    assert identity["state"] == "SMOKE_INCOMPLETE"


def test_amendment03_invocation_requires_single_flag_and_pinned_preflight(
    monkeypatch,
):
    expected_python = core._AMENDMENT03_CUDA_PYTHON.resolve()
    if Path(sys.executable).resolve() != expected_python:
        pytest.skip("requires the locked Amendment 03 CUDA interpreter")
    preflight = STUDY / "results" / recovery.ACCEPTED_PREFLIGHT_RUN_ID
    command = (
        "code/run_study.py --smoke --amendment-03-corrigendum-one "
        f"--preflight-run {preflight}"
    )
    monkeypatch.setenv("ABLATION_PYTHON", str(expected_python))
    core._assert_amendment03_invocation(
        preflight_run=preflight,
        command_line=command,
        identity={"run_id": recovery.ACCEPTED_PREFLIGHT_RUN_ID},
        immutable_gate={
            "accepted_preflight_run_id": recovery.ACCEPTED_PREFLIGHT_RUN_ID,
        },
    )
    with pytest.raises(core.IntegrityError, match="exact invocation"):
        core._assert_amendment03_invocation(
            preflight_run=preflight,
            command_line=command + " --amendment-03-corrigendum-one",
            identity={"run_id": recovery.ACCEPTED_PREFLIGHT_RUN_ID},
            immutable_gate={
                "accepted_preflight_run_id": recovery.ACCEPTED_PREFLIGHT_RUN_ID,
            },
        )


def test_core_amendment03_snapshot_copies_dual_probe_but_not_historical_zip(
    tmp_path, monkeypatch,
):
    study = tmp_path / "study"
    preflight = study / "results/accepted_preflight"
    preflight.mkdir(parents=True)
    bundle = Path(str(preflight) + "_review_bundle.zip")
    probe = b'{"classification":"VERIFIED","status":"PASS"}\n'
    mapping = {
        "config/run_identity.lock.json": (
            "provenance/preflight_snapshot/run_identity.lock.json"
        ),
        "provenance/evidence_gate_registry.json": (
            "provenance/preflight_snapshot/evidence_gate_registry.json"
        ),
        "OUTPUT_MANIFEST_FINAL.tsv": (
            "provenance/preflight_snapshot/OUTPUT_MANIFEST_FINAL.tsv"
        ),
        "BUNDLE_MANIFEST.tsv": (
            "provenance/preflight_snapshot/BUNDLE_MANIFEST.tsv"
        ),
        compat.SEALED_PROBE_RELATIVE_PATH: compat.NESTED_PROBE_DESTINATION,
        "__PREFLIGHT_REVIEW_BUNDLE__": (
            "provenance/preflight_snapshot/preflight_review_bundle.zip"
        ),
    }
    source_bytes = {
        "config/run_identity.lock.json": b"identity\n",
        "provenance/evidence_gate_registry.json": b"registry\n",
        "OUTPUT_MANIFEST_FINAL.tsv": b"output\n",
        "BUNDLE_MANIFEST.tsv": b"bundle\n",
        compat.SEALED_PROBE_RELATIVE_PATH: probe,
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        for relative, data in source_bytes.items():
            archive.writestr(f"{preflight.name}/{relative}", data)
    bundle_sha = recovery.sha256_file(bundle)
    contract = compat.SealedProbeContract(
        data=probe,
        evidence={
            "classification": "VERIFIED",
            "status": "PASS",
            "source_zip_sha256": bundle_sha,
            "source_member": compat.SEALED_PROBE_RELATIVE_PATH,
            "member_size_bytes": len(probe),
            "member_sha256": recovery.sha256_bytes(probe),
            "manifest_match": "PASS",
            "manifest_rows": {},
            "runtime_probe_used_as_source": False,
        },
    )
    monkeypatch.setattr(core, "STUDY_ROOT", study)
    monkeypatch.setattr(core, "smoke_copied_evidence_mapping", lambda protocol: mapping)
    monkeypatch.setattr(core, "verify_zip", lambda *args, **kwargs: {"status": "PASS"})
    smoke = study / "results/run_test_amendment03_cuda_smoke"
    smoke.mkdir()
    reference = core.snapshot_smoke_review_evidence(
        preflight_run=preflight,
        run_dir=smoke,
        validated_identity={
            "run_id": preflight.name,
            "state": "PREFLIGHT_COMPLETE",
            "protocol": "AMENDMENT_02",
            "code_tree": {},
            "reference_validation": {
                "review_bundle_live_byte_equality_status": "PASS",
                "review_bundle_sha256": bundle_sha,
            },
        },
        command_line="run_study.py --smoke --amendment-03-corrigendum-one",
        split_hash="1" * 64,
        amendment03_contract=contract,
        amendment03_allowed_after_hashes={},
        amendment03_prerun_hashes={},
    )
    root_probe = smoke / compat.ROOT_PROBE_DESTINATION
    nested_probe = smoke / compat.NESTED_PROBE_DESTINATION
    forbidden_zip = (
        smoke / "provenance/preflight_snapshot/preflight_review_bundle.zip"
    )
    assert root_probe.read_bytes() == nested_probe.read_bytes() == probe
    assert not forbidden_zip.exists()
    assert reference["packaged_preflight_review_bundle_path"] == (
        "NOT_BUNDLED_AMENDMENT03_RECOVERY_LINEAGE_ONLY"
    )
    assert str(forbidden_zip.relative_to(smoke)) not in (
        reference["copied_evidence_sha256"]
    )


def test_public_amendment03_runner_cleans_models_on_unhandled_late_failure(
    tmp_path, monkeypatch,
):
    study = tmp_path / "study"
    results = study / "results"
    results.mkdir(parents=True)
    run = results / "run_20260813_120000_deadbeef_amendment03_cuda_smoke"
    models = results / f"{run.name}_NON_SCIENTIFIC_smoke_models.zip"
    adjacent = results / (
        f"{run.name}_NON_SCIENTIFIC_smoke_models_package_verification.json"
    )
    monkeypatch.setattr(core, "STUDY_ROOT", study)

    def fail_after_models(**kwargs):
        for relative in ("config", "provenance", "logs"):
            (run / relative).mkdir(parents=True, exist_ok=True)
        models.write_bytes(b"owned models")
        record = {
            "run_id": run.name,
            "path": str(models),
            "sha256": core.sha256_file(models),
        }
        adjacent.write_text(json.dumps(record))
        (run / "provenance/smoke_model_bundle_verification.json").write_text(
            json.dumps(record)
        )
        raise OSError("injected report write failure")

    monkeypatch.setattr(core, "_run_cuda_smoke_impl", fail_after_models)
    with pytest.raises(OSError, match="injected report write failure"):
        core.run_cuda_smoke(
            preflight_run=results / "accepted",
            command_line="run_study.py --smoke --amendment-03-corrigendum-one",
            amendment03_replacement=True,
        )
    assert not models.exists()
    assert not adjacent.exists()
    assert not (run / "provenance/smoke_model_bundle_verification.json").exists()
    assert json.loads((run / "config/run_identity.lock.json").read_text())[
        "state"
    ] == "SMOKE_INCOMPLETE"
    failure = (run / "logs/smoke_failure.txt").read_text()
    assert "PHASE=UNHANDLED_POST_AUTHORIZATION_FAILURE" in failure
    assert "TRACEBACK_BEGIN" in failure


def test_preexisting_complete_amendment03_survives_pre_id_guard_failure(
    tmp_path, monkeypatch,
):
    study = tmp_path / "study"
    results = study / "results"
    existing = results / "run_20260813_120000_deadbeef_amendment03_cuda_smoke"
    (existing / "config").mkdir(parents=True)
    (existing / "config/run_identity.lock.json").write_text(json.dumps({
        "run_id": existing.name,
        "state": "SMOKE_COMPLETE",
    }))
    models = results / f"{existing.name}_NON_SCIENTIFIC_smoke_models.zip"
    models.write_bytes(b"preexisting complete models")
    adjacent = results / (
        f"{existing.name}_NON_SCIENTIFIC_smoke_models_package_verification.json"
    )
    adjacent.write_text(json.dumps({
        "run_id": existing.name,
        "path": str(models),
        "sha256": core.sha256_file(models),
    }))
    before = {
        "identity": (existing / "config/run_identity.lock.json").read_bytes(),
        "models": models.read_bytes(),
        "adjacent": adjacent.read_bytes(),
    }
    monkeypatch.setattr(core, "STUDY_ROOT", study)

    def pre_id_failure(**kwargs):
        raise core.IntegrityError("replacement authorization already consumed")

    monkeypatch.setattr(core, "_run_cuda_smoke_impl", pre_id_failure)
    with pytest.raises(core.IntegrityError, match="already consumed"):
        core.run_cuda_smoke(
            preflight_run=results / "accepted",
            command_line="run_study.py --smoke --amendment-03-corrigendum-one",
            amendment03_replacement=True,
        )
    assert (existing / "config/run_identity.lock.json").read_bytes() == before[
        "identity"
    ]
    assert models.read_bytes() == before["models"]
    assert adjacent.read_bytes() == before["adjacent"]
    assert not (existing / "logs/smoke_failure.txt").exists()


def test_production_runner_seals_dual_probe_before_resampling_and_fit():
    snapshot_source = inspect.getsource(core.snapshot_smoke_review_evidence)
    runner_source = inspect.getsource(core._run_cuda_smoke_impl)
    assert "copy_sealed_probe_to_smoke" in snapshot_source
    assert "amendment03_contract=amendment03_contract" in runner_source
    snapshot_index = runner_source.index("snapshot_smoke_review_evidence(")
    recovery_index = runner_source.index("build_executed_code_provenance(")
    resampling_index = runner_source.index("build_frozen_resampling_population(")
    fit_index = runner_source.index("classifier.fit(")
    assert snapshot_index < recovery_index < resampling_index < fit_index


def test_amendment03_package_contract_rejects_missing_root_probe(
    tmp_path, monkeypatch,
):
    run = tmp_path / "run_contract_amendment03_cuda_smoke"
    identity = {
        "run_id": run.name,
        "protocol": "AMENDMENT_02",
        "run_kind": "CUDA_SMOKE_NON_SCIENTIFIC",
        "state": "SMOKE_INCOMPLETE",
        "recovery_authorization": core._AMENDMENT03_AUTHORIZATION,
    }
    required = set(core._AMENDMENT03_SMOKE_REQUIRED_FILES) | {
        "logs/smoke_failure.txt",
    }
    root_probe = recovery.SEALED_PROBE_DESTINATIONS[0]
    assert root_probe in required
    for relative in sorted(required - {root_probe}):
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence\n")
    (run / "config/run_identity.lock.json").write_text(json.dumps(identity))
    (run / "RUN_STATUS.txt").write_text(
        "RUN_KIND=CUDA_SMOKE_NON_SCIENTIFIC\nRUN_STATE=SMOKE_INCOMPLETE\n"
    )
    monkeypatch.setattr(
        core,
        "validate_smoke_semantic_completeness",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    with pytest.raises(core.IntegrityError, match="cuda_active_fit_probe.json"):
        core.validate_package_completeness(run, identity)


def test_amendment03_command_log_requires_exact_envelope_and_command(
    monkeypatch,
):
    command = [str(core._AMENDMENT03_CUDA_PYTHON), "-m", "pytest", "-q", "tests"]
    data = (
        f"COMMAND={' '.join(command)}\n"
        f"CWD={STUDY}\n"
        "STDOUT_BEGIN\npassed\nSTDOUT_END\n"
        "STDERR_BEGIN\nSTDERR_END\n"
        "EXIT_CODE=0\n"
    ).encode()
    monkeypatch.setattr(core, "STUDY_ROOT", STUDY)
    result = core._validate_amendment03_command_log(
        data=data,
        relative="logs/full_tests.log",
        expected_command=command,
    )
    assert result["exit_code"] == 0
    assert result["sha256"] == recovery.sha256_bytes(data)
    with pytest.raises(core.IntegrityError, match="command log contract"):
        core._validate_amendment03_command_log(
            data=data.replace(b"EXIT_CODE=0", b"EXIT_CODE=1"),
            relative="logs/full_tests.log",
            expected_command=command,
        )
    with pytest.raises(core.IntegrityError, match="command log contract"):
        core._validate_amendment03_command_log(
            data=data.replace(b"pytest -q", b"pytest -q -x"),
            relative="logs/full_tests.log",
            expected_command=command,
        )
    with pytest.raises(core.IntegrityError, match="Duplicate COMMAND"):
        core._validate_amendment03_command_log(
            data=data + b"COMMAND=duplicate\n",
            relative="logs/full_tests.log",
            expected_command=command,
        )


def test_live_binding_records_sklearn_classifier_observation_and_exact_gpu():
    binding = json.loads((
        STUDY
        / ".runtime/amendment03_corrigendum/amendment03_live_cuda_binding.json"
    ).read_text())
    context = binding["context"]
    gpu = context["gpu_identity"]
    assert context["sklearn_is_classifier_xgbclassifier"] is False
    assert binding["checks"]["sklearn_classifier_tag_observed"] is True
    assert binding["checks"]["nvidia_smi_pass"] is True
    assert gpu["exit_code"] == 0
    assert gpu["stderr"] == ""
    assert "NVIDIA GeForce RTX 4090" in gpu["stdout"]
    assert "595.95" in gpu["stdout"]


def test_amendment03_reference_chain_accepts_exact_six_file_drift():
    immutable = recovery.validate_corrected_immutable_gate(study_root=STUDY)
    allowed = core.amendment03_code_drift_allowlist(
        immutable["reviewed_code_tree"]
    )
    assert set(allowed) == set(core._AMENDMENT03_ALLOWED_CODE_DRIFT_PATHS)

    identity = core.validate_amended_run_reference(
        Path(immutable["accepted_preflight_run"]),
        expected_kind="AMENDED_PREFLIGHT",
        allowed_states={"PREFLIGHT_COMPLETE"},
        amendment03_allowed_after_hashes=allowed,
    )

    semantic = identity["reference_validation"][
        "preflight_semantic_completeness"
    ]
    assert semantic["status"] == "PASS"
    assert semantic["checks"][
        "code_tree_snapshots_and_protocol_docs_bound"
    ] is True


def test_amendment03_reference_chain_rejects_arbitrary_allowed_hash():
    immutable = recovery.validate_corrected_immutable_gate(study_root=STUDY)
    allowed = core.amendment03_code_drift_allowlist(
        immutable["reviewed_code_tree"]
    )
    arbitrary = dict(allowed)
    arbitrary["code/amendment_core.py"] = "0" * 64

    with pytest.raises(
        core.IntegrityError,
        match="code_tree_snapshots_and_protocol_docs_bound",
    ):
        core.validate_amended_run_reference(
            Path(immutable["accepted_preflight_run"]),
            expected_kind="AMENDED_PREFLIGHT",
            allowed_states={"PREFLIGHT_COMPLETE"},
            amendment03_allowed_after_hashes=arbitrary,
        )
