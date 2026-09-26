from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


LAYOUT_ROOT = Path(__file__).resolve().parents[1]
if (LAYOUT_ROOT / "code").is_dir():
    STUDY_ROOT = LAYOUT_ROOT
    CODE_ROOT = STUDY_ROOT / "code"
elif (LAYOUT_ROOT / "code_snapshot").is_dir():
    STUDY_ROOT = LAYOUT_ROOT.parent
    CODE_ROOT = LAYOUT_ROOT / "code_snapshot"
else:
    raise RuntimeError(f"Cannot locate reviewer code beside {__file__}")
os.environ.setdefault("ABLATION_STUDY_ROOT", str(STUDY_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import evidence_gates as gates  # noqa: E402


def _all_verified_inputs() -> dict[str, dict[str, object]]:
    return {
        spec.key: {
            "classification": "VERIFIED",
            "passed": True,
            "evidence_refs": [f"provenance/{spec.key}.json"],
            "detail": "calculated acceptance rule passed",
        }
        for spec in gates.DEFAULT_GATE_SPECS
    }


def _complete_smoke_facts(**changes: object) -> gates.WorkflowFacts:
    values: dict[str, object] = {
        "no_pca_variant_status": "VERIFIED_EXACT_RAW_EMBEDDINGS_AVAILABLE",
        "no_pca_variant_runnable": True,
        "phase": gates.WorkflowPhase.CUDA_SMOKE,
        "official_smoke_variants_completed": 10,
        "exploratory_smoke_variants_completed": 1,
        "exploratory_smoke_variants_expected": 1,
        "full_scientific_run_executed": False,
        "full_variants_completed": 0,
        "full_variants_expected": 10,
    }
    values.update(changes)
    return gates.WorkflowFacts(**values)


def _preflight_facts(**changes: object) -> gates.WorkflowFacts:
    values: dict[str, object] = {
        "no_pca_variant_status": "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        "no_pca_variant_runnable": False,
    }
    values.update(changes)
    return gates.WorkflowFacts(**values)


def test_classification_enum_is_exact_and_invalid_value_is_rejected():
    assert [item.value for item in gates.EvidenceClassification] == [
        "VERIFIED", "INFERRED", "UNRESOLVED", "BLOCKED",
    ]
    with pytest.raises(gates.EvidenceGateError, match="Invalid evidence classification"):
        gates.EvidenceClassification.parse("PASS")


def test_protocol_amendment_defaults_to_amendment_01_and_rejects_unknown_value():
    facts = _preflight_facts()
    assert facts.protocol_amendment is gates.ProtocolAmendment.AMENDMENT_01

    with pytest.raises(gates.EvidenceGateError, match="Invalid protocol amendment"):
        _preflight_facts(protocol_amendment="AMENDMENT_03")


def test_verified_gate_requires_calculated_outcome_and_evidence_reference():
    with pytest.raises(gates.EvidenceGateError, match="requires evidence_refs"):
        gates.EvidenceGate(
            key="check", classification=gates.EvidenceClassification.VERIFIED,
            passed=True, evidence_refs=(),
        )
    with pytest.raises(gates.EvidenceGateError, match="requires a calculated outcome"):
        gates.EvidenceGate(
            key="check", classification=gates.EvidenceClassification.VERIFIED,
            passed=None, evidence_refs=("artifact.json",),
        )


def test_registry_rejects_missing_and_unknown_inputs():
    inputs = _all_verified_inputs()
    inputs.pop("cuda_capability")
    with pytest.raises(gates.EvidenceGateError, match="missing=.*cuda_capability"):
        gates.build_gate_registry(inputs)

    inputs = _all_verified_inputs()
    inputs["invented_gate"] = {
        "classification": "VERIFIED", "passed": True,
        "evidence_refs": ["invented.json"],
    }
    with pytest.raises(gates.EvidenceGateError, match="unknown=.*invented_gate"):
        gates.build_gate_registry(inputs)


def test_all_pass_derives_locked_split_complete_smoke_and_ready_state():
    registry = gates.build_gate_registry(_all_verified_inputs())
    state = gates.derive_workflow_state(registry, _complete_smoke_facts())

    assert state.data_source_decision == "HISTORICAL_R92_SNAPSHOT_VERIFIED_PRIMARY_PAIRED_ABLATION"
    assert state.split_locked
    assert state.smoke_eligible
    assert state.run_kind == "CUDA_SMOKE_NON_SCIENTIFIC"
    assert state.run_state == "SMOKE_COMPLETE"
    assert state.cuda_smoke_status == "PASS"
    assert state.facts.official_smoke_variants_expected == 10
    assert state.ready_for_full_awaiting_external_review
    assert state.blocker_codes == ()

    rendered = gates.render_status_lines(state)
    assert "DATA_SOURCE_DECISION=HISTORICAL_R92_SNAPSHOT_VERIFIED_PRIMARY_PAIRED_ABLATION" in rendered
    assert "CUDA_SMOKE_STATUS=PASS" in rendered
    assert "NO_PCA_STATUS=VERIFIED_EXACT_RAW_EMBEDDINGS_AVAILABLE" in rendered
    assert "NO_PCA_RUNNABLE=YES" in rendered
    assert "OFFICIAL_SMOKE_VARIANTS_COMPLETED=10/10" in rendered
    assert "READY_FOR_FULL_AWAITING_EXTERNAL_REVIEW=YES" in rendered
    assert "BLOCKERS=NONE" in rendered


def test_verified_historical_replay_failure_changes_decision_and_rendered_status():
    inputs = _all_verified_inputs()
    inputs["historical_prediction_replay"] = {
        "classification": "VERIFIED",
        "passed": False,
        "evidence_refs": ["provenance/historical_prediction_replay.csv.gz"],
        "detail": "calculated maximum probability difference exceeded tolerance",
    }
    registry = gates.build_gate_registry(inputs)
    state = gates.derive_workflow_state(registry, _preflight_facts())
    rendered = gates.render_status_lines(state)

    assert state.data_source_decision == "UNRESOLVED_REQUIRES_EXTERNAL_REVIEW"
    assert not state.split_locked
    assert not state.smoke_eligible
    assert state.run_state == "BLOCKED_PREFLIGHT_COMPLETE"
    assert "HISTORICAL_PREDICTION_REPLAY_FAILURE" in state.blocker_codes
    assert "R92_CONTROL_STATUS=PASS" in rendered
    assert "CUDA_SMOKE_STATUS=NOT_RUN_GLOBAL_GATES_FAILED" in rendered
    assert "READY_FOR_FULL_AWAITING_EXTERNAL_REVIEW=NO" in rendered


def test_unresolved_semantics_does_not_rewrite_data_decision_but_blocks_split_and_smoke():
    inputs = _all_verified_inputs()
    inputs["feature_semantics"] = {
        "classification": "UNRESOLVED",
        "passed": None,
        "evidence_refs": ["provenance/feature_semantics.json"],
        "detail": "transform lineage is unresolved",
    }
    state = gates.derive_workflow_state(
        gates.build_gate_registry(inputs), _preflight_facts(),
    )

    assert state.data_source_decision == "HISTORICAL_R92_SNAPSHOT_VERIFIED_PRIMARY_PAIRED_ABLATION"
    assert not state.split_locked
    assert state.split_lock_status == "NOT_LOCKED_SPLIT_GATES_FAILED"
    assert not state.smoke_eligible
    assert state.gate_statuses["feature_semantics"] == "UNRESOLVED"
    assert "FEATURE_SEMANTICS_FAILURE" in state.blocker_codes


def test_amendment_02_accepts_only_unresolved_pca_lineage_as_nonblocking_limitation():
    inputs = _all_verified_inputs()
    inputs["pca_prototype_compatibility"] = {
        "classification": "UNRESOLVED",
        "passed": None,
        "evidence_refs": ["provenance/pca_artifact_inventory.tsv"],
        "detail": "one common PCA/prototype basis is not established",
    }
    inputs["cuda_smoke_all_runnable"] = {
        "classification": "BLOCKED", "passed": None,
        "evidence_refs": ["metrics/SMOKE_REPORT.json"],
        "detail": "Smoke has not run",
    }
    inputs["model_save_reload"] = {
        "classification": "BLOCKED", "passed": None,
        "evidence_refs": ["metrics/SMOKE_REPORT.json"],
        "detail": "Smoke has not run",
    }

    state = gates.derive_workflow_state(
        gates.build_gate_registry(inputs),
        _preflight_facts(protocol_amendment="AMENDMENT_02"),
    )

    assert state.data_source_decision == (
        "ACCEPTED_HISTORICAL_FIXED_FEATURE_TABLE_WITH_LIMITATION"
    )
    assert state.split_locked
    assert state.smoke_eligible
    assert state.run_state == "PREFLIGHT_COMPLETE"
    assert state.gate_statuses["pca_prototype_compatibility"] == "UNRESOLVED"
    assert "PCA_PROTOTYPE_COMPATIBILITY_FAILURE" not in state.blocker_codes
    rendered = gates.render_status_lines(state)
    assert "PROTOCOL_AMENDMENT=AMENDMENT_02" in rendered
    assert (
        "DATA_SOURCE_DECISION="
        "ACCEPTED_HISTORICAL_FIXED_FEATURE_TABLE_WITH_LIMITATION"
    ) in rendered


@pytest.mark.parametrize(
    ("key", "classification", "passed", "blocker"),
    [
        (
            "historical_prediction_replay", "VERIFIED", False,
            "HISTORICAL_PREDICTION_REPLAY_FAILURE",
        ),
        (
            "feature_semantics", "UNRESOLVED", None,
            "FEATURE_SEMANTICS_FAILURE",
        ),
    ],
)
def test_amendment_02_does_not_waive_other_failed_or_unresolved_gates(
    key, classification, passed, blocker,
):
    inputs = _all_verified_inputs()
    inputs["pca_prototype_compatibility"] = {
        "classification": "UNRESOLVED", "passed": None,
        "evidence_refs": ["provenance/pca_artifact_inventory.tsv"],
    }
    inputs[key] = {
        "classification": classification, "passed": passed,
        "evidence_refs": [f"provenance/{key}.json"],
    }

    state = gates.derive_workflow_state(
        gates.build_gate_registry(inputs),
        _preflight_facts(protocol_amendment=gates.ProtocolAmendment.AMENDMENT_02),
    )

    assert not state.split_locked
    assert not state.smoke_eligible
    assert blocker in state.blocker_codes
    assert "PCA_PROTOTYPE_COMPATIBILITY_FAILURE" not in state.blocker_codes


def test_amendment_01_still_blocks_unresolved_pca_and_keeps_legacy_status_shape():
    inputs = _all_verified_inputs()
    inputs["pca_prototype_compatibility"] = {
        "classification": "UNRESOLVED", "passed": None,
        "evidence_refs": ["provenance/pca_artifact_inventory.tsv"],
    }

    state = gates.derive_workflow_state(
        gates.build_gate_registry(inputs), _preflight_facts(),
    )
    rendered = gates.render_status_lines(state)

    assert state.data_source_decision == "UNRESOLVED_REQUIRES_EXTERNAL_REVIEW"
    assert not state.split_locked
    assert not state.smoke_eligible
    assert "PCA_PROTOTYPE_COMPATIBILITY_FAILURE" in state.blocker_codes
    assert "PROTOCOL_AMENDMENT=" not in rendered


def test_inferred_true_and_blocked_cuda_are_not_accepted():
    inputs = _all_verified_inputs()
    inputs["expanded_tests"] = {
        "classification": "INFERRED", "passed": True,
        "evidence_refs": ["logs/partial-tests.log"],
    }
    inputs["cuda_capability"] = {
        "classification": "BLOCKED", "passed": None,
        "evidence_refs": ["provenance/cuda_probe.json"],
    }
    inputs["output_manifest_zip_verification"] = {
        "classification": "VERIFIED", "passed": False,
        "evidence_refs": ["provenance/package_verification.json"],
    }
    state = gates.derive_workflow_state(
        gates.build_gate_registry(inputs), _preflight_facts(),
    )

    assert not state.smoke_eligible
    assert state.gate_statuses["expanded_tests"] == "INFERRED_NOT_ACCEPTED"
    assert state.gate_statuses["cuda_capability"] == "BLOCKED"
    assert "EXPANDED_TESTS_FAILURE" in state.blocker_codes
    assert "CUDA_CAPABILITY_FAILURE" in state.blocker_codes
    assert "OUTPUT_MANIFEST_ZIP_FAILURE" in state.blocker_codes


def test_preflight_pass_is_not_ready_until_smoke_and_package_evidence_pass():
    inputs = _all_verified_inputs()
    for key in (
        "cuda_smoke_all_runnable", "model_save_reload",
        "output_manifest_zip_verification",
    ):
        inputs[key] = {
            "classification": "BLOCKED", "passed": None,
            "evidence_refs": [f"provenance/{key}.json"],
            "detail": "not executed yet",
        }
    registry = gates.build_gate_registry(inputs)
    state = gates.derive_workflow_state(registry, _preflight_facts())

    assert not state.smoke_eligible
    assert state.run_state == "BLOCKED_PREFLIGHT_COMPLETE"
    assert state.cuda_smoke_status == "NOT_RUN_GLOBAL_GATES_FAILED"
    assert not state.ready_for_full_awaiting_external_review
    assert state.blocker_codes == ("OUTPUT_MANIFEST_ZIP_FAILURE",)


def test_external_r92_control_is_independent_of_historical_prediction_replay():
    inputs = _all_verified_inputs()
    inputs["r92_external_control"] = {
        "classification": "VERIFIED", "passed": False,
        "evidence_refs": ["metrics/r92_control.json"],
    }
    state = gates.derive_workflow_state(
        gates.build_gate_registry(inputs), _preflight_facts(),
    )
    rendered = gates.render_status_lines(state)
    assert "R92_CONTROL_STATUS=FAIL" in rendered
    assert "R92_EXTERNAL_CONTROL_FAILURE" in state.blocker_codes


def test_blocked_true_no_pca_uses_nine_variant_smoke_and_completed_diagnostic():
    state = gates.derive_workflow_state(
        gates.build_gate_registry(_all_verified_inputs()),
        _complete_smoke_facts(
            no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
            no_pca_variant_runnable=False,
            official_smoke_variants_completed=9,
        ),
    )
    assert state.smoke_eligible
    assert state.cuda_smoke_status == "PASS"
    assert state.ready_for_full_awaiting_external_review
    assert "NO_PCA_VARIANT_NOT_VERIFIED" not in state.blocker_codes
    rendered = gates.render_status_lines(state)
    assert "NO_PCA_STATUS=NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE" in rendered
    assert "NO_PCA_RUNNABLE=NO" in rendered
    assert "OFFICIAL_SMOKE_VARIANTS_COMPLETED=9/9" in rendered
    assert "READY_FOR_FULL_AWAITING_EXTERNAL_REVIEW=YES" in rendered


def test_no_pca_facts_are_explicit_and_unknown_status_cannot_be_ready():
    with pytest.raises(TypeError, match="no_pca_variant"):
        gates.WorkflowFacts()
    with pytest.raises(gates.EvidenceGateError, match="no_pca_variant_runnable must be bool"):
        gates.WorkflowFacts(
            no_pca_variant_status="VERIFIED_EXACT_RAW_EMBEDDINGS_AVAILABLE",
            no_pca_variant_runnable=None,  # type: ignore[arg-type]
        )

    state = gates.derive_workflow_state(
        gates.build_gate_registry(_all_verified_inputs()),
        _complete_smoke_facts(no_pca_variant_status="UNKNOWN"),
    )
    assert state.facts.official_smoke_variants_expected == 10
    assert state.cuda_smoke_status == "NOT_RUN_GLOBAL_GATES_FAILED"
    assert not state.smoke_eligible
    assert not state.ready_for_full_awaiting_external_review
    assert "NO_PCA_VARIANT_NOT_VERIFIED" in state.blocker_codes
    assert "NO_PCA_STATUS=UNKNOWN" in gates.render_status_lines(state)


def test_incoherent_no_pca_facts_block_preflight_smoke_authorization():
    state = gates.derive_workflow_state(
        gates.build_gate_registry(_all_verified_inputs()),
        _preflight_facts(
            no_pca_variant_status="NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
            no_pca_variant_runnable=True,
        ),
    )
    assert state.smoke_eligible is False
    assert state.cuda_smoke_status == "NOT_RUN_GLOBAL_GATES_FAILED"
    assert "NO_PCA_VARIANT_NOT_VERIFIED" in state.blocker_codes


def test_runtime_counts_and_unauthorized_full_execution_prevent_ready_state():
    registry = gates.build_gate_registry(_all_verified_inputs())
    facts = _complete_smoke_facts(
        official_smoke_variants_completed=9,
        full_scientific_run_executed=True,
        full_variants_completed=1,
    )
    state = gates.derive_workflow_state(registry, facts)

    assert state.run_state == "SMOKE_NOT_AUTHORIZED"
    assert state.cuda_smoke_status == "NOT_RUN_GLOBAL_GATES_FAILED"
    assert not state.ready_for_full_awaiting_external_review
    assert "OFFICIAL_SMOKE_VARIANT_COUNT_MISMATCH" in state.blocker_codes
    assert "UNAUTHORIZED_FULL_EXECUTION" in state.blocker_codes
    rendered = gates.render_status_lines(state)
    assert "OFFICIAL_SMOKE_VARIANTS_COMPLETED=9/10" in rendered
    assert "FULL_SCIENTIFIC_RUN_EXECUTED=YES" in rendered
    assert "FULL_VARIANTS_COMPLETED=1/10" in rendered


def test_unauthorized_full_fact_blocks_preflight_smoke_authorization():
    state = gates.derive_workflow_state(
        gates.build_gate_registry(_all_verified_inputs()),
        _preflight_facts(
            full_scientific_run_executed=True,
            full_variants_completed=1,
        ),
    )

    assert state.run_state == "BLOCKED_PREFLIGHT_COMPLETE"
    assert state.smoke_eligible is False
    assert state.cuda_smoke_status == "NOT_RUN_GLOBAL_GATES_FAILED"
    assert "UNAUTHORIZED_FULL_EXECUTION" in state.blocker_codes
    rendered = gates.render_status_lines(state)
    assert "CUDA_SMOKE_ELIGIBLE=NO" in rendered
    assert "AMENDED_PREFLIGHT_STATUS=BLOCKED_BEFORE_CUDA_SMOKE" in rendered


def test_renderer_supports_artifact_lines_but_cannot_override_derived_status():
    state = gates.derive_workflow_state(
        gates.build_gate_registry(_all_verified_inputs()),
        _complete_smoke_facts(),
    )
    rendered = gates.render_status_lines(state, extra_lines={
        "REVIEW_BUNDLE_PATH": "/study/results/review.zip",
        "REVIEW_BUNDLE_SHA256": "abc123",
    })
    assert "REVIEW_BUNDLE_PATH=/study/results/review.zip" in rendered
    assert "REVIEW_BUNDLE_SHA256=abc123" in rendered
    with pytest.raises(gates.EvidenceGateError, match="cannot override"):
        gates.render_status_lines(state, extra_lines={"CUDA_SMOKE_STATUS": "FAKE"})
