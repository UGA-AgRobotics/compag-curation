"""Evidence-derived protocol-amendment workflow gates.

This module is deliberately independent from the study runner.  Callers provide
the outcomes calculated from artifacts; this module validates their evidence
classifications and derives workflow decisions without embedding any observed
result from a particular Run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


class EvidenceGateError(ValueError):
    """An evidence registry or workflow fact is invalid."""


class EvidenceClassification(str, Enum):
    """The only classifications allowed by Protocol Amendment 01."""

    VERIFIED = "VERIFIED"
    INFERRED = "INFERRED"
    UNRESOLVED = "UNRESOLVED"
    BLOCKED = "BLOCKED"

    @classmethod
    def parse(cls, value: "EvidenceClassification | str") -> "EvidenceClassification":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(item.value for item in cls)
            raise EvidenceGateError(
                f"Invalid evidence classification {value!r}; allowed values: {allowed}"
            ) from exc


class GateTarget(str, Enum):
    DATA_SOURCE = "DATA_SOURCE"
    SPLIT_LOCK = "SPLIT_LOCK"
    CUDA_SMOKE = "CUDA_SMOKE"
    READY_FOR_FULL = "READY_FOR_FULL"


class WorkflowPhase(str, Enum):
    AMENDED_PREFLIGHT = "AMENDED_PREFLIGHT"
    CUDA_SMOKE = "CUDA_SMOKE"

    @classmethod
    def parse(cls, value: "WorkflowPhase | str") -> "WorkflowPhase":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(item.value for item in cls)
            raise EvidenceGateError(
                f"Invalid workflow phase {value!r}; allowed values: {allowed}"
            ) from exc


class ProtocolAmendment(str, Enum):
    """Supported authorization policies, with Amendment 01 as the default."""

    AMENDMENT_01 = "AMENDMENT_01"
    AMENDMENT_02 = "AMENDMENT_02"

    @classmethod
    def parse(cls, value: "ProtocolAmendment | str") -> "ProtocolAmendment":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(item.value for item in cls)
            raise EvidenceGateError(
                f"Invalid protocol amendment {value!r}; allowed values: {allowed}"
            ) from exc


@dataclass(frozen=True)
class EvidenceGate:
    """One calculated check and the evidence quality supporting its outcome."""

    key: str
    classification: EvidenceClassification
    passed: bool | None
    evidence_refs: tuple[str, ...]
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise EvidenceGateError("Evidence gate key must be nonempty")
        object.__setattr__(self, "classification", EvidenceClassification.parse(self.classification))
        if self.passed is not None and type(self.passed) is not bool:
            raise EvidenceGateError(f"Gate {self.key!r} passed must be bool or None")
        refs = tuple(str(value).strip() for value in self.evidence_refs)
        if any(not value for value in refs):
            raise EvidenceGateError(f"Gate {self.key!r} contains an empty evidence reference")
        object.__setattr__(self, "evidence_refs", refs)
        if self.classification is EvidenceClassification.VERIFIED:
            if self.passed is None:
                raise EvidenceGateError(f"Verified gate {self.key!r} requires a calculated outcome")
            if not refs:
                raise EvidenceGateError(f"Verified gate {self.key!r} requires evidence_refs")
        elif self.classification in {
            EvidenceClassification.UNRESOLVED,
            EvidenceClassification.BLOCKED,
        } and self.passed is True:
            raise EvidenceGateError(
                f"{self.classification.value} gate {self.key!r} cannot claim passed=True"
            )

    @property
    def accepted(self) -> bool:
        """Only a verified, successful calculation can satisfy a scientific gate."""

        return self.classification is EvidenceClassification.VERIFIED and self.passed is True

    @property
    def status_token(self) -> str:
        if self.accepted:
            return "PASS"
        if self.classification is EvidenceClassification.VERIFIED:
            return "FAIL"
        if self.classification is EvidenceClassification.INFERRED:
            return "INFERRED_NOT_ACCEPTED"
        return self.classification.value


@dataclass(frozen=True)
class GateSpec:
    key: str
    blocker_code: str
    targets: frozenset[GateTarget]


def _targets(*values: GateTarget) -> frozenset[GateTarget]:
    return frozenset(values)


_DATA = GateTarget.DATA_SOURCE
_SPLIT = GateTarget.SPLIT_LOCK
_SMOKE = GateTarget.CUDA_SMOKE
_READY = GateTarget.READY_FOR_FULL


DEFAULT_GATE_SPECS: tuple[GateSpec, ...] = (
    GateSpec("prior_run_immutability", "PRIOR_RUN_IMMUTABILITY_FAILURE", _targets(_SMOKE, _READY)),
    GateSpec("source_input_integrity", "SOURCE_INPUT_INTEGRITY_FAILURE", _targets(_DATA, _SPLIT, _SMOKE, _READY)),
    GateSpec("historical_snapshot_provenance", "HISTORICAL_SNAPSHOT_PROVENANCE_FAILURE", _targets(_DATA, _SPLIT, _SMOKE, _READY)),
    GateSpec("current_snapshot_analysis", "CURRENT_SNAPSHOT_ANALYSIS_INCOMPLETE", _targets(_DATA, _SPLIT, _SMOKE, _READY)),
    GateSpec("historical_split_reproduction", "HISTORICAL_SPLIT_REPRODUCTION_FAILURE", _targets(_DATA, _SPLIT, _SMOKE, _READY)),
    GateSpec("historical_prediction_replay", "HISTORICAL_PREDICTION_REPLAY_FAILURE", _targets(_DATA, _SPLIT, _SMOKE, _READY)),
    GateSpec("historical_metric_tree_parity", "HISTORICAL_METRIC_TREE_PARITY_FAILURE", _targets(_DATA, _SPLIT, _SMOKE, _READY)),
    GateSpec("r92_external_control", "R92_EXTERNAL_CONTROL_FAILURE", _targets(_SMOKE, _READY)),
    GateSpec("pca_prototype_compatibility", "PCA_PROTOTYPE_COMPATIBILITY_FAILURE", _targets(_DATA, _SPLIT, _SMOKE, _READY)),
    GateSpec("review_weight_parity", "REVIEW_WEIGHT_PARITY_FAILURE", _targets(_DATA, _SPLIT, _SMOKE, _READY)),
    GateSpec("augmentation_shuffle_parity", "AUGMENTATION_SHUFFLE_PARITY_FAILURE", _targets(_DATA, _SPLIT, _SMOKE, _READY)),
    GateSpec("feature_dependency_closure", "FEATURE_DEPENDENCY_CLOSURE_FAILURE", _targets(_SPLIT, _SMOKE, _READY)),
    GateSpec("feature_semantics", "FEATURE_SEMANTICS_FAILURE", _targets(_SPLIT, _SMOKE, _READY)),
    GateSpec("gate_reconstruction", "GATE_RECONSTRUCTION_FAILURE", _targets(_SPLIT, _SMOKE, _READY)),
    GateSpec("safe_smote_isolation", "SAFE_SMOTE_ISOLATION_FAILURE", _targets(_SMOKE, _READY)),
    GateSpec("split_manifest_integrity", "SPLIT_MANIFEST_INTEGRITY_FAILURE", _targets(_SPLIT, _SMOKE, _READY)),
    GateSpec("split_zero_leakage", "SPLIT_LEAKAGE_FAILURE", _targets(_SPLIT, _SMOKE, _READY)),
    GateSpec("no_pca_diagnostic", "NO_PCA_DIAGNOSTIC_INCOMPLETE", _targets(_SMOKE, _READY)),
    GateSpec("expanded_tests", "EXPANDED_TESTS_FAILURE", _targets(_SMOKE, _READY)),
    GateSpec("cuda_capability", "CUDA_CAPABILITY_FAILURE", _targets(_SMOKE, _READY)),
    GateSpec("cuda_smoke_all_runnable", "CUDA_SMOKE_INCOMPLETE", _targets(_READY)),
    GateSpec("model_save_reload", "MODEL_SAVE_RELOAD_FAILURE", _targets(_READY)),
    GateSpec(
        "output_manifest_zip_verification",
        "OUTPUT_MANIFEST_ZIP_FAILURE",
        _targets(_SMOKE, _READY),
    ),
    GateSpec(
        "no_full_scientific_training",
        "UNAUTHORIZED_FULL_EXECUTION",
        _targets(_SMOKE, _READY),
    ),
)


@dataclass(frozen=True)
class Blocker:
    gate_key: str
    code: str
    classification: EvidenceClassification
    status: str
    reason: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class GateRegistry:
    """A complete, validated collection of calculated evidence gates."""

    specs: tuple[GateSpec, ...]
    evidence: Mapping[str, EvidenceGate]

    def __post_init__(self) -> None:
        spec_keys = [spec.key for spec in self.specs]
        if len(spec_keys) != len(set(spec_keys)):
            raise EvidenceGateError("Gate specifications contain duplicate keys")
        evidence = dict(self.evidence)
        missing = sorted(set(spec_keys) - set(evidence))
        unknown = sorted(set(evidence) - set(spec_keys))
        if missing or unknown:
            raise EvidenceGateError(
                f"Evidence registry mismatch: missing={missing}, unknown={unknown}"
            )
        for key, gate in evidence.items():
            if gate.key != key:
                raise EvidenceGateError(
                    f"Evidence mapping key {key!r} differs from gate key {gate.key!r}"
                )
        object.__setattr__(self, "evidence", MappingProxyType(evidence))

    def accepted(self, key: str) -> bool:
        return self.evidence[key].accepted

    def specs_for(self, target: GateTarget) -> tuple[GateSpec, ...]:
        return tuple(spec for spec in self.specs if target in spec.targets)

    def all_accepted(self, target: GateTarget) -> bool:
        return all(self.accepted(spec.key) for spec in self.specs_for(target))

    def blockers_for(self, target: GateTarget) -> tuple[Blocker, ...]:
        blockers = []
        for spec in self.specs_for(target):
            gate = self.evidence[spec.key]
            if gate.accepted:
                continue
            reason = gate.detail.strip() or _default_blocker_reason(gate)
            blockers.append(Blocker(
                gate_key=gate.key,
                code=spec.blocker_code,
                classification=gate.classification,
                status=gate.status_token,
                reason=reason,
                evidence_refs=gate.evidence_refs,
            ))
        return tuple(blockers)

    @property
    def status_tokens(self) -> Mapping[str, str]:
        return MappingProxyType({
            key: gate.status_token for key, gate in self.evidence.items()
        })


def _default_blocker_reason(gate: EvidenceGate) -> str:
    if gate.classification is EvidenceClassification.VERIFIED:
        return "Verified calculation failed its acceptance rule"
    return f"Evidence classification {gate.classification.value} is not sufficient for acceptance"


def _coerce_refs(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    raise EvidenceGateError(f"evidence_refs must be a string or iterable, got {type(value).__name__}")


def _gate_from_input(key: str, value: EvidenceGate | Mapping[str, Any]) -> EvidenceGate:
    if isinstance(value, EvidenceGate):
        return value
    if not isinstance(value, Mapping):
        raise EvidenceGateError(
            f"Gate {key!r} input must be EvidenceGate or mapping, got {type(value).__name__}"
        )
    allowed = {"classification", "passed", "evidence_refs", "detail"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise EvidenceGateError(f"Gate {key!r} has unknown input fields: {unknown}")
    if "classification" not in value or "passed" not in value:
        raise EvidenceGateError(f"Gate {key!r} requires classification and passed")
    return EvidenceGate(
        key=key,
        classification=EvidenceClassification.parse(value["classification"]),
        passed=value["passed"],
        evidence_refs=_coerce_refs(value.get("evidence_refs")),
        detail=str(value.get("detail", "")),
    )


def build_gate_registry(
    inputs: Mapping[str, EvidenceGate | Mapping[str, Any]],
    *,
    specs: Sequence[GateSpec] = DEFAULT_GATE_SPECS,
) -> GateRegistry:
    """Validate calculated inputs and return a complete fail-closed registry.

    Integration payload for each key::

        {
            "classification": "VERIFIED",
            "passed": calculated_boolean,
            "evidence_refs": ["provenance/calculated_artifact.json"],
            "detail": "optional calculated failure detail",
        }

    Missing and unknown keys are rejected.  An inferred result never satisfies a
    gate even when its provisional ``passed`` value is true.
    """

    expected_keys = {spec.key for spec in specs}
    missing = sorted(expected_keys - set(inputs))
    unknown = sorted(set(inputs) - expected_keys)
    if missing or unknown:
        raise EvidenceGateError(
            f"Evidence input mismatch: missing={missing}, unknown={unknown}"
        )
    evidence = {
        spec.key: _gate_from_input(spec.key, inputs[spec.key])
        for spec in specs
    }
    return GateRegistry(tuple(specs), evidence)


@dataclass(frozen=True)
class WorkflowFacts:
    """Execution facts that cannot be inferred from scientific gate artifacts."""

    no_pca_variant_status: str
    no_pca_variant_runnable: bool
    phase: WorkflowPhase = WorkflowPhase.AMENDED_PREFLIGHT
    official_smoke_variants_completed: int = 0
    exploratory_smoke_variants_completed: int = 0
    exploratory_smoke_variants_expected: int = 1
    full_scientific_run_executed: bool = False
    full_variants_completed: int = 0
    full_variants_expected: int = 10
    protocol_amendment: ProtocolAmendment = ProtocolAmendment.AMENDMENT_01

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol_amendment",
            ProtocolAmendment.parse(self.protocol_amendment),
        )
        object.__setattr__(self, "phase", WorkflowPhase.parse(self.phase))
        if type(self.no_pca_variant_runnable) is not bool:
            raise EvidenceGateError("no_pca_variant_runnable must be bool")
        if type(self.full_scientific_run_executed) is not bool:
            raise EvidenceGateError("full_scientific_run_executed must be bool")
        if not self.no_pca_variant_status or "\n" in self.no_pca_variant_status:
            raise EvidenceGateError("no_pca_variant_status must be one nonempty status token")
        for name in (
            "official_smoke_variants_completed",
            "exploratory_smoke_variants_completed", "exploratory_smoke_variants_expected",
            "full_variants_completed", "full_variants_expected",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise EvidenceGateError(f"{name} must be a nonnegative integer")
        for completed, expected in (
            (self.official_smoke_variants_completed, self.official_smoke_variants_expected),
            (self.exploratory_smoke_variants_completed, self.exploratory_smoke_variants_expected),
            (self.full_variants_completed, self.full_variants_expected),
        ):
            if completed > expected:
                raise EvidenceGateError("Completed variant count cannot exceed expected count")

    @property
    def official_smoke_variants_expected(self) -> int:
        """Derive the official Smoke denominator from true no-PCA availability."""

        return 10 if self.no_pca_variant_runnable else 9

    @property
    def no_pca_ready(self) -> bool:
        """Require a coherent true no-PCA outcome for the reviewed protocol.

        Amendment 01 explicitly permits Smoke of the other nine official variants
        when the machine diagnostic proves that true no-PCA is unavailable.  In
        that case the diagnostic gate, not a fabricated model, satisfies this fact.
        """

        if self.no_pca_variant_runnable:
            return (
                self.no_pca_variant_status
                == "VERIFIED_EXACT_RAW_EMBEDDINGS_AVAILABLE"
            )
        return (
            self.no_pca_variant_status
            == "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE"
        )


@dataclass(frozen=True)
class WorkflowState:
    run_kind: str
    run_state: str
    amended_preflight_status: str
    data_source_decision: str
    split_locked: bool
    split_lock_status: str
    smoke_eligible: bool
    cuda_smoke_status: str
    ready_for_full_awaiting_external_review: bool
    blocker_codes: tuple[str, ...]
    blockers: tuple[Blocker, ...]
    gate_statuses: Mapping[str, str]
    facts: WorkflowFacts


def _runtime_blocker(code: str, reason: str) -> Blocker:
    return Blocker(
        gate_key="runtime_facts",
        code=code,
        classification=EvidenceClassification.VERIFIED,
        status="FAIL",
        reason=reason,
        evidence_refs=(),
    )


def _deduplicate_blockers(blockers: Iterable[Blocker]) -> tuple[Blocker, ...]:
    output: list[Blocker] = []
    seen: set[str] = set()
    for blocker in blockers:
        if blocker.code not in seen:
            output.append(blocker)
            seen.add(blocker.code)
    return tuple(output)


def _nonblocking_limitations(facts: WorkflowFacts) -> frozenset[str]:
    """Return disclosed evidence gates excluded from transition authorization.

    Amendment 02 accepts unresolved PCA/prototype lineage specifically as a
    limitation of the retrospective fixed-feature-table design. The evidence
    gate keeps its original classification and status token; no other failed or
    unresolved gate is relaxed by this policy.
    """

    if facts.protocol_amendment is ProtocolAmendment.AMENDMENT_02:
        return frozenset({"pca_prototype_compatibility"})
    return frozenset()


def _all_accepted_for(
    registry: GateRegistry,
    target: GateTarget,
    nonblocking_limitations: frozenset[str],
) -> bool:
    return all(
        registry.accepted(spec.key) or spec.key in nonblocking_limitations
        for spec in registry.specs_for(target)
    )


def _blockers_for(
    registry: GateRegistry,
    target: GateTarget,
    nonblocking_limitations: frozenset[str],
) -> tuple[Blocker, ...]:
    return tuple(
        blocker for blocker in registry.blockers_for(target)
        if blocker.gate_key not in nonblocking_limitations
    )


def derive_workflow_state(
    registry: GateRegistry,
    facts: WorkflowFacts,
) -> WorkflowState:
    """Derive all workflow decisions from evidence and supplied execution facts."""

    nonblocking_limitations = _nonblocking_limitations(facts)
    data_source_verified = _all_accepted_for(
        registry, GateTarget.DATA_SOURCE, nonblocking_limitations,
    )
    split_locked = data_source_verified and _all_accepted_for(
        registry, GateTarget.SPLIT_LOCK, nonblocking_limitations,
    )
    smoke_blockers = _blockers_for(
        registry, GateTarget.CUDA_SMOKE, nonblocking_limitations,
    )
    if facts.phase is WorkflowPhase.CUDA_SMOKE:
        smoke_blockers = tuple(
            blocker for blocker in smoke_blockers
            if blocker.gate_key != "output_manifest_zip_verification"
        )
    no_full_execution = (
        not facts.full_scientific_run_executed and facts.full_variants_completed == 0
    )
    smoke_eligible = (
        not smoke_blockers
        and split_locked
        and facts.no_pca_ready
        and no_full_execution
    )

    official_complete = (
        facts.official_smoke_variants_completed == facts.official_smoke_variants_expected
    )
    exploratory_complete = (
        facts.exploratory_smoke_variants_completed == facts.exploratory_smoke_variants_expected
    )
    smoke_evidence_passed = registry.accepted("cuda_smoke_all_runnable")
    smoke_complete = smoke_eligible and smoke_evidence_passed and official_complete and exploratory_complete

    ready = (
        _all_accepted_for(
            registry, GateTarget.READY_FOR_FULL, nonblocking_limitations,
        )
        and split_locked
        and smoke_complete
        and no_full_execution
        and facts.no_pca_ready
    )

    if facts.phase is WorkflowPhase.AMENDED_PREFLIGHT:
        run_kind = "AMENDED_PREFLIGHT"
        run_state = "PREFLIGHT_COMPLETE" if smoke_eligible else "BLOCKED_PREFLIGHT_COMPLETE"
    else:
        run_kind = "CUDA_SMOKE_NON_SCIENTIFIC"
        if not smoke_eligible:
            run_state = "SMOKE_NOT_AUTHORIZED"
        else:
            run_state = "SMOKE_COMPLETE" if smoke_complete else "SMOKE_INCOMPLETE"

    if smoke_complete:
        cuda_smoke_status = "PASS"
    elif facts.phase is WorkflowPhase.CUDA_SMOKE and smoke_eligible:
        cuda_smoke_status = "INCOMPLETE"
    elif smoke_eligible:
        cuda_smoke_status = "NOT_RUN_AWAITING_EXECUTION"
    else:
        cuda_smoke_status = "NOT_RUN_GLOBAL_GATES_FAILED"

    if data_source_verified:
        data_source_decision = (
            "ACCEPTED_HISTORICAL_FIXED_FEATURE_TABLE_WITH_LIMITATION"
            if facts.protocol_amendment is ProtocolAmendment.AMENDMENT_02
            else "HISTORICAL_R92_SNAPSHOT_VERIFIED_PRIMARY_PAIRED_ABLATION"
        )
    else:
        data_source_decision = "UNRESOLVED_REQUIRES_EXTERNAL_REVIEW"
    if split_locked:
        split_lock_status = "LOCKED"
    elif not data_source_verified:
        split_lock_status = "NOT_LOCKED_DATA_SOURCE_DECISION_UNRESOLVED"
    else:
        split_lock_status = "NOT_LOCKED_SPLIT_GATES_FAILED"

    blockers: list[Blocker] = list(smoke_blockers)
    if facts.phase is WorkflowPhase.CUDA_SMOKE and not ready:
        blockers.extend(_blockers_for(
            registry, GateTarget.READY_FOR_FULL, nonblocking_limitations,
        ))
    if smoke_evidence_passed and not official_complete:
        blockers.append(_runtime_blocker(
            "OFFICIAL_SMOKE_VARIANT_COUNT_MISMATCH",
            "CUDA Smoke evidence passed but official completion count is incomplete",
        ))
    if smoke_evidence_passed and not exploratory_complete:
        blockers.append(_runtime_blocker(
            "EXPLORATORY_SMOKE_VARIANT_COUNT_MISMATCH",
            "CUDA Smoke evidence passed but exploratory completion count is incomplete",
        ))
    if not no_full_execution:
        blockers.append(_runtime_blocker(
            "UNAUTHORIZED_FULL_EXECUTION",
            "Runtime facts report Full execution or completed Full variants",
        ))
    if not facts.no_pca_ready:
        blockers.append(_runtime_blocker(
            "NO_PCA_VARIANT_NOT_VERIFIED",
            "True no-PCA runtime status is inconsistent with its reviewed mapping availability",
        ))
    blockers_tuple = _deduplicate_blockers(blockers)

    return WorkflowState(
        run_kind=run_kind,
        run_state=run_state,
        amended_preflight_status=(
            "PASS_READY_FOR_CUDA_SMOKE" if smoke_eligible else "BLOCKED_BEFORE_CUDA_SMOKE"
        ),
        data_source_decision=data_source_decision,
        split_locked=split_locked,
        split_lock_status=split_lock_status,
        smoke_eligible=smoke_eligible,
        cuda_smoke_status=cuda_smoke_status,
        ready_for_full_awaiting_external_review=ready,
        blocker_codes=tuple(blocker.code for blocker in blockers_tuple),
        blockers=blockers_tuple,
        gate_statuses=registry.status_tokens,
        facts=facts,
    )


def _composite_status(state: WorkflowState, keys: Sequence[str]) -> str:
    tokens = [state.gate_statuses[key] for key in keys]
    if all(token == "PASS" for token in tokens):
        return "PASS"
    for token in ("FAIL", "BLOCKED", "UNRESOLVED", "INFERRED_NOT_ACCEPTED"):
        if token in tokens:
            return token
    raise EvidenceGateError(f"Unrecognized composite gate statuses: {tokens}")


def render_status_lines(
    state: WorkflowState,
    *,
    extra_lines: Mapping[str, Any] | None = None,
) -> str:
    """Render machine-readable status lines exclusively from derived state.

    ``extra_lines`` is intended for artifact identifiers and hashes calculated by
    the integrating builder.  It may not override any derived status field.
    """

    facts = state.facts
    derived: list[tuple[str, Any]] = [
        ("RUN_KIND", state.run_kind),
        ("RUN_STATE", state.run_state),
        ("AMENDED_PREFLIGHT_STATUS", state.amended_preflight_status),
        ("DATA_SOURCE_DECISION", state.data_source_decision),
        ("SPLIT_LOCK_STATUS", state.split_lock_status),
        ("R92_CONTROL_STATUS", state.gate_statuses["r92_external_control"]),
        ("FEATURE_SEMANTICS_STATUS", state.gate_statuses["feature_semantics"]),
        ("GATE_RECONSTRUCTION_STATUS", state.gate_statuses["gate_reconstruction"]),
        ("REVIEW_WEIGHT_PARITY_STATUS", state.gate_statuses["review_weight_parity"]),
        ("NO_PCA_STATUS", facts.no_pca_variant_status),
        ("NO_PCA_RUNNABLE", "YES" if facts.no_pca_variant_runnable else "NO"),
        ("CUDA_CAPABILITY_STATUS", state.gate_statuses["cuda_capability"]),
        ("CUDA_SMOKE_ELIGIBLE", "YES" if state.smoke_eligible else "NO"),
        ("CUDA_SMOKE_STATUS", state.cuda_smoke_status),
        ("OFFICIAL_SMOKE_VARIANTS_COMPLETED", (
            f"{facts.official_smoke_variants_completed}/{facts.official_smoke_variants_expected}"
        )),
        ("EXPLORATORY_SMOKE_VARIANTS_COMPLETED", (
            f"{facts.exploratory_smoke_variants_completed}/{facts.exploratory_smoke_variants_expected}"
        )),
        ("FULL_SCIENTIFIC_RUN_EXECUTED", (
            "YES" if facts.full_scientific_run_executed else "NO"
        )),
        ("FULL_VARIANTS_COMPLETED", (
            f"{facts.full_variants_completed}/{facts.full_variants_expected}"
        )),
        ("READY_FOR_FULL_AWAITING_EXTERNAL_REVIEW", (
            "YES" if state.ready_for_full_awaiting_external_review else "NO"
        )),
        ("SOURCE_INPUT_INTEGRITY", state.gate_statuses["source_input_integrity"]),
        ("PRIOR_RUN_IMMUTABILITY", state.gate_statuses["prior_run_immutability"]),
        ("OUTPUT_MANIFEST_AND_ZIP_VERIFICATION", (
            state.gate_statuses["output_manifest_zip_verification"]
        )),
    ]
    if facts.protocol_amendment is ProtocolAmendment.AMENDMENT_02:
        derived.insert(1, ("PROTOCOL_AMENDMENT", facts.protocol_amendment.value))
    derived_keys = {key for key, _ in derived} | {"BLOCKERS", "NEXT_ACTION"}
    extras = list((extra_lines or {}).items())
    collisions = sorted(derived_keys & {key for key, _ in extras})
    if collisions:
        raise EvidenceGateError(f"extra_lines cannot override derived fields: {collisions}")
    lines = [f"{key}={value}" for key, value in derived + extras]
    lines.append("BLOCKERS=" + (";".join(state.blocker_codes) if state.blocker_codes else "NONE"))
    lines.append("NEXT_ACTION=STOP_AND_AWAIT_EXTERNAL_REVIEW")
    return "\n".join(lines) + "\n"


__all__ = [
    "Blocker",
    "DEFAULT_GATE_SPECS",
    "EvidenceClassification",
    "EvidenceGate",
    "EvidenceGateError",
    "GateRegistry",
    "GateSpec",
    "GateTarget",
    "ProtocolAmendment",
    "WorkflowFacts",
    "WorkflowPhase",
    "WorkflowState",
    "build_gate_registry",
    "derive_workflow_state",
    "render_status_lines",
]
