"""Fail-closed recovery provenance helpers for Amendment 03 Corrigendum One.

This module deliberately contains no training entry point.  It validates the
immutable recovery references, consumes the single replacement-Run slot, and
builds provenance that the existing Smoke workflow can package and revalidate.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


STUDY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = STUDY_ROOT / "results"

ACCEPTED_PREFLIGHT_RUN_ID = (
    "run_20260813_050405_9970c566_amendment02_preflight"
)
ACCEPTED_PREFLIGHT_ZIP_SHA256 = (
    "19d6d9ec381bab99ff41a081fd42016c653d4a4ce947f35694e0e32e57ae24cb"
)
FAILED_SMOKE_RUN_ID = "run_20260813_050604_437486_bc48422a_cuda_smoke"
LOCKED_SPLIT_SHA256 = (
    "f2cd1570d2c76af49ed52593d95657a3cbbe14d8f2f077edd05447e608698c64"
)
SEALED_PROBE_RELATIVE = "provenance/cuda_active_fit_probe.json"
SEALED_PROBE_DESTINATIONS = (
    "provenance/cuda_active_fit_probe.json",
    "provenance/preflight_snapshot/cuda_active_fit_probe.json",
)
AMENDMENT03_RECOVERY_REQUIRED_FILES = frozenset({
    "docs/PROTOCOL_AMENDMENT_03.md",
    "provenance/xgb211_reload_diagnostic.json",
    "provenance/cuda_active_fit_probe.json",
    "provenance/preflight_snapshot/cuda_active_fit_probe.json",
    "provenance/amendment03_live_cuda_binding.json",
    "provenance/sealed_probe_binding.json",
    "provenance/RECOVERY_LINEAGE.json",
    "provenance/executed_code_hashes.tsv",
    "provenance/frozen_resampling_parity.json",
    "logs/amendment03_targeted_tests.log",
    "logs/amendment03_syntax_compile.log",
    "logs/full_tests.log",
    "logs/cuda_reload_probe.log",
    "metrics/SMOKE_REPORT.json",
    "tables/SMOKE_REPORT.csv",
    "RUN_STATUS.txt",
})
RUNTIME_PROBE_SHA256 = (
    "41f4a3c35fd1d8308c791dab155c2a82b5e5ae20236f142645cfa71a652768d5"
)
CORRIGENDUM_PROMPT = Path(
    "/REVIEWER_INPUT_ROOT/03_RUN_NOW_Amendment03_Corrigendum_One_CUDA_Smoke.txt"
)
CORRIGENDUM_PROMPT_SHA256 = (
    "5ffa3fc2cec3b13e8e4d5d0ee2e18a906ee0677cd27502f41b0767f78fd0d2bc"
)

FAILURE_EVIDENCE_RELATIVE_HASHES = {
    ".runtime/amendment02_cuda_wheel/AMENDMENT02_HARD_STOP_REPORT.txt": (
        "b8eb3d24d7f403c5d2f794f9497b3801872fb4b4859e46ee46824adb81968ebf"
    ),
    (
        "results/run_20260813_050604_437486_bc48422a_cuda_smoke/"
        "logs/smoke_failure.txt"
    ): "e9a6d454bcd25aab8ddb05801c813f6f361afce8c6b108b20eaad48ab619ccc3",
    (
        "results/run_20260813_050604_437486_bc48422a_cuda_smoke/"
        "logs/final_package_failure.txt"
    ): "5906536e06f729e10d56ba85ca304c8a3a3b8b1aa68a120b45121a844e2e7f92",
}
ORPHAN_MODEL_RELATIVE = (
    ".runtime/smoke_models/run_20260813_050604_437486_bc48422a_cuda_smoke/"
    "full_new_reference.ubj"
)
ORPHAN_MODEL_SHA256 = (
    "c023f840f5899fd3276384e9d41ba6636c2b671e49b0eba620e0350ad3156dde"
)
RUNTIME_PROBE_RELATIVE = (
    ".runtime/amendment02_cuda_wheel/cuda_active_fit_probe.json"
)

_HASH_RE = re.compile(r"[0-9a-f]{64}")
_MANIFEST_COLUMNS = ("relative_path", "size_bytes", "sha256")


class RecoveryIntegrityError(RuntimeError):
    """An Amendment 03 recovery contract did not verify exactly."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and _HASH_RE.fullmatch(value) is not None


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_new_bytes(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RecoveryIntegrityError(f"Refusing to overwrite recovery evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_new_bytes(path, _json_bytes(payload))


def _status_fields(data: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in data.decode("utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            key = key.strip()
            if key in fields:
                raise RecoveryIntegrityError(f"Duplicate status key: {key}")
            fields[key] = value.strip()
    return fields


def _safe_member_name(name: str, expected_prefix: str) -> bool:
    path = PurePosixPath(name)
    return bool(
        name
        and "\\" not in name
        and not path.is_absolute()
        and ".." not in path.parts
        and name.startswith(expected_prefix)
        and not name.endswith("/")
    )


def _manifest_rows(data: bytes, label: str) -> dict[str, dict[str, Any]]:
    try:
        text = data.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    except Exception as exc:
        raise RecoveryIntegrityError(f"{label} is unreadable: {exc}") from exc
    if reader.fieldnames is None or tuple(reader.fieldnames[:3]) != _MANIFEST_COLUMNS:
        raise RecoveryIntegrityError(f"{label} has an invalid schema")
    rows: dict[str, dict[str, Any]] = {}
    for raw in reader:
        relative = str(raw.get("relative_path", ""))
        path = PurePosixPath(relative)
        if (
            not relative
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in relative
            or relative in rows
        ):
            raise RecoveryIntegrityError(f"{label} has an unsafe/duplicate path")
        try:
            size = int(str(raw.get("size_bytes", "")))
        except ValueError as exc:
            raise RecoveryIntegrityError(f"{label} has a noninteger size") from exc
        digest = str(raw.get("sha256", ""))
        if size < 0 or not _valid_hash(digest):
            raise RecoveryIntegrityError(f"{label} has an invalid size/hash row")
        rows[relative] = {
            "relative_path": relative,
            "size_bytes": size,
            "sha256": digest,
        }
    return rows


def _probe_contract_valid(probe: Mapping[str, Any]) -> bool:
    return bool(
        probe.get("classification") == "VERIFIED"
        and probe.get("status") == "PASS"
        and probe.get("xgboost_version") == "2.1.1"
        and probe.get("build_info", {}).get("USE_CUDA") is True
        and probe.get("requested_device") == "cuda"
        and probe.get("tree_method") == "hist"
        and probe.get("training_status") == "PASS"
        and probe.get("save_reload_predict_status") == "PASS"
        and probe.get("cpu_fallback_detected") is False
        and not probe.get("fallback_warnings")
    )


def validate_sealed_probe_contract(
    preflight_zip: Path,
    package_verification_path: Path,
    *,
    preflight_run_id: str = ACCEPTED_PREFLIGHT_RUN_ID,
    expected_zip_sha256: str = ACCEPTED_PREFLIGHT_ZIP_SHA256,
) -> tuple[dict[str, Any], bytes]:
    """Read and validate the sole authoritative historical CUDA probe."""

    preflight_zip = Path(preflight_zip)
    package_verification_path = Path(package_verification_path)
    if (
        not preflight_zip.is_file()
        or preflight_zip.is_symlink()
        or sha256_file(preflight_zip) != expected_zip_sha256
    ):
        raise RecoveryIntegrityError("Accepted Preflight ZIP hash/path contract failed")
    if not package_verification_path.is_file() or package_verification_path.is_symlink():
        raise RecoveryIntegrityError("Accepted package-verification evidence is missing")
    package = json.loads(package_verification_path.read_text())
    if (
        package.get("bundle_sha256") != expected_zip_sha256
        or package.get("crc_status") != "PASS"
        or package.get("independent_reopen_member_verification") != "PASS"
        or package.get("verification_failures")
        or package.get("package_completeness", {}).get(
            "preflight_semantic_completeness", {}
        ).get("status") != "PASS"
        or package.get("package_completeness", {}).get(
            "preflight_semantic_completeness", {}
        ).get("checks", {}).get("isolated_cuda_probe_active_pass") is not True
    ):
        raise RecoveryIntegrityError("Package verification does not accept the sealed probe")

    prefix = f"{preflight_run_id}/"
    probe_member = prefix + SEALED_PROBE_RELATIVE
    output_member = prefix + "OUTPUT_MANIFEST_FINAL.tsv"
    bundle_member = prefix + "BUNDLE_MANIFEST.tsv"
    with zipfile.ZipFile(preflight_zip) as archive:
        names = archive.namelist()
        if (
            archive.testzip() is not None
            or len(names) != len(set(names))
            or any(not _safe_member_name(name, prefix) for name in names)
        ):
            raise RecoveryIntegrityError("Accepted Preflight ZIP CRC/path contract failed")
        for member in (probe_member, output_member, bundle_member):
            if member not in names:
                raise RecoveryIntegrityError(f"Accepted Preflight ZIP omits {member}")
        probe_bytes = archive.read(probe_member)
        output_bytes = archive.read(output_member)
        bundle_bytes = archive.read(bundle_member)

    output_rows = _manifest_rows(output_bytes, "OUTPUT_MANIFEST_FINAL.tsv")
    bundle_rows = _manifest_rows(bundle_bytes, "BUNDLE_MANIFEST.tsv")
    output_row = output_rows.get(SEALED_PROBE_RELATIVE)
    bundle_row = bundle_rows.get(SEALED_PROBE_RELATIVE)
    actual = {
        "relative_path": SEALED_PROBE_RELATIVE,
        "size_bytes": len(probe_bytes),
        "sha256": sha256_bytes(probe_bytes),
    }
    if output_row != actual or bundle_row != actual:
        raise RecoveryIntegrityError("Sealed probe disagrees with a canonical Manifest")
    try:
        probe = json.loads(probe_bytes)
    except Exception as exc:
        raise RecoveryIntegrityError("Sealed probe JSON is malformed") from exc
    if not isinstance(probe, Mapping) or not _probe_contract_valid(probe):
        raise RecoveryIntegrityError("Sealed probe does not prove the locked CUDA fit")

    evidence = {
        "classification": "VERIFIED",
        "status": "PASS",
        "source_zip_path": str(preflight_zip),
        "source_zip_sha256": expected_zip_sha256,
        "source_member": SEALED_PROBE_RELATIVE,
        "source_member_full_name": probe_member,
        "member_size_bytes": len(probe_bytes),
        "member_sha256": actual["sha256"],
        "output_manifest_row": output_row,
        "bundle_manifest_row": bundle_row,
        "manifest_match": "PASS",
        "accepted_package_verification_path": str(package_verification_path),
        "accepted_package_verification_sha256": sha256_file(
            package_verification_path
        ),
        "accepted_package_verification_match": "PASS",
        "probe_contract_status": "PASS",
        "runtime_probe_used_as_source": False,
    }
    return evidence, probe_bytes


def validate_corrected_immutable_gate(
    *,
    study_root: Path = STUDY_ROOT,
    preflight_run_id: str = ACCEPTED_PREFLIGHT_RUN_ID,
    expected_zip_sha256: str = ACCEPTED_PREFLIGHT_ZIP_SHA256,
    expected_failure_hashes: Mapping[str, str] = FAILURE_EVIDENCE_RELATIVE_HASHES,
    expected_orphan_sha256: str = ORPHAN_MODEL_SHA256,
    expected_runtime_probe_sha256: str = RUNTIME_PROBE_SHA256,
) -> dict[str, Any]:
    """Validate every corrected pre-edit immutable reference without fallback."""

    root = Path(study_root)
    results = root / "results"
    preflight_run = results / preflight_run_id
    preflight_zip = results / f"{preflight_run_id}_review_bundle.zip"
    package_verification = results / f"{preflight_run_id}_package_verification.json"
    sealed, _ = validate_sealed_probe_contract(
        preflight_zip,
        package_verification,
        preflight_run_id=preflight_run_id,
        expected_zip_sha256=expected_zip_sha256,
    )

    with zipfile.ZipFile(preflight_zip) as archive:
        prefix = f"{preflight_run_id}/"
        status_bytes = archive.read(prefix + "RUN_STATUS.txt")
        identity_bytes = archive.read(prefix + "config/run_identity.lock.json")
        locked_split_bytes = archive.read(prefix + "config/locked_split_identity.json")
        output_bytes = archive.read(prefix + "OUTPUT_MANIFEST_FINAL.tsv")
        bundle_bytes = archive.read(prefix + "BUNDLE_MANIFEST.tsv")
    status = _status_fields(status_bytes)
    identity = json.loads(identity_bytes)
    locked_split = json.loads(locked_split_bytes)
    if not (
        status.get("RUN_STATE") == "PREFLIGHT_COMPLETE"
        and status.get("PROTOCOL_AMENDMENT") == "AMENDMENT_02"
        and status.get("SMOKE_ELIGIBLE") == "YES"
        and status.get("CUDA_SMOKE_ELIGIBLE") == "YES"
        and status.get("FULL_AUTHORIZED") == "NO"
        and status.get("BLOCKERS") == "NONE"
        and status.get("LOCKED_SPLIT_SHA256") == LOCKED_SPLIT_SHA256
        and status.get("BASE_SEED") == "42"
        and identity.get("run_id") == preflight_run_id
        and identity.get("state") == "PREFLIGHT_COMPLETE"
        and identity.get("protocol") == "AMENDMENT_02"
        and identity.get("candidate_split_hash") == LOCKED_SPLIT_SHA256
        and identity.get("base_seed") == 42
        and identity.get("full_authorized") is False
        and identity.get("split_locked") is True
        and locked_split.get("status") == "PASS"
        and locked_split.get("base_seed") == 42
        and locked_split.get("split_assignment_sha256") == LOCKED_SPLIT_SHA256
    ):
        raise RecoveryIntegrityError("Accepted Preflight authorization contract failed")

    output_rows = _manifest_rows(output_bytes, "OUTPUT_MANIFEST_FINAL.tsv")
    bundle_rows = _manifest_rows(bundle_bytes, "BUNDLE_MANIFEST.tsv")
    if set(bundle_rows) != set(output_rows) | {"OUTPUT_MANIFEST_FINAL.tsv"}:
        raise RecoveryIntegrityError("Accepted Preflight Manifest member sets disagree")
    if not preflight_run.is_dir() or preflight_run.is_symlink():
        raise RecoveryIntegrityError("Accepted live Preflight directory is missing")
    live_failures: list[str] = []
    for relative, row in bundle_rows.items():
        path = preflight_run / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row["size_bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            live_failures.append(relative)
    bundle_manifest_live = preflight_run / "BUNDLE_MANIFEST.tsv"
    if bundle_manifest_live.read_bytes() != bundle_bytes:
        live_failures.append("BUNDLE_MANIFEST.tsv")
    if live_failures:
        raise RecoveryIntegrityError(
            f"Accepted live Preflight differs from its sealed ZIP: {live_failures}"
        )

    corrected: dict[str, dict[str, Any]] = {}
    for relative, expected in expected_failure_hashes.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise RecoveryIntegrityError(f"Corrected failure-evidence pin failed: {path}")
        corrected[relative] = {"path": str(path), "sha256": expected}
    orphan = root / ORPHAN_MODEL_RELATIVE
    if (
        not orphan.is_file()
        or orphan.is_symlink()
        or sha256_file(orphan) != expected_orphan_sha256
    ):
        raise RecoveryIntegrityError("Failed orphan model pin failed")
    runtime_probe = root / RUNTIME_PROBE_RELATIVE
    if (
        not runtime_probe.is_file()
        or runtime_probe.is_symlink()
        or sha256_file(runtime_probe) != expected_runtime_probe_sha256
    ):
        raise RecoveryIntegrityError("Diagnostic-only runtime probe pin failed")

    code_tree = identity.get("code_tree")
    if not isinstance(code_tree, Mapping) or not code_tree or not all(
        isinstance(relative, str) and _valid_hash(digest)
        for relative, digest in code_tree.items()
    ):
        raise RecoveryIntegrityError("Accepted Preflight code tree is invalid")
    if (
        not CORRIGENDUM_PROMPT.is_file()
        or CORRIGENDUM_PROMPT.is_symlink()
        or sha256_file(CORRIGENDUM_PROMPT) != CORRIGENDUM_PROMPT_SHA256
    ):
        raise RecoveryIntegrityError("Corrigendum prompt path/hash pin failed")
    return {
        "classification": "VERIFIED",
        "status": "PASS",
        "accepted_preflight_run": str(preflight_run),
        "accepted_preflight_run_id": preflight_run_id,
        "accepted_preflight_zip": str(preflight_zip),
        "accepted_preflight_zip_sha256": expected_zip_sha256,
        "locked_split_sha256": LOCKED_SPLIT_SHA256,
        "base_seed": 42,
        "sealed_probe": sealed,
        "corrected_failure_evidence": corrected,
        "orphan_model": {
            "path": str(orphan),
            "sha256": expected_orphan_sha256,
            "diagnostic_only": True,
        },
        "runtime_probe": {
            "path": str(runtime_probe),
            "sha256": expected_runtime_probe_sha256,
            "used_as_source": False,
            "diagnostic_only": True,
        },
        "reviewed_code_tree": dict(sorted(code_tree.items())),
        "corrigendum_prompt": {
            "path": str(CORRIGENDUM_PROMPT),
            "sha256": CORRIGENDUM_PROMPT_SHA256,
            "immutable": True,
        },
    }


def assert_replacement_guard(
    *,
    results_root: Path = RESULTS_ROOT,
    new_run_id: str | None = None,
) -> dict[str, Any]:
    """Allow the pinned failed Smoke plus zero or one named replacement only."""

    root = Path(results_root)
    if not root.is_dir() or root.is_symlink():
        raise RecoveryIntegrityError("Results root is missing or unsafe")
    smoke_entries = sorted(
        path.name for path in root.iterdir() if "cuda_smoke" in path.name.lower()
    )
    allowed = {FAILED_SMOKE_RUN_ID}
    if new_run_id is not None:
        if (
            not new_run_id
            or new_run_id == FAILED_SMOKE_RUN_ID
            or "cuda_smoke" not in new_run_id.lower()
            or Path(new_run_id).name != new_run_id
        ):
            raise RecoveryIntegrityError("Invalid Amendment 03 replacement Run ID")
        allowed.update({
            new_run_id,
            f"{new_run_id}_review_bundle.zip",
            f"{new_run_id}_package_verification.json",
            f"{new_run_id}_NON_SCIENTIFIC_smoke_models.zip",
            (
                f"{new_run_id}_NON_SCIENTIFIC_smoke_models_"
                "package_verification.json"
            ),
        })
    unexpected = sorted(set(smoke_entries) - allowed)
    failed = root / FAILED_SMOKE_RUN_ID
    if not failed.is_dir() or failed.is_symlink() or unexpected:
        raise RecoveryIntegrityError(
            "Amendment 03 replacement guard failed; "
            f"unexpected Smoke outputs={unexpected}"
        )
    if new_run_id is None:
        if smoke_entries != [FAILED_SMOKE_RUN_ID]:
            raise RecoveryIntegrityError("Replacement authorization is not 0-of-1")
        authorization = "0_OF_1_BEFORE_RUN_CREATION"
    else:
        replacement = root / new_run_id
        if not replacement.is_dir() or replacement.is_symlink():
            raise RecoveryIntegrityError("Named replacement Run directory is absent")
        authorization = "1_OF_1_CONSUMED"
    return {
        "status": "PASS",
        "failed_smoke_run_id": FAILED_SMOKE_RUN_ID,
        "observed_smoke_outputs": smoke_entries,
        "replacement_run_id": new_run_id,
        "replacement_authorization": authorization,
    }


def create_authorized_replacement_run_directory(
    path: Path,
    *,
    results_root: Path = RESULTS_ROOT,
) -> dict[str, Any]:
    """Atomically consume the one replacement authorization by creating its Run."""

    target = Path(path)
    root = Path(results_root).resolve()
    if target.parent.resolve() != root:
        raise RecoveryIntegrityError("Replacement Run must be directly under results")
    assert_replacement_guard(results_root=root)
    try:
        target.mkdir(mode=0o755, exist_ok=False)
    except FileExistsError as exc:
        raise RecoveryIntegrityError("Replacement Run authorization already consumed") from exc
    return assert_replacement_guard(results_root=root, new_run_id=target.name)


def copy_sealed_probe_to_run(
    *,
    run_dir: Path,
    preflight_zip: Path,
    package_verification_path: Path,
    preflight_run_id: str = ACCEPTED_PREFLIGHT_RUN_ID,
    expected_zip_sha256: str = ACCEPTED_PREFLIGHT_ZIP_SHA256,
) -> dict[str, Any]:
    """Copy identical sealed member bytes to both required Smoke paths."""

    evidence, probe_bytes = validate_sealed_probe_contract(
        preflight_zip,
        package_verification_path,
        preflight_run_id=preflight_run_id,
        expected_zip_sha256=expected_zip_sha256,
    )
    destinations: dict[str, dict[str, Any]] = {}
    for relative in SEALED_PROBE_DESTINATIONS:
        destination = Path(run_dir) / relative
        _write_new_bytes(destination, probe_bytes)
        destinations[relative] = {
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "byte_identical_to_sealed_member": destination.read_bytes() == probe_bytes,
        }
    expected_hash = evidence["member_sha256"]
    if any(
        record["sha256"] != expected_hash
        or record["size_bytes"] != len(probe_bytes)
        or record["byte_identical_to_sealed_member"] is not True
        for record in destinations.values()
    ):
        raise RecoveryIntegrityError("A sealed probe destination changed during copy")
    binding = {
        **evidence,
        "destinations": destinations,
        "both_destinations_byte_identical": True,
        "runtime_probe_used_as_source": False,
    }
    _write_new_json(Path(run_dir) / "provenance/sealed_probe_binding.json", binding)
    return binding


def _live_code_tree(study_root: Path) -> dict[str, str]:
    root = Path(study_root)
    paths = [
        *sorted((root / "code").glob("*.py")),
        *sorted((root / "code").glob("*.sh")),
        *sorted((root / "tests").glob("*.py")),
    ]
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in paths if path.is_file() and not path.is_symlink()
    }


def build_executed_code_provenance(
    *,
    run_dir: Path,
    reviewed_code_tree: Mapping[str, str],
    allowed_after_hashes: Mapping[str, str],
    study_root: Path = STUDY_ROOT,
) -> dict[str, Any]:
    """Snapshot the exact live code/tests under a hash-specific drift allow-list."""

    root = Path(study_root)
    reviewed = dict(reviewed_code_tree)
    allowed = dict(allowed_after_hashes)
    if not all(_valid_hash(value) for value in [*reviewed.values(), *allowed.values()]):
        raise RecoveryIntegrityError("Executed-code provenance contains an invalid hash")
    if not set(allowed) <= (set(reviewed) | set(_live_code_tree(root))):
        raise RecoveryIntegrityError("Allow-list names a missing code/test path")
    live = _live_code_tree(root)
    universe = set(reviewed) | set(live)
    changed = {
        relative for relative in universe
        if reviewed.get(relative) != live.get(relative)
    }
    if changed != set(allowed):
        raise RecoveryIntegrityError(
            "Executed code drift is not the exact allow-list: "
            f"changed={sorted(changed)}, allowed={sorted(allowed)}"
        )
    if any(live.get(relative) != digest for relative, digest in allowed.items()):
        raise RecoveryIntegrityError("An allow-listed executed hash does not match live bytes")

    rows: list[dict[str, Any]] = []
    snapshot_hashes: dict[str, str] = {}
    for relative in sorted(live):
        source = root / relative
        category, name = relative.split("/", 1)
        snapshot_relative = (
            f"provenance/executed_{'code' if category == 'code' else 'tests'}_snapshot/"
            f"{name}"
        )
        destination = Path(run_dir) / snapshot_relative
        data = source.read_bytes()
        _write_new_bytes(destination, data)
        actual = sha256_bytes(data)
        if actual != live[relative] or sha256_file(destination) != actual:
            raise RecoveryIntegrityError("Executed snapshot bytes changed during copy")
        snapshot_hashes[snapshot_relative] = actual
        before = reviewed.get(relative, "NOT_PRESENT_IN_ACCEPTED_PREFLIGHT")
        rows.append({
            "relative_path": relative,
            "reviewed_sha256": before,
            "executed_sha256": actual,
            "snapshot_relative_path": snapshot_relative,
            "snapshot_sha256": actual,
            "drift_status": "ALLOWLISTED_CHANGE" if relative in changed else "UNCHANGED",
            "allowlisted": "YES" if relative in changed else "NO",
        })
    removed = sorted(set(reviewed) - set(live))
    if removed:
        raise RecoveryIntegrityError(f"Reviewed code/test files were removed: {removed}")

    columns = (
        "relative_path", "reviewed_sha256", "executed_sha256",
        "snapshot_relative_path", "snapshot_sha256", "drift_status",
        "allowlisted",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    tsv = stream.getvalue().encode("utf-8")
    tsv_path = Path(run_dir) / "provenance/executed_code_hashes.tsv"
    _write_new_bytes(tsv_path, tsv)
    return {
        "status": "PASS",
        "reviewed_file_count": len(reviewed),
        "executed_file_count": len(live),
        "changed_paths": sorted(changed),
        "allowed_after_hashes": dict(sorted(allowed.items())),
        "executed_code_hashes_tsv_sha256": sha256_bytes(tsv),
        "snapshot_hashes": snapshot_hashes,
    }


def validate_executed_code_provenance(
    *,
    run_dir: Path,
    reviewed_code_tree: Mapping[str, str],
    allowed_after_hashes: Mapping[str, str],
    study_root: Path = STUDY_ROOT,
) -> dict[str, Any]:
    """Recompute live, TSV, and snapshot hashes without trusting stored claims."""

    root = Path(study_root)
    run = Path(run_dir)
    tsv_path = run / "provenance/executed_code_hashes.tsv"
    if not tsv_path.is_file() or tsv_path.is_symlink():
        raise RecoveryIntegrityError("Executed-code hash TSV is missing")
    with tsv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    live = _live_code_tree(root)
    if len(rows) != len(live) or len({row.get("relative_path") for row in rows}) != len(rows):
        raise RecoveryIntegrityError("Executed-code TSV row set is invalid")
    reviewed = dict(reviewed_code_tree)
    allowed = dict(allowed_after_hashes)
    changed: set[str] = set()
    for row in rows:
        relative = str(row.get("relative_path", ""))
        executed = live.get(relative)
        snapshot_relative = str(row.get("snapshot_relative_path", ""))
        snapshot = run / snapshot_relative
        before = reviewed.get(relative, "NOT_PRESENT_IN_ACCEPTED_PREFLIGHT")
        is_changed = before != executed
        if is_changed:
            changed.add(relative)
        if not (
            executed is not None
            and row.get("reviewed_sha256") == before
            and row.get("executed_sha256") == executed
            and row.get("snapshot_sha256") == executed
            and snapshot.is_file()
            and not snapshot.is_symlink()
            and sha256_file(snapshot) == executed
            and row.get("drift_status")
            == ("ALLOWLISTED_CHANGE" if is_changed else "UNCHANGED")
            and row.get("allowlisted") == ("YES" if is_changed else "NO")
        ):
            raise RecoveryIntegrityError(f"Executed-code provenance mismatch: {relative}")
    if changed != set(allowed) or any(live.get(path) != digest for path, digest in allowed.items()):
        raise RecoveryIntegrityError("Executed-code allow-list no longer matches live bytes")
    return {
        "status": "PASS",
        "executed_file_count": len(rows),
        "changed_paths": sorted(changed),
        "executed_code_hashes_tsv_sha256": sha256_file(tsv_path),
    }


def build_recovery_lineage(
    *,
    run_dir: Path,
    new_run_id: str,
    immutable_gate: Mapping[str, Any],
    executed_code: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the replacement Run to every prior failure and executed-code byte."""

    if immutable_gate.get("status") != "PASS" or executed_code.get("status") != "PASS":
        raise RecoveryIntegrityError("Cannot create lineage from unverified evidence")
    if Path(run_dir).name != new_run_id:
        raise RecoveryIntegrityError("Recovery lineage Run ID/path mismatch")
    lineage = {
        "classification": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
        "status": "PASS",
        "authorization": "AMENDMENT_03_CORRIGENDUM_ONE",
        "new_smoke_run_id": new_run_id,
        "accepted_preflight": {
            "run_id": immutable_gate["accepted_preflight_run_id"],
            "run_path": immutable_gate["accepted_preflight_run"],
            "review_zip_path": immutable_gate["accepted_preflight_zip"],
            "review_zip_sha256": immutable_gate["accepted_preflight_zip_sha256"],
        },
        "sealed_probe": immutable_gate["sealed_probe"],
        "failed_smoke": {
            "run_id": FAILED_SMOKE_RUN_ID,
            "run_path": str(Path(STUDY_ROOT) / "results" / FAILED_SMOKE_RUN_ID),
            "preserved": True,
        },
        "corrected_failure_evidence": immutable_gate["corrected_failure_evidence"],
        "orphan_model": immutable_gate["orphan_model"],
        "runtime_probe": immutable_gate["runtime_probe"],
        "previous_prompt5": {
            "classification": "PROMPT_CONTRACT_FAILURE_ONLY",
            "stopped_pre_edit": True,
            "run_created": False,
            "replacement_authorization_consumed": False,
        },
        "scientific_contract": {
            "protocol": "AMENDMENT_02_UNCHANGED",
            "split_sha256": LOCKED_SPLIT_SHA256,
            "base_seed": 42,
            "full_authorized": False,
        },
        "executed_code": {
            "changed_paths": executed_code["changed_paths"],
            "allowed_after_hashes": executed_code["allowed_after_hashes"],
            "executed_code_hashes_tsv_sha256": executed_code[
                "executed_code_hashes_tsv_sha256"
            ],
            "snapshot_hashes": executed_code["snapshot_hashes"],
        },
        **({
            "corrigendum_prompt": immutable_gate["corrigendum_prompt"],
        } if immutable_gate.get("corrigendum_prompt") else {}),
    }
    _write_new_json(Path(run_dir) / "provenance/RECOVERY_LINEAGE.json", lineage)
    return lineage


def validate_recovery_artifacts(
    *,
    run_dir: Path,
    reviewed_code_tree: Mapping[str, str],
    allowed_after_hashes: Mapping[str, str],
    study_root: Path = STUDY_ROOT,
    require_prompt_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate the recovery-owned artifacts before Review packaging."""

    run = Path(run_dir)
    binding_path = run / "provenance/sealed_probe_binding.json"
    lineage_path = run / "provenance/RECOVERY_LINEAGE.json"
    if not binding_path.is_file() or not lineage_path.is_file():
        raise RecoveryIntegrityError("Recovery binding or lineage is missing")
    binding = json.loads(binding_path.read_text())
    lineage = json.loads(lineage_path.read_text())
    if require_prompt_artifacts:
        missing = sorted(
            relative for relative in AMENDMENT03_RECOVERY_REQUIRED_FILES
            if not (run / relative).is_file()
            or (run / relative).is_symlink()
            or (run / relative).stat().st_size == 0
        )
        executed_code_files = list(
            (run / "provenance/executed_code_snapshot").glob("*")
        )
        executed_test_files = list(
            (run / "provenance/executed_tests_snapshot").glob("*")
        )
        if missing or not executed_code_files or not executed_test_files:
            raise RecoveryIntegrityError(
                "Amendment 03 required artifacts are incomplete: "
                f"missing={missing}, code_snapshot={len(executed_code_files)}, "
                f"tests_snapshot={len(executed_test_files)}"
            )
    probe_hash = binding.get("member_sha256")
    destinations = binding.get("destinations", {})
    manifest_rows = binding.get("manifest_rows", {})
    output_manifest_row = binding.get("output_manifest_row") or (
        manifest_rows.get("OUTPUT_MANIFEST_FINAL.tsv")
        if isinstance(manifest_rows, Mapping) else None
    )
    bundle_manifest_row = binding.get("bundle_manifest_row") or (
        manifest_rows.get("BUNDLE_MANIFEST.tsv")
        if isinstance(manifest_rows, Mapping) else None
    )
    expected_probe_row = {
        "relative_path": SEALED_PROBE_RELATIVE,
        "size_bytes": binding.get("member_size_bytes"),
        "sha256": probe_hash,
    }
    if (
        binding.get("status") != "PASS"
        or binding.get("source_zip_sha256") != ACCEPTED_PREFLIGHT_ZIP_SHA256
        or binding.get("source_member") != SEALED_PROBE_RELATIVE
        or binding.get("manifest_match") != "PASS"
        or output_manifest_row != expected_probe_row
        or bundle_manifest_row != expected_probe_row
        or binding.get("runtime_probe_used_as_source") is not False
        or not (
            binding.get("both_destinations_byte_identical") is True
            or binding.get("root_nested_byte_equal") is True
        )
        or set(destinations) != set(SEALED_PROBE_DESTINATIONS)
        or not _valid_hash(probe_hash)
    ):
        raise RecoveryIntegrityError("Sealed-probe binding schema is invalid")
    destination_bytes: list[bytes] = []
    for relative in SEALED_PROBE_DESTINATIONS:
        path = run / relative
        record = destinations[relative]
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != probe_hash
            or record.get("sha256") != probe_hash
            or record.get("size_bytes") != path.stat().st_size
            or not (
                record.get("byte_identical_to_sealed_member") is True
                or record.get("byte_equal_to_sealed_member") is True
            )
        ):
            raise RecoveryIntegrityError(f"Sealed-probe copy failed validation: {relative}")
        destination_bytes.append(path.read_bytes())
    if destination_bytes[0] != destination_bytes[1]:
        raise RecoveryIntegrityError("Root/nested sealed probe bytes differ")

    executed = validate_executed_code_provenance(
        run_dir=run,
        reviewed_code_tree=reviewed_code_tree,
        allowed_after_hashes=allowed_after_hashes,
        study_root=study_root,
    )
    expected_failure_hashes = set(FAILURE_EVIDENCE_RELATIVE_HASHES.values())
    recorded_failure_hashes = {
        record.get("sha256")
        for record in lineage.get("corrected_failure_evidence", {}).values()
        if isinstance(record, Mapping)
    }
    if not (
        lineage.get("status") == "PASS"
        and lineage.get("authorization") == "AMENDMENT_03_CORRIGENDUM_ONE"
        and lineage.get("new_smoke_run_id") == run.name
        and lineage.get("accepted_preflight", {}).get("review_zip_sha256")
        == ACCEPTED_PREFLIGHT_ZIP_SHA256
        and lineage.get("sealed_probe", {}).get("member_sha256") == probe_hash
        and lineage.get("failed_smoke", {}).get("run_id") == FAILED_SMOKE_RUN_ID
        and lineage.get("failed_smoke", {}).get("preserved") is True
        and recorded_failure_hashes == expected_failure_hashes
        and lineage.get("orphan_model", {}).get("sha256") == ORPHAN_MODEL_SHA256
        and lineage.get("orphan_model", {}).get("diagnostic_only") is True
        and lineage.get("runtime_probe", {}).get("sha256") == RUNTIME_PROBE_SHA256
        and lineage.get("runtime_probe", {}).get("used_as_source") is False
        and lineage.get("previous_prompt5", {}).get("stopped_pre_edit") is True
        and lineage.get("previous_prompt5", {}).get("run_created") is False
        and lineage.get("scientific_contract", {}).get("split_sha256")
        == LOCKED_SPLIT_SHA256
        and lineage.get("scientific_contract", {}).get("base_seed") == 42
        and lineage.get("scientific_contract", {}).get("full_authorized") is False
        and lineage.get("executed_code", {}).get(
            "executed_code_hashes_tsv_sha256"
        ) == executed["executed_code_hashes_tsv_sha256"]
        and lineage.get("executed_code", {}).get("allowed_after_hashes")
        == dict(sorted(allowed_after_hashes.items()))
        and (
            "corrigendum_prompt" not in lineage
            or lineage.get("corrigendum_prompt") == {
                "path": str(CORRIGENDUM_PROMPT),
                "sha256": CORRIGENDUM_PROMPT_SHA256,
                "immutable": True,
            }
        )
    ):
        raise RecoveryIntegrityError("Recovery lineage contract is invalid")
    return {
        "status": "PASS",
        "sealed_probe_sha256": probe_hash,
        "executed_code": executed,
        "recovery_lineage_sha256": sha256_file(lineage_path),
    }


__all__ = [
    "ACCEPTED_PREFLIGHT_RUN_ID",
    "ACCEPTED_PREFLIGHT_ZIP_SHA256",
    "AMENDMENT03_RECOVERY_REQUIRED_FILES",
    "CORRIGENDUM_PROMPT",
    "CORRIGENDUM_PROMPT_SHA256",
    "FAILED_SMOKE_RUN_ID",
    "FAILURE_EVIDENCE_RELATIVE_HASHES",
    "LOCKED_SPLIT_SHA256",
    "ORPHAN_MODEL_SHA256",
    "RUNTIME_PROBE_SHA256",
    "RecoveryIntegrityError",
    "assert_replacement_guard",
    "build_executed_code_provenance",
    "build_recovery_lineage",
    "copy_sealed_probe_to_run",
    "create_authorized_replacement_run_directory",
    "sha256_bytes",
    "sha256_file",
    "validate_corrected_immutable_gate",
    "validate_executed_code_provenance",
    "validate_recovery_artifacts",
    "validate_sealed_probe_contract",
]
