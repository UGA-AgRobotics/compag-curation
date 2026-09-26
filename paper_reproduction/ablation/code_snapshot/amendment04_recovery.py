"""Fail-closed Amendment 04 CSV round-trip recovery helpers.

This module owns no training entry point.  It validates the immutable recovery
chain, implements the one-Run authorization guard, snapshots executed code,
and provides the sole exact CSV reader authorized by Amendment 04.
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shlex
import struct
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, MutableMapping, Sequence

import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = STUDY_ROOT / "results"

AUTHORIZATION = "AMENDMENT_04_CSV_ROUNDTRIP_ONE"
PROMPT_PATH = Path(
    "/REVIEWER_INPUT_ROOT/04_RUN_NOW_Amendment04_CSV_RoundTrip_One_CUDA_Smoke.txt"
)
PROMPT_SHA256 = (
    "8028d8a0f60caa9817214a1d973dd2f95f67f0291e7f06213191972fef9133a3"
)

CUDA_ENV = Path(
    "/REVIEWER_INPUT_ROOT/SAM_ablation_cuda_xgb211_wheel"
)
CUDA_PYTHON = CUDA_ENV / "bin/python"
XGBOOST_VERSION = "2.1.1"
SCIKIT_LEARN_VERSION = "1.7.2"
VISIBLE_GPU = "NVIDIA GeForce RTX 4090"
SEALED_PROBE_SHA256 = (
    "629ce45ce6c23bbc46feac26cbec3ec8cfa7f9e333c3ef595f74065e4c6820a9"
)
SEALED_PROBE_SIZE_BYTES = 16941
SEALED_PROBE_RELATIVE = "provenance/cuda_active_fit_probe.json"
SEALED_PROBE_DESTINATIONS = (
    "provenance/cuda_active_fit_probe.json",
    "provenance/preflight_snapshot/cuda_active_fit_probe.json",
)

ACCEPTED_PREFLIGHT_RUN_ID = (
    "run_20260813_050405_9970c566_amendment02_preflight"
)
ACCEPTED_PREFLIGHT_ZIP_SHA256 = (
    "19d6d9ec381bab99ff41a081fd42016c653d4a4ce947f35694e0e32e57ae24cb"
)
ACCEPTED_PREFLIGHT_PACKAGE_VERIFICATION_SHA256 = (
    "a970069fa83da886fe1a9e9c524b7066f21ce8ba322c49cd5476396e718f5d5f"
)
ACCEPTED_PREFLIGHT_TREE = {
    "file_count": 73,
    "total_size_bytes": 25543570,
    "tree_sha256": "e26fe5daa0af6415df694fb674eedebe329e6d5a53e8b5061eb4013186477641",
}
LOCKED_SPLIT_SHA256 = (
    "f2cd1570d2c76af49ed52593d95657a3cbbe14d8f2f077edd05447e608698c64"
)
BASE_SEED = 42

FAILED_AMENDMENT02_RUN_ID = "run_20260813_050604_437486_bc48422a_cuda_smoke"
FAILED_AMENDMENT02_TREE = {
    "file_count": 71,
    "total_size_bytes": 43447340,
    "tree_sha256": "68428806a048458fa848297449fc30c871b29a9aa0451d4eaf48cd76e882e508",
}
FAILED_AMENDMENT02_RUNTIME_TREE = {
    "file_count": 1,
    "total_size_bytes": 113129,
    "tree_sha256": "06d205e45032e423afba6e87bd22d6c392756426e17ab9490eca9eb93098926b",
}
FAILED_AMENDMENT02_EVIDENCE_HASHES = {
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
FAILED_AMENDMENT02_ORPHAN_MODEL_SHA256 = (
    "c023f840f5899fd3276384e9d41ba6636c2b671e49b0eba620e0350ad3156dde"
)
DIAGNOSTIC_RUNTIME_PROBE_SHA256 = (
    "41f4a3c35fd1d8308c791dab155c2a82b5e5ae20236f142645cfa71a652768d5"
)

FAILED_AMENDMENT03_RUN_ID = (
    "run_20260813_125445_373539_bc48422a_amendment03_cuda_smoke"
)
FAILED_AMENDMENT03_TREE = {
    "file_count": 99,
    "total_size_bytes": 21623453,
    "tree_sha256": "7f428c1bd7845adc509823f3986430336629cc0f22017748f7ab9c18c7c5841a",
}
FAILED_AMENDMENT03_RUNTIME_TREE = {
    "file_count": 11,
    "total_size_bytes": 1243356,
    "tree_sha256": "057dae64a99b0d7ae686f803252995891f825c5553dc3a6266ed7fea42f1dd63",
}
FAILED_AMENDMENT03_FORENSIC_ZIP_SHA256 = (
    "b067ab934f660b4c30c2143658028ab91bea0aae978c3ea35051bb1eebe9d791"
)
FAILED_AMENDMENT03_FORENSIC_MEMBERS = 114
FAILED_AMENDMENT03_FORENSIC_REGULAR_FILES = 99
FAILED_AMENDMENT03_FORENSIC_REGULAR_BYTES = 21623453

FAILED_AMENDMENT03_KEY_HASHES = {
    "RUN_STATUS.txt": "430ed00e609df616b8d9509ce17eb5767739be7b2440a13fce021a098a68e525",
    "metrics/SMOKE_REPORT.json": "43ec54661e1c795dbf48d45da5414798b19a3b5494af9cf26f6d309fb813d626",
    "tables/SMOKE_REPORT.csv": "ddf7755e216bee905f34606ef9e13fd238fde8ea2e136bfdb226a458d4704b23",
    "provenance/execution_ledger.json": "b5ff09ff81b1e1905d222bd9839ebabe43fbc16d5cac9fabe34c9f37977464b0",
    "provenance/frozen_resampling_parity.json": "29694e49fa50da5c7d256c1b07cfed0e0e306bca515cb2d2849ab0c5b289e154",
    "logs/smoke_failure.txt": "da04fe2fdadd3ce99532c85c619d1fda5aefe12174af806013a60c31de6188be",
    "logs/final_package_failure.txt": "da04fe2fdadd3ce99532c85c619d1fda5aefe12174af806013a60c31de6188be",
    "provenance/executed_code_hashes.tsv": "1fd28654a59de6091b659a7f123efd4179631d046e1d2d8ae32beb6e7f5a4d14",
    "config/run_identity.lock.json": "3a0cb28eaf1fd8fec2d1880d582c7faedc3bb0ad6cdfe09bcab8647216e3acd8",
    "provenance/evidence_gate_registry.json": "8da2959d9c14b199d67903af277379f4db10cf3384ba4032ea54787d15d69bc7",
    "config/preflight_reference.json": "718bdd061e43b9d59903a2fae860a3a92f8507f929bbf4246ef676b17b7e315c",
}

EXPECTED_VARIANT_IDS = frozenset({
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
})
METRIC_NAMES = (
    "average_precision",
    "roc_auc",
    "precision",
    "recall",
    "f1",
)

AMENDMENT04_ALLOWED_CODE_DRIFT_PATHS = frozenset({
    "code/amendment04_recovery.py",
    "code/amendment_core.py",
    "code/run_study.py",
    "tests/test_amendment04.py",
})
AMENDMENT04_REFERENCE_CODE_DRIFT_PATHS = frozenset({
    "code/amendment03_compat.py",
    "code/amendment03_recovery.py",
    "code/amendment04_recovery.py",
    "code/amendment_core.py",
    "code/run_study.py",
    "tests/test_amendment03.py",
    "tests/test_amendment03_recovery.py",
    "tests/test_amendment04.py",
})

AMENDMENT04_RUNTIME_RELATIVE = Path(".runtime/amendment04_csv_roundtrip")
AMENDMENT04_PRERUN_COPY_MAPPING = {
    "docs/PROTOCOL_AMENDMENT_03.md": "docs/PROTOCOL_AMENDMENT_03.md",
    "docs/PROTOCOL_AMENDMENT_04.md": "docs/PROTOCOL_AMENDMENT_04.md",
    str(AMENDMENT04_RUNTIME_RELATIVE / "amendment04_live_cuda_binding.json"): (
        "provenance/amendment04_live_cuda_binding.json"
    ),
    str(AMENDMENT04_RUNTIME_RELATIVE / "csv_roundtrip_diagnostic.json"): (
        "provenance/csv_roundtrip_diagnostic.json"
    ),
    str(AMENDMENT04_RUNTIME_RELATIVE / "amendment04_syntax_compile.log"): (
        "logs/amendment04_syntax_compile.log"
    ),
    str(AMENDMENT04_RUNTIME_RELATIVE / "amendment04_targeted_tests.log"): (
        "logs/amendment04_targeted_tests.log"
    ),
    str(AMENDMENT04_RUNTIME_RELATIVE / "full_tests.log"): "logs/full_tests.log",
    str(AMENDMENT04_RUNTIME_RELATIVE / "cuda_reload_probe.log"): (
        "logs/cuda_reload_probe.log"
    ),
    str(AMENDMENT04_RUNTIME_RELATIVE / "package_rehearsal.log"): (
        "logs/package_rehearsal.log"
    ),
}
AMENDMENT04_PRERUN_DESTINATIONS = frozenset(
    AMENDMENT04_PRERUN_COPY_MAPPING.values()
)

AMENDMENT04_RECOVERY_REQUIRED_FILES = frozenset({
    "docs/PROTOCOL_AMENDMENT_03.md",
    "docs/PROTOCOL_AMENDMENT_04.md",
    "provenance/AMENDMENT04_LINEAGE.json",
    "provenance/csv_roundtrip_diagnostic.json",
    "provenance/cuda_active_fit_probe.json",
    "provenance/preflight_snapshot/cuda_active_fit_probe.json",
    "provenance/amendment04_live_cuda_binding.json",
    "provenance/sealed_probe_binding.json",
    "provenance/executed_code_hashes.tsv",
    "provenance/frozen_resampling_parity.json",
    "logs/amendment04_syntax_compile.log",
    "logs/amendment04_targeted_tests.log",
    "logs/full_tests.log",
    "logs/cuda_reload_probe.log",
    "logs/package_rehearsal.log",
    "metrics/SMOKE_REPORT.json",
    "tables/SMOKE_REPORT.csv",
    "RUN_STATUS.txt",
})

_HASH_RE = re.compile(r"[0-9a-f]{64}")
_MANIFEST_COLUMNS = ("relative_path", "size_bytes", "sha256")


class RecoveryIntegrityError(RuntimeError):
    """An Amendment 04 recovery contract failed exact verification."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and _HASH_RE.fullmatch(value) is not None


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RecoveryIntegrityError("Recovery JSON is not finite/serializable") from exc
    return (text + "\n").encode("utf-8")


def _write_new_bytes(path: Path, data: bytes) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise RecoveryIntegrityError(f"Refusing to overwrite recovery evidence: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_new_bytes(path, _json_bytes(payload))


def _safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(
        value
        and "\\" not in value
        and not path.is_absolute()
        and ".." not in path.parts
        and not value.endswith("/")
    )


def _status_fields(data: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in data.decode("utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if key in fields:
            raise RecoveryIntegrityError(f"Duplicate status field: {key}")
        fields[key] = value.strip()
    return fields


def _tree_contract(path: Path) -> dict[str, Any]:
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise RecoveryIntegrityError(f"Immutable directory is absent/unsafe: {root}")
    symlinks = sorted(
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_symlink()
    )
    if symlinks:
        raise RecoveryIntegrityError(f"Immutable tree contains symlinks: {symlinks}")
    rows: list[tuple[bytes, str, int]] = []
    for item in root.rglob("*"):
        if item.is_file():
            relative = item.relative_to(root).as_posix().encode("utf-8")
            rows.append((relative, sha256_file(item), item.stat().st_size))
    rows.sort(key=lambda row: row[0])
    manifest = b"".join(
        digest.encode("ascii")
        + b"\t"
        + str(size).encode("ascii")
        + b"\t"
        + relative
        + b"\n"
        for relative, digest, size in rows
    )
    return {
        "path": str(root),
        "file_count": len(rows),
        "total_size_bytes": sum(row[2] for row in rows),
        "symlink_count": 0,
        "tree_sha256": sha256_bytes(manifest),
    }


def _require_tree(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = _tree_contract(path)
    for key in ("file_count", "total_size_bytes", "tree_sha256"):
        if observed[key] != expected[key]:
            raise RecoveryIntegrityError(
                f"Immutable tree pin failed for {path}: {key}={observed[key]!r}"
            )
    return observed


def _manifest_rows(data: bytes, label: str) -> dict[str, dict[str, Any]]:
    try:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t")
    except Exception as exc:
        raise RecoveryIntegrityError(f"{label} is unreadable") from exc
    if reader.fieldnames is None or tuple(reader.fieldnames[:3]) != _MANIFEST_COLUMNS:
        raise RecoveryIntegrityError(f"{label} has an invalid schema")
    rows: dict[str, dict[str, Any]] = {}
    for raw in reader:
        relative = str(raw.get("relative_path", ""))
        if not _safe_relative(relative) or relative in rows:
            raise RecoveryIntegrityError(f"{label} has an unsafe/duplicate path")
        try:
            size = int(str(raw.get("size_bytes", "")))
        except ValueError as exc:
            raise RecoveryIntegrityError(f"{label} has a noninteger size") from exc
        digest = str(raw.get("sha256", ""))
        if size < 0 or not _valid_hash(digest):
            raise RecoveryIntegrityError(f"{label} has an invalid size/hash")
        rows[relative] = {
            "relative_path": relative,
            "size_bytes": size,
            "sha256": digest,
        }
    return rows


def _binary64_bits(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def _ordered_binary64(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits | (1 << 63)


def _ulp_distance(left: float, right: float) -> int:
    return abs(_ordered_binary64(left) - _ordered_binary64(right))


def read_smoke_report_roundtrip(path: Path | str) -> pd.DataFrame:
    """Read Smoke CSV floats with exact C-parser binary64 round-trip recovery."""

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise RecoveryIntegrityError(f"Smoke report CSV is absent/unsafe: {source}")
    return pd.read_csv(source, engine="c", float_precision="round_trip")


def _metric_comparison(
    table: pd.DataFrame,
    details: Mapping[str, Any],
    *,
    value_key: str,
) -> list[dict[str, Any]]:
    required = {"variant", "status", *METRIC_NAMES}
    if not required <= set(table.columns):
        raise RecoveryIntegrityError(
            f"Smoke CSV metric schema is incomplete: {sorted(required - set(table.columns))}"
        )
    variants = table["variant"].astype(str)
    if variants.duplicated().any():
        raise RecoveryIntegrityError("Smoke CSV contains duplicate Variant rows")
    pass_table = table.loc[table["status"].astype(str).str.upper().eq("PASS")]
    if pass_table.empty:
        raise RecoveryIntegrityError("Smoke CSV contains no PASS rows")
    mismatches: list[dict[str, Any]] = []
    for row in pass_table.itertuples(index=False):
        variant = str(row.variant)
        detail = details.get(variant)
        if not isinstance(detail, Mapping):
            raise RecoveryIntegrityError(f"Smoke JSON omits Variant detail: {variant}")
        expected_metrics = detail.get("metrics_development_only")
        if not isinstance(expected_metrics, Mapping):
            raise RecoveryIntegrityError(f"Smoke JSON omits metric mapping: {variant}")
        for metric in METRIC_NAMES:
            if metric not in expected_metrics:
                raise RecoveryIntegrityError(f"Smoke JSON omits {metric}: {variant}")
            observed = float(getattr(row, metric))
            expected = float(expected_metrics[metric])
            if not math.isfinite(observed) or not math.isfinite(expected):
                raise RecoveryIntegrityError(
                    f"Smoke metric is non-finite: {variant}/{metric}"
                )
            if observed != expected:
                mismatches.append({
                    "metric": metric,
                    "variant": variant,
                    value_key: observed,
                    "json_value": expected,
                    "abs_difference": abs(observed - expected),
                    "ulp_distance": _ulp_distance(observed, expected),
                    f"{value_key}_binary64": _binary64_bits(observed),
                    "json_binary64": _binary64_bits(expected),
                })
    return mismatches


def _read_summary(mismatches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "mismatch_count": len(mismatches),
        "max_abs_difference": max(
            (float(row["abs_difference"]) for row in mismatches), default=0.0
        ),
        "max_ulp_distance": max(
            (int(row["ulp_distance"]) for row in mismatches), default=0
        ),
    }


def csv_roundtrip_diagnostic(
    csv_path: Path | str,
    json_path: Path | str,
) -> dict[str, Any]:
    """Compare default and exact-round-trip parsing against Smoke JSON metrics."""

    csv_source = Path(csv_path)
    json_source = Path(json_path)
    for source in (csv_source, json_source):
        if not source.is_file() or source.is_symlink():
            raise RecoveryIntegrityError(f"CSV diagnostic input is absent/unsafe: {source}")
    try:
        metrics = json.loads(json_source.read_text())
    except Exception as exc:
        raise RecoveryIntegrityError("Smoke JSON is malformed") from exc
    details = metrics.get("details") if isinstance(metrics, Mapping) else None
    if not isinstance(details, Mapping) or not details:
        raise RecoveryIntegrityError("Smoke JSON has no Variant details")

    default_table = pd.read_csv(csv_source)
    round_trip_table = read_smoke_report_roundtrip(csv_source)
    if list(default_table.columns) != list(round_trip_table.columns):
        raise RecoveryIntegrityError("Default and round-trip readers disagree on schema")
    if len(default_table) != len(round_trip_table):
        raise RecoveryIntegrityError("Default and round-trip readers disagree on row count")
    default_mismatches = _metric_comparison(
        default_table, details, value_key="default_value",
    )
    round_trip_mismatches = _metric_comparison(
        round_trip_table, details, value_key="round_trip_value",
    )
    default_summary = _read_summary(default_mismatches)
    round_trip_summary = _read_summary(round_trip_mismatches)
    pass_rows = int(
        round_trip_table["status"].astype(str).str.upper().eq("PASS").sum()
    )
    status = "PASS" if not round_trip_mismatches else "FAIL"
    return {
        "classification": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
        "status": status,
        "csv_path": str(csv_source),
        "csv_sha256": sha256_file(csv_source),
        "json_path": str(json_source),
        "json_sha256": sha256_file(json_source),
        "metric_names": list(METRIC_NAMES),
        "pass_row_count": pass_rows,
        "comparison_count": pass_rows * len(METRIC_NAMES),
        "default_read": {
            "expression": "pd.read_csv(path)",
            **default_summary,
        },
        "round_trip_read": {
            "expression": (
                'pd.read_csv(path, engine="c", float_precision="round_trip")'
            ),
            **round_trip_summary,
        },
        "mismatches": default_mismatches,
        "round_trip_mismatches": round_trip_mismatches,
        "default_all_mismatches_one_ulp": bool(
            default_mismatches
            and all(row["ulp_distance"] == 1 for row in default_mismatches)
        ),
        "exact_metric_equality_retained": True,
        "tolerance_or_rounding_added": False,
        "csv_or_json_rewritten": False,
    }


def _validate_preflight(study_root: Path) -> dict[str, Any]:
    results = study_root / "results"
    run = results / ACCEPTED_PREFLIGHT_RUN_ID
    review_zip = results / f"{ACCEPTED_PREFLIGHT_RUN_ID}_review_bundle.zip"
    adjacent = results / f"{ACCEPTED_PREFLIGHT_RUN_ID}_package_verification.json"
    tree = _require_tree(run, ACCEPTED_PREFLIGHT_TREE)
    if (
        not review_zip.is_file()
        or review_zip.is_symlink()
        or sha256_file(review_zip) != ACCEPTED_PREFLIGHT_ZIP_SHA256
    ):
        raise RecoveryIntegrityError("Accepted Preflight Review ZIP pin failed")
    if (
        not adjacent.is_file()
        or adjacent.is_symlink()
        or sha256_file(adjacent)
        != ACCEPTED_PREFLIGHT_PACKAGE_VERIFICATION_SHA256
    ):
        raise RecoveryIntegrityError("Accepted Preflight package verification pin failed")
    try:
        verification = json.loads(adjacent.read_text())
    except Exception as exc:
        raise RecoveryIntegrityError("Accepted package verification is malformed") from exc
    if not (
        verification.get("bundle_path") == str(review_zip)
        and verification.get("bundle_sha256") == ACCEPTED_PREFLIGHT_ZIP_SHA256
        and verification.get("crc_status") == "PASS"
        and verification.get("independent_reopen_member_verification") == "PASS"
        and verification.get("verification_failures") == []
        and int(verification.get("member_count", -1)) == 73
        and verification.get("package_completeness", {}).get(
            "preflight_semantic_completeness", {}
        ).get("status") == "PASS"
    ):
        raise RecoveryIntegrityError("Accepted package verification contract failed")

    prefix = f"{ACCEPTED_PREFLIGHT_RUN_ID}/"
    with zipfile.ZipFile(review_zip) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if (
            archive.testzip() is not None
            or len(infos) != 73
            or len(names) != len(set(names))
            or any(
                info.is_dir()
                or not info.filename.startswith(prefix)
                or not _safe_relative(info.filename[len(prefix):])
                for info in infos
            )
        ):
            raise RecoveryIntegrityError("Accepted Preflight ZIP CRC/path contract failed")
        output_bytes = archive.read(prefix + "OUTPUT_MANIFEST_FINAL.tsv")
        bundle_bytes = archive.read(prefix + "BUNDLE_MANIFEST.tsv")
        status_bytes = archive.read(prefix + "RUN_STATUS.txt")
        identity_bytes = archive.read(prefix + "config/run_identity.lock.json")
        split_bytes = archive.read(prefix + "config/locked_split_identity.json")
        probe_bytes = archive.read(prefix + SEALED_PROBE_RELATIVE)
        inventory_bytes = archive.read(
            prefix + "provenance/source_input_hashes_post.tsv"
        )
        archive_files = {
            info.filename[len(prefix):]: archive.read(info.filename)
            for info in infos
        }

    output_rows = _manifest_rows(output_bytes, "Preflight OUTPUT_MANIFEST_FINAL.tsv")
    bundle_rows = _manifest_rows(bundle_bytes, "Preflight BUNDLE_MANIFEST.tsv")
    if set(bundle_rows) != set(output_rows) | {"OUTPUT_MANIFEST_FINAL.tsv"}:
        raise RecoveryIntegrityError("Accepted Preflight Manifest member sets disagree")
    expected_probe_row = {
        "relative_path": SEALED_PROBE_RELATIVE,
        "size_bytes": SEALED_PROBE_SIZE_BYTES,
        "sha256": SEALED_PROBE_SHA256,
    }
    if (
        output_rows.get(SEALED_PROBE_RELATIVE) != expected_probe_row
        or bundle_rows.get(SEALED_PROBE_RELATIVE) != expected_probe_row
        or len(probe_bytes) != SEALED_PROBE_SIZE_BYTES
        or sha256_bytes(probe_bytes) != SEALED_PROBE_SHA256
    ):
        raise RecoveryIntegrityError("Accepted sealed CUDA probe Manifest pin failed")

    live_files = {
        item.relative_to(run).as_posix(): item
        for item in run.rglob("*")
        if item.is_file() and not item.is_symlink()
    }
    if set(live_files) != set(archive_files):
        raise RecoveryIntegrityError("Accepted live Preflight/ZIP member sets differ")
    for relative, data in archive_files.items():
        if live_files[relative].read_bytes() != data:
            raise RecoveryIntegrityError(
                f"Accepted live Preflight differs from ZIP: {relative}"
            )

    status = _status_fields(status_bytes)
    try:
        identity = json.loads(identity_bytes)
        locked_split = json.loads(split_bytes)
        probe = json.loads(probe_bytes)
    except Exception as exc:
        raise RecoveryIntegrityError("Accepted Preflight JSON evidence is malformed") from exc
    if not (
        status.get("RUN_STATE") == "PREFLIGHT_COMPLETE"
        and status.get("SMOKE_ELIGIBLE") == "YES"
        and status.get("CUDA_SMOKE_ELIGIBLE") == "YES"
        and status.get("FULL_AUTHORIZED") == "NO"
        and status.get("BLOCKERS") == "NONE"
        and status.get("LOCKED_SPLIT_SHA256") == LOCKED_SPLIT_SHA256
        and status.get("BASE_SEED") == str(BASE_SEED)
        and identity.get("run_id") == ACCEPTED_PREFLIGHT_RUN_ID
        and identity.get("protocol") == "AMENDMENT_02"
        and identity.get("state") == "PREFLIGHT_COMPLETE"
        and identity.get("candidate_split_hash") == LOCKED_SPLIT_SHA256
        and identity.get("base_seed") == BASE_SEED
        and identity.get("split_locked") is True
        and identity.get("full_authorized") is False
        and locked_split.get("status") == "PASS"
        and locked_split.get("base_seed") == BASE_SEED
        and locked_split.get("split_assignment_sha256") == LOCKED_SPLIT_SHA256
    ):
        raise RecoveryIntegrityError("Accepted Preflight authorization contract failed")
    raw_probe = probe.get("raw_probe", {})
    if not (
        probe.get("classification") == "VERIFIED"
        and probe.get("status") == "PASS"
        and probe.get("xgboost_version") == XGBOOST_VERSION
        and probe.get("build_info", {}).get("USE_CUDA") is True
        and probe.get("environment_path") == str(CUDA_ENV)
        and probe.get("interpreter") == str(CUDA_PYTHON)
        and probe.get("gpu_identity", {}).get("gpu_name") == VISIBLE_GPU
        and probe.get("requested_device") == "cuda"
        and probe.get("tree_method") == "hist"
        and probe.get("training_status") == "PASS"
        and probe.get("save_reload_predict_status") == "PASS"
        and probe.get("cpu_fallback_detected") is False
        and not probe.get("fallback_warnings")
        and raw_probe.get("status") == "PASS"
        and raw_probe.get("gpu_before", {}).get("driver_version") == "595.95"
        and raw_probe.get("gpu_after", {}).get("driver_version") == "595.95"
    ):
        raise RecoveryIntegrityError("Accepted sealed CUDA probe semantics failed")
    library = Path(str(probe.get("libxgboost_path", "")))
    if not (
        library.is_file()
        and not library.is_symlink()
        and probe.get("libxgboost_sha256") == sha256_file(library)
    ):
        raise RecoveryIntegrityError("Sealed libxgboost pin differs from live bytes")

    try:
        inventory_reader = csv.DictReader(
            io.StringIO(inventory_bytes.decode("utf-8")), delimiter="\t"
        )
        inventory_rows = list(inventory_reader)
    except Exception as exc:
        raise RecoveryIntegrityError("Preflight source inventory is malformed") from exc
    if inventory_reader.fieldnames != ["path", "size_bytes", "sha256"]:
        raise RecoveryIntegrityError("Preflight source inventory schema changed")
    if not inventory_rows or len({row["path"] for row in inventory_rows}) != len(
        inventory_rows
    ):
        raise RecoveryIntegrityError("Preflight source inventory rows are invalid")
    source_records: dict[str, dict[str, Any]] = {}
    for row in inventory_rows:
        source = Path(row["path"])
        try:
            expected_size = int(row["size_bytes"])
        except ValueError as exc:
            raise RecoveryIntegrityError("Source inventory has invalid size") from exc
        expected_hash = row["sha256"]
        if not (
            source.is_file()
            and not source.is_symlink()
            and source.stat().st_size == expected_size
            and _valid_hash(expected_hash)
            and sha256_file(source) == expected_hash
        ):
            raise RecoveryIntegrityError(f"Accepted source input drifted: {source}")
        source_records[str(source)] = {
            "size_bytes": expected_size,
            "sha256": expected_hash,
        }
    return {
        "status": "PASS",
        "run_id": ACCEPTED_PREFLIGHT_RUN_ID,
        "run_path": str(run),
        "tree": tree,
        "review_zip_path": str(review_zip),
        "review_zip_sha256": ACCEPTED_PREFLIGHT_ZIP_SHA256,
        "package_verification_path": str(adjacent),
        "package_verification_sha256": (
            ACCEPTED_PREFLIGHT_PACKAGE_VERIFICATION_SHA256
        ),
        "locked_split_sha256": LOCKED_SPLIT_SHA256,
        "base_seed": BASE_SEED,
        "sealed_probe": {
            "relative_path": SEALED_PROBE_RELATIVE,
            "size_bytes": SEALED_PROBE_SIZE_BYTES,
            "sha256": SEALED_PROBE_SHA256,
            "member_size_bytes": SEALED_PROBE_SIZE_BYTES,
            "member_sha256": SEALED_PROBE_SHA256,
            "output_manifest_row": output_rows[SEALED_PROBE_RELATIVE],
            "bundle_manifest_row": bundle_rows[SEALED_PROBE_RELATIVE],
        },
        "source_inputs": source_records,
        "reviewed_code_tree": dict(sorted(identity.get("code_tree", {}).items())),
    }


def _validate_a03_executed_baseline(
    study_root: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    run = study_root / "results" / FAILED_AMENDMENT03_RUN_ID
    tsv = run / "provenance/executed_code_hashes.tsv"
    if sha256_file(tsv) != FAILED_AMENDMENT03_KEY_HASHES[
        "provenance/executed_code_hashes.tsv"
    ]:
        raise RecoveryIntegrityError("Amendment 03 executed-code TSV pin failed")
    try:
        with tsv.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rows = list(reader)
    except Exception as exc:
        raise RecoveryIntegrityError("Amendment 03 executed-code TSV is malformed") from exc
    required = {
        "relative_path",
        "executed_sha256",
        "snapshot_relative_path",
        "snapshot_sha256",
    }
    if reader.fieldnames is None or not required <= set(reader.fieldnames):
        raise RecoveryIntegrityError("Amendment 03 executed-code TSV schema changed")
    if not rows or len({row["relative_path"] for row in rows}) != len(rows):
        raise RecoveryIntegrityError("Amendment 03 executed-code TSV row set is invalid")
    baseline: dict[str, str] = {}
    snapshot_hashes: dict[str, str] = {}
    for row in rows:
        relative = row["relative_path"]
        snapshot_relative = row["snapshot_relative_path"]
        executed = row["executed_sha256"]
        snapshot = run / snapshot_relative
        if not (
            _safe_relative(relative)
            and _safe_relative(snapshot_relative)
            and _valid_hash(executed)
            and row["snapshot_sha256"] == executed
            and snapshot.is_file()
            and not snapshot.is_symlink()
            and sha256_file(snapshot) == executed
        ):
            raise RecoveryIntegrityError(
                f"Amendment 03 executed snapshot pin failed: {relative}"
            )
        baseline[relative] = executed
        snapshot_hashes[snapshot_relative] = executed
    return baseline, {
        "path": str(tsv),
        "sha256": sha256_file(tsv),
        "row_count": len(rows),
        "executed_hashes": dict(sorted(baseline.items())),
        "snapshot_hashes": dict(sorted(snapshot_hashes.items())),
    }


def _validate_failed_amendment03(study_root: Path) -> dict[str, Any]:
    results = study_root / "results"
    run = results / FAILED_AMENDMENT03_RUN_ID
    tree = _require_tree(run, FAILED_AMENDMENT03_TREE)
    observed_hashes: dict[str, str] = {}
    for relative, expected in FAILED_AMENDMENT03_KEY_HASHES.items():
        path = run / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != expected
        ):
            raise RecoveryIntegrityError(
                f"Failed Amendment 03 key hash drifted: {relative}"
            )
        observed_hashes[relative] = expected

    status = _status_fields((run / "RUN_STATUS.txt").read_bytes())
    try:
        metrics = json.loads((run / "metrics/SMOKE_REPORT.json").read_text())
        ledger = json.loads(
            (run / "provenance/execution_ledger.json").read_text()
        )
        parity = json.loads(
            (run / "provenance/frozen_resampling_parity.json").read_text()
        )
        variants = json.loads(
            (run / "config/variant_feature_sets.json").read_text()
        )
    except Exception as exc:
        raise RecoveryIntegrityError("Failed Amendment 03 evidence is malformed") from exc
    completed = set(ledger.get("smoke_variants_completed_ids", []))
    details = metrics.get("details", {})
    if not (
        status.get("RUN_STATE") == "SMOKE_INCOMPLETE"
        and status.get("FULL_TRAINING_EXECUTED") == "NO"
        and status.get("SCIENTIFIC_RESULTS_CLAIMED") == "NO"
        and status.get("AMENDMENT03_REPLACEMENT_RUN_AUTHORIZATION")
        == "1_OF_1_CONSUMED"
        and metrics.get("state") == "SMOKE_INCOMPLETE"
        and metrics.get("classification") == "NON_SCIENTIFIC_DIAGNOSTIC_ONLY"
        and metrics.get("final_seal_failure")
        == "IntegrityError: Smoke semantic completeness failed checks: "
        "['smoke_table_matches_variant_details']"
        and metrics.get("official_variants_completed") == 9
        and metrics.get("exploratory_variants_completed") == 1
        and metrics.get("external_audit_scored") is False
        and metrics.get("true_no_pca_status")
        == "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE"
        and ledger.get("status") == "PASS"
        and ledger.get("official_smoke_variants_completed") == 9
        and ledger.get("exploratory_smoke_variants_completed") == 1
        and ledger.get("full_scientific_run_executed") is False
        and ledger.get("full_variants_completed") == 0
        and completed == EXPECTED_VARIANT_IDS
        and set(details) == EXPECTED_VARIANT_IDS
        and parity.get("status") == "PASS"
        and variants.get("no_pca", {}).get("status")
        == "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE"
        and variants.get("no_deep_pca_features", {}).get("status") == "RUNNABLE"
    ):
        raise RecoveryIntegrityError("Failed Amendment 03 state/Variant contract failed")
    for variant in sorted(EXPECTED_VARIANT_IDS):
        detail = details[variant]
        recovery = detail.get("amendment03_model_recovery", {})
        four_way = detail.get("four_way_reload_parity", {})
        raw = detail.get("raw_booster_parity", {})
        compatibility = detail.get("compatibility_classifier_parity", {})
        active = detail.get("active_reloaded_cuda_probe", {})
        bit_exact = four_way.get("bit_exact_against_fitted_classifier", {})
        maximum = four_way.get("maximum_absolute_difference", {})
        vectors = four_way.get("vector_sha256", {})
        if not (
            detail.get("save_reload_max_probability_difference") == 0.0
            and recovery.get("status") == "PASS"
            and four_way.get("status") == "PASS"
            and four_way.get("cpu_fallback_detected") is False
            and four_way.get("model_bytes_unchanged") is True
            and set(bit_exact) == {
                "fitted_classifier",
                "fitted_booster",
                "reloaded_booster",
                "compatibility_classifier",
            }
            and all(value is True for value in bit_exact.values())
            and set(maximum) == set(bit_exact)
            and all(float(value) == 0.0 for value in maximum.values())
            and set(vectors) == set(bit_exact)
            and len(set(vectors.values())) == 1
            and all(_valid_hash(value) for value in vectors.values())
            and raw.get("status") == "PASS"
            and raw.get("fitted_booster_bit_exact") is True
            and raw.get("reloaded_booster_bit_exact") is True
            and raw.get("fitted_booster_max_probability_difference") == 0.0
            and raw.get("reloaded_booster_max_probability_difference") == 0.0
            and raw.get("model_bytes_unchanged") is True
            and compatibility.get("status") == "PASS"
            and compatibility.get("bit_exact") is True
            and compatibility.get("max_probability_difference") == 0.0
            and active.get("status") == "PASS"
            and str(active.get("device", "")).startswith("cuda")
            and not active.get("warnings")
        ):
            raise RecoveryIntegrityError(
                f"Failed Amendment 03 CUDA/model parity failed: {variant}"
            )

    for relative in SEALED_PROBE_DESTINATIONS:
        probe = run / relative
        if not (
            probe.is_file()
            and not probe.is_symlink()
            and probe.stat().st_size == SEALED_PROBE_SIZE_BYTES
            and sha256_file(probe) == SEALED_PROBE_SHA256
        ):
            raise RecoveryIntegrityError(f"Failed A03 sealed probe drifted: {relative}")
    if (run / SEALED_PROBE_DESTINATIONS[0]).read_bytes() != (
        run / SEALED_PROBE_DESTINATIONS[1]
    ).read_bytes():
        raise RecoveryIntegrityError("Failed A03 root/nested probes differ")

    absent = (
        run / "OUTPUT_MANIFEST_FINAL.tsv",
        run / "BUNDLE_MANIFEST.tsv",
        run / "provenance/smoke_model_bundle_verification.json",
        results / f"{FAILED_AMENDMENT03_RUN_ID}_review_bundle.zip",
        results / f"{FAILED_AMENDMENT03_RUN_ID}_package_verification.json",
        results / f"{FAILED_AMENDMENT03_RUN_ID}_NON_SCIENTIFIC_smoke_models.zip",
        results
        / (
            f"{FAILED_AMENDMENT03_RUN_ID}_NON_SCIENTIFIC_smoke_models_"
            "package_verification.json"
        ),
    )
    if any(path.exists() or path.is_symlink() for path in absent):
        raise RecoveryIntegrityError("Failed A03 has manufactured final publications")

    forensic = results / f"{FAILED_AMENDMENT03_RUN_ID}.zip"
    if (
        not forensic.is_file()
        or forensic.is_symlink()
        or sha256_file(forensic) != FAILED_AMENDMENT03_FORENSIC_ZIP_SHA256
    ):
        raise RecoveryIntegrityError("Failed A03 forensic archive pin failed")
    prefix = f"{FAILED_AMENDMENT03_RUN_ID}/"
    with zipfile.ZipFile(forensic) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        regular = [info for info in infos if not info.is_dir()]
        if not (
            archive.testzip() is None
            and len(infos) == FAILED_AMENDMENT03_FORENSIC_MEMBERS
            and len(regular) == FAILED_AMENDMENT03_FORENSIC_REGULAR_FILES
            and sum(info.file_size for info in regular)
            == FAILED_AMENDMENT03_FORENSIC_REGULAR_BYTES
            and len(names) == len(set(names))
            and all(
                info.filename.startswith(prefix)
                and (
                    info.is_dir()
                    or _safe_relative(info.filename[len(prefix):])
                )
                for info in infos
            )
        ):
            raise RecoveryIntegrityError("Failed A03 forensic ZIP contract failed")
        archived_relatives = {
            info.filename[len(prefix):] for info in regular
        }
        live_relatives = {
            item.relative_to(run).as_posix()
            for item in run.rglob("*")
            if item.is_file() and not item.is_symlink()
        }
        if archived_relatives != live_relatives:
            raise RecoveryIntegrityError("A03 forensic/live member sets differ")
        for info in regular:
            relative = info.filename[len(prefix):]
            if archive.read(info.filename) != (run / relative).read_bytes():
                raise RecoveryIntegrityError(
                    f"A03 forensic/live bytes differ: {relative}"
                )

    baseline, baseline_evidence = _validate_a03_executed_baseline(study_root)
    return {
        "status": "PASS",
        "run_id": FAILED_AMENDMENT03_RUN_ID,
        "run_path": str(run),
        "tree": tree,
        "tree_sha256": tree["tree_sha256"],
        "file_count": tree["file_count"],
        "total_size_bytes": tree["total_size_bytes"],
        "key_hashes": observed_hashes,
        "forensic_zip": {
            "path": str(forensic),
            "sha256": FAILED_AMENDMENT03_FORENSIC_ZIP_SHA256,
            "member_count": FAILED_AMENDMENT03_FORENSIC_MEMBERS,
            "regular_file_count": FAILED_AMENDMENT03_FORENSIC_REGULAR_FILES,
            "regular_file_bytes": FAILED_AMENDMENT03_FORENSIC_REGULAR_BYTES,
            "crc_status": "PASS",
            "live_byte_equality": "PASS",
        },
        "executed_code_baseline": baseline_evidence,
        "completed_variant_ids": sorted(completed),
        "model_reload_parity": "PASS_10_OF_10_BIT_EXACT_MAX_DIFF_0",
        "cpu_fallback_detected": False,
        "full_training_executed": False,
        "_baseline_executed_hashes": baseline,
    }


def _validate_failed_amendment02(study_root: Path) -> dict[str, Any]:
    run = study_root / "results" / FAILED_AMENDMENT02_RUN_ID
    tree = _require_tree(run, FAILED_AMENDMENT02_TREE)
    evidence: dict[str, dict[str, Any]] = {}
    for relative, expected in FAILED_AMENDMENT02_EVIDENCE_HASHES.items():
        path = study_root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != expected
        ):
            raise RecoveryIntegrityError(
                f"Failed Amendment 02 evidence pin failed: {relative}"
            )
        evidence[relative] = {"path": str(path), "sha256": expected}
    runtime = (
        study_root / ".runtime/smoke_models" / FAILED_AMENDMENT02_RUN_ID
    )
    runtime_tree = _require_tree(runtime, FAILED_AMENDMENT02_RUNTIME_TREE)
    orphan = runtime / "full_new_reference.ubj"
    if sha256_file(orphan) != FAILED_AMENDMENT02_ORPHAN_MODEL_SHA256:
        raise RecoveryIntegrityError("Failed Amendment 02 orphan model pin failed")
    runtime_probe = (
        study_root / ".runtime/amendment02_cuda_wheel/cuda_active_fit_probe.json"
    )
    if (
        not runtime_probe.is_file()
        or runtime_probe.is_symlink()
        or sha256_file(runtime_probe) != DIAGNOSTIC_RUNTIME_PROBE_SHA256
    ):
        raise RecoveryIntegrityError("Diagnostic-only runtime probe pin failed")
    return {
        "status": "PASS",
        "run_id": FAILED_AMENDMENT02_RUN_ID,
        "run_path": str(run),
        "tree": tree,
        "failure_evidence": evidence,
        "runtime_models": {
            **runtime_tree,
            "orphan_model_sha256": FAILED_AMENDMENT02_ORPHAN_MODEL_SHA256,
            "diagnostic_only": True,
            "reused": False,
        },
        "runtime_probe": {
            "path": str(runtime_probe),
            "sha256": DIAGNOSTIC_RUNTIME_PROBE_SHA256,
            "diagnostic_only": True,
            "used_as_source": False,
        },
    }


def _prior_package_inventory(study_root: Path) -> dict[str, dict[str, Any]]:
    """Bind every publication that predates the distinct A04 Run namespace."""

    results = Path(study_root) / "results"
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(results.iterdir(), key=lambda item: item.name.encode("utf-8")):
        name = path.name
        if "_amendment04_cuda_smoke" in name:
            continue
        is_prior_publication = bool(
            name.endswith("_review_bundle.zip")
            or name.endswith("_package_verification.json")
            or name.endswith("_models.zip")
            or name.endswith("_models_package_verification.json")
            or name == f"{FAILED_AMENDMENT03_RUN_ID}.zip"
        )
        if not is_prior_publication:
            continue
        if not path.is_file() or path.is_symlink():
            raise RecoveryIntegrityError(f"Prior package is absent/unsafe: {path}")
        inventory[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if not inventory:
        raise RecoveryIntegrityError("Prior package inventory is unexpectedly empty")
    return inventory


def validate_immutable_gate(
    study_root: Path = STUDY_ROOT,
) -> dict[str, Any]:
    """Revalidate every pinned Preflight and failed-Run byte before A04 work."""

    root = Path(study_root)
    if not root.is_dir() or root.is_symlink():
        raise RecoveryIntegrityError("Study root is absent or unsafe")
    if (
        not PROMPT_PATH.is_file()
        or PROMPT_PATH.is_symlink()
        or sha256_file(PROMPT_PATH) != PROMPT_SHA256
    ):
        raise RecoveryIntegrityError("Amendment 04 prompt path/hash pin failed")
    preflight = _validate_preflight(root)
    failed_a02 = _validate_failed_amendment02(root)
    failed_a03 = _validate_failed_amendment03(root)
    a03_runtime = (
        root / ".runtime/smoke_models" / FAILED_AMENDMENT03_RUN_ID
    )
    a03_runtime_tree = _require_tree(
        a03_runtime, FAILED_AMENDMENT03_RUNTIME_TREE,
    )
    failed_a03["runtime_models"] = {
        **a03_runtime_tree,
        "diagnostic_only": True,
        "reused": False,
    }
    return {
        "classification": "VERIFIED",
        "status": "PASS",
        "authorization": AUTHORIZATION,
        "prompt": {
            "path": str(PROMPT_PATH),
            "sha256": PROMPT_SHA256,
            "immutable": True,
        },
        "accepted_preflight": preflight,
        "failed_amendment02_smoke": failed_a02,
        "failed_amendment03_smoke": failed_a03,
        "prior_package_artifacts": _prior_package_inventory(root),
        "locked_split_sha256": LOCKED_SPLIT_SHA256,
        "base_seed": BASE_SEED,
        "cuda_contract": {
            "environment": str(CUDA_ENV),
            "python": str(CUDA_PYTHON),
            "xgboost_version": XGBOOST_VERSION,
            "scikit_learn_version": SCIKIT_LEARN_VERSION,
            "visible_gpu": VISIBLE_GPU,
            "sealed_probe_sha256": SEALED_PROBE_SHA256,
            "sealed_probe_size_bytes": SEALED_PROBE_SIZE_BYTES,
        },
        "prior_artifacts_preserved": True,
        "prior_models_or_results_reused": False,
    }


def _prior_cuda_smoke_names() -> set[str]:
    return {
        FAILED_AMENDMENT02_RUN_ID,
        FAILED_AMENDMENT03_RUN_ID,
        f"{FAILED_AMENDMENT03_RUN_ID}.zip",
    }


def _valid_amendment04_run_id(value: str) -> bool:
    return bool(
        value
        and Path(value).name == value
        and value.startswith("run_")
        and value.endswith("_amendment04_cuda_smoke")
        and value not in {FAILED_AMENDMENT02_RUN_ID, FAILED_AMENDMENT03_RUN_ID}
    )


def assert_run_guard(
    results_root: Path = RESULTS_ROOT,
    new_run_id: str | None = None,
) -> dict[str, Any]:
    """Require the exact prior evidence plus zero or one named A04 Run."""

    root = Path(results_root)
    if not root.is_dir() or root.is_symlink():
        raise RecoveryIntegrityError("Results root is absent or unsafe")
    entries = {path.name: path for path in root.iterdir()}
    required_prior = _prior_cuda_smoke_names()
    observed_cuda = sorted(
        name for name in entries if "cuda_smoke" in name.lower()
    )
    missing_prior = sorted(required_prior - set(entries))
    if missing_prior:
        raise RecoveryIntegrityError(
            f"Amendment 04 guard is missing prior evidence: {missing_prior}"
        )
    if not (
        entries[FAILED_AMENDMENT02_RUN_ID].is_dir()
        and not entries[FAILED_AMENDMENT02_RUN_ID].is_symlink()
        and entries[FAILED_AMENDMENT03_RUN_ID].is_dir()
        and not entries[FAILED_AMENDMENT03_RUN_ID].is_symlink()
        and entries[f"{FAILED_AMENDMENT03_RUN_ID}.zip"].is_file()
        and not entries[f"{FAILED_AMENDMENT03_RUN_ID}.zip"].is_symlink()
    ):
        raise RecoveryIntegrityError("Amendment 04 prior evidence types are unsafe")

    observed_a04_dirs = sorted(
        name
        for name, path in entries.items()
        if name.endswith("_amendment04_cuda_smoke")
        and (path.is_dir() or path.is_symlink())
    )
    allowed = set(required_prior)
    if new_run_id is None:
        if observed_a04_dirs:
            raise RecoveryIntegrityError(
                "Amendment 04 authorization was already consumed: "
                f"{observed_a04_dirs}"
            )
        authorization = "0_OF_1_BEFORE_RUN_CREATION"
        replacement = None
    else:
        if not _valid_amendment04_run_id(new_run_id):
            raise RecoveryIntegrityError("Invalid Amendment 04 Run ID")
        replacement = entries.get(new_run_id)
        if (
            replacement is None
            or not replacement.is_dir()
            or replacement.is_symlink()
            or observed_a04_dirs != [new_run_id]
        ):
            raise RecoveryIntegrityError(
                "Named Amendment 04 Run is absent, unsafe, or not unique"
            )
        allowed.update({
            new_run_id,
            f"{new_run_id}_review_bundle.zip",
            f"{new_run_id}_review_bundle_staging_verified.zip",
            f"{new_run_id}_package_verification.json",
            f"{new_run_id}_NON_SCIENTIFIC_smoke_models.zip",
            (
                f"{new_run_id}_NON_SCIENTIFIC_smoke_models_"
                "package_verification.json"
            ),
        })
        authorization = "1_OF_1_CONSUMED"
    unexpected = sorted(set(observed_cuda) - allowed)
    if unexpected:
        raise RecoveryIntegrityError(
            f"Unexpected CUDA Smoke outputs under A04 guard: {unexpected}"
        )
    return {
        "status": "PASS",
        "authorization": authorization,
        "replacement_authorization": authorization,
        "amendment04_authorization": authorization,
        "amendment04_runs_created_before": 0 if new_run_id is None else 1,
        "required_prior_outputs": sorted(required_prior),
        "observed_cuda_smoke_outputs": observed_cuda,
        "new_run_id": new_run_id,
        "new_run_path": str(replacement) if replacement is not None else None,
    }


def create_authorized_run_directory(
    path: Path,
    results_root: Path = RESULTS_ROOT,
    *,
    created_run_holder: MutableMapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Atomically consume the single A04 authorization at directory creation."""

    target = Path(path)
    root = Path(results_root).resolve()
    if target.parent.resolve() != root or not _valid_amendment04_run_id(target.name):
        raise RecoveryIntegrityError(
            "Amendment 04 Run must be a valid direct child of results"
        )
    try:
        lock_fd = os.open(
            root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise RecoveryIntegrityError(
            "Could not lock the Amendment 04 results directory"
        ) from exc
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        assert_run_guard(root)
        try:
            target.mkdir(mode=0o755, exist_ok=False)
        except FileExistsError as exc:
            raise RecoveryIntegrityError(
                "Amendment 04 Run authorization was already consumed"
            ) from exc
        if created_run_holder is not None:
            created_run_holder["run_dir"] = target.resolve()
        # A failed post-create check deliberately leaves the directory in place:
        # authorization is consumed and the new path is forensic evidence.
        return assert_run_guard(root, new_run_id=target.name)
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _live_code_tree(study_root: Path) -> dict[str, str]:
    root = Path(study_root)
    paths = [
        *sorted((root / "code").glob("*.py")),
        *sorted((root / "code").glob("*.sh")),
        *sorted((root / "tests").glob("*.py")),
    ]
    live: dict[str, str] = {}
    for path in paths:
        if path.is_symlink():
            raise RecoveryIntegrityError(f"Live code/test path is a symlink: {path}")
        if path.is_file():
            live[path.relative_to(root).as_posix()] = sha256_file(path)
    return live


def _a03_baseline_hashes(study_root: Path) -> dict[str, str]:
    baseline, _ = _validate_a03_executed_baseline(Path(study_root))
    return baseline


def _exact_drift(
    baseline: Mapping[str, str],
    live: Mapping[str, str],
    expected_changed: set[str] | frozenset[str],
    *,
    label: str,
) -> dict[str, str]:
    if not baseline or not all(
        _safe_relative(path) and _valid_hash(digest)
        for path, digest in baseline.items()
    ):
        raise RecoveryIntegrityError(f"{label} baseline is invalid")
    if not live or not all(
        _safe_relative(path) and _valid_hash(digest)
        for path, digest in live.items()
    ):
        raise RecoveryIntegrityError(f"{label} live tree is invalid")
    changed = {
        relative
        for relative in set(baseline) | set(live)
        if baseline.get(relative) != live.get(relative)
    }
    if changed != set(expected_changed):
        raise RecoveryIntegrityError(
            f"{label} drift is not the exact allow-list: "
            f"changed={sorted(changed)}, expected={sorted(expected_changed)}"
        )
    if any(relative not in live for relative in expected_changed):
        raise RecoveryIntegrityError(f"{label} removes an allow-listed path")
    return {relative: live[relative] for relative in sorted(expected_changed)}


def code_drift_allowlist(
    study_root: Path = STUDY_ROOT,
    *,
    baseline_executed_hashes: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return exact A04 after-hashes relative to the executed A03 source tree."""

    root = Path(study_root)
    baseline = dict(
        baseline_executed_hashes
        if baseline_executed_hashes is not None
        else _a03_baseline_hashes(root)
    )
    return _exact_drift(
        baseline,
        _live_code_tree(root),
        AMENDMENT04_ALLOWED_CODE_DRIFT_PATHS,
        label="Amendment 04",
    )


def reference_code_drift_allowlist(
    reviewed_code_tree: Mapping[str, str],
    *,
    study_root: Path = STUDY_ROOT,
) -> dict[str, str]:
    """Return composite A03+A04 drift relative to the accepted A02 Preflight."""

    return _exact_drift(
        dict(reviewed_code_tree),
        _live_code_tree(Path(study_root)),
        AMENDMENT04_REFERENCE_CODE_DRIFT_PATHS,
        label="Accepted Preflight to Amendment 04",
    )


def build_executed_provenance(
    *,
    run_dir: Path,
    allowed_after_hashes: Mapping[str, str],
    study_root: Path = STUDY_ROOT,
    baseline_executed_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Snapshot exact live code/tests under the four-path A04 drift contract."""

    root = Path(study_root)
    run = Path(run_dir)
    baseline = dict(
        baseline_executed_hashes
        if baseline_executed_hashes is not None
        else _a03_baseline_hashes(root)
    )
    live = _live_code_tree(root)
    expected_after = _exact_drift(
        baseline,
        live,
        AMENDMENT04_ALLOWED_CODE_DRIFT_PATHS,
        label="Amendment 04 executed provenance",
    )
    if dict(allowed_after_hashes) != expected_after:
        raise RecoveryIntegrityError(
            "Executed provenance after-hashes differ from the exact A04 allow-list"
        )

    rows: list[dict[str, str]] = []
    before_hashes: dict[str, str] = {}
    snapshot_hashes: dict[str, str] = {}
    for relative in sorted(live):
        source = root / relative
        category, name = relative.split("/", 1)
        snapshot_relative = (
            "provenance/executed_code_snapshot/"
            if category == "code"
            else "provenance/executed_tests_snapshot/"
        ) + name
        destination = run / snapshot_relative
        data = source.read_bytes()
        digest = sha256_bytes(data)
        if digest != live[relative]:
            raise RecoveryIntegrityError(f"Live code changed during snapshot: {relative}")
        _write_new_bytes(destination, data)
        if sha256_file(destination) != digest:
            raise RecoveryIntegrityError(
                f"Executed snapshot changed during copy: {relative}"
            )
        before = baseline.get(relative, "NOT_PRESENT_IN_AMENDMENT03_EXECUTION")
        is_changed = relative in AMENDMENT04_ALLOWED_CODE_DRIFT_PATHS
        rows.append({
            "relative_path": relative,
            "reviewed_sha256": before,
            "executed_sha256": digest,
            "snapshot_relative_path": snapshot_relative,
            "snapshot_sha256": digest,
            "drift_status": "ALLOWLISTED_CHANGE" if is_changed else "UNCHANGED",
            "allowlisted": "YES" if is_changed else "NO",
        })
        before_hashes[relative] = before
        snapshot_hashes[snapshot_relative] = digest

    columns = (
        "relative_path",
        "reviewed_sha256",
        "executed_sha256",
        "snapshot_relative_path",
        "snapshot_sha256",
        "drift_status",
        "allowlisted",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=columns, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    tsv_bytes = stream.getvalue().encode("utf-8")
    tsv_path = run / "provenance/executed_code_hashes.tsv"
    _write_new_bytes(tsv_path, tsv_bytes)
    return {
        "status": "PASS",
        "baseline": "FAILED_AMENDMENT03_EXECUTED_CODE_HASHES_TSV",
        "baseline_tsv_sha256": FAILED_AMENDMENT03_KEY_HASHES[
            "provenance/executed_code_hashes.tsv"
        ],
        "executed_file_count": len(rows),
        "changed_paths": sorted(AMENDMENT04_ALLOWED_CODE_DRIFT_PATHS),
        "before_hashes": {
            path: before_hashes[path]
            for path in sorted(AMENDMENT04_ALLOWED_CODE_DRIFT_PATHS)
        },
        "after_hashes": dict(sorted(expected_after.items())),
        "allowed_after_hashes": dict(sorted(expected_after.items())),
        "snapshot_hashes": dict(sorted(snapshot_hashes.items())),
        "executed_code_hashes_tsv_sha256": sha256_bytes(tsv_bytes),
    }


def _validate_executed_provenance(
    *,
    run_dir: Path,
    allowed_after_hashes: Mapping[str, str],
    study_root: Path,
    baseline_executed_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(study_root)
    run = Path(run_dir)
    baseline = dict(
        baseline_executed_hashes
        if baseline_executed_hashes is not None
        else _a03_baseline_hashes(root)
    )
    live = _live_code_tree(root)
    expected_after = _exact_drift(
        baseline,
        live,
        AMENDMENT04_ALLOWED_CODE_DRIFT_PATHS,
        label="Amendment 04 executed revalidation",
    )
    if dict(allowed_after_hashes) != expected_after:
        raise RecoveryIntegrityError("A04 allowed after-hashes changed")
    tsv = run / "provenance/executed_code_hashes.tsv"
    if not tsv.is_file() or tsv.is_symlink():
        raise RecoveryIntegrityError("A04 executed-code TSV is absent/unsafe")
    try:
        with tsv.open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except Exception as exc:
        raise RecoveryIntegrityError("A04 executed-code TSV is malformed") from exc
    if len(rows) != len(live) or len({row.get("relative_path") for row in rows}) != len(rows):
        raise RecoveryIntegrityError("A04 executed-code TSV row set changed")
    changed: set[str] = set()
    snapshot_hashes: dict[str, str] = {}
    for row in rows:
        relative = str(row.get("relative_path", ""))
        snapshot_relative = str(row.get("snapshot_relative_path", ""))
        executed = live.get(relative)
        before = baseline.get(relative, "NOT_PRESENT_IN_AMENDMENT03_EXECUTION")
        snapshot = run / snapshot_relative
        is_changed = before != executed
        if is_changed:
            changed.add(relative)
        if not (
            executed is not None
            and _safe_relative(snapshot_relative)
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
            raise RecoveryIntegrityError(
                f"A04 executed provenance mismatch: {relative}"
            )
        snapshot_hashes[snapshot_relative] = executed
    if changed != set(AMENDMENT04_ALLOWED_CODE_DRIFT_PATHS):
        raise RecoveryIntegrityError("A04 executed provenance changed-path set differs")
    return {
        "status": "PASS",
        "executed_file_count": len(rows),
        "changed_paths": sorted(changed),
        "before_hashes": {
            path: baseline.get(path, "NOT_PRESENT_IN_AMENDMENT03_EXECUTION")
            for path in sorted(changed)
        },
        "after_hashes": dict(sorted(expected_after.items())),
        "snapshot_hashes": dict(sorted(snapshot_hashes.items())),
        "executed_code_hashes_tsv_sha256": sha256_file(tsv),
    }


def _parse_command_log(
    data: bytes,
    *,
    source_relative: str,
    study_root: Path,
) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecoveryIntegrityError(f"Command log is not UTF-8: {source_relative}") from exc
    markers = (
        "COMMAND=",
        "CWD=",
        "EXIT_CODE=",
        "WALL_SECONDS=",
        "STDOUT_BEGIN\n",
        "\nSTDOUT_END\n",
        "STDERR_BEGIN\n",
        "\nSTDERR_END",
    )
    if any(text.count(marker) != 1 for marker in markers):
        raise RecoveryIntegrityError(
            f"Command log envelope is not exact: {source_relative}"
        )
    stdout_start = text.index("STDOUT_BEGIN\n") + len("STDOUT_BEGIN\n")
    stdout_end = text.index("\nSTDOUT_END\n", stdout_start)
    stderr_start = text.index("STDERR_BEGIN\n", stdout_end) + len("STDERR_BEGIN\n")
    stderr_end = text.index("\nSTDERR_END", stderr_start)
    header = text[: text.index("STDOUT_BEGIN\n")]
    fields = _status_fields(header.encode("utf-8"))
    if set(fields) != {"COMMAND", "CWD", "EXIT_CODE", "WALL_SECONDS"}:
        raise RecoveryIntegrityError(
            f"Command log header fields changed: {source_relative}"
        )
    try:
        wall = float(fields["WALL_SECONDS"])
    except ValueError as exc:
        raise RecoveryIntegrityError(f"Command log wall time is invalid: {source_relative}") from exc
    if (
        fields["CWD"] != str(study_root)
        or fields["EXIT_CODE"] != "0"
        or not math.isfinite(wall)
        or wall < 0.0
    ):
        raise RecoveryIntegrityError(
            f"Command log did not complete in the pinned CWD: {source_relative}"
        )
    command = fields["COMMAND"]
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise RecoveryIntegrityError(f"Command log command is malformed: {source_relative}") from exc
    if tokens.count(str(CUDA_PYTHON)) != 1:
        raise RecoveryIntegrityError(
            f"Command log does not invoke exact CUDA Python: {source_relative}"
        )
    stdout = text[stdout_start:stdout_end]
    stderr = text[stderr_start:stderr_end]
    return {
        "command": command,
        "tokens": tokens,
        "cwd": fields["CWD"],
        "exit_code": 0,
        "wall_seconds": wall,
        "stdout": stdout,
        "stderr": stderr,
    }


def _require_log_command(
    record: Mapping[str, Any],
    *,
    label: str,
) -> None:
    command = str(record["command"])
    tokens = list(record["tokens"])
    stdout = str(record["stdout"])
    lowered_output = (stdout + "\n" + str(record["stderr"])).lower()
    if label == "syntax":
        required = {
            "code/amendment04_recovery.py",
            "code/amendment_core.py",
            "code/run_study.py",
            "tests/test_amendment04.py",
        }
        if not (
            required <= set(tokens)
            and ("py_compile" in command or "compile(" in command)
        ):
            raise RecoveryIntegrityError("A04 syntax command is not exact")
    elif label == "targeted":
        required = {
            "tests/test_amendment04.py",
            "tests/test_amendment03.py",
            "tests/test_amendment03_recovery.py",
            "tests/test_amendment02_resampling.py",
        }
        if not (
            required <= set(tokens)
            and "pytest" in tokens
            and "-p" in tokens
            and "no:cacheprovider" in tokens
            and "passed" in lowered_output
        ):
            raise RecoveryIntegrityError("A04 targeted-test command is not exact")
    elif label == "full":
        if not (
            "pytest" in tokens
            and "tests" in tokens
            and "-p" in tokens
            and "no:cacheprovider" in tokens
            and "passed" in lowered_output
        ):
            raise RecoveryIntegrityError("A04 full-test command is not exact")
    elif label == "cuda":
        if not (
            "AMENDMENT04_ACTIVE_CUDA_PROBE=1" in tokens
            and "pytest" in tokens
            and "test_amendment04_active_cuda_production_helper" in command
            and "passed" in lowered_output
            and "cpu_fallback_detected\": true" not in lowered_output
            and "cpu fallback" not in lowered_output
        ):
            raise RecoveryIntegrityError("A04 active CUDA command is not exact")
    elif label == "package":
        if not (
            "pytest" in tokens
            and "test_amendment04" in command
            and "package" in command.lower()
            and "rehearsal" in command.lower()
            and "passed" in lowered_output
            and "/results/" not in command
        ):
            raise RecoveryIntegrityError("A04 package-rehearsal command is not exact")
    else:
        raise RecoveryIntegrityError(f"Unknown A04 command-log label: {label}")


def _binding_value(payload: Mapping[str, Any], *paths: Sequence[str]) -> Any:
    for path in paths:
        value: Any = payload
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            return value
    return None


def _validate_live_cuda_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    executable = _binding_value(
        payload, ("sys_executable",), ("interpreter",), ("python",)
    )
    xgb_version = _binding_value(
        payload, ("xgboost_version",), ("versions", "xgboost")
    )
    sklearn_version = _binding_value(
        payload,
        ("scikit_learn_version",),
        ("sklearn_version",),
        ("versions", "scikit_learn"),
        ("versions", "sklearn"),
    )
    use_cuda = _binding_value(
        payload,
        ("xgboost_use_cuda",),
        ("build_info", "USE_CUDA"),
        ("xgboost_build_info", "USE_CUDA"),
    )
    gpu = _binding_value(
        payload,
        ("visible_gpu",),
        ("gpu_name",),
        ("gpu_identity", "gpu_name"),
    )
    active_fit = _binding_value(
        payload, ("active_cuda_fit",), ("active_cuda_fit_status",)
    )
    raw_reload = _binding_value(
        payload,
        ("raw_booster_reload_on_cuda",),
        ("raw_booster_reload_status",),
    )
    fallback = _binding_value(
        payload, ("cpu_fallback",), ("cpu_fallback_detected",)
    )
    if not (
        payload.get("status") == "PASS"
        and executable == str(CUDA_PYTHON)
        and xgb_version == XGBOOST_VERSION
        and sklearn_version == SCIKIT_LEARN_VERSION
        and use_cuda is True
        and gpu == VISIBLE_GPU
        and active_fit == "PASS"
        and raw_reload == "PASS"
        and fallback is False
    ):
        raise RecoveryIntegrityError("A04 live CUDA binding contract failed")
    return {
        "status": "PASS",
        "sys_executable": executable,
        "xgboost_version": xgb_version,
        "scikit_learn_version": sklearn_version,
        "xgboost_use_cuda": True,
        "visible_gpu": gpu,
        "active_cuda_fit": "PASS",
        "raw_booster_reload_on_cuda": "PASS",
        "cpu_fallback_detected": False,
    }


def _validate_stored_csv_diagnostic(payload: Mapping[str, Any]) -> dict[str, Any]:
    default = payload.get("default_read", {})
    round_trip = payload.get("round_trip_read", {})
    mismatches = payload.get("mismatches", [])
    if not (
        payload.get("status") == "PASS"
        and payload.get("csv_sha256")
        == FAILED_AMENDMENT03_KEY_HASHES["tables/SMOKE_REPORT.csv"]
        and payload.get("json_sha256")
        == FAILED_AMENDMENT03_KEY_HASHES["metrics/SMOKE_REPORT.json"]
        and payload.get("metric_names") == list(METRIC_NAMES)
        and payload.get("comparison_count") == 50
        and default.get("mismatch_count") == 22
        and default.get("max_abs_difference") == 1.1102230246251565e-16
        and default.get("max_ulp_distance") == 1
        and round_trip.get("mismatch_count") == 0
        and round_trip.get("max_abs_difference") == 0.0
        and round_trip.get("max_ulp_distance") == 0
        and isinstance(mismatches, list)
        and len(mismatches) == 22
        and all(
            isinstance(row, Mapping)
            and row.get("metric") in METRIC_NAMES
            and row.get("variant") in EXPECTED_VARIANT_IDS
            and math.isfinite(float(row.get("default_value")))
            and math.isfinite(float(row.get("json_value")))
            and row.get("abs_difference") == 1.1102230246251565e-16
            and row.get("ulp_distance") == 1
            for row in mismatches
        )
        and payload.get("default_all_mismatches_one_ulp") is True
        and payload.get("exact_metric_equality_retained") is True
        and payload.get("tolerance_or_rounding_added") is False
        and payload.get("csv_or_json_rewritten") is False
    ):
        raise RecoveryIntegrityError("Stored A04 CSV round-trip diagnostic failed")
    return {
        "status": "PASS",
        "default_mismatch_count": 22,
        "default_max_abs_difference": 1.1102230246251565e-16,
        "default_max_ulp_distance": 1,
        "round_trip_mismatch_count": 0,
        "all_default_mismatches_one_ulp": True,
    }


def validate_prerun_evidence(
    study_root: Path = STUDY_ROOT,
) -> dict[str, Any]:
    """Validate and return immutable bytes copied immediately after Run mkdir."""

    root = Path(study_root)
    source_bytes: dict[str, bytes] = {}
    destination_sha256: dict[str, str] = {}
    for source_relative, destination_relative in AMENDMENT04_PRERUN_COPY_MAPPING.items():
        source = root / source_relative
        if not source.is_file() or source.is_symlink() or source.stat().st_size == 0:
            raise RecoveryIntegrityError(f"A04 pre-Run evidence is absent: {source}")
        data = source.read_bytes()
        source_bytes[destination_relative] = data
        destination_sha256[destination_relative] = sha256_bytes(data)
    if set(source_bytes) != set(AMENDMENT04_PRERUN_DESTINATIONS):
        raise RecoveryIntegrityError("A04 pre-Run destination set changed")

    a03_doc = root / "docs/PROTOCOL_AMENDMENT_03.md"
    failed_a03_doc = (
        root / "results" / FAILED_AMENDMENT03_RUN_ID
        / "docs/PROTOCOL_AMENDMENT_03.md"
    )
    if a03_doc.read_bytes() != failed_a03_doc.read_bytes():
        raise RecoveryIntegrityError("Protocol Amendment 03 document drifted")
    a04_doc_text = (root / "docs/PROTOCOL_AMENDMENT_04.md").read_text()
    required_doc_tokens = (
        "Amendment 04",
        FAILED_AMENDMENT03_RUN_ID,
        "22",
        "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
        "Full",
    )
    if not all(token in a04_doc_text for token in required_doc_tokens) or not (
        "round-trip" in a04_doc_text.lower()
        or "round_trip" in a04_doc_text.lower()
    ):
        raise RecoveryIntegrityError("Protocol Amendment 04 document is incomplete")

    try:
        live_payload = json.loads(
            source_bytes["provenance/amendment04_live_cuda_binding.json"]
        )
        diagnostic_payload = json.loads(
            source_bytes["provenance/csv_roundtrip_diagnostic.json"]
        )
    except Exception as exc:
        raise RecoveryIntegrityError("A04 pre-Run JSON evidence is malformed") from exc
    live = _validate_live_cuda_binding(live_payload)
    diagnostic = _validate_stored_csv_diagnostic(diagnostic_payload)

    log_labels = {
        "logs/amendment04_syntax_compile.log": "syntax",
        "logs/amendment04_targeted_tests.log": "targeted",
        "logs/full_tests.log": "full",
        "logs/cuda_reload_probe.log": "cuda",
        "logs/package_rehearsal.log": "package",
    }
    logs: dict[str, dict[str, Any]] = {}
    for destination, label in log_labels.items():
        record = _parse_command_log(
            source_bytes[destination],
            source_relative=destination,
            study_root=root,
        )
        _require_log_command(record, label=label)
        logs[destination] = {
            "status": "PASS",
            "command": record["command"],
            "cwd": record["cwd"],
            "exit_code": record["exit_code"],
            "wall_seconds": record["wall_seconds"],
            "sha256": destination_sha256[destination],
        }
    return {
        "status": "PASS",
        "source_bytes": source_bytes,
        "destination_sha256": dict(sorted(destination_sha256.items())),
        "logs": logs,
        "live_cuda_binding": live,
        "csv_roundtrip_diagnostic": diagnostic_payload,
        "csv_roundtrip_diagnostic_validation": diagnostic,
    }


def build_lineage(
    *,
    run_dir: Path,
    new_run_id: str,
    immutable_gate: Mapping[str, Any],
    csv_diagnostic: Mapping[str, Any],
    executed_provenance: Mapping[str, Any] | None = None,
    executed_code: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write A04 lineage binding both failed Runs and the four-path repair."""

    run = Path(run_dir)
    executed = executed_provenance if executed_provenance is not None else executed_code
    if executed is None:
        raise RecoveryIntegrityError("A04 lineage requires executed provenance")
    if executed_provenance is not None and executed_code is not None:
        if dict(executed_provenance) != dict(executed_code):
            raise RecoveryIntegrityError("Conflicting A04 executed provenance aliases")
    if (
        run.name != new_run_id
        or not _valid_amendment04_run_id(new_run_id)
        or immutable_gate.get("status") != "PASS"
        or immutable_gate.get("authorization") != AUTHORIZATION
        or executed.get("status") != "PASS"
        or set(executed.get("changed_paths", []))
        != set(AMENDMENT04_ALLOWED_CODE_DRIFT_PATHS)
        or set(executed.get("before_hashes", {}))
        != set(AMENDMENT04_ALLOWED_CODE_DRIFT_PATHS)
        or set(executed.get("after_hashes", {}))
        != set(AMENDMENT04_ALLOWED_CODE_DRIFT_PATHS)
    ):
        raise RecoveryIntegrityError("A04 lineage inputs are not exact")
    diagnostic_summary = _validate_stored_csv_diagnostic(csv_diagnostic)
    guard = assert_run_guard(run.parent, new_run_id=new_run_id)
    if guard.get("authorization") != "1_OF_1_CONSUMED":
        raise RecoveryIntegrityError("A04 lineage cannot bind an unconsumed Run")

    preflight = immutable_gate["accepted_preflight"]
    failed_a02 = immutable_gate["failed_amendment02_smoke"]
    failed_a03 = immutable_gate["failed_amendment03_smoke"]
    lineage = {
        "classification": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
        "status": "PASS",
        "authorization": AUTHORIZATION,
        "run_authorization": "1_OF_1_CONSUMED",
        "new_smoke_run_id": new_run_id,
        "new_smoke_run_path": str(run),
        "selected_action": "FRESH_SMOKE_NOT_PACKAGE_ONLY",
        "prompt": immutable_gate["prompt"],
        "accepted_preflight": {
            "run_id": preflight["run_id"],
            "run_path": preflight["run_path"],
            "review_zip_path": preflight["review_zip_path"],
            "review_zip_sha256": preflight["review_zip_sha256"],
            "package_verification_sha256": preflight[
                "package_verification_sha256"
            ],
            "locked_split_sha256": preflight["locked_split_sha256"],
            "base_seed": preflight["base_seed"],
            "sealed_probe": preflight["sealed_probe"],
        },
        "failed_smokes": {
            "amendment02": {
                "run_id": failed_a02["run_id"],
                "run_path": failed_a02["run_path"],
                "tree": failed_a02["tree"],
                "failure_evidence": failed_a02["failure_evidence"],
                "runtime_models": failed_a02["runtime_models"],
                "preserved": True,
            },
            "amendment03": {
                "run_id": failed_a03["run_id"],
                "run_path": failed_a03["run_path"],
                "tree": failed_a03["tree"],
                "key_hashes": failed_a03["key_hashes"],
                "forensic_zip": failed_a03["forensic_zip"],
                "runtime_models": failed_a03["runtime_models"],
                "preserved": True,
            },
        },
        "csv_roundtrip_repair": {
            **diagnostic_summary,
            "csv_sha256": csv_diagnostic["csv_sha256"],
            "json_sha256": csv_diagnostic["json_sha256"],
            "reader": 'pd.read_csv(path, engine="c", float_precision="round_trip")',
            "exact_metric_equality_retained": True,
            "tolerance_or_rounding_added": False,
            "csv_or_json_rewritten": False,
            "writer_changed": False,
        },
        "executed_code": {
            "baseline": executed.get("baseline"),
            "baseline_tsv_sha256": executed.get("baseline_tsv_sha256"),
            "changed_paths": list(executed["changed_paths"]),
            "before_hashes": dict(sorted(executed["before_hashes"].items())),
            "after_hashes": dict(sorted(executed["after_hashes"].items())),
            "executed_code_hashes_tsv_sha256": executed[
                "executed_code_hashes_tsv_sha256"
            ],
            "snapshot_hashes": dict(sorted(executed["snapshot_hashes"].items())),
        },
        "reuse_contract": {
            "old_run_resumed": False,
            "old_models_reused": False,
            "old_raw_models_reused": False,
            "old_augmented_matrices_reused": False,
            "old_predictions_reused": False,
            "old_metric_values_reused": False,
            "new_resampling_required": True,
        },
        "scientific_contract": {
            "protocol": "AMENDMENT_02_UNCHANGED",
            "locked_split_sha256": LOCKED_SPLIT_SHA256,
            "base_seed": BASE_SEED,
            "full_authorized": False,
            "stability_authorized": False,
            "multi_seed_authorized": False,
            "external_audit_scoring_authorized": False,
            "scientific_results_claimed": False,
        },
        "cuda_contract": immutable_gate["cuda_contract"],
    }
    _write_new_json(run / "provenance/AMENDMENT04_LINEAGE.json", lineage)
    return lineage


def _validate_dual_probe(run: Path) -> dict[str, Any]:
    copies: dict[str, dict[str, Any]] = {}
    payloads: list[bytes] = []
    for relative in SEALED_PROBE_DESTINATIONS:
        path = run / relative
        if not (
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == SEALED_PROBE_SIZE_BYTES
            and sha256_file(path) == SEALED_PROBE_SHA256
        ):
            raise RecoveryIntegrityError(f"A04 sealed probe copy failed: {relative}")
        data = path.read_bytes()
        payloads.append(data)
        copies[relative] = {
            "size_bytes": len(data),
            "sha256": sha256_bytes(data),
        }
    if payloads[0] != payloads[1]:
        raise RecoveryIntegrityError("A04 root/nested sealed probe bytes differ")
    binding_path = run / "provenance/sealed_probe_binding.json"
    if not binding_path.is_file() or binding_path.is_symlink():
        raise RecoveryIntegrityError("A04 sealed-probe binding is absent/unsafe")
    try:
        binding = json.loads(binding_path.read_text())
    except Exception as exc:
        raise RecoveryIntegrityError("A04 sealed-probe binding is malformed") from exc
    destinations = binding.get("destinations", {})
    manifest_rows = binding.get("manifest_rows", {})
    expected_row = {
        "relative_path": SEALED_PROBE_RELATIVE,
        "size_bytes": SEALED_PROBE_SIZE_BYTES,
        "sha256": SEALED_PROBE_SHA256,
    }
    if not (
        binding.get("status") == "PASS"
        and binding.get("source_zip_sha256") == ACCEPTED_PREFLIGHT_ZIP_SHA256
        and binding.get("source_member") == SEALED_PROBE_RELATIVE
        and binding.get("member_sha256") == SEALED_PROBE_SHA256
        and binding.get("member_size_bytes") == SEALED_PROBE_SIZE_BYTES
        and binding.get("manifest_match") == "PASS"
        and binding.get("runtime_probe_used_as_source") is False
        and binding.get("root_nested_byte_equal") is True
        and manifest_rows.get("OUTPUT_MANIFEST_FINAL.tsv") == expected_row
        and manifest_rows.get("BUNDLE_MANIFEST.tsv") == expected_row
        and set(destinations) == set(SEALED_PROBE_DESTINATIONS)
        and all(
            destinations[relative].get("sha256") == SEALED_PROBE_SHA256
            and destinations[relative].get("size_bytes")
            == SEALED_PROBE_SIZE_BYTES
            and destinations[relative].get("byte_equal_to_sealed_member") is True
            for relative in SEALED_PROBE_DESTINATIONS
        )
    ):
        raise RecoveryIntegrityError("A04 sealed-probe binding contract failed")
    return {
        "status": "PASS",
        "sha256": SEALED_PROBE_SHA256,
        "size_bytes": SEALED_PROBE_SIZE_BYTES,
        "copies": copies,
        "binding_sha256": sha256_file(binding_path),
    }


def _lineage_contract_valid(
    lineage: Mapping[str, Any],
    *,
    run: Path,
    executed: Mapping[str, Any],
    immutable: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
) -> bool:
    failed = lineage.get("failed_smokes", {})
    reuse = lineage.get("reuse_contract", {})
    scientific = lineage.get("scientific_contract", {})
    repair = lineage.get("csv_roundtrip_repair", {})
    recorded_execution = lineage.get("executed_code", {})
    return bool(
        lineage.get("classification") == "NON_SCIENTIFIC_DIAGNOSTIC_ONLY"
        and lineage.get("status") == "PASS"
        and lineage.get("authorization") == AUTHORIZATION
        and lineage.get("run_authorization") == "1_OF_1_CONSUMED"
        and lineage.get("new_smoke_run_id") == run.name
        and lineage.get("new_smoke_run_path") == str(run)
        and lineage.get("selected_action") == "FRESH_SMOKE_NOT_PACKAGE_ONLY"
        and lineage.get("prompt") == immutable.get("prompt")
        and lineage.get("accepted_preflight", {}).get("run_id")
        == ACCEPTED_PREFLIGHT_RUN_ID
        and lineage.get("accepted_preflight", {}).get("review_zip_sha256")
        == ACCEPTED_PREFLIGHT_ZIP_SHA256
        and lineage.get("accepted_preflight", {}).get("locked_split_sha256")
        == LOCKED_SPLIT_SHA256
        and lineage.get("accepted_preflight", {}).get("base_seed") == BASE_SEED
        and failed.get("amendment02", {}).get("run_id")
        == FAILED_AMENDMENT02_RUN_ID
        and failed.get("amendment02", {}).get("tree", {}).get("tree_sha256")
        == FAILED_AMENDMENT02_TREE["tree_sha256"]
        and failed.get("amendment02", {}).get("runtime_models", {}).get(
            "tree_sha256"
        ) == FAILED_AMENDMENT02_RUNTIME_TREE["tree_sha256"]
        and failed.get("amendment02", {}).get("preserved") is True
        and failed.get("amendment03", {}).get("run_id")
        == FAILED_AMENDMENT03_RUN_ID
        and failed.get("amendment03", {}).get("tree", {}).get("tree_sha256")
        == FAILED_AMENDMENT03_TREE["tree_sha256"]
        and failed.get("amendment03", {}).get("key_hashes")
        == FAILED_AMENDMENT03_KEY_HASHES
        and failed.get("amendment03", {}).get("forensic_zip", {}).get("sha256")
        == FAILED_AMENDMENT03_FORENSIC_ZIP_SHA256
        and failed.get("amendment03", {}).get("runtime_models", {}).get(
            "tree_sha256"
        ) == FAILED_AMENDMENT03_RUNTIME_TREE["tree_sha256"]
        and failed.get("amendment03", {}).get("preserved") is True
        and repair.get("default_mismatch_count") == 22
        and repair.get("round_trip_mismatch_count") == 0
        and repair.get("default_max_ulp_distance") == 1
        and repair.get("csv_sha256")
        == FAILED_AMENDMENT03_KEY_HASHES["tables/SMOKE_REPORT.csv"]
        and repair.get("json_sha256")
        == FAILED_AMENDMENT03_KEY_HASHES["metrics/SMOKE_REPORT.json"]
        and repair.get("exact_metric_equality_retained") is True
        and repair.get("tolerance_or_rounding_added") is False
        and repair.get("csv_or_json_rewritten") is False
        and recorded_execution.get("changed_paths") == executed.get("changed_paths")
        and recorded_execution.get("before_hashes") == executed.get("before_hashes")
        and recorded_execution.get("after_hashes") == executed.get("after_hashes")
        and recorded_execution.get("executed_code_hashes_tsv_sha256")
        == executed.get("executed_code_hashes_tsv_sha256")
        and all(value is False for key, value in reuse.items() if key != "new_resampling_required")
        and reuse.get("new_resampling_required") is True
        and scientific.get("locked_split_sha256") == LOCKED_SPLIT_SHA256
        and scientific.get("base_seed") == BASE_SEED
        and scientific.get("full_authorized") is False
        and scientific.get("scientific_results_claimed") is False
        and diagnostic.get("status") == "PASS"
    )


def validate_recovery_artifacts(
    *,
    run_dir: Path,
    allowed_after_hashes: Mapping[str, str],
    study_root: Path = STUDY_ROOT,
    immutable_gate: Mapping[str, Any] | None = None,
    require_prompt_artifacts: bool = False,
    baseline_executed_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Recompute A04 lineage, code, probe, pre-Run, and immutable contracts."""

    root = Path(study_root)
    run = Path(run_dir)
    if not run.is_dir() or run.is_symlink() or not _valid_amendment04_run_id(run.name):
        raise RecoveryIntegrityError("A04 Run directory is absent/unsafe")
    immutable = validate_immutable_gate(root)
    if immutable_gate is not None:
        expected_scalars = (
            immutable_gate.get("status") == immutable.get("status") == "PASS"
            and immutable_gate.get("prompt") == immutable.get("prompt")
            and immutable_gate.get("accepted_preflight", {}).get("review_zip_sha256")
            == immutable.get("accepted_preflight", {}).get("review_zip_sha256")
            and immutable_gate.get("failed_amendment03_smoke", {}).get("tree")
            == immutable.get("failed_amendment03_smoke", {}).get("tree")
            and immutable_gate.get("failed_amendment03_smoke", {}).get("key_hashes")
            == immutable.get("failed_amendment03_smoke", {}).get("key_hashes")
        )
        if not expected_scalars:
            raise RecoveryIntegrityError("A04 immutable before/after gate differs")
    guard = assert_run_guard(run.parent, new_run_id=run.name)
    executed = _validate_executed_provenance(
        run_dir=run,
        allowed_after_hashes=allowed_after_hashes,
        study_root=root,
        baseline_executed_hashes=baseline_executed_hashes,
    )
    probe = _validate_dual_probe(run)

    lineage_path = run / "provenance/AMENDMENT04_LINEAGE.json"
    diagnostic_path = run / "provenance/csv_roundtrip_diagnostic.json"
    live_binding_path = run / "provenance/amendment04_live_cuda_binding.json"
    for path in (lineage_path, diagnostic_path, live_binding_path):
        if not path.is_file() or path.is_symlink():
            raise RecoveryIntegrityError(f"A04 recovery artifact is absent: {path}")
    try:
        lineage = json.loads(lineage_path.read_text())
        diagnostic_payload = json.loads(diagnostic_path.read_text())
        live_payload = json.loads(live_binding_path.read_text())
    except Exception as exc:
        raise RecoveryIntegrityError("A04 recovery JSON is malformed") from exc
    diagnostic = _validate_stored_csv_diagnostic(diagnostic_payload)
    live = _validate_live_cuda_binding(live_payload)
    if not _lineage_contract_valid(
        lineage,
        run=run,
        executed=executed,
        immutable=immutable,
        diagnostic=diagnostic,
    ):
        raise RecoveryIntegrityError("A04 lineage contract failed revalidation")

    prerun: dict[str, Any] | None = None
    if require_prompt_artifacts:
        missing = sorted(
            relative
            for relative in AMENDMENT04_RECOVERY_REQUIRED_FILES
            if not (run / relative).is_file()
            or (run / relative).is_symlink()
            or (run / relative).stat().st_size == 0
        )
        code_snapshots = sorted(
            path for path in (run / "provenance/executed_code_snapshot").glob("*")
            if path.is_file() and not path.is_symlink()
        )
        test_snapshots = sorted(
            path for path in (run / "provenance/executed_tests_snapshot").glob("*")
            if path.is_file() and not path.is_symlink()
        )
        if missing or not code_snapshots or not test_snapshots:
            raise RecoveryIntegrityError(
                "A04 required package evidence is incomplete: "
                f"missing={missing}, code_snapshots={len(code_snapshots)}, "
                f"test_snapshots={len(test_snapshots)}"
            )
        prerun = validate_prerun_evidence(root)
        for destination, expected in prerun["destination_sha256"].items():
            copied = run / destination
            if not (
                copied.is_file()
                and not copied.is_symlink()
                and sha256_file(copied) == expected
            ):
                raise RecoveryIntegrityError(
                    f"A04 copied pre-Run evidence drifted: {destination}"
                )
    return {
        "status": "PASS",
        "authorization": AUTHORIZATION,
        "run_authorization": guard["authorization"],
        "sealed_probe_sha256": probe["sha256"],
        "sealed_probe_binding_sha256": probe["binding_sha256"],
        "executed_code": executed,
        "executed_code_hashes_tsv_sha256": executed[
            "executed_code_hashes_tsv_sha256"
        ],
        "lineage_sha256": sha256_file(lineage_path),
        "csv_roundtrip_diagnostic_sha256": sha256_file(diagnostic_path),
        "live_cuda_binding_sha256": sha256_file(live_binding_path),
        "live_cuda_binding": live,
        "csv_roundtrip_diagnostic": diagnostic,
        "immutable_gate_status": immutable["status"],
        "required_prompt_artifacts": (
            "PASS" if require_prompt_artifacts else "NOT_REQUIRED_AT_THIS_PHASE"
        ),
        "prerun_evidence": (
            {
                "status": prerun["status"],
                "destination_sha256": prerun["destination_sha256"],
            }
            if prerun is not None
            else None
        ),
    }


__all__ = [
    "ACCEPTED_PREFLIGHT_RUN_ID",
    "ACCEPTED_PREFLIGHT_ZIP_SHA256",
    "AMENDMENT04_ALLOWED_CODE_DRIFT_PATHS",
    "AMENDMENT04_PRERUN_COPY_MAPPING",
    "AMENDMENT04_PRERUN_DESTINATIONS",
    "AMENDMENT04_RECOVERY_REQUIRED_FILES",
    "AUTHORIZATION",
    "BASE_SEED",
    "CUDA_ENV",
    "CUDA_PYTHON",
    "FAILED_AMENDMENT02_RUN_ID",
    "FAILED_AMENDMENT03_FORENSIC_ZIP_SHA256",
    "FAILED_AMENDMENT03_KEY_HASHES",
    "FAILED_AMENDMENT03_RUN_ID",
    "LOCKED_SPLIT_SHA256",
    "METRIC_NAMES",
    "PROMPT_PATH",
    "PROMPT_SHA256",
    "RecoveryIntegrityError",
    "SEALED_PROBE_SHA256",
    "SEALED_PROBE_SIZE_BYTES",
    "assert_run_guard",
    "build_executed_provenance",
    "build_lineage",
    "code_drift_allowlist",
    "create_authorized_run_directory",
    "csv_roundtrip_diagnostic",
    "read_smoke_report_roundtrip",
    "reference_code_drift_allowlist",
    "sha256_bytes",
    "sha256_file",
    "validate_immutable_gate",
    "validate_prerun_evidence",
    "validate_recovery_artifacts",
]
