"""Amendment 06 resumable Full scientific ablation runner.

This module is the scientific execution boundary for Amendment 06.  It never
delegates to the obsolete legacy Full runner.  Package construction is kept in
``amendment06_packaging`` and is imported only by :func:`package_dispatch`.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, suppress
import ctypes
import csv
from dataclasses import dataclass
from datetime import datetime
import errno
import fcntl
import gzip
import hashlib
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import stat
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence
import uuid
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import xgboost as xgb

import ablation_core as legacy
import amendment02_resampling as paired_resampling
import amendment03_compat as model_compat
import amendment_core as core
import amendment06_reporting as reporting


STUDY_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path("/REVIEWER_INPUT_ROOT/Clean")
RESULTS_ROOT = STUDY_ROOT / "results"
A06_RUNTIME = STUDY_ROOT / ".runtime/amendment06_full"
EXPECTED_RUNTIME_ROOT = A06_RUNTIME / "runtime"
CUDA_ENV = Path("/REVIEWER_INPUT_ROOT/SAM_ablation_cuda_xgb211_wheel")
CUDA_PYTHON = CUDA_ENV / "bin/python"

PROMPT_PATH = Path("/REVIEWER_INPUT_ROOT/06_RUN_NOW_Amendment06_Full_Scientific_Ablation.txt")
PROMPT_SHA256 = "246928379ce2871cf576dcba09a2f8f4551bed1060acb6d5a1d82ab96f126e5b"
AUTHORIZATION = "AMENDMENT_06_EXTERNAL_REVIEW_ACCEPTED_ONE_FULL"
EXTERNAL_REVIEW_DECISION = "ACCEPT_AMENDMENT05_SEALED_SMOKE_FOR_FULL"
RUN_KIND = "FULL_SCIENTIFIC_ABLATION"
RUN_SUFFIX = "_amendment06_full"

ACCEPTED_PREFLIGHT = RESULTS_ROOT / "run_20260813_050405_9970c566_amendment02_preflight"
ACCEPTED_PREFLIGHT_ZIP = Path(str(ACCEPTED_PREFLIGHT) + "_review_bundle.zip")
ACCEPTED_SMOKE = RESULTS_ROOT / "run_20260820_104526_263432_bc48422a_amendment04_cuda_smoke"
ACCEPTED_SMOKE_REVIEW_ZIP = Path(str(ACCEPTED_SMOKE) + "_review_bundle.zip")
ACCEPTED_SMOKE_REVIEW_VERIFICATION = RESULTS_ROOT / f"{ACCEPTED_SMOKE.name}_package_verification.json"
ACCEPTED_SMOKE_MODELS_ZIP = RESULTS_ROOT / f"{ACCEPTED_SMOKE.name}_NON_SCIENTIFIC_smoke_models.zip"
ACCEPTED_SMOKE_MODELS_VERIFICATION = RESULTS_ROOT / f"{ACCEPTED_SMOKE.name}_NON_SCIENTIFIC_smoke_models_package_verification.json"
A05_RECOVERY_VERIFICATION = A06_RUNTIME.parent / "amendment05_seal_only/final/amendment05_recovery_verification.json"
PRE_EDIT_ADMISSION = A06_RUNTIME / "pre_edit/admission_validation.json"
AUTHORIZED_CHANGE_LEDGER = A06_RUNTIME / "authorized_change_ledger.jsonl"
FINAL_PRERUN_EVIDENCE = A06_RUNTIME / "pre_run/final_prerun_evidence.json"
BASELINE_TREE_TSV = A06_RUNTIME / "pre_edit/baseline_tree_sha_size_rel.tsv"

PINNED_FILES: Mapping[Path, str] = {
    PROMPT_PATH: PROMPT_SHA256,
    ACCEPTED_PREFLIGHT_ZIP: "19d6d9ec381bab99ff41a081fd42016c653d4a4ce947f35694e0e32e57ae24cb",
    ACCEPTED_SMOKE_REVIEW_ZIP: "4705906df60b38965d2a4febad5483c7543ab7fa250eb866dcb99b2ebad48c0c",
    ACCEPTED_SMOKE_REVIEW_VERIFICATION: "94e64bcbae08a7d63129e576469dccd4528fc4b3711fdff6904b96a1d3f9b790",
    ACCEPTED_SMOKE_MODELS_ZIP: "e14a3b7cd7cc499384f88b471ad88a5c51eb30e5a0f92e1711b899c72ff9777e",
    ACCEPTED_SMOKE_MODELS_VERIFICATION: "f8e5845a6d8557c33baa4b841f45d89a48311f3a653d28a2663cdf7586126d44",
    A05_RECOVERY_VERIFICATION: "271c4cf4ae36e2e0bb589c45f11767d202f996c5eba512acc59020cd92a4dbc2",
    PRE_EDIT_ADMISSION: "b4281ee4f112c7d781a0ca528eb83189f295768eb2a632d4e00ee25a206bd8d3",
}

PRIMARY_TABLE = Path("/REVIEWER_INPUT_ROOT/IMG_9429 (2025-12-28_11-55-48)/features_train.csv")
PRIMARY_TABLE_SHA256 = "1836d704d15f27728e3ba09d7cddf0aea9458049bc66700dc98f14e7111ec29d"
EXTERNAL_AUDIT = Path("/REVIEWER_INPUT_ROOT/testset_labeled__xgb_recall.csv")
EXTERNAL_AUDIT_SHA256 = "c2be6932a56f13117b18e8ed5750dac5f5317fce77ff1ece41ddf948a0ffcee6"
SPLIT_ASSIGNMENT_SHA256 = "f2cd1570d2c76af49ed52593d95657a3cbbe14d8f2f077edd05447e608698c64"

TRAINING_CONFIGURATION_SHA256 = "3b0103a3d35f76448ddedcab8bb4d1a3d6ba37b675c163a922a437630ea9fcf3"
VARIANT_FEATURE_SETS_SHA256 = "5e2578570064e51a52301ef1cd065ea8c9a42c29a46ce11b2b22571d0c7f3be1"
FEATURE_MANIFEST_SHA256 = "8a4a922b53ddad9bd3d89e3452c8292774b3df2c5d149619c588a40877ca4e84"
FEATURE_DEPENDENCY_GRAPH_SHA256 = "2664c6c9433ffd97e33413c60f1f07e5efbf8ad6614afba3f72eca730cec1836"
ROW_SPLIT_MANIFEST_SHA256 = "8b2b81b2691f00167ef0537049cfd4e63d5a17d0cec52774c1d189a823470c1a"
SOURCE_INVENTORY_SHA256 = "795279d24a3b7e07a65ed0f3a95d87969e19feafd58db182a5d26c7d2b3f762d"
FULL_MODEL_PARAMETERS_SHA256 = "f8b69bce5432d077781427897b73ff2dd7c759c3ca6be9afd5fe042aea0ec14a"
CONTINUOUS_FEATURE_ORDER_SHA256 = "b035ac55a40010f68a8b194f6b03d2e9da03c754198712bb52969f3473802320"

BASE_SEED = 42
PRIMARY_MODEL_SEED = 42
DECISION_THRESHOLD = 0.5
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026081201

OFFICIAL_VARIANTS = (
    "full_new_reference",
    "manual_only",
    "deep_only",
    "no_color",
    "no_shape",
    "no_texture",
    "no_embed_sim",
    "no_review_aware_training_weights",
    "no_safe_smote",
)
EXPLORATORY_VARIANT = "no_deep_pca_features"
TRAINED_VARIANTS = (*OFFICIAL_VARIANTS, EXPLORATORY_VARIANT)
BLOCKED_VARIANT = "no_pca"
VARIANT_FEATURE_CONTRACT: Mapping[str, tuple[int, str]] = {
    "full_new_reference": (93, "a5478eb2e571809d80fbdf0e132d39f301d936b0b711ceb08a326ade0b24ed47"),
    "manual_only": (57, "a8f5cb9445994db38375367993fc54d93a595dc9dd9c8d4dc6518f9d99e7655c"),
    "deep_only": (35, "869d491c431cccf3e5a11ddea608ef8ed55f7534ecd624204c176be9848121ab"),
    "no_color": (65, "4c8131fa99c15ec3f7ed8adbcfe759ac1fb480c22e2694094e8b18506f1fecf9"),
    "no_shape": (78, "39ff51f6192fed1d47d17971fb8974f58b0d12afed587d28ef22a30e70625d54"),
    "no_texture": (86, "a72ba7cbff49608ea8266f61db1c2df7df3e3129383d9343ce074b6aa3372178"),
    "no_embed_sim": (90, "beffdb1f3966ac8a2cea3c9a1ba826d744cb1d9546c117e8b0c9780b42b059bc"),
    "no_review_aware_training_weights": (93, "a5478eb2e571809d80fbdf0e132d39f301d936b0b711ceb08a326ade0b24ed47"),
    "no_safe_smote": (93, "a5478eb2e571809d80fbdf0e132d39f301d936b0b711ceb08a326ade0b24ed47"),
    "no_deep_pca_features": (59, "abd482e8f546145db98362e57d77b27d7af0c8539d90d1f542e6a159aafd96ce"),
}

EXPECTED_STAGE_COUNTS: Mapping[str, tuple[int, int, int]] = {
    "raw_locked_post_split_shuffle": (291024, 215988, 75036),
    "post_mixup": (436536, 323982, 112554),
    "post_dropout": (873072, 647964, 225108),
    "post_jitter_pre_smote": (1309608, 971633, 337975),
    "post_shuffle": (1309608, 971633, 337975),
    "post_safe_smote": (1457449, 971633, 485816),
}
PRE_SMOTE_ROWS = 1_309_608
POST_SMOTE_ROWS = 1_457_449

REQUIRED_METRICS = (
    "average_precision", "roc_auc", "precision", "recall", "f1",
    "tn", "fp", "fn", "tp",
)
OUTPUT_MANIFEST_COLUMNS = (
    "relative_path", "size_bytes", "sha256", "artifact_role",
    "include_in_review_bundle", "include_in_models_bundle",
)
SCIENTIFIC_CODE_RELATIVES = (
    "code/ablation_core.py",
    "code/amendment02_resampling.py",
    "code/amendment03_compat.py",
    "code/amendment03_recovery.py",
    "code/amendment04_recovery.py",
    "code/amendment_inventory.py",
    "code/amendment_core.py",
    "code/amendment06_full.py",
    "code/amendment06_reporting.py",
    "code/evidence_gates.py",
    "code/run_ablation_study.sh",
    "code/run_study.py",
    "docs/PROTOCOL_AMENDMENT_06.md",
)
SCIENTIFIC_TEST_RELATIVES = (
    "tests/test_ablation.py",
    "tests/test_amendment02_resampling.py",
    "tests/test_amendment03.py",
    "tests/test_amendment03_recovery.py",
    "tests/test_amendment04.py",
    "tests/test_amendment06.py",
    "tests/test_amendment06_cli.py",
    "tests/test_amendment06_inventory.py",
    "tests/test_amendment06_reporting.py",
    "tests/test_amendment_inventory.py",
    "tests/test_evidence_gates.py",
)
PACKAGE_CODE_RELATIVES = (
    "code/amendment06_packaging.py",
    "tests/test_amendment06_packaging.py",
)
RUN_PHASES = (
    "FULL_IN_PROGRESS",
    "FULL_MODELS_FROZEN",
    "FULL_EXTERNAL_AUDIT_SCORED",
    "FULL_SCIENTIFIC_COMPLETE",
)


class Amendment06Error(RuntimeError):
    """Base Amendment 06 failure."""


class Amendment06IntegrityError(Amendment06Error):
    """An immutable or completion contract failed."""


class Amendment06ResumeError(Amendment06Error):
    """The only authorized Full Run cannot be resumed safely."""


class StrictJSONError(Amendment06Error):
    """New Full JSON is not strict RFC-8259 JSON."""


class PostFreezeScientificCallError(Amendment06Error):
    """A forbidden scientific-fit API was called after model freeze."""


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _progress(phase: str, status: str, **fields: Any) -> None:
    suffix = " ".join(f"{key.upper()}={value}" for key, value in fields.items())
    print(f"A06_PHASE_{status.upper()}={phase}" + (f" {suffix}" if suffix else ""), flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    view = memoryview(array).cast("B")
    digest = hashlib.sha256()
    chunk_size = 8 << 20
    for offset in range(0, len(view), chunk_size):
        digest.update(view[offset:offset + chunk_size])
    return digest.hexdigest()


def _strict_copy(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_strict_copy(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _strict_copy(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _strict_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_copy(item) for item in value]
    raise StrictJSONError(f"Unsupported strict JSON value: {type(value).__name__}")


def strict_full_json_bytes(value: Any) -> bytes:
    copied = _strict_copy(value)
    return (json.dumps(
        copied, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True,
    ) + "\n").encode("utf-8")


def strict_full_canonical_json_bytes(value: Any) -> bytes:
    copied = _strict_copy(value)
    return json.dumps(
        copied, allow_nan=False, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_json_constant(value: str) -> Any:
    raise StrictJSONError(f"Non-standard JSON numeric constant: {value}")


def strict_full_loads(value: bytes | str) -> Any:
    try:
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        return json.loads(text, parse_constant=_reject_json_constant)
    except StrictJSONError:
        raise
    except Exception as exc:
        raise StrictJSONError(f"Strict JSON parse failed: {type(exc).__name__}: {exc}") from exc


def strict_full_load_file(path: Path) -> Any:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise StrictJSONError(f"Strict JSON path is absent or unsafe: {target}")
    return strict_full_loads(target.read_bytes())


def assert_required_metrics_finite(
    metrics: Mapping[str, Any], required: Sequence[str] = REQUIRED_METRICS,
) -> None:
    invalid: list[str] = []
    for name in required:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            invalid.append(name)
            continue
        if not math.isfinite(float(value)):
            invalid.append(name)
    if invalid:
        raise Amendment06IntegrityError(f"Required metrics are missing/nonfinite: {invalid}")


def validate_strict_json_tree(
    root: Path, *, include: Iterable[Path] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    paths = sorted(include if include is not None else root.rglob("*.json"))
    checked: list[dict[str, Any]] = []
    for path in paths:
        target = Path(path)
        strict_full_load_file(target)
        checked.append({
            "relative_path": target.relative_to(root).as_posix(),
            "size_bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        })
    return {"status": "PASS", "strict_rfc8259": True, "file_count": len(checked), "files": checked}


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def atomic_write_strict_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, strict_full_json_bytes(value))


def publish_bytes_no_clobber(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(os, "O_TMPFILE"):
        raise Amendment06IntegrityError("Linux O_TMPFILE is required for atomic publication")
    descriptor = os.open(path.parent, os.O_WRONLY | os.O_TMPFILE, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        libc = ctypes.CDLL(None, use_errno=True)
        linkat = getattr(libc, "linkat", None)
        if linkat is None:
            raise Amendment06IntegrityError("linkat is required for atomic publication")
        linkat.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_int,
        ]
        linkat.restype = ctypes.c_int
        result = linkat(
            descriptor, b"", -100, os.fsencode(path), 0x1000,
        )
        if result != 0 and ctypes.get_errno() in {errno.ENOENT, errno.EPERM}:
            # Some kernels deny AT_EMPTY_PATH without CAP_DAC_READ_SEARCH.  The
            # documented /proc/self/fd form links the same anonymous inode.
            result = linkat(
                -100, os.fsencode(f"/proc/self/fd/{descriptor}"),
                -100, os.fsencode(path), 0x400,
            )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(f"No-clobber file publication collision: {path}")
            raise OSError(error, os.strerror(error), str(path))
        _fsync_parent(path)
    finally:
        os.close(descriptor)


def publish_strict_json_no_clobber(path: Path, value: Any) -> None:
    publish_bytes_no_clobber(path, strict_full_json_bytes(value))


def _rename_directory_no_clobber(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise Amendment06IntegrityError("renameat2 is required for no-clobber Run publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(source), -100, os.fsencode(destination), 1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"No-clobber directory publication collision: {destination}")
        raise OSError(error, os.strerror(error), str(destination))
    _fsync_parent(destination)


def _safe_regular(path: Path, expected_sha256: str | None = None) -> bool:
    return bool(
        path.is_file() and not path.is_symlink()
        and (expected_sha256 is None or sha256_file(path) == expected_sha256)
    )


def _load_concatenated_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder(parse_constant=_reject_json_constant)
    values: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        value, index = decoder.raw_decode(text, index)
        if not isinstance(value, dict):
            raise Amendment06IntegrityError("Authorized-change ledger entry is not an object")
        values.append(value)
    return values


def live_code_test_mapping(study_root: Path = STUDY_ROOT) -> dict[str, dict[str, Any]]:
    paths = sorted([
        *[path for path in (study_root / "code").glob("*.py") if path.is_file()],
        *[path for path in (study_root / "code").glob("*.sh") if path.is_file()],
        *[path for path in (study_root / "tests").glob("*.py") if path.is_file()],
    ], key=lambda path: path.relative_to(study_root).as_posix())
    if any(path.is_symlink() for path in paths):
        raise Amendment06IntegrityError("Live code/test tree contains a symlink")
    return {
        path.relative_to(study_root).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    }


def canonical_code_tree_sha256(mapping: Mapping[str, Mapping[str, Any]]) -> str:
    data = "".join(
        f"{mapping[relative]['sha256']}\t{int(mapping[relative]['size_bytes'])}\t{relative}\n"
        for relative in sorted(mapping)
    ).encode("utf-8")
    return sha256_bytes(data)


def _validated_tree_mapping(value: Any, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise Amendment06IntegrityError(f"{label} is not a nonempty tree map")
    result: dict[str, dict[str, Any]] = {}
    for relative, item in value.items():
        relative = str(relative)
        path = Path(relative)
        if (
            not relative or path.is_absolute() or ".." in path.parts
            or path.as_posix() != relative or not isinstance(item, Mapping)
        ):
            raise Amendment06IntegrityError(f"{label} has an unsafe tree record: {relative!r}")
        digest = str(item.get("sha256", ""))
        size = item.get("size_bytes")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise Amendment06IntegrityError(f"{label} has an invalid tree record: {relative}")
        result[relative] = {"size_bytes": size, "sha256": digest}
    return dict(sorted(result.items()))


def _baseline_tree_mapping() -> dict[str, dict[str, Any]]:
    if not _safe_regular(BASELINE_TREE_TSV):
        raise Amendment06IntegrityError("Canonical Amendment 06 baseline tree TSV is absent")
    frame = pd.read_csv(BASELINE_TREE_TSV, sep="\t", dtype=str, keep_default_na=False)
    if list(frame.columns) != ["sha256", "size_bytes", "relative_path"] or frame.relative_path.duplicated().any():
        raise Amendment06IntegrityError("Canonical Amendment 06 baseline tree TSV schema differs")
    mapping = {
        str(row.relative_path): {
            "size_bytes": int(row.size_bytes), "sha256": str(row.sha256),
        }
        for row in frame.itertuples(index=False)
    }
    validated = _validated_tree_mapping(mapping, label="baseline tree")
    if canonical_code_tree_sha256(validated) != "55992443799c7623911b2a908e79ff9c12e25e7ac9da15c2215ee1b01eac5075":
        raise Amendment06IntegrityError("Canonical Amendment 06 baseline tree hash differs")
    return validated


def _tree_summary(mapping: Mapping[str, Mapping[str, Any]]) -> tuple[str, int, int]:
    return (
        canonical_code_tree_sha256(mapping), len(mapping),
        sum(int(item["size_bytes"]) for item in mapping.values()),
    )


def _changed_tree_records(
    before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in sorted(set(before) | set(after)):
        old = before.get(relative)
        new = after.get(relative)
        if old == new:
            continue
        records.append({
            "relative_path": relative,
            "before_size_bytes": None if old is None else int(old["size_bytes"]),
            "before_sha256": None if old is None else str(old["sha256"]),
            "after_size_bytes": None if new is None else int(new["size_bytes"]),
            "after_sha256": None if new is None else str(new["sha256"]),
        })
    return records


def _validate_test_evidence_records(
    value: Any, *, label: str, require_live: bool = True,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise Amendment06IntegrityError(f"{label} test evidence is empty")
    records: list[dict[str, Any]] = []
    required = {
        "category", "status", "command", "exit_code", "log_path",
        "log_size_bytes", "log_sha256",
    }
    seen_categories: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or not required.issubset(item):
            raise Amendment06IntegrityError(f"{label} test evidence schema differs")
        category = item.get("category")
        log_path = item.get("log_path")
        path = Path(log_path) if isinstance(log_path, str) else Path()
        try:
            inside_runtime = (
                path.is_absolute()
                and A06_RUNTIME.resolve() in path.resolve().parents
            )
        except FileNotFoundError:
            inside_runtime = False
        sha256 = str(item.get("log_sha256", ""))
        size_bytes = item.get("log_size_bytes")
        record_valid = (
            item.get("status") == "PASS"
            and isinstance(item.get("exit_code"), int)
            and not isinstance(item.get("exit_code"), bool)
            and item.get("exit_code") == 0
            and isinstance(category, str) and category
            and category not in seen_categories
            and isinstance(item.get("command"), str) and item["command"]
            and inside_runtime
            and isinstance(size_bytes, int) and not isinstance(size_bytes, bool)
            and size_bytes >= 0
            and re.fullmatch(r"[0-9a-f]{64}", sha256) is not None
            and ("facts" not in item or isinstance(item.get("facts"), Mapping))
        )
        live_valid = (
            not require_live
            or (
                _safe_regular(path, sha256)
                and path.stat().st_size == size_bytes
            )
        )
        if not (record_valid and live_valid):
            raise Amendment06IntegrityError(f"{label} test log binding differs: {path}")
        seen_categories.add(category)
        records.append(dict(item))
    return records


def make_test_evidence_record(
    *, category: str, command: str, log_path: Path,
    facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(log_path).resolve()
    if not _safe_regular(path):
        raise Amendment06IntegrityError(f"Test-evidence log is absent/unsafe: {path}")
    record: dict[str, Any] = {
        "category": category, "status": "PASS", "command": command,
        "exit_code": 0, "log_path": str(path),
        "log_size_bytes": path.stat().st_size, "log_sha256": sha256_file(path),
    }
    if facts is not None:
        record["facts"] = dict(facts)
    _validate_test_evidence_records([record], label=f"generated {category}")
    return record


def build_authorized_edit_event(
    *, purpose: str, test_evidence: Sequence[Mapping[str, Any]],
    study_root: Path = STUDY_ROOT, ledger_path: Path = AUTHORIZED_CHANGE_LEDGER,
    final_byte_freeze: bool = True, recorded_at: str | None = None,
) -> dict[str, Any]:
    events = _load_concatenated_json(ledger_path)
    if len(events) < 2:
        raise Amendment06IntegrityError("Cannot extend an incomplete Amendment 06 ledger")
    for expected_index, event in enumerate(events):
        if event.get("event_index") != expected_index or (
            expected_index and event.get("previous_event_index") != expected_index - 1
        ):
            raise Amendment06IntegrityError("Cannot extend a noncontiguous Amendment 06 ledger")
    before = (
        _baseline_tree_mapping()
        if len(events) == 2
        else _validated_tree_mapping(events[-1].get("after_tree"), label="latest ledger after-tree")
    )
    after = live_code_test_mapping(study_root)
    changed = _changed_tree_records(before, after)
    if not changed:
        raise Amendment06IntegrityError("Final authorized edit event contains no changed path")
    evidence = [dict(item) for item in test_evidence]
    _validate_test_evidence_records(evidence, label="generated authorized edit")
    before_sha, before_count, before_bytes = _tree_summary(before)
    after_sha, after_count, after_bytes = _tree_summary(after)
    return {
        "event": "AUTHORIZED_EDIT_SET_COMMITTED",
        "event_index": len(events),
        "previous_event_index": len(events) - 1,
        "previous_event_sha256": sha256_bytes(
            strict_full_canonical_json_bytes(events[-1])
        ),
        "authorization": AUTHORIZATION, "prompt_sha256": PROMPT_SHA256,
        "recorded_at": recorded_at or _now(), "purpose": purpose,
        "scope": [item["relative_path"] for item in changed],
        "before_tree": before, "after_tree": after,
        "before_tree_sha256": before_sha, "after_tree_sha256": after_sha,
        "before_file_count": before_count, "after_file_count": after_count,
        "before_total_bytes": before_bytes, "after_total_bytes": after_bytes,
        "changed_paths": changed,
        "diff_sha256": sha256_bytes(strict_full_canonical_json_bytes(changed)),
        "test_evidence": evidence, "final_byte_freeze": final_byte_freeze,
    }


def append_authorized_edit_event(
    *, purpose: str, test_evidence: Sequence[Mapping[str, Any]],
    study_root: Path = STUDY_ROOT, ledger_path: Path = AUTHORIZED_CHANGE_LEDGER,
    final_byte_freeze: bool = True, recorded_at: str | None = None,
) -> dict[str, Any]:
    ledger_path = Path(ledger_path)
    if not _safe_regular(ledger_path):
        raise Amendment06IntegrityError("Authorized-change ledger is absent/unsafe")
    with ledger_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        event = build_authorized_edit_event(
            purpose=purpose, test_evidence=test_evidence,
            study_root=study_root, ledger_path=ledger_path,
            final_byte_freeze=final_byte_freeze, recorded_at=recorded_at,
        )
        handle.seek(0, os.SEEK_END)
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            trailing_newline = handle.read(1) == b"\n"
            handle.seek(0, os.SEEK_END)
            if not trailing_newline:
                handle.write(b"\n")
        handle.write(strict_full_canonical_json_bytes(event) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return event


def validate_authorized_change_chain(
    *, study_root: Path = STUDY_ROOT, ledger_path: Path = AUTHORIZED_CHANGE_LEDGER,
) -> dict[str, Any]:
    if not _safe_regular(ledger_path):
        raise Amendment06IntegrityError("Amendment 06 authorized-change ledger is absent")
    events = _load_concatenated_json(ledger_path)
    if len(events) < 2:
        raise Amendment06IntegrityError("Authorized-change ledger has no committed genesis")
    genesis, correction = events[:2]
    if not (
        genesis.get("event_index") == 0
        and genesis.get("event") == "PRE_EDIT_BASELINE_COMMITTED"
        and genesis.get("authorization") == AUTHORIZATION
        and genesis.get("prompt_sha256") == PROMPT_SHA256
        and correction.get("event_index") == 1
        and correction.get("previous_event_index") == 0
        and correction.get("event") == "PRE_EDIT_BASELINE_CANONICAL_ORDER_COMMITTED"
        and correction.get("canonical_tree_sha256")
        == "55992443799c7623911b2a908e79ff9c12e25e7ac9da15c2215ee1b01eac5075"
    ):
        raise Amendment06IntegrityError("Authorized-change ledger genesis differs from committed admission")
    for expected_index, event in enumerate(events):
        if event.get("event_index") != expected_index:
            raise Amendment06IntegrityError("Authorized-change ledger indices are not contiguous")
        if expected_index and event.get("previous_event_index") != expected_index - 1:
            raise Amendment06IntegrityError("Authorized-change ledger chain pointer is invalid")

    previous_tree = _baseline_tree_mapping()
    if len(events) < 3:
        raise Amendment06IntegrityError("Authorized-change ledger has no final committed edit event")
    for expected_index, event in enumerate(events[2:], start=2):
        before = _validated_tree_mapping(event.get("before_tree"), label=f"ledger event {expected_index} before")
        after = _validated_tree_mapping(event.get("after_tree"), label=f"ledger event {expected_index} after")
        before_sha, before_count, before_bytes = _tree_summary(before)
        after_sha, after_count, after_bytes = _tree_summary(after)
        changed = _changed_tree_records(before, after)
        if not changed:
            raise Amendment06IntegrityError(f"Ledger edit event {expected_index} contains no change")
        previous_event_sha = sha256_bytes(strict_full_canonical_json_bytes(events[expected_index - 1]))
        scope = [str(value) for value in event.get("scope", [])] if isinstance(event.get("scope"), list) else []
        if not (
            event.get("event") == "AUTHORIZED_EDIT_SET_COMMITTED"
            and event.get("authorization") == AUTHORIZATION
            and event.get("prompt_sha256") == PROMPT_SHA256
            and event.get("previous_event_sha256") == previous_event_sha
            and isinstance(event.get("purpose"), str) and event["purpose"].strip()
            and before == previous_tree
            and event.get("before_tree_sha256") == before_sha
            and event.get("after_tree_sha256") == after_sha
            and event.get("before_file_count") == before_count
            and event.get("after_file_count") == after_count
            and event.get("before_total_bytes") == before_bytes
            and event.get("after_total_bytes") == after_bytes
            and event.get("changed_paths") == changed
            and scope == [item["relative_path"] for item in changed]
            and event.get("diff_sha256")
            == sha256_bytes(strict_full_canonical_json_bytes(changed))
            and isinstance(event.get("final_byte_freeze"), bool)
        ):
            raise Amendment06IntegrityError(f"Authorized-change ledger event {expected_index} is incomplete")
        _validate_test_evidence_records(
            event.get("test_evidence"),
            label=f"ledger event {expected_index}",
            require_live=expected_index == len(events) - 1,
        )
        previous_tree = after
    if events[-1].get("final_byte_freeze") is not True:
        raise Amendment06IntegrityError("Latest authorized-change event is not the final byte freeze")

    mapping = live_code_test_mapping(study_root)
    live_tree_sha256 = canonical_code_tree_sha256(mapping)
    total_bytes = sum(int(item["size_bytes"]) for item in mapping.values())
    latest = events[-1]
    committed_sha = str(
        latest.get("post_edit_tree_sha256")
        or latest.get("canonical_tree_sha256")
        or latest.get("after_tree_sha256")
        or ""
    )
    if committed_sha != live_tree_sha256:
        raise Amendment06IntegrityError(
            "Live code/tests do not match the latest committed Amendment 06 tree: "
            f"committed={committed_sha!r}, live={live_tree_sha256}"
        )
    if previous_tree != mapping:
        raise Amendment06IntegrityError("Latest ledger after-tree differs from live code/tests")
    allowed_after_hashes = {path: item["sha256"] for path, item in mapping.items()}
    ledger_evidence = {
        "status": "PASS",
        "authorization": AUTHORIZATION,
        "chain_status": "PASS",
        "baseline_tree_sha256": "55992443799c7623911b2a908e79ff9c12e25e7ac9da15c2215ee1b01eac5075",
        "latest_committed_tree_sha256": live_tree_sha256,
        "latest_committed_file_count": len(mapping),
        "latest_committed_total_bytes": total_bytes,
        "allowed_after_hashes_sha256": sha256_bytes(
            core.canonical_json(dict(sorted(allowed_after_hashes.items())))
        ),
        "authorized_change_ledger_path": str(Path(ledger_path).resolve()),
        "authorized_change_ledger_sha256": sha256_file(ledger_path),
    }
    return {
        "status": "PASS", "event_count": len(events),
        "latest_event_index": len(events) - 1,
        "canonical_tree_sha256": live_tree_sha256,
        "mapping": mapping,
        "allowed_after_hashes": allowed_after_hashes,
        "change_ledger_evidence": ledger_evidence,
        "ledger_sha256": sha256_file(ledger_path),
        "latest_event_sha256": sha256_bytes(strict_full_canonical_json_bytes(latest)),
    }


PRERUN_CHECK_CATEGORIES = (
    "syntax_compile",
    "targeted_tests",
    "full_suite_exactly_once",
    "active_cuda_probe",
    "package_synthetic_rehearsal",
    "resume_rehearsal",
)


def _validated_final_prerun_checks(value: Any) -> list[dict[str, Any]]:
    checks = _validate_test_evidence_records(value, label="final pre-Run")
    if tuple(str(item["category"]) for item in checks) != PRERUN_CHECK_CATEGORIES:
        raise Amendment06IntegrityError("Final pre-Run check category/order differs")
    facts = {str(item["category"]): item.get("facts") for item in checks}
    if not all(isinstance(item, Mapping) for item in facts.values()) or not (
        facts["syntax_compile"].get("all_changed_python_files_compiled") is True
        and facts["targeted_tests"].get("amendment06_targeted_tests_passed") is True
        and facts["full_suite_exactly_once"].get("full_project_suite_passed") is True
        and facts["full_suite_exactly_once"].get("suite_invocation_count") == 1
        and facts["active_cuda_probe"].get("xgboost_version") == "2.1.1"
        and facts["active_cuda_probe"].get("scikit_learn_version") == "1.7.2"
        and facts["active_cuda_probe"].get("use_cuda") is True
        and "4090" in str(facts["active_cuda_probe"].get("gpu_name", ""))
        and facts["active_cuda_probe"].get("cpu_fallback_detected") is False
        and facts["active_cuda_probe"].get("fit_save_reload_predict_status") == "PASS"
        and facts["package_synthetic_rehearsal"].get("synthetic_full_fixture") is True
        and facts["package_synthetic_rehearsal"].get("production_packager_used") is True
        and facts["package_synthetic_rehearsal"].get("production_final_validator_used") is True
        and facts["package_synthetic_rehearsal"].get("package_retry_without_training") is True
        and facts["package_synthetic_rehearsal"].get("scientific_api_calls") == 0
        and facts["resume_rehearsal"].get("same_run_id_preserved") is True
        and facts["resume_rehearsal"].get("completed_variant_skipped") is True
        and facts["resume_rehearsal"].get("unpublished_incomplete_variant_retried") is True
        and facts["resume_rehearsal"].get("completed_variant_retrained") is False
        and facts["resume_rehearsal"].get("distinct_full_run_ids") == 1
    ):
        raise Amendment06IntegrityError("A mandatory final pre-Run rehearsal fact failed")
    return checks


def build_final_prerun_evidence(
    *, chain: Mapping[str, Any], checks: Sequence[Mapping[str, Any]],
    generated_at: str | None = None,
) -> dict[str, Any]:
    checked = _validated_final_prerun_checks([dict(item) for item in checks])
    mapping = chain["mapping"]
    return {
        "status": "PASS", "authorization": AUTHORIZATION,
        "prompt_sha256": PROMPT_SHA256, "generated_at": generated_at or _now(),
        "live_code_tree_sha256": chain["canonical_tree_sha256"],
        "live_code_tree_file_count": len(mapping),
        "live_code_tree_total_bytes": sum(
            int(item["size_bytes"]) for item in mapping.values()
        ),
        "authorized_change_ledger_path": str(AUTHORIZED_CHANGE_LEDGER.resolve()),
        "authorized_change_ledger_sha256": chain["ledger_sha256"],
        "latest_event_index": chain["latest_event_index"],
        "latest_event_sha256": chain["latest_event_sha256"],
        "full_suite_invocation_count": 1, "checks": checked,
    }


def write_final_prerun_evidence(
    *, checks: Sequence[Mapping[str, Any]], path: Path = FINAL_PRERUN_EVIDENCE,
    generated_at: str | None = None,
) -> dict[str, Any]:
    chain = validate_authorized_change_chain()
    payload = build_final_prerun_evidence(
        chain=chain, checks=checks, generated_at=generated_at,
    )
    data = strict_full_json_bytes(payload)
    path = Path(path)
    if path.exists():
        if not _safe_regular(path, sha256_bytes(data)):
            raise Amendment06IntegrityError("Final pre-Run evidence already differs")
    else:
        publish_bytes_no_clobber(path, data)
    return validate_final_prerun_evidence(chain, path=path)


def validate_final_prerun_evidence(
    chain: Mapping[str, Any], *, path: Path = FINAL_PRERUN_EVIDENCE,
) -> dict[str, Any]:
    evidence = strict_full_load_file(path)
    required_top = {
        "status", "authorization", "prompt_sha256", "generated_at",
        "live_code_tree_sha256", "live_code_tree_file_count", "live_code_tree_total_bytes",
        "authorized_change_ledger_path", "authorized_change_ledger_sha256",
        "latest_event_index", "latest_event_sha256", "full_suite_invocation_count",
        "checks",
    }
    mapping = chain["mapping"]
    if not (
        isinstance(evidence, Mapping) and set(evidence) == required_top
        and evidence.get("status") == "PASS"
        and evidence.get("authorization") == AUTHORIZATION
        and evidence.get("prompt_sha256") == PROMPT_SHA256
        and isinstance(evidence.get("generated_at"), str) and evidence["generated_at"]
        and evidence.get("live_code_tree_sha256") == chain["canonical_tree_sha256"]
        and evidence.get("live_code_tree_file_count") == len(mapping)
        and evidence.get("live_code_tree_total_bytes")
        == sum(int(item["size_bytes"]) for item in mapping.values())
        and Path(str(evidence.get("authorized_change_ledger_path"))).resolve()
        == AUTHORIZED_CHANGE_LEDGER.resolve()
        and evidence.get("authorized_change_ledger_sha256") == chain["ledger_sha256"]
        and evidence.get("latest_event_index") == chain["latest_event_index"]
        and evidence.get("latest_event_sha256") == chain["latest_event_sha256"]
        and evidence.get("full_suite_invocation_count") == 1
    ):
        raise Amendment06IntegrityError("Final pre-Run evidence does not bind the frozen code/ledger")
    _validated_final_prerun_checks(evidence.get("checks"))
    return {
        "status": "PASS", "path": str(path), "sha256": sha256_file(path),
        "checks": list(PRERUN_CHECK_CATEGORIES), "full_suite_invocation_count": 1,
    }


def _resolved_equal(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=True) == right.resolve(strict=True)
    except FileNotFoundError:
        return False


def assert_exact_invocation(
    *, preflight_run: Path, smoke_run: Path,
    expected_runtime_root: Path = EXPECTED_RUNTIME_ROOT,
) -> dict[str, Any]:
    runtime_environment = os.environ.get("ABLATION_RUNTIME_ROOT", "")
    if not _resolved_equal(Path(preflight_run), ACCEPTED_PREFLIGHT):
        raise Amendment06IntegrityError("Amendment 06 requires the exact accepted Preflight")
    if not _resolved_equal(Path(smoke_run), ACCEPTED_SMOKE):
        raise Amendment06IntegrityError("Amendment 06 requires the exact accepted Smoke")
    if Path(sys.executable).resolve() != CUDA_PYTHON.resolve() or Path(sys.prefix).resolve() != CUDA_ENV.resolve():
        raise Amendment06IntegrityError(f"Amendment 06 requires interpreter {CUDA_PYTHON}")
    if not runtime_environment or Path(runtime_environment).resolve() != expected_runtime_root.resolve():
        raise Amendment06IntegrityError(
            f"ABLATION_RUNTIME_ROOT must be exactly {expected_runtime_root}"
        )
    if not expected_runtime_root.is_dir() or expected_runtime_root.is_symlink():
        raise Amendment06IntegrityError("Amendment 06 runtime root is absent or unsafe")
    build = xgb.build_info()
    if xgb.__version__ != "2.1.1" or sklearn.__version__ != "1.7.2" or build.get("USE_CUDA") is not True:
        raise Amendment06IntegrityError("Pinned XGBoost/scikit-learn/CUDA environment differs")
    return {
        "status": "PASS", "cuda_env": str(CUDA_ENV),
        "python": str(Path(sys.executable).resolve()),
        "runtime_root": str(expected_runtime_root.resolve()),
        "xgboost_version": xgb.__version__, "scikit_learn_version": sklearn.__version__,
        "xgboost_use_cuda": True,
    }


def _smoke_identity_is_ready_for_full(identity: Mapping[str, Any]) -> bool:
    semantics = (
        identity.get("reference_validation", {})
        .get("smoke_semantic_completeness", {})
    )
    return bool(
        identity.get("run_id") == ACCEPTED_SMOKE.name
        and semantics.get("ready_for_full_awaiting_external_review") is True
        and semantics.get("status") == "PASS"
        and identity.get("full_authorized") is False
    )


def verify_reference_admission(
    *, preflight_run: Path, smoke_run: Path,
) -> dict[str, Any]:
    for path, expected in PINNED_FILES.items():
        if not _safe_regular(path, expected):
            raise Amendment06IntegrityError(f"Pinned Amendment 06 reference drifted: {path}")
    admission = strict_full_load_file(PRE_EDIT_ADMISSION)
    if not (
        admission.get("authorization") == AUTHORIZATION
        and admission.get("status") == "PASS"
        and admission.get("blockers") == []
        and admission.get("preflight_reference_validation", {}).get("status")
        == "PASS_SEALED_ONLY_NO_LIVE_AUDIT_PARSE"
        and admission.get("full_run_guard", {}).get("status") == "PASS"
        and admission.get("a05_recovery", {}).get("status") == "PASS"
    ):
        raise Amendment06IntegrityError("Committed pre-edit admission is not a clean PASS")
    chain = validate_authorized_change_chain()
    prerun = validate_final_prerun_evidence(chain)
    signature = inspect.signature(core.validate_amended_run_reference)
    if not {
        "amendment06_allowed_after_hashes", "amendment06_change_ledger_evidence",
    }.issubset(signature.parameters):
        raise Amendment06IntegrityError("Production Smoke validator lacks Amendment 06 drift routing")
    smoke_identity = core.validate_amended_run_reference(
        smoke_run,
        expected_kind="CUDA_SMOKE_NON_SCIENTIFIC",
        allowed_states={"SMOKE_COMPLETE"},
        amendment06_allowed_after_hashes=chain["allowed_after_hashes"],
        amendment06_change_ledger_evidence=chain["change_ledger_evidence"],
    )
    if not _smoke_identity_is_ready_for_full(smoke_identity):
        raise Amendment06IntegrityError("Accepted Smoke is not the sealed Full-admission reference")
    preflight_identity = strict_full_load_file(preflight_run / "config/run_identity.lock.json")
    if not (
        preflight_identity.get("run_kind") == "AMENDED_PREFLIGHT"
        and preflight_identity.get("state") == "PREFLIGHT_COMPLETE"
        and preflight_identity.get("full_authorized") is False
        and preflight_identity.get("candidate_split_hash") == SPLIT_ASSIGNMENT_SHA256
    ):
        raise Amendment06IntegrityError("Accepted Preflight identity differs from sealed admission")
    return {
        "status": "PASS", "preflight": "PASS_SEALED_ONLY_NO_LIVE_AUDIT_PARSE",
        "smoke": "PASS_PRODUCTION_VALIDATOR", "authorized_change_chain": chain,
        "final_prerun_evidence": prerun,
        "pre_edit_admission_sha256": sha256_file(PRE_EDIT_ADMISSION),
    }


@contextmanager
def amendment06_lock(runtime_root: Path = A06_RUNTIME) -> Iterator[None]:
    runtime_root.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_root / "amendment06_full.lock"
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Amendment06ResumeError("Another Amendment 06 process holds the execution lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def existing_full_runs(results_root: Path = RESULTS_ROOT) -> list[Path]:
    matches = sorted(results_root.glob(f"run_*{RUN_SUFFIX}"))
    runs: list[Path] = []
    for path in matches:
        mode = os.lstat(path).st_mode
        if not stat.S_ISDIR(mode):
            raise Amendment06IntegrityError(
                f"Unsafe one-Run guard entry occupies an Amendment 06 Full name: {path}"
            )
        runs.append(path)
    return runs


def _manifest_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    frame = pd.DataFrame(rows, columns=["relative_path", "size_bytes", "sha256", "role"])
    return frame.to_csv(index=False, sep="\t", lineterminator="\n").encode("utf-8")


def _code_manifest(
    relatives: Sequence[str], *, role: str,
) -> tuple[bytes, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for relative in relatives:
        path = STUDY_ROOT / relative
        if not _safe_regular(path):
            raise Amendment06IntegrityError(f"Code-manifest input is absent/unsafe: {relative}")
        digest = sha256_file(path)
        hashes[relative] = digest
        rows.append({
            "relative_path": relative, "size_bytes": path.stat().st_size,
            "sha256": digest,
            "role": role,
        })
    return _manifest_bytes(rows), hashes


def validate_frozen_code_manifests(run_dir: Path) -> dict[str, Any]:
    identity = strict_full_load_file(run_dir / "config/run_identity.lock.json")
    scientific_bytes, scientific_hashes = _code_manifest(
        (*SCIENTIFIC_CODE_RELATIVES, *SCIENTIFIC_TEST_RELATIVES),
        role="SCIENTIFIC_EXECUTION_OR_TEST",
    )
    scientific_path = run_dir / "provenance/scientific_execution_code_manifest.tsv"
    package_path = run_dir / "provenance/package_validator_code_manifest_initial.tsv"
    if not _safe_regular(package_path):
        raise Amendment06IntegrityError("Initial package-validator manifest is absent/unsafe")
    package_bytes = package_path.read_bytes()
    try:
        package_rows = list(csv.DictReader(
            io.StringIO(package_bytes.decode("utf-8")), delimiter="\t",
        ))
    except Exception as exc:
        raise Amendment06IntegrityError("Initial package-validator manifest is unreadable") from exc
    if (
        not package_rows
        or tuple(package_rows[0]) != ("relative_path", "size_bytes", "sha256", "role")
        or [str(row.get("relative_path")) for row in package_rows]
        != list(PACKAGE_CODE_RELATIVES)
    ):
        raise Amendment06IntegrityError("Initial package-validator manifest schema/order differs")
    package_hashes: dict[str, str] = {}
    for row in package_rows:
        relative = str(row.get("relative_path"))
        digest = str(row.get("sha256", ""))
        size = str(row.get("size_bytes", ""))
        if (
            row.get("role") != "PACKAGE_VALIDATOR_OR_TEST"
            or not re.fullmatch(r"[0-9]+", size)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise Amendment06IntegrityError(
                f"Initial package-validator manifest row differs: {relative}"
            )
        package_hashes[relative] = digest
    if not (
        _safe_regular(scientific_path, sha256_bytes(scientific_bytes))
        and identity.get("scientific_execution_code_manifest_sha256") == sha256_bytes(scientific_bytes)
        and identity.get("package_validator_code_manifest_initial_sha256") == sha256_bytes(package_bytes)
        and set(scientific_hashes).isdisjoint(package_hashes)
    ):
        raise Amendment06IntegrityError("Frozen scientific/package code manifests drifted")
    for relative, digest in scientific_hashes.items():
        snapshot_root = "tests_snapshot" if relative.startswith("tests/") else "code_snapshot"
        snapshot = run_dir / "provenance" / snapshot_root / Path(relative).name
        if not _safe_regular(snapshot, digest):
            raise Amendment06IntegrityError(
                f"Frozen scientific code/test snapshot drifted: {relative}"
            )
    return {
        "status": "PASS", "scientific": scientific_hashes,
        "package": package_hashes,
        "package_live_bytes_may_advance_via_isolated_lineage": True,
        "scientific_manifest_sha256": sha256_bytes(scientific_bytes),
        "package_manifest_sha256": sha256_bytes(package_bytes),
    }


@dataclass(frozen=True)
class LockedInputs:
    training_configuration: dict[str, Any]
    variant_features: dict[str, Any]
    full_features: tuple[str, ...]
    split: pd.DataFrame
    model_parameters: dict[str, Any]


def load_locked_inputs(preflight_run: Path = ACCEPTED_PREFLIGHT) -> LockedInputs:
    pins = {
        "config/training_configuration.lock.json": TRAINING_CONFIGURATION_SHA256,
        "config/variant_feature_sets.json": VARIANT_FEATURE_SETS_SHA256,
        "config/feature_manifest.csv": FEATURE_MANIFEST_SHA256,
        "config/feature_dependency_graph.json": FEATURE_DEPENDENCY_GRAPH_SHA256,
        "splits/row_split_manifest.csv.gz": ROW_SPLIT_MANIFEST_SHA256,
    }
    for relative, digest in pins.items():
        if not _safe_regular(preflight_run / relative, digest):
            raise Amendment06IntegrityError(f"Locked configuration drifted: {relative}")
    configuration = strict_full_load_file(preflight_run / "config/training_configuration.lock.json")
    variants = strict_full_load_file(preflight_run / "config/variant_feature_sets.json")
    split = pd.read_csv(preflight_run / "splits/row_split_manifest.csv.gz")
    required_split = [
        "stable_candidate_id", "file_name", "ann_id", "scale", "card_id",
        "label", "split", "eligible", "split_order",
    ]
    if list(split.columns) != required_split or split.stable_candidate_id.astype(str).duplicated().any():
        raise Amendment06IntegrityError("Locked Split manifest schema/identity is invalid")
    observed_split_hash = core.split_assignment_hash(
        split.stable_candidate_id, split.split, split.label, split.eligible, split.split_order,
    )
    if observed_split_hash != SPLIT_ASSIGNMENT_SHA256:
        raise Amendment06IntegrityError("Locked Split assignment hash differs")
    counts = {
        value: (
            int((split.split == value).sum()),
            int(split.loc[split.split == value, "label"].sum()),
            int(split.loc[split.split == value, "card_id"].nunique()),
        )
        for value in ("training", "validation", "historical_heldout_ineligible_border")
    }
    if counts != {
        "training": (291024, 75036, 66),
        "validation": (56843, 15952, 18),
        "historical_heldout_ineligible_border": (15696, 3080, 18),
    }:
        raise Amendment06IntegrityError(f"Locked Split counts differ: {counts}")
    full_features = tuple(configuration.get("feature_order", []))
    if len(full_features) != 93 or len(set(full_features)) != 93:
        raise Amendment06IntegrityError("Locked Full feature order is not unique 93D")
    for variant, (count, digest) in VARIANT_FEATURE_CONTRACT.items():
        item = variants.get(variant, {})
        if not (
            item.get("status") == "RUNNABLE"
            and item.get("feature_count") == count
            and item.get("feature_list_sha256") == digest
            and tuple(item.get("retained_features", []))
            == tuple(feature for feature in full_features if feature in set(item.get("retained_features", [])))
        ):
            raise Amendment06IntegrityError(f"Locked feature contract differs: {variant}")
    if variants.get(BLOCKED_VARIANT, {}).get("status") != "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE":
        raise Amendment06IntegrityError("True no-PCA must remain blocked")
    parameters = dict(configuration.get("model", {}).get("full_classifier_parameters", {}))
    if (
        configuration.get("model", {}).get("full_classifier_parameters_sha256")
        != FULL_MODEL_PARAMETERS_SHA256
        or sha256_bytes(core.canonical_json(parameters)) != FULL_MODEL_PARAMETERS_SHA256
    ):
        raise Amendment06IntegrityError("Locked Full XGBoost parameters differ")
    return LockedInputs(configuration, variants, full_features, split, parameters)


def _snapshot_file(source: Path, destination: Path) -> None:
    if not _safe_regular(source):
        raise Amendment06IntegrityError(f"Snapshot source is absent/unsafe: {source}")
    publish_bytes_no_clobber(destination, source.read_bytes())


def _initial_status(run_id: str) -> str:
    return (
        f"RUN_ID={run_id}\nRUN_KIND={RUN_KIND}\nRUN_STATE=FULL_IN_PROGRESS\n"
        f"FULL_AUTHORIZATION={AUTHORIZATION}\nFULL_MODELS_FROZEN=NO\n"
        "EXTERNAL_AUDIT_OPENED=NO\nFULL_TRAINING_EXECUTED=YES_IN_PROGRESS\n"
    )


def _run_intent_path() -> Path:
    return A06_RUNTIME / "authorization/full_run_intent.json"


def _run_receipt_path() -> Path:
    return A06_RUNTIME / "authorization/full_run_receipt.json"


def _new_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    binding = sha256_bytes(strict_full_canonical_json_bytes({
        "authorization": AUTHORIZATION, "preflight": ACCEPTED_PREFLIGHT.name,
        "smoke": ACCEPTED_SMOKE.name, "split": SPLIT_ASSIGNMENT_SHA256,
    }))[:8]
    return f"run_{timestamp}_{binding}{RUN_SUFFIX}"


def _reserve_full_run_id() -> dict[str, Any]:
    """Reserve the sole Run ID before any Results-directory side effect."""
    intent_path = _run_intent_path()
    if intent_path.exists():
        intent = strict_full_load_file(intent_path)
    else:
        run_id = _new_run_id()
        intent = {
            "authorization": AUTHORIZATION,
            "authorization_reserved": True,
            "run_id": run_id,
            "run_path": str(RESULTS_ROOT / run_id),
            "preflight_run_id": ACCEPTED_PREFLIGHT.name,
            "smoke_run_id": ACCEPTED_SMOKE.name,
            "candidate_split_hash": SPLIT_ASSIGNMENT_SHA256,
            "reserved_at": _now(),
        }
        publish_strict_json_no_clobber(intent_path, intent)
    run_id = str(intent.get("run_id", ""))
    if not (
        intent.get("authorization") == AUTHORIZATION
        and intent.get("authorization_reserved") is True
        and run_id.endswith(RUN_SUFFIX)
        and intent.get("run_path") == str(RESULTS_ROOT / run_id)
        and intent.get("preflight_run_id") == ACCEPTED_PREFLIGHT.name
        and intent.get("smoke_run_id") == ACCEPTED_SMOKE.name
        and intent.get("candidate_split_hash") == SPLIT_ASSIGNMENT_SHA256
    ):
        raise Amendment06IntegrityError("Full Run authorization intent is invalid")
    return intent


def _publish_or_validate_run_receipt(run_dir: Path) -> dict[str, Any]:
    identity_sha256 = sha256_file(run_dir / "config/run_identity.lock.json")
    receipt_path = _run_receipt_path()
    if receipt_path.exists():
        receipt = strict_full_load_file(receipt_path)
    else:
        receipt = {
            "authorization": AUTHORIZATION,
            "authorization_consumed": True,
            "run_id": run_dir.name,
            "run_path": str(run_dir),
            "run_identity_sha256": identity_sha256,
            "consumed_at": _now(),
        }
        publish_strict_json_no_clobber(receipt_path, receipt)
    if not (
        receipt.get("authorization") == AUTHORIZATION
        and receipt.get("authorization_consumed") is True
        and receipt.get("run_id") == run_dir.name
        and receipt.get("run_path") == str(run_dir)
        and receipt.get("run_identity_sha256") == identity_sha256
    ):
        raise Amendment06IntegrityError("Full Run authorization receipt disagrees")
    return receipt


def _state_marker_path(run_dir: Path, phase: str) -> Path:
    if phase not in RUN_PHASES:
        raise Amendment06IntegrityError(f"Unknown Full state: {phase}")
    return run_dir / f"provenance/state/{RUN_PHASES.index(phase):02d}_{phase}.json"


def _current_run_state(run_dir: Path) -> str:
    identity = strict_full_load_file(run_dir / "config/run_identity.lock.json")
    if identity.get("state") != "FULL_IN_PROGRESS":
        raise Amendment06IntegrityError("Immutable Run identity initial state changed")
    observed: list[str] = []
    previous_sha256: str | None = None
    for phase in RUN_PHASES:
        path = _state_marker_path(run_dir, phase)
        if path.exists():
            marker = strict_full_load_file(path)
            if not (
                marker.get("run_id") == run_dir.name
                and marker.get("run_kind") == RUN_KIND
                and marker.get("phase") == phase
                and marker.get("phase_index") == RUN_PHASES.index(phase)
                and marker.get("status") == "PASS"
                and marker.get("previous_marker_sha256") == previous_sha256
            ):
                raise Amendment06IntegrityError(f"Invalid Full state marker: {phase}")
            observed.append(phase)
            previous_sha256 = sha256_file(path)
        elif observed:
            # A later marker after this gap is checked below.
            if any(_state_marker_path(run_dir, later).exists() for later in RUN_PHASES[RUN_PHASES.index(phase) + 1:]):
                raise Amendment06IntegrityError("Full state marker chain has a gap")
            break
    if not observed or observed[0] != "FULL_IN_PROGRESS":
        raise Amendment06IntegrityError("Initial Full state marker is absent")
    return observed[-1]


def _create_authorized_run(
    *, command_line: str, admission: Mapping[str, Any], locked: LockedInputs,
) -> Path:
    intent = _reserve_full_run_id()
    run_id = str(intent["run_id"])
    temporary = EXPECTED_RUNTIME_ROOT / "run_creation" / run_id
    run_dir = RESULTS_ROOT / run_id
    runs = existing_full_runs()
    if runs:
        if runs == [run_dir]:
            _validate_run_identity(run_dir)
            return run_dir
        raise Amendment06IntegrityError("The one Amendment 06 Full Run authorization is already consumed")
    if temporary.exists():
        if temporary.is_symlink() or not temporary.is_dir():
            raise Amendment06IntegrityError("Full Run staging path is unsafe")
        quarantine = (
            EXPECTED_RUNTIME_ROOT / "run_creation_quarantine"
            / f"{run_id}_{uuid.uuid4().hex}"
        )
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        _rename_directory_no_clobber(temporary, quarantine)
    scientific_bytes, _ = _code_manifest(
        (*SCIENTIFIC_CODE_RELATIVES, *SCIENTIFIC_TEST_RELATIVES),
        role="SCIENTIFIC_EXECUTION_OR_TEST",
    )
    package_bytes, _ = _code_manifest(
        PACKAGE_CODE_RELATIVES, role="PACKAGE_VALIDATOR_OR_TEST",
    )
    temporary.mkdir(mode=0o700, parents=True)
    try:
        for relative in (
            "config", "splits", "models", "predictions", "metrics", "statistics",
            "tables", "figures", "timing", "provenance", "logs", "docs",
            "provenance/code_snapshot", "provenance/tests_snapshot", "provenance/state",
        ):
            (temporary / relative).mkdir(parents=True, exist_ok=True)
        identity = {
            "run_id": run_id, "run_kind": RUN_KIND, "state": "FULL_IN_PROGRESS",
            "full_authorization": AUTHORIZATION,
            "external_review_decision": EXTERNAL_REVIEW_DECISION,
            "accepted_preflight_path": str(ACCEPTED_PREFLIGHT),
            "accepted_preflight_run_id": ACCEPTED_PREFLIGHT.name,
            "accepted_preflight_review_zip_sha256": PINNED_FILES[ACCEPTED_PREFLIGHT_ZIP],
            "accepted_smoke_path": str(ACCEPTED_SMOKE),
            "accepted_smoke_run_id": ACCEPTED_SMOKE.name,
            "accepted_smoke_review_zip_sha256": PINNED_FILES[ACCEPTED_SMOKE_REVIEW_ZIP],
            "candidate_split_hash": SPLIT_ASSIGNMENT_SHA256,
            "base_seed": BASE_SEED, "primary_model_seed": PRIMARY_MODEL_SEED,
            "decision_threshold": DECISION_THRESHOLD,
            "created_at": _now(), "command_line": command_line,
            "scientific_execution_code_manifest_sha256": sha256_bytes(scientific_bytes),
            "package_validator_code_manifest_initial_sha256": sha256_bytes(package_bytes),
            "scientific_code_frozen": True,
            "new_preflight_created": False, "new_smoke_created": False,
            "stability_authorized": False, "threshold_tuning_authorized": False,
        }
        atomic_write_strict_json(temporary / "config/run_identity.lock.json", identity)
        atomic_write_bytes(temporary / "provenance/scientific_execution_code_manifest.tsv", scientific_bytes)
        atomic_write_bytes(temporary / "provenance/package_validator_code_manifest_initial.tsv", package_bytes)
        for relative in (
            "config/training_configuration.lock.json", "config/variant_feature_sets.json",
            "config/feature_manifest.csv", "config/feature_dependency_graph.json",
            "config/locked_split_identity.json", "splits/row_split_manifest.csv.gz",
        ):
            _snapshot_file(ACCEPTED_PREFLIGHT / relative, temporary / relative)
        _snapshot_file(PROMPT_PATH, temporary / "provenance/amendment06_prompt_snapshot.txt")
        atomic_write_strict_json(
            temporary / "provenance/amendment06_external_review_authorization.json",
            {
                "authorization": AUTHORIZATION, "decision": EXTERNAL_REVIEW_DECISION,
                "distinct_full_run_ids_authorized": 1, "run_id": run_id,
                "same_run_resume_authorized": True, "same_run_package_retry_authorized": True,
                "stability_or_multiseed_authorized": False,
                "threshold_tuning_authorized": False,
            },
        )
        atomic_write_strict_json(
            temporary / "provenance/amendment06_code_lineage.json",
            {
                "status": "PASS", "pre_edit_admission_sha256": sha256_file(PRE_EDIT_ADMISSION),
                "authorized_change_ledger_sha256": admission["authorized_change_chain"]["ledger_sha256"],
                "authorized_change_latest_event": admission["authorized_change_chain"]["latest_event_index"],
                "live_code_tree_sha256": admission["authorized_change_chain"]["canonical_tree_sha256"],
                "final_prerun_evidence_sha256": admission["final_prerun_evidence"]["sha256"],
                "scientific_manifest_sha256": sha256_bytes(scientific_bytes),
                "package_manifest_initial_sha256": sha256_bytes(package_bytes),
            },
        )
        for relative in (*SCIENTIFIC_CODE_RELATIVES, *SCIENTIFIC_TEST_RELATIVES):
            source = STUDY_ROOT / relative
            destination_root = "tests_snapshot" if relative.startswith("tests/") else "code_snapshot"
            _snapshot_file(source, temporary / "provenance" / destination_root / Path(relative).name)
        _snapshot_file(
            STUDY_ROOT / "docs/PROTOCOL_AMENDMENT_06.md",
            temporary / "docs/PROTOCOL_AMENDMENT_06.md",
        )
        publish_strict_json_no_clobber(
            _state_marker_path(temporary, "FULL_IN_PROGRESS"),
            {
                "run_id": run_id, "run_kind": RUN_KIND,
                "phase": "FULL_IN_PROGRESS", "phase_index": 0,
                "status": "PASS", "previous_marker_sha256": None,
                "recorded_at": _now(),
            },
        )
        _rename_directory_no_clobber(temporary, run_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _publish_or_validate_run_receipt(run_dir)
    return run_dir


def _validate_run_identity(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    if run_dir.parent != RESULTS_ROOT.resolve() or not run_dir.name.endswith(RUN_SUFFIX):
        raise Amendment06ResumeError("Resume target is not the authorized Amendment 06 Full Run")
    runs = existing_full_runs()
    if runs != [run_dir]:
        raise Amendment06ResumeError(f"Expected exactly this one Full Run, observed: {runs}")
    identity = strict_full_load_file(run_dir / "config/run_identity.lock.json")
    if not (
        identity.get("run_id") == run_dir.name
        and identity.get("run_kind") == RUN_KIND
        and identity.get("full_authorization") == AUTHORIZATION
        and identity.get("candidate_split_hash") == SPLIT_ASSIGNMENT_SHA256
        and identity.get("accepted_preflight_path") == str(ACCEPTED_PREFLIGHT)
        and identity.get("accepted_smoke_path") == str(ACCEPTED_SMOKE)
        and identity.get("state") == "FULL_IN_PROGRESS"
    ):
        raise Amendment06ResumeError("Full Run identity/authorization is invalid")
    intent = strict_full_load_file(_run_intent_path())
    if intent.get("run_id") != run_dir.name:
        raise Amendment06ResumeError("Full Run authorization intent disagrees")
    _publish_or_validate_run_receipt(run_dir)
    validate_frozen_code_manifests(run_dir)
    _current_run_state(run_dir)
    return identity


def _set_run_state(run_dir: Path, target: str) -> dict[str, Any]:
    current = _current_run_state(run_dir)
    if target not in RUN_PHASES or RUN_PHASES.index(target) < RUN_PHASES.index(current):
        raise Amendment06IntegrityError(f"Invalid Full state transition: {current} -> {target}")
    if target != current:
        if RUN_PHASES.index(target) != RUN_PHASES.index(current) + 1:
            raise Amendment06IntegrityError(f"Full state transition skips a phase: {current} -> {target}")
        publish_strict_json_no_clobber(
            _state_marker_path(run_dir, target),
            {
                "run_id": run_dir.name, "run_kind": RUN_KIND,
                "phase": target, "phase_index": RUN_PHASES.index(target),
                "status": "PASS", "previous_phase": current,
                "previous_marker_sha256": sha256_file(_state_marker_path(run_dir, current)),
                "recorded_at": _now(),
            },
        )
    return {"run_id": run_dir.name, "run_kind": RUN_KIND, "state": target}


@dataclass(frozen=True)
class DevelopmentPopulation:
    training: pd.DataFrame
    validation: pd.DataFrame
    training_labels: np.ndarray
    validation_labels: np.ndarray
    training_weights: np.ndarray
    validation_weights: np.ndarray


@dataclass(frozen=True)
class FrozenPopulationCache:
    root: Path
    manifest: dict[str, Any]
    imputer: dict[str, Any]
    full_features: tuple[str, ...]
    pre_X: np.ndarray
    pre_y: np.ndarray
    pre_weights: np.ndarray
    post_X: np.ndarray
    post_y: np.ndarray
    post_weights: np.ndarray
    validation_X: np.ndarray

    @property
    def binding_sha256(self) -> str:
        return str(self.manifest["frozen_population_sha256"])

    @property
    def imputer_sha256(self) -> str:
        return str(self.manifest["imputer_sha256"])


@dataclass(frozen=True)
class VariantMatrices:
    variant: str
    classification: str
    features: tuple[str, ...]
    indices: tuple[int, ...]
    X: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    validation_X: np.ndarray
    population: str
    population_evidence: dict[str, Any]


def _source_inventory_bytes() -> bytes:
    source = ACCEPTED_PREFLIGHT / "provenance/source_input_hashes_post.tsv"
    if not _safe_regular(source, SOURCE_INVENTORY_SHA256):
        raise Amendment06IntegrityError("Sealed source-input inventory is absent")
    raw = source.read_bytes()
    frame = pd.read_csv(
        io.BytesIO(raw), sep="\t", dtype={"path": str, "sha256": str},
    )
    if list(frame.columns) != ["path", "size_bytes", "sha256"] or len(frame) != 15:
        raise Amendment06IntegrityError("Sealed source-input inventory schema/count differs")
    if frame.path.duplicated().any():
        raise Amendment06IntegrityError("Sealed source-input inventory has duplicate paths")
    for row in frame.itertuples(index=False):
        path = Path(str(row.path))
        if not _safe_regular(path, str(row.sha256)) or path.stat().st_size != int(row.size_bytes):
            raise Amendment06IntegrityError(f"Pinned source input drifted: {path}")
    if not (
        ((frame.path == str(PRIMARY_TABLE)) & (frame.sha256 == PRIMARY_TABLE_SHA256)).sum() == 1
        and ((frame.path == str(EXTERNAL_AUDIT)) & (frame.sha256 == EXTERNAL_AUDIT_SHA256)).sum() == 1
    ):
        raise Amendment06IntegrityError("Primary/Audit source pins are absent from sealed inventory")
    if sha256_bytes(raw) != SOURCE_INVENTORY_SHA256:
        raise Amendment06IntegrityError("Sealed source-input inventory byte pin differs")
    return raw


def _snapshot_source_pre(run_dir: Path) -> None:
    destination = run_dir / "provenance/source_input_hashes_pre.tsv"
    data = _source_inventory_bytes()
    if destination.exists():
        if not _safe_regular(destination, sha256_bytes(data)):
            raise Amendment06IntegrityError("Full source pre-inventory drifted")
    else:
        publish_bytes_no_clobber(destination, data)


def _snapshot_source_post(run_dir: Path) -> None:
    destination = run_dir / "provenance/source_input_hashes_post.tsv"
    data = _source_inventory_bytes()
    if destination.exists():
        if not _safe_regular(destination, sha256_bytes(data)):
            raise Amendment06IntegrityError("Full source post-inventory drifted")
    else:
        publish_bytes_no_clobber(destination, data)


def load_development_population(locked: LockedInputs) -> DevelopmentPopulation:
    _progress("LOCKED_DEVELOPMENT_INPUT", "START")
    training = core.load_training_verified(PRIMARY_TABLE, PRIMARY_TABLE_SHA256)
    core.assert_training_matches_reviewed_split(training, locked.split)
    ordered_training = locked.split.loc[
        locked.split.split.astype(str).eq("training") & locked.split.eligible.astype(bool)
    ].sort_values("split_order", kind="stable")
    ordered_validation = locked.split.loc[
        locked.split.split.astype(str).eq("validation") & locked.split.eligible.astype(bool)
    ].sort_values("split_order", kind="stable")
    train = training.loc[ordered_training.index].copy().reset_index(drop=True)
    validation = training.loc[ordered_validation.index].copy().reset_index(drop=True)
    if not (
        len(train) == 291_024 and int(train.label.sum()) == 75_036
        and train.card_id.nunique() == 66
        and len(validation) == 56_843 and int(validation.label.sum()) == 15_952
        and validation.card_id.nunique() == 18
        and set(train.card_id.astype(str)).isdisjoint(validation.card_id.astype(str))
        and np.array_equal(
            train.stable_candidate_id.astype(str).to_numpy(),
            ordered_training.stable_candidate_id.astype(str).to_numpy(),
        )
        and np.array_equal(
            validation.stable_candidate_id.astype(str).to_numpy(),
            ordered_validation.stable_candidate_id.astype(str).to_numpy(),
        )
    ):
        raise Amendment06IntegrityError("Locked development population/order differs")
    training_weights, weight_evidence = core.review_weights(
        train, historical_geometry=False,
    )
    validation_weights, _ = core.review_weights(
        validation, historical_geometry=False,
    )
    if not (
        training_weights.dtype == np.float32
        and np.isfinite(training_weights).all()
        and np.all(training_weights > 0)
        and weight_evidence.get("weight_sha256_float32_le")
        == sha256_array(training_weights.astype("<f4", copy=False))
    ):
        raise Amendment06IntegrityError("Locked review-weight vector is invalid")
    _progress("LOCKED_DEVELOPMENT_INPUT", "COMPLETE", train_rows=len(train), validation_rows=len(validation))
    return DevelopmentPopulation(
        training=train, validation=validation,
        training_labels=train.label.to_numpy(np.int8),
        validation_labels=validation.label.to_numpy(np.int8),
        training_weights=training_weights,
        validation_weights=validation_weights,
    )


def _feature_list_sha256(features: Sequence[str]) -> str:
    return sha256_bytes(("\n".join(map(str, features)) + "\n").encode("utf-8"))


def _lineage_bytes(values: Sequence[Any]) -> bytes:
    return ("\n".join(map(str, values)) + "\n").encode("utf-8")


def _save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, np.ascontiguousarray(values), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _cache_file_records(root: Path, relatives: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in sorted(relatives):
        path = root / relative
        if not _safe_regular(path):
            raise Amendment06IntegrityError(f"Frozen cache member is absent/unsafe: {relative}")
        records.append({
            "relative_path": relative, "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return records


def _portable_imputer_payload(
    *, run_id: str, features: Sequence[str], statistics: np.ndarray,
) -> dict[str, Any]:
    medians = np.asarray(statistics, dtype=np.float64)
    if medians.shape != (93,) or not np.isfinite(medians).all():
        raise Amendment06IntegrityError("Training-only 93D median imputer is invalid")
    content: dict[str, Any] = {
        "run_id": run_id,
        "strategy": "median",
        "fit_population": "LOCKED_ORDERED_TRAINING_ONLY",
        "fit_rows": 291_024,
        "feature_count": 93,
        "feature_order": list(features),
        "feature_order_sha256": _feature_list_sha256(features),
        "statistics": medians.tolist(),
        "statistics_dtype": "float64",
        "transform_output_dtype": "float32",
        "primary_table_sha256": PRIMARY_TABLE_SHA256,
        "split_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
    }
    content["payload_sha256"] = sha256_bytes(strict_full_canonical_json_bytes(content))
    return content


def _validate_stage_counts(stages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_name = {str(stage.get("stage")): dict(stage) for stage in stages}
    if set(by_name) != set(EXPECTED_STAGE_COUNTS):
        raise Amendment06IntegrityError(f"Frozen stage set differs: {sorted(by_name)}")
    result: dict[str, Any] = {}
    for stage, (rows, negative, positive) in EXPECTED_STAGE_COUNTS.items():
        observed = by_name[stage]
        if (
            int(observed.get("rows", -1)), int(observed.get("negative", -1)),
            int(observed.get("positive", -1)),
        ) != (rows, negative, positive):
            raise Amendment06IntegrityError(f"Frozen stage count differs: {stage}")
        result[stage] = {"rows": rows, "negative": negative, "positive": positive}
    if by_name["post_shuffle"].get("shuffle_applied") is not False:
        raise Amendment06IntegrityError("Post-augmentation shuffle must be a no-op")
    if int(by_name["post_safe_smote"].get("safe_smote_synthetic_rows", -1)) != 147_841:
        raise Amendment06IntegrityError("Safe-SMOTE synthetic row count differs")
    result["post_shuffle"]["shuffle_applied"] = False
    result["post_safe_smote"]["safe_smote_synthetic_rows"] = 147_841
    return result


def _build_population_cache(
    *, run_dir: Path, locked: LockedInputs, development: DevelopmentPopulation,
    destination: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    temporary = destination.parent / f".{destination.name}.building.{os.getpid()}.{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        _progress("FROZEN_FULL_SPACE_RESAMPLING", "START", rows=len(development.training), features=93)
        if tuple(core.get_r92_feature_order()) != locked.full_features:
            raise Amendment06IntegrityError("Live canonical 93D order differs from locked order")
        frozen = paired_resampling.build_frozen_resampling_population(
            development.training.loc[:, locked.full_features],
            development.training_labels,
            development.training_weights,
            development.training.stable_candidate_id.astype(str),
            full_features=locked.full_features,
            seed=BASE_SEED,
            backend=core,
        )
        stage_counts = _validate_stage_counts(frozen.stages)
        validation = paired_resampling.transform_validation_full_space(
            frozen, development.validation.loc[:, locked.full_features],
        )
        arrays = {
            "pre_smote_X.npy": np.asarray(frozen.pre_smote_X, dtype=np.float32),
            "pre_smote_y.npy": np.asarray(frozen.pre_smote_y, dtype=np.int8),
            "pre_smote_weights.npy": np.asarray(frozen.pre_smote_weights, dtype=np.float32),
            "post_smote_X.npy": np.asarray(frozen.post_smote_X, dtype=np.float32),
            "post_smote_y.npy": np.asarray(frozen.post_smote_y, dtype=np.int8),
            "post_smote_weights.npy": np.asarray(frozen.post_smote_weights, dtype=np.float32),
            "validation_X.npy": np.asarray(validation.X, dtype=np.float32),
        }
        for relative, values in arrays.items():
            _save_npy(temporary / relative, values)
        publish_bytes_no_clobber(
            temporary / "pre_smote_lineage.txt", _lineage_bytes(frozen.pre_smote_lineage),
        )
        publish_bytes_no_clobber(
            temporary / "post_smote_lineage.txt", _lineage_bytes(frozen.post_smote_lineage),
        )
        imputer = _portable_imputer_payload(
            run_id=run_dir.name, features=locked.full_features,
            statistics=np.asarray(frozen.imputer.statistics_),
        )
        publish_strict_json_no_clobber(temporary / "portable_imputer.json", imputer)
        relatives = [*arrays, "pre_smote_lineage.txt", "post_smote_lineage.txt", "portable_imputer.json"]
        file_records = _cache_file_records(temporary, relatives)
        scientific_binding = {
            "run_id": run_dir.name,
            "primary_table_sha256": PRIMARY_TABLE_SHA256,
            "split_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
            "base_seed": BASE_SEED,
            "feature_order_sha256": frozen.feature_order_sha256,
            "raw_review_weight_sha256": sha256_array(development.training_weights.astype("<f4", copy=False)),
            "validation_X_sha256": sha256_array(arrays["validation_X.npy"]),
            "stages": [dict(stage) for stage in frozen.stages],
            "stage_counts": stage_counts,
            "cache_files": file_records,
        }
        frozen_population_sha256 = sha256_bytes(strict_full_canonical_json_bytes(scientific_binding))
        manifest = {
            **scientific_binding,
            "status": "PASS",
            "classification": "ONE_FROZEN_PAIRED_FULL_SPACE_POPULATION",
            "unique_realizations": 1,
            "augmentation_generation_count": 1,
            "safe_smote_generation_count": 1,
            "safe_smote_synthetic_rows": 147_841,
            "frozen_population_sha256": frozen_population_sha256,
            "imputer_sha256": sha256_bytes(strict_full_json_bytes(imputer)),
            "cache_packaged": False,
            "same_run_resume_cache": True,
        }
        publish_strict_json_no_clobber(temporary / "cache_manifest.json", manifest)
        _rename_directory_no_clobber(temporary, destination)
        _progress("FROZEN_FULL_SPACE_RESAMPLING", "COMPLETE", post_smote_rows=POST_SMOTE_ROWS)
        return manifest, imputer
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_population_cache(
    *, run_dir: Path, destination: Path, locked: LockedInputs,
    development: DevelopmentPopulation,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_all_members = {
        "cache_manifest.json", "pre_smote_X.npy", "pre_smote_y.npy",
        "pre_smote_weights.npy", "post_smote_X.npy", "post_smote_y.npy",
        "post_smote_weights.npy", "validation_X.npy",
        "pre_smote_lineage.txt", "post_smote_lineage.txt", "portable_imputer.json",
    }
    observed_members: set[str] = set()
    for member in destination.iterdir():
        if member.is_symlink() or not member.is_file():
            raise Amendment06IntegrityError(f"Frozen cache has an unsafe member: {member}")
        observed_members.add(member.name)
    if observed_members != expected_all_members:
        raise Amendment06IntegrityError("Frozen cache member set differs")
    manifest = strict_full_load_file(destination / "cache_manifest.json")
    imputer = strict_full_load_file(destination / "portable_imputer.json")
    expected_members = tuple(sorted((
        "pre_smote_X.npy", "pre_smote_y.npy", "pre_smote_weights.npy",
        "post_smote_X.npy", "post_smote_y.npy", "post_smote_weights.npy",
        "validation_X.npy", "pre_smote_lineage.txt", "post_smote_lineage.txt",
        "portable_imputer.json",
    )))
    records = manifest.get("cache_files")
    if (
        not isinstance(records, list)
        or tuple(str(item.get("relative_path")) for item in records) != expected_members
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"relative_path", "size_bytes", "sha256"}
            for item in records
        )
    ):
        raise Amendment06IntegrityError("Frozen cache manifest member set/order differs")
    for item in records:
        path = destination / str(item.get("relative_path"))
        if not (
            _safe_regular(path, str(item.get("sha256")))
            and path.stat().st_size == int(item.get("size_bytes", -1))
        ):
            raise Amendment06IntegrityError(f"Frozen cache member drifted: {path}")
    scientific_binding = {
        key: manifest[key]
        for key in (
            "run_id", "primary_table_sha256", "split_assignment_sha256", "base_seed",
            "feature_order_sha256", "raw_review_weight_sha256", "validation_X_sha256",
            "stages", "stage_counts", "cache_files",
        )
    }
    required_imputer_keys = {
        "run_id", "strategy", "fit_population", "fit_rows", "feature_count",
        "feature_order", "feature_order_sha256", "statistics", "statistics_dtype",
        "transform_output_dtype", "primary_table_sha256", "split_assignment_sha256",
        "payload_sha256",
    }
    if not (
        manifest.get("run_id") == run_dir.name
        and manifest.get("status") == "PASS"
        and manifest.get("classification") == "ONE_FROZEN_PAIRED_FULL_SPACE_POPULATION"
        and manifest.get("primary_table_sha256") == PRIMARY_TABLE_SHA256
        and manifest.get("split_assignment_sha256") == SPLIT_ASSIGNMENT_SHA256
        and manifest.get("base_seed") == BASE_SEED
        and manifest.get("unique_realizations") == 1
        and manifest.get("augmentation_generation_count") == 1
        and manifest.get("safe_smote_generation_count") == 1
        and manifest.get("safe_smote_synthetic_rows") == 147_841
        and manifest.get("cache_packaged") is False
        and manifest.get("same_run_resume_cache") is True
        and manifest.get("frozen_population_sha256")
        == sha256_bytes(strict_full_canonical_json_bytes(scientific_binding))
        and manifest.get("imputer_sha256") == sha256_file(destination / "portable_imputer.json")
        and set(imputer) == required_imputer_keys
        and imputer.get("run_id") == run_dir.name
        and imputer.get("strategy") == "median"
        and imputer.get("fit_population") == "LOCKED_ORDERED_TRAINING_ONLY"
        and imputer.get("fit_rows") == 291_024
        and imputer.get("feature_count") == 93
        and isinstance(imputer.get("feature_order"), list)
        and tuple(map(str, imputer["feature_order"])) == locked.full_features
        and len(imputer["feature_order"]) == 93
        and len(set(map(str, imputer["feature_order"]))) == 93
        and imputer.get("feature_order_sha256") == _feature_list_sha256(imputer["feature_order"])
        and manifest.get("feature_order_sha256") == imputer.get("feature_order_sha256")
        and imputer.get("statistics_dtype") == "float64"
        and imputer.get("transform_output_dtype") == "float32"
        and imputer.get("primary_table_sha256") == PRIMARY_TABLE_SHA256
        and imputer.get("split_assignment_sha256") == SPLIT_ASSIGNMENT_SHA256
        and isinstance(imputer.get("statistics"), list)
        and len(imputer["statistics"]) == 93
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) for value in imputer["statistics"]
        )
        and imputer.get("payload_sha256") == sha256_bytes(strict_full_canonical_json_bytes({
            key: value for key, value in imputer.items() if key != "payload_sha256"
        }))
    ):
        raise Amendment06IntegrityError("Frozen cache scientific binding differs")
    stage_counts = _validate_stage_counts(manifest["stages"])
    if manifest.get("stage_counts") != stage_counts:
        raise Amendment06IntegrityError("Frozen cache redundant stage counts differ")
    stages = list(manifest["stages"])
    if [str(stage.get("stage")) for stage in stages] != list(EXPECTED_STAGE_COUNTS):
        raise Amendment06IntegrityError("Frozen cache stage order differs")
    required_stage_keys = {
        "stage", "rows", "features", "negative", "positive",
        "feature_order_sha256", "X_sha256", "y_sha256", "weight_sha256",
        "source_lineage_sha256", "row_order_sha256",
    }
    for stage in stages:
        if not required_stage_keys.issubset(stage):
            raise Amendment06IntegrityError("Frozen cache stage hash schema differs")
        if not (
            stage.get("features") == 93
            and stage.get("feature_order_sha256") == manifest["feature_order_sha256"]
            and all(
                re.fullmatch(r"[0-9a-f]{64}", str(stage.get(key, "")))
                for key in (
                    "X_sha256", "y_sha256", "weight_sha256",
                    "source_lineage_sha256", "row_order_sha256",
                )
            )
            and stage.get("source_lineage_sha256") == stage.get("row_order_sha256")
        ):
            raise Amendment06IntegrityError(f"Frozen cache stage binding differs: {stage.get('stage')}")
    arrays = {
        name: np.load(destination / name, mmap_mode="r", allow_pickle=False)
        for name in (
            "pre_smote_X.npy", "pre_smote_y.npy", "pre_smote_weights.npy",
            "post_smote_X.npy", "post_smote_y.npy", "post_smote_weights.npy",
            "validation_X.npy",
        )
    }
    expected_arrays = {
        "pre_smote_X.npy": ((PRE_SMOTE_ROWS, 93), np.dtype("float32")),
        "pre_smote_y.npy": ((PRE_SMOTE_ROWS,), np.dtype("int8")),
        "pre_smote_weights.npy": ((PRE_SMOTE_ROWS,), np.dtype("float32")),
        "post_smote_X.npy": ((POST_SMOTE_ROWS, 93), np.dtype("float32")),
        "post_smote_y.npy": ((POST_SMOTE_ROWS,), np.dtype("int8")),
        "post_smote_weights.npy": ((POST_SMOTE_ROWS,), np.dtype("float32")),
        "validation_X.npy": ((56_843, 93), np.dtype("float32")),
    }
    for name, (shape, dtype) in expected_arrays.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise Amendment06IntegrityError(f"Frozen cache array shape/dtype differs: {name}")
    for name in ("pre_smote_X.npy", "post_smote_X.npy", "validation_X.npy"):
        if not np.isfinite(arrays[name]).all():
            raise Amendment06IntegrityError(f"Frozen cache matrix is nonfinite: {name}")
    for name in ("pre_smote_weights.npy", "post_smote_weights.npy"):
        if not (np.isfinite(arrays[name]).all() and np.all(arrays[name] > 0)):
            raise Amendment06IntegrityError(f"Frozen cache weights are invalid: {name}")
    for name in ("pre_smote_y.npy", "post_smote_y.npy"):
        if not np.isin(arrays[name], (0, 1)).all():
            raise Amendment06IntegrityError(f"Frozen cache labels are nonbinary: {name}")
    by_name = {str(stage["stage"]): stage for stage in stages}
    endpoint_bindings = (
        (
            "post_jitter_pre_smote", "pre_smote_X.npy", "pre_smote_y.npy",
            "pre_smote_weights.npy", "pre_smote_lineage.txt",
        ),
        (
            "post_shuffle", "pre_smote_X.npy", "pre_smote_y.npy",
            "pre_smote_weights.npy", "pre_smote_lineage.txt",
        ),
        (
            "post_safe_smote", "post_smote_X.npy", "post_smote_y.npy",
            "post_smote_weights.npy", "post_smote_lineage.txt",
        ),
    )
    for stage_name, x_name, y_name, weight_name, lineage_name in endpoint_bindings:
        stage = by_name[stage_name]
        lineage_sha = sha256_file(destination / lineage_name)
        if not (
            stage["X_sha256"] == sha256_array(arrays[x_name])
            and stage["y_sha256"] == sha256_array(arrays[y_name])
            and stage["weight_sha256"] == sha256_array(arrays[weight_name])
            and stage["source_lineage_sha256"] == lineage_sha
            and stage["row_order_sha256"] == lineage_sha
            and int((arrays[y_name] == 0).sum()) == stage["negative"]
            and int((arrays[y_name] == 1).sum()) == stage["positive"]
            and int(len(arrays[y_name])) == stage["rows"]
        ):
            raise Amendment06IntegrityError(f"Frozen cache endpoint bytes differ: {stage_name}")
    invariant_keys = (
        "rows", "features", "negative", "positive", "feature_order_sha256",
        "X_sha256", "y_sha256", "weight_sha256", "source_lineage_sha256",
        "row_order_sha256",
    )
    if any(
        by_name["post_jitter_pre_smote"].get(key) != by_name["post_shuffle"].get(key)
        for key in invariant_keys
    ):
        raise Amendment06IntegrityError("Frozen no-op post-shuffle bytes differ")
    if not (
        manifest.get("raw_review_weight_sha256")
        == sha256_array(development.training_weights.astype("<f4", copy=False))
        == by_name["raw_locked_post_split_shuffle"]["weight_sha256"]
        and manifest.get("validation_X_sha256") == sha256_array(arrays["validation_X.npy"])
    ):
        raise Amendment06IntegrityError("Frozen cache raw-weight/validation binding differs")
    recomputed_medians: list[float] = []
    for feature in locked.full_features:
        values = pd.to_numeric(
            development.training[feature], errors="coerce",
        ).to_numpy(np.float64)
        if np.isinf(values).any() or np.isnan(values).all():
            raise Amendment06IntegrityError(f"Locked imputer source is invalid: {feature}")
        recomputed_medians.append(float(np.nanmedian(values)))
    if not np.array_equal(
        np.asarray(recomputed_medians, dtype=np.float64),
        np.asarray(imputer["statistics"], dtype=np.float64),
    ):
        raise Amendment06IntegrityError("Frozen cache imputer medians differ from locked training")
    validation_values = development.validation.loc[:, locked.full_features].apply(
        pd.to_numeric, errors="coerce",
    ).to_numpy(np.float64)
    if np.isinf(validation_values).any():
        raise Amendment06IntegrityError("Locked validation contains infinity")
    missing = np.isnan(validation_values)
    if missing.any():
        validation_values[missing] = np.take(
            np.asarray(imputer["statistics"], dtype=np.float64),
            np.nonzero(missing)[1],
        )
    expected_validation = np.ascontiguousarray(validation_values, dtype=np.float32)
    if not np.array_equal(expected_validation, arrays["validation_X.npy"]):
        raise Amendment06IntegrityError("Frozen validation transform differs from locked bytes")
    for lineage_name, expected_rows in (
        ("pre_smote_lineage.txt", PRE_SMOTE_ROWS),
        ("post_smote_lineage.txt", POST_SMOTE_ROWS),
    ):
        lineage_path = destination / lineage_name
        with lineage_path.open("rb") as handle:
            rows = sum(1 for line in handle if line.endswith(b"\n") and line != b"\n")
        if rows != expected_rows:
            raise Amendment06IntegrityError(f"Frozen cache lineage row count differs: {lineage_name}")
    return manifest, imputer


def load_or_build_population_cache(
    *, run_dir: Path, locked: LockedInputs,
    development: DevelopmentPopulation | None,
) -> FrozenPopulationCache:
    destination = EXPECTED_RUNTIME_ROOT / "full_population" / run_dir.name
    published_manifest = run_dir / "provenance/frozen_full_resampling_manifest.json"
    cache_existed = destination.exists()
    operation = "CACHE_REMATERIALIZATION" if published_manifest.exists() else "CACHE_GENERATION"
    intent_name = f"{operation}_INTENT"
    intents = [
        event for event in _read_journal(run_dir)
        if event.get("event") == intent_name
    ]
    if len(intents) > 1:
        raise Amendment06IntegrityError(f"Duplicate frozen-cache intent: {intent_name}")
    if not cache_existed:
        if intents:
            intent = intents[0]
            if not (
                intent.get("cache_path") == str(destination)
                and intent.get("base_seed") == BASE_SEED
                and intent.get("primary_table_sha256") == PRIMARY_TABLE_SHA256
                and intent.get("split_assignment_sha256") == SPLIT_ASSIGNMENT_SHA256
            ):
                raise Amendment06IntegrityError("Frozen-cache retry intent differs")
        else:
            _journal_event(
                run_dir, intent_name, status="STARTED", cache_path=str(destination),
                base_seed=BASE_SEED, primary_table_sha256=PRIMARY_TABLE_SHA256,
                split_assignment_sha256=SPLIT_ASSIGNMENT_SHA256,
            )
    if cache_existed:
        if development is None:
            raise Amendment06ResumeError("Development inputs are required to validate the frozen cache")
        if not published_manifest.exists() and not intents:
            raise Amendment06IntegrityError(
                "An unowned first-use frozen population cache already exists"
            )
        manifest, imputer = _validate_population_cache(
            run_dir=run_dir, destination=destination, locked=locked,
            development=development,
        )
        if published_manifest.exists():
            _journal_event(run_dir, "CACHE_LOAD", status="PASS", cache_path=str(destination))
    else:
        if development is None:
            raise Amendment06ResumeError("Frozen population cache is absent and development inputs are unavailable")
        manifest, imputer = _build_population_cache(
            run_dir=run_dir, locked=locked, development=development, destination=destination,
        )
        manifest, imputer = _validate_population_cache(
            run_dir=run_dir, destination=destination, locked=locked,
            development=development,
        )
    if published_manifest.exists():
        published = strict_full_load_file(published_manifest)
        if published != manifest:
            raise Amendment06IntegrityError("Runtime cache differs from published frozen-population manifest")
    else:
        _ensure_unique_bound_journal_event(
            run_dir=run_dir, event_name="CACHE_GENERATION",
            binding_path=destination / "cache_manifest.json",
            binding_field="cache_manifest_sha256", recovered=cache_existed,
            extra={"status": "PASS", "cache_path": str(destination)},
        )
        publish_strict_json_no_clobber(published_manifest, manifest)
    rematerialization_intents = [
        event for event in _read_journal(run_dir)
        if event.get("event") == "CACHE_REMATERIALIZATION_INTENT"
    ]
    if operation == "CACHE_REMATERIALIZATION" and rematerialization_intents:
        _ensure_unique_bound_journal_event(
            run_dir=run_dir, event_name="CACHE_REMATERIALIZATION",
            binding_path=destination / "cache_manifest.json",
            binding_field="cache_manifest_sha256", recovered=cache_existed,
            extra={"status": "PASS", "cache_path": str(destination)},
        )
    imputer_path = run_dir / "provenance/frozen_full_imputer.json"
    if imputer_path.exists():
        if strict_full_load_file(imputer_path) != imputer:
            raise Amendment06IntegrityError("Portable imputer differs from frozen cache")
    else:
        publish_strict_json_no_clobber(imputer_path, imputer)
    arrays = {
        name: np.load(destination / name, mmap_mode="r", allow_pickle=False)
        for name in (
            "pre_smote_X.npy", "pre_smote_y.npy", "pre_smote_weights.npy",
            "post_smote_X.npy", "post_smote_y.npy", "post_smote_weights.npy",
            "validation_X.npy",
        )
    }
    expected_shapes = {
        "pre_smote_X.npy": (PRE_SMOTE_ROWS, 93), "pre_smote_y.npy": (PRE_SMOTE_ROWS,),
        "pre_smote_weights.npy": (PRE_SMOTE_ROWS,), "post_smote_X.npy": (POST_SMOTE_ROWS, 93),
        "post_smote_y.npy": (POST_SMOTE_ROWS,), "post_smote_weights.npy": (POST_SMOTE_ROWS,),
        "validation_X.npy": (56_843, 93),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise Amendment06IntegrityError(f"Frozen cache array shape differs: {name}")
    return FrozenPopulationCache(
        root=destination, manifest=manifest, imputer=imputer,
        full_features=tuple(imputer["feature_order"]),
        pre_X=arrays["pre_smote_X.npy"], pre_y=arrays["pre_smote_y.npy"],
        pre_weights=arrays["pre_smote_weights.npy"], post_X=arrays["post_smote_X.npy"],
        post_y=arrays["post_smote_y.npy"], post_weights=arrays["post_smote_weights.npy"],
        validation_X=arrays["validation_X.npy"],
    )


def project_variant(cache: FrozenPopulationCache, locked: LockedInputs, variant: str) -> VariantMatrices:
    if variant not in TRAINED_VARIANTS:
        raise Amendment06IntegrityError(f"Unauthorized Full Variant: {variant}")
    features = tuple(locked.variant_features[variant]["retained_features"])
    expected_count, expected_sha = VARIANT_FEATURE_CONTRACT[variant]
    if len(features) != expected_count or _feature_list_sha256(features) != expected_sha:
        raise Amendment06IntegrityError(f"Variant feature contract drifted: {variant}")
    index = {feature: offset for offset, feature in enumerate(cache.full_features)}
    indices = tuple(index[feature] for feature in features)
    if tuple(sorted(indices)) != indices:
        raise Amendment06IntegrityError(f"Variant projection reorders columns: {variant}")
    use_pre = variant == "no_safe_smote"
    source_X = cache.pre_X if use_pre else cache.post_X
    source_y = cache.pre_y if use_pre else cache.post_y
    source_weights = cache.pre_weights if use_pre else cache.post_weights
    X = source_X if indices == tuple(range(93)) else np.ascontiguousarray(source_X[:, indices], dtype=np.float32)
    validation_X = (
        cache.validation_X if indices == tuple(range(93))
        else np.ascontiguousarray(cache.validation_X[:, indices], dtype=np.float32)
    )
    weights = (
        np.ones(len(source_y), dtype=np.float32)
        if variant == "no_review_aware_training_weights" else source_weights
    )
    if not (np.isfinite(X).all() and np.isfinite(weights).all() and np.all(weights > 0)):
        raise Amendment06IntegrityError(f"Variant projection is nonfinite/invalid: {variant}")
    stage = "post_jitter_pre_smote" if use_pre else "post_safe_smote"
    stage_evidence = next(item for item in cache.manifest["stages"] if item["stage"] == stage)
    population_evidence = {
        "variant": variant, "source_stage": stage, "projection_only": True,
        "rows": int(len(source_y)), "feature_count": len(features),
        "feature_list_sha256": expected_sha,
        "full_space_X_sha256": stage_evidence["X_sha256"],
        "y_sha256": stage_evidence["y_sha256"],
        "source_lineage_sha256": stage_evidence["source_lineage_sha256"],
        "row_order_sha256": stage_evidence["row_order_sha256"],
        "weight_sha256": sha256_array(np.asarray(weights, dtype="<f4")),
        "review_weights_enabled": variant != "no_review_aware_training_weights",
        "safe_smote_enabled": not use_pre,
        "safe_smote_synthetic_rows": 0 if use_pre else 147_841,
    }
    return VariantMatrices(
        variant=variant,
        classification=("EXPLORATORY" if variant == EXPLORATORY_VARIANT else "OFFICIAL_RUNNABLE"),
        features=features, indices=indices, X=X, y=source_y, weights=weights,
        validation_X=validation_X, population=stage, population_evidence=population_evidence,
    )


def _journal_path(run_dir: Path) -> Path:
    return EXPECTED_RUNTIME_ROOT / "journals" / run_dir.name


def _journal_event_path(run_dir: Path, event_index: int) -> Path:
    return _journal_path(run_dir) / f"{event_index:08d}.json"


def _journal_lock_path(run_dir: Path) -> Path:
    return EXPECTED_RUNTIME_ROOT / "journals" / f".{run_dir.name}.lock"


def _read_journal(run_dir: Path) -> list[dict[str, Any]]:
    path = _journal_path(run_dir)
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_dir():
        raise Amendment06IntegrityError("Execution ownership journal is unsafe")
    members = sorted(path.iterdir(), key=lambda item: item.name)
    authoritative = [item for item in members if re.fullmatch(r"[0-9]{8}\.json", item.name)]
    unexpected = [item for item in members if item not in authoritative]
    if unexpected:
        raise Amendment06IntegrityError("Execution ownership journal has unexpected members")
    events: list[dict[str, Any]] = []
    previous_sha256: str | None = None
    for expected_index, member in enumerate(authoritative):
        if member != _journal_event_path(run_dir, expected_index) or not _safe_regular(member):
            raise Amendment06IntegrityError("Execution ownership journal is noncontiguous/unsafe")
        value = strict_full_load_file(member)
        if not (
            isinstance(value, dict)
            and value.get("event_index") == expected_index
            and value.get("run_id") == run_dir.name
            and value.get("previous_event_sha256") == previous_sha256
            and isinstance(value.get("event"), str) and value["event"]
            and isinstance(value.get("recorded_at"), str) and value["recorded_at"]
        ):
            raise Amendment06IntegrityError(
                f"Execution ownership journal is invalid at event {expected_index}"
            )
        events.append(value)
        previous_sha256 = sha256_bytes(strict_full_canonical_json_bytes(value))
    return events


def _journal_event(run_dir: Path, event: str, **fields: Any) -> dict[str, Any]:
    path = _journal_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise Amendment06IntegrityError("Execution ownership journal is unsafe")
    lock_path = _journal_lock_path(run_dir)
    descriptor = os.open(
        lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600,
    )
    with os.fdopen(descriptor, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _read_journal(run_dir)
        payload = {
            "event_index": len(existing), "event": event, "run_id": run_dir.name,
            "previous_event_sha256": (
                None if not existing else sha256_bytes(
                    strict_full_canonical_json_bytes(existing[-1])
                )
            ),
            "recorded_at": _now(), **fields,
        }
        publish_bytes_no_clobber(
            _journal_event_path(run_dir, len(existing)), strict_full_json_bytes(payload),
        )
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return payload


def _ensure_unique_bound_journal_event(
    *, run_dir: Path, event_name: str, binding_path: Path,
    binding_field: str, phase: str | None = None,
    variant: str | None = None, recovered: bool,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not _safe_regular(binding_path):
        raise Amendment06IntegrityError(
            f"Journal-bound artifact is absent/unsafe: {binding_path}"
        )
    digest = sha256_file(binding_path)
    matches = [
        item for item in _read_journal(run_dir)
        if item.get("event") == event_name
        and (phase is None or item.get("phase") == phase)
        and (variant is None or item.get("variant_id") == variant)
    ]
    if len(matches) > 1:
        raise Amendment06IntegrityError(f"Duplicate journal commit event: {event_name}/{variant}")
    if matches:
        observed = matches[0]
        if observed.get(binding_field) != digest:
            raise Amendment06IntegrityError(
                f"Journal commit binding differs: {event_name}/{variant}"
            )
        return observed
    fields: dict[str, Any] = {}
    if phase is not None:
        fields["phase"] = phase
    if variant is not None:
        fields["variant_id"] = variant
    fields.update(dict(extra or {}))
    fields[binding_field] = digest
    fields["recovered_after_interruption"] = recovered
    return _journal_event(run_dir, event_name, **fields)


def _record_interrupted_attempts(
    *, run_dir: Path, phase: str, variant: str,
) -> None:
    names = {
        "TRAINING": ("TRAINING_ATTEMPT_STARTED", "TRAINING_ATTEMPT_FAILED"),
        "AUDIT_SCORING": (
            "AUDIT_SCORING_ATTEMPT_STARTED", "AUDIT_SCORING_ATTEMPT_FAILED",
        ),
        "REPORTING": ("REPORTING_ATTEMPT_STARTED", "REPORTING_ATTEMPT_FAILED"),
    }
    if phase not in names:
        raise Amendment06IntegrityError(f"Unknown interruption phase: {phase}")
    start_name, failed_name = names[phase]
    events = _read_journal(run_dir)
    failed_ids = {
        str(event.get("attempt_id")) for event in events
        if event.get("event") == failed_name and event.get("phase") == phase
        and event.get("variant_id") == variant
    }
    committed_ids = {
        str(event.get("attempt_id")) for event in events
        if event.get("event") in {
            "TRAINING_COMPLETION_COMMITTED", "AUDIT_SCORING_COMPLETION_COMMITTED",
            "REPORTING_COMPLETION_COMMITTED",
        }
        and event.get("phase") == phase and event.get("variant_id") == variant
        and event.get("attempt_id") is not None
    }
    for event in events:
        attempt_id = str(event.get("attempt_id", ""))
        if (
            event.get("event") == start_name
            and event.get("phase") == phase and event.get("variant_id") == variant
            and attempt_id and attempt_id not in failed_ids | committed_ids
        ):
            _journal_event(
                run_dir, failed_name, phase=phase, variant_id=variant,
                attempt_id=attempt_id,
                failure_phase="PROCESS_INTERRUPTION_BEFORE_COMPLETION_COMMIT",
                error_type="InterruptedProcess",
                error="Recovered an unterminated attempt before same-Run retry",
                recovered_after_interruption=True,
            )


_SCIENTIFIC_COUNTERS: MutableMapping[str, int] = {
    "training_calls": 0,
    "augmentation_calls": 0,
    "safe_smote_calls": 0,
    "imputer_fit_calls": 0,
    "post_audit_training_calls": 0,
    "threshold_tuning_calls": 0,
    "stability_fits": 0,
}
_POST_FREEZE_GUARDS_INSTALLED = False


class BoundedFitProgress:
    """Emit bounded resource progress without interacting with model state."""

    def __init__(
        self, run_id: str, variant: str, *, interval_seconds: float = 60.0,
        snapshot: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("Fit progress interval must be positive")
        self.run_id = run_id
        self.variant = variant
        self.interval_seconds = float(interval_seconds)
        self.snapshot = snapshot
        self._stop = threading.Event()
        self._started = time.monotonic()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _resources(self) -> Mapping[str, Any]:
        if self.snapshot is not None:
            return self.snapshot()
        return {
            "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "gpu_memory_mib": core._gpu_used_memory_mib(),
        }

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                values = self._resources()
                _progress(
                    "FULL_VARIANT_FIT", "ACTIVE", run_id=self.run_id,
                    variant=self.variant,
                    elapsed_seconds=f"{time.monotonic() - self._started:.1f}",
                    rss_bytes=values.get("rss_bytes"),
                    gpu_memory_mib=values.get("gpu_memory_mib"),
                )
            except Exception as exc:
                _progress(
                    "FULL_VARIANT_FIT_MONITOR", "WARNING", variant=self.variant,
                    error_type=type(exc).__name__,
                )

    def __enter__(self) -> "BoundedFitProgress":
        self._started = time.monotonic()
        self._thread = threading.Thread(
            target=self._loop, name=f"a06-fit-progress-{self.variant}", daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        if self.running:
            raise Amendment06IntegrityError("Fit progress monitor did not stop")


def _binary_metrics(
    labels: Sequence[int], probabilities: Sequence[float], *,
    threshold: float = DECISION_THRESHOLD, weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    result = dict(reporting.binary_metrics(
        labels, probabilities, threshold=threshold, weights=weights,
    ))
    assert_required_metrics_finite(result)
    return result


PREDICTION_COLUMNS = reporting.PREDICTION_COLUMNS


def deterministic_csv_gzip_bytes(frame: pd.DataFrame) -> bytes:
    if list(frame.columns) != list(PREDICTION_COLUMNS):
        raise Amendment06IntegrityError("Prediction table schema/order differs")
    return reporting.deterministic_gzip_bytes(reporting.prediction_csv_bytes(frame))


def validate_prediction_roundtrip(
    data_or_path: bytes | Path, authoritative: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(data_or_path, Path):
        reopened = pd.read_csv(data_or_path, engine="c", float_precision="round_trip")
    else:
        reopened = pd.read_csv(
            io.BytesIO(data_or_path), compression="gzip", engine="c",
            float_precision="round_trip",
        )
    if list(reopened.columns) != list(PREDICTION_COLUMNS) or len(reopened) != len(authoritative):
        raise Amendment06IntegrityError("Prediction CSV roundtrip schema/row count differs")
    text_columns = ("stable_candidate_id", "card_id", "variant_id", "split_population")
    integer_columns = ("true_label", "prediction_at_0_5", "row_order")
    numeric_columns = ("probability", "review_weight")
    mismatches: dict[str, int] = {}
    for column in text_columns:
        count = int((reopened[column].astype(str).to_numpy() != authoritative[column].astype(str).to_numpy()).sum())
        mismatches[column] = count
    for column in integer_columns:
        left = pd.to_numeric(reopened[column], errors="raise").to_numpy(np.int64)
        right = pd.to_numeric(authoritative[column], errors="raise").to_numpy(np.int64)
        mismatches[column] = int((left != right).sum())
    for column in numeric_columns:
        left = pd.to_numeric(reopened[column], errors="coerce").to_numpy(np.float64)
        right = pd.to_numeric(authoritative[column], errors="coerce").to_numpy(np.float64)
        same_missing = np.array_equal(np.isnan(left), np.isnan(right))
        finite = np.isfinite(left) & np.isfinite(right)
        mismatches[column] = (
            int((left[finite] != right[finite]).sum())
            if same_missing else len(left)
        )
    total = sum(mismatches.values())
    if total:
        raise Amendment06IntegrityError(f"Prediction CSV exact roundtrip failed: {mismatches}")
    reporting_evidence = reporting.validate_prediction_csv_roundtrip(
        data_or_path, authoritative, compressed=True,
    )
    return reopened, {
        **reporting_evidence,
        "rows": len(reopened), "mismatch_count": 0,
        "column_mismatch_counts": mismatches,
    }


def exact_table_csv_bytes(frame: pd.DataFrame) -> tuple[bytes, dict[str, Any]]:
    data = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    reopened = pd.read_csv(
        io.BytesIO(data), engine="c", float_precision="round_trip",
    )
    if tuple(reopened.columns) != tuple(frame.columns) or len(reopened) != len(frame):
        raise Amendment06IntegrityError("Table CSV exact roundtrip shape/schema differs")
    for column in frame.columns:
        expected = frame[column]
        observed = reopened[column]
        expected_missing = pd.isna(expected).to_numpy()
        observed_missing = pd.isna(observed).to_numpy()
        if not np.array_equal(expected_missing, observed_missing):
            raise Amendment06IntegrityError(
                f"Table CSV missingness mask differs: {column}"
            )
        finite = ~expected_missing
        if pd.api.types.is_numeric_dtype(expected.dtype):
            left = pd.to_numeric(expected[finite], errors="raise").to_numpy(np.float64)
            right = pd.to_numeric(observed[finite], errors="raise").to_numpy(np.float64)
            if not np.array_equal(left, right):
                raise Amendment06IntegrityError(
                    f"Table CSV numeric roundtrip differs: {column}"
                )
        elif not np.array_equal(
            expected[finite].astype(str).to_numpy(),
            observed[finite].astype(str).to_numpy(),
        ):
            raise Amendment06IntegrityError(
                f"Table CSV text roundtrip differs: {column}"
            )
    return data, {
        "status": "PASS", "rows": len(frame), "columns": list(frame.columns),
        "engine": "c", "float_precision": "round_trip",
        "numeric_equality": "EXACT_FINITE_WITH_IDENTICAL_MISSINGNESS",
        "csv_sha256": sha256_bytes(data),
    }


def validate_exact_table_csv(
    path: Path, expected: pd.DataFrame,
) -> dict[str, Any]:
    expected_bytes, evidence = exact_table_csv_bytes(expected)
    if not _safe_regular(path, sha256_bytes(expected_bytes)):
        raise Amendment06IntegrityError(f"Table CSV bytes differ: {path}")
    return evidence


def _attempt_root(run_dir: Path, phase: str, variant: str, attempt_id: str) -> Path:
    return EXPECTED_RUNTIME_ROOT / "attempts" / run_dir.name / phase / variant / attempt_id


def _safe_run_relative(relative: str) -> str:
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise Amendment06IntegrityError(f"Unsafe Run-relative artifact path: {relative!r}")
    return relative


def _stage_bytes(staging: Path, relative: str, data: bytes) -> Path:
    path = staging / _safe_run_relative(relative)
    publish_bytes_no_clobber(path, data)
    return path


def _stage_json(staging: Path, relative: str, value: Any) -> Path:
    return _stage_bytes(staging, relative, strict_full_json_bytes(value))


def _publication_record(path: Path, relative: str) -> dict[str, Any]:
    return {
        "relative_path": relative, "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_exact_artifact_records(
    *, run_dir: Path, value: Any, expected_relatives: Sequence[str], label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(expected_relatives):
        raise Amendment06IntegrityError(f"{label} artifact record count differs")
    records: dict[str, dict[str, Any]] = {}
    observed: list[str] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "relative_path", "size_bytes", "sha256",
        }:
            raise Amendment06IntegrityError(f"{label} artifact record schema differs")
        relative = str(item["relative_path"])
        observed.append(relative)
        if relative in records:
            raise Amendment06IntegrityError(f"{label} artifact records are duplicated")
        member = run_dir / _safe_run_relative(relative)
        size = item["size_bytes"]
        digest = str(item["sha256"])
        if (
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not _safe_regular(member, digest) or member.stat().st_size != size
        ):
            raise Amendment06IntegrityError(f"{label} artifact drifted: {relative}")
        records[relative] = dict(item)
    if observed != list(expected_relatives):
        raise Amendment06IntegrityError(f"{label} artifact member set/order differs")
    return records


def _publish_attempt_file(
    *, run_dir: Path, staging: Path, relative: str, phase: str,
    variant: str, attempt_id: str,
) -> dict[str, Any]:
    relative = _safe_run_relative(relative)
    source = staging / relative
    destination = run_dir / relative
    if not _safe_regular(source):
        raise Amendment06IntegrityError(f"Attempt source is absent/unsafe: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise Amendment06IntegrityError(f"No-clobber scientific publication collision: {relative}")
    source_record = _publication_record(source, relative)
    _journal_event(
        run_dir, "ARTIFACT_PUBLICATION_INTENT", phase=phase, variant_id=variant,
        attempt_id=attempt_id, **source_record,
    )
    # Move, rather than hardlink, so a frozen Run inode has no writable alias in
    # the attempt cache after publication.  The pre-move INTENT makes a crash
    # after renameat2 but before PUBLISHED recoverable by exact hash ownership.
    _rename_directory_no_clobber(source, destination)
    record = _publication_record(destination, relative)
    _journal_event(
        run_dir, "ARTIFACT_PUBLISHED", phase=phase, variant_id=variant,
        attempt_id=attempt_id, **record,
    )
    return record


def _phase_publication_records(
    run_dir: Path, phase: str, variant: str,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for event in _read_journal(run_dir):
        if (
            event.get("event") in {"ARTIFACT_PUBLICATION_INTENT", "ARTIFACT_PUBLISHED"}
            and event.get("phase") == phase and event.get("variant_id") == variant
        ):
            records[str(event["relative_path"])] = event
    return records


def _recover_unpublished_paths(
    *, run_dir: Path, phase: str, variant: str,
    completion_relative: str, possible_relatives: Sequence[str],
) -> None:
    completion = run_dir / completion_relative
    if completion.exists():
        return
    events = _read_journal(run_dir)
    owned = _phase_publication_records(run_dir, phase, variant)
    committed_transactions = {
        str(event.get("transaction_id")) for event in events
        if event.get("event") == "ARTIFACT_QUARANTINE_COMMITTED"
    }
    pending = [
        event for event in events
        if event.get("event") == "ARTIFACT_QUARANTINE_INTENT"
        and event.get("phase") == phase and event.get("variant_id") == variant
        and str(event.get("transaction_id")) not in committed_transactions
    ]

    def reconcile(intent: Mapping[str, Any], *, recovered: bool) -> None:
        relative = _safe_run_relative(str(intent.get("relative_path")))
        transaction_id = str(intent.get("transaction_id", ""))
        source = run_dir / relative
        destination = Path(str(intent.get("quarantine_path", "")))
        expected_root = EXPECTED_RUNTIME_ROOT / "quarantine" / run_dir.name / phase / variant
        if not (
            re.fullmatch(r"[0-9a-f]{32}", transaction_id)
            and expected_root.resolve() in destination.resolve().parents
            and re.fullmatch(r"[0-9a-f]{64}", str(intent.get("sha256", "")))
            and isinstance(intent.get("size_bytes"), int)
            and not isinstance(intent.get("size_bytes"), bool)
            and int(intent["size_bytes"]) >= 0
        ):
            raise Amendment06IntegrityError("Quarantine intent schema/path differs")
        digest = str(intent["sha256"])
        size = int(intent["size_bytes"])
        source_valid = _safe_regular(source, digest) and source.stat().st_size == size
        destination_valid = (
            _safe_regular(destination, digest) and destination.stat().st_size == size
        )
        if source_valid and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            _rename_directory_no_clobber(source, destination)
            _fsync_parent(source)
            destination_valid = (
                _safe_regular(destination, digest) and destination.stat().st_size == size
            )
        if source.exists() or not destination_valid:
            raise Amendment06IntegrityError(
                f"Quarantine transaction cannot be reconciled: {relative}"
            )
        _journal_event(
            run_dir, "ARTIFACT_QUARANTINE_COMMITTED", phase=phase,
            variant_id=variant, attempt_id=intent.get("attempt_id"),
            transaction_id=transaction_id, relative_path=relative,
            sha256=digest, size_bytes=size, quarantine_path=str(destination),
            recovered_after_interruption=recovered,
        )

    for intent in pending:
        reconcile(intent, recovered=True)

    existing = [relative for relative in possible_relatives if (run_dir / relative).exists()]
    for relative in existing:
        relative = _safe_run_relative(str(relative))
        source = run_dir / relative
        record = owned.get(relative)
        if not (
            record and _safe_regular(source, str(record.get("sha256")))
            and source.stat().st_size == int(record.get("size_bytes", -1))
        ):
            raise Amendment06IntegrityError(
                f"Unpublished artifact lacks exact ownership proof: {relative}"
            )
        transaction_id = uuid.uuid4().hex
        destination = (
            EXPECTED_RUNTIME_ROOT / "quarantine" / run_dir.name / phase / variant
            / transaction_id / relative
        )
        intent = _journal_event(
            run_dir, "ARTIFACT_QUARANTINE_INTENT", phase=phase,
            variant_id=variant, attempt_id=record.get("attempt_id"),
            transaction_id=transaction_id, relative_path=relative,
            sha256=record["sha256"], size_bytes=record["size_bytes"],
            quarantine_path=str(destination),
        )
        reconcile(intent, recovered=False)


def _training_relatives(variant: str) -> tuple[str, ...]:
    return (
        f"models/{variant}/model.ubj",
        f"models/{variant}/feature_list.txt",
        f"models/{variant}/model_metadata.json",
        f"models/{variant}/transform_binding.json",
        f"predictions/{variant}_validation.csv.gz",
        f"timing/{variant}.json",
        f"logs/{variant}.stdout.log",
        f"logs/{variant}.stderr.log",
    )


def _training_completion_relative(variant: str) -> str:
    return f"models/{variant}/training_completion_manifest.json"


def _fallback_warnings(messages: Sequence[str]) -> list[str]:
    tokens = ("fallback", "not compiled with gpu", "mismatched devices", "cpu predictor")
    return [message for message in messages if any(token in message.lower() for token in tokens)]


def _gpu_name(identity: Mapping[str, Any]) -> str:
    for key in ("gpu_name", "name"):
        if identity.get(key):
            return str(identity[key])
    raise Amendment06IntegrityError("GPU identity has no name")


def _training_prediction_frame(
    *, development: DevelopmentPopulation, probabilities: np.ndarray,
    variant: str, population: str,
) -> pd.DataFrame:
    if len(probabilities) != len(development.validation):
        raise Amendment06IntegrityError("Validation probability row count differs")
    return pd.DataFrame({
        "stable_candidate_id": development.validation.stable_candidate_id.astype(str).to_numpy(),
        "card_id": development.validation.card_id.astype(str).to_numpy(),
        "true_label": development.validation_labels,
        # Promote float32 predictions exactly so pandas emits enough decimal
        # digits for C-engine round_trip to recover the scientific value.
        "probability": np.asarray(probabilities, dtype=np.float32).astype(np.float64),
        "prediction_at_0_5": (np.asarray(probabilities) >= DECISION_THRESHOLD).astype(np.int8),
        "review_weight": development.validation_weights.astype(np.float64),
        "variant_id": variant,
        "split_population": f"validation:{population}",
        "row_order": np.arange(len(probabilities), dtype=np.int64),
    }, columns=PREDICTION_COLUMNS)


def _model_common_metadata(
    *, run_dir: Path, variant_data: VariantMatrices, model_path: Path,
    cache: FrozenPopulationCache, gpu_identity: Mapping[str, Any],
    best_iteration: int, iteration_range: tuple[int, int],
) -> dict[str, Any]:
    return {
        "run_id": run_dir.name, "run_kind": RUN_KIND,
        "variant_id": variant_data.variant,
        "variant_classification": variant_data.classification,
        "model_sha256": sha256_file(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "feature_count": len(variant_data.features),
        "feature_list_sha256": _feature_list_sha256(variant_data.features),
        "full_parameter_lock_sha256": FULL_MODEL_PARAMETERS_SHA256,
        "frozen_population_sha256": cache.binding_sha256,
        "imputer_sha256": cache.imputer_sha256,
        "split_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
        "primary_table_sha256": PRIMARY_TABLE_SHA256,
        "source_input_hashes_pre_sha256": sha256_file(
            run_dir / "provenance/source_input_hashes_pre.tsv"
        ),
        "training_configuration_sha256": TRAINING_CONFIGURATION_SHA256,
        "variant_feature_sets_sha256": VARIANT_FEATURE_SETS_SHA256,
        "row_split_manifest_sha256": ROW_SPLIT_MANIFEST_SHA256,
        "scientific_execution_code_manifest_sha256": strict_full_load_file(
            run_dir / "config/run_identity.lock.json"
        )["scientific_execution_code_manifest_sha256"],
        "seed": PRIMARY_MODEL_SEED,
        "decision_threshold": DECISION_THRESHOLD,
        "gpu_name": _gpu_name(gpu_identity),
        "gpu_identity": dict(gpu_identity),
        "best_iteration": int(best_iteration),
        "iteration_range": [int(iteration_range[0]), int(iteration_range[1])],
    }


def _train_one_variant_impl(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    development: DevelopmentPopulation, variant: str,
) -> dict[str, Any]:
    completion_relative = _training_completion_relative(variant)
    if (run_dir / completion_relative).exists():
        result = validate_training_completion(
            run_dir=run_dir, locked=locked, cache=cache,
            development=development, variant=variant, active_prediction=True,
        )
        _ensure_unique_bound_journal_event(
            run_dir=run_dir, event_name="TRAINING_COMPLETION_COMMITTED",
            binding_path=run_dir / completion_relative,
            binding_field="completion_sha256", phase="TRAINING", variant=variant,
            recovered=True, extra={
                "attempt_id": _phase_publication_records(
                    run_dir, "TRAINING", variant,
                ).get(completion_relative, {}).get("attempt_id"),
            },
        )
        _journal_event(run_dir, "COMPLETED_VARIANT_SKIPPED", phase="TRAINING", variant_id=variant)
        return result
    _record_interrupted_attempts(
        run_dir=run_dir, phase="TRAINING", variant=variant,
    )
    _recover_unpublished_paths(
        run_dir=run_dir, phase="TRAINING", variant=variant,
        completion_relative=completion_relative,
        possible_relatives=_training_relatives(variant),
    )
    attempt_id = uuid.uuid4().hex
    staging = _attempt_root(run_dir, "TRAINING", variant, attempt_id)
    staging.mkdir(parents=True)
    _journal_event(run_dir, "TRAINING_ATTEMPT_STARTED", phase="TRAINING", variant_id=variant, attempt_id=attempt_id)
    _progress("FULL_VARIANT_TRAINING", "START", variant=variant, index=f"{TRAINED_VARIANTS.index(variant)+1}/10")
    variant_data = project_variant(cache, locked, variant)
    classifier = xgb.XGBClassifier(**locked.model_parameters)
    gpu_identity = core.gpu_identity_metadata()
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    monitor = legacy.PeakMonitor()
    fit_progress = BoundedFitProgress(run_dir.name, variant)
    caught_messages: list[str] = []
    try:
        _SCIENTIFIC_COUNTERS["training_calls"] += 1
        monitor.__enter__()
        fit_progress.__enter__()
        _journal_event(
            run_dir, "SCIENTIFIC_FIT_CALLED", phase="TRAINING",
            variant_id=variant, attempt_id=attempt_id,
            seed=PRIMARY_MODEL_SEED, stability_fit=False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            classifier.fit(
                variant_data.X, variant_data.y,
                sample_weight=variant_data.weights,
                eval_set=[(variant_data.validation_X, development.validation_labels)],
                verbose=False,
            )
        caught_messages = [str(item.message) for item in caught]
        monitor.__exit__(None, None, None)
    except Exception as exc:
        with suppress(Exception):
            monitor.__exit__(type(exc), exc, exc.__traceback__)
        _journal_event(
            run_dir, "TRAINING_ATTEMPT_FAILED", phase="TRAINING", variant_id=variant,
            attempt_id=attempt_id, error_type=type(exc).__name__, error=str(exc),
        )
        raise
    finally:
        fit_progress.__exit__(*sys.exc_info())
    booster = classifier.get_booster()
    configuration = strict_full_loads(booster.save_config())
    if "cuda" not in json.dumps(configuration).lower() or _fallback_warnings(caught_messages):
        raise Amendment06IntegrityError(f"CUDA execution/fallback gate failed: {variant}")
    best_iteration = int(classifier.best_iteration)
    iteration_range = model_compat.classifier_equivalent_iteration_range(booster)
    feature_sha = _feature_list_sha256(variant_data.features)
    model_compat.embed_binary_classifier_metadata(
        classifier, expected_feature_count=len(variant_data.features),
        feature_list_sha256=feature_sha,
    )
    embedded = {
        "run_id": run_dir.name,
        "variant_id": variant,
        "variant_classification": variant_data.classification,
        "feature_count": str(len(variant_data.features)),
        "feature_list_sha256": feature_sha,
        "full_parameter_lock_sha256": FULL_MODEL_PARAMETERS_SHA256,
        "frozen_population_sha256": cache.binding_sha256,
        "imputer_sha256": cache.imputer_sha256,
        "split_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
        "seed": str(PRIMARY_MODEL_SEED),
        "decision_threshold": str(DECISION_THRESHOLD),
        "gpu_name": _gpu_name(gpu_identity),
        "best_iteration": str(best_iteration),
        "iteration_range": strict_full_canonical_json_bytes(list(iteration_range)).decode("utf-8"),
    }
    booster.set_attr(**embedded)
    model_relative = f"models/{variant}/model.ubj"
    staged_model = staging / model_relative
    staged_model.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(staged_model)
    with staged_model.open("rb") as handle:
        os.fsync(handle.fileno())
    parity = model_compat.four_way_binary_reload_parity(
        classifier, staged_model, variant_data.validation_X,
        expected_feature_count=len(variant_data.features),
        expected_feature_list_sha256=feature_sha, device="cuda",
    )
    parity_differences = parity.evidence.get("maximum_absolute_difference", {})
    parity_maximum = max(map(float, parity_differences.values()), default=math.inf)
    if parity_maximum != 0.0:
        raise Amendment06IntegrityError(f"Four-way validation parity is not bit-exact: {variant}")
    active_probe = core._active_reloaded_booster_cuda_probe(
        parity.reloaded_booster, len(variant_data.features),
    )
    if active_probe.get("status") != "PASS":
        raise Amendment06IntegrityError(f"Reloaded active CUDA probe failed: {variant}")
    probabilities = parity.fitted_classifier_positive
    validation_frame = _training_prediction_frame(
        development=development, probabilities=probabilities,
        variant=variant, population=variant_data.population,
    )
    validation_metrics = _binary_metrics(development.validation_labels, probabilities)
    weighted_validation_metrics = _binary_metrics(
        development.validation_labels, probabilities,
        weights=development.validation_weights,
    )
    csv_bytes = deterministic_csv_gzip_bytes(validation_frame)
    reopened, roundtrip = validate_prediction_roundtrip(csv_bytes, validation_frame)
    recomputed = _binary_metrics(reopened.true_label, reopened.probability)
    if recomputed != validation_metrics:
        raise Amendment06IntegrityError(f"Saved validation metric recomputation differs: {variant}")
    common = _model_common_metadata(
        run_dir=run_dir, variant_data=variant_data, model_path=staged_model,
        cache=cache, gpu_identity=gpu_identity,
        best_iteration=best_iteration, iteration_range=iteration_range,
    )
    peak_gpu_memory_mib = getattr(monitor, "peak_gpu_memory_mib", None)
    if (
        isinstance(peak_gpu_memory_mib, bool)
        or not isinstance(peak_gpu_memory_mib, (int, float))
        or not math.isfinite(float(peak_gpu_memory_mib))
        or float(peak_gpu_memory_mib) < 0.0
    ):
        raise Amendment06IntegrityError(f"Peak GPU-memory evidence is invalid: {variant}")
    timing = {
        "run_id": run_dir.name, "variant_id": variant,
        "wall_seconds": float(time.perf_counter() - start_wall),
        "cpu_seconds": float(time.process_time() - start_cpu),
        "peak_ram_bytes": int(max(before_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024),
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
        "training_rows": len(variant_data.y), "validation_rows": len(development.validation_labels),
        "best_iteration": best_iteration, "boosted_rounds": int(booster.num_boosted_rounds()),
    }
    metadata = {
        **common, "status": "PASS", "requested_device": "cuda",
        "cpu_fallback_detected": False,
        "requested_model_parameters": locked.model_parameters,
        "booster_configuration_sha256": sha256_bytes(strict_full_canonical_json_bytes(configuration)),
        "booster_configuration": configuration,
        "captured_warning_messages": caught_messages,
        "fallback_warnings": _fallback_warnings(caught_messages),
        "population_binding": variant_data.population_evidence,
        "embedded_model_attributes": embedded,
    }
    transform = {
        "run_id": run_dir.name, "variant_id": variant, "status": "PASS",
        "fit_scope": "ONE_TRAINING_ONLY_93D_MEDIAN_IMPUTER",
        "imputer_sha256": cache.imputer_sha256,
        "full_feature_order_sha256": _feature_list_sha256(cache.full_features),
        "retained_feature_count": len(variant_data.features),
        "retained_feature_list_sha256": feature_sha,
        "retained_column_indices": list(variant_data.indices),
        "projection_only": True,
    }
    feature_bytes = ("\n".join(variant_data.features) + "\n").encode("utf-8")
    _stage_bytes(staging, f"models/{variant}/feature_list.txt", feature_bytes)
    _stage_json(staging, f"models/{variant}/model_metadata.json", metadata)
    _stage_json(staging, f"models/{variant}/transform_binding.json", transform)
    _stage_bytes(staging, f"predictions/{variant}_validation.csv.gz", csv_bytes)
    _stage_json(staging, f"timing/{variant}.json", timing)
    _stage_bytes(
        staging, f"logs/{variant}.stdout.log",
        (f"VARIANT={variant}\nPHASE=TRAINING_AND_VALIDATION_COMPLETE\nSTATUS=PASS\n").encode(),
    )
    _stage_bytes(
        staging, f"logs/{variant}.stderr.log",
        (("\n".join(caught_messages) + "\n") if caught_messages else "NO_WARNINGS\n").encode(),
    )
    artifact_records = [
        _publication_record(staging / relative, relative)
        for relative in _training_relatives(variant)
    ]
    completion = {
        **common,
        "phase": "TRAINING_AND_VALIDATION_COMPLETE", "status": "PASS",
        "cpu_fallback_detected": False,
        "four_way_parity_max_abs_diff": parity_maximum,
        "four_way_reload_parity": parity.evidence,
        "active_reloaded_cuda_probe": active_probe,
        "requested_model_parameters": locked.model_parameters,
        "booster_configuration_sha256": sha256_bytes(
            strict_full_canonical_json_bytes(configuration)
        ),
        "captured_warning_messages": caught_messages,
        "fallback_warnings": _fallback_warnings(caught_messages),
        "validation_prediction_sha256": sha256_bytes(csv_bytes),
        "validation_prediction_rows": len(validation_frame),
        "validation_order_sha256": sha256_bytes(_lineage_bytes(validation_frame.stable_candidate_id)),
        "validation_probability_sha256_float32": sha256_array(probabilities.astype("<f4", copy=False)),
        "validation_metrics": validation_metrics,
        "weighted_validation_metrics": weighted_validation_metrics,
        "csv_roundtrip": roundtrip,
        "artifact_records": artifact_records,
        "model_metadata_sha256": sha256_file(
            staging / f"models/{variant}/model_metadata.json"
        ),
        "transform_binding_sha256": sha256_file(
            staging / f"models/{variant}/transform_binding.json"
        ),
        "timing_resource_sha256": sha256_file(staging / f"timing/{variant}.json"),
        "source_code_manifest_sha256": common[
            "scientific_execution_code_manifest_sha256"
        ],
    }
    _stage_json(staging, completion_relative, completion)
    try:
        for relative in _training_relatives(variant):
            _publish_attempt_file(
                run_dir=run_dir, staging=staging, relative=relative,
                phase="TRAINING", variant=variant, attempt_id=attempt_id,
            )
        _publish_attempt_file(
            run_dir=run_dir, staging=staging, relative=completion_relative,
            phase="TRAINING", variant=variant, attempt_id=attempt_id,
        )
    except Exception as exc:
        _journal_event(
            run_dir, "TRAINING_PUBLICATION_INTERRUPTED", phase="TRAINING",
            variant_id=variant, attempt_id=attempt_id,
            error_type=type(exc).__name__, error=str(exc),
        )
        raise
    _ensure_unique_bound_journal_event(
        run_dir=run_dir, event_name="TRAINING_COMPLETION_COMMITTED",
        binding_path=run_dir / completion_relative,
        binding_field="completion_sha256", phase="TRAINING", variant=variant,
        recovered=False, extra={"attempt_id": attempt_id},
    )
    _progress("FULL_VARIANT_TRAINING", "COMPLETE", variant=variant, status="PASS")
    return validate_training_completion(
        run_dir=run_dir, locked=locked, cache=cache,
        development=development, variant=variant, active_prediction=False,
    )


def train_one_variant(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    development: DevelopmentPopulation, variant: str,
) -> dict[str, Any]:
    before = len(_read_journal(run_dir))
    try:
        return _train_one_variant_impl(
            run_dir=run_dir, locked=locked, cache=cache,
            development=development, variant=variant,
        )
    except Exception as exc:
        events = _read_journal(run_dir)[before:]
        starts = [event for event in events if event.get("event") == "TRAINING_ATTEMPT_STARTED"]
        if starts:
            attempt_id = str(starts[-1]["attempt_id"])
            already_recorded = any(
                event.get("event") == "TRAINING_ATTEMPT_FAILED"
                and event.get("attempt_id") == attempt_id
                for event in events
            )
            if not already_recorded:
                _journal_event(
                    run_dir, "TRAINING_ATTEMPT_FAILED", phase="TRAINING",
                    failure_phase="POST_START_THROUGH_COMPLETION_COMMIT",
                    variant_id=variant, attempt_id=attempt_id,
                    error_type=type(exc).__name__, error=str(exc),
                )
        raise


def _expected_variant_population_binding(
    *, cache: FrozenPopulationCache, locked: LockedInputs, variant: str,
) -> dict[str, Any]:
    features = tuple(locked.variant_features[variant]["retained_features"])
    expected_count, feature_sha = VARIANT_FEATURE_CONTRACT[variant]
    if len(features) != expected_count or _feature_list_sha256(features) != feature_sha:
        raise Amendment06IntegrityError(f"Variant feature contract drifted: {variant}")
    use_pre = variant == "no_safe_smote"
    stage_name = "post_jitter_pre_smote" if use_pre else "post_safe_smote"
    stage = next(
        (item for item in cache.manifest["stages"] if item.get("stage") == stage_name),
        None,
    )
    if not isinstance(stage, Mapping):
        raise Amendment06IntegrityError(f"Frozen population stage is absent: {stage_name}")
    rows = PRE_SMOTE_ROWS if use_pre else POST_SMOTE_ROWS
    weight_sha256 = (
        sha256_array(np.ones(rows, dtype=np.float32))
        if variant == "no_review_aware_training_weights"
        else str(stage["weight_sha256"])
    )
    return {
        "variant": variant, "source_stage": stage_name, "projection_only": True,
        "rows": rows, "feature_count": expected_count,
        "feature_list_sha256": feature_sha,
        "full_space_X_sha256": stage["X_sha256"],
        "y_sha256": stage["y_sha256"],
        "source_lineage_sha256": stage["source_lineage_sha256"],
        "row_order_sha256": stage["row_order_sha256"],
        "weight_sha256": weight_sha256,
        "review_weights_enabled": variant != "no_review_aware_training_weights",
        "safe_smote_enabled": not use_pre,
        "safe_smote_synthetic_rows": 0 if use_pre else 147_841,
    }


def validate_training_completion(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    development: DevelopmentPopulation, variant: str,
    active_prediction: bool = False,
) -> dict[str, Any]:
    if variant not in TRAINED_VARIANTS:
        raise Amendment06IntegrityError(f"Unauthorized completed Variant: {variant}")
    completion_path = run_dir / _training_completion_relative(variant)
    if not _safe_regular(completion_path):
        raise Amendment06IntegrityError(f"Training completion is absent: {variant}")
    completion = strict_full_load_file(completion_path)
    expected_classification = "EXPLORATORY" if variant == EXPLORATORY_VARIANT else "OFFICIAL_RUNNABLE"
    model_path = run_dir / f"models/{variant}/model.ubj"
    feature_path = run_dir / f"models/{variant}/feature_list.txt"
    metadata_path = run_dir / f"models/{variant}/model_metadata.json"
    transform_path = run_dir / f"models/{variant}/transform_binding.json"
    timing_path = run_dir / f"timing/{variant}.json"
    stderr_path = run_dir / f"logs/{variant}.stderr.log"
    stdout_path = run_dir / f"logs/{variant}.stdout.log"
    for required in (
        model_path, feature_path, metadata_path, transform_path, timing_path,
        stderr_path, stdout_path,
    ):
        if not _safe_regular(required):
            raise Amendment06IntegrityError(f"Completed training member is absent/unsafe: {required}")
    features = tuple(feature_path.read_text(encoding="utf-8").splitlines())
    expected_count, expected_feature_sha = VARIANT_FEATURE_CONTRACT[variant]
    identity = strict_full_load_file(run_dir / "config/run_identity.lock.json")
    scientific_manifest_sha = sha256_file(
        run_dir / "provenance/scientific_execution_code_manifest.tsv"
    )
    source_pre_sha = sha256_file(run_dir / "provenance/source_input_hashes_pre.tsv")
    population_binding = _expected_variant_population_binding(
        cache=cache, locked=locked, variant=variant,
    )
    expected_common = {
        "run_id": run_dir.name,
        "run_kind": RUN_KIND,
        "variant_id": variant,
        "variant_classification": expected_classification,
        "model_sha256": sha256_file(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "feature_count": expected_count,
        "feature_list_sha256": expected_feature_sha,
        "full_parameter_lock_sha256": FULL_MODEL_PARAMETERS_SHA256,
        "frozen_population_sha256": cache.binding_sha256,
        "imputer_sha256": cache.imputer_sha256,
        "split_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
        "primary_table_sha256": PRIMARY_TABLE_SHA256,
        "source_input_hashes_pre_sha256": source_pre_sha,
        "training_configuration_sha256": TRAINING_CONFIGURATION_SHA256,
        "variant_feature_sets_sha256": VARIANT_FEATURE_SETS_SHA256,
        "row_split_manifest_sha256": ROW_SPLIT_MANIFEST_SHA256,
        "scientific_execution_code_manifest_sha256": scientific_manifest_sha,
        "seed": PRIMARY_MODEL_SEED,
        "decision_threshold": DECISION_THRESHOLD,
    }
    if not (
        all(completion.get(key) == value for key, value in expected_common.items())
        and identity.get("scientific_execution_code_manifest_sha256") == scientific_manifest_sha
        and completion.get("phase") == "TRAINING_AND_VALIDATION_COMPLETE"
        and completion.get("status") == "PASS"
        and completion.get("cpu_fallback_detected") is False
        and completion.get("four_way_parity_max_abs_diff") == 0.0
        and len(features) == expected_count and _feature_list_sha256(features) == expected_feature_sha
        and completion.get("requested_model_parameters") == locked.model_parameters
        and completion.get("source_code_manifest_sha256") == scientific_manifest_sha
    ):
        raise Amendment06IntegrityError(f"Training completion binding differs: {variant}")
    _validate_exact_artifact_records(
        run_dir=run_dir, value=completion.get("artifact_records"),
        expected_relatives=_training_relatives(variant),
        label=f"Completed training {variant}",
    )
    metadata = strict_full_load_file(metadata_path)
    transform = strict_full_load_file(transform_path)
    timing = strict_full_load_file(timing_path)
    booster_configuration = metadata.get("booster_configuration")
    warning_messages = metadata.get("captured_warning_messages")
    parity = completion.get("four_way_reload_parity")
    active_probe = completion.get("active_reloaded_cuda_probe")
    if not (
        all(metadata.get(key) == value for key, value in expected_common.items())
        and metadata.get("status") == "PASS"
        and metadata.get("requested_device") == "cuda"
        and metadata.get("cpu_fallback_detected") is False
        and metadata.get("requested_model_parameters") == locked.model_parameters
        and isinstance(booster_configuration, Mapping)
        and "cuda" in json.dumps(booster_configuration).lower()
        and metadata.get("booster_configuration_sha256")
        == sha256_bytes(strict_full_canonical_json_bytes(booster_configuration))
        and metadata.get("population_binding") == population_binding
        and isinstance(warning_messages, list)
        and all(isinstance(value, str) for value in warning_messages)
        and metadata.get("fallback_warnings") == []
        and completion.get("captured_warning_messages") == warning_messages
        and completion.get("fallback_warnings") == []
        and completion.get("booster_configuration_sha256")
        == metadata.get("booster_configuration_sha256")
        and completion.get("model_metadata_sha256") == sha256_file(metadata_path)
        and completion.get("transform_binding_sha256") == sha256_file(transform_path)
        and completion.get("timing_resource_sha256") == sha256_file(timing_path)
        and isinstance(parity, Mapping)
        and parity.get("status") == "PASS"
        and parity.get("classification") == "VERIFIED"
        and parity.get("device") == "cuda"
        and parity.get("feature_list_sha256") == expected_feature_sha
        and parity.get("iteration_range") == completion.get("iteration_range")
        and parity.get("cpu_fallback_detected") is False
        and parity.get("model_sha256_before") == completion.get("model_sha256")
        and parity.get("model_sha256_after") == completion.get("model_sha256")
        and parity.get("model_bytes_unchanged") is True
        and isinstance(parity.get("maximum_absolute_difference"), Mapping)
        and set(parity["maximum_absolute_difference"]) == {
            "fitted_classifier", "fitted_booster", "reloaded_booster",
            "compatibility_classifier",
        }
        and set(parity["maximum_absolute_difference"].values()) == {0.0}
        and isinstance(parity.get("bit_exact_against_fitted_classifier"), Mapping)
        and all(value is True for value in parity["bit_exact_against_fitted_classifier"].values())
        and isinstance(parity.get("warnings"), Mapping)
        and all(value == [] for value in parity["warnings"].values())
        and isinstance(active_probe, Mapping)
        and active_probe.get("status") == "PASS"
        and "cuda" in str(active_probe.get("device", "")).lower()
        and active_probe.get("warnings") == []
    ):
        raise Amendment06IntegrityError(f"Completed training evidence differs: {variant}")
    expected_transform = {
        "run_id": run_dir.name, "variant_id": variant, "status": "PASS",
        "fit_scope": "ONE_TRAINING_ONLY_93D_MEDIAN_IMPUTER",
        "imputer_sha256": cache.imputer_sha256,
        "full_feature_order_sha256": _feature_list_sha256(cache.full_features),
        "retained_feature_count": expected_count,
        "retained_feature_list_sha256": expected_feature_sha,
        "retained_column_indices": [
            cache.full_features.index(feature) for feature in features
        ],
        "projection_only": True,
    }
    numeric_timing = (
        "wall_seconds", "cpu_seconds", "peak_ram_bytes", "peak_gpu_memory_mib",
    )
    if not (
        transform == expected_transform
        and timing.get("run_id") == run_dir.name
        and timing.get("variant_id") == variant
        and timing.get("training_rows") == population_binding["rows"]
        and timing.get("validation_rows") == 56_843
        and timing.get("best_iteration") == completion.get("best_iteration")
        and timing.get("boosted_rounds") >= completion.get("best_iteration") + 1
        and all(
            isinstance(timing.get(key), (int, float))
            and not isinstance(timing.get(key), bool)
            and math.isfinite(float(timing[key])) and float(timing[key]) >= 0.0
            for key in numeric_timing
        )
        and stdout_path.read_bytes() == (
            f"VARIANT={variant}\nPHASE=TRAINING_AND_VALIDATION_COMPLETE\nSTATUS=PASS\n"
        ).encode("utf-8")
        and stderr_path.read_text(encoding="utf-8") == (
            ("\n".join(warning_messages) + "\n") if warning_messages else "NO_WARNINGS\n"
        )
        and not _fallback_warnings(warning_messages)
    ):
        raise Amendment06IntegrityError(f"Completed transform/timing/warnings differ: {variant}")
    booster = xgb.Booster()
    booster.load_model(model_path)
    attributes = booster.attributes()
    required_attributes = {
        "run_id": run_dir.name, "variant_id": variant,
        "variant_classification": expected_classification,
        "feature_count": str(expected_count), "feature_list_sha256": expected_feature_sha,
        "full_parameter_lock_sha256": FULL_MODEL_PARAMETERS_SHA256,
        "frozen_population_sha256": cache.binding_sha256,
        "imputer_sha256": cache.imputer_sha256,
        "split_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
        "seed": str(PRIMARY_MODEL_SEED), "decision_threshold": str(DECISION_THRESHOLD),
        "gpu_name": str(completion["gpu_name"]),
        "best_iteration": str(completion["best_iteration"]),
        "iteration_range": strict_full_canonical_json_bytes(completion["iteration_range"]).decode(),
    }
    if any(attributes.get(key) != value for key, value in required_attributes.items()):
        raise Amendment06IntegrityError(f"Completed UBJ embedded attributes drifted: {variant}")
    reloaded_configuration = strict_full_loads(booster.save_config())
    if (
        sha256_bytes(strict_full_canonical_json_bytes(reloaded_configuration))
        != completion.get("booster_configuration_sha256")
        or "cuda" not in json.dumps(reloaded_configuration).lower()
    ):
        raise Amendment06IntegrityError(f"Completed UBJ configuration differs: {variant}")
    prediction_path = run_dir / f"predictions/{variant}_validation.csv.gz"
    authoritative = pd.read_csv(prediction_path, engine="c", float_precision="round_trip")
    authoritative = authoritative.loc[:, list(PREDICTION_COLUMNS)]
    reopened, roundtrip = validate_prediction_roundtrip(prediction_path, authoritative)
    if not (
        len(reopened) == len(development.validation)
        and np.array_equal(
            reopened.stable_candidate_id.astype(str).to_numpy(),
            development.validation.stable_candidate_id.astype(str).to_numpy(),
        )
        and np.array_equal(reopened.true_label.to_numpy(np.int8), development.validation_labels)
        and np.array_equal(
            reopened.card_id.astype(str).to_numpy(),
            development.validation.card_id.astype(str).to_numpy(),
        )
        and np.array_equal(
            reopened.review_weight.to_numpy(np.float64),
            development.validation_weights.astype(np.float64),
        )
        and np.array_equal(
            reopened.prediction_at_0_5.to_numpy(np.int8),
            (reopened.probability.to_numpy(np.float64) >= DECISION_THRESHOLD).astype(np.int8),
        )
        and np.array_equal(reopened.row_order.to_numpy(np.int64), np.arange(len(reopened)))
        and set(reopened.variant_id.astype(str)) == {variant}
        and set(reopened.split_population.astype(str)) == {
            f"validation:{population_binding['source_stage']}"
        }
        and completion.get("validation_prediction_sha256") == sha256_file(prediction_path)
        and completion.get("validation_prediction_rows") == len(reopened)
        and completion.get("validation_order_sha256")
        == sha256_bytes(_lineage_bytes(reopened.stable_candidate_id.astype(str)))
        and completion.get("validation_probability_sha256_float32")
        == sha256_array(reopened.probability.to_numpy(np.float32).astype("<f4", copy=False))
        and _binary_metrics(reopened.true_label, reopened.probability) == completion.get("validation_metrics")
        and _binary_metrics(
            reopened.true_label, reopened.probability, weights=reopened.review_weight,
        ) == completion.get("weighted_validation_metrics")
        and completion.get("csv_roundtrip") == roundtrip
        and roundtrip.get("mismatch_count") == 0
    ):
        raise Amendment06IntegrityError(f"Completed validation predictions/metrics drifted: {variant}")
    active_evidence: dict[str, Any] | None = None
    if active_prediction:
        indices = tuple(cache.full_features.index(feature) for feature in features)
        matrix_host = (
            cache.validation_X if indices == tuple(range(93))
            else np.ascontiguousarray(cache.validation_X[:, indices], dtype=np.float32)
        )
        loaded, raw_booster, load_evidence = load_independent_external_predictors(
            model_path,
            expected_feature_count=expected_count,
            expected_feature_list_sha256=expected_feature_sha,
            expected_model_sha256=completion["model_sha256"],
        )
        import cupy as cp
        matrix = cp.asarray(np.ascontiguousarray(matrix_host, dtype=np.float32))
        interval = tuple(map(int, completion["iteration_range"]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wrapper = cp.asnumpy(
                loaded.classifier.predict_proba(matrix, iteration_range=interval)[:, 1]
            )
            raw = cp.asnumpy(raw_booster.inplace_predict(matrix, iteration_range=interval))
        active_warnings = [str(item.message) for item in caught]
        live_probe = core._active_reloaded_booster_cuda_probe(raw_booster, expected_count)
        if not (np.array_equal(wrapper, raw) and np.array_equal(wrapper, reopened.probability.to_numpy(np.float32))):
            raise Amendment06IntegrityError(f"Completed model active prediction differs: {variant}")
        if _fallback_warnings(active_warnings) or live_probe.get("status") != "PASS":
            raise Amendment06IntegrityError(f"Completed model active CUDA probe differs: {variant}")
        active_evidence = {
            "status": "PASS", "maximum_absolute_difference": 0.0,
            "independent_load_evidence": load_evidence,
            "active_reloaded_cuda_probe": live_probe,
            "warnings": active_warnings,
        }
    return {
        "status": "PASS", "variant_id": variant,
        "completion_sha256": sha256_file(completion_path),
        "model_sha256": completion["model_sha256"],
        "active_prediction": active_evidence,
    }


def ensure_blocked_no_pca(run_dir: Path) -> None:
    status_path = run_dir / "models/no_pca/STATUS.txt"
    evidence_path = run_dir / "models/no_pca/blocked_evidence.json"
    status = (
        f"RUN_ID={run_dir.name}\nVARIANT_ID=no_pca\n"
        "STATUS=NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE\n"
        "MODEL_CREATED=NO\n"
    ).encode("utf-8")
    evidence = {
        "run_id": run_dir.name, "run_kind": RUN_KIND,
        "variant_id": BLOCKED_VARIANT,
        "variant_classification": "OFFICIAL_BLOCKED",
        "status": "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        "model_created": False, "matrix_created": False,
        "prediction_created": False, "metric_created": False,
        "reason": "RAW_EMBEDDINGS_UNAVAILABLE_AND_NEW_RECONSTRUCTION_NOT_AUTHORIZED",
        "no_deep_pca_features_is_not_true_no_pca": True,
    }
    for path, data in ((status_path, status), (evidence_path, strict_full_json_bytes(evidence))):
        if path.exists():
            if not _safe_regular(path, sha256_bytes(data)):
                raise Amendment06IntegrityError(f"Blocked no-PCA evidence drifted: {path}")
        else:
            publish_bytes_no_clobber(path, data)


_GUARDED_ORIGINALS: list[tuple[Any, str, Any]] = []


def _install_post_freeze_guards() -> None:
    global _POST_FREEZE_GUARDS_INSTALLED
    if _POST_FREEZE_GUARDS_INSTALLED:
        return

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise PostFreezeScientificCallError(
            "Scientific fit/resampling/imputer-fit API is forbidden after FULL_MODELS_FROZEN"
        )

    targets = (
        (xgb.XGBClassifier, "fit"),
        (xgb, "train"),
        (xgb, "cv"),
        (xgb.Booster, "update"),
        (xgb.Booster, "boost"),
        (SimpleImputer, "fit"),
        (SimpleImputer, "fit_transform"),
        (paired_resampling, "build_frozen_resampling_population"),
        (core, "augment_with_lineage"),
        (core, "augmentation_plan_counts"),
        (core, "replay_augmentation_weights"),
        (legacy, "apply_training_augmentation"),
        (legacy, "safe_smote_resample"),
        (legacy.SafeSMOTE, "fit_resample"),
    )
    for owner, name in targets:
        if not hasattr(owner, name):
            raise Amendment06IntegrityError(f"Freeze-guard target is absent: {owner}.{name}")
        _GUARDED_ORIGINALS.append((owner, name, getattr(owner, name)))
        setattr(owner, name, forbidden)
    _POST_FREEZE_GUARDS_INSTALLED = True


def validate_models_frozen_manifest(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    development: DevelopmentPopulation,
) -> dict[str, Any]:
    path = run_dir / "provenance/full_models_frozen_manifest.json"
    payload = strict_full_load_file(path)
    models = payload.get("models")
    events = _read_journal(run_dir)
    fit_attempts = sum(event.get("event") == "SCIENTIFIC_FIT_CALLED" for event in events)
    failed_attempts = sum(event.get("event") == "TRAINING_ATTEMPT_FAILED" for event in events)
    if not (
        payload.get("run_id") == run_dir.name
        and payload.get("run_kind") == RUN_KIND
        and payload.get("phase") == "FULL_MODELS_FROZEN"
        and payload.get("status") == "PASS"
        and payload.get("trained_model_count") == 10
        and payload.get("official_runnable_model_count") == 9
        and payload.get("exploratory_model_count") == 1
        and payload.get("true_no_pca_model_count") == 0
        and payload.get("frozen_population_sha256") == cache.binding_sha256
        and payload.get("imputer_sha256") == cache.imputer_sha256
        and payload.get("full_parameter_lock_sha256") == FULL_MODEL_PARAMETERS_SHA256
        and payload.get("split_assignment_sha256") == SPLIT_ASSIGNMENT_SHA256
        and payload.get("scientific_training_calls") == 10
        and payload.get("scientific_fit_attempts") == fit_attempts
        and payload.get("failed_unpublished_training_attempts") == failed_attempts
        and fit_attempts >= 10
        and payload.get("external_audit_opened") is False
        and isinstance(models, list) and [item.get("variant_id") for item in models] == list(TRAINED_VARIANTS)
    ):
        raise Amendment06IntegrityError("Full models-frozen manifest schema/count differs")
    for item in models:
        variant = str(item["variant_id"])
        expected_classification = (
            "EXPLORATORY" if variant == EXPLORATORY_VARIANT else "OFFICIAL_RUNNABLE"
        )
        if set(item) != {
            "variant_id", "variant_classification", "model_sha256",
            "training_completion_sha256",
        } or item.get("variant_classification") != expected_classification:
            raise Amendment06IntegrityError("Models-frozen member schema/classification differs")
        result = validate_training_completion(
            run_dir=run_dir, locked=locked, cache=cache,
            development=development, variant=variant,
            active_prediction=True,
        )
        if not (
            result["model_sha256"] == item.get("model_sha256")
            and result["completion_sha256"] == item.get("training_completion_sha256")
        ):
            raise Amendment06IntegrityError("Models-frozen member binding differs")
    _ensure_unique_bound_journal_event(
        run_dir=run_dir, event_name="MODELS_FROZEN",
        binding_path=path, binding_field="manifest_sha256", recovered=True,
        extra={"status": "PASS", "model_count": 10},
    )
    return payload


def freeze_all_models(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    development: DevelopmentPopulation,
) -> dict[str, Any]:
    _progress("FULL_MODELS_FREEZE", "START")
    path = run_dir / "provenance/full_models_frozen_manifest.json"
    manifest_created_now = not path.exists()
    if not manifest_created_now:
        payload = validate_models_frozen_manifest(
            run_dir=run_dir, locked=locked, cache=cache, development=development,
        )
    else:
        models: list[dict[str, Any]] = []
        for variant in TRAINED_VARIANTS:
            result = validate_training_completion(
                run_dir=run_dir, locked=locked, cache=cache,
                development=development, variant=variant, active_prediction=True,
            )
            models.append({
                "variant_id": variant,
                "variant_classification": (
                    "EXPLORATORY" if variant == EXPLORATORY_VARIANT else "OFFICIAL_RUNNABLE"
                ),
                "model_sha256": result["model_sha256"],
                "training_completion_sha256": result["completion_sha256"],
            })
        payload = {
            "run_id": run_dir.name, "run_kind": RUN_KIND,
            "phase": "FULL_MODELS_FROZEN", "status": "PASS",
            "frozen_at": _now(), "trained_model_count": 10,
            "official_runnable_model_count": 9, "exploratory_model_count": 1,
            "true_no_pca_model_count": 0,
            "frozen_population_sha256": cache.binding_sha256,
            "imputer_sha256": cache.imputer_sha256,
            "full_parameter_lock_sha256": FULL_MODEL_PARAMETERS_SHA256,
            "split_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
            "scientific_training_calls": 10,
            "scientific_fit_attempts": sum(
                event.get("event") == "SCIENTIFIC_FIT_CALLED"
                for event in _read_journal(run_dir)
            ),
            "failed_unpublished_training_attempts": sum(
                event.get("event") == "TRAINING_ATTEMPT_FAILED"
                for event in _read_journal(run_dir)
            ),
            "external_audit_opened": False,
            "models": models,
        }
        publish_strict_json_no_clobber(path, payload)
    current = _current_run_state(run_dir)
    if current == "FULL_IN_PROGRESS":
        _set_run_state(run_dir, "FULL_MODELS_FROZEN")
    elif RUN_PHASES.index(current) < RUN_PHASES.index("FULL_MODELS_FROZEN"):
        raise Amendment06IntegrityError("Models-frozen manifest/state disagree")
    _install_post_freeze_guards()
    _ensure_unique_bound_journal_event(
        run_dir=run_dir, event_name="MODELS_FROZEN",
        binding_path=path, binding_field="manifest_sha256",
        recovered=not manifest_created_now,
        extra={"status": "PASS", "model_count": 10},
    )
    _progress("FULL_MODELS_FREEZE", "COMPLETE", status="PASS")
    return payload


@dataclass(frozen=True)
class ExternalAuditPopulation:
    frame: pd.DataFrame
    full_X: np.ndarray
    labels: np.ndarray
    weights: np.ndarray


def _frozen_imputer_transform(cache: FrozenPopulationCache, frame: pd.DataFrame) -> np.ndarray:
    if tuple(frame.columns) != cache.full_features:
        raise Amendment06IntegrityError("Frozen imputer transform received wrong feature order")
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise Amendment06IntegrityError("Frozen imputer transform received infinity")
    medians = np.asarray(cache.imputer["statistics"], dtype=np.float64)
    missing = np.isnan(values)
    if missing.any():
        values[missing] = np.take(medians, np.nonzero(missing)[1])
    transformed = np.ascontiguousarray(values, dtype=np.float32)
    if transformed.shape != (len(frame), 93) or not np.isfinite(transformed).all():
        raise Amendment06IntegrityError("Frozen imputer transform output is invalid")
    return transformed


def open_external_audit_after_freeze(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    development: DevelopmentPopulation,
) -> ExternalAuditPopulation:
    if RUN_PHASES.index(_current_run_state(run_dir)) < RUN_PHASES.index("FULL_MODELS_FROZEN"):
        raise Amendment06IntegrityError("External Audit cannot open before FULL_MODELS_FROZEN")
    validate_models_frozen_manifest(
        run_dir=run_dir, locked=locked, cache=cache, development=development,
    )
    _install_post_freeze_guards()
    evidence_path = run_dir / "provenance/external_audit_exclusion_and_opening.json"
    evidence_created_now = not evidence_path.exists()
    if evidence_created_now:
        evidence = {
            "run_id": run_dir.name, "run_kind": RUN_KIND, "status": "PASS",
            "external_audit_path": str(EXTERNAL_AUDIT),
            "external_audit_sha256": EXTERNAL_AUDIT_SHA256,
            "external_audit_size_bytes": EXTERNAL_AUDIT.stat().st_size,
            "external_audit_opened_at": _now(),
            "opened_after_full_models_frozen": True,
            "models_frozen_manifest_sha256": sha256_file(
                run_dir / "provenance/full_models_frozen_manifest.json"
            ),
            "models_frozen_state_marker_sha256": sha256_file(
                _state_marker_path(run_dir, "FULL_MODELS_FROZEN")
            ),
            "pre_freeze_audit_access": "EXISTENCE_SIZE_SHA256_ONLY_NO_PARSE",
            "external_audit_used_for_fitting": False,
            "external_audit_used_for_imputer_fit": False,
            "external_audit_used_for_early_stopping": False,
            "external_audit_used_for_threshold_or_model_selection": False,
            "post_audit_training_calls": 0,
        }
        publish_strict_json_no_clobber(evidence_path, evidence)
    else:
        evidence = strict_full_load_file(evidence_path)
        if not (
            evidence.get("run_id") == run_dir.name
            and evidence.get("run_kind") == RUN_KIND
            and evidence.get("status") == "PASS"
            and evidence.get("external_audit_path") == str(EXTERNAL_AUDIT)
            and evidence.get("external_audit_sha256") == EXTERNAL_AUDIT_SHA256
            and evidence.get("external_audit_size_bytes") == EXTERNAL_AUDIT.stat().st_size
            and isinstance(evidence.get("external_audit_opened_at"), str)
            and evidence.get("models_frozen_manifest_sha256") == sha256_file(
                run_dir / "provenance/full_models_frozen_manifest.json"
            )
            and evidence.get("models_frozen_state_marker_sha256") == sha256_file(
                _state_marker_path(run_dir, "FULL_MODELS_FROZEN")
            )
            and evidence.get("opened_after_full_models_frozen") is True
            and evidence.get("pre_freeze_audit_access")
            == "EXISTENCE_SIZE_SHA256_ONLY_NO_PARSE"
            and evidence.get("external_audit_used_for_fitting") is False
            and evidence.get("external_audit_used_for_imputer_fit") is False
            and evidence.get("external_audit_used_for_early_stopping") is False
            and evidence.get("external_audit_used_for_threshold_or_model_selection") is False
            and evidence.get("post_audit_training_calls") == 0
        ):
            raise Amendment06IntegrityError("External Audit opening evidence differs")
    _ensure_unique_bound_journal_event(
        run_dir=run_dir, event_name="EXTERNAL_AUDIT_OPENED",
        binding_path=evidence_path, binding_field="evidence_sha256",
        recovered=not evidence_created_now, extra={"status": "PASS"},
    )
    _progress("EXTERNAL_AUDIT_OPEN", "START")
    audit = core._clean_canonical_audit_frame(EXTERNAL_AUDIT, EXTERNAL_AUDIT_SHA256)
    if not (
        len(audit) == 10_097 and int(audit.human_label.sum()) == 2_032
        and audit.card_id.nunique() == 10
        and set(audit.card_id.astype(str)).isdisjoint(locked.split.card_id.astype(str))
    ):
        raise Amendment06IntegrityError("Locked external Audit population/count differs")
    harmonized = core.harmonize_audit_features(audit, locked.full_features)
    full_X = _frozen_imputer_transform(cache, harmonized.loc[:, locked.full_features])
    labels = audit.human_label.to_numpy(np.int8)
    if "review_weight" in audit:
        weights = pd.to_numeric(audit.review_weight, errors="coerce").fillna(1.0).to_numpy(np.float64)
    else:
        weights = np.ones(len(audit), dtype=np.float64)
    if not (np.isfinite(weights).all() and np.all(weights > 0)):
        raise Amendment06IntegrityError("External Audit sensitivity weights are invalid")
    _progress("EXTERNAL_AUDIT_OPEN", "COMPLETE", rows=len(audit), cards=audit.card_id.nunique())
    return ExternalAuditPopulation(audit, full_X, labels, weights)


def _external_prediction_frame(
    *, audit: ExternalAuditPopulation, probabilities: np.ndarray, variant: str,
) -> pd.DataFrame:
    return pd.DataFrame({
        "stable_candidate_id": audit.frame.stable_candidate_id.astype(str).to_numpy(),
        "card_id": audit.frame.card_id.astype(str).to_numpy(),
        "true_label": audit.labels,
        "probability": np.asarray(probabilities, dtype=np.float32).astype(np.float64),
        "prediction_at_0_5": (np.asarray(probabilities) >= DECISION_THRESHOLD).astype(np.int8),
        "review_weight": audit.weights,
        "variant_id": variant,
        "split_population": "external_audit:locked_10_card",
        "row_order": np.arange(len(probabilities), dtype=np.int64),
    }, columns=PREDICTION_COLUMNS)


def _scoring_relatives(variant: str) -> tuple[str, ...]:
    return (
        f"predictions/{variant}_external_audit.csv.gz",
        f"metrics/{variant}.json",
        f"metrics/{variant}_per_card.csv",
        f"models/{variant}/STATUS.txt",
    )


def _scoring_completion_relative(variant: str) -> str:
    return f"models/{variant}/external_scoring_manifest.json"


def _variant_complete_status(run_id: str, variant: str) -> bytes:
    return (
        f"RUN_ID={run_id}\nVARIANT_ID={variant}\n"
        "TRAINING_AND_VALIDATION_COMPLETE\n"
        "EXTERNAL_AUDIT_SCORING_COMPLETE\nSTATUS=PASS\n"
    ).encode("utf-8")


def load_independent_external_predictors(
    model_path: Path, *, expected_feature_count: int,
    expected_feature_list_sha256: str, expected_model_sha256: str,
) -> tuple[model_compat.LoadedBinaryClassifier, xgb.Booster, dict[str, Any]]:
    model_path = Path(model_path)
    before = sha256_file(model_path)
    if before != expected_model_sha256:
        raise Amendment06IntegrityError("External scoring model differs before independent loads")
    compatibility = model_compat.load_binary_classifier_compat(
        model_path,
        expected_feature_count=expected_feature_count,
        expected_feature_list_sha256=expected_feature_list_sha256,
        expected_model_sha256=expected_model_sha256,
        device="cuda",
    )
    raw_booster = xgb.Booster()
    raw_booster.load_model(model_path)
    raw_booster.set_param({"device": "cuda"})
    after = sha256_file(model_path)
    if after != before:
        raise Amendment06IntegrityError("Independent external model loads changed UBJ bytes")
    if raw_booster is compatibility.classifier.get_booster():
        raise Amendment06IntegrityError("Raw Booster and compatibility classifier are not independent loads")
    return compatibility, raw_booster, {
        "status": "PASS", "compatibility_loader": "XGBClassifier.load_model",
        "raw_booster_loader": "INDEPENDENT_XGB_BOOSTER_LOAD_MODEL",
        "independent_objects": True,
        "model_sha256_before": before, "model_sha256_after": after,
    }


def _score_one_external_variant_impl(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    development: DevelopmentPopulation, audit: ExternalAuditPopulation,
    variant: str,
) -> dict[str, Any]:
    completion_relative = _scoring_completion_relative(variant)
    if (run_dir / completion_relative).exists():
        result = validate_external_scoring_completion(run_dir=run_dir, audit=audit, variant=variant)
        _ensure_unique_bound_journal_event(
            run_dir=run_dir, event_name="AUDIT_SCORING_COMPLETION_COMMITTED",
            binding_path=run_dir / completion_relative,
            binding_field="completion_sha256", phase="AUDIT_SCORING",
            variant=variant, recovered=True, extra={
                "attempt_id": _phase_publication_records(
                    run_dir, "AUDIT_SCORING", variant,
                ).get(completion_relative, {}).get("attempt_id"),
            },
        )
        _journal_event(run_dir, "COMPLETED_AUDIT_SCORING_SKIPPED", phase="AUDIT_SCORING", variant_id=variant)
        return result
    _record_interrupted_attempts(
        run_dir=run_dir, phase="AUDIT_SCORING", variant=variant,
    )
    _recover_unpublished_paths(
        run_dir=run_dir, phase="AUDIT_SCORING", variant=variant,
        completion_relative=completion_relative,
        possible_relatives=_scoring_relatives(variant),
    )
    validate_training_completion(
        run_dir=run_dir, locked=locked, cache=cache,
        development=development, variant=variant, active_prediction=False,
    )
    attempt_id = uuid.uuid4().hex
    staging = _attempt_root(run_dir, "AUDIT_SCORING", variant, attempt_id)
    staging.mkdir(parents=True)
    _journal_event(run_dir, "AUDIT_SCORING_ATTEMPT_STARTED", phase="AUDIT_SCORING", variant_id=variant, attempt_id=attempt_id)
    _progress("EXTERNAL_AUDIT_VARIANT_SCORING", "START", variant=variant, index=f"{TRAINED_VARIANTS.index(variant)+1}/10")
    features = tuple(locked.variant_features[variant]["retained_features"])
    indices = tuple(cache.full_features.index(feature) for feature in features)
    matrix = audit.full_X if indices == tuple(range(93)) else np.ascontiguousarray(audit.full_X[:, indices], dtype=np.float32)
    training_completion = strict_full_load_file(run_dir / _training_completion_relative(variant))
    model_path = run_dir / f"models/{variant}/model.ubj"
    loaded, independent_raw_booster, independent_load = load_independent_external_predictors(
        model_path,
        expected_feature_count=len(features),
        expected_feature_list_sha256=_feature_list_sha256(features),
        expected_model_sha256=training_completion["model_sha256"],
    )
    interval = tuple(map(int, training_completion["iteration_range"]))
    import cupy as cp
    device_matrix = cp.asarray(matrix)
    with warnings.catch_warnings(record=True) as wrapper_caught:
        warnings.simplefilter("always")
        wrapper_probability = loaded.classifier.predict_proba(
            device_matrix, iteration_range=interval,
        )[:, 1]
    with warnings.catch_warnings(record=True) as raw_caught:
        warnings.simplefilter("always")
        raw_probability = independent_raw_booster.inplace_predict(
            device_matrix, iteration_range=interval,
        )
    wrapper_host = np.ascontiguousarray(cp.asnumpy(wrapper_probability), dtype=np.float32)
    raw_host = np.ascontiguousarray(cp.asnumpy(raw_probability), dtype=np.float32)
    warning_text = [str(item.message) for item in (*wrapper_caught, *raw_caught)]
    if _fallback_warnings(warning_text) or not np.array_equal(wrapper_host, raw_host):
        raise Amendment06IntegrityError(f"External raw/compat CUDA parity failed: {variant}")
    active_probe = core._active_reloaded_booster_cuda_probe(
        independent_raw_booster, len(features),
    )
    if active_probe.get("status") != "PASS":
        raise Amendment06IntegrityError(f"External active CUDA probe failed: {variant}")
    prediction = _external_prediction_frame(audit=audit, probabilities=wrapper_host, variant=variant)
    raw_metrics = _binary_metrics(audit.labels, wrapper_host)
    weighted_metrics = _binary_metrics(audit.labels, wrapper_host, weights=audit.weights)
    csv_bytes = deterministic_csv_gzip_bytes(prediction)
    reopened, roundtrip = validate_prediction_roundtrip(csv_bytes, prediction)
    recomputed = _binary_metrics(reopened.true_label, reopened.probability)
    if recomputed != raw_metrics:
        raise Amendment06IntegrityError(f"External saved metric recomputation differs: {variant}")
    per_card = reporting.per_card_metrics(prediction)
    per_card_bytes, per_card_roundtrip = exact_table_csv_bytes(per_card)
    macro = {
        key: float(pd.to_numeric(per_card[key], errors="coerce").mean())
        for key in reporting.PRIMARY_METRICS
    }
    metrics_payload = {
        "run_id": run_dir.name, "run_kind": RUN_KIND, "variant_id": variant,
        "status": "PASS", "primary_endpoint": "raw_candidate_level_sklearn_average_precision",
        "decision_threshold": DECISION_THRESHOLD,
        "raw_candidate_level": raw_metrics,
        "review_weighted_sensitivity": weighted_metrics,
        "per_card_macro_secondary": macro,
        "external_audit_rows": len(audit.frame), "external_audit_cards": 10,
    }
    _stage_bytes(staging, f"predictions/{variant}_external_audit.csv.gz", csv_bytes)
    _stage_json(staging, f"metrics/{variant}.json", metrics_payload)
    _stage_bytes(staging, f"metrics/{variant}_per_card.csv", per_card_bytes)
    _stage_bytes(
        staging, f"models/{variant}/STATUS.txt",
        _variant_complete_status(run_dir.name, variant),
    )
    artifacts = [
        _publication_record(staging / relative, relative)
        for relative in _scoring_relatives(variant)
    ]
    scoring_manifest = {
        "run_id": run_dir.name, "run_kind": RUN_KIND,
        "variant_id": variant,
        "phase": "EXTERNAL_AUDIT_SCORING_COMPLETE", "status": "PASS",
        "external_audit_rows": 10_097, "external_audit_cards": 10,
        "external_audit_sha256": EXTERNAL_AUDIT_SHA256,
        "decision_threshold": DECISION_THRESHOLD,
        "cpu_fallback_detected": False,
        "raw_booster_parity_max_abs_diff": 0.0,
        "compatibility_loader_parity_max_abs_diff": 0.0,
        "independent_raw_booster_load": True,
        "independent_load_evidence": independent_load,
        "model_sha256_before_independent_load": independent_load["model_sha256_before"],
        "model_sha256_after_independent_load": sha256_file(model_path),
        "active_reloaded_cuda_probe": active_probe,
        "prediction_sha256": sha256_bytes(csv_bytes),
        "prediction_rows": len(prediction),
        "probability_sha256_float32": sha256_array(wrapper_host.astype("<f4", copy=False)),
        "raw_candidate_metrics": raw_metrics,
        "review_weighted_metrics": weighted_metrics,
        "csv_roundtrip": roundtrip,
        "per_card_csv_roundtrip": per_card_roundtrip,
        "training_completion_sha256": sha256_file(run_dir / _training_completion_relative(variant)),
        "artifact_records": artifacts,
    }
    _stage_json(staging, completion_relative, scoring_manifest)
    try:
        for relative in _scoring_relatives(variant):
            _publish_attempt_file(
                run_dir=run_dir, staging=staging, relative=relative,
                phase="AUDIT_SCORING", variant=variant, attempt_id=attempt_id,
            )
        _publish_attempt_file(
            run_dir=run_dir, staging=staging, relative=completion_relative,
            phase="AUDIT_SCORING", variant=variant, attempt_id=attempt_id,
        )
    except Exception as exc:
        _journal_event(
            run_dir, "AUDIT_SCORING_PUBLICATION_INTERRUPTED", phase="AUDIT_SCORING",
            variant_id=variant, attempt_id=attempt_id,
            error_type=type(exc).__name__, error=str(exc),
        )
        raise
    _ensure_unique_bound_journal_event(
        run_dir=run_dir, event_name="AUDIT_SCORING_COMPLETION_COMMITTED",
        binding_path=run_dir / completion_relative,
        binding_field="completion_sha256", phase="AUDIT_SCORING",
        variant=variant, recovered=False, extra={"attempt_id": attempt_id},
    )
    _progress("EXTERNAL_AUDIT_VARIANT_SCORING", "COMPLETE", variant=variant, status="PASS")
    return validate_external_scoring_completion(run_dir=run_dir, audit=audit, variant=variant)


def score_one_external_variant(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    development: DevelopmentPopulation, audit: ExternalAuditPopulation,
    variant: str,
) -> dict[str, Any]:
    before = len(_read_journal(run_dir))
    try:
        return _score_one_external_variant_impl(
            run_dir=run_dir, locked=locked, cache=cache,
            development=development, audit=audit, variant=variant,
        )
    except Exception as exc:
        events = _read_journal(run_dir)[before:]
        starts = [event for event in events if event.get("event") == "AUDIT_SCORING_ATTEMPT_STARTED"]
        if starts:
            attempt_id = str(starts[-1]["attempt_id"])
            if not any(
                event.get("event") == "AUDIT_SCORING_ATTEMPT_FAILED"
                and event.get("attempt_id") == attempt_id
                for event in events
            ):
                _journal_event(
                    run_dir, "AUDIT_SCORING_ATTEMPT_FAILED", phase="AUDIT_SCORING",
                    failure_phase="POST_START_THROUGH_COMPLETION_COMMIT",
                    variant_id=variant, attempt_id=attempt_id,
                    error_type=type(exc).__name__, error=str(exc),
                )
        raise


def validate_external_scoring_completion(
    *, run_dir: Path, audit: ExternalAuditPopulation, variant: str,
) -> dict[str, Any]:
    if variant not in TRAINED_VARIANTS:
        raise Amendment06IntegrityError(f"Unauthorized scored Variant: {variant}")
    path = run_dir / _scoring_completion_relative(variant)
    if not _safe_regular(path):
        raise Amendment06IntegrityError(f"External scoring completion is absent: {variant}")
    payload = strict_full_load_file(path)
    training_path = run_dir / _training_completion_relative(variant)
    model_path = run_dir / f"models/{variant}/model.ubj"
    training = strict_full_load_file(training_path)
    model_sha256 = sha256_file(model_path)
    independent = payload.get("independent_load_evidence")
    active_probe = payload.get("active_reloaded_cuda_probe")
    if not (
        payload.get("run_id") == run_dir.name and payload.get("run_kind") == RUN_KIND
        and payload.get("variant_id") == variant
        and payload.get("phase") == "EXTERNAL_AUDIT_SCORING_COMPLETE"
        and payload.get("status") == "PASS"
        and payload.get("external_audit_rows") == 10_097
        and payload.get("external_audit_cards") == 10
        and payload.get("external_audit_sha256") == EXTERNAL_AUDIT_SHA256
        and payload.get("decision_threshold") == DECISION_THRESHOLD
        and payload.get("cpu_fallback_detected") is False
        and payload.get("raw_booster_parity_max_abs_diff") == 0.0
        and payload.get("compatibility_loader_parity_max_abs_diff") == 0.0
        and payload.get("independent_raw_booster_load") is True
        and isinstance(independent, Mapping)
        and independent.get("status") == "PASS"
        and independent.get("compatibility_loader") == "XGBClassifier.load_model"
        and independent.get("raw_booster_loader") == "INDEPENDENT_XGB_BOOSTER_LOAD_MODEL"
        and independent.get("independent_objects") is True
        and independent.get("model_sha256_before") == model_sha256
        and independent.get("model_sha256_after") == model_sha256
        and payload.get("model_sha256_before_independent_load") == model_sha256
        and payload.get("model_sha256_after_independent_load") == model_sha256
        and training.get("model_sha256") == model_sha256
        and payload.get("training_completion_sha256") == sha256_file(training_path)
        and isinstance(active_probe, Mapping)
        and active_probe.get("status") == "PASS"
        and "cuda" in str(active_probe.get("device", "")).lower()
        and active_probe.get("warnings") == []
    ):
        raise Amendment06IntegrityError(f"External scoring completion binding differs: {variant}")
    _validate_exact_artifact_records(
        run_dir=run_dir, value=payload.get("artifact_records"),
        expected_relatives=_scoring_relatives(variant),
        label=f"Completed external scoring {variant}",
    )
    prediction_path = run_dir / f"predictions/{variant}_external_audit.csv.gz"
    prediction = pd.read_csv(prediction_path, engine="c", float_precision="round_trip")
    prediction = prediction.loc[:, list(PREDICTION_COLUMNS)]
    _, roundtrip = validate_prediction_roundtrip(prediction_path, prediction)
    raw = _binary_metrics(prediction.true_label, prediction.probability)
    weighted = _binary_metrics(prediction.true_label, prediction.probability, weights=prediction.review_weight)
    per_card = reporting.per_card_metrics(prediction)
    per_card_path = run_dir / f"metrics/{variant}_per_card.csv"
    per_card_roundtrip = validate_exact_table_csv(per_card_path, per_card)
    macro = {
        key: float(pd.to_numeric(per_card[key], errors="coerce").mean())
        for key in reporting.PRIMARY_METRICS
    }
    metrics_expected = {
        "run_id": run_dir.name, "run_kind": RUN_KIND, "variant_id": variant,
        "status": "PASS",
        "primary_endpoint": "raw_candidate_level_sklearn_average_precision",
        "decision_threshold": DECISION_THRESHOLD,
        "raw_candidate_level": raw,
        "review_weighted_sensitivity": weighted,
        "per_card_macro_secondary": macro,
        "external_audit_rows": len(audit.frame), "external_audit_cards": 10,
    }
    metrics_path = run_dir / f"metrics/{variant}.json"
    metrics_payload = strict_full_load_file(metrics_path)
    if not (
        len(prediction) == len(audit.frame)
        and np.array_equal(
            prediction.stable_candidate_id.astype(str).to_numpy(),
            audit.frame.stable_candidate_id.astype(str).to_numpy(),
        )
        and np.array_equal(
            prediction.card_id.astype(str).to_numpy(),
            audit.frame.card_id.astype(str).to_numpy(),
        )
        and np.array_equal(prediction.true_label.to_numpy(np.int8), audit.labels)
        and np.array_equal(
            prediction.review_weight.to_numpy(np.float64), audit.weights.astype(np.float64),
        )
        and np.array_equal(
            prediction.prediction_at_0_5.to_numpy(np.int8),
            (prediction.probability.to_numpy(np.float64) >= DECISION_THRESHOLD).astype(np.int8),
        )
        and np.array_equal(
            prediction.row_order.to_numpy(np.int64), np.arange(len(prediction)),
        )
        and set(prediction.variant_id.astype(str)) == {variant}
        and set(prediction.split_population.astype(str)) == {
            "external_audit:locked_10_card"
        }
        and payload.get("prediction_sha256") == sha256_file(prediction_path)
        and payload.get("prediction_rows") == len(prediction)
        and payload.get("probability_sha256_float32")
        == sha256_array(prediction.probability.to_numpy(np.float32).astype("<f4", copy=False))
        and raw == payload.get("raw_candidate_metrics")
        and weighted == payload.get("review_weighted_metrics")
        and payload.get("csv_roundtrip") == roundtrip
        and payload.get("per_card_csv_roundtrip") == per_card_roundtrip
        and roundtrip.get("status") == "PASS"
        and per_card_roundtrip.get("status") == "PASS"
        and metrics_payload == metrics_expected
        and (run_dir / f"models/{variant}/STATUS.txt").read_bytes()
        == _variant_complete_status(run_dir.name, variant)
    ):
        raise Amendment06IntegrityError(f"External scoring metric/row recomputation differs: {variant}")
    feature_path = run_dir / f"models/{variant}/feature_list.txt"
    features = tuple(feature_path.read_text(encoding="utf-8").splitlines())
    expected_count, expected_feature_sha = VARIANT_FEATURE_CONTRACT[variant]
    full_features = tuple(core.get_r92_feature_order())
    if not (
        len(full_features) == 93 and len(set(full_features)) == 93
        and len(features) == expected_count
        and _feature_list_sha256(features) == expected_feature_sha
    ):
        raise Amendment06IntegrityError(f"External scoring feature contract differs: {variant}")
    indices = tuple(full_features.index(feature) for feature in features)
    matrix_host = (
        audit.full_X if indices == tuple(range(93))
        else np.ascontiguousarray(audit.full_X[:, indices], dtype=np.float32)
    )
    loaded, raw_booster, live_load = load_independent_external_predictors(
        model_path, expected_feature_count=expected_count,
        expected_feature_list_sha256=expected_feature_sha,
        expected_model_sha256=model_sha256,
    )
    import cupy as cp
    device_matrix = cp.asarray(np.ascontiguousarray(matrix_host, dtype=np.float32))
    interval = tuple(map(int, training["iteration_range"]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapper_live = loaded.classifier.predict_proba(
            device_matrix, iteration_range=interval,
        )[:, 1]
        raw_live = raw_booster.inplace_predict(device_matrix, iteration_range=interval)
        cp.cuda.Stream.null.synchronize()
    wrapper_host = np.ascontiguousarray(cp.asnumpy(wrapper_live), dtype=np.float32)
    raw_host = np.ascontiguousarray(cp.asnumpy(raw_live), dtype=np.float32)
    live_warnings = [str(item.message) for item in caught]
    live_probe = core._active_reloaded_booster_cuda_probe(raw_booster, expected_count)
    saved_probability = prediction.probability.to_numpy(np.float32)
    if not (
        live_load == independent
        and not _fallback_warnings(live_warnings)
        and np.array_equal(wrapper_host, raw_host)
        and np.array_equal(wrapper_host, saved_probability)
        and live_probe.get("status") == "PASS"
        and "cuda" in str(live_probe.get("device", "")).lower()
        and live_probe.get("warnings") == []
        and sha256_file(model_path) == model_sha256
    ):
        raise Amendment06IntegrityError(
            f"External scoring live model/prediction parity differs: {variant}"
        )
    return {
        "status": "PASS", "variant_id": variant,
        "scoring_completion_sha256": sha256_file(path),
        "prediction": prediction,
        "raw_metrics": raw, "weighted_metrics": weighted,
        "per_card": per_card,
    }


def complete_external_audit_phase(
    *, run_dir: Path, audit: ExternalAuditPopulation,
) -> list[dict[str, Any]]:
    results = [
        validate_external_scoring_completion(run_dir=run_dir, audit=audit, variant=variant)
        for variant in TRAINED_VARIANTS
    ]
    if _SCIENTIFIC_COUNTERS["post_audit_training_calls"] != 0:
        raise Amendment06IntegrityError("A scientific training call occurred after Audit opening")
    current = _current_run_state(run_dir)
    if current == "FULL_MODELS_FROZEN":
        _set_run_state(run_dir, "FULL_EXTERNAL_AUDIT_SCORED")
    elif RUN_PHASES.index(current) < RUN_PHASES.index("FULL_EXTERNAL_AUDIT_SCORED"):
        raise Amendment06IntegrityError("Audit completion/state disagree")
    return results


def initialize_run_evidence(run_dir: Path) -> None:
    _snapshot_source_pre(run_dir)
    probe_source = ACCEPTED_PREFLIGHT / "provenance/cuda_active_fit_probe.json"
    probe_destination = run_dir / "provenance/cuda_active_fit_probe.json"
    probe_bytes = probe_source.read_bytes()
    if probe_destination.exists():
        if not _safe_regular(probe_destination, sha256_bytes(probe_bytes)):
            raise Amendment06IntegrityError("Full CUDA active-fit probe snapshot drifted")
    else:
        publish_bytes_no_clobber(probe_destination, probe_bytes)
    environment_path = run_dir / "provenance/cuda_environment.txt"
    if not environment_path.exists():
        command = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ], check=True, capture_output=True, text=True,
        )
        text_value = (
            f"RUN_ID={run_dir.name}\nPYTHON={Path(sys.executable).resolve()}\n"
            f"XGBOOST_VERSION={xgb.__version__}\nSKLEARN_VERSION={sklearn.__version__}\n"
            f"XGBOOST_USE_CUDA={xgb.build_info().get('USE_CUDA')}\n"
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<UNSET>')}\n"
            f"ABLATION_RUNTIME_ROOT={os.environ.get('ABLATION_RUNTIME_ROOT', '')}\n"
            f"NVIDIA_SMI={command.stdout.strip()}\n"
        )
        publish_bytes_no_clobber(environment_path, text_value.encode("utf-8"))


def _reporting_completion_path(run_dir: Path) -> Path:
    return run_dir / "provenance/reporting_completion_manifest.json"


REPORTING_REQUIRED_RELATIVES = frozenset({
    "tables/primary_metrics_official.csv",
    "tables/primary_metrics_exploratory.csv",
    "tables/review_weighted_sensitivity_official.csv",
    "tables/review_weighted_sensitivity_exploratory.csv",
    "tables/variant_minus_full_official.csv",
    "tables/variant_minus_full_exploratory.csv",
    "tables/paired_bootstrap_ci_official.csv",
    "tables/paired_bootstrap_ci_exploratory.csv",
    "tables/per_card_metrics_official.csv",
    "tables/per_card_metrics_exploratory.csv",
    "tables/official_blocked_variants.csv",
    "tables/class_balance_and_augmentation_stages.csv",
    "tables/training_resource_timing.csv",
    "tables/feature_and_resampling_parity.csv",
    "tables/r92_historical_control.csv",
    "tables/primary_metrics_official.tex",
    "tables/paired_bootstrap_ci_official.tex",
    "figures/official_average_precision.png",
    "figures/official_delta_average_precision.png",
    "figures/exploratory_average_precision.png",
    "docs/METHODS_DRAFT.md",
    "docs/RESULTS_DRAFT.md",
    "docs/REVIEWER_RESPONSE_DRAFT.md",
    "docs/REPRODUCE.md",
    "docs/WARNINGS_AND_LIMITATIONS.md",
    "statistics/paired_card_bootstrap.json",
    "metrics/r92_historical_control.json",
})


def _validate_reporting_completion(run_dir: Path) -> dict[str, Any]:
    path = _reporting_completion_path(run_dir)
    payload = strict_full_load_file(path)
    if not (
        payload.get("run_id") == run_dir.name
        and payload.get("run_kind") == RUN_KIND
        and payload.get("phase") == "SCIENTIFIC_REPORTING_COMPLETE"
        and payload.get("status") == "PASS"
        and payload.get("bootstrap_replicates") == BOOTSTRAP_REPLICATES
        and payload.get("bootstrap_seed") == BOOTSTRAP_SEED
        and payload.get("model_retraining") is False
        and payload.get("report_csv_roundtrip", {}).get("status") == "PASS"
        and payload.get("report_artifact_count") == 25
    ):
        raise Amendment06IntegrityError("Scientific reporting completion differs")
    records = payload.get("artifact_records", [])
    if not isinstance(records, list) or {
        str(record.get("relative_path")) for record in records
        if isinstance(record, Mapping)
    } != REPORTING_REQUIRED_RELATIVES or len(records) != len(REPORTING_REQUIRED_RELATIVES):
        raise Amendment06IntegrityError("Scientific reporting artifact set/count differs")
    for record in records:
        relative = str(record.get("relative_path"))
        member = run_dir / _safe_run_relative(relative)
        if not (
            _safe_regular(member, str(record.get("sha256")))
            and member.stat().st_size == int(record.get("size_bytes", -1))
        ):
            raise Amendment06IntegrityError(f"Completed reporting artifact drifted: {relative}")
    bootstrap = strict_full_load_file(run_dir / "statistics/paired_card_bootstrap.json")
    if not (
        bootstrap.get("status") == "PASS"
        and bootstrap.get("replicates") == BOOTSTRAP_REPLICATES
        and bootstrap.get("seed") == BOOTSTRAP_SEED
        and bootstrap.get("model_retraining") is False
    ):
        raise Amendment06IntegrityError("Paired bootstrap artifact differs")
    _recompute_and_validate_reporting_outputs(run_dir, payload)
    return payload


def _recompute_and_validate_reporting_outputs(
    run_dir: Path, completion: Mapping[str, Any],
) -> dict[str, Any]:
    predictions: dict[str, pd.DataFrame] = {}
    metrics_by_variant: dict[str, dict[str, Any]] = {}
    per_card_by_variant: dict[str, pd.DataFrame] = {}
    timing_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    for variant in TRAINED_VARIANTS:
        training = strict_full_load_file(run_dir / _training_completion_relative(variant))
        scoring = strict_full_load_file(run_dir / _scoring_completion_relative(variant))
        prediction_path = run_dir / f"predictions/{variant}_external_audit.csv.gz"
        prediction = pd.read_csv(
            prediction_path, engine="c", float_precision="round_trip",
        ).loc[:, list(PREDICTION_COLUMNS)]
        reporting.validate_prediction_csv_roundtrip(
            prediction_path, prediction,
            metrics=scoring["raw_candidate_metrics"], compressed=True,
        )
        raw_metrics = _binary_metrics(
            prediction.true_label, prediction.probability,
        )
        weighted_metrics = _binary_metrics(
            prediction.true_label, prediction.probability,
            weights=prediction.review_weight,
        )
        if (
            raw_metrics != scoring["raw_candidate_metrics"]
            or weighted_metrics != scoring["review_weighted_metrics"]
        ):
            raise Amendment06IntegrityError(
                f"Saved reporting metrics differ from prediction recomputation: {variant}"
            )
        predictions[variant] = prediction
        per_card = reporting.per_card_metrics(prediction)
        validate_exact_table_csv(run_dir / f"metrics/{variant}_per_card.csv", per_card)
        per_card_by_variant[variant] = per_card
        metrics_by_variant[variant] = {
            "feature_count": training["feature_count"],
            "best_iteration": training["best_iteration"],
            "external_raw": raw_metrics,
            "external_weighted_sensitivity": weighted_metrics,
        }
        timing_rows.append(strict_full_load_file(run_dir / f"timing/{variant}.json"))
        parity_rows.append({
            "variant": variant,
            "classification": training["variant_classification"],
            "feature_count": training["feature_count"],
            "feature_list_sha256": training["feature_list_sha256"],
            "population": (
                "post_jitter_pre_smote" if variant == "no_safe_smote"
                else "post_safe_smote"
            ),
            "frozen_population_sha256": training["frozen_population_sha256"],
            "four_way_parity_max_abs_diff": training["four_way_parity_max_abs_diff"],
            "external_raw_booster_parity_max_abs_diff": scoring[
                "raw_booster_parity_max_abs_diff"
            ],
            "projection_only": True,
        })
    cards = sorted(predictions["full_new_reference"].card_id.astype(str).unique())
    bootstrap = reporting.paired_card_bootstrap(
        predictions, cards, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
    )
    if bootstrap != strict_full_load_file(
        run_dir / "statistics/paired_card_bootstrap.json"
    ):
        raise Amendment06IntegrityError("Saved paired bootstrap differs from exact recomputation")
    cache_manifest = strict_full_load_file(
        run_dir / "provenance/frozen_full_resampling_manifest.json"
    )
    stage_rows = [
        {"stage": stage, **cache_manifest["stage_counts"][stage]}
        for stage in EXPECTED_STAGE_COUNTS
    ]
    r92_control = strict_full_load_file(run_dir / "metrics/r92_historical_control.json")
    staging = (
        EXPECTED_RUNTIME_ROOT / "report_validation" / run_dir.name / uuid.uuid4().hex
    )
    staging.mkdir(parents=True)
    try:
        evidence = reporting.build_all_reports(
            staging,
            metrics_by_variant=metrics_by_variant,
            per_card_by_variant=per_card_by_variant,
            bootstrap=bootstrap,
            stage_table=pd.DataFrame(stage_rows),
            timing_table=pd.DataFrame(timing_rows),
            parity_table=pd.DataFrame(parity_rows),
            r92_control=r92_control,
        )
        generated = {
            str(item["relative_path"]): item for item in evidence["artifacts"]
        }
        report_relatives = REPORTING_REQUIRED_RELATIVES - {
            "statistics/paired_card_bootstrap.json", "metrics/r92_historical_control.json",
        }
        if set(generated) != report_relatives or evidence.get("artifact_count") != 25:
            raise Amendment06IntegrityError("Regenerated report member set/count differs")
        for relative, record in generated.items():
            live = run_dir / relative
            regenerated = staging / relative
            if not (
                _safe_regular(live, str(record["sha256"]))
                and live.stat().st_size == int(record["size_bytes"])
                and live.read_bytes() == regenerated.read_bytes()
            ):
                raise Amendment06IntegrityError(
                    f"Regenerated report bytes differ: {relative}"
                )
        if completion.get("report_csv_roundtrip") != evidence.get("csv_roundtrip"):
            raise Amendment06IntegrityError("Reporting CSV-roundtrip evidence differs")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "status": "PASS", "bootstrap_recomputed": True,
        "report_artifacts_regenerated": 25,
    }


def _build_reports_and_bootstrap_impl(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    audit: ExternalAuditPopulation, scoring_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if _reporting_completion_path(run_dir).exists():
        result = _validate_reporting_completion(run_dir)
        _ensure_unique_bound_journal_event(
            run_dir=run_dir, event_name="REPORTING_COMPLETION_COMMITTED",
            binding_path=_reporting_completion_path(run_dir),
            binding_field="completion_sha256", phase="REPORTING",
            variant="GLOBAL", recovered=True, extra={
                "attempt_id": _phase_publication_records(
                    run_dir, "REPORTING", "GLOBAL",
                ).get("provenance/reporting_completion_manifest.json", {}).get(
                    "attempt_id"
                ),
            },
        )
        _journal_event(run_dir, "COMPLETED_REPORTING_SKIPPED", phase="REPORTING")
        return result
    _record_interrupted_attempts(
        run_dir=run_dir, phase="REPORTING", variant="GLOBAL",
    )
    owned = _phase_publication_records(run_dir, "REPORTING", "GLOBAL")
    _recover_unpublished_paths(
        run_dir=run_dir, phase="REPORTING", variant="GLOBAL",
        completion_relative="provenance/reporting_completion_manifest.json",
        possible_relatives=tuple(owned),
    )
    attempt_id = uuid.uuid4().hex
    staging = _attempt_root(run_dir, "REPORTING", "GLOBAL", attempt_id)
    staging.mkdir(parents=True)
    _journal_event(
        run_dir, "REPORTING_ATTEMPT_STARTED", phase="REPORTING",
        variant_id="GLOBAL", attempt_id=attempt_id,
    )
    _progress("PAIRED_CARD_BOOTSTRAP", "START", replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    result_by_variant = {str(item["variant_id"]): item for item in scoring_results}
    if set(result_by_variant) != set(TRAINED_VARIANTS):
        raise Amendment06IntegrityError("Reporting scoring-result Variant set differs")
    predictions = {
        variant: result_by_variant[variant]["prediction"]
        for variant in TRAINED_VARIANTS
    }
    cards = sorted(audit.frame.card_id.astype(str).unique())
    bootstrap = reporting.paired_card_bootstrap(
        predictions, cards, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
    )
    if bootstrap.get("retraining") is not False or bootstrap.get("model_retraining") is not False:
        raise Amendment06IntegrityError("Bootstrap attempted model retraining")
    bootstrap_sha256 = sha256_bytes(strict_full_canonical_json_bytes(bootstrap))
    bootstrap_events = [
        event for event in _read_journal(run_dir)
        if event.get("event") == "BOOTSTRAP_RESAMPLING"
    ]
    expected_bootstrap_event = {
        "phase": "REPORTING", "status": "PASS",
        "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
        "model_retraining": False, "bootstrap_sha256": bootstrap_sha256,
    }
    if len(bootstrap_events) > 1 or (
        bootstrap_events and any(
            bootstrap_events[0].get(key) != value
            for key, value in expected_bootstrap_event.items()
        )
    ):
        raise Amendment06IntegrityError("Paired-bootstrap journal binding differs")
    if not bootstrap_events:
        _journal_event(
            run_dir, "BOOTSTRAP_RESAMPLING", **expected_bootstrap_event,
        )
    _progress("PAIRED_CARD_BOOTSTRAP", "COMPLETE", status="PASS")
    _progress("SCIENTIFIC_REPORTING", "START")
    r92_control, _ = core.score_r92_control(audit.frame)
    if r92_control.get("status") != "PASS":
        raise Amendment06IntegrityError("Historical r92 control replay failed after model freeze")
    # Use the strict-JSON reopened key order at first publication so byte-for-byte
    # report regeneration after resume receives the same mapping representation.
    r92_control = strict_full_loads(strict_full_json_bytes(r92_control))
    metrics_by_variant: dict[str, dict[str, Any]] = {}
    per_card_by_variant: dict[str, pd.DataFrame] = {}
    timing_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    for variant in TRAINED_VARIANTS:
        scoring = result_by_variant[variant]
        training = strict_full_load_file(run_dir / _training_completion_relative(variant))
        metrics_by_variant[variant] = {
            "feature_count": training["feature_count"],
            "best_iteration": training["best_iteration"],
            "external_raw": scoring["raw_metrics"],
            "external_weighted_sensitivity": scoring["weighted_metrics"],
        }
        per_card_by_variant[variant] = scoring["per_card"]
        timing_rows.append(strict_full_load_file(run_dir / f"timing/{variant}.json"))
        parity_rows.append({
            "variant": variant,
            "classification": training["variant_classification"],
            "feature_count": training["feature_count"],
            "feature_list_sha256": training["feature_list_sha256"],
            "population": (
                "post_jitter_pre_smote" if variant == "no_safe_smote" else "post_safe_smote"
            ),
            "frozen_population_sha256": cache.binding_sha256,
            "four_way_parity_max_abs_diff": training["four_way_parity_max_abs_diff"],
            "external_raw_booster_parity_max_abs_diff": 0.0,
            "projection_only": True,
        })
    canonical_cache_manifest = strict_full_loads(strict_full_json_bytes(cache.manifest))
    stage_rows = []
    for stage in EXPECTED_STAGE_COUNTS:
        stage_rows.append({
            "stage": stage, **canonical_cache_manifest["stage_counts"][stage],
        })
    report_evidence = reporting.build_all_reports(
        staging,
        metrics_by_variant=metrics_by_variant,
        per_card_by_variant=per_card_by_variant,
        bootstrap=bootstrap,
        stage_table=pd.DataFrame(stage_rows),
        timing_table=pd.DataFrame(timing_rows),
        parity_table=pd.DataFrame(parity_rows),
        r92_control=r92_control,
    )
    if report_evidence.get("csv_roundtrip", {}).get("status") != "PASS":
        raise Amendment06IntegrityError("Generated report CSV exact roundtrip failed")
    _stage_json(staging, "statistics/paired_card_bootstrap.json", bootstrap)
    _stage_json(staging, "metrics/r92_historical_control.json", r92_control)
    relatives = [
        *(str(item["relative_path"]) for item in report_evidence["artifacts"]),
        "statistics/paired_card_bootstrap.json",
        "metrics/r92_historical_control.json",
    ]
    artifact_records = [_publication_record(staging / relative, relative) for relative in relatives]
    completion = {
        "run_id": run_dir.name, "run_kind": RUN_KIND,
        "phase": "SCIENTIFIC_REPORTING_COMPLETE", "status": "PASS",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED, "model_retraining": False,
        "report_artifact_count": report_evidence["artifact_count"],
        "report_csv_roundtrip": report_evidence["csv_roundtrip"],
        "official_runnable_count": 9, "official_blocked_true_no_pca_count": 1,
        "exploratory_count": 1, "delta_definition": "variant_minus_full",
        "artifact_records": artifact_records,
    }
    _stage_json(staging, "provenance/reporting_completion_manifest.json", completion)
    try:
        for relative in relatives:
            _publish_attempt_file(
                run_dir=run_dir, staging=staging, relative=relative,
                phase="REPORTING", variant="GLOBAL", attempt_id=attempt_id,
            )
        _publish_attempt_file(
            run_dir=run_dir, staging=staging,
            relative="provenance/reporting_completion_manifest.json",
            phase="REPORTING", variant="GLOBAL", attempt_id=attempt_id,
        )
    except Exception as exc:
        _journal_event(
            run_dir, "REPORTING_PUBLICATION_INTERRUPTED", phase="REPORTING",
            variant_id="GLOBAL", attempt_id=attempt_id,
            error_type=type(exc).__name__, error=str(exc),
        )
        raise
    _ensure_unique_bound_journal_event(
        run_dir=run_dir, event_name="REPORTING_COMPLETION_COMMITTED",
        binding_path=_reporting_completion_path(run_dir),
        binding_field="completion_sha256", phase="REPORTING",
        variant="GLOBAL", recovered=False, extra={"attempt_id": attempt_id},
    )
    _progress("SCIENTIFIC_REPORTING", "COMPLETE", status="PASS")
    return _validate_reporting_completion(run_dir)


def build_reports_and_bootstrap(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    audit: ExternalAuditPopulation, scoring_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    before = len(_read_journal(run_dir))
    try:
        return _build_reports_and_bootstrap_impl(
            run_dir=run_dir, locked=locked, cache=cache,
            audit=audit, scoring_results=scoring_results,
        )
    except Exception as exc:
        events = _read_journal(run_dir)[before:]
        starts = [event for event in events if event.get("event") == "REPORTING_ATTEMPT_STARTED"]
        if starts:
            attempt_id = str(starts[-1]["attempt_id"])
            if not any(
                event.get("event") == "REPORTING_ATTEMPT_FAILED"
                and event.get("attempt_id") == attempt_id
                for event in events
            ):
                _journal_event(
                    run_dir, "REPORTING_ATTEMPT_FAILED", phase="REPORTING",
                    failure_phase="POST_START_THROUGH_COMPLETION_COMMIT",
                    variant_id="GLOBAL", attempt_id=attempt_id,
                    error_type=type(exc).__name__, error=str(exc),
                )
        raise


def build_metric_recomputation(
    *, run_dir: Path, audit: ExternalAuditPopulation,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for variant in TRAINED_VARIANTS:
        training = strict_full_load_file(run_dir / _training_completion_relative(variant))
        validation_path = run_dir / f"predictions/{variant}_validation.csv.gz"
        validation = pd.read_csv(validation_path, engine="c", float_precision="round_trip")
        validation = validation.loc[:, list(PREDICTION_COLUMNS)]
        validation_evidence = reporting.validate_prediction_csv_roundtrip(
            validation_path, validation,
            metrics=training["validation_metrics"], compressed=True,
        )
        scoring = strict_full_load_file(run_dir / _scoring_completion_relative(variant))
        external_path = run_dir / f"predictions/{variant}_external_audit.csv.gz"
        external = pd.read_csv(external_path, engine="c", float_precision="round_trip")
        external = external.loc[:, list(PREDICTION_COLUMNS)]
        external_evidence = reporting.validate_prediction_csv_roundtrip(
            external_path, external,
            metrics=scoring["raw_candidate_metrics"], compressed=True,
        )
        raw = _binary_metrics(external.true_label, external.probability)
        weighted = _binary_metrics(
            external.true_label, external.probability, weights=external.review_weight,
        )
        if not (
            raw == scoring["raw_candidate_metrics"]
            and weighted == scoring["review_weighted_metrics"]
            and len(external) == len(audit.frame)
        ):
            raise Amendment06IntegrityError(f"Final external metric recomputation differs: {variant}")
        records.append({
            "variant_id": variant,
            "validation_prediction_sha256": sha256_file(validation_path),
            "validation_metric_recomputation": validation_evidence,
            "external_prediction_sha256": sha256_file(external_path),
            "external_metric_recomputation": external_evidence,
            "raw_metrics_exact": True, "weighted_metrics_exact": True,
        })
    return {
        "run_id": run_dir.name, "run_kind": RUN_KIND, "status": "PASS",
        "required_metrics_finite": True, "numeric_equality": "EXACT_NO_TOLERANCE",
        "csv_engine": "c", "float_precision": "round_trip",
        "variant_count": 10, "records": records,
    }


def build_execution_ledger(run_dir: Path) -> dict[str, Any]:
    all_events = _read_journal(run_dir)
    cutoffs = [
        index for index, event in enumerate(all_events)
        if event.get("event") == "REPORTING_COMPLETION_COMMITTED"
    ]
    if len(cutoffs) != 1:
        raise Amendment06IntegrityError("Execution ledger requires one reporting-completion cutoff")
    events = all_events[:cutoffs[0] + 1]
    event_counts = Counter(str(event.get("event")) for event in events)
    indexed = list(enumerate(events))
    training_commits = [
        (index, event) for index, event in indexed
        if event.get("event") == "TRAINING_COMPLETION_COMMITTED"
    ]
    scoring_commits = [
        (index, event) for index, event in indexed
        if event.get("event") == "AUDIT_SCORING_COMPLETION_COMMITTED"
    ]
    frozen_events = [
        (index, event) for index, event in indexed if event.get("event") == "MODELS_FROZEN"
    ]
    audit_events = [
        (index, event) for index, event in indexed if event.get("event") == "EXTERNAL_AUDIT_OPENED"
    ]
    if len(frozen_events) != 1 or len(audit_events) != 1:
        raise Amendment06IntegrityError("Execution ledger model-freeze/Audit-open cardinality differs")
    frozen_index = frozen_events[0][0]
    audit_index = audit_events[0][0]
    if not (frozen_index < audit_index < cutoffs[0]):
        raise Amendment06IntegrityError("Execution ledger phase ordering differs")
    completed_training = [str(event.get("variant_id")) for _, event in training_commits]
    completed_scoring = [str(event.get("variant_id")) for _, event in scoring_commits]
    if (
        completed_training != list(TRAINED_VARIANTS)
        or completed_scoring != list(TRAINED_VARIANTS)
        or any(index >= frozen_index for index, _ in training_commits)
        or any(index <= audit_index or index >= cutoffs[0] for index, _ in scoring_commits)
    ):
        raise Amendment06IntegrityError("Execution ledger cannot close before all completions")
    for index, event in training_commits:
        variant = str(event["variant_id"])
        completion_path = run_dir / _training_completion_relative(variant)
        starts = [
            start_index for start_index, candidate in indexed[:index]
            if candidate.get("event") == "TRAINING_ATTEMPT_STARTED"
            and candidate.get("variant_id") == variant
        ]
        fits = [
            fit_index for fit_index, candidate in indexed[:index]
            if candidate.get("event") == "SCIENTIFIC_FIT_CALLED"
            and candidate.get("variant_id") == variant
        ]
        if not (
            starts and fits and starts[-1] <= fits[-1] < index
            and event.get("completion_sha256") == sha256_file(completion_path)
        ):
            raise Amendment06IntegrityError(f"Execution ledger training chain differs: {variant}")
    for index, event in scoring_commits:
        variant = str(event["variant_id"])
        completion_path = run_dir / _scoring_completion_relative(variant)
        starts = [
            start_index for start_index, candidate in indexed[:index]
            if candidate.get("event") == "AUDIT_SCORING_ATTEMPT_STARTED"
            and candidate.get("variant_id") == variant
        ]
        if not starts or starts[-1] >= index or event.get("completion_sha256") != sha256_file(completion_path):
            raise Amendment06IntegrityError(f"Execution ledger scoring chain differs: {variant}")
    fit_events = [
        (index, event) for index, event in indexed
        if event.get("event") == "SCIENTIFIC_FIT_CALLED"
    ]
    start_events = [
        (index, event) for index, event in indexed
        if event.get("event") == "TRAINING_ATTEMPT_STARTED"
    ]
    failed_events = [
        (index, event) for index, event in indexed
        if event.get("event") == "TRAINING_ATTEMPT_FAILED"
    ]
    starts_by_attempt = Counter(str(event.get("attempt_id")) for _, event in start_events)
    fits_by_attempt = Counter(str(event.get("attempt_id")) for _, event in fit_events)
    failed_by_attempt = Counter(str(event.get("attempt_id")) for _, event in failed_events)
    committed_by_attempt = Counter(
        str(event.get("attempt_id")) for _, event in training_commits
    )
    attempt_ids = set(starts_by_attempt)
    if not attempt_ids or any(
        not re.fullmatch(r"[0-9a-f]{32}", attempt_id)
        or starts_by_attempt[attempt_id] != 1
        or fits_by_attempt[attempt_id] > 1
        or failed_by_attempt[attempt_id] + committed_by_attempt[attempt_id] != 1
        for attempt_id in attempt_ids
    ):
        raise Amendment06IntegrityError("Execution ledger training-attempt lifecycle differs")
    if set(fits_by_attempt) - attempt_ids or set(failed_by_attempt) - attempt_ids:
        raise Amendment06IntegrityError("Execution ledger has an orphan fit/failure event")
    if any(
        event.get("variant_id") not in TRAINED_VARIANTS
        or event.get("seed") != PRIMARY_MODEL_SEED
        or event.get("stability_fit") is not False
        or index >= frozen_index
        or str(event.get("attempt_id")) not in attempt_ids
        for index, event in fit_events
    ):
        raise Amendment06IntegrityError("Execution ledger fit authorization/order differs")
    successful_fit_variants = [
        str(event.get("variant_id")) for _, event in fit_events
        if committed_by_attempt[str(event.get("attempt_id"))] == 1
    ]
    if successful_fit_variants != list(TRAINED_VARIANTS):
        raise Amendment06IntegrityError("Execution ledger successful-fit Variant set/order differs")
    stability_fits = sum(
        event.get("stability_fit") is not False
        or event.get("seed") != PRIMARY_MODEL_SEED
        for _, event in fit_events
    )
    post_audit_training = sum(index > audit_index for index, _ in fit_events)
    threshold_calls = sum(
        event.get("event") == "THRESHOLD_TUNING_CALLED" for event in events
    )
    bootstrap_events = [
        (index, event) for index, event in indexed
        if event.get("event") == "BOOTSTRAP_RESAMPLING"
    ]
    last_scoring_index = max(index for index, _ in scoring_commits)
    if not (
        len(bootstrap_events) == 1
        and last_scoring_index < bootstrap_events[0][0] < cutoffs[0]
        and bootstrap_events[0][1].get("phase") == "REPORTING"
        and bootstrap_events[0][1].get("status") == "PASS"
        and bootstrap_events[0][1].get("replicates") == BOOTSTRAP_REPLICATES
        and bootstrap_events[0][1].get("seed") == BOOTSTRAP_SEED
        and bootstrap_events[0][1].get("model_retraining") is False
        and re.fullmatch(
            r"[0-9a-f]{64}", str(bootstrap_events[0][1].get("bootstrap_sha256", ""))
        )
    ):
        raise Amendment06IntegrityError("Execution ledger bootstrap binding/order differs")
    if stability_fits or post_audit_training or threshold_calls:
        raise Amendment06IntegrityError("Execution ledger records forbidden scientific calls")
    failed_training = event_counts.get("TRAINING_ATTEMPT_FAILED", 0)
    failed_scoring = event_counts.get("AUDIT_SCORING_ATTEMPT_FAILED", 0)
    failed_reporting = event_counts.get("REPORTING_ATTEMPT_FAILED", 0)
    return {
        "run_id": run_dir.name, "run_kind": RUN_KIND, "status": "PASS",
        "journal_schema": "ATOMIC_CONTIGUOUS_HASH_CHAINED_PER_EVENT_V1",
        "journal_phase_order_validation": "PASS",
        "completed_trained_models": len(completed_training),
        "completed_official_runnable_models": sum(
            variant in OFFICIAL_VARIANTS for variant in completed_training
        ),
        "completed_exploratory_models": sum(
            variant == EXPLORATORY_VARIANT for variant in completed_training
        ),
        "completed_true_no_pca_models": 0,
        "stability_fits": stability_fits,
        "threshold_tuning_calls": threshold_calls,
        "post_audit_training_calls": post_audit_training,
        "scientific_fit_attempts": len(fit_events),
        "failed_unpublished_training_attempts": failed_training,
        "failed_unpublished_scoring_attempts": failed_scoring,
        "failed_unpublished_reporting_attempts": failed_reporting,
        "prediction_only_reload_events": (
            event_counts.get("AUDIT_SCORING_ATTEMPT_STARTED", 0)
            + event_counts.get("COMPLETED_AUDIT_SCORING_SKIPPED", 0)
        ),
        "cache_generation_events": event_counts.get("CACHE_GENERATION", 0),
        "cache_load_events": event_counts.get("CACHE_LOAD", 0),
        "cache_rematerialization_events": event_counts.get("CACHE_REMATERIALIZATION", 0),
        "external_audit_open_events": len(audit_events),
        "bootstrap_resampling_events": len(bootstrap_events),
        "package_attempts_at_scientific_freeze": 0,
        "package_attempts_tracked_adjacent_to_frozen_run": True,
        "ownership_journal_path": str(_journal_path(run_dir)),
        "ownership_journal_prefix_sha256_at_scientific_freeze": sha256_bytes(b"".join(
            strict_full_canonical_json_bytes(event) + b"\n" for event in events
        )),
        "ownership_journal_event_count_at_scientific_freeze": len(events),
    }


MODEL_BUNDLE_GLOBALS = frozenset({
    "config/run_identity.lock.json",
    "config/training_configuration.lock.json",
    "config/variant_feature_sets.json",
    "config/feature_manifest.csv",
    "config/feature_dependency_graph.json",
    "config/locked_split_identity.json",
    "splits/row_split_manifest.csv.gz",
    "provenance/amendment06_external_review_authorization.json",
    "provenance/scientific_execution_code_manifest.tsv",
    "provenance/source_input_hashes_pre.tsv",
    "provenance/source_input_hashes_post.tsv",
    "provenance/frozen_full_imputer.json",
    "provenance/frozen_full_resampling_manifest.json",
    "provenance/full_models_frozen_manifest.json",
    "provenance/code_snapshot/amendment03_compat.py",
    "models/no_pca/STATUS.txt",
    "models/no_pca/blocked_evidence.json",
    *(
        f"provenance/state/{index:02d}_{phase}.json"
        for index, phase in enumerate(RUN_PHASES)
    ),
})


def _models_bundle_flag(relative: str) -> bool:
    if relative in MODEL_BUNDLE_GLOBALS:
        return True
    parts = Path(relative).parts
    return bool(
        len(parts) == 3 and parts[0] == "models" and parts[1] in TRAINED_VARIANTS
        and parts[2] in {
            "model.ubj", "feature_list.txt", "model_metadata.json",
            "transform_binding.json", "training_completion_manifest.json",
            "external_scoring_manifest.json", "STATUS.txt",
        }
    )


def _artifact_role(relative: str) -> str:
    if relative.endswith(".ubj"):
        return "SCIENTIFIC_MODEL_BINARY"
    if relative.startswith("predictions/"):
        return "SCIENTIFIC_PREDICTION"
    if relative.startswith("metrics/") or relative.startswith("statistics/"):
        return "SCIENTIFIC_METRIC_OR_STATISTIC"
    if relative.startswith("tables/") or relative.startswith("figures/") or relative.startswith("docs/"):
        return "SCIENTIFIC_REPORT"
    if relative.startswith("logs/") or relative.startswith("timing/"):
        return "EXECUTION_RESOURCE_EVIDENCE"
    if relative.startswith("models/"):
        return "MODEL_CONTRACT_OR_STATUS"
    if relative.startswith("config/") or relative.startswith("splits/"):
        return "LOCKED_CONFIGURATION_OR_SPLIT"
    return "PROVENANCE_AND_RUN_CONTROL"


def _walk_run_regular_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    for directory, names, filenames in os.walk(run_dir, followlinks=False):
        root = Path(directory)
        for name in names:
            child = root / name
            if child.is_symlink():
                raise Amendment06IntegrityError(f"Frozen Run contains symlink directory: {child}")
        for name in filenames:
            child = root / name
            mode = os.lstat(child).st_mode
            if not stat.S_ISREG(mode):
                raise Amendment06IntegrityError(f"Frozen Run contains unsafe file: {child}")
            files.append(child)
    return sorted(files, key=lambda path: path.relative_to(run_dir).as_posix().encode("utf-8"))


def expected_scientific_run_files() -> frozenset[str]:
    per_variant = {
        relative
        for variant in TRAINED_VARIANTS
        for relative in (
            *_training_relatives(variant), _training_completion_relative(variant),
            *_scoring_relatives(variant), _scoring_completion_relative(variant),
        )
    }
    snapshots = {
        *(
            f"provenance/code_snapshot/{Path(relative).name}"
            for relative in SCIENTIFIC_CODE_RELATIVES
        ),
        *(
            f"provenance/tests_snapshot/{Path(relative).name}"
            for relative in SCIENTIFIC_TEST_RELATIVES
        ),
    }
    return frozenset({
        "RUN_STATUS.txt", "BLOCKERS.md",
        "config/run_identity.lock.json",
        "config/training_configuration.lock.json",
        "config/variant_feature_sets.json",
        "config/feature_manifest.csv",
        "config/feature_dependency_graph.json",
        "config/locked_split_identity.json",
        "splits/row_split_manifest.csv.gz",
        "models/no_pca/STATUS.txt", "models/no_pca/blocked_evidence.json",
        "provenance/amendment06_prompt_snapshot.txt",
        "provenance/amendment06_external_review_authorization.json",
        "provenance/amendment06_code_lineage.json",
        "provenance/scientific_execution_code_manifest.tsv",
        "provenance/package_validator_code_manifest_initial.tsv",
        "provenance/source_input_hashes_pre.tsv",
        "provenance/source_input_hashes_post.tsv",
        "provenance/cuda_active_fit_probe.json",
        "provenance/cuda_environment.txt",
        "provenance/frozen_full_imputer.json",
        "provenance/frozen_full_resampling_manifest.json",
        "provenance/full_models_frozen_manifest.json",
        "provenance/external_audit_exclusion_and_opening.json",
        "provenance/reporting_completion_manifest.json",
        "provenance/metric_recomputation.json",
        "provenance/execution_ledger.json",
        "provenance/strict_json_validation.json",
        "docs/PROTOCOL_AMENDMENT_06.md",
        *(f"provenance/state/{index:02d}_{phase}.json" for index, phase in enumerate(RUN_PHASES)),
        *REPORTING_REQUIRED_RELATIVES,
        *per_variant, *snapshots,
    })


def output_manifest_bytes(
    run_dir: Path, *, require_exact_contract: bool = False,
) -> bytes:
    rows: list[dict[str, Any]] = []
    files = _walk_run_regular_files(run_dir)
    relatives = {
        path.relative_to(run_dir).as_posix() for path in files
        if path.relative_to(run_dir).as_posix() != "OUTPUT_MANIFEST_FINAL.tsv"
    }
    if require_exact_contract and relatives != expected_scientific_run_files():
        raise Amendment06IntegrityError(
            "Final scientific Run member set differs: "
            f"missing={sorted(expected_scientific_run_files() - relatives)}, "
            f"extra={sorted(relatives - expected_scientific_run_files())}"
        )
    for path in files:
        relative = path.relative_to(run_dir).as_posix()
        if relative == "OUTPUT_MANIFEST_FINAL.tsv":
            continue
        rows.append({
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "artifact_role": _artifact_role(relative),
            "include_in_review_bundle": not relative.endswith(".ubj"),
            "include_in_models_bundle": _models_bundle_flag(relative),
        })
    frame = pd.DataFrame(rows, columns=OUTPUT_MANIFEST_COLUMNS)
    return frame.to_csv(index=False, sep="\t", lineterminator="\n").encode("utf-8")


def validate_output_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "OUTPUT_MANIFEST_FINAL.tsv"
    if not _safe_regular(path):
        raise Amendment06IntegrityError("Final output manifest is absent/unsafe")
    expected = output_manifest_bytes(run_dir, require_exact_contract=True)
    if path.read_bytes() != expected:
        raise Amendment06IntegrityError("Final output manifest differs from immutable Run tree")
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if list(frame.columns) != list(OUTPUT_MANIFEST_COLUMNS):
        raise Amendment06IntegrityError("Final output manifest schema differs")
    if not (
        set(frame.include_in_review_bundle) <= {"True", "False"}
        and set(frame.include_in_models_bundle) <= {"True", "False"}
        and frame.relative_path.is_unique
        and "OUTPUT_MANIFEST_FINAL.tsv" not in set(frame.relative_path)
    ):
        raise Amendment06IntegrityError("Final output manifest flags/paths differ")
    return {
        "status": "PASS", "rows": len(frame), "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _final_status(run_dir: Path) -> str:
    return (
        f"RUN_ID={run_dir.name}\nRUN_KIND={RUN_KIND}\n"
        "RUN_STATE=FULL_SCIENTIFIC_COMPLETE\n"
        f"FULL_AUTHORIZATION={AUTHORIZATION}\n"
        "FULL_MODELS_FROZEN=YES\nEXTERNAL_AUDIT_OPENED=YES\n"
        "OFFICIAL_RUNNABLE_MODELS_COMPLETED=9_OF_9\n"
        "EXPLORATORY_MODELS_COMPLETED=1_OF_1\n"
        "OFFICIAL_TRUE_NO_PCA_STATUS=NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE\n"
        "PAIRED_CARD_BOOTSTRAP_REPLICATES=10000\n"
        "CPU_FALLBACK_DETECTED=NO\nSCIENTIFIC_RESULTS_CLAIMED=YES\n"
        "PACKAGE_PUBLICATION_STATE=ADJACENT_SEPARATELY_RETRYABLE\n"
    )


def finalize_scientific_run(
    *, run_dir: Path, audit: ExternalAuditPopulation,
) -> dict[str, Any]:
    manifest_path = run_dir / "OUTPUT_MANIFEST_FINAL.tsv"
    if manifest_path.exists():
        if _current_run_state(run_dir) != "FULL_SCIENTIFIC_COMPLETE":
            raise Amendment06IntegrityError("Final output manifest exists before final scientific state")
        return validate_output_manifest(run_dir)
    _validate_reporting_completion(run_dir)
    metric_path = run_dir / "provenance/metric_recomputation.json"
    metric_payload = build_metric_recomputation(run_dir=run_dir, audit=audit)
    if metric_path.exists():
        if strict_full_load_file(metric_path) != metric_payload:
            raise Amendment06IntegrityError("Final metric recomputation evidence drifted")
    else:
        publish_strict_json_no_clobber(metric_path, metric_payload)
    _snapshot_source_post(run_dir)
    current = _current_run_state(run_dir)
    if current == "FULL_EXTERNAL_AUDIT_SCORED":
        _set_run_state(run_dir, "FULL_SCIENTIFIC_COMPLETE")
    elif current != "FULL_SCIENTIFIC_COMPLETE":
        raise Amendment06IntegrityError("Scientific finalization state differs")
    ledger_path = run_dir / "provenance/execution_ledger.json"
    ledger = build_execution_ledger(run_dir)
    if ledger_path.exists():
        if strict_full_load_file(ledger_path) != ledger:
            raise Amendment06IntegrityError("Final execution ledger drifted")
    else:
        publish_strict_json_no_clobber(ledger_path, ledger)
    for status_path, status_bytes in (
        (run_dir / "RUN_STATUS.txt", _final_status(run_dir).encode("utf-8")),
        (run_dir / "BLOCKERS.md", b"# Blockers\n\nNone.\n"),
    ):
        if status_path.exists():
            if not _safe_regular(status_path, sha256_bytes(status_bytes)):
                raise Amendment06IntegrityError(
                    f"Final scientific status artifact drifted: {status_path}"
                )
        else:
            publish_bytes_no_clobber(status_path, status_bytes)
    strict_path = run_dir / "provenance/strict_json_validation.json"
    json_paths = sorted(
        (path for path in run_dir.rglob("*.json") if path != strict_path),
        key=lambda path: path.relative_to(run_dir).as_posix().encode("utf-8"),
    )
    strict_evidence = validate_strict_json_tree(run_dir, include=json_paths)
    strict_evidence.update({
        "run_id": run_dir.name, "self_excluded_by_non_circular_contract": True,
        "new_full_json_nonfinite_constants": 0,
    })
    if strict_path.exists():
        if strict_full_load_file(strict_path) != strict_evidence:
            raise Amendment06IntegrityError("Strict-JSON validation evidence drifted")
    else:
        publish_strict_json_no_clobber(strict_path, strict_evidence)
    strict_full_load_file(strict_path)
    publish_bytes_no_clobber(
        manifest_path, output_manifest_bytes(run_dir, require_exact_contract=True),
    )
    result = validate_output_manifest(run_dir)
    _progress("FULL_SCIENTIFIC_FINALIZATION", "COMPLETE", status="PASS", manifest_sha256=result["sha256"])
    return result


def validate_final_scientific_run(run_dir: Path) -> dict[str, Any]:
    if _current_run_state(run_dir) != "FULL_SCIENTIFIC_COMPLETE":
        raise Amendment06IntegrityError("Full Run is not scientifically complete")
    status = {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in (run_dir / "RUN_STATUS.txt").read_text().splitlines()
        if "=" in line
    }
    if status.get("RUN_STATE") != "FULL_SCIENTIFIC_COMPLETE":
        raise Amendment06IntegrityError("Final RUN_STATUS differs")
    strict_full_load_file(run_dir / "provenance/strict_json_validation.json")
    _validate_reporting_completion(run_dir)
    return validate_output_manifest(run_dir)


def _execute_scientific_full(
    *, run_dir: Path, preflight_run: Path, smoke_run: Path,
) -> Path:
    _validate_run_identity(run_dir)
    state = _current_run_state(run_dir)
    if state == "FULL_SCIENTIFIC_COMPLETE" and (
        run_dir / "OUTPUT_MANIFEST_FINAL.tsv"
    ).is_file():
        validate_final_scientific_run(run_dir)
        return run_dir
    initialize_run_evidence(run_dir)
    locked = load_locked_inputs(preflight_run)
    development = load_development_population(locked)
    cache_path = EXPECTED_RUNTIME_ROOT / "full_population" / run_dir.name
    if RUN_PHASES.index(state) >= RUN_PHASES.index("FULL_MODELS_FROZEN"):
        _install_post_freeze_guards()
        if not cache_path.is_dir() or cache_path.is_symlink():
            raise Amendment06IntegrityError("Frozen population cache cannot be rematerialized after model freeze")
    cache = load_or_build_population_cache(
        run_dir=run_dir, locked=locked, development=development,
    )
    ensure_blocked_no_pca(run_dir)
    state = _current_run_state(run_dir)
    if state == "FULL_IN_PROGRESS":
        for variant in TRAINED_VARIANTS:
            train_one_variant(
                run_dir=run_dir, locked=locked, cache=cache,
                development=development, variant=variant,
            )
        freeze_all_models(
            run_dir=run_dir, locked=locked, cache=cache, development=development,
        )
    else:
        validate_models_frozen_manifest(
            run_dir=run_dir, locked=locked, cache=cache, development=development,
        )
        _install_post_freeze_guards()
    audit = open_external_audit_after_freeze(
        run_dir=run_dir, locked=locked, cache=cache, development=development,
    )
    scoring_results = [
        score_one_external_variant(
            run_dir=run_dir, locked=locked, cache=cache,
            development=development, audit=audit, variant=variant,
        )
        for variant in TRAINED_VARIANTS
    ]
    complete_external_audit_phase(run_dir=run_dir, audit=audit)
    build_reports_and_bootstrap(
        run_dir=run_dir, locked=locked, cache=cache,
        audit=audit, scoring_results=scoring_results,
    )
    finalize_scientific_run(run_dir=run_dir, audit=audit)
    return run_dir


def package_dispatch(run_dir: Path, *, command_line: str = "") -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    _validate_run_identity(run_dir)
    validate_final_scientific_run(run_dir)
    package_module = importlib.import_module("amendment06_packaging")
    return package_module.package_existing_full(
        run_dir, command_line=command_line,
        preflight_run=ACCEPTED_PREFLIGHT, smoke_run=ACCEPTED_SMOKE,
    )


def run_full(
    *, preflight_run: Path, smoke_run: Path, command_line: str = "",
) -> Path:
    with amendment06_lock():
        assert_exact_invocation(preflight_run=preflight_run, smoke_run=smoke_run)
        admission = verify_reference_admission(
            preflight_run=preflight_run, smoke_run=smoke_run,
        )
        locked = load_locked_inputs(preflight_run)
        runs = existing_full_runs()
        if runs:
            intent = _reserve_full_run_id()
            if runs != [RESULTS_ROOT / str(intent["run_id"])]:
                raise Amendment06IntegrityError("Existing Full Run differs from reserved authorization")
            run_dir = runs[0]
            _validate_run_identity(run_dir)
        else:
            run_dir = _create_authorized_run(
                command_line=command_line, admission=admission, locked=locked,
            )
        _execute_scientific_full(
            run_dir=run_dir, preflight_run=Path(preflight_run), smoke_run=Path(smoke_run),
        )
        package_dispatch(run_dir, command_line=command_line)
        return run_dir


def resume_full(
    run_dir: Path, *, preflight_run: Path, smoke_run: Path,
    command_line: str = "",
) -> Path:
    with amendment06_lock():
        assert_exact_invocation(preflight_run=preflight_run, smoke_run=smoke_run)
        verify_reference_admission(preflight_run=preflight_run, smoke_run=smoke_run)
        target = Path(run_dir).resolve()
        _validate_run_identity(target)
        _execute_scientific_full(
            run_dir=target, preflight_run=Path(preflight_run), smoke_run=Path(smoke_run),
        )
        package_dispatch(target, command_line=command_line)
        return target


__all__ = [
    "AUTHORIZATION", "Amendment06Error", "Amendment06IntegrityError",
    "Amendment06ResumeError", "PostFreezeScientificCallError", "StrictJSONError",
    "append_authorized_edit_event", "assert_required_metrics_finite",
    "build_authorized_edit_event", "build_final_prerun_evidence",
    "make_test_evidence_record", "package_dispatch", "resume_full", "run_full",
    "strict_full_canonical_json_bytes", "strict_full_json_bytes", "strict_full_load_file",
    "strict_full_loads", "validate_strict_json_tree", "write_final_prerun_evidence",
]
