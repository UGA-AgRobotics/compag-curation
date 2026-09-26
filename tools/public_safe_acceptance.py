#!/usr/bin/env python3
"""Fresh-install, model-free acceptance for the rc9 browser-start source candidate.

This runner deliberately exercises only base CLI and selected synthetic tests.
It does not build release artifacts, load a model, run inference, fit, rescore,
contact a network endpoint, or treat excluded/deferred tests as passing credit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


SCHEMA = "compag-rc9-public-safe-acceptance/v3"
PACKAGING_REVISION = "rc10-repair-r5.4"
APPLICATION_VERSION = "1.9.4rc10"
BASE_SOURCE_SHA256 = "0c12a1ce2a51e82da607568dc325f65d42955ef284e28df96a749ed627be13d9"
ACTIVE_TEST_SCOPES = (
    "tests/test_public_safe_acceptance.py::PublicSafeAcceptanceTests",
    "tests/test_dual_profile_cli.py::DualProfileCliTests",
    "tests/test_public_contract.py::PublicContractTests::{test_public_module_imports_are_inert,test_cli_help_every_subcommand,test_all_variants_validate_and_plan}",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_revision(record: dict[str, object]) -> None:
    _require(record.get("packaging_revision") == PACKAGING_REVISION, "unexpected public packaging revision")
    _require(record.get("application_version") == APPLICATION_VERSION, "unexpected application version")
    _require(record.get("base_source_zip_sha256") == BASE_SOURCE_SHA256, "unexpected rc9 R1 source baseline")
    _require(record.get("parent_packaging_revision") == "rc9-paper-yolo-r5.3", "unexpected parent packaging revision")
    _require(record.get("parent_source_zip_sha256") == "f7c3c5eb79021dda5c2937302c7598f06e81a5660a1471bc00bed3c20a623b2a", "unexpected parent source ZIP")
    _require(record.get("separate_assets_required") is False, "r5.4 must include its four startup assets")
    _require(record.get("bundled_asset_count") == 4, "r5.4 must declare four startup assets")
    _require(record.get("application_code_changed") is True, "r5.4 must declare the hybrid implementation change")
    _require(record.get("public_package_bytes_changed") is True, "r5.4 source archives must change")
    _require(record.get("application_wheel_bytes_changed") is True, "r5.4 must declare the rebuilt wheel")
    _require(record.get("review_zoom_preserved_on_decision") is True, "r5.4 must retain zoom continuity")
    _require(record.get("onboarding_source_changed") is True, "r5.3 browser labels must be declared")
    _require(record.get("paper_training_helper_added") is True, "r5.4 must retain the paper training helper")
    _require(record.get("yolo_inference_overlay_added") is True, "r5.4 must retain the YOLO overlay")
    _require(record.get("yolo_retraining_in_browser") is False, "r5.4 must not claim browser YOLO retraining")
    _require(record.get("paper_optional_hybrid_formulation_implemented") is True, "r5.4 must declare the paper hybrid policy")
    _require(record.get("paper_optional_hybrid_labels_aligned") is True, "r5.4 must declare paper hybrid labels")
    _require(record.get("prior_fusion_receipts_reopen_with_original_policy") is True, "r5.4 must retain prior fusion receipts")
    _require(record.get("publication_authorized") is False, "local acceptance cannot authorize publication")
    _require(record.get("remote_write_authorized") is False, "this local acceptance cannot authorize upload")


def _display(argv: Sequence[str], *, python: Path, source: Path, work: Path) -> list[str]:
    replacements = {
        str(python): "$ACCEPTANCE_PYTHON",
        str(source): "$FRESH_SOURCE",
        str(work): "$ACCEPTANCE_WORK",
    }
    result: list[str] = []
    for value in argv:
        rendered = str(value)
        for old, new in sorted(replacements.items(), key=lambda item: -len(item[0])):
            rendered = rendered.replace(old, new)
        result.append(rendered)
    return result


def _run(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    python: Path,
    source: Path,
    work: Path,
) -> dict[str, object]:
    started = time.monotonic_ns()
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
    record: dict[str, object] = {
        "argv": _display(argv, python=python, source=source, work=work),
        "elapsed_ms": elapsed_ms,
        "returncode": completed.returncode,
        "stderr": completed.stderr,
        "stdout": completed.stdout,
    }
    _require(
        completed.returncode == 0,
        "acceptance command failed: " + " ".join(record["argv"]),
    )
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded rc9 public-source acceptance in a fresh local copy."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve(strict=True)
    work = args.work_dir.absolute()
    receipt = args.receipt.absolute()
    _require(root.is_dir() and not root.is_symlink(), "--root must be a regular directory")
    _require(not work.exists(), "--work-dir must not exist")
    _require(not receipt.exists(), "--receipt must not exist")
    _require(receipt.parent.is_dir(), "--receipt parent must already exist")
    _require(sys.version_info[:2] == (3, 12), "acceptance requires CPython 3.12")
    _require(platform.python_implementation() == "CPython", "acceptance requires CPython")
    revision = json.loads((root / "PUBLIC_PACKAGING_REVISION.json").read_text("utf-8"))
    _validate_revision(revision)

    work.mkdir(mode=0o755)
    source = work / "fresh-source"
    venv = work / "venv"
    shutil.copytree(root, source, symlinks=False)
    commands: list[dict[str, object]] = []

    bootstrap_env = {
        "HOME": str(work / "home"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PIP_NO_INPUT": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "SOURCE_DATE_EPOCH": "1788393600",
    }
    (work / "home").mkdir(mode=0o700)
    # The invoking reviewed Python 3.12 environment supplies the exact
    # setuptools/wheel build backend. Runtime acceptance remains base-only.
    commands.append(
        _run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
            cwd=work,
            env=bootstrap_env,
            python=Path(sys.executable),
            source=source,
            work=work,
        )
    )
    python = venv / "bin/python"
    env = dict(bootstrap_env)
    env["PATH"] = f"{venv / 'bin'}:/usr/bin:/bin"
    env["COMPAG_PUBLIC_ROOT"] = str(source)
    env["COMPAG_ACCEPTANCE_NO_NETWORK"] = "1"

    command_sets = [
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-cache-dir",
            "--no-build-isolation",
            "--no-deps",
            str(source),
        ],
        [str(python), "-m", "pip", "check"],
        [str(python), "-I", "-B", "-m", "compag_curation", "--help"],
        [str(python), "-I", "-B", "-m", "compag_curation", "--version"],
        [
            str(python),
            "-I",
            "-B",
            "-m",
            "compag_curation",
            "doctor",
            "--profile",
            "base",
        ],
    ]
    for command in command_sets:
        commands.append(
            _run(
                command,
                cwd=source,
                env=env,
                python=python,
                source=source,
                work=work,
            )
        )

    test_record = _run(
        [
            str(python),
            "-I",
            "-B",
            str(source / "tools/public_safe_test_runner.py"),
            "--root",
            str(source),
        ],
        cwd=source,
        env=env,
        python=python,
        source=source,
        work=work,
    )
    combined = str(test_record["stdout"]) + "\n" + str(test_record["stderr"])
    match = re.search(r"Ran ([0-9]+) tests?", combined)
    _require(match is not None and int(match.group(1)) > 0, "no active tests executed")
    _require("skipped=" not in combined.lower(), "skips are not acceptance credit")
    test_record["scopes"] = list(ACTIVE_TEST_SCOPES)
    test_record["tests_run"] = int(match.group(1))
    commands.append(test_record)

    inventory = json.loads((source / "PUBLIC_TEST_INVENTORY.json").read_text("utf-8"))
    counts: dict[str, int] = {}
    for row in inventory["tests"]:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "packaging_revision": PACKAGING_REVISION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_allowlist_sha256": _sha256(source / "PUBLIC_SOURCE_ALLOWLIST.json"),
        "network_operations": 0,
        "model_loads_or_unpickles": 0,
        "training_inference_rescoring": 0,
        "active_test_scopes": list(ACTIVE_TEST_SCOPES),
        "active_tests_run": int(test_record["tests_run"]),
        "active_test_skips": 0,
        "excluded_test_counts": counts,
        "excluded_tests_received_pass_credit": False,
        "temporary_source_install_build": {
            "application_version": APPLICATION_VERSION,
            "packaging_revision": PACKAGING_REVISION,
            "role": "EPHEMERAL_LOCAL_ACCEPTANCE_INSTALL_NOT_A_RELEASE_ARTIFACT",
            "published_or_retained_as_binary": False,
            "pip_stdout_sha256": hashlib.sha256(
                str(commands[1]["stdout"]).encode("utf-8")
            ).hexdigest(),
        },
        "commands": commands,
    }
    with receipt.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("schema", "status", "packaging_revision", "active_tests_run")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema": SCHEMA, "status": "FAIL", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)
