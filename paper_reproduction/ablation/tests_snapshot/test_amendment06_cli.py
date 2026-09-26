from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


STUDY = Path(__file__).resolve().parents[1]
CODE = STUDY / "code"
sys.path.insert(0, str(CODE))
os.environ.setdefault("ABLATION_STUDY_ROOT", str(STUDY))

import run_study  # noqa: E402


PREFLIGHT = STUDY / "results/run_20260813_050405_9970c566_amendment02_preflight"
SMOKE = STUDY / "results/run_20260820_104526_263432_bc48422a_amendment04_cuda_smoke"


@pytest.mark.parametrize(
    "argv",
    [
        ["run_study.py", "--full"],
        ["run_study.py", "--resume", "/tmp/full"],
    ],
)
def test_legacy_full_and_resume_remain_blocked(monkeypatch, argv, capsys):
    monkeypatch.setattr(sys, "argv", argv)
    assert run_study.main() == 3
    assert "PROTOCOL_AMENDMENT_FORBIDS_FULL_TRAINING" in capsys.readouterr().err


def test_amendment06_flag_requires_full_resume_or_package(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_study.py", "--smoke", "--amendment-06-full-one"],
    )
    assert run_study.main() == 3
    assert "AMENDMENT_06_FLAG_REQUIRES_FULL_RESUME_OR_PACKAGE" in capsys.readouterr().err


@pytest.mark.parametrize("action", [["--full"], ["--resume", "/tmp/full"], ["--package", "/tmp/full"]])
def test_amendment06_actions_require_both_exact_reference_arguments(
    monkeypatch, capsys, action,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_study.py", *action, "--amendment-06-full-one"],
    )
    assert run_study.main() == 3
    assert "AMENDMENT_06_REQUIRES_EXACT_PREFLIGHT_AND_SMOKE_REFERENCES" in capsys.readouterr().err


def _base(action: list[str]) -> list[str]:
    return [
        "run_study.py",
        *action,
        "--preflight-run",
        str(PREFLIGHT),
        "--smoke-run",
        str(SMOKE),
        "--amendment-06-full-one",
    ]


@pytest.fixture
def accepted_a06_environment(monkeypatch):
    monkeypatch.setattr(run_study, "assert_amendment06_cli_environment", lambda: None)


def test_full_routes_only_to_new_amendment06_runner(
    monkeypatch, capsys, accepted_a06_environment,
):
    calls = []
    module = SimpleNamespace(
        run_full=lambda **kwargs: calls.append(kwargs) or Path("/tmp/a06-full")
    )
    real_import = run_study.importlib.import_module
    monkeypatch.setattr(
        run_study.importlib,
        "import_module",
        lambda name: module if name == "amendment06_full" else real_import(name),
    )
    monkeypatch.setattr(sys, "argv", _base(["--full"]))
    assert run_study.main() == 0
    assert calls == [{
        "preflight_run": PREFLIGHT,
        "smoke_run": SMOKE,
        "command_line": " ".join(_base(["--full"])),
    }]
    assert "AMENDMENT_06_FULL_RUN=/tmp/a06-full" in capsys.readouterr().out


def test_resume_targets_the_same_run_and_new_runner(monkeypatch, accepted_a06_environment):
    run = Path("/tmp/existing-amendment06-full")
    calls = []
    module = SimpleNamespace(
        resume_full=lambda **kwargs: calls.append(kwargs) or run
    )
    real_import = run_study.importlib.import_module
    monkeypatch.setattr(
        run_study.importlib,
        "import_module",
        lambda name: module if name == "amendment06_full" else real_import(name),
    )
    monkeypatch.setattr(sys, "argv", _base(["--resume", str(run)]))
    assert run_study.main() == 0
    assert calls[0]["run_dir"] == run
    assert calls[0]["preflight_run"] == PREFLIGHT
    assert calls[0]["smoke_run"] == SMOKE


def test_package_routes_to_isolated_full_packager(monkeypatch, accepted_a06_environment):
    run = Path("/tmp/existing-amendment06-full")
    calls = []
    module = SimpleNamespace(
        package_existing_full=lambda run_dir, **kwargs: calls.append((run_dir, kwargs)) or {"status": "PASS"}
    )
    real_import = run_study.importlib.import_module
    monkeypatch.setattr(
        run_study.importlib,
        "import_module",
        lambda name: module if name == "amendment06_packaging" else real_import(name),
    )
    monkeypatch.setattr(sys, "argv", _base(["--package", str(run)]))
    assert run_study.main() == 0
    assert calls[0][0] == run
    assert calls[0][1]["preflight_run"] == PREFLIGHT
    assert calls[0][1]["smoke_run"] == SMOKE


def test_wrapper_exports_the_explicit_amendment_runtime_root():
    text = (CODE / "run_ablation_study.sh").read_text()
    assert 'export ABLATION_RUNTIME_ROOT="$RUNTIME_ROOT"' in text


def test_package_rejects_wrong_or_missing_runtime_before_dispatch(monkeypatch, capsys):
    monkeypatch.delenv("ABLATION_RUNTIME_ROOT", raising=False)
    monkeypatch.setattr(sys, "argv", _base(["--package", "/tmp/full"]))
    assert run_study.main() == 3
    assert "AMENDMENT_06_RUNTIME_ROOT_IS_REQUIRED" in capsys.readouterr().err


def test_amendment06_forbids_primary_run_argument(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", [*_base(["--full"]), "--primary-run", "/tmp/primary"],
    )
    assert run_study.main() == 3
    assert "AMENDMENT_06_FORBIDS_PRIMARY_RUN" in capsys.readouterr().err
