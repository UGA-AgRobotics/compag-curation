#!/usr/bin/env python3
"""Command-line controller for revision item 6 protocol amendments."""

from __future__ import annotations

import argparse
import importlib
import os
import shlex
import sys
import traceback
from pathlib import Path

import ablation_core as core


AMENDMENT_TRAINING_BLOCK = (
    "PROTOCOL_AMENDMENT_FORBIDS_FULL_TRAINING: this corrective phase permits "
    "amended preflight and, only after all gates pass, CUDA smoke. Full, resume, "
    "and stability execution must await external review."
)

AMENDMENT06_CUDA_ENV = Path(
    "/REVIEWER_INPUT_ROOT/SAM_ablation_cuda_xgb211_wheel"
)
AMENDMENT06_CUDA_PYTHON = AMENDMENT06_CUDA_ENV / "bin/python"
AMENDMENT06_RUNTIME_ROOT = Path(
    "/REVIEWER_INPUT_ROOT/runtime"
)


def assert_amendment06_cli_environment() -> None:
    """Fail before every Full/resume/package dispatch outside the sealed env."""

    runtime_value = os.environ.get("ABLATION_RUNTIME_ROOT", "")
    if not runtime_value:
        raise core.ScientificBlocker("AMENDMENT_06_RUNTIME_ROOT_IS_REQUIRED")
    if (
        Path(sys.executable).resolve() != AMENDMENT06_CUDA_PYTHON.resolve()
        or Path(sys.prefix).resolve() != AMENDMENT06_CUDA_ENV.resolve()
        or Path(runtime_value).resolve() != AMENDMENT06_RUNTIME_ROOT.resolve()
        or not AMENDMENT06_RUNTIME_ROOT.is_dir()
        or AMENDMENT06_RUNTIME_ROOT.is_symlink()
    ):
        raise core.ScientificBlocker(
            "AMENDMENT_06_REQUIRES_EXACT_CUDA_INTERPRETER_AND_RUNTIME_ROOT"
        )
    xgb = importlib.import_module("xgboost")
    sklearn = importlib.import_module("sklearn")
    if not (
        xgb.__version__ == "2.1.1"
        and sklearn.__version__ == "1.7.2"
        and xgb.build_info().get("USE_CUDA") is True
    ):
        raise core.ScientificBlocker(
            "AMENDMENT_06_REQUIRES_XGBOOST_2_1_1_SKLEARN_1_7_2_CUDA_BUILD"
        )


def package_amended_or_compatible(run_dir: Path):
    """Prefer the Amendment 01 packager without breaking legacy package access."""
    try:
        amendment = importlib.import_module("amendment_core")
    except ModuleNotFoundError as exc:
        if exc.name != "amendment_core":
            raise
    else:
        packager = getattr(amendment, "package_amended_run", None)
        if callable(packager):
            return packager(run_dir)
    return core.package_existing_run(run_dir)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Cotton-jassid candidate-classifier ablation study")
    group = result.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--amendment-02-preflight", action="store_true")
    group.add_argument("--legacy-blocked-preflight", action="store_true")
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--full", action="store_true")
    group.add_argument("--stability", action="store_true")
    group.add_argument("--resume", type=Path)
    group.add_argument("--package", dest="package_run", type=Path)
    result.add_argument(
        "--preflight-run",
        type=Path,
        help="Validated protocol-amendment Preflight Run required by --smoke",
    )
    result.add_argument(
        "--amendment-03-corrigendum-one",
        action="store_true",
        help=(
            "Consume the one authorized Amendment 03 replacement Smoke; valid "
            "only with --smoke and the pinned Amendment 02 Preflight"
        ),
    )
    result.add_argument(
        "--amendment-04-csv-roundtrip-one",
        action="store_true",
        help=(
            "Consume the one authorized Amendment 04 CSV round-trip recovery "
            "Smoke; valid only with --smoke and the pinned Amendment 02 Preflight"
        ),
    )
    result.add_argument(
        "--amendment-06-full-one",
        action="store_true",
        help=(
            "Consume or resume the single externally authorized Amendment 06 "
            "Full scientific Run; valid only with --full, --resume, or --package"
        ),
    )
    result.add_argument(
        "--smoke-run",
        type=Path,
        help="Exact accepted CUDA Smoke required by Amendment 06 Full actions",
    )
    result.add_argument("--primary-run", type=Path, help="Primary run for --stability")
    return result


def main() -> int:
    args = parser().parse_args()
    command_line = shlex.join(sys.argv)
    try:
        if args.amendment_03_corrigendum_one and not args.smoke:
            raise core.ScientificBlocker(
                "AMENDMENT_03_CORRIGENDUM_FLAG_REQUIRES_SMOKE"
            )
        if args.amendment_04_csv_roundtrip_one and not args.smoke:
            raise core.ScientificBlocker(
                "AMENDMENT_04_CSV_ROUNDTRIP_FLAG_REQUIRES_SMOKE"
            )
        if (
            args.amendment_03_corrigendum_one
            and args.amendment_04_csv_roundtrip_one
        ):
            raise core.ScientificBlocker(
                "AMENDMENT_RECOVERY_FLAGS_ARE_MUTUALLY_EXCLUSIVE"
            )
        amendment06_action = bool(args.full or args.resume or args.package_run)
        if args.amendment_06_full_one and not amendment06_action:
            raise core.ScientificBlocker(
                "AMENDMENT_06_FLAG_REQUIRES_FULL_RESUME_OR_PACKAGE"
            )
        if args.amendment_06_full_one and (
            args.amendment_03_corrigendum_one
            or args.amendment_04_csv_roundtrip_one
        ):
            raise core.ScientificBlocker(
                "AMENDMENT_RECOVERY_FLAGS_ARE_MUTUALLY_EXCLUSIVE"
            )
        if args.amendment_06_full_one and (
            args.preflight_run is None or args.smoke_run is None
        ):
            raise core.ScientificBlocker(
                "AMENDMENT_06_REQUIRES_EXACT_PREFLIGHT_AND_SMOKE_REFERENCES"
            )
        if args.amendment_06_full_one:
            if args.primary_run is not None:
                raise core.ScientificBlocker(
                    "AMENDMENT_06_FORBIDS_PRIMARY_RUN_OR_STABILITY_ARGUMENTS"
                )
            assert_amendment06_cli_environment()
        if args.preflight:
            amended_preflight = importlib.import_module("build_amended_preflight")
            amended_preflight.build()
        elif args.amendment_02_preflight:
            amendment02 = importlib.import_module("build_amendment02_preflight")
            result = amendment02.build()
            print(f"AMENDMENT_02_PREFLIGHT_RUN={result}")
        elif args.legacy_blocked_preflight:
            legacy = importlib.import_module("build_blocked_preflight")
            result = legacy.build()
            print("LEGACY_PREFLIGHT=BLOCKED_BEFORE_SCIENTIFIC_TRAINING")
            print(f"LEGACY_PREFLIGHT_RUN={result}")
        elif args.smoke:
            if args.preflight_run is None:
                raise core.ScientificBlocker(
                    "PROTOCOL_SMOKE_REQUIRES_PREFLIGHT_RUN: pass --preflight-run "
                    "with a validated amended-preflight Run directory."
                )
            amendment = importlib.import_module("amendment_core")
            smoke = amendment.run_cuda_smoke(
                preflight_run=args.preflight_run,
                command_line=command_line,
                amendment03_replacement=args.amendment_03_corrigendum_one,
                amendment04_csv_roundtrip=args.amendment_04_csv_roundtrip_one,
            )
            print(f"CUDA_SMOKE_RUN={smoke}")
        elif args.full:
            if not args.amendment_06_full_one:
                raise core.ScientificBlocker(AMENDMENT_TRAINING_BLOCK)
            amendment06 = importlib.import_module("amendment06_full")
            result = amendment06.run_full(
                preflight_run=args.preflight_run,
                smoke_run=args.smoke_run,
                command_line=command_line,
            )
            print(f"AMENDMENT_06_FULL_RUN={result}")
        elif args.resume:
            if not args.amendment_06_full_one:
                raise core.ScientificBlocker(AMENDMENT_TRAINING_BLOCK)
            amendment06 = importlib.import_module("amendment06_full")
            result = amendment06.resume_full(
                run_dir=args.resume,
                preflight_run=args.preflight_run,
                smoke_run=args.smoke_run,
                command_line=command_line,
            )
            print(f"AMENDMENT_06_FULL_RUN={result}")
        elif args.package_run:
            if args.amendment_06_full_one:
                amendment06_package = importlib.import_module(
                    "amendment06_packaging"
                )
                result = amendment06_package.package_existing_full(
                    args.package_run,
                    command_line=command_line,
                    preflight_run=args.preflight_run,
                    smoke_run=args.smoke_run,
                )
                print(f"AMENDMENT_06_FULL_PACKAGE={result}")
            else:
                print(package_amended_or_compatible(args.package_run))
        elif args.stability:
            raise core.ScientificBlocker(AMENDMENT_TRAINING_BLOCK)
        return 0
    except core.ScientificBlocker as exc:
        print(f"SCIENTIFIC_BLOCKER: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        try:
            amendment = importlib.import_module("amendment_core")
            amendment_blocker = getattr(amendment, "ScientificBlocker", ())
        except Exception:
            amendment_blocker = ()
        if amendment_blocker and isinstance(exc, amendment_blocker):
            print(f"SCIENTIFIC_BLOCKER: {exc}", file=sys.stderr)
            return 3
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
