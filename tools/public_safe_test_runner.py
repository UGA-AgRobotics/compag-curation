#!/usr/bin/env python3
"""Run the exact rc6 public-safe unittest selection with zero-skip policy."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load test module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    tests = root / "tests"
    public_contract = _load(
        "rc6_selected_public_contract", tests / "test_public_contract.py"
    )
    dual_profile = _load(
        "rc6_selected_dual_profile", tests / "test_dual_profile_cli.py"
    )
    contract_fix = _load(
        "rc6_selected_acceptance", tests / "test_public_safe_acceptance.py"
    )

    suite = unittest.TestSuite()
    for method in (
        "test_public_module_imports_are_inert",
        "test_cli_help_every_subcommand",
        "test_all_variants_validate_and_plan",
    ):
        suite.addTest(public_contract.PublicContractTests(method))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(dual_profile.DualProfileCliTests))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(contract_fix.PublicSafeAcceptanceTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        print("rc6 acceptance forbids skipped active tests", file=sys.stderr)
        return 1
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
