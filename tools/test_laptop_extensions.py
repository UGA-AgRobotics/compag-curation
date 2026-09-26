#!/usr/bin/env python3
"""Focused model-free checks for the new laptop entry and asset preflight."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from compag_curation.only_codes.asset_status import inspect_project_assets
from compag_curation.only_codes.guide import run_guide
from compag_curation.only_codes.project import import_project


class LaptopExtensionsTests(unittest.TestCase):
    def test_missing_assets_block_before_runtime_and_do_not_change_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "original"
            original.mkdir()
            project = import_project(legacy_root=original, output=root / "project",
                                     sam2_repo_root=root / "sam2", torch_home=root / "torch",
                                     image="IMG_9510")
            before = sorted(p.relative_to(original) for p in original.rglob("*"))
            report = inspect_project_assets(project, image="IMG_9510")
            self.assertEqual(report["stages"]["infer"]["status"], "BLOCKED_INPUTS")
            self.assertEqual(report["stages"]["merge"]["status"], "BLOCKED_INPUTS")
            self.assertEqual(before, sorted(p.relative_to(original) for p in original.rglob("*")))
            with self.assertRaisesRegex(ValueError, "IMG_1234"):
                inspect_project_assets(project, image="../other")

    def test_invalid_bundle_is_reported_without_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "original"
            original.mkdir()
            project = import_project(legacy_root=original, output=root / "project",
                                     sam2_repo_root=root / "sam2", torch_home=root / "torch")
            bundle = project.root / "bundles" / "r1_hybrid"
            bundle.mkdir()
            (bundle / "LEGACY_MODEL_BUNDLE.json").write_text(json.dumps({
                "schema": "compag-only-codes-legacy-model-bundle/v1",
                "verification": {"status": "PASS_EXACT"},
                "files": {"schema.json": "0" * 64},
            }), encoding="utf-8")
            project.manifest["model_bundles"]["r1_hybrid"] = {"bundle_dir": "bundles/r1_hybrid"}
            project.save()
            report = inspect_project_assets(project, model_round_dir="r1_hybrid")
            bundle_item = report["stages"]["infer"]["items"][1]
            self.assertEqual(bundle_item["status"], "INVALID")

    def test_launcher_can_exit_without_a_project(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {"XDG_CONFIG_HOME": folder}):
            with patch("builtins.input", return_value="0"), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(run_guide(), 0)
            self.assertIn("Import a compatible", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
