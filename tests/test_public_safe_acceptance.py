from __future__ import annotations

import io
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class PublicSafeAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(os.environ["COMPAG_PUBLIC_ROOT"]).resolve(strict=True)

    def test_packaging_revision_is_exact_current_local_release_candidate(self) -> None:
        record = json.loads(
            (self.root / "PUBLIC_PACKAGING_REVISION.json").read_text(encoding="utf-8")
        )
        self.assertEqual(record["application_version"], "1.9.4rc10")
        self.assertEqual(
            record["packaging_revision"], "rc10-repair-r5.4"
        )
        self.assertEqual(record["parent_packaging_revision"], "rc9-paper-yolo-r5.3")
        self.assertEqual(record["parent_source_zip_sha256"],
                         "f7c3c5eb79021dda5c2937302c7598f06e81a5660a1471bc00bed3c20a623b2a")
        self.assertTrue(record["paper_optional_hybrid_formulation_implemented"])
        self.assertTrue(record["paper_optional_hybrid_labels_aligned"])
        self.assertTrue(record["prior_fusion_receipts_reopen_with_original_policy"])
        self.assertTrue(record["local_artifact_staging_authorized"])
        self.assertFalse(record["publication_authorized"])
        self.assertFalse(record["remote_write_authorized"])
        self.assertEqual(record["author_selected_public_scope"], ["source_code", "native_r92_model_zip", "ten_original_photo_zip"])

    def test_four_bundled_startup_files_match_pinned_identities(self) -> None:
        bundled = self.root / "bundled-assets"
        wheel_name = "compag_curation-1.9.4rc10-py3-none-any.whl"
        sidecar = wheel_name + ".sha256"
        sidecar_parts = (bundled / sidecar).read_text(encoding="utf-8").split()
        self.assertEqual(len(sidecar_parts), 2)
        self.assertEqual(sidecar_parts[1], wheel_name)
        self.assertRegex(sidecar_parts[0], r"^[0-9a-f]{64}$")
        launcher = self.root / "start_compag.py"
        spec = importlib.util.spec_from_file_location("rc9_launcher_for_asset_test", launcher)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual((module.WHEEL, module.WHEEL_SHA256), (wheel_name, sidecar_parts[0]))
        expected = {
            wheel_name: sidecar_parts[0],
            "COMPAG_CJ_R92_INFERENCE_SAMPLE_20260924T200000Z.zip": "fffef7a85192b99dcca3bcc5750215150d2ca9659121363198041244bbfe57c1",
            "COMPAG_R92_TEN_PHOTO_INPUTS_RC6_R3_20260924.zip": "344c8d767a1719403a3713e0f51587971f4ab9ea18e1d1a44034ed72c9297914",
        }
        self.assertEqual({path.name for path in bundled.iterdir()}, set(expected) | {sidecar})
        for name, digest in expected.items():
            path = bundled / name
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertEqual(sidecar_parts, [expected[wheel_name], wheel_name])

    def test_runner_rejects_wrong_revision_or_application_identity(self) -> None:
        runner = self.root / "tools/public_safe_acceptance.py"
        spec = importlib.util.spec_from_file_location("rc9_acceptance_for_identity_test", runner)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        valid = json.loads((self.root / "PUBLIC_PACKAGING_REVISION.json").read_text(encoding="utf-8"))
        module._validate_revision(valid)
        for key, wrong in (("packaging_revision", "rc6-public-readiness-r1"),
                           ("application_version", "1.9.3"),
                           ("base_source_zip_sha256", "0" * 64),
                           ("parent_source_zip_sha256", "0" * 64),
                           ("paper_optional_hybrid_formulation_implemented", False)):
            mutated = dict(valid)
            mutated[key] = wrong
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                module._validate_revision(mutated)

    def test_test_inventory_is_complete_and_active_lane_is_exact(self) -> None:
        record = json.loads(
            (self.root / "PUBLIC_TEST_INVENTORY.json").read_text(encoding="utf-8")
        )
        present = {path.relative_to(self.root).as_posix() for path in (self.root / "tests").glob("test_*.py")}
        declared = {row["path"] for row in record["tests"]}
        self.assertEqual(declared, present)
        historical = {path.relative_to(self.root).as_posix() for path in (self.root / "historical_tests").glob("*.py")}
        self.assertEqual({row["path"] for row in record["historical_tests"]}, historical)
        self.assertTrue(all(not row["pass_credit"] for row in record["historical_tests"]))
        self.assertTrue(all(row["path"].startswith("tests/") for row in record["tests"]))
        self.assertTrue(all(row["path"].startswith("historical_tests/") for row in record["historical_tests"]))
        active = {
            row["path"].removeprefix("tests/")
            for row in record["tests"]
            if row["classification"] == "ACTIVE_PUBLIC_SAFE_EXECUTED"
        }
        self.assertEqual(
            active,
            {
                "test_public_safe_acceptance.py",
                "test_dual_profile_cli.py",
                "test_public_contract.py",
            },
        )
        self.assertTrue(
            all(row["pass_credit"] == (row["path"].removeprefix("tests/") in active) for row in record["tests"])
        )

    def test_active_validation_command_is_documented(self) -> None:
        inventory = json.loads((self.root / "PUBLIC_TEST_INVENTORY.json").read_text(encoding="utf-8"))
        docs = (self.root / "docs/DEVELOPMENT.md").read_text(encoding="utf-8")
        self.assertIn("tools/public_safe_acceptance.py", inventory["active_command"])
        self.assertIn("tools/public_safe_acceptance.py", docs)
        self.assertIn("rc10-repair-r5.4", docs)

    def test_r53_release_note_is_in_source_allowlist(self) -> None:
        record = json.loads((self.root / "PUBLIC_SOURCE_ALLOWLIST.json").read_text(encoding="utf-8"))
        paths = record["paths"]
        self.assertEqual(record["packaging_revision"], "rc10-repair-r5.4")
        self.assertEqual(paths, sorted(set(paths)))
        self.assertIn("PUBLIC_RELEASE_NOTES_RC10_R5_4.md", paths)
        self.assertIn("PUBLIC_RELEASE_NOTES_RC9_R5_3.md", paths)
        self.assertTrue((self.root / "PUBLIC_RELEASE_NOTES_RC9_R5_3.md").is_file())
        self.assertEqual(record["paths_sha256"],
                         hashlib.sha256(("\n".join(paths) + "\n").encode("ascii")).hexdigest())

    def test_browser_verifies_saved_training_without_python_311_file_digest(self) -> None:
        browser = self.root / "easy_start_web.py"
        spec = importlib.util.spec_from_file_location("rc9_browser_for_py310_test", browser)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def digest(path: Path) -> str:
            value = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(chunk)
            return value.hexdigest()

        def save(path: Path, value: dict) -> None:
            path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            round_dir = workspace / "rounds" / "round_py310_smoke"
            model = round_dir / "xgb_model"
            model_set = round_dir / "model_set"
            model.mkdir(parents=True)
            model_set.mkdir()
            snapshot = round_dir / "training_snapshot.json"
            save(snapshot, {"status": "PASS", "rows": [{"group_id": "card_1", "action": "accept"}]})
            classifier = model / "classifier.ubj"
            classifier.write_bytes(b"synthetic-checksum-only-fixture")
            save(model / "TRAINING_RECIPE.json", {
                "snapshot_sha256": digest(snapshot),
                "policy": "SMALL_DATA_OPERATIONAL_NO_HOLDOUT",
                "fitted_boosting_rounds": 1,
            })
            save(model / "PROJECT_MODEL_MANIFEST.json", {
                "status": "PASS", "classifier_sha256": digest(classifier),
                "snapshot_sha256": digest(snapshot),
                "files": {"classifier.ubj": digest(classifier)},
            })
            model_set_file = model_set / "MODEL_SET.json"
            save(model_set_file, {"status": "MODEL_SET_VERIFIED", "xgb_bundle": str(model)})
            save(workspace / "LATEST_MODEL_SET.json", {
                "path": str(model_set), "model_set_sha256": digest(model_set_file),
            })
            bridge = SimpleNamespace(workspace=workspace, backend=SimpleNamespace(_sha256=digest))
            with mock.patch.object(hashlib, "file_digest", None, create=True):
                evidence = module._Bridge._latest_training_evidence(
                    bridge, {"round_id": "round_py310_smoke", "model_set": str(model_set)}
                )
            self.assertEqual(evidence["status"], "VERIFIED")
            self.assertEqual(evidence["row_count"], 1)

    def test_current_docs_do_not_route_to_omitted_helpers(self) -> None:
        historical = {
            "CHANGELOG.md",
            "PUBLIC_RELEASE_NOTES.md",
            "docs/UPSTREAM_README_1.9.3_PRIVATE_BUILD_CONTEXT.md",
        }
        forbidden = (
            "tools/public_qa.py",
            "tools/install_gpu_profile.py",
            "tools/privacy_scan.py",
            "tools/build_public_metadata.py",
            "tools/run_documented_commands.py",
            "DOCUMENTED_COMMAND_EXECUTION_MATRIX.csv",
            "public_file_inventory.csv",
        )
        active_docs = sorted(
            path for path in self.root.rglob("*.md")
            if path.relative_to(self.root).as_posix() not in historical
        )
        self.assertGreater(len(active_docs), 10)
        for path in active_docs:
            relative = path.relative_to(self.root).as_posix()
            text = path.read_text(encoding="utf-8")
            for value in forbidden:
                self.assertNotIn(value, text, relative)

    def test_synthetic_project_plan_is_model_free_and_inert(self) -> None:
        from compag_curation.cli import main

        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "synthetic-project"
            with redirect_stdout(io.StringIO()) as capture:
                self.assertEqual(main(["init-project", "--output", str(project)]), 0)
            self.assertEqual(json.loads(capture.getvalue())["status"], "PASS")
            with redirect_stdout(io.StringIO()) as capture:
                self.assertEqual(main(["plan", "--config", str(project / "config.toml")]), 0)
            plan = json.loads(capture.getvalue())
            self.assertEqual(plan["status"], "VALID")
            self.assertFalse(plan["scientific_execution"])

    def test_restricted_model_request_fails_before_any_model_loader(self) -> None:
        from compag_curation.review import published_model

        restricted_root = published_model._resource_root()
        self.assertFalse(restricted_root.exists())
        with self.assertRaises(FileNotFoundError) as failure:
            published_model.load_published_model_transfer_assets()
        self.assertIn("compag_cj_r92", str(failure.exception))


if __name__ == "__main__":
    unittest.main()
