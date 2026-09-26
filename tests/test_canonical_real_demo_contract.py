from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HEAVY_MODULES = {
    "cv2", "imblearn", "matplotlib", "numpy", "pandas", "scipy", "sklearn",
    "torch", "torchvision", "xgboost",
}

CANONICAL_ASSETS = (
    ("resnet50-imagenet1k-v2-weights", "resnet50-11ad3fa6.pth"),
    ("sam2-apache-license", "SAM2-APACHE-2.0.txt"),
    ("sam2.1-hiera-large-checkpoint", "sam2.1_hiera_large.pt"),
    ("sam2.1-hiera-large-config", "sam2.1_hiera_l.yaml"),
    ("torchvision-bsd-license", "TORCHVISION-BSD-3-CLAUSE.txt"),
)


def _proposals() -> list[dict[str, str]]:
    rows = []
    for group_index in range(8):
        group = f"synthetic-group-{group_index + 1:02d}"
        image_sha = hashlib.sha256(f"image-{group}".encode("ascii")).hexdigest()
        for proposal_index in range(2):
            proposal = hashlib.sha256(
                f"proposal-{group}-{proposal_index}".encode("ascii")
            ).hexdigest()
            rows.append({
                "proposal_id": proposal,
                "proposal_sha256": proposal,
                "image_id": image_sha,
                "image_sha256": image_sha,
                "group_id": group,
            })
    return rows


def _gpu_dependencies(cuda_visible_devices: str = "3") -> dict[str, object]:
    from compag_curation.public_backend import SCIENCE_GPU_IMPORTS
    from compag_curation.public_io import compact_json_sha256
    from compag_curation.quick_demo import _expected_demo_gpu_archive_provenance
    from compag_curation.runtime_lock import (
        SCIENCE_GPU_LOCK_SHA256,
        SCIENCE_GPU_PYTHON,
        SCIENCE_GPU_SAM2_COMMIT,
        SCIENCE_GPU_SAM2_URL,
        SCIENCE_GPU_VERSIONS,
        cuda_visibility_identity,
        science_gpu_runtime_policy,
    )

    record: dict[str, object] = {
        "schema": "compag-curation-science-gpu-dependencies/v2",
        "python": SCIENCE_GPU_PYTHON,
        "implementation": "CPython",
        "platform": "Linux-fixture-x86_64",
        "versions": SCIENCE_GPU_VERSIONS,
        "missing": [],
        "sam2_direct_url": {
            "url": SCIENCE_GPU_SAM2_URL,
            "vcs_info": {
                "vcs": "git",
                "commit_id": SCIENCE_GPU_SAM2_COMMIT,
            },
        },
        "archive_provenance": _expected_demo_gpu_archive_provenance(),
        "lock_sha256": SCIENCE_GPU_LOCK_SHA256,
        "lock_resource": "resources/science_gpu_lock.json",
        "package_implementation": {
            "schema": "compag-curation-package-implementation-identity/v1",
            "sha256": "a" * 64,
            "file_count": 1,
            "size_bytes": 1,
        },
        "imports": list(SCIENCE_GPU_IMPORTS),
        "package_installation": "SYS_PREFIX_DISTRIBUTION_NON_EDITABLE_ISOLATED_NO_BYTECODE",
        "requested_device": "cuda",
        "resolved_device": "cuda:0",
        "cuda_visibility": cuda_visibility_identity(
            {"CUDA_VISIBLE_DEVICES": cuda_visible_devices}
        ),
        "gpu_runtime_policy": science_gpu_runtime_policy(),
        "cuda": {
            "device_index": 0,
            "device_name": "Fixture GPU",
            "compute_capability": [8, 9],
            "total_memory_bytes": 24 * 1024**3,
            "torch_cuda_runtime": "13.2",
            "cudnn_version": 92000,
            "driver_api_version": 13020,
        },
        "xgboost_build": {"USE_CUDA": True},
        "sam2_cuda_extension": {
            "schema": "compag-curation-sam2-cuda-extension-attestation/v1",
            "distribution": "sam-2",
            "distribution_version": SCIENCE_GPU_VERSIONS["sam-2"],
            "module": "sam2._C",
            "distribution_path": "sam2/_C.so",
            "sha256": "b" * 64,
            "size_bytes": 1024,
            "probe": {
                "function": "get_connected_componnets",
                "input": {
                    "shape": [1, 1, 4, 4],
                    "dtype": "uint8",
                    "device": "cuda:0",
                    "nonzero": 0,
                },
                "outputs": [
                    {
                        "shape": [1, 1, 4, 4],
                        "dtype": "int32",
                        "device": "cuda:0",
                        "nonzero": 0,
                    },
                    {
                        "shape": [1, 1, 4, 4],
                        "dtype": "int32",
                        "device": "cuda:0",
                        "nonzero": 0,
                    },
                ],
                "cpu_fallback": False,
            },
        },
    }
    record["sha256"] = compact_json_sha256(record)
    return record


class CanonicalRealDemoContractTests(unittest.TestCase):
    def test_real_demo_dependency_validator_rejects_archive_receipt_drift(self) -> None:
        from compag_curation import quick_demo
        from compag_curation.public_io import PublicIOError, compact_json_sha256

        dependencies = _gpu_dependencies()
        archive = dict(dependencies["archive_provenance"])
        archive["sha256"] = "0" * 64
        drifted = {**dependencies, "archive_provenance": archive}
        drifted["sha256"] = compact_json_sha256(
            {key: value for key, value in drifted.items() if key != "sha256"}
        )
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "dependency_inventory.json"
            receipt.write_bytes(quick_demo.canonical_json_bytes(drifted))
            with self.assertRaisesRegex(
                PublicIOError,
                "real-demo GPU dependency evidence is invalid",
            ):
                quick_demo._validated_demo_gpu_dependencies(
                    receipt,
                    expected_cuda_visibility=dependencies["cuda_visibility"],
                )

    def test_binary_review_fixture_default_remains_v1(self) -> None:
        from compag_curation.quick_demo import _reviewed_copy
        from compag_curation.review.exchange import (
            REVIEW_COLUMNS,
            validate_review_table,
            write_review_export,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "request.csv"
            reviewed = root / "reviewed.csv"
            write_review_export(_proposals(), request)
            _reviewed_copy(request, reviewed)
            with reviewed.open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(tuple(next(csv.reader(handle))), REVIEW_COLUMNS)
            rows, result = validate_review_table(request, reviewed)
            self.assertEqual(len(rows), 16)
            self.assertEqual(result["review_contract"], "BINARY_V1")
            self.assertTrue(all(row.review_action is None and row.review_weight is None for row in rows))

    def test_project_phase_subprocess_is_strict_json_and_no_clobber(self) -> None:
        from compag_curation.domain.inference import ProcessResult
        from compag_curation.public_io import PublicIOError
        from compag_curation.quick_demo import _run_fresh_demo_project_operation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            staging = root / "staging"
            project.mkdir()
            staging.mkdir()
            (project / "config.toml").write_text("fixture\n", encoding="ascii")
            run_root = staging / "run"

            class MalformedRunner:
                def run(
                    self,
                    argv,
                    *,
                    cwd: Path,
                    env: dict[str, str],
                    acceptable_exit_codes: frozenset[int],
                    capture_output: bool,
                ) -> ProcessResult:
                    self.assertions = (acceptable_exit_codes, capture_output)
                    return ProcessResult(tuple(argv), 3, "not-json\n", "", env, cwd)

            malformed = MalformedRunner()
            with self.assertRaisesRegex(PublicIOError, "stdout is not JSON"):
                _run_fresh_demo_project_operation(
                    project,
                    run_root,
                    staging,
                    malformed,
                    operation="pause",
                )
            self.assertEqual(malformed.assertions, (frozenset({3}), True))
            self.assertFalse(run_root.exists())

            protected = staging / "protected-run"
            protected.mkdir()
            marker = protected / "keep.txt"
            marker.write_text("unchanged\n", encoding="ascii")
            unused_runner = mock.Mock()
            with self.assertRaisesRegex(PublicIOError, "already exists"):
                _run_fresh_demo_project_operation(
                    project,
                    protected,
                    staging,
                    unused_runner,
                    operation="pause",
                )
            unused_runner.run.assert_not_called()
            self.assertEqual(marker.read_text(encoding="ascii"), "unchanged\n")

    def test_00_real_demo_selects_canonical_profile_assets_and_review_v2(self) -> None:
        before = set(sys.modules)
        from compag_curation import quick_demo
        from compag_curation.domain.inference import ProcessResult
        from compag_curation.review.exchange import (
            CANONICAL_REVIEW_COLUMNS,
            validate_review_table,
            write_review_export,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset_root = root / "assets"
            asset_root.mkdir()
            identities = {}
            for asset_id, filename in CANONICAL_ASSETS:
                payload = f"fixture:{asset_id}\n".encode("ascii")
                (asset_root / filename).write_bytes(payload)
                identities[asset_id] = {
                    "status": "PASS",
                    "asset_id": asset_id,
                    "filename": filename,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }

            process_calls = []
            review_result = {}
            child_run_id = "11111111-1111-4111-8111-111111111111"
            dependencies = _gpu_dependencies()

            def verify_asset(_root: Path, asset_id: str) -> dict[str, object]:
                self.assertEqual(_root, asset_root)
                return dict(identities[asset_id])

            def run_process(
                argv,
                *,
                cwd: Path,
                env: dict[str, str],
                acceptable_exit_codes: frozenset[int],
                capture_output: bool,
            ) -> ProcessResult:
                process_calls.append((tuple(argv), cwd, dict(env), acceptable_exit_codes, capture_output))
                self.assertEqual(tuple(argv[:6]), (sys.executable, "-I", "-B", "-m", "compag_curation", "run"))
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
                self.assertEqual(env["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
                self.assertEqual(env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"], "0")
                self.assertEqual(env["PYTHONPATH"], "")
                self.assertNotIn("SECRET_PARENT_VALUE", env)
                config_path = Path(argv[argv.index("--config") + 1])
                self.assertIn(
                    'profile = "canonical-xgb-recall-gpu-v1"',
                    config_path.read_text(encoding="ascii"),
                )
                if "--output" in argv:
                    self.assertEqual(acceptable_exit_codes, frozenset({3}))
                    output = Path(argv[argv.index("--output") + 1])
                    (output / ".staging").mkdir(parents=True)
                    request = output / "stages/20_proposals_features/review_request.csv"
                    request.parent.mkdir(parents=True)
                    write_review_export(_proposals(), request, canonical_actions=True)
                    dependency_path = output / "stages/00_input_inventory/dependency_inventory.json"
                    dependency_path.parent.mkdir(parents=True)
                    dependency_path.write_bytes(quick_demo.canonical_json_bytes(dependencies))
                    payload = {
                        "schema": "compag-curation-public-run-pause/v1",
                        "status": "PAUSED_FOR_REVIEW",
                        "exit_code": 3,
                        "run_id": child_run_id,
                        "review_request": "stages/20_proposals_features/review_request.csv",
                        "review_request_sha256": hashlib.sha256(request.read_bytes()).hexdigest(),
                        "resume_command": "python -I -B -m compag_curation run --config CONFIG --resume RUN_OUTPUT --review-labels REVIEWED_CSV",
                    }
                    return ProcessResult(
                        tuple(argv),
                        3,
                        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                        "",
                        env,
                        cwd,
                    )
                self.assertEqual(acceptable_exit_codes, frozenset({0}))
                resume = Path(argv[argv.index("--resume") + 1])
                reviewed_labels = Path(argv[argv.index("--review-labels") + 1])
                request = resume / "stages/20_proposals_features/review_request.csv"
                rows, result = validate_review_table(request, reviewed_labels)
                review_result.update(result)
                self.assertTrue(all(row.review_action == "accept" for row in rows))
                self.assertTrue(all(row.review_weight == 1.0 for row in rows))
                payload = {
                    "schema": "compag-curation-public-run-completion/v1",
                    "status": "PASS",
                    "run_id": child_run_id,
                    "stages": 8,
                    "report": "stages/70_report/report.json",
                    "model_bundle": "stages/50_train_bundle/model_bundle",
                    "fresh_inference": "stages/60_evaluate/predictions.csv",
                }
                completion = resume / "FINAL/RUN_COMPLETE.json"
                completion.parent.mkdir(parents=True)
                completion.write_bytes(quick_demo.canonical_json_bytes(payload))
                return ProcessResult(
                    tuple(argv),
                    0,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                    "",
                    env,
                    cwd,
                )

            output = root / "real-demo"
            process_runner = mock.Mock()
            process_runner.run.side_effect = run_process
            with (
                mock.patch.dict(
                    os.environ,
                    {"CUDA_VISIBLE_DEVICES": "3", "SECRET_PARENT_VALUE": "private"},
                    clear=True,
                ),
                mock.patch.object(quick_demo, "verify_asset", side_effect=verify_asset) as verify,
                mock.patch.object(
                    quick_demo,
                    "_run_fresh_demo_inference",
                    return_value={
                        "schema": "fixture",
                        "status": "PASS",
                        "dependency_sha256": dependencies["sha256"],
                    },
                ) as fresh,
            ):
                result = quick_demo.run_real_demo(output, asset_root, runner=process_runner)

            self.assertEqual(result["status"], "PASS")
            self.assertEqual(
                fresh.call_args.kwargs,
                {
                    "device": "cuda",
                    "cuda_visibility": dependencies["cuda_visibility"],
                    "captured_dependencies": dependencies,
                },
            )
            self.assertEqual([call.args[1] for call in verify.call_args_list], [row[0] for row in CANONICAL_ASSETS])
            self.assertEqual(len(process_calls), 2)
            self.assertIn("--output", process_calls[0][0])
            self.assertIn("--resume", process_calls[1][0])
            self.assertTrue(all(call[4] for call in process_calls))
            self.assertEqual(review_result["review_contract"], "CANONICAL_ACTION_WEIGHTED_V2")
            self.assertEqual(review_result["action_counts"]["accept"], 16)
            self.assertEqual(float(review_result["effective_weight"]), 16.0)

            config = (output / "project/config.toml").read_text(encoding="ascii")
            self.assertIn('profile = "canonical-xgb-recall-gpu-v1"', config)
            self.assertIn('checkpoint = "assets/sam2.1_hiera_large.pt"', config)
            self.assertIn('embedding_weights = "assets/resnet50-11ad3fa6.pth"', config)
            self.assertEqual(
                sorted(path.name for path in (output / "project/assets").iterdir()),
                sorted(filename for _asset_id, filename in CANONICAL_ASSETS),
            )
            inventory = json.loads((output / "asset_inventory.json").read_text(encoding="ascii"))
            self.assertEqual(set(inventory["assets"]), {asset_id for asset_id, _filename in CANONICAL_ASSETS})
            result_record = json.loads((output / "REAL_DEMO_RESULT.json").read_text(encoding="ascii"))
            self.assertEqual(result_record["profile"], "canonical-xgb-recall-gpu-v1")
            self.assertEqual(result_record["review_contract"], "CANONICAL_ACTION_WEIGHTED_V2")
            self.assertEqual(result_record["generated_training_groups"], 8)
            self.assertEqual(
                [row["operation"] for row in result_record["gpu_project_processes"]],
                ["pause", "resume"],
            )
            environment = json.loads((output / "environment.json").read_text(encoding="ascii"))
            self.assertEqual(environment["schema"], "compag-curation-demo-environment/v2")
            self.assertEqual(environment["dependencies"], dependencies)
            self.assertEqual(len(environment["gpu_processes"]), 3)
            self.assertEqual(
                len(list((output / "project/data/images").glob("synthetic-group-*__train.ppm"))),
                8,
            )
            with (output / "reviewed_fixture.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
                handle.seek(0)
                self.assertEqual(tuple(next(csv.reader(handle))), CANONICAL_REVIEW_COLUMNS)
            self.assertTrue(all(row["review_action"] == "accept" and row["review_weight"] == "1.0" for row in rows))

        introduced = {name.split(".", 1)[0] for name in set(sys.modules) - before}
        self.assertFalse(introduced & HEAVY_MODULES)


if __name__ == "__main__":
    unittest.main()
