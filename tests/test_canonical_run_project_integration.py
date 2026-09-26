from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from compag_curation import public_pipeline
from compag_curation.canonical import service
from compag_curation.canonical.spec import CANONICAL_GPU_PROFILE, CANONICAL_PROFILE
from compag_curation.public_config import initialize_project
from compag_curation.public_io import PublicIOError, write_new_bytes, write_new_json
from compag_curation.quick_demo import _reviewed_copy
from compag_curation.review.exchange import CANONICAL_REVIEW_COLUMNS
from compag_curation.run_state import (
    CANONICAL_STAGE60_REQUIRED_FILES,
    STAGE_NAMES,
    STAGE_REQUIRED_FILES,
    RunBindings,
    begin_stage,
    claim_run,
    completed_stage,
    open_resume,
    publish_stage,
)


def _proposal_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group_index in range(6):
        group_id = f"group{group_index:02d}"
        image_sha256 = hashlib.sha256(f"image:{group_id}".encode("ascii")).hexdigest()
        for proposal_index in range(2):
            proposal_id = hashlib.sha256(
                f"proposal:{group_id}:{proposal_index}".encode("ascii")
            ).hexdigest()
            row = {column: "0" for column in service.CANONICAL_PROPOSAL_COLUMNS}
            row.update(
                {
                    "proposal_id": proposal_id,
                    "proposal_sha256": proposal_id,
                    "image_id": image_sha256,
                    "image_sha256": image_sha256,
                    "image_name": f"{group_id}__train.ppm",
                    "group_id": group_id,
                    "tile_name": f"{group_id}__train_y00000x00000.jpg",
                    "tile_sha256": hashlib.sha256(f"tile:{group_id}".encode("ascii")).hexdigest(),
                    "mask_sha256": hashlib.sha256(
                        f"mask:{group_id}:{proposal_index}".encode("ascii")
                    ).hexdigest(),
                }
            )
            rows.append(row)
    return rows


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    import io

    with io.StringIO(newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        write_new_bytes(path, handle.getvalue().encode("ascii"))


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _fixture_bindings(name: str) -> RunBindings:
    return RunBindings(
        *(hashlib.sha256(f"{name}:{role}".encode("ascii")).hexdigest() for role in range(4))
    )


def _write_required_outputs(root: Path, required: tuple[str, ...]) -> None:
    for relative in required:
        path = root / relative
        path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        if path.suffix == ".json":
            write_new_json(path, {"fixture": relative})
        else:
            write_new_bytes(path, f"fixture:{relative}\n".encode("ascii"))


def _publish_through_stage_fifty(state: object) -> None:
    for name in STAGE_NAMES[:6]:
        workspace = begin_stage(state, name)
        _write_required_outputs(workspace.root, STAGE_REQUIRED_FILES[name])
        publish_stage(state, workspace, STAGE_REQUIRED_FILES[name])


class CanonicalRunProjectIntegrationTests(unittest.TestCase):
    def test_pause_resume_completion_and_completed_resume_are_integrated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            initialize_project(project, profile=CANONICAL_GPU_PROFILE)
            config_path = project / "config.toml"
            run_root = root / "run"
            bindings = RunBindings(
                hashlib.sha256(config_path.read_bytes()).hexdigest(),
                hashlib.sha256(b"fixture-inputs").hexdigest(),
                hashlib.sha256(b"fixture-dependencies").hexdigest(),
                hashlib.sha256(b"fixture-assets").hexdigest(),
            )

            def preflight(config: object) -> tuple[RunBindings, dict[str, object]]:
                self.assertTrue(getattr(config, "is_canonical"))
                self.assertEqual(getattr(config, "config_sha256"), bindings.config_sha256)
                return bindings, {
                    "schema": "compag-canonical-run-project-test-preflight/v1",
                    "status": "PASS",
                    "bindings": asdict(bindings),
                    "inspection": {"status": "PASS", "fixture": True},
                    "annotations": None,
                    "assets": [],
                    "dependencies": {
                        "schema": "compag-canonical-run-project-test-dependencies/v1",
                        "status": "PASS",
                        "sha256": bindings.dependency_sha256,
                    },
                    "decode_validation": {"status": "PASS"},
                }

            def assert_preflight(
                config: object,
                expected: RunBindings,
                _captured_dependencies: object,
            ) -> None:
                observed, _payload = preflight(config)
                self.assertEqual(observed, expected)

            def disk_preflight(
                _config: object,
                _output: Path,
                *,
                completed_stages: tuple[str, ...] = (),
            ) -> dict[str, object]:
                return {
                    "schema": "compag-canonical-run-project-test-disk/v1",
                    "status": "PASS",
                    "completed_stages": list(completed_stages),
                }

            def prepare_stage(_config: object, output: Path) -> None:
                write_new_bytes(output / "tiles_index.csv", b"fixture-tiles\n")
                write_new_json(output / "prepare_summary.json", {"status": "PASS"})

            def proposal_stage(
                _config: object,
                prepared: Path,
                output: Path,
                *,
                write_overlays: bool,
            ) -> None:
                self.assertEqual(prepared, run_root / "stages/10_prepare")
                self.assertTrue(write_overlays)
                _write_csv(
                    output / "proposals.csv",
                    service.CANONICAL_PROPOSAL_COLUMNS,
                    _proposal_rows(),
                )
                write_new_bytes(output / "features.csv", b"fixture-features\n")
                write_new_json(output / "proposal_config.json", {"status": "PASS"})
                write_new_json(output / "proposal_summary.json", {"status": "PASS"})

            def train_stage(
                _config: object,
                proposals: Path,
                reviewed: Path,
                split: Path,
                output: Path,
                *,
                features_path: Path,
            ) -> None:
                self.assertEqual(proposals, run_root / "stages/20_proposals_features/proposals.csv")
                self.assertEqual(features_path, run_root / "stages/20_proposals_features/features.csv")
                self.assertEqual(reviewed, run_root / "stages/30_review_import/reviewed.csv")
                self.assertEqual(split, run_root / "stages/40_group_split/split_manifest.json")
                write_new_bytes(output / "test_predictions.csv", b"fixture-test-predictions\n")
                write_new_json(
                    output / "training_result.json",
                    {
                        "status": "PASS",
                        "bundle": {"bundle_sha256": "a" * 64},
                        "threshold_selected_on": "FIXED_CANONICAL_METHOD",
                        "test_metrics": {"rows": 12},
                    },
                )
                bundle = output / "model_bundle"
                bundle.mkdir(mode=0o755)
                write_new_json(bundle / "bundle.json", {"status": "PASS"})
                write_new_bytes(bundle / "classifier.ubj", b"fixture-classifier\n")

            def inference_stage(
                _config: object,
                bundle: Path,
                output: Path,
                _captured_dependencies: object,
            ) -> None:
                self.assertEqual(bundle, run_root / "stages/50_train_bundle/model_bundle")
                write_new_bytes(output / "predictions.csv", b"fixture-predictions\n")
                write_new_bytes(output / "raw_features.csv", b"fixture-raw-features\n")
                write_new_json(
                    output / "inference_result.json",
                    {"schema": "compag-canonical-run-project-test-inference/v1", "status": "PASS"},
                )
                write_new_json(
                    output / "evaluation_status.json",
                    {
                        "schema": "compag-canonical-run-project-test-evaluation/v1",
                        "status": "PASS",
                        "point_coverage": {"status": "NOT_PROVIDED"},
                    },
                )

            with (
                mock.patch.object(public_pipeline, "_preflight", side_effect=preflight) as preflight_mock,
                mock.patch.object(
                    public_pipeline,
                    "_assert_preflight",
                    side_effect=assert_preflight,
                ) as assert_preflight_mock,
                mock.patch.object(public_pipeline, "_disk_preflight", side_effect=disk_preflight) as disk_mock,
                mock.patch.object(service, "prepare_canonical_stage", side_effect=prepare_stage) as prepare_mock,
                mock.patch.object(service, "generate_canonical_stage", side_effect=proposal_stage) as proposal_mock,
                mock.patch.object(service, "train_canonical_stage", side_effect=train_stage) as train_mock,
                mock.patch.object(public_pipeline, "_stage_sixty", side_effect=inference_stage) as inference_mock,
                mock.patch.object(
                    public_pipeline,
                    "_validated_canonical_published_stage_sixty",
                    return_value={"status": "PASS"},
                ) as canonical_validation_mock,
            ):
                pause, pause_rc = public_pipeline.run_project(config_path, output=run_root)
                self.assertEqual(pause_rc, 3)
                self.assertEqual(pause["status"], "PAUSED_FOR_REVIEW")
                self.assertEqual(
                    {path.name for path in (run_root / "stages").iterdir()},
                    set(STAGE_NAMES[:3]),
                )
                self.assertFalse((run_root / "FINAL").exists())
                prepare_mock.assert_called_once()
                proposal_mock.assert_called_once()
                train_mock.assert_not_called()
                inference_mock.assert_not_called()

                request = run_root / "stages/20_proposals_features/review_request.csv"
                with request.open("r", encoding="utf-8", newline="") as handle:
                    self.assertEqual(tuple(next(csv.reader(handle))), CANONICAL_REVIEW_COLUMNS)
                reviewed = root / "reviewed.csv"
                _reviewed_copy(request, reviewed, canonical_actions=True)

                complete, complete_rc = public_pipeline.run_project(
                    config_path,
                    resume=run_root,
                    reviewed_labels=reviewed,
                )
                self.assertEqual(complete_rc, 0)
                self.assertEqual(complete["status"], "PASS")
                self.assertEqual(complete["stages"], len(STAGE_NAMES))
                self.assertEqual(complete["run_id"], pause["run_id"])
                self.assertEqual(
                    {path.name for path in (run_root / "stages").iterdir()},
                    set(STAGE_NAMES),
                )
                self.assertEqual(
                    {path.name for path in (run_root / "FINAL").iterdir()},
                    {"RUN_COMPLETE.json", "OUTPUT_MANIFEST.json"},
                )
                prepare_mock.assert_called_once()
                proposal_mock.assert_called_once()
                train_mock.assert_called_once()
                inference_mock.assert_called_once()
                self.assertEqual(
                    public_pipeline._json(run_root / "stages/60_evaluate/_SUCCESS.json")["required"],
                    list(CANONICAL_STAGE60_REQUIRED_FILES),
                )

                before = _tree_bytes(run_root)
                repeated, repeated_rc = public_pipeline.run_project(config_path, resume=run_root)
                self.assertEqual(repeated_rc, 0)
                self.assertEqual(repeated, complete)
                self.assertEqual(_tree_bytes(run_root), before)
                prepare_mock.assert_called_once()
                proposal_mock.assert_called_once()
                train_mock.assert_called_once()
                inference_mock.assert_called_once()
                self.assertEqual(canonical_validation_mock.call_count, 2)
                self.assertEqual(preflight_mock.call_count, 3)
                self.assertEqual(assert_preflight_mock.call_count, 22)
                self.assertEqual(disk_mock.call_count, 2)


class StageSixtyRequiredFilesTests(unittest.TestCase):
    def test_balanced_receipt_remains_resumable_and_cannot_pose_as_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bindings = _fixture_bindings("balanced")
            state = claim_run(root / "run", bindings)
            with state:
                _publish_through_stage_fifty(state)
                workspace = begin_stage(state, "60_evaluate")
                required = STAGE_REQUIRED_FILES["60_evaluate"]
                _write_required_outputs(workspace.root, required)
                receipt = publish_stage(state, workspace, required)
                self.assertEqual(receipt["required"], list(required))

            resumed = open_resume(root / "run", bindings)
            with resumed:
                receipt = completed_stage(resumed, "60_evaluate")
                self.assertIsNotNone(receipt)
                self.assertEqual(receipt["required"], list(STAGE_REQUIRED_FILES["60_evaluate"]))
                balanced = SimpleNamespace(is_canonical=False)
                self.assertEqual(
                    public_pipeline._validate_published_stage_sixty_for_profile(balanced, resumed),
                    receipt,
                )
                with self.assertRaisesRegex(PublicIOError, "configured profile"):
                    public_pipeline._validate_published_stage_sixty_for_profile(
                        SimpleNamespace(is_canonical=True),
                        resumed,
                    )

    def test_canonical_variant_is_exact_semantically_reparsed_and_raw_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bindings = _fixture_bindings("canonical")
            state = claim_run(root / "run", bindings)
            with state:
                _publish_through_stage_fifty(state)
                workspace = begin_stage(state, "60_evaluate")
                _write_required_outputs(workspace.root, CANONICAL_STAGE60_REQUIRED_FILES)
                runtime = workspace.root / "fresh_process_runtime"
                runtime.mkdir(mode=0o700)
                write_new_json(
                    runtime / "ENVIRONMENT_POLICY.json",
                    {"schema": "fixture-runtime-policy/v1", "status": "PASS"},
                )
                invalid_variants = (
                    tuple(reversed(CANONICAL_STAGE60_REQUIRED_FILES)),
                    (*CANONICAL_STAGE60_REQUIRED_FILES, CANONICAL_STAGE60_REQUIRED_FILES[-1]),
                    (*CANONICAL_STAGE60_REQUIRED_FILES, "extra.csv"),
                )
                for invalid in invalid_variants:
                    with self.subTest(required=invalid):
                        with self.assertRaisesRegex(PublicIOError, "contract mismatch"):
                            publish_stage(state, workspace, invalid)
                receipt = publish_stage(state, workspace, CANONICAL_STAGE60_REQUIRED_FILES)
                self.assertEqual(receipt["required"], list(CANONICAL_STAGE60_REQUIRED_FILES))

                config = SimpleNamespace(
                    is_canonical=True,
                    inference_images=root / "images",
                    profile=CANONICAL_PROFILE,
                    device="cpu",
                )
                validated_result = {"schema": "fixture-validated-inference/v1", "status": "PASS"}
                with (
                    mock.patch.object(public_pipeline, "_input_images_identity", return_value=[]) as inputs_mock,
                    mock.patch.object(
                        public_pipeline,
                        "_validated_canonical_inference_result",
                        return_value=validated_result,
                    ) as result_mock,
                    mock.patch.object(
                        public_pipeline,
                        "_validated_canonical_predictions",
                        return_value={"status": "PASS"},
                    ) as predictions_mock,
                    mock.patch.object(
                        public_pipeline,
                        "_validated_canonical_raw_feature_archive",
                        return_value={"status": "PASS"},
                    ) as raw_mock,
                ):
                    self.assertEqual(
                        public_pipeline._validate_published_stage_sixty_for_profile(config, state),
                        receipt,
                    )
                inputs_mock.assert_called_once_with(config.inference_images)
                result_mock.assert_called_once()
                predictions_mock.assert_called_once_with(
                    state.stages / "60_evaluate/predictions.csv",
                    validated_result,
                )
                raw_mock.assert_called_once_with(state.stages / "60_evaluate", validated_result)

                (state.stages / "60_evaluate/raw_features.csv").unlink()
                with self.assertRaises(PublicIOError):
                    completed_stage(state, "60_evaluate")


if __name__ == "__main__":
    unittest.main()
