from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from tests.test_public_contract import _fixture


class SyntheticBoundary(RuntimeError):
    """Raised by a test double at the first scientific or process boundary."""


class TripwireRunner:
    """A process boundary that records any prohibited normal-window invocation."""

    def __init__(self) -> None:
        self.events: list[tuple[str, ...]] = []

    def run(
        self,
        argv: object,
        *,
        env: object,
        acceptable_exit_codes: object,
        cwd: object = None,
        capture_output: bool = False,
    ) -> object:
        event = tuple(str(value) for value in argv)  # type: ignore[arg-type]
        self.events.append(event)
        raise AssertionError("package runtime process boundary was invoked")


def _authorized_config(
    command: str,
    variant: str,
    root: Path,
    *,
    coverage_image_id: str = "synthetic-coverage-card-a",
):
    from compag_curation.config import (
        COMMANDS,
        DomainConfig,
        VARIANT_OUTPUT_ARTIFACT_FIELDS,
        validate_mapping,
    )
    from compag_curation.contracts import ExecutionPolicy

    mapping = _fixture((command, variant))
    if command == "evaluate" and variant.startswith("coverage-"):
        mapping["workflows"]["evaluate"]["image_id"] = coverage_image_id  # type: ignore[index]
    validated = validate_mapping(mapping, fixture=True, environment={})
    workflows = {name: validated.workflow(name) for name in COMMANDS}
    selected = workflows[command]
    outputs = VARIANT_OUTPUT_ARTIFACT_FIELDS[(command, variant)]
    for name, value in tuple(selected.items()):
        if not isinstance(value, dict) or "path" not in value:
            continue
        artifact = dict(value)
        artifact["path"] = str((root / command / variant / name).absolute())
        if name not in outputs:
            artifact["sha256"] = "0" * 64
        else:
            artifact.pop("sha256", None)
        selected[name] = artifact
    return DomainConfig(
        validated.schema,
        ExecutionPolicy(True, False, "fail", sys.executable),
        validated.notifications,
        workflows,
        validated.fixture_classification,
    )


class MappedSymbolContractTests(unittest.TestCase):
    """One authorized, symbol-specific fail-closed boundary contract per mapped API."""

    def setUp(self) -> None:
        self.runner = TripwireRunner()
        self.reports: list[str] = []
        self.temporary = tempfile.TemporaryDirectory(prefix="compag-symbol-contract-")
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.assertEqual(self.runner.events, [])
        self.temporary.cleanup()

    def _common_services(self):
        from compag_curation.contracts import HandlerServices

        return HandlerServices(
            report=self.reports.append,
            utc_now=lambda: "2026-08-27T00:00:00+00:00",
            runtime_identity=lambda: ("synthetic-user", "synthetic-host"),
        )

    def _config(self, command: str, variant: str):
        return _authorized_config(command, variant, self.root)

    def test_run_border_tiny_evaluation_contract(self) -> None:
        import compag_curation.domain.evaluation as module

        config = self._config("evaluate", "border-tiny")
        services = module.EvaluationServices(self._common_services())

        def boundary(received):
            self.assertIs(received, config)
            self.assertEqual(received.workflow("evaluate")["variant"], "border-tiny")
            raise SyntheticBoundary("border-tiny executor reached")

        with mock.patch.object(module, "validate_artifact_preconditions"), mock.patch.object(
            module, "execute_border_tiny_evaluation", side_effect=boundary
        ), self.assertRaisesRegex(SyntheticBoundary, "border-tiny executor reached"):
            module.run_border_tiny_evaluation(config, services)

    def test_run_coverage_evaluation_contract(self) -> None:
        import compag_curation.domain.evaluation as module

        for variant in (
            "coverage-basic",
            "coverage-points-overlay",
            "coverage-predictions-overlay",
            "coverage-reviewability",
        ):
            for image_id in ("synthetic-card-alpha.jpg", "synthetic-card-beta.jpg"):
                with self.subTest(variant=variant, image_id=image_id):
                    config = _authorized_config(
                        "evaluate",
                        variant,
                        self.root,
                        coverage_image_id=image_id,
                    )
                    services = module.EvaluationServices(
                        self._common_services(), coverage_runner=self.runner
                    )

                    def boundary(request, *, runner):
                        self.assertEqual(request.workflow, variant)
                        self.assertEqual(request.image_id, image_id)
                        self.assertEqual(request.script.sha256, "0" * 64)
                        self.assertEqual(request.tile_index.sha256, "0" * 64)
                        self.assertEqual(request.python_executable, Path(sys.executable).resolve())
                        self.assertIs(runner, self.runner)
                        self.assertEqual(request.images_root is not None, variant != "coverage-basic")
                        self.assertEqual(request.review_candidates is not None, variant == "coverage-reviewability")
                        raise SyntheticBoundary(f"coverage executor reached: {variant}")

                    with mock.patch.object(module, "validate_artifact_preconditions"), mock.patch.object(
                        module, "execute_coverage", side_effect=boundary
                    ), self.assertRaisesRegex(SyntheticBoundary, "coverage executor reached"):
                        module.run_coverage_evaluation(config, services)

    def test_run_meta_analysis_contract(self) -> None:
        import compag_curation.domain.evaluation as module

        config = self._config("evaluate", "meta-analysis")
        services = module.EvaluationServices(self._common_services())

        def boundary(received, injected):
            self.assertIs(received, config)
            self.assertIsNone(injected)
            self.assertTrue(received.execution.authorized)
            raise SyntheticBoundary("meta-analysis executor reached")

        with mock.patch.object(module, "validate_artifact_preconditions"), mock.patch.object(
            module, "execute_meta_analysis", side_effect=boundary
        ), self.assertRaisesRegex(SyntheticBoundary, "meta-analysis executor reached"):
            module.run_meta_analysis(config, services)

    def test_run_feature_extraction_workflow_contract(self) -> None:
        import compag_curation.domain.features as module

        config = self._config("extract-features", "coco-features")
        services = module.FeatureWorkflowServices(
            self._common_services(),
            lambda _checkpoint, _sha256, *, device: self.fail(
                f"feature backend factory invoked for {device}"
            ),
        )

        def boundary(sequence, injected):
            self.assertIs(injected, services)
            self.assertEqual(sequence.round.image_name, "IMG_1")
            self.assertEqual(sequence.round.mode, "xgb_recall")
            self.assertEqual(sequence.checkpoint_sha256, "0" * 64)
            self.assertEqual(sequence.device, "cpu")
            self.assertEqual(sequence.review_merge.mode, "xgb_recall")
            self.assertEqual(sequence.split.output_manifest, Path(config.workflow("extract-features")["split_manifest"]["path"]))
            raise SyntheticBoundary("feature preprocessing boundary reached")

        with mock.patch.object(module, "validate_artifact_preconditions"), mock.patch.object(
            module, "_run_feature_preprocessing_sequence", side_effect=boundary
        ), self.assertRaisesRegex(SyntheticBoundary, "feature preprocessing boundary reached"):
            module.run_feature_extraction_workflow(config, services)

    def test_run_single_image_inference_contract(self) -> None:
        import compag_curation.domain.inference as module

        config = self._config("infer", "sam2-xgb")
        services = module.InferenceWorkflowServices(
            self._common_services(), self.runner, {"OMP_NUM_THREADS": "1"}
        )

        def boundary(request, *, runner):
            self.assertIs(runner, self.runner)
            self.assertEqual(request.assets.sam2_checkpoint_sha256, "0" * 64)
            self.assertEqual(request.assets.xgb_model_sha256, "0" * 64)
            self.assertIsNone(request.assets.yolo_weights)
            self.assertEqual(request.options.python_executable, Path(sys.executable).resolve())
            self.assertFalse(request.options.allow_download)
            self.assertEqual(dict(request.environment), {"OMP_NUM_THREADS": "1"})
            raise SyntheticBoundary("inference executor reached")

        with mock.patch.object(module, "validate_artifact_preconditions"), mock.patch.object(
            module, "execute_inference", side_effect=boundary
        ), self.assertRaisesRegex(SyntheticBoundary, "inference executor reached"):
            module.run_single_image_inference(config, services)

    def test_prepare_yolo_dataset_contract(self) -> None:
        import compag_curation.domain.proposals as module
        import compag_curation.preprocessing.yolo_dataset as backend

        config = self._config("propose", "yolo-dataset-preparation")
        services = module.ProposalWorkflowServices(self._common_services())

        def boundary(request, common):
            values = config.workflow("propose")
            self.assertEqual(request.coco_annotations, Path(values["coco_annotations"]["path"]))
            self.assertEqual(request.tiles, Path(values["tiles"]["path"]))
            self.assertEqual(request.split_manifest, Path(values["split_manifest"]["path"]))
            self.assertEqual(request.output, Path(values["yolo_dataset_output"]["path"]))
            self.assertIs(common, services.common)
            raise SyntheticBoundary("YOLO dataset boundary reached")

        with mock.patch.object(module, "validate_artifact_preconditions"), mock.patch.object(
            backend, "prepare_yolo_dataset", side_effect=boundary
        ), self.assertRaisesRegex(SyntheticBoundary, "YOLO dataset boundary reached"):
            module.prepare_yolo_dataset(config, services)

    def test_build_plots_report_contract(self) -> None:
        import compag_curation.domain.reporting as module

        config = self._config("report", "plots")
        services = module.ReportingServices(self._common_services())

        def boundary(source, io, *, decision_threshold, active_learning_seed):
            self.assertEqual(source, Path(config.workflow("report")["source_tables"]["path"]))
            self.assertIsInstance(io, module.StdlibReportingTableIO)
            self.assertEqual(decision_threshold, 0.5)
            self.assertEqual(active_learning_seed, 0)
            raise SyntheticBoundary("precomputed report boundary reached")

        with mock.patch.object(module, "validate_artifact_preconditions"), mock.patch.object(
            module, "_validate_precomputed_testset_batch", side_effect=boundary
        ), self.assertRaisesRegex(SyntheticBoundary, "precomputed report boundary reached"):
            module.build_plots_report(config, services)

    def test_summarize_dataset_contract(self) -> None:
        import compag_curation.domain.reporting as module

        config = self._config("report", "dataset-summary")
        services = module.ReportingServices(self._common_services())

        def boundary(request):
            source = Path(config.workflow("report")["source_tables"]["path"])
            self.assertEqual(request.tiled_coco, source / "tiled_coco.json")
            self.assertEqual(request.original_coco, source / "original_coco.json")
            self.assertEqual(request.tiles_directory, source / "tiles")
            self.assertEqual(request.split_directory, source / "splits")
            self.assertEqual(request.tile_size, 512)
            raise SyntheticBoundary("dataset summary boundary reached")

        with mock.patch.object(module, "validate_artifact_preconditions"), mock.patch.object(
            module, "build_dataset_summary", side_effect=boundary
        ), self.assertRaisesRegex(SyntheticBoundary, "dataset summary boundary reached"):
            module.summarize_dataset(config, services)

    def test_run_al_review_contract(self) -> None:
        import compag_curation.domain.review as module

        config = self._config("review", "al-review-ui")
        services = module.ReviewWorkflowServices(
            self._common_services(),
            lambda _request: self.fail("active-learning launcher invoked"),
            lambda _request: self.fail("full-image launcher invoked"),
        )

        def boundary(request):
            values = config.workflow("review")
            self.assertEqual(request.current_image, "IMG_1")
            self.assertEqual(request.mode, "xgb_recall")
            self.assertEqual(request.image_root, Path(values["image_root"]["path"]))
            self.assertEqual(request.detections, Path(values["detections"]["path"]))
            self.assertEqual(request.images, Path(values["images"]["path"]))
            raise SyntheticBoundary("review path boundary reached")

        with mock.patch.object(module, "validate_artifact_preconditions"), mock.patch.object(
            module, "advance_review_paths", side_effect=boundary
        ), self.assertRaisesRegex(SyntheticBoundary, "review path boundary reached"):
            module.run_al_review(config, services)

    def _training_boundary(self, target, variant: str, executor_module: str, executor_name: str) -> None:
        import importlib
        import compag_curation.domain.training as module

        config = self._config("train", variant)
        services = module.TrainingWorkflowServices(
            self._common_services(),
            nb0001_dependencies=object() if variant == "coco-xgb" else None,
            nb6_dependencies=object() if variant == "xgb" else None,
            nb7_dependencies=object() if variant == "xgb-main" else None,
            thread_environment_setter=lambda _name, _value: self.fail("thread setter invoked"),
            nb6_fit_with_maybe_callbacks=(lambda *_args, **_kwargs: None) if variant == "xgb" else None,
            yolo_model_loader=lambda _path: self.fail("YOLO loader invoked"),
        )
        backend = importlib.import_module(executor_module)

        def boundary(batch, source_services, policy):
            self.assertEqual(batch.__class__.__name__, "SourceAlgorithmBatch")
            self.assertTrue(policy.authorized)
            self.assertEqual(source_services.cpu_count, 1)
            self.assertEqual(source_services.device_policy.mode, "cpu")
            self.assertGreaterEqual(len(source_services.approved_input_artifacts), 2)
            self.assertTrue(all(item.sha256 == "0" * 64 for item in source_services.approved_input_artifacts))
            raise SyntheticBoundary(f"{variant} source algorithm boundary reached")

        with mock.patch.object(module, "_require_xgb_dependency_surface"), mock.patch.object(
            module, "validate_artifact_preconditions"
        ), mock.patch.object(backend, executor_name, side_effect=boundary), self.assertRaisesRegex(
            SyntheticBoundary, f"{variant} source algorithm boundary reached"
        ):
            target(config, services)

    def test_run_coco_xgb_training_contract(self) -> None:
        from compag_curation.domain.training import run_coco_xgb_training

        self._training_boundary(
            run_coco_xgb_training,
            "coco-xgb",
            "compag_curation.training.xgb_nb0001",
            "execute_xgb_nb0001_source_algorithms",
        )

    def test_train_xgb_classic_contract(self) -> None:
        from compag_curation.domain.training import train_xgb_classic

        self._training_boundary(
            train_xgb_classic,
            "xgb",
            "compag_curation.training.xgb_nb6",
            "execute_xgb_nb6_source_algorithms",
        )

    def test_train_xgb_main_contract(self) -> None:
        from compag_curation.domain.training import train_xgb_main

        self._training_boundary(
            train_xgb_main,
            "xgb-main",
            "compag_curation.training.xgb_nb7",
            "execute_xgb_nb7_source_algorithms",
        )

    def test_train_yolo_local_contract(self) -> None:
        import compag_curation.domain.training as module

        config = self._config("train", "yolo")
        services = module.TrainingWorkflowServices(
            self._common_services(), yolo_model_loader=lambda _path: self.fail("YOLO loader invoked")
        )

        def boundary(request, common, *, model_loader):
            values = config.workflow("train")
            self.assertEqual(request.data_yaml, Path(values["data_yaml"]["path"]))
            self.assertEqual(request.dataset_root_sha256, "0" * 64)
            self.assertEqual(request.initial_weights_sha256, "0" * 64)
            self.assertEqual(request.output_root, Path(values["model_output"]["path"]))
            self.assertEqual(request.device, "cpu")
            self.assertIs(common, services.common)
            self.assertIs(model_loader, services.yolo_model_loader)
            raise SyntheticBoundary("YOLO training boundary reached")

        with mock.patch.object(module, "validate_artifact_preconditions"), mock.patch.object(
            module, "execute_yolo_training", side_effect=boundary
        ), self.assertRaisesRegex(SyntheticBoundary, "YOLO training boundary reached"):
            module.train_yolo_local(config, services)

    def _transfer_services(self):
        from compag_curation.domain.transfer import TransferWorkflowServices

        return TransferWorkflowServices(
            self._common_services(),
            self.runner,
            clock=lambda: datetime(2026, 8, 27, tzinfo=timezone.utc),
            environment={"LANG": "C.UTF-8"},
        )

    def test_transfer_csv_text_probe_contract(self) -> None:
        import compag_curation.domain.transfer as module

        config = self._config("transfer", "csv-and-text-probe")
        services = self._transfer_services()

        def boundary(request, *, runner, clock):
            self.assertEqual(request.variant, "csv-and-text-probe")
            self.assertEqual(request.include_patterns, ("*.csv", "*.txt"))
            self.assertEqual(request.probe_workspace, Path(config.workflow("transfer")["probe_workspace"]["path"]))
            self.assertEqual(dict(request.environment), {"LANG": "C.UTF-8"})
            self.assertIs(runner, self.runner)
            self.assertIs(clock, services.clock)
            raise SyntheticBoundary("CSV/text transfer boundary reached")

        with mock.patch.object(module, "validate_artifact_preconditions"), mock.patch.object(
            module, "execute_csv_text_probe_transfer", side_effect=boundary
        ), self.assertRaisesRegex(SyntheticBoundary, "CSV/text transfer boundary reached"):
            module.transfer_csv_text_probe(config, services)

    def test_transfer_result_tree_contract(self) -> None:
        import compag_curation.domain.transfer as module

        config = self._config("transfer", "result-tree")
        services = self._transfer_services()

        def boundary(request, *, runner):
            self.assertEqual(request.variant, "result-tree")
            self.assertEqual(request.include_patterns, ("*.*",))
            self.assertIsNone(request.probe_workspace)
            self.assertEqual(request.source, Path(config.workflow("transfer")["source"]["path"]))
            self.assertEqual(request.destination, Path(config.workflow("transfer")["destination"]["path"]))
            self.assertIs(runner, self.runner)
            raise SyntheticBoundary("result-tree transfer boundary reached")

        with mock.patch.object(module, "validate_artifact_preconditions"), mock.patch.object(
            module, "execute_result_tree_transfer", side_effect=boundary
        ), self.assertRaisesRegex(SyntheticBoundary, "result-tree transfer boundary reached"):
            module.transfer_result_tree(config, services)


if __name__ == "__main__":
    unittest.main()
