from __future__ import annotations

import ast
import contextlib
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import re
import stat
import sys
import tempfile
import types
import unittest
import urllib.request
from pathlib import Path
from unittest import mock


HEAVY_MODULES = {
    "cv2",
    "cupy",
    "dash",
    "hydra",
    "imblearn",
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "plotly",
    "pycocotools",
    "scipy",
    "sklearn",
    "torch",
    "torchvision",
    "ultralytics",
    "xgboost",
}


def _ppm(seed: int) -> bytes:
    color = bytes(((seed * 17) % 251, (seed * 29) % 251, (seed * 43) % 251))
    return b"P6\n64 64\n255\n" + color * (64 * 64)


def _file_tree(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_review_table(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class RunnableFacadeContractTests(unittest.TestCase):
    def test_00_facade_import_is_inert_and_legacy_registry_is_unchanged(self) -> None:
        from compag_curation.config import COMMANDS, VARIANT_FIELDS

        before = set(sys.modules)
        for name in (
            "compag_curation.assets",
            "compag_curation.point_annotations",
            "compag_curation.public_config",
            "compag_curation.public_pipeline",
            "compag_curation.quick_demo",
            "compag_curation.run_state",
            "compag_curation.runtime_lock",
        ):
            importlib.import_module(name)
        introduced = {name.split(".", 1)[0] for name in set(sys.modules) - before}
        self.assertFalse(introduced & HEAVY_MODULES)
        self.assertEqual(
            COMMANDS,
            ("propose", "extract-features", "review", "train", "infer", "evaluate", "transfer", "report"),
        )
        self.assertEqual(len(VARIANT_FIELDS), 21)

    def test_canonical_stage_fidelity_map_is_closed_sanitized_and_resolvable(self) -> None:
        release_root = Path(__file__).resolve().parents[1]
        with (release_root / "manifests/CANONICAL_STAGE_FIDELITY_MAP.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(
                tuple(reader.fieldnames or ()),
                (
                    "stage_id",
                    "stage_name",
                    "authority_refs",
                    "canonical_profile",
                    "public_symbols",
                    "defaults",
                    "input_contract",
                    "output_contract",
                    "invariants",
                    "comparison_status",
                    "comparison_evidence",
                    "profile_boundaries",
                    "known_adaptations",
                    "parity_claim",
                ),
            )
            rows = list(reader)

        self.assertEqual([row["stage_id"] for row in rows], ["10", "20", "30", "40", "50", "60", "61", "70"])
        boundary = (
            "public-safe-balanced-v1=COMPATIBILITY_ONLY_SEMANTICALLY_DIFFERENT;"
            "historical-deployed-yolo-multiscale-hybrid=LATER_NONCANONICAL_EXCLUDED"
        )
        expected_statuses = {
            "10": "PASS_EXACT_METHOD_DEFAULTS_AND_INVARIANTS",
            "20": "PASS_METHOD_DEFAULTS_WITH_EXECUTION_MEMORY_ADAPTATION",
            "30": "PASS_NONSEMANTIC_PORTABLE_REVIEW_INTERFACE",
            "40": "PASS_NONSEMANTIC_NEW_USER_DATA_GROUP_PARTITION",
            "50": "PASS_EXACT_METHOD_DEFAULTS_NEW_USER_MODEL",
            "60": "PASS_METHOD_DEFAULTS_WITH_EXECUTION_AND_PROVENANCE_ADAPTATIONS",
            "61": "PASS_EXACT_HISTORICAL_POINT_COVERAGE_WITH_SAFETY_ADAPTER",
            "70": "PASS_NONSEMANTIC_COMPACT_REPORTING",
        }
        source_cache: dict[Path, set[str]] = {}
        for row in rows:
            serialized = json.dumps(row, sort_keys=True)
            self.assertIsNone(
                re.search(
                    r"(?:/home/|/root(?:/|\b)|/Users/|/mnt/[a-z]/Users/|fi"
                    + r"le://|[A-Za-z]:\\\\|\\\\\\\\)",
                    serialized,
                ),
                row["stage_id"],
            )
            self.assertEqual(row["canonical_profile"], "canonical-xgb-recall-cpu-v1")
            self.assertEqual(row["comparison_status"], expected_statuses[row["stage_id"]])
            self.assertRegex(
                row["comparison_evidence"],
                rf"\AC8-STAGE-{row['stage_id']}@sha256:[0-9a-f]{{64}}\Z",
            )
            self.assertEqual(row["profile_boundaries"], boundary)
            self.assertNotEqual(row["parity_claim"], "NOT_CLAIMED")
            self.assertTrue(row["input_contract"] and row["output_contract"] and row["invariants"])
            self.assertTrue(row["known_adaptations"])
            for authority in row["authority_refs"].split("|"):
                self.assertRegex(authority, r"\AAUTH-[A-Z0-9-]+@sha256:[0-9a-f]{64}\Z")
            for mapped in row["public_symbols"].split("|"):
                match = re.fullmatch(r"(src/compag_curation/[a-z0-9_/]+\.py)::([A-Za-z_][A-Za-z0-9_]*)", mapped)
                self.assertIsNotNone(match, mapped)
                assert match is not None
                source = release_root / match.group(1)
                self.assertTrue(source.is_file(), mapped)
                if source not in source_cache:
                    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
                    source_cache[source] = {
                        node.name
                        for node in tree.body
                        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    }
                self.assertIn(match.group(2), source_cache[source], mapped)

        adjudication_path = release_root / "manifests/CANONICAL_STAGE_FIDELITY_ADJUDICATION.json"
        adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
        self.assertEqual(adjudication["status"], "PASS")
        self.assertEqual(
            adjudication["canonical_stage_fidelity_audit"],
            "PASS_METHOD_EXACT_WITH_DISCLOSED_GPU_NUMERICAL_ADAPTATION",
        )
        self.assertEqual(adjudication["stage_count"], 8)
        self.assertEqual(adjudication["material_algorithm_default_or_output_contract_deviations"], 0)
        self.assertEqual(adjudication["paper_result_reproduction_status"], "NOT_CLAIMED")
        self.assertEqual(adjudication["profile"], "canonical-xgb-recall-cpu-v1")
        self.assertEqual(
            adjudication["profile_boundaries"]["science-gpu-v1.3.0"],
            "SEPARATE_SYNTHETIC_RUNTIME_ACCEPTANCE_NOT_HISTORICAL_SCIENTIFIC_EVIDENCE",
        )
        self.assertEqual(
            adjudication["map_sha256"],
            hashlib.sha256(
                (release_root / "manifests/CANONICAL_STAGE_FIDELITY_MAP.csv").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(set(adjudication["stages"]), set(expected_statuses))
        for row in rows:
            stage = adjudication["stages"][row["stage_id"]]
            self.assertEqual(stage["comparison_status"], row["comparison_status"])
            self.assertEqual(stage["parity_claim"], row["parity_claim"])
            self.assertEqual(
                stage["evidence_sha256"],
                row["comparison_evidence"].split("@sha256:", 1)[1],
            )

    def test_canonical_stage_fidelity_output_contracts_match_public_writers(self) -> None:
        from compag_curation.canonical.active_learning import (
            CANONICAL_AL_DECISION_LOG_COLUMNS,
            CANONICAL_AL_SHORTLIST_COLUMNS,
        )
        from compag_curation.canonical.service import (
            CANONICAL_INFERENCE_COLUMNS,
            CANONICAL_PROPOSAL_COLUMNS,
            CANONICAL_RAW_FEATURE_TABLE_COLUMNS,
            CANONICAL_TEST_PREDICTION_COLUMNS,
            CANONICAL_TILE_COLUMNS,
        )
        from compag_curation.model_bundle import BUNDLE_V2_FILES
        from compag_curation.review.exchange import CANONICAL_REVIEW_COLUMNS

        release_root = Path(__file__).resolve().parents[1]
        with (release_root / "manifests/CANONICAL_STAGE_FIDELITY_MAP.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = {row["stage_id"]: row for row in csv.DictReader(handle)}

        review_schema = ",".join(CANONICAL_REVIEW_COLUMNS)
        self.assertIn("tiles_index.csv:" + ",".join(CANONICAL_TILE_COLUMNS), rows["10"]["output_contract"])
        self.assertIn("proposals.csv:" + ",".join(CANONICAL_PROPOSAL_COLUMNS), rows["20"]["output_contract"])
        self.assertIn("features.csv:" + ",".join(CANONICAL_RAW_FEATURE_TABLE_COLUMNS), rows["20"]["output_contract"])
        self.assertIn("overlay_manifest.json:compag-curation-canonical-overlay-manifest/v1", rows["20"]["output_contract"])
        self.assertIn("overlays/*.png:one deterministic labeled overlay per input tile", rows["20"]["output_contract"])
        self.assertIn("shortest collision-free proposal-ID prefix", rows["20"]["invariants"])
        self.assertIn("maps every proposal exactly once in proposal order", rows["20"]["invariants"])
        self.assertIn("proposal summary binds the overlay manifest and proposal-ID digest", rows["20"]["invariants"])
        self.assertIn("review_request.csv:" + review_schema, rows["20"]["output_contract"])
        self.assertIn("reviewed_supplied.csv:" + review_schema, rows["30"]["output_contract"])
        self.assertIn("reviewed.csv:" + review_schema, rows["30"]["output_contract"])
        self.assertIn("shortlist.csv:" + ",".join(CANONICAL_AL_SHORTLIST_COLUMNS), rows["30"]["output_contract"])
        self.assertIn("decision_log.csv:" + ",".join(CANONICAL_AL_DECISION_LOG_COLUMNS), rows["30"]["output_contract"])
        self.assertIn(
            "src/compag_curation/canonical/active_learning_facade.py::begin_image_round",
            rows["30"]["public_symbols"],
        )
        self.assertIn(
            "src/compag_curation/canonical/active_learning_facade.py::resume_image_round",
            rows["30"]["public_symbols"],
        )
        self.assertIn(
            "retrain_request.json:compag-curation-canonical-active-learning-retrain-request/v2 "
            "(legacy CPU /v1 accepted)",
            rows["30"]["output_contract"],
        )
        self.assertIn(
            "round_manifest.json:compag-curation-canonical-active-learning-selection/v3",
            rows["30"]["output_contract"],
        )
        self.assertIn(
            "round_completion.json:compag-curation-canonical-active-learning-completion/v3",
            rows["30"]["output_contract"],
        )
        self.assertIn(
            "accumulated_al_features.csv:"
            "compag-curation-canonical-active-learning-accumulated-raw-features/v1",
            rows["30"]["output_contract"],
        )
        self.assertIn(
            "model_lineage:compag-curation-canonical-active-learning-model-lineage/v1",
            rows["30"]["output_contract"],
        )
        self.assertIn("exactly one direct-child input image", rows["30"]["input_contract"])
        self.assertIn("at-most Top-K 50", rows["30"]["invariants"])
        self.assertIn("fresh full XGBoost retrain", rows["30"]["invariants"])
        self.assertIn("genesis test groups and PCA remain frozen", rows["30"]["invariants"])
        self.assertIn("model label is project-rN", rows["30"]["invariants"])
        self.assertIn("no r92 lineage import or r93 output", rows["30"]["invariants"])
        self.assertIn("v1.8 operational adaptation", rows["30"]["invariants"])
        self.assertIn(
            "split_manifest.json:schema,status,profile,seed,method,train_groups,test_groups,target_test_rows,"
            "observed_test_rows,group_cv_folds,coverage,effective_rows,skipped_rows,reviewed_sha256",
            rows["40"]["output_contract"],
        )
        self.assertNotIn("validation_groups", rows["40"]["output_contract"])
        self.assertIn("test_predictions.csv:" + ",".join(CANONICAL_TEST_PREDICTION_COLUMNS), rows["50"]["output_contract"])
        self.assertIn("model_bundle/bundle.json:compag-curation-model-bundle/v2", rows["50"]["output_contract"])
        for member in BUNDLE_V2_FILES:
            self.assertIn(member, rows["50"]["output_contract"])
        self.assertIn("predictions.csv:" + ",".join(CANONICAL_INFERENCE_COLUMNS), rows["60"]["output_contract"])
        self.assertIn(
            "raw_features.csv:CANONICAL_RAW_FEATURE_TABLE_COLUMNS at scales 0.67|0.80|1.00|1.25",
            rows["60"]["output_contract"],
        )
        self.assertIn(
            "sam2_execution_points_per_batch=32",
            rows["60"]["output_contract"],
        )
        self.assertIn(
            "inference_only_gpu_sam2_execution_microbatch_32",
            rows["60"]["known_adaptations"],
        )
        self.assertIn("CANONICAL_EVALUATION.json:compag-curation-canonical-point-evaluation/v1", rows["61"]["output_contract"])
        self.assertIn("report.json:compag-curation-public-report/v1", rows["70"]["output_contract"])
        self.assertIn("report.md:ASCII Markdown", rows["70"]["output_contract"])

    def test_canonical_stage_fidelity_defaults_match_frozen_public_specs(self) -> None:
        from compag_curation.canonical.active_learning import CANONICAL_AL_MARGIN, CANONICAL_AL_TOP_K
        from compag_curation.canonical.spec import (
            CANONICAL_DECISION_THRESHOLD,
            CANONICAL_FEATURE_CROP_SCALES,
            CANONICAL_FEATURE_ORDER,
            CANONICAL_FEATURE_ORDER_SHA256,
            CANONICAL_FULL_IMAGE_NMS_IOU,
            CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
            CANONICAL_STAGE20_AMG_POINTS_PER_BATCH,
            CanonicalAMGSettings,
        )
        from compag_curation.canonical.training import CanonicalTrainingConfig
        from compag_curation.public_config import (
            RESNET50_WEIGHTS_SHA256,
            SAM2_LARGE_CHECKPOINT_SHA256,
            SAM2_LARGE_CONFIG_SHA256,
        )

        release_root = Path(__file__).resolve().parents[1]
        with (release_root / "manifests/CANONICAL_STAGE_FIDELITY_MAP.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            defaults = {
                row["stage_id"]: dict(field.split("=", 1) for field in row["defaults"].split(";"))
                for row in csv.DictReader(handle)
            }

        amg = CanonicalAMGSettings()
        training = CanonicalTrainingConfig()
        self.assertEqual(defaults["10"]["tile_size"], "512")
        self.assertEqual(defaults["10"]["tile_stride"], "512")
        self.assertEqual(defaults["10"]["tile_format"], "jpg")
        self.assertEqual(defaults["20"]["sam2_architecture"], amg.architecture)
        self.assertEqual(defaults["20"]["sam2_checkpoint_sha256"], SAM2_LARGE_CHECKPOINT_SHA256)
        self.assertEqual(defaults["20"]["sam2_config_sha256"], SAM2_LARGE_CONFIG_SHA256)
        self.assertEqual(defaults["20"]["resnet50_weights_sha256"], RESNET50_WEIGHTS_SHA256)
        self.assertEqual(int(defaults["20"]["points_per_side"]), amg.points_per_side)
        self.assertEqual(int(defaults["20"]["points_per_batch"]), amg.points_per_batch)
        self.assertEqual(
            int(defaults["20"]["sam2_gpu_execution_microbatch"]),
            CANONICAL_STAGE20_AMG_POINTS_PER_BATCH,
        )
        self.assertEqual(
            int(defaults["60"]["sam2_method_points_per_batch"]),
            amg.points_per_batch,
        )
        self.assertEqual(
            int(defaults["60"]["sam2_gpu_execution_microbatch"]),
            CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
        )
        self.assertEqual(int(defaults["60"]["archived_sam_score_decimals"]), 6)
        self.assertEqual(float(defaults["20"]["pred_iou_thresh"]), amg.pred_iou_thresh)
        self.assertEqual(float(defaults["20"]["stability_score_thresh"]), amg.stability_score_thresh)
        self.assertEqual(float(defaults["20"]["merge_full_mask_iou"]), amg.merge_iou_threshold)
        self.assertEqual(int(defaults["20"]["max_masks_per_tile"]), amg.max_masks_per_tile)
        self.assertEqual(defaults["20"]["write_overlays"], "true")
        self.assertEqual(
            tuple(float(value) for value in defaults["20"]["feature_crop_scales"].split("|")),
            CANONICAL_FEATURE_CROP_SCALES,
        )
        self.assertEqual(float(defaults["30"]["active_learning_margin"]), CANONICAL_AL_MARGIN)
        self.assertEqual(int(defaults["30"]["active_learning_top_k"]), CANONICAL_AL_TOP_K)
        self.assertEqual(float(defaults["40"]["test_fraction"]), training.test_fraction)
        self.assertEqual(int(defaults["40"]["group_cv_folds"]), training.group_folds)
        self.assertEqual(int(defaults["50"]["feature_count"]), len(CANONICAL_FEATURE_ORDER))
        self.assertEqual(defaults["50"]["feature_order_sha256"], CANONICAL_FEATURE_ORDER_SHA256)
        self.assertEqual(int(defaults["50"]["random_search_iterations"]), training.search_iterations)
        self.assertEqual(int(defaults["50"]["early_stopping_rounds"]), training.early_stopping_rounds)
        self.assertEqual(float(defaults["50"]["decision_threshold"]), CANONICAL_DECISION_THRESHOLD)
        self.assertEqual(float(defaults["60"]["prediction_feature_scale"]), 1.0)
        self.assertEqual(
            tuple(float(value) for value in defaults["60"]["archive_feature_crop_scales"].split("|")),
            CANONICAL_FEATURE_CROP_SCALES,
        )
        self.assertEqual(float(defaults["60"]["threshold"]), CANONICAL_DECISION_THRESHOLD)
        self.assertEqual(float(defaults["60"]["full_image_nms_iou"]), CANONICAL_FULL_IMAGE_NMS_IOU)
        self.assertEqual(float(defaults["61"]["active_learning_margin"]), CANONICAL_AL_MARGIN)
        self.assertEqual(int(defaults["61"]["active_learning_top_k"]), CANONICAL_AL_TOP_K)

    def test_cli_help_and_execution_argument_boundaries(self) -> None:
        from compag_curation.cli import main

        help_routes = (
            ["doctor", "--help"],
            ["init-project", "--help"],
            ["inspect-data", "--help"],
            ["validate", "--help"],
            ["plan", "--help"],
            ["assets", "--help"],
            ["assets", "fetch", "--help"],
            ["assets", "verify", "--help"],
            ["demo", "--help"],
            ["run", "--help"],
            ["active-learning", "--help"],
            ["active-learning", "initial-export", "--help"],
            ["active-learning", "initial-resume", "--help"],
            ["active-learning", "begin-round", "--help"],
            ["active-learning", "resume-round", "--help"],
            ["active-learning", "begin-image-round", "--help"],
            ["active-learning", "resume-image-round", "--help"],
            ["active-learning", "import-r92-transfer-baseline", "--help"],
            ["active-learning", "infer-r92-image", "--help"],
            ["active-learning", "begin-transfer-image-round", "--help"],
            ["active-learning", "resume-transfer-image-round", "--help"],
            ["convert-annotations", "--help"],
        )
        for argv in help_routes:
            with self.subTest(argv=argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main(argv)
                self.assertEqual(caught.exception.code, 0)

        invalid_routes = (
            ["infer", "--execute", "--images", "images", "--bundle", "bundle"],
            ["infer", "--plan", "--images", "images"],
            ["infer", "--execute", "--config", "config.json", "--images", "images", "--bundle", "bundle", "--output", "output"],
            ["run", "--config", "config.toml"],
            ["run", "--config", "config.toml", "--output", "output", "--review-labels", "reviewed.csv"],
        )
        for argv in invalid_routes:
            with self.subTest(argv=argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main(argv)
                self.assertEqual(caught.exception.code, 2)

    def test_cli_active_learning_dispatch_is_lazy_and_exact(self) -> None:
        from compag_curation.canonical import active_learning_facade as active
        from compag_curation.cli import main

        cases = (
            (
                "initial_export",
                [
                    "active-learning", "initial-export",
                    "--stage20-root", "stage20",
                    "--output", "initial-export",
                ],
                (Path("stage20"), Path("initial-export")),
                3,
            ),
            (
                "initial_resume",
                [
                    "active-learning", "initial-resume",
                    "--initial-root", "initial-export",
                    "--reviewed", "reviewed.csv",
                    "--output", "initial-completion",
                ],
                (Path("initial-export"), Path("reviewed.csv"), Path("initial-completion")),
                0,
            ),
            (
                "begin_round",
                [
                    "active-learning", "begin-round",
                    "--stage60-root", "pool-inference",
                    "--bundle", "bundle",
                    "--split", "split.json",
                    "--decision-log", "decisions.csv",
                    "--round-number", "1",
                    "--ancestry-root", "initial-completion",
                    "--output", "round-selection",
                ],
                (
                    Path("pool-inference"), Path("bundle"), Path("split.json"),
                    Path("decisions.csv"), 1, Path("initial-completion"),
                    Path("round-selection"),
                ),
                3,
            ),
            (
                "resume_round",
                [
                    "active-learning", "resume-round",
                    "--selection-root", "round-selection",
                    "--stage60-root", "pool-inference",
                    "--stage20-features", "stage20/features.csv",
                    "--source-bundle", "bundle",
                    "--split", "split.json",
                    "--decision-log", "decisions.csv",
                    "--reviewed", "round-reviewed.csv",
                    "--round-number", "1",
                    "--ancestry-root", "initial-completion",
                    "--output", "round-completion",
                ],
                (
                    Path("round-selection"), Path("pool-inference"),
                    Path("stage20/features.csv"), Path("bundle"),
                    Path("split.json"), Path("decisions.csv"),
                    Path("round-reviewed.csv"), 1, Path("initial-completion"),
                    Path("round-completion"),
                ),
                0,
            ),
            (
                "begin_image_round",
                [
                    "active-learning", "begin-image-round",
                    "--stage60-root", "image-inference",
                    "--bundle", "bundle",
                    "--stage20-features", "stage20/features.csv",
                    "--genesis-split", "split.json",
                    "--decision-log", "decisions.csv",
                    "--round-number", "1",
                    "--ancestry-root", "initial-completion",
                    "--output", "image-selection",
                ],
                (
                    Path("image-inference"), Path("bundle"),
                    Path("stage20/features.csv"), Path("split.json"),
                    Path("decisions.csv"), 1, Path("initial-completion"),
                    Path("image-selection"),
                ),
                3,
            ),
            (
                "resume_image_round",
                [
                    "active-learning", "resume-image-round",
                    "--selection-root", "image-selection",
                    "--stage60-root", "image-inference",
                    "--stage20-features", "stage20/features.csv",
                    "--source-bundle", "bundle",
                    "--genesis-split", "split.json",
                    "--decision-log", "decisions.csv",
                    "--reviewed", "image-reviewed.csv",
                    "--round-number", "1",
                    "--ancestry-root", "initial-completion",
                    "--output", "image-completion",
                ],
                (
                    Path("image-selection"), Path("image-inference"),
                    Path("stage20/features.csv"), Path("bundle"),
                    Path("split.json"), Path("decisions.csv"),
                    Path("image-reviewed.csv"), 1,
                    Path("initial-completion"), Path("image-completion"),
                ),
                0,
            ),
        )
        for name, argv, expected, return_code in cases:
            with self.subTest(name=name):
                payload = {
                    "schema": "compag-curation-active-learning-operation-result/v1",
                    "status": "PAUSED_FOR_REVIEW" if return_code == 3 else "PASS",
                }
                capture = io.StringIO()
                with (
                    mock.patch.object(active, name, return_value=(payload, return_code)) as call,
                    contextlib.redirect_stdout(capture),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(main(argv), return_code)
                call.assert_called_once_with(*expected)
                self.assertEqual(json.loads(capture.getvalue()), payload)

    def test_cli_transfer_active_learning_dispatch_is_exact(self) -> None:
        from types import SimpleNamespace

        from compag_curation.canonical import active_learning_facade as active
        from compag_curation.canonical import transfer_baseline, transfer_inference
        from compag_curation.cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_root = root / "baseline"
            baseline_root.mkdir()
            (baseline_root / "manifest.json").write_text(
                "{}\n", encoding="ascii"
            )
            baseline = SimpleNamespace(
                root=baseline_root,
                source_sha256="1" * 64,
                source_size_bytes=582_968_556,
                source_groups_sha256="2" * 64,
                source_groups=frozenset({"g0", "g1"}),
                row_count=8,
                proposal_count=2,
                train_groups=frozenset({"g0"}),
                test_groups=frozenset({"g1"}),
            )
            capture = io.StringIO()
            with (
                mock.patch.object(
                    transfer_baseline,
                    "import_r92_transfer_baseline",
                    return_value=baseline,
                ) as call,
                contextlib.redirect_stdout(capture),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            "active-learning",
                            "import-r92-transfer-baseline",
                            "--source-features",
                            "private.csv",
                            "--output",
                            str(baseline_root),
                        ]
                    ),
                    0,
                )
            call.assert_called_once_with(Path("private.csv"), baseline_root)
            self.assertEqual(json.loads(capture.getvalue())["status"], "PASS")

            transfer_payload = {"status": "PASS", "schema": "transfer-inference"}
            capture = io.StringIO()
            with (
                mock.patch.object(
                    transfer_inference,
                    "infer_r92_transfer_image",
                    return_value=(transfer_payload, 0),
                ) as call,
                contextlib.redirect_stdout(capture),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            "active-learning",
                            "infer-r92-image",
                            "--config",
                            "config.toml",
                            "--images",
                            "one-image",
                            "--output",
                            "inference",
                        ]
                    ),
                    0,
                )
            call.assert_called_once_with(
                Path("config.toml"), Path("one-image"), Path("inference")
            )
            self.assertEqual(json.loads(capture.getvalue()), transfer_payload)

            pause_payload = {"status": "PAUSED_FOR_REVIEW", "schema": "transfer-selection"}
            capture = io.StringIO()
            with (
                mock.patch.object(
                    active,
                    "begin_transfer_image_round",
                    return_value=(pause_payload, 3),
                ) as call,
                contextlib.redirect_stdout(capture),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            "active-learning",
                            "begin-transfer-image-round",
                            "--stage60-root",
                            "inference",
                            "--transfer-baseline",
                            "baseline",
                            "--round-number",
                            "2",
                            "--source-bundle",
                            "project-r1",
                            "--decision-log",
                            "decisions.csv",
                            "--prior-completion",
                            "prior",
                            "--output",
                            "selection",
                        ]
                    ),
                    3,
                )
            call.assert_called_once_with(
                Path("inference"),
                Path("baseline"),
                2,
                Path("selection"),
                source_bundle=Path("project-r1"),
                decision_log=Path("decisions.csv"),
                prior_completion=Path("prior"),
            )
            self.assertEqual(json.loads(capture.getvalue()), pause_payload)

            completion_payload = {"status": "PASS", "schema": "transfer-completion"}
            capture = io.StringIO()
            with (
                mock.patch.object(
                    active,
                    "resume_transfer_image_round",
                    return_value=(completion_payload, 0),
                ) as call,
                contextlib.redirect_stdout(capture),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            "active-learning",
                            "resume-transfer-image-round",
                            "--selection-root",
                            "selection",
                            "--stage60-root",
                            "inference",
                            "--transfer-baseline",
                            "baseline",
                            "--config",
                            "config.toml",
                            "--reviewed",
                            "reviewed.csv",
                            "--round-number",
                            "1",
                            "--output",
                            "completion",
                        ]
                    ),
                    0,
                )
            call.assert_called_once_with(
                Path("selection"),
                Path("inference"),
                Path("baseline"),
                Path("reviewed.csv"),
                1,
                Path("completion"),
                config=Path("config.toml"),
                source_bundle=None,
                decision_log=None,
                prior_completion=None,
            )
            self.assertEqual(json.loads(capture.getvalue()), completion_payload)

    def test_cli_init_inspect_plan_and_no_clobber(self) -> None:
        from compag_curation import public_config
        from compag_curation.cli import main
        from compag_curation.public_io import PublicIOError

        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "user project"
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["init-project", "--output", str(project)]), 0)
            created = json.loads(capture.getvalue())
            self.assertEqual(created["status"], "PASS")
            self.assertTrue((project / "config.toml").is_file())
            self.assertTrue((project / "data/images").is_dir())
            self.assertTrue((project / "data/inference_images").is_dir())

            for command, expected_status in (("inspect-data", "PASS"), ("plan", "VALID")):
                capture = io.StringIO()
                with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main([command, "--config", str(project / "config.toml")]), 0)
                self.assertEqual(json.loads(capture.getvalue())["status"], expected_status)

            before = _file_tree(project)
            error = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
                self.assertEqual(main(["init-project", "--output", str(project)]), 1)
            self.assertEqual(json.loads(error.getvalue())["status"], "ERROR")
            self.assertEqual(_file_tree(project), before)

            interrupted = Path(directory) / "interrupted-project"
            real_publish = public_config.publish_directory_noreplace

            def fail_publish(source: Path, destination: Path) -> None:
                if destination == interrupted:
                    raise PublicIOError("injected project initialization interruption")
                real_publish(source, destination)

            with mock.patch.object(public_config, "publish_directory_noreplace", side_effect=fail_publish):
                with self.assertRaisesRegex(PublicIOError, "initialization interruption"):
                    public_config.initialize_project(interrupted)
            self.assertFalse(interrupted.exists())
            self.assertEqual(len(list(Path(directory).glob(".interrupted-project.project-init.*"))), 1)
            self.assertEqual(public_config.initialize_project(interrupted)["status"], "PASS")

            recovered = Path(directory) / "recovered-project"
            from compag_curation import public_io

            real_rename = public_io.rename_noreplace

            def project_publish_then_raise(source: Path, destination: Path) -> None:
                real_rename(source, destination)
                if destination == recovered:
                    raise OSError("injected project destination fsync failure")

            with mock.patch.object(public_io, "rename_noreplace", side_effect=project_publish_then_raise):
                self.assertEqual(public_config.initialize_project(recovered)["status"], "PASS")

    def test_cli_pause_and_bundle_inference_dispatch_are_lazy(self) -> None:
        from compag_curation import public_pipeline
        from compag_curation.cli import main

        pause = {"schema": "fixture-pause/v1", "status": "PAUSED_FOR_REVIEW", "exit_code": 3}
        with mock.patch.object(public_pipeline, "run_project", return_value=(pause, 3)) as target:
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(io.StringIO()):
                rc = main(["run", "--config", "project.toml", "--output", "new-output"])
            self.assertEqual(rc, 3)
            self.assertEqual(json.loads(capture.getvalue()), pause)
            target.assert_called_once_with(
                Path("project.toml"),
                output=Path("new-output"),
                resume=None,
                reviewed_labels=None,
                dry_run=False,
                invocation={
                    "source": "CLI",
                    "command": "run",
                    "mode": "NEW",
                    "exact_argv_sha256": mock.ANY,
                },
            )

        error = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
            with self.assertRaises(SystemExit) as rejected:
                main([
                    "infer", "--execute", "--images", "images", "--bundle", "bundle",
                    "--output", "predictions", "--device", "cpu",
                ])
        self.assertEqual(rejected.exception.code, 2)
        self.assertIn("invalid choice", error.getvalue())

        result = {"schema": "fixture-inference/v1", "status": "PASS"}
        with mock.patch.object(public_pipeline, "infer_bundle", return_value=(result, 0)) as target:
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(io.StringIO()):
                rc = main([
                    "infer", "--execute", "--images", "images", "--bundle", "bundle",
                    "--output", "predictions", "--device", "cuda",
                ])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(capture.getvalue()), result)
            target.assert_called_once_with(
                Path("images"),
                Path("bundle"),
                Path("predictions"),
                device="cuda",
                invocation={"source": "CLI", "command": "infer", "exact_argv_sha256": mock.ANY},
            )

    def test_image_inspection_and_missing_asset_failure_are_nonexecuting(self) -> None:
        from compag_curation import public_pipeline
        from compag_curation.cli import main

        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            public_pipeline.init_project(project)
            for index in range(6):
                (project / "data/images" / f"group-{index + 1:02d}__card.ppm").write_bytes(_ppm(index + 1))
            heldout = project / "data/inference_images/heldout__card.ppm"
            heldout.write_bytes(_ppm(20))

            inspected, rc = public_pipeline.inspect_project(project / "config.toml")
            self.assertEqual(rc, 0)
            self.assertEqual((inspected["image_count"], inspected["group_count"]), (6, 6))
            self.assertEqual(inspected["inference_image_count"], 1)

            error = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
                self.assertEqual(main(["validate", "--config", str(project / "config.toml")]), 1)
            failure = json.loads(error.getvalue())
            self.assertEqual(failure["status"], "ERROR")
            self.assertIn("SAM2 checkpoint", failure["error"])

            duplicate = project / "data/inference_images/duplicate__card.ppm"
            duplicate.write_bytes(_ppm(1))
            with self.assertRaisesRegex(RuntimeError, "overlap by content hash"):
                public_pipeline.inspect_project(project / "config.toml")

            too_few_groups = Path(directory) / "too-few-groups"
            public_pipeline.init_project(too_few_groups)
            for index in range(5):
                (too_few_groups / "data/images" / f"group-{index + 1:02d}__card.ppm").write_bytes(_ppm(index + 30))
            (too_few_groups / "data/inference_images/heldout__card.ppm").write_bytes(_ppm(40))
            with self.assertRaisesRegex(RuntimeError, "at least six independent training groups"):
                public_pipeline.validate_project(too_few_groups / "config.toml")

            no_inference = Path(directory) / "no-inference"
            public_pipeline.init_project(no_inference)
            for index in range(6):
                (no_inference / "data/images" / f"group-{index + 1:02d}__card.ppm").write_bytes(_ppm(index + 50))
            with self.assertRaisesRegex(RuntimeError, "inference image"):
                public_pipeline.validate_project(no_inference / "config.toml")

            comma_name = Path(directory) / "comma-name"
            public_pipeline.init_project(comma_name)
            for index in range(4):
                (comma_name / "data/images" / f"group-{index + 1:02d}__card.ppm").write_bytes(
                    _ppm(index + 60)
                )
            (comma_name / "data/inference_images/heldout__view,1.ppm").write_bytes(_ppm(70))
            with self.assertRaisesRegex(RuntimeError, "reserved separator"):
                public_pipeline.inspect_project(comma_name / "config.toml")

    def test_annotation_conversion_is_no_clobber_and_preserves_input(self) -> None:
        from compag_curation.cli import main

        xml = (
            '<annotations><image id="0" name="heldout.ppm" width="64" height="64">'
            '<points label="cj" points="10.5,20.5"/>'
            '<point label="cj" points="30,40"/>'
            "</image></annotations>\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "points.xml"
            output = root / "points.json"
            source.write_text(xml, encoding="ascii")
            before_info = source.stat()
            before_bytes = source.read_bytes()

            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["convert-annotations", "--input", str(source), "--output", str(output)]), 0)
            self.assertEqual(json.loads(capture.getvalue())["rows"], 2)
            converted = json.loads(output.read_text(encoding="ascii"))
            self.assertEqual(converted["schema"], "compag-curation-canonical-point-annotations/v1")
            self.assertEqual(len(converted["rows"]), 2)
            from compag_curation.point_annotations import inspect_cvat_points, inspect_point_annotations

            xml_rows, _xml_summary = inspect_cvat_points(source)
            json_rows, json_summary = inspect_point_annotations(output)
            self.assertEqual(json_rows, xml_rows)
            self.assertEqual(json_summary["point_rows"], len(xml_rows))

            canonical_payload = output.read_bytes()
            output.write_bytes(b'{"schema":"duplicate",' + canonical_payload[1:])
            with self.assertRaisesRegex(RuntimeError, "canonical point annotation JSON is invalid"):
                inspect_point_annotations(output)
            output.write_bytes(b'{"nonfinite":NaN,' + canonical_payload[1:])
            with self.assertRaisesRegex(RuntimeError, "canonical point annotation JSON is invalid"):
                inspect_point_annotations(output)
            output.write_bytes(canonical_payload)

            error = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
                self.assertEqual(main(["convert-annotations", "--input", str(source), "--output", str(output)]), 1)
            self.assertEqual(json.loads(error.getvalue())["status"], "ERROR")
            after_info = source.stat()
            self.assertEqual(source.read_bytes(), before_bytes)
            for field in ("st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"):
                self.assertEqual(getattr(after_info, field), getattr(before_info, field))

            invalid = root / "invalid.xml"
            invalid.write_text(xml.replace("30,40", "70,40"), encoding="ascii")
            invalid_output = root / "invalid.json"
            error = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
                self.assertEqual(main(["convert-annotations", "--input", str(invalid), "--output", str(invalid_output)]), 1)
            self.assertIn("outside the declared image", json.loads(error.getvalue())["error"])
            self.assertFalse(invalid_output.exists())

            comma_image = root / "comma-image.xml"
            comma_image.write_text(
                xml.replace("heldout.ppm", "heldout,view.ppm"), encoding="ascii"
            )
            comma_image_output = root / "comma-image.json"
            error = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
                self.assertEqual(
                    main(
                        [
                            "convert-annotations",
                            "--input",
                            str(comma_image),
                            "--output",
                            str(comma_image_output),
                        ]
                    ),
                    1,
                )
            self.assertIn("canonical annotation name alphabet", json.loads(error.getvalue())["error"])
            self.assertFalse(comma_image_output.exists())

            comma_source = root / "points,1.xml"
            comma_source.write_text(xml, encoding="ascii")
            comma_source_output = root / "comma-source.json"
            error = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
                self.assertEqual(
                    main(
                        [
                            "convert-annotations",
                            "--input",
                            str(comma_source),
                            "--output",
                            str(comma_source_output),
                        ]
                    ),
                    1,
                )
            self.assertIn("canonical annotation name alphabet", json.loads(error.getvalue())["error"])
            self.assertFalse(comma_source_output.exists())

            wrong_suffix = root / "points.txt"
            error = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
                self.assertEqual(main(["convert-annotations", "--input", str(source), "--output", str(wrong_suffix)]), 1)
            self.assertIn(".json suffix", json.loads(error.getvalue())["error"])
            self.assertFalse(wrong_suffix.exists())

            recovered_output = root / "recovered.json"
            from compag_curation import public_io

            real_rename = public_io.rename_noreplace

            def annotation_publish_then_raise(staging: Path, destination: Path) -> None:
                real_rename(staging, destination)
                if destination == recovered_output:
                    raise OSError("injected annotation destination fsync failure")

            with mock.patch.object(public_io, "rename_noreplace", side_effect=annotation_publish_then_raise):
                capture = io.StringIO()
                with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["convert-annotations", "--input", str(source), "--output", str(recovered_output)]), 0)
            self.assertEqual(json.loads(capture.getvalue())["status"], "PASS")

    def test_review_exchange_round_trip_and_identity_rejections(self) -> None:
        from compag_curation.public_io import PublicIOError
        from compag_curation.review.exchange import (
            REVIEW_COLUMNS,
            validate_review_table,
            write_review_export,
            write_review_import,
        )

        proposals = []
        for index in range(2):
            proposal = hashlib.sha256(f"proposal-{index}".encode("ascii")).hexdigest()
            proposals.append({
                "proposal_id": proposal,
                "proposal_sha256": proposal,
                "image_id": hashlib.sha256(f"image-id-{index}".encode("ascii")).hexdigest(),
                "image_sha256": hashlib.sha256(f"image-{index}".encode("ascii")).hexdigest(),
                "group_id": f"group-{index}",
            })

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exported = root / "request.csv"
            write_review_export(proposals, exported)
            before = exported.read_bytes()
            with exported.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for index, row in enumerate(rows):
                row["label"] = str(index)
                row["review_status"] = "reviewed"
            reviewed = root / "reviewed.csv"
            _write_review_table(reviewed, rows, REVIEW_COLUMNS)
            normalized_rows, summary = validate_review_table(exported, reviewed)
            self.assertEqual((summary["positive_rows"], summary["negative_rows"]), (1, 1))
            imported = root / "imported.csv"
            self.assertEqual(write_review_import(normalized_rows, imported)["rows"], 2)
            self.assertEqual(exported.read_bytes(), before)

            duplicate_rows = [dict(rows[0]), dict(rows[0]), dict(rows[1])]
            missing_rows = [dict(rows[0])]
            unknown_rows = [dict(row) for row in rows]
            unknown_rows[0]["proposal_id"] = "f" * 64
            unknown_rows[0]["proposal_sha256"] = "f" * 64
            stale_rows = [dict(row) for row in rows]
            stale_rows[0]["image_sha256"] = "e" * 64
            cases = {
                "duplicate": duplicate_rows,
                "missing": missing_rows,
                "unknown": unknown_rows,
                "stale": stale_rows,
            }
            for label, candidate_rows in cases.items():
                candidate = root / f"{label}.csv"
                _write_review_table(candidate, candidate_rows, REVIEW_COLUMNS)
                with self.subTest(case=label), self.assertRaises(PublicIOError):
                    validate_review_table(exported, candidate)

    def test_canonical_review_actions_round_trip_and_weights_fail_closed(self) -> None:
        from compag_curation.public_io import PublicIOError
        from compag_curation.review.exchange import (
            CANONICAL_REVIEW_COLUMNS,
            REVIEW_ACTION_WEIGHTS,
            validate_review_table,
            write_review_export,
            write_review_import,
        )

        proposals = []
        for index in range(len(REVIEW_ACTION_WEIGHTS)):
            proposal = hashlib.sha256(f"canonical-proposal-{index}".encode("ascii")).hexdigest()
            image = hashlib.sha256(f"canonical-image-{index}".encode("ascii")).hexdigest()
            proposals.append({
                "proposal_id": proposal,
                "proposal_sha256": proposal,
                "image_id": image,
                "image_sha256": image,
                "group_id": f"canonical-{index}",
            })

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exported = root / "canonical-request.csv"
            result = write_review_export(proposals, exported, canonical_actions=True)
            self.assertEqual(result["review_contract"], "CANONICAL_ACTION_WEIGHTED_V2")
            with exported.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
                handle.seek(0)
                self.assertEqual(tuple(next(csv.reader(handle))), CANONICAL_REVIEW_COLUMNS)

            actions = tuple(REVIEW_ACTION_WEIGHTS)
            for index, row in enumerate(rows):
                action = actions[index]
                row.update({
                    "label": str(index % 2),
                    "review_action": action,
                    "review_weight": format(REVIEW_ACTION_WEIGHTS[action], ".1f"),
                    "review_status": "reviewed",
                })
            reviewed = root / "canonical-reviewed.csv"
            _write_review_table(reviewed, rows, CANONICAL_REVIEW_COLUMNS)
            normalized, summary = validate_review_table(exported, reviewed)
            self.assertEqual(summary["review_contract"], "CANONICAL_ACTION_WEIGHTED_V2")
            self.assertEqual(summary["action_counts"], {action: 1 for action in actions})
            self.assertAlmostEqual(float(summary["effective_weight"]), 2.8)
            imported = root / "canonical-imported.csv"
            self.assertEqual(write_review_import(normalized, imported)["review_contract"], "CANONICAL_ACTION_WEIGHTED_V2")
            self.assertEqual(imported.read_bytes(), reviewed.read_bytes())

            wrong_weight = [dict(row) for row in rows]
            wrong_weight[0]["review_weight"] = "0.4"
            invalid = root / "wrong-weight.csv"
            _write_review_table(invalid, wrong_weight, CANONICAL_REVIEW_COLUMNS)
            with self.assertRaisesRegex(PublicIOError, "weight differs"):
                validate_review_table(exported, invalid)

            missing_action = [
                {key: value for key, value in row.items() if key not in {"review_action", "review_weight"}}
                for row in rows
            ]
            legacy_header = tuple(
                key for key in CANONICAL_REVIEW_COLUMNS if key not in {"review_action", "review_weight"}
            )
            invalid_header = root / "legacy-reviewed.csv"
            _write_review_table(invalid_header, missing_action, legacy_header)
            with self.assertRaisesRegex(PublicIOError, "schema differs"):
                validate_review_table(exported, invalid_header)

    def test_canonical_inference_result_requires_exact_method_and_count_closure(self) -> None:
        from compag_curation.canonical.spec import (
            CANONICAL_FEATURE_ORDER_SHA256,
            CANONICAL_GPU_PROFILE,
            CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
            CANONICAL_PROFILE,
        )
        from compag_curation.canonical.active_learning_service import (
            CANONICAL_AL_RAW_FEATURE_ARCHIVE_SCHEMA,
        )
        from compag_curation.canonical.service import (
            CANONICAL_INFERENCE_COLUMNS,
            CANONICAL_RAW_FEATURE_TABLE_COLUMNS,
        )
        from compag_curation.public_io import (
            PublicIOError,
            compact_json_sha256,
            sha256_file,
            write_new_json,
        )
        from compag_curation.public_pipeline import (
            _validated_canonical_inference_result,
            _validated_canonical_predictions,
            _validated_canonical_raw_feature_archive,
        )

        bundle_sha256 = "a" * 64
        inputs = [{
            "filename": "heldout.ppm",
            "sha256": "b" * 64,
            "size_bytes": 128,
            "width": 64,
            "height": 64,
            "exif_orientation": None,
        }]
        raw_feature_payload = (
            ",".join(CANONICAL_RAW_FEATURE_TABLE_COLUMNS) + "\n"
        ).encode("utf-8")
        raw_feature_receipt = {
            "schema": CANONICAL_AL_RAW_FEATURE_ARCHIVE_SCHEMA,
            "status": "PASS",
            "profile": CANONICAL_PROFILE,
            "basename": "raw_features.csv",
            "columns_sha256": compact_json_sha256(
                list(CANONICAL_RAW_FEATURE_TABLE_COLUMNS)
            ),
            "crop_scales": [0.67, 0.8, 1.0, 1.25],
            "rows_per_proposal": 4,
            "proposal_count": 0,
            "row_count": 0,
            "bundle_sha256": bundle_sha256,
            "predictions_sha256": "c" * 64,
            "sha256": hashlib.sha256(raw_feature_payload).hexdigest(),
            "size_bytes": len(raw_feature_payload),
        }
        result = {
            "schema": "compag-curation-canonical-inference/v1",
            "status": "PASS",
            "profile": CANONICAL_PROFILE,
            "device": "cpu",
            "bundle_sha256": bundle_sha256,
            "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
            "threshold": 0.5,
            "threshold_method": "FIXED_CANONICAL_METHOD",
            "full_image_nms_iou": 0.5,
            "sam2_execution_points_per_batch": CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
            "image_count": 1,
            "tile_count": 1,
            "proposal_count": 0,
            "prediction_rows": 0,
            "positive_rows": 0,
            "kept_rows": 0,
            "predictions_sha256": "c" * 64,
            "raw_feature_archive": raw_feature_receipt,
            "input_inventory": [{
                "image_name": "heldout.ppm",
                "image_id": hashlib.sha256(
                    b"compag-image-v1\0heldout.ppm\0heldout\0" + b"b" * 64
                ).hexdigest(),
                "image_sha256": "b" * 64,
                "group_id": "heldout",
                "size_bytes": 128,
                "width": 64,
                "height": 64,
                "tile_count": 1,
                "candidate_rows": 0,
            }],
            "paper_result_reproduction": "NOT_CLAIMED",
        }
        self.assertEqual(
            _validated_canonical_inference_result(
                result,
                bundle_sha256=bundle_sha256,
                expected_inputs=inputs,
            ),
            result,
        )
        gpu_result = {
            **result,
            "profile": CANONICAL_GPU_PROFILE,
            "device": "cuda",
            "raw_feature_archive": {
                **raw_feature_receipt,
                "profile": CANONICAL_GPU_PROFILE,
            },
        }
        self.assertEqual(
            _validated_canonical_inference_result(
                gpu_result,
                bundle_sha256=bundle_sha256,
                expected_inputs=inputs,
                expected_profile=CANONICAL_GPU_PROFILE,
                expected_device="cuda",
            ),
            gpu_result,
        )
        with self.assertRaisesRegex(PublicIOError, "profile/device pairing"):
            _validated_canonical_inference_result(
                gpu_result,
                bundle_sha256=bundle_sha256,
                expected_inputs=inputs,
                expected_profile=CANONICAL_GPU_PROFILE,
                expected_device="cpu",
            )
        for name, changed in (
            ("missing field", {key: value for key, value in result.items() if key != "threshold_method"}),
            ("count drift", {**result, "prediction_rows": 1}),
            (
                "method batch cannot pose as execution microbatch",
                {**result, "sam2_execution_points_per_batch": 512},
            ),
            (
                "legacy SAM2 execution microbatch",
                {**result, "sam2_execution_points_per_batch": 256},
            ),
            ("wrong profile", {**result, "profile": "public-safe-balanced-v1"}),
            (
                "forged group",
                {
                    **result,
                    "input_inventory": [{**result["input_inventory"][0], "group_id": "forged"}],
                },
            ),
            (
                "forged image identity",
                {
                    **result,
                    "input_inventory": [{**result["input_inventory"][0], "image_id": "d" * 64}],
                },
            ),
        ):
            with self.subTest(name=name), self.assertRaises(PublicIOError):
                _validated_canonical_inference_result(
                    changed,
                    bundle_sha256=bundle_sha256,
                    expected_inputs=inputs,
                )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.csv"
            predictions.write_text(",".join(CANONICAL_INFERENCE_COLUMNS) + "\n", encoding="utf-8")
            prediction_sha256 = sha256_file(predictions)
            zero_result = {
                **result,
                "predictions_sha256": prediction_sha256,
                "raw_feature_archive": {
                    **raw_feature_receipt,
                    "predictions_sha256": prediction_sha256,
                },
            }
            self.assertEqual(
                _validated_canonical_predictions(predictions, zero_result),
                {
                    "rows": 0,
                    "positive_rows": 0,
                    "kept_rows": 0,
                    "sha256": zero_result["predictions_sha256"],
                },
            )
            raw_features = root / "raw_features.csv"
            raw_features.write_bytes(raw_feature_payload)
            write_new_json(root / "inference_result.json", zero_result)
            self.assertEqual(
                _validated_canonical_raw_feature_archive(root, zero_result),
                {
                    "schema": "compag-curation-canonical-raw-feature-validation/v1",
                    "status": "PASS",
                    "proposal_count": 0,
                    "row_count": 0,
                    "sha256": raw_feature_receipt["sha256"],
                },
            )
            with raw_features.open("ab") as handle:
                handle.write(b"tamper\n")
            with self.assertRaises(PublicIOError):
                _validated_canonical_raw_feature_archive(root, zero_result)
            malformed = root / "malformed.csv"
            malformed.write_text("this is not a predictions CSV\n", encoding="utf-8")
            with self.assertRaisesRegex(PublicIOError, "CSV header mismatch"):
                _validated_canonical_predictions(
                    malformed,
                    {**result, "predictions_sha256": sha256_file(malformed)},
                )

            proposal_ids = ("1" * 64, "2" * 64)
            nms_rows = []
            for ordinal, proposal_id in enumerate(proposal_ids, start=1):
                scale = "1"
                detection_id = hashlib.sha256(
                    b"compag-canonical-detection-v1\0"
                    + proposal_id.encode("ascii")
                    + b"\0"
                    + scale.encode("ascii")
                ).hexdigest()
                nms_rows.append({
                    "id": str(ordinal),
                    "detection_id": detection_id,
                    "proposal_id": proposal_id,
                    "proposal_sha256": proposal_id,
                    "image": "heldout.ppm",
                    "full_image": "heldout.ppm",
                    "image_id": result["input_inventory"][0]["image_id"],
                    "image_sha256": "b" * 64,
                    "group_id": "heldout",
                    "tile_name": "heldout_y00000x00000.jpg",
                    "tile_sha256": "e" * 64,
                    "scale": scale,
                    "mask_sha256": str(ordinal + 2) * 64,
                    "x": "0",
                    "y": "0",
                    "w": "10",
                    "h": "10",
                    "bbox_x1": "0",
                    "bbox_y1": "0",
                    "bbox_x2": "10",
                    "bbox_y2": "10",
                    "orig_w": "64",
                    "orig_h": "64",
                    "poly": "[]",
                    "xgb_p": "1",
                    "probability": "1",
                    "prediction": "1",
                    "kept": "1",
                })

            def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=CANONICAL_INFERENCE_COLUMNS,
                        lineterminator="\n",
                    )
                    writer.writeheader()
                    writer.writerows(rows)

            over_kept = root / "over-kept.csv"
            write_predictions(over_kept, nms_rows)
            nms_result = {
                **result,
                "proposal_count": 2,
                "prediction_rows": 2,
                "positive_rows": 2,
                "kept_rows": 2,
                "predictions_sha256": sha256_file(over_kept),
                "input_inventory": [
                    {**result["input_inventory"][0], "candidate_rows": 2},
                ],
            }
            with self.assertRaisesRegex(PublicIOError, "full-image NMS"):
                _validated_canonical_predictions(over_kept, nms_result)

            expected_kept = {0}
            forged_tie_rows = [
                {**row, "kept": str(int(index == 1))}
                for index, row in enumerate(nms_rows)
            ]
            forged_tie = root / "forged-tie.csv"
            write_predictions(forged_tie, forged_tie_rows)
            with self.assertRaisesRegex(PublicIOError, "full-image NMS"):
                _validated_canonical_predictions(
                    forged_tie,
                    {
                        **nms_result,
                        "kept_rows": 1,
                        "predictions_sha256": sha256_file(forged_tie),
                    },
                )
            valid_rows = [
                {**row, "kept": str(int(index in expected_kept))}
                for index, row in enumerate(nms_rows)
            ]
            nms_valid = root / "nms-valid.csv"
            write_predictions(nms_valid, valid_rows)
            valid_result = {
                **nms_result,
                "kept_rows": len(expected_kept),
                "predictions_sha256": sha256_file(nms_valid),
            }
            self.assertEqual(
                _validated_canonical_predictions(nms_valid, valid_result)["kept_rows"],
                len(expected_kept),
            )

    def test_review_feasibility_fails_before_immutable_stage_publication(self) -> None:
        from compag_curation import public_pipeline
        from compag_curation.public_backend import review_split_feasibility
        from compag_curation.public_io import PublicIOError, sha256_file, write_new_bytes
        from compag_curation.review.exchange import REVIEW_COLUMNS, write_review_export
        from compag_curation.run_state import (
            STAGE_REQUIRED_FILES,
            RunBindings,
            begin_stage,
            claim_run,
            completed_stage,
            publish_stage,
        )

        proposals: list[dict[str, str]] = []
        for group_index in range(4):
            group = f"group{group_index}"
            image_sha = hashlib.sha256(group.encode("ascii")).hexdigest()
            for label in (0, 1):
                proposal_sha = hashlib.sha256(f"{group}:{label}".encode("ascii")).hexdigest()
                proposals.append(
                    {
                        "proposal_id": proposal_sha,
                        "proposal_sha256": proposal_sha,
                        "image_id": image_sha,
                        "image_sha256": image_sha,
                        "group_id": group,
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exported = root / "review_request.csv"
            write_review_export(proposals, exported)
            with exported.open("r", encoding="utf-8", newline="") as handle:
                pending = list(csv.DictReader(handle))

            invalid_rows = [dict(row, label="0", review_status="reviewed") for row in pending]
            invalid = root / "invalid.csv"
            _write_review_table(invalid, invalid_rows, REVIEW_COLUMNS)
            with self.assertRaisesRegex(PublicIOError, "partition does not contain both labels"):
                public_pipeline._validated_review_source(exported, invalid)

            valid_rows = []
            for row in pending:
                label = str(int(row["proposal_id"] == hashlib.sha256(f"{row['group_id']}:1".encode("ascii")).hexdigest()))
                valid_rows.append(dict(row, label=label, review_status="reviewed"))
            valid = root / "valid.csv"
            _write_review_table(valid, valid_rows, REVIEW_COLUMNS)
            expected_sha, row_count = public_pipeline._validated_review_source(exported, valid)
            self.assertEqual(row_count, 8)

            plan = review_split_feasibility([(row["group_id"], int(row["label"])) for row in valid_rows])
            train_groups = list(plan["train_groups"])
            smote_invalid: list[tuple[str, int]] = []
            for group_index in range(4):
                group = f"group{group_index}"
                if group in train_groups:
                    smote_invalid.extend((group, 0) for _ in range(5))
                    if group == train_groups[0]:
                        smote_invalid.append((group, 1))
                else:
                    smote_invalid.extend(((group, 0), (group, 1)))
            with self.assertRaisesRegex(PublicIOError, "too few minority rows"):
                review_split_feasibility(smote_invalid)

            run = root / "run"
            state = claim_run(run, RunBindings(*(("0" * 64,) * 4)))
            try:
                for name in ("00_input_inventory", "10_prepare"):
                    workspace = begin_stage(state, name)
                    for relative in STAGE_REQUIRED_FILES[name]:
                        destination = workspace.root / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        write_new_bytes(destination, f"fixture:{relative}\n".encode("ascii"))
                    publish_stage(state, workspace, STAGE_REQUIRED_FILES[name])

                workspace = begin_stage(state, "20_proposals_features")
                for relative in STAGE_REQUIRED_FILES["20_proposals_features"]:
                    destination = workspace.root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if relative == "review_request.csv":
                        write_review_export(proposals, destination)
                    else:
                        write_new_bytes(destination, f"fixture:{relative}\n".encode("ascii"))
                publish_stage(state, workspace, STAGE_REQUIRED_FILES["20_proposals_features"])
                published_export = state.stages / "20_proposals_features/review_request.csv"

                with self.assertRaisesRegex(PublicIOError, "partition does not contain both labels"):
                    public_pipeline._produce_stage(
                        state,
                        "30_review_import",
                        STAGE_REQUIRED_FILES["30_review_import"],
                        lambda target: public_pipeline._stage_review(
                            published_export,
                            invalid,
                            sha256_file(invalid),
                            target,
                        ),
                    )
                self.assertIsNone(completed_stage(state, "30_review_import"))

                public_pipeline._produce_stage(
                    state,
                    "30_review_import",
                    STAGE_REQUIRED_FILES["30_review_import"],
                    lambda target: public_pipeline._stage_review(published_export, valid, expected_sha, target),
                )
                self.assertIsNotNone(completed_stage(state, "30_review_import"))
            finally:
                state.close()

    def test_run_state_lock_resume_drift_and_stage_tamper(self) -> None:
        from compag_curation import public_pipeline, run_state
        from compag_curation.model_bundle import BUNDLE_SCHEMA
        from compag_curation.public_io import PublicIOError, manifest_rows, write_new_bytes
        from compag_curation.public_pipeline import _existing_completion, _finalize_run
        from compag_curation.run_state import (
            STAGE_NAMES,
            STAGE_REQUIRED_FILES,
            RunBindings,
            begin_stage,
            claim_run,
            completed_stage,
            open_resume,
            publish_stage,
        )

        bindings = RunBindings(*(("0" * 64,) * 4))
        changed = RunBindings("1" * 64, *(("0" * 64,) * 3))

        def publish_fixture_stage(state: object, name: str, label: str) -> None:
            workspace = begin_stage(state, name)
            for relative in STAGE_REQUIRED_FILES[name]:
                destination = workspace.root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if relative == "model_bundle/bundle.json":
                    public_pipeline.write_new_json(destination, {"schema": BUNDLE_SCHEMA})
                else:
                    write_new_bytes(destination, f"{label}:{relative}\n".encode("ascii"))
            publish_stage(state, workspace, STAGE_REQUIRED_FILES[name])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            state = claim_run(output, bindings)
            try:
                with self.assertRaisesRegex(PublicIOError, "RUN_ALREADY_ACTIVE"):
                    open_resume(output, bindings)
                publish_fixture_stage(state, "00_input_inventory", "original")
            finally:
                state.close()

            with open_resume(output, bindings) as resumed:
                self.assertIsNotNone(completed_stage(resumed, "00_input_inventory"))
            with self.assertRaisesRegex(PublicIOError, "RESUME_DRIFT"):
                open_resume(output, changed)

            (output / "stages/00_input_inventory/config.toml").write_bytes(b"tampered\n")
            with self.assertRaisesRegex(PublicIOError, "stage drift"):
                open_resume(output, bindings)

            interrupted = Path(directory) / "interrupted-run"
            real_publish = run_state.publish_directory_noreplace

            def fail_final_publish(source: Path, destination: Path) -> None:
                if destination == interrupted:
                    raise PublicIOError("injected initialization interruption")
                real_publish(source, destination)

            with mock.patch.object(run_state, "publish_directory_noreplace", side_effect=fail_final_publish):
                with self.assertRaisesRegex(PublicIOError, "initialization interruption"):
                    claim_run(interrupted, bindings)
            self.assertFalse(interrupted.exists())
            self.assertEqual(len(list(Path(directory).glob(".interrupted-run.run-init.*"))), 1)
            with claim_run(interrupted, bindings):
                pass

            recovered_run = Path(directory) / "recovered-run"
            from compag_curation import public_io

            real_rename = public_io.rename_noreplace

            def run_publish_then_raise(source: Path, destination: Path) -> None:
                real_rename(source, destination)
                if destination == recovered_run:
                    raise OSError("injected run destination fsync failure")

            with mock.patch.object(public_io, "rename_noreplace", side_effect=run_publish_then_raise):
                with claim_run(recovered_run, bindings) as recovered_state:
                    self.assertEqual(recovered_state.root, recovered_run)

            stage_interrupted = Path(directory) / "stage-interrupted-run"
            state = claim_run(stage_interrupted, bindings)
            real_write_json = run_state.write_new_json

            def fail_attempt_marker(path: Path, value: object, mode: int = 0o644) -> None:
                if path.name == "_ATTEMPT.json":
                    raise PublicIOError("injected attempt marker interruption")
                real_write_json(path, value, mode)

            try:
                with mock.patch.object(run_state, "write_new_json", side_effect=fail_attempt_marker):
                    with self.assertRaisesRegex(PublicIOError, "attempt marker interruption"):
                        begin_stage(state, STAGE_NAMES[0])
            finally:
                state.close()
            self.assertEqual(list((stage_interrupted / ".staging").iterdir()), [])
            self.assertEqual(len(list(Path(directory).glob(".compag-stage-init.*"))), 1)
            with open_resume(stage_interrupted, bindings) as resumed:
                for index, name in enumerate(STAGE_NAMES):
                    publish_fixture_stage(resumed, name, f"recovered-{index}")
                self.assertEqual(_finalize_run(resumed)["status"], "PASS")

            partial_event = Path(directory) / "partial-event-run"
            with claim_run(partial_event, bindings):
                pass
            staged_event = partial_event / ".staging/event.00000001.00000000-0000-4000-8000-000000000000.json"
            write_new_bytes(staged_event, b'{"partial":')
            nonfinite_events = []
            for sequence, constant in enumerate(("NaN", "Infinity", "-Infinity"), start=2):
                path = partial_event / f".staging/event.{sequence:08d}.00000000-0000-4000-8000-00000000000{sequence}.json"
                write_new_bytes(path, f'{{"junk":{constant}}}\n'.encode("ascii"))
                nonfinite_events.append(path)
            with open_resume(partial_event, bindings) as resumed:
                sidecar = staged_event.with_name(f"{staged_event.name}.classification.json")
                classified = json.loads(sidecar.read_text(encoding="utf-8"))
                self.assertEqual(classified["classification"], "PARTIAL_OR_INVALID_EVENT")
                for path in nonfinite_events:
                    record = json.loads(path.with_name(f"{path.name}.classification.json").read_text(encoding="ascii"))
                    self.assertEqual(record["classification"], "PARTIAL_OR_INVALID_EVENT")
                for index, name in enumerate(STAGE_NAMES):
                    publish_fixture_stage(resumed, name, f"classified-{index}")
                self.assertEqual(_finalize_run(resumed)["status"], "PASS")
            with open_resume(partial_event, bindings) as resumed:
                self.assertEqual(_existing_completion(resumed)["status"], "PASS")

            failed_residue = Path(directory) / "failed-residue-run"
            state = claim_run(failed_residue, bindings)
            try:
                workspace = begin_stage(state, STAGE_NAMES[0])
                (workspace.root / "producer-created-empty").mkdir()
                run_state.record_stage_failure(state, workspace, PublicIOError("injected producer failure"))
            finally:
                state.close()
            with open_resume(failed_residue, bindings) as resumed:
                for index, name in enumerate(STAGE_NAMES):
                    publish_fixture_stage(resumed, name, f"residue-{index}")
                self.assertEqual(_finalize_run(resumed)["status"], "PASS")
            self.assertTrue(
                any(
                    (path / "producer-created-empty").is_dir()
                    for path in (failed_residue / ".staging").iterdir()
                    if path.is_dir()
                )
            )
            residue_attempt = next(path for path in (failed_residue / ".staging").iterdir() if path.is_dir())
            late_empty = residue_attempt / "late-empty"
            late_empty.mkdir()
            with open_resume(failed_residue, bindings) as resumed:
                with self.assertRaisesRegex(PublicIOError, "completed run output changed"):
                    _existing_completion(resumed)
            late_empty.rmdir()
            original_empty = residue_attempt / "producer-created-empty"
            original_empty.rmdir()
            with open_resume(failed_residue, bindings) as resumed:
                with self.assertRaisesRegex(PublicIOError, "completed run output changed"):
                    _existing_completion(resumed)

            recovered_publication = Path(directory) / "recovered-publication-run"
            state = claim_run(recovered_publication, bindings)
            original_append = state.append_event

            def fail_publication_event(event: str, **fields: object) -> None:
                if event == "STAGE_PUBLISHED":
                    raise PublicIOError("injected post-publication event interruption")
                original_append(event, **fields)

            try:
                with mock.patch.object(state, "append_event", side_effect=fail_publication_event):
                    with self.assertRaisesRegex(PublicIOError, "event interruption"):
                        publish_fixture_stage(state, STAGE_NAMES[0], "recovered-anchor")
            finally:
                state.close()
            with open_resume(recovered_publication, bindings) as resumed:
                events = [json.loads(path.read_text(encoding="ascii")) for path in sorted((recovered_publication / "events").iterdir())]
                self.assertEqual(events[-1]["event"], "STAGE_PUBLICATION_RECOVERED")
                self.assertIsNotNone(completed_stage(resumed, STAGE_NAMES[0]))
                with self.assertRaisesRegex(PublicIOError, "exact completed stage set"):
                    _finalize_run(resumed)

            diagnostic_interruption = Path(directory) / "diagnostic-interruption-run"
            state = claim_run(diagnostic_interruption, bindings)
            workspace = begin_stage(state, STAGE_NAMES[0])
            for relative in STAGE_REQUIRED_FILES[STAGE_NAMES[0]]:
                destination = workspace.root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                write_new_bytes(destination, f"diagnostic:{relative}\n".encode("ascii"))
            publish_stage(state, workspace, STAGE_REQUIRED_FILES[STAGE_NAMES[0]])
            real_failure_write = run_state.write_new_json

            def interrupt_failure_marker(path: Path, value: object, mode: int = 0o644) -> None:
                if path.parent.name.startswith(".compag-stage-failure-init.") and path.name == "_FAILURE.json":
                    raise PublicIOError("injected failure-marker interruption")
                real_failure_write(path, value, mode)

            try:
                with mock.patch.object(run_state, "write_new_json", side_effect=interrupt_failure_marker):
                    run_state.record_stage_failure(state, workspace, PublicIOError("post-publish failure"))
            finally:
                state.close()
            self.assertFalse(any((diagnostic_interruption / ".staging").glob("*.postpublish-failure.*")))
            self.assertEqual(len(list(Path(directory).glob(".compag-stage-failure-init.*"))), 1)
            with open_resume(diagnostic_interruption, bindings) as resumed:
                self.assertIsNotNone(completed_stage(resumed, STAGE_NAMES[0]))

            receipt_tamper = Path(directory) / "receipt-anchor-tamper-run"
            state = claim_run(receipt_tamper, bindings)
            try:
                for index, name in enumerate(STAGE_NAMES):
                    publish_fixture_stage(state, name, f"anchored-{index}")
                stage_root = receipt_tamper / "stages" / STAGE_NAMES[0]
                (stage_root / "config.toml").write_bytes(b"co-tampered\n")
                receipt_path = stage_root / "_SUCCESS.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                tampered_rows = manifest_rows(stage_root, excluded={"_SUCCESS.json"})
                receipt["files"] = tampered_rows
                receipt["files_sha256"] = public_pipeline.compact_json_sha256(tampered_rows)
                receipt_path.write_bytes(public_pipeline.canonical_json_bytes(receipt))
                with self.assertRaisesRegex(PublicIOError, "ledger anchor changed"):
                    _finalize_run(state)
            finally:
                state.close()
            with self.assertRaisesRegex(PublicIOError, "ledger anchor changed"):
                open_resume(receipt_tamper, bindings)

            unknown_top = Path(directory) / "unknown-top-run"
            state = claim_run(unknown_top, bindings)
            try:
                for index, name in enumerate(STAGE_NAMES):
                    publish_fixture_stage(state, name, f"unknown-top-{index}")
                accident = unknown_top / "PRIVATE_ACCIDENT.txt"
                write_new_bytes(accident, b"private accident\n")
                with self.assertRaisesRegex(PublicIOError, "top-level namespace closure"):
                    _finalize_run(state)
                self.assertEqual(accident.read_bytes(), b"private accident\n")
            finally:
                state.close()

            unknown_staging = Path(directory) / "unknown-staging-run"
            state = claim_run(unknown_staging, bindings)
            try:
                for index, name in enumerate(STAGE_NAMES):
                    publish_fixture_stage(state, name, f"unknown-staging-{index}")
                accident = unknown_staging / ".staging/unknown.txt"
                write_new_bytes(accident, b"unknown staging\n")
                with self.assertRaisesRegex(PublicIOError, "staging namespace"):
                    _finalize_run(state)
                self.assertEqual(accident.read_bytes(), b"unknown staging\n")
            finally:
                state.close()

            private_locator = Path(directory) / "private-locator-run"
            state = claim_run(private_locator, bindings)
            try:
                workspace = begin_stage(state, STAGE_NAMES[0])
                write_new_bytes(workspace.root / "diagnostic.txt", b"source=/ho" + b"me/alice/secret.csv\n")
                private_empty = workspace.root / "home/alice/Clean"
                private_empty.mkdir(parents=True)
                run_state.record_stage_failure(state, workspace, PublicIOError("injected producer failure"))
            finally:
                state.close()
            with open_resume(private_locator, bindings) as resumed:
                for index, name in enumerate(STAGE_NAMES):
                    publish_fixture_stage(resumed, name, f"private-locator-{index}")
                with self.assertRaisesRegex(PublicIOError, "empty output directory"):
                    _finalize_run(resumed)
                private_empty.rmdir()
                private_empty.parent.rmdir()
                private_empty.parent.parent.rmdir()
                with self.assertRaisesRegex(PublicIOError, "private locator"):
                    _finalize_run(resumed)
            self.assertTrue(any(b"/ho" + b"me/alice/" in path.read_bytes() for path in (private_locator / ".staging").glob("*/diagnostic.txt")))

            completed = Path(directory) / "completed-run"
            state = claim_run(completed, bindings)
            try:
                for index, name in enumerate(STAGE_NAMES):
                    publish_fixture_stage(state, name, f"stage-{index}")
                self.assertEqual(_finalize_run(state)["status"], "PASS")
            finally:
                state.close()
            before_resume = manifest_rows(completed, allowed_empty_directories={".staging"})
            with open_resume(completed, bindings) as resumed:
                self.assertEqual(_existing_completion(resumed)["status"], "PASS")
            self.assertEqual(manifest_rows(completed, allowed_empty_directories={".staging"}), before_resume)

            durability = Path(directory) / "durability-run"
            state = claim_run(durability, bindings)
            try:
                for index, name in enumerate(STAGE_NAMES):
                    publish_fixture_stage(state, name, f"durability-{index}")
                real_final_publish = public_pipeline.rename_noreplace

                def publish_then_raise(source: Path, destination: Path) -> None:
                    real_final_publish(source, destination)
                    if destination == durability / "FINAL":
                        raise OSError("injected post-rename durability error")

                with mock.patch.object(public_pipeline, "rename_noreplace", side_effect=publish_then_raise):
                    with self.assertRaisesRegex(PublicIOError, "durability confirmation failed"):
                        _finalize_run(state)
            finally:
                state.close()
            sealed_before = manifest_rows(durability, allowed_empty_directories={".staging"})
            with open_resume(durability, bindings) as resumed:
                self.assertEqual(_existing_completion(resumed)["status"], "PASS")
            self.assertEqual(manifest_rows(durability, allowed_empty_directories={".staging"}), sealed_before)

            extra = completed / "unexpected-empty-directory"
            extra.mkdir()
            with self.assertRaisesRegex(PublicIOError, "undeclared empty directory"):
                manifest_rows(completed, allowed_empty_directories={".staging"})

    def test_run_json_recovery_and_directory_metadata_are_fail_closed(self) -> None:
        from compag_curation import public_pipeline, run_state
        from compag_curation.public_io import PublicIOError, canonical_json_bytes, compact_json_sha256, write_new_bytes
        from compag_curation.public_pipeline import _existing_completion, _finalize_run
        from compag_curation.run_state import (
            STAGE_NAMES,
            STAGE_REQUIRED_FILES,
            RunBindings,
            begin_stage,
            claim_run,
            open_resume,
            publish_stage,
        )

        bindings = RunBindings(*(("0" * 64,) * 4))

        def publish_all(state: object) -> None:
            for index, name in enumerate(STAGE_NAMES):
                workspace = begin_stage(state, name)
                for relative in STAGE_REQUIRED_FILES[name]:
                    destination = workspace.root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    write_new_bytes(destination, f"strict-{index}:{relative}\n".encode("ascii"))
                publish_stage(state, workspace, STAGE_REQUIRED_FILES[name])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty_run = root / "empty-run"
            with claim_run(empty_run, bindings):
                pass
            with open_resume(empty_run, bindings) as state:
                self.assertIsNone(_existing_completion(state))
            closed_state = claim_run(root / "closed-state-run", bindings)
            closed_state.close()
            with self.assertRaisesRegex(PublicIOError, "closed and cannot be mutated"):
                closed_state.append_event("AFTER_CLOSE")

            missing_genesis = root / "missing-genesis-run"
            with claim_run(missing_genesis, bindings):
                pass
            (missing_genesis / "events/00000000.json").rename(root / "preserved-genesis-event.json")
            with self.assertRaisesRegex(PublicIOError, "missing its genesis"):
                open_resume(missing_genesis, bindings)

            policy_run = root / "policy-run"
            with claim_run(policy_run, bindings):
                pass
            owner_path = policy_run / "RUN_OWNER.json"
            owner = json.loads(owner_path.read_text(encoding="ascii"))
            owner["output_policy"] = "ALLOW_CLOBBER"
            owner_path.write_bytes(canonical_json_bytes(owner))
            with self.assertRaisesRegex(PublicIOError, "owner schema"):
                open_resume(policy_run, bindings)

            event_run = root / "event-run"
            with claim_run(event_run, bindings) as state:
                run_id = state.run_id
            tip = json.loads((event_run / "events/00000000.json").read_text(encoding="ascii"))["event_sha256"]
            event = {
                "sequence": 1,
                "previous_event_sha256": tip,
                "timestamp_utc": "2026-08-27T00:00:00Z",
                "run_id": run_id,
                "event": "INTERRUPTED_FIXTURE",
            }
            event["event_sha256"] = compact_json_sha256(event)
            staged = event_run / ".staging/event.00000001.00000000-0000-4000-8000-000000000010.json"
            write_new_bytes(staged, run_state._event_payload(event))
            deep = event_run / ".staging/event.00000002.00000000-0000-4000-8000-000000000011.json"
            write_new_bytes(deep, b"[" * 2000 + b"0" + b"]" * 2000 + b"\n")
            wrong_tip = dict(event)
            wrong_tip["previous_event_sha256"] = "f" * 64
            wrong_tip["event_sha256"] = compact_json_sha256({key: value for key, value in wrong_tip.items() if key != "event_sha256"})
            wrong = event_run / ".staging/event.00000001.00000000-0000-4000-8000-000000000012.json"
            write_new_bytes(wrong, run_state._event_payload(wrong_tip))
            with open_resume(event_run, bindings, record_event=True):
                pass
            complete_sidecar = json.loads(staged.with_name(f"{staged.name}.classification.json").read_text(encoding="ascii"))
            self.assertEqual(complete_sidecar["classification"], "UNPUBLISHED_COMPLETE_EVENT")
            self.assertEqual(
                json.loads(deep.with_name(f"{deep.name}.classification.json").read_text(encoding="ascii"))["classification"],
                "PARTIAL_OR_INVALID_EVENT",
            )
            self.assertEqual(
                json.loads(wrong.with_name(f"{wrong.name}.classification.json").read_text(encoding="ascii"))["classification"],
                "PARTIAL_OR_INVALID_EVENT",
            )
            with open_resume(event_run, bindings):
                pass
            self.assertEqual(
                json.loads(staged.with_name(f"{staged.name}.classification.json").read_text(encoding="ascii")),
                complete_sidecar,
            )

            failure_run = root / "failure-run"
            state = claim_run(failure_run, bindings)
            workspace = begin_stage(state, STAGE_NAMES[0])
            real_write = run_state.write_new_json

            def partial_failure_write(path: Path, value: object, mode: int = 0o644) -> None:
                if path.name.startswith(".compag-stage-failure-file-init."):
                    path.write_bytes(b"{")
                    raise OSError("injected partial failure receipt")
                real_write(path, value, mode)

            try:
                with mock.patch.object(run_state, "write_new_json", side_effect=partial_failure_write):
                    run_state.record_stage_failure(state, workspace, PublicIOError("fixture failure"))
                self.assertFalse((workspace.root / "_FAILURE.json").exists())
            finally:
                state.close()
            with open_resume(failure_run, bindings):
                pass

            bounded_receipt_run = root / "bounded-receipt-run"
            with claim_run(bounded_receipt_run, bindings) as state:
                workspace = begin_stage(state, STAGE_NAMES[0])
                for relative in STAGE_REQUIRED_FILES[STAGE_NAMES[0]]:
                    destination = workspace.root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    write_new_bytes(destination, b"bounded\n")
                publish_stage(state, workspace, STAGE_REQUIRED_FILES[STAGE_NAMES[0]])
            with (
                mock.patch.object(run_state, "MAX_STAGE_RECEIPT_BYTES", 8),
                self.assertRaisesRegex(PublicIOError, "stage receipt cannot be parsed"),
            ):
                open_resume(bounded_receipt_run, bindings)

            completed = root / "metadata-run"
            with claim_run(completed, bindings) as state:
                publish_all(state)
                self.assertEqual(_finalize_run(state)["status"], "PASS")

            os.chmod(completed, 0o777)
            with self.assertRaisesRegex(PublicIOError, "run output is unsafe"):
                open_resume(completed, bindings)
            os.chmod(completed, 0o755)

            os.chmod(completed / "FINAL", 0o777)
            with self.assertRaisesRegex(PublicIOError, "evidence directory metadata"):
                open_resume(completed, bindings)
            os.chmod(completed / "FINAL", 0o755)

            nested = completed / "stages/50_train_bundle/model_bundle"
            original_mode = stat.S_IMODE(nested.stat().st_mode)
            os.chmod(nested, 0o700 if original_mode != 0o700 else 0o755)
            with self.assertRaisesRegex(PublicIOError, "directory metadata changed"):
                open_resume(completed, bindings)
            os.chmod(nested, original_mode)

            manifest_path = completed / "FINAL/OUTPUT_MANIFEST.json"
            manifest_payload = manifest_path.read_bytes()
            event_count = len(list((completed / "events").iterdir()))
            with self.assertRaisesRegex(PublicIOError, "finalized run cannot be mutated"):
                open_resume(completed, bindings, record_event=True)
            self.assertEqual(len(list((completed / "events").iterdir())), event_count)

            os.chmod(manifest_path, 0o600)
            with open_resume(completed, bindings) as state:
                with self.assertRaisesRegex(PublicIOError, "evidence file metadata"):
                    _existing_completion(state)
            os.chmod(manifest_path, 0o644)

            manifest_path.write_bytes(b'{"status":"FAIL",' + manifest_payload[1:])
            with open_resume(completed, bindings) as state:
                with self.assertRaisesRegex(PublicIOError, "not strict JSON"):
                    _existing_completion(state)
            manifest_path.write_bytes(b'{"nonfinite":NaN,' + manifest_payload[1:])
            with open_resume(completed, bindings) as state:
                with self.assertRaisesRegex(PublicIOError, "not strict JSON"):
                    _existing_completion(state)
            manifest_path.write_bytes(manifest_payload)
            with open_resume(completed, bindings) as state:
                self.assertEqual(_existing_completion(state)["status"], "PASS")

            no_chmod = root / "final-mode-run"
            state = claim_run(no_chmod, bindings)
            try:
                publish_all(state)
                with (
                    mock.patch.object(public_pipeline.os, "chmod", return_value=None),
                    self.assertRaisesRegex(PublicIOError, "FINAL staging directory metadata"),
                ):
                    _finalize_run(state)
                self.assertFalse((no_chmod / "FINAL").exists())
            finally:
                state.close()

    def test_doctor_and_duplicate_asset_preflight_remain_metadata_only(self) -> None:
        from compag_curation import assets, public_backend, public_pipeline, quick_demo
        from compag_curation.assets import AssetSpec
        from compag_curation.public_io import PublicIOError

        def version(name: str) -> str:
            if name == "compag-curation":
                return "1.9.3"
            raise importlib.metadata.PackageNotFoundError(name)

        with (
            mock.patch.object(public_pipeline.importlib.metadata, "version", side_effect=version),
            mock.patch.object(public_backend.platform, "platform", side_effect=AssertionError("subprocess-capable platform probe")),
        ):
            base, base_rc = public_pipeline.doctor("base")
            quick, quick_rc = public_pipeline.doctor("quick")
        self.assertEqual((base["status"], base_rc), ("PASS", 0))
        self.assertEqual((quick["status"], quick_rc), ("FAIL", 1))
        self.assertEqual(set(quick["checks"]["missing"]), {"numpy", "opencv-python"})

        with (
            mock.patch("compag_curation.public_backend.require_quick_dependencies", side_effect=PublicIOError("lock mismatch")),
            mock.patch.object(quick_demo, "_demo_output") as claim_output,
            self.assertRaisesRegex(PublicIOError, "lock mismatch"),
        ):
            quick_demo.run_quick_demo(Path("unused-output"))
        claim_output.assert_not_called()

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(public_pipeline, "fetch_asset") as fetch:
            with self.assertRaisesRegex(PublicIOError, "duplicate asset request"):
                public_pipeline.fetch_assets(Path(directory), ["sam2-apache-license", "sam2-apache-license"])
            fetch.assert_not_called()

        request = urllib.request.Request("https://raw.githubusercontent.com/allowed/start")
        handler = assets._AllowlistedRedirectHandler()
        with mock.patch.object(urllib.request.HTTPRedirectHandler, "redirect_request") as parent:
            with self.assertRaisesRegex(PublicIOError, "redirect.*allowlist"):
                handler.redirect_request(request, None, 302, "redirect", {}, "https://unreviewed.invalid/payload")
            parent.assert_not_called()
            handler.redirect_request(request, None, 302, "redirect", {}, "/facebookresearch/sam2/raw")
            parent.assert_called_once()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"fixture-asset\n"
            original = root / "original.bin"
            linked = root / "fixture.bin"
            original.write_bytes(payload)
            os.link(original, linked)
            spec = AssetSpec(
                asset_id="fixture",
                kind="test",
                filename=linked.name,
                url="https://raw.githubusercontent.com/facebookresearch/sam2/fixture",
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                license_spdx="Apache-2.0",
                license_url="https://www.apache.org/licenses/LICENSE-2.0",
                publisher="fixture",
                source_commit="0" * 40,
                bundled_in_repository=False,
                device_requirements="none",
                acquisition_method="test_only",
            )
            with mock.patch.dict(assets._ASSETS, {"fixture": spec}, clear=True):
                with self.assertRaisesRegex(PublicIOError, "single-link"):
                    assets.verify_asset(root, "fixture")

    def test_real_demo_seal_declares_only_its_contractual_empty_directories(self) -> None:
        from compag_curation import public_io
        from compag_curation.public_io import PublicIOError
        from compag_curation.quick_demo import _seal_demo

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            output = root / "real-demo"
            for relative in ("project/annotations", "project/review", "run/.staging"):
                (staging / relative).mkdir(parents=True, exist_ok=True)
            (staging / "payload.txt").write_bytes(b"real-demo-fixture\n")
            result = _seal_demo(output, staging, "real", "00000000-0000-0000-0000-000000000000")
            self.assertEqual(result["status"], "PASS")

            recovered_staging = root / "recovered-staging"
            recovered_output = root / "recovered-demo"
            (recovered_staging / "project/annotations").mkdir(parents=True)
            (recovered_staging / "project/review").mkdir(parents=True)
            (recovered_staging / "run/.staging").mkdir(parents=True)
            (recovered_staging / "payload.txt").write_bytes(b"post-rename-recovery\n")
            real_publish = public_io.rename_noreplace

            def publish_then_raise(source: Path, destination: Path) -> None:
                real_publish(source, destination)
                if destination == recovered_output:
                    raise OSError("injected destination fsync failure")

            with mock.patch.object(public_io, "rename_noreplace", side_effect=publish_then_raise):
                recovered = _seal_demo(
                    recovered_output,
                    recovered_staging,
                    "real",
                    "00000000-0000-0000-0000-000000000002",
                )
            self.assertEqual(recovered["status"], "PASS")

            tampered_staging = root / "tampered-staging"
            tampered_output = root / "tampered-demo"
            (tampered_staging / "payload.txt").mkdir(parents=True)
            (tampered_staging / "payload.txt/data.txt").write_bytes(b"metadata-tamper\n")

            def publish_then_tamper(source: Path, destination: Path) -> None:
                real_publish(source, destination)
                manifest_path = destination / "OUTPUT_MANIFEST.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["private_path"] = "/ho" + "me/private/should-not-pass"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with mock.patch("compag_curation.quick_demo.publish_directory_noreplace", side_effect=publish_then_tamper):
                with self.assertRaisesRegex(PublicIOError, "manifest verification|completion evidence"):
                    _seal_demo(
                        tampered_output,
                        tampered_staging,
                        "quick",
                        "00000000-0000-0000-0000-000000000003",
                    )

            invalid_staging = root / "invalid-staging"
            invalid_output = root / "invalid-demo"
            (invalid_staging / "project/annotations").mkdir(parents=True)
            (invalid_staging / "unexpected-empty").mkdir()
            with self.assertRaisesRegex(PublicIOError, "undeclared empty directory"):
                _seal_demo(invalid_output, invalid_staging, "real", "00000000-0000-0000-0000-000000000001")
            self.assertFalse(invalid_output.exists())

    def test_real_demo_fresh_process_contract_binds_runner_environment_inputs_and_result(self) -> None:
        from compag_curation.domain.inference import ProcessResult
        from compag_curation.public_io import (
            PublicIOError,
            compact_json_sha256,
            manifest_rows,
            sha256_file,
            write_new_bytes,
            write_new_json,
        )
        from compag_curation.quick_demo import _run_fresh_demo_inference

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            run_root = root / "run"
            images = project / "data/inference_images"
            bundle = run_root / "stages/50_train_bundle/model_bundle"
            stage_zero = run_root / "stages/00_input_inventory"
            stage_sixty = run_root / "stages/60_evaluate"
            for path in (images, bundle, stage_zero, stage_sixty):
                path.mkdir(parents=True, exist_ok=True)
            image_payload = _ppm(91)
            image = images / "heldout__card.ppm"
            image.write_bytes(image_payload)
            image_row = {
                "filename": image.name,
                "group_id": "heldout",
                "sha256": hashlib.sha256(image_payload).hexdigest(),
                "size_bytes": len(image_payload),
                "width": 64,
                "height": 64,
                "exif_orientation": None,
            }
            write_new_json(bundle / "bundle.json", {"schema": "fixture-bundle/v1"})
            predictions = b"proposal_id,image_id,group_id,x,y,w,h,probability,prediction,kept\n"
            write_new_bytes(stage_sixty / "predictions.csv", predictions)
            expected_result = {
                "schema": "compag-curation-bundle-inference/v1",
                "status": "PASS",
                "bundle_sha256": sha256_file(bundle / "bundle.json"),
                "feature_order_sha256": "a" * 64,
                "images": 1,
                "proposals": 0,
                "positive_predictions": 0,
                "kept_predictions": 0,
                "predictions_sha256": hashlib.sha256(predictions).hexdigest(),
            }
            write_new_json(stage_sixty / "inference_result.json", expected_result)
            write_new_json(stage_zero / "input_inventory.json", {
                "schema": "compag-curation-run-input-inventory/v1",
                "inspection": {"inference_images": [image_row]},
            })

            class FixtureRunner:
                def __init__(
                    self,
                    *,
                    wrong_contract: bool = False,
                    forged_result: bool = False,
                    malformed_input_manifest: bool = False,
                ) -> None:
                    self.wrong_contract = wrong_contract
                    self.forged_result = forged_result
                    self.malformed_input_manifest = malformed_input_manifest
                    self.calls: list[tuple[tuple[str, ...], Path | None, dict[str, str]]] = []

                def run(
                    self,
                    argv: object,
                    *,
                    cwd: Path | None,
                    env: object,
                    acceptable_exit_codes: frozenset[int],
                    capture_output: bool,
                ) -> ProcessResult:
                    actual_argv = tuple(str(value) for value in argv)
                    actual_env = dict(env)
                    self.calls.append((actual_argv, cwd, actual_env))
                    cache = Path(actual_env["TRITON_CACHE_DIR"]) / "9/0"
                    cache.mkdir(parents=True, mode=0o700)
                    write_new_bytes(
                        cache / "c25eed8a829ad6",
                        b"compiled cache containing /home/private\n",
                    )
                    output = Path(actual_argv[actual_argv.index("--output") + 1])
                    output.mkdir(mode=0o700)
                    write_new_bytes(output / "predictions.csv", predictions)
                    result = dict(expected_result)
                    if self.forged_result:
                        result["schema"] = "forged-result/v1"
                    write_new_json(output / "inference_result.json", result)
                    run_id = "00000000-0000-4000-8000-000000000020"
                    write_new_json(output / "FINAL_STATUS.json", {
                        "schema": "compag-curation-inference-completion/v1",
                        "status": "PASS",
                        "run_id": run_id,
                        "output_name": output.name,
                        "publication_condition": "VALID_ONLY_AT_DECLARED_OUTPUT_NAME_AFTER_ATOMIC_RENAME",
                        "bundle_sha256": expected_result["bundle_sha256"],
                        "inference_result_sha256": sha256_file(output / "inference_result.json"),
                    })
                    input_identity = [{
                        **{key: image_row[key] for key in ("filename", "sha256", "size_bytes", "width", "height", "exif_orientation")},
                        "mode_octal": "0644",
                        "uid": os.getuid(),
                        "gid": os.getgid(),
                        "nlink": 1,
                        "mtime_ns": image.stat().st_mtime_ns,
                        "ctime_ns": image.stat().st_ctime_ns,
                    }]
                    if self.malformed_input_manifest:
                        input_identity.append({})
                    members = manifest_rows(output, excluded={"OUTPUT_MANIFEST.json"})
                    write_new_json(output / "OUTPUT_MANIFEST.json", {
                        "schema": "compag-curation-inference-output-manifest/v1",
                        "status": "PASS",
                        "run_id": run_id,
                        "bundle_sha256": expected_result["bundle_sha256"],
                        "input_images": input_identity,
                        "members": members,
                        "members_sha256": compact_json_sha256(members),
                    })
                    stdout = json.dumps({
                        **result,
                        "run_id": run_id,
                        "output_name": output.name,
                        "output_manifest_sha256": sha256_file(output / "OUTPUT_MANIFEST.json"),
                    }, sort_keys=True, separators=(",", ":")) + "\n"
                    returned_argv = (*actual_argv, "--forged") if self.wrong_contract else actual_argv
                    return ProcessResult(returned_argv, 0, stdout, "", actual_env, cwd)

            staging = root / "staging"
            staging.mkdir()
            runner = FixtureRunner()
            receipt = _run_fresh_demo_inference(project, run_root, staging, runner)
            self.assertEqual(receipt["status"], "PASS")
            self.assertTrue(receipt["matches_stage60_predictions"])
            argv, cwd, environment = runner.calls[0]
            self.assertEqual(argv[:5], (sys.executable, "-I", "-B", "-m", "compag_curation"))
            self.assertEqual(cwd, staging)
            for key in (
                "HOME", "TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "MPLCONFIGDIR",
                "NUMBA_CACHE_DIR", "JOBLIB_TEMP_FOLDER", "CUDA_CACHE_PATH", "CUPY_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
            ):
                self.assertTrue(environment[key].startswith(str(staging / "fresh_process_runtime")))
            self.assertEqual(environment["MPLCONFIGDIR"], str(staging / "fresh_process_runtime"))
            self.assertFalse((staging / "fresh_process_runtime/matplotlib").exists())
            self.assertFalse(
                (staging / "fresh_process_runtime/9/0/c25eed8a829ad6").exists()
            )
            self.assertEqual(environment["PIP_CONFIG_FILE"], "/dev/null")
            self.assertEqual(environment["PYTHONPATH"], "")
            self.assertNotIn("LD_LIBRARY_PATH", environment)
            self.assertNotIn("SSL_CERT_FILE", environment)

            wrong_staging = root / "wrong-staging"
            wrong_staging.mkdir()
            with self.assertRaisesRegex(PublicIOError, "fresh-process inference contract failed"):
                _run_fresh_demo_inference(project, run_root, wrong_staging, FixtureRunner(wrong_contract=True))
            self.assertFalse(
                (
                    wrong_staging
                    / "fresh_process_runtime/9/0/c25eed8a829ad6"
                ).exists()
            )

            forged_staging = root / "forged-staging"
            forged_staging.mkdir()
            with self.assertRaisesRegex(PublicIOError, "differs from the completed run"):
                _run_fresh_demo_inference(project, run_root, forged_staging, FixtureRunner(forged_result=True))

            malformed_manifest_staging = root / "malformed-manifest-staging"
            malformed_manifest_staging.mkdir()
            with self.assertRaisesRegex(PublicIOError, "input manifest row closure"):
                _run_fresh_demo_inference(
                    project,
                    run_root,
                    malformed_manifest_staging,
                    FixtureRunner(malformed_input_manifest=True),
                )

    def test_canonical_stage60_reuses_nonempty_runtime_for_matplotlib_config(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "src/compag_curation/public_pipeline.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"MPLCONFIGDIR": str(runtime),', source)
        self.assertNotIn('"MPLCONFIGDIR": str(runtime / "matplotlib"),', source)

    def test_verified_descriptor_pins_checkpoint_bytes_and_rejects_path_swap(self) -> None:
        from compag_curation.public_io import PublicIOError, verified_file_path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            original = b"verified-checkpoint\n"
            checkpoint.write_bytes(original)
            digest = hashlib.sha256(original).hexdigest()
            with self.assertRaisesRegex(
                PublicIOError,
                r"verified file(?: path)? changed while in use",
            ):
                with verified_file_path(checkpoint, digest) as (pinned, _snapshot):
                    moved = root / "model.original.pt"
                    checkpoint.rename(moved)
                    checkpoint.write_bytes(b"substituted-checkpoint\n")
                    self.assertEqual(pinned.read_bytes(), original)

    def test_lock_initialization_failure_does_not_leak_a_file_descriptor(self) -> None:
        from compag_curation import run_state

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = len(list(Path("/proc/self/fd").iterdir()))
            with (
                mock.patch.object(run_state, "fsync_directory", side_effect=OSError("injected fsync failure")),
                self.assertRaisesRegex(OSError, "injected fsync failure"),
            ):
                run_state._open_lock(root, create=True)
            after = len(list(Path("/proc/self/fd").iterdir()))
            self.assertEqual(after, before)

    def test_manifest_directory_identity_failure_does_not_leak_a_file_descriptor(self) -> None:
        from compag_curation import public_io
        from compag_curation.public_io import PublicIOError, manifest_rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child"
            child.mkdir()
            (child / "member.txt").write_text("member\n", encoding="ascii")
            real_identity = public_io._stat_identity
            calls = 0

            def mismatched_child_identity(info: os.stat_result) -> tuple[int, ...]:
                nonlocal calls
                calls += 1
                identity = real_identity(info)
                if calls == 3:
                    return (identity[0] + 1, *identity[1:])
                return identity

            before = len(list(Path("/proc/self/fd").iterdir()))
            with (
                mock.patch.object(public_io, "_stat_identity", side_effect=mismatched_child_identity),
                self.assertRaisesRegex(PublicIOError, "directory changed while opening"),
            ):
                manifest_rows(root)
            after = len(list(Path("/proc/self/fd").iterdir()))
            self.assertEqual(after, before)

    def test_smote_plan_uses_an_explicit_integer_minority_target(self) -> None:
        from compag_curation.public_backend import _smote_plan

        self.assertEqual(_smote_plan(2, 5), (True, 0, 3))
        self.assertEqual(_smote_plan(3, 7), (True, 0, 4))
        self.assertEqual(_smote_plan(8, 3), (True, 1, 4))
        self.assertEqual(_smote_plan(4, 8), (False, 0, 4))

    def test_resume_disk_estimate_counts_only_unfinished_large_stages(self) -> None:
        from compag_curation import public_pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"x" * 1024)
            config = types.SimpleNamespace(checkpoint=checkpoint)
            inspection = {
                "images": [{"width": 512, "height": 512}] * 4,
                "inference_images": [{"width": 1024, "height": 1024}],
            }
            statvfs = types.SimpleNamespace(f_bavail=10**9, f_frsize=4096)
            with (
                mock.patch.object(public_pipeline, "inspect_public_data", return_value=inspection),
                mock.patch.object(public_pipeline.os, "statvfs", return_value=statvfs),
            ):
                full = public_pipeline._disk_preflight(config, root / "new-run")
                paused = public_pipeline._disk_preflight(
                    config,
                    root / "paused-run",
                    completed_stages=("00_input_inventory", "10_prepare", "20_proposals_features"),
                )
                trained = public_pipeline._disk_preflight(
                    config,
                    root / "trained-run",
                    completed_stages=(
                        "00_input_inventory", "10_prepare", "20_proposals_features",
                        "30_review_import", "40_group_split", "50_train_bundle",
                    ),
                )
            self.assertEqual(paused["estimate_components"]["training_tiles_overlays_and_tables"], 0)
            self.assertEqual(trained["estimate_components"]["checkpoint_bundle_and_staging"], 0)
            self.assertLess(paused["required_free_bytes"], full["required_free_bytes"])
            self.assertLess(trained["required_free_bytes"], paused["required_free_bytes"])

    def test_csv_outputs_are_0644_under_restrictive_umask(self) -> None:
        from compag_curation.public_backend import _write_csv
        from compag_curation.quick_demo import _write_csv_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = os.umask(0o077)
            try:
                backend = root / "backend.csv"
                demo = root / "demo.csv"
                _write_csv(backend, ("value",), ({"value": "backend"},))
                _write_csv_file(demo, ("value",), ({"value": "demo"},))
            finally:
                os.umask(previous)
            self.assertEqual(stat.S_IMODE(backend.stat().st_mode), 0o644)
            self.assertEqual(stat.S_IMODE(demo.stat().st_mode), 0o644)

    def test_sam_scores_and_geometry_proposal_ids_are_cross_process_stable(self) -> None:
        from compag_curation.public_backend import _canonical_sam_score, _format_feature_value, _proposal_id

        canonical_iou = _canonical_sam_score(0.98239409923553467, "predicted IoU")
        self.assertEqual(canonical_iou, _canonical_sam_score(0.98239398002624512, "predicted IoU"))
        self.assertEqual(_format_feature_value("pred_iou", canonical_iou), "0.982394")
        self.assertEqual(
            _canonical_sam_score(0.98454582691192627, "predicted IoU"),
            _canonical_sam_score(0.98454570770263672, "predicted IoU"),
        )
        identity = {
            "image_sha256": "1" * 64,
            "tile_sha256": "2" * 64,
            "tile_x": 0,
            "tile_y": 0,
            "mask_sha256": "3" * 64,
            "bbox": (2, 450, 62, 62),
            "polygon": [3, 450, 2, 452, 2, 509],
            "proposal_settings_sha256": "4" * 64,
        }
        proposal_id = _proposal_id(**identity)
        self.assertRegex(proposal_id, r"^[0-9a-f]{64}$")
        self.assertEqual(proposal_id, _proposal_id(**identity))
        changed_mask = dict(identity)
        changed_mask["mask_sha256"] = "5" * 64
        self.assertNotEqual(proposal_id, _proposal_id(**changed_mask))

    def test_science_cpu_runtime_policy_is_enforced_and_idempotent(self) -> None:
        from compag_curation import public_backend
        from compag_curation.public_io import PublicIOError
        from compag_curation.runtime_lock import science_cpu_runtime_policy

        class FakeTorch:
            intra = 12
            inter = 24
            deterministic = False
            inter_set_calls = 0

            @classmethod
            def set_num_threads(cls, value: int) -> None:
                cls.intra = value

            @classmethod
            def get_num_threads(cls) -> int:
                return cls.intra

            @classmethod
            def set_num_interop_threads(cls, value: int) -> None:
                cls.inter_set_calls += 1
                cls.inter = value

            @classmethod
            def get_num_interop_threads(cls) -> int:
                return cls.inter

            @classmethod
            def use_deterministic_algorithms(cls, enabled: bool) -> None:
                cls.deterministic = enabled

            @classmethod
            def are_deterministic_algorithms_enabled(cls) -> bool:
                return cls.deterministic

        class FakeOpenCL:
            enabled = True

            @classmethod
            def setUseOpenCL(cls, enabled: bool) -> None:
                cls.enabled = enabled

            @classmethod
            def useOpenCL(cls) -> bool:
                return cls.enabled

        class FakeCv2:
            threads = 8
            ocl = FakeOpenCL

            @classmethod
            def setNumThreads(cls, value: int) -> None:
                cls.threads = value

            @classmethod
            def getNumThreads(cls) -> int:
                return cls.threads

        class FakeThreadpoolctl:
            limit_calls = 0

            @classmethod
            def threadpool_limits(cls, *, limits: int) -> object:
                cls.limit_calls += 1
                self.assertEqual(limits, 1)
                return object()

            @staticmethod
            def threadpool_info() -> list[dict[str, object]]:
                return [
                    {"user_api": "blas", "num_threads": 1},
                    {"user_api": "openmp", "num_threads": 1},
                ]

        policy = science_cpu_runtime_policy()
        with (
            mock.patch.dict(
                os.environ,
                {"OMP_NUM_THREADS": "12", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "16"},
                clear=False,
            ),
            mock.patch.object(public_backend, "_SCIENCE_THREADPOOL_LIMITER", None),
        ):
            public_backend._set_science_cpu_environment(policy)
            first = public_backend._enforce_science_cpu_runtime(
                policy,
                torch_module=FakeTorch,
                cv2_module=FakeCv2,
                threadpoolctl_module=FakeThreadpoolctl,
            )
            second = public_backend._enforce_science_cpu_runtime(
                policy,
                torch_module=FakeTorch,
                cv2_module=FakeCv2,
                threadpoolctl_module=FakeThreadpoolctl,
            )
            self.assertEqual(first, policy)
            self.assertEqual(second, policy)
            self.assertEqual(FakeTorch.inter_set_calls, 1)
            self.assertEqual(FakeThreadpoolctl.limit_calls, 2)
            for name, value in policy["environment"].items():
                self.assertEqual(os.environ[name], value)

        class IneffectiveCv2(FakeCv2):
            threads = 8

            @classmethod
            def setNumThreads(cls, value: int) -> None:
                pass

        with self.assertRaisesRegex(PublicIOError, "OpenCV thread limit"):
            public_backend._enforce_science_cpu_runtime(
                policy,
                torch_module=FakeTorch,
                cv2_module=IneffectiveCv2,
                threadpoolctl_module=FakeThreadpoolctl,
            )

    def test_historical_v1_model_bundle_is_verify_only_and_semantically_closed(self) -> None:
        from compag_curation import model_bundle
        from compag_curation.model_bundle import BundleWriteRequest, verify_model_bundle, write_model_bundle
        from compag_curation.public_config import FEATURE_ORDER_SHA256, PUBLIC_FEATURE_ORDER
        from compag_curation.public_io import PublicIOError
        from compag_curation.runtime_lock import science_cpu_runtime_policy

        class FakeBooster:
            load_calls = 0

            def __init__(self) -> None:
                self.feature_names = list(PUBLIC_FEATURE_ORDER)

            def load_model(self, payload: bytearray) -> None:
                type(self).load_calls += 1
                if bytes(payload) != b"FAKE-XGBOOST-UBJ":
                    raise ValueError("invalid fixture classifier")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            config = root / "config.yaml"
            license_path = root / "LICENSE.txt"
            checkpoint.write_bytes(b"fixture-checkpoint\n")
            config.write_bytes(b"fixture-config\n")
            license_path.write_bytes(b"fixture-license\n")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
            license_sha = hashlib.sha256(license_path.read_bytes()).hexdigest()
            profile = model_bundle.PUBLIC_BUNDLE_PROFILE
            profile_record = {
                "locator": "configs/sam2.1/sam2.1_hiera_t",
                "checkpoint_sha256": checkpoint_sha,
                "config_sha256": config_sha,
            }
            normalized = {
                "schema": "compag-curation-normalized-public-config/v1",
                "profile": profile,
                "device": "cpu",
                "seed": 42,
                "checkpoint_sha256": checkpoint_sha,
                "sam2_config_sha256": config_sha,
                "feature_order_sha256": FEATURE_ORDER_SHA256,
            }
            preprocessing = {
                "schema": "compag-curation-preprocessing/v1",
                "tile_size": 512,
                "tile_overlap": 64,
                "image_color": "OpenCV BGR decoded then SAM2 RGB and OpenCV uint8 LAB features",
                "exif_orientation": "absent_or_1",
            }
            runtime_policy = science_cpu_runtime_policy()
            proposal = {
                "schema": "compag-curation-proposal-config/v2",
                "profile": profile,
                "sam2_config_locator": profile_record["locator"],
                "sam2_checkpoint_sha256": checkpoint_sha,
                "sam2_config_sha256": config_sha,
                "source_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
                "device": "cpu",
                "points_per_side": 8,
                "points_per_batch": 16,
                "pred_iou_threshold": 0.0,
                "stability_threshold": 0.0,
                "stability_score_offset": 1.0,
                "box_nms_threshold": 0.7,
                "crop_layers": 0,
                "crop_nms_threshold": 0.7,
                "crop_overlap_ratio": 512 / 1500,
                "crop_points_downscale": 1,
                "min_mask_region_area": 0,
                "multiscale": [1.0],
                "multiscale_iou": 0.75,
                "sam_score_decimal_places": 6,
                "proposal_identity": "geometry-mask-v2",
                "cpu_runtime_policy": runtime_policy,
                "max_proposals_per_tile": 24,
                "min_mask_area": 16,
                "feature_order_sha256": FEATURE_ORDER_SHA256,
            }
            compatibility = {
                "schema": "compag-curation-model-compatibility/v2",
                "python": model_bundle.SCIENCE_CPU_PYTHON,
                "versions": dict(model_bundle.SCIENCE_CPU_VERSIONS),
                "lock_sha256": model_bundle.SCIENCE_CPU_LOCK_SHA256,
                "cpu_runtime_policy_sha256": runtime_policy["sha256"],
                "feature_math": "gate-core-balanced-v1",
                "classifier": "xgboost-ubj",
            }
            provenance = {
                "schema": "compag-curation-model-provenance/v1",
                "profile": profile,
                "seed": 42,
                "reviewed_table_sha256": "0" * 64,
                "proposal_table_sha256": "1" * 64,
                "split_manifest_sha256": "2" * 64,
                "feature_order_sha256": FEATURE_ORDER_SHA256,
                "training_rows_before_smote": 8,
                "training_rows_after_smote": 8,
                "smote_applied": False,
                "validation_rows": 2,
                "test_rows": 2,
                "paper_result_reproduction": "NOT_CLAIMED",
            }

            def request(
                output: Path,
                proposal_value: dict[str, object],
                compatibility_value: dict[str, object] | None = None,
            ) -> BundleWriteRequest:
                return BundleWriteRequest(
                    output=output,
                    classifier_ubj=b"FAKE-XGBOOST-UBJ",
                    imputer_statistics={name: 0.0 for name in PUBLIC_FEATURE_ORDER},
                    threshold=0.5,
                    normalized_config=normalized,
                    preprocessing=preprocessing,
                    proposal_config=proposal_value,
                    compatibility=compatibility_value or compatibility,
                    provenance=provenance,
                    sam2_config=config,
                    sam2_checkpoint=checkpoint,
                    sam2_license=license_path,
                )

            from compag_curation import public_io

            def materialize_historical_bundle(output: Path) -> None:
                output.mkdir()
                for relative in ("sam2/configs/sam2.1", "sam2/checkpoints", "licenses"):
                    output.joinpath(*relative.split("/")).mkdir(parents=True)
                public_io.write_new_bytes(output / "classifier.ubj", b"FAKE-XGBOOST-UBJ")
                public_io.write_new_json(
                    output / "imputer.json",
                    {
                        "schema": "compag-curation-imputer/v1",
                        "strategy": "median",
                        "feature_order": list(PUBLIC_FEATURE_ORDER),
                        "statistics": {name: 0.0 for name in PUBLIC_FEATURE_ORDER},
                    },
                )
                public_io.write_new_json(
                    output / "feature_schema.json",
                    {
                        "schema": "compag-curation-feature-schema/v1",
                        "profile": profile,
                        "feature_order": list(PUBLIC_FEATURE_ORDER),
                        "feature_order_sha256": FEATURE_ORDER_SHA256,
                        "scaler": None,
                        "pca": None,
                        "embedding": None,
                    },
                )
                public_io.write_new_json(
                    output / "threshold.json",
                    {
                        "schema": "compag-curation-threshold/v1",
                        "selection_partition": "validation",
                        "value": 0.5,
                    },
                )
                public_io.write_new_json(
                    output / "class_map.json",
                    {"schema": "compag-curation-class-map/v1", "negative": 0, "positive": 1},
                )
                for relative, value in (
                    ("normalized_config.json", normalized),
                    ("preprocessing.json", preprocessing),
                    ("proposal_config.json", proposal),
                    ("compatibility.json", compatibility),
                    ("provenance.json", provenance),
                ):
                    public_io.write_new_json(output / relative, value)
                config_copy = public_io.copy_new(
                    config,
                    output / "sam2/configs/sam2.1/model.yaml",
                )
                checkpoint_copy = public_io.copy_new(
                    checkpoint,
                    output / "sam2/checkpoints/model.pt",
                )
                license_copy = public_io.copy_new(
                    license_path,
                    output / "licenses/SAM2-APACHE-2.0.txt",
                )
                rows = public_io.manifest_rows(output, excluded={"bundle.json"})
                public_io.write_new_json(
                    output / "bundle.json",
                    {
                        "schema": model_bundle.BUNDLE_SCHEMA,
                        "status": "PASS",
                        "profile": profile,
                        "feature_order_sha256": FEATURE_ORDER_SHA256,
                        "classifier_format": "XGBOOST_UBJ",
                        "checkpoint": checkpoint_copy,
                        "sam2_config": config_copy,
                        "sam2_license": license_copy,
                        "members": rows,
                    },
                )

            fake_xgboost = types.SimpleNamespace(Booster=FakeBooster)
            with (
                mock.patch.dict(sys.modules, {"xgboost": fake_xgboost}),
                mock.patch.dict(model_bundle.BUNDLE_ASSET_PROFILES, {profile: profile_record}, clear=True),
                mock.patch.object(model_bundle, "SAM2_LICENSE_SHA256", license_sha),
            ):
                rejected = root / "rejected-v1-writer"
                with self.assertRaisesRegex(PublicIOError, "historical read/inspect/verify-only"):
                    write_model_bundle(request(rejected, proposal))
                self.assertFalse(rejected.exists())
                self.assertEqual(list(root.glob(".rejected-v1-writer.bundle.*")), [])

                final = root / "historical-model-bundle"
                materialize_historical_bundle(final)
                verified = verify_model_bundle(final)

                bundle_manifest = final / "bundle.json"
                original_manifest = bundle_manifest.read_bytes()
                FakeBooster.load_calls = 0
                bundle_manifest.write_bytes(b'{"schema":"duplicate",' + original_manifest[1:])
                with self.assertRaisesRegex(PublicIOError, "not strict JSON"):
                    verify_model_bundle(final)
                self.assertEqual(FakeBooster.load_calls, 0)
                bundle_manifest.write_bytes(b'{"nonfinite":NaN,' + original_manifest[1:])
                with self.assertRaisesRegex(PublicIOError, "not strict JSON"):
                    verify_model_bundle(final)
                self.assertEqual(FakeBooster.load_calls, 0)
                bundle_manifest.write_bytes(original_manifest)
                self.assertEqual(verify_model_bundle(final).bundle_sha256, verified.bundle_sha256)

                unexpected = final / "000-oversized-unknown.bin"
                with unexpected.open("wb") as handle:
                    handle.truncate(model_bundle.MAX_BUNDLE_JSON_BYTES + 1)
                with self.assertRaisesRegex(PublicIOError, "unexpected file"):
                    verify_model_bundle(final)
                unexpected.rename(root / "preserved-oversized-unknown.bin")

                (final / "unexpected-empty-directory").mkdir()
                with self.assertRaisesRegex(PublicIOError, "undeclared empty directory"):
                    verify_model_bundle(final)


if __name__ == "__main__":
    unittest.main()
