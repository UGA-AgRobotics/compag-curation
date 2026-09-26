from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from compag_curation.canonical import active_learning
from compag_curation.canonical import active_learning_facade
from compag_curation.canonical import active_learning_service
from compag_curation.canonical.service import CANONICAL_RAW_FEATURE_TABLE_COLUMNS
from compag_curation.canonical.spec import (
    CANONICAL_CPU_PROFILE,
    CANONICAL_FEATURE_CROP_SCALES,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_GPU_PROFILE,
    EFFICIENT_GPU_PROFILE,
    EFFICIENT_INFERENCE_AMG_POINTS_PER_BATCH,
    FULL_IMAGE_GPU_PROFILE,
    FULL_IMAGE_INFERENCE_AMG_POINTS_PER_BATCH,
)
from compag_curation.model_bundle import BUNDLE_SCHEMA_V2
from compag_curation.public_io import (
    PublicIOError,
    canonical_json_bytes,
    compact_json_sha256,
)
from compag_curation import public_pipeline
from compag_curation.public_config import initialize_project, load_public_config


class DualProfileExecutionGateTests(unittest.TestCase):
    @staticmethod
    def _request(profile: str, device: str, provenance: dict[str, object]):
        return active_learning.CanonicalActiveLearningRetrainRequest(
            round_number=1,
            feature_state=mock.sentinel.state,
            effective_decisions=(),
            training_rows=(),
            train_groups=frozenset(),
            test_groups=frozenset(),
            decision_log_sha256="a" * 64,
            training_rows_sha256="b" * 64,
            source_bundle_sha256="c" * 64,
            frozen_split_sha256="d" * 64,
            prototype_sha256="e" * 64,
            pca_components_sha256="f" * 64,
            pca_mean_sha256="0" * 64,
            expected_bundle_provenance=provenance,
            profile=profile,
            device=device,
        )

    @staticmethod
    def _empty_lite_inference_result() -> dict[str, object]:
        bundle_sha256 = "a" * 64
        predictions_sha256 = "b" * 64
        return {
            "schema": "compag-curation-canonical-inference/v1",
            "status": "PASS",
            "profile": EFFICIENT_GPU_PROFILE,
            "device": "cuda",
            "bundle_sha256": bundle_sha256,
            "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
            "threshold": 0.5,
            "threshold_method": "FIXED_CANONICAL_METHOD",
            "full_image_nms_iou": 0.5,
            "sam2_execution_points_per_batch": (
                EFFICIENT_INFERENCE_AMG_POINTS_PER_BATCH
            ),
            "image_count": 0,
            "tile_count": 0,
            "proposal_count": 0,
            "prediction_rows": 0,
            "positive_rows": 0,
            "kept_rows": 0,
            "predictions_sha256": predictions_sha256,
            "raw_feature_archive": {
                "schema": (
                    active_learning_service.CANONICAL_AL_RAW_FEATURE_ARCHIVE_SCHEMA
                ),
                "status": "PASS",
                "profile": EFFICIENT_GPU_PROFILE,
                "basename": "raw_features.csv",
                "columns_sha256": compact_json_sha256(
                    list(CANONICAL_RAW_FEATURE_TABLE_COLUMNS)
                ),
                "crop_scales": list(CANONICAL_FEATURE_CROP_SCALES),
                "rows_per_proposal": len(CANONICAL_FEATURE_CROP_SCALES),
                "proposal_count": 0,
                "row_count": 0,
                "bundle_sha256": bundle_sha256,
                "predictions_sha256": predictions_sha256,
                "sha256": "c" * 64,
                "size_bytes": 1,
            },
            "input_inventory": [],
            "paper_result_reproduction": "NOT_CLAIMED",
        }

    def test_v2_read_profiles_and_gpu_execution_profiles_are_distinct(self) -> None:
        self.assertEqual(
            active_learning._canonical_device_for_profile(CANONICAL_CPU_PROFILE),
            "cpu",
        )
        self.assertEqual(
            active_learning._canonical_device_for_profile(CANONICAL_GPU_PROFILE),
            "cuda",
        )
        self.assertEqual(
            active_learning._canonical_device_for_profile(EFFICIENT_GPU_PROFILE),
            "cuda",
        )
        self.assertEqual(
            active_learning._canonical_device_for_profile(FULL_IMAGE_GPU_PROFILE),
            "cuda",
        )
        with self.assertRaisesRegex(PublicIOError, "requires device 'cuda'"):
            active_learning._canonical_profile_device(EFFICIENT_GPU_PROFILE, "cpu")

    def test_lite_retrain_identity_is_mandatory_and_profile_bound(self) -> None:
        with self.assertRaisesRegex(PublicIOError, "provenance profile/device"):
            self._request(EFFICIENT_GPU_PROFILE, "cuda", {})
        with self.assertRaisesRegex(PublicIOError, "provenance profile/device"):
            self._request(
                EFFICIENT_GPU_PROFILE,
                "cuda",
                {"profile": CANONICAL_GPU_PROFILE, "device": "cuda"},
            )
        request = self._request(
            EFFICIENT_GPU_PROFILE,
            "cuda",
            {"profile": EFFICIENT_GPU_PROFILE, "device": "cuda"},
        )
        self.assertEqual(request.profile, EFFICIENT_GPU_PROFILE)

    def test_lite_inference_microbatch_is_profile_specific(self) -> None:
        result = self._empty_lite_inference_result()
        validated = public_pipeline._validated_canonical_inference_result(
            result,
            bundle_sha256="a" * 64,
            expected_inputs=(),
            expected_profile=EFFICIENT_GPU_PROFILE,
            expected_device="cuda",
        )
        self.assertEqual(
            validated["sam2_execution_points_per_batch"],
            EFFICIENT_INFERENCE_AMG_POINTS_PER_BATCH,
        )
        changed = dict(result)
        changed["sam2_execution_points_per_batch"] = 32
        with self.assertRaisesRegex(PublicIOError, "method or field closure"):
            public_pipeline._validated_canonical_inference_result(
                changed,
                bundle_sha256="a" * 64,
                expected_inputs=(),
                expected_profile=EFFICIENT_GPU_PROFILE,
                expected_device="cuda",
            )

    def test_active_learning_archive_parser_uses_lite_microbatch(self) -> None:
        predictions_payload = b"fixture-predictions\n"
        feature_payload = b"fixture-features\n"
        image_name = "tilegroup_001.png"
        image_sha256 = "d" * 64
        group_id, image_id = active_learning_service._canonical_image_identity(
            image_name,
            image_sha256,
        )
        result = self._empty_lite_inference_result()
        result.update(
            image_count=1,
            tile_count=1,
            predictions_sha256=hashlib.sha256(predictions_payload).hexdigest(),
            input_inventory=[
                {
                    "image_name": image_name,
                    "image_id": image_id,
                    "image_sha256": image_sha256,
                    "group_id": group_id,
                    "size_bytes": 1,
                    "width": 64,
                    "height": 64,
                    "tile_count": 1,
                    "candidate_rows": 0,
                }
            ],
        )
        result["raw_feature_archive"] = active_learning_service._archive_receipt(
            feature_payload,
            bundle_sha256="a" * 64,
            predictions_sha256=str(result["predictions_sha256"]),
            proposal_count=0,
            row_count=0,
            profile=EFFICIENT_GPU_PROFILE,
        )
        parsed = active_learning_service._parse_inference_result(
            canonical_json_bytes(result),
            predictions_payload=predictions_payload,
            feature_payload=feature_payload,
            prediction_rows=(),
            feature_row_count=0,
        )
        self.assertEqual(parsed, result)
        changed = dict(result)
        changed["sam2_execution_points_per_batch"] = 32
        with self.assertRaisesRegex(PublicIOError, "method identity"):
            active_learning_service._parse_inference_result(
                canonical_json_bytes(changed),
                predictions_payload=predictions_payload,
                feature_payload=feature_payload,
                prediction_rows=(),
                feature_row_count=0,
            )

    def test_full_image_archive_accepts_one_processing_unit_per_image_only(self) -> None:
        predictions_payload = b"fixture-predictions\n"
        feature_payload = b"fixture-features\n"
        image_name = "whole-image.png"
        image_sha256 = "d" * 64
        group_id, image_id = active_learning_service._canonical_image_identity(
            image_name,
            image_sha256,
        )
        result = self._empty_lite_inference_result()
        result.update(
            profile=FULL_IMAGE_GPU_PROFILE,
            sam2_execution_points_per_batch=(
                FULL_IMAGE_INFERENCE_AMG_POINTS_PER_BATCH
            ),
            image_count=1,
            tile_count=1,
            predictions_sha256=hashlib.sha256(predictions_payload).hexdigest(),
            input_inventory=[
                {
                    "image_name": image_name,
                    "image_id": image_id,
                    "image_sha256": image_sha256,
                    "group_id": group_id,
                    "size_bytes": 1,
                    "width": 4032,
                    "height": 3024,
                    "tile_count": 1,
                    "candidate_rows": 0,
                }
            ],
        )
        result["raw_feature_archive"] = active_learning_service._archive_receipt(
            feature_payload,
            bundle_sha256="a" * 64,
            predictions_sha256=str(result["predictions_sha256"]),
            proposal_count=0,
            row_count=0,
            profile=FULL_IMAGE_GPU_PROFILE,
        )
        self.assertEqual(
            active_learning_service._parse_inference_result(
                canonical_json_bytes(result),
                predictions_payload=predictions_payload,
                feature_payload=feature_payload,
                prediction_rows=(),
                feature_row_count=0,
            ),
            result,
        )
        public_inputs = (
            {
                "filename": image_name,
                "sha256": image_sha256,
                "size_bytes": 1,
                "width": 4032,
                "height": 3024,
                "exif_orientation": None,
            },
        )
        self.assertEqual(
            public_pipeline._validated_canonical_inference_result(
                result,
                bundle_sha256="a" * 64,
                expected_inputs=public_inputs,
                expected_profile=FULL_IMAGE_GPU_PROFILE,
                expected_device="cuda",
            ),
            result,
        )
        forged = {
            **result,
            "tile_count": 2,
            "input_inventory": [
                {**result["input_inventory"][0], "tile_count": 2}
            ],
        }
        with self.assertRaisesRegex(PublicIOError, "compatibility counter"):
            active_learning_service._parse_inference_result(
                canonical_json_bytes(forged),
                predictions_payload=predictions_payload,
                feature_payload=feature_payload,
                prediction_rows=(),
                feature_row_count=0,
            )
        with self.assertRaisesRegex(PublicIOError, "input-inventory row"):
            public_pipeline._validated_canonical_inference_result(
                forged,
                bundle_sha256="a" * 64,
                expected_inputs=public_inputs,
                expected_profile=FULL_IMAGE_GPU_PROFILE,
                expected_device="cuda",
            )

    def test_stage_zero_adds_full_image_execution_identity_only_for_full_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executions: dict[str, dict[str, object]] = {}
            for profile in (CANONICAL_GPU_PROFILE, FULL_IMAGE_GPU_PROFILE):
                project = root / profile
                initialize_project(project, profile=profile)
                config = load_public_config(project / "config.toml")
                output = root / f"stage-zero-{profile}"
                output.mkdir()
                preflight = {
                    "bindings": {
                        "config_sha256": config.config_sha256,
                        "input_manifest_sha256": "a" * 64,
                        "dependency_sha256": "b" * 64,
                        "asset_sha256": "c" * 64,
                    },
                    "inspection": {},
                    "annotations": {},
                    "assets": [],
                    "dependencies": {},
                    "disk_preflight": {},
                }
                public_pipeline._stage_zero(config, preflight, {}, output)
                normalized = json.loads(
                    (output / "normalized_config.json").read_text(
                        encoding="ascii"
                    )
                )
                executions[profile] = normalized["execution"]
            identity_fields = {
                "spatial_mode",
                "execution_points_per_batch",
                "max_proposals_per_image",
            }
            self.assertTrue(
                identity_fields.isdisjoint(executions[CANONICAL_GPU_PROFILE])
            )
            self.assertEqual(
                {
                    key: executions[FULL_IMAGE_GPU_PROFILE][key]
                    for key in identity_fields
                },
                {
                    "spatial_mode": "full-image-multiscale",
                    "execution_points_per_batch": 8,
                    "max_proposals_per_image": 1000,
                },
            )

    def test_published_r92_reader_rejects_full_image_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = {
                "schema": active_learning_service.R92_TRANSFER_INFERENCE_SCHEMA,
                "status": "PASS",
                "workflow": active_learning_service.R92_TRANSFER_INFERENCE_WORKFLOW,
                "lineage_policy": active_learning_service.R92_TRANSFER_LINEAGE_POLICY,
                "reproduction_claim": active_learning_service.R92_TRANSFER_REPRODUCTION_CLAIM,
                "preset_id": "compag-cj-r92",
                "profile": FULL_IMAGE_GPU_PROFILE,
                "device": "cuda",
                "threshold": 0.5,
                "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
                "resource_manifest_sha256": "1" * 64,
                "classifier_sha256": "2" * 64,
                "feature_state_sha256": "3" * 64,
                "scorer_identity_sha256": "4" * 64,
                "config_sha256": "5" * 64,
                "asset_inventory": [],
                "asset_inventory_sha256": "6" * 64,
                "image_inventory_sha256": "7" * 64,
                "image": {},
                "predictions_sha256": "8" * 64,
                "raw_features_sha256": "9" * 64,
                "inference_result_sha256": "a" * 64,
            }
            (root / "transfer_scorer.json").write_bytes(
                canonical_json_bytes(receipt)
            )
            inference = SimpleNamespace(
                profile=FULL_IMAGE_GPU_PROFILE,
                device="cuda",
                bundle_sha256="4" * 64,
                predictions_sha256="8" * 64,
                raw_features_sha256="9" * 64,
                inference_result_sha256="a" * 64,
            )
            with mock.patch.object(
                active_learning_service,
                "read_canonical_active_learning_inference",
                return_value=inference,
            ), self.assertRaisesRegex(PublicIOError, "does not bind"):
                active_learning_service.read_canonical_r92_transfer_inference(
                    root
                )

    def test_public_runtime_and_facade_accept_all_gpu_profiles_only(self) -> None:
        for profile in (
            CANONICAL_GPU_PROFILE,
            EFFICIENT_GPU_PROFILE,
            FULL_IMAGE_GPU_PROFILE,
        ):
            public_pipeline._require_gpu_executable_profile(
                SimpleNamespace(profile=profile, device="cuda")
            )
            bundle = SimpleNamespace(schema=BUNDLE_SCHEMA_V2, profile=profile)
            with mock.patch.object(
                active_learning_facade,
                "verify_model_bundle",
                return_value=bundle,
            ):
                self.assertIs(
                    active_learning_facade._verified_gpu_execution_bundle(
                        mock.sentinel.root
                    ),
                    bundle,
                )
        with self.assertRaisesRegex(PublicIOError, "historical verification-only"):
            public_pipeline._require_gpu_executable_profile(
                SimpleNamespace(profile=CANONICAL_CPU_PROFILE, device="cpu")
            )

    def test_real_demo_forwards_selected_execution_profile(self) -> None:
        with mock.patch(
            "compag_curation.quick_demo.run_real_demo",
            return_value={"status": "PASS"},
        ) as run_real:
            result, code = public_pipeline.run_demo(
                "real",
                Path("out"),
                Path("assets"),
                execution_profile=EFFICIENT_GPU_PROFILE,
            )
        self.assertEqual((result, code), ({"status": "PASS"}, 0))
        run_real.assert_called_once_with(
            Path("out"),
            Path("assets"),
            execution_profile=EFFICIENT_GPU_PROFILE,
            invocation=None,
        )


if __name__ == "__main__":
    unittest.main()
