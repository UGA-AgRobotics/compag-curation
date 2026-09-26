from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from compag_curation.canonical import active_learning
from compag_curation.canonical import active_learning_facade as facade
from compag_curation.canonical import active_learning_service as service
from compag_curation.canonical.spec import (
    CANONICAL_CPU_PROFILE,
    CANONICAL_GPU_PROFILE,
)
from compag_curation.model_bundle import BUNDLE_SCHEMA_V2
from compag_curation.public_io import PublicIOError, write_new_json


class CanonicalActiveLearningGpuProfileTests(unittest.TestCase):
    @staticmethod
    def _retrain_record(*, schema: str, profile: str, device: str | None) -> dict[str, object]:
        record: dict[str, object] = {
            "schema": schema,
            "profile": profile,
            "round_number": 1,
            "source_bundle_sha256": "1" * 64,
            "frozen_split_sha256": "2" * 64,
            "decision_log_sha256": "3" * 64,
            "training_rows_sha256": "4" * 64,
            "pool_feature_archive_sha256": "5" * 64,
            "effective_proposals": 12,
            "feature_rows": 48,
            "train_groups_sha256": "6" * 64,
            "test_groups_sha256": "7" * 64,
            "feature_state_training_rows": 48,
            "feature_state_positive_rows": 24,
            "prototype_sha256": "8" * 64,
            "pca_components_sha256": "9" * 64,
            "pca_mean_sha256": "a" * 64,
            "pca_policy": "FROZEN_FROM_PARENT_BUNDLE",
            "prototype_policy": "RECOMPUTED_ACCUMULATED_OUTER_TRAIN_POSITIVES",
        }
        if device is not None:
            record["device"] = device
        return record

    def test_profile_device_pairs_are_fail_closed(self) -> None:
        self.assertEqual(
            active_learning._canonical_device_for_profile(CANONICAL_CPU_PROFILE),
            "cpu",
        )
        self.assertEqual(
            active_learning._canonical_device_for_profile(CANONICAL_GPU_PROFILE),
            "cuda",
        )
        with self.assertRaisesRegex(PublicIOError, "requires device 'cuda'"):
            active_learning._canonical_profile_device(
                CANONICAL_GPU_PROFILE,
                "cpu",
            )
        with self.assertRaisesRegex(PublicIOError, "supported canonical profile"):
            active_learning._canonical_device_for_profile("canonical-unknown")

    def test_gpu_retrain_request_requires_profiled_provenance(self) -> None:
        fields = {
            "round_number": 1,
            "feature_state": mock.sentinel.state,
            "effective_decisions": (),
            "training_rows": (),
            "train_groups": frozenset(),
            "test_groups": frozenset(),
            "decision_log_sha256": "a" * 64,
            "training_rows_sha256": "b" * 64,
            "source_bundle_sha256": "c" * 64,
            "frozen_split_sha256": "d" * 64,
            "prototype_sha256": "e" * 64,
            "pca_components_sha256": "f" * 64,
            "pca_mean_sha256": "0" * 64,
            "profile": CANONICAL_GPU_PROFILE,
            "device": "cuda",
        }
        with self.assertRaisesRegex(PublicIOError, "provenance profile/device"):
            active_learning.CanonicalActiveLearningRetrainRequest(
                **fields,
                expected_bundle_provenance={},
            )
        request = active_learning.CanonicalActiveLearningRetrainRequest(
            **fields,
            expected_bundle_provenance={
                "profile": CANONICAL_GPU_PROFILE,
                "device": "cuda",
            },
        )
        self.assertEqual((request.profile, request.device), (CANONICAL_GPU_PROFILE, "cuda"))

    def test_retrain_request_reader_accepts_cpu_v1_and_paired_v2_only(self) -> None:
        v1 = "compag-curation-canonical-active-learning-retrain-request/v1"
        v2 = "compag-curation-canonical-active-learning-retrain-request/v2"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cpu_path = root / "cpu-v1.json"
            gpu_path = root / "gpu-v2.json"
            write_new_json(
                cpu_path,
                self._retrain_record(
                    schema=v1,
                    profile=CANONICAL_CPU_PROFILE,
                    device=None,
                ),
            )
            write_new_json(
                gpu_path,
                self._retrain_record(
                    schema=v2,
                    profile=CANONICAL_GPU_PROFILE,
                    device="cuda",
                ),
            )
            self.assertEqual(
                active_learning._retrain_request_manifest(cpu_path)[0]["profile"],
                CANONICAL_CPU_PROFILE,
            )
            self.assertEqual(
                active_learning._retrain_request_manifest(gpu_path)[0]["device"],
                "cuda",
            )

            invalid_v1 = root / "invalid-gpu-v1.json"
            invalid_pair = root / "invalid-gpu-pair.json"
            write_new_json(
                invalid_v1,
                self._retrain_record(
                    schema=v1,
                    profile=CANONICAL_GPU_PROFILE,
                    device=None,
                ),
            )
            write_new_json(
                invalid_pair,
                self._retrain_record(
                    schema=v2,
                    profile=CANONICAL_GPU_PROFILE,
                    device="cpu",
                ),
            )
            with self.assertRaisesRegex(PublicIOError, "exact CPU v1"):
                active_learning._retrain_request_manifest(invalid_v1)
            with self.assertRaisesRegex(PublicIOError, "requires device 'cuda'"):
                active_learning._retrain_request_manifest(invalid_pair)

    def test_gpu_rescore_passes_cuda_to_portable_predictor(self) -> None:
        row = active_learning.CanonicalActiveLearningPoolRow(
            proposal_id="a" * 64,
            proposal_sha256="a" * 64,
            image_id="b" * 64,
            image_sha256="c" * 64,
            group_id="g0",
            scale=1.0,
            xgb_p=0.5,
        )
        bundle = SimpleNamespace(
            profile=CANONICAL_GPU_PROFILE,
            classifier=b"fixture",
            imputer=(0.0,),
        )
        with mock.patch(
            "compag_curation.canonical.serialization.load_portable_predictor",
            return_value=object(),
        ) as loader, mock.patch(
            "compag_curation.canonical.serialization.predict_portable_probabilities",
            return_value=[0.25],
        ):
            probabilities = active_learning._predict_rescored_probabilities(
                bundle,
                (row,),
                ({},),
                device="cuda",
            )
        self.assertEqual(probabilities, (0.25,))
        loader.assert_called_once_with(
            bundle.classifier,
            bundle.imputer,
            device="cuda",
        )

    def test_inference_archive_and_receipt_preserve_gpu_identity(self) -> None:
        archive = service.CanonicalActiveLearningInferenceArchive(
            pool_rows=(),
            feature_rows=(),
            bundle_sha256="a" * 64,
            predictions_sha256="b" * 64,
            raw_features_sha256="c" * 64,
            feature_archive_sha256="d" * 64,
            inference_result_sha256="e" * 64,
            profile=CANONICAL_GPU_PROFILE,
            device="cuda",
        )
        self.assertEqual((archive.profile, archive.device), (CANONICAL_GPU_PROFILE, "cuda"))
        receipt = service._archive_receipt(
            b"features\n",
            bundle_sha256="a" * 64,
            predictions_sha256="b" * 64,
            proposal_count=1,
            row_count=4,
            profile=CANONICAL_GPU_PROFILE,
        )
        self.assertEqual(receipt["profile"], CANONICAL_GPU_PROFILE)
        with self.assertRaisesRegex(PublicIOError, "requires device 'cuda'"):
            service.CanonicalActiveLearningInferenceArchive(
                pool_rows=(),
                feature_rows=(),
                bundle_sha256="a" * 64,
                predictions_sha256="b" * 64,
                raw_features_sha256="c" * 64,
                feature_archive_sha256="d" * 64,
                inference_result_sha256="e" * 64,
                profile=CANONICAL_GPU_PROFILE,
                device="cpu",
            )

    def test_facade_records_science_gpu_and_accepts_verified_gpu_bundle(self) -> None:
        self.assertEqual(facade._science_device_for_mode("gpu"), "cuda")
        dependencies = {
            "schema": "test-science-gpu/v1",
            "sha256": "f" * 64,
        }
        with mock.patch.object(
            facade,
            "package_implementation_identity",
            return_value={"sha256": "e" * 64, "size_bytes": 1},
        ):
            artifacts = facade._environment_artifacts(
                dependencies,
                science_mode="gpu",
            )
        self.assertEqual(artifacts[-1]["basename"], "science-gpu")
        bundle = SimpleNamespace(
            schema=BUNDLE_SCHEMA_V2,
            profile=CANONICAL_GPU_PROFILE,
        )
        with mock.patch.object(facade, "verify_model_bundle", return_value=bundle):
            self.assertIs(facade._verified_canonical_bundle(mock.sentinel.root), bundle)
            self.assertIs(
                facade._verified_gpu_execution_bundle(mock.sentinel.root),
                bundle,
            )

    def test_historical_cpu_bundle_is_structurally_readable_but_not_executable(self) -> None:
        bundle = SimpleNamespace(
            schema=BUNDLE_SCHEMA_V2,
            profile=CANONICAL_CPU_PROFILE,
        )
        with mock.patch.object(
            facade,
            "verify_model_bundle",
            return_value=bundle,
        ), mock.patch.object(
            facade,
            "require_science_dependencies",
            side_effect=AssertionError("historical CPU inspection invoked live preflight"),
        ) as preflight:
            self.assertIs(
                facade._verified_canonical_bundle(mock.sentinel.root),
                bundle,
            )
            with self.assertRaisesRegex(PublicIOError, "requires a canonical GPU/CUDA bundle"):
                facade._verified_gpu_execution_bundle(mock.sentinel.root)
        preflight.assert_not_called()

        cpu_attempt = SimpleNamespace(
            science_dependencies={"schema": "historical-science-cpu/v1"},
            science_mode="cpu",
        )
        with self.assertRaisesRegex(PublicIOError, "require science-gpu"):
            facade._dependencies_for_sealing(cpu_attempt)

    def test_gpu_sealing_revalidates_captured_preflight_without_reinitializing(self) -> None:
        dependencies = {"schema": "test-science-gpu/v1", "sha256": "f" * 64}
        attempt = SimpleNamespace(
            science_dependencies=dependencies,
            science_mode="gpu",
        )
        revalidated = dict(dependencies)
        with mock.patch.object(
            facade,
            "revalidate_science_gpu_dependencies",
            return_value=revalidated,
        ) as revalidation, mock.patch.object(
            facade,
            "require_science_dependencies",
            side_effect=AssertionError("GPU preflight was repeated after CUDA init"),
        ) as preflight:
            self.assertIs(facade._dependencies_for_sealing(attempt), revalidated)
        revalidation.assert_called_once_with("cuda", dependencies)
        preflight.assert_not_called()

    def test_gpu_sealing_rejects_revalidated_receipt_drift(self) -> None:
        dependencies = {"schema": "test-science-gpu/v1", "sha256": "f" * 64}
        attempt = SimpleNamespace(
            science_dependencies=dependencies,
            science_mode="gpu",
        )
        drifted = {**dependencies, "sha256": "e" * 64}
        with mock.patch.object(
            facade,
            "revalidate_science_gpu_dependencies",
            return_value=drifted,
        ), mock.patch.object(
            facade,
            "require_science_dependencies",
            side_effect=AssertionError("GPU full preflight must not repeat"),
        ) as preflight:
            with self.assertRaisesRegex(
                PublicIOError,
                "dependency identity changed during operation sealing",
            ):
                facade._dependencies_for_sealing(attempt)
        preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
