from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from compag_curation.canonical import service
from compag_curation.canonical.spec import (
    CANONICAL_CPU_PROFILE,
    CANONICAL_FEATURE_CROP_SCALES,
    CANONICAL_GPU_PROFILE,
    CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
    CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256,
    CANONICAL_SAM2_HIERA_L_CONFIG_SHA256,
    CANONICAL_STAGE20_AMG_POINTS_PER_BATCH,
)
from compag_curation.public_config import RESNET50_WEIGHTS_SHA256
from compag_curation.public_io import PublicIOError


def _config(profile: str, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        profile=profile,
        is_canonical=True,
        device=device,
        seed=42,
        tile_size=512,
        tile_stride=512,
        tile_overlap=0,
        checkpoint=Path("/assets/sam2.1_hiera_large.pt"),
        checkpoint_sha256=CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256,
        sam2_config=Path("/assets/sam2.1_hiera_l.yaml"),
        sam2_config_sha256=CANONICAL_SAM2_HIERA_L_CONFIG_SHA256,
        sam2_config_locator="configs/sam2.1/sam2.1_hiera_l",
        embedding_weights=Path("/assets/resnet50-11ad3fa6.pth"),
        embedding_weights_sha256=RESNET50_WEIGHTS_SHA256,
        yolo_enabled=False,
        proposal_scales=(1.0,),
        feature_crop_scales=CANONICAL_FEATURE_CROP_SCALES,
        inference_threshold=0.5,
        nms_iou_threshold=0.5,
    )


class CanonicalServiceGpuProfileTests(unittest.TestCase):
    def test_gpu_profile_requires_cuda_and_builds_cuda_training_config(self) -> None:
        config = _config(CANONICAL_GPU_PROFILE, "cuda")

        service._canonical_config(config)
        with mock.patch.object(service.os, "cpu_count", return_value=24):
            training = service._canonical_training_config(config)

        self.assertEqual(training.device, "cuda")
        self.assertEqual(training.thread_count, 1)
        with self.assertRaisesRegex(PublicIOError, "profile/device/seed"):
            service._canonical_config(_config(CANONICAL_GPU_PROFILE, "cpu"))

    def test_gpu_bundle_config_maps_retain_profile_and_execution_device(self) -> None:
        config = _config(CANONICAL_GPU_PROFILE, "cuda")

        normalized, preprocessing, proposal, features, training = (
            service._bundle_config_maps(config)
        )

        self.assertEqual(normalized["device"], "cuda")
        for record in (normalized, preprocessing, proposal, features, training):
            self.assertEqual(record["profile"], CANONICAL_GPU_PROFILE)
        self.assertEqual(proposal["points_per_batch"], 512)
        self.assertNotIn("sam2_execution_points_per_batch", proposal)
        self.assertNotIn("device", proposal)
        self.assertNotIn("device", features)
        self.assertNotIn("device", training)

    def test_runtime_loader_forwards_explicit_cuda_to_sam2_and_resnet(self) -> None:
        assets = service.CanonicalRuntimeAssets(
            sam2_config=Path("sam.yaml"),
            sam2_checkpoint=Path("sam.pt"),
            sam2_config_locator="configs/sam2.1/sam2.1_hiera_l",
            sam2_config_sha256=CANONICAL_SAM2_HIERA_L_CONFIG_SHA256,
            sam2_checkpoint_sha256=CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256,
            resnet50_weights=Path("resnet.pth"),
            resnet50_weights_sha256=RESNET50_WEIGHTS_SHA256,
        )
        model = object()
        with (
            mock.patch.object(
                service,
                "load_canonical_sam2_model",
                return_value=model,
            ) as sam2_loader,
            mock.patch.object(service, "build_canonical_amg", return_value="amg"),
            mock.patch.object(
                service.CanonicalResNet50Embedder,
                "from_local_weights",
                return_value="embedder",
            ) as resnet_loader,
            mock.patch.object(
                service,
                "_attest_canonical_stage_runtime_cuda",
            ) as runtime_attestation,
        ):
            runtime = service.load_canonical_runtime(assets, device="cuda")

        self.assertEqual(runtime.amg, "amg")
        self.assertEqual(runtime.embedder, "embedder")
        self.assertEqual(sam2_loader.call_args.kwargs["device"], "cuda")
        self.assertEqual(resnet_loader.call_args.kwargs["device"], "cuda")
        runtime_attestation.assert_called_once_with(runtime)

    def test_runtime_loader_accepts_only_sealed_amg_execution_microbatches(self) -> None:
        assets = service.CanonicalRuntimeAssets(
            sam2_config=Path("sam.yaml"),
            sam2_checkpoint=Path("sam.pt"),
            sam2_config_locator="configs/sam2.1/sam2.1_hiera_l",
            sam2_config_sha256=CANONICAL_SAM2_HIERA_L_CONFIG_SHA256,
            sam2_checkpoint_sha256=CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256,
            resnet50_weights=Path("resnet.pth"),
            resnet50_weights_sha256=RESNET50_WEIGHTS_SHA256,
        )
        with (
            mock.patch.object(
                service,
                "load_canonical_sam2_model",
                return_value=mock.sentinel.model,
            ) as sam2_loader,
            mock.patch.object(
                service,
                "build_canonical_amg",
                return_value=mock.sentinel.amg,
            ) as amg_builder,
            mock.patch.object(
                service.CanonicalResNet50Embedder,
                "from_local_weights",
                return_value=mock.sentinel.embedder,
            ),
            mock.patch.object(service, "_attest_canonical_stage_runtime_cuda"),
        ):
            for microbatch in (
                CANONICAL_STAGE20_AMG_POINTS_PER_BATCH,
                CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
            ):
                with self.subTest(microbatch=microbatch):
                    service.load_canonical_runtime(
                        assets,
                        device="cuda",
                        amg_execution_points_per_batch=microbatch,
                    )
                    self.assertEqual(
                        amg_builder.call_args.kwargs["execution_points_per_batch"],
                        microbatch,
                    )
            for unsupported in (True, 64, 128, 1024):
                with self.subTest(unsupported=unsupported), self.assertRaisesRegex(
                    PublicIOError,
                    "microbatch",
                ):
                    service.load_canonical_runtime(
                        assets,
                        device="cuda",
                        amg_execution_points_per_batch=unsupported,
                    )

        self.assertEqual(sam2_loader.call_count, 2)

    def test_production_dependency_loaders_forward_cuda_without_fallback(self) -> None:
        assets = mock.sentinel.assets
        runtime = mock.sentinel.runtime
        predictor = mock.sentinel.predictor
        with (
            mock.patch.object(
                service,
                "load_canonical_runtime",
                return_value=runtime,
            ) as runtime_loader,
            mock.patch.object(
                service,
                "load_portable_predictor",
                return_value=predictor,
            ) as predictor_loader,
        ):
            observed_runtime = service._load_stage_runtime(assets, "cuda", None)
            observed_predictor = service._load_bundle_predictor(
                b"ubj",
                (0.0,),
                "cuda",
                None,
            )

        self.assertIs(observed_runtime, runtime)
        self.assertIs(observed_predictor, predictor)
        runtime_loader.assert_called_once_with(assets, device="cuda")
        predictor_loader.assert_called_once_with(b"ubj", (0.0,), device="cuda")

    def test_explicit_execution_microbatch_is_forwarded_to_runtime_loader(self) -> None:
        assets = mock.sentinel.assets
        runtime = mock.sentinel.runtime
        with mock.patch.object(
            service,
            "load_canonical_runtime",
            return_value=runtime,
        ) as runtime_loader:
            observed = service._load_stage_runtime(
                assets,
                "cuda",
                None,
                amg_execution_points_per_batch=CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
            )

        self.assertIs(observed, runtime)
        runtime_loader.assert_called_once_with(
            assets,
            device="cuda",
            amg_execution_points_per_batch=CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
        )

    def test_stage20_runtime_helper_seals_execution_only_microbatch(self) -> None:
        runtime = mock.sentinel.runtime
        with mock.patch.object(
            service,
            "_load_stage_runtime",
            return_value=runtime,
        ) as stage_loader:
            observed = service._load_stage20_runtime(
                mock.sentinel.assets,
                "cuda",
                None,
            )

        self.assertIs(observed, runtime)
        stage_loader.assert_called_once_with(
            mock.sentinel.assets,
            "cuda",
            None,
            amg_execution_points_per_batch=CANONICAL_STAGE20_AMG_POINTS_PER_BATCH,
        )

    def test_inference_runtime_helper_seals_the_vram_safe_microbatch(self) -> None:
        runtime = mock.sentinel.runtime
        with mock.patch.object(
            service,
            "_load_stage_runtime",
            return_value=runtime,
        ) as stage_loader:
            observed = service._load_inference_runtime(
                mock.sentinel.assets,
                "cuda",
                None,
            )

        self.assertIs(observed, runtime)
        stage_loader.assert_called_once_with(
            mock.sentinel.assets,
            "cuda",
            None,
            amg_execution_points_per_batch=CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
        )

    def test_cpu_production_loaders_reject_before_model_or_predictor_work(self) -> None:
        assets = mock.sentinel.assets
        with (
            mock.patch.object(
                service,
                "load_canonical_sam2_model",
                side_effect=AssertionError("SAM2 loader must not run"),
            ) as sam2_loader,
            mock.patch.object(
                service,
                "load_portable_predictor",
                side_effect=AssertionError("predictor loader must not run"),
            ) as predictor_loader,
        ):
            with self.assertRaisesRegex(PublicIOError, "CUDA-only.*CPU"):
                service.load_canonical_runtime(assets, device="cpu")
            with self.assertRaisesRegex(PublicIOError, "CUDA-only.*CPU"):
                service._load_stage_runtime(assets, "cpu", None)
            with self.assertRaisesRegex(PublicIOError, "CUDA-only.*CPU"):
                service._load_bundle_predictor(b"ubj", (), "cpu", None)
        sam2_loader.assert_not_called()
        predictor_loader.assert_not_called()

    def test_public_infer_and_execution_stages_reject_cpu_before_filesystem_work(self) -> None:
        cpu = _config(CANONICAL_CPU_PROFILE, "cpu")
        with mock.patch.object(
            service,
            "_infer_canonical_bundle_with_dependencies",
            side_effect=AssertionError("inference harness must not run"),
        ) as inference:
            with self.assertRaisesRegex(PublicIOError, "CUDA-only.*CPU"):
                service.infer_canonical_bundle(
                    Path("missing-images"),
                    Path("missing-bundle"),
                    Path("missing-output"),
                    device="cpu",
                )
        inference.assert_not_called()
        with self.assertRaisesRegex(PublicIOError, "CUDA-only.*CPU"):
            service.generate_canonical_stage(
                cpu,
                Path("missing-prepared"),
                Path("missing-output"),
            )
        with self.assertRaisesRegex(PublicIOError, "CUDA-only.*CPU"):
            service.train_canonical_stage(
                cpu,
                Path("missing-proposals"),
                Path("missing-reviews"),
                Path("missing-split"),
                Path("missing-output"),
            )

    def test_tile_generation_reattests_amg_after_upstream_generate(self) -> None:
        import numpy as np
        import sys
        import types

        amg = mock.Mock()
        amg.generate.return_value = []
        runtime = service.CanonicalStageRuntime(
            amg=amg,
            embedder=mock.sentinel.embedder,
        )
        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.COLOR_BGR2RGB = object()
        fake_cv2.cvtColor = lambda value, _conversion: value
        with (
            mock.patch.dict(sys.modules, {"cv2": fake_cv2}),
            mock.patch.object(
                service,
                "_attest_canonical_amg_cuda",
                side_effect=(0, RuntimeError("post-generate CUDA drift")),
            ) as attestation,
        ):
            with self.assertRaisesRegex(RuntimeError, "post-generate CUDA drift"):
                service._generate_tile_records(
                    runtime,
                    np.zeros((2, 2, 3), dtype=np.uint8),
                    {},
                    {},
                    attest_cuda=True,
                )
        amg.generate.assert_called_once()
        self.assertEqual(attestation.call_count, 2)


if __name__ == "__main__":
    unittest.main()
