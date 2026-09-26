from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from compag_curation.canonical import model_loading, service
from compag_curation.canonical.spec import (
    CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256,
    EFFICIENT_GPU_PROFILE,
    EFFICIENT_SAM2_HIERA_T_CHECKPOINT_SHA256,
    EFFICIENT_SAM2_HIERA_T_CONFIG_SHA256,
)
from compag_curation.public_config import RESNET50_WEIGHTS_SHA256


def _lite_config() -> SimpleNamespace:
    return SimpleNamespace(
        profile=EFFICIENT_GPU_PROFILE,
        device="cuda",
        seed=42,
        tile_size=512,
        tile_stride=512,
        tile_overlap=0,
        checkpoint=Path("/assets/sam2.1_hiera_tiny.pt"),
        checkpoint_sha256=EFFICIENT_SAM2_HIERA_T_CHECKPOINT_SHA256,
        sam2_config=Path("/assets/sam2.1_hiera_t.yaml"),
        sam2_config_locator="configs/sam2.1/sam2.1_hiera_t",
        sam2_config_sha256=EFFICIENT_SAM2_HIERA_T_CONFIG_SHA256,
        embedding_weights=Path("/assets/resnet50.pth"),
        embedding_weights_sha256=RESNET50_WEIGHTS_SHA256,
        yolo_enabled=False,
        proposal_scales=(1.0,),
        feature_crop_scales=(0.67, 0.8, 1.0, 1.25),
        inference_threshold=0.5,
        nms_iou_threshold=0.5,
    )


class DualProfileRuntimeTests(unittest.TestCase):
    def test_lite_config_is_cuda_and_bound_to_tiny(self) -> None:
        config = _lite_config()
        service._canonical_config(config)
        self.assertEqual(service._canonical_device(config.profile), "cuda")
        assets = service._runtime_assets_from_config(config)
        self.assertEqual(assets.profile, EFFICIENT_GPU_PROFILE)
        self.assertEqual(
            assets.sam2_config_locator, "configs/sam2.1/sam2.1_hiera_t"
        )

        config.checkpoint_sha256 = CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256
        with self.assertRaisesRegex(Exception, "checkpoint identity"):
            service._canonical_config(config)

    def test_lite_runtime_uses_tiny_identity_and_microbatch_16(self) -> None:
        assets = service.CanonicalRuntimeAssets(
            sam2_config=Path("/assets/sam2.1_hiera_t.yaml"),
            sam2_checkpoint=Path("/assets/sam2.1_hiera_tiny.pt"),
            sam2_config_locator="configs/sam2.1/sam2.1_hiera_t",
            sam2_config_sha256=EFFICIENT_SAM2_HIERA_T_CONFIG_SHA256,
            sam2_checkpoint_sha256=EFFICIENT_SAM2_HIERA_T_CHECKPOINT_SHA256,
            resnet50_weights=Path("/assets/resnet50.pth"),
            resnet50_weights_sha256=RESNET50_WEIGHTS_SHA256,
            profile=EFFICIENT_GPU_PROFILE,
        )
        embedder = mock.sentinel.embedder
        with (
            mock.patch.object(service, "_require_cuda_service_device", return_value="cuda"),
            mock.patch.object(
                service, "load_canonical_sam2_model", return_value=mock.sentinel.model
            ) as load_model,
            mock.patch.object(
                service, "build_canonical_amg", return_value=mock.sentinel.amg
            ) as build_amg,
            mock.patch.object(
                service.CanonicalResNet50Embedder,
                "from_local_weights",
                return_value=embedder,
            ),
            mock.patch.object(service, "_attest_canonical_stage_runtime_cuda"),
        ):
            runtime = service.load_canonical_runtime(
                assets, amg_execution_points_per_batch=16
            )
        self.assertIs(runtime.amg, mock.sentinel.amg)
        self.assertEqual(load_model.call_args.kwargs["profile"], EFFICIENT_GPU_PROFILE)
        self.assertEqual(build_amg.call_args.kwargs["profile"], EFFICIENT_GPU_PROFILE)
        self.assertEqual(build_amg.call_args.kwargs["execution_points_per_batch"], 16)
        with self.assertRaisesRegex(Exception, "microbatch"):
            service.load_canonical_runtime(
                assets, amg_execution_points_per_batch=32
            )

    def test_lite_bundle_runtime_assets_are_tiny_and_profile_bound(self) -> None:
        bundle = SimpleNamespace(
            profile=EFFICIENT_GPU_PROFILE,
            sam2_config=Path("/bundle/sam2.1_hiera_t.yaml"),
            checkpoint=Path("/bundle/sam2.1_hiera_tiny.pt"),
            resnet50_weights=Path("/bundle/resnet50.pth"),
        )
        assets = service._runtime_assets_from_bundle(bundle)
        self.assertEqual(assets.profile, EFFICIENT_GPU_PROFILE)
        self.assertEqual(
            assets.sam2_checkpoint_sha256,
            EFFICIENT_SAM2_HIERA_T_CHECKPOINT_SHA256,
        )

    def test_model_loader_rejects_full_assets_for_lite_before_sam2_import(self) -> None:
        with mock.patch.object(model_loading, "_resolve_torch_device", return_value="cuda"):
            with self.assertRaisesRegex(ValueError, "does not match the profile"):
                model_loading.load_canonical_sam2_model(
                    Path("/assets/sam2.1_hiera_t.yaml"),
                    Path("/assets/sam2.1_hiera_large.pt"),
                    "configs/sam2.1/sam2.1_hiera_t",
                    EFFICIENT_SAM2_HIERA_T_CONFIG_SHA256,
                    CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256,
                    profile=EFFICIENT_GPU_PROFILE,
                )


if __name__ == "__main__":
    unittest.main()
