from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from compag_curation import model_bundle
from compag_curation.canonical import service
from compag_curation.canonical.spec import (
    CANONICAL_EMBEDDING_DIMENSIONS,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_GPU_PROFILE,
    CANONICAL_PCA_DIMENSIONS,
    CANONICAL_PROFILES,
    EFFICIENT_GPU_PROFILE,
    EFFICIENT_SAM2_HIERA_T_CHECKPOINT_SHA256,
    EFFICIENT_SAM2_HIERA_T_CONFIG_SHA256,
    FULL_IMAGE_GPU_PROFILE,
)
from compag_curation.canonical.full_image_features import (
    FULL_IMAGE_FEATURE_SEMANTICS_ID,
)
from compag_curation.model_bundle import (
    BUNDLE_V2_ASSET_PROFILES,
    BUNDLE_V2_FILES,
    BUNDLE_V2_FILES_BY_PROFILE,
    BundleV2WriteRequest,
    encode_float32_npy,
    verify_model_bundle,
    write_model_bundle_v2,
)
from compag_curation.public_config import RESNET50_WEIGHTS_SHA256
from compag_curation.public_io import PublicIOError, canonical_json_bytes


class ModelBundleDualProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.payloads = {
            "tiny.yaml": b"fixture-hiera-t-config\n",
            "tiny.pt": b"fixture-hiera-t-checkpoint\n",
            "sam-license.txt": b"fixture-sam-license\n",
            "resnet50.pth": b"fixture-resnet50-weights\n",
            "resnet-license.txt": b"fixture-resnet-license\n",
        }
        for name, payload in self.payloads.items():
            (self.inputs / name).write_bytes(payload)

        def digest(name: str) -> str:
            return hashlib.sha256(self.payloads[name]).hexdigest()

        self.lite_asset_profile = {
            "sam2_architecture": "sam2.1_hiera_tiny",
            "sam2_config_locator": "configs/sam2.1/sam2.1_hiera_t",
            "sam2_config_bundle_path": "sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
            "sam2_checkpoint_bundle_path": "sam2/checkpoints/sam2.1_hiera_tiny.pt",
            "sam2_checkpoint_sha256": digest("tiny.pt"),
            "sam2_config_sha256": digest("tiny.yaml"),
            "sam2_license_sha256": digest("sam-license.txt"),
            "resnet50_architecture": "torchvision.models.resnet50",
            "resnet50_weights_identity": "ResNet50_Weights.IMAGENET1K_V2",
            "resnet50_weights_sha256": digest("resnet50.pth"),
            "resnet50_license_sha256": digest("resnet-license.txt"),
        }
        self.full_image_asset_profile = {
            **self.lite_asset_profile,
            "sam2_architecture": "sam2.1_hiera_large",
            "sam2_config_locator": "configs/sam2.1/sam2.1_hiera_l",
            "sam2_config_bundle_path": "sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
            "sam2_checkpoint_bundle_path": "sam2/checkpoints/sam2.1_hiera_large.pt",
        }
        self.prototype_npy = encode_float32_npy(
            (1.0, *(0.0 for _ in range(CANONICAL_EMBEDDING_DIMENSIONS - 1))),
            shape=(CANONICAL_EMBEDDING_DIMENSIONS,),
            role="fixture prototype",
        )
        self.pca_mean_npy = encode_float32_npy(
            (0.0,) * CANONICAL_EMBEDDING_DIMENSIONS,
            shape=(CANONICAL_EMBEDDING_DIMENSIONS,),
            role="fixture PCA mean",
        )
        self.pca_components_npy = encode_float32_npy(
            tuple(
                1.0 if row == column else 0.0
                for row in range(CANONICAL_PCA_DIMENSIONS)
                for column in range(CANONICAL_EMBEDDING_DIMENSIONS)
            ),
            shape=(CANONICAL_PCA_DIMENSIONS, CANONICAL_EMBEDDING_DIMENSIONS),
            role="fixture PCA components",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _request(self, output: str = "lite-bundle") -> BundleV2WriteRequest:
        profile = EFFICIENT_GPU_PROFILE
        configs = model_bundle._canonical_config_contracts(
            profile=profile,
            asset_profile=self.lite_asset_profile,
            feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
        )
        return BundleV2WriteRequest(
            output=self.root / output,
            profile=profile,
            classifier_ubj=b"{Lfixture-canonical-ubj",
            feature_order=CANONICAL_FEATURE_ORDER,
            feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
            imputer_statistics={name: 0.0 for name in CANONICAL_FEATURE_ORDER},
            prototype_npy=self.prototype_npy,
            pca_mean_npy=self.pca_mean_npy,
            pca_components_npy=self.pca_components_npy,
            pca_explained_variance=(1.0,) * CANONICAL_PCA_DIMENSIONS,
            normalized_config=configs["normalized"],
            preprocessing=configs["preprocessing"],
            proposal_config=configs["proposal"],
            feature_config=configs["features"],
            training_config=configs["training"],
            compatibility={
                "classifier_format": "XGBOOST_UBJ",
                "no_pickle_or_joblib": True,
                "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
                "fixed_threshold": 0.5,
            },
            provenance={
                "profile": profile,
                "device": "cuda",
                "reviewed_sha256": "a" * 64,
                "split_manifest_sha256": "b" * 64,
                "raw_features_sha256": "c" * 64,
                "feature_state_fit_group_sha256": "d" * 64,
                "feature_state_fit_rows_sha256": "e" * 64,
                "feature_state_fit_scope": "OUTER_TRAIN_ROWS_BEFORE_CV",
                "validation_feature_scale": 1.0,
                "cv_score_interpretation": "CONDITIONAL_ON_OUTER_TRAIN_PREFIT_PROTOTYPE_AND_PCA",
                "training_row_count": 32,
                "positive_training_row_count": 16,
                "fixed_threshold": 0.5,
                "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
                "ubj_parity_status": "PASS",
                "paper_result_reproduction": "NOT_CLAIMED",
            },
            sam2_config=self.inputs / "tiny.yaml",
            sam2_checkpoint=self.inputs / "tiny.pt",
            sam2_license=self.inputs / "sam-license.txt",
            resnet50_weights=self.inputs / "resnet50.pth",
            resnet50_license=self.inputs / "resnet-license.txt",
        )

    def _full_image_request(
        self,
        output: str = "full-image-bundle",
    ) -> BundleV2WriteRequest:
        profile = FULL_IMAGE_GPU_PROFILE
        configs = model_bundle._canonical_config_contracts(
            profile=profile,
            asset_profile=self.full_image_asset_profile,
            feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
        )
        base = self._request(output)
        provenance = dict(base.provenance)
        provenance.update(profile=profile, device="cuda")
        return replace(
            base,
            profile=profile,
            normalized_config=configs["normalized"],
            preprocessing=configs["preprocessing"],
            proposal_config=configs["proposal"],
            feature_config=configs["features"],
            training_config=configs["training"],
            provenance=provenance,
        )

    def _dependencies(self) -> ExitStack:
        full_asset_profile = dict(BUNDLE_V2_ASSET_PROFILES[CANONICAL_GPU_PROFILE])
        stack = ExitStack()
        stack.enter_context(
            mock.patch.dict(
                BUNDLE_V2_ASSET_PROFILES,
                {
                    CANONICAL_GPU_PROFILE: full_asset_profile,
                    EFFICIENT_GPU_PROFILE: self.lite_asset_profile,
                    FULL_IMAGE_GPU_PROFILE: self.full_image_asset_profile,
                },
                clear=True,
            )
        )
        stack.enter_context(
            mock.patch.object(model_bundle, "_validate_canonical_ubj", return_value=None)
        )
        stack.enter_context(
            mock.patch.object(
                model_bundle,
                "package_implementation_identity",
                return_value={
                    "schema": "compag-curation-package-implementation-identity/v1",
                    "sha256": "f" * 64,
                    "file_count": 1,
                    "size_bytes": 1,
                },
            )
        )
        return stack

    def test_profile_sets_keep_full_canonical_and_select_exact_sam_paths(self) -> None:
        self.assertNotIn(EFFICIENT_GPU_PROFILE, CANONICAL_PROFILES)
        self.assertNotIn(FULL_IMAGE_GPU_PROFILE, CANONICAL_PROFILES)
        self.assertIs(BUNDLE_V2_FILES_BY_PROFILE[CANONICAL_GPU_PROFILE], BUNDLE_V2_FILES)
        self.assertIs(BUNDLE_V2_FILES_BY_PROFILE[FULL_IMAGE_GPU_PROFILE], BUNDLE_V2_FILES)
        self.assertIn(
            "sam2/checkpoints/sam2.1_hiera_large.pt",
            BUNDLE_V2_FILES_BY_PROFILE[CANONICAL_GPU_PROFILE],
        )
        self.assertNotIn(
            "sam2/checkpoints/sam2.1_hiera_tiny.pt",
            BUNDLE_V2_FILES_BY_PROFILE[CANONICAL_GPU_PROFILE],
        )
        self.assertIn(
            "sam2/checkpoints/sam2.1_hiera_tiny.pt",
            BUNDLE_V2_FILES_BY_PROFILE[EFFICIENT_GPU_PROFILE],
        )
        self.assertNotIn(
            "sam2/checkpoints/sam2.1_hiera_large.pt",
            BUNDLE_V2_FILES_BY_PROFILE[EFFICIENT_GPU_PROFILE],
        )
        self.assertEqual(
            BUNDLE_V2_ASSET_PROFILES[FULL_IMAGE_GPU_PROFILE],
            BUNDLE_V2_ASSET_PROFILES[CANONICAL_GPU_PROFILE],
        )

    def _reseal_json_member(self, bundle: Path, relative: str, value: object) -> None:
        payload = canonical_json_bytes(value)
        (bundle / relative).write_bytes(payload)
        manifest_path = bundle / "bundle.json"
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        for record in manifest["members"]:
            if record["path"] == relative:
                record["sha256"] = hashlib.sha256(payload).hexdigest()
                record["size_bytes"] = len(payload)
                break
        else:  # pragma: no cover - malformed fixture guard
            self.fail(f"missing fixture manifest member: {relative}")
        manifest_path.write_bytes(canonical_json_bytes(manifest))

    def test_full_image_write_verify_and_spatial_semantics_fail_closed(self) -> None:
        request = self._full_image_request()
        with self._dependencies():
            result = write_model_bundle_v2(request)
            verified = verify_model_bundle(request.output)

            self.assertEqual(result["profile"], FULL_IMAGE_GPU_PROFILE)
            self.assertEqual(verified.profile, FULL_IMAGE_GPU_PROFILE)
            self.assertEqual(
                verified.sam2_config.relative_to(request.output).as_posix(),
                "sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
            )
            self.assertEqual(
                verified.checkpoint.relative_to(request.output).as_posix(),
                "sam2/checkpoints/sam2.1_hiera_large.pt",
            )
            self.assertEqual(
                verified.normalized_config["spatial_mode"],
                "full-image-multiscale",
            )
            self.assertEqual(
                verified.feature_config["feature_semantics_id"],
                FULL_IMAGE_FEATURE_SEMANTICS_ID,
            )
            self.assertFalse(verified.preprocessing["external_tiling"])
            self.assertEqual(
                verified.proposal_config["feature_semantics_id"],
                FULL_IMAGE_FEATURE_SEMANTICS_ID,
            )

            tampered = self.root / "full-image-spatial-tampered"
            shutil.copytree(request.output, tampered)
            feature_path = tampered / "feature_config.json"
            feature = json.loads(feature_path.read_text(encoding="ascii"))
            feature["config"]["spatial_mode"] = "tiled"
            self._reseal_json_member(tampered, "feature_config.json", feature)
            with self.assertRaisesRegex(PublicIOError, "features config"):
                verify_model_bundle(tampered)

        tiled_configs = model_bundle._canonical_config_contracts(
            profile=CANONICAL_GPU_PROFILE,
            asset_profile=self.full_image_asset_profile,
            feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
        )
        self.assertNotIn("spatial_mode", tiled_configs["normalized"])
        self.assertNotIn("feature_semantics_id", tiled_configs["features"])
        mismatched = replace(
            self._full_image_request("full-image-with-tiled-config"),
            normalized_config=tiled_configs["normalized"],
        )
        with self._dependencies(), self.assertRaisesRegex(
            PublicIOError, "normalized config"
        ):
            write_model_bundle_v2(mismatched)
        self.assertFalse(mismatched.output.exists())

        wrong_features = dict(
            self._full_image_request("full-image-wrong-semantics").feature_config
        )
        wrong_features["feature_semantics_id"] = "compag-tiled-ultra-features-v1"
        mismatched = replace(
            self._full_image_request("full-image-wrong-semantics"),
            feature_config=wrong_features,
        )
        with self._dependencies(), self.assertRaisesRegex(
            PublicIOError, "features config"
        ):
            write_model_bundle_v2(mismatched)
        self.assertFalse(mismatched.output.exists())

    def test_lite_write_verify_and_profile_tamper_fail_closed(self) -> None:
        request = self._request()
        with self._dependencies():
            result = write_model_bundle_v2(request)
            verified = verify_model_bundle(request.output)

            self.assertEqual(result["profile"], EFFICIENT_GPU_PROFILE)
            self.assertEqual(verified.profile, EFFICIENT_GPU_PROFILE)
            self.assertEqual(
                verified.sam2_config.relative_to(request.output).as_posix(),
                "sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
            )
            self.assertEqual(
                verified.checkpoint.relative_to(request.output).as_posix(),
                "sam2/checkpoints/sam2.1_hiera_tiny.pt",
            )
            self.assertEqual(verified.proposal_config["architecture"], "sam2.1_hiera_tiny")
            self.assertEqual(
                verified.proposal_config["config_locator"],
                "configs/sam2.1/sam2.1_hiera_t",
            )

            tampered = self.root / "profile-tampered"
            shutil.copytree(request.output, tampered)
            manifest_path = tampered / "bundle.json"
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            manifest["profile"] = CANONICAL_GPU_PROFILE
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(
                PublicIOError, "payload|closure|not allowed|unexpected file"
            ):
                verify_model_bundle(tampered)

    def test_lite_writer_rejects_full_proposal_identity(self) -> None:
        request = self._request("wrong-proposal")
        proposal = dict(request.proposal_config)
        proposal.update(
            architecture="sam2.1_hiera_large",
            config_locator="configs/sam2.1/sam2.1_hiera_l",
        )
        request = replace(request, proposal_config=proposal)
        with self._dependencies(), self.assertRaisesRegex(
            PublicIOError, "proposal config"
        ):
            write_model_bundle_v2(request)
        self.assertFalse(request.output.exists())

    def test_stage50_lite_config_maps_are_accepted_by_the_bundle_writer(self) -> None:
        config = SimpleNamespace(
            profile=EFFICIENT_GPU_PROFILE,
            device="cuda",
            seed=42,
            tile_size=512,
            tile_stride=512,
            tile_overlap=0,
            checkpoint=Path("/assets/sam2.1_hiera_tiny.pt"),
            checkpoint_sha256=EFFICIENT_SAM2_HIERA_T_CHECKPOINT_SHA256,
            sam2_config=Path("/assets/sam2.1_hiera_t.yaml"),
            sam2_config_sha256=EFFICIENT_SAM2_HIERA_T_CONFIG_SHA256,
            sam2_config_locator="configs/sam2.1/sam2.1_hiera_t",
            embedding_weights=Path("/assets/resnet50.pth"),
            embedding_weights_sha256=RESNET50_WEIGHTS_SHA256,
            yolo_enabled=False,
            proposal_scales=(1.0,),
            feature_crop_scales=(0.67, 0.8, 1.0, 1.25),
            inference_threshold=0.5,
            nms_iou_threshold=0.5,
        )
        maps = service._bundle_config_maps(config)
        request = replace(
            self._request("stage50-lite-contract"),
            normalized_config=maps[0],
            preprocessing=maps[1],
            proposal_config=maps[2],
            feature_config=maps[3],
            training_config=maps[4],
        )
        with (
            mock.patch.object(model_bundle, "_validate_canonical_ubj"),
            mock.patch.object(
                model_bundle,
                "package_implementation_identity",
                return_value={
                    "schema": "compag-curation-package-implementation-identity/v1",
                    "sha256": "f" * 64,
                    "file_count": 1,
                    "size_bytes": 1,
                },
            ),
            mock.patch.object(
                model_bundle,
                "_write_model_bundle_v2_staging",
                return_value={"status": "PASS"},
            ) as staging,
        ):
            self.assertEqual(write_model_bundle_v2(request), {"status": "PASS"})
        staging.assert_called_once()


if __name__ == "__main__":
    unittest.main()
