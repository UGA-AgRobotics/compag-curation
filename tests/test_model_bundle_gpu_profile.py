from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from unittest import mock

from compag_curation import model_bundle
from compag_curation.canonical.spec import (
    CANONICAL_CPU_PROFILE,
    CANONICAL_EMBEDDING_DIMENSIONS,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_GPU_PROFILE,
    CANONICAL_PCA_DIMENSIONS,
    EFFICIENT_GPU_PROFILE,
    FULL_IMAGE_GPU_PROFILE,
)
from compag_curation.model_bundle import (
    BUNDLE_V2_ASSET_PROFILES,
    BundleV2WriteRequest,
    HISTORICAL_V1_2_PACKAGE_IMPLEMENTATION_IDENTITY,
    encode_float32_npy,
    verify_model_bundle,
    write_model_bundle_v2,
)
from compag_curation.public_io import PublicIOError, canonical_json_bytes
from compag_curation.runtime_lock import (
    SCIENCE_CPU_LOCK_SHA256,
    SCIENCE_CPU_PYTHON,
    SCIENCE_CPU_VERSIONS,
    SCIENCE_GPU_LOCK_SHA256,
    SCIENCE_GPU_PYTHON,
    SCIENCE_GPU_VERSIONS,
    science_cpu_runtime_policy,
    science_gpu_runtime_policy,
)


class ModelBundleGPUProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        payloads = {
            "hiera.yaml": b"fixture-hiera-l-config\n",
            "hiera.pt": b"fixture-hiera-l-checkpoint\n",
            "sam-license.txt": b"fixture-sam-license\n",
            "resnet50.pth": b"fixture-resnet50-weights\n",
            "resnet-license.txt": b"fixture-resnet-license\n",
        }
        for name, payload in payloads.items():
            (self.inputs / name).write_bytes(payload)

        def digest(name: str) -> str:
            return hashlib.sha256(payloads[name]).hexdigest()

        self.asset_profile = {
            "sam2_architecture": "sam2.1_hiera_large",
            "sam2_config_locator": "configs/sam2.1/sam2.1_hiera_l",
            "sam2_checkpoint_sha256": digest("hiera.pt"),
            "sam2_config_sha256": digest("hiera.yaml"),
            "sam2_license_sha256": digest("sam-license.txt"),
            "resnet50_architecture": "torchvision.models.resnet50",
            "resnet50_weights_identity": "ResNet50_Weights.IMAGENET1K_V2",
            "resnet50_weights_sha256": digest("resnet50.pth"),
            "resnet50_license_sha256": digest("resnet-license.txt"),
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
                1.0 if column == row else 0.0
                for row in range(CANONICAL_PCA_DIMENSIONS)
                for column in range(CANONICAL_EMBEDDING_DIMENSIONS)
            ),
            shape=(CANONICAL_PCA_DIMENSIONS, CANONICAL_EMBEDDING_DIMENSIONS),
            role="fixture PCA components",
        )
        self.identity_calls: list[str] = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _request(
        self,
        profile: str,
        output: str,
        *,
        include_provenance_profile: bool = True,
    ) -> BundleV2WriteRequest:
        device = {
            CANONICAL_CPU_PROFILE: "cpu",
            CANONICAL_GPU_PROFILE: "cuda",
        }[profile]
        configs = model_bundle._canonical_config_contracts(
            profile=profile,
            asset_profile=self.asset_profile,
            feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
        )
        provenance: dict[str, object] = {
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
        }
        if include_provenance_profile:
            provenance.update({"profile": profile, "device": device})
        return BundleV2WriteRequest(
            output=self.root / output,
            profile=profile,
            classifier_ubj=b"{Lcanonical-test-ubj",
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
            provenance=provenance,
            sam2_config=self.inputs / "hiera.yaml",
            sam2_checkpoint=self.inputs / "hiera.pt",
            sam2_license=self.inputs / "sam-license.txt",
            resnet50_weights=self.inputs / "resnet50.pth",
            resnet50_license=self.inputs / "resnet-license.txt",
        )

    @contextmanager
    def _bundle_dependencies(self):
        identity = {
            "schema": "compag-curation-package-implementation-identity/v1",
            "sha256": "f" * 64,
            "file_count": 1,
            "size_bytes": 1,
        }

        def package_identity(**kwargs: object) -> dict[str, object]:
            self.identity_calls.append(str(kwargs.get("expected_distribution_version")))
            return dict(identity)

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(
                    BUNDLE_V2_ASSET_PROFILES,
                    {
                        CANONICAL_CPU_PROFILE: self.asset_profile,
                        CANONICAL_GPU_PROFILE: self.asset_profile,
                    },
                    clear=True,
                )
            )
            stack.enter_context(
                mock.patch.object(model_bundle, "_validate_canonical_ubj", return_value=object())
            )
            stack.enter_context(
                mock.patch.object(
                    model_bundle,
                    "package_implementation_identity",
                    side_effect=package_identity,
                )
            )
            yield

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
            self.fail(f"missing manifest member: {relative}")
        manifest_path.write_bytes(canonical_json_bytes(manifest))

    def _convert_gpu_bundle_to_historical_cpu(
        self,
        bundle: Path,
        *,
        profiled_provenance: bool,
    ) -> None:
        feature_schema = json.loads(
            (bundle / "feature_schema.json").read_text(encoding="ascii")
        )
        feature_schema["profile"] = CANONICAL_CPU_PROFILE
        self._reseal_json_member(bundle, "feature_schema.json", feature_schema)
        for relative in (
            "normalized_config.json",
            "preprocessing.json",
            "proposal_config.json",
            "feature_config.json",
            "training_config.json",
        ):
            wrapper = json.loads((bundle / relative).read_text(encoding="ascii"))
            wrapper["profile"] = CANONICAL_CPU_PROFILE
            wrapper["config"]["profile"] = CANONICAL_CPU_PROFILE
            if relative == "normalized_config.json":
                wrapper["config"]["device"] = "cpu"
            self._reseal_json_member(bundle, relative, wrapper)

        compatibility = json.loads(
            (bundle / "compatibility.json").read_text(encoding="ascii")
        )
        compatibility.update(
            profile=CANONICAL_CPU_PROFILE,
            python=SCIENCE_CPU_PYTHON,
            versions=dict(SCIENCE_CPU_VERSIONS),
            lock_sha256=SCIENCE_CPU_LOCK_SHA256,
            implementation_identity=dict(
                HISTORICAL_V1_2_PACKAGE_IMPLEMENTATION_IDENTITY
            ),
        )
        compatibility.pop("gpu_runtime_policy_sha256")
        compatibility["cpu_runtime_policy_sha256"] = science_cpu_runtime_policy()[
            "sha256"
        ]
        self._reseal_json_member(bundle, "compatibility.json", compatibility)

        provenance = json.loads(
            (bundle / "provenance.json").read_text(encoding="ascii")
        )
        provenance["profile"] = CANONICAL_CPU_PROFILE
        if profiled_provenance:
            provenance["details"].update(
                profile=CANONICAL_CPU_PROFILE,
                device="cpu",
            )
        else:
            provenance["details"].pop("profile")
            provenance["details"].pop("device")
        self._reseal_json_member(bundle, "provenance.json", provenance)

        manifest_path = bundle / "bundle.json"
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        manifest["profile"] = CANONICAL_CPU_PROFILE
        manifest_path.write_bytes(canonical_json_bytes(manifest))

    def test_registered_cpu_and_gpu_profiles_share_the_same_reviewed_assets(self) -> None:
        self.assertEqual(
            set(BUNDLE_V2_ASSET_PROFILES),
            {
                CANONICAL_CPU_PROFILE,
                CANONICAL_GPU_PROFILE,
                EFFICIENT_GPU_PROFILE,
                FULL_IMAGE_GPU_PROFILE,
            },
        )
        self.assertEqual(
            BUNDLE_V2_ASSET_PROFILES[CANONICAL_CPU_PROFILE],
            BUNDLE_V2_ASSET_PROFILES[CANONICAL_GPU_PROFILE],
        )
        self.assertEqual(
            BUNDLE_V2_ASSET_PROFILES[FULL_IMAGE_GPU_PROFILE],
            BUNDLE_V2_ASSET_PROFILES[CANONICAL_GPU_PROFILE],
        )
        self.assertEqual(
            BUNDLE_V2_ASSET_PROFILES[EFFICIENT_GPU_PROFILE]["sam2_architecture"],
            "sam2.1_hiera_tiny",
        )

    def test_gpu_write_and_verify_use_only_the_gpu_runtime_contract(self) -> None:
        request = self._request(CANONICAL_GPU_PROFILE, "gpu-bundle")
        with (
            self._bundle_dependencies(),
            mock.patch.object(
                model_bundle,
                "science_cpu_lock_identity",
                side_effect=AssertionError("GPU bundle consulted the CPU lock"),
            ),
            mock.patch.object(
                model_bundle,
                "science_cpu_runtime_policy",
                side_effect=AssertionError("GPU bundle consulted the CPU policy"),
            ),
        ):
            result = write_model_bundle_v2(request)
            verified = verify_model_bundle(request.output)

        self.assertEqual(result["profile"], CANONICAL_GPU_PROFILE)
        self.assertEqual(verified.profile, CANONICAL_GPU_PROFILE)
        self.assertEqual(verified.normalized_config["device"], "cuda")
        self.assertEqual(verified.compatibility["python"], SCIENCE_GPU_PYTHON)
        self.assertEqual(verified.compatibility["versions"], SCIENCE_GPU_VERSIONS)
        self.assertEqual(verified.compatibility["lock_sha256"], SCIENCE_GPU_LOCK_SHA256)
        self.assertEqual(
            verified.compatibility["gpu_runtime_policy_sha256"],
            science_gpu_runtime_policy()["sha256"],
        )
        self.assertNotIn("cpu_runtime_policy_sha256", verified.compatibility)
        self.assertEqual(verified.provenance["details"]["device"], "cuda")
        self.assertTrue(self.identity_calls)
        self.assertEqual(set(self.identity_calls), {"1.9.3"})

    def test_cpu_writer_is_closed_before_staging(self) -> None:
        for include_profile in (False, True):
            request = self._request(
                CANONICAL_CPU_PROFILE,
                f"rejected-cpu-{include_profile}",
                include_provenance_profile=include_profile,
            )
            with self.assertRaisesRegex(PublicIOError, "historical read/inspect/verify-only"):
                write_model_bundle_v2(request)
            self.assertFalse(request.output.exists())
            self.assertEqual(
                list(request.output.parent.glob(f".{request.output.name}.bundle.*")),
                [],
            )
        self.assertEqual(self.identity_calls, [])

    def test_historical_cpu_legacy_and_profiled_bundles_verify_against_frozen_identity(self) -> None:
        verified_bundles = []
        with self._bundle_dependencies():
            for profiled in (False, True):
                request = self._request(
                    CANONICAL_GPU_PROFILE,
                    f"historical-cpu-{profiled}",
                )
                write_model_bundle_v2(request)
                self._convert_gpu_bundle_to_historical_cpu(
                    request.output,
                    profiled_provenance=profiled,
                )
                with (
                    mock.patch.object(
                        model_bundle,
                        "science_gpu_lock_identity",
                        side_effect=AssertionError("CPU verification consulted the GPU lock"),
                    ),
                    mock.patch.object(
                        model_bundle,
                        "science_gpu_runtime_policy",
                        side_effect=AssertionError("CPU verification consulted the GPU policy"),
                    ),
                ):
                    verified_bundles.append(verify_model_bundle(request.output))

        legacy_verified, profiled_verified = verified_bundles
        for verified in verified_bundles:
            self.assertEqual(verified.profile, CANONICAL_CPU_PROFILE)
            self.assertEqual(verified.compatibility["python"], SCIENCE_CPU_PYTHON)
            self.assertEqual(verified.compatibility["versions"], SCIENCE_CPU_VERSIONS)
            self.assertEqual(verified.compatibility["lock_sha256"], SCIENCE_CPU_LOCK_SHA256)
            self.assertEqual(
                verified.compatibility["implementation_identity"],
                HISTORICAL_V1_2_PACKAGE_IMPLEMENTATION_IDENTITY,
            )
            self.assertEqual(
                verified.compatibility["cpu_runtime_policy_sha256"],
                science_cpu_runtime_policy()["sha256"],
            )
            self.assertNotIn("gpu_runtime_policy_sha256", verified.compatibility)
        self.assertNotIn("profile", legacy_verified.provenance["details"])
        self.assertEqual(profiled_verified.provenance["details"]["device"], "cpu")
        # Each GPU mint is checked before and after publication; historical CPU
        # verification never adds a v1.2 live-identity lookup.
        self.assertEqual(self.identity_calls, ["1.9.3"] * 6)

        compatibility = json.loads(
            (profiled_verified.root / "compatibility.json").read_text(encoding="ascii")
        )
        compatibility["implementation_identity"]["sha256"] = "0" * 64
        self._reseal_json_member(
            profiled_verified.root,
            "compatibility.json",
            compatibility,
        )
        with self._bundle_dependencies(), self.assertRaisesRegex(
            PublicIOError,
            "dependency compatibility",
        ):
            verify_model_bundle(profiled_verified.root)

    def test_gpu_write_rejects_missing_or_mismatched_profile_device_identity(self) -> None:
        base = self._request(CANONICAL_GPU_PROFILE, "unused")
        missing = replace(
            base,
            output=self.root / "missing-provenance-profile",
            provenance={
                key: value
                for key, value in base.provenance.items()
                if key not in {"profile", "device"}
            },
        )
        wrong_provenance = replace(
            base,
            output=self.root / "wrong-provenance-device",
            provenance={**base.provenance, "device": "cpu"},
        )
        wrong_normalized = replace(
            base,
            output=self.root / "wrong-normalized-device",
            normalized_config={**base.normalized_config, "device": "cpu"},
        )
        cases = (
            (missing, "field closure"),
            (wrong_provenance, "profile/device pairing"),
            (wrong_normalized, "normalized config"),
        )
        with self._bundle_dependencies():
            for candidate, pattern in cases:
                with self.subTest(output=candidate.output.name):
                    with self.assertRaisesRegex(PublicIOError, pattern):
                        write_model_bundle_v2(candidate)
                    self.assertFalse(candidate.output.exists())
                    self.assertEqual(
                        list(candidate.output.parent.glob(f".{candidate.output.name}.bundle.*")),
                        [],
                    )

    def test_gpu_verifier_rejects_resealed_cpu_policy_or_device(self) -> None:
        policy_request = self._request(CANONICAL_GPU_PROFILE, "tampered-policy")
        device_request = self._request(CANONICAL_GPU_PROFILE, "tampered-device")
        with self._bundle_dependencies():
            write_model_bundle_v2(policy_request)
            compatibility = json.loads(
                (policy_request.output / "compatibility.json").read_text(encoding="ascii")
            )
            compatibility["cpu_runtime_policy_sha256"] = compatibility.pop(
                "gpu_runtime_policy_sha256"
            )
            self._reseal_json_member(policy_request.output, "compatibility.json", compatibility)
            with self.assertRaisesRegex(PublicIOError, "dependency compatibility"):
                verify_model_bundle(policy_request.output)

            write_model_bundle_v2(device_request)
            provenance = json.loads(
                (device_request.output / "provenance.json").read_text(encoding="ascii")
            )
            provenance["details"]["device"] = "cpu"
            self._reseal_json_member(device_request.output, "provenance.json", provenance)
            with self.assertRaisesRegex(PublicIOError, "profile/device pairing"):
                verify_model_bundle(device_request.output)


if __name__ == "__main__":
    unittest.main()
