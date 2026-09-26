from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock


HEAVY_MODULES = {
    "cv2", "imblearn", "joblib", "matplotlib", "numpy", "pandas", "scipy",
    "sklearn", "torch", "torchvision", "ultralytics", "xgboost",
}


class CanonicalProfileContractTests(unittest.TestCase):
    def test_profile_initialization_and_validation_are_inert(self) -> None:
        before = set(sys.modules)
        from compag_curation.public_config import (
            BALANCED_PROFILE,
            BALANCED_PROJECT_TOML,
            CANONICAL_CPU_PROJECT_TOML,
            CANONICAL_GPU_PROFILE,
            CANONICAL_PROFILE,
            CANONICAL_FEATURE_ORDER_SHA256,
            FEATURE_ORDER_SHA256,
            PublicConfigurationError,
            build_public_plan,
            image_files,
            initialize_project,
            load_public_config,
        )

        introduced = {name.split(".", 1)[0] for name in set(sys.modules) - before}
        self.assertFalse(introduced & HEAVY_MODULES)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu_root = root / "gpu"
            balanced_root = root / "balanced"
            canonical_root = root / "canonical-cpu"
            result = initialize_project(gpu_root)
            self.assertEqual(result["profile"], CANONICAL_GPU_PROFILE)
            with self.assertRaisesRegex(PublicConfigurationError, "historical read/inspect/verify-only"):
                initialize_project(root / "rejected-balanced", profile=BALANCED_PROFILE)
            with self.assertRaisesRegex(PublicConfigurationError, "historical read/inspect/verify-only"):
                initialize_project(root / "rejected-cpu", profile=CANONICAL_PROFILE)
            self.assertFalse((root / "rejected-balanced").exists())
            self.assertFalse((root / "rejected-cpu").exists())

            balanced_root.mkdir()
            canonical_root.mkdir()
            (balanced_root / "data/images").mkdir(parents=True)
            (canonical_root / "data/images").mkdir(parents=True)
            (balanced_root / "config.toml").write_text(BALANCED_PROJECT_TOML, encoding="ascii")
            (canonical_root / "config.toml").write_text(CANONICAL_CPU_PROJECT_TOML, encoding="ascii")
            balanced = load_public_config(balanced_root / "config.toml")
            canonical = load_public_config(canonical_root / "config.toml")
            gpu = load_public_config(gpu_root / "config.toml")
            self.assertEqual(balanced.profile, BALANCED_PROFILE)
            self.assertEqual((balanced.tile_size, balanced.tile_overlap), (512, 64))
            self.assertFalse(balanced.is_canonical)
            self.assertTrue(canonical.is_canonical)
            self.assertEqual((gpu.profile, gpu.device), (CANONICAL_GPU_PROFILE, "cuda"))
            self.assertEqual((canonical.tile_size, canonical.tile_stride), (512, 512))
            self.assertEqual(canonical.tile_edge_alignment, "far-edge")
            self.assertEqual(canonical.tile_padding, "bottom-right-edge-value")
            self.assertEqual(canonical.tile_format, "jpg")
            self.assertEqual(canonical.proposal_scales, (1.0,))
            self.assertEqual((canonical.points_per_side, canonical.points_per_batch), (64, 512))
            from compag_curation.canonical.spec import (
                CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
                CANONICAL_STAGE20_AMG_POINTS_PER_BATCH,
            )

            self.assertEqual(CANONICAL_STAGE20_AMG_POINTS_PER_BATCH, 32)
            self.assertEqual(CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH, 32)
            self.assertEqual(
                CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
                CANONICAL_STAGE20_AMG_POINTS_PER_BATCH,
            )
            self.assertNotEqual(
                CANONICAL_STAGE20_AMG_POINTS_PER_BATCH,
                canonical.points_per_batch,
            )
            self.assertNotEqual(
                CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
                canonical.points_per_batch,
            )
            self.assertEqual((canonical.pred_iou_threshold, canonical.stability_threshold), (0.8, 0.88))
            self.assertEqual((canonical.crop_n_layers, canonical.crop_n_points_downscale_factor), (0, 2))
            self.assertEqual(canonical.crop_overlap_ratio, 0.4)
            self.assertEqual(canonical.feature_mode, "ultra")
            self.assertEqual(canonical.feature_crop_scales, (0.67, 0.8, 1.0, 1.25))
            self.assertEqual((canonical.embedding_backbone, canonical.embedding_dimensions), ("resnet50-imagenet1k-v2", 2048))
            self.assertEqual((canonical.masked_crop_padding, canonical.pca_components), (0.1, 32))
            self.assertEqual((canonical.group_cv_splits, canonical.hyperparameter_search_iterations), (5, 30))
            self.assertEqual((canonical.early_stopping_rounds, canonical.inner_validation_fraction), (30, 0.3))
            self.assertEqual(
                (
                    canonical.review_accept_weight,
                    canonical.review_flip_weight,
                    canonical.review_suspect_weight,
                    canonical.review_skip_weight,
                ),
                (1.0, 1.0, 0.4, 0.0),
            )
            self.assertTrue(canonical.safe_smote)
            self.assertEqual((canonical.inference_threshold, canonical.nms_iou_threshold), (0.5, 0.5))
            self.assertFalse(canonical.yolo_enabled)
            self.assertEqual(build_public_plan(balanced)["feature_order_sha256"], FEATURE_ORDER_SHA256)
            self.assertEqual(
                build_public_plan(canonical)["feature_order_sha256"],
                CANONICAL_FEATURE_ORDER_SHA256,
            )
            canonical.images.joinpath(f"{'a' * 240}.ppm").write_bytes(b"placeholder")
            with self.assertRaisesRegex(PublicConfigurationError, r"derived \.jpg tile name"):
                image_files(canonical)

    def test_canonical_profile_rejects_semantic_drift(self) -> None:
        from compag_curation.public_config import (
            BALANCED_PROJECT_TOML,
            CANONICAL_PROJECT_TOML,
            PublicConfigurationError,
            validate_public_mapping,
        )

        mapping = tomllib.loads(CANONICAL_PROJECT_TOML)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(CANONICAL_PROJECT_TOML, encoding="ascii")
            mutations = (
                ("yolo_enabled", True),
                ("proposal_scales", [0.8, 1.0, 1.25]),
                ("inference_threshold", 0.5843676924705505),
                ("tile_stride", 448),
            )
            for field, value in mutations:
                with self.subTest(field=field):
                    candidate = copy.deepcopy(mapping)
                    candidate["execution"][field] = value
                    with self.assertRaises(PublicConfigurationError):
                        validate_public_mapping(candidate, config_path=config_path, config_sha256="0" * 64)

            mismatched = copy.deepcopy(mapping)
            mismatched["assets"] = tomllib.loads(BALANCED_PROJECT_TOML)["assets"]
            with self.assertRaises(PublicConfigurationError):
                validate_public_mapping(mismatched, config_path=config_path, config_sha256="0" * 64)

    def test_gpu_profile_changes_only_profile_and_device(self) -> None:
        from compag_curation.canonical.spec import (
            CANONICAL_CPU_PROFILE as SPEC_CPU_PROFILE,
            CANONICAL_GPU_PROFILE as SPEC_GPU_PROFILE,
            CANONICAL_PROFILE as SPEC_COMPAT_PROFILE,
            CANONICAL_PROFILES as SPEC_PROFILES,
            CanonicalAMGSettings,
        )
        from compag_curation.public_config import (
            CANONICAL_CPU_PROFILE,
            CANONICAL_CPU_PROJECT_TOML,
            CANONICAL_GPU_PROFILE,
            CANONICAL_GPU_PROJECT_TOML,
            CANONICAL_PROFILE,
            CANONICAL_PROFILES,
            PublicConfigurationError,
            load_public_config,
            validate_public_mapping,
        )

        self.assertEqual(CANONICAL_PROFILE, CANONICAL_CPU_PROFILE)
        self.assertEqual(CANONICAL_PROFILES, (CANONICAL_CPU_PROFILE, CANONICAL_GPU_PROFILE))
        self.assertEqual(
            (SPEC_CPU_PROFILE, SPEC_GPU_PROFILE, SPEC_COMPAT_PROFILE, SPEC_PROFILES),
            (CANONICAL_CPU_PROFILE, CANONICAL_GPU_PROFILE, CANONICAL_PROFILE, CANONICAL_PROFILES),
        )
        self.assertEqual(
            CanonicalAMGSettings().provenance_record(profile=CANONICAL_GPU_PROFILE)["profile"],
            CANONICAL_GPU_PROFILE,
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            CanonicalAMGSettings().provenance_record(profile="forged-profile")
        cpu_mapping = tomllib.loads(CANONICAL_CPU_PROJECT_TOML)
        gpu_mapping = tomllib.loads(CANONICAL_GPU_PROJECT_TOML)
        self.assertEqual(cpu_mapping["execution"]["device"], "cpu")
        self.assertEqual(gpu_mapping["execution"]["device"], "cuda")
        for field in set(cpu_mapping["execution"]) - {"profile", "device"}:
            self.assertEqual(gpu_mapping["execution"][field], cpu_mapping["execution"][field])
        self.assertEqual(gpu_mapping["assets"], cpu_mapping["assets"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpu_path = root / "cpu.toml"
            gpu_path = root / "gpu.toml"
            cpu_path.write_text(CANONICAL_CPU_PROJECT_TOML, encoding="ascii")
            gpu_path.write_text(CANONICAL_GPU_PROJECT_TOML, encoding="ascii")
            cpu = load_public_config(cpu_path)
            gpu = load_public_config(gpu_path)
            self.assertEqual((cpu.profile, cpu.device), (CANONICAL_CPU_PROFILE, "cpu"))
            self.assertEqual((gpu.profile, gpu.device), (CANONICAL_GPU_PROFILE, "cuda"))
            self.assertTrue(cpu.is_canonical)
            self.assertTrue(gpu.is_canonical)

            for mapping, device in ((cpu_mapping, "cuda"), (gpu_mapping, "cpu")):
                mismatched = copy.deepcopy(mapping)
                mismatched["execution"]["device"] = device
                with self.assertRaisesRegex(PublicConfigurationError, "requires execution.device"):
                    validate_public_mapping(
                        mismatched,
                        config_path=cpu_path,
                        config_sha256="0" * 64,
                    )

    def test_registry_contains_exact_profile_assets_without_fetching(self) -> None:
        from compag_curation.assets import asset_ids_for_profile, asset_registry, registry_payload
        from compag_curation.public_config import (
            BALANCED_PROFILE,
            CANONICAL_GPU_PROFILE,
            CANONICAL_PROFILE,
        )

        registry = asset_registry()
        self.assertEqual(
            asset_ids_for_profile(BALANCED_PROFILE),
            ("sam2-apache-license", "sam2.1-hiera-tiny-checkpoint", "sam2.1-hiera-tiny-config"),
        )
        self.assertEqual(
            asset_ids_for_profile(CANONICAL_PROFILE),
            (
                "resnet50-imagenet1k-v2-weights", "sam2-apache-license",
                "sam2.1-hiera-large-checkpoint", "sam2.1-hiera-large-config",
                "torchvision-bsd-license",
            ),
        )
        self.assertEqual(
            asset_ids_for_profile(CANONICAL_GPU_PROFILE),
            asset_ids_for_profile(CANONICAL_PROFILE),
        )
        expected = {
            "sam2.1-hiera-large-checkpoint": (
                "sam2.1_hiera_large.pt", 898_083_611,
                "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
            ),
            "sam2.1-hiera-large-config": (
                "sam2.1_hiera_l.yaml", 3_798,
                "1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107",
            ),
            "resnet50-imagenet1k-v2-weights": (
                "resnet50-11ad3fa6.pth", 102_540_417,
                "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
            ),
        }
        for asset_id, identity in expected.items():
            spec = registry[asset_id]
            self.assertEqual((spec.filename, spec.size_bytes, spec.sha256), identity)
            self.assertTrue(spec.url.startswith("https://"))
            self.assertFalse(spec.bundled_in_repository)
        self.assertEqual(registry["resnet50-imagenet1k-v2-weights"].license_spdx, "BSD-3-Clause")
        release_root = Path(__file__).resolve().parents[1]
        recorded = json.loads((release_root / "manifests/PUBLIC_ASSET_REGISTRY.json").read_text(encoding="utf-8"))
        self.assertEqual(recorded, registry_payload())

    def test_efficient_gpu_profile_is_distinct_v2_tiny_contract(self) -> None:
        from compag_curation.canonical.spec import (
            CANONICAL_GPU_PROFILE,
            CANONICAL_PROFILES,
            EFFICIENT_GPU_PROFILE,
            FULL_IMAGE_GPU_PROFILE,
            GPU_EXECUTION_PROFILES,
            V2_PIPELINE_PROFILES,
            CanonicalAMGSettings,
            EfficientAMGSettings,
            amg_settings_for_profile,
            inference_amg_execution_points_per_batch,
            stage20_amg_execution_points_per_batch,
        )
        from compag_curation.assets import asset_ids_for_profile
        from compag_curation.public_config import (
            CANONICAL_GPU_PROJECT_TOML,
            CANONICAL_FEATURE_ORDER_SHA256,
            EFFICIENT_GPU_PROJECT_TOML,
            PublicConfigurationError,
            build_public_plan,
            initialize_project,
            load_public_config,
            validate_public_mapping,
        )

        self.assertNotIn(EFFICIENT_GPU_PROFILE, CANONICAL_PROFILES)
        self.assertEqual(
            GPU_EXECUTION_PROFILES,
            (
                CANONICAL_GPU_PROFILE,
                EFFICIENT_GPU_PROFILE,
                FULL_IMAGE_GPU_PROFILE,
            ),
        )
        self.assertIn(EFFICIENT_GPU_PROFILE, V2_PIPELINE_PROFILES)
        self.assertEqual(stage20_amg_execution_points_per_batch(CANONICAL_GPU_PROFILE), 32)
        self.assertEqual(inference_amg_execution_points_per_batch(CANONICAL_GPU_PROFILE), 32)
        self.assertEqual(stage20_amg_execution_points_per_batch(EFFICIENT_GPU_PROFILE), 16)
        self.assertEqual(inference_amg_execution_points_per_batch(EFFICIENT_GPU_PROFILE), 16)
        with self.assertRaisesRegex(ValueError, "unsupported GPU execution profile"):
            stage20_amg_execution_points_per_batch("forged-profile")

        full_settings = amg_settings_for_profile(CANONICAL_GPU_PROFILE)
        efficient_settings = amg_settings_for_profile(EFFICIENT_GPU_PROFILE)
        self.assertIsInstance(full_settings, CanonicalAMGSettings)
        self.assertNotIsInstance(full_settings, EfficientAMGSettings)
        self.assertIsInstance(efficient_settings, EfficientAMGSettings)
        self.assertEqual(
            (efficient_settings.architecture, efficient_settings.config_locator),
            ("sam2.1_hiera_tiny", "configs/sam2.1/sam2.1_hiera_t"),
        )
        efficient_provenance = efficient_settings.provenance_record()
        self.assertEqual(efficient_provenance["profile"], EFFICIENT_GPU_PROFILE)
        self.assertEqual(
            efficient_provenance["method_classification"],
            "EFFICIENT_SAM2_TINY_NON_EQUIVALENT_VARIANT",
        )
        self.assertEqual(efficient_provenance["canonical_method_equivalence"], "NOT_CLAIMED")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            CanonicalAMGSettings().provenance_record(profile=EFFICIENT_GPU_PROFILE)

        full_mapping = tomllib.loads(CANONICAL_GPU_PROJECT_TOML)
        efficient_mapping = tomllib.loads(EFFICIENT_GPU_PROJECT_TOML)
        self.assertEqual(efficient_mapping["execution"]["profile"], EFFICIENT_GPU_PROFILE)
        self.assertEqual(efficient_mapping["execution"]["device"], "cuda")
        for field in set(full_mapping["execution"]) - {"profile"}:
            self.assertEqual(
                efficient_mapping["execution"][field],
                full_mapping["execution"][field],
            )
        self.assertEqual(
            efficient_mapping["assets"]["checkpoint"],
            "assets/sam2.1_hiera_tiny.pt",
        )
        self.assertEqual(
            efficient_mapping["assets"]["sam2_config_locator"],
            "configs/sam2.1/sam2.1_hiera_t",
        )
        self.assertEqual(
            efficient_mapping["assets"]["embedding_weights_sha256"],
            full_mapping["assets"]["embedding_weights_sha256"],
        )
        self.assertEqual(
            asset_ids_for_profile(EFFICIENT_GPU_PROFILE),
            (
                "resnet50-imagenet1k-v2-weights",
                "sam2-apache-license",
                "sam2.1-hiera-tiny-checkpoint",
                "sam2.1-hiera-tiny-config",
                "torchvision-bsd-license",
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "efficient"
            result = initialize_project(project, profile=EFFICIENT_GPU_PROFILE)
            self.assertEqual(result["profile"], EFFICIENT_GPU_PROFILE)
            config = load_public_config(project / "config.toml")
            self.assertFalse(config.is_canonical)
            self.assertTrue(config.uses_v2_pipeline)
            self.assertTrue(config.is_gpu_execution_profile)
            self.assertEqual(
                build_public_plan(config)["feature_order_sha256"],
                CANONICAL_FEATURE_ORDER_SHA256,
            )

            mismatched = copy.deepcopy(efficient_mapping)
            mismatched["assets"] = full_mapping["assets"]
            with self.assertRaisesRegex(PublicConfigurationError, "Hiera Tiny"):
                validate_public_mapping(
                    mismatched,
                    config_path=project / "config.toml",
                    config_sha256="0" * 64,
                )
            cpu_fallback = copy.deepcopy(efficient_mapping)
            cpu_fallback["execution"]["device"] = "cpu"
            with self.assertRaisesRegex(PublicConfigurationError, "requires execution.device=cuda"):
                validate_public_mapping(
                    cpu_fallback,
                    config_path=project / "config.toml",
                    config_sha256="0" * 64,
                )

        schema = Path(__file__).resolve().parents[1] / "schemas/public_project.schema.json"
        packaged_schema = (
            Path(__file__).resolve().parents[1]
            / "src/compag_curation/resources/public_project.schema.json"
        )
        self.assertEqual(schema.read_bytes(), packaged_schema.read_bytes())
        schema_value = json.loads(schema.read_text(encoding="utf-8"))
        self.assertIn("efficientAssets", schema_value["$defs"])

    def test_full_image_profile_v2_cli_schema_assets_and_legacy_v1_compatibility(self) -> None:
        from compag_curation import public_pipeline
        from compag_curation.assets import asset_ids_for_profile
        from compag_curation.canonical.spec import (
            CANONICAL_GPU_PROFILE,
            CANONICAL_PROFILES,
            FULL_IMAGE_GPU_PROFILE,
            FULL_IMAGE_INFERENCE_AMG_POINTS_PER_BATCH,
            FULL_IMAGE_PROPOSAL_IDENTITY,
            FULL_IMAGE_STAGE20_AMG_POINTS_PER_BATCH,
            FullImageMultiscaleAMGSettings,
            amg_settings_for_profile,
            inference_amg_execution_points_per_batch,
            proposal_identity_for_profile,
            stage20_amg_execution_points_per_batch,
        )
        from compag_curation.cli import main
        from compag_curation.public_config import (
            CANONICAL_GPU_PROJECT_TOML,
            FULL_IMAGE_GPU_PROJECT_TOML,
            FULL_IMAGE_SPATIAL_MODE,
            LEGACY_SCHEMA,
            SCHEMA,
            TILED_SPATIAL_MODE,
            PublicConfigurationError,
            image_files,
            load_public_config,
            validate_public_mapping,
        )

        self.assertNotIn(FULL_IMAGE_GPU_PROFILE, CANONICAL_PROFILES)
        settings = amg_settings_for_profile(FULL_IMAGE_GPU_PROFILE)
        self.assertIsInstance(settings, FullImageMultiscaleAMGSettings)
        self.assertEqual(
            (
                settings.architecture,
                settings.points_per_side,
                settings.points_per_batch,
                settings.execution_points_per_batch,
                settings.crop_n_layers,
                settings.crop_n_points_downscale_factor,
                settings.crop_overlap_ratio,
                settings.max_masks_per_image,
            ),
            (
                "sam2.1_hiera_large",
                64,
                512,
                8,
                2,
                1,
                512 / 1500,
                1000,
            ),
        )
        self.assertEqual(
            (settings.pred_iou_thresh, settings.stability_score_thresh),
            (0.80, 0.88),
        )
        self.assertEqual(
            proposal_identity_for_profile(FULL_IMAGE_GPU_PROFILE),
            FULL_IMAGE_PROPOSAL_IDENTITY,
        )
        self.assertEqual(
            stage20_amg_execution_points_per_batch(FULL_IMAGE_GPU_PROFILE),
            FULL_IMAGE_STAGE20_AMG_POINTS_PER_BATCH,
        )
        self.assertEqual(
            inference_amg_execution_points_per_batch(FULL_IMAGE_GPU_PROFILE),
            FULL_IMAGE_INFERENCE_AMG_POINTS_PER_BATCH,
        )
        self.assertLess(FULL_IMAGE_STAGE20_AMG_POINTS_PER_BATCH, settings.points_per_batch)
        provenance = settings.provenance_record()
        self.assertEqual(provenance["spatial_mode"], FULL_IMAGE_SPATIAL_MODE)
        self.assertEqual(provenance["canonical_method_equivalence"], "NOT_CLAIMED")
        self.assertEqual(
            asset_ids_for_profile(FULL_IMAGE_GPU_PROFILE),
            asset_ids_for_profile(CANONICAL_GPU_PROFILE),
        )

        template = tomllib.loads(FULL_IMAGE_GPU_PROJECT_TOML)
        execution = template["execution"]
        self.assertEqual(template["schema"], SCHEMA)
        self.assertEqual(execution["profile"], FULL_IMAGE_GPU_PROFILE)
        self.assertEqual(execution["spatial_mode"], FULL_IMAGE_SPATIAL_MODE)
        self.assertEqual(execution["execution_points_per_batch"], 8)
        self.assertEqual(execution["max_proposals_per_image"], 1000)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "full-image"
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(["init-project", "--profile", "full-image", "--output", str(output)]),
                    0,
                )
            self.assertEqual(json.loads(capture.getvalue())["profile"], FULL_IMAGE_GPU_PROFILE)
            config = load_public_config(output / "config.toml")
            self.assertEqual(config.spatial_mode, FULL_IMAGE_SPATIAL_MODE)
            self.assertTrue(config.uses_full_image_multiscale)
            self.assertFalse(config.is_canonical)
            self.assertEqual(config.max_proposals_per_image, 1000)
            long_image = config.images / f"{'a' * 240}.ppm"
            long_image.write_bytes(b"placeholder")
            self.assertEqual(image_files(config), [long_image])

            expected_assets = asset_ids_for_profile(FULL_IMAGE_GPU_PROFILE)
            with mock.patch.object(
                public_pipeline,
                "verify_assets",
                return_value=({"status": "PASS"}, 0),
            ) as verify_assets, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main([
                        "assets", "verify", "--profile", "full-image",
                        "--asset-root", str(config.asset_root),
                    ]),
                    0,
                )
            verify_assets.assert_called_once_with(config.asset_root, expected_assets)

            legacy_path = root / "legacy-v1.toml"
            legacy_payload = (
                CANONICAL_GPU_PROJECT_TOML
                .replace(f'schema = "{SCHEMA}"', f'schema = "{LEGACY_SCHEMA}"', 1)
                .replace(f'spatial_mode = "{TILED_SPATIAL_MODE}"\n', "", 1)
                .encode("ascii")
            )
            legacy_path.write_bytes(legacy_payload)
            legacy = load_public_config(legacy_path)
            self.assertEqual(legacy.spatial_mode, TILED_SPATIAL_MODE)
            self.assertFalse(legacy.uses_full_image_multiscale)
            self.assertEqual(legacy_path.read_bytes(), legacy_payload)

            invalid_legacy = copy.deepcopy(template)
            invalid_legacy["schema"] = LEGACY_SCHEMA
            invalid_legacy["execution"].pop("spatial_mode")
            with self.assertRaisesRegex(PublicConfigurationError, "execution.profile"):
                validate_public_mapping(
                    invalid_legacy,
                    config_path=output / "config.toml",
                    config_sha256="0" * 64,
                )

            wrong_mode = copy.deepcopy(template)
            wrong_mode["execution"]["spatial_mode"] = TILED_SPATIAL_MODE
            with self.assertRaisesRegex(PublicConfigurationError, "requires execution.spatial_mode"):
                validate_public_mapping(
                    wrong_mode,
                    config_path=output / "config.toml",
                    config_sha256="0" * 64,
                )

        schema_path = Path(__file__).resolve().parents[1] / "schemas/public_project.schema.json"
        packaged_schema = (
            Path(__file__).resolve().parents[1]
            / "src/compag_curation/resources/public_project.schema.json"
        )
        self.assertEqual(schema_path.read_bytes(), packaged_schema.read_bytes())
        schema_value = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema_value["$id"], "urn:compag-curation:public-project:v2")
        self.assertEqual(schema_value["properties"]["schema"]["const"], SCHEMA)
        full_schema = schema_value["$defs"]["fullImageExecution"]
        self.assertFalse(full_schema["additionalProperties"])
        self.assertIn("execution_points_per_batch", full_schema["required"])
        self.assertEqual(
            full_schema["properties"]["profile"]["const"],
            FULL_IMAGE_GPU_PROFILE,
        )
        self.assertEqual(
            full_schema["properties"]["spatial_mode"]["const"],
            FULL_IMAGE_SPATIAL_MODE,
        )
        self.assertEqual(
            full_schema["properties"]["execution_points_per_batch"]["const"],
            8,
        )
        self.assertEqual(
            full_schema["properties"]["max_proposals_per_image"]["const"],
            1000,
        )

    def test_cli_init_accepts_only_gpu_without_model_imports(self) -> None:
        from compag_curation.cli import main

        before = set(sys.modules)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rejected = root / "cpu-project"
            output = root / "gpu-project"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(["init-project", "--profile", "canonical-xgb-recall-cpu-v1", "--output", str(rejected)])
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(rejected.exists())
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(["init-project", "--profile", "canonical-xgb-recall-gpu-v1", "--output", str(output)]),
                    0,
                )
            self.assertEqual(json.loads(capture.getvalue())["profile"], "canonical-xgb-recall-gpu-v1")
            self.assertIn('profile = "canonical-xgb-recall-gpu-v1"', (output / "config.toml").read_text(encoding="ascii"))
            manifest = json.loads((output / "PROJECT_MANIFEST.json").read_text(encoding="ascii"))
            self.assertEqual(
                manifest["next_command"],
                "python -I -B -m compag_curation inspect-data --config config.toml",
            )
        introduced = {name.split(".", 1)[0] for name in set(sys.modules) - before}
        self.assertFalse(introduced & HEAVY_MODULES)

    def test_cli_defaults_to_gpu_profile_and_cuda_inference(self) -> None:
        from compag_curation import public_pipeline
        from compag_curation.cli import main
        from compag_curation.public_config import CANONICAL_GPU_PROFILE, load_public_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "gpu-project"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["init-project", "--output", str(output)]), 0)
            config = load_public_config(output / "config.toml")
            self.assertEqual((config.profile, config.device), (CANONICAL_GPU_PROFILE, "cuda"))

            result = {"schema": "fixture-inference/v1", "status": "PASS"}
            with mock.patch.object(
                public_pipeline,
                "infer_bundle",
                return_value=(result, 0),
            ) as infer_bundle, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main([
                        "infer", "--execute", "--images", "images", "--bundle", "bundle",
                        "--output", "predictions",
                    ]),
                    0,
                )
            self.assertEqual(infer_bundle.call_args.kwargs["device"], "cuda")

    def test_public_template_default_is_gpu_while_historical_templates_remain_available(self) -> None:
        from compag_curation.public_config import (
            BALANCED_PROFILE,
            BALANCED_PROJECT_TOML,
            CANONICAL_CPU_PROFILE,
            CANONICAL_CPU_PROJECT_TOML,
            CANONICAL_GPU_PROFILE,
            PROFILE,
            PROJECT_TOML,
        )

        self.assertEqual(PROFILE, CANONICAL_GPU_PROFILE)
        self.assertEqual(
            tomllib.loads(PROJECT_TOML)["execution"]["profile"],
            CANONICAL_GPU_PROFILE,
        )
        self.assertEqual(
            tomllib.loads(BALANCED_PROJECT_TOML)["execution"]["profile"],
            BALANCED_PROFILE,
        )
        self.assertEqual(
            tomllib.loads(CANONICAL_CPU_PROJECT_TOML)["execution"]["profile"],
            CANONICAL_CPU_PROFILE,
        )

    def test_cli_maps_canonical_profile_to_registered_assets(self) -> None:
        from compag_curation import public_pipeline
        from compag_curation.cli import main

        expected = (
            "resnet50-imagenet1k-v2-weights",
            "sam2-apache-license",
            "sam2.1-hiera-large-checkpoint",
            "sam2.1-hiera-large-config",
            "torchvision-bsd-license",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(
                public_pipeline,
                "verify_assets",
                return_value=({"status": "PASS"}, 0),
            ) as verify_assets, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main([
                        "assets", "verify", "--profile", "canonical-xgb-recall-cpu-v1",
                        "--asset-root", str(root),
                    ]),
                    0,
                )
            verify_assets.assert_called_once_with(root, expected)


if __name__ == "__main__":
    unittest.main()
