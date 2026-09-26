from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
import warnings
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest import mock

from compag_curation.canonical.spec import (
    CANONICAL_EMBEDDING_DIMENSIONS,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_PCA_DIMENSIONS,
    CANONICAL_GPU_PROFILE,
)
from compag_curation.model_bundle import (
    BUNDLE_SCHEMA_V2,
    BUNDLE_V2_ASSET_PROFILES,
    BundleV2WriteRequest,
    FIXED_CANONICAL_METHOD,
    compute_feature_order_sha256,
    encode_float32_npy,
    feature_matrix,
    _validate_canonical_ubj,
    verify_model_bundle,
    write_model_bundle_v2,
)
from compag_curation.public_io import PublicIOError, canonical_json_bytes


class ModelBundleV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.source_payloads = {
            "hiera.yaml": b"fixture-hiera-l-config\n",
            "hiera.pt": b"fixture-hiera-l-checkpoint\n",
            "sam-license.txt": b"fixture-sam-license\n",
            "resnet50.pth": b"fixture-resnet50-weights\n",
            "resnet-license.txt": b"fixture-resnet-license\n",
        }
        for name, payload in self.source_payloads.items():
            (self.inputs / name).write_bytes(payload)
        self.asset_profile = {
            "sam2_architecture": "sam2.1_hiera_large",
            "sam2_config_locator": "configs/sam2.1/sam2.1_hiera_l",
            "sam2_checkpoint_sha256": self._digest("hiera.pt"),
            "sam2_config_sha256": self._digest("hiera.yaml"),
            "sam2_license_sha256": self._digest("sam-license.txt"),
            "resnet50_architecture": "torchvision.models.resnet50",
            "resnet50_weights_identity": "ResNet50_Weights.IMAGENET1K_V2",
            "resnet50_weights_sha256": self._digest("resnet50.pth"),
            "resnet50_license_sha256": self._digest("resnet-license.txt"),
        }
        self.prototype = (1.0, *(0.0 for _ in range(CANONICAL_EMBEDDING_DIMENSIONS - 1)))
        self.prototype_npy = encode_float32_npy(
            self.prototype,
            shape=(CANONICAL_EMBEDDING_DIMENSIONS,),
            role="fixture prototype",
        )
        self.pca_mean_npy = encode_float32_npy(
            (0.0,) * CANONICAL_EMBEDDING_DIMENSIONS,
            shape=(CANONICAL_EMBEDDING_DIMENSIONS,),
            role="fixture PCA mean",
        )
        components = tuple(
            1.0 if column == row else 0.0
            for row in range(CANONICAL_PCA_DIMENSIONS)
            for column in range(CANONICAL_EMBEDDING_DIMENSIONS)
        )
        self.pca_components_npy = encode_float32_npy(
            components,
            shape=(CANONICAL_PCA_DIMENSIONS, CANONICAL_EMBEDDING_DIMENSIONS),
            role="fixture PCA components",
        )
        self.classifier_ubj = self._classifier_bytes()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _digest(self, name: str) -> str:
        return hashlib.sha256(self.source_payloads[name]).hexdigest()

    def _classifier_bytes(
        self,
        *,
        objective: str = "binary:logistic",
        feature_order: tuple[str, ...] = CANONICAL_FEATURE_ORDER,
        attribute: tuple[str, str] | None = None,
        raw_format: str = "ubj",
    ) -> bytes:
        import numpy as np
        import xgboost as xgb

        self.assertEqual(xgb.__version__, "2.1.1")
        matrix = np.zeros((32, len(feature_order)), dtype=np.float32)
        matrix[:, 0] = np.arange(32, dtype=np.float32) % 2
        labels = np.arange(32, dtype=np.float32) % 2
        booster = xgb.train(
            {
                "objective": objective,
                "max_depth": 1,
                "eta": 1.0,
                "nthread": 1,
                "seed": 42,
            },
            xgb.DMatrix(matrix, label=labels, feature_names=list(feature_order)),
            num_boost_round=1,
        )
        if attribute is not None:
            booster.set_attr(**{attribute[0]: attribute[1]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bytes(booster.save_raw(raw_format=raw_format))

    def _request(self, output: str = "bundle-v2", **changes: object) -> BundleV2WriteRequest:
        feature_hash = CANONICAL_FEATURE_ORDER_SHA256
        profile = self.asset_profile
        request = BundleV2WriteRequest(
            output=self.root / output,
            profile=CANONICAL_GPU_PROFILE,
            classifier_ubj=self.classifier_ubj,
            feature_order=CANONICAL_FEATURE_ORDER,
            feature_order_sha256=feature_hash,
            imputer_statistics={
                name: float(index) for index, name in enumerate(CANONICAL_FEATURE_ORDER)
            },
            prototype_npy=self.prototype_npy,
            pca_mean_npy=self.pca_mean_npy,
            pca_components_npy=self.pca_components_npy,
            pca_explained_variance=(1.0,) * CANONICAL_PCA_DIMENSIONS,
            normalized_config={
                "profile": CANONICAL_GPU_PROFILE,
                "device": "cuda",
                "seed": 42,
                "tile_size": 512,
                "tile_stride": 512,
                "sam2_checkpoint_sha256": profile["sam2_checkpoint_sha256"],
                "sam2_config_sha256": profile["sam2_config_sha256"],
                "sam2_config_locator": profile["sam2_config_locator"],
                "embedding_weights_sha256": profile["resnet50_weights_sha256"],
                "embedding_dimensions": CANONICAL_EMBEDDING_DIMENSIONS,
                "pca_components": CANONICAL_PCA_DIMENSIONS,
                "proposal_scales": [1.0],
                "yolo_enabled": False,
                "feature_order_sha256": feature_hash,
                "inference_threshold": 0.5,
                "nms_iou_threshold": 0.5,
                "threshold_method": FIXED_CANONICAL_METHOD,
            },
            preprocessing={
                "profile": CANONICAL_GPU_PROFILE,
                "tile_size": 512,
                "tile_stride": 512,
                "far_edge_alignment": True,
                "padding": "bottom-right-edge-value",
                "tile_format": "jpg",
                "jpeg_quality": 95,
                "yellow_card_hsv_lower": [15, 40, 60],
                "yellow_card_hsv_upper": [40, 255, 255],
                "yellow_card_quad_epsilon_fraction": 0.02,
                "yellow_card_fallback": "identity",
            },
            proposal_config={
                "profile": CANONICAL_GPU_PROFILE,
                "architecture": "sam2.1_hiera_large",
                "config_locator": "configs/sam2.1/sam2.1_hiera_l",
                "points_per_side": 64,
                "points_per_batch": 512,
                "pred_iou_thresh": 0.8,
                "stability_score_thresh": 0.88,
                "crop_n_layers": 0,
                "crop_n_points_downscale_factor": 2,
                "crop_overlap_ratio": 0.4,
                "exclude_largest": False,
                "proposal_scales": [1.0],
                "merge_iou_threshold": 0.75,
                "max_masks_per_tile": 500,
                "yolo_enabled": False,
                "proposal_policy": "amg_only",
                "proposal_identity": "compag-canonical-proposal-v2",
                "feature_mode": "ultra",
                "feature_crop_scales": [0.67, 0.8, 1.0, 1.25],
                "decision_threshold": 0.5,
                "full_image_nms_iou": 0.5,
                "sam2_checkpoint_sha256": profile["sam2_checkpoint_sha256"],
                "sam2_config_sha256": profile["sam2_config_sha256"],
                "sam2_config_locator": profile["sam2_config_locator"],
                "feature_order_sha256": feature_hash,
            },
            feature_config={
                "profile": CANONICAL_GPU_PROFILE,
                "mode": "ultra",
                "feature_crop_scales": [0.67, 0.8, 1.0, 1.25],
                "masked_crop_padding": 0.1,
                "embedding_backbone": "resnet50-imagenet1k-v2",
                "embedding_weights_sha256": profile["resnet50_weights_sha256"],
                "embedding_dimensions": CANONICAL_EMBEDDING_DIMENSIONS,
                "prototype": "train_positive_mean_l2_normalized",
                "pca_components": CANONICAL_PCA_DIMENSIONS,
                "feature_order_sha256": feature_hash,
                "missing_feature_policy": "reject",
                "zero_fill": False,
            },
            training_config={
                "profile": CANONICAL_GPU_PROFILE,
                "group_test_fraction": 0.2,
                "group_cv_folds": 5,
                "search_iterations": 30,
                "inner_validation_fraction": 0.3,
                "validation_scale": 1.0,
                "early_stopping_rounds": 30,
                "review_action_weights": {
                    "accept": 1.0,
                    "flip": 1.0,
                    "sus_accept": 0.4,
                    "sus_flip": 0.4,
                    "skip": 0.0,
                },
                "tabular_augmentation_enabled": False,
                "safe_smote_train_fold_only": True,
                "scale_pos_weight": 1.0,
                "xgboost_host_thread_count": 1,
                "inference_threshold": 0.5,
                "threshold_method": FIXED_CANONICAL_METHOD,
                "full_image_nms_iou": 0.5,
                "feature_order_sha256": feature_hash,
            },
            compatibility={
                "classifier_format": "XGBOOST_UBJ",
                "no_pickle_or_joblib": True,
                "feature_order_sha256": feature_hash,
                "fixed_threshold": 0.5,
            },
            provenance={
                "profile": CANONICAL_GPU_PROFILE,
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
                "feature_order_sha256": feature_hash,
                "ubj_parity_status": "PASS",
                "paper_result_reproduction": "NOT_CLAIMED",
            },
            sam2_config=self.inputs / "hiera.yaml",
            sam2_checkpoint=self.inputs / "hiera.pt",
            sam2_license=self.inputs / "sam-license.txt",
            resnet50_weights=self.inputs / "resnet50.pth",
            resnet50_license=self.inputs / "resnet-license.txt",
        )
        return replace(request, **changes)

    @contextmanager
    def _profile(self):
        with mock.patch.dict(
            BUNDLE_V2_ASSET_PROFILES,
            {CANONICAL_GPU_PROFILE: self.asset_profile},
            clear=True,
        ):
            yield

    def _assert_no_staging(self, output: Path) -> None:
        self.assertFalse(output.exists())
        self.assertEqual(list(output.parent.glob(f".{output.name}.bundle.*")), [])

    def _reseal_json_member(self, bundle: Path, relative: str, value: object) -> None:
        member_path = bundle / relative
        payload = canonical_json_bytes(value)
        member_path.write_bytes(payload)
        manifest_path = bundle / "bundle.json"
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        for record in manifest["members"]:
            if record["path"] == relative:
                record["sha256"] = hashlib.sha256(payload).hexdigest()
                record["size_bytes"] = len(payload)
                break
        else:
            self.fail(f"missing fixture manifest member: {relative}")
        manifest_path.write_bytes(canonical_json_bytes(manifest))

    def test_canonical_feature_digest_uses_the_frozen_line_encoding(self) -> None:
        self.assertEqual(
            compute_feature_order_sha256(CANONICAL_FEATURE_ORDER),
            CANONICAL_FEATURE_ORDER_SHA256,
        )
        self.assertNotEqual(
            compute_feature_order_sha256(tuple(reversed(CANONICAL_FEATURE_ORDER))),
            CANONICAL_FEATURE_ORDER_SHA256,
        )

    def test_real_xgboost_211_round_trip_and_npy_normalization(self) -> None:
        header_length = int.from_bytes(self.prototype_npy[8:10], "little")
        header = self.prototype_npy[10 : 10 + header_length]
        noncanonical_header = header.replace(b", 'shape'", b",  'shape'", 1)
        noncanonical_header = noncanonical_header[:-2] + noncanonical_header[-1:]
        self.assertEqual(len(noncanonical_header), len(header))
        noncanonical_npy = (
            self.prototype_npy[:10]
            + noncanonical_header
            + self.prototype_npy[10 + header_length :]
        )
        self.assertNotEqual(noncanonical_npy, self.prototype_npy)
        request = self._request(prototype_npy=noncanonical_npy)
        with self._profile():
            result = write_model_bundle_v2(request)
            verified = verify_model_bundle(request.output)
        self.assertEqual(result["schema"], BUNDLE_SCHEMA_V2)
        self.assertEqual(verified.feature_order, CANONICAL_FEATURE_ORDER)
        self.assertEqual(verified.feature_order_sha256, CANONICAL_FEATURE_ORDER_SHA256)
        self.assertEqual(verified.prototype_npy, self.prototype_npy)
        self.assertEqual(verified.prototype, self.prototype)
        self.assertEqual(len(verified.imputer), 93)
        row = {name: None for name in CANONICAL_FEATURE_ORDER}
        self.assertEqual(
            feature_matrix([row], verified.imputer, feature_order=verified.feature_order),
            [list(verified.imputer)],
        )

    def test_sparse_config_and_provenance_closures_fail_before_staging(self) -> None:
        base = self._request()
        cases: list[tuple[str, BundleV2WriteRequest, str]] = []
        normalized = dict(base.normalized_config)
        normalized.pop("tile_stride")
        cases.append(("normalized", replace(base, normalized_config=normalized), "normalized config"))
        training = dict(base.training_config)
        training["group_cv_folds"] = 5.0
        cases.append(("training type", replace(base, training_config=training), "training config"))
        provenance = dict(base.provenance)
        provenance.pop("feature_state_fit_rows_sha256")
        cases.append(("provenance", replace(base, provenance=provenance), "field closure"))
        for index, (name, candidate, pattern) in enumerate(cases):
            candidate = replace(candidate, output=self.root / f"sparse-{index}")
            with self.subTest(name=name), self._profile():
                with self.assertRaisesRegex(PublicIOError, pattern):
                    write_model_bundle_v2(candidate)
                self._assert_no_staging(candidate.output)

    def test_verifier_rejects_resealed_sparse_config(self) -> None:
        request = self._request()
        with self._profile():
            write_model_bundle_v2(request)
            wrapper = json.loads(
                (request.output / "normalized_config.json").read_text(encoding="ascii")
            )
            wrapper["config"].pop("tile_stride")
            self._reseal_json_member(request.output, "normalized_config.json", wrapper)
            with self.assertRaisesRegex(PublicIOError, "normalized config"):
                verify_model_bundle(request.output)
            identity_request = replace(self._request(), output=self.root / "proposal-identity")
            write_model_bundle_v2(identity_request)
            proposal = json.loads(
                (identity_request.output / "proposal_config.json").read_text(encoding="ascii")
            )
            proposal["config"]["proposal_identity"] = "compag-canonical-proposal-v1"
            self._reseal_json_member(identity_request.output, "proposal_config.json", proposal)
            with self.assertRaisesRegex(PublicIOError, "proposal config"):
                verify_model_bundle(identity_request.output)
            implementation_request = replace(self._request(), output=self.root / "implementation-identity")
            write_model_bundle_v2(implementation_request)
            compatibility = json.loads(
                (implementation_request.output / "compatibility.json").read_text(encoding="ascii")
            )
            compatibility["implementation_identity"]["sha256"] = "f" * 64
            self._reseal_json_member(
                implementation_request.output,
                "compatibility.json",
                compatibility,
            )
            with self.assertRaisesRegex(PublicIOError, "dependency compatibility"):
                verify_model_bundle(implementation_request.output)

    def test_feature_state_and_training_count_invariants(self) -> None:
        base = self._request()
        bad_prototype = encode_float32_npy(
            (2.0, *(0.0 for _ in range(CANONICAL_EMBEDDING_DIMENSIONS - 1))),
            shape=(CANONICAL_EMBEDDING_DIMENSIONS,),
        )
        nonfinite_mean = self.pca_mean_npy[:-4] + b"\x00\x00\xc0\x7f"
        cases = [
            (replace(base, prototype_npy=bad_prototype), "unit L2"),
            (replace(base, pca_mean_npy=nonfinite_mean), "nonfinite"),
            (
                replace(base, provenance={**base.provenance, "training_row_count": 31}),
                "integer >= 32",
            ),
            (
                replace(base, provenance={**base.provenance, "positive_training_row_count": 33}),
                "positive_training_row_count",
            ),
        ]
        for index, (candidate, pattern) in enumerate(cases):
            candidate = replace(candidate, output=self.root / f"bad-state-{index}")
            with self.subTest(pattern=pattern), self._profile():
                with self.assertRaisesRegex(PublicIOError, pattern):
                    write_model_bundle_v2(candidate)
                self._assert_no_staging(candidate.output)

    def test_ubj_format_objective_attributes_and_private_metadata_are_closed(self) -> None:
        base = self._request()
        cases = [
            (self.classifier_ubj + b"trailing", "trailing"),
            (self._classifier_bytes(raw_format="json"), "canonical XGBoost UBJ"),
            (self._classifier_bytes(raw_format="deprecated"), "canonical XGBoost UBJ"),
            (self._classifier_bytes(objective="reg:squarederror"), "single-output semantics"),
            (self._classifier_bytes(attribute=("unexpected", "value")), "unexpected Booster attributes"),
            (
                self._classifier_bytes(
                    attribute=("source", "/" + "home/private/model")
                ),
                "private locator metadata",
            ),
            (
                self._classifier_bytes(feature_order=tuple(reversed(CANONICAL_FEATURE_ORDER))),
                "feature names differ",
            ),
        ]
        for index, (payload, pattern) in enumerate(cases):
            candidate = replace(base, output=self.root / f"bad-ubj-{index}", classifier_ubj=payload)
            with self.subTest(pattern=pattern), self._profile():
                with self.assertRaisesRegex(PublicIOError, pattern):
                    write_model_bundle_v2(candidate)
                self._assert_no_staging(candidate.output)

    def test_ubj_numeric_bytes_that_resemble_unc_locator_are_accepted(self) -> None:
        # XGBoost UBJ stores learned float arrays as raw big-endian IEEE-754
        # bytes.  Replace the fixture's negative leaf value with a finite float
        # whose representation contains two 0x5c bytes.  The result is still a
        # canonical, attribute-free XGBoost model; those bytes are numeric data,
        # not a private UNC locator.
        original_weight = b"\xbf\xcc\xcc\xcd"
        colliding_weight = b"\x3f\x5c\x5c\x00"
        self.assertEqual(self.classifier_ubj.count(original_weight), 2)
        classifier = self.classifier_ubj.replace(original_weight, colliding_weight)
        self.assertIn(b"\\\\", classifier)

        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(bytearray(classifier))
        self.assertEqual(bytes(booster.save_raw(raw_format="ubj")), classifier)
        self.assertEqual(booster.attributes(), {})

        request = self._request("numeric-locator-collision", classifier_ubj=classifier)
        with self._profile():
            write_model_bundle_v2(request)
            verify_model_bundle(request.output)
        self.assertEqual((request.output / "classifier.ubj").read_bytes(), classifier)

    def test_ubj_complete_decoded_json_private_metadata_is_closed(self) -> None:
        # Exercise a textual string in the complete model projection that is
        # deliberately outside save_config(), Booster attributes, and the
        # feature-name/type views.  This keeps the privacy check closed if a
        # future XGBoost model schema adds another string-valued metadata field.
        import xgboost as xgb

        real_save_raw = xgb.Booster.save_raw

        def projected_model_with_private_locator(
            booster: object,
            *,
            raw_format: str = "deprecated",
        ) -> bytearray:
            rendered = bytes(real_save_raw(booster, raw_format=raw_format))
            if raw_format != "json":
                return bytearray(rendered)
            decoded = json.loads(rendered)
            decoded["learner"]["gradient_booster"]["future_metadata"] = (
                "C:\\Users\\private\\classifier.ubj"
            )
            return bytearray(
                json.dumps(
                    decoded,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
            )

        with mock.patch.object(
            xgb.Booster,
            "save_raw",
            autospec=True,
            side_effect=projected_model_with_private_locator,
        ):
            with self.assertRaisesRegex(PublicIOError, "private locator metadata"):
                _validate_canonical_ubj(self.classifier_ubj)

    def test_ubj_complete_decoded_json_is_bounded_and_strict(self) -> None:
        with mock.patch("compag_curation.model_bundle.MAX_CLASSIFIER_JSON_BYTES", 1):
            with self.assertRaisesRegex(PublicIOError, "exceeds its size bound"):
                _validate_canonical_ubj(self.classifier_ubj)

        import xgboost as xgb

        real_save_raw = xgb.Booster.save_raw

        def projected_model_with_duplicate_keys(
            booster: object,
            *,
            raw_format: str = "deprecated",
        ) -> bytearray:
            if raw_format == "json":
                return bytearray(b'{"learner":{},"learner":{}}')
            return bytearray(real_save_raw(booster, raw_format=raw_format))

        with mock.patch.object(
            xgb.Booster,
            "save_raw",
            autospec=True,
            side_effect=projected_model_with_duplicate_keys,
        ):
            with self.assertRaisesRegex(PublicIOError, "is not strict JSON"):
                _validate_canonical_ubj(self.classifier_ubj)

    def test_published_r92_ubj_binary_locator_collisions_are_not_metadata(self) -> None:
        classifier = (
            Path(__file__).resolve().parents[1]
            / "src/compag_curation/resources/presets/compag_cj_r92/classifier.ubj"
        ).read_bytes()
        self.assertGreater(classifier.count(b"\\\\"), 0)
        self.assertIsNotNone(re.search(rb"[a-z]:\\", classifier.lower()))

        booster = _validate_canonical_ubj(classifier)
        self.assertEqual(booster.attributes(), {})
        self.assertEqual(booster.num_features(), 93)

    def test_asset_oversize_hash_and_post_write_parser_failures_clean_staging(self) -> None:
        wrong_weights = self.inputs / "wrong-resnet50.pth"
        wrong_weights.write_bytes(b"wrong fixture weights\n")
        bad_hash = self._request("bad-hash", resnet50_weights=wrong_weights)
        with self._profile():
            with self.assertRaisesRegex(PublicIOError, "ResNet50 weights hash mismatch"):
                write_model_bundle_v2(bad_hash)
        self._assert_no_staging(bad_hash.output)

        oversize = self._request("oversize")
        with self._profile(), mock.patch(
            "compag_curation.model_bundle.MAX_RESNET50_WEIGHTS_BYTES", 4
        ):
            with self.assertRaisesRegex(PublicIOError, "exceeds its size bound"):
                write_model_bundle_v2(oversize)
        self._assert_no_staging(oversize.output)

        parser_failure = self._request("parser-failure")
        with self._profile(), mock.patch(
            "compag_curation.model_bundle.verify_model_bundle",
            side_effect=PublicIOError("fixture parser failure"),
        ):
            with self.assertRaisesRegex(PublicIOError, "fixture parser failure"):
                write_model_bundle_v2(parser_failure)
        self._assert_no_staging(parser_failure.output)


if __name__ == "__main__":
    unittest.main()
