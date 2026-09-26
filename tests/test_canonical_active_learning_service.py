from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from compag_curation.canonical import active_learning_service as adapter
from compag_curation.canonical.active_learning import (
    CanonicalActiveLearningDecision,
    CanonicalActiveLearningRetrainRequest,
    CanonicalActiveLearningTrainingRow,
    _expected_bundle_provenance,
    _training_row_hash,
)
from compag_curation.canonical.features import (
    CANONICAL_RAW_FEATURE_ORDER,
    CanonicalFeatureState,
    CanonicalRawFeature,
    finalize_canonical_feature,
)
from compag_curation.canonical.serialization import (
    load_portable_predictor,
    portable_classifier_bytes,
    predict_portable_probabilities,
    serialize_canonical_feature_state,
)
from compag_curation.canonical.spec import (
    CANONICAL_FEATURE_CROP_SCALES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_GPU_PROFILE,
    CANONICAL_PROFILE,
)
from compag_curation.model_bundle import (
    BUNDLE_V2_ASSET_PROFILES,
    BundleV2WriteRequest,
    _canonical_config_contracts,
    verify_model_bundle,
    write_model_bundle_v2,
)
from compag_curation.public_io import PublicIOError


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _classifier() -> object:
    import numpy as np
    from xgboost import XGBClassifier

    features = np.zeros((32, len(CANONICAL_FEATURE_ORDER)), dtype=np.float32)
    labels = np.arange(32, dtype=np.int32) % 2
    features[:, 0] = labels
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=2,
        max_depth=1,
        learning_rate=0.5,
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    classifier.fit(features, labels, verbose=False)
    classifier.get_booster().feature_names = list(CANONICAL_FEATURE_ORDER)
    return classifier


def _state(prototype_axis: int) -> CanonicalFeatureState:
    import numpy as np

    prototype = np.zeros(2048, dtype=np.float32)
    prototype[prototype_axis] = 1.0
    components = np.zeros((32, 2048), dtype=np.float32)
    for index in range(32):
        components[index, index] = 1.0
    mean = np.zeros(2048, dtype=np.float32)
    for value in (prototype, components, mean):
        value.setflags(write=False)
    return CanonicalFeatureState(
        prototype=prototype,
        pca_components=components,
        pca_mean=mean,
        training_row_count=40,
        positive_row_count=20,
    )


class CanonicalActiveLearningServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
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
        digest = {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in payloads.items()
        }
        self.asset_profile = {
            "sam2_architecture": "sam2.1_hiera_large",
            "sam2_config_locator": "configs/sam2.1/sam2.1_hiera_l",
            "sam2_checkpoint_sha256": digest["hiera.pt"],
            "sam2_config_sha256": digest["hiera.yaml"],
            "sam2_license_sha256": digest["sam-license.txt"],
            "resnet50_architecture": "torchvision.models.resnet50",
            "resnet50_weights_identity": "ResNet50_Weights.IMAGENET1K_V2",
            "resnet50_weights_sha256": digest["resnet50.pth"],
            "resnet50_license_sha256": digest["resnet-license.txt"],
        }
        patcher = mock.patch.dict(
            BUNDLE_V2_ASSET_PROFILES,
            {
                CANONICAL_PROFILE: self.asset_profile,
                CANONICAL_GPU_PROFILE: self.asset_profile,
            },
            clear=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_zero_candidate_inventory_uses_filename_derived_identity(self) -> None:
        image_name = "g0__empty.jpg"
        image_sha256 = _sha("empty-image-bytes")
        claimed_group = "g1"
        claimed_image_id = hashlib.sha256(
            b"compag-image-v1\0"
            + image_name.encode("utf-8")
            + b"\0"
            + claimed_group.encode("ascii")
            + b"\0"
            + image_sha256.encode("ascii")
        ).hexdigest()
        with self.assertRaisesRegex(PublicIOError, "inventory row"):
            adapter._validate_inventory(
                [
                    {
                        "image_name": image_name,
                        "image_id": claimed_image_id,
                        "image_sha256": image_sha256,
                        "group_id": claimed_group,
                        "size_bytes": 1,
                        "width": 1,
                        "height": 1,
                        "tile_count": 1,
                        "candidate_rows": 0,
                    }
                ]
            )

    def _source_bundle(
        self,
        state: CanonicalFeatureState,
        *,
        profile: str = CANONICAL_GPU_PROFILE,
    ) -> Path:
        configs = _canonical_config_contracts(
            profile=profile,
            asset_profile=self.asset_profile,
            feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
        )
        payloads = serialize_canonical_feature_state(state)
        output = self.root / "source_bundle_gpu"
        provenance = {
            "reviewed_sha256": "a" * 64,
            "split_manifest_sha256": "b" * 64,
            "raw_features_sha256": "c" * 64,
            "feature_state_fit_group_sha256": "d" * 64,
            "feature_state_fit_rows_sha256": "e" * 64,
            "feature_state_fit_scope": "OUTER_TRAIN_ROWS_BEFORE_CV",
            "validation_feature_scale": 1.0,
            "cv_score_interpretation": "CONDITIONAL_ON_OUTER_TRAIN_PREFIT_PROTOTYPE_AND_PCA",
            "training_row_count": state.training_row_count,
            "positive_training_row_count": state.positive_row_count,
            "fixed_threshold": 0.5,
            "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
            "ubj_parity_status": "PASS",
            "paper_result_reproduction": "NOT_CLAIMED",
        }
        provenance.update(profile=profile, device="cuda")
        write_model_bundle_v2(
            BundleV2WriteRequest(
                output=output,
                profile=profile,
                classifier_ubj=portable_classifier_bytes(_classifier()),
                feature_order=CANONICAL_FEATURE_ORDER,
                feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
                imputer_statistics={name: 0.0 for name in CANONICAL_FEATURE_ORDER},
                prototype_npy=payloads.prototype_npy,
                pca_mean_npy=payloads.pca_mean_npy,
                pca_components_npy=payloads.pca_components_npy,
                pca_explained_variance=(1.0,) * 32,
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
        )
        return output

    def _request(
        self,
        source_bundle: Path,
        state: CanonicalFeatureState,
        *,
        profile: str = CANONICAL_GPU_PROFILE,
        device: str = "cuda",
    ) -> CanonicalActiveLearningRetrainRequest:
        import numpy as np

        specifications = [
            (f"train-{group}-{label}", group, label)
            for group in ("g0", "g1", "g2", "g3", "g4")
            for label in (0, 1)
        ]
        specifications += [("test-0", "g5", 0), ("test-1", "g5", 1)]
        identities = sorted(
            (
                _sha(f"proposal:{name}"),
                _sha(f"image:{name}"),
                _sha(f"image-bytes:{name}"),
                group,
                label,
            )
            for name, group, label in specifications
        )
        decisions = tuple(
            CanonicalActiveLearningDecision(
                sequence=index,
                phase="ACTIVE_LEARNING_ROUND",
                round_number=1,
                decision_timestamp_utc="2026-08-28T07:00:00Z",
                proposal_id=proposal_id,
                proposal_sha256=proposal_id,
                image_id=image_id,
                image_sha256=image_sha256,
                group_id=group,
                label=label,
                review_action="accept",
                review_weight=1.0,
            )
            for index, (proposal_id, image_id, image_sha256, group, label)
            in enumerate(identities, start=1)
        )
        rows: list[CanonicalActiveLearningTrainingRow] = []
        for proposal_index, decision in enumerate(decisions, start=1):
            for scale in CANONICAL_FEATURE_CROP_SCALES:
                embedding = np.zeros(2048, dtype=np.float32)
                embedding[decision.label] = 1.0
                embedding.setflags(write=False)
                rows.append(
                    CanonicalActiveLearningTrainingRow(
                        proposal_id=decision.proposal_id,
                        proposal_sha256=decision.proposal_sha256,
                        image_id=decision.image_id,
                        image_sha256=decision.image_sha256,
                        group_id=decision.group_id,
                        raw_feature=CanonicalRawFeature(
                            proposal_index=proposal_index,
                            scale=scale,
                            predicted_iou=0.9,
                            stability_score=0.95,
                            values={
                                name: float(proposal_index)
                                for name in CANONICAL_RAW_FEATURE_ORDER
                            },
                            embedding=embedding,
                        ),
                        source_sha256="9" * 64,
                    )
                )
        payloads = serialize_canonical_feature_state(state)
        latest = {decision.proposal_id: decision for decision in decisions}
        training_rows_sha256 = _training_row_hash(rows, latest)
        train_groups = frozenset({"g0", "g1", "g2", "g3", "g4"})
        provenance = _expected_bundle_provenance(
            rows,
            latest,
            train_groups,
            decision_log_sha256="1" * 64,
            split_sha256="2" * 64,
            training_rows_sha256=training_rows_sha256,
            state=state,
            profile=profile,
            device=device,
        )
        return CanonicalActiveLearningRetrainRequest(
            round_number=1,
            feature_state=state,
            effective_decisions=decisions,
            training_rows=tuple(rows),
            train_groups=train_groups,
            test_groups=frozenset({"g5"}),
            decision_log_sha256="1" * 64,
            training_rows_sha256=training_rows_sha256,
            source_bundle_sha256=verify_model_bundle(source_bundle).bundle_sha256,
            frozen_split_sha256="2" * 64,
            prototype_sha256=hashlib.sha256(payloads.prototype_npy).hexdigest(),
            pca_components_sha256=hashlib.sha256(
                payloads.pca_components_npy
            ).hexdigest(),
            pca_mean_sha256=hashlib.sha256(payloads.pca_mean_npy).hexdigest(),
            expected_bundle_provenance=provenance,
            profile=profile,
            device=device,
        )

    def test_gpu_retrain_uses_real_v2_writer_and_portable_scorer(self) -> None:
        source_state = _state(0)
        source_bundle = self._source_bundle(source_state)
        request = self._request(source_bundle, _state(1))
        classifier = _classifier()
        calls: list[dict[str, object]] = []

        def fake_search(
            feature_rows: object,
            labels: object,
            groups: object,
            actions: object,
            weights: object,
            *,
            scales: object,
            split: object,
            config: object,
        ) -> object:
            calls.append(
                {
                    "feature_rows": feature_rows,
                    "labels": labels,
                    "groups": groups,
                    "actions": actions,
                    "weights": weights,
                    "scales": scales,
                    "split": split,
                    "config": config,
                }
            )
            return SimpleNamespace(
                classifier=classifier,
                imputer_statistics=(0.0,) * len(CANONICAL_FEATURE_ORDER),
                feature_order=CANONICAL_FEATURE_ORDER,
                fixed_threshold=0.5,
                split=split,
            )

        output = self.root / "retrained_bundle"
        with mock.patch.object(adapter, "train_canonical_xgb", side_effect=fake_search):
            source_before = verify_model_bundle(source_bundle)
            with self.assertRaisesRegex(PublicIOError, "overlaps"):
                adapter.retrain_canonical_active_learning_bundle(
                    request,
                    source_bundle / "nested_bundle",
                    source_bundle_root=source_bundle,
                )
            source_after_overlap = verify_model_bundle(source_bundle)
            self.assertEqual(source_after_overlap.bundle_sha256, source_before.bundle_sha256)
            self.assertEqual(source_after_overlap.tree_identity, source_before.tree_identity)
            self.assertFalse((source_bundle / "nested_bundle").exists())
            adapter.retrain_canonical_active_learning_bundle(
                request,
                output,
                source_bundle_root=source_bundle,
            )
            with self.assertRaisesRegex(PublicIOError, "already exists"):
                adapter.retrain_canonical_active_learning_bundle(
                    request,
                    output,
                    source_bundle_root=source_bundle,
                )
            forged_hash = "f" * 64
            forged_request = replace(
                request,
                training_rows_sha256=forged_hash,
                expected_bundle_provenance={
                    **request.expected_bundle_provenance,
                    "raw_features_sha256": forged_hash,
                },
            )
            with self.assertRaisesRegex(PublicIOError, "training-row receipt"):
                adapter.retrain_canonical_active_learning_bundle(
                    forged_request,
                    self.root / "forged_hash_bundle",
                    source_bundle_root=source_bundle,
                )
            stale_state = _state(0)
            stale_payloads = serialize_canonical_feature_state(stale_state)
            stale_request = replace(
                request,
                feature_state=stale_state,
                prototype_sha256=hashlib.sha256(
                    stale_payloads.prototype_npy
                ).hexdigest(),
            )
            with self.assertRaisesRegex(PublicIOError, "not recomputed"):
                adapter.retrain_canonical_active_learning_bundle(
                    stale_request,
                    self.root / "stale_prototype_bundle",
                    source_bundle_root=source_bundle,
                )
            forged_provenance = replace(
                request,
                expected_bundle_provenance={
                    **request.expected_bundle_provenance,
                    "feature_state_fit_group_sha256": "0" * 64,
                },
            )
            with self.assertRaisesRegex(PublicIOError, "expected provenance"):
                adapter.retrain_canonical_active_learning_bundle(
                    forged_provenance,
                    self.root / "forged_provenance_bundle",
                    source_bundle_root=source_bundle,
                )
            malformed_values = dict(request.training_rows[0].raw_feature.values)
            malformed_values["unexpected"] = 0.0
            malformed_first = replace(
                request.training_rows[0],
                raw_feature=replace(
                    request.training_rows[0].raw_feature,
                    values=malformed_values,
                ),
            )
            malformed_request = replace(
                request,
                training_rows=(malformed_first, *request.training_rows[1:]),
            )
            with self.assertRaisesRegex(PublicIOError, "raw feature schema"):
                adapter.retrain_canonical_active_learning_bundle(
                    malformed_request,
                    self.root / "malformed_features_bundle",
                    source_bundle_root=source_bundle,
                )
            mixed_source_row = replace(
                request.training_rows[0],
                source_sha256="8" * 64,
            )
            mixed_source_request = replace(
                request,
                training_rows=(mixed_source_row, *request.training_rows[1:]),
            )
            with self.assertRaisesRegex(PublicIOError, "source closure"):
                adapter.retrain_canonical_active_learning_bundle(
                    mixed_source_request,
                    self.root / "mixed_source_bundle",
                    source_bundle_root=source_bundle,
                )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["split"].train_groups, request.train_groups)
        self.assertEqual(calls[0]["split"].test_groups, request.test_groups)
        self.assertEqual(calls[0]["config"].device, "cuda")
        self.assertEqual(len(calls[0]["feature_rows"]), len(request.training_rows))
        verified = verify_model_bundle(output)
        source = verify_model_bundle(source_bundle)
        self.assertEqual(
            verified.provenance["details"],
            dict(request.expected_bundle_provenance),
        )
        self.assertEqual(verified.pca_components_npy, source.pca_components_npy)
        self.assertEqual(verified.pca_mean_npy, source.pca_mean_npy)
        self.assertNotEqual(verified.prototype_npy, source.prototype_npy)
        predictor = load_portable_predictor(verified.classifier, verified.imputer)
        score_rows = tuple(
            finalize_canonical_feature(row.raw_feature, request.feature_state)
            for row in request.training_rows
            if row.raw_feature.scale == 1.0
        )
        probabilities = predict_portable_probabilities(predictor, score_rows)
        self.assertEqual(len(probabilities), len(request.effective_decisions))
        self.assertTrue(all(0.0 <= float(value) <= 1.0 for value in probabilities))

    def test_gpu_retrain_propagates_cuda_to_training_parity_and_bundle(self) -> None:
        source_bundle = self._source_bundle(
            _state(0),
            profile=CANONICAL_GPU_PROFILE,
        )
        request = self._request(
            source_bundle,
            _state(1),
            profile=CANONICAL_GPU_PROFILE,
            device="cuda",
        )
        classifier = _classifier()
        observed: dict[str, object] = {}

        def fake_search(
            feature_rows: object,
            labels: object,
            groups: object,
            actions: object,
            weights: object,
            *,
            scales: object,
            split: object,
            config: object,
        ) -> object:
            observed["training_device"] = config.device
            observed["training_host_threads"] = config.thread_count
            return SimpleNamespace(
                classifier=classifier,
                imputer_statistics=(0.0,) * len(CANONICAL_FEATURE_ORDER),
                feature_order=CANONICAL_FEATURE_ORDER,
                fixed_threshold=0.5,
                split=split,
            )

        def fake_parity(*_args: object, **kwargs: object) -> dict[str, object]:
            observed["parity_device"] = kwargs.get("device")
            return {"status": "PASS", "device": kwargs.get("device")}

        output = self.root / "retrained_bundle_gpu"
        with mock.patch.object(
            adapter,
            "train_canonical_xgb",
            side_effect=fake_search,
        ), mock.patch.object(
            adapter,
            "verify_probability_parity",
            side_effect=fake_parity,
        ):
            adapter.retrain_canonical_active_learning_bundle(
                request,
                output,
                source_bundle_root=source_bundle,
            )

        self.assertEqual(
            observed,
            {
                "training_device": "cuda",
                "training_host_threads": 1,
                "parity_device": "cuda",
            },
        )
        verified = verify_model_bundle(output)
        self.assertEqual(verified.profile, CANONICAL_GPU_PROFILE)
        self.assertEqual(verified.normalized_config["device"], "cuda")
        self.assertEqual(
            (
                verified.provenance["details"]["profile"],
                verified.provenance["details"]["device"],
            ),
            (CANONICAL_GPU_PROFILE, "cuda"),
        )

    def test_cpu_retrain_request_is_rejected_before_output_or_training(self) -> None:
        source_bundle = self._source_bundle(_state(0))
        gpu_request = self._request(source_bundle, _state(1))
        cpu_request = replace(
            gpu_request,
            profile=CANONICAL_PROFILE,
            device="cpu",
            expected_bundle_provenance={
                key: value
                for key, value in gpu_request.expected_bundle_provenance.items()
                if key not in {"profile", "device"}
            },
        )
        output = self.root / "rejected_cpu_retrain"
        with mock.patch.object(adapter, "train_canonical_xgb") as trainer:
            with self.assertRaisesRegex(PublicIOError, "requires the GPU/CUDA profile"):
                adapter.retrain_canonical_active_learning_bundle(
                    cpu_request,
                    output,
                    source_bundle_root=source_bundle,
                )
        trainer.assert_not_called()
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(f".{output.name}.bundle.*")), [])


if __name__ == "__main__":
    unittest.main()
