from __future__ import annotations

import unittest

from compag_curation.canonical.spec import (
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_GPU_PROFILE,
)
from compag_curation.model_bundle import (
    POST_R92_TRANSFER_CV_INTERPRETATION,
    POST_R92_TRANSFER_FEATURE_STATE_SCOPE,
    POST_R92_TRANSFER_FROZEN_STATE_POLICY,
    POST_R92_TRANSFER_LINEAGE_POLICY,
    POST_R92_TRANSFER_PCA_VARIANCE_POLICY,
    POST_R92_TRANSFER_REPRODUCTION_CLAIM,
    POST_R92_TRANSFER_WORKFLOW,
    _validate_v2_details,
)
from compag_curation.public_io import PublicIOError


class ModelBundleTransferLineageTests(unittest.TestCase):
    def _compatibility(self) -> dict[str, object]:
        return {
            "classifier_format": "XGBOOST_UBJ",
            "no_pickle_or_joblib": True,
            "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
            "fixed_threshold": 0.5,
        }

    def _provenance(self) -> dict[str, object]:
        digest = "a" * 64
        return {
            "profile": CANONICAL_GPU_PROFILE,
            "device": "cuda",
            "reviewed_sha256": digest,
            "split_manifest_sha256": digest,
            "raw_features_sha256": digest,
            "feature_state_fit_group_sha256": digest,
            "feature_state_fit_rows_sha256": digest,
            "feature_state_fit_scope": POST_R92_TRANSFER_FEATURE_STATE_SCOPE,
            "validation_feature_scale": 1.0,
            "cv_score_interpretation": POST_R92_TRANSFER_CV_INTERPRETATION,
            "training_row_count": 128,
            "positive_training_row_count": 32,
            "fixed_threshold": 0.5,
            "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
            "ubj_parity_status": "PASS",
            "paper_result_reproduction": "NOT_CLAIMED",
            "active_learning_workflow": POST_R92_TRANSFER_WORKFLOW,
            "lineage_policy": POST_R92_TRANSFER_LINEAGE_POLICY,
            "reproduction_claim": POST_R92_TRANSFER_REPRODUCTION_CLAIM,
            "model_label": "project-r1",
            "active_learning_round_number": 1,
            "parent_model_identity_sha256": digest,
            "transfer_baseline_sha256": digest,
            "transfer_split_sha256": digest,
            "r92_resource_manifest_sha256": digest,
            "r92_classifier_sha256": digest,
            "r92_feature_state_sha256": digest,
            "frozen_feature_state_policy": POST_R92_TRANSFER_FROZEN_STATE_POLICY,
            "pca_explained_variance_policy": POST_R92_TRANSFER_PCA_VARIANCE_POLICY,
            "round_image_id": digest,
            "round_image_sha256": digest,
            "round_group_id": "img_9475",
            "cumulative_al_feature_archive_sha256": digest,
            "effective_train_groups_sha256": digest,
            "frozen_test_groups_sha256": digest,
        }

    def _validate(self, provenance: dict[str, object]) -> tuple[int, int]:
        return _validate_v2_details(
            self._compatibility(),
            provenance,
            profile=CANONICAL_GPU_PROFILE,
            feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
        )

    def test_accepts_exact_post_r92_transfer_lineage(self) -> None:
        self.assertEqual(self._validate(self._provenance()), (128, 32))

    def test_rejects_sparse_or_extra_transfer_lineage(self) -> None:
        base = self._provenance()
        missing = dict(base)
        missing.pop("transfer_split_sha256")
        extra = {**base, "r92_reproduction": True}
        for provenance in (missing, extra):
            with self.subTest(fields=set(provenance)):
                with self.assertRaisesRegex(PublicIOError, "field closure"):
                    self._validate(provenance)

    def test_rejects_transfer_policy_or_frozen_state_drift(self) -> None:
        base = self._provenance()
        cases = (
            ({**base, "reproduction_claim": "R92_REPRODUCTION"}, "identity"),
            (
                {**base, "feature_state_fit_scope": "OUTER_TRAIN_ROWS_BEFORE_CV"},
                "feature-state fit scope",
            ),
            (
                {
                    **base,
                    "cv_score_interpretation": (
                        "CONDITIONAL_ON_OUTER_TRAIN_PREFIT_PROTOTYPE_AND_PCA"
                    ),
                },
                "feature-state fit scope",
            ),
            (
                {
                    **base,
                    "pca_explained_variance_policy": "HISTORICAL_R92_FIT_VARIANCE",
                },
                "identity",
            ),
            ({**base, "r92_classifier_sha256": "A" * 64}, "r92_classifier"),
            ({**base, "model_label": "project-r2"}, "identity"),
        )
        for provenance, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(PublicIOError, pattern):
                    self._validate(provenance)


if __name__ == "__main__":
    unittest.main()
