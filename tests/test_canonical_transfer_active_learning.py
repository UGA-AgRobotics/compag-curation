from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from compag_curation.canonical import active_learning_facade as facade
from compag_curation.canonical.active_learning import (
    CanonicalActiveLearningDecision,
    CanonicalActiveLearningPoolRow,
    CanonicalTransferBaselineReference,
    CanonicalTransferScorerReference,
    _feature_archive_hash,
    _feature_state_hashes,
    _parse_transfer_decision_rows,
    _transfer_decision_csv_row,
    _training_row_hash,
    begin_canonical_transfer_image_round,
    resume_canonical_transfer_image_round,
    write_canonical_active_learning_pool,
)
from compag_curation.canonical.active_learning_service import (
    read_canonical_r92_transfer_inference,
    train_canonical_transfer_request,
)
from compag_curation.assets import asset_ids_for_profile, asset_registry
from compag_curation.canonical.transfer_baseline import TransferBaselineRow
from compag_curation.review.exchange import ReviewRow
from compag_curation.canonical.spec import (
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_GPU_PROFILE,
)
from compag_curation.model_bundle import BUNDLE_SCHEMA_V2
from compag_curation.public_io import (
    PublicIOError,
    compact_json_sha256,
    manifest_rows,
    write_new_json,
)

from tests.test_canonical_active_learning import (
    _raw_rows,
    _reviewed_copy,
    _sha,
    _state,
)


def _same_image_rows(
    count: int,
    group_id: str,
    *,
    probability: float = 0.5,
    image_name: str = "new-card",
) -> list[CanonicalActiveLearningPoolRow]:
    image_id = _sha(f"image:{image_name}")
    image_sha256 = _sha(f"image-bytes:{image_name}")
    rows: list[CanonicalActiveLearningPoolRow] = []
    for index in range(count):
        proposal_id = _sha(f"proposal:{image_name}:{index:04d}")
        rows.append(
            CanonicalActiveLearningPoolRow(
                proposal_id=proposal_id,
                proposal_sha256=proposal_id,
                image_id=image_id,
                image_sha256=image_sha256,
                group_id=group_id,
                scale=1.0,
                xgb_p=probability,
            )
        )
    return rows


def _baseline_reference() -> CanonicalTransferBaselineReference:
    source_groups = frozenset({"g0", "g1", "g2", "g3", "g4", "g5"})
    return CanonicalTransferBaselineReference(
        archive_sha256=_sha("transfer-baseline"),
        source_sha256=_sha("legacy-source"),
        source_size_bytes=582_968_556,
        split_sha256=_sha("transfer-split"),
        source_groups_sha256=compact_json_sha256(sorted(source_groups)),
        feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
        row_count=40,
        proposal_count=10,
        source_groups=source_groups,
        train_groups=frozenset({"g0", "g1", "g2", "g3", "g4"}),
        test_groups=frozenset({"g5"}),
    )


def _scorer(kind: str, state: object, scorer_sha256: str) -> CanonicalTransferScorerReference:
    hashes = _feature_state_hashes(state)
    return CanonicalTransferScorerReference(
        kind=kind,
        scorer_sha256=scorer_sha256,
        scores_sha256="0" * 64,
        pool_scores_sha256="0" * 64,
        feature_state_sha256=_sha("published-r92-feature-state"),
        prototype_sha256=hashes["prototype_sha256"],
        pca_components_sha256=hashes["pca_components_sha256"],
        pca_mean_sha256=hashes["pca_mean_sha256"],
        r92_classifier_sha256=_sha("r92-classifier"),
        r92_resource_manifest_sha256=_sha("r92-resources"),
        preset_id="compag-cj-r92" if kind == "PUBLISHED_R92_TRANSFER_SCORER" else None,
    )


def _reviewed_with_action(
    request: Path,
    reviewed: Path,
    *,
    action: str,
    weight: str,
) -> None:
    with request.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    with reviewed.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            row["label"] = "1" if index % 2 == 0 else "0"
            row["review_action"] = action
            row["review_weight"] = weight
            row["review_status"] = "reviewed"
            writer.writerow(row)


def _registered_transfer_assets() -> list[dict[str, object]]:
    roles = {
        "sam2.1-hiera-large-checkpoint": "SAM2 checkpoint",
        "sam2.1-hiera-large-config": "SAM2 configuration",
        "resnet50-imagenet1k-v2-weights": "ResNet50 weights",
        "sam2-apache-license": "SAM2 license",
        "torchvision-bsd-license": "torchvision license",
    }
    registry = asset_registry()
    return [
        {
            "role": roles[asset_id],
            "asset_id": asset_id,
            "filename": registry[asset_id].filename,
            "sha256": registry[asset_id].sha256,
            "size_bytes": registry[asset_id].size_bytes,
        }
        for asset_id in sorted(asset_ids_for_profile(CANONICAL_GPU_PROFILE))
    ]


class _TransferBundleHarness:
    def __init__(self) -> None:
        self.states: dict[Path, object] = {}
        self.provenance: dict[Path, dict[str, object]] = {}

    def callback(self, observed: list[object]):
        def write(request: object, output: Path) -> None:
            observed.append(request)
            output.mkdir()
            (output / "bundle.json").write_bytes(
                f"project-r{request.round_number}\n".encode("ascii")
            )
            self.states[output.resolve()] = request.feature_state
            self.provenance[output.resolve()] = dict(
                request.expected_bundle_provenance
            )

        return write

    def verify(self, root: Path) -> object:
        resolved = root.resolve(strict=True)
        state = self.states[resolved]
        digest = hashlib.sha256((resolved / "bundle.json").read_bytes()).hexdigest()
        return SimpleNamespace(
            schema=BUNDLE_SCHEMA_V2,
            profile=CANONICAL_GPU_PROFILE,
            threshold=0.5,
            feature_order=CANONICAL_FEATURE_ORDER,
            feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
            prototype=state.prototype,
            pca_components=state.pca_components,
            pca_mean=state.pca_mean,
            bundle_sha256=digest,
            provenance={"details": self.provenance[resolved]},
        )


class CanonicalTransferActiveLearningTests(unittest.TestCase):
    def test_transfer_scorer_kind_changes_after_round_one(self) -> None:
        state = _state()
        published = _scorer(
            "PUBLISHED_R92_TRANSFER_SCORER",
            state,
            _sha("published-scorer"),
        )
        project = _scorer(
            "PROJECT_TRANSFER_BUNDLE",
            state,
            _sha("project-r1"),
        )
        pool_rows = (
            _same_image_rows(1, "new-a", image_name="new-a")[0],
            _same_image_rows(1, "new-b", image_name="new-b")[0],
        )
        reviews = tuple(
            ReviewRow(
                row.proposal_id,
                row.proposal_sha256,
                row.image_id,
                row.image_sha256,
                row.group_id,
                index % 2,
                "reviewed",
                "accept",
                1.0,
            )
            for index, row in enumerate(pool_rows)
        )
        first = _transfer_decision_csv_row(
            reviews[0],
            sequence=1,
            round_number=1,
            timestamp_utc="2026-09-02T01:00:00Z",
            scorer=published,
            round_manifest_sha256=_sha("round-1"),
            reviewed_sha256=_sha("reviewed-1"),
        )
        invalid_second = _transfer_decision_csv_row(
            reviews[1],
            sequence=2,
            round_number=2,
            timestamp_utc="2026-09-02T02:00:00Z",
            scorer=published,
            round_manifest_sha256=_sha("round-2"),
            reviewed_sha256=_sha("reviewed-2"),
        )
        with self.assertRaisesRegex(
            PublicIOError,
            "published r92 may score only the first transfer round",
        ):
            _parse_transfer_decision_rows((first, invalid_second))
        valid_second = _transfer_decision_csv_row(
            reviews[1],
            sequence=2,
            round_number=2,
            timestamp_utc="2026-09-02T02:00:00Z",
            scorer=project,
            round_manifest_sha256=_sha("round-2"),
            reviewed_sha256=_sha("reviewed-2"),
        )
        self.assertEqual(
            len(_parse_transfer_decision_rows((first, valid_second))),
            2,
        )

    def test_two_transfer_rounds_bind_project_parent_and_cumulative_ancestry(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_root = root / "baseline"
            baseline_root.mkdir()
            baseline = _baseline_reference()
            state = _state()
            source_sha256 = _sha("raw-feature-source")
            harness = _TransferBundleHarness()
            observed: list[object] = []

            first_rows = _same_image_rows(3, "g6", image_name="round-1")
            first_features = tuple(_raw_rows(first_rows))
            first_scorer = _scorer(
                "PUBLISHED_R92_TRANSFER_SCORER",
                state,
                _sha("published-r92-transfer-scorer"),
            )
            first_pool = root / "round-1-pool"
            first_pool_result = write_canonical_active_learning_pool(
                first_rows,
                first_pool,
                bundle_sha256=first_scorer.scorer_sha256,
                source_sha256=source_sha256,
                feature_archive_sha256=_feature_archive_hash(first_features),
                profile=CANONICAL_GPU_PROFILE,
            )
            first_scorer = CanonicalTransferScorerReference(
                **{
                    **first_scorer.__dict__,
                    "pool_scores_sha256": first_pool_result["predictions_sha256"],
                }
            )
            first_selection = root / "round-1-selection"
            begin_canonical_transfer_image_round(
                first_pool,
                baseline,
                first_scorer,
                first_selection,
                round_number=1,
                created_at_utc="2026-09-02T01:00:00Z",
            )
            first_reviewed = root / "round-1-reviewed.csv"
            _reviewed_copy(
                first_selection / "review_request.csv",
                first_reviewed,
            )
            first_completion = root / "round-1-completion"
            with mock.patch(
                "compag_curation.model_bundle.verify_model_bundle",
                side_effect=harness.verify,
            ):
                first_result = resume_canonical_transfer_image_round(
                    first_selection,
                    first_pool,
                    baseline_root,
                    baseline,
                    first_scorer,
                    first_reviewed,
                    state,
                    first_features,
                    first_features,
                    first_completion,
                    completed_at_utc="2026-09-02T02:00:00Z",
                    retrain=harness.callback(observed),
                )

            second_rows = _same_image_rows(4, "g7", image_name="round-2")
            second_features = tuple(_raw_rows(second_rows))
            second_scorer = _scorer(
                "PROJECT_TRANSFER_BUNDLE",
                state,
                str(first_result["outputs"]["new_bundle_sha256"]),
            )
            second_pool = root / "round-2-pool"
            second_pool_result = write_canonical_active_learning_pool(
                second_rows,
                second_pool,
                bundle_sha256=second_scorer.scorer_sha256,
                source_sha256=source_sha256,
                feature_archive_sha256=_feature_archive_hash(second_features),
                profile=CANONICAL_GPU_PROFILE,
            )
            second_scorer = CanonicalTransferScorerReference(
                **{
                    **second_scorer.__dict__,
                    "scores_sha256": _sha("round-2-stage60-predictions"),
                    "pool_scores_sha256": second_pool_result["predictions_sha256"],
                }
            )
            prior_completion = first_completion / "round_completion.json"
            prior_log = first_completion / "decision_log.csv"
            changed_log = root / "changed-decision-log.csv"
            changed_log.write_bytes(prior_log.read_bytes() + b"\n")
            with self.assertRaises(PublicIOError):
                begin_canonical_transfer_image_round(
                    second_pool,
                    baseline,
                    second_scorer,
                    root / "invalid-round-2-selection",
                    round_number=2,
                    created_at_utc="2026-09-02T03:00:00Z",
                    prior_round_completion_path=prior_completion,
                    decision_log_path=changed_log,
                )
            self.assertFalse((root / "invalid-round-2-selection").exists())

            second_selection = root / "round-2-selection"
            second_pause = begin_canonical_transfer_image_round(
                second_pool,
                baseline,
                second_scorer,
                second_selection,
                round_number=2,
                created_at_utc="2026-09-02T03:00:00Z",
                prior_round_completion_path=prior_completion,
                decision_log_path=prior_log,
            )
            self.assertEqual(
                second_pause["chain"]["source_scorer"]["kind"],
                "PROJECT_TRANSFER_BUNDLE",
            )
            self.assertEqual(
                second_pause["chain"]["source_scorer"]["scorer_sha256"],
                first_result["outputs"]["new_bundle_sha256"],
            )
            second_reviewed = root / "round-2-reviewed.csv"
            _reviewed_copy(
                second_selection / "review_request.csv",
                second_reviewed,
            )
            second_completion = root / "round-2-completion"
            with mock.patch(
                "compag_curation.model_bundle.verify_model_bundle",
                side_effect=harness.verify,
            ):
                second_result = resume_canonical_transfer_image_round(
                    second_selection,
                    second_pool,
                    baseline_root,
                    baseline,
                    second_scorer,
                    second_reviewed,
                    state,
                    first_features + second_features,
                    second_features,
                    second_completion,
                    completed_at_utc="2026-09-02T04:00:00Z",
                    retrain=harness.callback(observed),
                    prior_round_completion_path=prior_completion,
                    decision_log_path=prior_log,
                )
            self.assertEqual(second_result["round_number"], 2)
            self.assertEqual(second_result["counts"]["decision_log_rows"], 7)
            self.assertEqual(
                second_result["counts"]["cumulative_al_feature_rows"],
                28,
            )
            self.assertEqual(len(observed), 2)
            self.assertEqual(
                observed[-1].expected_bundle_provenance["model_label"],
                "project-r2",
            )
            self.assertEqual(
                observed[-1].expected_bundle_provenance[
                    "parent_model_identity_sha256"
                ],
                first_result["outputs"]["new_bundle_sha256"],
            )

    def test_transfer_inference_output_is_disjoint_from_inputs(self) -> None:
        from compag_curation.canonical.transfer_inference import _safe_output

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "one-image"
            assets = root / "assets"
            images.mkdir()
            assets.mkdir()
            config = root / "project.toml"
            config.write_text("fixture = true\n", encoding="ascii")
            final, staging = _safe_output(
                root / "result",
                protected_inputs=(images, config, assets),
            )
            self.assertEqual(final, root / "result")
            self.assertEqual(staging.parent, root)
            for overlapping in (images / "result", assets / "result"):
                with self.assertRaisesRegex(PublicIOError, "overlaps"):
                    _safe_output(
                        overlapping,
                        protected_inputs=(images, config, assets),
                    )

    def test_r92_reader_rejects_resealed_arbitrary_scorer_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = {
                "image_name": "new-card.ppm",
                "image_id": _sha("image-id"),
                "image_sha256": _sha("image-bytes"),
                "group_id": "new-card",
                "size_bytes": 123,
                "width": 64,
                "height": 64,
                "tile_count": 1,
                "candidate_rows": 1,
            }
            result_path = root / "inference_result.json"
            write_new_json(result_path, {"input_inventory": [image]})
            result_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
            arbitrary_scorer = _sha("attacker-selected-scorer")
            predictions_sha256 = _sha("predictions")
            raw_features_sha256 = _sha("raw-features")
            assets = _registered_transfer_assets()
            receipt = {
                "schema": "compag-curation-r92-transfer-image-inference/v1",
                "status": "PASS",
                "workflow": "SINGLE_NEW_IMAGE_POST_R92_TRANSFER_V1",
                "lineage_policy": "POST_R92_REVIEWED_TRANSFER_BASELINE",
                "reproduction_claim": "NOT_R92_REPRODUCTION",
                "preset_id": "compag-cj-r92",
                "profile": CANONICAL_GPU_PROFILE,
                "device": "cuda",
                "threshold": 0.5,
                "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
                "resource_manifest_sha256": _sha("r92-resources"),
                "classifier_sha256": _sha("r92-classifier"),
                "feature_state_sha256": _sha("r92-state"),
                # A forged receipt and a resealed inference result agree with
                # each other, but not with the canonical identity formula.
                "scorer_identity_sha256": arbitrary_scorer,
                "config_sha256": _sha("config"),
                "asset_inventory": assets,
                "asset_inventory_sha256": compact_json_sha256(assets),
                "image_inventory_sha256": compact_json_sha256([image]),
                "image": image,
                "predictions_sha256": predictions_sha256,
                "raw_features_sha256": raw_features_sha256,
                "inference_result_sha256": result_sha256,
            }
            write_new_json(root / "transfer_scorer.json", receipt)
            archive = SimpleNamespace(
                profile=CANONICAL_GPU_PROFILE,
                device="cuda",
                bundle_sha256=arbitrary_scorer,
                predictions_sha256=predictions_sha256,
                raw_features_sha256=raw_features_sha256,
                inference_result_sha256=result_sha256,
            )
            with mock.patch(
                "compag_curation.canonical.active_learning_service."
                "read_canonical_active_learning_inference",
                return_value=archive,
            ):
                with self.assertRaisesRegex(
                    PublicIOError,
                    "identity differs from its preset and public assets",
                ):
                    read_canonical_r92_transfer_inference(root)

    def test_baseline_blank_actions_are_only_numeric_weight_adapted(self) -> None:
        from compag_curation.canonical.active_learning import (
            CanonicalTransferRetrainRequest,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_root = root / "baseline"
            baseline_root.mkdir()
            split_path = baseline_root / "split.json"
            split_path.write_bytes(b"{}\n")
            source_groups = frozenset({"g0", "g1", "g2", "g3", "g4", "g5"})
            baseline = CanonicalTransferBaselineReference(
                archive_sha256=compact_json_sha256(manifest_rows(baseline_root)),
                source_sha256=_sha("legacy-source"),
                source_size_bytes=582_968_556,
                split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                source_groups_sha256=compact_json_sha256(sorted(source_groups)),
                feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
                row_count=40,
                proposal_count=10,
                source_groups=source_groups,
                train_groups=frozenset({"g0", "g1", "g2", "g3", "g4"}),
                test_groups=frozenset({"g5"}),
            )
            baseline_rows: list[TransferBaselineRow] = []
            groups = ("g0", "g1", "g2", "g3", "g4", "g5", "g0", "g1", "g2", "g3")
            for proposal_index, group_id in enumerate(groups):
                proposal_id = _sha(f"baseline:{proposal_index}")
                for scale in (0.67, 0.8, 1.0, 1.25):
                    reviewed = proposal_index == 0
                    baseline_rows.append(
                        TransferBaselineRow(
                            proposal_id=proposal_id,
                            group_id=group_id,
                            scale=scale,
                            features=tuple(
                                float(proposal_index + feature_index / 1000)
                                for feature_index in range(len(CANONICAL_FEATURE_ORDER))
                            ),
                            label=proposal_index % 2,
                            reviewed=reviewed,
                            review_action="sus_flip" if reviewed else "",
                            review_weight=0.4 if reviewed else 1.0,
                        )
                    )
            candidate = _same_image_rows(1, "g6")[0]
            al_rows = tuple(_raw_rows([candidate]))
            decision = CanonicalActiveLearningDecision(
                sequence=1,
                phase="ACTIVE_LEARNING_ROUND",
                round_number=1,
                decision_timestamp_utc="2026-09-02T02:00:00Z",
                proposal_id=candidate.proposal_id,
                proposal_sha256=candidate.proposal_sha256,
                image_id=candidate.image_id,
                image_sha256=candidate.image_sha256,
                group_id=candidate.group_id,
                label=1,
                review_action="accept",
                review_weight=1.0,
            )
            request = CanonicalTransferRetrainRequest(
                round_number=1,
                baseline_root=baseline_root,
                baseline=baseline,
                feature_state=_state(),
                effective_decisions=(decision,),
                cumulative_al_rows=al_rows,
                train_groups=frozenset({"g0", "g1", "g2", "g3", "g4", "g6"}),
                test_groups=frozenset({"g5"}),
                decision_log_sha256=_sha("decisions"),
                cumulative_al_rows_sha256=_training_row_hash(
                    al_rows, {decision.proposal_id: decision}
                ),
                source_scorer_sha256=_sha("scorer"),
                selection_manifest_sha256=_sha("selection"),
                expected_bundle_provenance={},
                profile=CANONICAL_GPU_PROFILE,
                device="cuda",
            )
            fake_baseline = SimpleNamespace(
                root=baseline_root,
                split_path=split_path,
                source_sha256=baseline.source_sha256,
                source_size_bytes=baseline.source_size_bytes,
                source_groups=source_groups,
                train_groups=baseline.train_groups,
                test_groups=baseline.test_groups,
                row_count=baseline.row_count,
                proposal_count=baseline.proposal_count,
            )
            captured: dict[str, object] = {}

            def fake_train(
                feature_rows: object,
                labels: object,
                row_groups: object,
                actions: object,
                weights: object,
                *,
                scales: object,
                split: object,
                config: object,
            ) -> object:
                captured.update(
                    actions=tuple(actions),
                    weights=tuple(weights),
                    groups=tuple(row_groups),
                    config=config,
                )
                return SimpleNamespace(
                    split=split,
                    feature_order=CANONICAL_FEATURE_ORDER,
                    fixed_threshold=0.5,
                )

            finalized = {name: float(index) for index, name in enumerate(CANONICAL_FEATURE_ORDER)}
            with mock.patch(
                "compag_curation.canonical.transfer_baseline.verify_transfer_baseline",
                return_value=fake_baseline,
            ), mock.patch(
                "compag_curation.canonical.transfer_baseline.iter_transfer_baseline_rows",
                return_value=iter(baseline_rows),
            ), mock.patch(
                "compag_curation.canonical.active_learning_service.finalize_canonical_feature",
                return_value=finalized,
            ), mock.patch(
                "compag_curation.canonical.active_learning_service.train_canonical_xgb",
                side_effect=fake_train,
            ):
                result = train_canonical_transfer_request(request)
            self.assertEqual(captured["actions"][:4], ("sus_flip",) * 4)
            self.assertNotIn("", captured["actions"])
            self.assertEqual(captured["actions"][4:40], ("accept",) * 36)
            self.assertEqual(captured["weights"][4:40], (1.0,) * 36)
            self.assertEqual(result.reviewed_baseline_feature_rows, 4)
            self.assertEqual(result.unreviewed_baseline_feature_rows, 36)
            self.assertIn(
                "NUMERIC_WEIGHT_VALIDATION_ONLY",
                result.baseline_action_adapter_policy,
            )

    def test_first_round_is_one_image_top50_and_fresh_project_r1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_root = root / "baseline"
            baseline_root.mkdir()
            baseline = _baseline_reference()
            state = _state()
            pool_rows = _same_image_rows(55, "g6")
            pool_features = tuple(_raw_rows(pool_rows))
            scorer = _scorer(
                "PUBLISHED_R92_TRANSFER_SCORER",
                state,
                _sha("r92-transfer-scorer"),
            )
            pool_root = root / "pool"
            write_canonical_active_learning_pool(
                pool_rows,
                pool_root,
                bundle_sha256=scorer.scorer_sha256,
                source_sha256=_sha("raw-feature-source"),
                feature_archive_sha256=_feature_archive_hash(pool_features),
                profile=CANONICAL_GPU_PROFILE,
            )
            scorer = CanonicalTransferScorerReference(
                **{
                    **scorer.__dict__,
                    "pool_scores_sha256": json.loads(
                        (pool_root / "pool_result.json").read_text()
                    )["predictions_sha256"],
                }
            )
            selection = root / "selection"
            paused = begin_canonical_transfer_image_round(
                pool_root,
                baseline,
                scorer,
                selection,
                round_number=1,
                created_at_utc="2026-09-02T01:00:00Z",
            )
            self.assertEqual(paused["counts"]["selected"], 50)
            self.assertEqual(
                paused["chain"]["source_scorer"]["scores_sha256"],
                "0" * 64,
            )
            self.assertEqual(
                paused["chain"]["source_scorer"]["pool_scores_sha256"],
                paused["chain"]["source_predictions_sha256"],
            )
            self.assertEqual(paused["image"]["group_id"], "g6")
            self.assertEqual(
                paused["scientific_lineage"]["reproduction_claim"],
                "NOT_R92_REPRODUCTION",
            )
            with (selection / "review_request.csv").open(
                "r", encoding="ascii", newline=""
            ) as handle:
                selected_ids = {row["proposal_id"] for row in csv.DictReader(handle)}
            reviewed = root / "reviewed.csv"
            _reviewed_copy(selection / "review_request.csv", reviewed)
            cumulative = tuple(
                row for row in pool_features if row.proposal_id in selected_ids
            )
            all_skip = root / "reviewed-all-skip.csv"
            _reviewed_with_action(
                selection / "review_request.csv",
                all_skip,
                action="skip",
                weight="0.0",
            )
            with self.assertRaisesRegex(
                PublicIOError,
                "cannot be all skip",
            ):
                resume_canonical_transfer_image_round(
                    selection,
                    pool_root,
                    baseline_root,
                    baseline,
                    scorer,
                    all_skip,
                    state,
                    cumulative,
                    pool_features,
                    root / "all-skip-completion",
                    completed_at_utc="2026-09-02T02:00:00Z",
                    retrain=lambda _request, _output: None,
                )
            self.assertFalse((root / "all-skip-completion").exists())
            harness = _TransferBundleHarness()
            observed: list[object] = []
            with mock.patch(
                "compag_curation.model_bundle.verify_model_bundle",
                side_effect=harness.verify,
            ):
                completed = resume_canonical_transfer_image_round(
                    selection,
                    pool_root,
                    baseline_root,
                    baseline,
                    scorer,
                    reviewed,
                    state,
                    cumulative,
                    pool_features,
                    root / "completion",
                    completed_at_utc="2026-09-02T02:00:00Z",
                    retrain=harness.callback(observed),
                )
            self.assertEqual(len(observed), 1)
            request = observed[0]
            self.assertEqual(request.round_number, 1)
            self.assertEqual(
                request.expected_bundle_provenance["model_label"], "project-r1"
            )
            self.assertEqual(
                request.expected_bundle_provenance["parent_model_identity_sha256"],
                scorer.scorer_sha256,
            )
            self.assertEqual(completed["counts"]["reviewed_batch_rows"], 50)
            retrain_receipt = json.loads(
                (root / "completion/retrain_request.json").read_text()
            )
            self.assertFalse(retrain_receipt["warm_start"])
            self.assertIn("NUMERIC_WEIGHT_VALIDATION_ONLY", retrain_receipt["baseline_action_adapter_policy"])

    def test_transfer_begin_rejects_multiple_or_baseline_image_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = _baseline_reference()
            state = _state()
            scorer = _scorer(
                "PUBLISHED_R92_TRANSFER_SCORER",
                state,
                _sha("r92-transfer-scorer"),
            )
            for name, rows, message in (
                (
                    "multiple",
                    [*_same_image_rows(1, "g6", image_name="a"), *_same_image_rows(1, "g7", image_name="b")],
                    "exactly one image",
                ),
                ("baseline", _same_image_rows(1, "g0"), "previously unseen"),
            ):
                pool = root / f"{name}-pool"
                write_canonical_active_learning_pool(
                    rows,
                    pool,
                    bundle_sha256=scorer.scorer_sha256,
                    source_sha256=_sha("raw-feature-source"),
                    feature_archive_sha256=_sha(f"{name}-features"),
                    profile=CANONICAL_GPU_PROFILE,
                )
                value = json.loads((pool / "pool_result.json").read_text())
                bound_scorer = CanonicalTransferScorerReference(
                    **{
                        **scorer.__dict__,
                        "pool_scores_sha256": value["predictions_sha256"],
                    }
                )
                with self.assertRaisesRegex(PublicIOError, message):
                    begin_canonical_transfer_image_round(
                        pool,
                        baseline,
                        bound_scorer,
                        root / f"{name}-selection",
                        round_number=1,
                        created_at_utc="2026-09-02T01:00:00Z",
                    )

    def test_facade_publishes_pause_and_resume_with_injected_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage60 = root / "stage60"
            baseline_root = root / "baseline"
            stage60.mkdir()
            baseline_root.mkdir()
            (stage60 / "fixture.txt").write_text("stage60\n", encoding="ascii")
            (baseline_root / "fixture.txt").write_text("baseline\n", encoding="ascii")
            config = root / "project.toml"
            config.write_text("fixture = true\n", encoding="ascii")
            state = _state()
            baseline = _baseline_reference()
            rows = _same_image_rows(3, "g6")
            features = tuple(_raw_rows(rows))
            scorer = _scorer(
                "PUBLISHED_R92_TRANSFER_SCORER",
                state,
                _sha("r92-transfer-scorer"),
            )
            # The facade creates the pool, so bind the scorer to those exact rows.
            scratch = root / "scratch-pool"
            write_canonical_active_learning_pool(
                rows,
                scratch,
                bundle_sha256=scorer.scorer_sha256,
                source_sha256=_sha("raw-feature-source"),
                feature_archive_sha256=_feature_archive_hash(features),
                profile=CANONICAL_GPU_PROFILE,
            )
            pool_result = json.loads((scratch / "pool_result.json").read_text())
            scorer = CanonicalTransferScorerReference(
                **{
                    **scorer.__dict__,
                    "pool_scores_sha256": pool_result["predictions_sha256"],
                }
            )
            archive = SimpleNamespace(
                pool_rows=tuple(rows),
                feature_rows=features,
                raw_features_sha256=_sha("raw-feature-source"),
                feature_archive_sha256=_feature_archive_hash(features),
            )
            context = (archive, scorer, state, None, None, None)
            project_config = SimpleNamespace(config_sha256=_sha("project-config"))
            common = (
                mock.patch.object(
                    facade,
                    "_transfer_baseline_reference",
                    return_value=(object(), baseline),
                ),
                mock.patch.object(
                    facade,
                    "_transfer_round_context",
                    return_value=context,
                ),
                mock.patch.object(
                    facade,
                    "_verify_transfer_project_config",
                    return_value=project_config,
                ),
            )
            for patcher in common:
                patcher.start()
                self.addCleanup(patcher.stop)
            begin_root = root / "begin"
            result, code = facade.begin_transfer_image_round(
                stage60,
                baseline_root,
                1,
                begin_root,
            )
            self.assertEqual((result["status"], code), ("PAUSED_FOR_REVIEW", 3))
            reviewed = root / "reviewed.csv"
            _reviewed_copy(
                begin_root / "payload/selection/review_request.csv",
                reviewed,
            )
            harness = _TransferBundleHarness()
            observed: list[object] = []
            dependency = {"sha256": _sha("science-gpu")}
            with mock.patch(
                "compag_curation.model_bundle.verify_model_bundle",
                side_effect=harness.verify,
            ), mock.patch.object(
                facade, "require_science_dependencies", return_value=dependency
            ), mock.patch.object(
                facade,
                "revalidate_science_gpu_dependencies",
                return_value=dependency,
            ):
                result, code = facade.resume_transfer_image_round(
                    begin_root,
                    stage60,
                    baseline_root,
                    reviewed,
                    1,
                    root / "resume",
                    config=config,
                    retrain=harness.callback(observed),
                )
            self.assertEqual((result["status"], code), ("PASS", 0))
            self.assertEqual(len(observed), 1)
            self.assertTrue(
                (root / "resume/payload/round_completion/model_bundle/bundle.json").is_file()
            )
            default_harness = _TransferBundleHarness()
            default_observed: list[object] = []
            concrete_calls: list[tuple[Path, str, Path | None]] = []

            def fake_concrete(
                request: object,
                destination: Path,
                *,
                config_path: Path,
                expected_config_sha256: str,
                source_bundle_root: Path | None,
            ) -> None:
                concrete_calls.append(
                    (config_path, expected_config_sha256, source_bundle_root)
                )
                default_harness.callback(default_observed)(request, destination)

            with mock.patch(
                "compag_curation.model_bundle.verify_model_bundle",
                side_effect=default_harness.verify,
            ), mock.patch.object(
                facade, "require_science_dependencies", return_value=dependency
            ), mock.patch.object(
                facade,
                "revalidate_science_gpu_dependencies",
                return_value=dependency,
            ), mock.patch.object(
                facade,
                "retrain_canonical_transfer_bundle",
                side_effect=fake_concrete,
            ):
                result, code = facade.resume_transfer_image_round(
                    begin_root,
                    stage60,
                    baseline_root,
                    reviewed,
                    1,
                    root / "resume-default-writer",
                    config=config,
                )
            self.assertEqual((result["status"], code), ("PASS", 0))
            self.assertEqual(len(default_observed), 1)
            self.assertEqual(
                concrete_calls,
                [(config, project_config.config_sha256, None)],
            )


if __name__ == "__main__":
    unittest.main()
