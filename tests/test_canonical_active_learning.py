from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from compag_curation.canonical.active_learning import (
    CANONICAL_AL_SHORTLIST_COLUMNS,
    CanonicalActiveLearningPoolRow,
    CanonicalActiveLearningTrainingRow,
    _feature_archive_hash,
    begin_canonical_active_learning_round,
    complete_initial_labeling_export_all,
    read_canonical_accumulated_feature_archive,
    resume_canonical_active_learning_round,
    write_canonical_active_learning_pool,
    write_initial_labeling_export_all,
)
from compag_curation.canonical.features import (
    CANONICAL_RAW_FEATURE_ORDER,
    CanonicalFeatureState,
    CanonicalRawFeature,
)
from compag_curation.canonical.spec import (
    CANONICAL_FEATURE_CROP_SCALES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_CPU_PROFILE,
    CANONICAL_GPU_PROFILE,
)
from compag_curation.model_bundle import BUNDLE_SCHEMA_V2
from compag_curation.public_io import PublicIOError, write_new_json


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _pool_row(name: str, group: str, probability: float | None) -> CanonicalActiveLearningPoolRow:
    proposal = _sha(f"proposal:{name}")
    return CanonicalActiveLearningPoolRow(
        proposal_id=proposal,
        proposal_sha256=proposal,
        image_id=_sha(f"image:{name}"),
        image_sha256=_sha(f"image-bytes:{name}"),
        group_id=group,
        scale=1.0,
        xgb_p=probability,
    )


def _split(path: Path, initial_completion: Path) -> None:
    completion = json.loads((initial_completion / "initial_completion.json").read_text())
    with (initial_completion / "decision_log.csv").open("r", encoding="ascii", newline="") as handle:
        decisions = list(csv.DictReader(handle))

    def counts(groups: set[str]) -> dict[str, int]:
        rows = [
            row for row in decisions
            if row["group_id"] in groups and float(row["review_weight"]) > 0.0
        ]
        return {
            "rows": len(rows),
            "negative": sum(row["label"] == "0" for row in rows),
            "positive": sum(row["label"] == "1" for row in rows),
        }

    train_counts = counts({"g0", "g1", "g2", "g3", "g4"})
    test_counts = counts({"g5"})
    write_new_json(
        path,
        {
            "schema": "compag-curation-canonical-group-split/v1",
            "status": "PASS",
            "profile": CANONICAL_GPU_PROFILE,
            "seed": 42,
            "method": "SOURCE_TILE_BALANCED_GROUP_PURE_NEAREST_80_20",
            "train_groups": ["g0", "g1", "g2", "g3", "g4"],
            "test_groups": ["g5"],
            "target_test_rows": 2,
            "observed_test_rows": 2,
            "group_cv_folds": 5,
            "coverage": {"train": train_counts, "test": test_counts},
            "effective_rows": train_counts["rows"] + test_counts["rows"],
            "skipped_rows": 0,
            "reviewed_sha256": completion["reviewed_batch_sha256"],
        },
    )


def _reviewed_copy(request: Path, reviewed: Path, *, label: str = "1") -> None:
    with request.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    with reviewed.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            row["label"] = label if index % 2 == 0 else "0"
            row["review_action"] = "accept"
            row["review_weight"] = "1.0"
            row["review_status"] = "reviewed"
            writer.writerow(row)


def _state() -> CanonicalFeatureState:
    import numpy as np

    prototype = np.zeros(2048, dtype=np.float32)
    prototype[0] = 1.0
    components = np.zeros((32, 2048), dtype=np.float32)
    mean = np.zeros(2048, dtype=np.float32)
    for value in (prototype, components, mean):
        value.setflags(write=False)
    return CanonicalFeatureState(prototype, components, mean, 40, 20)


class _BundleHarness:
    def __init__(
        self,
        source: Path,
        state: CanonicalFeatureState,
        *,
        profile: str = CANONICAL_GPU_PROFILE,
    ) -> None:
        self.profile = profile
        self.states: dict[Path, CanonicalFeatureState] = {}
        self.provenance: dict[Path, dict[str, object]] = {}
        source.mkdir()
        (source / "bundle.json").write_bytes(b"source-bundle\n")
        self.states[source.resolve()] = state
        self.provenance[source.resolve()] = {
            "raw_features_sha256": _sha("genesis-features"),
        }

    def verify(self, root: Path) -> SimpleNamespace:
        root = root.resolve(strict=True)
        state = self.states[root]
        payload = (root / "bundle.json").read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        return SimpleNamespace(
            schema=BUNDLE_SCHEMA_V2,
            profile=self.profile,
            feature_order=CANONICAL_FEATURE_ORDER,
            feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
            threshold=0.5,
            bundle_sha256=digest,
            tree_identity=(("bundle.json", digest),),
            prototype=state.prototype,
            pca_components=state.pca_components,
            pca_mean=state.pca_mean,
            classifier=b"fixture-classifier",
            imputer=(0.0,) * len(CANONICAL_FEATURE_ORDER),
            provenance={
                "schema": "compag-curation-model-provenance/v2",
                "profile": self.profile,
                "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
                "threshold_method": "FIXED_CANONICAL_METHOD",
                "details": self.provenance[root],
            },
        )

    def hash(self, root: Path) -> str:
        return self.verify(root).bundle_sha256

    def retrain(self, observed: list[object]):
        def callback(request: object, output: Path) -> None:
            observed.append(request)
            output.mkdir()
            (output / "bundle.json").write_bytes(b"retrained-bundle\n")
            self.states[output.resolve()] = request.feature_state
            self.provenance[output.resolve()] = dict(request.expected_bundle_provenance)

        return callback


def _cold_start(root: Path, rows: list[CanonicalActiveLearningPoolRow]) -> tuple[Path, Path]:
    initial_pool = root / "initial_pool"
    write_canonical_active_learning_pool(
        rows,
        initial_pool,
        bundle_sha256=None,
        profile=CANONICAL_GPU_PROFILE,
    )
    initial_request = root / "initial_request"
    manifest = write_initial_labeling_export_all(
        initial_pool,
        initial_request,
        created_at_utc="2026-08-28T05:00:00Z",
    )
    if manifest["kind"] != "INITIAL_LABELING_EXPORT_ALL" or manifest["round_number"] is not None:
        raise AssertionError("cold start was represented as an active-learning round")
    reviewed = root / "initial_reviewed.csv"
    _reviewed_copy(initial_request / "review_request.csv", reviewed)
    initial_completion = root / "initial_completion"
    complete_initial_labeling_export_all(
        initial_request,
        initial_pool,
        reviewed,
        initial_completion,
        decision_timestamp_utc="2026-08-28T05:30:00Z",
    )
    return initial_completion / "decision_log.csv", initial_completion


def _raw_rows(
    decisions: list[CanonicalActiveLearningPoolRow],
    *,
    positive_tail: frozenset[str] = frozenset(),
) -> list[CanonicalActiveLearningTrainingRow]:
    import numpy as np

    output: list[CanonicalActiveLearningTrainingRow] = []
    for row in sorted(decisions, key=lambda value: value.proposal_id):
        proposal_index = int(row.proposal_id[:8], 16) + 1
        for scale in CANONICAL_FEATURE_CROP_SCALES:
            embedding = np.zeros(2048, dtype=np.float32)
            embedding[1 if row.proposal_id in positive_tail else 0] = 1.0
            raw = CanonicalRawFeature(
                proposal_index=proposal_index,
                scale=scale,
                predicted_iou=0.9,
                stability_score=0.95,
                values={name: float(proposal_index) for name in CANONICAL_RAW_FEATURE_ORDER},
                embedding=embedding,
            )
            output.append(
                CanonicalActiveLearningTrainingRow(
                    proposal_id=row.proposal_id,
                    proposal_sha256=row.proposal_sha256,
                    image_id=row.image_id,
                    image_sha256=row.image_sha256,
                    group_id=row.group_id,
                    raw_feature=raw,
                    source_sha256=_sha("raw-feature-source"),
                )
            )
    return output


class CanonicalActiveLearningTests(unittest.TestCase):
    def test_two_single_image_rounds_full_retrain_from_cumulative_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = [
                _pool_row(f"initial-{index}", f"g{index % 5}", None)
                for index in range(10)
            ]
            initial += [
                _pool_row("initial-test-0", "g5", None),
                _pool_row("initial-test-1", "g5", None),
            ]
            decision_log, initial_completion = _cold_start(root, initial)
            split = root / "split.json"
            _split(split, initial_completion)
            state = _state()
            source_bundle = root / "source_bundle"
            harness = _BundleHarness(source_bundle, state)
            self.enterContext(
                mock.patch(
                    "compag_curation.model_bundle.verify_model_bundle",
                    side_effect=harness.verify,
                )
            )
            self.enterContext(
                mock.patch(
                    "compag_curation.canonical.serialization.load_portable_predictor",
                    return_value=object(),
                )
            )
            self.enterContext(
                mock.patch(
                    "compag_curation.canonical.serialization.predict_portable_probabilities",
                    side_effect=lambda _predictor, rows: [0.4] * len(rows),
                )
            )

            round1_candidate = _pool_row("new-card-a", "g6", 0.5)
            multi_image_pool = root / "multi-image-pool"
            write_canonical_active_learning_pool(
                [round1_candidate, _pool_row("other-card", "g7", 0.5)],
                multi_image_pool,
                bundle_sha256=harness.hash(source_bundle),
                source_sha256=_sha("raw-feature-source"),
                feature_archive_sha256=_sha("multi-image-features"),
                profile=CANONICAL_GPU_PROFILE,
            )
            with self.assertRaisesRegex(PublicIOError, "exactly one image"):
                begin_canonical_active_learning_round(
                    multi_image_pool,
                    source_bundle,
                    split,
                    decision_log,
                    root / "multi-image-selection",
                    round_number=1,
                    created_at_utc="2026-08-28T06:00:00Z",
                    initial_completion_path=initial_completion
                    / "initial_completion.json",
                    image_scoped=True,
                )
            round1_pool = root / "round1-pool"
            round1_features = _raw_rows(
                [round1_candidate],
                positive_tail=frozenset({round1_candidate.proposal_id}),
            )
            write_canonical_active_learning_pool(
                [round1_candidate],
                round1_pool,
                bundle_sha256=harness.hash(source_bundle),
                source_sha256=_sha("raw-feature-source"),
                feature_archive_sha256=_feature_archive_hash(round1_features),
                profile=CANONICAL_GPU_PROFILE,
            )
            selection1 = root / "selection1"
            paused1 = begin_canonical_active_learning_round(
                round1_pool,
                source_bundle,
                split,
                decision_log,
                selection1,
                round_number=1,
                created_at_utc="2026-08-28T06:00:00Z",
                initial_completion_path=initial_completion
                / "initial_completion.json",
                image_scoped=True,
            )
            self.assertEqual(
                paused1["schema"],
                "compag-curation-canonical-active-learning-selection/v3",
            )
            self.assertEqual(paused1["image"]["group_id"], "g6")
            reviewed1 = root / "reviewed1.csv"
            _reviewed_copy(selection1 / "review_request.csv", reviewed1)
            retrain_calls: list[object] = []
            completion1 = root / "completion1"
            result1 = resume_canonical_active_learning_round(
                selection1,
                round1_pool,
                source_bundle,
                split,
                decision_log,
                reviewed1,
                state,
                [*_raw_rows(initial), *round1_features],
                round1_features,
                completion1,
                completed_at_utc="2026-08-28T07:00:00Z",
                retrain=harness.retrain(retrain_calls),
                initial_completion_path=initial_completion
                / "initial_completion.json",
                image_scoped=True,
            )
            self.assertEqual(
                result1["schema"],
                "compag-curation-canonical-active-learning-completion/v3",
            )
            lineage1 = json.loads(
                (completion1 / "model_lineage.json").read_text(encoding="ascii")
            )
            self.assertEqual(lineage1["model_label"], "project-r1")
            archive1, receipt1 = read_canonical_accumulated_feature_archive(
                completion1 / "accumulated_al_features.csv"
            )
            self.assertEqual(receipt1["proposal_count"], 1)
            self.assertEqual({row.group_id for row in archive1}, {"g6"})
            self.assertEqual(retrain_calls[-1].train_groups, frozenset({"g0", "g1", "g2", "g3", "g4", "g6"}))

            round2_candidate = _pool_row("new-card-b", "g7", 0.5)
            round2_pool = root / "round2-pool"
            round2_features = _raw_rows([round2_candidate])
            round1_bundle = completion1 / "model_bundle"
            write_canonical_active_learning_pool(
                [round2_candidate],
                round2_pool,
                bundle_sha256=harness.hash(round1_bundle),
                source_sha256=_sha("raw-feature-source"),
                feature_archive_sha256=_feature_archive_hash(round2_features),
                profile=CANONICAL_GPU_PROFILE,
            )

            completion1_manifest = completion1 / "round_completion.json"
            completion1_lineage = completion1 / "model_lineage.json"
            completion1_archive = completion1 / "accumulated_al_features.csv"

            def replace_json(path: Path, value: object) -> None:
                path.unlink()
                write_new_json(path, value)

            def file_record(path: Path, role: str) -> dict[str, object]:
                payload = path.read_bytes()
                return {
                    "role": role,
                    "basename": path.name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }

            original_ancestry = {
                path: path.read_bytes()
                for path in (
                    completion1_manifest,
                    completion1_lineage,
                    completion1_archive,
                )
            }
            with completion1_archive.open(
                "r", encoding="ascii", newline=""
            ) as handle:
                archive_reader = csv.DictReader(handle)
                archive_fields = tuple(archive_reader.fieldnames or ())
                forged_archive_rows = list(archive_reader)
            forged_proposal = forged_archive_rows[0]["proposal_id"]
            for archive_row in forged_archive_rows:
                if archive_row["proposal_id"] == forged_proposal:
                    archive_row["predicted_iou"] = format(0.8, ".17g")
            with completion1_archive.open(
                "w", encoding="ascii", newline=""
            ) as handle:
                archive_writer = csv.DictWriter(
                    handle,
                    fieldnames=archive_fields,
                    lineterminator="\n",
                )
                archive_writer.writeheader()
                archive_writer.writerows(forged_archive_rows)
            _forged_rows, forged_archive_receipt = (
                read_canonical_accumulated_feature_archive(completion1_archive)
            )
            forged_lineage = json.loads(
                completion1_lineage.read_text(encoding="ascii")
            )
            forged_lineage["cumulative_al_feature_archive_sha256"] = (
                forged_archive_receipt["logical_sha256"]
            )
            replace_json(completion1_lineage, forged_lineage)
            forged_completion = json.loads(
                completion1_manifest.read_text(encoding="ascii")
            )
            forged_completion["outputs"]["accumulated_al_features"] = (
                forged_archive_receipt
            )
            forged_completion["outputs"]["model_lineage"] = file_record(
                completion1_lineage,
                "sequential image model lineage",
            )
            replace_json(completion1_manifest, forged_completion)
            with self.assertRaisesRegex(
                PublicIOError,
                "parent bundle provenance differs",
            ):
                begin_canonical_active_learning_round(
                    round2_pool,
                    round1_bundle,
                    split,
                    completion1 / "decision_log.csv",
                    root / "forged-archive-selection",
                    round_number=2,
                    created_at_utc="2026-08-28T08:00:00Z",
                    prior_round_completion_path=completion1_manifest,
                    image_scoped=True,
                )
            self.assertFalse((root / "forged-archive-selection").exists())
            for path, payload in original_ancestry.items():
                path.write_bytes(payload)

            with completion1_archive.open(
                "r", encoding="ascii", newline=""
            ) as handle:
                archive_reader = csv.DictReader(handle)
                archive_fields = tuple(archive_reader.fieldnames or ())
                forged_identity_rows = list(archive_reader)
            original_proposal = forged_identity_rows[0]["proposal_id"]
            forged_proposal = _sha("resealed-forged-archive-proposal")
            for archive_row in forged_identity_rows:
                if archive_row["proposal_id"] == original_proposal:
                    archive_row["proposal_id"] = forged_proposal
                    archive_row["proposal_sha256"] = forged_proposal
            with completion1_archive.open(
                "w", encoding="ascii", newline=""
            ) as handle:
                archive_writer = csv.DictWriter(
                    handle,
                    fieldnames=archive_fields,
                    lineterminator="\n",
                )
                archive_writer.writeheader()
                archive_writer.writerows(forged_identity_rows)
            _forged_rows, forged_identity_receipt = (
                read_canonical_accumulated_feature_archive(completion1_archive)
            )
            forged_lineage = json.loads(
                completion1_lineage.read_text(encoding="ascii")
            )
            forged_lineage["cumulative_al_feature_archive_sha256"] = (
                forged_identity_receipt["logical_sha256"]
            )
            replace_json(completion1_lineage, forged_lineage)
            forged_completion = json.loads(
                completion1_manifest.read_text(encoding="ascii")
            )
            forged_completion["outputs"]["accumulated_al_features"] = (
                forged_identity_receipt
            )
            forged_completion["outputs"]["model_lineage"] = file_record(
                completion1_lineage,
                "sequential image model lineage",
            )
            replace_json(completion1_manifest, forged_completion)
            with self.assertRaisesRegex(
                PublicIOError,
                "cumulative archive does not close",
            ):
                begin_canonical_active_learning_round(
                    round2_pool,
                    round1_bundle,
                    split,
                    completion1 / "decision_log.csv",
                    root / "forged-archive-identity-selection",
                    round_number=2,
                    created_at_utc="2026-08-28T08:00:00Z",
                    prior_round_completion_path=completion1_manifest,
                    image_scoped=True,
                )
            self.assertFalse(
                (root / "forged-archive-identity-selection").exists()
            )
            for path, payload in original_ancestry.items():
                path.write_bytes(payload)

            forged_parent = "f" * 64
            forged_lineage = json.loads(
                completion1_lineage.read_text(encoding="ascii")
            )
            forged_lineage["parent_bundle_sha256"] = forged_parent
            replace_json(completion1_lineage, forged_lineage)
            forged_completion = json.loads(
                completion1_manifest.read_text(encoding="ascii")
            )
            forged_completion["chain"]["source_bundle_sha256"] = forged_parent
            forged_completion["outputs"]["model_lineage"] = file_record(
                completion1_lineage,
                "sequential image model lineage",
            )
            replace_json(completion1_manifest, forged_completion)
            with self.assertRaisesRegex(
                PublicIOError,
                "parent bundle provenance differs",
            ):
                begin_canonical_active_learning_round(
                    round2_pool,
                    round1_bundle,
                    split,
                    completion1 / "decision_log.csv",
                    root / "forged-parent-selection",
                    round_number=2,
                    created_at_utc="2026-08-28T08:00:00Z",
                    prior_round_completion_path=completion1_manifest,
                    image_scoped=True,
                )
            self.assertFalse((root / "forged-parent-selection").exists())
            for path, payload in original_ancestry.items():
                path.write_bytes(payload)

            selection2 = root / "selection2"
            begin_canonical_active_learning_round(
                round2_pool,
                round1_bundle,
                split,
                completion1 / "decision_log.csv",
                selection2,
                round_number=2,
                created_at_utc="2026-08-28T08:00:00Z",
                prior_round_completion_path=completion1
                / "round_completion.json",
                image_scoped=True,
            )
            reviewed2 = root / "reviewed2.csv"
            _reviewed_copy(selection2 / "review_request.csv", reviewed2)
            completion2 = root / "completion2"
            result2 = resume_canonical_active_learning_round(
                selection2,
                round2_pool,
                round1_bundle,
                split,
                completion1 / "decision_log.csv",
                reviewed2,
                retrain_calls[-1].feature_state,
                [*_raw_rows(initial), *archive1, *round2_features],
                round2_features,
                completion2,
                completed_at_utc="2026-08-28T09:00:00Z",
                retrain=harness.retrain(retrain_calls),
                prior_round_completion_path=completion1
                / "round_completion.json",
                image_scoped=True,
            )
            self.assertEqual(result2["image"]["group_id"], "g7")
            archive2, receipt2 = read_canonical_accumulated_feature_archive(
                completion2 / "accumulated_al_features.csv"
            )
            self.assertEqual(receipt2["proposal_count"], 2)
            self.assertEqual({row.group_id for row in archive2}, {"g6", "g7"})
            self.assertEqual(
                retrain_calls[-1].train_groups,
                frozenset({"g0", "g1", "g2", "g3", "g4", "g6", "g7"}),
            )
            self.assertEqual(
                retrain_calls[-1].expected_bundle_provenance["model_label"],
                "project-r2",
            )

            duplicate_group_pool = root / "duplicate-group-pool"
            duplicate_group = _pool_row("different-bytes-same-card", "g7", 0.5)
            duplicate_features = _raw_rows([duplicate_group])
            write_canonical_active_learning_pool(
                [duplicate_group],
                duplicate_group_pool,
                bundle_sha256=harness.hash(completion2 / "model_bundle"),
                source_sha256=_sha("raw-feature-source"),
                feature_archive_sha256=_feature_archive_hash(duplicate_features),
                profile=CANONICAL_GPU_PROFILE,
            )
            with self.assertRaisesRegex(PublicIOError, "previously unseen image group"):
                begin_canonical_active_learning_round(
                    duplicate_group_pool,
                    completion2 / "model_bundle",
                    split,
                    completion2 / "decision_log.csv",
                    root / "duplicate-group-selection",
                    round_number=3,
                    created_at_utc="2026-08-28T10:00:00Z",
                    prior_round_completion_path=completion2
                    / "round_completion.json",
                    image_scoped=True,
                )

            renamed_duplicate = replace(
                _pool_row("renamed-copy", "g8", 0.5),
                image_sha256=round2_candidate.image_sha256,
            )
            renamed_pool = root / "renamed-duplicate-image-pool"
            renamed_features = _raw_rows([renamed_duplicate])
            write_canonical_active_learning_pool(
                [renamed_duplicate],
                renamed_pool,
                bundle_sha256=harness.hash(completion2 / "model_bundle"),
                source_sha256=_sha("raw-feature-source"),
                feature_archive_sha256=_feature_archive_hash(renamed_features),
                profile=CANONICAL_GPU_PROFILE,
            )
            with self.assertRaisesRegex(PublicIOError, "image identity"):
                begin_canonical_active_learning_round(
                    renamed_pool,
                    completion2 / "model_bundle",
                    split,
                    completion2 / "decision_log.csv",
                    root / "renamed-duplicate-image-selection",
                    round_number=3,
                    created_at_utc="2026-08-28T10:00:00Z",
                    prior_round_completion_path=completion2
                    / "round_completion.json",
                    image_scoped=True,
                )

    def test_historical_cpu_bundle_cannot_begin_a_new_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = _pool_row("historical-cpu", "g0", 0.5)
            state = _state()
            source_bundle = root / "historical_cpu_bundle"
            harness = _BundleHarness(
                source_bundle,
                state,
                profile=CANONICAL_CPU_PROFILE,
            )
            pool = root / "historical_cpu_pool"
            write_canonical_active_learning_pool(
                [row],
                pool,
                bundle_sha256=harness.hash(source_bundle),
                source_sha256=_sha("historical-physical-archive"),
                feature_archive_sha256=_sha("historical-logical-archive"),
                profile=CANONICAL_CPU_PROFILE,
            )
            with mock.patch(
                "compag_curation.model_bundle.verify_model_bundle",
                side_effect=harness.verify,
            ):
                output = root / "rejected_cpu_round"
                with self.assertRaisesRegex(PublicIOError, "requires a canonical GPU bundle"):
                    begin_canonical_active_learning_round(
                        pool,
                        source_bundle,
                        root / "unused-split.json",
                        root / "unused-decisions.csv",
                        output,
                        round_number=1,
                        created_at_utc="2026-08-28T06:00:00Z",
                    )
            self.assertFalse(output.exists())

    def test_cold_start_is_distinct_and_selection_is_exact_top50(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = [_pool_row(f"initial-{index}", f"g{index % 5}", None) for index in range(10)]
            initial += [_pool_row("initial-test-0", "g5", None), _pool_row("initial-test-1", "g5", None)]
            with self.assertRaisesRegex(PublicIOError, "duplicate proposal"):
                write_canonical_active_learning_pool(
                    [initial[0], initial[0]],
                    root / "duplicate_pool",
                    bundle_sha256=None,
                )
            self.assertFalse((root / "duplicate_pool").exists())
            real_parent = root / "real_parent"
            real_parent.mkdir()
            (root / "linked_parent").symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(PublicIOError, "parent is unsafe"):
                write_canonical_active_learning_pool(
                    [initial[0]],
                    root / "linked_parent/pool",
                    bundle_sha256=None,
                )
            self.assertFalse((real_parent / "pool").exists())
            decision_log, initial_completion = _cold_start(root, initial)
            state = _state()
            source_bundle = root / "source_bundle"
            harness = _BundleHarness(source_bundle, state)
            self.enterContext(
                mock.patch(
                    "compag_curation.model_bundle.verify_model_bundle",
                    side_effect=harness.verify,
                )
            )

            scored_initial = [replace(row, xgb_p=0.5) for row in initial]
            ties = [_pool_row(f"tie-{index:02d}", f"g{index % 5}", 0.5) for index in range(55)]
            boundary = _pool_row("boundary", "g0", 0.7)
            outside = _pool_row("outside", "g0", 0.700001)
            frozen_test = _pool_row("frozen-test", "g5", 0.5)
            legacy_pool = root / "legacy_scored_pool"
            with self.assertRaisesRegex(PublicIOError, "physical and logical"):
                write_canonical_active_learning_pool(
                    [outside],
                    legacy_pool,
                    bundle_sha256=harness.hash(source_bundle),
                    profile=CANONICAL_GPU_PROFILE,
                )
            self.assertFalse(legacy_pool.exists())
            pool_root = root / "scored_pool"
            pool_source_sha256 = _sha("selection-raw-feature-archive")
            write_canonical_active_learning_pool(
                [*scored_initial, *ties, boundary, outside, frozen_test],
                pool_root,
                bundle_sha256=harness.hash(source_bundle),
                source_sha256=pool_source_sha256,
                feature_archive_sha256=_sha("selection-logical-feature-archive"),
                profile=CANONICAL_GPU_PROFILE,
            )
            self.assertEqual(
                json.loads((pool_root / "pool_result.json").read_text(encoding="ascii"))[
                    "source_sha256"
                ],
                pool_source_sha256,
            )
            split = root / "split.json"
            _split(split, initial_completion)
            bad_split = root / "bad_split.json"
            bad_split_value = json.loads(split.read_text(encoding="ascii"))
            bad_split_value["reviewed_sha256"] = "b" * 64
            write_new_json(bad_split, bad_split_value)
            with self.assertRaisesRegex(PublicIOError, "initial reviewed batch"):
                begin_canonical_active_learning_round(
                    pool_root,
                    source_bundle,
                    bad_split,
                    decision_log,
                    root / "bad_genesis_selection",
                    round_number=1,
                    created_at_utc="2026-08-28T06:00:00Z",
                    initial_completion_path=initial_completion / "initial_completion.json",
                )
            self.assertFalse((root / "bad_genesis_selection").exists())
            selection_root = root / "round1"
            result = begin_canonical_active_learning_round(
                pool_root,
                source_bundle,
                split,
                decision_log,
                selection_root,
                round_number=1,
                created_at_utc="2026-08-28T06:00:00Z",
                initial_completion_path=initial_completion / "initial_completion.json",
            )
            self.assertEqual(result["status"], "PAUSED_FOR_REVIEW")
            self.assertEqual(result["kind"], "ACTIVE_LEARNING_ROUND")
            self.assertEqual(result["counts"]["reviewed_excluded"], len(initial))
            self.assertEqual(result["counts"]["frozen_test_group_excluded"], 1)
            self.assertEqual(result["counts"]["outside_margin_excluded"], 1)
            self.assertEqual(result["counts"]["margin_eligible"], 56)
            self.assertEqual(result["counts"]["selected"], 50)
            with (selection_root / "shortlist.csv").open("r", encoding="ascii", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(tuple(rows[0]), CANONICAL_AL_SHORTLIST_COLUMNS)
            expected_ties = sorted(row.proposal_id for row in ties)[:50]
            self.assertEqual([row["proposal_id"] for row in rows], expected_ties)
            self.assertNotIn(boundary.proposal_id, {row["proposal_id"] for row in rows})
            with self.assertRaises(PublicIOError):
                begin_canonical_active_learning_round(
                    pool_root,
                    source_bundle,
                    split,
                    decision_log,
                    selection_root,
                    round_number=1,
                    created_at_utc="2026-08-28T06:00:00Z",
                    initial_completion_path=initial_completion / "initial_completion.json",
                )
            reviewed = root / "tamper_reviewed.csv"
            _reviewed_copy(selection_root / "review_request.csv", reviewed)
            manifest_path = selection_root / "round_manifest.json"
            tampered = json.loads(manifest_path.read_text(encoding="ascii"))
            tampered["policy"]["margin"] = 0.21
            manifest_path.write_text(
                json.dumps(tampered, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(PublicIOError, "paused manifest"):
                resume_canonical_active_learning_round(
                    selection_root,
                    pool_root,
                    source_bundle,
                    split,
                    decision_log,
                    reviewed,
                    state,
                    (),
                    (),
                    root / "tampered_completion",
                    completed_at_utc="2026-08-28T07:00:00Z",
                    retrain=lambda _request, _output: self.fail("tampered selection reached retrain"),
                    initial_completion_path=initial_completion / "initial_completion.json",
                )
            self.assertFalse((root / "tampered_completion").exists())

    def test_resume_freezes_pca_recomputes_prototype_and_binds_rescore(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = [_pool_row(f"initial-{index}", f"g{index % 5}", None) for index in range(10)]
            initial += [_pool_row("initial-test-0", "g5", None), _pool_row("initial-test-1", "g5", None)]
            decision_log, initial_completion = _cold_start(root, initial)
            state = _state()
            source_bundle = root / "source_bundle"
            harness = _BundleHarness(source_bundle, state)
            self.enterContext(
                mock.patch(
                    "compag_curation.model_bundle.verify_model_bundle",
                    side_effect=harness.verify,
                )
            )
            selected = _pool_row("new-positive", "g0", 0.5)
            next_candidate = _pool_row("next-round", "g1", 0.1)
            pool_rows = [
                *[replace(row, xgb_p=0.1) for row in initial],
                selected,
                next_candidate,
            ]
            pool_root = root / "scored_pool"
            source_archive_sha256 = _sha("raw-feature-source")
            pool_features = _raw_rows(
                pool_rows,
                positive_tail=frozenset({selected.proposal_id}),
            )
            write_canonical_active_learning_pool(
                pool_rows,
                pool_root,
                bundle_sha256=harness.hash(source_bundle),
                source_sha256=source_archive_sha256,
                feature_archive_sha256=_feature_archive_hash(
                    pool_features
                ),
                profile=CANONICAL_GPU_PROFILE,
            )
            split = root / "split.json"
            _split(split, initial_completion)
            selection_root = root / "selection"
            selection_result = begin_canonical_active_learning_round(
                pool_root,
                source_bundle,
                split,
                decision_log,
                selection_root,
                round_number=1,
                created_at_utc="2026-08-28T06:00:00Z",
                initial_completion_path=initial_completion / "initial_completion.json",
            )
            self.assertEqual(
                selection_result["chain"]["source_pool_source_sha256"],
                source_archive_sha256,
            )
            reviewed = root / "round_reviewed.csv"
            _reviewed_copy(selection_root / "review_request.csv", reviewed, label="1")
            accumulated = [*initial, selected]
            features = _raw_rows(
                accumulated,
                positive_tail=frozenset({selected.proposal_id}),
            )
            retrain_calls: list[object] = []

            def predict(_predictor: object, finalized: tuple[object, ...]) -> list[float]:
                self.assertTrue(retrain_calls)
                self.assertEqual(len(finalized), len(pool_rows))
                return [
                    0.5 if row.proposal_id == next_candidate.proposal_id else 0.25
                    for row in sorted(pool_rows, key=lambda item: item.proposal_id)
                ]

            self.enterContext(
                mock.patch(
                    "compag_curation.canonical.serialization.load_portable_predictor",
                    return_value=object(),
                )
            )
            self.enterContext(
                mock.patch(
                    "compag_curation.canonical.serialization.predict_portable_probabilities",
                    side_effect=predict,
                )
            )

            output = root / "completion"
            mixed_training_sources = list(features)
            mixed_training_sources[0] = replace(
                mixed_training_sources[0],
                source_sha256=_sha("wrong-accumulated-feature-archive"),
            )
            with self.assertRaisesRegex(PublicIOError, "scale/source closure"):
                resume_canonical_active_learning_round(
                    selection_root,
                    pool_root,
                    source_bundle,
                    split,
                    decision_log,
                    reviewed,
                    state,
                    mixed_training_sources,
                    pool_features,
                    root / "mixed_training_source_completion",
                    completed_at_utc="2026-08-28T07:00:00Z",
                    retrain=lambda _request, _output: self.fail(
                        "mixed-source training rows reached retrain"
                    ),
                    initial_completion_path=initial_completion
                    / "initial_completion.json",
                )
            self.assertFalse(
                (root / "mixed_training_source_completion").exists()
            )
            mismatched_pool_features = tuple(
                replace(row, source_sha256=_sha("wrong-stage60-feature-archive"))
                for row in pool_features
            )
            with self.assertRaisesRegex(PublicIOError, "physical or logical archive"):
                resume_canonical_active_learning_round(
                    selection_root,
                    pool_root,
                    source_bundle,
                    split,
                    decision_log,
                    reviewed,
                    state,
                    features,
                    mismatched_pool_features,
                    root / "source_mismatch_completion",
                    completed_at_utc="2026-08-28T07:00:00Z",
                    retrain=lambda _request, _output: self.fail(
                        "source-mismatched features reached retrain"
                    ),
                    initial_completion_path=initial_completion
                    / "initial_completion.json",
                )
            self.assertFalse((root / "source_mismatch_completion").exists())
            result = resume_canonical_active_learning_round(
                selection_root,
                pool_root,
                source_bundle,
                split,
                decision_log,
                reviewed,
                state,
                features,
                pool_features,
                output,
                completed_at_utc="2026-08-28T07:00:00Z",
                retrain=harness.retrain(retrain_calls),
                initial_completion_path=initial_completion / "initial_completion.json",
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(len(retrain_calls), 1)
            request = retrain_calls[0]
            self.assertEqual(request.train_groups, frozenset({"g0", "g1", "g2", "g3", "g4"}))
            self.assertEqual(request.test_groups, frozenset({"g5"}))
            self.assertEqual(
                np.asarray(request.feature_state.pca_components).tobytes(),
                np.asarray(state.pca_components).tobytes(),
            )
            self.assertEqual(
                np.asarray(request.feature_state.pca_mean).tobytes(),
                np.asarray(state.pca_mean).tobytes(),
            )
            self.assertGreater(float(request.feature_state.prototype[1]), 0.0)
            self.assertAlmostEqual(float(np.linalg.norm(request.feature_state.prototype)), 1.0, places=6)
            self.assertEqual(
                result["outputs"]["new_bundle_sha256"],
                harness.hash(output / "model_bundle"),
            )
            self.assertEqual(
                result["outputs"]["rescored_predictions_sha256"],
                json.loads((output / "rescored_pool/pool_result.json").read_text())["predictions_sha256"],
            )
            added_candidate = _pool_row("round-two-added", "g2", 0.65)
            round_two_rows = [
                *[
                    replace(
                        row,
                        xgb_p=(0.5 if row.proposal_id == next_candidate.proposal_id else 0.25),
                    )
                    for row in pool_rows
                ],
                added_candidate,
            ]
            round_two_features = _raw_rows(round_two_rows)
            omitted_pool = root / "round2_omits_reviewed_pool"
            write_canonical_active_learning_pool(
                [
                    row
                    for row in round_two_rows
                    if row.proposal_id != selected.proposal_id
                ],
                omitted_pool,
                bundle_sha256=harness.hash(output / "model_bundle"),
                source_sha256=source_archive_sha256,
                feature_archive_sha256=_feature_archive_hash(
                    [
                        row
                        for row in round_two_features
                        if row.proposal_id != selected.proposal_id
                    ]
                ),
                profile=CANONICAL_GPU_PROFILE,
            )
            with self.assertRaisesRegex(PublicIOError, "prior reviewed candidate"):
                begin_canonical_active_learning_round(
                    omitted_pool,
                    output / "model_bundle",
                    split,
                    output / "decision_log.csv",
                    root / "round2_omits_reviewed_selection",
                    round_number=2,
                    created_at_utc="2026-08-28T08:00:00Z",
                    prior_round_completion_path=output / "round_completion.json",
                )
            cumulative_pool = root / "round2_cumulative_pool"
            write_canonical_active_learning_pool(
                round_two_rows,
                cumulative_pool,
                bundle_sha256=harness.hash(output / "model_bundle"),
                source_sha256=source_archive_sha256,
                feature_archive_sha256=_feature_archive_hash(
                    round_two_features
                ),
                profile=CANONICAL_GPU_PROFILE,
            )
            round_two = root / "round2_selection"
            round_two_result = begin_canonical_active_learning_round(
                cumulative_pool,
                output / "model_bundle",
                split,
                output / "decision_log.csv",
                round_two,
                round_number=2,
                created_at_utc="2026-08-28T08:00:00Z",
                prior_round_completion_path=output / "round_completion.json",
            )
            self.assertEqual(round_two_result["counts"]["selected"], 2)
            with (round_two / "shortlist.csv").open("r", encoding="ascii", newline="") as handle:
                round_two_rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["proposal_id"] for row in round_two_rows},
                {next_candidate.proposal_id, added_candidate.proposal_id},
            )

    def test_invalid_weight_or_failed_retrain_never_writes_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = [_pool_row(f"initial-{index}", f"g{index % 5}", None) for index in range(10)]
            initial += [_pool_row("initial-test-0", "g5", None), _pool_row("initial-test-1", "g5", None)]
            decision_log, initial_completion = _cold_start(root, initial)
            state = _state()
            source_bundle = root / "source_bundle"
            harness = _BundleHarness(source_bundle, state)
            self.enterContext(
                mock.patch(
                    "compag_curation.model_bundle.verify_model_bundle",
                    side_effect=harness.verify,
                )
            )
            selected = _pool_row("new-positive", "g0", 0.5)
            pool_rows = [*[replace(row, xgb_p=0.1) for row in initial], selected]
            pool_root = root / "scored_pool"
            pool_features = _raw_rows(pool_rows)
            write_canonical_active_learning_pool(
                pool_rows,
                pool_root,
                bundle_sha256=harness.hash(source_bundle),
                source_sha256=_sha("raw-feature-source"),
                feature_archive_sha256=_feature_archive_hash(
                    pool_features
                ),
                profile=CANONICAL_GPU_PROFILE,
            )
            split = root / "split.json"
            _split(split, initial_completion)
            selection_root = root / "selection"
            begin_canonical_active_learning_round(
                pool_root,
                source_bundle,
                split,
                decision_log,
                selection_root,
                round_number=1,
                created_at_utc="2026-08-28T06:00:00Z",
                initial_completion_path=initial_completion / "initial_completion.json",
            )
            invalid = root / "invalid_review.csv"
            _reviewed_copy(selection_root / "review_request.csv", invalid)
            text = invalid.read_text(encoding="ascii").replace(",accept,1.0,reviewed", ",accept,0.4,reviewed")
            invalid.write_text(text, encoding="ascii")
            with self.assertRaises(PublicIOError):
                resume_canonical_active_learning_round(
                    selection_root,
                    pool_root,
                    source_bundle,
                    split,
                    decision_log,
                    invalid,
                    state,
                    _raw_rows([*initial, selected]),
                    _raw_rows(pool_rows),
                    root / "invalid_completion",
                    completed_at_utc="2026-08-28T07:00:00Z",
                    retrain=lambda _request, _output: self.fail("invalid review reached retrain"),
                    initial_completion_path=initial_completion / "initial_completion.json",
                )
            self.assertFalse((root / "invalid_completion/round_completion.json").exists())

            valid = root / "valid_review.csv"
            _reviewed_copy(selection_root / "review_request.csv", valid)

            def fail_retrain(_request: object, _output: Path) -> None:
                raise RuntimeError("synthetic retrain failure")

            failed_output = root / "failed_completion"
            with self.assertRaisesRegex(RuntimeError, "synthetic retrain failure"):
                resume_canonical_active_learning_round(
                    selection_root,
                    pool_root,
                    source_bundle,
                    split,
                    decision_log,
                    valid,
                    state,
                    _raw_rows([*initial, selected]),
                    _raw_rows(pool_rows),
                    failed_output,
                    completed_at_utc="2026-08-28T07:00:00Z",
                    retrain=fail_retrain,
                    initial_completion_path=initial_completion / "initial_completion.json",
                )
            self.assertFalse((failed_output / "round_completion.json").exists())

            stale_provenance_output = root / "stale_provenance_completion"
            stale_retrain_calls: list[object] = []
            valid_retrain = harness.retrain(stale_retrain_calls)

            def stale_provenance_retrain(request: object, output: Path) -> None:
                valid_retrain(request, output)
                harness.provenance[output.resolve()] = {}

            with self.assertRaisesRegex(PublicIOError, "provenance"):
                resume_canonical_active_learning_round(
                    selection_root,
                    pool_root,
                    source_bundle,
                    split,
                    decision_log,
                    valid,
                    state,
                    _raw_rows([*initial, selected]),
                    _raw_rows(pool_rows),
                    stale_provenance_output,
                    completed_at_utc="2026-08-28T07:00:00Z",
                    retrain=stale_provenance_retrain,
                    initial_completion_path=initial_completion / "initial_completion.json",
                )
            self.assertEqual(len(stale_retrain_calls), 1)
            self.assertFalse((stale_provenance_output / "round_completion.json").exists())

            bad_rescore_output = root / "bad_rescore_completion"
            retrain_calls: list[object] = []

            def truncate_predictions(
                _predictor: object,
                finalized: tuple[object, ...],
            ) -> list[float]:
                return [0.25 for _row in finalized[:-1]]

            self.enterContext(
                mock.patch(
                    "compag_curation.canonical.serialization.load_portable_predictor",
                    return_value=object(),
                )
            )
            self.enterContext(
                mock.patch(
                    "compag_curation.canonical.serialization.predict_portable_probabilities",
                    side_effect=truncate_predictions,
                )
            )

            with self.assertRaisesRegex(PublicIOError, "predictor result"):
                resume_canonical_active_learning_round(
                    selection_root,
                    pool_root,
                    source_bundle,
                    split,
                    decision_log,
                    valid,
                    state,
                    _raw_rows([*initial, selected]),
                    _raw_rows(pool_rows),
                    bad_rescore_output,
                    completed_at_utc="2026-08-28T07:00:00Z",
                    retrain=harness.retrain(retrain_calls),
                    initial_completion_path=initial_completion / "initial_completion.json",
                )
            self.assertEqual(len(retrain_calls), 1)
            self.assertFalse((bad_rescore_output / "round_completion.json").exists())


if __name__ == "__main__":
    unittest.main()
