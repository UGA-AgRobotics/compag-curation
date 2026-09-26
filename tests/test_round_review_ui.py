from __future__ import annotations

import csv
import hashlib
import json
import struct
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from compag_curation.canonical.active_learning import (
    CANONICAL_AL_SHORTLIST_COLUMNS,
    CanonicalActiveLearningPoolRow,
    _format_float,
    _input_record,
    _policy,
    _read_csv,
    _read_pool,
    write_canonical_active_learning_pool,
)
from compag_curation.canonical.active_learning_facade import _directory_record
from compag_curation.canonical.active_learning_service import _canonical_image_identity
from compag_curation.canonical.service import CANONICAL_INFERENCE_COLUMNS
from compag_curation.canonical.spec import (
    CANONICAL_DECISION_THRESHOLD,
    CANONICAL_FEATURE_CROP_SCALES,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_FULL_IMAGE_NMS_IOU,
    CANONICAL_GPU_PROFILE,
    CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
)
from compag_curation.public_io import PublicIOError, compact_json_sha256, write_new_json
from compag_curation.review.exchange import read_review_export, validate_review_table, write_review_export
from compag_curation.review.round_session import (
    MAX_IMAGE_DIRECTORY_EXTRA_MEMBERS,
    RoundReviewSession,
    SOURCE_KIND,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class _RoundFixture:
    def __init__(
        self,
        root: Path,
        *,
        probabilities: tuple[float, ...],
        image_scoped: bool = False,
        transfer_scoped: bool = False,
    ) -> None:
        if image_scoped and transfer_scoped:
            raise AssertionError("round fixture scope is ambiguous")
        self.root = root
        self.image_root = root / "images"
        self.image_root.mkdir()
        self.image_name = "orchard__frame.ppm"
        width = height = 64
        raster = bytes((index * 17) % 256 for index in range(width * height * 3))
        image_payload = f"P6\n{width} {height}\n255\n".encode("ascii") + raster
        self.image_path = self.image_root / self.image_name
        self.image_path.write_bytes(image_payload)
        image_sha = hashlib.sha256(image_payload).hexdigest()
        group_id, image_id = _canonical_image_identity(self.image_name, image_sha)

        candidates: list[tuple[str, float]] = [
            (_sha(f"proposal-{index}"), probability)
            for index, probability in enumerate(probabilities, start=1)
        ]
        candidates.sort(key=lambda item: item[0])
        prediction_rows: list[dict[str, str]] = []
        pool_rows: list[CanonicalActiveLearningPoolRow] = []
        for ordinal, (proposal_id, probability) in enumerate(candidates, start=1):
            x = 4 + ordinal * 7
            y = 5 + ordinal * 6
            w = 13
            h = 11
            scale = "1"
            detection_id = hashlib.sha256(
                b"compag-canonical-detection-v1\0"
                + proposal_id.encode("ascii")
                + b"\0"
                + scale.encode("ascii")
            ).hexdigest()
            prediction_rows.append(
                {
                    "id": str(ordinal),
                    "detection_id": detection_id,
                    "proposal_id": proposal_id,
                    "proposal_sha256": proposal_id,
                    "image": self.image_name,
                    "full_image": self.image_name,
                    "image_id": image_id,
                    "image_sha256": image_sha,
                    "group_id": group_id,
                    "tile_name": "orchard__frame_y00000x00000.png",
                    "tile_sha256": _sha("tile"),
                    "scale": scale,
                    "mask_sha256": _sha(f"mask-{ordinal}"),
                    "x": str(x),
                    "y": str(y),
                    "w": str(w),
                    "h": str(h),
                    "bbox_x1": str(x),
                    "bbox_y1": str(y),
                    "bbox_x2": str(x + w),
                    "bbox_y2": str(y + h),
                    "orig_w": str(width),
                    "orig_h": str(height),
                    "poly": json.dumps([x, y, x + w, y, x + w, y + h, x, y + h], separators=(",", ":")),
                    "xgb_p": _format_float(probability),
                    "probability": _format_float(probability),
                    "prediction": str(int(probability >= CANONICAL_DECISION_THRESHOLD)),
                    "kept": str(int(probability >= CANONICAL_DECISION_THRESHOLD)),
                }
            )
            pool_rows.append(
                CanonicalActiveLearningPoolRow(
                    proposal_id=proposal_id,
                    proposal_sha256=proposal_id,
                    image_id=image_id,
                    image_sha256=image_sha,
                    group_id=group_id,
                    scale=1.0,
                    xgb_p=probability,
                )
            )

        self.stage60 = root / "stage60"
        self.stage60.mkdir()
        _csv(self.stage60 / "predictions.csv", CANONICAL_INFERENCE_COLUMNS, prediction_rows)
        predictions_payload = (self.stage60 / "predictions.csv").read_bytes()
        predictions_sha = hashlib.sha256(predictions_payload).hexdigest()
        raw_payload = b"fixture raw feature archive\n"
        (self.stage60 / "raw_features.csv").write_bytes(raw_payload)
        bundle_sha = _sha("bundle")
        feature_archive_sha = _sha("logical-feature-archive")
        raw_receipt = {
            "schema": "compag-curation-canonical-active-learning-raw-features/v1",
            "status": "PASS",
            "profile": CANONICAL_GPU_PROFILE,
            "basename": "raw_features.csv",
            "columns_sha256": _sha("columns"),
            "crop_scales": list(CANONICAL_FEATURE_CROP_SCALES),
            "rows_per_proposal": len(CANONICAL_FEATURE_CROP_SCALES),
            "proposal_count": len(prediction_rows),
            "row_count": len(prediction_rows) * len(CANONICAL_FEATURE_CROP_SCALES),
            "bundle_sha256": bundle_sha,
            "predictions_sha256": predictions_sha,
            "sha256": hashlib.sha256(raw_payload).hexdigest(),
            "size_bytes": len(raw_payload),
        }
        inference = {
            "schema": "compag-curation-canonical-inference/v1",
            "status": "PASS",
            "profile": CANONICAL_GPU_PROFILE,
            "device": "cuda",
            "bundle_sha256": bundle_sha,
            "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
            "threshold": CANONICAL_DECISION_THRESHOLD,
            "threshold_method": "FIXED_CANONICAL_METHOD",
            "full_image_nms_iou": CANONICAL_FULL_IMAGE_NMS_IOU,
            "sam2_execution_points_per_batch": CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
            "image_count": 1,
            "tile_count": 1,
            "proposal_count": len(prediction_rows),
            "prediction_rows": len(prediction_rows),
            "positive_rows": sum(row["prediction"] == "1" for row in prediction_rows),
            "kept_rows": sum(row["kept"] == "1" for row in prediction_rows),
            "predictions_sha256": predictions_sha,
            "raw_feature_archive": raw_receipt,
            "input_inventory": [
                {
                    "image_name": self.image_name,
                    "image_id": image_id,
                    "image_sha256": image_sha,
                    "group_id": group_id,
                    "size_bytes": len(image_payload),
                    "width": width,
                    "height": height,
                    "tile_count": 1,
                    "candidate_rows": len(prediction_rows),
                }
            ],
            "paper_result_reproduction": "NOT_CLAIMED",
        }
        write_new_json(self.stage60 / "inference_result.json", inference)

        self.operation = root / "begin-round"
        pool_root = self.operation / "payload/round_pool"
        selection = self.operation / "payload/selection"
        pool_root.parent.mkdir(parents=True)
        pool_result = write_canonical_active_learning_pool(
            pool_rows,
            pool_root,
            bundle_sha256=bundle_sha,
            source_sha256=raw_receipt["sha256"],
            feature_archive_sha256=feature_archive_sha,
            profile=CANONICAL_GPU_PROFILE,
        )
        selection.mkdir()
        ranked = sorted(
            pool_rows,
            key=lambda row: (abs(float(row.xgb_p) - CANONICAL_DECISION_THRESHOLD), row.proposal_id),
        )
        shortlist_rows: list[dict[str, str]] = []
        for rank, row in enumerate(ranked, start=1):
            distance = abs(float(row.xgb_p) - CANONICAL_DECISION_THRESHOLD)
            uncertainty = max(0.0, 1.0 - min(1.0, distance / float(_policy()["margin"])))
            shortlist_rows.append(
                {
                    "proposal_id": row.proposal_id,
                    "proposal_sha256": row.proposal_sha256,
                    "image_id": row.image_id,
                    "image_sha256": row.image_sha256,
                    "group_id": row.group_id,
                    "scale": "1",
                    "xgb_p": _format_float(float(row.xgb_p)),
                    "distance_to_threshold": _format_float(distance),
                    "uncertainty": _format_float(uncertainty),
                    "selection_rank": str(rank),
                }
            )
        _csv(selection / "shortlist.csv", CANONICAL_AL_SHORTLIST_COLUMNS, shortlist_rows)
        write_review_export(
            [
                {
                    "proposal_id": row.proposal_id,
                    "proposal_sha256": row.proposal_sha256,
                    "image_id": row.image_id,
                    "image_sha256": row.image_sha256,
                    "group_id": row.group_id,
                }
                for row in ranked
            ],
            selection / "review_request.csv",
            canonical_actions=True,
        )
        _pool_values, pool_value, pool_records = _read_pool(pool_root, require_scores=True)
        _shortlist_values, shortlist_record = _read_csv(
            selection / "shortlist.csv",
            CANONICAL_AL_SHORTLIST_COLUMNS,
            (
                "transfer active-learning shortlist"
                if transfer_scoped
                else "active-learning shortlist"
            ),
        )
        review_request_payload = selection / "review_request.csv"
        review_record = _input_record(
            review_request_payload,
            (
                "transfer active-learning review request"
                if transfer_scoped
                else "active-learning review request"
            ),
            max_bytes=256 * 1024 * 1024,
        )
        if transfer_scoped:
            source_groups = ["historical-a", "historical-b"]
            chain = {
                "transfer_baseline": {
                    "archive_sha256": _sha("transfer-baseline"),
                    "source_sha256": _sha("historical-source"),
                    "source_size_bytes": 582_968_556,
                    "split_sha256": _sha("transfer-split"),
                    "source_groups_sha256": compact_json_sha256(source_groups),
                    "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
                    "row_count": 40,
                    "proposal_count": 10,
                    "train_groups_sha256": compact_json_sha256(["historical-a"]),
                    "test_groups_sha256": compact_json_sha256(["historical-b"]),
                },
                "prior_round_completion_sha256": None,
                "prior_review_log_sha256": None,
                "source_scorer": {
                    "kind": "PUBLISHED_R92_TRANSFER_SCORER",
                    "scorer_sha256": bundle_sha,
                    "scores_sha256": predictions_sha,
                    "pool_scores_sha256": pool_result["predictions_sha256"],
                    "feature_state_sha256": _sha("r92-feature-state"),
                    "prototype_sha256": _sha("r92-prototype"),
                    "pca_components_sha256": _sha("r92-components"),
                    "pca_mean_sha256": _sha("r92-mean"),
                    "r92_classifier_sha256": _sha("r92-classifier"),
                    "r92_resource_manifest_sha256": _sha("r92-resources"),
                    "preset_id": "compag-cj-r92",
                },
                "source_predictions_sha256": pool_result["predictions_sha256"],
                "source_pool_source_sha256": raw_receipt["sha256"],
                "source_feature_archive_sha256": feature_archive_sha,
            }
        else:
            chain = {
                "genesis_initial_completion_sha256": _sha("genesis"),
                "prior_round_completion_sha256": None,
                "prior_review_log_sha256": _sha("review-log"),
                "source_bundle_sha256": bundle_sha,
                "source_predictions_sha256": pool_result["predictions_sha256"],
                "source_pool_source_sha256": raw_receipt["sha256"],
                "source_feature_archive_sha256": feature_archive_sha,
                "frozen_split_sha256": _sha("split"),
            }
        image = {
            "image_id": image_id,
            "image_sha256": image_sha,
            "group_id": group_id,
            "proposal_count": len(pool_rows),
        }
        if image_scoped:
            chain["genesis_stage20_features_sha256"] = _sha(
                "genesis-stage20-features"
            )
            chain["genesis_split_sha256"] = _sha("split")
        round_id = compact_json_sha256(
            {
                "domain": (
                    "compag-curation-transfer-image-round-id/v1"
                    if transfer_scoped
                    else
                    "compag-curation-canonical-active-learning-image-round-id/v3"
                    if image_scoped
                    else "compag-curation-canonical-active-learning-round-id/v2"
                ),
                "round_number": 1,
                "policy": _policy(),
                "chain": chain,
                "shortlist_sha256": shortlist_record["sha256"],
                **(
                    {
                        "workflow": (
                            "SINGLE_NEW_IMAGE_POST_R92_TRANSFER_V1"
                            if transfer_scoped
                            else "SINGLE_NEW_IMAGE_UNCERTAINTY_BATCH_FULL_RETRAIN_V1"
                        ),
                        "image": image,
                    }
                    if image_scoped or transfer_scoped
                    else {}
                ),
            }
        )
        manifest = {
            "schema": (
                "compag-curation-canonical-transfer-active-learning-selection/v1"
                if transfer_scoped
                else
                "compag-curation-canonical-active-learning-selection/v3"
                if image_scoped
                else "compag-curation-canonical-active-learning-selection/v2"
            ),
            "status": "PAUSED_FOR_REVIEW",
            "profile": CANONICAL_GPU_PROFILE,
            "kind": SOURCE_KIND,
            "round_number": 1,
            "round_id": round_id,
            "created_at_utc": "2026-08-31T00:00:00Z",
            "policy": _policy(),
            "chain": chain,
            "inputs": {
                "pool": {"files": list(pool_records), "result": dict(pool_value)},
                **(
                    {"decision_log": None}
                    if transfer_scoped
                    else {
                        "frozen_split": {"role": "frozen split", "basename": "split.json", "sha256": _sha("split"), "size_bytes": 1},
                        "decision_log": {"role": "decision log", "basename": "log.csv", "sha256": _sha("review-log"), "size_bytes": 1},
                    }
                ),
            },
            "counts": {
                "pool_rows": len(pool_rows),
                "reviewed_excluded": 0,
                "frozen_test_group_excluded": 0,
                "outside_margin_excluded": 0,
                "margin_eligible": len(pool_rows),
                "selected": len(ranked),
            },
            "shortlist": shortlist_record,
            "review_request": review_record,
            "review_contract": "CANONICAL_ACTION_WEIGHTED_V2",
            "paper_result_reproduction": "NOT_CLAIMED",
        }
        if image_scoped or transfer_scoped:
            manifest["workflow"] = (
                "SINGLE_NEW_IMAGE_POST_R92_TRANSFER_V1"
                if transfer_scoped
                else "SINGLE_NEW_IMAGE_UNCERTAINTY_BATCH_FULL_RETRAIN_V1"
            )
            manifest["image"] = image
        if transfer_scoped:
            manifest["scientific_lineage"] = {
                "source_corpus_role": "POST_R92_REVIEWED_TRANSFER_BASELINE",
                "reproduction_claim": "NOT_R92_REPRODUCTION",
                "feature_state_policy": "FROZEN_PUBLISHED_R92_PCA_AND_PROTOTYPE",
            }
        write_new_json(selection / "round_manifest.json", manifest)
        self.stage_record = _directory_record(self.stage60, "canonical_inference_root")
        self.operation_id = str(uuid.uuid4())
        self.operation_result = {
            "root": self.operation,
            "status": {
                "operation": (
                    "begin_transfer_image_round"
                    if transfer_scoped
                    else "begin_image_round"
                    if image_scoped
                    else "begin_round"
                ),
                "status": "PAUSED_FOR_REVIEW",
                "exit_code": 3,
                "round_number": 1,
                "operation_id": self.operation_id,
            },
            "inputs": (self.stage_record,),
            "output_manifest_sha256": _sha("outer-output-manifest"),
        }
        expected_operation = (
            "begin_transfer_image_round"
            if transfer_scoped
            else "begin_image_round"
            if image_scoped
            else "begin_round"
        )

        def verify_operation(
            _root: Path,
            *,
            operation: str,
            **_kwargs: object,
        ) -> dict[str, object]:
            expected_round = _kwargs.get("round_number", ...)
            if operation != expected_operation:
                raise PublicIOError("operation kind mismatch")
            if expected_round is not ... and expected_round != 1:
                raise PublicIOError("operation round mismatch")
            return self.operation_result

        self.verifier = mock.patch(
            "compag_curation.review.round_session._verify_published_operation",
            side_effect=verify_operation,
        )
        self.verifier.start()

    def close(self) -> None:
        self.verifier.stop()

    def open(self) -> RoundReviewSession:
        return RoundReviewSession.open(
            self.operation,
            self.stage60,
            self.image_root,
            self.root / "reviewed-round.csv",
            self.root / "round-state",
        )


class RoundReviewSessionTests(unittest.TestCase):
    def test_transfer_selection_uses_the_same_durable_reviewer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _RoundFixture(
                Path(directory),
                probabilities=(0.49, 0.54),
                transfer_scoped=True,
            )
            self.addCleanup(fixture.close)
            with fixture.open() as session:
                self.assertEqual(session.index.groups, ("orchard",))
                self.assertEqual(len(session.index.order), 2)
                self.assertEqual(session.summary()["status"], "IN_PROGRESS")

    def test_sequential_image_selection_uses_the_same_durable_reviewer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _RoundFixture(
                Path(directory),
                probabilities=(0.49, 0.54),
                image_scoped=True,
            )
            self.addCleanup(fixture.close)
            with fixture.open() as session:
                self.assertEqual(session.index.groups, ("orchard",))
                self.assertEqual(len(session.index.order), 2)
                self.assertEqual(session.summary()["status"], "IN_PROGRESS")

    def test_happy_path_uses_selection_rank_and_verified_visual(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _RoundFixture(Path(directory), probabilities=(0.66, 0.49, 0.54))
            self.addCleanup(fixture.close)
            with fixture.open() as session:
                ranks = [session.index.proposals[item].selection_rank for item in session.index.order]
                self.assertEqual(ranks, [1, 2, 3])
                probabilities = [session.index.proposals[item].xgb_p for item in session.index.order]
                self.assertEqual(probabilities, [0.49, 0.54, 0.66])
                summary = session.summary()
                self.assertEqual(summary["source_kind"], SOURCE_KIND)
                self.assertEqual(summary["round_number"], 1)
                payload = session.proposal_payload(session.index.order[0])
                self.assertIn("xgb_p", payload["proposal"])
                self.assertNotIn("predicted_iou", payload["proposal"])
                png = session.visual_bytes(0)
                self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(struct.unpack(">II", png[16:24]), (512, 512))

                full_image = session.scene_payload(session.index.order[0])
                scene = full_image["scene"]
                self.assertEqual(scene["kind"], "VERIFIED_STAGE60_ORIGINAL_FULL_IMAGE")
                self.assertEqual((scene["width"], scene["height"]), (64, 64))
                self.assertEqual(scene["overlay_defaults"], {"mask_outline": True, "bbox": False})
                self.assertEqual(scene["reviewable_proposal_count"], 3)
                self.assertTrue(all(item["reviewable"] for item in scene["proposals"]))
                self.assertTrue(all("model" in item for item in scene["proposals"]))
                self.assertEqual(
                    {item["model"]["prediction"] for item in scene["proposals"]},
                    {0, 1},
                )
                scene_png = session.scene_bytes(scene["scene_id"])
                self.assertEqual(scene_png[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(struct.unpack(">II", scene_png[16:24]), (64, 64))

    def test_stage60_tamper_breaks_exact_begin_round_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _RoundFixture(Path(directory), probabilities=(0.49,))
            self.addCleanup(fixture.close)
            with (fixture.stage60 / "predictions.csv").open("ab") as handle:
                handle.write(b"tamper\n")
            with self.assertRaisesRegex(PublicIOError, "differs from begin-round input"):
                fixture.open()

    def test_image_tamper_is_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _RoundFixture(Path(directory), probabilities=(0.49,))
            self.addCleanup(fixture.close)
            with fixture.open() as session:
                session.set_decision(session.index.order[0], "target")
                fixture.image_path.write_bytes(fixture.image_path.read_bytes() + b"tamper")
                with self.assertRaisesRegex(PublicIOError, "original image"):
                    session.finalize()
                self.assertFalse(session.output.exists())

    def test_one_row_round_finalizes_without_split_feasibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _RoundFixture(Path(directory), probabilities=(0.49,))
            self.addCleanup(fixture.close)
            with fixture.open() as session:
                proposal_id = session.index.order[0]
                session.set_decision(proposal_id, "target_uncertain", request_id="one-row-request")
                final = session.finalize()
                self.assertEqual(final["status"], "PASS")
                self.assertEqual(final["rows"], 1)
                self.assertNotIn("split_feasibility", final)
                rows, result = validate_review_table(session.snapshot.review_request, session.output)
                self.assertEqual(len(rows), 1)
                self.assertEqual(result["action_counts"]["sus_accept"], 1)
            request_rows, _meta = read_review_export(fixture.operation / "payload/selection/review_request.csv")
            self.assertEqual(len(request_rows), 1)

    def test_trailing_event_deletion_is_detected_on_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _RoundFixture(Path(directory), probabilities=(0.49, 0.51))
            self.addCleanup(fixture.close)
            with fixture.open() as session:
                session.set_decision(session.index.order[0], "target", request_id="trailing-event-01")
                session.set_decision(session.index.order[1], "other", request_id="trailing-event-02")
            (fixture.root / "round-state/events/00000002.json").unlink()
            with self.assertRaisesRegex(PublicIOError, "event head is ahead"):
                fixture.open()

    def test_event_deletion_while_open_blocks_final_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _RoundFixture(Path(directory), probabilities=(0.49,))
            self.addCleanup(fixture.close)
            with fixture.open() as session:
                session.set_decision(session.index.order[0], "target", request_id="open-delete-01")
                (fixture.root / "round-state/events/00000001.json").unlink()
                with self.assertRaisesRegex(PublicIOError, "event head is ahead"):
                    session.finalize()
                self.assertFalse(session.output.exists())

    def test_event_head_hash_tamper_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _RoundFixture(Path(directory), probabilities=(0.49,))
            self.addCleanup(fixture.close)
            with fixture.open() as session:
                session.set_decision(session.index.order[0], "target", request_id="head-tamper-01")
            head_path = fixture.root / "round-state/EVENT_HEAD.json"
            head = json.loads(head_path.read_text(encoding="ascii"))
            head["event_sha256"] = "f" * 64
            head_path.write_text(
                json.dumps(head, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(PublicIOError, "event head hash"):
                fixture.open()

    def test_valid_event_ahead_of_head_is_recovered_after_crash_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _RoundFixture(Path(directory), probabilities=(0.49, 0.51))
            self.addCleanup(fixture.close)
            head_path = fixture.root / "round-state/EVENT_HEAD.json"
            with fixture.open() as session:
                session.set_decision(session.index.order[0], "target", request_id="ahead-event-01")
                prior_head = head_path.read_bytes()
                session.set_decision(session.index.order[1], "other", request_id="ahead-event-02")
            # Model the only recoverable commit window: the event rename reached
            # durable storage but the separately atomic head advancement did not.
            head_path.write_bytes(prior_head)
            with fixture.open() as recovered:
                self.assertEqual(recovered.summary()["reviewed"], 2)
                self.assertEqual(recovered.summary()["event_sequence"], 2)
            repaired = json.loads(head_path.read_text(encoding="ascii"))
            self.assertEqual(repaired["sequence"], 2)

    def test_session_timestamp_is_bound_by_event_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _RoundFixture(Path(directory), probabilities=(0.49,))
            self.addCleanup(fixture.close)
            with fixture.open():
                pass
            receipt_path = fixture.root / "round-state/SESSION.json"
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            receipt["created_at_utc"] = "2026-08-30T00:00:00Z"
            receipt_path.write_text(
                json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(PublicIOError, "event head binding"):
                fixture.open()

    def test_original_image_directory_member_count_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _RoundFixture(Path(directory), probabilities=(0.49,))
            self.addCleanup(fixture.close)
            for index in range(MAX_IMAGE_DIRECTORY_EXTRA_MEMBERS + 1):
                (fixture.image_root / f"ignored-{index:03d}.txt").write_bytes(b"")
            with self.assertRaisesRegex(PublicIOError, "member-count bound"):
                fixture.open()

    def test_event_member_and_mutation_counts_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _RoundFixture(Path(directory), probabilities=(0.49, 0.51))
            self.addCleanup(fixture.close)
            limit = mock.patch("compag_curation.review.round_session.MAX_EVENT_COUNT", 1)
            with limit:
                with fixture.open() as session:
                    session.set_decision(session.index.order[0], "target", request_id="bounded-event-01")
                    with self.assertRaisesRegex(PublicIOError, "event-count limit"):
                        session.set_decision(session.index.order[1], "other", request_id="bounded-event-02")
            with fixture.open() as session:
                session.set_decision(session.index.order[1], "other", request_id="bounded-event-03")
            with mock.patch("compag_curation.review.round_session.MAX_EVENT_COUNT", 1):
                with self.assertRaisesRegex(PublicIOError, "member-count bound"):
                    fixture.open()

    def test_reopen_rejects_symlinked_output_before_any_unbounded_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _RoundFixture(Path(directory), probabilities=(0.49,))
            self.addCleanup(fixture.close)
            with fixture.open() as session:
                session.set_decision(session.index.order[0], "target")
                session.finalize()
                output = session.output
            output.unlink()
            sentinel = Path(directory) / "sentinel.txt"
            sentinel.write_text("must not be hashed through a symlink\n", encoding="ascii")
            output.symlink_to(sentinel)

            with mock.patch(
                "compag_curation.review.round_session.sha256_file",
                side_effect=AssertionError("unsafe pre-validation hash"),
            ):
                with self.assertRaisesRegex(PublicIOError, "single-link"):
                    fixture.open()


if __name__ == "__main__":
    unittest.main()
