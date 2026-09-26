from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from compag_curation import cli
from compag_curation.canonical.spec import (
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_GPU_PROFILE,
)
from compag_curation.public_io import (
    PublicIOError,
    canonical_json_bytes,
    compact_json_sha256,
)
from compag_curation.review.exchange import CANONICAL_REVIEW_COLUMNS
from compag_curation.review import published_model
from compag_curation.review.published_model import (
    ASSISTANCE_FILE_SCHEMA,
    PRESET_DISPLAY_NAME,
    PRESET_ID,
    PRESET_RESOURCE_MANIFEST_SHA256,
    PRESET_THRESHOLD,
    SCORE_SCHEMA,
    PublishedModelContext,
    PublishedModelScore,
    _load_assets,
)
from compag_curation.review.session import (
    HUMAN_DECISION_ORIGIN,
    MODEL_ASSISTED_SESSION_SCHEMA,
    MODEL_DECISION_ORIGIN,
    ProposalView,
    ReviewIndex,
    ReviewSession,
    TileView,
)


class _Snapshot:
    def __init__(self, root: Path) -> None:
        self.root = root / "sealed-run"
        self.run_id = "00000000-0000-4000-8000-000000000170"
        self.stage10_receipt_sha256 = "8" * 64
        self.stage20_receipt_sha256 = "9" * 64
        self.review_request = root / "sealed-review-request.csv"
        self.review_request_sha256 = "a" * 64
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _fixture_index(root: Path) -> tuple[ReviewIndex, tuple[str, ...]]:
    proposal_ids = tuple(f"{index:064x}" for index in range(1, 4))
    image_id = "c" * 64
    image_sha256 = "d" * 64
    rows: list[dict[str, str]] = []
    proposals: dict[str, ProposalView] = {}
    for index, proposal_id in enumerate(proposal_ids, start=1):
        rows.append(
            {
                "proposal_id": proposal_id,
                "proposal_sha256": proposal_id,
                "image_id": image_id,
                "image_sha256": image_sha256,
                "group_id": "group-1",
                "label": "",
                "review_action": "",
                "review_weight": "",
                "review_status": "pending",
            }
        )
        proposals[proposal_id] = ProposalView(
            proposal_id=proposal_id,
            image_id=image_id,
            image_name="fixture.jpg",
            group_id="group-1",
            tile_name="fixture_y00000x00000.png",
            tile_index=0,
            proposal_index=index,
            label_prefix=proposal_id[:8],
            tile_x=0,
            tile_y=0,
            bbox_x=index,
            bbox_y=index,
            bbox_w=8,
            bbox_h=8,
            predicted_iou=0.9,
            stability_score=0.8,
        )

    overlay = root / "fixture-overlay.png"
    if not overlay.exists():
        overlay.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    with (root / "sealed-review-request.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CANONICAL_REVIEW_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    return (
        ReviewIndex(
            request_rows=tuple(rows),
            request_columns=CANONICAL_REVIEW_COLUMNS,
            order=proposal_ids,
            proposals=proposals,
            tiles=(
                TileView(
                    index=0,
                    tile_name="fixture_y00000x00000.png",
                    overlay_path=overlay,
                    overlay_sha256=hashlib.sha256(
                        overlay.read_bytes()
                    ).hexdigest(),
                    proposal_ids=proposal_ids,
                ),
            ),
            groups=("group-1",),
            overlay_manifest_sha256="e" * 64,
            proposals_sha256="f" * 64,
        ),
        proposal_ids,
    )


def _model_context(
    snapshot: _Snapshot,
    proposal_ids: tuple[str, ...],
) -> PublishedModelContext:
    probabilities = (0.51, 0.90, 0.10)
    scores = {
        proposal_id: PublishedModelScore(
            probability=probability,
            prediction=int(probability >= PRESET_THRESHOLD),
            uncertainty=abs(probability - PRESET_THRESHOLD),
        )
        for proposal_id, probability in zip(
            proposal_ids, probabilities, strict=True
        )
    }
    score_rows = [
        {
            "proposal_id": proposal_id,
            "probability_hex": score.probability.hex(),
            "prediction": score.prediction,
        }
        for proposal_id, score in sorted(scores.items())
    ]
    feature_source_sha256 = "4" * 64
    scores_sha256 = compact_json_sha256(
        {
            "schema": SCORE_SCHEMA,
            "preset_id": PRESET_ID,
            "threshold": PRESET_THRESHOLD,
            "feature_source_sha256": feature_source_sha256,
            "rows": score_rows,
        }
    )
    order = tuple(
        sorted(
            scores,
            key=lambda proposal_id: (
                scores[proposal_id].uncertainty,
                proposal_id,
            ),
        )
    )
    navigation_order_sha256 = compact_json_sha256(list(order))
    resource_manifest_sha256 = "5" * 64
    classifier_sha256 = "6" * 64
    feature_state_sha256 = "7" * 64
    assistance_record = {
        "schema": ASSISTANCE_FILE_SCHEMA,
        "preset_id": PRESET_ID,
        "display_name": PRESET_DISPLAY_NAME,
        "assist_mode": "HISTORICAL_R92_TRANSFER_ASSIST",
        "profile": CANONICAL_GPU_PROFILE,
        "threshold_method": "FIXED_MANUSCRIPT_THRESHOLD",
        "threshold": PRESET_THRESHOLD,
        "run_id": snapshot.run_id,
        "review_request_sha256": snapshot.review_request_sha256,
        "stage20_receipt_sha256": snapshot.stage20_receipt_sha256,
        "feature_source_sha256": feature_source_sha256,
        "resource_manifest_sha256": resource_manifest_sha256,
        "classifier_sha256": classifier_sha256,
        "feature_state_sha256": feature_state_sha256,
        "feature_order_sha256": CANONICAL_FEATURE_ORDER_SHA256,
        "scores_sha256": scores_sha256,
        "proposal_count": len(order),
        "navigation_order": list(order),
        "navigation_order_sha256": navigation_order_sha256,
        "scores": score_rows,
        "device": "cuda",
        "lineage_policy": "TRANSFER_ASSIST_NOT_FRESH_CANONICAL_EQUIVALENCE",
    }
    return PublishedModelContext(
        preset_id=PRESET_ID,
        display_name=PRESET_DISPLAY_NAME,
        threshold=PRESET_THRESHOLD,
        order=order,
        scores=scores,
        scene_models={
            proposal_id: {
                "preset_id": PRESET_ID,
                "xgb_p": score.probability,
                "prediction": score.prediction,
                "uncertainty": score.uncertainty,
                "threshold": PRESET_THRESHOLD,
            }
            for proposal_id, score in scores.items()
        },
        resource_manifest_sha256=resource_manifest_sha256,
        classifier_sha256=classifier_sha256,
        feature_state_sha256=feature_state_sha256,
        feature_source_sha256=feature_source_sha256,
        scores_sha256=scores_sha256,
        navigation_order_sha256=navigation_order_sha256,
        assistance_record=assistance_record,
        assistance_file_sha256=hashlib.sha256(
            canonical_json_bytes(assistance_record)
        ).hexdigest(),
    )


def _manual_session(
    root: Path,
    *,
    assisted: bool,
) -> tuple[ReviewSession, tuple[str, ...], PublishedModelContext | None]:
    index, proposal_ids = _fixture_index(root)
    snapshot = _Snapshot(root)
    context = _model_context(snapshot, proposal_ids) if assisted else None
    if context is not None:
        index = replace(index, order=context.order)
        proposal_ids = context.order
    state = root / ("assisted-state" if assisted else "cold-state")
    output = root / ("assisted-reviewed.csv" if assisted else "cold-reviewed.csv")
    if not state.exists() and not state.is_symlink():
        ReviewSession._create_state(
            snapshot,
            index,
            state,
            output,
            model_context=context,
        )
    lock_path = state / "SESSION.LOCK"
    lock_fd = os.open(
        lock_path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        session_sha256 = ReviewSession._validate_receipt(
            snapshot,
            index,
            state,
            output,
            model_context=context,
        )
        session = ReviewSession(
            snapshot,
            index,
            state,
            output,
            lock_fd,
            session_sha256,
            context,
        )
    except BaseException:
        os.close(lock_fd)
        snapshot.close()
        raise
    return session, proposal_ids, context


class PublishedModelAssistedSessionTests(unittest.TestCase):
    def test_published_r92_assets_are_exact_safe_and_93_feature_closed(self) -> None:
        assets = _load_assets()
        self.assertEqual(
            PRESET_RESOURCE_MANIFEST_SHA256,
            "0fb259e5d4bb366d4368068b32f26d54dd361892ed3811c4824d9cdac4e405df",
        )
        self.assertEqual(
            assets.manifest_sha256,
            PRESET_RESOURCE_MANIFEST_SHA256,
        )
        self.assertEqual(
            assets.classifier_sha256,
            "ebcd0761bd4d8b5db466e2724c6c15882286fdc28ab29633c9f4c49350375bc4",
        )
        self.assertEqual(
            hashlib.sha256(assets.classifier_ubj).hexdigest(),
            assets.classifier_sha256,
        )
        self.assertEqual(len(CANONICAL_FEATURE_ORDER), 93)
        self.assertEqual(len(assets.imputer_statistics), 93)
        self.assertEqual(assets.feature_state.prototype.shape, (2048,))
        self.assertEqual(assets.feature_state.pca_mean.shape, (2048,))
        self.assertEqual(
            assets.feature_state.pca_components.shape,
            (32, 2048),
        )

        resource_root = published_model._resource_root()
        unsafe = sorted(
            path.name
            for path in resource_root.iterdir()
            if path.suffix.lower() in {".pkl", ".pickle", ".joblib"}
        )
        self.assertEqual(unsafe, [])

    def test_published_r92_asset_tampering_fails_closed(self) -> None:
        source = published_model._resource_root()
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "compag_cj_r92"
            shutil.copytree(source, copied)
            classifier = copied / "classifier.ubj"
            payload = bytearray(classifier.read_bytes())
            payload[len(payload) // 2] ^= 0x01
            classifier.write_bytes(payload)
            os.chmod(classifier, 0o644)

            with (
                mock.patch.object(
                    published_model,
                    "_resource_root",
                    return_value=copied,
                ),
                self.assertRaisesRegex(
                    PublicIOError,
                    "preset resource hash changed: classifier[.]ubj",
                ),
            ):
                _load_assets()

    def test_manual_correction_bulk_idempotence_undo_and_reopen_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, proposal_ids, context = _manual_session(
                root, assisted=True
            )
            self.assertIsNotNone(context)
            try:
                session.set_decision(
                    proposal_ids[0],
                    "other",
                    request_id="manual-correction-0001",
                )
                first = session.confirm_model_remainder(
                    request_id="bulk-confirm-0001"
                )
                repeated = session.confirm_model_remainder(
                    request_id="bulk-confirm-0001"
                )
                self.assertEqual(first, repeated)
                self.assertEqual(session.summary()["event_sequence"], 2)
                self.assertEqual(
                    session._decision_origins,
                    {
                        proposal_ids[0]: HUMAN_DECISION_ORIGIN,
                        proposal_ids[1]: MODEL_DECISION_ORIGIN,
                        proposal_ids[2]: MODEL_DECISION_ORIGIN,
                    },
                )
                rows = {
                    row["proposal_id"]: row
                    for row in csv.DictReader(
                        io.StringIO(
                            session._render_csv().decode("utf-8")
                        )
                    )
                }
                self.assertEqual(
                    (
                        rows[proposal_ids[0]]["label"],
                        rows[proposal_ids[0]]["review_action"],
                        rows[proposal_ids[0]]["review_weight"],
                    ),
                    ("0", "flip", "1.0"),
                )
                self.assertEqual(
                    (
                        rows[proposal_ids[1]]["label"],
                        rows[proposal_ids[1]]["review_action"],
                        rows[proposal_ids[1]]["review_weight"],
                    ),
                    ("1", "sus_accept", "0.4"),
                )
                self.assertEqual(
                    (
                        rows[proposal_ids[2]]["label"],
                        rows[proposal_ids[2]]["review_action"],
                        rows[proposal_ids[2]]["review_weight"],
                    ),
                    ("0", "sus_flip", "0.4"),
                )
            finally:
                session.close()

            replayed, replayed_ids, _context = _manual_session(
                root, assisted=True
            )
            try:
                self.assertEqual(replayed_ids, proposal_ids)
                self.assertEqual(replayed.summary()["reviewed"], 3)
                self.assertEqual(
                    replayed.summary()["decision_origins"]["counts"],
                    {
                        HUMAN_DECISION_ORIGIN: 1,
                        MODEL_DECISION_ORIGIN: 2,
                    },
                )
                replayed.confirm_model_remainder(
                    request_id="bulk-confirm-0001"
                )
                self.assertEqual(replayed.summary()["event_sequence"], 2)
                replayed.undo(request_id="bulk-undo-0001")
                self.assertEqual(replayed.summary()["reviewed"], 1)
                self.assertEqual(
                    replayed.proposal_payload(proposal_ids[0])["proposal"][
                        "decision"
                    ]["choice"],
                    "other",
                )
                self.assertIsNone(
                    replayed.proposal_payload(proposal_ids[1])["proposal"][
                        "decision"
                    ]
                )
                self.assertIsNone(
                    replayed.proposal_payload(proposal_ids[2])["proposal"][
                        "decision"
                    ]
                )
            finally:
                replayed.close()

            replayed_again, _ids, _context = _manual_session(
                root, assisted=True
            )
            try:
                summary = replayed_again.summary()
                self.assertEqual(summary["event_sequence"], 3)
                self.assertEqual(summary["reviewed"], 1)
                self.assertEqual(
                    summary["decision_origins"]["counts"],
                    {
                        HUMAN_DECISION_ORIGIN: 1,
                        MODEL_DECISION_ORIGIN: 0,
                    },
                )
            finally:
                replayed_again.close()

    def test_assisted_receipt_cache_and_tampered_bulk_event_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, proposal_ids, context = _manual_session(
                root, assisted=True
            )
            assert context is not None
            state = root / "assisted-state"
            try:
                cache = state / "MODEL_ASSISTANCE.json"
                self.assertEqual(
                    cache.read_bytes(),
                    canonical_json_bytes(context.assistance_record),
                )
                self.assertEqual(stat.S_IMODE(cache.stat().st_mode), 0o600)
                self.assertEqual(
                    hashlib.sha256(cache.read_bytes()).hexdigest(),
                    context.assistance_file_sha256,
                )
                receipt = json.loads(
                    (state / "SESSION.json").read_text(encoding="ascii")
                )
                self.assertEqual(
                    receipt["schema"], MODEL_ASSISTED_SESSION_SCHEMA
                )
                self.assertEqual(
                    receipt["published_model_assist"],
                    context.receipt_record(),
                )
                session.set_decision(
                    proposal_ids[0],
                    "other",
                    request_id="tamper-manual-0001",
                )
                session.confirm_model_remainder(
                    request_id="tamper-bulk-0001"
                )
            finally:
                session.close()

            event_path = state / "events" / "00000002.json"
            event = json.loads(event_path.read_text(encoding="ascii"))
            event["proposal_count"] = int(event["proposal_count"]) - 1
            event["event_sha256"] = compact_json_sha256(
                {
                    key: value
                    for key, value in event.items()
                    if key != "event_sha256"
                }
            )
            event_path.write_bytes(canonical_json_bytes(event))
            os.chmod(event_path, 0o600)
            with self.assertRaisesRegex(
                PublicIOError, "confirmation set changed"
            ):
                _manual_session(root, assisted=True)

    def test_cold_session_rejects_model_remainder_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session, _proposal_ids, _context = _manual_session(
                Path(directory), assisted=False
            )
            try:
                with self.assertRaisesRegex(
                    PublicIOError, "assistance is not active"
                ):
                    session.confirm_model_remainder(
                        request_id="cold-bulk-reject-0001"
                    )
                self.assertEqual(session.summary()["event_sequence"], 0)
            finally:
                session.close()

    def test_cli_rejects_r92_with_any_non_cj_label_vocabulary(self) -> None:
        cases = (
            ("Target", "Non-target"),
            ("CJ", "Other"),
            ("cj", "Non-CJ"),
        )
        for target, non_target in cases:
            with (
                self.subTest(labels=(target, non_target)),
                contextlib.redirect_stderr(io.StringIO()) as stderr,
                self.assertRaises(SystemExit) as raised,
            ):
                cli.main(
                    [
                        "review-ui",
                        "status",
                        "--run",
                        "/tmp/sealed-run",
                        "--output",
                        "/tmp/reviewed.csv",
                        "--model-preset",
                        PRESET_ID,
                        "--target-label",
                        target,
                        "--non-target-label",
                        non_target,
                    ]
                )
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("fixes the labels", stderr.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
