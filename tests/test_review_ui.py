from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import http.client
import io
import json
import os
import socket
import stat
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from unittest import mock

from compag_curation import cli
from compag_curation.public_io import (
    PublicIOError,
    canonical_json_bytes,
    compact_json_sha256,
    write_new_json,
)
from compag_curation.review.desktop import (
    DesktopReviewController,
    _CHOICE_LABELS,
    _TkReviewer,
)
from compag_curation.review.exchange import CANONICAL_REVIEW_COLUMNS
from compag_curation.review.session import (
    DECISIONS,
    ProposalView,
    ReviewIndex,
    ReviewSession,
    TileView,
    _external_path,
)
from compag_curation.review.web import (
    _HTML,
    LocalReviewPageOpener,
    create_server,
    open_local_review_page,
)


EXPECTED_DECISIONS = {
    "target": (1, "accept", 1.0),
    "other": (0, "flip", 1.0),
    "target_uncertain": (1, "sus_accept", 0.4),
    "other_uncertain": (0, "sus_flip", 0.4),
    "skip": (0, "skip", 0.0),
}


class _Snapshot:
    def __init__(self, root: Path) -> None:
        self.root = root / "sealed-run"
        self.run_id = "00000000-0000-4000-8000-000000000001"
        self.stage10_receipt_sha256 = "8" * 64
        self.stage20_receipt_sha256 = "9" * 64
        self.review_request = root / "sealed-review-request.csv"
        self.review_request_sha256 = "a" * 64
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _manual_session(
    root: Path,
    *,
    proposal_count: int = 2,
) -> tuple[ReviewSession, tuple[str, ...], _Snapshot]:
    """Create a tiny non-scientific session without opening a sealed run."""

    state = root / "state"
    initialize_state = not state.exists() and not state.is_symlink()
    events = state / "events"
    staging = state / ".staging"
    for directory in (state, events, staging):
        if not directory.exists():
            directory.mkdir(mode=0o700)
        os.chmod(directory, 0o700)

    lock_path = state / "TEST.LOCK"
    if not lock_path.exists():
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
    os.chmod(lock_path, 0o600)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)

    proposal_ids = tuple(f"{index + 1:064x}" for index in range(proposal_count))
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
    overlay_digest = hashlib.sha256(overlay.read_bytes()).hexdigest()
    index = ReviewIndex(
        request_rows=tuple(rows),
        request_columns=CANONICAL_REVIEW_COLUMNS,
        order=proposal_ids,
        proposals=proposals,
        tiles=(
            TileView(
                index=0,
                tile_name="fixture_y00000x00000.png",
                overlay_path=overlay,
                overlay_sha256=overlay_digest,
                proposal_ids=proposal_ids,
            ),
        ),
        groups=("group-1",),
        overlay_manifest_sha256="e" * 64,
        proposals_sha256="f" * 64,
    )
    review_request = root / "sealed-review-request.csv"
    with review_request.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CANONICAL_REVIEW_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    snapshot = _Snapshot(root)
    try:
        output = root / "reviewed.csv"
        if initialize_state:
            receipt = ReviewSession._receipt(
                snapshot,
                index,
                output,
                created_at_utc="2026-08-31T00:00:00Z",
            )
            write_new_json(state / "SESSION.json", receipt, mode=0o600)
            session_sha256 = hashlib.sha256(
                canonical_json_bytes(receipt)
            ).hexdigest()
            write_new_json(
                state / "EVENT_HEAD.json",
                ReviewSession._head_record(
                    snapshot,
                    index,
                    session_sha256,
                    0,
                    "0" * 64,
                ),
                mode=0o600,
            )
        session_sha256 = ReviewSession._validate_receipt(
            snapshot,
            index,
            state,
            output,
        )
        session = ReviewSession(
            snapshot,
            index,
            state,
            output,
            lock_fd,
            session_sha256,
        )
    except BaseException:
        os.close(lock_fd)
        snapshot.close()
        raise
    return session, proposal_ids, snapshot


class ReviewDecisionAndJournalTests(unittest.TestCase):
    def test_atomic_publication_name_length_is_rejected_before_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / ("x" * 220 + ".csv")
            derived = f".{output.name}.compag-review.00000000-0000-4000-8000-000000000001.tmp"
            with self.assertRaisesRegex(PublicIOError, "too long"):
                _external_path(
                    output,
                    root / "run",
                    "review output",
                    derived_basename=derived,
                )

    def test_five_choices_map_to_the_canonical_label_action_and_weight(self) -> None:
        self.assertEqual(set(DECISIONS), set(EXPECTED_DECISIONS))
        for choice, expected in EXPECTED_DECISIONS.items():
            with self.subTest(choice=choice):
                decision = DECISIONS[choice]
                self.assertEqual(decision.choice, choice)
                self.assertEqual(
                    (decision.label, decision.review_action, decision.review_weight),
                    expected,
                )

    def test_rendered_rows_preserve_all_five_choice_mappings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session, proposal_ids, _snapshot = _manual_session(
                Path(directory), proposal_count=len(EXPECTED_DECISIONS)
            )
            try:
                for proposal_id, choice in zip(proposal_ids, EXPECTED_DECISIONS, strict=True):
                    session.set_decision(proposal_id, choice)
                rows = list(
                    csv.DictReader(io.StringIO(session._render_csv().decode("utf-8")))
                )
            finally:
                session.close()

        self.assertEqual(len(rows), len(EXPECTED_DECISIONS))
        for row, choice in zip(rows, EXPECTED_DECISIONS, strict=True):
            label, action, weight = EXPECTED_DECISIONS[choice]
            self.assertEqual(row["label"], str(label))
            self.assertEqual(row["review_action"], action)
            self.assertEqual(row["review_weight"], f"{weight:.1f}")
            self.assertEqual(row["review_status"], "reviewed")

    def test_append_only_journal_recovers_undo_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, proposal_ids, first_snapshot = _manual_session(root)
            first.set_decision(proposal_ids[0], "target")
            first.set_decision(proposal_ids[1], "skip")
            first.undo()
            self.assertEqual(first.summary()["event_sequence"], 3)
            first.close()
            self.assertTrue(first_snapshot.closed)

            event_paths = sorted((root / "state" / "events").iterdir())
            self.assertEqual([path.name for path in event_paths], [
                "00000001.json", "00000002.json", "00000003.json"
            ])
            previous_hash = "0" * 64
            for sequence, path in enumerate(event_paths, start=1):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                event = json.loads(path.read_text(encoding="utf-8"))
                supplied_hash = event.pop("event_sha256")
                self.assertEqual(event["sequence"], sequence)
                self.assertEqual(event["previous_event_sha256"], previous_hash)
                self.assertEqual(supplied_hash, compact_json_sha256(event))
                previous_hash = supplied_hash

            recovered, recovered_ids, _snapshot = _manual_session(root)
            try:
                self.assertEqual(recovered_ids, proposal_ids)
                self.assertEqual(recovered.summary()["reviewed"], 1)
                self.assertEqual(
                    recovered.proposal_payload(proposal_ids[0])["proposal"]["decision"]["choice"],
                    "target",
                )
                self.assertIsNone(
                    recovered.proposal_payload(proposal_ids[1])["proposal"]["decision"]
                )
            finally:
                recovered.close()

            tampered = json.loads(event_paths[0].read_text(encoding="utf-8"))
            tampered["decision"]["label"] = 0
            event_paths[0].write_text(
                json.dumps(tampered, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            os.chmod(event_paths[0], 0o600)
            descriptor = -1
            try:
                descriptor = os.open(root / "state" / "TEST.LOCK", os.O_RDWR)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                with self.assertRaises(PublicIOError):
                    ReviewSession(
                        _Snapshot(root),
                        recovered.index,
                        root / "state",
                        root / "reviewed.csv",
                        descriptor,
                        hashlib.sha256(
                            (root / "state" / "SESSION.json").read_bytes()
                        ).hexdigest(),
                    )
            finally:
                if descriptor >= 0:
                    os.close(descriptor)

    def test_committed_trailing_event_deletion_is_rejected_by_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, proposal_ids, _snapshot = _manual_session(root)
            session.set_decision(proposal_ids[0], "target")
            session.close()

            (root / "state" / "events" / "00000001.json").unlink()
            with self.assertRaisesRegex(PublicIOError, "head is ahead"):
                _manual_session(root)

    def test_finalize_replays_events_and_rejects_in_open_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, proposal_ids, _snapshot = _manual_session(root)
            try:
                session.set_decision(proposal_ids[0], "target")
                session.set_decision(proposal_ids[1], "other")
                (root / "state" / "events" / "00000002.json").unlink()
                with self.assertRaisesRegex(PublicIOError, "head is ahead"):
                    session.finalize()
                self.assertFalse((root / "reviewed.csv").exists())
                self.assertFalse((root / "state" / "FINAL.json").exists())
            finally:
                session.close()

    def test_event_head_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, proposal_ids, _snapshot = _manual_session(root)
            session.set_decision(proposal_ids[0], "target")
            session.close()

            head_path = root / "state" / "EVENT_HEAD.json"
            head = json.loads(head_path.read_text(encoding="ascii"))
            head["event_sha256"] = "b" * 64
            head_path.write_bytes(canonical_json_bytes(head))
            os.chmod(head_path, 0o600)
            with self.assertRaisesRegex(PublicIOError, "head hash"):
                _manual_session(root)

    def test_valid_event_suffix_ahead_of_head_is_recovered_after_crash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, proposal_ids, _snapshot = _manual_session(root)
            session.set_decision(proposal_ids[0], "target")
            with (
                mock.patch.object(
                    session,
                    "_replace_head",
                    side_effect=OSError("injected head-advance crash"),
                ),
                self.assertRaisesRegex(OSError, "head-advance crash"),
            ):
                session.set_decision(
                    proposal_ids[1],
                    "other",
                    request_id="crash-ahead-0002",
                )
            self.assertEqual(session.summary()["event_sequence"], 1)
            session.close()

            recovered, recovered_ids, _snapshot = _manual_session(root)
            try:
                self.assertEqual(recovered_ids, proposal_ids)
                self.assertEqual(recovered.summary()["event_sequence"], 2)
                self.assertEqual(recovered.summary()["reviewed"], 2)
                head = json.loads(
                    (root / "state" / "EVENT_HEAD.json").read_text(
                        encoding="ascii"
                    )
                )
                event = json.loads(
                    (root / "state" / "events" / "00000002.json").read_text(
                        encoding="ascii"
                    )
                )
                self.assertEqual(head["sequence"], 2)
                self.assertEqual(head["event_sha256"], event["event_sha256"])
            finally:
                recovered.close()

    def test_event_mutation_ceiling_is_finite_and_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, proposal_ids, _snapshot = _manual_session(
                root, proposal_count=1
            )
            try:
                limit = session._event_limit()
                self.assertEqual(limit, 64)
                for sequence in range(1, limit + 1):
                    session.set_decision(
                        proposal_ids[0],
                        "target" if sequence % 2 else "other",
                        request_id=f"bounded-{sequence:06d}",
                    )
                with self.assertRaisesRegex(PublicIOError, "ceiling"):
                    session.set_decision(
                        proposal_ids[0],
                        "skip",
                        request_id="bounded-overflow",
                    )
                self.assertEqual(
                    len(list((root / "state" / "events").iterdir())),
                    limit,
                )
            finally:
                session.close()
            with (
                mock.patch(
                    "compag_curation.review.session.MAX_EVENT_COUNT", 1
                ),
                self.assertRaisesRegex(PublicIOError, "count exceeds"),
            ):
                _manual_session(root, proposal_count=1)

    def test_session_creation_timestamp_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, _proposal_ids, _snapshot = _manual_session(root)
            session.close()

            receipt_path = root / "state" / "SESSION.json"
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            receipt["created_at_utc"] = "2026-09-01T00:00:00Z"
            receipt_path.write_bytes(canonical_json_bytes(receipt))
            os.chmod(receipt_path, 0o600)
            with self.assertRaisesRegex(PublicIOError, "head binding"):
                _manual_session(root)

    def test_finalize_rejects_live_state_not_derived_from_durable_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, proposal_ids, _snapshot = _manual_session(root)
            try:
                session.set_decision(proposal_ids[0], "target")
                session.set_decision(proposal_ids[1], "other")
                session._decisions[proposal_ids[0]] = DECISIONS["skip"]
                with self.assertRaisesRegex(PublicIOError, "differ from the live"):
                    session.finalize()
                self.assertFalse((root / "reviewed.csv").exists())
            finally:
                session.close()

    def test_mutation_request_ids_are_idempotent_and_cannot_be_repurposed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session, proposal_ids, _snapshot = _manual_session(Path(directory))
            try:
                first = session.set_decision(
                    proposal_ids[0], "target", request_id="request-0001"
                )
                repeated = session.set_decision(
                    proposal_ids[0], "target", request_id="request-0001"
                )
                self.assertEqual(first, repeated)
                self.assertEqual(session.summary()["event_sequence"], 1)
                with self.assertRaisesRegex(PublicIOError, "reused"):
                    session.set_decision(
                        proposal_ids[0], "other", request_id="request-0001"
                    )

                undone = session.undo(request_id="request-0002")
                repeated_undo = session.undo(request_id="request-0002")
                self.assertEqual(undone, repeated_undo)
                self.assertEqual(session.summary()["event_sequence"], 2)
                self.assertEqual(session.summary()["reviewed"], 0)
                with self.assertRaisesRegex(PublicIOError, "different action"):
                    session.set_decision(
                        proposal_ids[1], "skip", request_id="request-0002"
                    )
            finally:
                session.close()

    def test_failed_undo_publication_keeps_the_live_undo_stack_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session, proposal_ids, _snapshot = _manual_session(Path(directory))
            try:
                session.set_decision(proposal_ids[0], "target")
                with (
                    mock.patch.object(
                        session,
                        "_append",
                        side_effect=OSError("injected event publication failure"),
                    ),
                    self.assertRaisesRegex(OSError, "injected"),
                ):
                    session.undo(request_id="failed-undo-request")
                self.assertEqual(session.summary()["reviewed"], 1)
                self.assertEqual(
                    session.proposal_payload(proposal_ids[0])["proposal"]["decision"]["choice"],
                    "target",
                )
                session.undo(request_id="successful-undo-request")
                self.assertEqual(session.summary()["reviewed"], 0)
            finally:
                session.close()

    def test_final_receipt_and_private_output_are_fully_revalidated(self) -> None:
        feasibility = {"status": "PASS", "fixture": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, proposal_ids, _snapshot = _manual_session(root)
            try:
                session.set_decision(proposal_ids[0], "target")
                session.set_decision(proposal_ids[1], "other")
                with mock.patch.object(
                    session,
                    "_split_feasibility",
                    return_value=feasibility,
                ):
                    session.finalize()
            finally:
                session.close()

            final_path = root / "state" / "FINAL.json"
            final = json.loads(final_path.read_text(encoding="ascii"))
            final["positive_rows"] = 999
            final_path.write_bytes(canonical_json_bytes(final))
            os.chmod(final_path, 0o600)
            with (
                mock.patch.object(
                    ReviewSession,
                    "_split_feasibility",
                    return_value=feasibility,
                ),
                self.assertRaises(PublicIOError),
            ):
                _manual_session(root)

            final["positive_rows"] = 1
            final_path.write_bytes(canonical_json_bytes(final))
            os.chmod(final_path, 0o600)
            os.chmod(root / "reviewed.csv", 0o644)
            with (
                mock.patch.object(
                    ReviewSession,
                    "_split_feasibility",
                    return_value=feasibility,
                ),
                self.assertRaisesRegex(PublicIOError, "mode-0600"),
            ):
                _manual_session(root)

    def test_postpublication_receipt_error_fails_closed_against_more_decisions(self) -> None:
        feasibility = {"status": "PASS", "fixture": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, proposal_ids, _snapshot = _manual_session(root)
            try:
                session.set_decision(proposal_ids[0], "target")
                session.set_decision(proposal_ids[1], "other")

                def publish_final_then_fail(path: Path, value: object, mode: int = 0o644) -> None:
                    write_new_json(path, value, mode)
                    raise OSError("injected post-publication receipt failure")

                with (
                    mock.patch.object(
                        session,
                        "_split_feasibility",
                        return_value=feasibility,
                    ),
                    mock.patch(
                        "compag_curation.review.session.write_new_json",
                        side_effect=publish_final_then_fail,
                    ),
                    self.assertRaisesRegex(OSError, "post-publication"),
                ):
                    session.finalize()
                self.assertTrue((root / "reviewed.csv").is_file())
                self.assertTrue((root / "state" / "FINAL.json").is_file())
                sequence = session.summary()["event_sequence"]
                with self.assertRaises(PublicIOError):
                    session.set_decision(proposal_ids[0], "other")
                self.assertEqual(session.summary()["event_sequence"], sequence)
            finally:
                session.close()


class ReviewCliTests(unittest.TestCase):
    def test_review_ui_help_is_available_for_every_mode(self) -> None:
        cases = (
            ["review-ui", "--help"],
            ["review-ui", "web", "--help"],
            ["review-ui", "desktop", "--help"],
            ["review-ui", "status", "--help"],
        )
        for argv in cases:
            with self.subTest(argv=argv):
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    cli.main(argv)
                self.assertEqual(raised.exception.code, 0)

    def test_parser_exposes_secure_web_defaults_and_required_paths(self) -> None:
        args = cli._parser().parse_args(
            [
                "review-ui",
                "web",
                "--run",
                "/tmp/a-run",
                "--output",
                "/tmp/reviewed.csv",
                "--no-open-browser",
            ]
        )
        self.assertEqual(args.review_ui_mode, "web")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 0)
        self.assertTrue(args.no_open_browser)
        self.assertEqual(args.run, Path("/tmp/a-run"))
        self.assertEqual(args.output, Path("/tmp/reviewed.csv"))

    def test_cli_dispatches_status_without_scientific_execution(self) -> None:
        with (
            mock.patch(
                "compag_curation.review.facade.run_review_ui",
                return_value=({"status": "IN_PROGRESS", "scientific_execution": False}, 0),
            ) as run_review_ui,
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            result = cli.main(
                [
                    "review-ui",
                    "status",
                    "--run",
                    "/tmp/a-run",
                    "--output",
                    "/tmp/reviewed.csv",
                    "--state",
                    "/tmp/review-state",
                ]
            )
        self.assertEqual(result, 0)
        self.assertFalse(json.loads(stdout.getvalue())["scientific_execution"])
        run_review_ui.assert_called_once_with(
            "status",
            run=Path("/tmp/a-run"),
            selection_root=None,
            stage60_root=None,
            images=None,
            output=Path("/tmp/reviewed.csv"),
            state=Path("/tmp/review-state"),
            host="127.0.0.1",
            port=0,
            open_browser=True,
            target_label="Target",
            non_target_label="Non-target",
            model_preset=None,
        )

    def test_parser_requires_the_complete_round_visual_source_tuple(self) -> None:
        parser = cli._parser()
        parsed = parser.parse_args(
            [
                "review-ui",
                "desktop",
                "--selection-root",
                "/tmp/round-selection",
                "--stage60-root",
                "/tmp/stage60",
                "--images",
                "/tmp/images",
                "--output",
                "/tmp/reviewed.csv",
            ]
        )
        self.assertIsNone(parsed.run)
        self.assertEqual(parsed.selection_root, Path("/tmp/round-selection"))
        self.assertEqual(parsed.stage60_root, Path("/tmp/stage60"))
        self.assertEqual(parsed.images, Path("/tmp/images"))

        invalid_cases = (
            [
                "review-ui", "status", "--selection-root", "/tmp/selection",
                "--output", "/tmp/out.csv",
            ],
            [
                "review-ui", "status", "--run", "/tmp/run",
                "--stage60-root", "/tmp/stage60", "--output", "/tmp/out.csv",
            ],
            [
                "review-ui", "status", "--run", "/tmp/run",
                "--selection-root", "/tmp/selection", "--output", "/tmp/out.csv",
            ],
        )
        for argv in invalid_cases:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    cli.main(argv)
                self.assertEqual(raised.exception.code, 2)

    def test_cli_rejects_non_loopback_host_and_out_of_range_port(self) -> None:
        parser = cli._parser()
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as host_error,
        ):
            parser.parse_args(
                [
                    "review-ui", "web", "--run", "/tmp/run", "--output",
                    "/tmp/out.csv", "--host", "0.0.0.0",
                ]
            )
        self.assertEqual(host_error.exception.code, 2)
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as port_error,
        ):
            cli.main(
                [
                    "review-ui", "web", "--run", "/tmp/run", "--output",
                    "/tmp/out.csv", "--port", "65536", "--no-open-browser",
                ]
            )
        self.assertEqual(port_error.exception.code, 2)


class _FakeWebSession:
    def __init__(self) -> None:
        self.proposal_id = "1" * 64
        self.scene_id = "a" * 64
        self.index = SimpleNamespace(order=(self.proposal_id,))
        self.calls: list[tuple[object, ...]] = []
        self.choice: str | None = None
        self.bulk_request_ids: set[str] = set()
        self.bulk_commit_count = 0

    def summary(self) -> dict[str, object]:
        return {
            "status": "IN_PROGRESS",
            "first_pending": self.proposal_id,
            "remaining": 1,
        }

    def proposal_payload(self, proposal_id: str) -> dict[str, object]:
        self.calls.append(("proposal", proposal_id))
        return {"proposal": {"proposal_id": proposal_id}}

    def scene_payload(self, proposal_id: str) -> dict[str, object]:
        self.calls.append(("scene_payload", proposal_id))
        return {
            "proposal": {
                "proposal_id": proposal_id,
                "decision": (
                    None
                    if self.choice is None
                    else {"choice": self.choice, "label": 1, "review_weight": 1.0}
                ),
            },
            "scene": {
                "scene_id": self.scene_id,
                "width": 640,
                "height": 480,
                "proposals": [
                    {
                        "proposal_id": proposal_id,
                        "label_prefix": proposal_id[:8],
                        "polygon": [1, 1, 9, 1, 9, 9, 1, 9],
                        "bbox": [1, 1, 8, 8],
                        "choice": self.choice,
                        "reviewable": True,
                        "selected": True,
                    }
                ],
                "overlay_defaults": {"mask_outline": True, "bbox": False},
            },
            "summary": self.summary(),
        }

    def scene_bytes(self, scene_id: str) -> bytes:
        self.calls.append(("scene_bytes", scene_id))
        return b"\x89PNG\r\n\x1a\nfixture"

    def set_decision(
        self,
        proposal_id: str,
        choice: str,
        *,
        request_id: str | None = None,
    ) -> dict[str, object]:
        self.calls.append(("decision", proposal_id, choice, request_id))
        self.choice = choice
        return {"proposal": {"proposal_id": proposal_id}}

    def undo(self, *, request_id: str | None = None) -> dict[str, object]:
        self.calls.append(("undo", request_id))
        self.choice = None
        return {"proposal": {"proposal_id": self.proposal_id}}

    def confirm_model_remainder(
        self,
        *,
        request_id: str | None = None,
    ) -> dict[str, object]:
        self.calls.append(("confirm_model_remainder", request_id))
        if request_id not in self.bulk_request_ids:
            self.bulk_request_ids.add(str(request_id))
            self.bulk_commit_count += 1
            self.choice = "target_uncertain"
        return {"proposal": {"proposal_id": self.proposal_id}}

    def navigate(
        self,
        current: str | None,
        direction: int,
        *,
        state_filter: str,
        group: str | None,
    ) -> dict[str, object]:
        self.calls.append(("navigate", current, direction, state_filter, group))
        return {"proposal": {"proposal_id": self.proposal_id}}

    def search(self, prefix: str) -> dict[str, object]:
        self.calls.append(("search", prefix))
        return {"proposal": {"proposal_id": self.proposal_id}}

    def finalize(self) -> dict[str, object]:
        self.calls.append(("finalize",))
        return {"status": "PASS"}


class ReviewWebSecurityTests(unittest.TestCase):
    @staticmethod
    def _request(
        port: int,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            payload = response.read()
            return response.status, {name.lower(): value for name, value in response.getheaders()}, payload
        finally:
            connection.close()

    def test_server_rejects_every_non_loopback_bind(self) -> None:
        session = _FakeWebSession()
        for host in ("0.0.0.0", "localhost", "::1", "127.0.0.2"):
            with self.subTest(host=host), self.assertRaisesRegex(PublicIOError, "127.0.0.1"):
                create_server(session, host=host)
        for port in (-1, 65536):
            with self.subTest(port=port), self.assertRaisesRegex(PublicIOError, "port"):
                create_server(session, port=port)

        invalid_labels = (
            ("", "Non-target"),
            (" Target", "Non-target"),
            ("Target", "Target"),
            ("Target\nInjected", "Non-target"),
        )
        for target_label, non_target_label in invalid_labels:
            with self.subTest(labels=(target_label, non_target_label)), self.assertRaises(PublicIOError):
                create_server(
                    session,
                    target_label=target_label,
                    non_target_label=non_target_label,
                )

    def test_local_page_opener_is_an_injected_nonshell_boundary(self) -> None:
        class RecordingOpener:
            def __init__(self, *, fail: bool = False) -> None:
                self.fail = fail
                self.argv: tuple[str, ...] | None = None

            def run(self, argv: object) -> None:
                self.argv = tuple(str(part) for part in argv)
                if self.fail:
                    raise OSError("controlled opener failure")

        url = "http://127.0.0.1:43123/?token=fixture"
        expected = ("/usr/bin/xdg-open", url)
        opener = RecordingOpener()
        with mock.patch(
            "compag_curation.review.web._opener_command",
            return_value=list(expected),
        ):
            self.assertTrue(open_local_review_page(url, opener=opener))
            self.assertEqual(opener.argv, expected)
            self.assertFalse(open_local_review_page(url, opener=RecordingOpener(fail=True)))

    def test_local_page_opener_process_uses_detached_argv_without_a_shell(self) -> None:
        argv = ("/usr/bin/xdg-open", "http://127.0.0.1:43123/?token=fixture")
        with mock.patch("compag_curation.review.web.subprocess.Popen") as popen:
            LocalReviewPageOpener().run(argv)

        popen.assert_called_once_with(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )

    def test_web_decodes_full_image_before_atomic_proposal_commit_and_fails_closed(self) -> None:
        decode = _HTML.index("nextUrl=await decodedScene(scene)")
        commit = _HTML.index("current=proposal")
        self.assertLess(decode, commit)
        self.assertIn("catch(error){if(nextUrl)URL.revokeObjectURL(nextUrl);clearReview();throw error}", _HTML)
        self.assertIn("current=null;currentScene=null;enableReviewActions(false)", _HTML)
        self.assertIn("probe.naturalWidth!==scene.width||probe.naturalHeight!==scene.height", _HTML)
        self.assertIn('id="undo" title="Undo last decision (0 or Z)" disabled', _HTML)
        self.assertEqual(_HTML.count("data-choice=\""), 5)
        for choice in EXPECTED_DECISIONS:
            self.assertIn(f'data-choice="{choice}" disabled', _HTML)
        self.assertIn("if(busy||!current)return", _HTML)
        self.assertIn(".stage{position:absolute;left:0;top:0", _HTML)
        self.assertNotIn(".stage{width:512px;height:512px", _HTML)
        self.assertIn('id="maskOutline" checked', _HTML)
        self.assertIn('id="bboxToggle">', _HTML)
        self.assertNotIn('id="bboxToggle" checked', _HTML)
        self.assertIn('id="followSelected" checked', _HTML)
        self.assertIn(
            "if(sceneChanged)resetView();else followSelected(scene);enableReviewActions",
            _HTML,
        )
        self.assertIn("if(proposal.reviewable){hit=svgShape", _HTML)
        self.assertIn("if(!proposal.reviewable&&!showContext)continue", _HTML)
        self.assertIn("event.repeat||event.ctrlKey||event.metaKey||event.altKey", _HTML)
        self.assertIn("label 0, action skip, and review weight 0.0", _HTML)
        self.assertIn(
            "exactFlag(proposal.prediction,presentation.target_label,presentation.non_target_label)",
            _HTML,
        )
        self.assertIn("applyPresentation(await api('/api/presentation'))", _HTML)
        self.assertIn(
            'id="confirmModelRemainder" hidden disabled',
            _HTML,
        )
        self.assertIn("STAGE20_PUBLISHED_MODEL_ASSIST", _HTML)
        self.assertIn("Confirm exactly ${remaining.toLocaleString()}", _HTML)
        self.assertIn("review weight 0.4", _HTML)
        self.assertIn("/api/confirm-model-remainder", _HTML)
        self.assertIn("scene.model?.mode==='PUBLISHED_TRANSFER_ASSIST'", _HTML)

    def test_bulk_confirm_route_requires_capability_origin_exact_body_and_is_idempotent(self) -> None:
        session = _FakeWebSession()
        server, capability_url = create_server(session)
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
        )
        thread.start()
        try:
            token = parse_qs(urlsplit(capability_url).query)["token"][0]
            token_header = {"X-COMPAG-Review-Token": token}
            origin = f"http://127.0.0.1:{server.server_port}"
            body = json.dumps({"request_id": "bulk-confirm-idempotent"}).encode(
                "ascii"
            )

            status, _headers, _body = self._request(
                server.server_port,
                "POST",
                "/api/confirm-model-remainder",
                headers={**token_header, "Content-Type": "application/json"},
                body=body,
            )
            self.assertEqual(status, 403)
            self.assertEqual(session.bulk_commit_count, 0)

            status, _headers, _body = self._request(
                server.server_port,
                "POST",
                "/api/confirm-model-remainder",
                headers={
                    **token_header,
                    "Origin": origin,
                    "Content-Type": "application/json",
                },
                body=json.dumps(
                    {"request_id": "bulk-confirm-idempotent", "extra": True}
                ).encode("ascii"),
            )
            self.assertEqual(status, 400)
            self.assertEqual(session.bulk_commit_count, 0)

            for _attempt in range(2):
                status, _headers, response = self._request(
                    server.server_port,
                    "POST",
                    "/api/confirm-model-remainder",
                    headers={
                        **token_header,
                        "Origin": origin,
                        "Content-Type": "application/json",
                    },
                    body=body,
                )
                self.assertEqual(status, 200)
                self.assertEqual(
                    json.loads(response)["proposal"]["decision"]["choice"],
                    "target_uncertain",
                )
            self.assertEqual(session.bulk_commit_count, 1)
            self.assertEqual(
                session.calls.count(
                    ("confirm_model_remainder", "bulk-confirm-idempotent")
                ),
                2,
            )
        finally:
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()
        self.assertFalse(thread.is_alive())

    def test_http_capability_origin_csp_and_route_denial(self) -> None:
        session = _FakeWebSession()
        server, capability_url = create_server(
            session,
            target_label="CJ",
            non_target_label="Non-CJ",
        )
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            parsed = urlsplit(capability_url)
            token = parse_qs(parsed.query)["token"][0]
            origin = f"http://127.0.0.1:{server.server_port}"
            token_header = {"X-COMPAG-Review-Token": token}

            status, headers, body = self._request(
                server.server_port, "GET", f"/?token={token}"
            )
            self.assertEqual(status, 200)
            self.assertEqual(headers["cache-control"], "no-store, max-age=0")
            self.assertEqual(headers["x-frame-options"], "DENY")
            self.assertEqual(headers["x-content-type-options"], "nosniff")
            self.assertIn("default-src 'none'", headers["content-security-policy"])
            self.assertIn(
                f"script-src 'nonce-{server.page_nonce}'",
                headers["content-security-policy"],
            )
            self.assertNotIn("unsafe-eval", headers["content-security-policy"])
            self.assertNotIn(token.encode("ascii"), body)
            self.assertIn(f'nonce="{server.page_nonce}"'.encode("ascii"), body)

            status, _headers, body = self._request(
                server.server_port,
                "GET",
                "/api/presentation",
                headers=token_header,
            )
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["target_label"], "CJ")
            self.assertEqual(json.loads(body)["non_target_label"], "Non-CJ")

            status, _headers, body = self._request(
                server.server_port,
                "GET",
                f"/api/proposal?id={session.proposal_id}",
                headers=token_header,
            )
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["scene"]["scene_id"], session.scene_id)
            self.assertIn(("scene_payload", session.proposal_id), session.calls)

            status, headers, body = self._request(
                server.server_port,
                "GET",
                f"/api/scene?scene={session.scene_id}",
                headers=token_header,
            )
            self.assertEqual(status, 200)
            self.assertEqual(headers["content-type"], "image/png")
            self.assertTrue(body.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertIn(("scene_bytes", session.scene_id), session.calls)

            status, _headers, _body = self._request(
                server.server_port, "GET", "/api/status"
            )
            self.assertEqual(status, 403)
            status, _headers, _body = self._request(
                server.server_port,
                "GET",
                "/api/status",
                headers={"X-COMPAG-Review-Token": "wrong"},
            )
            self.assertEqual(status, 403)
            status, _headers, _body = self._request(
                server.server_port,
                "GET",
                "/api/status",
                headers={**token_header, "Host": f"localhost:{server.server_port}"},
            )
            self.assertEqual(status, 403)

            decision = json.dumps(
                {
                    "proposal_id": session.proposal_id,
                    "choice": "target",
                    "request_id": "unit-test",
                }
            ).encode("ascii")
            status, _headers, _body = self._request(
                server.server_port,
                "POST",
                "/api/decision",
                headers={**token_header, "Content-Type": "application/json"},
                body=decision,
            )
            self.assertEqual(status, 403)
            self.assertFalse(any(call[:3] == ("decision", session.proposal_id, "target") for call in session.calls))

            status, _headers, body = self._request(
                server.server_port,
                "POST",
                "/api/decision",
                headers={
                    **token_header,
                    "Origin": origin,
                    "Content-Type": "application/json",
                },
                body=decision,
            )
            self.assertEqual(status, 200)
            self.assertEqual(
                json.loads(body)["proposal"]["decision"]["choice"],
                "target",
            )
            self.assertIn(
                ("decision", session.proposal_id, "target", "unit-test"),
                session.calls,
            )

            for method, path in (
                ("GET", "/../../etc/passwd"),
                ("GET", "/api/overlay?tile=0&extra=1"),
                ("GET", f"/api/scene?scene={session.scene_id}&extra=1"),
                ("POST", "/api/not-a-route"),
            ):
                with self.subTest(method=method, path=path):
                    request_body = b"{}" if method == "POST" else None
                    route_headers = dict(token_header)
                    if method == "POST":
                        route_headers.update(
                            {"Origin": origin, "Content-Type": "application/json"}
                        )
                    status, _headers, _body = self._request(
                        server.server_port,
                        method,
                        path,
                        headers=route_headers,
                        body=request_body,
                    )
                    self.assertEqual(status, 404)

            status, _headers, _body = self._request(
                server.server_port,
                "PUT",
                "/api/decision",
                headers={**token_header, "Origin": origin},
            )
            self.assertEqual(status, 405)
        finally:
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()
        self.assertFalse(thread.is_alive())

    def test_incomplete_local_connection_cannot_block_authorized_requests(self) -> None:
        session = _FakeWebSession()
        server, capability_url = create_server(session)
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
        )
        thread.start()
        raw = socket.create_connection(("127.0.0.1", server.server_port), timeout=2)
        try:
            raw.sendall(
                f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n".encode(
                    "ascii"
                )
            )
            token = parse_qs(urlsplit(capability_url).query)["token"][0]
            status, _headers, body = self._request(
                server.server_port,
                "GET",
                "/api/status",
                headers={"X-COMPAG-Review-Token": token},
            )
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["status"], "IN_PROGRESS")
        finally:
            raw.close()
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()
        self.assertFalse(thread.is_alive())

    def test_incomplete_connections_never_exceed_the_hard_handler_cap(self) -> None:
        session = _FakeWebSession()
        server, capability_url = create_server(session)
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
        )
        thread.start()
        sockets: list[socket.socket] = []
        try:
            for _index in range(server.MAX_CONCURRENT_HANDLERS + 24):
                try:
                    raw = socket.create_connection(
                        ("127.0.0.1", server.server_port), timeout=0.5
                    )
                    raw.sendall(
                        (
                            "GET / HTTP/1.1\r\n"
                            f"Host: 127.0.0.1:{server.server_port}\r\n"
                        ).encode("ascii")
                    )
                    sockets.append(raw)
                except OSError:
                    continue
            threading.Event().wait(0.1)
            self.assertLessEqual(
                server.peak_concurrent_handlers,
                server.MAX_CONCURRENT_HANDLERS,
            )
            self.assertGreater(server.peak_concurrent_handlers, 0)
        finally:
            for raw in sockets:
                raw.close()

        for _index in range(100):
            if server.active_concurrent_handlers == 0:
                break
            threading.Event().wait(0.01)
        try:
            token = parse_qs(urlsplit(capability_url).query)["token"][0]
            status, _headers, body = self._request(
                server.server_port,
                "GET",
                "/api/status",
                headers={"X-COMPAG-Review-Token": token},
            )
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["status"], "IN_PROGRESS")
        finally:
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()
        self.assertFalse(thread.is_alive())


class _FakeDesktopSession:
    def __init__(self) -> None:
        self.ids = ("1" * 64, "2" * 64, "3" * 64)
        self.index = SimpleNamespace(order=self.ids)
        self.calls: list[tuple[object, ...]] = []

    def summary(self) -> dict[str, object]:
        return {"first_pending": self.ids[0], "remaining": 3}

    def proposal_payload(self, proposal_id: str) -> dict[str, object]:
        self.calls.append(("payload", proposal_id))
        return {"proposal": {"proposal_id": proposal_id}, "summary": {"remaining": 3}}

    def scene_payload(self, proposal_id: str) -> dict[str, object]:
        return {
            "proposal": {"proposal_id": proposal_id},
            "summary": {"remaining": 3},
            "scene": {"scene_id": f"scene-{proposal_id}"},
        }

    def set_decision(self, proposal_id: str, choice: str) -> dict[str, object]:
        self.calls.append(("decision", proposal_id, choice))
        return {"proposal": {"proposal_id": proposal_id}, "summary": {"remaining": 0}}

    def navigate(
        self,
        current: str,
        direction: int,
        *,
        state_filter: str,
        group: str | None,
    ) -> dict[str, object]:
        self.calls.append(("navigate", current, direction, state_filter, group))
        selected = self.ids[(self.ids.index(current) + direction) % len(self.ids)]
        return {"proposal": {"proposal_id": selected}, "summary": {"remaining": 3}}

    def undo(self) -> dict[str, object]:
        self.calls.append(("undo",))
        return {"proposal": {"proposal_id": self.ids[1]}, "summary": {"remaining": 3}}

    def confirm_model_remainder(self) -> dict[str, object]:
        self.calls.append(("confirm_model_remainder",))
        return {"proposal": {"proposal_id": self.ids[0]}, "summary": {"remaining": 0}}


class _HeadlessTkVariable:
    def __init__(self, value: object = None) -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class _HeadlessTkWidget:
    def __init__(self, _parent: object = None, **options: object) -> None:
        self.options = dict(options)

    def bind(self, *_args: object, **_kwargs: object) -> None:
        pass

    def columnconfigure(self, *_args: object, **_kwargs: object) -> None:
        pass

    def configure(self, *_args: object, **options: object) -> None:
        self.options.update(options)

    def grid(self, *_args: object, **_kwargs: object) -> None:
        pass

    def pack(self, *_args: object, **_kwargs: object) -> None:
        self.packed = True

    def place(self, *_args: object, **_kwargs: object) -> None:
        pass

    def rowconfigure(self, *_args: object, **_kwargs: object) -> None:
        pass


class _HeadlessTkRoot(_HeadlessTkWidget):
    def title(self, _value: str) -> None:
        pass

    def geometry(self, _value: str) -> None:
        pass

    def minsize(self, _width: int, _height: int) -> None:
        pass

    def protocol(self, *_args: object) -> None:
        pass

    def destroy(self) -> None:
        pass


class _HeadlessTkStyle(_HeadlessTkWidget):
    def theme_use(self, _value: str) -> None:
        pass


class DesktopControllerTests(unittest.TestCase):
    @staticmethod
    def _reviewer_payload(proposal_id: str) -> dict[str, object]:
        return {
            "proposal": {
                "proposal_id": proposal_id,
                "label_prefix": proposal_id[:8],
                "image_name": "fixture.jpg",
                "position": 1,
                "total": 2,
                "group_id": "group-1",
                "predicted_iou": 0.9,
                "stability_score": 0.8,
                "decision": None,
            },
            "tile": {"index": 1, "tile_name": "fixture.png", "proposals": []},
            "scene": {
                "schema": "compag-curation-full-image-review-scene/v1",
                "scene_id": f"scene-{proposal_id}",
                "index": 0,
                "kind": "SEALED_STAGE10_FULL_IMAGE_MOSAIC",
                "image_id": "a" * 64,
                "image_name": "fixture.jpg",
                "group_id": "group-1",
                "width": 640,
                "height": 480,
                "mime_type": "image/png",
                "selected_proposal_id": proposal_id,
                "proposal_count": 1,
                "proposals": [
                    {
                        "proposal_id": proposal_id,
                        "label_prefix": proposal_id[:8],
                        "polygon": [10, 10, 30, 10, 30, 30, 10, 30],
                        "bbox": [10, 10, 20, 20],
                        "choice": None,
                        "reviewable": True,
                        "selected": True,
                    }
                ],
                "overlay_defaults": {"mask_outline": True, "bbox": False},
                "outline_source": "SEALED_MASK_DERIVED_POLYGON",
            },
            "summary": {
                "reviewed": 0,
                "remaining": 2,
                "progress_percent": 0.0,
                "status": "IN_PROGRESS",
            },
        }

    @staticmethod
    def _bare_reviewer(session: object, old_id: str) -> _TkReviewer:
        reviewer = object.__new__(_TkReviewer)
        reviewer.session = session
        reviewer.controller = SimpleNamespace(current_id="2" * 64)
        reviewer.payload = {"proposal": {"proposal_id": old_id}}
        reviewer.scene = {"scene_id": f"scene-{old_id}"}
        reviewer.source_image = SimpleNamespace(size=(640, 480))
        reviewer.photo = object()
        reviewer.photo_origin = (0.0, 0.0)
        reviewer.review_ready = True
        reviewer.drag_start = None
        reviewer.canvas = mock.Mock()
        reviewer.id_label = mock.Mock()
        reviewer.context_label = mock.Mock()
        reviewer.tile_label = mock.Mock()
        reviewer.facts_label = mock.Mock()
        reviewer.decision_label = mock.Mock()
        reviewer.progress_text = mock.Mock()
        reviewer.progress = mock.Mock()
        reviewer.finalize_button = mock.Mock()
        reviewer.undo_button = mock.Mock()
        reviewer.bulk_confirm_button = mock.Mock()
        reviewer.decision_buttons = [mock.Mock(), mock.Mock()]
        reviewer.assisted_mode = False
        reviewer.target_label = "Target"
        reviewer.non_target_label = "Non-target"
        reviewer.choice_labels = dict(_CHOICE_LABELS)
        reviewer.outline_var = SimpleNamespace(get=lambda: True)
        reviewer.bbox_var = SimpleNamespace(get=lambda: False)
        reviewer.context_var = SimpleNamespace(get=lambda: False, set=mock.Mock())
        reviewer.context_button = mock.Mock()
        reviewer.follow_var = SimpleNamespace(get=lambda: True)
        reviewer.scale = 1.0
        reviewer.offset_x = 0.0
        reviewer.offset_y = 0.0
        reviewer.redraw = mock.Mock()  # type: ignore[method-assign]
        reviewer.fit = mock.Mock()  # type: ignore[method-assign]
        return reviewer

    def test_native_constructor_builds_custom_decision_buttons_and_actions(self) -> None:
        proposal_id = "1" * 64
        session = SimpleNamespace(
            summary=lambda: {"first_pending": proposal_id},
            index=SimpleNamespace(order=(proposal_id,), groups=("group-1",)),
            scene_payload=lambda selected: {"proposal": {"proposal_id": selected}},
        )
        tk = SimpleNamespace(
            TclError=RuntimeError,
            StringVar=_HeadlessTkVariable,
            BooleanVar=_HeadlessTkVariable,
            Canvas=_HeadlessTkWidget,
        )
        ttk = SimpleNamespace(
            Style=_HeadlessTkStyle,
            Frame=_HeadlessTkWidget,
            Label=_HeadlessTkWidget,
            Progressbar=_HeadlessTkWidget,
            Button=_HeadlessTkWidget,
            Combobox=_HeadlessTkWidget,
            Checkbutton=_HeadlessTkWidget,
            Entry=_HeadlessTkWidget,
        )

        with mock.patch.object(_TkReviewer, "show", autospec=True) as show:
            reviewer = _TkReviewer(
                _HeadlessTkRoot(),
                session,
                tk,
                ttk,
                image_module=object(),
                image_tk=object(),
                target_label="CJ",
                non_target_label="Non-CJ",
            )

        show.assert_called_once()
        self.assertEqual(
            [button.options["text"] for button in reviewer.decision_buttons],
            [
                "CJ · confident    [1 / A]",
                "Non-CJ · confident    [2 / R]",
                "CJ · uncertain    [4 / W]",
                "Non-CJ · uncertain    [5 / U]",
                "Skip · zero weight    [3 / S]",
            ],
        )

        reviewer.decide = mock.Mock()  # type: ignore[method-assign]
        for button in reviewer.decision_buttons:
            command = button.options["command"]
            self.assertTrue(callable(command))
            command()
        self.assertEqual(
            reviewer.decide.call_args_list,
            [
                mock.call("target"),
                mock.call("other"),
                mock.call("target_uncertain"),
                mock.call("other_uncertain"),
                mock.call("skip"),
            ],
        )
        self.assertFalse(reviewer.assisted_mode)
        self.assertFalse(hasattr(reviewer.bulk_confirm_button, "packed"))

    def test_native_assisted_mode_displays_model_metrics_and_only_then_packs_bulk_action(self) -> None:
        proposal_id = "1" * 64
        session = SimpleNamespace(
            summary=lambda: {
                "first_pending": proposal_id,
                "source_kind": "STAGE20_PUBLISHED_MODEL_ASSIST",
            },
            index=SimpleNamespace(order=(proposal_id,), groups=("group-1",)),
            scene_payload=lambda selected: {"proposal": {"proposal_id": selected}},
        )
        tk = SimpleNamespace(
            TclError=RuntimeError,
            StringVar=_HeadlessTkVariable,
            BooleanVar=_HeadlessTkVariable,
            Canvas=_HeadlessTkWidget,
        )
        ttk = SimpleNamespace(
            Style=_HeadlessTkStyle,
            Frame=_HeadlessTkWidget,
            Label=_HeadlessTkWidget,
            Progressbar=_HeadlessTkWidget,
            Button=_HeadlessTkWidget,
            Combobox=_HeadlessTkWidget,
            Checkbutton=_HeadlessTkWidget,
            Entry=_HeadlessTkWidget,
        )
        with mock.patch.object(_TkReviewer, "show", autospec=True):
            reviewer = _TkReviewer(
                _HeadlessTkRoot(),
                session,
                tk,
                ttk,
                image_module=object(),
                image_tk=object(),
                target_label="CJ",
                non_target_label="Non-CJ",
            )
        self.assertTrue(reviewer.assisted_mode)
        self.assertTrue(reviewer.bulk_confirm_button.packed)
        self.assertEqual(
            reviewer.bulk_confirm_button.options["text"],
            "Confirm remaining XGB suggestions · weight 0.4",
        )

        payload = self._reviewer_payload(proposal_id)
        payload["proposal"].update(
            {"xgb_p": 0.81, "prediction": 1, "uncertainty": 0.19, "selection_rank": 7}
        )
        payload["summary"].update(
            {
                "source_kind": "STAGE20_PUBLISHED_MODEL_ASSIST",
                "can_confirm_model_remainder": True,
            }
        )
        payload["scene"]["model"] = {
            "mode": "PUBLISHED_TRANSFER_ASSIST",
            "model_suggestions_are_human_context": True,
        }
        payload["scene"]["proposals"][0]["model"] = {
            "preset_id": "compag-cj-r92",
            "xgb_p": 0.81,
            "prediction": 1,
            "uncertainty": 0.19,
            "threshold": 0.5,
        }
        display = reviewer._stage_display(payload)
        self.assertIn("Published XGBoost r92 transfer assist", display["context"])
        self.assertIn("model CJ", display["tile_text"])
        self.assertIn("XGB probability: 0.8100", display["facts"])
        self.assertIn("Uncertainty: 0.1900", display["facts"])
        self.assertTrue(display["can_confirm_model_remainder"])
        self.assertEqual(
            reviewer._overlay_color(payload["scene"]["proposals"][0]),
            "#ffd166",
        )
        model_row = dict(payload["scene"]["proposals"][0])
        model_row["selected"] = False
        self.assertEqual(reviewer._overlay_color(model_row), "#55dfad")
        model_row["model"] = {**model_row["model"], "prediction": 0}
        self.assertEqual(reviewer._overlay_color(model_row), "#ff8b98")

    def test_desktop_controller_bulk_confirmation_uses_session_and_verified_scene(self) -> None:
        session = _FakeDesktopSession()
        controller = DesktopReviewController(session)

        payload = controller.confirm_model_remainder()

        self.assertEqual(controller.current_id, session.ids[0])
        self.assertEqual(payload["scene"]["scene_id"], f"scene-{session.ids[0]}")
        self.assertEqual(session.calls, [("confirm_model_remainder",)])

    def test_native_overlay_failure_clears_stale_image_and_disables_mutation(self) -> None:
        old_id = "1" * 64
        new_id = "2" * 64
        session = SimpleNamespace(
            scene_bytes=mock.Mock(side_effect=PublicIOError("tampered scene"))
        )
        reviewer = self._bare_reviewer(session, old_id)
        reviewer.Image = mock.Mock()

        with self.assertRaisesRegex(PublicIOError, "tampered"):
            reviewer.show(self._reviewer_payload(new_id))

        self.assertIsNone(reviewer.payload)
        self.assertIsNone(reviewer.source_image)
        self.assertIsNone(reviewer.photo)
        self.assertFalse(reviewer.review_ready)
        self.assertEqual(reviewer.controller.current_id, old_id)
        reviewer.canvas.delete.assert_called_with("all")
        for button in reviewer.decision_buttons:
            button.configure.assert_called_with(state="disabled")
        reviewer.undo_button.configure.assert_called_with(state="disabled")
        reviewer.finalize_button.configure.assert_called_with(state="disabled")

    def test_native_new_overlay_invalidates_same_scale_photo_cache_before_commit(self) -> None:
        old_id = "1" * 64
        new_id = "2" * 64
        converted = object()
        decoded = mock.Mock(size=(640, 480))
        decoded.convert.return_value = converted
        reviewer = self._bare_reviewer(
            SimpleNamespace(scene_bytes=mock.Mock(return_value=b"valid-png")),
            old_id,
        )
        reviewer.Image = SimpleNamespace(open=mock.Mock(return_value=decoded))
        payload = self._reviewer_payload(new_id)

        reviewer.show(payload)

        decoded.load.assert_called_once_with()
        decoded.convert.assert_called_once_with("RGB")
        self.assertIs(reviewer.payload, payload)
        self.assertIs(reviewer.source_image, converted)
        self.assertIsNone(reviewer.photo)
        self.assertIsNone(reviewer.photo_origin)
        self.assertTrue(reviewer.review_ready)
        self.assertEqual(reviewer.controller.current_id, new_id)
        reviewer.fit.assert_called_once_with()

    def test_native_shortcuts_ignore_editable_widgets_and_use_explicit_key_events(self) -> None:
        reviewer = object.__new__(_TkReviewer)
        reviewer.decide = mock.Mock()  # type: ignore[method-assign]
        reviewer.navigate = mock.Mock()  # type: ignore[method-assign]
        reviewer.undo = mock.Mock()  # type: ignore[method-assign]
        editing_event = SimpleNamespace(
            widget=SimpleNamespace(winfo_class=lambda: "TEntry")
        )
        canvas_event = SimpleNamespace(
            widget=SimpleNamespace(winfo_class=lambda: "Canvas")
        )

        self.assertIsNone(reviewer._decision_shortcut(editing_event, "target"))
        self.assertIsNone(reviewer._navigation_shortcut(editing_event, 1))
        self.assertIsNone(reviewer._undo_shortcut(editing_event))
        reviewer.decide.assert_not_called()
        reviewer.navigate.assert_not_called()
        reviewer.undo.assert_not_called()

        self.assertEqual(reviewer._decision_shortcut(canvas_event, "target"), "break")
        self.assertEqual(reviewer._navigation_shortcut(canvas_event, -1), "break")
        self.assertEqual(reviewer._undo_shortcut(canvas_event), "break")
        reviewer.decide.assert_called_once_with("target")
        reviewer.navigate.assert_called_once_with(-1)
        reviewer.undo.assert_called_once_with()

    def test_native_shortcuts_reject_modifiers_and_auto_repeat(self) -> None:
        reviewer = object.__new__(_TkReviewer)
        reviewer._shortcut_keys_down = set()
        reviewer.decide = mock.Mock()  # type: ignore[method-assign]
        reviewer.navigate = mock.Mock()  # type: ignore[method-assign]
        reviewer.undo = mock.Mock()  # type: ignore[method-assign]
        widget = SimpleNamespace(winfo_class=lambda: "Canvas")

        modified = SimpleNamespace(widget=widget, state=0x04, keysym="r")
        self.assertEqual(reviewer._decision_shortcut(modified, "other"), "break")
        reviewer.decide.assert_not_called()

        first = SimpleNamespace(widget=widget, state=0, keysym="a")
        repeated = SimpleNamespace(widget=widget, state=0, keysym="a")
        self.assertEqual(reviewer._decision_shortcut(first, "target"), "break")
        self.assertEqual(reviewer._decision_shortcut(repeated, "target"), "break")
        reviewer.decide.assert_called_once_with("target")
        reviewer._shortcut_released(first)
        self.assertEqual(reviewer._decision_shortcut(first, "target"), "break")
        self.assertEqual(reviewer.decide.call_count, 2)

    def test_native_round_keeps_model_context_optional_and_off_by_default(self) -> None:
        old_id = "1" * 64
        new_id = "2" * 64
        decoded = mock.Mock(size=(640, 480))
        decoded.convert.return_value = object()
        reviewer = self._bare_reviewer(
            SimpleNamespace(scene_bytes=mock.Mock(return_value=b"valid-png")),
            old_id,
        )
        reviewer.Image = SimpleNamespace(open=mock.Mock(return_value=decoded))
        payload = self._reviewer_payload(new_id)
        payload["summary"]["source_kind"] = "ACTIVE_LEARNING_ROUND"
        payload["summary"]["round_number"] = 3
        payload["scene"]["kind"] = "VERIFIED_STAGE60_ORIGINAL_FULL_IMAGE"

        reviewer.show(payload)

        self.assertTrue(reviewer.review_ready)
        tile_text = reviewer.tile_label.configure.call_args.kwargs["text"]
        facts = reviewer.facts_label.configure.call_args.kwargs["text"]
        self.assertIn("rank Unavailable", tile_text)
        self.assertIn("model unavailable", tile_text)
        self.assertIn("XGB probability: Unavailable", facts)
        self.assertIn("Uncertainty: Unavailable", facts)
        reviewer.context_button.configure.assert_called_with(state="normal")
        reviewer.context_var.set.assert_not_called()

    def test_native_round_hides_nonreviewable_model_context_by_default(self) -> None:
        reviewer = object.__new__(_TkReviewer)
        reviewer.scene = {
            "proposals": [
                {"proposal_id": "hard-case", "reviewable": True},
                {"proposal_id": "context-only", "reviewable": False},
            ]
        }
        reviewer.context_var = SimpleNamespace(get=lambda: False)
        self.assertEqual(
            [row["proposal_id"] for row in reviewer._display_scene_rows()],
            ["hard-case"],
        )
        reviewer.context_var = SimpleNamespace(get=lambda: True)
        self.assertEqual(
            [row["proposal_id"] for row in reviewer._display_scene_rows()],
            ["hard-case", "context-only"],
        )

    def test_native_follow_selected_pans_minimally_without_changing_zoom(self) -> None:
        reviewer = object.__new__(_TkReviewer)
        reviewer.source_image = SimpleNamespace(size=(1000, 800))
        reviewer.scene = {
            "proposals": [
                {
                    "proposal_id": "selected",
                    "bbox": [900, 700, 50, 50],
                    "selected": True,
                }
            ]
        }
        reviewer.canvas = SimpleNamespace(
            winfo_width=lambda: 300,
            winfo_height=lambda: 200,
        )
        reviewer.scale = 1.0
        reviewer.offset_x = 0.0
        reviewer.offset_y = 0.0

        reviewer._ensure_selected_visible()

        self.assertEqual(reviewer.scale, 1.0)
        self.assertEqual(reviewer.offset_x, -674.0)
        self.assertEqual(reviewer.offset_y, -574.0)

    def test_native_render_failure_clears_new_identity_and_stale_pixels(self) -> None:
        old_id = "1" * 64
        new_id = "2" * 64
        decoded = mock.Mock(size=(640, 480))
        decoded.convert.return_value = object()
        reviewer = self._bare_reviewer(
            SimpleNamespace(scene_bytes=mock.Mock(return_value=b"valid-png")),
            old_id,
        )
        reviewer.Image = SimpleNamespace(open=mock.Mock(return_value=decoded))
        reviewer.fit.side_effect = RuntimeError("injected render failure")

        with self.assertRaisesRegex(PublicIOError, "failed safely"):
            reviewer.show(self._reviewer_payload(new_id))

        self.assertFalse(reviewer.review_ready)
        self.assertIsNone(reviewer.payload)
        self.assertIsNone(reviewer.source_image)
        self.assertEqual(reviewer.controller.current_id, old_id)
        reviewer.canvas.delete.assert_called_with("all")

    def test_controller_forwards_all_decisions_and_navigation_without_gui(self) -> None:
        session = _FakeDesktopSession()
        controller = DesktopReviewController(session)
        controller.auto_next = False
        self.assertEqual(controller.current_id, session.ids[0])
        self.assertEqual(controller.current()["proposal"]["proposal_id"], session.ids[0])

        for choice in EXPECTED_DECISIONS:
            with self.subTest(choice=choice):
                result = controller.decide(choice)
                self.assertEqual(result["proposal"]["proposal_id"], session.ids[0])
                self.assertEqual(session.calls[-1], ("decision", session.ids[0], choice))

        controller.state_filter = "reviewed"
        controller.group = "group-1"
        moved = controller.navigate(1)
        self.assertEqual(moved["proposal"]["proposal_id"], session.ids[1])
        self.assertEqual(controller.current_id, session.ids[1])
        self.assertEqual(
            session.calls[-1],
            ("navigate", session.ids[0], 1, "reviewed", "group-1"),
        )
        undone = controller.undo()
        self.assertEqual(undone["proposal"]["proposal_id"], session.ids[1])
        self.assertEqual(controller.current_id, session.ids[1])

    def test_controller_auto_next_uses_pending_filter(self) -> None:
        session = _FakeDesktopSession()
        controller = DesktopReviewController(session)

        def saved_with_remaining(proposal_id: str, choice: str) -> dict[str, object]:
            session.calls.append(("decision", proposal_id, choice))
            return {"proposal": {"proposal_id": proposal_id}, "summary": {"remaining": 2}}

        session.set_decision = saved_with_remaining  # type: ignore[method-assign]
        result = controller.decide("skip")
        self.assertEqual(result["proposal"]["proposal_id"], session.ids[1])
        self.assertEqual(controller.current_id, session.ids[1])
        self.assertEqual(
            session.calls[-1],
            ("navigate", session.ids[0], 1, "pending", None),
        )


if __name__ == "__main__":
    unittest.main()
