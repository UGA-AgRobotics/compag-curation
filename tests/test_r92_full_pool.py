"""Project reviewer regression with 120 synthetic scored candidates."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class FullPoolReview(unittest.TestCase):
    def test_all_candidates_shortlist_filters_and_51_decisions(self):
        import cv2
        import numpy as np
        from compag_curation.r92_review import effective_events, export, make_session
        from compag_curation.r92_project_model import review_plan
        from compag_curation.only_codes.review import key_str

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tiles = root / "tiles"
            tiles.mkdir()
            names = [f"CARD_{card}_y00000x{offset:05d}.jpg" for card in ("A", "B") for offset in (0, 512)]
            for name in names:
                self.assertTrue(cv2.imwrite(str(tiles / name), np.zeros((512, 512, 3), np.uint8)))
            scores = root / "scores"
            scores.mkdir()
            scored = scores / "detections.csv"
            fields = ["img_folder", "image", "id", "xgb_p", "xgb_pred", "threshold",
                      "poly", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
            rows = []
            for i in range(120):
                p = (.45 if i % 2 == 0 else .55) if i < 50 else (.48 if i >= 100 else .05 if i % 2 == 0 else .95)
                image = names[i // 30]
                rows.append({"img_folder": image.split("_y")[0] + ".jpg", "image": image,
                             "id": str(i % 30 + 1), "xgb_p": str(p), "xgb_pred": str(int(p >= .5)),
                             "threshold": ".5", "poly": "", "bbox_x": str((i % 20) * 22),
                             "bbox_y": str((i % 10) * 22), "bbox_w": "14", "bbox_h": "14"})
            with scored.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            digest = hashlib.sha256(scored.read_bytes()).hexdigest()
            (scores / "SCORE_RECEIPT.json").write_text(json.dumps({
                "detections_sha256": digest, "execution_variant": "cpu_tabular_xgboost_2.1.1",
                "threshold": .5, "model_sha256": "synthetic-test-model"}))
            selection = root / "top50.csv"
            with selection.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["image", "id"])
                writer.writeheader()
                writer.writerows({"image": r["image"], "id": r["id"]} for r in rows[:50])
            state = root / "state"
            sess = make_session(scored, tiles, state, selection=selection)
            self.assertEqual(len(sess.df), 120)
            self.assertEqual(len(sess.project_selection), 50)
            all_items = [item for base in sess.bases for item in sess.items(base)]
            self.assertEqual(len(all_items), 120)
            self.assertEqual(sum(item["shortlisted"] for item in all_items), 50)
            self.assertEqual(sum(item["uncertain"] for item in all_items), 70)
            target = key_str(rows[50]["image"], rows[50]["id"])
            before = next(item for item in all_items if item["key"] == target)
            self.assertFalse(before["shortlisted"])
            self.assertEqual(before["pred"], 0)
            self.assertFalse(before["uncertain"])
            for r in rows[:50]:
                sess.act("accept", key_str(r["image"], r["id"]))
            sess.act("flip", target)
            self.assertEqual(len(effective_events(state / "review_labels.csv")), 51)
            reopened = make_session(scored, tiles, state)
            after = next(item for item in reopened.items("CARD_A") + reopened.items("CARD_B") if item["key"] == target)
            self.assertEqual(after["human"], 1)
            self.assertEqual(after["pred"], 0)
            self.assertFalse(after["uncertain"])
            self.assertEqual(after["review_status"], "confident")
            self.assertFalse(after["shortlisted"])
            reopened.act("sus_accept", target)
            uncertain = next(item for item in reopened.items("CARD_A") + reopened.items("CARD_B") if item["key"] == target)
            self.assertEqual(uncertain["review_status"], "human_uncertain")
            self.assertFalse(uncertain["uncertain"])
            reopened.act("undo")
            restored = next(item for item in reopened.items("CARD_A") + reopened.items("CARD_B") if item["key"] == target)
            self.assertEqual(restored["human"], 1)
            self.assertEqual(restored["action"], "flip")
            self.assertEqual(len(effective_events(state / "review_labels.csv")), 51)
            reopened.act("delete", target)
            deleted = next(item for item in reopened.items("CARD_A") + reopened.items("CARD_B") if item["key"] == target)
            self.assertEqual(deleted["review_status"], "deleted")
            self.assertIsNone(deleted["human"])
            self.assertEqual(deleted["pred"], 0)
            self.assertEqual(review_plan(state)["eligible_labeled_count"], 50)
            self.assertEqual(review_plan(state)["deleted_count"], 1)
            preview = reopened.bulk_plan()
            self.assertEqual(preview["remaining_count"], 69)
            self.assertEqual(preview["model_cj_count"] + preview["model_noncj_count"], 69)
            with self.assertRaisesRegex(ValueError, "preview and confirm again"):
                reopened.act("accept_remaining", expected_count=68, expected_log_sha256=preview["review_log_sha256"])
            with self.assertRaisesRegex(ValueError, "preview and confirm again"):
                reopened.act("accept_remaining", expected_count=69, expected_log_sha256="stale")
            reopened.act("accept_remaining", expected_count=69, expected_log_sha256=preview["review_log_sha256"])
            plan = review_plan(state, scope="all")
            self.assertEqual(plan["terminal_count"], 120)
            self.assertEqual(plan["eligible_labeled_count"], 119)
            self.assertEqual(plan["deleted_count"], 1)
            self.assertEqual(plan["bulk_accepted_count"], 69)
            self.assertNotIn([rows[50]["image"], rows[50]["id"]], plan["included_keys"])
            bulk_item = next(item for item in reopened.items("CARD_B") if item["key"] == key_str(rows[60]["image"], rows[60]["id"]))
            self.assertEqual(bulk_item["review_status"], "bulk_accepted")
            self.assertEqual(bulk_item["human"], bulk_item["pred"])
            reopened.act("undo_bulk")
            self.assertEqual(reopened.bulk_plan()["remaining_count"], 69)
            with self.assertRaisesRegex(ValueError, "already undone"):
                reopened.act("undo_bulk")
            reopened.act("undo")
            restored_again = next(item for item in reopened.items("CARD_A") if item["key"] == target)
            self.assertEqual(restored_again["human"], 1)
            self.assertEqual(restored_again["review_status"], "confident")
            for r in rows[51:]:
                reopened.act("accept", key_str(r["image"], r["id"]))
            reopened.act("delete", target)
            self.assertEqual(review_plan(state, scope="all")["deleted_count"], 1)
            with self.assertRaisesRegex(ValueError, "effective individual"):
                export(state, root / "missing_original_coco.json", root / "missing_tiled_coco.json",
                       root / "strict_export", confirm_review_complete=True)


if __name__ == "__main__":
    unittest.main()
