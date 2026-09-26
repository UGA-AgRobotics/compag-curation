"""Regression for nonempty masks whose display contour has zero area."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path


class R92MaskGeometryTests(unittest.TestCase):
    def test_valid_contour_matches_existing_approximation(self):
        import cv2
        import numpy as np
        from compag_curation.r92_live import _mask_polygon

        mask = np.zeros((512, 512), np.uint8)
        mask[10:30, 20:45] = 1
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        expected = [int(v) for v in cv2.approxPolyDP(
            max(contours, key=cv2.contourArea), 1.0, True).reshape(-1)]
        self.assertEqual(_mask_polygon(mask), expected)

    def test_degenerate_contour_uses_existing_review_bbox_path(self):
        import cv2
        import numpy as np
        from compag_curation.r92_live import _mask_polygon
        from compag_curation.r92_review import _preflight_rows

        mask = np.zeros((512, 512), np.uint8)
        mask[20, 30:34] = 1
        self.assertIsNone(_mask_polygon(mask))
        self.assertEqual(int(mask.sum()), 4)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tiles = root / "tiles"
            tiles.mkdir()
            tile_name = "IMG_9317_y00000x00000.jpg"
            self.assertTrue(cv2.imwrite(str(tiles / tile_name),
                                       np.zeros((512, 512, 3), np.uint8)))
            scored = root / "detections.csv"
            with scored.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["img_folder", "image", "id",
                    "xgb_p", "poly", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
                writer.writeheader()
                writer.writerow({"img_folder": "IMG_9317.jpg", "image": tile_name,
                    "id": "1", "xgb_p": ".72", "poly": "", "bbox_x": "30",
                    "bbox_y": "20", "bbox_w": "4", "bbox_h": "1"})
            rows = _preflight_rows(scored, tiles)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["xgb_p"], ".72")
            self.assertEqual(rows[0]["poly"], "")

    def test_empty_or_wrong_shape_mask_still_fails(self):
        import numpy as np
        from compag_curation.r92_live import _mask_polygon

        with self.assertRaisesRegex(ValueError, "empty"):
            _mask_polygon(np.zeros((512, 512), np.uint8))
        with self.assertRaisesRegex(ValueError, "512x512"):
            _mask_polygon(np.ones((8, 8), np.uint8))


if __name__ == "__main__":
    unittest.main()
