from __future__ import annotations

import unittest
from types import SimpleNamespace

from compag_curation.canonical.features import (
    CANONICAL_RAW_FEATURE_ORDER,
    CanonicalGridContext,
)
from compag_curation.canonical.full_image_features import (
    FULL_IMAGE_FEATURE_SEMANTICS_ID,
    extract_full_image_raw_features,
)


class _Embedder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed_many(self, patches: object) -> object:
        import numpy as np

        patches = tuple(patches)
        self.batch_sizes.append(len(patches))
        result = np.zeros((len(patches), 2048), dtype=np.float32)
        result[:, 0] = 1.0
        return result


class FullImageFeatureTests(unittest.TestCase):
    def test_roi_features_use_full_frame_geometry_without_full_masks(self) -> None:
        import numpy as np

        image = np.full((200, 300, 3), (0, 220, 255), dtype=np.uint8)
        mask_crop = np.ones((20, 30), dtype=np.uint8)
        proposal = SimpleNamespace(
            bbox=(50, 60, 30, 20),
            mask_crop=mask_crop,
            proposal_index=7,
            predicted_iou=0.91,
            stability_score=0.93,
        )
        embedder = _Embedder()
        rows = extract_full_image_raw_features(
            image,
            (proposal,),
            embedder,
            grid=CanonicalGridContext(
                row_lines=(40, 80, 120, 160),
                column_lines=(75, 150, 225),
            ),
        )
        self.assertEqual(FULL_IMAGE_FEATURE_SEMANTICS_ID, "compag-full-image-roi-ultra-features-v1")
        self.assertEqual(len(rows), 4)
        self.assertEqual(embedder.batch_sizes, [4])
        self.assertTrue(all(row.proposal_index == 7 for row in rows))
        self.assertTrue(all(tuple(row.values) == CANONICAL_RAW_FEATURE_ORDER for row in rows))
        scale_one = next(row for row in rows if row.scale == 1.0)
        self.assertGreaterEqual(scale_one.values["bbox_x"], 50.0)
        self.assertGreaterEqual(scale_one.values["bbox_y"], 60.0)
        self.assertLess(scale_one.values["area_norm"], 0.02)
        self.assertAlmostEqual(
            scale_one.values["scale_diag"],
            (200**2 + 300**2) ** 0.5,
            places=5,
        )
        self.assertFalse(mask_crop.flags.writeable is False)

    def test_invalid_crop_canvas_closure_fails(self) -> None:
        import numpy as np

        image = np.zeros((32, 32, 3), dtype=np.uint8)
        proposal = SimpleNamespace(
            bbox=(30, 30, 4, 4),
            mask_crop=np.ones((4, 4), dtype=np.uint8),
            proposal_index=1,
            predicted_iou=0.9,
            stability_score=0.9,
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            extract_full_image_raw_features(
                image,
                (proposal,),
                _Embedder(),
                grid=CanonicalGridContext((), ()),
            )


if __name__ == "__main__":
    unittest.main()
