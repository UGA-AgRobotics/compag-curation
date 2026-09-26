from __future__ import annotations

import base64
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from compag_curation.canonical.full_image import (
    full_image_proposal_sha256,
    pack_uncompressed_rle,
)
from compag_curation.public_io import PublicIOError, write_new_json
from compag_curation.review.session import _stream_proposals


_PROPOSAL_COLUMNS = (
    "proposal_id",
    "proposal_sha256",
    "image_id",
    "image_sha256",
    "image_name",
    "group_id",
    "tile_name",
    "tile_sha256",
    "tile_x",
    "tile_y",
    "source_index",
    "proposal_index",
    "mask_sha256",
    "mask_area",
    "mask_encoding",
    "mask_height",
    "mask_width",
    "mask_packbits_base64",
    "tile_bbox_x",
    "tile_bbox_y",
    "tile_bbox_w",
    "tile_bbox_h",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "poly",
    "predicted_iou",
    "stability_score",
)


class FullImageReviewSessionTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        mask_encoding: str = "base64-packbits-little-column-major-bbox-v1",
        bbox_x: int = 620,
    ) -> tuple[SimpleNamespace, dict[str, dict[str, object]], str]:
        stage10 = root / "stages/10_prepare"
        stage20 = root / "stages/20_proposals_features"
        stage10.mkdir(parents=True)
        stage20.mkdir(parents=True)
        image_id = "1" * 64
        image_sha = "2" * 64
        unit_sha = "3" * 64
        unit_name = "card__full.png"

        index_path = stage10 / "tiles_index.csv"
        unit_columns = (
            "unit_name", "image_id", "image_sha256", "image_name", "group_id",
            "unit_sha256", "x", "y", "width", "height", "orig_w", "orig_h",
            "processing_unit_kind",
        )
        with index_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=unit_columns, lineterminator="\n")
            writer.writeheader()
            writer.writerow(
                {
                    "unit_name": unit_name,
                    "image_id": image_id,
                    "image_sha256": image_sha,
                    "image_name": "card.png",
                    "group_id": "card",
                    "unit_sha256": unit_sha,
                    "x": "0",
                    "y": "0",
                    "width": "700",
                    "height": "600",
                    "orig_w": "700",
                    "orig_h": "600",
                    "processing_unit_kind": "full_warped_frame",
                }
            )
        write_new_json(
            stage10 / "prepare_summary.json",
            {
                "schema": "compag-curation-full-image-prepare/v1",
                "status": "PASS",
                "profile": "full-image-multiscale-xgb-recall-gpu-v1",
                "spatial_mode": "full-image-multiscale",
                "image_count": 1,
                "group_count": 1,
                "processing_unit_count": 1,
                "processing_unit_kind": "full_warped_frame",
                "external_tiling": False,
                "unit_format": "png",
                "unit_lossless": True,
                "tiles_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                "images": [
                    {
                        "image_name": "card.png",
                        "image_id": image_id,
                        "image_sha256": image_sha,
                        "group_id": "card",
                        "source_width": 700,
                        "source_height": 600,
                        "warped_width": 700,
                        "warped_height": 600,
                        "warp_mode": "identity_fallback",
                        "inverse_warp": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "row_lines": [],
                        "column_lines": [],
                        "processing_unit_count": 1,
                        "preprocessing_evidence_sha256": "4" * 64,
                    }
                ],
            },
        )

        flat_index = 620 * 600 + 530
        mask = pack_uncompressed_rle(
            {
                "size": [600, 700],
                "counts": [flat_index, 1, 600 * 700 - flat_index - 1],
            },
            canvas_height=600,
            canvas_width=700,
        )
        assert mask is not None
        proposal_id = full_image_proposal_sha256(mask)
        row = {
            "proposal_id": proposal_id,
            "proposal_sha256": proposal_id,
            "image_id": image_id,
            "image_sha256": image_sha,
            "image_name": "card.png",
            "group_id": "card",
            "tile_name": unit_name,
            "tile_sha256": unit_sha,
            "tile_x": "0",
            "tile_y": "0",
            "source_index": "1",
            "proposal_index": "1",
            "mask_sha256": mask.mask_sha256,
            "mask_area": str(mask.area),
            "mask_encoding": mask_encoding,
            "mask_height": str(mask.canvas_height),
            "mask_width": str(mask.canvas_width),
            "mask_packbits_base64": base64.b64encode(mask.packed).decode("ascii"),
            "tile_bbox_x": str(bbox_x),
            "tile_bbox_y": str(mask.origin_y),
            "tile_bbox_w": str(mask.width),
            "tile_bbox_h": str(mask.height),
            "bbox_x": "620",
            "bbox_y": "530",
            "bbox_w": "1",
            "bbox_h": "1",
            "poly": json.dumps([620, 530, 621, 530, 621, 531, 620, 531]),
            "predicted_iou": "0.9",
            "stability_score": "0.8",
        }
        with (stage20 / "proposals.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=_PROPOSAL_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
        overlay = {
            proposal_id: {
                "tile_index": 0,
                "tile_name": unit_name,
                "proposal_index": 1,
                "label_prefix": proposal_id[:8],
                "mask_sha256": mask.mask_sha256,
            }
        }
        return SimpleNamespace(root=root), overlay, proposal_id

    def test_dynamic_bbox_and_compact_mask_are_accepted_beyond_512(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot, overlay, proposal_id = self._fixture(Path(directory))
            proposals, _polygons, digest = _stream_proposals(snapshot, overlay)
            self.assertEqual(proposals[proposal_id].bbox_x, 620)
            self.assertEqual(proposals[proposal_id].bbox_y, 530)
            self.assertEqual(len(digest), 64)

    def test_legacy_mask_encoding_is_rejected_for_full_image_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot, overlay, _proposal_id = self._fixture(
                Path(directory),
                mask_encoding="base64-packbits-little-row-major-v1",
            )
            with self.assertRaisesRegex(PublicIOError, "mask contract changed"):
                _stream_proposals(snapshot, overlay)

    def test_full_image_bbox_outside_dynamic_canvas_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot, overlay, _proposal_id = self._fixture(
                Path(directory),
                bbox_x=700,
            )
            with self.assertRaisesRegex(PublicIOError, "outside its supported range"):
                _stream_proposals(snapshot, overlay)


if __name__ == "__main__":
    unittest.main()
