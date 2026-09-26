from __future__ import annotations

import csv
import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from compag_curation.public_io import PublicIOError, write_new_json
from compag_curation.review.scene import (
    FULL_IMAGE_STAGE20_SCENE_KIND,
    ROUND_SCENE_KIND,
    STAGE20_SCENE_KIND,
    RoundSceneCatalog,
    SceneProposal,
    Stage20SceneCatalog,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jpeg(path: Path, color: tuple[int, int, int]) -> str:
    image = Image.new("RGB", (512, 512), color)
    image.save(path, format="JPEG", quality=95, subsampling=0)
    return _sha(path.read_bytes())


def _png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", size, color)
    image.save(path, format="PNG", optimize=False, compress_level=6)
    return path.read_bytes()


class FullImageStage20SceneTests(unittest.TestCase):
    def _catalog(self, root: Path) -> tuple[Stage20SceneCatalog, str, Path]:
        stage10 = root / "stages/10_prepare"
        stage20 = root / "stages/20_proposals_features"
        tiles = stage10 / "tiles"
        tiles.mkdir(parents=True)
        stage20.mkdir(parents=True)

        image_id = "1" * 64
        image_sha = "2" * 64
        proposal_id = "3" * 64
        first_name = "card_y00000x00000.jpg"
        second_name = "card_y00000x00256.jpg"
        first_sha = _jpeg(tiles / first_name, (240, 10, 10))
        second_sha = _jpeg(tiles / second_name, (10, 20, 240))

        tile_index = stage10 / "tiles_index.csv"
        columns = (
            "tile_name", "image_id", "image_sha256", "image_name", "group_id",
            "tile_sha256", "x", "y", "crop_w", "crop_h", "orig_w", "orig_h",
        )
        with tile_index.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "tile_name": first_name,
                        "image_id": image_id,
                        "image_sha256": image_sha,
                        "image_name": "card.jpg",
                        "group_id": "card",
                        "tile_sha256": first_sha,
                        "x": "0",
                        "y": "0",
                        "crop_w": "512",
                        "crop_h": "512",
                        "orig_w": "768",
                        "orig_h": "512",
                    },
                    {
                        "tile_name": second_name,
                        "image_id": image_id,
                        "image_sha256": image_sha,
                        "image_name": "card.jpg",
                        "group_id": "card",
                        "tile_sha256": second_sha,
                        "x": "256",
                        "y": "0",
                        "crop_w": "512",
                        "crop_h": "512",
                        "orig_w": "768",
                        "orig_h": "512",
                    },
                ]
            )
        summary = {
            "schema": "compag-curation-canonical-prepare/v1",
            "status": "PASS",
            "profile": "canonical-xgb-recall-gpu-v1",
            "image_count": 1,
            "group_count": 1,
            "tile_count": 2,
            "tile_size": 512,
            "tile_stride": 512,
            "far_edge_alignment": True,
            "padding": "bottom-right-edge-value",
            "tile_format": "jpg",
            "tiles_index_sha256": _sha(tile_index.read_bytes()),
            "images": [
                {
                    "image_name": "card.jpg",
                    "image_id": image_id,
                    "image_sha256": image_sha,
                    "group_id": "card",
                    "source_width": 768,
                    "source_height": 512,
                    "warped_width": 768,
                    "warped_height": 512,
                    "warp_mode": "identity",
                    "inverse_warp": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "row_lines": [],
                    "column_lines": [],
                    "tile_count": 2,
                    "preprocessing_evidence_sha256": "4" * 64,
                }
            ],
        }
        write_new_json(stage10 / "prepare_summary.json", summary)
        (stage10 / "_SUCCESS.json").write_bytes(b"stage-10 receipt\n")
        (stage20 / "_SUCCESS.json").write_bytes(b"stage-20 receipt\n")
        proposals_path = stage20 / "proposals.csv"
        proposals_path.write_bytes(b"sealed proposal table\n")
        overlay_path = stage20 / "overlay_manifest.json"
        overlay_path.write_bytes(b"sealed overlay manifest\n")

        view = SimpleNamespace(
            image_id=image_id,
            image_name="card.jpg",
            group_id="card",
            label_prefix=proposal_id[:12],
            tile_x=0,
            tile_y=0,
            bbox_x=10,
            bbox_y=20,
            bbox_w=30,
            bbox_h=40,
        )
        catalog = Stage20SceneCatalog.open(
            root,
            stage10_receipt_sha256=_sha((stage10 / "_SUCCESS.json").read_bytes()),
            stage20_receipt_sha256=_sha((stage20 / "_SUCCESS.json").read_bytes()),
            proposals_sha256=_sha(proposals_path.read_bytes()),
            overlay_manifest_sha256=_sha(overlay_path.read_bytes()),
            proposal_views={proposal_id: view},
            proposal_order=(proposal_id,),
            source_polygons={proposal_id: (10, 20, 40, 20, 40, 60, 10, 60)},
        )
        return catalog, proposal_id, proposals_path

    def _direct_catalog(
        self,
        root: Path,
    ) -> tuple[Stage20SceneCatalog, str, Path, bytes]:
        stage10 = root / "stages/10_prepare"
        stage20 = root / "stages/20_proposals_features"
        units = stage10 / "processing_units"
        units.mkdir(parents=True)
        stage20.mkdir(parents=True)

        image_id = "a" * 64
        image_sha = "b" * 64
        proposal_id = "c" * 64
        unit_name = "card__full.png"
        png_payload = _png(units / unit_name, (700, 600), (21, 43, 65))
        unit_sha = _sha(png_payload)
        unit_index = stage10 / "tiles_index.csv"
        columns = (
            "unit_name", "image_id", "image_sha256", "image_name", "group_id",
            "unit_sha256", "x", "y", "width", "height", "orig_w", "orig_h",
            "processing_unit_kind",
        )
        with unit_index.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
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
        summary = {
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
            "tiles_index_sha256": _sha(unit_index.read_bytes()),
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
                    "preprocessing_evidence_sha256": "d" * 64,
                }
            ],
        }
        write_new_json(stage10 / "prepare_summary.json", summary)
        (stage10 / "_SUCCESS.json").write_bytes(b"stage-10 full-image receipt\n")
        (stage20 / "_SUCCESS.json").write_bytes(b"stage-20 full-image receipt\n")
        proposals_path = stage20 / "proposals.csv"
        proposals_path.write_bytes(b"sealed full-image proposal table\n")
        overlay_path = stage20 / "overlay_manifest.json"
        overlay_path.write_bytes(b"sealed full-image overlay manifest\n")
        view = SimpleNamespace(
            image_id=image_id,
            image_name="card.png",
            group_id="card",
            tile_name=unit_name,
            label_prefix=proposal_id[:12],
            tile_x=0,
            tile_y=0,
            bbox_x=620,
            bbox_y=530,
            bbox_w=60,
            bbox_h=50,
        )
        catalog = Stage20SceneCatalog.open(
            root,
            stage10_receipt_sha256=_sha((stage10 / "_SUCCESS.json").read_bytes()),
            stage20_receipt_sha256=_sha((stage20 / "_SUCCESS.json").read_bytes()),
            proposals_sha256=_sha(proposals_path.read_bytes()),
            overlay_manifest_sha256=_sha(overlay_path.read_bytes()),
            proposal_views={proposal_id: view},
            proposal_order=(proposal_id,),
            source_polygons={proposal_id: (620, 530, 680, 530, 680, 580, 620, 580)},
        )
        return catalog, proposal_id, units / unit_name, png_payload

    def test_stitches_native_full_image_and_keeps_outline_separate_from_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog, proposal_id, _proposals_path = self._catalog(Path(directory))
            payload = catalog.payload(proposal_id, {proposal_id: None})
            self.assertEqual(payload["kind"], STAGE20_SCENE_KIND)
            self.assertEqual((payload["width"], payload["height"]), (768, 512))
            self.assertEqual(payload["overlay_defaults"], {"mask_outline": True, "bbox": False})
            geometry = payload["proposals"][0]
            self.assertEqual(geometry["polygon"], [10, 20, 40, 20, 40, 60, 10, 60])
            self.assertEqual(geometry["bbox"], [10, 20, 30, 40])

            first = catalog.bytes(payload["scene_id"])
            second = catalog.bytes(payload["scene_id"])
            self.assertEqual(first, second)
            self.assertEqual(struct.unpack(">II", first[16:24]), (768, 512))
            with Image.open(io.BytesIO(first)) as mosaic:
                self.assertGreater(mosaic.getpixel((20, 20))[0], 200)
                # The deterministic sorted stitch makes the later x=256 tile
                # own the overlap and the far-right part of the mosaic.
                self.assertGreater(mosaic.getpixel((300, 20))[2], 200)
                self.assertGreater(mosaic.getpixel((740, 20))[2], 200)

    def test_metadata_tamper_fails_closed_before_payload_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog, proposal_id, proposals_path = self._catalog(Path(directory))
            proposals_path.write_bytes(b"tampered proposal table with a different size\n")
            with self.assertRaisesRegex(PublicIOError, "changed after scene indexing"):
                catalog.payload(proposal_id, {})

    def test_full_profile_uses_verified_dynamic_png_without_stitching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog, proposal_id, _unit_path, expected_png = self._direct_catalog(
                Path(directory)
            )
            payload = catalog.payload(proposal_id, {proposal_id: None})
            self.assertEqual(payload["kind"], FULL_IMAGE_STAGE20_SCENE_KIND)
            self.assertEqual((payload["width"], payload["height"]), (700, 600))
            self.assertEqual(
                payload["stitch_policy"],
                "DIRECT_VERIFIED_SINGLE_LOSSLESS_PNG_NO_STITCH",
            )
            self.assertEqual(payload["proposals"][0]["bbox"], [620, 530, 60, 50])
            self.assertEqual(catalog.bytes(payload["scene_id"]), expected_png)

    def test_full_profile_png_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog, proposal_id, unit_path, png_payload = self._direct_catalog(
                Path(directory)
            )
            unit_path.write_bytes(png_payload + b"tamper")
            with self.assertRaisesRegex(PublicIOError, "changed after scene indexing"):
                catalog.payload(proposal_id, {})


class FullImageRoundSceneTests(unittest.TestCase):
    def test_round_scene_uses_full_original_and_all_stage60_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage60 = root / "stage60"
            images = root / "images"
            stage60.mkdir()
            images.mkdir()
            predictions = b"sealed predictions fixture\n"
            inference = b"sealed inference fixture\n"
            (stage60 / "predictions.csv").write_bytes(predictions)
            (stage60 / "inference_result.json").write_bytes(inference)
            image_path = images / "heldout.png"
            image_payload = _png(image_path, (100, 60), (12, 34, 56))
            image_id = "5" * 64
            image_sha = _sha(image_payload)
            first_id, second_id = "6" * 64, "7" * 64
            source = SimpleNamespace(
                name="heldout.png",
                path=image_path,
                image_id=image_id,
                image_sha256=image_sha,
                group_id="heldout",
                size_bytes=len(image_payload),
                width=100,
                height=60,
            )
            internal = {
                "image_name": "heldout.png",
                "image_sha256": image_sha,
                "group_id": "heldout",
                "width": 100,
                "height": 60,
            }
            proposals = {
                first_id: SceneProposal(
                    first_id,
                    image_id,
                    first_id[:12],
                    (2, 3, 10, 11),
                    (2, 3, 12, 3, 12, 14, 2, 14),
                    {**internal, "xgb_p": 0.49, "prediction": 0, "kept": 0},
                ),
                second_id: SceneProposal(
                    second_id,
                    image_id,
                    second_id[:12],
                    (40, 20, 12, 8),
                    (40, 20, 52, 20, 52, 28, 40, 28),
                    {**internal, "xgb_p": 0.9, "prediction": 1, "kept": 1},
                ),
            }
            catalog = RoundSceneCatalog.open(
                stage60,
                predictions_sha256=_sha(predictions),
                inference_result_sha256=_sha(inference),
                images=(source,),
                proposals=proposals,
                reviewable_order=(first_id,),
                profile="canonical-xgb-recall-gpu-v1",
                bundle_sha256="8" * 64,
            )
            payload = catalog.payload(first_id, {first_id: "target"})
            self.assertEqual(payload["kind"], ROUND_SCENE_KIND)
            self.assertEqual((payload["width"], payload["height"]), (100, 60))
            self.assertEqual(payload["proposal_count"], 2)
            self.assertEqual(payload["reviewable_proposal_count"], 1)
            by_id = {item["proposal_id"]: item for item in payload["proposals"]}
            self.assertTrue(by_id[first_id]["reviewable"])
            self.assertFalse(by_id[second_id]["reviewable"])
            self.assertEqual(by_id[second_id]["model"]["prediction"], 1)
            rendered = catalog.bytes(payload["scene_id"])
            self.assertEqual(struct.unpack(">II", rendered[16:24]), (100, 60))

            image_path.write_bytes(image_payload + b"tamper")
            with self.assertRaisesRegex(PublicIOError, "changed after scene indexing"):
                catalog.bytes(payload["scene_id"])


if __name__ == "__main__":
    unittest.main()
