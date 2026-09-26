from __future__ import annotations

import csv
import hashlib
import inspect
import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from compag_curation.canonical import service
from compag_curation.canonical.active_learning_service import (
    read_canonical_active_learning_inference,
    write_initial_canonical_active_learning_pool,
)
from compag_curation.canonical.features import (
    CANONICAL_RAW_FEATURE_ORDER,
    CanonicalFeatureState,
    CanonicalRawFeature,
    canonical_raw_feature_csv_row,
)
from compag_curation.canonical.proposals import canonical_mask_sha256
from compag_curation.canonical.preprocessing import (
    CanonicalPreparedImage,
    CanonicalTile,
)
from compag_curation.canonical.spec import (
    CANONICAL_FEATURE_CROP_SCALES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FEATURE_ORDER_SHA256,
    CANONICAL_GPU_PROFILE,
    CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
    CANONICAL_PROFILE,
    CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256,
    CANONICAL_SAM2_HIERA_L_CONFIG_SHA256,
    CANONICAL_STAGE20_AMG_POINTS_PER_BATCH,
    EFFICIENT_GPU_PROFILE,
)
from compag_curation.model_bundle import BUNDLE_SCHEMA_V2
from compag_curation.public_config import RESNET50_WEIGHTS_SHA256
from compag_curation.public_io import PublicIOError, write_new_json


def _write_ppm(path: Path, width: int = 64, height: int = 64) -> None:
    path.write_bytes(
        f"P6\n{width} {height}\n255\n".encode("ascii")
        + bytes([255, 220, 0]) * width * height
    )


def _config(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        profile=CANONICAL_PROFILE,
        is_canonical=True,
        device="cpu",
        seed=42,
        tile_size=512,
        tile_stride=512,
        tile_overlap=0,
        tile_format="jpg",
        checkpoint=root / "assets" / "sam.pt",
        checkpoint_sha256=CANONICAL_SAM2_HIERA_L_CHECKPOINT_SHA256,
        sam2_config=root / "assets" / "sam.yaml",
        sam2_config_sha256=CANONICAL_SAM2_HIERA_L_CONFIG_SHA256,
        sam2_config_locator="configs/sam2.1/sam2.1_hiera_l",
        embedding_weights=root / "assets" / "resnet.pth",
        embedding_weights_sha256=RESNET50_WEIGHTS_SHA256,
        yolo_enabled=False,
        proposal_scales=(1.0,),
        feature_crop_scales=CANONICAL_FEATURE_CROP_SCALES,
        inference_threshold=0.5,
        nms_iou_threshold=0.5,
        images=root / "images",
        inference_images=root / "inference_images",
        asset_root=root / "assets",
    )


def _prepared(image_name: str, pixels: object) -> CanonicalPreparedImage:
    import numpy as np

    tile = CanonicalTile(
        name=f"{Path(image_name).stem}_y00000x00000.jpg",
        x=0,
        y=0,
        crop_width=64,
        crop_height=64,
        pixels=pixels,
    )
    evidence = {
        "schema": "compag-curation-canonical-preprocessing/v1",
        "warp": {"inverse_matrix": np.eye(3, dtype=np.float32).tolist()},
        "grid": {},
        "tiling": {},
    }
    return CanonicalPreparedImage(
        warped_bgr=pixels,
        inverse_warp=np.eye(3, dtype=np.float32),
        warp_mode="identity_fallback",
        row_lines=(10, 20, 30, 40, 50),
        column_lines=(10, 30, 50),
        tiles=(tile,),
        evidence=evidence,
    )


class _FakeAMG:
    def generate(self, image: object) -> list[dict[str, object]]:
        import numpy as np

        # The tile is deliberately BGR=(0,220,255); AMG must receive RGB.
        if not (int(image[0, 0, 0]) > int(image[0, 0, 2])):
            raise AssertionError("canonical AMG input was not converted BGR -> RGB")
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[8:24, 12:30] = 1
        return [
            {
                "segmentation": mask,
                "predicted_iou": 0.91,
                "stability_score": 0.93,
            }
        ]


class _FakeEmbedder:
    def embed_many(self, patches: object) -> object:
        raise AssertionError("raw feature fake should own embedding output")


class CanonicalServiceTests(unittest.TestCase):
    def test_historical_evaluator_preserves_public_ppm_names(self) -> None:
        from compag_curation.evaluation.points_coverage_base import norm_img_name

        self.assertEqual(norm_img_name("synthetic-heldout__card.ppm"), "synthetic-heldout__card.ppm")
        self.assertEqual(norm_img_name("synthetic-heldout__card"), "synthetic-heldout__card.jpg")

    def test_stage10_and_stage20_strict_lossless_artifacts(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("images", "inference_images", "assets", "stage10", "stage20"):
                (root / name).mkdir()
            source = root / "images" / "card01__view.ppm"
            _write_ppm(source)
            config = _config(root)
            pixels = np.full((512, 512, 3), (0, 220, 255), dtype=np.uint8)

            def fake_prepare(_decoded: object, image_name: str, *, locate_grid: bool) -> object:
                self.assertTrue(locate_grid)
                return _prepared(image_name, pixels)

            def jpeg(tile: CanonicalTile) -> bytes:
                ok, encoded = cv2.imencode(".jpg", tile.pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
                self.assertTrue(ok)
                return bytes(encoded)

            summary = service.prepare_canonical_stage(
                config,
                root / "stage10",
                prepare_image=fake_prepare,
                jpeg_encoder=jpeg,
            )
            self.assertEqual(summary["tile_count"], 1)
            tile_path = next((root / "stage10" / "tiles").iterdir())
            self.assertEqual(stat.S_IMODE(tile_path.stat().st_mode), 0o644)
            with (root / "stage10" / "tiles_index.csv").open(newline="") as handle:
                self.assertEqual(tuple(csv.DictReader(handle).fieldnames or ()), service.CANONICAL_TILE_COLUMNS)

            def fake_raw(
                _image: object,
                proposals: object,
                _embedder: object,
                *,
                grid: object,
            ) -> tuple[CanonicalRawFeature, ...]:
                self.assertEqual(grid.tile_x, 0)
                vector = np.zeros(2048, dtype=np.float32)
                vector[0] = 1.0
                proposal = tuple(proposals)[0]
                return tuple(
                    CanonicalRawFeature(
                        proposal_index=proposal.proposal_index,
                        scale=scale,
                        predicted_iou=proposal.predicted_iou,
                        stability_score=proposal.stability_score,
                        values={name: 1.0 for name in CANONICAL_RAW_FEATURE_ORDER},
                        embedding=vector,
                    )
                    for scale in CANONICAL_FEATURE_CROP_SCALES
                )

            def runtime_factory(_assets: object) -> service.CanonicalStageRuntime:
                return service.CanonicalStageRuntime(
                    amg=_FakeAMG(),
                    embedder=_FakeEmbedder(),
                )
            with (
                mock.patch.object(service, "extract_canonical_raw_features", fake_raw),
                mock.patch.object(cv2, "putText", wraps=cv2.putText) as put_text,
                mock.patch.object(
                    service,
                    "_load_stage20_runtime",
                    wraps=service._load_stage20_runtime,
                ) as stage20_loader,
            ):
                proposal_summary = service._generate_canonical_stage_with_dependencies(
                    config,
                    root / "stage10",
                    root / "stage20",
                    runtime_factory=runtime_factory,
                    write_overlays=True,
                )
            stage20_loader.assert_called_once()
            self.assertEqual(
                stage20_loader.call_args.args[1:],
                ("cpu", runtime_factory),
            )
            self.assertEqual(
                set(proposal_summary),
                {
                    "schema",
                    "status",
                    "profile",
                    "sam2_execution_points_per_batch",
                    "tile_count",
                    "proposal_count",
                    "raw_feature_row_count",
                    "raw_rows_per_proposal",
                    "amg_input_count",
                    "empty_mask_count",
                    "merge_iou_suppressed_count",
                    "merge_iou_threshold",
                    "post_dedup_cap_count",
                    "full_masks_retained_losslessly",
                    "mask_encoding",
                    "proposals_sha256",
                    "raw_features_sha256",
                    "proposal_config_sha256",
                    "overlays",
                    "overlay_manifest",
                    "overlay_manifest_sha256",
                    "overlay_tile_count",
                    "overlay_proposal_count",
                    "overlay_proposal_ids_sha256",
                },
            )
            self.assertEqual(proposal_summary["proposal_count"], 1)
            self.assertEqual(proposal_summary["raw_feature_row_count"], 4)
            self.assertEqual(
                proposal_summary["sam2_execution_points_per_batch"],
                CANONICAL_STAGE20_AMG_POINTS_PER_BATCH,
            )
            proposal_config = json.loads(
                (root / "stage20" / "proposal_config.json").read_text(
                    encoding="ascii"
                )
            )
            self.assertEqual(
                proposal_config["proposal_identity"],
                "compag-canonical-proposal-v2",
            )
            self.assertEqual(proposal_config["points_per_batch"], 512)
            self.assertEqual(
                proposal_config["sam2_execution_points_per_batch"],
                CANONICAL_STAGE20_AMG_POINTS_PER_BATCH,
            )
            self.assertEqual(
                proposal_summary["sam2_execution_points_per_batch"],
                proposal_config["sam2_execution_points_per_batch"],
            )
            self.assertEqual(
                proposal_summary["proposal_config_sha256"],
                hashlib.sha256(
                    (root / "stage20" / "proposal_config.json").read_bytes()
                ).hexdigest(),
            )
            with (root / "stage20" / "proposals.csv").open(newline="") as handle:
                proposal_rows = list(csv.DictReader(handle))
            proposal_id = proposal_rows[0]["proposal_id"]
            overlay_manifest_path = root / "stage20" / "overlay_manifest.json"
            overlay_manifest = json.loads(
                overlay_manifest_path.read_text(encoding="ascii")
            )
            overlay_row = overlay_manifest["tiles"][0]
            overlay_path = root / "stage20" / overlay_row["overlay"]
            expected_label = proposal_id[:8]
            self.assertEqual(proposal_summary["overlays"], "overlays")
            self.assertEqual(
                proposal_summary["overlay_manifest_sha256"],
                hashlib.sha256(overlay_manifest_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(proposal_summary["overlay_proposal_count"], 1)
            self.assertEqual(overlay_manifest["proposal_count"], 1)
            self.assertEqual(
                proposal_summary["overlay_proposal_ids_sha256"],
                service.compact_json_sha256([proposal_id]),
            )
            self.assertEqual(
                proposal_summary["overlay_proposal_ids_sha256"],
                overlay_manifest["proposal_ids_sha256"],
            )
            self.assertEqual(
                overlay_manifest["tiles_sha256"],
                service.compact_json_sha256(overlay_manifest["tiles"]),
            )
            self.assertEqual(
                overlay_row["overlay_sha256"],
                hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                overlay_row["proposal_labels"],
                [
                    {
                        "label": expected_label,
                        "proposal_id": proposal_id,
                        "proposal_index": int(proposal_rows[0]["proposal_index"]),
                        "mask_sha256": proposal_rows[0]["mask_sha256"],
                    }
                ],
            )
            self.assertEqual(
                overlay_row["proposal_labels_sha256"],
                service.compact_json_sha256(overlay_row["proposal_labels"]),
            )
            self.assertEqual(len(put_text.call_args_list), 2)
            self.assertTrue(
                all(call.args[1] == expected_label for call in put_text.call_args_list)
            )
            decoded = service.decode_lossless_mask(
                proposal_rows[0]["mask_height"],
                proposal_rows[0]["mask_width"],
                proposal_rows[0]["mask_packbits_base64"],
            )
            self.assertEqual(int(decoded.sum()), 16 * 18)
            with (root / "stage20" / "features.csv").open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), service.CANONICAL_RAW_FEATURE_TABLE_COLUMNS)
                self.assertEqual(len(list(reader)), 4)

    def test_stage20_rejects_a_genesis_image_without_any_proposal(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("images", "inference_images", "assets", "stage10"):
                (root / name).mkdir()
            _write_ppm(root / "images/card01__view.ppm")
            _write_ppm(root / "images/card02__view.ppm", width=63)
            config = _config(root)
            pixels = np.full((512, 512, 3), (0, 220, 255), dtype=np.uint8)

            def fake_prepare(
                _decoded: object,
                image_name: str,
                *,
                locate_grid: bool,
            ) -> object:
                self.assertTrue(locate_grid)
                return _prepared(image_name, pixels)

            def jpeg(tile: CanonicalTile) -> bytes:
                ok, encoded = cv2.imencode(
                    ".jpg",
                    tile.pixels,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                self.assertTrue(ok)
                return bytes(encoded)

            service.prepare_canonical_stage(
                config,
                root / "stage10",
                prepare_image=fake_prepare,
                jpeg_encoder=jpeg,
            )

            class FirstImageOnlyAMG(_FakeAMG):
                def __init__(self) -> None:
                    self.calls = 0

                def generate(self, image: object) -> list[dict[str, object]]:
                    self.calls += 1
                    if self.calls == 2:
                        return []
                    return super().generate(image)

            def fake_raw(
                _image: object,
                proposals: object,
                _embedder: object,
                *,
                grid: object,
            ) -> tuple[CanonicalRawFeature, ...]:
                vector = np.zeros(2048, dtype=np.float32)
                vector[0] = 1.0
                proposal_rows = tuple(proposals)
                if not proposal_rows:
                    return ()
                proposal = proposal_rows[0]
                return tuple(
                    CanonicalRawFeature(
                        proposal_index=proposal.proposal_index,
                        scale=scale,
                        predicted_iou=proposal.predicted_iou,
                        stability_score=proposal.stability_score,
                        values={
                            name: 1.0 for name in CANONICAL_RAW_FEATURE_ORDER
                        },
                        embedding=vector,
                    )
                    for scale in CANONICAL_FEATURE_CROP_SCALES
                )

            amg = FirstImageOnlyAMG()

            def runtime_factory(_assets: object) -> service.CanonicalStageRuntime:
                return service.CanonicalStageRuntime(
                    amg=amg,
                    embedder=_FakeEmbedder(),
                )

            output = root / "stage20"
            output.mkdir()
            with mock.patch.object(
                service,
                "extract_canonical_raw_features",
                fake_raw,
            ):
                with self.assertRaisesRegex(
                    PublicIOError,
                    "at least one proposal for every genesis image",
                ):
                    service._generate_canonical_stage_with_dependencies(
                        config,
                        root / "stage10",
                        output,
                        runtime_factory=runtime_factory,
                    )
            self.assertEqual(amg.calls, 2)
            self.assertFalse((output / "proposals.csv").exists())
            self.assertFalse((output / "features.csv").exists())

    def test_overlay_labels_extend_colliding_proposal_id_prefixes(self) -> None:
        mask_hashes = ("1" * 64, "2" * 64)
        proposal_ids = ("a" * 8 + "0" * 56, "a" * 8 + "1" * 56)
        proposals = tuple(
            SimpleNamespace(proposal_index=index, mask_sha256=mask_hash)
            for index, mask_hash in enumerate(mask_hashes)
        )
        rows = tuple(
            {
                "proposal_id": proposal_id,
                "proposal_sha256": proposal_id,
                "proposal_index": str(index),
                "mask_sha256": mask_hashes[index],
            }
            for index, proposal_id in enumerate(proposal_ids)
        )

        prefix_length, labels = service._overlay_proposal_labels(proposals, rows)

        self.assertEqual(prefix_length, 9)
        self.assertEqual(
            [row["label"] for row in labels],
            [proposal_id[:9] for proposal_id in proposal_ids],
        )

    def test_canonical_nms_breaks_probability_ties_by_proposal_id(self) -> None:
        import numpy as np

        kept = service._canonical_nms_indices(
            np.asarray([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32),
            np.asarray([0.75, 0.75], dtype=np.float32),
            ("b" * 64, "a" * 64),
            0.5,
        )

        self.assertEqual(kept, [1])

    def test_group_split_excludes_skip_and_proves_five_folds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reviewed = root / "reviewed.csv"
            rows = []
            for group_index in range(6):
                for label in (0, 1):
                    proposal = hashlib.sha256(f"{group_index}:{label}".encode("ascii")).hexdigest()
                    rows.append(
                        {
                            "proposal_id": proposal,
                            "proposal_sha256": proposal,
                            "image_id": hashlib.sha256(f"image:{group_index}".encode("ascii")).hexdigest(),
                            "image_sha256": hashlib.sha256(f"bytes:{group_index}".encode("ascii")).hexdigest(),
                            "group_id": f"group{group_index}",
                            "label": str(label),
                            "review_action": "accept",
                            "review_weight": "1.0",
                            "review_status": "reviewed",
                        }
                    )
            skipped = hashlib.sha256(b"skipped").hexdigest()
            rows.append(
                {
                    "proposal_id": skipped,
                    "proposal_sha256": skipped,
                    "image_id": hashlib.sha256(b"skip-image").hexdigest(),
                    "image_sha256": hashlib.sha256(b"skip-bytes").hexdigest(),
                    "group_id": "skiponly",
                    "label": "1",
                    "review_action": "skip",
                    "review_weight": "0.0",
                    "review_status": "reviewed",
                }
            )
            with reviewed.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=service.CANONICAL_REVIEW_COLUMNS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            output = root / "stage40"
            output.mkdir()
            result = service.build_canonical_group_split(reviewed, output)
            self.assertEqual(result["group_cv_folds"], 5)
            self.assertEqual(result["skipped_rows"], 1)
            self.assertNotIn("skiponly", result["train_groups"])
            self.assertNotIn("skiponly", result["test_groups"])
            self.assertFalse(set(result["train_groups"]) & set(result["test_groups"]))
            tampered = dict(result)
            tampered["coverage"] = {
                **result["coverage"],
                "test": {**result["coverage"]["test"], "positive": 99},
            }
            tampered_path = root / "tampered-split.json"
            write_new_json(tampered_path, tampered)
            reviewed_payload = reviewed.read_bytes()
            with self.assertRaisesRegex(PublicIOError, "differs from the reviewed table"):
                service._canonical_split(
                    tampered_path,
                    hashlib.sha256(reviewed_payload).hexdigest(),
                    service._canonical_review_rows(reviewed, payload=reviewed_payload),
                )

    def test_stage50_builds_v2_request_from_train_only_state(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("images", "inference_images", "assets", "stage20", "stage40", "stage50"):
                (root / name).mkdir()
            config = _config(root)
            config.profile = CANONICAL_GPU_PROFILE
            config.device = "cuda"
            mask = np.zeros((512, 512), dtype=np.uint8)
            mask[2:8, 3:10] = 1
            mask_sha256 = canonical_mask_sha256(mask)
            mask_height, mask_width, encoded_mask = service._encode_mask(mask)
            embedding = np.zeros(2048, dtype=np.float32)
            embedding[0] = 1.0
            proposal_rows = []
            raw_rows = []
            review_rows = []
            for group_index in range(6):
                image_name = f"group{group_index}__view.ppm"
                group_id = f"group{group_index}"
                image_sha256 = hashlib.sha256(f"image-bytes:{group_index}".encode("ascii")).hexdigest()
                image_id = hashlib.sha256(
                    b"compag-image-v1\0"
                    + image_name.encode("utf-8")
                    + b"\0"
                    + group_id.encode("ascii")
                    + b"\0"
                    + image_sha256.encode("ascii")
                ).hexdigest()
                tile_sha256 = hashlib.sha256(f"tile:{group_index}".encode("ascii")).hexdigest()
                for label in (0, 1):
                    proposal_id = hashlib.sha256(f"proposal:{group_index}:{label}".encode("ascii")).hexdigest()
                    proposal = {
                        "proposal_id": proposal_id,
                        "proposal_sha256": proposal_id,
                        "image_id": image_id,
                        "image_sha256": image_sha256,
                        "image_name": image_name,
                        "group_id": group_id,
                        "tile_name": f"group{group_index}__view_y00000x00000.jpg",
                        "tile_sha256": tile_sha256,
                        "tile_x": "0",
                        "tile_y": "0",
                        "source_index": str(label),
                        "proposal_index": str(label + 1),
                        "mask_sha256": mask_sha256,
                        "mask_area": str(int(mask.sum())),
                        "mask_encoding": service.MASK_ENCODING,
                        "mask_height": str(mask_height),
                        "mask_width": str(mask_width),
                        "mask_packbits_base64": encoded_mask,
                        "tile_bbox_x": "3",
                        "tile_bbox_y": "2",
                        "tile_bbox_w": "7",
                        "tile_bbox_h": "6",
                        "bbox_x": "3",
                        "bbox_y": "2",
                        "bbox_w": "7",
                        "bbox_h": "6",
                        "poly": "[3,2,10,2,10,8,3,8]",
                        "predicted_iou": "0.91",
                        "stability_score": "0.93",
                    }
                    proposal_rows.append(proposal)
                    identity = {
                        name: proposal[name]
                        for name in service._RAW_IDENTITY_COLUMNS
                    }
                    for scale in CANONICAL_FEATURE_CROP_SCALES:
                        raw = CanonicalRawFeature(
                            proposal_index=label + 1,
                            scale=scale,
                            predicted_iou=0.91,
                            stability_score=0.93,
                            values={name: 1.0 for name in CANONICAL_RAW_FEATURE_ORDER},
                            embedding=embedding,
                        )
                        raw_rows.append({**identity, **canonical_raw_feature_csv_row(raw)})
                    review_rows.append(
                        {
                            "proposal_id": proposal_id,
                            "proposal_sha256": proposal_id,
                            "image_id": image_id,
                            "image_sha256": image_sha256,
                            "group_id": group_id,
                            "label": str(label),
                            "review_action": "accept",
                            "review_weight": "1.0",
                            "review_status": "reviewed",
                        }
                    )
            proposals_path = root / "stage20" / "proposals.csv"
            features_path = root / "stage20" / "features.csv"
            reviewed_path = root / "reviewed.csv"
            service._write_csv_exact(proposals_path, service.CANONICAL_PROPOSAL_COLUMNS, proposal_rows)
            service._write_csv_exact(features_path, service.CANONICAL_RAW_FEATURE_TABLE_COLUMNS, raw_rows)
            service._write_csv_exact(reviewed_path, service.CANONICAL_REVIEW_COLUMNS, review_rows)
            self.assertNotEqual(
                [row["proposal_id"] for row in raw_rows[::4]],
                sorted(row["proposal_id"] for row in raw_rows[::4]),
            )
            initial_pool_result = write_initial_canonical_active_learning_pool(
                features_path,
                root / "initial_pool",
            )
            self.assertEqual(initial_pool_result["prediction_rows"], len(review_rows))
            self.assertEqual(
                initial_pool_result["source_sha256"],
                hashlib.sha256(features_path.read_bytes()).hexdigest(),
            )
            service.build_canonical_group_split(
                reviewed_path,
                root / "stage40",
                profile=CANONICAL_GPU_PROFILE,
            )

            class FakeBooster:
                feature_names = None

            class FakeClassifier:
                def __init__(self) -> None:
                    self.booster = FakeBooster()

                def get_booster(self) -> object:
                    return self.booster

            classifier = FakeClassifier()

            def fake_state(
                _raw: object,
                labels: object,
                training_indices: object,
            ) -> CanonicalFeatureState:
                indices = list(training_indices)
                prototype = np.zeros(2048, dtype=np.float32)
                prototype[0] = 1.0
                return CanonicalFeatureState(
                    prototype=prototype,
                    pca_components=np.zeros((32, 2048), dtype=np.float32),
                    pca_mean=np.zeros(2048, dtype=np.float32),
                    training_row_count=len(indices),
                    positive_row_count=sum(int(labels[index]) == 1 for index in indices),
                )

            def fake_train(
                _features: object,
                labels: object,
                groups: object,
                _actions: object,
                weights: object,
                *,
                scales: object,
                split: object,
                config: object,
            ) -> object:
                self.assertEqual(config.device, "cuda")
                self.assertEqual(len(scales), len(labels))
                self.assertEqual(set(scales), {0.67, 0.8, 1.0, 1.25})
                selected = [
                    index
                    for index, (group, weight, scale) in enumerate(
                        zip(groups, weights, scales, strict=True)
                    )
                    if group in split.test_groups
                    and float(weight) > 0.0
                    and float(scale) == 1.0
                ]
                return SimpleNamespace(
                    classifier=classifier,
                    imputer_statistics=(0.0,) * 93,
                    feature_order=CANONICAL_FEATURE_ORDER,
                    fixed_threshold=0.5,
                    test_probabilities=(0.75,) * len(selected),
                    test_labels=tuple(int(labels[index]) for index in selected),
                    test_groups=tuple(str(groups[index]) for index in selected),
                    used_trees=17,
                    search=SimpleNamespace(
                        sampled_configuration_count=30,
                        fold_count=5,
                        best_average_precision=0.8,
                        best_parameters={"max_depth": 4},
                        score_ledger=(0.8,) * 30,
                    ),
                    test_metrics={"average_precision": 0.8, "f1": 0.7},
                    retrospective_test_best_f1_threshold=0.6,
                )

            captured = []

            def fake_bundle_writer(request: object) -> dict[str, object]:
                captured.append(request)
                (root / "stage50" / "model_bundle").mkdir()
                return {
                    "status": "PASS",
                    "root": str(root / "stage50" / "model_bundle"),
                    "bundle_sha256": "b" * 64,
                    "members": 22,
                }

            with (
                mock.patch.object(service, "fit_canonical_feature_state", side_effect=fake_state),
                mock.patch.object(service, "finalize_canonical_feature", side_effect=lambda _raw, _state: {name: 1.0 for name in CANONICAL_FEATURE_ORDER}),
                mock.patch.object(service, "train_canonical_xgb", side_effect=fake_train),
                mock.patch.object(service, "portable_classifier_bytes", return_value=b"UBJ"),
                mock.patch.object(service, "verify_probability_parity", return_value={"status": "PASS", "row_count": 8}),
                mock.patch.object(service, "_pca_explained_variance", return_value=(0.0,) * 32),
            ):
                result = service.train_canonical_stage(
                    config,
                    proposals_path,
                    reviewed_path,
                    root / "stage40" / "split_manifest.json",
                    root / "stage50",
                    bundle_writer=fake_bundle_writer,
                )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["threshold_selected_on"], "FIXED_CANONICAL_METHOD")
            self.assertEqual(len(captured), 1)
            request = captured[0]
            from compag_curation.model_bundle import (
                BUNDLE_V2_ASSET_PROFILES,
                _canonical_config_contracts,
            )

            expected_configs = _canonical_config_contracts(
                profile=CANONICAL_GPU_PROFILE,
                asset_profile=BUNDLE_V2_ASSET_PROFILES[CANONICAL_GPU_PROFILE],
                feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
            )
            for role, observed in (
                ("normalized", request.normalized_config),
                ("preprocessing", request.preprocessing),
                ("proposal", request.proposal_config),
                ("features", request.feature_config),
                ("training", request.training_config),
            ):
                self.assertEqual(observed, expected_configs[role])
            self.assertEqual(tuple(request.feature_order), CANONICAL_FEATURE_ORDER)
            self.assertEqual(request.feature_order_sha256, CANONICAL_FEATURE_ORDER_SHA256)
            self.assertEqual(request.provenance["ubj_parity_status"], "PASS")
            self.assertGreaterEqual(request.provenance["training_row_count"], 32)
            self.assertTrue((root / "stage50" / "finalized_features.csv").is_file())
            self.assertTrue((root / "stage50" / "test_predictions.csv").is_file())
            self.assertTrue((root / "stage50" / "training_result.json").is_file())

    def test_fresh_inference_preserves_attempt_and_writes_evaluator_schema(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            bundle_root = root / "bundle"
            output = root / "stage60"
            for path in (images, bundle_root, output):
                path.mkdir()
            _write_ppm(images / "infer01__view.ppm")
            (output / "_ATTEMPT.json").write_text("{}\n", encoding="ascii")
            pixels = np.full((512, 512, 3), (0, 220, 255), dtype=np.uint8)
            prototype = np.zeros(2048, dtype=np.float32)
            prototype[0] = 1.0
            fake_bundle = SimpleNamespace(
                root=bundle_root,
                schema=BUNDLE_SCHEMA_V2,
                profile=CANONICAL_PROFILE,
                feature_order=CANONICAL_FEATURE_ORDER,
                feature_order_sha256=CANONICAL_FEATURE_ORDER_SHA256,
                threshold=0.5,
                classifier=b"ubj",
                imputer=(0.0,) * 93,
                prototype=tuple(float(value) for value in prototype),
                pca_mean=(0.0,) * 2048,
                pca_components=tuple((0.0,) * 2048 for _ in range(32)),
                provenance={"details": {"training_row_count": 32, "positive_training_row_count": 1}},
                sam2_config=bundle_root / "sam.yaml",
                checkpoint=bundle_root / "sam.pt",
                resnet50_weights=bundle_root / "resnet.pth",
                bundle_sha256="f" * 64,
                tree_identity=(),
            )

            def fake_generate(
                _runtime: object,
                _tile: object,
                tile_row: object,
                _image_record: object,
                *,
                attest_cuda: bool,
            ) -> object:
                self.assertFalse(attest_cuda)
                proposal_id = hashlib.sha256(
                    str(tile_row["image_id"]).encode("ascii")
                ).hexdigest()
                proposal = {
                    "proposal_id": proposal_id,
                    "proposal_sha256": proposal_id,
                    "image_id": tile_row["image_id"],
                    "image_sha256": tile_row["image_sha256"],
                    "image_name": tile_row["image_name"],
                    "group_id": tile_row["group_id"],
                    "tile_name": tile_row["tile_name"],
                    "tile_sha256": tile_row["tile_sha256"],
                    "mask_sha256": "e" * 64,
                    "bbox_x": "4",
                    "bbox_y": "5",
                    "bbox_w": "12",
                    "bbox_h": "13",
                    "poly": "[4,5,16,5,16,18,4,18]",
                }
                embedding = np.zeros(2048, dtype=np.float32)
                embedding[0] = 1.0
                raw_features = tuple(
                    CanonicalRawFeature(
                        proposal_index=1,
                        scale=scale,
                        predicted_iou=0.9,
                        stability_score=0.95,
                        values={name: 1.0 for name in CANONICAL_RAW_FEATURE_ORDER},
                        embedding=embedding,
                    )
                    for scale in CANONICAL_FEATURE_CROP_SCALES
                )
                identity = {
                    "proposal_id": proposal_id,
                    "proposal_sha256": proposal_id,
                    "image_id": tile_row["image_id"],
                    "image_sha256": tile_row["image_sha256"],
                    "image_name": tile_row["image_name"],
                    "group_id": tile_row["group_id"],
                    "tile_name": tile_row["tile_name"],
                    "tile_sha256": tile_row["tile_sha256"],
                    "tile_x": tile_row["x"],
                    "tile_y": tile_row["y"],
                    "mask_sha256": "e" * 64,
                }
                raw_rows = tuple(
                    {**identity, **canonical_raw_feature_csv_row(raw)}
                    for raw in raw_features
                )
                return service._GeneratedTile(
                    proposal_rows=(proposal,),
                    raw_rows=raw_rows,
                    raw_features=raw_features,
                    selection=SimpleNamespace(proposals=()),
                )

            with (
                mock.patch.object(service, "canonical_prepare_images", side_effect=lambda _image, name, locate_grid=True: _prepared(name, pixels)),
                mock.patch.object(service, "_generate_tile_records", side_effect=fake_generate),
                mock.patch.object(service, "finalize_canonical_feature", side_effect=lambda _raw, _state: {name: 1.0 for name in CANONICAL_FEATURE_ORDER}),
            ):
                self.assertEqual(
                    tuple(inspect.signature(service.infer_canonical_bundle).parameters),
                    ("images", "bundle_root", "output", "device"),
                )
                bundle_members_before = tuple(
                    sorted(path.relative_to(bundle_root) for path in bundle_root.rglob("*"))
                )
                with self.assertRaisesRegex(PublicIOError, "overlaps"):
                    service._infer_canonical_bundle_with_dependencies(
                        images,
                        bundle_root,
                        bundle_root,
                    )
                self.assertEqual(
                    tuple(
                        sorted(
                            path.relative_to(bundle_root)
                            for path in bundle_root.rglob("*")
                        )
                    ),
                    bundle_members_before,
                )
                with self.assertRaisesRegex(PublicIOError, "overlaps"):
                    service._infer_canonical_bundle_with_dependencies(
                        images,
                        bundle_root,
                        images,
                    )
                self.assertFalse((images / "predictions.csv").exists())
                result = service._infer_canonical_bundle_with_dependencies(
                    images,
                    bundle_root,
                    output,
                    device="cpu",
                    runtime_factory=lambda _assets: service.CanonicalStageRuntime(_FakeAMG(), _FakeEmbedder()),
                    bundle_verifier=lambda _path: fake_bundle,
                    predictor_loader=lambda _classifier, _imputer: object(),
                    probability_predictor=lambda _predictor, _rows: np.asarray([0.9], dtype=np.float32),
                    nms_function=lambda _boxes, _scores, _threshold: np.asarray([0], dtype=int),
                )
            self.assertTrue((output / "_ATTEMPT.json").is_file())
            self.assertEqual(result["prediction_rows"], 1)
            self.assertEqual(result["positive_rows"], 1)
            self.assertEqual(result["kept_rows"], 1)
            self.assertEqual(
                result["sam2_execution_points_per_batch"],
                CANONICAL_INFERENCE_AMG_POINTS_PER_BATCH,
            )
            self.assertEqual(result["raw_feature_archive"]["row_count"], 4)
            self.assertTrue((output / "raw_features.csv").is_file())
            archive = read_canonical_active_learning_inference(output)
            self.assertEqual(len(archive.pool_rows), 1)
            self.assertEqual(len(archive.feature_rows), 4)
            self.assertEqual(archive.bundle_sha256, fake_bundle.bundle_sha256)
            with (output / "raw_features.csv").open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(PublicIOError, "raw-feature archive"):
                read_canonical_active_learning_inference(output)
            with (output / "predictions.csv").open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), service.CANONICAL_INFERENCE_COLUMNS)
                prediction_rows = list(reader)
            self.assertEqual(sum(int(row["kept"]) for row in prediction_rows), 1)
            self.assertTrue(all(row["full_image"] == "infer01__view.ppm" for row in prediction_rows))

            empty_output = root / "stage60-empty"
            empty_output.mkdir()
            (empty_output / "_ATTEMPT.json").write_text("{}\n", encoding="ascii")
            empty_generated = service._GeneratedTile(
                proposal_rows=(),
                raw_rows=(),
                raw_features=(),
                selection=SimpleNamespace(proposals=()),
            )
            predictor = mock.Mock(side_effect=AssertionError("empty inference called predictor"))
            with (
                mock.patch.object(service, "canonical_prepare_images", side_effect=lambda _image, name, locate_grid=True: _prepared(name, pixels)),
                mock.patch.object(service, "_generate_tile_records", return_value=empty_generated),
            ):
                empty_result = service._infer_canonical_bundle_with_dependencies(
                    images,
                    bundle_root,
                    empty_output,
                    device="cpu",
                    runtime_factory=lambda _assets: service.CanonicalStageRuntime(_FakeAMG(), _FakeEmbedder()),
                    bundle_verifier=lambda _path: fake_bundle,
                    predictor_loader=lambda _classifier, _imputer: object(),
                    probability_predictor=predictor,
                    nms_function=lambda _boxes, _scores, _threshold: np.asarray([], dtype=int),
                )
            predictor.assert_not_called()
            self.assertEqual(
                (
                    empty_result["proposal_count"],
                    empty_result["prediction_rows"],
                    empty_result["positive_rows"],
                    empty_result["kept_rows"],
                ),
                (0, 0, 0, 0),
            )
            self.assertEqual(empty_result["raw_feature_archive"]["row_count"], 0)
            self.assertTrue((empty_output / "raw_features.csv").is_file())
            empty_archive = read_canonical_active_learning_inference(empty_output)
            self.assertEqual(empty_archive.pool_rows, ())
            self.assertEqual(empty_archive.feature_rows, ())
            with (empty_output / "predictions.csv").open(newline="") as handle:
                empty_reader = csv.DictReader(handle)
                self.assertEqual(tuple(empty_reader.fieldnames or ()), service.CANONICAL_INFERENCE_COLUMNS)
                self.assertEqual(list(empty_reader), [])

    def test_proposal_identity_binds_tile_position_even_for_identical_tile_bytes(self) -> None:
        from compag_curation.canonical import service
        from compag_curation.canonical.proposals import CanonicalProposal

        proposal = CanonicalProposal(
            source_index=0,
            proposal_index=0,
            mask=None,
            mask_sha256="c" * 64,
            area=1,
            bbox=(0, 0, 1, 1),
            predicted_iou=0.9,
            stability_score=0.9,
        )
        first = service._proposal_id(
            "a" * 64,
            "card_y00000x00000.jpg",
            "b" * 64,
            0,
            0,
            proposal,
        )
        second = service._proposal_id(
            "a" * 64,
            "card_y00000x00512.jpg",
            "b" * 64,
            512,
            0,
            proposal,
        )
        self.assertNotEqual(first, second)
        lite = service._proposal_id(
            "a" * 64,
            "card_y00000x00000.jpg",
            "b" * 64,
            0,
            0,
            proposal,
            profile=EFFICIENT_GPU_PROFILE,
        )
        self.assertNotEqual(first, lite)
        expected_lite_payload = (
            b"compag-efficient-sam2-tiny-proposal-v1\0"
            + ("a" * 64).encode("ascii")
            + b"\0card_y00000x00000.jpg\0"
            + ("b" * 64).encode("ascii")
            + b"\0"
            + b"0"
            + b"\0"
            + b"0"
            + b"\0"
            + ("c" * 64).encode("ascii")
            + b"\0"
            + b"0"
        )
        self.assertEqual(lite, hashlib.sha256(expected_lite_payload).hexdigest())
        self.assertEqual(
            first,
            service._proposal_id(
                "a" * 64,
                "card_y00000x00000.jpg",
                "b" * 64,
                0,
                0,
                proposal,
            ),
        )


if __name__ == "__main__":
    unittest.main()
