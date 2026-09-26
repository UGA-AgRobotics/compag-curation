"""Small contract regressions for the new public project path."""
from __future__ import annotations

import csv
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class ContinuationContracts(unittest.TestCase):
    def test_laptop_guide_forwards_cumulative_parent_and_resume_checkpoint(self):
        from compag_curation.only_codes import guide

        answers = iter(["22", "workspace", "project_r2", "review_state", "inference",
                        "native_model", "reviewed", "prior_snapshot", "prior_set",
                        "new_round", "REVIEW COMPLETE", "17", "model_set", "prepared",
                        "assets", "next_inference", "boxes", "tile_checkpoint", "0"])
        with mock.patch.object(guide, "_recent_project", return_value=None), \
                mock.patch.object(guide, "_run_r92", return_value=0) as delegated, \
                mock.patch("builtins.input", side_effect=lambda prompt: next(answers)), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(guide.run_guide(), 0)
        first = delegated.call_args_list[0].args[0]
        second = delegated.call_args_list[1].args[0]
        self.assertEqual(first[0], "complete-xgb-round")
        self.assertEqual(first[first.index("--scope")+1], "reviewed")
        self.assertNotIn("--selection", first)
        self.assertEqual(first[first.index("--parent-snapshot")+1], "prior_snapshot")
        self.assertEqual(first[first.index("--parent-model-set")+1], "prior_set")
        self.assertEqual(second[0], "infer-model-set")
        self.assertEqual(second[second.index("--checkpoint-dir")+1], "tile_checkpoint")
        self.assertEqual(second[second.index("--yolo-boxes")+1], "boxes")

    def test_ordered_review_skip_and_uncertain_factor(self):
        from compag_curation.r92_project_model import _review_events

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "review.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image","id","action","human_label",
                                                           "review_weight","timestamp"])
                writer.writeheader()
                writer.writerows([
                    {"image":"tile.jpg","id":"1","action":"accept","human_label":"1","review_weight":"1","timestamp":"10"},
                    {"image":"tile.jpg","id":"1","action":"skip","human_label":"","review_weight":"0","timestamp":"11"},
                    {"image":"tile.jpg","id":"2","action":"sus_flip","human_label":"0","review_weight":"0.4","timestamp":"12"},
                ])
            events = _review_events(path)
            self.assertEqual(events[("tile.jpg","1")]["action"], "skip")
            self.assertIsNone(events[("tile.jpg","1")]["label"])
            self.assertEqual(events[("tile.jpg","2")]["review_factor"], .4)

    def test_original_mask_rle_preserves_nonrectangular_pixels(self):
        import numpy as np
        from compag_curation.r92_project_model import _rle

        mask = np.asarray([[0,1,0],[0,1,1],[0,0,0]], dtype=np.uint8)
        encoded = _rle(mask)
        runs = []
        bit = 0
        for count in encoded["counts"]:
            runs.extend([bit]*count)
            bit = 1-bit
        self.assertEqual(encoded["size"], [3,3])
        self.assertTrue(np.array_equal(np.asarray(runs,dtype=np.uint8).reshape(3,3,order="F"),mask))

    def test_yolo_complete_tile_split_and_box_guards(self):
        import cv2
        import numpy as np
        from compag_curation.r92_yolo import prepare_yolo_dataset, verify_yolo_dataset

        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); project=root/"project"; (project/"tiles").mkdir(parents=True)
            images=[]; units=[]; hashes={}
            for card, role in (("a","train"),("b","val")):
                name=f"{card}_y00000x00000.jpg"
                cv2.imwrite(str(project/"tiles"/name),np.zeros((512,512,3),dtype=np.uint8))
                digest=hashlib.sha256((project/"tiles"/name).read_bytes()).hexdigest()
                hashes["tiles/"+name]=digest
                images.append({"file_name":name,"meta":{"orig_file_name":card}})
                units.append({"tile":name,"tile_sha256":digest,"split_role":role,
                    "complete_for":"all_CJ_targets_in_512x512_tile","certified_by":"fixture",
                    "targets":[] if card=="b" else [{"physical_object_id":"a1","bbox_xyxy":[10,10,30,30]}]})
            (project/"tiled_coco.json").write_text(json.dumps({"images":images}))
            (project/"PROJECT_MANIFEST.json").write_text(json.dumps({"schema":"compag-r92-full-project-manifest/v1",
                "sha256_by_relative_path":hashes}))
            receipt={"schema":"compag-r92-image-project/v2","tile_count":2,
                "project_manifest_sha256":hashlib.sha256((project/"PROJECT_MANIFEST.json").read_bytes()).hexdigest()}
            (project/"PROJECT_RECEIPT.json").write_text(json.dumps(receipt))
            qa={"schema":"compag-r92-yolo-complete-tile-qa/v1","target_class":"CJ",
                "project_receipt_sha256":hashlib.sha256((project/"PROJECT_RECEIPT.json").read_bytes()).hexdigest(),
                "annotation_origin":"test_fixture","units":units}
            path=root/"qa.json"; path.write_text(json.dumps(qa))
            result=prepare_yolo_dataset(project,path,root/"dataset")
            self.assertEqual(result["empty_target_tile_count"],1)
            verify_yolo_dataset(root/"dataset")
            qa["units"][0].pop("complete_for")
            path.write_text(json.dumps(qa))
            with self.assertRaisesRegex(ValueError,"not explicitly complete"):
                prepare_yolo_dataset(project,path,root/"rejected")

    def test_yolo_match_keeps_missing_distinct(self):
        from compag_curation.r92_yolo import matched_box

        boxes=[{"bbox_xyxy":[10,10,30,30],"confidence":.91}]
        self.assertIsNone(matched_box((100,100,20,20),boxes))
        self.assertEqual(matched_box((10,10,20,20),boxes)["confidence"],.91)

    def test_paper_hybrid_review_selection_uses_effective_margin(self):
        from compag_curation.r92_project_model import select_review
        from compag_curation.r92_yolo import PAPER_HYBRID_POLICY

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inference = root / "inference"
            scored = inference / "scores" / "detections.csv"
            scored.parent.mkdir(parents=True)
            with scored.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image", "id", "xgb_p", "threshold",
                                                          "p_ui", "final_threshold"])
                writer.writeheader()
                writer.writerows([
                    {"image": "tile.jpg", "id": "1", "xgb_p": ".50", "threshold": ".50",
                     "p_ui": ".95", "final_threshold": ".50"},
                    {"image": "tile.jpg", "id": "2", "xgb_p": ".10", "threshold": ".50",
                     "p_ui": ".51", "final_threshold": ".50"},
                    {"image": "tile.jpg", "id": "3", "xgb_p": ".90", "threshold": ".50",
                     "p_ui": ".81", "final_threshold": ".50"},
                    {"image": "tile.jpg", "id": "4", "xgb_p": ".25", "threshold": ".50",
                     "p_ui": ".68", "final_threshold": ".50"},
                ])
            (inference / "FULL_INFERENCE_RECEIPT.json").write_text(json.dumps({
                "schema": "compag-r92-infer-full/v1", "status": "PASS",
                "expected_tile_count": 1, "processed_tile_count": 1, "candidate_count": 4,
                "score_receipt": {"detections_sha256": hashlib.sha256(scored.read_bytes()).hexdigest(),
                                  "fusion_policy": PAPER_HYBRID_POLICY}}))
            selection = root / "selection.csv"
            result = select_review(inference, selection, count=50)
            self.assertEqual(result["selected_count"], 2)
            self.assertEqual(result["al_margin"], .20)
            with selection.open(newline="", encoding="utf-8") as stream:
                self.assertEqual([row["id"] for row in csv.DictReader(stream)], ["2", "4"])

    def test_paper_hybrid_scoring_keeps_decision_review_score_and_fusion_distinct(self):
        from compag_curation.r92_yolo import apply_fusion

        tile = "card_y00000x00000.jpg"
        fields = ["image", "id", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                  "xgb_p", "threshold", "xgb_pred"]
        candidates = [
            # Both scores support CJ: use confidence-normalized convex fusion.
            {"image": tile, "id": "0", "bbox_x": "10", "bbox_y": "10",
             "bbox_w": "20", "bbox_h": "20", "xgb_p": "0.4",
             "threshold": "0.5", "xgb_pred": "0"},
            # XGB-high/YOLO-low: beta=0.8 must outweigh normalized YOLO weight.
            {"image": tile, "id": "1", "bbox_x": "40", "bbox_y": "40",
             "bbox_w": "20", "bbox_h": "20", "xgb_p": "0.9",
             "threshold": "0.5", "xgb_pred": "1"},
            # No matching box: reject even when raw XGBoost strongly favors CJ.
            {"image": tile, "id": "2", "bbox_x": "100", "bbox_y": "100",
             "bbox_w": "20", "bbox_h": "20", "xgb_p": "0.9",
             "threshold": "0.5", "xgb_pred": "1"},
            # A geometric match below the YOLO confidence threshold is invalid support.
            {"image": tile, "id": "3", "bbox_x": "70", "bbox_y": "70",
             "bbox_w": "20", "bbox_h": "20", "xgb_p": "0.75",
             "threshold": "0.5", "xgb_pred": "1"},
            # A valid lower-IoU box must win over an invalid exact box.
            {"image": tile, "id": "4", "bbox_x": "130", "bbox_y": "130",
             "bbox_w": "20", "bbox_h": "20", "xgb_p": "0.4",
             "threshold": "0.5", "xgb_pred": "0"},
        ]
        boxes = {tile: [
            {"bbox_xyxy": [10, 10, 30, 30], "confidence": 0.8},
            {"bbox_xyxy": [40, 40, 60, 60], "confidence": 0.3},
            {"bbox_xyxy": [70, 70, 90, 90], "confidence": 0.15},
            {"bbox_xyxy": [130, 130, 150, 150], "confidence": 0.15},
            {"bbox_xyxy": [131, 131, 150, 150], "confidence": 0.8},
        ]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for mode in ("fusion", "both"):
                with self.subTest(mode=mode):
                    scored = root / f"{mode}.csv"
                    receipt_path = root / f"{mode}.json"
                    with scored.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(candidates)
                    receipt_path.write_text(json.dumps({"threshold": 0.5}))
                    receipt = apply_fusion(scored, receipt_path, boxes, mode)
                    with scored.open(newline="", encoding="utf-8") as stream:
                        rows = {row["id"]: row for row in csv.DictReader(stream)}

                    self.assertEqual(receipt["fusion_policy"], "paper_hybrid_reject_missing_v1")
                    self.assertEqual(receipt["detections_sha256"],
                                     hashlib.sha256(scored.read_bytes()).hexdigest())
                    self.assertEqual((rows["0"]["final_pred"], rows["0"]["yolo_match"]), ("1", "1"))
                    expected_normalized = (0.8 / (0.8 + 0.4)) * 0.8 + (0.4 / (0.8 + 0.4)) * 0.4
                    self.assertAlmostEqual(float(rows["0"]["p_fused"]), expected_normalized)
                    self.assertAlmostEqual(float(rows["0"]["p_ui"]), expected_normalized)
                    self.assertEqual((rows["1"]["final_pred"], rows["1"]["yolo_match"]), ("0", "1"))
                    self.assertAlmostEqual(float(rows["1"]["p_fused"]), 0.8 * 0.3 + 0.2 * 0.9)
                    self.assertAlmostEqual(float(rows["1"]["p_ui"]), 0.42)
                    for candidate_id, raw_xgb, matched in (("2", 0.9, "0"), ("3", 0.75, "1")):
                        row = rows[candidate_id]
                        self.assertEqual((row["final_pred"], row["yolo_match"]), ("0", matched))
                        self.assertEqual(row["fused_p"], "")
                        self.assertEqual(row["p_fused"], "")
                        self.assertAlmostEqual(float(row["p_ui"]), raw_xgb)
                        self.assertEqual(float(row["final_threshold"]), 0.5)
                    self.assertEqual(rows["4"]["yolo_valid"], "1")
                    self.assertEqual(rows["4"]["yolo_conf"], repr(0.8))
                    self.assertEqual(rows["4"]["final_pred"], "1")

    def test_paper_and_legacy_fusion_receipts_reopen_with_bound_decisions(self):
        import cv2
        import numpy as np
        from compag_curation.r92_review import make_session

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tiles = root / "tiles"
            tiles.mkdir()
            tile = "card_y00000x00000.jpg"
            self.assertTrue(cv2.imwrite(str(tiles / tile), np.zeros((512, 512, 3), dtype=np.uint8)))
            base_row = {"img_folder": "card", "image": tile, "id": "0", "xgb_p": "0.9",
                        "xgb_pred": "1", "threshold": "0.5", "poly": "",
                        "bbox_x": "10", "bbox_y": "10", "bbox_w": "20", "bbox_h": "20",
                        "yolo_conf": "0.0", "yolo_iou": "0.0", "final_threshold": "0.5"}
            policies = (
                ("retained_Only_codes_review_hybrid_ignore_missing_v1", "ignore", "1",
                 {"p_fused": "0.9"}),
                ("paper_hybrid_reject_missing_v1", "reject", "0",
                 {"yolo_valid": "0", "p_ui": "0.9", "fused_p": "", "p_fused": ""}),
            )
            for policy, missing, expected_pred, extra in policies:
                with self.subTest(policy=policy):
                    case = root / missing
                    scores = case / "scores"
                    scores.mkdir(parents=True)
                    scored = scores / "detections.csv"
                    row = {**base_row, "final_pred": expected_pred, **extra}
                    with scored.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerow(row)
                    (scores / "SCORE_RECEIPT.json").write_text(json.dumps({
                        "detections_sha256": hashlib.sha256(scored.read_bytes()).hexdigest(),
                        "execution_variant": "cpu_tabular_xgboost_2.1.1", "threshold": 0.5,
                        "model_sha256": "fixture-model", "yolo_mode": "fusion",
                        "fusion_policy": policy}))

                    session = make_session(scored, tiles, case / "review")
                    self.assertEqual(session.params.det_missing, missing)
                    item, = session.items("card")
                    self.assertEqual(item["pred"], int(expected_pred))
                    self.assertAlmostEqual(item["p_ui"], 0.9)
                    self.assertEqual(item["p_fused"], None if missing == "reject" else 0.9)

    def test_external_segment_checkpoint_boxes_and_model_set_binding(self):
        """A segmentation checkpoint can feed fusion, with tampering rejected."""
        import numpy as np
        from compag_curation import r92_model_set, r92_yolo

        class SegmentModel:
            task = "segment"
            names = {0: "CJ"}

            def __init__(self, path):
                self.path = path

            def predict(self, **kwargs):
                self.options = kwargs
                box = SimpleNamespace(cls=[0], xyxy=np.array([[10., 20., 30., 40.]]), conf=[.9])
                return [SimpleNamespace(boxes=[box], masks=[object()])]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / "segment-package"
            (package / "weights").mkdir(parents=True)
            (package / "inference").mkdir()
            checkpoint = package / "weights" / "best.pt"
            checkpoint.write_bytes(b"fixture-segment-checkpoint")
            weight_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            (package / "MODEL_MANIFEST.json").write_text(json.dumps({
                "checkpoint": "weights/best.pt", "checkpoint_sha256": weight_sha,
                "task": "instance segmentation", "class_mapping": {"0": "CJ"},
                "verified_runtime": {"ultralytics": "8.4.26"}}))
            config_path = package / "inference" / "inference_config.json"
            config_path.write_text(json.dumps(r92_yolo.SEGMENT_CONFIG))
            project = root / "prepared"
            (project / "tiles").mkdir(parents=True)
            tile = project / "tiles" / "card_y00000x00000.jpg"
            tile.write_bytes(b"fixture-tile")
            (project / "tiled_coco.json").write_text(json.dumps({"images": [{"file_name": tile.name}]}))
            (project / "PROJECT_MANIFEST.json").write_text(json.dumps({
                "sha256_by_relative_path": {"tiles/" + tile.name: hashlib.sha256(tile.read_bytes()).hexdigest()}}))
            (project / "PROJECT_RECEIPT.json").write_text(json.dumps({
                "schema": "compag-r92-image-project/v2", "tile_count": 1,
                "project_manifest_sha256": hashlib.sha256((project / "PROJECT_MANIFEST.json").read_bytes()).hexdigest()}))
            optional_backend = SimpleNamespace(__version__="8.4.26", YOLO=SegmentModel)
            with mock.patch.dict(sys.modules, {"ultralytics": optional_backend}):
                attestation = root / "attestation.json"
                r92_yolo.verify_checkpoint_record(checkpoint, weight_sha, attestation,
                                                  model_package=package)
                boxes_path = root / "boxes.json"
                result = r92_yolo.precompute_boxes(project, checkpoint, weight_sha, boxes_path,
                                                   model_package=package)
            self.assertEqual(result["box_count"], 1)
            self.assertEqual(r92_yolo.load_precomputed_boxes(boxes_path, project, weight_sha)[tile.name][0]["confidence"], .9)

            workspace = root / "workspace"
            xgb = workspace / "xgb_model"
            xgb.mkdir(parents=True)
            snapshot = workspace / "training_snapshot.json"
            snapshot.write_text(json.dumps({"review_origin": "test_fixture"}))
            native = workspace / "native.zip"
            native.write_bytes(b"fixture-native")
            (xgb / "PROJECT_MODEL_MANIFEST.json").write_text(json.dumps({
                "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "native_model_sha256": hashlib.sha256(native.read_bytes()).hexdigest()}))
            with mock.patch.object(r92_model_set, "verify_snapshot"), \
                    mock.patch.object(r92_model_set, "load_project_model"):
                selected = r92_model_set.activate_model_set(
                    workspace, "first_round_external", snapshot, xgb, native,
                    workspace / "selected_model_set", yolo_mode="fusion",
                    yolo_checkpoint=checkpoint, yolo_sha256=weight_sha,
                    yolo_attestation=attestation, yolo_external=True)
                self.assertEqual(selected["yolo_status"], "YOLO_EXTERNAL_ATTESTED")
                r92_model_set.verify_model_set(workspace / "selected_model_set")
                config_path.write_text(json.dumps({**r92_yolo.SEGMENT_CONFIG, "confidence": .25}))
                with self.assertRaisesRegex(ValueError, "inference settings differ"):
                    r92_yolo.load_precomputed_boxes(boxes_path, project, weight_sha)
                with self.assertRaisesRegex(ValueError, "inference settings differ"):
                    r92_model_set.verify_model_set(workspace / "selected_model_set")


if __name__ == "__main__":
    unittest.main()
