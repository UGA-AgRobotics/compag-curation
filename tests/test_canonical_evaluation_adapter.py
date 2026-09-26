from __future__ import annotations

import json
import io
import contextlib
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HEAVY_MODULES = {
    "cv2", "matplotlib", "numpy", "pandas", "scipy", "sklearn", "torch",
    "torchvision", "xgboost",
}


def _write_source_outputs(args, *, unexpected: bool = False) -> None:
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction = Path(args.pred_csv[0]).read_text(encoding="ascii")
    if "10,10,20,20" not in prediction or prediction.rstrip().splitlines()[-1].endswith(",[]"):
        raise AssertionError("bbox fallback prediction was not passed through unchanged")
    view = args.stage
    (output / f"summary__{view}.csv").write_text(
        "image,stage,n_gt_points,n_pred_instances,covered_points,recall_cov\n"
        f"card.jpg,{view},2,1,1,0.5\n",
        encoding="ascii",
    )
    (output / f"points_detail__{view}.csv").write_text(
        "image,gt_idx,px,py,covered_stage\ncard.jpg,0,15,15,1\n",
        encoding="ascii",
    )
    overall = {
        "stage": view,
        "cvat_label": args.cvat_label,
        "n_images": 1,
        "total_gt_points": 2,
        "total_pred_instances": 1,
        "total_covered_points": 1,
        "micro_recall_cov": 0.5,
        "micro_proposal_cov_recall": 0.5,
        "micro_al_cov_recall": 0.5,
        "micro_miss_within_margin_rate": 0.5,
        "micro_miss_in_al_topk_rate": 0.5,
        "micro_mask_hit_rate": 1.0,
        "micro_fp_rate_masks": 0.0,
        "micro_merge_rate_masks_ge2pts": 0.0,
        "micro_redundancy_mean": 0.5,
        "micro_multi_cover_rate": 0.0,
        "params": {
            "thr_xgb": args.thr_xgb,
            "det_policy": args.det_policy,
            "nms_iou": args.nms_iou,
            "al_margin": args.al_margin,
            "al_topk": args.al_topk,
        },
        "notes": {
            "definition": "A GT point is detected inside a prediction (polygon preferred; bbox fallback).",
        },
    }
    (output / f"overall__{view}.json").write_text(
        json.dumps(overall, allow_nan=False),
        encoding="ascii",
    )
    (output / f"README__{view}.md").write_text("# Evaluation\n", encoding="ascii")
    (output / f"fig_recall_by_image__{view}.png").write_bytes(b"fixture-png")
    if unexpected:
        (output / "unexpected.txt").write_text("not allowed\n", encoding="ascii")


class CanonicalEvaluationAdapterTests(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path]:
        xml = root / "points.xml"
        xml.write_text(
            '<annotations><image name="card.jpg" width="100" height="100">'
            '<points label="cj" points="15,15;80,80"/></image></annotations>\n',
            encoding="ascii",
        )
        predictions = root / "predictions.csv"
        predictions.write_text(
            "image,x,y,w,h,xgb_p,kept,poly\ncard.jpg,10,10,20,20,0.9,1,\n",
            encoding="ascii",
        )
        return xml, predictions

    def test_00_import_is_inert_and_bbox_fallback_views_are_preserved(self) -> None:
        before = set(sys.modules)
        from compag_curation.canonical import evaluation

        introduced = {name.split(".", 1)[0] for name in set(sys.modules) - before}
        self.assertFalse(introduced & HEAVY_MODULES)
        calls = []

        def runner(args) -> None:
            calls.append(args)
            print(f"[OK] Wrote: {args.out_dir}")
            _write_source_outputs(args)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml, predictions = self._inputs(root)
            output = root / "evaluation"
            with mock.patch.object(evaluation, "_load_run_eval", return_value=runner):
                visible_stdout = io.StringIO()
                with contextlib.redirect_stdout(visible_stdout):
                    result = evaluation.evaluate_canonical_points(xml, (predictions,), output)
            self.assertEqual(visible_stdout.getvalue(), "")
            self.assertEqual([args.stage for args in calls], ["all", "kept", "xgb"])
            self.assertTrue(all(args.det_policy == "xgb" and not args.use_p_fused for args in calls))
            self.assertTrue(all(args.thr_xgb == args.nms_iou == 0.5 for args in calls))
            self.assertEqual(result["geometry_policy"], "polygon-preferred-bbox-fallback")
            self.assertEqual(result["profile"], "canonical-xgb-recall-cpu-v1")
            self.assertEqual(result["annotation_adapter"], "DIRECT_PINNED_CVAT_XML")
            self.assertEqual(result["views"], ["all", "kept", "xgb"])
            self.assertEqual(result["stages"]["xgb"]["micro_recall_cov"], 0.5)
            self.assertEqual(
                result["inputs"]["files"],
                [
                    {
                        "role": "CVAT points XML",
                        "basename": xml.name,
                        "sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
                        "size_bytes": xml.stat().st_size,
                    },
                    {
                        "role": "prediction CSV 1",
                        "basename": predictions.name,
                        "sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                        "size_bytes": predictions.stat().st_size,
                    },
                ],
            )
            self.assertEqual(
                result["parameters"],
                {
                    "selected_images": [],
                    "cvat_label": "cj",
                    "max_area_fraction": None,
                    "al_margin": 0.2,
                    "al_topk": 50,
                },
            )
            self.assertEqual(
                result["source_stream_policy"],
                "CAPTURED_BOUNDED_NOT_RETAINED_PRIVATE_LOCATORS",
            )
            self.assertNotIn(str(root), json.dumps(result, sort_keys=True))
            self.assertTrue((output / "CANONICAL_EVALUATION.json").is_file())
            self.assertEqual(
                json.loads((output / "CANONICAL_EVALUATION.json").read_text(encoding="ascii")),
                result,
            )

    def test_canonical_json_is_normalized_ephemerally_and_original_is_bound(self) -> None:
        from compag_curation.canonical import evaluation
        from compag_curation.public_io import canonical_json_bytes

        seen_annotation_paths: list[Path] = []

        def runner(args) -> None:
            annotation = Path(args.cvat_xml)
            seen_annotation_paths.append(annotation)
            text = annotation.read_text(encoding="utf-8")
            self.assertIn('<image id="0" name="card.jpg" width="100" height="100">', text)
            self.assertIn('label="cj" points="15,15"', text)
            _write_source_outputs(args)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _xml, predictions = self._inputs(root)
            annotations = root / "points.json"
            annotations.write_bytes(
                canonical_json_bytes(
                    {
                        "schema": "compag-curation-canonical-point-annotations/v1",
                        "status": "PASS",
                        "source_filename": "points.xml",
                        "source_sha256": "a" * 64,
                        "rows": [
                            {
                                "image": "card.jpg",
                                "label": "cj",
                                "x": 15.0,
                                "y": 15.0,
                                "width": 100,
                                "height": 100,
                            }
                        ],
                    },
                )
            )
            output = root / "evaluation-json"
            with mock.patch.object(evaluation, "_load_run_eval", return_value=runner):
                result = evaluation.evaluate_canonical_points(
                    annotations,
                    (predictions,),
                    output,
                    images=("card.jpg",),
                    views=("xgb",),
                )
            self.assertEqual(result["annotation_adapter"], "CANONICAL_JSON_TO_EPHEMERAL_CVAT_XML")
            self.assertEqual(result["parameters"]["selected_images"], ["card.jpg"])
            self.assertEqual(result["inputs"]["files"][0]["role"], "canonical point JSON")
            self.assertEqual(result["inputs"]["files"][0]["basename"], "points.json")
            self.assertEqual(len(seen_annotation_paths), 1)
            self.assertNotEqual(seen_annotation_paths[0], annotations)
            self.assertFalse(seen_annotation_paths[0].exists())

    def test_unexpected_output_fails_closure_without_publishing(self) -> None:
        from compag_curation.canonical import evaluation
        from compag_curation.public_io import PublicIOError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml, predictions = self._inputs(root)
            output = root / "evaluation"

            def runner(args) -> None:
                _write_source_outputs(args, unexpected=True)

            with mock.patch.object(evaluation, "_load_run_eval", return_value=runner):
                with self.assertRaisesRegex(PublicIOError, "unexpected file"):
                    evaluation.evaluate_canonical_points(
                        xml,
                        (predictions,),
                        output,
                        views=("xgb",),
                    )
            self.assertFalse(output.exists())
            self.assertEqual(len(list(root.glob(".evaluation.canonical-eval.*"))), 1)

    def test_existing_output_is_no_clobber_and_runner_stays_inert(self) -> None:
        from compag_curation.canonical import evaluation
        from compag_curation.public_io import PublicIOError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml, predictions = self._inputs(root)
            output = root / "evaluation"
            output.mkdir()
            marker = output / "private.txt"
            marker.write_text("unchanged\n", encoding="ascii")
            with mock.patch.object(evaluation, "_load_run_eval") as load_runner:
                with self.assertRaisesRegex(PublicIOError, "already exists"):
                    evaluation.evaluate_canonical_points(xml, (predictions,), output)
            load_runner.assert_not_called()
            self.assertEqual(marker.read_text(encoding="ascii"), "unchanged\n")

    def test_stage_sixty_passes_portable_image_names_to_canonical_evaluator(self) -> None:
        from compag_curation import public_pipeline
        from compag_curation.canonical import evaluation
        from compag_curation.public_io import canonical_json_bytes

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "stage60"
            output.mkdir()
            (output / "inference_result.json").write_bytes(
                canonical_json_bytes({"schema": "fixture-inference/v1", "status": "PASS"})
            )
            config = mock.Mock(
                is_canonical=True,
                profile="canonical-xgb-recall-gpu-v1",
                annotations=root / "points.json",
                inference_images=root / "inference-images",
            )
            inspection = {
                "inference_images": [
                    {"filename": "heldout-card.jpg", "width": 512, "height": 512}
                ]
            }
            dependencies = {"schema": "fixture-dependencies/v1"}
            with (
                mock.patch.object(
                    public_pipeline,
                    "_fresh_canonical_inference",
                    return_value={"status": "PASS"},
                ) as fresh,
                mock.patch.object(public_pipeline, "inspect_public_data", return_value=inspection),
                mock.patch.object(evaluation, "evaluate_canonical_points", return_value={"status": "PASS"}) as evaluate,
            ):
                public_pipeline._stage_sixty(
                    config,
                    root / "bundle",
                    output,
                    dependencies,
                )
            fresh.assert_called_once_with(
                config,
                root / "bundle",
                output,
                dependencies,
            )
            self.assertEqual(evaluate.call_args.kwargs["images"], ("heldout-card.jpg",))
            self.assertEqual(
                evaluate.call_args.kwargs["profile"],
                "canonical-xgb-recall-gpu-v1",
            )

    def test_gpu_profile_is_preserved_in_published_evaluation(self) -> None:
        from compag_curation.canonical import evaluation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml, predictions = self._inputs(root)
            output = root / "gpu-evaluation"
            with mock.patch.object(
                evaluation,
                "_load_run_eval",
                return_value=_write_source_outputs,
            ):
                result = evaluation.evaluate_canonical_points(
                    xml,
                    (predictions,),
                    output,
                    views=("xgb",),
                    profile="canonical-xgb-recall-gpu-v1",
                )
            self.assertEqual(result["profile"], "canonical-xgb-recall-gpu-v1")
            self.assertEqual(
                json.loads(
                    (output / "CANONICAL_EVALUATION.json").read_text(encoding="ascii")
                )["profile"],
                "canonical-xgb-recall-gpu-v1",
            )


if __name__ == "__main__":
    unittest.main()
