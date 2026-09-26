"""Regression tests for the 1.9.1 remaining-fix set (relocated inputs, state commit, resolver).

Synthetic fixtures only, with a deterministic embedder: this is an I/O and state contract test, not
a ResNet50 acceptance test.  The scientific loops themselves are covered by the parity kit.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


def _has(*names: str) -> bool:
    return all(importlib.util.find_spec(n) is not None for n in names)


HAVE_SCI = _has("numpy", "pandas", "cv2", "joblib")
IN_HEAVY_CHILD = os.environ.get("ONLY_CODES_HEAVY_CHILD") == "1"
heavy = unittest.skipUnless(IN_HEAVY_CHILD, "runs in the isolated scientific child process")

SCALES = 3  # SCALES_TRAIN = [0.85, 1.00, 1.15]


class InputResolverTests(unittest.TestCase):
    """Rule precedence, single application, and the explicit refusal of unsupported paths."""

    def _resolver(self, base, rules, reader="pack"):
        from compag_curation.only_codes.inputs import resolver_for
        from compag_curation.only_codes.state import PathMapper

        return resolver_for(reader, base, PathMapper(rules), record_identity=False)

    def test_longest_component_rule_applied_once(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "n2/Clean/tiles").mkdir(parents=True)
            (t / "n2/Clean/tiles/a.jpg").write_bytes(b"x")
            r = self._resolver(t / "base", [("/old", str(t / "n1")), ("/old/Clean", str(t / "n2/Clean"))])
            got = r.resolve("/old/Clean/tiles/a.jpg")
            self.assertTrue(got.ok)
            self.assertEqual(got.strategy, "MAPPED_BY_EXPLICIT_RULE")
            self.assertEqual(got.rule["old_prefix"], "/old/Clean")
            self.assertEqual(got.resolved, str(t / "n2/Clean/tiles/a.jpg"))

    def test_substring_prefix_is_not_a_match(self):
        with tempfile.TemporaryDirectory() as t:
            r = self._resolver(Path(t), [("/old", "/new")])
            got = r.resolve("/oldish/a.jpg")
            self.assertFalse(got.ok)
            self.assertIsNone(got.rule)
            self.assertEqual([c["strategy"] for c in got.candidates], ["RECORDED_ABSOLUTE_PATH"])

    def test_recorded_path_used_when_no_rule_and_readable(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "real").mkdir()
            (t / "real/a.jpg").write_bytes(b"x")
            r = self._resolver(t / "base", [])
            got = r.resolve(str(t / "real/a.jpg"))
            self.assertTrue(got.ok)
            self.assertEqual(got.strategy, "RECORDED_ABSOLUTE_PATH")

    def test_missing_mapped_destination_falls_back_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "old").mkdir(); (t / "old/a.jpg").write_bytes(b"x")
            r = self._resolver(t / "base", [(str(t / "old"), str(t / "gone"))])
            got = r.resolve(str(t / "old/a.jpg"))
            self.assertTrue(got.ok)                                   # the recorded file still exists
            self.assertEqual(got.strategy, "RECORDED_ABSOLUTE_PATH")
            self.assertEqual(got.candidates[0]["strategy"], "MAPPED_BY_EXPLICIT_RULE")
            self.assertFalse(got.candidates[0]["exists"])
            r2 = self._resolver(t / "base", [(str(t / "old"), str(t / "gone"))])
            missing = r2.resolve(str(t / "old/never.jpg"))
            self.assertEqual(missing.outcome, "NOT_FOUND")

    def test_explicit_rule_wins_when_both_sides_exist(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for d in ("old", "new"):
                (t / d).mkdir(); (t / d / "a.jpg").write_bytes(d.encode())
            r = self._resolver(t / "base", [(str(t / "old"), str(t / "new"))])
            got = r.resolve(str(t / "old/a.jpg"))
            self.assertEqual(got.resolved, str(t / "new/a.jpg"))
            self.assertEqual(got.strategy, "MAPPED_BY_EXPLICIT_RULE")

    def test_windows_path_rejected_unless_mapped(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "win").mkdir(); (t / "win/a.jpg").write_bytes(b"x")
            r = self._resolver(t / "base", [])
            got = r.resolve(r"C:\Data\tiles\a.jpg")
            self.assertEqual(got.outcome, "UNSUPPORTED_WINDOWS_PATH")
            r2 = self._resolver(t / "base", [("C:/Data/tiles", str(t / "win"))])
            ok = r2.resolve(r"C:\Data\tiles\a.jpg")
            self.assertTrue(ok.ok)
            self.assertEqual(ok.resolved, str(t / "win/a.jpg"))

    def test_features_reader_keeps_the_reference_basename_join(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "tiles").mkdir(); (t / "tiles/a.jpg").write_bytes(b"x")
            r = self._resolver(t / "tiles", [], reader="features")
            got = r.resolve("/gone/elsewhere/a.jpg")
            self.assertTrue(got.ok)
            self.assertEqual(got.strategy, "LEGACY_BASENAME_JOIN")
            # the pack reader must NOT pick a same-named file from another directory
            p = self._resolver(t / "tiles", [])
            self.assertFalse(p.resolve("/gone/elsewhere/a.jpg").ok)


@unittest.skipUnless(HAVE_SCI, "numpy/pandas/cv2/joblib required")
@heavy
class RelocatedInputPackTests(unittest.TestCase):
    """F01/E07: relocated absolute COCO paths, and no completion for a failed input."""

    def setUp(self):
        import numpy as np

        self.np = np

    # -- fixture ---------------------------------------------------------
    def _fixture(self, root: Path, *, file_name_for_image, n_images=1, extra_ann_without_patch=False):
        import cv2
        import numpy as np

        tiles = root / "this_machine/img_tiles_512"
        tiles.mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        for i in range(1, n_images + 1):
            name = f"t{i}.jpg"
            img = np.full((40, 40, 3), (40, 60, 200), np.uint8)
            img[10:30, 10:30] = (30, 40, 150)
            cv2.imwrite(str(tiles / name), img)
            images.append({"id": i, "file_name": file_name_for_image(i, name, tiles), "width": 40, "height": 40,
                           "meta": {"orig_image_id": 100 + i}})
            annotations.append({"id": 10 + i, "image_id": i, "category_id": 1, "iscrowd": 0,
                                "segmentation": [[5, 5, 30, 5, 30, 30, 5, 30]], "bbox": [5, 5, 25, 25], "area": 625})
        if extra_ann_without_patch:
            annotations.append({"id": 999, "image_id": 1, "category_id": 1, "iscrowd": 0,
                                "segmentation": [[0, 0, 0, 0, 0, 0]], "bbox": [0, 0, 0, 0], "area": 0})
        coco = {"images": images, "categories": [{"id": 1, "name": "cj"}], "annotations": annotations}
        orig = root / "orig"
        (orig / "Shared/PC_codes/annotations").mkdir(parents=True, exist_ok=True)
        (orig / "Shared/PC_codes/annotations/tiles.json").write_text(json.dumps(coco))
        splits = orig / "Shared/PC_codes/npy_npz_csv/tiles_512/splits"
        splits.mkdir(parents=True, exist_ok=True)
        (splits / "used_img_ids_train.json").write_text(json.dumps(list(range(1, n_images + 1))))
        return orig, tiles

    def _fake_embedder(self):
        import numpy as np

        class FakeEmbedder:
            fallback_count = 0

            def embed(self, patch):
                v = np.zeros(8, np.float32)
                v[0], v[1], v[2] = float(patch.shape[0]), float(patch.shape[1]), 1.0
                return v

        return lambda *a, **k: FakeEmbedder()

    def _run(self, orig, overlay, tiles, rules=()):
        from compag_curation.only_codes.pack import run_pack
        from compag_curation.only_codes.state import LegacyState, PathMapper

        return run_pack(state=LegacyState(orig, overlay),
                        coco_json=orig / "Shared/PC_codes/annotations/tiles.json", images_dir=tiles,
                        art_rel="Shared/PC_codes/npy_npz_csv/tiles_512",
                        split_rel="Shared/PC_codes/npy_npz_csv/tiles_512/splits", fold_tag="train_foldA",
                        resnet50_weights="unused", device="cpu", mapper=PathMapper(list(rules)),
                        embedder_factory=self._fake_embedder())

    def _emb(self, overlay):
        p = overlay / "Shared/PC_codes/npy_npz_csv/tiles_512/stage1_embeddings/train_foldA/embeddings_train_pos.npy"
        return self.np.load(p) if p.exists() else self.np.zeros((0, 0), "float32")

    def _used(self, overlay):
        p = overlay / "Shared/PC_codes/npy_npz_csv/tiles_512/stage2_proto/train_foldA/used_img_ids.json"
        return json.loads(p.read_text()) if p.exists() else []

    # -- tests -----------------------------------------------------------
    def test_mapped_absolute_path_matches_the_relative_control(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            ctrl_root = t / "ctrl"
            orig_c, tiles_c = self._fixture(ctrl_root, file_name_for_image=lambda i, n, d: n)
            self._run(orig_c, ctrl_root / "ov", tiles_c)
            control = self._emb(ctrl_root / "ov")

            rel_root = t / "reloc"
            orig_r, tiles_r = self._fixture(
                rel_root, file_name_for_image=lambda i, n, d: str(rel_root / "old_machine/img_tiles_512" / n))
            rep = self._run(orig_r, rel_root / "ov", tiles_r,
                            rules=[(str(rel_root / "old_machine/img_tiles_512"), str(tiles_r))])
            mapped = self._emb(rel_root / "ov")

            self.assertEqual(control.shape[0], SCALES)
            self.assertEqual(mapped.shape, control.shape)
            self.assertEqual(mapped.tobytes(), control.tobytes())
            self.assertEqual(self._used(rel_root / "ov"), [1])
            st1 = rep["stages"]["stage1"]
            self.assertEqual(st1["contributed_ids"], [1])
            self.assertEqual(st1["input_resolution"]["by_strategy"], {"MAPPED_BY_EXPLICIT_RULE": 1})

    def test_readable_absolute_path_without_mapping_still_works(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            orig, tiles = self._fixture(t, file_name_for_image=lambda i, n, d: str(d / n))
            rep = self._run(orig, t / "ov", tiles)
            self.assertEqual(self._emb(t / "ov").shape[0], SCALES)
            self.assertEqual(rep["stages"]["stage1"]["input_resolution"]["by_strategy"],
                             {"RECORDED_ABSOLUTE_PATH": 1})

    def test_failed_input_is_not_completed_and_resume_produces_the_clean_result_once(self):
        from compag_curation.only_codes.pack import PackInputError

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            orig, tiles = self._fixture(t, file_name_for_image=lambda i, n, d: str(t / "old_machine/tiles" / n))
            ov = t / "ov"
            with self.assertRaises(PackInputError) as ctx:
                self._run(orig, ov, tiles)                      # no mapping: the image cannot be read
            rep = ctx.exception.report
            self.assertEqual(rep["stages"]["stage1"], "ABORTED_UNREADABLE_INPUTS")
            self.assertFalse(rep["published"])
            self.assertEqual(self._used(ov), [])                # nothing marked as used
            self.assertEqual(self._emb(ov).shape[0], 0)
            pack_p = ov / "Shared/PC_codes/npy_npz_csv/tiles_512/stage2_proto/train_foldA/foldsafe_pack.joblib"
            self.assertFalse(pack_p.exists())                   # nothing published

            rep2 = self._run(orig, ov, tiles, rules=[(str(t / "old_machine/tiles"), str(tiles))])
            self.assertEqual(self._emb(ov).shape[0], SCALES)    # exactly the clean-run content, once
            self.assertEqual(self._used(ov), [1])
            self.assertTrue(rep2["published"])

            ctrl = t / "ctrl"
            orig_c, tiles_c = self._fixture(ctrl, file_name_for_image=lambda i, n, d: n)
            self._run(orig_c, ctrl / "ov", tiles_c)
            self.assertEqual(self._emb(ov).tobytes(), self._emb(ctrl / "ov").tobytes())

    def test_mixed_batch_aborts_and_keeps_previous_successful_state(self):
        from compag_curation.only_codes.pack import PackInputError

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            # round 1: one good image
            orig, tiles = self._fixture(t, file_name_for_image=lambda i, n, d: n, n_images=1)
            ov = t / "ov"
            self._run(orig, ov, tiles)
            before = self._emb(ov).tobytes()
            self.assertEqual(self._used(ov), [1])

            # round 2: a second image whose recorded path is unreadable
            coco_p = orig / "Shared/PC_codes/annotations/tiles.json"
            coco = json.loads(coco_p.read_text())
            coco["images"].append({"id": 2, "file_name": str(t / "gone/t2.jpg"), "width": 40, "height": 40,
                                   "meta": {"orig_image_id": 102}})
            coco["annotations"].append({"id": 12, "image_id": 2, "category_id": 1, "iscrowd": 0,
                                        "segmentation": [[5, 5, 30, 5, 30, 30, 5, 30]], "bbox": [5, 5, 25, 25],
                                        "area": 625})
            coco_p.write_text(json.dumps(coco))
            (orig / "Shared/PC_codes/npy_npz_csv/tiles_512/splits/used_img_ids_train.json").write_text("[1, 2]")
            with self.assertRaises(PackInputError):
                self._run(orig, ov, tiles)
            self.assertEqual(self._used(ov), [1])               # unchanged
            self.assertEqual(self._emb(ov).tobytes(), before)   # unchanged

    def test_zero_contribution_image_is_completed_not_failed(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            orig, tiles = self._fixture(t, file_name_for_image=lambda i, n, d: n, n_images=2)
            coco_p = orig / "Shared/PC_codes/annotations/tiles.json"
            coco = json.loads(coco_p.read_text())
            coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] != 2]   # image 2: no instances
            coco_p.write_text(json.dumps(coco))
            rep = self._run(orig, t / "ov", tiles)
            st1 = rep["stages"]["stage1"]
            self.assertEqual(st1["contributed_ids"], [1])
            self.assertEqual(st1["zero_contribution_ids"], [2])
            self.assertEqual(st1["failed_ids"], [])
            self.assertEqual(self._used(t / "ov"), [1, 2])      # a legitimate zero contribution completes
            self.assertEqual(self._emb(t / "ov").shape[0], SCALES)

    def test_receipts_record_every_image_outcome(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            orig, tiles = self._fixture(t, file_name_for_image=lambda i, n, d: n)
            self._run(orig, t / "ov", tiles)
            rp = t / "ov/Shared/PC_codes/npy_npz_csv/tiles_512/stage2_proto/train_foldA/stage1_receipts.json"
            runs = json.loads(rp.read_text())
            self.assertEqual(runs[-1]["outcome"], "PUBLISHED")
            self.assertEqual(runs[-1]["images"]["1"]["outcome"], "CONTRIBUTED")
            self.assertEqual(runs[-1]["images"]["1"]["vectors"], SCALES)

    def test_atomic_write_of_numpy_array_lands_on_the_declared_path(self):
        """numpy appends '.npy' to a path without that suffix: the temporary name must keep it last."""

        import numpy as np

        from compag_curation.only_codes import pack as P

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            target = t / "embeddings_train_pos.npy"
            P._atomic_write(target, lambda tmp: np.save(tmp, np.arange(4, dtype=np.float32)))
            self.assertTrue(target.is_file())
            self.assertEqual(np.load(target).tolist(), [0.0, 1.0, 2.0, 3.0])
            self.assertEqual(sorted(x.name for x in t.iterdir()), ["embeddings_train_pos.npy"])

    def test_interrupted_publication_leaves_no_partial_file(self):
        import numpy as np

        from compag_curation.only_codes import pack as P

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            target = t / "out.npy"

            def boom(tmp):
                np.save(tmp, np.zeros(3, np.float32))
                raise RuntimeError("interrupted while publishing")

            with self.assertRaises(RuntimeError):
                P._atomic_write(target, boom)
            self.assertFalse(target.exists())
            self.assertEqual(list(t.glob("*.publishing")), [])


@unittest.skipUnless(HAVE_SCI, "numpy/pandas/cv2/joblib required")
@heavy
class RelocatedInputFeatureTests(unittest.TestCase):
    """The feature export uses the same resolver and refuses to half-write a relocated round."""

    def test_feature_export_maps_and_aborts_without_writing_rows(self):
        import cv2
        import numpy as np

        from unittest import mock

        from compag_curation.only_codes.features import FeatureInputError, run_feature_export
        from compag_curation.only_codes.legacy import gate_core
        from compag_curation.only_codes.state import LegacyState, PathMapper

        class _DummyModel:                       # no ResNet50 download/read in a unit test
            def to(self, *_a, **_k):
                return self

            def eval(self):
                return self

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            tiles = t / "this/img_tiles_512"; tiles.mkdir(parents=True)
            img = np.full((40, 40, 3), (40, 60, 200), np.uint8); img[10:30, 10:30] = (30, 40, 150)
            cv2.imwrite(str(tiles / "t1.jpg"), img)
            cv2.imwrite(str(tiles / "t2.jpg"), img)
            def _im(i):
                return {"id": i, "file_name": str(t / f"old/img_tiles_512/t{i}.jpg"), "width": 40, "height": 40,
                        "meta": {"orig_image_id": 100 + i}}
            def _ann(i, cat):
                return {"id": 10 + i, "image_id": i, "category_id": cat, "iscrowd": 0,
                        "segmentation": [[5, 5, 30, 5, 30, 30, 5, 30]], "bbox": [5, 5, 25, 25], "area": 625,
                        "meta": {"review_id": i}}
            coco = {"images": [_im(1), _im(2)],
                    "categories": [{"id": 1, "name": "cj"}, {"id": 2, "name": "noncj"}],
                    "annotations": [_ann(1, 1), _ann(2, 2)]}          # the export requires a mixed subset
            orig = t / "orig"; (orig / "ann").mkdir(parents=True)
            (orig / "ann/tiles.json").write_text(json.dumps(coco))
            (orig / "ann/used.json").write_text("[1, 2]")
            st = LegacyState(orig, t / "ov")
            project_tiles = t / "project_tiles"; project_tiles.mkdir()   # the round's tiles dir is empty:
            # neither the recorded absolute path nor the reference basename join can find the images
            kw = dict(state=st, coco_json=orig / "ann/tiles.json", images_dir=project_tiles,
                      pack_path=t / "missing.joblib",
                      out_csv_rel="features/features_train.csv", allowed_img_ids_json=orig / "ann/used.json",
                      review_csv=None, resnet50_weights="unused", device="cpu", feature_mode="basic")
            with mock.patch.object(gate_core, "build_embed_model", lambda *a, **k: (_DummyModel(), None)), \
                    mock.patch.object(gate_core, "embed_patch", lambda *a, **k: np.ones(8, np.float32)):
                with self.assertRaises(FeatureInputError):
                    run_feature_export(**kw)
                self.assertFalse((t / "ov/features/features_train.csv").exists())

                rep = run_feature_export(**kw, mapper=PathMapper([(str(t / "old/img_tiles_512"), str(tiles))]))
            csv_p = t / "ov/features/features_train.csv"  # noqa: E501
            self.assertTrue(csv_p.exists())
            self.assertGreater(rep["appended_rows"], 0)
            self.assertEqual(rep["input_resolution"]["by_strategy"], {"MAPPED_BY_EXPLICIT_RULE": 2})
            head = csv_p.read_text().splitlines()[1].split(",")  # first data row
            self.assertIn("t1.jpg", head)                      # recorded name stays the reference basename


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_SCI, "numpy/pandas/cv2/joblib required")
@heavy
class ReadinessVsVerifiedParityTests(unittest.TestCase):
    """F05/E08: choosing the reference preset is not evidence that outputs match."""

    class _FakeBundle:                      # stands in for a verified pickle-free bundle
        features = ["f1", "f2"]
        threshold = 0.5
        manifest: dict = {"source": {"sha256": "0" * 64}}

        def proba_fn(self):
            return lambda X: 0.0

    def _patched_run(self, **kw):
        """run_legacy_inference with the vision pipeline and the bundle loader stubbed out."""

        from unittest import mock

        from compag_curation.only_codes import inference, model_bundle
        from compag_curation.only_codes.legacy import pipeline as legacy_pipeline

        calls = {}

        def fake_pipeline(args):
            calls["yolo"] = getattr(args, "yolo_weights", "unset")

        with mock.patch.object(legacy_pipeline, "run_pipeline", fake_pipeline), \
                mock.patch.object(model_bundle, "load_safe_bundle", lambda d: self._FakeBundle()):
            man = inference.run_legacy_inference(**kw)
        return man, calls

    def _kw(self, t: Path, *, yolo=True, bundle=True):
        import numpy as np

        for name in ("proto.csv", "pca.npz", "sam2.pt", "resnet.pth", "yolo.pt"):
            (t / name).write_bytes(b"x")
        np.savez(t / "pca.npz", components=np.zeros((2, 2), np.float32), mean=np.zeros(2, np.float32))
        (t / "tiles").mkdir(exist_ok=True)
        bdir = t / "bundle"
        bdir.mkdir(exist_ok=True)
        if bundle:
            from compag_curation.only_codes.model_bundle import BUNDLE_MANIFEST

            (bdir / BUNDLE_MANIFEST).write_text("{}")
        return {"tiles_dir": t / "tiles", "run_root": t / "run", "model_bundle_dir": bdir,
                "padded_proto": t / "proto.csv", "pca_npz": t / "pca.npz", "sam2_ckpt": t / "sam2.pt",
                "resnet50_weights": t / "resnet.pth",
                "yolo_weights": (t / "yolo.pt") if yolo else (t / "absent.pt")}

    def test_missing_detector_blocks_the_reference_preset(self):
        from compag_curation.only_codes.inference import InferencePreflightError, run_legacy_inference

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            with self.assertRaises(InferencePreflightError) as ctx:
                run_legacy_inference(**self._kw(t, yolo=False))
            rep = ctx.exception.report
            self.assertTrue(any(p["input"] == "yolo_weights" for p in rep["preflight"]))
            self.assertFalse((t / "run").exists())          # nothing was published

    def test_mismatched_pinned_hash_blocks_the_run(self):
        from compag_curation.only_codes.inference import InferencePreflightError, run_legacy_inference

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            with self.assertRaises(InferencePreflightError) as ctx:
                run_legacy_inference(**self._kw(t), pinned_inputs={"yolo_weights": "0" * 64})
            self.assertTrue(any(p["status"] == "MISMATCH_WITH_PINNED_REFERENCE"
                                for p in ctx.exception.report["preflight"]))

    def test_missing_required_dependency_blocks_the_run(self):
        from compag_curation.only_codes.inference import InferencePreflightError, run_legacy_inference

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            kw = self._kw(t)
            kw["sam2_ckpt"] = t / "nope.pt"
            with self.assertRaises(InferencePreflightError):
                run_legacy_inference(**kw)

    def test_declared_detector_free_variant_is_allowed_and_labelled(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            kw = self._kw(t, yolo=False)
            man, seen = self._patched_run(**kw, detector="none")
            self.assertFalse(man["reference_preset_requested"])
            self.assertEqual(man["execution_variant"]["detector"], "none")
            self.assertIsNone(seen["yolo"])
            self.assertEqual(man["output_parity_verification"]["status"], "NOT_VERIFIED_IN_THIS_RUN")
            self.assertTrue(man["run_completed"])

    def test_successful_reference_run_does_not_claim_output_parity(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            man, _ = self._patched_run(**self._kw(t))
            self.assertTrue(man["reference_preset_requested"])
            self.assertEqual(man["effective_configuration_match"]["status"], "MATCH")
            self.assertEqual(man["inputs_verified"]["status"], "HASHES_RECORDED_WITHOUT_INDEPENDENT_PIN")
            self.assertEqual(man["output_parity_verification"]["status"], "NOT_VERIFIED_IN_THIS_RUN")
            self.assertEqual(man["manifest_schema"], "compag-only-codes-inference-run/v2")

    def test_pinned_inputs_are_reported_as_verified(self):
        import hashlib

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            kw = self._kw(t)
            pins = {k: hashlib.sha256(Path(kw[k]).read_bytes()).hexdigest()
                    for k in ("yolo_weights", "sam2_ckpt", "resnet50_weights", "padded_proto")}
            from compag_curation.only_codes.model_bundle import BUNDLE_MANIFEST

            pins["model_bundle"] = hashlib.sha256(
                (kw["model_bundle_dir"] / BUNDLE_MANIFEST).read_bytes()).hexdigest()
            pins["pca_npz"] = hashlib.sha256(Path(kw["pca_npz"]).read_bytes()).hexdigest()
            man, _ = self._patched_run(**kw, pinned_inputs=pins)
            self.assertEqual(man["inputs_verified"]["status"], "VERIFIED_AGAINST_PINNED_REFERENCE")

    def test_doctor_separates_software_hardware_and_parity(self):
        from compag_curation.only_codes.doctor import environment_report

        rep = environment_report()
        self.assertEqual(rep["schema"], "compag-only-codes-doctor/v2")
        self.assertIn(rep["software_environment_match"], ("MATCH", "DIFFERENT"))
        self.assertIn(rep["hardware_match"], ("SAME_MODEL_AND_DRIVER_AS_REFERENCE", "DIFFERENT_FROM_REFERENCE",
                                              "UNKNOWN"))
        self.assertEqual(rep["output_parity_verification"], "NOT_ESTABLISHED_BY_DOCTOR")
        self.assertIn("gpu_name", rep["hardware_observed"])


@unittest.skipUnless(HAVE_SCI, "numpy/pandas/cv2/joblib required")
@heavy
class ReferenceReaderContractTests(unittest.TestCase):
    """R1: each reader keeps its own reference path contract, with or without a mapping rule."""

    def _tile(self, path: Path, value: int):
        import cv2
        import numpy as np

        img = np.full((40, 40, 3), (value, value, value), np.uint8)
        img[10:30, 10:30] = (max(value - 20, 0),) * 3
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)

    def _fixture(self, root: Path):
        """IMAGES_DIR holds the reference images; a same-named pair exists elsewhere and in subdir/."""

        images = root / "IMAGES_DIR"
        for name in ("t1.png", "t2.png"):
            self._tile(images / name, 40)                 # what the reference exporter must read
            self._tile(root / "elsewhere" / name, 220)    # different content, same name
            self._tile(images / "subdir" / name, 220)
        return images

    def _coco(self, root: Path, names):
        imgs, anns = [], []
        for i, name in enumerate(names, start=1):
            imgs.append({"id": i, "file_name": str(name), "width": 40, "height": 40,
                         "meta": {"orig_image_id": 100 + i}})
            anns.append({"id": 10 + i, "image_id": i, "category_id": 1 if i == 1 else 2, "iscrowd": 0,
                         "segmentation": [[5, 5, 30, 5, 30, 30, 5, 30]], "bbox": [5, 5, 25, 25],
                         "area": 625, "meta": {"review_id": i}})
        orig = root / "orig"
        (orig / "ann").mkdir(parents=True, exist_ok=True)
        (orig / "ann/tiles.json").write_text(json.dumps(
            {"images": imgs, "categories": [{"id": 1, "name": "cj"}, {"id": 2, "name": "noncj"}],
             "annotations": anns}))
        (orig / "ann/used.json").write_text("[1, 2]")
        return orig

    def _export(self, root: Path, images_dir: Path, names, *, rules=()):
        from unittest import mock

        import numpy as np

        from compag_curation.only_codes.features import run_feature_export
        from compag_curation.only_codes.legacy import gate_core
        from compag_curation.only_codes.state import LegacyState, PathMapper

        class _DummyModel:
            def to(self, *_a, **_k):
                return self

            def eval(self):
                return self

        orig = self._coco(root, names)
        st = LegacyState(orig, root / "ov")
        with mock.patch.object(gate_core, "build_embed_model", lambda *a, **k: (_DummyModel(), None)), \
                mock.patch.object(gate_core, "embed_patch", lambda *a, **k: np.ones(8, np.float32)):
            rep = run_feature_export(state=st, coco_json=orig / "ann/tiles.json", images_dir=images_dir,
                                     pack_path=root / "missing.joblib",
                                     out_csv_rel="features/features_train.csv",
                                     allowed_img_ids_json=orig / "ann/used.json", review_csv=None,
                                     resnet50_weights="unused", device="cpu", feature_mode="basic",
                                     mapper=PathMapper(list(rules)))
        rows = (root / "ov/features/features_train.csv").read_text().splitlines()
        header, first = rows[0].split(","), rows[1].split(",")
        return {"n_rows": len(rows) - 1, "first": dict(zip(header, first)), "report": rep}

    # -- the reference contract of the feature exporter -------------------
    def test_feature_export_reads_images_dir_basename_for_every_file_name_form(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            out = {}
            for case, names in (("basename", ["t1.png", "t2.png"]),
                                ("absolute_elsewhere", None),
                                ("relative_subdir", ["subdir/t1.png", "subdir/t2.png"])):
                root = t / case
                images = self._fixture(root)
                if names is None:
                    names = [str(root / "elsewhere/t1.png"), str(root / "elsewhere/t2.png")]
                out[case] = self._export(root, images, names)
            ref = out["basename"]["first"]["mean_L"]
            for case in ("absolute_elsewhere", "relative_subdir"):
                self.assertEqual(out[case]["first"]["mean_L"], ref,
                                 f"{case} did not read IMAGES_DIR/basename like the reference exporter")
                self.assertEqual(out[case]["n_rows"], out["basename"]["n_rows"])
                self.assertEqual(out[case]["report"]["input_resolution"]["by_strategy"],
                                 {"LEGACY_BASENAME_JOIN": 2})

    def test_identical_same_name_contents_also_resolve_to_images_dir(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            root = t / "same"
            images = self._fixture(root)
            for name in ("t1.png", "t2.png"):           # make the alternative identical
                self._tile(root / "elsewhere" / name, 40)
            got = self._export(root, images, [str(root / "elsewhere/t1.png"), str(root / "elsewhere/t2.png")])
            res = got["report"]["input_resolution"]
            self.assertEqual(res["by_strategy"], {"LEGACY_BASENAME_JOIN": 2})
            self.assertEqual(res["reference_contract_reads"], 2)

    def test_explicit_mapping_selects_the_mapped_root_and_records_it(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            root = t / "mapped"
            images = self._fixture(root)
            names = [str(root / "old_machine/tiles/t1.png"), str(root / "old_machine/tiles/t2.png")]
            got = self._export(root, images, names,
                               rules=[(str(root / "old_machine/tiles"), str(root / "elsewhere"))])
            self.assertEqual(got["report"]["input_resolution"]["by_strategy"],
                             {"MAPPED_BY_EXPLICIT_RULE": 2})
            self.assertEqual(got["report"]["input_resolution"]["relocated_reads"], 2)
            # the mapped images are the bright ones, so the feature value must differ from the
            # reference-contract read of IMAGES_DIR
            plain = self._export(t / "plain", self._fixture(t / "plain"), ["t1.png", "t2.png"])
            self.assertNotEqual(got["first"]["mean_L"], plain["first"]["mean_L"])

    def test_mapped_destination_missing_falls_back_to_the_reference_candidate(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            root = t / "fallback"
            images = self._fixture(root)
            names = [str(root / "old/t1.png"), str(root / "old/t2.png")]
            got = self._export(root, images, names, rules=[(str(root / "old"), str(root / "gone"))])
            self.assertEqual(got["report"]["input_resolution"]["by_strategy"], {"LEGACY_BASENAME_JOIN": 2})
            plain = self._export(t / "plain2", self._fixture(t / "plain2"), ["t1.png", "t2.png"])
            self.assertEqual(got["first"]["mean_L"], plain["first"]["mean_L"])

    def test_unreadable_input_still_aborts_before_any_row_is_written(self):
        from compag_curation.only_codes.features import FeatureInputError

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            root = t / "missing"
            self._fixture(root)
            empty = root / "EMPTY_IMAGES"
            empty.mkdir()
            with self.assertRaises(FeatureInputError):
                self._export(root, empty, ["t1.png", "t2.png"])
            self.assertFalse((root / "ov/features/features_train.csv").exists())

    # -- the pack reader keeps its own, different contract ----------------
    def test_pack_reader_does_not_use_the_basename_contract(self):
        from compag_curation.only_codes.inputs import resolver_for
        from compag_curation.only_codes.state import PathMapper

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            images = t / "IMAGES_DIR"
            self._tile(images / "t1.png", 40)
            pack = resolver_for("pack", images, PathMapper([]), record_identity=False)
            absent = pack.resolve("/elsewhere/t1.png")
            self.assertFalse(absent.ok)                     # never a same-named file under IMAGES_DIR
            self.assertEqual([c["strategy"] for c in absent.candidates], ["RECORDED_ABSOLUTE_PATH"])
            rel = pack.resolve("t1.png")
            self.assertTrue(rel.ok)
            self.assertEqual(rel.strategy, "RELATIVE_JOIN")
            sub = pack.resolve("subdir/t1.png")             # relative join keeps the sub-directory
            self.assertEqual([c["path"] for c in sub.candidates], [str(images / "subdir/t1.png")])

    def test_windows_name_without_rule_is_refused_and_marked_as_extension_when_mapped(self):
        from compag_curation.only_codes.inputs import resolver_for
        from compag_curation.only_codes.state import PathMapper

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            images = t / "IMAGES_DIR"
            self._tile(images / "t1.png", 40)
            plain = resolver_for("features", images, PathMapper([]), record_identity=False)
            got = plain.resolve(r"C:\Data\tiles\t1.png")
            self.assertEqual(got.outcome, "UNSUPPORTED_WINDOWS_PATH")
            mapped = resolver_for("features", images, PathMapper([("C:/Data/tiles", str(images))]),
                                  record_identity=False)
            ok = mapped.resolve(r"C:\Data\tiles\t1.png")
            self.assertTrue(ok.ok)
            self.assertEqual(ok.strategy, "MAPPED_BY_EXPLICIT_RULE")
            self.assertFalse(ok.follows_reference_contract)
