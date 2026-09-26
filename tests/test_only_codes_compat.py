"""Unit tests for the only-codes-compat-v1 workflow (synthetic fixtures only).

Real-data parity (OLD oracle vs NEW) is run by
the separate ONLY_CODES_PARITY_KIT (``run_parity_suite.py``); these tests pin the
contract, the argument construction, the pickle-free model path, review and
merge semantics, the copy-on-write state and project reopen guards.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path

def _has(*names: str) -> bool:
    return all(importlib.util.find_spec(n) is not None for n in names)


# Heavy scientific modules are imported lazily inside the tests so that test
# discovery keeps the public import-inertness contract intact.
HAVE_SCI = _has("numpy", "pandas")
HAVE_CV2 = _has("cv2")
HAVE_ML = _has("sklearn", "xgboost", "imblearn", "joblib")
np = pd = cv2 = None
# Tests that import the scientific stack run in a child interpreter so the
# parent test process keeps the public import-inertness contract.
IN_HEAVY_CHILD = os.environ.get("ONLY_CODES_HEAVY_CHILD") == "1"
heavy = unittest.skipUnless(IN_HEAVY_CHILD, "runs in the isolated scientific child process")


def _sci():
    global np, pd, cv2
    import numpy as _np
    import pandas as _pd

    np, pd = _np, _pd
    if HAVE_CV2:
        import cv2 as _cv2

        cv2 = _cv2


class ContractTests(unittest.TestCase):
    def test_profile_and_reference_values(self):
        from compag_curation.only_codes import PROFILE_ID
        from compag_curation.only_codes.contract import REFERENCE_INFERENCE as R, contract, contract_sha256

        self.assertEqual(PROFILE_ID, "only-codes-compat-v1")
        self.assertEqual(R["det_missing"], "reject")
        self.assertEqual(R["ms_scales"], "1.0")
        self.assertEqual(R["points_per_batch"], 512)
        self.assertEqual((R["pred_iou"], R["stability"]), (0.80, 0.88))
        self.assertEqual(R["crop_overlap_ratio"], 0.4)
        self.assertFalse(R["exclude_largest"])
        self.assertEqual((R["yolo_conf"], R["yolo_iou"], R["yolo_imgsz"], R["yolo_max_det"]), (0.20, 0.60, 512, 300))
        self.assertEqual(R["device"], "cuda")
        self.assertEqual(len(contract_sha256()), 64)
        json.dumps(contract())
        self.assertEqual(contract_sha256(), contract_sha256())

    def test_training_constants(self):
        from compag_curation.only_codes.contract import REFERENCE_TRAINING as T

        self.assertEqual(T["early_stop_rounds_effective"], 30)
        self.assertEqual(T["search_every_n_rounds"], 500)
        self.assertFalse(T["force_hparam_search"])
        self.assertEqual(T["aug"]["mixup"], {"enabled": True, "alpha": 0.20, "mult": 0.50})
        self.assertEqual(T["aug"]["jitter"]["clip_q"], [0.001, 0.999])


class RunnerArgvTests(unittest.TestCase):
    REFERENCE_TAIL = ("--points-per-side 64 --points-per-batch 512 --crop-n-layers 0 --crop-n-points-downscale-factor 2 "
                      "--crop-overlap-ratio 0.4 --pred-iou 0.80 --stability 0.88 --no-exclude-largest --min-true-gates 3 "
                      "--export-features --use-xgb-inference --xgb-model M --xgb-features auto --xgb-policy replace "
                      "--xgb-threshold 0.50 --det-policy hybrid --det-missing reject --det-thr 0.5 --hybrid-yolo-bias 0.8 "
                      "--thr_yolo_raw 0.20 --thr_yolo_iou 0.60")

    def _argv(self, yolo):
        from compag_curation.only_codes.runner import build_runner_argv

        return build_runner_argv(images_dir="I", out_dir="O", sam2_config="configs/sam2.1/sam2.1_hiera_l", sam2_ckpt="C",
                                 padded_proto="P", pca_npz="Q", yolo_weights=yolo, xgb_model="M")

    def test_argv_matches_original_runner(self):
        with tempfile.TemporaryDirectory() as d:
            y = Path(d) / "best.pt"
            y.write_bytes(b"x")
            argv = self._argv(str(y))
        head = (f"--images I --out O --sam2-config configs/sam2.1/sam2.1_hiera_l --sam2-ckpt C --sam2-policy both "
                f"--feature-mode ultra --embed-csv P --embed-pca Q --yolo-weights {y} --yolo-conf 0.20 --yolo-iou 0.60 "
                f"--yolo-hint off --yolo-imgsz 512 --yolo-device 0 --yolo-max-det 300 --debug-yolo-dump ")
        self.assertEqual(" ".join(argv), head + self.REFERENCE_TAIL)

    def test_parsed_effective_arguments(self):
        from compag_curation.only_codes.runner import parse_legacy_args

        a = parse_legacy_args(self._argv(None))
        self.assertEqual(a.det_missing, "reject")
        self.assertEqual(a.ms_scales, "1.0")
        self.assertEqual((a.ms_merge_iou, a.ms_max_masks), (0.75, 500))
        self.assertEqual(a.points_per_batch, 512)
        self.assertEqual(a.max_num_masks, 0)
        self.assertTrue(a.smart_hybrid)
        self.assertEqual((a.al_topk, a.al_margin, a.al_disagree_abs), (50, 0.20, 0.55))
        self.assertTrue(a.al_only_uncertain and a.al_use_disagreement and not a.al_fill_to_topk)
        self.assertEqual(a.yolo_conf, 0.20)
        self.assertEqual(a.yolo_iou, 0.60)
        self.assertFalse(hasattr(a, "thr_yolo_raw"))
        self.assertEqual(a.yolo_pad_frac, 0.02)
        self.assertEqual(a.xgb_features, "auto")

    def test_environment_cannot_change_preset(self):
        from compag_curation.only_codes.runner import parse_legacy_args

        keys = {"CJ_SAVE_ALL": "0", "DET_MISSING": "ignore", "JASSID_FEATURE_MODE": "balanced", "SAM2_POLICY": "auto",
                "CJ_SAM2_POLICY": "prompt", "JASSID_EMBED_BACKBONE": "resnet18"}
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ.update(keys)
            a = parse_legacy_args(self._argv(None))
            self.assertEqual((a.det_missing, a.sam2_policy, a.feature_mode), ("reject", "both", "ultra"))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    @heavy
    @unittest.skipUnless(HAVE_SCI, "pandas required")
    def test_proto_repadding_written_to_work_dir(self):
        _sci()
        from compag_curation.only_codes.runner import pad_proto_csv, parse_legacy_args

        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "in" / "proto_padded.csv"
            src.parent.mkdir()
            src.write_text("mu_embed_0000,mu_embed_0001,embed_thresh\n0.1,0.2,0.7\n")
            before = src.read_bytes()
            a = parse_legacy_args(self._argv(None))
            a.embed_csv = str(src)
            rec = pad_proto_csv(a, Path(d) / "work")
            self.assertEqual(Path(rec["padded"]).name, "proto_padded_padded.csv")
            self.assertEqual(Path(rec["padded"]).parent, Path(d) / "work")
            self.assertEqual(src.read_bytes(), before)
            self.assertFalse((src.parent / "proto_padded_padded.csv").exists())


@unittest.skipUnless(HAVE_SCI and HAVE_ML, "scientific stack required")
@heavy
class ModelBundleTests(unittest.TestCase):
    def setUp(self):
        _sci()

    def _toy(self, d: Path):
        import joblib
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.impute import SimpleImputer
        from xgboost import XGBClassifier

        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(300, 5).astype(np.float32), columns=[f"f{i}" for i in range(5)])
        X.iloc[::17, 2] = np.nan
        y = (X["f0"].fillna(0) + 0.3 * rng.randn(300) > 0.8).astype(int)
        pipe = ImbPipeline([("imp", SimpleImputer(strategy="median")),
                            ("smote", SMOTE(k_neighbors=3, random_state=42, sampling_strategy=0.5)),
                            ("clf", XGBClassifier(n_estimators=20, max_depth=3, tree_method="hist", random_state=42))])
        pipe.fit(X, y)
        p = d / "toy_xgb_r3_hybrid.pkl"
        joblib.dump({"pipeline": pipe, "features": list(X.columns), "threshold": 0.61, "meta": {"a": 1}}, p)
        return p, X

    def test_exact_conversion_and_pickle_free_load(self):
        from compag_curation.only_codes.model_bundle import LegacyModelError, convert_trusted_legacy_pickle, load_safe_bundle

        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            p, X = self._toy(d)
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            with self.assertRaises(LegacyModelError):
                convert_trusted_legacy_pickle(p, d / "b0", trusted_sha256="0" * 64, verification_rows=X)
            man = convert_trusted_legacy_pickle(p, d / "b1", trusted_sha256=sha, verification_rows=X)
            self.assertEqual(man["verification"]["status"], "PASS_EXACT")
            b = load_safe_bundle(d / "b1")
            self.assertEqual(b.features, list(X.columns))
            self.assertEqual(b.threshold, 0.61)
            fn = b.proba_fn()
            self.assertIsInstance(fn(pd.DataFrame([X.iloc[0].to_dict()], columns=X.columns)), float)
            (d / "b1" / "schema.json").write_text("{}")
            with self.assertRaises(LegacyModelError):
                load_safe_bundle(d / "b1")

    def test_imputer_replica_matches_sklearn(self):
        from sklearn.impute import SimpleImputer

        from compag_curation.only_codes.model_bundle import LegacyMedianImputer

        rng = np.random.RandomState(1)
        X = pd.DataFrame(rng.randn(50, 4).astype(np.float32), columns=list("abcd"))
        X.iloc[::5, 1] = np.nan
        imp = SimpleImputer(strategy="median").fit(X)
        T = X.astype(np.float64).copy()
        T.iloc[::3, 0] = np.nan
        a = imp.transform(T)
        b = LegacyMedianImputer(imp.statistics_, list(X.columns)).transform(T)
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.tobytes(), b.tobytes())


@unittest.skipUnless(HAVE_SCI and HAVE_ML, "joblib required")
@heavy
class SafeJoblibTests(unittest.TestCase):
    def setUp(self):
        _sci()

    def test_numpy_pack_round_trip_and_refusal(self):
        import joblib

        from compag_curation.only_codes.safe_joblib import RestrictedPickleError, load_data_pack

        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "pack.joblib"
            pack = {"used_img_ids": [1, 2, 3], "mu": np.arange(4, dtype=np.float32), "embed_thresh": 0.7}
            joblib.dump(pack, p)
            got = load_data_pack(p)
            self.assertEqual(got["used_img_ids"], [1, 2, 3])
            self.assertEqual(got["mu"].tobytes(), pack["mu"].tobytes())

            class Evil:
                def __reduce__(self):
                    return (os.system, ("echo pwned",))

            bad = Path(t) / "bad.joblib"
            bad.write_bytes(pickle.dumps({"x": Evil()}))
            with self.assertRaises(RestrictedPickleError):
                load_data_pack(bad)


@unittest.skipUnless(HAVE_SCI and HAVE_CV2, "pandas/cv2 required")
@heavy
class ReviewTests(unittest.TestCase):
    def setUp(self):
        _sci()

    def _session(self, d: Path, clock):
        from compag_curation.only_codes.review import LegacyReviewSession

        tiles = d / "tiles"
        tiles.mkdir()
        cv2.imwrite(str(tiles / "IMG_1_y00000x00000.jpg"), np.full((64, 64, 3), 200, np.uint8))
        rows = [
            {"image": "IMG_1_y00000x00000.jpg", "id": 1, "x": 1, "y": 2, "w": 10, "h": 8, "kept": 1, "xgb_p": 0.62,
             "yolo_conf": 0.0, "yolo_iou": 0.0, "poly": "[1, 2, 11, 2, 11, 10, 1, 10]"},
            {"image": "IMG_1_y00000x00000.jpg", "id": 2, "x": 20, "y": 20, "w": 5, "h": 5, "kept": 0, "xgb_p": 0.39,
             "yolo_conf": 0.7, "yolo_iou": 0.8, "poly": ""},
        ]
        det = d / "detections.csv"
        pd.DataFrame(rows).to_csv(det, index=False)
        return LegacyReviewSession(detections_csv=det, tiles_dir=tiles, review_csv=d / "review_labels.csv",
                                   seen_json=d / "seen.json", cache_dir=d / "cache", clock=clock)

    def test_gui_decision_policy_is_xgb_ignore_040(self):
        from compag_curation.only_codes.review import compute_final_pred_and_ui

        self.assertEqual(compute_final_pred_and_ui({"xgb_p": 0.40})[0], 1)
        self.assertEqual(compute_final_pred_and_ui({"xgb_p": 0.3999})[0], 0)
        self.assertEqual(compute_final_pred_and_ui({"xgb_p": 0.2})[2], 0.40)

    def test_actions_undo_skip_and_latest_rule(self):
        ts = iter(range(20260101000000, 20260101000100))
        with tempfile.TemporaryDirectory() as t:
            s = self._session(Path(t), lambda: next(ts))
            k1, k2 = "IMG_1_y00000x00000.jpg||1", "IMG_1_y00000x00000.jpg||2"
            s.act("accept", sel=k1)
            s.act("flip", sel=k1)
            self.assertEqual(s.labels_eff[k1]["human_label"], 0)
            self.assertEqual(s.labels_eff[k1]["action"], "flip")
            s.act("skip", sel=k2)
            self.assertIn(k2, s.seen)
            self.assertIn("un-saw", s.act("undo"))
            self.assertNotIn(k2, s.seen)
            df = pd.read_csv(Path(t) / "review_labels.csv")
            self.assertEqual(list(df.columns)[:5], ["image", "id", "human_label", "action", "review_weight"])
            self.assertEqual(df["click_x"].tolist(), [6, 6])
            items = {i["key"]: i for i in s.items(s.bases[0])}
            self.assertEqual(items[k2]["geometry"], "bbox_rectangle")
            self.assertEqual(items[k1]["geometry"], "poly")


@heavy
class MergeSplitTests(unittest.TestCase):
    def test_merge_latest_equal_timestamp_and_fallbacks(self):
        from compag_curation.only_codes.merge import merge_reviews

        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            orig = {"images": [{"id": 5, "file_name": "IMG_2.jpg"}], "annotations": [], "categories": []}
            tiled = {"images": [], "annotations": [], "categories": []}
            (d / "o.json").write_text(json.dumps(orig)); (d / "t.json").write_text(json.dumps(tiled))
            (d / "det.csv").write_text("image,id,poly,bbox_x,bbox_y,bbox_w,bbox_h\n"
                                       "IMG_3_y00000x00000.jpg,1,\"[0, 0, 4, 0, 4, 4, 0, 4]\",0,0,4,4\n"
                                       "IMG_3_y00000x00000.jpg,2,,5,5,3,2\n")
            (d / "rev.csv").write_text("image,id,human_label,timestamp\n"
                                       "IMG_3_y00000x00000.jpg,1,1,10\nIMG_3_y00000x00000.jpg,1,0,10\n"
                                       "IMG_3_y00000x00000.jpg,2,,11\nIMG_3_y00000x00000.jpg,9,1,12\n")
            rep = merge_reviews(orig_coco_in=d / "o.json", tiled_coco_in=d / "t.json", detections_csv=d / "det.csv",
                                reviews_csv=d / "rev.csv", orig_coco_out=d / "o2.json", tiled_coco_out=d / "t2.json",
                                new_orig_ids_out=d / "new.json")
            t2 = json.loads((d / "t2.json").read_text())
            cats = {a["meta"]["review_id"]: a["category_id"] for a in t2["annotations"]}
            self.assertEqual(cats, {1: 2, 2: 2})  # equal ts -> later row; skip/NaN -> non-CJ (LB-16)
            self.assertEqual(rep["geometry"]["bbox_rectangle_fallback"], 1)
            self.assertEqual(json.loads((d / "new.json").read_text()), [6])
            self.assertEqual((d / "t2.json").read_text(), json.dumps(t2, indent=2))

    def test_splits_append_train_only(self):
        from compag_curation.only_codes.splits import update_splits

        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "o.json").write_text(json.dumps({"images": [{"id": i} for i in (1, 2, 3, 4)]}))
            sd = d / "splits"; sd.mkdir()
            (sd / "orig_train_ids.json").write_text("[1, 2]"); (sd / "orig_val_ids.json").write_text("[3]")
            (sd / "orig_test_ids.json").write_text("[]")
            (d / "new.json").write_text("[3, 4]")
            rep = update_splits(orig_coco_json=d / "o.json", split_dir=sd, append_new_orig_ids=d / "new.json")
            self.assertEqual(rep["mode"], "APPEND_TRAIN_ONLY")
            self.assertEqual(json.loads((sd / "orig_train_ids.json").read_text()), [1, 2, 3, 4])
            self.assertEqual(json.loads((sd / "orig_val_ids.json").read_text()), [])
            self.assertTrue((d / "new.applied.json").exists())


class StateProjectTests(unittest.TestCase):
    def test_path_mapper_longest_prefix(self):
        from compag_curation.only_codes.state import PathMapper

        m = PathMapper([("/old", "/n1"), ("/old/Clean", "/n2")])
        self.assertEqual(m.map("/old/Clean/a.csv"), "/n2/a.csv")
        self.assertEqual(m.map("/old/x"), "/n1/x")
        self.assertEqual(m.map("/other"), "/other")
        self.assertEqual(m.map("/oldish/x"), "/oldish/x")

    def test_copy_on_write_never_touches_original(self):
        from compag_curation.only_codes.state import LegacyState

        with tempfile.TemporaryDirectory() as t:
            orig = Path(t) / "orig"; ov = Path(t) / "ov"
            (orig / "a").mkdir(parents=True)
            (orig / "a" / "f.txt").write_text("orig")
            st = LegacyState(orig, ov)
            self.assertEqual(st.path("a/f.txt"), orig / "a" / "f.txt")
            st.ensure_overlay_copy("a/f.txt").write_text("changed")
            self.assertEqual((orig / "a" / "f.txt").read_text(), "orig")
            self.assertEqual(st.path("a/f.txt").read_text(), "changed")
            with self.assertRaises(ValueError):
                st.path("../escape")

    def test_import_and_reopen_guards(self):
        from compag_curation.only_codes import project as P

        with tempfile.TemporaryDirectory() as t:
            legacy = Path(t) / "Clean"
            (legacy / "always_same/jupyter/models/cj_classifier/r5_hybrid").mkdir(parents=True)
            (legacy / "Shared/maskout_tile/IMG_10/tiles").mkdir(parents=True)
            (legacy / "Shared/maskout_tile/IMG_12").mkdir(parents=True)
            proj = P.import_project(legacy_root=legacy, output=Path(t) / "proj", sam2_repo_root=Path(t) / "sam2",
                                    torch_home=Path(t) / "torch", image="IMG_10", hash_large=False)
            self.assertTrue((Path(t) / "proj" / "INPUT_COMPATIBILITY_MATRIX.csv").exists())
            again = P.open_project(Path(t) / "proj")
            self.assertEqual(again.manifest["profile"], "only-codes-compat-v1")
            with self.assertRaises(P.ProjectError):
                P.import_project(legacy_root=legacy, output=Path(t) / "proj", sam2_repo_root=Path(t),
                                 torch_home=Path(t), hash_large=False)
            man = json.loads((Path(t) / "proj" / P.PROJECT_MANIFEST).read_text())
            man["contract_sha256"] = "0" * 64
            (Path(t) / "proj" / P.PROJECT_MANIFEST).write_text(json.dumps(man))
            with self.assertRaises(P.ProjectError):
                P.open_project(Path(t) / "proj")
            roots = [legacy / "Shared/maskout_tile"]
            self.assertEqual(P.next_image_name(roots, "IMG_10"), "IMG_12")
            self.assertEqual(P.next_image_name(roots, "IMG_12"), "IMG_12")
            name, tag = P.alloc_next_model_name([legacy / "always_same/jupyter/models/cj_classifier"],
                                                "cj_ultra_tilesafe_xgb_r5_hybrid.pkl")
            self.assertEqual((name, tag), ("cj_ultra_tilesafe_xgb_r6_hybrid.pkl", "r6_hybrid"))


@unittest.skipUnless(HAVE_SCI, "numpy/pandas required")
@heavy
class TrainingHelperTests(unittest.TestCase):
    def setUp(self):
        _sci()

    def test_tile_balanced_split_deterministic_and_group_pure(self):
        from compag_curation.only_codes.training import original_image_group, pick_groups_tile_balanced

        groups = np.array([f"IMG_{i % 13}" for i in range(500)])
        a = pick_groups_tile_balanced(groups, test_frac=0.2, random_state=42, tol=0.01, max_tries=3000)
        b = pick_groups_tile_balanced(groups, test_frac=0.2, random_state=42, tol=0.01, max_tries=3000)
        self.assertEqual(a[1], b[1])
        self.assertFalse(a[0] & a[1])
        self.assertEqual(original_image_group("IMG_9510_y00512x01024.jpg"), "IMG_9510")

    def test_augmentations_are_seeded(self):
        from compag_curation.only_codes.training import apply_augs_in_3A
        from compag_curation.only_codes.contract import REFERENCE_TRAINING

        rng = np.random.RandomState(3)
        X = pd.DataFrame(rng.rand(60, 4).astype(np.float32), columns=list("abcd"))
        X["d"] = (X["d"] > 0.5).astype(np.float32)  # discrete column: excluded from mixing
        y = (rng.rand(60) > 0.7).astype(int); w = np.ones(60, np.float32); g = np.array(["G1", "G2"] * 30)
        cfg = json.loads(json.dumps(REFERENCE_TRAINING["aug"])); cfg["jitter"]["clip_q"] = tuple(cfg["jitter"]["clip_q"])
        r1 = apply_augs_in_3A(X, y, w, g, cfg, random_state=42, log=lambda *_: None)
        r2 = apply_augs_in_3A(X, y, w, g, cfg, random_state=42, log=lambda *_: None)
        self.assertEqual(r1[0].to_numpy().tobytes(), r2[0].to_numpy().tobytes())
        self.assertTrue(set(np.unique(r1[0]["d"])) <= {0.0, 1.0})


@heavy
class VendoredSourceIntegrityTests(unittest.TestCase):
    """Vendored copies differ from Only_codes only in documented plumbing patches."""

    VENDORED = {"gate_core.py": "gate_core.py", "pipeline.py": "sam2_pipeline/pipeline.py",
                "light_gate.py": "sam2_pipeline/light_gate.py", "pca_utils.py": "sam2_pipeline/pca_utils.py"}
    UNCHANGED_FUNCTIONS = {
        "gate_core.py": ["warp_card_to_rect", "detect_grid_lines", "cell_of_point", "compute_features", "yellow_bg_stats",
                         "crop_masked_patch", "embed_patch", "load_embed_csv", "load_embed_pca", "project_embed_pca",
                         "_lbp_uniform", "_glcm_contrast_homogeneity", "_grid_one_pass", "_grid_run_sweep"],
        "pipeline.py": ["_prepare_feature_export_cols", "_resolve_xgb_feature_names", "_bbox_iou_xyxy", "_yolo_support",
                        "_al_uncertainty_from_dist", "_load_thresholds_and_stats", "_resolve_embed_proto",
                        "_mask_to_poly_and_bbox_in_original", "_mask_rle_in_original", "_mask_iou", "_amg_multiscale",
                        "_draw_masks", "_render_info_panel"],
    }

    @staticmethod
    def _funcs(src: str) -> dict:
        return {n.name: ast.dump(n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}

    def test_unchanged_functions_ast_equal_to_only_codes(self):
        clean = os.environ.get("ONLY_CODES_CLEAN")
        if not clean or not Path(clean).is_dir():
            self.skipTest("set ONLY_CODES_CLEAN=<Only_codes/Clean> to verify vendored sources")
        import compag_curation.only_codes.legacy as L

        vdir = Path(L.__file__).parent
        for vname, oname in self.VENDORED.items():
            v = self._funcs((vdir / vname).read_text())
            o = self._funcs((Path(clean) / oname).read_text())
            for fn in self.UNCHANGED_FUNCTIONS.get(vname, []):
                self.assertEqual(v[fn], o[fn], f"{vname}:{fn} differs from Only_codes")

    def test_run_pipeline_differs_only_in_tagged_patches(self):
        clean = os.environ.get("ONLY_CODES_CLEAN")
        if not clean or not Path(clean).is_dir():
            self.skipTest("set ONLY_CODES_CLEAN=<Only_codes/Clean>")
        import difflib

        import compag_curation.only_codes.legacy as L

        v = (Path(L.__file__).parent / "pipeline.py").read_text().splitlines()
        o = (Path(clean) / "sam2_pipeline/pipeline.py").read_text().splitlines()
        added = [ln for ln in difflib.unified_diff(o, v, lineterm="", n=0) if ln.startswith("+") and not ln.startswith("+++")]
        # every changed region is inside a tagged patch block: count patch tags vs changed blocks
        self.assertTrue(any("ONLY-CODES-COMPAT PATCH PL-4" in ln for ln in added))
        for ln in added:
            self.assertNotIn("os.getenv", ln)

    def test_new_gate_core_reuse_is_verified(self):
        from compag_curation.proposals.sam2_pipeline import gate_core as new_gc
        import compag_curation.only_codes.legacy as L

        vendored = self._funcs((Path(L.__file__).parent / "gate_core.py").read_text())
        new = self._funcs(Path(new_gc.__file__).read_text())
        for fn in ("warp_card_to_rect", "detect_grid_lines", "cell_of_point", "yellow_bg_stats", "crop_masked_patch",
                   "embed_patch", "load_embed_csv", "load_embed_pca", "project_embed_pca"):
            self.assertEqual(vendored[fn], new[fn], f"canonical gate_core.{fn} drifted from the legacy reference")


class IsolatedScientificChildTests(unittest.TestCase):
    @unittest.skipIf(IN_HEAVY_CHILD, "parent-only orchestrator")
    def test_scientific_units_pass_in_isolated_process(self):
        import subprocess
        import sys

        env = dict(os.environ, ONLY_CODES_HEAVY_CHILD="1", PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run([sys.executable, "-B", "-m", "unittest", "-q", "tests.test_only_codes_compat"],
                              cwd=str(Path(__file__).resolve().parents[1]), env=env, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])


class CliTests(unittest.TestCase):
    def test_only_codes_command_group_parses(self):
        from compag_curation.cli import _parser

        a = _parser().parse_args(["only-codes", "infer", "--project", "P", "--image", "IMG_1", "--model", "r1_hybrid"])
        self.assertEqual((a.command, a.oc_action, a.device), ("only-codes", "infer", "cuda"))
        with self.assertRaises(SystemExit):
            _parser().parse_args(["only-codes", "infer", "--project", "P", "--model", "r1"])


if __name__ == "__main__":
    unittest.main()
