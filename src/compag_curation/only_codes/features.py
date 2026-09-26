"""Port of notebook cell 5 "JUPYTER CELL 2: COCO → Features (subset-only; uses TRAIN pack)".

Training-feature export with scales [0.67, 0.80, 1.00, 1.25]: the tile image
(INTER_LINEAR) and mask (INTER_NEAREST) are resized *first*, then features,
background statistics and the masked crop are computed on the scaled image.
``embed_sim`` is ``clip((dot(z, mu) + 1) / 2)``; ``grid_*`` use the scaled
tile size with 3x3 bins; ``pred_iou = stability = 1.0``.

The historical ``features_train.csv`` is consumed as-is: its resume keys
``(file_name, ann_id, scale)`` (with the ``'*'`` wildcard) decide which rows are
already present, exactly as in the notebook.  Existing rows are never
recomputed or rewritten.  New rows are appended to a *project copy* (created
only when something is appended or a schema migration is required); a
provenance sidecar records which rows were computed by this application and
with which transform identity.
"""

from __future__ import annotations

import csv as _csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .inputs import resolver_for
from .state import LegacyState, PathMapper

MULTISCALE_SCALES = [0.67, 0.80, 1.00, 1.25]
WEIGHT_MAP = {"accept": 1.0, "flip": 1.0, "sus_accept": 0.4, "sus_flip": 0.4, "skip": 0.0}
_WEIGHT_KEYS = set(WEIGHT_MAP.keys())

FEAT_BASE = [
    "file_name", "ann_id", "scale",
    "area_px", "area_norm",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "cx", "cy",
    "mean_L", "mean_a", "mean_b", "std_L", "std_a", "std_b", "median_L", "median_a", "median_b",
    "delta_a", "delta_b",
    "elongation", "eccentricity", "solidity", "aspect_ratio", "circularity", "extent", "perimeter",
    "bbox_diag_frac", "perim_over_sqrt_area", "scale_diag",
    "hu1", "hu2", "hu4", "components_count", "holes_ratio", "touching_border",
    "LBP_u5", "GLCM_contrast", "GLCM_homogeneity", "grad_mean", "grad_p90",
    "grid_r", "grid_c", "grid_r_norm", "grid_c_norm",
    "pred_iou", "stability", "embed_sim",
    "g_quality", "g_light", "g_color", "g_shape", "g_embed", "g_robust", "g_maha", "g_border",
]
FEAT_RICH = [
    "delta_b_med", "delta_a_med", "median_b_in", "median_b_ring", "median_a_in", "median_a_ring",
    "histb_q1", "histb_q2", "histb_q3", "histb_q4",
    "histab_q11", "histab_q12", "histab_q21", "histab_q22",
]


def _safe_div(a, b):
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def _clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def _detect_review_tag_col(df_rev) -> Optional[str]:
    best = None
    best_rate = 0.0
    for c in df_rev.columns:
        vals = df_rev[c].astype(str).str.strip().str.lower()
        rate = float(vals.isin(_WEIGHT_KEYS).mean())
        if rate > best_rate:
            best_rate = rate
            best = c
    return best if best_rate >= 0.01 else None


def load_review_map(review_csv: Path) -> Tuple[Dict[Tuple[str, str], str], Optional[str]]:
    import pandas as pd

    if not Path(review_csv).exists():
        return {}, None
    df = pd.read_csv(review_csv, low_memory=False)
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    else:
        df["id"] = df.index.astype(str)
    img_col = None
    for c in ["image", "file_name", "file", "filename"]:
        if c in df.columns:
            img_col = c
            break
    if img_col is None:
        df["_file"] = ""
    else:
        df["_file"] = df[img_col].astype(str).map(lambda s: Path(s).name)
    tag_col = _detect_review_tag_col(df)
    if tag_col is not None:
        df["_tag"] = df[tag_col].astype(str).str.strip()
    else:
        df["_tag"] = ""
    return dict(zip(zip(df["_file"], df["id"]), df["_tag"])), tag_col


def _segmentation_to_mask(ann: Dict[str, Any], img_h: int, img_w: int) -> np.ndarray:
    import cv2

    m = np.zeros((img_h, img_w), dtype=np.uint8)
    seg = ann.get("segmentation", None)
    if isinstance(seg, list) and seg:
        polys = []
        for poly in seg:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            polys.append(pts.astype(np.int32))
        cv2.fillPoly(m, polys, 255)
        return m
    if isinstance(seg, dict) and "counts" in seg:
        try:
            import pycocotools.mask as maskUtils
            rle = seg if isinstance(seg.get("counts"), (list, str, bytes)) else maskUtils.frPyObjects(seg, img_h, img_w)
            mm = maskUtils.decode(rle)
            if mm.ndim == 3:
                mm = mm.max(axis=2)
            return (mm.astype(np.uint8) * 255)
        except Exception:
            pass
    x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
    x0 = int(max(0, x)); y0 = int(max(0, y))
    x1 = int(min(img_w - 1, x + w)); y1 = int(min(img_h - 1, y + h))
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = 255
    return m


def load_done_triples(csv_path: Path) -> Set[Tuple[str, str, str]]:
    if not Path(csv_path).exists():
        return set()
    done: Set[Tuple[str, str, str]] = set()
    try:
        with Path(csv_path).open("r", newline="", encoding="utf-8", errors="replace") as fin:
            rdr = _csv.reader(fin)
            header = next(rdr, None)
            if not header:
                return set()

            def _idx(name: str) -> int:
                try:
                    return header.index(name)
                except ValueError:
                    return -1
            i_fn = _idx("file_name"); i_id = _idx("ann_id"); i_sc = _idx("scale")
            if i_fn < 0 or i_id < 0:
                return set()
            for row in rdr:
                try:
                    fn = str(row[i_fn]); aid = str(row[i_id])
                    sc = (str(row[i_sc]) if i_sc >= 0 else "*")
                    if sc.strip() == "":
                        sc = "*"
                    done.add((fn, aid, sc))
                except Exception:
                    continue
    except Exception:
        return set()
    return done


def _needs_migration(csv_path: Path) -> Tuple[bool, bool]:
    if not csv_path.exists():
        return False, False
    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as fin:
        header = next(_csv.reader(fin), None) or []
    return ("scale" not in header), ("review_tag" not in header)


def _insert_blank_column(csv_path: Path, name: str, after: str | None, before: str | None, bak_suffix: str) -> None:
    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as fin:
        rdr = _csv.reader(fin)
        old_header = next(rdr, None)
        if name == "scale":
            ins_idx = (old_header.index("ann_id") + 1) if "ann_id" in old_header else 2
        else:
            if "reviewed" in old_header:
                ins_idx = old_header.index("reviewed") + 1
            elif "class_id" in old_header:
                ins_idx = old_header.index("class_id")
            else:
                ins_idx = len(old_header)
        new_header = old_header[:ins_idx] + [name] + old_header[ins_idx:]
        tmp = csv_path.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fout:
            wtr = _csv.writer(fout)
            wtr.writerow(new_header)
            N_old = len(old_header)
            for row in rdr:
                if len(row) < N_old:
                    row = row + [""] * (N_old - len(row))
                elif len(row) > N_old:
                    row = row[:N_old - 1] + [",".join(row[N_old - 1:])]
                wtr.writerow(row[:ins_idx] + [""] + row[ins_idx:])
    bkp = csv_path.with_suffix(bak_suffix)
    shutil.move(str(csv_path), str(bkp))
    shutil.move(str(tmp), str(csv_path))


class FeatureInputError(RuntimeError):
    """An image of this export could not be read; no row and no resume key was written.

    The reference cell skips an unreadable image silently, which would append the remaining rows
    and leave the missing ones to be appended out of order later.  This profile aborts instead, so
    that a retry after fixing the input reproduces the clean-run row order exactly once.  This is a
    documented departure for *failed* inputs only; successful inputs follow the reference exactly.
    """

    def __init__(self, message: str, report: Dict[str, Any]):
        super().__init__(message)
        self.report = report


def run_feature_export(*, state: LegacyState, coco_json: Path, images_dir: Path, pack_path: Path,
                       out_csv_rel: str, allowed_img_ids_json: Path, review_csv: Path | None,
                       resnet50_weights: str, device: str, feature_mode: str = "ultra",
                       pad_frac: float = 0.10, pos_name: str = "cj",
                       transform_identity: Dict[str, Any] | None = None,
                       mapper: "PathMapper | None" = None) -> Dict[str, Any]:
    import cv2

    from .legacy import gate_core
    from .legacy.gate_core import compute_features, crop_masked_patch, embed_patch, yellow_bg_stats
    from .safe_joblib import load_data_pack

    review_tag_map, tag_col = load_review_map(review_csv) if review_csv else ({}, None)
    review_pairs = set(review_tag_map.keys())

    data = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    id_to_cat = {int(c["id"]): (c.get("name", "") or "") for c in data.get("categories", [])}
    pos_cat_ids = {cid for cid, nm in id_to_cat.items() if (nm or "").lower() in {pos_name.lower()}}
    im_has_pos = {int(im["id"]): False for im in data.get("images", [])}
    for a in data.get("annotations", []):
        if int(a.get("category_id", -1)) in pos_cat_ids:
            im_has_pos[int(a.get("image_id", -1))] = True

    if not Path(allowed_img_ids_json).exists():
        raise RuntimeError("No allowed_img_ids provided. Aborting to prevent leakage.")
    allowed = set(map(int, json.loads(Path(allowed_img_ids_json).read_text(encoding="utf-8"))))
    flags = [bool(im_has_pos.get(i, False)) for i in allowed]
    n_pos = int(np.sum(flags)); n_neg = int(len(flags) - n_pos)
    if not ((n_pos > 0) and (n_neg > 0)):
        raise SystemExit(f"[ABORT] Allowed subset not mixed: pos={n_pos}, neg={n_neg}.")

    id_to_img = {int(im["id"]): im for im in data.get("images", [])}
    gate_core.set_feature_mode(feature_mode)

    csv_read_path = state.path(out_csv_rel)
    migrated: list[str] = []
    need_scale, need_tag = _needs_migration(csv_read_path)
    if need_scale or need_tag:
        ov = state.ensure_overlay_copy(out_csv_rel)
        if need_scale:
            _insert_blank_column(ov, "scale", None, None, ".pre_scale.bak.csv"); migrated.append("scale")
        if need_tag:
            _insert_blank_column(ov, "review_tag", None, None, ".pre_reviewtag.bak.csv"); migrated.append("review_tag")
        csv_read_path = ov

    embed_model, embed_transform = gate_core.build_embed_model(resnet50_weights)
    embed_model = embed_model.to(device).eval()
    embed_fallbacks = 0

    pack = load_data_pack(pack_path) if Path(pack_path).exists() else None
    mu = thr = comps = mean = None
    if pack is not None:
        mu = pack.get("mu", None)
        thr = pack.get("embed_thresh", None)
        comps = pack.get("pca_components", None)
        mean = pack.get("pca_mean", None)
        if isinstance(mu, np.ndarray):
            mu = (mu / (np.linalg.norm(mu) + 1e-12)).astype(np.float32)
        comps = np.asarray(comps, np.float32) if comps is not None else None
        mean = np.asarray(mean, np.float32) if mean is not None else None
    have_mu = isinstance(mu, np.ndarray)
    have_pca = isinstance(comps, np.ndarray) and isinstance(mean, np.ndarray)
    PCA_NAMES = [f"embed_pca_{i}" for i in range(int(comps.shape[0]))] if (feature_mode == "ultra" and have_pca) else []

    exists_before = csv_read_path.exists()
    done_triples = load_done_triples(csv_read_path) if exists_before else set()
    header = (FEAT_BASE + (FEAT_RICH if feature_mode in ("rich", "ultra") else []) + PCA_NAMES
              + ["reviewed", "review_tag", "class_id", "class_name", "label"])

    new_rows: List[list] = []
    new_keys: List[Tuple[str, str, str]] = []
    img_cache: Dict[int, np.ndarray] = {}
    _last_img_id: Optional[int] = None
    _precomp_scaled: Dict[float, tuple] = {}
    unreadable: list[str] = []
    resolver = resolver_for("features", images_dir, mapper)
    resolved_cache: Dict[int, Any] = {}
    failed_images: Dict[str, Any] = {}
    anns = [a for a in data.get("annotations", []) if int(a.get("image_id", -1)) in allowed]
    for k, ann in enumerate(anns):
        img_id = int(ann.get("image_id", -1))
        if img_id not in allowed:
            continue
        info = id_to_img.get(img_id)
        if not info:
            continue
        file_name = Path(str(info.get("file_name", ""))).name   # recorded column: reference basename
        ann_id_str = str(ann.get("id", k))
        if img_id not in resolved_cache:
            resolved_cache[img_id] = resolver.resolve(str(info.get("file_name", "")))
        resolution = resolved_cache[img_id]
        img_path = resolution.path      # single authoritative reader-aware resolution
        rid = int(ann.get("meta", {}).get("review_id", -1))
        pair = (file_name, str(rid))
        reviewed_flag = 1 if (pair in review_pairs) else 0
        review_tag = str(review_tag_map.get(pair, "") if reviewed_flag else "").strip()
        if img_id not in img_cache:
            im = cv2.imread(str(img_path), cv2.IMREAD_COLOR) if resolution.ok else None
            if im is None:
                unreadable.append(str(img_path) if img_path else resolution.logical)
                failed_images[str(img_id)] = {"file_name": file_name,
                                              "path": str(img_path) if img_path else None,
                                              "resolution": resolution.to_dict()}
                continue
            img_cache[img_id] = im
        bgr = img_cache[img_id]
        H, W = bgr.shape[:2]
        m0 = _segmentation_to_mask(ann, H, W)
        if _last_img_id != img_id:
            _precomp_scaled = {}
            _last_img_id = img_id
        for s in MULTISCALE_SCALES:
            s_key = round(float(s), 4)
            s_str = f"{s_key:.4f}"
            if (file_name, ann_id_str, s_str) in done_triples or (file_name, ann_id_str, "*") in done_triples:
                continue
            if s_key in _precomp_scaled:
                bgr_s, lab_s, bg_a_s, bg_b_s, bg_bgr_s = _precomp_scaled[s_key]
            else:
                bgr_s = bgr if abs(s - 1.0) < 1e-4 else cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                lab_s = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2LAB)
                bg_a_s, bg_b_s, bg_bgr_s = yellow_bg_stats(bgr_s)
                _precomp_scaled[s_key] = (bgr_s, lab_s, bg_a_s, bg_b_s, bg_bgr_s)
            m_s = m0 if abs(s - 1.0) < 1e-4 else cv2.resize(m0, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            Hs, Ws = bgr_s.shape[:2]
            f = compute_features(m_s, lab_s, mode=feature_mode)
            if not f:
                continue
            f["delta_a"] = bg_a_s - f.get("mean_a", 0.0)
            f["delta_b"] = f.get("mean_b", 0.0) - bg_b_s
            GRID_R, GRID_C = 3, 3
            cy = float(f.get("cy", 0.0)); cx = float(f.get("cx", 0.0))
            f["grid_r_norm"] = float(cy) / max(1.0, Hs)
            f["grid_c_norm"] = float(cx) / max(1.0, Ws)
            f["grid_r"] = int(np.clip(int(f["grid_r_norm"] * GRID_R), 0, GRID_R - 1))
            f["grid_c"] = int(np.clip(int(f["grid_c_norm"] * GRID_C), 0, GRID_C - 1))
            if ("hu1" not in f) or ("hu2" not in f) or ("hu4" not in f):
                mm = (m_s > 0).astype(np.uint8)
                M = cv2.moments(mm, binaryImage=True)
                hu = cv2.HuMoments(M).flatten()
                hu = np.sign(hu) * np.log1p(np.abs(hu))
                f["hu1"], f["hu2"], f["hu4"] = float(hu[0]), float(hu[1]), float(hu[3])

            pred_iou, stability = 1.0, 1.0
            embed_sim = 0.0
            z_vec = None
            patch = crop_masked_patch(bgr_s, m_s, pad_frac=pad_frac, bg_bgr=bg_bgr_s)
            if patch is not None:
                try:
                    z = embed_patch(patch, embed_model, embed_transform, device=device).astype(np.float32)
                except Exception:
                    embed_fallbacks += 1
                    try:
                        z = embed_patch(patch, embed_model.to("cpu").eval(), embed_transform, device="cpu").astype(np.float32)
                    except Exception:
                        z = None
                if z is not None:
                    z /= (np.linalg.norm(z) + 1e-12)
                    z_vec = z
                    if have_mu:
                        embed_sim = float(_clamp((float(np.dot(z_vec, mu)) + 1.0) * 0.5))

            pca_vals: List[float] = []
            if (z_vec is not None) and have_pca:
                try:
                    y = comps @ (z_vec - mean)
                    pca_vals = [float(v) for v in y.tolist()]
                except Exception:
                    pca_vals = [0.0] * len(PCA_NAMES)
            else:
                pca_vals = [0.0] * len(PCA_NAMES)

            mean_L = float(f.get("mean_L", 0.0)); std_L = float(f.get("std_L", 0.0))
            delta_a = float(f.get("delta_a", 0.0)); delta_b = float(f.get("delta_b", 0.0))
            ecc = float(f.get("eccentricity", 0.0)); circ = float(f.get("circularity", 0.0))
            solid = float(f.get("solidity", 0.0)); touch = int(f.get("touching_border", 0))
            compc = int(f.get("components_count", 1)); extent = float(f.get("extent", 0.0))
            grad_p90 = float(f.get("grad_p90", 0.0))

            g_light = _clamp(0.5 * _safe_div(mean_L, 255.0) + 0.5 * _clamp(std_L / 64.0))
            g_color = _clamp(np.hypot(delta_a, delta_b) / 200.0)
            g_shape = _clamp(0.4 * circ + 0.4 * solid + 0.2 * (1.0 - _clamp(ecc)))
            g_embed = float(embed_sim) if have_mu else 0.5
            g_robust = _clamp(0.6 * (1.0 if compc == 1 else 0.6) + 0.2 * (1.0 - abs(extent - 0.5) * 2.0) + 0.2 * _clamp(grad_p90 / 200.0))
            g_robust *= (0.7 if touch else 1.0)
            if PCA_NAMES:
                d = float(np.linalg.norm(np.array(pca_vals, dtype=np.float32)))
                g_maha = _clamp(1.0 - (d / (d + 5.0)))
            else:
                g_maha = 0.5
            g_border = 1.0 if touch else 0.0
            parts = [g_light, g_color, g_shape, g_embed, g_robust, g_maha, (1.0 - 0.5 * g_border)]
            g_quality = _clamp(sum(parts) / len(parts))

            row = []
            row += [file_name, ann_id_str, s_str]
            row += [int(f.get("area_px", 0)), float(f.get("area_norm", 0.0))]
            row += [int(f.get("bbox_x", 0)), int(f.get("bbox_y", 0)), int(f.get("bbox_w", 0)), int(f.get("bbox_h", 0))]
            row += [float(f.get("cx", 0.0)), float(f.get("cy", 0.0))]
            row += [float(f.get("mean_L", 0.0)), float(f.get("mean_a", 0.0)), float(f.get("mean_b", 0.0)),
                    float(f.get("std_L", 0.0)), float(f.get("std_a", 0.0)), float(f.get("std_b", 0.0)),
                    float(f.get("median_L", 0.0)), float(f.get("median_a", 0.0)), float(f.get("median_b", 0.0))]
            row += [float(f.get("delta_a", 0.0)), float(f.get("delta_b", 0.0))]
            row += [float(f.get("elongation", 0.0)), float(f.get("eccentricity", 0.0)), float(f.get("solidity", 0.0)),
                    float(f.get("aspect_ratio", 0.0)), float(f.get("circularity", 0.0)), float(f.get("extent", 0.0)),
                    float(f.get("perimeter", 0.0))]
            row += [float(f.get("bbox_diag_frac", 0.0)), float(f.get("perim_over_sqrt_area", 0.0)), float(f.get("scale_diag", 0.0))]
            row += [float(f.get("hu1", 0.0)), float(f.get("hu2", 0.0)), float(f.get("hu4", 0.0)),
                    int(f.get("components_count", 1)), float(f.get("holes_ratio", 0.0)), int(f.get("touching_border", 0))]
            row += [float(f.get("LBP_u5", 0.0)), float(f.get("GLCM_contrast", 0.0)), float(f.get("GLCM_homogeneity", 0.0)),
                    float(f.get("grad_mean", 0.0)), float(f.get("grad_p90", 0.0))]
            row += [int(f.get("grid_r", 0)), int(f.get("grid_c", 0)),
                    float(f.get("grid_r_norm", 0.0)), float(f.get("grid_c_norm", 0.0))]
            row += [float(pred_iou), float(stability), float(embed_sim)]
            row += [float(g_quality), float(g_light), float(g_color), float(g_shape),
                    float(g_embed), float(g_robust), float(g_maha), float(g_border)]
            if feature_mode in ("rich", "ultra"):
                row += [float(f.get("delta_b_med", 0.0)), float(f.get("delta_a_med", 0.0)),
                        float(f.get("median_b_in", 0.0)), float(f.get("median_b_ring", 0.0)),
                        float(f.get("median_a_in", 0.0)), float(f.get("median_a_ring", 0.0)),
                        float(f.get("histb_q1", 0.0)), float(f.get("histb_q2", 0.0)),
                        float(f.get("histb_q3", 0.0)), float(f.get("histb_q4", 0.0)),
                        float(f.get("histab_q11", 0.0)), float(f.get("histab_q12", 0.0)),
                        float(f.get("histab_q21", 0.0)), float(f.get("histab_q22", 0.0))]
            if PCA_NAMES:
                row += [float(v) for v in pca_vals]
            row += [int(reviewed_flag), review_tag]
            cid = int(ann.get("category_id", -1))
            row += [cid, str(id_to_cat.get(cid, "")).strip(), int(cid in pos_cat_ids)]
            new_rows.append(row)
            new_keys.append((file_name, ann_id_str, s_str))

    if failed_images:
        raise FeatureInputError(
            f"{len(failed_images)} tile image(s) could not be read; no feature row and no resume key "
            "was written (supply --map OLD=NEW for relocated inputs, then rerun features)",
            {"failed_images": failed_images, "input_resolution": resolver.summary(),
             "rows_not_written": len(new_rows), "published": False})
    written_path = None
    if new_rows or not exists_before:
        out_path = state.ensure_overlay_copy(out_csv_rel) if exists_before else state.write_path(out_csv_rel)
        mode = "a" if exists_before else "w"
        with out_path.open(mode, newline="", encoding="utf-8") as fcsv:
            w = _csv.writer(fcsv)
            if not exists_before:
                w.writerow(header)
            for row in new_rows:
                w.writerow(row)
        written_path = str(out_path)
    prov = {
        "schema": "compag-only-codes-feature-provenance/v1",
        "csv": out_csv_rel,
        "existing_rows_source": state.source_of(out_csv_rel) if not new_rows else "ORIGINAL_ROWS_PRESERVED_THEN_APPENDED",
        "existing_done_keys": len(done_triples),
        "appended_rows": len(new_rows),
        "appended_keys_sha256": hashlib.sha256(json.dumps(new_keys).encode()).hexdigest(),
        "migrations": migrated,
        "review_tag_column": tag_col,
        "transform_identity": transform_identity or {},
        "historical_rows_transform": "UNVERIFIED (legacy cache has no transform hash; rows imported as-is)",
        "embed_cpu_fallbacks": embed_fallbacks,
        "unreadable_images": unreadable,
        "input_resolution": resolver.summary(),
        "pos": n_pos, "neg": n_neg,
    }
    sidecar = state.write_path(out_csv_rel + ".only_codes_provenance.json")
    history = []
    if sidecar.exists():
        try:
            history = json.loads(sidecar.read_text(encoding="utf-8")).get("history", [])
        except Exception:
            history = []
    history.append(prov)
    sidecar.write_text(json.dumps({"history": history}, indent=2, sort_keys=True), encoding="utf-8")
    return {**prov, "written_path": written_path}
