# gate_core.py
# -*- coding: utf-8 -*-
"""
Core utilities for JASSID gates & features.
This module is intentionally self-contained and **must not** import itself.

Exposed API (used by sam2_and_filter.py):
- warp_card_to_rect(img_bgr) -> (warped_bgr, Minv)
- detect_grid_lines(card_bgr) -> (row_lines, col_lines)
- cell_of_point(x,y,row_lines,col_lines) -> (r,c)
- compute_features(mask, lab, mode='balanced') -> Dict[str, float]
- yellow_bg_stats(card_bgr) -> (bg_a, bg_b, bg_bgr)
- build_embed_model(backbone, weights_path) -> (model, transform)
  [optional; explicit local weights, used only if embed proto provided]
- crop_masked_patch(card_bgr, mask, pad_frac=0.10, bg_bgr=(255,255,255)) -> np.ndarray|None
- embed_patch(patch_bgr, model, transform, device='cpu') -> 1D np.ndarray (unit length)
- load_embed_csv(path) -> (headers, {'mu_embed': np.ndarray, 'embed_thresh': float})
- load_unified_proto_csv(path) -> (feat_names, {'mu': np.ndarray, 'mad': np.ndarray, 'cov': np.ndarray})
- load_embed_pca(path) -> {'components': (k,D) np.ndarray, 'mean': (D,), 'names': [str,...]}
- project_embed_pca(z, pca_dict) -> (k,) np.ndarray
"""

from typing import Dict, Any, BinaryIO, List, Tuple, Optional
from pathlib import Path
import json, math, csv as _csv

import numpy as np
import cv2

# ============ Geometry & pre-processing ============
def mask_to_poly_and_bbox_in_original(mask_card, inverse_matrix, original_shape):
    """Map the largest full-mask contour into original-image coordinates."""

    height, width = int(original_shape[0]), int(original_shape[1])
    binary = (np.asarray(mask_card) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], (0, 0, 0, 0)
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    points = contour.reshape(-1, 1, 2).astype(np.float32)
    mapped = cv2.perspectiveTransform(points, np.asarray(inverse_matrix, dtype=np.float32)).reshape(-1, 2)
    mapped[:, 0] = np.clip(mapped[:, 0], 0, width - 1)
    mapped[:, 1] = np.clip(mapped[:, 1], 0, height - 1)
    polygon = np.round(mapped).astype(np.int32)
    x, y, box_width, box_height = cv2.boundingRect(polygon)
    return polygon.reshape(-1).tolist(), (
        int(x), int(y), int(box_width), int(box_height)
    )


def warp_card_to_rect(img_bgr):
    """Best effort: try to find the yellow card quad; fall back to identity warp."""
    try:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([15, 40, 60], np.uint8)
        upper = np.array([40, 255, 255], np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise RuntimeError("no yellow region")
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            pts = approx.reshape(-1,2).astype(np.float32)
            # order: tl,tr,br,bl
            s = pts.sum(axis=1); diff = np.diff(pts, axis=1).reshape(-1)
            tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
            tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
            src = np.array([tl,tr,br,bl], np.float32)
            def dist(a,b): return float(np.linalg.norm(a-b))
            W = int(max(dist(tr,tl), dist(br,bl))); H = int(max(dist(tr,br), dist(tl,bl)))
            W = max(W, 1); H = max(H, 1)
            dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)
            M  = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(img_bgr, M, (W,H))
            Minv = cv2.getPerspectiveTransform(dst, src)
            return warped, Minv
    except Exception:
        pass
    H,W = img_bgr.shape[:2]
    return img_bgr.copy(), np.eye(3, dtype=np.float32)

# ============================================================
# Dashed-grid localisation (spacing-constrained, sweep)
# ------------------------------------------------------------
# Replaces the old Hough-based grid detector because dashed lines often
# break Hough + insects/noise make it unstable.
#
# Mirrors your working standalone script:
# - Darkness map (local background - gray)
# - Percentile threshold -> BW ink mask
# - 1D projections (row/col sums) + smoothing
# - Search best 5 horizontal + 3 vertical peaks under expected spacing
# - If constraints fail, sweep p_thresh + border_frac and pick best.
# ============================================================

# --- measured on a reference warped card (keep these in sync with your calibration) ---
_GRID_BASE_W = 2173
_GRID_BASE_H = 3692
_GRID_DX_PX = 730.0   # vertical-to-vertical spacing at _GRID_BASE_W
_GRID_DY_PX = 744.0   # horizontal-to-horizontal spacing at _GRID_BASE_H
_GRID_DX_RATIO = _GRID_DX_PX / float(_GRID_BASE_W)
_GRID_DY_RATIO = _GRID_DY_PX / float(_GRID_BASE_H)

# Expected number of dashed grid lines in the warped card
_GRID_EXPECTED_V = 3
_GRID_EXPECTED_H = 5

# Darkness map params
_GRID_SIGMA_BG_FRAC = 0.02
_GRID_SIGMA_BG_MIN = 10
_GRID_T_MIN = 10
_GRID_T_MAX = 80

# Projection smoothing + peak picking/search
_GRID_SMOOTH_SIGMA = 6.0
_GRID_POS_SEARCH_RADIUS_PX = 28
_GRID_MIN_SEPARATION_PX = 6

# Spacing constraints
_GRID_DX_TOL_FRAC = 0.15
_GRID_DY_TOL_FRAC = 0.15

# Sweep lists (robust to lighting / border artifacts)
_GRID_BORDER_FRAC_LIST = [0.05, 0.04, 0.03, 0.02]
_GRID_P_THRESH_LIST = [92.0, 90.0, 88.0, 85.0, 82.0, 80.0]

# Optional margin prior (disabled by default; keep weight=0 unless you want it)
_GRID_LEFT_MARGIN_RATIO = 133.0 / float(_GRID_BASE_W)
_GRID_TOP_MARGIN_RATIO  = 412.0 / float(_GRID_BASE_H)
_GRID_PRIOR_WEIGHT = 0.0

# Acceptance based on normalized peak strength
_GRID_MIN_MEAN_NORM_SCORE_H = 0.10
_GRID_MIN_MEAN_NORM_SCORE_V = 0.10


def _grid_gaussian1d_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x.astype(np.float32)
    k = int(round(sigma * 6))
    k = max(k, 3)
    if k % 2 == 0:
        k += 1
    x_col = x.reshape(-1, 1).astype(np.float32)
    y = cv2.GaussianBlur(x_col, (1, k), sigmaX=0, sigmaY=sigma)
    return y.reshape(-1)


def _grid_compute_darkness(img_bgr: np.ndarray, sigma_bg: float) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma_bg, sigmaY=sigma_bg)
    dark = cv2.subtract(bg, gray)  # uint8
    return dark


def _grid_threshold_darkness(dark: np.ndarray, p_thresh: float) -> int:
    T = int(np.percentile(dark.reshape(-1), p_thresh))
    return int(np.clip(T, _GRID_T_MIN, _GRID_T_MAX))


def _grid_apply_border_mask(bw01: np.ndarray, border_frac: float) -> Tuple[np.ndarray, Dict[str, int]]:
    H, W = bw01.shape[:2]
    mx = int(round(W * border_frac))
    my = int(round(H * border_frac))
    out = bw01.copy()
    if mx > 0:
        out[:, :mx] = 0
        out[:, W - mx:] = 0
    if my > 0:
        out[:my, :] = 0
        out[H - my:, :] = 0
    return out, {"mx": mx, "my": my}


def _grid_comb_search_positions(
    score: np.ndarray,
    start_min: int,
    start_max: int,
    n_lines: int,
    spacing: float,
    search_radius: int,
    min_separation: int,
    prior_start: Optional[float],
    prior_weight: float,
) -> Tuple[float, int, List[int], List[float]]:
    """Search integer start position and pick n_lines maxima near start+i*spacing."""
    L = int(len(score))
    best_total = -1e18
    best_start = int(start_min)
    best_pos: List[int] = []
    best_peaks: List[float] = []

    for s in range(int(start_min), int(start_max) + 1):
        pos: List[int] = []
        peaks: List[float] = []
        total = 0.0
        last = -10**9
        ok = True

        for i in range(int(n_lines)):
            p = float(s) + float(i) * float(spacing)
            p_int = int(round(p))
            a = max(0, p_int - int(search_radius))
            b = min(L - 1, p_int + int(search_radius))
            if b < a:
                ok = False
                break

            local = score[a:b + 1]
            if local.size == 0:
                ok = False
                break

            idx = int(a + int(np.argmax(local)))
            if idx <= last + int(min_separation):
                ok = False
                break

            last = idx
            pos.append(idx)
            pk = float(score[idx])
            peaks.append(pk)
            total += pk

        if not ok:
            continue

        if prior_start is not None and prior_weight > 0:
            total += float(prior_weight) * (-abs(float(s) - float(prior_start)) / (float(spacing) + 1e-6))

        if total > best_total:
            best_total = float(total)
            best_start = int(s)
            best_pos = pos
            best_peaks = peaks

    return float(best_total), int(best_start), best_pos, best_peaks


def _grid_validate_spacing_all(pos: List[int], expected_spacing: float, tol_frac: float) -> Tuple[bool, List[int], Tuple[float, float]]:
    if len(pos) < 2:
        return False, [], (0.0, 0.0)
    diffs = [int(pos[i + 1] - pos[i]) for i in range(len(pos) - 1)]
    lo = float(expected_spacing) * (1.0 - float(tol_frac))
    hi = float(expected_spacing) * (1.0 + float(tol_frac))
    ok = all(lo <= float(d) <= hi for d in diffs)
    return bool(ok), diffs, (float(lo), float(hi))


def _grid_mean_norm_strength(peaks: List[float], score: np.ndarray) -> float:
    med = float(np.median(score))
    p95 = float(np.percentile(score, 95))
    denom = max(1e-6, p95 - med)
    norms = [(float(p) - med) / denom for p in peaks]
    return float(np.mean(norms)) if norms else 0.0


def _grid_one_pass(img_bgr: np.ndarray, dark: np.ndarray, p_thresh: float, border_frac: float,
                   expected_h: int, expected_v: int) -> Dict[str, Any]:
    H, W = img_bgr.shape[:2]
    dx = _GRID_DX_RATIO * float(W)
    dy = _GRID_DY_RATIO * float(H)

    T = _grid_threshold_darkness(dark, p_thresh)
    bw = (dark > T).astype(np.uint8)  # 0/1

    bw, border = _grid_apply_border_mask(bw, border_frac=border_frac)
    mx, my = int(border["mx"]), int(border["my"])
    x1, x2 = mx, W - mx - 1
    y1, y2 = my, H - my - 1

    row_sum = bw[:, x1:x2 + 1].sum(axis=1).astype(np.float32)
    col_sum = bw[y1:y2 + 1, :].sum(axis=0).astype(np.float32)

    row_score = _grid_gaussian1d_smooth(row_sum, sigma=_GRID_SMOOTH_SIGMA)
    col_score = _grid_gaussian1d_smooth(col_sum, sigma=_GRID_SMOOTH_SIGMA)

    y0_min = y1
    y0_max = int(y2 - (expected_h - 1) * dy)
    x0_min = x1
    x0_max = int(x2 - (expected_v - 1) * dx)

    if y0_max < y0_min or x0_max < x0_min:
        return {"ok": False, "reason": "image too small for expected spacing"}

    prior_y0 = _GRID_TOP_MARGIN_RATIO * float(H)
    prior_x0 = _GRID_LEFT_MARGIN_RATIO * float(W)

    h_total, h_start, h_pos, h_peaks = _grid_comb_search_positions(
        row_score, y0_min, y0_max,
        n_lines=expected_h,
        spacing=dy,
        search_radius=_GRID_POS_SEARCH_RADIUS_PX,
        min_separation=_GRID_MIN_SEPARATION_PX,
        prior_start=prior_y0,
        prior_weight=_GRID_PRIOR_WEIGHT,
    )

    v_total, v_start, v_pos, v_peaks = _grid_comb_search_positions(
        col_score, x0_min, x0_max,
        n_lines=expected_v,
        spacing=dx,
        search_radius=_GRID_POS_SEARCH_RADIUS_PX,
        min_separation=_GRID_MIN_SEPARATION_PX,
        prior_start=prior_x0,
        prior_weight=_GRID_PRIOR_WEIGHT,
    )

    ok_h, h_diffs, h_rng = _grid_validate_spacing_all(h_pos, dy, _GRID_DY_TOL_FRAC)
    ok_v, v_diffs, v_rng = _grid_validate_spacing_all(v_pos, dx, _GRID_DX_TOL_FRAC)

    strength_h = _grid_mean_norm_strength(h_peaks, row_score)
    strength_v = _grid_mean_norm_strength(v_peaks, col_score)
    ok_strength = (strength_h >= _GRID_MIN_MEAN_NORM_SCORE_H) and (strength_v >= _GRID_MIN_MEAN_NORM_SCORE_V)

    ok = bool(ok_h and ok_v and ok_strength)

    quality = float(h_total + v_total + 1000.0 * (strength_h + strength_v))
    if not ok_h:
        quality -= 1e6
    if not ok_v:
        quality -= 1e6
    if not ok_strength:
        quality -= 5e5

    return {
        "ok": ok,
        "quality": quality,
        "H": int(H), "W": int(W),
        "dx_expected": float(dx),
        "dy_expected": float(dy),
        "T_dark": int(T),
        "p_thresh": float(p_thresh),
        "border_frac": float(border_frac),
        "border": border,
        "h": {"pos": h_pos, "peaks": h_peaks, "diffs": h_diffs, "range": h_rng, "total": float(h_total),
              "strength_mean_norm": float(strength_h), "start": int(h_start)},
        "v": {"pos": v_pos, "peaks": v_peaks, "diffs": v_diffs, "range": v_rng, "total": float(v_total),
              "strength_mean_norm": float(strength_v), "start": int(v_start)},
        "debug": {"bw": bw},
    }


def _grid_run_sweep(img_bgr: np.ndarray, expected_h: int, expected_v: int) -> Dict[str, Any]:
    H, W = img_bgr.shape[:2]
    sigma_bg = max(_GRID_SIGMA_BG_MIN, int(round(min(H, W) * _GRID_SIGMA_BG_FRAC)))
    dark = _grid_compute_darkness(img_bgr, sigma_bg=float(sigma_bg))

    best_ok = None
    best_any = None

    for p in _GRID_P_THRESH_LIST:
        for bf in _GRID_BORDER_FRAC_LIST:
            res = _grid_one_pass(img_bgr, dark, p_thresh=float(p), border_frac=float(bf),
                                 expected_h=int(expected_h), expected_v=int(expected_v))
            res["sigma_bg"] = int(sigma_bg)

            if best_any is None or res.get("quality", -1e18) > best_any.get("quality", -1e18):
                best_any = res

            if res.get("ok", False):
                if best_ok is None or res["quality"] > best_ok["quality"]:
                    best_ok = res

    return best_ok if best_ok is not None else best_any


def detect_grid_lines(card_bgr, expected_h: int = _GRID_EXPECTED_H, expected_v: int = _GRID_EXPECTED_V):
    """Detect dashed printed grid lines on the *warped* card image.

    Returns:
      row_lines: list[int]  (y positions, sorted)
      col_lines: list[int]  (x positions, sorted)
    """
    try:
        chosen = _grid_run_sweep(card_bgr, expected_h=int(expected_h), expected_v=int(expected_v))
        if chosen is None:
            return [], []
        row_lines = [int(y) for y in chosen["h"]["pos"]]
        col_lines = [int(x) for x in chosen["v"]["pos"]]
        row_lines.sort()
        col_lines.sort()
        return row_lines, col_lines
    except Exception:
        # Keep the pipeline robust: return empty if something unexpected happens.
        return [], []

def cell_of_point(x,y,row_lines,col_lines):
    rows = [-1]+row_lines+[10**9]; cols = [-1]+col_lines+[10**9]
    r=c=0
    for i in range(len(rows)-1):
        if rows[i]<y<=rows[i+1]:
            r=i; break
    for j in range(len(cols)-1):
        if cols[j]<x<=cols[j+1]:
            c=j; break
    return r,c

# ============ Feature extraction ============
def _lbp_uniform(img_u8, radius=1, neighbors=8):
    """Uniform LBP histogram (59 bins). Returns scalar uniform-count (cheap proxy)."""
    try:
        H,W = img_u8.shape
        ys, xs = np.mgrid[-radius:radius+1, -radius:radius+1]
        ys = ys.astype(np.int32); xs = xs.astype(np.int32)
        lbp = np.zeros_like(img_u8, dtype=np.uint8)
        for n,(dy,dx) in enumerate(zip(ys.flatten(), xs.flatten())):
            if dy==0 and dx==0: continue
            shifted = np.roll(np.roll(img_u8, dy, axis=0), dx, axis=1)
            lbp = lbp + ((shifted >= img_u8).astype(np.uint8)<<n)
        # crude uniform approx: how many neighbors exceeded center on average
        return float(lbp.mean())
    except Exception:
        return 0.0

def _glcm_contrast_homogeneity(gray_u8):
    try:
        # extremely light-weight proxy using gradients
        gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1,0, ksize=3)
        gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0,1, ksize=3)
        g = np.hypot(gx,gy)
        contrast = float(np.mean(g))
        hom = float(1.0/(1.0+np.var(g)))
        return contrast, hom
    except Exception:
        return 0.0, 0.0

def _quantiles(v: np.ndarray, qs=(0.25,0.5,0.75,0.9)):
    if v.size == 0: return [0.0]*len(qs)
    return [float(np.quantile(v, q)) for q in qs]

def compute_features(mask: np.ndarray, lab: np.ndarray, mode: str='balanced') -> Dict[str, float]:
    """
    Compute geometry + basic color + a few texture cues.
    mode: explicit 'balanced' | 'rich' | 'ultra'
    """
    mode = mode if mode in ('balanced', 'rich', 'ultra') else 'balanced'
    cnts,_ = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return {}
    c = max(cnts, key=cv2.contourArea)
    A = float(cv2.contourArea(c))
    x,y,w,h = cv2.boundingRect(c)
    P = float(cv2.arcLength(c, True))
    aspect = w/float(h+1e-6)
    hull = cv2.convexHull(c); hullA = float(cv2.contourArea(hull)) if hull is not None else A
    solidity = float(A/(hullA+1e-6))
    circularity = float((4*math.pi*A)/((P+1e-6)**2)) if P>0 else 0.0

    M = cv2.moments(c)
    cx = float(M["m10"]/M["m00"]) if M["m00"]!=0 else (x+w/2)
    cy = float(M["m01"]/M["m00"]) if M["m00"]!=0 else (y+h/2)

    # elongation via PCA of contour points
    pts = c.reshape(-1,2).astype(np.float32)
    if len(pts)>=5:
        mean,_ = cv2.PCACompute(pts, mean=None)
        ptsc = pts - mean
        cov = np.cov(ptsc.T); evals,_ = np.linalg.eig(cov); evals=np.sort(evals)[::-1]
        elong = float((evals[0]+1e-6)/(evals[-1]+1e-6)) if len(evals)>=2 else 1.0
        ecc = float(np.sqrt(1.0 - (evals[-1]+1e-6)/(evals[0]+1e-6))) if len(evals)>=2 else 0.0
    else:
        elong = 1.0; ecc = 0.0

    L,Aa,Bb = cv2.split(lab); roi = mask>0
    meanL = float(np.mean(L[roi])) if np.any(roi) else 0.0
    meana = float(np.mean(Aa[roi])) if np.any(roi) else 0.0
    meanb = float(np.mean(Bb[roi])) if np.any(roi) else 0.0
    stdL  = float(np.std(L[roi])) if np.any(roi) else 0.0
    stda  = float(np.std(Aa[roi])) if np.any(roi) else 0.0
    stdb  = float(np.std(Bb[roi])) if np.any(roi) else 0.0
    medL  = float(np.median(L[roi])) if np.any(roi) else 0.0
    meda  = float(np.median(Aa[roi])) if np.any(roi) else 0.0
    medb  = float(np.median(Bb[roi])) if np.any(roi) else 0.0

    H,W = mask.shape
    touching_border = int(x<=0 or y<=0 or (x+w)>=W-1 or (y+h)>=H-1)

    # texture / gradients
    gray = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (3,3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1,0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0,1, ksize=3)
    mag = np.hypot(gx, gy)[roi]
    grad_mean = float(np.mean(mag)) if mag.size else 0.0
    grad_p90  = float(np.quantile(mag, 0.90)) if mag.size else 0.0

    # cheap LBP/GLCM proxies on the mask ROI
    gray_roi = gray.copy()
    gray_roi[~roi] = gray[~roi].mean() if gray.size else 0
    LBP_u5 = _lbp_uniform(gray_roi, radius=1, neighbors=8)
    GLCM_contrast, GLCM_homogeneity = _glcm_contrast_homogeneity(gray_roi)

    # base dict
    out = {
        "area_px": int(A),
        "area_norm": float(A/(H*W+1e-6)),
        "bbox_x": int(x), "bbox_y": int(y), "bbox_w": int(w), "bbox_h": int(h),
        "cx": float(cx), "cy": float(cy),
        "mean_L": meanL, "mean_a": meana, "mean_b": meanb,
        "std_L": stdL, "std_a": stda, "std_b": stdb,
        "median_L": medL, "median_a": meda, "median_b": medb,
        "elongation": float(elong), "eccentricity": float(ecc),
        "solidity": float(solidity), "aspect_ratio": float(aspect),
        "circularity": float(circularity), "extent": float(A/float(w*h+1e-6)),
        "perimeter": float(P), "perim_sqrt": float(np.sqrt(P+1e-6)),
        "hu1": 0.0, "hu2": 0.0, "hu4": 0.0,  # optional; keep placeholders
        "components_count": 1, "holes_ratio": 0.0,
        "touching_border": int(touching_border),
        "LBP_u5": float(LBP_u5),
        "GLCM_contrast": float(GLCM_contrast), "GLCM_homogeneity": float(GLCM_homogeneity),
        "grad_mean": float(grad_mean), "grad_p90": float(grad_p90),
        # grid_r/grid_c filled later by caller
        "grid_r": 0, "grid_c": 0,
    }
    
    # --- scale-aware, mostly scale-invariant extras ---
    card_diag = float((H**2 + W**2) ** 0.5)
    bbox_diag = float((w**2 + h**2) ** 0.5)
    bbox_diag_frac = float(bbox_diag / (card_diag + 1e-6))
    perim_over_sqrt_area = float(P / (A**0.5 + 1e-6)) if (P > 0 and A > 0) else 0.0
    out.update({
        "bbox_diag_frac": bbox_diag_frac,
        "perim_over_sqrt_area": perim_over_sqrt_area,
        "scale_diag": card_diag,
    })


    if mode in ('rich','ultra'):
        # ring medians (thin dilation border)
        rp = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rp+1, 2*rp+1))
        dil = cv2.dilate((mask>0).astype(np.uint8), kernel)
        ring = (dil.astype(bool) & (~(mask>0))).astype(np.uint8)
        a_in = Aa[mask>0]; a_ring = Aa[ring>0] if np.any(ring) else Aa[mask==0]
        b_in = Bb[mask>0]; b_ring = Bb[ring>0] if np.any(ring) else Bb[mask==0]
        med_ai = float(np.median(a_in)) if a_in.size else 0.0
        med_ar = float(np.median(a_ring)) if a_ring.size else 0.0
        med_bi = float(np.median(b_in)) if b_in.size else 0.0
        med_br = float(np.median(b_ring)) if b_ring.size else 0.0
        out.update({
            "median_a_in": med_ai, "median_a_ring": med_ar,
            "median_b_in": med_bi, "median_b_ring": med_br,
            "delta_a_med": float(med_ai - med_ar),
            "delta_b_med": float(med_bi - med_br),
        })
        # compact histograms (b channel and 2x2 (a,b) quadrants)
        b_vals = b_in.astype(np.float32) if b_in.size else np.array([0], np.float32)
        q1,q2,q3,q4 = _quantiles(b_vals, (0.2,0.4,0.6,0.8))
        out.update({"histb_q1":q1,"histb_q2":q2,"histb_q3":q3,"histb_q4":q4})
        # 2x2 bins on (a,b) centered at (med_a, med_b)
        ca, cb = med_ai, med_bi
        qa11 = float(np.mean((a_in<=ca)&(b_in<=cb))) if a_in.size else 0.0
        qa12 = float(np.mean((a_in<=ca)&(b_in> cb))) if a_in.size else 0.0
        qa21 = float(np.mean((a_in> ca)&(b_in<=cb))) if a_in.size else 0.0
        qa22 = float(np.mean((a_in> ca)&(b_in> cb))) if a_in.size else 0.0
        out.update({"histab_q11":qa11,"histab_q12":qa12,"histab_q21":qa21,"histab_q22":qa22})

    return out

def yellow_bg_stats(card_bgr: np.ndarray):
    """Estimate yellow background median (a,b) and return an RGB sample of bg for padding."""
    lab = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2LAB)
    H,W = lab.shape[:2]
    border = max(2, int(min(H,W)*0.05))
    ring = np.zeros((H,W), np.uint8)
    ring[:border,:] = 1; ring[-border:,:] = 1; ring[:,:border] = 1; ring[:,-border:] = 1
    a = lab[:,:,1]; b = lab[:,:,2]
    a_vals = a[ring>0]; b_vals = b[ring>0]
    med_a = float(np.median(a_vals)) if a_vals.size else 0.0
    med_b = float(np.median(b_vals)) if b_vals.size else 0.0
    # bg color sample (BGR) from corner
    bg_bgr = card_bgr[0:5,0:5].reshape(-1,3).mean(axis=0).astype(np.uint8).tolist()
    return med_a, med_b, tuple(int(x) for x in bg_bgr)

# ============ Embedding helpers (optional) ============
def build_embed_model(backbone: str, weights_path: str | Path | BinaryIO):
    """Build an embedding model from an explicit local state dictionary.

    Every torchvision constructor receives ``weights=None``.  Consequently no
    registered model name or default-weight enum can initiate a download. Path
    callers remain supported; a descriptor-backed binary stream can be used to
    keep an upstream verified-file context pinned through ``torch.load``.
    """
    import os
    import stat

    normalized_backbone = str(backbone).strip().lower()
    if normalized_backbone not in {"resnet18", "resnet34", "resnet50", "resnet101"}:
        raise ValueError("embedding backbone must be explicitly selected")

    import torch
    import torchvision.transforms as T
    from torchvision import models

    constructors = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
    }
    m = constructors[normalized_backbone](weights=None)

    def identity(info):
        return (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_uid,
            info.st_gid,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    opened_weights = None
    local_weights = None
    path_before = None
    if isinstance(weights_path, (str, os.PathLike)):
        local_weights = Path(weights_path)
        path_before = local_weights.lstat()
        if (
            not stat.S_ISREG(path_before.st_mode)
            or stat.S_ISLNK(path_before.st_mode)
            or path_before.st_nlink != 1
        ):
            raise ValueError(
                "embedding backbone weights must be a single-link regular local file"
            )
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        descriptor = os.open(local_weights, flags)
        try:
            opened_weights = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
    else:
        try:
            source_descriptor = weights_path.fileno()
            source_before = os.fstat(source_descriptor)
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise ValueError(
                "embedding backbone weights must be a descriptor-backed binary file"
            ) from exc
        duplicate = os.dup(source_descriptor)
        try:
            opened_weights = os.fdopen(duplicate, "rb")
        except BaseException:
            os.close(duplicate)
            raise
        if identity(os.fstat(opened_weights.fileno())) != identity(source_before):
            opened_weights.close()
            raise ValueError("embedding backbone weights changed while duplicating")

    weights_file = opened_weights

    try:
        descriptor = weights_file.fileno()
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(
                "embedding backbone weights must be a single-link regular local file"
            )
        if before.st_size > 256 * 1024 * 1024:
            raise ValueError("embedding backbone weights exceed the supported size bound")
        if path_before is not None and identity(before) != identity(path_before):
            raise ValueError("embedding backbone weights changed while opening")

        weights_file.seek(0)
        state = torch.load(weights_file, map_location="cpu", weights_only=True)

        after = os.fstat(descriptor)
        if identity(after) != identity(before):
            raise ValueError("embedding backbone weights changed while loading")
        if local_weights is not None:
            path_after = local_weights.lstat()
            if identity(path_after) != identity(before):
                raise ValueError("embedding backbone weights path changed while loading")
    finally:
        opened_weights.close()

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    m.load_state_dict(state, strict=True)

    m.fc = torch.nn.Identity()
    m.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return m, transform

def crop_masked_patch(card_bgr: np.ndarray, mask: np.ndarray, pad_frac: float=0.10, bg_bgr=(255,255,255)):
    ys,xs = np.where(mask>0)
    if xs.size==0 or ys.size==0:
        return None
    x0,x1 = int(xs.min()), int(xs.max())
    y0,y1 = int(ys.min()), int(ys.max())
    H,W = mask.shape
    ph = int((y1-y0+1)*pad_frac); pw = int((x1-x0+1)*pad_frac)
    x0 = max(0, x0-pw); y0 = max(0, y0-ph)
    x1 = min(W-1, x1+pw); y1 = min(H-1, y1+ph)
    crop = card_bgr[y0:y1+1, x0:x1+1].copy()
    m = mask[y0:y1+1, x0:x1+1]
    bg = np.full_like(crop, bg_bgr, dtype=np.uint8)
    crop[~(m>0)] = bg[~(m>0)]
    return crop

def embed_patch(patch_bgr, model, transform, device='cpu'):
    import torch
    with torch.no_grad():
        x = transform(cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        z = model(x).detach().cpu().numpy().reshape(-1).astype(np.float32)
    z /= (np.linalg.norm(z)+1e-12)
    return z

# ============ Proto IO ============
def _load_kv_csv(path: Path) -> Dict[str, Any]:
    D: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        r = _csv.DictReader(f)
        headers = set(r.fieldnames or [])
        if headers == {"key","value"}:
            for row in r:
                D[row["key"]] = row["value"]
        else:
            rows = list(r)
            if len(rows)>=1:
                D.update(rows[0])
    return D

def load_unified_proto_csv(path: Path) -> Tuple[List[str], Dict[str, Any]]:
    D = _load_kv_csv(path)

    # embed
    embed_thresh = float(D.get("embed_thresh", "0.70"))
    mu_embed = []
    i = 0
    while True:
        k = f"mu_embed_{i:04d}"
        if k not in D: break
        try: mu_embed.append(float(D[k]))
        except: mu_embed.append(0.0)
        i += 1
    mu_embed = np.array(mu_embed, dtype=np.float32) if mu_embed else None
    if mu_embed is not None and mu_embed.size:
        mu_embed = mu_embed / (np.linalg.norm(mu_embed)+1e-12)

    # features
    feat_names = [s.strip() for s in D.get("feat_names","").split(",") if s.strip()]
    K = len(feat_names)
    mu  = np.array([float(D.get(f"mu_feat_{i:03d}", "0"))  for i in range(K)], dtype=np.float64) if K else np.array([])
    mad = np.array([float(D.get(f"mad_feat_{i:03d}", "1e-6")) for i in range(K)], dtype=np.float64) if K else np.array([])

    cov = None
    if K:
        cov = np.zeros((K,K), dtype=np.float64)
        have_cov=True
        for i in range(K):
            for j in range(K):
                key = f"cov_feat_{i:03d}_{j:03d}"
                if key not in D: have_cov=False; break
                cov[i,j] = float(D[key])
            if not have_cov: break
        if not have_cov: cov = None

    pack = {"mu": mu, "mad": mad, "cov": cov, "mu_embed": mu_embed, "embed_thresh": embed_thresh}
    return feat_names, pack

def load_embed_csv(path: Path):
    """CSV with two cols key,value containing embed proto fields (mu_embed_0000..., embed_thresh)."""
    D = _load_kv_csv(path)
    mu = []
    i = 0
    while True:
        k = f"mu_embed_{i:04d}"
        if k not in D: break
        try: mu.append(float(D[k]))
        except: mu.append(0.0)
        i += 1
    mu = np.array(mu, dtype=np.float32) if mu else None
    if mu is not None and mu.size:
        mu = mu / (np.linalg.norm(mu)+1e-12)
    return list(D.keys()), {"mu_embed": mu, "embed_thresh": float(D.get("embed_thresh", "0.70"))}

# ============ PCA helpers ============
def load_embed_pca(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".npz":
        d = np.load(str(p))
        comps = d["components"].astype(np.float32)
        mean = d["mean"].astype(np.float32) if "mean" in d else np.zeros((comps.shape[1],), np.float32)
    elif p.suffix.lower() == ".npy":
        comps = np.load(str(p)).astype(np.float32)
        mean = np.zeros((comps.shape[1],), np.float32)
    elif p.suffix.lower() == ".json":
        J = json.loads(Path(p).read_text(encoding="utf-8"))
        comps = np.array(J["components"], dtype=np.float32)
        mean = np.array(J.get("mean",[0]*comps.shape[1]), dtype=np.float32)
    else:
        raise ValueError("Unsupported PCA format. Use .npz/.npy/.json")
    names = [f"embed_pca_{i}" for i in range(comps.shape[0])]
    return {"components": comps, "mean": mean, "names": names}

def project_embed_pca(z: np.ndarray, pca_dict: Dict[str, Any]) -> np.ndarray:
    comps = pca_dict["components"]  # (k,D)
    mean  = pca_dict.get("mean", np.zeros((comps.shape[1],), np.float32))
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    y = comps @ (z - mean)
    return y.astype(np.float32)
