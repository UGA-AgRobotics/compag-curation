# [ONLY-CODES-COMPAT] Vendored verbatim from Only_codes/Clean/sam2_pipeline/pipeline.py
# (see only_codes/contract.py:SOURCE_HASHES). Only documented plumbing patches
# (tagged "ONLY-CODES-COMPAT PATCH") differ; decision, feature, AL and writer
# logic are unchanged.
# pipeline.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import sys
import os
import csv as _csv
import gc

import numpy as np
import cv2
import pandas as pd

try:
    from ultralytics import YOLO as _YOLO
except Exception:
    _YOLO = None

# Local imports (package or flat file)
# [ONLY-CODES-COMPAT PATCH PL-0] vendored package-relative imports only; the
# original's flat-layout fallback resolved to the same module contents.
from .defaults import (
    PAD_FRAC, DEFAULT_COMBO_GATES, DEFAULT_GATE_ORDER, DEFAULT_USE_GATES,
    XGB_FEATS_BALANCED_DEFAULT, XGB_FEATS_RICH_DEFAULT, all_gate_names
)
from .pca_utils import resolve_pca_path
from .xgb_utils import finalize_auto_xgb_names, uncertainty
from .light_gate import evaluate_light_gate

# Helper module (must be next to this file)
from .gate_core import (
    warp_card_to_rect, detect_grid_lines, cell_of_point,
    compute_features, yellow_bg_stats,
    build_embed_model, crop_masked_patch, embed_patch,
    load_embed_csv, load_unified_proto_csv,
    load_embed_pca, project_embed_pca,
    set_feature_mode as _set_feature_mode,
)


def _env_truthy(name: str, default: str = "1") -> bool:
    v = str(os.getenv(name, default)).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


# [ONLY-CODES-COMPAT PATCH PL-1] CJ_SAVE_ALL is not read from the environment;
# run_pipeline() takes the recorded value from args.save_all_outputs (reference: True).
SAVE_ALL_OUTPUTS = True


def _draw_masks(
    base: np.ndarray,
    kept_masks: List[np.ndarray],
    rej_masks: List[np.ndarray],
    kept_badges=None,
    rej_badges=None,
    legend: Optional[Dict[str, Any]] = None,
    kept_ids: Optional[List[str]] = None,
    rej_ids: Optional[List[str]] = None,
    show_badges: bool = True,
) -> np.ndarray:
    """Draw contours with small id labels. Long badges are moved to a bottom panel."""
    img = base.copy()
    kept_badges = kept_badges or [None] * len(kept_masks)
    rej_badges = rej_badges or [None] * len(rej_masks)

    # kept (green)
    for idx, m in enumerate(kept_masks):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
        if cnts and kept_ids is not None and idx < len(kept_ids):
            x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            cv2.putText(
                img,
                f"id{kept_ids[idx]}",
                (x, max(12, y - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # rejected (red)
    for idx, m in enumerate(rej_masks):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 0, 255), 2)
        if cnts and rej_ids is not None and idx < len(rej_ids):
            x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            cv2.putText(
                img,
                f"id{rej_ids[idx]}",
                (x, max(12, y - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # optional top-left legend (rare; normally the panel has it)
    if legend and show_badges:
        lines = []
        pol = str(legend.get("policy", "")).lower()
        lines.append(f"Policy: {pol} (missing={legend.get('missing', '')})")
        lines.append(
            f"thr_xgb={legend.get('thr_xgb', 'NA')}  thr_yolo={legend.get('thr_yolo', 'NA')}  thr_iou={legend.get('thr_iou', 'NA')}"
        )
        if pol == "hybrid":
            lines.append(f"det_thr={legend.get('det_thr', 'NA')}")
        x0, y0 = 8, 8
        for k, ln in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                img,
                (x0 - 4, y0 + 4 + k * (th + 8)),
                (x0 + tw + 4, y0 + 4 + k * (th + 8) + th + 6),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                img,
                ln,
                (x0, y0 + k * (th + 8) + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return img


def _prepare_feature_export_cols(feature_mode: str, embed_pca_names: List[str]) -> List[str]:
    FEAT_EXPORT_BASE = [
        "image", "id",
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
        "light_fail_reason",
    ]
    FEAT_EXPORT_RICH_EXTRAS = [
        "delta_b_med", "delta_a_med", "median_b_in", "median_b_ring", "median_a_in", "median_a_ring",
        "histb_q1", "histb_q2", "histb_q3", "histb_q4",
        "histab_q11", "histab_q12", "histab_q21", "histab_q22",
    ]
    feat_cols = list(FEAT_EXPORT_BASE)
    if feature_mode in ("rich", "ultra"):
        feat_cols += FEAT_EXPORT_RICH_EXTRAS
    if feature_mode == "ultra" and embed_pca_names:
        feat_cols += list(embed_pca_names)
    return feat_cols


def _render_info_panel(
    entries: List[Dict[str, Any]],
    width: int,
    legend: Optional[Dict[str, Any]] = None,
    line_h: int = 22,
    margin: int = 8,
) -> np.ndarray:
    """Create a dark bottom panel with legend + per-id probabilities."""
    def fmt(v):
        return "NA" if v is None else f"{float(v):.2f}"

    lines: List[str] = []
    if legend:
        pol = str(legend.get("policy", ""))
        lines.append(f"Policy: {pol} (missing={legend.get('missing', '')})")
        lines.append(
            f"thr_xgb={legend.get('thr_xgb', 'NA')}   thr_yolo={legend.get('thr_yolo', 'NA')}   thr_iou={legend.get('thr_iou', 'NA')}"
        )
        if str(pol).lower() == "hybrid":
            lines.append(f"det_thr={legend.get('det_thr', 'NA')}")
    for e in entries:
        lines.append(
            f"id{e['id']} [{e['state']}]:  XGB={fmt(e.get('xgb_p'))}   YOLO={fmt(e.get('yolo_conf'))}/{fmt(e.get('yolo_iou'))}   F={fmt(e.get('p_fused'))}"
        )

    H = margin * 2 + max(1, len(lines)) * line_h
    panel = np.zeros((H, max(1, width), 3), dtype=np.uint8)
    panel[:] = (32, 32, 32)
    y = margin + line_h
    for ln in lines:
        cv2.putText(panel, ln, (margin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += line_h
    return panel


def _stack_bottom(img: np.ndarray, panel: np.ndarray) -> np.ndarray:
    if panel.shape[1] != img.shape[1]:
        panel = cv2.resize(panel, (img.shape[1], panel.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.vstack([img, panel])


def _read_feature_list_file(path: Path) -> List[str]:
    txt = path.read_text(encoding="utf-8")
    if txt and txt[0] == "﻿":
        txt = txt.lstrip("﻿")
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return []
    if len(lines) == 1 and ("," in lines[0]):
        return [s.strip() for s in lines[0].split(",") if s.strip()]
    return lines


def _resolve_xgb_feature_names(xgb_raw: str, feature_mode: str, embed_pca_names: List[str]) -> Tuple[List[str], str]:
    raw = (xgb_raw or "").strip()
    low = raw.lower()
    if low in ("", "auto", "default"):
        feats = finalize_auto_xgb_names(feature_mode, embed_pca_names)
        return feats, "auto"

    if raw.startswith("@"):
        p = Path(raw[1:])
        if p.exists() and p.is_file():
            feats = _read_feature_list_file(p)
            if feats:
                return feats, f"file:{p}"

    p = Path(raw)
    if ("," not in raw) and p.exists() and p.is_file():
        feats = _read_feature_list_file(p)
        if feats:
            return feats, f"file:{p}"

    feats = [s.strip() for s in raw.split(",") if s.strip()]
    return feats, "inline"


def _bbox_iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / (ua + 1e-6))


def _yolo_support(ann_mask: np.ndarray, yolo_boxes: List[Tuple[float, float, float, float, float]]) -> Tuple[float, float]:
    """Return (yolo_conf, yolo_iou) of best YOLO box (selection by conf*IoU)."""
    if not yolo_boxes:
        return 0.0, 0.0
    x, y, w, h = cv2.boundingRect(ann_mask.astype(np.uint8))
    best_conf, best_iou, best_s = 0.0, 0.0, 0.0
    for (bx1, by1, bx2, by2, conf) in yolo_boxes:
        iou = _bbox_iou_xyxy((x, y, x + w, y + h), (bx1, by1, bx2, by2))
        s = float(conf) * float(iou)
        if s > best_s:
            best_conf, best_iou, best_s = float(conf), float(iou), s
    return best_conf, best_iou


def _al_effective_threshold(policy: str, yolo_ok: bool, thr_xgb: float, det_thr: float) -> float:
    """For hybrid: use det_thr when YOLO is valid, else fall back to thr_xgb."""
    return float(det_thr) if (str(policy).lower() == "hybrid" and bool(yolo_ok)) else float(thr_xgb)


def _al_uncertainty_from_dist(dist: float, margin: float) -> float:
    """Map |p - thr_eff| into [0,1] uncertainty (1=most uncertain)."""
    m = float(margin) if (margin is not None and float(margin) > 0) else 1e-6
    d = float(dist)
    return float(max(0.0, 1.0 - min(1.0, d / m)))


def _load_thresholds_and_stats(args):
    proto: Dict[str, Any] = {}
    feat_names: List[str] = []
    mu = np.array([])
    mad = np.array([])
    cov = np.array([])

    if getattr(args, "proto", "") and Path(args.proto).exists():
        try:
            with open(args.proto, "r", encoding="utf-8") as f:
                proto = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read proto JSON: {e}. Using defaults.")

    if getattr(args, "unified_proto_csv", ""):
        try:
            feat_names, pack = load_unified_proto_csv(Path(args.unified_proto_csv))
            mu, mad, cov = pack.get("mu"), pack.get("mad"), pack.get("cov")
        except Exception as e:
            print(f"[WARN] Failed to read unified proto CSV: {e}. Trying embed-only CSV if provided...")

    if not feat_names and getattr(args, "proto_csv", ""):
        try:
            feat_names, pack = load_unified_proto_csv(Path(args.proto_csv))
            mu, mad, cov = pack.get("mu"), pack.get("mad"), pack.get("cov")
        except Exception as e:
            print(f"[WARN] Failed to read proto CSV: {e}. Using JSON/defaults.")

    if not feat_names:
        feat_names = proto.get("feature_names", [])
        mu = np.array(proto.get("mu", []), dtype=np.float64) if feat_names else np.array([])
        mad = np.array(proto.get("mad", []), dtype=np.float64) if feat_names else np.array([])
        cov = np.array(proto.get("cov", []), dtype=np.float64) if feat_names and proto.get("cov") is not None else np.array([])

    k_mad = float(getattr(args, "k_mad", None) if getattr(args, "k_mad", None) is not None else proto.get("k_mad", 6.0))
    params = {
        "chi2_tau": float(proto.get("chi2_tau", 25.0)),
        "min_L": float(proto.get("min_L", 80.0)),
        "delta_a_min": float(proto.get("delta_a_min", 0.0)),
        "delta_b_max": float(proto.get("delta_b_max", 20.0)),
        "elong_min": float(proto.get("elong_min", 0.8)),
        "elong_max": float(proto.get("elong_max", 12.0)),
        "min_solidity": float(proto.get("min_solidity", 0.50)),
        # light gate
        "light_low_T": float(proto.get("light_low_T", 5.0)),
        "light_high_T": float(proto.get("light_high_T", 250.0)),
        "light_low_clip_max": float(proto.get("light_low_clip_max", 0.10)),
        "light_high_clip_max": float(proto.get("light_high_clip_max", 0.08)),
        "light_std_max": float(proto.get("light_std_max", 40.0)),
        "light_bg_deltaL_max": float(proto.get("light_bg_deltaL_max", 25.0)),
        "light_ring_px": int(proto.get("light_ring_px", 5)),
        "k_mad": k_mad,
    }
    return proto, feat_names, mu, mad, cov, params


def _try_build_embed_model(has_embed_proto: bool, weights_path=None):
    # [ONLY-CODES-COMPAT PATCH PL-2] explicit local ResNet50 weights; a missing or
    # unloadable backbone is a blocker instead of silently disabling embed features.
    embed_model = None
    embed_transform = None
    has_embed = has_embed_proto
    if has_embed:
        try:
            embed_model, embed_transform = build_embed_model(weights_path)
        except Exception as e:
            raise RuntimeError(f"embedding model unavailable: {e}") from e
    return has_embed, embed_model, embed_transform


def _resolve_embed_proto(args, proto):
    embed_mu = None
    embed_thresh = None
    if getattr(args, "embed_csv", ""):
        _, packed = load_embed_csv(Path(args.embed_csv))
        embed_mu = packed.get("mu_embed")
        if embed_mu is not None and hasattr(embed_mu, "size") and embed_mu.size:
            embed_mu = embed_mu / (np.linalg.norm(embed_mu) + 1e-12)
        embed_thresh = float(packed.get("embed_thresh", 0.70))
    else:
        if "mu_embed" in proto:
            try:
                me = list(map(float, proto.get("mu_embed", [])))
                embed_mu = np.array(me, dtype=np.float32)
                embed_mu = embed_mu / (np.linalg.norm(embed_mu) + 1e-12)
                embed_thresh = float(proto.get("embed_thresh", 0.70))
            except Exception:
                embed_mu = None
                embed_thresh = None
    return embed_mu, embed_thresh


def _mask_to_poly_and_bbox_in_original(m_card: np.ndarray, Minv: np.ndarray, orig_shape) -> Tuple[List[int], Tuple[int, int, int, int]]:
    H0, W0 = int(orig_shape[0]), int(orig_shape[1])
    mm = (m_card > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [], (0, 0, 0, 0)
    c = max(cnts, key=cv2.contourArea)
    eps = 0.005 * cv2.arcLength(c, True)
    c = cv2.approxPolyDP(c, eps, True)

    pts = c.reshape(-1, 1, 2).astype(np.float32)
    pts_o = cv2.perspectiveTransform(pts, Minv).reshape(-1, 2)
    pts_o[:, 0] = np.clip(pts_o[:, 0], 0, W0 - 1)
    pts_o[:, 1] = np.clip(pts_o[:, 1], 0, H0 - 1)
    poly_i = np.round(pts_o).astype(np.int32)
    x, y, w, h = cv2.boundingRect(poly_i)
    return poly_i.reshape(-1).tolist(), (int(x), int(y), int(w), int(h))


def _mask_rle_in_original(m_card: np.ndarray, Minv: np.ndarray, orig_shape) -> str:
    try:
        import pycocotools.mask as maskUtils
        H0, W0 = int(orig_shape[0]), int(orig_shape[1])
        m_orig = cv2.warpPerspective((m_card > 0).astype(np.uint8), Minv, (W0, H0), flags=cv2.INTER_NEAREST)
        rle = maskUtils.encode(np.asfortranarray(m_orig.astype(np.uint8)))
        if isinstance(rle.get("counts"), (bytes, bytearray)):
            rle["counts"] = rle["counts"].decode("ascii")
        return json.dumps(rle)
    except Exception:
        return ""


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = (a > 0).astype(np.uint8)
    bb = (b > 0).astype(np.uint8)
    inter = int((aa & bb).sum())
    union = int((aa | bb).sum())
    return float(inter) / float(union + 1e-6)


def _amg_multiscale(
    amg,
    card_bgr: np.ndarray,
    scales: List[float],
    iou_merge_thresh: float = 0.70,
    max_masks_total: int = 300,
):
    """Generate masks at multiple scales and merge by mask IoU (NMS-like)."""
    H, W = card_bgr.shape[:2]
    all_preds: List[Dict[str, Any]] = []

    for s in scales:
        if abs(s - 1.0) < 1e-6:
            img_s = card_bgr
        else:
            img_s = cv2.resize(card_bgr, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_LINEAR)

        rgb_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
        anns_s = amg.generate(rgb_s) or []

        for ann in anns_s:
            m = ann["segmentation"].astype(np.uint8)
            if m.ndim == 3:
                m = m[..., 0]
            m_back = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            all_preds.append({"m": m_back, "p": float(ann.get("predicted_iou", 1.0)), "stab": float(ann.get("stability_score", 1.0))})

    all_preds.sort(key=lambda d: (d["p"], d["stab"]), reverse=True)

    kept: List[Dict[str, Any]] = []
    for d in all_preds:
        if len(kept) >= max_masks_total:
            break
        overlap = False
        for k in kept:
            if _mask_iou(d["m"], k["m"]) >= iou_merge_thresh:
                overlap = True
                break
        if not overlap:
            kept.append(d)

    return [{"segmentation": np.asarray(d["m"], dtype=np.uint8), "predicted_iou": d["p"], "stability_score": d["stab"]} for d in kept]


def run_pipeline(args):
    # [ONLY-CODES-COMPAT PATCH PL-1] recorded SAVE_ALL value (never the environment)
    SAVE_ALL_OUTPUTS = bool(getattr(args, "save_all_outputs", True))

    # Apply global feature-mode to gate_core
    _set_feature_mode(getattr(args, "feature_mode", "balanced"))

    # --- Load thresholds (JSON or unified CSV) ---
    proto, feat_names, mu, mad, cov, params = _load_thresholds_and_stats(args)

    # embedding proto
    embed_mu, embed_thresh = _resolve_embed_proto(args, proto)
    has_embed_proto = (embed_mu is not None) and (embed_thresh is not None)
    has_embed, embed_model, embed_transform = _try_build_embed_model(
        has_embed_proto, getattr(args, "embed_backbone_weights", None))

    # ULTRA: optional PCA (explicit or preset)
    embed_pca = None
    embed_pca_names: List[str] = []
    pca_path_resolved = resolve_pca_path(args, Path(getattr(args, "out", "."))) if getattr(args, "feature_mode", "balanced") == "ultra" else ""
    if getattr(args, "feature_mode", "balanced") == "ultra" and pca_path_resolved:
        try:
            pca_dict = load_embed_pca(pca_path_resolved)
            embed_pca = pca_dict
            embed_pca_names = list(pca_dict.get("names", []))
            print(f"[INFO] Loaded embed PCA: {len(embed_pca_names)} dims from {pca_path_resolved}")
        except Exception as e:
            print(f"[WARN] Failed to load embed PCA: {e}")
            embed_pca = None
            embed_pca_names = []

    # --- YOLO load ---
    yolo_model = None
    if getattr(args, "yolo_weights", ""):
        # [ONLY-CODES-COMPAT PATCH PL-3] configured-but-unloadable YOLO weights are a
        # blocker (the original silently continued without a detector). The
        # "no detector loaded" branch is reached by not configuring weights.
        if _YOLO is None:
            raise RuntimeError("ultralytics is not installed but YOLO weights are configured")
        try:
            yolo_model = _YOLO(args.yolo_weights)
            print(f"[INFO] YOLO loaded: {args.yolo_weights}")
        except Exception as e:
            raise RuntimeError(f"YOLO load failed: {e}") from e

    # --- SAM2 setup ---
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import torch
    except Exception as e:
        sys.exit(f"[ERROR] sam2/torch not available: {e}")

    # [ONLY-CODES-COMPAT PATCH PL-5] explicit device matched to the baseline (the
    # original inferred cuda-if-available); no silent CPU fallback.
    device = str(getattr(args, "device", "") or "")
    if device not in ("cuda", "cpu"):
        raise RuntimeError("an explicit device (cuda|cpu) is required")
    if device == "cuda" and not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        raise RuntimeError("configured CUDA device is unavailable")

    # Accept external runner modes; treat any non-'auto' as 'auto' here.
    if str(getattr(args, "mode", "auto")).lower() not in ("auto", "gate", "xgb", "xgb_recall"):
        print(f"[WARN] unsupported --mode={getattr(args, 'mode', None)}; using 'auto'.")

    sam_model = build_sam2(args.sam2_config, args.sam2_ckpt, device=device, apply_postprocessing=False)

    amg = SAM2AutomaticMaskGenerator(
        sam_model,
        points_per_side=getattr(args, "points_per_side", 32),
        points_per_batch=getattr(args, "points_per_batch", 256),
        pred_iou_thresh=getattr(args, "pred_iou", 0.88),
        stability_score_thresh=getattr(args, "stability", 0.90),
        crop_n_layers=getattr(args, "crop_n_layers", 0),
        crop_n_points_downscale_factor=getattr(args, "crop_n_points_downscale_factor", 2),
        crop_overlap_ratio=getattr(args, "crop_overlap_ratio", 0.40),
        output_mode="binary_mask",
        max_num_masks=(None if int(getattr(args, "max_num_masks", 0) or 0) == 0 else int(getattr(args, "max_num_masks"))),
    )

    embed_device = device
    if has_embed and (embed_model is not None):
        try:
            embed_model = embed_model.to(embed_device)
            embed_model.eval()
        except Exception as e:
            print(f"[WARN] Embedding model to {embed_device} failed: {e}")
            has_embed = False

    # --- IO ---
    img_dir = Path(getattr(args, "images", ""))
    if not img_dir.exists():
        sys.exit(f"[ERROR] images dir not found: {img_dir}")

    out_dir = Path(getattr(args, "out", "."))
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Optional: load XGBoost model ---
    # [ONLY-CODES-COMPAT PATCH PL-4] the classifier comes from a verified safe bundle
    # converted from the trusted legacy pickle (see only_codes/model_bundle.py);
    # no deserialization happens here.
    xgb_proba_fn = getattr(args, "xgb_proba_fn", None)

    # Resolve XGB feature list (auto OR custom via @file/path/inline)
    xgb_raw = str(getattr(args, "xgb_features", "auto")).strip()
    xgb_feat_names, xgb_feat_src = _resolve_xgb_feature_names(xgb_raw, getattr(args, "feature_mode", "balanced"), embed_pca_names)
    print(f"[INFO] XGB features ({xgb_feat_src}): {len(xgb_feat_names)} cols")

    xgb_extra_cols: List[str] = []
    if xgb_proba_fn is not None and xgb_feat_names:
        xgb_extra_cols = ["xgb_p", "xgb_score", "xgb_unc"] + [f"xf_{k}" for k in xgb_feat_names]

    # Feature export header (mode-aware)
    feat_fp = None
    feat_writer = None
    feat_cols = _prepare_feature_export_cols(getattr(args, "feature_mode", "balanced"), embed_pca_names)

    if SAVE_ALL_OUTPUTS and bool(getattr(args, "export_features", False)):
        feat_path = Path(getattr(args, "export_features_path", "")) if getattr(args, "export_features_path", "") else (out_dir / "features_pool.csv")
        feat_fp = open(feat_path, "w", newline="", encoding="utf-8")
        feat_writer = _csv.writer(feat_fp)
        header_feat = feat_cols + (["xgb_proba", "uncertainty"] if xgb_proba_fn is not None else [])
        feat_writer.writerow(header_feat)

    # Active-learning rows: [image,id,p_for_al,thr_eff]
    al_rows: List[List[Any]] = []

    # Detections CSV
    csv_path = out_dir / "detections.csv"

    extra_prob_cols: List[str] = []
    if (xgb_proba_fn is not None) and bool(getattr(args, "use_xgb_inference", False)):
        extra_prob_cols = ["p_fused", "yolo_conf", "yolo_iou", "fuse_w"]

    header = [
        "image", "id", "x", "y", "w", "h", "area_px", "area_mm2", "mean_L", "mean_a", "mean_b",
        "delta_a", "delta_b", "elongation", "solidity", "aspect_ratio", "circularity",
        "centroid_x", "centroid_y", "grid_r", "grid_c", "touching_border",
        "pred_iou", "stability", "embed_sim", "kept",
        "g_quality", "g_light", "g_color", "g_shape", "g_embed", "g_robust", "g_maha", "g_border",
        "first_killer", "largest_mask", "true_count", "min_true_req", "passed_combos",
        "poly", "rle",
    ] + xgb_extra_cols + extra_prob_cols

    gate_names_all = all_gate_names()
    fail_counts = {g: 0 for g in gate_names_all}
    first_killer_counts = {g: 0 for g in gate_names_all}
    total_props = 0
    kept_props = 0

    order = [g.strip().lower() for g in str(getattr(args, "gate_order", DEFAULT_GATE_ORDER)).split(",") if g.strip()]
    for g in order:
        if g not in gate_names_all:
            sys.exit(f"[ERROR] Unknown gate in --gate-order: {g}. Valid: {gate_names_all}")
    seq_pass_counts = [0] * len(order)

    img_paths = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")])

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = _csv.writer(fcsv)
        writer.writerow(header)

        for img_index, img_path in enumerate(img_paths, start=1):
            if (img_index % 5) == 0:
                try:
                    del amg
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                amg = SAM2AutomaticMaskGenerator(
                    sam_model,
                    points_per_side=getattr(args, "points_per_side", 32),
                    points_per_batch=getattr(args, "points_per_batch", 256),
                    pred_iou_thresh=getattr(args, "pred_iou", 0.88),
                    stability_score_thresh=getattr(args, "stability", 0.90),
                    crop_n_layers=getattr(args, "crop_n_layers", 0),
                    crop_n_points_downscale_factor=getattr(args, "crop_n_points_downscale_factor", 2),
                    crop_overlap_ratio=getattr(args, "crop_overlap_ratio", 0.40),
                    output_mode="binary_mask",
                    max_num_masks=(None if int(getattr(args, "max_num_masks", 0) or 0) == 0 else int(getattr(args, "max_num_masks"))),
                )

            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[WARN] cannot read: {img_path}")
                continue

            card, Minv = warp_card_to_rect(bgr)
            lab = cv2.cvtColor(card, cv2.COLOR_BGR2LAB)
            bg_a, bg_b, bg_bgr = yellow_bg_stats(card)
            rows, cols = detect_grid_lines(card)

            # SAM2 mask-generation policy: both | auto | prompt
            sam2_policy = str(getattr(args, "sam2_policy", "both")).lower()
            if sam2_policy not in ("both", "auto", "prompt"):
                sam2_policy = "both"

            # ---- YOLO + SAM prompt-first; AMG fallback ----
            yolo_boxes_xyxy: List[Tuple[float, float, float, float, float]] = []
            if yolo_model is not None:
                try:
                    res1 = yolo_model(
                        card,
                        imgsz=int(getattr(args, "yolo_imgsz", 512)),
                        conf=float(getattr(args, "yolo_conf", 0.20)),
                        iou=float(getattr(args, "yolo_iou", 0.60)),
                        device=getattr(args, "yolo_device", None),
                        max_det=int(getattr(args, "yolo_max_det", 300)),
                        verbose=False,
                    )
                    n1 = int(res1[0].boxes.shape[0]) if getattr(res1[0], "boxes", None) is not None else 0

                    res2, n2 = None, 0
                    if n1 == 0:
                        rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
                        res2 = yolo_model.predict(
                            source=rgb,
                            imgsz=int(getattr(args, "yolo_imgsz", 512)),
                            conf=float(getattr(args, "yolo_conf", 0.20)),
                            iou=float(getattr(args, "yolo_iou", 0.60)),
                            device=getattr(args, "yolo_device", None),
                            max_det=int(getattr(args, "yolo_max_det", 300)),
                            verbose=False,
                        )
                        n2 = int(res2[0].boxes.shape[0]) if getattr(res2[0], "boxes", None) is not None else 0

                    rr = res1[0] if n1 >= n2 else (res2[0] if res2 is not None else res1[0])

                    names = getattr(rr, "names", None) or getattr(yolo_model, "names", None) or {}
                    boxes = getattr(rr, "boxes", None)
                    n_total = int(boxes.shape[0]) if boxes is not None else 0
                    counts_by_cls: Dict[str, int] = {}
                    n_cj = 0
                    if boxes is not None and getattr(boxes, "cls", None) is not None:
                        cls_ids = [int(c) for c in boxes.cls.view(-1).tolist()]
                        for k in cls_ids:
                            nm = (names[k] if isinstance(names, dict) and k in names else str(k))
                            counts_by_cls[nm] = counts_by_cls.get(nm, 0) + 1
                        cj_name = str(getattr(args, "yolo_cj_name", "cj")).lower()
                        cj_idx = getattr(args, "yolo_cj_idx", None)
                        n_cj = sum(1 for c in cls_ids if cj_idx is not None and int(c) == int(cj_idx)) if cj_idx is not None else sum(v for k, v in counts_by_cls.items() if str(k).lower() == cj_name)
                        if n_cj == 0 and n_total > 0 and (len(counts_by_cls) <= 1):
                            n_cj = n_total

                    print(f"[YOLO] {img_path.name}: boxes={n_total}, CJ={n_cj}, by_class={counts_by_cls} (try1={n1}{', try2='+str(n2) if n1==0 else ''})")

                    if getattr(rr, "boxes", None) is not None:
                        for b in rr.boxes:
                            x1, y1, x2, y2 = b.xyxy[0].tolist()
                            yolo_boxes_xyxy.append((float(x1), float(y1), float(x2), float(y2), float(b.conf[0])))

                    if SAVE_ALL_OUTPUTS and len(yolo_boxes_xyxy) == 0 and bool(getattr(args, "debug_yolo_dump", False)):
                        if SAVE_ALL_OUTPUTS: cv2.imwrite(str(out_dir / f"{img_path.stem}_yolo_input.jpg"), card)

                except Exception as e:
                    print(f"[WARN] YOLO prediction failed on {img_path.name}: {e}")

            print(f"[YOLO] {img_path.name}: {len(yolo_boxes_xyxy)} boxes")

            anns_prompt: List[Dict[str, Any]] = []
            if sam2_policy in ("both", "prompt") and len(yolo_boxes_xyxy) > 0:
                pred = SAM2ImagePredictor(sam_model)
                pred.set_image(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
                H, W = card.shape[:2]
                PAD = float(getattr(args, "yolo_pad_frac", 0.02))
                for (x1, y1, x2, y2, conf) in yolo_boxes_xyxy:
                    dx = PAD * (x2 - x1)
                    dy = PAD * (y2 - y1)
                    bx = np.array([max(0.0, x1 - dx), max(0.0, y1 - dy), min(W - 1.0, x2 + dx), min(H - 1.0, y2 + dy)], dtype=np.float32)
                    masks, scores, _ = pred.predict(point_coords=None, point_labels=None, box=bx[None, :], multimask_output=True)
                    k = int(np.argmax(scores))
                    m = (masks[k] > 0).astype(np.uint8)
                    anns_prompt.append({"segmentation": m, "predicted_iou": float(scores[k]), "stability_score": float(scores[k]), "src": "prompt"})

            anns_auto: List[Dict[str, Any]] = []
            if sam2_policy in ("both", "auto"):
                ms_scales_str = str(getattr(args, "ms_scales", "1.0")).strip()
                try:
                    MS_SCALES = [float(s) for s in ms_scales_str.split(",") if s.strip()]
                except Exception:
                    MS_SCALES = [1.0]
                MS_IOU = float(getattr(args, "ms_merge_iou", 0.75))
                MS_MAX = int(getattr(args, "ms_max_masks", 150))
                anns_auto = _amg_multiscale(amg, card, MS_SCALES, iou_merge_thresh=MS_IOU, max_masks_total=MS_MAX)
                for a in anns_auto:
                    a["src"] = "auto"

            if sam2_policy == "prompt":
                anns = anns_prompt
            elif sam2_policy == "auto":
                anns = anns_auto
            else:
                anns = anns_prompt + anns_auto

            print(f"[SAM2] policy={sam2_policy}, prompt={len(anns_prompt)}, auto={len(anns_auto)}, total={len(anns)}")

            areas = [int(ann["segmentation"].astype(np.uint8).sum()) for ann in anns] if anns else []
            largest_idx = (int(np.argmax(areas)) + 1) if areas else -1

            kept_masks_main, kept_ids_main = [], []
            rej_masks_main, rej_ids_main = [], []
            kept_masks_prompt, rej_masks_prompt = [], []
            kept_ids_prompt, rej_ids_prompt = [], []
            kept_badges_main, rej_badges_main = [], []
            kept_badges_prompt, rej_badges_prompt = [], []

            thr_xgb_last = thr_yolo_last = thr_iou_last = det_thr_last = None
            policy_last = str(getattr(args, "det_policy", "and")).lower()
            missing_last = str(getattr(args, "det_missing", "reject")).lower()

            # precompute features
            precomp = []
            for i, ann in enumerate(anns, start=1):
                m = ann["segmentation"].astype(np.uint8) * 255
                pred_iou = float(ann.get("predicted_iou", 1.0))
                stab = float(ann.get("stability_score", 1.0))
                src = ann.get("src", "auto")

                core_mode = ("rich" if getattr(args, "feature_mode", "balanced") == "ultra" else getattr(args, "feature_mode", "balanced"))
                f = compute_features(m, lab, mode=core_mode)
                if not f:
                    precomp.append((i, m, pred_iou, stab, None, src))
                else:
                    f["delta_a"] = bg_a - f.get("mean_a", 0.0)
                    f["delta_b"] = f.get("mean_b", 0.0) - bg_b
                    r, c = cell_of_point(f["cx"], f["cy"], rows, cols)
                    f["grid_r"], f["grid_c"] = r, c
                    precomp.append((i, m, pred_iou, stab, f, src))

            for (i, m, pred_iou, stab, f, src) in precomp:
                total_props += 1

                # Defaults for any branch
                xgb_ok = None
                yolo_ok = None
                xgb_p = None
                unc_xgb = None
                p_fused = None
                yolo_conf = 0.0
                yolo_iou = 0.0
                policy = str(getattr(args, "det_policy", "and")).lower()
                missing = str(getattr(args, "det_missing", "reject")).lower()
                thr_xgb = float(getattr(args, "thr_xgb_raw", getattr(args, "xgb_threshold", 0.5)))
                thr_yolo = float(getattr(args, "thr_yolo_raw", getattr(args, "yolo_conf", 0.20)))
                thr_iou = float(getattr(args, "thr_yolo_iou", getattr(args, "yolo_iou", 0.30)))
                det_thr = float(getattr(args, "det_thr", 0.5))

                if not f:
                    poly_flat, (xo, yo, wo, ho) = _mask_to_poly_and_bbox_in_original(m, Minv, bgr.shape)
                    poly_str = json.dumps(poly_flat) if poly_flat else ""
                    rle_str = _mask_rle_in_original(m, Minv, bgr.shape)

                    row_out = [
                        img_path.name, i,
                        int(xo), int(yo), int(wo), int(ho),
                        "", "", "", "", "", "", "", "", "", "", "",
                        "", "", "", "", 0, round(pred_iou, 3), round(stab, 3), "", 0,
                        0, 0, 0, 0, 0, 0, 0, "no_features", 0, 0, int(getattr(args, "min_true_gates", 0)), "[]",
                        poly_str, rle_str,
                    ]
                    if xgb_extra_cols:
                        row_out += ["", "", ""] + [0.0] * len(xgb_feat_names)
                    if extra_prob_cols:
                        row_out += ["", "", "", ""]
                    writer.writerow(row_out)
                    first_killer_counts["quality"] += 1
                    continue

                # ---------- EMBEDDING ----------
                embed_sim = ""
                g_embed = True
                z = None
                has_embed_local = bool(has_embed) and (embed_thresh is not None) and (embed_mu is not None)
                if has_embed_local:
                    patch = crop_masked_patch(card, m, pad_frac=PAD_FRAC, bg_bgr=bg_bgr)
                    if patch is not None:
                        try:
                            z = embed_patch(patch, embed_model, embed_transform, device=embed_device)
                        except Exception:
                            z = None
                    if z is not None:
                        embed_sim = float(np.dot(z, embed_mu))
                        g_embed = (embed_sim >= float(embed_thresh))
                    else:
                        embed_sim = 0.0
                        g_embed = False

                # ---------- PCA (ULTRA) ----------
                embed_pca_vals: List[float] = []
                if getattr(args, "feature_mode", "balanced") == "ultra" and (embed_pca is not None) and (z is not None):
                    try:
                        y = project_embed_pca(z, embed_pca)
                        embed_pca_vals = [float(v) for v in y.tolist()]
                    except Exception as e:
                        print(f"[WARN] PCA projection failed on {img_path.name}#{i}: {e}")
                        embed_pca_vals = []

                # ---------- GATES (for logging only) ----------
                g_quality = (pred_iou >= float(getattr(args, "pred_iou", 0.88))) and (stab >= float(getattr(args, "stability", 0.90))) and (float(f["mean_L"]) >= float(params["min_L"]))
                g_light, light_fail_reason = evaluate_light_gate(lab, m, params)

                color_mode = str(getattr(args, "color_mode", "ab")).lower()
                if color_mode == "none":
                    g_color = True
                elif color_mode == "b_only":
                    g_color = (float(f["delta_b"]) <= float(getattr(args, "delta_b_max_bonly", params["delta_b_max"])))
                else:
                    g_color = (float(f["delta_a"]) >= float(params["delta_a_min"])) and (float(f["delta_b"]) <= float(params["delta_b_max"]))

                g_shape = (float(f["elongation"]) >= float(params["elong_min"])) and (float(f["elongation"]) <= float(params["elong_max"])) and (float(f["solidity"]) >= float(params["min_solidity"]))
                g_border = (int(f.get("touching_border", 0)) == 0)

                g_robust = True
                g_maha = True
                if len(feat_names) > 0:
                    xvec = np.array([float(f.get(k, 0.0)) for k in feat_names], dtype=np.float64)
                    if mu is not None and mad is not None and getattr(mu, "size", 0) > 0 and getattr(mad, "size", 0) > 0:
                        g_robust = bool(np.all(np.abs(xvec - mu) <= (float(params["k_mad"]) * mad)))
                    if hasattr(cov, "size") and getattr(cov, "size", 0) > 0 and (not bool(getattr(args, "skip_maha", False))):
                        d2 = float((xvec - mu) @ np.linalg.pinv(cov) @ (xvec - mu))
                        g_maha = (d2 <= float(params["chi2_tau"]))

                gate_flags = {
                    "quality": g_quality, "light": g_light, "color": g_color, "shape": g_shape,
                    "embed": g_embed, "robust": g_robust, "maha": g_maha, "border": g_border
                }

                for g, ok in gate_flags.items():
                    if (not ok) and g in fail_counts:
                        fail_counts[g] += 1

                first_killer = ""
                running = True
                for j, g in enumerate([gg for gg in str(getattr(args, "gate_order", DEFAULT_GATE_ORDER)).split(",") if gg]):
                    running = running and gate_flags.get(g, True)
                    if running:
                        seq_pass_counts[j] += 1
                    else:
                        first_killer = g
                        if g in first_killer_counts:
                            first_killer_counts[g] += 1
                        break

                is_largest = 1 if (bool(getattr(args, "exclude_largest", False)) and i == largest_idx) else 0
                true_count = 0  # gates are not used for decision

                # ---------- XGB / Feature export / AL ----------
                f_all = dict(f)
                f_all.update({
                    "pred_iou": float(pred_iou),
                    "stability": float(stab),
                    "embed_sim": float(embed_sim) if embed_sim != "" else 0.0,
                    "grid_r": int(f.get("grid_r", 0)),
                    "grid_c": int(f.get("grid_c", 0)),
                    "light_fail_reason": light_fail_reason,
                })
                try:
                    f_all["grid_r_norm"] = float(f_all["grid_r"]) / max(1.0, float(len(rows) - 1))
                    f_all["grid_c_norm"] = float(f_all["grid_c"]) / max(1.0, float(len(cols) - 1))
                except Exception:
                    f_all["grid_r_norm"] = 0.0
                    f_all["grid_c_norm"] = 0.0

                if getattr(args, "feature_mode", "balanced") == "ultra" and embed_pca_names:
                    for j, name in enumerate(embed_pca_names):
                        val = embed_pca_vals[j] if j < len(embed_pca_vals) else 0.0
                        f_all[name] = float(val)

                xgb_feat_values = [0.0] * len(xgb_feat_names)
                if (xgb_proba_fn is not None) and xgb_feat_names:
                    try:
                        xgb_feat_values = [float(f_all.get(k, 0.0)) for k in xgb_feat_names]
                        X_df = pd.DataFrame([dict(zip(xgb_feat_names, xgb_feat_values))], columns=xgb_feat_names)
                        xgb_p = float(xgb_proba_fn(X_df))
                        unc_xgb = uncertainty(xgb_p, getattr(args, "al_metric", "default"))
                    except Exception as e:
                        print(f"[WARN] XGB predict failed on {img_path.name}#{i}: {e}")
                        xgb_p = None
                        unc_xgb = None

                # --- Write feature export row ---
                if feat_writer is not None:
                    row = []
                    for key in feat_cols:
                        if key == "image":
                            row.append(img_path.name)
                        elif key == "id":
                            row.append(i)
                        else:
                            row.append(f_all.get(key, f.get(key, "")))
                    if xgb_p is not None:
                        row += [xgb_p, unc_xgb]
                    feat_writer.writerow(row)

                # --- Make decision only with XGB/YOLO; Gates are ignored ---
                keep_main = False
                if bool(getattr(args, "use_xgb_inference", False)):
                    yolo_conf, yolo_iou = _yolo_support(m, yolo_boxes_xyxy)
                    xgb_p_eff = float(xgb_p) if xgb_p is not None else 0.0

                    xgb_has = (xgb_p is not None)
                    yolo_has = (yolo_conf > 0.0)

                    xgb_ok = bool(xgb_has and (xgb_p_eff >= thr_xgb))
                    yolo_ok = bool(yolo_has and (yolo_conf >= thr_yolo) and (yolo_iou >= thr_iou))

                    thr_xgb_last, thr_yolo_last, thr_iou_last, det_thr_last = thr_xgb, thr_yolo, thr_iou, det_thr
                    policy_last, missing_last = policy, missing

                    p_yolo = float(yolo_conf)
                    fuse_w = float(yolo_conf)  # for CSV

                    def _combine_default(x_ok, y_ok, x_has_, y_has_, pol, miss):
                        present = []
                        if x_has_:
                            present.append(bool(x_ok))
                        if y_has_:
                            present.append(bool(y_ok))
                        if not present:
                            return False
                        if pol in ("xgb", "xgb_only"):
                            return bool(x_ok) if x_has_ else (False if miss == "reject" else False)
                        if pol in ("yolo", "yolo_only"):
                            return bool(y_ok) if y_has_ else (False if miss == "reject" else False)
                        if pol == "or":
                            return (bool(x_ok) or bool(y_ok)) if miss == "reject" else any(present)
                        return (bool(x_ok) and bool(y_ok)) if miss == "reject" else all(present)

                    # If YOLO is not OK:
                    if not yolo_ok:
                        if missing == "reject":
                            keep_main = False
                            first_killer = "yolo_not_detected"
                            p_fused = None
                        else:
                            if xgb_has:
                                keep_main = bool(xgb_ok)
                                p_fused = xgb_p_eff
                                first_killer = "<none>" if keep_main else "yolo_missing_xgb_low"
                            else:
                                keep_main = False
                                first_killer = "both_missing"
                                p_fused = None
                    else:
                        # Policy logic (executed only when YOLO is present)
                        if policy == "hybrid":
                            def _norm(a, b):
                                s = float(a) + float(b)
                                return (float(a) / s, float(b) / s) if s > 0 else (0.5, 0.5)

                            # --- SMART HYBRID OVERRIDES (rule-based) ---
                            # These match the Dash reviewer logic and are ON by default.
                            did_override = False
                            if bool(getattr(args, "smart_hybrid", True)) and xgb_has:
                                try:
                                    x_lo = float(getattr(args, "smart_noncj_xgb_max", 0.20))
                                    y_lo = float(getattr(args, "smart_noncj_yolo_max", 0.70))
                                    y_hi = float(getattr(args, "smart_cj_yolo_min", 0.85))
                                    x_hi = float(getattr(args, "smart_cj_xgb_min", 0.30))

                                    # If XGB is very low, require a stronger YOLO to override it.
                                    if (xgb_p_eff <= x_lo) and (p_yolo <= y_lo):
                                        keep_main = False
                                        p_fused = min(float(p_yolo), float(xgb_p_eff))
                                        first_killer = "smart_low_xgb_low_yolo"
                                        did_override = True

                                    # If both are reasonably confident CJ, accept early.
                                    elif (p_yolo >= y_hi) and (xgb_p_eff >= x_hi):
                                        keep_main = True
                                        p_fused = max(float(p_yolo), float(xgb_p_eff))
                                        first_killer = "<none>"
                                        did_override = True
                                except Exception:
                                    did_override = False

                            if xgb_has:
                                if did_override:
                                    pass
                                elif (xgb_p_eff >= det_thr) and (p_yolo >= det_thr):
                                    wy, wx = _norm(p_yolo, xgb_p_eff)
                                    p_fused = wy * p_yolo + (1.0 - wy) * xgb_p_eff
                                    keep_main = True
                                    first_killer = "<none>"
                                elif (p_yolo >= det_thr) and (xgb_p_eff < det_thr):
                                    wy, wx = _norm(p_yolo, xgb_p_eff)
                                    p_fused = wy * p_yolo + (1.0 - wy) * xgb_p_eff
                                    keep_main = (p_fused >= det_thr)
                                    first_killer = "<none>" if keep_main else "hybrid_weight_low"
                                elif (xgb_p_eff >= det_thr) and (p_yolo < det_thr):
                                    bias = float(getattr(args, "hybrid_yolo_bias", 0.6))
                                    wy_raw, _ = _norm(p_yolo, xgb_p_eff)
                                    wy = max(bias, wy_raw)
                                    wx = 1.0 - wy
                                    p_fused = wy * p_yolo + wx * xgb_p_eff
                                    keep_main = (p_fused >= det_thr)
                                    first_killer = "<none>" if keep_main else "hybrid_weight_low"
                                else:
                                    wy, wx = _norm(p_yolo, xgb_p_eff)
                                    p_fused = wy * p_yolo + (1.0 - wy) * xgb_p_eff
                                    keep_main = False
                                    first_killer = "hybrid_both_low"
                            else:
                                keep_main = (p_yolo >= det_thr)
                                p_fused = p_yolo
                                first_killer = "<none>" if keep_main else "hybrid_only_yolo_low"
                        else:
                            keep_main = _combine_default(xgb_ok, yolo_ok, xgb_has, yolo_has, policy, missing)
                            p_fused = (1.0 - fuse_w) * xgb_p_eff + fuse_w * p_yolo
                            if not keep_main:
                                if not xgb_has and not yolo_has:
                                    first_killer = "both_missing"
                                elif policy in ("xgb", "xgb_only"):
                                    first_killer = "xgb_only_fail" if not xgb_ok else "<none>"
                                elif policy in ("yolo", "yolo_only"):
                                    first_killer = "yolo_only_fail" if not yolo_ok else "<none>"
                                elif policy == "and":
                                    if (not xgb_ok) and (not yolo_ok):
                                        first_killer = "xgb_yolo_fail"
                                    elif not xgb_ok:
                                        first_killer = "xgb_fail"
                                    elif not yolo_ok:
                                        first_killer = "yolo_fail"
                                elif policy == "or":
                                    first_killer = "or_both_fail" if (not xgb_ok and not yolo_ok) else "<none>"
                            else:
                                first_killer = "<none>"

                    # Active-learning shortlist: uncertainty near threshold + (optionally) XGB-vs-YOLO disagreement.
                    # Works even if you bypass XGB (falls back to YOLO conf for AL if available).
                    if (xgb_proba_fn is not None) or (yolo_model is not None):
                        try:
                            use_pf = (str(policy).lower() == "hybrid" and bool(yolo_ok) and (p_fused is not None))
                            if use_pf:
                                p_for_al = float(p_fused)
                                thr_eff = float(det_thr)
                            elif xgb_p is not None:
                                p_for_al = float(xgb_p_eff)
                                thr_eff = float(thr_xgb)
                            elif yolo_has:
                                p_for_al = float(p_yolo)
                                thr_eff = float(thr_yolo)
                            else:
                                p_for_al = None
                                thr_eff = None

                            if p_for_al is not None and thr_eff is not None:
                                al_rows.append([img_path.name, i, float(p_for_al), float(thr_eff), (float(xgb_p_eff) if xgb_p is not None else None), (float(p_yolo) if yolo_has else None), (float(yolo_iou) if yolo_has else None), int(bool(yolo_ok))])
                        except Exception:
                            pass

                # Build per-proposal badge (booleans + scores)
                badge = {
                    "xgb_ok": (bool(xgb_ok) if xgb_ok is not None else None),
                    "yolo_ok": (bool(yolo_ok) if yolo_ok is not None else None),
                    "keep": bool(keep_main),
                    "xgb_p": (float(xgb_p) if xgb_p is not None else None),
                    "yolo_conf": (float(yolo_conf) if yolo_conf is not None else None),
                    "yolo_iou": (float(yolo_iou) if yolo_iou is not None else None),
                    "p_fused": (float(p_fused) if p_fused is not None else None),
                }

                if not is_largest:
                    if keep_main:
                        kept_props += 1
                        kept_masks_main.append(m)
                        kept_ids_main.append(str(i))
                        kept_badges_main.append(badge)
                        if src == "prompt":
                            kept_masks_prompt.append(m)
                            kept_ids_prompt.append(str(i))
                            kept_badges_prompt.append(badge)
                    else:
                        rej_masks_main.append(m)
                        rej_ids_main.append(str(i))
                        rej_badges_main.append(badge)
                        if src == "prompt":
                            rej_masks_prompt.append(m)
                            rej_ids_prompt.append(str(i))
                            rej_badges_prompt.append(badge)

                # Passed-combo IDs (kept for logging; optional)
                passed_combo_ids: List[int] = []
                combo_gates = [g.strip() for g in str(getattr(args, "combo_gates", DEFAULT_COMBO_GATES)).split(",") if g.strip()]
                if len(combo_gates) >= 2:
                    from itertools import combinations
                    cid = 0
                    for k in range(1, len(combo_gates)):
                        for sub in combinations(combo_gates, k):
                            cid += 1
                            if all(gate_flags.get(g, True) for g in sub):
                                passed_combo_ids.append(cid)

                poly_flat, (xo, yo, wo, ho) = _mask_to_poly_and_bbox_in_original(m, Minv, bgr.shape)
                poly_str = json.dumps(poly_flat) if poly_flat else ""
                rle_str = _mask_rle_in_original(m, Minv, bgr.shape)

                row_out = [
                    img_path.name, i,
                    int(xo), int(yo), int(wo), int(ho),
                    int(f["area_px"]), "",
                    round(float(f["mean_L"]), 1), round(float(f["mean_a"]), 1), round(float(f["mean_b"]), 1),
                    round(float(f["delta_a"]), 1), round(float(f["delta_b"]), 1),
                    round(float(f["elongation"]), 2), round(float(f["solidity"]), 2), round(float(f["aspect_ratio"]), 2), round(float(f["circularity"]), 2),
                    round(float(f["cx"]), 1), round(float(f["cy"]), 1), int(f["grid_r"]), int(f["grid_c"]), int(f.get("touching_border", 0)),
                    round(float(pred_iou), 3), round(float(stab), 3), (round(float(embed_sim), 3) if embed_sim != "" else ""),
                    int(bool(keep_main)),
                    int(g_quality), int(g_light), int(g_color), int(g_shape), int(g_embed), int(g_robust), int(g_maha), int(g_border),
                    (first_killer if first_killer else ("<none>" if keep_main else "<unknown>")),
                    int(is_largest), int(true_count), int(getattr(args, "min_true_gates", 0)),
                    "[" + ",".join(str(x) for x in passed_combo_ids) + "]",
                    poly_str, rle_str,
                ]

                if xgb_extra_cols:
                    row_out += [
                        (round(float(xgb_p), 6) if xgb_p is not None else ""),
                        (round(float(xgb_p), 6) if xgb_p is not None else ""),
                        (round(float(unc_xgb), 6) if unc_xgb is not None else ""),
                    ] + [round(float(v), 6) for v in xgb_feat_values]

                if extra_prob_cols:
                    if p_fused is not None:
                        row_out += [round(float(p_fused), 6), round(float(yolo_conf), 6), round(float(yolo_iou), 6), round(float(yolo_conf), 6)]
                    else:
                        row_out += ["", "", "", ""]

                writer.writerow(row_out)

            # --- draw overlays and panels for this image ---
            H0, W0 = bgr.shape[:2]

            def _warp_back(masks):
                out = []
                for mm in masks:
                    m0 = cv2.warpPerspective(mm, Minv, (W0, H0), flags=cv2.INTER_NEAREST)
                    out.append(m0)
                return out

            kept_back = _warp_back(kept_masks_main)
            rej_back = _warp_back(rej_masks_main)

            legend = {
                "policy": policy_last, "missing": missing_last,
                "thr_xgb": thr_xgb_last, "thr_yolo": thr_yolo_last, "thr_iou": thr_iou_last,
                "det_thr": det_thr_last,
            }

            img_keep_core = _draw_masks(bgr, kept_back, [], legend=None, kept_ids=kept_ids_main, rej_ids=[], show_badges=False)
            img_rej_core = _draw_masks(bgr, [], rej_back, legend=None, kept_ids=[], rej_ids=rej_ids_main, show_badges=False)
            if SAVE_ALL_OUTPUTS: cv2.imwrite(str(out_dir / f"{img_path.stem}_main_kept.jpg"), img_keep_core)
            if SAVE_ALL_OUTPUTS: cv2.imwrite(str(out_dir / f"{img_path.stem}_main_rejected.jpg"), img_rej_core)

            entries_keep = [{"id": int(sid), "state": "K", **(b or {})} for sid, b in zip(kept_ids_main, kept_badges_main)]
            entries_rej = [{"id": int(sid), "state": "R", **(b or {})} for sid, b in zip(rej_ids_main, rej_badges_main)]
            entries_keep = sorted(entries_keep, key=lambda d: d["id"])
            entries_rej = sorted(entries_rej, key=lambda d: d["id"])

            panel_keep = _render_info_panel(entries_keep, W0, legend=legend)
            panel_rej = _render_info_panel(entries_rej, W0, legend=legend)
            if SAVE_ALL_OUTPUTS: cv2.imwrite(str(out_dir / f"{img_path.stem}_main_kept_list.jpg"), _stack_bottom(img_keep_core, panel_keep))
            if SAVE_ALL_OUTPUTS: cv2.imwrite(str(out_dir / f"{img_path.stem}_main_rejected_list.jpg"), _stack_bottom(img_rej_core, panel_rej))

            if sam2_policy in ("both", "prompt"):
                kept_prompt_back = _warp_back(kept_masks_prompt)
                rej_prompt_back = _warp_back(rej_masks_prompt)
                img_prompt_core = _draw_masks(bgr, kept_prompt_back, rej_prompt_back, legend=None, kept_ids=kept_ids_prompt, rej_ids=rej_ids_prompt, show_badges=False)
                if SAVE_ALL_OUTPUTS: cv2.imwrite(str(out_dir / f"{img_path.stem}_yolo_prompt.jpg"), img_prompt_core)

                entries_prompt = (
                    [{"id": int(sid), "state": "K", **(b or {})} for sid, b in zip(kept_ids_prompt, kept_badges_prompt)]
                    + [{"id": int(sid), "state": "R", **(b or {})} for sid, b in zip(rej_ids_prompt, rej_badges_prompt)]
                )
                entries_prompt = sorted(entries_prompt, key=lambda d: d["id"])
                panel_prompt = _render_info_panel(entries_prompt, W0, legend=legend)
                if SAVE_ALL_OUTPUTS: cv2.imwrite(str(out_dir / f"{img_path.stem}_yolo_prompt_list.jpg"), _stack_bottom(img_prompt_core, panel_prompt))

    # --- summary CSV ---
    summary_csv = out_dir / "gate_summary.csv"
    if SAVE_ALL_OUTPUTS:
        with open(summary_csv, "w", newline="", encoding="utf-8") as fs:
            w = _csv.writer(fs)
            w.writerow(["metric", "gate", "count", "rate"])
            w.writerow(["overall", "total_proposals", total_props, "1.000"])
            w.writerow(["overall", "kept", kept_props, f"{(kept_props / max(1, total_props)):.3f}"])
            for g in all_gate_names():
                rate = fail_counts[g] / max(1, total_props)
                w.writerow(["independent_fail", g, fail_counts[g], f"{rate:.3f}"])
            prev = total_props
            for j, g in enumerate([gg for gg in str(getattr(args, "gate_order", DEFAULT_GATE_ORDER)).split(",") if gg]):
                kept_after = seq_pass_counts[j] if j < len(seq_pass_counts) else 0
                dropped_here = prev - kept_after
                rate = dropped_here / max(1, total_props)
                w.writerow(["sequential_drop", g, dropped_here, f"{rate:.3f}"])
                prev = kept_after

    # --- finalize optional exports ---
    if feat_fp is not None:
        try:
            feat_fp.close()
        except Exception:
            pass

    # --- Active-learning outputs ---
    if SAVE_ALL_OUTPUTS and len(al_rows) > 0:
        al_path = Path(getattr(args, "al_out", "")) if getattr(args, "al_out", "") else (out_dir / "al_candidates.csv")
        try:
            al_topk = int(getattr(args, "al_topk", 50))
            al_margin = float(getattr(args, "al_margin", 0.20))
            al_only_uncertain = bool(getattr(args, "al_only_uncertain", True))
            al_fill_to_topk = bool(getattr(args, "al_fill_to_topk", False))

            # Disagreement-driven AL: include severe XGB-vs-YOLO disagreements in addition to near-threshold samples.
            al_use_disagreement = bool(getattr(args, "al_use_disagreement", True))
            al_disagree_abs = float(getattr(args, "al_disagree_abs", 0.55))
            al_disagree_xgb_hi = float(getattr(args, "al_disagree_xgb_hi", 0.80))
            al_disagree_xgb_lo = float(getattr(args, "al_disagree_xgb_lo", 0.20))
            al_disagree_yolo_hi = float(getattr(args, "al_disagree_yolo_hi", 0.85))
            al_disagree_yolo_lo = float(getattr(args, "al_disagree_yolo_lo", 0.15))
            al_unc_w = float(getattr(args, "al_unc_weight", 1.0))
            al_dis_w = float(getattr(args, "al_disagree_weight", 1.0))

            scored = []
            for r in al_rows:
                if not r or len(r) < 4:
                    continue

                img = r[0]
                rid = r[1]
                p_for_al = float(r[2])
                thr_eff = float(r[3])

                # Optional extra fields (added in this version)
                xgb_p_row = None
                yolo_conf_row = None
                yolo_iou_row = None
                yolo_ok_row = None
                if len(r) >= 8:
                    try:
                        xgb_p_row = None if r[4] is None else float(r[4])
                    except Exception:
                        xgb_p_row = None
                    try:
                        yolo_conf_row = None if r[5] is None else float(r[5])
                    except Exception:
                        yolo_conf_row = None
                    try:
                        yolo_iou_row = None if r[6] is None else float(r[6])
                    except Exception:
                        yolo_iou_row = None
                    try:
                        yolo_ok_row = bool(int(r[7]))
                    except Exception:
                        try:
                            yolo_ok_row = bool(r[7])
                        except Exception:
                            yolo_ok_row = None

                dist = abs(p_for_al - thr_eff)
                in_unc_band = (dist <= al_margin)
                unc_eff = _al_uncertainty_from_dist(dist, al_margin)

                disagree_abs = None
                disagree_flag = False
                if al_use_disagreement and (xgb_p_row is not None) and (yolo_conf_row is not None):
                    try:
                        disagree_abs = abs(float(xgb_p_row) - float(yolo_conf_row))
                        severe_abs = (disagree_abs >= al_disagree_abs)
                        severe_conf = (
                            (float(xgb_p_row) >= al_disagree_xgb_hi and float(yolo_conf_row) <= al_disagree_yolo_lo)
                            or (float(xgb_p_row) <= al_disagree_xgb_lo and float(yolo_conf_row) >= al_disagree_yolo_hi)
                        )
                        disagree_flag = bool(severe_abs or severe_conf)
                    except Exception:
                        disagree_abs = None
                        disagree_flag = False

                disagree_score = float(disagree_abs) if (disagree_flag and disagree_abs is not None) else 0.0
                al_score = (al_unc_w * float(unc_eff)) + (al_dis_w * float(disagree_score))

                if in_unc_band and disagree_flag:
                    reason = "both"
                elif in_unc_band:
                    reason = "uncertain"
                elif disagree_flag:
                    reason = "disagree"
                else:
                    reason = ""

                scored.append([
                    img, rid,
                    p_for_al, float(unc_eff), float(dist), float(thr_eff),
                    xgb_p_row, yolo_conf_row, yolo_iou_row,
                    disagree_abs, bool(disagree_flag),
                    float(al_score), reason,
                ])

            # Sort helper: higher al_score first, then closer to threshold, then larger disagreement.
            def _sort_key(rr):
                try:
                    dis = float(rr[9]) if rr[9] is not None else -1.0
                except Exception:
                    dis = -1.0
                return (-float(rr[11]), float(rr[4]), -dis)

            if al_only_uncertain:
                pool = []
                for rr in scored:
                    in_unc = bool(rr[4] <= al_margin)
                    in_dis = bool(al_use_disagreement and rr[10])
                    if in_unc or in_dis:
                        pool.append(rr)
                pool.sort(key=_sort_key)
                chosen = pool[:al_topk]

                if al_fill_to_topk and len(chosen) < al_topk:
                    chosen_keys = set((str(rr[0]), str(rr[1])) for rr in chosen)
                    rest = [rr for rr in scored if (str(rr[0]), str(rr[1])) not in chosen_keys]
                    rest.sort(key=_sort_key)
                    chosen += rest[: (al_topk - len(chosen))]
            else:
                scored.sort(key=_sort_key)
                chosen = scored[:al_topk]

            with open(al_path, "w", newline="", encoding="utf-8") as fal:
                w = _csv.writer(fal)
                # Keep first 4 cols compatible with previous versions.
                w.writerow([
                    "image", "id", "xgb_proba", "uncertainty",
                    "al_score", "al_reason",
                    "xgb_p", "yolo_conf", "yolo_iou",
                    "disagree_abs", "dist", "thr_eff",
                ])
                for rr in chosen:
                    img, rid = rr[0], rr[1]
                    p_for_al, unc_eff, dist, thr_eff = rr[2], rr[3], rr[4], rr[5]
                    xgb_p_row, yolo_conf_row, yolo_iou_row = rr[6], rr[7], rr[8]
                    disagree_abs, al_score, reason = rr[9], rr[11], rr[12]
                    w.writerow([
                        img, rid, p_for_al, unc_eff,
                        al_score, reason,
                        xgb_p_row, yolo_conf_row, yolo_iou_row,
                        disagree_abs, dist, thr_eff,
                    ])

            print(
                f"[DONE] Active-learning candidates: {al_path} (rows={len(chosen)}, only_uncertain={al_only_uncertain}, "
                f"margin={al_margin}, disagreement={al_use_disagreement}, disagree_abs={al_disagree_abs})"
            )

        except Exception as e:
            print(f"[WARN] Failed to write AL candidates: {e}")

    print(f"[DONE] CSV: {csv_path}")
    if SAVE_ALL_OUTPUTS:

        print(f"[DONE] Summary: {summary_csv}")
    print(f"Feature-mode: {getattr(args, 'feature_mode', 'balanced')}")
    if embed_pca_names:
        print(f"Embed PCA dims: {len(embed_pca_names)}")
    print(f"XGB feature set: {xgb_feat_src} ({len(xgb_feat_names)} cols)")
    print(f"Use-gates: {getattr(args, 'use_gates', DEFAULT_USE_GATES)}, min_true={getattr(args, 'min_true_gates', 0)}")
    print(f"Combo-gates: {getattr(args, 'combo_gates', DEFAULT_COMBO_GATES)}  save_all_combos={getattr(args, 'save_all_combos', False)}  exclude_largest={getattr(args, 'exclude_largest', False)}")