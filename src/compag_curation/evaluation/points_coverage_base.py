
#!/usr/bin/env python3
"""
eval_points_coverage_v3.py

Point-based insect-level evaluation (CVAT points) for your SAM2+XGB(+YOLO fusion) pipeline,
WITH an additional "uncertainty / AL recoverability" analysis.

✅ Design choices (matches your request):
- **No bipartite / one-to-one matching.**
- Each GT point (CJ insect) is detected if it lies inside >=1 predicted instance
  (polygon preferred; bbox fallback).
- Works on one image OR multiple images (subset selection via --images).
- Supports:
  - a merged predictions CSV (many images) OR
  - per-image detections.csv OR
  - multiple detections.csv files OR
  - --run_dir convenience (auto-find detections.csv, tile index, al_candidates.csv)

✅ New in v3:
- For each GT point, we compute whether it would be "surfaced" by Active Learning:
  1) Does it have ANY covering proposal within AL margin (|p_for_al - thr_eff| <= AL_MARGIN)?
  2) Does it have ANY covering proposal that is actually present in al_candidates.csv (top-k shortlist)?

This lets you write a statement like:
"Among insect-level misses at τ=0.5, X% were within the AL uncertainty margin and Y% appeared
in the top-50 uncertainty shortlist—indicating they are likely to be prioritized for human review."

Important: with point-only GT, we do NOT claim true instance-level precision. We report:
- coverage recall (main)
- redundancy / merge / FP proxies (diagnostics)

---------------------------
Typical usage
---------------------------

A) One image (per-image outputs):
  python eval_points_coverage_v3.py \
    --cvat_xml annotations.xml \
    --pred_csv synthetic-card-a/run_xgb_recall/detections.csv \
    --tile_index synthetic-card-a/tile_info/tiles_index.csv \
    --al_candidates_csv synthetic-card-a/run_xgb_recall/al_candidates.csv \
    --images synthetic-card-a \
    --stage xgb --thr_xgb 0.50 \
    --out_dir eval_synthetic_card_a_xgb05

B) Two images (merged CSV):
  python eval_points_coverage_v3.py \
    --cvat_xml annotations.xml \
    --pred_csv test_detections__xgb_recall.csv \
    --tile_index dataset_index.csv \
    --al_candidates_csv synthetic-card-a/run_xgb_recall/al_candidates.csv \
    --al_candidates_csv synthetic-card-b/run_xgb_recall/al_candidates.csv \
    --images synthetic-card-a,synthetic-card-b \
    --stage xgb --thr_xgb 0.50 \
    --out_dir eval_both_xgb05

C) Convenience using run dirs (auto-detect al_candidates + tile_index):
  python eval_points_coverage_v3.py \
    --cvat_xml annotations.xml \
    --run_dir .../synthetic-card-a/run_xgb_recall \
    --run_dir .../synthetic-card-b/run_xgb_recall \
    --stage xgb --thr_xgb 0.50 \
    --out_dir eval_both_xgb05

Optional:
  --nms_iou 0.5
  --max_area_frac 0.05
  --images_root /path/to/full_images   (write overlays)

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2

from .geometry import bbox_iou, nms_xyxy


# -------------------------- CVAT XML (points) --------------------------


def parse_cvat_points_xml(xml_path: Path, *, label: str = "cj") -> Tuple[Dict[str, List[Tuple[float, float]]], Dict[str, Tuple[int, int]]]:
    """
    Parse CVAT XML exported annotations that contain <points> shapes.

    Returns:
      points_by_image: dict image_name -> list of (x,y) points (float)
      size_by_image:   dict image_name -> (width,height)
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    points_by_image: Dict[str, List[Tuple[float, float]]] = {}
    size_by_image: Dict[str, Tuple[int, int]] = {}

    for img in root.findall("image"):
        name = (img.attrib.get("name", "") or "").strip()
        if not name:
            continue
        w = int(float(img.attrib.get("width", "0")))
        h = int(float(img.attrib.get("height", "0")))
        size_by_image[name] = (w, h)

        pts: List[Tuple[float, float]] = []
        for p in img.findall("points"):
            if (p.attrib.get("label", "") or "").strip() != label:
                continue
            s = (p.attrib.get("points", "") or "").strip()
            if not s:
                continue
            # CVAT may store multiple points as "x1,y1;x2,y2;..."
            for part in s.split(";"):
                part = part.strip()
                if not part:
                    continue
                try:
                    xs, ys = part.split(",")
                    pts.append((float(xs), float(ys)))
                except Exception:
                    pass

        points_by_image[name] = pts

    return points_by_image, size_by_image


# -------------------------- Helpers -----------------------------------


def norm_img_name(s: str) -> str:
    """Preserve supported image suffixes and add .jpg only when absent."""
    s = s.strip()
    if not s:
        return s
    p = Path(s)
    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".tif", ".tiff"):
        return p.name
    return f"{p.name}.jpg"


def safe_json_list(s: str) -> Optional[List[float]]:
    s = str(s).strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
    except Exception:
        return None
    return None


def poly_to_contour(poly_flat: Optional[List[float]]) -> Optional[np.ndarray]:
    """Convert [x1,y1,x2,y2,...] -> contour (N,1,2) float32."""
    if poly_flat is None:
        return None
    if len(poly_flat) < 6 or (len(poly_flat) % 2 != 0):
        return None
    pts = np.array(poly_flat, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return None
    return pts.reshape(-1, 1, 2)


def detect_tile_col(df: pd.DataFrame) -> str:
    for c in ("tile_image", "image", "img", "filename", "file"):
        if c in df.columns:
            return c
    raise KeyError("Could not find image filename column in predictions CSV. Expected one of: tile_image, image, img, filename, file")


def detect_bbox_cols(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    if all(c in df.columns for c in ("x", "y", "w", "h")):
        return ("x", "y", "w", "h")
    if all(c in df.columns for c in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")):
        return ("bbox_x", "bbox_y", "bbox_w", "bbox_h")
    raise KeyError("Could not find bbox columns. Expected x,y,w,h or bbox_x,bbox_y,bbox_w,bbox_h")


# -------------------------- Loading / auto paths -----------------------


def infer_paths_from_run_dir(run_dir: Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """
    Given .../synthetic-card-a/run_xgb_recall, try to find:
      - detections.csv   (required)
      - tile index       (optional): ../tile_info/tiles_index.csv
      - al_candidates.csv (optional): run_dir/al_candidates.csv
    """
    run_dir = Path(run_dir)
    pred_csv = run_dir / "detections.csv"
    if not pred_csv.is_file():
        raise FileNotFoundError(f"run_dir does not contain detections.csv: {pred_csv}")

    # tile index: try parent sibling tile_info/tiles_index.csv
    tile_index = run_dir.parent / "tile_info" / "tiles_index.csv"
    if not tile_index.is_file():
        tile_index = None

    al_csv = run_dir / "al_candidates.csv"
    if not al_csv.is_file():
        al_csv = None

    return pred_csv, tile_index, al_csv


def load_many_csv(paths: List[Path]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(p)
        dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame()


def load_al_candidates(paths: List[Path]) -> pd.DataFrame:
    """Load and concatenate al_candidates.csv files."""
    if not paths:
        return pd.DataFrame()
    df = load_many_csv(paths)
    # Standardize dtypes if possible
    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    if "image" in df.columns:
        df["image"] = df["image"].astype(str)
    return df


# -------------------------- Coordinate alignment -----------------------


def ensure_full_coords(
    pred: pd.DataFrame,
    *,
    tile_index: Optional[pd.DataFrame],
    cvat_sizes: Dict[str, Tuple[int, int]],
) -> pd.DataFrame:
    """
    If tile_index is provided, merge tile origin (x0,y0) and orig_name (full image) and shift
    prediction geometry to full-image coordinates.

    If tile_index is None, assume predictions are already in full-image coords with 'image' matching CVAT names.
    """
    pred = pred.copy()

    tile_col = detect_tile_col(pred)
    pred["tile_base"] = pred[tile_col].astype(str).apply(lambda s: Path(s).name)

    if tile_index is None:
        pred["full_image"] = pred[tile_col].astype(str).apply(lambda s: Path(s).name)
        pred["orig_w"] = pred["full_image"].map(lambda n: cvat_sizes.get(n, (np.nan, np.nan))[0])
        pred["orig_h"] = pred["full_image"].map(lambda n: cvat_sizes.get(n, (np.nan, np.nan))[1])
        pred["tile_x0"] = 0
        pred["tile_y0"] = 0
        # bbox columns to unified names
        bx, by, bw, bh = detect_bbox_cols(pred)
        pred["bbox_x1"] = pred[bx].astype(float)
        pred["bbox_y1"] = pred[by].astype(float)
        pred["bbox_x2"] = pred["bbox_x1"] + pred[bw].astype(float)
        pred["bbox_y2"] = pred["bbox_y1"] + pred[bh].astype(float)
        # polygon
        if "poly" in pred.columns:
            pred["poly_full"] = pred["poly"].astype(str)
        else:
            pred["poly_full"] = ""
        return pred

    idx = tile_index.copy()
    if "tile_name" in idx.columns:
        idx["tile_base"] = idx["tile_name"].astype(str).apply(lambda s: Path(s).name)
    elif "tile_base" not in idx.columns:
        raise KeyError("tile_index must include tile_name or tile_base column")

    for c in ("x", "y", "orig_name"):
        if c not in idx.columns:
            raise KeyError(f"tile_index missing required column: {c}")

    idx = idx.rename(columns={"x": "tile_x0", "y": "tile_y0"})
    merged = pred.merge(
        idx[["tile_base", "tile_x0", "tile_y0", "orig_name", "orig_w", "orig_h"]],
        on="tile_base", how="left", validate="many_to_one"
    )
    merged = merged.rename(columns={"orig_name": "full_image"})

    # Fill missing sizes from CVAT
    merged["orig_w"] = merged["orig_w"].where(~merged["orig_w"].isna(), merged["full_image"].map(lambda n: cvat_sizes.get(n, (np.nan, np.nan))[0]))
    merged["orig_h"] = merged["orig_h"].where(~merged["orig_h"].isna(), merged["full_image"].map(lambda n: cvat_sizes.get(n, (np.nan, np.nan))[1]))

    bx, by, bw, bh = detect_bbox_cols(merged)
    merged["bbox_x1"] = merged[bx].astype(float) + merged["tile_x0"].astype(float)
    merged["bbox_y1"] = merged[by].astype(float) + merged["tile_y0"].astype(float)
    merged["bbox_x2"] = merged["bbox_x1"] + merged[bw].astype(float)
    merged["bbox_y2"] = merged["bbox_y1"] + merged[bh].astype(float)

    # Shift polygon if present
    if "poly" in merged.columns:
        polys_full: List[str] = []
        for _, r in merged.iterrows():
            poly = safe_json_list(r.get("poly", ""))
            if not poly:
                polys_full.append("")
                continue
            arr = np.array(poly, dtype=np.float32)
            arr[0::2] += float(r["tile_x0"])
            arr[1::2] += float(r["tile_y0"])
            polys_full.append(json.dumps(arr.tolist()))
        merged["poly_full"] = polys_full
    else:
        merged["poly_full"] = ""

    # Clip bbox to image bounds if known
    def _clip_row(r):
        W = r.get("orig_w")
        H = r.get("orig_h")
        if pd.isna(W) or pd.isna(H):
            return r
        W = float(W); H = float(H)
        r["bbox_x1"] = float(np.clip(r["bbox_x1"], 0, max(0.0, W - 1)))
        r["bbox_y1"] = float(np.clip(r["bbox_y1"], 0, max(0.0, H - 1)))
        r["bbox_x2"] = float(np.clip(r["bbox_x2"], 0, max(0.0, W - 1)))
        r["bbox_y2"] = float(np.clip(r["bbox_y2"], 0, max(0.0, H - 1)))
        return r

    merged = merged.apply(_clip_row, axis=1)
    return merged


# -------------------------- Instances ----------------------------------


@dataclass
class PredInstance:
    det_id: str
    tile_image: str
    bbox_xyxy: np.ndarray
    contour: Optional[np.ndarray]
    score: float
    p_for_al: float
    dist: float
    uncertainty: float
    in_al_topk: bool


def build_instances(df_img: pd.DataFrame, *, score_col: Optional[str], al_key_set: Optional[set]) -> List[PredInstance]:
    out: List[PredInstance] = []

    # score
    if score_col and (score_col in df_img.columns):
        scores = df_img[score_col].astype(float).fillna(0.0).values
    else:
        scores = np.zeros(len(df_img), dtype=np.float32)

    ids = df_img["id"].astype(str).values if "id" in df_img.columns else df_img.index.astype(str).values
    tile_img_col = detect_tile_col(df_img)
    tile_imgs = df_img[tile_img_col].astype(str).apply(lambda s: Path(s).name).values

    # p_for_al fields
    p_for_al = df_img.get("p_for_al", pd.Series([np.nan]*len(df_img))).astype(float).fillna(np.nan).values
    dist = df_img.get("dist_to_thr", pd.Series([np.nan]*len(df_img))).astype(float).fillna(np.nan).values
    unc = df_img.get("uncertainty", pd.Series([0.0]*len(df_img))).astype(float).fillna(0.0).values

    for i in range(len(df_img)):
        bbox = np.array([df_img.iloc[i]["bbox_x1"], df_img.iloc[i]["bbox_y1"], df_img.iloc[i]["bbox_x2"], df_img.iloc[i]["bbox_y2"]], dtype=np.float32)
        if not np.isfinite(bbox).all():
            continue
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue

        poly = safe_json_list(df_img.iloc[i].get("poly_full", ""))
        contour = poly_to_contour(poly) if poly else None

        key = (tile_imgs[i], int(df_img.iloc[i]["id"])) if ("id" in df_img.columns and al_key_set is not None) else None
        in_topk = bool(key in al_key_set) if (key is not None) else False

        out.append(PredInstance(
            det_id=str(ids[i]),
            tile_image=str(tile_imgs[i]),
            bbox_xyxy=bbox,
            contour=contour,
            score=float(scores[i]),
            p_for_al=float(p_for_al[i]) if np.isfinite(p_for_al[i]) else float("nan"),
            dist=float(dist[i]) if np.isfinite(dist[i]) else float("nan"),
            uncertainty=float(unc[i]),
            in_al_topk=in_topk,
        ))
    return out


def point_in_instance(px: float, py: float, inst: PredInstance) -> bool:
    x1, y1, x2, y2 = inst.bbox_xyxy
    if px < x1 or px > x2 or py < y1 or py > y2:
        return False
    if inst.contour is None:
        return True
    return cv2.pointPolygonTest(inst.contour, (float(px), float(py)), False) >= 0


# -------------------------- AL / uncertainty fields ---------------------


def add_uncertainty_fields(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Add:
      - yolo_ok
      - p_for_al
      - thr_eff
      - dist_to_thr
      - uncertainty  (AL_MARGIN-based)

    Mirrors your pipeline logic (simplified):
      if det_policy=hybrid and yolo_ok and p_fused available -> p_for_al=p_fused else xgb_p
      thr_eff = det_thr if yolo_ok else thr_xgb
      dist = |p_for_al - thr_eff|
      uncertainty = max(0, 1 - min(1, dist/AL_MARGIN))
    """
    df = df.copy()

    # yolo_ok
    if ("yolo_conf" in df.columns) and ("yolo_iou" in df.columns):
        y_ok = (df["yolo_conf"].astype(float).fillna(-1) >= float(args.yolo_conf_thr)) & \
               (df["yolo_iou"].astype(float).fillna(-1) >= float(args.yolo_iou_thr))
    else:
        y_ok = pd.Series([False]*len(df), index=df.index)
    df["yolo_ok"] = y_ok

    # thr_eff
    thr_xgb = float(args.thr_xgb)
    det_thr = float(args.det_thr)
    if args.det_policy.lower() == "hybrid":
        df["thr_eff"] = np.where(df["yolo_ok"].to_numpy(), det_thr, thr_xgb)
    else:
        df["thr_eff"] = thr_xgb

    # p_for_al
    if args.det_policy.lower() == "hybrid" and ("p_fused" in df.columns) and args.use_p_fused:
        pf = df["p_fused"].astype(float)
        df["p_for_al"] = np.where(df["yolo_ok"].to_numpy() & np.isfinite(pf.to_numpy()), pf.to_numpy(), df["xgb_p"].astype(float).to_numpy())
    else:
        df["p_for_al"] = df["xgb_p"].astype(float)

    # dist + uncertainty
    df["dist_to_thr"] = np.abs(df["p_for_al"].astype(float) - df["thr_eff"].astype(float))
    m = float(args.al_margin)
    d = df["dist_to_thr"].astype(float).to_numpy()
    unc = np.maximum(0.0, 1.0 - np.minimum(1.0, d / m))
    df["uncertainty"] = unc
    return df


def compute_al_topk_keys(df_img_all: pd.DataFrame, args: argparse.Namespace) -> set:
    """
    Compute synthetic AL shortlist from the full candidate pool (for ONE image), if al_candidates.csv isn't provided.
    Sort by dist_to_thr ascending; optionally keep only uncertain ones (dist <= AL_MARGIN).
    """
    if len(df_img_all) == 0:
        return set()

    df = df_img_all.copy()
    if "dist_to_thr" not in df.columns or "thr_eff" not in df.columns or "p_for_al" not in df.columns:
        raise RuntimeError("compute_al_topk_keys requires uncertainty fields; call add_uncertainty_fields first.")

    if args.al_only_uncertain:
        df = df[df["dist_to_thr"].astype(float) <= float(args.al_margin)].copy()

    df = df.sort_values("dist_to_thr", ascending=True)
    df = df.head(int(args.al_topk))

    tile_col = detect_tile_col(df)
    keys = set(zip(df[tile_col].astype(str).apply(lambda s: Path(s).name), df["id"].astype(int)))
    return keys


# -------------------------- Figures ------------------------------------


def make_figures(out_dir: Path, stage: str, per_image: pd.DataFrame,
                 point_cover_counts: Dict[str, np.ndarray], mask_points_counts: Dict[str, np.ndarray],
                 per_point: pd.DataFrame) -> List[Path]:
    import matplotlib.pyplot as plt

    out_paths: List[Path] = []

    # 1) Recall by image
    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    ax.bar(per_image["image"], per_image["recall_cov"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Point-coverage recall")
    ax.set_title(f"Insect-level coverage recall ({stage})")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    p = out_dir / f"fig_recall_by_image__{stage}.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    out_paths.append(p)

    # 2) Histogram: how many masks cover each GT point (redundancy)
    all_cc = np.concatenate([v for v in point_cover_counts.values()]) if point_cover_counts else np.array([], dtype=np.int32)
    if all_cc.size > 0:
        fig = plt.figure(figsize=(6, 3.2))
        ax = fig.add_subplot(111)
        maxv = int(np.max(all_cc))
        bins = np.arange(-0.5, maxv + 1.5, 1.0)
        ax.hist(all_cc, bins=bins)
        ax.set_xlabel("#predicted instances covering a GT point")
        ax.set_ylabel("#GT points")
        ax.set_title(f"Redundancy over GT points ({stage})")
        fig.tight_layout()
        p = out_dir / f"fig_point_cover_hist__{stage}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        out_paths.append(p)

    # 3) Histogram: how many GT points are inside each predicted mask (merge + FP proxy)
    all_pc = np.concatenate([v for v in mask_points_counts.values()]) if mask_points_counts else np.array([], dtype=np.int32)
    if all_pc.size > 0:
        fig = plt.figure(figsize=(6, 3.2))
        ax = fig.add_subplot(111)
        maxv = int(np.max(all_pc))
        bins = np.arange(-0.5, maxv + 1.5, 1.0)
        ax.hist(all_pc, bins=bins)
        ax.set_xlabel("#GT points inside a predicted instance")
        ax.set_ylabel("#predicted instances")
        ax.set_title(f"Points-per-mask distribution ({stage})")
        fig.tight_layout()
        p = out_dir / f"fig_points_per_mask_hist__{stage}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        out_paths.append(p)

    # 4) New: uncertainty distribution for HIT vs MISS (per point)
    if "covered_stage" in per_point.columns and "min_dist_all" in per_point.columns:
        pp = per_point.copy()
        pp = pp[np.isfinite(pp["min_dist_all"].astype(float))]
        if len(pp) > 0 and pp["covered_stage"].nunique() > 1:
            fig = plt.figure(figsize=(6, 3.2))
            ax = fig.add_subplot(111)
            hit = pp[pp["covered_stage"] == 1]["min_dist_all"].astype(float).to_numpy()
            miss = pp[pp["covered_stage"] == 0]["min_dist_all"].astype(float).to_numpy()
            bins = np.linspace(0.0, 0.5, 26)
            ax.hist(hit, bins=bins, alpha=0.7, label="Covered (stage)")
            ax.hist(miss, bins=bins, alpha=0.7, label="Missed (stage)")
            ax.set_xlabel("min |p_for_al - thr_eff| among covering proposals")
            ax.set_ylabel("#GT points")
            ax.set_title(f"Uncertainty of available proposals (hit vs miss) ({stage})")
            ax.legend()
            fig.tight_layout()
            p = out_dir / f"fig_min_dist_hit_vs_miss__{stage}.png"
            fig.savefig(p, dpi=200)
            plt.close(fig)
            out_paths.append(p)

    # 5) New: bar chart for miss recoverability
    if {"miss_within_margin_rate", "miss_in_al_topk_rate"}.issubset(set(per_image.columns)):
        # For multi-image, show bars per image
        fig = plt.figure(figsize=(6.5, 3.2))
        ax = fig.add_subplot(111)
        x = np.arange(len(per_image))
        ax.bar(x - 0.18, per_image["miss_within_margin_rate"].fillna(0.0), width=0.36, label="Misses with proposal within AL margin")
        ax.bar(x + 0.18, per_image["miss_in_al_topk_rate"].fillna(0.0), width=0.36, label="Misses covered by AL shortlist")
        ax.set_xticks(x)
        ax.set_xticklabels(per_image["image"], rotation=20)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Fraction of missed GT points")
        ax.set_title(f"How many misses are 'reviewable' by AL ({stage})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / f"fig_miss_recoverability__{stage}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        out_paths.append(p)

    return out_paths


def write_overlays(out_dir: Path, stage: str, images_root: Path,
                   points_by_image: Dict[str, List[Tuple[float, float]]],
                   covered_flags: Dict[str, np.ndarray]) -> List[Path]:
    out_paths: List[Path] = []
    out_sub = out_dir / f"overlays__{stage}"
    out_sub.mkdir(parents=True, exist_ok=True)

    for name, pts in points_by_image.items():
        img_path = images_root / name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        covered = covered_flags.get(name, np.zeros(len(pts), dtype=bool))

        H, W = img.shape[:2]
        max_side = max(H, W)
        scale = 1.0
        if max_side > 1600:
            scale = 1600.0 / max_side
            img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)

        for (i, (x, y)) in enumerate(pts):
            cx = int(round(x * scale))
            cy = int(round(y * scale))
            col = (0, 200, 0) if bool(covered[i]) else (0, 0, 200)  # BGR
            cv2.circle(img, (cx, cy), 6, col, thickness=2, lineType=cv2.LINE_AA)

        out_path = out_sub / f"{Path(name).stem}__{stage}.png"
        cv2.imwrite(str(out_path), img)
        out_paths.append(out_path)

    return out_paths


# -------------------------- Core evaluation ----------------------------


def run_eval(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points_by_image, cvat_sizes = parse_cvat_points_xml(Path(args.cvat_xml), label=args.cvat_label)

    # Select images
    if args.images:
        wanted = [norm_img_name(x) for x in args.images.split(",") if x.strip()]
        points_by_image = {k: v for k, v in points_by_image.items() if k in wanted}
        if not points_by_image:
            raise ValueError(f"--images selected {wanted}, but none found in CVAT XML")

    # Load predictions
    pred_paths: List[Path] = []
    tile_index_paths: List[Path] = []
    al_paths: List[Path] = []

    if args.run_dir:
        for rd in args.run_dir:
            p, ti, al = infer_paths_from_run_dir(Path(rd))
            pred_paths.append(p)
            if ti is not None:
                tile_index_paths.append(ti)
            if al is not None:
                al_paths.append(al)

    if args.pred_csv:
        pred_paths.extend([Path(p) for p in args.pred_csv])

    if args.tile_index:
        tile_index_paths.extend([Path(p) for p in args.tile_index])

    if args.al_candidates_csv:
        al_paths.extend([Path(p) for p in args.al_candidates_csv])

    if not pred_paths:
        raise ValueError("No predictions provided. Use --pred_csv and/or --run_dir.")

    # Load & concat predictions
    df_pred = load_many_csv(pred_paths)

    # Tile index handling:
    # - if user provides exactly 1 tile index file, use it (global dataset_index)
    # - if provides multiple (per-image tiles_index.csv), concat them
    tile_index_df = None
    if tile_index_paths:
        tile_index_df = load_many_csv(tile_index_paths)

    # Coordinate shift
    df_pred = ensure_full_coords(df_pred, tile_index=tile_index_df, cvat_sizes=cvat_sizes)

    # Keep only relevant images
    df_pred = df_pred[df_pred["full_image"].isin(points_by_image.keys())].copy()

    # Add uncertainty fields (for AL analysis)
    if "xgb_p" not in df_pred.columns:
        raise KeyError("predictions CSV must include xgb_p for uncertainty analysis / stage filtering.")
    df_pred = add_uncertainty_fields(df_pred, args)

    # Load AL candidates (optional)
    df_al = load_al_candidates(al_paths)
    al_key_set_global: Optional[set] = None
    if len(df_al) > 0:
        if ("image" in df_al.columns) and ("id" in df_al.columns):
            al_key_set_global = set(zip(df_al["image"].astype(str).apply(lambda s: Path(s).name),
                                        df_al["id"].astype(int)))
        else:
            print("[WARN] al_candidates.csv missing image/id columns; ignoring.")
            al_key_set_global = None

    # Evaluate per image
    stage = args.stage.lower().strip()
    score_col = args.score_col or ("xgb_p" if "xgb_p" in df_pred.columns else None)

    per_image_rows = []
    point_cover_counts: Dict[str, np.ndarray] = {}
    mask_points_counts: Dict[str, np.ndarray] = {}
    covered_flags: Dict[str, np.ndarray] = {}

    # Per-point detail rows (for uncertainty analysis)
    per_point_rows: List[dict] = []

    for img_name, pts in points_by_image.items():
        pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 2)
        n_gt = int(len(pts_arr))

        g_all = df_pred[df_pred["full_image"] == img_name].copy()

        # Optional area filter (applied to both stage and proposal pool so coverage isn't dominated by giant masks)
        if args.max_area_frac is not None:
            frac = float(args.max_area_frac)
            g_all["bbox_area"] = (g_all["bbox_x2"] - g_all["bbox_x1"]).clip(lower=0) * (g_all["bbox_y2"] - g_all["bbox_y1"]).clip(lower=0)
            img_area = float(g_all["orig_w"].iloc[0]) * float(g_all["orig_h"].iloc[0]) if len(g_all) else float("nan")
            if np.isfinite(img_area):
                g_all = g_all[g_all["bbox_area"] <= frac * img_area].copy()

        # Determine AL shortlist keys for this image:
        # - Prefer actual al_candidates.csv if provided
        # - Otherwise, compute synthetic top-k from g_all using dist_to_thr.
        al_key_set_img = None
        if al_key_set_global is not None:
            # filter to this image's tile names: we keep global set but membership check is still valid
            al_key_set_img = al_key_set_global
        else:
            al_key_set_img = compute_al_topk_keys(g_all, args)

        # Build "proposal pool" instances (ALL proposals) for uncertainty analysis
        inst_all = build_instances(g_all, score_col=score_col, al_key_set=al_key_set_img)

        # Stage filtering (end-to-end predictions)
        g_stage = g_all.copy()
        if stage == "kept":
            if "kept" not in g_stage.columns:
                raise KeyError("stage=kept requested but predictions CSV has no 'kept' column")
            g_stage = g_stage[g_stage["kept"].astype(int) == 1].copy()
        elif stage == "xgb":
            g_stage = g_stage[g_stage["xgb_p"].astype(float) >= float(args.thr_xgb)].copy()
        elif stage == "all":
            pass
        else:
            raise ValueError("stage must be one of: all, kept, xgb")

        # Optional NMS on stage predictions
        if args.nms_iou is not None and len(g_stage) > 0:
            boxes = g_stage[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(np.float32)
            scores = g_stage[score_col].astype(float).fillna(0.0).to_numpy(np.float32) if score_col in g_stage.columns else np.zeros(len(g_stage), dtype=np.float32)
            keep_idx = nms_xyxy(boxes, scores, float(args.nms_iou))
            g_stage = g_stage.reset_index(drop=True).iloc[keep_idx].copy()

        inst_stage = build_instances(g_stage, score_col=score_col, al_key_set=al_key_set_img)

        # Coverage computations
        n_pred = len(inst_stage)
        cover_count_stage = np.zeros(n_gt, dtype=np.int32)   # how many stage masks cover each point
        cover_count_all = np.zeros(n_gt, dtype=np.int32)     # how many proposal masks cover each point
        points_in_mask = np.zeros(n_pred, dtype=np.int32)    # how many GT points per stage mask

        # per-point uncertainty signals
        min_dist_all = np.full(n_gt, np.inf, dtype=np.float32)
        max_unc_all = np.zeros(n_gt, dtype=np.float32)
        any_in_al_topk = np.zeros(n_gt, dtype=bool)
        any_within_margin = np.zeros(n_gt, dtype=bool)

        if n_gt > 0 and len(inst_all) > 0:
            boxes_all = np.stack([x.bbox_xyxy for x in inst_all], axis=0)
            x1a, y1a, x2a, y2a = boxes_all[:, 0], boxes_all[:, 1], boxes_all[:, 2], boxes_all[:, 3]

            for pi, (px, py) in enumerate(pts_arr):
                cand = np.where((px >= x1a) & (px <= x2a) & (py >= y1a) & (py <= y2a))[0]
                if cand.size == 0:
                    continue
                for mi in cand.tolist():
                    if point_in_instance(float(px), float(py), inst_all[mi]):
                        cover_count_all[pi] += 1
                        # uncertainty fields
                        d = inst_all[mi].dist
                        u = inst_all[mi].uncertainty
                        if np.isfinite(d):
                            min_dist_all[pi] = min(min_dist_all[pi], float(d))
                        max_unc_all[pi] = max(max_unc_all[pi], float(u))
                        if inst_all[mi].in_al_topk:
                            any_in_al_topk[pi] = True

                if np.isfinite(min_dist_all[pi]) and min_dist_all[pi] <= float(args.al_margin):
                    any_within_margin[pi] = True

        # Stage coverage + points per stage mask
        if n_gt > 0 and n_pred > 0:
            boxes = np.stack([x.bbox_xyxy for x in inst_stage], axis=0)
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

            for pi, (px, py) in enumerate(pts_arr):
                cand = np.where((px >= x1) & (px <= x2) & (py >= y1) & (py <= y2))[0]
                if cand.size == 0:
                    continue
                for mi in cand.tolist():
                    if point_in_instance(float(px), float(py), inst_stage[mi]):
                        cover_count_stage[pi] += 1
                        points_in_mask[mi] += 1

        covered_stage = (cover_count_stage > 0)
        covered_all = (cover_count_all > 0)
        covered_flags[img_name] = covered_stage

        # Summary metrics
        cov_recall = float(covered_stage.mean()) if n_gt > 0 else float("nan")
        proposal_cov_recall = float(covered_all.mean()) if n_gt > 0 else float("nan")

        redundancy_mean = float(cover_count_stage.mean()) if n_gt > 0 else float("nan")
        redundancy_p90 = float(np.percentile(cover_count_stage, 90)) if n_gt > 0 else float("nan")
        multi_cover_rate = float((cover_count_stage >= 2).mean()) if n_gt > 0 else float("nan")

        mask_hit_rate = float((points_in_mask > 0).mean()) if n_pred > 0 else float("nan")
        fp_rate = float((points_in_mask == 0).mean()) if n_pred > 0 else float("nan")
        merge_rate = float((points_in_mask >= 2).mean()) if n_pred > 0 else float("nan")
        avg_points_per_mask = float(points_in_mask.mean()) if n_pred > 0 else float("nan")

        # Miss diagnostics
        miss_mask = ~covered_stage
        n_miss = int(miss_mask.sum())
        # Among missed points, how many are "reviewable"?
        miss_within_margin_rate = float(any_within_margin[miss_mask].mean()) if n_miss > 0 else float("nan")
        miss_in_al_topk_rate = float(any_in_al_topk[miss_mask].mean()) if n_miss > 0 else float("nan")

        # AL coverage recall: if you ONLY review AL top-k candidates, how many insects are touched?
        al_cov_recall = float(any_in_al_topk.mean()) if n_gt > 0 else float("nan")

        per_image_rows.append({
            "image": img_name,
            "stage": stage,
            "n_gt_points": n_gt,
            "n_pred_instances": n_pred,
            "covered_points": int(covered_stage.sum()),
            "recall_cov": cov_recall,

            "proposal_cov_recall": proposal_cov_recall,
            "al_cov_recall": al_cov_recall,

            "missed_points": n_miss,
            "miss_within_margin_rate": miss_within_margin_rate,
            "miss_in_al_topk_rate": miss_in_al_topk_rate,

            "redundancy_mean": redundancy_mean,
            "redundancy_p90": redundancy_p90,
            "multi_cover_rate": multi_cover_rate,
            "mask_hit_rate": mask_hit_rate,
            "fp_rate_masks": fp_rate,
            "merge_rate_masks_ge2pts": merge_rate,
            "avg_points_per_mask": avg_points_per_mask,
        })

        point_cover_counts[img_name] = cover_count_stage
        mask_points_counts[img_name] = points_in_mask

        # per-point details
        for i in range(n_gt):
            per_point_rows.append({
                "image": img_name,
                "gt_idx": i,
                "px": float(pts_arr[i, 0]),
                "py": float(pts_arr[i, 1]),
                "covered_stage": int(covered_stage[i]),
                "covered_all": int(covered_all[i]),
                "cover_count_stage": int(cover_count_stage[i]),
                "cover_count_all": int(cover_count_all[i]),
                "min_dist_all": float(min_dist_all[i]) if np.isfinite(min_dist_all[i]) else float("inf"),
                "max_unc_all": float(max_unc_all[i]),
                "any_within_margin": int(any_within_margin[i]),
                "any_in_al_topk": int(any_in_al_topk[i]),
            })

    per_image = pd.DataFrame(per_image_rows).sort_values("image")
    per_image.to_csv(out_dir / f"summary__{stage}.csv", index=False)

    per_point_df = pd.DataFrame(per_point_rows)
    per_point_df.to_csv(out_dir / f"points_detail__{stage}.csv", index=False)

    # Overall (micro)
    total_gt = int(per_image["n_gt_points"].sum())
    total_cov = int(per_image["covered_points"].sum())
    total_pred = int(per_image["n_pred_instances"].sum())
    micro_recall = float(total_cov / total_gt) if total_gt > 0 else float("nan")

    micro_proposal_cov = float((per_point_df["covered_all"] == 1).mean()) if len(per_point_df) else float("nan")
    micro_al_cov = float((per_point_df["any_in_al_topk"] == 1).mean()) if len(per_point_df) else float("nan")

    miss_df = per_point_df[per_point_df["covered_stage"] == 0]
    micro_miss_within_margin = float((miss_df["any_within_margin"] == 1).mean()) if len(miss_df) else float("nan")
    micro_miss_in_al_topk = float((miss_df["any_in_al_topk"] == 1).mean()) if len(miss_df) else float("nan")

    # Diagnostics micro (stage)
    all_mask_counts = np.concatenate([v for v in mask_points_counts.values()]) if mask_points_counts else np.array([], dtype=np.int32)
    micro_mask_hit = float((all_mask_counts > 0).mean()) if all_mask_counts.size > 0 else float("nan")
    micro_fp_rate = float((all_mask_counts == 0).mean()) if all_mask_counts.size > 0 else float("nan")
    micro_merge_rate = float((all_mask_counts >= 2).mean()) if all_mask_counts.size > 0 else float("nan")

    all_cover_counts = np.concatenate([v for v in point_cover_counts.values()]) if point_cover_counts else np.array([], dtype=np.int32)
    micro_redundancy_mean = float(all_cover_counts.mean()) if all_cover_counts.size > 0 else float("nan")
    micro_multi_cover_rate = float((all_cover_counts >= 2).mean()) if all_cover_counts.size > 0 else float("nan")

    overall = {
        "stage": stage,
        "cvat_label": args.cvat_label,
        "n_images": int(len(points_by_image)),
        "total_gt_points": total_gt,
        "total_pred_instances": total_pred,
        "total_covered_points": total_cov,
        "micro_recall_cov": micro_recall,

        "micro_proposal_cov_recall": micro_proposal_cov,
        "micro_al_cov_recall": micro_al_cov,
        "micro_miss_within_margin_rate": micro_miss_within_margin,
        "micro_miss_in_al_topk_rate": micro_miss_in_al_topk,

        "micro_mask_hit_rate": micro_mask_hit,
        "micro_fp_rate_masks": micro_fp_rate,
        "micro_merge_rate_masks_ge2pts": micro_merge_rate,
        "micro_redundancy_mean": micro_redundancy_mean,
        "micro_multi_cover_rate": micro_multi_cover_rate,

        "params": {
            "thr_xgb": float(args.thr_xgb),
            "det_policy": args.det_policy,
            "det_thr": float(args.det_thr),
            "yolo_conf_thr": float(args.yolo_conf_thr),
            "yolo_iou_thr": float(args.yolo_iou_thr),
            "al_margin": float(args.al_margin),
            "al_topk": int(args.al_topk),
            "al_only_uncertain": bool(args.al_only_uncertain),
            "nms_iou": args.nms_iou,
            "max_area_frac": args.max_area_frac,
            "score_col": score_col,
            "al_candidates_provided": bool(al_key_set_global is not None),
        },
        "notes": {
            "definition": "A GT point is detected if it lies inside >=1 predicted instance (polygon preferred; bbox fallback). No one-to-one matching is used.",
            "uncertainty_analysis": "For each GT point, we examine the proposal pool (all masks) and report whether any covering proposal is within the AL margin, and whether any covering proposal appears in the AL shortlist (al_candidates.csv or synthetic top-k).",
            "precision_note": "With point-only GT, mask_hit_rate and fp_rate_masks are proxies, not strict precision.",
        },
    }
    (out_dir / f"overall__{stage}.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    # Figures
    fig_paths = make_figures(out_dir, stage, per_image, point_cover_counts, mask_points_counts, per_point_df)

    # Optional overlays
    overlay_paths: List[Path] = []
    if args.images_root:
        overlay_paths = write_overlays(out_dir, stage, Path(args.images_root), points_by_image, covered_flags)

    # README
    md = []
    md.append(f"# Point-based insect-level evaluation ({stage})\n")
    md.append(f"- Images evaluated: **{len(points_by_image)}**\n")
    md.append(f"- Total GT points: **{total_gt}**\n")
    md.append(f"- Total predictions (stage): **{total_pred}**\n")
    md.append(f"- Micro coverage recall (stage): **{micro_recall:.3f}**\n")
    if not math.isnan(micro_proposal_cov):
        md.append(f"- Micro proposal coverage recall (all masks): **{micro_proposal_cov:.3f}**\n")
    if not math.isnan(micro_al_cov):
        md.append(f"- Micro AL coverage recall (top-k shortlist): **{micro_al_cov:.3f}**\n")
    if not math.isnan(micro_miss_within_margin):
        md.append(f"- Among misses: within-AL-margin rate: **{micro_miss_within_margin:.3f}**\n")
        md.append(f"- Among misses: covered by AL shortlist rate: **{micro_miss_in_al_topk:.3f}**\n")

    md.append("\n## Per-image summary\n")
    md.append(per_image.to_markdown(index=False))

    md.append("\n\n## Outputs\n")
    for p in [out_dir / f"summary__{stage}.csv", out_dir / f"points_detail__{stage}.csv", out_dir / f"overall__{stage}.json", *fig_paths, *overlay_paths]:
        md.append(f"- {p.name}\n")
    (out_dir / f"README__{stage}.md").write_text("\n".join(md), encoding="utf-8")

    print("[OK] Wrote:", out_dir)
    print("  -", out_dir / f"overall__{stage}.json")
    print("  -", out_dir / f"summary__{stage}.csv")
    print("  -", out_dir / f"points_detail__{stage}.csv")
    for p in fig_paths[:5]:
        print("  -", p.name)
    if overlay_paths:
        print("  - overlays:", len(overlay_paths))


# -------------------------- CLI ----------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Point-based insect-level coverage evaluation (no bipartite matching) + AL/uncertainty analysis.")
    ap.add_argument("--cvat_xml", required=True, help="CVAT XML export containing <points> annotations.")
    ap.add_argument("--cvat_label", default="cj", help="Label name in CVAT XML to treat as GT points (default: cj).")

    ap.add_argument("--pred_csv", action="append", default=[], help="Predictions CSV path (repeatable).")
    ap.add_argument("--tile_index", action="append", default=[], help="Tile index CSV path (repeatable). dataset_index.csv or per-image tiles_index.csv.")
    ap.add_argument("--run_dir", action="append", default=[], help="Convenience: run directory containing detections.csv (repeatable).")

    ap.add_argument("--al_candidates_csv", action="append", default=[], help="al_candidates.csv path(s) (repeatable). If omitted, a synthetic top-k is computed.")
    ap.add_argument("--images", default="", help="Comma-separated list of images to evaluate (e.g., synthetic-card-a,synthetic-card-b). If empty, evaluate all in CVAT XML.")

    ap.add_argument("--out_dir", required=True, help="Output directory for metrics + figures.")
    ap.add_argument("--images_root", default="", help="Optional: folder containing full images (for overlays). Filenames must match CVAT image names.")

    ap.add_argument("--stage", default="xgb", choices=["all", "kept", "xgb"],
                    help="Which predictions to evaluate as the end-to-end output: all proposals, kept==1, or xgb_p>=thr_xgb.")
    ap.add_argument("--thr_xgb", type=float, default=0.50, help="Threshold for stage=xgb and AL thr_xgb (default: 0.50).")
    ap.add_argument("--score_col", default="", help="Score column for NMS ordering/plots (default auto: xgb_p).")

    ap.add_argument("--max_area_frac", type=float, default=None,
                    help="Optional: remove predictions whose bbox area > max_area_frac * image_area (e.g., 0.05).")
    ap.add_argument("--nms_iou", type=float, default=None,
                    help="Optional: apply bbox-NMS with this IoU threshold to reduce duplicates (e.g., 0.5).")

    # AL / uncertainty config (should match your pipeline defaults)
    ap.add_argument("--al_margin", type=float, default=0.20, help="AL margin (default: 0.20).")
    ap.add_argument("--al_topk", type=int, default=50, help="AL top-k (default: 50). Used when al_candidates.csv not provided.")
    ap.add_argument("--al_only_uncertain", action="store_true", help="If set, synthetic AL shortlist keeps only candidates within AL margin (default pipeline behavior).")

    ap.add_argument("--det_policy", default="hybrid", choices=["hybrid", "xgb"],
                    help="Policy used to compute p_for_al/thr_eff for uncertainty analysis (default: hybrid).")
    ap.add_argument("--det_thr", type=float, default=0.50, help="det_thr used when det_policy=hybrid and yolo_ok (default: 0.50).")
    ap.add_argument("--yolo_conf_thr", type=float, default=0.20, help="YOLO conf threshold for yolo_ok (default: 0.20).")
    ap.add_argument("--yolo_iou_thr", type=float, default=0.60, help="YOLO IoU threshold for yolo_ok (default: 0.60).")
    ap.add_argument("--use_p_fused", action="store_true", help="If set (recommended), use p_fused when yolo_ok for AL p_for_al.")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    # Normalize empty strings
    args.images_root = args.images_root.strip()
    args.images = args.images.strip()
    args.score_col = args.score_col.strip()
    run_eval(args)


if __name__ == "__main__":
    main()
