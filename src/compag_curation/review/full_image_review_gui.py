# dash_mosaic_review_labels_v4_weighted.py
# Full-image mosaic reviewer (Plotly/Dash) + click-to-label, compatible with the *working* v1 mosaic code path.
# - No Qt (browser UI). Works on headless servers via SSH port-forward.
# - Shows model preds (CJ / Non-CJ / Uncertain), supports "Only mismatches" QA view.
# - Lets you label any object ad-hoc (no need to review all sequentially). Writes review_labels.csv as append-log.
#
# v4_weighted tweak:
#   - Adds `review_weight` column (derived from action) to the CSV rows.
#   - Weight map matches the training code (cell3A_action_weighting): accept/flip=1.0, sus_*=0.4, skip=0.0
#
# Keyboard shortcuts (after selecting a dot):
#   Accept (correct det):              1 or A
#   Flip (incorrect det):              2 or R
#   Skip (seen only; logged):         3 or S
#   Sus accept (uncertain-keep label): 4 or W
#   Sus flip (uncertain-flip label):   5 or U
#   Undo last CSV row:                 0 or Z
#
# Auto-next (optional, toggle from menu):
#   - If enabled, after each per-item action (accept/flip/sus*/skip) selection jumps to nearest visible dot.

from __future__ import annotations

import base64
import json
import re
import time
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None

# Human-readable timestamp: YYYYMMDDHHMMSS (e.g., 20251222124902)
TS_TZ = "America/New_York"  # change if you want server-local time
def now_ts_compact() -> int:
    if ZoneInfo is not None:
        try:
            return int(datetime.now(ZoneInfo(TS_TZ)).strftime("%Y%m%d%H%M%S"))
        except Exception:
            pass
    return int(datetime.now().strftime("%Y%m%d%H%M%S"))

from pathlib import Path
from urllib.parse import unquote_plus

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update, ctx

# ============================== CONFIG (EDIT HERE) ==============================
# Decision params (match pipeline/gui)
USE_XGB  = True
USE_YOLO = True

DET_POLICY  = "xgb"      # "xgb" | "hybrid" | "yolo" | "and" | "or"
DET_MISSING = "ignore"   # "reject" | "ignore"

THR_XGB  = 0.40
THR_YOLO = 0.20
THR_IOU  = 0.60
DET_THR  = 0.50
HYBRID_YOLO_BIAS = 0.80

AL_MARGIN = 0.20  # uncertain if |p_ui - thr_eff| <= AL_MARGIN

# Mosaic stitch
SCAN_ALL_TILES_FOR_BASE = True    # stitch all tiles for base, not only those present in detections rows
MISSING_TILE_BGR = (32, 32, 32)

# Output mosaic encoding
MOSAIC_FORMAT = "jpg"  # "jpg" or "png"
JPG_QUALITY   = 98     # keep it high; mosaic is your "full-res" view
# ==============================================================================

_TILE_PAT = re.compile(r"^(?P<base>.+?)_y(?P<y>\d{1,8})x(?P<x>\d{1,8})\.(?P<ext>[^.]+)$", re.IGNORECASE)

# Action -> training weight (kept consistent with cell3A_action_weighting.py)
WEIGHT_MAP = {
    "accept": 1.0,
    "flip": 1.0,
    "sus_accept": 0.4,
    "sus_flip": 0.4,
    "skip": 0.0,
    "auto_accept": 0.0,  # bulk/as-is acceptance; excluded from effort by default
}

REVIEW_COLS = [
    "image","id","human_label","action","review_weight","click_x","click_y",
    "prob","xgb_prob","final_pred","xgb_pred","kept",
    "yolo_conf","yolo_iou",
    "det_policy","det_missing",
    "thr_xgb","thr_yolo","thr_iou","det_thr","hybrid_yolo_bias",
    "use_xgb","use_yolo","timestamp"
]

# ============================== utils ==============================

def _safe_float(v, default=0.0) -> float:
    try:
        if v is None:
            return float(default)
        if isinstance(v, float) and np.isnan(v):
            return float(default)
        return float(v)
    except Exception:
        try:
            return float(str(v).strip())
        except Exception:
            return float(default)

def parse_tile(name: str):
    m = _TILE_PAT.match(Path(name).name)
    if not m:
        return None
    return (m.group("base"), int(m.group("y")), int(m.group("x")), m.group("ext"))

def key_str(image: str, det_id) -> str:
    return f"{str(image)}||{str(det_id)}"

def _parse_poly(row: dict):
    poly = row.get("poly") or row.get("polygon") or row.get("poly_str")
    if poly is None:
        return None
    if isinstance(poly, float) and np.isnan(poly):
        return None
    s = str(poly).strip()
    if not s:
        return None
    try:
        arr = json.loads(s) if isinstance(poly, str) else poly
        if isinstance(arr, (list, tuple)) and len(arr) >= 6:
            flat = np.array(arr, dtype=np.float32).reshape(-1)
            if flat.size % 2 != 0:
                return None
            pts = flat.reshape(-1, 2)
            if len(pts) >= 3:
                return pts.astype(np.float32)
    except Exception:
        return None
    return None

def _parse_bbox(row: dict, poly):
    for keys in [("x","y","w","h"), ("bbox_x","bbox_y","bbox_w","bbox_h")]:
        if all(k in row for k in keys):
            x=_safe_float(row.get(keys[0]),0); y=_safe_float(row.get(keys[1]),0)
            w=_safe_float(row.get(keys[2]),1); h=_safe_float(row.get(keys[3]),1)
            return (float(x), float(y), max(1.0,float(w)), max(1.0,float(h)))
    if poly is not None and len(poly) >= 3:
        x,y,w,h = cv2.boundingRect(poly.astype(np.int32))
        return (float(x), float(y), float(w), float(h))
    cx=_safe_float(row.get("cx", row.get("centroid_x", 50)), 50)
    cy=_safe_float(row.get("cy", row.get("centroid_y", 50)), 50)
    return (float(cx-15), float(cy-15), 30.0, 30.0)

def _combine_default(x_ok, y_ok, x_has, y_has, pol: str, miss: str) -> bool:
    present=[]
    if x_has:
        present.append(bool(x_ok))
    if y_has:
        present.append(bool(y_ok))
    if not present:
        return False
    pol = str(pol).lower()
    miss = str(miss).lower()
    if pol in ("xgb","xgb_only"):
        return bool(x_ok) if x_has else False
    if pol in ("yolo","yolo_only"):
        return bool(y_ok) if y_has else False
    if pol == "or":
        return (bool(x_ok) or bool(y_ok)) if miss=="reject" else any(present)
    return (bool(x_ok) and bool(y_ok)) if miss=="reject" else all(present)

def _norm(a: float, b: float):
    s = float(a) + float(b)
    if s <= 0:
        return (0.5, 0.5)
    return (float(a)/s, float(b)/s)

def compute_final_pred_and_ui(row: dict):
    """
    Returns: pred(int), p_ui(float|None), thr_eff(float)

    thr_eff for hybrid (to match GUI/pipeline intent):
      - if YOLO OK -> DET_THR
      - else -> THR_XGB
    """
    policy  = str(DET_POLICY).lower()
    missing = str(DET_MISSING).lower()

    xgb_p  = _safe_float(row.get("xgb_p", row.get("xgb_score", row.get("prob", 0.0))), 0.0) if USE_XGB else 0.0
    y_conf = _safe_float(row.get("yolo_conf", 0.0), 0.0) if USE_YOLO else 0.0
    y_iou  = _safe_float(row.get("yolo_iou", 0.0), 0.0) if USE_YOLO else 0.0

    x_has = bool(USE_XGB)
    y_has = bool(USE_YOLO and (y_conf > 0.0))
    x_ok  = bool(x_has and (xgb_p >= float(THR_XGB)))
    y_ok  = bool(y_has and (y_conf >= float(THR_YOLO)) and (y_iou >= float(THR_IOU)))

    if policy == "yolo":
        thr_eff = float(THR_YOLO)
    elif policy == "hybrid":
        thr_eff = float(DET_THR if y_ok else THR_XGB)
    else:
        thr_eff = float(THR_XGB)

    # Hybrid: if YOLO not OK => missing rule
    if policy == "hybrid" and not y_ok:
        if missing == "reject":
            return 0, None, thr_eff
        return (1 if x_ok else 0), float(xgb_p), thr_eff

    if policy == "hybrid":
        p_y = float(y_conf)
        x   = float(xgb_p)
        det_thr = float(DET_THR)

        if (x >= det_thr) and (p_y >= det_thr):
            wy,_ = _norm(p_y, x)
            p_f = wy*p_y + (1.0-wy)*x
            return 1, p_f, thr_eff
        elif (p_y >= det_thr) and (x < det_thr):
            wy,_ = _norm(p_y, x)
            p_f = wy*p_y + (1.0-wy)*x
            return (1 if p_f >= det_thr else 0), p_f, thr_eff
        elif (x >= det_thr) and (p_y < det_thr):
            wy_raw,_ = _norm(p_y, x)
            wy = max(float(HYBRID_YOLO_BIAS), float(wy_raw))
            p_f = wy*p_y + (1.0-wy)*x
            return (1 if p_f >= det_thr else 0), p_f, thr_eff
        else:
            wy,_ = _norm(p_y, x)
            p_f = wy*p_y + (1.0-wy)*x
            return 0, p_f, thr_eff

    keep = _combine_default(x_ok, y_ok, x_has, y_has, policy, missing)
    p_ui = float(y_conf) if policy in ("yolo","yolo_only") else (float(xgb_p) if x_has else None)
    return (1 if keep else 0), p_ui, thr_eff

def is_uncertain(p_ui, thr_eff):
    if p_ui is None:
        return False
    return abs(float(p_ui) - float(thr_eff)) <= float(AL_MARGIN)

# ============================== mosaic IO ==============================

def load_tile_rgb(p: Path):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def build_mosaic(base: str, tiles_dir: Path, tiles: list[tuple[str,int,int]]):
    # find sample tile size
    sample = None
    for name, yy, xx in tiles:
        im = load_tile_rgb(tiles_dir / name)
        if im is not None:
            sample = im
            break
    if sample is None:
        raise RuntimeError(f"No readable tiles for base={base} (check TILES_DIR).")

    tile_h, tile_w = sample.shape[:2]
    max_y = max(int(yy) for _,yy,_ in tiles)
    max_x = max(int(xx) for _,_,xx in tiles)

    H = int(max_y + tile_h)
    W = int(max_x + tile_w)

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = np.array(MISSING_TILE_BGR, dtype=np.uint8)[None,None,:]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    name_by_pos = {(int(yy), int(xx)): str(name) for (name,yy,xx) in tiles}

    # stitch
    for (yy, xx), name in name_by_pos.items():
        im = load_tile_rgb(tiles_dir / name)
        if im is None:
            continue
        if im.shape[0]!=tile_h or im.shape[1]!=tile_w:
            im = cv2.resize(im, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        y0, x0 = int(yy), int(xx)
        canvas[y0:y0+tile_h, x0:x0+tile_w] = im

    return canvas

def write_mosaic_and_get_data_uri(mosaic: np.ndarray, out_path: Path) -> tuple[str,int,int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
    if out_path.suffix.lower() == ".jpg":
        cv2.imwrite(str(out_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPG_QUALITY)])
    else:
        cv2.imwrite(str(out_path), bgr)

    im = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Failed to re-read mosaic: {out_path}")
    H, W = im.shape[:2]

    data = out_path.read_bytes()
    mime = "image/jpeg" if out_path.suffix.lower()==".jpg" else "image/png"
    uri = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    return uri, W, H

# ============================== seen / labels ==============================

def load_seen(path: Path) -> set[str]:
    try:
        if path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                return set(map(str, obj))
    except Exception:
        pass
    return set()

def save_seen(path: Path, seen: set[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(list(seen))), encoding="utf-8")

def load_labels_effective(path: Path) -> dict[str, dict]:
    """
    Returns mapping key_str -> last label record (effective).
    We treat the CSV as an append-log; effective label is last row for each (image,id).
    Only rows with integer human_label are considered.
    """
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "image" not in df.columns or "id" not in df.columns:
        return {}

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=True)

    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        k = key_str(r.get("image",""), r.get("id",""))
        hl = r.get("human_label", None)
        if hl is None or (isinstance(hl, float) and np.isnan(hl)):
            continue
        try:
            hl_i = int(hl)
        except Exception:
            continue
        out[k] = {
            "human_label": int(hl_i),
            "action": str(r.get("action","")),
            "review_weight": _safe_float(r.get("review_weight", np.nan), np.nan),
            "timestamp": int(r.get("timestamp", 0)) if not pd.isna(r.get("timestamp", 0)) else 0,
        }
    return out

def append_review_row(csv_path: Path, rec: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = pd.DataFrame(columns=REVIEW_COLS)
    else:
        df = pd.DataFrame(columns=REVIEW_COLS)

    for c in REVIEW_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df = pd.concat([df, pd.DataFrame([rec], columns=REVIEW_COLS)], ignore_index=True)
    df.to_csv(csv_path, index=False)

def append_review_rows(csv_path: Path, recs: list[dict]) -> int:
    """Append many review rows in one CSV write. Returns number of rows appended."""
    if not recs:
        return 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = pd.DataFrame(columns=REVIEW_COLS)
    else:
        df = pd.DataFrame(columns=REVIEW_COLS)

    for c in REVIEW_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df = pd.concat([df, pd.DataFrame(recs, columns=REVIEW_COLS)], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return int(len(recs))

def pop_last_row(csv_path: Path) -> dict | None:
    """Pop the last row from review_labels.csv and return it as a dict (or None if empty)."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if len(df) == 0:
        return None
    last = df.iloc[-1].to_dict()
    df = df.iloc[:-1].reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    return last

def undo_last_row(csv_path: Path) -> bool:
    """Backward-compatible helper: returns True iff a row was removed."""
    return pop_last_row(csv_path) is not None

# ============================== drawing ==============================

def _shape_xy_in_mosaic(row: dict, off_x: int, off_y: int):
    poly = _parse_poly(row)
    x,y,w,h = _parse_bbox(row, poly)

    if poly is not None and len(poly) >= 3:
        pts = poly.copy()
        pts[:,0] += off_x
        pts[:,1] += off_y
        xs = pts[:,0].tolist() + [pts[0,0]]
        ys = pts[:,1].tolist() + [pts[0,1]]
    else:
        x1 = x + off_x; y1 = y + off_y
        x2 = x1 + w;    y2 = y1 + h
        xs = [x1,x2,x2,x1,x1]
        ys = [y1,y1,y2,y2,y1]
    return xs, ys

def pack_traces(
    df_base: pd.DataFrame,
    filter_mode: str,
    opts: list[str],
    labels_eff: dict[str, dict],
    seen_set: set[str],
    selected_key: str|None,
):
    show_outlines = ("out" in (opts or []))
    hide_reviewed = ("hide_rev" in (opts or []))
    hide_seen     = ("hide_seen" in (opts or []))
    only_mismatch = ("mismatch" in (opts or []))
    human_outline = ("human_outline" in (opts or []))

    # line groups
    x_model_cj=[]; y_model_cj=[]
    x_model_nj=[]; y_model_nj=[]
    x_unc=[];      y_unc=[]
    x_h_cj=[];     y_h_cj=[]
    x_h_nj=[];     y_h_nj=[]
    x_sel=[];      y_sel=[]
    x_mis=[];      y_mis=[]

    # markers
    mx=[]; my=[]; mtext=[]; mcolor=[]; mcustom=[]
    msize=[]; mlinew=[]; mlinec=[]

    for _, rr in df_base.iterrows():
        row = rr.to_dict()
        tile = str(row["image"])
        pt = parse_tile(tile)
        if not pt:
            continue
        _, oy, ox, _ = pt
        off_x, off_y = int(ox), int(oy)

        det_id = row.get("id","?")
        k = key_str(tile, det_id)

        pred, p_ui, thr_eff = compute_final_pred_and_ui(row)
        unc = is_uncertain(p_ui, thr_eff)
        reviewed = (k in labels_eff)
        seen = (k in seen_set)

        if hide_reviewed and reviewed:
            continue
        if hide_seen and seen:
            continue

        # Only mismatches QA mode
        human = None
        mismatch = False
        w_eff = np.nan
        if reviewed:
            human = int(labels_eff[k]["human_label"])
            mismatch = (human in (0,1)) and (int(human) != int(pred))
            w_eff = labels_eff[k].get("review_weight", np.nan)

        if only_mismatch:
            if not mismatch:
                continue

        # dropdown filter (applies to model-state when not reviewed, to human-state when reviewed and human in {0,1})
        disp_pred = int(pred)
        disp_unc = bool(unc)
        if reviewed and (human in (0,1)):
            disp_pred = int(human)
            disp_unc = False

        if not only_mismatch:
            if filter_mode == "uncertain" and not disp_unc:
                continue
            if filter_mode == "certain" and disp_unc:
                continue
            if filter_mode == "cj" and disp_pred != 1:
                continue
            if filter_mode == "noncj" and disp_pred != 0:
                continue
            if filter_mode == "reviewed" and not reviewed:
                continue
            if filter_mode == "unreviewed" and reviewed:
                continue

        xs, ys = _shape_xy_in_mosaic(row, off_x, off_y)

        # outlines
        if show_outlines:
            if k == selected_key:
                x_sel += xs + [None]; y_sel += ys + [None]
            elif mismatch:
                x_mis += xs + [None]; y_mis += ys + [None]
            elif reviewed and human_outline and (human in (0,1)):
                if human == 1:
                    x_h_cj += xs + [None]; y_h_cj += ys + [None]
                else:
                    x_h_nj += xs + [None]; y_h_nj += ys + [None]
            else:
                if disp_unc:
                    x_unc += xs + [None]; y_unc += ys + [None]
                elif disp_pred == 1:
                    x_model_cj += xs + [None]; y_model_cj += ys + [None]
                else:
                    x_model_nj += xs + [None]; y_model_nj += ys + [None]

        # centroid marker
        cx = (min(xs[:-1]) + max(xs[:-1]))/2.0
        cy = (min(ys[:-1]) + max(ys[:-1]))/2.0

        xgb_p = _safe_float(row.get("xgb_p", row.get("xgb_score", 0.0)), 0.0)
        yconf = _safe_float(row.get("yolo_conf", 0.0), 0.0)
        yiou  = _safe_float(row.get("yolo_iou", 0.0), 0.0)

        txt = f"id={det_id} reviewed={reviewed} seen={seen} mismatch={mismatch}"
        txt += f"<br>model_pred={'CJ' if pred==1 else 'Non-CJ'} p={('NA' if p_ui is None else f'{p_ui:.3f}')} thr={thr_eff:.2f}" + (" UNC" if unc else "")
        if reviewed:
            txt += f"<br><b>HUMAN={human}</b>  action={labels_eff[k].get('action','')}  weight={('NA' if (w_eff is None or (isinstance(w_eff,float) and np.isnan(w_eff))) else f'{float(w_eff):.2f}')}"
        txt += f"<br>tile={tile}"
        txt += f"<br>xgb_p={xgb_p:.3f} yolo={yconf:.3f}/{yiou:.3f}"

        mx.append(cx); my.append(cy); mtext.append(txt)
        mcustom.append([
            str(det_id), tile, int(pred),
            (None if p_ui is None else float(p_ui)), float(thr_eff), bool(unc),
            float(xgb_p), float(yconf), float(yiou),
            int(off_x), int(off_y),
            (None if human is None else int(human)),
            bool(mismatch),
            (None if (w_eff is None or (isinstance(w_eff,float) and np.isnan(w_eff))) else float(w_eff)),
        ])

        # marker styles
        if reviewed and (human in (0,1)):
            # reviewed: fill by human label; cyan ring; bigger dot
            msize.append(11 if mismatch else 10)
            mlinew.append(3 if mismatch else 2)
            mlinec.append("magenta" if mismatch else "cyan")
            mcolor.append("lime" if human==1 else "red")
        else:
            msize.append(6)
            mlinew.append(0)
            mlinec.append("rgba(0,0,0,0)")
            if unc:
                mcolor.append("yellow")
            elif pred==1:
                mcolor.append("lime")
            else:
                mcolor.append("red")

    traces = []

    if show_outlines:
        if x_model_cj:
            traces.append(go.Scattergl(x=x_model_cj, y=y_model_cj, mode="lines", name="CJ (model)", line=dict(width=2), hoverinfo="skip"))
        if x_model_nj:
            traces.append(go.Scattergl(x=x_model_nj, y=y_model_nj, mode="lines", name="Non-CJ (model)", line=dict(width=2), hoverinfo="skip"))
        if x_unc:
            traces.append(go.Scattergl(x=x_unc, y=y_unc, mode="lines", name="Uncertain", line=dict(width=3), hoverinfo="skip"))

        if x_h_cj:
            traces.append(go.Scattergl(x=x_h_cj, y=y_h_cj, mode="lines", name="CJ (human)", line=dict(width=4, dash="dash"), hoverinfo="skip"))
        if x_h_nj:
            traces.append(go.Scattergl(x=x_h_nj, y=y_h_nj, mode="lines", name="Non-CJ (human)", line=dict(width=4, dash="dash"), hoverinfo="skip"))

        if x_mis:
            traces.append(go.Scattergl(x=x_mis, y=y_mis, mode="lines", name="Mismatch (QA)", line=dict(width=6), hoverinfo="skip"))
        if x_sel:
            traces.append(go.Scattergl(x=x_sel, y=y_sel, mode="lines", name="Selected", line=dict(width=7), hoverinfo="skip"))

    traces.append(go.Scattergl(
        x=mx, y=my, mode="markers", name="clickable",
        marker=dict(size=msize, opacity=0.80, color=mcolor, line=dict(width=mlinew, color=mlinec)),
        text=mtext, hoverinfo="text", customdata=mcustom
    ))
    return traces

# ============================== keyboard hash ==============================

def _parse_hash_key(h: str | None) -> str | None:
    """Expected: '#k=<key>&ts=<...>' -> returns key (lower)."""
    if not h:
        return None
    s = str(h).lstrip("#").strip()
    if not s:
        return None
    parts = s.split("&")
    kv = {}
    for p in parts:
        if "=" in p:
            a, b = p.split("=", 1)
            kv[a.strip()] = b.strip()
    if "k" not in kv:
        return None
    try:
        return unquote_plus(kv["k"]).lower()
    except Exception:
        return str(kv["k"]).lower()

def _key_to_action(k: str | None) -> str | None:
    """
    Returns canonical action string:
      accept | flip | skip | sus_accept | sus_flip | undo
    """
    if not k:
        return None
    k = str(k).lower()
    if k in ("1", "a"):
        return "accept"
    if k in ("2", "r"):
        return "flip"
    if k in ("3", "s"):
        return "skip"
    if k in ("4", "w"):
        return "sus_accept"
    if k in ("5", "u"):
        return "sus_flip"
    if k in ("0", "z"):
        return "undo"
    return None

# ============================== main app ==============================

def main(
    *,
    detections_csv: Path,
    tiles_dir: Path,
    cache_dir: Path,
    review_labels_csv: Path,
    seen_json: Path,
    host: str,
    port: int,
) -> None:
    det_csv = Path(detections_csv)
    tiles_dir = Path(tiles_dir)
    cache_dir = Path(cache_dir)
    review_csv = Path(review_labels_csv)
    seen_json = Path(seen_json)

    if not det_csv.exists():
        raise SystemExit(f"DETECTIONS_CSV not found: {det_csv}")
    if not tiles_dir.exists():
        raise SystemExit(f"TILES_DIR not found: {tiles_dir}")

    df = pd.read_csv(det_csv)
    if "image" not in df.columns or "id" not in df.columns:
        raise SystemExit("detections.csv must contain columns: image, id")

    # base(s)
    tile_info = df["image"].astype(str).map(parse_tile)
    df["base"] = tile_info.map(lambda t: t[0] if t else "")
    bases = sorted([b for b in df["base"].unique().tolist() if b])
    if not bases:
        raise SystemExit("No bases found in image names (expected *_y####x####.*).")

    labels_eff0 = load_labels_effective(review_csv)
    seen0 = load_seen(seen_json)

    print(f"[LOAD] rows={len(df)} bases={len(bases)}")
    print(f"[CFG] policy={DET_POLICY}/{DET_MISSING} thr_xgb={THR_XGB} thr_yolo={THR_YOLO}/{THR_IOU} det_thr={DET_THR} margin={AL_MARGIN}")
    print(f"[STATE] labels={len(labels_eff0)} seen={len(seen0)}")
    print("[ZOOM] mouse-wheel / trackpad pinch works (scrollZoom=ON). Double-click resets.")
    print("[KEYS] 1/A=accept 2/R=flip 3/S=skip 4/W=sus_accept 5/U=sus_flip 0/Z=undo")

    def get_mosaic_for_base(base: str):
        tiles: list[tuple[str,int,int]] = []
        if SCAN_ALL_TILES_FOR_BASE:
            for p in tiles_dir.glob(f"{base}_y*x*.*"):
                pt = parse_tile(p.name)
                if pt:
                    tiles.append((p.name, pt[1], pt[2]))
        else:
            dfb = df[df["base"].astype(str)==str(base)]
            for t in dfb["image"].astype(str).unique().tolist():
                pt = parse_tile(t)
                if pt:
                    tiles.append((t, pt[1], pt[2]))

        if not tiles:
            raise RuntimeError(f"No tiles found for base={base} under {tiles_dir}")

        mosaic_path = cache_dir / f"{base}.{'jpg' if MOSAIC_FORMAT=='jpg' else 'png'}"
        if mosaic_path.exists():
            data = mosaic_path.read_bytes()
            mime = "image/jpeg" if mosaic_path.suffix.lower()==".jpg" else "image/png"
            uri = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
            im = cv2.imread(str(mosaic_path), cv2.IMREAD_COLOR)
            if im is None:
                mosaic_path.unlink(missing_ok=True)
                raise RuntimeError("Cached mosaic unreadable; delete and rebuild.")
            H, W = im.shape[:2]
            return uri, int(W), int(H)

        print(f"[BUILD] stitching mosaic for base={base} (native tile resolution)...")
        mosaic = build_mosaic(base, tiles_dir, tiles)
        uri, W, H = write_mosaic_and_get_data_uri(mosaic, mosaic_path)
        print(f"[BUILD] saved mosaic: {mosaic_path} size={W}x{H}")
        return uri, int(W), int(H)

    def _centroid_for_row_in_mosaic(row: dict):
        tile = str(row.get("image",""))
        pt = parse_tile(tile)
        if not pt:
            return None
        _, oy, ox, _ = pt
        off_x, off_y = int(ox), int(oy)
        xs, ys = _shape_xy_in_mosaic(row, off_x, off_y)
        cx = (min(xs[:-1]) + max(xs[:-1]))/2.0
        cy = (min(ys[:-1]) + max(ys[:-1]))/2.0
        return float(cx), float(cy)

    def _find_row_by_key(sel_key: str):
        try:
            tile, det_id = str(sel_key).split("||", 1)
        except Exception:
            return None
        df_match = df[(df["image"].astype(str)==str(tile)) & (df["id"].astype(str)==str(det_id))]
        if len(df_match) == 0:
            return None
        return df_match.iloc[0].to_dict()

    def _visible_centroids_for_base(
        base_now: str,
        filter_mode: str,
        opts: list[str],
        labels_eff: dict[str, dict],
        seen_set: set[str],
    ) -> dict[str, tuple[float,float]]:
        """
        Build mapping key -> (cx,cy) for dots that would be visible under current filters/options.
        Mirrors pack_traces filtering logic (but ignores outlines).
        """
        labels_eff = labels_eff or {}
        show_outlines = ("out" in (opts or []))  # unused here but kept for parity
        hide_reviewed = ("hide_rev" in (opts or []))
        hide_seen     = ("hide_seen" in (opts or []))
        only_mismatch = ("mismatch" in (opts or []))
        human_outline = ("human_outline" in (opts or []))  # unused here but kept for parity

        dfb = df[df["base"].astype(str) == str(base_now)].copy()
        out: dict[str, tuple[float,float]] = {}

        for _, rr in dfb.iterrows():
            row = rr.to_dict()
            tile = str(row.get("image",""))
            pt = parse_tile(tile)
            if not pt:
                continue
            det_id = row.get("id","?")
            k = key_str(tile, det_id)

            pred, p_ui, thr_eff = compute_final_pred_and_ui(row)
            unc = is_uncertain(p_ui, thr_eff)
            reviewed = (k in labels_eff)
            seen = (k in seen_set)

            if hide_reviewed and reviewed:
                continue
            if hide_seen and seen:
                continue

            human = None
            mismatch = False
            if reviewed:
                try:
                    human = int(labels_eff[k]["human_label"])
                except Exception:
                    human = None
                mismatch = (human in (0,1)) and (int(human) != int(pred))

            if only_mismatch and (not mismatch):
                continue

            disp_pred = int(pred)
            disp_unc = bool(unc)
            if reviewed and (human in (0,1)):
                disp_pred = int(human)
                disp_unc = False

            if not only_mismatch:
                if filter_mode == "uncertain" and not disp_unc:
                    continue
                if filter_mode == "certain" and disp_unc:
                    continue
                if filter_mode == "cj" and disp_pred != 1:
                    continue
                if filter_mode == "noncj" and disp_pred != 0:
                    continue
                if filter_mode == "reviewed" and not reviewed:
                    continue
                if filter_mode == "unreviewed" and reviewed:
                    continue

            cen = _centroid_for_row_in_mosaic(row)
            if cen is None:
                continue
            out[k] = cen

        return out

    def _nearest_key(
        sel_key: str,
        base_now: str,
        filter_mode: str,
        opts: list[str],
        labels_eff: dict[str, dict],
        seen_set: set[str],
    ) -> str | None:
        row_sel = _find_row_by_key(sel_key)
        if row_sel is None:
            return None
        cen_sel = _centroid_for_row_in_mosaic(row_sel)
        if cen_sel is None:
            return None
        sx, sy = cen_sel

        cand = _visible_centroids_for_base(base_now, filter_mode, opts, labels_eff, seen_set)
        if not cand:
            return None

        best_k = None
        best_d2 = None
        for k, (cx, cy) in cand.items():
            if str(k) == str(sel_key):
                continue
            dx = float(cx) - float(sx)
            dy = float(cy) - float(sy)
            d2 = dx*dx + dy*dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_k = k
        return best_k

    # initial base
    base0 = bases[0]
    uri0, W0, H0 = get_mosaic_for_base(base0)

    # Dash app
    app = Dash(__name__)
    app.title = "Mosaic Review"

    app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
        <script>
        (function() {{
            function shouldIgnore() {{
                var el = document.activeElement;
                if (!el) return false;
                var tag = (el.tagName || '').toLowerCase();
                if (tag === 'input' || tag === 'textarea' || tag === 'select') return true;
                if (el.isContentEditable) return true;
                return false;
            }}
            document.addEventListener('keydown', function(e) {{
                try {{
                    if (!e) return;
                    if (e.defaultPrevented) return;
                    if (e.ctrlKey || e.metaKey || e.altKey) return;
                    if (shouldIgnore()) return;

                    var k = (e.key || '').toLowerCase();
                    var allow = {{
                        '1':1,'a':1,
                        '2':1,'r':1,
                        '3':1,'s':1,
                        '4':1,'w':1,
                        '5':1,'u':1,
                        '0':1,'z':1,'l':1
                    }};
                    if (!allow[k]) return;

                    var ts = Date.now();
                    window.location.hash = 'k=' + encodeURIComponent(k) + '&ts=' + ts;

                    e.preventDefault();
                    e.stopPropagation();
                }} catch (err) {{
                }}
            }}, true);
        }})();
        </script>
    </body>
</html>
"""

    def make_figure(meta, filter_mode, opts, labels_eff, seen_list, selected, dragmode, view, view_rev=0):
        base = meta["base"]; uri = meta["uri"]; W = int(meta["W"]); H = int(meta["H"])
        dfb = df[df["base"].astype(str)==str(base)].copy()
        labels_eff = labels_eff or {}
        seen_set = set(map(str, seen_list or []))

        fig = go.Figure()
        fig.add_layout_image(dict(
            source=uri,
            xref="x", yref="y",
            x=0, y=0,
            sizex=W, sizey=H,
            xanchor="left", yanchor="top",
            layer="below"
        ))

        traces = pack_traces(dfb, filter_mode, opts or [], labels_eff, seen_set, selected)
        for t in traces:
            fig.add_trace(t)

        x0, x1 = 0.0, float(W)
        y0, y1 = 0.0, float(H)
        if isinstance(view, dict) and str(view.get("base","")) == str(base):
            try:
                if "x0" in view and "x1" in view:
                    x0, x1 = float(view["x0"]), float(view["x1"])
                if "y0" in view and "y1" in view:
                    y0, y1 = float(view["y0"]), float(view["y1"])
            except Exception:
                pass


        # Canonicalize ranges (Plotly may emit reversed ranges on some interactions).
        try:
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
        except Exception:
            pass
        x0 = float(max(0.0, min(float(W), x0)))
        x1 = float(max(0.0, min(float(W), x1)))
        y0 = float(max(0.0, min(float(H), y0)))
        y1 = float(max(0.0, min(float(H), y1)))

        # Follow selection: pan so the selected object becomes visible, while preserving zoom.
        # We keep the current window size (zoom level) and apply the *minimal* shift that
        # guarantees the selected object's bbox is within view (with a small padding),
        # then clamp to image bounds.
        if selected:
            try:
                row_sel = _find_row_by_key(str(selected))
            except Exception:
                row_sel = None

            if row_sel is not None:
                try:
                    tile = str(row_sel.get("image", ""))
                    pt = parse_tile(tile)
                except Exception:
                    pt = None

                if pt:
                    try:
                        _, oy, ox, _ = pt
                        off_x, off_y = int(ox), int(oy)
                        xs, ys = _shape_xy_in_mosaic(row_sel, off_x, off_y)
                        xmin = float(min(xs[:-1])); xmax = float(max(xs[:-1]))
                        ymin = float(min(ys[:-1])); ymax = float(max(ys[:-1]))

                        # Current view window (canonical).
                        x_lo, x_hi = float(x0), float(x1)
                        y_lo, y_hi = float(y0), float(y1)
                        w = float(max(1.0, x_hi - x_lo))
                        h = float(max(1.0, y_hi - y_lo))
                        w = min(w, float(W))
                        h = min(h, float(H))

                        # Padding so the object isn't glued to the edge after panning.
                        pad_x = float(max(10.0, min(w * 0.08, 80.0)))
                        pad_y = float(max(10.0, min(h * 0.08, 80.0)))

                        # Minimal shift that puts [xmin,xmax] inside [x_lo+pad, x_hi-pad], if feasible.
                        if (xmax - xmin) > (w - 2.0 * pad_x):
                            dx = ((xmin + xmax) / 2.0) - ((x_lo + x_hi) / 2.0)
                        else:
                            lower = (xmax + pad_x) - x_hi   # need dx >= lower
                            upper = (xmin - pad_x) - x_lo   # need dx <= upper
                            if lower > upper:
                                dx = ((xmin + xmax) / 2.0) - ((x_lo + x_hi) / 2.0)
                            else:
                                dx = 0.0
                                dx = min(max(dx, lower), upper)

                        if (ymax - ymin) > (h - 2.0 * pad_y):
                            dy = ((ymin + ymax) / 2.0) - ((y_lo + y_hi) / 2.0)
                        else:
                            lower = (ymax + pad_y) - y_hi
                            upper = (ymin - pad_y) - y_lo
                            if lower > upper:
                                dy = ((ymin + ymax) / 2.0) - ((y_lo + y_hi) / 2.0)
                            else:
                                dy = 0.0
                                dy = min(max(dy, lower), upper)

                        x0n = x_lo + dx
                        x1n = x0n + w
                        y0n = y_lo + dy
                        y1n = y0n + h

                        # Clamp to image bounds, keeping window size when possible.
                        if x0n < 0.0:
                            x1n -= x0n
                            x0n = 0.0
                        if x1n > float(W):
                            x0n -= (x1n - float(W))
                            x1n = float(W)
                        if y0n < 0.0:
                            y1n -= y0n
                            y0n = 0.0
                        if y1n > float(H):
                            y0n -= (y1n - float(H))
                            y1n = float(H)

                        # Final safety clamp.
                        x0, x1 = float(max(0.0, x0n)), float(min(float(W), x1n))
                        y0, y1 = float(max(0.0, y0n)), float(min(float(H), y1n))
                    except Exception:
                        pass
        fig.update_xaxes(range=[x0, x1], visible=False)
        fig.update_yaxes(range=[y1, y0], visible=False, autorange=False, scaleanchor="x", scaleratio=1)

        fig.update_layout(
            uirevision=str(base) + "|" + str(selected or ""),
            margin=dict(l=0,r=0,t=0,b=0),
            dragmode=str(dragmode or "pan"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
        )
        return fig

    meta0 = {"base": base0, "uri": uri0, "W": W0, "H": H0}
    fig0 = make_figure(meta0, "all", ["out","hide_seen","human_outline","auto_next"], labels_eff0, sorted(list(seen0)), None, "pan", None, view_rev=0)

    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="labels_store", data=labels_eff0),
        dcc.Store(id="seen_store", data=sorted(list(seen0))),
        dcc.Store(id="selected_store", data=None),
        dcc.Store(id="mosaic_meta_store", data=meta0),
        dcc.Store(id="view_store", data=None),
        dcc.Store(id="action_pulse_store", data=None),
        dcc.Store(id="zoom_lock_store", data=True),
        dcc.Store(id="view_rev_store", data=0),

        html.Div([
            html.Div([
                html.Span("Base:", style={"marginRight":"6px"}),
                dcc.Dropdown(
                    id="base_dd",
                    options=[{"label":b, "value":b} for b in bases],
                    value=base0,
                    clearable=False,
                    style={"width":"260px"}
                ),
                html.Span("Filter:", style={"marginLeft":"12px", "marginRight":"6px"}),
                dcc.Dropdown(
                    id="filter",
                    options=[
                        {"label":"All", "value":"all"},
                        {"label":"Uncertain (near threshold)", "value":"uncertain"},
                        {"label":"Certain (far from threshold)", "value":"certain"},
                        {"label":"CJ", "value":"cj"},
                        {"label":"Non-CJ", "value":"noncj"},
                        {"label":"Reviewed", "value":"reviewed"},
                        {"label":"Unreviewed", "value":"unreviewed"},
                    ],
                    value="all",
                    clearable=False,
                    style={"width":"220px"}
                ),
                html.Span("Drag:", style={"marginLeft":"12px", "marginRight":"6px"}),
                dcc.RadioItems(
                    id="dragmode",
                    options=[{"label":"Pan", "value":"pan"}, {"label":"Zoom", "value":"zoom"}],
                    value="pan",
                    inline=True,
                ),
                dcc.Checklist(
                    id="opts",
                    options=[
                        {"label":"Show outlines (polygons/bboxes)", "value":"out"},
                        {"label":"Hide reviewed dots", "value":"hide_rev"},
                        {"label":"Hide seen/skip dots", "value":"hide_seen"},
                        {"label":"Only mismatches (QA)", "value":"mismatch"},
                        {"label":"Human-style outlines for reviewed", "value":"human_outline"},
                        {"label":"Auto-next selection after action", "value":"auto_next"},
                    ],
                    value=["out","hide_seen","human_outline","auto_next"],
                    style={"marginLeft":"12px", "display":"inline-block"}
                ),
            ], style={"display":"flex", "alignItems":"center", "flexWrap":"wrap", "gap":"10px"}),

            html.Div(id="status_line", style={"fontFamily":"monospace", "marginTop":"8px"}),

            html.Div([
                html.Div(
                    "Actions (select a dot first). Keyboard: 1/A accept, 2/R flip, 3/S skip, 4/W sus-accept, 5/U sus-flip, 0/Z undo.",
                    style={"fontFamily":"monospace", "marginBottom":"6px"}
                ),
                html.Button("Accept (correct det)", id="btn_accept", n_clicks=0, title="Keep model label. action=accept, weight=1.0  (keys: 1/A)"),
                html.Button("Flip (incorrect det)", id="btn_flip", n_clicks=0, title="Flip model label. action=flip, weight=1.0  (keys: 2/R)", style={"marginLeft":"8px"}),
                html.Button("Sus accept (uncertain - keep label)", id="btn_sus_acc", n_clicks=0, title="Keep model label with lower weight. action=sus_accept, weight=0.4  (keys: 4/W)", style={"marginLeft":"8px"}),
                html.Button("Sus flip (uncertain - flip label)", id="btn_sus_flip", n_clicks=0, title="Flip model label with lower weight. action=sus_flip, weight=0.4  (keys: 5/U)", style={"marginLeft":"8px"}),
                html.Button("Skip (seen only)", id="btn_skip", n_clicks=0, title="Mark as seen (logged to review_labels.csv, weight=0.0).  (keys: 3/S)", style={"marginLeft":"14px"}),
                html.Button("Accept remaining (auto, this base)", id="btn_accept_rest", n_clicks=0, title="Auto-accept remaining unreviewed+unseen items in current base (logged as action=auto_accept; excluded from effort).", style={"marginLeft":"8px"}),
                html.Button("Unsee (remove from seen)", id="btn_unsee", n_clicks=0, title="Remove this item from seen.json.", style={"marginLeft":"8px"}),
                html.Button("Undo last write", id="btn_undo", n_clicks=0, title="Removes only the last row in review_labels.csv.  (keys: 0/Z)", style={"marginLeft":"14px"}),
                html.Button("Zoom lock: ON (L)", id="btn_zoom_lock", n_clicks=0, title="Lock current zoom (prevents double-click reset) and auto-pan to selected object without changing zoom.", style={"marginLeft":"8px"}),
            ], style={"marginTop":"8px"}),

            html.Div(id="selected_info", style={"fontFamily":"monospace", "marginTop":"8px"}),
            html.Div(id="action_msg", style={"fontFamily":"monospace", "marginTop":"6px"}),
        ], style={"padding":"10px"}),

        dcc.Graph(
            id="graph",
            figure=fig0,
            style={"height":"90vh"},
            config={
                "scrollZoom": True,
                "doubleClick": "reset",
                "displaylogo": False,
            },
        ),
    ])

    @app.callback(
        Output("mosaic_meta_store","data"),
        Output("view_store","data"),
        Output("view_rev_store","data"),
        Input("base_dd","value"),
        Input("graph","relayoutData"),
        Input("selected_store","data"),
        State("mosaic_meta_store","data"),
        State("view_store","data"),
        State("view_rev_store","data"),
        State("zoom_lock_store","data"),
        prevent_initial_call=True,
    )
    def on_base_or_view(base, relayout, selected, meta_prev, view_prev, view_rev_prev, zoom_lock):
        trig = (ctx.triggered_id or "")
        view_rev_prev = int(view_rev_prev or 0)
        zoom_lock = bool(zoom_lock)

        # ----------------- Base changed -----------------
        if trig == "base_dd":
            if not base:
                return meta_prev, None, 0
            try:
                uri, W, H = get_mosaic_for_base(base)
                meta_new = {"base": base, "uri": uri, "W": int(W), "H": int(H)}
                return meta_new, None, 0
            except Exception as e:
                print(f"[ERR] base change failed: {e}")
                return meta_prev, None, 0

        # ----------------- User zoom/pan (relayout) -----------------
        if trig == "graph":
            if not relayout:
                return no_update, view_prev, view_rev_prev

            # Note: plotly may send yaxis.autorange="reversed" (string) always; treat only boolean True as reset.
            xauto = relayout.get("xaxis.autorange", None)
            yauto = relayout.get("yaxis.autorange", None)
            is_reset = (xauto is True) or (yauto is True)
            if is_reset:
                # If zoom-lock is ON, immediately restore previous view (no reset).
                if zoom_lock and isinstance(view_prev, dict) and view_prev:
                    return no_update, view_prev, (view_rev_prev + 1)
                return no_update, None, view_rev_prev

            def _get_range(axis: str):
                k0 = f"{axis}.range[0]"
                k1 = f"{axis}.range[1]"
                if k0 in relayout and k1 in relayout:
                    try:
                        return float(relayout[k0]), float(relayout[k1])
                    except Exception:
                        return None
                k = f"{axis}.range"
                if k in relayout and isinstance(relayout[k], (list, tuple)) and len(relayout[k]) == 2:
                    try:
                        return float(relayout[k][0]), float(relayout[k][1])
                    except Exception:
                        return None
                return None

            xr = _get_range("xaxis")
            yr = _get_range("yaxis")
            if xr is None and yr is None:
                return no_update, view_prev, view_rev_prev

            base_now = str((meta_prev or {}).get("base", ""))
            out = dict(view_prev) if (isinstance(view_prev, dict) and view_prev and str(view_prev.get("base","")) == base_now) else {"base": base_now}
            if xr is not None:
                x0_, x1_ = float(xr[0]), float(xr[1])
                out["x0"], out["x1"] = (min(x0_, x1_), max(x0_, x1_))
            if yr is not None:
                y0_, y1_ = float(yr[0]), float(yr[1])
                out["y0"], out["y1"] = (min(y0_, y1_), max(y0_, y1_))

            # De-jitter: if essentially unchanged, keep previous object to avoid loops.
            def _same(a, b, eps=1e-3):
                try:
                    return abs(float(a) - float(b)) <= eps
                except Exception:
                    return False
            if isinstance(view_prev, dict) and view_prev and str(view_prev.get("base","")) == base_now:
                ok = True
                for k in ("x0","x1","y0","y1"):
                    if (k in out) and (k in view_prev) and (not _same(out[k], view_prev[k])):
                        ok = False
                        break
                # If out has fewer keys (e.g., only x), still update.
                if ok and set(out.keys()) <= set(view_prev.keys()):
                    return no_update, view_prev, view_rev_prev

            return no_update, out, view_rev_prev

        # ----------------- Selection changed: (optional) follow without changing zoom -----------------
        if trig == "selected_store":
            # follow/pan only when zoom-lock is ON (so the toggle has a clear effect)
            if not zoom_lock:
                return no_update, view_prev, view_rev_prev

            sel_key = str(selected or "").strip()
            if not sel_key:
                return no_update, view_prev, view_rev_prev

            meta_now = meta_prev or {}
            base_now = str(meta_now.get("base", ""))
            if not base_now:
                return no_update, view_prev, view_rev_prev

            # Need a current view window to preserve zoom; if none, do not auto-zoom.
            if not (isinstance(view_prev, dict) and view_prev and str(view_prev.get("base","")) == base_now):
                return no_update, view_prev, view_rev_prev

            row = _find_row_by_key(sel_key)
            if row is None:
                return no_update, view_prev, view_rev_prev

            # Ensure selected belongs to current base
            tile = str(row.get("image",""))
            pt = parse_tile(tile)
            if (not pt) or (str(pt[0]) != base_now):
                return no_update, view_prev, view_rev_prev
            _, oy, ox, _ = pt
            off_x, off_y = int(ox), int(oy)
            xs, ys = _shape_xy_in_mosaic(row, off_x, off_y)
            if (not xs) or (not ys):
                return no_update, view_prev, view_rev_prev

            # bbox in mosaic coords
            xmn = float(min(xs[:-1] if len(xs) > 1 else xs))
            xmx = float(max(xs[:-1] if len(xs) > 1 else xs))
            ymn = float(min(ys[:-1] if len(ys) > 1 else ys))
            ymx = float(max(ys[:-1] if len(ys) > 1 else ys))

            try:
                W = float(meta_now.get("W", 0))
                H = float(meta_now.get("H", 0))
            except Exception:
                W, H = 0.0, 0.0
            if W <= 0 or H <= 0:
                return no_update, view_prev, view_rev_prev

            # Current window (canonical: x0<x1, y0<y1)
            try:
                x0 = float(view_prev.get("x0", 0.0)); x1 = float(view_prev.get("x1", W))
                y0 = float(view_prev.get("y0", 0.0)); y1 = float(view_prev.get("y1", H))
            except Exception:
                return no_update, view_prev, view_rev_prev
            x0, x1 = (min(x0, x1), max(x0, x1))
            y0, y1 = (min(y0, y1), max(y0, y1))
            vw = float(x1 - x0); vh = float(y1 - y0)
            if vw <= 2 or vh <= 2:
                return no_update, view_prev, view_rev_prev

            # Padding for comfortable visibility
            pad_x = min(max(20.0, 0.06 * vw), 0.25 * vw)
            pad_y = min(max(20.0, 0.06 * vh), 0.25 * vh)

            inside = (xmn >= x0 + pad_x) and (xmx <= x1 - pad_x) and (ymn >= y0 + pad_y) and (ymx <= y1 - pad_y)
            if inside:
                return no_update, view_prev, view_rev_prev

            def _shift_axis(curr0: float, win: float, bb0: float, bb1: float, pad: float, maxv: float) -> float:
                if win >= maxv:
                    return 0.0
                lo = bb1 + pad - win
                hi = bb0 - pad
                if lo > hi:
                    new0 = (bb0 + bb1) / 2.0 - win / 2.0
                else:
                    # minimal move: keep current if possible, else clamp into feasible interval
                    new0 = min(max(curr0, lo), hi)
                new0 = max(0.0, min(float(new0), float(maxv - win)))
                return float(new0)

            new_x0 = _shift_axis(x0, vw, xmn, xmx, pad_x, W)
            new_y0 = _shift_axis(y0, vh, ymn, ymx, pad_y, H)
            new_view = {
                "base": base_now,
                "x0": float(new_x0),
                "x1": float(new_x0 + vw),
                "y0": float(new_y0),
                "y1": float(new_y0 + vh),
            }

            # If no effective change, keep.
            def _same2(a, b, eps=1e-3):
                try:
                    return abs(float(a) - float(b)) <= eps
                except Exception:
                    return False
            if all(_same2(new_view.get(k), view_prev.get(k)) for k in ("x0","x1","y0","y1")):
                return no_update, view_prev, view_rev_prev

            return no_update, new_view, (view_rev_prev + 1)

        return no_update, view_prev, view_rev_prev
    @app.callback(
        Output("selected_store","data"),
        Input("graph","clickData"),
        Input("action_pulse_store","data"),
        State("selected_store","data"),
        State("mosaic_meta_store","data"),
        State("filter","value"),
        State("opts","value"),
        State("labels_store","data"),
        State("seen_store","data"),
        prevent_initial_call=True,
    )
    def update_selection(clickData, pulse, prev_sel, meta, filt, opts, labels_eff, seen_list):
        trig = (ctx.triggered_id or "")

        if trig == "graph":
            if not clickData:
                return prev_sel
            try:
                pt = clickData["points"][0]
                c = pt.get("customdata", None)
                if not c:
                    return prev_sel
                det_id, tile = c[0], c[1]
                return key_str(tile, det_id)
            except Exception:
                return prev_sel

        if trig == "action_pulse_store":
            if not pulse or not isinstance(pulse, dict):
                return prev_sel
            if not bool(pulse.get("do_auto_next", False)):
                return prev_sel

            opts = opts or []
            if "auto_next" not in opts:
                return prev_sel

            sel_key = str(pulse.get("sel") or prev_sel or "").strip()
            if not sel_key:
                return prev_sel

            base_now = str((meta or {}).get("base","")).strip()
            if not base_now:
                return prev_sel

            seen_set = set(map(str, seen_list or []))
            next_k = _nearest_key(sel_key, base_now, str(filt or "all"), opts, labels_eff or {}, seen_set)
            if next_k:
                return next_k
            return None

        return prev_sel

    @app.callback(
        Output("selected_info","children"),
        Input("selected_store","data"),
        State("labels_store","data"),
        State("seen_store","data"),
    )
    def update_selected_info(sel, labels_eff, seen_list):
        if not sel:
            return "Click a dot (centroid) to select an object."
        labels_eff = labels_eff or {}
        seen_set = set(map(str, seen_list or []))

        row = _find_row_by_key(str(sel))
        if row is None:
            return f"SELECTED key={sel} | [ERR] cannot find row in detections."

        try:
            tile, det_id = str(sel).split("||", 1)
        except Exception:
            tile, det_id = str(row.get("image","")), str(row.get("id","?"))

        pred, p_ui, thr_eff = compute_final_pred_and_ui(row)
        unc = is_uncertain(p_ui, thr_eff)

        xgb_p = _safe_float(row.get("xgb_p", row.get("xgb_score", 0.0)), 0.0)
        yconf = _safe_float(row.get("yolo_conf", 0.0), 0.0)
        yiou  = _safe_float(row.get("yolo_iou", 0.0), 0.0)

        reviewed = (str(sel) in labels_eff)
        seen = (str(sel) in seen_set)

        human_prev = None
        mismatch = False
        w_prev = None
        act_prev = ""
        if reviewed:
            try:
                human_prev = int(labels_eff[str(sel)]["human_label"])
            except Exception:
                human_prev = None
            act_prev = str(labels_eff[str(sel)].get("action",""))
            w_prev = labels_eff[str(sel)].get("review_weight", None)
            mismatch = (human_prev in (0,1)) and (int(human_prev) != int(pred))

        msg = f"SELECTED key={sel} | pred={pred} unc={unc} p={p_ui} thr={thr_eff} xgb_p={xgb_p} yolo={yconf}/{yiou} reviewed={reviewed} seen={seen}"
        if reviewed:
            msg += f" | HUMAN(prev)={human_prev} action={act_prev} mismatch={mismatch} w={w_prev}"
        return msg

    @app.callback(
        Output("labels_store","data"),
        Output("seen_store","data"),
        Output("action_msg","children"),
        Output("action_pulse_store","data"),
        Input("btn_accept","n_clicks"),
        Input("btn_flip","n_clicks"),
        Input("btn_sus_acc","n_clicks"),
        Input("btn_sus_flip","n_clicks"),
        Input("btn_skip","n_clicks"),
        Input("btn_accept_rest","n_clicks"),
        Input("btn_unsee","n_clicks"),
        Input("btn_undo","n_clicks"),
        Input("url","hash"),
        State("selected_store","data"),
        State("labels_store","data"),
        State("seen_store","data"),
        State("mosaic_meta_store","data"),
        prevent_initial_call=True,
    )
    def on_action(n_acc, n_flip, n_sus_acc, n_sus_flip, n_skip, n_bulk, n_unsee, n_undo, url_hash, sel, labels_eff, seen_list, meta):
        labels_eff = labels_eff or {}
        seen_set = set(map(str, seen_list or []))

        trig = (ctx.triggered_id or "")
        if not trig:
            return labels_eff, sorted(list(seen_set)), no_update, no_update

        if trig == "url":
            k = _parse_hash_key(url_hash)
            act = _key_to_action(k)
            if not act:
                return labels_eff, sorted(list(seen_set)), no_update, no_update
            if act == "accept":
                trig = "btn_accept"
            elif act == "flip":
                trig = "btn_flip"
            elif act == "sus_accept":
                trig = "btn_sus_acc"
            elif act == "sus_flip":
                trig = "btn_sus_flip"
            elif act == "skip":
                trig = "btn_skip"
            elif act == "undo":
                trig = "btn_undo"
            else:
                return labels_eff, sorted(list(seen_set)), no_update, no_update

        if trig == "btn_undo":
            last = pop_last_row(review_csv)
            if last is None:
                msg = "[UNDO] nothing to undo."
            else:
                act_last = str(last.get("action", "")).lower().strip()
                try:
                    k_undo = key_str(str(last.get("image", "")), str(last.get("id", "")))
                except Exception:
                    k_undo = None
                if act_last == "skip" and k_undo and (k_undo in seen_set):
                    seen_set.remove(k_undo)
                    save_seen(seen_json, seen_set)
                    msg = f"[UNDO] removed last row (action=skip) and un-saw: {k_undo}"
                else:
                    msg = f"[UNDO] removed last row (action={act_last or 'unknown'})."
            labels_eff = load_labels_effective(review_csv)
            pulse = {"ts": int(time.time()*1000), "do_auto_next": False, "sel": sel}
            return labels_eff, sorted(list(seen_set)), msg, pulse

        if trig == "btn_accept_rest":
            try:
                base_now = str((meta or {}).get("base", "")).strip()
            except Exception:
                base_now = ""
            if not base_now:
                pulse = {"ts": int(time.time()*1000), "do_auto_next": False, "sel": sel}
                return labels_eff, sorted(list(seen_set)), "[BULK] No base selected.", pulse

            dfb = df[df["base"].astype(str) == base_now].copy()
            if len(dfb) == 0:
                pulse = {"ts": int(time.time()*1000), "do_auto_next": False, "sel": sel}
                return labels_eff, sorted(list(seen_set)), f"[BULK] base={base_now}: no rows.", pulse

            recs = []
            n_reviewed = 0
            n_seen = 0
            for _, rr in dfb.iterrows():
                row = rr.to_dict()
                tile = str(row.get("image", ""))
                det_id = str(row.get("id", ""))
                k = key_str(tile, det_id)

                if k in labels_eff:
                    n_reviewed += 1
                    continue
                if k in seen_set:
                    n_seen += 1
                    continue

                pred, p_ui, thr_eff = compute_final_pred_and_ui(row)

                poly = _parse_poly(row)
                bx, by, bw, bh = _parse_bbox(row, poly)
                cx_tile = int(round(bx + bw / 2.0))
                cy_tile = int(round(by + bh / 2.0))

                xgb_p = _safe_float(row.get("xgb_p", row.get("xgb_score", 0.0)), 0.0)
                yconf = _safe_float(row.get("yolo_conf", 0.0), 0.0)
                yiou  = _safe_float(row.get("yolo_iou", 0.0), 0.0)
                kept = int(_safe_float(row.get("kept", -1), -1))

                recs.append({
                    "image": str(tile),
                    "id": str(det_id),
                    "human_label": int(pred),
                    "action": "auto_accept",
                    "review_weight": float(WEIGHT_MAP.get("auto_accept", 0.0)),
                    "click_x": int(cx_tile),
                    "click_y": int(cy_tile),
                    "prob": (None if p_ui is None else float(p_ui)),
                    "xgb_prob": float(xgb_p),
                    "final_pred": int(pred),
                    "xgb_pred": int(float(xgb_p) >= float(THR_XGB)),
                    "kept": int(kept),
                    "yolo_conf": float(yconf),
                    "yolo_iou": float(yiou),
                    "det_policy": str(DET_POLICY),
                    "det_missing": str(DET_MISSING),
                    "thr_xgb": float(THR_XGB),
                    "thr_yolo": float(THR_YOLO),
                    "thr_iou": float(THR_IOU),
                    "det_thr": float(DET_THR),
                    "hybrid_yolo_bias": float(HYBRID_YOLO_BIAS),
                    "use_xgb": bool(USE_XGB),
                    "use_yolo": bool(USE_YOLO),
                    "timestamp": now_ts_compact(),
                })

            n_app = append_review_rows(review_csv, recs)
            if n_app > 0:
                labels_eff = load_labels_effective(review_csv)

            msg = (
                f"[BULK] base={base_now}: auto-accepted {n_app} remaining (action=auto_accept; excluded from effort). "
                f"(already reviewed={n_reviewed}, seen/skip={n_seen}). "
                f"Note: Undo removes only the last CSV row."
            )
            pulse = {"ts": int(time.time()*1000), "do_auto_next": False, "sel": sel}
            return labels_eff, sorted(list(seen_set)), msg, pulse

        if not sel:
            pulse = {"ts": int(time.time()*1000), "do_auto_next": False, "sel": sel}
            return labels_eff, sorted(list(seen_set)), "No selection. Click a dot first.", pulse

        if trig == "btn_unsee":
            if str(sel) in seen_set:
                seen_set.remove(str(sel))
                save_seen(seen_json, seen_set)
                msg = f"[SEEN] removed {sel}."
            else:
                msg = f"[SEEN] {sel} was not in seen."
            pulse = {"ts": int(time.time()*1000), "do_auto_next": False, "sel": sel}
            return labels_eff, sorted(list(seen_set)), msg, pulse

        try:
            tile, det_id = str(sel).split("||", 1)
        except Exception:
            pulse = {"ts": int(time.time()*1000), "do_auto_next": False, "sel": sel}
            return labels_eff, sorted(list(seen_set)), f"[ERR] bad selected key: {sel}", pulse

        df_match = df[(df["image"].astype(str)==str(tile)) & (df["id"].astype(str)==str(det_id))]
        if len(df_match) == 0:
            pulse = {"ts": int(time.time()*1000), "do_auto_next": False, "sel": sel}
            return labels_eff, sorted(list(seen_set)), f"[ERR] cannot find row in detections for {sel}", pulse
        row = df_match.iloc[0].to_dict()

        pred, p_ui, thr_eff = compute_final_pred_and_ui(row)
        xgb_p = _safe_float(row.get("xgb_p", row.get("xgb_score", 0.0)), 0.0)
        yconf = _safe_float(row.get("yolo_conf", 0.0), 0.0)
        yiou  = _safe_float(row.get("yolo_iou", 0.0), 0.0)
        kept = int(_safe_float(row.get("kept", -1), -1))

        poly = _parse_poly(row)
        bx, by, bw, bh = _parse_bbox(row, poly)
        cx_tile = int(round(bx + bw/2.0))
        cy_tile = int(round(by + bh/2.0))

        # Skip: mark as seen + append a CSV log row (weight=0.0), so Undo works like other actions.
        if trig == "btn_skip":
            rec = {
                "image": str(tile),
                "id": str(det_id),
                "human_label": np.nan,
                "action": "skip",
                "review_weight": float(WEIGHT_MAP.get("skip", 0.0)),
                "click_x": int(cx_tile),
                "click_y": int(cy_tile),
                "prob": (None if p_ui is None else float(p_ui)),
                "xgb_prob": float(xgb_p),
                "final_pred": int(pred),
                "xgb_pred": int(float(xgb_p) >= float(THR_XGB)),
                "kept": int(kept),
                "yolo_conf": float(yconf),
                "yolo_iou": float(yiou),
                "det_policy": str(DET_POLICY),
                "det_missing": str(DET_MISSING),
                "thr_xgb": float(THR_XGB),
                "thr_yolo": float(THR_YOLO),
                "thr_iou": float(THR_IOU),
                "det_thr": float(DET_THR),
                "hybrid_yolo_bias": float(HYBRID_YOLO_BIAS),
                "use_xgb": bool(USE_XGB),
                "use_yolo": bool(USE_YOLO),
                "timestamp": now_ts_compact(),
            }
            append_review_row(review_csv, rec)
            seen_set.add(str(sel))
            save_seen(seen_json, seen_set)
            msg = f"[SKIP] logged + marked {sel} as seen (appended to review_labels.csv, weight=0.0)."
            pulse = {"ts": int(time.time()*1000), "do_auto_next": True, "sel": sel}
            return labels_eff, sorted(list(seen_set)), msg, pulse

        if trig == "btn_accept":
            human = int(pred)
            action = "accept"
        elif trig == "btn_flip":
            human = int(1 - int(pred))
            action = "flip"
        elif trig == "btn_sus_acc":
            human = int(pred)
            action = "sus_accept"
        elif trig == "btn_sus_flip":
            human = int(1 - int(pred))
            action = "sus_flip"
        else:
            pulse = {"ts": int(time.time()*1000), "do_auto_next": False, "sel": sel}
            return labels_eff, sorted(list(seen_set)), no_update, pulse

        w = float(WEIGHT_MAP.get(str(action), 1.0))

        rec = {
            "image": str(tile),
            "id": str(det_id),
            "human_label": int(human),
            "action": action,
            "review_weight": float(w),
            "click_x": int(cx_tile),
            "click_y": int(cy_tile),
            "prob": (None if p_ui is None else float(p_ui)),
            "xgb_prob": float(xgb_p),
            "final_pred": int(pred),
            "xgb_pred": int(float(xgb_p) >= float(THR_XGB)),
            "kept": int(kept),
            "yolo_conf": float(yconf),
            "yolo_iou": float(yiou),
            "det_policy": str(DET_POLICY),
            "det_missing": str(DET_MISSING),
            "thr_xgb": float(THR_XGB),
            "thr_yolo": float(THR_YOLO),
            "thr_iou": float(THR_IOU),
            "det_thr": float(DET_THR),
            "hybrid_yolo_bias": float(HYBRID_YOLO_BIAS),
            "use_xgb": bool(USE_XGB),
            "use_yolo": bool(USE_YOLO),
            "timestamp": now_ts_compact(),
        }
        append_review_row(review_csv, rec)
        labels_eff = load_labels_effective(review_csv)
        msg = f"[SAVE] {action} -> human={human} w={w:.2f} for {sel}  (appended to review_labels.csv)"
        pulse = {"ts": int(time.time()*1000), "do_auto_next": True, "sel": sel}
        return labels_eff, sorted(list(seen_set)), msg, pulse


    @app.callback(
        Output("zoom_lock_store","data"),
        Output("btn_zoom_lock","children"),
        Input("btn_zoom_lock","n_clicks"),
        Input("url","hash"),
        State("zoom_lock_store","data"),
        prevent_initial_call=True,
    )
    def on_zoom_lock(n_clicks, url_hash, prev_lock):
        trig = (ctx.triggered_id or "")
        # Keyboard toggle (L) comes in via URL hash
        if trig == "url":
            k = _parse_hash_key(url_hash)
            if (k or "").lower() != "l":
                return no_update, no_update
        new_lock = (not bool(prev_lock))
        label = ("Zoom lock: ON (L)" if new_lock else "Zoom lock: OFF (L)")
        return new_lock, label

    @app.callback(
        Output("graph","figure"),
        Output("status_line","children"),
        Input("mosaic_meta_store","data"),
        Input("filter","value"),
        Input("opts","value"),
        Input("labels_store","data"),
        Input("seen_store","data"),
        Input("selected_store","data"),
        Input("dragmode","value"),
        Input("zoom_lock_store","data"),
        Input("view_rev_store","data"),
        State("view_store","data"),
    )
    def update_fig(meta, filt, opts, labels_eff, seen_list, selected, dragmode, zoom_lock, view_rev, view):
        try:
            fig = make_figure(meta, filt, opts, labels_eff, seen_list, selected, dragmode, view, view_rev=view_rev)
            base = meta["base"]; W = int(meta["W"]); H = int(meta["H"])
            dfb = df[df["base"].astype(str)==str(base)]
            status = (
                f"base={base} size={W}x{H} rows={len(dfb)} "
                f"labels={len(labels_eff or {})} seen={len(seen_list or [])} "
                f"policy={DET_POLICY}/{DET_MISSING}  scrollZoom=ON  zoom_lock={int(bool(zoom_lock))}  view_rev={int(view_rev or 0)}"
            )
            return fig, status
        except Exception as e:
            print(f"[ERR] update_fig: {e}")
            return no_update, f"[ERR] update_fig crashed: {e}"

    print("[RUN] loopback review server started")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    raise SystemExit("Use the typed compag-curation review handler")
