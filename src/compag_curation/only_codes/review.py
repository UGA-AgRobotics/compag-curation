"""Legacy review state: exact port of ``full_image_review_gui.py`` semantics.

Only the Dash/Plotly presentation is replaced (see :mod:`review_web`).  The
decision shown to the reviewer, the effective-label rule, the append-only
``review_labels.csv`` writer (pandas read -> concat -> write), ``seen.json``,
skip/undo/auto-accept/unsee behaviour, click coordinates and every recorded
column are the original code paths, parameterized by the recorded GUI
constants instead of module globals.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .contract import REFERENCE_REVIEW_GUI

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None

TS_TZ = "America/New_York"


def now_ts_compact() -> int:
    if ZoneInfo is not None:
        try:
            return int(datetime.now(ZoneInfo(TS_TZ)).strftime("%Y%m%d%H%M%S"))
        except Exception:
            pass
    return int(datetime.now().strftime("%Y%m%d%H%M%S"))


_TILE_PAT = re.compile(r"^(?P<base>.+?)_y(?P<y>\d{1,8})x(?P<x>\d{1,8})\.(?P<ext>[^.]+)$", re.IGNORECASE)

WEIGHT_MAP = dict(REFERENCE_REVIEW_GUI["weight_map"])

REVIEW_COLS = [
    "image", "id", "human_label", "action", "review_weight", "click_x", "click_y",
    "prob", "xgb_prob", "final_pred", "xgb_pred", "kept",
    "yolo_conf", "yolo_iou",
    "det_policy", "det_missing",
    "thr_xgb", "thr_yolo", "thr_iou", "det_thr", "hybrid_yolo_bias",
    "use_xgb", "use_yolo", "timestamp",
]


@dataclass(frozen=True)
class GuiParams:
    use_xgb: bool = bool(REFERENCE_REVIEW_GUI["use_xgb"])
    use_yolo: bool = bool(REFERENCE_REVIEW_GUI["use_yolo"])
    det_policy: str = str(REFERENCE_REVIEW_GUI["det_policy"])
    det_missing: str = str(REFERENCE_REVIEW_GUI["det_missing"])
    thr_xgb: float = float(REFERENCE_REVIEW_GUI["thr_xgb"])
    thr_yolo: float = float(REFERENCE_REVIEW_GUI["thr_yolo"])
    thr_iou: float = float(REFERENCE_REVIEW_GUI["thr_iou"])
    det_thr: float = float(REFERENCE_REVIEW_GUI["det_thr"])
    hybrid_yolo_bias: float = float(REFERENCE_REVIEW_GUI["hybrid_yolo_bias"])
    al_margin: float = float(REFERENCE_REVIEW_GUI["al_margin"])


# ============================== utils (verbatim) ==============================

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
    import cv2

    for keys in [("x", "y", "w", "h"), ("bbox_x", "bbox_y", "bbox_w", "bbox_h")]:
        if all(k in row for k in keys):
            x = _safe_float(row.get(keys[0]), 0); y = _safe_float(row.get(keys[1]), 0)
            w = _safe_float(row.get(keys[2]), 1); h = _safe_float(row.get(keys[3]), 1)
            return (float(x), float(y), max(1.0, float(w)), max(1.0, float(h)))
    if poly is not None and len(poly) >= 3:
        x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
        return (float(x), float(y), float(w), float(h))
    cx = _safe_float(row.get("cx", row.get("centroid_x", 50)), 50)
    cy = _safe_float(row.get("cy", row.get("centroid_y", 50)), 50)
    return (float(cx - 15), float(cy - 15), 30.0, 30.0)


def _combine_default(x_ok, y_ok, x_has, y_has, pol: str, miss: str) -> bool:
    present = []
    if x_has:
        present.append(bool(x_ok))
    if y_has:
        present.append(bool(y_ok))
    if not present:
        return False
    pol = str(pol).lower()
    miss = str(miss).lower()
    if pol in ("xgb", "xgb_only"):
        return bool(x_ok) if x_has else False
    if pol in ("yolo", "yolo_only"):
        return bool(y_ok) if y_has else False
    if pol == "or":
        return (bool(x_ok) or bool(y_ok)) if miss == "reject" else any(present)
    return (bool(x_ok) and bool(y_ok)) if miss == "reject" else all(present)


def _norm(a: float, b: float):
    s = float(a) + float(b)
    if s <= 0:
        return (0.5, 0.5)
    return (float(a) / s, float(b) / s)


def compute_final_pred_and_ui(row: dict, P: GuiParams = GuiParams()):
    """Returns: pred(int), p_ui(float|None), thr_eff(float) -- GUI decision."""

    policy = str(P.det_policy).lower()
    missing = str(P.det_missing).lower()

    xgb_p = _safe_float(row.get("xgb_p", row.get("xgb_score", row.get("prob", 0.0))), 0.0) if P.use_xgb else 0.0
    y_conf = _safe_float(row.get("yolo_conf", 0.0), 0.0) if P.use_yolo else 0.0
    y_iou = _safe_float(row.get("yolo_iou", 0.0), 0.0) if P.use_yolo else 0.0

    x_has = bool(P.use_xgb)
    y_has = bool(P.use_yolo and (y_conf > 0.0))
    x_ok = bool(x_has and (xgb_p >= float(P.thr_xgb)))
    y_ok = bool(y_has and (y_conf >= float(P.thr_yolo)) and (y_iou >= float(P.thr_iou)))

    if policy == "yolo":
        thr_eff = float(P.thr_yolo)
    elif policy == "hybrid":
        thr_eff = float(P.det_thr if y_ok else P.thr_xgb)
    else:
        thr_eff = float(P.thr_xgb)

    if policy == "hybrid" and not P.use_yolo:
        return (1 if x_ok else 0), float(xgb_p), float(P.thr_xgb)

    if policy == "hybrid" and not y_ok:
        if missing == "reject":
            # The paper rejects an unsupported candidate, but still uses its
            # raw XGBoost score for review priority (p_ui, tau_eff).
            return 0, float(xgb_p), thr_eff
        return (1 if x_ok else 0), float(xgb_p), thr_eff

    if policy == "hybrid":
        p_y = float(y_conf)
        x = float(xgb_p)
        det_thr = float(P.det_thr)
        if (x >= det_thr) and (p_y >= det_thr):
            wy, _ = _norm(p_y, x)
            p_f = wy * p_y + (1.0 - wy) * x
            return 1, p_f, thr_eff
        elif (p_y >= det_thr) and (x < det_thr):
            wy, _ = _norm(p_y, x)
            p_f = wy * p_y + (1.0 - wy) * x
            return (1 if p_f >= det_thr else 0), p_f, thr_eff
        elif (x >= det_thr) and (p_y < det_thr):
            wy_raw, _ = _norm(p_y, x)
            wy = max(float(P.hybrid_yolo_bias), float(wy_raw))
            p_f = wy * p_y + (1.0 - wy) * x
            return (1 if p_f >= det_thr else 0), p_f, thr_eff
        else:
            wy, _ = _norm(p_y, x)
            p_f = wy * p_y + (1.0 - wy) * x
            return 0, p_f, thr_eff

    keep = _combine_default(x_ok, y_ok, x_has, y_has, policy, missing)
    p_ui = float(y_conf) if policy in ("yolo", "yolo_only") else (float(xgb_p) if x_has else None)
    return (1 if keep else 0), p_ui, thr_eff


def is_uncertain(p_ui, thr_eff, P: GuiParams = GuiParams()):
    if p_ui is None:
        return False
    return abs(float(p_ui) - float(thr_eff)) <= float(P.al_margin)


# ============================== mosaic IO (verbatim) ==============================
MISSING_TILE_BGR = tuple(REFERENCE_REVIEW_GUI["missing_tile_bgr"])
JPG_QUALITY = int(REFERENCE_REVIEW_GUI["jpg_quality"])


def load_tile_rgb(p: Path):
    import cv2

    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        return None
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def build_mosaic(base: str, tiles_dir: Path, tiles: list[tuple[str, int, int]]):
    import cv2

    sample = None
    for name, yy, xx in tiles:
        im = load_tile_rgb(tiles_dir / name)
        if im is not None:
            sample = im
            break
    if sample is None:
        raise RuntimeError(f"No readable tiles for base={base} (check TILES_DIR).")
    tile_h, tile_w = sample.shape[:2]
    max_y = max(int(yy) for _, yy, _ in tiles)
    max_x = max(int(xx) for _, _, xx in tiles)
    H = int(max_y + tile_h)
    W = int(max_x + tile_w)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = np.array(MISSING_TILE_BGR, dtype=np.uint8)[None, None, :]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    name_by_pos = {(int(yy), int(xx)): str(name) for (name, yy, xx) in tiles}
    for (yy, xx), name in name_by_pos.items():
        im = load_tile_rgb(tiles_dir / name)
        if im is None:
            continue
        if im.shape[0] != tile_h or im.shape[1] != tile_w:
            im = cv2.resize(im, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        y0, x0 = int(yy), int(xx)
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = im
    return canvas


def write_mosaic(mosaic: np.ndarray, out_path: Path) -> tuple[int, int]:
    import cv2

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
    return W, H


# ============================== seen / labels (verbatim) ==============================

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
    """Effective label = last row per (image,id) after pandas timestamp sort (LB-15)."""

    import pandas as pd

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
        k = key_str(r.get("image", ""), r.get("id", ""))
        hl = r.get("human_label", None)
        if hl is None or (isinstance(hl, float) and np.isnan(hl)):
            continue
        try:
            hl_i = int(hl)
        except Exception:
            continue
        out[k] = {
            "human_label": int(hl_i),
            "action": str(r.get("action", "")),
            "review_weight": _safe_float(r.get("review_weight", np.nan), np.nan),
            "timestamp": int(r.get("timestamp", 0)) if not pd.isna(r.get("timestamp", 0)) else 0,
        }
    return out


def append_review_row(csv_path: Path, rec: dict):
    import pandas as pd

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
    import pandas as pd

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
    import pandas as pd

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


def _shape_xy_in_mosaic(row: dict, off_x: int, off_y: int):
    poly = _parse_poly(row)
    x, y, w, h = _parse_bbox(row, poly)
    if poly is not None and len(poly) >= 3:
        pts = poly.copy()
        pts[:, 0] += off_x
        pts[:, 1] += off_y
        xs = pts[:, 0].tolist() + [pts[0, 0]]
        ys = pts[:, 1].tolist() + [pts[0, 1]]
    else:
        x1 = x + off_x; y1 = y + off_y
        x2 = x1 + w; y2 = y1 + h
        xs = [x1, x2, x2, x1, x1]
        ys = [y1, y1, y2, y2, y1]
    return xs, ys


# ============================== session ==============================
ACTIONS = ("accept", "flip", "sus_accept", "sus_flip", "skip", "auto_accept_rest", "unsee", "undo")


@dataclass
class LegacyReviewSession:
    """State of one legacy review (detections + review_labels.csv + seen.json).

    ``review_csv`` / ``seen_json`` are the files written by the session.  When
    an imported project reopens historical review state, the historical files
    are first copied into the project (originals are never modified).
    """

    detections_csv: Path
    tiles_dir: Path
    review_csv: Path
    seen_json: Path
    cache_dir: Path
    params: GuiParams = field(default_factory=GuiParams)
    clock: Callable[[], int] = now_ts_compact

    def __post_init__(self):
        import pandas as pd

        self.detections_csv = Path(self.detections_csv)
        self.tiles_dir = Path(self.tiles_dir)
        self.review_csv = Path(self.review_csv)
        self.seen_json = Path(self.seen_json)
        self.cache_dir = Path(self.cache_dir)
        if not self.detections_csv.exists():
            raise FileNotFoundError(f"DETECTIONS_CSV not found: {self.detections_csv}")
        if not self.tiles_dir.exists():
            raise FileNotFoundError(f"TILES_DIR not found: {self.tiles_dir}")
        df = pd.read_csv(self.detections_csv)
        if "image" not in df.columns or "id" not in df.columns:
            raise ValueError("detections.csv must contain columns: image, id")
        tile_info = df["image"].astype(str).map(parse_tile)
        df["base"] = tile_info.map(lambda t: t[0] if t else "")
        self.df = df
        self.bases = sorted([b for b in df["base"].unique().tolist() if b])
        if not self.bases:
            raise ValueError("No bases found in image names (expected *_y####x####.*).")
        self.labels_eff = load_labels_effective(self.review_csv)
        self.seen = load_seen(self.seen_json)

    # ---------------- presentation data ----------------
    def mosaic(self, base: str) -> tuple[Path, int, int]:
        import cv2

        tiles: list[tuple[str, int, int]] = []
        for p in self.tiles_dir.glob(f"{base}_y*x*.*"):
            pt = parse_tile(p.name)
            if pt:
                tiles.append((p.name, pt[1], pt[2]))
        if not tiles:
            raise RuntimeError(f"No tiles found for base={base} under {self.tiles_dir}")
        path = self.cache_dir / f"{base}.jpg"
        if path.exists():
            im = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if im is None:
                path.unlink(missing_ok=True)
                raise RuntimeError("Cached mosaic unreadable; delete and rebuild.")
            H, W = im.shape[:2]
            return path, int(W), int(H)
        W, H = write_mosaic(build_mosaic(base, self.tiles_dir, tiles), path)
        return path, int(W), int(H)

    def items(self, base: str) -> list[dict[str, Any]]:
        out = []
        dfb = self.df[self.df["base"].astype(str) == str(base)]
        for _, rr in dfb.iterrows():
            row = rr.to_dict()
            tile = str(row["image"])
            pt = parse_tile(tile)
            if not pt:
                continue
            _, oy, ox, _ = pt
            det_id = row.get("id", "?")
            k = key_str(tile, det_id)
            pred, p_ui, thr_eff = compute_final_pred_and_ui(row, self.params)
            unc = is_uncertain(p_ui, thr_eff, self.params)
            reviewed = k in self.labels_eff
            human = int(self.labels_eff[k]["human_label"]) if reviewed else None
            mismatch = bool(reviewed and human in (0, 1) and int(human) != int(pred))
            xs, ys = _shape_xy_in_mosaic(row, int(ox), int(oy))
            poly_src = "poly" if _parse_poly(row) is not None else "bbox_rectangle"
            out.append({
                "key": k, "image": tile, "id": str(det_id), "pred": int(pred),
                "p_ui": None if p_ui is None else float(p_ui), "thr_eff": float(thr_eff),
                "uncertain": bool(unc), "reviewed": bool(reviewed), "human": human,
                "action": self.labels_eff[k]["action"] if reviewed else None,
                "seen": k in self.seen, "mismatch": mismatch,
                "xs": [float(v) for v in xs], "ys": [float(v) for v in ys], "geometry": poly_src,
                "xgb_p": _safe_float(row.get("xgb_p", row.get("xgb_score", 0.0)), 0.0),
                "yolo_conf": _safe_float(row.get("yolo_conf", 0.0), 0.0),
                "yolo_iou": _safe_float(row.get("yolo_iou", 0.0), 0.0),
                "kept_pipeline": int(_safe_float(row.get("kept", -1), -1)),
                "p_fused": None if str(row.get("p_fused", "")) in ("", "nan") else _safe_float(row.get("p_fused"), 0.0),
            })
        return out

    # ---------------- actions (on_action port) ----------------
    def _row_for(self, sel: str) -> dict | None:
        tile, det_id = str(sel).split("||", 1)
        df = self.df
        m = df[(df["image"].astype(str) == str(tile)) & (df["id"].astype(str) == str(det_id))]
        if len(m) == 0:
            return None
        return m.iloc[0].to_dict()

    def _record(self, tile, det_id, human, action, row) -> dict:
        P = self.params
        pred, p_ui, thr_eff = compute_final_pred_and_ui(row, P)
        xgb_p = _safe_float(row.get("xgb_p", row.get("xgb_score", 0.0)), 0.0)
        yconf = _safe_float(row.get("yolo_conf", 0.0), 0.0)
        yiou = _safe_float(row.get("yolo_iou", 0.0), 0.0)
        kept = int(_safe_float(row.get("kept", -1), -1))
        poly = _parse_poly(row)
        bx, by, bw, bh = _parse_bbox(row, poly)
        cx_tile = int(round(bx + bw / 2.0))
        cy_tile = int(round(by + bh / 2.0))
        weight = float(WEIGHT_MAP.get(str(action), 1.0)) if action not in ("skip", "auto_accept") \
            else float(WEIGHT_MAP.get(action, 0.0))
        return {
            "image": str(tile), "id": str(det_id), "human_label": human, "action": action,
            "review_weight": weight, "click_x": int(cx_tile), "click_y": int(cy_tile),
            "prob": (None if p_ui is None else float(p_ui)), "xgb_prob": float(xgb_p),
            "final_pred": int(pred), "xgb_pred": int(float(xgb_p) >= float(P.thr_xgb)),
            "kept": int(kept), "yolo_conf": float(yconf), "yolo_iou": float(yiou),
            "det_policy": str(P.det_policy), "det_missing": str(P.det_missing),
            "thr_xgb": float(P.thr_xgb), "thr_yolo": float(P.thr_yolo), "thr_iou": float(P.thr_iou),
            "det_thr": float(P.det_thr), "hybrid_yolo_bias": float(P.hybrid_yolo_bias),
            "use_xgb": bool(P.use_xgb), "use_yolo": bool(P.use_yolo), "timestamp": self.clock(),
        }

    def act(self, action: str, sel: str | None = None, base: str | None = None) -> str:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        if action == "undo":
            last = pop_last_row(self.review_csv)
            if last is None:
                msg = "[UNDO] nothing to undo."
            else:
                act_last = str(last.get("action", "")).lower().strip()
                try:
                    k_undo = key_str(str(last.get("image", "")), str(last.get("id", "")))
                except Exception:
                    k_undo = None
                if act_last == "skip" and k_undo and (k_undo in self.seen):
                    self.seen.remove(k_undo)
                    save_seen(self.seen_json, self.seen)
                    msg = f"[UNDO] removed last row (action=skip) and un-saw: {k_undo}"
                else:
                    msg = f"[UNDO] removed last row (action={act_last or 'unknown'})."
            self.labels_eff = load_labels_effective(self.review_csv)
            return msg
        if action == "auto_accept_rest":
            base_now = str(base or "").strip()
            if not base_now:
                return "[BULK] No base selected."
            dfb = self.df[self.df["base"].astype(str) == base_now].copy()
            if len(dfb) == 0:
                return f"[BULK] base={base_now}: no rows."
            recs = []
            n_reviewed = n_seen = 0
            for _, rr in dfb.iterrows():
                row = rr.to_dict()
                tile = str(row.get("image", ""))
                det_id = str(row.get("id", ""))
                k = key_str(tile, det_id)
                if k in self.labels_eff:
                    n_reviewed += 1
                    continue
                if k in self.seen:
                    n_seen += 1
                    continue
                pred, _, _ = compute_final_pred_and_ui(row, self.params)
                recs.append(self._record(tile, det_id, int(pred), "auto_accept", row))
            n_app = append_review_rows(self.review_csv, recs)
            if n_app > 0:
                self.labels_eff = load_labels_effective(self.review_csv)
            return (f"[BULK] base={base_now}: auto-accepted {n_app} remaining (action=auto_accept; excluded from effort). "
                    f"(already reviewed={n_reviewed}, seen/skip={n_seen}). Note: Undo removes only the last CSV row.")
        if not sel:
            return "No selection. Click a dot first."
        if action == "unsee":
            if str(sel) in self.seen:
                self.seen.remove(str(sel))
                save_seen(self.seen_json, self.seen)
                return f"[SEEN] removed {sel}."
            return f"[SEEN] {sel} was not in seen."
        try:
            tile, det_id = str(sel).split("||", 1)
        except Exception:
            return f"[ERR] bad selected key: {sel}"
        row = self._row_for(sel)
        if row is None:
            return f"[ERR] cannot find row in detections for {sel}"
        pred, _, _ = compute_final_pred_and_ui(row, self.params)
        if action == "skip":
            rec = self._record(tile, det_id, np.nan, "skip", row)
            append_review_row(self.review_csv, rec)
            self.seen.add(str(sel))
            save_seen(self.seen_json, self.seen)
            return f"[SKIP] logged + marked {sel} as seen (appended to review_labels.csv, weight=0.0)."
        human = int(pred) if action in ("accept", "sus_accept") else int(1 - int(pred))
        rec = self._record(tile, det_id, int(human), action, row)
        append_review_row(self.review_csv, rec)
        self.labels_eff = load_labels_effective(self.review_csv)
        w = float(WEIGHT_MAP.get(str(action), 1.0))
        return f"[SAVE] {action} -> human={human} w={w:.2f} for {sel}  (appended to review_labels.csv)"

    def mosaic_data_uri(self, base: str) -> str:
        path, _, _ = self.mosaic(base)
        return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
