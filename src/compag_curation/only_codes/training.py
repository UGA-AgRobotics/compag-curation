"""Port of notebook cells 6 (Cell3A) and 8 (Cell3B): data preparation,
augmentation, group-safe weighted search / early-stopping refit / full-train
refit, threshold selection and artifact writing.

The statements follow the notebook order so that every NumPy / scikit-learn /
imbalanced-learn / XGBoost random stream is consumed identically.  Differences
are plumbing only:

* paths are explicit arguments (the notebook hard-coded ``CSV_PATH`` and read
  ``CJ_*`` environment variables); legacy environment variables are ignored;
* the effective device is explicit (the notebook enabled CUDA when ``import
  cupy`` succeeded) and is recorded;
* round directories are written into the project overlay; previous-round state
  (``tile_preds.csv``, ``threshold_*.json``, ``best_params_used.json``) is read
  through the copy-on-write view, preserving the notebook's directory scan;
* a new model receives a new artifact identity (``ONLY_CODES_MODEL_IDENTITY.json``)
  plus a pickle-free bundle; the legacy ``.pkl`` is still written because it is
  part of the legacy output contract.

Legacy limitations are preserved on purpose (LB-09..LB-12 in contract.py).
"""

from __future__ import annotations

import inspect
import json
import os
import re
import time
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .contract import REFERENCE_TRAINING, contract_sha256
from .state import LegacyState


class Timer:
    def __init__(self, label, log):
        self.label = label
        self.log = log

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.log(f"[TIMER] {self.label} ...")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.log(f"[TIMER] {self.label} took {time.perf_counter() - self.t0:.2f} sec")


# ------------------------------------------------------------------ Cell3A helpers
def normalize_features_csv_add_review_tag(csv_in: Path, csv_out: Path) -> Path:
    import csv as _csv

    with csv_in.open("r", newline="", encoding="utf-8", errors="replace") as fin:
        rdr = _csv.reader(fin)
        header = next(rdr, None)
        if not header:
            raise RuntimeError("CSV is empty or missing header.")
        header = [h.strip() for h in header]
        has_review_tag = ("review_tag" in header)
        if has_review_tag:
            tag_idx = header.index("review_tag")
            out_header = header
            expected_n = len(out_header)
        else:
            if "reviewed" in header:
                tag_idx = header.index("reviewed") + 1
            elif "class_id" in header:
                tag_idx = header.index("class_id")
            else:
                tag_idx = len(header)
            out_header = header[:tag_idx] + ["review_tag"] + header[tag_idx:]
            expected_n = len(out_header)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="", encoding="utf-8") as fout:
            wtr = _csv.writer(fout)
            wtr.writerow(out_header)
            in_n = len(header)
            for row in rdr:
                if not row:
                    continue
                if not has_review_tag:
                    if len(row) == in_n:
                        row = row[:tag_idx] + [""] + row[tag_idx:]
                    elif len(row) == in_n + 1:
                        pass
                    elif len(row) < in_n:
                        row = row + [""] * (in_n - len(row))
                        row = row[:tag_idx] + [""] + row[tag_idx:]
                    else:
                        row = row[:(in_n)] + [",".join(row[in_n:])]
                        row = row[:tag_idx] + [""] + row[tag_idx:]
                else:
                    if len(row) < expected_n:
                        row = row + [""] * (expected_n - len(row))
                    elif len(row) > expected_n:
                        row = row[:(expected_n - 1)] + [",".join(row[expected_n - 1:])]
                if len(row) < expected_n:
                    row = row + [""] * (expected_n - len(row))
                elif len(row) > expected_n:
                    row = row[:(expected_n - 1)] + [",".join(row[expected_n - 1:])]
                wtr.writerow(row)
    return csv_out


def read_features_csv_robust(csv_path: Path, normalized_out: Path):
    import pandas as pd
    from pandas.errors import ParserError

    try:
        df = pd.read_csv(csv_path, low_memory=False, keep_default_na=False)
        used = csv_path
    except ParserError:
        norm = normalize_features_csv_add_review_tag(csv_path, normalized_out)
        df = pd.read_csv(norm, low_memory=False, keep_default_na=False)
        used = norm
    if "review_tag" not in df.columns:
        df["review_tag"] = ""
    df["review_tag"] = df["review_tag"].astype(str).str.strip()
    df.loc[df["review_tag"].str.lower().isin({"nan", "none"}), "review_tag"] = ""
    if "reviewed" in df.columns:
        df["reviewed"] = pd.to_numeric(df["reviewed"], errors="coerce").fillna(0).astype(int)
    else:
        df["reviewed"] = 0
    if "label" not in df.columns:
        raise RuntimeError("CSV missing required column: label")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    if "scale" in df.columns:
        df["scale"] = pd.to_numeric(df["scale"], errors="coerce")
    return df, used


def leakage_scan_vectorized(df, y, drop_always: set, leak_patterns: List[str], corr_thr: float = 0.999):
    import pandas as pd

    num_df = df.drop(columns=list(drop_always), errors="ignore")
    num_df = num_df.select_dtypes(include=[np.number]).astype(np.float32)
    pat = "|".join(f"(?:{p})" for p in leak_patterns)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression", category=UserWarning)
        name_hit = num_df.columns.to_series().str.contains(pat, case=False, regex=True, na=False)
    drop_by_name = set(num_df.columns[name_hit])
    num_df_f = num_df.fillna(0.0)
    y_s = pd.Series(y.astype(np.float32), index=num_df_f.index)
    eq_y = num_df_f.eq(y_s, axis=0).all(axis=0)
    eq_inv = num_df_f.eq(1.0 - y_s, axis=0).all(axis=0)
    drop_by_equal = set(num_df_f.columns[eq_y | eq_inv])
    const_mask = (num_df_f.nunique(dropna=False) <= 1)
    corr = num_df_f.loc[:, ~const_mask].corrwith(y_s, method="pearson")
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    drop_by_corr = set(corr.index[corr.abs() > float(corr_thr)])
    to_drop = drop_by_name | drop_by_equal | drop_by_corr
    keep_cols = [c for c in num_df.columns if c not in to_drop and not bool(const_mask.get(c, False))]
    return keep_cols, sorted(to_drop)


def _ensure_numeric_series(s, default=0.0):
    import pandas as pd

    return pd.to_numeric(s, errors="coerce").fillna(default).astype(float)


def border_and_tiny_masks_from_df_auto(df, name_col: str, tile_size_default: int = 512, border_px: int = 4,
                                       tiny_min_wh: int = 32, tiny_max_area: int = 1500):
    import pandas as pd

    bx = _ensure_numeric_series(df.get("bbox_x", 0))
    by = _ensure_numeric_series(df.get("bbox_y", 0))
    bw = _ensure_numeric_series(df.get("bbox_w", 0))
    bh = _ensure_numeric_series(df.get("bbox_h", 0))
    if {"tile_w", "tile_h"}.issubset(df.columns):
        tw = _ensure_numeric_series(df["tile_w"]).clip(lower=1.0)
        th = _ensure_numeric_series(df["tile_h"]).clip(lower=1.0)
    else:
        per_tile = pd.DataFrame({
            name_col: df[name_col].astype(str),
            "mx_w": (bx + bw),
            "mx_h": (by + bh),
        }).groupby(name_col, sort=False).agg({"mx_w": "max", "mx_h": "max"})
        tw = df[name_col].astype(str).map(per_tile["mx_w"]).fillna(float(tile_size_default)).clip(lower=1.0)
        th = df[name_col].astype(str).map(per_tile["mx_h"]).fillna(float(tile_size_default)).clip(lower=1.0)
    border_frac = float(border_px) / float(tile_size_default)
    tiny_min_frac = float(tiny_min_wh) / float(tile_size_default)
    tiny_area_frac = float(tiny_max_area) / float(tile_size_default * tile_size_default)
    is_border = ((bx <= border_frac * tw) | (by <= border_frac * th) |
                 ((bx + bw) >= (1.0 - border_frac) * tw) | ((by + bh) >= (1.0 - border_frac) * th)).to_numpy()
    is_tiny = ((bw < tiny_min_frac * tw) | (bh < tiny_min_frac * th) |
               ((bw * bh) < (tiny_area_frac * (tw * th)))).to_numpy()
    return is_border, is_tiny


def pick_name_col(df) -> str:
    for c in ["image", "file_name", "filename", "file"]:
        if c in df.columns:
            return c
    raise ValueError("CSV must contain one of: 'image', 'file_name', 'filename', 'file'.")


def original_image_group(name: str) -> str:
    base = Path(str(name)).name
    stem = Path(base).stem
    for pat in [r"^(?P<root>.+?)_y\d{1,8}x\d{1,8}$", r"^(?P<root>.+?)_x\d{1,8}_y\d{1,8}$",
                r"^(?P<root>.+?)(?:__tile-\d+|-tile-\d+)$"]:
        m = re.match(pat, stem, flags=re.IGNORECASE)
        if m:
            return m.group("root")
    return stem


def pick_groups_tile_balanced(groups_vec: np.ndarray, test_frac: float = 0.20, random_state: int = 42,
                              tol: float = 0.01, max_tries: int = 2000):
    uniq, inv = np.unique(groups_vec, return_inverse=True)
    counts = np.bincount(inv).astype(int)
    total = int(counts.sum())
    target = int(round(total * test_frac))
    rs = np.random.RandomState(random_state)
    best_sel, best_sum, best_diff = None, 0, float("inf")

    def greedy_from(order_idx):
        s = 0; sel = set()
        for idx in order_idx:
            c = int(counts[idx])
            if abs((s + c) - target) <= abs(s - target):
                sel.add(uniq[idx]); s += c
        return sel, s

    orders = [np.argsort(-counts), np.argsort(counts)]
    for t in range(max_tries):
        order = orders[t] if t < len(orders) else rs.permutation(len(uniq))
        sel, s = greedy_from(order)
        diff = abs(s - target)
        if diff < best_diff:
            best_sel, best_sum, best_diff = sel, s, diff
            if diff <= max(1, int(target * tol)):
                break
    test_groups = set(best_sel) if best_sel is not None else set()
    train_groups = set(uniq) - test_groups
    return train_groups, test_groups, dict(zip(uniq, counts)), target, best_sum


def _extract_round_num_from_name(s: str) -> Optional[int]:
    m = re.search(r"r(\d+)(?:\D|$)", str(s), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _iter_round_dirs(state: LegacyState, model_root_rel: str) -> List[Tuple[str, Path]]:
    """Directory scan equivalent to ``root.iterdir()`` over original + overlay.

    The original directory order (os.scandir order) is kept first; directories
    that exist only in the overlay are appended.  Each entry resolves reads
    through the copy-on-write view.
    """

    out: List[Tuple[str, Path]] = []
    seen = set()
    for base in (state.original(model_root_rel), state.overlay(model_root_rel)):
        if base is None or not base.exists():
            continue
        for d in base.iterdir():
            if d.name in seen:
                continue
            seen.add(d.name)
            out.append((d.name, d))
    return out


def _glob_in_view(state: LegacyState, rel_dir: str, pattern: str) -> List[Path]:
    """``list(dir.glob(pattern))`` order-preserving over the copy-on-write view."""

    res: List[Path] = []
    names = set()
    for base in (state.original(rel_dir), state.overlay(rel_dir)):
        if base is None or not base.exists():
            continue
        for p in base.glob(pattern):
            if p.name in names:
                continue
            names.add(p.name)
            res.append(state.path(f"{rel_dir}/{p.name}"))
    return res


def _find_prev_round_dir(state: LegacyState, model_root_rel: str, current_round_tag: str,
                         round_dir_name: str) -> Optional[str]:
    cur = _extract_round_num_from_name(current_round_tag)
    if cur is None:
        return None
    best = None
    best_r = -1
    for name, d in _iter_round_dirs(state, model_root_rel):
        rel = f"{model_root_rel}/{name}"
        if not state.path(rel).is_dir():
            continue
        if name == round_dir_name:
            continue
        tp = state.path(f"{rel}/tile_preds.csv")
        ths = _glob_in_view(state, rel, "threshold_*.json")
        if not tp.exists() or not ths:
            continue
        rnums = [_extract_round_num_from_name(x.name) for x in ths] + [_extract_round_num_from_name(name)]
        rnums = [r for r in rnums if r is not None]
        if not rnums:
            continue
        r = max(rnums)
        if r < cur and r > best_r:
            best_r = r
            best = rel
    return best


def _load_prev_preds_and_thr(prev_preds_path: Path, prev_thr_path: Path, name_col: str):
    import pandas as pd

    try:
        prev = pd.read_csv(prev_preds_path, low_memory=False, keep_default_na=False)
        with open(prev_thr_path, "r", encoding="utf-8") as f:
            thr_obj = json.load(f)
        thr = float(thr_obj.get("threshold", thr_obj.get("thr", 0.5)))
        if name_col not in prev.columns or "proba" not in prev.columns:
            return None
        prev2 = prev[[name_col, "proba"]].copy()
        prev2["proba"] = pd.to_numeric(prev2["proba"], errors="coerce")
        prev2 = prev2.groupby(name_col, as_index=False, sort=False)["proba"].max()
        prev2["pred"] = (prev2["proba"] >= thr).astype(int)
        mapping = {str(n): (int(p), float(s)) for n, p, s in zip(prev2[name_col], prev2["pred"], prev2["proba"])}
        mapping["__thr__"] = (0, float(thr))
        return mapping
    except Exception:
        return None


def mixup_same_class_with_groups(X, y, w, g, alpha=0.2, mult=0.5, exclude_discrete_max_nunique=10, rs=None):
    import pandas as pd

    if rs is None:
        rs = np.random.RandomState(42)
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=np.float32); g = np.asarray(g)
    nunq = pd.DataFrame(X).nunique().to_numpy()
    cont_idx = np.where(nunq > int(exclude_discrete_max_nunique))[0]
    if cont_idx.size == 0:
        return X, y, w, g
    Xc_list, yc_list, wc_list, gc_list = [X], [y], [w], [g]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        n = len(idx)
        if n < 2:
            continue
        m = int(np.ceil(n * float(mult)))
        i1 = rs.choice(idx, size=m, replace=(n < m))
        i2 = rs.choice(idx, size=m, replace=(n < m))
        lam = rs.beta(alpha, alpha, size=m).astype(np.float32)
        Xn = X[i1].copy()
        diff = (X[i2] - X[i1]).astype(np.float32)
        Xn[:, cont_idx] = (X[i1][:, cont_idx] + (lam[:, None] * diff[:, cont_idx])).astype(np.float32)
        yn = np.full(m, cls, dtype=int)
        wn = 0.5 * (w[i1] + w[i2])
        gn = np.where(rs.rand(m) < 0.5, g[i1], g[i2])
        Xc_list.append(Xn); yc_list.append(yn); wc_list.append(wn); gc_list.append(gn)
    return (np.vstack(Xc_list).astype(np.float32), np.concatenate(yc_list).astype(int),
            np.concatenate(wc_list).astype(np.float32), np.concatenate(gc_list))


def dropout_augment_tabular_with_groups(X, y, w, g, p=0.05, mult=1.0, strategy="median_by_class",
                                        exclude_discrete_max_nunique=10, rs=None):
    import pandas as pd

    if rs is None:
        rs = np.random.RandomState(42)
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=np.float32); g = np.asarray(g)
    n, d = X.shape
    nunq = pd.DataFrame(X).nunique().to_numpy()
    cont_idx = np.where(nunq > int(exclude_discrete_max_nunique))[0]
    if cont_idx.size == 0:
        return X, y, w, g
    m = int(np.ceil(n * float(mult)))
    sel = rs.choice(np.arange(n), size=m, replace=(n < m))
    Xn = X[sel].copy(); yn = y[sel].copy(); wn = w[sel].copy(); gn = g[sel].copy()
    if strategy.startswith("median"):
        if "class" in strategy and len(np.unique(y)) >= 2:
            med = {int(c): np.median(X[y == c][:, cont_idx], axis=0) for c in np.unique(y)}
            fills = np.vstack([med[int(c)] for c in yn])
        else:
            med_all = np.median(X[:, cont_idx], axis=0)
            fills = np.repeat(med_all[None, :], m, axis=0)
    else:
        fills = np.zeros((m, cont_idx.size), dtype=np.float32)
    mask = (rs.rand(m, cont_idx.size) < float(p))
    Xn[:, cont_idx] = np.where(mask, fills, Xn[:, cont_idx])
    return (np.vstack([X, Xn]).astype(np.float32), np.concatenate([y, yn]).astype(int),
            np.concatenate([w, wn]).astype(np.float32), np.concatenate([g, gn]))


def jitter_augment_tabular_with_groups(X, y, w, g, cont_idx, mult=0.5, sigma=0.01, per_feature=True,
                                       clip_low=None, clip_high=None, rs=None):
    if rs is None:
        rs = np.random.RandomState(42)
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=np.float32); g = np.asarray(g)
    n, d = X.shape
    if cont_idx.size == 0:
        return X, y, w, g
    m = int(np.ceil(n * float(mult)))
    sel = rs.choice(np.arange(n), size=m, replace=(n < m))
    Xn = X[sel].copy(); yn = y[sel].copy(); wn = w[sel].copy(); gn = g[sel].copy()
    if per_feature:
        std = X[:, cont_idx].std(axis=0, ddof=0) + 1e-6
        noise = rs.normal(0.0, 1.0, size=(m, cont_idx.size)).astype(np.float32)
        Xn[:, cont_idx] = Xn[:, cont_idx] + noise * (float(sigma) * std)
    else:
        gstd = float(np.std(X[:, cont_idx])) + 1e-6
        noise = rs.normal(0.0, 1.0, size=(m, cont_idx.size)).astype(np.float32)
        Xn[:, cont_idx] = Xn[:, cont_idx] + noise * (float(sigma) * gstd)
    if clip_low is not None and clip_high is not None:
        Xn[:, cont_idx] = np.minimum(np.maximum(Xn[:, cont_idx], clip_low), clip_high)
    return (np.vstack([X, Xn]).astype(np.float32), np.concatenate([y, yn]).astype(int),
            np.concatenate([w, wn]).astype(np.float32), np.concatenate([g, gn]))


def apply_augs_in_3A(X_df, y_np, w_np, g_np, cfg, random_state=42, log=print):
    import pandas as pd
    from sklearn.impute import SimpleImputer

    rs = np.random.RandomState(random_state)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_df)
    nunq = X_df.nunique().to_numpy()
    cont_idx = np.where(nunq > int(cfg["exclude_discrete_max_nunique"]))[0]
    ql, qh = None, None
    if cfg["jitter"]["enabled"] and cont_idx.size > 0:
        ql = np.quantile(X_imp[:, cont_idx], cfg["jitter"]["clip_q"][0], axis=0)
        qh = np.quantile(X_imp[:, cont_idx], cfg["jitter"]["clip_q"][1], axis=0)
    Xo, yo, wo, go = X_imp, y_np.copy(), w_np.copy(), g_np.copy()
    if cfg["mixup"]["enabled"]:
        Xo, yo, wo, go = mixup_same_class_with_groups(Xo, yo, wo, go, alpha=cfg["mixup"]["alpha"],
                                                      mult=cfg["mixup"]["mult"],
                                                      exclude_discrete_max_nunique=cfg["exclude_discrete_max_nunique"], rs=rs)
        log(f"[AUG] mixup -> total {len(yo)} samples")
    if cfg["dropout"]["enabled"]:
        Xo, yo, wo, go = dropout_augment_tabular_with_groups(Xo, yo, wo, go, p=cfg["dropout"]["p"],
                                                             mult=cfg["dropout"]["mult"],
                                                             strategy=cfg["dropout"]["strategy"],
                                                             exclude_discrete_max_nunique=cfg["exclude_discrete_max_nunique"], rs=rs)
        log(f"[AUG] dropout -> total {len(yo)} samples")
    if cfg["jitter"]["enabled"]:
        Xo, yo, wo, go = jitter_augment_tabular_with_groups(Xo, yo, wo, go, cont_idx=cont_idx,
                                                            mult=cfg["jitter"]["mult"], sigma=cfg["jitter"]["sigma"],
                                                            per_feature=cfg["jitter"]["per_feature"],
                                                            clip_low=ql, clip_high=qh, rs=rs)
        log(f"[AUG] jitter -> total {len(yo)} samples")
    return pd.DataFrame(Xo, columns=X_df.columns), yo, wo, go


# ------------------------------------------------------------------ Cell3A driver
def prepare_training_data(*, df, csv_used: str, name_col: str | None = None, state: LegacyState,
                          model_root_rel: str, round_tag: str, round_dir_name: str,
                          prev_tile_preds_path: Path | None = None, prev_thresh_json_path: Path | None = None,
                          cfg: Dict[str, Any] | None = None, log=print) -> Dict[str, Any]:
    """Everything Cell3A computes, returned as a dict (the notebook's globals)."""

    import pandas as pd

    C = dict(REFERENCE_TRAINING if cfg is None else cfg)
    TEST_SIZE = C["test_size"]; RANDOM_STATE = C["random_state"]
    BORDER_PX = C["border_px"]; TILE_SIZE = C["tile_size"]; TINY_MIN_WH = C["tiny_min_wh"]
    TINY_MAX_AREA = C["tiny_max_area"]; BORDER_WEIGHT = C["border_weight"]
    UNCERT_MARGIN = C["uncert_margin"]; UNCERT_MULT = C["uncert_mult"]
    DROP_TINY_IN_TRAIN = C["drop_tiny_in_train"]; DROP_BORDER_FROM_TEST = C["drop_border_from_test"]
    USE_REVIEW_WEIGHTING = C["use_review_weighting"]; WEIGHT_CORRECT = C["weight_correct"]
    WEIGHT_WRONG = C["weight_wrong"]; AUTO_DETECT_PREV_ROUND = C["auto_detect_prev_round"]
    WEIGHT_MAP = dict(C["weight_map"]); USE_REVIEW_TAG_WEIGHTS = C["use_review_tag_weights"]
    DROP_SKIP_ROWS = C["drop_skip_rows"]; AUG_CFG = json.loads(json.dumps(C["aug"]))
    AUG_CFG["jitter"]["clip_q"] = tuple(AUG_CFG["jitter"]["clip_q"])
    TINY_NEG_WEIGHT = C["tiny_neg_weight"]

    USE_SCALE_FEATURE = False
    if "scale" in df.columns:
        df["scale"] = pd.to_numeric(df["scale"], errors="coerce")
    NAME_COL = name_col or pick_name_col(df)
    groups = df[NAME_COL].astype(str).map(original_image_group).values
    y = df["label"].astype(int).values
    drop_always = {NAME_COL, "image", "file_name", "filename", "file", "id", "ann_id", "label", "group",
                   "area_px", "perim_sqrt", "reviewed", "review_tag", "class_id", "class_name"}
    if not USE_SCALE_FEATURE:
        drop_always.add("scale")
    LEAK_PATTERNS = [r"^class(_?id)?$", r"^category(_?id)?$", r"^cat(_?id)?$", r"^cid$", r"^name$",
                     r"^target$", r"^y$", r".*(_|^)class(_|$).*", r".*(_|^)category(_|$).*",
                     r".*(_|^)catname(_|$).*", r".*(_|^)classname(_|$).*"]
    num_cols, dropped_cols = leakage_scan_vectorized(df, y, drop_always, LEAK_PATTERNS, corr_thr=0.999)
    MUST_KEEP = ["pred_iou", "stability", "embed_sim", "g_quality", "g_light", "g_color", "g_shape",
                 "g_embed", "g_robust", "g_maha", "g_border"]
    PCA_KEEP = [c for c in df.columns if c.startswith("embed_pca_")]

    def _safe_promote(cols, df, y):
        ys = pd.Series(pd.to_numeric(y, errors="coerce"), index=df.index).astype(float).fillna(0.0)
        keep = []
        for c in cols:
            if c not in df.columns:
                continue
            s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            if s.nunique(dropna=False) <= 1:
                continue
            if s.equals(ys) or s.equals(1.0 - ys):
                continue
            r = np.corrcoef(s.to_numpy(), ys.to_numpy())[0, 1]
            if np.isfinite(r) and abs(r) > 0.999:
                continue
            keep.append(c)
        return keep

    keepers = _safe_promote(MUST_KEEP + PCA_KEEP, df, y)
    num_cols = sorted(set(num_cols).union(keepers))
    log(f"[FEATURES] final numeric columns: {len(num_cols)}")
    if not num_cols:
        raise RuntimeError("No usable numeric features after leakage/variance filters.")
    X = df[num_cols].copy().astype(np.float32)

    g_tr_set, g_te_set, g_counts, target_tiles_test, got_tiles_test = pick_groups_tile_balanced(
        groups_vec=groups, test_frac=float(TEST_SIZE), random_state=RANDOM_STATE, tol=0.01, max_tries=3000)
    train_idx = np.where(np.isin(groups, list(g_tr_set)))[0]
    test_idx = np.where(np.isin(groups, list(g_te_set)))[0]
    if AUG_CFG.get("shuffle_rows", True):
        rs = np.random.RandomState(RANDOM_STATE)
        train_idx = rs.permutation(train_idx)
        test_idx = rs.permutation(test_idx)
    split_record = {"test_groups": sorted(map(str, g_te_set)), "train_groups": sorted(map(str, g_tr_set)),
                    "target_tiles_test": int(target_tiles_test), "got_tiles_test": int(got_tiles_test),
                    "train_idx_initial": train_idx.copy(), "test_idx_initial": test_idx.copy()}
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]
    _is_b_tr, _is_t_tr = border_and_tiny_masks_from_df_auto(df.iloc[train_idx], name_col=NAME_COL,
                                                            tile_size_default=TILE_SIZE, border_px=BORDER_PX,
                                                            tiny_min_wh=TINY_MIN_WH, tiny_max_area=TINY_MAX_AREA)
    _is_b_te, _ = border_and_tiny_masks_from_df_auto(df.iloc[test_idx], name_col=NAME_COL,
                                                     tile_size_default=TILE_SIZE, border_px=BORDER_PX,
                                                     tiny_min_wh=TINY_MIN_WH, tiny_max_area=TINY_MAX_AREA)
    if DROP_TINY_IN_TRAIN and _is_t_tr.any():
        _drop_tr = (_is_t_tr & (y_train == 0))
        _keep_tr = ~_drop_tr
        X_train = X_train.iloc[_keep_tr]; y_train = y_train[_keep_tr]
        groups_train = groups_train[_keep_tr]; train_idx = train_idx[_keep_tr]
        _is_b_tr = _is_b_tr[_keep_tr]; _is_t_tr = _is_t_tr[_keep_tr]
    n_border_test = int(_is_b_te.sum())
    if DROP_BORDER_FROM_TEST and _is_b_te.any():
        _keep_te = ~_is_b_te
        X_test = X_test.iloc[_keep_te]; y_test = y_test[_keep_te]
        groups_test = groups_test[_keep_te]; test_idx = test_idx[_keep_te]
    log(f"Train size: {len(train_idx)}, Test size: {len(test_idx)} | drop border in TEST: {n_border_test}")

    # ---------------- prev-round (review) weights ----------------
    prev_record: Dict[str, Any] = {"auto_detect": AUTO_DETECT_PREV_ROUND}

    def build_review_weights_for_train(train_idx, y_train):
        prev_preds_path = Path(prev_tile_preds_path) if prev_tile_preds_path else None
        prev_thr_path = Path(prev_thresh_json_path) if prev_thresh_json_path else None
        if AUTO_DETECT_PREV_ROUND and (prev_preds_path is None or prev_thr_path is None):
            prev_rel = _find_prev_round_dir(state, model_root_rel, round_tag, round_dir_name)
            prev_record["prev_dir"] = prev_rel
            if prev_rel is not None:
                prev_preds_path = state.path(f"{prev_rel}/tile_preds.csv")
                cands = _glob_in_view(state, prev_rel, "threshold_*.json")
                prev_thr_path = cands[0] if cands else None
        prev_record["tile_preds"] = None if prev_preds_path is None else str(prev_preds_path)
        prev_record["threshold_json"] = None if prev_thr_path is None else str(prev_thr_path)
        mapping = None
        if prev_preds_path is not None and prev_thr_path is not None and prev_preds_path.exists() and prev_thr_path.exists():
            mapping = _load_prev_preds_and_thr(prev_preds_path, prev_thr_path, NAME_COL)
        w = np.full(len(train_idx), float(WEIGHT_WRONG), dtype=np.float32)
        if not mapping:
            return w
        thr = float(mapping.get("__thr__", (0, 0.5))[1])
        prev_record["prev_threshold"] = thr
        names_train = df.iloc[train_idx][NAME_COL].astype(str).values
        preds_prev = np.full(len(train_idx), -1, dtype=int)
        prob_prev = np.full(len(train_idx), np.nan, dtype=float)
        for i, n in enumerate(names_train):
            v = mapping.get(str(n), None)
            if v is None:
                continue
            preds_prev[i] = int(v[0])
            prob_prev[i] = float(v[1])
        known = preds_prev >= 0
        if known.any():
            correct = (preds_prev[known] == y_train[known])
            w_sub = np.where(correct, float(WEIGHT_CORRECT), float(WEIGHT_WRONG)).astype(np.float32)
            close = (np.abs(prob_prev[known] - thr) <= float(UNCERT_MARGIN))
            w_sub = w_sub * np.where(close, float(UNCERT_MULT), 1.0).astype(np.float32)
            w[known] = w_sub
        prev_record["known_rows"] = int(known.sum())
        return w

    review_weights_train = np.ones_like(y_train, dtype=np.float32)
    review_mask_train = np.zeros_like(y_train, dtype=bool)
    if USE_REVIEW_WEIGHTING:
        review_weights_train = build_review_weights_for_train(train_idx, y_train)
        review_mask_train = (df.iloc[train_idx]["reviewed"].astype(int).values == 1)
        review_weights_train = np.where(review_mask_train, review_weights_train, 1.0).astype(np.float32)
    base_review_weights = review_weights_train.copy()
    if _is_b_tr.any() and review_mask_train.any():
        review_weights_train = review_weights_train * np.where(review_mask_train & _is_b_tr, float(BORDER_WEIGHT), 1.0).astype(np.float32)
    review_weights_train = review_weights_train * np.where(_is_t_tr & (y_train == 0), float(TINY_NEG_WEIGHT), 1.0).astype(np.float32)

    def _get_review_tag_series(df_slice):
        if "review_tag" not in df_slice.columns:
            return pd.Series([""] * len(df_slice), index=df_slice.index)
        s = df_slice["review_tag"].astype(str).str.strip().str.lower()
        s = s.where(~s.isin({"nan", "none"}), "")
        return s

    n_skip_train = n_skip_test = 0
    if USE_REVIEW_TAG_WEIGHTS:
        tags_tr = _get_review_tag_series(df.iloc[train_idx])
        mult_tr = tags_tr.map(lambda t: float(WEIGHT_MAP.get(t, 1.0))).astype(np.float32).to_numpy()
        if DROP_SKIP_ROWS:
            keep = mult_tr > 0.0
            if (~keep).any():
                n_skip_train = int((~keep).sum())
                X_train = X_train.iloc[keep]; y_train = y_train[keep]; groups_train = groups_train[keep]
                review_weights_train = review_weights_train[keep]; base_review_weights = base_review_weights[keep]
                _is_b_tr = _is_b_tr[keep]; _is_t_tr = _is_t_tr[keep]
                train_idx = train_idx[keep]; mult_tr = mult_tr[keep]
        review_weights_train = (review_weights_train.astype(np.float32) * mult_tr.astype(np.float32))
        tags_te = _get_review_tag_series(df.iloc[test_idx])
        mult_te = tags_te.map(lambda t: float(WEIGHT_MAP.get(t, 1.0))).astype(np.float32).to_numpy()
        if DROP_SKIP_ROWS:
            keep = mult_te > 0.0
            if (~keep).any():
                n_skip_test = int((~keep).sum())
                X_test = X_test.iloc[keep]; y_test = y_test[keep]; groups_test = groups_test[keep]
                test_idx = test_idx[keep]; mult_te = mult_te[keep]
    log(f"[WEIGHTS] mean={review_weights_train.mean():.3f} | skip dropped train={n_skip_train} test={n_skip_test}")

    X_train_preaug = X_train.copy(); y_train_preaug = y_train.copy()
    groups_train_preaug = groups_train.copy(); review_weights_train_preaug = review_weights_train.copy()
    train_idx_final = train_idx.copy()
    if AUG_CFG.get("apply_in_3A", True):
        X_train, y_train, review_weights_train, groups_train = apply_augs_in_3A(
            X_df=X_train, y_np=y_train, w_np=review_weights_train, g_np=groups_train, cfg=AUG_CFG,
            random_state=RANDOM_STATE, log=log)
    return {
        "df": df, "csv_used": csv_used, "NAME_COL": NAME_COL, "groups": groups, "y": y,
        "num_cols": num_cols, "dropped_cols": dropped_cols,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "groups_train": groups_train, "groups_test": groups_test,
        "review_weights_train": review_weights_train, "base_review_weights_preaug": base_review_weights,
        "fit_weights_preaug": review_weights_train_preaug, "X_train_preaug": X_train_preaug,
        "y_train_preaug": y_train_preaug, "groups_train_preaug": groups_train_preaug,
        "train_idx": train_idx_final, "test_idx": test_idx, "split": split_record, "prev": prev_record,
        "n_border_test_dropped": n_border_test, "n_skip_train": n_skip_train, "n_skip_test": n_skip_test,
        "cfg": C, "AUG_CFG": AUG_CFG,
    }


def xgb_setup(device: str, n_cpu: int | None = None, random_state: int = 42) -> Dict[str, Any]:
    """Cell3A XGBoost setup with the device made explicit (notebook: import cupy)."""

    import xgboost as xgb
    from packaging import version

    use_cuda = (str(device) == "cuda")
    if use_cuda:
        import cupy  # noqa: F401  (the notebook's CUDA criterion; must succeed)
    N_CPU = n_cpu or os.cpu_count() or 1
    if use_cuda:
        grid_n_jobs = 1
        xgb_n_jobs = max(1, min(8, N_CPU))
    else:
        grid_n_jobs = max(1, min(4, N_CPU))
        xgb_n_jobs = max(1, N_CPU // max(1, grid_n_jobs))
    os.environ["OMP_NUM_THREADS"] = str(xgb_n_jobs)
    os.environ["MKL_NUM_THREADS"] = str(xgb_n_jobs)
    xgb_kwargs = dict(objective="binary:logistic", eval_metric="aucpr", n_estimators=300, learning_rate=0.05,
                      max_depth=6, subsample=0.9, colsample_bytree=0.9, min_child_weight=1.0, reg_lambda=1.0,
                      scale_pos_weight=1.0, random_state=random_state, n_jobs=xgb_n_jobs, verbosity=0)
    if version.parse(xgb.__version__) >= version.parse("2.0.0"):
        xgb_kwargs["tree_method"] = "hist"
        xgb_kwargs["max_bin"] = 256
        if use_cuda:
            xgb_kwargs["device"] = "cuda"
    else:
        xgb_kwargs["tree_method"] = "gpu_hist" if use_cuda else "hist"
    return {"use_cuda": use_cuda, "grid_n_jobs": grid_n_jobs, "xgb_n_jobs": xgb_n_jobs, "xgb_kwargs": xgb_kwargs,
            "n_cpu": N_CPU}


def build_smote(y_like: np.ndarray, pos_label: int = 1, sampling_strategy: float = 0.5, random_state: int = 42):
    from imblearn.over_sampling import SMOTE

    pos_count = int((y_like == pos_label).sum())
    if pos_count < 2:
        return None
    k = max(1, min(5, pos_count - 1))
    return SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k, random_state=random_state)


def fit_with_maybe_callbacks(model, X, y, **kw):
    sig = inspect.signature(model.fit)
    if "callbacks" not in sig.parameters and "callbacks" in kw:
        kw.pop("callbacks", None)
    return model.fit(X, y, **kw)


# ------------------------------------------------------------------ Cell3B driver
def fit_and_evaluate(prep: Dict[str, Any], setup: Dict[str, Any], *, round_tag: str, model_name: str,
                     prev_best_params: Dict[str, Any], log=print) -> Dict[str, Any]:
    import pandas as pd
    import xgboost as xgb
    from imblearn.pipeline import Pipeline as ImbPipeline
    from scipy.stats import loguniform, randint, uniform
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterSampler
    from xgboost import XGBClassifier

    C = prep["cfg"]
    RANDOM_STATE = int(C["random_state"]); POS_LABEL = int(C["pos_label"])
    EARLY_STOP_ROUNDS = int(C["early_stop_rounds_effective"])
    EARLY_STOP_VAL_FRAC = float(C["early_stop_val_frac"]); N_ESTIMATORS_BIG = int(C["n_estimators_big"])
    N_SPLITS_CV = int(C["n_splits_cv"]); FORCE_HPARAM_SEARCH = bool(C["force_hparam_search"])
    HPARAM_N_ITER = int(C["hparam_n_iter"]); USE_WEIGHTED_SEARCH = bool(C["use_weighted_search"])
    SEARCH_EVERY_N_ROUNDS = int(C["search_every_n_rounds"]); N_EST_SEARCH = int(C["n_est_search"])
    xgb_kwargs = dict(setup["xgb_kwargs"])

    num_cols = list(prep["num_cols"])
    X_train = prep["X_train"]; X_test = prep["X_test"]
    y_train = np.asarray(prep["y_train"]).astype(int); y_test = np.asarray(prep["y_test"]).astype(int)
    groups_train = np.asarray(prep["groups_train"]); groups_test = np.asarray(prep["groups_test"])
    review_weights_train = np.asarray(prep["review_weights_train"]).astype(np.float32)

    def ensure_numeric_df(d, cols):
        out = d.copy()
        for c in cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    X_train = ensure_numeric_df(X_train, num_cols)
    X_test = ensure_numeric_df(X_test, num_cols)

    def build_cv_pairs(groups_vec, y_vec, target_k):
        uniq_g = np.unique(groups_vec)
        if len(uniq_g) < 2:
            return []
        k = min(int(target_k), len(uniq_g))
        if k < 2:
            return []
        gkf = GroupKFold(n_splits=k)
        pairs = []
        for tr_idx, va_idx in gkf.split(np.zeros(len(y_vec)), y_vec, groups_vec):
            if len(np.unique(y_vec[tr_idx])) < 2 or len(np.unique(y_vec[va_idx])) < 2:
                continue
            pairs.append((tr_idx, va_idx))
        return pairs

    cv_pairs = build_cv_pairs(groups_train, y_train, N_SPLITS_CV)
    if not cv_pairs:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE + 999)
        tr_idx, va_idx = next(gss.split(np.zeros(len(y_train)), y_train, groups_train))
        cv_pairs = [(tr_idx, va_idx)]

    def make_weights_after_smote(original_w, y_original, y_resampled):
        y_original = np.asarray(y_original); y_resampled = np.asarray(y_resampled)
        w0 = np.asarray(original_w, dtype=np.float32)
        n0 = len(y_original); n1 = len(y_resampled)
        w = np.empty(n1, dtype=np.float32)
        w[:n0] = w0[:n0]
        if n1 > n0:
            synth_y = y_resampled[n0:]
            if synth_y.size > 0:
                cls = int(pd.Series(synth_y).mode().iloc[0])
                m = float(w0[y_original == cls].mean()) if np.any(y_original == cls) else float(w0.mean())
                w[n0:] = m
            else:
                w[n0:] = float(w0.mean())
        return w

    def pick_threshold_by_pr_f1(y_true, proba):
        prec, rec, thr = precision_recall_curve(y_true, proba)
        if thr.size == 0:
            return 0.5, {"best_f1": float("nan"), "best_precision": float(prec[-1]), "best_recall": float(rec[-1])}, \
                pd.DataFrame({"precision": prec, "recall": rec, "threshold": np.r_[np.nan]})
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        bi = int(np.nanargmax(f1))
        thr_use = float(thr[max(bi - 1, 0)])
        pr_df = pd.DataFrame({"precision": prec, "recall": rec, "threshold": np.r_[np.nan, thr]})
        stats = {"best_f1": float(f1[bi]), "best_precision": float(prec[bi]), "best_recall": float(rec[bi])}
        return thr_use, stats, pr_df

    cur_r = _extract_round_num_from_name(round_tag) or _extract_round_num_from_name(model_name) or 0
    do_search = bool(FORCE_HPARAM_SEARCH)
    if not do_search and SEARCH_EVERY_N_ROUNDS and cur_r > 0:
        do_search = (cur_r % int(SEARCH_EVERY_N_ROUNDS) == 0)

    param_distributions = {
        "clf__max_depth": randint(3, 10),
        "clf__min_child_weight": loguniform(0.5, 10.0),
        "clf__subsample": uniform(0.7, 0.3),
        "clf__colsample_bytree": uniform(0.7, 0.3),
        "clf__reg_lambda": loguniform(0.1, 10.0),
        "clf__reg_alpha": loguniform(1e-3, 1.0),
        "clf__gamma": loguniform(1e-3, 1.0),
        "clf__learning_rate": loguniform(0.02, 0.2),
    }
    if "max_bin" in xgb_kwargs:
        param_distributions["clf__max_bin"] = [xgb_kwargs["max_bin"]]
    smote_possible = build_smote(np.asarray(y_train), pos_label=POS_LABEL, sampling_strategy=0.5,
                                 random_state=RANDOM_STATE) is not None
    if smote_possible:
        param_distributions.update({"smote__sampling_strategy": [0.4, 0.5, 0.6], "smote__k_neighbors": randint(2, 6)})

    def safe_smote_fit_resample(X_imp, y_tr, ss, kn):
        if not smote_possible:
            return X_imp, y_tr, None
        y_tr = np.asarray(y_tr)
        n_pos = int((y_tr == POS_LABEL).sum()); n_neg = int((y_tr != POS_LABEL).sum())
        n_min = min(n_pos, n_neg); n_maj = max(n_pos, n_neg)
        if n_maj <= 0 or n_min <= 1:
            return X_imp, y_tr, None
        cur_ratio = n_min / float(n_maj)
        ss = float(ss) if ss is not None else 0.5
        if cur_ratio >= ss - 1e-9:
            return X_imp, y_tr, None
        if kn is not None:
            kn = int(kn)
            kn = max(1, min(kn, n_min - 1))
        sm = build_smote(y_tr, pos_label=POS_LABEL, sampling_strategy=ss, random_state=RANDOM_STATE)
        if sm is None:
            return X_imp, y_tr, None
        if kn is not None and hasattr(sm, "k_neighbors"):
            sm.k_neighbors = kn
        try:
            X_fit, y_fit = sm.fit_resample(X_imp, y_tr)
            return X_fit, y_fit, sm
        except ValueError:
            return X_imp, y_tr, None

    def _search(n_iter, random_state, weighted: bool):
        sampler = list(ParameterSampler(param_distributions, n_iter=int(n_iter), random_state=int(random_state)))
        best_score = -1.0
        best_params: Dict[str, Any] = {}
        for i, params in enumerate(sampler, start=1):
            fold_scores = []
            for tr_idx, va_idx in cv_pairs:
                X_tr = X_train.iloc[tr_idx]; y_tr = y_train[tr_idx]
                X_va = X_train.iloc[va_idx]; y_va = y_train[va_idx]
                w_tr = review_weights_train[tr_idx].astype(np.float32)
                imp = SimpleImputer(strategy="median")
                X_tr_imp = imp.fit_transform(X_tr)
                X_va_imp = imp.transform(X_va)
                if weighted:
                    X_fit, y_fit, w_fit = X_tr_imp, y_tr, w_tr
                    if smote_possible:
                        ss = float(params.get("smote__sampling_strategy", 0.5))
                        kn = params.get("smote__k_neighbors", None)
                        X_fit, y_fit, sm_used = safe_smote_fit_resample(X_tr_imp, y_tr, ss, kn)
                        w_fit = make_weights_after_smote(w_tr, y_tr, y_fit) if (sm_used is not None) else w_tr
                else:
                    if smote_possible:
                        ss = float(params.get("smote__sampling_strategy", 0.5))
                        kn = params.get("smote__k_neighbors", None)
                        X_fit, y_fit, _ = safe_smote_fit_resample(X_tr_imp, y_tr, ss, kn)
                    else:
                        X_fit, y_fit = X_tr_imp, y_tr
                clf = XGBClassifier(**xgb_kwargs)
                clf_only = {k.replace("clf__", ""): v for k, v in params.items() if k.startswith("clf__")}
                if clf_only:
                    clf.set_params(**clf_only)
                clf.set_params(n_estimators=int(N_EST_SEARCH), early_stopping_rounds=int(EARLY_STOP_ROUNDS),
                               eval_metric="aucpr")
                if weighted:
                    clf.fit(X_fit, y_fit, sample_weight=w_fit, eval_set=[(X_va_imp, y_va)], verbose=False)
                else:
                    clf.fit(X_fit, y_fit, eval_set=[(X_va_imp, y_va)], verbose=False)
                proba = clf.predict_proba(X_va_imp)[:, 1]
                fold_scores.append(float(average_precision_score(y_va, proba)))
            mean_sc = float(np.mean(fold_scores)) if fold_scores else float("nan")
            if np.isfinite(mean_sc) and mean_sc > best_score:
                best_score = mean_sc
                best_params = dict(params)
        return best_params, float(best_score)

    best_params: Dict[str, Any] = {}
    cv_best_pr = None
    param_source = "base"
    if len(np.unique(y_train)) < 2:
        best_params = {}
    else:
        if (not do_search) and prev_best_params:
            best_params = dict(prev_best_params)
            param_source = "prev_best"
        elif do_search and USE_WEIGHTED_SEARCH:
            with Timer("weighted_search", log):
                best_params, cv_best_pr = _search(HPARAM_N_ITER, RANDOM_STATE, weighted=True)
            param_source = "weighted_search"
        elif do_search and (not USE_WEIGHTED_SEARCH):
            with Timer("unweighted_search", log):
                best_params, cv_best_pr = _search(HPARAM_N_ITER, RANDOM_STATE, weighted=False)
            param_source = "unweighted_search"
        else:
            best_params = {}
            param_source = "base"
    log(f"[HP] do_search={do_search} source={param_source} best_params={best_params}")

    def refit_es_pipeline(X_tr_df, y_tr_np, g_tr_np, w_tr_np, params):
        if len(np.unique(g_tr_np)) >= 2 and len(y_tr_np) >= 50:
            gss = GroupShuffleSplit(n_splits=1, test_size=float(EARLY_STOP_VAL_FRAC), random_state=RANDOM_STATE + 123)
            tr_idx, va_idx = next(gss.split(np.zeros(len(y_tr_np)), y_tr_np, g_tr_np))
        else:
            idx = np.arange(len(y_tr_np))
            rs = np.random.RandomState(RANDOM_STATE + 123); rs.shuffle(idx)
            cut = max(1, int(len(idx) * float(EARLY_STOP_VAL_FRAC)))
            va_idx, tr_idx = idx[:cut], idx[cut:]
        if len(np.unique(y_tr_np[va_idx])) < 2:
            idx = np.arange(len(y_tr_np))
            rs = np.random.RandomState(RANDOM_STATE + 777); rs.shuffle(idx)
            cut = max(1, int(len(idx) * float(EARLY_STOP_VAL_FRAC)))
            va_idx, tr_idx = idx[:cut], idx[cut:]
        X_tr = X_tr_df.iloc[tr_idx]; y_tr = y_tr_np[tr_idx]; w_tr = w_tr_np[tr_idx].astype(np.float32)
        X_va = X_tr_df.iloc[va_idx]; y_va = y_tr_np[va_idx]
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_tr)
        X_va_imp = imp.transform(X_va)
        X_fit, y_fit, w_fit = X_tr_imp, y_tr, w_tr
        sm_used = None
        if smote_possible:
            ss = float(params.get("smote__sampling_strategy", 0.5))
            kn = params.get("smote__k_neighbors", None)
            X_fit, y_fit, sm_used = safe_smote_fit_resample(X_tr_imp, y_tr, ss, kn)
            if sm_used is not None:
                w_fit = make_weights_after_smote(w_tr, y_tr, y_fit)
        clf = XGBClassifier(**xgb_kwargs)
        clf_only = {k.replace("clf__", ""): v for k, v in params.items() if k.startswith("clf__")}
        if clf_only:
            clf.set_params(**clf_only)
        clf.set_params(n_estimators=int(N_ESTIMATORS_BIG), early_stopping_rounds=int(EARLY_STOP_ROUNDS), eval_metric="aucpr")
        with Timer("fit ES", log):
            fit_with_maybe_callbacks(clf, X_fit, y_fit, sample_weight=w_fit, eval_set=[(X_va_imp, y_va)], verbose=False,
                                     callbacks=[xgb.callback.EvaluationMonitor(period=25)])
        es_record = {"es_train_rows": int(len(tr_idx)), "es_val_rows": int(len(va_idx)),
                     "es_val_idx_sha": _sha_array(np.asarray(va_idx)), "smote_used": sm_used is not None,
                     "fit_rows": int(len(y_fit))}
        return ImbPipeline([("imp", imp), ("smote", sm_used if sm_used is not None else "passthrough"), ("clf", clf)]), es_record

    def full_train_refit(pipe_es, X_tr_df, y_tr_np, w_tr_np, params):
        imp = pipe_es.named_steps["imp"]
        clf_es = pipe_es.named_steps["clf"]
        best_iter = getattr(clf_es, "best_iteration", None)
        used_trees = int(best_iter) + 1 if best_iter is not None else int(getattr(clf_es, "n_estimators", N_ESTIMATORS_BIG))
        X_imp = imp.fit_transform(X_tr_df)
        X_fit, y_fit, w_fit = X_imp, y_tr_np, w_tr_np.astype(np.float32)
        sm_used = None
        if smote_possible:
            ss = float(params.get("smote__sampling_strategy", 0.5))
            kn = params.get("smote__k_neighbors", None)
            X_fit, y_fit, sm_used = safe_smote_fit_resample(X_imp, y_tr_np, ss, kn)
            if sm_used is not None:
                w_fit = make_weights_after_smote(w_tr_np.astype(np.float32), y_tr_np, y_fit)
        clf = XGBClassifier(**xgb_kwargs)
        clf_only = {k.replace("clf__", ""): v for k, v in params.items() if k.startswith("clf__")}
        if clf_only:
            clf.set_params(**clf_only)
        clf.set_params(n_estimators=int(used_trees), early_stopping_rounds=None, eval_metric="aucpr")
        with Timer("full-train fit", log):
            fit_with_maybe_callbacks(clf, X_fit, y_fit, sample_weight=w_fit, verbose=False,
                                     callbacks=[xgb.callback.EvaluationMonitor(period=25)])
        return ImbPipeline([("imp", imp), ("smote", sm_used if sm_used is not None else "passthrough"), ("clf", clf)]), int(used_trees), \
            {"full_fit_rows": int(len(y_fit)), "smote_used": sm_used is not None}

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("TRAIN is single-class after filtering; cannot train XGB. (Check skip-dropping / split / labels)")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with Timer("refit_es_pipeline", log):
            pipe_es, es_record = refit_es_pipeline(X_train, y_train, groups_train, review_weights_train, best_params)
        with Timer("full_train_refit", log):
            best_model, used_trees, full_record = full_train_refit(pipe_es, X_train, y_train, review_weights_train, best_params)
    log(f"[MODEL] used_trees={used_trees}")

    def predict_proba_batched(model, Xdf, batch=50000):
        n = len(Xdf)
        out = np.empty(n, dtype=np.float32)
        for i in range(0, n, batch):
            j = min(i + batch, n)
            out[i:j] = model.predict_proba(Xdf.iloc[i:j])[:, 1]
        return out

    use_test = (len(y_test) > 0) and (len(np.unique(y_test)) >= 2)
    eval_split = "test" if use_test else "train"
    if use_test:
        proba_eval = predict_proba_batched(best_model, X_test); y_eval = y_test; g_eval = groups_test
    else:
        proba_eval = predict_proba_batched(best_model, X_train); y_eval = y_train; g_eval = groups_train
    tile_pr_auc = float(average_precision_score(y_eval, proba_eval))
    tile_roc_auc = float(roc_auc_score(y_eval, proba_eval)) if len(np.unique(y_eval)) >= 2 else float("nan")
    thr_use, thr_stats, pr_df = pick_threshold_by_pr_f1(y_eval, proba_eval)
    y_pred = (proba_eval >= thr_use).astype(int)
    cm = confusion_matrix(y_eval, y_pred)
    fpr, tpr, roc_thr = roc_curve(y_eval, proba_eval)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr})
    df = prep["df"]; NAME_COL = prep["NAME_COL"]
    tile_preds_df = None
    img_preds_df = None
    img_pr_auc = float("nan"); img_roc_auc = float("nan")
    if eval_split == "test":
        test_idx = prep["test_idx"]
        if test_idx is None or len(test_idx) != len(y_test):
            base = pd.DataFrame({NAME_COL: np.arange(len(y_test)).astype(str)})
            base["group"] = groups_test; base["label"] = y_test; base["proba"] = proba_eval
            base["pred"] = (base["proba"] >= thr_use).astype(int)
            tile_preds_df = base
        else:
            base = df.iloc[np.asarray(test_idx)].copy()
            base["group"] = groups_test; base["label"] = y_test; base["proba"] = proba_eval
            base["pred"] = (base["proba"] >= thr_use).astype(int)
            keep_cols = [c for c in [NAME_COL, "group", "label", "proba", "pred", "reviewed", "review_tag", "scale",
                                     "ann_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"] if c in base.columns]
            tile_preds_df = base[keep_cols].copy()
        agg = tile_preds_df.groupby("group", as_index=False).agg(y_img=("label", "max"), p_img=("proba", "max"),
                                                                 n_tiles=("proba", "size"))
        agg["pred_img"] = (agg["p_img"] >= thr_use).astype(int)
        img_preds_df = agg
        if len(np.unique(agg["y_img"].values)) >= 2:
            img_pr_auc = float(average_precision_score(agg["y_img"].values, agg["p_img"].values))
            img_roc_auc = float(roc_auc_score(agg["y_img"].values, agg["p_img"].values))
    clf_final = best_model.named_steps["clf"]
    fi = getattr(clf_final, "feature_importances_", None)
    fi_df = None
    if fi is not None and len(fi) == len(num_cols):
        fi_df = pd.DataFrame({"feature": num_cols, "importance": fi.astype(float)})
        fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return {
        "best_model": best_model, "used_trees": used_trees, "best_params": best_params, "param_source": param_source,
        "cv_best_pr": cv_best_pr, "do_search": do_search, "eval_split": eval_split, "proba_eval": proba_eval,
        "y_eval": y_eval, "tile_pr_auc": tile_pr_auc, "tile_roc_auc": tile_roc_auc, "thr_use": thr_use,
        "thr_stats": thr_stats, "pr_df": pr_df, "roc_df": roc_df, "cm": cm, "tile_preds_df": tile_preds_df,
        "img_preds_df": img_preds_df, "img_pr_auc": img_pr_auc, "img_roc_auc": img_roc_auc, "fi_df": fi_df,
        "es": es_record, "full": full_record, "cv_folds": len(cv_pairs), "xgb_kwargs": xgb_kwargs,
        "smote_possible": smote_possible,
    }


def _sha_array(a: np.ndarray) -> str:
    import hashlib

    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes() + str(a.dtype).encode() + str(a.shape).encode()).hexdigest()


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def load_prev_best_params(state: LegacyState, model_root_rel: str, round_tag: str, round_dir_name: str,
                          auto_detect: bool = True) -> Tuple[Dict[str, Any], Optional[str]]:
    if not auto_detect:
        return {}, None
    cur = _extract_round_num_from_name(round_tag)
    if cur is None:
        return {}, None
    best = None; best_r = -1
    for name, _ in _iter_round_dirs(state, model_root_rel):
        rel = f"{model_root_rel}/{name}"
        if not state.path(rel).is_dir():
            continue
        if name == round_dir_name:
            continue
        bp = state.path(f"{rel}/best_params_used.json")
        if not bp.exists():
            continue
        r = _extract_round_num_from_name(name) or _extract_round_num_from_name(str(state.path(rel)))
        if r is None:
            continue
        if r < cur and r > best_r:
            best_r = r; best = rel
    if best is None:
        return {}, None
    try:
        obj = json.loads(state.path(f"{best}/best_params_used.json").read_text(encoding="utf-8"))
        return (obj if isinstance(obj, dict) else {}), best
    except Exception:
        return {}, best


def write_round_artifacts(*, state: LegacyState, model_root_rel: str, round_dir_name: str, round_tag: str,
                          model_name: str, prep: Dict[str, Any], fit: Dict[str, Any], csv_path_recorded: str,
                          lineage: Dict[str, Any], write_legacy_pickle: bool = True, log=print) -> Dict[str, Any]:
    """Cell3B 'Save everything' with the original filenames and serializers."""

    import joblib
    import pandas as pd
    import xgboost as xgb

    rd = f"{model_root_rel}/{round_dir_name}"
    P = lambda name: state.write_path(f"{rd}/{name}")  # noqa: E731
    best_params = fit["best_params"]
    (P("best_params_used.json")).write_text(json.dumps({k: _jsonable(v) for k, v in (best_params or {}).items()}, indent=2), encoding="utf-8")
    y_train = np.asarray(prep["y_train"]); y_test = np.asarray(prep["y_test"])
    img_pr_auc = fit["img_pr_auc"]; img_roc_auc = fit["img_roc_auc"]
    meta = {
        "csv_used": str(prep["csv_used"]), "csv_path": str(csv_path_recorded), "round_tag": round_tag,
        "model_name": model_name, "xgboost_version": xgb.__version__, "params_source": fit["param_source"],
        "cv_best_pr_auc": (None if fit["cv_best_pr"] is None else float(fit["cv_best_pr"])),
        "used_trees": int(fit["used_trees"]), "tile_pr_auc": float(fit["tile_pr_auc"]),
        "tile_roc_auc": float(fit["tile_roc_auc"]),
        "img_pr_auc": (None if np.isnan(img_pr_auc) else float(img_pr_auc)),
        "img_roc_auc": (None if np.isnan(img_roc_auc) else float(img_roc_auc)),
        "threshold": float(fit["thr_use"]), "threshold_stats": fit["thr_stats"],
        "train_n": int(len(y_train)), "test_n": int(len(y_test)),
        "pos_rate_train": float(np.mean(y_train == 1)),
        "pos_rate_test": float(np.mean(y_test == 1)) if len(y_test) else None,
        "weights_mean_train": float(np.mean(prep["review_weights_train"])),
        "best_params": best_params if isinstance(best_params, dict) else {},
    }
    model_path = P(model_name)
    if write_legacy_pickle:
        joblib.dump({"pipeline": fit["best_model"], "features": list(prep["num_cols"]),
                     "threshold": float(fit["thr_use"]), "meta": meta}, model_path)
    P(f"threshold_{round_tag}.json").write_text(json.dumps({"threshold": float(fit["thr_use"])}, indent=2), encoding="utf-8")
    fit["pr_df"].to_csv(P("pr_curve.csv"), index=False)
    fit["roc_df"].to_csv(P("roc_curve.csv"), index=False)
    if fit["tile_preds_df"] is not None:
        fit["tile_preds_df"].to_csv(P("tile_preds.csv"), index=False)
    if fit["img_preds_df"] is not None:
        fit["img_preds_df"].to_csv(P("image_preds.csv"), index=False)
    pd.DataFrame(fit["cm"], index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(P("confusion_matrix.csv"))
    if fit["fi_df"] is not None:
        fit["fi_df"].to_csv(P("feature_importances.csv"), index=False)
    P("dropped_cols.json").write_text(json.dumps({"dropped_cols": prep["dropped_cols"]}, indent=2), encoding="utf-8")
    C = prep["cfg"]
    cfg_out = {
        "TEST_SIZE": C["test_size"], "RANDOM_STATE": C["random_state"], "N_SPLITS_CV": C["n_splits_cv"],
        "POS_LABEL": C["pos_label"], "BORDER_PX": C["border_px"], "TILE_SIZE": C["tile_size"],
        "TINY_MIN_WH": C["tiny_min_wh"], "TINY_MAX_AREA": C["tiny_max_area"], "BORDER_WEIGHT": C["border_weight"],
        "UNCERT_MARGIN": C["uncert_margin"], "UNCERT_MULT": C["uncert_mult"],
        "DROP_TINY_IN_TRAIN": C["drop_tiny_in_train"], "DROP_BORDER_FROM_TEST": C["drop_border_from_test"],
        "EARLY_STOP_ROUNDS": C["early_stop_rounds_effective"], "EARLY_STOP_VAL_FRAC": C["early_stop_val_frac"],
        "N_ESTIMATORS_BIG": C["n_estimators_big"], "USE_REVIEW_WEIGHTING": C["use_review_weighting"],
        "WEIGHT_CORRECT": C["weight_correct"], "WEIGHT_WRONG": C["weight_wrong"],
        "AUTO_DETECT_PREV_ROUND": C["auto_detect_prev_round"], "USE_REVIEW_TAG_WEIGHTS": C["use_review_tag_weights"],
        "WEIGHT_MAP": C["weight_map"], "DROP_SKIP_ROWS": C["drop_skip_rows"], "AUG_CFG": prep["AUG_CFG"],
    }
    cfg_out.update({"xgb_kwargs": fit["xgb_kwargs"], "best_params": best_params, "param_source": fit["param_source"],
                    "csv_used": str(prep["csv_used"]), "n_features": int(len(prep["num_cols"]))})
    P("config.json").write_text(json.dumps(cfg_out, indent=2, default=str), encoding="utf-8")
    metrics_out = {
        "tile": {"split": fit["eval_split"], "pr_auc": float(fit["tile_pr_auc"]), "roc_auc": float(fit["tile_roc_auc"]),
                 "threshold": float(fit["thr_use"]), "confusion_matrix": fit["cm"].tolist(),
                 "threshold_stats": fit["thr_stats"]},
        "image": {"available": bool(fit["img_preds_df"] is not None),
                  "pr_auc": (None if np.isnan(img_pr_auc) else float(img_pr_auc)),
                  "roc_auc": (None if np.isnan(img_roc_auc) else float(img_roc_auc))},
        "meta": meta,
    }
    P("metrics.json").write_text(json.dumps(metrics_out, indent=2, default=str), encoding="utf-8")

    # history.csv append + progress.png (model root)
    hist_rel = f"{model_root_rel}/history.csv"
    row = {"ts": pd.Timestamp.utcnow().isoformat(), "round_tag": round_tag, "model_name": model_name,
           "param_source": fit["param_source"], "tile_pr_auc": float(fit["tile_pr_auc"]),
           "tile_roc_auc": float(fit["tile_roc_auc"]) if np.isfinite(fit["tile_roc_auc"]) else np.nan,
           "img_pr_auc": (np.nan if np.isnan(img_pr_auc) else float(img_pr_auc)),
           "img_roc_auc": (np.nan if np.isnan(img_roc_auc) else float(img_roc_auc)),
           "threshold": float(fit["thr_use"]), "n_features": int(len(prep["num_cols"])),
           "train_n": int(len(y_train)), "test_n": int(len(y_test))}
    hist = None
    if state.exists(hist_rel):
        try:
            hist = pd.read_csv(state.path(hist_rel))
        except Exception:
            hist = None
    if hist is None:
        hist = pd.DataFrame(columns=list(row.keys()))
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(state.write_path(hist_rel), index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _rnum(rt):
            m = re.search(r"r(\d+)", str(rt), flags=re.IGNORECASE)
            return float(m.group(1)) if m else float("nan")
        hist["_rnum"] = hist["round_tag"].map(_rnum)
        hist2 = hist.sort_values(["_rnum", "ts"], ascending=[True, True])
        plt.figure()
        plt.plot(hist2["_rnum"].to_numpy(), hist2["tile_pr_auc"].to_numpy(), marker="o")
        plt.xlabel("round"); plt.ylabel("tile PR-AUC"); plt.title("Training progress"); plt.grid(True)
        plt.savefig(state.write_path(f"{model_root_rel}/progress.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:  # pragma: no cover
        log(f"[WARN] progress plot failed: {e}")

    identity = {
        "schema": "compag-only-codes-model-identity/v1",
        "model_id": f"only-codes-compat-{uuid.uuid4()}",
        "legacy_round_tag": round_tag,
        "legacy_file_name": model_name,
        "note": ("Newly fitted by the compatibility workflow. The legacy round tag and file name are part of the "
                 "legacy output contract only; they do not imply identity with any historical artifact of that name."),
        "contract_sha256": contract_sha256(),
        "lineage": lineage,
        "split": {"test_groups": prep["split"]["test_groups"],
                  "train_rows_sha256": _sha_array(np.asarray(prep["train_idx"])),
                  "test_rows_sha256": _sha_array(np.asarray(prep["test_idx"]))},
        "weights": {"base_review_weights_mean_preaug": float(np.mean(prep["base_review_weights_preaug"])),
                    "fit_weights_mean_preaug": float(np.mean(prep["fit_weights_preaug"])),
                    "fit_weights_mean_postaug": float(np.mean(prep["review_weights_train"]))},
        "es": fit["es"], "full": fit["full"], "prev": prep["prev"],
    }
    P("ONLY_CODES_MODEL_IDENTITY.json").write_text(json.dumps(identity, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return {"round_dir": str(state.overlay(rd)), "model_path": str(model_path), "identity": identity, "meta": meta}
