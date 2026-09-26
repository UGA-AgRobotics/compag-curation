# sam2_pipeline/xgb_utils.py
from __future__ import annotations
from typing import Optional, Callable, Any, List
from pathlib import Path
import hashlib
import math
import json
import os
import re
import stat
from typing import Tuple, Dict
import numpy as np

try:
    import pandas as pd  # Used to pass column names to scikit-learn
except Exception:
    pd = None  # If pandas is not available, it still works but may raise warnings

# -------------------- Border / Uncertainty helpers (training-time) --------------------
def border_and_tiny_masks_from_df(
    df_like: "pd.DataFrame",
    *,
    tile_size: int = 512,
    border_px: int = 4,
    xcol: str = "bbox_x",
    ycol: str = "bbox_y",
    wcol: str = "bbox_w",
    hcol: str = "bbox_h",
    touch_col: str = "touching_border",
    tiny_min_wh: int = 32,
    tiny_max_area: int = 1500,
) -> "tuple[np.ndarray, np.ndarray]":
    x = pd.to_numeric(df_like.get(xcol, 0), errors="coerce").fillna(0.0).to_numpy(np.float32)
    y = pd.to_numeric(df_like.get(ycol, 0), errors="coerce").fillna(0.0).to_numpy(np.float32)
    w = pd.to_numeric(df_like.get(wcol, 0), errors="coerce").fillna(0.0).to_numpy(np.float32)
    h = pd.to_numeric(df_like.get(hcol, 0), errors="coerce").fillna(0.0).to_numpy(np.float32)
    touch = pd.to_numeric(df_like.get(touch_col, 0), errors="coerce").fillna(0).to_numpy(np.int32) == 1
    is_border = (
        touch
        | (x <= float(border_px))
        | (y <= float(border_px))
        | ((x + w) >= float(tile_size - border_px))
        | ((y + h) >= float(tile_size - border_px))
    )
    is_tiny = (np.minimum(w, h) < float(tiny_min_wh)) | ((w * h) < float(tiny_max_area))
    return is_border.astype(bool), is_tiny.astype(bool)


def load_prev_prob_map_and_thr(
    prev_preds_path: Path,
    prev_thr_path: Path,
    name_col: str,
) -> "Optional[tuple[dict[str, tuple[int, float]], float]]":
    try:
        prev = pd.read_csv(prev_preds_path)
    except Exception:
        return None
    try:
        with open(prev_thr_path, "r") as f:
            thr = float(json.load(f)["threshold"])
    except Exception:
        return None
    if name_col not in prev.columns or "proba" not in prev.columns:
        return None
    tmp = prev[[name_col, "proba"]].copy()
    tmp = tmp.groupby(name_col, as_index=False, sort=False)["proba"].max()
    tmp["pred"] = (tmp["proba"] >= thr).astype(int)
    mapping = {str(n): (int(p), float(s)) for n, p, s in zip(tmp[name_col].astype(str), tmp["pred"], tmp["proba"])}
    return mapping, thr


def apply_border_and_uncertainty_weighting(
    df_like: "pd.DataFrame",
    row_indices: "np.ndarray",
    base_weights: "np.ndarray",
    *,
    name_col: str,
    prev_preds_path: "Optional[Path]" = None,
    prev_thr_path: "Optional[Path]" = None,
    reviewed_col: str = "reviewed",
    tile_size: int = 512,
    border_px: int = 4,
    tiny_min_wh: int = 32,
    tiny_max_area: int = 1500,
    border_weight: float = 0.5,
    uncert_margin: float = 0.05,
    uncert_mult: float = 0.7,
    drop_tiny: bool = True,
) -> "tuple[np.ndarray, np.ndarray]":
    idx = np.asarray(row_indices, dtype=np.int64)
    w = np.asarray(base_weights, dtype=np.float32)
    view = df_like.iloc[idx]
    is_border, is_tiny = border_and_tiny_masks_from_df(
        view, tile_size=tile_size, border_px=border_px,
        tiny_min_wh=tiny_min_wh, tiny_max_area=tiny_max_area
    )
    if drop_tiny and is_tiny.any():
        keep = ~is_tiny
        idx = idx[keep]
        w = w[keep]
        view = df_like.iloc[idx]
        is_border = is_border[keep]
    reviewed = pd.to_numeric(view.get(reviewed_col, 0), errors="coerce").fillna(0).to_numpy(np.int32) == 1
    if np.any(reviewed & is_border):
        w = w * np.where(reviewed & is_border, float(border_weight), 1.0).astype(np.float32)
    if prev_preds_path is not None and prev_thr_path is not None:
        prev = load_prev_prob_map_and_thr(prev_preds_path, prev_thr_path, name_col=name_col)
        if prev is not None:
            mapping, thr = prev
            names = view[name_col].astype(str).to_numpy()
            proba_prev = np.array([mapping.get(n, (-1, np.nan))[1] for n in names], dtype=np.float32)
            pred_prev = np.array([mapping.get(n, (-1, np.nan))[0] for n in names], dtype=np.int32)
            known = pred_prev >= 0
            if np.any(known & reviewed):
                close = np.abs(proba_prev - float(thr)) <= float(uncert_margin)
                w = w * np.where(known & reviewed & close, float(uncert_mult), 1.0).astype(np.float32)
    return idx, w


def drop_border_from_indices_for_eval(
    df_like: "pd.DataFrame",
    row_indices: "np.ndarray",
    *,
    tile_size: int = 512,
    border_px: int = 4,
) -> "np.ndarray":
    idx = np.asarray(row_indices, dtype=np.int64)
    is_border, _ = border_and_tiny_masks_from_df(df_like.iloc[idx], tile_size=tile_size, border_px=border_px)
    return idx[~is_border]


# Keep this if used elsewhere
def finalize_auto_xgb_names(feature_mode: str, embed_pca_names: List[str]) -> List[str]:
    from .defaults import XGB_FEATS_BALANCED_DEFAULT, XGB_FEATS_RICH_DEFAULT
    base = (XGB_FEATS_RICH_DEFAULT if feature_mode in ("rich", "ultra") else XGB_FEATS_BALANCED_DEFAULT)
    if feature_mode == "ultra" and embed_pca_names:
        return base + list(embed_pca_names)
    return base

def uncertainty(p: float, metric: str) -> float:
    if metric == "entropy":
        p = max(1e-9, min(1 - 1e-9, p))
        return float(-(p * math.log(p) + (1 - p) * math.log(1 - p)))
    else:  # margin
        return float(1.0 - abs(2 * p - 1.0))

def _to_dataframe(X: Any, features: Optional[List[str]]):
    """
    Convert X to a DataFrame with columns aligned to features.
    If features=None, use available columns if possible.
    """
    if pd is None:
        # Fallback: no pandas
        arr = np.asarray(X, dtype=float)
        if features is not None:
            # Try to align shape with expected number of columns
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != len(features):
                raise ValueError(f"Shape mismatch: X has {arr.shape[1]} cols but model expects {len(features)}")
        return arr  # Might raise a warning, but still works

    # dict → single row
    if isinstance(X, dict):
        if features is not None:
            data = {k: float(X.get(k, 0.0)) for k in features}
            return pd.DataFrame([data], columns=features)
        return pd.DataFrame([X])

    # DataFrame → reindex
    if hasattr(X, "reindex"):
        if features is not None:
            return X.reindex(columns=features, fill_value=0.0)
        return X

    # numpy/list → DataFrame with column names
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if features is not None:
        return pd.DataFrame(arr, columns=features)
    return pd.DataFrame(arr)

# --- NEW: robust predict_proba for sklearn/imblearn/cuML pipelines ---
def _final_estimator(mdl):
    # If it's a Pipeline, return the last step (estimator); otherwise return mdl itself
    if hasattr(mdl, "steps") and mdl.steps:
        return mdl.steps[-1][1]
    return mdl

def _predict_proba_robust(mdl, X_use):
    # First try the object itself (may be a Pipeline)
    try:
        return mdl.predict_proba(X_use)
    except Exception as e:
        msg = str(e)
        # Handle imblearn/sklearn checks that break with cuML (no __sklearn_tags__)
        if "__sklearn_tags__" in msg or "check_is_fitted" in msg:
            est = _final_estimator(mdl)
            Xu = X_use.values if hasattr(X_use, "values") else np.asarray(X_use, dtype=float)
            return est.predict_proba(Xu)
        raise

_SAFE_XGB_MODEL_SUFFIXES = frozenset({".json", ".ubj"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_XGB_MODEL_BYTES = 512 * 1024 * 1024


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _read_safe_xgb_model(
    model_path: str,
    expected_sha256: Optional[str],
) -> tuple[bytearray, str]:
    """Read one safe-format model from a stable, non-linked file descriptor."""

    rendered = str(model_path).strip()
    if not rendered:
        raise ValueError("XGBoost model path must not be empty")
    absolute = Path(os.path.abspath(rendered))
    suffix = absolute.suffix.lower()
    if suffix not in _SAFE_XGB_MODEL_SUFFIXES:
        raise ValueError("XGBoost model must use the safe .json or .ubj format")
    if expected_sha256 is not None and _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("XGBoost model expected SHA-256 must be lowercase hexadecimal")

    try:
        initial = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("XGBoost model file is unavailable") from exc
    if resolved != absolute:
        raise RuntimeError("XGBoost model locator must not contain symbolic links")
    if (
        absolute.is_symlink()
        or not stat.S_ISREG(initial.st_mode)
        or initial.st_nlink != 1
    ):
        raise RuntimeError("XGBoost model must be a regular non-linked file")
    if initial.st_size <= 0 or initial.st_size > _MAX_XGB_MODEL_BYTES:
        raise RuntimeError("XGBoost model size is outside the supported bound")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise RuntimeError("XGBoost model could not be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not _same_file_identity(initial, opened)
        ):
            raise RuntimeError("XGBoost model file identity changed before reading")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_XGB_MODEL_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_XGB_MODEL_BYTES:
                raise RuntimeError("XGBoost model size is outside the supported bound")
        completed = os.fstat(descriptor)
    except OSError as exc:
        raise RuntimeError("XGBoost model could not be read safely") from exc
    finally:
        os.close(descriptor)

    try:
        current = absolute.lstat()
    except OSError as exc:
        raise RuntimeError("XGBoost model path changed while reading") from exc
    stable_metadata = (
        opened.st_size == completed.st_size == total
        and opened.st_mtime_ns == completed.st_mtime_ns
        and opened.st_ctime_ns == completed.st_ctime_ns
    )
    if (
        not stable_metadata
        or not _same_file_identity(opened, completed)
        or not _same_file_identity(opened, current)
        or not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
    ):
        raise RuntimeError("XGBoost model file identity changed while reading")

    payload = b"".join(chunks)
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise RuntimeError("XGBoost model SHA-256 does not match")
    stripped = payload.lstrip()
    if not stripped or stripped.startswith(b"\x80"):
        raise ValueError("XGBoost model payload is empty or pickle-like")
    if suffix == ".json":
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("XGBoost JSON model payload is invalid") from exc
        if (
            not isinstance(parsed, dict)
            or not isinstance(parsed.get("learner"), dict)
            or not isinstance(parsed.get("version"), list)
        ):
            raise ValueError("XGBoost JSON model structure is invalid")
    elif not payload.startswith(b"{L"):
        raise ValueError("XGBoost UBJ model header is invalid or pickle-like")
    return bytearray(payload), suffix


def load_xgb_model(
    model_path: str,
    expected_sha256: Optional[str] = None,
) -> Optional[Callable[[Any], float]]:
    """
    Load a safe XGBoost JSON/UBJ Booster and return class-1 probability.

    Pickle, joblib, legacy binary, unknown, linked, and unstable inputs are
    rejected before XGBoost receives bytes.  A supplied SHA-256 is checked on
    the exact descriptor-bound bytes passed to XGBoost.
    """
    if not model_path:
        return None
    payload, _suffix = _read_safe_xgb_model(model_path, expected_sha256)
    try:
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(payload)
    except (Exception, RecursionError) as exc:
        raise RuntimeError("XGBoost safe-format model could not be loaded") from exc

    def xgb_proba_fn(X: Any) -> float:
        if pd is not None and hasattr(X, "values"):
            X = X.values
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        d = xgb.DMatrix(arr)
        p = booster.predict(d)
        return float(p[0])

    return xgb_proba_fn
