"""Trusted legacy classifier import and the pickle-free runtime bundle.

Only_codes saved each round model as ``joblib.dump({"pipeline": ImbPipeline(
[imp, smote, clf]), "features": [...], "threshold": float, "meta": {...}})``.
Loading such a file executes arbitrary pickle code, so the application never
does it at runtime.  Instead, the owner converts a *trusted* file once, locally,
after pinning its SHA-256:

    compag-curation only-codes import-model --model OLD.pkl \
        --trust-legacy-pickle <sha256> --output BUNDLE_DIR

The conversion extracts the exact fitted state into a safe bundle:

* ``imputer_statistics.npy`` (``allow_pickle=False``) with the fitted median
  statistics and dtype of the ``SimpleImputer``;
* ``classifier.ubj`` (XGBoost UBJSON model) plus ``booster_config.json`` (the
  booster learner/generic configuration, including the training ``device``)
  and the sklearn wrapper parameters, so prediction runs through the same
  ``XGBClassifier.predict_proba`` code path;
* ``schema.json`` with the ordered feature list, threshold and sanitized meta.

The bundle is accepted only after the reconstructed predictor reproduces the
original pipeline's probabilities bit-for-bit on verification rows (single-row
DataFrames exactly as the legacy inference calls it, plus batch and NaN rows).
A failed verification is a blocker; it never falls back to retraining.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

BUNDLE_SCHEMA = "compag-only-codes-legacy-model-bundle/v1"
BUNDLE_MANIFEST = "LEGACY_MODEL_BUNDLE.json"
_MAX_PICKLE_BYTES = 2 * 1024 * 1024 * 1024


class LegacyModelError(RuntimeError):
    """Raised when a legacy model cannot be imported or verified exactly."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite_float__": repr(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# ---------------------------------------------------------------------------
# Pickle-free replica of the fitted SimpleImputer(strategy="median").transform
# ---------------------------------------------------------------------------
_FLOAT_DTYPES = (np.dtype(np.float64), np.dtype(np.float32), np.dtype(np.float16))


class LegacyMedianImputer:
    """Replicates ``sklearn.impute.SimpleImputer.transform`` for the fitted
    legacy configuration (dense input, ``missing_values=nan``, median strategy,
    ``add_indicator=False``, ``keep_empty_features=False``, ``copy=True``).
    Equality with sklearn is verified during import."""

    def __init__(self, statistics: np.ndarray, feature_names: Sequence[str] | None):
        self.statistics_ = np.asarray(statistics)
        self.feature_names_in_ = None if feature_names is None else [str(x) for x in feature_names]

    def transform(self, X: Any) -> np.ndarray:
        if hasattr(X, "columns"):
            if self.feature_names_in_ is not None and [str(c) for c in X.columns] != self.feature_names_in_:
                raise LegacyModelError("feature names differ from the fitted imputer")
            arr = X.to_numpy()
        else:
            arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.dtype not in _FLOAT_DTYPES:
            arr = arr.astype(np.float64)
        arr = np.array(arr, copy=True)
        stats = self.statistics_
        if arr.shape[1] != stats.shape[0]:
            raise LegacyModelError("X has %d features per sample, expected %d" % (arr.shape[1], stats.shape[0]))
        missing_mask = np.isnan(arr)
        invalid_mask = np.isnan(stats)
        valid_statistics = stats[~invalid_mask]
        valid_idx = np.flatnonzero(~invalid_mask)
        if invalid_mask.any():
            arr = arr[:, valid_idx]
            mask = missing_mask[:, valid_idx]
        else:
            mask = missing_mask
        n_missing = np.sum(mask, axis=0)
        values = np.repeat(valid_statistics, n_missing)
        coordinates = np.where(mask.transpose())[::-1]
        arr[coordinates] = values
        return arr


@dataclass
class LegacyClassifier:
    """Pickle-free predictor equivalent to the legacy ``ImbPipeline``."""

    features: list[str] | None
    threshold: float | None
    meta: dict[str, Any]
    imputer: LegacyMedianImputer | None
    classifier: Any  # xgboost.XGBClassifier reconstructed from UBJ + config
    manifest: dict[str, Any]

    def predict_proba(self, X: Any) -> np.ndarray:
        Xt = self.imputer.transform(X) if self.imputer is not None else X
        return self.classifier.predict_proba(Xt)

    def proba_fn(self) -> Callable[[Any], float]:
        """Return the exact equivalent of Only_codes ``load_xgb_model`` closure."""

        from .legacy.xgb_utils import _to_dataframe

        feats = self.features
        clf = self.classifier

        def xgb_proba_fn(X: Any) -> float:
            X_use = _to_dataframe(X, feats)
            p = np.asarray(self.predict_proba(X_use))
            if p.ndim == 1:
                return float(p[0])
            idx = 1
            classes = getattr(clf, "classes_", None)
            if classes is not None and len(classes) == p.shape[1]:
                try:
                    ids = np.where(np.asarray(classes) == 1)[0]
                    if ids.size:
                        idx = int(ids[0])
                except Exception:
                    pass
            return float(p[0, idx])

        return xgb_proba_fn


def _decode_params(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__nonfinite_float__"}:
            return float(value["__nonfinite_float__"])
        return {k: _decode_params(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_params(v) for v in value]
    return value


def _regular_file(path: Path, max_bytes: int) -> os.stat_result:
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise LegacyModelError(f"{path} must be a regular non-symlink file")
    if info.st_size <= 0 or info.st_size > max_bytes:
        raise LegacyModelError(f"{path} size is outside the supported bound")
    return info


def _unwrap(obj: Any) -> tuple[Any, list[str] | None, float | None, dict[str, Any]]:
    if isinstance(obj, dict):
        mdl = obj.get("pipeline", obj)
        feats = obj.get("features", None)
        feats = [str(c) for c in feats] if isinstance(feats, (list, tuple)) else None
        thr = obj.get("threshold", None)
        meta = obj.get("meta", {}) if isinstance(obj.get("meta", {}), dict) else {}
        return mdl, feats, (float(thr) if thr is not None else None), meta
    return obj, None, None, {}


def _split_pipeline(mdl: Any) -> tuple[Any, Any, Any, list[str]]:
    from sklearn.impute import SimpleImputer

    steps = list(getattr(mdl, "steps", []) or [])
    names = [str(n) for n, _ in steps]
    if not steps:
        return None, None, mdl, names
    imp = None
    sampler = None
    for name, step in steps[:-1]:
        if isinstance(step, SimpleImputer):
            if imp is not None:
                raise LegacyModelError("more than one imputer step is not supported")
            imp = step
        elif step == "passthrough" or step is None:
            continue
        elif hasattr(step, "fit_resample"):
            sampler = step  # skipped at predict time by imblearn
        else:
            raise LegacyModelError(f"unsupported pipeline step {name!r}: {type(step).__name__}")
    return imp, sampler, steps[-1][1], names


def _check_imputer(imp: Any) -> None:
    if imp is None:
        return
    if str(getattr(imp, "strategy", "")) != "median":
        raise LegacyModelError("only the legacy median imputer is supported")
    mv = getattr(imp, "missing_values", None)
    if not (isinstance(mv, float) and math.isnan(mv)):
        raise LegacyModelError("only missing_values=nan is supported")
    if bool(getattr(imp, "add_indicator", False)) or bool(getattr(imp, "keep_empty_features", False)):
        raise LegacyModelError("imputer indicator/keep_empty_features are not supported")


def load_safe_bundle(bundle_dir: Path | str, *, verify_hashes: bool = True) -> LegacyClassifier:
    """Load a converted bundle without any pickle deserialization."""

    import xgboost as xgb

    root = Path(bundle_dir)
    manifest = json.loads((root / BUNDLE_MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise LegacyModelError("unsupported legacy model bundle schema")
    if manifest.get("verification", {}).get("status") != "PASS_EXACT":
        raise LegacyModelError("legacy model bundle was not verified bit-exactly")
    if verify_hashes:
        for rel, digest in manifest["files"].items():
            path = root / rel
            _regular_file(path, 4 * 1024 * 1024 * 1024)
            if _sha256_file(path) != digest:
                raise LegacyModelError(f"bundle member hash mismatch: {rel}")
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    imputer = None
    if manifest.get("imputer") is not None:
        stats = np.load(root / "imputer_statistics.npy", allow_pickle=False)
        imputer = LegacyMedianImputer(stats, manifest["imputer"].get("feature_names_in"))
    params = _decode_params(dict(manifest["classifier"]["params"]))
    clf = xgb.XGBClassifier(**params)
    clf.load_model(str(root / "classifier.ubj"))
    config = (root / "booster_config.json").read_text(encoding="utf-8")
    clf.get_booster().load_config(config)
    # XGBClassifier.fit() sets n_classes_; load_model() cannot recover it under
    # xgboost 2.1.x + scikit-learn >= 1.6 (is_classifier tag lookup), so the
    # recorded value of the original fitted object is restored explicitly.
    n_classes = manifest["classifier"].get("n_classes")
    if n_classes is not None:
        clf.n_classes_ = int(n_classes)
    return LegacyClassifier(
        features=schema.get("features"),
        threshold=schema.get("threshold"),
        meta=schema.get("meta", {}),
        imputer=imputer,
        classifier=clf,
        manifest=manifest,
    )


def convert_trusted_legacy_pickle(
    model_path: Path | str,
    output_dir: Path | str,
    *,
    trusted_sha256: str,
    verification_rows: Any = None,
    max_single_rows: int = 2000,
) -> dict[str, Any]:
    """Convert one explicitly trusted legacy joblib model into a safe bundle.

    ``verification_rows`` is a DataFrame of real legacy feature rows (for
    example model-schema columns from ``features_train.csv`` or the ``xf_*``
    columns of a legacy ``detections.csv``).  Synthetic NaN rows are added to
    exercise the imputer.  Every row is predicted by the original pipeline and
    by the reconstructed bundle; any bit difference fails the import.
    """

    import joblib  # local, trusted conversion path only
    import pandas as pd
    import sklearn
    import xgboost as xgb

    src = Path(model_path)
    out = Path(output_dir)
    if not (isinstance(trusted_sha256, str) and len(trusted_sha256) == 64
            and all(c in "0123456789abcdef" for c in trusted_sha256)):
        raise LegacyModelError("--trust-legacy-pickle requires the lowercase SHA-256 of the file")
    _regular_file(src, _MAX_PICKLE_BYTES)
    observed = _sha256_file(src)
    if observed != trusted_sha256:
        raise LegacyModelError(f"legacy model SHA-256 {observed} differs from the trusted value")
    if out.exists():
        raise LegacyModelError(f"output bundle directory already exists: {out}")

    obj = joblib.load(str(src))
    source = {"path": str(src), "sha256": observed, "size_bytes": src.stat().st_size,
              "trust": "EXPLICIT_OWNER_SHA256_PIN", "format": "joblib-pickle"}
    return bundle_from_object(obj, out, source=source, verification_rows=verification_rows,
                              max_single_rows=max_single_rows)


def bundle_from_object(obj: Any, output_dir: Path | str, *, source: dict[str, Any],
                       verification_rows: Any = None, max_single_rows: int = 2000) -> dict[str, Any]:
    """Build and exactly verify a safe bundle from an in-memory legacy model object
    (used for trusted imports and for models trained by the compatibility workflow)."""

    import pandas as pd
    import sklearn
    import xgboost as xgb

    out = Path(output_dir)
    if out.exists():
        raise LegacyModelError(f"output bundle directory already exists: {out}")
    mdl, feats, thr, meta = _unwrap(obj)
    imp, sampler, clf, step_names = _split_pipeline(mdl)
    if not isinstance(clf, xgb.XGBClassifier):
        raise LegacyModelError(f"final estimator must be XGBClassifier, found {type(clf).__name__}")
    _check_imputer(imp)
    if feats is None:
        raise LegacyModelError("legacy model does not record its ordered feature list")
    if imp is not None and hasattr(imp, "feature_names_in_"):
        if [str(c) for c in imp.feature_names_in_] != feats:
            raise LegacyModelError("model feature list differs from the imputer's fitted feature names")

    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        raise LegacyModelError(f"stale partial bundle exists: {tmp}")
    tmp.mkdir(parents=True)
    files: dict[str, str] = {}
    imputer_record = None
    if imp is not None:
        stats = np.asarray(imp.statistics_)
        np.save(tmp / "imputer_statistics.npy", stats, allow_pickle=False)
        imputer_record = {
            "class": type(imp).__module__ + "." + type(imp).__name__,
            "strategy": "median",
            "missing_values": "nan",
            "statistics_dtype": str(stats.dtype),
            "n_features_in": int(getattr(imp, "n_features_in_", stats.shape[0])),
            "feature_names_in": [str(c) for c in getattr(imp, "feature_names_in_", feats)],
            "fit_dtype": str(getattr(imp, "_fit_dtype", "")),
            "copy": bool(getattr(imp, "copy", True)),
        }
    clf.save_model(str(tmp / "classifier.ubj"))
    (tmp / "booster_config.json").write_text(clf.get_booster().save_config(), encoding="utf-8")
    params = _jsonable(clf.get_params())
    schema = {"features": feats, "threshold": thr, "meta": _jsonable(meta)}
    (tmp / "schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    for rel in ("classifier.ubj", "booster_config.json", "schema.json") + (("imputer_statistics.npy",) if imp is not None else ()):
        files[rel] = _sha256_file(tmp / rel)
    manifest: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "source": source,
        "pipeline_steps": step_names,
        "sampler": None if sampler is None else {
            "class": type(sampler).__module__ + "." + type(sampler).__name__,
            "params": _jsonable(sampler.get_params()),
            "note": "samplers are skipped by imblearn Pipeline at predict time",
        },
        "imputer": imputer_record,
        "classifier": {
            "class": "xgboost.XGBClassifier",
            "params": params,
            "classes": _jsonable(getattr(clf, "classes_", None)),
            "n_classes": _jsonable(getattr(clf, "n_classes_", None)),
            "n_features_in": _jsonable(getattr(clf, "n_features_in_", None)),
        },
        "versions": {"xgboost": xgb.__version__, "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "files": files,
    }
    try:
        import imblearn
        manifest["versions"]["imblearn"] = imblearn.__version__
    except Exception:
        pass
    (tmp / BUNDLE_MANIFEST).write_text(json.dumps({**manifest, "verification": {"status": "PENDING"}},
                                                  indent=2, sort_keys=True), encoding="utf-8")

    # ---------------- exact verification ----------------
    import warnings

    from .legacy.xgb_utils import _final_estimator, _to_dataframe  # noqa: F401

    def original_single(row: dict) -> float:
        X_df = pd.DataFrame([row], columns=feats)
        X_use = _to_dataframe(X_df, feats)
        p = np.asarray(mdl.predict_proba(X_use))
        idx = 1
        classes = getattr(_final_estimator(mdl), "classes_", None)
        if classes is not None and len(classes) == p.shape[1]:
            ids = np.where(np.asarray(classes) == 1)[0]
            if ids.size:
                idx = int(ids[0])
        return float(p[0, idx])

    tmp_manifest = {**manifest, "verification": {"status": "PASS_EXACT"}}
    (tmp / BUNDLE_MANIFEST).write_text(json.dumps(tmp_manifest, indent=2, sort_keys=True), encoding="utf-8")
    safe = load_safe_bundle(tmp)
    safe_fn = safe.proba_fn()

    frames = []
    if verification_rows is not None:
        vr = pd.DataFrame(verification_rows).reindex(columns=feats)
        frames.append(vr.apply(pd.to_numeric, errors="coerce").astype(np.float64))
    rng = np.random.RandomState(20260922)
    base = frames[0].head(64).copy() if frames else pd.DataFrame(np.zeros((8, len(feats))), columns=feats)
    nan_rows = base.copy()
    mask = rng.rand(*nan_rows.shape) < 0.15
    nan_rows = nan_rows.mask(mask)
    frames.append(nan_rows.astype(np.float64))
    allrows = pd.concat(frames, ignore_index=True)

    n_single = 0
    n_batch = 0
    max_abs = 0.0
    mismatches = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # batch equality (full matrix)
        p_orig = np.asarray(mdl.predict_proba(allrows))[:, 1]
        p_safe = np.asarray(safe.predict_proba(allrows))[:, 1]
        n_batch = int(len(allrows))
        if p_orig.dtype != p_safe.dtype or p_orig.shape != p_safe.shape:
            raise LegacyModelError("batch prediction dtype/shape mismatch")
        diff = np.abs(p_orig.astype(np.float64) - p_safe.astype(np.float64))
        max_abs = float(np.nanmax(diff)) if diff.size else 0.0
        bad = np.flatnonzero(p_orig.view(np.uint32) != p_safe.view(np.uint32)) if p_orig.dtype == np.float32 \
            else np.flatnonzero(p_orig != p_safe)
        for i in bad[:20]:
            mismatches.append({"kind": "batch", "row": int(i), "orig": float(p_orig[i]), "safe": float(p_safe[i])})
        # single-row equality through the exact legacy call shape
        sel = list(range(min(max_single_rows, len(allrows))))
        for i in sel:
            row = {k: float(v) for k, v in allrows.iloc[i].items()}
            a = original_single(row)
            b = safe_fn(pd.DataFrame([row], columns=feats))
            n_single += 1
            if not (a == b or (math.isnan(a) and math.isnan(b))):
                mismatches.append({"kind": "single", "row": int(i), "orig": a, "safe": b})
                max_abs = max(max_abs, abs(a - b))
    status = "PASS_EXACT" if not mismatches else "FAIL"
    manifest["verification"] = {
        "status": status,
        "batch_rows": n_batch,
        "single_rows": n_single,
        "nan_rows": int(len(nan_rows)),
        "max_abs_diff": max_abs,
        "mismatches": mismatches,
        "comparison": "IEEE-754 bitwise equality of predict_proba[:,1]",
    }
    (tmp / BUNDLE_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if status != "PASS_EXACT":
        raise LegacyModelError(f"safe bundle does not reproduce the legacy model exactly: {mismatches[:3]}")
    tmp.rename(out)
    return manifest
