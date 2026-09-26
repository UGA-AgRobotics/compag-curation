# COMPAG Curation 1.9.4rc9 — paper optional hybrid policy (R5.3)

R5.3 aligns the optional XGBoost + YOLO decision with Section 2.9 and Table S1 of the paper. A candidate has valid YOLO support when a matched box has confidence at least 0.20 and intersection over union at least 0.60. With valid support, the hybrid score is `p_fused = w_y × p_yolo + (1 − w_y) × p_xgb`, where `w_y = p_yolo / (p_yolo + p_xgb)` (or 0.5 when both scores are zero). In the XGBoost-high/YOLO-low case (`p_xgb ≥ 0.50` and `p_yolo < 0.50`), `w_y` is raised to at least 0.80. The final hybrid decision uses threshold 0.50.

The paper's `det_missing=reject` setting now rejects candidates without valid YOLO support, even when their raw XGBoost score is high. Such candidates have no fused score; their review and active-learning score `p_ui` falls back to raw `p_xgb`, with effective threshold 0.50. The output keeps the raw, fused, review, and final-decision fields separate. For new hybrid scores, `select-review` ranks `|p_ui - tau_eff|`, selects only candidates within `al_margin=0.20`, and uses a default maximum of 50 candidates. Previously saved R5.2 fusion receipts retain their original `ignore` policy when reopened.

The browser names the four choices with the paper's decision and SAM2 policy terms: `xgb_recall` with YOLO disabled, `sam2_policy=both` with `det_policy=xgb`, `sam2_policy=auto` with `det_policy=hybrid`, and `sam2_policy=both` with `det_policy=hybrid`. The stored mode values remain `off`, `prompt`, `fusion`, and `both` for existing model-set compatibility. YOLO is disabled in the canonical paper run, and this update does not change that default. YOLO weights remain an optional, externally supplied local asset; the browser does not retrain YOLO.

The application wheel is rebuilt under the existing version `1.9.4rc9`. An installation of the same version needs a one-time force reinstall of the new bundled wheel to receive the R5.3 behavior. Stop the reviewer and local guide first; use the installation folder selected in the guide if it differs from this default:

```bash
SOURCE="$(pwd)"
PREFIX="$HOME/.local/share/compag-r92-rc9-gpu"
( umask 022; env -u PYTHONPATH -u PYTHONHOME "$PREFIX/bin/python" -I -B -m pip install --no-index --no-deps --no-compile --force-reinstall \
    "$SOURCE/bundled-assets/compag_curation-1.9.4rc9-py3-none-any.whl" )
env -u PYTHONPATH -u PYTHONHOME "$PREFIX/bin/python" -I -B -m compag_curation doctor --profile science-gpu
python3 start_compag.py
```

The native r92 model ZIP, ten-photo ZIP, and saved human review decisions are unchanged. This is a local delivery candidate; no remote GitHub publication or biological-performance validation is claimed.
