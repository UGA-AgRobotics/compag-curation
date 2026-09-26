# External CJ YOLO segmentation checkpoint in the r92 pipeline

## Browser workflow in the current R5 source

The simplest route is `python3 start_compag.py`. Complete an XGBoost round in the local browser, then select the intended YOLO mode at the optional YOLO step. The guide finds saved and nearby model packages automatically and remembers the locations after you click **Apply YOLO setting**. Leave the package field blank for automatic discovery; an explicit path is only needed when the package is elsewhere or several different models are available. The folder must contain `MODEL_MANIFEST.json` and `weights/best.pt`; a standalone `.pt` file is insufficient for package verification. Choose the proposal and decision policy for the next inference from the browser options below. Leave the advanced YOLO Python field blank for automatic creation of a separate, hash-pinned Ultralytics 8.4.26 environment; the verified science-GPU environment remains unchanged. Choose **Use the latest model trained from my previous review** for the next photo so the attached setting is used. Later XGBoost retraining carries forward the selected, attested YOLO inference setting. This route uses the supplied checkpoint as a detector and does not retrain YOLO. The command examples below remain available for advanced use.

| Internal `--yolo-mode` value | Paper-aligned browser label | Effect |
| --- | --- | --- |
| `off` | `xgb_recall: YOLO disabled (XGBoost only)` | Canonical paper decision path; AMG proposals and XGBoost-only decisions. |
| `prompt` | `sam2_policy=both (AMG + YOLO prompts), det_policy=xgb` | Adds YOLO-box-prompted SAM2 masks to AMG; decisions remain XGBoost-only. |
| `fusion` | `sam2_policy=auto (AMG), det_policy=hybrid` | AMG proposals with optional XGBoost + YOLO hybrid decisions. |
| `both` | `sam2_policy=both (AMG + YOLO prompts), det_policy=hybrid` | Adds prompted masks to AMG and applies hybrid decisions. |

The external model package `YOLO_latset_versin` supplies a one-class CJ
instance-segmentation checkpoint. COMPAG verifies its package manifest,
checkpoint SHA-256, class map, Ultralytics 8.4.26 runtime, and exact inference
configuration before using it. The optional detector stage runs in a separate
Ultralytics environment and produces a hash-bound box receipt for every prepared
512-pixel tile. The science-GPU stage verifies that receipt without importing
Ultralytics.

This adapter uses the segmentation model's **boxes and confidences**. In
`fusion`, AMG supplies candidates; in `prompt`, YOLO boxes prompt SAM2 and
those masks are combined with AMG. `both` combines prompting and hybrid
decisions. The segmentation checkpoint's own masks are checked for one-to-one
correspondence with boxes, but are not inserted as candidate masks.

The paper's optional `det_policy=hybrid` requires a matched YOLO box with
confidence at least 0.20 and candidate-box IoU at least 0.60. For a valid
match, `w_y = p_yolo / (p_yolo + p_xgb)` (or 0.5 when both scores are zero).
If `p_xgb >= 0.50` and `p_yolo < 0.50`, `w_y = max(w_y, 0.80)`. The fused
score is `p_fused = w_y*p_yolo + (1-w_y)*p_xgb`, and the final decision is
positive when `p_fused >= 0.50`. With YOLO enabled but no valid match,
`det_missing=reject` makes the final decision negative regardless of XGBoost;
the review score `p_ui` still falls back to `p_xgb` for uncertainty ranking.
With YOLO disabled, the decision is XGBoost-only at 0.50.

The external weight is **not** included in the public code, source ZIP, wheel,
sdist, or GitHub Release assets. It remains at a user-selected local path. The
original native r92 model and XGBoost classifier are still required. This
integration does not claim that the external checkpoint's reported evaluation
metrics transfer to the public example cards.

## A new card with the native XGBoost model and YOLO fusion

Set these paths to the local source tree, separate Ultralytics 8.4.26 Python,
science-GPU Python, externally supplied model package, native r92 model ZIP,
assets, and a **new** prepared card project. The producer's optional
Ultralytics Python 3.10 environment can run `tools/yolo_segment_backend.py`;
the main CLI still requires Python 3.12.

```bash
SOURCE=/absolute/path/to/cloned-COMPAG-repository
YOLO_PACKAGE=/absolute/path/to/YOLO_latset_versin
YOLO_PY=/absolute/path/to/ultralytics-8.4.26-python
GPU_PY=/absolute/path/to/compag-science-gpu-python3.12
MODEL=/absolute/path/to/COMPAG_CJ_R92_INFERENCE_SAMPLE_20260924T200000Z.zip
ASSETS=/absolute/path/to/verified-science-assets
CARD=/absolute/path/to/second-original-card-folder
RUN=/absolute/path/to/new-project-workspace
YOLO_SHA=554a6c7a0c33dc14e7565650bf13d27229a4131455e4f75605993ee06e8fa20e
mkdir -p "$RUN"

env -u PYTHONPATH "$GPU_PY" -I -B -m compag_curation r92 init-images \
  --images "$CARD" --model "$MODEL" --output "$RUN/card2_prepared"

"$YOLO_PY" -B "$SOURCE/tools/yolo_segment_backend.py" attest \
  --model-package "$YOLO_PACKAGE" --checkpoint "$YOLO_PACKAGE/weights/best.pt" \
  --sha256 "$YOLO_SHA" --output "$RUN/yolo_segment_attestation.json"
"$YOLO_PY" -B "$SOURCE/tools/yolo_segment_backend.py" predict-boxes \
  --project "$RUN/card2_prepared" --model-package "$YOLO_PACKAGE" \
  --checkpoint "$YOLO_PACKAGE/weights/best.pt" --sha256 "$YOLO_SHA" \
  --output "$RUN/card2_yolo_boxes.json"

env -u PYTHONPATH "$GPU_PY" -I -B -m compag_curation r92 infer-full \
  --project "$RUN/card2_prepared" --model "$MODEL" --asset-root "$ASSETS" \
  --yolo-mode fusion --yolo-checkpoint "$YOLO_PACKAGE/weights/best.pt" \
  --yolo-sha256 "$YOLO_SHA" --yolo-boxes "$RUN/card2_yolo_boxes.json" \
  --checkpoint-dir "$RUN/card2_fusion_checkpoints" \
  --output "$RUN/card2_fusion_inference"
env -u PYTHONPATH "$GPU_PY" -I -B -m compag_curation r92 review \
  --scored "$RUN/card2_fusion_inference/scores/detections.csv" \
  --tiles "$RUN/card2_prepared/tiles" --state "$RUN/card2_fusion_review"
```

Use `--yolo-mode prompt` or `--yolo-mode both` with the same verified box
receipt to enable SAM2 box prompting. Use a distinct checkpoint/output
directory for each mode. Switching a review-page filter between Effective
machine decision and Raw XGBoost changes the display/filter; the inference
receipt fixes which policy made the effective decision.

## Attach the external YOLO model to a completed XGBoost round

After an independently chosen and completed XGBoost round has produced a
verified snapshot and XGBoost bundle, create a new model-set revision. The
`--yolo-external` flag requires the v2 attestation above and marks the model as
an externally supplied segmentation checkpoint; it does not claim local YOLO
training. Use the resulting model set with `infer-model-set --yolo-boxes`.

```bash
env -u PYTHONPATH "$GPU_PY" -I -B -m compag_curation r92 activate-model-set \
  --workspace "$RUN" --round-id project_r1_external_yolo \
  --snapshot "$RUN/project_r1/training_snapshot.json" \
  --xgb-bundle "$RUN/project_r1/xgb_model" --model "$MODEL" \
  --yolo-mode fusion --yolo-external \
  --yolo-checkpoint "$YOLO_PACKAGE/weights/best.pt" --yolo-sha256 "$YOLO_SHA" \
  --yolo-attestation "$RUN/yolo_segment_attestation.json" \
  --output "$RUN/project_r1_external_yolo/model_set"
env -u PYTHONPATH "$GPU_PY" -I -B -m compag_curation r92 verify-model-set \
  --model-set "$RUN/project_r1_external_yolo/model_set"
env -u PYTHONPATH "$GPU_PY" -I -B -m compag_curation r92 infer-model-set \
  --model-set "$RUN/project_r1_external_yolo/model_set" \
  --project "$RUN/card2_prepared" --asset-root "$ASSETS" \
  --yolo-boxes "$RUN/card2_yolo_boxes.json" \
  --checkpoint-dir "$RUN/card2_external_model_set_checkpoints" \
  --output "$RUN/card2_external_model_set_inference"
```

The existing `train-yolo` path trains a one-class **detection** checkpoint from
complete-tile box QA. It does not fine-tune this segmentation checkpoint from
bulk-accepted predictions. Segmentation-mask retraining would require a
separate, exhaustive mask QA contract and training path.
