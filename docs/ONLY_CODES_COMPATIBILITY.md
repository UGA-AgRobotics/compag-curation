# Only_codes drop-in compatibility workflow (`only-codes-compat-v1`)

Version 1.9.0 added an explicitly selected legacy workflow that accepts the
original Only_codes project files directly (the `Clean` workspace used with
the `COCO_2_features` notebook, `gate_core.py`, `sam2_and_filter.py`,
`sam2_pipeline/`, the shell runner and the Dash reviewer) and reproduces
their computations and output files.  Version 1.9.1 fixes the handling of
explicitly relocated inputs, makes the round's state commit transactional and
separates preset selection and environment readiness from verified output
parity.  Version 1.9.2 restores the reference path contract of each input
reader and keeps explicit relocation under a documented mapped-reader policy
(see `CHANGELOG.md`).

The behavioural authority is the supplied Only_codes source and the effective
runtime arguments of its runner (mode `xgb_recall`), not the manuscript, the
SAM3 pipelines or the canonical Full/Lite/Full-image/r92 defaults. Those
canonical workflows are unchanged.

## What is preserved

* **Inputs, read in place and never rewritten**: tiled/original COCO JSON,
  training tile images, splits, Stage-1 embeddings, prototypes, PCA,
  fold-safe packs, `features_train.csv` (existing rows and resume keys are
  kept, never recomputed), round model directories (`tile_preds.csv`,
  `threshold_*.json`, `best_params_used.json`, `history.csv`), inference
  prototype/PCA (`always_same`), YOLO and SAM2 checkpoints, ResNet50 weights,
  per-image tiles, detections, AL shortlists, `review_labels.csv` and
  `seen.json`.
* **Copy-on-write project state**: every step that used to modify files in
  place writes into `<project>/state/` with the original relative layout.
  Overwritten project files are snapshotted and each transition is appended to
  `TRANSITIONS.jsonl`.
* **Inference**: the vendored original `run_pipeline` (documented plumbing
  patches only) with SAM2.1 Hiera-L (`apply_postprocessing=False`), policy
  `both`, 64 points per side, points-per-batch 512, pred-IoU 0.80, stability
  0.88, crop layers 0, overlap 0.4, AMG scales `[1.0]` with merge IoU 0.75 and
  cap 500, prompt boxes padded by 0.02, YOLO conf 0.20 / IoU 0.60 / imgsz 512 /
  max-det 300 with the BGR-then-RGB retry, hybrid decision with detector
  missing = `reject`, Smart Hybrid rules, and the disagreement-aware AL
  shortlist (top-K 50, margin 0.20). The XGBoost feature order is the
  selected model's recorded schema.
* **Training/export/merge/review**: ports of the notebook cells with the same
  statement order, random streams, weights, augmentation, early stopping,
  full-train refit, threshold rule and writers.

## Known legacy behaviours (kept on purpose, not corrected)

They are listed with identifiers `LB-01` ... `LB-18` in
`compag_curation/only_codes/contract.py`, for example the zero-filled `g_*`
inference predictors (`legacy_gate_input_zero_fill`), the different
`embed_sim` and grid definitions between training export and inference,
augmentation before inner splitting, the one-step-shifted internal
threshold, and the reviewer's own `xgb`/`ignore`/0.40 display policy.

## Trust model for legacy pickles

Round models are `joblib` pickles. They are never loaded by the runtime.
`only-codes import-model` converts one model after the owner pins its
SHA-256 (`--trust-legacy-pickle`), writes a pickle-free bundle (XGBoost UBJ,
booster configuration, imputer statistics as `.npy`, ordered schema) and
accepts it only if the reconstructed predictor reproduces the original
pipeline's probabilities bit for bit. Fold-safe data packs are read with a
NumPy-only restricted unpickler.

## Environment

Exact equality is claimed only in the matched reference environment
(Python 3.12.7, the interpreter and libraries used by the original runner,
including its SAM2 source and `ultralytics`). Other environments are
execution variants and are labelled as such. `--device cpu` or a smaller
`--points-per-batch` are recorded as `NON_EQUIVALENT_EXECUTION_VARIANT`.
Legacy environment variables (`CJ_*`, `DET_*`, `JASSID_*`, `FORCE_XGB_P`, ...)
never change the imported preset; their presence is recorded.

## Commands

The commands need the private workspace and a CUDA GPU; they are not part of
the hosted command matrix.

```console
compag-curation only-codes import --legacy-root OLD_CLEAN --sam2-repo-root SAM2_REPO \
    --torch-home TORCH_HOME --image IMG_NNNN --output PROJECT
compag-curation only-codes import-model --project PROJECT --round-dir rNNN_hybrid \
    --trust-legacy-pickle SHA256_OF_PKL
compag-curation only-codes infer --project PROJECT --image IMG_NNNN --model rNNN_hybrid
compag-curation only-codes review --project PROJECT --image IMG_NNNN
compag-curation only-codes round --project PROJECT --image IMG_NNNN --confirm-review-complete
compag-curation only-codes ui
```

Individual stages are also available: `merge`, `splits`, `pack`,
`features`, `train`, `review-status` and `status`.

## Parity harness

The separate parity kit shipped beside the release (`ONLY_CODES_PARITY_KIT`, script
`only_codes_parity/run_parity_suite.py`) runs the original Only_codes
source as an independent oracle in separate processes (read-only, sandboxed
paths, notifications refused) and compares every output bitwise, reporting
`PASS_EXACT`, `PASS_NUMERIC_WITH_TOLERANCE`, `FAIL`, `BLOCKED_MISSING_INPUT`,
`BLOCKED_ENVIRONMENT` or `NOT_RUN` per stage.
