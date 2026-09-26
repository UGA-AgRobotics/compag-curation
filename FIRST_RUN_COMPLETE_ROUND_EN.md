# Complete laptop round: full-pool r92 review to new project model (rc9)

For a guided installation and first run after cloning GitHub, use [START_HERE_EN.md](START_HERE_EN.md) and `python3 start_compag.py`. The commands below remain available for advanced or scripted use.

Run in a clean Python 3.12 science-gpu environment with the rc9 wheel installed noneditable and the rc4-compatible pinned SAM2/ResNet assets already verified. Run YOLO training in a separate optional environment containing Ultralytics 8.3.221. The native r92 ZIP is included under `bundled-assets/` in the source repository. These commands operate only on local files and never use the desktop.

For a fresh GPU environment, clone this repository or unpack the rc9 R4 public repository ZIP. The matching rc9 wheel, its `.sha256` sidecar, native r92 ZIP, and ten-photo ZIP are included in `bundled-assets/`. The ZIP contains the top-level `COMPAG_R92_RC9_R4_PUBLIC_REPOSITORY` directory. See `README.md` for the verified example-preparation command. `--prefix` must name a new absolute directory. The first invocation prints a plan; the second installs. The dependency lock intentionally retains the accepted 1.9.3 science-GPU dependency identities while the application wheel is rc9.

```bash
SOURCE=/absolute/path/to/cloned-or-unpacked/COMPAG_R92_RC9_R4_PUBLIC_REPOSITORY
WHEEL="$SOURCE/bundled-assets/compag_curation-1.9.4rc10-py3-none-any.whl"
PREFIX=/absolute/path/to/new-compag-rc9-gpu-env
python3 "$SOURCE/install_r92_gpu.py" --wheel "$WHEEL" --prefix "$PREFIX"
python3 "$SOURCE/install_r92_gpu.py" --wheel "$WHEEL" --prefix "$PREFIX" --execute
env -u PYTHONPATH "$PREFIX/bin/python" -I -B -m compag_curation doctor --profile science-gpu
```

In the commands below, `compag-curation` is `$PREFIX/bin/compag-curation` or the same installed entry point in an already verified rc9 science-GPU environment. Clear a host `PYTHONPATH` before invoking the pinned environment.

Use `tools/prepare_r92_public_example.py` as shown in `README.md` to produce `EXAMPLE/model`, `EXAMPLE/cards/<card-id>` for all ten original cards from the exact bundled model and ten-photo ZIPs. Set the paths to your actual inputs. `RUN` must be a new directory you own. `CARD1` and `CARD2` are two of the ten folders of original full photographs; the second folder must contain a different source card. The shown commands are executable CLI commands; angle-bracket placeholders occur only in the separate QA JSON example.

```bash
MODEL=/absolute/path/to/COMPAG_CJ_R92_INFERENCE_SAMPLE_20260924T200000Z.zip
ASSETS=/absolute/path/to/verified-assets
EXAMPLE=/absolute/path/to/prepared-r92-example-inputs
CARD1="$EXAMPLE/cards/IMG_9317"
CARD2="$EXAMPLE/cards/IMG_9319"
RUN=/absolute/path/to/new-local-project-workspace
mkdir -p "$RUN"
compag-curation r92 verify --model "$MODEL"
compag-curation assets fetch --asset-root "$ASSETS" --profile full
compag-curation assets verify --asset-root "$ASSETS" --profile full
compag-curation r92 init-images --images "$CARD1" --model "$MODEL" --output "$RUN/card1_prepared"
compag-curation r92 infer-full --project "$RUN/card1_prepared" --model "$MODEL" --asset-root "$ASSETS" --checkpoint-dir "$RUN/card1_tile_checkpoint" --output "$RUN/card1_inference"
compag-curation r92 select-review --inference "$RUN/card1_inference" --count 50 --output "$RUN/card1_selection.csv" # optional recommendation
compag-curation r92 review --scored "$RUN/card1_inference/scores/detections.csv" --tiles "$RUN/card1_prepared/tiles" --selection "$RUN/card1_selection.csv" --state "$RUN/card1_review"
```

The reviewer opens the entire verified scored pool. Drag with the right or left mouse button to pan the image; right click on the image does not open the browser menu. Left click selects a candidate. An existing rc7 scored result and review state can be reopened with rc9 without rerunning inference or discarding decisions. Stop an older review server before starting the rc9 reviewer on the same state, then open the new local URL. `--selection` binds a recommended Top-50 rank; it does not restrict what you can select. You can omit it. Use independent prediction, model difficulty, shortlist, review status, and human-label filters; Reset shows all candidates, including deleted masks. **Delete mask** removes the selected proposal from the active review pool and later training/COCO output. The scored inference record stays immutable for audit. Undo last event restores the prior decision if deletion was the latest event; you can also select a deleted mask after clearing Hide deleted and make a new decision.

**Preview / accept remaining model predictions** shows the count and model CJ/non-CJ split across **all cards** in the bound scored pool, independent of current filters and Seen. The browser asks for confirmation before committing. Only candidates with no effective decision are included; individual decisions, skipped and deleted masks are unchanged. Saved actions are marked `bulk_accept`, distinct from individually inspected labels, and have the same 1.0 review factor as an ordinary Accept. **Undo last batch** restores the exact earlier event log while no other decision has been made after that batch. The browser writes ordered decisions to `card1_review/review_labels.csv`; closing it does not finalize a batch. Reopen with the same command. A fresh fit needs at least one eligible CJ and non-CJ. Review the bulk labels carefully before training: they reproduce model predictions and do not provide independent ground truth.

If full-card inference is interrupted, rerun the same `infer-full` command with the **same** `--checkpoint-dir` and a **new** `--output` directory. Completed tiles are reused only when project, model, assets, YOLO configuration, and code identity match. Each tile is checkpointed atomically; `PROGRESS.json` records the active tile, completed count, and failure reason when the process can catch it. A killed process leaves its last active tile marked unfinished. When upgrading from rc6 R3, use a **new checkpoint directory**: the 12 old tile checkpoints cannot be mixed with the later inference code. Keep the rc6 R3 directory as an audit record; run all 40 tiles again when upgrading from rc6 R3. Existing rc7/rc8 OFF-mode checkpoint tiles keep the same inference code identity in rc9. YOLO-enabled runs use distinct mode and box-receipt identities. Never mark a partial tile set complete. The rc9 receipt reports `polygon_bbox_fallback_count` by tile and in total. This count affects display geometry only; canonical masks, features, scoring and candidate membership remain independent of it.

Preview exactly what will enter training, then complete the reviewed-decision batch. Every effective labelled decision, including one outside Top-50 or a confirmed bulk acceptance, enters the cumulative snapshot. Unreviewed, skipped and deleted candidates are excluded. `plan-review` reports deleted and bulk-accepted counts separately. `--scope all` instead requires a terminal event for every scored candidate. `--scope locked --selection FILE` is only for a deliberate restricted batch. Excluding a candidate already committed in a parent snapshot requires a new output revision and `--correction-revision`.

```bash
compag-curation r92 plan-review --state "$RUN/card1_review" --scope reviewed
compag-curation r92 complete-xgb-round --workspace "$RUN" --round-id project_r1 --state "$RUN/card1_review" --inference "$RUN/card1_inference" --model "$MODEL" --scope reviewed --output "$RUN/project_r1"
```

It writes `training_snapshot.json`, `xgb_model/`, `model_set/`, and durable `ROUND_STATE.json` under `project_r1`. A failed stage leaves completed earlier stage files for inspection and does not activate a partial model set. Use a new output revision after changing review or training inputs.

For original-image COCO of **committed reviewed candidates**, preserving mapped full masks, use:

```bash
compag-curation r92 export-original-coco --snapshot "$RUN/project_r1/training_snapshot.json" --inference "$RUN/card1_inference" --output "$RUN/card1_selected_original_coco.json"
```

This is not exhaustive whole-image ground truth. Overlapping masks that appear to be the same physical object require explicit QA rather than silent duplication.

Prepare the next full card and infer with the newly fitted, selected model set. The next review begins from the resulting full-card scores:

```bash
compag-curation r92 verify-model-set --model-set "$RUN/project_r1/model_set"
compag-curation r92 init-images --images "$CARD2" --model "$MODEL" --output "$RUN/card2_prepared"
compag-curation r92 infer-model-set --model-set "$RUN/project_r1/model_set" --project "$RUN/card2_prepared" --asset-root "$ASSETS" --checkpoint-dir "$RUN/card2_tile_checkpoint" --output "$RUN/card2_inference"
compag-curation r92 select-review --inference "$RUN/card2_inference" --count 50 --output "$RUN/card2_selection.csv" # optional recommendation
compag-curation r92 review --scored "$RUN/card2_inference/scores/detections.csv" --tiles "$RUN/card2_prepared/tiles" --selection "$RUN/card2_selection.csv" --state "$RUN/card2_review"
```

For another XGBoost round, use `complete-xgb-round` with `--parent-snapshot "$RUN/project_r1/training_snapshot.json"` and `--parent-model-set "$RUN/project_r1/model_set"`, a new round ID and output directory. Existing model-set paths remain valid for explicit older-model selection; `LATEST_MODEL_SET.json` changes only after complete verification.

## Optional YOLO branch

First perform complete target QA on inspected tiles and save the JSON contract described in `OPTIONAL_YOLO_TRAINING_AND_INFERENCE_EN.md`. Merely reviewing the AL shortlist is insufficient. The QA file must contain independent train and validation source cards.

Create a separate Python 3.12 detector environment if you do not already have one. The wheel's `yolo-local` extra pins Ultralytics 8.3.221; the isolated installer resolves that backend's own CPU dependencies. `PY312` must point to a working Python 3.12 executable. The science-GPU environment remains separate.

```bash
PY312=/absolute/path/to/python3.12
"$PY312" -m venv "$RUN/yolo_env"
"$RUN/yolo_env/bin/python" -m pip install "$WHEEL[yolo-local]"
"$RUN/yolo_env/bin/python" -m pip check
```

```bash
QA=/absolute/path/to/completed-tile-qa.json
YOLO_CARDS=/absolute/path/to/at-least-two-reviewed-source-cards
INITIAL_YOLO=/absolute/path/to/one-class-CJ-checkpoint-or-yaml
YOLO_PYTHON="$RUN/yolo_env/bin/python"
compag-curation r92 init-images --images "$YOLO_CARDS" --model "$MODEL" --output "$RUN/yolo_prepared"
compag-curation r92 prepare-yolo --project "$RUN/yolo_prepared" --qa "$QA" --output "$RUN/yolo_dataset_r1"
"$YOLO_PYTHON" -m compag_curation r92 train-yolo --dataset "$RUN/yolo_dataset_r1" --initial-weights "$INITIAL_YOLO" --output "$RUN/yolo_fit_r1" --epochs 60 --batch 16 --device cpu --imgsz 512
```

Use the exact `checkpoint_sha256` and `checkpoint` from `yolo_fit_r1/YOLO_PROJECT_MODEL.json`. Set these two local variables to those exact values, then run the optional detector stage and activate a new combined model set. The combined model set can reuse the verified XGBoost snapshot without retraining XGBoost:

```bash
YOLO_CHECKPOINT=/absolute/path/from/YOLO_PROJECT_MODEL.json
YOLO_SHA256=the-exact-checkpoint_sha256-from-YOLO_PROJECT_MODEL.json
"$YOLO_PYTHON" -m compag_curation r92 yolo-predict-boxes --project "$RUN/card2_prepared" --checkpoint "$YOLO_CHECKPOINT" --sha256 "$YOLO_SHA256" --output "$RUN/card2_yolo_boxes.json"
compag-curation r92 activate-model-set --workspace "$RUN" --round-id project_r1_both --snapshot "$RUN/project_r1/training_snapshot.json" --xgb-bundle "$RUN/project_r1/xgb_model" --model "$MODEL" --output "$RUN/project_r1_both" --parent-model-set "$RUN/project_r1/model_set" --yolo-mode both --yolo-checkpoint "$YOLO_CHECKPOINT" --yolo-sha256 "$YOLO_SHA256" --yolo-training-record "$RUN/yolo_fit_r1/YOLO_PROJECT_MODEL.json"
compag-curation r92 infer-model-set --model-set "$RUN/project_r1_both" --project "$RUN/card2_prepared" --asset-root "$ASSETS" --yolo-boxes "$RUN/card2_yolo_boxes.json" --checkpoint-dir "$RUN/card2_both_tile_checkpoint" --output "$RUN/card2_inference_both"
```

The same `activate-model-set` command accepts `--yolo-mode prompt` or `fusion`. A retained older verified detector instead needs `--retained-from-round`, `--retained-model-set` pointing to the verified earlier set that owns the checkpoint, and a `verify-yolo-checkpoint` attestation. The selected checkpoint hash must match that earlier set. `train-yolo` by itself does not switch inference on. When YOLO is OFF, no detector install, checkpoint, or box receipt is required.

To preserve work in an old 50-row rc5 session, run `compag-curation r92 expand-review --old-state OLD_STATE --new-state NEW_STATE`, then reopen `NEW_STATE` with the same scored and tile paths. The old event log stays byte-for-byte unchanged, and the new state grants access to every scored candidate. Verify the migration receipt before continuing. A finalized prior round needs explicit correction lineage for changed committed labels.

All-candidate review access does not certify that every physical CJ target has a complete, correct candidate mask. Keep the separate YOLO target-completeness and geometry QA route. Software-test labels, browser checks, and synthetic YOLO QA are separate from genuine human decisions and scientific evaluation.

## External one-class segmentation YOLO

The separately supplied YOLO 8.4.26 segment checkpoint can be used for prompt, fusion, or both inference without bundling its weights. See [EXTERNAL_YOLO_SEGMENT_INTEGRATION_EN.md](EXTERNAL_YOLO_SEGMENT_INTEGRATION_EN.md). This is an inference adapter; `train-yolo` remains a complete-tile detection training path.
