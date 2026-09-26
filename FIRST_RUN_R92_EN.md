> Historical rc4 two-photo guide. For the current rc6 R3 ten-photo package, start with README.md and FIRST_RUN_COMPLETE_ROUND_EN.md. The command paths below describe the older rc4 candidate.

# First run — one complete fixed-r92 public round (1.9.4rc4)

This local candidate processes **every canonical 512×512 tile of every selected warped card**. The stages are explicit: install → prepare → infer-full → review → export → verify-round. The supplied two photos are examples previously used in audits, not a held-out accuracy test. No labels or review decisions are included.

Use Linux x86-64, CPython 3.12 for the CPU installer, Conda for the pinned science GPU environment, an NVIDIA CUDA GPU compatible with the locked stack, at least 15 GiB free for the GPU environment, and network access to the official locked archives. The CPU scorer works independently of the GPU packages. Choose new output paths outside the source and image folders. Keep `PYTHONPATH` unset.

## 1. Extract and install

Extract the rc4 source ZIP and two-photo input ZIP. Keep the rc4 wheel and its `.sha256` file together. Set these paths in a terminal:

```bash
SOURCE=/absolute/path/COMPAG_CURATION_LAPTOP_RUNNABLE_CANDIDATE_R92_RC4_20260924
WHEEL=/absolute/path/compag_curation-1.9.4rc4-py3-none-any.whl
MODEL=/absolute/path/COMPAG_CJ_R92_INFERENCE_SAMPLE_20260924T200000Z.zip
IMAGES=/absolute/path/COMPAG_R92_TWO_PHOTO_INPUTS_RC4_20260924/originals
WORK=/absolute/path/new-r92-round-work
GPU_PREFIX=/absolute/path/secure-new-r92-gpu-environment
PY=python3.12
unset PYTHONPATH PYTHONHOME
mkdir -p "$WORK"
"$PY" "$SOURCE/install_laptop.py" --profile r92-sample --wheel "$WHEEL" --venv "$WORK/cpu"
"$PY" "$SOURCE/install_laptop.py" --profile r92-sample --wheel "$WHEEL" --venv "$WORK/cpu" --execute
CPU="$WORK/cpu/bin/compag-curation"
"$CPU" r92 verify --model "$MODEL"
```

The first installer command prints a plan; the second performs the isolated, hash-pinned CPU installation. If `python3.12` is not on `PATH`, set `PY` to an existing Conda Python 3.12 executable. This does **not** install into that Conda environment.

Create a **separate new** Full-profile GPU environment. The installer prints a plan first and then uses the bundled exact Conda/pip locks, the pinned official SAM2 commit, and the selected wheel. It does not change system drivers or install into the CPU environment:

```bash
"$PY" "$SOURCE/install_r92_gpu.py" --wheel "$WHEEL" --prefix "$GPU_PREFIX"
"$PY" "$SOURCE/install_r92_gpu.py" --wheel "$WHEEL" --prefix "$GPU_PREFIX" --execute
GPU_PY="$GPU_PREFIX/bin/python"
ASSETS="$WORK/verified-assets"
"$CPU" assets fetch --asset-root "$ASSETS" --profile full
```

The GPU installer requires a new prefix whose parent chain is not group/world writable or symlinked. A private user path such as `~/.local/share/compag-r92-rc4-gpu` is suitable when new; `WORK` for projects may be elsewhere. It invokes the `science-gpu` doctor in Python isolated/no-bytecode mode; that check must pass before calling the environment fully attested. Use the same `-I -B -m compag_curation` form for GPU commands. Official Full-profile assets include the SAM2.1 Hiera Large checkpoint/config, ResNet50 IMAGENET1K_V2 weights and notices. Asset hashes are checked during fetch and again at inference. On a 6 GiB GPU, close other GPU workloads before starting.

## 2. Prepare all tiles and infer every tile

For your own project, set `IMAGES` to a directory containing 1–20 regular JPG/PNG files with unique portable stems. Each image must be at least 512×512. The two supplied originals can be used unchanged for the first round check.

```bash
PROJECT="$WORK/project"
INFERENCE="$WORK/inference"
"$CPU" r92 init-images --images "$IMAGES" --model "$MODEL" --output "$PROJECT"
"$GPU_PY" -I -B -m compag_curation r92 infer-full --project "$PROJECT" --model "$MODEL" --asset-root "$ASSETS" --output "$INFERENCE"
```

`init-images` copies originals byte for byte, runs the existing yellow-card warp/grid and far-edge tiler, and creates empty-label original/tiled COCO registries. `PROJECT_MANIFEST.json` lists every original and tile hash. `infer-full` logs tile `i/N` as it runs and refuses a missing or modified tile before loading the model. `FULL_INFERENCE_RECEIPT.json` records expected and processed tile counts, each tile's proposal count, feature/score hashes, model and official asset identities. A tile with zero proposals is still counted as processed. No tile or image is silently omitted.

## 3. Review, export and verify

```bash
STATE="$WORK/review-state"
EXPORT="$WORK/reviewed-export"
"$CPU" r92 review --scored "$INFERENCE/scores/detections.csv" --tiles "$PROJECT/tiles" --state "$STATE"
```

The browser opens the local review UI. Review **every** scored candidate. Reuse the same `review` command and `STATE` directory to resume after a pause. After finishing, stop the local review server and run:

```bash
"$CPU" r92 export --state "$STATE" --orig-coco "$PROJECT/original_coco.json" --tiled-coco "$PROJECT/tiled_coco.json" --output "$EXPORT" --confirm-review-complete
"$CPU" r92 verify-round --project "$PROJECT" --inference "$INFERENCE" --export "$EXPORT" --images "$IMAGES"
```

Export refuses missing individual decisions, duplicate normalized keys, invalid geometry, bad image dimensions or changed project/score receipts. It writes **new tiled COCO annotations** to `reviewed-export/tiled_coco.json`; `original_coco.json` remains an image registry and receives no new original-coordinate polygons. `EXPORT_MANIFEST.json` hashes the output files. `verify-round` checks full tile coverage, candidate accounting, export hashes and preservation of both the project copies and the optional original input folder.

## Scope

The fixed r92 model is unchanged and is used for inference only. A genuine review → prepare → fit → next-model cycle is **not included** in this public round. Software-test review actions are not human labels. These two example cards were previously audited and do not support a fresh accuracy or held-out claim. The candidate has not been published remotely; record model/photo asset terms and a destination before doing so. See `LIMITATIONS_R92_RC4_EN.md`.
