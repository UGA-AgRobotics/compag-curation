# Pipeline Execution

## State Machine

1. `00_input_inventory` seals configuration, normalized configuration, command,
   data, annotation, dependency, asset, and disk-capacity evidence.
2. `10_prepare` decodes inputs with stable hashes. Tiled Full and Lite rectify
   the card/grid and use 512-pixel stride, far-edge alignment, edge padding,
   and JPEG tiles. Full-image retains exactly one lossless native processing
   unit per image and creates no external/persisted tiles.
3. `20_proposals_features` runs profile-bound official SAM2 AMG, extracts the
   selected feature contract, and exports the review request. Full uses Hiera
   Large and Lite uses Hiera Tiny; both tiled modes use scale 1.0 proposals and
   four export-only ResNet50 crop scales. Full-image uses Hiera Large and SAM2's
   internal overlap crop pyramid (`points_per_side=64`, configured batch 512,
   execution microbatch 8, crop layers 2, downscale 1, overlap `512/1500`, cap
   1000 proposals/image), then extracts the four ROI scales in its distinct
   full-image feature-semantics domain.
4. The process returns `PAUSED_FOR_REVIEW` with exit code 3.
5. `30_review_import` validates a separately supplied reviewed copy.
6. `40_group_split` creates an outer tile-balanced, group-pure 80/20
   train/test split; inner group-safe validation exists only inside training.
7. `50_train_bundle` trains deterministic CUDA XGBoost and evaluates untouched
   test candidates. Canonical uses five-fold group search, fold-local
   Safe-SMOTE, group-safe early stopping, and the fixed 0.50 method threshold.
   The stage seals a portable
   profile-specific bundle.
8. `60_evaluate` launches bundle-only inference on new images and optional point
   coverage.
9. `70_report` writes a bounded human-readable and structured report.

## Publication And Resume

Each stage is built in `.staging/<stage>.<uuid>`, validated, fsynced, and
published with Linux no-replace rename. `_SUCCESS.json` binds the run ID and
every member. Failed staging is retained as diagnostics. Completed stages form
a contiguous prefix and cannot be republished or overwritten.

The persistent sibling lock permits one process per run. Resume checks run
ownership, lock identity, event-ledger closure, stage order, exact completed
stage manifests, and current config/input/dependency/asset hashes. Final
completion is atomically published as `FINAL/`.

## Determinism Boundary

Balanced v1 and canonical CPU v2 stage records remain structurally
loadable/verifiable as historical evidence, but this v1.7 state machine does
not create or execute them.

Ordering, geometry-and-mask proposal identities, feature order, split seed,
XGBoost seed/thread, review import, bundle manifest, and NMS tie-breaks are
fixed. All GPU profiles use finite raw SAM2 scores for ordering,
deduplication, and the cap, then quantizes only the selected archival confidence
fields to six decimal places. Those fields are not part of the 93-feature
predictor matrix. Each profile seals its precision and identity policy in
`proposal_config.json`. Before numerical work,
all GPU profiles set `CUBLAS_WORKSPACE_CONFIG` and force
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` before CUDA context creation where possible,
enable deterministic Torch algorithms, disable cuDNN benchmarking, and disable
TF32. They record requested/resolved device, Torch CUDA runtime,
device index/name/capability, observable driver, and XGBoost `USE_CUDA`.
CUDA absence or a CPU resolution fails closed. CPU remains limited to bounded
image decoding, card/grid preparation, I/O, provenance, and other pre/postprocess
work without a supported GPU kernel. That exact runtime policy is sealed in
proposal and dependency evidence. Scientific compatibility remains limited to
the locked `science-gpu` profile on exact CPython 3.12.7, Linux x86-64, and an
NVIDIA CUDA device. Reported metrics are functional fixture outcomes, not
promised biological performance.

Stage 20 retains the frozen 64-points-per-side, batch-512 AMG method and bundle
identity. Stage 20, Stage 60 fresh bundle inference, and standalone inference
evaluate the same configured lattice with a sealed execution-only 32-point
tiled Full, 16-point Lite, or 8-point Full-image CUDA microbatch. For
Full-image, SAM2's crop layer is internal and does not create Stage-10 tile
files.
Non-M2M transient `low_res_masks` are discarded before upstream aggregation
when they cannot be consumed. This bounds VRAM residency; it does not reduce
sampling, expose a configuration knob, or enable CPU fallback.

Every stage, bundle, resume, and active-learning identity stays bound to the
originating immutable Full, Lite, or Full-image profile. Cross-profile reuse
fails closed; Lite and Full-image do not inherit the tiled Full canonical claim,
and Full-image equivalence is `NOT_CLAIMED`.
