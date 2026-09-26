# Current reproducibility scope — R5.4 / 1.9.4rc10 (2026-09-26)

This versioned software companion contains retained scientific source, documented
release adaptations and the exact native r92 ZIP in `bundled-assets/`.
The current workflow is described in [README](../README.md),
[R5.4 notes](../PUBLIC_RELEASE_NOTES_RC10_R5_4.md) and the
[controlled-input runbook](../paper_reproduction/CURRENT_REPLAY_EN.md).
Author-owned paper-reproduction Python source is MIT under [LICENSES](../LICENSES.md).
New launcher/Stop/COCO changes are not historical paper-result producers.
Current r92 review exposes all candidates; Top-K is recommendation metadata.
The selected public assets do not supply the entire historical study workspace.

## Historical software contract — retained 1.9.3 context

Everything below is a dated account of the earlier software line and its
recorded experiments, not fresh R5.4 execution evidence or current acceptance.
Earlier restricted-shortlist and bundled/private wording applies only to those
historical interfaces. The 2026-09-24 JR09 code-only exclusion is superseded for
this release by the hash-bound selected-asset scope in [PUBLIC_RELEASE_SCOPE](../PUBLIC_RELEASE_SCOPE.md).
Scientific limitations, including missing historical fit lineage, remain.

### Retained reproducibility record

## Software Contract

Version 1.9.3 retains the hash-closed strict post-r92 sequential one-image
active-learning workflow introduced in v1.8.1 on the tiled Full scientific profile while retaining the
Full/Lite project surfaces, adding experimental Full-image execution, and retaining general cumulative-pool
rounds, and the v1.8.0 genesis-based image-round contract. The transfer path
adds exactly one unseen image, one uncertainty batch, and one fresh full
XGBoost retrain per image-round. Acceptance or publication claims apply only
to the exact 1.9.3
artifacts and evidence that accompany a release; earlier-version acceptance is
historical evidence, not acceptance of this version.
Version 1.8 supports the base and quick surfaces on CPython 3.12 and binds all
three `science-gpu` profiles to exact CPython 3.12.7, Linux x86-64, and an NVIDIA
CUDA device. It pins build tools, GPU scientific archives, PyTorch
`2.13.0+cu132`, torchvision `0.28.0+cu132`, CuPy `14.2.0`, XGBoost `2.1.1`,
the full official SAM2 source commit, CUDA 13.2 Update 2 `nvcc` 13.2.86 and
required development headers, public assets, feature order, configuration,
split/training seeds, and package metadata.

The deterministic FP32 runtime policy sets `CUBLAS_WORKSPACE_CONFIG` and forces
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` before a CUDA context where possible,
enables deterministic Torch algorithms, disables cuDNN benchmarking and TF32,
records the selected device and observable driver,
and requires XGBoost `USE_CUDA`. It fails closed rather than falling back to
CPU. CPU remains a bounded pre/postprocess boundary for image I/O, card/grid
preparation, CSV/provenance work, and operations without a supported GPU kernel.
The v1.2 `science-cpu` closure is retained only for historical provenance and
is unsupported because its Torch line is covered by a security advisory.
Balanced v1 and canonical CPU v2 model configurations/bundles are likewise
historical structural-verification inputs, not executable profiles. The
generated quick demo is the only supported standalone CPU computation.
The proposal method and bundle remain batch 512. Full is immutable
`canonical-xgb-recall-gpu-v1` with Hiera Large and a sealed 32-point CUDA
execution microbatch. Lite is immutable `efficient-xgb-recall-gpu-v1` with
Hiera Tiny and a sealed 16-point CUDA execution microbatch. Stage 20, Stage 60,
and standalone inference use the selected profile's microbatch over the same
prompt lattice; non-M2M transient `low_res_masks` are discarded before
upstream aggregation only when they cannot be consumed. Selected SAM2
confidence evidence is quantized to six decimals only after raw-score
selection. These execution adaptations add neither a public knob nor a CPU
model route.

Full-image is immutable `full-image-multiscale-xgb-recall-gpu-v1`. It keeps one
lossless native processing unit per image, writes no external tiles, and uses
Hiera Large with SAM2's internal overlap crop pyramid: 64 points per side,
configured batch 512, execution microbatch 8, crop layers 2, downscale 1,
overlap `512/1500`, and cap 1000 proposals/image. Its proposal IDs and ROI
feature semantics form a separate domain.

The two-layer/downscale-1 choice is backed only by a bounded engineering
benchmark on `IMG_9475.jpg`: 884 retained proposals, 98.56% reference-point
coverage at IoU 0.25, 90.65% at IoU 0.50, median best IoU 0.875, 144.3 seconds,
and peak CUDA memory of 2.86 GiB allocated / 5.91 GiB reserved. A three-layer
comparison reached 92.09% at IoU 0.50 only with the 1000-proposal cap binding
and took 481 seconds. The +1.44 percentage-point change for more than 3.3 times
the elapsed time was rejected for the initial preset. This does not establish
general accuracy, an optimum, or equivalence.

Stage-20 provenance binds the exact proposal/feature CSV hashes, compact-mask
encoding `bbox-cropped-column-packbits-little-v1`, and component-selection rule
`largest-8-connected-pixel-area-top-left-tie-v1`; training revalidates these
bindings before fitting.

All three profiles install from one wheel with one exact dependency lock, but their
scientific results are not equivalent. Configuration, assets, outputs, fitted
state, model bundles, resume state, and active-learning lineage remain
profile-bound. Lite and Full-image do not inherit the tiled Full canonical
claim; Full-image equivalence is `NOT_CLAIMED`. Tiled r92 and tiled bundles
cannot enter Full-image, which must train and continue within its own profile.

The v1.7 review applications are outside scientific execution and immutable
scientific ledgers. Initial review verifies a run paused after exactly Stage
20. Tiled modes deterministically reconstruct the complete prepared card from
sealed Stage-10 tiles; Full-image displays its sealed native processing unit.
Round review verifies the complete sealed `begin_round`
operation, its exact Stage-60 input, original inference-image inventory, and
verified `compag-curation-model-bundle/v2`, then displays the complete original
image. Each binds external session state to immutable input hashes, records
explicit human set/undo events in a hash chain, and publishes the strict
reviewed CSV without replacement only after complete validation. Web and
native modes share the same decision interface. Mask outlines default on and
bounding rectangles default off; these display choices, zoom, pan, and
selection do not alter immutable inputs or CSV semantics. Human choice and
action order remain inputs, so the review session itself is not a claim of
automatic or cross-reviewer byte reproducibility.

Project-1/CJ Stage-20 review uses the exact hash-closed r92 transfer preset,
fixed threshold 0.50, and uncertainty ordering. Manual corrections remain
explicit; the optional remainder operation is separately confirmed,
reversible, low-weight, and provenance-distinct. This is not fresh canonical
training or a current-dataset performance claim. Cold-start Stage-20 review
has no fitted classifier and therefore requires an explicit decision for every
proposal. Model-assisted round review makes only
the sealed hard-case shortlist selectable; other detections and their
prediction/score/uncertainty/rank stay display context. CJ/Non-CJ,
Target/Non-target, and custom class names are presentation-only aliases for
durable `1`/`0`. The applications never infer a human label, resume a run or
round, or train.

## Sequential Image-Round Reproducibility

An image-round begins with a directory containing exactly one previously
unseen image. `infer --execute` binds the image, SAM2 proposals, four-scale raw
features, parent bundle, and XGBoost scores. `begin-image-round` then applies
one immutable policy: threshold `0.5`, distance `abs(xgb_p-0.5)`, eligibility
margin `0.2`, Top-K `50`, and proposal-ID tie breaking. Selection is therefore
deterministic for the exact inference archive and parent bundle. Human choices
remain experimental inputs and are not automatically reproducible.

After complete review, `resume-image-round` starts a new full XGBoost fit over
genesis Stage-20 rows plus every cumulative reviewed image-round row. The
genesis test groups and PCA stay frozen; new image groups are train-only and
the prototype is recomputed from accumulated outer-train positives. A sealed
cumulative raw-feature archive prevents prior images from being proposed or
feature-extracted again while preserving their exact selected feature rows.

Every `project-rN` bundle binds the immediate parent bundle, round image,
genesis split, accumulated archive, effective training groups, and frozen test
groups. Round-number gaps, repeated images/groups, changed genesis state,
copied review requests, missing archive rows, or cross-profile inputs fail
closed. This guarantees traceable project lineage, not cross-device floating-
point bit parity or biological accuracy.

Version 1.9.3 can combine the bundled `compag-cj-r92` scoring closure with a
separately supplied, exact recovered fit-time feature snapshot. The importer
requires size `582968556` bytes and SHA-256
`1836d704d15f27728e3ba09d7cddf0aea9458049bc66700dc98f14e7111ec29d`,
then seals its own deterministic group-pure transfer split. The private source
and imported baseline are never public release artifacts.

This closes the identity of the transfer starting point but does not recover
the complete historical r1-r92 human-decision, split, and retraining sequence.
Every new bundle therefore states `NOT_R92_REPRODUCTION`, remains named
`project-rN`, and must not be described as manuscript `r93`. Evaluation is
conditional on the frozen published r92 prototype/PCA state; reported PCA
variance is the observed transfer-training projection variance, not an r92
fit-time reconstruction.

The historical local v1.4 installed-wheel synthetic real-backend runs passed on an RTX
4090 in the exact CUDA 13.2/Torch cu132 environment. Lite completed in 5:53.26
with 2,252,164 KiB maximum process RSS, 3,503 MiB sampled maximum total GPU
memory, and 74% sampled maximum utilization. Full completed in 6:36.10 with
3,108,076 KiB maximum process RSS, 4,650 MiB sampled maximum total GPU memory,
and 88% sampled maximum utilization. Both completed pause/resume, bundle
creation, evaluation, and fresh inference on resolved `cuda:0` with
`allow_cpu_fallback=false`.

The locked v1.3 RTX 4090 Full comparison found a small method-512-to-execution-32 hardware
numerical difference: one of 528 masks changed by one pixel, seven SAM2 IoU
values by 1e-6, and two stability scores by approximately 2.65e-4. The new-32
Stage-20 baseline was byte-stable across two runs, and full Stage-60,
fresh-process, and repeat inference outputs were byte-identical. Resource
measurements were 24,018 MiB old Stage-20 peak total, 4,441 MiB final Stage-20
peak total from an 837 MiB baseline, 4,431 MiB on repeat, and 4,913 MiB for the
passing 6:28 Full demo. These are historical v1.3 exact-host reproducibility
evidence, separate from the v1.4 acceptance above. Neither evidence is laptop
acceptance or a cross-device bit-parity, memory, or performance claim. The exact common dependency doctor
has passed on an RTX 1000 Ada 6 GB laptop with driver 580.173.02, but the Lite
end-to-end demo has not yet run there. No preset-selection VRAM threshold or
exact Lite/Full resource claim follows.
The historical clean v1.7.2 wheel, sdist, source ZIP, and online wrapper passed independent
byte-identical build checks. Their exact artifact hashes belong only in the
adjacent sidecars and release evidence, not in this neutral source document.

GitHub-hosted CPU runners exercise static and mocked GPU contracts only. They
do not establish a real GPU acceptance or performance claim; that requires the
exact locked environment on an NVIDIA host.

## Quick Demonstration

The quick profile generates pixels locally and exercises the real balanced
feature and review-exchange code without SAM2, XGBoost, or a model bundle. It
proves a lightweight installed-wheel workflow, not the real backend.

## Public Real Backend

The required real demos are classified `SYNTHETIC_REAL_BACKEND_DEMO`. Full
uses public Hiera Large and ResNet50 assets; Lite uses public Hiera Tiny and
the same ResNet50 contract. Both use generated images and the production
pause/import/resume state machine, canonical group-aware XGBoost training,
bundle sealing, fresh-process inference, and reporting. Their deterministic
pre-reviewed labels are explicitly non-human fixture data. Both required local
acceptance runs passed as recorded above; this is functional evidence, never
biological-performance evidence.

## Private Copied-Data Validation

Private acceptance, when performed for a release candidate, uses an
OS-read-only bounded copy and publishes only aggregate, redacted findings. A
successful execution can prove schema, stage, bundle, and fresh-inference
interoperability for the exact tested candidate. It does not by itself prove
proposal-distribution equivalence, accuracy parity, fitted-model parity, or
paper-result reproduction. Original paths, images, annotations, review rows,
predictions, and model outputs remain private.

## Paper Results

Original research data, fit-time split/snapshots, training rows, private
predictions, and most fitted study state are not bundled. The narrow exception
is the safe Project-1/CJ r92 transfer-assist closure documented above; it does
not make the unavailable historical training state or results reproducible.
The Full canonical profile also makes the recovered manuscript algorithm
available for newly supplied user data. The later
YOLO-assisted/multiscale/hybrid deployed route is separate and is not claimed
as the canonical paper method. Lite is an efficient alternative, not a
canonical-equivalence route. Exact paper-result reproduction is not claimed.
