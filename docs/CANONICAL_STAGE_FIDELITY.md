> **JR09 public-code candidate boundary (2026-09-24).** The fitted Project-1/CJ r92 model closure is research material and is **not included** in this public-code candidate. Any inherited statement below calling that preset "bundled" or "public" describes the frozen private 1.9.3 build and is superseded for this candidate. The r92-assisted lane requires the separately controlled reviewer asset. Application/scientific algorithms in `src/` are unchanged. This is a local candidate pending repository confirmation and asset-specific rights review, not a deployed release.

# Canonical stage fidelity

The eight-stage map and adjudication preserve the historical CPU/manuscript
method-fidelity evidence for `canonical-xgb-recall-cpu-v1`. That evidence is
mapped stage by stage in
[`manifests/CANONICAL_STAGE_FIDELITY_MAP.csv`](../manifests/CANONICAL_STAGE_FIDELITY_MAP.csv).
The map names the exact public symbols, immutable defaults, input and output
schemas, invariants, sanitized authority identifiers, known adaptations, and
hash-bound comparison evidence for stages 10, 20, 30, 40, 50, 60, 61, and 70.
The sanitized adjudication is
[`manifests/CANONICAL_STAGE_FIDELITY_ADJUDICATION.json`](../manifests/CANONICAL_STAGE_FIDELITY_ADJUDICATION.json).

All eight historical CPU rows are closed as exact
method/default/invariant matches or as enumerated bounded adaptations. The
aggregate result is
`PASS_METHOD_EXACT_WITH_DISCLOSED_GPU_NUMERICAL_ADAPTATION`. This means fidelity
of the method, defaults, and public contracts for newly supplied compatible user data,
with the separately measured GPU scheduling numerical deviation disclosed below.
It is not a GPU acceptance result. It also does not establish historical
fitted-model identity, the unavailable historical fit-time split, numerical
parity with private historical outputs, or reproduction of the paper's reported
values.

The v1.3 GPU-first runtime has separate synthetic/runtime contract acceptance.
Those checks verify CUDA profile/device propagation, fail-closed behavior, and
GPU kernel routing; they do not replace, rename, or re-hash the historical CPU
scientific evidence. A release-level GPU claim still requires the separately
documented real-NVIDIA acceptance run.

Version 1.8 retains the locally accepted v1.4 Full and Lite execution
identities. `canonical-xgb-recall-gpu-v1` remains Full;
`efficient-xgb-recall-gpu-v1` remains Lite and uses Hiera Tiny with CUDA prompt
microbatch 16. Lite is not included in the canonical fidelity set and does not
inherit this adjudication or claim scientific, numerical, or canonical
equivalence to Full. Full/Lite bundles, outputs, and resume state are not
interchangeable.

Version 1.9.3 also exposes experimental
`full-image-multiscale-xgb-recall-gpu-v1`. It is deliberately outside this
canonical tiled-fidelity adjudication: it writes no external/persisted tiles,
uses a distinct proposal/ROI-feature domain, and runs Hiera Large with SAM2's
internal crop pyramid at execution microbatch 8. Its scientific and numerical
equivalence to canonical tiled Full is `NOT_CLAIMED`; tiled r92 and tiled
bundles cannot cross into it.

The accepted adaptations are bounded and explicit:

- portable content identities, fail-closed validation, and no-clobber outputs;
- lossless hash-bound proposal overlays and provenance records;
- a strict CSV pause/import/resume scientific boundary, with v1.7 web and
  native applications acting only as human-authored front ends for initial
  Stage-20 and iterative-round reviewed CSVs rather than reproducing
  historical GUI state;
- a deterministic group-pure partition generated for new user data because the
  exact historical fit split was not archived;
- portable UBJ and isolated bundle inference for a newly trained user-owned
  model rather than distribution of the private fitted study model;
- recall-style point coverage rather than one-to-one instance precision; and
- a compact public report rather than manuscript figures and tables.

## Profile boundaries

`canonical-xgb-recall-cpu-v1` is the profile named by the historical fidelity
map and adjudication. Its hashes remain the authority for the eight-stage
method comparison and are not relabeled as GPU evidence.

`canonical-xgb-recall-gpu-v1` follows the same frozen method: warped-card
grid localization; 512-pixel stride-512 far-edge JPEG tiling; SAM 2.1 Hiera-L
AMG-only proposals at scale 1.0; the Ultra feature route with four export crop
scales, ResNet-50, a train-positive prototype, and train-only frozen PCA32; the
exact 93-feature schema; group-pure 80:20 splitting; five-fold group-aware random
search; review-aware weights and train-fold-only Safe-SMOTE; fixed threshold 0.5;
and full-image NMS at IoU 0.5. YOLO is disabled.

The v1.3 execution adaptation moves supported tensor/model work to deterministic
FP32 CUDA while retaining the frozen method and defaults. Its evidence is the
separate GPU synthetic/runtime acceptance described above, not the historical
CPU evidence hashes. SAM2, ResNet50, XGBoost training, and supported inference
must resolve to CUDA; CPU remains only at bounded image I/O, card/grid,
CSV/provenance, and other pre/postprocess boundaries without a supported GPU
kernel. CUDA absence is a hard error, never a CPU fallback.

The method and bundle retain the 512-point AMG batch. Stage 20, Stage 60, and
standalone inference use a sealed 32-point CUDA execution microbatch over the
unchanged 64-point lattice, and non-M2M transient `low_res_masks` are discarded
before upstream aggregation when they cannot be consumed. This is a scheduling
and lifetime adaptation, not a public method parameter. Against the old-512
locked-host baseline, one of 528 masks changed by one pixel, seven SAM2 IoU
values by 1e-6, and two stability scores by approximately 2.65e-4; this small
hardware-numerical deviation is disclosed rather than described as exact output
parity. The new-32 Stage-20 baseline was byte-stable across two runs, and full
Stage-60/fresh/repeat inference was byte-identical. SAM2 score ordering,
deduplication, and the proposal cap remain raw-score operations; only selected
archival confidence fields are quantized to six decimals afterward, and they
are not classifier features.

`public-safe-balanced-v1` remains a historical structural-verification
compatibility profile, not an executable v1.7 model profile. Its overlapping PNG
tiling, Hiera-Tiny asset, 25-feature subset, and simpler training route are
semantic differences, so it is not evidence for the canonical manuscript profile.

## Review-interface fidelity boundary

The v1.7 review applications do not alter proposal generation, feature values,
review semantics, split logic, training, or inference. Initial review opens a
hash-verified run paused after exactly Stage 20. Iterative review opens a
hash-verified `begin_round` operation only when its exact bound Stage-60 root
and original inference-image inventory are also supplied. Both store
append-only human actions outside immutable scientific inputs and finalize the
same strict reviewed-CSV contracts used by their command-line resume
boundaries. Round visuals are a portability/interface adaptation: verified
original-image crops carry the Stage-60 polygon and bbox, not the Stage-20
lossless mask overlay. Project-1/CJ may show verified r92 transfer predictions
and offers only an explicit, reversible low-weight remainder confirmation;
cold start and project-trained rounds never expose that operation. No
application performs a scientific resume or retraining or claims behavioral
identity with a historical notebook GUI.

The strict post-r92 image scheduling introduced in v1.8.1 and retained in
Version 1.9.3 is an operational extension,
not an assertion that the manuscript defined one image as one round. It keeps
the manuscript acquisition rule inside each episode—XGBoost distance to 0.5,
margin 0.2, deterministic Top-K with `K=50`—and performs a fresh cumulative
full retrain after review. When fewer than 50 proposals are eligible, all are
reviewed; zero eligible rows fail closed. Only selected proposals enter the
new-image training archive, and the same image bytes/group cannot be submitted
for a second transfer image-round.

The only accepted historical training-row source is the separately supplied
582,968,556-byte snapshot with SHA-256
`1836d704d15f27728e3ba09d7cddf0aea9458049bc66700dc98f14e7111ec29d`;
it is never publicly bundled. Round 1 uses the verified r92 classifier and
frozen published prototype/PCA state, then fresh-fits `project-r1`; later rounds
use the immediately prior `project-rN`. This lineage is explicitly
`NOT_R92_REPRODUCTION`: no `r93` label is emitted, and no exact historical
round-order, split, decision sequence, or result reproduction is claimed.

The preserved deployed YOLO-prompt, multiscale-AMG, and hybrid-score code and
its private state are a later historical operational profile and remain
excluded from the canonical claim. The public canonical training route is
AMG-only and XGBoost-only, and public users can train their own hash-bound model
bundle. Separately, the narrow Project-1/CJ review exception distributes only
the manuscript-selected XGBoost r92 closure as reviewed UBJ/JSON/NPY transfer
assistance; its original pickle, private training rows, splits, predictions,
and other historical fitted state are not distributed.

## Authority privacy

The public map contains only authority IDs and SHA-256 digests. Absolute historical
paths are private engineering evidence and must never be copied into a public file,
package, report, or runtime output. The private authority registry resolves those
IDs for fidelity work without making the locators distributable.

## Comparison rule

The fidelity gate requires stage execution, invariant checks, and a comparison to
unchanged frozen evidence wherever the comparison is semantically valid. Interface,
serialization, path-safety, and provenance changes may be accepted only when shown
to be non-semantic. Any algorithm, default, or output-contract difference remains
`NOT_READY` until repaired or removed from the supported canonical claim. The
closed map records no material algorithm, default, or output-contract deviation.
Exact historical fit-split, fitted-model identity, private-output parity, and
paper-result parity remain `NOT_CLAIMED` because those are outside the available
and distributable evidence.
