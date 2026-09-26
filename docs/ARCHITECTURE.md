> **JR09 public-code candidate boundary (2026-09-24).** The fitted Project-1/CJ r92 model closure is research material and is **not included** in this public-code candidate. Any inherited statement below calling that preset "bundled" or "public" describes the frozen private 1.9.3 build and is superseded for this candidate. The r92-assisted lane requires the separately controlled reviewer asset. Application/scientific algorithms in `src/` are unchanged. This is a local candidate pending repository confirmation and asset-specific rights review, not a deployed release.

# Architecture

COMPAG Curation 1.9.3 has two deliberately separate software surfaces. The original
eight-command, 21-variant domain interface remains a dependency-light contract
validator and inert planner. The public-project facade is the supported
review-gated execution path. Both dispatch through one console script, but the
new facade does not modify the legacy registry or variant schemas.

Within the public facade, one source tree, one wheel, and one hash-locked CUDA
dependency environment support three explicit GPU profiles. Tiled Full is
`canonical-xgb-recall-gpu-v1`; Lite is `efficient-xgb-recall-gpu-v1`; the
experimental non-tiled mode is
`full-image-multiscale-xgb-recall-gpu-v1`. Friendly
CLI aliases are resolved only at the input boundary, while immutable profile
IDs remain sealed in configuration and provenance.

## Public Project Layers

- `public_config.py` owns the closed TOML project and bounded input inspection.
- `assets.py` owns explicit reviewed acquisition and byte verification.
- `public_io.py` provides no-follow stable reads, exclusive writes, manifests,
  portable path validation, fsync, and no-replace publication.
- `run_state.py` owns one-writer locking, run identity, atomic stages, events,
  resume, and drift rejection.
- `review/exchange.py` owns the full-hash review CSV boundary.
- The public execution backend connects profile-specific spatial preparation, official SAM2
  AMG, feature extraction, group-aware XGBoost, inference, and point coverage.
- `model_bundle.py` seals and verifies portable UBJ bundles.
- `public_pipeline.py` orchestrates the pause/resume state machine and reports.
- `canonical/active_learning.py`, `active_learning_service.py`, and
  `active_learning_facade.py` own immutable genesis, selection, cumulative
  feature-archive, full-retrain, and lineage contracts for both general-pool
  and sequential image-round active learning.
- `canonical/transfer_baseline.py` performs the exact SHA-256/size-gated
  private r92 snapshot import; `canonical/transfer_inference.py` closes
  one-image proposal/feature/r92 scoring. The private snapshot is never a
  package or public-release resource.
- `quick_demo.py` retains generated-data demonstration implementation as code
  evidence. No quick/GPU/scientific execution route is accepted in this
  model-free packaging candidate.

Heavy libraries are imported only inside explicit execution or decode checks;
package import, help, legacy validation/planning, initialization, and bounded
structural inspection remain inert. The source layout prevents the repository
root from masquerading as an installed package.

## State And Safety

Inputs are stable-read through non-following file descriptors and never
rewritten. New outputs must be disjoint and absent. Each run has a sibling
flock, owner marker, UUID, normalized config, command/environment/input/asset
records, append-only events, ordered stages, and final manifest. Stages publish
only after closure checks using Linux no-replace rename; failures remain as
diagnostics. Resume rechecks every bound identity.

## Scientific Boundary

`public-safe-balanced-v1` and `canonical-xgb-recall-cpu-v1` are retained as
historical structural load/inspect/verify contracts; their project and bundle
writers and model execution routes are closed in v1.7.
Full uses the manuscript-bound Hiera Large AMG-only single-scale proposal
route and a sealed CUDA prompt microbatch of 32. Lite uses Hiera Tiny and a
sealed CUDA prompt microbatch of 16. Both use Ultra and ResNet50/PCA32 features,
train-fitted prototype/PCA state, group-aware XGBoost, and fixed 0.50 inference
threshold. Their common science surface is fail-closed CUDA:
SAM2, ResNet50, XGBoost training, and supported model inference must resolve to
the NVIDIA device, while bounded image I/O, card/grid preparation, provenance,
and other preprocessing/postprocessing may remain on CPU. There is no silent
CPU fallback. YOLO is disabled and is not represented as part of the canonical
scientific run. No result parity with unavailable original data or fitted model
state is claimed.

Full-image uses Hiera Large without external or persisted tiling: Stage 10
retains one lossless native processing unit per image and SAM2 supplies the
internal overlap crop pyramid. Its initial operating point is
`points_per_side=64`, configured scientific batch 512, execution-only
microbatch 8, crop layers 2, downscale 1, overlap `512/1500`, and cap 1000
proposals/image. It has distinct proposal IDs and ROI feature semantics.
Scientific or numerical equivalence with either tiled profile is
`NOT_CLAIMED`.

The method and bundle retain `points_per_batch = 512`, while Stage 20, Stage 60,
and standalone inference use the profile's sealed execution-only CUDA microbatch over
the same 64-point prompt lattice. Non-M2M transient `low_res_masks` are discarded
before upstream aggregation when they cannot be consumed. This memory-lifetime
boundary is internal, not a user configuration field or a CPU fallback.

Full, Lite, and Full-image are separate scientific identities, not
interchangeable quality labels. The selected SAM2 asset, configuration, spatial
mode, bundle closure, output state, resume identity, and active-learning
ancestry stay profile-bound. Lite and Full-image do not inherit the tiled Full
canonical claim. Tiled r92 and tiled bundles cannot cross into Full-image;
Full-image must complete its own review/training and continue only with the
same profile.

## Strict Post-r92 Image-Round State Machine

Version 1.9.3 retains the transfer state machine introduced in v1.8.1 alongside the retained v2
general-pool and v3 genesis-based image-round operations:

```text
exact private r92 transfer snapshot -> verified sealed baseline/split
one unseen image
  -> infer-r92-image: SAM2 proposals + four-scale features + r92 scores
  -> begin-transfer-image-round: abs(p-0.5) <= 0.2, deterministic Top-K 50
  -> explicit human review of every selected row and finalization
  -> resume-transfer-image-round: fresh full XGB fit over baseline + reviewed rows
  -> verified project-r1 bundle
next unseen image
  -> infer with project-r1 -> review -> fresh full fit -> project-r2 -> ...
```

The private source boundary is fixed at 582,968,556 bytes and SHA-256
`1836d704d15f27728e3ba09d7cddf0aea9458049bc66700dc98f14e7111ec29d`.
Round 1 permits only the published r92 scorer; later rounds require the exact
immediately preceding project bundle, decision log, and completion operation.
The baseline split and published r92 prototype/PCA state remain frozen. Each
new image group is train-only, and each successful resume starts a new full
XGBoost fit rather than appending parent trees.

The transfer lineage is labeled `POST_R92_REVIEWED_TRANSFER_BASELINE` and
`NOT_R92_REPRODUCTION`. It does not reconstruct the historical r1-r92 decision
trajectory, so the first new model is `project-r1`, not `r93`.

## Retained Genesis-Based Image-Round State Machine

Version 1.8.0 added a v3 active-learning operation alongside the v2
general-pool round:

```text
one unseen image
  -> infer: SAM2 proposals + four-scale features + parent XGB scores
  -> begin-image-round: abs(p-0.5) <= 0.2, deterministic Top-K 50
  -> explicit human review and finalization
  -> resume-image-round: fresh full XGB fit
  -> verified project-rN bundle
```

Round 1 is rooted in a sealed initial-completion operation. Round `N>1` is
rooted in the complete immediately prior image-round completion. Each edge is
closed by physical and logical SHA-256 records: genesis Stage-20 features,
genesis split, decision log, parent bundle, current inference archive,
selection/review, and cumulative active-learning raw-feature archive.

The genesis test groups are immutable. The current unseen image group is added
to effective training groups only. Training always starts a new XGBoost fit
over genesis plus all cumulative reviewed image-round rows; prior trees are not
updated in place. PCA stays frozen from the parent lineage. The prototype is
recomputed from accumulated outer-train positives. Previously reviewed images
are not proposed or feature-extracted again because their exact selected raw
feature rows are carried in the sealed cumulative archive.

The model lineage label `project-rN` describes only the current public
project. In this retained genesis-based path, the packaged r92 state is a
separate Stage-20 transfer-review assist. In the v1.9.3 strict transfer path,
the separately supplied exact snapshot provides training rows, but still does
not supply the complete historical r1-r92 decision chain. Neither path is
evidence that `project-r1` is historical `r93`.

Only the CLI and `public_pipeline.py` facade are supported standalone execution
boundaries. Symbols under `canonical.service` and the `canonical` re-export
surface are implementation APIs: even though their model-heavy operations are
CUDA-only and device-attested, callers must not invoke them without a successful,
still-current `science-gpu` preflight and dependency receipt owned by the public
facade.

The source-controlled diagram is [architecture.mmd](architecture.mmd).
