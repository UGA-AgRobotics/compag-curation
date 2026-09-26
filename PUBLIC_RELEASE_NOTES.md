> Historical release notes for earlier 1.9.3/JR09 and rc4 context. For the current rc6 entry path use `README.md`, `FIRST_RUN_COMPLETE_ROUND_EN.md`, and `PUBLIC_RELEASE_SCOPE.md`.

> **Current rc4 candidate (1.9.4rc4).** The fixed native r92 model and two approved example photos are separately packaged local release candidates. `r92 init-images` and `r92 infer-full` cover every prepared tile. Exact asset-specific terms and remote destination remain pending. The older JR09 model-exclusion boundary and the notes below describe historical 1.9.3 context, not this rc4 candidate.

> **JR09 public-code candidate boundary (2026-09-24).** The fitted Project-1/CJ r92 model closure is research material and is **not included** in this public-code candidate. Any inherited statement below calling that preset "bundled" or "public" describes the frozen private 1.9.3 build and is superseded for this candidate. The r92-assisted lane requires the separately controlled reviewer asset. Application/scientific algorithms in `src/` are unchanged. This is a local candidate pending repository confirmation and asset-specific rights review, not a deployed release.

# COMPAG Curation 1.9.3 Release-Candidate Notes

## v1.9.3 Parity-kit model-acceptance obligations

Version 1.9.3 corrects the parity kit's model-acceptance contract: the stored decision threshold and
an explicitly declared transform identity are separate, enforced obligations, and acceptance is
decided by structured statuses instead of a verdict string. The application's computational modules
are byte-identical to 1.9.2 and no formula, threshold used at inference, RNG, split or training
behaviour changed. Distribution remains through a verified GitHub Release only; nothing is published
automatically.

## v1.9.2 Only_codes input-reader contract

Version 1.9.2 is a corrective patch of the `only-codes-compat-v1` profile. It restores the reference
path contract of each legacy input reader (the feature exporter reads the basename under IMAGES_DIR,
the pack reader keeps its own absolute/relative behaviour), keeps explicit relocation under a
documented mapped-reader policy, and leaves every 1.9.1 state-safety fix in place. No scientific
formula, gate, scale list, split, augmentation, imputer, threshold or RNG stream was changed, and the
canonical Full, Lite, Full-image and r92 profiles are untouched. Distribution remains through a
verified GitHub Release only; nothing is published automatically.

## v1.9.1 Only_codes compatibility fixes

Version 1.9.1 is a targeted patch of the `only-codes-compat-v1` profile introduced in 1.9.0. It
fixes the handling of explicitly relocated inputs, makes a round's state commit transactional,
and separates preset selection and environment readiness from verified output parity. The
canonical Full, Lite, Full-image and r92 profiles are unchanged, and no scientific formula,
gate, scale list, split, augmentation, imputer or threshold rule was modified.
Distribution remains through a verified GitHub Release only; nothing is published automatically.

## v1.9.0 Only_codes Drop-in Compatibility

Version 1.9.0 adds the explicitly selected legacy profile
`only-codes-compat-v1`. It accepts the original Only_codes workspace directly
and reproduces the original computations and output files: see
`docs/ONLY_CODES_COMPATIBILITY.md`. Exact equality is claimed only in the
matched reference environment; other environments are labelled as execution
variants. The canonical Full, Lite, Full-image and r92 profiles are unchanged.
Distribution remains through a verified GitHub Release only; nothing is
published automatically.


Version 1.8.3 adds an explicitly experimental full-image execution profile
beside the retained tiled Full and Lite profiles. Distribution is through the
verified GitHub Release only. Do not upload the wheel or sdist to PyPI: the
supported GPU dependency metadata intentionally uses hash-bound direct archive
URLs and an exact SAM2 VCS commit.

## v1.8.3 Experimental Full-image Mode

`--profile full-image` and installer `--preset full-image` resolve to the
immutable ID `full-image-multiscale-xgb-recall-gpu-v1`. This mode:

- performs no external or persisted tiling and writes one lossless processing
  unit for each original image;
- uses SAM2's internal overlapping crop pyramid with Hiera-L, 64 points per
  side, configured `points_per_batch = 512`, two crop layers, crop-point
  downscale factor 1, overlap ratio `512/1500`, and a 1000-proposal image cap;
- executes the unchanged prompt lattice using a memory-bounded CUDA
  microbatch of 8; configured batch 512 and execution batch 8 are separate,
  hash-bound identities;
- retains the 93 exported feature-column names but gives them an explicit
  full-image ROI semantics identity, so the numerical feature/model domain is
  not interchangeable with tiled Full or Lite.

The initial two-layer/downscale-1 setting was selected from a bounded
engineering benchmark on `IMG_9475.jpg`. It retained 884 proposals, covered
98.56% of reference points at IoU 0.25 and 90.65% at IoU 0.50, had median best
IoU 0.875, took 144.3 seconds, and peaked at 2.86 GiB CUDA allocated / 5.91 GiB
reserved. A three-layer comparison reached 92.09% at IoU 0.50 only with the
1000-proposal cap binding and took 481 seconds; the +1.44 percentage-point
change did not justify more than 3.3 times the elapsed time for this initial
preset. These measurements are one-image engineering evidence, not a general
performance, biological-accuracy, canonical-method, or manuscript-equivalence
claim.

Stage 20 binds its exact `proposals.csv` and `features.csv` hashes alongside
the durable compact-mask encoding
`bbox-cropped-column-packbits-little-v1` and deterministic connected-component
policy `largest-8-connected-pixel-area-top-left-tie-v1`. Training revalidates
that record before fitting, so rows cannot be detached from the proposal and
feature tables that created them.

Full-image compact-mask fields can exceed Python's default CSV field limit.
Every package CSV reader therefore uses one shared, process-global,
serialized 64 MiB field-limit guard and restores the caller's prior limit on
success or failure. Regression coverage includes a real 1,745,468-byte compact
mask field from the GPU Stage-20 output.

Local installed-wheel GPU acceptance on an RTX 4090 completed six full-size
training images through the human-review boundary without external tiling. It
produced six full-frame processing units, 5,183 proposals, 20,732 four-scale
raw-feature rows, and a 5,183-row review request, then exited with the required
code 3 and `PAUSED_FOR_REVIEW`. The sealed proposal and feature tables were
byte-identical to the pre-fix diagnostic attempt; the fix changed CSV reading,
not scientific output. The complete process-isolated suite passed 469/469 with
no failures, errors, or skips. No review labels were invented, and review,
training, evaluation, remote CI, tagging, and publication remain separate
acceptance boundaries.

Scientific equivalence to the manuscript or tiled Full method is
`NOT_CLAIMED`. The published r92 scorer and strict post-r92 transfer workflow
remain tiled-Full-only. No tiled r92 resource, tiled bundle, feature archive,
review state, resume state, or active-learning artifact may seed a full-image
model. A full-image project must complete its own initial review/training and
may continue only with a verified full-image bundle. Conversely, a full-image
bundle cannot score or resume a tiled project.

The tiled canonical command remains unchanged:

```bash
set -euo pipefail
compag-curation init-project --profile full --output "$HOME/compag-workspaces/compag-full-project"
compag-curation run --config "$HOME/compag-workspaces/compag-full-project/config.toml" --output "$HOME/compag-workspaces/compag-full-run"
```

The experimental full-image command is:

```bash
set -euo pipefail
compag-curation init-project --profile full-image --output "$HOME/compag-workspaces/compag-full-image-project"
compag-curation run --config "$HOME/compag-workspaces/compag-full-image-project/config.toml" --output "$HOME/compag-workspaces/compag-full-image-run"
```

Both runs pause at the explicit human-review boundary. Use the reviewed CSV
only with the exact run/profile that created its request. The local
install-to-review GPU boundary passed as recorded above; exact release-artifact
hashes and installer verification are recorded during final assembly. Human
review-to-training E2E, remote CI, immutable tag, and publication remain
pending.

## Historical v1.8.2 Release-Candidate Notes

Version 1.8.2 retains the strict post-r92, one-new-image active-learning
workflow introduced in v1.8.1 while retaining the Full/Lite CUDA profiles, full-image review, general
cumulative-pool rounds, the v1.8.0 genesis-based image-round contract, and
fail-closed run/bundle/provenance contracts. Release acceptance, remote CI,
immutable tagging, and publication must be established for the exact 1.8.2
artifact set; historical acceptance below does not constitute v1.8.2
acceptance.

Distribution is through the verified GitHub Release only. Do not upload the
wheel or sdist to PyPI: the supported GPU dependency metadata
intentionally uses hash-bound direct archive URLs and an exact SAM2 VCS commit.

## v1.8.2 Corrections

- Native sealed-round review now preserves the distinction between an omitted
  round-number filter and an explicit null value, allowing the reviewer to
  discover the exact published begin operation before enforcing its manifest.
- Transfer-round shortlist and review-request inputs are verified using the
  transfer-specific role strings written into their sealed records.
- GPU installation now ignores Mamba's unreliable `sha256_in_prefix` as an
  authority for the relocated Conda Python executable. It instead binds the
  exact Python build and archive, the source member path/digest/size, and the
  one permitted installation-prefix relocation before hardening the binary.
- Regression tests cover both reviewer corrections and the fail-closed Python
  archive/member/relocation checks.

These are implementation and attestation corrections. They do not change the
scientific method, model/data contract, dependency-package closure, feature
set, thresholds, or sequential transfer policy. Exact 1.8.2
source/package/artifact/installer and real System-B acceptance remain pending.
Any v1.8.1 smoke run that supplies compatibility shims is diagnostic evidence,
not native acceptance of the corrected 1.8.2 paths.

## Historical v1.8.1 Strict Post-r92 One-Image Active Learning

The transfer workflow introduced in v1.8.1 closes one complete learning episode around exactly
one previously unseen image:

1. A strict importer verifies a separately supplied private r92 fit-time
   feature CSV by exact size `582968556` bytes and SHA-256
   `1836d704d15f27728e3ba09d7cddf0aea9458049bc66700dc98f14e7111ec29d`.
2. In round 1, `active-learning infer-r92-image` generates SAM2 proposals,
   extracts canonical four-scale features, and scores the one image with the
   safe published r92 classifier and frozen published prototype/PCA state.
3. `active-learning begin-transfer-image-round` seals the deterministic Top-K
   (`K=50`) proposals inside margin `0.2` of threshold `0.5`. When fewer than
   50 qualify, all eligible proposals are selected; zero eligible proposals
   fail closed.
4. The Web or native reviewer collects an explicit human action for every
   selected proposal. Predictions remain context and unselected proposals are
   not silently labeled.
5. `active-learning resume-transfer-image-round` starts a fresh full XGBoost
   fit over the verified baseline plus cumulative non-skip reviewed image-round
   rows and publishes `project-r1`.

Later rounds repeat the cycle for exactly one new image, score it with the
immediately preceding `project-rN`, verify unbroken bundle/decision/completion
ancestry, and fresh-fit `project-r(N+1)`. The baseline split, published r92
prototype, and published r92 PCA remain frozen; new image groups are
train-only.

This is a `POST_R92_REVIEWED_TRANSFER_BASELINE` workflow with the explicit
claim `NOT_R92_REPRODUCTION`. The recovered snapshot and published safe scorer
do not recreate the complete historical r1-r92 review and retraining sequence,
so `project-r1` is not `r93`. Transfer CV is conditional on the frozen r92
prototype/PCA, and reported PCA variance describes observed transfer-training
projections rather than historical fit-time r92 PCA variance.

The private source CSV and its imported baseline are supplied separately and
must never be included in the public repository, source ZIP, wheel, sdist,
online installer, public model resources, or GitHub Release. The public package
contains only code for strict import/verification and the already disclosed
safe portable r92 scoring resource.

See `docs/SEQUENTIAL_IMAGE_ACTIVE_LEARNING.md` for the complete command and
artifact contract.

### Private/local transfer-baseline artifact

The optional author-controlled transfer package is named
`COMPAG_CURATION_1.8.2_R92_TRANSFER_BASELINE_LOCAL_TRANSFER_ONLY.zip`, with an
adjacent `.sha256` sidecar. It is a private/local companion artifact, not a
public release deliverable. It must remain outside the public allowlist,
repository, source ZIP, wheel, sdist, online-installer ZIP, and GitHub Release.
The public importer still independently verifies the enclosed source feature
CSV against the fixed source byte size and SHA-256 before producing a usable
local baseline.

## Historical v1.8.0 Genesis-Based Image-Round Active Learning

The v1.8.0 workflow closed one complete learning episode around exactly one
previously unseen image:

1. `infer --execute` generates SAM2 proposals, extracts canonical four-scale
   features, and scores them with the current verified parent XGBoost bundle.
2. `active-learning begin-image-round` selects a deterministic uncertainty
   batch using threshold `0.5`, margin `0.2`, and Top-K `50`, then pauses with
   exit code 3.
3. The existing Web or native reviewer collects an explicit human action for
   every selected proposal and finalizes a strict reviewed CSV.
4. `active-learning resume-image-round` starts a fresh full XGBoost fit over
   genesis plus every cumulative reviewed image-round and publishes the next
   `project-rN` bundle.

The genesis test groups and PCA remain frozen. Each new image group is
train-only; the prototype is recomputed from accumulated outer-train positives.
A sealed cumulative raw-feature archive carries prior reviewed rows forward,
so older images are not reprocessed by SAM2 or feature extraction. Selection,
review, feature archive, parent bundle, genesis split, effective train groups,
frozen test groups, and model ancestry are hash-closed. Empty uncertainty
batches, repeated images/groups, lineage gaps, changed genesis artifacts, and
cross-profile inputs fail closed.

The packaged `compag-cj-r92` closure remains historical transfer assistance
for initial Project-1/CJ Stage-20 review only. It does not contain the original
training corpus or r1-r92 decision lineage. Therefore the first public
sequential model is `project-r1`, not `r93`, unless the genuine historical
lineage is separately supplied and verified.

That genesis-based workflow remains available as a separate contract. Its
initial-export, split, decision log, parent bundle, and ancestry artifacts must
not be mixed with the new strict transfer workflow.

### v1.8.0 release-native local-transfer cache

The optional local Conda/CUDA transfer cache for that release had the exact
basename
`COMPAG_CURATION_1.8.0_VERIFIED_CONDA_CUDA_CACHE_LINUX_X86_64_LOCAL_TRANSFER_ONLY.zip`.
The 1.8.0 wrapper accepts only that release-native archive and its exact 1.8.0
manifest/lock binding. It does not accept, rename, or normalize a v1.7.1 or
other predecessor cache. This cache contains third-party binaries, remains
outside the public GitHub artifact set, and does not make pip, SAM2 source, or
model-asset acquisition offline.

### Historical v1.8.0 local source QA evidence

For the v1.8.0 candidate, the guarded public-contract partition passed
`146/146` and the locked science partition passed `267/267`, both with no
skips. Its source privacy scan covered exactly 272 files and 14,990,970 bytes
with zero findings. These are historical v1.8.0 source-test and source-privacy
results only.

The counts above do not establish v1.8.2 package, artifact, installer,
remote-CI, publication, or real-GPU science acceptance.

## Historical v1.7.2 Baseline

The remaining v1.7.2 material records the implementation and local acceptance
baseline inherited by 1.8.0. It is historical and should not be read as a
successful build, GPU run, or publication claim for 1.8.0.

## v1.7.2 UBJ Validation Hotfix

Version 1.7.2 corrects a fail-closed Stage-50 false positive found during a
fresh v1.7.1 installation test. The v1.7.1 validator searched the complete
binary UBJ serialization for path-like byte sequences. Learned XGBoost typed
arrays contain arbitrary IEEE-754 and integer bytes, so a valid model could
incidentally contain the same bytes as a UNC or filesystem-locator fragment.
No failed candidate model bundle was published.

The v1.7.2 validator first loads the model with XGBoost, requires an exact
canonical UBJ round-trip, and recursively inspects the decoded textual
metadata surfaces. The existing empty-attribute, canonical feature-name,
empty feature-type, objective, feature-count, and single-output checks remain
closed. Private locators in decoded semantic metadata still fail. This
hotfix changes serialization validation only; it makes no new numerical,
biological-performance, or historical-paper-reproduction claim.

Package implementation bytes and the dependency receipt are hash-bound. A
v1.7.1 run therefore must not be resumed under v1.7.2 by weakening the run
binding. Preserve the old run as diagnostic evidence, create a new v1.7.2
run, and reuse a finalized v1.7.1 review CSV only when the new Stage-20 review
request passes the normal exact identity validation.

## Scope

COMPAG Curation is a review-gated binary target-versus-other workflow for
compatible yellow sticky-card imagery. It provides project inspection,
validated public-asset acquisition, SAM2 proposal generation, explicit human
review with a strict CSV output, resume, group-separated XGBoost training,
portable model bundles, fresh-image
inference, and optional recall-style point coverage.

## One Package, Two Explicit GPU Choices

Version 1.7.2 uses one source tree, one wheel, and one exact hash-locked
`science-gpu` dependency closure. Installation does not choose between two
code packages. The user explicitly selects one runtime/asset preset:

| Choice | Immutable profile | Proposal model | CUDA prompt microbatch | Claim boundary |
| --- | --- | --- | ---: | --- |
| `full` | `canonical-xgb-recall-gpu-v1` | SAM2.1 Hiera Large | 32 | Full canonical method/default/contract profile |
| `lite` | `efficient-xgb-recall-gpu-v1` | SAM2.1 Hiera Tiny | 16 | Efficient lower-footprint profile; canonical equivalence is not claimed |

Both choices keep `points_per_batch = 512` in method and bundle provenance and
evaluate the same 64-points-per-side prompt lattice. Both require CUDA for
SAM2, ResNet50, XGBoost training/scoring, and supported model inference.
Bounded image decoding, card/grid preparation, file/CSV handling, provenance,
serialization, and other pre/postprocessing without a supported GPU kernel may
run on CPU. No hidden or automatic CPU model fallback exists.

`tools/install_gpu_profile.py` requires an explicit `lite` or `full` preset and
prints a no-mutation plan by default. Repeating the verified request with
`--execute` creates the common locked CUDA environment, installs the one exact
wheel, runs the GPU doctor, records the selected profile in the environment
receipt, leaves the requested project path absent, and prints
matching initialization and asset commands. Friendly CLI aliases are
not persisted; immutable IDs are stored in configuration and provenance.

Lite and Full are different scientific profiles. Their proposal outputs,
trained model state, bundles, run outputs, resume records, and active-learning
lineage are profile-bound and are not interchangeable. A Full artifact cannot
be relabeled as Lite or vice versa.

### Exact online and verified local-cache installation

Version 1.7.2 provides two ways to create the same exact 71-package Conda/CUDA
environment. The normal online path consumes one authoritative `@EXPLICIT`
HTTPS lock instead of performing a new channel solve. Mamba is constrained to
one download thread and bounded retries, backoff, connect timeout, and read
timeout. A transient HTTP 5xx response can therefore be retried without
changing package identity or silently substituting another build.

When the NVIDIA package service is temporarily unavailable, the user may copy
`COMPAG_CURATION_1.7.1_VERIFIED_CONDA_CUDA_CACHE_LINUX_X86_64_LOCAL_TRANSFER_ONLY.zip`
from an already accepted machine and select it explicitly. This basename is
intentionally retained: the cache is the byte-exact v1.7.1 predecessor
artifact and is neither renamed nor republished as a v1.7.2 cache. Before any
installation mutation, the v1.7.2 wrapper verifies the outer archive, its
original manifest, and all 71 exact name/version/build/subdir rows, filenames,
sizes, and SHA-256 values against the unchanged package closure. In its private
extraction root it then writes the canonical v1.7.2 cache manifest bound to the
current v1.7.2 Conda lock before invoking the strict source installer. The
installer generates an explicit specification bound to local package paths and
runs Mamba with `--offline --always-copy`; the resulting `conda-meta` records
are attested against the current lock before the environment is accepted.

The outer acquisition receipt records the predecessor cache ZIP and manifest
identities as well as the normalized current lock and manifest identities. The
environment receipt records the current v1.7.2 lock/manifest contract. The
source installer itself accepts only its canonical v1.7.2 cache manifest and
does not contain a generic predecessor-cache exception.

Each execution uses its own private Mamba root and package cache next to the
new environment. A successful installation removes that private workspace
only after attestation; a failed installation preserves the partial
environment but also safely removes the private package workspace so hidden
multi-gigabyte cache state is not abandoned. It neither seeds nor cleans the
user's global Conda package cache, and the installer never runs
`mamba clean --all`.

The local cache is deliberately limited to the exact Conda/CUDA package
closure. It is not a fully offline COMPAG installer: pip packages, the pinned
SAM2 Git source, registered model assets, and a managed Miniforge bootstrap
when one is not already accepted still require network access. The host/WSL
NVIDIA driver must already make `nvidia-smi` work; neither installation path
installs that driver.

The predecessor v1.7.1 real local fallback proof extracted the accepted
71-package cache, built
the exact local explicit specification, and created an environment with real
Mamba under `--offline --always-copy` while network access was made
unavailable through invalid proxy endpoints. CPython 3.12.7 and all 71 exact
`conda-meta` records matched lock SHA-256
`3f80d14cdb2d5e04548ead7d9310b9bcd5d207bcc3d42d54d047dd00ff9d975c`.
The before/after identity of the pre-existing global Anaconda package-cache
member set was unchanged. This is evidence for the Conda/CUDA fallback stage
only, not a claim that pip, SAM2, or model-asset acquisition ran offline.

The retained v1.4 local installed-wheel acceptance ran in the exact environment
on an RTX 4090 with the CUDA 13.2 driver API and Torch cu132 runtime. Lite
(`efficient-xgb-recall-gpu-v1`, Hiera Tiny, microbatch 16) passed in 5:53.26
with 2,252,164 KiB maximum process RSS (about 2.15 GiB), 3,503 MiB sampled
maximum total GPU memory, and 74% sampled maximum GPU utilization. Full
(`canonical-xgb-recall-gpu-v1`, Hiera Large, microbatch 32) passed in 6:36.10
with 3,108,076 KiB maximum process RSS (about 2.96 GiB), 4,650 MiB sampled
maximum total GPU memory, and 88% sampled maximum GPU utilization. Both passed
the review pause/resume boundary, bundle creation, evaluation, and fresh
inference, resolved `cuda:0`, and recorded `allow_cpu_fallback=false`.

The retained v1.4 current-source local QA also passed: public QA `141/141`,
mocked GPU tests 110 passed with one expected skip, installed and independently extracted command
matrices `74/74` each, zero privacy findings, and deterministic artifact
checks. These numbers are not v1.7.2 acceptance. The v1.7.2 public artifacts
will use separately measured current acceptance; their exact hashes are
intentionally stated only in adjacent sidecars.

The predecessor v1.7.1 local software acceptance passed the public-contract partition
`29/29` in a fresh process and the remaining source suite `373/373`, both
without skips. The separate cache-builder suite passed `6/6`; the online
installer/builder/recovery harness passed `54/54` against the complete bundle
without skips. Metadata generation/checking, Twine distribution checks, and
privacy scans of the source tree, independently extracted public source ZIP,
wheel, sdist, and online-installer inputs passed with zero privacy findings.
Two independent wheel, normalized-sdist, public-source-ZIP, and online-wrapper
builds were byte-identical. These are retained v1.7.1 results; they are not
v1.7.2 acceptance, remote CI, or a new end-to-end GPU science run.

Final v1.7.2 local acceptance passed the public-contract partition `29/29` in a
fresh process and the remaining source suite `378/378`. Installed-wheel exact
CUDA/XGBoost checks passed `13/13`, and the online-installer suite passed
`56/56`; all four partitions completed without skips. Twine passed. Privacy
scans of the source, independently extracted tree, wheel, and sdist had zero
findings. The installer-template privacy scan covered exactly 6 files and
153903 bytes with zero findings. Independent source-ZIP, wheel, sdist, and
online-wrapper builds were byte-identical, and wrapper verify-only passed.
These results establish local software/package/artifact acceptance only. They
do not claim remote CI or a new real end-to-end GPU science/System-B run.

The predecessor v1.7.1 exact shared science-gpu doctor separately passed on an RTX 1000 Ada 6 GB
laptop with driver 580.173.02. That is dependency compatibility evidence, not
Lite end-to-end acceptance: the Lite real demo has not run on that laptop. The
RTX 4090 observations are bounded synthetic-demo measurements, not an
automatic VRAM threshold, cross-device guarantee, or biological-performance
claim.

## Supported Human Review

Version 1.7.2 retains one safe review engine with three CLI modes and upgrades
both graphical modes to the shared full-image scene:

- `review-ui web` opens a loopback-only browser application whose server uses
  the Python standard library; verified round visuals use Pillow from the
  locked GPU profile;
- `review-ui desktop` opens a native Ubuntu/WSLg Tk/Pillow application;
- `review-ui status` prints saved progress without opening an interface.

The reviewer accepts two exact source contracts and exposes three separate
scientific workflows. Stage-20 review may be either Project-1/CJ historical
r92 transfer-assistance or model-free cold start. Both accept an exactly
paused run and verify its run/stage ledgers, immutable review export, proposal
table, overlay manifest, sealed Stage-10 tiles, and full-image scene source
hashes. The third workflow, project-trained round review, accepts the exact
whole output root of one
sealed `active-learning begin-round` operation only when its exact bound Stage-60 root
and original inference-image directory are also supplied. The CLI names these
inputs `--selection-root`, `--stage60-root`, and `--images`; `--output` names a
new reviewed CSV and optional `--state` names another new external session
directory. It verifies the operation, selection and prediction lineage, source
bundle/profile, and complete original image inventory. Every human set/undo
action is an append-only hash-chained event, so closing and reopening the
application with identical arguments preserves completed work.

`EVENT_HEAD.json` binds each session/source to the committed sequence and hash.
On reopen, the reviewer replays the bounded contiguous event directory and
checks every hash link and set/undo transition. A fully valid durable event
suffix beyond an older head is the recoverable interrupted-commit case; a head
ahead of events, gap, malformed transition, changed binding, or hash mismatch
fails closed. The durable journal is replayed again immediately before final
CSV publication.

Both interfaces and both source contracts provide the same full-image scene,
proposal selection, zoom/pan, unique proposal-prefix search, pending/reviewed
and group filters, keyboard shortcuts, auto-next navigation, undo, progress,
and explicit finalization. Mask-derived outlines are enabled by default.
Bounding rectangles are an independent display-only toggle and are disabled by
default. Auto-next changes only the selected proposal and never creates a
decision. No interface silently predicts, propagates, auto-accepts, finalizes,
resumes, or trains. The sole multi-row decision is the explicit, confirmed,
reversible r92 remainder action described below.

Project-1/CJ review may opt into the safe bundled paper-selected r92 portable
closure. It is accepted only for the canonical Full profile, fixes CJ/Non-CJ
labels and the paper operating threshold at 0.50, scores on CUDA, and orders
the full-image proposals by increasing distance from 0.50. Manual corrections
remain explicit and authoritative. After corrections, the user may confirm an
exact displayed count of still-pending suggestions; positives become
`sus_accept` and negatives become `sus_flip`, both at weight 0.4. That whole
confirmation is one append-only action and one Undo reverses it before
finalization while it remains the latest action. Receipts distinguish
`HUMAN_EXPLICIT` from
`PUBLISHED_MODEL_BULK_CONFIRMED` decisions and bind the preset/resource/feature
hashes, threshold, scores, uncertainty order, and counts. This is historical
transfer-assistance only: it makes no fresh-canonical-equivalence,
current-dataset performance, or paper-result-reproduction claim.

The exact Web command is:

```bash
set -euo pipefail
VENV="$HOME/.local/share/compag-curation/venvs/science-gpu-full-1.8.0"
RUN="$HOME/compag-workspaces/compag-canonical-run"
REVIEW="$HOME/compag-workspaces/compag-cj-r92-reviewed.csv"
test ! -e "$REVIEW"
"$VENV/bin/python" -I -B -m compag_curation review-ui web \
  --run "$RUN" --output "$REVIEW" \
  --target-label CJ --non-target-label Non-CJ \
  --model-preset compag-cj-r92
```

Replace `web` with `desktop` for the native Ubuntu/WSLg application.

Model-free cold-start review omits `--model-preset`. It reconstructs the
complete prepared card from sealed Stage-10 tiles and maps each Stage-20
mask-derived polygon into that full-card coordinate system. No fitted
classifier exists in this workflow, so every exported proposal remains
reviewable and must receive an explicit human decision before finalization.

Project-trained iterative review is the third, separate boundary. It requires
a verified safe `compag-curation-model-bundle/v2`, completed Stage-60 scoring,
and a sealed `begin-round` hard-case selection. The reviewer renders the
complete original inference image with Stage-60 mask-derived outlines. Only
proposals in the sealed shortlist are selectable; other detections remain
visible as muted context. Model score, prediction, kept flag, uncertainty, and
rank are display context only and are never converted into human labels. It
does not accept an r92 preset or expose the r92 remainder confirmation. The
review application does not train or replace the separately sealed
inference/selection workflow.

The UI may present the classes as CJ/Non-CJ, Target/Non-target, or two concise
custom names. These names are presentation-only: durable labels remain `1` and
`0`, and the action/weight mapping below is unchanged.

The five choices have a fixed output mapping:

| Human choice | Output mapping |
| --- | --- |
| Target, confident | `label=1`, `review_action=accept`, `review_weight=1.0` |
| Non-target, confident | `label=0`, `review_action=flip`, `review_weight=1.0` |
| Target, uncertain | `label=1`, `review_action=sus_accept`, `review_weight=0.4` |
| Non-target, uncertain | `label=0`, `review_action=sus_flip`, `review_weight=0.4` |
| Skip, zero weight | `label=0`, `review_action=skip`, `review_weight=0.0` |

Finalization remains disabled while any proposal is pending. It validates exact
identity parity and the action/weight contract, then creates the reviewed CSV
without replacement. Initial Stage-20 review also validates group-split
feasibility; round review validates the canonical shortlist review table
without applying that initial-dataset gate. Closing the application does not
finalize, and successful finalization does not resume a run or round or train a
model.

After project initialization, the installer can add no-clobber launcher pairs.
`START_COMPAG_REVIEW_UBUNTU.desktop` / `.sh` is the master chooser: it offers
Project-1/CJ r92 transfer-assist, model-free cold start, or an existing
project-trained round. R92 fixes CJ/Non-CJ and defaults to
`review/cj-r92-reviewed.csv`; cold start offers CJ/Non-CJ, Target/Non-target,
or custom presentation names and defaults to `review/reviewed.csv`; the round
choice delegates to the direct launcher. Stage-20 choices ask for Application
or private loopback Web, an exact paused-run path, and a new output CSV.
`START_COMPAG_ROUND_REVIEW_UBUNTU.desktop` / `.sh` remains a direct shortcut to
the sealed-round path. The launchers prompt for exact absolute Linux paths and
never label, resume, or train automatically.

Web mode binds only to `127.0.0.1` and protects its local API with a per-process
capability token, exact Host/Origin checks, strict bounded requests, no-store
responses, and a restrictive Content Security Policy. It is not designed for
remote serving, port forwarding, or proxy publication.

The v1.7.2 reviewer may finalize a structurally compatible run still paused under
v1.4 or v1.5 when its sealed Stage-10/Stage-20 structure satisfies the current
full-image scene contract. The scientific run remains bound to the
package/dependency identity that created it and must be resumed with its
unchanged originating environment; upgrading that environment in place or
resuming under v1.7.2 must fail as drift.

## Prior v1.3 Full GPU Evidence And Contract Retained

Version 1.3.0 replaces the supported canonical CPU surface with the
fail-closed `canonical-xgb-recall-gpu-v1` profile. The supported science
environment is exact CPython 3.12.7 on Linux x86_64 with an NVIDIA GPU. Its
reviewed closure includes PyTorch `2.13.0+cu132`, torchvision
`0.28.0+cu132`, CuPy `14.2.0`, and XGBoost `2.1.1`; runtime validation also
requires XGBoost to report a CUDA-enabled build.

SAM2 is pinned to commit
`2b90b9f5ceec907a1c18123530e92e794ad901a4` and built with CUDA 13.2 Update
2 `nvcc` 13.2.86. The build environment must provide the CUDA runtime,
cuBLAS, cuSPARSE, and cuSOLVER development headers documented in
`docs/INSTALL_GPU.md`.

CUDA is required for SAM2 proposal generation, ResNet50 embedding, canonical
XGBoost training and scoring, and supported canonical model inference. Missing
or incompatible CUDA hardware, driver/runtime state, CUDA-enabled dependency
builds, or a non-CUDA device request stops execution. There is no automatic or
silent CPU fallback. Bounded image decoding, card/grid preparation, row and
array assembly, CSV/review handling, provenance, serialization, and report
generation remain CPU preprocessing/postprocessing boundaries.

The frozen proposal method remains 64 points per side with
`points_per_batch = 512` recorded in method and bundle provenance. Stage 20,
Stage 60, and standalone bundle inference use a sealed execution-only CUDA
prompt microbatch of 32. For proposals that do not enter M2M refinement,
transient `low_res_masks` are discarded before upstream aggregation because
they cannot be consumed. The same full prompt lattice is evaluated; neither
adaptation is user-configurable and neither permits CPU fallback. Proposal
selection, deduplication, and the 500-mask cap use the original unrounded SAM2
scores. Only the selected archival confidence fields are then quantized to six
decimal places; those two fields are not inputs to the canonical 93-feature
classifier.

On the locked RTX 4090 host, the v1.3 old method-batch-512 Stage-20 run peaked at
24,018 MiB total device use. Execution microbatch 32 peaked at 4,441 MiB from an
837 MiB baseline (3,604 MiB incremental), and repeated at 4,431 MiB total/3,594
MiB incremental. The final full demo passed in 6:28 with a 4,913 MiB total peak;
Stage 50 reached 1,427 MiB and standalone inference reached 4,431 MiB. These
measurements are historical v1.3 Full evidence only. They are not v1.4 through
v1.7.2 Full/Lite acceptance, acceptance of a laptop GPU, a preset-selection threshold,
or a general memory/performance promise.

The execution adaptation has a disclosed small hardware-numerical effect versus
the old-512 host baseline: one of 528 masks changed by one pixel, seven SAM2 IoU
values changed by 1e-6, and two stability scores changed by approximately
2.65e-4. The new-32 Stage-20 baseline was byte-stable across two runs, while the
full Stage-60, fresh-process, and repeat inference outputs were byte-identical.

The v1.2.0 `science-cpu` lock and `canonical-xgb-recall-cpu-v1` profile are
retained only as historical provenance. They are not supported v1.3.0 install
or execution surfaces because the v1.2 lock contains a Torch version affected
by a security advisory. Do not create a new environment from that historical
lock.

## Compatibility And Evidence Boundary

Base and synthetic quick-demo surfaces continue to support CPython 3.12 on
Linux x86-64. The canonical science profile has the narrower exact environment
above. macOS, Windows, non-NVIDIA accelerators, and containers are not tested
release profiles; no container is provided.

GitHub-hosted CI builds and checks the distributions, exercises base and quick
surfaces, and runs static/mocked GPU contract tests. Those hosted runners do
not provide release GPU acceptance, do not run `doctor --profile science-gpu`,
and are not evidence of actual CUDA execution. The separate local acceptance
on the documented real NVIDIA environment is recorded above. The measured
times and resource maxima describe only those bounded synthetic exact-host
runs; no general throughput, latency, accuracy, preset-selection threshold, or
resource guarantee is made.

The bounded v1.2.0 copied-data CPU result (8 groups, 346 review rows, 69 test
rows with 5 positives, confusion matrix `[[64,0],[1,4]]`, and recall-style
point coverage over 13 points) is historical functional evidence only. It is
not v1.3.0 GPU acceptance, a benchmark, a resource guarantee, historical
fitted-model/split identity, private-output parity, or reproduction of paper
results.

CVAT/JSON points remain evaluation-only. Existing box/mask/class datasets are
not directly ingestible, and the supported review applications produce the
same strict binary CSV contract; no arbitrary-image annotation pipeline is
included. Inspect Stage 10 warp/grid evidence and Stage 20 overlays before
trusting proposals.

## Safety And Release Corrections Retained

- The hash-locked bootstrap now includes `packaging 26.3`, closing the runtime
  dependency of `wheel 0.48.0`; a fresh base install must pass `pip check`.
- Public model loading fails closed for pickle/joblib, unknown, linked,
  hash-mismatched, and pickle-like inputs before unsafe deserialization;
  reviewed XGBoost JSON/UBJ is the public model interchange.
- Evaluation identity comes from validated user input rather than a historical
  hard-coded image name.
- Descriptor/path-swap QA accepts both legitimate fail-closed race outcomes
  without weakening production verification.
- PEP 610 verification enumerates only the active interpreter's attested
  top-level install roots, so metadata for dependencies vendored inside another
  package is not mistaken for a second install; physically distinct top-level
  duplicates remain terminal.
- Transient CUDA runtime-probe tensors are released after synchronization, and
  the exact pinned Torch runtime's persistent cuBLAS workspaces are explicitly
  cleared before cache eviction, so the parent Torch/CuPy allocator-zero check
  remains strict before spawning independent inference.
- Fresh-process accelerator caches are confined to an already nonempty,
  run-owned runtime root. After successful child validation, the complete
  closed cache tree is checked for safe ownership and node types, generated
  cache files are removed, and only explicit environment/allocator-release
  evidence remains hash-bound. Failed runs retain their structured diagnostics
  without opaque accelerator caches. Compiler- and host-derived cache names
  therefore cannot enter Stage 60 or real-demo output.
- The privacy scanner has no author-specific self-bypass and scans concatenated
  locator literals.
- Source and GitHub Release wheel onboarding are independent and fresh-shell
  safe. Environment-creation blocks set exact `umask 0022` before creating or
  installing anything, so the installed package and Conda/Mamba history
  satisfy the same permission policy enforced by `doctor`, even when the
  caller inherited collaborative `umask 0002`. A failed GPU installation
  attestation now includes its underlying diagnostic, such as unsafe package
  permissions, without relaxing the fail-closed check. User projects, data,
  review files, models, predictions, and outputs use external workspaces with
  defense-in-depth ignore rules.

## Artifact And Remote Identity

The intended public 1.8.2 set is
`COMPAG_D_26_02120_PUBLIC_RUNNABLE_GITHUB_v1.8.2.zip`,
`compag_curation-1.8.2-py3-none-any.whl`,
`compag_curation-1.8.2.tar.gz`, the byte-identical bootstrap lock, this
`PUBLIC_RELEASE_NOTES.md`, and
`COMPAG_CURATION_1.8.2_ONLINE_INSTALLER_LINUX_X86_64_NVIDIA.zip`; each receives
an adjacent sidecar formed by appending `.sha256` to its exact basename. The
historical v1.7.2 local artifact set passed reproducible-build, privacy,
distribution, and wrapper verification acceptance. The 1.8.2 set requires its
own complete acceptance before release. Exact byte identities remain confined
to the adjacent sidecars and release evidence. No source-tree statement asserts that a
remote release or remote CI has passed, and no new real end-to-end GPU
science/System-B run is claimed.

The optional current cache
`COMPAG_CURATION_1.8.2_VERIFIED_CONDA_CUDA_CACHE_LINUX_X86_64_LOCAL_TRANSFER_ONLY.zip`
and its adjacent sidecar are a separate machine-to-machine recovery aid. They
contain third-party Conda/CUDA binaries, are excluded from the intended public
artifact set above, and must not be uploaded to the public GitHub repository or
GitHub Release. Its acceptance does not change the identities of the public
source ZIP, wheel, sdist, or online-installer ZIP. The separate
`COMPAG_CURATION_1.8.2_R92_TRANSFER_BASELINE_LOCAL_TRANSFER_ONLY.zip` is also
excluded from the public set; it contains private transfer data rather than
third-party package archives and is governed by the stricter source-hash
import boundary described above.

The source tree remains repository-identifier-neutral by design; it is not
mutated to embed a repository URL or commit after acceptance. The author must
push the exact accepted source commit, wait for the first remote GitHub Actions
run on that full commit to pass, and only then create the immutable version tag
at the same commit. After pushing the unmoved tag, its workflow on that same
commit must also pass. `tools/release_provenance_verifier.py` can subsequently
perform a no-concurrent-mutation, point-in-time check with trusted
`/usr/bin/git`, without modifying release inputs, that the local tag, index,
tracked worktree, and independently extracted source ZIP are identical and
that the configured remote URL matches. It does not query GitHub or claim
remote CI, remote-tag publication, or release-page state; those facts remain
explicit author checks before creating the public GitHub Release. Real local
GPU acceptance is the separate evidence recorded above.
