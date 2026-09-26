# 1.9.4rc10 / R5.4

See [R5.4 release notes](PUBLIC_RELEASE_NOTES_RC10_R5_4.md) for owned-process Stop, duplicate-session rejection, current documentation and controlled-replay binding.

> **JR09 public-code candidate boundary (2026-09-24).** The fitted Project-1/CJ r92 model closure is research material and is **not included** in this public-code candidate. Any inherited statement below calling that preset "bundled" or "public" describes the frozen private 1.9.3 build and is superseded for this candidate. The r92-assisted lane requires the separately controlled reviewer asset. The rc6 project review correction retains the native r92 and Only_codes contracts. This is a local candidate, not a deployed release.

## 1.9.4rc6 - 2026-09-24 (full scored-pool review and explicit commit scope)

The r92 project reviewer now exposes every valid scored candidate, with optional bound Top-K rank and independent model, shortlist, review-status, and human-label filters. It supports reversible per-mask Delete and a confirmed Accept remaining action across all undecided candidates using recorded model predictions. Bulk accepted labels retain distinct action provenance and a batch undo receipt; deleted masks are excluded from downstream snapshots, COCO export and training. Restricted v1 sessions can be expanded into a new v2 state with authentic events preserved. Reviewed/all/locked finalization scopes are explicit, and the default XGBoost round commits all eligible effective labels, including ones outside the shortlist. The Only_codes compatibility reviewer remains unchanged.

# Changelog

## 1.9.4rc9 easy-start R2 - 2026-09-25 (guided onboarding)

- Add a desktop guide with a terminal fallback for matching Release file checks, pinned GPU setup, ten-card preparation, full-card inference, review and explicit next-round XGBoost training.
- Keep the 1.9.4rc9 application wheel, native model, photos and scientific pipeline byte-identical to the preceding rc9 release candidate. User results stay outside the source repository.

## 1.9.4rc9 - 2026-09-25 (external CJ YOLO segmentation adapter)

- Attest the externally supplied one-class CJ segmentation package and its exact Ultralytics 8.4.26 inference settings.
- Precompute boxes and confidences for every prepared 512-pixel tile in the separate optional environment; validate a v2 source, model and configuration bound receipt in the science-GPU environment.
- Feed those boxes into existing SAM2 prompt, XGBoost/YOLO fusion, or both modes, and allow explicit external checkpoint activation in a verified model set.
- Keep the external checkpoint and its instance masks outside the public package; OFF-mode inference and existing detection training remain unchanged.

## 1.9.4rc8 - 2026-09-25 (review image right-button pan)

The full-pool project reviewer suppresses the native browser context menu on its image canvas. The first right-button drag pans immediately; left-button drag and left-click selection remain available. Existing scored inference and review state remain usable. The rc7 inference, scoring, model and asset bytes are unchanged.

## 1.9.4rc7 - 2026-09-25 (degenerate SAM2 contour review fallback)

Keep a nonempty canonical mask and its scored candidate when its 1-pixel contour approximation has no area. Supply the existing reviewed bbox display fallback, count such cases in the full inference receipt, and reject truly empty or malformed masks. The native r92 model, full mask records, feature extraction, threshold, and numerical scoring are unchanged. The rc6 R3 checkpoint identity intentionally does not match this version; start a new checkpoint directory when upgrading.

## 1.9.4rc5 - 2026-09-24 (public project review and retraining continuation)

- Complete tiled full-card proposals now preserve full masks and source bindings for selected original-coordinate COCO export.
- Add transparent review selection, resumable selected review display, ordered-event finalization, cumulative snapshots, genuine fresh project XGBoost fitting, verified model sets, and next-card inference.
- Add completeness-certified one-class CJ YOLO dataset conversion, optional real local training and checkpoint reload, and explicit OFF/PROMPT/FUSION/BOTH inference routing.
- Add hash-bound per-tile inference checkpoints for safe resume and verify retained YOLO checkpoints against an explicitly named earlier model set.
- Keep human acceptance, independent scientific validation, YOLO redistribution rights, and publication status separate from local software-test evidence.

## 1.9.3 - 2026-09-22 (parity kit: structured model-acceptance obligations)

- Parity kit only. The `only-codes-compat-v1` computational modules of the package are byte-identical
  to 1.9.2; this release exists so that every distributed copy of the comparator carries the fix.
- `model_compare` publishes one obligation per claim (learned tree state, tree count, runtime config,
  feature schema, class mapping, imputer state, stored decision threshold, declared transform
  identity, predictions, provenance) with an explicit PASS / FAIL / NOT_SUPPLIED / NOT_RUN status.
  The stored decision threshold is a required scientific field and is compared, never applied in
  place of the application's configured runtime threshold; `transform_identity` now means the
  explicitly declared identity only, generic `meta` is provenance and is normalised for recorded
  path roots and timestamps.
- `recheck_training_rounds` and `run_model_comparison` accept through those obligations under a
  declared model scope instead of a verdict string, so a favourable label can no longer pass a failed
  or absent required obligation, and an absent transform identity is reported as not verified rather
  than as agreement.

## 1.9.2 - 2026-09-22 (Only_codes compatibility: reader contract, stage identity, training recheck)

- The feature-export reader again reads `IMAGES_DIR/Path(file_name).name` exactly as the reference
  cell does, even when `file_name` carries an absolute directory or a relative sub-directory and even
  when a same-named file exists there; the pack reader keeps its own absolute/relative contract. With
  no applicable `--map` rule each reader offers exactly one candidate, so file existence can no
  longer decide which image is read. Explicit relocation stays supported (mapped path first, then the
  reader's reference candidate) and every attempt records the rule, the selected path, its hash and
  what the reference reader would have read.
- `features.py` and `pack.py` no longer join a second path outside the resolver: one authoritative,
  reader-aware resolution decides what is read, so a successful read cannot contradict the recorded
  outcome. The 1.9.1 protections (abort on unreadable input, no completion for a failed image, atomic
  `.npy` publication, retry and overlay safety) are unchanged.

## 1.9.1 - 2026-09-22 (Only_codes compatibility: relocated inputs, state commit, evidence)

- `only-codes` now applies the explicit `--map OLD=NEW` relocation rules at the COCO image-path
  boundary of the pack and feature-export readers (`only_codes/inputs.py`). With no rule the
  candidate order is exactly the reference one, so a successful legacy read is unchanged; a rule is
  applied once, on whole path components, longest rule first, and a Windows-style path is refused
  unless a rule maps it. Every attempt is recorded (candidates, rule, resolved path, identity).
- A round no longer completes on a failed input: the pack aborts before publishing anything when an
  image cannot be read (no embeddings appended, no `used_img_ids` advanced, previous pack,
  prototype and PCA untouched), the feature export refuses to write a partial set of rows, and
  publication is atomic. Each run writes stage-1 receipts, so a later inspection can distinguish a
  contribution, a legitimate zero contribution and a failure.
- Added `only-codes inspect-state [--rebuild-into DIR]`, which reports what a pre-1.9.1 overlay can
  and cannot prove and can rebuild only the derived TRAIN scope into a new overlay; historical packs
  and feature tables are never rewritten in place.
- The inference run manifest (schema v2) separates `reference_preset_requested`,
  `effective_configuration_match`, `inputs_verified`, software/hardware observations,
  `execution_variant`, `run_completed` and `output_parity_verification`; a completed run starts as
  `NOT_VERIFIED_IN_THIS_RUN`. The reference hybrid preset preflights what it consumes and a missing
  or pin-mismatching detector blocks the run instead of silently selecting the detector-free branch
  (`--detector none` selects that branch explicitly as a declared variant).
- `only-codes doctor` (schema v2) lists the checked, differing and unknown software factors, reports
  the observed GPU separately, and states that readiness is not output parity.
- `test_public_module_imports_are_inert` now measures the imports caused by the operation in a fresh
  child interpreter, so the process-global module state of unrelated tests cannot decide it.

## 1.9.0 - 2026-09-22 (Only_codes drop-in compatibility workflow)

- Added the explicitly selected legacy workflow `only-codes-compat-v1`
  (`compag-curation only-codes ...`) that imports an existing Only_codes
  workspace directly (COCO JSON, tiles, splits, embeddings, prototypes, PCA,
  fold-safe packs, `features_train.csv`, round models and state, detections,
  AL shortlists and review logs) without manual conversion, relabeling,
  retiling or rebuilding. Originals are read-only; updates go to a
  copy-on-write project state with the legacy relative layout.
- Vendored the original computational sources (`gate_core.py`,
  `sam2_pipeline/*`) with documented plumbing patches only, and ported the
  notebook merge, split, pack, feature-export, Cell3A and Cell3B logic and the
  reviewer state machine with identical statement order and random streams.
- Added the trusted legacy model import (`import-model --trust-legacy-pickle`)
  that converts a pinned joblib pickle into a pickle-free bundle and accepts it
  only after bit-exact probability verification; data packs use a NumPy-only
  restricted reader.
- Added a loopback-only legacy reviewer and an import form.
- Added an independent parity harness (`tools/only_codes_parity`) that runs
  the original source as an oracle and compares outputs bitwise.
- The canonical Full, Lite, Full-image and r92 workflows are unchanged.

## 1.8.3 - 2026-09-03 (experimental full-image candidate; acceptance pending)

- Added the noncanonical `full-image` execution choice with immutable profile
  ID `full-image-multiscale-xgb-recall-gpu-v1`. The existing `full` choice
  remains the canonical 512-by-512 tiled Hiera-L workflow and remains the
  default; `lite` remains the Hiera-T tiled workflow.
- Full-image mode persists exactly one lossless processing unit per input
  image and does not create or persist an external tile grid. SAM2 still uses
  its internal overlapping crop pyramid: 64 points per side, two crop layers,
  crop-point downscale factor 1, overlap ratio `512/1500`, and a deterministic
  cap of 1000 proposals per image.
- Separated the configured scientific `points_per_batch = 512` identity from
  the memory-bounded CUDA execution microbatch of 8. The smaller execution
  batch does not change the prompt lattice or become a user-tunable setting.
- Added full-image ROI-bounded Ultra feature extraction, profile-bound model
  bundle semantics, Stage-00/10/20/60 provenance, review-scene support, and
  same-profile general active-learning continuation.
- The retained two-layer/downscale-1 operating point was measured on
  `IMG_9475.jpg`: 884 retained proposals, 98.56% point coverage at IoU 0.25,
  90.65% at IoU 0.50, median best IoU 0.875, 144.3 seconds, and peak CUDA
  memory of 2.86 GiB allocated / 5.91 GiB reserved. Three crop layers reached
  92.09% at IoU 0.50 only after the 1000-proposal cap, took 481 seconds, and
  was rejected for this initial operating point (+1.44 percentage points for
  more than 3.3 times the elapsed time). This single-image engineering
  benchmark is not a biological-accuracy, general-performance, or equivalence
  claim.
- Full-image Stage 20 now records the compact-mask encoding
  `bbox-cropped-column-packbits-little-v1`, the deterministic component rule
  `largest-8-connected-pixel-area-top-left-tie-v1`, and exact proposal and raw
  feature table hashes. Training validates these bindings before fitting.
- Raised full-image CSV reading through a shared, serialized 64 MiB field
  guard that restores the process setting after every read. This fixes the
  Stage-20 failure exposed by a valid 1,745,468-byte compact-mask field and is
  regression-tested across public-pipeline, service, review, transfer, and
  published-model readers.
- Declared `canonical_method_equivalence = NOT_CLAIMED`. Tiled r92 resources,
  tiled bundles, features, review state, resumes, and strict post-r92 transfer
  artifacts cannot cross into full-image mode. A full-image project must train
  its own first classifier and continue only with full-image bundles.
- Extended project schema v2, the CLI/init templates, GPU installer planning,
  public release metadata, documentation, and focused fail-closed tests while
  retaining read-only compatibility with schema-v1 tiled projects.
- Local installed-wheel GPU acceptance on an RTX 4090 processed six full-size
  images with no external tile directory, sealed 5,183 proposals and 20,732
  four-scale feature rows, and reached the required `PAUSED_FOR_REVIEW` state.
  The process-isolated suite passed 469/469. Final release-artifact and
  installer hashes are established during assembly; human review-to-training
  E2E, remote CI, tag, and publication remain pending.

## 1.8.2 - 2026-09-03 (correction candidate; acceptance pending)

- Corrected native sealed-round discovery so an omitted round number remains
  distinct from an explicitly supplied null value. The reviewer now delegates
  operation discovery without injecting a round-number constraint unless the
  caller actually supplied one.
- Corrected transfer-round reviewer input identities. Transfer shortlists and
  review requests are now verified with their transfer-specific roles instead
  of the general active-learning role strings recorded by other round types.
- Corrected Conda Python attestation in the GPU profile installer. The
  installer no longer trusts Mamba's unreliable `sha256_in_prefix`; it binds
  the exact Python package build/archive, source member path/digest/size, and
  proves the installed executable differs by exactly one expected prefix
  relocation before hardening it.
- Added regression coverage for omitted versus explicit round-number
  verification, transfer-specific review roles, exact Python archive/member
  identity, the permitted single relocation, and rejection of drift.
- Retained the 1.8.1 scientific method, model assets, feature contract,
  dependency-package closure, thresholds, and transfer workflow. Exact 1.8.2
  source/package/artifact/installer and real System-B acceptance remain
  pending; a v1.8.1 shim-assisted smoke run is not native 1.8.2 acceptance.

## 1.8.1 - 2026-09-02 (candidate; acceptance pending)

- Added the strict `SINGLE_NEW_IMAGE_POST_R92_TRANSFER_V1` workflow. Round 1
  performs proposal generation and canonical feature extraction for exactly
  one unseen image, then scores it with the verified published r92 classifier.
  Later rounds score exactly one new image with the immediately preceding
  `project-rN` bundle.
- Added `active-learning import-r92-transfer-baseline`, which accepts only the
  separately supplied recovered fit-time CSV with size `582968556` bytes and
  SHA-256
  `1836d704d15f27728e3ba09d7cddf0aea9458049bc66700dc98f14e7111ec29d`.
  The private CSV and imported baseline are never bundled in the public source,
  source ZIP, wheel, sdist, online installer, model resources, or GitHub
  Release.
- The strict importer retains all 90,887 complete labeled non-skip proposals
  and their 363,548 canonical four-scale rows. It preserves reviewed versus
  unreviewed provenance; historical unreviewed labels receive unit numerical
  weight without being rewritten or claimed as explicit human accepts.
- Added `active-learning infer-r92-image`,
  `begin-transfer-image-round`, and `resume-transfer-image-round`. Selection is
  deterministic Top-K (`K=50`) inside the inclusive margin `0.3..0.7`; when
  only 1 through 49 proposals qualify, every eligible proposal is reviewed.
  Zero eligible proposals fail closed.
- Each completed transfer round requires explicit review of its complete
  sealed shortlist and starts a fresh full XGBoost fit over the verified
  baseline plus every cumulative non-skip reviewed image-round row. Round 1
  publishes `project-r1`; each later round publishes the next `project-rN`.
  No warm start or incremental tree append is permitted.
- Bound all transfer models to `POST_R92_REVIEWED_TRANSFER_BASELINE`, frozen
  published r92 prototype/PCA state, and
  `CONDITIONAL_ON_FROZEN_R92_TRANSFER_PROTOTYPE_AND_PCA` evaluation. Observed
  transfer projection variance is not represented as historical r92 fit-time
  PCA variance.
- Labeled the workflow `NOT_R92_REPRODUCTION`. The private snapshot and safe
  public scorer do not reconstruct the complete historical r1-r92 decision
  trajectory, so `project-r1` is not described as `r93`.
- Preserved the v1.8.0 genesis-based image-round and general-pool interfaces as
  separate contracts. Their run, review, split, decision-log, model, and
  ancestry artifacts cannot be mixed into the new transfer lineage.

## 1.8.0 - 2026-09-02 (candidate; acceptance pending)

- Added `active-learning begin-image-round` and `resume-image-round` for an
  explicit one-new-image workflow. Standalone `infer --execute` performs SAM2
  proposal generation, canonical four-scale feature extraction, and scoring
  with the current parent XGBoost bundle before selection.
- Bound image-round acquisition to the manuscript policy: XGBoost-only,
  threshold 0.5, absolute threshold distance, margin 0.2, deterministic Top-K
  50, and proposal-ID tie breaking. Empty in-margin batches fail closed.
- Required exactly one previously unseen image/group per image-round, excluded
  frozen genesis test groups, and rejected repeated genesis or prior-round
  images before review publication.
- Closed the complete image inventory rather than only proposal-bearing rows:
  a hidden second inference image with zero candidates is rejected, and every
  prepared genesis image must contribute at least one Stage-20 proposal.
- Added cumulative raw-feature archives so later image-rounds retrain from
  genesis plus all prior and current reviewed rows without rerunning SAM2 or
  feature extraction on earlier images. Physical and logical archive hashes
  are closed into the ancestry.
- Added fresh full XGBoost retraining after each finalized image review. The
  genesis test split and PCA remain frozen; the prototype is recomputed from
  accumulated outer-train positives. New image groups are train-only.
- Added project model labels `project-r1`, `project-r2`, and so on, with
  lineage binding the parent bundle, round image, genesis split, genesis
  Stage-20 feature hash, cumulative archive, effective train groups, and
  frozen test groups. Previously seen image bytes are rejected by SHA-256 even
  when the file has been renamed or assigned a different candidate group name.
- Extended round review validation to the sealed image-round v3 schema. The
  reviewer remains decision-only and never trains or resumes automatically.
- Clarified that `compag-cj-r92` is historical Stage-20 transfer assistance,
  not a bundled historical training corpus or r1-r92 decision chain. A new
  `project-r1` must not be called `r93` without the genuine historical lineage.
- Added `docs/SEQUENTIAL_IMAGE_ACTIVE_LEARNING.md` and updated current 1.8.0
  commands, architecture, reproducibility, review, citation, and release
  documentation. Historical changelog entries remain unchanged.

## 1.7.2 - 2026-09-01 (locally accepted; remote publication pending)

- Corrected a fail-closed false positive in canonical XGBoost UBJ privacy
  validation. UBJ typed arrays contain arbitrary numeric bytes, so v1.7.2 no
  longer interprets incidental binary byte sequences as filesystem locators.
  The validator still requires a loadable byte-identical canonical UBJ
  round-trip and recursively checks decoded XGBoost textual metadata while
  retaining the closed attribute, feature-name, feature-type, objective, and
  single-output contracts.
- Added regression coverage for valid canonical UBJ numeric payloads and for
  private locators in decoded semantic metadata. This is a serialization
  validator correction; it does not change proposal generation, feature
  extraction, group splitting, XGBoost training defaults, thresholds, review
  decisions, or the Full/Lite scientific profile boundaries.
- Bumped the package, runtime implementation, release metadata, source ZIP,
  wheel/sdist, installer, and current Conda-lock contracts to 1.7.2. A run
  bound to the v1.7.1 package implementation must not be resumed by bypassing
  dependency identity checks; its finalized review CSV may be reused only
  after a new v1.7.2 Stage-20 review request passes exact validation.
- Retained the byte-exact v1.7.1 `LOCAL_TRANSFER_ONLY` Conda/CUDA cache as an
  explicitly identified predecessor artifact. The v1.7.2 convenience wrapper
  verifies that outer artifact and its exact 71-package closure, then records
  the predecessor and normalized current-lock identities. The source
  installer remains strict to its canonical v1.7.2 Conda manifest. The cache
  is not renamed, republished, or added to the public GitHub artifact set.
- Completed final local v1.7.2 source/package/artifact acceptance: public
  contracts `29/29` in a fresh process, the remaining source suite `378/378`,
  installed-wheel exact CUDA/XGBoost checks `13/13`, and the online-installer
  suite `56/56`, all without skips. Twine passed; source/tree/wheel/sdist and
  installer-template privacy scans covered 6 files and 153903 bytes with zero
  findings; independent
  source-ZIP, wheel, sdist, and online-wrapper builds were byte-identical; and
  wrapper verify-only passed. Remote GitHub CI, tag, and Release remain
  pending. A new real end-to-end GPU science/System-B run is not claimed.

## 1.7.1 (release candidate; acceptance pending)

- Replaced the two mutable Conda solve steps with one canonical explicit lock
  for the complete 71-package CPython/CUDA closure. Every package name,
  version, build, subdir, official URL, byte size, and SHA-256 is now bound and
  the installed `conda-meta` closure is attested before pip or doctor runs.
- Added a release-bound Verified Conda/CUDA Cache route for networks where an
  official Conda/NVIDIA endpoint repeatedly returns a transient failure such
  as HTTP 503. Cache mode re-verifies all 71 archives and uses Mamba
  `--offline --always-copy`; the online mode uses the same lock, one download
  thread, and bounded retries. Neither path runs an automatic cache clean.
- Kept the large third-party cache outside the public GitHub installer and
  marked the private transfer artifact `LOCAL_TRANSFER_ONLY` pending a
  separate redistribution-license review. pip, SAM2 source, and model assets
  remain online acquisitions, so the cache is not described as full offline.
- Replaced tile-by-tile initial review and cropped round review with a shared
  full-image scene in both the loopback Web application and native Ubuntu/WSLg
  application. Initial scenes reconstruct the complete prepared card from the
  sealed Stage-10 tiles; round scenes show the complete original inference
  image.
- Added mask-derived proposal outlines as the default selection layer. Added
  bounding rectangles as an independent display-only option that is off by
  default. Zoom, pan, selection, and either geometry toggle leave proposal
  identity and scientific outputs unchanged.
- Defined three explicit review workflows. Project-1/CJ Stage-20 review uses
  the manuscript-selected r92 XGBoost model as historical transfer assistance,
  with a fixed 0.50 threshold, CUDA-only scoring, uncertainty-first ordering,
  and a hash-closed UBJ/JSON/NPY 93-feature closure. Cold-start Stage-20 review
  has no fitted classifier and remains exhaustive. Project-trained round review
  still requires a verified `compag-curation-model-bundle/v2`, complete
  Stage-60 scoring, and a sealed `begin-round` hard-case selection.
- Added an explicit, reversible **Confirm remaining XGB suggestions** action
  only for Project-1/CJ. Manually corrected rows remain weight 1.0 with
  `HUMAN_EXPLICIT` provenance; confirmed remaining suggestions use
  `sus_accept`/`sus_flip`, weight 0.4, and
  `PUBLISHED_MODEL_BULK_CONFIRMED` provenance.
- Added presentation-only CJ/Non-CJ, Target/Non-target, and custom binary class
  names. Durable labels remain `1`/`0`, and the established decision action,
  weight, review CSV, event journal, finalization, and resume contracts are
  unchanged.
- Expanded the installed master launcher to choose Project-1/CJ r92 assistance,
  model-free cold start, or an existing project-trained hard-case round, then
  native Application or private loopback Web. Retained a direct sealed-round
  launcher. No path silently auto-labels, resumes, or trains.
- Kept the legacy pickle and private annotation state outside the runtime. The
  narrowly scoped r92 transfer preset contains only reviewed, hash-bound safe
  UBJ/JSON/NPY assets and never imports arbitrary legacy models or zero-fills
  missing features.
- Retained the Full and Lite scientific profiles unchanged; this review-layer
  release makes no new numerical-equivalence, biological-performance, or
  real-GPU-acceptance claim. Final v1.7 source/package/review-UI/privacy/
  determinism acceptance and remote publication remain pending.

## 1.5.0 (locally accepted; publication pending)

- Added the supported `review-ui web`, `review-ui desktop`, and
  `review-ui status` commands for both an initial run paused after exactly
  Stage 20 and a sealed iterative active-learning `begin-round` selection. Web
  mode uses a standard-library loopback-only server, with Pillow from the
  locked GPU profile used for verified round visuals; desktop mode is a native
  Ubuntu/WSLg Tk/Pillow application. Both workflows share one verified
  decision/session interface while retaining source-specific hash bindings.
- Replaced manual CSV editing in the primary user workflow with five explicit
  human decisions: Target confident `(label=1, accept, 1.0)`, Non-target
  confident `(label=0, flip, 1.0)`, Target uncertain `(label=1, sus_accept,
  0.4)`, Non-target uncertain `(label=0, sus_flip, 0.4)`, and Skip `(label=0,
  skip, 0.0)`. Added keyboard navigation, proposal selection, zoom/pan,
  proposal-prefix search, state/group filters, auto-next navigation, and undo.
- Added external durable review state bound to the verified paused run, Stage
  20 receipt, review request, proposal table, overlay manifest, visual order,
  output path, and decision contract. Every set/undo is an append-only,
  hash-chained private event; closing either UI preserves completed decisions.
- Added round-review binding to the exact sealed `begin-round` operation root,
  its exact Stage-60 input, and the original inference-image directory. Round
  visuals are verified original-image crops with the Stage-60 polygon and
  bounding box; they do not reconstruct a Stage-20 mask overlay.
- Added no-clobber native Ubuntu launchers for initial review and round review:
  `START_COMPAG_REVIEW_UBUNTU` and `START_COMPAG_ROUND_REVIEW_UBUNTU` shell and
  desktop entry pairs.
- Added single-writer session locking and a shared immutable run snapshot so
  review and scientific resume cannot race. The reviewer never writes inside
  the run or changes `review_request.csv`.
- Added fail-closed finalization: every proposal requires an explicit human
  action, the complete CSV and group-split feasibility are validated, and the
  reviewed output plus final receipt are published without replacement. No UI
  predicts labels, bulk-labels, finalizes automatically, or invokes resume.
- Hardened web mode with loopback-only binding, a per-process capability token,
  exact Host/Origin checks, bounded strict JSON requests, verified overlay-only
  serving, no-store responses, and a restrictive nonce-based Content Security
  Policy.
- Preserved review compatibility for structurally valid v1.4 paused runs while
  retaining scientific environment binding: a reviewed v1.4 run must still be
  resumed through its unchanged v1.4 package/dependency environment.
- Retained the v1.4 Full/Lite GPU profiles and scientific method contracts.
- Completed v1.5 local acceptance: public QA `141/141`, the complete locked
  science/UI partition `228/228`, mocked-GPU CI coverage `111/111`, the focused
  human-review suites `58/58`, installed-wheel and independently extracted
  command matrices `78/78` each, zero privacy findings, and byte-identical
  repeated wheel, normalized-sdist, and public-source-ZIP builds. Remote CI,
  the immutable tag, and the GitHub Release remain pending.

## 1.4.0 (locally accepted; publication pending)

- Added one explicit dual-profile GPU installation surface to the same source
  tree and wheel. `tools/install_gpu_profile.py` prints a no-write plan by
  default and requires `--execute` plus an explicit `lite` or `full` preset to
  create the exact shared `science-gpu` environment.
- Kept Full as immutable `canonical-xgb-recall-gpu-v1` with SAM2.1 Hiera
  Large and sealed execution microbatch 32. Added Lite as immutable
  `efficient-xgb-recall-gpu-v1` with SAM2.1 Hiera Tiny and sealed execution
  microbatch 16. Both are CUDA-only for SAM2, ResNet50, XGBoost
  training/scoring, and supported inference; bounded preprocessing and
  postprocessing may continue on CPU without becoming a CPU model fallback.
- Added friendly `full` and `lite` CLI input aliases for project
  initialization and profile asset selection. Stored configurations, receipts,
  bundles, outputs, and provenance retain immutable profile IDs.
- Bound model assets, bundle contents, run/resume state, and active-learning
  lineage to the selected profile. Full and Lite scientific outputs, trained
  models, bundles, and resume artifacts are not interchangeable, and Lite does
  not claim canonical or numerical equivalence to Full.
- Retained one hash-locked GPU dependency closure and pinned SAM2 build for
  both choices; Lite is an asset/runtime preset rather than a separate or
  reduced-integrity dependency package.
- Accepted both profiles from the installed wheel in the exact environment on
  an RTX 4090. Lite passed in 5:53.26 with 2,252,164 KiB maximum process RSS,
  3,503 MiB sampled maximum total GPU memory, and 74% maximum sampled GPU
  utilization. Full passed in 6:36.10 with 3,108,076 KiB maximum process RSS,
  4,650 MiB sampled maximum total GPU memory, and 88% maximum sampled GPU
  utilization. Both exercised pause/resume, bundle creation, evaluation, and
  fresh inference on `cuda:0` with CPU fallback disabled.
- Completed current-source local QA: public QA `141/141`, mocked GPU tests 110
  passed with one expected skip, installed and extracted command matrices
  `74/74` each, zero privacy findings, and passing deterministic artifact
  checks. Final artifact rebuilding, the exact remote commit/CI, immutable tag,
  and GitHub Release are still pending; the package has not been published.
- Recorded that the exact shared science-gpu doctor passed on an RTX 1000 Ada
  6 GB laptop with driver 580.173.02. This is dependency compatibility only:
  no Lite E2E laptop acceptance, automatic VRAM threshold, or cross-device
  memory/performance guarantee is claimed.

## 1.3.0

- Replaced the supported canonical science surface with the fail-closed
  `canonical-xgb-recall-gpu-v1` profile on exact CPython 3.12.7 and Linux
  x86_64 with an NVIDIA GPU; canonical execution never silently falls back to
  CPU.
- Locked PyTorch `2.13.0+cu132`, torchvision `0.28.0+cu132`, CuPy `14.2.0`,
  CUDA-enabled XGBoost `2.1.1`, and SAM2 commit
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`, built with CUDA 13.2 Update 2
  `nvcc` 13.2.86 and the required CUDA development headers.
- Routed SAM2 proposals, ResNet50 embeddings, canonical XGBoost training and
  scoring, and supported canonical model inference through CUDA while keeping
  bounded image I/O, card/grid preparation, review/provenance, serialization,
  and reporting at explicit CPU preprocessing/postprocessing boundaries.
- Kept the frozen AMG method and bundle at 64 points per side and batch 512,
  while sealing Stage 20, Stage 60, and standalone inference to an
  execution-only 32-point CUDA microbatch. Non-M2M transient `low_res_masks`
  are discarded before upstream aggregation when they cannot be consumed. The
  full prompt lattice remains evaluated, with no user knob or CPU fallback.
- Reduced locked-host Stage-20 peak total device use from 24,018 MiB at method
  batch 512 to 4,441 MiB at execution microbatch 32 (837 MiB baseline; 3,604
  MiB incremental), with a repeat at 4,431/3,594 MiB. The final full demo
  passed in 6:28 at 4,913 MiB peak; Stage 50 peaked at 1,427 MiB and standalone
  inference at 4,431 MiB. Against the old-512 host baseline, one of 528 masks
  changed by one pixel, seven SAM2 IoU values by 1e-6, and two stability scores
  by approximately 2.65e-4. The new-32 Stage-20 baseline was byte-stable across
  two runs, and full Stage-60/fresh/repeat inference was byte-identical. These
  are RTX 4090 host measurements, not laptop acceptance or a memory guarantee.
- Canonicalized selected SAM2 confidence evidence to six decimal places only
  after raw-score ordering, deduplication, and the 500-mask cap. These scores
  are archival metadata, not members of the 93-feature predictor matrix.
- Added executable GPU runtime identity and deterministic-FP32 policy checks,
  CUDA device/driver provenance, CUDA-enabled XGBoost build verification, and
  GPU-profile model-bundle provenance.
- Bound PEP 610 archive enumeration to the active interpreter's attested
  top-level installation roots. Vendored metadata added to `sys.path` by an
  imported package is excluded, while physically distinct duplicate installs
  still fail closed.
- Made every documented environment-creation path set exact `umask 0022`
  before directory creation or installation, so a caller inheriting common
  collaborative `umask 0002` cannot produce group-writable package files or
  Conda/Mamba history. GPU `doctor` now includes the underlying installation-
  attestation diagnostic while retaining the same fail-closed permission
  policy.
- Released the transient CUDA execution-probe tensor immediately after its
  synchronized check, including when PyTorch retains an importing caller frame,
  and explicitly cleared the pinned Torch runtime's persistent cuBLAS
  workspaces before cache eviction so the parent allocator can be proven empty
  before fresh-process inference.
- Confined CUDA, CuPy, TorchInductor, and Triton caches to the nonempty owned
  fresh-process runtime root. After successful child validation, a fail-closed
  cleanup validates the complete closed tree, removes generated cache files,
  and retains only the explicit environment and allocator-release evidence;
  failed runs retain their structured diagnostics without publishing opaque
  accelerator caches. This prevents compiler-generated or
  host-account-derived cache names from entering Stage 60 and real-demo output.
- Retained the v1.2.0 `science-cpu` lock and
  `canonical-xgb-recall-cpu-v1` only as historical provenance; they are not
  supported v1.3.0 surfaces because the historical lock contains a Torch
  version affected by a security advisory.
- Added mocked/static GPU contract coverage to hosted CI without presenting a
  CPU-only hosted runner as real NVIDIA execution or release GPU acceptance.
- Closed the hash-locked bootstrap environment by adding `packaging 26.3`, the
  runtime dependency required by `wheel 0.48.0`, so fresh base installs pass
  `pip check` without relying on undeclared environment state.

## 1.2.0

- Added the opt-in `canonical-xgb-recall-cpu-v1` profile with manuscript-bound
  Hiera Large AMG settings, Ultra features, explicit ResNet50 weights, PCA32,
  group-aware training defaults, fixed 0.50 inference threshold, and YOLO off.
- Retained `public-safe-balanced-v1` and its generated configuration as the
  backward-compatible default.
- Added profile-selective project initialization and explicit registries for
  the Hiera Large checkpoint/configuration, ResNet50 weights, and license text.

## 1.1.0

- Added the supported `public-safe-balanced-v1` CPU execution path from images
  through review pause/resume, group-separated training, portable model bundle,
  fresh inference, point coverage, and report.
- Added generated quick and real demonstrations, explicit public asset
  acquisition, versioned user-data/review/annotation schemas, atomic run state,
  drift rejection, and full provenance.
- Preserved the eight legacy workflow commands and 21 validation/planning
  variants unchanged.
- Added clean-install documentation, pinned CI actions, privacy/license/SBOM
  contracts, and durable repository-identifier-neutral release metadata free of
  an asserted project-repository URL.
- Canonicalized SAM2 scores to six decimal places and made proposal identities
  geometry-and-mask based so supported fresh-process inference is byte-stable.
- Enforced and bundle-sealed a single-thread Torch/OpenCV/native numerical
  runtime policy so supported wheel and source executions produce identical
  proposal features and trained model bytes.

## 1.0.0 - 2026-08-26

- Prepared the first notebook-independent, GitHub-ready source release
  candidate.
- Organized eight workflow groups and 21 variants behind typed configuration,
  validation, inert planning, and importable Python handler interfaces.
- Added the `migrate-feature-csv` command with `PRESERVE_BACKUPS` behavior
  for `.pre_scale.bak.csv` and `.pre_reviewtag.bak.csv`.
- Added public documentation, software citation metadata, explicit license
  scope, community files, and read-only CI.
- Replaced legacy lexical acceptance gates with workflow capability and
  operational-cell traceability.
- Performed release QA without scientific execution. Scientific and behavioral
  parity were `NOT_RETESTED_BY_DESIGN`.

No repository URL or DOI is asserted in the neutral 1.0.0 source metadata.
