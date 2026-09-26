> **JR09 public-code candidate boundary (2026-09-24).** The fitted Project-1/CJ r92 model closure is research material and is **not included** in this public-code candidate. Any inherited statement below calling that preset "bundled" or "public" describes the frozen private 1.9.3 build and is superseded for this candidate. The r92-assisted lane requires the separately controlled reviewer asset. Application/scientific algorithms in `src/` are unchanged. This is a local candidate pending repository confirmation and asset-specific rights review, not a deployed release.

# Known Limitations

- Version 1.9.3 is the strict post-r92 sequential-image active-learning release
  candidate. The recorded installed-wheel GPU acceptance below was
  performed on v1.4 and remains historical evidence for the unchanged
  scientific profiles; it is not a new v1.9.3 real-GPU rerun. Historical v1.7.2
  local source/package/review-UI/privacy/determinism and artifact QA passed;
  equivalent v1.9.3 acceptance is not yet claimed.
  Upload, remote GitHub CI, tagging, and release-page creation remain pending;
  a real v1.9.3 end-to-end GPU science/System-B run is not claimed; an
  earlier-version System-B result is not acceptance of the image-round code.
  In particular, a v1.8.1 smoke run that patches reviewer role strings or the
  omitted round-number sentinel is diagnostic only and is not native 1.9.3
  reviewer acceptance.
- Base and quick are supported on CPython 3.12 on Linux x86-64. All three GPU
  profiles are narrower: exact CPython 3.12.7, Linux x86-64, an NVIDIA CUDA
  GPU, and the same locked CUDA 13.2 environment.
- CPU-only scientific execution, non-NVIDIA accelerators, containers, YOLO,
  remotely exposed or multi-user review hosting, non-Ubuntu native desktop
  review, and arbitrary SAM2 configurations are not supported release
  surfaces. The supported web reviewer is loopback-only; the native reviewer
  targets Ubuntu/WSLg. GPU preflight fails closed without CPU fallback.
- Iterative review requires the exact whole sealed `begin-round`,
  `begin-image-round`, or `begin-transfer-image-round` operation root, its exact
  bound Stage-60 inference root,
  and the original inference-image directory; copied or substituted inputs are
  rejected. It requires a verified `compag-curation-model-bundle/v2`. The full
  original image displays Stage-60 mask-derived outlines, but only the sealed
  hard-case shortlist is selectable. Other detections and model values are
  context, not labels.
- One-image scheduling is an operational contract, not a claim that the
  manuscript defined one image as one active-learning round. A transfer
  image-round reviews and trains on only deterministic Top-K (`K=50`, or all
  eligible rows when fewer qualify) selected in-margin proposals.
  Unselected proposals from that image do not enter training, and the same
  image bytes or group cannot re-enter a later image-round to obtain another
  batch. The workflow always emits `project-rN` and does not emit `r93`.
- The v1.9.3 transfer baseline is usable only when the user separately supplies
  the exact 582,968,556-byte source whose SHA-256 is
  `1836d704d15f27728e3ba09d7cddf0aea9458049bc66700dc98f14e7111ec29d`.
  Neither that source nor its imported baseline is publicly bundled. The
  importer creates a deterministic transfer split, but does not reconstruct an
  unavailable historical r92 split or complete decision trajectory.
- Initial Stage-20 review is exhaustive because no fitted classifier exists at
  the model-free cold-start boundary. Full-image display improves card-level
  context but does not reduce the number of explicit human decisions.
  Genesis creation fails closed if any prepared input image has no retained
  Stage-20 proposal, so every genesis image remains inside the sealed identity
  boundary used to reject later duplicate-image rounds. This guarantee covers
  Stage-20 emitted by the canonical producer; a deliberately re-sealed
  standalone Stage-20 artifact without its Stage-10/Stage-00 inventory
  ancestry is outside the supported sequential-lineage provenance boundary.
  Mask-derived outlines default on and optional bounding rectangles default
  off in both graphical modes. Neither graphical mode silently predicts,
  automatically finalizes, resumes scientific execution, or trains.
- Project-1/CJ transfer scoring is a narrow exception, not a generic
  pretrained-model surface. It requires canonical Full, fixes CJ/Non-CJ and
  threshold 0.50, and uses the bundled paper-selected safe r92 closure. Its
  uncertainty ordering does not establish fresh-canonical equivalence or
  performance on new data. The separate initial Stage-20 reviewer may still
  use its explicit remainder confirmation: bulk-confirmed suggestions are
  reduced-weight 0.4 `sus_*` rows with distinct provenance and are reversible
  as one operation before finalization. The strict transfer image-round
  reviewer has no bulk-confirm action; every selected proposal requires an
  explicit decision. Lite and custom labels reject this preset.
- Transfer retraining freezes the published r92 prototype and PCA rather than
  refitting them. Its CV and projection-variance outputs are therefore
  conditional on that frozen feature state and must not be interpreted as a
  historical r92 fit reconstruction. The formal claim is
  `NOT_R92_REPRODUCTION`; `project-r1` is not manuscript `r93`.
- CJ/Non-CJ, Target/Non-target, and custom class names are presentation-only;
  COMPAG remains a binary `1`/`0` workflow with the same decision actions and
  weights. It is not a general multiclass annotation application.
- The quick demo does not run SAM2 or XGBoost. The real demo uses synthetic
  pixels and deterministic non-human review fixture labels.
- The balanced compatibility profile and canonical CPU profile are historical
  structural verification contracts, not v1.9.3 execution profiles. The Full
  GPU canonical profile's supported method/default/contract comparison is closed
  with exact or enumerated bounded adaptations, including the disclosed GPU
  numerical deviation. It trains new
  user-owned state; unavailable original data, fit-time split, fitted transforms,
  classifier, and predictions prevent historical fitted-model or paper-result
  parity.
- Lite uses Hiera Tiny and execution microbatch 16 to reduce model/asset and
  execution footprint. It does not claim scientific, numerical, or canonical
  equivalence to Full. Full-image is an experimental Hiera Large mode with no
  external/persisted tiling. It uses SAM2's internal overlap crop pyramid
  (`points_per_side=64`, configured batch 512, execution microbatch 8, crop
  layers 2, downscale 1, overlap `512/1500`, cap 1000 proposals/image). These are
  an initial operating point, not an optimization or equivalence claim. Lite,
  tiled Full, and Full-image configurations, proposals, ROI features, trained
  state, bundles, outputs, resume records, and active-learning lineage are not
  interchangeable. Tiled r92 and tiled bundles cannot initialize Full-image;
  it must train and continue within its own profile.
- The Full-image crop-pyramid preset is an initial operating point, not a
  general optimum. On the single engineering image `IMG_9475.jpg`, two crop
  layers/downscale 1 retained 884 proposals, reached 98.56% reference-point
  coverage at IoU 0.25 and 90.65% at IoU 0.50 (median best IoU 0.875), took
  144.3 seconds, and peaked at 2.86 GiB allocated / 5.91 GiB reserved CUDA
  memory. A three-layer run reached 92.09% at IoU 0.50 only with the
  1000-proposal cap binding and took 481 seconds. The +1.44 percentage-point
  change at more than 3.3 times the elapsed time was not retained. Neither
  result establishes biological accuracy, dataset-wide performance, or
  equivalence to the tiled method or manuscript.
- Original research images, annotations, review rows, predictions, and private
  scientific evidence are not bundled. The sole fitted-state exception is the
  safe portable `compag-cj-r92` transfer-assist closure and its disclosed
  frozen feature state.
- Legacy pickle/joblib classifiers and loose historical model state are not a
  supported model-assisted shortcut. The r92 exception never bundles or loads
  the legacy pickle; arbitrary project-assisted use requires a newly written
  and verified `compag-curation-model-bundle/v2` with its exact feature order.
- Point coverage is recall-style polygon-preferred, bounding-box-fallback
  coverage and is not instance precision or a general detection metric.
- Full, Lite, and Full-image execution require at least six independent training groups and a
  feasible five-fold group search; its outer train/test and inner early-stop
  partitions require both labels. The retained balanced structural contract
  records its historical four-group train/validation/test requirement but is
  not executable in v1.9.3.
- Reproducibility is bound to exact supported dependencies and assets. Results
  on other platforms or versions are outside the claim.
- The v1.2 `science-cpu` lock is historical provenance only and is unsupported
  in v1.9.3 because its Torch line is covered by a security advisory.
- The generated quick demo is the only supported standalone CPU computation.
  CPU work inside any GPU profile is limited to documented bounded pre/postprocess
  boundaries and is not a CPU model fallback.
- GitHub-hosted CPU CI exercises mocked/static GPU contracts, not a real GPU
  acceptance or performance workload.
- Historical local v1.4 installed-wheel acceptance on an RTX 4090 passed for Lite in
  5:53.26 at 2,252,164 KiB maximum process RSS, 3,503 MiB sampled maximum total
  GPU memory, and 74% maximum sampled utilization, and for Full in 6:36.10 at
  3,108,076 KiB maximum process RSS, 4,650 MiB sampled maximum total GPU memory,
  and 88% maximum sampled utilization. Both passed pause/resume, bundle,
  evaluation, and fresh inference on `cuda:0` without CPU fallback. These
  bounded synthetic exact-host observations validate the retained scientific
  profiles, are distinct from the completed historical v1.7.2 artifact acceptance, and are
  not guarantees across input sizes, proposal counts, drivers, concurrent GPU
  contexts, or other devices.
- The final locked v1.3 RTX 4090 Full demo passed in 6:28 at 4,913 MiB peak
  total device use. Stage 20 at execution microbatch 32 repeated at 4,441 and
  4,431 MiB total, and standalone inference reached 4,431 MiB. Those figures
  remain historical v1.3 evidence and are not the v1.4 measurements above.
- Execution microbatch 32 preserves method/bundle batch 512 and the full prompt
  lattice, but it is not claimed bit-identical to old batch-512 SAM2 execution
  across hardware: the locked-host comparison changed one of 528 masks by one
  pixel, seven IoU values by 1e-6, and two stability scores by approximately
  2.65e-4. The selected new-32 baseline was repeat-byte-stable, and the complete
  Stage-60/fresh/repeat inference outputs were byte-identical.
- The Full-image memory-bounded backend was benchmarked on one RTX 4090 with a
  4032x3024 input at execution microbatch 8: observed Torch memory peaked near
  2.8 GiB allocated and 6.6 GiB reserved. This is narrow implementation
  evidence, not a complete v1.9.3 end-to-end acceptance, quality result,
  equivalence result, capacity guarantee, or automatic VRAM-selection rule.
- The exact shared science-gpu doctor passed on an RTX 1000 Ada 6 GB laptop
  with driver 580.173.02. The Lite end-to-end demo has not yet run on that
  laptop, so this is dependency compatibility evidence only. No automatic
  VRAM selection threshold or exact Lite/Full/Full-image VRAM or performance guarantee is
  claimed.
- No repository URL, DOI, performance guarantee, or exact paper-result
  reproduction is asserted by the neutral artifact.
