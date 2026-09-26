> **JR09 public-code candidate boundary (2026-09-24).** The fitted Project-1/CJ r92 model closure is research material and is **not included** in this public-code candidate. Any inherited statement below calling that preset "bundled" or "public" describes the frozen private 1.9.3 build and is superseded for this candidate. The r92-assisted lane requires the separately controlled reviewer asset. Application/scientific algorithms in `src/` are unchanged. This is a local candidate pending repository confirmation and asset-specific rights review, not a deployed release.

# Compatibility And Capability

COMPAG curation is intentionally narrow. A successful structural validation
does not establish that arbitrary imagery is scientifically compatible.

| Input or capability | Status | Public contract |
| --- | --- | --- |
| Binary target-versus-other proposal review, training, and inference | SUPPORTED | Review labels are `0` or `1`; the classifier is not a general multiclass learner. |
| Compatible yellow sticky-card images | SUPPORTED_WITH_CONSTRAINTS | The full card should be visible, upright with its long axis vertical, and sufficiently yellow under OpenCV HSV; moderate perspective is corrected only when a four-corner card contour is recovered. |
| PNG, JPEG, P3/P6 PPM | SUPPORTED_WITH_CONSTRAINTS | 8-bit, 64 MiB maximum encoded size, 64x64 minimum, 16384 maximum per side, 100 million decoded pixels maximum; JPEG EXIF orientation must be absent or 1. |
| Existing box, polygon, mask, or class-label datasets | UNSUPPORTED | No tested direct-ingestion adapter is included. Images enter proposal generation; classifier labels enter through the canonical review CSV. |
| CVAT point XML and canonical point JSON | EVALUATION_ONLY | Points measure recall-style coverage for configured inference images. They never train the classifier and do not establish one-to-one instance precision. |
| Initial Stage-20 human review | SUPPORTED_UI_AND_CSV | A loopback-only Web application and a native Ubuntu/WSLg Tk/Pillow application share one durable engine and full-card scene. An exactly paused Stage-20 run is a cold start, so every proposal requires an explicit human decision before the strict reviewed CSV can be finalized for explicit resume. Direct preparation of a separate compatible CSV remains supported. |
| Iterative active-learning round review | SUPPORTED_UI_AND_CSV | The same loopback Web and native Ubuntu/WSLg interfaces accept an exact sealed `begin-round`, `begin-image-round`, or `begin-transfer-image-round` operation root together with its exact Stage-60 input and original inference-image directory. They show the full image but make only the sealed hard-case shortlist selectable, then publish the strict reviewed CSV required by the matching explicit resume command. |
| Strict post-r92 one-image active learning | SUPPORTED_SOURCE_CONTRACT_GPU_E2E_PENDING | v1.9.3 accepts exactly one unseen image, uses published r92 scoring in round 1 and the immediately prior `project-rN` later, reviews deterministic Top-K (`K=50`, or every eligible row when fewer qualify) inside margin 0.2, and fresh-fits over the exact private transfer baseline plus cumulative reviewed rows. The baseline is separately supplied and SHA-256/size gated, never public. The claim is `NOT_R92_REPRODUCTION`; `project-r1` is not `r93`. |
| Retained genesis-based one-image active learning | SUPPORTED_SEPARATE_CONTRACT | The v1.8.0 contract starts from a project-created genesis, accepts exactly one unseen image, reviews at most 50 in-margin proposals, and full-retrains `project-rN` from genesis plus cumulative reviewed rows. Its ancestry must not be mixed with strict transfer rounds. |
| Safe model-assisted review | SUPPORTED_EXPLICIT_BOUNDARIES | Project-1/CJ Stage-20 assistance accepts only the bundled hash-closed r92 UBJ/JSON/NPY preset with canonical Full and fixed threshold 0.50. Project-trained round assistance separately requires a verified `compag-curation-model-bundle/v2`, complete Stage-60 scoring, and a sealed general-pool or image-round selection. Arbitrary pickle/joblib classifiers, loose historical models, feature-order substitution, and hidden automatic labels are unsupported. |
| Linux x86-64 CPU, CPython 3.12 | SUPPORTED_BASE_QUICK_ONLY | Base validation/planning and the synthetic quick demo run on CPU; they are not canonical scientific execution. |
| NVIDIA GPU Full, Linux x86-64, CPython 3.12.7 | SUPPORTED_EXACT_PROFILE_RETAINED_ACCEPTANCE | `full` / `canonical-xgb-recall-gpu-v1` uses Hiera Large and CUDA microbatch 32 with the exact `science-gpu` closure. The unchanged scientific profile passed the documented v1.4 installed-wheel run on the RTX 4090 host, and historical v1.7.2 local source/package/artifact QA passed. A new v1.9.3 real end-to-end GPU science/System-B run is not claimed. |
| NVIDIA GPU Lite, Linux x86-64, CPython 3.12.7 | SUPPORTED_EXACT_PROFILE_RETAINED_ACCEPTANCE | `lite` / `efficient-xgb-recall-gpu-v1` uses Hiera Tiny and CUDA microbatch 16 with the same exact `science-gpu` closure. The unchanged scientific profile passed the documented v1.4 installed-wheel run on the RTX 4090 host, and historical v1.7.2 local source/package/artifact QA passed. A new v1.9.3 real end-to-end GPU science/System-B run and canonical equivalence are not claimed. |
| NVIDIA GPU Full-image, Linux x86-64, CPython 3.12.7 | SUPPORTED_EXPERIMENTAL_GPU_E2E_PENDING | `full-image` / `full-image-multiscale-xgb-recall-gpu-v1` keeps one lossless native processing unit per image instead of external/persisted tiles. It uses Hiera Large, SAM2's internal overlap crop pyramid (`points_per_side=64`, configured batch 512, execution microbatch 8, crop layers 2, downscale 1, overlap `512/1500`, cap 1000 proposals/image), and the exact shared `science-gpu` closure. Source-level and focused memory-bounded backend tests passed; a complete installed-wheel/System-B scientific run and equivalence to either tiled profile are `NOT_CLAIMED`. |
| CPU-only scientific execution | UNSUPPORTED_V1_8 | The v1.2 `science-cpu` lock is historical provenance only and is unsupported because its Torch line is covered by a security advisory. |
| Balanced v1 / canonical CPU v2 artifacts | VERIFY_ONLY | Existing configs and bundles may be structurally loaded, inspected, and verified; v1.8 does not initialize, write, train, infer, or resume model-driven active learning with them. |
| macOS, Windows, or non-NVIDIA accelerator execution | NOT_TESTED | These are not accepted 1.9.3 scientific-execution platforms. The native reviewer is supported only on Ubuntu/WSLg, while the web reviewer remains loopback-only on the supported Linux installation. |
| Container execution | NOT_PROVIDED | No container image or Dockerfile was built and tested. This is nonblocking for the supported Linux installation. |
| YOLO-assisted proposals | UNSUPPORTED_RELEASE_PROFILE | YOLO is disabled and was not part of the canonical scientific execution. |

CPU work inside any GPU profile is limited to bounded boundaries such
as image decoding, card/grid preparation, file I/O, CSV/provenance handling,
and postprocessing without a supported GPU kernel. SAM2, ResNet50, XGBoost
training, and supported model inference are CUDA operations.
The generated quick demo is the only supported standalone CPU computation.

Full, Lite, and Full-image use one codebase, one wheel, and one dependency lock,
but they are not labels for the same scientific result. Profile identity is
sealed into configuration, bundles, outputs, resume state, and active-learning
lineage; cross-profile mixing fails. The Full canonical claim does not extend
to Lite or Full-image. In particular, tiled r92 and tiled bundles cannot seed a
Full-image run; Full-image must train from its own review and continue only with
the same profile.

The historical v1.4 installed-wheel synthetic real-backend acceptance host was an RTX
4090 in the exact CUDA 13.2/Torch cu132 environment. Lite passed in 5:53.26
with 3,503 MiB sampled maximum total GPU memory; Full passed in 6:36.10 with
4,650 MiB. Both passed pause/resume, bundle creation, evaluation, and fresh
inference on `cuda:0` with `allow_cpu_fallback=false`. These exact-host
observations do not create an automatic selection threshold or resource
guarantee.

The shared science-gpu doctor has passed on an RTX 1000 Ada 6 GB laptop with
driver 580.173.02. That demonstrates dependency compatibility on the observed
host, not Lite end-to-end acceptance. The Lite real demo has not yet run there,
and no automatic GPU-memory threshold or exact Lite/Full VRAM/performance claim
is currently published.

## Human-Review Interface Boundary

The v1.8 Web and native applications expose the same five-decision contract for
two verified source kinds. Initial review validates an immutable run paused
after exactly Stage 20. Tiled profiles reconstruct the complete prepared card
from sealed Stage-10 tiles; Full-image displays the sealed native full-image
processing unit and does not depend on external tiles. Iterative review
validates an exact sealed `begin-round`
or `begin-image-round`
operation, its exact Stage-60 inference input, and the original inference-image
inventory, then displays the complete original image. Mask-derived outlines
default on in both workflows; bounding rectangles are a separate optional
display layer and default off.

Project-1/CJ Stage-20 review separately verifies the bundled safe r92 closure,
requires canonical Full, fixes CJ/Non-CJ and threshold 0.50, and orders
proposals by uncertainty. Its optional remainder confirmation is explicit,
reversible, low-weight, and provenance-distinct from manual corrections.
Cold-start Stage-20 review is exhaustive because no fitted classifier exists.
Iterative model-assisted review requires a verified
`compag-curation-model-bundle/v2`,
Stage-60 scoring, and a sealed general-pool or image-round sequence; only sealed hard cases are
selectable, while all other detections and model values remain context. Both
workflows preserve `review_request.csv`, keep session state and output outside
immutable inputs, require one explicit human decision for every reviewable
proposal, and publish a separate reviewed CSV without replacement. The
applications never silently infer a human label, finalize an incomplete
review, resume scientific execution, or train. The r92 remainder operation is
unavailable outside Project-1/CJ.

CJ/Non-CJ, Target/Non-target, and concise custom class names are presentation
options only. They do not change binary `1`/`0` values, actions, weights,
proposal identity, or the strict CSV import contract.

Web mode binds only to loopback and uses a per-session token. Native mode uses
Tk 8.6 and Pillow on Ubuntu or WSLg. Neither interface is a remotely hosted,
multi-user annotation service. Both initial and round workflows retain a strict
CSV boundary: `run --resume`, `active-learning resume-round`, and
`active-learning resume-image-round` remain separate explicit commands.

## Card Warp Contract

The canonical Stage 10 implementation converts BGR to HSV, thresholds between
`[15, 40, 60]` and `[40, 255, 255]`, selects the largest yellow connected
contour, and applies `approxPolyDP` with epsilon `0.02 * perimeter`. Exactly
four recovered corners are ordered top-left, top-right, bottom-right,
bottom-left and perspective-warped to a rectangle.

If no yellow contour is found, the contour is not a quadrilateral, or the warp
raises an error, the implementation returns an identity transform and
continues. `identity_fallback` is a recorded fallback, not a statement that the
image is suitable. Severe shadows, glare, color casts, a clipped card boundary,
extreme perspective, or incorrect orientation can trigger or degrade it.

## Dashed Grid Contract

Grid localization operates on the warped image. It expects five horizontal and
three vertical dashed lines. Spacing scales from a reference warped card of
2173 by 3692 pixels: vertical-line spacing is `730 / 2173` of width and
horizontal-line spacing is `744 / 3692` of height, each with 15% tolerance.
The detector sweeps darkness percentiles 92, 90, 88, 85, 82, and 80 and border
fractions 5%, 4%, 3%, and 2%.

Unexpected failures return empty line arrays; a low-quality sweep can still
return its best candidate. The pipeline can therefore continue with weak or
empty grid evidence and fallback cell coordinates. This is deliberate runtime
robustness, not a compatibility gate. Users must inspect the evidence before
interpreting grid-derived features.

## Required Human Inspection

After the first run reaches `PAUSED_FOR_REVIEW`, inspect every image record in
`stages/10_prepare/prepare_summary.json`:

- Confirm `warp_mode` is appropriate and warped dimensions/orientation match
  the visible card.
- Confirm `row_lines` and `column_lines` align with the printed five-by-three
  dashed-line layout.
- Inspect Stage 20 `overlays/` and `overlay_manifest.json`; proposal labels are
  hash-bound to the lossless proposal rows.
- Reject the run if the card, grid, or proposals are visibly unsuitable. Make
  a corrected input copy and start a new output; do not modify a sealed run.

Only then launch `review-ui web` or `review-ui desktop` with external output
and state paths, make one explicit decision for every proposal, and finalize
the separate reviewed CSV. The applications preserve the five identity
columns and enforce the binary label/action/weight/status contract. An expert
may instead prepare a separate compatible CSV directly, but must never edit
the sealed `review_request.csv`. Resume explicitly with `--review-labels`; the
reviewer never resumes the run automatically. See
[Review Workflow](REVIEW_WORKFLOW.md) and [User Data Guide](USER_DATA_GUIDE.md)
for the full no-clobber sequence.

For an iterative round, pass the exact whole `begin-round --output` root, its
exact bound Stage-60 root, and the original inference-image directory to the
same reviewer. The complete original image and Stage-60 polygon context are not
a reconstruction of a Stage-20 lossless mask overlay; use explicit Skip with
zero weight when the available visual evidence does not support a reliable
decision. Finalize the separate CSV, then invoke `active-learning resume-round`
explicitly.

## Claim Boundary

The public fidelity closure covers method, defaults, and contracts for this
supported new-user-data implementation, including enumerated portability,
safety, and interface adaptations. It does not prove compatibility for inputs
outside this table, historical fitted-model or split identity, private-output
parity, biological robustness, or reproduction of paper results.
