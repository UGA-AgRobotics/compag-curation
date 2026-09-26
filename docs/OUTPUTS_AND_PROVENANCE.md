> **JR09 public-code candidate boundary (2026-09-24).** The fitted Project-1/CJ r92 model closure is research material and is **not included** in this public-code candidate. Any inherited statement below calling that preset "bundled" or "public" describes the frozen private 1.9.3 build and is superseded for this candidate. The r92-assisted lane requires the separately controlled reviewer asset. Application/scientific algorithms in `src/` are unchanged. This is a local candidate pending repository confirmation and asset-specific rights review, not a deployed release.

# Outputs And Provenance

A run output is an immutable sequence of stage directories plus an append-only
event ledger. Every completed stage contains `_SUCCESS.json` with its run ID,
stage name, member paths, sizes, modes, hashes, and manifest hash. Stage zero
records the normalized path-free configuration and structured invocation.

The model bundle contains a profile-specific closed schema. Both versions
contain:

- `classifier.ubj`, never pickle or joblib;
- the exact ordered feature contract and median imputer values;
- a profile-bound threshold contract;
- class map and fixed preprocessing/proposal configuration;
- dependency, Python, requested/resolved device, and exact numerical-runtime-policy compatibility;
- reviewed/proposal/split provenance hashes and row counts;
- `bundle.json`, whose self-excluding member list closes every payload byte.

Historical balanced v1 carries the 25-feature contract, validation-selected
threshold, and verified SAM2 Tiny assets. Historical canonical CPU v2 and the
supported GPU v2 profiles carry the exact 93-feature hash,
strict little-endian float32 prototype/PCA32 state, fixed method threshold
0.50, verified profile-specific SAM2 and ResNet50 assets, both license texts, and the
path-independent package implementation identity. Neither bundle carries
training rows or private fitted study state.

The v1.9.3 writer mints only GPU v2 bundles: tiled Full and Full-image contain
Hiera Large, while Lite contains Hiera Tiny. Balanced v1 and canonical CPU v2
bundles remain structurally loadable and verifiable, but they cannot authorize
new model execution. CPU v2 verification compares its recorded package closure
with the frozen v1.2 package identity rather than current v1.7 source bytes.

Fresh inference verifies all members and compatibility before loading XGBoost or
SAM2. Within the installed exact locked `science-gpu` runtime, its only workflow
inputs are raw images and the bundle; it does not require training tables,
review files, development source, notebook state, or a model cache. The bundle
does not include Python or scientific libraries and is not a substitute for
that verified runtime.

Stage-20 `proposal_config.json` and `proposal_summary.json`, and Stage-60 or
standalone `inference_result.json`, record the selected profile's actual fixed
CUDA execution microbatch: 32 for tiled Full, 16 for Lite, or 8 for Full-image.
The method and model
bundle retain `points_per_batch = 512`; the
execution-only value is not copied into the bundle as a configurable method
parameter. Discarded non-M2M transient `low_res_masks` are runtime intermediates
and are not output or provenance members.

For Full-image, Stage-20 `proposal_config.json` closes the exact
`proposals.csv` and `features.csv` SHA-256 values. It also binds durable mask
encoding `bbox-cropped-column-packbits-little-v1`, CSV transport encoding
`base64-packbits-little-column-major-bbox-v1`, and connected-component policy
`largest-8-connected-pixel-area-top-left-tie-v1`. Model-bundle training
revalidates the same encoding/component contract and the Stage-20 table hashes
before fitting.

GPU v2 compatibility binds Linux x86-64, CPython 3.12.7, the
science-GPU lock and deterministic FP32 policy, requested/resolved CUDA device,
Torch CUDA runtime, device index/name/capability, observable driver, and
XGBoost `USE_CUDA`. A mismatch fails before model work; the bundle never
authorizes CPU fallback.

The immutable profile also binds bundle member paths, asset hashes, spatial
mode, and feature semantics. Full, Lite, and Full-image bundles, outputs,
resume state, and active-learning lineage are not interchangeable; Lite and
Full-image do not claim canonical equivalence.

Full proposal IDs are domain-separated with
`compag-canonical-proposal-v2`; Lite proposal IDs use
`compag-efficient-sam2-tiny-proposal-v1`. Initial active-learning export
accepts the complete sealed Stage-20 root, revalidates its receipt,
configuration, hashes, and profile, and carries that profile into the genesis
pool/export/decision lineage. A loose `features.csv` cannot establish genesis,
and later cross-profile reuse fails closed.

Canonical active-learning operations are separate immutable output roots; they
never append to or reopen the sealed base run. Each root contains
`command_record.json`, `input_manifest.json`, `environment.json`, a `payload/`
tree, `OUTPUT_MANIFEST.json`, and a last-written `FINAL_STATUS.json`. Export and
selection operations seal `PAUSED_FOR_REVIEW` with exit code 3. Resume
operations publish `PASS` only after their decision log, retrain request,
verified v2 bundle, rescored pool, and completion record all exist. Input and
environment records carry roles, portable basenames, hashes, and sizes rather
than local paths. Under the retained v2 general-pool contract, a later pool
must retain every earlier reviewed active-learning candidate as well as its new
compatible candidates.

Sequential image-round selection/completion uses the v3 contract. It binds
exactly one unseen image and one at-most-50 uncertainty shortlist. Completion
writes `accumulated_al_features.csv`, containing only reviewed raw-feature rows
from prior and current image-rounds; unselected proposals do not enter that
archive. `round_completion.json` and the new bundle provenance bind the genesis
Stage-20 feature hash, frozen genesis split, parent bundle, round image/group
and image SHA-256, cumulative archive, effective train groups, frozen test
groups, and `project-rN` label. Matching image bytes are rejected after rename
and cannot enter another image-round.

The separate post-r92 transfer path introduced in v1.8.1 and retained in
v1.9.3 begins with a sealed local baseline
created only from the exact private SHA-256/size-pinned source. Round-1
inference additionally writes `transfer_scorer.json`, which binds the r92
resource manifest, classifier, frozen feature state, project config, asset
inventory, one-image inventory, predictions, and raw features. Transfer
selection/completion operation roots bind that baseline, its deterministic
split, the source scorer, the one image, the complete human-reviewed shortlist,
the cumulative transfer decision/feature archive, and the immediately
preceding completion for rounds after the first.

The published classifier/PCA/prototype resource may be public, but neither the
private source CSV nor the imported baseline is a public artifact. Transfer
bundles are labeled `project-rN` and state `NOT_R92_REPRODUCTION`; they are not
labeled `r93`.

## Human-Review State

The v1.8 review application is deliberately outside the run ledger. For final
output `reviewed.csv`, the default session root is
`reviewed.csv.review-session`; a different external root may be supplied with
`--state`. For initial review, `SESSION.json` binds the run ID, Stage-20
receipt, `review_request.csv`, overlay manifest, proposal table, proposal count
and visual order, CSV columns, final output path, and five-decision contract.
Project-1/CJ initial review additionally binds the exact safe r92 preset,
93-feature order, threshold, verified CUDA score table, uncertainty order, and
assistance cache; model-free cold start omits all fitted state.
`SESSION.LOCK` permits one session writer. The scientific run remains
shared-locked while the initial reviewer is open, preventing a concurrent
resume.

For iterative review, the session receipt instead binds the complete sealed
`begin_round`, `begin_image_round`, or `begin_transfer_image_round` operation
and manifest, round number/ID, exact Stage-60 directory
record and inference result, predictions, shortlist and request hashes, source
bundle/profile, complete original inference-image inventory, proposal/visual
order, output, and the same five-decision contract. The output and state must
remain outside the selection, Stage-60, and image inputs. A rendered round
visual is a bounded verified original-image crop with the Stage-60 polygon and
bbox; it is not a Stage-20 lossless mask overlay.

Each human decision or undo is a new private event under `events/`. Sequence
numbers are contiguous and each event records the prior event hash, creating an
append-only chain. Undo restores the previous decision by appending another
event; it never rewrites history. Auto-next, scores, and UI filters do not
silently create decisions. The only compact multi-row event is the explicit,
reversible Project-1/CJ remainder confirmation; it records low-weight uncertain
actions and distinct model-confirmed provenance. The session contains
human-review state and must
be protected as user data; it never becomes a member of the public repository
or sealed run.

`EVENT_HEAD.json` is a small durable commit pointer bound to `SESSION.json`,
the source identity, and visual order. It records the last committed sequence
and event hash. Opening a session replays the complete bounded event directory,
verifies contiguous names, hashes, previous-hash links, request IDs, and
set/undo state transitions, then compares the reconstructed chain with the
head. Because an event file is published and synced before the head advances, a
fully valid suffix with an older head is recoverable by verified replay and a
durable head advance. A head ahead of events, a gap, changed binding, malformed
transition, or hash mismatch is rejected. Finalization performs another
independent replay immediately before publishing the CSV.

After all rows have a decision, finalization stages a CSV and validates
immutable identity parity plus exact action/weight mappings. Initial Stage-20
review additionally validates split feasibility, and its `FINAL.json` records
that result. Round review validates the complete canonical table without the
initial split-feasibility gate. Both publish the requested output without
replacement; their `FINAL.json` records its hash, row/label/action counts,
effective weight, and terminal event-chain anchor. The source
`review_request.csv` remains unchanged.
Closing the UI does not create either final file and does not resume a run or
round. No reviewer path trains a model.

The five canonical mappings are Target confident = `(1, accept, 1.0)`,
Non-target confident = `(0, flip, 1.0)`, Target uncertain =
`(1, sus_accept, 0.4)`, Non-target uncertain = `(0, sus_flip, 0.4)`, and
Skip = `(0, skip, 0.0)`. `review_status` is written as `reviewed` only during final
CSV rendering.

A v1.7 reviewer may bind to a structurally compatible paused v1.4 or v1.5 run. The
review session and CSV do not change that run's scientific environment
provenance: resume must still use the unchanged v1.4 package/dependency closure
that created the run.

`FINAL/RUN_COMPLETE.json` points to the report, bundle, and fresh predictions.
`FINAL/OUTPUT_MANIFEST.json` binds all non-final members. Paths are relative;
private or author-local locators are forbidden. Inputs and registered downloads
are not copied into reports except the licensed assets intentionally carried by
the model bundle.
