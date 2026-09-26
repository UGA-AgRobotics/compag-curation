> **JR09 public-code candidate boundary (2026-09-24).** The fitted Project-1/CJ r92 model closure is research material and is **not included** in this public-code candidate. Any inherited statement below calling that preset "bundled" or "public" describes the frozen private 1.9.3 build and is superseded for this candidate. The r92-assisted lane requires the separately controlled reviewer asset. Application/scientific algorithms in `src/` are unchanged. This is a local candidate pending repository confirmation and asset-specific rights review, not a deployed release.

# Architecture Decision Record

## Decision 8: Correct Native Review Discovery And Conda Python Attestation

Correction candidate for COMPAG Curation 1.9.3; exact
package/artifact/installer acceptance and a native real System-B execution
remain separate required checkpoints.

Preserve omission of the optional round-number constraint while discovering a
sealed begin operation, and verify transfer shortlist/review-request records
using the transfer-specific roles with which they were published. Keep these
changes inside the durable reviewer boundary; they do not change selection,
review decisions, or training.

For the exact locked Conda Python package, bind the package build, archive
digest/size, source-member path/digest/size, and one expected prefix
relocation. Do not treat Mamba's `sha256_in_prefix` as authoritative. Reject
any additional binary change, unexpected relocation, metadata drift, or
package-archive mismatch before the installed interpreter is hardened or run.

## Decision 7: Strict Post-r92 Transfer Rounds

Candidate for COMPAG Curation 1.8.1; exact package/artifact/installer acceptance
and a real v1.8.1 System-B execution remained separate required checkpoints.

Allow one historical starting point only through an exact private-data gate:
the separately supplied 582,968,556-byte CSV with SHA-256
`1836d704d15f27728e3ba09d7cddf0aea9458049bc66700dc98f14e7111ec29d`.
Never bundle that CSV or its imported baseline in any public artifact.

In round 1, generate proposals and features for exactly one unseen image and
score them with the verified published r92 classifier and frozen published
prototype/PCA state. Select deterministic Top-K (`K=50`) only inside margin
`0.2` of threshold `0.5`; review all selected rows explicitly. Fresh-fit
XGBoost over the verified transfer baseline plus reviewed image rows to publish
`project-r1`. Later rounds repeat with one new image and the immediately prior
`project-rN`, decision log, and completion root.

Freeze the imported baseline split and published r92 feature state. Treat new
image groups as train-only, retain cumulative reviewed raw rows, and never warm
start or append trees. Record
`SINGLE_NEW_IMAGE_POST_R92_TRANSFER_V1`,
`POST_R92_REVIEWED_TRANSFER_BASELINE`, and `NOT_R92_REPRODUCTION` in model
lineage. Do not call `project-r1` r93 because the complete historical r1-r92
decision and retraining sequence remains unavailable.

## Decision 6: Sequential One-Image Active Learning

Candidate for COMPAG Curation 1.8.0. Source contract tests have passed at
recorded checkpoints, while final package/artifact/installer acceptance and a
real v1.8.0 System-B GPU end-to-end run remain pending.

Add a v3 image-round beside the retained v2 cumulative-pool round. One
image-round accepts exactly one unseen image, runs proposal generation,
four-scale feature extraction, and parent-XGBoost scoring, then selects one
deterministic uncertainty batch at threshold 0.5, margin 0.2, and Top-K 50.
Only selected proposals receive review decisions or enter training. Matching
image bytes/group cannot re-enter a later image-round, so unselected proposals
from that image do not receive a second batch.

After finalized review, perform a fresh full XGBoost fit over genesis plus the
cumulative reviewed image-round feature archive. Freeze the genesis test groups
and PCA, recompute the prototype from accumulated outer-train positives, add
the new group to training only, and bind the genesis Stage-20 feature hash,
parent bundle, image SHA-256, cumulative archive, effective train groups, and
frozen test groups into the `project-rN` lineage.

This one-image mapping is a v1.8.0 operational scheduling decision, not the
manuscript's definition of an active-learning round or reproduction of its
historical sequence. The packaged r92 state remains initial-review transfer
assistance. There is no supported r92 corpus/lineage import, and this workflow
never emits an `r93` label.

## Decision 5: One Wheel With Explicit Full And Lite GPU Profiles

Accepted locally for COMPAG Curation 1.4.0 after installed-wheel Full and Lite
real-backend runs in the exact NVIDIA environment. Historical v1.7.2 local
source/package/artifact acceptance is also complete. Current v1.9.3 acceptance,
remote GitHub CI, tagging, and publication remain pending; a new real end-to-end GPU science/System-B run
is not claimed.

Keep one source tree, one wheel, and the existing exact `science-gpu`
dependency closure. Require users to select `full` or `lite` explicitly in the
profile installer. Planning is read-only by default and `--execute` is required
for mutation. Store only immutable profile IDs after the CLI boundary.

Full remains `canonical-xgb-recall-gpu-v1` with SAM2.1 Hiera Large and sealed
CUDA prompt microbatch 32. Add Lite as `efficient-xgb-recall-gpu-v1` with
SAM2.1 Hiera Tiny and sealed CUDA prompt microbatch 16. Keep the common
ResNet50/Ultra/PCA32/XGBoost pipeline and `points_per_batch = 512` method and
bundle setting. Both profiles require CUDA for SAM2, ResNet50, XGBoost
training/scoring, and supported inference; CPU is permitted only for bounded
pre/postprocessing without a supported GPU kernel. There is no CPU fallback.

Do not add Lite to the canonical-fidelity identity set. Lite and Full outputs,
models, bundles, resume state, and active-learning lineage are profile-bound
and fail closed when crossed. Hiera Tiny is a lower-footprint alternative, not
a claim of scientific, numerical, or canonical equivalence.

Do not publish an automatic GPU-memory selection threshold from the bounded
acceptance measurements. On the RTX 4090 host, Lite passed at 3,503 MiB sampled
maximum total GPU memory and Full at 4,650 MiB; both completed the full
synthetic real-backend workflow without CPU fallback. The common dependency
doctor passed on an RTX 1000 Ada 6 GB laptop with driver 580.173.02, but the
Lite end-to-end demo has not passed there. Laptop acceptance remains a
separate evidence gate.

## Decision 4: GPU-First Canonical Profile

Accepted for COMPAG Curation 1.3.0.

Replace the supported canonical installation surface with the explicit
`canonical-xgb-recall-gpu-v1` contract on Linux x86-64, exact CPython 3.12.7,
and the locked NVIDIA CUDA environment. SAM2, ResNet50, XGBoost training, and
supported bundle inference must execute on CUDA. Image decoding, card/grid
preparation, file I/O, provenance, and bounded preprocessing/postprocessing may
remain on CPU where no supported GPU kernel exists.

Keep the frozen AMG method and bundle at batch 512. Use a sealed 32-point CUDA
execution microbatch for Stage 20, Stage 60, and standalone bundle inference
over the same prompt lattice. Discard non-M2M transient `low_res_masks` before
upstream aggregation only when they cannot be consumed, and quantize selected
confidence evidence only after raw-score selection. This bounds VRAM residency
without adding a user knob, a CPU route, or a predictor-feature change.

This execution decision is not claimed to be bit-identical to method-batch-512
SAM2 on the acceptance hardware. On the locked RTX 4090 host, one of 528 masks
changed by one pixel, seven IoU values by 1e-6, and two stability scores by
approximately 2.65e-4. The selected 32-point execution baseline was byte-stable
across two Stage-20 runs, and Stage-60/fresh/repeat inference was byte-identical.

CUDA and XGBoost CUDA-build checks fail closed and the runtime never silently
falls back to CPU. The deterministic FP32 policy disables TF32 and cuDNN
benchmarking and enables deterministic Torch algorithms. The v1.2
`science-cpu` lock remains historical provenance only and is unsupported due
to its Torch security advisory. Hosted CPU CI tests mocked/static contracts;
real acceptance requires the exact locked environment on an NVIDIA host.

## Decision 3: Additive Canonical CPU Profile

Accepted for COMPAG Curation 1.2.0.

Historical decision, superseded as a supported install surface by Decision 4
and retained under Decision 5's dual-GPU facade.
Its balanced-v1 and canonical-CPU-v2 project/bundle identities are now
structural load/inspect/verify-only; v1.4 writers and execution entry points
accept only the two explicit GPU profiles.

At v1.2, keep `public-safe-balanced-v1` unchanged as the generated default and add the
explicit `canonical-xgb-recall-cpu-v1` contract. The canonical branch binds
Hiera Large AMG-only proposals, Ultra plus locally verified ResNet50 features,
train-only prototype/PCA32 fitting, group-aware XGBoost defaults, fixed 0.50
inference, and YOLO off. No private fitted study asset becomes a public input,
and no paper-result reproduction claim follows from method availability.

## Decision 2: Runnable Public Project

Accepted for COMPAG Curation 1.1.0.

The 1.0 validation/planning design below remains a compatibility contract. The
1.1 release adds a separate versioned public-project facade with explicit
execution, registered public assets, a mandatory review pause/import boundary,
group-separated XGBoost training, portable bundle-only inference, atomic run
state, and generated demonstrations. At 1.1, only
`public-safe-balanced-v1` was a supported real profile and it was intentionally
not the historical paper profile. Decision 3 supersedes that scope for 1.2
while retaining the balanced compatibility contract.

This supersedes the earlier consequence that the CLI had no runnable path; it
does not change the closed legacy registry or inflate historical parity claims.

## Decision 1: Notebook-Independent Contracts

## Status

Accepted for COMPAG Curation 1.0.0 on 2026-08-26.

## Context

The source authority contained logic originating from 22 notebooks and 236
code-cell occurrences. The reconciled inventory classifies 84 cells as
operational and 152 as non-operational. The public release requires every
operational capability to have a current implementation symbol or real manual
entry point and a public test, without executing notebooks or rerunning sealed
science.

Earlier internal reviews emphasized lexical and statement-level diagnostics.
Those private records do not define the professional public API.

## Decision

Use the smallest coherent `src`-layout package that covers the eight actual
workflow groups:

- keep a closed registry of 21 variants;
- separate configuration, contracts, plans, domain handlers, and optional
  adapters;
- expose CLI validation and inert planning for every workflow;
- keep scientific execution behind typed Python services rather than an
  advertised generic CLI execution switch;
- represent initial and iterative active learning as explicit, append-only CSV
  pause/resume operations rather than an advertised GUI;
- resolve heavy scientific and GUI dependencies lazily;
- pass state through typed objects and named artifacts, never notebook globals;
- use fail-on-collision for normal outputs;
- retain the two legacy feature-CSV backups under
  `PRESERVE_BACKUPS`;
- maintain cell-to-symbol and cell-to-test traceability without requiring
  one-to-one lexical identity.

The public tree excludes notebooks, private controls, raw research data,
row-level evidence, predictions, scores, models, checkpoints, and sealed
scientific evidence.

## Consequences

Package import, CLI help, validation, and planning are usable without notebook
runtime or scientific asset initialization. Integrators can inspect complete
workflow contracts before supplying optional dependencies, data, models, and
typed services.

The CLI intentionally does not promise a one-command scientific run. Users
requiring real execution must construct a controlled Python integration and
satisfy the handler's explicit artifact and service contracts.

Software QA can cover imports, contracts, plans, callability, UI handoffs,
backup behavior, and packaging with mocks and synthetic fixtures. It cannot
establish historical real-data behavior or scientific-result parity.

## Scientific Status

```text
SCIENTIFIC_EXECUTION_CALLS=0
SCIENTIFIC_PARITY_STATUS=NOT_RETESTED_BY_DESIGN
BEHAVIORAL_PARITY_STATUS=NOT_RETESTED_BY_DESIGN
```

The release refactor must not be described as having generated the already
sealed study results.
