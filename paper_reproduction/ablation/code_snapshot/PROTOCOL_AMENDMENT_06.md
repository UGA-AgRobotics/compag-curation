# Protocol Amendment 06: One Full Scientific Ablation

## Authorization

Amendment 06 implements and authorizes exactly one resumable Full scientific
Run under:

```text
AMENDMENT_06_EXTERNAL_REVIEW_ACCEPTED_ONE_FULL
```

It does not authorize another Preflight, another CUDA Smoke, a second Full Run
ID, stability fits, threshold tuning, new PCA/prototype fitting, or raw embedding
reconstruction. The accepted Amendment 02 Preflight and Amendment 04 CUDA Smoke
remain immutable references and are never scientific model inputs.

## Entry Points

The scientific controller is `code/amendment06_full.py`:

- `run_full(preflight_run=..., smoke_run=..., command_line=...)`
- `resume_full(run_dir, preflight_run=..., smoke_run=..., command_line=...)`
- `package_dispatch(run_dir, command_line=...)`

The required interpreter is:

```text
/REVIEWER_INPUT_ROOT/python
```

The required runtime root is:

```text
/REVIEWER_INPUT_ROOT/runtime
```

The caller must supply the exact accepted Preflight and Smoke paths and the
Amendment 06 CLI authorization token. Invocation under another Python prefix or
runtime root fails before Run creation.

## Admission And One-Run Control

Before reserving a Run ID, admission verifies immutable reference hashes, the
sealed Preflight identity without parsing the external Audit, the accepted Smoke
through the production reference validator, and the complete Amendment 06
authorized-change chain. Every post-genesis edit event binds its predecessor,
complete before/after code and test trees, exact changed paths, diff hash, purpose,
scope, and hashed test logs. The final event must bind the live tree and declare
the final byte freeze.

Structured pre-Run evidence must prove final-byte syntax compilation, targeted
tests, one complete project-suite invocation, active CUDA fit/save/reload/predict,
production packaging and final-validator rehearsal, and interruption/resume
rehearsal. This evidence is checked before the one-Run intent is written.
`make_test_evidence_record`, `build_authorized_edit_event`,
`append_authorized_edit_event`, `build_final_prerun_evidence`, and
`write_final_prerun_evidence` are the deterministic construction APIs for these
records; the admission validator independently reopens every bound log and live
tree byte.

The controller writes an immutable Run intent before any Results-directory side
effect. The intent reserves the sole Run ID. Creation uses one intent-owned,
fixed staging path below the Amendment 06 runtime root on the Results filesystem;
it never creates an untracked `.creating*` directory in Results. A complete
staging directory is published with Linux `renameat2(RENAME_NOREPLACE)`. If the
process stops after directory publication but before the consumption receipt,
resume validates the same identity and publishes the missing receipt. A matching
symlink, FIFO, socket, device, or other unsafe one-Run name is an integrity
failure.

## Immutable Identity And State

`config/run_identity.lock.json` is immutable from Run creation. Its `state` is
the initial `FULL_IN_PROGRESS` state. Effective state is derived from a contiguous,
hash-chained sequence of no-clobber markers:

```text
provenance/state/00_FULL_IN_PROGRESS.json
provenance/state/01_FULL_MODELS_FROZEN.json
provenance/state/02_FULL_EXTERNAL_AUDIT_SCORED.json
provenance/state/03_FULL_SCIENTIFIC_COMPLETE.json
```

Each marker binds the Run ID, Run kind, phase index, PASS status, and prior marker
hash. Missing, reordered, or conflicting markers fail closed.

Scientific execution and package/reference-validator code use disjoint manifests.
The scientific manifest includes the preprocessing, paired-resampling,
compatibility, scoring, metric, bootstrap, reporting, controller, wrapper, and
focused test sources. These bytes are snapshotted into the Run and revalidated on
every resume. Package-only corrections after scientific completion cannot change
the scientific manifest or invoke a scientific API.

## Locked Design

This is a retrospective fixed-feature-table experiment. Its primary table has
363,563 rows. The immutable Split contains 291,024 ordered training rows across
66 cards and 56,843 ordered eligible validation rows across 18 cards. The Split
is selected exclusively by the sealed row manifest and `split_order`; no Split
generator is called.

The design retains the unresolved PCA/prototype lineage as an accepted fixed-table
limitation. It does not claim a fold-safe common PCA/prototype basis. The
exploratory `no_deep_pca_features` Variant removes a PCA-derived embedding feature
block and is not described as true no-PCA. Prespecified true `no_pca` remains:

```text
NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE
```

## Frozen Paired Population

One `SimpleImputer(strategy="median")` is fit on the complete ordered 291,024-row,
93-feature training population. Its feature order and 93 medians are persisted in
strict portable JSON. Validation is transformed once in the same full space.

The accepted Amendment 02 implementation produces one shared realization using
one augmentation `RandomState(42)` stream and one Safe-SMOTE instance with its own
`random_state=42`. Post-augmentation shuffle is a deliberate no-op. Required
counts are:

| Stage | Rows | Negative | Positive |
|---|---:|---:|---:|
| raw locked order | 291,024 | 215,988 | 75,036 |
| post Mixup | 436,536 | 323,982 | 112,554 |
| post dropout | 873,072 | 647,964 | 225,108 |
| post jitter/pre-SMOTE | 1,309,608 | 971,633 | 337,975 |
| post no-op shuffle | 1,309,608 | 971,633 | 337,975 |
| post Safe-SMOTE | 1,457,449 | 971,633 | 485,816 |

The Full-specific runtime cache stores the pre- and post-SMOTE matrices, labels,
weights, lineages, validation matrix, portable imputer, and exact hashes. It is
not packaged. Resume memory-maps this one cache. Validation rejects extra or
unsafe members and binds the exact locked 93-feature order, live raw weights,
training-column medians, independently transformed validation matrix, stage
counts, lineages, array hashes, dtypes, shapes, finiteness, and positive weights.
Before model freeze, missing cache bytes may be deterministically rematerialized
only when every published scientific binding matches. Rematerialization after
model freeze is forbidden.

Feature ablations are sequential retained-column projections. The controller does
not retain ten projected matrices at once. `no_review_aware_training_weights`
changes only the sample-weight vector. `no_safe_smote` uses the exact frozen
post-jitter/pre-SMOTE population. No Variant fits an imputer or calls augmentation
or SMOTE.

## Training And Completion

The ten fresh XGBoost 2.1.1 CUDA fits execute sequentially in the prespecified
order: nine official runnable Variants followed by the separately labeled
exploratory Variant. Every model uses Seed 42, the sealed Full parameters, early
stopping only on the locked validation rows, and evaluation threshold 0.5. Smoke
and historical models are never initializers.

Each attempt is built outside the Run. The ownership journal consists of atomic,
contiguous, hash-chained per-event files bound to the Run ID. It records a
publication intent before each `renameat2(RENAME_NOREPLACE)` move into the Run and
records the completed publication afterward. The move leaves no writable staging
hard-link alias. Standalone no-clobber JSON/text publication uses an anonymous
same-filesystem `O_TMPFILE`, `fsync`, and atomic `linkat`, so interruption cannot
leave a partial final file or a named temporary file inside the Run.
`training_completion_manifest.json` is published last.
A completed Variant is skipped only after model attributes, model bytes, feature
contract, population/imputer/Split/config bindings, prediction bytes, exact
metric recomputation, CSV round-trip, CUDA evidence, and four-way Amendment 03
parity validate.

An incomplete Variant may be retried in the same Run. Only paths with an exact
ownership intent or publication record and matching size/hash may be moved into
a unique no-clobber quarantine. The moved bytes are rehashed before the
transaction is recorded. Unknown or corrupted paths are never removed. A corrupt
completion is an integrity failure and is not permission to retrain.
Quarantine itself is two-phase: an intent containing the exact source,
destination, size, and hash precedes the move, and resume reconciles an
intent-only interruption before writing the committed event.

## Model Freeze And Audit Exclusion

After all ten training completions validate, the no-clobber models-frozen manifest
binds all model and completion hashes. A process-level guard then replaces all
training, XGBoost train, imputer fit, augmentation, and Safe-SMOTE entry points
with fail-closed exceptions. Resume reinstalls the guard whenever the effective
state is at or after model freeze.

Before this gate, external-Audit access is limited to existence, size, and SHA-256.
The Audit-opening evidence is published before parsing and binds the models-frozen
manifest and state marker. The locked Audit has 10,097 candidates, 2,032 positives,
and ten cards. It is harmonized once using the accepted affine `embed_sim`, gate,
and 3x3 grid semantics, then transformed using the frozen training-only imputer.

Every model is loaded from its new UBJ and scores only its retained-column Audit
projection. Compatibility-classifier and raw-Booster CUDA predictions must be
bit-exact. External prediction, raw and review-weighted metrics, and per-card
metrics are published before `external_scoring_manifest.json`, which is the last
per-Variant scoring artifact.

## Metrics, Bootstrap, And Reporting

The primary endpoint is raw candidate-level scikit-learn Average Precision on the
locked external Audit. ROC-AUC, precision, recall, F1, and confusion counts use the
fixed threshold 0.5. Review-weighted results and card-macro summaries are
secondary. Required aggregate metrics must be finite.

All prediction CSVs use deterministic gzip. Scientific float32 values are exactly
promoted to float64 before serialization so `pandas.read_csv(engine="c",
float_precision="round_trip")` recovers the exact value. Every generated CSV is
reopened against its authoritative frame with exact finite-cell and missing-mask
equality. Metrics are recomputed from the saved predictions without tolerance.

The reporting module runs exactly 10,000 prediction-only paired card-cluster
bootstrap replicates with Seed 2026081201. Each replicate applies one common
ten-card multiplicity vector to Full and every compared Variant. It reports 95%
percentile intervals for Variant-minus-Full AP, F1, and recall deltas. It performs
no model fit, emits no p-values, and makes no significance claim.

Official tables contain the nine runnable official Variants and the separate
blocked true no-PCA record. Exploratory tables and figures contain only
`no_deep_pca_features`. The historical r92 replay is a separate control and never
the new baseline.

## Final Seal And Packaging

Scientific finalization revalidates models, training and scoring completions,
predictions, metrics, bootstrap, reports, source post-hashes, strict JSON, and the
zero post-Audit training ledger. It publishes the final state marker, final
`RUN_STATUS.txt`, and `BLOCKERS.md` before creating the manifest.

`OUTPUT_MANIFEST_FINAL.tsv` is the final and only Run mutation. It lists every
other regular Run file, excludes itself, and uses exact `True`/`False` bundle
flags. UBJ files are excluded from Review and selected into Models. Models also
select their training contracts, portable imputer, compatibility snapshot,
locked configs and Split/source bindings, blocked outcome, and all four state
markers.

Packaging runs in the isolated `amendment06_packaging` layer and publishes two
adjacent ZIPs and two adjacent verification JSONs. Package staging manifests do
not enter the frozen Run. A package failure preserves the Run and all scientific
artifacts; package-only retry targets the same Run and makes zero scientific API
calls.
