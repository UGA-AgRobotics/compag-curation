> **JR09 public-code candidate boundary (2026-09-24).** The fitted Project-1/CJ r92 model closure is research material and is **not included** in this public-code candidate. Any inherited statement below calling that preset "bundled" or "public" describes the frozen private 1.9.3 build and is superseded for this candidate. The r92-assisted lane requires the separately controlled reviewer asset. Application/scientific algorithms in `src/` are unchanged. This is a local candidate pending repository confirmation and asset-specific rights review, not a deployed release.

# Data Schema

## Project Configuration

New `config.toml` files use schema `compag-curation-public-project/v2`. Its four tables
are `project`, `paths`, `execution`, and `assets`; `schema` is a top-level key.
Unknown or missing fields fail. The structural machine contract is
`schemas/public_project.schema.json`; runtime validation adds filesystem,
cross-field, and local-asset checks. The balanced
`examples/public_fixture/config.toml` remains a structural compatibility
fixture, not a v1.7 execution template.

Paths are project-relative and cannot traverse, contain unsafe symlinks, or
alias output. Image basenames use portable printable ASCII without commas;
commas are reserved by the source-backed evaluation selector. The schema can
structurally read the historical balanced v1 and canonical CPU v2 branches,
but they are inspection/verification-only and cannot be initialized or
executed in v1.9.3. Existing schema-v1 projects remain read-only compatible and
are interpreted as tiled without being rewritten. New projects use the Full
GPU canonical branch, which locks Hiera Large and CUDA microbatch 32; the Lite
efficient branch, which locks Hiera Tiny and CUDA microbatch 16; or the
experimental Full-image branch, which locks Hiera Large, no external/persisted
tiling, and execution microbatch 8. Full-image also seals the internal SAM2
crop pyramid (64 points per side, configured batch 512, crop layers 2, downscale
1, overlap `512/1500`, cap 1000 proposals/image). All lock
Ultra/ResNet50/PCA32, group-aware training, threshold, NMS, and YOLO-off values.
Every branch requires exact lowercase asset SHA-256 values, and profile
execution/asset branches cannot be mixed.

For Full-image, Stage-20 `proposal_config.json` additionally records
`durable_mask_encoding = "bbox-cropped-column-packbits-little-v1"`,
`connected_component_policy =
"largest-8-connected-pixel-area-top-left-tie-v1"`, and the exact SHA-256 of
`proposals.csv` and `features.csv`. The CSV transport encoding remains
`base64-packbits-little-column-major-bbox-v1`. Review/training consumes these
as a closed source binding and fails before fitting when any field or bound
table differs.

## Review Table

The balanced compatibility CSV header is exactly:

```text
proposal_id,proposal_sha256,image_id,image_sha256,group_id,label,review_status
```

The Full/Lite/Full-image v2 profile header is exactly:

```text
proposal_id,proposal_sha256,image_id,image_sha256,group_id,label,review_action,review_weight,review_status
```

An export has blank label/action/weight fields and `pending`. A reviewed
GPU-profile import pairs `accept`/`flip` with `1.0`, `sus_accept`/`sus_flip` with
`0.4`, or `skip` with `0.0`; its label is `0` or `1` and status is `reviewed`.
Proposal and image identities are immutable full lowercase SHA-256 values. The
import row set must equal the export row set. See
`schemas/review_table.schema.json`, the balanced examples
`examples/public_fixture/review_request_example.csv` and
`examples/public_fixture/reviewed_example.csv`, and the canonical examples
`examples/public_fixture/canonical_review_request_example.csv` and
`examples/public_fixture/canonical_reviewed_example.csv`.

Proposal hashes are profile-domain separated before they enter this table:
Full uses `compag-canonical-proposal-v2`, while Lite uses
`compag-efficient-sam2-tiny-proposal-v1`; Full-image uses its distinct
full-image multiscale proposal domain. Review rows cannot be transferred
between these domains.

## Active-Learning Operation Tables

The scored pool header is
`proposal_id,proposal_sha256,image_id,image_sha256,group_id,scale,xgb_p`.
The shortlist appends `distance_to_threshold,uncertainty,selection_rank`.
Decision logs additionally bind sequence, phase, round number/time, the exact
human action/weight, fixed acquisition policy, source bundle/predictions,
selection manifest, and reviewed-batch hash.

General cumulative-pool selection/completion uses the v2 active-learning
schemas. Sequential image-round selection/completion uses
`compag-curation-canonical-active-learning-selection/v3` and
`compag-curation-canonical-active-learning-completion/v3`. The v3 workflow
accepts one unseen image and only its at-most-50 selected proposals can appear
as new decisions/training rows.

Its `accumulated_al_features.csv` uses schema
`compag-curation-canonical-active-learning-accumulated-raw-features/v1` and
stores immutable proposal/image/group identity, source hash, proposal index,
four crop-scale rows with IoU/stability and the canonical raw feature order,
plus the encoded float32 embedding. Physical and logical archive hashes are
bound into `compag-curation-canonical-active-learning-model-lineage/v1`
together with the genesis Stage-20 feature hash, genesis split, parent bundle,
round image SHA-256, effective train groups, frozen test groups, round number,
and `project-rN` label. The same image bytes/group cannot re-enter.

Version 1.9.3 retains the separate strict post-r92 transfer family introduced
in v1.8.1. The sealed
baseline uses `compag-curation-transfer-baseline/v1`; round-1 inference uses
`compag-curation-r92-transfer-image-inference/v1`; selection/completion use
`compag-curation-canonical-transfer-active-learning-selection/v1` and
`compag-curation-canonical-transfer-active-learning-completion/v1`; model
ancestry uses
`compag-curation-canonical-transfer-active-learning-model-lineage/v1`.
The baseline import accepts only the separately supplied 582,968,556-byte
source with SHA-256
`1836d704d15f27728e3ba09d7cddf0aea9458049bc66700dc98f14e7111ec29d`.
That private source is not a public artifact. Transfer models remain labeled
`project-rN`, and their explicit claim is `NOT_R92_REPRODUCTION`; no `r93`
output label is emitted.

The JSON Schema describes each row's structure and paired state. Runtime
validation additionally requires `proposal_sha256 == proposal_id` and exact
identity/set equality between the exported and reviewed files.

## Point Ground Truth

CVAT XML `<point>` and `<points>` nodes with label `cj` are accepted. Multiple
semicolon-separated coordinates are expanded. DTD/entity declarations,
nonfinite coordinates, duplicate image declarations, dimension mismatch, and
out-of-bounds points fail. The non-destructive converter emits schema
`compag-curation-canonical-point-annotations/v1` with rows containing
`image,label,x,y,width,height` and source provenance.

The JSON Schema bounds row structure and count. Runtime validation additionally
enforces `x < width`, `y < height`, configured-image dimensions, and source
identity.

## Split Manifest

The balanced split record contains disjoint train/validation/test groups and
the reviewed-table SHA-256. The canonical split record contains the recovered
tile-balanced, group-pure outer train/test groups, exact group union and row
coverage, five-fold search contract, and reviewed-table SHA-256. Canonical
early-stop validation is derived group-safely inside each training context and
is never an outer manifest partition. The outer test is evaluated once at
scale 1.0; the canonical operating threshold remains fixed at 0.50.

## Output Tables

`tiles_index.csv` binds source and tile hashes, sanitized names, group, geometry,
and dimensions. Canonical `proposals.csv` binds full-mask and positional
proposal identities; raw `features.csv` preserves four lossless export scales
and the ResNet50 embedding needed to fit the exact 93-feature state. The
balanced compatibility profile retains its fixed 25-feature table.
`predictions.csv` records proposal/image/group identity, full-image box,
probability, prediction, and deterministic NMS keep status.
All CSV files are UTF-8, RFC 4180 compatible, LF terminated, and exact-header.
