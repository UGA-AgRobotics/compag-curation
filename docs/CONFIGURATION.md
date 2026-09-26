> **JR09 public-code candidate boundary (2026-09-24).** The fitted Project-1/CJ r92 model closure is research material and is **not included** in this public-code candidate. Any inherited statement below calling that preset "bundled" or "public" describes the frozen private 1.9.3 build and is superseded for this candidate. The r92-assisted lane requires the separately controlled reviewer asset. Application/scientific algorithms in `src/` are unchanged. This is a local candidate pending repository confirmation and asset-specific rights review, not a deployed release.

# Configuration

## Runnable Project TOML

New user-data projects use `compag-curation-public-project/v2`. The authoritative
schema is `schemas/public_project.schema.json`. Existing schema-v1 projects
remain readable as tiled projects without being rewritten. Generated projects
use one of the three GPU execution templates; `examples/public_fixture/config.toml` is a
historical balanced structural-validation fixture, not a v1.7 execution
template.

The closed tables define project name; training, inference, annotation, and
asset paths; fixed profile/device/seed/tiling/proposal values; and exact model
asset locators and lowercase SHA-256 values. Unknown fields fail.
All paths are relative to the canonical config parent. Traversal, unsafe
symlinks, input/output overlap, nonportable names, and collisions fail.

Public initialization accepts Full (`canonical-xgb-recall-gpu-v1`), Lite
(`efficient-xgb-recall-gpu-v1`), and experimental Full-image
(`full-image-multiscale-xgb-recall-gpu-v1`). CLI inputs may use the friendly
aliases `full`, `lite`, and `full-image`; only immutable IDs are persisted. Full remains the API
and CLI default when `--profile` is omitted. The historical
`public-safe-balanced-v1` and `canonical-xgb-recall-cpu-v1` mappings remain
accepted by structural loaders so existing projects can be inspected and
verified, but v1.7 does not initialize, execute, or mint bundles for them.

The two tiled GPU configurations are immutable and bind `device = "cuda"`: 512-pixel
stride with far-edge alignment and bottom/right edge-value
padding, JPEG tiles, AMG-only proposals at scale 1.0, 64 points per
side, batch 512, predicted-IoU 0.80, stability 0.88, zero crop layers, crop
downscale 2, overlap 0.40, and a 500-proposal cap. They use Ultra handcrafted
features, a 2048-value ImageNet ResNet50 embedding, 0.10 masked-crop padding,
four export-only feature crop scales, train-fitted prototype and PCA32 state,
group-aware XGBoost defaults, review weights accept/flip 1.0, suspect 0.4, and
skip 0.0, fixed inference threshold 0.50, NMS 0.50, and YOLO disabled. Changing
a profile-bound value, locator, or registered hash fails.

Full binds the SAM2.1 Hiera Large checkpoint/configuration and a sealed
execution-only CUDA prompt microbatch of 32. Lite binds the SAM2.1 Hiera Tiny
checkpoint/configuration and a sealed execution-only CUDA prompt microbatch of
16. Both retain the ResNet50 and 93-feature training/inference contract, but
the SAM2 architecture difference means their proposals, scientific outputs,
fitted state, bundles, and resume artifacts are not equivalent or
interchangeable. Canonical equivalence is claimed only for Full, not Lite.
Proposal identifiers also use separate hash domains: Full uses
`compag-canonical-proposal-v2`, while Lite uses
`compag-efficient-sam2-tiny-proposal-v1`. Identical-looking geometry cannot be
relabelled across profiles as the same proposal identity.

Full-image also binds `device = "cuda"`, but sets
`spatial_mode = "full-image-multiscale"` and does not create or persist an
external tile grid. Each input image becomes exactly one lossless processing
unit. SAM2 Hiera-L uses its internal overlapping crop pyramid with 64 points
per side, configured batch 512, two crop layers, point-downscale factor 1,
overlap ratio `512/1500`, and a maximum of 1000 proposals per image. Its
execution-only prompt microbatch is 8. The profile has a distinct proposal-ID
domain and full-image ROI feature-semantics ID even though the exported table
keeps the 93 established column names. Canonical-method and paper-result
equivalence are `NOT_CLAIMED`.

Tiled r92 state and tiled model bundles cannot initialize, score, resume, or
continue Full-image. Full-image must perform its own initial human review and
classifier fit and may continue only with a verified same-profile bundle. The
reverse crossing into tiled projects is also rejected.

The recorded batch 512 is the frozen proposal-method and bundle setting. Stage
20, Stage 60, and standalone bundle inference keep the same 64-point sampling
lattice but use the profile's sealed execution-only CUDA microbatch. Non-M2M transient
`low_res_masks` are discarded before upstream aggregation when they cannot be
consumed. These are implementation constants, not public configuration fields;
they cannot authorize a different proposal method or CPU execution. Raw SAM2
scores control ordering, deduplication, and the cap; selected confidence
evidence is quantized to six decimal places afterward for stable archives.

All three GPU runtimes use the same exact `science-gpu` closure on Linux x86-64 and
CPython 3.12.7. They fail closed when CUDA is unavailable or XGBoost lacks CUDA
support; `device = "cpu"` is not an accepted fallback. CPU remains an explicit
boundary for bounded image I/O, card/grid preparation, CSV/provenance work,
and other preprocessing/postprocessing without a supported GPU kernel.
The generated quick demo is the only supported standalone CPU computation.

Use `inspect-data` for bounded structural inspection, `validate` for complete
local dependency/asset/decode/annotation checks, and `run --dry-run` to add
output path and disk-capacity validation. None runs SAM2.

## Legacy Domain Configuration

The compatibility surface retains `compag-curation-domain-config/v2`,
`configs/schema/static_plan.schema.json`, and
`configs/examples/static_plan.json`. It still contains all eight workflow
sections and 21 variants for `--validate` and `--plan`. It does not configure
the runnable public-project facade and cannot authorize scientific execution.

Environment interpolation in the legacy contract remains limited to the
documented `COMPAG_*_ROOT` and interpreter names. Secret-like names are
rejected. The public project format performs no interpolation.
