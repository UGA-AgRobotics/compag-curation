# Model and execution assets for rc9 R4

The native r92 scoring closure is `bundled-assets/COMPAG_CJ_R92_INFERENCE_SAMPLE_20260924T200000Z.zip` (SHA-256 `fffef7a85192b99dcca3bcc5750215150d2ca9659121363198041244bbfe57c1`). It is included in the repository and source archives, but not embedded in the application wheel. It contains an XGBoost UBJ classifier, imputer, PCA/prototype arrays and metadata, but no training rows or photos. Its author-selected asset terms are recorded in `ASSET_LICENSES/PUBLIC_ASSET_SCOPE_EN.md`; other rights-holder authority has not been documented.

The ten author-selected original example photos are included in `bundled-assets/COMPAG_R92_TEN_PHOTO_INPUTS_RC6_R3_20260924.zip` (SHA-256 `344c8d767a1719403a3713e0f51587971f4ab9ea18e1d1a44034ed72c9297914`). They are full-card examples, not independent validation. See `RELEASE_ASSETS_PLAN.md` for the bundled layout and optional Release mirror.

Official SAM2.1 Hiera Large and ResNet50 V2 execution weights and their license texts are fetched by the user into an external asset root with `compag-curation assets fetch --asset-root "$ASSETS" --profile full`, then checked with `assets verify`. Their exact URLs, sizes and hashes are in `manifests/PUBLIC_ASSET_REGISTRY.json`; downloads are explicit and outside this package. Optional YOLO checkpoints are produced or supplied locally and are never bundled by default.
