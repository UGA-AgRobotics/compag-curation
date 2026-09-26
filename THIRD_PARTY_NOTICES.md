# Third-party notices — R5.4

No third-party package source, third-party wheel, SAM2/ResNet checkpoint, Ultralytics package, or external YOLO weight is vendored in the repository, wheel, or sdist. The exact author-selected native r92 model and ten original photographs are included in `bundled-assets/` in the repository and source archives; their scope is recorded in `ASSET_LICENSES/PUBLIC_ASSET_SCOPE_EN.md`. The base software package has no runtime dependencies. Optional dependency groups and pinned versions are declared in `pyproject.toml` and `requirements/`; the observed license inventory is `manifests/DEPENDENCY_LICENSE_REPORT.csv` and `manifests/SBOM.cdx.json`. Those historical inventories should be refreshed before any later dependency update.

The supported science-GPU route uses pinned PyTorch/torchvision, CuPy, XGBoost, and official Meta SAM2 dependencies. Official SAM2.1 Hiera and torchvision ResNet50 assets are downloaded only on explicit `assets fetch` and verified against `manifests/PUBLIC_ASSET_REGISTRY.json`; they are not redistributed by this candidate. SAM2's reviewed license is Apache-2.0. Torchvision's license is BSD-3-Clause, while pretrained weight use may involve separate training-data terms.

The optional `yolo-local` extra pins Ultralytics 8.3.221. Its [pinned upstream license](https://github.com/ultralytics/ultralytics/blob/v8.3.221/LICENSE) is AGPL-3.0; [Ultralytics also describes an Enterprise option](https://github.com/ultralytics/ultralytics/blob/v8.3.221/README.md). This repository does not bundle Ultralytics or a YOLO checkpoint. Assess obligations for the planned combined distribution or service before publication; this notice makes no legal determination or blanket change to the project's approved MIT/CC BY split.

The native author-selected r92 model ZIP and ten-photo ZIP are included as separate files under `bundled-assets/`. Their author-selected terms are documented in `ASSET_LICENSES/PUBLIC_ASSET_SCOPE_EN.md`; the software's MIT license does not grant rights to those assets. Other rights-holder authority remains unverified.

The separate optional `yolo-segment-local` environment pins **Ultralytics 8.4.26**
in `requirements/yolo-segment-overlay-cp312-linux-x86_64.txt` and its setup helper.
Its [version-specific upstream license](https://github.com/ultralytics/ultralytics/blob/v8.4.26/LICENSE)
is separate from the older supported `yolo-local` 8.3.221 profile. Neither
profile is upgraded here. Both upstream versions use AGPL-3.0 with upstream
commercial licensing options; this records the applicable notices, not a legal
finding. External checkpoint provenance remains the user's responsibility.
