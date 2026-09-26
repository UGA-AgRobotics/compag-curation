# GPU installation for current COMPAG 1.9.4rc10

For a fresh GitHub clone, run `python3 start_compag.py` and follow its local browser steps and [Start Here](../START_HERE_EN.md). The browser checks the four files included in `bundled-assets/` automatically. The installer step uses the exact Conda science-GPU lock, pip archive lock, pinned SAM2 source and included matching wheel through `install_r92_gpu.py`. The GPU environment remains separate from optional Ultralytics; it must pass `doctor --profile science-gpu` before full-card inference. [Advanced commands](../FIRST_RUN_COMPLETE_ROUND_EN.md) remain available.

## Historical JR09 note — no longer applies to the current candidate

This JR09 public-code candidate intentionally has no supported GPU installation
route. The previously documented GPU installer, frozen 1.9.3 wheel, fitted r92
resource, and model-assuming packaging/CI surfaces are not part of this
model-free source package.

Do not substitute an older wheel, source archive, model, cache, or installer,
and do not combine the public tree with the reviewer-only evidence package.
Those artifacts have different audiences and identities.

Historical GPU profiles and implementation source remain visible for code
review, but they are not accepted or executable release surfaces here. A future
GPU release requires a new, separately versioned public-only distribution,
exact dependency and asset identities, real-hardware acceptance, rights review,
and explicit deployment authorization. None of those gates is claimed by this
candidate.

For the only supported current verification path, use the base, model-free
source acceptance in [Development](DEVELOPMENT.md).
