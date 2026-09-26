# Supported Environment Records

For the current 1.9.4rc10 GitHub clone, use [Start Here](../START_HERE_EN.md) and `python3 start_compag.py`. The guide calls the pinned GPU installer and checks the installed environment. The older candidate notes below explain lock provenance and should not be used as a first-run checklist.

Candidate 1.9.4rc5 uses the retained exact science-gpu Conda/pip/SAM2 locks through
`install_r92_gpu.py` for `r92 infer-full`. This is a separate environment from
the CPU `install_laptop.py` entry. See `FIRST_RUN_R92_EN.md` for the complete
round commands. The older records below describe their original 1.9.3 lock
provenance and are not an rc4 installation instruction by themselves.

Candidate 1.9.4rc5 adds `constraints-r92-cpu-cp312-linux-x86_64.txt`, the
small pinned SciPy 1.16.2/XGBoost CPU 2.1.1 addition to the quick/review
entry environment. It supports the separate native r92 tabular sample, not the
recorded CUDA image/reference environment. Older records below are historical.


The base distribution has no runtime dependencies. Files in this directory are
observed, platform-specific records from clean acceptance environments. Base
and quick support is CPython 3.12 on Linux x86-64. The v1.9.3 `science-gpu`
closure is bound to exact CPython 3.12.7, Linux x86-64, and PyTorch's CUDA 13.2
wheel family. The v1.2.0 `science-cpu` files are retained for provenance only;
they are not a supported v1.9.3 installation surface.

- `constraints-bootstrap-cp312-linux-x86_64.txt` pins the complete build
  bootstrap, including the `packaging` runtime dependency required by `wheel`.
- `constraints-laptop-review-cp312-linux-x86_64.txt` pins the additional
  pandas and timezone dependencies used by the local Only_codes reviewer.
  This small UI closure is not the recorded reference GPU environment.
- `constraints-quick-demo-cp312-linux-x86_64.txt` pins the quick environment.
- `constraints-science-cpu-cp312-linux-x86_64.txt` pins the public CPU archive
  closure from v1.2.0, including every observed runtime transitive dependency;
  do not use it to create a v1.9.3 environment.
- `constraints-science-gpu-cp312-linux-x86_64.txt` pins every archive in the
  shared v1.9.3 Lite/Full GPU closure, including the official PyTorch cu132 wheels and CUDA
  runtime packages. Its two NCCL distributions have different owners:
  `nvidia-nccl-cu13` is required by PyTorch cu132, while
  `nvidia-nccl-cu12` is required by the XGBoost 2.1.1 Linux wheel.
- `conda-science-gpu-cp312-linux-x86_64.lock.json` binds the complete shared
  Lite/Full Conda closure: all 71 package names, versions, builds, subdirs,
  official HTTPS URLs, byte sizes, and SHA-256 digests. The GPU installer
  converts this same lock into an explicit Mamba input for both the normal
  online route and the verified Conda/CUDA-cache route; it does not solve a
  different environment at installation time.
- `requirements-sam2-cp312-linux-x86_64.txt` retains the v1.2.0 CPU SAM2 build
  record.
- `requirements-sam2-gpu-cp312-linux-x86_64.txt` binds the same reviewed
  official SAM2 Git commit for a CUDA-extension build.
- `constraints-dev-cp312-linux-x86_64.txt` pins the build and QA closure.

Install a selected supported `constraints-*.txt` archive lock with
`--require-hashes --no-deps`. The science-gpu lock additionally requires
`--no-compile --no-build-isolation --force-reinstall`, so every one of its 62
archive-backed distributions receives the exact PEP 610 URL/SHA-256 record
validated at runtime, even when a bootstrap or Conda package already supplied
the same version. Install the GPU SAM2 VCS requirement separately with
`SAM2_BUILD_CUDA=1`, `SAM2_BUILD_ALLOW_ERRORS=0`, `--no-build-isolation`, and
`--no-deps`; its PEP 610 record must contain the same official URL and full
commit. That Git requirement has no archive SHA-256 or publisher-signature
claim. Install the final project wheel separately with
`--no-compile --no-deps --force-reinstall` after checking the
publisher-provided or release-page SHA-256 through a trusted channel. Direct
science-gpu requirements also remain pinned in `pyproject.toml` for package
metadata consumers.

The separately distributed Verified Conda/CUDA Cache is a transport for the
71 archives in the Conda lock only. It does not contain the 62 pip archives,
the pinned SAM2 Git source, or registered SAM2/ResNet model assets. Therefore
it must not be described as a fully offline COMPAG installer.
