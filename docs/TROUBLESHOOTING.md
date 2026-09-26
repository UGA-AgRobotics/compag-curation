# Troubleshooting - JR09 model-free candidate

## Python version rejected

The current acceptance requires CPython 3.12 on Linux x86-64. Do not replace or
relink the system interpreter. Use a side-by-side, user-managed Python 3.12 and
rerun the single command in [Development](DEVELOPMENT.md) with new output paths.

## Acceptance output already exists

The runner is no-clobber by design. Preserve the earlier work/receipt for
inspection and choose new absent paths. Do not delete a prior acceptance result
merely to reuse its name.

## Build backend unavailable

Invoke acceptance from a reviewed Python 3.12 build environment containing the
exact setuptools and wheel versions declared in `pyproject.toml`. The runner
uses no index and does not resolve or download dependencies. A missing backend
is an environment failure, not permission to enable network access.

## Model or GPU feature requested

The fitted r92 closure, GPU installer, frozen wheel/sdist, and scientific
dependency environment are intentionally absent. Those features are not
repairable from this public tree and must fail closed. Do not copy material
from the reviewer package into the public source. The only accepted current
route is base help/version/doctor plus the selected model-free synthetic tests.

## A retained test is not in the active lane

Consult `PUBLIC_TEST_INVENTORY.json`. Restricted, historical, deferred, and
non-selected tests are preserved as code evidence but receive no pass credit.
Do not run full discovery or reinterpret a skip as success.
