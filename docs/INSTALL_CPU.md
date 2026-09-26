# Historical JR09 base-source installation

This page describes an older model-free source candidate. For current COMPAG 1.9.4rc10 full-card installation after cloning GitHub, use [Start Here](../START_HERE_EN.md) and `python3 start_compag.py`. The current full workflow requires the pinned science-GPU environment and the four files included in `bundled-assets/`. The older commands below are retained only for historical reference.

The supported platform for this local acceptance is Linux x86-64 with CPython
3.12. This candidate does not provide or authorize a wheel, source
distribution, CPU science profile, or GPU science profile.

From an extracted copy of this source tree, create a new environment and
install only the base package. The invoking build environment must already
contain the exact build backend versions declared in `pyproject.toml`:

```bash
set -euo pipefail
ROOT="$(pwd -P)"
VENV="$ROOT/.venv-jr09-base"
test -f "$ROOT/PUBLIC_PACKAGING_REVISION.json"
test ! -e "$VENV"
python3.12 -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --no-index --no-build-isolation --no-deps "$ROOT"
"$VENV/bin/python" -m pip check
"$VENV/bin/python" -I -B -m compag_curation --help
"$VENV/bin/python" -I -B -m compag_curation --version
"$VENV/bin/python" -I -B -m compag_curation doctor --profile base
```

For the complete fresh-copy check, including the selected synthetic tests and
machine-readable receipt, use the single command in
[Development](DEVELOPMENT.md). Model-assisted, training, inference, and
scientific execution routes are unavailable from this package.
