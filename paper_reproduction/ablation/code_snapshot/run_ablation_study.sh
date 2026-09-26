#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
STUDY_ROOT="${ABLATION_STUDY_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}"
PROJECT_ROOT="${ABLATION_PROJECT_ROOT:-/REVIEWER_INPUT_ROOT/Clean}"
PYTHON="${ABLATION_PYTHON:-/REVIEWER_INPUT_ROOT/python}"
RUNTIME_ROOT="${ABLATION_RUNTIME_ROOT:-$STUDY_ROOT/.runtime}"

export ABLATION_STUDY_ROOT="$STUDY_ROOT"
export ABLATION_PROJECT_ROOT="$PROJECT_ROOT"
export ABLATION_PYTHON="$PYTHON"
export ABLATION_RUNTIME_ROOT="$RUNTIME_ROOT"

mkdir -p \
    "$RUNTIME_ROOT/tmp" \
    "$RUNTIME_ROOT/cache" \
    "$RUNTIME_ROOT/config" \
    "$RUNTIME_ROOT/matplotlib" \
    "$RUNTIME_ROOT/numba" \
    "$RUNTIME_ROOT/cuda" \
    "$RUNTIME_ROOT/joblib"

export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TMPDIR="$RUNTIME_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$RUNTIME_ROOT/cache"
export XDG_CONFIG_HOME="$RUNTIME_ROOT/config"
export MPLCONFIGDIR="$RUNTIME_ROOT/matplotlib"
export NUMBA_CACHE_DIR="$RUNTIME_ROOT/numba"
export CUDA_CACHE_PATH="$RUNTIME_ROOT/cuda"
export CUPY_CACHE_DIR="$RUNTIME_ROOT/cuda/cupy"
export JOBLIB_TEMP_FOLDER="$RUNTIME_ROOT/joblib"
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -p no:cacheprovider"

exec "$PYTHON" "$SCRIPT_DIR/run_study.py" "$@"
