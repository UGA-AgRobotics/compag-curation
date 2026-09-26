#!/usr/bin/env bash
# Major-revision items 3+4 (audit v3.0): canonical candidate-metric
# reconciliation and cautious model-selection/card-split provenance audit.
#
# Default project root: /REVIEWER_INPUT_ROOT/Clean
# This script is read-only with respect to all original project data/models.
# It writes only below:
#   /REVIEWER_INPUT_ROOT/results

set -Eeuo pipefail

SOURCE_ROOT="${CJ_CLEAN_ROOT:-/REVIEWER_INPUT_ROOT/Clean}"
TASK_ROOT="${SOURCE_ROOT}/revision/03_04_candidate_metrics_model_selection"
CODE_DIR="${TASK_ROOT}/code"
RESULTS_ROOT="${TASK_ROOT}/results"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TOKEN="$(printf '%06x' "$(( (RANDOM << 15) ^ RANDOM ))")"
RUN_DIR="${RESULTS_ROOT}/run_${RUN_STAMP}_${RUN_TOKEN:-audit}"

mkdir -p "$CODE_DIR" "$RUN_DIR"
mkdir -p "$RUN_DIR/matplotlib_cache"
export MPLCONFIGDIR="$RUN_DIR/matplotlib_cache"
LOG_FILE="$RUN_DIR/terminal.log"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
    rc=$?
    printf '\n[ERROR] Audit stopped (exit code %s).\n' "$rc"
    printf 'Read: %s\n' "$LOG_FILE"
    printf 'Partial outputs, if any: %s\n' "$RUN_DIR"
    exit "$rc"
}
trap on_error ERR

if [ ! -d "$SOURCE_ROOT" ]; then
    echo "[ERROR] Project root does not exist: $SOURCE_ROOT"
    exit 2
fi

if [ -x "${SOURCE_ROOT}/../anaconda3/envs/SAM/bin/python" ]; then
    PYTHON_BIN="${SOURCE_ROOT}/../anaconda3/envs/SAM/bin/python"
elif [ -x "/REVIEWER_INPUT_ROOT/python" ]; then
    PYTHON_BIN="/REVIEWER_INPUT_ROOT/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "[ERROR] Python 3 was not found."
    exit 2
fi

SCRIPT_PATH="$(readlink -f "$0" 2>/dev/null || printf '%s' "$0")"
if [ -f "$SCRIPT_PATH" ]; then
    cp -a "$SCRIPT_PATH" "$RUN_DIR/$(basename "$SCRIPT_PATH")"
    sha256sum "$SCRIPT_PATH" > "$RUN_DIR/script_sha256.txt"
fi

cat > "$RUN_DIR/run_configuration.txt" <<EOF
source_root=$SOURCE_ROOT
task_root=$TASK_ROOT
run_dir=$RUN_DIR
python=$PYTHON_BIN
fixed_threshold=${FIXED_THRESHOLD:-0.5}
target_round=${TARGET_ROUND:-92}
table4_rounds=${TABLE4_ROUNDS:-92,76,106,73,121,82,112,110,74,53}
paper_testset_override=${PAPER_TESTSET:-}
r92_model_override=${R92_MODEL:-}
model_root_override=${MODEL_ROOT:-}
training_features_override=${TRAINING_FEATURES:-}
point_audit_cards=${POINT_AUDIT_CARDS:-IMG_9406,IMG_9459,IMG_9460,IMG_9486,IMG_9487}
instance_revision_cards=${INSTANCE_REVISION_CARDS:-IMG_9459,IMG_9486,IMG_9487}
audit_script_version=3.0
EOF

echo "============================================================"
echo "REVISION ITEMS 3+4: METRIC + MODEL-SELECTION AUDIT"
echo "Project root: $SOURCE_ROOT"
echo "Output:       $RUN_DIR"
echo "Python:       $PYTHON_BIN"
echo "============================================================"

"$PYTHON_BIN" - "$SOURCE_ROOT" "$RUN_DIR" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupShuffleSplit
except Exception as exc:
    raise SystemExit(
        "Required Python packages are missing. Activate the SAM Conda environment "
        "and install pandas, numpy, scikit-learn, joblib, and matplotlib. "
        f"Original import error: {exc}"
    )

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit(f"matplotlib is required to generate Figure 7: {exc}")


SOURCE_ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)

TAU = float(os.environ.get("FIXED_THRESHOLD", "0.5"))
TARGET_ROUND = int(os.environ.get("TARGET_ROUND", "92"))
TABLE4_ROUNDS = [
    int(x.strip())
    for x in os.environ.get(
        "TABLE4_ROUNDS", "92,76,106,73,121,82,112,110,74,53"
    ).split(",")
    if x.strip()
]
POSITIVE_CLASS = "CJ (label=1)"

TEST_CARD_REFERENCE = [
    "IMG_9406", "IMG_9443", "IMG_9448", "IMG_9452", "IMG_9468",
    "IMG_9473", "IMG_9480", "IMG_9487", "IMG_9497", "IMG_9508",
]
POINT_AUDIT_REFERENCE_RAW = os.environ.get(
    "POINT_AUDIT_CARDS", "IMG_9406,IMG_9459,IMG_9460,IMG_9486,IMG_9487"
)
INSTANCE_REVISION_REFERENCE_RAW = os.environ.get(
    "INSTANCE_REVISION_CARDS", "IMG_9459,IMG_9486,IMG_9487"
)

WEIGHT_MAP = {
    "accept": 1.0,
    "flip": 1.0,
    "sus_accept": 0.4,
    "sus_flip": 0.4,
    "skip": 0.0,
    "auto_accept": 0.0,
}

LEGACY_FIG7 = {"tn": 7888, "fp": 177, "fn": 79, "tp": 1953}
LEGACY_TABLE4_R92 = {"precision": 0.901, "recall": 0.966, "f1": 0.932}

warnings: list[str] = []
errors: list[str] = []
discovery_rows: list[dict[str, Any]] = []
model_meta_by_round: dict[int, dict[str, Any]] = {}


def note(msg: str) -> None:
    print(msg, flush=True)


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(SOURCE_ROOT))
    except Exception:
        return str(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def record_artifact(artifact_id: str, path: Optional[Path], status: str, note_text: str = "") -> None:
    if path is not None and path.exists():
        st = path.stat()
        discovery_rows.append({
            "artifact_id": artifact_id,
            "status": status,
            "path": str(path),
            "relative_to_clean": safe_rel(path),
            "size_bytes": int(st.st_size),
            "modified_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "note": note_text,
        })
    else:
        discovery_rows.append({
            "artifact_id": artifact_id,
            "status": status,
            "path": "" if path is None else str(path),
            "relative_to_clean": "",
            "size_bytes": "",
            "modified_utc": "",
            "note": note_text,
        })


def path_from_override(env_name: str) -> Optional[Path]:
    value = os.environ.get(env_name, "").strip()
    if not value:
        return None
    p = Path(value).expanduser().resolve()
    return p if p.exists() else None


PRUNE_DIRS = {
    ".git", "__pycache__", ".cache", "anaconda3", "miniconda3",
    "node_modules", "revision", "$RECYCLE.BIN",
}


def find_named(names: Iterable[str], max_results: int = 100) -> list[Path]:
    wanted = set(names)
    out: list[Path] = []
    for root, dirs, files in os.walk(SOURCE_ROOT):
        dirs[:] = [d for d in dirs if d not in PRUNE_DIRS and not d.startswith("run_202")]
        for f in files:
            if f in wanted:
                out.append(Path(root) / f)
                if len(out) >= max_results:
                    return out
    return out


def score_candidate_path(path: Path, preferred_parts: list[str]) -> tuple:
    s = str(path)
    exact_bonus = sum(1 for part in preferred_parts if part in s)
    release_penalty = int("/_release/" in s or "/github_repo/" in s)
    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = 0
    return (exact_bonus, -release_penalty, mtime, -len(s))


def choose_path(
    override_env: str,
    exact_candidates: list[Path],
    fallback_names: list[str],
    preferred_parts: list[str],
) -> Optional[Path]:
    over = path_from_override(override_env)
    if over is not None:
        return over
    for p in exact_candidates:
        if p.exists():
            return p.resolve()
    found = find_named(fallback_names)
    if not found:
        return None
    return sorted(found, key=lambda p: score_candidate_path(p, preferred_parts), reverse=True)[0].resolve()


paper_testset = choose_path(
    "PAPER_TESTSET",
    [
        SOURCE_ROOT / "Shared/maskout_tile/test_set/_paper_testset__xgb_recall/testset_labeled__xgb_recall.csv",
    ],
    ["testset_labeled__xgb_recall.csv"],
    ["Shared/maskout_tile/test_set", "_paper_testset__xgb_recall"],
)

model_root = path_from_override("MODEL_ROOT")
if model_root is None:
    exact_model_root = SOURCE_ROOT / "always_same/jupyter/models/cj_classifier"
    model_root = exact_model_root if exact_model_root.exists() else None

r92_model = choose_path(
    "R92_MODEL",
    [
        SOURCE_ROOT / "always_same/jupyter/models/cj_classifier/r92_hybrid/cj_ultra_tilesafe_xgb_r92_hybrid.pkl",
    ],
    ["cj_ultra_tilesafe_xgb_r92_hybrid.pkl"],
    ["always_same/jupyter/models/cj_classifier", "r92_hybrid"],
)
if model_root is None and r92_model is not None:
    model_root = r92_model.parent.parent

r92_tile_preds = choose_path(
    "R92_TILE_PREDS",
    [SOURCE_ROOT / "always_same/jupyter/models/cj_classifier/r92_hybrid/tile_preds.csv"],
    ["tile_preds.csv"],
    ["models/cj_classifier/r92_hybrid"],
)

history_csv = choose_path(
    "MODEL_HISTORY",
    [SOURCE_ROOT / "always_same/jupyter/models/cj_classifier/history.csv"],
    ["history.csv"],
    ["models/cj_classifier"],
)

training_features = path_from_override("TRAINING_FEATURES")
known_training_candidates = [
    SOURCE_ROOT / "Shared/PC_codes/annotation_2_features/cj_noncj_512_tiles/features_train.csv",
    SOURCE_ROOT / "always_same/jupyter/fetures/features_train.csv",
]
if training_features is None:
    training_features = next((p.resolve() for p in known_training_candidates if p.exists()), None)

record_artifact("paper_testset_labeled", paper_testset, "FOUND" if paper_testset else "MISSING")
record_artifact("model_root", model_root, "FOUND" if model_root else "MISSING")
record_artifact("xgb_r92", r92_model, "FOUND" if r92_model else "MISSING")
record_artifact("r92_internal_test_predictions", r92_tile_preds, "FOUND" if r92_tile_preds else "MISSING")
record_artifact("model_history", history_csv, "FOUND" if history_csv else "MISSING")
record_artifact("training_feature_table", training_features, "FOUND" if training_features else "MISSING")

pd.DataFrame(discovery_rows).to_csv(OUT / "discovered_inputs.tsv", sep="\t", index=False)

if paper_testset is None:
    errors.append("Canonical paper test-set CSV was not found.")
if r92_model is None:
    errors.append("The r92 XGBoost model was not found.")
if errors:
    (OUT / "BLOCKERS.txt").write_text("\n".join(f"- {e}" for e in errors) + "\n")
    raise SystemExit("Required inputs are missing; see BLOCKERS.txt and discovered_inputs.tsv")


# The historical pickle can reference this class in __main__. During inference,
# a sampler is a no-op. Defining it here is necessary for safe unpickling of the
# project's own local model artifact.
class SafeSMOTE:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def get_params(self, deep=True):
        return dict(self._kwargs)

    def set_params(self, **params):
        self._kwargs.update(params)
        return self

    def fit(self, X, y=None):
        return self

    def fit_resample(self, X, y):
        return X, y


def round_number_from_text(value: Any, require_r_prefix: bool = False) -> Optional[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    patterns = [r"(?:^|[_-])r(\d+)(?:[_\-.]|$)"]
    if not require_r_prefix:
        patterns.append(r"^\s*(\d+)(?:\.0)?\s*$")
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I)
        if m:
            return int(m.group(1))
    return None


def parse_round(path: Path) -> Optional[int]:
    for text in [path.name, path.parent.name]:
        value = round_number_from_text(text, require_r_prefix=True)
        if value is not None:
            return value
    return None


def history_round_info(row: pd.Series) -> tuple[Optional[int], str]:
    # Prefer the model filename because some legacy rows contain a stale
    # round_tag (for example r1) next to a model filename carrying the actual
    # checkpoint number. Then fall back to round_tag and round.
    for column, require_prefix in [
        ("model_name", True),
        ("round_tag", False),
        ("round", False),
    ]:
        if column not in row.index:
            continue
        value = round_number_from_text(row.get(column), require_r_prefix=require_prefix)
        if value is not None:
            return value, column
    return None, "unparsed"


def discover_round_models(root: Path, rounds: list[int]) -> dict[int, Path]:
    wanted = set(rounds)
    candidates: dict[int, list[Path]] = defaultdict(list)
    if not root.exists():
        return {}
    for p in root.rglob("*.pkl"):
        r = parse_round(p)
        if r in wanted and "xgb" in p.name.lower():
            candidates[r].append(p.resolve())
    chosen: dict[int, Path] = {}
    for r in rounds:
        arr = candidates.get(r, [])
        if not arr:
            continue
        arr = sorted(
            arr,
            key=lambda p: (
                int("hybrid" in p.name.lower()),
                int(f"r{r}_hybrid" in str(p.parent).lower()),
                p.stat().st_mtime,
            ),
            reverse=True,
        )
        chosen[r] = arr[0]
    return chosen


round_models = discover_round_models(model_root, sorted(set(TABLE4_ROUNDS + [TARGET_ROUND])))
round_models[TARGET_ROUND] = r92_model
sidecars_by_round: dict[int, dict[str, Path]] = {}
sidecar_snapshot_dir = OUT / "provenance_sidecars"
sidecar_snapshot_dir.mkdir(parents=True, exist_ok=True)
for r in sorted(set(TABLE4_ROUNDS + [TARGET_ROUND])):
    record_artifact(
        f"xgb_r{r}", round_models.get(r),
        "FOUND" if r in round_models else "MISSING",
        "Checkpoint requested for reconciled Table 4",
    )
    model_path = round_models.get(r)
    if model_path is None:
        continue
    model_dir = model_path.parent
    candidates = {
        "config": model_dir / "config.json",
        "metrics": model_dir / "metrics.json",
        "threshold_round": model_dir / f"threshold_r{r}.json",
        "threshold_generic": model_dir / "threshold.json",
    }
    found_for_round: dict[str, Path] = {}
    for kind, candidate in candidates.items():
        if candidate.exists() and candidate.is_file():
            found_for_round[kind] = candidate.resolve()
            record_artifact(
                f"r{r}_{kind}_sidecar", candidate.resolve(), "FOUND",
                "Checkpoint-side provenance file copied into the audit output.",
            )
            destination = sidecar_snapshot_dir / f"r{r}__{kind}__{candidate.name}"
            shutil.copy2(candidate, destination)
    sidecars_by_round[r] = found_for_round
pd.DataFrame(discovery_rows).to_csv(OUT / "discovered_inputs.tsv", sep="\t", index=False)


def normalize_image_name(value: Any) -> str:
    return Path(str(value)).name


TILE_PATTERNS = [
    re.compile(r"^(?P<root>.+?)_y\d{1,8}x\d{1,8}$", re.I),
    re.compile(r"^(?P<root>.+?)_x\d{1,8}_y\d{1,8}$", re.I),
    re.compile(r"^(?P<root>.+?)(?:__tile-\d+|-tile-\d+)$", re.I),
]


def original_card(value: Any) -> str:
    stem = Path(str(value)).stem
    for pat in TILE_PATTERNS:
        m = pat.match(stem)
        if m:
            return m.group("root")
    m = re.search(r"(IMG_\d+)", stem, flags=re.I)
    if m:
        return m.group(1).upper()
    return stem


POINT_AUDIT_REFERENCE = sorted({
    original_card(x) for x in POINT_AUDIT_REFERENCE_RAW.split(",") if x.strip()
})
INSTANCE_REVISION_REFERENCE = sorted({
    original_card(x) for x in INSTANCE_REVISION_REFERENCE_RAW.split(",") if x.strip()
})


def load_and_clean_testset(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    note(f"[LOAD] Candidate audit table: {path}")
    df = pd.read_csv(path, low_memory=False)
    original_n = len(df)
    if "human_label" not in df.columns:
        for c in ["label", "y_true", "target", "gt"]:
            if c in df.columns:
                df["human_label"] = df[c]
                break
    if "human_label" not in df.columns:
        raise RuntimeError("Test-set table does not contain a human label column.")
    df["human_label"] = pd.to_numeric(df["human_label"], errors="coerce")
    df = df[df["human_label"].isin([0, 1])].copy()
    df["human_label"] = df["human_label"].astype(np.int8)

    if "action" not in df.columns:
        df["action"] = ""
    actions = df["action"].astype(str).str.strip().str.lower()
    if "review_weight" in df.columns:
        w = pd.to_numeric(df["review_weight"], errors="coerce")
        derived = actions.map(WEIGHT_MAP)
        df["review_weight"] = w.where(w.notna(), derived).fillna(1.0)
    else:
        df["review_weight"] = actions.map(WEIGHT_MAP).fillna(1.0)
    df["review_weight"] = df["review_weight"].astype(float)

    # Enforce one effective record per candidate if a stale merged table has duplicates.
    key_cols = [c for c in ["img_folder", "image", "id"] if c in df.columns]
    duplicate_rows = 0
    if len(key_cols) >= 2:
        if "timestamp" in df.columns:
            df["_audit_ts"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(-1)
            df = df.sort_values(key_cols + ["_audit_ts"])
        duplicate_rows = int(df.duplicated(key_cols, keep="last").sum())
        df = df.drop_duplicates(key_cols, keep="last").copy()
        df = df.drop(columns=["_audit_ts"], errors="ignore")

    zero_weight = int((df["review_weight"] <= 0).sum())
    df = df[df["review_weight"] > 0].copy()

    if "img_folder" not in df.columns:
        if "root" in df.columns:
            df["img_folder"] = df["root"].map(original_card)
        elif "image" in df.columns:
            df["img_folder"] = df["image"].map(original_card)
        else:
            raise RuntimeError("Could not infer original card identity from the test-set table.")
    df["img_folder"] = df["img_folder"].astype(str).map(original_card)

    if "image" in df.columns:
        df["image"] = df["image"].astype(str).map(normalize_image_name)

    info = {
        "rows_read": int(original_n),
        "valid_binary_labels_before_zero_weight_drop": int(original_n - (original_n - len(df) - zero_weight)),
        "duplicate_effective_rows_removed": int(duplicate_rows),
        "zero_weight_rows_removed": int(zero_weight),
        "canonical_evaluation_rows": int(len(df)),
        "positives": int(df["human_label"].sum()),
        "negatives": int((df["human_label"] == 0).sum()),
        "cards": sorted(df["img_folder"].unique().tolist()),
        "weight_rule": WEIGHT_MAP,
        "positive_class": POSITIVE_CLASS,
    }
    return df.reset_index(drop=True), info


df_eval, testset_info = load_and_clean_testset(paper_testset)
(OUT / "evaluation_population.json").write_text(json.dumps(testset_info, indent=2, sort_keys=True))

cards_found = set(testset_info["cards"])
missing_reference_cards = sorted(set(TEST_CARD_REFERENCE) - cards_found)
extra_cards = sorted(cards_found - set(TEST_CARD_REFERENCE))
if missing_reference_cards or extra_cards:
    warnings.append(
        f"Candidate audit card list differs from the manuscript reference. "
        f"Missing={missing_reference_cards}; extra={extra_cards}"
    )


PREFERRED_DICT_KEYS = [
    "pipeline", "model", "estimator", "clf", "classifier", "pipe",
    "xgb", "xgb_model", "booster", "bst", "xgb_booster", "sk_model",
]


def looks_scorable(obj: Any) -> bool:
    if obj is None:
        return False
    if any(hasattr(obj, a) for a in ["predict_proba", "decision_function", "predict"]):
        return True
    return obj.__class__.__module__.startswith("xgboost") and obj.__class__.__name__ == "Booster"


def unwrap_model(obj: Any, max_depth: int = 8) -> tuple[Optional[Any], str]:
    if looks_scorable(obj):
        return obj, "."
    if max_depth <= 0:
        return None, ""
    if isinstance(obj, dict):
        for k in PREFERRED_DICT_KEYS:
            if k in obj:
                m, p = unwrap_model(obj[k], max_depth - 1)
                if m is not None:
                    return m, f".{k}{p}"
        for k, v in obj.items():
            m, p = unwrap_model(v, max_depth - 1)
            if m is not None:
                return m, f".{k}{p}"
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            m, p = unwrap_model(v, max_depth - 1)
            if m is not None:
                return m, f"[{i}]{p}"
    else:
        for attr in PREFERRED_DICT_KEYS:
            if hasattr(obj, attr):
                try:
                    v = getattr(obj, attr)
                except Exception:
                    continue
                m, p = unwrap_model(v, max_depth - 1)
                if m is not None:
                    return m, f".{attr}{p}"
    return None, ""


def first_feature_names(obj: Any, seen: Optional[set[int]] = None) -> Optional[list[str]]:
    if obj is None:
        return None
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return None
    seen.add(oid)
    for attr in ["feature_names_in_", "feature_name_", "feature_names"]:
        if hasattr(obj, attr):
            try:
                vals = list(getattr(obj, attr))
                if vals:
                    return [str(x) for x in vals]
            except Exception:
                pass
    if hasattr(obj, "get_booster"):
        try:
            vals = obj.get_booster().feature_names
            if vals:
                return [str(x) for x in vals]
        except Exception:
            pass
    if isinstance(obj, dict):
        for k in ["features", "feature_names", "xgb_features", "cols", "columns"]:
            if k in obj and isinstance(obj[k], (list, tuple, np.ndarray, pd.Index)):
                vals = [str(x) for x in list(obj[k])]
                if vals:
                    return vals
        for v in obj.values():
            vals = first_feature_names(v, seen)
            if vals:
                return vals
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            vals = first_feature_names(v, seen)
            if vals:
                return vals
    if hasattr(obj, "named_steps"):
        try:
            for v in obj.named_steps.values():
                vals = first_feature_names(v, seen)
                if vals:
                    return vals
        except Exception:
            pass
    return None


NON_FEATURES = {
    "human_label", "label", "y_true", "target", "action", "review_weight",
    "timestamp", "image", "id", "root", "img_folder", "key",
}


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def predict_score(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        return p[:, 1].astype(float) if p.ndim == 2 else p.astype(float)
    if hasattr(model, "decision_function"):
        return sigmoid(np.asarray(model.decision_function(X), dtype=float))
    if model.__class__.__module__.startswith("xgboost") and model.__class__.__name__ == "Booster":
        import xgboost as xgb
        p = np.asarray(model.predict(xgb.DMatrix(X, feature_names=list(X.columns))))
        return p.astype(float) if np.nanmin(p) >= 0 and np.nanmax(p) <= 1 else sigmoid(p)
    if hasattr(model, "predict"):
        p = np.asarray(model.predict(X), dtype=float)
        if p.ndim == 2:
            p = p[:, -1]
        return p if np.nanmin(p) >= 0 and np.nanmax(p) <= 1 else sigmoid(p)
    raise RuntimeError(f"Model type is not scorable: {type(model)}")


def model_meta(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict) and isinstance(raw.get("meta"), dict):
        return dict(raw["meta"])
    return {}


def metadata_round_number(meta: dict[str, Any]) -> Optional[int]:
    for key, require_prefix in [
        ("model_name", True),
        ("round_tag", False),
        ("round", False),
    ]:
        value = round_number_from_text(meta.get(key), require_r_prefix=require_prefix)
        if value is not None:
            return value
    return None


def sanitize_json(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return str(type(value))
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): sanitize_json(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return [sanitize_json(v, depth + 1) for v in list(value)]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def load_json_dict(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        warnings.append(f"Could not parse JSON sidecar {path}: {exc}")
        return {}


def augmentation_semantics(config: dict[str, Any]) -> dict[str, Any]:
    """Describe train_n semantics without comparing augmented rows to a raw CSV."""
    aug_cfg = config.get("AUG_CFG")
    result: dict[str, Any] = {
        "config_available": bool(config),
        "aug_cfg_available": isinstance(aug_cfg, dict),
        "apply_in_3A": None,
        "enabled_stages": [],
        "nominal_sequential_multiplier": None,
        "model_train_n_semantics": "UNKNOWN",
        "row_counts_directly_comparable_to_raw_feature_csv": False,
    }
    if not isinstance(aug_cfg, dict):
        return result
    enabled_master = bool(aug_cfg.get("apply_in_3A", False))
    result["apply_in_3A"] = enabled_master
    multiplier = 1.0
    enabled_stages: list[dict[str, Any]] = []
    if enabled_master:
        # The training notebook applies these sequentially to the expanding
        # training array: mixup -> dropout -> jitter.
        for stage in ["mixup", "dropout", "jitter"]:
            cfg = aug_cfg.get(stage)
            if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
                continue
            try:
                mult = float(cfg.get("mult", 0.0))
            except Exception:
                mult = 0.0
            enabled_stages.append({"stage": stage, "mult": mult})
            multiplier *= 1.0 + mult
        result["enabled_stages"] = enabled_stages
        result["nominal_sequential_multiplier"] = multiplier
        result["model_train_n_semantics"] = "POST_AUGMENTATION_TRAINING_ROWS"
    else:
        result["nominal_sequential_multiplier"] = 1.0
        result["model_train_n_semantics"] = "UNAUGMENTED_OR_FILTERED_TRAINING_ROWS"
    return result


def latex_escape_text(value: Any) -> str:
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(mapping.get(ch, ch) for ch in str(value))


def dataframe_to_latex(df: pd.DataFrame, digits: int = 3) -> str:
    """Small dependency-free LaTeX exporter with escaped headers/text."""
    align = "".join(
        "r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in df.columns
    )
    lines = [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape_text(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, (int, np.integer)):
                cells.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.{digits}f}")
            else:
                cells.append(latex_escape_text(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def score_one_model(path: Path, df: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    note(f"[MODEL] Loading {path}")
    raw = joblib.load(path)
    model, unwrap_path = unwrap_model(raw)
    if model is None:
        raise RuntimeError(f"Could not unwrap a scorable model from {path}")
    features = first_feature_names(raw) or first_feature_names(model)
    if not features:
        features = [
            c for c in df.columns
            if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])
        ]
        feature_source = "numeric_columns_fallback"
    else:
        feature_source = "model_artifact"
    features = [f for f in features if f not in NON_FEATURES]
    missing = [f for f in features if f not in df.columns]
    extra = [c for c in df.columns if c not in set(features) and c not in NON_FEATURES]
    X = df.reindex(columns=features).copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    fill_used = "nan"
    try:
        p = predict_score(model, X)
    except Exception:
        fill_used = "zero_after_nan_failure"
        p = predict_score(model, X.fillna(0.0))
    p = np.asarray(p, dtype=float).reshape(-1)
    if len(p) != len(df):
        raise RuntimeError(f"Prediction length mismatch: {len(p)} != {len(df)}")
    if not np.isfinite(p).all():
        raise RuntimeError("Model produced NaN/Inf scores.")
    if np.nanmin(p) < -1e-6 or np.nanmax(p) > 1 + 1e-6:
        raise RuntimeError("Model scores are outside [0,1].")
    diag = {
        "model_path": str(path),
        "model_sha256": sha256_file(path),
        "model_size_bytes": path.stat().st_size,
        "raw_type": str(type(raw)),
        "unwrapped_type": str(type(model)),
        "unwrap_path": unwrap_path,
        "feature_source": feature_source,
        "n_features": len(features),
        "n_missing_features": len(missing),
        "missing_features": missing[:100],
        "n_extra_columns": len(extra),
        "fill_policy": fill_used,
    }
    return p, diag, model_meta(raw)


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def counts_and_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray, tau: float) -> dict[str, float]:
    pred = p >= tau
    y1 = y == 1
    y0 = ~y1
    ww = np.asarray(w, dtype=float)
    tp = float(ww[y1 & pred].sum())
    fp = float(ww[y0 & pred].sum())
    fn = float(ww[y1 & ~pred].sum())
    tn = float(ww[y0 & ~pred].sum())
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    specificity = safe_ratio(tn, tn + fp)
    fpr = safe_ratio(fp, fp + tn)
    fnr = safe_ratio(fn, fn + tp)
    accuracy = safe_ratio(tp + tn, tp + tn + fp + fn)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": precision, "recall": recall, "f1": f1,
        "specificity": specificity, "fpr": fpr, "fnr": fnr,
        "accuracy": accuracy,
    }


def best_f1(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y, p, sample_weight=w)
    if len(thresholds) == 0:
        return {"best_threshold": 1.0, "best_precision": 0.0, "best_recall": 0.0, "best_f1": 0.0}
    f1s = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    idx = int(np.nanargmax(f1s))
    return {
        "best_threshold": float(thresholds[idx]),
        "best_precision": float(precision[idx]),
        "best_recall": float(recall[idx]),
        "best_f1": float(f1s[idx]),
    }


def full_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray, tau: float) -> dict[str, float]:
    out = counts_and_metrics(y, p, w, tau)
    out["pr_auc"] = float(average_precision_score(y, p, sample_weight=w))
    out["roc_auc"] = float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) > 1 else float("nan")
    out.update(best_f1(y, p, w))
    return out


y = df_eval["human_label"].to_numpy(dtype=np.int8)
review_w = df_eval["review_weight"].to_numpy(dtype=float)
raw_w = np.ones(len(df_eval), dtype=float)

model_rows: list[dict[str, Any]] = []
score_by_round: dict[int, np.ndarray] = {}
diag_rows: list[dict[str, Any]] = []

for round_id in TABLE4_ROUNDS:
    model_path = round_models.get(round_id)
    if model_path is None:
        warnings.append(f"Checkpoint r{round_id} was not found; its Table 4 row was not regenerated.")
        continue
    try:
        p, diag, meta = score_one_model(model_path, df_eval)
        score_by_round[round_id] = p
        model_meta_by_round[round_id] = sanitize_json(meta)
        diag["round"] = round_id
        diag_rows.append(diag)
        if int(diag.get("n_missing_features", 0)) > 0:
            missing_preview = ", ".join(diag.get("missing_features", [])[:20])
            warnings.append(
                f"Checkpoint r{round_id} was scored with {diag['n_missing_features']} expected "
                f"feature(s) absent from the audit table ({missing_preview}). Missing values were "
                f"passed as NaN to the model; this checkpoint is not feature-complete for a strict "
                f"cross-checkpoint comparison."
            )
        meta_round = metadata_round_number(meta)
        if meta_round is not None and meta_round != round_id:
            warnings.append(
                f"Checkpoint path r{round_id} contains metadata identifying r{meta_round}; "
                "treat its provenance metadata as mismatched even though its scores were generated."
            )
        raw = full_metrics(y, p, raw_w, TAU)
        weighted = full_metrics(y, p, review_w, TAU)
        row: dict[str, Any] = {
            "round": round_id,
            "model_path": str(model_path),
            "threshold": TAU,
            "n_candidates": len(y),
            "n_positive": int(y.sum()),
            "sum_review_weights": float(review_w.sum()),
        }
        row.update({f"raw_{k}": v for k, v in raw.items()})
        row.update({f"weighted_{k}": v for k, v in weighted.items()})
        model_rows.append(row)
        note(
            f"[r{round_id}] raw P/R/F1={raw['precision']:.4f}/{raw['recall']:.4f}/{raw['f1']:.4f}; "
            f"weighted={weighted['precision']:.4f}/{weighted['recall']:.4f}/{weighted['f1']:.4f}"
        )
    except Exception as exc:
        warnings.append(f"Checkpoint r{round_id} could not be evaluated: {exc}")
        (OUT / f"r{round_id}_error.txt").write_text(traceback.format_exc())

if TARGET_ROUND not in score_by_round:
    raise SystemExit(f"The primary checkpoint r{TARGET_ROUND} could not be evaluated; see warnings/errors.")

metrics_df = pd.DataFrame(model_rows)
metrics_df.to_csv(OUT / "table4_reconciled_raw_and_weighted.csv", index=False)
pd.DataFrame(diag_rows).to_csv(OUT / "model_scoring_diagnostics.tsv", sep="\t", index=False)
(OUT / "model_metadata_by_round.json").write_text(json.dumps(model_meta_by_round, indent=2, sort_keys=True))

# Preserve every regenerated score vector so all checkpoint rows can be
# independently recomputed without reopening the large model files.
score_export = df_eval[[c for c in ["img_folder", "image", "id", "human_label", "action", "review_weight"] if c in df_eval.columns]].copy()
for round_id in sorted(score_by_round):
    score_export[f"score_r{round_id}"] = score_by_round[round_id]
score_export.to_csv(OUT / "checkpoint_scores_all_rounds.csv", index=False)

compact_cols = [
    "round", "threshold", "n_candidates",
    "raw_pr_auc", "raw_precision", "raw_recall", "raw_f1",
    "weighted_pr_auc", "weighted_precision", "weighted_recall", "weighted_f1",
    "raw_best_threshold", "raw_best_f1",
    "weighted_best_threshold", "weighted_best_f1",
]
compact = metrics_df[[c for c in compact_cols if c in metrics_df.columns]].copy()
compact.to_csv(OUT / "table4_manuscript_compact.csv", index=False)
try:
    (OUT / "table4_manuscript_compact.tex").write_text(
        dataframe_to_latex(compact, digits=3)
    )
except Exception as exc:
    warnings.append(f"LaTeX export for Table 4 failed: {exc}")

# Journal-friendly alternatives: a compact raw-count main table and a
# side-by-side raw/weighted supplementary table. These avoid the overly wide
# 15-column diagnostic export above.
main_table = metrics_df[[
    "round", "raw_pr_auc", "raw_roc_auc", "raw_f1", "raw_precision",
    "raw_recall", "raw_best_f1", "raw_best_threshold",
]].rename(columns={
    "round": "Round",
    "raw_pr_auc": "AP",
    "raw_roc_auc": "ROC-AUC",
    "raw_f1": "F1@0.5",
    "raw_precision": "P@0.5",
    "raw_recall": "R@0.5",
    "raw_best_f1": "Audit best F1*",
    "raw_best_threshold": "Audit tau*",
})
main_table.to_csv(OUT / "table4_maintext_raw.csv", index=False)
(OUT / "table4_maintext_raw.tex").write_text(
    dataframe_to_latex(main_table, digits=4)
)

weighted_supp = metrics_df[[
    "round", "raw_pr_auc", "raw_precision", "raw_recall", "raw_f1",
    "weighted_pr_auc", "weighted_precision", "weighted_recall", "weighted_f1",
]].rename(columns={
    "round": "Round",
    "raw_pr_auc": "Raw AP",
    "raw_precision": "Raw P",
    "raw_recall": "Raw R",
    "raw_f1": "Raw F1",
    "weighted_pr_auc": "Wtd AP",
    "weighted_precision": "Wtd P",
    "weighted_recall": "Wtd R",
    "weighted_f1": "Wtd F1",
})
weighted_supp.to_csv(OUT / "table4_supplement_raw_weighted.csv", index=False)
(OUT / "table4_supplement_raw_weighted.tex").write_text(
    dataframe_to_latex(weighted_supp, digits=4)
)


# Canonical scored candidate table for r92. All subsequent outputs derive from this.
p92 = score_by_round[TARGET_ROUND]
scored = df_eval.copy()
scored["canonical_score"] = p92
scored["canonical_prediction"] = (p92 >= TAU).astype(np.int8)
keep_scored = [
    c for c in [
        "img_folder", "root", "image", "id", "human_label", "action", "review_weight",
        "is_border", "is_tiny", "is_border_or_tiny", "touching_border",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "canonical_score", "canonical_prediction",
    ] if c in scored.columns
]
scored[keep_scored].to_csv(OUT / f"scored_candidates_r{TARGET_ROUND}.csv", index=False)


def compute_border_tiny(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    existing = None
    for c in ["is_border_or_tiny", "border_or_tiny"]:
        if c in out.columns:
            existing = c
            break
    if existing is not None:
        vals = out[existing]
        if vals.dtype == bool:
            out["audit_border_or_tiny"] = vals
        else:
            out["audit_border_or_tiny"] = vals.astype(str).str.lower().isin(["1", "true", "yes"])
        return out

    def num(c: str, default: float = 0.0) -> np.ndarray:
        if c not in out.columns:
            return np.full(len(out), default, dtype=float)
        return pd.to_numeric(out[c], errors="coerce").fillna(default).to_numpy(dtype=float)

    if {"bbox_x", "bbox_y", "bbox_w", "bbox_h"}.issubset(out.columns):
        x, yy, bw, bh = num("bbox_x"), num("bbox_y"), num("bbox_w"), num("bbox_h")
    elif {"x", "y", "w", "h"}.issubset(out.columns):
        x, yy, bw, bh = num("x"), num("y"), num("w"), num("h")
    else:
        raise RuntimeError("Border/tiny audit requires bbox columns or an existing border/tiny flag.")
    touch = num("touching_border", 0) == 1
    tile_w = num("tile_w", 512)
    tile_h = num("tile_h", 512)
    border = touch | (x <= 4) | (yy <= 4) | ((x + bw) >= (tile_w - 4)) | ((yy + bh) >= (tile_h - 4))
    tiny = (np.minimum(bw, bh) < 32) | ((bw * bh) < 1500)
    out["audit_border_or_tiny"] = border | tiny
    return out


scored = compute_border_tiny(scored)


def metric_row_for_subset(name: str, sub: pd.DataFrame) -> dict[str, Any]:
    ys = sub["human_label"].to_numpy(dtype=np.int8)
    ps = sub["canonical_score"].to_numpy(dtype=float)
    ws = sub["review_weight"].to_numpy(dtype=float)
    raw = full_metrics(ys, ps, np.ones(len(sub)), TAU)
    weighted = full_metrics(ys, ps, ws, TAU)
    row: dict[str, Any] = {
        "slice": name,
        "n_candidates": len(sub),
        "raw_positives": int(ys.sum()),
        "weighted_candidate_mass": float(ws.sum()),
        "weighted_positive_mass": float(ws[ys == 1].sum()),
        "threshold": TAU,
    }
    row.update({f"raw_{k}": v for k, v in raw.items()})
    row.update({f"weighted_{k}": v for k, v in weighted.items()})
    return row


s2_rows = [
    metric_row_for_subset("border-or-tiny", scored[scored["audit_border_or_tiny"]]),
    metric_row_for_subset("non-border/tiny", scored[~scored["audit_border_or_tiny"]]),
    metric_row_for_subset("OVERALL", scored),
]
s2 = pd.DataFrame(s2_rows)
s2.to_csv(OUT / "supp_table_s2_reconciled_raw_and_weighted.csv", index=False)
s2_compact_cols = [
    "slice", "n_candidates", "raw_positives", "threshold",
    "raw_pr_auc", "raw_precision", "raw_recall", "raw_f1",
    "weighted_pr_auc", "weighted_precision", "weighted_recall", "weighted_f1",
]
s2_compact = s2[[c for c in s2_compact_cols if c in s2.columns]]
s2_compact.to_csv(OUT / "supp_table_s2_manuscript_compact.csv", index=False)
try:
    (OUT / "supp_table_s2_manuscript_compact.tex").write_text(
        dataframe_to_latex(s2_compact, digits=3)
    )
except Exception as exc:
    warnings.append(f"LaTeX export for Supplementary Table S2 failed: {exc}")


per_card_rows = []
for card, sub in scored.groupby("img_folder", sort=True):
    row = metric_row_for_subset(str(card), sub)
    row["card_id"] = str(card)
    per_card_rows.append(row)
pd.DataFrame(per_card_rows).to_csv(OUT / "candidate_metrics_per_card_raw_and_weighted.csv", index=False)


# Reconciled Figure 7: raw integer counts and review-weighted mass are visibly separate.
primary = metrics_df[metrics_df["round"] == TARGET_ROUND].iloc[0]
raw_cm = np.array([
    [primary["raw_tn"], primary["raw_fp"]],
    [primary["raw_fn"], primary["raw_tp"]],
], dtype=float)
weighted_cm = np.array([
    [primary["weighted_tn"], primary["weighted_fp"]],
    [primary["weighted_fn"], primary["weighted_tp"]],
], dtype=float)


def row_normalize(cm: np.ndarray) -> np.ndarray:
    den = np.maximum(cm.sum(axis=1, keepdims=True), 1e-15)
    return cm / den


plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
for ax, cm, title, count_kind in [
    (axes[0], raw_cm, "Raw candidate counts", "int"),
    (axes[1], weighted_cm, "Review-weighted candidate mass", "float"),
]:
    norm = row_normalize(cm)
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks([0, 1], ["non-CJ", "CJ"])
    ax.set_yticks([0, 1], ["non-CJ", "CJ"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            count_text = f"{int(round(cm[i,j]))}" if count_kind == "int" else f"{cm[i,j]:.1f}"
            color = "white" if norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{count_text}\n({100*norm[i,j]:.1f}%)", ha="center", va="center", color=color)

axes[0].text(
    0.5, -0.25,
    f"Raw: P={primary['raw_precision']:.3f}, R={primary['raw_recall']:.3f}, F1={primary['raw_f1']:.3f}",
    transform=axes[0].transAxes, ha="center", va="top",
)
axes[1].text(
    0.5, -0.25,
    f"Weighted: P={primary['weighted_precision']:.3f}, R={primary['weighted_recall']:.3f}, F1={primary['weighted_f1']:.3f}",
    transform=axes[1].transAxes, ha="center", va="top",
)
fig.suptitle(f"Candidate-level confusion matrices for r{TARGET_ROUND} at score threshold {TAU:.2f}", fontsize=10)
fig.savefig(OUT / "figure7_reconciled_raw_and_weighted.pdf", bbox_inches="tight")
fig.savefig(OUT / "figure7_reconciled_raw_and_weighted.png", dpi=400, bbox_inches="tight")
plt.close(fig)


# Legacy-versus-canonical comparison, with formulas checked from the published counts.
legacy_raw_precision = LEGACY_FIG7["tp"] / (LEGACY_FIG7["tp"] + LEGACY_FIG7["fp"])
legacy_raw_recall = LEGACY_FIG7["tp"] / (LEGACY_FIG7["tp"] + LEGACY_FIG7["fn"])
legacy_raw_f1 = 2 * legacy_raw_precision * legacy_raw_recall / (legacy_raw_precision + legacy_raw_recall)
legacy_compare = pd.DataFrame([
    {
        "source": "legacy_Figure_7_raw_counts_recalculated",
        **LEGACY_FIG7,
        "precision": legacy_raw_precision,
        "recall": legacy_raw_recall,
        "f1": legacy_raw_f1,
    },
    {
        "source": "legacy_Table_4_reported_scalars",
        "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan,
        **LEGACY_TABLE4_R92,
    },
    {
        "source": "canonical_r92_raw",
        "tn": primary["raw_tn"], "fp": primary["raw_fp"],
        "fn": primary["raw_fn"], "tp": primary["raw_tp"],
        "precision": primary["raw_precision"], "recall": primary["raw_recall"], "f1": primary["raw_f1"],
    },
    {
        "source": "canonical_r92_review_weighted",
        "tn": primary["weighted_tn"], "fp": primary["weighted_fp"],
        "fn": primary["weighted_fn"], "tp": primary["weighted_tp"],
        "precision": primary["weighted_precision"], "recall": primary["weighted_recall"], "f1": primary["weighted_f1"],
    },
])
legacy_compare.to_csv(OUT / "legacy_vs_reconciled_r92.csv", index=False)


# ---------------- Model provenance, split, and leakage audit ----------------
r92_meta = model_meta_by_round.get(TARGET_ROUND, {})
r92_sidecars = sidecars_by_round.get(TARGET_ROUND, {})
r92_config_path = r92_sidecars.get("config")
r92_metrics_path = r92_sidecars.get("metrics")
r92_config = load_json_dict(r92_config_path)
r92_metrics_sidecar = load_json_dict(r92_metrics_path)
r92_augmentation = augmentation_semantics(r92_config)

sidecar_inventory_rows: list[dict[str, Any]] = []
for round_id, files in sorted(sidecars_by_round.items()):
    for kind, path in sorted(files.items()):
        sidecar_inventory_rows.append({
            "round": round_id,
            "kind": kind,
            "source_path": str(path),
            "source_sha256": sha256_file(path),
            "snapshot_file": f"provenance_sidecars/r{round_id}__{kind}__{path.name}",
        })
pd.DataFrame(sidecar_inventory_rows).to_csv(
    OUT / "checkpoint_sidecar_inventory.csv", index=False
)


def expand_recorded_path(value: Any) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.replace("${CJ_CLEAN_ROOT}", str(SOURCE_ROOT))
    p = Path(text).expanduser()
    return p.resolve() if p.exists() else None


for key in ["csv_path", "csv_used", "features_csv"]:
    meta_path = expand_recorded_path(r92_meta.get(key))
    if meta_path is not None:
        training_features = meta_path
        break

record_artifact("training_feature_table_resolved_from_r92", training_features, "FOUND" if training_features and training_features.exists() else "MISSING")
pd.DataFrame(discovery_rows).to_csv(OUT / "discovered_inputs.tsv", sep="\t", index=False)


def read_csv_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def pick_name_col(cols: list[str]) -> Optional[str]:
    for c in ["image", "file_name", "filename", "file", "name"]:
        if c in cols:
            return c
    return None


def aggregate_training_groups(path: Path) -> pd.DataFrame:
    cols = read_csv_header(path)
    name_col = pick_name_col(cols)
    if name_col is None:
        raise RuntimeError(f"No image/name column in training feature table: {path}")
    label_col = next((c for c in ["label", "human_label", "y", "target"] if c in cols), None)
    usecols = [name_col] + ([label_col] if label_col else [])
    counts: Counter[str] = Counter()
    positives: Counter[str] = Counter()
    total = 0
    note(f"[SPLIT] Reading only {usecols} from training feature table ({path.stat().st_size/1024**2:.1f} MiB)")
    for chunk_i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=200_000, low_memory=False), 1):
        groups = chunk[name_col].astype(str).map(original_card)
        vc = groups.value_counts()
        counts.update({str(k): int(v) for k, v in vc.items()})
        if label_col:
            labels = pd.to_numeric(chunk[label_col], errors="coerce").fillna(0)
            pos = pd.DataFrame({"g": groups, "p": (labels == 1).astype(int)}).groupby("g")["p"].sum()
            positives.update({str(k): int(v) for k, v in pos.items()})
        total += len(chunk)
        note(f"[SPLIT] processed {total:,} training rows")
    rows = [
        {"card_id": g, "n_rows": counts[g], "n_positive_rows": positives[g] if label_col else np.nan}
        for g in sorted(counts)
    ]
    return pd.DataFrame(rows)


training_groups_df = pd.DataFrame(columns=["card_id", "n_rows", "n_positive_rows"])
if training_features is not None and training_features.exists():
    try:
        training_groups_df = aggregate_training_groups(training_features)
        training_groups_df.to_csv(OUT / "training_source_card_counts.csv", index=False)
    except Exception as exc:
        warnings.append(f"Could not read training card identities: {exc}")
        (OUT / "training_group_error.txt").write_text(traceback.format_exc())
else:
    warnings.append("Training feature table was not found; direct train-vs-audit card overlap could not be checked.")


def internal_test_groups_from_preds(path: Optional[Path]) -> set[str]:
    if path is None or not path.exists():
        return set()
    df = pd.read_csv(path, low_memory=False)
    if "group" in df.columns:
        return set(df["group"].astype(str).map(original_card).unique())
    col = pick_name_col(list(df.columns))
    if col:
        return set(df[col].astype(str).map(original_card).unique())
    return set()


internal_test_groups = internal_test_groups_from_preds(r92_tile_preds)
all_training_source_groups = set(training_groups_df["card_id"].astype(str)) if len(training_groups_df) else set()
internal_train_groups = all_training_source_groups - internal_test_groups
external_audit_groups = set(df_eval["img_folder"].astype(str).map(original_card).unique())
point_audit_groups = set(POINT_AUDIT_REFERENCE)
instance_revision_groups = set(INSTANCE_REVISION_REFERENCE)

union_cards = sorted(
    all_training_source_groups
    | internal_test_groups
    | external_audit_groups
    | point_audit_groups
    | instance_revision_groups
)
split_rows = []
for card in union_cards:
    in_source = card in all_training_source_groups
    in_internal_test = card in internal_test_groups
    in_internal_train = card in internal_train_groups
    in_external = card in external_audit_groups
    in_point_audit = card in point_audit_groups
    in_instance_revision = card in instance_revision_groups
    if in_external and in_source:
        leakage = "YES_DIRECT_CARD_OVERLAP"
    else:
        leakage = "NO_DIRECT_CARD_OVERLAP"
    split_rows.append({
        "card_id": card,
        "in_model_source_table": in_source,
        "in_internal_train_or_validation": in_internal_train,
        "in_internal_test": in_internal_test,
        "in_external_candidate_audit": in_external,
        "in_five_card_point_audit": in_point_audit,
        "in_three_card_instance_revision": in_instance_revision,
        "direct_training_audit_overlap": leakage,
    })
split_manifest = pd.DataFrame(split_rows)
split_manifest.to_csv(OUT / "card_split_manifest.csv", index=False)

direct_overlap = sorted(external_audit_groups & all_training_source_groups)
internal_test_train_overlap = sorted(internal_test_groups & internal_train_groups)

evaluation_sets = {
    "model_source_current_snapshot": all_training_source_groups,
    "internal_test_saved_predictions": internal_test_groups,
    "candidate_audit_10": external_audit_groups,
    "point_audit_5": point_audit_groups,
    "instance_revision_3": instance_revision_groups,
}
overlap_rows = []
set_names = list(evaluation_sets)
for i, set_a in enumerate(set_names):
    for set_b in set_names[i:]:
        overlap = sorted(evaluation_sets[set_a] & evaluation_sets[set_b])
        overlap_rows.append({
            "set_a": set_a,
            "set_b": set_b,
            "n_set_a": len(evaluation_sets[set_a]),
            "n_set_b": len(evaluation_sets[set_b]),
            "n_overlap": len(overlap),
            "overlap_cards": ";".join(overlap),
        })
pd.DataFrame(overlap_rows).to_csv(OUT / "evaluation_set_overlap_matrix.csv", index=False)

candidate_point_overlap = sorted(external_audit_groups & point_audit_groups)
candidate_instance_overlap = sorted(external_audit_groups & instance_revision_groups)
training_point_overlap = sorted(all_training_source_groups & point_audit_groups)
training_instance_overlap = sorted(all_training_source_groups & instance_revision_groups)
if candidate_point_overlap:
    warnings.append(
        "The five-card point audit is not disjoint from the 10-card candidate audit; "
        f"overlap={candidate_point_overlap}."
    )
if candidate_instance_overlap:
    warnings.append(
        "The three-card instance/mask revision subset is not disjoint from the 10-card "
        f"candidate audit; overlap={candidate_instance_overlap}."
    )


# Reconstruct group-level 5-fold allocation and early-stopping holdout for documented reproducibility.
fold_rows: list[dict[str, Any]] = []
early_rows: list[dict[str, Any]] = []
if len(training_groups_df) and internal_train_groups:
    train_counts = training_groups_df[training_groups_df["card_id"].isin(internal_train_groups)].copy()
    # Equivalent to sklearn GroupKFold's non-shuffled group-size balancing.
    fold_load = [0] * 5
    fold_assignment: dict[str, int] = {}
    for _, row in train_counts.sort_values(["n_rows", "card_id"], ascending=[False, True]).iterrows():
        fold = int(np.argmin(fold_load))
        card = str(row["card_id"])
        fold_assignment[card] = fold + 1
        fold_load[fold] += int(row["n_rows"])
    for card in sorted(internal_train_groups):
        fold_rows.append({
            "card_id": card,
            "groupkfold_validation_fold": fold_assignment.get(card),
            "reconstruction_basis": "GroupKFold(n_splits=5, shuffle=False), group-size-balanced",
        })

    cards_arr = np.array(sorted(internal_train_groups), dtype=object)
    dummy_y = np.zeros(len(cards_arr), dtype=np.int8)
    if len(cards_arr) >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=165)
        tr_idx, va_idx = next(gss.split(cards_arr, dummy_y, groups=cards_arr))
        es_val = set(cards_arr[va_idx].tolist())
        for card in sorted(internal_train_groups):
            early_rows.append({
                "card_id": card,
                "early_stopping_role": "validation" if card in es_val else "refit_train",
                "random_state": 165,
                "validation_fraction_of_groups": 0.30,
                "reconstruction_note": "Reconstructed from documented GroupShuffleSplit rule; compare with saved artifacts if available.",
            })

pd.DataFrame(fold_rows).to_csv(OUT / "groupkfold_card_assignments_reconstructed.csv", index=False)
pd.DataFrame(early_rows).to_csv(OUT / "early_stopping_card_split_reconstructed.csv", index=False)


history_info: dict[str, Any] = {"available": False}
hist_enriched = pd.DataFrame()
hist_round_summary = pd.DataFrame()
if history_csv is not None and history_csv.exists():
    try:
        hist_enriched = pd.read_csv(history_csv, low_memory=False)
        parsed = hist_enriched.apply(history_round_info, axis=1, result_type="expand")
        hist_enriched["round_number"] = pd.to_numeric(parsed[0], errors="coerce")
        hist_enriched["round_number_source"] = parsed[1].astype(str)
        hist_enriched.to_csv(OUT / "model_history_snapshot.csv", index=False)

        valid_history = hist_enriched[hist_enriched["round_number"].notna()].copy()
        valid_history["round_number"] = valid_history["round_number"].astype(int)
        if "ts" in valid_history.columns:
            valid_history["_history_ts"] = pd.to_datetime(
                valid_history["ts"], errors="coerce", utc=True
            )
        else:
            valid_history["_history_ts"] = pd.NaT
        # Keep the latest recorded row for each parsed checkpoint in the
        # one-row-per-round summary used for merging and ranking.
        hist_round_summary = (
            valid_history.sort_values(["round_number", "_history_ts"], na_position="first")
            .drop_duplicates("round_number", keep="last")
            .drop(columns=["_history_ts"], errors="ignore")
        )
        hist_round_summary.to_csv(OUT / "model_history_by_round_parsed.csv", index=False)

        r_target_history = valid_history[valid_history["round_number"] == TARGET_ROUND]
        history_info = {
            "available": True,
            "rows": len(hist_enriched),
            "columns": list(hist_enriched.columns),
            "unparsed_rows": int(hist_enriched["round_number"].isna().sum()),
            "r92_rows": sanitize_json(r_target_history.to_dict("records")),
        }

        if "tile_pr_auc" in hist_round_summary.columns:
            ranked_history = hist_round_summary.copy()
            ranked_history["tile_pr_auc"] = pd.to_numeric(
                ranked_history["tile_pr_auc"], errors="coerce"
            )
            ranked_history = ranked_history[ranked_history["tile_pr_auc"].notna()].sort_values(
                ["tile_pr_auc", "round_number"], ascending=[False, True]
            ).reset_index(drop=True)
            target_positions = ranked_history.index[
                ranked_history["round_number"] == TARGET_ROUND
            ].tolist()
            history_info["r92_tile_pr_auc_rank_all_recorded_rounds"] = (
                target_positions[0] + 1 if target_positions else None
            )
            target_rows = ranked_history[ranked_history["round_number"] == TARGET_ROUND]
            target_ap = float(target_rows.iloc[0]["tile_pr_auc"]) if len(target_rows) else None
            history_info["r92_tile_pr_auc"] = target_ap
            if target_ap is not None:
                prior = ranked_history[ranked_history["round_number"] <= TARGET_ROUND]
                history_info["r92_tile_pr_auc_rank_through_r92"] = int(
                    1 + (prior["tile_pr_auc"] > target_ap).sum()
                )
                history_info["higher_tile_pr_auc_rounds_through_r92"] = sanitize_json(
                    prior[prior["tile_pr_auc"] > target_ap][
                        ["round_number", "tile_pr_auc"]
                    ].to_dict("records")
                )
    except Exception as exc:
        warnings.append(f"Model history could not be read: {exc}")


# Selection audit table combines internal-history metrics (where available) and
# the newly regenerated external-audit metrics.
selection_df = compact.copy()
if len(hist_round_summary):
    try:
        keep_hist = [
            c for c in [
                "round_number", "round_number_source", "cv_best_pr_auc",
                "tile_pr_auc", "tile_roc_auc", "n_tiles_train", "n_tiles_test",
                "n_groups_train", "n_groups_test", "threshold", "ts", "model_name",
            ] if c in hist_round_summary.columns
        ]
        history_merge = hist_round_summary[keep_hist].copy()
        history_merge = history_merge.rename(columns={
            c: f"history_{c}" for c in keep_hist if c != "round_number"
        })
        selection_df = selection_df.merge(
            history_merge, left_on="round", right_on="round_number", how="left"
        )
    except Exception as exc:
        warnings.append(f"Could not merge model history into checkpoint audit: {exc}")

# Add artifact metadata beside history metadata. This also exposes legacy
# mismatches such as an r53 path whose embedded round_tag says r1.
artifact_meta_rows = []
for round_id in sorted(metrics_df["round"].astype(int).tolist()):
    meta = model_meta_by_round.get(round_id, {}) or {}
    tile_metrics = meta.get("tile_metrics", {}) if isinstance(meta.get("tile_metrics"), dict) else {}
    artifact_meta_rows.append({
        "round": round_id,
        "artifact_meta_round": metadata_round_number(meta),
        "artifact_round_tag": meta.get("round_tag"),
        "artifact_model_name": meta.get("model_name"),
        "artifact_tile_pr_auc": meta.get("tile_pr_auc", tile_metrics.get("pr_auc_test")),
        "artifact_tile_roc_auc": meta.get("tile_roc_auc", tile_metrics.get("roc_auc_test")),
        "artifact_stored_threshold": meta.get("threshold"),
        "artifact_train_n": meta.get("train_n"),
        "artifact_test_n": meta.get("test_n"),
    })
selection_df = selection_df.merge(pd.DataFrame(artifact_meta_rows), on="round", how="left")
selection_df.to_csv(OUT / "checkpoint_selection_audit.csv", index=False)


selection_notes = []
if direct_overlap:
    training_leakage_status = "FAIL_DIRECT_CARD_OVERLAP"
    selection_notes.append(f"Direct card overlap between model-source table and external candidate audit: {direct_overlap}")
elif all_training_source_groups:
    training_leakage_status = "PASS_CURRENT_SOURCE_SNAPSHOT_NO_CARD_OVERLAP"
    selection_notes.append(
        "No external candidate-audit card was found in the current file at the feature-table path recorded by the model."
    )
else:
    training_leakage_status = "INDETERMINATE_TRAINING_SOURCE_NOT_AUDITED"
    selection_notes.append("Training-source card identities were not available, so direct leakage was not testable.")

internal_test_outside_source = sorted(internal_test_groups - all_training_source_groups)
if internal_test_outside_source and all_training_source_groups:
    internal_split_status = "INCONSISTENT_SAVED_TEST_CARDS_OUTSIDE_CURRENT_SOURCE"
elif internal_test_groups and all_training_source_groups:
    internal_split_status = "PARTIAL_RECONSTRUCTED_NOT_FIT_PROOF"
    selection_notes.append(
        "Internal train/validation membership was inferred by subtracting saved test-prediction cards "
        "from the current source table. This reconstruction cannot prove that those test cards were "
        "excluded from the actual fit performed when r92 was created."
    )
else:
    internal_split_status = "INDETERMINATE"

current_training_rows = int(training_groups_df["n_rows"].sum()) if len(training_groups_df) else None
meta_train_n = pd.to_numeric(pd.Series([r92_meta.get("train_n")]), errors="coerce").iloc[0]
meta_test_n = pd.to_numeric(pd.Series([r92_meta.get("test_n")]), errors="coerce").iloc[0]
meta_train_n_value = None if pd.isna(meta_train_n) else int(meta_train_n)
meta_test_n_value = None if pd.isna(meta_test_n) else int(meta_test_n)
nominal_aug_multiplier = r92_augmentation.get("nominal_sequential_multiplier")
inferred_preaugmentation_train_rows = None
if (
    meta_train_n_value is not None
    and isinstance(nominal_aug_multiplier, (int, float))
    and float(nominal_aug_multiplier) > 0
    and r92_augmentation.get("model_train_n_semantics") == "POST_AUGMENTATION_TRAINING_ROWS"
):
    inferred_preaugmentation_train_rows = meta_train_n_value / float(nominal_aug_multiplier)

# A raw/current feature-table row count is not directly comparable with the
# artifact's train_n/test_n: train_n is post-augmentation when AUG_CFG is active,
# while filtering/splitting also occurs before model fitting. v3 intentionally
# makes no PASS/FAIL claim from this apples-to-oranges comparison.
row_counts_directly_comparable = False
exact_fit_snapshot_status = "NOT_PROVEN_NO_IMMUTABLE_FIT_TIME_SNAPSHOT"
training_snapshot_modified_utc = (
    datetime.fromtimestamp(training_features.stat().st_mtime, timezone.utc).isoformat()
    if training_features is not None and training_features.exists()
    else None
)
r92_history_timestamps = []
for row in history_info.get("r92_rows", []):
    if isinstance(row, dict) and row.get("ts"):
        r92_history_timestamps.append(str(row["ts"]))

r92_history_datetime = pd.NaT
if r92_history_timestamps:
    parsed_times = pd.to_datetime(pd.Series(r92_history_timestamps), errors="coerce", utc=True).dropna()
    if len(parsed_times):
        r92_history_datetime = parsed_times.max()
training_snapshot_datetime = pd.to_datetime(training_snapshot_modified_utc, errors="coerce", utc=True)
current_snapshot_postdates_r92 = (
    bool(training_snapshot_datetime > r92_history_datetime)
    if pd.notna(training_snapshot_datetime) and pd.notna(r92_history_datetime)
    else None
)

if not r92_config:
    warnings.append(
        "The r92 config.json sidecar was not found, so train_n augmentation semantics could not "
        "be confirmed directly from the checkpoint directory. No raw-row-count comparison was made."
    )
elif r92_augmentation.get("model_train_n_semantics") == "POST_AUGMENTATION_TRAINING_ROWS":
    warnings.append(
        "r92 metadata train_n is post-augmentation, whereas the current feature CSV is a raw/current "
        "source table. These row counts are not directly comparable; v3 does not use them to decide "
        "whether the exact fit-time snapshot matches."
    )
if current_snapshot_postdates_r92 is True:
    warnings.append(
        "The current feature file at the path referenced by r92 metadata has a modification time after "
        "the recorded r92 history timestamp. Its card-level no-overlap result is useful supporting "
        "evidence, but the exact fit-time feature snapshot remains unavailable."
    )

stored_threshold = r92_meta.get("threshold")
if stored_threshold is not None and abs(float(stored_threshold) - TAU) > 1e-12:
    warnings.append(
        f"r92 stores threshold={float(stored_threshold):.9f}, whereas this audit intentionally uses "
        f"the manuscript fixed threshold tau={TAU:.2f}. The audit best-F1 threshold is a third, "
        "retrospectively optimized quantity and must not be called an independent deployment threshold."
    )

selection_rank_raw = None
selection_rank_weighted = None
selection_rank_raw_f1 = None
selection_rank_weighted_f1 = None
if len(metrics_df):
    ranked_raw = metrics_df.sort_values(["raw_pr_auc", "round"], ascending=[False, True]).reset_index(drop=True)
    ranked_weighted = metrics_df.sort_values(["weighted_pr_auc", "round"], ascending=[False, True]).reset_index(drop=True)
    raw_matches = ranked_raw.index[ranked_raw["round"] == TARGET_ROUND].tolist()
    weighted_matches = ranked_weighted.index[ranked_weighted["round"] == TARGET_ROUND].tolist()
    selection_rank_raw = raw_matches[0] + 1 if raw_matches else None
    selection_rank_weighted = weighted_matches[0] + 1 if weighted_matches else None
    target_metric_rows = metrics_df[metrics_df["round"] == TARGET_ROUND]
    if len(target_metric_rows):
        target_raw_f1 = float(target_metric_rows.iloc[0]["raw_f1"])
        target_weighted_f1 = float(target_metric_rows.iloc[0]["weighted_f1"])
        selection_rank_raw_f1 = int(1 + (metrics_df["raw_f1"] > target_raw_f1 + 1e-15).sum())
        selection_rank_weighted_f1 = int(1 + (metrics_df["weighted_f1"] > target_weighted_f1 + 1e-15).sum())

selection_independence_status = "NOT_DEMONSTRATED"
selection_notes.append(
    "Multiple checkpoints were evaluated on the same 10-card candidate-audit set. "
    "Unless a training-only/CV rule selecting r92 is documented before inspecting this audit set, "
    "the set must be described as a candidate-level audit set, not an independent model-selection test set."
)

audit_summary = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "source_root": str(SOURCE_ROOT),
    "positive_class": POSITIVE_CLASS,
    "fixed_threshold": TAU,
    "target_checkpoint": f"r{TARGET_ROUND}",
    "table4_rounds_requested": TABLE4_ROUNDS,
    "table4_rounds_regenerated": sorted(metrics_df["round"].astype(int).tolist()),
    "external_candidate_audit_cards": sorted(external_audit_groups),
    "five_card_point_audit_cards": sorted(point_audit_groups),
    "three_card_instance_revision_cards": sorted(instance_revision_groups),
    "candidate_audit_point_audit_overlap": candidate_point_overlap,
    "candidate_audit_instance_revision_overlap": candidate_instance_overlap,
    "current_training_point_audit_overlap": training_point_overlap,
    "current_training_instance_revision_overlap": training_instance_overlap,
    "model_source_cards_count": len(all_training_source_groups),
    "internal_train_or_validation_cards_count": len(internal_train_groups),
    "internal_test_cards_count": len(internal_test_groups),
    "internal_test_cards_outside_current_source": internal_test_outside_source,
    "direct_external_audit_training_overlap": direct_overlap,
    "training_leakage_status": training_leakage_status,
    "internal_split_status": internal_split_status,
    "checkpoint_selection_independence": selection_independence_status,
    "r92_external_raw_pr_auc_rank_among_regenerated_rounds": selection_rank_raw,
    "r92_external_weighted_pr_auc_rank_among_regenerated_rounds": selection_rank_weighted,
    "r92_external_raw_f1_at_fixed_threshold_rank_among_regenerated_rounds": selection_rank_raw_f1,
    "r92_external_weighted_f1_at_fixed_threshold_rank_among_regenerated_rounds": selection_rank_weighted_f1,
    "r92_model_sha256": sha256_file(r92_model),
    "paper_testset_sha256": sha256_file(paper_testset),
    "training_features_sha256": sha256_file(training_features) if training_features and training_features.exists() else None,
    "current_training_feature_rows": current_training_rows,
    "r92_metadata_train_n": meta_train_n_value,
    "r92_metadata_test_n": meta_test_n_value,
    "r92_train_n_semantics": r92_augmentation.get("model_train_n_semantics"),
    "r92_augmentation": r92_augmentation,
    "inferred_preaugmentation_train_rows_approximate": inferred_preaugmentation_train_rows,
    "row_counts_directly_comparable_to_current_raw_csv": row_counts_directly_comparable,
    "exact_fit_training_snapshot_status": exact_fit_snapshot_status,
    "current_training_snapshot_modified_utc": training_snapshot_modified_utc,
    "current_training_snapshot_postdates_r92": current_snapshot_postdates_r92,
    "r92_history_timestamps": r92_history_timestamps,
    "r92_config_sidecar_path": str(r92_config_path) if r92_config_path else None,
    "r92_config_sidecar_sha256": sha256_file(r92_config_path) if r92_config_path else None,
    "r92_metrics_sidecar_path": str(r92_metrics_path) if r92_metrics_path else None,
    "r92_metrics_sidecar_sha256": sha256_file(r92_metrics_path) if r92_metrics_path else None,
    "r92_metrics_sidecar": r92_metrics_sidecar,
    "r92_metadata": r92_meta,
    "model_history": history_info,
    "selection_notes": selection_notes,
}
(OUT / "model_selection_and_leakage_audit.json").write_text(json.dumps(sanitize_json(audit_summary), indent=2, sort_keys=True))

pd.DataFrame([{
    "current_training_feature_path": str(training_features) if training_features else "",
    "current_training_feature_sha256": sha256_file(training_features) if training_features and training_features.exists() else "",
    "current_training_feature_modified_utc": training_snapshot_modified_utc,
    "current_training_feature_rows": current_training_rows,
    "r92_metadata_train_n": meta_train_n_value,
    "r92_metadata_train_n_semantics": r92_augmentation.get("model_train_n_semantics"),
    "r92_metadata_test_n": meta_test_n_value,
    "r92_metadata_test_n_semantics": "UNAUGMENTED_INTERNAL_TEST_ROWS",
    "nominal_sequential_augmentation_multiplier": nominal_aug_multiplier,
    "inferred_preaugmentation_train_rows_approximate": inferred_preaugmentation_train_rows,
    "row_counts_directly_comparable": row_counts_directly_comparable,
    "exact_fit_snapshot_status": exact_fit_snapshot_status,
    "current_snapshot_postdates_r92": current_snapshot_postdates_r92,
    "r92_config_sidecar": str(r92_config_path) if r92_config_path else "",
    "r92_config_sha256": sha256_file(r92_config_path) if r92_config_path else "",
    "r92_history_timestamps": ";".join(r92_history_timestamps),
    "interpretation": (
        "Exact fit snapshot not proven. Raw/current CSV rows are not compared with augmented "
        "artifact train_n; an immutable fit-time manifest or archived feature-table hash is required."
    ),
}]).to_csv(OUT / "training_snapshot_provenance.csv", index=False)


# Automated consistency checks.
checks: list[tuple[str, bool, str]] = []


def add_check(name: str, passed: bool, detail: str) -> None:
    checks.append((name, bool(passed), detail))


for prefix in ["raw", "weighted"]:
    tp = float(primary[f"{prefix}_tp"])
    fp = float(primary[f"{prefix}_fp"])
    fn = float(primary[f"{prefix}_fn"])
    p_calc = safe_ratio(tp, tp + fp)
    r_calc = safe_ratio(tp, tp + fn)
    f_calc = safe_ratio(2 * p_calc * r_calc, p_calc + r_calc)
    add_check(f"r92_{prefix}_precision_matches_counts", abs(p_calc - primary[f"{prefix}_precision"]) < 1e-12, f"{p_calc}")
    add_check(f"r92_{prefix}_recall_matches_counts", abs(r_calc - primary[f"{prefix}_recall"]) < 1e-12, f"{r_calc}")
    add_check(f"r92_{prefix}_f1_matches_counts", abs(f_calc - primary[f"{prefix}_f1"]) < 1e-12, f"{f_calc}")

add_check("candidate_scores_finite", bool(np.isfinite(p92).all()), f"n={len(p92)}")
add_check("candidate_scores_in_unit_interval", bool((p92.min() >= 0) and (p92.max() <= 1)), f"min={p92.min()}, max={p92.max()}")
add_check("ten_candidate_audit_cards_present", len(external_audit_groups) == 10, str(sorted(external_audit_groups)))
add_check(
    "saved_internal_test_cards_are_in_current_source_snapshot",
    not internal_test_outside_source if all_training_source_groups else False,
    str(internal_test_outside_source),
)
add_check(
    "actual_fit_exclusion_of_internal_test_cards_proven",
    False,
    "No immutable fit-time train/test manifest was found; available split is reconstructed.",
)
add_check("no_direct_training_external_audit_card_overlap", not direct_overlap if all_training_source_groups else False, str(direct_overlap))
add_check("five_card_point_audit_disjoint_from_candidate_audit", not candidate_point_overlap, str(candidate_point_overlap))
add_check("three_card_instance_revision_disjoint_from_candidate_audit", not candidate_instance_overlap, str(candidate_instance_overlap))
add_check(
    "raw_current_rows_not_compared_with_augmented_model_train_n",
    row_counts_directly_comparable is False,
    (
        f"current_raw_rows={current_training_rows}; model_train_n={meta_train_n_value}; "
        f"model_train_n_semantics={r92_augmentation.get('model_train_n_semantics')}"
    ),
)
add_check("all_requested_table4_rounds_regenerated", set(TABLE4_ROUNDS).issubset(set(metrics_df["round"].astype(int))), str(sorted(metrics_df["round"].astype(int).tolist())))

check_df = pd.DataFrame(checks, columns=["check", "passed", "detail"])
check_df.to_csv(OUT / "consistency_checks.csv", index=False)

metric_checks = check_df[check_df["check"].str.contains("matches_counts|scores_")]
all_rounds_regenerated = set(TABLE4_ROUNDS).issubset(set(metrics_df["round"].astype(int)))
metric_consistency_pass = bool(metric_checks["passed"].all() and all_rounds_regenerated)
leakage_evidence_complete = False
diag_df = pd.DataFrame(diag_rows)
missing_feature_rounds = sorted(
    diag_df.loc[diag_df["n_missing_features"] > 0, "round"].astype(int).tolist()
) if len(diag_df) else []
metadata_mismatch_rounds = sorted({
    int(round_id)
    for round_id, meta in model_meta_by_round.items()
    if metadata_round_number(meta or {}) is not None
    and metadata_round_number(meta or {}) != int(round_id)
})
history_target_found = bool(history_info.get("r92_rows"))

status_lines = [
    "AUDIT_SCRIPT_VERSION=3.0",
    f"METRIC_RECONCILIATION={'PASS' if metric_consistency_pass else 'FAIL'}",
    f"TRAINING_CARD_LEAKAGE={training_leakage_status}",
    f"INTERNAL_SPLIT={internal_split_status}",
    f"CHECKPOINT_SELECTION_INDEPENDENCE={selection_independence_status}",
    f"LEAKAGE_EVIDENCE_COMPLETE={'YES' if leakage_evidence_complete else 'NO'}",
    f"HISTORY_R{TARGET_ROUND}_FOUND={'YES' if history_target_found else 'NO'}",
    f"EXACT_FIT_TRAINING_SNAPSHOT={exact_fit_snapshot_status}",
    f"R{TARGET_ROUND}_TRAIN_N_SEMANTICS={r92_augmentation.get('model_train_n_semantics')}",
    f"RAW_CURRENT_ROWS_DIRECTLY_COMPARABLE_TO_R{TARGET_ROUND}_TRAIN_N={'YES' if row_counts_directly_comparable else 'NO'}",
    f"R{TARGET_ROUND}_CONFIG_SIDECAR={'FOUND' if r92_config else 'NOT_FOUND'}",
    f"TABLE4_FEATURE_COMPLETENESS={'PASS' if not missing_feature_rounds else 'PARTIAL_MISSING_FEATURES_R' + '_R'.join(map(str, missing_feature_rounds))}",
    f"MODEL_METADATA_MISMATCH_ROUNDS={'NONE' if not metadata_mismatch_rounds else ','.join('r'+str(x) for x in metadata_mismatch_rounds)}",
    f"R{TARGET_ROUND}_EXTERNAL_AP_RANK_AMONG_REGENERATED={selection_rank_raw}",
    f"R{TARGET_ROUND}_EXTERNAL_F1_AT_FIXED_TAU_RANK_AMONG_REGENERATED={selection_rank_raw_f1}",
    f"POINT_AUDIT_DISJOINT_FROM_CANDIDATE_AUDIT={'YES' if not candidate_point_overlap else 'NO_' + '_'.join(candidate_point_overlap)}",
    f"INSTANCE_REVISION_DISJOINT_FROM_CANDIDATE_AUDIT={'YES' if not candidate_instance_overlap else 'NO_' + '_'.join(candidate_instance_overlap)}",
    f"TABLE4_ROUNDS_REGENERATED={','.join(map(str, sorted(metrics_df['round'].astype(int).tolist())))}",
    f"PRIMARY_R92_RAW_P_R_F1={primary['raw_precision']:.9f},{primary['raw_recall']:.9f},{primary['raw_f1']:.9f}",
    f"PRIMARY_R92_WEIGHTED_P_R_F1={primary['weighted_precision']:.9f},{primary['weighted_recall']:.9f},{primary['weighted_f1']:.9f}",
]
(OUT / "AUDIT_STATUS.txt").write_text("\n".join(status_lines) + "\n")


methods_text = f"""Candidate-level metric reconciliation protocol

Evaluation population: {len(df_eval):,} effectively labelled candidate instances from {len(external_audit_groups)} fully reviewed cards.
Positive class: CJ (label 1). Fixed operating threshold: tau={TAU:.2f}.

Raw counts use one unit per candidate. Review-weighted counts use action weights accept/flip=1.0 and sus_accept/sus_flip=0.4; zero-weight skip/auto-accept records are excluded. For each mode, TP, FP, FN, and TN are computed from the same score vector and evaluation population. Precision=TP/(TP+FP), recall=TP/(TP+FN), and F1=2PR/(P+R). Average precision (AP) is computed with sklearn.average_precision_score; it is not a trapezoidal PR-curve integral. ROC-AUC is computed with the corresponding sample weights. The compact main-text table uses raw candidate counts so its scalar metrics and Figure 7 integer counts have the same denominator; review-weighted metrics are reported separately in the supplementary comparison.

Border-or-tiny is defined as touching a tile boundary within 4 pixels, min(width,height)<32 pixels, or bounding-box area<1500 pixels squared. Supplementary Table S2 uses the same r{TARGET_ROUND} score vector, tau, labels, and weight rules as Table 4 and Figure 7.

Figure 7 displays raw integer confusion counts separately from review-weighted candidate mass. A weighted scalar is never printed as though it were derived from raw integer counts. The fixed manuscript operating threshold tau={TAU:.2f}, the threshold stored in a model artifact, and a best-F1 threshold optimized retrospectively on the audit set are distinct quantities and must be labelled separately.
"""
(OUT / "METHODS_METRIC_RECONCILIATION.txt").write_text(methods_text)

selection_text = f"""Model-selection and leakage interpretation

Direct card-level training leakage status: {training_leakage_status}.
Internal train/test split status: {internal_split_status}.
Checkpoint-selection independence: {selection_independence_status}.

The script found {len(all_training_source_groups)} cards in the current file at the feature-table path referenced by r92 metadata, {len(internal_test_groups)} cards in the saved internal test predictions, and {len(external_audit_groups)} cards in the candidate audit. Direct overlap between the candidate-audit cards and this current file: {direct_overlap or 'none detected'}. This is supporting evidence only: the file is not an immutable fit-time snapshot, and the internal train/validation list is reconstructed by subtraction rather than recovered from a saved fit-time split manifest.

Row-count provenance is handled conservatively. The artifact records train_n={meta_train_n_value if meta_train_n_value is not None else 'unknown'} with semantics {r92_augmentation.get('model_train_n_semantics')}, while the current raw/source CSV contains {current_training_rows if current_training_rows is not None else 'an unknown number of'} rows. Because augmentation, filtering, and splitting change these populations, v3 does not compare those row counts or use them as a snapshot-match test. Exact fit-time identity remains {exact_fit_snapshot_status}.

The five-card point audit overlaps the candidate audit on {candidate_point_overlap or 'no cards'}, and the three-card instance/mask revision subset overlaps it on {candidate_instance_overlap or 'no cards'}. These sets must not be described as fully disjoint or fully independent.

Even when there is no direct training-card overlap, comparing multiple checkpoints on the same 10-card set and calling r92 the best model makes that set unsuitable as an independent model-selection test set unless an earlier training-only/CV selection rule is documented. The defensible wording is therefore 'fully reviewed candidate-level audit set' until provenance proves that r92 was frozen before this set was inspected.

Among the regenerated checkpoints, r92 has raw external-audit AP rank {selection_rank_raw}, but fixed-threshold F1 rank {selection_rank_raw_f1}. Therefore it must not be described as best on every metric; this ranking is retrospective on the audit set.
"""
(OUT / "MODEL_SELECTION_INTERPRETATION.txt").write_text(selection_text)

artifact_guidance = f"""Manuscript artifact guidance

Ready after adding an accurate caption:
- figure7_reconciled_raw_and_weighted.pdf (state that percentages are row-normalized and tau={TAU:.2f})
- supplementary Table S2 outputs derived from the feature-complete r{TARGET_ROUND} score vector

Use with explicit caveats:
- table4_maintext_raw.tex and table4_supplement_raw_weighted.tex. Checkpoints {missing_feature_rounds or 'none'} have missing audit features, and checkpoints {metadata_mismatch_rounds or 'none'} have model-metadata mismatches. Do not present this as a strict, fully provenance-verified comparison unless those rows are repaired, omitted, or footnoted.
- AP means sklearn average_precision_score, not trapezoidal PR-curve area.
- Columns marked Audit best are retrospective optimizations on this same audit set and are not deployment thresholds.

Do not insert directly:
- table4_manuscript_compact.tex
- supp_table_s2_manuscript_compact.tex
These diagnostic fragments are too wide for normal journal text width.

Scientific status:
- Candidate-metric reconciliation: {'PASS' if metric_consistency_pass else 'FAIL'}.
- Exact fit-time training snapshot: {exact_fit_snapshot_status}.
- Checkpoint-selection independence: {selection_independence_status}.
"""
(OUT / "MANUSCRIPT_ARTIFACT_GUIDANCE.txt").write_text(artifact_guidance)

if warnings:
    (OUT / "WARNINGS.txt").write_text("\n".join(f"- {w}" for w in warnings) + "\n")
else:
    (OUT / "WARNINGS.txt").write_text("No warnings.\n")

manifest_rows = []
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.name != "OUTPUT_MANIFEST.tsv" and "matplotlib_cache" not in p.parts:
        manifest_rows.append({
            "file": str(p.relative_to(OUT)),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
pd.DataFrame(manifest_rows).to_csv(OUT / "OUTPUT_MANIFEST.tsv", sep="\t", index=False)

note("\n============================================================")
note("CANONICAL CANDIDATE-METRIC / MODEL-SELECTION AUDIT FINISHED")
for line in status_lines:
    note(line)
note(f"Output: {OUT}")
note("============================================================")
PY

# Rebuild the manifest after terminal.log has received the Python completion.
if command -v sha256sum >/dev/null 2>&1; then
    find "$RUN_DIR" -type f ! -path '*/matplotlib_cache/*' ! -name 'OUTPUT_MANIFEST_FINAL.tsv' -print0 \
        | sort -z \
        | while IFS= read -r -d '' f; do
            size="$(stat -c '%s' "$f")"
            sum="$(sha256sum "$f" | awk '{print $1}')"
            rel="${f#"$RUN_DIR"/}"
            printf '%s\t%s\t%s\n' "$rel" "$size" "$sum"
          done > "$RUN_DIR/OUTPUT_MANIFEST_FINAL.tsv"
fi

# A compact archive makes it easy to send the complete result back for review.
if command -v zip >/dev/null 2>&1; then
    (
        cd "$RESULTS_ROOT"
        zip -qr "$(basename "$RUN_DIR")_review_bundle.zip" "$(basename "$RUN_DIR")" \
            -x '*/matplotlib_cache/*'
    )
    REVIEW_ARCHIVE="$RESULTS_ROOT/$(basename "$RUN_DIR")_review_bundle.zip"
else
    REVIEW_ARCHIVE="$RESULTS_ROOT/$(basename "$RUN_DIR")_review_bundle.tar.gz"
    tar -czf "$REVIEW_ARCHIVE" -C "$RESULTS_ROOT" "$(basename "$RUN_DIR")"
fi

echo
echo "============================================================"
echo "AUDIT COMPLETED"
echo "Output folder:"
echo "  $RUN_DIR"
echo
echo "Archive to send back for review:"
echo "  $REVIEW_ARCHIVE"
echo
echo "Primary files:"
echo "  $RUN_DIR/AUDIT_STATUS.txt"
echo "  $RUN_DIR/table4_reconciled_raw_and_weighted.csv"
echo "  $RUN_DIR/figure7_reconciled_raw_and_weighted.pdf"
echo "  $RUN_DIR/supp_table_s2_reconciled_raw_and_weighted.csv"
echo "  $RUN_DIR/model_selection_and_leakage_audit.json"
echo "  $RUN_DIR/card_split_manifest.csv"
echo "  $RUN_DIR/evaluation_set_overlap_matrix.csv"
echo "  $RUN_DIR/training_snapshot_provenance.csv"
echo "  $RUN_DIR/checkpoint_selection_audit.csv"
echo "  $RUN_DIR/checkpoint_scores_all_rounds.csv"
echo "  $RUN_DIR/WARNINGS.txt"
echo "============================================================"
