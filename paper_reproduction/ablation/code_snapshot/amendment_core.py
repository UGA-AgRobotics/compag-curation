"""Protocol Amendment 01 evidence and execution utilities.

The module separates historical replay from the corrected prospective protocol.
Historical quirks are reproduced only as evidence; prospective helpers implement
the amended semantics.  All writes are constrained to the study root and the
previous accepted Run and ZIP are explicitly immutable.
"""

from __future__ import annotations

import contextlib
import ctypes
import csv
import errno
import gzip
import hashlib
import io
import importlib.metadata
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import joblib
import numpy as np
import pandas as pd
import psutil
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import ablation_core as legacy
import amendment02_resampling as paired_resampling
import amendment03_compat as amendment03_compat
import amendment03_recovery as amendment03_recovery
import amendment04_recovery as amendment04_recovery
import amendment_inventory as inventory_tools
import evidence_gates as evidence_workflow


_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_STUDY_ROOT = _MODULE_DIR.parent
STUDY_ROOT = Path(os.environ.get("ABLATION_STUDY_ROOT", str(_DEFAULT_STUDY_ROOT)))
PROJECT_ROOT = Path(os.environ.get("ABLATION_PROJECT_ROOT", "/REVIEWER_INPUT_ROOT/Clean"))
PYTHON = Path(os.environ.get("ABLATION_PYTHON", "/REVIEWER_INPUT_ROOT/python"))
TASK_SPEC = STUDY_ROOT / "docs/PROTOCOL_AMENDMENT_01_TASK_SPEC.md"
PROMPT_SOURCE = Path(os.environ.get("ABLATION_AMENDMENT_PROMPT", "/REVIEWER_INPUT_ROOT/2.txt"))
AMENDMENT02_TASK_SPEC = STUDY_ROOT / "docs/PROTOCOL_AMENDMENT_02_TASK_SPEC.md"
AMENDMENT02_PROMPT_SOURCE = Path(os.environ.get(
    "ABLATION_AMENDMENT02_PROMPT",
    "/REVIEWER_INPUT_ROOT/01_RUN_NOW_Protocol_Amendment_CUDA_Smoke.txt",
))

HISTORICAL_TRAINING_CSV = Path(os.environ.get(
    "ABLATION_HISTORICAL_TRAINING_CSV",
    str(PROJECT_ROOT / "Shared/maskout_tile/IMG_9429 (2025-12-28_11-55-48)/features_train.csv"),
))
HISTORICAL_TILED_COCO = HISTORICAL_TRAINING_CSV.parent / "CJ_NOCJ_tiles_512.json"
TRAINING_TILE_ROOT = Path(os.environ.get(
    "ABLATION_TRAINING_TILE_ROOT",
    str(PROJECT_ROOT / "Shared/PC_codes/img_tiles_512"),
))
CURRENT_TRAINING_CSV = Path(os.environ.get(
    "ABLATION_CURRENT_TRAINING_CSV",
    str(PROJECT_ROOT / "Shared/PC_codes/annotation_2_features/cj_noncj_512_tiles/features_train.csv"),
))
AUDIT_CSV = Path(os.environ.get(
    "ABLATION_AUDIT_CSV",
    str(PROJECT_ROOT / "Shared/maskout_tile/test_set/_paper_testset__xgb_recall/testset_labeled__xgb_recall.csv"),
))
R92_DIR = Path(os.environ.get(
    "ABLATION_R92_DIR",
    str(PROJECT_ROOT / "always_same/jupyter/models/cj_classifier/r92_hybrid"),
))
R92_MODEL = R92_DIR / "cj_ultra_tilesafe_xgb_r92_hybrid.pkl"
R92_PREDICTIONS = R92_DIR / "tile_preds.csv"
R91_DIR = Path(os.environ.get(
    "ABLATION_R91_DIR",
    str(PROJECT_ROOT / "always_same/jupyter/models/cj_classifier/r91_hybrid"),
))
R91_PREDICTIONS = R91_DIR / "tile_preds.csv"
R91_THRESHOLD = R91_DIR / "threshold_r91.json"
SELECTED_V3_RUN = Path(os.environ.get(
    "ABLATION_SELECTED_V3_RUN",
    str(PROJECT_ROOT / "revision/03_04_candidate_metrics_model_selection/results/run_20260805_073743_1efaf813"),
))
SELECTED_V3_ZIP = Path(str(SELECTED_V3_RUN) + "_review_bundle.zip")
SELECTED_V3_SCORES = SELECTED_V3_RUN / "scored_candidates_r92.csv"
SELECTED_V3_POPULATION = SELECTED_V3_RUN / "evaluation_population.json"
PRIOR_RUN = STUDY_ROOT / "results/run_20260812_104222_5a97b4a9_blocked_preflight"
PRIOR_ZIP = Path(str(PRIOR_RUN) + "_review_bundle.zip")
AMENDMENT02_REFERENCE_RUN = STUDY_ROOT / (
    "results/run_20260812_210736_b717f212_amended_preflight"
)
AMENDMENT02_REFERENCE_ZIP = Path(str(AMENDMENT02_REFERENCE_RUN) + "_review_bundle.zip")
AMENDMENT02_REFERENCE_VERIFICATION = AMENDMENT02_REFERENCE_RUN.parent / (
    AMENDMENT02_REFERENCE_RUN.name + "_package_verification.json"
)
NOTEBOOK = PROJECT_ROOT / "COCO_2_features.ipynb"
GATE_SOURCE = PROJECT_ROOT / "Shared/inference_bash/code/sam2_pipeline/gate_core.py"
PIPELINE_SOURCE = PROJECT_ROOT / "Shared/inference_bash/code/sam2_pipeline/pipeline.py"
PLOTS_NOTEBOOK = PROJECT_ROOT / "Shared/plots.ipynb"
FROZEN_PCA = PROJECT_ROOT / "always_same/stage3/train_foldA/embed_pca_32.npz"
FROZEN_PROTO = PROJECT_ROOT / "always_same/stage2/train_foldA/jassid_unified_proto.csv"
RAW_EMBEDDINGS = PROJECT_ROOT / "Shared/PC_codes/npy_npz_csv/tiles_512/stage1_embeddings/train_foldA/embeddings_train_all.npy"
POST_R92_DIR = PROJECT_ROOT / "Shared/maskout_tile/IMG_9431 (2025-12-28_12-26-31)"
POST_R92_RAW = POST_R92_DIR / "embeddings_train_all.npy"
POST_R92_PCA = POST_R92_DIR / "embed_pca_32.npz"
POST_R92_PROTO = POST_R92_DIR / "jassid_unified_proto.csv"
PRE_R92_PCA = HISTORICAL_TRAINING_CSV.parent / "embed_pca_32.npz"
PRE_R92_PROTO = HISTORICAL_TRAINING_CSV.parent / "jassid_unified_proto.csv"
ENCODER_WEIGHTS = Path("/REVIEWER_INPUT_ROOT/resnet50-11ad3fa6.pth")

EXPECTED_HASHES = {
    str(HISTORICAL_TRAINING_CSV): "1836d704d15f27728e3ba09d7cddf0aea9458049bc66700dc98f14e7111ec29d",
    str(CURRENT_TRAINING_CSV): "f2c68d4c61b402f9d81be44c7ba5340960da2130dd35288253e7a184b495b998",
    str(AUDIT_CSV): "c2be6932a56f13117b18e8ed5750dac5f5317fce77ff1ece41ddf948a0ffcee6",
    str(R92_MODEL): "f8e13bcdd7308ed87329e46b0c0fd7229028f6026125660e0186ae111e3d9857",
    str(SELECTED_V3_ZIP): "91cda13167cb904f6cb625fdc4891c30434d82cdf2733ff374f91fa6467068ff",
    str(PRIOR_ZIP): "1d99965412be8c8fe9226746d55dc672fc319caa4d073a2b41fc1d26ef8375bb",
    str(TASK_SPEC): "2a5bdae1a78eca765cbd69aa061f090541ff7c0aaf9eded9059396831d9d0eaf",
    str(AMENDMENT02_TASK_SPEC): "844e61d789d8b54279122796c2e8fd24ea4ba8c130bd050fffb64a419298dfc7",
    str(AMENDMENT02_REFERENCE_ZIP): "fb2f41f2e38efa8a8a58f6107a7df935c254bfb049d8ef7e10ed85640cf3da48",
}

IMMUTABLE_DIRECTORY_TARGETS = (
    PRIOR_RUN.resolve(), AMENDMENT02_REFERENCE_RUN.resolve(),
)
IMMUTABLE_FILE_TARGETS = (
    PRIOR_ZIP.resolve(), AMENDMENT02_REFERENCE_ZIP.resolve(),
    AMENDMENT02_REFERENCE_VERIFICATION.resolve(),
)
BASE_SEED = 42
TILE_SIZE = 512.0
BORDER_PX = 4.0
TINY_MIN_WH = 32.0
TINY_MAX_AREA = 1500.0
GATE_TOLERANCE = 1e-12
GATE_REQUIRED_INPUTS = (
    "mean_L", "std_L", "delta_a", "delta_b", "circularity", "solidity",
    "eccentricity", "components_count", "extent", "grad_p90",
    "touching_border", "embed_sim", *(f"embed_pca_{index}" for index in range(32)),
)

FAMILIES = (
    "color", "shape_morphology", "texture", "spatial_context",
    "deep_pca", "deep_similarity",
)
OFFICIAL_VARIANTS = (
    "full_new_reference", "manual_only", "deep_only", "no_color",
    "no_shape", "no_texture", "no_pca", "no_embed_sim",
    "no_review_aware_training_weights", "no_safe_smote",
)
EXPLORATORY_VARIANT = "no_deep_pca_features"
_AMENDMENT03_MODEL_METADATA = {
    "amendment_estimator_type": "XGBClassifier",
    "amendment_class_labels_json": "[0,1]",
    "amendment_positive_class": "1",
    "amendment_objective": "binary:logistic",
}
_AMENDMENT03_AUTHORIZATION = "AMENDMENT_03_CORRIGENDUM_ONE"
_AMENDMENT04_AUTHORIZATION = "AMENDMENT_04_CSV_ROUNDTRIP_ONE"
_AMENDMENT03_CUDA_PYTHON = Path(
    "/REVIEWER_INPUT_ROOT/python"
)
_AMENDMENT03_CUDA_PREFIX = _AMENDMENT03_CUDA_PYTHON.parent.parent
_AMENDMENT03_RUNTIME_RELATIVE = Path(".runtime/amendment03_corrigendum")
_AMENDMENT03_ALLOWED_CODE_DRIFT_PATHS = frozenset({
    "code/amendment03_compat.py",
    "code/amendment03_recovery.py",
    "code/amendment_core.py",
    "code/run_study.py",
    "tests/test_amendment03.py",
    "tests/test_amendment03_recovery.py",
})
_AMENDMENT03_PRERUN_COPY_MAPPING = {
    "docs/PROTOCOL_AMENDMENT_03.md": "docs/PROTOCOL_AMENDMENT_03.md",
    (
        ".runtime/amendment03_corrigendum/"
        "amendment03_live_cuda_binding.json"
    ): "provenance/amendment03_live_cuda_binding.json",
    (
        ".runtime/amendment03_corrigendum/"
        "xgb211_reload_diagnostic.json"
    ): "provenance/xgb211_reload_diagnostic.json",
    (
        ".runtime/amendment03_corrigendum/"
        "amendment03_targeted_tests.log"
    ): "logs/amendment03_targeted_tests.log",
    (
        ".runtime/amendment03_corrigendum/"
        "amendment03_syntax_compile.log"
    ): "logs/amendment03_syntax_compile.log",
    ".runtime/amendment03_corrigendum/full_tests.log": "logs/full_tests.log",
    (
        ".runtime/amendment03_corrigendum/"
        "cuda_reload_probe.log"
    ): "logs/cuda_reload_probe.log",
}
_NON_SCIENTIFIC_SMOKE_MARKER = (
    "All artifacts in this Run are non-scientific, diagnostic-only CUDA evidence.\n"
)
SMOKE_SUBSET_SELECTION_RULE = (
    "ELIGIBLE_IMMUTABLE_SPLIT_ORDER_LEXICOGRAPHIC_BOTH_CLASS_CARDS_"
    "DETERMINISTIC_CLASS_FALLBACK_NO_PERFORMANCE_INPUT"
)

_AMENDED_PREFLIGHT_REQUIRED_FILES = frozenset({
    "RUN_STATUS.txt",
    "BLOCKERS.md",
    "RECOVERY_REPORT.md",
    "COMMANDS_RUN.txt",
    "targeted_test_results.txt",
    "full_test_results.txt",
    "r92_population_canonicalization_diagnostic.json",
    "docs/PROTOCOL_AMENDMENT_01_TASK_SPEC.md",
    "docs/PROTOCOL_AMENDMENT_01.md",
    "docs/DATA_SOURCE_DECISION.md",
    "docs/FEATURE_SEMANTICS_RECONCILIATION.md",
    "docs/NO_PCA_DIAGNOSTIC.md",
    "docs/CODE_FIXES_AND_TESTS.md",
    "docs/INITIAL_DISCOVERY_AND_PREFLIGHT_PLAN.md",
    "config/run_identity.lock.json",
    "config/feature_manifest.csv",
    "config/feature_dependency_graph.json",
    "config/variant_feature_sets.json",
    "config/training_configuration.lock.json",
    "config/resolved_paths.json",
    "splits/card_split_manifest.csv",
    "splits/row_split_manifest.csv.gz",
    "splits/split_summary.csv",
    "splits/overlap_checks.csv",
    "splits/split_hashes.json",
    "metrics/r92_control.json",
    "provenance/PREFLIGHT_REPORT.json",
    "provenance/historical_snapshot_validation.json",
    "provenance/historical_prediction_replay.csv.gz",
    "provenance/historical_prediction_replay_schema.json",
    "provenance/current_snapshot_validation.json",
    "provenance/embed_sim_semantics.json",
    "provenance/gate_reconstruction_validation.csv",
    "provenance/gate_formula_manifest.json",
    "provenance/audit_gate_upstream_coverage.csv",
    "provenance/harmonized_feature_layer_manifest.json",
    "provenance/raw_embedding_mapping_diagnostic.json",
    "provenance/raw_embedding_join_diagnostic.csv",
    "provenance/review_weight_validation.json",
    "provenance/source_input_hashes_pre.tsv",
    "provenance/source_input_hashes_post.tsv",
    "provenance/hardware_and_software.json",
    "provenance/execution_ledger.json",
    "provenance/initial_readonly_preflight.json",
    "provenance/data_source_gate_evidence.json",
    "provenance/evidence_gate_registry.json",
    "provenance/pca_artifact_inventory.tsv",
    "provenance/raw_embedding_source_inventory.tsv",
    "provenance/prior_immutable_verification_pre.json",
    "provenance/prior_immutable_verification_post.json",
    "provenance/code_diff_vs_previous_blocked_run.patch",
    "provenance/code_snapshot/ablation_core.py",
    "provenance/code_snapshot/amendment_core.py",
    "provenance/code_snapshot/amendment_inventory.py",
    "provenance/code_snapshot/build_amended_preflight.py",
    "provenance/code_snapshot/build_blocked_preflight.py",
    "provenance/code_snapshot/run_ablation_study.sh",
    "provenance/code_snapshot/run_study.py",
    "provenance/code_snapshot/evidence_gates.py",
    "provenance/tests_snapshot/test_ablation.py",
    "provenance/tests_snapshot/test_amendment_inventory.py",
    "provenance/tests_snapshot/test_evidence_gates.py",
    "code/ablation_core.py",
    "code/amendment_core.py",
    "code/amendment_inventory.py",
    "code/build_amended_preflight.py",
    "code/build_blocked_preflight.py",
    "code/evidence_gates.py",
    "code/run_ablation_study.sh",
    "code/run_study.py",
    "tests/test_ablation.py",
    "tests/test_amendment_inventory.py",
    "tests/test_evidence_gates.py",
    "logs/syntax_compile.log",
    "logs/pytest.log",
    "logs/reviewer_snapshot_syntax_compile.log",
    "logs/reviewer_snapshot_pytest.log",
})

_CUDA_SMOKE_REQUIRED_FILES = frozenset({
    "RUN_STATUS.txt",
    "BLOCKERS.md",
    "NON_SCIENTIFIC_DIAGNOSTIC_ONLY.txt",
    "docs/PROTOCOL_AMENDMENT_01_TASK_SPEC.md",
    "docs/PROTOCOL_AMENDMENT_01.md",
    "docs/DATA_SOURCE_DECISION.md",
    "docs/FEATURE_SEMANTICS_RECONCILIATION.md",
    "docs/NO_PCA_DIAGNOSTIC.md",
    "docs/CODE_FIXES_AND_TESTS.md",
    "docs/INITIAL_DISCOVERY_AND_PREFLIGHT_PLAN.md",
    "config/run_identity.lock.json",
    "config/preflight_reference.json",
    "config/smoke_subset.json",
    "config/feature_manifest.csv",
    "config/feature_dependency_graph.json",
    "config/variant_feature_sets.json",
    "config/training_configuration.lock.json",
    "config/resolved_paths.json",
    "splits/card_split_manifest.csv",
    "splits/row_split_manifest.csv.gz",
    "splits/split_summary.csv",
    "splits/overlap_checks.csv",
    "splits/split_hashes.json",
    "tables/SMOKE_REPORT.csv",
    "metrics/SMOKE_REPORT.json",
    "provenance/evidence_gate_registry.json",
    "provenance/execution_ledger.json",
    "provenance/review_weight_subset.json",
    "provenance/preflight_snapshot/PREFLIGHT_REPORT.json",
    "provenance/preflight_snapshot/evidence_gate_registry.json",
    "provenance/preflight_snapshot/source_input_hashes_post.tsv",
    "provenance/preflight_snapshot/cuda_capability_probe.json",
    "provenance/preflight_snapshot/run_identity.lock.json",
    "provenance/preflight_snapshot/OUTPUT_MANIFEST_FINAL.tsv",
    "provenance/preflight_snapshot/BUNDLE_MANIFEST.tsv",
    "provenance/preflight_snapshot/preflight_review_bundle.zip",
    "provenance/code_snapshot/ablation_core.py",
    "provenance/code_snapshot/amendment_core.py",
    "provenance/code_snapshot/amendment_inventory.py",
    "provenance/code_snapshot/build_amended_preflight.py",
    "provenance/code_snapshot/build_blocked_preflight.py",
    "provenance/code_snapshot/evidence_gates.py",
    "provenance/code_snapshot/run_ablation_study.sh",
    "provenance/code_snapshot/run_study.py",
    "provenance/tests_snapshot/test_ablation.py",
    "provenance/tests_snapshot/test_amendment_inventory.py",
    "provenance/tests_snapshot/test_evidence_gates.py",
    "code/ablation_core.py",
    "code/amendment_core.py",
    "code/amendment_inventory.py",
    "code/build_amended_preflight.py",
    "code/build_blocked_preflight.py",
    "code/evidence_gates.py",
    "code/run_ablation_study.sh",
    "code/run_study.py",
    "tests/test_ablation.py",
    "tests/test_amendment_inventory.py",
    "tests/test_evidence_gates.py",
    "logs/preflight_syntax_compile.log",
    "logs/preflight_pytest.log",
    "logs/preflight_reviewer_snapshot_syntax_compile.log",
    "logs/preflight_reviewer_snapshot_pytest.log",
})

_AMENDMENT02_PREFLIGHT_REQUIRED_FILES = frozenset({
    "RUN_STATUS.txt",
    "BLOCKERS.md",
    "docs/PROTOCOL_AMENDMENT_02_TASK_SPEC.md",
    "docs/PROTOCOL_AMENDMENT_02.md",
    "docs/DATA_SOURCE_DECISION.md",
    "docs/FIXED_TABLE_LIMITATIONS.md",
    "config/run_identity.lock.json",
    "config/feature_manifest.csv",
    "config/feature_dependency_graph.json",
    "config/variant_feature_sets.json",
    "config/training_configuration.lock.json",
    "config/locked_split_identity.json",
    "config/resolved_paths.json",
    "splits/card_split_manifest.csv",
    "splits/row_split_manifest.csv.gz",
    "splits/split_summary.csv",
    "splits/overlap_checks.csv",
    "splits/split_hashes.json",
    "metrics/r92_control.json",
    "provenance/PREFLIGHT_REPORT.json",
    "provenance/amendment02_policy_evidence.json",
    "provenance/reference_preflight_validation.json",
    "provenance/historical_prediction_replay.csv.gz",
    "provenance/historical_prediction_replay_schema.json",
    "provenance/pca_artifact_inventory.tsv",
    "provenance/embed_sim_semantics.json",
    "provenance/gate_reconstruction_validation.csv",
    "provenance/gate_formula_manifest.json",
    "provenance/audit_gate_upstream_coverage.csv",
    "provenance/source_input_hashes_pre.tsv",
    "provenance/source_input_hashes_post.tsv",
    "provenance/evidence_gate_registry.json",
    "provenance/execution_ledger.json",
    "provenance/cuda_environment_before.txt",
    "provenance/cuda_environment_after.txt",
    "provenance/cuda_active_fit_probe.json",
    "provenance/code_snapshot/ablation_core.py",
    "provenance/code_snapshot/amendment_core.py",
    "provenance/code_snapshot/amendment_inventory.py",
    "provenance/code_snapshot/amendment02_resampling.py",
    "provenance/code_snapshot/build_amended_preflight.py",
    "provenance/code_snapshot/build_amendment02_preflight.py",
    "provenance/code_snapshot/build_blocked_preflight.py",
    "provenance/code_snapshot/evidence_gates.py",
    "provenance/code_snapshot/run_ablation_study.sh",
    "provenance/code_snapshot/run_study.py",
    "provenance/tests_snapshot/test_ablation.py",
    "provenance/tests_snapshot/test_amendment_inventory.py",
    "provenance/tests_snapshot/test_amendment02_resampling.py",
    "provenance/tests_snapshot/test_evidence_gates.py",
    "code/ablation_core.py",
    "code/amendment_core.py",
    "code/amendment_inventory.py",
    "code/amendment02_resampling.py",
    "code/build_amended_preflight.py",
    "code/build_amendment02_preflight.py",
    "code/build_blocked_preflight.py",
    "code/evidence_gates.py",
    "code/run_ablation_study.sh",
    "code/run_study.py",
    "tests/test_ablation.py",
    "tests/test_amendment_inventory.py",
    "tests/test_amendment02_resampling.py",
    "tests/test_evidence_gates.py",
    "logs/targeted_tests.log",
    "logs/full_tests.log",
    "logs/reviewer_snapshot_syntax_compile.log",
    "logs/reviewer_snapshot_pytest.log",
})

_AMENDMENT02_SMOKE_REQUIRED_FILES = frozenset({
    "RUN_STATUS.txt",
    "BLOCKERS.md",
    "NON_SCIENTIFIC_DIAGNOSTIC_ONLY.txt",
    "docs/PROTOCOL_AMENDMENT_02_TASK_SPEC.md",
    "docs/PROTOCOL_AMENDMENT_02.md",
    "docs/DATA_SOURCE_DECISION.md",
    "docs/FIXED_TABLE_LIMITATIONS.md",
    "config/run_identity.lock.json",
    "config/preflight_reference.json",
    "config/smoke_subset.json",
    "config/feature_manifest.csv",
    "config/feature_dependency_graph.json",
    "config/variant_feature_sets.json",
    "config/training_configuration.lock.json",
    "config/locked_split_identity.json",
    "config/resolved_paths.json",
    "splits/card_split_manifest.csv",
    "splits/row_split_manifest.csv.gz",
    "splits/split_summary.csv",
    "splits/overlap_checks.csv",
    "splits/split_hashes.json",
    "tables/SMOKE_REPORT.csv",
    "metrics/SMOKE_REPORT.json",
    "provenance/frozen_resampling_parity.json",
    "provenance/evidence_gate_registry.json",
    "provenance/execution_ledger.json",
    "provenance/review_weight_subset.json",
    "provenance/cuda_environment_before.txt",
    "provenance/cuda_environment_after.txt",
    "provenance/cuda_active_fit_probe.json",
    "provenance/preflight_snapshot/PREFLIGHT_REPORT.json",
    "provenance/preflight_snapshot/evidence_gate_registry.json",
    "provenance/preflight_snapshot/source_input_hashes_post.tsv",
    "provenance/preflight_snapshot/cuda_active_fit_probe.json",
    "provenance/preflight_snapshot/run_identity.lock.json",
    "provenance/preflight_snapshot/OUTPUT_MANIFEST_FINAL.tsv",
    "provenance/preflight_snapshot/BUNDLE_MANIFEST.tsv",
    "provenance/preflight_snapshot/preflight_review_bundle.zip",
    "provenance/code_snapshot/ablation_core.py",
    "provenance/code_snapshot/amendment_core.py",
    "provenance/code_snapshot/amendment_inventory.py",
    "provenance/code_snapshot/amendment02_resampling.py",
    "provenance/code_snapshot/build_amended_preflight.py",
    "provenance/code_snapshot/build_amendment02_preflight.py",
    "provenance/code_snapshot/build_blocked_preflight.py",
    "provenance/code_snapshot/evidence_gates.py",
    "provenance/code_snapshot/run_ablation_study.sh",
    "provenance/code_snapshot/run_study.py",
    "provenance/tests_snapshot/test_ablation.py",
    "provenance/tests_snapshot/test_amendment_inventory.py",
    "provenance/tests_snapshot/test_amendment02_resampling.py",
    "provenance/tests_snapshot/test_evidence_gates.py",
    "code/ablation_core.py",
    "code/amendment_core.py",
    "code/amendment_inventory.py",
    "code/amendment02_resampling.py",
    "code/build_amended_preflight.py",
    "code/build_amendment02_preflight.py",
    "code/build_blocked_preflight.py",
    "code/evidence_gates.py",
    "code/run_ablation_study.sh",
    "code/run_study.py",
    "tests/test_ablation.py",
    "tests/test_amendment_inventory.py",
    "tests/test_amendment02_resampling.py",
    "tests/test_evidence_gates.py",
    "logs/preflight_targeted_tests.log",
    "logs/preflight_full_tests.log",
    "logs/preflight_reviewer_snapshot_syntax_compile.log",
    "logs/preflight_reviewer_snapshot_pytest.log",
})

_AMENDMENT03_SMOKE_REQUIRED_FILES = (
    (
        _AMENDMENT02_SMOKE_REQUIRED_FILES
        - {"provenance/preflight_snapshot/preflight_review_bundle.zip"}
    )
    | amendment03_recovery.AMENDMENT03_RECOVERY_REQUIRED_FILES
    | {"logs/amendment03_syntax_compile.log"}
)
_AMENDMENT04_SMOKE_REQUIRED_FILES = (
    (
        _AMENDMENT02_SMOKE_REQUIRED_FILES
        - {"provenance/preflight_snapshot/preflight_review_bundle.zip"}
    )
    | amendment04_recovery.AMENDMENT04_RECOVERY_REQUIRED_FILES
)

PACKAGE_REQUIRED_FILES: dict[tuple[str, str], frozenset[str]] = {
    ("AMENDED_PREFLIGHT", "BLOCKED_PREFLIGHT_COMPLETE"): _AMENDED_PREFLIGHT_REQUIRED_FILES,
    ("AMENDED_PREFLIGHT", "PREFLIGHT_COMPLETE"): _AMENDED_PREFLIGHT_REQUIRED_FILES,
    ("CUDA_SMOKE_NON_SCIENTIFIC", "SMOKE_COMPLETE"): (
        _CUDA_SMOKE_REQUIRED_FILES | {"provenance/smoke_model_bundle_verification.json"}
    ),
    ("CUDA_SMOKE_NON_SCIENTIFIC", "SMOKE_INCOMPLETE"): (
        _CUDA_SMOKE_REQUIRED_FILES | {"logs/smoke_failure.txt"}
    ),
}


class AmendmentError(RuntimeError):
    """Controlled Amendment 01 workflow error."""


class ScientificBlocker(legacy.ScientificBlocker):
    """A gate that prevents CUDA Smoke or later scientific training."""


class IntegrityError(AmendmentError):
    """An immutable input or package integrity failure."""


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    wall_seconds: float


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def file_is_byte_prefix(prefix_path: Path, complete_path: Path, chunk_size: int = 8 << 20) -> bool:
    if prefix_path.stat().st_size > complete_path.stat().st_size:
        return False
    remaining = prefix_path.stat().st_size
    with prefix_path.open("rb") as prefix, complete_path.open("rb") as complete:
        while remaining:
            size = min(chunk_size, remaining)
            if prefix.read(size) != complete.read(size):
                return False
            remaining -= size
    return True


def canonical_pca_artifact_inventory() -> pd.DataFrame:
    """Enumerate the complete PCA/prototype lineage evidence scope."""

    root = PROJECT_ROOT / "Shared/maskout_tile"
    records = [
        *inventory_tools.enumerate_transform_artifacts(root),
        *inventory_tools.enumerate_prefix_csv_artifacts(root),
    ]
    return inventory_tools.inventory_frame(records)


def pca_lineage_evidence(inventory: pd.DataFrame) -> dict[str, Any]:
    """Derive PCA/prototype compatibility from the pinned artifact inventory."""

    lineage = inventory.loc[
        inventory.role.astype(str).eq("timestamped PCA/prototype lineage")
    ].copy()
    prefixes = inventory.loc[
        inventory.role.astype(str).eq("historical append-prefix verification")
    ].copy()
    pca_rows = lineage.loc[
        lineage.logical_name.astype(str).str.endswith("_embed_pca_32")
    ]
    pack_rows = lineage.loc[
        lineage.logical_name.astype(str).str.endswith("_foldsafe_pack")
    ]
    prototype_rows = lineage.loc[
        lineage.logical_name.astype(str).str.endswith("_jassid_unified_proto")
    ]
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, path in (
        ("frozen", FROZEN_PCA),
        ("pre_IMG_9429", PRE_R92_PCA),
        ("post_IMG_9429", POST_R92_PCA),
    ):
        payload = np.load(path, allow_pickle=False)
        arrays[name] = {key: payload[key] for key in payload.files}
    comparisons = {
        f"frozen_vs_{name}": {
            "components_max_absolute_difference": float(np.max(np.abs(
                arrays["frozen"]["components"] - arrays[name]["components"]
            ))),
            "mean_max_absolute_difference": float(np.max(np.abs(
                arrays["frozen"]["mean"] - arrays[name]["mean"]
            ))),
        }
        for name in ("pre_IMG_9429", "post_IMG_9429")
    }
    prefix_paths: dict[str, Path] = {}
    for row in prefixes.itertuples(index=False):
        match = re.match(r"prefix_csv_(IMG_\d+)_", str(row.logical_name))
        if match:
            prefix_paths[match.group(1)] = Path(str(row.path))
    previous = prefix_paths.get("IMG_9428")
    following = prefix_paths.get("IMG_9431")
    per_row_hash_columns = [
        column
        for column in pd.read_csv(HISTORICAL_TRAINING_CSV, nrows=0).columns
        if "hash" in column.lower()
        and ("pca" in column.lower() or "proto" in column.lower())
    ]
    frozen_pca_hash = sha256_file(FROZEN_PCA)
    frozen_proto_hash = sha256_file(FROZEN_PROTO)
    pca_hashes = set(pca_rows.sha256.astype(str))
    pack_hashes = set(pack_rows.sha256.astype(str))
    prototype_hashes = set(prototype_rows.sha256.astype(str))
    artifact_set_count = len(pca_rows)
    compatible = bool(
        len(pca_hashes) == 1
        and len(prototype_hashes) == 1
        and pca_hashes == {frozen_pca_hash}
        and prototype_hashes == {frozen_proto_hash}
        and per_row_hash_columns
    )
    inventory_checks = {
        "three_transform_artifacts_per_backup": (
            len(lineage) == artifact_set_count * 3
        ),
        "pca_count_matches_backup_count": len(pca_rows) == artifact_set_count,
        "pack_count_matches_backup_count": len(pack_rows) == artifact_set_count,
        "prototype_count_matches_backup_count": (
            len(prototype_rows) == artifact_set_count
        ),
        "both_prefix_tables_inventoried": len(prefixes) == 2,
        "all_inventory_hashes_present": bool(
            inventory.sha256.astype(str).map(_valid_sha256).all()
        ),
    }
    return {
        "classification": "VERIFIED" if compatible else "UNRESOLVED",
        "status": (
            "VERIFIED_SINGLE_COMPATIBLE_TRANSFORM"
            if compatible
            else "UNRESOLVED_HETEROGENEOUS_APPEND_BUILT_TRANSFORMS"
        ),
        "artifact_backup_scope": (
            "timestamped backup directories through first post-IMG9429 "
            "preserving backup IMG_9431"
        ),
        "artifact_set_count": artifact_set_count,
        "distinct_pca_hashes": len(pca_hashes),
        "distinct_pack_hashes": len(pack_hashes),
        "distinct_prototype_hashes": len(prototype_hashes),
        "frozen_pca_sha256": frozen_pca_hash,
        "pre_IMG_9429_pca_sha256": sha256_file(PRE_R92_PCA),
        "post_IMG_9429_pca_sha256": sha256_file(POST_R92_PCA),
        "frozen_prototype_sha256": frozen_proto_hash,
        "pre_IMG_9429_prototype_sha256": sha256_file(PRE_R92_PROTO),
        "post_IMG_9429_prototype_sha256": sha256_file(POST_R92_PROTO),
        "comparisons": comparisons,
        "IMG9428_snapshot_is_byte_prefix_of_IMG9429": bool(
            previous and file_is_byte_prefix(previous, HISTORICAL_TRAINING_CSV)
        ),
        "IMG9429_snapshot_is_byte_prefix_of_IMG9431": bool(
            following and file_is_byte_prefix(HISTORICAL_TRAINING_CSV, following)
        ),
        "source_behavior": (
            "PCA/prototype refit when new IDs arrive; feature CSV skips "
            "completed triples and appends only unseen rows"
        ),
        "per_row_transform_hash_columns": per_row_hash_columns,
        "per_row_transform_hash_persisted": bool(per_row_hash_columns),
        "compatible_single_basis": compatible,
        "compatibility_conclusion": (
            "A single frozen PCA/prototype basis is proven for every row."
            if compatible
            else "Stored fixed columns replay r92, but no single PCA/prototype "
            "basis is compatible with every training row and the audit."
        ),
        "dedicated_artifact_inventory": {
            "classification": (
                "VERIFIED" if all(inventory_checks.values()) else "BLOCKED"
            ),
            "status": "PASS" if all(inventory_checks.values()) else "FAIL",
            "row_count": len(inventory),
            "lineage_row_count": len(lineage),
            "prefix_row_count": len(prefixes),
            "checks": inventory_checks,
        },
    }


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default).encode("utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value).__name__)


def ensure_write_path(path: Path) -> Path:
    resolved = path.resolve()
    root = STUDY_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise AmendmentError(f"Refusing write outside study root: {resolved}")
    for immutable in IMMUTABLE_DIRECTORY_TARGETS:
        if resolved == immutable or immutable in resolved.parents:
            raise IntegrityError(f"Refusing write to immutable prior artifact: {resolved}")
    for immutable in IMMUTABLE_FILE_TARGETS:
        if resolved == immutable:
            raise IntegrityError(f"Refusing write to immutable prior artifact: {resolved}")
    amendment03_immutable_directories = (
        STUDY_ROOT / "results" / amendment03_recovery.ACCEPTED_PREFLIGHT_RUN_ID,
        STUDY_ROOT / "results" / amendment03_recovery.FAILED_SMOKE_RUN_ID,
        STUDY_ROOT / Path(amendment03_recovery.ORPHAN_MODEL_RELATIVE).parent,
    )
    amendment03_immutable_files = (
        STUDY_ROOT / "results" / (
            amendment03_recovery.ACCEPTED_PREFLIGHT_RUN_ID + "_review_bundle.zip"
        ),
        STUDY_ROOT / "results" / (
            amendment03_recovery.ACCEPTED_PREFLIGHT_RUN_ID
            + "_package_verification.json"
        ),
        STUDY_ROOT / amendment03_recovery.RUNTIME_PROBE_RELATIVE,
    )
    for immutable in amendment03_immutable_directories:
        immutable = immutable.resolve()
        if resolved == immutable or immutable in resolved.parents:
            raise IntegrityError(
                f"Refusing write to Amendment 03 immutable artifact: {resolved}"
            )
    if any(resolved == immutable.resolve() for immutable in amendment03_immutable_files):
        raise IntegrityError(
            f"Refusing write to Amendment 03 immutable artifact: {resolved}"
        )
    amendment04_immutable_directories = (
        STUDY_ROOT / "results" / (
            "run_20260813_125445_373539_bc48422a_amendment03_cuda_smoke"
        ),
        STUDY_ROOT / ".runtime/smoke_models" / (
            "run_20260813_125445_373539_bc48422a_amendment03_cuda_smoke"
        ),
    )
    amendment04_immutable_files = (
        STUDY_ROOT / "results" / (
            "run_20260813_125445_373539_bc48422a_"
            "amendment03_cuda_smoke.zip"
        ),
    )
    for immutable in amendment04_immutable_directories:
        immutable = immutable.resolve()
        if resolved == immutable or immutable in resolved.parents:
            raise IntegrityError(
                f"Refusing write to Amendment 04 immutable evidence: {resolved}"
            )
    if any(resolved == immutable.resolve() for immutable in amendment04_immutable_files):
        raise IntegrityError(
            f"Refusing write to Amendment 04 immutable evidence: {resolved}"
        )
    return resolved


def mkdir(path: Path) -> Path:
    target = ensure_write_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


@dataclass(frozen=True)
class PublicationOwnershipToken:
    file_type: str
    st_dev: int
    st_ino: int
    size_bytes: int
    sha256: str


def _publication_key(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _capture_regular_publication_token(
    path: Path,
    *,
    expected_sha256: str,
) -> PublicationOwnershipToken:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise IntegrityError(f"Publication is not a regular file: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 8 << 20):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    observed_sha256 = digest.hexdigest()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or observed_sha256 != expected_sha256
    ):
        raise IntegrityError(f"Publication bytes changed while captured: {path}")
    return PublicationOwnershipToken(
        file_type="REGULAR_FILE",
        st_dev=int(after.st_dev),
        st_ino=int(after.st_ino),
        size_bytes=int(after.st_size),
        sha256=observed_sha256,
    )


def _publication_matches_token(
    path: Path,
    token: PublicationOwnershipToken,
) -> bool:
    if token.file_type != "REGULAR_FILE":
        return False
    try:
        observed = _capture_regular_publication_token(
            path, expected_sha256=token.sha256,
        )
        leaf = os.lstat(path)
    except (FileNotFoundError, OSError, IntegrityError):
        return False
    return bool(
        observed == token
        and stat.S_ISREG(leaf.st_mode)
        and int(leaf.st_dev) == token.st_dev
        and int(leaf.st_ino) == token.st_ino
        and int(leaf.st_size) == token.size_bytes
    )


def _register_owned_publication(
    path: Path,
    token: PublicationOwnershipToken,
    ownership: MutableMapping[Path, PublicationOwnershipToken],
) -> None:
    if not _publication_matches_token(path, token):
        raise IntegrityError(f"Published path differs from owned inode: {path}")
    ownership[_publication_key(path)] = token


def _rename_no_clobber(source: Path, destination: Path) -> bool:
    """Atomically rename source only when destination is absent."""

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise IntegrityError("Atomic no-clobber rename requires renameat2")
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(source), -100, os.fsencode(destination), 1,
    )
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        return False
    raise OSError(error_number, os.strerror(error_number), str(source))


def _lstat_type(mode: int) -> str:
    predicates = (
        (stat.S_ISREG, "REGULAR_FILE"),
        (stat.S_ISDIR, "DIRECTORY"),
        (stat.S_ISLNK, "SYMLINK"),
        (stat.S_ISFIFO, "FIFO"),
        (stat.S_ISSOCK, "SOCKET"),
        (stat.S_ISCHR, "CHARACTER_DEVICE"),
        (stat.S_ISBLK, "BLOCK_DEVICE"),
    )
    return next((label for predicate, label in predicates if predicate(mode)), "OTHER")


def _path_collision_evidence(path: Path) -> dict[str, Any] | None:
    """Describe an occupied leaf without following or reading its target."""

    try:
        before = os.lstat(path)
    except FileNotFoundError:
        return None
    file_type = _lstat_type(before.st_mode)
    link_target = os.readlink(path) if file_type == "SYMLINK" else None
    try:
        after = os.lstat(path)
    except FileNotFoundError as exc:
        raise IntegrityError(f"Collision path changed while inspected: {path}") from exc
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or stat.S_IFMT(before.st_mode) != stat.S_IFMT(after.st_mode)
    ):
        raise IntegrityError(f"Collision path changed while inspected: {path}")
    return {
        "status": "PREEXISTING_UNOWNED_COLLISION",
        "path": str(path),
        "lstat_type": file_type,
        "st_dev": int(after.st_dev),
        "st_ino": int(after.st_ino),
        "link_target": link_target,
    }


def atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    track_publication: bool = False,
    ownership: MutableMapping[Path, PublicationOwnershipToken] | None = None,
) -> PublicationOwnershipToken | None:
    if track_publication and ownership is None:
        raise ValueError("Tracked publication requires an explicit ownership map")
    target = ensure_write_path(path)
    mkdir(target.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        token = (
            _capture_regular_publication_token(
                Path(temporary), expected_sha256=sha256_bytes(data),
            )
            if track_publication else None
        )
        os.replace(temporary, target)
        if token is not None:
            ownership[_publication_key(target)] = token
            _register_owned_publication(target, token, ownership)
        return token
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, json_bytes(value))


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n"
    ).encode("utf-8")


def publish_bytes_no_clobber(
    path: Path,
    data: bytes,
    *,
    ownership: MutableMapping[Path, PublicationOwnershipToken],
) -> str:
    """Atomically publish verified bytes without replacing an existing path."""

    target = ensure_write_path(path)
    mkdir(target.parent)
    if target.exists() or target.is_symlink():
        raise IntegrityError(f"Refusing to overwrite existing publication: {target}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
    )
    published = False
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(temporary_name)
        expected_sha256 = sha256_bytes(data)
        if sha256_file(temporary) != expected_sha256:
            raise IntegrityError(f"Temporary publication bytes drifted: {target}")
        token = _capture_regular_publication_token(
            temporary, expected_sha256=expected_sha256,
        )
        try:
            os.link(temporary, target)
            published = True
        except FileExistsError as exc:
            raise IntegrityError(
                f"Refusing to overwrite existing publication: {target}"
            ) from exc
        _register_owned_publication(target, token, ownership)
        return expected_sha256
    except Exception:
        if published:
            try:
                _remove_publication_with_token(target, token)
            finally:
                ownership.pop(_publication_key(target), None)
        raise
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)


def publish_json_no_clobber(
    path: Path,
    value: Any,
    *,
    ownership: MutableMapping[Path, PublicationOwnershipToken],
) -> str:
    return publish_bytes_no_clobber(path, json_bytes(value), ownership=ownership)


def move_owned_publication_no_clobber(
    source: Path,
    destination: Path,
    *,
    ownership: MutableMapping[Path, PublicationOwnershipToken],
) -> None:
    """Move an owned regular file without overwriting a concurrent destination."""

    source = ensure_write_path(source)
    destination = ensure_write_path(destination)
    if not source.is_file() or source.is_symlink():
        raise IntegrityError(f"Owned publication source is absent/invalid: {source}")
    if destination.exists() or destination.is_symlink():
        raise IntegrityError(
            f"Refusing to overwrite existing staging publication: {destination}"
        )
    source_key = _publication_key(source)
    source_token = ownership.get(source_key)
    if source_token is None or not _publication_matches_token(source, source_token):
        raise IntegrityError(f"Publication source is not owned by this invocation: {source}")
    linked = False
    try:
        os.link(source, destination)
        linked = True
        _register_owned_publication(destination, source_token, ownership)
        if not remove_owned_publication(source, ownership=ownership):
            raise IntegrityError("Owned publication source changed before staging move")
    except Exception:
        if linked:
            destination_key = _publication_key(destination)
            destination_token = ownership.get(destination_key, source_token)
            _remove_publication_with_token(destination, destination_token)
            ownership.pop(destination_key, None)
        raise


def atomic_write_frame(
    frame: pd.DataFrame,
    path: Path,
    *,
    sep: str = ",",
    float_format: str | None = None,
) -> None:
    target = ensure_write_path(path)
    mkdir(target.parent)
    suffix = ".csv.gz" if target.name.endswith(".gz") else ".csv"
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=suffix, dir=target.parent)
    os.close(fd)
    try:
        compression = {"method": "gzip", "compresslevel": 9, "mtime": 0} if target.name.endswith(".gz") else None
        frame.to_csv(
            temporary,
            sep=sep,
            index=False,
            lineterminator="\n",
            compression=compression,
            float_format=float_format,
        )
        os.replace(temporary, target)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def run_command(command: Sequence[str], *, cwd: Path = STUDY_ROOT, env: Mapping[str, str] | None = None) -> CommandResult:
    started = time.perf_counter()
    proc = subprocess.run(
        list(command), cwd=cwd, env=dict(env) if env is not None else None,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return CommandResult(
        command=list(command), cwd=str(cwd), returncode=int(proc.returncode),
        stdout=proc.stdout, stderr=proc.stderr,
        wall_seconds=float(time.perf_counter() - started),
    )


def assert_scientific_training_unblocked(blockers: Mapping[str, Any] | Sequence[Any] | None) -> None:
    if not blockers:
        return
    if isinstance(blockers, Mapping):
        detail = "; ".join(f"{key}: {value}" for key, value in blockers.items())
    else:
        detail = "; ".join(map(str, blockers))
    raise ScientificBlocker(detail)


def normalize_card_id(value: Any) -> str:
    return legacy.normalize_card_id(value)


def normalize_scale(value: Any) -> str:
    return legacy.normalize_scale(value)


def stable_training_ids(frame: pd.DataFrame) -> pd.Series:
    required = ("file_name", "ann_id", "scale")
    missing = [name for name in required if name not in frame]
    if missing:
        raise ScientificBlocker(f"Training key columns missing: {missing}")
    names = frame["file_name"].astype(str).map(lambda value: Path(value).name)
    anns = frame["ann_id"].astype(str)
    scales = frame["scale"].map(normalize_scale)
    keys = pd.DataFrame({"file_name": names, "ann_id": anns, "scale": scales})
    if keys.isna().any().any() or keys.duplicated().any():
        raise ScientificBlocker("Training [file_name,ann_id,scale] key is null or duplicated")
    values = [
        "train_" + sha256_bytes(canonical_json(list(row)))
        for row in keys.itertuples(index=False, name=None)
    ]
    return pd.Series(values, index=frame.index, dtype="string")


def _finish_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(np.int8)
    frame["stable_candidate_id"] = stable_training_ids(frame)
    frame["card_id"] = frame["file_name"].map(normalize_card_id)
    return frame


def load_training(path: Path) -> pd.DataFrame:
    return _finish_training_frame(
        pd.read_csv(path, low_memory=False, keep_default_na=False)
    )


def _sha256_open_binary_stream(stream: Any) -> str:
    position = stream.tell()
    stream.seek(0)
    digest = hashlib.sha256()
    while chunk := stream.read(8 << 20):
        digest.update(chunk)
    stream.seek(position)
    return digest.hexdigest()


def read_csv_verified(
    path: Path,
    expected_sha256: str,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Parse one open descriptor bracketed by the reviewed content hash."""

    if not _valid_sha256(expected_sha256) or not path.is_file() or path.is_symlink():
        raise IntegrityError("Reviewed CSV path/hash is invalid")
    with path.open("rb") as stream:
        before = _sha256_open_binary_stream(stream)
        if before != expected_sha256:
            raise IntegrityError("CSV differs from the reviewed source hash")
        stream.seek(0)
        frame = pd.read_csv(stream, **read_csv_kwargs)
        after = _sha256_open_binary_stream(stream)
    if after != before:
        raise IntegrityError("CSV changed while its reviewed bytes were parsed")
    return frame


def load_training_verified(path: Path, expected_sha256: str) -> pd.DataFrame:
    """Parse the reviewed training bytes and derive stable row identities."""

    return _finish_training_frame(read_csv_verified(
        path,
        expected_sha256,
        low_memory=False,
        keep_default_na=False,
    ))


def load_historical_predictions_verified(
    path: Path,
    expected_sha256: str,
) -> pd.DataFrame:
    """Parse reviewed prediction bytes without losing probability literals."""

    frame = read_csv_verified(
        path,
        expected_sha256,
        low_memory=False,
        keep_default_na=False,
        dtype={"proba": "string"},
    )
    required = {"file_name", "ann_id", "scale", "label", "proba"}
    missing = sorted(required - set(frame.columns))
    if missing or frame.proba.isna().any() or frame.proba.astype(str).eq("").any():
        raise IntegrityError(
            f"Historical prediction source schema/literals are invalid: {missing}"
        )
    return frame


def canonicalize_audit_labels(frame: pd.DataFrame) -> pd.Series:
    """Return the one canonical, strict representation of audit labels."""

    if "human_label" not in frame:
        raise IntegrityError("Audit table is missing required human_label column")
    labels = pd.to_numeric(frame["human_label"], errors="coerce")
    invalid = labels.isna() | ~labels.isin([0, 1])
    if invalid.any():
        raise IntegrityError(
            "Audit human_label values must be non-null binary values {0, 1}; "
            f"invalid_rows={int(invalid.sum())}"
        )
    return labels.astype(np.int8)


def assert_training_matches_reviewed_split(
    training: pd.DataFrame,
    split: pd.DataFrame,
) -> None:
    if len(training) != len(split):
        raise IntegrityError("Training row count differs from accepted split manifest")
    if not np.array_equal(
        training.stable_candidate_id.astype(str),
        split.stable_candidate_id.astype(str),
    ):
        raise IntegrityError("Training rows differ from accepted split manifest")
    observed_labels = pd.to_numeric(training.label, errors="raise").to_numpy(np.int8)
    reviewed_labels = pd.to_numeric(split.label, errors="raise").to_numpy(np.int8)
    if not np.array_equal(observed_labels, reviewed_labels):
        raise IntegrityError("Training labels differ from accepted split manifest")


def historical_group_split(groups: Sequence[str], *, test_frac: float = 0.2, seed: int = 42,
                           tolerance: float = 0.01, max_tries: int = 3000) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    groups_array = np.asarray(groups, dtype=str)
    unique, inverse = np.unique(groups_array, return_inverse=True)
    counts = np.bincount(inverse).astype(int)
    target = int(round(int(counts.sum()) * float(test_frac)))
    rng = np.random.RandomState(seed)
    best_selection: set[str] | None = None
    best_sum = 0
    best_difference = float("inf")

    def greedy(order: np.ndarray) -> tuple[set[str], int]:
        total = 0
        selection: set[str] = set()
        for index in order:
            count = int(counts[index])
            if abs(total + count - target) <= abs(total - target):
                selection.add(str(unique[index]))
                total += count
        return selection, total

    fixed_orders = [np.argsort(-counts), np.argsort(counts)]
    for attempt in range(max_tries):
        order = fixed_orders[attempt] if attempt < 2 else rng.permutation(len(unique))
        selection, selected_count = greedy(order)
        difference = abs(selected_count - target)
        if difference < best_difference:
            best_selection, best_sum, best_difference = selection, selected_count, difference
            if difference <= max(1, int(target * tolerance)):
                break
    if best_selection is None:
        raise ScientificBlocker("Historical group split search produced no selection")
    heldout_groups = best_selection
    train_groups = set(map(str, unique)) - heldout_groups
    train_indices = np.where(np.isin(groups_array, list(train_groups)))[0]
    heldout_indices = np.where(np.isin(groups_array, list(heldout_groups)))[0]
    shuffle = np.random.RandomState(seed)
    train_indices = shuffle.permutation(train_indices)
    heldout_indices = shuffle.permutation(heldout_indices)
    return train_indices.astype(np.int64), heldout_indices.astype(np.int64), {
        "algorithm": "pick_groups_tile_balanced_then_sequential_permutation",
        "seed": seed, "target_rows": target, "selected_rows": int(best_sum),
        "train_cards": sorted(train_groups), "heldout_cards": sorted(heldout_groups),
    }


def _strict_numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    """Load a required numeric field without inventing missing values."""
    if name not in frame:
        raise ScientificBlocker(f"Required numeric input is absent: {name}")
    values = pd.to_numeric(frame[name], errors="coerce").astype(float)
    invalid = ~np.isfinite(values.to_numpy())
    if invalid.any():
        raise ScientificBlocker(
            f"Required numeric input {name} has {int(invalid.sum())} missing/nonfinite rows"
        )
    return values


def historical_border_tiny_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the r92 fallback exactly; never use for amended training."""
    x, y, width, height = (_strict_numeric(frame, key) for key in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
    extents = pd.DataFrame({
        "file_name": frame["file_name"].astype(str), "right": x + width, "bottom": y + height,
    }).groupby("file_name", sort=False).agg({"right": "max", "bottom": "max"})
    tile_width = frame["file_name"].astype(str).map(extents["right"]).fillna(TILE_SIZE).clip(lower=1)
    tile_height = frame["file_name"].astype(str).map(extents["bottom"]).fillna(TILE_SIZE).clip(lower=1)
    border_fraction = BORDER_PX / TILE_SIZE
    tiny_fraction = TINY_MIN_WH / TILE_SIZE
    tiny_area_fraction = TINY_MAX_AREA / (TILE_SIZE * TILE_SIZE)
    border = (
        (x <= border_fraction * tile_width) | (y <= border_fraction * tile_height)
        | (x + width >= (1 - border_fraction) * tile_width)
        | (y + height >= (1 - border_fraction) * tile_height)
    )
    tiny = (
        (width < tiny_fraction * tile_width) | (height < tiny_fraction * tile_height)
        | (width * height < tiny_area_fraction * tile_width * tile_height)
    )
    return border.to_numpy(bool), tiny.to_numpy(bool)


def corrected_border_tiny_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Amended source-defined fixed 512-coordinate boundary rule."""
    x, y, width, height = (_strict_numeric(frame, key) for key in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
    touching = _strict_numeric(frame, "touching_border").to_numpy() == 1
    border = (
        touching | (x.to_numpy() <= BORDER_PX) | (y.to_numpy() <= BORDER_PX)
        | ((x + width).to_numpy() >= TILE_SIZE - BORDER_PX)
        | ((y + height).to_numpy() >= TILE_SIZE - BORDER_PX)
    )
    tiny = (
        (width.to_numpy() < TINY_MIN_WH) | (height.to_numpy() < TINY_MIN_WH)
        | ((width * height).to_numpy() < TINY_MAX_AREA)
    )
    return np.asarray(border, bool), np.asarray(tiny, bool)


def get_r92_feature_order() -> list[str]:
    payload = joblib.load(R92_MODEL)
    features = list(payload["features"])
    if len(features) != 93 or len(set(features)) != 93:
        raise ScientificBlocker(f"Expected 93 unique r92 predictors, got {len(features)}")
    return features


def historical_metric_sidecar_evidence() -> dict[str, Any]:
    metrics_path = R92_DIR / "metrics.json"
    threshold_path = R92_DIR / "threshold_r92.json"
    metrics = json.loads(metrics_path.read_text())
    threshold = json.loads(threshold_path.read_text())
    checkpoint = joblib.load(R92_MODEL)
    used_trees = int(
        checkpoint["pipeline"].named_steps["clf"].get_booster().num_boosted_rounds()
    )
    return {
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "threshold_path": str(threshold_path),
        "threshold_sha256": sha256_file(threshold_path),
        "checkpoint_path": str(R92_MODEL),
        "checkpoint_sha256": sha256_file(R92_MODEL),
        "average_precision": float(metrics["tile"]["pr_auc"]),
        "roc_auc": float(metrics["tile"]["roc_auc"]),
        "confusion_matrix": metrics["tile"]["confusion_matrix"],
        "threshold": float(threshold["threshold"]),
        "sidecar_used_trees": int(metrics["meta"]["used_trees"]),
        "checkpoint_used_trees": used_trees,
    }


def feature_primary_groups(features: Sequence[str] | None = None) -> dict[str, str]:
    mapping = legacy.feature_group_mapping(list(features or get_r92_feature_order()))
    for feature in FEATURE_DEPENDENCIES:
        if feature in mapping:
            mapping[feature] = "cross_family_composite"
    return mapping


FEATURE_DEPENDENCIES: dict[str, set[str]] = {
    "g_quality": set(FAMILIES),
    "g_robust": {"shape_morphology", "texture", "spatial_context"},
}


def dependency_closure(features: Sequence[str], mapping: Mapping[str, str], removed_families: Iterable[str]) -> dict[str, str]:
    removed = set(removed_families)
    reasons: dict[str, str] = {}
    for feature in features:
        if mapping[feature] in removed:
            reasons[feature] = f"primary_group_removed:{mapping[feature]}"
        dependencies = FEATURE_DEPENDENCIES.get(feature, set())
        hit = sorted(dependencies & removed)
        if hit:
            reasons[feature] = "dependency_closure:" + "|".join(hit)
    return reasons


def _valid_sha256(value: Any) -> bool:
    return bool(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value))


def validate_no_pca_raw_layer_contract(
    contract: Mapping[str, Any] | None,
    raw_embedding_features: Sequence[str],
) -> dict[str, Any]:
    """Validate every file needed to load exact Training/Validation/Audit raw layers."""

    raw_features = list(map(str, raw_embedding_features))
    expected_feature_hash = sha256_bytes(("\n".join(raw_features) + "\n").encode())
    top_checks = {
        "contract_mapping": isinstance(contract, Mapping),
        "contract_version": isinstance(contract, Mapping) and contract.get("version") == 1,
        "contract_status": isinstance(contract, Mapping)
        and contract.get("status") == "VERIFIED_EXACT_RAW_LAYER_CONTRACT",
        "raw_feature_count": len(raw_features) == 2048,
        "raw_feature_unique": len(set(raw_features)) == 2048,
        "raw_feature_list": isinstance(contract, Mapping)
        and contract.get("ordered_raw_feature_columns") == raw_features,
        "raw_feature_hash": isinstance(contract, Mapping)
        and contract.get("ordered_raw_feature_columns_sha256") == expected_feature_hash,
    }
    populations = contract.get("populations", {}) if isinstance(contract, Mapping) else {}
    required_populations = {"training", "validation", "audit"}
    top_checks["all_required_populations"] = (
        isinstance(populations, Mapping) and set(populations) == required_populations
    )
    population_results: dict[str, Any] = {}
    file_hash_cache: dict[str, str | None] = {}
    candidate_source_signatures: dict[str, tuple[str, str | None]] = {}

    def observed_file_hash(path_value: Any) -> str | None:
        path = Path(str(path_value))
        key = str(path.resolve(strict=False))
        if key not in file_hash_cache:
            file_hash_cache[key] = sha256_file(path) if path.is_file() else None
        return file_hash_cache[key]

    semantics_path = Path(str(
        contract.get("embedding_semantics_manifest_path", "")
        if isinstance(contract, Mapping) else ""
    ))
    expected_semantics_hash = (
        contract.get("embedding_semantics_manifest_sha256")
        if isinstance(contract, Mapping) else None
    )
    top_checks["semantics_manifest_hash"] = bool(
        _valid_sha256(expected_semantics_hash)
        and observed_file_hash(semantics_path) == expected_semantics_hash
    )
    semantics_manifest: dict[str, Any] = {}
    if top_checks["semantics_manifest_hash"]:
        try:
            loaded = json.loads(semantics_path.read_text())
            semantics_manifest = loaded if isinstance(loaded, dict) else {}
        except Exception:
            semantics_manifest = {}
    top_checks["semantics_manifest_schema"] = bool(
        semantics_manifest.get("version") == 1
        and semantics_manifest.get("status") == "VERIFIED_EXACT_EMBEDDING_SEMANTICS"
        and set(semantics_manifest.get("population_cache_links", {}))
        == required_populations
    )
    encoder = semantics_manifest.get("encoder", {})
    preprocessing = semantics_manifest.get("preprocessing", {})
    crop_mask = semantics_manifest.get("crop_mask", {})
    top_checks["semantics_encoder_artifact"] = bool(
        isinstance(encoder.get("backbone"), str)
        and bool(encoder.get("backbone"))
        and _valid_sha256(encoder.get("weights_sha256"))
        and observed_file_hash(encoder.get("weights_path", ""))
        == encoder.get("weights_sha256")
    )
    top_checks["semantics_preprocessing_artifact"] = bool(
        isinstance(preprocessing.get("ordered_steps"), list)
        and bool(preprocessing.get("ordered_steps"))
        and all(isinstance(step, str) and step for step in preprocessing.get("ordered_steps", []))
        and _valid_sha256(preprocessing.get("source_sha256"))
        and observed_file_hash(preprocessing.get("source_path", ""))
        == preprocessing.get("source_sha256")
    )
    top_checks["semantics_crop_mask_artifact"] = bool(
        crop_mask.get("status") == "VERIFIED_IDENTICAL_ACROSS_POPULATIONS"
        and _valid_sha256(crop_mask.get("source_sha256"))
        and observed_file_hash(crop_mask.get("source_path", ""))
        == crop_mask.get("source_sha256")
    )

    for population in sorted(required_populations):
        reference = populations.get(population, {}) if isinstance(populations, Mapping) else {}
        checks: dict[str, bool] = {
            "reference_mapping": isinstance(reference, Mapping),
            "population_role": isinstance(reference, Mapping)
            and reference.get("population") == population,
            "reference_status": isinstance(reference, Mapping)
            and reference.get("status") == "VERIFIED_ALIGNED_RAW_LAYER_REFERENCE",
            "row_count_positive": isinstance(reference, Mapping)
            and isinstance(reference.get("row_count"), int)
            and reference.get("row_count", 0) > 0,
            "dimension_count": isinstance(reference, Mapping)
            and reference.get("dimension_count") == 2048,
            "dtype": isinstance(reference, Mapping) and reference.get("dtype") == "float32",
            "ordered_raw_features": isinstance(reference, Mapping)
            and reference.get("ordered_raw_feature_columns") == raw_features
            and reference.get("ordered_raw_feature_columns_sha256") == expected_feature_hash,
        }
        checks["embedding_semantics_manifest"] = bool(
            reference.get("embedding_semantics_manifest_path") == str(semantics_path.resolve(strict=False))
            and reference.get("embedding_semantics_manifest_sha256") == expected_semantics_hash
            and top_checks["semantics_manifest_schema"]
            and top_checks["semantics_encoder_artifact"]
            and top_checks["semantics_preprocessing_artifact"]
            and top_checks["semantics_crop_mask_artifact"]
        )
        raw_path = Path(str(reference.get("raw_cache_path", "")))
        alignment_path = Path(str(reference.get("alignment_index_path", "")))
        table_path = Path(str(reference.get("candidate_table_path", "")))
        checks["raw_cache_hash"] = bool(
            _valid_sha256(reference.get("raw_cache_sha256"))
            and observed_file_hash(raw_path) == reference.get("raw_cache_sha256")
        )
        checks["alignment_index_hash"] = bool(
            _valid_sha256(reference.get("alignment_index_sha256"))
            and observed_file_hash(alignment_path) == reference.get("alignment_index_sha256")
        )
        checks["candidate_table_hash"] = bool(
            _valid_sha256(reference.get("candidate_table_sha256"))
            and observed_file_hash(table_path) == reference.get("candidate_table_sha256")
        )
        alignment: np.ndarray | None = None
        cache: np.ndarray | None = None
        if checks["raw_cache_hash"]:
            try:
                cache = np.load(raw_path, mmap_mode="r", allow_pickle=False)
            except Exception:
                cache = None
        checks["raw_cache_shape_and_dtype"] = bool(
            cache is not None
            and cache.ndim == 2
            and cache.shape[1] == 2048
            and cache.dtype == np.dtype("float32")
        )
        if checks["alignment_index_hash"]:
            try:
                with np.load(alignment_path, allow_pickle=False) as payload:
                    alignment = np.asarray(payload["cache_row_index"])
            except Exception:
                alignment = None
        row_count = reference.get("row_count", -1) if isinstance(reference, Mapping) else -1
        checks["raw_cache_row_count_exact"] = bool(
            cache is not None and len(cache) == row_count
        )
        checks["alignment_index_shape_and_dtype"] = bool(
            alignment is not None
            and alignment.ndim == 1
            and np.issubdtype(alignment.dtype, np.integer)
            and len(alignment) == row_count
        )
        checks["alignment_mapping_hash"] = bool(
            alignment is not None
            and _valid_sha256(reference.get("alignment_mapping_sha256_int64_le"))
            and sha256_array(alignment.astype("<i8", copy=False))
            == reference.get("alignment_mapping_sha256_int64_le")
        )
        checks["alignment_indices_in_range_and_unique"] = bool(
            alignment is not None
            and cache is not None
            and ((alignment >= 0) & (alignment < len(cache))).all()
            and len(np.unique(alignment)) == len(alignment)
        )
        aligned_values_finite = False
        if checks["alignment_indices_in_range_and_unique"]:
            aligned_values_finite = True
            for start in range(0, len(alignment), 4096):
                selected = np.asarray(cache[alignment[start:start + 4096]])
                if not np.isfinite(selected).all():
                    aligned_values_finite = False
                    break
        checks["all_aligned_raw_values_finite"] = aligned_values_finite
        key_columns = reference.get("candidate_key_columns", []) if isinstance(reference, Mapping) else []
        normalized_key_columns = [str(value).lower() for value in key_columns] if isinstance(key_columns, list) else []
        checks["population_key_schema"] = bool(
            normalized_key_columns == (
                ["img_folder", "image", "id"]
                if population == "audit" else ["file_name", "ann_id", "scale"]
            )
        )
        checks["audit_source_is_canonical"] = bool(
            population != "audit"
            or table_path.resolve(strict=False) == AUDIT_CSV.resolve(strict=False)
        )
        candidate_key_hash = None
        table_rows = None
        if checks["candidate_table_hash"] and isinstance(key_columns, list) and key_columns:
            try:
                table_frame = pd.read_csv(
                    table_path,
                    usecols=key_columns,
                    keep_default_na=False,
                    low_memory=False,
                )
                table_rows = len(table_frame)
                candidate_key_hash = inventory_tools.canonical_key_digest(table_frame, key_columns)
            except Exception:
                candidate_key_hash = None
        checks["candidate_table_rows_and_key_hash"] = bool(
            table_rows == row_count
            and _valid_sha256(reference.get("candidate_key_sha256"))
            and candidate_key_hash == reference.get("candidate_key_sha256")
        )
        checks["candidate_row_order_key_hash"] = bool(
            _valid_sha256(reference.get("candidate_row_order_key_sha256"))
            and candidate_key_hash == reference.get("candidate_row_order_key_sha256")
        )
        sidecar_path = Path(str(reference.get("sidecar_path", "")))
        checks["sidecar_hash"] = bool(
            _valid_sha256(reference.get("sidecar_sha256"))
            and observed_file_hash(sidecar_path) == reference.get("sidecar_sha256")
        )
        recomputed_alignment: np.ndarray | None = None
        sidecar_join_counts: dict[str, int | float] = {}
        if checks["sidecar_hash"] and candidate_key_hash is not None:
            sidecar_frame, sidecar_error = inventory_tools._read_sidecar_frame(
                sidecar_path, 64 << 20,
            )
            try:
                sidecar_keys = inventory_tools.canonical_key_frame(sidecar_frame, key_columns)
                table_keys = inventory_tools.canonical_key_frame(table_frame, key_columns)
                _, sidecar_offsets, offset_checks = inventory_tools._explicit_row_index(sidecar_frame)
                valid_sidecar = bool(
                    sidecar_error is None
                    and sidecar_offsets is not None
                    and all(offset_checks.values())
                    and not sidecar_keys.eq("").any(axis=1).any()
                    and not sidecar_keys.duplicated(keep=False).any()
                )
                if valid_sidecar:
                    sidecar_index = pd.MultiIndex.from_frame(sidecar_keys)
                    table_index = pd.MultiIndex.from_frame(table_keys)
                    mapping = pd.Series(sidecar_offsets, index=sidecar_index).reindex(table_index)
                    if mapping.notna().all():
                        recomputed_alignment = mapping.to_numpy(np.int64)
                    sidecar_join_counts = {
                        "mapped": int(mapping.notna().sum()),
                        "unmatched": int(mapping.isna().sum()),
                        "orphan": int((~sidecar_index.isin(table_index)).sum()),
                        "collisions": int(mapping.dropna().duplicated(keep=False).sum()),
                        "rate": float(mapping.notna().mean()) if len(mapping) else 1.0,
                    }
            except Exception:
                recomputed_alignment = None
        checks["sidecar_actual_join_matches_alignment"] = bool(
            recomputed_alignment is not None
            and alignment is not None
            and sidecar_frame is not None
            and len(sidecar_frame) == row_count
            and np.array_equal(recomputed_alignment, alignment.astype(np.int64, copy=False))
            and sidecar_join_counts == {
                "mapped": row_count,
                "unmatched": 0,
                "orphan": 0,
                "collisions": 0,
                "rate": 1.0,
            }
            and reference.get("measured_exact_join_rate") == 1.0
            and reference.get("measured_mapped_candidate_rows") == row_count
            and reference.get("measured_unmatched_candidate_rows") == 0
            and reference.get("measured_orphan_sidecar_rows") == 0
            and reference.get("measured_cache_collision_rows") == 0
        )
        semantics_link = semantics_manifest.get("population_cache_links", {}).get(population, {})
        checks["semantics_population_cache_link"] = bool(
            semantics_link.get("raw_cache_path") == str(raw_path.resolve(strict=False))
            and semantics_link.get("raw_cache_sha256") == reference.get("raw_cache_sha256")
            and semantics_link.get("sidecar_path") == str(sidecar_path.resolve(strict=False))
            and semantics_link.get("sidecar_sha256") == reference.get("sidecar_sha256")
            and semantics_link.get("alignment_mapping_sha256_int64_le")
            == reference.get("alignment_mapping_sha256_int64_le")
        )
        candidate_source_signatures[population] = (
            str(table_path.resolve(strict=False)),
            reference.get("candidate_table_sha256") if isinstance(reference, Mapping) else None,
        )
        population_results[population] = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        }
    top_checks["same_encoder_preprocessing_semantics"] = bool(
        top_checks["semantics_manifest_hash"]
        and all(
            result["checks"].get("embedding_semantics_manifest", False)
            and result["checks"].get("semantics_population_cache_link", False)
            for result in population_results.values()
        )
    )
    top_checks["training_validation_same_candidate_universe"] = bool(
        candidate_source_signatures.get("training")
        == candidate_source_signatures.get("validation")
    )
    passed = all(top_checks.values()) and all(
        result["status"] == "PASS" for result in population_results.values()
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "classification": "VERIFIED" if passed else "BLOCKED",
        "checks": top_checks,
        "failed_checks": sorted(name for name, value in top_checks.items() if not value),
        "populations": population_results,
    }


def load_no_pca_smoke_raw_layers(
    contract: Mapping[str, Any] | None,
    training: pd.DataFrame,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    raw_embedding_features: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load only requested Smoke rows after independently validating the layer contract."""

    validation = validate_no_pca_raw_layer_contract(contract, raw_embedding_features)
    if validation["status"] != "PASS" or not isinstance(contract, Mapping):
        raise ScientificBlocker(
            "no_pca is marked runnable without a valid materialized raw-layer contract: "
            f"{validation['failed_checks']}"
        )
    populations = contract["populations"]

    def load(population: str, requested: Sequence[int]) -> tuple[np.ndarray, dict[str, Any]]:
        reference = populations[population]
        key_columns = list(reference["candidate_key_columns"])
        if len(training) != int(reference["row_count"]):
            raise ScientificBlocker(
                f"no_pca {population} layer row count does not match the accepted training table"
            )
        observed_key_hash = inventory_tools.canonical_key_digest(training, key_columns)
        if observed_key_hash != reference["candidate_key_sha256"]:
            raise ScientificBlocker(
                f"no_pca {population} layer keys/order differ from the accepted training table"
            )
        selected = np.asarray(requested, dtype=np.int64)
        if selected.ndim != 1 or ((selected < 0) | (selected >= len(training))).any():
            raise ScientificBlocker(f"no_pca {population} requested rows are out of bounds")
        with np.load(Path(reference["alignment_index_path"]), allow_pickle=False) as payload:
            alignment = np.asarray(payload["cache_row_index"], dtype=np.int64)
        cache = np.load(Path(reference["raw_cache_path"]), mmap_mode="r", allow_pickle=False)
        values = np.asarray(cache[alignment[selected]], dtype=np.float32)
        del cache
        if values.shape != (len(selected), 2048):
            raise ScientificBlocker(
                f"no_pca {population} materialized layer has unexpected shape {values.shape}"
            )
        return values, {
            "population": population,
            "selected_rows": len(selected),
            "raw_cache_sha256": reference["raw_cache_sha256"],
            "alignment_index_sha256": reference["alignment_index_sha256"],
            "candidate_key_sha256": observed_key_hash,
            "embedding_semantics_manifest_sha256": reference[
                "embedding_semantics_manifest_sha256"
            ],
            "selected_raw_values_sha256_float32_le": sha256_array(
                values.astype("<f4", copy=False)
            ),
        }

    training_raw, training_evidence = load("training", train_indices)
    validation_raw, validation_evidence = load("validation", validation_indices)
    post_load_validation = validate_no_pca_raw_layer_contract(
        contract, raw_embedding_features,
    )
    if post_load_validation != validation or post_load_validation["status"] != "PASS":
        raise ScientificBlocker(
            "no_pca raw-layer contract changed while selected rows were materialized"
        )
    return training_raw, validation_raw, {
        "contract_validation": validation,
        "post_load_contract_validation": post_load_validation,
        "training": training_evidence,
        "validation": validation_evidence,
        "audit_reference_validated": validation["populations"]["audit"]["status"] == "PASS",
    }


def build_variant_feature_sets(
    features: Sequence[str] | None = None,
    *,
    raw_embedding_features: Sequence[str] | None = None,
    raw_mapping_verified: bool = False,
    raw_layer_contract: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    ordered = list(features or get_r92_feature_order())
    mapping = feature_primary_groups(ordered)
    definitions: dict[str, dict[str, Any]] = {
        "full_new_reference": {"remove_families": set(), "kind": "OFFICIAL"},
        "manual_only": {"remove_families": {"deep_pca", "deep_similarity"}, "kind": "OFFICIAL"},
        "deep_only": {"remove_families": {"color", "shape_morphology", "texture", "spatial_context"}, "kind": "OFFICIAL"},
        "no_color": {"remove_families": {"color"}, "kind": "OFFICIAL"},
        "no_shape": {"remove_families": {"shape_morphology"}, "kind": "OFFICIAL"},
        "no_texture": {"remove_families": {"texture"}, "kind": "OFFICIAL"},
        "no_embed_sim": {"remove_families": {"deep_similarity"}, "kind": "OFFICIAL"},
        "no_review_aware_training_weights": {"remove_families": set(), "kind": "OFFICIAL"},
        "no_safe_smote": {"remove_families": set(), "kind": "OFFICIAL"},
        "no_deep_pca_features": {"remove_families": {"deep_pca"}, "kind": "EXPLORATORY_PCA_DERIVED_FEATURE_BLOCK_REMOVAL"},
    }
    output: dict[str, dict[str, Any]] = {}
    for variant, definition in definitions.items():
        reasons = dependency_closure(ordered, mapping, definition["remove_families"])
        retained = [feature for feature in ordered if feature not in reasons]
        removed = [feature for feature in ordered if feature in reasons]
        violations = [
            feature for feature in retained
            if FEATURE_DEPENDENCIES.get(feature, set()) & set(definition["remove_families"])
        ]
        output[variant] = {
            "kind": definition["kind"], "status": "RUNNABLE",
            "removed_families": sorted(definition["remove_families"]),
            "retained_features": retained, "removed_features": removed,
            "removal_reasons": {feature: reasons[feature] for feature in removed},
            "feature_count": len(retained),
            "feature_list_sha256": sha256_bytes(("\n".join(retained) + "\n").encode()),
            "dependency_closure_linter": "PASS" if not violations else "FAIL",
            "dependency_violations": violations,
        }
    base_non_pca = output["no_deep_pca_features"]["retained_features"]
    raw_features = list(raw_embedding_features or [])
    raw_layer_validation = validate_no_pca_raw_layer_contract(
        raw_layer_contract, raw_features,
    )
    raw_feature_integrity = bool(
        raw_mapping_verified
        and len(raw_features) == 2048
        and len(set(raw_features)) == 2048
        and not (set(raw_features) & set(ordered))
        and raw_layer_validation["status"] == "PASS"
    )
    if raw_feature_integrity:
        retained_no_pca = [*base_non_pca, *raw_features]
        output["no_pca"] = {
            "kind": "OFFICIAL", "status": "RUNNABLE",
            "base_non_pca_features": base_non_pca,
            "base_non_pca_count": len(base_non_pca),
            "required_raw_embedding_count": len(raw_features),
            "expected_total_feature_count_if_mapped": len(retained_no_pca),
            "retained_features": retained_no_pca,
            "removed_features": output["no_deep_pca_features"]["removed_features"],
            "removal_reasons": output["no_deep_pca_features"]["removal_reasons"],
            "feature_count": len(retained_no_pca),
            "feature_list_sha256": sha256_bytes(("\n".join(retained_no_pca) + "\n").encode()),
            "raw_mapping_verified": True,
            "raw_layer_contract": dict(raw_layer_contract or {}),
            "raw_layer_contract_validation": raw_layer_validation,
            "dependency_closure_linter": "PASS",
            "dependency_violations": [],
        }
    else:
        output["no_pca"] = {
            "kind": "OFFICIAL", "status": "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
            "base_non_pca_features": base_non_pca,
            "base_non_pca_count": len(base_non_pca),
            "required_raw_embedding_count": 2048,
            "expected_total_feature_count_if_mapped": len(base_non_pca) + 2048,
            "retained_features": None,
            "removed_features": output["no_deep_pca_features"]["removed_features"],
            "removal_reasons": output["no_deep_pca_features"]["removal_reasons"],
            "feature_count": None,
            "feature_list_sha256": None,
            "raw_mapping_verified": False,
            "raw_layer_contract": None,
            "raw_layer_contract_validation": raw_layer_validation,
            "dependency_closure_linter": "NOT_APPLICABLE_BLOCKED_RAW_MAPPING",
            "dependency_violations": [],
        }
    expected = {
        "full_new_reference": 93, "manual_only": 57, "deep_only": 35,
        "no_color": 65, "no_shape": 78, "no_texture": 86,
        "no_embed_sim": 90, "no_review_aware_training_weights": 93,
        "no_safe_smote": 93, "no_deep_pca_features": 59,
    }
    actual = {name: output[name]["feature_count"] for name in expected}
    if actual != expected:
        raise ScientificBlocker(f"Dependency-aware feature counts differ: {actual} != {expected}")
    return output


def feature_manifest_frame(features: Sequence[str] | None = None) -> pd.DataFrame:
    ordered = list(features or get_r92_feature_order())
    mapping = feature_primary_groups(ordered)
    rows = []
    for index, feature in enumerate(ordered):
        dependencies = sorted(FEATURE_DEPENDENCIES.get(feature, {mapping[feature]}))
        source, symbol, evidence = legacy.feature_evidence(feature, mapping[feature])
        if feature in FEATURE_DEPENDENCIES:
            evidence = (
                "Verified cross-family composite with dependency closure over: "
                + ", ".join(dependencies)
                + ". `other_manual` is retained only as the legacy storage bucket, "
                "not as a scientific family assignment."
            )
        rows.append({
            "feature_name": feature, "original_order": index,
            "primary_group": mapping[feature], "dependencies": "|".join(dependencies),
            "is_cross_family_composite": feature in FEATURE_DEPENDENCIES,
            "legacy_storage_bucket": (
                "other_manual" if feature in FEATURE_DEPENDENCIES else mapping[feature]
            ),
            "primary_group_status": (
                "VERIFIED_CROSS_FAMILY_COMPOSITE"
                if feature in FEATURE_DEPENDENCIES else "VERIFIED_SINGLE_FAMILY"
            ),
            "source_code_path": source, "source_line_or_symbol": symbol,
            "evidence": evidence, "verification_status": "VERIFIED",
        })
    return pd.DataFrame(rows)


def feature_dependency_graph_payload() -> dict[str, Any]:
    return {
        "families": list(FAMILIES),
        "dependencies": {
            name: sorted(value) for name, value in FEATURE_DEPENDENCIES.items()
        },
        "closure_rule": (
            "Remove a predictor when its primary group or any transitive family "
            "dependency is removed; never recompute a partial composite."
        ),
    }


def runnable_official_variants(
    feature_sets: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return official variants whose reviewed feature definitions are runnable."""

    return tuple(
        variant for variant in OFFICIAL_VARIANTS
        if feature_sets.get(variant, {}).get("status") == "RUNNABLE"
    )


def canonical_embed_similarity(raw_cosine: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(raw_cosine, dtype=float)
    return np.clip((values + 1.0) / 2.0, 0.0, 1.0)


def reconstruct_gates(
    frame: pd.DataFrame,
    *,
    prototype_available: bool,
    audit_raw_cosine: bool = False,
) -> pd.DataFrame:
    if prototype_available is not True:
        raise ScientificBlocker(
            "Gate reconstruction requires a verified prototype; source fallback values are not admissible"
        )
    mean_l = _strict_numeric(frame, "mean_L").to_numpy()
    std_l = _strict_numeric(frame, "std_L").to_numpy()
    delta_a = _strict_numeric(frame, "delta_a").to_numpy()
    delta_b = _strict_numeric(frame, "delta_b").to_numpy()
    circularity = _strict_numeric(frame, "circularity").to_numpy()
    solidity = _strict_numeric(frame, "solidity").to_numpy()
    eccentricity = _strict_numeric(frame, "eccentricity").to_numpy()
    components = _strict_numeric(frame, "components_count").to_numpy()
    extent = _strict_numeric(frame, "extent").to_numpy()
    gradient = _strict_numeric(frame, "grad_p90").to_numpy()
    touching = _strict_numeric(frame, "touching_border").to_numpy()
    components_int = components.astype(np.int64)
    touching_int = touching.astype(np.int64)
    similarity = _strict_numeric(frame, "embed_sim").to_numpy()
    if audit_raw_cosine:
        similarity = canonical_embed_similarity(similarity)
    pca_columns = [f"embed_pca_{index}" for index in range(32)]
    missing = [name for name in pca_columns if name not in frame]
    if missing:
        raise ScientificBlocker(f"Gate reconstruction missing PCA inputs: {missing}")
    pca = frame[pca_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(pca).all():
        raise ScientificBlocker("Gate reconstruction PCA inputs contain nonfinite values")
    # Match the notebook's scalar source formulas exactly. In particular,
    # mean_L is divided but not independently clipped, and g_embed uses the
    # explicit have_mu fallback rather than an additional clamp.
    g_light = np.clip(0.5 * (mean_l / 255.0) + 0.5 * np.clip(std_l / 64.0, 0, 1), 0, 1)
    g_color = np.clip(np.hypot(delta_a, delta_b) / 200.0, 0, 1)
    g_shape = np.clip(0.4 * circularity + 0.4 * solidity + 0.2 * (1.0 - np.clip(eccentricity, 0, 1)), 0, 1)
    g_embed = similarity.astype(float, copy=True)
    robust_base = 0.6 * np.where(components_int == 1, 1.0, 0.6) + 0.2 * (1.0 - np.abs(extent - 0.5) * 2.0) + 0.2 * np.clip(gradient / 200.0, 0, 1)
    g_robust = np.clip(robust_base, 0, 1) * np.where(touching_int != 0, 0.7, 1.0)
    # The notebook calls np.linalg.norm on one float32 row at a time.  Preserve
    # that reduction order for validation against the decimal feature table.
    distance = np.fromiter(
        (float(np.linalg.norm(row)) for row in pca), dtype=float, count=len(pca),
    )
    g_maha = np.clip(1.0 - distance / (distance + 5.0), 0, 1)
    g_border = (touching_int != 0).astype(float)
    g_quality = np.clip(np.mean(np.column_stack([
        g_light, g_color, g_shape, g_embed, g_robust, g_maha, 1.0 - 0.5 * g_border,
    ]), axis=1), 0, 1)
    return pd.DataFrame({
        "g_border": g_border, "g_color": g_color, "g_embed": g_embed,
        "g_light": g_light, "g_maha": g_maha, "g_quality": g_quality,
        "g_robust": g_robust, "g_shape": g_shape,
    }, index=frame.index)


def reconstruct_uniform_grid(frame: pd.DataFrame, *, audit: bool) -> pd.DataFrame:
    if audit:
        width = np.full(len(frame), 512.0)
        height = np.full(len(frame), 512.0)
    else:
        scale = _strict_numeric(frame, "scale").to_numpy()
        width = np.rint(512.0 * scale)
        height = np.rint(512.0 * scale)
    x_norm = np.clip(_strict_numeric(frame, "cx").to_numpy() / width, 0.0, np.nextafter(1.0, 0.0))
    y_norm = np.clip(_strict_numeric(frame, "cy").to_numpy() / height, 0.0, np.nextafter(1.0, 0.0))
    return pd.DataFrame({
        "grid_r_norm": y_norm, "grid_c_norm": x_norm,
        "grid_r": np.floor(y_norm * 3).astype(int),
        "grid_c": np.floor(x_norm * 3).astype(int),
    }, index=frame.index)


def gate_validation_rows(frame: pd.DataFrame, snapshot: str) -> list[dict[str, Any]]:
    reconstructed = reconstruct_gates(frame, prototype_available=True)
    rows: list[dict[str, Any]] = []
    for feature in reconstructed.columns:
        observed = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
        expected = reconstructed[feature].to_numpy(float)
        mask = np.isfinite(observed) & np.isfinite(expected)
        errors = np.abs(observed[mask] - expected[mask])
        rows.append({
            "snapshot": snapshot, "feature": feature, "rows_total": len(frame),
            "rows_compared": int(mask.sum()), "coverage": float(mask.mean()),
            "max_absolute_error": float(errors.max(initial=0)),
            "error_q50": float(np.quantile(errors, 0.5)) if len(errors) else None,
            "error_q95": float(np.quantile(errors, 0.95)) if len(errors) else None,
            "error_q99": float(np.quantile(errors, 0.99)) if len(errors) else None,
            "mismatch_count_gt_1e_12": int((errors > GATE_TOLERANCE).sum()),
            "status": "PASS" if len(errors) == len(frame) and not (errors > GATE_TOLERANCE).any() else "FAIL",
            "classification": "VERIFIED" if len(errors) == len(frame) and not (errors > GATE_TOLERANCE).any() else "BLOCKED",
        })
    return rows


def required_numeric_coverage(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for column in columns:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            valid = np.isfinite(values)
        else:
            valid = np.zeros(len(frame), dtype=bool)
        rows.append({
            "feature": column, "rows": len(frame), "finite_rows": int(valid.sum()),
            "missing_or_nonfinite_rows": int((~valid).sum()),
            "coverage": float(valid.mean()) if len(valid) else 0.0,
            "classification": "VERIFIED" if valid.all() else "BLOCKED",
        })
    return rows


def harmonize_audit_features(audit: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    result = audit[list(features)].copy()
    for column in result:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["embed_sim"] = canonical_embed_similarity(result["embed_sim"].to_numpy())
    gates = reconstruct_gates(
        audit, audit_raw_cosine=True, prototype_available=True,
    )
    for column in gates:
        result[column] = gates[column]
    grid = reconstruct_uniform_grid(audit, audit=True)
    for column in grid:
        result[column] = grid[column]
    if result.isna().any().any() or not np.isfinite(result.to_numpy(dtype=float)).all():
        bad = result.columns[result.isna().any()].tolist()
        raise ScientificBlocker(f"Harmonized audit layer has unresolved predictors: {bad}")
    return result.astype(np.float32)


def feature_semantics_evidence(
    historical: pd.DataFrame,
    audit: pd.DataFrame,
    features: Sequence[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Rebuild the label-free audit harmonization and its source evidence."""

    harmonized = harmonize_audit_features(audit, features)
    grid_historical = reconstruct_uniform_grid(historical, audit=False)
    grid_errors: dict[str, dict[str, Any]] = {}
    for column in grid_historical:
        observed = pd.to_numeric(historical[column], errors="coerce").to_numpy(float)
        expected = grid_historical[column].to_numpy(float)
        error = np.abs(observed - expected)
        grid_errors[column] = {
            "max_absolute_error": float(np.max(error)),
            "mismatch_count_gt_1e_12": int((error > 1e-12).sum()),
        }
    evidence = {
        "classification": "VERIFIED",
        "training_definition": "clamp((cosine+1)/2,0,1)",
        "audit_export_definition": "raw cosine dot(z,mu)",
        "canonical_prospective_definition": "training affine definition",
        "audit_transformation": "clamp((raw_embed_sim+1)/2,0,1)",
        "training_range": [
            float(pd.to_numeric(historical.embed_sim).min()),
            float(pd.to_numeric(historical.embed_sim).max()),
        ],
        "audit_raw_range": [
            float(pd.to_numeric(audit.embed_sim).min()),
            float(pd.to_numeric(audit.embed_sim).max()),
        ],
        "audit_harmonized_range": [
            float(harmonized.embed_sim.min()),
            float(harmonized.embed_sim.max()),
        ],
        "audit_labels_used": False,
        "source_evidence": {
            "training_notebook_path": str(NOTEBOOK),
            "training_notebook_sha256": sha256_file(NOTEBOOK),
            "training_symbols": [
                "embed_sim", "g_light", "g_color", "g_shape", "g_embed",
                "g_robust", "g_maha", "g_border", "g_quality",
            ],
            "audit_pipeline_path": str(PIPELINE_SOURCE),
            "audit_pipeline_sha256": sha256_file(PIPELINE_SOURCE),
            "audit_symbols": [
                "embed_sim", "grid_r_norm", "grid_c_norm", "f_all",
            ],
        },
        "grid_semantics": {
            "training": "uniform 3x3 floor(norm*3), norm=cx/Ws or cy/Hs",
            "audit_export": "detected grid-line indices and normalization",
            "prospective_harmonization": "training uniform 3x3 semantics",
            "training_validation": grid_errors,
        },
    }
    return evidence, harmonized


def harmonized_layer_manifest(
    pca_evidence: Mapping[str, Any],
    harmonized: pd.DataFrame,
    features: Sequence[str],
    *,
    relative_path: str,
    layer_sha256: str,
) -> dict[str, Any]:
    compatible = bool(pca_evidence.get("compatible_single_basis"))
    return {
        "classification": "VERIFIED" if compatible else "UNRESOLVED",
        "status": (
            "VERIFIED_HARMONIZED_FEATURE_LAYER"
            if compatible
            else "DERIVED_LAYER_CREATED_BUT_GLOBAL_PCA_PROTOTYPE_COMPATIBILITY_UNRESOLVED"
        ),
        "historical_control_modified": False,
        "audit_labels_used": False,
        "derived_audit_layer_path": relative_path,
        "derived_audit_layer_sha256": layer_sha256,
        "row_count": len(harmonized),
        "feature_count": len(features),
        "transformations": [
            "embed_sim affine", "eight source gate formulas",
            "training-style uniform 3x3 grid",
        ],
        "unresolved": pca_evidence.get("compatibility_conclusion"),
    }


def _effective_tags(frame: pd.DataFrame) -> pd.Series:
    if "review_tag" not in frame:
        return pd.Series("", index=frame.index, dtype="string")
    tags = frame["review_tag"].astype("string").fillna("").str.strip().str.lower()
    return tags.where(~tags.isin({"nan", "none", "<na>"}), "")


def r91_probability_mapping() -> tuple[dict[str, float], dict[str, Any]]:
    previous = pd.read_csv(R91_PREDICTIONS, low_memory=False)
    threshold_object = json.loads(R91_THRESHOLD.read_text())
    threshold = float(threshold_object.get("threshold", threshold_object.get("thr", 0.5)))
    source_counts = previous.groupby("file_name", sort=False).size()
    duplicate_excess = int(previous.duplicated("file_name", keep="first").sum())
    grouped = previous[["file_name", "proba"]].copy()
    grouped["proba"] = pd.to_numeric(grouped["proba"], errors="coerce")
    grouped = grouped.groupby("file_name", as_index=False, sort=False)["proba"].max()
    mapping = dict(zip(grouped["file_name"].astype(str), grouped["proba"].astype(float)))
    return mapping, {
        "source_behavior": "groupby(file_name,sort=False).proba.max",
        "threshold": threshold, "source_rows": int(len(previous)),
        "unique_file_names": int(len(grouped)), "duplicate_excess_rows": duplicate_excess,
        "duplicate_file_name_keys": int((source_counts > 1).sum()),
        "maximum_rows_per_file_name": int(source_counts.max()),
        "source_explicitly_justifies_maximum": True,
    }


REVIEW_TAG_MULTIPLIERS = {"accept": 1.0, "flip": 1.0, "sus_accept": 0.4, "sus_flip": 0.4, "skip": 0.0}


def review_weights(frame: pd.DataFrame, *, historical_geometry: bool) -> tuple[np.ndarray, dict[str, Any]]:
    mapping, join = r91_probability_mapping()
    threshold = float(join["threshold"])
    labels = frame["label"].to_numpy(np.int8)
    reviewed = _strict_numeric(frame, "reviewed").to_numpy(int) == 1
    probability = frame["file_name"].astype(str).map(mapping).to_numpy(float)
    mapped = np.isfinite(probability)
    known_reviewed = mapped & reviewed
    weights = np.ones(len(frame), dtype=np.float32)
    prediction = probability >= threshold
    correct = prediction == labels
    weights[known_reviewed] = np.where(correct[known_reviewed], 0.7, 1.0).astype(np.float32)
    uncertain = np.abs(probability - threshold) <= 0.05
    weights[known_reviewed & uncertain] *= np.float32(0.7)
    border, tiny = historical_border_tiny_masks(frame) if historical_geometry else corrected_border_tiny_masks(frame)
    weights[reviewed & border] *= np.float32(0.5)
    weights[tiny & (labels == 0)] *= np.float32(0.6)
    tags = _effective_tags(frame)
    multipliers = tags.map(lambda value: REVIEW_TAG_MULTIPLIERS.get(str(value), 1.0)).to_numpy(np.float32)
    weights *= multipliers
    if np.any(weights <= 0):
        raise ScientificBlocker("Skip rows must be removed before final weight construction")
    previous_keys = pd.read_csv(R91_PREDICTIONS, usecols=["file_name"])["file_name"].astype(str)
    train_counts = frame["file_name"].astype(str).value_counts(sort=False)
    previous_counts = previous_keys.value_counts(sort=False)
    shared = train_counts.index.intersection(previous_counts.index)
    left = train_counts.loc[shared]
    right = previous_counts.loc[shared]
    key_cases = {
        "one_to_one": int(((left == 1) & (right == 1)).sum()),
        "one_to_many": int(((left == 1) & (right > 1)).sum()),
        "many_to_one": int(((left > 1) & (right == 1)).sum()),
        "many_to_many": int(((left > 1) & (right > 1)).sum()),
    }
    join.update({
        "train_rows": len(frame), "mapped_rows": int(mapped.sum()),
        "missing_rows": int((~mapped).sum()), "join_rate": float(mapped.mean()),
        "missing_rate": float((~mapped).mean()),
        "train_unique_file_names": int(len(train_counts)),
        "train_duplicate_file_name_keys": int((train_counts > 1).sum()),
        "train_maximum_rows_per_file_name": int(train_counts.max()),
        "shared_unique_file_names": int(len(shared)),
        "train_only_unique_file_names": int(len(train_counts.index.difference(previous_counts.index))),
        "r91_only_unique_file_names": int(len(previous_counts.index.difference(train_counts.index))),
        "shared_key_cardinality_cases": key_cases,
        "join_treatment": "aggregate r91 by file_name maximum before many-to-one lookup onto training rows",
        "reviewed_rows": int(reviewed.sum()), "reviewed_mapped_rows": int(known_reviewed.sum()),
        "reviewed_missing_rows": int((reviewed & ~mapped).sum()),
        "unaggregated_many_to_many_join_rows": int(frame[["file_name"]].merge(
            pd.DataFrame({"file_name": previous_keys}), on="file_name", how="inner"
        ).shape[0]),
        "geometry": "historical_inferred_fractional_extents" if historical_geometry else "corrected_fixed_512_with_touching_border",
        "border_rows": int(border.sum()), "reviewed_border_rows": int((reviewed & border).sum()),
        "tiny_rows": int(tiny.sum()), "tiny_negative_rows": int((tiny & (labels == 0)).sum()),
        "mean_weight": float(weights.mean()), "min_weight": float(weights.min()), "max_weight": float(weights.max()),
        "weight_sha256_float32_le": sha256_array(weights.astype("<f4", copy=False)),
        "review_tag_counts": {str(key): int(value) for key, value in tags.value_counts(dropna=False).items()},
    })
    return weights, join


def deterministic_permutation(length: int, seed: int) -> np.ndarray:
    return np.random.RandomState(seed).permutation(np.arange(length, dtype=np.int64))


def split_assignment_hash(
    stable_ids: Sequence[str], splits: Sequence[str], labels: Sequence[int], eligible: Sequence[bool],
    split_order: Sequence[int] | None = None,
) -> str:
    frame = pd.DataFrame({
        "stable_candidate_id": list(stable_ids), "split": list(splits),
        "label": np.asarray(labels, dtype=np.int8), "eligible": np.asarray(eligible, dtype=bool),
    })
    if split_order is not None:
        frame["split_order"] = np.asarray(split_order, dtype=np.int64)
    return sha256_bytes(frame.to_csv(index=False, lineterminator="\n").encode())


def validate_card_split(card_ids: Sequence[str], splits: Sequence[str]) -> dict[str, Any]:
    frame = pd.DataFrame({"card_id": list(card_ids), "split": list(splits)})
    memberships = frame.groupby("card_id").split.nunique()
    leaking = sorted(memberships[memberships > 1].index.astype(str))
    return {
        "status": "PASS" if not leaking else "FAIL",
        "leaking_cards": leaking, "leaking_card_count": len(leaking),
    }


def variant_training_semantics(variant: str) -> dict[str, Any]:
    if variant not in OFFICIAL_VARIANTS and variant != EXPLORATORY_VARIANT:
        raise AmendmentError(f"Unknown variant: {variant}")
    return {
        "augmentation_enabled": True,
        "mixup_enabled": True,
        "dropout_enabled": True,
        "jitter_enabled": True,
        "raw_split_shuffle_enabled": True,
        "review_weights_enabled": variant != "no_review_aware_training_weights",
        "safe_smote_enabled": variant != "no_safe_smote",
        "seed": BASE_SEED,
    }


def r92_control_gate(
    *, ordered_population_match: bool, expected_metrics_match: bool,
    maximum_probability_difference: float, selected_score_hash_match: bool,
    tolerance: float = 1e-12,
) -> bool:
    return bool(
        ordered_population_match and expected_metrics_match and selected_score_hash_match
        and np.isfinite(maximum_probability_difference)
        and maximum_probability_difference <= tolerance
    )


def derive_ready_status(evidence: Mapping[str, bool]) -> dict[str, Any]:
    failed = sorted(key for key, passed in evidence.items() if not bool(passed))
    return {
        "ready_for_full_awaiting_external_review": not failed,
        "status": "READY" if not failed else "BLOCKED",
        "failed_gates": failed,
    }


def validate_run_transition(identity: Mapping[str, Any], target_state: str) -> None:
    permitted = set(identity.get("permitted_transitions", []))
    if target_state not in permitted:
        raise IntegrityError(
            f"Transition {identity.get('state')} -> {target_state} is not permitted"
        )


def augmentation_plan_counts(
    y: np.ndarray, *, continuous_feature_count: int, seed: int = BASE_SEED,
) -> dict[str, Any]:
    """Replay historical augmentation RNG sufficiently to obtain exact class counts."""
    labels = np.asarray(y, dtype=np.int8)
    raw_counts = Counter(labels.tolist())
    rng = np.random.RandomState(seed)
    mixed_labels = [labels]
    mix_added: dict[int, int] = {}
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        count = int(math.ceil(len(indices) * 0.5))
        mix_added[int(label)] = count
        rng.choice(indices, size=count, replace=len(indices) < count)
        rng.choice(indices, size=count, replace=len(indices) < count)
        rng.beta(0.2, 0.2, size=count)
        rng.rand(count)  # historical synthetic-group source choice
        mixed_labels.append(np.full(count, label, dtype=np.int8))
    labels_after_mix = np.concatenate(mixed_labels)
    dropout_count = len(labels_after_mix)
    selected = rng.choice(
        np.arange(len(labels_after_mix)), size=dropout_count,
        replace=len(labels_after_mix) < dropout_count,
    )
    labels_after_dropout = np.concatenate([labels_after_mix, labels_after_mix[selected]])
    # Consume the exact dropout-mask stream without allocating its full matrix.
    remaining = dropout_count
    while remaining:
        rows = min(8192, remaining)
        rng.rand(rows, continuous_feature_count)
        remaining -= rows
    jitter_count = int(math.ceil(len(labels_after_dropout) * 0.5))
    selected = rng.choice(
        np.arange(len(labels_after_dropout)), size=jitter_count,
        replace=len(labels_after_dropout) < jitter_count,
    )
    labels_after_jitter = np.concatenate([labels_after_dropout, labels_after_dropout[selected]])
    after_mix = Counter(labels_after_mix.tolist())
    after_dropout = Counter(labels_after_dropout.tolist())
    after_jitter = Counter(labels_after_jitter.tolist())
    majority = max(after_jitter.values())
    minority_label = min(after_jitter, key=after_jitter.get)
    smote_target = int(math.floor(majority * 0.5))
    smote_added = max(0, smote_target - after_jitter[minority_label])
    return {
        "raw": {str(k): int(v) for k, v in sorted(raw_counts.items())},
        "mixup_added": {str(k): int(v) for k, v in sorted(mix_added.items())},
        "post_mixup": {str(k): int(v) for k, v in sorted(after_mix.items())},
        "post_dropout": {str(k): int(v) for k, v in sorted(after_dropout.items())},
        "post_jitter": {str(k): int(v) for k, v in sorted(after_jitter.items())},
        "rng_seed": seed, "continuous_feature_count": continuous_feature_count,
        "safe_smote_synthetic_rows": int(smote_added),
        "final_fit_rows": int(sum(after_jitter.values()) + smote_added),
    }


def replay_augmentation_weights(
    y: np.ndarray, weights: np.ndarray, *, seed: int = BASE_SEED,
    continuous_feature_count: int, include_smote: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Replay source RNG and weight propagation without materializing feature matrices."""
    labels = np.asarray(y, dtype=np.int8)
    values = np.asarray(weights, dtype=np.float32)
    if len(labels) != len(values):
        raise AmendmentError("Label and weight vectors have different lengths")
    rng = np.random.RandomState(seed)
    mixed_labels = [labels]
    mixed_weights = [values]
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        count = int(math.ceil(len(indices) * 0.5))
        first = rng.choice(indices, size=count, replace=len(indices) < count)
        second = rng.choice(indices, size=count, replace=len(indices) < count)
        rng.beta(0.2, 0.2, size=count)
        rng.rand(count)
        mixed_labels.append(np.full(count, label, dtype=np.int8))
        mixed_weights.append((0.5 * (values[first] + values[second])).astype(np.float32))
    labels = np.concatenate(mixed_labels)
    values = np.concatenate(mixed_weights).astype(np.float32, copy=False)
    count = len(labels)
    selected = rng.choice(np.arange(count), size=count, replace=False)
    labels = np.concatenate([labels, labels[selected]])
    values = np.concatenate([values, values[selected]]).astype(np.float32, copy=False)
    remaining = count
    while remaining:
        rows = min(8192, remaining)
        rng.rand(rows, continuous_feature_count)
        remaining -= rows
    count = int(math.ceil(len(labels) * 0.5))
    selected = rng.choice(np.arange(len(labels)), size=count, replace=False)
    labels = np.concatenate([labels, labels[selected]])
    values = np.concatenate([values, values[selected]]).astype(np.float32, copy=False)
    pre_smote_length = len(labels)
    synthetic_rows = 0
    if include_smote:
        class_counts = Counter(labels.tolist())
        minority = min(class_counts, key=class_counts.get)
        target = int(math.floor(max(class_counts.values()) * 0.5))
        synthetic_rows = max(0, target - class_counts[minority])
        if synthetic_rows:
            labels = np.concatenate([labels, np.full(synthetic_rows, minority, dtype=np.int8)])
            synthetic_weight = np.float32(values[labels[:pre_smote_length] == minority].mean())
            values = np.concatenate([values, np.full(synthetic_rows, synthetic_weight, dtype=np.float32)])
    return labels, values, {
        "pre_smote_rows": pre_smote_length,
        "safe_smote_synthetic_rows": synthetic_rows,
        "final_rows": len(labels), "mean_weight": float(values.mean()),
        "weight_sha256_float32_le": sha256_array(values.astype("<f4", copy=False)),
        "class_counts": {str(k): int(v) for k, v in sorted(Counter(labels.tolist()).items())},
    }


def recompute_review_weight_evidence(
    stored: Mapping[str, Any],
    historical: pd.DataFrame,
    split: pd.DataFrame,
    features: Sequence[str],
) -> tuple[bool, bool, bool]:
    """Recompute weight and augmentation predicates from canonical sources."""

    train_idx, heldout_idx, _ = historical_group_split(
        historical.card_id.astype(str).to_numpy(),
    )
    expected_train_order = split.loc[
        split.split.astype(str).eq("training")
    ].sort_values("split_order").stable_candidate_id.astype(str).to_numpy()
    if not np.array_equal(
        historical.iloc[train_idx].stable_candidate_id.astype(str).to_numpy(),
        expected_train_order,
    ):
        return False, False, False
    train = historical.iloc[train_idx].copy()
    continuous = [
        feature for feature in features
        if train[feature].nunique(dropna=False) > 10
    ]
    historical_weights, historical_info = review_weights(
        train, historical_geometry=True,
    )
    corrected_weights, corrected_info = review_weights(
        train, historical_geometry=False,
    )
    plan = augmentation_plan_counts(
        train.label.to_numpy(), continuous_feature_count=len(continuous),
    )
    historical_y, historical_w, historical_aug = replay_augmentation_weights(
        train.label.to_numpy(), historical_weights,
        continuous_feature_count=len(continuous), include_smote=False,
    )
    _, final_w, final_aug = replay_augmentation_weights(
        train.label.to_numpy(), historical_weights,
        continuous_feature_count=len(continuous), include_smote=True,
    )
    corrected_y, corrected_w, _ = replay_augmentation_weights(
        train.label.to_numpy(), corrected_weights,
        continuous_feature_count=len(continuous), include_smote=False,
    )
    historical_section = stored.get("historical_legacy_geometry", {})
    corrected_section = stored.get("corrected_fixed_512_geometry", {})
    expected_mean = float(
        json.loads((R92_DIR / "metrics.json").read_text())["meta"][
            "weights_mean_train"
        ]
    )
    historical_info_matches = all(
        historical_section.get(key) == value
        for key, value in historical_info.items()
    )
    corrected_info_matches = all(
        corrected_section.get(key) == value
        for key, value in corrected_info.items()
    )
    review_weight_ok = bool(
        historical_info_matches
        and corrected_info_matches
        and historical_section.get("post_augmentation_rows") == len(historical_w)
        and historical_section.get("post_augmentation_mean_weight")
        == float(historical_w.mean())
        and historical_section.get("expected_r92_metadata_mean_weight")
        == expected_mean
        and historical_section.get("post_augmentation_mean_parity")
        is (abs(expected_mean - float(historical_w.mean())) <= 1e-12)
        and historical_section.get("post_augmentation_weight_sha256_float32_le")
        == sha256_array(historical_w.astype("<f4", copy=False))
        and historical_section.get("post_augmentation_label_sha256_int8")
        == sha256_array(historical_y)
        and historical_section.get("final_fit_weight_validation") == final_aug
        and corrected_section.get("post_augmentation_rows") == len(corrected_w)
        and corrected_section.get("post_augmentation_mean_weight")
        == float(corrected_w.mean())
        and corrected_section.get("post_augmentation_weight_sha256_float32_le")
        == sha256_array(corrected_w.astype("<f4", copy=False))
        and corrected_section.get("post_augmentation_label_sha256_int8")
        == sha256_array(corrected_y)
    )
    raw_train = np.where(~np.isin(
        historical.card_id.astype(str).to_numpy(),
        historical.iloc[heldout_idx].card_id.astype(str).unique(),
    ))[0]
    raw_heldout = np.where(np.isin(
        historical.card_id.astype(str).to_numpy(),
        historical.iloc[heldout_idx].card_id.astype(str).unique(),
    ))[0]
    shuffle = stored.get("shuffle_evidence", {})
    augmentation_ok = bool(
        stored.get("augmentation_counts") == plan
        and stored.get("continuous_feature_derivation") == {
            "rule": "nunique(dropna=False)>10 on the ordered r92 training predictors",
            "count": len(continuous),
            "ordered_features": continuous,
            "ordered_feature_list_sha256": sha256_bytes(
                ("\n".join(continuous) + "\n").encode()
            ),
        }
        and stored.get("historical_augmented_label_counts") == {
            "negative": int((historical_y == 0).sum()),
            "positive": int((historical_y == 1).sum()),
        }
        and stored.get("historical_safe_smote") == {
            "synthetic_positive_rows": final_aug["safe_smote_synthetic_rows"],
            "final_rows": final_aug["final_rows"],
            "k_neighbors": 3,
            "sampling_strategy": 0.5,
            "final_weight_sha256_float32_le": sha256_array(
                final_w.astype("<f4", copy=False)
            ),
        }
        and shuffle.get("training_before_ordered_id_sha256") == sha256_bytes(
            ("\n".join(historical.iloc[raw_train].stable_candidate_id.astype(str)) + "\n").encode()
        )
        and shuffle.get("training_after_ordered_id_sha256") == sha256_bytes(
            ("\n".join(historical.iloc[train_idx].stable_candidate_id.astype(str)) + "\n").encode()
        )
        and shuffle.get("heldout_before_ordered_id_sha256") == sha256_bytes(
            ("\n".join(historical.iloc[raw_heldout].stable_candidate_id.astype(str)) + "\n").encode()
        )
        and shuffle.get("heldout_after_ordered_id_sha256") == sha256_bytes(
            ("\n".join(historical.iloc[heldout_idx].stable_candidate_id.astype(str)) + "\n").encode()
        )
        and shuffle.get("training_order_changed") is True
        and shuffle.get("heldout_order_changed") is True
    )
    review_parity = bool(
        review_weight_ok
        and historical_section.get("post_augmentation_mean_parity") is True
    )
    return review_weight_ok and augmentation_ok, review_parity, augmentation_ok


def score_r92_control(audit: pd.DataFrame) -> tuple[dict[str, Any], np.ndarray]:
    payload = joblib.load(R92_MODEL)
    features = list(payload["features"])
    matrix = audit[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    probability = np.asarray(payload["pipeline"].predict_proba(matrix)[:, 1], dtype=float)
    selected = pd.read_csv(SELECTED_V3_SCORES, low_memory=False)
    selected_population = json.loads(SELECTED_V3_POPULATION.read_text())
    comparison = ["img_folder", "image", "id", "human_label"]
    ordered_population_match = audit[comparison].astype(str).reset_index(drop=True).equals(
        selected[comparison].astype(str).reset_index(drop=True)
    )
    maximum_difference = float(np.max(np.abs(probability - selected["canonical_score"].to_numpy(float))))
    labels = audit["human_label"].to_numpy(np.int8)
    predictions = probability >= 0.5
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    selected_score_member = immutable_zip_member_evidence(
        SELECTED_V3_ZIP, "scored_candidates_r92.csv",
    )
    selected_population_member = immutable_zip_member_evidence(
        SELECTED_V3_ZIP, "evaluation_population.json",
    )
    hash_match = bool(
        sha256_file(SELECTED_V3_SCORES) == selected_score_member["member_sha256"]
        and SELECTED_V3_SCORES.stat().st_size == selected_score_member["member_size_bytes"]
        and sha256_file(SELECTED_V3_POPULATION) == selected_population_member["member_sha256"]
        and SELECTED_V3_POPULATION.stat().st_size == selected_population_member["member_size_bytes"]
    )
    selected_labels = pd.to_numeric(selected["human_label"], errors="raise").to_numpy(np.int8)
    selected_probability = pd.to_numeric(selected["canonical_score"], errors="raise").to_numpy(float)
    selected_prediction = selected_probability >= 0.5
    expected_tn, expected_fp, expected_fn, expected_tp = confusion_matrix(
        selected_labels, selected_prediction, labels=[0, 1],
    ).ravel()
    expected = {
        "n": int(len(selected_labels)),
        "positives": int(selected_labels.sum()),
        "negatives": int((selected_labels == 0).sum()),
        "tn": int(expected_tn), "fp": int(expected_fp),
        "fn": int(expected_fn), "tp": int(expected_tp),
        "average_precision": float(average_precision_score(selected_labels, selected_probability)),
        "roc_auc": float(roc_auc_score(selected_labels, selected_probability)),
    }
    expected_cards = sorted(map(str, selected_population.get("cards", [])))
    observed_cards = sorted(audit["img_folder"].astype(str).unique())
    population_metadata_match = bool(
        len(audit) == int(selected_population.get("canonical_evaluation_rows", -1))
        and int(labels.sum()) == int(selected_population.get("positives", -1))
        and int((labels == 0).sum()) == int(selected_population.get("negatives", -1))
        and observed_cards == expected_cards
    )
    population_match = bool(ordered_population_match and population_metadata_match)
    result = {
        "ordered_population_match": ordered_population_match,
        "population_metadata_match": population_metadata_match,
        "expected_population_source": {
            "path": str(SELECTED_V3_POPULATION),
            "sha256": sha256_file(SELECTED_V3_POPULATION),
            "cards": expected_cards,
        },
        "selected_score_source_path": str(SELECTED_V3_SCORES),
        "selected_score_source_sha256": sha256_file(SELECTED_V3_SCORES),
        "selected_score_source_hash_match": hash_match,
        "immutable_zip_score_member": selected_score_member,
        "immutable_zip_population_member": selected_population_member,
        "max_probability_difference": maximum_difference, "tolerance": 1e-12,
        "n": len(labels), "positives": int(labels.sum()), "negatives": int((labels == 0).sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "average_precision": float(average_precision_score(labels, probability)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "model_sha256": sha256_file(R92_MODEL), "feature_count": len(features),
        "expected_metrics_source": {
            "path": str(SELECTED_V3_SCORES),
            "sha256": sha256_file(SELECTED_V3_SCORES),
            "row_selector": "all rows in saved order; human_label and canonical_score; threshold=0.5",
            "derived_metrics": expected,
        },
    }
    result["expected_metrics_match"] = all(
        abs(float(result[key]) - value) <= 1e-12 for key, value in expected.items()
    )
    result["status"] = "PASS" if r92_control_gate(
        ordered_population_match=population_match,
        expected_metrics_match=result["expected_metrics_match"],
        maximum_probability_difference=maximum_difference,
        selected_score_hash_match=hash_match,
    ) else "FAIL"
    result["classification"] = "VERIFIED" if result["status"] == "PASS" else "BLOCKED"
    return result, probability


def verify_zip(path: Path, *, embedded_run_manifest: bool = False) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        bad_crc = archive.testzip()
        names = archive.namelist()
        duplicate_count = len(names) - len(set(names))
        result: dict[str, Any] = {
            "path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
            "member_count": len(names), "duplicate_member_count": duplicate_count,
            "crc_status": "PASS" if bad_crc is None else "FAIL", "first_bad_crc_member": bad_crc,
        }
        if embedded_run_manifest:
            manifests = [
                name for name in names
                if len(Path(name).parts) == 2
                and name.endswith("/OUTPUT_MANIFEST_FINAL.tsv")
            ]
            failures: list[dict[str, Any]] = []
            checked = 0
            if len(manifests) == 1:
                prefix = manifests[0].rsplit("/", 1)[0] + "/"
                manifest_text = archive.read(manifests[0]).decode()
                parsed = list(csv.reader(io.StringIO(manifest_text), delimiter="\t"))
                if parsed and parsed[0][:3] == ["relative_path", "size_bytes", "sha256"]:
                    rows = list(csv.DictReader(io.StringIO(manifest_text), delimiter="\t"))
                    manifest_schema = "HEADERED_AT_LEAST_3_COLUMNS"
                elif parsed and all(
                    len(fields) == 3 and fields[1].isdigit()
                    and re.fullmatch(r"[0-9a-f]{64}", fields[2])
                    for fields in parsed
                ):
                    rows = [
                        {"relative_path": fields[0], "size_bytes": fields[1], "sha256": fields[2]}
                        for fields in parsed
                    ]
                    manifest_schema = "HEADERLESS_3_COLUMNS"
                else:
                    rows = []
                    manifest_schema = "UNRECOGNIZED"
                    failures.append({"reason": "manifest_schema_unrecognized"})
                for row in rows:
                    member = prefix + row["relative_path"]
                    checked += 1
                    if member not in names:
                        failures.append({"member": member, "reason": "missing"})
                        continue
                    data = archive.read(member)
                    if len(data) != int(row["size_bytes"]) or sha256_bytes(data) != row["sha256"]:
                        failures.append({"member": member, "reason": "size_or_hash"})
            else:
                failures.append({"reason": "manifest_count", "count": len(manifests)})
            result.update({
                "embedded_manifest_rows_checked": checked,
                "embedded_manifest_schema": locals().get("manifest_schema", "NOT_AVAILABLE"),
                "embedded_manifest_failures": failures,
                "embedded_manifest_status": "PASS" if not failures else "FAIL",
            })
        result["status"] = "PASS" if bad_crc is None and duplicate_count == 0 and not result.get("embedded_manifest_failures") else "FAIL"
        return result


def verify_published_review_bundle(
    run_dir: Path,
    bundle_path: Path,
    package_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen a canonical Review publication and bind every member to the live Run."""

    if not bundle_path.is_file() or bundle_path.is_symlink():
        raise IntegrityError("Canonical Review bundle is not a regular file")
    reopened = verify_zip(bundle_path, embedded_run_manifest=True)
    live_files = {
        path.relative_to(run_dir).as_posix(): path
        for path in run_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    expected_names = {
        f"{run_dir.name}/{relative}" for relative in live_files
    }
    failures: list[str] = []
    with zipfile.ZipFile(bundle_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            failures.append("duplicate_members")
        if set(names) != expected_names:
            failures.append("member_set")
        if any(
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or Path(name).parts[:1] != (run_dir.name,)
            for name in names
        ):
            failures.append("unsafe_member")
        for relative, live_path in live_files.items():
            member = f"{run_dir.name}/{relative}"
            if member not in names or archive.read(member) != live_path.read_bytes():
                failures.append(f"live_byte_mismatch:{relative}")
    expected_uncompressed = sum(path.stat().st_size for path in live_files.values())
    if (
        reopened.get("status") != "PASS"
        or reopened.get("crc_status") != "PASS"
        or reopened.get("embedded_manifest_status") != "PASS"
        or reopened.get("sha256") != package_verification.get("bundle_sha256")
        or reopened.get("size_bytes") != package_verification.get("bundle_size_bytes")
        or reopened.get("member_count") != package_verification.get("member_count")
        or reopened.get("member_count") != len(live_files)
        or package_verification.get("uncompressed_size_bytes") != expected_uncompressed
        or package_verification.get("independent_reopen_member_verification") != "PASS"
        or package_verification.get("verification_failures")
    ):
        failures.append("package_verification_binding")
    if failures:
        raise IntegrityError(
            "Canonical Review bundle reopen failed: " + ",".join(failures)
        )
    return {
        "status": "PASS",
        "bundle_path": str(bundle_path),
        "bundle_sha256": reopened["sha256"],
        "bundle_size_bytes": reopened["size_bytes"],
        "member_count": reopened["member_count"],
        "crc_status": reopened["crc_status"],
        "embedded_manifest_status": reopened["embedded_manifest_status"],
        "unique_safe_member_set": "PASS",
        "live_member_byte_equality": "PASS",
    }


def immutable_zip_member_evidence(zip_path: Path, member_suffix: str) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        matches = [name for name in archive.namelist() if name.endswith("/" + member_suffix)]
        if len(matches) != 1:
            raise IntegrityError(
                f"Expected one {member_suffix!r} member in {zip_path}, found {matches}"
            )
        member = matches[0]
        data = archive.read(member)
    return {
        "zip_path": str(zip_path), "zip_sha256": sha256_file(zip_path),
        "member": member, "member_size_bytes": len(data),
        "member_sha256": sha256_bytes(data),
    }


def verify_prior_run_inventory() -> dict[str, Any]:
    manifest = pd.read_csv(PRIOR_RUN / "OUTPUT_MANIFEST_FINAL.tsv", sep="\t")
    failures = []
    for row in manifest.itertuples(index=False):
        path = PRIOR_RUN / row.relative_path
        if not path.exists() or path.stat().st_size != int(row.size_bytes) or sha256_file(path) != row.sha256:
            failures.append(str(row.relative_path))
    digest = hashlib.sha256()
    for path in sorted(p for p in PRIOR_RUN.rglob("*") if p.is_file()):
        relative = path.relative_to(PRIOR_RUN).as_posix()
        digest.update(f"{relative}\t{sha256_file(path)}\n".encode())
    return {
        "manifest_rows": int(len(manifest)), "manifest_failures": failures,
        "manifest_status": "PASS" if not failures else "FAIL",
        "manifest_sha256": sha256_file(PRIOR_RUN / "OUTPUT_MANIFEST_FINAL.tsv"),
        "deterministic_tree_hash_relpath_tab_sha_lines": digest.hexdigest(),
        "zip": verify_zip(PRIOR_ZIP, embedded_run_manifest=True),
    }


def source_inventory() -> list[tuple[str, Path, str]]:
    return [
        ("protocol_amendment_task_spec", TASK_SPEC, "locked protocol"),
        ("historical_training_snapshot", HISTORICAL_TRAINING_CSV, "candidate exact r92 fixed table"),
        ("historical_tiled_coco", HISTORICAL_TILED_COCO, "training crop/mask provenance"),
        ("current_training_snapshot", CURRENT_TRAINING_CSV, "comparison table"),
        ("canonical_external_audit", AUDIT_CSV, "locked external audit"),
        ("r92_checkpoint", R92_MODEL, "historical control"),
        ("r92_historical_predictions", R92_PREDICTIONS, "historical replay source"),
        ("r92_config", R92_DIR / "config.json", "historical configuration"),
        ("r92_metrics", R92_DIR / "metrics.json", "historical metrics"),
        ("r92_threshold", R92_DIR / "threshold_r92.json", "historical threshold"),
        ("r91_predictions", R91_PREDICTIONS, "review-weight source"),
        ("r91_threshold", R91_THRESHOLD, "review-weight threshold"),
        ("selected_v3_scores", SELECTED_V3_SCORES, "r92 audit comparison"),
        ("selected_v3_evaluation_population", SELECTED_V3_POPULATION, "r92 population definition"),
        ("selected_v3_review_zip", SELECTED_V3_ZIP, "immutable audit evidence"),
        ("prior_amendment_review_zip", PRIOR_ZIP, "immutable prior evidence"),
        ("training_notebook", NOTEBOOK, "historical source"),
        ("gate_source", GATE_SOURCE, "feature source"),
        ("inference_pipeline_source", PIPELINE_SOURCE, "audit feature source"),
        ("fixed_boundary_source", PLOTS_NOTEBOOK, "corrected boundary source"),
        ("frozen_pca", FROZEN_PCA, "deployment transform"),
        ("frozen_prototype", FROZEN_PROTO, "deployment prototype"),
        ("pre_r92_pca", PRE_R92_PCA, "pre-IMG9429 evolving transform"),
        ("pre_r92_prototype", PRE_R92_PROTO, "pre-IMG9429 evolving prototype"),
        ("post_r92_pca", POST_R92_PCA, "post-IMG9429 evolving transform"),
        ("post_r92_prototype", POST_R92_PROTO, "post-IMG9429 evolving prototype"),
        ("candidate_raw_embeddings", RAW_EMBEDDINGS, "unmapped current raw cache"),
        ("post_r92_raw_embeddings", POST_R92_RAW, "unmapped historical raw cache"),
        ("encoder_weights", ENCODER_WEIGHTS, "ResNet50 ImageNet1K V2"),
    ]


def inventory_frame() -> pd.DataFrame:
    rows = []
    for logical_name, path, role in source_inventory():
        rows.append({
            "logical_name": logical_name, "path": str(path), "role": role,
            "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None,
            "classification": "VERIFIED" if path.exists() and path.is_file() else "BLOCKED",
            "readable": os.access(path, os.R_OK) if path.exists() else False,
            "writable_by_process": os.access(path, os.W_OK) if path.exists() else False,
            "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
            "expected_sha256": EXPECTED_HASHES.get(str(path)),
            "expected_hash_match": (
                sha256_file(path) == EXPECTED_HASHES[str(path)] if path.exists() and str(path) in EXPECTED_HASHES else None
            ),
        })
    return pd.DataFrame(rows)


def hardware_software() -> dict[str, Any]:
    packages = {}
    for module in ("numpy", "pandas", "sklearn", "imblearn", "xgboost", "joblib", "scipy", "pytest", "torch", "cupy"):
        try:
            loaded = __import__(module)
            distribution = {"sklearn": "scikit-learn", "imblearn": "imbalanced-learn"}.get(module, module)
            try:
                installed_metadata = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                installed_metadata = None
            runtime_version = getattr(loaded, "__version__", "unknown")
            packages[module] = {
                "runtime_version": runtime_version,
                "installed_distribution_version": installed_metadata,
                "versions_match": str(runtime_version) == str(installed_metadata),
            }
        except Exception as exc:
            packages[module] = {"status": "UNAVAILABLE", "error": str(exc)}
    nvidia = run_command([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ], cwd=STUDY_ROOT)
    try:
        import torch
        torch_info = {
            "version": torch.__version__, "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        torch_info = {"error": str(exc), "cuda_available": False}
    build_info = xgb.build_info()
    return {
        "python": sys.version, "executable": sys.executable, "packages": packages,
        "cpu_logical": os.cpu_count(), "ram_total_bytes": psutil.virtual_memory().total,
        "disk_free_bytes": shutil.disk_usage(STUDY_ROOT).free,
        "nvidia_smi": asdict(nvidia), "torch": torch_info,
        "xgboost_build_info": build_info,
        "xgboost_cuda_compiled": bool(build_info.get("USE_CUDA", False)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
    }


def xgboost_cuda_capability_probe() -> dict[str, Any]:
    build_info = xgb.build_info()
    result: dict[str, Any] = {
        "xgboost_version": xgb.__version__, "build_info": build_info,
        "use_cuda_compiled": bool(build_info.get("USE_CUDA", False)),
        "requested_device": "cuda", "status": "FAIL",
    }
    if not result["use_cuda_compiled"]:
        result["failure_reason"] = "XGBoost build_info USE_CUDA=false; CPU or fallback execution is forbidden"
        return result
    try:
        matrix = xgb.DMatrix(np.array([[0.0], [1.0], [2.0], [3.0]], np.float32), label=np.array([0, 0, 1, 1]))
        with warnings.catch_warnings(record=True) as caught:
            booster = xgb.train({"tree_method": "hist", "device": "cuda", "objective": "binary:logistic"}, matrix, num_boost_round=1)
        configuration = json.loads(booster.save_config())
        warning_text = [str(item.message) for item in caught]
        serialized = json.dumps(configuration).lower()
        result.update({"booster_configuration": configuration, "warnings": warning_text})
        if "cuda" in serialized and not any("fallback" in text.lower() or "not compiled" in text.lower() for text in warning_text):
            result["status"] = "PASS"
        else:
            result["failure_reason"] = "CUDA device not proven in saved booster configuration"
    except Exception as exc:
        result["failure_reason"] = repr(exc)
    return result


def r92_model_parameter_lock() -> dict[str, Any]:
    """Derive the amended classifier lock from the hashed r92 configuration."""

    source = R92_DIR / "config.json"
    payload = json.loads(source.read_text())
    historical = payload["xgb_kwargs"]
    best = payload["best_params"]
    parameters = {
        "objective": historical["objective"],
        "eval_metric": historical["eval_metric"],
        "learning_rate": best["clf__learning_rate"],
        "max_depth": best["clf__max_depth"],
        "subsample": best["clf__subsample"],
        "colsample_bytree": best["clf__colsample_bytree"],
        "gamma": best["clf__gamma"],
        "min_child_weight": best["clf__min_child_weight"],
        "reg_alpha": best["clf__reg_alpha"],
        "reg_lambda": best["clf__reg_lambda"],
        "max_bin": best["clf__max_bin"],
        "scale_pos_weight": historical["scale_pos_weight"],
        "tree_method": historical["tree_method"],
        "random_state": historical["random_state"],
        "n_jobs": historical["n_jobs"],
        "verbosity": historical["verbosity"],
    }
    checkpoint = joblib.load(R92_MODEL)
    checkpoint_classifier = checkpoint["pipeline"].named_steps["clf"]
    checkpoint_parameters = checkpoint_classifier.get_params()
    checkpoint_fields = {
        key: checkpoint_parameters[key] for key in parameters
    }
    if any(
        checkpoint_fields[key] != parameters[key] for key in parameters
    ):
        raise IntegrityError(
            "r92 config best parameters differ from the immutable checkpoint"
        )
    full = {
        **parameters,
        "n_estimators": int(payload["N_ESTIMATORS_BIG"]),
        "early_stopping_rounds": int(payload["EARLY_STOP_ROUNDS"]),
        "device": historical["device"],
    }
    smoke = {
        **parameters,
        "n_estimators": 16,
        "early_stopping_rounds": 4,
        "device": "cuda",
    }
    return {
        "classification": "VERIFIED",
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "parameter_source": payload["param_source"],
        "historical_classifier_parameters": parameters,
        "checkpoint_path": str(R92_MODEL),
        "checkpoint_sha256": sha256_file(R92_MODEL),
        "checkpoint_classifier_parameters": checkpoint_fields,
        "full_classifier_parameters": full,
        "full_classifier_parameters_sha256": sha256_bytes(canonical_json(full)),
        "smoke_classifier_parameters": smoke,
        "allowed_smoke_overrides": {
            "n_estimators": 16,
            "early_stopping_rounds": 4,
            "device": "cuda",
        },
        "smoke_classifier_parameters_sha256": sha256_bytes(canonical_json(smoke)),
    }


def _persisted_booster_hyperparameters(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    learner = configuration.get("learner", {})
    tree = learner.get("gradient_booster", {}).get("tree_train_param", {})
    objective = learner.get("objective", {})
    train = learner.get("learner_train_param", {})
    generic = learner.get("generic_param", {})
    metrics = learner.get("metrics", [])
    gradient = learner.get("gradient_booster", {})
    return {
        "booster": train.get("booster"),
        "objective": train.get("objective"),
        "eval_metric": [
            metric.get("name") for metric in metrics if isinstance(metric, Mapping)
        ],
        "tree_method": gradient.get("gbtree_train_param", {}).get("tree_method"),
        "random_state": generic.get("random_state"),
        "n_jobs": generic.get("n_jobs"),
        "learning_rate": tree.get("learning_rate"),
        "max_depth": tree.get("max_depth"),
        "subsample": tree.get("subsample"),
        "colsample_bytree": tree.get("colsample_bytree"),
        "gamma": tree.get("gamma"),
        "min_child_weight": tree.get("min_child_weight"),
        "reg_alpha": tree.get("reg_alpha"),
        "reg_lambda": tree.get("reg_lambda"),
        "max_bin": tree.get("max_bin"),
        "scale_pos_weight": objective.get("reg_loss_param", {}).get(
            "scale_pos_weight"
        ),
    }


def booster_matches_parameter_lock(
    configuration: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    actual = _persisted_booster_hyperparameters(configuration)
    if (
        actual.get("booster") != "gbtree"
        or actual.get("objective") != expected.get("objective")
        or actual.get("eval_metric") != [expected.get("eval_metric")]
        or actual.get("tree_method") != expected.get("tree_method")
    ):
        return False
    integer_keys = ("max_depth", "max_bin", "random_state", "n_jobs")
    numeric_keys = (
        "learning_rate", "subsample", "colsample_bytree", "gamma",
        "min_child_weight", "reg_alpha", "reg_lambda", "scale_pos_weight",
    )
    try:
        return bool(
            all(int(actual[key]) == int(expected[key]) for key in integer_keys)
            and all(
                np.isclose(
                    float(actual[key]), float(expected[key]), rtol=1e-6, atol=1e-12,
                )
                for key in numeric_keys
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def manifest_rows(
    run_dir: Path,
    *,
    run_kind: str | None = None,
) -> pd.DataFrame:
    rows = []
    excluded = {"OUTPUT_MANIFEST_FINAL.tsv", "BUNDLE_MANIFEST.tsv"}
    allowed_nested_zips = {
        "provenance/preflight_snapshot/preflight_review_bundle.zip",
    }
    forbidden_suffixes = {".pyc", ".pkl", ".joblib", ".ubj", ".npy", ".npz"}
    symlinks = sorted(
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*") if path.is_symlink()
    )
    if symlinks:
        raise IntegrityError(f"Review Run contains symlinks: {symlinks}")
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(run_dir).as_posix()
        if relative in excluded or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        if path.suffix.lower() in forbidden_suffixes:
            raise IntegrityError(f"Review Run contains forbidden model/cache artifact: {relative}")
        if path.suffix.lower() == ".zip" and (
            run_kind != "CUDA_SMOKE_NON_SCIENTIFIC"
            or relative not in allowed_nested_zips
        ):
            raise IntegrityError(f"Review Run contains unrelated ZIP artifact: {relative}")
        role = relative.split("/", 1)[0] if "/" in relative else "run_status"
        rows.append({
            "relative_path": relative, "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path), "artifact_role": role,
            "include_in_review_bundle": True, "include_in_models_bundle": False,
        })
    return pd.DataFrame(rows, columns=[
        "relative_path", "size_bytes", "sha256", "artifact_role",
        "include_in_review_bundle", "include_in_models_bundle",
    ])


def validate_package_completeness(
    run_dir: Path,
    identity: Mapping[str, Any],
    *,
    amendment03_allowed_after_hashes: Mapping[str, str] | None = None,
    amendment04_reference_allowed_after_hashes: Mapping[str, str] | None = None,
    amendment06_allowed_after_hashes: Mapping[str, str] | None = None,
    amendment06_change_ledger_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the evidence contract before writing package manifests or ZIPs."""
    run_kind = str(identity.get("run_kind", ""))
    run_state = str(identity.get("state", ""))
    protocol = str(identity.get("protocol", "AMENDMENT_01"))
    amendment03_replacement = (
        identity.get("recovery_authorization") == _AMENDMENT03_AUTHORIZATION
    )
    amendment04_recovery_run = (
        identity.get("recovery_authorization") == _AMENDMENT04_AUTHORIZATION
    )
    contract = (run_kind, run_state)
    if protocol == "AMENDMENT_02" and run_kind == "AMENDED_PREFLIGHT":
        required = (
            _AMENDMENT02_PREFLIGHT_REQUIRED_FILES
            if run_state in {"PREFLIGHT_COMPLETE", "BLOCKED_PREFLIGHT_COMPLETE"}
            else None
        )
    elif protocol == "AMENDMENT_02" and run_kind == "CUDA_SMOKE_NON_SCIENTIFIC":
        if run_state == "SMOKE_COMPLETE":
            smoke_required = (
                _AMENDMENT04_SMOKE_REQUIRED_FILES
                if amendment04_recovery_run
                else _AMENDMENT03_SMOKE_REQUIRED_FILES
                if amendment03_replacement
                else _AMENDMENT02_SMOKE_REQUIRED_FILES
            )
            required = smoke_required | {
                "provenance/smoke_model_bundle_verification.json",
            }
        elif run_state == "SMOKE_INCOMPLETE":
            smoke_required = (
                _AMENDMENT04_SMOKE_REQUIRED_FILES
                if amendment04_recovery_run
                else _AMENDMENT03_SMOKE_REQUIRED_FILES
                if amendment03_replacement
                else _AMENDMENT02_SMOKE_REQUIRED_FILES
            )
            required = smoke_required | {
                "logs/smoke_failure.txt",
            }
        else:
            required = None
    else:
        required = PACKAGE_REQUIRED_FILES.get(contract)
    if required is None:
        raise IntegrityError(f"Unsupported package Run kind/state contract: {contract}")
    if identity.get("run_id") != run_dir.name:
        raise IntegrityError(
            f"Run identity/name mismatch: run_id={identity.get('run_id')!r}, "
            f"directory={run_dir.name!r}"
        )

    missing = []
    invalid = []
    for relative in sorted(required):
        path = run_dir / relative
        if not path.exists():
            missing.append(relative)
        elif not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            invalid.append(relative)

    status_fields: dict[str, str] = {}
    status_path = run_dir / "RUN_STATUS.txt"
    if status_path.is_file() and not status_path.is_symlink():
        for line in status_path.read_text().splitlines():
            key, separator, value = line.partition("=")
            if separator:
                status_fields[key.strip()] = value.strip()
    status_mismatches = []
    if status_fields.get("RUN_KIND") != run_kind:
        status_mismatches.append(
            f"RUN_KIND={status_fields.get('RUN_KIND')!r}, identity={run_kind!r}"
        )
    if status_fields.get("RUN_STATE") != run_state:
        status_mismatches.append(
            f"RUN_STATE={status_fields.get('RUN_STATE')!r}, identity={run_state!r}"
        )

    if missing or invalid or status_mismatches:
        raise IntegrityError(
            "Run completeness validation failed: "
            f"kind={run_kind!r}, state={run_state!r}, missing={missing}, "
            f"invalid={invalid}, status_mismatches={status_mismatches}"
        )
    result = {
        "run_kind": run_kind,
        "run_state": run_state,
        "required_file_count": len(required),
        "required_files_status": "PASS",
        "status_identity_match": "PASS",
    }
    if run_kind == "AMENDED_PREFLIGHT":
        result["preflight_semantic_completeness"] = (
            validate_amendment02_preflight_semantic_completeness(
                run_dir,
                identity,
                amendment03_allowed_after_hashes=(
                    amendment03_allowed_after_hashes
                ),
                amendment04_reference_allowed_after_hashes=(
                    amendment04_reference_allowed_after_hashes
                ),
            )
            if protocol == "AMENDMENT_02"
            else validate_preflight_semantic_completeness(run_dir, identity)
        )
    elif run_kind == "CUDA_SMOKE_NON_SCIENTIFIC":
        result["smoke_semantic_completeness"] = (
            validate_smoke_semantic_completeness(
                run_dir,
                identity,
                amendment06_allowed_after_hashes=(
                    amendment06_allowed_after_hashes
                ),
                amendment06_change_ledger_evidence=(
                    amendment06_change_ledger_evidence
                ),
            )
        )
    return result


def _status_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            fields[key.strip()] = value.strip()
    return fields


def _is_review_relative_regular_file(run_dir: Path, relative: Any) -> bool:
    try:
        relative_path = Path(str(relative))
    except Exception:
        return False
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or ".." in relative_path.parts
    ):
        return False
    path = run_dir / relative_path
    return path.is_file() and not path.is_symlink()


def _gate_inputs_from_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], Mapping[str, Any]]:
    stored_gates = payload.get("gates", {})
    if not isinstance(stored_gates, Mapping):
        raise IntegrityError("Evidence gate registry has no gate mapping")
    gate_inputs: dict[str, dict[str, Any]] = {}
    for specification in evidence_workflow.DEFAULT_GATE_SPECS:
        stored = stored_gates.get(specification.key)
        if not isinstance(stored, Mapping):
            raise IntegrityError(
                f"Evidence gate registry omits {specification.key}"
            )
        gate_inputs[specification.key] = {
            "classification": stored.get("classification"),
            "passed": stored.get("passed"),
            "evidence_refs": stored.get("evidence_refs"),
            "detail": stored.get("detail", ""),
        }
    if set(stored_gates) != set(gate_inputs):
        raise IntegrityError("Evidence gate registry has unknown gate keys")
    return gate_inputs, stored_gates


def _preflight_derived_payload(
    state: evidence_workflow.WorkflowState,
) -> dict[str, Any]:
    return {
        "run_kind": state.run_kind,
        "run_state": state.run_state,
        "amended_preflight_status": state.amended_preflight_status,
        "data_source_decision": state.data_source_decision,
        "split_locked": state.split_locked,
        "split_lock_status": state.split_lock_status,
        "smoke_eligible": state.smoke_eligible,
        "cuda_smoke_status": state.cuda_smoke_status,
        "ready_for_full_awaiting_external_review": (
            state.ready_for_full_awaiting_external_review
        ),
        "blocker_codes": list(state.blocker_codes),
    }


PREFLIGHT_GATE_EVIDENCE_REFS: dict[str, tuple[str, ...]] = {
    "prior_run_immutability": ("provenance/prior_immutable_verification_post.json",),
    "source_input_integrity": (
        "provenance/source_input_hashes_pre.tsv",
        "provenance/source_input_hashes_post.tsv",
    ),
    "historical_snapshot_provenance": ("provenance/historical_snapshot_validation.json",),
    "current_snapshot_analysis": ("provenance/current_snapshot_validation.json",),
    "historical_split_reproduction": (
        "provenance/historical_snapshot_validation.json",
        "splits/row_split_manifest.csv.gz",
    ),
    "historical_prediction_replay": (
        "provenance/historical_prediction_replay.csv.gz",
        "provenance/historical_prediction_replay_schema.json",
    ),
    "historical_metric_tree_parity": ("provenance/historical_snapshot_validation.json",),
    "r92_external_control": ("metrics/r92_control.json",),
    "pca_prototype_compatibility": (
        "provenance/pca_artifact_inventory.tsv",
        "provenance/historical_snapshot_validation.json",
    ),
    "review_weight_parity": ("provenance/review_weight_validation.json",),
    "augmentation_shuffle_parity": ("provenance/review_weight_validation.json",),
    "feature_dependency_closure": (
        "config/feature_manifest.csv",
        "config/variant_feature_sets.json",
    ),
    "feature_semantics": (
        "provenance/embed_sim_semantics.json",
        "provenance/gate_reconstruction_validation.csv",
        "provenance/pca_artifact_inventory.tsv",
    ),
    "gate_reconstruction": (
        "provenance/gate_formula_manifest.json",
        "provenance/gate_reconstruction_validation.csv",
        "provenance/audit_gate_upstream_coverage.csv",
    ),
    "safe_smote_isolation": ("config/variant_feature_sets.json", "logs/pytest.log"),
    "split_manifest_integrity": (
        "splits/row_split_manifest.csv.gz",
        "splits/split_hashes.json",
    ),
    "split_zero_leakage": ("splits/overlap_checks.csv",),
    "no_pca_diagnostic": (
        "provenance/raw_embedding_mapping_diagnostic.json",
        "provenance/raw_embedding_join_diagnostic.csv",
        "provenance/raw_embedding_source_inventory.tsv",
    ),
    "expanded_tests": (
        "logs/syntax_compile.log",
        "logs/pytest.log",
        "logs/reviewer_snapshot_syntax_compile.log",
        "logs/reviewer_snapshot_pytest.log",
    ),
    "cuda_capability": (
        "provenance/cuda_capability_probe.json",
        "provenance/hardware_and_software.json",
    ),
    "cuda_smoke_all_runnable": ("provenance/PREFLIGHT_REPORT.json",),
    "model_save_reload": ("provenance/PREFLIGHT_REPORT.json",),
    "output_manifest_zip_verification": ("provenance/package_staging_verification.json",),
    "no_full_scientific_training": ("provenance/execution_ledger.json",),
}

AMENDMENT02_GATE_EVIDENCE_REFS: dict[str, tuple[str, ...]] = {
    "prior_run_immutability": (
        "provenance/reference_preflight_validation.json",
    ),
    "source_input_integrity": (
        "provenance/source_input_hashes_pre.tsv",
        "provenance/source_input_hashes_post.tsv",
    ),
    "historical_snapshot_provenance": (
        "provenance/amendment02_policy_evidence.json",
    ),
    "current_snapshot_analysis": (
        "provenance/reference_preflight_validation.json",
    ),
    "historical_split_reproduction": (
        "splits/row_split_manifest.csv.gz",
        "config/locked_split_identity.json",
    ),
    "historical_prediction_replay": (
        "provenance/historical_prediction_replay.csv.gz",
        "provenance/historical_prediction_replay_schema.json",
    ),
    "historical_metric_tree_parity": (
        "provenance/amendment02_policy_evidence.json",
    ),
    "r92_external_control": ("metrics/r92_control.json",),
    "pca_prototype_compatibility": (
        "provenance/pca_artifact_inventory.tsv",
        "docs/FIXED_TABLE_LIMITATIONS.md",
    ),
    "review_weight_parity": (
        "provenance/amendment02_policy_evidence.json",
    ),
    "augmentation_shuffle_parity": (
        "provenance/amendment02_policy_evidence.json",
    ),
    "feature_dependency_closure": (
        "config/feature_manifest.csv",
        "config/variant_feature_sets.json",
    ),
    "feature_semantics": (
        "provenance/embed_sim_semantics.json",
        "provenance/gate_reconstruction_validation.csv",
        "provenance/audit_gate_upstream_coverage.csv",
    ),
    "gate_reconstruction": (
        "provenance/gate_formula_manifest.json",
        "provenance/gate_reconstruction_validation.csv",
        "provenance/audit_gate_upstream_coverage.csv",
    ),
    "safe_smote_isolation": (
        "docs/PROTOCOL_AMENDMENT_02.md",
        "logs/targeted_tests.log",
    ),
    "split_manifest_integrity": (
        "splits/row_split_manifest.csv.gz",
        "config/locked_split_identity.json",
    ),
    "split_zero_leakage": ("splits/overlap_checks.csv",),
    "no_pca_diagnostic": (
        "provenance/amendment02_policy_evidence.json",
    ),
    "expanded_tests": (
        "logs/targeted_tests.log",
        "logs/full_tests.log",
        "logs/reviewer_snapshot_syntax_compile.log",
        "logs/reviewer_snapshot_pytest.log",
    ),
    "cuda_capability": (
        "provenance/cuda_active_fit_probe.json",
        "provenance/cuda_environment_after.txt",
    ),
    "cuda_smoke_all_runnable": ("provenance/PREFLIGHT_REPORT.json",),
    "model_save_reload": ("provenance/PREFLIGHT_REPORT.json",),
    "output_manifest_zip_verification": (
        "provenance/package_staging_verification.json",
    ),
    "no_full_scientific_training": (
        "provenance/execution_ledger.json",
    ),
}


def _expected_blocker_details(
    state: evidence_workflow.WorkflowState,
) -> dict[str, dict[str, Any]]:
    return {
        blocker.code: {
            "gate_key": blocker.gate_key,
            "classification": blocker.classification.value,
            "status": blocker.status,
            "reason": blocker.reason,
            "evidence_refs": list(blocker.evidence_refs),
        }
        for blocker in state.blockers
    }


def _command_payload_matches_log(payload: Any, path: Path) -> bool:
    if not isinstance(payload, Mapping):
        return False
    try:
        expected = (
            "COMMAND=" + " ".join(map(str, payload["command"])) + "\n"
            + f"CWD={payload['cwd']}\nEXIT_CODE={int(payload['returncode'])}\n"
            + f"WALL_SECONDS={float(payload['wall_seconds']):.9f}\n"
            + "--- STDOUT ---\n" + str(payload["stdout"])
            + ("\n" if payload["stdout"] and not str(payload["stdout"]).endswith("\n") else "")
            + "--- STDERR ---\n" + str(payload["stderr"])
            + ("\n" if payload["stderr"] and not str(payload["stderr"]).endswith("\n") else "")
        )
    except (KeyError, TypeError, ValueError):
        return False
    return path.is_file() and not path.is_symlink() and path.read_text() == expected


def _command_log_exit_zero(path: Path) -> bool:
    """Validate a persisted command log without trusting a separate PASS token."""

    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        return False
    exit_lines = [
        line for line in path.read_text(errors="replace").splitlines()
        if line.startswith("EXIT_CODE=")
    ]
    return len(exit_lines) == 1 and exit_lines[0] == "EXIT_CODE=0"


def _clean_canonical_audit_frame(path: Path, expected_sha256: str) -> pd.DataFrame:
    frame = read_csv_verified(path, expected_sha256, low_memory=False)
    frame["human_label"] = canonicalize_audit_labels(frame)
    if "review_weight" in frame:
        weights = pd.to_numeric(frame.review_weight, errors="coerce").fillna(1.0)
        frame = frame[weights > 0].copy()
    keys = ["img_folder", "image", "id"]
    if "timestamp" in frame:
        frame["_timestamp"] = pd.to_numeric(frame.timestamp, errors="coerce").fillna(-1)
        frame = frame.sort_values(keys + ["_timestamp"])
    frame = frame.drop_duplicates(keys, keep="last").reset_index(drop=True)
    frame["img_folder"] = frame.img_folder.map(normalize_card_id)
    frame["card_id"] = frame.img_folder
    frame["stable_candidate_id"] = [
        "audit_" + sha256_bytes(canonical_json(list(row)))
        for row in frame[keys].astype(str).itertuples(index=False, name=None)
    ]
    return frame


def _canonical_audit_card_ids(path: Path) -> set[str]:
    frame = _clean_canonical_audit_frame(
        path, EXPECTED_HASHES.get(str(path), sha256_file(path)),
    )
    return set(frame.card_id.astype(str))


def _frames_equal_ignoring_dtype(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
) -> bool:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except (AssertionError, TypeError, ValueError):
        return False
    return True


def _numeric_evidence_frames_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    absolute_tolerance: float = 1e-15,
) -> bool:
    if list(observed.columns) != list(expected.columns) or len(observed) != len(expected):
        return False
    for column in observed:
        expected_numeric = pd.to_numeric(expected[column], errors="coerce")
        observed_numeric = pd.to_numeric(observed[column], errors="coerce")
        if expected_numeric.notna().all() and observed_numeric.notna().all():
            if not np.allclose(
                observed_numeric.to_numpy(float), expected_numeric.to_numpy(float),
                rtol=0.0, atol=absolute_tolerance, equal_nan=True,
            ):
                return False
        elif not np.array_equal(
            observed[column].fillna("").astype(str).to_numpy(),
            expected[column].fillna("").astype(str).to_numpy(),
        ):
            return False
    return True


def _float32_layer_frames_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    identity_columns: Sequence[str] = ("stable_candidate_id", "card_id"),
) -> bool:
    if list(observed.columns) != list(expected.columns) or len(observed) != len(expected):
        return False
    for column in identity_columns:
        if not np.array_equal(
            observed[column].astype(str).to_numpy(),
            expected[column].astype(str).to_numpy(),
        ):
            return False
    numeric = [column for column in expected if column not in identity_columns]
    try:
        observed_values = observed[numeric].to_numpy(dtype=np.float32)
        expected_values = expected[numeric].to_numpy(dtype=np.float32)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(observed_values).all()
        and np.array_equal(observed_values.view(np.uint32), expected_values.view(np.uint32))
    )


def _fresh_gate_formula_manifest() -> dict[str, Any]:
    builder = importlib.import_module("build_amended_preflight")
    expected_builder = (_MODULE_DIR / "build_amended_preflight.py").resolve()
    if Path(str(builder.__file__)).resolve() != expected_builder:
        raise IntegrityError(
            "Gate formula producer was imported outside the reviewed sibling code tree"
        )
    manifest = builder._gate_formula_manifest()
    if not isinstance(manifest, dict):
        raise IntegrityError("Gate formula producer did not return a mapping")
    return manifest


def _fresh_no_pca_diagnostic_validation(
    diagnostic: dict[str, Any],
    join: pd.DataFrame,
) -> dict[str, Any]:
    builder = importlib.import_module("build_amended_preflight")
    expected_builder = (_MODULE_DIR / "build_amended_preflight.py").resolve()
    if Path(str(builder.__file__)).resolve() != expected_builder:
        raise IntegrityError(
            "No-PCA validator was imported outside the reviewed sibling code tree"
        )
    result = builder._validate_no_pca_diagnostic(diagnostic, join)
    if not isinstance(result, dict):
        raise IntegrityError("No-PCA diagnostic validator did not return a mapping")
    return result


def _raw_scale_values(diagnostic: Mapping[str, Any]) -> list[float]:
    source = diagnostic.get("notebook_scale_source", {})
    explicit = source.get("raw_embedding_scales", {})
    if isinstance(explicit, Mapping) and isinstance(explicit.get("values"), list):
        return [float(value) for value in explicit["values"]]
    assignments = source.get("assignments", [])
    if isinstance(assignments, Mapping):
        values = assignments.get("raw_embedding_scales")
        if isinstance(values, list):
            return [float(value) for value in values]
    for row in assignments if isinstance(assignments, list) else []:
        if row.get("symbol") == "SCALES_TRAIN" and isinstance(row.get("values"), list):
            return [float(value) for value in row["values"]]
    return []


def _no_pca_mapping_observation(
    mapping: Mapping[str, Any],
    *,
    expected_candidate_path: Path,
    expected_raw_path: Path,
    raw_scales: Sequence[float],
    required_dimensions: int,
    source_hashes: Mapping[str, str],
) -> tuple[bool, dict[str, Any]]:
    try:
        raw = mapping["raw_cache"]
        candidate = mapping["candidate_table"]
        raw_path = Path(str(raw["path"]))
        candidate_path = Path(str(candidate["path"]))
        observed_raw = inventory_tools.inspect_raw_cache(
            raw_path, scale_sequence=raw_scales,
        )
        observed_candidate = inventory_tools.inspect_candidate_table(candidate_path)
        observed_sidecars = inventory_tools.discover_stable_key_sidecars(
            raw_path,
            search_roots=[RAW_EMBEDDINGS.parent, POST_R92_RAW.parent],
        )
        observed_mapping = inventory_tools.raw_cache_table_diagnostics(
            raw_path,
            candidate_path,
            raw_scale_sequence=raw_scales,
            expected_dimension_count=required_dimensions,
            sidecar_diagnostic=observed_sidecars,
            population="training",
            embedding_semantics_manifest_path=None,
            embedding_semantics_manifest_sha256=None,
        )
        actual = mapping["actual_stable_key_join"]
        sidecar_verified = bool(
            actual.get("actual_join_attempted")
            and actual.get("selected_sidecar_path")
            and actual.get("selected_sidecar_sha256")
        )
        semantics_verified = bool(
            mapping.get("aligned_raw_layer_reference")
            and mapping["aligned_raw_layer_reference"].get(
                "embedding_semantics_manifest_sha256"
            )
        )
        normalized_raw_scales = [normalize_scale(value) for value in raw_scales]
        expected_checks = {
            "two_dimensional": observed_raw.get("ndim") == 2,
            "expected_dtype": observed_raw.get("dtype") == "float32",
            "expected_dimension_count": (
                observed_raw.get("dimension_count") == required_dimensions
            ),
            "cache_rows_divisible_by_raw_scale_count": (
                observed_raw.get("row_count_divisible_by_scale_count") is True
            ),
            "candidate_count_equal": (
                observed_candidate.get("base_candidate_count")
                == observed_raw.get("implicit_candidate_count")
            ),
            "scale_sequence_equal": (
                normalized_raw_scales
                == observed_candidate.get("scale_sequence_first_appearance")
            ),
            "stable_table_key_nonnull": (
                observed_candidate.get("stable_key_null_rows") == 0
            ),
            "stable_table_key_unique": (
                observed_candidate.get("stable_key_duplicate_rows") == 0
            ),
            "qualifying_stable_key_sidecar": sidecar_verified,
            "actual_table_join_complete": (
                actual.get("mapped_candidate_rows")
                == observed_candidate.get("rows")
            ),
            "actual_cache_key_coverage_complete": (
                actual.get("orphan_sidecar_rows") == 0
            ),
            "actual_mapping_has_no_cache_collisions": (
                actual.get("mapped_cache_collision_rows") == 0
            ),
            "unambiguous_stable_key_mapping": (
                actual.get("unambiguous_complete_mapping") is True
            ),
            "verified_embedding_semantics_manifest": semantics_verified,
        }
        failed_checks = sorted(
            key for key, value in expected_checks.items() if not value
        )
        raw_columns = [
            f"raw_embed_{index:04d}" for index in range(required_dimensions)
        ]
        raw_columns_hash = sha256_bytes(("\n".join(raw_columns) + "\n").encode())
        expected_status = (
            "VERIFIED_EXACT_RAW_MAPPING_PREREQUISITES"
            if not failed_checks else "BLOCKED_RAW_MAPPING_PREREQUISITES"
        )
        expected_classification = "VERIFIED" if not failed_checks else "BLOCKED"
        expected_deficit = (
            observed_candidate.get("base_candidate_count")
            - observed_raw.get("implicit_candidate_count")
            if observed_raw.get("implicit_candidate_count") is not None else None
        )
        source_binding = bool(
            candidate_path.resolve() == expected_candidate_path.resolve()
            and raw_path.resolve() == expected_raw_path.resolve()
            and source_hashes.get(str(candidate_path.resolve()))
            == observed_candidate.get("sha256")
            and source_hashes.get(str(raw_path.resolve())) == observed_raw.get("sha256")
        )
        actual_consistency = bool(
            actual.get("candidate_table_rows") == observed_candidate.get("rows")
            and actual.get("candidate_table_duplicate_key_rows")
            == observed_candidate.get("stable_key_duplicate_rows")
            and int(actual.get("mapped_candidate_rows", -1))
            + int(actual.get("unmatched_candidate_rows", -1))
            == observed_candidate.get("rows")
        )
        passed = bool(
            source_binding
            and (
                mapping == observed_mapping
                if mapping.get("aligned_raw_layer_reference") is None
                else True
            )
            and mapping.get("raw_cache") == observed_raw
            and mapping.get("candidate_table") == observed_candidate
            and mapping.get("checks") == expected_checks
            and mapping.get("failed_checks") == failed_checks
            and mapping.get("candidate_deficit") == expected_deficit
            and mapping.get("raw_and_feature_scale_sequence_match")
            is expected_checks["scale_sequence_equal"]
            and mapping.get("ordered_raw_feature_columns") == raw_columns
            and mapping.get("ordered_raw_feature_columns_sha256") == raw_columns_hash
            and mapping.get("status") == expected_status
            and mapping.get("evidence_classification") == expected_classification
            and actual_consistency
        )
        return passed, {
            "candidate_rows": observed_candidate["rows"],
            "base_candidates": observed_candidate["base_candidate_count"],
            "raw_cache_rows": observed_raw["row_count"],
            "implicit_raw_candidates": observed_raw["implicit_candidate_count"],
            "candidate_deficit": expected_deficit,
            "stable_key_duplicate_rows": observed_candidate["stable_key_duplicate_rows"],
            "verified_exact_raw_embedding_rows": int(actual["mapped_candidate_rows"]),
            "exact_join_rate": actual["exact_join_rate"],
            "join_rate_classification": (
                "VERIFIED"
                if actual.get("actual_join_attempted")
                and actual.get("exact_join_rate") is not None
                else "BLOCKED"
            ),
            "join_rate_reason": (
                "Measured by actual stable-key sidecar join"
                if actual.get("actual_join_attempted")
                and actual.get("exact_join_rate") is not None
                else "No parseable stable-key sidecar permits an actual numeric join"
            ),
            "status": expected_status,
            "classification": expected_classification,
        }
    except Exception:
        return False, {}


def validate_no_pca_evidence_artifacts(
    diagnostic: Mapping[str, Any],
    join: pd.DataFrame,
    raw_source_inventory: pd.DataFrame,
    pre_sources: pd.DataFrame,
    *,
    historical_path: Path,
    current_path: Path,
    audit_frame: pd.DataFrame,
) -> dict[str, bool]:
    required_dimensions = int(diagnostic.get("required_raw_dimensions", -1))
    protocol_dimension_count = required_dimensions == 2048
    raw_scales = _raw_scale_values(diagnostic)
    source_hashes = (
        dict(zip(pre_sources.path.astype(str), pre_sources.sha256.astype(str)))
        if {"path", "sha256"} <= set(pre_sources) else {}
    )
    historical_ok, historical_join = _no_pca_mapping_observation(
        diagnostic.get("historical_training_mapping", {}),
        expected_candidate_path=historical_path,
        expected_raw_path=POST_R92_RAW,
        raw_scales=raw_scales,
        required_dimensions=required_dimensions,
        source_hashes=source_hashes,
    )
    current_ok, current_join = _no_pca_mapping_observation(
        diagnostic.get("current_training_mapping", {}),
        expected_candidate_path=current_path,
        expected_raw_path=RAW_EMBEDDINGS,
        raw_scales=raw_scales,
        required_dimensions=required_dimensions,
        source_hashes=source_hashes,
    )
    audit = diagnostic.get("audit_mapping", {})
    try:
        expected_detection_records = (
            inventory_tools.enumerate_audit_detection_artifacts(
                PROJECT_ROOT / "Shared/maskout_tile/test_set",
                audit_csv=AUDIT_CSV,
            )
        )
        expected_detection_bindings = {
            str(Path(record.path).resolve()): (
                int(record.size_bytes), str(record.sha256),
            )
            for record in expected_detection_records
        }
        detection_paths = [Path(path) for path in expected_detection_bindings]
        observed_detection_rows = raw_source_inventory.loc[
            raw_source_inventory.role.astype(str).eq(
                "card-aware audit detection source"
            )
        ] if "role" in raw_source_inventory else pd.DataFrame()
        observed_detection_bindings = {
            str(Path(str(row.path)).resolve()): (
                int(row.size_bytes), str(row.sha256),
            )
            for row in observed_detection_rows.itertuples(index=False)
        } if {"path", "size_bytes", "sha256"} <= set(
            observed_detection_rows
        ) else {}
        detection_inventory_bound = bool(
            observed_detection_bindings == expected_detection_bindings
            and all(
                source_hashes.get(path) == sha256
                for path, (_, sha256) in expected_detection_bindings.items()
            )
        )
    except Exception:
        detection_paths = []
        detection_inventory_bound = False
    try:
        observed_audit = inventory_tools.audit_detection_diagnostics(
            AUDIT_CSV, detection_paths,
        )
        observed_audit["polygon_treatment"] = {
            "classification": "VERIFIED",
            "eligible_as_exact_mask": False,
            "reason": (
                "The inference source saves a rounded perspective-transformed "
                "approxPolyDP external contour, not the full pre-transform mask "
                "used for embedding crops."
            ),
            "source_path": str(PIPELINE_SOURCE),
            "source_sha256": sha256_file(PIPELINE_SOURCE),
            "source_symbols": ["cv2.approxPolyDP", "crop_masked_patch"],
        }
        observed_audit["aligned_raw_layer_reference"] = {
            "classification": "BLOCKED",
            "status": "BLOCKED_NO_MATERIALIZED_AUDIT_RAW_EMBEDDING_LAYER",
            "reason": (
                "Crop/mask availability is diagnostic input evidence, not a "
                "persisted 2048D audit embedding layer with stable-key alignment."
            ),
        }
        audit_payload_exact = audit == observed_audit
    except Exception:
        audit_payload_exact = False
    audit_keys = ["img_folder", "image", "id"]
    audit_unique = len(audit_frame[audit_keys].drop_duplicates())
    audit_expected = {
        "candidate_rows": len(audit_frame),
        "base_candidates": audit_unique,
        "raw_cache_rows": 0,
        "implicit_raw_candidates": None,
        "candidate_deficit": len(audit_frame),
        "stable_key_duplicate_rows": int(
            audit_frame.duplicated(audit_keys, keep=False).sum()
        ),
        "verified_exact_raw_embedding_rows": (
            len(audit_frame)
            if audit.get("aligned_raw_layer_reference", {}).get("status")
            == "VERIFIED_ALIGNED_RAW_LAYER_REFERENCE" else 0
        ),
        "exact_join_rate": (
            1.0
            if audit.get("aligned_raw_layer_reference", {}).get("status")
            == "VERIFIED_ALIGNED_RAW_LAYER_REFERENCE" else None
        ),
        "join_rate_classification": (
            "VERIFIED"
            if audit.get("aligned_raw_layer_reference", {}).get("status")
            == "VERIFIED_ALIGNED_RAW_LAYER_REFERENCE" else "BLOCKED"
        ),
        "join_rate_reason": (
            "No materialized audit raw-embedding layer and stable-key alignment; "
            "crop/mask diagnostics are not a raw-layer join"
        ),
        "status": audit.get("raw_embedding_input_status"),
        "classification": audit.get("evidence_classification"),
    }
    expected_join = pd.DataFrame([
        {"population": "historical_training", **historical_join},
        {"population": "current_training", **current_join},
        {"population": "external_audit", **audit_expected},
    ])
    raw_role_pattern = re.compile(
        "raw-cache|raw cache|audit detection|embedding|training crop|ResNet50",
        flags=re.IGNORECASE,
    )
    expected_inventory = (
        pre_sources[
            pre_sources.role.astype(str).map(
                lambda value: bool(raw_role_pattern.search(value))
            )
        ].reset_index(drop=True)
        if "role" in pre_sources else pd.DataFrame()
    )
    raw_columns = [
        f"raw_embed_{index:04d}" for index in range(required_dimensions)
    ] if required_dimensions > 0 else []
    audit_bound = bool(
        audit_payload_exact
        and detection_inventory_bound
        and diagnostic.get("notebook_scale_source")
        == inventory_tools.extract_notebook_scale_evidence(NOTEBOOK)
        and audit.get("audit_rows") == len(audit_frame)
        and audit.get("audit_unique_card_aware_keys") == audit_unique
        and audit.get("audit_duplicate_key_rows")
        == int(audit_frame.duplicated(audit_keys, keep=False).sum())
        and audit.get("audit_keys_matched_card_aware")
        + audit.get("audit_keys_unmatched_card_aware") == len(audit_frame)
        and all(
            isinstance(audit.get(key), int) and audit.get(key) >= 0
            for key in (
                "detection_file_count_requested", "detection_file_count_read",
                "detection_rows", "detection_unique_card_aware_keys",
                "audit_keys_matched_card_aware", "audit_keys_unmatched_card_aware",
                "audit_duplicate_key_groups", "cross_file_collision_key_groups",
                "conflicting_mask_payload_key_groups",
                "audit_keys_with_parseable_rle",
                "audit_keys_with_existing_explicit_mask",
                "audit_keys_with_reconstructible_crop_mask_input",
            )
        )
        and audit.get("audit_keys_with_reconstructible_crop_mask_input")
        <= len(audit_frame)
        and audit.get("polygon_treatment", {}).get("eligible_as_exact_mask") is False
    )
    exact_training = bool(
        diagnostic.get("historical_training_mapping", {}).get(
            "aligned_raw_layer_reference"
        )
    )
    exact_audit = bool(
        audit.get("aligned_raw_layer_reference", {}).get("status")
        == "VERIFIED_ALIGNED_RAW_LAYER_REFERENCE"
    )
    expected_status = (
        "VERIFIED_EXACT_RAW_EMBEDDINGS_AVAILABLE"
        if exact_training and exact_audit
        else "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE"
    )
    historical_layer = diagnostic.get("historical_training_mapping", {}).get(
        "aligned_raw_layer_reference"
    ) or {}
    current_layer = diagnostic.get("current_training_mapping", {}).get(
        "aligned_raw_layer_reference"
    ) or {}
    expected_prerequisites = {
        "historical_training": bool(
            historical_layer.get("status")
            == "VERIFIED_ALIGNED_RAW_LAYER_REFERENCE"
        ),
        "external_audit": exact_audit,
        "current_training_comparison_only": bool(
            current_layer.get("status")
            == "VERIFIED_ALIGNED_RAW_LAYER_REFERENCE"
        ),
        "same_encoder_preprocessing_semantics": False,
        "all_training_validation_audit_layers_materialized": False,
    }
    encoder = diagnostic.get("encoder", {})
    available_weights = encoder.get("available_weights_artifact", {})
    historical_encoder = encoder.get("historical_encoder_identity", {})
    encoder_path = Path(str(available_weights.get("path", "")))
    source_path = Path(str(encoder.get("source_path", "")))
    expected_encoder = {
        "available_weights_artifact": {
            "classification": "VERIFIED",
            "path": str(ENCODER_WEIGHTS),
            "sha256": sha256_file(ENCODER_WEIGHTS),
        },
        "historical_encoder_identity": {
            "classification": "UNRESOLVED",
            "status": "UNRESOLVED_NO_PERSISTED_CACHE_TO_WEIGHT_LINKAGE",
            "persisted_linkage_found": False,
            "reason": (
                "The source permits an environment-selected ResNet backbone and "
                "a weights fallback; no raw-cache sidecar records the exact selected "
                "backbone/weight artifact."
            ),
        },
        "source_supported_default": (
            "torchvision ResNet50 IMAGENET1K_V2 with DEFAULT fallback, "
            "fc=Identity, L2-normalized"
        ),
        "preprocessing": (
            "RGB, Resize(224), CenterCrop(224), ImageNet normalization, "
            "L2 normalization"
        ),
        "source_path": str(GATE_SOURCE),
        "source_sha256": sha256_file(GATE_SOURCE),
        "source_symbols": [
            "JASSID_EMBED_BACKBONE", "build_embed_model", "embed_patch",
        ],
    }
    encoder_bound = bool(
        encoder == expected_encoder
        and encoder_path.is_file() and not encoder_path.is_symlink()
        and source_hashes.get(str(encoder_path.resolve()))
        == available_weights.get("sha256")
        and source_path.is_file() and not source_path.is_symlink()
        and source_hashes.get(str(source_path.resolve()))
        == encoder.get("source_sha256")
    )
    crop = diagnostic.get("training_crop_mask_availability", {})
    tiled_coco_path = HISTORICAL_TILED_COCO.resolve(strict=False)
    try:
        historical_keys = pd.read_csv(
            historical_path, usecols=["file_name", "ann_id"], low_memory=False,
        ).drop_duplicates().copy()
        historical_keys["file_name"] = historical_keys.file_name.astype(str)
        historical_keys["ann_id"] = historical_keys.ann_id.astype(str)
        tiled_payload = json.loads(tiled_coco_path.read_text())
        image_names = {
            int(row["id"]): str(row["file_name"])
            for row in tiled_payload.get("images", [])
        }
        annotation_rows = pd.DataFrame([
            {
                "file_name": image_names.get(int(row.get("image_id", -1)), ""),
                "ann_id": str(row.get("id", "")),
                "segmentation_available": bool(row.get("segmentation")),
            }
            for row in tiled_payload.get("annotations", [])
        ], columns=["file_name", "ann_id", "segmentation_available"])
        training_assets = historical_keys.merge(
            annotation_rows, on=["file_name", "ann_id"], how="left",
        )
        referenced_tiles = sorted(historical_keys.file_name.unique())
        tile_root = TRAINING_TILE_ROOT.resolve(strict=False)
        expected_crop = {
            "classification": "VERIFIED",
            "tiled_coco_path": str(tiled_coco_path),
            "tiled_coco_sha256": sha256_file(tiled_coco_path),
            "feature_base_candidate_count": len(historical_keys),
            "feature_base_candidates_joining_coco_annotation": int(
                training_assets.segmentation_available.notna().sum()
            ),
            "feature_base_candidates_with_segmentation": int(
                training_assets.segmentation_available.eq(True).sum()
            ),
            "feature_base_candidates_without_exact_coco_join": int(
                training_assets.segmentation_available.isna().sum()
            ),
            "referenced_tile_image_count": len(referenced_tiles),
            "referenced_tile_images_existing": sum(
                (tile_root / name).is_file() for name in referenced_tiles
            ),
            "tile_root": str(tile_root),
            "tile_content_hash_inventory": (
                "NOT_CAPTURED; availability claim is path-existence only"
            ),
        }
    except Exception:
        expected_crop = {}
    crop_bound = bool(
        crop == expected_crop
        and Path(str(crop.get("tiled_coco_path", ""))).resolve(strict=False)
        == tiled_coco_path
        and tiled_coco_path.is_file() and not tiled_coco_path.is_symlink()
        and crop.get("tiled_coco_sha256") == sha256_file(tiled_coco_path)
        == source_hashes.get(str(tiled_coco_path.resolve()))
    )
    return {
        "required_raw_dimension_is_protocol_2048": protocol_dimension_count,
        "mapping_payloads_recomputed": historical_ok and current_ok,
        "join_frame_rederived_exactly": _numeric_evidence_frames_equal(
            join, expected_join,
        ),
        "raw_source_inventory_exact_and_pinned": bool(
            not expected_inventory.empty
            and _frames_equal_ignoring_dtype(
                raw_source_inventory, expected_inventory,
            )
            and all(
                source_hashes.get(str(row.path)) == str(row.sha256)
                for row in raw_source_inventory.itertuples(index=False)
            )
        ),
        "audit_mapping_bound": audit_bound,
        "encoder_and_training_crop_sources_pinned": encoder_bound and crop_bound,
        "exact_mapping_prerequisites_derived": (
            diagnostic.get("exact_mapping_prerequisites") == expected_prerequisites
        ),
        "ordered_raw_columns_canonical": bool(
            diagnostic.get("proposed_unmapped_raw_feature_columns") == raw_columns
            and diagnostic.get("proposed_unmapped_raw_feature_columns_sha256")
            == sha256_bytes(("\n".join(raw_columns) + "\n").encode())
        ),
        "top_level_status_derived": bool(
            diagnostic.get("status") == expected_status
            and diagnostic.get("classification")
            == ("VERIFIED" if exact_training and exact_audit else "BLOCKED")
        ),
    }


def _decode_float32_hex_bits(values: Iterable[Any]) -> np.ndarray:
    encoded = np.fromiter(
        (int(str(value), 16) for value in values), dtype=np.uint32,
    )
    return encoded.view(np.float32)


def _historical_replay_reopen_frame(path: Path) -> pd.DataFrame:
    string_columns = {
        "historical_saved_probability_literal": "string",
        "historical_saved_probability_float64_hex": "string",
        "historical_saved_probability_float32_bits_hex": "string",
        "replayed_probability_float32_bits_hex": "string",
        "absolute_difference_direct_decimal_float64_hex": "string",
    }
    return pd.read_csv(
        path, dtype=string_columns, keep_default_na=False,
        float_precision="round_trip",
    )


def _historical_prediction_source_matches_replay(
    source: pd.DataFrame,
    replay: pd.DataFrame,
) -> bool:
    """Bind replay rows and preserved probability text to the reviewed source."""

    try:
        source_float32_bits = np.fromiter(
            (float(value) for value in source.proba.astype(str)),
            dtype=np.float32,
            count=len(source),
        ).view(np.uint32)
        replay_float32_bits = np.fromiter(
            (
                int(str(value), 16)
                for value in replay.historical_saved_probability_float32_bits_hex
            ),
            dtype=np.uint32,
            count=len(replay),
        )
        return bool(
            len(source) == len(replay)
            and np.array_equal(
                source.file_name.astype(str), replay.file_name.astype(str),
            )
            and np.array_equal(
                source.ann_id.astype(str), replay.ann_id.astype(str),
            )
            and np.array_equal(
                source.scale.map(normalize_scale).astype(str),
                replay.scale.map(normalize_scale).astype(str),
            )
            and np.array_equal(
                pd.to_numeric(source.label, errors="raise").to_numpy(np.int8),
                pd.to_numeric(replay.label, errors="raise").to_numpy(np.int8),
            )
            and np.array_equal(
                source.proba.astype(str),
                replay.historical_saved_probability_literal.astype(str),
            )
            and np.array_equal(source_float32_bits, replay_float32_bits)
        )
    except Exception:
        return False


def _historical_replay_reopen_summary(path: Path) -> tuple[dict[str, Any], bool]:
    frame = _historical_replay_reopen_frame(path)
    saved_decimal = np.fromiter(
        (float(value) for value in frame.historical_saved_probability_literal),
        dtype=np.float64, count=len(frame),
    )
    saved_decimal_hex = np.fromiter(
        (float.fromhex(str(value)) for value in frame.historical_saved_probability_float64_hex),
        dtype=np.float64, count=len(frame),
    )
    saved_float32 = _decode_float32_hex_bits(
        frame.historical_saved_probability_float32_bits_hex,
    )
    replayed_float32 = _decode_float32_hex_bits(
        frame.replayed_probability_float32_bits_hex,
    )
    direct_error = np.abs(saved_decimal - replayed_float32.astype(np.float64))
    source_dtype_error = np.abs(saved_float32 - replayed_float32)
    stored_error = pd.to_numeric(
        frame.absolute_difference_direct_decimal, errors="raise",
    ).to_numpy(np.float64)
    stored_error_hex = np.fromiter(
        (
            float.fromhex(str(value))
            for value in frame.absolute_difference_direct_decimal_float64_hex
        ),
        dtype=np.float64, count=len(frame),
    )
    checks_pass = bool(
        np.array_equal(saved_decimal.view(np.uint64), saved_decimal_hex.view(np.uint64))
        and np.array_equal(stored_error.view(np.uint64), direct_error.view(np.uint64))
        and np.array_equal(stored_error_hex.view(np.uint64), direct_error.view(np.uint64))
        and np.array_equal(
            pd.to_numeric(
                frame.historical_saved_probability_float32, errors="raise",
            ).to_numpy(np.float32).view(np.uint32),
            saved_float32.view(np.uint32),
        )
        and np.array_equal(
            pd.to_numeric(
                frame.replayed_probability_float32, errors="raise",
            ).to_numpy(np.float32).view(np.uint32),
            replayed_float32.view(np.uint32),
        )
    )
    summary = {
        "row_count": len(frame),
        "direct_decimal_max_absolute_difference": float(
            direct_error.max(initial=0.0)
        ),
        "direct_decimal_mean_absolute_difference": float(
            direct_error.mean() if len(frame) else 0.0
        ),
        "direct_decimal_mismatch_count_gt_1e_12": int(
            (direct_error > 1e-12).sum()
        ),
        "source_dtype_float32_max_absolute_difference": float(
            source_dtype_error.max(initial=np.float32(0.0))
        ),
        "source_dtype_float32_mismatch_count_gt_1e_12": int(
            (source_dtype_error > 1e-12).sum()
        ),
        "saved_vs_replayed_float32_bit_mismatch_count": int(
            (
                saved_float32.view(np.uint32)
                != replayed_float32.view(np.uint32)
            ).sum()
        ),
    }
    return summary, checks_pass


def _load_gate_registry_state(
    payload: Mapping[str, Any],
    *,
    protocol: str,
    phase: str,
    no_pca_status: str,
    no_pca_runnable: bool,
    ledger: Mapping[str, Any],
) -> tuple[evidence_workflow.GateRegistry, evidence_workflow.WorkflowState]:
    gate_inputs, stored_gates = _gate_inputs_from_payload(payload)
    registry = evidence_workflow.build_gate_registry(gate_inputs)
    if any(
        stored_gates[key].get("accepted") is not gate.accepted
        or stored_gates[key].get("status") != gate.status_token
        for key, gate in registry.evidence.items()
    ):
        raise IntegrityError("Evidence gate registry stores inconsistent tokens")
    facts = evidence_workflow.WorkflowFacts(
        phase=phase,
        official_smoke_variants_completed=int(
            ledger.get("official_smoke_variants_completed", 0)
        ),
        exploratory_smoke_variants_completed=int(
            ledger.get("exploratory_smoke_variants_completed", 0)
        ),
        full_scientific_run_executed=bool(
            ledger.get("full_scientific_run_executed", False)
        ),
        full_variants_completed=int(ledger.get("full_variants_completed", 0)),
        no_pca_variant_status=no_pca_status,
        no_pca_variant_runnable=no_pca_runnable,
        protocol_amendment=protocol,
    )
    return registry, evidence_workflow.derive_workflow_state(registry, facts)


def validate_amendment02_preflight_semantic_completeness(
    run_dir: Path,
    identity: Mapping[str, Any],
    *,
    amendment03_allowed_after_hashes: Mapping[str, str] | None = None,
    amendment04_reference_allowed_after_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Recompute the bounded Amendment 02 authorization contract."""

    try:
        report = json.loads(
            (run_dir / "provenance/PREFLIGHT_REPORT.json").read_text()
        )
        policy = json.loads(
            (run_dir / "provenance/amendment02_policy_evidence.json").read_text()
        )
        reference = json.loads(
            (run_dir / "provenance/reference_preflight_validation.json").read_text()
        )
        registry_payload = json.loads(
            (run_dir / "provenance/evidence_gate_registry.json").read_text()
        )
        ledger = json.loads(
            (run_dir / "provenance/execution_ledger.json").read_text()
        )
        locked_split = json.loads(
            (run_dir / "config/locked_split_identity.json").read_text()
        )
        probe = json.loads(
            (run_dir / "provenance/cuda_active_fit_probe.json").read_text()
        )
        cuda_environment_before = (
            run_dir / "provenance/cuda_environment_before.txt"
        ).read_text()
        cuda_environment_after = (
            run_dir / "provenance/cuda_environment_after.txt"
        ).read_text()
        variants = json.loads(
            (run_dir / "config/variant_feature_sets.json").read_text()
        )
        split = pd.read_csv(run_dir / "splits/row_split_manifest.csv.gz")
        overlap = pd.read_csv(run_dir / "splits/overlap_checks.csv")
        replay_path = run_dir / "provenance/historical_prediction_replay.csv.gz"
        replay_schema = json.loads(
            (run_dir / "provenance/historical_prediction_replay_schema.json").read_text()
        )
        r92 = json.loads((run_dir / "metrics/r92_control.json").read_text())
        pre_sources = pd.read_csv(
            run_dir / "provenance/source_input_hashes_pre.tsv", sep="\t",
        )
        post_sources = pd.read_csv(
            run_dir / "provenance/source_input_hashes_post.tsv", sep="\t",
        )
        status_path = run_dir / "RUN_STATUS.txt"
        status = _status_fields(status_path)
    except Exception as exc:
        raise IntegrityError(
            f"Amendment 02 Preflight evidence is unreadable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    checks: dict[str, bool] = {}
    no_pca_status = str(policy.get("true_no_pca_status", ""))
    no_pca_runnable = variants.get("no_pca", {}).get("status") == "RUNNABLE"
    registry, state = _load_gate_registry_state(
        registry_payload,
        protocol="AMENDMENT_02",
        phase="AMENDED_PREFLIGHT",
        no_pca_status=no_pca_status,
        no_pca_runnable=no_pca_runnable,
        ledger=ledger,
    )
    expected_refs = AMENDMENT02_GATE_EVIDENCE_REFS
    checks["registry_schema_and_evidence_refs"] = bool(
        set(expected_refs) == set(registry.evidence)
        and all(
            tuple(registry.evidence[key].evidence_refs) == tuple(expected_refs[key])
            and all(_is_review_relative_regular_file(run_dir, relative) for relative in refs)
            for key, refs in expected_refs.items()
            if not (
                key == "output_manifest_zip_verification"
                and not registry.accepted(key)
            )
        )
        and registry_payload.get("derived_workflow")
        == _preflight_derived_payload(state)
    )

    expected_prompt_sha = EXPECTED_HASHES[str(AMENDMENT02_TASK_SPEC)]
    expected_reference_sha = EXPECTED_HASHES[str(AMENDMENT02_REFERENCE_ZIP)]
    checks["identity_protocol_and_reference"] = bool(
        identity.get("protocol") == "AMENDMENT_02"
        and identity.get("run_id") == run_dir.name
        and identity.get("run_kind") == state.run_kind
        and identity.get("state") == state.run_state
        and identity.get("task_spec_sha256") == expected_prompt_sha
        and identity.get("reference_run_id") == AMENDMENT02_REFERENCE_RUN.name
        and identity.get("reference_review_zip_sha256") == expected_reference_sha
        and identity.get("full_authorized") is False
        and identity.get("stability_authorized") is False
        and identity.get("base_seed") == BASE_SEED
        and identity.get("candidate_split_hash")
        == "f2cd1570d2c76af49ed52593d95657a3cbbe14d8f2f077edd05447e608698c64"
    )
    live_code_paths = {
        path.relative_to(STUDY_ROOT).as_posix()
        for path in [
            *sorted((STUDY_ROOT / "code").glob("*.py")),
            *sorted((STUDY_ROOT / "code").glob("*.sh")),
            *sorted((STUDY_ROOT / "tests").glob("*.py")),
        ]
        if path.is_file()
    }
    code_tree = identity.get("code_tree", {})
    sealed_code_tree_bound = isinstance(code_tree, Mapping) and bool(code_tree)
    if sealed_code_tree_bound:
        try:
            sealed_code_tree_bound = all(
                _valid_sha256(expected)
                and (run_dir / relative).is_file()
                and not (run_dir / relative).is_symlink()
                and sha256_file(run_dir / relative) == expected
                and (
                    run_dir
                    / "provenance"
                    / ("tests_snapshot" if relative.startswith("tests/") else "code_snapshot")
                    / Path(relative).name
                ).is_file()
                and sha256_file(
                    run_dir
                    / "provenance"
                    / ("tests_snapshot" if relative.startswith("tests/") else "code_snapshot")
                    / Path(relative).name
                ) == expected
                for relative, expected in code_tree.items()
            )
        except Exception:
            sealed_code_tree_bound = False
    if (
        amendment03_allowed_after_hashes is None
        and amendment04_reference_allowed_after_hashes is None
    ):
        try:
            live_code_tree_bound = bool(
                isinstance(code_tree, Mapping)
                and set(code_tree) == live_code_paths
                and all(
                    (STUDY_ROOT / relative).is_file()
                    and not (STUDY_ROOT / relative).is_symlink()
                    and sha256_file(STUDY_ROOT / relative) == expected
                    for relative, expected in code_tree.items()
                )
            )
        except Exception:
            live_code_tree_bound = False
    elif amendment03_allowed_after_hashes is not None:
        try:
            live_code_tree_bound = bool(
                isinstance(code_tree, Mapping)
                and dict(amendment03_allowed_after_hashes)
                == amendment03_code_drift_allowlist(code_tree)
            )
        except Exception:
            live_code_tree_bound = False
    else:
        try:
            live_code_tree_bound = bool(
                isinstance(code_tree, Mapping)
                and dict(amendment04_reference_allowed_after_hashes or {})
                == amendment04_recovery.reference_code_drift_allowlist(
                    code_tree, study_root=STUDY_ROOT,
                )
            )
        except Exception:
            live_code_tree_bound = False
    task_copy = run_dir / "docs/PROTOCOL_AMENDMENT_02_TASK_SPEC.md"
    protocol_copy = run_dir / "docs/PROTOCOL_AMENDMENT_02.md"
    protocol_source = STUDY_ROOT / "docs/PROTOCOL_AMENDMENT_02.md"
    checks["code_tree_snapshots_and_protocol_docs_bound"] = bool(
        sealed_code_tree_bound
        and live_code_tree_bound
        and sha256_file(AMENDMENT02_TASK_SPEC) == expected_prompt_sha
        and task_copy.is_file()
        and sha256_file(task_copy) == expected_prompt_sha
        and protocol_source.is_file()
        and protocol_copy.is_file()
        and sha256_file(protocol_copy) == sha256_file(protocol_source)
        and identity.get("protocol_document_sha256")
        == sha256_file(protocol_source)
    )
    checks["reference_preflight_and_zip_pinned"] = bool(
        reference.get("status") == "PASS"
        and reference.get("reference_run_id") == AMENDMENT02_REFERENCE_RUN.name
        and reference.get("reference_run_path") == str(AMENDMENT02_REFERENCE_RUN)
        and reference.get("reference_review_zip_path")
        == str(AMENDMENT02_REFERENCE_ZIP)
        and reference.get("reference_review_zip_sha256") == expected_reference_sha
        and AMENDMENT02_REFERENCE_ZIP.is_file()
        and not AMENDMENT02_REFERENCE_ZIP.is_symlink()
        and sha256_file(AMENDMENT02_REFERENCE_ZIP) == expected_reference_sha
        and verify_zip(
            AMENDMENT02_REFERENCE_ZIP, embedded_run_manifest=True,
        ).get("status") == "PASS"
    )

    source_columns = ["path", "size_bytes", "sha256"]
    source_frames_valid = bool(
        source_columns == list(pre_sources.columns)
        == list(post_sources.columns)
        and pre_sources.equals(post_sources)
        and not pre_sources.path.astype(str).duplicated().any()
    )
    source_live_valid = False
    if source_frames_valid:
        try:
            source_live_valid = all(
                Path(str(row.path)).is_file()
                and not Path(str(row.path)).is_symlink()
                and Path(str(row.path)).stat().st_size == int(row.size_bytes)
                and sha256_file(Path(str(row.path))) == str(row.sha256)
                for row in post_sources.itertuples(index=False)
            )
        except Exception:
            source_live_valid = False
    checks["source_inputs_unchanged"] = bool(
        source_frames_valid
        and source_live_valid
        and registry.accepted("source_input_integrity")
    )

    replay_summary, replay_internal = _historical_replay_reopen_summary(replay_path)
    source_prediction_sha = str(replay_schema.get("source_prediction_sha256", ""))
    try:
        source_prediction = load_historical_predictions_verified(
            R92_PREDICTIONS, source_prediction_sha,
        )
        replay_frame = _historical_replay_reopen_frame(replay_path)
        replay_source_match = _historical_prediction_source_matches_replay(
            source_prediction, replay_frame,
        )
    except Exception:
        replay_source_match = False
    replay_pass = bool(
        replay_internal
        and replay_source_match
        and replay_summary == {
            "row_count": 56843,
            "direct_decimal_max_absolute_difference": 2.9790496847148518e-08,
            "direct_decimal_mean_absolute_difference": 2.011113556158508e-09,
            "direct_decimal_mismatch_count_gt_1e_12": 14201,
            "source_dtype_float32_max_absolute_difference": 0.0,
            "source_dtype_float32_mismatch_count_gt_1e_12": 0,
            "saved_vs_replayed_float32_bit_mismatch_count": 0,
        }
        and replay_schema.get("source_prediction_path") == str(R92_PREDICTIONS)
        and replay_schema.get("source_prediction_sha256")
        == sha256_file(R92_PREDICTIONS)
    )
    checks["historical_float32_replay_exact"] = bool(
        replay_pass
        and registry.accepted("historical_prediction_replay")
        and policy.get("historical_float32_replay_status") == "PASS"
    )

    split_hash = split_assignment_hash(
        split.stable_candidate_id, split.split, split.label, split.eligible,
        split.split_order,
    )
    split_counts = {
        "training": {
            "rows": int((split.split == "training").sum()),
            "positive": int(split.loc[split.split == "training", "label"].sum()),
            "cards": int(split.loc[split.split == "training", "card_id"].nunique()),
        },
        "validation": {
            "rows": int((split.split == "validation").sum()),
            "positive": int(split.loc[split.split == "validation", "label"].sum()),
            "cards": int(split.loc[split.split == "validation", "card_id"].nunique()),
        },
        "validation_ineligible_border_rows": int(
            (split.split == "historical_heldout_ineligible_border").sum()
        ),
    }
    split_expected = {
        "training": {"rows": 291024, "positive": 75036, "cards": 66},
        "validation": {"rows": 56843, "positive": 15952, "cards": 18},
        "validation_ineligible_border_rows": 15696,
    }
    split_pass = bool(
        split_hash == "f2cd1570d2c76af49ed52593d95657a3cbbe14d8f2f077edd05447e608698c64"
        and split_counts == split_expected
        and len(overlap) == 2
        and (pd.to_numeric(overlap.overlap_count, errors="raise") == 0).all()
        and (overlap.status.astype(str) == "PASS").all()
        and locked_split.get("status") == "PASS"
        and locked_split.get("split_lock_status") == "LOCKED"
        and locked_split.get("locked_for_scientific_use") is True
        and locked_split.get("split_assignment_sha256") == split_hash
        and locked_split.get("base_seed") == BASE_SEED
        and locked_split.get("counts") == split_counts
    )
    checks["split_and_seed_locked_exactly"] = bool(
        split_pass
        and registry.accepted("split_manifest_integrity")
        and registry.accepted("split_zero_leakage")
        and state.split_locked
    )

    policy_checks = policy.get("checks", {})
    checks["authorized_policy_exact"] = bool(
        policy.get("classification") == "VERIFIED"
        and policy.get("data_source_decision")
        == "ACCEPTED_HISTORICAL_FIXED_FEATURE_TABLE_WITH_LIMITATION"
        and policy.get("primary_table_path") == str(HISTORICAL_TRAINING_CSV)
        and policy.get("primary_table_sha256")
        == EXPECTED_HASHES[str(HISTORICAL_TRAINING_CSV)]
        and policy.get("primary_table_rows") == 363563
        and policy.get("pca_lineage_policy")
        == "UNRESOLVED_ACCEPTED_AS_FIXED_TABLE_LIMITATION"
        and policy.get("feature_semantics_status") == "PASS"
        and policy.get("true_no_pca_status")
        == "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE"
        and isinstance(policy_checks, Mapping)
        and set(policy_checks) == {
            "primary_fixed_table_bound",
            "historical_float32_replay_bit_exact",
            "pca_limitation_disclosed_without_compatibility_claim",
            "feature_semantics_verified_independently",
            "true_no_pca_variant_remains_blocked",
            "paired_resampling_tests_passed",
        }
        and all(value is True for value in policy_checks.values())
        and registry.evidence["pca_prototype_compatibility"].classification.value
        == "UNRESOLVED"
        and not registry.evidence["pca_prototype_compatibility"].accepted
        and registry.accepted("feature_semantics")
        and registry.accepted("feature_dependency_closure")
        and registry.accepted("gate_reconstruction")
        and registry.accepted("safe_smote_isolation")
    )

    build_info = probe.get("build_info", {})
    environment_path = str(probe.get("environment_path", ""))
    visible_device_configuration = probe.get("visible_device_configuration", {})
    visibility_evidence_bound = bool(
        isinstance(visible_device_configuration, Mapping)
        and set(visible_device_configuration) == {
            "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
        }
        and all(
            isinstance(value, str) and bool(value)
            for value in visible_device_configuration.values()
        )
        and all(
            f"{key}={value}" in cuda_environment_before
            and f"{key}={value}" in cuda_environment_after
            for key, value in visible_device_configuration.items()
        )
    )
    checks["isolated_cuda_probe_active_pass"] = bool(
        probe.get("status") == "PASS"
        and probe.get("classification") == "VERIFIED"
        and Path(environment_path).is_absolute()
        and Path(environment_path).resolve()
        != Path("/REVIEWER_INPUT_ROOT/SAM").resolve()
        and identity.get("cuda_environment_path") == environment_path
        and report.get("cuda_capability", {}).get("environment_path")
        == environment_path
        and probe.get("xgboost_version") == "2.1.1"
        and build_info.get("USE_CUDA") is True
        and probe.get("requested_device") == "cuda"
        and probe.get("tree_method") == "hist"
        and probe.get("training_status") == "PASS"
        and probe.get("save_reload_predict_status") == "PASS"
        and probe.get("cpu_fallback_detected") is False
        and visibility_evidence_bound
        and registry.accepted("cuda_capability")
    )
    checks["tests_and_no_forbidden_execution"] = bool(
        registry.accepted("expanded_tests")
        and registry.accepted("no_full_scientific_training")
        and ledger.get("status") == "PASS"
        and ledger.get("official_smoke_variants_completed") == 0
        and ledger.get("exploratory_smoke_variants_completed") == 0
        and ledger.get("full_scientific_run_executed") is False
        and ledger.get("full_variants_completed") == 0
        and all(
            _command_log_exit_zero(run_dir / relative)
            for relative in (
                "logs/targeted_tests.log",
                "logs/full_tests.log",
                "logs/reviewer_snapshot_syntax_compile.log",
                "logs/reviewer_snapshot_pytest.log",
            )
        )
    )
    checks["r92_control_preserved"] = bool(
        r92.get("status") == "PASS"
        and r92.get("n") == 10097
        and float(r92.get("max_probability_difference", math.inf)) <= 1e-12
        and registry.accepted("r92_external_control")
    )
    expected_report = bool(
        report.get("protocol") == "AMENDMENT_02"
        and report.get("run_kind") == state.run_kind
        and report.get("status") == state.amended_preflight_status
        and report.get("data_source_decision") == state.data_source_decision
        and report.get("training_table_path") == str(HISTORICAL_TRAINING_CSV)
        and report.get("training_table_sha256")
        == EXPECTED_HASHES[str(HISTORICAL_TRAINING_CSV)]
        and report.get("training_table_rows") == 363563
        and report.get("pca_lineage_policy")
        == "UNRESOLVED_ACCEPTED_AS_FIXED_TABLE_LIMITATION"
        and report.get("feature_semantics_status") == "PASS"
        and report.get("true_no_pca_status") == no_pca_status
        and report.get("candidate_split_hash") == split_hash
        and report.get("base_seed") == BASE_SEED
        and report.get("split_lock_status") == state.split_lock_status
        and report.get("smoke_eligible") is state.smoke_eligible
        and report.get("full_authorized") is False
        and report.get("global_blockers") == _expected_blocker_details(state)
        and report.get("r92_control") == r92
    )
    checks["report_and_status_match_derived_state"] = bool(
        expected_report
        and status.get("PROTOCOL_AMENDMENT") == "AMENDMENT_02"
        and status.get("RUN_KIND") == state.run_kind
        and status.get("RUN_STATE") == state.run_state
        and status.get("AMENDED_PREFLIGHT_STATUS")
        == state.amended_preflight_status
        and status.get("DATA_SOURCE_DECISION") == state.data_source_decision
        and status.get("SPLIT_LOCK_STATUS") == state.split_lock_status
        and status.get("CUDA_SMOKE_ELIGIBLE")
        == ("YES" if state.smoke_eligible else "NO")
        and status.get("FULL_AUTHORIZED") == "NO"
        and status.get("PCA_LINEAGE_POLICY")
        == "UNRESOLVED_ACCEPTED_AS_FIXED_TABLE_LIMITATION"
        and status.get("BLOCKERS")
        == (";".join(state.blocker_codes) if state.blocker_codes else "NONE")
    )

    staging_path = run_dir / "provenance/package_staging_verification.json"
    if registry.accepted("output_manifest_zip_verification"):
        try:
            staging = json.loads(staging_path.read_text())
            nested = staging.get("package_completeness", {}).get(
                "preflight_semantic_completeness", {},
            )
            staging_bound = bool(
                staging.get("crc_status") == "PASS"
                and staging.get("independent_reopen_member_verification") == "PASS"
                and not staging.get("verification_failures")
                and nested.get("status") == "PASS"
                and nested.get("phase") == "STAGING"
                and _valid_sha256(staging.get("bundle_sha256"))
                and int(staging.get("bundle_size_bytes", 0)) > 0
                and int(staging.get("member_count", 0)) > 0
            )
        except Exception:
            staging_bound = False
    else:
        staging_bound = bool(
            registry.evidence["output_manifest_zip_verification"].passed is None
            and not staging_path.exists()
            and not state.smoke_eligible
        )
    checks["package_gate_bound"] = staging_bound

    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise IntegrityError(
            f"Amendment 02 Preflight semantic completeness failed: {failed}"
        )
    return {
        "status": "PASS",
        "checks": checks,
        "protocol": "AMENDMENT_02",
        "run_state": state.run_state,
        "smoke_eligible": state.smoke_eligible,
        "split_locked": state.split_locked,
        "blocker_codes": list(state.blocker_codes),
        "phase": "FINAL" if staging_path.is_file() else "STAGING",
    }


def validate_preflight_semantic_completeness(
    run_dir: Path,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the Preflight evidence contract before either package pass."""

    try:
        report = json.loads(
            (run_dir / "provenance/PREFLIGHT_REPORT.json").read_text()
        )
        registry_payload = json.loads(
            (run_dir / "provenance/evidence_gate_registry.json").read_text()
        )
        ledger = json.loads(
            (run_dir / "provenance/execution_ledger.json").read_text()
        )
        variants = json.loads(
            (run_dir / "config/variant_feature_sets.json").read_text()
        )
        dependency_graph = json.loads(
            (run_dir / "config/feature_dependency_graph.json").read_text()
        )
        split_hashes = json.loads((run_dir / "splits/split_hashes.json").read_text())
        historical = json.loads(
            (run_dir / "provenance/historical_snapshot_validation.json").read_text()
        )
        current = json.loads(
            (run_dir / "provenance/current_snapshot_validation.json").read_text()
        )
        replay_schema = json.loads(
            (run_dir / "provenance/historical_prediction_replay_schema.json").read_text()
        )
        raw_diagnostic = json.loads(
            (run_dir / "provenance/raw_embedding_mapping_diagnostic.json").read_text()
        )
        raw_join = pd.read_csv(
            run_dir / "provenance/raw_embedding_join_diagnostic.csv"
        )
        raw_source_inventory = pd.read_csv(
            run_dir / "provenance/raw_embedding_source_inventory.tsv", sep="\t",
        )
        pca_inventory = pd.read_csv(
            run_dir / "provenance/pca_artifact_inventory.tsv", sep="\t",
        )
        embed_semantics = json.loads(
            (run_dir / "provenance/embed_sim_semantics.json").read_text()
        )
        harmonized_manifest = json.loads(
            (run_dir / "provenance/harmonized_feature_layer_manifest.json").read_text()
        )
        harmonized_layer = pd.read_csv(
            run_dir / "provenance/harmonized_audit_feature_layer.csv.gz"
        )
        weights = json.loads(
            (run_dir / "provenance/review_weight_validation.json").read_text()
        )
        cuda = json.loads(
            (run_dir / "provenance/cuda_capability_probe.json").read_text()
        )
        gate_formula = json.loads(
            (run_dir / "provenance/gate_formula_manifest.json").read_text()
        )
        r92 = json.loads((run_dir / "metrics/r92_control.json").read_text())
        prior_pre = json.loads(
            (run_dir / "provenance/prior_immutable_verification_pre.json").read_text()
        )
        prior_post = json.loads(
            (run_dir / "provenance/prior_immutable_verification_post.json").read_text()
        )
        split = pd.read_csv(run_dir / "splits/row_split_manifest.csv.gz")
        card_split = pd.read_csv(run_dir / "splits/card_split_manifest.csv")
        split_summary = pd.read_csv(run_dir / "splits/split_summary.csv")
        overlap = pd.read_csv(run_dir / "splits/overlap_checks.csv")
        gate_frame = pd.read_csv(
            run_dir / "provenance/gate_reconstruction_validation.csv"
        )
        audit_coverage = pd.read_csv(
            run_dir / "provenance/audit_gate_upstream_coverage.csv"
        )
        feature_manifest = pd.read_csv(run_dir / "config/feature_manifest.csv")
        pre_sources = pd.read_csv(
            run_dir / "provenance/source_input_hashes_pre.tsv", sep="\t",
        )
        post_sources = pd.read_csv(
            run_dir / "provenance/source_input_hashes_post.tsv", sep="\t",
        )
        status_path = run_dir / "RUN_STATUS.txt"
        status_text = status_path.read_text()
        status = _status_fields(status_path)
        status_keys = [
            line.partition("=")[0].strip()
            for line in status_text.splitlines() if "=" in line
        ]
    except Exception as exc:
        raise IntegrityError(
            f"Preflight semantic evidence is unreadable: {type(exc).__name__}: {exc}"
        ) from exc

    gate_inputs, stored_gates = _gate_inputs_from_payload(registry_payload)
    try:
        registry = evidence_workflow.build_gate_registry(gate_inputs)
    except Exception as exc:
        raise IntegrityError(f"Preflight gate registry is invalid: {exc}") from exc
    no_pca_variant = variants.get("no_pca", {})
    no_pca_runnable = no_pca_variant.get("status") == "RUNNABLE"
    no_pca_status = (
        "VERIFIED_EXACT_RAW_EMBEDDINGS_AVAILABLE"
        if no_pca_runnable else no_pca_variant.get("status")
    )
    facts = evidence_workflow.WorkflowFacts(
        phase="AMENDED_PREFLIGHT",
        official_smoke_variants_completed=int(
            ledger.get("official_smoke_variants_completed", -1)
        ),
        exploratory_smoke_variants_completed=int(
            ledger.get("exploratory_smoke_variants_completed", -1)
        ),
        full_scientific_run_executed=ledger.get("full_scientific_run_executed"),
        full_variants_completed=int(ledger.get("full_variants_completed", -1)),
        no_pca_variant_status=str(no_pca_status),
        no_pca_variant_runnable=no_pca_runnable,
        protocol_amendment=str(
            identity.get("protocol", "AMENDMENT_01")
        ),
    )
    state = evidence_workflow.derive_workflow_state(registry, facts)
    gate_matches = lambda key, observed: (
        registry.accepted(key) is bool(observed)
    )

    replay_path = run_dir / "provenance/historical_prediction_replay.csv.gz"
    try:
        replay_summary, replay_rows_valid = _historical_replay_reopen_summary(
            replay_path,
        )
    except Exception:
        replay_summary, replay_rows_valid = {}, False
    stored_replay = replay_schema.get("serialized_reopen_verification", {})
    stored_replay_summary = stored_replay.get("summary", {})
    recomputed_split_hash = split_assignment_hash(
        split.stable_candidate_id, split.split, split.label,
        split.eligible, split.split_order,
    )
    row_split_columns = [
        "stable_candidate_id", "file_name", "ann_id", "scale", "card_id",
        "label", "split", "eligible", "split_order",
    ]
    card_split_columns = [
        "card_id", "split", "rows", "positive", "eligible", "negative",
    ]
    split_summary_columns = [
        "split", "rows", "positive", "cards", "eligible", "negative",
    ]
    overlap_columns = ["comparison", "overlap_count", "status"]
    allowed_splits = {
        "training", "validation", "historical_heldout_ineligible_border",
    }
    split_labels = pd.to_numeric(split.label, errors="coerce")
    split_orders = pd.to_numeric(split.split_order, errors="coerce")
    split_eligible_text = split.eligible.astype(str).str.lower()
    split_eligible_domain_valid = split_eligible_text.isin({"true", "false"}).all()
    observed_eligible = split_eligible_text.eq("true")
    normalized_scales = split.scale.map(normalize_scale)
    reconstructed_ids_valid = False
    reconstructed_cards_valid = False
    try:
        reconstructed_ids_valid = np.array_equal(
            stable_training_ids(split).astype(str),
            split.stable_candidate_id.astype(str),
        )
        reconstructed_cards_valid = np.array_equal(
            split.file_name.map(normalize_card_id).astype(str),
            split.card_id.astype(str),
        )
    except Exception:
        pass
    expected_eligible = split.split.astype(str).isin({"training", "validation"})
    expected_orders_valid = True
    for split_name in allowed_splits:
        mask = split.split.astype(str) == split_name
        observed = split_orders[mask].to_numpy(np.int64, na_value=-2)
        expected = (
            np.arange(int(mask.sum()), dtype=np.int64)
            if split_name in {"training", "validation"}
            else np.full(int(mask.sum()), -1, dtype=np.int64)
        )
        expected_orders_valid = expected_orders_valid and np.array_equal(
            np.sort(observed), expected,
        )
    expected_card_split = (
        split.assign(
            label=split_labels.astype(np.int64),
            eligible=expected_eligible.astype(bool),
        )
        .groupby(["card_id", "split"], as_index=False)
        .agg(
            rows=("stable_candidate_id", "size"),
            positive=("label", "sum"),
            eligible=("eligible", "sum"),
        )
    )
    expected_card_split["negative"] = (
        expected_card_split.rows - expected_card_split.positive
    )
    expected_split_summary = (
        split.assign(
            label=split_labels.astype(np.int64),
            eligible=expected_eligible.astype(bool),
        )
        .groupby("split", as_index=False)
        .agg(
            rows=("stable_candidate_id", "size"),
            positive=("label", "sum"),
            cards=("card_id", "nunique"),
            eligible=("eligible", "sum"),
        )
    )
    expected_split_summary["negative"] = (
        expected_split_summary.rows - expected_split_summary.positive
    )
    training_cards = set(
        split.loc[split.split.astype(str) == "training", "card_id"].astype(str)
    )
    validation_cards = set(
        split.loc[split.split.astype(str) == "validation", "card_id"].astype(str)
    )
    audit_cards: set[str] = set()
    try:
        audit_cards = _canonical_audit_card_ids(AUDIT_CSV)
    except Exception:
        pass
    expected_overlap = pd.DataFrame([
        {
            "comparison": "training_vs_validation",
            "overlap_count": len(training_cards & validation_cards),
        },
        {
            "comparison": "development_vs_external_audit",
            "overlap_count": len((training_cards | validation_cards) & audit_cards),
        },
    ])
    expected_overlap["status"] = np.where(
        expected_overlap.overlap_count == 0, "PASS", "FAIL",
    )

    code_tree = identity.get("code_tree", {})
    source_tree = identity.get("source_hashes", {})
    expected_code_paths = {
        path.relative_to(run_dir).as_posix()
        for root in (run_dir / "code", run_dir / "tests")
        for path in root.glob("*")
        if path.is_file() and path.suffix in {".py", ".sh"}
    }
    code_snapshot_matches = all(
        (run_dir / relative).read_bytes()
        == (
            run_dir
            / "provenance"
            / ("code_snapshot" if relative.startswith("code/") else "tests_snapshot")
            / Path(relative).name
        ).read_bytes()
        for relative in expected_code_paths
    )
    source_rows = post_sources.set_index("path", drop=False)
    source_hash_map = {
        str(path): str(value) for path, value in source_tree.items()
    } if isinstance(source_tree, Mapping) else {}

    expected_status = _preflight_derived_payload(state)
    derived_status_fields = dict(
        line.split("=", 1)
        for line in evidence_workflow.render_status_lines(state).splitlines()
    )
    blockers_text = (run_dir / "BLOCKERS.md").read_text()
    global_blocker_text = blockers_text.split(
        "## Variant-Specific Limitation", 1,
    )[0]
    blocker_rows = re.findall(
        r"^- `([^`]+)`", global_blocker_text, flags=re.MULTILINE,
    )
    expected_global_blocker_lines = [
        f"- `{code}` ({value['classification']}/{value['status']}): "
        f"{value['reason']} Evidence: {', '.join(value['evidence_refs'])}."
        for code, value in _expected_blocker_details(state).items()
    ] or ["- None."]
    expected_global_blocker_text = (
        "# Global Blockers\n\n" + "\n".join(expected_global_blocker_lines)
    )
    expected_identity = bool(
        identity.get("run_id") == run_dir.name
        and identity.get("run_kind") == state.run_kind
        and identity.get("state") == state.run_state
        and identity.get("data_source_decision") == state.data_source_decision
        and identity.get("candidate_split_hash") == recomputed_split_hash
        and identity.get("split_locked") is state.split_locked
        and identity.get("full_authorized") is False
        and identity.get("stability_authorized") is False
        and identity.get("permitted_transitions") == (
            ["CUDA_SMOKE_NON_SCIENTIFIC", "PACKAGE_REVIEW_EVIDENCE"]
            if state.smoke_eligible else ["PACKAGE_REVIEW_EVIDENCE"]
        )
    )

    checks: dict[str, bool] = {}
    checks["gate_registry_schema_and_calculated_tokens"] = bool(
        registry_payload.get("allowed_classifications")
        == [value.value for value in evidence_workflow.EvidenceClassification]
        and all(
            stored_gates[key].get("accepted") is gate.accepted
            and stored_gates[key].get("status") == gate.status_token
            and tuple(gate.evidence_refs) == PREFLIGHT_GATE_EVIDENCE_REFS[key]
            and isinstance(stored_gates[key].get("detail"), str)
            and bool(stored_gates[key].get("detail", "").strip())
            and all(
                _is_review_relative_regular_file(run_dir, relative)
                or (
                    key == "output_manifest_zip_verification"
                    and not gate.accepted
                    and relative
                    == "provenance/package_staging_verification.json"
                )
                for relative in gate.evidence_refs
            )
            for key, gate in registry.evidence.items()
        )
    )
    checks["workflow_rederived_exactly"] = bool(
        registry_payload.get("derived_workflow") == expected_status
    )
    checks["identity_matches_derived_workflow"] = expected_identity
    checks["run_status_matches_derived_workflow"] = bool(
        len(status_keys) == len(set(status_keys))
        and all(status.get(key) == value for key, value in derived_status_fields.items())
        and status.get("TRAINING_TABLE_PATH") == report.get("training_table_path")
        and status.get("TRAINING_TABLE_SHA256") == report.get("training_table_sha256")
        and status.get("HISTORICAL_LITERAL_REPLAY_TOLERANCE_STATUS")
        == report.get("historical_replay", {}).get(
            "literal_probability_tolerance_status"
        )
        and status.get("BLOCKERS") == ";".join(state.blocker_codes)
        and status.get("NEXT_ACTION") == "STOP_AND_AWAIT_EXTERNAL_REVIEW"
    )
    checks["blocker_document_matches_derived_workflow"] = bool(
        len(blocker_rows) == len(set(blocker_rows))
        and set(blocker_rows) == set(state.blocker_codes)
        and (("- None." in blockers_text) is (not state.blocker_codes))
        and global_blocker_text.strip() == expected_global_blocker_text.strip()
    )
    expected_blockers = _expected_blocker_details(state)
    report_blockers_match = report.get("global_blockers") == expected_blockers
    checks["report_matches_derived_workflow"] = bool(
        report.get("protocol") == "AMENDMENT_01"
        and report.get("run_kind") == state.run_kind
        and report.get("status") == state.amended_preflight_status
        and report.get("data_source_decision") == state.data_source_decision
        and isinstance(report.get("elapsed_seconds"), (int, float))
        and math.isfinite(float(report.get("elapsed_seconds")))
        and float(report.get("elapsed_seconds")) >= 0
        and report_blockers_match
    )
    checks["report_scientific_statuses_match_registry"] = bool(
        report.get("feature_dependency_closure")
        == state.gate_statuses["feature_dependency_closure"]
        and report.get("feature_semantics")
        == state.gate_statuses["feature_semantics"]
        and report.get("gate_reconstruction")
        == state.gate_statuses["gate_reconstruction"]
        and report.get("review_weight_parity")
        == state.gate_statuses["review_weight_parity"]
        and report.get("cuda_smoke") == state.cuda_smoke_status
        and report.get("ready_for_full_awaiting_external_review")
        is state.ready_for_full_awaiting_external_review
        and report.get("output_manifest_zip_verification")
        == state.gate_statuses["output_manifest_zip_verification"]
        and report.get("no_pca") == no_pca_status
    )
    expected_historical_report = {
        "literal_probability_tolerance_status": historical.get(
            "literal_probability_tolerance_status"
        ),
        "direct_max_difference": historical.get(
            "direct_decimal_probability_max_absolute_difference"
        ),
        "source_dtype_roundtrip_status": historical.get(
            "source_dtype_roundtrip_status"
        ),
        "source_dtype_max_difference": historical.get(
            "source_dtype_float32_probability_max_absolute_difference"
        ),
        "serialized_reopen_status": historical.get(
            "serialized_replay_artifact", {}
        ).get("reopen_verification", {}).get("status"),
        "serialized_reopen_summary_matches": historical.get(
            "serialized_replay_artifact", {}
        ).get("reopen_verification", {}).get(
            "summary_matches_in_memory_calculation"
        ),
    }
    expected_not_applicable = {
        "tables/SMOKE_REPORT.csv": (
            "Smoke is a separate transition not yet executed"
            if state.smoke_eligible else "No Smoke because global gates failed"
        ),
        "metrics/SMOKE_REPORT.json": (
            "Smoke is a separate transition not yet executed"
            if state.smoke_eligible else "No Smoke because global gates failed"
        ),
        "smoke_models_bundle": "No Smoke models were created in this Preflight",
    }
    checks["report_duplicate_evidence_payloads_are_bound"] = bool(
        report.get("r92_control") == r92
        and report.get("execution_ledger") == ledger
        and report.get("historical_replay") == expected_historical_report
        and report.get("candidate_split_hash") == split_hashes
        and report.get("prior_run_immutability") == (
            "PASS" if (
                prior_pre == prior_post
                and prior_post.get("prior", {}).get("manifest_status") == "PASS"
                and prior_post.get("prior", {}).get("zip", {}).get("status") == "PASS"
                and prior_post.get("selected_v3", {}).get("status") == "PASS"
            ) else "FAIL"
        )
        and report.get("full_scientific_run_executed")
        is ledger.get("full_scientific_run_executed")
        and report.get("full_variants_completed")
        == f"{ledger.get('full_variants_completed')}/10"
        and report.get("not_applicable_outputs") == expected_not_applicable
        and _command_payload_matches_log(
            report.get("syntax_compile"), run_dir / "logs/syntax_compile.log",
        )
        and _command_payload_matches_log(
            report.get("tests"), run_dir / "logs/pytest.log",
        )
        and _command_payload_matches_log(
            report.get("reviewer_snapshot_syntax_compile"),
            run_dir / "logs/reviewer_snapshot_syntax_compile.log",
        )
        and _command_payload_matches_log(
            report.get("reviewer_snapshot_tests"),
            run_dir / "logs/reviewer_snapshot_pytest.log",
        )
    )
    split_schema_and_derivations = bool(
        list(split.columns) == row_split_columns
        and list(card_split.columns) == card_split_columns
        and list(split_summary.columns) == split_summary_columns
        and list(overlap.columns) == overlap_columns
        and split.stable_candidate_id.is_unique
        and not split.stable_candidate_id.astype(str).eq("").any()
        and set(split.split.astype(str)) == allowed_splits
        and split_labels.notna().all()
        and split_labels.isin([0, 1]).all()
        and split_orders.notna().all()
        and split_eligible_domain_valid
        and np.array_equal(
            observed_eligible.to_numpy(),
            expected_eligible.to_numpy(),
        )
        and reconstructed_ids_valid
        and reconstructed_cards_valid
        and normalized_scales.astype(str).ne("").all()
        and expected_orders_valid
        and _frames_equal_ignoring_dtype(
            card_split[card_split_columns], expected_card_split[card_split_columns],
        )
        and _frames_equal_ignoring_dtype(
            split_summary[split_summary_columns],
            expected_split_summary[split_summary_columns],
        )
        and _frames_equal_ignoring_dtype(
            overlap[overlap_columns], expected_overlap[overlap_columns],
        )
    )
    checks["split_manifest_recomputed_and_bound"] = bool(
        split_schema_and_derivations
        and recomputed_split_hash
        == split_hashes.get("candidate_split_manifest_sha256")
        == identity.get("candidate_split_hash")
        == report.get("candidate_split_hash", {}).get(
            "candidate_split_manifest_sha256"
        )
        and split_hashes.get("locked_for_scientific_use") is state.split_locked
        and split_hashes.get("lock_status") == state.split_lock_status
        and report.get("candidate_split_hash") == split_hashes
    )
    split_manifest_observed = bool(
        split_schema_and_derivations
        and recomputed_split_hash
        == split_hashes.get("candidate_split_manifest_sha256")
    )
    split_zero_leakage_observed = bool(
        split_schema_and_derivations
        and sha256_file(AUDIT_CSV) == EXPECTED_HASHES.get(str(AUDIT_CSV))
        and len(overlap) == 2
        and (overlap.status == "PASS").all()
        and (pd.to_numeric(overlap.overlap_count, errors="raise") == 0).all()
    )
    checks["split_gate_tokens_match_recomputed_evidence"] = bool(
        gate_matches("split_manifest_integrity", split_manifest_observed)
        and gate_matches("split_zero_leakage", split_zero_leakage_observed)
    )
    replay_frame = _historical_replay_reopen_frame(replay_path)
    expected_replay_population = split.loc[
        split.split.astype(str).eq("validation")
        & expected_eligible.astype(bool)
    ].sort_values("split_order", kind="stable").reset_index(drop=True)
    replay_identity_label_match = bool(
        len(replay_frame) == len(expected_replay_population)
        and np.array_equal(
            replay_frame.stable_candidate_id.astype(str),
            expected_replay_population.stable_candidate_id.astype(str),
        )
        and np.array_equal(
            replay_frame.file_name.astype(str),
            expected_replay_population.file_name.astype(str),
        )
        and np.array_equal(
            replay_frame.ann_id.astype(str),
            expected_replay_population.ann_id.astype(str),
        )
        and np.array_equal(
            replay_frame.scale.map(normalize_scale).astype(str),
            expected_replay_population.scale.map(normalize_scale).astype(str),
        )
        and np.array_equal(
            pd.to_numeric(replay_frame.label, errors="raise").to_numpy(np.int8),
            expected_replay_population.label.to_numpy(np.int8),
        )
    )
    canonical_prediction_path = str(R92_PREDICTIONS)
    prediction_source_sha256 = str(
        replay_schema.get("source_prediction_sha256", "")
    )
    prediction_pre_rows = pre_sources.loc[
        pre_sources.path.astype(str).eq(canonical_prediction_path)
    ] if "path" in pre_sources else pd.DataFrame()
    prediction_post_rows = post_sources.loc[
        post_sources.path.astype(str).eq(canonical_prediction_path)
    ] if "path" in post_sources else pd.DataFrame()
    prediction_source_inventory_bound = bool(
        replay_schema.get("source_prediction_path") == canonical_prediction_path
        and historical.get("saved_prediction_source_path")
        == canonical_prediction_path
        and historical.get("saved_prediction_source_sha256")
        == prediction_source_sha256
        and _valid_sha256(prediction_source_sha256)
        and len(prediction_pre_rows) == 1
        and len(prediction_post_rows) == 1
        and str(prediction_pre_rows.iloc[0].get("sha256", ""))
        == prediction_source_sha256
        and str(prediction_post_rows.iloc[0].get("sha256", ""))
        == prediction_source_sha256
        and source_hash_map.get(canonical_prediction_path)
        == prediction_source_sha256
    )
    prediction_source_frame: pd.DataFrame | None = None
    if prediction_source_inventory_bound:
        try:
            prediction_source_frame = load_historical_predictions_verified(
                R92_PREDICTIONS,
                prediction_source_sha256,
            )
        except Exception:
            pass
    prediction_source_replay_match = bool(
        prediction_source_frame is not None
        and _historical_prediction_source_matches_replay(
            prediction_source_frame,
            replay_frame,
        )
    )
    checks["historical_replay_reopened_and_bound"] = bool(
        replay_rows_valid
        and replay_identity_label_match
        and prediction_source_inventory_bound
        and prediction_source_replay_match
        and historical.get("ordered_rows_compared") == len(replay_frame)
        and historical.get("ordered_candidate_identity_and_label_match")
        is replay_identity_label_match
        and sha256_file(replay_path) == replay_schema.get("comparison_sha256")
        == stored_replay.get("artifact_sha256")
        and replay_summary == stored_replay_summary
        and stored_replay.get("status") == "PASS"
        and stored_replay.get("summary_matches_in_memory_calculation") is True
        and historical.get("direct_decimal_probability_max_absolute_difference")
        == replay_summary.get("direct_decimal_max_absolute_difference")
        and historical.get("direct_decimal_rows_exceeding_1e_12")
        == replay_summary.get("direct_decimal_mismatch_count_gt_1e_12")
        and registry.accepted("historical_prediction_replay")
        is (
            replay_summary.get("direct_decimal_max_absolute_difference", math.inf)
            <= 1e-12
            and replay_summary.get("direct_decimal_mismatch_count_gt_1e_12") == 0
        )
    )
    historical_provenance_observed = bool(
        historical.get("sha256") == report.get("training_table_sha256")
        == identity.get("source_hashes", {}).get(report.get("training_table_path"))
        == EXPECTED_HASHES.get(str(report.get("training_table_path")))
        and historical.get("path") == report.get("training_table_path")
        and historical.get("stable_candidate_id_unique") is True
        and historical.get("stable_key_null_rows") == 0
        and historical.get("stable_key_duplicate_rows") == 0
        and historical.get("external_audit_card_overlap") == 0
        and historical.get("all_features_present", False) is True
    )
    expected_pca_inventory = pd.DataFrame()
    expected_pca_evidence: dict[str, Any] = {}
    try:
        expected_pca_inventory = canonical_pca_artifact_inventory()
        expected_pca_evidence = pca_lineage_evidence(expected_pca_inventory)
    except Exception:
        pass
    pca_evidence_exact = bool(
        not expected_pca_inventory.empty
        and _frames_equal_ignoring_dtype(
            pca_inventory, expected_pca_inventory,
        )
        and historical.get("pca_prototype_compatibility")
        == expected_pca_evidence
    )
    pca_compatible_observed = bool(
        pca_evidence_exact
        and expected_pca_evidence.get("compatible_single_basis") is True
    )
    checks["historical_snapshot_and_pca_gate_bound"] = bool(
        gate_matches("historical_snapshot_provenance", historical_provenance_observed)
        and pca_evidence_exact
        and gate_matches("pca_prototype_compatibility", pca_compatible_observed)
    )
    split_reproduction_observed = bool(
        replay_identity_label_match
        and {"training", "validation"} <= set(split.split.astype(str))
        and (split.split.astype(str) == "training").any()
        and (split.split.astype(str) == "validation").any()
        and split_zero_leakage_observed
    )
    replay_labels = pd.to_numeric(
        replay_frame.label, errors="raise",
    ).to_numpy(np.int8)
    replay_probability = _decode_float32_hex_bits(
        replay_frame.replayed_probability_float32_bits_hex,
    ).astype(float)
    historical_metrics = historical.get("metrics", {})
    try:
        historical_sidecars = historical_metric_sidecar_evidence()
    except Exception:
        historical_sidecars = {}
    replay_metric_values: dict[str, Any] = {}
    try:
        replay_threshold = float(historical_metrics["threshold"])
        replay_prediction = replay_probability >= replay_threshold
        replay_tn, replay_fp, replay_fn, replay_tp = confusion_matrix(
            replay_labels, replay_prediction, labels=[0, 1],
        ).ravel()
        replay_metric_values = {
            "average_precision": float(
                average_precision_score(replay_labels, replay_probability)
            ),
            "roc_auc": float(roc_auc_score(replay_labels, replay_probability)),
            "tn": int(replay_tn), "fp": int(replay_fp),
            "fn": int(replay_fn), "tp": int(replay_tp),
        }
    except Exception:
        pass
    historical_metric_observed = bool(
        replay_identity_label_match
        and replay_metric_values
        and all(
            abs(float(historical_metrics.get(key, math.inf)) - value) <= 1e-12
            for key, value in replay_metric_values.items()
            if key in {"average_precision", "roc_auc"}
        )
        and all(
            historical_metrics.get(key) == value
            for key, value in replay_metric_values.items()
            if key in {"tn", "fp", "fn", "tp"}
        )
        and historical_metrics.get("sidecar_average_precision")
        == historical_metrics.get("average_precision")
        and historical_metrics.get("sidecar_roc_auc")
        == historical_metrics.get("roc_auc")
        and historical_metrics.get("sidecar_used_trees")
        == historical_metrics.get("used_trees")
        == historical_sidecars.get("sidecar_used_trees")
        == historical_sidecars.get("checkpoint_used_trees")
        and historical_metrics.get("threshold")
        == historical_sidecars.get("threshold")
        and historical_metrics.get("sidecar_average_precision")
        == historical_sidecars.get("average_precision")
        and historical_metrics.get("sidecar_roc_auc")
        == historical_sidecars.get("roc_auc")
        and [
            [historical_metrics.get("tn"), historical_metrics.get("fp")],
            [historical_metrics.get("fn"), historical_metrics.get("tp")],
        ] == historical_sidecars.get("confusion_matrix")
        and all(
            source_hash_map.get(str(historical_sidecars.get(path_key)))
            == historical_sidecars.get(hash_key)
            for path_key, hash_key in (
                ("metrics_path", "metrics_sha256"),
                ("threshold_path", "threshold_sha256"),
                ("checkpoint_path", "checkpoint_sha256"),
            )
        )
        and historical_metrics.get("metric_and_tree_parity") is True
    )
    checks["historical_split_and_metric_gates_bound"] = bool(
        gate_matches("historical_split_reproduction", split_reproduction_observed)
        and gate_matches("historical_metric_tree_parity", historical_metric_observed)
    )
    current_observed = bool(
        current.get("stable_candidate_id_unique") is True
        and current.get("stable_key_null_rows") == 0
        and current.get("stable_key_duplicate_rows") == 0
        and current.get("all_features_present") is True
        and current.get("path") in source_hash_map
        and current.get("sha256") == source_hash_map.get(current.get("path"))
        == EXPECTED_HASHES.get(str(current.get("path")))
    )
    checks["current_snapshot_analysis_bound"] = bool(
        gate_matches("current_snapshot_analysis", current_observed)
    )
    source_columns = ["path", "size_bytes", "sha256"]
    source_columns_available = all(
        column in pre_sources and column in post_sources
        for column in source_columns
    )
    pre_source_view = (
        pre_sources[source_columns].reset_index(drop=True)
        if source_columns_available else pd.DataFrame()
    )
    post_source_view = (
        post_sources[source_columns].reset_index(drop=True)
        if source_columns_available else pd.DataFrame()
    )
    pre_source_map = (
        dict(zip(pre_sources.path.astype(str), pre_sources.sha256.astype(str)))
        if {"path", "sha256"} <= set(pre_sources) else {}
    )
    live_source_rows_valid = bool(source_columns_available)
    if live_source_rows_valid:
        for row in post_source_view.itertuples(index=False):
            path = Path(str(row.path))
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != int(row.size_bytes)
                or sha256_file(path) != str(row.sha256)
            ):
                live_source_rows_valid = False
                break
    source_integrity_observed = bool(
        source_columns_available
        and pre_source_view.equals(post_source_view)
        and pre_sources.path.is_unique
        and post_sources.path.is_unique
        and source_hash_map == pre_source_map
        and live_source_rows_valid
        and all(
            pre_source_map.get(path) == expected
            for path, expected in EXPECTED_HASHES.items()
        )
    )
    checks["source_inventory_pre_post_and_identity_bound"] = bool(
        gate_matches("source_input_integrity", source_integrity_observed)
        and report.get("source_input_integrity")
        == ("PASS" if source_integrity_observed else "FAIL")
    )
    historical_source_frame: pd.DataFrame | None = None
    current_source_frame: pd.DataFrame | None = None
    audit_source_frame: pd.DataFrame | None = None
    try:
        historical_source_frame = load_training_verified(
            HISTORICAL_TRAINING_CSV,
            EXPECTED_HASHES[str(HISTORICAL_TRAINING_CSV)],
        )
        current_source_frame = load_training_verified(
            CURRENT_TRAINING_CSV,
            EXPECTED_HASHES[str(CURRENT_TRAINING_CSV)],
        )
        audit_source_frame = _clean_canonical_audit_frame(
            AUDIT_CSV, EXPECTED_HASHES[str(AUDIT_CSV)],
        )
    except Exception:
        pass
    features = get_r92_feature_order()
    historical_rows_bound = bool(
        historical_source_frame is not None
        and len(historical_source_frame) == len(split)
        and np.array_equal(
            historical_source_frame.stable_candidate_id.astype(str),
            split.stable_candidate_id.astype(str),
        )
        and np.array_equal(
            historical_source_frame.file_name.astype(str),
            split.file_name.astype(str),
        )
        and np.array_equal(
            historical_source_frame.ann_id.astype(str),
            split.ann_id.astype(str),
        )
        and np.array_equal(
            historical_source_frame.scale.map(normalize_scale).astype(str),
            split.scale.map(normalize_scale).astype(str),
        )
        and np.array_equal(
            historical_source_frame.card_id.astype(str), split.card_id.astype(str),
        )
        and np.array_equal(
            historical_source_frame.label.to_numpy(np.int8),
            split_labels.to_numpy(np.int8),
        )
    )
    checks["source_tables_reopened_and_split_rows_bound"] = bool(
        historical_rows_bound
        and current_source_frame is not None
        and audit_source_frame is not None
        and Path(str(historical.get("path"))).resolve()
        == HISTORICAL_TRAINING_CSV.resolve()
        and historical.get("sha256")
        == EXPECTED_HASHES[str(HISTORICAL_TRAINING_CSV)]
        and Path(str(current.get("path"))).resolve()
        == CURRENT_TRAINING_CSV.resolve()
        and current.get("sha256")
        == EXPECTED_HASHES[str(CURRENT_TRAINING_CSV)]
        and historical.get("rows") == len(historical_source_frame)
        and historical.get("cards") == int(historical_source_frame.card_id.nunique())
        and historical.get("negative_rows")
        == int((historical_source_frame.label == 0).sum())
        and historical.get("positive_rows")
        == int((historical_source_frame.label == 1).sum())
        and historical.get("feature_count") == len(features)
        and historical.get("all_features_present")
        is all(feature in historical_source_frame for feature in features)
        and current.get("rows") == len(current_source_frame)
        and current.get("cards") == int(current_source_frame.card_id.nunique())
        and current.get("negative_rows") == int((current_source_frame.label == 0).sum())
        and current.get("positive_rows") == int((current_source_frame.label == 1).sum())
        and current.get("feature_count") == len(features)
        and current.get("all_features_present")
        is all(feature in current_source_frame for feature in features)
    )
    checks["code_tree_and_snapshots_bound"] = bool(
        isinstance(code_tree, Mapping)
        and set(code_tree) == expected_code_paths
        and all(
            sha256_file(run_dir / relative) == expected
            for relative, expected in code_tree.items()
        )
        and code_snapshot_matches
    )
    identity_seed = {
        "protocol": identity.get("protocol"),
        "run_kind": identity.get("run_kind"),
        "command_line": identity.get("command_line"),
        "working_directory": identity.get("working_directory"),
        "task_spec_sha256": identity.get("task_spec_sha256"),
        "source_hashes": identity.get("source_hashes"),
        "code_tree": identity.get("code_tree"),
    }
    identity_suffix = sha256_bytes(canonical_json(identity_seed))[:8]
    checks["identity_seed_expected_hashes_and_run_suffix_bound"] = bool(
        identity.get("protocol") == "AMENDMENT_01"
        and isinstance(identity.get("command_line"), list)
        and isinstance(identity.get("working_directory"), str)
        and identity.get("task_spec_sha256") == EXPECTED_HASHES.get(str(TASK_SPEC))
        and sha256_file(run_dir / "docs/PROTOCOL_AMENDMENT_01_TASK_SPEC.md")
        == identity.get("task_spec_sha256")
        and re.fullmatch(
            rf"run_\d{{8}}_\d{{6}}_{identity_suffix}_amended_preflight",
            run_dir.name,
        ) is not None
    )
    expected_feature_manifest = feature_manifest_frame(features)
    checks["feature_manifest_matches_verified_ontology_exactly"] = (
        _frames_equal_ignoring_dtype(feature_manifest, expected_feature_manifest)
    )
    claimed_raw_features = raw_diagnostic.get(
        "exact_ordered_raw_feature_columns"
    ) or []
    claimed_raw_contract = raw_diagnostic.get("raw_layer_contract")
    claimed_raw_verified = bool(
        raw_diagnostic.get("status")
        == "VERIFIED_EXACT_RAW_EMBEDDINGS_AVAILABLE"
        and raw_diagnostic.get("classification") == "VERIFIED"
    )
    expected_variants = build_variant_feature_sets(
        features,
        raw_embedding_features=claimed_raw_features,
        raw_mapping_verified=claimed_raw_verified,
        raw_layer_contract=claimed_raw_contract,
    )
    checks["feature_dependency_graph_and_variants_rederived_exactly"] = bool(
        dependency_graph == feature_dependency_graph_payload()
        and variants == expected_variants
    )
    expected_gate_frame = pd.DataFrame()
    expected_audit_coverage = pd.DataFrame()
    if (
        historical_source_frame is not None
        and current_source_frame is not None
        and audit_source_frame is not None
    ):
        try:
            expected_gate_frame = pd.DataFrame(
                gate_validation_rows(
                    historical_source_frame, "historical_363563",
                )
                + gate_validation_rows(current_source_frame, "current_511082")
            )
            expected_audit_coverage = pd.DataFrame(
                required_numeric_coverage(audit_source_frame, GATE_REQUIRED_INPUTS)
            )
        except Exception:
            expected_gate_frame = pd.DataFrame()
            expected_audit_coverage = pd.DataFrame()
    gate_formula_expected: dict[str, Any] = {}
    gate_formula_observed = dict(gate_formula)
    try:
        gate_formula_expected = _fresh_gate_formula_manifest()
    except Exception:
        pass
    implementation_path = Path(str(gate_formula.get("implementation_path", "")))
    implementation_path_valid = bool(
        tuple(implementation_path.parts[-2:]) == ("code", "amendment_core.py")
    )
    expected_implementation_path = Path(str(
        gate_formula_expected.get("implementation_path", "")
    ))
    expected_implementation_path_valid = bool(
        tuple(expected_implementation_path.parts[-2:])
        == ("code", "amendment_core.py")
    )
    if implementation_path_valid and expected_implementation_path_valid:
        gate_formula_observed["implementation_path"] = "code/amendment_core.py"
        gate_formula_expected["implementation_path"] = "code/amendment_core.py"
    gate_reconstruction_observed = bool(
        not expected_gate_frame.empty
        and not expected_audit_coverage.empty
        and _numeric_evidence_frames_equal(gate_frame, expected_gate_frame)
        and _numeric_evidence_frames_equal(
            audit_coverage, expected_audit_coverage,
        )
        and gate_formula_observed == gate_formula_expected
        and gate_formula.get("status") == "PASS"
        and all(gate_formula.get("checks", {}).values())
        and gate_formula.get("source_notebook_sha256")
        == EXPECTED_HASHES.get(str(NOTEBOOK), sha256_file(NOTEBOOK))
        and gate_formula.get("implementation_sha256")
        == sha256_file(run_dir / "code/amendment_core.py")
        and implementation_path_valid
        and expected_implementation_path_valid
    )
    checks["gate_reconstruction_machine_evidence_bound"] = gate_matches(
        "gate_reconstruction", gate_reconstruction_observed,
    )
    semantic_evidence_exact = False
    if historical_source_frame is not None and audit_source_frame is not None:
        try:
            expected_embed_semantics, expected_harmonized = (
                feature_semantics_evidence(
                    historical_source_frame, audit_source_frame, features,
                )
            )
            expected_harmonized_layer = pd.concat(
                [
                    audit_source_frame[["stable_candidate_id", "card_id"]],
                    expected_harmonized,
                ],
                axis=1,
            )
            layer_relative = (
                "provenance/harmonized_audit_feature_layer.csv.gz"
            )
            expected_harmonized_manifest = harmonized_layer_manifest(
                expected_pca_evidence,
                expected_harmonized,
                features,
                relative_path=layer_relative,
                layer_sha256=sha256_file(run_dir / layer_relative),
            )
            semantic_evidence_exact = bool(
                embed_semantics == expected_embed_semantics
                and _float32_layer_frames_equal(
                    harmonized_layer, expected_harmonized_layer,
                )
                and harmonized_manifest == expected_harmonized_manifest
                and harmonized_manifest.get("audit_labels_used") is False
                and "human_label" not in harmonized_layer
            )
        except Exception:
            semantic_evidence_exact = False
    feature_semantics_observed = bool(
        pca_compatible_observed
        and semantic_evidence_exact
        and (gate_frame.status.astype(str) == "PASS").all()
        and (audit_coverage.classification.astype(str) == "VERIFIED").all()
    )
    checks["feature_semantics_gate_bound"] = bool(
        semantic_evidence_exact
        and gate_matches("feature_semantics", feature_semantics_observed)
    )
    dependency_observed = bool(
        checks["feature_manifest_matches_verified_ontology_exactly"]
        and checks["feature_dependency_graph_and_variants_rederived_exactly"]
        and all(
            value.get("dependency_closure_linter") == "PASS"
            for value in variants.values()
            if value.get("status") == "RUNNABLE"
        )
    )
    checks["feature_sets_dependency_closure_bound"] = bool(
        gate_matches("feature_dependency_closure", dependency_observed)
    )
    no_safe = variants.get("no_safe_smote", {})
    full_variant = variants.get("full_new_reference", {})
    test_logs_observed = bool(
        all(
            "EXIT_CODE=0" in (run_dir / relative).read_text()
            for relative in (
                "logs/syntax_compile.log", "logs/pytest.log",
                "logs/reviewer_snapshot_syntax_compile.log",
                "logs/reviewer_snapshot_pytest.log",
            )
        )
    )
    no_safe_semantics = variant_training_semantics("no_safe_smote")
    safe_smote_isolation_observed = bool(
        test_logs_observed
        and
        no_safe.get("status") == "RUNNABLE"
        and no_safe.get("feature_list_sha256")
        == full_variant.get("feature_list_sha256")
        and no_safe.get("feature_count") == full_variant.get("feature_count")
        and no_safe_semantics.get("safe_smote_enabled") is False
        and no_safe_semantics.get("augmentation_enabled") is True
    )
    checks["safe_smote_isolation_gate_bound"] = gate_matches(
        "safe_smote_isolation", safe_smote_isolation_observed,
    )
    no_pca_validation: dict[str, Any] = {}
    try:
        no_pca_validation = _fresh_no_pca_diagnostic_validation(
            raw_diagnostic, raw_join,
        )
    except Exception:
        pass
    no_pca_machine_checks: dict[str, bool] = {}
    if audit_source_frame is not None:
        no_pca_machine_checks = validate_no_pca_evidence_artifacts(
            raw_diagnostic,
            raw_join,
            raw_source_inventory,
            pre_sources,
            historical_path=HISTORICAL_TRAINING_CSV,
            current_path=CURRENT_TRAINING_CSV,
            audit_frame=audit_source_frame,
        )
    no_pca_diagnostic_observed = bool(
        no_pca_validation.get("status") == "PASS"
        and raw_diagnostic.get("diagnostic_completeness") == no_pca_validation
        and no_pca_machine_checks
        and all(no_pca_machine_checks.values())
    )
    checks["no_pca_diagnostic_and_variant_bound"] = bool(
        gate_matches("no_pca_diagnostic", no_pca_diagnostic_observed)
        and raw_diagnostic.get("status") == no_pca_status
        and no_pca_variant.get("status") == (
            "RUNNABLE" if no_pca_runnable else no_pca_status
        )
    )
    weight_evidence_exact = False
    review_weight_observed = False
    augmentation_observed = False
    if historical_source_frame is not None:
        try:
            (
                weight_evidence_exact,
                review_weight_observed,
                augmentation_observed,
            ) = (
                recompute_review_weight_evidence(
                    weights, historical_source_frame, split, features,
                )
            )
        except Exception:
            review_weight_observed = False
            augmentation_observed = False
    checks["review_weight_and_augmentation_parity_bound"] = bool(
        weight_evidence_exact
        and gate_matches("review_weight_parity", review_weight_observed)
        and gate_matches("augmentation_shuffle_parity", augmentation_observed)
    )
    expected_r92: dict[str, Any] = {}
    if audit_source_frame is not None:
        try:
            expected_r92, _ = score_r92_control(audit_source_frame)
        except Exception:
            expected_r92 = {}
    r92_evidence_exact = bool(r92 == expected_r92)
    r92_observed = bool(r92_evidence_exact and r92.get("status") == "PASS")
    checks["r92_external_control_bound"] = bool(
        r92_evidence_exact
        and gate_matches("r92_external_control", r92_observed)
    )
    try:
        expected_cuda = xgboost_cuda_capability_probe()
    except Exception:
        expected_cuda = {}
    checks["cuda_capability_gate_bound"] = bool(
        report.get("cuda_capability") == cuda == expected_cuda
        and registry.accepted("cuda_capability")
        is (
            expected_cuda.get("status") == "PASS"
            and expected_cuda.get("use_cuda_compiled") is True
        )
    )
    try:
        expected_prior = {
            "prior": verify_prior_run_inventory(),
            "selected_v3": verify_zip(
                SELECTED_V3_ZIP, embedded_run_manifest=True,
            ),
        }
    except Exception:
        expected_prior = {}
    prior_immutable_observed = bool(
        prior_pre == prior_post == expected_prior
        and prior_post.get("prior", {}).get("manifest_status") == "PASS"
        and prior_post.get("prior", {}).get("zip", {}).get("status") == "PASS"
        and prior_post.get("selected_v3", {}).get("status") == "PASS"
    )
    checks["prior_immutable_evidence_bound"] = gate_matches(
        "prior_run_immutability", prior_immutable_observed,
    )
    no_full_observed = bool(
        ledger.get("status") == "PASS"
        and all(ledger.get("checks", {}).values())
        and ledger.get("smoke_report_status") == "ABSENT"
        and ledger.get("official_smoke_variants_completed") == 0
        and ledger.get("exploratory_smoke_variants_completed") == 0
        and ledger.get("full_scientific_run_executed") is False
        and ledger.get("full_variants_completed") == 0
        and ledger.get("stability_executed") is False
        and ledger.get("multiseed_executed") is False
    )
    checks["execution_ledger_no_smoke_full_stability_multiseed"] = bool(
        gate_matches("no_full_scientific_training", no_full_observed)
    )
    staging_path = run_dir / "provenance/package_staging_verification.json"
    if registry.accepted("output_manifest_zip_verification"):
        try:
            staging = json.loads(staging_path.read_text())
        except Exception:
            staging = {}
        package_semantics = staging.get("package_completeness", {}).get(
            "preflight_semantic_completeness", {},
        )
        staging_bundle = Path(str(staging.get(
            "staging_bundle_temporary_path_after_rename", "",
        )))
        expected_staging_name = (
            f"{run_dir.name}_review_bundle_staging_verified.zip"
        )
        staging_common = bool(
            staging.get("crc_status") == "PASS"
            and staging.get("independent_reopen_member_verification") == "PASS"
            and not staging.get("verification_failures")
            and staging.get("staging_bundle_lifecycle")
            == "SUPERSEDED_BY_FINAL_BUNDLE_AND_DELETED_AFTER_FINAL_REOPEN"
            and staging.get("publication_method")
            == "VERIFIED_TEMPORARY_THEN_ATOMIC_NO_CLOBBER_LINK"
            and package_semantics.get("status") == "PASS"
            and package_semantics.get("phase") == "STAGING"
            and staging_bundle.name == expected_staging_name
            and _valid_sha256(staging.get("bundle_sha256"))
            and int(staging.get("bundle_size_bytes", 0)) > 0
            and int(staging.get("member_count", 0)) > 0
            and int(staging.get("uncompressed_size_bytes", 0)) > 0
        )
        staging_live_verified = bool(
            staging_bundle.is_file()
            and not staging_bundle.is_symlink()
            and staging_bundle.parent == run_dir.parent
            and staging_bundle.stat().st_size
            == int(staging.get("bundle_size_bytes", -1))
            and sha256_file(staging_bundle) == staging.get("bundle_sha256")
            and verify_zip(
                staging_bundle, embedded_run_manifest=True,
            ).get("status") == "PASS"
        )
        staging_deleted_after_final_seal = bool(
            not staging_bundle.exists()
            and live_run_manifests_match_content(
                run_dir, run_kind="AMENDED_PREFLIGHT",
            )
        )
        checks["package_gate_bound_to_staging_verification"] = bool(
            staging_common
            and (staging_live_verified or staging_deleted_after_final_seal)
        )
    else:
        checks["package_gate_bound_to_staging_verification"] = bool(
            registry.evidence[
                "output_manifest_zip_verification"
            ].classification.value in {"BLOCKED", "VERIFIED"}
            and not state.smoke_eligible
        )
    checks["test_and_compile_logs_pass"] = gate_matches(
        "expanded_tests", test_logs_observed,
    )

    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise IntegrityError(
            f"Preflight semantic completeness failed checks: {failed}"
        )
    return {
        "status": "PASS",
        "checks": checks,
        "run_state": state.run_state,
        "smoke_eligible": state.smoke_eligible,
        "split_locked": state.split_locked,
        "blocker_codes": list(state.blocker_codes),
        "phase": (
            "FINAL" if (
                run_dir / "provenance/package_staging_verification.json"
            ).is_file() else "STAGING"
        ),
    }


def live_run_manifests_match_content(
    run_dir: Path,
    *,
    run_kind: str,
) -> bool:
    """Recompute both in-Run manifests without trusting their stored rows."""

    try:
        output_path = run_dir / "OUTPUT_MANIFEST_FINAL.tsv"
        bundle_path = run_dir / "BUNDLE_MANIFEST.tsv"
        expected_output = manifest_rows(run_dir, run_kind=run_kind)
        output_bytes = _frame_bytes(expected_output, sep="\t")
        expected_bundle = pd.concat([
            expected_output[["relative_path", "size_bytes", "sha256"]],
            pd.DataFrame([{
                "relative_path": "OUTPUT_MANIFEST_FINAL.tsv",
                "size_bytes": len(output_bytes),
                "sha256": sha256_bytes(output_bytes),
            }]),
        ], ignore_index=True)
        return bool(
            output_path.is_file()
            and not output_path.is_symlink()
            and output_path.read_bytes() == output_bytes
            and bundle_path.is_file()
            and not bundle_path.is_symlink()
            and bundle_path.read_bytes() == _frame_bytes(expected_bundle, sep="\t")
        )
    except Exception:
        return False


_SMOKE_STAGE_SEQUENCE = (
    "raw_locked_post_split_shuffle",
    "post_mixup",
    "post_dropout",
    "post_jitter_pre_smote",
    "post_shuffle",
    "post_safe_smote",
)


def smoke_resampling_parity(
    details: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconstruct the cross-variant augmentation parity evidence from stages."""

    parity: dict[str, Any] = {}
    for variant, detail in details.items():
        stages = detail.get("stages", [])
        by_name = {
            str(stage.get("stage")): stage
            for stage in stages if isinstance(stage, Mapping)
        }
        if {
            "post_jitter_pre_smote", "post_shuffle", "post_safe_smote",
        } <= set(by_name):
            parity[str(variant)] = {
                "post_jitter_pre_smote": by_name["post_jitter_pre_smote"],
                "post_shuffle": by_name["post_shuffle"],
                "post_safe_smote": by_name["post_safe_smote"],
            }
    comparison_inputs = {
        "full_new_reference", "no_review_aware_training_weights", "no_safe_smote",
    }
    if comparison_inputs <= set(parity):
        full = parity["full_new_reference"]
        no_weight = parity["no_review_aware_training_weights"]
        no_smote = parity["no_safe_smote"]
        parity["full_vs_no_weight"] = {
            key: full["post_safe_smote"].get(key)
            == no_weight["post_safe_smote"].get(key)
            for key in (
                "X_sha256", "y_sha256", "source_lineage_sha256",
                "row_order_sha256",
            )
        }
        parity["full_vs_no_safe_smote_pre_smote"] = {
            key: full["post_shuffle"].get(key)
            == no_smote["post_shuffle"].get(key)
            for key in (
                "X_sha256", "y_sha256", "weight_sha256",
                "source_lineage_sha256", "row_order_sha256",
            )
        }
    return parity


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and (number > 0 if positive else number >= 0)


def _actual_booster_configuration(booster: xgb.Booster) -> dict[str, Any]:
    """Return the configuration persisted by the model being verified."""

    return json.loads(booster.save_config())


def _stable_booster_configuration(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Project config to model fields that XGBoost preserves through UBJ reload."""

    learner = configuration.get("learner", {})
    train = learner.get("learner_train_param", {})
    objective = learner.get("objective", {})
    gradient = learner.get("gradient_booster", {})
    return {
        "learner_model_param": learner.get("learner_model_param"),
        "learner_train_param": {
            key: train.get(key)
            for key in ("booster", "multi_strategy", "objective")
        },
        "objective_name": objective.get("name"),
        "gradient_booster_name": gradient.get("name"),
    }


def _active_reloaded_booster_cuda_probe(
    booster: xgb.Booster,
    feature_count: int,
) -> dict[str, Any]:
    """Execute the reloaded model on a CUDA array and reject fallback."""

    try:
        import cupy as cp

        booster.set_param({"device": "cuda"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prediction = booster.inplace_predict(
                cp.zeros((2, feature_count), dtype=cp.float32)
            )
            cp.cuda.Stream.null.synchronize()
        configuration = json.loads(booster.save_config())
        device = str(
            configuration.get("learner", {})
            .get("generic_param", {})
            .get("device", "")
        ).lower()
        warning_text = [str(item.message) for item in caught]
        fallback = any(
            token in value.lower()
            for value in warning_text
            for token in ("fallback", "not compiled", "cpu", "mismatched devices")
        )
        values = cp.asnumpy(prediction)
        passed = bool(
            "cuda" in device
            and not fallback
            and values.shape == (2,)
            and np.isfinite(values).all()
        )
        return {
            "status": "PASS" if passed else "FAIL",
            "device": device,
            "warnings": warning_text,
        }
    except Exception as exc:
        return {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _cuda_classifier_predict_proba(
    classifier: xgb.XGBClassifier,
    matrix: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Predict from a CUDA array and reject any device fallback warning."""

    import cupy as cp

    cuda_matrix = cp.asarray(np.asarray(matrix, dtype=np.float32))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prediction = classifier.predict_proba(cuda_matrix)[:, 1]
        cp.cuda.Stream.null.synchronize()
    warning_text = [str(item.message) for item in caught]
    if any(
        token in value.lower()
        for value in warning_text
        for token in (
            "fallback", "falling back", "mismatched devices", "not compiled",
            "cpu",
        )
    ):
        raise ScientificBlocker(
            "CUDA prediction emitted a CPU/device-fallback warning"
        )
    values = (
        cp.asnumpy(prediction)
        if isinstance(prediction, cp.ndarray)
        else np.asarray(prediction)
    )
    if values.ndim != 1 or len(values) != len(matrix) or not np.isfinite(values).all():
        raise ScientificBlocker("CUDA prediction returned an invalid probability vector")
    return np.asarray(values, dtype=np.float32), warning_text


def _amendment03_model_detail_failures(
    *,
    detail: Mapping[str, Any],
    booster: xgb.Booster,
    expected_attributes: Mapping[str, str],
) -> list[str]:
    """Validate the Corrigendum-only model metadata and four-way parity proof."""

    recovery = detail.get("amendment03_model_recovery")
    if recovery is None:
        return []
    failures: list[str] = []
    parity = detail.get("four_way_reload_parity")
    raw_parity = detail.get("raw_booster_parity")
    compatibility_parity = detail.get("compatibility_classifier_parity")
    expected_paths = {
        "fitted_classifier", "fitted_booster", "reloaded_booster",
        "compatibility_classifier",
    }
    try:
        expected_iteration_range = list(
            amendment03_compat.classifier_equivalent_iteration_range(booster)
        )
    except Exception:
        expected_iteration_range = None
    if not (
        isinstance(recovery, Mapping)
        and recovery.get("classification") == "VERIFIED"
        and recovery.get("status") == "PASS"
        and recovery.get("authorization") == "AMENDMENT_03_CORRIGENDUM_ONE"
        and recovery.get("compatibility_loader")
        == "PROJECT_OWNED_XGB211_BINARY_CLASSIFIER_LOADER"
        and recovery.get("four_way_parity_required") is True
    ):
        failures.append("amendment03_model_recovery_identity")

    if not isinstance(parity, Mapping):
        failures.append("amendment03_four_way_reload_parity")
        return failures
    maximum = parity.get("maximum_absolute_difference")
    bit_exact = parity.get("bit_exact_against_fitted_classifier")
    vector_hashes = parity.get("vector_sha256")
    warning_map = parity.get("warnings")
    loader = parity.get("compatibility_loader")
    model_sha256 = detail.get("model_sha256")
    parity_valid = bool(
        parity.get("classification") == "VERIFIED"
        and parity.get("status") == "PASS"
        and parity.get("device") == "cuda"
        and parity.get("input_dtype") == "float32"
        and parity.get("input_contiguous") is True
        and parity.get("feature_list_sha256")
        == detail.get("feature_list_sha256")
        and parity.get("iteration_range") == expected_iteration_range
        and isinstance(maximum, Mapping)
        and set(maximum) == expected_paths
        and all(float(maximum[path]) == 0.0 for path in expected_paths)
        and isinstance(bit_exact, Mapping)
        and set(bit_exact) == expected_paths
        and all(bit_exact[path] is True for path in expected_paths)
        and isinstance(vector_hashes, Mapping)
        and set(vector_hashes) == expected_paths
        and all(_valid_sha256(vector_hashes[path]) for path in expected_paths)
        and len(set(vector_hashes.values())) == 1
        and isinstance(warning_map, Mapping)
        and set(warning_map) == expected_paths
        and all(
            isinstance(warning_map[path], list)
            and all(isinstance(value, str) for value in warning_map[path])
            and not any(
                token in value.lower()
                for value in warning_map[path]
                for token in (
                    "fallback", "falling back", "mismatched devices",
                    "not compiled", "cpu",
                )
            )
            for path in expected_paths
        )
        and parity.get("cpu_fallback_detected") is False
        and parity.get("model_bytes_unchanged") is True
        and parity.get("model_sha256_before") == model_sha256
        and parity.get("model_sha256_after") == model_sha256
    )
    if not parity_valid:
        failures.append("amendment03_four_way_reload_parity")

    loader_metadata = {
        **_AMENDMENT03_MODEL_METADATA,
        "feature_list_sha256": str(detail.get("feature_list_sha256", "")),
    }
    loader_valid = bool(
        isinstance(loader, Mapping)
        and loader.get("loader") == "XGBClassifier.load_model"
        and loader.get("requested_device") == "cuda"
        and loader.get("objective") == "binary:logistic"
        and int(loader.get("native_num_class", -1)) < 2
        and loader.get("derived_num_classes") == 2
        and loader.get("compatibility_repair_applied") is True
        and loader.get("n_classes_present_before_repair") is False
        and loader.get("n_classes_after_repair") == 2
        and loader.get("repair_scope") == "MISSING_N_CLASSES_ONLY"
        and loader.get("feature_count") == detail.get("feature_count")
        and loader.get("feature_list_sha256")
        == detail.get("feature_list_sha256")
        and loader.get("project_model_metadata") == loader_metadata
        and loader.get("model_sha256_before") == model_sha256
        and loader.get("model_sha256_after") == model_sha256
        and loader.get("model_bytes_unchanged") is True
    )
    if not loader_valid:
        failures.append("amendment03_compatibility_loader")

    raw_valid = bool(
        isinstance(raw_parity, Mapping)
        and raw_parity.get("status") == "PASS"
        and raw_parity.get("iteration_range") == expected_iteration_range
        and raw_parity.get("fitted_booster_bit_exact") is True
        and raw_parity.get("reloaded_booster_bit_exact") is True
        and raw_parity.get("fitted_booster_max_probability_difference") == 0.0
        and raw_parity.get("reloaded_booster_max_probability_difference") == 0.0
        and raw_parity.get("model_bytes_unchanged") is True
    )
    if not raw_valid:
        failures.append("amendment03_raw_booster_parity")
    compatibility_valid = bool(
        isinstance(compatibility_parity, Mapping)
        and compatibility_parity.get("status") == "PASS"
        and compatibility_parity.get("iteration_range") == expected_iteration_range
        and compatibility_parity.get("bit_exact") is True
        and compatibility_parity.get("max_probability_difference") == 0.0
        and compatibility_parity.get("repair_applied") is True
    )
    if not compatibility_valid:
        failures.append("amendment03_compatibility_classifier_parity")

    attributes = booster.attributes()
    mandatory = {
        **_AMENDMENT03_MODEL_METADATA,
        "feature_list_sha256": str(detail.get("feature_list_sha256", "")),
    }
    if (
        any(expected_attributes.get(key) != value for key, value in mandatory.items())
        or any(attributes.get(key) != value for key, value in mandatory.items())
    ):
        failures.append("amendment03_embedded_binary_metadata")
    return failures


def _smoke_variant_detail_failures(
    *,
    variant: str,
    detail: Mapping[str, Any],
    specification: Mapping[str, Any],
    booster: xgb.Booster,
    expected_raw_rows: int,
    run_id: str,
    expected_model_parameters: Mapping[str, Any],
    expected_parameter_lock_sha256: str,
) -> list[str]:
    failures: list[str] = []
    expected_feature_count = len(specification.get("retained_features", []))
    configuration = detail.get("booster_configuration")
    gpu = detail.get("gpu_identity_and_visibility")
    warnings_value = detail.get("warnings")
    stages = detail.get("stages")
    metrics = detail.get("metrics_development_only")
    paired = detail.get("paired_resampling")
    paired_mode = isinstance(paired, Mapping)
    configuration_sha256 = (
        sha256_bytes(canonical_json(configuration))
        if isinstance(configuration, Mapping) else ""
    )
    gpu_sha256 = (
        sha256_bytes(canonical_json(gpu)) if isinstance(gpu, Mapping) else ""
    )
    serialized_configuration = (
        json.dumps(configuration, sort_keys=True).lower()
        if isinstance(configuration, Mapping) else ""
    )
    try:
        actual_configuration = _actual_booster_configuration(booster)
    except Exception:
        actual_configuration = None
    if (
        not isinstance(configuration, Mapping)
        or not isinstance(actual_configuration, Mapping)
        or _stable_booster_configuration(actual_configuration)
        != _stable_booster_configuration(configuration)
        or "cuda" not in serialized_configuration
    ):
        failures.append("booster_cuda_configuration")
    if (
        detail.get("requested_model_parameters") != expected_model_parameters
        or detail.get("model_parameter_lock_sha256")
        != expected_parameter_lock_sha256
        or not booster_matches_parameter_lock(
            configuration or {}, expected_model_parameters,
        )
        or booster.num_boosted_rounds()
        > int(expected_model_parameters.get("n_estimators", -1))
    ):
        failures.append("r92_model_parameter_lock")
    if _active_reloaded_booster_cuda_probe(
        booster, expected_feature_count,
    ).get("status") != "PASS":
        failures.append("active_reloaded_model_cuda_execution")
    if not isinstance(warnings_value, list) or any(
        token in str(value).lower()
        for value in (warnings_value if isinstance(warnings_value, list) else [])
        for token in ("fallback", "not compiled", "cpu")
    ):
        failures.append("warnings_or_fallback")
    if not (
        isinstance(gpu, Mapping)
        and gpu.get("requested_device") == "cuda"
        and gpu.get("xgboost_cuda_compiled") is True
        and gpu.get("nvidia_smi_query_returncode") == 0
        and int(gpu.get("visible_gpu_count_reported_by_nvidia_smi", 0)) > 0
        and isinstance(gpu.get("cuda_visible_devices"), str)
        and isinstance(gpu.get("nvidia_visible_devices"), str)
        and isinstance(gpu.get("devices"), list)
        and bool(gpu.get("devices"))
        and all(
            isinstance(device, Mapping)
            and all(str(device.get(key, "")).strip() for key in (
                "physical_index", "gpu_name", "driver_version",
            ))
            for device in gpu.get("devices", [])
        )
    ):
        failures.append("gpu_identity_schema")
    try:
        resource_relations = bool(
            float(detail["wall_seconds"]) >= float(detail["fit_wall_seconds"])
            and float(detail["cpu_seconds"]) >= float(detail["fit_cpu_seconds"])
            and float(detail["peak_gpu_memory_mib"])
            >= float(detail["baseline_process_gpu_memory_mib"])
            and np.isclose(
                float(detail["peak_gpu_memory_delta_mib"]),
                max(
                    0.0,
                    float(detail["peak_gpu_memory_mib"])
                    - float(detail["baseline_process_gpu_memory_mib"]),
                ),
                rtol=0.0,
                atol=1e-12,
            )
        )
    except (KeyError, TypeError, ValueError):
        resource_relations = False
    if not (
        detail.get("resource_measurement_scope") == "END_TO_END_VARIANT"
        and _finite_number(detail.get("wall_seconds"), positive=True)
        and _finite_number(detail.get("cpu_seconds"))
        and _finite_number(detail.get("fit_wall_seconds"), positive=True)
        and _finite_number(detail.get("fit_cpu_seconds"))
        and (
            not paired_mode
            or _finite_number(detail.get("prediction_wall_seconds"), positive=True)
        )
        and _finite_number(detail.get("peak_ram_bytes"), positive=True)
        and _finite_number(detail.get("baseline_process_gpu_memory_mib"))
        and _finite_number(detail.get("peak_gpu_memory_mib"), positive=True)
        and _finite_number(detail.get("peak_gpu_memory_delta_mib"))
        and resource_relations
    ):
        failures.append("resource_measurements")
    if not (
        isinstance(metrics, Mapping)
        and all(
            _finite_number(metrics.get(key))
            and 0.0 <= float(metrics[key]) <= 1.0
            for key in (
                ("average_precision", "roc_auc", "precision", "recall", "f1")
                if paired_mode else ("average_precision", "roc_auc")
            )
        )
        and (
            not paired_mode
            or (
                int(detail.get("training_rows", 0)) > 0
                and int(detail.get("training_raw_rows", -1)) == expected_raw_rows
                and int(detail.get("validation_rows", 0)) > 0
                and detail.get("requested_device") == "cuda"
                and "cuda" in str(detail.get("observed_device", "")).lower()
                and int(detail.get("tree_count", 0)) > 0
            )
        )
    ):
        failures.append("development_metrics")
    if not isinstance(stages, list) or [
        stage.get("stage") if isinstance(stage, Mapping) else None for stage in stages
    ] != list(_SMOKE_STAGE_SEQUENCE):
        failures.append("stage_sequence")
    else:
        previous_rows = 0
        for index, stage in enumerate(stages):
            required_hashes = (
                "X_sha256", "y_sha256", "weight_sha256",
                "source_lineage_sha256", "row_order_sha256",
            )
            rows = int(stage.get("rows", -1))
            if (
                rows <= 0
                or rows != int(stage.get("negative", -1)) + int(stage.get("positive", -1))
                or int(stage.get("features", -1))
                != (93 if paired_mode else expected_feature_count)
                or rows < previous_rows
                or not all(_valid_sha256(stage.get(key)) for key in required_hashes)
            ):
                failures.append(f"stage_evidence:{stage.get('stage', index)}")
            previous_rows = rows
        if int(stages[0].get("rows", -1)) != expected_raw_rows:
            failures.append("stage_raw_row_count")
        expected_smote_enabled = variant != "no_safe_smote"
        shuffle_stage = stages[-2]
        if not (
            shuffle_stage.get("shuffle_applied") is False
            and shuffle_stage.get("rule") == "NO_POST_AUGMENTATION_SHUFFLE"
            and all(
                shuffle_stage.get(key) == stages[-3].get(key)
                for key in (
                    "rows", "features", "negative", "positive", "X_sha256",
                    "y_sha256", "weight_sha256", "source_lineage_sha256",
                    "row_order_sha256",
                )
            )
        ):
            failures.append("post_shuffle_noop_evidence")
        final_stage = stages[-1]
        synthetic_rows = int(final_stage.get("rows", -1)) - int(stages[-2].get("rows", -1))
        if (
            final_stage.get("safe_smote_enabled") is not expected_smote_enabled
            or int(final_stage.get("safe_smote_synthetic_rows", -1)) != synthetic_rows
            or (not expected_smote_enabled and synthetic_rows != 0)
        ):
            failures.append("safe_smote_stage")
    if paired_mode:
        projection = paired.get("projection", {})
        projection_rows = (
            int(stages[-2].get("rows", -1))
            if variant == "no_safe_smote" and isinstance(stages, list)
            and len(stages) == len(_SMOKE_STAGE_SEQUENCE)
            else int(stages[-1].get("rows", -1))
            if isinstance(stages, list) and stages else -1
        )
        if not (
            paired.get("status") == "PASS"
            and paired.get("generation_scope") == "ONE_FULL_SPACE_POPULATION"
            and _valid_sha256(paired.get("frozen_parity_sha256"))
            and isinstance(projection, Mapping)
            and projection.get("variant") == variant
            and projection.get("projection_only") is True
            and int(projection.get("features", -1)) == expected_feature_count
            and int(projection.get("rows", -1)) == projection_rows
            and all(
                _valid_sha256(projection.get(key))
                for key in (
                    "feature_order_sha256", "X_sha256", "y_sha256",
                    "weight_sha256", "source_lineage_sha256",
                    "row_order_sha256", "full_space_source_X_sha256",
                )
            )
            and projection.get("safe_smote_enabled")
            is (variant != "no_safe_smote")
            and int(projection.get("safe_smote_synthetic_rows", -1))
            == (0 if variant == "no_safe_smote" else synthetic_rows)
            and int(detail.get("safe_smote_synthetic_rows", -1))
            == int(projection.get("safe_smote_synthetic_rows", -2))
            and detail.get("variant_classification")
            == ("EXPLORATORY" if variant == EXPLORATORY_VARIANT else "OFFICIAL")
            and isinstance(detail.get("best_iteration"), int)
            and 0 <= int(detail.get("best_iteration"))
            < int(detail.get("tree_count", 0))
        ):
            failures.append("paired_frozen_projection")
    attributes = booster.attributes()
    expected_attributes = {
        "amendment_classification": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
        "amendment_device": "cuda",
        "amendment_variant": variant,
        "amendment_run_id": run_id,
        "booster_configuration_sha256": configuration_sha256,
        "feature_list_sha256": str(specification.get("feature_list_sha256", "")),
        "gpu_identity_sha256": gpu_sha256,
        "model_parameter_lock_sha256": expected_parameter_lock_sha256,
    }
    if detail.get("amendment03_model_recovery") is not None:
        expected_attributes.update(_AMENDMENT03_MODEL_METADATA)
    if (
        detail.get("embedded_model_attributes") != expected_attributes
        or any(attributes.get(key) != value for key, value in expected_attributes.items())
    ):
        failures.append("embedded_model_attributes")
    failures.extend(_amendment03_model_detail_failures(
        detail=detail,
        booster=booster,
        expected_attributes=expected_attributes,
    ))
    return failures


def verify_smoke_model_bundle_semantics(
    verification: Mapping[str, Any],
    completed_variant_ids: set[str],
    expected_model_details: Mapping[str, Mapping[str, Any]],
    variant_specs: Mapping[str, Mapping[str, Any]],
    *,
    run_dir: Path,
    expected_raw_rows: int,
    model_parameter_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen, load, and bind each Smoke model to its variant evidence."""

    filename = str(verification.get("filename", ""))
    portable_reference = str(verification.get("portable_relative_to_run", ""))
    path = run_dir.parent / filename
    expected_model_paths = {f"{variant}.ubj" for variant in completed_variant_ids}
    expected_filename = f"{run_dir.name}_NON_SCIENTIFIC_smoke_models.zip"
    expected_archive_root = f"{run_dir.name}_smoke_models"
    failures: list[str] = []
    expected_model_parameters = model_parameter_lock.get(
        "smoke_classifier_parameters", {}
    )
    expected_parameter_lock_sha256 = model_parameter_lock.get(
        "smoke_classifier_parameters_sha256", ""
    )
    if (
        verification.get("classification") != "NON_SCIENTIFIC_DIAGNOSTIC_ONLY"
        or verification.get("crc_status") != "PASS"
        or verification.get("reopen_member_hash_status") != "PASS"
        or not filename
        or filename != expected_filename
        or portable_reference != f"../{filename}"
        or Path(str(verification.get("path", ""))).name != filename
        or verification.get("run_id") != run_dir.name
        or verification.get("archive_root") != expected_archive_root
        or set(verification.get("completed_variant_ids", []))
        != completed_variant_ids
        or not isinstance(verification.get("excluded_uncompleted_artifacts"), list)
        or not path.is_file()
        or path.is_symlink()
        or int(verification.get("size_bytes", -1)) != path.stat().st_size
        or sha256_file(path) != verification.get("sha256")
        or set(expected_model_details) != completed_variant_ids
        or not completed_variant_ids <= set(variant_specs)
        or model_parameter_lock.get("classification") != "VERIFIED"
        or model_parameter_lock != r92_model_parameter_lock()
        or not isinstance(expected_model_parameters, Mapping)
        or not _valid_sha256(expected_parameter_lock_sha256)
    ):
        failures.append("outer_verification")
        return {"status": "FAIL", "failures": failures}
    try:
        with zipfile.ZipFile(path) as archive:
            bad_crc = archive.testzip()
            names = archive.namelist()
            if bad_crc is not None:
                failures.append(f"crc:{bad_crc}")
            if len(names) != len(set(names)):
                failures.append("duplicate_members")
            roots = {name.split("/", 1)[0] for name in names if "/" in name}
            if len(roots) != 1 or any(
                name.startswith("/") or ".." in Path(name).parts or name.endswith("/")
                for name in names
            ):
                failures.append("unsafe_or_multiple_roots")
            root = next(iter(roots), "")
            if verification.get("archive_root") != root:
                failures.append("archive_root")
            manifest_name = f"{root}/BUNDLE_MANIFEST.tsv"
            if manifest_name not in names:
                failures.append("missing_internal_manifest")
                manifest = pd.DataFrame()
            else:
                manifest_bytes = archive.read(manifest_name)
                manifest = pd.read_csv(io.BytesIO(manifest_bytes), sep="\t")
            required_columns = {"relative_path", "size_bytes", "sha256"}
            if not required_columns <= set(manifest):
                failures.append("internal_manifest_schema")
            else:
                relative_paths = manifest.relative_path.astype(str)
                if relative_paths.duplicated().any():
                    failures.append("internal_manifest_duplicates")
                if set(relative_paths) != expected_model_paths:
                    failures.append("variant_member_set")
                expected_names = {
                    f"{root}/{relative}" for relative in relative_paths
                } | {manifest_name}
                if set(names) != expected_names:
                    failures.append("archive_member_set")
                for row in manifest.itertuples(index=False):
                    data = archive.read(f"{root}/{row.relative_path}")
                    member_sha256 = sha256_bytes(data)
                    if (
                        len(data) != int(row.size_bytes)
                        or member_sha256 != row.sha256
                    ):
                        failures.append(str(row.relative_path))
                    variant = Path(str(row.relative_path)).stem
                    detail = expected_model_details.get(variant, {})
                    specification = variant_specs.get(variant, {})
                    if detail.get("model_sha256") != row.sha256:
                        failures.append(f"metrics_model_sha:{variant}")
                    expected_feature_count = len(
                        specification.get("retained_features", [])
                    )
                    if (
                        detail.get("classification")
                        != "NON_SCIENTIFIC_DIAGNOSTIC_ONLY"
                        or detail.get("save_reload_max_probability_difference") != 0.0
                        or detail.get("external_audit_scored") is not False
                        or int(detail.get("feature_count", -1))
                        != expected_feature_count
                        or detail.get("feature_list_sha256")
                        != specification.get("feature_list_sha256")
                    ):
                        failures.append(f"metrics_model_evidence:{variant}")
                    try:
                        booster = xgb.Booster()
                        booster.load_model(bytearray(data))
                        if (
                            booster.num_boosted_rounds() <= 0
                            or booster.num_features() != expected_feature_count
                        ):
                            failures.append(f"model_shape:{variant}")
                        failures.extend(
                            f"{failure}:{variant}" for failure in
                            _smoke_variant_detail_failures(
                                variant=variant,
                                detail=detail,
                                specification=specification,
                                booster=booster,
                                expected_raw_rows=expected_raw_rows,
                                run_id=run_dir.name,
                                expected_model_parameters=expected_model_parameters,
                                expected_parameter_lock_sha256=(
                                    expected_parameter_lock_sha256
                                ),
                            )
                        )
                    except Exception:
                        failures.append(f"model_load:{variant}")
            if int(verification.get("member_count", -1)) != len(names):
                failures.append("outer_member_count")
    except Exception as exc:
        failures.append(f"unreadable:{type(exc).__name__}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "completed_variant_ids": sorted(completed_variant_ids),
        "expected_model_paths": sorted(expected_model_paths),
    }


def validate_smoke_semantic_completeness(
    run_dir: Path,
    identity: Mapping[str, Any],
    *,
    amendment06_allowed_after_hashes: Mapping[str, str] | None = None,
    amendment06_change_ledger_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject structurally present but semantically fabricated Smoke evidence."""

    if (amendment06_allowed_after_hashes is None) != (
        amendment06_change_ledger_evidence is None
    ):
        raise IntegrityError(
            "Amendment 06 code map and ledger evidence must be supplied together"
        )
    amendment06_code_chain: dict[str, Any] | None = None
    if amendment06_allowed_after_hashes is not None:
        amendment06_code_chain = validate_amendment06_reference_code_chain(
            allowed_after_hashes=amendment06_allowed_after_hashes,
            ledger_evidence=amendment06_change_ledger_evidence or {},
            study_root=STUDY_ROOT,
        )

    try:
        marker = (run_dir / "NON_SCIENTIFIC_DIAGNOSTIC_ONLY.txt").read_text()
        reference = json.loads((run_dir / "config/preflight_reference.json").read_text())
        variants = json.loads((run_dir / "config/variant_feature_sets.json").read_text())
        training_configuration = json.loads(
            (run_dir / "config/training_configuration.lock.json").read_text()
        )
        registry_payload = json.loads(
            (run_dir / "provenance/evidence_gate_registry.json").read_text()
        )
        ledger = json.loads((run_dir / "provenance/execution_ledger.json").read_text())
        metrics = json.loads((run_dir / "metrics/SMOKE_REPORT.json").read_text())
        subset = json.loads((run_dir / "config/smoke_subset.json").read_text())
        table = amendment04_recovery.read_smoke_report_roundtrip(
            run_dir / "tables/SMOKE_REPORT.csv"
        )
        status_path = run_dir / "RUN_STATUS.txt"
        status_text = status_path.read_text()
        status = _status_fields(status_path)
        status_keys = [
            line.partition("=")[0].strip()
            for line in status_text.splitlines() if "=" in line
        ]
    except Exception as exc:
        raise IntegrityError(
            f"Smoke semantic evidence is unreadable: {type(exc).__name__}: {exc}"
        ) from exc

    checks: dict[str, bool] = {}
    checks["non_scientific_marker"] = (
        "non-scientific" in marker.lower() and "diagnostic" in marker.lower()
    )
    try:
        reviewed_split = pd.read_csv(run_dir / "splits/row_split_manifest.csv.gz")
        _, _, expected_subset = deterministic_smoke_subset(reviewed_split)
        subset_exact = canonical_json(subset) == canonical_json(expected_subset)
    except Exception:
        subset_exact = False
    checks["deterministic_subset_recomputed_exactly"] = subset_exact
    reference_validation = reference.get("reference_validation") or {}
    checks["preflight_reference_pinned"] = bool(
        reference.get("classification") == "VERIFIED"
        and reference.get("preflight_run_id") == identity.get("preflight_run_id")
        and reference.get("split_hash") == identity.get("candidate_split_hash")
        and reference_validation.get("required_files_status") == "PASS"
        and reference_validation.get("status_identity_match") == "PASS"
        and reference_validation.get("live_file_set_status") == "PASS"
        and reference_validation.get("bundle_manifest_status") == "PASS"
        and reference_validation.get("review_bundle_live_byte_equality_status")
        == "PASS"
        and reference_validation.get("review_bundle_sha256")
        == reference.get("preflight_review_bundle_sha256")
        and reference_validation.get("split_hash_status") == "PASS"
        and re.fullmatch(r"[0-9a-f]{64}", str(reference.get("preflight_identity_sha256", "")))
        and re.fullmatch(r"[0-9a-f]{64}", str(reference.get("preflight_registry_sha256", "")))
        and re.fullmatch(r"[0-9a-f]{64}", str(reference.get("preflight_review_bundle_sha256", "")))
        and reference.get("preflight_review_bundle_verification", {}).get("status") == "PASS"
    )
    copied = reference.get("copied_evidence_sha256", {})
    protocol = str(identity.get("protocol", "AMENDMENT_01"))
    amendment03_replacement = (
        identity.get("recovery_authorization") == _AMENDMENT03_AUTHORIZATION
    )
    amendment04_recovery_run = (
        identity.get("recovery_authorization") == _AMENDMENT04_AUTHORIZATION
    )
    compatibility_recovery = amendment03_replacement or amendment04_recovery_run
    copied_mapping = smoke_copied_evidence_mapping(protocol)
    expected_copied_files = set(copied_mapping.values())
    if compatibility_recovery:
        expected_copied_files.add(amendment03_compat.ROOT_PROBE_DESTINATION)
        expected_copied_files.discard(
            "provenance/preflight_snapshot/preflight_review_bundle.zip"
        )
    packaged_preflight_relative = (
        "provenance/preflight_snapshot/preflight_review_bundle.zip"
    )
    checks["copied_preflight_evidence_hashes"] = bool(
        isinstance(copied, dict)
        and set(copied) == expected_copied_files
        and all(
            (run_dir / relative).is_file()
            and sha256_file(run_dir / relative) == expected
            for relative, expected in copied.items()
        )
        and copied.get("provenance/preflight_snapshot/run_identity.lock.json")
        == reference.get("preflight_identity_sha256")
        and copied.get("provenance/preflight_snapshot/evidence_gate_registry.json")
        == reference.get("preflight_registry_sha256")
        and copied.get("provenance/preflight_snapshot/OUTPUT_MANIFEST_FINAL.tsv")
        == reference.get("preflight_output_manifest_sha256")
        and copied.get("provenance/preflight_snapshot/BUNDLE_MANIFEST.tsv")
        == reference.get("preflight_bundle_manifest_sha256")
        and (
            compatibility_recovery
            or copied.get(packaged_preflight_relative)
            == reference.get("preflight_review_bundle_sha256")
        )
    )
    packaged_zip = run_dir / packaged_preflight_relative
    if compatibility_recovery:
        checks["packaged_preflight_review_zip_matches_pin"] = bool(
            reference.get("packaged_preflight_review_bundle_path")
            == (
                "NOT_BUNDLED_AMENDMENT04_LINEAGE_ONLY"
                if amendment04_recovery_run
                else "NOT_BUNDLED_AMENDMENT03_RECOVERY_LINEAGE_ONLY"
            )
            and not packaged_zip.exists()
            and reference.get("preflight_review_bundle_sha256")
            == amendment03_recovery.ACCEPTED_PREFLIGHT_ZIP_SHA256
        )
    else:
        checks["packaged_preflight_review_zip_matches_pin"] = bool(
            reference.get("packaged_preflight_review_bundle_path")
            == packaged_preflight_relative
            and packaged_zip.is_file()
            and packaged_zip.stat().st_size
            == int(reference.get("preflight_review_bundle_size_bytes", -1))
            and sha256_file(packaged_zip)
            == reference.get("preflight_review_bundle_sha256")
            and verify_zip(packaged_zip, embedded_run_manifest=True).get("status") == "PASS"
        )
    copied_member_equality = False
    source_preflight_zip = (
        Path(str(reference.get("preflight_review_bundle_path", "")))
        if compatibility_recovery else packaged_zip
    )
    if checks["packaged_preflight_review_zip_matches_pin"]:
        try:
            with zipfile.ZipFile(source_preflight_zip) as archive:
                prefix = f"{reference['preflight_run_id']}/"
                copied_member_equality = all(
                    (source == "__PREFLIGHT_REVIEW_BUNDLE__" and compatibility_recovery)
                    or source == "__PREFLIGHT_REVIEW_BUNDLE__"
                    or archive.read(f"{prefix}{source}")
                    == (run_dir / destination).read_bytes()
                    for source, destination in copied_mapping.items()
                    if not (
                        compatibility_recovery
                        and source == "__PREFLIGHT_REVIEW_BUNDLE__"
                    )
                )
                if compatibility_recovery:
                    sealed_probe = archive.read(
                        f"{prefix}{amendment03_compat.SEALED_PROBE_RELATIVE_PATH}"
                    )
                    copied_member_equality = bool(
                        copied_member_equality
                        and sealed_probe
                        == (run_dir / amendment03_compat.ROOT_PROBE_DESTINATION).read_bytes()
                        == (run_dir / amendment03_compat.NESTED_PROBE_DESTINATION).read_bytes()
                    )
        except Exception:
            copied_member_equality = False
    checks["direct_copies_equal_packaged_preflight_members"] = copied_member_equality

    if amendment03_replacement:
        reviewed_code_tree = reference.get("reviewed_code_tree", {})
        allowed_after_hashes = reference.get(
            "amendment03_allowed_after_hashes", {}
        )
        try:
            recovery_validation = amendment03_recovery.validate_recovery_artifacts(
                run_dir=run_dir,
                reviewed_code_tree=reviewed_code_tree,
                allowed_after_hashes=allowed_after_hashes,
                study_root=STUDY_ROOT,
                require_prompt_artifacts=True,
            )
            checks["amendment03_recovery_artifacts"] = bool(
                recovery_validation.get("status") == "PASS"
                and reference.get("recovery_authorization")
                == _AMENDMENT03_AUTHORIZATION
                and set(allowed_after_hashes)
                == set(_AMENDMENT03_ALLOWED_CODE_DRIFT_PATHS)
            )
        except Exception:
            checks["amendment03_recovery_artifacts"] = False
        prerun_hashes = reference.get("amendment03_prerun_evidence_sha256", {})
        checks["amendment03_prerun_evidence_exact"] = bool(
            isinstance(prerun_hashes, Mapping)
            and set(prerun_hashes) == set(_AMENDMENT03_PRERUN_COPY_MAPPING.values())
            and all(
                (run_dir / relative).is_file()
                and not (run_dir / relative).is_symlink()
                and sha256_file(run_dir / relative) == digest
                for relative, digest in prerun_hashes.items()
            )
        )
    if amendment04_recovery_run:
        reviewed_code_tree = reference.get("reviewed_code_tree", {})
        allowed_after_hashes = reference.get(
            "amendment04_allowed_after_hashes", {}
        )
        try:
            recovery_validation = (
                validate_sealed_a04_execution_for_amendment06(
                    run_dir,
                    identity,
                    study_root=STUDY_ROOT,
                )
                if amendment06_code_chain is not None
                else amendment04_recovery.validate_recovery_artifacts(
                    run_dir=run_dir,
                    allowed_after_hashes=allowed_after_hashes,
                    study_root=STUDY_ROOT,
                    require_prompt_artifacts=True,
                )
            )
            checks["amendment04_recovery_artifacts"] = bool(
                recovery_validation.get("status") == "PASS"
                and reference.get("recovery_authorization")
                == _AMENDMENT04_AUTHORIZATION
            )
        except Exception:
            checks["amendment04_recovery_artifacts"] = False
        prerun_hashes = reference.get("amendment04_prerun_evidence_sha256", {})
        checks["amendment04_prerun_evidence_exact"] = bool(
            isinstance(prerun_hashes, Mapping)
            and set(prerun_hashes)
            == set(amendment04_recovery.AMENDMENT04_PRERUN_DESTINATIONS)
            and all(
                (run_dir / relative).is_file()
                and not (run_dir / relative).is_symlink()
                and sha256_file(run_dir / relative) == digest
                for relative, digest in prerun_hashes.items()
            )
        )

    gate_inputs: dict[str, dict[str, Any]] = {}
    stored_gates = registry_payload.get("gates", {})
    for specification in evidence_workflow.DEFAULT_GATE_SPECS:
        stored = stored_gates.get(specification.key, {})
        gate_inputs[specification.key] = {
            "classification": stored.get("classification"),
            "passed": stored.get("passed"),
            "evidence_refs": stored.get("evidence_refs"),
            "detail": stored.get("detail", ""),
        }
    try:
        registry = evidence_workflow.build_gate_registry(gate_inputs)
    except Exception as exc:
        raise IntegrityError(f"Smoke registry is invalid: {exc}") from exc
    checks["registry_tokens_recomputed"] = all(
        stored_gates[key].get("accepted") is gate.accepted
        and stored_gates[key].get("status") == gate.status_token
        for key, gate in registry.evidence.items()
    )
    checks["non_scientific_global_typing"] = bool(
        registry_payload.get("classification") == "VERIFIED"
        and metrics.get("classification") == "NON_SCIENTIFIC_DIAGNOSTIC_ONLY"
        and metrics.get("external_audit_scored") is False
        and status.get("CLASSIFICATION") == "NON_SCIENTIFIC_DIAGNOSTIC_ONLY"
        and status.get("EXTERNAL_AUDIT_SCORED") == "NO"
    )
    checks["registry_evidence_refs_exist"] = all(
        (run_dir / relative).is_file()
        or (
            gate.key == "output_manifest_zip_verification"
            and gate.classification.value == "BLOCKED"
            and relative == "provenance/package_staging_verification.json"
        )
        for gate in registry.evidence.values()
        for relative in gate.evidence_refs
    )

    no_pca_runnable = variants.get("no_pca", {}).get("status") == "RUNNABLE"
    try:
        facts = evidence_workflow.WorkflowFacts(
            phase="CUDA_SMOKE",
            official_smoke_variants_completed=int(
                ledger["official_smoke_variants_completed"]
            ),
            exploratory_smoke_variants_completed=int(
                ledger["exploratory_smoke_variants_completed"]
            ),
            full_scientific_run_executed=ledger["full_scientific_run_executed"],
            full_variants_completed=int(ledger["full_variants_completed"]),
            no_pca_variant_status=str(metrics["true_no_pca_status"]),
            no_pca_variant_runnable=no_pca_runnable,
            protocol_amendment=str(
                identity.get("protocol", "AMENDMENT_01")
            ),
        )
        state = evidence_workflow.derive_workflow_state(registry, facts)
    except Exception as exc:
        raise IntegrityError(f"Smoke workflow facts are invalid: {exc}") from exc

    derived = registry_payload.get("derived", {})
    checks["derived_registry_state"] = derived == {
        "run_kind": state.run_kind,
        "run_state": state.run_state,
        "cuda_smoke_status": state.cuda_smoke_status,
        "ready_for_full_awaiting_external_review": (
            state.ready_for_full_awaiting_external_review
        ),
        "blockers": list(state.blocker_codes),
    }
    checks["identity_matches_derived_state"] = bool(
        identity.get("run_kind") == state.run_kind
        and identity.get("state") == state.run_state
        and identity.get("full_authorized") is False
        and identity.get("stability_authorized") is False
    )

    protocol = str(identity.get("protocol", "AMENDMENT_01"))
    required_table_columns = {
        "variant", "status", "feature_count", "average_precision", "roc_auc",
        "wall_seconds", "cpu_seconds", "peak_ram_bytes",
        "baseline_process_gpu_memory_mib", "peak_gpu_memory_mib",
        "peak_gpu_memory_delta_mib",
    }
    if protocol == "AMENDMENT_02":
        required_table_columns.update({
            "precision", "recall", "f1", "training_rows", "validation_rows",
            "prediction_wall_seconds", "best_iteration", "tree_count",
        })
    table_valid = required_table_columns <= set(table)
    completed_ids = set(
        table.loc[table.status.astype(str).str.upper() == "PASS", "variant"].astype(str)
    ) if table_valid else set()
    runnable_official = set(runnable_official_variants(variants))
    expected_ids = runnable_official | {EXPLORATORY_VARIANT}
    checks["smoke_table_ids_and_counts"] = bool(
        table_valid
        and not table.variant.astype(str).duplicated().any()
        and set(table.variant.astype(str)) <= expected_ids
        and completed_ids == set(ledger.get("smoke_variants_completed_ids", []))
        and len(completed_ids & runnable_official)
        == int(ledger["official_smoke_variants_completed"])
        and int(EXPLORATORY_VARIANT in completed_ids)
        == int(ledger["exploratory_smoke_variants_completed"])
        and int(metrics["official_variants_completed"])
        == int(ledger["official_smoke_variants_completed"])
        and int(metrics["official_variants_expected"])
        == facts.official_smoke_variants_expected
        and int(metrics["exploratory_variants_completed"])
        == int(ledger["exploratory_smoke_variants_completed"])
        and set(metrics.get("details", {})) == completed_ids
    )
    expected_variant_order = [
        *runnable_official_variants(variants), EXPLORATORY_VARIANT,
    ]
    observed_variant_order = (
        table.loc[
            table.status.astype(str).str.upper() == "PASS", "variant"
        ].astype(str).tolist()
        if table_valid else []
    )
    checks["smoke_table_order"] = bool(
        observed_variant_order
        == expected_variant_order[:len(observed_variant_order)]
        and (
            identity.get("state") != "SMOKE_COMPLETE"
            or observed_variant_order == expected_variant_order
        )
    )
    if compatibility_recovery:
        checks["compatibility_recovery_identity_and_counts"] = bool(
            identity.get("recovery_authorization")
            in {_AMENDMENT03_AUTHORIZATION, _AMENDMENT04_AUTHORIZATION}
            and identity.get("preflight_run_id")
            == amendment03_recovery.ACCEPTED_PREFLIGHT_RUN_ID
            and identity.get("candidate_split_hash")
            == amendment03_recovery.LOCKED_SPLIT_SHA256
            and (
                identity.get("state") == "SMOKE_INCOMPLETE"
                or (
                    identity.get("state") == "SMOKE_COMPLETE"
                    and completed_ids == expected_ids
                    and len(completed_ids & runnable_official) == 9
                    and EXPLORATORY_VARIANT in completed_ids
                )
            )
        )
        checks["completed_models_have_compatibility_recovery_parity"] = all(
            isinstance(metrics.get("details", {}).get(variant), Mapping)
            and metrics["details"][variant].get(
                "amendment03_model_recovery", {}
            ).get("authorization") == _AMENDMENT03_AUTHORIZATION
            and metrics["details"][variant].get(
                "amendment03_model_recovery", {}
            ).get("status") == "PASS"
            and metrics["details"][variant].get(
                "four_way_reload_parity", {}
            ).get("status") == "PASS"
            and metrics["details"][variant].get(
                "four_way_reload_parity", {}
            ).get("cpu_fallback_detected") is False
            and metrics["details"][variant].get(
                "raw_booster_parity", {}
            ).get("status") == "PASS"
            and metrics["details"][variant].get(
                "compatibility_classifier_parity", {}
            ).get("status") == "PASS"
            for variant in completed_ids
        )
    if amendment04_recovery_run:
        lineage_path = run_dir / "provenance/AMENDMENT04_LINEAGE.json"
        lineage_sha256 = (
            sha256_file(lineage_path)
            if lineage_path.is_file() and not lineage_path.is_symlink()
            else ""
        )
        checks["amendment04_identity_and_status"] = bool(
            identity.get("recovery_authorization") == _AMENDMENT04_AUTHORIZATION
            and identity.get("amendment03_authorization") == "1_OF_1_CONSUMED"
            and identity.get("replacement_run_authorization") == "1_OF_1_CONSUMED"
            and identity.get("amendment04_lineage_sha256") == lineage_sha256
            and identity.get("full_training_executed") is False
            and identity.get("scientific_results_claimed") is False
            and status.get("RECOVERY_AUTHORIZATION") == _AMENDMENT04_AUTHORIZATION
            and status.get("AMENDMENT03_AUTHORIZATION") == "1_OF_1_CONSUMED"
            and status.get("AMENDMENT04_FRESH_RUN_AUTHORIZATION")
            == "1_OF_1_CONSUMED"
            and status.get("CSV_READER_REPAIR")
            == "PANDAS_C_ENGINE_FLOAT_PRECISION_ROUND_TRIP"
            and status.get("CSV_DEFAULT_MISMATCH_COUNT") == "22"
            and status.get("CSV_ROUND_TRIP_MISMATCH_COUNT") == "0"
            and status.get("AMENDMENT04_LINEAGE_SHA256") == lineage_sha256
        )
    checks["smoke_table_matches_variant_details"] = bool(
        table_valid
        and all(
            int(row.feature_count)
            == int(metrics["details"][str(row.variant)].get("feature_count", -1))
            and float(row.average_precision)
            == float(
                metrics["details"][str(row.variant)]
                .get("metrics_development_only", {})
                .get("average_precision", float("nan"))
            )
            and float(row.roc_auc)
            == float(
                metrics["details"][str(row.variant)]
                .get("metrics_development_only", {})
                .get("roc_auc", float("nan"))
            )
            and (
                protocol != "AMENDMENT_02"
                or (
                    all(
                        float(getattr(row, key))
                        == float(
                            metrics["details"][str(row.variant)]
                            .get("metrics_development_only", {})
                            .get(key, float("nan"))
                        )
                        for key in ("precision", "recall", "f1")
                    )
                    and int(row.training_rows) == int(
                        metrics["details"][str(row.variant)].get(
                            "training_rows", -1
                        )
                    )
                    and int(row.validation_rows) == int(
                        metrics["details"][str(row.variant)].get(
                            "validation_rows", -1
                        )
                    )
                    and int(row.tree_count) == int(
                        metrics["details"][str(row.variant)].get("tree_count", -1)
                    )
                    and np.isclose(
                        float(row.prediction_wall_seconds),
                        float(metrics["details"][str(row.variant)].get(
                            "prediction_wall_seconds", float("nan")
                        )),
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
            )
            and all(
                np.isclose(
                    float(getattr(row, key)),
                    float(metrics["details"][str(row.variant)].get(
                        key, float("nan")
                    )),
                    rtol=0.0,
                    atol=1e-12,
                )
                for key in (
                    "wall_seconds", "cpu_seconds", "peak_ram_bytes",
                    "baseline_process_gpu_memory_mib",
                    "peak_gpu_memory_mib", "peak_gpu_memory_delta_mib",
                )
            )
            for row in table.itertuples(index=False)
            if str(row.status).upper() == "PASS"
        )
    )
    stored_parity = metrics.get("parity")
    if str(identity.get("protocol", "AMENDMENT_01")) == "AMENDMENT_02":
        try:
            parity_artifact = json.loads(
                (run_dir / "provenance/frozen_resampling_parity.json").read_text()
            )
        except Exception:
            parity_artifact = {}
        comparisons = parity_artifact.get("comparisons", {})
        full_weight_checks = comparisons.get("full_vs_no_weight", {})
        no_smote_checks = comparisons.get(
            "full_vs_no_safe_smote_pre_smote", {}
        )
        post_checks = comparisons.get("post_smote_variant_projections", {})
        expected_post_ids = expected_ids - {"no_safe_smote"}
        parity_sha = sha256_bytes(canonical_json(parity_artifact))
        parity_stages = parity_artifact.get("stages", [])
        stage_counts = parity_artifact.get("stage_counts", {})
        stage_by_name = {
            str(stage.get("stage")): stage
            for stage in parity_stages if isinstance(stage, Mapping)
        }
        population_keys = {
            "rows", "features", "negative", "positive",
            "feature_order_sha256", "X_sha256", "y_sha256",
            "weight_sha256", "source_lineage_sha256", "row_order_sha256",
        }
        pre_smote = parity_artifact.get("pre_smote", {})
        post_smote = parity_artifact.get("post_smote", {})
        variants_evidence = parity_artifact.get("variants", {})
        full_evidence = variants_evidence.get("full_new_reference", {})
        no_weight_evidence = variants_evidence.get(
            "no_review_aware_training_weights", {}
        )
        no_smote_evidence = variants_evidence.get("no_safe_smote", {})
        detail_stages_bound = all(
            (
                canonical_json(detail.get("stages", []))
                == canonical_json(parity_stages)
            )
            if variant != "no_safe_smote"
            else (
                canonical_json(detail.get("stages", [])[:-1])
                == canonical_json(parity_stages[:-1])
                and len(detail.get("stages", [])) == len(_SMOKE_STAGE_SEQUENCE)
                and detail.get("stages", [])[-1].get("stage")
                == "post_safe_smote"
                and detail.get("stages", [])[-1].get("safe_smote_enabled")
                is False
                and detail.get("stages", [])[-1].get(
                    "safe_smote_synthetic_rows"
                ) == 0
                and all(
                    detail.get("stages", [])[-1].get(key)
                    == pre_smote.get(key)
                    for key in population_keys
                )
            )
            for variant, detail in metrics.get("details", {}).items()
        )
        stage_structure_exact = bool(
            [stage.get("stage") for stage in parity_stages]
            == list(_SMOKE_STAGE_SEQUENCE)
            and set(stage_counts) == set(_SMOKE_STAGE_SEQUENCE)
            and all(
                stage_counts.get(name) == {
                    "rows": int(stage_by_name[name].get("rows", -1)),
                    "negative": int(stage_by_name[name].get("negative", -1)),
                    "positive": int(stage_by_name[name].get("positive", -1)),
                }
                for name in _SMOKE_STAGE_SEQUENCE
            )
            and all(
                pre_smote.get(key)
                == stage_by_name["post_jitter_pre_smote"].get(key)
                == stage_by_name["post_shuffle"].get(key)
                for key in population_keys
            )
            and all(
                post_smote.get(key)
                == stage_by_name["post_safe_smote"].get(key)
                for key in population_keys
            )
            and full_evidence.get("full_space_source_X_sha256")
            == post_smote.get("X_sha256")
            and no_weight_evidence.get("full_space_source_X_sha256")
            == post_smote.get("X_sha256")
            and all(
                no_smote_evidence.get(key) == pre_smote.get(key)
                for key in (
                    "X_sha256", "y_sha256", "weight_sha256",
                    "source_lineage_sha256", "row_order_sha256",
                )
            )
            and all(
                full_evidence.get(key) == no_weight_evidence.get(key)
                for key in (
                    "X_sha256", "y_sha256", "source_lineage_sha256",
                    "row_order_sha256",
                )
            )
            and full_evidence.get("weight_sha256")
            != no_weight_evidence.get("weight_sha256")
            and detail_stages_bound
        )
        checks["resampling_parity_recomputed_exactly"] = bool(
            isinstance(stored_parity, Mapping)
            and canonical_json(stored_parity) == canonical_json(parity_artifact)
            and stage_structure_exact
            and parity_artifact.get("classification") == "VERIFIED"
            and parity_artifact.get("status") == "PASS"
            and parity_artifact.get("generation", {}).get(
                "augmentation_generation_count"
            ) == 1
            and parity_artifact.get("generation", {}).get(
                "safe_smote_generation_count"
            ) == 1
            and parity_artifact.get("generation", {}).get(
                "full_feature_count"
            ) == 93
            and parity_artifact.get("generation", {}).get("scope")
            == "ONE_FULL_SPACE_POPULATION"
            and parity_artifact.get("generation", {}).get("seed") == BASE_SEED
            and parity_artifact.get("generation", {}).get(
                "full_feature_order_sha256"
            ) == variants.get("full_new_reference", {}).get(
                "feature_list_sha256"
            )
            and set(full_weight_checks) == {
                "X_exact", "y_exact", "lineage_exact", "row_order_exact",
                "only_weight_policy_changes",
            }
            and all(value is True for value in full_weight_checks.values())
            and set(no_smote_checks) == {
                "X_exact", "y_exact", "weight_exact", "lineage_exact",
                "row_order_exact", "zero_safe_smote_rows",
            }
            and all(value is True for value in no_smote_checks.values())
            and set(post_checks) == expected_post_ids
            and all(
                set(variant_checks) == {
                    "X_is_exact_column_projection", "y_exact", "lineage_exact",
                    "row_order_exact", "feature_order_hash_exact",
                }
                and all(value is True for value in variant_checks.values())
                for variant_checks in post_checks.values()
            )
            and set(parity_artifact.get("variants", {})) == expected_ids
            and parity_artifact.get("true_no_pca_status")
            == "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE"
            and parity_artifact.get("exploratory_variant") == EXPLORATORY_VARIANT
            and parity_artifact.get(
                "no_safe_smote_retains_mixup_dropout_jitter"
            ) is True
            and all(
                detail.get("paired_resampling", {}).get("frozen_parity_sha256")
                == parity_sha
                and canonical_json(
                    detail.get("paired_resampling", {}).get("projection", {})
                ) == canonical_json(
                    parity_artifact.get("variants", {}).get(variant, {})
                )
                for variant, detail in metrics.get("details", {}).items()
            )
        )
    else:
        expected_parity = smoke_resampling_parity(metrics.get("details", {}))
        comparison_names = (
            "full_vs_no_weight", "full_vs_no_safe_smote_pre_smote",
        )
        checks["resampling_parity_recomputed_exactly"] = bool(
            isinstance(stored_parity, Mapping)
            and canonical_json(stored_parity) == canonical_json(expected_parity)
            and (
                identity.get("state") != "SMOKE_COMPLETE"
                or all(
                    name in expected_parity
                    and all(expected_parity[name].values())
                    for name in comparison_names
                )
            )
        )
    checks["execution_ledger_no_full"] = bool(
        ledger.get("status") == "PASS"
        and not ledger.get("full_scientific_run_executed")
        and int(ledger.get("full_variants_completed", -1)) == 0
        and not ledger.get("forbidden_arguments_detected")
        and not ledger.get("full_scientific_artifacts_detected")
        and not ledger.get("stability_artifacts_detected")
        and not ledger.get("multiseed_artifacts_detected")
    )
    observed_ledger = smoke_execution_ledger(
        run_dir,
        command_line=str(reference.get("command_line", "")),
        runnable_official=sorted(runnable_official),
    )
    checks["execution_ledger_recomputed_exactly"] = (
        canonical_json(observed_ledger) == canonical_json(ledger)
    )

    expected_status = {
        line.partition("=")[0]: line.partition("=")[2]
        for line in evidence_workflow.render_status_lines(state).splitlines()
        if "=" in line
    }
    expected_status.update({
        "CLASSIFICATION": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
        "EXTERNAL_AUDIT_SCORED": "NO",
    })
    if protocol == "AMENDMENT_02":
        expected_status["FULL_AUTHORIZED"] = "NO"
    checks["run_status_matches_derived_state"] = bool(
        len(status_keys) == len(set(status_keys))
        and all(status.get(key) == value for key, value in expected_status.items())
    )
    blockers_text = (run_dir / "BLOCKERS.md").read_text()
    blocker_rows = re.findall(r"^- `([^`]+)`", blockers_text, flags=re.MULTILINE)
    checks["blocker_document_matches_derived_state"] = bool(
        len(blocker_rows) == len(set(blocker_rows))
        and set(blocker_rows) == set(state.blocker_codes)
        and (("- None." in blockers_text) is (not state.blocker_codes))
    )

    stage_verification_path = run_dir / "provenance/package_staging_verification.json"
    if stage_verification_path.is_file():
        staging = json.loads(stage_verification_path.read_text())
        staging_path = Path(str(staging.get(
            "staging_bundle_temporary_path_after_rename", "",
        )))
        staging_semantics = staging.get("package_completeness", {}).get(
            "smoke_semantic_completeness", {},
        )
        staging_common = bool(
            staging.get("crc_status") == "PASS"
            and staging.get("independent_reopen_member_verification") == "PASS"
            and not staging.get("verification_failures")
            and staging.get("staging_bundle_lifecycle")
            == "SUPERSEDED_BY_FINAL_BUNDLE_AND_DELETED_AFTER_FINAL_REOPEN"
            and staging_semantics.get("status") == "PASS"
            and staging_semantics.get("phase") == "STAGING"
            and registry.accepted("output_manifest_zip_verification")
        )
        staging_live_verified = bool(
            staging_path.is_file()
            and not staging_path.is_symlink()
            and sha256_file(staging_path) == staging.get("bundle_sha256")
            and staging_path.stat().st_size == int(
                staging.get("bundle_size_bytes", -1)
            )
            and verify_zip(staging_path, embedded_run_manifest=True).get("status")
            == "PASS"
        )
        staging_deleted_after_final_seal = bool(
            not staging_path.exists()
            and _valid_sha256(staging.get("bundle_sha256"))
            and int(staging.get("bundle_size_bytes", 0)) > 0
            and int(staging.get("member_count", 0)) > 0
            and int(staging.get("uncompressed_size_bytes", 0)) > 0
            and staging.get("publication_method")
            == "VERIFIED_TEMPORARY_THEN_ATOMIC_NO_CLOBBER_LINK"
            and live_run_manifests_match_content(
                run_dir, run_kind=str(identity.get("run_kind", "")),
            )
        )
        checks["staging_verification_matches_output_gate"] = bool(
            staging_common
            and (staging_live_verified or staging_deleted_after_final_seal)
        )
        checks["metrics_match_final_derived_state"] = bool(
            metrics.get("state") == state.run_state
            and metrics.get("cuda_smoke_status") == state.cuda_smoke_status
            and metrics.get("ready_for_full_awaiting_external_review")
            is state.ready_for_full_awaiting_external_review
            and metrics.get("blockers") == list(state.blocker_codes)
            and metrics.get("output_manifest_zip_verification")
            == state.gate_statuses["output_manifest_zip_verification"]
        )
    else:
        checks["staging_verification_matches_output_gate"] = bool(
            registry.evidence["output_manifest_zip_verification"].classification.value
            == "BLOCKED"
            and not registry.accepted("output_manifest_zip_verification")
            and not state.ready_for_full_awaiting_external_review
        )
        checks["metrics_match_final_derived_state"] = bool(
            metrics.get("state") == state.run_state
            and "ready_for_full_awaiting_external_review" not in metrics
        )

    model_verification_path = run_dir / "provenance/smoke_model_bundle_verification.json"
    failure_record_path = run_dir / "provenance/smoke_model_bundle_failure.json"
    if compatibility_recovery and identity.get("state") == "SMOKE_INCOMPLETE":
        companion_zip = run_dir.parent / (
            f"{run_dir.name}_NON_SCIENTIFIC_smoke_models.zip"
        )
        companion_verification = run_dir.parent / (
            f"{run_dir.name}_NON_SCIENTIFIC_smoke_models_"
            "package_verification.json"
        )
        no_models = metrics.get("smoke_models_bundle", {})
        try:
            failure_record = (
                json.loads(failure_record_path.read_text())
                if failure_record_path.is_file() else None
            )
        except Exception:
            failure_record = {}
        no_model_claim = bool(
            no_models.get("path") == "NOT_CREATED_NO_SMOKE_MODELS"
            and no_models.get("sha256") == "NOT_APPLICABLE"
            and failure_record is None
        )
        failed_publication_claim = bool(
            isinstance(failure_record, Mapping)
            and failure_record.get("classification")
            == "NON_SCIENTIFIC_DIAGNOSTIC_ONLY"
            and failure_record.get("status") == "FAIL"
            and failure_record.get("path") == "NOT_CREATED_MODEL_BUNDLE_FAILURE"
            and failure_record.get("sha256") == "NOT_APPLICABLE"
            and failure_record.get("run_id") == run_dir.name
            and no_models == failure_record
        )
        checks["recovery_incomplete_has_no_models_bundle"] = bool(
            not model_verification_path.exists()
            and not companion_zip.exists()
            and not companion_verification.exists()
            and not registry.accepted("cuda_smoke_all_runnable")
            and not registry.accepted("model_save_reload")
            and (no_model_claim or failed_publication_claim)
        )
        checks["model_bundle_verified"] = checks[
            "recovery_incomplete_has_no_models_bundle"
        ]
        checks["run_status_model_bundle_reference"] = bool(
            (
                no_model_claim
                and status.get("SMOKE_MODELS_BUNDLE_PATH")
                == "NOT_CREATED_NO_SMOKE_MODELS"
                and status.get("SMOKE_MODELS_BUNDLE_RUN_ID") == "NOT_APPLICABLE"
            )
            or (
                failed_publication_claim
                and status.get("SMOKE_MODELS_BUNDLE_PATH")
                == "NOT_CREATED_MODEL_BUNDLE_FAILURE"
                and status.get("SMOKE_MODELS_BUNDLE_RUN_ID") == run_dir.name
            )
        ) and status.get("SMOKE_MODELS_BUNDLE_SHA256") == "NOT_APPLICABLE"
    elif completed_ids:
        if model_verification_path.is_file():
            try:
                model = json.loads(model_verification_path.read_text())
                model_semantics = verify_smoke_model_bundle_semantics(
                    model,
                    completed_ids,
                    metrics["details"],
                    variants,
                    run_dir=run_dir,
                    expected_raw_rows=int(subset["training_raw_rows"]),
                    model_parameter_lock=training_configuration.get("model", {}),
                )
                checks["model_bundle_verified"] = bool(
                    model_semantics["status"] == "PASS"
                    and metrics.get("smoke_models_bundle") == model
                    and not failure_record_path.exists()
                )
                checks["run_status_model_bundle_reference"] = bool(
                    status.get("SMOKE_MODELS_BUNDLE_PATH")
                    == model.get("portable_relative_to_run")
                    and status.get("SMOKE_MODELS_BUNDLE_SHA256") == model.get("sha256")
                    and status.get("SMOKE_MODELS_BUNDLE_RUN_ID") == identity.get("run_id")
                )
            except Exception:
                checks["model_bundle_verified"] = False
                checks["run_status_model_bundle_reference"] = False
        else:
            try:
                failure_record = json.loads(failure_record_path.read_text())
            except Exception:
                failure_record = {}
            expected_companion = (
                run_dir.parent
                / f"{run_dir.name}_NON_SCIENTIFIC_smoke_models.zip"
            )
            collision = failure_record.get("preexisting_unowned_collision")
            collision_bound = bool(
                "preexisting_unowned_collision" in failure_record
                and collision == _path_collision_evidence(expected_companion)
            )
            checks["model_bundle_verified"] = bool(
                identity.get("state") == "SMOKE_INCOMPLETE"
                and failure_record.get("classification")
                == "NON_SCIENTIFIC_DIAGNOSTIC_ONLY"
                and failure_record.get("status") == "FAIL"
                and failure_record.get("path")
                == "NOT_CREATED_MODEL_BUNDLE_FAILURE"
                and failure_record.get("portable_relative_to_run")
                == "NOT_CREATED_MODEL_BUNDLE_FAILURE"
                and failure_record.get("sha256") == "NOT_APPLICABLE"
                and failure_record.get("run_id") == run_dir.name
                and set(failure_record.get("completed_variant_ids", []))
                == completed_ids
                and isinstance(failure_record.get("failure"), str)
                and bool(failure_record.get("failure"))
                and metrics.get("smoke_models_bundle") == failure_record
                and not registry.accepted("cuda_smoke_all_runnable")
                and not registry.accepted("model_save_reload")
                and collision_bound
            )
            checks["run_status_model_bundle_reference"] = bool(
                status.get("SMOKE_MODELS_BUNDLE_PATH")
                == "NOT_CREATED_MODEL_BUNDLE_FAILURE"
                and status.get("SMOKE_MODELS_BUNDLE_SHA256")
                == "NOT_APPLICABLE"
                and status.get("SMOKE_MODELS_BUNDLE_RUN_ID") == run_dir.name
            )
    else:
        no_models = metrics.get("smoke_models_bundle", {})
        checks["model_bundle_verified"] = bool(
            not model_verification_path.exists()
            and not registry.accepted("model_save_reload")
            and no_models.get("path") == "NOT_CREATED_NO_SMOKE_MODELS"
            and no_models.get("sha256") == "NOT_APPLICABLE"
        )
        checks["run_status_model_bundle_reference"] = bool(
            status.get("SMOKE_MODELS_BUNDLE_PATH")
            == "NOT_CREATED_NO_SMOKE_MODELS"
            and status.get("SMOKE_MODELS_BUNDLE_SHA256") == "NOT_APPLICABLE"
            and status.get("SMOKE_MODELS_BUNDLE_RUN_ID") == "NOT_APPLICABLE"
        )

    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise IntegrityError(
            f"Smoke semantic completeness failed checks: {failed}"
        )
    result = {
        "status": "PASS",
        "checks": checks,
        "phase": "FINAL" if stage_verification_path.is_file() else "STAGING",
        "official_variants_completed": facts.official_smoke_variants_completed,
        "exploratory_variants_completed": facts.exploratory_smoke_variants_completed,
        "ready_for_full_awaiting_external_review": (
            state.ready_for_full_awaiting_external_review
        ),
    }
    if amendment06_code_chain is not None:
        result["amendment06_reference_code_chain"] = amendment06_code_chain
    return result


def _frame_bytes(frame: pd.DataFrame, *, sep: str) -> bytes:
    return frame.to_csv(sep=sep, index=False, lineterminator="\n").encode("utf-8")


def _snapshot_regular_files(paths: Sequence[Path]) -> dict[Path, bytes | None]:
    snapshot: dict[Path, bytes | None] = {}
    for path in paths:
        if path.exists() or path.is_symlink():
            if not path.is_file() or path.is_symlink():
                raise IntegrityError(f"Transactional output is not a regular file: {path}")
            snapshot[path] = path.read_bytes()
        else:
            snapshot[path] = None
    return snapshot


def _restore_file_snapshot(
    snapshot: Mapping[Path, bytes | None],
    *,
    ownership: MutableMapping[Path, PublicationOwnershipToken],
    previous_ownership: Mapping[Path, PublicationOwnershipToken | None],
) -> None:
    failures: list[str] = []
    for path, data in snapshot.items():
        key = _publication_key(path)
        if key not in ownership:
            continue
        if not remove_owned_publication(path, ownership=ownership):
            failures.append(f"changed:{path}")
            continue
        if data is None:
            continue
        restored_ownership = (
            ownership if previous_ownership.get(path) is not None else {}
        )
        try:
            publish_bytes_no_clobber(
                path, data, ownership=restored_ownership,
            )
        except Exception as exc:
            failures.append(f"restore:{path}:{type(exc).__name__}:{exc}")
    if failures:
        raise IntegrityError(
            "Transactional snapshot restoration was incomplete: "
            + "; ".join(failures)
        )


def package_amended_run(
    run_dir: Path,
    *,
    ownership: MutableMapping[Path, PublicationOwnershipToken] | None = None,
) -> dict[str, Any]:
    """Build, verify, and atomically publish a self-manifested Review ZIP."""

    run_dir = ensure_write_path(run_dir)
    if ownership is None:
        ownership = {}
    destination = STUDY_ROOT / "results" / f"{run_dir.name}_review_bundle.zip"
    if destination.exists() or destination.is_symlink():
        raise IntegrityError(f"Refusing to overwrite existing bundle: {destination}")
    identity_path = run_dir / "config/run_identity.lock.json"
    if not identity_path.exists():
        raise IntegrityError("Amended Run identity lock missing")
    identity_bytes = identity_path.read_bytes()
    identity = json.loads(identity_bytes)
    if identity.get("run_kind") not in {"AMENDED_PREFLIGHT", "CUDA_SMOKE_NON_SCIENTIFIC"}:
        raise IntegrityError(f"Unsupported package Run kind: {identity.get('run_kind')}")
    before_manifest = manifest_rows(
        run_dir, run_kind=str(identity.get("run_kind", "")),
    )
    identity_rows = before_manifest.loc[
        before_manifest.relative_path.astype(str).eq(
            "config/run_identity.lock.json"
        )
    ]
    if (
        len(identity_rows) != 1
        or int(identity_rows.iloc[0].size_bytes) != len(identity_bytes)
        or str(identity_rows.iloc[0].sha256) != sha256_bytes(identity_bytes)
    ):
        raise IntegrityError("Run identity changed before semantic validation")
    completeness = validate_package_completeness(run_dir, identity)
    manifest = manifest_rows(run_dir, run_kind=str(identity.get("run_kind", "")))
    if not before_manifest.equals(manifest):
        raise IntegrityError("Run changed during semantic package validation")
    output_manifest_bytes = _frame_bytes(manifest, sep="\t")
    bundle_manifest = pd.concat([
        manifest[["relative_path", "size_bytes", "sha256"]],
        pd.DataFrame([{
            "relative_path": "OUTPUT_MANIFEST_FINAL.tsv",
            "size_bytes": len(output_manifest_bytes),
            "sha256": sha256_bytes(output_manifest_bytes),
        }]),
    ], ignore_index=True)
    bundle_manifest_bytes = _frame_bytes(bundle_manifest, sep="\t")
    final_members = pd.concat([
        bundle_manifest,
        pd.DataFrame([{
            "relative_path": "BUNDLE_MANIFEST.tsv",
            "size_bytes": len(bundle_manifest_bytes),
            "sha256": sha256_bytes(bundle_manifest_bytes),
        }]),
    ], ignore_index=True)
    generated_members = {
        "OUTPUT_MANIFEST_FINAL.tsv": output_manifest_bytes,
        "BUNDLE_MANIFEST.tsv": bundle_manifest_bytes,
    }
    manifest_paths = tuple(run_dir / name for name in generated_members)
    previous_manifests = _snapshot_regular_files(manifest_paths)
    previous_manifest_ownership = {
        path: ownership.get(_publication_key(path)) for path in manifest_paths
    }
    for path, token in previous_manifest_ownership.items():
        if token is not None and not _publication_matches_token(path, token):
            raise IntegrityError(
                f"Previously owned generated manifest changed before package: {path}"
            )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    os.close(fd)
    os.unlink(temporary_name)
    temporary = Path(temporary_name)
    published = False
    manifests_published = False
    try:
        with zipfile.ZipFile(
            temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9,
        ) as archive:
            for row in final_members.itertuples(index=False):
                relative = str(row.relative_path)
                arcname = f"{run_dir.name}/{relative}"
                if relative in generated_members:
                    archive.writestr(arcname, generated_members[relative])
                else:
                    archive.write(run_dir / relative, arcname=arcname)
        failures = []
        with zipfile.ZipFile(temporary) as archive:
            bad_crc = archive.testzip()
            expected_names = {
                f"{run_dir.name}/{value}" for value in final_members.relative_path
            }
            actual_names = archive.namelist()
            if len(actual_names) != len(set(actual_names)):
                failures.append("duplicate_members")
            if set(actual_names) != expected_names:
                failures.append("member_set")
            for row in final_members.itertuples(index=False):
                data = archive.read(f"{run_dir.name}/{row.relative_path}")
                if len(data) != int(row.size_bytes) or sha256_bytes(data) != row.sha256:
                    failures.append(str(row.relative_path))
        verification = {
            "bundle_path": str(destination), "bundle_sha256": sha256_file(temporary),
            "bundle_size_bytes": temporary.stat().st_size,
            "member_count": int(len(final_members)),
            "uncompressed_size_bytes": int(final_members.size_bytes.sum()),
            "crc_status": "PASS" if bad_crc is None else "FAIL",
            "independent_reopen_member_verification": "PASS" if not failures else "FAIL",
            "verification_failures": failures,
            "package_completeness": completeness,
            "models_bundle_path": "NOT_CREATED_NO_SMOKE_MODELS",
            "models_bundle_sha256": "NOT_APPLICABLE",
            "publication_method": "VERIFIED_TEMPORARY_THEN_ATOMIC_NO_CLOBBER_LINK",
        }
        if bad_crc is not None or failures:
            raise IntegrityError(f"Bundle verification failed: {verification}")

        manifests_published = True
        for relative, data in generated_members.items():
            path = run_dir / relative
            atomic_write_bytes(
                path, data, track_publication=True, ownership=ownership,
            )
        for row in final_members.itertuples(index=False):
            path = run_dir / str(row.relative_path)
            if (
                not path.is_file() or path.is_symlink()
                or path.stat().st_size != int(row.size_bytes)
                or sha256_file(path) != row.sha256
            ):
                raise IntegrityError(
                    f"Live Run drifted before atomic publication: {row.relative_path}"
                )
        publication_token = _capture_regular_publication_token(
            temporary, expected_sha256=verification["bundle_sha256"],
        )
        try:
            os.link(temporary, destination)
            published = True
        except FileExistsError as exc:
            raise IntegrityError(
                f"Refusing to overwrite existing bundle: {destination}"
            ) from exc
        _register_owned_publication(destination, publication_token, ownership)
        return verification
    except Exception:
        if published:
            try:
                _remove_publication_with_token(destination, publication_token)
            finally:
                ownership.pop(_publication_key(destination), None)
        if manifests_published:
            _restore_file_snapshot(
                previous_manifests,
                ownership=ownership,
                previous_ownership=previous_manifest_ownership,
            )
        raise
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


_REFERENCE_SOURCE_INVENTORY_BY_RUN_KIND = {
    "AMENDED_PREFLIGHT": "provenance/source_input_hashes_post.tsv",
    "CUDA_SMOKE_NON_SCIENTIFIC": (
        "provenance/preflight_snapshot/source_input_hashes_post.tsv"
    ),
    "FULL_SCIENTIFIC_ABLATION": "provenance/source_input_hashes_post.tsv",
}
_REFERENCE_SOURCE_INVENTORY_COLUMNS = ("path", "size_bytes", "sha256")
_AMENDMENT06_AUTHORIZATION = "AMENDMENT_06_EXTERNAL_REVIEW_ACCEPTED_ONE_FULL"
_AMENDMENT06_BASELINE_TREE_SHA256 = (
    "55992443799c7623911b2a908e79ff9c12e25e7ac9da15c2215ee1b01eac5075"
)
_AMENDMENT06_ACCEPTED_SMOKE_RUN_ID = (
    "run_20260820_104526_263432_bc48422a_amendment04_cuda_smoke"
)
_AMENDMENT06_A05_VERIFICATION_SHA256 = (
    "271c4cf4ae36e2e0bb589c45f11767d202f996c5eba512acc59020cd92a4dbc2"
)
_AMENDMENT06_A04_EXECUTED_TSV_SHA256 = (
    "579f298931c82e7214a8db7954791de10c0bc345f68047a1777d0268e40b4014"
)
_AMENDMENT06_LEDGER_EVIDENCE_FIELDS = {
    "status",
    "authorization",
    "chain_status",
    "baseline_tree_sha256",
    "latest_committed_tree_sha256",
    "latest_committed_file_count",
    "latest_committed_total_bytes",
    "allowed_after_hashes_sha256",
    "authorized_change_ledger_path",
    "authorized_change_ledger_sha256",
}


def amendment06_live_code_tree(
    study_root: Path = STUDY_ROOT,
) -> dict[str, dict[str, Any]]:
    """Return the complete regular production code/test tree for A06 binding."""

    root = Path(study_root)
    paths = [
        *sorted((root / "code").glob("*.py")),
        *sorted((root / "code").glob("*.sh")),
        *sorted((root / "tests").glob("*.py")),
    ]
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        leaf = os.lstat(path)
        if not stat.S_ISREG(leaf.st_mode) or relative in records:
            raise IntegrityError(
                f"Amendment 06 code tree contains an unsafe/duplicate path: {relative}"
            )
        records[relative] = {
            "size_bytes": int(leaf.st_size),
            "sha256": sha256_file(path),
        }
    if not records:
        raise IntegrityError("Amendment 06 code/test tree is empty")
    return dict(sorted(records.items()))


def amendment06_code_tree_summary(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Hash an exact A06 tree using the committed pre-edit record contract."""

    stream = b"".join(
        (
            f"{records[relative]['sha256']}\t"
            f"{int(records[relative]['size_bytes'])}\t{relative}\n"
        ).encode("utf-8")
        for relative in sorted(records, key=lambda value: value.encode("utf-8"))
    )
    return {
        "tree_sha256": sha256_bytes(stream),
        "file_count": len(records),
        "total_bytes": sum(
            int(record["size_bytes"]) for record in records.values()
        ),
    }


def validate_amendment06_reference_code_chain(
    *,
    allowed_after_hashes: Mapping[str, str],
    ledger_evidence: Mapping[str, Any],
    study_root: Path = STUDY_ROOT,
) -> dict[str, Any]:
    """Bind live code/tests to a complete externally validated A06 ledger tip."""

    root = Path(study_root)
    records = amendment06_live_code_tree(root)
    observed_hashes = {
        relative: str(record["sha256"])
        for relative, record in records.items()
    }
    supplied_hashes = dict(allowed_after_hashes)
    if (
        set(supplied_hashes) != set(observed_hashes)
        or any(not re.fullmatch(r"[0-9a-f]{64}", str(value))
               for value in supplied_hashes.values())
        or supplied_hashes != observed_hashes
    ):
        raise IntegrityError(
            "Amendment 06 allowed-after map is not the complete live code/test tree"
        )
    if set(ledger_evidence) != _AMENDMENT06_LEDGER_EVIDENCE_FIELDS:
        raise IntegrityError("Amendment 06 ledger evidence schema is not exact")

    summary = amendment06_code_tree_summary(records)
    mapping_sha256 = sha256_bytes(canonical_json(dict(sorted(supplied_hashes.items()))))
    expected_ledger = (
        root / ".runtime/amendment06_full/authorized_change_ledger.jsonl"
    )
    supplied_ledger = Path(str(ledger_evidence["authorized_change_ledger_path"]))
    try:
        ledger_lstat = os.lstat(supplied_ledger)
    except FileNotFoundError as exc:
        raise IntegrityError("Amendment 06 authorized-change ledger is missing") from exc
    if (
        supplied_ledger.resolve() != expected_ledger.resolve()
        or not stat.S_ISREG(ledger_lstat.st_mode)
    ):
        raise IntegrityError("Amendment 06 authorized-change ledger path is unsafe")
    ledger_sha256 = sha256_file(supplied_ledger)
    if not (
        ledger_evidence["status"] == "PASS"
        and ledger_evidence["authorization"] == _AMENDMENT06_AUTHORIZATION
        and ledger_evidence["chain_status"] == "PASS"
        and ledger_evidence["baseline_tree_sha256"]
        == _AMENDMENT06_BASELINE_TREE_SHA256
        and ledger_evidence["latest_committed_tree_sha256"]
        == summary["tree_sha256"]
        and int(ledger_evidence["latest_committed_file_count"])
        == summary["file_count"]
        and int(ledger_evidence["latest_committed_total_bytes"])
        == summary["total_bytes"]
        and ledger_evidence["allowed_after_hashes_sha256"] == mapping_sha256
        and ledger_evidence["authorized_change_ledger_sha256"] == ledger_sha256
    ):
        raise IntegrityError("Amendment 06 ledger evidence does not bind the live tree")
    return {
        "status": "PASS",
        "authorization": _AMENDMENT06_AUTHORIZATION,
        "baseline_tree_sha256": _AMENDMENT06_BASELINE_TREE_SHA256,
        "latest_committed_tree_sha256": summary["tree_sha256"],
        "latest_committed_file_count": summary["file_count"],
        "latest_committed_total_bytes": summary["total_bytes"],
        "allowed_after_hashes_sha256": mapping_sha256,
        "authorized_change_ledger_sha256": ledger_sha256,
    }


def validate_sealed_a04_execution_for_amendment06(
    run_dir: Path,
    identity: Mapping[str, Any],
    *,
    study_root: Path = STUDY_ROOT,
) -> dict[str, Any]:
    """Revalidate frozen A04 execution snapshots without consulting changed live code."""

    root = Path(study_root)
    run = Path(run_dir)
    if (
        run.name != _AMENDMENT06_ACCEPTED_SMOKE_RUN_ID
        or identity.get("run_id") != _AMENDMENT06_ACCEPTED_SMOKE_RUN_ID
        or identity.get("recovery_authorization")
        != _AMENDMENT04_AUTHORIZATION
    ):
        raise IntegrityError("Amendment 06 received an unaccepted Smoke Run")

    a05_path = (
        root
        / ".runtime/amendment05_seal_only/final/amendment05_recovery_verification.json"
    )
    try:
        a05_lstat = os.lstat(a05_path)
    except FileNotFoundError as exc:
        raise IntegrityError("Pinned Amendment 05 verification is missing") from exc
    if (
        not stat.S_ISREG(a05_lstat.st_mode)
        or sha256_file(a05_path) != _AMENDMENT06_A05_VERIFICATION_SHA256
    ):
        raise IntegrityError("Pinned Amendment 05 verification changed")
    try:
        a05 = json.loads(a05_path.read_text())
    except Exception as exc:
        raise IntegrityError("Pinned Amendment 05 verification is malformed") from exc
    if not (
        a05.get("status") == "PASS"
        and a05.get("production_final_reference_validation") == "PASS"
        and a05.get("final_review_reopen") == "PASS"
        and a05.get("target_run") == str(run)
        and a05.get("live_code_changed") is False
        and a05.get("no_training", {}).get("TRAINING_API_CALL_COUNT") == 0
        and a05.get("no_training", {}).get("RESAMPLING_CALL_COUNT") == 0
        and a05.get("no_training", {}).get("METRIC_RECOMPUTATION_COUNT") == 0
    ):
        raise IntegrityError("Pinned Amendment 05 acceptance facts changed")

    tsv = run / "provenance/executed_code_hashes.tsv"
    try:
        tsv_lstat = os.lstat(tsv)
    except FileNotFoundError as exc:
        raise IntegrityError("Sealed A04 executed-code TSV is missing") from exc
    if (
        not stat.S_ISREG(tsv_lstat.st_mode)
        or sha256_file(tsv) != _AMENDMENT06_A04_EXECUTED_TSV_SHA256
    ):
        raise IntegrityError("Sealed A04 executed-code TSV changed")
    with tsv.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    columns = [
        "relative_path", "reviewed_sha256", "executed_sha256",
        "snapshot_relative_path", "snapshot_sha256", "drift_status",
        "allowlisted",
    ]
    if reader.fieldnames != columns or not rows:
        raise IntegrityError("Sealed A04 executed-code TSV schema is invalid")
    relatives = [str(row["relative_path"]) for row in rows]
    snapshots = [str(row["snapshot_relative_path"]) for row in rows]
    if (
        len(relatives) != len(set(relatives))
        or len(snapshots) != len(set(snapshots))
    ):
        raise IntegrityError("Sealed A04 executed-code TSV has duplicate paths")

    changed: dict[str, str] = {}
    for row in rows:
        relative = str(row["relative_path"])
        snapshot_relative = str(row["snapshot_relative_path"])
        for value in (relative, snapshot_relative):
            candidate = Path(value)
            if (
                not value
                or candidate.is_absolute()
                or ".." in candidate.parts
                or candidate.as_posix() != value
            ):
                raise IntegrityError(f"Unsafe sealed A04 snapshot path: {value!r}")
        executed = str(row["executed_sha256"])
        snapshot = run / snapshot_relative
        try:
            snapshot_lstat = os.lstat(snapshot)
        except FileNotFoundError as exc:
            raise IntegrityError(
                f"Sealed A04 execution snapshot is missing: {snapshot_relative}"
            ) from exc
        is_changed = row["reviewed_sha256"] != executed
        if not (
            re.fullmatch(r"[0-9a-f]{64}", executed)
            and row["snapshot_sha256"] == executed
            and stat.S_ISREG(snapshot_lstat.st_mode)
            and sha256_file(snapshot) == executed
            and row["drift_status"]
            == ("ALLOWLISTED_CHANGE" if is_changed else "UNCHANGED")
            and row["allowlisted"] == ("YES" if is_changed else "NO")
        ):
            raise IntegrityError(
                f"Sealed A04 execution snapshot is invalid: {relative}"
            )
        if is_changed:
            changed[relative] = executed
    if changed != dict(identity.get("amendment04_allowed_after_hashes", {})):
        raise IntegrityError("Sealed A04 allowed execution changes differ")
    return {
        "status": "PASS",
        "a05_recovery_verification_sha256": _AMENDMENT06_A05_VERIFICATION_SHA256,
        "executed_code_hashes_tsv_sha256": _AMENDMENT06_A04_EXECUTED_TSV_SHA256,
        "executed_file_count": len(rows),
        "sealed_snapshot_status": "PASS",
    }


def reference_source_inventory_relative_path(run_kind: str) -> str:
    """Select the sole authorized source inventory for a sealed Run kind."""

    try:
        return _REFERENCE_SOURCE_INVENTORY_BY_RUN_KIND[run_kind]
    except KeyError as exc:
        raise IntegrityError(
            f"Unsupported Run kind for source inventory routing: {run_kind!r}"
        ) from exc


def validate_reference_source_inventory(
    run_dir: Path,
    *,
    run_kind: str,
    output_manifest: pd.DataFrame,
    bundle_manifest: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the explicitly routed, manifest-bound source inventory."""

    # Select before touching a path so unsupported kinds fail without discovery.
    relative = reference_source_inventory_relative_path(run_kind)
    for label, frame in (
        ("OUTPUT_MANIFEST_FINAL.tsv", output_manifest),
        ("BUNDLE_MANIFEST.tsv", bundle_manifest),
    ):
        required = {"relative_path", "size_bytes", "sha256"}
        if not required <= set(frame.columns):
            raise IntegrityError(
                f"{label} cannot bind the selected source inventory"
            )
        matches = frame.loc[
            frame["relative_path"].astype(str) == relative
        ]
        if len(matches) != 1:
            raise IntegrityError(
                f"Selected source inventory is not uniquely sealed in {label}: "
                f"{relative}"
            )

    inventory_path = run_dir / relative
    try:
        inventory_lstat = os.lstat(inventory_path)
    except FileNotFoundError as exc:
        raise IntegrityError(
            f"Selected source inventory is missing: {relative}"
        ) from exc
    if not stat.S_ISREG(inventory_lstat.st_mode):
        raise IntegrityError(
            f"Selected source inventory is not a regular non-symlink file: {relative}"
        )
    inventory_bytes = inventory_path.read_bytes()
    inventory_sha256 = sha256_bytes(inventory_bytes)
    for label, frame in (
        ("OUTPUT_MANIFEST_FINAL.tsv", output_manifest),
        ("BUNDLE_MANIFEST.tsv", bundle_manifest),
    ):
        row = frame.loc[
            frame["relative_path"].astype(str) == relative
        ].iloc[0]
        try:
            sealed_size = int(row["size_bytes"])
        except (TypeError, ValueError) as exc:
            raise IntegrityError(
                f"{label} has an invalid source-inventory size"
            ) from exc
        if (
            sealed_size != len(inventory_bytes)
            or str(row["sha256"]) != inventory_sha256
        ):
            raise IntegrityError(
                f"Selected source inventory differs from {label}: {relative}"
            )

    try:
        inventory_text = inventory_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IntegrityError("Selected source inventory is not UTF-8 TSV") from exc
    parsed = list(csv.reader(io.StringIO(inventory_text), delimiter="\t"))
    if not parsed or tuple(parsed[0]) != _REFERENCE_SOURCE_INVENTORY_COLUMNS:
        raise IntegrityError(
            "Selected source inventory schema must be exactly "
            "path,size_bytes,sha256"
        )
    if len(parsed) == 1 or any(
        len(fields) != len(_REFERENCE_SOURCE_INVENTORY_COLUMNS)
        for fields in parsed[1:]
    ):
        raise IntegrityError("Selected source inventory has invalid TSV rows")

    rows = [dict(zip(_REFERENCE_SOURCE_INVENTORY_COLUMNS, fields))
            for fields in parsed[1:]]
    source_paths = [row["path"] for row in rows]
    if len(source_paths) != len(set(source_paths)):
        raise IntegrityError("Selected source inventory has duplicate source paths")

    for row in rows:
        source_text = row["path"]
        source = Path(source_text)
        if not source_text or "\x00" in source_text or not source.is_absolute():
            raise IntegrityError(
                f"Selected source inventory has an unsafe source path: {source_text!r}"
            )
        if not re.fullmatch(r"(?:0|[1-9][0-9]*)", row["size_bytes"]):
            raise IntegrityError(
                f"Selected source inventory has an invalid size: {source}"
            )
        expected_size = int(row["size_bytes"])
        expected_sha256 = row["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise IntegrityError(
                f"Selected source inventory has an invalid SHA-256: {source}"
            )
        try:
            source_lstat = os.lstat(source)
        except FileNotFoundError as exc:
            raise IntegrityError(
                f"Source is missing from referenced Run inventory: {source}"
            ) from exc
        if not stat.S_ISREG(source_lstat.st_mode):
            raise IntegrityError(
                f"Source is not a regular non-symlink file: {source}"
            )
        if (
            source_lstat.st_size != expected_size
            or sha256_file(source) != expected_sha256
        ):
            raise IntegrityError(f"Source changed since referenced Run: {source}")

    return {
        "status": "PASS",
        "selected_relative_path": relative,
        "sha256": inventory_sha256,
        "row_count": len(rows),
    }


def validate_amended_run_reference(
    run_dir: Path,
    *,
    expected_kind: str,
    allowed_states: set[str],
    amendment03_allowed_after_hashes: Mapping[str, str] | None = None,
    amendment04_reference_allowed_after_hashes: Mapping[str, str] | None = None,
    amendment06_allowed_after_hashes: Mapping[str, str] | None = None,
    amendment06_change_ledger_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # Reject unsupported expected kinds before resolving or reading any Run path.
    reference_source_inventory_relative_path(expected_kind)
    target = run_dir.resolve()
    if STUDY_ROOT.resolve() not in target.parents:
        raise IntegrityError("Run reference is outside study root")
    identity_path = target / "config/run_identity.lock.json"
    if not identity_path.exists():
        raise IntegrityError("Run reference has no identity lock")
    identity = json.loads(identity_path.read_text())
    sealed_run_kind = identity.get("run_kind")
    if sealed_run_kind != expected_kind or identity.get("state") not in allowed_states:
        raise IntegrityError(f"Run kind/state mismatch: {identity}")
    reference_source_inventory_relative_path(str(sealed_run_kind))
    completeness = validate_package_completeness(
        target,
        identity,
        amendment03_allowed_after_hashes=amendment03_allowed_after_hashes,
        amendment04_reference_allowed_after_hashes=(
            amendment04_reference_allowed_after_hashes
        ),
        amendment06_allowed_after_hashes=amendment06_allowed_after_hashes,
        amendment06_change_ledger_evidence=(
            amendment06_change_ledger_evidence
        ),
    )
    manifest_path = target / "OUTPUT_MANIFEST_FINAL.tsv"
    bundle_manifest_path = target / "BUNDLE_MANIFEST.tsv"
    if not manifest_path.is_file() or not bundle_manifest_path.is_file():
        raise IntegrityError("Run reference has no sealed output/bundle manifest")
    manifest = pd.read_csv(manifest_path, sep="\t")
    bundle_manifest = pd.read_csv(bundle_manifest_path, sep="\t")
    failures = []
    output_columns = [
        "relative_path", "size_bytes", "sha256", "artifact_role",
        "include_in_review_bundle", "include_in_models_bundle",
    ]
    bundle_columns = ["relative_path", "size_bytes", "sha256"]
    if list(manifest.columns) != output_columns:
        failures.append("output_manifest_schema")
    if list(bundle_manifest.columns) != bundle_columns:
        failures.append("bundle_manifest_schema")
    for label, frame in (("output", manifest), ("bundle", bundle_manifest)):
        if "relative_path" not in frame or frame.relative_path.astype(str).duplicated().any():
            failures.append(f"{label}_manifest_duplicates")
            continue
        for relative in frame.relative_path.astype(str):
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                failures.append(f"{label}_manifest_unsafe_path:{relative}")
    output_paths = set(manifest.relative_path.astype(str)) if "relative_path" in manifest else set()
    bundle_paths = (
        set(bundle_manifest.relative_path.astype(str))
        if "relative_path" in bundle_manifest else set()
    )
    if bundle_paths != output_paths | {"OUTPUT_MANIFEST_FINAL.tsv"}:
        failures.append("bundle_manifest_member_set")
    for row in bundle_manifest.itertuples(index=False):
        path = target / row.relative_path
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != int(row.size_bytes)
            or sha256_file(path) != row.sha256
        ):
            failures.append(str(row.relative_path))
    if (
        not bundle_manifest_path.is_file() or bundle_manifest_path.is_symlink()
        or bundle_manifest_path.stat().st_size == 0
    ):
        failures.append("bundle_manifest_live_file")
    live_files = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*") if path.is_file()
    }
    expected_files = bundle_paths | {"BUNDLE_MANIFEST.tsv"}
    if live_files != expected_files:
        failures.append("live_file_set_differs_from_sealed_manifests")

    review_bundle = Path(str(target) + "_review_bundle.zip")
    adjacent = target.parent / f"{target.name}_package_verification.json"
    if (
        not review_bundle.is_file() or review_bundle.is_symlink()
        or not adjacent.is_file() or adjacent.is_symlink()
    ):
        failures.append("review_bundle_or_adjacent_missing")
        package_verification: dict[str, Any] = {}
        zip_verification: dict[str, Any] = {}
    else:
        try:
            package_verification = json.loads(adjacent.read_text())
            zip_verification = verify_zip(
                review_bundle, embedded_run_manifest=True,
            )
            expected_archive_names = {
                f"{target.name}/{relative}" for relative in live_files
            }
            with zipfile.ZipFile(review_bundle) as archive:
                archive_names = archive.namelist()
                if set(archive_names) != expected_archive_names:
                    failures.append("review_bundle_member_set")
                if len(archive_names) != len(set(archive_names)):
                    failures.append("review_bundle_duplicate_members")
                for relative in live_files:
                    member = f"{target.name}/{relative}"
                    if (
                        member not in archive_names
                        or archive.read(member) != (target / relative).read_bytes()
                    ):
                        failures.append(f"review_bundle_live_mismatch:{relative}")
            if (
                package_verification.get("bundle_path") != str(review_bundle)
                or package_verification.get("bundle_sha256")
                != sha256_file(review_bundle)
                or int(package_verification.get("bundle_size_bytes", -1))
                != review_bundle.stat().st_size
                or int(package_verification.get("member_count", -1))
                != len(live_files)
                or int(package_verification.get("uncompressed_size_bytes", -1))
                != sum((target / relative).stat().st_size for relative in live_files)
                or package_verification.get("crc_status") != "PASS"
                or package_verification.get("independent_reopen_member_verification")
                != "PASS"
                or package_verification.get("verification_failures")
                or zip_verification.get("status") != "PASS"
            ):
                failures.append("review_bundle_adjacent_verification")
        except Exception as exc:
            failures.append(f"review_bundle_unreadable:{type(exc).__name__}")
    if failures:
        raise IntegrityError(f"Run reference manifest verification failed: {failures}")
    source_inventory = validate_reference_source_inventory(
        target,
        run_kind=str(sealed_run_kind),
        output_manifest=manifest,
        bundle_manifest=bundle_manifest,
    )
    split = pd.read_csv(target / "splits/row_split_manifest.csv.gz")
    split_hash = split_assignment_hash(
        split.stable_candidate_id, split.split, split.label, split.eligible,
        split.split_order if "split_order" in split else None,
    )
    if split_hash != identity.get("candidate_split_hash"):
        raise IntegrityError("Referenced Run split hash differs from identity lock")
    identity["reference_validation"] = {
        **completeness,
        "output_manifest_rows_verified": int(len(manifest)),
        "bundle_manifest_rows_verified": int(len(bundle_manifest)),
        "live_file_set_status": "PASS",
        "bundle_manifest_status": "PASS",
        "review_bundle_live_byte_equality_status": "PASS",
        "review_bundle_sha256": sha256_file(review_bundle),
        "package_verification_sha256": sha256_file(adjacent),
        "split_hash_status": "PASS",
        "source_inventory_selected_relative_path": source_inventory[
            "selected_relative_path"
        ],
        "source_inventory_sha256": source_inventory["sha256"],
        "source_inventory_rows_verified": source_inventory["row_count"],
    }
    return identity


def _lineage_hash(values: np.ndarray) -> str:
    return sha256_bytes(("\n".join(map(str, values.tolist())) + "\n").encode())


def _stage_evidence(
    stage: str, X: np.ndarray, y: np.ndarray, weights: np.ndarray, lineage: np.ndarray,
) -> dict[str, Any]:
    return {
        "stage": stage, "rows": len(y), "features": int(X.shape[1]),
        "negative": int((y == 0).sum()), "positive": int((y == 1).sum()),
        "X_sha256": sha256_array(np.asarray(X, dtype=np.float32)),
        "y_sha256": sha256_array(np.asarray(y, dtype=np.int8)),
        "weight_sha256": sha256_array(np.asarray(weights, dtype="<f4")),
        "source_lineage_sha256": _lineage_hash(lineage),
        "row_order_sha256": _lineage_hash(lineage),
    }


def augment_with_lineage(
    frame: pd.DataFrame, labels: np.ndarray, weights: np.ndarray,
    source_ids: Sequence[str], *, seed: int = BASE_SEED, smote_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SimpleImputer, list[dict[str, Any]]]:
    """Small-run implementation of the historical augmentation with provenance."""
    labels = np.asarray(labels, dtype=np.int8)
    weights = np.asarray(weights, dtype=np.float32)
    lineage = np.asarray(list(map(str, source_ids)), dtype=object)
    if not (len(frame) == len(labels) == len(weights) == len(lineage)):
        raise AmendmentError("Augmentation inputs have inconsistent lengths")
    imputer = SimpleImputer(strategy="median")
    X = np.asarray(imputer.fit_transform(frame), dtype=np.float32)
    continuous = np.where(frame.nunique(dropna=False).to_numpy() > 10)[0]
    # Input order is the immutable historical split order, which was produced by
    # its own RandomState(42). Augmentation historically resets RandomState(42).
    rng = np.random.RandomState(seed)
    stages = [_stage_evidence("raw_locked_post_split_shuffle", X, labels, weights, lineage)]
    if len(continuous):
        q_low = np.quantile(X[:, continuous], 0.001, axis=0)
        q_high = np.quantile(X[:, continuous], 0.999, axis=0)
        new_x, new_y, new_weights, new_lineage = [X], [labels], [weights], [lineage]
        for label in np.unique(labels):
            indices = np.where(labels == label)[0]
            count = int(math.ceil(len(indices) * 0.5))
            if len(indices) < 2:
                continue
            first = rng.choice(indices, size=count, replace=len(indices) < count)
            second = rng.choice(indices, size=count, replace=len(indices) < count)
            lam = rng.beta(0.2, 0.2, size=count).astype(np.float32)
            values = X[first].copy()
            values[:, continuous] = X[first][:, continuous] + lam[:, None] * (
                X[second][:, continuous] - X[first][:, continuous]
            )
            choose_first = rng.rand(count) < 0.5
            parent = np.where(choose_first, lineage[first], lineage[second])
            synthetic_ids = np.asarray([
                f"mix_{sha256_bytes(f'{lineage[a]}|{lineage[b]}|{index}'.encode())}"
                for index, (a, b) in enumerate(zip(first, second))
            ], dtype=object)
            new_x.append(values.astype(np.float32))
            new_y.append(np.full(count, label, dtype=np.int8))
            new_weights.append((0.5 * (weights[first] + weights[second])).astype(np.float32))
            new_lineage.append(np.char.add(np.char.add(parent.astype(str), "|"), synthetic_ids.astype(str)))
        X = np.vstack(new_x).astype(np.float32)
        labels = np.concatenate(new_y).astype(np.int8)
        weights = np.concatenate(new_weights).astype(np.float32)
        lineage = np.concatenate(new_lineage).astype(object)
    stages.append(_stage_evidence("post_mixup", X, labels, weights, lineage))
    if len(continuous):
        count = len(labels)
        selected = rng.choice(np.arange(len(labels)), size=count, replace=False)
        values, selected_labels = X[selected].copy(), labels[selected].copy()
        medians = {int(label): np.median(X[labels == label][:, continuous], axis=0) for label in np.unique(labels)}
        fills = np.vstack([medians[int(label)] for label in selected_labels])
        mask = rng.rand(count, len(continuous)) < 0.05
        values[:, continuous] = np.where(mask, fills, values[:, continuous])
        dropout_ids = np.asarray([
            f"drop_{sha256_bytes(f'{lineage[index]}|{offset}'.encode())}"
            for offset, index in enumerate(selected)
        ], dtype=object)
        X = np.vstack([X, values]).astype(np.float32)
        labels = np.concatenate([labels, selected_labels]).astype(np.int8)
        weights = np.concatenate([weights, weights[selected]]).astype(np.float32)
        lineage = np.concatenate([lineage, dropout_ids]).astype(object)
    stages.append(_stage_evidence("post_dropout", X, labels, weights, lineage))
    if len(continuous):
        count = int(math.ceil(len(labels) * 0.5))
        selected = rng.choice(np.arange(len(labels)), size=count, replace=False)
        values = X[selected].copy()
        std = X[:, continuous].std(axis=0, ddof=0) + 1e-6
        noise = rng.normal(0.0, 1.0, size=(count, len(continuous))).astype(np.float32)
        values[:, continuous] += noise * (0.01 * std)
        values[:, continuous] = np.minimum(np.maximum(values[:, continuous], q_low), q_high)
        jitter_ids = np.asarray([
            f"jitter_{sha256_bytes(f'{lineage[index]}|{offset}'.encode())}"
            for offset, index in enumerate(selected)
        ], dtype=object)
        X = np.vstack([X, values]).astype(np.float32)
        labels = np.concatenate([labels, labels[selected]]).astype(np.int8)
        weights = np.concatenate([weights, weights[selected]]).astype(np.float32)
        lineage = np.concatenate([lineage, jitter_ids]).astype(object)
    stages.append(_stage_evidence("post_jitter_pre_smote", X, labels, weights, lineage))
    post_shuffle = _stage_evidence("post_shuffle", X, labels, weights, lineage)
    post_shuffle.update({
        "shuffle_applied": False,
        "rule": "NO_POST_AUGMENTATION_SHUFFLE",
        "reason": (
            "Verified historical shuffle occurs once before augmentation; the "
            "post-Jitter order is preserved into SMOTE."
        ),
    })
    stages.append(post_shuffle)
    if smote_enabled:
        before = len(labels)
        sampler = legacy.SafeSMOTE(sampling_strategy=0.5, k_neighbors=3, random_state=seed)
        X_result, labels_result = sampler.fit_resample(X, labels)
        X_result = np.asarray(X_result, dtype=np.float32)
        labels_result = np.asarray(labels_result, dtype=np.int8)
        synthetic = len(labels_result) - before
        if synthetic:
            synthetic_labels = labels_result[before:]
            synthetic_weights = np.asarray([
                weights[labels == label].mean() for label in synthetic_labels
            ], dtype=np.float32)
            weights = np.concatenate([weights, synthetic_weights]).astype(np.float32)
            synthetic_ids = np.asarray([
                f"smote_{sha256_bytes(f'{seed}|{offset}|{int(label)}'.encode())}"
                for offset, label in enumerate(synthetic_labels)
            ], dtype=object)
            lineage = np.concatenate([lineage, synthetic_ids]).astype(object)
        X, labels = X_result, labels_result
    stages.append(_stage_evidence("post_safe_smote", X, labels, weights, lineage))
    stages[-1]["safe_smote_enabled"] = smote_enabled
    stages[-1]["safe_smote_synthetic_rows"] = stages[-1]["rows"] - stages[-2]["rows"]
    return X, labels, weights, lineage, imputer, stages


def deterministic_smoke_subset(
    split_manifest: pd.DataFrame, *, train_cap: int = 4000, validation_cap: int = 2000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    def choose(name: str, cap: int) -> np.ndarray:
        available = split_manifest[(split_manifest.split == name) & split_manifest.eligible.astype(bool)].copy()
        if "split_order" not in available:
            raise ScientificBlocker("Split manifest lacks immutable within-split order")
        available = available.sort_values("split_order", kind="stable")
        card_summary = available.groupby("card_id").label.nunique()
        both_class_cards = sorted(card_summary[card_summary == 2].index.astype(str))
        if not both_class_cards:
            raise ScientificBlocker(f"No {name} card contains both classes")
        selected = available[available.card_id.astype(str).isin(both_class_cards)]
        indices = selected.index.to_numpy(dtype=np.int64)[:cap]
        if split_manifest.loc[indices, "label"].nunique() != 2:
            positives = selected[selected.label == 1].index.to_numpy(dtype=np.int64)
            negatives = selected[selected.label == 0].index.to_numpy(dtype=np.int64)
            half = cap // 2
            indices = np.concatenate([negatives[:half], positives[:half]])
            indices = split_manifest.loc[indices].sort_values("split_order", kind="stable").index.to_numpy(dtype=np.int64)
        if len(indices) > cap or split_manifest.loc[indices, "label"].nunique() != 2:
            raise ScientificBlocker(f"Deterministic {name} Smoke subset is invalid")
        return indices

    training = choose("training", train_cap)
    validation = choose("validation", validation_cap)
    def ordered_hash(values: Sequence[Any]) -> str:
        return sha256_bytes(("\n".join(map(str, values)) + "\n").encode())

    training_rows = split_manifest.loc[training]
    validation_rows = split_manifest.loc[validation]
    return training, validation, {
        "selection_rule": SMOKE_SUBSET_SELECTION_RULE,
        "training_manifest_indices": training.tolist(),
        "validation_manifest_indices": validation.tolist(),
        "training_stable_candidate_id_sha256": ordered_hash(
            training_rows.stable_candidate_id.astype(str)
        ),
        "validation_stable_candidate_id_sha256": ordered_hash(
            validation_rows.stable_candidate_id.astype(str)
        ),
        "training_label_sha256": ordered_hash(training_rows.label.astype(int)),
        "validation_label_sha256": ordered_hash(validation_rows.label.astype(int)),
        "training_split_order_sha256": ordered_hash(
            training_rows.split_order.astype(int)
        ),
        "validation_split_order_sha256": ordered_hash(
            validation_rows.split_order.astype(int)
        ),
        "training_raw_rows": len(training), "validation_rows": len(validation),
        "training_cards": sorted(split_manifest.loc[training, "card_id"].astype(str).unique()),
        "validation_cards": sorted(split_manifest.loc[validation, "card_id"].astype(str).unique()),
        "training_positive": int(split_manifest.loc[training, "label"].sum()),
        "validation_positive": int(split_manifest.loc[validation, "label"].sum()),
    }


def gpu_identity_metadata() -> dict[str, Any]:
    """Capture the device identity recorded with each prospective Smoke variant."""
    result = run_command([
        "nvidia-smi", "--query-gpu=index,name,driver_version",
        "--format=csv,noheader,nounits",
    ], cwd=STUDY_ROOT)
    devices = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            fields = [part.strip() for part in line.split(",", 2)]
            if len(fields) == 3:
                devices.append({
                    "physical_index": fields[0],
                    "gpu_name": fields[1],
                    "driver_version": fields[2],
                })
    return {
        "requested_device": "cuda",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", "UNSET"),
        "nvidia_smi_query_returncode": result.returncode,
        "nvidia_smi_query_stderr": result.stderr,
        "visible_gpu_count_reported_by_nvidia_smi": len(devices),
        "devices": devices,
        "xgboost_cuda_compiled": bool(xgb.build_info().get("USE_CUDA", False)),
    }


def _gpu_used_memory_mib() -> float | None:
    """Measure process GPU memory, falling back to device-global WSL evidence."""

    return legacy.gpu_memory_usage_mib(os.getpid())[0]


def _package_smoke_models(
    model_root: Path,
    run_name: str,
    completed_variant_ids: Sequence[str] | None = None,
    *,
    ownership: MutableMapping[Path, PublicationOwnershipToken] | None = None,
) -> dict[str, Any]:
    if ownership is None:
        ownership = {}
    symlinks = [path for path in model_root.rglob("*") if path.is_symlink()]
    available_files = sorted(
        path for path in model_root.rglob("*")
        if path.is_file() and path.relative_to(model_root).as_posix() != "BUNDLE_MANIFEST.tsv"
    )
    requested_names = (
        {f"{variant}.ubj" for variant in completed_variant_ids}
        if completed_variant_ids is not None else None
    )
    files = [
        path for path in available_files
        if requested_names is None
        or path.relative_to(model_root).as_posix() in requested_names
    ]
    if (
        not files or symlinks
        or any(path.stat().st_size == 0 or path.suffix.lower() != ".ubj" for path in files)
        or (
            requested_names is not None
            and {path.relative_to(model_root).as_posix() for path in files}
            != requested_names
        )
    ):
        raise IntegrityError("Smoke model bundle requires nonempty regular model artifacts")
    for path in files:
        try:
            booster = xgb.Booster()
            booster.load_model(path)
            if booster.num_boosted_rounds() <= 0:
                raise ValueError("model has no boosted rounds")
        except Exception as exc:
            raise IntegrityError(
                f"Smoke model artifact is not a loadable XGBoost model: {path.name}"
            ) from exc
    manifest = pd.DataFrame([
        {"relative_path": path.relative_to(model_root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    ])
    manifest_path = model_root / "BUNDLE_MANIFEST.tsv"
    manifest_bytes = _frame_bytes(manifest, sep="\t")
    destination = STUDY_ROOT / "results" / f"{run_name}_NON_SCIENTIFIC_smoke_models.zip"
    if destination.exists() or destination.is_symlink():
        raise IntegrityError(f"Refusing model-bundle overwrite: {destination}")
    previous_manifest = _snapshot_regular_files((manifest_path,))
    previous_manifest_ownership = {
        manifest_path: ownership.get(_publication_key(manifest_path))
    }
    previous_token = previous_manifest_ownership[manifest_path]
    if previous_token is not None and not _publication_matches_token(
        manifest_path, previous_token,
    ):
        raise IntegrityError(
            f"Previously owned model manifest changed before package: {manifest_path}"
        )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    os.close(fd)
    os.unlink(temporary_name)
    temporary = Path(temporary_name)
    published = False
    manifest_published = False
    archive_root = f"{run_name}_smoke_models"
    try:
        with zipfile.ZipFile(
            temporary, "x", zipfile.ZIP_DEFLATED, compresslevel=9,
        ) as archive:
            for path in files:
                archive.write(
                    path,
                    arcname=f"{archive_root}/{path.relative_to(model_root).as_posix()}",
                )
            archive.writestr(f"{archive_root}/BUNDLE_MANIFEST.tsv", manifest_bytes)
        with zipfile.ZipFile(temporary) as archive:
            bad_crc = archive.testzip()
            failures = []
            actual_names = archive.namelist()
            expected_names = {
                f"{archive_root}/{path.relative_to(model_root).as_posix()}"
                for path in files
            } | {f"{archive_root}/BUNDLE_MANIFEST.tsv"}
            if len(actual_names) != len(set(actual_names)):
                failures.append("duplicate_members")
            if set(actual_names) != expected_names:
                failures.append("member_set")
            for row in manifest.itertuples(index=False):
                data = archive.read(f"{archive_root}/{row.relative_path}")
                if len(data) != row.size_bytes or sha256_bytes(data) != row.sha256:
                    failures.append(row.relative_path)
            manifest_member = f"{archive_root}/BUNDLE_MANIFEST.tsv"
            if archive.read(manifest_member) != manifest_bytes:
                failures.append("BUNDLE_MANIFEST.tsv")
        if bad_crc or failures:
            raise IntegrityError(
                f"Smoke model bundle verification failed: crc={bad_crc}, "
                f"failures={failures}"
            )
        verified_sha = sha256_file(temporary)
        atomic_write_bytes(
            manifest_path, manifest_bytes, track_publication=True,
            ownership=ownership,
        )
        manifest_published = True
        publication_token = _capture_regular_publication_token(
            temporary, expected_sha256=verified_sha,
        )
        try:
            os.link(temporary, destination)
            published = True
        except FileExistsError as exc:
            raise IntegrityError(
                f"Refusing model-bundle overwrite: {destination}"
            ) from exc
        _register_owned_publication(destination, publication_token, ownership)
        return {
            "path": str(destination), "sha256": verified_sha,
            "filename": destination.name,
            "portable_relative_to_run": f"../{destination.name}",
            "archive_root": archive_root,
            "size_bytes": destination.stat().st_size,
            "member_count": len(files) + 1,
            "crc_status": "PASS", "reopen_member_hash_status": "PASS",
            "classification": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
            "run_id": run_name,
            "completed_variant_ids": sorted(
                path.stem for path in files
            ),
            "excluded_uncompleted_artifacts": sorted(
                path.relative_to(model_root).as_posix()
                for path in available_files if path not in files
            ),
            "publication_method": "VERIFIED_TEMPORARY_THEN_ATOMIC_NO_CLOBBER_LINK",
        }
    except Exception:
        if published:
            try:
                _remove_publication_with_token(destination, publication_token)
            finally:
                ownership.pop(_publication_key(destination), None)
        if manifest_published:
            _restore_file_snapshot(
                previous_manifest,
                ownership=ownership,
                previous_ownership=previous_manifest_ownership,
            )
        raise
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def create_immutable_run_directory(path: Path) -> Path:
    """Create a new Run directory without permitting reuse or concurrent clobber."""

    target = ensure_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise IntegrityError(f"Run/model directory already exists: {target}") from exc
    return target


def smoke_copied_evidence_mapping(
    protocol: str = "AMENDMENT_01",
) -> dict[str, str]:
    """Return the exact reviewed Preflight source-to-Smoke snapshot contract."""
    if protocol == "AMENDMENT_02":
        direct_files = [
            "docs/PROTOCOL_AMENDMENT_02_TASK_SPEC.md",
            "docs/PROTOCOL_AMENDMENT_02.md",
            "docs/DATA_SOURCE_DECISION.md",
            "docs/FIXED_TABLE_LIMITATIONS.md",
            "config/feature_manifest.csv",
            "config/feature_dependency_graph.json",
            "config/variant_feature_sets.json",
            "config/training_configuration.lock.json",
            "config/locked_split_identity.json",
            "config/resolved_paths.json",
            "splits/card_split_manifest.csv",
            "splits/row_split_manifest.csv.gz",
            "splits/split_summary.csv",
            "splits/overlap_checks.csv",
            "splits/split_hashes.json",
            "provenance/cuda_environment_before.txt",
            "provenance/cuda_environment_after.txt",
            "provenance/cuda_active_fit_probe.json",
            *[f"code/{name}" for name in (
                "ablation_core.py", "amendment_core.py", "amendment_inventory.py",
                "amendment02_resampling.py", "build_amended_preflight.py",
                "build_amendment02_preflight.py", "build_blocked_preflight.py",
                "evidence_gates.py", "run_ablation_study.sh", "run_study.py",
            )],
            *[f"tests/{name}" for name in (
                "test_ablation.py", "test_amendment_inventory.py",
                "test_amendment02_resampling.py", "test_evidence_gates.py",
            )],
            *[f"provenance/code_snapshot/{name}" for name in (
                "ablation_core.py", "amendment_core.py", "amendment_inventory.py",
                "amendment02_resampling.py", "build_amended_preflight.py",
                "build_amendment02_preflight.py", "build_blocked_preflight.py",
                "evidence_gates.py", "run_ablation_study.sh", "run_study.py",
            )],
            *[f"provenance/tests_snapshot/{name}" for name in (
                "test_ablation.py", "test_amendment_inventory.py",
                "test_amendment02_resampling.py", "test_evidence_gates.py",
            )],
        ]
        log_mapping = {
            "logs/targeted_tests.log": "logs/preflight_targeted_tests.log",
            "logs/full_tests.log": "logs/preflight_full_tests.log",
            "logs/reviewer_snapshot_syntax_compile.log": (
                "logs/preflight_reviewer_snapshot_syntax_compile.log"
            ),
            "logs/reviewer_snapshot_pytest.log": (
                "logs/preflight_reviewer_snapshot_pytest.log"
            ),
        }
        snapshot_mapping = {
            "provenance/PREFLIGHT_REPORT.json": (
                "provenance/preflight_snapshot/PREFLIGHT_REPORT.json"
            ),
            "provenance/evidence_gate_registry.json": (
                "provenance/preflight_snapshot/evidence_gate_registry.json"
            ),
            "provenance/source_input_hashes_post.tsv": (
                "provenance/preflight_snapshot/source_input_hashes_post.tsv"
            ),
            "provenance/cuda_active_fit_probe.json": (
                "provenance/preflight_snapshot/cuda_active_fit_probe.json"
            ),
            "config/run_identity.lock.json": (
                "provenance/preflight_snapshot/run_identity.lock.json"
            ),
            "OUTPUT_MANIFEST_FINAL.tsv": (
                "provenance/preflight_snapshot/OUTPUT_MANIFEST_FINAL.tsv"
            ),
            "BUNDLE_MANIFEST.tsv": (
                "provenance/preflight_snapshot/BUNDLE_MANIFEST.tsv"
            ),
            "__PREFLIGHT_REVIEW_BUNDLE__": (
                "provenance/preflight_snapshot/preflight_review_bundle.zip"
            ),
        }
        return {
            **{value: value for value in direct_files},
            **log_mapping,
            **snapshot_mapping,
        }
    if protocol != "AMENDMENT_01":
        raise IntegrityError(f"Unsupported Smoke evidence protocol: {protocol}")
    direct_files = [
        "docs/PROTOCOL_AMENDMENT_01_TASK_SPEC.md",
        "docs/PROTOCOL_AMENDMENT_01.md",
        "docs/DATA_SOURCE_DECISION.md",
        "docs/FEATURE_SEMANTICS_RECONCILIATION.md",
        "docs/NO_PCA_DIAGNOSTIC.md",
        "docs/CODE_FIXES_AND_TESTS.md",
        "docs/INITIAL_DISCOVERY_AND_PREFLIGHT_PLAN.md",
        "config/feature_manifest.csv",
        "config/feature_dependency_graph.json",
        "config/variant_feature_sets.json",
        "config/training_configuration.lock.json",
        "config/resolved_paths.json",
        "splits/card_split_manifest.csv",
        "splits/row_split_manifest.csv.gz",
        "splits/split_summary.csv",
        "splits/overlap_checks.csv",
        "splits/split_hashes.json",
        *[f"code/{name}" for name in (
            "ablation_core.py", "amendment_core.py", "amendment_inventory.py",
            "build_amended_preflight.py", "build_blocked_preflight.py",
            "evidence_gates.py", "run_ablation_study.sh", "run_study.py",
        )],
        *[f"tests/{name}" for name in (
            "test_ablation.py", "test_amendment_inventory.py",
            "test_evidence_gates.py",
        )],
        *[f"provenance/code_snapshot/{name}" for name in (
            "ablation_core.py", "amendment_core.py", "amendment_inventory.py",
            "build_amended_preflight.py", "build_blocked_preflight.py",
            "evidence_gates.py", "run_ablation_study.sh", "run_study.py",
        )],
        *[f"provenance/tests_snapshot/{name}" for name in (
            "test_ablation.py", "test_amendment_inventory.py",
            "test_evidence_gates.py",
        )],
    ]
    log_mapping = {
        "logs/syntax_compile.log": "logs/preflight_syntax_compile.log",
        "logs/pytest.log": "logs/preflight_pytest.log",
        "logs/reviewer_snapshot_syntax_compile.log": (
            "logs/preflight_reviewer_snapshot_syntax_compile.log"
        ),
        "logs/reviewer_snapshot_pytest.log": (
            "logs/preflight_reviewer_snapshot_pytest.log"
        ),
    }
    snapshot_mapping = {
        "provenance/PREFLIGHT_REPORT.json": (
            "provenance/preflight_snapshot/PREFLIGHT_REPORT.json"
        ),
        "provenance/evidence_gate_registry.json": (
            "provenance/preflight_snapshot/evidence_gate_registry.json"
        ),
        "provenance/source_input_hashes_post.tsv": (
            "provenance/preflight_snapshot/source_input_hashes_post.tsv"
        ),
        "provenance/cuda_capability_probe.json": (
            "provenance/preflight_snapshot/cuda_capability_probe.json"
        ),
        "config/run_identity.lock.json": (
            "provenance/preflight_snapshot/run_identity.lock.json"
        ),
        "OUTPUT_MANIFEST_FINAL.tsv": (
            "provenance/preflight_snapshot/OUTPUT_MANIFEST_FINAL.tsv"
        ),
        "BUNDLE_MANIFEST.tsv": "provenance/preflight_snapshot/BUNDLE_MANIFEST.tsv",
        "__PREFLIGHT_REVIEW_BUNDLE__": (
            "provenance/preflight_snapshot/preflight_review_bundle.zip"
        ),
    }
    return {
        **{value: value for value in direct_files},
        **log_mapping,
        **snapshot_mapping,
    }


def read_reviewed_preflight_members(
    preflight_run: Path,
    relative_paths: Sequence[str],
    *,
    expected_bundle_sha256: str,
) -> tuple[dict[str, bytes], set[str]]:
    """Read reviewed inputs only from the immutable, hash-pinned Preflight ZIP."""

    review_bundle = Path(str(preflight_run) + "_review_bundle.zip")
    if (
        not review_bundle.is_file()
        or review_bundle.is_symlink()
        or sha256_file(review_bundle) != expected_bundle_sha256
    ):
        raise IntegrityError("Reviewed Preflight ZIP does not match its validated hash")
    verification = verify_zip(review_bundle, embedded_run_manifest=True)
    if verification.get("status") != "PASS":
        raise IntegrityError("Reviewed Preflight ZIP failed CRC/member verification")
    with zipfile.ZipFile(review_bundle) as archive:
        prefix = f"{preflight_run.name}/"
        archive_names = archive.namelist()
        if any(
            not name.startswith(prefix) or name.endswith("/")
            for name in archive_names
        ):
            raise IntegrityError("Reviewed Preflight ZIP has an unexpected root/layout")
        reviewed_paths = {name[len(prefix):] for name in archive_names}
        requested = set(map(str, relative_paths))
        missing = sorted(requested - reviewed_paths)
        if missing:
            raise IntegrityError(
                f"Reviewed Preflight ZIP omits requested evidence: {missing}"
            )
        return {
            relative: archive.read(f"{prefix}{relative}")
            for relative in requested
        }, reviewed_paths


def snapshot_smoke_review_evidence(
    *,
    preflight_run: Path,
    run_dir: Path,
    validated_identity: Mapping[str, Any],
    command_line: str,
    split_hash: str,
    amendment03_contract: amendment03_compat.SealedProbeContract | None = None,
    amendment03_allowed_after_hashes: Mapping[str, str] | None = None,
    amendment03_prerun_hashes: Mapping[str, str] | None = None,
    recovery_authorization: str | None = None,
    recovery_allowed_after_hashes: Mapping[str, str] | None = None,
    recovery_prerun_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Copy and pin the reviewed Preflight evidence needed to audit a Smoke Run."""

    preflight_zip = Path(str(preflight_run) + "_review_bundle.zip")
    preflight_zip_verification = verify_zip(
        preflight_zip, embedded_run_manifest=True,
    )
    reference_validation = validated_identity.get("reference_validation") or {}
    if (
        preflight_zip_verification.get("status") != "PASS"
        or reference_validation.get("review_bundle_live_byte_equality_status")
        != "PASS"
        or reference_validation.get("review_bundle_sha256")
        != sha256_file(preflight_zip)
    ):
        raise IntegrityError("Reviewed Preflight ZIP is not the validated sealed artifact")

    copied: dict[str, str] = {}
    with zipfile.ZipFile(preflight_zip) as archive:
        archive_names = set(archive.namelist())
        for source_relative, destination_relative in (
            smoke_copied_evidence_mapping(
                str(validated_identity.get("protocol", "AMENDMENT_01"))
            ).items()
        ):
            if amendment03_contract is not None and (
                source_relative == amendment03_compat.SEALED_PROBE_RELATIVE_PATH
                or source_relative == "__PREFLIGHT_REVIEW_BUNDLE__"
            ):
                continue
            if source_relative == "__PREFLIGHT_REVIEW_BUNDLE__":
                data = preflight_zip.read_bytes()
            else:
                member = f"{preflight_run.name}/{source_relative}"
                if member not in archive_names:
                    raise IntegrityError(
                        f"Reviewed Preflight ZIP omits required evidence: {member}"
                    )
                data = archive.read(member)
            destination = run_dir / destination_relative
            if destination.exists() or destination.is_symlink():
                raise IntegrityError(f"Refusing Smoke evidence overwrite: {destination}")
            atomic_write_bytes(destination, data)
            copied[destination_relative] = sha256_bytes(data)

    sealed_binding = None
    if amendment03_contract is not None:
        sealed_binding = amendment03_compat.copy_sealed_probe_to_smoke(
            run_dir, amendment03_contract,
        )
        atomic_write_json(
            run_dir / "provenance/sealed_probe_binding.json", sealed_binding,
        )
        for relative in (
            amendment03_compat.ROOT_PROBE_DESTINATION,
            amendment03_compat.NESTED_PROBE_DESTINATION,
        ):
            copied[relative] = sha256_file(run_dir / relative)

    recovery_token = (
        recovery_authorization
        or (_AMENDMENT03_AUTHORIZATION if amendment03_contract is not None else None)
    )
    reference = {
        "classification": "VERIFIED",
        "preflight_run": str(preflight_run),
        "preflight_run_id": validated_identity["run_id"],
        "preflight_state": validated_identity["state"],
        "split_hash": split_hash,
        "command_line": command_line,
        "reference_validation": reference_validation,
        "preflight_identity_sha256": copied[
            "provenance/preflight_snapshot/run_identity.lock.json"
        ],
        "preflight_registry_sha256": copied[
            "provenance/preflight_snapshot/evidence_gate_registry.json"
        ],
        "preflight_output_manifest_sha256": copied[
            "provenance/preflight_snapshot/OUTPUT_MANIFEST_FINAL.tsv"
        ],
        "preflight_bundle_manifest_sha256": copied[
            "provenance/preflight_snapshot/BUNDLE_MANIFEST.tsv"
        ],
        "preflight_review_bundle_path": str(preflight_zip),
        "packaged_preflight_review_bundle_path": (
            (
                "NOT_BUNDLED_AMENDMENT04_LINEAGE_ONLY"
                if recovery_token == _AMENDMENT04_AUTHORIZATION
                else "NOT_BUNDLED_AMENDMENT03_RECOVERY_LINEAGE_ONLY"
            )
            if amendment03_contract is not None
            else "provenance/preflight_snapshot/preflight_review_bundle.zip"
        ),
        "preflight_review_bundle_sha256": sha256_file(preflight_zip),
        "preflight_review_bundle_size_bytes": preflight_zip.stat().st_size,
        "preflight_review_bundle_verification": preflight_zip_verification,
        "copied_evidence_sha256": copied,
    }
    if amendment03_contract is not None:
        reference.update({
            "recovery_authorization": recovery_token,
            "reviewed_code_tree": dict(sorted(
                validated_identity.get("code_tree", {}).items()
            )),
            "sealed_probe_binding_sha256": sha256_bytes(
                canonical_json(sealed_binding)
            ),
        })
        if recovery_token == _AMENDMENT04_AUTHORIZATION:
            reference.update({
                "amendment04_allowed_after_hashes": dict(sorted(
                    (recovery_allowed_after_hashes or {}).items()
                )),
                "amendment04_prerun_evidence_sha256": dict(sorted(
                    (recovery_prerun_hashes or {}).items()
                )),
            })
        else:
            reference.update({
                "amendment03_allowed_after_hashes": dict(sorted(
                    (amendment03_allowed_after_hashes or {}).items()
                )),
                "amendment03_prerun_evidence_sha256": dict(sorted(
                    (amendment03_prerun_hashes or {}).items()
                )),
            })
    atomic_write_json(run_dir / "config/preflight_reference.json", reference)
    return reference


def amendment03_code_drift_allowlist(
    reviewed_code_tree: Mapping[str, str],
) -> dict[str, str]:
    """Return the sealed six-path A03 execution drift, not later live drift."""

    reviewed = dict(reviewed_code_tree)
    failed_run = STUDY_ROOT / "results" / (
        amendment04_recovery.FAILED_AMENDMENT03_RUN_ID
    )
    tsv = failed_run / "provenance/executed_code_hashes.tsv"
    expected_tsv_sha256 = amendment04_recovery.FAILED_AMENDMENT03_KEY_HASHES[
        "provenance/executed_code_hashes.tsv"
    ]
    if (
        not tsv.is_file() or tsv.is_symlink()
        or sha256_file(tsv) != expected_tsv_sha256
    ):
        raise IntegrityError("Amendment 03 executed-code TSV pin changed")
    frame = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
    required = {
        "relative_path", "executed_sha256", "snapshot_relative_path",
        "snapshot_sha256",
    }
    if not required <= set(frame.columns) or frame.relative_path.duplicated().any():
        raise IntegrityError("Amendment 03 executed-code TSV schema changed")
    executed: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        relative = str(row.relative_path)
        digest = str(row.executed_sha256)
        snapshot_relative = str(row.snapshot_relative_path)
        snapshot = failed_run / snapshot_relative
        if not (
            _valid_sha256(digest)
            and str(row.snapshot_sha256) == digest
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts
            and not Path(snapshot_relative).is_absolute()
            and ".." not in Path(snapshot_relative).parts
            and snapshot.is_file()
            and not snapshot.is_symlink()
            and sha256_file(snapshot) == digest
        ):
            raise IntegrityError(
                f"Amendment 03 executed snapshot pin changed: {relative}"
            )
        executed[relative] = digest
    changed = {
        relative for relative in set(reviewed) | set(executed)
        if reviewed.get(relative) != executed.get(relative)
    }
    if changed != set(_AMENDMENT03_ALLOWED_CODE_DRIFT_PATHS):
        raise IntegrityError(
            "Amendment 03 code/test drift is not the exact narrow path set: "
            f"changed={sorted(changed)}"
        )
    if set(reviewed) - set(executed):
        raise IntegrityError("Amendment 03 removed a Preflight-reviewed code/test file")
    return {relative: executed[relative] for relative in sorted(changed)}


def _validate_amendment03_command_log(
    *,
    data: bytes,
    relative: str,
    expected_command: Sequence[str],
) -> dict[str, Any]:
    """Validate one exact pre-ID command-log envelope and hash its raw bytes."""

    text = data.decode("utf-8", errors="strict")
    lines = text.splitlines()
    fields: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key in {"COMMAND", "CWD", "EXIT_CODE"}:
            if key in fields:
                raise IntegrityError(f"Duplicate {key} in {relative}")
            fields[key] = value
    marker_order = [
        "STDOUT_BEGIN", "STDOUT_END", "STDERR_BEGIN", "STDERR_END",
    ]
    try:
        marker_indexes = [lines.index(marker) for marker in marker_order]
        observed_command = shlex.split(fields.get("COMMAND", ""))
    except (ValueError, KeyError) as exc:
        raise IntegrityError(
            f"Amendment 03 command log schema failed: {relative}"
        ) from exc
    if not (
        set(fields) == {"COMMAND", "CWD", "EXIT_CODE"}
        and fields["CWD"] == str(STUDY_ROOT)
        and fields["EXIT_CODE"] == "0"
        and observed_command == list(expected_command)
        and all(lines.count(marker) == 1 for marker in marker_order)
        and marker_indexes == sorted(marker_indexes)
        and marker_indexes[0] < marker_indexes[1] < marker_indexes[2]
        < marker_indexes[3]
    ):
        raise IntegrityError(f"Amendment 03 command log contract failed: {relative}")
    return {
        "command": fields["COMMAND"],
        "cwd": fields["CWD"],
        "exit_code": 0,
        "sha256": sha256_bytes(data),
    }


def _validate_amendment03_prerun_evidence() -> dict[str, Any]:
    """Validate the completed pre-ID diagnostics/tests and return exact bytes."""

    sources: dict[str, bytes] = {}
    for source_relative, destination_relative in _AMENDMENT03_PRERUN_COPY_MAPPING.items():
        source = STUDY_ROOT / source_relative
        if not source.is_file() or source.is_symlink() or source.stat().st_size == 0:
            raise IntegrityError(
                f"Amendment 03 pre-Run evidence is missing/unsafe: {source}"
            )
        sources[destination_relative] = source.read_bytes()
    try:
        live = json.loads(sources["provenance/amendment03_live_cuda_binding.json"])
        orphan = json.loads(sources["provenance/xgb211_reload_diagnostic.json"])
    except Exception as exc:
        raise IntegrityError(f"Amendment 03 JSON diagnostic is unreadable: {exc}") from exc
    live_context = live.get("context", {})
    orphan_context = orphan.get("context", {})
    gpu_identity = live_context.get("gpu_identity", {})
    gpu_output = str(gpu_identity.get("stdout", ""))
    if not (
        live.get("status") == "PASS"
        and live.get("cpu_fallback_detected") is False
        and live.get("checks", {}).get("active_fit_cuda") is True
        and live.get("checks", {}).get("reload_cuda") is True
        and live.get("checks", {}).get("model_immutable") is True
        and live_context.get("sys_executable") == str(_AMENDMENT03_CUDA_PYTHON)
        and live_context.get("sys_prefix") == str(_AMENDMENT03_CUDA_PREFIX)
        and live_context.get("xgboost") == "2.1.1"
        and live_context.get("sklearn_is_classifier_xgbclassifier") is False
        and live.get("checks", {}).get("sklearn_classifier_tag_observed") is True
        and live.get("checks", {}).get("nvidia_smi_pass") is True
        and live_context.get("xgboost_build_info", {}).get("USE_CUDA") is True
        and gpu_identity.get("exit_code") == 0
        and gpu_identity.get("stderr") == ""
        and bool(gpu_output.strip())
        and "NVIDIA GeForce RTX 4090" in gpu_output
        and "595.95" in gpu_output
        and orphan.get("status") == "PASS"
        and orphan.get("model_unchanged") is True
        and orphan.get("xgb_classifier", {}).get("is_classifier") is False
        and orphan_context.get("sys_executable") == str(_AMENDMENT03_CUDA_PYTHON)
        and orphan_context.get("xgboost") == "2.1.1"
        and orphan_context.get("scikit_learn") == live_context.get("scikit_learn")
    ):
        raise IntegrityError("Amendment 03 live/reload diagnostic contract failed")
    python = str(_AMENDMENT03_CUDA_PYTHON)
    log_requirements = {
        "logs/amendment03_syntax_compile.log": [
            python, "-m", "py_compile",
            "code/amendment03_compat.py", "code/amendment03_recovery.py",
            "code/amendment_core.py", "code/run_study.py",
            "tests/test_amendment03.py", "tests/test_amendment03_recovery.py",
        ],
        "logs/amendment03_targeted_tests.log": [
            python, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "tests/test_amendment03.py",
            "tests/test_amendment03_recovery.py",
            "tests/test_amendment02_resampling.py",
        ],
        "logs/full_tests.log": [
            python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests",
        ],
        "logs/cuda_reload_probe.log": [
            "AMENDMENT03_ACTIVE_CUDA_PROBE=1", python, "-m", "pytest",
            "-q", "-s", "-p", "no:cacheprovider",
            (
                "tests/test_amendment03.py::"
                "test_amendment03_active_cuda_production_helper"
            ),
        ],
    }
    log_contract: dict[str, dict[str, Any]] = {}
    for relative, expected_command in log_requirements.items():
        log_contract[relative] = _validate_amendment03_command_log(
            data=sources[relative],
            relative=relative,
            expected_command=expected_command,
        )
    return {
        "status": "PASS",
        "source_bytes": sources,
        "destination_sha256": {
            relative: sha256_bytes(data) for relative, data in sources.items()
        },
        "logs": log_contract,
    }


def _assert_amendment03_invocation(
    *,
    preflight_run: Path,
    command_line: str,
    identity: Mapping[str, Any],
    immutable_gate: Mapping[str, Any],
) -> None:
    """Require the exact one-shot CLI, Preflight, interpreter, and prompt pins."""

    try:
        tokens = shlex.split(command_line)
    except ValueError as exc:
        raise IntegrityError("Amendment 03 command line is malformed") from exc
    forbidden = {"--full", "--resume", "--stability", "--preflight", "--audit"}
    prompt = Path(
        "/REVIEWER_INPUT_ROOT/03_RUN_NOW_Amendment03_Corrigendum_One_CUDA_Smoke.txt"
    )
    expected_preflight = STUDY_ROOT / "results" / (
        amendment03_recovery.ACCEPTED_PREFLIGHT_RUN_ID
    )
    preflight_values = [
        tokens[index + 1] for index, token in enumerate(tokens[:-1])
        if token == "--preflight-run"
    ]
    if not (
        command_line
        and tokens.count("--smoke") == 1
        and tokens.count("--amendment-03-corrigendum-one") == 1
        and tokens.count("--preflight-run") == 1
        and len(preflight_values) == 1
        and Path(preflight_values[0]).resolve() == expected_preflight.resolve()
        and not (set(tokens) & forbidden)
        and Path(preflight_run).resolve() == expected_preflight.resolve()
        and immutable_gate.get("accepted_preflight_run_id")
        == amendment03_recovery.ACCEPTED_PREFLIGHT_RUN_ID
        and identity.get("run_id") == amendment03_recovery.ACCEPTED_PREFLIGHT_RUN_ID
        and Path(sys.executable).resolve() == _AMENDMENT03_CUDA_PYTHON.resolve()
        and Path(sys.prefix).resolve() == _AMENDMENT03_CUDA_PREFIX.resolve()
        and Path(os.environ.get("ABLATION_PYTHON", "")).resolve()
        == _AMENDMENT03_CUDA_PYTHON.resolve()
        and prompt.is_file()
        and not prompt.is_symlink()
        and sha256_file(prompt)
        == "5ffa3fc2cec3b13e8e4d5d0ee2e18a906ee0677cd27502f41b0767f78fd0d2bc"
    ):
        raise IntegrityError("Amendment 03 exact invocation/immutable identity failed")


def _assert_amendment04_invocation(
    *,
    preflight_run: Path,
    command_line: str,
    identity: Mapping[str, Any],
    immutable_gate: Mapping[str, Any],
) -> None:
    """Require the exact one-shot Amendment 04 command and runtime pins."""

    try:
        tokens = shlex.split(command_line)
    except ValueError as exc:
        raise IntegrityError("Amendment 04 command line is malformed") from exc
    expected_preflight = STUDY_ROOT / "results" / (
        amendment04_recovery.ACCEPTED_PREFLIGHT_RUN_ID
    )
    expected_tokens = [
        "code/run_study.py",
        "--smoke",
        "--preflight-run",
        str(expected_preflight),
        "--amendment-04-csv-roundtrip-one",
    ]
    accepted = immutable_gate.get("accepted_preflight", {}) or {}
    if not (
        command_line
        and tokens == expected_tokens
        and Path(preflight_run).resolve() == expected_preflight.resolve()
        and identity.get("run_id") == amendment04_recovery.ACCEPTED_PREFLIGHT_RUN_ID
        and accepted.get("run_id") == amendment04_recovery.ACCEPTED_PREFLIGHT_RUN_ID
        and accepted.get("review_zip_sha256")
        == amendment04_recovery.ACCEPTED_PREFLIGHT_ZIP_SHA256
        and Path(sys.executable).resolve() == amendment04_recovery.CUDA_PYTHON.resolve()
        and Path(sys.prefix).resolve() == amendment04_recovery.CUDA_ENV.resolve()
        and Path(os.environ.get("ABLATION_PYTHON", "")).resolve()
        == amendment04_recovery.CUDA_PYTHON.resolve()
        and amendment04_recovery.PROMPT_PATH.is_file()
        and not amendment04_recovery.PROMPT_PATH.is_symlink()
        and sha256_file(amendment04_recovery.PROMPT_PATH)
        == amendment04_recovery.PROMPT_SHA256
    ):
        raise IntegrityError("Amendment 04 exact invocation/immutable identity failed")


def authorize_cuda_smoke_reference(
    *,
    preflight_run: Path,
    validated_identity: Mapping[str, Any],
    preflight_report: Mapping[str, Any],
    features_by_variant: Mapping[str, Any],
    registry_payload: Mapping[str, Any],
    execution_ledger: Mapping[str, Any],
    reviewed_member_paths: set[str],
    amendment03_allowed_after_hashes: Mapping[str, str] | None = None,
    amendment04_reference_allowed_after_hashes: Mapping[str, str] | None = None,
) -> evidence_workflow.WorkflowState:
    """Re-derive every preflight authorization predicate before any model work."""

    validate_run_transition(validated_identity, "CUDA_SMOKE_NON_SCIENTIFIC")
    code_tree = validated_identity.get("code_tree", {})
    expected_code_tree_paths = {
        path.relative_to(STUDY_ROOT).as_posix()
        for path in [
            *sorted((STUDY_ROOT / "code").glob("*.py")),
            *sorted((STUDY_ROOT / "code").glob("*.sh")),
            *sorted((STUDY_ROOT / "tests").glob("*.py")),
        ]
        if path.is_file()
    }
    if (
        amendment03_allowed_after_hashes is None
        and amendment04_reference_allowed_after_hashes is None
    ):
        if not isinstance(code_tree, dict) or set(code_tree) != expected_code_tree_paths:
            raise IntegrityError("Preflight identity does not pin the executing code tree")
        code_drift = [
            relative for relative, expected in code_tree.items()
            if not (STUDY_ROOT / relative).is_file()
            or sha256_file(STUDY_ROOT / relative) != expected
        ]
        if code_drift:
            raise IntegrityError(
                f"Executing code/tests differ from reviewed Preflight: {code_drift}"
            )
    elif amendment03_allowed_after_hashes is not None:
        observed_allowlist = amendment03_code_drift_allowlist(code_tree)
        if dict(amendment03_allowed_after_hashes) != observed_allowlist:
            raise IntegrityError("Amendment 03 hash-specific code drift changed")
    else:
        observed_allowlist = amendment04_recovery.reference_code_drift_allowlist(
            code_tree, study_root=STUDY_ROOT,
        )
        if dict(amendment04_reference_allowed_after_hashes or {}) != observed_allowlist:
            raise IntegrityError("Amendment 04 hash-specific code drift changed")
    gate_inputs: dict[str, dict[str, Any]] = {}
    stored_gates = registry_payload.get("gates", {})
    for specification in evidence_workflow.DEFAULT_GATE_SPECS:
        if specification.key not in stored_gates:
            raise IntegrityError(
                f"Preflight registry omits gate {specification.key}"
            )
        stored = stored_gates[specification.key]
        gate_inputs[specification.key] = {
            "classification": stored.get("classification"),
            "passed": stored.get("passed"),
            "evidence_refs": stored.get("evidence_refs"),
            "detail": stored.get("detail", ""),
        }
    registry = evidence_workflow.build_gate_registry(gate_inputs)
    for key, gate in registry.evidence.items():
        stored = stored_gates[key]
        if stored.get("accepted") is not gate.accepted:
            raise IntegrityError(f"Preflight gate accepted flag drifted: {key}")
        if stored.get("status") != gate.status_token:
            raise IntegrityError(f"Preflight gate status token drifted: {key}")
        for relative in gate.evidence_refs:
            if relative not in reviewed_member_paths:
                raise IntegrityError(
                    f"Preflight gate {key} references absent evidence: {relative}"
                )

    ledger = execution_ledger
    no_pca_variant = features_by_variant.get("no_pca", {})
    no_pca_runnable = no_pca_variant.get("status") == "RUNNABLE"
    no_pca_status = (
        "VERIFIED_EXACT_RAW_EMBEDDINGS_AVAILABLE"
        if no_pca_runnable else no_pca_variant.get("status")
    )
    facts = evidence_workflow.WorkflowFacts(
        phase="AMENDED_PREFLIGHT",
        official_smoke_variants_completed=int(
            ledger.get("official_smoke_variants_completed", -1)
        ),
        exploratory_smoke_variants_completed=int(
            ledger.get("exploratory_smoke_variants_completed", -1)
        ),
        full_scientific_run_executed=ledger.get(
            "full_scientific_run_executed"
        ),
        full_variants_completed=int(ledger.get("full_variants_completed", -1)),
        no_pca_variant_status=str(no_pca_status),
        no_pca_variant_runnable=no_pca_runnable,
        protocol_amendment=str(
            validated_identity.get("protocol", "AMENDMENT_01")
        ),
    )
    state = evidence_workflow.derive_workflow_state(registry, facts)
    stored_derived = registry_payload.get("derived_workflow", {})
    expected_derived = {
        "run_kind": state.run_kind,
        "run_state": state.run_state,
        "amended_preflight_status": state.amended_preflight_status,
        "data_source_decision": state.data_source_decision,
        "split_locked": state.split_locked,
        "split_lock_status": state.split_lock_status,
        "smoke_eligible": state.smoke_eligible,
        "cuda_smoke_status": state.cuda_smoke_status,
        "ready_for_full_awaiting_external_review": (
            state.ready_for_full_awaiting_external_review
        ),
        "blocker_codes": list(state.blocker_codes),
    }
    if stored_derived != expected_derived:
        raise IntegrityError("Stored Preflight workflow differs from fresh derivation")
    if (
        not state.smoke_eligible
        or state.run_state != "PREFLIGHT_COMPLETE"
        or validated_identity.get("state") != state.run_state
        or validated_identity.get("split_locked") is not True
        or preflight_report.get("global_blockers")
        or preflight_report.get("status") != state.amended_preflight_status
    ):
        raise ScientificBlocker(
            "CUDA_SMOKE_NOT_AUTHORIZED_BY_FRESH_PREFLIGHT_DERIVATION"
        )
    return state


def smoke_execution_ledger(
    run_dir: Path,
    *,
    command_line: str,
    runnable_official: Sequence[str],
) -> dict[str, Any]:
    """Derive Smoke and no-Full facts from the current Run and invocation."""

    try:
        invocation = shlex.split(command_line)
        invocation_parse_status = "PASS"
    except ValueError as exc:
        invocation = []
        invocation_parse_status = f"FAIL:{type(exc).__name__}"
    forbidden_arguments = {
        "--full", "--resume", "--stability", "--multiseed", "--multi-seed",
    }
    invoked_forbidden = sorted(set(invocation) & forbidden_arguments)
    invocation_is_smoke = "--smoke" in invocation

    report_path = run_dir / "tables/SMOKE_REPORT.csv"
    completed: set[str] = set()
    report_status = "ABSENT"
    unknown: list[str] = []
    duplicates: list[str] = []
    conflicts: list[str] = []
    invalid_status_rows = 0
    allowed = set(runnable_official) | {EXPLORATORY_VARIANT}
    if report_path.is_file():
        try:
            report = pd.read_csv(report_path)
            if {"variant", "status"} <= set(report):
                variant = report.variant.astype(str)
                status = report.status.astype(str).str.upper()
                unknown = sorted(set(variant) - allowed)
                duplicates = sorted(variant[variant.duplicated(keep=False)].unique())
                conflicts = sorted(
                    key for key, values in pd.DataFrame({
                        "variant": variant, "status": status,
                    }).groupby("variant", sort=False).status
                    if values.nunique() > 1
                )
                valid_statuses = {"PASS", "FAIL", "INCOMPLETE"}
                invalid_status_rows = int((~status.isin(valid_statuses)).sum())
                if not (unknown or duplicates or conflicts or invalid_status_rows):
                    completed = set(variant[status == "PASS"])
                    report_status = "PARSED_VALID"
                else:
                    report_status = "INVALID_CONTENT"
            else:
                report_status = "INVALID_SCHEMA"
        except Exception as exc:
            report_status = f"UNREADABLE:{type(exc).__name__}"

    forbidden_directories = {"models", "predictions", "statistics", "timing", "resources"}
    forbidden_names = {
        "ablation_summary_raw.csv", "ablation_summary_weighted_sensitivity.csv",
        "ablation_delta_vs_full.csv", "feature_manifest_by_variant.tsv",
        "resampling_manifest_by_variant.tsv", "metrics_recomputation_checks.csv",
    }
    artifacts = sorted(path for path in run_dir.rglob("*") if path.is_file())
    full_artifacts = sorted(
        path.relative_to(run_dir).as_posix()
        for path in artifacts
        if (
            forbidden_directories & set(path.relative_to(run_dir).parts[:-1])
            or path.name in forbidden_names
        )
    )
    lowered = [path.relative_to(run_dir).as_posix().lower() for path in artifacts]
    stability_artifacts = sorted(
        value for value in lowered
        if "stability" in value or "seed_stability" in value
    )
    multiseed_artifacts = sorted(
        value for value in lowered
        if "multiseed" in value or "multi_seed" in value or "multi-seed" in value
    )
    completed_full_variants: set[str] = set()
    invalid_full_completion_manifests: list[str] = []
    models_root = run_dir / "models"
    if models_root.is_dir():
        for completion in sorted(models_root.glob("*/completion_manifest.json")):
            variant_name = completion.parent.name
            status_path = completion.parent / "STATUS.txt"
            try:
                record = json.loads(completion.read_text())
                files = record.get("files", {})
                valid = bool(
                    record.get("variant") == variant_name
                    and status_path.is_file()
                    and status_path.read_text().strip() == "DONE"
                    and isinstance(files, dict)
                    and files
                    and all(
                        (run_dir / relative).is_file()
                        and sha256_file(run_dir / relative) == expected_hash
                        for relative, expected_hash in files.items()
                    )
                )
            except Exception:
                valid = False
            if valid:
                completed_full_variants.add(variant_name)
            else:
                invalid_full_completion_manifests.append(
                    completion.relative_to(run_dir).as_posix()
                )
    full_executed = bool(
        invoked_forbidden or full_artifacts or stability_artifacts or multiseed_artifacts
        or completed_full_variants or invalid_full_completion_manifests
    )
    checks = {
        "invocation_parsed": invocation_parse_status == "PASS",
        "invocation_is_cuda_smoke": invocation_is_smoke,
        "no_forbidden_execution_arguments": not invoked_forbidden,
        "smoke_report_schema_and_rows_valid": report_status == "PARSED_VALID",
        "no_full_scientific_artifacts": not full_artifacts,
        "no_stability_artifacts": not stability_artifacts,
        "no_multiseed_artifacts": not multiseed_artifacts,
        "no_full_completion_manifests": not (
            completed_full_variants or invalid_full_completion_manifests
        ),
    }
    return {
        "classification": "VERIFIED" if all(checks.values()) else "BLOCKED",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "invocation_argv": invocation,
        "invocation_parse_status": invocation_parse_status,
        "forbidden_arguments_detected": invoked_forbidden,
        "smoke_report_status": report_status,
        "smoke_unknown_variants": unknown,
        "smoke_duplicate_variants": duplicates,
        "smoke_conflicting_variants": conflicts,
        "smoke_invalid_status_rows": invalid_status_rows,
        "smoke_variants_completed_ids": sorted(completed),
        "official_smoke_variants_completed": len(completed & set(runnable_official)),
        "exploratory_smoke_variants_completed": int(EXPLORATORY_VARIANT in completed),
        "full_scientific_artifacts_detected": full_artifacts,
        "validated_full_variant_completions": sorted(completed_full_variants),
        "invalid_full_completion_manifests": invalid_full_completion_manifests,
        "stability_artifacts_detected": stability_artifacts,
        "multiseed_artifacts_detected": multiseed_artifacts,
        "full_scientific_run_executed": full_executed,
        "full_variants_completed": len(completed_full_variants),
    }


def _smoke_registry_payload(
    registry: evidence_workflow.GateRegistry,
    state: evidence_workflow.WorkflowState,
) -> dict[str, Any]:
    return {
        "classification": "VERIFIED",
        "gates": {
            key: {
                "classification": gate.classification.value,
                "passed": gate.passed,
                "accepted": gate.accepted,
                "status": gate.status_token,
                "evidence_refs": list(gate.evidence_refs),
                "detail": gate.detail,
            }
            for key, gate in registry.evidence.items()
        },
        "derived": {
            "run_kind": state.run_kind,
            "run_state": state.run_state,
            "cuda_smoke_status": state.cuda_smoke_status,
            "ready_for_full_awaiting_external_review": (
                state.ready_for_full_awaiting_external_review
            ),
            "blockers": list(state.blocker_codes),
        },
    }


def promote_smoke_package_gate(
    gate_inputs: Mapping[str, Mapping[str, Any]],
    facts: evidence_workflow.WorkflowFacts,
    verification: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], evidence_workflow.GateRegistry, evidence_workflow.WorkflowState]:
    """Promote package evidence only after a complete independent reopen passes."""

    completeness = verification.get("package_completeness", {})
    passed = bool(
        verification.get("crc_status") == "PASS"
        and verification.get("independent_reopen_member_verification") == "PASS"
        and not verification.get("verification_failures")
        and completeness.get("required_files_status") == "PASS"
        and completeness.get("status_identity_match") == "PASS"
        and completeness.get("smoke_semantic_completeness", {}).get("status")
        == "PASS"
        and completeness.get("smoke_semantic_completeness", {}).get("phase")
        == "STAGING"
    )
    promoted = {key: dict(value) for key, value in gate_inputs.items()}
    promoted["output_manifest_zip_verification"] = {
        "classification": "VERIFIED",
        "passed": passed,
        "evidence_refs": ["provenance/package_staging_verification.json"],
        "detail": (
            "A complete staging bundle passed CRC, required-file validation, "
            "and independent member size/hash verification before final sealing."
        ),
    }
    registry = evidence_workflow.build_gate_registry(promoted)
    state = evidence_workflow.derive_workflow_state(registry, facts)
    return promoted, registry, state


def _smoke_status_text(
    *,
    state: evidence_workflow.WorkflowState,
    run_dir: Path,
    preflight_run: Path,
    preflight_report: Mapping[str, Any],
    split_hash: str,
    model_bundle: Mapping[str, Any] | None,
    review_bundle: Mapping[str, Any] | None = None,
    amendment03_recovery: Mapping[str, Any] | None = None,
    amendment04_recovery_validation: Mapping[str, Any] | None = None,
) -> str:
    r92 = preflight_report["r92_control"]
    return evidence_workflow.render_status_lines(state, extra_lines={
        "AMENDED_PREFLIGHT_RUN": str(preflight_run),
        "CUDA_SMOKE_RUN": str(run_dir),
        "TRAINING_TABLE_PATH": preflight_report["training_table_path"],
        "TRAINING_TABLE_SHA256": preflight_report["training_table_sha256"],
        "LOCKED_SPLIT_SHA256": split_hash,
        "R92_MAX_PROBABILITY_DIFFERENCE": r92["max_probability_difference"],
        "REVIEW_BUNDLE_PATH": (
            review_bundle["bundle_path"] if review_bundle else "PENDING_FINAL_SEAL"
        ),
        "REVIEW_BUNDLE_SHA256": (
            review_bundle["bundle_sha256"] if review_bundle else "PENDING_FINAL_SEAL"
        ),
        "SMOKE_MODELS_BUNDLE_PATH": (
            model_bundle["portable_relative_to_run"]
            if model_bundle else "NOT_CREATED_NO_SMOKE_MODELS"
        ),
        "SMOKE_MODELS_BUNDLE_SHA256": (
            model_bundle["sha256"] if model_bundle else "NOT_APPLICABLE"
        ),
        "SMOKE_MODELS_BUNDLE_RUN_ID": (
            model_bundle["run_id"] if model_bundle else "NOT_APPLICABLE"
        ),
        "CLASSIFICATION": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
        "EXTERNAL_AUDIT_SCORED": "NO",
        **(
            {"FULL_AUTHORIZED": "NO"}
            if preflight_report.get("protocol") == "AMENDMENT_02" else {}
        ),
        **(
            {
                "RECOVERY_AUTHORIZATION": _AMENDMENT03_AUTHORIZATION,
                "AMENDMENT03_REPLACEMENT_RUN_AUTHORIZATION": "1_OF_1_CONSUMED",
                "PREVIOUS_PROMPT5_PRE_EDIT_STOP_ACKNOWLEDGED": "YES",
                "SEALED_PREFLIGHT_PROBE_SHA256": amendment03_recovery.get(
                    "sealed_probe_sha256", ""
                ),
                "RECOVERY_LINEAGE_SHA256": amendment03_recovery.get(
                    "recovery_lineage_sha256", ""
                ),
                "FULL_TRAINING_EXECUTED": "NO",
                "SCIENTIFIC_RESULTS_CLAIMED": "NO",
            }
            if amendment03_recovery else {}
        ),
        **(
            {
                "RECOVERY_AUTHORIZATION": _AMENDMENT04_AUTHORIZATION,
                "AMENDMENT03_AUTHORIZATION": "1_OF_1_CONSUMED",
                "AMENDMENT04_FRESH_RUN_AUTHORIZATION": "1_OF_1_CONSUMED",
                "REFERENCE_AMENDMENT03_RUN": (
                    amendment04_recovery.FAILED_AMENDMENT03_RUN_ID
                ),
                "CSV_READER_REPAIR": (
                    "PANDAS_C_ENGINE_FLOAT_PRECISION_ROUND_TRIP"
                ),
                "CSV_DEFAULT_MISMATCH_COUNT": "22",
                "CSV_ROUND_TRIP_MISMATCH_COUNT": "0",
                "SEALED_PREFLIGHT_PROBE_SHA256": (
                    amendment04_recovery_validation.get(
                        "sealed_probe_sha256", ""
                    )
                ),
                "AMENDMENT04_LINEAGE_SHA256": (
                    amendment04_recovery_validation.get(
                        "lineage_sha256", ""
                    )
                ),
                "FULL_TRAINING_EXECUTED": "NO",
                "SCIENTIFIC_RESULTS_CLAIMED": "NO",
            }
            if amendment04_recovery_validation else {}
        ),
    })


def failed_smoke_model_bundle_record(
    run_id: str,
    completed_variant_ids: Sequence[str],
    error: Exception | str,
) -> dict[str, Any]:
    """Describe a failed companion-model publication without claiming a ZIP."""

    failure = (
        error if isinstance(error, str)
        else f"{type(error).__name__}: {error}"
    )
    collision_path = STUDY_ROOT / "results" / (
        f"{run_id}_NON_SCIENTIFIC_smoke_models.zip"
    )
    collision = _path_collision_evidence(collision_path)
    return {
        "classification": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
        "status": "FAIL",
        "path": "NOT_CREATED_MODEL_BUNDLE_FAILURE",
        "portable_relative_to_run": "NOT_CREATED_MODEL_BUNDLE_FAILURE",
        "sha256": "NOT_APPLICABLE",
        "run_id": run_id,
        "completed_variant_ids": sorted(map(str, completed_variant_ids)),
        "failure": failure,
        "preexisting_unowned_collision": collision,
    }


def _write_smoke_blockers(
    run_dir: Path,
    state: evidence_workflow.WorkflowState,
) -> None:
    rows = [
        f"- `{blocker.code}` ({blocker.classification.value}/{blocker.status}): "
        f"{blocker.reason} Evidence: {', '.join(blocker.evidence_refs) or 'runtime facts'}."
        for blocker in state.blockers
    ] or ["- None."]
    atomic_write_text(
        run_dir / "BLOCKERS.md",
        "# CUDA Smoke Blockers\n\n"
        + "\n".join(rows)
        + "\n\nThis Run is non-scientific diagnostic evidence. Full, resume, "
        "stability, and multi-seed execution remain unauthorized.\n",
    )


def _remove_publication_with_token(
    path: Path,
    token: PublicationOwnershipToken,
) -> bool:
    """Atomically isolate a pathname before deciding whether its inode is ours."""

    path = _publication_key(path)
    quarantine_dir = Path(tempfile.mkdtemp(
        prefix=f".{path.name}.rollback-", dir=path.parent,
    ))
    os.chmod(quarantine_dir, 0o700)
    quarantine = quarantine_dir / "publication"
    try:
        try:
            os.replace(path, quarantine)
        except FileNotFoundError:
            return False
        if _publication_matches_token(quarantine, token):
            quarantine.unlink()
            return True
        if _rename_no_clobber(quarantine, path):
            return False
        raise IntegrityError(
            "Publication changed during cleanup; preserved unexpected object at "
            f"{quarantine} because {path} is concurrently occupied"
        )
    finally:
        with contextlib.suppress(OSError):
            quarantine_dir.rmdir()


def remove_owned_publication(
    path: Path,
    *,
    ownership: MutableMapping[Path, PublicationOwnershipToken],
) -> bool:
    """Remove only the exact publication authorized by an explicit token map."""

    key = _publication_key(path)
    token = ownership.get(key)
    if token is None:
        return False
    try:
        return _remove_publication_with_token(path, token)
    finally:
        if ownership.get(key) == token:
            ownership.pop(key, None)


def remove_owned_publication_required(
    path: Path,
    *,
    ownership: MutableMapping[Path, PublicationOwnershipToken],
    purpose: str,
) -> None:
    removed = remove_owned_publication(path, ownership=ownership)
    if not removed or os.path.lexists(path):
        raise IntegrityError(f"Owned {purpose} publication changed before cleanup")


def rollback_smoke_final_seal_failure(
    *,
    run_dir: Path,
    preflight_run: Path,
    preflight_report: Mapping[str, Any],
    split_hash: str,
    model_bundle: Mapping[str, Any] | None,
    gate_inputs: Mapping[str, Mapping[str, Any]],
    facts: evidence_workflow.WorkflowFacts,
    smoke_metrics: dict[str, Any],
    error: Exception,
    ownership: MutableMapping[Path, PublicationOwnershipToken],
    amendment03_replacement: bool = False,
    amendment03_recovery_validation: Mapping[str, Any] | None = None,
    amendment04_csv_roundtrip: bool = False,
    amendment04_recovery_validation: Mapping[str, Any] | None = None,
    command_line: str = "",
) -> evidence_workflow.WorkflowState:
    """Remove any canonical publication and persist a fail-closed live status."""

    for stale_manifest in (
        run_dir / "OUTPUT_MANIFEST_FINAL.tsv", run_dir / "BUNDLE_MANIFEST.tsv",
    ):
        remove_owned_publication(stale_manifest, ownership=ownership)
    canonical_bundle = STUDY_ROOT / "results" / f"{run_dir.name}_review_bundle.zip"
    remove_owned_publication(canonical_bundle, ownership=ownership)
    adjacent = STUDY_ROOT / "results" / f"{run_dir.name}_package_verification.json"
    remove_owned_publication(adjacent, ownership=ownership)
    compatibility_recovery = amendment03_replacement or amendment04_csv_roundtrip
    if compatibility_recovery:
        if model_bundle and Path(str(model_bundle.get("path", ""))).is_absolute():
            remove_owned_publication(
                Path(str(model_bundle["path"])), ownership=ownership,
            )
        model_adjacent = STUDY_ROOT / "results" / (
            f"{run_dir.name}_NON_SCIENTIFIC_smoke_models_"
            "package_verification.json"
        )
        remove_owned_publication(model_adjacent, ownership=ownership)
        model_verification = (
            run_dir / "provenance/smoke_model_bundle_verification.json"
        )
        if model_verification.is_file() and not model_verification.is_symlink():
            model_verification.unlink()

    failed_inputs = {key: dict(value) for key, value in gate_inputs.items()}
    failed_inputs["output_manifest_zip_verification"] = {
        "classification": "VERIFIED",
        "passed": False,
        "evidence_refs": ["provenance/package_staging_verification.json"],
        "detail": (
            "Final atomic bundle publication/reopen failed: "
            f"{type(error).__name__}: {error}"
        ),
    }
    for key in ("cuda_smoke_all_runnable", "model_save_reload"):
        failed_inputs[key] = {
            "classification": "VERIFIED",
            "passed": False,
            "evidence_refs": [
                "metrics/SMOKE_REPORT.json",
                "logs/final_package_failure.txt",
            ],
            "detail": (
                "Final Smoke semantic/package sealing did not complete: "
                f"{type(error).__name__}: {error}"
            ),
        }
    failed_registry = evidence_workflow.build_gate_registry(failed_inputs)
    failed_state = evidence_workflow.derive_workflow_state(failed_registry, facts)
    atomic_write_json(
        run_dir / "provenance/evidence_gate_registry.json",
        _smoke_registry_payload(failed_registry, failed_state),
    )
    identity_path = run_dir / "config/run_identity.lock.json"
    if identity_path.is_file() and not identity_path.is_symlink():
        failed_identity = json.loads(identity_path.read_text())
        failed_identity["state"] = failed_state.run_state
        if compatibility_recovery:
            failed_identity.update({
                "recovery_authorization": (
                    _AMENDMENT04_AUTHORIZATION
                    if amendment04_csv_roundtrip
                    else _AMENDMENT03_AUTHORIZATION
                ),
                "replacement_run_authorization": "1_OF_1_CONSUMED",
                "full_authorized": False,
                "stability_authorized": False,
                "full_training_executed": False,
                "scientific_results_claimed": False,
            })
            if amendment04_csv_roundtrip:
                failed_identity["amendment03_authorization"] = "1_OF_1_CONSUMED"
        atomic_write_json(identity_path, failed_identity)
    smoke_metrics.update({
        "state": failed_state.run_state,
        "cuda_smoke_status": failed_state.cuda_smoke_status,
        "ready_for_full_awaiting_external_review": False,
        "blockers": list(failed_state.blocker_codes),
        "output_manifest_zip_verification": "FAIL",
        "final_seal_failure": f"{type(error).__name__}: {error}",
        **({
            "smoke_models_bundle": {
                "path": "NOT_CREATED_NO_SMOKE_MODELS",
                "sha256": "NOT_APPLICABLE",
            },
        } if compatibility_recovery else {}),
    })
    atomic_write_json(run_dir / "metrics/SMOKE_REPORT.json", smoke_metrics)
    _write_smoke_blockers(run_dir, failed_state)
    atomic_write_text(run_dir / "RUN_STATUS.txt", _smoke_status_text(
        state=failed_state,
        run_dir=run_dir,
        preflight_run=preflight_run,
        preflight_report=preflight_report,
        split_hash=split_hash,
        model_bundle=None if compatibility_recovery else model_bundle,
        review_bundle={
            "bundle_path": "FINAL_SEAL_FAILED_NO_CANONICAL_BUNDLE",
            "bundle_sha256": "NOT_APPLICABLE",
        },
        amendment03_recovery=amendment03_recovery_validation,
        amendment04_recovery_validation=amendment04_recovery_validation,
    ))
    if compatibility_recovery:
        traceback_text = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        structured_failure = (
            "PHASE=CUDA_SMOKE_REVIEW_PACKAGE\n"
            "ACTIVE_VARIANT=NONE\n"
            "ACTIVE_VARIANT_INDEX=NONE\n"
            f"CANONICAL_INVOCATION={command_line}\n"
            f"EXCEPTION_TYPE={type(error).__name__}\n"
            f"EXCEPTION_REPR={error!r}\n"
            f"RUN_PATH={run_dir}\n"
            f"LOG_PATH={run_dir / 'logs/final_package_failure.txt'}\n"
            "TRACEBACK_BEGIN\n"
            + traceback_text.rstrip("\n")
            + "\nTRACEBACK_END\n"
        )
        final_failure_path = run_dir / "logs/final_package_failure.txt"
        atomic_write_text(final_failure_path, structured_failure)
        smoke_failure_path = run_dir / "logs/smoke_failure.txt"
        existing_primary = (
            smoke_failure_path.read_text()
            if smoke_failure_path.is_file() and not smoke_failure_path.is_symlink()
            else ""
        )
        if not (
            "PHASE=" in existing_primary
            and "EXCEPTION_TYPE=" in existing_primary
            and "TRACEBACK_BEGIN" in existing_primary
            and "TRACEBACK_END" in existing_primary
        ):
            atomic_write_text(
                smoke_failure_path,
                structured_failure.replace(
                    f"LOG_PATH={final_failure_path}",
                    f"LOG_PATH={smoke_failure_path}",
                ),
            )
    else:
        atomic_write_text(
            run_dir / "logs/final_package_failure.txt", repr(error) + "\n",
        )
    return failed_state


def assert_amendment02_cuda_runtime(
    *,
    identity: Mapping[str, Any],
    report: Mapping[str, Any],
    probe: Mapping[str, Any],
) -> None:
    """Bind the executing interpreter and XGBoost binary to the sealed probe."""

    if str(identity.get("protocol")) != "AMENDMENT_02":
        return
    environment_path = Path(str(probe.get("environment_path", "")))
    build_info = xgb.build_info()
    library = Path(str(build_info.get("libxgboost", "")))
    reported_cuda = report.get("cuda_capability", {})
    expected_library_sha = str(probe.get("libxgboost_sha256", ""))
    probe_interpreter = Path(str(probe.get("interpreter", "")))
    if not (
        probe.get("classification") == "VERIFIED"
        and probe.get("status") == "PASS"
        and probe.get("xgboost_version") == "2.1.1"
        and probe.get("build_info", {}).get("USE_CUDA") is True
        and probe.get("requested_device") == "cuda"
        and probe.get("tree_method") == "hist"
        and probe.get("training_status") == "PASS"
        and probe.get("save_reload_predict_status") == "PASS"
        and probe.get("cpu_fallback_detected") is False
        and environment_path.is_dir()
        and environment_path.resolve() == Path(sys.prefix).resolve()
        and probe_interpreter.is_file()
        and probe_interpreter.resolve() == Path(sys.executable).resolve()
        and str(identity.get("cuda_environment_path")) == str(environment_path)
        and reported_cuda.get("environment_path") == str(environment_path)
        and reported_cuda.get("status") == "PASS"
        and probe.get("visible_device_configuration") == {
            "CUDA_VISIBLE_DEVICES": os.environ.get(
                "CUDA_VISIBLE_DEVICES", "<UNSET>"
            ),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get(
                "NVIDIA_VISIBLE_DEVICES", "<UNSET>"
            ),
        }
        and xgb.__version__ == "2.1.1"
        and build_info.get("USE_CUDA") is True
        and library.is_file()
        and environment_path.resolve() in library.resolve().parents
        and Path(str(probe.get("libxgboost_path", ""))).resolve()
        == library.resolve()
        and _valid_sha256(expected_library_sha)
        and sha256_file(library) == expected_library_sha
    ):
        raise ScientificBlocker(
            "AMENDMENT_02_CUDA_RUNTIME_DIFFERS_FROM_SEALED_ACTIVE_PROBE"
        )


def existing_cuda_smoke_outputs() -> list[str]:
    """Return any prior Smoke Run/publication that would violate the one-run limit."""

    results = STUDY_ROOT / "results"
    if not results.is_dir():
        return []
    return sorted(
        path.name for path in results.iterdir()
        if "cuda_smoke" in path.name.lower()
    )


def _write_amendment03_structured_failure(
    *,
    run_dir: Path,
    phase: str,
    command_line: str,
    error: Exception,
    active_variant: str | None = None,
    active_variant_index: int | None = None,
) -> None:
    """Persist a fail-closed, machine-readable one-shot recovery failure."""

    traceback_text = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    fields = [
        f"PHASE={phase}",
        f"ACTIVE_VARIANT={active_variant or 'NONE'}",
        f"ACTIVE_VARIANT_INDEX={active_variant_index or 'NONE'}",
        f"CANONICAL_INVOCATION={command_line}",
        f"EXCEPTION_TYPE={type(error).__name__}",
        f"EXCEPTION_REPR={error!r}",
        f"RUN_PATH={run_dir}",
        f"LOG_PATH={run_dir / 'logs/smoke_failure.txt'}",
        "TRACEBACK_BEGIN",
        traceback_text.rstrip("\n"),
        "TRACEBACK_END",
    ]
    atomic_write_text(run_dir / "logs/smoke_failure.txt", "\n".join(fields) + "\n")
    atomic_write_json(run_dir / "config/run_identity.lock.json", {
        "run_id": run_dir.name,
        "protocol": "AMENDMENT_02",
        "run_kind": "CUDA_SMOKE_NON_SCIENTIFIC",
        "state": "SMOKE_INCOMPLETE",
        "recovery_authorization": _AMENDMENT03_AUTHORIZATION,
        "replacement_run_authorization": "1_OF_1_CONSUMED",
        "permitted_transitions": ["PACKAGE_REVIEW_EVIDENCE"],
        "full_authorized": False,
        "stability_authorized": False,
    })
    atomic_write_text(
        run_dir / "RUN_STATUS.txt",
        "RUN_KIND=CUDA_SMOKE_NON_SCIENTIFIC\n"
        "RUN_STATE=SMOKE_INCOMPLETE\n"
        "PROTOCOL_AMENDMENT=AMENDMENT_02\n"
        f"RECOVERY_AUTHORIZATION={_AMENDMENT03_AUTHORIZATION}\n"
        "AMENDMENT03_REPLACEMENT_RUN_AUTHORIZATION=1_OF_1_CONSUMED\n"
        "CLASSIFICATION=NON_SCIENTIFIC_DIAGNOSTIC_ONLY\n"
        "FULL_AUTHORIZED=NO\n"
        "FULL_TRAINING_EXECUTED=NO\n"
        "SCIENTIFIC_RESULTS_CLAIMED=NO\n"
        f"FAILURE_PHASE={phase}\n"
        f"FAILURE_LOG={run_dir / 'logs/smoke_failure.txt'}\n",
    )


def _write_amendment04_structured_failure(
    *,
    run_dir: Path,
    phase: str,
    command_line: str,
    error: Exception,
    active_variant: str | None = None,
    active_variant_index: int | None = None,
) -> None:
    """Persist a fail-closed Amendment 04 one-shot failure."""

    traceback_text = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    fields = [
        f"PHASE={phase}",
        f"ACTIVE_VARIANT={active_variant or 'NONE'}",
        f"ACTIVE_VARIANT_INDEX={active_variant_index or 'NONE'}",
        f"CANONICAL_INVOCATION={command_line}",
        f"EXCEPTION_TYPE={type(error).__name__}",
        f"EXCEPTION_REPR={error!r}",
        f"RUN_PATH={run_dir}",
        f"LOG_PATH={run_dir / 'logs/smoke_failure.txt'}",
        "TRACEBACK_BEGIN",
        traceback_text.rstrip("\n"),
        "TRACEBACK_END",
    ]
    atomic_write_text(run_dir / "logs/smoke_failure.txt", "\n".join(fields) + "\n")
    atomic_write_json(run_dir / "config/run_identity.lock.json", {
        "run_id": run_dir.name,
        "protocol": "AMENDMENT_02",
        "run_kind": "CUDA_SMOKE_NON_SCIENTIFIC",
        "state": "SMOKE_INCOMPLETE",
        "recovery_authorization": _AMENDMENT04_AUTHORIZATION,
        "amendment03_authorization": "1_OF_1_CONSUMED",
        "replacement_run_authorization": "1_OF_1_CONSUMED",
        "permitted_transitions": ["PACKAGE_REVIEW_EVIDENCE"],
        "full_authorized": False,
        "stability_authorized": False,
        "full_training_executed": False,
        "scientific_results_claimed": False,
    })
    atomic_write_text(
        run_dir / "RUN_STATUS.txt",
        "RUN_KIND=CUDA_SMOKE_NON_SCIENTIFIC\n"
        "RUN_STATE=SMOKE_INCOMPLETE\n"
        "PROTOCOL_AMENDMENT=AMENDMENT_02\n"
        f"RECOVERY_AUTHORIZATION={_AMENDMENT04_AUTHORIZATION}\n"
        "AMENDMENT03_AUTHORIZATION=1_OF_1_CONSUMED\n"
        "AMENDMENT04_FRESH_RUN_AUTHORIZATION=1_OF_1_CONSUMED\n"
        "CLASSIFICATION=NON_SCIENTIFIC_DIAGNOSTIC_ONLY\n"
        "FULL_AUTHORIZED=NO\n"
        "FULL_TRAINING_EXECUTED=NO\n"
        "SCIENTIFIC_RESULTS_CLAIMED=NO\n"
        f"FAILURE_PHASE={phase}\n"
        f"FAILURE_LOG={run_dir / 'logs/smoke_failure.txt'}\n",
    )


def smoke_model_package_eligible(
    *,
    completed: Sequence[str],
    variants: Sequence[str],
    runnable_official: Sequence[str],
    state: str,
    amendment03_replacement: bool,
    amendment04_csv_roundtrip: bool = False,
) -> bool:
    """Enforce the Corrigendum's all-or-nothing Models ZIP publication rule."""

    if not completed:
        return False
    if not (amendment03_replacement or amendment04_csv_roundtrip):
        return True
    completed_ids = set(completed)
    return bool(
        state == "SMOKE_COMPLETE"
        and completed_ids == set(variants)
        and len(completed_ids & set(runnable_official)) == 9
        and len(completed_ids & {EXPLORATORY_VARIANT}) == 1
    )


def _run_cuda_smoke_impl(
    *,
    preflight_run: Path,
    command_line: str = "",
    amendment03_replacement: bool = False,
    amendment04_csv_roundtrip: bool = False,
    created_run_holder: MutableMapping[str, Path] | None = None,
) -> Path:
    """Run every permitted variant on one locked, development-only CUDA subset."""
    if amendment03_replacement and amendment04_csv_roundtrip:
        raise ScientificBlocker("AMENDMENT_RECOVERY_FLAGS_ARE_MUTUALLY_EXCLUSIVE")
    compatibility_recovery = amendment03_replacement or amendment04_csv_roundtrip
    print("PHASE_START=CUDA_SMOKE_AUTHORIZATION", flush=True)
    immutable_gate: dict[str, Any] | None = None
    amendment03_contract: amendment03_compat.SealedProbeContract | None = None
    amendment03_allowed_after_hashes: dict[str, str] | None = None
    amendment03_prerun: dict[str, Any] | None = None
    amendment04_allowed_after_hashes: dict[str, str] | None = None
    amendment04_reference_allowed_after_hashes: dict[str, str] | None = None
    amendment04_prerun: dict[str, Any] | None = None
    if amendment03_replacement:
        immutable_gate = amendment03_recovery.validate_corrected_immutable_gate(
            study_root=STUDY_ROOT,
        )
        amendment03_recovery.assert_replacement_guard(
            results_root=STUDY_ROOT / "results",
        )
        amendment03_allowed_after_hashes = amendment03_code_drift_allowlist(
            immutable_gate.get("reviewed_code_tree", {})
        )
    elif amendment04_csv_roundtrip:
        immutable_gate = amendment04_recovery.validate_immutable_gate(
            study_root=STUDY_ROOT,
        )
        amendment04_recovery.assert_run_guard(
            results_root=STUDY_ROOT / "results",
        )
        amendment04_allowed_after_hashes = amendment04_recovery.code_drift_allowlist(
            study_root=STUDY_ROOT,
        )
        amendment04_reference_allowed_after_hashes = (
            amendment04_recovery.reference_code_drift_allowlist(
                (immutable_gate.get("accepted_preflight", {}) or {}).get(
                    "reviewed_code_tree", {}
                ),
                study_root=STUDY_ROOT,
            )
        )
    identity = validate_amended_run_reference(
        preflight_run,
        expected_kind="AMENDED_PREFLIGHT",
        allowed_states={"PREFLIGHT_COMPLETE"},
        amendment03_allowed_after_hashes=amendment03_allowed_after_hashes,
        amendment04_reference_allowed_after_hashes=(
            amendment04_reference_allowed_after_hashes
        ),
    )
    if amendment03_replacement:
        _assert_amendment03_invocation(
            preflight_run=preflight_run,
            command_line=command_line,
            identity=identity,
            immutable_gate=immutable_gate or {},
        )
        amendment03_contract = amendment03_compat.validate_sealed_preflight_probe(
            preflight_run,
            expected_bundle_sha256=(
                amendment03_recovery.ACCEPTED_PREFLIGHT_ZIP_SHA256
            ),
        )
        recovered_probe = (immutable_gate or {}).get("sealed_probe", {})
        compat_manifest_rows = amendment03_contract.evidence.get(
            "manifest_rows", {}
        )
        if not (
            amendment03_contract.evidence.get("member_sha256")
            == recovered_probe.get("member_sha256")
            == "629ce45ce6c23bbc46feac26cbec3ec8cfa7f9e333c3ef595f74065e4c6820a9"
            and amendment03_contract.evidence.get("member_size_bytes")
            == recovered_probe.get("member_size_bytes") == 16941
            and compat_manifest_rows.get("OUTPUT_MANIFEST_FINAL.tsv")
            == recovered_probe.get("output_manifest_row")
            and compat_manifest_rows.get("BUNDLE_MANIFEST.tsv")
            == recovered_probe.get("bundle_manifest_row")
        ):
            raise IntegrityError(
                "Amendment 03 independent sealed-probe validators disagree"
            )
        amendment03_prerun = _validate_amendment03_prerun_evidence()
    elif amendment04_csv_roundtrip:
        _assert_amendment04_invocation(
            preflight_run=preflight_run,
            command_line=command_line,
            identity=identity,
            immutable_gate=immutable_gate or {},
        )
        amendment03_contract = amendment03_compat.validate_sealed_preflight_probe(
            preflight_run,
            expected_bundle_sha256=(
                amendment04_recovery.ACCEPTED_PREFLIGHT_ZIP_SHA256
            ),
        )
        recovered_probe = (
            (immutable_gate or {}).get("accepted_preflight", {}) or {}
        ).get("sealed_probe", {})
        compat_manifest_rows = amendment03_contract.evidence.get(
            "manifest_rows", {}
        )
        if not (
            amendment03_contract.evidence.get("member_sha256")
            == recovered_probe.get("sha256")
            == amendment04_recovery.SEALED_PROBE_SHA256
            and amendment03_contract.evidence.get("member_size_bytes")
            == recovered_probe.get("size_bytes")
            == amendment04_recovery.SEALED_PROBE_SIZE_BYTES
            and compat_manifest_rows.get("OUTPUT_MANIFEST_FINAL.tsv")
            == recovered_probe.get("output_manifest_row")
            and compat_manifest_rows.get("BUNDLE_MANIFEST.tsv")
            == recovered_probe.get("bundle_manifest_row")
        ):
            raise IntegrityError(
                "Amendment 04 independent sealed-probe validators disagree"
            )
        amendment04_prerun = amendment04_recovery.validate_prerun_evidence(
            study_root=STUDY_ROOT,
        )
    reviewed_files, reviewed_member_paths = read_reviewed_preflight_members(
        preflight_run,
        (
            "provenance/PREFLIGHT_REPORT.json",
            "config/variant_feature_sets.json",
            "config/training_configuration.lock.json",
            "provenance/evidence_gate_registry.json",
            "provenance/execution_ledger.json",
            "provenance/source_input_hashes_post.tsv",
            "provenance/cuda_active_fit_probe.json",
            "splits/row_split_manifest.csv.gz",
        ),
        expected_bundle_sha256=identity["reference_validation"][
            "review_bundle_sha256"
        ],
    )
    report = json.loads(reviewed_files["provenance/PREFLIGHT_REPORT.json"])
    features_by_variant = json.loads(
        reviewed_files["config/variant_feature_sets.json"]
    )
    training_configuration = json.loads(
        reviewed_files["config/training_configuration.lock.json"]
    )
    model_parameter_lock = training_configuration.get("model", {})
    if model_parameter_lock != r92_model_parameter_lock():
        raise ScientificBlocker(
            "CUDA_SMOKE_BLOCKED_R92_MODEL_PARAMETER_LOCK_MISMATCH"
        )
    smoke_model_parameters = dict(
        model_parameter_lock["smoke_classifier_parameters"]
    )
    model_parameter_lock_sha256 = model_parameter_lock[
        "smoke_classifier_parameters_sha256"
    ]
    registry_payload = json.loads(
        reviewed_files["provenance/evidence_gate_registry.json"]
    )
    execution_ledger = json.loads(
        reviewed_files["provenance/execution_ledger.json"]
    )
    cuda_active_probe = json.loads(
        reviewed_files["provenance/cuda_active_fit_probe.json"]
    )
    authorize_cuda_smoke_reference(
        preflight_run=preflight_run,
        validated_identity=identity,
        preflight_report=report,
        features_by_variant=features_by_variant,
        registry_payload=registry_payload,
        execution_ledger=execution_ledger,
        reviewed_member_paths=reviewed_member_paths,
        amendment03_allowed_after_hashes=amendment03_allowed_after_hashes,
        amendment04_reference_allowed_after_hashes=(
            amendment04_reference_allowed_after_hashes
        ),
    )
    blockers = report.get("global_blockers", {})
    assert_scientific_training_unblocked(blockers)
    if report.get("cuda_capability", {}).get("status") != "PASS":
        raise ScientificBlocker("CUDA_SMOKE_BLOCKED_XGBOOST_CUDA_UNAVAILABLE")
    assert_amendment02_cuda_runtime(
        identity=identity,
        report=report,
        probe=cuda_active_probe,
    )
    if str(identity.get("protocol")) == "AMENDMENT_02" and not compatibility_recovery:
        existing_smoke = existing_cuda_smoke_outputs()
        if existing_smoke:
            raise ScientificBlocker(
                "AMENDMENT_02_REFUSES_SECOND_CUDA_SMOKE_OUTPUT: "
                + repr(existing_smoke)
            )
    split = pd.read_csv(
        io.BytesIO(reviewed_files["splits/row_split_manifest.csv.gz"]),
        compression="gzip",
    )
    split_hash = split_assignment_hash(
        split.stable_candidate_id, split.split, split.label, split.eligible,
        split.split_order if "split_order" in split else None,
    )
    if split_hash != identity.get("candidate_split_hash"):
        raise IntegrityError("Smoke split differs from accepted Preflight lock")
    reviewed_inventory = pd.read_csv(
        io.BytesIO(reviewed_files["provenance/source_input_hashes_post.tsv"]),
        sep="\t",
    )
    training_path = Path(str(report["training_table_path"]))
    training_sha256 = str(report["training_table_sha256"])
    inventory_match = reviewed_inventory[
        reviewed_inventory.path.astype(str) == str(training_path)
    ]
    if (
        len(inventory_match) != 1
        or str(inventory_match.iloc[0].sha256) != training_sha256
        or not _valid_sha256(training_sha256)
    ):
        raise IntegrityError(
            "Smoke training table is not bound to the reviewed source inventory"
        )
    reviewed_source_hashes = identity.get("source_hashes", {})
    for source_path in (R91_PREDICTIONS, R91_THRESHOLD):
        matches = reviewed_inventory[
            reviewed_inventory.path.astype(str) == str(source_path)
        ]
        expected_source_sha = reviewed_source_hashes.get(str(source_path))
        if not (
            len(matches) == 1
            and _valid_sha256(expected_source_sha)
            and str(matches.iloc[0].sha256) == expected_source_sha
            and source_path.is_file()
            and not source_path.is_symlink()
            and sha256_file(source_path) == expected_source_sha
        ):
            raise IntegrityError(
                f"Smoke review-weight source is not hash-pinned: {source_path}"
            )
    training = load_training_verified(training_path, training_sha256)
    assert_training_matches_reviewed_split(training, split)
    train_indices, validation_indices, subset_info = deterministic_smoke_subset(split)
    runnable_official = runnable_official_variants(features_by_variant)
    variants = [*runnable_official, EXPLORATORY_VARIANT]
    validation = training.iloc[validation_indices]
    training_subset = training.iloc[train_indices]
    no_pca_training_raw = None
    no_pca_validation_raw = None
    no_pca_layer_evidence = None
    no_pca_contract = None
    no_pca_raw_features: list[str] = []
    if "no_pca" in runnable_official:
        no_pca_spec = features_by_variant["no_pca"]
        no_pca_contract = no_pca_spec.get("raw_layer_contract")
        no_pca_raw_features = list(
            (no_pca_contract or {}).get("ordered_raw_feature_columns", [])
        )
    base_weights, weight_info = review_weights(
        training_subset, historical_geometry=False,
    )
    protocol = str(identity.get("protocol", "AMENDMENT_01"))
    transaction_ownership: dict[Path, PublicationOwnershipToken] = {}
    run_name: str | None = None
    run_dir: Path | None = None
    amendment03_recovery_validation: dict[str, Any] | None = None
    amendment04_recovery_validation: dict[str, Any] | None = None
    if amendment03_replacement:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_name = (
            f"run_{timestamp}_"
            f"{sha256_bytes(canonical_json({'preflight': identity['run_id'], 'split': split_hash}))[:8]}_"
            "amendment03_cuda_smoke"
        )
        run_dir = STUDY_ROOT / "results" / run_name
        amendment03_recovery.create_authorized_replacement_run_directory(
            run_dir,
            results_root=STUDY_ROOT / "results",
        )
        try:
            for relative in ("config", "provenance", "tables", "metrics", "logs"):
                mkdir(run_dir / relative)
            atomic_write_text(
                run_dir / "NON_SCIENTIFIC_DIAGNOSTIC_ONLY.txt",
                _NON_SCIENTIFIC_SMOKE_MARKER,
            )
            assert amendment03_prerun is not None
            for destination_relative, data in amendment03_prerun["source_bytes"].items():
                atomic_write_bytes(run_dir / destination_relative, data)
            snapshot_smoke_review_evidence(
                preflight_run=preflight_run,
                run_dir=run_dir,
                validated_identity=identity,
                command_line=command_line,
                split_hash=split_hash,
                amendment03_contract=amendment03_contract,
                amendment03_allowed_after_hashes=amendment03_allowed_after_hashes,
                amendment03_prerun_hashes=amendment03_prerun[
                    "destination_sha256"
                ],
            )
            atomic_write_json(run_dir / "config/smoke_subset.json", subset_info)
            executed_code = amendment03_recovery.build_executed_code_provenance(
                run_dir=run_dir,
                reviewed_code_tree=identity.get("code_tree", {}),
                allowed_after_hashes=amendment03_allowed_after_hashes or {},
                study_root=STUDY_ROOT,
            )
            amendment03_recovery.build_recovery_lineage(
                run_dir=run_dir,
                new_run_id=run_name,
                immutable_gate=immutable_gate or {},
                executed_code=executed_code,
            )
            amendment03_recovery_validation = (
                amendment03_recovery.validate_recovery_artifacts(
                    run_dir=run_dir,
                    reviewed_code_tree=identity.get("code_tree", {}),
                    allowed_after_hashes=amendment03_allowed_after_hashes or {},
                    study_root=STUDY_ROOT,
                    require_prompt_artifacts=False,
                )
            )
        except Exception as exc:
            _write_amendment03_structured_failure(
                run_dir=run_dir,
                phase="RUN_INITIALIZATION_BEFORE_RESAMPLING",
                command_line=command_line,
                error=exc,
            )
            raise
    elif amendment04_csv_roundtrip:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_name = (
            f"run_{timestamp}_"
            f"{sha256_bytes(canonical_json({'preflight': identity['run_id'], 'split': split_hash}))[:8]}_"
            "amendment04_cuda_smoke"
        )
        run_dir = STUDY_ROOT / "results" / run_name
        amendment04_recovery.create_authorized_run_directory(
            run_dir,
            results_root=STUDY_ROOT / "results",
            created_run_holder=created_run_holder,
        )
        try:
            for relative in ("config", "provenance", "tables", "metrics", "logs"):
                mkdir(run_dir / relative)
            atomic_write_text(
                run_dir / "NON_SCIENTIFIC_DIAGNOSTIC_ONLY.txt",
                _NON_SCIENTIFIC_SMOKE_MARKER,
            )
            assert amendment04_prerun is not None
            for destination_relative, data in amendment04_prerun[
                "source_bytes"
            ].items():
                atomic_write_bytes(run_dir / destination_relative, data)
            snapshot_smoke_review_evidence(
                preflight_run=preflight_run,
                run_dir=run_dir,
                validated_identity=identity,
                command_line=command_line,
                split_hash=split_hash,
                amendment03_contract=amendment03_contract,
                recovery_authorization=_AMENDMENT04_AUTHORIZATION,
                recovery_allowed_after_hashes=amendment04_allowed_after_hashes,
                recovery_prerun_hashes=amendment04_prerun[
                    "destination_sha256"
                ],
            )
            atomic_write_json(run_dir / "config/smoke_subset.json", subset_info)
            executed_code = amendment04_recovery.build_executed_provenance(
                run_dir=run_dir,
                allowed_after_hashes=amendment04_allowed_after_hashes or {},
                study_root=STUDY_ROOT,
            )
            amendment04_recovery.build_lineage(
                run_dir=run_dir,
                new_run_id=run_name,
                immutable_gate=immutable_gate or {},
                csv_diagnostic=amendment04_prerun[
                    "csv_roundtrip_diagnostic"
                ],
                executed_provenance=executed_code,
            )
            amendment04_recovery_validation = (
                amendment04_recovery.validate_recovery_artifacts(
                    run_dir=run_dir,
                    allowed_after_hashes=amendment04_allowed_after_hashes or {},
                    study_root=STUDY_ROOT,
                    immutable_gate=immutable_gate,
                    require_prompt_artifacts=False,
                )
            )
        except Exception as exc:
            _write_amendment04_structured_failure(
                run_dir=run_dir,
                phase="RUN_INITIALIZATION_BEFORE_RESAMPLING",
                command_line=command_line,
                error=exc,
            )
            raise
    paired_frozen = None
    paired_projections: dict[str, paired_resampling.VariantProjection] = {}
    paired_validation = None
    paired_parity: dict[str, Any] = {}
    if protocol == "AMENDMENT_02":
        try:
            print(
                "PHASE_START=FROZEN_FULL_SPACE_RESAMPLING "
                f"TRAIN_ROWS={len(training_subset)} FEATURES=93",
                flush=True,
            )
            full_features = get_r92_feature_order()
            if features_by_variant != build_variant_feature_sets(full_features):
                raise IntegrityError(
                    "Reviewed Amendment 02 Variant definitions differ from executing code"
                )
            paired_frozen = paired_resampling.build_frozen_resampling_population(
                training_subset.loc[:, full_features],
                training_subset.label.to_numpy(np.int8),
                base_weights,
                training_subset.stable_candidate_id.astype(str),
                full_features=full_features,
                seed=BASE_SEED,
                backend=sys.modules[__name__],
            )
            paired_projections = paired_resampling.project_all_variants(
                paired_frozen, backend=sys.modules[__name__],
            )
            paired_validation = paired_resampling.transform_validation_full_space(
                paired_frozen, validation.loc[:, full_features],
            )
            paired_parity = paired_resampling.frozen_resampling_parity(
                paired_frozen, paired_projections, backend=sys.modules[__name__],
            )
            if paired_parity.get("status") != "PASS":
                raise ScientificBlocker(
                    "AMENDMENT_02_FROZEN_RESAMPLING_PARITY_FAILED"
                )
            print(
                "PHASE_COMPLETE=FROZEN_FULL_SPACE_RESAMPLING "
                f"POST_SMOTE_ROWS={len(paired_frozen.post_smote_y)} STATUS=PASS",
                flush=True,
            )
        except Exception as exc:
            if compatibility_recovery and run_dir is not None:
                if amendment04_csv_roundtrip:
                    _write_amendment04_structured_failure(
                        run_dir=run_dir,
                        phase="FROZEN_FULL_SPACE_RESAMPLING",
                        command_line=command_line,
                        error=exc,
                    )
                else:
                    _write_amendment03_structured_failure(
                        run_dir=run_dir,
                        phase="FROZEN_FULL_SPACE_RESAMPLING",
                        command_line=command_line,
                        error=exc,
                    )
            raise
    print("PHASE_COMPLETE=CUDA_SMOKE_AUTHORIZATION STATUS=PASS", flush=True)
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_name = f"run_{timestamp}_{sha256_bytes(canonical_json({'preflight': identity['run_id'], 'split': split_hash}))[:8]}_cuda_smoke"
        run_dir = STUDY_ROOT / "results" / run_name
        create_immutable_run_directory(run_dir)
        for relative in ("config", "provenance", "tables", "metrics", "logs"):
            mkdir(run_dir / relative)
        atomic_write_text(
            run_dir / "NON_SCIENTIFIC_DIAGNOSTIC_ONLY.txt",
            _NON_SCIENTIFIC_SMOKE_MARKER,
        )
        snapshot_smoke_review_evidence(
            preflight_run=preflight_run,
            run_dir=run_dir,
            validated_identity=identity,
            command_line=command_line,
            split_hash=split_hash,
        )
        atomic_write_json(run_dir / "config/smoke_subset.json", subset_info)
    assert run_name is not None
    try:
        if protocol == "AMENDMENT_02":
            atomic_write_json(
                run_dir / "provenance/frozen_resampling_parity.json",
                paired_parity,
            )
        model_root = STUDY_ROOT / ".runtime" / "smoke_models" / run_name
        mkdir(model_root.parent)
        create_immutable_run_directory(model_root)
        smoke_gpu_identity = gpu_identity_metadata()
    except Exception as exc:
        if compatibility_recovery:
            if amendment04_csv_roundtrip:
                _write_amendment04_structured_failure(
                    run_dir=run_dir,
                    phase="POST_AUTHORIZATION_PRE_VARIANT_SETUP",
                    command_line=command_line,
                    error=exc,
                )
            else:
                _write_amendment03_structured_failure(
                    run_dir=run_dir,
                    phase="POST_AUTHORIZATION_PRE_VARIANT_SETUP",
                    command_line=command_line,
                    error=exc,
                )
        raise
    rows = []
    details = {}
    parity = {}
    completed = []
    state = "SMOKE_COMPLETE"
    active_variant_monitor: legacy.PeakMonitor | None = None
    active_variant: str | None = None
    print("PHASE_START=CUDA_SMOKE_VARIANTS", flush=True)
    try:
        for variant in variants:
            active_variant = variant
            print(
                f"SMOKE_VARIANT_START={variant} "
                f"INDEX={variants.index(variant) + 1}/{len(variants)}",
                flush=True,
            )
            variant_started_wall = time.perf_counter()
            variant_started_cpu = time.process_time()
            baseline_gpu = _gpu_used_memory_mib()
            baseline_gpu = 0.0 if baseline_gpu is None else baseline_gpu
            active_variant_monitor = legacy.PeakMonitor()
            active_variant_monitor.__enter__()
            feature_names = features_by_variant[variant]["retained_features"]
            raw_features = (
                list(features_by_variant["no_pca"]["raw_layer_contract"]["ordered_raw_feature_columns"])
                if variant == "no_pca" else []
            )
            table_features = [name for name in feature_names if name not in set(raw_features)]
            missing_features = [name for name in table_features if name not in training]
            if missing_features:
                raise ScientificBlocker(
                    f"Smoke {variant} is marked runnable but its reviewed training "
                    f"layer is missing {len(missing_features)} predictors"
                )
            weights = np.ones(len(training_subset), np.float32) if variant == "no_review_aware_training_weights" else base_weights.copy()
            smote_enabled = variant != "no_safe_smote"
            if protocol == "AMENDMENT_02":
                if variant == "no_pca":
                    raise ScientificBlocker(
                        "TRUE_NO_PCA_MUST_NOT_RUN_UNDER_AMENDMENT_02"
                    )
                projection = paired_projections[variant]
                X = projection.X
                y = projection.y
                fit_weights = projection.weights
                lineage = projection.lineage
                stages = [dict(stage) for stage in paired_frozen.stages]
                if variant == "no_safe_smote":
                    no_smote_final = dict(projection.evidence)
                    no_smote_final.update({
                        "stage": "post_safe_smote",
                        "features": 93,
                        "safe_smote_enabled": False,
                        "safe_smote_synthetic_rows": 0,
                        "generation_scope": (
                            "FROZEN_PRE_SMOTE_POPULATION_NO_NEW_GENERATION"
                        ),
                    })
                    stages[-1] = no_smote_final
                validation_matrix = (
                    paired_resampling.project_validation_population(
                        paired_validation,
                        projection,
                        backend=sys.modules[__name__],
                    )
                )
            elif variant == "no_pca":
                (
                    no_pca_training_raw,
                    no_pca_validation_raw,
                    no_pca_layer_evidence,
                ) = load_no_pca_smoke_raw_layers(
                    no_pca_contract,
                    training,
                    train_indices,
                    validation_indices,
                    no_pca_raw_features,
                )
                if no_pca_training_raw is None or no_pca_validation_raw is None:
                    raise ScientificBlocker("no_pca raw layers were not validated and loaded")
                training_input = pd.concat([
                    training_subset[table_features].reset_index(drop=True),
                    pd.DataFrame(no_pca_training_raw, columns=raw_features),
                ], axis=1)[feature_names]
                validation_input = pd.concat([
                    validation[table_features].reset_index(drop=True),
                    pd.DataFrame(no_pca_validation_raw, columns=raw_features),
                ], axis=1)[feature_names]
            else:
                training_input = training_subset[feature_names]
                validation_input = validation[feature_names]
            if protocol != "AMENDMENT_02":
                X, y, fit_weights, lineage, imputer, stages = augment_with_lineage(
                    training_input, training_subset.label.to_numpy(), weights,
                    training_subset.stable_candidate_id.astype(str), seed=BASE_SEED,
                    smote_enabled=smote_enabled,
                )
                validation_matrix = np.asarray(
                    imputer.transform(validation_input), dtype=np.float32,
                )
            validation_labels = validation.label.to_numpy(np.int8)
            classifier = xgb.XGBClassifier(**smoke_model_parameters)
            fit_started_wall = time.perf_counter()
            fit_started_cpu = time.process_time()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                classifier.fit(
                    X, y, sample_weight=fit_weights,
                    eval_set=[(validation_matrix, validation_labels)], verbose=False,
                )
            fit_wall = time.perf_counter() - fit_started_wall
            fit_cpu = time.process_time() - fit_started_cpu
            warning_text = [str(item.message) for item in caught]
            configuration = json.loads(classifier.get_booster().save_config())
            serialized_configuration = json.dumps(configuration).lower()
            fallback = any("fallback" in value.lower() or "not compiled with gpu" in value.lower() for value in warning_text)
            observed_gpu_after_fit = _gpu_used_memory_mib()
            if (
                "cuda" not in serialized_configuration
                or fallback
                or (
                    active_variant_monitor.peak_gpu_memory_mib is None
                    and observed_gpu_after_fit is None
                )
            ):
                raise ScientificBlocker(f"Smoke {variant} did not prove real CUDA execution")
            model_path = model_root / f"{variant}.ubj"
            embedded_model_attributes = {
                "amendment_classification": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
                "amendment_device": "cuda",
                "amendment_variant": variant,
                "amendment_run_id": run_name,
                "booster_configuration_sha256": sha256_bytes(
                    canonical_json(configuration)
                ),
                "feature_list_sha256": features_by_variant[variant][
                    "feature_list_sha256"
                ],
                "gpu_identity_sha256": sha256_bytes(
                    canonical_json(smoke_gpu_identity)
                ),
                "model_parameter_lock_sha256": model_parameter_lock_sha256,
            }
            if compatibility_recovery:
                embedded_model_attributes.update(
                    amendment03_compat.embed_binary_classifier_metadata(
                        classifier,
                        expected_feature_count=len(feature_names),
                        feature_list_sha256=features_by_variant[variant][
                            "feature_list_sha256"
                        ],
                    )
                )
            classifier.get_booster().set_attr(**embedded_model_attributes)
            classifier.save_model(model_path)
            prediction_started = time.perf_counter()
            amendment03_parity = None
            if compatibility_recovery:
                amendment03_parity = amendment03_compat.four_way_binary_reload_parity(
                    classifier,
                    model_path,
                    validation_matrix,
                    expected_feature_count=len(feature_names),
                    expected_feature_list_sha256=features_by_variant[variant][
                        "feature_list_sha256"
                    ],
                    device="cuda",
                )
                prediction = amendment03_parity.fitted_classifier_positive
                reloaded_booster = amendment03_parity.reloaded_booster
                parity_warnings = amendment03_parity.evidence["warnings"]
                prediction_warnings = list(parity_warnings["fitted_classifier"])
                reload_warnings = [
                    *parity_warnings["fitted_booster"],
                    *parity_warnings["reloaded_booster"],
                    *parity_warnings["compatibility_classifier"],
                ]
                reload_difference = max(
                    amendment03_parity.evidence[
                        "maximum_absolute_difference"
                    ].values()
                )
            else:
                prediction, prediction_warnings = _cuda_classifier_predict_proba(
                    classifier, validation_matrix,
                )
                reloaded = xgb.XGBClassifier()
                reloaded.load_model(model_path)
                reloaded.set_params(device="cuda")
                reloaded.get_booster().set_param({"device": "cuda"})
                reloaded_prediction, reload_warnings = _cuda_classifier_predict_proba(
                    reloaded, validation_matrix,
                )
                reloaded_booster = reloaded.get_booster()
                reload_difference = float(
                    np.max(np.abs(prediction - reloaded_prediction))
                )
            prediction_wall = time.perf_counter() - prediction_started
            if reload_difference != 0.0:
                raise ScientificBlocker(
                    f"Smoke {variant} save/reload predictions differ"
                )
            binary_prediction = prediction >= 0.5
            metrics = {
                "average_precision": float(average_precision_score(validation_labels, prediction)),
                "roc_auc": float(roc_auc_score(validation_labels, prediction)),
                "precision": float(precision_score(
                    validation_labels, binary_prediction, zero_division=0,
                )),
                "recall": float(recall_score(
                    validation_labels, binary_prediction, zero_division=0,
                )),
                "f1": float(f1_score(
                    validation_labels, binary_prediction, zero_division=0,
                )),
            }
            active_cuda_probe = _active_reloaded_booster_cuda_probe(
                reloaded_booster, len(feature_names),
            )
            if active_cuda_probe.get("status") != "PASS":
                raise ScientificBlocker(
                    f"Smoke {variant} reloaded model did not execute on CUDA"
                )
            observed_gpu_after_reload = _gpu_used_memory_mib()
            active_variant_monitor.__exit__(None, None, None)
            monitor = active_variant_monitor
            active_variant_monitor = None
            wall = time.perf_counter() - variant_started_wall
            cpu = time.process_time() - variant_started_cpu
            peak_gpu_candidates = [
                value for value in (
                    monitor.peak_gpu_memory_mib,
                    observed_gpu_after_fit,
                    observed_gpu_after_reload,
                ) if value is not None
            ]
            peak_gpu = max(peak_gpu_candidates)
            peak_gpu_delta = max(0.0, peak_gpu - baseline_gpu)
            variant_detail = {
                "classification": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
                "variant_classification": (
                    "EXPLORATORY" if variant == EXPLORATORY_VARIANT else "OFFICIAL"
                ),
                "feature_count": len(feature_names), "feature_list_sha256": features_by_variant[variant]["feature_list_sha256"],
                "stages": stages, "metrics_development_only": metrics,
                "booster_configuration": configuration,
                "warnings": [
                    *warning_text, *prediction_warnings, *reload_warnings,
                ],
                "requested_model_parameters": smoke_model_parameters,
                "model_parameter_lock_sha256": model_parameter_lock_sha256,
                "resource_measurement_scope": "END_TO_END_VARIANT",
                "gpu_memory_measurement_scope": (
                    monitor.gpu_memory_measurement_scope
                    or "PROCESS_OR_GLOBAL_DEVICE_WSL_SNAPSHOT"
                ),
                "wall_seconds": wall, "cpu_seconds": cpu,
                "fit_wall_seconds": fit_wall, "fit_cpu_seconds": fit_cpu,
                "prediction_wall_seconds": prediction_wall,
                "training_rows": int(len(y)),
                "training_raw_rows": int(len(training_subset)),
                "validation_rows": int(len(validation_labels)),
                "requested_device": "cuda",
                "observed_device": str(
                    configuration.get("learner", {})
                    .get("generic_param", {}).get("device", "")
                ),
                "best_iteration": (
                    int(classifier.best_iteration)
                    if hasattr(classifier, "best_iteration") else None
                ),
                "tree_count": int(classifier.get_booster().num_boosted_rounds()),
                "safe_smote_synthetic_rows": int(
                    projection.evidence["safe_smote_synthetic_rows"]
                    if protocol == "AMENDMENT_02"
                    else stages[-1]["safe_smote_synthetic_rows"]
                ),
                "peak_ram_bytes": monitor.peak_rss_bytes,
                "peak_gpu_memory_mib": peak_gpu,
                "baseline_process_gpu_memory_mib": baseline_gpu,
                "peak_gpu_memory_delta_mib": peak_gpu_delta,
                "gpu_identity_and_visibility": smoke_gpu_identity,
                "active_reloaded_cuda_probe": active_cuda_probe,
                "embedded_model_attributes": embedded_model_attributes,
                "model_sha256": sha256_file(model_path),
                "save_reload_max_probability_difference": reload_difference,
                "external_audit_scored": False,
            }
            if compatibility_recovery:
                parity_evidence = amendment03_parity.evidence
                parity_maximum = parity_evidence[
                    "maximum_absolute_difference"
                ]
                parity_bit_exact = parity_evidence[
                    "bit_exact_against_fitted_classifier"
                ]
                variant_detail.update({
                    "amendment03_model_recovery": {
                        "classification": "VERIFIED",
                        "status": "PASS",
                        "authorization": "AMENDMENT_03_CORRIGENDUM_ONE",
                        "compatibility_loader": (
                            "PROJECT_OWNED_XGB211_BINARY_CLASSIFIER_LOADER"
                        ),
                        "four_way_parity_required": True,
                        "reload_root_cause": (
                            "XGBOOST_2_1_1_SKLEARN_TAG_PROTOCOL_DID_NOT_RESTORE_"
                            "N_CLASSES"
                        ),
                    },
                    "four_way_reload_parity": parity_evidence,
                    "raw_booster_parity": {
                        "status": "PASS",
                        "iteration_range": parity_evidence["iteration_range"],
                        "fitted_booster_bit_exact": parity_bit_exact[
                            "fitted_booster"
                        ],
                        "reloaded_booster_bit_exact": parity_bit_exact[
                            "reloaded_booster"
                        ],
                        "fitted_booster_max_probability_difference": (
                            parity_maximum["fitted_booster"]
                        ),
                        "reloaded_booster_max_probability_difference": (
                            parity_maximum["reloaded_booster"]
                        ),
                        "model_bytes_unchanged": parity_evidence[
                            "model_bytes_unchanged"
                        ],
                    },
                    "compatibility_classifier_parity": {
                        "status": "PASS",
                        "iteration_range": parity_evidence["iteration_range"],
                        "bit_exact": parity_bit_exact[
                            "compatibility_classifier"
                        ],
                        "max_probability_difference": parity_maximum[
                            "compatibility_classifier"
                        ],
                        "repair_applied": parity_evidence[
                            "compatibility_loader"
                        ]["compatibility_repair_applied"],
                    },
                })
            if protocol == "AMENDMENT_02":
                variant_detail["paired_resampling"] = {
                    "status": "PASS",
                    "generation_scope": "ONE_FULL_SPACE_POPULATION",
                    "projection": dict(projection.evidence),
                    "frozen_parity_sha256": sha256_bytes(
                        canonical_json(paired_parity)
                    ),
                }
            if variant == "no_pca":
                variant_detail["raw_layer_load_evidence"] = no_pca_layer_evidence
            semantic_failures = _smoke_variant_detail_failures(
                variant=variant,
                detail=variant_detail,
                specification=features_by_variant[variant],
                booster=reloaded_booster,
                expected_raw_rows=int(subset_info["training_raw_rows"]),
                run_id=run_name,
                expected_model_parameters=smoke_model_parameters,
                expected_parameter_lock_sha256=model_parameter_lock_sha256,
            )
            if semantic_failures:
                raise ScientificBlocker(
                    f"Smoke {variant} semantic completion failed: "
                    + ",".join(semantic_failures)
                )
            details[variant] = variant_detail
            rows.append({
                "variant": variant, "status": "PASS",
                "feature_count": len(feature_names), **metrics,
                "wall_seconds": wall, "cpu_seconds": cpu,
                "prediction_wall_seconds": prediction_wall,
                "training_rows": int(len(y)),
                "validation_rows": int(len(validation_labels)),
                "best_iteration": variant_detail["best_iteration"],
                "tree_count": variant_detail["tree_count"],
                "peak_ram_bytes": monitor.peak_rss_bytes,
                "baseline_process_gpu_memory_mib": baseline_gpu,
                "peak_gpu_memory_mib": peak_gpu,
                "peak_gpu_memory_delta_mib": peak_gpu_delta,
            })
            completed.append(variant)
            print(
                f"SMOKE_VARIANT_COMPLETE={variant} STATUS=PASS "
                f"WALL_SECONDS={wall:.6f}",
                flush=True,
            )
            active_variant = None
        parity = (
            paired_parity if protocol == "AMENDMENT_02"
            else smoke_resampling_parity(details)
        )
        if protocol == "AMENDMENT_02":
            if parity.get("status") != "PASS":
                raise ScientificBlocker("Smoke resampling parity invariant failed")
        elif not all(parity["full_vs_no_weight"].values()) or not all(parity["full_vs_no_safe_smote_pre_smote"].values()):
            raise ScientificBlocker("Smoke resampling parity invariant failed")
    except Exception as exc:
        if active_variant_monitor is not None:
            active_variant_monitor.__exit__(type(exc), exc, exc.__traceback__)
            active_variant_monitor = None
        state = "SMOKE_INCOMPLETE"
        if active_variant is not None:
            print(
                f"SMOKE_VARIANT_COMPLETE={active_variant} STATUS=FAIL "
                f"ERROR_TYPE={type(exc).__name__}",
                flush=True,
            )
        if compatibility_recovery:
            writer = (
                _write_amendment04_structured_failure
                if amendment04_csv_roundtrip
                else _write_amendment03_structured_failure
            )
            writer(
                run_dir=run_dir,
                phase="CUDA_SMOKE_VARIANTS",
                command_line=command_line,
                error=exc,
                active_variant=active_variant,
                active_variant_index=(
                    variants.index(active_variant) + 1
                    if active_variant in variants else None
                ),
            )
        else:
            atomic_write_text(run_dir / "logs/smoke_failure.txt", repr(exc) + "\n")
    print(
        f"PHASE_COMPLETE=CUDA_SMOKE_VARIANTS STATUS="
        f"{'PASS' if state == 'SMOKE_COMPLETE' else 'FAIL'} ",
        f"COMPLETED={len(completed)}/{len(variants)}",
        flush=True,
    )
    parity = (
        paired_parity if protocol == "AMENDMENT_02"
        else smoke_resampling_parity(details)
    )
    model_bundle = None
    model_bundle_failure = None
    model_bundle_adjacent: Path | None = None
    print("PHASE_START=CUDA_SMOKE_MODEL_PACKAGE", flush=True)
    should_package_models = smoke_model_package_eligible(
        completed=completed,
        variants=variants,
        runnable_official=runnable_official,
        state=state,
        amendment03_replacement=amendment03_replacement,
        amendment04_csv_roundtrip=amendment04_csv_roundtrip,
    )
    if should_package_models:
        try:
            model_bundle = _package_smoke_models(
                model_root, run_name, completed_variant_ids=completed,
                ownership=transaction_ownership,
            )
            model_semantics = verify_smoke_model_bundle_semantics(
                model_bundle,
                set(completed),
                details,
                features_by_variant,
                run_dir=run_dir,
                expected_raw_rows=int(subset_info["training_raw_rows"]),
                model_parameter_lock=model_parameter_lock,
            )
            if model_semantics["status"] != "PASS":
                raise IntegrityError(
                    "Smoke model bundle failed pre-package semantic verification: "
                    + ",".join(model_semantics["failures"])
                )
            model_bundle["semantic_verification"] = model_semantics
            model_bundle_adjacent = STUDY_ROOT / "results" / (
                f"{run_name}_NON_SCIENTIFIC_smoke_models_package_verification.json"
            )
            publish_json_no_clobber(
                model_bundle_adjacent, model_bundle,
                ownership=transaction_ownership,
            )
        except Exception as exc:
            if model_bundle is not None:
                remove_owned_publication(
                    Path(model_bundle["path"]), ownership=transaction_ownership,
                )
                model_bundle = None
            state = "SMOKE_INCOMPLETE"
            model_bundle_failure = failed_smoke_model_bundle_record(
                run_name, completed, exc,
            )
            atomic_write_json(
                run_dir / "provenance/smoke_model_bundle_failure.json",
                model_bundle_failure,
            )
            if compatibility_recovery:
                writer = (
                    _write_amendment04_structured_failure
                    if amendment04_csv_roundtrip
                    else _write_amendment03_structured_failure
                )
                writer(
                    run_dir=run_dir,
                    phase="CUDA_SMOKE_MODEL_PACKAGE",
                    command_line=command_line,
                    error=exc,
                )
            else:
                failure_path = run_dir / "logs/smoke_failure.txt"
                previous = failure_path.read_text() if failure_path.is_file() else ""
                atomic_write_text(failure_path, previous + f"model_bundle: {exc!r}\n")
    print(
        "PHASE_COMPLETE=CUDA_SMOKE_MODEL_PACKAGE STATUS="
        + ("PASS" if model_bundle else "FAIL"),
        flush=True,
    )
    if model_bundle:
        atomic_write_json(
            run_dir / "provenance/smoke_model_bundle_verification.json",
            model_bundle,
        )
    model_bundle_reference = model_bundle or model_bundle_failure

    smoke_columns = [
        "variant", "status", "feature_count", "average_precision", "roc_auc",
        "precision", "recall", "f1", "training_rows", "validation_rows",
        "prediction_wall_seconds", "best_iteration", "tree_count",
        "wall_seconds", "cpu_seconds", "peak_ram_bytes",
        "baseline_process_gpu_memory_mib", "peak_gpu_memory_mib",
        "peak_gpu_memory_delta_mib",
    ]
    atomic_write_frame(
        pd.DataFrame(rows, columns=smoke_columns),
        run_dir / "tables/SMOKE_REPORT.csv",
    )
    no_pca_evidence_status = (
        "VERIFIED_EXACT_RAW_EMBEDDINGS_AVAILABLE"
        if features_by_variant["no_pca"]["status"] == "RUNNABLE"
        else features_by_variant["no_pca"]["status"]
    )
    smoke_metrics = {
        "classification": "NON_SCIENTIFIC_DIAGNOSTIC_ONLY", "state": state,
        "official_variants_completed": len(set(completed) & set(runnable_official)),
        "official_variants_expected": len(runnable_official),
        "exploratory_variants_completed": len(set(completed) & {EXPLORATORY_VARIANT}),
        "true_no_pca_status": no_pca_evidence_status,
        "details": details, "parity": parity, "external_audit_scored": False,
        "smoke_models_bundle": model_bundle or model_bundle_failure or {
            "path": "NOT_CREATED_NO_SMOKE_MODELS", "sha256": "NOT_APPLICABLE",
        },
    }
    atomic_write_json(run_dir / "metrics/SMOKE_REPORT.json", smoke_metrics)
    atomic_write_json(run_dir / "provenance/review_weight_subset.json", weight_info)

    execution_ledger = smoke_execution_ledger(
        run_dir, command_line=command_line, runnable_official=runnable_official,
    )
    atomic_write_json(run_dir / "provenance/execution_ledger.json", execution_ledger)
    preflight_registry_payload = registry_payload
    smoke_completed_official = execution_ledger["official_smoke_variants_completed"]
    smoke_completed_exploratory = execution_ledger[
        "exploratory_smoke_variants_completed"
    ]
    smoke_gate_inputs: dict[str, dict[str, Any]] = {}
    for specification in evidence_workflow.DEFAULT_GATE_SPECS:
        previous = preflight_registry_payload["gates"][specification.key]
        smoke_gate_inputs[specification.key] = {
            "classification": previous["classification"],
            "passed": previous["passed"],
            "evidence_refs": ["config/preflight_reference.json"],
            "detail": (
                f"Inherited from validated Preflight {identity['run_id']}; "
                f"source evidence refs={previous['evidence_refs']}. "
                + previous.get("detail", "")
            ),
        }
    smoke_gate_inputs["cuda_smoke_all_runnable"] = {
        "classification": "VERIFIED",
        "passed": state == "SMOKE_COMPLETE"
        and execution_ledger["status"] == "PASS"
        and smoke_completed_official == len(runnable_official)
        and smoke_completed_exploratory == 1,
        "evidence_refs": [
            "tables/SMOKE_REPORT.csv", "metrics/SMOKE_REPORT.json",
            "provenance/execution_ledger.json",
        ],
        "detail": "Derived from validated official/exploratory Smoke variant IDs and statuses.",
    }
    smoke_gate_inputs["model_save_reload"] = {
        "classification": "VERIFIED",
        "passed": state == "SMOKE_COMPLETE"
        and len(completed) == len(variants)
        and model_bundle is not None
        and all(
            value.get("save_reload_max_probability_difference") == 0.0
            for key, value in details.items() if key in completed
        ),
        "evidence_refs": ["metrics/SMOKE_REPORT.json"] + (
            ["provenance/smoke_model_bundle_verification.json"]
            if model_bundle else []
        ),
        "detail": "Every completed Smoke model was reopened with bit-identical probabilities.",
    }
    smoke_gate_inputs["no_full_scientific_training"] = {
        "classification": "VERIFIED",
        "passed": execution_ledger["status"] == "PASS"
        and not execution_ledger["full_scientific_run_executed"]
        and execution_ledger["full_variants_completed"] == 0,
        "evidence_refs": ["provenance/execution_ledger.json"],
        "detail": "The current Smoke invocation and Run artifacts were scanned for Full, resume, stability, and multi-seed execution.",
    }
    smoke_gate_inputs["output_manifest_zip_verification"] = {
        "classification": "BLOCKED", "passed": None,
        "evidence_refs": ["provenance/package_staging_verification.json"],
        "detail": "Pending package verification.",
    }
    smoke_facts = evidence_workflow.WorkflowFacts(
        phase="CUDA_SMOKE",
        official_smoke_variants_completed=smoke_completed_official,
        exploratory_smoke_variants_completed=smoke_completed_exploratory,
        full_scientific_run_executed=execution_ledger[
            "full_scientific_run_executed"
        ],
        full_variants_completed=execution_ledger["full_variants_completed"],
        no_pca_variant_status=no_pca_evidence_status,
        no_pca_variant_runnable=features_by_variant["no_pca"]["status"] == "RUNNABLE",
        protocol_amendment=str(identity.get("protocol", "AMENDMENT_01")),
    )
    smoke_registry = evidence_workflow.build_gate_registry(smoke_gate_inputs)
    smoke_workflow_state = evidence_workflow.derive_workflow_state(
        smoke_registry, smoke_facts,
    )
    if smoke_workflow_state.run_state == "SMOKE_INCOMPLETE":
        failure_path = run_dir / "logs/smoke_failure.txt"
        if not failure_path.is_file():
            atomic_write_text(
                failure_path,
                "Evidence-derived Smoke state is incomplete; inspect the gate registry.\n",
            )
    smoke_identity = {
        "run_id": run_name,
        "protocol": str(identity.get("protocol", "AMENDMENT_01")),
        "run_kind": smoke_workflow_state.run_kind,
        "state": smoke_workflow_state.run_state,
        "preflight_run_id": identity["run_id"],
        "candidate_split_hash": split_hash,
        "permitted_transitions": ["PACKAGE_REVIEW_EVIDENCE"],
        "full_authorized": False,
        "stability_authorized": False,
        **({
            "recovery_authorization": _AMENDMENT03_AUTHORIZATION,
            "replacement_run_authorization": "1_OF_1_CONSUMED",
            "previous_prompt5_pre_edit_stop_acknowledged": True,
            "sealed_preflight_probe_sha256": (
                amendment03_recovery_validation or {}
            ).get("sealed_probe_sha256"),
            "recovery_lineage_sha256": (
                amendment03_recovery_validation or {}
            ).get("recovery_lineage_sha256"),
            "amendment03_allowed_after_hashes": dict(sorted(
                (amendment03_allowed_after_hashes or {}).items()
            )),
            "full_training_executed": False,
            "scientific_results_claimed": False,
        } if amendment03_replacement else {}),
        **({
            "recovery_authorization": _AMENDMENT04_AUTHORIZATION,
            "amendment03_authorization": "1_OF_1_CONSUMED",
            "replacement_run_authorization": "1_OF_1_CONSUMED",
            "reference_amendment03_run": (
                amendment04_recovery.FAILED_AMENDMENT03_RUN_ID
            ),
            "sealed_preflight_probe_sha256": (
                amendment04_recovery_validation or {}
            ).get("sealed_probe_sha256"),
            "amendment04_lineage_sha256": (
                amendment04_recovery_validation or {}
            ).get("lineage_sha256", (
                amendment04_recovery_validation or {}
            ).get("recovery_lineage_sha256")),
            "amendment04_allowed_after_hashes": dict(sorted(
                (amendment04_allowed_after_hashes or {}).items()
            )),
            "amendment04_prerun_evidence_sha256": dict(sorted(
                (amendment04_prerun or {}).get(
                    "destination_sha256", {}
                ).items()
            )),
            "base_seed": BASE_SEED,
            "old_run_resumed": False,
            "old_models_reused": False,
            "full_training_executed": False,
            "scientific_results_claimed": False,
        } if amendment04_csv_roundtrip else {}),
    }
    atomic_write_json(run_dir / "config/run_identity.lock.json", smoke_identity)
    atomic_write_json(
        run_dir / "provenance/evidence_gate_registry.json",
        _smoke_registry_payload(smoke_registry, smoke_workflow_state),
    )
    _write_smoke_blockers(run_dir, smoke_workflow_state)
    atomic_write_text(run_dir / "RUN_STATUS.txt", _smoke_status_text(
        state=smoke_workflow_state,
        run_dir=run_dir,
        preflight_run=preflight_run,
        preflight_report=report,
        split_hash=split_hash,
        model_bundle=model_bundle_reference,
        amendment03_recovery=amendment03_recovery_validation,
        amendment04_recovery_validation=amendment04_recovery_validation,
    ))

    staging_destination: Path | None = None
    print("PHASE_START=CUDA_SMOKE_REVIEW_PACKAGE", flush=True)
    try:
        staging = package_amended_run(
            run_dir, ownership=transaction_ownership,
        )
        if model_bundle:
            staging["models_bundle_path"] = model_bundle["path"]
            staging["models_bundle_sha256"] = model_bundle["sha256"]
        elif model_bundle_failure:
            staging["models_bundle_path"] = model_bundle_failure["path"]
            staging["models_bundle_sha256"] = model_bundle_failure["sha256"]
        staging_path = Path(staging["bundle_path"])
        staging_destination = staging_path.with_name(
            staging_path.stem + "_staging_verified.zip"
        )
        move_owned_publication_no_clobber(
            staging_path, staging_destination,
            ownership=transaction_ownership,
        )
        staging["staging_bundle_original_path_before_rename"] = staging.pop("bundle_path")
        staging["staging_bundle_temporary_path_after_rename"] = str(staging_destination)
        staging["staging_bundle_lifecycle"] = (
            "SUPERSEDED_BY_FINAL_BUNDLE_AND_DELETED_AFTER_FINAL_REOPEN"
        )
        staging["purpose"] = (
            "First-pass complete Smoke package verification used to derive the sealed "
            "workflow state; the final bundle is independently rebuilt and reopened."
        )
        atomic_write_json(
            run_dir / "provenance/package_staging_verification.json", staging,
        )
        smoke_gate_inputs, smoke_registry, smoke_workflow_state = (
            promote_smoke_package_gate(smoke_gate_inputs, smoke_facts, staging)
        )
        smoke_identity["state"] = smoke_workflow_state.run_state
        atomic_write_json(run_dir / "config/run_identity.lock.json", smoke_identity)
        atomic_write_json(
            run_dir / "provenance/evidence_gate_registry.json",
            _smoke_registry_payload(smoke_registry, smoke_workflow_state),
        )
        _write_smoke_blockers(run_dir, smoke_workflow_state)
        smoke_metrics.update({
            "state": smoke_workflow_state.run_state,
            "cuda_smoke_status": smoke_workflow_state.cuda_smoke_status,
            "ready_for_full_awaiting_external_review": (
                smoke_workflow_state.ready_for_full_awaiting_external_review
            ),
            "blockers": list(smoke_workflow_state.blocker_codes),
            "output_manifest_zip_verification": smoke_workflow_state.gate_statuses[
                "output_manifest_zip_verification"
            ],
        })
        atomic_write_json(run_dir / "metrics/SMOKE_REPORT.json", smoke_metrics)
        final_bundle_reference = {
            "bundle_path": str(
                STUDY_ROOT / "results" / f"{run_dir.name}_review_bundle.zip"
            ),
            "bundle_sha256": "SEE_ADJACENT_PACKAGE_VERIFICATION_JSON_AFTER_FINAL_REOPEN",
        }
        atomic_write_text(run_dir / "RUN_STATUS.txt", _smoke_status_text(
            state=smoke_workflow_state,
            run_dir=run_dir,
            preflight_run=preflight_run,
            preflight_report=report,
            split_hash=split_hash,
            model_bundle=model_bundle_reference,
            review_bundle=final_bundle_reference,
            amendment03_recovery=amendment03_recovery_validation,
            amendment04_recovery_validation=amendment04_recovery_validation,
        ))
        final_bundle = package_amended_run(
            run_dir, ownership=transaction_ownership,
        )
        if model_bundle:
            final_bundle["models_bundle_path"] = model_bundle["path"]
            final_bundle["models_bundle_sha256"] = model_bundle["sha256"]
        elif model_bundle_failure:
            final_bundle["models_bundle_path"] = model_bundle_failure["path"]
            final_bundle["models_bundle_sha256"] = model_bundle_failure["sha256"]
        if amendment03_replacement:
            remove_owned_publication_required(
                staging_destination,
                ownership=transaction_ownership,
                purpose="staging bundle",
            )
            staging_destination = None
            post_immutable_gate = (
                amendment03_recovery.validate_corrected_immutable_gate(
                    study_root=STUDY_ROOT,
                )
            )
            post_replacement_guard = amendment03_recovery.assert_replacement_guard(
                results_root=STUDY_ROOT / "results",
                new_run_id=run_name,
            )
            post_recovery = amendment03_recovery.validate_recovery_artifacts(
                run_dir=run_dir,
                reviewed_code_tree=identity.get("code_tree", {}),
                allowed_after_hashes=amendment03_allowed_after_hashes or {},
                study_root=STUDY_ROOT,
                require_prompt_artifacts=True,
            )
            if not (
                canonical_json(post_immutable_gate) == canonical_json(immutable_gate)
                and post_replacement_guard.get("replacement_authorization")
                == "1_OF_1_CONSUMED"
                and post_recovery.get("status") == "PASS"
                and smoke_workflow_state.run_state in {
                    "SMOKE_COMPLETE", "SMOKE_INCOMPLETE",
                }
                and (
                    smoke_workflow_state.run_state != "SMOKE_COMPLETE"
                    or (
                        model_bundle is not None
                        and smoke_completed_official == 9
                        and smoke_completed_exploratory == 1
                    )
                )
            ):
                raise IntegrityError("Amendment 03 final immutable/recovery gate failed")
            final_bundle["amendment03_final_validation"] = {
                "status": "PASS",
                "authorization": _AMENDMENT03_AUTHORIZATION,
                "replacement_authorization": "1_OF_1_CONSUMED",
                "accepted_preflight_zip_sha256_before": (
                    immutable_gate or {}
                ).get("accepted_preflight_zip_sha256"),
                "accepted_preflight_zip_sha256_after": post_immutable_gate.get(
                    "accepted_preflight_zip_sha256"
                ),
                "sealed_probe_sha256": post_recovery.get(
                    "sealed_probe_sha256"
                ),
                "recovery_lineage_sha256": post_recovery.get(
                    "recovery_lineage_sha256"
                ),
                "official_smoke_variants_completed": smoke_completed_official,
                "exploratory_smoke_variants_completed": (
                    smoke_completed_exploratory
                ),
                "models_bundle_created": model_bundle is not None,
                "full_training_executed": False,
                "scientific_results_claimed": False,
            }
        elif amendment04_csv_roundtrip:
            remove_owned_publication_required(
                staging_destination,
                ownership=transaction_ownership,
                purpose="staging bundle",
            )
            staging_destination = None
            post_immutable_gate = amendment04_recovery.validate_immutable_gate(
                study_root=STUDY_ROOT,
            )
            post_replacement_guard = amendment04_recovery.assert_run_guard(
                results_root=STUDY_ROOT / "results",
                new_run_id=run_name,
            )
            post_recovery = amendment04_recovery.validate_recovery_artifacts(
                run_dir=run_dir,
                allowed_after_hashes=amendment04_allowed_after_hashes or {},
                study_root=STUDY_ROOT,
                immutable_gate=post_immutable_gate,
                require_prompt_artifacts=True,
            )
            models_path = Path(str((model_bundle or {}).get("path", "")))
            reopened_models_adjacent = (
                json.loads(model_bundle_adjacent.read_text())
                if model_bundle_adjacent is not None
                and model_bundle_adjacent.is_file()
                and not model_bundle_adjacent.is_symlink()
                else None
            )
            if not (
                canonical_json(post_immutable_gate) == canonical_json(immutable_gate)
                and post_replacement_guard.get("replacement_authorization")
                == "1_OF_1_CONSUMED"
                and post_recovery.get("status") == "PASS"
                and smoke_workflow_state.run_state == "SMOKE_COMPLETE"
                and model_bundle is not None
                and smoke_completed_official == 9
                and smoke_completed_exploratory == 1
                and models_path.is_file()
                and not models_path.is_symlink()
                and sha256_file(models_path) == model_bundle.get("sha256")
                and reopened_models_adjacent == model_bundle
                and model_bundle.get("crc_status") == "PASS"
                and model_bundle.get("reopen_member_hash_status") == "PASS"
                and model_bundle.get("semantic_verification", {}).get("status")
                == "PASS"
                and amendment04_recovery.read_smoke_report_roundtrip(
                    run_dir / "tables/SMOKE_REPORT.csv"
                ).shape[0] == 10
            ):
                raise IntegrityError("Amendment 04 final immutable/recovery gate failed")
            final_bundle["amendment04_final_validation"] = {
                "status": "PASS",
                "authorization": _AMENDMENT04_AUTHORIZATION,
                "replacement_authorization": "1_OF_1_CONSUMED",
                "amendment03_authorization": "1_OF_1_CONSUMED",
                "accepted_preflight_zip_sha256_before": (
                    (immutable_gate or {}).get("accepted_preflight", {}) or {}
                ).get("review_zip_sha256"),
                "accepted_preflight_zip_sha256_after": (
                    post_immutable_gate.get("accepted_preflight", {}) or {}
                ).get("review_zip_sha256"),
                "failed_amendment03_tree_sha256_before": (
                    (immutable_gate or {}).get("failed_amendment03_smoke", {}) or {}
                ).get("tree_sha256"),
                "failed_amendment03_tree_sha256_after": (
                    post_immutable_gate.get("failed_amendment03_smoke", {}) or {}
                ).get("tree_sha256"),
                "sealed_probe_sha256": post_recovery.get(
                    "sealed_probe_sha256"
                ),
                "amendment04_lineage_sha256": post_recovery.get(
                    "lineage_sha256", post_recovery.get("recovery_lineage_sha256")
                ),
                "csv_round_trip_mismatch_count": 0,
                "official_smoke_variants_completed": smoke_completed_official,
                "exploratory_smoke_variants_completed": (
                    smoke_completed_exploratory
                ),
                "models_bundle_created": True,
                "full_training_executed": False,
                "scientific_results_claimed": False,
            }
        canonical_bundle = Path(final_bundle["bundle_path"])
        if amendment04_csv_roundtrip:
            final_bundle["amendment04_canonical_review_reopen"] = (
                verify_published_review_bundle(
                    run_dir, canonical_bundle, final_bundle,
                )
            )
        adjacent_path = (
            STUDY_ROOT / "results" / f"{run_dir.name}_package_verification.json"
        )
        publish_json_no_clobber(
            adjacent_path, final_bundle, ownership=transaction_ownership,
        )
        reopened_adjacent = json.loads(adjacent_path.read_text())
        if (
            reopened_adjacent != final_bundle
            or not canonical_bundle.is_file()
            or sha256_file(canonical_bundle) != final_bundle["bundle_sha256"]
        ):
            raise IntegrityError("Final Smoke bundle or adjacent verification drifted")
        if amendment04_csv_roundtrip:
            validate_amended_run_reference(
                run_dir,
                expected_kind="CUDA_SMOKE_NON_SCIENTIFIC",
                allowed_states={"SMOKE_COMPLETE"},
            )
            final_immutable_gate = amendment04_recovery.validate_immutable_gate(
                study_root=STUDY_ROOT,
            )
            final_guard = amendment04_recovery.assert_run_guard(
                results_root=STUDY_ROOT / "results",
                new_run_id=run_name,
            )
            final_recovery = amendment04_recovery.validate_recovery_artifacts(
                run_dir=run_dir,
                allowed_after_hashes=amendment04_allowed_after_hashes or {},
                study_root=STUDY_ROOT,
                immutable_gate=final_immutable_gate,
                require_prompt_artifacts=True,
            )
            if not (
                canonical_json(final_immutable_gate) == canonical_json(immutable_gate)
                and final_guard.get("authorization") == "1_OF_1_CONSUMED"
                and final_recovery.get("status") == "PASS"
                and reopened_adjacent.get(
                    "amendment04_final_validation", {}
                ).get("status") == "PASS"
            ):
                raise IntegrityError("Amendment 04 post-publication recheck failed")
        if staging_destination is not None:
            remove_owned_publication_required(
                staging_destination,
                ownership=transaction_ownership,
                purpose="staging bundle",
            )
        print(_smoke_status_text(
            state=smoke_workflow_state,
            run_dir=run_dir,
            preflight_run=preflight_run,
            preflight_report=report,
            split_hash=split_hash,
            model_bundle=model_bundle_reference,
            review_bundle=final_bundle,
            amendment03_recovery=amendment03_recovery_validation,
            amendment04_recovery_validation=amendment04_recovery_validation,
        ), end="")
        print("PHASE_COMPLETE=CUDA_SMOKE_REVIEW_PACKAGE STATUS=PASS", flush=True)
    except Exception as exc:
        try:
            rollback_smoke_final_seal_failure(
                run_dir=run_dir,
                preflight_run=preflight_run,
                preflight_report=report,
                split_hash=split_hash,
                model_bundle=model_bundle_reference,
                gate_inputs=smoke_gate_inputs,
                facts=smoke_facts,
                smoke_metrics=smoke_metrics,
                error=exc,
                ownership=transaction_ownership,
                amendment03_replacement=amendment03_replacement,
                amendment03_recovery_validation=(
                    amendment03_recovery_validation
                ),
                amendment04_csv_roundtrip=amendment04_csv_roundtrip,
                amendment04_recovery_validation=(
                    amendment04_recovery_validation
                ),
                command_line=command_line,
            )
        finally:
            if staging_destination is not None:
                remove_owned_publication(
                    staging_destination, ownership=transaction_ownership,
                )
        post_failure_recheck_error: Exception | None = None
        if amendment04_csv_roundtrip:
            try:
                post_failure_gate = amendment04_recovery.validate_immutable_gate(
                    study_root=STUDY_ROOT,
                )
                post_failure_guard = amendment04_recovery.assert_run_guard(
                    results_root=STUDY_ROOT / "results",
                    new_run_id=run_name,
                )
                if not (
                    canonical_json(post_failure_gate) == canonical_json(immutable_gate)
                    and post_failure_guard.get("replacement_authorization")
                    == "1_OF_1_CONSUMED"
                ):
                    raise IntegrityError(
                        "Amendment 04 post-failure immutable/authorization drift"
                    )
                atomic_write_json(
                    run_dir / "provenance/amendment04_post_failure_recheck.json",
                    {
                        "status": "PASS",
                        "authorization": _AMENDMENT04_AUTHORIZATION,
                        "immutable_gate_status": post_failure_gate.get("status"),
                        "replacement_authorization": post_failure_guard.get(
                            "replacement_authorization"
                        ),
                        "failed_phase": "CUDA_SMOKE_REVIEW_PACKAGE",
                    },
                )
            except Exception as recheck_exc:
                post_failure_recheck_error = recheck_exc
        if post_failure_recheck_error is not None:
            raise IntegrityError(
                "Smoke package transaction failed and the Amendment 04 "
                f"post-failure recheck also failed: {post_failure_recheck_error}"
            ) from exc
        raise IntegrityError(
            f"Smoke package transaction failed and was rolled back: {exc}"
        ) from exc
    return run_dir


def _cleanup_amendment03_models_after_unhandled_failure(run_dir: Path) -> None:
    """Remove only a cryptographically bound Models publication for this Run."""

    models = run_dir.parent / f"{run_dir.name}_NON_SCIENTIFIC_smoke_models.zip"
    adjacent = run_dir.parent / (
        f"{run_dir.name}_NON_SCIENTIFIC_smoke_models_package_verification.json"
    )
    models_exists = models.is_file() and not models.is_symlink()
    adjacent_exists = adjacent.is_file() and not adjacent.is_symlink()
    if not models_exists and not adjacent_exists:
        return
    try:
        record = json.loads(adjacent.read_text()) if adjacent_exists else None
        if record is not None and (
            record.get("run_id") != run_dir.name
            or record.get("path") != str(models)
            or (
                models_exists
                and record.get("sha256") != sha256_file(models)
            )
        ):
            raise IntegrityError(
                "A partial Models publication is not bound to this Amendment Run"
            )
        if models_exists and record is None:
            verification = verify_zip(models)
            expected_prefix = f"{run_dir.name}_smoke_models/"
            with zipfile.ZipFile(models) as archive:
                model_names = archive.namelist()
            if not (
                verification.get("status") == "PASS"
                and model_names
                and len(model_names) == len(set(model_names))
                and all(name.startswith(expected_prefix) for name in model_names)
                and f"{expected_prefix}BUNDLE_MANIFEST.tsv" in model_names
            ):
                raise IntegrityError(
                    "Unaccompanied Models ZIP is not provably owned by this Run"
                )
        model_removed = True
        if models_exists:
            model_token = _capture_regular_publication_token(
                models,
                expected_sha256=(
                    str(record["sha256"]) if record is not None
                    else sha256_file(models)
                ),
            )
            model_removed = _remove_publication_with_token(models, model_token)
        adjacent_removed = True
        if adjacent_exists:
            adjacent_token = _capture_regular_publication_token(
                adjacent, expected_sha256=sha256_file(adjacent),
            )
            adjacent_removed = _remove_publication_with_token(
                adjacent, adjacent_token,
            )
        if not (model_removed and adjacent_removed):
            raise IntegrityError(
                "A bound recovery Models publication changed during cleanup"
            )
        verification = run_dir / "provenance/smoke_model_bundle_verification.json"
        if verification.is_file() and not verification.is_symlink():
            verification.unlink()
    except Exception:
        raise


def run_cuda_smoke(
    *,
    preflight_run: Path,
    command_line: str = "",
    amendment03_replacement: bool = False,
    amendment04_csv_roundtrip: bool = False,
) -> Path:
    """Run CUDA Smoke, with an outer fail-closed Corrigendum transaction guard."""

    preexisting_amendment03_runs = {
        path.name
        for path in (STUDY_ROOT / "results").glob(
            "run_*_amendment03_cuda_smoke"
        )
        if path.is_dir() and not path.is_symlink()
    } if amendment03_replacement else set()
    amendment04_created_run: dict[str, Path] = {}
    try:
        return _run_cuda_smoke_impl(
            preflight_run=preflight_run,
            command_line=command_line,
            amendment03_replacement=amendment03_replacement,
            amendment04_csv_roundtrip=amendment04_csv_roundtrip,
            created_run_holder=(
                amendment04_created_run if amendment04_csv_roundtrip else None
            ),
        )
    except Exception as exc:
        if amendment03_replacement:
            candidates = sorted(
                path for path in (STUDY_ROOT / "results").glob(
                    "run_*_amendment03_cuda_smoke"
                )
                if path.is_dir() and not path.is_symlink()
                and path.name not in preexisting_amendment03_runs
            )
            if len(candidates) == 1:
                run_dir = candidates[0]
                _cleanup_amendment03_models_after_unhandled_failure(run_dir)
                smoke_failure = run_dir / "logs/smoke_failure.txt"
                existing = (
                    smoke_failure.read_text()
                    if smoke_failure.is_file() and not smoke_failure.is_symlink()
                    else ""
                )
                if "PHASE=" not in existing or "TRACEBACK_BEGIN" not in existing:
                    _write_amendment03_structured_failure(
                        run_dir=run_dir,
                        phase="UNHANDLED_POST_AUTHORIZATION_FAILURE",
                        command_line=command_line,
                        error=exc,
                    )
        elif amendment04_csv_roundtrip:
            owned_run = amendment04_created_run.get("run_dir")
            if (
                owned_run is not None
                and owned_run.parent.resolve()
                == (STUDY_ROOT / "results").resolve()
                and owned_run.is_dir()
                and not owned_run.is_symlink()
            ):
                run_dir = owned_run
                _cleanup_amendment03_models_after_unhandled_failure(run_dir)
                smoke_failure = run_dir / "logs/smoke_failure.txt"
                existing = (
                    smoke_failure.read_text()
                    if smoke_failure.is_file() and not smoke_failure.is_symlink()
                    else ""
                )
                if "PHASE=" not in existing or "TRACEBACK_BEGIN" not in existing:
                    _write_amendment04_structured_failure(
                        run_dir=run_dir,
                        phase="UNHANDLED_POST_AUTHORIZATION_FAILURE",
                        command_line=command_line,
                        error=exc,
                    )
                try:
                    failure_gate = amendment04_recovery.validate_immutable_gate(
                        study_root=STUDY_ROOT,
                    )
                    failure_guard = amendment04_recovery.assert_run_guard(
                        results_root=STUDY_ROOT / "results",
                        new_run_id=run_dir.name,
                    )
                    if not (
                        failure_gate.get("status") == "PASS"
                        and failure_guard.get("replacement_authorization")
                        == "1_OF_1_CONSUMED"
                    ):
                        raise IntegrityError(
                            "Amendment 04 post-failure immutable/authorization drift"
                        )
                    recheck_path = (
                        run_dir
                        / "provenance/amendment04_post_failure_recheck.json"
                    )
                    if not recheck_path.exists():
                        atomic_write_json(
                            recheck_path,
                            {
                                "status": "PASS",
                                "authorization": _AMENDMENT04_AUTHORIZATION,
                                "immutable_gate_status": failure_gate.get("status"),
                                "replacement_authorization": failure_guard.get(
                                    "replacement_authorization"
                                ),
                                "failed_phase": (
                                    "UNHANDLED_POST_AUTHORIZATION_FAILURE"
                                ),
                            },
                        )
                except Exception as recheck_exc:
                    raise IntegrityError(
                        "Amendment 04 failure evidence was written, but its final "
                        f"immutable recheck failed: {recheck_exc}"
                    ) from exc
        raise
