"""Amendment 06 resumable Full scientific ablation runner.

This module is the scientific execution boundary for Amendment 06.  It never
delegates to the obsolete legacy Full runner.  Package construction is kept in
``amendment06_packaging`` and is imported only by :func:`package_dispatch`.
"""

from __future__ import annotations

import base64
import binascii
from collections import Counter
from contextlib import contextmanager, suppress
from contextvars import ContextVar
import ctypes
import csv
from dataclasses import dataclass
from datetime import datetime
import difflib
import errno
import fcntl
import gzip
import hashlib
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import stat
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence
import uuid
import warnings
import zipfile

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import xgboost as xgb

import ablation_core as legacy
import amendment02_resampling as paired_resampling
import amendment03_compat as model_compat
import amendment_core as core
import amendment06_reporting as reporting


STUDY_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path("/REVIEWER_INPUT_ROOT/Clean")
RESULTS_ROOT = STUDY_ROOT / "results"
A06_RUNTIME = STUDY_ROOT / ".runtime/amendment06_full"
EXPECTED_RUNTIME_ROOT = A06_RUNTIME / "runtime"
PRODUCTION_RESULTS_ROOT = (STUDY_ROOT / "results").resolve()
PRODUCTION_EXPECTED_RUNTIME_ROOT = (
    STUDY_ROOT / ".runtime/amendment06_full/runtime"
).resolve()
CUDA_ENV = Path("/REVIEWER_INPUT_ROOT/SAM_ablation_cuda_xgb211_wheel")
CUDA_PYTHON = CUDA_ENV / "bin/python"

A07_PROMPT_PATH = Path(
    "/REVIEWER_INPUT_ROOT/07_RUN_NOW_Amendment07_GPU_Identity_Same_Run_Resume.txt"
)
A07_PROMPT_SHA256 = "6508bbee0457d59ab468ec7480fc486cc562a23217158e384aed71e3b4ec7c13"
A07_AUTHORIZATION = "AMENDMENT_07_GPU_IDENTITY_CORRIGENDUM_SAME_RUN_RESUME"
A07_TARGET_RUN_ID = "run_20260820_152459_818052_e1120801_amendment06_full"
A07_TARGET_RUN = RESULTS_ROOT / A07_TARGET_RUN_ID
A07_RUNTIME = STUDY_ROOT / ".runtime/amendment07_gpu_identity_corrigendum"
A07_A06_BOUND_RUNTIME = A06_RUNTIME / "corrigendum07_gpu_identity"
A07_INSTALL_INTENT = A07_RUNTIME / "authorization/amendment07_overlay_install_intent.json"
A07_FORENSIC_ZIP = (
    A07_RUNTIME / "forensic/amendment07_pre_mutation_forensic_bundle.zip"
)
A07_FORENSIC_VERIFICATION = (
    A07_RUNTIME / "forensic/amendment07_pre_mutation_forensic_verification.json"
)
A07_FORENSIC_RECONSTRUCTION = (
    A07_RUNTIME / "forensic/amendment07_pre_correction_reconstruction.json"
)
A07_FAILED_ATTEMPT_QUARANTINE_INTENT = (
    A07_RUNTIME / "forensic/failed_attempt_quarantine_intent.json"
)
A07_FAILED_ATTEMPT_QUARANTINE_RECEIPT = (
    A07_RUNTIME / "forensic/failed_attempt_quarantine_receipt.json"
)

A07_BASE_IDENTITY_SHA256 = "ba324ffe3ecb3c186a48437cb0cafa0c983b82fb05d0a65183db112095652da3"
A07_BASE_SCIENTIFIC_MANIFEST_SHA256 = (
    "c99ae66631266d3d920d9079cee5d33a16d112cfb40eacd45a13fd5716e7fd55"
)
A07_BASE_CODE_LINEAGE_SHA256 = "1867a029eed19f61bed46f0ca4bf6b5f7d0883e82d0d3495d18d96726f1b71eb"
A07_OLD_FULL_SHA256 = "fdafceabff535a94415b068fa651642b1149f7fb783df08be6388e081fd9b719"
A07_OLD_TEST_SHA256 = "3d673326fc8b614ff9953b7d525352029e5ff90b6fc7d35cca25f921f7f9c095"
A07_UNCHANGED_CORE_SHA256 = "e8641e4cd9dede67f83acb491acee79eeb62da0e34a12f777caf719ada047e18"
A07_BASE_LIVE_CODE_TREE_SHA256 = (
    "22bab2b51e818e8ba0150a8216a76ac2df3a7a9e2b1e545f132473f94f97766d"
)
A07_A06_INTENT_SHA256 = "92def243dd13892ce2fb2172e0f5d21994cf8166d44476cb31b7ab4053f90513"
A07_A06_RECEIPT_SHA256 = "8fe482e886911f8558b83ef18f475b536453abfe8a0706613af65c5e4d259e13"
A07_ORIGINAL_LEDGER_SHA256 = "8a6e5b9579756116c16318b097bd1ca25f6224b813fbf8b31184887ca8d25803"
A07_ORIGINAL_LEDGER_SIZE_BYTES = 41_804
A07_ORIGINAL_PRERUN_SHA256 = "74146ca93b2d5c36c28c07542c53e154e1273c609e3273d558fc1e55e2f7f02b"
A07_ORIGINAL_FULL_LOG_SHA256 = "3ef479550bfd329477209fadcaccce22a1ce9ed5a12fd233c06ef0fcb38ab9b9"
A07_CACHE_MANIFEST_SHA256 = "66ef5e350bd14a5afd8a799c02ac13d89c0ae3232a15336c9e878308a0643752"
A07_FROZEN_POPULATION_SHA256 = "902fc0984bbc593ed08df1c0e8e5293ec0cf9ccb32570649a4a822f299f8b967"
A07_PORTABLE_IMPUTER_SHA256 = "ed416353af9aa4a166b22558a789c064f44747999fcabebbb919f33ef6899939"
A07_IMMUTABLE_CUDA_PROBE_SHA256 = (
    "629ce45ce6c23bbc46feac26cbec3ec8cfa7f9e333c3ef595f74065e4c6820a9"
)
A07_FORENSIC_ZIP_SHA256 = "43d5b40a3342ab71d61e76c4d35ef5766801b1d40875367b745b4d052762abc1"
A07_FORENSIC_VERIFICATION_SHA256 = (
    "a236048593694c52d03007e6a31daeed105458d1b02bb708ba243da9ba3c09db"
)
A07_FORENSIC_SOURCE_INVENTORY_SHA256 = (
    "af20e28f6a8c1b06f8ba7be3681e7b71a13e2b234ba03d4fc7706097c3d22b69"
)
A07_FORENSIC_SOURCE_INVENTORY_FILE_SHA256 = (
    "7c082d7f55a9bf597c36f457031389fb99ecfb34261dcf2d18b0c898e5dfec82"
)
A07_FORENSIC_CACHE_INVENTORY_SHA256 = (
    "b794e2f1df39eeba68578a90643dc2c0e12253aea25fdb81b1cc784c6869fc00"
)
A07_FORENSIC_FACTS_SHA256 = "3864f7f01cf6ffdfd72679c581f0116d624815d238b2c3c9cf64cf0d091dc356"
A07_FAILED_ATTEMPT_QUARANTINE_INTENT_SHA256 = (
    "31df35c8768026ce5c12b83cd27df19972e0923433451cd5b6e7bbecddb7ba94"
)
A07_FAILED_ATTEMPT_QUARANTINE_RECEIPT_SHA256 = (
    "3d98718589dcc57d59041b9ff9aae0d9ed49cc7060924dadcddd0e2f16cdb89b"
)
A07_FORENSIC_RECONSTRUCTION_SHA256 = (
    "22f092f845757d6f9d8f060ba36ec0a278022a67def22eeb1f8af544e9d01f10"
)
A07_FAILED_ATTEMPT_ID = "4673f3f5746a4ca0840eb9dc8c315e8c"
A07_FAILED_JOURNAL_EVENT_RAW_SHA256 = (
    "1ea9441a56e3bfd6c1897bad7e2928280e423c5a5ad0036aba7006b165f8dd11"
)
A07_PRECORRIGENDUM_JOURNAL_HEAD_SHA256 = (
    "807e256ef78b4423a393eec64d52a173929ee1d3b7ef20835db4003fbdf5e5c8"
)
A07_JOURNAL_PREFIX_RAW_SHA256 = (
    "2d2a931df30202a2932ca13e7535338dfc23a62c3e4f8269579503572bd01a63"
)
A07_JOURNAL_RAW_SHA256 = (
    "1692939ada7effbbce2eb7da313406317fb1a276ad303d160e98f768ea042355",
    "8285db1a2b1a512177104864429afee8292235ae6cfbfbbd6b7adc76da2c26e6",
    "9fd9430c19a86e33cb2d2ac170f3b589ad518a38fa3c563e84c7887eea11b905",
    "127a23e215ff571b1d1171840bf7a574729f64fbcd360e7b77523dbbc7902f0b",
    A07_FAILED_JOURNAL_EVENT_RAW_SHA256,
)
A07_CHANGED_SCIENTIFIC_RELATIVES = (
    "code/amendment06_full.py",
    "tests/test_amendment06.py",
)
A07_OVERLAY_RUN_RELATIVES = (
    "provenance/amendment07_external_authorization.json",
    "provenance/amendment07_prompt_snapshot.txt",
    "provenance/amendment07_code_corrigendum.json",
    "provenance/scientific_execution_code_manifest_effective_amendment07.tsv",
    "provenance/amendment07_failed_attempt_forensics.json",
    "provenance/amendment07_resume_precheck.json",
    "provenance/amendment07_cuda_producer_consumer_probe.json",
    "provenance/amendment07_overlay_install_receipt.json",
    "provenance/code_snapshot_amendment07/amendment06_full.py",
    "provenance/tests_snapshot_amendment07/test_amendment06.py",
)
A07_MODEL_BUNDLE_GLOBALS = frozenset({
    "provenance/amendment07_external_authorization.json",
    "provenance/amendment07_code_corrigendum.json",
    "provenance/scientific_execution_code_manifest_effective_amendment07.tsv",
    "provenance/amendment07_failed_attempt_forensics.json",
    "provenance/amendment07_resume_precheck.json",
    "provenance/amendment07_overlay_install_receipt.json",
    "provenance/code_snapshot_amendment07/amendment06_full.py",
})

A08_PROMPT_PATH = Path(
    "/REVIEWER_INPUT_ROOT/08_RUN_NOW_Amendment08_Journal_Adjudication_Same_Run_Resume.txt"
)
A08_PROMPT_SHA256 = "3ebb61bb1872c2d9e0c9e68c56a8c794ed07e5d4b1cb511298ba335cf9e2ce60"
A08_AUTHORIZATION = (
    "AMENDMENT_08_EXACT_PYTEST_JOURNAL_ADJUDICATION_GPU_CORRIGENDUM_"
    "SAME_RUN_CONTINUATION"
)
A08_TARGET_RUN_ID = A07_TARGET_RUN_ID
A08_TARGET_RUN = RESULTS_ROOT / A08_TARGET_RUN_ID
A08_RUNTIME = STUDY_ROOT / ".runtime/amendment08_journal_adjudication"
A08_A06_BOUND_RUNTIME = A06_RUNTIME / "corrigendum08_journal_adjudication"
A08_INSTALL_INTENT = A08_RUNTIME / "authorization/amendment08_overlay_install_intent.json"
A08_SOURCE_FREEZE_INTENT = A08_RUNTIME / "source_freeze/install_intent.json"
A08_SOURCE_FREEZE_RECEIPT = A08_RUNTIME / "source_freeze/install_receipt.json"
A08_ISOLATED_REHEARSAL_PROJECTION_SCHEMA = (
    "amendment08_isolated_rehearsal_projection/v1"
)
A08_ISOLATED_REHEARSAL_PROJECTION_MODE = "ISOLATED_REHEARSAL_PROJECTION"
A08_ISOLATED_REHEARSAL_PROJECTION_DOMAIN = (
    "amendment08_isolated_rehearsal_projection_basis/v1"
)
A08_REHEARSAL_HANDOFF_SCHEMA = "amendment08_resume_rehearsal_handoff/v1"
A08_REHEARSAL_HANDOFF_KEYS = frozenset({
    "schema", "status", "run_id", "run_path", "runtime_path",
    "output_manifest_sha256", "output_manifest_rows", "run_tree_sha256",
    "cache_source_inventory_sha256", "cache_clone_inventory_sha256",
    "live_journal_before_sha256", "live_journal_after_sha256",
})
A08_PRELIVE_JOURNAL_SENTINEL_SHA256 = (
    "1540d41a501f13a0b43eae104532da205aa762dcb20ebca50c66d409248fcc5f"
)
A08_CACHE_MEMBER_COUNT = 10
A08_CACHE_MEMBER_BYTES = 1_280_873_017
A08_REHEARSAL_CACHE_FILE_COUNT = 11
A08_REHEARSAL_BASE_RUN_FILE_COUNT = 45
A08_REHEARSAL_BASE_RUN_SIZE_BYTES = 19_100_427
A08_REHEARSAL_BASE_RUN_INVENTORY_SHA256 = (
    "0e47f82bc1e7763cb4d8b431fa3d5c2d1f01c0299cd3fbfdf7d669e1cf409909"
)

_AMENDMENT08_REHEARSAL_CONTEXT: ContextVar[tuple[Path, Path] | None] = ContextVar(
    "amendment08_isolated_rehearsal_context", default=None,
)
A08_FORENSIC_ZIP = (
    A08_RUNTIME / "forensic/amendment08_pre_mutation_forensic_bundle.zip"
)
A08_FORENSIC_VERIFICATION = (
    A08_RUNTIME / "forensic/amendment08_pre_mutation_forensic_verification.json"
)
A08_EXTERNAL_REVIEW_ZIP = (
    RESULTS_ROOT / "amendment07_journal_contamination_review_20260820_182751.zip"
)
A08_FORENSIC_ZIP_SHA256 = "e39681488d77154bdeecf6e988537898e96e63d6e9ef852d665c1e0894e1f841"
A08_FORENSIC_ZIP_SIZE_BYTES = 61_222_626
A08_FORENSIC_VERIFICATION_SHA256 = (
    "af3b70bf03b1d237936e8c21ce84b658e98008a598e3b08436b9696ed311ff33"
)
A08_FORENSIC_VERIFICATION_SIZE_BYTES = 26_472
A08_FORENSIC_TREE_SHA256 = "5d463c9c70f0dd8b19f43e4effade5ca8b57b37602d73f47c5d76211ce7f19bb"
A08_EXTERNAL_REVIEW_ZIP_SHA256 = "b94c117a144733af08e51770245d2f3c397c71e62b9293094c6cc5db987839a3"
A08_EXTERNAL_REVIEW_ZIP_SIZE_BYTES = 18_211_946
A08_PRE_FULL_SHA256 = "c1490bb85026298fed7a4cfe9244e4331aec35881e6777128155886e935a96e9"
A08_PRE_TEST_SHA256 = "6e7a421720219ddbc017c13ee74460973a9b4e7ec0b5247a2965f5002c52ef08"
A08_PRE_PACKAGE_SHA256 = "e4e36d87056f8a8ce60dfccaa3267562928134de49432c4b102a912476908b35"
A08_PRE_PACKAGE_TEST_SHA256 = "e09d16ad30639d50d3edfbc64dba4262fe81260d2076e71a5cdc617308c98981"
A08_PRE_SOURCE_HASHES: Mapping[str, str] = {
    "code/amendment06_full.py": A08_PRE_FULL_SHA256,
    "code/amendment06_packaging.py": A08_PRE_PACKAGE_SHA256,
    "tests/test_amendment06.py": A08_PRE_TEST_SHA256,
    "tests/test_amendment06_packaging.py": A08_PRE_PACKAGE_TEST_SHA256,
}
A08_CHANGED_SCIENTIFIC_RELATIVES = (
    "code/amendment06_full.py", "tests/test_amendment06.py",
)
A08_CHANGED_PACKAGE_RELATIVES = (
    "code/amendment06_packaging.py", "tests/test_amendment06_packaging.py",
)
A08_CHANGED_ALL_RELATIVES = (
    "code/amendment06_full.py", "code/amendment06_packaging.py",
    "tests/test_amendment06.py", "tests/test_amendment06_packaging.py",
)
A08_INITIAL_PACKAGE_MANIFEST_SHA256 = (
    "491595ca490a6ab6d8acc2c824ac0366bb97cde773bbe6fdb2cf30859a53ff9d"
)
A08_INITIAL_PACKAGE_HASHES: Mapping[str, str] = {
    "code/amendment06_packaging.py": "39ab3c7dfaf30054447c038e839f12e06d6439e402cd18966b937d040f732c31",
    "tests/test_amendment06_packaging.py": "4941b3777102c3009cf70e67f4e2a12662ee346f9f6964458d41fe682e8c7099",
}
A08_PACKAGE_SET_ID = "AMENDMENT08_PRELIVE_PACKAGE_ALIGNMENT"
A08_PACKAGE_LINEAGE_ROOT = (
    EXPECTED_RUNTIME_ROOT / "package_code_lineage" / A08_TARGET_RUN_ID
)
A08_PACKAGE_A08_LINEAGE_ROOT = A08_PACKAGE_LINEAGE_ROOT / "amendment08"
A08_REQUIRED_ZERO_SCIENCE_API_NAMES = frozenset({
    "ablation_core.SafeSMOTE.fit_resample",
    "ablation_core.apply_training_augmentation",
    "ablation_core.safe_smote_resample",
    "amendment02_resampling.build_frozen_resampling_population",
    "amendment_core.augment_with_lineage",
    "amendment_core.augmentation_plan_counts",
    "amendment_core.replay_augmentation_weights",
    "amendment06_full._binary_metrics",
    "amendment06_full._build_population_cache",
    "amendment06_full._build_reports_and_bootstrap_impl",
    "amendment06_full._execute_scientific_full",
    "amendment06_full._external_prediction_frame",
    "amendment06_full._frozen_imputer_transform",
    "amendment06_full._journal_event",
    "amendment06_full._publish_attempt_file",
    "amendment06_full._recompute_and_validate_reporting_outputs",
    "amendment06_full._record_interrupted_attempts",
    "amendment06_full._recover_unpublished_paths",
    "amendment06_full._score_one_external_variant_impl",
    "amendment06_full._set_run_state",
    "amendment06_full._snapshot_source_post",
    "amendment06_full._stage_bytes",
    "amendment06_full._stage_json",
    "amendment06_full._training_prediction_frame",
    "amendment06_full._validate_reporting_completion",
    "amendment06_full._validate_amendment08_corrected_training_history",
    "amendment06_full._validate_amendment08_frozen_prerequisite_receipts",
    "amendment06_full.build_metric_recomputation",
    "amendment06_full.build_reports_and_bootstrap",
    "amendment06_full.complete_external_audit_phase",
    "amendment06_full.deterministic_csv_gzip_bytes",
    "amendment06_full.ensure_blocked_no_pca",
    "amendment06_full.exact_table_csv_bytes",
    "amendment06_full.finalize_scientific_run",
    "amendment06_full.freeze_all_models",
    "amendment06_full.initialize_run_evidence",
    "amendment06_full.install_amendment08_overlay",
    "amendment06_full.install_amendment08_source_freeze",
    "amendment06_full.load_or_build_population_cache",
    "amendment06_full.open_external_audit_after_freeze",
    "amendment06_full.output_manifest_bytes",
    "amendment06_full.publish_bytes_no_clobber",
    "amendment06_full.publish_strict_json_no_clobber",
    "amendment06_full.project_variant",
    "amendment06_full.read_amendment08_journal_views",
    "amendment06_full.resume_full",
    "amendment06_full.run_amendment08_cuda_producer_consumer_probe",
    "amendment06_full.run_full",
    "amendment06_full.score_one_external_variant",
    "amendment06_full.train_one_variant",
    "amendment06_full.validate_amendment07_overlay",
    "amendment06_full.validate_amendment08_overlay",
    "amendment06_full.validate_amendment08_pre_resume",
    "amendment06_full.validate_exact_table_csv",
    "amendment06_full.validate_external_scoring_completion",
    "amendment06_full.validate_final_scientific_run",
    "amendment06_full.validate_prediction_roundtrip",
    "amendment06_reporting._bootstrap_metrics",
    "amendment06_reporting.binary_metrics",
    "amendment06_reporting.build_all_reports",
    "amendment06_reporting.build_scientific_reports",
    "amendment06_reporting.exact_prediction_csv_bytes",
    "amendment06_reporting.paired_card_bootstrap",
    "amendment06_reporting.per_card_metrics",
    "amendment06_packaging._binary_metrics_from_frame",
    "amendment06_packaging._per_card_expected",
    "amendment06_packaging._read_prediction",
    "amendment06_packaging._validate_external_source_population",
    "amendment06_packaging._validate_locked_validation_population",
    "amendment06_packaging._validate_reports",
    "sklearn.impute.SimpleImputer.fit",
    "sklearn.impute.SimpleImputer.fit_transform",
    "xgboost.Booster.boost",
    "xgboost.Booster.inplace_predict",
    "xgboost.Booster.predict",
    "xgboost.Booster.update",
    "xgboost.XGBClassifier.fit",
    "xgboost.XGBClassifier.predict",
    "xgboost.XGBClassifier.predict_proba",
    "xgboost.cv",
    "xgboost.train",
})
A08_PACKAGE_LINEAGE_EVENT_KEYS = frozenset({
    "schema", "status", "authorization", "prompt_sha256", "target_run_id",
    "package_set_id", "set_event_index", "sequence", "target_relative_path",
    "previous_code_sha256", "new_code_sha256", "previous_event_sha256",
    "initial_package_validator_manifest_sha256", "targeted_tests_log_sha256",
    "package_rehearsal_log_sha256", "package_rehearsal_evidence_sha256",
    "zero_science_evidence_path", "zero_science_evidence_sha256",
    "source_snapshot_path", "source_snapshot_sha256", "scientific_api_calls",
})
A08_PACKAGE_LINEAGE_EVIDENCE_KEYS = frozenset({
    "schema", "status", "authorization", "prompt_sha256", "target_run_id",
    "package_set_id", "sequence", "target_relative_path", "previous_code_sha256",
    "new_code_sha256", "source_snapshot_path", "source_snapshot_sha256",
    "targeted_tests_log", "package_rehearsal_log",
    "package_rehearsal_evidence_path", "package_rehearsal_evidence_sha256",
    "guarded_scientific_api_calls", "guarded_api_names", "run_mutation_calls",
    "live_runtime_journal_reads", "receipt_only_validation",
})
A08_PACKAGE_LINEAGE_RECORD_KEYS = frozenset({
    "sequence", "target_relative_path", "previous_code_sha256", "new_code_sha256",
    "snapshot_relative_path", "snapshot_sha256", "evidence_relative_path",
    "evidence_sha256", "event_relative_path", "event_sha256",
    "event_canonical_sha256",
})
A08_PACKAGE_LINEAGE_INTENT_KEYS = frozenset({
    "schema", "status", "authorization", "prompt_sha256", "target_run_id",
    "package_set_id", "set_kind", "created_at",
    "initial_package_validator_manifest_sha256", "initial_hashes",
    "forensic_pre_a08_draft_hashes", "final_hashes", "base_lineage_event_count",
    "base_lineage_head_sha256", "targeted_tests_log", "package_rehearsal_log",
    "package_rehearsal_evidence", "event_count", "records",
})
A08_PACKAGE_LINEAGE_RECEIPT_KEYS = frozenset({
    "schema", "status", "authorization", "prompt_sha256", "target_run_id",
    "package_set_id", "set_kind", "intent_sha256",
    "initial_package_validator_manifest_sha256", "initial_hashes",
    "forensic_pre_a08_draft_hashes", "final_hashes", "event_count",
    "event_hashes", "lineage_event_count", "lineage_sha256",
    "lineage_head_sha256", "evidence_hashes", "snapshot_hashes",
    "targeted_tests_log", "package_rehearsal_log", "package_rehearsal_evidence",
    "scientific_api_calls", "installed_at",
})
A08_PACKAGE_REHEARSAL_EVIDENCE_KEYS = frozenset({
    "schema", "status", "authorization", "prompt_sha256", "target_run_id",
    "package_set_id", "recorded_at", "fixture_run_id", "targeted_tests_log",
    "package_rehearsal_log", "initial_package_validator_manifest_sha256",
    "initial_hashes", "final_hashes", "expected_run_manifest_rows",
    "actual_run_regular_files", "review_selected_run_rows",
    "review_embedded_manifest_rows", "review_zip_file_entry_count",
    "models_selected_run_rows", "models_embedded_manifest_rows",
    "models_zip_file_entry_count", "receipt_only_validation",
    "independent_zip_reopen", "package_retry_status",
    "guarded_scientific_api_calls", "guarded_api_names", "run_mutation_calls",
    "live_runtime_journal_reads",
})

A08_RAW_JOURNAL_SHA256 = (
    "1692939ada7effbbce2eb7da313406317fb1a276ad303d160e98f768ea042355",
    "8285db1a2b1a512177104864429afee8292235ae6cfbfbbd6b7adc76da2c26e6",
    "9fd9430c19a86e33cb2d2ac170f3b589ad518a38fa3c563e84c7887eea11b905",
    "127a23e215ff571b1d1171840bf7a574729f64fbcd360e7b77523dbbc7902f0b",
    "1ea9441a56e3bfd6c1897bad7e2928280e423c5a5ad0036aba7006b165f8dd11",
    "7b7439542d7770c044880cffb1dd36f54d3210d43f95d52c6a8689fd521ef306",
    "631e84745fe4dfba9a9a2ab4cd6d6c7a2873bb96d5cbb628cc4950718514a511",
)
A08_CANONICAL_JOURNAL_SHA256 = (
    "a26f9d55ba93a5b638a3f062cd0cf77701dda7efbd2fc7b82493d5d11dc8b5d5",
    "3b64454ef9bc0e3c84df17912201c9f46b3d3a7f2031fb94aec51246b9c97c47",
    "7d4d58f003a9cffabaa85eb6dfd29a4f520db58f660e025a196d063db33da5a1",
    "04371aca73a3454e771d487dfafa9a66de2b1ddb72891cf6de72db37278bef94",
    "807e256ef78b4423a393eec64d52a173929ee1d3b7ef20835db4003fbdf5e5c8",
    "3a3d07335bc86d681d276318266c1850ceb857e1d37e9a86c9e8360b30604b7a",
    "f69e665e5dd05f6c5109a436c02505c0f4198b8e86b83f94806e15fc64021881",
)
A08_RAW_EVENTS_0_TO_6_CONCAT_SHA256 = (
    "e44e6fa1c262690d258949a7c9342cc867bfba7bb40bb2404df3929a234d5719"
)
A08_CANONICAL_EVENTS_0_TO_6_LINES_SHA256 = (
    "6f79efd337ed1609b784b360b019ff378411076800c601a68ea87ef0088d95c6"
)
A08_CONTAMINATION_INDICES = (5, 6)
A08_CONTAMINATION_CLASSIFICATION = "NON_SCIENTIFIC_PYTEST_CONTAMINATION"
A08_TEST_FIXTURE_SHA256 = "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"
A08_CONTAMINATING_TEST = (
    "tests/test_amendment06.py::"
    "test_execution_ledger_invokes_strict_amendment07_final_history"
)
A08_CONTAMINATING_FUNCTION_SHA256 = (
    "7ae8e7aca7b432431a7f719677b95b580d6493eb1b0448528079f12822903faa"
)

A08_LINEAGE_FIELDS = (
    "base_scientific_execution_code_manifest_sha256",
    "effective_scientific_execution_code_manifest_sha256",
    "amendment08_code_corrigendum_sha256",
    "amendment08_journal_contamination_adjudication_sha256",
    "amendment08_authorization", "amendment08_prompt_sha256",
)
A08_REHEARSAL_SOURCE_PROJECTION_KEYS = (
    "source_freeze_receipt_sha256",
    "current_a06_authorized_change_ledger_sha256",
    "a08_source_ledger_event_index",
    "current_a06_final_prerun_evidence_sha256",
    "a08_source_ledger_event_canonical_sha256",
)
A08_REHEARSAL_PACKAGE_PROJECTION_KEYS = (
    "lineage_sha256", "receipt_sha256", "final_code_sha256",
    "final_test_sha256",
)
A08_REHEARSAL_REFERENCE_PROJECTION_KEYS = (
    "status", "validation_mode", "pre_edit_admission_sha256",
    "original_final_prerun_evidence_sha256", "accepted_preflight_run_id",
    "accepted_smoke_run_id", "sealed_smoke_semantic_status",
    "production_validator_signature_status", "external_audit_tabular_parse_calls",
)
A08_REHEARSAL_CACHE_RECORD_KEYS = (
    "relative_path", "size_bytes", "sha256", "source_device",
    "source_inode", "clone_device", "clone_inode",
)
A08_ISOLATED_REHEARSAL_PROJECTION_KEYS = frozenset({
    "schema", "status", "mode", "authorization", "prompt_sha256",
    "target_run_id", "run_path", "runtime_path", "recorded_at",
    "base_identity_sha256", "base_scientific_manifest_sha256",
    "base_code_lineage_sha256", "initial_package_validator_manifest_sha256",
    "initial_package_hashes", "final_source_hashes",
    "raw_events_0_to_6_concat_sha256",
    "canonical_json_lines_events_0_to_6_sha256", "forensic_zip_sha256",
    "forensic_verification_sha256", "base_run_regular_file_count",
    "base_run_inventory_sha256", "cache_source_path", "cache_clone_path",
    "cache_manifest_sha256", "frozen_population_sha256",
    "portable_imputer_sha256", "cache_member_count", "cache_member_bytes",
    "cache_inventory_file_count", "cache_source_inventory_sha256",
    "cache_clone_inventory_sha256", "cache_clone_distinct_inodes",
    "cache_clone_records", "reference_projection", "source_projection",
    "package_projection", "authorization_relocation", "projection_basis_sha256",
    "new_full_run_ids_created", "new_resampling_realizations",
    "production_amendment08_receipts_published",
    "production_authorization_consumed",
    "projection_durable_authorization_consumption",
})
A08_OVERLAY_RUN_RELATIVES = (
    "provenance/amendment08_external_authorization.json",
    "provenance/amendment08_prompt_snapshot.txt",
    "provenance/amendment08_code_corrigendum.json",
    "provenance/scientific_execution_code_manifest_effective_amendment08.tsv",
    "provenance/amendment08_failed_attempt_forensics.json",
    "provenance/amendment08_journal_contamination_adjudication.json",
    "provenance/amendment08_resume_precheck.json",
    "provenance/amendment08_cuda_producer_consumer_probe.json",
    "provenance/amendment08_overlay_install_receipt.json",
    "provenance/code_snapshot_amendment08/amendment06_full.py",
    "provenance/tests_snapshot_amendment08/test_amendment06.py",
)
A08_RAW_JOURNAL_CUTOFF_RELATIVE = (
    "provenance/amendment08_raw_journal_at_reporting_cutoff.json"
)
A08_RAW_JOURNAL_RECORD_KEYS = frozenset({
    "physical_event_index", "relative_path", "raw_size_bytes", "raw_sha256",
    "canonical_sha256", "event", "raw_bytes_base64",
})
A08_RAW_JOURNAL_CUTOFF_KEYS = frozenset({
    "schema", "status", "authorization", "prompt_sha256", "target_run_id",
    "created_at", "record_count", "records",
    "raw_journal_event_count_at_reporting_cutoff",
    "raw_journal_head_sha256_at_reporting_cutoff",
    "raw_journal_canonical_json_lines_sha256_at_reporting_cutoff",
    "semantic_journal_event_count_at_reporting_cutoff",
    "semantic_journal_canonical_json_lines_sha256_at_reporting_cutoff",
    "classified_test_contamination_event_count",
    "classified_test_contamination_event_indices",
    "classified_test_contamination_raw_sha256",
    "classified_test_contamination_canonical_sha256",
    "raw_reporting_completion_events", "semantic_reporting_completion_events",
    "corrigendum_journal_event_index", "corrigendum_journal_event_raw_sha256",
    "corrigendum_journal_head_sha256",
    "amendment08_overlay_install_receipt_sha256",
    "genuine_reporting_completion_event_index",
    "genuine_reporting_completion_event_raw_sha256",
    "genuine_reporting_completion_event_canonical_sha256",
    "genuine_reporting_completion_manifest_sha256",
    "base_scientific_execution_code_manifest_sha256",
    "effective_scientific_execution_code_manifest_sha256",
    "amendment08_code_corrigendum_sha256",
    "amendment08_journal_contamination_adjudication_sha256",
    "full_models_frozen_manifest_sha256", "external_audit_opening_sha256",
    "reporting_completion_manifest_sha256", "metric_recomputation_sha256",
    "source_input_hashes_post_sha256", "previous_state_marker_sha256",
    "planned_full_scientific_complete_marker_recorded_at",
    "planned_full_scientific_complete_marker",
    "planned_full_scientific_complete_marker_sha256",
    "prerequisite_validation_status",
})
A08_EXECUTION_LEDGER_ADDED_FIELDS = frozenset({
    *A08_LINEAGE_FIELDS,
    "amendment08_raw_journal_cutoff_sha256",
    "amendment08_raw_journal_at_reporting_cutoff_sha256",
    "raw_journal_event_count_at_reporting_cutoff",
    "raw_journal_head_sha256_at_reporting_cutoff",
    "raw_journal_canonical_json_lines_sha256_at_reporting_cutoff",
    "semantic_journal_event_count_at_reporting_cutoff",
    "semantic_journal_canonical_json_lines_sha256_at_reporting_cutoff",
    "classified_test_contamination_event_count",
    "classified_test_contamination_event_indices",
    "classified_test_contamination_raw_sha256",
    "classified_test_contamination_canonical_sha256",
    "raw_reporting_completion_events", "semantic_reporting_completion_events",
    "genuine_reporting_completion_event_index",
    "genuine_reporting_completion_event_raw_sha256",
    "genuine_reporting_completion_event_canonical_sha256",
    "current_physical_journal_event_count",
    "current_physical_journal_head_sha256",
    "current_physical_journal_canonical_json_lines_sha256",
    "amendment08_corrigendum_installations",
})
A08_MODEL_BUNDLE_GLOBALS = frozenset({
    "provenance/amendment08_external_authorization.json",
    "provenance/amendment08_code_corrigendum.json",
    "provenance/scientific_execution_code_manifest_effective_amendment08.tsv",
    "provenance/amendment08_failed_attempt_forensics.json",
    "provenance/amendment08_journal_contamination_adjudication.json",
    "provenance/amendment08_resume_precheck.json",
    "provenance/amendment08_overlay_install_receipt.json",
    "provenance/code_snapshot_amendment08/amendment06_full.py",
    A08_RAW_JOURNAL_CUTOFF_RELATIVE,
    "provenance/execution_ledger.json",
})

PROMPT_PATH = Path("/REVIEWER_INPUT_ROOT/06_RUN_NOW_Amendment06_Full_Scientific_Ablation.txt")
PROMPT_SHA256 = "246928379ce2871cf576dcba09a2f8f4551bed1060acb6d5a1d82ab96f126e5b"
AUTHORIZATION = "AMENDMENT_06_EXTERNAL_REVIEW_ACCEPTED_ONE_FULL"
EXTERNAL_REVIEW_DECISION = "ACCEPT_AMENDMENT05_SEALED_SMOKE_FOR_FULL"
RUN_KIND = "FULL_SCIENTIFIC_ABLATION"
RUN_SUFFIX = "_amendment06_full"

ACCEPTED_PREFLIGHT = RESULTS_ROOT / "run_20260813_050405_9970c566_amendment02_preflight"
ACCEPTED_PREFLIGHT_ZIP = Path(str(ACCEPTED_PREFLIGHT) + "_review_bundle.zip")
ACCEPTED_SMOKE = RESULTS_ROOT / "run_20260820_104526_263432_bc48422a_amendment04_cuda_smoke"
ACCEPTED_SMOKE_REVIEW_ZIP = Path(str(ACCEPTED_SMOKE) + "_review_bundle.zip")
ACCEPTED_SMOKE_REVIEW_VERIFICATION = RESULTS_ROOT / f"{ACCEPTED_SMOKE.name}_package_verification.json"
ACCEPTED_SMOKE_MODELS_ZIP = RESULTS_ROOT / f"{ACCEPTED_SMOKE.name}_NON_SCIENTIFIC_smoke_models.zip"
ACCEPTED_SMOKE_MODELS_VERIFICATION = RESULTS_ROOT / f"{ACCEPTED_SMOKE.name}_NON_SCIENTIFIC_smoke_models_package_verification.json"
A05_RECOVERY_VERIFICATION = A06_RUNTIME.parent / "amendment05_seal_only/final/amendment05_recovery_verification.json"
PRE_EDIT_ADMISSION = A06_RUNTIME / "pre_edit/admission_validation.json"
AUTHORIZED_CHANGE_LEDGER = A06_RUNTIME / "authorized_change_ledger.jsonl"
FINAL_PRERUN_EVIDENCE = A06_RUNTIME / "pre_run/final_prerun_evidence.json"
BASELINE_TREE_TSV = A06_RUNTIME / "pre_edit/baseline_tree_sha_size_rel.tsv"

PINNED_FILES: Mapping[Path, str] = {
    PROMPT_PATH: PROMPT_SHA256,
    ACCEPTED_PREFLIGHT_ZIP: "19d6d9ec381bab99ff41a081fd42016c653d4a4ce947f35694e0e32e57ae24cb",
    ACCEPTED_SMOKE_REVIEW_ZIP: "4705906df60b38965d2a4febad5483c7543ab7fa250eb866dcb99b2ebad48c0c",
    ACCEPTED_SMOKE_REVIEW_VERIFICATION: "94e64bcbae08a7d63129e576469dccd4528fc4b3711fdff6904b96a1d3f9b790",
    ACCEPTED_SMOKE_MODELS_ZIP: "e14a3b7cd7cc499384f88b471ad88a5c51eb30e5a0f92e1711b899c72ff9777e",
    ACCEPTED_SMOKE_MODELS_VERIFICATION: "f8e5845a6d8557c33baa4b841f45d89a48311f3a653d28a2663cdf7586126d44",
    A05_RECOVERY_VERIFICATION: "271c4cf4ae36e2e0bb589c45f11767d202f996c5eba512acc59020cd92a4dbc2",
    PRE_EDIT_ADMISSION: "b4281ee4f112c7d781a0ca528eb83189f295768eb2a632d4e00ee25a206bd8d3",
}

PRIMARY_TABLE = Path("/REVIEWER_INPUT_ROOT/IMG_9429 (2025-12-28_11-55-48)/features_train.csv")
PRIMARY_TABLE_SHA256 = "1836d704d15f27728e3ba09d7cddf0aea9458049bc66700dc98f14e7111ec29d"
EXTERNAL_AUDIT = Path("/REVIEWER_INPUT_ROOT/testset_labeled__xgb_recall.csv")
EXTERNAL_AUDIT_SHA256 = "c2be6932a56f13117b18e8ed5750dac5f5317fce77ff1ece41ddf948a0ffcee6"
SPLIT_ASSIGNMENT_SHA256 = "f2cd1570d2c76af49ed52593d95657a3cbbe14d8f2f077edd05447e608698c64"

TRAINING_CONFIGURATION_SHA256 = "3b0103a3d35f76448ddedcab8bb4d1a3d6ba37b675c163a922a437630ea9fcf3"
VARIANT_FEATURE_SETS_SHA256 = "5e2578570064e51a52301ef1cd065ea8c9a42c29a46ce11b2b22571d0c7f3be1"
FEATURE_MANIFEST_SHA256 = "8a4a922b53ddad9bd3d89e3452c8292774b3df2c5d149619c588a40877ca4e84"
FEATURE_DEPENDENCY_GRAPH_SHA256 = "2664c6c9433ffd97e33413c60f1f07e5efbf8ad6614afba3f72eca730cec1836"
ROW_SPLIT_MANIFEST_SHA256 = "8b2b81b2691f00167ef0537049cfd4e63d5a17d0cec52774c1d189a823470c1a"
SOURCE_INVENTORY_SHA256 = "795279d24a3b7e07a65ed0f3a95d87969e19feafd58db182a5d26c7d2b3f762d"
FULL_MODEL_PARAMETERS_SHA256 = "f8b69bce5432d077781427897b73ff2dd7c759c3ca6be9afd5fe042aea0ec14a"
CONTINUOUS_FEATURE_ORDER_SHA256 = "b035ac55a40010f68a8b194f6b03d2e9da03c754198712bb52969f3473802320"

BASE_SEED = 42
PRIMARY_MODEL_SEED = 42
DECISION_THRESHOLD = 0.5
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026081201

OFFICIAL_VARIANTS = (
    "full_new_reference",
    "manual_only",
    "deep_only",
    "no_color",
    "no_shape",
    "no_texture",
    "no_embed_sim",
    "no_review_aware_training_weights",
    "no_safe_smote",
)
EXPLORATORY_VARIANT = "no_deep_pca_features"
TRAINED_VARIANTS = (*OFFICIAL_VARIANTS, EXPLORATORY_VARIANT)
BLOCKED_VARIANT = "no_pca"
VARIANT_FEATURE_CONTRACT: Mapping[str, tuple[int, str]] = {
    "full_new_reference": (93, "a5478eb2e571809d80fbdf0e132d39f301d936b0b711ceb08a326ade0b24ed47"),
    "manual_only": (57, "a8f5cb9445994db38375367993fc54d93a595dc9dd9c8d4dc6518f9d99e7655c"),
    "deep_only": (35, "869d491c431cccf3e5a11ddea608ef8ed55f7534ecd624204c176be9848121ab"),
    "no_color": (65, "4c8131fa99c15ec3f7ed8adbcfe759ac1fb480c22e2694094e8b18506f1fecf9"),
    "no_shape": (78, "39ff51f6192fed1d47d17971fb8974f58b0d12afed587d28ef22a30e70625d54"),
    "no_texture": (86, "a72ba7cbff49608ea8266f61db1c2df7df3e3129383d9343ce074b6aa3372178"),
    "no_embed_sim": (90, "beffdb1f3966ac8a2cea3c9a1ba826d744cb1d9546c117e8b0c9780b42b059bc"),
    "no_review_aware_training_weights": (93, "a5478eb2e571809d80fbdf0e132d39f301d936b0b711ceb08a326ade0b24ed47"),
    "no_safe_smote": (93, "a5478eb2e571809d80fbdf0e132d39f301d936b0b711ceb08a326ade0b24ed47"),
    "no_deep_pca_features": (59, "abd482e8f546145db98362e57d77b27d7af0c8539d90d1f542e6a159aafd96ce"),
}

EXPECTED_STAGE_COUNTS: Mapping[str, tuple[int, int, int]] = {
    "raw_locked_post_split_shuffle": (291024, 215988, 75036),
    "post_mixup": (436536, 323982, 112554),
    "post_dropout": (873072, 647964, 225108),
    "post_jitter_pre_smote": (1309608, 971633, 337975),
    "post_shuffle": (1309608, 971633, 337975),
    "post_safe_smote": (1457449, 971633, 485816),
}
PRE_SMOTE_ROWS = 1_309_608
POST_SMOTE_ROWS = 1_457_449

REQUIRED_METRICS = (
    "average_precision", "roc_auc", "precision", "recall", "f1",
    "tn", "fp", "fn", "tp",
)
OUTPUT_MANIFEST_COLUMNS = (
    "relative_path", "size_bytes", "sha256", "artifact_role",
    "include_in_review_bundle", "include_in_models_bundle",
)
SCIENTIFIC_CODE_RELATIVES = (
    "code/ablation_core.py",
    "code/amendment02_resampling.py",
    "code/amendment03_compat.py",
    "code/amendment03_recovery.py",
    "code/amendment04_recovery.py",
    "code/amendment_inventory.py",
    "code/amendment_core.py",
    "code/amendment06_full.py",
    "code/amendment06_reporting.py",
    "code/evidence_gates.py",
    "code/run_ablation_study.sh",
    "code/run_study.py",
    "docs/PROTOCOL_AMENDMENT_06.md",
)
SCIENTIFIC_TEST_RELATIVES = (
    "tests/test_ablation.py",
    "tests/test_amendment02_resampling.py",
    "tests/test_amendment03.py",
    "tests/test_amendment03_recovery.py",
    "tests/test_amendment04.py",
    "tests/test_amendment06.py",
    "tests/test_amendment06_cli.py",
    "tests/test_amendment06_inventory.py",
    "tests/test_amendment06_reporting.py",
    "tests/test_amendment_inventory.py",
    "tests/test_evidence_gates.py",
)
PACKAGE_CODE_RELATIVES = (
    "code/amendment06_packaging.py",
    "tests/test_amendment06_packaging.py",
)
RUN_PHASES = (
    "FULL_IN_PROGRESS",
    "FULL_MODELS_FROZEN",
    "FULL_EXTERNAL_AUDIT_SCORED",
    "FULL_SCIENTIFIC_COMPLETE",
)


class Amendment06Error(RuntimeError):
    """Base Amendment 06 failure."""


class Amendment06IntegrityError(Amendment06Error):
    """An immutable or completion contract failed."""


class Amendment06ResumeError(Amendment06Error):
    """The only authorized Full Run cannot be resumed safely."""


class StrictJSONError(Amendment06Error):
    """New Full JSON is not strict RFC-8259 JSON."""


class PostFreezeScientificCallError(Amendment06Error):
    """A forbidden scientific-fit API was called after model freeze."""


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _progress(phase: str, phase_status: str, **fields: Any) -> None:
    suffix = " ".join(f"{key.upper()}={value}" for key, value in fields.items())
    print(
        f"A06_PHASE_{phase_status.upper()}={phase}" + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    view = memoryview(array).cast("B")
    digest = hashlib.sha256()
    chunk_size = 8 << 20
    for offset in range(0, len(view), chunk_size):
        digest.update(view[offset:offset + chunk_size])
    return digest.hexdigest()


def _strict_copy(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_strict_copy(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _strict_copy(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _strict_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_copy(item) for item in value]
    raise StrictJSONError(f"Unsupported strict JSON value: {type(value).__name__}")


def strict_full_json_bytes(value: Any) -> bytes:
    copied = _strict_copy(value)
    return (json.dumps(
        copied, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True,
    ) + "\n").encode("utf-8")


def strict_full_canonical_json_bytes(value: Any) -> bytes:
    copied = _strict_copy(value)
    return json.dumps(
        copied, allow_nan=False, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_json_constant(value: str) -> Any:
    raise StrictJSONError(f"Non-standard JSON numeric constant: {value}")


def _reject_duplicate_object_pairs(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def strict_full_loads(value: bytes | str) -> Any:
    try:
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        return json.loads(
            text, parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
    except StrictJSONError:
        raise
    except Exception as exc:
        raise StrictJSONError(f"Strict JSON parse failed: {type(exc).__name__}: {exc}") from exc


def strict_full_load_file(path: Path) -> Any:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise StrictJSONError(f"Strict JSON path is absent or unsafe: {target}")
    return strict_full_loads(target.read_bytes())


def assert_required_metrics_finite(
    metrics: Mapping[str, Any], required: Sequence[str] = REQUIRED_METRICS,
) -> None:
    invalid: list[str] = []
    for name in required:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            invalid.append(name)
            continue
        if not math.isfinite(float(value)):
            invalid.append(name)
    if invalid:
        raise Amendment06IntegrityError(f"Required metrics are missing/nonfinite: {invalid}")


def validate_strict_json_tree(
    root: Path, *, include: Iterable[Path] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    paths = sorted(include if include is not None else root.rglob("*.json"))
    checked: list[dict[str, Any]] = []
    for path in paths:
        target = Path(path)
        strict_full_load_file(target)
        checked.append({
            "relative_path": target.relative_to(root).as_posix(),
            "size_bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        })
    return {"status": "PASS", "strict_rfc8259": True, "file_count": len(checked), "files": checked}


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def atomic_write_strict_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, strict_full_json_bytes(value))


def publish_bytes_no_clobber(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(os, "O_TMPFILE"):
        raise Amendment06IntegrityError("Linux O_TMPFILE is required for atomic publication")
    descriptor = os.open(path.parent, os.O_WRONLY | os.O_TMPFILE, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        libc = ctypes.CDLL(None, use_errno=True)
        linkat = getattr(libc, "linkat", None)
        if linkat is None:
            raise Amendment06IntegrityError("linkat is required for atomic publication")
        linkat.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_int,
        ]
        linkat.restype = ctypes.c_int
        result = linkat(
            descriptor, b"", -100, os.fsencode(path), 0x1000,
        )
        if result != 0 and ctypes.get_errno() in {errno.ENOENT, errno.EPERM}:
            # Some kernels deny AT_EMPTY_PATH without CAP_DAC_READ_SEARCH.  The
            # documented /proc/self/fd form links the same anonymous inode.
            result = linkat(
                -100, os.fsencode(f"/proc/self/fd/{descriptor}"),
                -100, os.fsencode(path), 0x400,
            )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(f"No-clobber file publication collision: {path}")
            raise OSError(error, os.strerror(error), str(path))
        _fsync_parent(path)
    finally:
        os.close(descriptor)


def publish_strict_json_no_clobber(path: Path, value: Any) -> None:
    publish_bytes_no_clobber(path, strict_full_json_bytes(value))


def _rename_directory_no_clobber(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise Amendment06IntegrityError("renameat2 is required for no-clobber Run publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(source), -100, os.fsencode(destination), 1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"No-clobber directory publication collision: {destination}")
        raise OSError(error, os.strerror(error), str(destination))
    _fsync_parent(destination)


def _safe_regular(path: Path, expected_sha256: str | None = None) -> bool:
    return bool(
        path.is_file() and not path.is_symlink()
        and (expected_sha256 is None or sha256_file(path) == expected_sha256)
    )


def _load_concatenated_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder(
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_object_pairs,
    )
    values: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        value, index = decoder.raw_decode(text, index)
        if not isinstance(value, dict):
            raise Amendment06IntegrityError("Authorized-change ledger entry is not an object")
        values.append(value)
    return values


def live_code_test_mapping(
    study_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    if study_root is None:
        study_root = STUDY_ROOT
    paths = sorted([
        *[path for path in (study_root / "code").glob("*.py") if path.is_file()],
        *[path for path in (study_root / "code").glob("*.sh") if path.is_file()],
        *[path for path in (study_root / "tests").glob("*.py") if path.is_file()],
    ], key=lambda path: path.relative_to(study_root).as_posix())
    if any(path.is_symlink() for path in paths):
        raise Amendment06IntegrityError("Live code/test tree contains a symlink")
    return {
        path.relative_to(study_root).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    }


def canonical_code_tree_sha256(mapping: Mapping[str, Mapping[str, Any]]) -> str:
    data = "".join(
        f"{mapping[relative]['sha256']}\t{int(mapping[relative]['size_bytes'])}\t{relative}\n"
        for relative in sorted(mapping)
    ).encode("utf-8")
    return sha256_bytes(data)


def _validated_tree_mapping(value: Any, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise Amendment06IntegrityError(f"{label} is not a nonempty tree map")
    result: dict[str, dict[str, Any]] = {}
    for relative, item in value.items():
        relative = str(relative)
        path = Path(relative)
        if (
            not relative or path.is_absolute() or ".." in path.parts
            or path.as_posix() != relative or not isinstance(item, Mapping)
        ):
            raise Amendment06IntegrityError(f"{label} has an unsafe tree record: {relative!r}")
        digest = str(item.get("sha256", ""))
        size = item.get("size_bytes")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise Amendment06IntegrityError(f"{label} has an invalid tree record: {relative}")
        result[relative] = {"size_bytes": size, "sha256": digest}
    return dict(sorted(result.items()))


def _baseline_tree_mapping() -> dict[str, dict[str, Any]]:
    if not _safe_regular(BASELINE_TREE_TSV):
        raise Amendment06IntegrityError("Canonical Amendment 06 baseline tree TSV is absent")
    frame = pd.read_csv(BASELINE_TREE_TSV, sep="\t", dtype=str, keep_default_na=False)
    if list(frame.columns) != ["sha256", "size_bytes", "relative_path"] or frame.relative_path.duplicated().any():
        raise Amendment06IntegrityError("Canonical Amendment 06 baseline tree TSV schema differs")
    mapping = {
        str(row.relative_path): {
            "size_bytes": int(row.size_bytes), "sha256": str(row.sha256),
        }
        for row in frame.itertuples(index=False)
    }
    validated = _validated_tree_mapping(mapping, label="baseline tree")
    if canonical_code_tree_sha256(validated) != "55992443799c7623911b2a908e79ff9c12e25e7ac9da15c2215ee1b01eac5075":
        raise Amendment06IntegrityError("Canonical Amendment 06 baseline tree hash differs")
    return validated


def _tree_summary(mapping: Mapping[str, Mapping[str, Any]]) -> tuple[str, int, int]:
    return (
        canonical_code_tree_sha256(mapping), len(mapping),
        sum(int(item["size_bytes"]) for item in mapping.values()),
    )


def _changed_tree_records(
    before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in sorted(set(before) | set(after)):
        old = before.get(relative)
        new = after.get(relative)
        if old == new:
            continue
        records.append({
            "relative_path": relative,
            "before_size_bytes": None if old is None else int(old["size_bytes"]),
            "before_sha256": None if old is None else str(old["sha256"]),
            "after_size_bytes": None if new is None else int(new["size_bytes"]),
            "after_sha256": None if new is None else str(new["sha256"]),
        })
    return records


def _validate_test_evidence_records(
    value: Any, *, label: str, require_live: bool = True,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise Amendment06IntegrityError(f"{label} test evidence is empty")
    records: list[dict[str, Any]] = []
    required = {
        "category", "status", "command", "exit_code", "log_path",
        "log_size_bytes", "log_sha256",
    }
    seen_categories: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or not required.issubset(item):
            raise Amendment06IntegrityError(f"{label} test evidence schema differs")
        category = item.get("category")
        log_path = item.get("log_path")
        path = Path(log_path) if isinstance(log_path, str) else Path()
        try:
            inside_runtime = (
                path.is_absolute()
                and A06_RUNTIME.resolve() in path.resolve().parents
            )
        except FileNotFoundError:
            inside_runtime = False
        sha256 = str(item.get("log_sha256", ""))
        size_bytes = item.get("log_size_bytes")
        record_valid = (
            item.get("status") == "PASS"
            and isinstance(item.get("exit_code"), int)
            and not isinstance(item.get("exit_code"), bool)
            and item.get("exit_code") == 0
            and isinstance(category, str) and category
            and category not in seen_categories
            and isinstance(item.get("command"), str) and item["command"]
            and inside_runtime
            and isinstance(size_bytes, int) and not isinstance(size_bytes, bool)
            and size_bytes >= 0
            and re.fullmatch(r"[0-9a-f]{64}", sha256) is not None
            and ("facts" not in item or isinstance(item.get("facts"), Mapping))
        )
        live_valid = (
            not require_live
            or (
                _safe_regular(path, sha256)
                and path.stat().st_size == size_bytes
            )
        )
        if not (record_valid and live_valid):
            raise Amendment06IntegrityError(f"{label} test log binding differs: {path}")
        seen_categories.add(category)
        records.append(dict(item))
    return records


def make_test_evidence_record(
    *, category: str, command: str, log_path: Path,
    facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(log_path).resolve()
    if not _safe_regular(path):
        raise Amendment06IntegrityError(f"Test-evidence log is absent/unsafe: {path}")
    record: dict[str, Any] = {
        "category": category, "status": "PASS", "command": command,
        "exit_code": 0, "log_path": str(path),
        "log_size_bytes": path.stat().st_size, "log_sha256": sha256_file(path),
    }
    if facts is not None:
        record["facts"] = dict(facts)
    _validate_test_evidence_records([record], label=f"generated {category}")
    return record


def build_authorized_edit_event(
    *, purpose: str, test_evidence: Sequence[Mapping[str, Any]],
    study_root: Path | None = None, ledger_path: Path | None = None,
    final_byte_freeze: bool = True, recorded_at: str | None = None,
) -> dict[str, Any]:
    if study_root is None:
        study_root = STUDY_ROOT
    if ledger_path is None:
        ledger_path = AUTHORIZED_CHANGE_LEDGER
    events = _load_concatenated_json(ledger_path)
    if len(events) < 2:
        raise Amendment06IntegrityError("Cannot extend an incomplete Amendment 06 ledger")
    for expected_index, event in enumerate(events):
        if event.get("event_index") != expected_index or (
            expected_index and event.get("previous_event_index") != expected_index - 1
        ):
            raise Amendment06IntegrityError("Cannot extend a noncontiguous Amendment 06 ledger")
    before = (
        _baseline_tree_mapping()
        if len(events) == 2
        else _validated_tree_mapping(events[-1].get("after_tree"), label="latest ledger after-tree")
    )
    after = live_code_test_mapping(study_root)
    changed = _changed_tree_records(before, after)
    if not changed:
        raise Amendment06IntegrityError("Final authorized edit event contains no changed path")
    evidence = [dict(item) for item in test_evidence]
    _validate_test_evidence_records(evidence, label="generated authorized edit")
    before_sha, before_count, before_bytes = _tree_summary(before)
    after_sha, after_count, after_bytes = _tree_summary(after)
    return {
        "event": "AUTHORIZED_EDIT_SET_COMMITTED",
        "event_index": len(events),
        "previous_event_index": len(events) - 1,
        "previous_event_sha256": sha256_bytes(
            strict_full_canonical_json_bytes(events[-1])
        ),
        "authorization": AUTHORIZATION, "prompt_sha256": PROMPT_SHA256,
        "recorded_at": recorded_at or _now(), "purpose": purpose,
        "scope": [item["relative_path"] for item in changed],
        "before_tree": before, "after_tree": after,
        "before_tree_sha256": before_sha, "after_tree_sha256": after_sha,
        "before_file_count": before_count, "after_file_count": after_count,
        "before_total_bytes": before_bytes, "after_total_bytes": after_bytes,
        "changed_paths": changed,
        "diff_sha256": sha256_bytes(strict_full_canonical_json_bytes(changed)),
        "test_evidence": evidence, "final_byte_freeze": final_byte_freeze,
    }


def append_authorized_edit_event(
    *, purpose: str, test_evidence: Sequence[Mapping[str, Any]],
    study_root: Path | None = None, ledger_path: Path | None = None,
    final_byte_freeze: bool = True, recorded_at: str | None = None,
) -> dict[str, Any]:
    if study_root is None:
        study_root = STUDY_ROOT
    if ledger_path is None:
        ledger_path = AUTHORIZED_CHANGE_LEDGER
    ledger_path = Path(ledger_path)
    if not _safe_regular(ledger_path):
        raise Amendment06IntegrityError("Authorized-change ledger is absent/unsafe")
    with ledger_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        event = build_authorized_edit_event(
            purpose=purpose, test_evidence=test_evidence,
            study_root=study_root, ledger_path=ledger_path,
            final_byte_freeze=final_byte_freeze, recorded_at=recorded_at,
        )
        handle.seek(0, os.SEEK_END)
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            trailing_newline = handle.read(1) == b"\n"
            handle.seek(0, os.SEEK_END)
            if not trailing_newline:
                handle.write(b"\n")
        handle.write(strict_full_canonical_json_bytes(event) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return event


def _validate_amendment07_ledger_event(
    event: Mapping[str, Any], *, require_latest_tree: bool,
) -> None:
    scope = event.get("scope")
    if not isinstance(scope, list) or not all(isinstance(item, str) for item in scope):
        raise Amendment06IntegrityError("Amendment 07 ledger scope differs")
    allowed = {
        *A07_CHANGED_SCIENTIFIC_RELATIVES,
        "code/amendment06_packaging.py",
        "tests/test_amendment06_packaging.py",
    }
    if not (
        event.get("corrigendum_authorization") == A07_AUTHORIZATION
        and event.get("corrigendum_prompt_sha256") == A07_PROMPT_SHA256
        and A07_AUTHORIZATION in str(event.get("purpose", ""))
        and A07_PROMPT_SHA256 in str(event.get("purpose", ""))
        and set(A07_CHANGED_SCIENTIFIC_RELATIVES).issubset(scope)
        and set(scope).issubset(allowed)
        and event.get("corrigendum_scientific_scope")
        == list(A07_CHANGED_SCIENTIFIC_RELATIVES)
        and event.get("conditional_package_scope")
        == [item for item in scope if item not in A07_CHANGED_SCIENTIFIC_RELATIVES]
        and event.get("final_byte_freeze") is True
    ):
        raise Amendment06IntegrityError("Amendment 07 authorized ledger event differs")
    if require_latest_tree:
        mapping = live_code_test_mapping()
        after = _validated_tree_mapping(
            event.get("after_tree"), label="Amendment 07 ledger after-tree",
        )
        if after != mapping:
            raise Amendment06IntegrityError("Amendment 07 latest ledger tree drifted")


def _validate_amendment08_ledger_event(
    event: Mapping[str, Any], *, require_latest_tree: bool,
) -> None:
    extra_keys = {
        "corrigendum_authorization", "corrigendum_prompt_sha256",
        "corrigendum_scientific_scope", "corrigendum_package_scope",
        "pre_a08_source_hashes", "source_freeze_transaction_id",
    }
    standard_keys = {
        "event", "event_index", "previous_event_index", "previous_event_sha256",
        "authorization", "prompt_sha256", "recorded_at", "purpose", "scope",
        "before_tree", "after_tree", "before_tree_sha256", "after_tree_sha256",
        "before_file_count", "after_file_count", "before_total_bytes",
        "after_total_bytes", "changed_paths", "diff_sha256", "test_evidence",
        "final_byte_freeze",
    }
    scope = event.get("scope")
    if not (
        set(event) == standard_keys | extra_keys
        and event.get("event_index") == 5
        and event.get("authorization") == AUTHORIZATION
        and event.get("prompt_sha256") == PROMPT_SHA256
        and event.get("corrigendum_authorization") == A08_AUTHORIZATION
        and event.get("corrigendum_prompt_sha256") == A08_PROMPT_SHA256
        and event.get("corrigendum_scientific_scope")
        == list(A08_CHANGED_SCIENTIFIC_RELATIVES)
        and event.get("corrigendum_package_scope")
        == list(A08_CHANGED_PACKAGE_RELATIVES)
        and scope == list(A08_CHANGED_ALL_RELATIVES)
        and event.get("pre_a08_source_hashes") == dict(A08_PRE_SOURCE_HASHES)
        and re.fullmatch(
            r"[0-9a-f]{32}", str(event.get("source_freeze_transaction_id", ""))
        )
        and event.get("final_byte_freeze") is True
        and A08_AUTHORIZATION in str(event.get("purpose", ""))
        and A08_PROMPT_SHA256 in str(event.get("purpose", ""))
    ):
        raise Amendment06IntegrityError("Amendment 08 authorized ledger event differs")
    checks = _validate_test_evidence_records(
        event.get("test_evidence"), label="Amendment 08 ledger event",
        require_live=require_latest_tree,
    )
    if tuple(item["category"] for item in checks) != PRERUN_CHECK_CATEGORIES:
        raise Amendment06IntegrityError("Amendment 08 source-freeze check order differs")
    for record in checks:
        path = Path(str(record["log_path"])).resolve()
        if A08_A06_BOUND_RUNTIME.resolve() not in path.parents:
            raise Amendment06IntegrityError(
                "Amendment 08 source-freeze log is outside A06-bound runtime"
            )
    if require_latest_tree:
        after = _validated_tree_mapping(
            event.get("after_tree"), label="Amendment 08 ledger after-tree",
        )
        if after != live_code_test_mapping():
            raise Amendment06IntegrityError("Amendment 08 latest ledger tree drifted")


def append_amendment07_authorized_edit_event(
    *, test_evidence: Sequence[Mapping[str, Any]],
    recorded_at: str | None = None,
) -> dict[str, Any]:
    requested_evidence = _validate_test_evidence_records(
        [dict(item) for item in test_evidence],
        label="requested Amendment 07 ledger event",
    )
    validate_amendment07_precorrection()
    if not _safe_regular(AUTHORIZED_CHANGE_LEDGER):
        raise Amendment06IntegrityError("Authorized-change ledger is absent/unsafe")
    purpose = (
        f"{A07_AUTHORIZATION}:{A07_PROMPT_SHA256}:"
        "GPU_IDENTITY_CORRIGENDUM_AND_CONDITIONAL_PACKAGE_ALIGNMENT"
    )
    append_intent_path = (
        A07_RUNTIME / "transactions/source_ledger_append_intent.json"
    )
    # Lock a stable inode outside the ledger.  The ledger itself is atomically
    # replaced, so locking its old inode would not serialize a concurrent retry.
    with amendment06_lock(A06_RUNTIME):
        existing = _load_concatenated_json(AUTHORIZED_CHANGE_LEDGER)
        matches = [
            event for event in existing
            if event.get("corrigendum_authorization") == A07_AUTHORIZATION
        ]
        if matches:
            if len(matches) != 1 or matches[0] is not existing[-1]:
                raise Amendment06IntegrityError("Amendment 07 ledger event cardinality differs")
            _validate_amendment07_ledger_event(matches[0], require_latest_tree=True)
            observed_evidence = _validate_test_evidence_records(
                matches[0].get("test_evidence"),
                label="installed Amendment 07 ledger event",
            )
            if observed_evidence != requested_evidence:
                raise Amendment06IntegrityError(
                    "Amendment 07 ledger retry requested different test evidence"
                )
            return dict(matches[0])
        intent: Mapping[str, Any] | None = None
        if append_intent_path.exists():
            intent = _require_exact_keys(
                strict_full_load_file(append_intent_path),
                {"schema", "status", "authorization", "prompt_sha256", "event"},
                label="Amendment 07 source-ledger append intent",
            )
            if not (
                intent["schema"] == "amendment07_source_ledger_append_intent/v1"
                and intent["status"] == "INTENT"
                and intent["authorization"] == A07_AUTHORIZATION
                and intent["prompt_sha256"] == A07_PROMPT_SHA256
                and isinstance(intent["event"], Mapping)
            ):
                raise Amendment06IntegrityError("Amendment 07 ledger append intent differs")
            recorded_at = str(intent["event"].get("recorded_at", ""))
        event = build_authorized_edit_event(
            purpose=purpose, test_evidence=requested_evidence,
            final_byte_freeze=True, recorded_at=recorded_at,
        )
        scope = [str(item) for item in event["scope"]]
        event["corrigendum_authorization"] = A07_AUTHORIZATION
        event["corrigendum_prompt_sha256"] = A07_PROMPT_SHA256
        event["corrigendum_scientific_scope"] = list(
            A07_CHANGED_SCIENTIFIC_RELATIVES
        )
        event["conditional_package_scope"] = [
            item for item in scope if item not in A07_CHANGED_SCIENTIFIC_RELATIVES
        ]
        _validate_amendment07_ledger_event(event, require_latest_tree=True)
        expected_intent = {
            "schema": "amendment07_source_ledger_append_intent/v1",
            "status": "INTENT", "authorization": A07_AUTHORIZATION,
            "prompt_sha256": A07_PROMPT_SHA256, "event": event,
        }
        if intent is not None:
            if dict(intent) != expected_intent:
                raise Amendment06IntegrityError("Amendment 07 ledger append intent/event drifted")
        else:
            publish_strict_json_no_clobber(append_intent_path, expected_intent)
        before_bytes = AUTHORIZED_CHANGE_LEDGER.read_bytes()
        separator = b"" if before_bytes.endswith(b"\n") else b"\n"
        after_bytes = (
            before_bytes + separator
            + strict_full_canonical_json_bytes(event) + b"\n"
        )
        staging = (
            A07_RUNTIME / "transactions/source_ledger_append"
            / f"{sha256_bytes(after_bytes)}.jsonl"
        )
        if staging.exists():
            if not _safe_regular(staging, sha256_bytes(after_bytes)):
                raise Amendment06IntegrityError("A07 staged source-ledger bytes drifted")
        else:
            publish_bytes_no_clobber(staging, after_bytes)
        if staging.stat().st_dev != AUTHORIZED_CHANGE_LEDGER.stat().st_dev:
            raise Amendment06IntegrityError("A07 source-ledger append crosses filesystems")
        if AUTHORIZED_CHANGE_LEDGER.read_bytes() != before_bytes:
            raise Amendment06IntegrityError("A07 source ledger changed during append")
        os.replace(staging, AUTHORIZED_CHANGE_LEDGER)
        _fsync_parent(AUTHORIZED_CHANGE_LEDGER)
        validate_authorized_change_chain()
    return event


def validate_authorized_change_chain(
    *, study_root: Path | None = None, ledger_path: Path | None = None,
) -> dict[str, Any]:
    if study_root is None:
        study_root = STUDY_ROOT
    if ledger_path is None:
        ledger_path = AUTHORIZED_CHANGE_LEDGER
    if not _safe_regular(ledger_path):
        raise Amendment06IntegrityError("Amendment 06 authorized-change ledger is absent")
    events = _load_concatenated_json(ledger_path)
    if len(events) < 2:
        raise Amendment06IntegrityError("Authorized-change ledger has no committed genesis")
    genesis, correction = events[:2]
    if not (
        genesis.get("event_index") == 0
        and genesis.get("event") == "PRE_EDIT_BASELINE_COMMITTED"
        and genesis.get("authorization") == AUTHORIZATION
        and genesis.get("prompt_sha256") == PROMPT_SHA256
        and correction.get("event_index") == 1
        and correction.get("previous_event_index") == 0
        and correction.get("event") == "PRE_EDIT_BASELINE_CANONICAL_ORDER_COMMITTED"
        and correction.get("canonical_tree_sha256")
        == "55992443799c7623911b2a908e79ff9c12e25e7ac9da15c2215ee1b01eac5075"
    ):
        raise Amendment06IntegrityError("Authorized-change ledger genesis differs from committed admission")
    for expected_index, event in enumerate(events):
        if event.get("event_index") != expected_index:
            raise Amendment06IntegrityError("Authorized-change ledger indices are not contiguous")
        if expected_index and event.get("previous_event_index") != expected_index - 1:
            raise Amendment06IntegrityError("Authorized-change ledger chain pointer is invalid")

    previous_tree = _baseline_tree_mapping()
    if len(events) < 3:
        raise Amendment06IntegrityError("Authorized-change ledger has no final committed edit event")
    for expected_index, event in enumerate(events[2:], start=2):
        before = _validated_tree_mapping(event.get("before_tree"), label=f"ledger event {expected_index} before")
        after = _validated_tree_mapping(event.get("after_tree"), label=f"ledger event {expected_index} after")
        before_sha, before_count, before_bytes = _tree_summary(before)
        after_sha, after_count, after_bytes = _tree_summary(after)
        changed = _changed_tree_records(before, after)
        if not changed:
            raise Amendment06IntegrityError(f"Ledger edit event {expected_index} contains no change")
        previous_event_sha = sha256_bytes(strict_full_canonical_json_bytes(events[expected_index - 1]))
        scope = [str(value) for value in event.get("scope", [])] if isinstance(event.get("scope"), list) else []
        if not (
            event.get("event") == "AUTHORIZED_EDIT_SET_COMMITTED"
            and event.get("authorization") == AUTHORIZATION
            and event.get("prompt_sha256") == PROMPT_SHA256
            and event.get("previous_event_sha256") == previous_event_sha
            and isinstance(event.get("purpose"), str) and event["purpose"].strip()
            and before == previous_tree
            and event.get("before_tree_sha256") == before_sha
            and event.get("after_tree_sha256") == after_sha
            and event.get("before_file_count") == before_count
            and event.get("after_file_count") == after_count
            and event.get("before_total_bytes") == before_bytes
            and event.get("after_total_bytes") == after_bytes
            and event.get("changed_paths") == changed
            and scope == [item["relative_path"] for item in changed]
            and event.get("diff_sha256")
            == sha256_bytes(strict_full_canonical_json_bytes(changed))
            and isinstance(event.get("final_byte_freeze"), bool)
        ):
            raise Amendment06IntegrityError(f"Authorized-change ledger event {expected_index} is incomplete")
        _validate_test_evidence_records(
            event.get("test_evidence"),
            label=f"ledger event {expected_index}",
            require_live=expected_index == len(events) - 1,
        )
        if "corrigendum_authorization" in event:
            if event.get("corrigendum_authorization") == A07_AUTHORIZATION:
                _validate_amendment07_ledger_event(
                    event, require_latest_tree=expected_index == len(events) - 1,
                )
            elif event.get("corrigendum_authorization") == A08_AUTHORIZATION:
                _validate_amendment08_ledger_event(
                    event, require_latest_tree=expected_index == len(events) - 1,
                )
            else:
                raise Amendment06IntegrityError(
                    f"Ledger event {expected_index} has unknown corrigendum authority"
                )
        previous_tree = after
    if events[-1].get("final_byte_freeze") is not True:
        raise Amendment06IntegrityError("Latest authorized-change event is not the final byte freeze")

    mapping = live_code_test_mapping(study_root)
    live_tree_sha256 = canonical_code_tree_sha256(mapping)
    total_bytes = sum(int(item["size_bytes"]) for item in mapping.values())
    latest = events[-1]
    committed_sha = str(
        latest.get("post_edit_tree_sha256")
        or latest.get("canonical_tree_sha256")
        or latest.get("after_tree_sha256")
        or ""
    )
    if committed_sha != live_tree_sha256:
        raise Amendment06IntegrityError(
            "Live code/tests do not match the latest committed Amendment 06 tree: "
            f"committed={committed_sha!r}, live={live_tree_sha256}"
        )
    if previous_tree != mapping:
        raise Amendment06IntegrityError("Latest ledger after-tree differs from live code/tests")
    allowed_after_hashes = {path: item["sha256"] for path, item in mapping.items()}
    ledger_evidence = {
        "status": "PASS",
        "authorization": AUTHORIZATION,
        "chain_status": "PASS",
        "baseline_tree_sha256": "55992443799c7623911b2a908e79ff9c12e25e7ac9da15c2215ee1b01eac5075",
        "latest_committed_tree_sha256": live_tree_sha256,
        "latest_committed_file_count": len(mapping),
        "latest_committed_total_bytes": total_bytes,
        "allowed_after_hashes_sha256": sha256_bytes(
            core.canonical_json(dict(sorted(allowed_after_hashes.items())))
        ),
        "authorized_change_ledger_path": str(Path(ledger_path).resolve()),
        "authorized_change_ledger_sha256": sha256_file(ledger_path),
    }
    return {
        "status": "PASS", "event_count": len(events),
        "latest_event_index": len(events) - 1,
        "canonical_tree_sha256": live_tree_sha256,
        "mapping": mapping,
        "allowed_after_hashes": allowed_after_hashes,
        "change_ledger_evidence": ledger_evidence,
        "ledger_sha256": sha256_file(ledger_path),
        "latest_event_sha256": sha256_bytes(strict_full_canonical_json_bytes(latest)),
    }


PRERUN_CHECK_CATEGORIES = (
    "syntax_compile",
    "targeted_tests",
    "full_suite_exactly_once",
    "active_cuda_probe",
    "package_synthetic_rehearsal",
    "resume_rehearsal",
)


def _validated_final_prerun_checks(value: Any) -> list[dict[str, Any]]:
    checks = _validate_test_evidence_records(value, label="final pre-Run")
    if tuple(str(item["category"]) for item in checks) != PRERUN_CHECK_CATEGORIES:
        raise Amendment06IntegrityError("Final pre-Run check category/order differs")
    facts = {str(item["category"]): item.get("facts") for item in checks}
    if not all(isinstance(item, Mapping) for item in facts.values()) or not (
        facts["syntax_compile"].get("all_changed_python_files_compiled") is True
        and facts["targeted_tests"].get("amendment06_targeted_tests_passed") is True
        and facts["full_suite_exactly_once"].get("full_project_suite_passed") is True
        and facts["full_suite_exactly_once"].get("suite_invocation_count") == 1
        and facts["active_cuda_probe"].get("xgboost_version") == "2.1.1"
        and facts["active_cuda_probe"].get("scikit_learn_version") == "1.7.2"
        and facts["active_cuda_probe"].get("use_cuda") is True
        and "4090" in str(facts["active_cuda_probe"].get("gpu_name", ""))
        and facts["active_cuda_probe"].get("cpu_fallback_detected") is False
        and facts["active_cuda_probe"].get("fit_save_reload_predict_status") == "PASS"
        and facts["package_synthetic_rehearsal"].get("synthetic_full_fixture") is True
        and facts["package_synthetic_rehearsal"].get("production_packager_used") is True
        and facts["package_synthetic_rehearsal"].get("production_final_validator_used") is True
        and facts["package_synthetic_rehearsal"].get("package_retry_without_training") is True
        and facts["package_synthetic_rehearsal"].get("scientific_api_calls") == 0
        and facts["resume_rehearsal"].get("same_run_id_preserved") is True
        and facts["resume_rehearsal"].get("completed_variant_skipped") is True
        and facts["resume_rehearsal"].get("unpublished_incomplete_variant_retried") is True
        and facts["resume_rehearsal"].get("completed_variant_retrained") is False
        and facts["resume_rehearsal"].get("distinct_full_run_ids") == 1
    ):
        raise Amendment06IntegrityError("A mandatory final pre-Run rehearsal fact failed")
    return checks


def build_final_prerun_evidence(
    *, chain: Mapping[str, Any], checks: Sequence[Mapping[str, Any]],
    generated_at: str | None = None,
) -> dict[str, Any]:
    checked = _validated_final_prerun_checks([dict(item) for item in checks])
    mapping = chain["mapping"]
    return {
        "status": "PASS", "authorization": AUTHORIZATION,
        "prompt_sha256": PROMPT_SHA256, "generated_at": generated_at or _now(),
        "live_code_tree_sha256": chain["canonical_tree_sha256"],
        "live_code_tree_file_count": len(mapping),
        "live_code_tree_total_bytes": sum(
            int(item["size_bytes"]) for item in mapping.values()
        ),
        "authorized_change_ledger_path": str(AUTHORIZED_CHANGE_LEDGER.resolve()),
        "authorized_change_ledger_sha256": chain["ledger_sha256"],
        "latest_event_index": chain["latest_event_index"],
        "latest_event_sha256": chain["latest_event_sha256"],
        "full_suite_invocation_count": 1, "checks": checked,
    }


def write_final_prerun_evidence(
    *, checks: Sequence[Mapping[str, Any]], path: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if path is None:
        path = FINAL_PRERUN_EVIDENCE
    chain = validate_authorized_change_chain()
    payload = build_final_prerun_evidence(
        chain=chain, checks=checks, generated_at=generated_at,
    )
    data = strict_full_json_bytes(payload)
    path = Path(path)
    if path.exists():
        if not _safe_regular(path, sha256_bytes(data)):
            raise Amendment06IntegrityError("Final pre-Run evidence already differs")
    else:
        publish_bytes_no_clobber(path, data)
    return validate_final_prerun_evidence(chain, path=path)


def _atomic_replace_expected_bytes(
    path: Path, data: bytes, *, expected_old_sha256: str,
) -> None:
    path = Path(path)
    if not _safe_regular(path, expected_old_sha256):
        raise Amendment06IntegrityError(
            f"Atomic authorized replacement old-byte pin differs: {path}"
        )
    staging_root = A07_RUNTIME / "transactions/final_prerun_replacement"
    staging_root.mkdir(parents=True, exist_ok=True)
    staged = staging_root / f"{sha256_bytes(data)}.json"
    if staged.exists():
        if not _safe_regular(staged, sha256_bytes(data)):
            raise Amendment06IntegrityError("A07 final pre-Run replacement staging drifted")
    else:
        publish_bytes_no_clobber(staged, data)
    if staged.stat().st_dev != path.stat().st_dev:
        raise Amendment06IntegrityError("A07 final pre-Run replacement crosses filesystems")
    if not _safe_regular(path, expected_old_sha256):
        raise Amendment06IntegrityError("A07 final pre-Run evidence changed during replacement")
    os.replace(staged, path)
    _fsync_parent(path)


def replace_amendment07_final_prerun_evidence(
    *, checks: Sequence[Mapping[str, Any]], generated_at: str | None = None,
) -> dict[str, Any]:
    chain = validate_authorized_change_chain()
    events = _load_concatenated_json(AUTHORIZED_CHANGE_LEDGER)
    if not events or events[-1].get("corrigendum_authorization") != A07_AUTHORIZATION:
        raise Amendment06IntegrityError("Latest source-ledger event is not Amendment 07")
    _validate_amendment07_ledger_event(events[-1], require_latest_tree=True)
    requested_checks = _validated_final_prerun_checks(
        [dict(item) for item in checks]
    )
    ledger_checks = _validated_final_prerun_checks(
        events[-1].get("test_evidence")
    )
    if requested_checks != ledger_checks:
        raise Amendment06IntegrityError(
            "Amendment 07 final pre-Run checks differ from the latest ledger event"
        )
    for record in requested_checks:
        path = Path(str(record.get("log_path", "")))
        try:
            inside = A07_A06_BOUND_RUNTIME.resolve() in path.resolve().parents
        except FileNotFoundError:
            inside = False
        if not inside:
            raise Amendment06IntegrityError(
                f"Amendment 07 final log is outside the A06-bound runtime: {path}"
            )
    payload = build_final_prerun_evidence(
        chain=chain, checks=requested_checks, generated_at=generated_at,
    )
    data = strict_full_json_bytes(payload)
    if _safe_regular(FINAL_PRERUN_EVIDENCE, sha256_bytes(data)):
        return validate_final_prerun_evidence(chain)
    if _safe_regular(FINAL_PRERUN_EVIDENCE, A07_ORIGINAL_PRERUN_SHA256):
        forensic = _validate_amendment07_forensic_snapshot()
        if forensic["original_prerun_forensic_sha256"] != A07_ORIGINAL_PRERUN_SHA256:
            raise Amendment06IntegrityError("Original A06 pre-Run evidence is not preserved")
        _atomic_replace_expected_bytes(
            FINAL_PRERUN_EVIDENCE, data,
            expected_old_sha256=A07_ORIGINAL_PRERUN_SHA256,
        )
    else:
        # A completed replacement is reusable only if it validates the exact
        # current ledger tip and six live logs.
        return validate_final_prerun_evidence(chain)
    return validate_final_prerun_evidence(chain)


def validate_final_prerun_evidence(
    chain: Mapping[str, Any], *, path: Path | None = None,
) -> dict[str, Any]:
    if path is None:
        path = FINAL_PRERUN_EVIDENCE
    evidence = strict_full_load_file(path)
    required_top = {
        "status", "authorization", "prompt_sha256", "generated_at",
        "live_code_tree_sha256", "live_code_tree_file_count", "live_code_tree_total_bytes",
        "authorized_change_ledger_path", "authorized_change_ledger_sha256",
        "latest_event_index", "latest_event_sha256", "full_suite_invocation_count",
        "checks",
    }
    mapping = chain["mapping"]
    if not (
        isinstance(evidence, Mapping) and set(evidence) == required_top
        and evidence.get("status") == "PASS"
        and evidence.get("authorization") == AUTHORIZATION
        and evidence.get("prompt_sha256") == PROMPT_SHA256
        and isinstance(evidence.get("generated_at"), str) and evidence["generated_at"]
        and evidence.get("live_code_tree_sha256") == chain["canonical_tree_sha256"]
        and evidence.get("live_code_tree_file_count") == len(mapping)
        and evidence.get("live_code_tree_total_bytes")
        == sum(int(item["size_bytes"]) for item in mapping.values())
        and Path(str(evidence.get("authorized_change_ledger_path"))).resolve()
        == AUTHORIZED_CHANGE_LEDGER.resolve()
        and evidence.get("authorized_change_ledger_sha256") == chain["ledger_sha256"]
        and evidence.get("latest_event_index") == chain["latest_event_index"]
        and evidence.get("latest_event_sha256") == chain["latest_event_sha256"]
        and evidence.get("full_suite_invocation_count") == 1
    ):
        raise Amendment06IntegrityError("Final pre-Run evidence does not bind the frozen code/ledger")
    checked = _validated_final_prerun_checks(evidence.get("checks"))
    events = _load_concatenated_json(AUTHORIZED_CHANGE_LEDGER)
    if events and events[-1].get("corrigendum_authorization") in {
        A07_AUTHORIZATION, A08_AUTHORIZATION,
    }:
        ledger_checks = _validated_final_prerun_checks(
            events[-1].get("test_evidence")
        )
        if checked != ledger_checks:
            raise Amendment06IntegrityError(
                "Corrigendum final pre-Run checks differ from the latest ledger event"
            )
    return {
        "status": "PASS", "path": str(path), "sha256": sha256_file(path),
        "checks": list(PRERUN_CHECK_CATEGORIES), "full_suite_invocation_count": 1,
    }


def _replace_exact_file_from_a08_staging(
    *, path: Path, desired: bytes, allowed_old_sha256: str, label: str,
) -> None:
    path = Path(path)
    desired_sha256 = sha256_bytes(desired)
    if _safe_regular(path, desired_sha256) and path.stat().st_size == len(desired):
        return
    if not _safe_regular(path, allowed_old_sha256):
        raise Amendment06IntegrityError(f"{label} old-byte pin differs")
    staging = A08_RUNTIME / "source_freeze/staging" / f"{desired_sha256}.bin"
    if staging.exists():
        if not (
            _safe_regular(staging, desired_sha256)
            and staging.stat().st_size == len(desired)
        ):
            raise Amendment06IntegrityError(f"{label} staging bytes drifted")
    else:
        publish_bytes_no_clobber(staging, desired)
    if staging.stat().st_dev != path.stat().st_dev:
        raise Amendment06IntegrityError(f"{label} replacement crosses filesystems")
    if not _safe_regular(path, allowed_old_sha256):
        raise Amendment06IntegrityError(f"{label} changed during replacement")
    os.replace(staging, path)
    _fsync_parent(path)


def _amendment08_source_freeze_material(
    *, checks: Sequence[Mapping[str, Any]], recorded_at: str,
    evidence_generated_at: str, transaction_id: str,
) -> dict[str, Any]:
    checked = _validated_final_prerun_checks([dict(item) for item in checks])
    if tuple(item["category"] for item in checked) != PRERUN_CHECK_CATEGORIES:
        raise Amendment06IntegrityError("Amendment 08 final check order differs")
    for item in checked:
        path = Path(str(item["log_path"])).resolve()
        if A08_A06_BOUND_RUNTIME.resolve() not in path.parents:
            raise Amendment06IntegrityError(
                f"Amendment 08 final log is outside A06-bound runtime: {path}"
            )
    if not (
        _safe_regular(AUTHORIZED_CHANGE_LEDGER, A07_ORIGINAL_LEDGER_SHA256)
        and AUTHORIZED_CHANGE_LEDGER.stat().st_size == A07_ORIGINAL_LEDGER_SIZE_BYTES
        and _safe_regular(FINAL_PRERUN_EVIDENCE, A07_ORIGINAL_PRERUN_SHA256)
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 source freeze requires the exact original A06 ledger/evidence"
        )
    event = build_authorized_edit_event(
        purpose=(
            f"{A08_AUTHORIZATION}:{A08_PROMPT_SHA256}:"
            "JOURNAL_ADJUDICATION_GPU_CORRIGENDUM_AND_PACKAGE_ALIGNMENT"
        ),
        test_evidence=checked, final_byte_freeze=True, recorded_at=recorded_at,
    )
    event.update({
        "corrigendum_authorization": A08_AUTHORIZATION,
        "corrigendum_prompt_sha256": A08_PROMPT_SHA256,
        "corrigendum_scientific_scope": list(A08_CHANGED_SCIENTIFIC_RELATIVES),
        "corrigendum_package_scope": list(A08_CHANGED_PACKAGE_RELATIVES),
        "pre_a08_source_hashes": dict(A08_PRE_SOURCE_HASHES),
        "source_freeze_transaction_id": transaction_id,
    })
    _validate_amendment08_ledger_event(event, require_latest_tree=True)
    original_ledger = AUTHORIZED_CHANGE_LEDGER.read_bytes()
    ledger_bytes = (
        original_ledger + (b"" if original_ledger.endswith(b"\n") else b"\n")
        + strict_full_canonical_json_bytes(event) + b"\n"
    )
    mapping = _validated_tree_mapping(
        event["after_tree"], label="projected Amendment 08 after-tree",
    )
    chain = {
        "mapping": mapping,
        "canonical_tree_sha256": canonical_code_tree_sha256(mapping),
        "ledger_sha256": sha256_bytes(ledger_bytes),
        "latest_event_index": 5,
        "latest_event_sha256": sha256_bytes(
            strict_full_canonical_json_bytes(event)
        ),
    }
    evidence = build_final_prerun_evidence(
        chain=chain, checks=checked, generated_at=evidence_generated_at,
    )
    evidence_bytes = strict_full_json_bytes(evidence)
    return {
        "event": event, "event_canonical_sha256": chain["latest_event_sha256"],
        "ledger_bytes": ledger_bytes, "ledger_sha256": chain["ledger_sha256"],
        "evidence": evidence, "evidence_bytes": evidence_bytes,
        "evidence_sha256": sha256_bytes(evidence_bytes),
        "mapping": mapping, "checks": checked,
    }


def _amendment08_source_freeze_intent_payload(
    *, material: Mapping[str, Any], transaction_id: str,
    recorded_at: str, evidence_generated_at: str,
) -> dict[str, Any]:
    return {
        "schema": "amendment08_source_freeze_install_intent/v1",
        "status": "INTENT", "authorization": A08_AUTHORIZATION,
        "prompt_sha256": A08_PROMPT_SHA256, "target_run_id": A08_TARGET_RUN_ID,
        "transaction_id": transaction_id, "created_at": recorded_at,
        "evidence_generated_at": evidence_generated_at,
        "old_authorized_change_ledger_sha256": A07_ORIGINAL_LEDGER_SHA256,
        "old_authorized_change_ledger_size_bytes": A07_ORIGINAL_LEDGER_SIZE_BYTES,
        "old_final_prerun_evidence_sha256": A07_ORIGINAL_PRERUN_SHA256,
        "projected_source_ledger_event": material["event"],
        "projected_source_ledger_event_canonical_sha256": material[
            "event_canonical_sha256"
        ],
        "projected_authorized_change_ledger_sha256": material["ledger_sha256"],
        "projected_authorized_change_ledger_size_bytes": len(material["ledger_bytes"]),
        "projected_final_prerun_evidence": material["evidence"],
        "projected_final_prerun_evidence_sha256": material["evidence_sha256"],
        "projected_final_prerun_evidence_size_bytes": len(material["evidence_bytes"]),
        "final_source_hashes": {
            relative: str(material["mapping"][relative]["sha256"])
            for relative in A08_CHANGED_ALL_RELATIVES
        },
        "test_evidence": material["checks"],
    }


def _amendment08_source_freeze_material_from_intent(
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    event = dict(intent.get("projected_source_ledger_event", {}))
    _validate_amendment08_ledger_event(event, require_latest_tree=True)
    ledger_raw = AUTHORIZED_CHANGE_LEDGER.read_bytes()
    if len(ledger_raw) < A07_ORIGINAL_LEDGER_SIZE_BYTES:
        raise Amendment06IntegrityError("Amendment 08 source-ledger prefix disappeared")
    original = ledger_raw[:A07_ORIGINAL_LEDGER_SIZE_BYTES]
    if sha256_bytes(original) != A07_ORIGINAL_LEDGER_SHA256:
        raise Amendment06IntegrityError("Amendment 08 original source-ledger prefix drifted")
    ledger_bytes = (
        original + (b"" if original.endswith(b"\n") else b"\n")
        + strict_full_canonical_json_bytes(event) + b"\n"
    )
    evidence = dict(intent.get("projected_final_prerun_evidence", {}))
    evidence_bytes = strict_full_json_bytes(evidence)
    mapping = _validated_tree_mapping(
        event.get("after_tree"), label="installed Amendment 08 after-tree",
    )
    chain = {
        "mapping": mapping,
        "canonical_tree_sha256": canonical_code_tree_sha256(mapping),
        "ledger_sha256": sha256_bytes(ledger_bytes),
        "latest_event_index": 5,
        "latest_event_sha256": sha256_bytes(
            strict_full_canonical_json_bytes(event)
        ),
    }
    checks = _validated_final_prerun_checks(intent.get("test_evidence"))
    expected_evidence = build_final_prerun_evidence(
        chain=chain, checks=checks,
        generated_at=str(intent.get("evidence_generated_at", "")),
    )
    if evidence != expected_evidence:
        raise Amendment06IntegrityError("Amendment 08 projected final evidence drifted")
    material = {
        "event": event, "event_canonical_sha256": chain["latest_event_sha256"],
        "ledger_bytes": ledger_bytes, "ledger_sha256": chain["ledger_sha256"],
        "evidence": evidence, "evidence_bytes": evidence_bytes,
        "evidence_sha256": sha256_bytes(evidence_bytes),
        "mapping": mapping, "checks": checks,
    }
    expected = _amendment08_source_freeze_intent_payload(
        material=material, transaction_id=str(intent.get("transaction_id", "")),
        recorded_at=str(intent.get("created_at", "")),
        evidence_generated_at=str(intent.get("evidence_generated_at", "")),
    )
    if dict(intent) != expected:
        raise Amendment06IntegrityError("Amendment 08 source-freeze intent drifted")
    return material


def validate_amendment08_source_freeze() -> dict[str, Any]:
    if not (_safe_regular(A08_SOURCE_FREEZE_INTENT) and _safe_regular(A08_SOURCE_FREEZE_RECEIPT)):
        raise Amendment06IntegrityError("Amendment 08 source-freeze transaction is incomplete")
    intent = strict_full_load_file(A08_SOURCE_FREEZE_INTENT)
    required_intent = {
        "schema", "status", "authorization", "prompt_sha256", "target_run_id",
        "transaction_id", "created_at", "evidence_generated_at",
        "old_authorized_change_ledger_sha256",
        "old_authorized_change_ledger_size_bytes", "old_final_prerun_evidence_sha256",
        "projected_source_ledger_event",
        "projected_source_ledger_event_canonical_sha256",
        "projected_authorized_change_ledger_sha256",
        "projected_authorized_change_ledger_size_bytes",
        "projected_final_prerun_evidence",
        "projected_final_prerun_evidence_sha256",
        "projected_final_prerun_evidence_size_bytes", "final_source_hashes",
        "test_evidence",
    }
    _require_exact_keys(intent, required_intent, label="Amendment 08 source-freeze intent")
    if not (
        intent["schema"] == "amendment08_source_freeze_install_intent/v1"
        and intent["status"] == "INTENT"
        and intent["authorization"] == A08_AUTHORIZATION
        and intent["prompt_sha256"] == A08_PROMPT_SHA256
        and intent["target_run_id"] == A08_TARGET_RUN_ID
        and re.fullmatch(r"[0-9a-f]{32}", str(intent["transaction_id"]))
        and intent["old_authorized_change_ledger_sha256"] == A07_ORIGINAL_LEDGER_SHA256
        and intent["old_authorized_change_ledger_size_bytes"] == A07_ORIGINAL_LEDGER_SIZE_BYTES
        and intent["old_final_prerun_evidence_sha256"] == A07_ORIGINAL_PRERUN_SHA256
    ):
        raise Amendment06IntegrityError("Amendment 08 source-freeze intent semantics differ")
    event = intent["projected_source_ledger_event"]
    _validate_amendment08_ledger_event(event, require_latest_tree=True)
    if not (
        event.get("source_freeze_transaction_id") == intent["transaction_id"]
        and sha256_bytes(strict_full_canonical_json_bytes(event))
        == intent["projected_source_ledger_event_canonical_sha256"]
        and _safe_regular(
            AUTHORIZED_CHANGE_LEDGER,
            str(intent["projected_authorized_change_ledger_sha256"]),
        )
        and AUTHORIZED_CHANGE_LEDGER.stat().st_size
        == intent["projected_authorized_change_ledger_size_bytes"]
        and _safe_regular(
            FINAL_PRERUN_EVIDENCE,
            str(intent["projected_final_prerun_evidence_sha256"]),
        )
        and FINAL_PRERUN_EVIDENCE.stat().st_size
        == intent["projected_final_prerun_evidence_size_bytes"]
        and strict_full_load_file(FINAL_PRERUN_EVIDENCE)
        == intent["projected_final_prerun_evidence"]
    ):
        raise Amendment06IntegrityError("Amendment 08 source-freeze installed bytes differ")
    chain = validate_authorized_change_chain()
    validate_final_prerun_evidence(chain)
    receipt = strict_full_load_file(A08_SOURCE_FREEZE_RECEIPT)
    required_receipt = {
        "schema", "status", "authorization", "prompt_sha256", "target_run_id",
        "transaction_id", "intent_sha256", "source_ledger_event_index",
        "source_ledger_event_canonical_sha256", "authorized_change_ledger_sha256",
        "authorized_change_ledger_size_bytes", "final_prerun_evidence_sha256",
        "final_prerun_evidence_size_bytes", "final_source_hashes", "test_evidence",
        "installed_at",
    }
    _require_exact_keys(
        receipt, required_receipt, label="Amendment 08 source-freeze receipt",
    )
    expected_source_hashes = {
        relative: chain["mapping"][relative]["sha256"]
        for relative in A08_CHANGED_ALL_RELATIVES
    }
    if not (
        receipt["schema"] == "amendment08_source_freeze_install_receipt/v1"
        and receipt["status"] == "PASS"
        and receipt["authorization"] == A08_AUTHORIZATION
        and receipt["prompt_sha256"] == A08_PROMPT_SHA256
        and receipt["target_run_id"] == A08_TARGET_RUN_ID
        and receipt["transaction_id"] == intent["transaction_id"]
        and receipt["intent_sha256"] == sha256_file(A08_SOURCE_FREEZE_INTENT)
        and receipt["source_ledger_event_index"] == 5
        and receipt["source_ledger_event_canonical_sha256"]
        == intent["projected_source_ledger_event_canonical_sha256"]
        and receipt["authorized_change_ledger_sha256"] == chain["ledger_sha256"]
        and receipt["authorized_change_ledger_size_bytes"]
        == AUTHORIZED_CHANGE_LEDGER.stat().st_size
        and receipt["final_prerun_evidence_sha256"] == sha256_file(FINAL_PRERUN_EVIDENCE)
        and receipt["final_prerun_evidence_size_bytes"] == FINAL_PRERUN_EVIDENCE.stat().st_size
        and receipt["final_source_hashes"] == expected_source_hashes
        and receipt["test_evidence"] == intent["test_evidence"]
        and isinstance(receipt["installed_at"], str) and receipt["installed_at"]
    ):
        raise Amendment06IntegrityError("Amendment 08 source-freeze receipt differs")
    return {
        "status": "PASS", "authorization": A08_AUTHORIZATION,
        "prompt_sha256": A08_PROMPT_SHA256, "target_run_id": A08_TARGET_RUN_ID,
        "transaction_id": intent["transaction_id"],
        "source_freeze_intent_sha256": sha256_file(A08_SOURCE_FREEZE_INTENT),
        "source_freeze_receipt_sha256": sha256_file(A08_SOURCE_FREEZE_RECEIPT),
        "current_a06_authorized_change_ledger_sha256": chain["ledger_sha256"],
        "a08_source_ledger_event_index": 5,
        "a08_source_ledger_event_canonical_sha256": chain["latest_event_sha256"],
        "current_a06_final_prerun_evidence_sha256": sha256_file(FINAL_PRERUN_EVIDENCE),
        "final_source_hashes": expected_source_hashes,
    }


def install_amendment08_source_freeze(
    *, checks: Sequence[Mapping[str, Any]], recorded_at: str | None = None,
    evidence_generated_at: str | None = None,
) -> dict[str, Any]:
    if A08_SOURCE_FREEZE_RECEIPT.exists():
        return validate_amendment08_source_freeze()
    with amendment06_lock(A08_RUNTIME / "source_freeze"):
        if A08_SOURCE_FREEZE_RECEIPT.exists():
            return validate_amendment08_source_freeze()
        if A08_SOURCE_FREEZE_INTENT.exists():
            intent = strict_full_load_file(A08_SOURCE_FREEZE_INTENT)
            transaction_id = str(intent.get("transaction_id", ""))
            recorded_at = str(intent.get("created_at", ""))
            evidence_generated_at = str(intent.get("evidence_generated_at", ""))
            material = _amendment08_source_freeze_material_from_intent(intent)
            if material["checks"] != _validated_final_prerun_checks(
                [dict(item) for item in checks]
            ):
                raise Amendment06IntegrityError(
                    "Amendment 08 source-freeze retry requested different test evidence"
                )
        else:
            transaction_id = uuid.uuid4().hex
            recorded_at = recorded_at or _now()
            evidence_generated_at = evidence_generated_at or _now()
            material = _amendment08_source_freeze_material(
                checks=checks, recorded_at=str(recorded_at),
                evidence_generated_at=str(evidence_generated_at),
                transaction_id=transaction_id,
            )
        expected_intent = _amendment08_source_freeze_intent_payload(
            material=material, transaction_id=transaction_id,
            recorded_at=str(recorded_at),
            evidence_generated_at=str(evidence_generated_at),
        )
        if A08_SOURCE_FREEZE_INTENT.exists():
            if strict_full_load_file(A08_SOURCE_FREEZE_INTENT) != expected_intent:
                raise Amendment06IntegrityError("Amendment 08 source-freeze intent drifted")
        else:
            publish_strict_json_no_clobber(A08_SOURCE_FREEZE_INTENT, expected_intent)
        _replace_exact_file_from_a08_staging(
            path=AUTHORIZED_CHANGE_LEDGER, desired=material["ledger_bytes"],
            allowed_old_sha256=A07_ORIGINAL_LEDGER_SHA256,
            label="Amendment 08 source ledger",
        )
        _replace_exact_file_from_a08_staging(
            path=FINAL_PRERUN_EVIDENCE, desired=material["evidence_bytes"],
            allowed_old_sha256=A07_ORIGINAL_PRERUN_SHA256,
            label="Amendment 08 final pre-Run evidence",
        )
        chain = validate_authorized_change_chain()
        validate_final_prerun_evidence(chain)
        receipt = {
            "schema": "amendment08_source_freeze_install_receipt/v1",
            "status": "PASS", "authorization": A08_AUTHORIZATION,
            "prompt_sha256": A08_PROMPT_SHA256, "target_run_id": A08_TARGET_RUN_ID,
            "transaction_id": transaction_id,
            "intent_sha256": sha256_file(A08_SOURCE_FREEZE_INTENT),
            "source_ledger_event_index": 5,
            "source_ledger_event_canonical_sha256": chain["latest_event_sha256"],
            "authorized_change_ledger_sha256": chain["ledger_sha256"],
            "authorized_change_ledger_size_bytes": AUTHORIZED_CHANGE_LEDGER.stat().st_size,
            "final_prerun_evidence_sha256": sha256_file(FINAL_PRERUN_EVIDENCE),
            "final_prerun_evidence_size_bytes": FINAL_PRERUN_EVIDENCE.stat().st_size,
            "final_source_hashes": {
                relative: chain["mapping"][relative]["sha256"]
                for relative in A08_CHANGED_ALL_RELATIVES
            },
            "test_evidence": material["checks"], "installed_at": _now(),
        }
        publish_strict_json_no_clobber(A08_SOURCE_FREEZE_RECEIPT, receipt)
    return validate_amendment08_source_freeze()


def _resolved_equal(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=True) == right.resolve(strict=True)
    except FileNotFoundError:
        return False


def assert_exact_invocation(
    *, preflight_run: Path, smoke_run: Path,
    expected_runtime_root: Path | None = None,
) -> dict[str, Any]:
    if expected_runtime_root is None:
        expected_runtime_root = EXPECTED_RUNTIME_ROOT
    runtime_environment = os.environ.get("ABLATION_RUNTIME_ROOT", "")
    if not _resolved_equal(Path(preflight_run), ACCEPTED_PREFLIGHT):
        raise Amendment06IntegrityError("Amendment 06 requires the exact accepted Preflight")
    if not _resolved_equal(Path(smoke_run), ACCEPTED_SMOKE):
        raise Amendment06IntegrityError("Amendment 06 requires the exact accepted Smoke")
    if Path(sys.executable).resolve() != CUDA_PYTHON.resolve() or Path(sys.prefix).resolve() != CUDA_ENV.resolve():
        raise Amendment06IntegrityError(f"Amendment 06 requires interpreter {CUDA_PYTHON}")
    if not runtime_environment or Path(runtime_environment).resolve() != expected_runtime_root.resolve():
        raise Amendment06IntegrityError(
            f"ABLATION_RUNTIME_ROOT must be exactly {expected_runtime_root}"
        )
    if not expected_runtime_root.is_dir() or expected_runtime_root.is_symlink():
        raise Amendment06IntegrityError("Amendment 06 runtime root is absent or unsafe")
    build = xgb.build_info()
    if xgb.__version__ != "2.1.1" or sklearn.__version__ != "1.7.2" or build.get("USE_CUDA") is not True:
        raise Amendment06IntegrityError("Pinned XGBoost/scikit-learn/CUDA environment differs")
    return {
        "status": "PASS", "cuda_env": str(CUDA_ENV),
        "python": str(Path(sys.executable).resolve()),
        "runtime_root": str(expected_runtime_root.resolve()),
        "xgboost_version": xgb.__version__, "scikit_learn_version": sklearn.__version__,
        "xgboost_use_cuda": True,
    }


def _smoke_identity_is_ready_for_full(identity: Mapping[str, Any]) -> bool:
    semantics = (
        identity.get("reference_validation", {})
        .get("smoke_semantic_completeness", {})
    )
    return bool(
        identity.get("run_id") == ACCEPTED_SMOKE.name
        and semantics.get("ready_for_full_awaiting_external_review") is True
        and semantics.get("status") == "PASS"
        and identity.get("full_authorized") is False
    )


def verify_reference_admission(
    *, preflight_run: Path, smoke_run: Path,
) -> dict[str, Any]:
    for path, expected in PINNED_FILES.items():
        if not _safe_regular(path, expected):
            raise Amendment06IntegrityError(f"Pinned Amendment 06 reference drifted: {path}")
    admission = strict_full_load_file(PRE_EDIT_ADMISSION)
    if not (
        admission.get("authorization") == AUTHORIZATION
        and admission.get("status") == "PASS"
        and admission.get("blockers") == []
        and admission.get("preflight_reference_validation", {}).get("status")
        == "PASS_SEALED_ONLY_NO_LIVE_AUDIT_PARSE"
        and admission.get("full_run_guard", {}).get("status") == "PASS"
        and admission.get("a05_recovery", {}).get("status") == "PASS"
    ):
        raise Amendment06IntegrityError("Committed pre-edit admission is not a clean PASS")
    chain = validate_authorized_change_chain()
    prerun = validate_final_prerun_evidence(chain)
    signature = inspect.signature(core.validate_amended_run_reference)
    if not {
        "amendment06_allowed_after_hashes", "amendment06_change_ledger_evidence",
    }.issubset(signature.parameters):
        raise Amendment06IntegrityError("Production Smoke validator lacks Amendment 06 drift routing")
    smoke_identity = core.validate_amended_run_reference(
        smoke_run,
        expected_kind="CUDA_SMOKE_NON_SCIENTIFIC",
        allowed_states={"SMOKE_COMPLETE"},
        amendment06_allowed_after_hashes=chain["allowed_after_hashes"],
        amendment06_change_ledger_evidence=chain["change_ledger_evidence"],
    )
    if not _smoke_identity_is_ready_for_full(smoke_identity):
        raise Amendment06IntegrityError("Accepted Smoke is not the sealed Full-admission reference")
    preflight_identity = strict_full_load_file(preflight_run / "config/run_identity.lock.json")
    if not (
        preflight_identity.get("run_kind") == "AMENDED_PREFLIGHT"
        and preflight_identity.get("state") == "PREFLIGHT_COMPLETE"
        and preflight_identity.get("full_authorized") is False
        and preflight_identity.get("candidate_split_hash") == SPLIT_ASSIGNMENT_SHA256
    ):
        raise Amendment06IntegrityError("Accepted Preflight identity differs from sealed admission")
    return {
        "status": "PASS", "preflight": "PASS_SEALED_ONLY_NO_LIVE_AUDIT_PARSE",
        "smoke": "PASS_PRODUCTION_VALIDATOR", "authorized_change_chain": chain,
        "final_prerun_evidence": prerun,
        "pre_edit_admission_sha256": sha256_file(PRE_EDIT_ADMISSION),
    }


@contextmanager
def amendment06_lock(runtime_root: Path | None = None) -> Iterator[None]:
    if runtime_root is None:
        runtime_root = A06_RUNTIME
    runtime_root.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_root / "amendment06_full.lock"
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Amendment06ResumeError("Another Amendment 06 process holds the execution lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def amendment08_isolated_rehearsal_context(
    *, run_dir: Path, runtime_root: Path,
) -> Iterator[None]:
    """Authorize one exact-basename rehearsal clone without relaxing live receipt bytes."""

    resolved_run = Path(run_dir).resolve()
    resolved_runtime = Path(runtime_root).resolve()
    isolated_root = resolved_runtime.parent
    resolved_a06_runtime = Path(A06_RUNTIME).resolve()
    resolved_a08_runtime = Path(A08_RUNTIME).resolve()
    expected_package_root = (
        resolved_runtime / "package_code_lineage" / A08_TARGET_RUN_ID
    )
    if not (
        os.environ.get("AMENDMENT08_ENABLE_TRUE_REHEARSAL") == "1"
        and "PYTEST_CURRENT_TEST" in os.environ
        and resolved_run.name == A08_TARGET_RUN_ID
        and resolved_run == Path(A08_TARGET_RUN).resolve()
        and resolved_run.parent == Path(RESULTS_ROOT).resolve()
        and resolved_runtime == Path(EXPECTED_RUNTIME_ROOT).resolve()
        and resolved_run.parent.name == "results"
        and resolved_runtime.name == "runtime"
        and resolved_run.parent.parent == resolved_runtime.parent
        and _isolated_exact_target_journal_pair(resolved_run, resolved_runtime)
        and Path(A07_TARGET_RUN).resolve() == resolved_run
        and Path(RESULTS_ROOT).resolve() == isolated_root / "results"
        and resolved_a06_runtime == isolated_root / "a06_runtime"
        and resolved_a08_runtime == isolated_root / "a08_runtime"
        and resolved_a06_runtime.is_dir() and not resolved_a06_runtime.is_symlink()
        and resolved_a08_runtime.is_dir() and not resolved_a08_runtime.is_symlink()
        and Path(A08_A06_BOUND_RUNTIME).resolve()
        == resolved_a06_runtime / "corrigendum08_journal_adjudication"
        and Path(A08_INSTALL_INTENT).resolve()
        == resolved_a08_runtime / "authorization/amendment08_overlay_install_intent.json"
        and Path(A08_SOURCE_FREEZE_INTENT).resolve()
        == resolved_a08_runtime / "source_freeze/install_intent.json"
        and Path(A08_SOURCE_FREEZE_RECEIPT).resolve()
        == resolved_a08_runtime / "source_freeze/install_receipt.json"
        and Path(AUTHORIZED_CHANGE_LEDGER).resolve()
        == resolved_a06_runtime / "authorized_change_ledger.jsonl"
        and Path(FINAL_PRERUN_EVIDENCE).resolve()
        == resolved_a06_runtime / "pre_run/final_prerun_evidence.json"
        and Path(BASELINE_TREE_TSV).resolve()
        == resolved_a06_runtime / "pre_edit/baseline_tree_sha_size_rel.tsv"
        and Path(A08_PACKAGE_LINEAGE_ROOT).resolve() == expected_package_root
        and Path(A08_PACKAGE_A08_LINEAGE_ROOT).resolve()
        == expected_package_root / "amendment08"
        and _safe_regular(
            resolved_a06_runtime / "authorization/full_run_intent.json",
            A07_A06_INTENT_SHA256,
        )
        and _safe_regular(
            resolved_a06_runtime / "authorization/full_run_receipt.json",
            A07_A06_RECEIPT_SHA256,
        )
        and _safe_regular(
            resolved_a06_runtime / "authorized_change_ledger.jsonl",
            A07_ORIGINAL_LEDGER_SHA256,
        )
        and _safe_regular(
            resolved_a06_runtime / "pre_run/final_prerun_evidence.json",
            A07_ORIGINAL_PRERUN_SHA256,
        )
        and _safe_regular(
            resolved_a06_runtime / "live/full_run.log",
            A07_ORIGINAL_FULL_LOG_SHA256,
        )
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 rehearsal context is not an exact isolated pytest Run/Runtime pair"
        )
    token = _AMENDMENT08_REHEARSAL_CONTEXT.set((resolved_run, resolved_runtime))
    try:
        yield
    finally:
        _AMENDMENT08_REHEARSAL_CONTEXT.reset(token)


def _validate_amendment08_rehearsal_authorization_relocation(
    run_dir: Path,
) -> dict[str, Any]:
    context_run, runtime_root = _amendment08_rehearsal_context_paths()
    selected = Path(run_dir).resolve()
    clone_a06_runtime = Path(A06_RUNTIME).resolve()
    identity_path = selected / "config/run_identity.lock.json"
    intent_path = _run_intent_path()
    receipt_path = _run_receipt_path()
    production_run = Path(PRODUCTION_RESULTS_ROOT).resolve() / A08_TARGET_RUN_ID
    if not (
        selected == context_run
        and clone_a06_runtime == runtime_root.parent / "a06_runtime"
        and intent_path == clone_a06_runtime / "authorization/full_run_intent.json"
        and receipt_path == clone_a06_runtime / "authorization/full_run_receipt.json"
        and _safe_regular(identity_path, A07_BASE_IDENTITY_SHA256)
        and _safe_regular(intent_path, A07_A06_INTENT_SHA256)
        and _safe_regular(receipt_path, A07_A06_RECEIPT_SHA256)
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 rehearsal authorization relocation pins differ"
        )
    identity = strict_full_load_file(identity_path)
    intent = _require_exact_keys(
        strict_full_load_file(intent_path), {
            "authorization", "authorization_reserved", "run_id", "run_path",
            "preflight_run_id", "smoke_run_id", "candidate_split_hash",
            "reserved_at",
        }, label="Amendment 08 relocated A06 intent",
    )
    receipt = _require_exact_keys(
        strict_full_load_file(receipt_path), {
            "authorization", "authorization_consumed", "run_id", "run_path",
            "run_identity_sha256", "consumed_at",
        }, label="Amendment 08 relocated A06 receipt",
    )
    if not (
        identity.get("run_id") == selected.name == A08_TARGET_RUN_ID
        and identity.get("run_kind") == RUN_KIND
        and identity.get("state") == "FULL_IN_PROGRESS"
        and identity.get("full_authorization") == AUTHORIZATION
        and intent.get("authorization") == AUTHORIZATION
        and intent.get("authorization_reserved") is True
        and intent.get("run_id") == selected.name
        and intent.get("run_path") == str(production_run)
        and intent.get("candidate_split_hash") == SPLIT_ASSIGNMENT_SHA256
        and receipt.get("authorization") == AUTHORIZATION
        and receipt.get("authorization_consumed") is True
        and receipt.get("run_id") == selected.name
        and receipt.get("run_path") == str(production_run)
        and receipt.get("run_identity_sha256") == A07_BASE_IDENTITY_SHA256
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 rehearsal authorization relocation semantics differ"
        )
    return {
        "schema": "amendment08_isolated_rehearsal_authorization_relocation/v1",
        "status": "PASS", "mode": A08_ISOLATED_REHEARSAL_PROJECTION_MODE,
        "target_run_id": selected.name, "clone_run_path": str(selected),
        "clone_runtime_path": str(runtime_root),
        "clone_a06_runtime_path": str(clone_a06_runtime),
        "production_run_path": str(production_run),
        "base_identity_sha256": A07_BASE_IDENTITY_SHA256,
        "full_run_intent_sha256": A07_A06_INTENT_SHA256,
        "full_run_receipt_sha256": A07_A06_RECEIPT_SHA256,
        "stored_intent_run_path": str(intent["run_path"]),
        "stored_receipt_run_path": str(receipt["run_path"]),
        "receipt_rewritten": False, "new_receipt_created": False,
        "production_authorization_reconsumed": False,
    }


def _amendment08_rehearsal_receipt_path_matches(
    *, run_dir: Path, receipt: Mapping[str, Any], identity_sha256: str,
) -> bool:
    context = _AMENDMENT08_REHEARSAL_CONTEXT.get()
    if context is None:
        return False
    resolved_run = Path(run_dir).resolve()
    try:
        relocation = _validate_amendment08_rehearsal_authorization_relocation(
            resolved_run
        )
    except Amendment06IntegrityError:
        return False
    return bool(
        context == (resolved_run, Path(EXPECTED_RUNTIME_ROOT).resolve())
        and relocation["base_identity_sha256"] == identity_sha256
        and dict(receipt) == strict_full_load_file(_run_receipt_path())
    )


def existing_full_runs(results_root: Path | None = None) -> list[Path]:
    if results_root is None:
        results_root = RESULTS_ROOT
    matches = sorted(results_root.glob(f"run_*{RUN_SUFFIX}"))
    runs: list[Path] = []
    for path in matches:
        mode = os.lstat(path).st_mode
        if not stat.S_ISDIR(mode):
            raise Amendment06IntegrityError(
                f"Unsafe one-Run guard entry occupies an Amendment 06 Full name: {path}"
            )
        runs.append(path)
    return runs


def _manifest_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    frame = pd.DataFrame(rows, columns=["relative_path", "size_bytes", "sha256", "role"])
    return frame.to_csv(index=False, sep="\t", lineterminator="\n").encode("utf-8")


def _code_manifest(
    relatives: Sequence[str], *, role: str,
) -> tuple[bytes, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for relative in relatives:
        path = STUDY_ROOT / relative
        if not _safe_regular(path):
            raise Amendment06IntegrityError(f"Code-manifest input is absent/unsafe: {relative}")
        digest = sha256_file(path)
        hashes[relative] = digest
        rows.append({
            "relative_path": relative, "size_bytes": path.stat().st_size,
            "sha256": digest,
            "role": role,
        })
    return _manifest_bytes(rows), hashes


def _parse_code_manifest_bytes(data: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t"))
    except Exception as exc:
        raise Amendment06IntegrityError(f"{label} is unreadable") from exc
    if not rows or tuple(rows[0]) != (
        "relative_path", "size_bytes", "sha256", "role",
    ):
        raise Amendment06IntegrityError(f"{label} schema differs")
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        relative = str(row.get("relative_path", ""))
        relative_path = Path(relative)
        size_text = str(row.get("size_bytes", ""))
        digest = str(row.get("sha256", ""))
        role = str(row.get("role", ""))
        if (
            not relative or relative in seen or relative_path.is_absolute()
            or ".." in relative_path.parts or relative_path.as_posix() != relative
            or not re.fullmatch(r"[0-9]+", size_text)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not role
        ):
            raise Amendment06IntegrityError(f"{label} row differs: {relative!r}")
        seen.add(relative)
        parsed.append({
            "relative_path": relative,
            "size_bytes": int(size_text),
            "sha256": digest,
            "role": role,
        })
    return parsed


def _validate_initial_package_manifest(
    run_dir: Path, identity: Mapping[str, Any],
) -> dict[str, str]:
    package_path = run_dir / "provenance/package_validator_code_manifest_initial.tsv"
    if not _safe_regular(package_path):
        raise Amendment06IntegrityError("Initial package-validator manifest is absent/unsafe")
    package_bytes = package_path.read_bytes()
    package_rows = _parse_code_manifest_bytes(
        package_bytes, label="Initial package-validator manifest",
    )
    if [row["relative_path"] for row in package_rows] != list(PACKAGE_CODE_RELATIVES):
        raise Amendment06IntegrityError("Initial package-validator manifest order differs")
    package_hashes: dict[str, str] = {}
    for row in package_rows:
        relative = str(row["relative_path"])
        if row["role"] != "PACKAGE_VALIDATOR_OR_TEST":
            raise Amendment06IntegrityError(
                f"Initial package-validator manifest role differs: {relative}"
            )
        package_hashes[relative] = str(row["sha256"])
    if identity.get("package_validator_code_manifest_initial_sha256") != sha256_bytes(
        package_bytes
    ):
        raise Amendment06IntegrityError("Initial package-validator identity binding differs")
    return package_hashes


def _validate_amendment07_base_history(run_dir: Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir).resolve()
    identity_path = run_dir / "config/run_identity.lock.json"
    manifest_path = run_dir / "provenance/scientific_execution_code_manifest.tsv"
    lineage_path = run_dir / "provenance/amendment06_code_lineage.json"
    if not (
        _safe_regular(identity_path, A07_BASE_IDENTITY_SHA256)
        and _safe_regular(manifest_path, A07_BASE_SCIENTIFIC_MANIFEST_SHA256)
        and _safe_regular(lineage_path, A07_BASE_CODE_LINEAGE_SHA256)
        and _safe_regular(_run_intent_path(), A07_A06_INTENT_SHA256)
        and _safe_regular(_run_receipt_path(), A07_A06_RECEIPT_SHA256)
        and _safe_regular(
            A06_RUNTIME / "live/full_run.log", A07_ORIGINAL_FULL_LOG_SHA256,
        )
        and _safe_regular(
            run_dir / "provenance/cuda_active_fit_probe.json",
            A07_IMMUTABLE_CUDA_PROBE_SHA256,
        )
    ):
        raise Amendment06IntegrityError("Immutable Amendment 06 base history drifted")
    identity = strict_full_load_file(identity_path)
    if not (
        identity.get("run_id") == run_dir.name == A07_TARGET_RUN_ID
        and identity.get("run_kind") == RUN_KIND
        and identity.get("scientific_execution_code_manifest_sha256")
        == A07_BASE_SCIENTIFIC_MANIFEST_SHA256
    ):
        raise Amendment06IntegrityError("Immutable Amendment 06 identity binding drifted")
    rows = _parse_code_manifest_bytes(
        manifest_path.read_bytes(), label="Base scientific manifest",
    )
    expected = (*SCIENTIFIC_CODE_RELATIVES, *SCIENTIFIC_TEST_RELATIVES)
    if [row["relative_path"] for row in rows] != list(expected) or any(
        row["role"] != "SCIENTIFIC_EXECUTION_OR_TEST" for row in rows
    ):
        raise Amendment06IntegrityError("Base scientific manifest order/role differs")
    for row in rows:
        relative = str(row["relative_path"])
        snapshot_root = "tests_snapshot" if relative.startswith("tests/") else "code_snapshot"
        snapshot = run_dir / "provenance" / snapshot_root / Path(relative).name
        if not (
            _safe_regular(snapshot, str(row["sha256"]))
            and snapshot.stat().st_size == int(row["size_bytes"])
        ):
            raise Amendment06IntegrityError(
                f"Immutable Amendment 06 base snapshot drifted: {relative}"
            )
    return rows


def _effective_amendment07_manifest_bytes(run_dir: Path) -> bytes:
    rows = _validate_amendment07_base_history(run_dir)
    effective: list[dict[str, Any]] = []
    for row in rows:
        relative = str(row["relative_path"])
        live = STUDY_ROOT / relative
        if not _safe_regular(live):
            raise Amendment06IntegrityError(
                f"Effective Amendment 07 source is absent/unsafe: {relative}"
            )
        effective.append({
            "relative_path": relative,
            "size_bytes": live.stat().st_size,
            "sha256": sha256_file(live),
            "role": "SCIENTIFIC_EXECUTION_OR_TEST",
        })
    return _manifest_bytes(effective)


def _amendment07_unified_diff_bytes(run_dir: Path) -> bytes:
    base_rows = _validate_amendment07_base_history(run_dir)
    by_relative = {str(row["relative_path"]): row for row in base_rows}
    pieces: list[str] = []
    for relative in A07_CHANGED_SCIENTIFIC_RELATIVES:
        row = by_relative[relative]
        snapshot_root = "tests_snapshot" if relative.startswith("tests/") else "code_snapshot"
        before_path = run_dir / "provenance" / snapshot_root / Path(relative).name
        after_path = STUDY_ROOT / relative
        before = before_path.read_text(encoding="utf-8").splitlines(keepends=True)
        after = after_path.read_text(encoding="utf-8").splitlines(keepends=True)
        pieces.extend(difflib.unified_diff(
            before, after, fromfile=f"a/{relative}", tofile=f"b/{relative}",
        ))
        if sha256_file(before_path) != str(row["sha256"]):
            raise Amendment06IntegrityError(f"Base diff input drifted: {relative}")
    data = "".join(pieces).encode("utf-8")
    if not data:
        raise Amendment06IntegrityError("Amendment 07 unified diff is empty")
    return data


def _validate_amendment07_code_manifests(run_dir: Path) -> dict[str, Any]:
    base_rows = _validate_amendment07_base_history(run_dir)
    effective_path = (
        run_dir
        / "provenance/scientific_execution_code_manifest_effective_amendment07.tsv"
    )
    expected_bytes = _effective_amendment07_manifest_bytes(run_dir)
    if not _safe_regular(effective_path, sha256_bytes(expected_bytes)):
        raise Amendment06IntegrityError("Effective Amendment 07 manifest drifted")
    effective_rows = _parse_code_manifest_bytes(
        effective_path.read_bytes(), label="Effective Amendment 07 scientific manifest",
    )
    changed: list[str] = []
    effective_hashes: dict[str, str] = {}
    for base, effective in zip(base_rows, effective_rows, strict=True):
        if base["relative_path"] != effective["relative_path"]:
            raise Amendment06IntegrityError("Effective Amendment 07 manifest order differs")
        relative = str(base["relative_path"])
        live = STUDY_ROOT / relative
        if not (
            effective["role"] == "SCIENTIFIC_EXECUTION_OR_TEST"
            and _safe_regular(live, str(effective["sha256"]))
            and live.stat().st_size == int(effective["size_bytes"])
        ):
            raise Amendment06IntegrityError(
                f"Effective Amendment 07 live source drifted: {relative}"
            )
        effective_hashes[relative] = str(effective["sha256"])
        if (base["size_bytes"], base["sha256"]) != (
            effective["size_bytes"], effective["sha256"],
        ):
            changed.append(relative)
    if tuple(changed) != A07_CHANGED_SCIENTIFIC_RELATIVES:
        raise Amendment06IntegrityError(
            f"Amendment 07 changed-path set differs: {changed}"
        )
    old_expected = {
        "code/amendment06_full.py": A07_OLD_FULL_SHA256,
        "tests/test_amendment06.py": A07_OLD_TEST_SHA256,
    }
    base_by_relative = {str(row["relative_path"]): row for row in base_rows}
    for relative, digest in old_expected.items():
        if base_by_relative[relative]["sha256"] != digest:
            raise Amendment06IntegrityError(f"Amendment 07 old-byte pin differs: {relative}")
        snapshot_root = (
            "tests_snapshot_amendment07"
            if relative.startswith("tests/") else "code_snapshot_amendment07"
        )
        corrected = run_dir / "provenance" / snapshot_root / Path(relative).name
        if not _safe_regular(corrected, effective_hashes[relative]):
            raise Amendment06IntegrityError(
                f"Corrected Amendment 07 snapshot drifted: {relative}"
            )
    if effective_hashes.get("code/amendment_core.py") != A07_UNCHANGED_CORE_SHA256:
        raise Amendment06IntegrityError("Amendment core changed under Amendment 07")
    package_hashes = _validate_initial_package_manifest(
        run_dir, strict_full_load_file(run_dir / "config/run_identity.lock.json"),
    )
    return {
        "status": "PASS",
        "base_manifest_sha256": A07_BASE_SCIENTIFIC_MANIFEST_SHA256,
        "effective_manifest_sha256": sha256_file(effective_path),
        "changed_paths": changed,
        "scientific": effective_hashes,
        "package": package_hashes,
        "package_manifest_sha256": sha256_file(
            run_dir / "provenance/package_validator_code_manifest_initial.tsv"
        ),
    }


def validate_frozen_code_manifests(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    if run_dir.name == A08_TARGET_RUN_ID or any(
        (run_dir / relative).exists() for relative in A08_OVERLAY_RUN_RELATIVES
    ):
        receipt = run_dir / "provenance/amendment08_overlay_install_receipt.json"
        if not _safe_regular(receipt):
            raise Amendment06IntegrityError(
                "Amendment 08 target has no valid active overlay receipt"
            )
        overlay = validate_amendment08_overlay(run_dir)
        code = dict(overlay["code_manifests"])
        effective_path = (
            run_dir
            / "provenance/scientific_execution_code_manifest_effective_amendment08.tsv"
        )
        rows = _parse_code_manifest_bytes(
            effective_path.read_bytes(), label="Amendment 08 effective manifest",
        )
        scientific = {
            str(row["relative_path"]): str(row["sha256"]) for row in rows
        }
        package = dict(overlay["package_lineage"]["final_hashes"])
        if set(scientific).intersection(package):
            raise Amendment06IntegrityError(
                "Amendment 08 scientific/package manifests are not disjoint"
            )
        return {
            "status": "PASS", "scientific": scientific, "package": package,
            "base_manifest_sha256": code["base_manifest_sha256"],
            "effective_manifest_sha256": code["effective_manifest_sha256"],
            "changed_paths": code["changed_scientific_paths"],
            "package_lineage_sha256": overlay["package_lineage"]["lineage_sha256"],
            "package_manifest_sha256": A08_INITIAL_PACKAGE_MANIFEST_SHA256,
        }
    if any(
        (run_dir / relative).exists() for relative in A07_OVERLAY_RUN_RELATIVES
    ):
        overlay = validate_amendment07_overlay(run_dir)
        return dict(overlay["code_manifests"])
    identity = strict_full_load_file(run_dir / "config/run_identity.lock.json")
    scientific_bytes, scientific_hashes = _code_manifest(
        (*SCIENTIFIC_CODE_RELATIVES, *SCIENTIFIC_TEST_RELATIVES),
        role="SCIENTIFIC_EXECUTION_OR_TEST",
    )
    scientific_path = run_dir / "provenance/scientific_execution_code_manifest.tsv"
    package_path = run_dir / "provenance/package_validator_code_manifest_initial.tsv"
    package_hashes = _validate_initial_package_manifest(run_dir, identity)
    package_bytes = package_path.read_bytes()
    if not (
        _safe_regular(scientific_path, sha256_bytes(scientific_bytes))
        and identity.get("scientific_execution_code_manifest_sha256") == sha256_bytes(scientific_bytes)
        and identity.get("package_validator_code_manifest_initial_sha256") == sha256_bytes(package_bytes)
        and set(scientific_hashes).isdisjoint(package_hashes)
    ):
        raise Amendment06IntegrityError("Frozen scientific/package code manifests drifted")
    for relative, digest in scientific_hashes.items():
        snapshot_root = "tests_snapshot" if relative.startswith("tests/") else "code_snapshot"
        snapshot = run_dir / "provenance" / snapshot_root / Path(relative).name
        if not _safe_regular(snapshot, digest):
            raise Amendment06IntegrityError(
                f"Frozen scientific code/test snapshot drifted: {relative}"
            )
    return {
        "status": "PASS", "scientific": scientific_hashes,
        "package": package_hashes,
        "package_live_bytes_may_advance_via_isolated_lineage": True,
        "scientific_manifest_sha256": sha256_bytes(scientific_bytes),
        "package_manifest_sha256": sha256_bytes(package_bytes),
    }


def _require_exact_keys(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise Amendment06IntegrityError(f"{label} strict schema differs: {observed}")
    return value


def _validate_amendment07_forensic_snapshot() -> dict[str, Any]:
    if not (
        _safe_regular(A07_PROMPT_PATH, A07_PROMPT_SHA256)
        and _safe_regular(
            A07_RUNTIME / "prompt" / A07_PROMPT_PATH.name, A07_PROMPT_SHA256,
        )
        and _safe_regular(A07_FORENSIC_ZIP, A07_FORENSIC_ZIP_SHA256)
        and _safe_regular(
            A07_FORENSIC_VERIFICATION, A07_FORENSIC_VERIFICATION_SHA256,
        )
        and _safe_regular(
            A07_FORENSIC_RECONSTRUCTION, A07_FORENSIC_RECONSTRUCTION_SHA256,
        )
        and _safe_regular(
            A07_FAILED_ATTEMPT_QUARANTINE_INTENT,
            A07_FAILED_ATTEMPT_QUARANTINE_INTENT_SHA256,
        )
        and _safe_regular(
            A07_FAILED_ATTEMPT_QUARANTINE_RECEIPT,
            A07_FAILED_ATTEMPT_QUARANTINE_RECEIPT_SHA256,
        )
    ):
        raise Amendment06IntegrityError("Amendment 07 forensic inputs are absent/unsafe")
    verification = _require_exact_keys(
        strict_full_load_file(A07_FORENSIC_VERIFICATION),
        {
            "schema", "status", "authorization", "a07_prompt_sha256",
            "bundle_path", "bundle_sha256", "bundle_size_bytes", "member_count",
            "members", "crc_status", "safe_names", "unique_names",
            "independent_reopen_status", "live_source_member_match",
            "cache_members_rehashed", "cache_inventory_sha256",
            "canonical_source_inventory_sha256",
        },
        label="Amendment 07 forensic verification",
    )
    members = verification["members"]
    if not (
        verification["schema"] == "amendment07_pre_mutation_forensic_verification/v1"
        and verification["status"] == "PASS"
        and verification["authorization"] == A07_AUTHORIZATION
        and verification["a07_prompt_sha256"] == A07_PROMPT_SHA256
        and Path(str(verification["bundle_path"])).resolve() == A07_FORENSIC_ZIP.resolve()
        and verification["bundle_sha256"] == A07_FORENSIC_ZIP_SHA256
        and verification["bundle_size_bytes"] == A07_FORENSIC_ZIP.stat().st_size
        and isinstance(members, list)
        and verification["member_count"] == len(members) == 64
        and verification["crc_status"] == "PASS"
        and verification["safe_names"] is True
        and verification["unique_names"] is True
        and verification["independent_reopen_status"] == "PASS"
        and verification["live_source_member_match"] == "PASS"
        and verification["cache_members_rehashed"] == "PASS_10_OF_10"
        and verification["cache_inventory_sha256"]
        == A07_FORENSIC_CACHE_INVENTORY_SHA256
        and verification["canonical_source_inventory_sha256"]
        == A07_FORENSIC_SOURCE_INVENTORY_SHA256
    ):
        raise Amendment06IntegrityError("Amendment 07 forensic verification differs")
    expected_records: dict[str, tuple[int, str]] = {}
    for item in members:
        record = _require_exact_keys(
            item, {"member_path", "size_bytes", "sha256", "crc32"},
            label="Amendment 07 forensic member",
        )
        name = str(record["member_path"])
        path = Path(name)
        size = record["size_bytes"]
        digest = str(record["sha256"])
        if (
            not name or path.is_absolute() or ".." in path.parts
            or path.as_posix() != name or name in expected_records
            or isinstance(size, bool) or not isinstance(size, int) or size < 0
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not re.fullmatch(r"[0-9a-f]{8}", str(record["crc32"]))
        ):
            raise Amendment06IntegrityError("Amendment 07 forensic member record differs")
        expected_records[name] = (size, digest)
    root = "amendment07_pre_mutation_forensic"
    source_inventory_name = f"{root}/metadata/source_member_inventory.tsv"
    cache_inventory_name = f"{root}/metadata/cache_member_inventory.tsv"
    facts_name = f"{root}/metadata/forensic_facts.json"
    metadata_records = {
        source_inventory_name: (16_155, A07_FORENSIC_SOURCE_INVENTORY_FILE_SHA256),
        cache_inventory_name: (2_652, A07_FORENSIC_CACHE_INVENTORY_SHA256),
        facts_name: (1_126, A07_FORENSIC_FACTS_SHA256),
    }
    if any(expected_records.get(name) != record for name, record in metadata_records.items()):
        raise Amendment06IntegrityError("Amendment 07 forensic metadata inventory differs")

    original_prerun_bytes: bytes
    with zipfile.ZipFile(A07_FORENSIC_ZIP, "r") as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if (
            archive.testzip() is not None or len(names) != len(set(names))
            or set(names) != set(expected_records)
        ):
            raise Amendment06IntegrityError("Amendment 07 forensic ZIP member set/CRC differs")
        for info in infos:
            size, digest = expected_records[info.filename]
            with archive.open(info, "r") as handle:
                hasher = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            if info.file_size != size or hasher.hexdigest() != digest:
                raise Amendment06IntegrityError(
                    f"Amendment 07 forensic ZIP member drifted: {info.filename}"
                )
        source_inventory_bytes = archive.read(source_inventory_name)
        cache_inventory_bytes = archive.read(cache_inventory_name)
        facts_bytes = archive.read(facts_name)

        try:
            source_reader = csv.DictReader(
                io.StringIO(source_inventory_bytes.decode("utf-8")), delimiter="\t",
            )
            if source_reader.fieldnames != [
                "member_path", "source_path", "size_bytes", "sha256",
            ]:
                raise ValueError("source inventory header")
            source_rows = list(source_reader)
            cache_reader = csv.DictReader(
                io.StringIO(cache_inventory_bytes.decode("utf-8")), delimiter="\t",
            )
            if cache_reader.fieldnames != [
                "relative_path", "absolute_path", "size_bytes", "sha256",
            ]:
                raise ValueError("cache inventory header")
            cache_rows = list(cache_reader)
        except (UnicodeDecodeError, csv.Error, ValueError) as exc:
            raise Amendment06IntegrityError(
                "Amendment 07 forensic TSV inventory is malformed"
            ) from exc

        source_records: dict[str, tuple[int, str]] = {}
        source_member_bytes = 0
        prefix_counts: Counter[str] = Counter()
        for row in source_rows:
            relative = str(row.get("member_path", ""))
            relative_path = Path(relative)
            source_path = str(row.get("source_path", ""))
            digest = str(row.get("sha256", ""))
            try:
                size = int(str(row.get("size_bytes", "")))
            except ValueError as exc:
                raise Amendment06IntegrityError(
                    "Amendment 07 forensic source inventory size differs"
                ) from exc
            if (
                not relative or relative_path.is_absolute() or ".." in relative_path.parts
                or relative_path.as_posix() != relative or relative in source_records
                or not Path(source_path).is_absolute() or size < 0
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise Amendment06IntegrityError(
                    "Amendment 07 forensic source inventory record differs"
                )
            source_records[relative] = (size, digest)
            source_member_bytes += size
            prefix_counts[relative_path.parts[0]] += 1
        source_canonical = "".join(
            f"{digest}\t{size}\t{relative}\n"
            for relative, (size, digest) in sorted(source_records.items())
        ).encode("utf-8")
        if not (
            len(source_records) == 61
            and source_member_bytes == 20_037_183
            and prefix_counts == {
                "a06_runtime": 10, "a07_prompt": 1, "cache": 2,
                "original_sources": 3, "target_run": 45,
            }
            and sha256_bytes(source_canonical)
            == A07_FORENSIC_SOURCE_INVENTORY_SHA256
            and set(expected_records)
            == {f"{root}/{relative}" for relative in source_records} | set(metadata_records)
            and all(
                expected_records[f"{root}/{relative}"] == record
                for relative, record in source_records.items()
            )
        ):
            raise Amendment06IntegrityError(
                "Amendment 07 forensic source member set/inventory differs"
            )

        cache_records: dict[str, tuple[int, str]] = {}
        cache_member_bytes = 0
        cache_root = EXPECTED_RUNTIME_ROOT / "full_population" / A07_TARGET_RUN_ID
        for row in cache_rows:
            relative = str(row.get("relative_path", ""))
            relative_path = Path(relative)
            absolute = str(row.get("absolute_path", ""))
            digest = str(row.get("sha256", ""))
            try:
                size = int(str(row.get("size_bytes", "")))
            except ValueError as exc:
                raise Amendment06IntegrityError(
                    "Amendment 07 forensic cache inventory size differs"
                ) from exc
            if (
                not relative or relative_path.is_absolute() or len(relative_path.parts) != 1
                or relative_path.as_posix() != relative or relative in cache_records
                or absolute != str(cache_root / relative) or size < 0
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise Amendment06IntegrityError(
                    "Amendment 07 forensic cache inventory record differs"
                )
            cache_records[relative] = (size, digest)
            cache_member_bytes += size
        archived_cache_manifest = strict_full_loads(
            archive.read(f"{root}/cache/cache_manifest.json")
        )
        archived_cache_records = {
            str(item["relative_path"]): (int(item["size_bytes"]), str(item["sha256"]))
            for item in archived_cache_manifest.get("cache_files", [])
            if isinstance(item, Mapping)
            and set(item) == {"relative_path", "size_bytes", "sha256"}
        }
        if not (
            len(cache_records) == 10
            and cache_member_bytes == 1_280_873_017
            and cache_records == archived_cache_records
            and cache_records.get("portable_imputer.json")
            == (4_333, A07_PORTABLE_IMPUTER_SHA256)
        ):
            raise Amendment06IntegrityError(
                "Amendment 07 forensic cache member inventory differs"
            )

        facts = _require_exact_keys(
            strict_full_loads(facts_bytes),
            {
                "schema", "status", "authorization", "a07_prompt_sha256",
                "target_run_id", "target_run_path", "failed_attempt_id",
                "journal_event_count", "journal_head_canonical_sha256",
                "cache_manifest_sha256", "cache_member_count", "cache_member_bytes",
                "source_member_count", "source_member_bytes",
                "canonical_source_inventory_sha256",
                "external_failure_review_zip_sha256",
                "external_audit_tabular_parse_performed",
            },
            label="Amendment 07 forensic facts",
        )
        if not (
            facts["schema"] == "amendment07_pre_mutation_forensic/v1"
            and facts["status"] == "PASS"
            and facts["authorization"] == A07_AUTHORIZATION
            and facts["a07_prompt_sha256"] == A07_PROMPT_SHA256
            and facts["target_run_id"] == A07_TARGET_RUN_ID
            and facts["target_run_path"] == str(A07_TARGET_RUN)
            and facts["failed_attempt_id"] == A07_FAILED_ATTEMPT_ID
            and facts["journal_event_count"] == 5
            and facts["journal_head_canonical_sha256"]
            == A07_PRECORRIGENDUM_JOURNAL_HEAD_SHA256
            and facts["cache_manifest_sha256"] == A07_CACHE_MANIFEST_SHA256
            and facts["cache_member_count"] == len(cache_records)
            and facts["cache_member_bytes"] == cache_member_bytes
            and facts["source_member_count"] == len(source_records)
            and facts["source_member_bytes"] == source_member_bytes
            and facts["canonical_source_inventory_sha256"]
            == A07_FORENSIC_SOURCE_INVENTORY_SHA256
            and facts["external_audit_tabular_parse_performed"] is False
        ):
            raise Amendment06IntegrityError("Amendment 07 forensic facts differ")

        original_prerun_name = f"{root}/a06_runtime/pre_run/final_prerun_evidence.json"
        original_prerun_bytes = archive.read(original_prerun_name)
        if not (
            expected_records.get(original_prerun_name)
            == (5_843, A07_ORIGINAL_PRERUN_SHA256)
            and sha256_bytes(original_prerun_bytes) == A07_ORIGINAL_PRERUN_SHA256
            and isinstance(strict_full_loads(original_prerun_bytes), Mapping)
        ):
            raise Amendment06IntegrityError(
                "Original A06 final pre-Run evidence is absent from the forensic bundle"
            )

    quarantine_source = (
        EXPECTED_RUNTIME_ROOT / "attempts" / A07_TARGET_RUN_ID / "TRAINING"
        / "full_new_reference" / A07_FAILED_ATTEMPT_ID
    )
    quarantine_destination = (
        A07_RUNTIME / "forensic/quarantine" / A07_TARGET_RUN_ID / "TRAINING"
        / "full_new_reference" / A07_FAILED_ATTEMPT_ID
    )
    intent = _require_exact_keys(
        strict_full_load_file(A07_FAILED_ATTEMPT_QUARANTINE_INTENT),
        {
            "schema", "status", "authorization", "a07_prompt_sha256",
            "target_run_id", "failed_attempt_id", "original_path",
            "quarantine_path", "member_count", "members",
        },
        label="Amendment 07 failed-attempt quarantine intent",
    )
    receipt = _require_exact_keys(
        strict_full_load_file(A07_FAILED_ATTEMPT_QUARANTINE_RECEIPT),
        {
            "schema", "status", "authorization", "a07_prompt_sha256",
            "target_run_id", "failed_attempt_id", "original_path",
            "quarantine_path", "member_count", "members", "intent_sha256",
            "source_absent", "destination_verified",
        },
        label="Amendment 07 failed-attempt quarantine receipt",
    )
    if not (
        intent["schema"] == "amendment07_failed_attempt_quarantine_intent/v1"
        and intent["status"] == "INTENT"
        and intent["authorization"] == A07_AUTHORIZATION
        and intent["a07_prompt_sha256"] == A07_PROMPT_SHA256
        and intent["target_run_id"] == A07_TARGET_RUN_ID
        and intent["failed_attempt_id"] == A07_FAILED_ATTEMPT_ID
        and intent["original_path"] == str(quarantine_source)
        and intent["quarantine_path"] == str(quarantine_destination)
        and intent["member_count"] == 0 and intent["members"] == []
        and receipt["schema"] == "amendment07_failed_attempt_quarantine_receipt/v1"
        and receipt["status"] == "PASS"
        and receipt["authorization"] == A07_AUTHORIZATION
        and receipt["a07_prompt_sha256"] == A07_PROMPT_SHA256
        and receipt["target_run_id"] == A07_TARGET_RUN_ID
        and receipt["failed_attempt_id"] == A07_FAILED_ATTEMPT_ID
        and receipt["original_path"] == str(quarantine_source)
        and receipt["quarantine_path"] == str(quarantine_destination)
        and receipt["intent_sha256"] == A07_FAILED_ATTEMPT_QUARANTINE_INTENT_SHA256
        and receipt["member_count"] == 0 and receipt["members"] == []
        and receipt["source_absent"] is True
        and receipt["destination_verified"] is True
        and not quarantine_source.exists() and not quarantine_source.is_symlink()
        and quarantine_destination.is_dir() and not quarantine_destination.is_symlink()
        and not any(quarantine_destination.iterdir())
    ):
        raise Amendment06IntegrityError("Amendment 07 failed-attempt quarantine differs")
    reconstruction = strict_full_load_file(A07_FORENSIC_RECONSTRUCTION)
    if not (
        reconstruction.get("schema") == "amendment07_pre_correction_reconstruction/v1"
        and reconstruction.get("status") == "PASS"
        and reconstruction.get("authorization") == A07_AUTHORIZATION
        and reconstruction.get("a07_prompt_sha256") == A07_PROMPT_SHA256
        and reconstruction.get("forensic", {}).get("bundle_sha256")
        == A07_FORENSIC_ZIP_SHA256
        and reconstruction.get("forensic", {}).get("verification_sha256")
        == A07_FORENSIC_VERIFICATION_SHA256
        and reconstruction.get("forensic", {}).get("quarantine_receipt_sha256")
        == A07_FAILED_ATTEMPT_QUARANTINE_RECEIPT_SHA256
    ):
        raise Amendment06IntegrityError("Amendment 07 forensic reconstruction differs")
    return {
        "status": "PASS",
        "forensic_zip_sha256": A07_FORENSIC_ZIP_SHA256,
        "forensic_verification_sha256": A07_FORENSIC_VERIFICATION_SHA256,
        "quarantine_receipt_sha256": A07_FAILED_ATTEMPT_QUARANTINE_RECEIPT_SHA256,
        "original_prerun_forensic_sha256": sha256_bytes(original_prerun_bytes),
    }


def _validate_amendment07_journal_prefix(run_dir: Path) -> dict[str, Any]:
    events = _read_journal(run_dir)
    if len(events) < 5:
        raise Amendment06IntegrityError("Amendment 07 Journal has fewer than five base events")
    prefix_bytes = b""
    for index, expected_raw in enumerate(A07_JOURNAL_RAW_SHA256):
        path = _journal_event_path(run_dir, index)
        if not _safe_regular(path, expected_raw):
            raise Amendment06IntegrityError(f"Amendment 07 Journal base event drifted: {index}")
        prefix_bytes += strict_full_canonical_json_bytes(events[index]) + b"\n"
    if not (
        sha256_bytes(prefix_bytes) == A07_JOURNAL_PREFIX_RAW_SHA256
        and sha256_bytes(strict_full_canonical_json_bytes(events[4]))
        == A07_PRECORRIGENDUM_JOURNAL_HEAD_SHA256
        and events[0].get("event") == "CACHE_GENERATION_INTENT"
        and events[1].get("event") == "CACHE_GENERATION"
        and events[1].get("status") == "PASS"
        and events[2].get("event") == "TRAINING_ATTEMPT_STARTED"
        and events[3].get("event") == "SCIENTIFIC_FIT_CALLED"
        and events[4].get("event") == "TRAINING_ATTEMPT_FAILED"
        and all(event.get("variant_id") == "full_new_reference" for event in events[2:5])
        and all(event.get("attempt_id") == A07_FAILED_ATTEMPT_ID for event in events[2:5])
        and events[4].get("error_type") == "Amendment06IntegrityError"
        and events[4].get("error") == "GPU identity has no name"
    ):
        raise Amendment06IntegrityError("Amendment 07 failed-boundary Journal differs")
    return {
        "status": "PASS", "event_count": len(events),
        "base_event_count": 5,
        "pre_corrigendum_head_sha256": A07_PRECORRIGENDUM_JOURNAL_HEAD_SHA256,
    }


def _validate_preserved_target_cache_files(run_dir: Path) -> dict[str, Any]:
    destination = EXPECTED_RUNTIME_ROOT / "full_population" / run_dir.name
    runtime_manifest = destination / "cache_manifest.json"
    run_manifest = run_dir / "provenance/frozen_full_resampling_manifest.json"
    run_imputer = run_dir / "provenance/frozen_full_imputer.json"
    if not (
        destination.is_dir() and not destination.is_symlink()
        and _safe_regular(runtime_manifest, A07_CACHE_MANIFEST_SHA256)
        and _safe_regular(run_manifest, A07_CACHE_MANIFEST_SHA256)
        and runtime_manifest.read_bytes() == run_manifest.read_bytes()
        and _safe_regular(run_imputer, A07_PORTABLE_IMPUTER_SHA256)
    ):
        raise Amendment06IntegrityError("Amendment 07 frozen-cache pins drifted")
    manifest = strict_full_load_file(runtime_manifest)
    records = manifest.get("cache_files")
    if not isinstance(records, list) or len(records) != 10:
        raise Amendment06IntegrityError("Amendment 07 cache member records differ")
    expected_names: list[str] = []
    total_bytes = 0
    for item in records:
        record = _require_exact_keys(
            item, {"relative_path", "size_bytes", "sha256"},
            label="Amendment 07 cache member",
        )
        relative = str(record["relative_path"])
        relative_path = Path(relative)
        size = record["size_bytes"]
        digest = str(record["sha256"])
        if (
            not relative or relative_path.is_absolute() or ".." in relative_path.parts
            or relative_path.as_posix() != relative or relative in expected_names
            or isinstance(size, bool) or not isinstance(size, int) or size < 0
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise Amendment06IntegrityError("Amendment 07 cache member schema differs")
        path = destination / relative
        if not (_safe_regular(path, digest) and path.stat().st_size == size):
            raise Amendment06IntegrityError(f"Amendment 07 cache member drifted: {relative}")
        expected_names.append(relative)
        total_bytes += size
    actual_names = sorted(
        path.name for path in destination.iterdir() if path.name != "cache_manifest.json"
    )
    if actual_names != sorted(expected_names):
        raise Amendment06IntegrityError("Amendment 07 cache member set differs")
    post = manifest.get("stage_counts", {}).get("post_safe_smote", {})
    if not (
        manifest.get("status") == "PASS"
        and manifest.get("unique_realizations") == 1
        and manifest.get("augmentation_generation_count") == 1
        and manifest.get("safe_smote_generation_count") == 1
        and manifest.get("same_run_resume_cache") is True
        and post.get("rows") == POST_SMOTE_ROWS
        and manifest.get("frozen_population_sha256") == A07_FROZEN_POPULATION_SHA256
        and manifest.get("imputer_sha256") == A07_PORTABLE_IMPUTER_SHA256
    ):
        raise Amendment06IntegrityError("Amendment 07 cache semantics differ")
    return {
        "status": "PASS", "cache_path": str(destination),
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        "frozen_population_sha256": A07_FROZEN_POPULATION_SHA256,
        "portable_imputer_sha256": A07_PORTABLE_IMPUTER_SHA256,
        "cache_member_count": 10, "cache_member_bytes": total_bytes,
        "unique_realizations": 1,
    }


def _validate_amendment07_cache_files(run_dir: Path) -> dict[str, Any]:
    return _validate_preserved_target_cache_files(run_dir)


def _validate_amendment07_ledger_prefix() -> dict[str, Any]:
    if not _safe_regular(AUTHORIZED_CHANGE_LEDGER):
        raise Amendment06IntegrityError("Amendment 06 source ledger is absent/unsafe")
    raw = AUTHORIZED_CHANGE_LEDGER.read_bytes()
    if (
        len(raw) < A07_ORIGINAL_LEDGER_SIZE_BYTES
        or sha256_bytes(raw[:A07_ORIGINAL_LEDGER_SIZE_BYTES])
        != A07_ORIGINAL_LEDGER_SHA256
    ):
        raise Amendment06IntegrityError("Original Amendment 06 ledger prefix drifted")
    return {
        "status": "PASS", "original_prefix_sha256": A07_ORIGINAL_LEDGER_SHA256,
        "current_ledger_sha256": sha256_bytes(raw), "current_size_bytes": len(raw),
    }


def validate_amendment07_precorrection(
    run_dir: Path = A07_TARGET_RUN,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    runs = existing_full_runs()
    if run_dir != A07_TARGET_RUN.resolve() or runs != [run_dir]:
        raise Amendment06IntegrityError(
            f"Amendment 07 requires exactly the preserved target Run: {runs}"
        )
    base_rows = _validate_amendment07_base_history(run_dir)
    if not (
        _safe_regular(_run_intent_path(), A07_A06_INTENT_SHA256)
        and _safe_regular(_run_receipt_path(), A07_A06_RECEIPT_SHA256)
        and _safe_regular(A06_RUNTIME / "live/full_run.log", A07_ORIGINAL_FULL_LOG_SHA256)
        and _safe_regular(
            run_dir / "provenance/cuda_active_fit_probe.json",
            A07_IMMUTABLE_CUDA_PROBE_SHA256,
        )
        and _safe_regular(EXTERNAL_AUDIT, EXTERNAL_AUDIT_SHA256)
    ):
        raise Amendment06IntegrityError("Amendment 07 immutable pre-correction pin drifted")
    if _current_run_state(run_dir) != "FULL_IN_PROGRESS":
        raise Amendment06IntegrityError("Amendment 07 target is not FULL_IN_PROGRESS")
    allowed = set(A07_CHANGED_SCIENTIFIC_RELATIVES)
    changed: list[str] = []
    for row in base_rows:
        relative = str(row["relative_path"])
        live = STUDY_ROOT / relative
        if not _safe_regular(live):
            raise Amendment06IntegrityError(f"Amendment 07 live source is unsafe: {relative}")
        differs = (
            live.stat().st_size != int(row["size_bytes"])
            or sha256_file(live) != str(row["sha256"])
        )
        if differs:
            changed.append(relative)
            if relative not in allowed:
                raise Amendment06IntegrityError(
                    f"Unauthorized Amendment 07 scientific change: {relative}"
                )
    if changed and tuple(changed) != A07_CHANGED_SCIENTIFIC_RELATIVES:
        raise Amendment06IntegrityError("Amendment 07 correction is only partially installed")
    if sha256_file(STUDY_ROOT / "code/amendment_core.py") != A07_UNCHANGED_CORE_SHA256:
        raise Amendment06IntegrityError("Amendment core changed under Amendment 07")
    journal = _validate_amendment07_journal_prefix(run_dir)
    if len(_read_journal(run_dir)) not in {5, 6} and not (
        run_dir / "provenance/amendment07_overlay_install_receipt.json"
    ).exists():
        raise Amendment06IntegrityError("Unexpected Journal event before Amendment 07 install")
    forbidden = [
        *run_dir.glob("models/*/model.ubj"),
        *run_dir.glob("models/*/training_completion_manifest.json"),
        *run_dir.glob("predictions/*"), *run_dir.glob("metrics/*"),
    ]
    forbidden.extend(
        run_dir / relative for relative in (
            "provenance/full_models_frozen_manifest.json",
            "provenance/external_audit_exclusion_and_opening.json",
            "provenance/reporting_completion_manifest.json",
            "OUTPUT_MANIFEST_FINAL.tsv",
        ) if (run_dir / relative).exists()
    )
    if forbidden:
        raise Amendment06IntegrityError(
            f"Amendment 07 failed boundary has published scientific artifacts: {forbidden}"
        )
    external_review = RESULTS_ROOT / "amendment06_gpu_name_failure_review_20260820_164007.zip"
    if external_review.exists() and not _safe_regular(
        external_review,
        "03abe6480fe66aa651a0666994e2326d2bb333f1892819834273b37b311a053f",
    ):
        raise Amendment06IntegrityError("Local external failure-review ZIP differs")
    forensic = _validate_amendment07_forensic_snapshot()
    cache = _validate_amendment07_cache_files(run_dir)
    ledger = _validate_amendment07_ledger_prefix()
    return {
        "status": "PASS", "authorization": A07_AUTHORIZATION,
        "prompt_sha256": A07_PROMPT_SHA256, "target_run_id": run_dir.name,
        "new_full_run_ids": 0, "base_history": "PASS",
        "changed_scientific_paths": changed,
        "journal": journal, "cache": cache, "forensic": forensic,
        "ledger_prefix": ledger, "external_audit_tabular_parse_calls": 0,
    }


def validate_amendment07_cuda_producer_consumer_probe(
    path: Path,
) -> dict[str, Any]:
    payload = _require_exact_keys(
        strict_full_load_file(path),
        {
            "schema", "status", "authorization", "prompt_sha256",
            "target_run_id", "recorded_at", "xgboost_version",
            "scikit_learn_version", "use_cuda", "gpu_identity", "gpu_name",
            "producer_consumer_status", "fit_save_reload_predict_status",
            "four_way_bit_exact_max_abs_diff", "model_bytes_unchanged",
            "cpu_fallback_detected", "fallback_warnings",
            "active_reloaded_cuda_probe",
        },
        label="Amendment 07 CUDA producer-consumer probe",
    )
    gpu_identity = payload["gpu_identity"]
    parsed_name = _gpu_name(gpu_identity)
    active = payload["active_reloaded_cuda_probe"]
    if not (
        payload["schema"] == "amendment07_cuda_producer_consumer_probe/v1"
        and payload["status"] == "PASS"
        and payload["authorization"] == A07_AUTHORIZATION
        and payload["prompt_sha256"] == A07_PROMPT_SHA256
        and payload["target_run_id"] == A07_TARGET_RUN_ID
        and isinstance(payload["recorded_at"], str) and payload["recorded_at"]
        and payload["xgboost_version"] == "2.1.1"
        and payload["scikit_learn_version"] == "1.7.2"
        and payload["use_cuda"] is True
        and payload["gpu_name"] == parsed_name
        and "RTX 4090" in parsed_name
        and payload["producer_consumer_status"] == "PASS"
        and payload["fit_save_reload_predict_status"] == "PASS"
        and payload["four_way_bit_exact_max_abs_diff"] == 0.0
        and payload["model_bytes_unchanged"] is True
        and payload["cpu_fallback_detected"] is False
        and payload["fallback_warnings"] == []
        and isinstance(active, Mapping)
        and active.get("status") == "PASS"
        and "cuda" in str(active.get("device", "")).lower()
        and active.get("warnings") == []
    ):
        raise Amendment06IntegrityError("Amendment 07 CUDA producer-consumer probe failed")
    return dict(payload)


def run_amendment07_cuda_producer_consumer_probe(
    path: Path = A07_RUNTIME / "pre_run/amendment07_cuda_producer_consumer_probe.json",
) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        return validate_amendment07_cuda_producer_consumer_probe(path)
    if not (
        Path(sys.executable).resolve() == CUDA_PYTHON.resolve()
        and Path(sys.prefix).resolve() == CUDA_ENV.resolve()
        and xgb.__version__ == "2.1.1"
        and sklearn.__version__ == "1.7.2"
        and xgb.build_info().get("USE_CUDA") is True
    ):
        raise Amendment06IntegrityError("Amendment 07 CUDA probe environment differs")
    identity = core.gpu_identity_metadata()
    gpu_name = _gpu_name(identity)
    if "RTX 4090" not in gpu_name:
        raise Amendment06IntegrityError("Amendment 07 CUDA probe GPU is not RTX 4090")
    rng = np.random.default_rng(20260820)
    matrix = np.ascontiguousarray(rng.normal(size=(128, 4)), dtype=np.float32)
    labels = (matrix[:, 0] + 0.4 * matrix[:, 1] > 0.0).astype(np.int8)
    classifier = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        n_estimators=24, learning_rate=0.15, max_depth=3,
        subsample=1.0, colsample_bytree=1.0, tree_method="hist",
        device="cuda", random_state=42, n_jobs=2, verbosity=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier.fit(matrix, labels, eval_set=[(matrix, labels)], verbose=False)
    warning_text = [str(item.message) for item in caught]
    fallback = _fallback_warnings(warning_text)
    configuration = strict_full_loads(classifier.get_booster().save_config())
    if fallback or "cuda" not in json.dumps(configuration).lower():
        raise Amendment06IntegrityError("Amendment 07 CUDA probe fit/fallback gate failed")
    features = ("probe_0", "probe_1", "probe_2", "probe_3")
    feature_sha256 = _feature_list_sha256(features)
    model_compat.embed_binary_classifier_metadata(
        classifier, expected_feature_count=4,
        feature_list_sha256=feature_sha256,
    )
    work = A07_RUNTIME / "probe_work" / uuid.uuid4().hex
    work.mkdir(parents=True, exist_ok=False)
    model_path = work / "amendment07_cuda_probe.ubj"
    try:
        classifier.save_model(model_path)
        before = sha256_file(model_path)
        parity = model_compat.four_way_binary_reload_parity(
            classifier, model_path, matrix,
            expected_feature_count=4,
            expected_feature_list_sha256=feature_sha256,
            device="cuda",
        )
        differences = parity.evidence.get("maximum_absolute_difference", {})
        maximum = max((float(value) for value in differences.values()), default=math.inf)
        active = core._active_reloaded_booster_cuda_probe(
            parity.reloaded_booster, 4,
        )
        after = sha256_file(model_path)
        payload = {
            "schema": "amendment07_cuda_producer_consumer_probe/v1",
            "status": "PASS", "authorization": A07_AUTHORIZATION,
            "prompt_sha256": A07_PROMPT_SHA256,
            "target_run_id": A07_TARGET_RUN_ID,
            "recorded_at": _now(), "xgboost_version": xgb.__version__,
            "scikit_learn_version": sklearn.__version__,
            "use_cuda": True, "gpu_identity": identity, "gpu_name": gpu_name,
            "producer_consumer_status": "PASS",
            "fit_save_reload_predict_status": "PASS",
            "four_way_bit_exact_max_abs_diff": maximum,
            "model_bytes_unchanged": before == after,
            "cpu_fallback_detected": False,
            "fallback_warnings": fallback,
            "active_reloaded_cuda_probe": active,
        }
        validate_amendment07_cuda_producer_consumer_probe_bytes = strict_full_json_bytes(
            payload
        )
        publish_bytes_no_clobber(path, validate_amendment07_cuda_producer_consumer_probe_bytes)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return validate_amendment07_cuda_producer_consumer_probe(path)


def _amendment07_external_authorization_payload(
    *, recorded_at: str, forensic: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "amendment07_external_authorization/v1",
        "status": "PASS", "authorization": A07_AUTHORIZATION,
        "prompt_sha256": A07_PROMPT_SHA256,
        "target_run_id": A07_TARGET_RUN_ID,
        "target_run_path": str(A07_TARGET_RUN),
        "recorded_at": recorded_at,
        "new_preflight_runs_authorized": 0,
        "new_smoke_runs_authorized": 0,
        "new_full_run_ids_authorized": 0,
        "target_run_continuation_authorized": True,
        "corrigendum_installations_authorized": 1,
        "frozen_population_reuse_required": True,
        "new_resampling_realizations_authorized": 0,
        "stability_or_multiseed_authorized": False,
        "threshold_tuning_authorized": False,
        "full_reference_determinism_retrain_authorized": False,
        "legacy_full_path_authorized": False,
        "external_audit_use_for_fitting": False,
        "base_run_identity_sha256": A07_BASE_IDENTITY_SHA256,
        "external_failure_review_zip_sha256": (
            "03abe6480fe66aa651a0666994e2326d2bb333f1892819834273b37b311a053f"
        ),
        "forensic_snapshot_sha256": forensic["forensic_zip_sha256"],
    }


def _amendment07_failed_forensics_payload(
    *, forensic: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "amendment07_failed_attempt_forensics/v1",
        "status": "PASS", "authorization": A07_AUTHORIZATION,
        "prompt_sha256": A07_PROMPT_SHA256,
        "target_run_id": A07_TARGET_RUN_ID,
        "failed_attempt_id": A07_FAILED_ATTEMPT_ID,
        "failed_event_index": 4,
        "failed_event_raw_sha256": A07_FAILED_JOURNAL_EVENT_RAW_SHA256,
        "pre_corrigendum_journal_event_count": 5,
        "pre_corrigendum_journal_head_sha256": A07_PRECORRIGENDUM_JOURNAL_HEAD_SHA256,
        "error_type": "Amendment06IntegrityError",
        "error": "GPU identity has no name",
        "variant_id": "full_new_reference",
        "classifier_fit_reached_cuda": True,
        "model_save_reached": False,
        "failed_attempt_publication_count": 0,
        "published_model_count": 0,
        "published_prediction_count": 0,
        "published_metric_count": 0,
        "external_audit_tabular_parse_calls": 0,
        "forensic_snapshot_sha256": forensic["forensic_zip_sha256"],
        "forensic_verification_sha256": forensic["forensic_verification_sha256"],
        "failed_attempt_quarantine_receipt_sha256": forensic[
            "quarantine_receipt_sha256"
        ],
    }


def _amendment07_corrigendum_payload(
    *, run_dir: Path, effective_manifest_sha256: str,
    current_chain: Mapping[str, Any], current_prerun_sha256: str,
    probe_sha256: str, forensic: Mapping[str, Any], generated_at: str,
) -> dict[str, Any]:
    base_rows = _validate_amendment07_base_history(run_dir)
    by_relative = {str(row["relative_path"]): row for row in base_rows}
    current_full = STUDY_ROOT / "code/amendment06_full.py"
    current_test = STUDY_ROOT / "tests/test_amendment06.py"
    return {
        "schema": "amendment07_code_corrigendum/v1",
        "status": "PASS", "authorization": A07_AUTHORIZATION,
        "prompt_sha256": A07_PROMPT_SHA256,
        "target_run_id": run_dir.name, "target_run_path": str(run_dir),
        "generated_at": generated_at,
        "original_run_identity_sha256": A07_BASE_IDENTITY_SHA256,
        "base_scientific_execution_code_manifest_sha256": A07_BASE_SCIENTIFIC_MANIFEST_SHA256,
        "original_amendment06_code_lineage_sha256": A07_BASE_CODE_LINEAGE_SHA256,
        "old_amendment06_full_py_sha256": A07_OLD_FULL_SHA256,
        "new_amendment06_full_py_sha256": sha256_file(current_full),
        "old_test_amendment06_py_sha256": A07_OLD_TEST_SHA256,
        "new_test_amendment06_py_sha256": sha256_file(current_test),
        "unchanged_amendment_core_py_sha256": A07_UNCHANGED_CORE_SHA256,
        "effective_scientific_execution_code_manifest_sha256": effective_manifest_sha256,
        "changed_scientific_paths": list(A07_CHANGED_SCIENTIFIC_RELATIVES),
        "unified_diff_sha256": sha256_bytes(_amendment07_unified_diff_bytes(run_dir)),
        "original_a06_authorized_change_ledger_sha256": A07_ORIGINAL_LEDGER_SHA256,
        "current_a06_authorized_change_ledger_sha256": current_chain["ledger_sha256"],
        "original_a06_final_prerun_evidence_sha256": A07_ORIGINAL_PRERUN_SHA256,
        "current_a06_final_prerun_evidence_sha256": current_prerun_sha256,
        "pre_corrigendum_ownership_journal_event_count": 5,
        "pre_corrigendum_ownership_journal_head_sha256": A07_PRECORRIGENDUM_JOURNAL_HEAD_SHA256,
        "failed_attempt_id": A07_FAILED_ATTEMPT_ID,
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        "frozen_population_sha256": A07_FROZEN_POPULATION_SHA256,
        "portable_imputer_sha256": A07_PORTABLE_IMPUTER_SHA256,
        "immutable_target_cuda_probe_sha256": A07_IMMUTABLE_CUDA_PROBE_SHA256,
        "amendment07_cuda_producer_consumer_probe_sha256": probe_sha256,
        "forensic_snapshot_sha256": forensic["forensic_zip_sha256"],
        "forensic_verification_sha256": forensic["forensic_verification_sha256"],
        "base_snapshots_preserved": True,
        "conditional_package_paths_are_not_scientific_manifest_members": True,
        "old_source_manifest_rows": {
            relative: {
                "size_bytes": int(by_relative[relative]["size_bytes"]),
                "sha256": str(by_relative[relative]["sha256"]),
            }
            for relative in A07_CHANGED_SCIENTIFIC_RELATIVES
        },
    }


def _amendment07_event5_payload(
    *, run_dir: Path, recorded_at: str, effective_manifest_sha256: str,
    corrigendum_sha256: str,
) -> dict[str, Any]:
    return {
        "event_index": 5,
        "event": "AMENDMENT07_CORRIGENDUM_INSTALLED",
        "run_id": run_dir.name,
        "previous_event_sha256": A07_PRECORRIGENDUM_JOURNAL_HEAD_SHA256,
        "recorded_at": recorded_at,
        "status": "PASS", "authorization": A07_AUTHORIZATION,
        "prompt_sha256": A07_PROMPT_SHA256,
        "effective_scientific_execution_code_manifest_sha256": effective_manifest_sha256,
        "amendment07_code_corrigendum_sha256": corrigendum_sha256,
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
    }


def _amendment07_resume_precheck_payload(
    *, run_dir: Path, generated_at: str, effective_manifest_sha256: str,
    corrigendum_sha256: str, event5: Mapping[str, Any],
    cache: Mapping[str, Any], chain: Mapping[str, Any], prerun_sha256: str,
    probe_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "amendment07_resume_precheck/v1",
        "status": "PASS", "authorization": A07_AUTHORIZATION,
        "prompt_sha256": A07_PROMPT_SHA256,
        "target_run_id": run_dir.name, "target_run_path": str(run_dir),
        "generated_at": generated_at,
        "exact_full_run_count": 1,
        "run_state": "FULL_IN_PROGRESS",
        "base_scientific_execution_code_manifest_sha256": A07_BASE_SCIENTIFIC_MANIFEST_SHA256,
        "effective_scientific_execution_code_manifest_sha256": effective_manifest_sha256,
        "amendment07_code_corrigendum_sha256": corrigendum_sha256,
        "corrigendum_journal_event_index": 5,
        "corrigendum_journal_event_sha256": sha256_bytes(
            strict_full_json_bytes(event5)
        ),
        "corrigendum_journal_head_sha256": sha256_bytes(
            strict_full_canonical_json_bytes(event5)
        ),
        "journal_event_count_after_corrigendum": 6,
        "failed_attempt_id_preserved": A07_FAILED_ATTEMPT_ID,
        "failed_attempt_publications": 0,
        "cache_manifest_sha256": cache["cache_manifest_sha256"],
        "frozen_population_sha256": cache["frozen_population_sha256"],
        "portable_imputer_sha256": cache["portable_imputer_sha256"],
        "cache_members_rehashed": cache["cache_member_count"],
        "unique_full_resampling_realizations": cache["unique_realizations"],
        "cache_generation_events": 1,
        "cache_load_events_before_resume": 0,
        "current_a06_authorized_change_ledger_sha256": chain["ledger_sha256"],
        "current_a06_final_prerun_evidence_sha256": prerun_sha256,
        "amendment07_cuda_producer_consumer_probe_sha256": probe_sha256,
        "accepted_preflight_validation": "PASS_SEALED_ONLY_NO_LIVE_AUDIT_PARSE",
        "accepted_smoke_validation": "PASS_PRODUCTION_VALIDATOR",
        "external_audit_tabular_parse_calls": 0,
        "new_full_run_ids_created": 0,
    }


def _overlay_record(relative: str, data: bytes) -> dict[str, Any]:
    return {
        "relative_path": relative,
        "size_bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def _build_amendment07_overlay_material(
    *, run_dir: Path, probe_path: Path, generated_at: str,
    chain: Mapping[str, Any], prerun_sha256: str,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    forensic = _validate_amendment07_forensic_snapshot()
    cache = _validate_amendment07_cache_files(run_dir)
    probe = validate_amendment07_cuda_producer_consumer_probe(probe_path)
    probe_bytes = strict_full_json_bytes(probe)
    effective_bytes = _effective_amendment07_manifest_bytes(run_dir)
    effective_sha256 = sha256_bytes(effective_bytes)
    authorization = _amendment07_external_authorization_payload(
        recorded_at=generated_at, forensic=forensic,
    )
    failed = _amendment07_failed_forensics_payload(forensic=forensic)
    corrigendum = _amendment07_corrigendum_payload(
        run_dir=run_dir, effective_manifest_sha256=effective_sha256,
        current_chain=chain, current_prerun_sha256=prerun_sha256,
        probe_sha256=sha256_bytes(probe_bytes), forensic=forensic,
        generated_at=generated_at,
    )
    corrigendum_bytes = strict_full_json_bytes(corrigendum)
    corrigendum_sha256 = sha256_bytes(corrigendum_bytes)
    event5 = _amendment07_event5_payload(
        run_dir=run_dir, recorded_at=generated_at,
        effective_manifest_sha256=effective_sha256,
        corrigendum_sha256=corrigendum_sha256,
    )
    precheck = _amendment07_resume_precheck_payload(
        run_dir=run_dir, generated_at=generated_at,
        effective_manifest_sha256=effective_sha256,
        corrigendum_sha256=corrigendum_sha256, event5=event5,
        cache=cache, chain=chain, prerun_sha256=prerun_sha256,
        probe_sha256=sha256_bytes(probe_bytes),
    )
    material = {
        "provenance/amendment07_external_authorization.json": strict_full_json_bytes(
            authorization
        ),
        "provenance/amendment07_prompt_snapshot.txt": A07_PROMPT_PATH.read_bytes(),
        "provenance/amendment07_code_corrigendum.json": corrigendum_bytes,
        "provenance/scientific_execution_code_manifest_effective_amendment07.tsv": effective_bytes,
        "provenance/amendment07_failed_attempt_forensics.json": strict_full_json_bytes(
            failed
        ),
        "provenance/amendment07_resume_precheck.json": strict_full_json_bytes(precheck),
        "provenance/amendment07_cuda_producer_consumer_probe.json": probe_bytes,
        "provenance/code_snapshot_amendment07/amendment06_full.py": (
            STUDY_ROOT / "code/amendment06_full.py"
        ).read_bytes(),
        "provenance/tests_snapshot_amendment07/test_amendment06.py": (
            STUDY_ROOT / "tests/test_amendment06.py"
        ).read_bytes(),
    }
    expected = set(A07_OVERLAY_RUN_RELATIVES) - {
        "provenance/amendment07_overlay_install_receipt.json"
    }
    if set(material) != expected:
        raise Amendment06IntegrityError("Internal Amendment 07 material set differs")
    return material, {
        "forensic": forensic, "cache": cache, "probe": probe,
        "effective_manifest_sha256": effective_sha256,
        "corrigendum_sha256": corrigendum_sha256, "event5": event5,
    }


def _amendment07_install_intent_payload(
    *, generated_at: str, transaction_id: str,
    material: Mapping[str, bytes], bindings: Mapping[str, Any],
    chain: Mapping[str, Any], prerun_sha256: str,
) -> dict[str, Any]:
    event5 = bindings["event5"]
    records = [_overlay_record(relative, material[relative]) for relative in sorted(material)]
    return {
        "schema": "amendment07_overlay_install_intent/v1",
        "status": "INTENT", "authorization": A07_AUTHORIZATION,
        "prompt_sha256": A07_PROMPT_SHA256,
        "target_run_id": A07_TARGET_RUN_ID,
        "transaction_id": transaction_id, "generated_at": generated_at,
        "corrigendum_installations_authorized": 1,
        "declared_member_count": len(records), "declared_members": records,
        "receipt_relative_path": "provenance/amendment07_overlay_install_receipt.json",
        "projected_corrigendum_journal_event": dict(event5),
        "projected_corrigendum_journal_event_sha256": sha256_bytes(
            strict_full_json_bytes(event5)
        ),
        "projected_corrigendum_journal_head_sha256": sha256_bytes(
            strict_full_canonical_json_bytes(event5)
        ),
        "base_scientific_execution_code_manifest_sha256": A07_BASE_SCIENTIFIC_MANIFEST_SHA256,
        "effective_scientific_execution_code_manifest_sha256": bindings[
            "effective_manifest_sha256"
        ],
        "amendment07_code_corrigendum_sha256": bindings["corrigendum_sha256"],
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        "current_a06_authorized_change_ledger_sha256": chain["ledger_sha256"],
        "current_a06_final_prerun_evidence_sha256": prerun_sha256,
    }


def _amendment07_install_receipt_payload(
    *, intent_sha256: str, generated_at: str,
    material: Mapping[str, bytes], bindings: Mapping[str, Any],
) -> dict[str, Any]:
    event5 = bindings["event5"]
    records = [_overlay_record(relative, material[relative]) for relative in sorted(material)]
    return {
        "schema": "amendment07_overlay_install_receipt/v1",
        "status": "PASS", "authorization": A07_AUTHORIZATION,
        "prompt_sha256": A07_PROMPT_SHA256,
        "target_run_id": A07_TARGET_RUN_ID,
        "installation_intent_sha256": intent_sha256,
        "corrigendum_installations": 1,
        "member_count": len(records), "members": records,
        "corrigendum_journal_event_index": 5,
        "corrigendum_journal_event_sha256": sha256_bytes(
            strict_full_json_bytes(event5)
        ),
        "corrigendum_journal_head_sha256": sha256_bytes(
            strict_full_canonical_json_bytes(event5)
        ),
        "base_scientific_execution_code_manifest_sha256": A07_BASE_SCIENTIFIC_MANIFEST_SHA256,
        "effective_scientific_execution_code_manifest_sha256": bindings[
            "effective_manifest_sha256"
        ],
        "amendment07_code_corrigendum_sha256": bindings["corrigendum_sha256"],
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        "frozen_population_sha256": A07_FROZEN_POPULATION_SHA256,
        "installed_at": generated_at,
    }


def _quarantine_amendment07_overlay_mismatch(
    *, path: Path, relative: str, transaction_id: str,
) -> None:
    quarantine_root = (
        A07_RUNTIME / "forensic/quarantine/overlay_install_mismatch" / transaction_id
    )
    destination = quarantine_root / relative
    intent_path = (
        A07_RUNTIME / "transactions/overlay_mismatch"
        / f"{transaction_id}_{sha256_bytes(relative.encode())}.intent.json"
    )
    mode = os.lstat(path).st_mode
    observed_sha256 = sha256_file(path) if stat.S_ISREG(mode) else None
    observed_size = path.stat().st_size if stat.S_ISREG(mode) else None
    payload = {
        "schema": "amendment07_overlay_mismatch_quarantine_intent/v1",
        "status": "INTENT", "authorization": A07_AUTHORIZATION,
        "prompt_sha256": A07_PROMPT_SHA256,
        "target_run_id": A07_TARGET_RUN_ID,
        "transaction_id": transaction_id, "relative_path": relative,
        "source_path": str(path), "quarantine_path": str(destination),
        "observed_size_bytes": observed_size,
        "observed_sha256": observed_sha256,
    }
    if intent_path.exists():
        if strict_full_load_file(intent_path) != payload:
            raise Amendment06IntegrityError("A07 overlay mismatch intent drifted")
    else:
        publish_strict_json_no_clobber(intent_path, payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _rename_directory_no_clobber(path, destination)
    receipt = {**payload, "schema": "amendment07_overlay_mismatch_quarantine_receipt/v1", "status": "PASS"}
    publish_strict_json_no_clobber(intent_path.with_suffix(".receipt.json"), receipt)


def _assert_no_amendment07_overlay_mismatch() -> None:
    root = A07_RUNTIME / "transactions/overlay_mismatch"
    if root.exists():
        if root.is_symlink() or not root.is_dir() or any(root.iterdir()):
            raise Amendment06IntegrityError(
                "A quarantined Amendment 07 overlay mismatch permanently blocks Resume"
            )


def _publish_or_validate_amendment07_member(
    *, run_dir: Path, relative: str, data: bytes, transaction_id: str,
) -> None:
    path = run_dir / relative
    if path.exists() or path.is_symlink():
        if _safe_regular(path, sha256_bytes(data)) and path.stat().st_size == len(data):
            return
        _quarantine_amendment07_overlay_mismatch(
            path=path, relative=relative, transaction_id=transaction_id,
        )
        raise Amendment06IntegrityError(
            f"Mismatched Amendment 07 member was quarantined: {relative}"
        )
    publish_bytes_no_clobber(path, data)


def install_amendment07_overlay(
    *, probe_path: Path, run_dir: Path = A07_TARGET_RUN,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    _assert_no_amendment07_overlay_mismatch()
    receipt_path = run_dir / "provenance/amendment07_overlay_install_receipt.json"
    if receipt_path.exists():
        return validate_amendment07_overlay(run_dir)
    validate_amendment07_precorrection(run_dir)
    chain = validate_authorized_change_chain()
    events = _load_concatenated_json(AUTHORIZED_CHANGE_LEDGER)
    if not events or events[-1].get("corrigendum_authorization") != A07_AUTHORIZATION:
        raise Amendment06IntegrityError("Amendment 07 source-ledger event is not installed")
    _validate_amendment07_ledger_event(events[-1], require_latest_tree=True)
    prerun = validate_final_prerun_evidence(chain)
    prerun_sha256 = str(prerun["sha256"])
    probe_path = Path(probe_path).resolve()

    if A07_INSTALL_INTENT.exists():
        intent = strict_full_load_file(A07_INSTALL_INTENT)
        generated_at = str(intent.get("generated_at", ""))
        transaction_id = str(intent.get("transaction_id", ""))
        if not generated_at or not re.fullmatch(r"[0-9a-f]{32}", transaction_id):
            raise Amendment06IntegrityError("Amendment 07 installation intent identity differs")
    else:
        generated_at = _now()
        transaction_id = uuid.uuid4().hex

    material, bindings = _build_amendment07_overlay_material(
        run_dir=run_dir, probe_path=probe_path, generated_at=generated_at,
        chain=chain, prerun_sha256=prerun_sha256,
    )
    expected_intent = _amendment07_install_intent_payload(
        generated_at=generated_at, transaction_id=transaction_id,
        material=material, bindings=bindings, chain=chain,
        prerun_sha256=prerun_sha256,
    )
    if A07_INSTALL_INTENT.exists():
        if strict_full_load_file(A07_INSTALL_INTENT) != expected_intent:
            raise Amendment06IntegrityError("Amendment 07 installation intent drifted")
    else:
        publish_strict_json_no_clobber(A07_INSTALL_INTENT, expected_intent)
    intent_sha256 = sha256_file(A07_INSTALL_INTENT)

    staging = A07_RUNTIME / "install_staging" / transaction_id
    staging.mkdir(parents=True, exist_ok=True)
    if staging.is_symlink():
        raise Amendment06IntegrityError("Amendment 07 installation staging is unsafe")
    for relative, data in material.items():
        staged = staging / relative
        if staged.exists():
            if not _safe_regular(staged, sha256_bytes(data)):
                raise Amendment06IntegrityError(
                    f"Amendment 07 staged member drifted: {relative}"
                )
        else:
            publish_bytes_no_clobber(staged, data)

    pre_event_relatives = set(material) - {"provenance/amendment07_resume_precheck.json"}
    for relative in sorted(pre_event_relatives):
        _publish_or_validate_amendment07_member(
            run_dir=run_dir, relative=relative, data=material[relative],
            transaction_id=transaction_id,
        )

    journal = _read_journal(run_dir)
    projected_event5 = bindings["event5"]
    if len(journal) == 5:
        event_fields = dict(projected_event5)
        event_name = str(event_fields.pop("event"))
        for key in ("event_index", "run_id", "previous_event_sha256"):
            event_fields.pop(key)
        observed_event5 = _journal_event(run_dir, event_name, **event_fields)
    elif len(journal) >= 6:
        observed_event5 = journal[5]
    else:
        raise Amendment06IntegrityError("Amendment 07 Journal prefix disappeared")
    if observed_event5 != projected_event5:
        raise Amendment06IntegrityError("Amendment 07 corrigendum Journal event differs")
    if len(_read_journal(run_dir)) != 6:
        raise Amendment06IntegrityError("Unexpected event before Amendment 07 receipt")

    precheck_relative = "provenance/amendment07_resume_precheck.json"
    _publish_or_validate_amendment07_member(
        run_dir=run_dir, relative=precheck_relative,
        data=material[precheck_relative], transaction_id=transaction_id,
    )
    receipt = _amendment07_install_receipt_payload(
        intent_sha256=intent_sha256, generated_at=generated_at,
        material=material, bindings=bindings,
    )
    receipt_bytes = strict_full_json_bytes(receipt)
    staged_receipt = staging / "provenance/amendment07_overlay_install_receipt.json"
    if staged_receipt.exists():
        if not _safe_regular(staged_receipt, sha256_bytes(receipt_bytes)):
            raise Amendment06IntegrityError("Amendment 07 staged receipt drifted")
    else:
        publish_bytes_no_clobber(staged_receipt, receipt_bytes)
    _publish_or_validate_amendment07_member(
        run_dir=run_dir,
        relative="provenance/amendment07_overlay_install_receipt.json",
        data=receipt_bytes, transaction_id=transaction_id,
    )
    return validate_amendment07_overlay(run_dir)


def validate_amendment07_overlay(
    run_dir: Path = A07_TARGET_RUN,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    _assert_no_amendment07_overlay_mismatch()
    if run_dir != A07_TARGET_RUN.resolve():
        raise Amendment06IntegrityError("Amendment 07 overlay target Run differs")
    _validate_amendment07_base_history(run_dir)
    _validate_amendment07_ledger_prefix()
    _validate_amendment07_journal_prefix(run_dir)
    forensic = _validate_amendment07_forensic_snapshot()
    cache = _validate_amendment07_cache_files(run_dir)
    if not _safe_regular(A07_INSTALL_INTENT):
        raise Amendment06IntegrityError("Amendment 07 installation intent is absent")
    intent = _require_exact_keys(
        strict_full_load_file(A07_INSTALL_INTENT),
        {
            "schema", "status", "authorization", "prompt_sha256", "target_run_id",
            "transaction_id", "generated_at", "corrigendum_installations_authorized",
            "declared_member_count", "declared_members", "receipt_relative_path",
            "projected_corrigendum_journal_event",
            "projected_corrigendum_journal_event_sha256",
            "projected_corrigendum_journal_head_sha256",
            "base_scientific_execution_code_manifest_sha256",
            "effective_scientific_execution_code_manifest_sha256",
            "amendment07_code_corrigendum_sha256", "cache_manifest_sha256",
            "current_a06_authorized_change_ledger_sha256",
            "current_a06_final_prerun_evidence_sha256",
        },
        label="Amendment 07 installation intent",
    )
    if not (
        intent["schema"] == "amendment07_overlay_install_intent/v1"
        and intent["status"] == "INTENT"
        and intent["authorization"] == A07_AUTHORIZATION
        and intent["prompt_sha256"] == A07_PROMPT_SHA256
        and intent["target_run_id"] == run_dir.name
        and intent["corrigendum_installations_authorized"] == 1
        and intent["base_scientific_execution_code_manifest_sha256"]
        == A07_BASE_SCIENTIFIC_MANIFEST_SHA256
        and intent["cache_manifest_sha256"] == A07_CACHE_MANIFEST_SHA256
    ):
        raise Amendment06IntegrityError("Amendment 07 installation intent semantics differ")
    generated_at = str(intent["generated_at"])
    probe_path = run_dir / "provenance/amendment07_cuda_producer_consumer_probe.json"
    probe = validate_amendment07_cuda_producer_consumer_probe(probe_path)
    # Rebuild using the installed probe bytes without creating a validation file.
    chain_stub = {"ledger_sha256": str(intent["current_a06_authorized_change_ledger_sha256"])}
    material, bindings = _build_amendment07_overlay_material(
        run_dir=run_dir, probe_path=probe_path, generated_at=generated_at,
        chain=chain_stub,
        prerun_sha256=str(intent["current_a06_final_prerun_evidence_sha256"]),
    )
    del probe
    expected_intent = _amendment07_install_intent_payload(
        generated_at=generated_at, transaction_id=str(intent["transaction_id"]),
        material=material, bindings=bindings, chain=chain_stub,
        prerun_sha256=str(intent["current_a06_final_prerun_evidence_sha256"]),
    )
    if dict(intent) != expected_intent:
        raise Amendment06IntegrityError("Amendment 07 installation intent/member plan differs")
    for relative, data in material.items():
        path = run_dir / relative
        if not (_safe_regular(path, sha256_bytes(data)) and path.stat().st_size == len(data)):
            raise Amendment06IntegrityError(f"Amendment 07 overlay member drifted: {relative}")
    event5 = bindings["event5"]
    journal = _read_journal(run_dir)
    if len(journal) < 6 or journal[5] != event5:
        raise Amendment06IntegrityError("Amendment 07 Journal install event differs")
    receipt_path = run_dir / "provenance/amendment07_overlay_install_receipt.json"
    expected_receipt = _amendment07_install_receipt_payload(
        intent_sha256=sha256_file(A07_INSTALL_INTENT), generated_at=generated_at,
        material=material, bindings=bindings,
    )
    receipt = _require_exact_keys(
        strict_full_load_file(receipt_path), set(expected_receipt),
        label="Amendment 07 overlay install receipt",
    )
    if dict(receipt) != expected_receipt:
        raise Amendment06IntegrityError("Amendment 07 overlay install receipt differs")
    code_manifests = _validate_amendment07_code_manifests(run_dir)
    if not (
        intent["effective_scientific_execution_code_manifest_sha256"]
        == bindings["effective_manifest_sha256"]
        == code_manifests["effective_manifest_sha256"]
        and intent["amendment07_code_corrigendum_sha256"]
        == bindings["corrigendum_sha256"]
        and sha256_file(AUTHORIZED_CHANGE_LEDGER)
        == intent["current_a06_authorized_change_ledger_sha256"]
        and sha256_file(FINAL_PRERUN_EVIDENCE)
        == intent["current_a06_final_prerun_evidence_sha256"]
        and cache["cache_manifest_sha256"] == A07_CACHE_MANIFEST_SHA256
        and forensic["forensic_zip_sha256"]
        == strict_full_load_file(
            run_dir / "provenance/amendment07_code_corrigendum.json"
        )["forensic_snapshot_sha256"]
    ):
        raise Amendment06IntegrityError("Amendment 07 cross-artifact binding differs")
    return {
        "status": "PASS", "run_id": run_dir.name,
        "authorization": A07_AUTHORIZATION, "prompt_sha256": A07_PROMPT_SHA256,
        "base_scientific_execution_code_manifest_sha256": A07_BASE_SCIENTIFIC_MANIFEST_SHA256,
        "effective_scientific_execution_code_manifest_sha256": bindings[
            "effective_manifest_sha256"
        ],
        "amendment07_code_corrigendum_sha256": bindings["corrigendum_sha256"],
        "overlay_install_receipt_sha256": sha256_file(receipt_path),
        "corrigendum_journal_event_index": 5,
        "corrigendum_journal_head_sha256": sha256_bytes(
            strict_full_canonical_json_bytes(event5)
        ),
        "code_manifests": code_manifests,
    }


def _amendment08_frozen_lineage_bindings(run_dir: Path) -> dict[str, str]:
    """Validate the installed Run overlay without consulting mutable Runtime state."""

    run_dir = Path(run_dir).resolve()
    receipt_path = run_dir / "provenance/amendment08_overlay_install_receipt.json"
    receipt = _require_exact_keys(
        strict_full_load_file(receipt_path),
        {
            "schema", "status", "authorization", "prompt_sha256", "target_run_id",
            "transaction_id", "installation_intent_sha256",
            "overlay_installations", "member_count", "members",
            "corrigendum_journal_event_index",
            "corrigendum_journal_event_raw_sha256",
            "corrigendum_journal_head_sha256", "precheck_sha256",
            "base_scientific_execution_code_manifest_sha256",
            "effective_scientific_execution_code_manifest_sha256",
            "amendment08_code_corrigendum_sha256",
            "amendment08_journal_contamination_adjudication_sha256",
            "source_freeze_receipt_sha256",
            "current_a06_authorized_change_ledger_sha256",
            "current_a06_final_prerun_evidence_sha256",
            "a08_prelive_package_lineage_sha256",
            "a08_prelive_package_receipt_sha256",
            "a08_prelive_package_code_sha256",
            "a08_prelive_package_test_sha256", "cache_manifest_sha256",
            "frozen_population_sha256", "installed_at",
        },
        label="Frozen Amendment 08 overlay receipt",
    )
    member_relatives = tuple(
        relative for relative in A08_OVERLAY_RUN_RELATIVES
        if relative != "provenance/amendment08_overlay_install_receipt.json"
    )
    members = receipt["members"]
    if not (
        receipt["schema"] == "amendment08_overlay_install_receipt/v1"
        and receipt["status"] == "PASS"
        and receipt["authorization"] == A08_AUTHORIZATION
        and receipt["prompt_sha256"] == A08_PROMPT_SHA256
        and receipt["target_run_id"] == run_dir.name == A08_TARGET_RUN_ID
        and receipt["overlay_installations"] == 1
        and receipt["member_count"] == 10
        and isinstance(members, list) and len(members) == 10
        and [item.get("relative_path") for item in members]
        == list(member_relatives)
        and receipt["base_scientific_execution_code_manifest_sha256"]
        == A07_BASE_SCIENTIFIC_MANIFEST_SHA256
        and receipt["cache_manifest_sha256"] == A07_CACHE_MANIFEST_SHA256
        and receipt["frozen_population_sha256"] == A07_FROZEN_POPULATION_SHA256
        and receipt["corrigendum_journal_event_index"] == 7
    ):
        raise Amendment06IntegrityError("Frozen Amendment 08 overlay receipt differs")
    for record, relative in zip(members, member_relatives, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "relative_path", "size_bytes", "sha256",
        }:
            raise Amendment06IntegrityError("Frozen Amendment 08 member record differs")
        member = run_dir / relative
        if not (
            _safe_regular(member, str(record["sha256"]))
            and member.stat().st_size == record["size_bytes"]
        ):
            raise Amendment06IntegrityError(
                f"Frozen Amendment 08 member drifted: {relative}"
            )
    effective_path = (
        run_dir
        / "provenance/scientific_execution_code_manifest_effective_amendment08.tsv"
    )
    corrigendum_path = run_dir / "provenance/amendment08_code_corrigendum.json"
    adjudication_path = (
        run_dir / "provenance/amendment08_journal_contamination_adjudication.json"
    )
    prompt_path = run_dir / "provenance/amendment08_prompt_snapshot.txt"
    if not (
        sha256_file(effective_path)
        == receipt["effective_scientific_execution_code_manifest_sha256"]
        and sha256_file(corrigendum_path)
        == receipt["amendment08_code_corrigendum_sha256"]
        and sha256_file(adjudication_path)
        == receipt["amendment08_journal_contamination_adjudication_sha256"]
        and _safe_regular(prompt_path, A08_PROMPT_SHA256)
        and sha256_file(
            run_dir / "provenance/amendment08_resume_precheck.json"
        ) == receipt["precheck_sha256"]
    ):
        raise Amendment06IntegrityError("Frozen Amendment 08 overlay cross-binding differs")
    corrigendum = strict_full_load_file(corrigendum_path)
    adjudication = strict_full_load_file(adjudication_path)
    if not (
        corrigendum.get("authorization") == A08_AUTHORIZATION
        and corrigendum.get("prompt_sha256") == A08_PROMPT_SHA256
        and corrigendum.get("target_run_id") == run_dir.name
        and adjudication.get("authorization") == A08_AUTHORIZATION
        and adjudication.get("prompt_sha256") == A08_PROMPT_SHA256
        and adjudication.get("target_run_id") == run_dir.name
    ):
        raise Amendment06IntegrityError("Frozen Amendment 08 overlay identity differs")
    return {
        "base_scientific_execution_code_manifest_sha256": (
            A07_BASE_SCIENTIFIC_MANIFEST_SHA256
        ),
        "effective_scientific_execution_code_manifest_sha256": str(
            receipt["effective_scientific_execution_code_manifest_sha256"]
        ),
        "amendment08_code_corrigendum_sha256": str(
            receipt["amendment08_code_corrigendum_sha256"]
        ),
        "amendment08_journal_contamination_adjudication_sha256": str(
            receipt["amendment08_journal_contamination_adjudication_sha256"]
        ),
        "amendment08_authorization": A08_AUTHORIZATION,
        "amendment08_prompt_sha256": A08_PROMPT_SHA256,
    }


def _scientific_lineage_bindings(run_dir: Path) -> dict[str, str]:
    run_dir = Path(run_dir).resolve()
    if run_dir.name == A08_TARGET_RUN_ID or (
        run_dir / "provenance/amendment08_overlay_install_receipt.json"
    ).exists():
        receipt_path = run_dir / "provenance/amendment08_overlay_install_receipt.json"
        if not _safe_regular(receipt_path):
            raise Amendment06IntegrityError(
                "Amendment 08 target lacks its active overlay receipt"
            )
        return _amendment08_frozen_lineage_bindings(run_dir)
    if (
        run_dir / "provenance/amendment07_overlay_install_receipt.json"
    ).exists():
        code = _validate_amendment07_code_manifests(run_dir)
        corrigendum_path = run_dir / "provenance/amendment07_code_corrigendum.json"
        receipt_path = run_dir / "provenance/amendment07_overlay_install_receipt.json"
        if not (_safe_regular(corrigendum_path) and _safe_regular(receipt_path)):
            raise Amendment06IntegrityError("Amendment 07 lineage artifacts are absent")
        corrigendum = strict_full_load_file(corrigendum_path)
        receipt = strict_full_load_file(receipt_path)
        effective = code["effective_manifest_sha256"]
        corrigendum_sha256 = sha256_file(corrigendum_path)
        if not (
            corrigendum.get("authorization") == A07_AUTHORIZATION
            and corrigendum.get("prompt_sha256") == A07_PROMPT_SHA256
            and corrigendum.get("base_scientific_execution_code_manifest_sha256")
            == A07_BASE_SCIENTIFIC_MANIFEST_SHA256
            and corrigendum.get("effective_scientific_execution_code_manifest_sha256")
            == effective
            and receipt.get("effective_scientific_execution_code_manifest_sha256")
            == effective
            and receipt.get("amendment07_code_corrigendum_sha256")
            == corrigendum_sha256
        ):
            raise Amendment06IntegrityError("Amendment 07 lightweight lineage binding differs")
        return {
            "base_scientific_execution_code_manifest_sha256": A07_BASE_SCIENTIFIC_MANIFEST_SHA256,
            "effective_scientific_execution_code_manifest_sha256": effective,
            "amendment07_code_corrigendum_sha256": corrigendum_sha256,
        }
    base = sha256_file(run_dir / "provenance/scientific_execution_code_manifest.tsv")
    return {
        "base_scientific_execution_code_manifest_sha256": base,
        "effective_scientific_execution_code_manifest_sha256": base,
        "amendment07_code_corrigendum_sha256": "",
    }


def validate_amendment07_pre_resume(
    *, run_dir: Path = A07_TARGET_RUN,
    preflight_run: Path = ACCEPTED_PREFLIGHT,
    smoke_run: Path = ACCEPTED_SMOKE,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    assert_exact_invocation(preflight_run=preflight_run, smoke_run=smoke_run)
    if _AMENDMENT08_REHEARSAL_CONTEXT.get() is None:
        admission = verify_reference_admission(
            preflight_run=preflight_run, smoke_run=smoke_run,
        )
    else:
        projection = validate_amendment08_isolated_rehearsal_projection(run_dir)
        admission = dict(projection["reference_projection"])
    if not _safe_regular(EXTERNAL_AUDIT, EXTERNAL_AUDIT_SHA256):
        raise Amendment06IntegrityError(
            "Amendment 07 external Audit streaming hash differs before Resume"
        )
    overlay = validate_amendment07_overlay(run_dir)
    cache = _validate_amendment07_cache_files(run_dir)
    events = _read_journal(run_dir)
    if len(events) < 6 or events[5].get("event") != "AMENDMENT07_CORRIGENDUM_INSTALLED":
        raise Amendment06IntegrityError("Amendment 07 install event is absent before Resume")
    cache_loads = [event for event in events if event.get("event") == "CACHE_LOAD"]
    history: Mapping[str, Any] | None = None
    if cache_loads:
        history = _validate_amendment07_corrected_training_history(
            run_dir=run_dir, overlay=overlay, allow_unterminated=True,
        )
    elif len(events) != 6:
        raise Amendment06IntegrityError("Unexpected event precedes Amendment 07 CACHE_LOAD")
    return {
        "status": "PASS", "authorization": A07_AUTHORIZATION,
        "prompt_sha256": A07_PROMPT_SHA256, "target_run_id": run_dir.name,
        "overlay": overlay, "cache": cache,
        "reference_admission_status": admission["status"],
        "journal_event_count": len(events),
        "cache_load_event_count": len(cache_loads),
        "corrected_training_history": history,
        "external_audit_tabular_parse_calls": 0,
        "new_full_run_ids_created": 0,
    }


def _amendment08_expected_raw_prefix_events() -> tuple[dict[str, Any], ...]:
    cache_path = str(
        PRODUCTION_EXPECTED_RUNTIME_ROOT / "full_population" / A08_TARGET_RUN_ID
    )
    return (
        {
            "base_seed": 42, "cache_path": cache_path,
            "event": "CACHE_GENERATION_INTENT", "event_index": 0,
            "previous_event_sha256": None,
            "primary_table_sha256": PRIMARY_TABLE_SHA256,
            "recorded_at": "2026-08-20T15:25:11.338269-04:00",
            "run_id": A08_TARGET_RUN_ID,
            "split_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
            "status": "STARTED",
        },
        {
            "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
            "cache_path": cache_path, "event": "CACHE_GENERATION",
            "event_index": 1,
            "previous_event_sha256": A08_CANONICAL_JOURNAL_SHA256[0],
            "recorded_at": "2026-08-20T15:33:53.122563-04:00",
            "recovered_after_interruption": False,
            "run_id": A08_TARGET_RUN_ID, "status": "PASS",
        },
        {
            "attempt_id": A07_FAILED_ATTEMPT_ID,
            "event": "TRAINING_ATTEMPT_STARTED", "event_index": 2,
            "phase": "TRAINING",
            "previous_event_sha256": A08_CANONICAL_JOURNAL_SHA256[1],
            "recorded_at": "2026-08-20T15:33:53.139341-04:00",
            "run_id": A08_TARGET_RUN_ID, "variant_id": "full_new_reference",
        },
        {
            "attempt_id": A07_FAILED_ATTEMPT_ID,
            "event": "SCIENTIFIC_FIT_CALLED", "event_index": 3,
            "phase": "TRAINING",
            "previous_event_sha256": A08_CANONICAL_JOURNAL_SHA256[2],
            "recorded_at": "2026-08-20T15:33:53.253449-04:00",
            "run_id": A08_TARGET_RUN_ID, "seed": PRIMARY_MODEL_SEED,
            "stability_fit": False, "variant_id": "full_new_reference",
        },
        {
            "attempt_id": A07_FAILED_ATTEMPT_ID,
            "error": "GPU identity has no name",
            "error_type": "Amendment06IntegrityError",
            "event": "TRAINING_ATTEMPT_FAILED", "event_index": 4,
            "failure_phase": "POST_START_THROUGH_COMPLETION_COMMIT",
            "phase": "TRAINING",
            "previous_event_sha256": A08_CANONICAL_JOURNAL_SHA256[3],
            "recorded_at": "2026-08-20T15:33:58.006792-04:00",
            "run_id": A08_TARGET_RUN_ID, "variant_id": "full_new_reference",
        },
        {
            "attempt_id": "a" * 32,
            "completion_sha256": A08_TEST_FIXTURE_SHA256,
            "event": "REPORTING_COMPLETION_COMMITTED", "event_index": 5,
            "phase": "REPORTING",
            "previous_event_sha256": A08_CANONICAL_JOURNAL_SHA256[4],
            "recorded_at": "2026-08-20T17:58:49.643305-04:00",
            "run_id": A08_TARGET_RUN_ID, "variant_id": "GLOBAL",
        },
        {
            "attempt_id": "a" * 32,
            "completion_sha256": A08_TEST_FIXTURE_SHA256,
            "event": "REPORTING_COMPLETION_COMMITTED", "event_index": 6,
            "phase": "REPORTING",
            "previous_event_sha256": A08_CANONICAL_JOURNAL_SHA256[5],
            "recorded_at": "2026-08-20T18:03:08.631177-04:00",
            "run_id": A08_TARGET_RUN_ID, "variant_id": "GLOBAL",
        },
    )


def _amendment08_journal_directory_inventory_bytes(journal: Path) -> bytes:
    journal = Path(journal)
    if journal.is_symlink() or not journal.is_dir():
        raise Amendment06IntegrityError("Amendment 08 Journal directory is absent/unsafe")
    rows: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for member in journal.iterdir():
        relative = member.name
        if (
            relative in seen or not re.fullmatch(r"[0-9]{8}\.json", relative)
            or not _safe_regular(member)
        ):
            raise Amendment06IntegrityError("Amendment 08 Journal inventory is unsafe")
        seen.add(relative)
        rows.append((sha256_file(member), member.stat().st_size, relative))
    return b"".join(
        f"{digest}\t{size}\t{relative}\n".encode("utf-8")
        for digest, size, relative in sorted(rows, key=lambda row: row[2].encode("utf-8"))
    )


def amendment08_live_journal_inventory_bytes(run_dir: Path | None = None) -> bytes:
    if run_dir is None:
        run_dir = A08_TARGET_RUN
    journal = _journal_path(Path(run_dir).resolve())
    return _amendment08_journal_directory_inventory_bytes(journal)


def amendment08_production_live_journal_inventory_bytes() -> bytes:
    """Read the production seven-event sentinel without consulting patched roots."""

    journal = (
        Path(PRODUCTION_EXPECTED_RUNTIME_ROOT) / "journals" / A08_TARGET_RUN_ID
    )
    data = _amendment08_journal_directory_inventory_bytes(journal)
    expected_names = {f"{index:08d}.json" for index in range(7)}
    observed_names = {
        line.rsplit(b"\t", 1)[-1].decode("utf-8").rstrip("\n")
        for line in data.splitlines(keepends=True)
    }
    if not (
        observed_names == expected_names
        and sha256_bytes(data) == A08_PRELIVE_JOURNAL_SENTINEL_SHA256
    ):
        raise Amendment06IntegrityError(
            "Production Amendment 08 Journal sentinel differs"
        )
    return data


def _validate_amendment08_raw_prefix(
    run_dir: Path | None = None, *, require_exact_boundary: bool = False,
) -> dict[str, Any]:
    if run_dir is None:
        run_dir = A08_TARGET_RUN
    run_dir = Path(run_dir).resolve()
    if run_dir.name != A08_TARGET_RUN_ID:
        raise Amendment06IntegrityError("Amendment 08 raw prefix Run ID differs")
    events = _read_journal(run_dir)
    expected = _amendment08_expected_raw_prefix_events()
    if len(events) < 7 or (require_exact_boundary and len(events) != 7):
        raise Amendment06IntegrityError("Amendment 08 raw Journal boundary count differs")
    raw_concat = bytearray()
    canonical_lines = bytearray()
    records: list[dict[str, Any]] = []
    for index, expected_event in enumerate(expected):
        path = _journal_event_path(run_dir, index)
        raw = path.read_bytes() if _safe_regular(path) else b""
        canonical = strict_full_canonical_json_bytes(events[index])
        if not (
            sha256_bytes(raw) == A08_RAW_JOURNAL_SHA256[index]
            and events[index] == expected_event
            and sha256_bytes(canonical) == A08_CANONICAL_JOURNAL_SHA256[index]
        ):
            raise Amendment06IntegrityError(
                f"Amendment 08 raw Journal event {index} differs"
            )
        raw_concat.extend(raw)
        canonical_lines.extend(canonical + b"\n")
        records.append({
            "physical_event_index": index,
            "relative_path": f"{index:08d}.json",
            "raw_size_bytes": len(raw), "raw_sha256": sha256_bytes(raw),
            "canonical_sha256": sha256_bytes(canonical),
            "event": dict(events[index]),
        })
    if not (
        sha256_bytes(bytes(raw_concat)) == A08_RAW_EVENTS_0_TO_6_CONCAT_SHA256
        and sha256_bytes(bytes(canonical_lines))
        == A08_CANONICAL_EVENTS_0_TO_6_LINES_SHA256
    ):
        raise Amendment06IntegrityError("Amendment 08 raw Journal aggregate differs")
    return {
        "status": "PASS", "target_run_id": run_dir.name,
        "raw_event_count": len(events), "base_event_count": 7,
        "pre_a08_raw_head_sha256": A08_CANONICAL_JOURNAL_SHA256[6],
        "raw_events_0_to_6_concat_sha256": A08_RAW_EVENTS_0_TO_6_CONCAT_SHA256,
        "canonical_json_lines_events_0_to_6_sha256": (
            A08_CANONICAL_EVENTS_0_TO_6_LINES_SHA256
        ),
        "classified_records": records[5:7], "events": events,
    }


def _validate_amendment08_forensic_snapshot() -> dict[str, Any]:
    if not (
        _safe_regular(A08_FORENSIC_ZIP, A08_FORENSIC_ZIP_SHA256)
        and A08_FORENSIC_ZIP.stat().st_size == A08_FORENSIC_ZIP_SIZE_BYTES
        and _safe_regular(
            A08_FORENSIC_VERIFICATION, A08_FORENSIC_VERIFICATION_SHA256,
        )
        and A08_FORENSIC_VERIFICATION.stat().st_size
        == A08_FORENSIC_VERIFICATION_SIZE_BYTES
        and _safe_regular(A08_PROMPT_PATH, A08_PROMPT_SHA256)
    ):
        raise Amendment06IntegrityError("Pinned Amendment 08 forensic bytes differ")
    verification = strict_full_load_file(A08_FORENSIC_VERIFICATION)
    required = {
        "schema", "status", "authorization", "prompt_sha256", "verified_at",
        "bundle_path", "bundle_sha256", "bundle_size_bytes", "member_count",
        "members", "canonical_tree_inventory_format",
        "canonical_tree_inventory_final_lf", "canonical_tree_inventory_sha256",
        "independent_reopen_status", "crc_status", "unique_member_names",
        "safe_member_names", "encrypted_members", "directory_entries",
        "symlink_entries", "non_regular_entries", "all_member_size_hashes_verified",
        "prompt_snapshot_path", "prompt_snapshot_sha256", "prompt_snapshot_size_bytes",
        "authoritative_prompt_byte_identity", "external_a08_review_zip_sha256",
        "external_a08_review_zip_validation", "target_run_id",
        "target_run_regular_files_snapshotted", "target_run_complete_file_tree",
        "target_run_directory_inventory_included", "raw_journal_event_count",
        "raw_journal_head_sha256", "raw_events_0_to_6_concat_sha256",
        "canonical_json_lines_events_0_to_6_sha256", "raw_journal_files_snapshotted",
        "contamination_classification", "contamination_event_indices",
        "cache_manifest_sha256", "cache_members_rehashed",
        "cache_large_members_metadata_only", "repo_source_or_test_writes",
        "target_run_writes", "pytest_or_collection_commands", "audit_tabular_parse_calls",
        "overlay_installations", "resume_commands", "package_commands",
        "no_clobber_publication",
        "verification_self_excluded_from_bundle_to_avoid_circular_hash",
    }
    _require_exact_keys(verification, required, label="Amendment 08 forensic verification")
    members = verification["members"]
    if not (
        verification["schema"] == "amendment08_pre_mutation_forensic_verification/v1"
        and verification["status"] == "PASS"
        and verification["authorization"] == A08_AUTHORIZATION
        and verification["prompt_sha256"] == A08_PROMPT_SHA256
        and verification["bundle_path"] == str(A08_FORENSIC_ZIP)
        and verification["bundle_sha256"] == A08_FORENSIC_ZIP_SHA256
        and verification["bundle_size_bytes"] == A08_FORENSIC_ZIP_SIZE_BYTES
        and verification["member_count"] == 81
        and isinstance(members, list) and len(members) == 81
        and verification["canonical_tree_inventory_final_lf"] is True
        and verification["canonical_tree_inventory_sha256"] == A08_FORENSIC_TREE_SHA256
        and verification["independent_reopen_status"] == "PASS"
        and verification["crc_status"] == "PASS"
        and verification["unique_member_names"] is True
        and verification["safe_member_names"] is True
        and verification["encrypted_members"] == 0
        and verification["directory_entries"] == 0
        and verification["symlink_entries"] == 0
        and verification["non_regular_entries"] == 0
        and verification["all_member_size_hashes_verified"] is True
        and verification["prompt_snapshot_sha256"] == A08_PROMPT_SHA256
        and verification["prompt_snapshot_size_bytes"] == A08_PROMPT_PATH.stat().st_size
        and verification["authoritative_prompt_byte_identity"] == "PASS"
        and verification["external_a08_review_zip_sha256"] == A08_EXTERNAL_REVIEW_ZIP_SHA256
        and verification["target_run_id"] == A08_TARGET_RUN_ID
        and verification["target_run_regular_files_snapshotted"] == 45
        and verification["target_run_complete_file_tree"] == "PASS"
        and verification["raw_journal_event_count"] == 7
        and verification["raw_journal_head_sha256"] == A08_CANONICAL_JOURNAL_SHA256[6]
        and verification["raw_events_0_to_6_concat_sha256"]
        == A08_RAW_EVENTS_0_TO_6_CONCAT_SHA256
        and verification["canonical_json_lines_events_0_to_6_sha256"]
        == A08_CANONICAL_EVENTS_0_TO_6_LINES_SHA256
        and verification["raw_journal_files_snapshotted"] == "PASS_7_OF_7"
        and verification["contamination_classification"]
        == A08_CONTAMINATION_CLASSIFICATION
        and verification["contamination_event_indices"] == [5, 6]
        and verification["cache_manifest_sha256"] == A07_CACHE_MANIFEST_SHA256
        and verification["cache_members_rehashed"] == "PASS_10_OF_10"
        and verification["repo_source_or_test_writes"] == 0
        and verification["target_run_writes"] == 0
        and verification["pytest_or_collection_commands"] == 0
        and verification["audit_tabular_parse_calls"] == 0
        and verification["overlay_installations"] == 0
        and verification["resume_commands"] == 0
        and verification["package_commands"] == 0
        and verification["no_clobber_publication"] is True
    ):
        raise Amendment06IntegrityError("Amendment 08 forensic verification semantics differ")
    expected: dict[str, Mapping[str, Any]] = {}
    for item in members:
        record = _require_exact_keys(
            item, {"compression_method", "crc32", "member_path", "sha256", "size_bytes"},
            label="Amendment 08 forensic member",
        )
        name = str(record["member_path"])
        if (
            name in expected or Path(name).is_absolute() or ".." in Path(name).parts
            or Path(name).as_posix() != name
            or record["compression_method"] not in {0, 8}
            or not re.fullmatch(r"[0-9a-f]{8}", str(record["crc32"]))
            or not re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"]))
            or isinstance(record["size_bytes"], bool)
            or not isinstance(record["size_bytes"], int) or record["size_bytes"] < 0
        ):
            raise Amendment06IntegrityError("Amendment 08 forensic member record differs")
        expected[name] = record
    root = "amendment08_pre_mutation_forensic"
    required_members = {
        f"{root}/pre_a08_sources/code/amendment06_full.py": A08_PRE_FULL_SHA256,
        f"{root}/pre_a08_sources/tests/test_amendment06.py": A08_PRE_TEST_SHA256,
        f"{root}/pre_a08_sources/code/amendment06_packaging.py": A08_PRE_PACKAGE_SHA256,
        f"{root}/pre_a08_sources/tests/test_amendment06_packaging.py": A08_PRE_PACKAGE_TEST_SHA256,
        f"{root}/a06_runtime/pre_run/final_prerun_evidence.json": A07_ORIGINAL_PRERUN_SHA256,
        f"{root}/metadata/journal_member_inventory.tsv": "70eef7089288594ba0c744fb3b5b3e98466b950e76f9fc761a118073735ff61b",
        f"{root}/metadata/cache_member_inventory.tsv": A07_FORENSIC_CACHE_INVENTORY_SHA256,
    }
    if any(
        name not in expected or expected[name]["sha256"] != digest
        for name, digest in required_members.items()
    ):
        raise Amendment06IntegrityError("Amendment 08 mandatory forensic inventory differs")
    with zipfile.ZipFile(A08_FORENSIC_ZIP, "r") as archive:
        infos = archive.infolist()
        if archive.testzip() is not None or len(infos) != 81:
            raise Amendment06IntegrityError("Amendment 08 forensic ZIP CRC/count differs")
        if [info.filename for info in infos] != list(expected):
            raise Amendment06IntegrityError("Amendment 08 forensic ZIP order/member set differs")
        for info in infos:
            record = expected[info.filename]
            mode = (info.external_attr >> 16) & 0xFFFF
            if (
                info.flag_bits & 0x1 or info.is_dir()
                or (mode and not stat.S_ISREG(mode))
                or info.file_size != record["size_bytes"]
                or f"{info.CRC & 0xffffffff:08x}" != record["crc32"]
            ):
                raise Amendment06IntegrityError(
                    f"Amendment 08 forensic ZIP metadata differs: {info.filename}"
                )
            digest = hashlib.sha256()
            with archive.open(info) as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise Amendment06IntegrityError(
                    f"Amendment 08 forensic ZIP bytes differ: {info.filename}"
                )
    return {
        "status": "PASS", "authorization": A08_AUTHORIZATION,
        "prompt_sha256": A08_PROMPT_SHA256, "target_run_id": A08_TARGET_RUN_ID,
        "forensic_zip_sha256": A08_FORENSIC_ZIP_SHA256,
        "forensic_verification_sha256": A08_FORENSIC_VERIFICATION_SHA256,
        "canonical_tree_inventory_sha256": A08_FORENSIC_TREE_SHA256,
        "original_prerun_forensic_sha256": A07_ORIGINAL_PRERUN_SHA256,
    }


def _amendment08_rehearsal_context_paths() -> tuple[Path, Path]:
    context = _AMENDMENT08_REHEARSAL_CONTEXT.get()
    if context is None:
        raise Amendment06IntegrityError(
            "Amendment 08 rehearsal projection is unavailable outside its isolated context"
        )
    run_dir, runtime_root = context
    if not (
        run_dir == Path(A08_TARGET_RUN).resolve()
        and runtime_root == Path(EXPECTED_RUNTIME_ROOT).resolve()
        and run_dir.parent.parent == runtime_root.parent
        and run_dir.parent.name == "results" and runtime_root.name == "runtime"
        and _isolated_exact_target_journal_pair(run_dir, runtime_root)
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 rehearsal projection context drifted"
        )
    return run_dir, runtime_root


def amendment08_isolated_rehearsal_projection_path(
    run_dir: Path | None = None,
) -> Path:
    context_run, runtime_root = _amendment08_rehearsal_context_paths()
    selected = context_run if run_dir is None else Path(run_dir).resolve()
    if selected != context_run:
        raise Amendment06IntegrityError(
            "Amendment 08 rehearsal projection Run differs from its context"
        )
    return runtime_root / "rehearsal_projection" / selected.name / "admission.json"


def _amendment08_forensic_target_run_records() -> tuple[dict[str, dict[str, Any]], str]:
    verification = strict_full_load_file(A08_FORENSIC_VERIFICATION)
    members = verification.get("members")
    if not isinstance(members, list):
        raise Amendment06IntegrityError("Amendment 08 forensic members are absent")
    prefix = "amendment08_pre_mutation_forensic/target_run/"
    records: dict[str, dict[str, Any]] = {}
    for item in members:
        if not isinstance(item, Mapping):
            raise Amendment06IntegrityError("Amendment 08 forensic member is malformed")
        name = str(item.get("member_path", ""))
        if not name.startswith(prefix):
            continue
        relative = name[len(prefix):]
        relative_path = Path(relative)
        size = item.get("size_bytes")
        digest = str(item.get("sha256", ""))
        if (
            not relative or relative_path.is_absolute() or ".." in relative_path.parts
            or relative_path.as_posix() != relative or relative in records
            or isinstance(size, bool) or not isinstance(size, int) or size < 0
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise Amendment06IntegrityError(
                "Amendment 08 forensic target-Run record differs"
            )
        records[relative] = {"size_bytes": size, "sha256": digest}
    inventory = b"".join(
        f"{records[relative]['sha256']}\t{records[relative]['size_bytes']}\t{relative}\n".encode(
            "utf-8"
        )
        for relative in sorted(records, key=lambda value: value.encode("utf-8"))
    )
    if not (
        len(records) == A08_REHEARSAL_BASE_RUN_FILE_COUNT
        and sum(int(record["size_bytes"]) for record in records.values())
        == A08_REHEARSAL_BASE_RUN_SIZE_BYTES
        and sha256_bytes(inventory) == A08_REHEARSAL_BASE_RUN_INVENTORY_SHA256
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 forensic target-Run inventory is not exact 45"
        )
    return records, sha256_bytes(inventory)


def _validate_amendment08_rehearsal_base_run(
    run_dir: Path, *, require_exact_set: bool,
) -> dict[str, Any]:
    records, inventory_sha256 = _amendment08_forensic_target_run_records()
    observed: set[str] = set()
    for path in run_dir.rglob("*"):
        relative = path.relative_to(run_dir).as_posix()
        mode = os.lstat(path).st_mode
        if stat.S_ISDIR(mode) and not path.is_symlink():
            continue
        if not stat.S_ISREG(mode) or path.is_symlink():
            raise Amendment06IntegrityError(
                f"Amendment 08 rehearsal Run has an unsafe member: {relative}"
            )
        observed.add(relative)
    if require_exact_set and observed != set(records):
        raise Amendment06IntegrityError(
            "Amendment 08 rehearsal Run is not the exact 45-file forensic clone"
        )
    if not set(records).issubset(observed):
        raise Amendment06IntegrityError(
            "Amendment 08 rehearsal Run lost a forensic base member"
        )
    for relative, record in records.items():
        path = run_dir / relative
        if not (
            _safe_regular(path, str(record["sha256"]))
            and path.stat().st_size == record["size_bytes"]
        ):
            raise Amendment06IntegrityError(
                f"Amendment 08 rehearsal base member drifted: {relative}"
            )
    return {
        "status": "PASS", "base_run_regular_file_count": len(records),
        "base_run_inventory_sha256": inventory_sha256,
    }


def _amendment08_cache_inventory_bytes(
    records: Sequence[Mapping[str, Any]],
) -> bytes:
    return b"".join(
        f"{record['sha256']}\t{int(record['size_bytes'])}\t{record['relative_path']}\n".encode(
            "utf-8"
        )
        for record in sorted(
            records, key=lambda item: str(item["relative_path"]).encode("utf-8")
        )
    )


def _amendment08_rehearsal_cache_clone_evidence(
    run_dir: Path, *, rehash_source_and_clone: bool,
) -> dict[str, Any]:
    source = Path(PRODUCTION_EXPECTED_RUNTIME_ROOT) / "full_population" / run_dir.name
    clone = Path(EXPECTED_RUNTIME_ROOT) / "full_population" / run_dir.name
    if not (
        source.resolve() != clone.resolve()
        and source.is_dir() and not source.is_symlink()
        and clone.is_dir() and not clone.is_symlink()
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 rehearsal cache source/clone layout differs"
        )
    manifest_path = clone / "cache_manifest.json"
    if not _safe_regular(manifest_path, A07_CACHE_MANIFEST_SHA256):
        raise Amendment06IntegrityError("Amendment 08 rehearsal cache manifest differs")
    manifest = strict_full_load_file(manifest_path)
    cache_files = manifest.get("cache_files")
    if not isinstance(cache_files, list) or len(cache_files) != A08_CACHE_MEMBER_COUNT:
        raise Amendment06IntegrityError("Amendment 08 rehearsal cache member plan differs")
    expected: dict[str, tuple[int, str]] = {
        "cache_manifest.json": (manifest_path.stat().st_size, A07_CACHE_MANIFEST_SHA256),
    }
    cache_member_bytes = 0
    for item in cache_files:
        record = _require_exact_keys(
            item, {"relative_path", "size_bytes", "sha256"},
            label="Amendment 08 rehearsal cache member",
        )
        relative = str(record["relative_path"])
        relative_path = Path(relative)
        size = record["size_bytes"]
        digest = str(record["sha256"])
        if (
            not relative or relative_path.is_absolute() or len(relative_path.parts) != 1
            or relative_path.as_posix() != relative or relative in expected
            or isinstance(size, bool) or not isinstance(size, int) or size < 0
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise Amendment06IntegrityError(
                "Amendment 08 rehearsal cache member schema differs"
            )
        expected[relative] = (size, digest)
        cache_member_bytes += size
    if cache_member_bytes != A08_CACHE_MEMBER_BYTES:
        raise Amendment06IntegrityError(
            "Amendment 08 rehearsal cache byte total differs"
        )
    expected_names = set(expected)
    for directory, label in ((source, "source"), (clone, "clone")):
        actual = {path.name for path in directory.iterdir()}
        if actual != expected_names:
            raise Amendment06IntegrityError(
                f"Amendment 08 rehearsal cache {label} member set differs"
            )
    records: list[dict[str, Any]] = []
    for relative in sorted(expected, key=lambda value: value.encode("utf-8")):
        size, digest = expected[relative]
        source_path = source / relative
        clone_path = clone / relative
        source_stat = os.lstat(source_path)
        clone_stat = os.lstat(clone_path)
        if not (
            stat.S_ISREG(source_stat.st_mode) and not source_path.is_symlink()
            and stat.S_ISREG(clone_stat.st_mode) and not clone_path.is_symlink()
            and source_stat.st_size == clone_stat.st_size == size
            and (source_stat.st_dev, source_stat.st_ino)
            != (clone_stat.st_dev, clone_stat.st_ino)
        ):
            raise Amendment06IntegrityError(
                f"Amendment 08 rehearsal cache clone aliases/drifts: {relative}"
            )
        if rehash_source_and_clone and not (
            sha256_file(source_path) == digest and sha256_file(clone_path) == digest
        ):
            raise Amendment06IntegrityError(
                f"Amendment 08 rehearsal cache clone bytes differ: {relative}"
            )
        records.append({
            "relative_path": relative, "size_bytes": size, "sha256": digest,
            "source_device": int(source_stat.st_dev),
            "source_inode": int(source_stat.st_ino),
            "clone_device": int(clone_stat.st_dev),
            "clone_inode": int(clone_stat.st_ino),
        })
    inventory_sha256 = sha256_bytes(_amendment08_cache_inventory_bytes(records))
    return {
        "status": "PASS", "cache_source_path": str(source.resolve()),
        "cache_clone_path": str(clone.resolve()),
        "cache_member_count": A08_CACHE_MEMBER_COUNT,
        "cache_member_bytes": cache_member_bytes,
        "cache_inventory_file_count": len(records),
        "cache_source_inventory_sha256": inventory_sha256,
        "cache_clone_inventory_sha256": inventory_sha256,
        "cache_clone_distinct_inodes": True, "cache_clone_records": records,
    }


def _validate_amendment08_rehearsal_reference_projection(
    *, preflight_run: Path, smoke_run: Path,
) -> dict[str, Any]:
    if not (
        _resolved_equal(Path(preflight_run), ACCEPTED_PREFLIGHT)
        and _resolved_equal(Path(smoke_run), ACCEPTED_SMOKE)
        and all(_safe_regular(path, digest) for path, digest in PINNED_FILES.items())
        and _safe_regular(
            Path(A06_RUNTIME) / "pre_run/final_prerun_evidence.json",
            A07_ORIGINAL_PRERUN_SHA256,
        )
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 rehearsal sealed reference projection differs"
        )
    signature = inspect.signature(core.validate_amended_run_reference)
    if not {
        "amendment06_allowed_after_hashes", "amendment06_change_ledger_evidence",
    }.issubset(signature.parameters):
        raise Amendment06IntegrityError(
            "Production reference validator signature lacks Amendment 06 routing"
        )
    preflight_identity = strict_full_load_file(
        Path(preflight_run) / "config/run_identity.lock.json"
    )
    smoke_identity = strict_full_load_file(
        Path(smoke_run) / "config/run_identity.lock.json"
    )
    if not (
        preflight_identity.get("run_kind") == "AMENDED_PREFLIGHT"
        and preflight_identity.get("state") == "PREFLIGHT_COMPLETE"
        and preflight_identity.get("full_authorized") is False
        and preflight_identity.get("candidate_split_hash") == SPLIT_ASSIGNMENT_SHA256
        and smoke_identity.get("run_id") == ACCEPTED_SMOKE.name
        and smoke_identity.get("run_kind") == "CUDA_SMOKE_NON_SCIENTIFIC"
        and smoke_identity.get("state") == "SMOKE_COMPLETE"
        and smoke_identity.get("full_authorized") is False
        and smoke_identity.get("candidate_split_hash") == SPLIT_ASSIGNMENT_SHA256
        and smoke_identity.get("full_training_executed") is False
        and smoke_identity.get("scientific_results_claimed") is False
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 rehearsal sealed reference identity differs"
        )
    result = {
        "status": "PASS", "validation_mode": "SEALED_EVIDENCE_PROJECTION",
        "pre_edit_admission_sha256": sha256_file(PRE_EDIT_ADMISSION),
        "original_final_prerun_evidence_sha256": A07_ORIGINAL_PRERUN_SHA256,
        "accepted_preflight_run_id": Path(preflight_run).name,
        "accepted_smoke_run_id": Path(smoke_run).name,
        "sealed_smoke_semantic_status": "PASS",
        "production_validator_signature_status": "PASS",
        "external_audit_tabular_parse_calls": 0,
    }
    _require_exact_keys(
        result, set(A08_REHEARSAL_REFERENCE_PROJECTION_KEYS),
        label="Amendment 08 rehearsal reference projection",
    )
    return result


def _amendment08_projection_basis_sha256(payload: Mapping[str, Any]) -> str:
    excluded = {"projection_basis_sha256", "source_projection", "package_projection"}
    basis = {key: value for key, value in payload.items() if key not in excluded}
    return sha256_bytes(strict_full_canonical_json_bytes({
        "domain": A08_ISOLATED_REHEARSAL_PROJECTION_DOMAIN,
        "projection": basis,
    }))


def _amendment08_projection_derived_sha256(basis_sha256: str, label: str) -> str:
    return sha256_bytes(
        (
            A08_ISOLATED_REHEARSAL_PROJECTION_DOMAIN + "\0"
            + basis_sha256 + "\0" + label
        ).encode("utf-8")
    )


def _amendment08_projection_bindings(
    basis_sha256: str, final_source_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    source = {
        "source_freeze_receipt_sha256": _amendment08_projection_derived_sha256(
            basis_sha256, "source_freeze_receipt",
        ),
        "current_a06_authorized_change_ledger_sha256": (
            _amendment08_projection_derived_sha256(
                basis_sha256, "current_a06_authorized_change_ledger",
            )
        ),
        "a08_source_ledger_event_index": 5,
        "current_a06_final_prerun_evidence_sha256": (
            _amendment08_projection_derived_sha256(
                basis_sha256, "current_a06_final_prerun_evidence",
            )
        ),
        "a08_source_ledger_event_canonical_sha256": (
            _amendment08_projection_derived_sha256(
                basis_sha256, "a08_source_ledger_event",
            )
        ),
    }
    package = {
        "lineage_sha256": _amendment08_projection_derived_sha256(
            basis_sha256, "package_lineage",
        ),
        "receipt_sha256": _amendment08_projection_derived_sha256(
            basis_sha256, "package_receipt",
        ),
        "final_code_sha256": final_source_hashes[
            "code/amendment06_packaging.py"
        ],
        "final_test_sha256": final_source_hashes[
            "tests/test_amendment06_packaging.py"
        ],
    }
    return source, package


def install_amendment08_isolated_rehearsal_projection(
    *, run_dir: Path, preflight_run: Path | None = None,
    smoke_run: Path | None = None, recorded_at: str | None = None,
) -> dict[str, Any]:
    requested_run = Path(run_dir).resolve()
    context_run, runtime_root = _amendment08_rehearsal_context_paths()
    if requested_run != context_run or requested_run != Path(A08_TARGET_RUN).resolve():
        raise Amendment06IntegrityError("Amendment 08 rehearsal projection target differs")
    run_dir = requested_run
    if preflight_run is None:
        preflight_run = ACCEPTED_PREFLIGHT
    if smoke_run is None:
        smoke_run = ACCEPTED_SMOKE
    path = amendment08_isolated_rehearsal_projection_path(run_dir)
    if path.exists():
        return validate_amendment08_isolated_rehearsal_projection(run_dir)
    precorrection = validate_amendment08_precorrection(run_dir)
    base_run = _validate_amendment08_rehearsal_base_run(
        run_dir, require_exact_set=True,
    )
    cache = _amendment08_rehearsal_cache_clone_evidence(
        run_dir, rehash_source_and_clone=True,
    )
    reference = _validate_amendment08_rehearsal_reference_projection(
        preflight_run=preflight_run, smoke_run=smoke_run,
    )
    relocation = _validate_amendment08_rehearsal_authorization_relocation(run_dir)
    final_source_hashes = {
        relative: sha256_file(STUDY_ROOT / relative)
        for relative in A08_CHANGED_ALL_RELATIVES
    }
    if any(
        not _safe_regular(STUDY_ROOT / relative, digest)
        for relative, digest in final_source_hashes.items()
    ):
        raise Amendment06IntegrityError("Amendment 08 rehearsal live source bytes are unsafe")
    base = precorrection["base_history"]
    raw = precorrection["raw_journal"]
    payload: dict[str, Any] = {
        "schema": A08_ISOLATED_REHEARSAL_PROJECTION_SCHEMA,
        "status": "PASS", "mode": A08_ISOLATED_REHEARSAL_PROJECTION_MODE,
        "authorization": A08_AUTHORIZATION, "prompt_sha256": A08_PROMPT_SHA256,
        "target_run_id": run_dir.name, "run_path": str(run_dir),
        "runtime_path": str(runtime_root), "recorded_at": recorded_at or _now(),
        "base_identity_sha256": base["identity_sha256"],
        "base_scientific_manifest_sha256": base[
            "base_scientific_manifest_sha256"
        ],
        "base_code_lineage_sha256": base["base_code_lineage_sha256"],
        "initial_package_validator_manifest_sha256": (
            A08_INITIAL_PACKAGE_MANIFEST_SHA256
        ),
        "initial_package_hashes": dict(A08_INITIAL_PACKAGE_HASHES),
        "final_source_hashes": final_source_hashes,
        "raw_events_0_to_6_concat_sha256": raw[
            "raw_events_0_to_6_concat_sha256"
        ],
        "canonical_json_lines_events_0_to_6_sha256": raw[
            "canonical_json_lines_events_0_to_6_sha256"
        ],
        "forensic_zip_sha256": A08_FORENSIC_ZIP_SHA256,
        "forensic_verification_sha256": A08_FORENSIC_VERIFICATION_SHA256,
        **{key: base_run[key] for key in (
            "base_run_regular_file_count", "base_run_inventory_sha256",
        )},
        **{key: cache[key] for key in (
            "cache_source_path", "cache_clone_path", "cache_member_count",
            "cache_member_bytes", "cache_inventory_file_count",
            "cache_source_inventory_sha256", "cache_clone_inventory_sha256",
            "cache_clone_distinct_inodes", "cache_clone_records",
        )},
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        "frozen_population_sha256": A07_FROZEN_POPULATION_SHA256,
        "portable_imputer_sha256": A07_PORTABLE_IMPUTER_SHA256,
        "reference_projection": reference,
        "authorization_relocation": relocation,
        "new_full_run_ids_created": 0, "new_resampling_realizations": 0,
        "production_amendment08_receipts_published": 0,
        "production_authorization_consumed": False,
        "projection_durable_authorization_consumption": False,
    }
    basis_sha256 = _amendment08_projection_basis_sha256(payload)
    source, package = _amendment08_projection_bindings(
        basis_sha256, final_source_hashes,
    )
    payload.update({
        "source_projection": source, "package_projection": package,
        "projection_basis_sha256": basis_sha256,
    })
    _require_exact_keys(
        payload, set(A08_ISOLATED_REHEARSAL_PROJECTION_KEYS),
        label="Amendment 08 isolated rehearsal projection",
    )
    publish_strict_json_no_clobber(path, payload)
    return validate_amendment08_isolated_rehearsal_projection(run_dir)


def validate_amendment08_isolated_rehearsal_projection(
    run_dir: Path | None = None,
) -> dict[str, Any]:
    context_run, runtime_root = _amendment08_rehearsal_context_paths()
    selected = context_run if run_dir is None else Path(run_dir).resolve()
    path = amendment08_isolated_rehearsal_projection_path(selected)
    payload = _require_exact_keys(
        strict_full_load_file(path), set(A08_ISOLATED_REHEARSAL_PROJECTION_KEYS),
        label="Amendment 08 isolated rehearsal projection",
    )
    final_source_hashes = payload["final_source_hashes"]
    if not isinstance(final_source_hashes, Mapping) or set(final_source_hashes) != set(
        A08_CHANGED_ALL_RELATIVES
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 rehearsal final-source mapping differs"
        )
    if not (
        payload["schema"] == A08_ISOLATED_REHEARSAL_PROJECTION_SCHEMA
        and payload["status"] == "PASS"
        and payload["mode"] == A08_ISOLATED_REHEARSAL_PROJECTION_MODE
        and payload["authorization"] == A08_AUTHORIZATION
        and payload["prompt_sha256"] == A08_PROMPT_SHA256
        and payload["target_run_id"] == selected.name == A08_TARGET_RUN_ID
        and payload["run_path"] == str(selected)
        and payload["runtime_path"] == str(runtime_root)
        and isinstance(payload["recorded_at"], str) and payload["recorded_at"]
        and payload["base_identity_sha256"] == A07_BASE_IDENTITY_SHA256
        and payload["base_scientific_manifest_sha256"]
        == A07_BASE_SCIENTIFIC_MANIFEST_SHA256
        and payload["base_code_lineage_sha256"] == A07_BASE_CODE_LINEAGE_SHA256
        and payload["initial_package_validator_manifest_sha256"]
        == A08_INITIAL_PACKAGE_MANIFEST_SHA256
        and payload["initial_package_hashes"] == dict(A08_INITIAL_PACKAGE_HASHES)
        and payload["raw_events_0_to_6_concat_sha256"]
        == A08_RAW_EVENTS_0_TO_6_CONCAT_SHA256
        and payload["canonical_json_lines_events_0_to_6_sha256"]
        == A08_CANONICAL_EVENTS_0_TO_6_LINES_SHA256
        and payload["forensic_zip_sha256"] == A08_FORENSIC_ZIP_SHA256
        and payload["forensic_verification_sha256"]
        == A08_FORENSIC_VERIFICATION_SHA256
        and _amendment08_exact_int(payload["base_run_regular_file_count"], 45)
        and payload["cache_manifest_sha256"] == A07_CACHE_MANIFEST_SHA256
        and payload["frozen_population_sha256"] == A07_FROZEN_POPULATION_SHA256
        and payload["portable_imputer_sha256"] == A07_PORTABLE_IMPUTER_SHA256
        and _amendment08_exact_int(payload["cache_member_count"], 10)
        and _amendment08_exact_int(
            payload["cache_member_bytes"], A08_CACHE_MEMBER_BYTES,
        )
        and _amendment08_exact_int(payload["cache_inventory_file_count"], 11)
        and payload["cache_source_inventory_sha256"]
        == payload["cache_clone_inventory_sha256"]
        and payload["cache_clone_distinct_inodes"] is True
        and _amendment08_exact_int(payload["new_full_run_ids_created"], 0)
        and _amendment08_exact_int(payload["new_resampling_realizations"], 0)
        and _amendment08_exact_int(
            payload["production_amendment08_receipts_published"], 0,
        )
        and payload["production_authorization_consumed"] is False
        and payload["projection_durable_authorization_consumption"] is False
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 isolated rehearsal projection semantics differ"
        )
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", str(digest))
        or not _safe_regular(STUDY_ROOT / relative, str(digest))
        for relative, digest in final_source_hashes.items()
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 rehearsal projected live-source bytes drifted"
        )
    if not (
        _safe_regular(A08_FORENSIC_ZIP, A08_FORENSIC_ZIP_SHA256)
        and A08_FORENSIC_ZIP.stat().st_size == A08_FORENSIC_ZIP_SIZE_BYTES
        and _safe_regular(
            A08_FORENSIC_VERIFICATION, A08_FORENSIC_VERIFICATION_SHA256,
        )
        and A08_FORENSIC_VERIFICATION.stat().st_size
        == A08_FORENSIC_VERIFICATION_SIZE_BYTES
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 rehearsal forensic projection bytes drifted"
        )
    _validate_amendment08_base_history(selected)
    raw = _validate_amendment08_raw_prefix(selected, require_exact_boundary=False)
    base_run = _validate_amendment08_rehearsal_base_run(
        selected, require_exact_set=False,
    )
    cache = _amendment08_rehearsal_cache_clone_evidence(
        selected, rehash_source_and_clone=False,
    )
    reference = _validate_amendment08_rehearsal_reference_projection(
        preflight_run=ACCEPTED_PREFLIGHT, smoke_run=ACCEPTED_SMOKE,
    )
    relocation = _validate_amendment08_rehearsal_authorization_relocation(selected)
    if not (
        payload["base_run_inventory_sha256"]
        == base_run["base_run_inventory_sha256"]
        and payload["raw_events_0_to_6_concat_sha256"]
        == raw["raw_events_0_to_6_concat_sha256"]
        and payload["reference_projection"] == reference
        and payload["authorization_relocation"] == relocation
        and payload["cache_source_path"] == cache["cache_source_path"]
        and payload["cache_clone_path"] == cache["cache_clone_path"]
        and payload["cache_source_inventory_sha256"]
        == cache["cache_source_inventory_sha256"]
        and payload["cache_clone_inventory_sha256"]
        == cache["cache_clone_inventory_sha256"]
        and payload["cache_clone_records"] == cache["cache_clone_records"]
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 isolated rehearsal projection evidence drifted"
        )
    basis_sha256 = _amendment08_projection_basis_sha256(payload)
    source, package = _amendment08_projection_bindings(
        basis_sha256, final_source_hashes,
    )
    if not (
        payload["projection_basis_sha256"] == basis_sha256
        and set(payload["source_projection"])
        == set(A08_REHEARSAL_SOURCE_PROJECTION_KEYS)
        and dict(payload["source_projection"]) == source
        and set(payload["package_projection"])
        == set(A08_REHEARSAL_PACKAGE_PROJECTION_KEYS)
        and dict(payload["package_projection"]) == package
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 isolated rehearsal projection derivation differs"
        )
    return {
        "status": "PASS", "mode": A08_ISOLATED_REHEARSAL_PROJECTION_MODE,
        "authorization": A08_AUTHORIZATION, "prompt_sha256": A08_PROMPT_SHA256,
        "target_run_id": selected.name, "projection_path": str(path.resolve()),
        "projection_sha256": sha256_file(path),
        "projection_basis_sha256": basis_sha256,
        "final_source_hashes": dict(final_source_hashes),
        "source_projection": source, "package_projection": package,
        "cache_source_inventory_sha256": cache[
            "cache_source_inventory_sha256"
        ],
        "cache_clone_inventory_sha256": cache["cache_clone_inventory_sha256"],
        "cache_clone_distinct_inodes": True,
        "reference_projection": reference, "authorization_relocation": relocation,
    }


def _amendment08_overlay_admission_bindings(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _AMENDMENT08_REHEARSAL_CONTEXT.get() is None:
        return (
            validate_amendment08_source_freeze(),
            validate_amendment08_prelive_package_lineage(run_dir),
        )
    projection = validate_amendment08_isolated_rehearsal_projection(run_dir)
    source_projection = dict(projection["source_projection"])
    package_projection = dict(projection["package_projection"])
    final_hashes = dict(projection["final_source_hashes"])
    source = {
        "status": "PASS", "authorization": A08_AUTHORIZATION,
        "prompt_sha256": A08_PROMPT_SHA256, "target_run_id": run_dir.name,
        **source_projection,
        "final_source_hashes": final_hashes,
        "projection_mode": A08_ISOLATED_REHEARSAL_PROJECTION_MODE,
        "projection_sha256": projection["projection_sha256"],
    }
    package = {
        "status": "PASS", "authorization": A08_AUTHORIZATION,
        "prompt_sha256": A08_PROMPT_SHA256, "target_run_id": run_dir.name,
        "package_set_id": A08_PACKAGE_SET_ID,
        "initial_package_validator_manifest_sha256": (
            A08_INITIAL_PACKAGE_MANIFEST_SHA256
        ),
        "initial_hashes": dict(A08_INITIAL_PACKAGE_HASHES),
        "final_hashes": {
            relative: final_hashes[relative]
            for relative in A08_CHANGED_PACKAGE_RELATIVES
        },
        **package_projection,
        "event_count": 2, "scientific_api_calls": 0,
        "projection_mode": A08_ISOLATED_REHEARSAL_PROJECTION_MODE,
        "projection_sha256": projection["projection_sha256"],
    }
    return source, package


def _effective_amendment08_manifest_bytes(run_dir: Path) -> bytes:
    base_path = run_dir / "provenance/scientific_execution_code_manifest.tsv"
    if not _safe_regular(base_path, A07_BASE_SCIENTIFIC_MANIFEST_SHA256):
        raise Amendment06IntegrityError("Amendment 08 base scientific manifest drifted")
    rows = _parse_code_manifest_bytes(base_path.read_bytes(), label="A08 base manifest")
    if [str(row["relative_path"]) for row in rows] != list(
        (*SCIENTIFIC_CODE_RELATIVES, *SCIENTIFIC_TEST_RELATIVES)
    ):
        raise Amendment06IntegrityError("Amendment 08 base scientific manifest order differs")
    changed: list[str] = []
    for row in rows:
        relative = str(row["relative_path"])
        if relative not in A08_CHANGED_SCIENTIFIC_RELATIVES:
            continue
        source = STUDY_ROOT / relative
        row["size_bytes"] = source.stat().st_size
        row["sha256"] = sha256_file(source)
        changed.append(relative)
    if changed != list(A08_CHANGED_SCIENTIFIC_RELATIVES):
        raise Amendment06IntegrityError("Amendment 08 scientific delta differs")
    return _manifest_bytes(rows)


def _validate_amendment08_effective_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "provenance/scientific_execution_code_manifest_effective_amendment08.tsv"
    expected = _effective_amendment08_manifest_bytes(run_dir)
    if not (_safe_regular(path, sha256_bytes(expected)) and path.read_bytes() == expected):
        raise Amendment06IntegrityError("Amendment 08 effective manifest differs")
    base_rows = _parse_code_manifest_bytes(
        (run_dir / "provenance/scientific_execution_code_manifest.tsv").read_bytes(),
        label="Amendment 08 base manifest",
    )
    effective_rows = _parse_code_manifest_bytes(
        expected, label="Amendment 08 effective manifest",
    )
    changed: list[str] = []
    for base, effective in zip(base_rows, effective_rows, strict=True):
        relative = str(effective["relative_path"])
        live = STUDY_ROOT / relative
        if not (
            base["relative_path"] == relative
            and effective["role"] == "SCIENTIFIC_EXECUTION_OR_TEST"
            and _safe_regular(live, str(effective["sha256"]))
            and live.stat().st_size == int(effective["size_bytes"])
        ):
            raise Amendment06IntegrityError(
                f"Amendment 08 effective live source drifted: {relative}"
            )
        if (base["size_bytes"], base["sha256"]) != (
            effective["size_bytes"], effective["sha256"],
        ):
            changed.append(relative)
    if changed != list(A08_CHANGED_SCIENTIFIC_RELATIVES):
        raise Amendment06IntegrityError(
            f"Amendment 08 effective changed-path set differs: {changed}"
        )
    snapshots = {
        "code/amendment06_full.py": (
            run_dir / "provenance/code_snapshot_amendment08/amendment06_full.py"
        ),
        "tests/test_amendment06.py": (
            run_dir / "provenance/tests_snapshot_amendment08/test_amendment06.py"
        ),
    }
    by_relative = {str(row["relative_path"]): row for row in effective_rows}
    if any(
        not _safe_regular(snapshot, str(by_relative[relative]["sha256"]))
        or snapshot.stat().st_size != int(by_relative[relative]["size_bytes"])
        for relative, snapshot in snapshots.items()
    ):
        raise Amendment06IntegrityError("Amendment 08 corrected source snapshot drifted")
    return {
        "base_manifest_sha256": A07_BASE_SCIENTIFIC_MANIFEST_SHA256,
        "effective_manifest_sha256": sha256_bytes(expected),
        "changed_scientific_paths": changed,
        "final_scientific_hashes": {
            relative: sha256_file(STUDY_ROOT / relative)
            for relative in A08_CHANGED_SCIENTIFIC_RELATIVES
        },
    }


def _amendment08_package_paths(run_dir: Path) -> dict[str, Path]:
    lineage_root = (
        Path(EXPECTED_RUNTIME_ROOT) / "package_code_lineage" / run_dir.name
    )
    prelive = lineage_root / "amendment08/prelive"
    return {
        "lineage": lineage_root / "package_code_lineage.jsonl",
        "intent": prelive / "install_intent.json",
        "receipt": prelive / "install_receipt.json",
        "event0": prelive / "events/00000001.json",
        "event1": prelive / "events/00000002.json",
        "evidence0": prelive / "evidence/00000001.json",
        "evidence1": prelive / "evidence/00000002.json",
        "code_snapshot": prelive / "snapshots/code/amendment06_packaging.py",
        "test_snapshot": prelive / "snapshots/tests/test_amendment06_packaging.py",
    }


def _validate_amendment08_package_log_binding(value: Any, *, label: str) -> dict[str, Any]:
    record = _require_exact_keys(
        value, {"path", "size_bytes", "sha256", "exit_code", "command"},
        label=label,
    )
    declared_path = Path(str(record["path"]))
    if not declared_path.is_absolute():
        raise Amendment06IntegrityError(f"{label} path is not absolute")
    try:
        path = declared_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Amendment06IntegrityError(f"{label} path cannot be resolved") from exc
    if not (
        declared_path == path and isinstance(record["size_bytes"], int)
        and not isinstance(record["size_bytes"], bool) and record["size_bytes"] >= 0
        and re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"]))
        and record["exit_code"] == 0 and not isinstance(record["exit_code"], bool)
        and isinstance(record["command"], str) and record["command"].strip()
        and _safe_regular(path)
    ):
        raise Amendment06IntegrityError(f"{label} differs")
    data = path.read_bytes()
    try:
        lines = data.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise Amendment06IntegrityError(f"{label} is not UTF-8") from exc
    pytest_success = re.compile(
        r"^\d+ passed(?:, \d+ skipped)? in \d+(?:\.\d+)?s$", re.IGNORECASE,
    )
    status_success = re.compile(
        r"^STATUS=(?:PASS|PASS_EXPECTED_FAIL_CLOSED)$", re.IGNORECASE,
    )
    failed_summary = re.compile(
        r"(?:^|[, ])\d+ (?:failed|errors?)(?:[, ]|$)", re.IGNORECASE,
    )
    if (
        not lines
        or any(failed_summary.search(line) for line in lines)
        or any(
            re.fullmatch(
                r"STATUS=(?:FAIL|FAILED|ERROR).*", line.strip(), re.IGNORECASE,
            )
            for line in lines
        )
        or not any(
            pytest_success.fullmatch(line.strip())
            or status_success.fullmatch(line.strip())
            for line in lines
        )
    ):
        raise Amendment06IntegrityError(f"{label} has no unambiguous success trailer")
    checked = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_bytes(data),
        "exit_code": 0,
        "command": record["command"].strip(),
    }
    if (
        checked != dict(record)
        or len(data) != checked["size_bytes"]
        or not _safe_regular(path, checked["sha256"])
    ):
        raise Amendment06IntegrityError(f"{label} binding differs")
    return checked


def _amendment08_exact_int(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def validate_amendment08_prelive_package_lineage(
    run_dir: Path | None = None,
) -> dict[str, Any]:
    if run_dir is None:
        run_dir = A08_TARGET_RUN
    run_dir = Path(run_dir).resolve()
    if run_dir.name != A08_TARGET_RUN_ID:
        raise Amendment06IntegrityError("A08 package lineage target differs")
    paths = _amendment08_package_paths(run_dir)
    root = paths["lineage"].parent
    lock_path = paths["intent"].parent / "install.lock"
    if not (
        root.is_dir() and not root.is_symlink()
        and root.resolve(strict=True) == root
        and all(
            _safe_regular(path) and path.resolve(strict=True) == path
            for path in (*paths.values(), lock_path)
        )
    ):
        raise Amendment06IntegrityError("A08 pre-live package lineage is incomplete")
    observed_members: set[str] = set()
    for directory, names, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        if parent.is_symlink() or parent.resolve(strict=True) != parent:
            raise Amendment06IntegrityError("A08 package lineage directory is unsafe")
        for name in names:
            child = parent / name
            if not stat.S_ISDIR(os.lstat(child).st_mode) or child.is_symlink():
                raise Amendment06IntegrityError(
                    f"A08 package lineage directory is unsafe: {child}"
                )
        for name in filenames:
            child = parent / name
            if not _safe_regular(child) or child.resolve(strict=True) != child:
                raise Amendment06IntegrityError(
                    f"A08 package lineage file is unsafe: {child}"
                )
            observed_members.add(child.relative_to(root).as_posix())
    expected_members = {
        path.relative_to(root).as_posix() for path in (*paths.values(), lock_path)
    }
    if observed_members != expected_members:
        raise Amendment06IntegrityError(
            "A08 pre-live package lineage member inventory differs: "
            f"missing={sorted(expected_members - observed_members)}, "
            f"extra={sorted(observed_members - expected_members)}"
        )
    intent = _require_exact_keys(
        strict_full_load_file(paths["intent"]), set(A08_PACKAGE_LINEAGE_INTENT_KEYS),
        label="A08 package lineage intent",
    )
    receipt = _require_exact_keys(
        strict_full_load_file(paths["receipt"]), set(A08_PACKAGE_LINEAGE_RECEIPT_KEYS),
        label="A08 package lineage receipt",
    )
    events = [
        _require_exact_keys(
            strict_full_load_file(paths[f"event{index}"]),
            set(A08_PACKAGE_LINEAGE_EVENT_KEYS),
            label=f"A08 package lineage event {index}",
        )
        for index in range(2)
    ]
    for index, event in enumerate(events):
        expected_event_bytes = strict_full_canonical_json_bytes(event) + b"\n"
        if paths[f"event{index}"].read_bytes() != expected_event_bytes:
            raise Amendment06IntegrityError(
                f"A08 package lineage event sidecar {index} bytes differ"
            )
    lineage_bytes = b"".join(
        strict_full_canonical_json_bytes(event) + b"\n" for event in events
    )
    if paths["lineage"].read_bytes() != lineage_bytes:
        raise Amendment06IntegrityError("A08 package lineage JSONL bytes differ")
    if any(not _safe_regular(STUDY_ROOT / relative) for relative in A08_CHANGED_PACKAGE_RELATIVES):
        raise Amendment06IntegrityError("A08 live package code/test is unsafe")
    final_hashes = {
        relative: sha256_file(STUDY_ROOT / relative)
        for relative in A08_CHANGED_PACKAGE_RELATIVES
    }
    initial = dict(A08_INITIAL_PACKAGE_HASHES)
    forensic = {
        "code/amendment06_packaging.py": A08_PRE_PACKAGE_SHA256,
        "tests/test_amendment06_packaging.py": A08_PRE_PACKAGE_TEST_SHA256,
    }
    if not (
        intent["schema"] == "amendment08_package_code_lineage_install_intent/v1"
        and intent["status"] == "INTENT"
        and intent["authorization"] == A08_AUTHORIZATION
        and intent["prompt_sha256"] == A08_PROMPT_SHA256
        and intent["target_run_id"] == run_dir.name
        and intent["package_set_id"] == A08_PACKAGE_SET_ID
        and intent["set_kind"] == "PRELIVE"
        and isinstance(intent["created_at"], str) and bool(intent["created_at"])
        and intent["initial_package_validator_manifest_sha256"]
        == A08_INITIAL_PACKAGE_MANIFEST_SHA256
        and intent["initial_hashes"] == initial
        and intent["forensic_pre_a08_draft_hashes"] == forensic
        and intent["final_hashes"] == final_hashes
        and intent["base_lineage_event_count"] == 0
        and intent["base_lineage_head_sha256"] is None
        and _amendment08_exact_int(intent["base_lineage_event_count"], 0)
        and _amendment08_exact_int(intent["event_count"], 2)
        and isinstance(intent["records"], list) and len(intent["records"]) == 2
    ):
        raise Amendment06IntegrityError("A08 package lineage intent semantics differ")
    logs = {
        "targeted_tests": _validate_amendment08_package_log_binding(
            intent["targeted_tests_log"], label="A08 targeted-tests log",
        ),
        "package_rehearsal": _validate_amendment08_package_log_binding(
            intent["package_rehearsal_log"], label="A08 package-rehearsal log",
        ),
    }
    rehearsal_binding = _require_exact_keys(
        intent["package_rehearsal_evidence"],
        {"path", "size_bytes", "sha256"},
        label="A08 package rehearsal evidence binding",
    )
    declared_rehearsal_path = Path(str(rehearsal_binding["path"]))
    try:
        rehearsal_path = declared_rehearsal_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Amendment06IntegrityError(
            "A08 package rehearsal evidence path cannot be resolved"
        ) from exc
    if not (
        declared_rehearsal_path.is_absolute()
        and declared_rehearsal_path == rehearsal_path
        and isinstance(rehearsal_binding["size_bytes"], int)
        and not isinstance(rehearsal_binding["size_bytes"], bool)
        and rehearsal_binding["size_bytes"] >= 0
        and re.fullmatch(r"[0-9a-f]{64}", str(rehearsal_binding["sha256"]))
        and _safe_regular(rehearsal_path, str(rehearsal_binding["sha256"]))
        and rehearsal_path.stat().st_size == rehearsal_binding["size_bytes"]
    ):
        raise Amendment06IntegrityError("A08 package rehearsal evidence binding differs")
    target_order = list(A08_CHANGED_PACKAGE_RELATIVES)
    rehearsal = _require_exact_keys(
        strict_full_load_file(rehearsal_path),
        set(A08_PACKAGE_REHEARSAL_EVIDENCE_KEYS),
        label="A08 package rehearsal evidence",
    )
    guarded_names = sorted(A08_REQUIRED_ZERO_SCIENCE_API_NAMES)
    if len(A08_REQUIRED_ZERO_SCIENCE_API_NAMES) != 81:
        raise Amendment06IntegrityError("A08 zero-science API inventory differs")
    if not (
        rehearsal["schema"] == "amendment08_package_rehearsal_evidence/v1"
        and rehearsal["status"] == "PASS"
        and rehearsal["authorization"] == A08_AUTHORIZATION
        and rehearsal["prompt_sha256"] == A08_PROMPT_SHA256
        and rehearsal["target_run_id"] == run_dir.name
        and rehearsal["package_set_id"] == A08_PACKAGE_SET_ID
        and isinstance(rehearsal["recorded_at"], str) and bool(rehearsal["recorded_at"])
        and rehearsal["fixture_run_id"] == run_dir.name == A08_TARGET_RUN_ID
        and rehearsal["targeted_tests_log"] == logs["targeted_tests"]
        and rehearsal["package_rehearsal_log"] == logs["package_rehearsal"]
        and rehearsal["initial_package_validator_manifest_sha256"]
        == A08_INITIAL_PACKAGE_MANIFEST_SHA256
        and rehearsal["initial_hashes"] == initial
        and rehearsal["final_hashes"] == final_hashes
        and _amendment08_exact_int(rehearsal["expected_run_manifest_rows"], 236)
        and _amendment08_exact_int(rehearsal["actual_run_regular_files"], 237)
        and _amendment08_exact_int(rehearsal["review_selected_run_rows"], 226)
        and _amendment08_exact_int(rehearsal["review_embedded_manifest_rows"], 227)
        and _amendment08_exact_int(rehearsal["review_zip_file_entry_count"], 228)
        and _amendment08_exact_int(rehearsal["models_selected_run_rows"], 101)
        and _amendment08_exact_int(rehearsal["models_embedded_manifest_rows"], 102)
        and _amendment08_exact_int(rehearsal["models_zip_file_entry_count"], 103)
        and rehearsal["receipt_only_validation"] == "PASS"
        and rehearsal["independent_zip_reopen"] == "PASS"
        and rehearsal["package_retry_status"] == "PASS"
        and _amendment08_exact_int(rehearsal["guarded_scientific_api_calls"], 0)
        and rehearsal["guarded_api_names"] == guarded_names
        and _amendment08_exact_int(rehearsal["run_mutation_calls"], 0)
        and _amendment08_exact_int(rehearsal["live_runtime_journal_reads"], 0)
    ):
        raise Amendment06IntegrityError("A08 package rehearsal evidence differs")
    checked_rehearsal_binding = {
        "path": str(rehearsal_path),
        "size_bytes": rehearsal_path.stat().st_size,
        "sha256": sha256_file(rehearsal_path),
    }
    if dict(rehearsal_binding) != checked_rehearsal_binding:
        raise Amendment06IntegrityError("A08 package rehearsal evidence binding drifted")

    event_hashes: list[str] = []
    evidence_hashes: list[str] = []
    snapshot_hashes: dict[str, str] = {}
    for index, (event, relative, raw_record) in enumerate(zip(
        events, target_order, intent["records"], strict=True,
    )):
        sequence = index + 1
        expected_previous = (
            None if index == 0
            else sha256_bytes(strict_full_canonical_json_bytes(events[index - 1]))
        )
        evidence_path = paths[f"evidence{index}"]
        snapshot_path = paths["code_snapshot" if index == 0 else "test_snapshot"]
        evidence = _require_exact_keys(
            strict_full_load_file(evidence_path),
            set(A08_PACKAGE_LINEAGE_EVIDENCE_KEYS),
            label=f"A08 package rehearsal guard {sequence}",
        )
        evidence_sha256 = sha256_file(evidence_path)
        snapshot_sha256 = sha256_file(snapshot_path)
        event_canonical_sha256 = sha256_bytes(strict_full_canonical_json_bytes(event))
        if not (
            event["schema"] == "amendment08_package_code_lineage_event/v1"
            and event["status"] == "PASS"
            and event["authorization"] == A08_AUTHORIZATION
            and event["prompt_sha256"] == A08_PROMPT_SHA256
            and event["target_run_id"] == run_dir.name
            and event["package_set_id"] == A08_PACKAGE_SET_ID
            and _amendment08_exact_int(event["set_event_index"], index)
            and _amendment08_exact_int(event["sequence"], sequence)
            and event["target_relative_path"] == relative
            and event["previous_code_sha256"] == initial[relative]
            and event["new_code_sha256"] == final_hashes[relative]
            and event["previous_event_sha256"] == expected_previous
            and event["initial_package_validator_manifest_sha256"]
            == A08_INITIAL_PACKAGE_MANIFEST_SHA256
            and event["targeted_tests_log_sha256"] == logs["targeted_tests"]["sha256"]
            and event["package_rehearsal_log_sha256"]
            == logs["package_rehearsal"]["sha256"]
            and event["package_rehearsal_evidence_sha256"]
            == checked_rehearsal_binding["sha256"]
            and event["zero_science_evidence_path"] == str(evidence_path.resolve())
            and event["zero_science_evidence_sha256"] == evidence_sha256
            and event["source_snapshot_path"] == str(snapshot_path.resolve())
            and event["source_snapshot_sha256"] == final_hashes[relative]
            and snapshot_sha256 == final_hashes[relative]
            and _amendment08_exact_int(event["scientific_api_calls"], 0)
        ):
            raise Amendment06IntegrityError(f"A08 package lineage event {index} differs")
        if not (
            evidence["schema"] == "amendment08_package_rehearsal_guard/v1"
            and evidence["status"] == "PASS"
            and evidence["authorization"] == A08_AUTHORIZATION
            and evidence["prompt_sha256"] == A08_PROMPT_SHA256
            and evidence["target_run_id"] == run_dir.name
            and evidence["package_set_id"] == A08_PACKAGE_SET_ID
            and _amendment08_exact_int(evidence["sequence"], sequence)
            and evidence["target_relative_path"] == relative
            and evidence["previous_code_sha256"] == initial[relative]
            and evidence["new_code_sha256"] == final_hashes[relative]
            and evidence["source_snapshot_path"] == str(snapshot_path.resolve())
            and evidence["source_snapshot_sha256"] == final_hashes[relative]
            and evidence["targeted_tests_log"] == logs["targeted_tests"]
            and evidence["package_rehearsal_log"] == logs["package_rehearsal"]
            and evidence["package_rehearsal_evidence_path"]
            == checked_rehearsal_binding["path"]
            and evidence["package_rehearsal_evidence_sha256"]
            == checked_rehearsal_binding["sha256"]
            and _amendment08_exact_int(evidence["guarded_scientific_api_calls"], 0)
            and evidence["guarded_api_names"] == guarded_names
            and _amendment08_exact_int(evidence["run_mutation_calls"], 0)
            and _amendment08_exact_int(evidence["live_runtime_journal_reads"], 0)
            and evidence["receipt_only_validation"] == "PASS"
        ):
            raise Amendment06IntegrityError(
                f"A08 package rehearsal guard {sequence} differs"
            )
        record = _require_exact_keys(
            raw_record, set(A08_PACKAGE_LINEAGE_RECORD_KEYS),
            label=f"A08 package lineage intent record {sequence}",
        )
        expected_record = {
            "sequence": sequence,
            "target_relative_path": relative,
            "previous_code_sha256": initial[relative],
            "new_code_sha256": final_hashes[relative],
            "snapshot_relative_path": snapshot_path.relative_to(root).as_posix(),
            "snapshot_sha256": snapshot_sha256,
            "evidence_relative_path": evidence_path.relative_to(root).as_posix(),
            "evidence_sha256": evidence_sha256,
            "event_relative_path": paths[f"event{index}"].relative_to(root).as_posix(),
            "event_sha256": sha256_file(paths[f"event{index}"]),
            "event_canonical_sha256": event_canonical_sha256,
        }
        if (
            not _amendment08_exact_int(record["sequence"], sequence)
            or dict(record) != expected_record
        ):
            raise Amendment06IntegrityError(
                f"A08 package lineage intent record {sequence} differs"
            )
        event_hashes.append(event_canonical_sha256)
        evidence_hashes.append(evidence_sha256)
        snapshot_hashes[relative] = snapshot_sha256
    if not (
        receipt["schema"] == "amendment08_package_code_lineage_install_receipt/v1"
        and receipt["status"] == "PASS"
        and receipt["authorization"] == A08_AUTHORIZATION
        and receipt["prompt_sha256"] == A08_PROMPT_SHA256
        and receipt["target_run_id"] == run_dir.name
        and receipt["package_set_id"] == A08_PACKAGE_SET_ID
        and receipt["set_kind"] == "PRELIVE"
        and receipt["intent_sha256"] == sha256_file(paths["intent"])
        and receipt["initial_package_validator_manifest_sha256"]
        == A08_INITIAL_PACKAGE_MANIFEST_SHA256
        and receipt["initial_hashes"] == initial
        and receipt["forensic_pre_a08_draft_hashes"] == forensic
        and receipt["final_hashes"] == final_hashes
        and _amendment08_exact_int(receipt["event_count"], 2)
        and receipt["event_hashes"] == event_hashes
        and _amendment08_exact_int(receipt["lineage_event_count"], 2)
        and receipt["lineage_sha256"] == sha256_bytes(lineage_bytes)
        and receipt["lineage_head_sha256"] == event_hashes[-1]
        and receipt["evidence_hashes"] == evidence_hashes
        and receipt["snapshot_hashes"] == snapshot_hashes
        and receipt["targeted_tests_log"] == logs["targeted_tests"]
        and receipt["package_rehearsal_log"] == logs["package_rehearsal"]
        and receipt["package_rehearsal_evidence"] == checked_rehearsal_binding
        and _amendment08_exact_int(receipt["scientific_api_calls"], 0)
        and isinstance(receipt["installed_at"], str) and bool(receipt["installed_at"])
    ):
        raise Amendment06IntegrityError("A08 package lineage receipt differs")
    if any(
        not _safe_regular(STUDY_ROOT / relative, digest)
        for relative, digest in final_hashes.items()
    ):
        raise Amendment06IntegrityError("A08 live package bytes changed during validation")
    return {
        "status": "PASS", "authorization": A08_AUTHORIZATION,
        "prompt_sha256": A08_PROMPT_SHA256, "target_run_id": run_dir.name,
        "package_set_id": A08_PACKAGE_SET_ID,
        "initial_package_validator_manifest_sha256": A08_INITIAL_PACKAGE_MANIFEST_SHA256,
        "initial_hashes": initial, "forensic_pre_a08_draft_hashes": forensic,
        "final_hashes": final_hashes,
        "final_code_sha256": final_hashes["code/amendment06_packaging.py"],
        "final_test_sha256": final_hashes["tests/test_amendment06_packaging.py"],
        "event_count": 2, "lineage_sha256": sha256_bytes(lineage_bytes),
        "lineage_head_sha256": event_hashes[-1],
        "intent_sha256": sha256_file(paths["intent"]),
        "receipt_sha256": sha256_file(paths["receipt"]),
        "scientific_api_calls": 0,
    }


def validate_amendment08_cuda_producer_consumer_probe(path: Path) -> dict[str, Any]:
    keys = {
        "schema", "status", "authorization", "prompt_sha256", "target_run_id",
        "recorded_at", "xgboost_version", "scikit_learn_version", "use_cuda",
        "gpu_identity", "gpu_name", "producer_consumer_status",
        "fit_save_reload_predict_status", "four_way_bit_exact_max_abs_diff",
        "model_bytes_unchanged", "cpu_fallback_detected", "fallback_warnings",
        "active_reloaded_cuda_probe",
    }
    payload = _require_exact_keys(
        strict_full_load_file(path), keys,
        label="Amendment 08 CUDA producer-consumer probe",
    )
    active = payload["active_reloaded_cuda_probe"]
    parsed_name = _gpu_name(payload["gpu_identity"])
    if not (
        payload["schema"] == "amendment08_cuda_producer_consumer_probe/v1"
        and payload["status"] == "PASS"
        and payload["authorization"] == A08_AUTHORIZATION
        and payload["prompt_sha256"] == A08_PROMPT_SHA256
        and payload["target_run_id"] == A08_TARGET_RUN_ID
        and isinstance(payload["recorded_at"], str) and payload["recorded_at"]
        and payload["xgboost_version"] == "2.1.1"
        and payload["scikit_learn_version"] == "1.7.2"
        and payload["use_cuda"] is True and payload["gpu_name"] == parsed_name
        and "RTX 4090" in parsed_name
        and payload["producer_consumer_status"] == "PASS"
        and payload["fit_save_reload_predict_status"] == "PASS"
        and payload["four_way_bit_exact_max_abs_diff"] == 0.0
        and payload["model_bytes_unchanged"] is True
        and payload["cpu_fallback_detected"] is False
        and payload["fallback_warnings"] == []
        and isinstance(active, Mapping) and active.get("status") == "PASS"
        and "cuda" in str(active.get("device", "")).lower()
        and active.get("warnings") == []
    ):
        raise Amendment06IntegrityError("Amendment 08 CUDA probe differs")
    return dict(payload)


def run_amendment08_cuda_producer_consumer_probe(
    path: Path | None = None,
) -> dict[str, Any]:
    if path is None:
        path = A08_RUNTIME / "pre_run/amendment08_cuda_producer_consumer_probe.json"
    path = Path(path)
    if path.exists():
        return validate_amendment08_cuda_producer_consumer_probe(path)
    if not (
        Path(sys.executable).resolve() == CUDA_PYTHON.resolve()
        and Path(sys.prefix).resolve() == CUDA_ENV.resolve()
        and xgb.__version__ == "2.1.1" and sklearn.__version__ == "1.7.2"
        and xgb.build_info().get("USE_CUDA") is True
    ):
        raise Amendment06IntegrityError("Amendment 08 CUDA probe environment differs")
    identity = core.gpu_identity_metadata()
    gpu_name = _gpu_name(identity)
    if "RTX 4090" not in gpu_name:
        raise Amendment06IntegrityError("Amendment 08 CUDA probe GPU is not RTX 4090")
    rng = np.random.default_rng(20260820)
    matrix = np.ascontiguousarray(rng.normal(size=(128, 4)), dtype=np.float32)
    labels = (matrix[:, 0] + 0.4 * matrix[:, 1] > 0.0).astype(np.int8)
    classifier = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=24,
        learning_rate=0.15, max_depth=3, subsample=1.0, colsample_bytree=1.0,
        tree_method="hist", device="cuda", random_state=42, n_jobs=2, verbosity=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier.fit(matrix, labels, eval_set=[(matrix, labels)], verbose=False)
    warning_text = [str(item.message) for item in caught]
    fallback = _fallback_warnings(warning_text)
    configuration = strict_full_loads(classifier.get_booster().save_config())
    if fallback or "cuda" not in json.dumps(configuration).lower():
        raise Amendment06IntegrityError("Amendment 08 CUDA probe fit/fallback gate failed")
    features = ("probe_0", "probe_1", "probe_2", "probe_3")
    feature_sha256 = _feature_list_sha256(features)
    model_compat.embed_binary_classifier_metadata(
        classifier, expected_feature_count=4,
        feature_list_sha256=feature_sha256,
    )
    work = A08_RUNTIME / "probe_work" / uuid.uuid4().hex
    work.mkdir(parents=True, exist_ok=False)
    model_path = work / "amendment08_cuda_probe.ubj"
    try:
        classifier.save_model(model_path)
        before = sha256_file(model_path)
        parity = model_compat.four_way_binary_reload_parity(
            classifier, model_path, matrix, expected_feature_count=4,
            expected_feature_list_sha256=feature_sha256, device="cuda",
        )
        differences = parity.evidence.get("maximum_absolute_difference", {})
        maximum = max((float(value) for value in differences.values()), default=math.inf)
        active = core._active_reloaded_booster_cuda_probe(parity.reloaded_booster, 4)
        payload = {
            "schema": "amendment08_cuda_producer_consumer_probe/v1",
            "status": "PASS", "authorization": A08_AUTHORIZATION,
            "prompt_sha256": A08_PROMPT_SHA256, "target_run_id": A08_TARGET_RUN_ID,
            "recorded_at": _now(), "xgboost_version": xgb.__version__,
            "scikit_learn_version": sklearn.__version__, "use_cuda": True,
            "gpu_identity": identity, "gpu_name": gpu_name,
            "producer_consumer_status": "PASS",
            "fit_save_reload_predict_status": "PASS",
            "four_way_bit_exact_max_abs_diff": maximum,
            "model_bytes_unchanged": before == sha256_file(model_path),
            "cpu_fallback_detected": False, "fallback_warnings": fallback,
            "active_reloaded_cuda_probe": active,
        }
        publish_strict_json_no_clobber(path, payload)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return validate_amendment08_cuda_producer_consumer_probe(path)


def _amendment08_pre_event7_absent_relatives() -> tuple[str, ...]:
    global_relatives = (
        "provenance/full_models_frozen_manifest.json",
        "provenance/external_audit_exclusion_and_opening.json",
        "provenance/reporting_completion_manifest.json",
        "provenance/metric_recomputation.json",
        "statistics/paired_card_bootstrap.json",
        "OUTPUT_MANIFEST_FINAL.tsv",
    )
    per_variant = tuple(
        relative
        for variant in TRAINED_VARIANTS
        for relative in (
            f"models/{variant}/model.ubj",
            f"models/{variant}/training_completion_manifest.json",
            f"models/{variant}/external_scoring_manifest.json",
            f"predictions/{variant}_validation.csv.gz",
            f"predictions/{variant}_external_audit.csv.gz",
            f"metrics/{variant}.json",
            f"metrics/{variant}_per_card.csv",
        )
    )
    return (*global_relatives, *per_variant)


def _validate_amendment08_base_history(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    base_paths = {
        "identity": (
            run_dir / "config/run_identity.lock.json", A07_BASE_IDENTITY_SHA256,
        ),
        "manifest": (
            run_dir / "provenance/scientific_execution_code_manifest.tsv",
            A07_BASE_SCIENTIFIC_MANIFEST_SHA256,
        ),
        "lineage": (
            run_dir / "provenance/amendment06_code_lineage.json",
            A07_BASE_CODE_LINEAGE_SHA256,
        ),
        "intent": (_run_intent_path(), A07_A06_INTENT_SHA256),
        "receipt": (_run_receipt_path(), A07_A06_RECEIPT_SHA256),
        "full_log": (
            A06_RUNTIME / "live/full_run.log", A07_ORIGINAL_FULL_LOG_SHA256,
        ),
        "cuda_probe": (
            run_dir / "provenance/cuda_active_fit_probe.json",
            A07_IMMUTABLE_CUDA_PROBE_SHA256,
        ),
    }
    if any(not _safe_regular(path, digest) for path, digest in base_paths.values()):
        raise Amendment06IntegrityError("Amendment 08 immutable base history drifted")
    identity = strict_full_load_file(base_paths["identity"][0])
    if not (
        identity.get("run_id") == run_dir.name == A08_TARGET_RUN_ID
        and identity.get("run_kind") == RUN_KIND
        and identity.get("state") == "FULL_IN_PROGRESS"
        and identity.get("scientific_execution_code_manifest_sha256")
        == A07_BASE_SCIENTIFIC_MANIFEST_SHA256
    ):
        raise Amendment06IntegrityError("Amendment 08 base identity differs")
    base_rows = _parse_code_manifest_bytes(
        base_paths["manifest"][0].read_bytes(), label="Amendment 08 base manifest",
    )
    if [str(row["relative_path"]) for row in base_rows] != list(
        (*SCIENTIFIC_CODE_RELATIVES, *SCIENTIFIC_TEST_RELATIVES)
    ):
        raise Amendment06IntegrityError("Amendment 08 base manifest path order differs")
    original = {
        "code/amendment06_full.py": A07_OLD_FULL_SHA256,
        "tests/test_amendment06.py": A07_OLD_TEST_SHA256,
    }
    for relative, digest in original.items():
        row = next(item for item in base_rows if item["relative_path"] == relative)
        snapshot_root = "tests_snapshot" if relative.startswith("tests/") else "code_snapshot"
        snapshot = run_dir / "provenance" / snapshot_root / Path(relative).name
        if row["sha256"] != digest or not _safe_regular(snapshot, digest):
            raise Amendment06IntegrityError(
                f"Amendment 08 immutable base snapshot differs: {relative}"
            )
    return {
        "status": "PASS", "identity_sha256": A07_BASE_IDENTITY_SHA256,
        "base_scientific_manifest_sha256": A07_BASE_SCIENTIFIC_MANIFEST_SHA256,
        "base_code_lineage_sha256": A07_BASE_CODE_LINEAGE_SHA256,
    }


def validate_amendment08_precorrection(
    run_dir: Path | None = None,
) -> dict[str, Any]:
    if run_dir is None:
        run_dir = A08_TARGET_RUN
    run_dir = Path(run_dir).resolve()
    if run_dir != Path(A08_TARGET_RUN).resolve():
        raise Amendment06IntegrityError("Amendment 08 target Run path differs")
    runs = existing_full_runs()
    if runs != [run_dir]:
        raise Amendment06IntegrityError(
            f"Amendment 08 requires the one preserved Full Run: {runs}"
        )
    base = _validate_amendment08_base_history(run_dir)
    raw = _validate_amendment08_raw_prefix(run_dir, require_exact_boundary=True)
    forensic = _validate_amendment08_forensic_snapshot()
    cache = _validate_preserved_target_cache_files(run_dir)
    if any((run_dir / relative).exists() for relative in A07_OVERLAY_RUN_RELATIVES):
        raise Amendment06IntegrityError("An Amendment 07 overlay was installed unexpectedly")
    if any((run_dir / relative).exists() for relative in A08_OVERLAY_RUN_RELATIVES):
        raise Amendment06IntegrityError("An Amendment 08 overlay already occupies the target")
    absent = list(_amendment08_pre_event7_absent_relatives())
    if any((run_dir / relative).exists() for relative in absent):
        raise Amendment06IntegrityError(
            "A real scientific artifact exists at the pre-event-7 adjudication boundary"
        )
    return {
        "status": "PASS", "authorization": A08_AUTHORIZATION,
        "prompt_sha256": A08_PROMPT_SHA256, "target_run_id": run_dir.name,
        "base_history": base, "raw_journal": raw, "forensic": forensic,
        "cache": cache, "pre_event7_absent_scientific_artifacts": absent,
        "a07_overlay_installed": False, "new_full_run_ids_created": 0,
        "external_audit_tabular_parse_calls": 0,
    }


def _amendment08_external_authorization_payload(
    *, recorded_at: str, forensic: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "amendment08_external_authorization/v1", "status": "PASS",
        "authorization": A08_AUTHORIZATION, "prompt_sha256": A08_PROMPT_SHA256,
        "target_run_id": A08_TARGET_RUN_ID, "target_run_path": str(A08_TARGET_RUN),
        "recorded_at": recorded_at, "new_preflight_runs_authorized": 0,
        "new_smoke_runs_authorized": 0, "new_full_run_ids_authorized": 0,
        "same_run_resume_authorized": True, "overlay_installations_authorized": 1,
        "classified_test_contamination_events": 2,
        "frozen_population_reuse_required": True,
        "new_resampling_realizations_authorized": 0,
        "stability_or_multiseed_authorized": False,
        "threshold_tuning_authorized": False,
        "external_audit_use_for_fitting": False,
        "base_run_identity_sha256": A07_BASE_IDENTITY_SHA256,
        "base_scientific_manifest_sha256": A07_BASE_SCIENTIFIC_MANIFEST_SHA256,
        "forensic_snapshot_sha256": forensic["forensic_zip_sha256"],
        "forensic_verification_sha256": forensic["forensic_verification_sha256"],
        "external_a08_review_zip_sha256": A08_EXTERNAL_REVIEW_ZIP_SHA256,
        "a07_overlay_installed": False,
        "a07_continuation_authorization_consumed": False,
    }


def _amendment08_failed_attempt_payload(
    *, recorded_at: str,
) -> dict[str, Any]:
    absent = list(_amendment08_pre_event7_absent_relatives())
    return {
        "schema": "amendment08_failed_attempt_forensics/v1", "status": "PASS",
        "authorization": A08_AUTHORIZATION, "prompt_sha256": A08_PROMPT_SHA256,
        "target_run_id": A08_TARGET_RUN_ID, "recorded_at": recorded_at,
        "failed_attempt_id": A07_FAILED_ATTEMPT_ID,
        "training_start_event_index": 2, "scientific_fit_event_index": 3,
        "failed_event_index": 4,
        "failed_event_raw_sha256": A08_RAW_JOURNAL_SHA256[4],
        "failed_event_canonical_sha256": A08_CANONICAL_JOURNAL_SHA256[4],
        "error_type": "Amendment06IntegrityError",
        "error": "GPU identity has no name", "published_artifact_count": 0,
        "classifier_saved": False, "failed_in_memory_classifier_reused": False,
        "pre_event7_absent_scientific_artifacts": absent,
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        "frozen_population_sha256": A07_FROZEN_POPULATION_SHA256,
        "failed_attempt_quarantine_intent_sha256": (
            A07_FAILED_ATTEMPT_QUARANTINE_INTENT_SHA256
        ),
        "failed_attempt_quarantine_receipt_sha256": (
            A07_FAILED_ATTEMPT_QUARANTINE_RECEIPT_SHA256
        ),
    }


def _amendment08_adjudication_payload(
    *, recorded_at: str, raw: Mapping[str, Any],
) -> dict[str, Any]:
    records = [dict(item) for item in raw["classified_records"]]
    return {
        "schema": "amendment08_journal_contamination_adjudication/v1",
        "status": "PASS", "authorization": A08_AUTHORIZATION,
        "prompt_sha256": A08_PROMPT_SHA256, "target_run_id": A08_TARGET_RUN_ID,
        "recorded_at": recorded_at,
        "contamination_classification": A08_CONTAMINATION_CLASSIFICATION,
        "excluded_physical_event_indices": [5, 6], "semantic_exclusion_count": 2,
        "classified_records": records,
        "pre_a08_raw_journal_event_count": 7,
        "pre_a08_raw_journal_head_sha256": A08_CANONICAL_JOURNAL_SHA256[6],
        "raw_events_0_to_6_concat_sha256": A08_RAW_EVENTS_0_TO_6_CONCAT_SHA256,
        "canonical_json_lines_events_0_to_6_sha256": (
            A08_CANONICAL_EVENTS_0_TO_6_LINES_SHA256
        ),
        "test_fixture_bytes_sha256": A08_TEST_FIXTURE_SHA256,
        "contaminating_test": A08_CONTAMINATING_TEST,
        "contaminating_function_pre_a08_sha256": A08_CONTAMINATING_FUNCTION_SHA256,
        "pre_event7_absent_scientific_artifacts": list(
            _amendment08_pre_event7_absent_relatives()
        ),
        "pre_event7_scientific_artifact_absence_verified": True,
        "pre_a08_source_hashes": dict(A08_PRE_SOURCE_HASHES),
        "external_a08_review_zip_sha256": A08_EXTERNAL_REVIEW_ZIP_SHA256,
        "amendment07_prompt_sha256": A07_PROMPT_SHA256,
        "amendment07_forensic_zip_sha256": A07_FORENSIC_ZIP_SHA256,
        "amendment07_forensic_verification_sha256": A07_FORENSIC_VERIFICATION_SHA256,
        "amendment07_reconstruction_sha256": A07_FORENSIC_RECONSTRUCTION_SHA256,
        "a07_overlay_installed": False,
        "classification_is_exact_not_generic": True,
    }


def _amendment08_corrigendum_payload(
    *, recorded_at: str, effective_manifest_sha256: str,
    adjudication_sha256: str, source: Mapping[str, Any],
    package: Mapping[str, Any], probe_sha256: str,
) -> dict[str, Any]:
    final_hashes = source["final_source_hashes"]
    return {
        "schema": "amendment08_code_corrigendum/v1", "status": "PASS",
        "authorization": A08_AUTHORIZATION, "prompt_sha256": A08_PROMPT_SHA256,
        "target_run_id": A08_TARGET_RUN_ID, "recorded_at": recorded_at,
        "base_run_identity_sha256": A07_BASE_IDENTITY_SHA256,
        "base_scientific_execution_code_manifest_sha256": (
            A07_BASE_SCIENTIFIC_MANIFEST_SHA256
        ),
        "base_amendment06_code_lineage_sha256": A07_BASE_CODE_LINEAGE_SHA256,
        "effective_scientific_execution_code_manifest_sha256": effective_manifest_sha256,
        "final_amendment06_full_py_sha256": final_hashes["code/amendment06_full.py"],
        "final_test_amendment06_py_sha256": final_hashes["tests/test_amendment06.py"],
        "gpu_identity_correction": {
            "nested_devices_gpu_name_sole_source": True,
            "canonical_unpadded_name_required": True,
            "optional_duplicate_names_exact": True,
            "visible_device_count_exact_integer_one": True,
            "parse_once_before_attempt_start_and_fit": True,
        },
        "journal_path_guard": {
            "production_results_only": True,
            "pytest_live_target_write_blocked_before_side_effect": True,
            "exact_target_fixture_requires_isolated_pair": True,
        },
        "semantic_journal_view_correction": {
            "raw_reader_writer_unchanged": True,
            "excluded_physical_event_indices": [5, 6],
            "original_physical_indices_retained": True,
            "receipt_gated": True,
        },
        "amendment08_journal_contamination_adjudication_sha256": adjudication_sha256,
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        "frozen_population_sha256": A07_FROZEN_POPULATION_SHA256,
        "portable_imputer_sha256": A07_PORTABLE_IMPUTER_SHA256,
        "failed_attempt_id": A07_FAILED_ATTEMPT_ID,
        "failed_attempt_publications": 0,
        "pre_event7_raw_journal_head_sha256": A08_CANONICAL_JOURNAL_SHA256[6],
        "forensic_snapshot_sha256": A08_FORENSIC_ZIP_SHA256,
        "forensic_verification_sha256": A08_FORENSIC_VERIFICATION_SHA256,
        "cuda_producer_consumer_probe_sha256": probe_sha256,
        "source_freeze_receipt_sha256": source["source_freeze_receipt_sha256"],
        "current_a06_authorized_change_ledger_sha256": source[
            "current_a06_authorized_change_ledger_sha256"
        ],
        "a08_source_ledger_event_index": source["a08_source_ledger_event_index"],
        "a08_source_ledger_event_canonical_sha256": source[
            "a08_source_ledger_event_canonical_sha256"
        ],
        "current_a06_final_prerun_evidence_sha256": source[
            "current_a06_final_prerun_evidence_sha256"
        ],
        "a08_prelive_package_lineage_sha256": package["lineage_sha256"],
        "a08_prelive_package_receipt_sha256": package["receipt_sha256"],
        "a08_prelive_package_code_sha256": package["final_code_sha256"],
        "a08_prelive_package_test_sha256": package["final_test_sha256"],
    }


def _amendment08_event7_payload(
    *, recorded_at: str, transaction_id: str, effective_manifest_sha256: str,
    corrigendum_sha256: str, adjudication_sha256: str,
    source: Mapping[str, Any], package: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "event": "AMENDMENT08_CORRIGENDUM_INSTALLED", "event_index": 7,
        "run_id": A08_TARGET_RUN_ID,
        "previous_event_sha256": A08_CANONICAL_JOURNAL_SHA256[6],
        "recorded_at": recorded_at, "transaction_id": transaction_id,
        "status": "PASS", "authorization": A08_AUTHORIZATION,
        "prompt_sha256": A08_PROMPT_SHA256,
        "base_scientific_execution_code_manifest_sha256": (
            A07_BASE_SCIENTIFIC_MANIFEST_SHA256
        ),
        "effective_scientific_execution_code_manifest_sha256": effective_manifest_sha256,
        "amendment08_code_corrigendum_sha256": corrigendum_sha256,
        "amendment08_journal_contamination_adjudication_sha256": adjudication_sha256,
        "source_freeze_receipt_sha256": source["source_freeze_receipt_sha256"],
        "current_a06_authorized_change_ledger_sha256": source[
            "current_a06_authorized_change_ledger_sha256"
        ],
        "a08_source_ledger_event_index": source["a08_source_ledger_event_index"],
        "a08_source_ledger_event_canonical_sha256": source[
            "a08_source_ledger_event_canonical_sha256"
        ],
        "current_a06_final_prerun_evidence_sha256": source[
            "current_a06_final_prerun_evidence_sha256"
        ],
        "a08_prelive_package_lineage_sha256": package["lineage_sha256"],
        "a08_prelive_package_receipt_sha256": package["receipt_sha256"],
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        "contamination_event_indices": [5, 6],
        "contamination_event_raw_sha256": list(A08_RAW_JOURNAL_SHA256[5:7]),
        "contamination_event_canonical_sha256": list(
            A08_CANONICAL_JOURNAL_SHA256[5:7]
        ),
        "a07_overlay_installed": False,
        "a07_continuation_authorization_consumed": False,
    }


def _amendment08_precheck_payload(
    *, recorded_at: str, transaction_id: str, event7: Mapping[str, Any],
    effective_manifest_sha256: str, corrigendum_sha256: str,
    adjudication_sha256: str, source: Mapping[str, Any],
    package: Mapping[str, Any],
) -> dict[str, Any]:
    raw_events = (*_amendment08_expected_raw_prefix_events(), dict(event7))
    raw_lines = b"".join(
        strict_full_canonical_json_bytes(event) + b"\n" for event in raw_events
    )
    semantic_events = (*raw_events[:5], raw_events[7])
    semantic_lines = b"".join(
        strict_full_canonical_json_bytes(event) + b"\n" for event in semantic_events
    )
    event7_bytes = strict_full_json_bytes(event7)
    event7_head = sha256_bytes(strict_full_canonical_json_bytes(event7))
    return {
        "schema": "amendment08_resume_precheck/v1", "status": "PASS",
        "authorization": A08_AUTHORIZATION, "prompt_sha256": A08_PROMPT_SHA256,
        "target_run_id": A08_TARGET_RUN_ID, "recorded_at": recorded_at,
        "transaction_id": transaction_id,
        "corrigendum_journal_event_index": 7,
        "corrigendum_journal_event_raw_sha256": sha256_bytes(event7_bytes),
        "corrigendum_journal_head_sha256": event7_head,
        "raw_journal_event_count": 8, "raw_journal_head_sha256": event7_head,
        "raw_journal_canonical_json_lines_sha256": sha256_bytes(raw_lines),
        "semantic_journal_event_count": 6,
        "semantic_journal_canonical_json_lines_sha256": sha256_bytes(semantic_lines),
        "classified_test_contamination_event_count": 2,
        "classified_test_contamination_event_indices": [5, 6],
        "classified_test_contamination_raw_sha256": list(A08_RAW_JOURNAL_SHA256[5:7]),
        "classified_test_contamination_canonical_sha256": list(
            A08_CANONICAL_JOURNAL_SHA256[5:7]
        ),
        "base_scientific_execution_code_manifest_sha256": (
            A07_BASE_SCIENTIFIC_MANIFEST_SHA256
        ),
        "effective_scientific_execution_code_manifest_sha256": effective_manifest_sha256,
        "amendment08_code_corrigendum_sha256": corrigendum_sha256,
        "amendment08_journal_contamination_adjudication_sha256": adjudication_sha256,
        "source_freeze_receipt_sha256": source["source_freeze_receipt_sha256"],
        "current_a06_authorized_change_ledger_sha256": source[
            "current_a06_authorized_change_ledger_sha256"
        ],
        "current_a06_final_prerun_evidence_sha256": source[
            "current_a06_final_prerun_evidence_sha256"
        ],
        "a08_prelive_package_lineage_sha256": package["lineage_sha256"],
        "a08_prelive_package_receipt_sha256": package["receipt_sha256"],
        "a08_prelive_package_code_sha256": package["final_code_sha256"],
        "a08_prelive_package_test_sha256": package["final_test_sha256"],
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        "frozen_population_sha256": A07_FROZEN_POPULATION_SHA256,
        "cache_generation_calls": 0, "cache_rematerialization_calls": 0,
        "external_audit_tabular_parse_calls": 0, "new_full_run_ids_created": 0,
        "a07_overlay_installed": False,
        "projected_semantic_view_status": "PASS",
    }


def _amendment08_overlay_record(relative: str, data: bytes) -> dict[str, Any]:
    return {
        "relative_path": relative, "size_bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def _build_amendment08_overlay_material(
    *, run_dir: Path, probe_path: Path, generated_at: str,
    event7_recorded_at: str, transaction_id: str,
    source: Mapping[str, Any], package: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    raw = _validate_amendment08_raw_prefix(run_dir, require_exact_boundary=False)
    forensic = _validate_amendment08_forensic_snapshot()
    probe = validate_amendment08_cuda_producer_consumer_probe(probe_path)
    effective_bytes = _effective_amendment08_manifest_bytes(run_dir)
    effective_sha256 = sha256_bytes(effective_bytes)
    adjudication = _amendment08_adjudication_payload(
        recorded_at=generated_at, raw=raw,
    )
    adjudication_bytes = strict_full_json_bytes(adjudication)
    adjudication_sha256 = sha256_bytes(adjudication_bytes)
    probe_bytes = Path(probe_path).read_bytes()
    corrigendum = _amendment08_corrigendum_payload(
        recorded_at=generated_at, effective_manifest_sha256=effective_sha256,
        adjudication_sha256=adjudication_sha256, source=source, package=package,
        probe_sha256=sha256_bytes(probe_bytes),
    )
    corrigendum_bytes = strict_full_json_bytes(corrigendum)
    corrigendum_sha256 = sha256_bytes(corrigendum_bytes)
    event7 = _amendment08_event7_payload(
        recorded_at=event7_recorded_at, transaction_id=transaction_id,
        effective_manifest_sha256=effective_sha256,
        corrigendum_sha256=corrigendum_sha256,
        adjudication_sha256=adjudication_sha256,
        source=source, package=package,
    )
    precheck = _amendment08_precheck_payload(
        recorded_at=generated_at, transaction_id=transaction_id, event7=event7,
        effective_manifest_sha256=effective_sha256,
        corrigendum_sha256=corrigendum_sha256,
        adjudication_sha256=adjudication_sha256,
        source=source, package=package,
    )
    material = {
        "provenance/amendment08_external_authorization.json": strict_full_json_bytes(
            _amendment08_external_authorization_payload(
                recorded_at=generated_at, forensic=forensic,
            )
        ),
        "provenance/amendment08_prompt_snapshot.txt": A08_PROMPT_PATH.read_bytes(),
        "provenance/amendment08_code_corrigendum.json": corrigendum_bytes,
        "provenance/scientific_execution_code_manifest_effective_amendment08.tsv": (
            effective_bytes
        ),
        "provenance/amendment08_failed_attempt_forensics.json": strict_full_json_bytes(
            _amendment08_failed_attempt_payload(recorded_at=generated_at)
        ),
        "provenance/amendment08_journal_contamination_adjudication.json": (
            adjudication_bytes
        ),
        "provenance/amendment08_resume_precheck.json": strict_full_json_bytes(precheck),
        "provenance/amendment08_cuda_producer_consumer_probe.json": probe_bytes,
        "provenance/code_snapshot_amendment08/amendment06_full.py": (
            STUDY_ROOT / "code/amendment06_full.py"
        ).read_bytes(),
        "provenance/tests_snapshot_amendment08/test_amendment06.py": (
            STUDY_ROOT / "tests/test_amendment06.py"
        ).read_bytes(),
    }
    expected_order = [
        relative for relative in A08_OVERLAY_RUN_RELATIVES
        if relative != "provenance/amendment08_overlay_install_receipt.json"
    ]
    if list(material) != expected_order:
        raise Amendment06IntegrityError("Amendment 08 overlay material order differs")
    final_hashes = source["final_source_hashes"]
    if not (
        sha256_bytes(material["provenance/code_snapshot_amendment08/amendment06_full.py"])
        == final_hashes["code/amendment06_full.py"]
        and sha256_bytes(
            material["provenance/tests_snapshot_amendment08/test_amendment06.py"]
        ) == final_hashes["tests/test_amendment06.py"]
        and probe["status"] == "PASS"
    ):
        raise Amendment06IntegrityError("Amendment 08 overlay source/probe binding differs")
    return material, {
        "effective_manifest_sha256": effective_sha256,
        "corrigendum_sha256": corrigendum_sha256,
        "adjudication_sha256": adjudication_sha256,
        "event7": event7, "precheck": precheck,
    }


def _amendment08_install_intent_payload(
    *, generated_at: str, event7_recorded_at: str, transaction_id: str,
    material: Mapping[str, bytes], bindings: Mapping[str, Any],
    source: Mapping[str, Any], package: Mapping[str, Any],
) -> dict[str, Any]:
    event7 = bindings["event7"]
    records = [_amendment08_overlay_record(relative, material[relative]) for relative in material]
    return {
        "schema": "amendment08_overlay_install_intent/v1", "status": "INTENT",
        "authorization": A08_AUTHORIZATION, "prompt_sha256": A08_PROMPT_SHA256,
        "target_run_id": A08_TARGET_RUN_ID, "transaction_id": transaction_id,
        "generated_at": generated_at, "event7_recorded_at": event7_recorded_at,
        "overlay_installations_authorized": 1, "declared_member_count": 10,
        "declared_members": records,
        "receipt_relative_path": "provenance/amendment08_overlay_install_receipt.json",
        "projected_corrigendum_journal_event": event7,
        "projected_corrigendum_journal_event_raw_sha256": sha256_bytes(
            strict_full_json_bytes(event7)
        ),
        "projected_corrigendum_journal_head_sha256": sha256_bytes(
            strict_full_canonical_json_bytes(event7)
        ),
        "base_scientific_execution_code_manifest_sha256": (
            A07_BASE_SCIENTIFIC_MANIFEST_SHA256
        ),
        "effective_scientific_execution_code_manifest_sha256": bindings[
            "effective_manifest_sha256"
        ],
        "amendment08_code_corrigendum_sha256": bindings["corrigendum_sha256"],
        "amendment08_journal_contamination_adjudication_sha256": bindings[
            "adjudication_sha256"
        ],
        "source_freeze_receipt_sha256": source["source_freeze_receipt_sha256"],
        "current_a06_authorized_change_ledger_sha256": source[
            "current_a06_authorized_change_ledger_sha256"
        ],
        "current_a06_final_prerun_evidence_sha256": source[
            "current_a06_final_prerun_evidence_sha256"
        ],
        "a08_prelive_package_lineage_sha256": package["lineage_sha256"],
        "a08_prelive_package_receipt_sha256": package["receipt_sha256"],
        "a08_prelive_package_code_sha256": package["final_code_sha256"],
        "a08_prelive_package_test_sha256": package["final_test_sha256"],
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        "frozen_population_sha256": A07_FROZEN_POPULATION_SHA256,
    }


def _amendment08_install_receipt_payload(
    *, intent_sha256: str, installed_at: str, material: Mapping[str, bytes],
    bindings: Mapping[str, Any], source: Mapping[str, Any],
    package: Mapping[str, Any],
) -> dict[str, Any]:
    event7 = bindings["event7"]
    precheck_relative = "provenance/amendment08_resume_precheck.json"
    return {
        "schema": "amendment08_overlay_install_receipt/v1", "status": "PASS",
        "authorization": A08_AUTHORIZATION, "prompt_sha256": A08_PROMPT_SHA256,
        "target_run_id": A08_TARGET_RUN_ID,
        "transaction_id": event7["transaction_id"],
        "installation_intent_sha256": intent_sha256,
        "overlay_installations": 1, "member_count": 10,
        "members": [
            _amendment08_overlay_record(relative, material[relative])
            for relative in material
        ],
        "corrigendum_journal_event_index": 7,
        "corrigendum_journal_event_raw_sha256": sha256_bytes(
            strict_full_json_bytes(event7)
        ),
        "corrigendum_journal_head_sha256": sha256_bytes(
            strict_full_canonical_json_bytes(event7)
        ),
        "precheck_sha256": sha256_bytes(material[precheck_relative]),
        "base_scientific_execution_code_manifest_sha256": (
            A07_BASE_SCIENTIFIC_MANIFEST_SHA256
        ),
        "effective_scientific_execution_code_manifest_sha256": bindings[
            "effective_manifest_sha256"
        ],
        "amendment08_code_corrigendum_sha256": bindings["corrigendum_sha256"],
        "amendment08_journal_contamination_adjudication_sha256": bindings[
            "adjudication_sha256"
        ],
        "source_freeze_receipt_sha256": source["source_freeze_receipt_sha256"],
        "current_a06_authorized_change_ledger_sha256": source[
            "current_a06_authorized_change_ledger_sha256"
        ],
        "current_a06_final_prerun_evidence_sha256": source[
            "current_a06_final_prerun_evidence_sha256"
        ],
        "a08_prelive_package_lineage_sha256": package["lineage_sha256"],
        "a08_prelive_package_receipt_sha256": package["receipt_sha256"],
        "a08_prelive_package_code_sha256": package["final_code_sha256"],
        "a08_prelive_package_test_sha256": package["final_test_sha256"],
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        "frozen_population_sha256": A07_FROZEN_POPULATION_SHA256,
        "installed_at": installed_at,
    }


def _amendment08_overlay_runtime_root() -> Path:
    return Path(A08_INSTALL_INTENT).resolve().parent.parent


def _amendment08_overlay_mismatch_root() -> Path:
    return _amendment08_overlay_runtime_root() / "transactions/overlay_mismatch"


def _amendment08_overlay_mismatch_paths(
    *, transaction_id: str, source_scope: str, relative: str,
) -> tuple[Path, Path, Path]:
    token = sha256_bytes(f"{source_scope}\0{relative}".encode("utf-8"))
    transaction_root = _amendment08_overlay_mismatch_root() / transaction_id
    intent = transaction_root / f"{token}.intent.json"
    receipt = transaction_root / f"{token}.receipt.json"
    destination = (
        _amendment08_overlay_runtime_root()
        / "forensic/quarantine/overlay_install_mismatch"
        / transaction_id / source_scope.lower() / relative
    )
    return intent, receipt, destination


def _lexical_absolute_path(path: Path) -> Path:
    """Return an absolute path without dereferencing its final symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def _amendment08_overlay_mismatch_payload(
    *, path: Path, run_dir: Path, relative: str, transaction_id: str,
    source_scope: str,
) -> dict[str, Any]:
    relative = _safe_run_relative(relative)
    if source_scope not in {"RUN", "STAGING"}:
        raise Amendment06IntegrityError("Amendment 08 mismatch source scope differs")
    path = _lexical_absolute_path(Path(path))
    run_dir = Path(run_dir).resolve()
    expected_source = _lexical_absolute_path(
        run_dir / relative if source_scope == "RUN" else
        _amendment08_overlay_runtime_root()
        / "install_staging" / transaction_id / relative
    )
    if path != expected_source:
        raise Amendment06IntegrityError("Amendment 08 mismatch source path differs")
    mode = os.lstat(path).st_mode
    regular = stat.S_ISREG(mode)
    intent_path, _, destination = _amendment08_overlay_mismatch_paths(
        transaction_id=transaction_id, source_scope=source_scope, relative=relative,
    )
    del intent_path
    return {
        "schema": "amendment08_overlay_mismatch_quarantine_intent/v1",
        "status": "INTENT", "authorization": A08_AUTHORIZATION,
        "prompt_sha256": A08_PROMPT_SHA256, "target_run_id": run_dir.name,
        "target_run_path": str(run_dir), "transaction_id": transaction_id,
        "source_scope": source_scope, "relative_path": relative,
        "source_path": str(path), "quarantine_path": str(destination),
        "observed_mode": int(mode),
        "observed_size_bytes": int(path.stat().st_size) if regular else None,
        "observed_sha256": sha256_file(path) if regular else None,
    }


def _reconcile_amendment08_overlay_mismatch_intent(intent_path: Path) -> None:
    payload = _require_exact_keys(
        strict_full_load_file(intent_path),
        {
            "schema", "status", "authorization", "prompt_sha256", "target_run_id",
            "target_run_path", "transaction_id", "source_scope", "relative_path",
            "source_path", "quarantine_path", "observed_mode",
            "observed_size_bytes", "observed_sha256",
        },
        label="Amendment 08 overlay mismatch quarantine intent",
    )
    transaction_id = str(payload["transaction_id"])
    source_scope = str(payload["source_scope"])
    relative = _safe_run_relative(str(payload["relative_path"]))
    run_dir = Path(str(payload["target_run_path"])).resolve()
    expected_intent, receipt_path, destination = _amendment08_overlay_mismatch_paths(
        transaction_id=transaction_id, source_scope=source_scope, relative=relative,
    )
    expected_source = _lexical_absolute_path(
        run_dir / relative if source_scope == "RUN" else
        _amendment08_overlay_runtime_root()
        / "install_staging" / transaction_id / relative
    )
    if not (
        intent_path.resolve() == expected_intent.resolve()
        and payload["schema"] == "amendment08_overlay_mismatch_quarantine_intent/v1"
        and payload["status"] == "INTENT"
        and payload["authorization"] == A08_AUTHORIZATION
        and payload["prompt_sha256"] == A08_PROMPT_SHA256
        and payload["target_run_id"] == run_dir.name == A08_TARGET_RUN_ID
        and re.fullmatch(r"[0-9a-f]{32}", transaction_id)
        and source_scope in {"RUN", "STAGING"}
        and _lexical_absolute_path(Path(str(payload["source_path"])))
        == expected_source
        and _lexical_absolute_path(Path(str(payload["quarantine_path"])))
        == _lexical_absolute_path(destination)
        and isinstance(payload["observed_mode"], int)
        and not isinstance(payload["observed_mode"], bool)
    ):
        raise Amendment06IntegrityError("Amendment 08 overlay mismatch intent differs")
    regular = stat.S_ISREG(int(payload["observed_mode"]))
    if regular:
        if not (
            isinstance(payload["observed_size_bytes"], int)
            and not isinstance(payload["observed_size_bytes"], bool)
            and int(payload["observed_size_bytes"]) >= 0
            and re.fullmatch(r"[0-9a-f]{64}", str(payload["observed_sha256"]))
        ):
            raise Amendment06IntegrityError(
                "Amendment 08 regular mismatch evidence differs"
            )
    elif payload["observed_size_bytes"] is not None or payload["observed_sha256"] is not None:
        raise Amendment06IntegrityError("Amendment 08 special-file mismatch evidence differs")

    source_present = expected_source.exists() or expected_source.is_symlink()
    destination_present = destination.exists() or destination.is_symlink()
    if source_present and destination_present:
        raise Amendment06IntegrityError("Amendment 08 mismatch exists in two locations")
    if source_present:
        if os.lstat(expected_source).st_mode != int(payload["observed_mode"]):
            raise Amendment06IntegrityError("Amendment 08 mismatch mode changed before quarantine")
        if regular and not (
            _safe_regular(expected_source, str(payload["observed_sha256"]))
            and expected_source.stat().st_size == payload["observed_size_bytes"]
        ):
            raise Amendment06IntegrityError("Amendment 08 mismatch bytes changed before quarantine")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if os.lstat(expected_source).st_dev != destination.parent.stat().st_dev:
            raise Amendment06IntegrityError("Amendment 08 mismatch quarantine crosses filesystems")
        _rename_directory_no_clobber(expected_source, destination)
        _fsync_parent(expected_source)
        destination_present = True
    if not destination_present or os.lstat(destination).st_mode != int(payload["observed_mode"]):
        raise Amendment06IntegrityError("Amendment 08 mismatch quarantine is incomplete")
    if regular and not (
        _safe_regular(destination, str(payload["observed_sha256"]))
        and destination.stat().st_size == payload["observed_size_bytes"]
    ):
        raise Amendment06IntegrityError("Amendment 08 quarantined mismatch bytes differ")
    receipt = {
        **dict(payload),
        "schema": "amendment08_overlay_mismatch_quarantine_receipt/v1",
        "status": "PASS", "intent_sha256": sha256_file(intent_path),
    }
    if receipt_path.exists():
        if strict_full_load_file(receipt_path) != receipt:
            raise Amendment06IntegrityError("Amendment 08 mismatch receipt drifted")
    else:
        publish_strict_json_no_clobber(receipt_path, receipt)


def _quarantine_amendment08_overlay_mismatch(
    *, path: Path, run_dir: Path, relative: str, transaction_id: str,
    source_scope: str,
) -> None:
    payload = _amendment08_overlay_mismatch_payload(
        path=path, run_dir=run_dir, relative=relative,
        transaction_id=transaction_id, source_scope=source_scope,
    )
    intent_path, _, _ = _amendment08_overlay_mismatch_paths(
        transaction_id=transaction_id, source_scope=source_scope, relative=relative,
    )
    if intent_path.exists():
        if strict_full_load_file(intent_path) != payload:
            raise Amendment06IntegrityError("Amendment 08 mismatch intent drifted")
    else:
        publish_strict_json_no_clobber(intent_path, payload)
    _reconcile_amendment08_overlay_mismatch_intent(intent_path)


def _assert_no_amendment08_overlay_mismatch() -> None:
    root = _amendment08_overlay_mismatch_root()
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise Amendment06IntegrityError("Amendment 08 mismatch transaction root is unsafe")
    intents = sorted(root.rglob("*.intent.json"))
    unexpected = [
        path for path in root.rglob("*")
        if path.is_file() and not (
            path.name.endswith(".intent.json") or path.name.endswith(".receipt.json")
        )
    ]
    if unexpected:
        raise Amendment06IntegrityError("Amendment 08 mismatch transaction has extra files")
    for intent_path in intents:
        _reconcile_amendment08_overlay_mismatch_intent(intent_path)
    if intents or any(root.iterdir()):
        raise Amendment06IntegrityError(
            "A quarantined Amendment 08 overlay mismatch permanently blocks Resume"
        )


def _publish_or_validate_amendment08_member(
    *, run_dir: Path, relative: str, data: bytes, transaction_id: str,
    source_scope: str = "RUN",
) -> None:
    relative = _safe_run_relative(relative)
    path = (
        Path(run_dir).resolve() / relative if source_scope == "RUN" else
        _amendment08_overlay_runtime_root()
        / "install_staging" / transaction_id / relative
    )
    if path.exists() or path.is_symlink():
        if _safe_regular(path, sha256_bytes(data)) and path.stat().st_size == len(data):
            return
        _quarantine_amendment08_overlay_mismatch(
            path=path, run_dir=run_dir, relative=relative,
            transaction_id=transaction_id, source_scope=source_scope,
        )
        raise Amendment06IntegrityError(
            f"Mismatched Amendment 08 {source_scope.lower()} member was quarantined: {relative}"
        )
    publish_bytes_no_clobber(path, data)
    if not (_safe_regular(path, sha256_bytes(data)) and path.stat().st_size == len(data)):
        raise Amendment06IntegrityError(f"Amendment 08 publication verification failed: {relative}")


def _append_or_validate_exact_journal_event(
    run_dir: Path, expected_event: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    expected = dict(expected_event)
    required_base = {
        "event_index", "event", "run_id", "previous_event_sha256", "recorded_at",
    }
    if not required_base <= set(expected):
        raise Amendment06IntegrityError("Exact Journal event lacks base fields")
    index = expected["event_index"]
    if (
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        or expected["run_id"] != run_dir.name
        or not isinstance(expected["event"], str) or not expected["event"]
        or not isinstance(expected["recorded_at"], str) or not expected["recorded_at"]
    ):
        raise Amendment06IntegrityError("Exact Journal event base fields differ")
    journal = _journal_path(run_dir, for_write=True)
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.mkdir(exist_ok=True)
    lock_path = _journal_lock_path(run_dir, for_write=True)
    descriptor = os.open(
        lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600,
    )
    with os.fdopen(descriptor, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        events = _read_journal(run_dir)
        if len(events) > index:
            if events[index] != expected:
                raise Amendment06IntegrityError(
                    f"Existing exact Journal event {index} differs"
                )
        elif len(events) == index:
            previous = (
                None if not events else
                sha256_bytes(strict_full_canonical_json_bytes(events[-1]))
            )
            if expected["previous_event_sha256"] != previous:
                raise Amendment06IntegrityError(
                    f"Exact Journal event {index} previous head differs"
                )
            publish_bytes_no_clobber(
                _journal_event_path(run_dir, index), strict_full_json_bytes(expected),
            )
        else:
            raise Amendment06IntegrityError(
                f"Exact Journal event {index} would leave a physical gap"
            )
        observed = _read_journal(run_dir)
        if len(observed) <= index or observed[index] != expected:
            raise Amendment06IntegrityError(
                f"Exact Journal event {index} publication did not converge"
            )
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return dict(expected)


def _amendment08_install_intent_keys() -> set[str]:
    return {
        "schema", "status", "authorization", "prompt_sha256", "target_run_id",
        "transaction_id", "generated_at", "event7_recorded_at",
        "overlay_installations_authorized", "declared_member_count",
        "declared_members", "receipt_relative_path",
        "projected_corrigendum_journal_event",
        "projected_corrigendum_journal_event_raw_sha256",
        "projected_corrigendum_journal_head_sha256",
        "base_scientific_execution_code_manifest_sha256",
        "effective_scientific_execution_code_manifest_sha256",
        "amendment08_code_corrigendum_sha256",
        "amendment08_journal_contamination_adjudication_sha256",
        "source_freeze_receipt_sha256", "current_a06_authorized_change_ledger_sha256",
        "current_a06_final_prerun_evidence_sha256",
        "a08_prelive_package_lineage_sha256", "a08_prelive_package_receipt_sha256",
        "a08_prelive_package_code_sha256", "a08_prelive_package_test_sha256",
        "cache_manifest_sha256", "frozen_population_sha256",
    }


def _amendment08_probe_for_install_retry(
    *, run_dir: Path, probe_path: Path, intent: Mapping[str, Any] | None,
) -> Path:
    candidates = (
        Path(probe_path).resolve(),
        run_dir / "provenance/amendment08_cuda_producer_consumer_probe.json",
    )
    expected_digest: str | None = None
    if intent is not None and isinstance(intent.get("declared_members"), list):
        records = {
            str(record.get("relative_path")): record
            for record in intent["declared_members"] if isinstance(record, Mapping)
        }
        record = records.get("provenance/amendment08_cuda_producer_consumer_probe.json")
        if isinstance(record, Mapping):
            expected_digest = str(record.get("sha256", ""))
    for candidate in candidates:
        if _safe_regular(candidate) and (
            expected_digest is None or sha256_file(candidate) == expected_digest
        ):
            return candidate
    raise Amendment06IntegrityError("Amendment 08 CUDA probe bytes are unavailable for retry")


def install_amendment08_overlay(
    *, probe_path: Path, run_dir: Path | None = None,
) -> dict[str, Any]:
    if run_dir is None:
        run_dir = A08_TARGET_RUN
    run_dir = Path(run_dir).resolve()
    _assert_no_amendment08_overlay_mismatch()
    receipt_path = run_dir / "provenance/amendment08_overlay_install_receipt.json"
    if receipt_path.exists():
        return validate_amendment08_overlay(run_dir)
    runtime = _amendment08_overlay_runtime_root()
    with amendment06_lock(runtime / "overlay_install_lock"):
        _assert_no_amendment08_overlay_mismatch()
        if receipt_path.exists():
            return validate_amendment08_overlay(run_dir)
        source, package = _amendment08_overlay_admission_bindings(run_dir)
        intent: Mapping[str, Any] | None = None
        if A08_INSTALL_INTENT.exists():
            intent = _require_exact_keys(
                strict_full_load_file(A08_INSTALL_INTENT),
                _amendment08_install_intent_keys(),
                label="Amendment 08 overlay installation intent",
            )
            generated_at = str(intent["generated_at"])
            event7_recorded_at = str(intent["event7_recorded_at"])
            transaction_id = str(intent["transaction_id"])
        else:
            validate_amendment08_precorrection(run_dir)
            generated_at = _now()
            event7_recorded_at = _now()
            transaction_id = uuid.uuid4().hex
        if not (
            re.fullmatch(r"[0-9a-f]{32}", transaction_id)
            and generated_at and event7_recorded_at
        ):
            raise Amendment06IntegrityError("Amendment 08 reserved install identity differs")
        selected_probe = _amendment08_probe_for_install_retry(
            run_dir=run_dir, probe_path=probe_path, intent=intent,
        )
        material, bindings = _build_amendment08_overlay_material(
            run_dir=run_dir, probe_path=selected_probe, generated_at=generated_at,
            event7_recorded_at=event7_recorded_at, transaction_id=transaction_id,
            source=source, package=package,
        )
        expected_intent = _amendment08_install_intent_payload(
            generated_at=generated_at, event7_recorded_at=event7_recorded_at,
            transaction_id=transaction_id, material=material, bindings=bindings,
            source=source, package=package,
        )
        if intent is not None:
            if dict(intent) != expected_intent:
                raise Amendment06IntegrityError("Amendment 08 installation intent drifted")
        else:
            publish_strict_json_no_clobber(A08_INSTALL_INTENT, expected_intent)
        intent_sha256 = sha256_file(A08_INSTALL_INTENT)

        staging = runtime / "install_staging" / transaction_id
        staging.mkdir(parents=True, exist_ok=True)
        if staging.is_symlink() or not staging.is_dir():
            raise Amendment06IntegrityError("Amendment 08 installation staging is unsafe")
        if staging.stat().st_dev != run_dir.stat().st_dev:
            raise Amendment06IntegrityError("Amendment 08 overlay staging crosses filesystems")
        for relative, data in material.items():
            _publish_or_validate_amendment08_member(
                run_dir=run_dir, relative=relative, data=data,
                transaction_id=transaction_id, source_scope="STAGING",
            )

        pre_event = set(material) - {"provenance/amendment08_resume_precheck.json"}
        for relative in sorted(pre_event, key=lambda value: value.encode("utf-8")):
            _publish_or_validate_amendment08_member(
                run_dir=run_dir, relative=relative, data=material[relative],
                transaction_id=transaction_id,
            )
        _validate_preserved_target_cache_files(run_dir)
        observed_event7 = _append_or_validate_exact_journal_event(
            run_dir, bindings["event7"],
        )
        if observed_event7 != bindings["event7"]:
            raise Amendment06IntegrityError("Amendment 08 event 7 differs")
        _publish_or_validate_amendment08_member(
            run_dir=run_dir, relative="provenance/amendment08_resume_precheck.json",
            data=material["provenance/amendment08_resume_precheck.json"],
            transaction_id=transaction_id,
        )
        receipt = _amendment08_install_receipt_payload(
            intent_sha256=intent_sha256, installed_at=generated_at,
            material=material, bindings=bindings, source=source, package=package,
        )
        receipt_bytes = strict_full_json_bytes(receipt)
        receipt_relative = "provenance/amendment08_overlay_install_receipt.json"
        _publish_or_validate_amendment08_member(
            run_dir=run_dir, relative=receipt_relative, data=receipt_bytes,
            transaction_id=transaction_id, source_scope="STAGING",
        )
        _publish_or_validate_amendment08_member(
            run_dir=run_dir, relative=receipt_relative, data=receipt_bytes,
            transaction_id=transaction_id,
        )
    return validate_amendment08_overlay(run_dir)


def validate_amendment08_overlay(
    run_dir: Path | None = None,
) -> dict[str, Any]:
    if run_dir is None:
        run_dir = A08_TARGET_RUN
    run_dir = Path(run_dir).resolve()
    _assert_no_amendment08_overlay_mismatch()
    if run_dir.name != A08_TARGET_RUN_ID:
        raise Amendment06IntegrityError("Amendment 08 overlay target Run differs")
    _validate_amendment08_base_history(run_dir)
    raw = _validate_amendment08_raw_prefix(run_dir, require_exact_boundary=False)
    source, package = _amendment08_overlay_admission_bindings(run_dir)
    intent = _require_exact_keys(
        strict_full_load_file(A08_INSTALL_INTENT),
        _amendment08_install_intent_keys(),
        label="Amendment 08 overlay installation intent",
    )
    if not (
        intent["schema"] == "amendment08_overlay_install_intent/v1"
        and intent["status"] == "INTENT"
        and intent["authorization"] == A08_AUTHORIZATION
        and intent["prompt_sha256"] == A08_PROMPT_SHA256
        and intent["target_run_id"] == run_dir.name
        and intent["overlay_installations_authorized"] == 1
        and intent["declared_member_count"] == 10
        and re.fullmatch(r"[0-9a-f]{32}", str(intent["transaction_id"]))
        and isinstance(intent["generated_at"], str) and intent["generated_at"]
        and isinstance(intent["event7_recorded_at"], str) and intent["event7_recorded_at"]
    ):
        raise Amendment06IntegrityError("Amendment 08 overlay intent semantics differ")
    probe_path = run_dir / "provenance/amendment08_cuda_producer_consumer_probe.json"
    material, bindings = _build_amendment08_overlay_material(
        run_dir=run_dir, probe_path=probe_path,
        generated_at=str(intent["generated_at"]),
        event7_recorded_at=str(intent["event7_recorded_at"]),
        transaction_id=str(intent["transaction_id"]), source=source, package=package,
    )
    expected_intent = _amendment08_install_intent_payload(
        generated_at=str(intent["generated_at"]),
        event7_recorded_at=str(intent["event7_recorded_at"]),
        transaction_id=str(intent["transaction_id"]), material=material,
        bindings=bindings, source=source, package=package,
    )
    if dict(intent) != expected_intent:
        raise Amendment06IntegrityError("Amendment 08 overlay intent/member plan differs")
    for relative, data in material.items():
        path = run_dir / relative
        if not (_safe_regular(path, sha256_bytes(data)) and path.stat().st_size == len(data)):
            raise Amendment06IntegrityError(f"Amendment 08 overlay member drifted: {relative}")
    event7 = bindings["event7"]
    events = raw["events"]
    event7_path = _journal_event_path(run_dir, 7)
    if not (
        len(events) >= 8 and events[7] == event7
        and _safe_regular(event7_path, sha256_bytes(strict_full_json_bytes(event7)))
    ):
        raise Amendment06IntegrityError("Amendment 08 corrigendum Journal event differs")
    receipt_path = run_dir / "provenance/amendment08_overlay_install_receipt.json"
    expected_receipt = _amendment08_install_receipt_payload(
        intent_sha256=sha256_file(A08_INSTALL_INTENT),
        installed_at=str(intent["generated_at"]), material=material,
        bindings=bindings, source=source, package=package,
    )
    receipt = _require_exact_keys(
        strict_full_load_file(receipt_path), set(expected_receipt),
        label="Amendment 08 overlay install receipt",
    )
    if dict(receipt) != expected_receipt:
        raise Amendment06IntegrityError("Amendment 08 overlay install receipt differs")
    code_manifests = _validate_amendment08_effective_manifest(run_dir)
    cache = _validate_preserved_target_cache_files(run_dir)
    if not (
        code_manifests["effective_manifest_sha256"]
        == bindings["effective_manifest_sha256"]
        and sha256_file(
            run_dir / "provenance/amendment08_code_corrigendum.json"
        ) == bindings["corrigendum_sha256"]
        and sha256_file(
            run_dir / "provenance/amendment08_journal_contamination_adjudication.json"
        ) == bindings["adjudication_sha256"]
        and cache["cache_manifest_sha256"] == A07_CACHE_MANIFEST_SHA256
    ):
        raise Amendment06IntegrityError("Amendment 08 overlay cross-binding differs")
    # The receipt is the activation boundary.  Once it exists, also exercise
    # the receipt-gated semantic view before declaring installation complete.
    read_amendment08_journal_views(run_dir)
    return {
        "status": "PASS", "run_id": run_dir.name,
        "authorization": A08_AUTHORIZATION, "prompt_sha256": A08_PROMPT_SHA256,
        "base_scientific_execution_code_manifest_sha256": (
            A07_BASE_SCIENTIFIC_MANIFEST_SHA256
        ),
        "effective_scientific_execution_code_manifest_sha256": bindings[
            "effective_manifest_sha256"
        ],
        "amendment08_code_corrigendum_sha256": bindings["corrigendum_sha256"],
        "amendment08_journal_contamination_adjudication_sha256": bindings[
            "adjudication_sha256"
        ],
        "overlay_install_receipt_sha256": sha256_file(receipt_path),
        "corrigendum_journal_event_index": 7,
        "corrigendum_journal_event_raw_sha256": sha256_file(event7_path),
        "corrigendum_journal_head_sha256": sha256_bytes(
            strict_full_canonical_json_bytes(event7)
        ),
        "source_freeze": source, "package_lineage": package,
        "code_manifests": code_manifests, "cache": cache,
    }


def read_amendment08_journal_views(
    run_dir: Path | None = None,
) -> dict[str, Any]:
    if run_dir is None:
        run_dir = A08_TARGET_RUN
    run_dir = Path(run_dir).resolve()
    _amendment08_frozen_lineage_bindings(run_dir)
    raw_validation = _validate_amendment08_raw_prefix(
        run_dir, require_exact_boundary=False,
    )
    raw_events = [dict(event) for event in raw_validation["events"]]
    adjudication_path = (
        run_dir / "provenance/amendment08_journal_contamination_adjudication.json"
    )
    receipt_path = run_dir / "provenance/amendment08_overlay_install_receipt.json"
    adjudication = strict_full_load_file(adjudication_path)
    receipt = strict_full_load_file(receipt_path)
    expected_prefix = _amendment08_expected_raw_prefix_events()
    classified_records = adjudication.get("classified_records")
    if not (
        adjudication.get("schema")
        == "amendment08_journal_contamination_adjudication/v1"
        and adjudication.get("status") == "PASS"
        and adjudication.get("authorization") == A08_AUTHORIZATION
        and adjudication.get("prompt_sha256") == A08_PROMPT_SHA256
        and adjudication.get("target_run_id") == run_dir.name
        and adjudication.get("contamination_classification")
        == A08_CONTAMINATION_CLASSIFICATION
        and adjudication.get("excluded_physical_event_indices") == [5, 6]
        and adjudication.get("semantic_exclusion_count") == 2
        and isinstance(classified_records, list) and len(classified_records) == 2
        and receipt.get("schema") == "amendment08_overlay_install_receipt/v1"
        and receipt.get("status") == "PASS"
        and receipt.get("amendment08_journal_contamination_adjudication_sha256")
        == sha256_file(adjudication_path)
        and _safe_regular(receipt_path)
    ):
        raise Amendment06IntegrityError("Amendment 08 semantic Journal gate differs")
    for offset, physical_index in enumerate(A08_CONTAMINATION_INDICES):
        path = _journal_event_path(run_dir, physical_index)
        raw = path.read_bytes() if _safe_regular(path) else b""
        event = raw_events[physical_index]
        canonical = strict_full_canonical_json_bytes(event)
        expected_record = {
            "physical_event_index": physical_index,
            "relative_path": f"{physical_index:08d}.json",
            "raw_size_bytes": len(raw), "raw_sha256": sha256_bytes(raw),
            "canonical_sha256": sha256_bytes(canonical), "event": event,
        }
        if not (
            event == expected_prefix[physical_index]
            and sha256_bytes(raw) == A08_RAW_JOURNAL_SHA256[physical_index]
            and sha256_bytes(canonical) == A08_CANONICAL_JOURNAL_SHA256[physical_index]
            and classified_records[offset] == expected_record
        ):
            raise Amendment06IntegrityError(
                f"Amendment 08 classified Journal record differs: {physical_index}"
            )
    event7 = raw_events[7] if len(raw_events) > 7 else None
    corrigendum = strict_full_load_file(
        run_dir / "provenance/amendment08_code_corrigendum.json"
    )
    event7_base_keys = {
        "event_index", "event", "run_id", "previous_event_sha256", "recorded_at",
    }
    event7_binding_keys = {
        "transaction_id", "status", "authorization", "prompt_sha256",
        "base_scientific_execution_code_manifest_sha256",
        "effective_scientific_execution_code_manifest_sha256",
        "amendment08_code_corrigendum_sha256",
        "amendment08_journal_contamination_adjudication_sha256",
        "source_freeze_receipt_sha256",
        "current_a06_authorized_change_ledger_sha256",
        "a08_source_ledger_event_index",
        "a08_source_ledger_event_canonical_sha256",
        "current_a06_final_prerun_evidence_sha256",
        "a08_prelive_package_lineage_sha256",
        "a08_prelive_package_receipt_sha256", "cache_manifest_sha256",
        "contamination_event_indices", "contamination_event_raw_sha256",
        "contamination_event_canonical_sha256", "a07_overlay_installed",
        "a07_continuation_authorization_consumed",
    }
    if not (
        isinstance(event7, Mapping)
        and set(event7) == event7_base_keys | event7_binding_keys
        and event7.get("event") == "AMENDMENT08_CORRIGENDUM_INSTALLED"
        and event7.get("event_index") == 7
        and event7.get("run_id") == run_dir.name == A08_TARGET_RUN_ID
        and event7.get("previous_event_sha256") == A08_CANONICAL_JOURNAL_SHA256[6]
        and isinstance(event7.get("recorded_at"), str) and event7["recorded_at"]
        and re.fullmatch(r"[0-9a-f]{32}", str(event7.get("transaction_id", "")))
        and event7.get("transaction_id") == receipt.get("transaction_id")
        and event7.get("status") == "PASS"
        and event7.get("authorization") == A08_AUTHORIZATION
        and event7.get("prompt_sha256") == A08_PROMPT_SHA256
        and event7.get("base_scientific_execution_code_manifest_sha256")
        == A07_BASE_SCIENTIFIC_MANIFEST_SHA256
        and event7.get("effective_scientific_execution_code_manifest_sha256")
        == receipt.get("effective_scientific_execution_code_manifest_sha256")
        and event7.get("amendment08_code_corrigendum_sha256")
        == receipt.get("amendment08_code_corrigendum_sha256")
        and event7.get("amendment08_journal_contamination_adjudication_sha256")
        == receipt.get("amendment08_journal_contamination_adjudication_sha256")
        and event7.get("source_freeze_receipt_sha256")
        == receipt.get("source_freeze_receipt_sha256")
        == corrigendum.get("source_freeze_receipt_sha256")
        and event7.get("current_a06_authorized_change_ledger_sha256")
        == receipt.get("current_a06_authorized_change_ledger_sha256")
        == corrigendum.get("current_a06_authorized_change_ledger_sha256")
        and event7.get("a08_source_ledger_event_index")
        == corrigendum.get("a08_source_ledger_event_index")
        == 5
        and event7.get("a08_source_ledger_event_canonical_sha256")
        == corrigendum.get("a08_source_ledger_event_canonical_sha256")
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(event7.get("a08_source_ledger_event_canonical_sha256", "")),
        )
        and event7.get("current_a06_final_prerun_evidence_sha256")
        == receipt.get("current_a06_final_prerun_evidence_sha256")
        == corrigendum.get("current_a06_final_prerun_evidence_sha256")
        and event7.get("a08_prelive_package_lineage_sha256")
        == receipt.get("a08_prelive_package_lineage_sha256")
        == corrigendum.get("a08_prelive_package_lineage_sha256")
        and event7.get("a08_prelive_package_receipt_sha256")
        == receipt.get("a08_prelive_package_receipt_sha256")
        == corrigendum.get("a08_prelive_package_receipt_sha256")
        and event7.get("cache_manifest_sha256") == A07_CACHE_MANIFEST_SHA256
        and event7.get("contamination_event_indices") == [5, 6]
        and event7.get("contamination_event_raw_sha256")
        == list(A08_RAW_JOURNAL_SHA256[5:7])
        and event7.get("contamination_event_canonical_sha256")
        == list(A08_CANONICAL_JOURNAL_SHA256[5:7])
        and event7.get("a07_overlay_installed") is False
        and event7.get("a07_continuation_authorization_consumed") is False
        and sha256_file(_journal_event_path(run_dir, 7))
        == receipt.get("corrigendum_journal_event_raw_sha256")
        and sha256_bytes(strict_full_canonical_json_bytes(event7))
        == receipt.get("corrigendum_journal_head_sha256")
    ):
        raise Amendment06IntegrityError("Amendment 08 semantic gate event 7 differs")
    synthetic_signature = {
        "event": "REPORTING_COMPLETION_COMMITTED", "phase": "REPORTING",
        "variant_id": "GLOBAL", "attempt_id": "a" * 32,
        "completion_sha256": A08_TEST_FIXTURE_SHA256, "run_id": run_dir.name,
    }
    if any(
        int(event["event_index"]) > 6
        and all(event.get(key) == value for key, value in synthetic_signature.items())
        for event in raw_events
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 found a third exact pytest-contamination signature"
        )
    semantic_events = [
        event for event in raw_events
        if int(event["event_index"]) not in A08_CONTAMINATION_INDICES
    ]
    if len(semantic_events) != len(raw_events) - 2 or any(
        event.get("event_index") in A08_CONTAMINATION_INDICES
        for event in semantic_events
    ):
        raise Amendment06IntegrityError("Amendment 08 semantic exclusion cardinality differs")
    raw_lines = b"".join(
        strict_full_canonical_json_bytes(event) + b"\n" for event in raw_events
    )
    semantic_lines = b"".join(
        strict_full_canonical_json_bytes(event) + b"\n" for event in semantic_events
    )
    raw_reporting = sum(
        event.get("event") == "REPORTING_COMPLETION_COMMITTED"
        for event in raw_events
    )
    semantic_reporting = sum(
        event.get("event") == "REPORTING_COMPLETION_COMMITTED"
        for event in semantic_events
    )
    return {
        "status": "PASS", "authorization": A08_AUTHORIZATION,
        "prompt_sha256": A08_PROMPT_SHA256, "target_run_id": run_dir.name,
        "raw_events": raw_events, "semantic_events": semantic_events,
        "semantic_events_by_physical_index": {
            int(event["event_index"]): event for event in semantic_events
        },
        "raw_journal_event_count": len(raw_events),
        "raw_journal_head_sha256": sha256_bytes(
            strict_full_canonical_json_bytes(raw_events[-1])
        ),
        "raw_journal_canonical_json_lines_sha256": sha256_bytes(raw_lines),
        "semantic_journal_event_count": len(semantic_events),
        "semantic_journal_canonical_json_lines_sha256": sha256_bytes(semantic_lines),
        "classified_test_contamination_event_count": 2,
        "classified_test_contamination_event_indices": [5, 6],
        "classified_test_contamination_raw_sha256": list(A08_RAW_JOURNAL_SHA256[5:7]),
        "classified_test_contamination_canonical_sha256": list(
            A08_CANONICAL_JOURNAL_SHA256[5:7]
        ),
        "raw_reporting_completion_events": raw_reporting,
        "semantic_reporting_completion_events": semantic_reporting,
        "overlay_install_receipt_sha256": sha256_file(receipt_path),
        "adjudication_sha256": sha256_file(adjudication_path),
    }


def _scientific_journal_events(run_dir: Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir).resolve()
    if run_dir.name == A08_TARGET_RUN_ID:
        receipt = run_dir / "provenance/amendment08_overlay_install_receipt.json"
        if not _safe_regular(receipt):
            raise Amendment06IntegrityError(
                "The Amendment 08 target requires its receipt-gated semantic Journal view"
            )
        return [dict(event) for event in read_amendment08_journal_views(run_dir)["semantic_events"]]
    return _read_journal(run_dir)


@dataclass(frozen=True)
class LockedInputs:
    training_configuration: dict[str, Any]
    variant_features: dict[str, Any]
    full_features: tuple[str, ...]
    split: pd.DataFrame
    model_parameters: dict[str, Any]


def load_locked_inputs(preflight_run: Path | None = None) -> LockedInputs:
    if preflight_run is None:
        preflight_run = ACCEPTED_PREFLIGHT
    pins = {
        "config/training_configuration.lock.json": TRAINING_CONFIGURATION_SHA256,
        "config/variant_feature_sets.json": VARIANT_FEATURE_SETS_SHA256,
        "config/feature_manifest.csv": FEATURE_MANIFEST_SHA256,
        "config/feature_dependency_graph.json": FEATURE_DEPENDENCY_GRAPH_SHA256,
        "splits/row_split_manifest.csv.gz": ROW_SPLIT_MANIFEST_SHA256,
    }
    for relative, digest in pins.items():
        if not _safe_regular(preflight_run / relative, digest):
            raise Amendment06IntegrityError(f"Locked configuration drifted: {relative}")
    configuration = strict_full_load_file(preflight_run / "config/training_configuration.lock.json")
    variants = strict_full_load_file(preflight_run / "config/variant_feature_sets.json")
    split = pd.read_csv(preflight_run / "splits/row_split_manifest.csv.gz")
    required_split = [
        "stable_candidate_id", "file_name", "ann_id", "scale", "card_id",
        "label", "split", "eligible", "split_order",
    ]
    if list(split.columns) != required_split or split.stable_candidate_id.astype(str).duplicated().any():
        raise Amendment06IntegrityError("Locked Split manifest schema/identity is invalid")
    observed_split_hash = core.split_assignment_hash(
        split.stable_candidate_id, split.split, split.label, split.eligible, split.split_order,
    )
    if observed_split_hash != SPLIT_ASSIGNMENT_SHA256:
        raise Amendment06IntegrityError("Locked Split assignment hash differs")
    counts = {
        value: (
            int((split.split == value).sum()),
            int(split.loc[split.split == value, "label"].sum()),
            int(split.loc[split.split == value, "card_id"].nunique()),
        )
        for value in ("training", "validation", "historical_heldout_ineligible_border")
    }
    if counts != {
        "training": (291024, 75036, 66),
        "validation": (56843, 15952, 18),
        "historical_heldout_ineligible_border": (15696, 3080, 18),
    }:
        raise Amendment06IntegrityError(f"Locked Split counts differ: {counts}")
    full_features = tuple(configuration.get("feature_order", []))
    if len(full_features) != 93 or len(set(full_features)) != 93:
        raise Amendment06IntegrityError("Locked Full feature order is not unique 93D")
    for variant, (count, digest) in VARIANT_FEATURE_CONTRACT.items():
        item = variants.get(variant, {})
        if not (
            item.get("status") == "RUNNABLE"
            and item.get("feature_count") == count
            and item.get("feature_list_sha256") == digest
            and tuple(item.get("retained_features", []))
            == tuple(feature for feature in full_features if feature in set(item.get("retained_features", [])))
        ):
            raise Amendment06IntegrityError(f"Locked feature contract differs: {variant}")
    if variants.get(BLOCKED_VARIANT, {}).get("status") != "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE":
        raise Amendment06IntegrityError("True no-PCA must remain blocked")
    parameters = dict(configuration.get("model", {}).get("full_classifier_parameters", {}))
    if (
        configuration.get("model", {}).get("full_classifier_parameters_sha256")
        != FULL_MODEL_PARAMETERS_SHA256
        or sha256_bytes(core.canonical_json(parameters)) != FULL_MODEL_PARAMETERS_SHA256
    ):
        raise Amendment06IntegrityError("Locked Full XGBoost parameters differ")
    return LockedInputs(configuration, variants, full_features, split, parameters)


def _snapshot_file(source: Path, destination: Path) -> None:
    if not _safe_regular(source):
        raise Amendment06IntegrityError(f"Snapshot source is absent/unsafe: {source}")
    publish_bytes_no_clobber(destination, source.read_bytes())


def _initial_status(run_id: str) -> str:
    return (
        f"RUN_ID={run_id}\nRUN_KIND={RUN_KIND}\nRUN_STATE=FULL_IN_PROGRESS\n"
        f"FULL_AUTHORIZATION={AUTHORIZATION}\nFULL_MODELS_FROZEN=NO\n"
        "EXTERNAL_AUDIT_OPENED=NO\nFULL_TRAINING_EXECUTED=YES_IN_PROGRESS\n"
    )


def _run_intent_path() -> Path:
    return A06_RUNTIME / "authorization/full_run_intent.json"


def _run_receipt_path() -> Path:
    return A06_RUNTIME / "authorization/full_run_receipt.json"


def _new_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    binding = sha256_bytes(strict_full_canonical_json_bytes({
        "authorization": AUTHORIZATION, "preflight": ACCEPTED_PREFLIGHT.name,
        "smoke": ACCEPTED_SMOKE.name, "split": SPLIT_ASSIGNMENT_SHA256,
    }))[:8]
    return f"run_{timestamp}_{binding}{RUN_SUFFIX}"


def _reserve_full_run_id() -> dict[str, Any]:
    """Reserve the sole Run ID before any Results-directory side effect."""
    intent_path = _run_intent_path()
    if intent_path.exists():
        intent = strict_full_load_file(intent_path)
    else:
        run_id = _new_run_id()
        intent = {
            "authorization": AUTHORIZATION,
            "authorization_reserved": True,
            "run_id": run_id,
            "run_path": str(RESULTS_ROOT / run_id),
            "preflight_run_id": ACCEPTED_PREFLIGHT.name,
            "smoke_run_id": ACCEPTED_SMOKE.name,
            "candidate_split_hash": SPLIT_ASSIGNMENT_SHA256,
            "reserved_at": _now(),
        }
        publish_strict_json_no_clobber(intent_path, intent)
    run_id = str(intent.get("run_id", ""))
    if not (
        intent.get("authorization") == AUTHORIZATION
        and intent.get("authorization_reserved") is True
        and run_id.endswith(RUN_SUFFIX)
        and intent.get("run_path") == str(RESULTS_ROOT / run_id)
        and intent.get("preflight_run_id") == ACCEPTED_PREFLIGHT.name
        and intent.get("smoke_run_id") == ACCEPTED_SMOKE.name
        and intent.get("candidate_split_hash") == SPLIT_ASSIGNMENT_SHA256
    ):
        raise Amendment06IntegrityError("Full Run authorization intent is invalid")
    return intent


def _publish_or_validate_run_receipt(run_dir: Path) -> dict[str, Any]:
    identity_sha256 = sha256_file(run_dir / "config/run_identity.lock.json")
    receipt_path = _run_receipt_path()
    if receipt_path.exists():
        receipt = strict_full_load_file(receipt_path)
    else:
        if _AMENDMENT08_REHEARSAL_CONTEXT.get() is not None:
            raise Amendment06IntegrityError(
                "The isolated rehearsal requires the byte-exact copied A06 receipt"
            )
        receipt = {
            "authorization": AUTHORIZATION,
            "authorization_consumed": True,
            "run_id": run_dir.name,
            "run_path": str(run_dir),
            "run_identity_sha256": identity_sha256,
            "consumed_at": _now(),
        }
        publish_strict_json_no_clobber(receipt_path, receipt)
    path_matches = receipt.get("run_path") == str(run_dir)
    if not path_matches:
        path_matches = _amendment08_rehearsal_receipt_path_matches(
            run_dir=run_dir, receipt=receipt, identity_sha256=identity_sha256,
        )
    if not (
        receipt.get("authorization") == AUTHORIZATION
        and receipt.get("authorization_consumed") is True
        and receipt.get("run_id") == run_dir.name
        and path_matches
        and receipt.get("run_identity_sha256") == identity_sha256
    ):
        raise Amendment06IntegrityError("Full Run authorization receipt disagrees")
    return receipt


def _state_marker_path(run_dir: Path, phase: str) -> Path:
    if phase not in RUN_PHASES:
        raise Amendment06IntegrityError(f"Unknown Full state: {phase}")
    return run_dir / f"provenance/state/{RUN_PHASES.index(phase):02d}_{phase}.json"


def _current_run_state(run_dir: Path) -> str:
    identity = strict_full_load_file(run_dir / "config/run_identity.lock.json")
    if identity.get("state") != "FULL_IN_PROGRESS":
        raise Amendment06IntegrityError("Immutable Run identity initial state changed")
    observed: list[str] = []
    previous_sha256: str | None = None
    for phase in RUN_PHASES:
        path = _state_marker_path(run_dir, phase)
        if path.exists():
            marker = strict_full_load_file(path)
            if not (
                marker.get("run_id") == run_dir.name
                and marker.get("run_kind") == RUN_KIND
                and marker.get("phase") == phase
                and marker.get("phase_index") == RUN_PHASES.index(phase)
                and marker.get("status") == "PASS"
                and marker.get("previous_marker_sha256") == previous_sha256
            ):
                raise Amendment06IntegrityError(f"Invalid Full state marker: {phase}")
            observed.append(phase)
            previous_sha256 = sha256_file(path)
        elif observed:
            # A later marker after this gap is checked below.
            if any(_state_marker_path(run_dir, later).exists() for later in RUN_PHASES[RUN_PHASES.index(phase) + 1:]):
                raise Amendment06IntegrityError("Full state marker chain has a gap")
            break
    if not observed or observed[0] != "FULL_IN_PROGRESS":
        raise Amendment06IntegrityError("Initial Full state marker is absent")
    return observed[-1]


def _create_authorized_run(
    *, command_line: str, admission: Mapping[str, Any], locked: LockedInputs,
) -> Path:
    intent = _reserve_full_run_id()
    run_id = str(intent["run_id"])
    temporary = EXPECTED_RUNTIME_ROOT / "run_creation" / run_id
    run_dir = RESULTS_ROOT / run_id
    runs = existing_full_runs()
    if runs:
        if runs == [run_dir]:
            _validate_run_identity(run_dir)
            return run_dir
        raise Amendment06IntegrityError("The one Amendment 06 Full Run authorization is already consumed")
    if temporary.exists():
        if temporary.is_symlink() or not temporary.is_dir():
            raise Amendment06IntegrityError("Full Run staging path is unsafe")
        quarantine = (
            EXPECTED_RUNTIME_ROOT / "run_creation_quarantine"
            / f"{run_id}_{uuid.uuid4().hex}"
        )
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        _rename_directory_no_clobber(temporary, quarantine)
    scientific_bytes, _ = _code_manifest(
        (*SCIENTIFIC_CODE_RELATIVES, *SCIENTIFIC_TEST_RELATIVES),
        role="SCIENTIFIC_EXECUTION_OR_TEST",
    )
    package_bytes, _ = _code_manifest(
        PACKAGE_CODE_RELATIVES, role="PACKAGE_VALIDATOR_OR_TEST",
    )
    temporary.mkdir(mode=0o700, parents=True)
    try:
        for relative in (
            "config", "splits", "models", "predictions", "metrics", "statistics",
            "tables", "figures", "timing", "provenance", "logs", "docs",
            "provenance/code_snapshot", "provenance/tests_snapshot", "provenance/state",
        ):
            (temporary / relative).mkdir(parents=True, exist_ok=True)
        identity = {
            "run_id": run_id, "run_kind": RUN_KIND, "state": "FULL_IN_PROGRESS",
            "full_authorization": AUTHORIZATION,
            "external_review_decision": EXTERNAL_REVIEW_DECISION,
            "accepted_preflight_path": str(ACCEPTED_PREFLIGHT),
            "accepted_preflight_run_id": ACCEPTED_PREFLIGHT.name,
            "accepted_preflight_review_zip_sha256": PINNED_FILES[ACCEPTED_PREFLIGHT_ZIP],
            "accepted_smoke_path": str(ACCEPTED_SMOKE),
            "accepted_smoke_run_id": ACCEPTED_SMOKE.name,
            "accepted_smoke_review_zip_sha256": PINNED_FILES[ACCEPTED_SMOKE_REVIEW_ZIP],
            "candidate_split_hash": SPLIT_ASSIGNMENT_SHA256,
            "base_seed": BASE_SEED, "primary_model_seed": PRIMARY_MODEL_SEED,
            "decision_threshold": DECISION_THRESHOLD,
            "created_at": _now(), "command_line": command_line,
            "scientific_execution_code_manifest_sha256": sha256_bytes(scientific_bytes),
            "package_validator_code_manifest_initial_sha256": sha256_bytes(package_bytes),
            "scientific_code_frozen": True,
            "new_preflight_created": False, "new_smoke_created": False,
            "stability_authorized": False, "threshold_tuning_authorized": False,
        }
        atomic_write_strict_json(temporary / "config/run_identity.lock.json", identity)
        atomic_write_bytes(temporary / "provenance/scientific_execution_code_manifest.tsv", scientific_bytes)
        atomic_write_bytes(temporary / "provenance/package_validator_code_manifest_initial.tsv", package_bytes)
        for relative in (
            "config/training_configuration.lock.json", "config/variant_feature_sets.json",
            "config/feature_manifest.csv", "config/feature_dependency_graph.json",
            "config/locked_split_identity.json", "splits/row_split_manifest.csv.gz",
        ):
            _snapshot_file(ACCEPTED_PREFLIGHT / relative, temporary / relative)
        _snapshot_file(PROMPT_PATH, temporary / "provenance/amendment06_prompt_snapshot.txt")
        atomic_write_strict_json(
            temporary / "provenance/amendment06_external_review_authorization.json",
            {
                "authorization": AUTHORIZATION, "decision": EXTERNAL_REVIEW_DECISION,
                "distinct_full_run_ids_authorized": 1, "run_id": run_id,
                "same_run_resume_authorized": True, "same_run_package_retry_authorized": True,
                "stability_or_multiseed_authorized": False,
                "threshold_tuning_authorized": False,
            },
        )
        atomic_write_strict_json(
            temporary / "provenance/amendment06_code_lineage.json",
            {
                "status": "PASS", "pre_edit_admission_sha256": sha256_file(PRE_EDIT_ADMISSION),
                "authorized_change_ledger_sha256": admission["authorized_change_chain"]["ledger_sha256"],
                "authorized_change_latest_event": admission["authorized_change_chain"]["latest_event_index"],
                "live_code_tree_sha256": admission["authorized_change_chain"]["canonical_tree_sha256"],
                "final_prerun_evidence_sha256": admission["final_prerun_evidence"]["sha256"],
                "scientific_manifest_sha256": sha256_bytes(scientific_bytes),
                "package_manifest_initial_sha256": sha256_bytes(package_bytes),
            },
        )
        for relative in (*SCIENTIFIC_CODE_RELATIVES, *SCIENTIFIC_TEST_RELATIVES):
            source = STUDY_ROOT / relative
            destination_root = "tests_snapshot" if relative.startswith("tests/") else "code_snapshot"
            _snapshot_file(source, temporary / "provenance" / destination_root / Path(relative).name)
        _snapshot_file(
            STUDY_ROOT / "docs/PROTOCOL_AMENDMENT_06.md",
            temporary / "docs/PROTOCOL_AMENDMENT_06.md",
        )
        publish_strict_json_no_clobber(
            _state_marker_path(temporary, "FULL_IN_PROGRESS"),
            {
                "run_id": run_id, "run_kind": RUN_KIND,
                "phase": "FULL_IN_PROGRESS", "phase_index": 0,
                "status": "PASS", "previous_marker_sha256": None,
                "recorded_at": _now(),
            },
        )
        _rename_directory_no_clobber(temporary, run_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _publish_or_validate_run_receipt(run_dir)
    return run_dir


def _validate_run_identity(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    if run_dir.parent != RESULTS_ROOT.resolve() or not run_dir.name.endswith(RUN_SUFFIX):
        raise Amendment06ResumeError("Resume target is not the authorized Amendment 06 Full Run")
    runs = existing_full_runs()
    if runs != [run_dir]:
        raise Amendment06ResumeError(f"Expected exactly this one Full Run, observed: {runs}")
    identity = strict_full_load_file(run_dir / "config/run_identity.lock.json")
    if not (
        identity.get("run_id") == run_dir.name
        and identity.get("run_kind") == RUN_KIND
        and identity.get("full_authorization") == AUTHORIZATION
        and identity.get("candidate_split_hash") == SPLIT_ASSIGNMENT_SHA256
        and identity.get("accepted_preflight_path") == str(ACCEPTED_PREFLIGHT)
        and identity.get("accepted_smoke_path") == str(ACCEPTED_SMOKE)
        and identity.get("state") == "FULL_IN_PROGRESS"
    ):
        raise Amendment06ResumeError("Full Run identity/authorization is invalid")
    intent = strict_full_load_file(_run_intent_path())
    if intent.get("run_id") != run_dir.name:
        raise Amendment06ResumeError("Full Run authorization intent disagrees")
    _publish_or_validate_run_receipt(run_dir)
    validate_frozen_code_manifests(run_dir)
    _current_run_state(run_dir)
    return identity


def _set_run_state(run_dir: Path, target: str) -> dict[str, Any]:
    current = _current_run_state(run_dir)
    if target not in RUN_PHASES or RUN_PHASES.index(target) < RUN_PHASES.index(current):
        raise Amendment06IntegrityError(f"Invalid Full state transition: {current} -> {target}")
    if target != current:
        if RUN_PHASES.index(target) != RUN_PHASES.index(current) + 1:
            raise Amendment06IntegrityError(f"Full state transition skips a phase: {current} -> {target}")
        publish_strict_json_no_clobber(
            _state_marker_path(run_dir, target),
            {
                "run_id": run_dir.name, "run_kind": RUN_KIND,
                "phase": target, "phase_index": RUN_PHASES.index(target),
                "status": "PASS", "previous_phase": current,
                "previous_marker_sha256": sha256_file(_state_marker_path(run_dir, current)),
                "recorded_at": _now(),
            },
        )
    return {"run_id": run_dir.name, "run_kind": RUN_KIND, "state": target}


@dataclass(frozen=True)
class DevelopmentPopulation:
    training: pd.DataFrame
    validation: pd.DataFrame
    training_labels: np.ndarray
    validation_labels: np.ndarray
    training_weights: np.ndarray
    validation_weights: np.ndarray


@dataclass(frozen=True)
class FrozenPopulationCache:
    root: Path
    manifest: dict[str, Any]
    imputer: dict[str, Any]
    full_features: tuple[str, ...]
    pre_X: np.ndarray
    pre_y: np.ndarray
    pre_weights: np.ndarray
    post_X: np.ndarray
    post_y: np.ndarray
    post_weights: np.ndarray
    validation_X: np.ndarray

    @property
    def binding_sha256(self) -> str:
        return str(self.manifest["frozen_population_sha256"])

    @property
    def imputer_sha256(self) -> str:
        return str(self.manifest["imputer_sha256"])


@dataclass(frozen=True)
class VariantMatrices:
    variant: str
    classification: str
    features: tuple[str, ...]
    indices: tuple[int, ...]
    X: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    validation_X: np.ndarray
    population: str
    population_evidence: dict[str, Any]


def _source_inventory_bytes() -> bytes:
    source = ACCEPTED_PREFLIGHT / "provenance/source_input_hashes_post.tsv"
    if not _safe_regular(source, SOURCE_INVENTORY_SHA256):
        raise Amendment06IntegrityError("Sealed source-input inventory is absent")
    raw = source.read_bytes()
    frame = pd.read_csv(
        io.BytesIO(raw), sep="\t", dtype={"path": str, "sha256": str},
    )
    if list(frame.columns) != ["path", "size_bytes", "sha256"] or len(frame) != 15:
        raise Amendment06IntegrityError("Sealed source-input inventory schema/count differs")
    if frame.path.duplicated().any():
        raise Amendment06IntegrityError("Sealed source-input inventory has duplicate paths")
    for row in frame.itertuples(index=False):
        path = Path(str(row.path))
        if not _safe_regular(path, str(row.sha256)) or path.stat().st_size != int(row.size_bytes):
            raise Amendment06IntegrityError(f"Pinned source input drifted: {path}")
    if not (
        ((frame.path == str(PRIMARY_TABLE)) & (frame.sha256 == PRIMARY_TABLE_SHA256)).sum() == 1
        and ((frame.path == str(EXTERNAL_AUDIT)) & (frame.sha256 == EXTERNAL_AUDIT_SHA256)).sum() == 1
    ):
        raise Amendment06IntegrityError("Primary/Audit source pins are absent from sealed inventory")
    if sha256_bytes(raw) != SOURCE_INVENTORY_SHA256:
        raise Amendment06IntegrityError("Sealed source-input inventory byte pin differs")
    return raw


def _snapshot_source_pre(run_dir: Path) -> None:
    destination = run_dir / "provenance/source_input_hashes_pre.tsv"
    data = _source_inventory_bytes()
    if destination.exists():
        if not _safe_regular(destination, sha256_bytes(data)):
            raise Amendment06IntegrityError("Full source pre-inventory drifted")
    else:
        publish_bytes_no_clobber(destination, data)


def _snapshot_source_post(run_dir: Path) -> None:
    destination = run_dir / "provenance/source_input_hashes_post.tsv"
    data = _source_inventory_bytes()
    if destination.exists():
        if not _safe_regular(destination, sha256_bytes(data)):
            raise Amendment06IntegrityError("Full source post-inventory drifted")
    else:
        publish_bytes_no_clobber(destination, data)


def load_development_population(locked: LockedInputs) -> DevelopmentPopulation:
    _progress("LOCKED_DEVELOPMENT_INPUT", "START")
    training = core.load_training_verified(PRIMARY_TABLE, PRIMARY_TABLE_SHA256)
    core.assert_training_matches_reviewed_split(training, locked.split)
    ordered_training = locked.split.loc[
        locked.split.split.astype(str).eq("training") & locked.split.eligible.astype(bool)
    ].sort_values("split_order", kind="stable")
    ordered_validation = locked.split.loc[
        locked.split.split.astype(str).eq("validation") & locked.split.eligible.astype(bool)
    ].sort_values("split_order", kind="stable")
    train = training.loc[ordered_training.index].copy().reset_index(drop=True)
    validation = training.loc[ordered_validation.index].copy().reset_index(drop=True)
    if not (
        len(train) == 291_024 and int(train.label.sum()) == 75_036
        and train.card_id.nunique() == 66
        and len(validation) == 56_843 and int(validation.label.sum()) == 15_952
        and validation.card_id.nunique() == 18
        and set(train.card_id.astype(str)).isdisjoint(validation.card_id.astype(str))
        and np.array_equal(
            train.stable_candidate_id.astype(str).to_numpy(),
            ordered_training.stable_candidate_id.astype(str).to_numpy(),
        )
        and np.array_equal(
            validation.stable_candidate_id.astype(str).to_numpy(),
            ordered_validation.stable_candidate_id.astype(str).to_numpy(),
        )
    ):
        raise Amendment06IntegrityError("Locked development population/order differs")
    training_weights, weight_evidence = core.review_weights(
        train, historical_geometry=False,
    )
    validation_weights, _ = core.review_weights(
        validation, historical_geometry=False,
    )
    if not (
        training_weights.dtype == np.float32
        and np.isfinite(training_weights).all()
        and np.all(training_weights > 0)
        and weight_evidence.get("weight_sha256_float32_le")
        == sha256_array(training_weights.astype("<f4", copy=False))
    ):
        raise Amendment06IntegrityError("Locked review-weight vector is invalid")
    _progress("LOCKED_DEVELOPMENT_INPUT", "COMPLETE", train_rows=len(train), validation_rows=len(validation))
    return DevelopmentPopulation(
        training=train, validation=validation,
        training_labels=train.label.to_numpy(np.int8),
        validation_labels=validation.label.to_numpy(np.int8),
        training_weights=training_weights,
        validation_weights=validation_weights,
    )


def _feature_list_sha256(features: Sequence[str]) -> str:
    return sha256_bytes(("\n".join(map(str, features)) + "\n").encode("utf-8"))


def _lineage_bytes(values: Sequence[Any]) -> bytes:
    return ("\n".join(map(str, values)) + "\n").encode("utf-8")


def _save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, np.ascontiguousarray(values), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _cache_file_records(root: Path, relatives: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in sorted(relatives):
        path = root / relative
        if not _safe_regular(path):
            raise Amendment06IntegrityError(f"Frozen cache member is absent/unsafe: {relative}")
        records.append({
            "relative_path": relative, "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return records


def _portable_imputer_payload(
    *, run_id: str, features: Sequence[str], statistics: np.ndarray,
) -> dict[str, Any]:
    medians = np.asarray(statistics, dtype=np.float64)
    if medians.shape != (93,) or not np.isfinite(medians).all():
        raise Amendment06IntegrityError("Training-only 93D median imputer is invalid")
    content: dict[str, Any] = {
        "run_id": run_id,
        "strategy": "median",
        "fit_population": "LOCKED_ORDERED_TRAINING_ONLY",
        "fit_rows": 291_024,
        "feature_count": 93,
        "feature_order": list(features),
        "feature_order_sha256": _feature_list_sha256(features),
        "statistics": medians.tolist(),
        "statistics_dtype": "float64",
        "transform_output_dtype": "float32",
        "primary_table_sha256": PRIMARY_TABLE_SHA256,
        "split_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
    }
    content["payload_sha256"] = sha256_bytes(strict_full_canonical_json_bytes(content))
    return content


def _validate_stage_counts(stages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_name = {str(stage.get("stage")): dict(stage) for stage in stages}
    if set(by_name) != set(EXPECTED_STAGE_COUNTS):
        raise Amendment06IntegrityError(f"Frozen stage set differs: {sorted(by_name)}")
    result: dict[str, Any] = {}
    for stage, (rows, negative, positive) in EXPECTED_STAGE_COUNTS.items():
        observed = by_name[stage]
        if (
            int(observed.get("rows", -1)), int(observed.get("negative", -1)),
            int(observed.get("positive", -1)),
        ) != (rows, negative, positive):
            raise Amendment06IntegrityError(f"Frozen stage count differs: {stage}")
        result[stage] = {"rows": rows, "negative": negative, "positive": positive}
    if by_name["post_shuffle"].get("shuffle_applied") is not False:
        raise Amendment06IntegrityError("Post-augmentation shuffle must be a no-op")
    if int(by_name["post_safe_smote"].get("safe_smote_synthetic_rows", -1)) != 147_841:
        raise Amendment06IntegrityError("Safe-SMOTE synthetic row count differs")
    result["post_shuffle"]["shuffle_applied"] = False
    result["post_safe_smote"]["safe_smote_synthetic_rows"] = 147_841
    return result


def _build_population_cache(
    *, run_dir: Path, locked: LockedInputs, development: DevelopmentPopulation,
    destination: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    temporary = destination.parent / f".{destination.name}.building.{os.getpid()}.{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        _progress("FROZEN_FULL_SPACE_RESAMPLING", "START", rows=len(development.training), features=93)
        if tuple(core.get_r92_feature_order()) != locked.full_features:
            raise Amendment06IntegrityError("Live canonical 93D order differs from locked order")
        frozen = paired_resampling.build_frozen_resampling_population(
            development.training.loc[:, locked.full_features],
            development.training_labels,
            development.training_weights,
            development.training.stable_candidate_id.astype(str),
            full_features=locked.full_features,
            seed=BASE_SEED,
            backend=core,
        )
        stage_counts = _validate_stage_counts(frozen.stages)
        validation = paired_resampling.transform_validation_full_space(
            frozen, development.validation.loc[:, locked.full_features],
        )
        arrays = {
            "pre_smote_X.npy": np.asarray(frozen.pre_smote_X, dtype=np.float32),
            "pre_smote_y.npy": np.asarray(frozen.pre_smote_y, dtype=np.int8),
            "pre_smote_weights.npy": np.asarray(frozen.pre_smote_weights, dtype=np.float32),
            "post_smote_X.npy": np.asarray(frozen.post_smote_X, dtype=np.float32),
            "post_smote_y.npy": np.asarray(frozen.post_smote_y, dtype=np.int8),
            "post_smote_weights.npy": np.asarray(frozen.post_smote_weights, dtype=np.float32),
            "validation_X.npy": np.asarray(validation.X, dtype=np.float32),
        }
        for relative, values in arrays.items():
            _save_npy(temporary / relative, values)
        publish_bytes_no_clobber(
            temporary / "pre_smote_lineage.txt", _lineage_bytes(frozen.pre_smote_lineage),
        )
        publish_bytes_no_clobber(
            temporary / "post_smote_lineage.txt", _lineage_bytes(frozen.post_smote_lineage),
        )
        imputer = _portable_imputer_payload(
            run_id=run_dir.name, features=locked.full_features,
            statistics=np.asarray(frozen.imputer.statistics_),
        )
        publish_strict_json_no_clobber(temporary / "portable_imputer.json", imputer)
        relatives = [*arrays, "pre_smote_lineage.txt", "post_smote_lineage.txt", "portable_imputer.json"]
        file_records = _cache_file_records(temporary, relatives)
        scientific_binding = {
            "run_id": run_dir.name,
            "primary_table_sha256": PRIMARY_TABLE_SHA256,
            "split_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
            "base_seed": BASE_SEED,
            "feature_order_sha256": frozen.feature_order_sha256,
            "raw_review_weight_sha256": sha256_array(development.training_weights.astype("<f4", copy=False)),
            "validation_X_sha256": sha256_array(arrays["validation_X.npy"]),
            "stages": [dict(stage) for stage in frozen.stages],
            "stage_counts": stage_counts,
            "cache_files": file_records,
        }
        frozen_population_sha256 = sha256_bytes(strict_full_canonical_json_bytes(scientific_binding))
        manifest = {
            **scientific_binding,
            "status": "PASS",
            "classification": "ONE_FROZEN_PAIRED_FULL_SPACE_POPULATION",
            "unique_realizations": 1,
            "augmentation_generation_count": 1,
            "safe_smote_generation_count": 1,
            "safe_smote_synthetic_rows": 147_841,
            "frozen_population_sha256": frozen_population_sha256,
            "imputer_sha256": sha256_bytes(strict_full_json_bytes(imputer)),
            "cache_packaged": False,
            "same_run_resume_cache": True,
        }
        publish_strict_json_no_clobber(temporary / "cache_manifest.json", manifest)
        _rename_directory_no_clobber(temporary, destination)
        _progress("FROZEN_FULL_SPACE_RESAMPLING", "COMPLETE", post_smote_rows=POST_SMOTE_ROWS)
        return manifest, imputer
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_population_cache(
    *, run_dir: Path, destination: Path, locked: LockedInputs,
    development: DevelopmentPopulation,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_all_members = {
        "cache_manifest.json", "pre_smote_X.npy", "pre_smote_y.npy",
        "pre_smote_weights.npy", "post_smote_X.npy", "post_smote_y.npy",
        "post_smote_weights.npy", "validation_X.npy",
        "pre_smote_lineage.txt", "post_smote_lineage.txt", "portable_imputer.json",
    }
    observed_members: set[str] = set()
    for member in destination.iterdir():
        if member.is_symlink() or not member.is_file():
            raise Amendment06IntegrityError(f"Frozen cache has an unsafe member: {member}")
        observed_members.add(member.name)
    if observed_members != expected_all_members:
        raise Amendment06IntegrityError("Frozen cache member set differs")
    manifest = strict_full_load_file(destination / "cache_manifest.json")
    imputer = strict_full_load_file(destination / "portable_imputer.json")
    expected_members = tuple(sorted((
        "pre_smote_X.npy", "pre_smote_y.npy", "pre_smote_weights.npy",
        "post_smote_X.npy", "post_smote_y.npy", "post_smote_weights.npy",
        "validation_X.npy", "pre_smote_lineage.txt", "post_smote_lineage.txt",
        "portable_imputer.json",
    )))
    records = manifest.get("cache_files")
    if (
        not isinstance(records, list)
        or tuple(str(item.get("relative_path")) for item in records) != expected_members
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"relative_path", "size_bytes", "sha256"}
            for item in records
        )
    ):
        raise Amendment06IntegrityError("Frozen cache manifest member set/order differs")
    for item in records:
        path = destination / str(item.get("relative_path"))
        if not (
            _safe_regular(path, str(item.get("sha256")))
            and path.stat().st_size == int(item.get("size_bytes", -1))
        ):
            raise Amendment06IntegrityError(f"Frozen cache member drifted: {path}")
    scientific_binding = {
        key: manifest[key]
        for key in (
            "run_id", "primary_table_sha256", "split_assignment_sha256", "base_seed",
            "feature_order_sha256", "raw_review_weight_sha256", "validation_X_sha256",
            "stages", "stage_counts", "cache_files",
        )
    }
    required_imputer_keys = {
        "run_id", "strategy", "fit_population", "fit_rows", "feature_count",
        "feature_order", "feature_order_sha256", "statistics", "statistics_dtype",
        "transform_output_dtype", "primary_table_sha256", "split_assignment_sha256",
        "payload_sha256",
    }
    if not (
        manifest.get("run_id") == run_dir.name
        and manifest.get("status") == "PASS"
        and manifest.get("classification") == "ONE_FROZEN_PAIRED_FULL_SPACE_POPULATION"
        and manifest.get("primary_table_sha256") == PRIMARY_TABLE_SHA256
        and manifest.get("split_assignment_sha256") == SPLIT_ASSIGNMENT_SHA256
        and manifest.get("base_seed") == BASE_SEED
        and manifest.get("unique_realizations") == 1
        and manifest.get("augmentation_generation_count") == 1
        and manifest.get("safe_smote_generation_count") == 1
        and manifest.get("safe_smote_synthetic_rows") == 147_841
        and manifest.get("cache_packaged") is False
        and manifest.get("same_run_resume_cache") is True
        and manifest.get("frozen_population_sha256")
        == sha256_bytes(strict_full_canonical_json_bytes(scientific_binding))
        and manifest.get("imputer_sha256") == sha256_file(destination / "portable_imputer.json")
        and set(imputer) == required_imputer_keys
        and imputer.get("run_id") == run_dir.name
        and imputer.get("strategy") == "median"
        and imputer.get("fit_population") == "LOCKED_ORDERED_TRAINING_ONLY"
        and imputer.get("fit_rows") == 291_024
        and imputer.get("feature_count") == 93
        and isinstance(imputer.get("feature_order"), list)
        and tuple(map(str, imputer["feature_order"])) == locked.full_features
        and len(imputer["feature_order"]) == 93
        and len(set(map(str, imputer["feature_order"]))) == 93
        and imputer.get("feature_order_sha256") == _feature_list_sha256(imputer["feature_order"])
        and manifest.get("feature_order_sha256") == imputer.get("feature_order_sha256")
        and imputer.get("statistics_dtype") == "float64"
        and imputer.get("transform_output_dtype") == "float32"
        and imputer.get("primary_table_sha256") == PRIMARY_TABLE_SHA256
        and imputer.get("split_assignment_sha256") == SPLIT_ASSIGNMENT_SHA256
        and isinstance(imputer.get("statistics"), list)
        and len(imputer["statistics"]) == 93
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) for value in imputer["statistics"]
        )
        and imputer.get("payload_sha256") == sha256_bytes(strict_full_canonical_json_bytes({
            key: value for key, value in imputer.items() if key != "payload_sha256"
        }))
    ):
        raise Amendment06IntegrityError("Frozen cache scientific binding differs")
    stage_counts = _validate_stage_counts(manifest["stages"])
    if manifest.get("stage_counts") != stage_counts:
        raise Amendment06IntegrityError("Frozen cache redundant stage counts differ")
    stages = list(manifest["stages"])
    if [str(stage.get("stage")) for stage in stages] != list(EXPECTED_STAGE_COUNTS):
        raise Amendment06IntegrityError("Frozen cache stage order differs")
    required_stage_keys = {
        "stage", "rows", "features", "negative", "positive",
        "feature_order_sha256", "X_sha256", "y_sha256", "weight_sha256",
        "source_lineage_sha256", "row_order_sha256",
    }
    for stage in stages:
        if not required_stage_keys.issubset(stage):
            raise Amendment06IntegrityError("Frozen cache stage hash schema differs")
        if not (
            stage.get("features") == 93
            and stage.get("feature_order_sha256") == manifest["feature_order_sha256"]
            and all(
                re.fullmatch(r"[0-9a-f]{64}", str(stage.get(key, "")))
                for key in (
                    "X_sha256", "y_sha256", "weight_sha256",
                    "source_lineage_sha256", "row_order_sha256",
                )
            )
            and stage.get("source_lineage_sha256") == stage.get("row_order_sha256")
        ):
            raise Amendment06IntegrityError(f"Frozen cache stage binding differs: {stage.get('stage')}")
    arrays = {
        name: np.load(destination / name, mmap_mode="r", allow_pickle=False)
        for name in (
            "pre_smote_X.npy", "pre_smote_y.npy", "pre_smote_weights.npy",
            "post_smote_X.npy", "post_smote_y.npy", "post_smote_weights.npy",
            "validation_X.npy",
        )
    }
    expected_arrays = {
        "pre_smote_X.npy": ((PRE_SMOTE_ROWS, 93), np.dtype("float32")),
        "pre_smote_y.npy": ((PRE_SMOTE_ROWS,), np.dtype("int8")),
        "pre_smote_weights.npy": ((PRE_SMOTE_ROWS,), np.dtype("float32")),
        "post_smote_X.npy": ((POST_SMOTE_ROWS, 93), np.dtype("float32")),
        "post_smote_y.npy": ((POST_SMOTE_ROWS,), np.dtype("int8")),
        "post_smote_weights.npy": ((POST_SMOTE_ROWS,), np.dtype("float32")),
        "validation_X.npy": ((56_843, 93), np.dtype("float32")),
    }
    for name, (shape, dtype) in expected_arrays.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise Amendment06IntegrityError(f"Frozen cache array shape/dtype differs: {name}")
    for name in ("pre_smote_X.npy", "post_smote_X.npy", "validation_X.npy"):
        if not np.isfinite(arrays[name]).all():
            raise Amendment06IntegrityError(f"Frozen cache matrix is nonfinite: {name}")
    for name in ("pre_smote_weights.npy", "post_smote_weights.npy"):
        if not (np.isfinite(arrays[name]).all() and np.all(arrays[name] > 0)):
            raise Amendment06IntegrityError(f"Frozen cache weights are invalid: {name}")
    for name in ("pre_smote_y.npy", "post_smote_y.npy"):
        if not np.isin(arrays[name], (0, 1)).all():
            raise Amendment06IntegrityError(f"Frozen cache labels are nonbinary: {name}")
    by_name = {str(stage["stage"]): stage for stage in stages}
    endpoint_bindings = (
        (
            "post_jitter_pre_smote", "pre_smote_X.npy", "pre_smote_y.npy",
            "pre_smote_weights.npy", "pre_smote_lineage.txt",
        ),
        (
            "post_shuffle", "pre_smote_X.npy", "pre_smote_y.npy",
            "pre_smote_weights.npy", "pre_smote_lineage.txt",
        ),
        (
            "post_safe_smote", "post_smote_X.npy", "post_smote_y.npy",
            "post_smote_weights.npy", "post_smote_lineage.txt",
        ),
    )
    for stage_name, x_name, y_name, weight_name, lineage_name in endpoint_bindings:
        stage = by_name[stage_name]
        lineage_sha = sha256_file(destination / lineage_name)
        if not (
            stage["X_sha256"] == sha256_array(arrays[x_name])
            and stage["y_sha256"] == sha256_array(arrays[y_name])
            and stage["weight_sha256"] == sha256_array(arrays[weight_name])
            and stage["source_lineage_sha256"] == lineage_sha
            and stage["row_order_sha256"] == lineage_sha
            and int((arrays[y_name] == 0).sum()) == stage["negative"]
            and int((arrays[y_name] == 1).sum()) == stage["positive"]
            and int(len(arrays[y_name])) == stage["rows"]
        ):
            raise Amendment06IntegrityError(f"Frozen cache endpoint bytes differ: {stage_name}")
    invariant_keys = (
        "rows", "features", "negative", "positive", "feature_order_sha256",
        "X_sha256", "y_sha256", "weight_sha256", "source_lineage_sha256",
        "row_order_sha256",
    )
    if any(
        by_name["post_jitter_pre_smote"].get(key) != by_name["post_shuffle"].get(key)
        for key in invariant_keys
    ):
        raise Amendment06IntegrityError("Frozen no-op post-shuffle bytes differ")
    if not (
        manifest.get("raw_review_weight_sha256")
        == sha256_array(development.training_weights.astype("<f4", copy=False))
        == by_name["raw_locked_post_split_shuffle"]["weight_sha256"]
        and manifest.get("validation_X_sha256") == sha256_array(arrays["validation_X.npy"])
    ):
        raise Amendment06IntegrityError("Frozen cache raw-weight/validation binding differs")
    recomputed_medians: list[float] = []
    for feature in locked.full_features:
        values = pd.to_numeric(
            development.training[feature], errors="coerce",
        ).to_numpy(np.float64)
        if np.isinf(values).any() or np.isnan(values).all():
            raise Amendment06IntegrityError(f"Locked imputer source is invalid: {feature}")
        recomputed_medians.append(float(np.nanmedian(values)))
    if not np.array_equal(
        np.asarray(recomputed_medians, dtype=np.float64),
        np.asarray(imputer["statistics"], dtype=np.float64),
    ):
        raise Amendment06IntegrityError("Frozen cache imputer medians differ from locked training")
    validation_values = development.validation.loc[:, locked.full_features].apply(
        pd.to_numeric, errors="coerce",
    ).to_numpy(np.float64)
    if np.isinf(validation_values).any():
        raise Amendment06IntegrityError("Locked validation contains infinity")
    missing = np.isnan(validation_values)
    if missing.any():
        validation_values[missing] = np.take(
            np.asarray(imputer["statistics"], dtype=np.float64),
            np.nonzero(missing)[1],
        )
    expected_validation = np.ascontiguousarray(validation_values, dtype=np.float32)
    if not np.array_equal(expected_validation, arrays["validation_X.npy"]):
        raise Amendment06IntegrityError("Frozen validation transform differs from locked bytes")
    for lineage_name, expected_rows in (
        ("pre_smote_lineage.txt", PRE_SMOTE_ROWS),
        ("post_smote_lineage.txt", POST_SMOTE_ROWS),
    ):
        lineage_path = destination / lineage_name
        with lineage_path.open("rb") as handle:
            rows = sum(1 for line in handle if line.endswith(b"\n") and line != b"\n")
        if rows != expected_rows:
            raise Amendment06IntegrityError(f"Frozen cache lineage row count differs: {lineage_name}")
    return manifest, imputer


def load_or_build_population_cache(
    *, run_dir: Path, locked: LockedInputs,
    development: DevelopmentPopulation | None,
) -> FrozenPopulationCache:
    destination = EXPECTED_RUNTIME_ROOT / "full_population" / run_dir.name
    published_manifest = run_dir / "provenance/frozen_full_resampling_manifest.json"
    cache_existed = destination.exists()
    amendment08_active = (
        Path(run_dir).resolve().name == A08_TARGET_RUN_ID
        and (run_dir / "provenance/amendment08_overlay_install_receipt.json").exists()
    )
    amendment07_active = (
        not amendment08_active
        and Path(run_dir).resolve() == A07_TARGET_RUN.resolve()
        and (run_dir / "provenance/amendment07_overlay_install_receipt.json").exists()
    )
    if Path(run_dir).resolve().name == A08_TARGET_RUN_ID and not amendment08_active:
        raise Amendment06IntegrityError(
            "The Amendment 08 target cannot load science before its overlay receipt"
        )
    if amendment08_active and (not cache_existed or not published_manifest.exists()):
        raise Amendment06IntegrityError(
            "Amendment 08 forbids cache generation/rematerialization"
        )
    if amendment07_active and (not cache_existed or not published_manifest.exists()):
        raise Amendment06IntegrityError(
            "Amendment 07 forbids cache generation/rematerialization"
        )
    operation = "CACHE_REMATERIALIZATION" if published_manifest.exists() else "CACHE_GENERATION"
    intent_name = f"{operation}_INTENT"
    intents = [
        event for event in _scientific_journal_events(run_dir)
        if event.get("event") == intent_name
    ]
    if len(intents) > 1:
        raise Amendment06IntegrityError(f"Duplicate frozen-cache intent: {intent_name}")
    if not cache_existed:
        if intents:
            intent = intents[0]
            if not (
                intent.get("cache_path") == str(destination)
                and intent.get("base_seed") == BASE_SEED
                and intent.get("primary_table_sha256") == PRIMARY_TABLE_SHA256
                and intent.get("split_assignment_sha256") == SPLIT_ASSIGNMENT_SHA256
            ):
                raise Amendment06IntegrityError("Frozen-cache retry intent differs")
        else:
            _journal_event(
                run_dir, intent_name, status="STARTED", cache_path=str(destination),
                base_seed=BASE_SEED, primary_table_sha256=PRIMARY_TABLE_SHA256,
                split_assignment_sha256=SPLIT_ASSIGNMENT_SHA256,
            )
    if cache_existed:
        if development is None:
            raise Amendment06ResumeError("Development inputs are required to validate the frozen cache")
        if not published_manifest.exists() and not intents:
            raise Amendment06IntegrityError(
                "An unowned first-use frozen population cache already exists"
            )
        manifest, imputer = _validate_population_cache(
            run_dir=run_dir, destination=destination, locked=locked,
            development=development,
        )
        if published_manifest.exists():
            if amendment08_active:
                _ensure_amendment08_cache_load_event(
                    run_dir=run_dir, destination=destination,
                )
            elif amendment07_active:
                _ensure_amendment07_cache_load_event(
                    run_dir=run_dir, destination=destination,
                )
            else:
                _journal_event(
                    run_dir, "CACHE_LOAD", status="PASS", cache_path=str(destination),
                )
    else:
        if development is None:
            raise Amendment06ResumeError("Frozen population cache is absent and development inputs are unavailable")
        manifest, imputer = _build_population_cache(
            run_dir=run_dir, locked=locked, development=development, destination=destination,
        )
        manifest, imputer = _validate_population_cache(
            run_dir=run_dir, destination=destination, locked=locked,
            development=development,
        )
    if published_manifest.exists():
        published = strict_full_load_file(published_manifest)
        if published != manifest:
            raise Amendment06IntegrityError("Runtime cache differs from published frozen-population manifest")
    else:
        _ensure_unique_bound_journal_event(
            run_dir=run_dir, event_name="CACHE_GENERATION",
            binding_path=destination / "cache_manifest.json",
            binding_field="cache_manifest_sha256", recovered=cache_existed,
            extra={"status": "PASS", "cache_path": str(destination)},
        )
        publish_strict_json_no_clobber(published_manifest, manifest)
    rematerialization_intents = [
        event for event in _scientific_journal_events(run_dir)
        if event.get("event") == "CACHE_REMATERIALIZATION_INTENT"
    ]
    if operation == "CACHE_REMATERIALIZATION" and rematerialization_intents:
        _ensure_unique_bound_journal_event(
            run_dir=run_dir, event_name="CACHE_REMATERIALIZATION",
            binding_path=destination / "cache_manifest.json",
            binding_field="cache_manifest_sha256", recovered=cache_existed,
            extra={"status": "PASS", "cache_path": str(destination)},
        )
    imputer_path = run_dir / "provenance/frozen_full_imputer.json"
    if imputer_path.exists():
        if strict_full_load_file(imputer_path) != imputer:
            raise Amendment06IntegrityError("Portable imputer differs from frozen cache")
    else:
        publish_strict_json_no_clobber(imputer_path, imputer)
    arrays = {
        name: np.load(destination / name, mmap_mode="r", allow_pickle=False)
        for name in (
            "pre_smote_X.npy", "pre_smote_y.npy", "pre_smote_weights.npy",
            "post_smote_X.npy", "post_smote_y.npy", "post_smote_weights.npy",
            "validation_X.npy",
        )
    }
    expected_shapes = {
        "pre_smote_X.npy": (PRE_SMOTE_ROWS, 93), "pre_smote_y.npy": (PRE_SMOTE_ROWS,),
        "pre_smote_weights.npy": (PRE_SMOTE_ROWS,), "post_smote_X.npy": (POST_SMOTE_ROWS, 93),
        "post_smote_y.npy": (POST_SMOTE_ROWS,), "post_smote_weights.npy": (POST_SMOTE_ROWS,),
        "validation_X.npy": (56_843, 93),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise Amendment06IntegrityError(f"Frozen cache array shape differs: {name}")
    return FrozenPopulationCache(
        root=destination, manifest=manifest, imputer=imputer,
        full_features=tuple(imputer["feature_order"]),
        pre_X=arrays["pre_smote_X.npy"], pre_y=arrays["pre_smote_y.npy"],
        pre_weights=arrays["pre_smote_weights.npy"], post_X=arrays["post_smote_X.npy"],
        post_y=arrays["post_smote_y.npy"], post_weights=arrays["post_smote_weights.npy"],
        validation_X=arrays["validation_X.npy"],
    )


def _ensure_amendment08_cache_load_event(
    *, run_dir: Path, destination: Path,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    overlay = validate_amendment08_overlay(run_dir)
    cache = _validate_preserved_target_cache_files(run_dir)
    views = read_amendment08_journal_views(run_dir)
    events = views["semantic_events"]
    lineage = {
        "amendment08_authorization": A08_AUTHORIZATION,
        "amendment08_prompt_sha256": A08_PROMPT_SHA256,
        "base_scientific_execution_code_manifest_sha256": overlay[
            "base_scientific_execution_code_manifest_sha256"
        ],
        "effective_scientific_execution_code_manifest_sha256": overlay[
            "effective_scientific_execution_code_manifest_sha256"
        ],
        "amendment08_code_corrigendum_sha256": overlay[
            "amendment08_code_corrigendum_sha256"
        ],
        "amendment08_journal_contamination_adjudication_sha256": overlay[
            "amendment08_journal_contamination_adjudication_sha256"
        ],
    }
    expected = {
        "status": "PASS", "cache_path": str(destination),
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        "frozen_population_sha256": A07_FROZEN_POPULATION_SHA256,
        "unique_realizations": 1, "augmentation_generation_count": 1,
        "safe_smote_generation_count": 1, **lineage,
    }
    matches = [event for event in events if event.get("event") == "CACHE_LOAD"]
    base_keys = {
        "event_index", "event", "run_id", "previous_event_sha256", "recorded_at",
    }
    if matches:
        event = matches[0]
        if not (
            len(matches) == 1 and event.get("event_index") == 8
            and set(event) == base_keys | set(expected)
            and all(event.get(key) == value for key, value in expected.items())
        ):
            raise Amendment06IntegrityError("Amendment 08 CACHE_LOAD event differs")
        return dict(event)
    raw_events = views["raw_events"]
    if not (
        len(raw_events) == 8
        and raw_events[7].get("event") == "AMENDMENT08_CORRIGENDUM_INSTALLED"
        and cache["cache_manifest_sha256"] == A07_CACHE_MANIFEST_SHA256
        and cache["cache_member_count"] == 10
        and cache["unique_realizations"] == 1
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 CACHE_LOAD is not the exact post-corrigendum event 8"
        )
    event = _journal_event(run_dir, "CACHE_LOAD", **expected)
    if not (
        event.get("event_index") == 8
        and set(event) == base_keys | set(expected)
        and all(event.get(key) == value for key, value in expected.items())
    ):
        raise Amendment06IntegrityError("Amendment 08 CACHE_LOAD publication differs")
    return event


def _ensure_amendment07_cache_load_event(
    *, run_dir: Path, destination: Path,
) -> dict[str, Any]:
    overlay = validate_amendment07_overlay(run_dir)
    events = _read_journal(run_dir)
    matches = [event for event in events if event.get("event") == "CACHE_LOAD"]
    expected = {
        "status": "PASS", "cache_path": str(destination),
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        "frozen_population_sha256": A07_FROZEN_POPULATION_SHA256,
        "unique_realizations": 1,
        "augmentation_generation_count": 1,
        "safe_smote_generation_count": 1,
        "effective_scientific_execution_code_manifest_sha256": overlay[
            "effective_scientific_execution_code_manifest_sha256"
        ],
        "amendment07_code_corrigendum_sha256": overlay[
            "amendment07_code_corrigendum_sha256"
        ],
    }
    if matches:
        event = matches[0]
        if len(matches) != 1 or event.get("event_index") != 6 or any(
            event.get(key) != value for key, value in expected.items()
        ):
            raise Amendment06IntegrityError("Amendment 07 CACHE_LOAD event differs")
        return event
    if not (
        len(events) == 6
        and events[5].get("event") == "AMENDMENT07_CORRIGENDUM_INSTALLED"
        and not any(
            event.get("event") in {
                "CACHE_REMATERIALIZATION_INTENT", "CACHE_REMATERIALIZATION",
                "CACHE_GENERATION_INTENT", "CACHE_GENERATION",
            }
            for event in events[2:]
        )
    ):
        raise Amendment06IntegrityError(
            "Amendment 07 CACHE_LOAD is not the exact post-corrigendum event 6"
        )
    event = _journal_event(run_dir, "CACHE_LOAD", **expected)
    if event.get("event_index") != 6:
        raise Amendment06IntegrityError("Amendment 07 CACHE_LOAD index differs")
    return event


def project_variant(cache: FrozenPopulationCache, locked: LockedInputs, variant: str) -> VariantMatrices:
    if variant not in TRAINED_VARIANTS:
        raise Amendment06IntegrityError(f"Unauthorized Full Variant: {variant}")
    features = tuple(locked.variant_features[variant]["retained_features"])
    expected_count, expected_sha = VARIANT_FEATURE_CONTRACT[variant]
    if len(features) != expected_count or _feature_list_sha256(features) != expected_sha:
        raise Amendment06IntegrityError(f"Variant feature contract drifted: {variant}")
    index = {feature: offset for offset, feature in enumerate(cache.full_features)}
    indices = tuple(index[feature] for feature in features)
    if tuple(sorted(indices)) != indices:
        raise Amendment06IntegrityError(f"Variant projection reorders columns: {variant}")
    use_pre = variant == "no_safe_smote"
    source_X = cache.pre_X if use_pre else cache.post_X
    source_y = cache.pre_y if use_pre else cache.post_y
    source_weights = cache.pre_weights if use_pre else cache.post_weights
    X = source_X if indices == tuple(range(93)) else np.ascontiguousarray(source_X[:, indices], dtype=np.float32)
    validation_X = (
        cache.validation_X if indices == tuple(range(93))
        else np.ascontiguousarray(cache.validation_X[:, indices], dtype=np.float32)
    )
    weights = (
        np.ones(len(source_y), dtype=np.float32)
        if variant == "no_review_aware_training_weights" else source_weights
    )
    if not (np.isfinite(X).all() and np.isfinite(weights).all() and np.all(weights > 0)):
        raise Amendment06IntegrityError(f"Variant projection is nonfinite/invalid: {variant}")
    stage = "post_jitter_pre_smote" if use_pre else "post_safe_smote"
    stage_evidence = next(item for item in cache.manifest["stages"] if item["stage"] == stage)
    population_evidence = {
        "variant": variant, "source_stage": stage, "projection_only": True,
        "rows": int(len(source_y)), "feature_count": len(features),
        "feature_list_sha256": expected_sha,
        "full_space_X_sha256": stage_evidence["X_sha256"],
        "y_sha256": stage_evidence["y_sha256"],
        "source_lineage_sha256": stage_evidence["source_lineage_sha256"],
        "row_order_sha256": stage_evidence["row_order_sha256"],
        "weight_sha256": sha256_array(np.asarray(weights, dtype="<f4")),
        "review_weights_enabled": variant != "no_review_aware_training_weights",
        "safe_smote_enabled": not use_pre,
        "safe_smote_synthetic_rows": 0 if use_pre else 147_841,
    }
    return VariantMatrices(
        variant=variant,
        classification=("EXPLORATORY" if variant == EXPLORATORY_VARIANT else "OFFICIAL_RUNNABLE"),
        features=features, indices=indices, X=X, y=source_y, weights=weights,
        validation_X=validation_X, population=stage, population_evidence=population_evidence,
    )


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _isolated_exact_target_journal_pair(run_dir: Path, runtime_root: Path) -> bool:
    common = Path(os.path.commonpath((str(run_dir), str(runtime_root)))).resolve()
    production_results = Path(PRODUCTION_RESULTS_ROOT).resolve()
    production_runtime = Path(PRODUCTION_EXPECTED_RUNTIME_ROOT).resolve()
    forbidden = {
        Path("/").resolve(), Path("/tmp").resolve(), Path.home().resolve(),
        PROJECT_ROOT.resolve(), STUDY_ROOT.resolve(),
    }
    return (
        common not in forbidden
        and not _path_is_within(run_dir, production_results)
        and not _path_is_within(runtime_root, production_runtime)
        and _path_is_within(run_dir, common)
        and _path_is_within(runtime_root, common)
    )


def _journal_path(run_dir: Path, *, for_write: bool = False) -> Path:
    resolved_run = Path(run_dir).resolve()
    runtime_root = Path(EXPECTED_RUNTIME_ROOT).resolve()
    production_runtime = Path(PRODUCTION_EXPECTED_RUNTIME_ROOT).resolve()
    production_results = Path(PRODUCTION_RESULTS_ROOT).resolve()

    if runtime_root == production_runtime:
        if resolved_run.parent != production_results:
            raise Amendment06IntegrityError(
                "Production Journal routing rejects a Run outside canonical RESULTS_ROOT"
            )
        live_target = production_results / A07_TARGET_RUN_ID
        if (
            for_write
            and resolved_run == live_target
            and "PYTEST_CURRENT_TEST" in os.environ
        ):
            raise Amendment06IntegrityError(
                "pytest cannot write the live production target Journal"
            )
    elif (
        resolved_run.name == A07_TARGET_RUN_ID
        and not _isolated_exact_target_journal_pair(resolved_run, runtime_root)
    ):
        raise Amendment06IntegrityError(
            "An exact-target fixture Journal requires one isolated Run/Runtime tree"
        )
    return runtime_root / "journals" / resolved_run.name


def _journal_event_path(run_dir: Path, event_index: int) -> Path:
    return _journal_path(run_dir) / f"{event_index:08d}.json"


def _journal_lock_path(run_dir: Path, *, for_write: bool = False) -> Path:
    journal = _journal_path(run_dir, for_write=for_write)
    return journal.parent / f".{journal.name}.lock"


def _read_journal(run_dir: Path) -> list[dict[str, Any]]:
    path = _journal_path(run_dir)
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_dir():
        raise Amendment06IntegrityError("Execution ownership journal is unsafe")
    members = sorted(path.iterdir(), key=lambda item: item.name)
    authoritative = [item for item in members if re.fullmatch(r"[0-9]{8}\.json", item.name)]
    unexpected = [item for item in members if item not in authoritative]
    if unexpected:
        raise Amendment06IntegrityError("Execution ownership journal has unexpected members")
    events: list[dict[str, Any]] = []
    previous_sha256: str | None = None
    for expected_index, member in enumerate(authoritative):
        if member != _journal_event_path(run_dir, expected_index) or not _safe_regular(member):
            raise Amendment06IntegrityError("Execution ownership journal is noncontiguous/unsafe")
        raw = member.read_bytes()
        value = strict_full_loads(raw)
        if not (
            isinstance(value, dict)
            and raw == strict_full_json_bytes(value)
            and value.get("event_index") == expected_index
            and value.get("run_id") == run_dir.name
            and value.get("previous_event_sha256") == previous_sha256
            and isinstance(value.get("event"), str) and value["event"]
            and isinstance(value.get("recorded_at"), str) and value["recorded_at"]
        ):
            raise Amendment06IntegrityError(
                f"Execution ownership journal is invalid at event {expected_index}"
            )
        events.append(value)
        previous_sha256 = sha256_bytes(strict_full_canonical_json_bytes(value))
    return events


def _journal_event(run_dir: Path, event: str, **fields: Any) -> dict[str, Any]:
    path = _journal_path(run_dir, for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise Amendment06IntegrityError("Execution ownership journal is unsafe")
    lock_path = _journal_lock_path(run_dir, for_write=True)
    descriptor = os.open(
        lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600,
    )
    with os.fdopen(descriptor, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _read_journal(run_dir)
        payload = {
            "event_index": len(existing), "event": event, "run_id": run_dir.name,
            "previous_event_sha256": (
                None if not existing else sha256_bytes(
                    strict_full_canonical_json_bytes(existing[-1])
                )
            ),
            "recorded_at": _now(), **fields,
        }
        publish_bytes_no_clobber(
            _journal_event_path(run_dir, len(existing)), strict_full_json_bytes(payload),
        )
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return payload


def _ensure_unique_bound_journal_event(
    *, run_dir: Path, event_name: str, binding_path: Path,
    binding_field: str, phase: str | None = None,
    variant: str | None = None, recovered: bool,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not _safe_regular(binding_path):
        raise Amendment06IntegrityError(
            f"Journal-bound artifact is absent/unsafe: {binding_path}"
        )
    digest = sha256_file(binding_path)
    matches = [
        item for item in _scientific_journal_events(run_dir)
        if item.get("event") == event_name
        and (phase is None or item.get("phase") == phase)
        and (variant is None or item.get("variant_id") == variant)
    ]
    if len(matches) > 1:
        raise Amendment06IntegrityError(f"Duplicate journal commit event: {event_name}/{variant}")
    if matches:
        observed = matches[0]
        if observed.get(binding_field) != digest:
            raise Amendment06IntegrityError(
                f"Journal commit binding differs: {event_name}/{variant}"
            )
        return observed
    fields: dict[str, Any] = {}
    if phase is not None:
        fields["phase"] = phase
    if variant is not None:
        fields["variant_id"] = variant
    fields.update(dict(extra or {}))
    fields[binding_field] = digest
    fields["recovered_after_interruption"] = recovered
    return _journal_event(run_dir, event_name, **fields)


def _record_interrupted_attempts(
    *, run_dir: Path, phase: str, variant: str,
) -> None:
    names = {
        "TRAINING": ("TRAINING_ATTEMPT_STARTED", "TRAINING_ATTEMPT_FAILED"),
        "AUDIT_SCORING": (
            "AUDIT_SCORING_ATTEMPT_STARTED", "AUDIT_SCORING_ATTEMPT_FAILED",
        ),
        "REPORTING": ("REPORTING_ATTEMPT_STARTED", "REPORTING_ATTEMPT_FAILED"),
    }
    if phase not in names:
        raise Amendment06IntegrityError(f"Unknown interruption phase: {phase}")
    start_name, failed_name = names[phase]
    events = _scientific_journal_events(run_dir)
    failed_ids = {
        str(event.get("attempt_id")) for event in events
        if event.get("event") == failed_name and event.get("phase") == phase
        and event.get("variant_id") == variant
    }
    committed_ids = {
        str(event.get("attempt_id")) for event in events
        if event.get("event") in {
            "TRAINING_COMPLETION_COMMITTED", "AUDIT_SCORING_COMPLETION_COMMITTED",
            "REPORTING_COMPLETION_COMMITTED",
        }
        and event.get("phase") == phase and event.get("variant_id") == variant
        and event.get("attempt_id") is not None
    }
    for event in events:
        attempt_id = str(event.get("attempt_id", ""))
        if (
            event.get("event") == start_name
            and event.get("phase") == phase and event.get("variant_id") == variant
            and attempt_id and attempt_id not in failed_ids | committed_ids
        ):
            _journal_event(
                run_dir, failed_name, phase=phase, variant_id=variant,
                attempt_id=attempt_id,
                failure_phase="PROCESS_INTERRUPTION_BEFORE_COMPLETION_COMMIT",
                error_type="InterruptedProcess",
                error="Recovered an unterminated attempt before same-Run retry",
                recovered_after_interruption=True,
            )


_SCIENTIFIC_COUNTERS: MutableMapping[str, int] = {
    "training_calls": 0,
    "augmentation_calls": 0,
    "safe_smote_calls": 0,
    "imputer_fit_calls": 0,
    "post_audit_training_calls": 0,
    "threshold_tuning_calls": 0,
    "stability_fits": 0,
}
_POST_FREEZE_GUARDS_INSTALLED = False


class BoundedFitProgress:
    """Emit bounded resource progress without interacting with model state."""

    def __init__(
        self, run_id: str, variant: str, *, interval_seconds: float = 60.0,
        snapshot: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("Fit progress interval must be positive")
        self.run_id = run_id
        self.variant = variant
        self.interval_seconds = float(interval_seconds)
        self.snapshot = snapshot
        self._stop = threading.Event()
        self._started = time.monotonic()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _resources(self) -> Mapping[str, Any]:
        if self.snapshot is not None:
            return self.snapshot()
        return {
            "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "gpu_memory_mib": core._gpu_used_memory_mib(),
        }

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                values = self._resources()
                _progress(
                    "FULL_VARIANT_FIT", "ACTIVE", run_id=self.run_id,
                    variant=self.variant,
                    elapsed_seconds=f"{time.monotonic() - self._started:.1f}",
                    rss_bytes=values.get("rss_bytes"),
                    gpu_memory_mib=values.get("gpu_memory_mib"),
                )
            except Exception as exc:
                _progress(
                    "FULL_VARIANT_FIT_MONITOR", "WARNING", variant=self.variant,
                    error_type=type(exc).__name__,
                )

    def __enter__(self) -> "BoundedFitProgress":
        self._started = time.monotonic()
        self._thread = threading.Thread(
            target=self._loop, name=f"a06-fit-progress-{self.variant}", daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        if self.running:
            raise Amendment06IntegrityError("Fit progress monitor did not stop")


def _binary_metrics(
    labels: Sequence[int], probabilities: Sequence[float], *,
    threshold: float = DECISION_THRESHOLD, weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    result = dict(reporting.binary_metrics(
        labels, probabilities, threshold=threshold, weights=weights,
    ))
    assert_required_metrics_finite(result)
    return result


PREDICTION_COLUMNS = reporting.PREDICTION_COLUMNS


def deterministic_csv_gzip_bytes(frame: pd.DataFrame) -> bytes:
    if list(frame.columns) != list(PREDICTION_COLUMNS):
        raise Amendment06IntegrityError("Prediction table schema/order differs")
    return reporting.deterministic_gzip_bytes(reporting.prediction_csv_bytes(frame))


def validate_prediction_roundtrip(
    data_or_path: bytes | Path, authoritative: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(data_or_path, Path):
        reopened = pd.read_csv(data_or_path, engine="c", float_precision="round_trip")
    else:
        reopened = pd.read_csv(
            io.BytesIO(data_or_path), compression="gzip", engine="c",
            float_precision="round_trip",
        )
    if list(reopened.columns) != list(PREDICTION_COLUMNS) or len(reopened) != len(authoritative):
        raise Amendment06IntegrityError("Prediction CSV roundtrip schema/row count differs")
    text_columns = ("stable_candidate_id", "card_id", "variant_id", "split_population")
    integer_columns = ("true_label", "prediction_at_0_5", "row_order")
    numeric_columns = ("probability", "review_weight")
    mismatches: dict[str, int] = {}
    for column in text_columns:
        count = int((reopened[column].astype(str).to_numpy() != authoritative[column].astype(str).to_numpy()).sum())
        mismatches[column] = count
    for column in integer_columns:
        left = pd.to_numeric(reopened[column], errors="raise").to_numpy(np.int64)
        right = pd.to_numeric(authoritative[column], errors="raise").to_numpy(np.int64)
        mismatches[column] = int((left != right).sum())
    for column in numeric_columns:
        left = pd.to_numeric(reopened[column], errors="coerce").to_numpy(np.float64)
        right = pd.to_numeric(authoritative[column], errors="coerce").to_numpy(np.float64)
        same_missing = np.array_equal(np.isnan(left), np.isnan(right))
        finite = np.isfinite(left) & np.isfinite(right)
        mismatches[column] = (
            int((left[finite] != right[finite]).sum())
            if same_missing else len(left)
        )
    total = sum(mismatches.values())
    if total:
        raise Amendment06IntegrityError(f"Prediction CSV exact roundtrip failed: {mismatches}")
    reporting_evidence = reporting.validate_prediction_csv_roundtrip(
        data_or_path, authoritative, compressed=True,
    )
    return reopened, {
        **reporting_evidence,
        "rows": len(reopened), "mismatch_count": 0,
        "column_mismatch_counts": mismatches,
    }


def exact_table_csv_bytes(frame: pd.DataFrame) -> tuple[bytes, dict[str, Any]]:
    data = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    reopened = pd.read_csv(
        io.BytesIO(data), engine="c", float_precision="round_trip",
    )
    if tuple(reopened.columns) != tuple(frame.columns) or len(reopened) != len(frame):
        raise Amendment06IntegrityError("Table CSV exact roundtrip shape/schema differs")
    for column in frame.columns:
        expected = frame[column]
        observed = reopened[column]
        expected_missing = pd.isna(expected).to_numpy()
        observed_missing = pd.isna(observed).to_numpy()
        if not np.array_equal(expected_missing, observed_missing):
            raise Amendment06IntegrityError(
                f"Table CSV missingness mask differs: {column}"
            )
        finite = ~expected_missing
        if pd.api.types.is_numeric_dtype(expected.dtype):
            left = pd.to_numeric(expected[finite], errors="raise").to_numpy(np.float64)
            right = pd.to_numeric(observed[finite], errors="raise").to_numpy(np.float64)
            if not np.array_equal(left, right):
                raise Amendment06IntegrityError(
                    f"Table CSV numeric roundtrip differs: {column}"
                )
        elif not np.array_equal(
            expected[finite].astype(str).to_numpy(),
            observed[finite].astype(str).to_numpy(),
        ):
            raise Amendment06IntegrityError(
                f"Table CSV text roundtrip differs: {column}"
            )
    return data, {
        "status": "PASS", "rows": len(frame), "columns": list(frame.columns),
        "engine": "c", "float_precision": "round_trip",
        "numeric_equality": "EXACT_FINITE_WITH_IDENTICAL_MISSINGNESS",
        "csv_sha256": sha256_bytes(data),
    }


def validate_exact_table_csv(
    path: Path, expected: pd.DataFrame,
) -> dict[str, Any]:
    expected_bytes, evidence = exact_table_csv_bytes(expected)
    if not _safe_regular(path, sha256_bytes(expected_bytes)):
        raise Amendment06IntegrityError(f"Table CSV bytes differ: {path}")
    return evidence


def _attempt_root(run_dir: Path, phase: str, variant: str, attempt_id: str) -> Path:
    return EXPECTED_RUNTIME_ROOT / "attempts" / run_dir.name / phase / variant / attempt_id


def _safe_run_relative(relative: str) -> str:
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise Amendment06IntegrityError(f"Unsafe Run-relative artifact path: {relative!r}")
    return relative


def _stage_bytes(staging: Path, relative: str, data: bytes) -> Path:
    path = staging / _safe_run_relative(relative)
    publish_bytes_no_clobber(path, data)
    return path


def _stage_json(staging: Path, relative: str, value: Any) -> Path:
    return _stage_bytes(staging, relative, strict_full_json_bytes(value))


def _publication_record(path: Path, relative: str) -> dict[str, Any]:
    return {
        "relative_path": relative, "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_exact_artifact_records(
    *, run_dir: Path, value: Any, expected_relatives: Sequence[str], label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(expected_relatives):
        raise Amendment06IntegrityError(f"{label} artifact record count differs")
    records: dict[str, dict[str, Any]] = {}
    observed: list[str] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "relative_path", "size_bytes", "sha256",
        }:
            raise Amendment06IntegrityError(f"{label} artifact record schema differs")
        relative = str(item["relative_path"])
        observed.append(relative)
        if relative in records:
            raise Amendment06IntegrityError(f"{label} artifact records are duplicated")
        member = run_dir / _safe_run_relative(relative)
        size = item["size_bytes"]
        digest = str(item["sha256"])
        if (
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not _safe_regular(member, digest) or member.stat().st_size != size
        ):
            raise Amendment06IntegrityError(f"{label} artifact drifted: {relative}")
        records[relative] = dict(item)
    if observed != list(expected_relatives):
        raise Amendment06IntegrityError(f"{label} artifact member set/order differs")
    return records


def _publish_attempt_file(
    *, run_dir: Path, staging: Path, relative: str, phase: str,
    variant: str, attempt_id: str,
) -> dict[str, Any]:
    relative = _safe_run_relative(relative)
    source = staging / relative
    destination = run_dir / relative
    if not _safe_regular(source):
        raise Amendment06IntegrityError(f"Attempt source is absent/unsafe: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise Amendment06IntegrityError(f"No-clobber scientific publication collision: {relative}")
    source_record = _publication_record(source, relative)
    _journal_event(
        run_dir, "ARTIFACT_PUBLICATION_INTENT", phase=phase, variant_id=variant,
        attempt_id=attempt_id, **source_record,
    )
    # Move, rather than hardlink, so a frozen Run inode has no writable alias in
    # the attempt cache after publication.  The pre-move INTENT makes a crash
    # after renameat2 but before PUBLISHED recoverable by exact hash ownership.
    _rename_directory_no_clobber(source, destination)
    record = _publication_record(destination, relative)
    _journal_event(
        run_dir, "ARTIFACT_PUBLISHED", phase=phase, variant_id=variant,
        attempt_id=attempt_id, **record,
    )
    return record


def _phase_publication_records(
    run_dir: Path, phase: str, variant: str,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for event in _scientific_journal_events(run_dir):
        if (
            event.get("event") in {"ARTIFACT_PUBLICATION_INTENT", "ARTIFACT_PUBLISHED"}
            and event.get("phase") == phase and event.get("variant_id") == variant
        ):
            records[str(event["relative_path"])] = event
    return records


def _recover_unpublished_paths(
    *, run_dir: Path, phase: str, variant: str,
    completion_relative: str, possible_relatives: Sequence[str],
) -> None:
    completion = run_dir / completion_relative
    if completion.exists():
        return
    events = _scientific_journal_events(run_dir)
    owned = _phase_publication_records(run_dir, phase, variant)
    committed_transactions = {
        str(event.get("transaction_id")) for event in events
        if event.get("event") == "ARTIFACT_QUARANTINE_COMMITTED"
    }
    pending = [
        event for event in events
        if event.get("event") == "ARTIFACT_QUARANTINE_INTENT"
        and event.get("phase") == phase and event.get("variant_id") == variant
        and str(event.get("transaction_id")) not in committed_transactions
    ]

    def reconcile(intent: Mapping[str, Any], *, recovered: bool) -> None:
        relative = _safe_run_relative(str(intent.get("relative_path")))
        transaction_id = str(intent.get("transaction_id", ""))
        source = run_dir / relative
        destination = Path(str(intent.get("quarantine_path", "")))
        expected_root = EXPECTED_RUNTIME_ROOT / "quarantine" / run_dir.name / phase / variant
        if not (
            re.fullmatch(r"[0-9a-f]{32}", transaction_id)
            and expected_root.resolve() in destination.resolve().parents
            and re.fullmatch(r"[0-9a-f]{64}", str(intent.get("sha256", "")))
            and isinstance(intent.get("size_bytes"), int)
            and not isinstance(intent.get("size_bytes"), bool)
            and int(intent["size_bytes"]) >= 0
        ):
            raise Amendment06IntegrityError("Quarantine intent schema/path differs")
        digest = str(intent["sha256"])
        size = int(intent["size_bytes"])
        source_valid = _safe_regular(source, digest) and source.stat().st_size == size
        destination_valid = (
            _safe_regular(destination, digest) and destination.stat().st_size == size
        )
        if source_valid and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            _rename_directory_no_clobber(source, destination)
            _fsync_parent(source)
            destination_valid = (
                _safe_regular(destination, digest) and destination.stat().st_size == size
            )
        if source.exists() or not destination_valid:
            raise Amendment06IntegrityError(
                f"Quarantine transaction cannot be reconciled: {relative}"
            )
        _journal_event(
            run_dir, "ARTIFACT_QUARANTINE_COMMITTED", phase=phase,
            variant_id=variant, attempt_id=intent.get("attempt_id"),
            transaction_id=transaction_id, relative_path=relative,
            sha256=digest, size_bytes=size, quarantine_path=str(destination),
            recovered_after_interruption=recovered,
        )

    for intent in pending:
        reconcile(intent, recovered=True)

    existing = [relative for relative in possible_relatives if (run_dir / relative).exists()]
    for relative in existing:
        relative = _safe_run_relative(str(relative))
        source = run_dir / relative
        record = owned.get(relative)
        if not (
            record and _safe_regular(source, str(record.get("sha256")))
            and source.stat().st_size == int(record.get("size_bytes", -1))
        ):
            raise Amendment06IntegrityError(
                f"Unpublished artifact lacks exact ownership proof: {relative}"
            )
        transaction_id = uuid.uuid4().hex
        destination = (
            EXPECTED_RUNTIME_ROOT / "quarantine" / run_dir.name / phase / variant
            / transaction_id / relative
        )
        intent = _journal_event(
            run_dir, "ARTIFACT_QUARANTINE_INTENT", phase=phase,
            variant_id=variant, attempt_id=record.get("attempt_id"),
            transaction_id=transaction_id, relative_path=relative,
            sha256=record["sha256"], size_bytes=record["size_bytes"],
            quarantine_path=str(destination),
        )
        reconcile(intent, recovered=False)


def _training_relatives(variant: str) -> tuple[str, ...]:
    return (
        f"models/{variant}/model.ubj",
        f"models/{variant}/feature_list.txt",
        f"models/{variant}/model_metadata.json",
        f"models/{variant}/transform_binding.json",
        f"predictions/{variant}_validation.csv.gz",
        f"timing/{variant}.json",
        f"logs/{variant}.stdout.log",
        f"logs/{variant}.stderr.log",
    )


def _training_completion_relative(variant: str) -> str:
    return f"models/{variant}/training_completion_manifest.json"


def _fallback_warnings(messages: Sequence[str]) -> list[str]:
    tokens = ("fallback", "not compiled with gpu", "mismatched devices", "cpu predictor")
    return [message for message in messages if any(token in message.lower() for token in tokens)]


def _gpu_name(identity: Mapping[str, Any]) -> str:
    if not isinstance(identity, Mapping):
        raise Amendment06IntegrityError("GPU identity is not a Mapping")
    devices = identity.get("devices")
    if (
        not isinstance(devices, Sequence)
        or isinstance(devices, (str, bytes, bytearray))
    ):
        raise Amendment06IntegrityError("GPU identity devices is not a real sequence")
    if len(devices) != 1:
        raise Amendment06IntegrityError(
            f"GPU identity must contain exactly one device, observed {len(devices)}"
        )
    device = devices[0]
    if not isinstance(device, Mapping):
        raise Amendment06IntegrityError("GPU identity device record is not a Mapping")

    nested_name = device.get("gpu_name")
    if not isinstance(nested_name, str) or not nested_name.strip():
        raise Amendment06IntegrityError("GPU identity devices[0].gpu_name is missing/blank")
    if nested_name != nested_name.strip():
        raise Amendment06IntegrityError(
            "GPU identity devices[0].gpu_name is not canonical"
        )

    def require_duplicate(owner: Mapping[str, Any], key: str, *, label: str) -> None:
        if key not in owner:
            return
        value = owner[key]
        if not isinstance(value, str) or not value.strip():
            raise Amendment06IntegrityError(f"GPU identity {label} is not a nonempty string")
        if value != nested_name:
            raise Amendment06IntegrityError(f"GPU identity {label} conflicts with nested gpu_name")

    require_duplicate(device, "name", label="devices[0].name")
    require_duplicate(identity, "gpu_name", label="top-level gpu_name")
    require_duplicate(identity, "name", label="top-level name")
    if "visible_gpu_count_reported_by_nvidia_smi" in identity:
        count = identity["visible_gpu_count_reported_by_nvidia_smi"]
        if isinstance(count, bool) or not isinstance(count, int):
            raise Amendment06IntegrityError("GPU identity visible-device count is not an integer")
        if count != 1 or count != len(devices):
            raise Amendment06IntegrityError("GPU identity visible-device count disagrees")
    return nested_name


def _training_prediction_frame(
    *, development: DevelopmentPopulation, probabilities: np.ndarray,
    variant: str, population: str,
) -> pd.DataFrame:
    if len(probabilities) != len(development.validation):
        raise Amendment06IntegrityError("Validation probability row count differs")
    return pd.DataFrame({
        "stable_candidate_id": development.validation.stable_candidate_id.astype(str).to_numpy(),
        "card_id": development.validation.card_id.astype(str).to_numpy(),
        "true_label": development.validation_labels,
        # Promote float32 predictions exactly so pandas emits enough decimal
        # digits for C-engine round_trip to recover the scientific value.
        "probability": np.asarray(probabilities, dtype=np.float32).astype(np.float64),
        "prediction_at_0_5": (np.asarray(probabilities) >= DECISION_THRESHOLD).astype(np.int8),
        "review_weight": development.validation_weights.astype(np.float64),
        "variant_id": variant,
        "split_population": f"validation:{population}",
        "row_order": np.arange(len(probabilities), dtype=np.int64),
    }, columns=PREDICTION_COLUMNS)


def _model_common_metadata(
    *, run_dir: Path, variant_data: VariantMatrices, model_path: Path,
    cache: FrozenPopulationCache, gpu_identity: Mapping[str, Any], gpu_name: str,
    best_iteration: int, iteration_range: tuple[int, int],
) -> dict[str, Any]:
    lineage = _scientific_lineage_bindings(run_dir)
    return {
        "run_id": run_dir.name, "run_kind": RUN_KIND,
        "variant_id": variant_data.variant,
        "variant_classification": variant_data.classification,
        "model_sha256": sha256_file(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "feature_count": len(variant_data.features),
        "feature_list_sha256": _feature_list_sha256(variant_data.features),
        "full_parameter_lock_sha256": FULL_MODEL_PARAMETERS_SHA256,
        "frozen_population_sha256": cache.binding_sha256,
        "imputer_sha256": cache.imputer_sha256,
        "split_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
        "primary_table_sha256": PRIMARY_TABLE_SHA256,
        "source_input_hashes_pre_sha256": sha256_file(
            run_dir / "provenance/source_input_hashes_pre.tsv"
        ),
        "training_configuration_sha256": TRAINING_CONFIGURATION_SHA256,
        "variant_feature_sets_sha256": VARIANT_FEATURE_SETS_SHA256,
        "row_split_manifest_sha256": ROW_SPLIT_MANIFEST_SHA256,
        **lineage,
        "scientific_execution_code_manifest_sha256": lineage[
            "effective_scientific_execution_code_manifest_sha256"
        ],
        "seed": PRIMARY_MODEL_SEED,
        "decision_threshold": DECISION_THRESHOLD,
        "gpu_name": gpu_name,
        "gpu_identity": dict(gpu_identity),
        "best_iteration": int(best_iteration),
        "iteration_range": [int(iteration_range[0]), int(iteration_range[1])],
    }


def _amendment07_training_event_fields(
    *, run_dir: Path, variant: str,
) -> dict[str, Any]:
    if not (
        Path(run_dir).resolve() == A07_TARGET_RUN.resolve()
        and (run_dir / "provenance/amendment07_overlay_install_receipt.json").exists()
    ):
        return {}
    overlay = validate_amendment07_overlay(run_dir)
    lineage = {
        "amendment07_authorization": A07_AUTHORIZATION,
        "amendment07_prompt_sha256": A07_PROMPT_SHA256,
        "base_scientific_execution_code_manifest_sha256": overlay[
            "base_scientific_execution_code_manifest_sha256"
        ],
        "effective_scientific_execution_code_manifest_sha256": overlay[
            "effective_scientific_execution_code_manifest_sha256"
        ],
        "amendment07_code_corrigendum_sha256": overlay[
            "amendment07_code_corrigendum_sha256"
        ],
    }
    history = _validate_amendment07_corrected_training_history(
        run_dir=run_dir, overlay=overlay, allow_unterminated=False,
    )
    if history["next_variant"] != variant:
        raise Amendment06IntegrityError(
            "Amendment 07 corrected Variant order differs before fit"
        )
    return lineage


def _validate_amendment07_corrected_training_history(
    *, run_dir: Path, overlay: Mapping[str, Any] | None = None,
    allow_unterminated: bool,
) -> dict[str, Any]:
    overlay = dict(overlay or validate_amendment07_overlay(run_dir))
    lineage = {
        "amendment07_authorization": A07_AUTHORIZATION,
        "amendment07_prompt_sha256": A07_PROMPT_SHA256,
        "base_scientific_execution_code_manifest_sha256": overlay[
            "base_scientific_execution_code_manifest_sha256"
        ],
        "effective_scientific_execution_code_manifest_sha256": overlay[
            "effective_scientific_execution_code_manifest_sha256"
        ],
        "amendment07_code_corrigendum_sha256": overlay[
            "amendment07_code_corrigendum_sha256"
        ],
    }
    events = _read_journal(run_dir)
    cache_path = str(EXPECTED_RUNTIME_ROOT / "full_population" / run_dir.name)
    expected_cache = {
        "status": "PASS", "cache_path": cache_path,
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        "frozen_population_sha256": A07_FROZEN_POPULATION_SHA256,
        "unique_realizations": 1, "augmentation_generation_count": 1,
        "safe_smote_generation_count": 1,
        "effective_scientific_execution_code_manifest_sha256": lineage[
            "effective_scientific_execution_code_manifest_sha256"
        ],
        "amendment07_code_corrigendum_sha256": lineage[
            "amendment07_code_corrigendum_sha256"
        ],
    }
    journal_base_keys = {
        "event_index", "event", "run_id", "previous_event_sha256", "recorded_at",
    }
    expected_install = {
        "status": "PASS", "authorization": A07_AUTHORIZATION,
        "prompt_sha256": A07_PROMPT_SHA256,
        "effective_scientific_execution_code_manifest_sha256": lineage[
            "effective_scientific_execution_code_manifest_sha256"
        ],
        "amendment07_code_corrigendum_sha256": lineage[
            "amendment07_code_corrigendum_sha256"
        ],
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
    }
    if not (
        len(events) >= 7
        and events[4].get("event") == "TRAINING_ATTEMPT_FAILED"
        and events[4].get("attempt_id") == A07_FAILED_ATTEMPT_ID
        and events[5].get("event") == "AMENDMENT07_CORRIGENDUM_INSTALLED"
        and events[5].get("event_index") == 5
        and set(events[5]) == journal_base_keys | set(expected_install)
        and all(events[5].get(key) == value for key, value in expected_install.items())
        and events[6].get("event") == "CACHE_LOAD"
        and events[6].get("event_index") == 6
        and set(events[6]) == journal_base_keys | set(expected_cache)
        and all(events[6].get(key) == value for key, value in expected_cache.items())
        and sum(event.get("event") == "CACHE_LOAD" for event in events) == 1
    ):
        raise Amendment06IntegrityError(
            "Amendment 07 corrected training history lacks exact CACHE_LOAD event 6"
        )

    start_keys = {
        *journal_base_keys,
        "phase", "variant_id", "attempt_id", *lineage,
    }
    fit_keys = {
        *start_keys, "seed", "stability_fit",
    }
    failed_keys = {
        *journal_base_keys,
        "phase", "variant_id", "attempt_id", "error_type", "error",
    }
    attempts: dict[str, dict[str, Any]] = {}
    completed_variants: list[str] = []
    for index, event in enumerate(events[7:], start=7):
        name = event.get("event")
        attempt_id = str(event.get("attempt_id", ""))
        if name == "TRAINING_ATTEMPT_STARTED":
            next_variant = next(
                (item for item in TRAINED_VARIANTS if item not in completed_variants),
                None,
            )
            if (
                set(event) != start_keys
                or event.get("event_index") != index
                or event.get("phase") != "TRAINING"
                or event.get("variant_id") != next_variant
                or not re.fullmatch(r"[0-9a-f]{32}", attempt_id)
                or attempt_id in attempts
                or any(item["terminal"] is None for item in attempts.values())
                or any(event.get(key) != value for key, value in lineage.items())
            ):
                raise Amendment06IntegrityError(
                    "Amendment 07 corrected training-attempt start differs"
                )
            attempts[attempt_id] = {
                "variant_id": str(next_variant), "start_index": index,
                "fit_index": None, "terminal": None,
            }
            if len(attempts) == 1 and not (
                index == 7 and next_variant == "full_new_reference"
            ):
                raise Amendment06IntegrityError(
                    "First corrected training attempt is not exact Journal event 7"
                )
        elif name == "SCIENTIFIC_FIT_CALLED" and index > 6:
            state = attempts.get(attempt_id)
            if (
                set(event) != fit_keys or state is None
                or state["terminal"] is not None or state["fit_index"] is not None
                or event.get("phase") != "TRAINING"
                or event.get("variant_id") != state["variant_id"]
                or event.get("seed") != PRIMARY_MODEL_SEED
                or event.get("stability_fit") is not False
                or any(event.get(key) != value for key, value in lineage.items())
            ):
                raise Amendment06IntegrityError(
                    "Amendment 07 corrected scientific-fit event differs"
                )
            state["fit_index"] = index
        elif name == "TRAINING_ATTEMPT_FAILED":
            state = attempts.get(attempt_id)
            observed_keys = set(event)
            failure_phase = event.get("failure_phase")
            recovered = event.get("recovered_after_interruption")
            schema_is_valid = (
                observed_keys == failed_keys
                and failure_phase is None and recovered is None
            ) or (
                observed_keys == failed_keys | {"failure_phase"}
                and failure_phase == "POST_START_THROUGH_COMPLETION_COMMIT"
                and recovered is None
            ) or (
                observed_keys
                == failed_keys | {"failure_phase", "recovered_after_interruption"}
                and failure_phase == "PROCESS_INTERRUPTION_BEFORE_COMPLETION_COMMIT"
                and recovered is True
            )
            if (
                state is None or not schema_is_valid
                or state["terminal"] is not None
                or event.get("phase") != "TRAINING"
                or event.get("variant_id") != state["variant_id"]
                or index <= int(state["start_index"])
                or not isinstance(event.get("error_type"), str)
                or not str(event.get("error_type", "")).strip()
                or not isinstance(event.get("error"), str)
                or not str(event.get("error", "")).strip()
            ):
                raise Amendment06IntegrityError(
                    "Amendment 07 corrected failed-attempt lifecycle differs"
                )
            state["terminal"] = "FAILED"
        elif name == "TRAINING_COMPLETION_COMMITTED":
            state = attempts.get(attempt_id)
            if (
                state is None or state["terminal"] is not None
                or state["fit_index"] is None
                or event.get("phase") != "TRAINING"
                or event.get("variant_id") != state["variant_id"]
                or state["variant_id"] in completed_variants
            ):
                raise Amendment06IntegrityError(
                    "Amendment 07 corrected completion lifecycle differs"
                )
            state["terminal"] = "COMMITTED"
            completed_variants.append(str(state["variant_id"]))

    unterminated = [
        attempt_id for attempt_id, state in attempts.items()
        if state["terminal"] is None
    ]
    if (
        (not attempts and len(events) != 7)
        or
        len(unterminated) > 1
        or (unterminated and not allow_unterminated)
        or completed_variants != list(TRAINED_VARIANTS[:len(completed_variants)])
    ):
        raise Amendment06IntegrityError(
            "Amendment 07 corrected attempt terminal/order history differs"
        )
    next_variant = next(
        (item for item in TRAINED_VARIANTS if item not in completed_variants), None,
    )
    return {
        "status": "PASS", "corrected_attempt_count": len(attempts),
        "completed_variants": completed_variants,
        "unterminated_attempt_ids": unterminated,
        "next_variant": next_variant,
    }


def _amendment08_training_lineage(
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "amendment08_authorization": A08_AUTHORIZATION,
        "amendment08_prompt_sha256": A08_PROMPT_SHA256,
        "base_scientific_execution_code_manifest_sha256": overlay[
            "base_scientific_execution_code_manifest_sha256"
        ],
        "effective_scientific_execution_code_manifest_sha256": overlay[
            "effective_scientific_execution_code_manifest_sha256"
        ],
        "amendment08_code_corrigendum_sha256": overlay[
            "amendment08_code_corrigendum_sha256"
        ],
        "amendment08_journal_contamination_adjudication_sha256": overlay[
            "amendment08_journal_contamination_adjudication_sha256"
        ],
    }


def _validate_amendment08_corrected_training_history(
    *, run_dir: Path, overlay: Mapping[str, Any] | None = None,
    allow_unterminated: bool,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    overlay = dict(overlay or validate_amendment08_overlay(run_dir))
    lineage = _amendment08_training_lineage(overlay)
    views = read_amendment08_journal_views(run_dir)
    raw_events = views["raw_events"]
    events = views["semantic_events"]
    if not (
        len(raw_events) >= 9
        and raw_events[4] == _amendment08_expected_raw_prefix_events()[4]
        and raw_events[7].get("event") == "AMENDMENT08_CORRIGENDUM_INSTALLED"
        and raw_events[7].get("event_index") == 7
        and raw_events[8].get("event") == "CACHE_LOAD"
        and raw_events[8].get("event_index") == 8
        and all(
            not (
                event.get("event") == "AMENDMENT07_CORRIGENDUM_INSTALLED"
                or event.get("amendment07_authorization") == A07_AUTHORIZATION
            )
            for event in raw_events
        )
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 corrected history lacks exact event 7 -> event 8"
        )
    base_keys = {
        "event_index", "event", "run_id", "previous_event_sha256", "recorded_at",
    }
    expected_cache = {
        "status": "PASS",
        "cache_path": str(EXPECTED_RUNTIME_ROOT / "full_population" / run_dir.name),
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        "frozen_population_sha256": A07_FROZEN_POPULATION_SHA256,
        "unique_realizations": 1, "augmentation_generation_count": 1,
        "safe_smote_generation_count": 1, **lineage,
    }
    cache_event = raw_events[8]
    if not (
        set(cache_event) == base_keys | set(expected_cache)
        and all(cache_event.get(key) == value for key, value in expected_cache.items())
        and sum(event.get("event") == "CACHE_LOAD" for event in events) == 1
    ):
        raise Amendment06IntegrityError("Amendment 08 exact CACHE_LOAD event 8 differs")

    start_keys = {*base_keys, "phase", "variant_id", "attempt_id", *lineage}
    fit_keys = {*start_keys, "seed", "stability_fit"}
    failure_base_keys = {
        *base_keys, "phase", "variant_id", "attempt_id", "error_type", "error",
    }
    completion_keys = {
        *base_keys, "phase", "variant_id", "attempt_id", "completion_sha256",
        "recovered_after_interruption",
    }
    attempts: dict[str, dict[str, Any]] = {}
    completed_variants: list[str] = []
    first_start_index: int | None = None
    for event in events:
        physical_index = event.get("event_index")
        if isinstance(physical_index, bool) or not isinstance(physical_index, int):
            raise Amendment06IntegrityError("Amendment 08 semantic physical index differs")
        if physical_index <= 8:
            continue
        name = event.get("event")
        attempt_id = str(event.get("attempt_id", ""))
        if name == "TRAINING_ATTEMPT_STARTED":
            next_variant = next(
                (variant for variant in TRAINED_VARIANTS if variant not in completed_variants),
                None,
            )
            if (
                set(event) != start_keys
                or event.get("phase") != "TRAINING"
                or event.get("variant_id") != next_variant
                or not re.fullmatch(r"[0-9a-f]{32}", attempt_id)
                or attempt_id in {A07_FAILED_ATTEMPT_ID, "a" * 32}
                or attempt_id in attempts
                or any(state["terminal"] is None for state in attempts.values())
                or any(event.get(key) != value for key, value in lineage.items())
            ):
                raise Amendment06IntegrityError(
                    "Amendment 08 corrected training-attempt start differs"
                )
            attempts[attempt_id] = {
                "variant_id": str(next_variant), "start_index": physical_index,
                "fit_index": None, "terminal": None,
            }
            if first_start_index is None:
                first_start_index = physical_index
                if physical_index != 9 or next_variant != "full_new_reference":
                    raise Amendment06IntegrityError(
                        "First Amendment 08 corrected attempt is not physical event 9"
                    )
        elif name == "SCIENTIFIC_FIT_CALLED":
            state = attempts.get(attempt_id)
            if (
                set(event) != fit_keys or state is None
                or state["terminal"] is not None or state["fit_index"] is not None
                or event.get("phase") != "TRAINING"
                or event.get("variant_id") != state["variant_id"]
                or event.get("seed") != PRIMARY_MODEL_SEED
                or event.get("stability_fit") is not False
                or physical_index <= int(state["start_index"])
                or any(event.get(key) != value for key, value in lineage.items())
            ):
                raise Amendment06IntegrityError(
                    "Amendment 08 corrected scientific-fit event differs"
                )
            state["fit_index"] = physical_index
        elif name == "TRAINING_ATTEMPT_FAILED":
            state = attempts.get(attempt_id)
            observed_keys = set(event)
            failure_phase = event.get("failure_phase")
            recovered = event.get("recovered_after_interruption")
            schema_valid = (
                observed_keys == failure_base_keys
                and failure_phase is None and recovered is None
            ) or (
                observed_keys == failure_base_keys | {"failure_phase"}
                and failure_phase == "POST_START_THROUGH_COMPLETION_COMMIT"
                and recovered is None
            ) or (
                observed_keys
                == failure_base_keys | {"failure_phase", "recovered_after_interruption"}
                and failure_phase == "PROCESS_INTERRUPTION_BEFORE_COMPLETION_COMMIT"
                and recovered is True
            )
            if (
                state is None or not schema_valid or state["terminal"] is not None
                or event.get("phase") != "TRAINING"
                or event.get("variant_id") != state["variant_id"]
                or physical_index <= int(state["start_index"])
                or not isinstance(event.get("error_type"), str)
                or not str(event.get("error_type", "")).strip()
                or not isinstance(event.get("error"), str)
                or not str(event.get("error", "")).strip()
            ):
                raise Amendment06IntegrityError(
                    "Amendment 08 corrected failed-attempt lifecycle differs"
                )
            state["terminal"] = "FAILED"
        elif name == "TRAINING_COMPLETION_COMMITTED":
            state = attempts.get(attempt_id)
            variant = str(event.get("variant_id", ""))
            completion_path = run_dir / _training_completion_relative(variant)
            if (
                set(event) != completion_keys or state is None
                or state["terminal"] is not None or state["fit_index"] is None
                or event.get("phase") != "TRAINING"
                or variant != state["variant_id"] or variant in completed_variants
                or physical_index <= int(state["fit_index"])
                or not _safe_regular(completion_path, str(event.get("completion_sha256", "")))
                or not isinstance(event.get("recovered_after_interruption"), bool)
            ):
                raise Amendment06IntegrityError(
                    "Amendment 08 corrected completion lifecycle differs"
                )
            state["terminal"] = "COMMITTED"
            completed_variants.append(variant)

    unterminated = [
        attempt_id for attempt_id, state in attempts.items()
        if state["terminal"] is None
    ]
    if (
        (not attempts and len(raw_events) != 9)
        or len(unterminated) > 1
        or (unterminated and not allow_unterminated)
        or completed_variants != list(TRAINED_VARIANTS[:len(completed_variants)])
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 corrected attempt terminal/order history differs"
        )
    next_variant = next(
        (variant for variant in TRAINED_VARIANTS if variant not in completed_variants),
        None,
    )
    return {
        "status": "PASS", "corrected_attempt_count": len(attempts),
        "completed_variants": completed_variants,
        "unterminated_attempt_ids": unterminated, "next_variant": next_variant,
        "first_corrected_training_start_event_index": first_start_index,
        "cache_load_event_index": 8,
        "semantic_event_count": len(events), "raw_event_count": len(raw_events),
    }


def _amendment08_training_event_fields(
    *, run_dir: Path, variant: str,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    if run_dir.name != A08_TARGET_RUN_ID:
        return {}
    overlay = validate_amendment08_overlay(run_dir)
    history = _validate_amendment08_corrected_training_history(
        run_dir=run_dir, overlay=overlay, allow_unterminated=False,
    )
    if history["next_variant"] != variant:
        raise Amendment06IntegrityError(
            "Amendment 08 corrected Variant order differs before fit"
        )
    return _amendment08_training_lineage(overlay)


def validate_amendment08_pre_resume(
    *, run_dir: Path | None = None,
    preflight_run: Path | None = None,
    smoke_run: Path | None = None,
) -> dict[str, Any]:
    if run_dir is None:
        run_dir = A08_TARGET_RUN
    if preflight_run is None:
        preflight_run = ACCEPTED_PREFLIGHT
    if smoke_run is None:
        smoke_run = ACCEPTED_SMOKE
    run_dir = Path(run_dir).resolve()
    assert_exact_invocation(preflight_run=preflight_run, smoke_run=smoke_run)
    if _AMENDMENT08_REHEARSAL_CONTEXT.get() is None:
        admission = verify_reference_admission(
            preflight_run=preflight_run, smoke_run=smoke_run,
        )
    else:
        projection = validate_amendment08_isolated_rehearsal_projection(run_dir)
        admission = dict(projection["reference_projection"])
    if not _safe_regular(EXTERNAL_AUDIT, EXTERNAL_AUDIT_SHA256):
        raise Amendment06IntegrityError(
            "Amendment 08 external Audit streaming hash differs before Resume"
        )
    overlay = validate_amendment08_overlay(run_dir)
    cache = _validate_preserved_target_cache_files(run_dir)
    views = read_amendment08_journal_views(run_dir)
    cache_loads = [
        event for event in views["semantic_events"] if event.get("event") == "CACHE_LOAD"
    ]
    history: Mapping[str, Any] | None = None
    if cache_loads:
        history = _validate_amendment08_corrected_training_history(
            run_dir=run_dir, overlay=overlay, allow_unterminated=True,
        )
    elif len(views["raw_events"]) != 8:
        raise Amendment06IntegrityError(
            "Unexpected event precedes Amendment 08 CACHE_LOAD"
        )
    return {
        "status": "PASS", "authorization": A08_AUTHORIZATION,
        "prompt_sha256": A08_PROMPT_SHA256, "target_run_id": run_dir.name,
        "overlay": overlay, "cache": cache,
        "reference_admission_status": admission["status"],
        "raw_journal_event_count": len(views["raw_events"]),
        "semantic_journal_event_count": len(views["semantic_events"]),
        "cache_load_event_count": len(cache_loads),
        "corrected_training_history": history,
        "external_audit_tabular_parse_calls": 0,
        "new_full_run_ids_created": 0,
    }


def _train_one_variant_impl(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    development: DevelopmentPopulation, variant: str,
) -> dict[str, Any]:
    completion_relative = _training_completion_relative(variant)
    if (run_dir / completion_relative).exists():
        result = validate_training_completion(
            run_dir=run_dir, locked=locked, cache=cache,
            development=development, variant=variant, active_prediction=True,
        )
        _ensure_unique_bound_journal_event(
            run_dir=run_dir, event_name="TRAINING_COMPLETION_COMMITTED",
            binding_path=run_dir / completion_relative,
            binding_field="completion_sha256", phase="TRAINING", variant=variant,
            recovered=True, extra={
                "attempt_id": _phase_publication_records(
                    run_dir, "TRAINING", variant,
                ).get(completion_relative, {}).get("attempt_id"),
            },
        )
        _journal_event(run_dir, "COMPLETED_VARIANT_SKIPPED", phase="TRAINING", variant_id=variant)
        return result
    _record_interrupted_attempts(
        run_dir=run_dir, phase="TRAINING", variant=variant,
    )
    _recover_unpublished_paths(
        run_dir=run_dir, phase="TRAINING", variant=variant,
        completion_relative=completion_relative,
        possible_relatives=_training_relatives(variant),
    )
    gpu_identity = core.gpu_identity_metadata()
    gpu_name = _gpu_name(gpu_identity)
    scientific_event_fields = (
        _amendment08_training_event_fields(run_dir=run_dir, variant=variant)
        if Path(run_dir).resolve().name == A08_TARGET_RUN_ID
        else _amendment07_training_event_fields(run_dir=run_dir, variant=variant)
    )
    attempt_id = uuid.uuid4().hex
    staging = _attempt_root(run_dir, "TRAINING", variant, attempt_id)
    staging.mkdir(parents=True)
    started = _journal_event(
        run_dir, "TRAINING_ATTEMPT_STARTED", phase="TRAINING",
        variant_id=variant, attempt_id=attempt_id, **scientific_event_fields,
    )
    if started.get("attempt_id") != attempt_id:
        raise Amendment06IntegrityError("Corrected training attempt start was not committed")
    _progress("FULL_VARIANT_TRAINING", "START", variant=variant, index=f"{TRAINED_VARIANTS.index(variant)+1}/10")
    variant_data = project_variant(cache, locked, variant)
    classifier = xgb.XGBClassifier(**locked.model_parameters)
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    monitor = legacy.PeakMonitor()
    fit_progress = BoundedFitProgress(run_dir.name, variant)
    caught_messages: list[str] = []
    try:
        _SCIENTIFIC_COUNTERS["training_calls"] += 1
        monitor.__enter__()
        fit_progress.__enter__()
        _journal_event(
            run_dir, "SCIENTIFIC_FIT_CALLED", phase="TRAINING",
            variant_id=variant, attempt_id=attempt_id,
            seed=PRIMARY_MODEL_SEED, stability_fit=False,
            **scientific_event_fields,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            classifier.fit(
                variant_data.X, variant_data.y,
                sample_weight=variant_data.weights,
                eval_set=[(variant_data.validation_X, development.validation_labels)],
                verbose=False,
            )
        caught_messages = [str(item.message) for item in caught]
        monitor.__exit__(None, None, None)
    except Exception as exc:
        with suppress(Exception):
            monitor.__exit__(type(exc), exc, exc.__traceback__)
        _journal_event(
            run_dir, "TRAINING_ATTEMPT_FAILED", phase="TRAINING", variant_id=variant,
            attempt_id=attempt_id, error_type=type(exc).__name__, error=str(exc),
        )
        raise
    finally:
        fit_progress.__exit__(*sys.exc_info())
    booster = classifier.get_booster()
    configuration = strict_full_loads(booster.save_config())
    if "cuda" not in json.dumps(configuration).lower() or _fallback_warnings(caught_messages):
        raise Amendment06IntegrityError(f"CUDA execution/fallback gate failed: {variant}")
    best_iteration = int(classifier.best_iteration)
    iteration_range = model_compat.classifier_equivalent_iteration_range(booster)
    feature_sha = _feature_list_sha256(variant_data.features)
    model_compat.embed_binary_classifier_metadata(
        classifier, expected_feature_count=len(variant_data.features),
        feature_list_sha256=feature_sha,
    )
    embedded = {
        "run_id": run_dir.name,
        "variant_id": variant,
        "variant_classification": variant_data.classification,
        "feature_count": str(len(variant_data.features)),
        "feature_list_sha256": feature_sha,
        "full_parameter_lock_sha256": FULL_MODEL_PARAMETERS_SHA256,
        "frozen_population_sha256": cache.binding_sha256,
        "imputer_sha256": cache.imputer_sha256,
        "split_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
        **{
            key: str(value)
            for key, value in _scientific_lineage_bindings(run_dir).items()
        },
        "seed": str(PRIMARY_MODEL_SEED),
        "decision_threshold": str(DECISION_THRESHOLD),
        "gpu_name": gpu_name,
        "best_iteration": str(best_iteration),
        "iteration_range": strict_full_canonical_json_bytes(list(iteration_range)).decode("utf-8"),
    }
    booster.set_attr(**embedded)
    model_relative = f"models/{variant}/model.ubj"
    staged_model = staging / model_relative
    staged_model.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(staged_model)
    with staged_model.open("rb") as handle:
        os.fsync(handle.fileno())
    parity = model_compat.four_way_binary_reload_parity(
        classifier, staged_model, variant_data.validation_X,
        expected_feature_count=len(variant_data.features),
        expected_feature_list_sha256=feature_sha, device="cuda",
    )
    parity_differences = parity.evidence.get("maximum_absolute_difference", {})
    parity_maximum = max(map(float, parity_differences.values()), default=math.inf)
    if parity_maximum != 0.0:
        raise Amendment06IntegrityError(f"Four-way validation parity is not bit-exact: {variant}")
    active_probe = core._active_reloaded_booster_cuda_probe(
        parity.reloaded_booster, len(variant_data.features),
    )
    if active_probe.get("status") != "PASS":
        raise Amendment06IntegrityError(f"Reloaded active CUDA probe failed: {variant}")
    probabilities = parity.fitted_classifier_positive
    validation_frame = _training_prediction_frame(
        development=development, probabilities=probabilities,
        variant=variant, population=variant_data.population,
    )
    validation_metrics = _binary_metrics(development.validation_labels, probabilities)
    weighted_validation_metrics = _binary_metrics(
        development.validation_labels, probabilities,
        weights=development.validation_weights,
    )
    csv_bytes = deterministic_csv_gzip_bytes(validation_frame)
    reopened, roundtrip = validate_prediction_roundtrip(csv_bytes, validation_frame)
    recomputed = _binary_metrics(reopened.true_label, reopened.probability)
    if recomputed != validation_metrics:
        raise Amendment06IntegrityError(f"Saved validation metric recomputation differs: {variant}")
    common = _model_common_metadata(
        run_dir=run_dir, variant_data=variant_data, model_path=staged_model,
        cache=cache, gpu_identity=gpu_identity, gpu_name=gpu_name,
        best_iteration=best_iteration, iteration_range=iteration_range,
    )
    peak_gpu_memory_mib = getattr(monitor, "peak_gpu_memory_mib", None)
    if (
        isinstance(peak_gpu_memory_mib, bool)
        or not isinstance(peak_gpu_memory_mib, (int, float))
        or not math.isfinite(float(peak_gpu_memory_mib))
        or float(peak_gpu_memory_mib) < 0.0
    ):
        raise Amendment06IntegrityError(f"Peak GPU-memory evidence is invalid: {variant}")
    timing = {
        "run_id": run_dir.name, "variant_id": variant,
        "wall_seconds": float(time.perf_counter() - start_wall),
        "cpu_seconds": float(time.process_time() - start_cpu),
        "peak_ram_bytes": int(max(before_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024),
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
        "training_rows": len(variant_data.y), "validation_rows": len(development.validation_labels),
        "best_iteration": best_iteration, "boosted_rounds": int(booster.num_boosted_rounds()),
    }
    metadata = {
        **common, "status": "PASS", "requested_device": "cuda",
        "cpu_fallback_detected": False,
        "requested_model_parameters": locked.model_parameters,
        "booster_configuration_sha256": sha256_bytes(strict_full_canonical_json_bytes(configuration)),
        "booster_configuration": configuration,
        "captured_warning_messages": caught_messages,
        "fallback_warnings": _fallback_warnings(caught_messages),
        "population_binding": variant_data.population_evidence,
        "embedded_model_attributes": embedded,
    }
    transform = {
        "run_id": run_dir.name, "variant_id": variant, "status": "PASS",
        "fit_scope": "ONE_TRAINING_ONLY_93D_MEDIAN_IMPUTER",
        "imputer_sha256": cache.imputer_sha256,
        "full_feature_order_sha256": _feature_list_sha256(cache.full_features),
        "retained_feature_count": len(variant_data.features),
        "retained_feature_list_sha256": feature_sha,
        "retained_column_indices": list(variant_data.indices),
        "projection_only": True,
    }
    feature_bytes = ("\n".join(variant_data.features) + "\n").encode("utf-8")
    _stage_bytes(staging, f"models/{variant}/feature_list.txt", feature_bytes)
    _stage_json(staging, f"models/{variant}/model_metadata.json", metadata)
    _stage_json(staging, f"models/{variant}/transform_binding.json", transform)
    _stage_bytes(staging, f"predictions/{variant}_validation.csv.gz", csv_bytes)
    _stage_json(staging, f"timing/{variant}.json", timing)
    _stage_bytes(
        staging, f"logs/{variant}.stdout.log",
        (f"VARIANT={variant}\nPHASE=TRAINING_AND_VALIDATION_COMPLETE\nSTATUS=PASS\n").encode(),
    )
    _stage_bytes(
        staging, f"logs/{variant}.stderr.log",
        (("\n".join(caught_messages) + "\n") if caught_messages else "NO_WARNINGS\n").encode(),
    )
    artifact_records = [
        _publication_record(staging / relative, relative)
        for relative in _training_relatives(variant)
    ]
    completion = {
        **common,
        "phase": "TRAINING_AND_VALIDATION_COMPLETE", "status": "PASS",
        "cpu_fallback_detected": False,
        "four_way_parity_max_abs_diff": parity_maximum,
        "four_way_reload_parity": parity.evidence,
        "active_reloaded_cuda_probe": active_probe,
        "requested_model_parameters": locked.model_parameters,
        "booster_configuration_sha256": sha256_bytes(
            strict_full_canonical_json_bytes(configuration)
        ),
        "captured_warning_messages": caught_messages,
        "fallback_warnings": _fallback_warnings(caught_messages),
        "validation_prediction_sha256": sha256_bytes(csv_bytes),
        "validation_prediction_rows": len(validation_frame),
        "validation_order_sha256": sha256_bytes(_lineage_bytes(validation_frame.stable_candidate_id)),
        "validation_probability_sha256_float32": sha256_array(probabilities.astype("<f4", copy=False)),
        "validation_metrics": validation_metrics,
        "weighted_validation_metrics": weighted_validation_metrics,
        "csv_roundtrip": roundtrip,
        "artifact_records": artifact_records,
        "model_metadata_sha256": sha256_file(
            staging / f"models/{variant}/model_metadata.json"
        ),
        "transform_binding_sha256": sha256_file(
            staging / f"models/{variant}/transform_binding.json"
        ),
        "timing_resource_sha256": sha256_file(staging / f"timing/{variant}.json"),
        "source_code_manifest_sha256": common[
            "scientific_execution_code_manifest_sha256"
        ],
    }
    _stage_json(staging, completion_relative, completion)
    try:
        for relative in _training_relatives(variant):
            _publish_attempt_file(
                run_dir=run_dir, staging=staging, relative=relative,
                phase="TRAINING", variant=variant, attempt_id=attempt_id,
            )
        _publish_attempt_file(
            run_dir=run_dir, staging=staging, relative=completion_relative,
            phase="TRAINING", variant=variant, attempt_id=attempt_id,
        )
    except Exception as exc:
        _journal_event(
            run_dir, "TRAINING_PUBLICATION_INTERRUPTED", phase="TRAINING",
            variant_id=variant, attempt_id=attempt_id,
            error_type=type(exc).__name__, error=str(exc),
        )
        raise
    _ensure_unique_bound_journal_event(
        run_dir=run_dir, event_name="TRAINING_COMPLETION_COMMITTED",
        binding_path=run_dir / completion_relative,
        binding_field="completion_sha256", phase="TRAINING", variant=variant,
        recovered=False, extra={"attempt_id": attempt_id},
    )
    _progress("FULL_VARIANT_TRAINING", "COMPLETE", variant=variant, status="PASS")
    return validate_training_completion(
        run_dir=run_dir, locked=locked, cache=cache,
        development=development, variant=variant, active_prediction=False,
    )


def train_one_variant(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    development: DevelopmentPopulation, variant: str,
) -> dict[str, Any]:
    before = len(_scientific_journal_events(run_dir))
    try:
        return _train_one_variant_impl(
            run_dir=run_dir, locked=locked, cache=cache,
            development=development, variant=variant,
        )
    except Exception as exc:
        events = _scientific_journal_events(run_dir)[before:]
        starts = [event for event in events if event.get("event") == "TRAINING_ATTEMPT_STARTED"]
        if starts:
            attempt_id = str(starts[-1]["attempt_id"])
            already_recorded = any(
                event.get("event") == "TRAINING_ATTEMPT_FAILED"
                and event.get("attempt_id") == attempt_id
                for event in events
            )
            if not already_recorded:
                _journal_event(
                    run_dir, "TRAINING_ATTEMPT_FAILED", phase="TRAINING",
                    failure_phase="POST_START_THROUGH_COMPLETION_COMMIT",
                    variant_id=variant, attempt_id=attempt_id,
                    error_type=type(exc).__name__, error=str(exc),
                )
        raise


def _expected_variant_population_binding(
    *, cache: FrozenPopulationCache, locked: LockedInputs, variant: str,
) -> dict[str, Any]:
    features = tuple(locked.variant_features[variant]["retained_features"])
    expected_count, feature_sha = VARIANT_FEATURE_CONTRACT[variant]
    if len(features) != expected_count or _feature_list_sha256(features) != feature_sha:
        raise Amendment06IntegrityError(f"Variant feature contract drifted: {variant}")
    use_pre = variant == "no_safe_smote"
    stage_name = "post_jitter_pre_smote" if use_pre else "post_safe_smote"
    stage = next(
        (item for item in cache.manifest["stages"] if item.get("stage") == stage_name),
        None,
    )
    if not isinstance(stage, Mapping):
        raise Amendment06IntegrityError(f"Frozen population stage is absent: {stage_name}")
    rows = PRE_SMOTE_ROWS if use_pre else POST_SMOTE_ROWS
    weight_sha256 = (
        sha256_array(np.ones(rows, dtype=np.float32))
        if variant == "no_review_aware_training_weights"
        else str(stage["weight_sha256"])
    )
    return {
        "variant": variant, "source_stage": stage_name, "projection_only": True,
        "rows": rows, "feature_count": expected_count,
        "feature_list_sha256": feature_sha,
        "full_space_X_sha256": stage["X_sha256"],
        "y_sha256": stage["y_sha256"],
        "source_lineage_sha256": stage["source_lineage_sha256"],
        "row_order_sha256": stage["row_order_sha256"],
        "weight_sha256": weight_sha256,
        "review_weights_enabled": variant != "no_review_aware_training_weights",
        "safe_smote_enabled": not use_pre,
        "safe_smote_synthetic_rows": 0 if use_pre else 147_841,
    }


def validate_training_completion(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    development: DevelopmentPopulation, variant: str,
    active_prediction: bool = False,
) -> dict[str, Any]:
    if variant not in TRAINED_VARIANTS:
        raise Amendment06IntegrityError(f"Unauthorized completed Variant: {variant}")
    completion_path = run_dir / _training_completion_relative(variant)
    if not _safe_regular(completion_path):
        raise Amendment06IntegrityError(f"Training completion is absent: {variant}")
    completion = strict_full_load_file(completion_path)
    expected_classification = "EXPLORATORY" if variant == EXPLORATORY_VARIANT else "OFFICIAL_RUNNABLE"
    model_path = run_dir / f"models/{variant}/model.ubj"
    feature_path = run_dir / f"models/{variant}/feature_list.txt"
    metadata_path = run_dir / f"models/{variant}/model_metadata.json"
    transform_path = run_dir / f"models/{variant}/transform_binding.json"
    timing_path = run_dir / f"timing/{variant}.json"
    stderr_path = run_dir / f"logs/{variant}.stderr.log"
    stdout_path = run_dir / f"logs/{variant}.stdout.log"
    for required in (
        model_path, feature_path, metadata_path, transform_path, timing_path,
        stderr_path, stdout_path,
    ):
        if not _safe_regular(required):
            raise Amendment06IntegrityError(f"Completed training member is absent/unsafe: {required}")
    features = tuple(feature_path.read_text(encoding="utf-8").splitlines())
    expected_count, expected_feature_sha = VARIANT_FEATURE_CONTRACT[variant]
    identity = strict_full_load_file(run_dir / "config/run_identity.lock.json")
    lineage = _scientific_lineage_bindings(run_dir)
    scientific_manifest_sha = lineage[
        "effective_scientific_execution_code_manifest_sha256"
    ]
    source_pre_sha = sha256_file(run_dir / "provenance/source_input_hashes_pre.tsv")
    population_binding = _expected_variant_population_binding(
        cache=cache, locked=locked, variant=variant,
    )
    expected_common = {
        "run_id": run_dir.name,
        "run_kind": RUN_KIND,
        "variant_id": variant,
        "variant_classification": expected_classification,
        "model_sha256": sha256_file(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "feature_count": expected_count,
        "feature_list_sha256": expected_feature_sha,
        "full_parameter_lock_sha256": FULL_MODEL_PARAMETERS_SHA256,
        "frozen_population_sha256": cache.binding_sha256,
        "imputer_sha256": cache.imputer_sha256,
        "split_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
        "primary_table_sha256": PRIMARY_TABLE_SHA256,
        "source_input_hashes_pre_sha256": source_pre_sha,
        "training_configuration_sha256": TRAINING_CONFIGURATION_SHA256,
        "variant_feature_sets_sha256": VARIANT_FEATURE_SETS_SHA256,
        "row_split_manifest_sha256": ROW_SPLIT_MANIFEST_SHA256,
        **lineage,
        "scientific_execution_code_manifest_sha256": scientific_manifest_sha,
        "seed": PRIMARY_MODEL_SEED,
        "decision_threshold": DECISION_THRESHOLD,
    }
    if not (
        all(completion.get(key) == value for key, value in expected_common.items())
        and identity.get("scientific_execution_code_manifest_sha256")
        == lineage["base_scientific_execution_code_manifest_sha256"]
        and completion.get("phase") == "TRAINING_AND_VALIDATION_COMPLETE"
        and completion.get("status") == "PASS"
        and completion.get("cpu_fallback_detected") is False
        and completion.get("four_way_parity_max_abs_diff") == 0.0
        and len(features) == expected_count and _feature_list_sha256(features) == expected_feature_sha
        and completion.get("requested_model_parameters") == locked.model_parameters
        and completion.get("source_code_manifest_sha256") == scientific_manifest_sha
    ):
        raise Amendment06IntegrityError(f"Training completion binding differs: {variant}")
    _validate_exact_artifact_records(
        run_dir=run_dir, value=completion.get("artifact_records"),
        expected_relatives=_training_relatives(variant),
        label=f"Completed training {variant}",
    )
    metadata = strict_full_load_file(metadata_path)
    transform = strict_full_load_file(transform_path)
    timing = strict_full_load_file(timing_path)
    booster_configuration = metadata.get("booster_configuration")
    warning_messages = metadata.get("captured_warning_messages")
    parity = completion.get("four_way_reload_parity")
    active_probe = completion.get("active_reloaded_cuda_probe")
    if not (
        all(metadata.get(key) == value for key, value in expected_common.items())
        and metadata.get("status") == "PASS"
        and metadata.get("requested_device") == "cuda"
        and metadata.get("cpu_fallback_detected") is False
        and metadata.get("requested_model_parameters") == locked.model_parameters
        and isinstance(booster_configuration, Mapping)
        and "cuda" in json.dumps(booster_configuration).lower()
        and metadata.get("booster_configuration_sha256")
        == sha256_bytes(strict_full_canonical_json_bytes(booster_configuration))
        and metadata.get("population_binding") == population_binding
        and isinstance(warning_messages, list)
        and all(isinstance(value, str) for value in warning_messages)
        and metadata.get("fallback_warnings") == []
        and completion.get("captured_warning_messages") == warning_messages
        and completion.get("fallback_warnings") == []
        and completion.get("booster_configuration_sha256")
        == metadata.get("booster_configuration_sha256")
        and completion.get("model_metadata_sha256") == sha256_file(metadata_path)
        and completion.get("transform_binding_sha256") == sha256_file(transform_path)
        and completion.get("timing_resource_sha256") == sha256_file(timing_path)
        and isinstance(parity, Mapping)
        and parity.get("status") == "PASS"
        and parity.get("classification") == "VERIFIED"
        and parity.get("device") == "cuda"
        and parity.get("feature_list_sha256") == expected_feature_sha
        and parity.get("iteration_range") == completion.get("iteration_range")
        and parity.get("cpu_fallback_detected") is False
        and parity.get("model_sha256_before") == completion.get("model_sha256")
        and parity.get("model_sha256_after") == completion.get("model_sha256")
        and parity.get("model_bytes_unchanged") is True
        and isinstance(parity.get("maximum_absolute_difference"), Mapping)
        and set(parity["maximum_absolute_difference"]) == {
            "fitted_classifier", "fitted_booster", "reloaded_booster",
            "compatibility_classifier",
        }
        and set(parity["maximum_absolute_difference"].values()) == {0.0}
        and isinstance(parity.get("bit_exact_against_fitted_classifier"), Mapping)
        and all(value is True for value in parity["bit_exact_against_fitted_classifier"].values())
        and isinstance(parity.get("warnings"), Mapping)
        and all(value == [] for value in parity["warnings"].values())
        and isinstance(active_probe, Mapping)
        and active_probe.get("status") == "PASS"
        and "cuda" in str(active_probe.get("device", "")).lower()
        and active_probe.get("warnings") == []
    ):
        raise Amendment06IntegrityError(f"Completed training evidence differs: {variant}")
    expected_transform = {
        "run_id": run_dir.name, "variant_id": variant, "status": "PASS",
        "fit_scope": "ONE_TRAINING_ONLY_93D_MEDIAN_IMPUTER",
        "imputer_sha256": cache.imputer_sha256,
        "full_feature_order_sha256": _feature_list_sha256(cache.full_features),
        "retained_feature_count": expected_count,
        "retained_feature_list_sha256": expected_feature_sha,
        "retained_column_indices": [
            cache.full_features.index(feature) for feature in features
        ],
        "projection_only": True,
    }
    numeric_timing = (
        "wall_seconds", "cpu_seconds", "peak_ram_bytes", "peak_gpu_memory_mib",
    )
    if not (
        transform == expected_transform
        and timing.get("run_id") == run_dir.name
        and timing.get("variant_id") == variant
        and timing.get("training_rows") == population_binding["rows"]
        and timing.get("validation_rows") == 56_843
        and timing.get("best_iteration") == completion.get("best_iteration")
        and timing.get("boosted_rounds") >= completion.get("best_iteration") + 1
        and all(
            isinstance(timing.get(key), (int, float))
            and not isinstance(timing.get(key), bool)
            and math.isfinite(float(timing[key])) and float(timing[key]) >= 0.0
            for key in numeric_timing
        )
        and stdout_path.read_bytes() == (
            f"VARIANT={variant}\nPHASE=TRAINING_AND_VALIDATION_COMPLETE\nSTATUS=PASS\n"
        ).encode("utf-8")
        and stderr_path.read_text(encoding="utf-8") == (
            ("\n".join(warning_messages) + "\n") if warning_messages else "NO_WARNINGS\n"
        )
        and not _fallback_warnings(warning_messages)
    ):
        raise Amendment06IntegrityError(f"Completed transform/timing/warnings differ: {variant}")
    booster = xgb.Booster()
    booster.load_model(model_path)
    attributes = booster.attributes()
    required_attributes = {
        "run_id": run_dir.name, "variant_id": variant,
        "variant_classification": expected_classification,
        "feature_count": str(expected_count), "feature_list_sha256": expected_feature_sha,
        "full_parameter_lock_sha256": FULL_MODEL_PARAMETERS_SHA256,
        "frozen_population_sha256": cache.binding_sha256,
        "imputer_sha256": cache.imputer_sha256,
        "split_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
        **lineage,
        "seed": str(PRIMARY_MODEL_SEED), "decision_threshold": str(DECISION_THRESHOLD),
        "gpu_name": str(completion["gpu_name"]),
        "best_iteration": str(completion["best_iteration"]),
        "iteration_range": strict_full_canonical_json_bytes(completion["iteration_range"]).decode(),
    }
    if any(attributes.get(key) != value for key, value in required_attributes.items()):
        raise Amendment06IntegrityError(f"Completed UBJ embedded attributes drifted: {variant}")
    raw_reloaded_configuration = strict_full_loads(booster.save_config())
    if (
        core._stable_booster_configuration(raw_reloaded_configuration)
        != core._stable_booster_configuration(booster_configuration)
    ):
        raise Amendment06IntegrityError(
            f"Completed UBJ stable configuration differs: {variant}"
        )
    try:
        booster.load_config(
            strict_full_canonical_json_bytes(booster_configuration).decode("utf-8")
        )
    except Exception as exc:
        raise Amendment06IntegrityError(
            f"Completed UBJ configuration cannot be restored: {variant}"
        ) from exc
    reloaded_configuration = strict_full_loads(booster.save_config())
    if (
        sha256_bytes(strict_full_canonical_json_bytes(reloaded_configuration))
        != completion.get("booster_configuration_sha256")
        or "cuda" not in json.dumps(reloaded_configuration).lower()
    ):
        raise Amendment06IntegrityError(f"Completed UBJ configuration differs: {variant}")
    prediction_path = run_dir / f"predictions/{variant}_validation.csv.gz"
    authoritative = pd.read_csv(prediction_path, engine="c", float_precision="round_trip")
    authoritative = authoritative.loc[:, list(PREDICTION_COLUMNS)]
    reopened, roundtrip = validate_prediction_roundtrip(prediction_path, authoritative)
    if not (
        len(reopened) == len(development.validation)
        and np.array_equal(
            reopened.stable_candidate_id.astype(str).to_numpy(),
            development.validation.stable_candidate_id.astype(str).to_numpy(),
        )
        and np.array_equal(reopened.true_label.to_numpy(np.int8), development.validation_labels)
        and np.array_equal(
            reopened.card_id.astype(str).to_numpy(),
            development.validation.card_id.astype(str).to_numpy(),
        )
        and np.array_equal(
            reopened.review_weight.to_numpy(np.float64),
            development.validation_weights.astype(np.float64),
        )
        and np.array_equal(
            reopened.prediction_at_0_5.to_numpy(np.int8),
            (reopened.probability.to_numpy(np.float64) >= DECISION_THRESHOLD).astype(np.int8),
        )
        and np.array_equal(reopened.row_order.to_numpy(np.int64), np.arange(len(reopened)))
        and set(reopened.variant_id.astype(str)) == {variant}
        and set(reopened.split_population.astype(str)) == {
            f"validation:{population_binding['source_stage']}"
        }
        and completion.get("validation_prediction_sha256") == sha256_file(prediction_path)
        and completion.get("validation_prediction_rows") == len(reopened)
        and completion.get("validation_order_sha256")
        == sha256_bytes(_lineage_bytes(reopened.stable_candidate_id.astype(str)))
        and completion.get("validation_probability_sha256_float32")
        == sha256_array(reopened.probability.to_numpy(np.float32).astype("<f4", copy=False))
        and _binary_metrics(reopened.true_label, reopened.probability) == completion.get("validation_metrics")
        and _binary_metrics(
            reopened.true_label, reopened.probability, weights=reopened.review_weight,
        ) == completion.get("weighted_validation_metrics")
        and completion.get("csv_roundtrip") == roundtrip
        and roundtrip.get("mismatch_count") == 0
    ):
        raise Amendment06IntegrityError(f"Completed validation predictions/metrics drifted: {variant}")
    active_evidence: dict[str, Any] | None = None
    if active_prediction:
        indices = tuple(cache.full_features.index(feature) for feature in features)
        matrix_host = (
            cache.validation_X if indices == tuple(range(93))
            else np.ascontiguousarray(cache.validation_X[:, indices], dtype=np.float32)
        )
        loaded, raw_booster, load_evidence = load_independent_external_predictors(
            model_path,
            expected_feature_count=expected_count,
            expected_feature_list_sha256=expected_feature_sha,
            expected_model_sha256=completion["model_sha256"],
        )
        import cupy as cp
        matrix = cp.asarray(np.ascontiguousarray(matrix_host, dtype=np.float32))
        interval = tuple(map(int, completion["iteration_range"]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wrapper = cp.asnumpy(
                loaded.classifier.predict_proba(matrix, iteration_range=interval)[:, 1]
            )
            raw = cp.asnumpy(raw_booster.inplace_predict(matrix, iteration_range=interval))
        active_warnings = [str(item.message) for item in caught]
        live_probe = core._active_reloaded_booster_cuda_probe(raw_booster, expected_count)
        if not (np.array_equal(wrapper, raw) and np.array_equal(wrapper, reopened.probability.to_numpy(np.float32))):
            raise Amendment06IntegrityError(f"Completed model active prediction differs: {variant}")
        if _fallback_warnings(active_warnings) or live_probe.get("status") != "PASS":
            raise Amendment06IntegrityError(f"Completed model active CUDA probe differs: {variant}")
        active_evidence = {
            "status": "PASS", "maximum_absolute_difference": 0.0,
            "independent_load_evidence": load_evidence,
            "active_reloaded_cuda_probe": live_probe,
            "warnings": active_warnings,
        }
    return {
        "status": "PASS", "variant_id": variant,
        "completion_sha256": sha256_file(completion_path),
        "model_sha256": completion["model_sha256"],
        "active_prediction": active_evidence,
    }


def ensure_blocked_no_pca(run_dir: Path) -> None:
    status_path = run_dir / "models/no_pca/STATUS.txt"
    evidence_path = run_dir / "models/no_pca/blocked_evidence.json"
    status = (
        f"RUN_ID={run_dir.name}\nVARIANT_ID=no_pca\n"
        "STATUS=NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE\n"
        "MODEL_CREATED=NO\n"
    ).encode("utf-8")
    evidence = {
        "run_id": run_dir.name, "run_kind": RUN_KIND,
        "variant_id": BLOCKED_VARIANT,
        "variant_classification": "OFFICIAL_BLOCKED",
        "status": "NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE",
        "model_created": False, "matrix_created": False,
        "prediction_created": False, "metric_created": False,
        "reason": "RAW_EMBEDDINGS_UNAVAILABLE_AND_NEW_RECONSTRUCTION_NOT_AUTHORIZED",
        "no_deep_pca_features_is_not_true_no_pca": True,
    }
    for path, data in ((status_path, status), (evidence_path, strict_full_json_bytes(evidence))):
        if path.exists():
            if not _safe_regular(path, sha256_bytes(data)):
                raise Amendment06IntegrityError(f"Blocked no-PCA evidence drifted: {path}")
        else:
            publish_bytes_no_clobber(path, data)


_GUARDED_ORIGINALS: list[tuple[Any, str, Any]] = []


def _install_post_freeze_guards() -> None:
    global _POST_FREEZE_GUARDS_INSTALLED
    if _POST_FREEZE_GUARDS_INSTALLED:
        return

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise PostFreezeScientificCallError(
            "Scientific fit/resampling/imputer-fit API is forbidden after FULL_MODELS_FROZEN"
        )

    targets = (
        (xgb.XGBClassifier, "fit"),
        (xgb, "train"),
        (xgb, "cv"),
        (xgb.Booster, "update"),
        (xgb.Booster, "boost"),
        (SimpleImputer, "fit"),
        (SimpleImputer, "fit_transform"),
        (paired_resampling, "build_frozen_resampling_population"),
        (core, "augment_with_lineage"),
        (core, "augmentation_plan_counts"),
        (core, "replay_augmentation_weights"),
        (legacy, "apply_training_augmentation"),
        (legacy, "safe_smote_resample"),
        (legacy.SafeSMOTE, "fit_resample"),
    )
    for owner, name in targets:
        if not hasattr(owner, name):
            raise Amendment06IntegrityError(f"Freeze-guard target is absent: {owner}.{name}")
        _GUARDED_ORIGINALS.append((owner, name, getattr(owner, name)))
        setattr(owner, name, forbidden)
    _POST_FREEZE_GUARDS_INSTALLED = True


def validate_models_frozen_manifest(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    development: DevelopmentPopulation,
) -> dict[str, Any]:
    path = run_dir / "provenance/full_models_frozen_manifest.json"
    payload = strict_full_load_file(path)
    models = payload.get("models")
    events = _scientific_journal_events(run_dir)
    lineage = _scientific_lineage_bindings(run_dir)
    if Path(run_dir).resolve().name == A08_TARGET_RUN_ID:
        history = _validate_amendment08_corrected_training_history(
            run_dir=run_dir, allow_unterminated=False,
        )
        if history["completed_variants"] != list(TRAINED_VARIANTS):
            raise Amendment06IntegrityError(
                "Amendment 08 models cannot freeze before all corrected completions"
            )
    fit_attempts = sum(event.get("event") == "SCIENTIFIC_FIT_CALLED" for event in events)
    failed_attempts = sum(event.get("event") == "TRAINING_ATTEMPT_FAILED" for event in events)
    if not (
        payload.get("run_id") == run_dir.name
        and payload.get("run_kind") == RUN_KIND
        and payload.get("phase") == "FULL_MODELS_FROZEN"
        and payload.get("status") == "PASS"
        and payload.get("trained_model_count") == 10
        and payload.get("official_runnable_model_count") == 9
        and payload.get("exploratory_model_count") == 1
        and payload.get("true_no_pca_model_count") == 0
        and payload.get("frozen_population_sha256") == cache.binding_sha256
        and payload.get("imputer_sha256") == cache.imputer_sha256
        and payload.get("full_parameter_lock_sha256") == FULL_MODEL_PARAMETERS_SHA256
        and payload.get("split_assignment_sha256") == SPLIT_ASSIGNMENT_SHA256
        and all(payload.get(key) == value for key, value in lineage.items())
        and payload.get("scientific_training_calls") == 10
        and payload.get("scientific_fit_attempts") == fit_attempts
        and payload.get("failed_unpublished_training_attempts") == failed_attempts
        and fit_attempts >= 10
        and payload.get("external_audit_opened") is False
        and isinstance(models, list) and [item.get("variant_id") for item in models] == list(TRAINED_VARIANTS)
    ):
        raise Amendment06IntegrityError("Full models-frozen manifest schema/count differs")
    for item in models:
        variant = str(item["variant_id"])
        expected_classification = (
            "EXPLORATORY" if variant == EXPLORATORY_VARIANT else "OFFICIAL_RUNNABLE"
        )
        if set(item) != {
            "variant_id", "variant_classification", "model_sha256",
            "training_completion_sha256",
        } or item.get("variant_classification") != expected_classification:
            raise Amendment06IntegrityError("Models-frozen member schema/classification differs")
        result = validate_training_completion(
            run_dir=run_dir, locked=locked, cache=cache,
            development=development, variant=variant,
            active_prediction=True,
        )
        if not (
            result["model_sha256"] == item.get("model_sha256")
            and result["completion_sha256"] == item.get("training_completion_sha256")
        ):
            raise Amendment06IntegrityError("Models-frozen member binding differs")
    _ensure_unique_bound_journal_event(
        run_dir=run_dir, event_name="MODELS_FROZEN",
        binding_path=path, binding_field="manifest_sha256", recovered=True,
        extra={"status": "PASS", "model_count": 10},
    )
    return payload


def freeze_all_models(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    development: DevelopmentPopulation,
) -> dict[str, Any]:
    _progress("FULL_MODELS_FREEZE", "START")
    path = run_dir / "provenance/full_models_frozen_manifest.json"
    manifest_created_now = not path.exists()
    if not manifest_created_now:
        payload = validate_models_frozen_manifest(
            run_dir=run_dir, locked=locked, cache=cache, development=development,
        )
    else:
        lineage = _scientific_lineage_bindings(run_dir)
        models: list[dict[str, Any]] = []
        for variant in TRAINED_VARIANTS:
            result = validate_training_completion(
                run_dir=run_dir, locked=locked, cache=cache,
                development=development, variant=variant, active_prediction=True,
            )
            models.append({
                "variant_id": variant,
                "variant_classification": (
                    "EXPLORATORY" if variant == EXPLORATORY_VARIANT else "OFFICIAL_RUNNABLE"
                ),
                "model_sha256": result["model_sha256"],
                "training_completion_sha256": result["completion_sha256"],
            })
        payload = {
            "run_id": run_dir.name, "run_kind": RUN_KIND,
            "phase": "FULL_MODELS_FROZEN", "status": "PASS",
            "frozen_at": _now(), "trained_model_count": 10,
            "official_runnable_model_count": 9, "exploratory_model_count": 1,
            "true_no_pca_model_count": 0,
            "frozen_population_sha256": cache.binding_sha256,
            "imputer_sha256": cache.imputer_sha256,
            "full_parameter_lock_sha256": FULL_MODEL_PARAMETERS_SHA256,
            "split_assignment_sha256": SPLIT_ASSIGNMENT_SHA256,
            **lineage,
            "scientific_training_calls": 10,
            "scientific_fit_attempts": sum(
                event.get("event") == "SCIENTIFIC_FIT_CALLED"
                for event in _scientific_journal_events(run_dir)
            ),
            "failed_unpublished_training_attempts": sum(
                event.get("event") == "TRAINING_ATTEMPT_FAILED"
                for event in _scientific_journal_events(run_dir)
            ),
            "external_audit_opened": False,
            "models": models,
        }
        publish_strict_json_no_clobber(path, payload)
    current = _current_run_state(run_dir)
    if current == "FULL_IN_PROGRESS":
        _set_run_state(run_dir, "FULL_MODELS_FROZEN")
    elif RUN_PHASES.index(current) < RUN_PHASES.index("FULL_MODELS_FROZEN"):
        raise Amendment06IntegrityError("Models-frozen manifest/state disagree")
    _install_post_freeze_guards()
    _ensure_unique_bound_journal_event(
        run_dir=run_dir, event_name="MODELS_FROZEN",
        binding_path=path, binding_field="manifest_sha256",
        recovered=not manifest_created_now,
        extra={"status": "PASS", "model_count": 10},
    )
    _progress("FULL_MODELS_FREEZE", "COMPLETE", status="PASS")
    return payload


@dataclass(frozen=True)
class ExternalAuditPopulation:
    frame: pd.DataFrame
    full_X: np.ndarray
    labels: np.ndarray
    weights: np.ndarray


def _frozen_imputer_transform(cache: FrozenPopulationCache, frame: pd.DataFrame) -> np.ndarray:
    if tuple(frame.columns) != cache.full_features:
        raise Amendment06IntegrityError("Frozen imputer transform received wrong feature order")
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise Amendment06IntegrityError("Frozen imputer transform received infinity")
    medians = np.asarray(cache.imputer["statistics"], dtype=np.float64)
    missing = np.isnan(values)
    if missing.any():
        values[missing] = np.take(medians, np.nonzero(missing)[1])
    transformed = np.ascontiguousarray(values, dtype=np.float32)
    if transformed.shape != (len(frame), 93) or not np.isfinite(transformed).all():
        raise Amendment06IntegrityError("Frozen imputer transform output is invalid")
    return transformed


def open_external_audit_after_freeze(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    development: DevelopmentPopulation,
) -> ExternalAuditPopulation:
    if RUN_PHASES.index(_current_run_state(run_dir)) < RUN_PHASES.index("FULL_MODELS_FROZEN"):
        raise Amendment06IntegrityError("External Audit cannot open before FULL_MODELS_FROZEN")
    validate_models_frozen_manifest(
        run_dir=run_dir, locked=locked, cache=cache, development=development,
    )
    _install_post_freeze_guards()
    evidence_path = run_dir / "provenance/external_audit_exclusion_and_opening.json"
    evidence_created_now = not evidence_path.exists()
    if evidence_created_now:
        evidence = {
            "run_id": run_dir.name, "run_kind": RUN_KIND, "status": "PASS",
            "external_audit_path": str(EXTERNAL_AUDIT),
            "external_audit_sha256": EXTERNAL_AUDIT_SHA256,
            "external_audit_size_bytes": EXTERNAL_AUDIT.stat().st_size,
            "external_audit_opened_at": _now(),
            "opened_after_full_models_frozen": True,
            "models_frozen_manifest_sha256": sha256_file(
                run_dir / "provenance/full_models_frozen_manifest.json"
            ),
            "models_frozen_state_marker_sha256": sha256_file(
                _state_marker_path(run_dir, "FULL_MODELS_FROZEN")
            ),
            "pre_freeze_audit_access": "EXISTENCE_SIZE_SHA256_ONLY_NO_PARSE",
            "external_audit_used_for_fitting": False,
            "external_audit_used_for_imputer_fit": False,
            "external_audit_used_for_early_stopping": False,
            "external_audit_used_for_threshold_or_model_selection": False,
            "post_audit_training_calls": 0,
        }
        publish_strict_json_no_clobber(evidence_path, evidence)
    else:
        evidence = strict_full_load_file(evidence_path)
        if not (
            evidence.get("run_id") == run_dir.name
            and evidence.get("run_kind") == RUN_KIND
            and evidence.get("status") == "PASS"
            and evidence.get("external_audit_path") == str(EXTERNAL_AUDIT)
            and evidence.get("external_audit_sha256") == EXTERNAL_AUDIT_SHA256
            and evidence.get("external_audit_size_bytes") == EXTERNAL_AUDIT.stat().st_size
            and isinstance(evidence.get("external_audit_opened_at"), str)
            and evidence.get("models_frozen_manifest_sha256") == sha256_file(
                run_dir / "provenance/full_models_frozen_manifest.json"
            )
            and evidence.get("models_frozen_state_marker_sha256") == sha256_file(
                _state_marker_path(run_dir, "FULL_MODELS_FROZEN")
            )
            and evidence.get("opened_after_full_models_frozen") is True
            and evidence.get("pre_freeze_audit_access")
            == "EXISTENCE_SIZE_SHA256_ONLY_NO_PARSE"
            and evidence.get("external_audit_used_for_fitting") is False
            and evidence.get("external_audit_used_for_imputer_fit") is False
            and evidence.get("external_audit_used_for_early_stopping") is False
            and evidence.get("external_audit_used_for_threshold_or_model_selection") is False
            and evidence.get("post_audit_training_calls") == 0
        ):
            raise Amendment06IntegrityError("External Audit opening evidence differs")
    _ensure_unique_bound_journal_event(
        run_dir=run_dir, event_name="EXTERNAL_AUDIT_OPENED",
        binding_path=evidence_path, binding_field="evidence_sha256",
        recovered=not evidence_created_now, extra={"status": "PASS"},
    )
    _progress("EXTERNAL_AUDIT_OPEN", "START")
    audit = core._clean_canonical_audit_frame(EXTERNAL_AUDIT, EXTERNAL_AUDIT_SHA256)
    if not (
        len(audit) == 10_097 and int(audit.human_label.sum()) == 2_032
        and audit.card_id.nunique() == 10
        and set(audit.card_id.astype(str)).isdisjoint(locked.split.card_id.astype(str))
    ):
        raise Amendment06IntegrityError("Locked external Audit population/count differs")
    harmonized = core.harmonize_audit_features(audit, locked.full_features)
    full_X = _frozen_imputer_transform(cache, harmonized.loc[:, locked.full_features])
    labels = audit.human_label.to_numpy(np.int8)
    if "review_weight" in audit:
        weights = pd.to_numeric(audit.review_weight, errors="coerce").fillna(1.0).to_numpy(np.float64)
    else:
        weights = np.ones(len(audit), dtype=np.float64)
    if not (np.isfinite(weights).all() and np.all(weights > 0)):
        raise Amendment06IntegrityError("External Audit sensitivity weights are invalid")
    _progress("EXTERNAL_AUDIT_OPEN", "COMPLETE", rows=len(audit), cards=audit.card_id.nunique())
    return ExternalAuditPopulation(audit, full_X, labels, weights)


def _external_prediction_frame(
    *, audit: ExternalAuditPopulation, probabilities: np.ndarray, variant: str,
) -> pd.DataFrame:
    return pd.DataFrame({
        "stable_candidate_id": audit.frame.stable_candidate_id.astype(str).to_numpy(),
        "card_id": audit.frame.card_id.astype(str).to_numpy(),
        "true_label": audit.labels,
        "probability": np.asarray(probabilities, dtype=np.float32).astype(np.float64),
        "prediction_at_0_5": (np.asarray(probabilities) >= DECISION_THRESHOLD).astype(np.int8),
        "review_weight": audit.weights,
        "variant_id": variant,
        "split_population": "external_audit:locked_10_card",
        "row_order": np.arange(len(probabilities), dtype=np.int64),
    }, columns=PREDICTION_COLUMNS)


def _scoring_relatives(variant: str) -> tuple[str, ...]:
    return (
        f"predictions/{variant}_external_audit.csv.gz",
        f"metrics/{variant}.json",
        f"metrics/{variant}_per_card.csv",
        f"models/{variant}/STATUS.txt",
    )


def _scoring_completion_relative(variant: str) -> str:
    return f"models/{variant}/external_scoring_manifest.json"


def _variant_complete_status(run_id: str, variant: str) -> bytes:
    return (
        f"RUN_ID={run_id}\nVARIANT_ID={variant}\n"
        "TRAINING_AND_VALIDATION_COMPLETE\n"
        "EXTERNAL_AUDIT_SCORING_COMPLETE\nSTATUS=PASS\n"
    ).encode("utf-8")


def load_independent_external_predictors(
    model_path: Path, *, expected_feature_count: int,
    expected_feature_list_sha256: str, expected_model_sha256: str,
) -> tuple[model_compat.LoadedBinaryClassifier, xgb.Booster, dict[str, Any]]:
    model_path = Path(model_path)
    before = sha256_file(model_path)
    if before != expected_model_sha256:
        raise Amendment06IntegrityError("External scoring model differs before independent loads")
    compatibility = model_compat.load_binary_classifier_compat(
        model_path,
        expected_feature_count=expected_feature_count,
        expected_feature_list_sha256=expected_feature_list_sha256,
        expected_model_sha256=expected_model_sha256,
        device="cuda",
    )
    raw_booster = xgb.Booster()
    raw_booster.load_model(model_path)
    raw_booster.set_param({"device": "cuda"})
    after = sha256_file(model_path)
    if after != before:
        raise Amendment06IntegrityError("Independent external model loads changed UBJ bytes")
    if raw_booster is compatibility.classifier.get_booster():
        raise Amendment06IntegrityError("Raw Booster and compatibility classifier are not independent loads")
    return compatibility, raw_booster, {
        "status": "PASS", "compatibility_loader": "XGBClassifier.load_model",
        "raw_booster_loader": "INDEPENDENT_XGB_BOOSTER_LOAD_MODEL",
        "independent_objects": True,
        "model_sha256_before": before, "model_sha256_after": after,
    }


def _score_one_external_variant_impl(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    development: DevelopmentPopulation, audit: ExternalAuditPopulation,
    variant: str,
) -> dict[str, Any]:
    completion_relative = _scoring_completion_relative(variant)
    lineage = _scientific_lineage_bindings(run_dir)
    if (run_dir / completion_relative).exists():
        result = validate_external_scoring_completion(run_dir=run_dir, audit=audit, variant=variant)
        _ensure_unique_bound_journal_event(
            run_dir=run_dir, event_name="AUDIT_SCORING_COMPLETION_COMMITTED",
            binding_path=run_dir / completion_relative,
            binding_field="completion_sha256", phase="AUDIT_SCORING",
            variant=variant, recovered=True, extra={
                "attempt_id": _phase_publication_records(
                    run_dir, "AUDIT_SCORING", variant,
                ).get(completion_relative, {}).get("attempt_id"),
            },
        )
        _journal_event(run_dir, "COMPLETED_AUDIT_SCORING_SKIPPED", phase="AUDIT_SCORING", variant_id=variant)
        return result
    _record_interrupted_attempts(
        run_dir=run_dir, phase="AUDIT_SCORING", variant=variant,
    )
    _recover_unpublished_paths(
        run_dir=run_dir, phase="AUDIT_SCORING", variant=variant,
        completion_relative=completion_relative,
        possible_relatives=_scoring_relatives(variant),
    )
    validate_training_completion(
        run_dir=run_dir, locked=locked, cache=cache,
        development=development, variant=variant, active_prediction=False,
    )
    attempt_id = uuid.uuid4().hex
    staging = _attempt_root(run_dir, "AUDIT_SCORING", variant, attempt_id)
    staging.mkdir(parents=True)
    _journal_event(run_dir, "AUDIT_SCORING_ATTEMPT_STARTED", phase="AUDIT_SCORING", variant_id=variant, attempt_id=attempt_id)
    _progress("EXTERNAL_AUDIT_VARIANT_SCORING", "START", variant=variant, index=f"{TRAINED_VARIANTS.index(variant)+1}/10")
    features = tuple(locked.variant_features[variant]["retained_features"])
    indices = tuple(cache.full_features.index(feature) for feature in features)
    matrix = audit.full_X if indices == tuple(range(93)) else np.ascontiguousarray(audit.full_X[:, indices], dtype=np.float32)
    training_completion = strict_full_load_file(run_dir / _training_completion_relative(variant))
    model_path = run_dir / f"models/{variant}/model.ubj"
    loaded, independent_raw_booster, independent_load = load_independent_external_predictors(
        model_path,
        expected_feature_count=len(features),
        expected_feature_list_sha256=_feature_list_sha256(features),
        expected_model_sha256=training_completion["model_sha256"],
    )
    interval = tuple(map(int, training_completion["iteration_range"]))
    import cupy as cp
    device_matrix = cp.asarray(matrix)
    with warnings.catch_warnings(record=True) as wrapper_caught:
        warnings.simplefilter("always")
        wrapper_probability = loaded.classifier.predict_proba(
            device_matrix, iteration_range=interval,
        )[:, 1]
    with warnings.catch_warnings(record=True) as raw_caught:
        warnings.simplefilter("always")
        raw_probability = independent_raw_booster.inplace_predict(
            device_matrix, iteration_range=interval,
        )
    wrapper_host = np.ascontiguousarray(cp.asnumpy(wrapper_probability), dtype=np.float32)
    raw_host = np.ascontiguousarray(cp.asnumpy(raw_probability), dtype=np.float32)
    warning_text = [str(item.message) for item in (*wrapper_caught, *raw_caught)]
    if _fallback_warnings(warning_text) or not np.array_equal(wrapper_host, raw_host):
        raise Amendment06IntegrityError(f"External raw/compat CUDA parity failed: {variant}")
    active_probe = core._active_reloaded_booster_cuda_probe(
        independent_raw_booster, len(features),
    )
    if active_probe.get("status") != "PASS":
        raise Amendment06IntegrityError(f"External active CUDA probe failed: {variant}")
    prediction = _external_prediction_frame(audit=audit, probabilities=wrapper_host, variant=variant)
    raw_metrics = _binary_metrics(audit.labels, wrapper_host)
    weighted_metrics = _binary_metrics(audit.labels, wrapper_host, weights=audit.weights)
    csv_bytes = deterministic_csv_gzip_bytes(prediction)
    reopened, roundtrip = validate_prediction_roundtrip(csv_bytes, prediction)
    recomputed = _binary_metrics(reopened.true_label, reopened.probability)
    if recomputed != raw_metrics:
        raise Amendment06IntegrityError(f"External saved metric recomputation differs: {variant}")
    per_card = reporting.per_card_metrics(prediction)
    per_card_bytes, per_card_roundtrip = exact_table_csv_bytes(per_card)
    macro = {
        key: float(pd.to_numeric(per_card[key], errors="coerce").mean())
        for key in reporting.PRIMARY_METRICS
    }
    metrics_payload = {
        "run_id": run_dir.name, "run_kind": RUN_KIND, "variant_id": variant,
        "status": "PASS", "primary_endpoint": "raw_candidate_level_sklearn_average_precision",
        "decision_threshold": DECISION_THRESHOLD,
        "raw_candidate_level": raw_metrics,
        "review_weighted_sensitivity": weighted_metrics,
        "per_card_macro_secondary": macro,
        "external_audit_rows": len(audit.frame), "external_audit_cards": 10,
    }
    _stage_bytes(staging, f"predictions/{variant}_external_audit.csv.gz", csv_bytes)
    _stage_json(staging, f"metrics/{variant}.json", metrics_payload)
    _stage_bytes(staging, f"metrics/{variant}_per_card.csv", per_card_bytes)
    _stage_bytes(
        staging, f"models/{variant}/STATUS.txt",
        _variant_complete_status(run_dir.name, variant),
    )
    artifacts = [
        _publication_record(staging / relative, relative)
        for relative in _scoring_relatives(variant)
    ]
    scoring_manifest = {
        "run_id": run_dir.name, "run_kind": RUN_KIND,
        "variant_id": variant,
        "phase": "EXTERNAL_AUDIT_SCORING_COMPLETE", "status": "PASS",
        **lineage,
        "external_audit_rows": 10_097, "external_audit_cards": 10,
        "external_audit_sha256": EXTERNAL_AUDIT_SHA256,
        "decision_threshold": DECISION_THRESHOLD,
        "cpu_fallback_detected": False,
        "raw_booster_parity_max_abs_diff": 0.0,
        "compatibility_loader_parity_max_abs_diff": 0.0,
        "independent_raw_booster_load": True,
        "independent_load_evidence": independent_load,
        "model_sha256_before_independent_load": independent_load["model_sha256_before"],
        "model_sha256_after_independent_load": sha256_file(model_path),
        "active_reloaded_cuda_probe": active_probe,
        "prediction_sha256": sha256_bytes(csv_bytes),
        "prediction_rows": len(prediction),
        "probability_sha256_float32": sha256_array(wrapper_host.astype("<f4", copy=False)),
        "raw_candidate_metrics": raw_metrics,
        "review_weighted_metrics": weighted_metrics,
        "csv_roundtrip": roundtrip,
        "per_card_csv_roundtrip": per_card_roundtrip,
        "training_completion_sha256": sha256_file(run_dir / _training_completion_relative(variant)),
        "artifact_records": artifacts,
    }
    _stage_json(staging, completion_relative, scoring_manifest)
    try:
        for relative in _scoring_relatives(variant):
            _publish_attempt_file(
                run_dir=run_dir, staging=staging, relative=relative,
                phase="AUDIT_SCORING", variant=variant, attempt_id=attempt_id,
            )
        _publish_attempt_file(
            run_dir=run_dir, staging=staging, relative=completion_relative,
            phase="AUDIT_SCORING", variant=variant, attempt_id=attempt_id,
        )
    except Exception as exc:
        _journal_event(
            run_dir, "AUDIT_SCORING_PUBLICATION_INTERRUPTED", phase="AUDIT_SCORING",
            variant_id=variant, attempt_id=attempt_id,
            error_type=type(exc).__name__, error=str(exc),
        )
        raise
    _ensure_unique_bound_journal_event(
        run_dir=run_dir, event_name="AUDIT_SCORING_COMPLETION_COMMITTED",
        binding_path=run_dir / completion_relative,
        binding_field="completion_sha256", phase="AUDIT_SCORING",
        variant=variant, recovered=False, extra={"attempt_id": attempt_id},
    )
    _progress("EXTERNAL_AUDIT_VARIANT_SCORING", "COMPLETE", variant=variant, status="PASS")
    return validate_external_scoring_completion(run_dir=run_dir, audit=audit, variant=variant)


def score_one_external_variant(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    development: DevelopmentPopulation, audit: ExternalAuditPopulation,
    variant: str,
) -> dict[str, Any]:
    before = len(_scientific_journal_events(run_dir))
    try:
        return _score_one_external_variant_impl(
            run_dir=run_dir, locked=locked, cache=cache,
            development=development, audit=audit, variant=variant,
        )
    except Exception as exc:
        events = _scientific_journal_events(run_dir)[before:]
        starts = [event for event in events if event.get("event") == "AUDIT_SCORING_ATTEMPT_STARTED"]
        if starts:
            attempt_id = str(starts[-1]["attempt_id"])
            if not any(
                event.get("event") == "AUDIT_SCORING_ATTEMPT_FAILED"
                and event.get("attempt_id") == attempt_id
                for event in events
            ):
                _journal_event(
                    run_dir, "AUDIT_SCORING_ATTEMPT_FAILED", phase="AUDIT_SCORING",
                    failure_phase="POST_START_THROUGH_COMPLETION_COMMIT",
                    variant_id=variant, attempt_id=attempt_id,
                    error_type=type(exc).__name__, error=str(exc),
                )
        raise


def validate_external_scoring_completion(
    *, run_dir: Path, audit: ExternalAuditPopulation, variant: str,
) -> dict[str, Any]:
    if variant not in TRAINED_VARIANTS:
        raise Amendment06IntegrityError(f"Unauthorized scored Variant: {variant}")
    path = run_dir / _scoring_completion_relative(variant)
    if not _safe_regular(path):
        raise Amendment06IntegrityError(f"External scoring completion is absent: {variant}")
    payload = strict_full_load_file(path)
    training_path = run_dir / _training_completion_relative(variant)
    model_path = run_dir / f"models/{variant}/model.ubj"
    training = strict_full_load_file(training_path)
    model_sha256 = sha256_file(model_path)
    independent = payload.get("independent_load_evidence")
    active_probe = payload.get("active_reloaded_cuda_probe")
    lineage = _scientific_lineage_bindings(run_dir)
    if not (
        payload.get("run_id") == run_dir.name and payload.get("run_kind") == RUN_KIND
        and payload.get("variant_id") == variant
        and payload.get("phase") == "EXTERNAL_AUDIT_SCORING_COMPLETE"
        and payload.get("status") == "PASS"
        and payload.get("external_audit_rows") == 10_097
        and payload.get("external_audit_cards") == 10
        and payload.get("external_audit_sha256") == EXTERNAL_AUDIT_SHA256
        and payload.get("decision_threshold") == DECISION_THRESHOLD
        and payload.get("cpu_fallback_detected") is False
        and payload.get("raw_booster_parity_max_abs_diff") == 0.0
        and payload.get("compatibility_loader_parity_max_abs_diff") == 0.0
        and payload.get("independent_raw_booster_load") is True
        and isinstance(independent, Mapping)
        and independent.get("status") == "PASS"
        and independent.get("compatibility_loader") == "XGBClassifier.load_model"
        and independent.get("raw_booster_loader") == "INDEPENDENT_XGB_BOOSTER_LOAD_MODEL"
        and independent.get("independent_objects") is True
        and independent.get("model_sha256_before") == model_sha256
        and independent.get("model_sha256_after") == model_sha256
        and payload.get("model_sha256_before_independent_load") == model_sha256
        and payload.get("model_sha256_after_independent_load") == model_sha256
        and training.get("model_sha256") == model_sha256
        and payload.get("training_completion_sha256") == sha256_file(training_path)
        and all(
            payload.get(key) == lineage[key]
            for key in (
                "base_scientific_execution_code_manifest_sha256",
                "effective_scientific_execution_code_manifest_sha256",
                "amendment08_code_corrigendum_sha256",
                "amendment08_journal_contamination_adjudication_sha256",
            )
        )
        and isinstance(active_probe, Mapping)
        and active_probe.get("status") == "PASS"
        and "cuda" in str(active_probe.get("device", "")).lower()
        and active_probe.get("warnings") == []
    ):
        raise Amendment06IntegrityError(f"External scoring completion binding differs: {variant}")
    _validate_exact_artifact_records(
        run_dir=run_dir, value=payload.get("artifact_records"),
        expected_relatives=_scoring_relatives(variant),
        label=f"Completed external scoring {variant}",
    )
    prediction_path = run_dir / f"predictions/{variant}_external_audit.csv.gz"
    prediction = pd.read_csv(prediction_path, engine="c", float_precision="round_trip")
    prediction = prediction.loc[:, list(PREDICTION_COLUMNS)]
    _, roundtrip = validate_prediction_roundtrip(prediction_path, prediction)
    raw = _binary_metrics(prediction.true_label, prediction.probability)
    weighted = _binary_metrics(prediction.true_label, prediction.probability, weights=prediction.review_weight)
    per_card = reporting.per_card_metrics(prediction)
    per_card_path = run_dir / f"metrics/{variant}_per_card.csv"
    per_card_roundtrip = validate_exact_table_csv(per_card_path, per_card)
    macro = {
        key: float(pd.to_numeric(per_card[key], errors="coerce").mean())
        for key in reporting.PRIMARY_METRICS
    }
    metrics_expected = {
        "run_id": run_dir.name, "run_kind": RUN_KIND, "variant_id": variant,
        "status": "PASS",
        "primary_endpoint": "raw_candidate_level_sklearn_average_precision",
        "decision_threshold": DECISION_THRESHOLD,
        "raw_candidate_level": raw,
        "review_weighted_sensitivity": weighted,
        "per_card_macro_secondary": macro,
        "external_audit_rows": len(audit.frame), "external_audit_cards": 10,
    }
    metrics_path = run_dir / f"metrics/{variant}.json"
    metrics_payload = strict_full_load_file(metrics_path)
    if not (
        len(prediction) == len(audit.frame)
        and np.array_equal(
            prediction.stable_candidate_id.astype(str).to_numpy(),
            audit.frame.stable_candidate_id.astype(str).to_numpy(),
        )
        and np.array_equal(
            prediction.card_id.astype(str).to_numpy(),
            audit.frame.card_id.astype(str).to_numpy(),
        )
        and np.array_equal(prediction.true_label.to_numpy(np.int8), audit.labels)
        and np.array_equal(
            prediction.review_weight.to_numpy(np.float64), audit.weights.astype(np.float64),
        )
        and np.array_equal(
            prediction.prediction_at_0_5.to_numpy(np.int8),
            (prediction.probability.to_numpy(np.float64) >= DECISION_THRESHOLD).astype(np.int8),
        )
        and np.array_equal(
            prediction.row_order.to_numpy(np.int64), np.arange(len(prediction)),
        )
        and set(prediction.variant_id.astype(str)) == {variant}
        and set(prediction.split_population.astype(str)) == {
            "external_audit:locked_10_card"
        }
        and payload.get("prediction_sha256") == sha256_file(prediction_path)
        and payload.get("prediction_rows") == len(prediction)
        and payload.get("probability_sha256_float32")
        == sha256_array(prediction.probability.to_numpy(np.float32).astype("<f4", copy=False))
        and raw == payload.get("raw_candidate_metrics")
        and weighted == payload.get("review_weighted_metrics")
        and payload.get("csv_roundtrip") == roundtrip
        and payload.get("per_card_csv_roundtrip") == per_card_roundtrip
        and roundtrip.get("status") == "PASS"
        and per_card_roundtrip.get("status") == "PASS"
        and metrics_payload == metrics_expected
        and (run_dir / f"models/{variant}/STATUS.txt").read_bytes()
        == _variant_complete_status(run_dir.name, variant)
    ):
        raise Amendment06IntegrityError(f"External scoring metric/row recomputation differs: {variant}")
    feature_path = run_dir / f"models/{variant}/feature_list.txt"
    features = tuple(feature_path.read_text(encoding="utf-8").splitlines())
    expected_count, expected_feature_sha = VARIANT_FEATURE_CONTRACT[variant]
    full_features = tuple(core.get_r92_feature_order())
    if not (
        len(full_features) == 93 and len(set(full_features)) == 93
        and len(features) == expected_count
        and _feature_list_sha256(features) == expected_feature_sha
    ):
        raise Amendment06IntegrityError(f"External scoring feature contract differs: {variant}")
    indices = tuple(full_features.index(feature) for feature in features)
    matrix_host = (
        audit.full_X if indices == tuple(range(93))
        else np.ascontiguousarray(audit.full_X[:, indices], dtype=np.float32)
    )
    loaded, raw_booster, live_load = load_independent_external_predictors(
        model_path, expected_feature_count=expected_count,
        expected_feature_list_sha256=expected_feature_sha,
        expected_model_sha256=model_sha256,
    )
    import cupy as cp
    device_matrix = cp.asarray(np.ascontiguousarray(matrix_host, dtype=np.float32))
    interval = tuple(map(int, training["iteration_range"]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapper_live = loaded.classifier.predict_proba(
            device_matrix, iteration_range=interval,
        )[:, 1]
        raw_live = raw_booster.inplace_predict(device_matrix, iteration_range=interval)
        cp.cuda.Stream.null.synchronize()
    wrapper_host = np.ascontiguousarray(cp.asnumpy(wrapper_live), dtype=np.float32)
    raw_host = np.ascontiguousarray(cp.asnumpy(raw_live), dtype=np.float32)
    live_warnings = [str(item.message) for item in caught]
    live_probe = core._active_reloaded_booster_cuda_probe(raw_booster, expected_count)
    saved_probability = prediction.probability.to_numpy(np.float32)
    if not (
        live_load == independent
        and not _fallback_warnings(live_warnings)
        and np.array_equal(wrapper_host, raw_host)
        and np.array_equal(wrapper_host, saved_probability)
        and live_probe.get("status") == "PASS"
        and "cuda" in str(live_probe.get("device", "")).lower()
        and live_probe.get("warnings") == []
        and sha256_file(model_path) == model_sha256
    ):
        raise Amendment06IntegrityError(
            f"External scoring live model/prediction parity differs: {variant}"
        )
    return {
        "status": "PASS", "variant_id": variant,
        "scoring_completion_sha256": sha256_file(path),
        "prediction": prediction,
        "raw_metrics": raw, "weighted_metrics": weighted,
        "per_card": per_card,
    }


def complete_external_audit_phase(
    *, run_dir: Path, audit: ExternalAuditPopulation,
) -> list[dict[str, Any]]:
    results = [
        validate_external_scoring_completion(run_dir=run_dir, audit=audit, variant=variant)
        for variant in TRAINED_VARIANTS
    ]
    if _SCIENTIFIC_COUNTERS["post_audit_training_calls"] != 0:
        raise Amendment06IntegrityError("A scientific training call occurred after Audit opening")
    current = _current_run_state(run_dir)
    if current == "FULL_MODELS_FROZEN":
        _set_run_state(run_dir, "FULL_EXTERNAL_AUDIT_SCORED")
    elif RUN_PHASES.index(current) < RUN_PHASES.index("FULL_EXTERNAL_AUDIT_SCORED"):
        raise Amendment06IntegrityError("Audit completion/state disagree")
    return results


def initialize_run_evidence(run_dir: Path) -> None:
    _snapshot_source_pre(run_dir)
    probe_source = ACCEPTED_PREFLIGHT / "provenance/cuda_active_fit_probe.json"
    probe_destination = run_dir / "provenance/cuda_active_fit_probe.json"
    probe_bytes = probe_source.read_bytes()
    if probe_destination.exists():
        if not _safe_regular(probe_destination, sha256_bytes(probe_bytes)):
            raise Amendment06IntegrityError("Full CUDA active-fit probe snapshot drifted")
    else:
        publish_bytes_no_clobber(probe_destination, probe_bytes)
    environment_path = run_dir / "provenance/cuda_environment.txt"
    if not environment_path.exists():
        command = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ], check=True, capture_output=True, text=True,
        )
        text_value = (
            f"RUN_ID={run_dir.name}\nPYTHON={Path(sys.executable).resolve()}\n"
            f"XGBOOST_VERSION={xgb.__version__}\nSKLEARN_VERSION={sklearn.__version__}\n"
            f"XGBOOST_USE_CUDA={xgb.build_info().get('USE_CUDA')}\n"
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<UNSET>')}\n"
            f"ABLATION_RUNTIME_ROOT={os.environ.get('ABLATION_RUNTIME_ROOT', '')}\n"
            f"NVIDIA_SMI={command.stdout.strip()}\n"
        )
        publish_bytes_no_clobber(environment_path, text_value.encode("utf-8"))


def _reporting_completion_path(run_dir: Path) -> Path:
    return run_dir / "provenance/reporting_completion_manifest.json"


REPORTING_REQUIRED_RELATIVES = frozenset({
    "tables/primary_metrics_official.csv",
    "tables/primary_metrics_exploratory.csv",
    "tables/review_weighted_sensitivity_official.csv",
    "tables/review_weighted_sensitivity_exploratory.csv",
    "tables/variant_minus_full_official.csv",
    "tables/variant_minus_full_exploratory.csv",
    "tables/paired_bootstrap_ci_official.csv",
    "tables/paired_bootstrap_ci_exploratory.csv",
    "tables/per_card_metrics_official.csv",
    "tables/per_card_metrics_exploratory.csv",
    "tables/official_blocked_variants.csv",
    "tables/class_balance_and_augmentation_stages.csv",
    "tables/training_resource_timing.csv",
    "tables/feature_and_resampling_parity.csv",
    "tables/r92_historical_control.csv",
    "tables/primary_metrics_official.tex",
    "tables/paired_bootstrap_ci_official.tex",
    "figures/official_average_precision.png",
    "figures/official_delta_average_precision.png",
    "figures/exploratory_average_precision.png",
    "docs/METHODS_DRAFT.md",
    "docs/RESULTS_DRAFT.md",
    "docs/REVIEWER_RESPONSE_DRAFT.md",
    "docs/REPRODUCE.md",
    "docs/WARNINGS_AND_LIMITATIONS.md",
    "statistics/paired_card_bootstrap.json",
    "metrics/r92_historical_control.json",
})


def _validate_reporting_completion_receipt(run_dir: Path) -> dict[str, Any]:
    path = _reporting_completion_path(run_dir)
    payload = strict_full_load_file(path)
    if not (
        payload.get("run_id") == run_dir.name
        and payload.get("run_kind") == RUN_KIND
        and payload.get("phase") == "SCIENTIFIC_REPORTING_COMPLETE"
        and payload.get("status") == "PASS"
        and payload.get("bootstrap_replicates") == BOOTSTRAP_REPLICATES
        and payload.get("bootstrap_seed") == BOOTSTRAP_SEED
        and payload.get("model_retraining") is False
        and payload.get("report_csv_roundtrip", {}).get("status") == "PASS"
        and payload.get("report_artifact_count") == 25
    ):
        raise Amendment06IntegrityError("Scientific reporting completion differs")
    records = payload.get("artifact_records", [])
    if not isinstance(records, list) or {
        str(record.get("relative_path")) for record in records
        if isinstance(record, Mapping)
    } != REPORTING_REQUIRED_RELATIVES or len(records) != len(REPORTING_REQUIRED_RELATIVES):
        raise Amendment06IntegrityError("Scientific reporting artifact set/count differs")
    for record in records:
        relative = str(record.get("relative_path"))
        member = run_dir / _safe_run_relative(relative)
        if not (
            _safe_regular(member, str(record.get("sha256")))
            and member.stat().st_size == int(record.get("size_bytes", -1))
        ):
            raise Amendment06IntegrityError(f"Completed reporting artifact drifted: {relative}")
    bootstrap = strict_full_load_file(run_dir / "statistics/paired_card_bootstrap.json")
    if not (
        bootstrap.get("status") == "PASS"
        and bootstrap.get("replicates") == BOOTSTRAP_REPLICATES
        and bootstrap.get("seed") == BOOTSTRAP_SEED
        and bootstrap.get("model_retraining") is False
    ):
        raise Amendment06IntegrityError("Paired bootstrap artifact differs")
    return payload


def _validate_reporting_completion(run_dir: Path) -> dict[str, Any]:
    payload = _validate_reporting_completion_receipt(run_dir)
    _recompute_and_validate_reporting_outputs(run_dir, payload)
    return payload


def _recompute_and_validate_reporting_outputs(
    run_dir: Path, completion: Mapping[str, Any],
) -> dict[str, Any]:
    predictions: dict[str, pd.DataFrame] = {}
    metrics_by_variant: dict[str, dict[str, Any]] = {}
    per_card_by_variant: dict[str, pd.DataFrame] = {}
    timing_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    for variant in TRAINED_VARIANTS:
        training = strict_full_load_file(run_dir / _training_completion_relative(variant))
        scoring = strict_full_load_file(run_dir / _scoring_completion_relative(variant))
        prediction_path = run_dir / f"predictions/{variant}_external_audit.csv.gz"
        prediction = pd.read_csv(
            prediction_path, engine="c", float_precision="round_trip",
        ).loc[:, list(PREDICTION_COLUMNS)]
        reporting.validate_prediction_csv_roundtrip(
            prediction_path, prediction,
            metrics=scoring["raw_candidate_metrics"], compressed=True,
        )
        raw_metrics = _binary_metrics(
            prediction.true_label, prediction.probability,
        )
        weighted_metrics = _binary_metrics(
            prediction.true_label, prediction.probability,
            weights=prediction.review_weight,
        )
        if (
            raw_metrics != scoring["raw_candidate_metrics"]
            or weighted_metrics != scoring["review_weighted_metrics"]
        ):
            raise Amendment06IntegrityError(
                f"Saved reporting metrics differ from prediction recomputation: {variant}"
            )
        predictions[variant] = prediction
        per_card = reporting.per_card_metrics(prediction)
        validate_exact_table_csv(run_dir / f"metrics/{variant}_per_card.csv", per_card)
        per_card_by_variant[variant] = per_card
        metrics_by_variant[variant] = {
            "feature_count": training["feature_count"],
            "best_iteration": training["best_iteration"],
            "external_raw": raw_metrics,
            "external_weighted_sensitivity": weighted_metrics,
        }
        timing_rows.append(strict_full_load_file(run_dir / f"timing/{variant}.json"))
        parity_rows.append({
            "variant": variant,
            "classification": training["variant_classification"],
            "feature_count": training["feature_count"],
            "feature_list_sha256": training["feature_list_sha256"],
            "population": (
                "post_jitter_pre_smote" if variant == "no_safe_smote"
                else "post_safe_smote"
            ),
            "frozen_population_sha256": training["frozen_population_sha256"],
            "four_way_parity_max_abs_diff": training["four_way_parity_max_abs_diff"],
            "external_raw_booster_parity_max_abs_diff": scoring[
                "raw_booster_parity_max_abs_diff"
            ],
            "projection_only": True,
        })
    cards = sorted(predictions["full_new_reference"].card_id.astype(str).unique())
    bootstrap = reporting.paired_card_bootstrap(
        predictions, cards, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
    )
    if bootstrap != strict_full_load_file(
        run_dir / "statistics/paired_card_bootstrap.json"
    ):
        raise Amendment06IntegrityError("Saved paired bootstrap differs from exact recomputation")
    cache_manifest = strict_full_load_file(
        run_dir / "provenance/frozen_full_resampling_manifest.json"
    )
    stage_rows = [
        {"stage": stage, **cache_manifest["stage_counts"][stage]}
        for stage in EXPECTED_STAGE_COUNTS
    ]
    r92_control = strict_full_load_file(run_dir / "metrics/r92_historical_control.json")
    staging = (
        EXPECTED_RUNTIME_ROOT / "report_validation" / run_dir.name / uuid.uuid4().hex
    )
    staging.mkdir(parents=True)
    try:
        evidence = reporting.build_all_reports(
            staging,
            metrics_by_variant=metrics_by_variant,
            per_card_by_variant=per_card_by_variant,
            bootstrap=bootstrap,
            stage_table=pd.DataFrame(stage_rows),
            timing_table=pd.DataFrame(timing_rows),
            parity_table=pd.DataFrame(parity_rows),
            r92_control=r92_control,
        )
        generated = {
            str(item["relative_path"]): item for item in evidence["artifacts"]
        }
        report_relatives = REPORTING_REQUIRED_RELATIVES - {
            "statistics/paired_card_bootstrap.json", "metrics/r92_historical_control.json",
        }
        if set(generated) != report_relatives or evidence.get("artifact_count") != 25:
            raise Amendment06IntegrityError("Regenerated report member set/count differs")
        for relative, record in generated.items():
            live = run_dir / relative
            regenerated = staging / relative
            if not (
                _safe_regular(live, str(record["sha256"]))
                and live.stat().st_size == int(record["size_bytes"])
                and live.read_bytes() == regenerated.read_bytes()
            ):
                raise Amendment06IntegrityError(
                    f"Regenerated report bytes differ: {relative}"
                )
        if completion.get("report_csv_roundtrip") != evidence.get("csv_roundtrip"):
            raise Amendment06IntegrityError("Reporting CSV-roundtrip evidence differs")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "status": "PASS", "bootstrap_recomputed": True,
        "report_artifacts_regenerated": 25,
    }


def _build_reports_and_bootstrap_impl(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    audit: ExternalAuditPopulation, scoring_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if _reporting_completion_path(run_dir).exists():
        result = _validate_reporting_completion(run_dir)
        _ensure_unique_bound_journal_event(
            run_dir=run_dir, event_name="REPORTING_COMPLETION_COMMITTED",
            binding_path=_reporting_completion_path(run_dir),
            binding_field="completion_sha256", phase="REPORTING",
            variant="GLOBAL", recovered=True, extra={
                "attempt_id": _phase_publication_records(
                    run_dir, "REPORTING", "GLOBAL",
                ).get("provenance/reporting_completion_manifest.json", {}).get(
                    "attempt_id"
                ),
            },
        )
        _journal_event(run_dir, "COMPLETED_REPORTING_SKIPPED", phase="REPORTING")
        return result
    _record_interrupted_attempts(
        run_dir=run_dir, phase="REPORTING", variant="GLOBAL",
    )
    owned = _phase_publication_records(run_dir, "REPORTING", "GLOBAL")
    _recover_unpublished_paths(
        run_dir=run_dir, phase="REPORTING", variant="GLOBAL",
        completion_relative="provenance/reporting_completion_manifest.json",
        possible_relatives=tuple(owned),
    )
    attempt_id = uuid.uuid4().hex
    staging = _attempt_root(run_dir, "REPORTING", "GLOBAL", attempt_id)
    staging.mkdir(parents=True)
    _journal_event(
        run_dir, "REPORTING_ATTEMPT_STARTED", phase="REPORTING",
        variant_id="GLOBAL", attempt_id=attempt_id,
    )
    _progress("PAIRED_CARD_BOOTSTRAP", "START", replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    result_by_variant = {str(item["variant_id"]): item for item in scoring_results}
    if set(result_by_variant) != set(TRAINED_VARIANTS):
        raise Amendment06IntegrityError("Reporting scoring-result Variant set differs")
    predictions = {
        variant: result_by_variant[variant]["prediction"]
        for variant in TRAINED_VARIANTS
    }
    cards = sorted(audit.frame.card_id.astype(str).unique())
    bootstrap = reporting.paired_card_bootstrap(
        predictions, cards, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
    )
    if bootstrap.get("retraining") is not False or bootstrap.get("model_retraining") is not False:
        raise Amendment06IntegrityError("Bootstrap attempted model retraining")
    bootstrap_sha256 = sha256_bytes(strict_full_canonical_json_bytes(bootstrap))
    bootstrap_events = [
        event for event in _scientific_journal_events(run_dir)
        if event.get("event") == "BOOTSTRAP_RESAMPLING"
    ]
    expected_bootstrap_event = {
        "phase": "REPORTING", "status": "PASS",
        "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
        "model_retraining": False, "bootstrap_sha256": bootstrap_sha256,
    }
    if len(bootstrap_events) > 1 or (
        bootstrap_events and any(
            bootstrap_events[0].get(key) != value
            for key, value in expected_bootstrap_event.items()
        )
    ):
        raise Amendment06IntegrityError("Paired-bootstrap journal binding differs")
    if not bootstrap_events:
        _journal_event(
            run_dir, "BOOTSTRAP_RESAMPLING", **expected_bootstrap_event,
        )
    _progress("PAIRED_CARD_BOOTSTRAP", "COMPLETE", status="PASS")
    _progress("SCIENTIFIC_REPORTING", "START")
    r92_control, _ = core.score_r92_control(audit.frame)
    if r92_control.get("status") != "PASS":
        raise Amendment06IntegrityError("Historical r92 control replay failed after model freeze")
    # Use the strict-JSON reopened key order at first publication so byte-for-byte
    # report regeneration after resume receives the same mapping representation.
    r92_control = strict_full_loads(strict_full_json_bytes(r92_control))
    metrics_by_variant: dict[str, dict[str, Any]] = {}
    per_card_by_variant: dict[str, pd.DataFrame] = {}
    timing_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    for variant in TRAINED_VARIANTS:
        scoring = result_by_variant[variant]
        training = strict_full_load_file(run_dir / _training_completion_relative(variant))
        metrics_by_variant[variant] = {
            "feature_count": training["feature_count"],
            "best_iteration": training["best_iteration"],
            "external_raw": scoring["raw_metrics"],
            "external_weighted_sensitivity": scoring["weighted_metrics"],
        }
        per_card_by_variant[variant] = scoring["per_card"]
        timing_rows.append(strict_full_load_file(run_dir / f"timing/{variant}.json"))
        parity_rows.append({
            "variant": variant,
            "classification": training["variant_classification"],
            "feature_count": training["feature_count"],
            "feature_list_sha256": training["feature_list_sha256"],
            "population": (
                "post_jitter_pre_smote" if variant == "no_safe_smote" else "post_safe_smote"
            ),
            "frozen_population_sha256": cache.binding_sha256,
            "four_way_parity_max_abs_diff": training["four_way_parity_max_abs_diff"],
            "external_raw_booster_parity_max_abs_diff": 0.0,
            "projection_only": True,
        })
    canonical_cache_manifest = strict_full_loads(strict_full_json_bytes(cache.manifest))
    stage_rows = []
    for stage in EXPECTED_STAGE_COUNTS:
        stage_rows.append({
            "stage": stage, **canonical_cache_manifest["stage_counts"][stage],
        })
    report_evidence = reporting.build_all_reports(
        staging,
        metrics_by_variant=metrics_by_variant,
        per_card_by_variant=per_card_by_variant,
        bootstrap=bootstrap,
        stage_table=pd.DataFrame(stage_rows),
        timing_table=pd.DataFrame(timing_rows),
        parity_table=pd.DataFrame(parity_rows),
        r92_control=r92_control,
    )
    if report_evidence.get("csv_roundtrip", {}).get("status") != "PASS":
        raise Amendment06IntegrityError("Generated report CSV exact roundtrip failed")
    _stage_json(staging, "statistics/paired_card_bootstrap.json", bootstrap)
    _stage_json(staging, "metrics/r92_historical_control.json", r92_control)
    relatives = [
        *(str(item["relative_path"]) for item in report_evidence["artifacts"]),
        "statistics/paired_card_bootstrap.json",
        "metrics/r92_historical_control.json",
    ]
    artifact_records = [_publication_record(staging / relative, relative) for relative in relatives]
    completion = {
        "run_id": run_dir.name, "run_kind": RUN_KIND,
        "phase": "SCIENTIFIC_REPORTING_COMPLETE", "status": "PASS",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED, "model_retraining": False,
        "report_artifact_count": report_evidence["artifact_count"],
        "report_csv_roundtrip": report_evidence["csv_roundtrip"],
        "official_runnable_count": 9, "official_blocked_true_no_pca_count": 1,
        "exploratory_count": 1, "delta_definition": "variant_minus_full",
        "artifact_records": artifact_records,
    }
    _stage_json(staging, "provenance/reporting_completion_manifest.json", completion)
    try:
        for relative in relatives:
            _publish_attempt_file(
                run_dir=run_dir, staging=staging, relative=relative,
                phase="REPORTING", variant="GLOBAL", attempt_id=attempt_id,
            )
        _publish_attempt_file(
            run_dir=run_dir, staging=staging,
            relative="provenance/reporting_completion_manifest.json",
            phase="REPORTING", variant="GLOBAL", attempt_id=attempt_id,
        )
    except Exception as exc:
        _journal_event(
            run_dir, "REPORTING_PUBLICATION_INTERRUPTED", phase="REPORTING",
            variant_id="GLOBAL", attempt_id=attempt_id,
            error_type=type(exc).__name__, error=str(exc),
        )
        raise
    _ensure_unique_bound_journal_event(
        run_dir=run_dir, event_name="REPORTING_COMPLETION_COMMITTED",
        binding_path=_reporting_completion_path(run_dir),
        binding_field="completion_sha256", phase="REPORTING",
        variant="GLOBAL", recovered=False, extra={"attempt_id": attempt_id},
    )
    _progress("SCIENTIFIC_REPORTING", "COMPLETE", status="PASS")
    return _validate_reporting_completion(run_dir)


def build_reports_and_bootstrap(
    *, run_dir: Path, locked: LockedInputs, cache: FrozenPopulationCache,
    audit: ExternalAuditPopulation, scoring_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    before = len(_scientific_journal_events(run_dir))
    try:
        return _build_reports_and_bootstrap_impl(
            run_dir=run_dir, locked=locked, cache=cache,
            audit=audit, scoring_results=scoring_results,
        )
    except Exception as exc:
        events = _scientific_journal_events(run_dir)[before:]
        starts = [event for event in events if event.get("event") == "REPORTING_ATTEMPT_STARTED"]
        if starts:
            attempt_id = str(starts[-1]["attempt_id"])
            if not any(
                event.get("event") == "REPORTING_ATTEMPT_FAILED"
                and event.get("attempt_id") == attempt_id
                for event in events
            ):
                _journal_event(
                    run_dir, "REPORTING_ATTEMPT_FAILED", phase="REPORTING",
                    failure_phase="POST_START_THROUGH_COMPLETION_COMMIT",
                    variant_id="GLOBAL", attempt_id=attempt_id,
                    error_type=type(exc).__name__, error=str(exc),
                )
        raise


def build_metric_recomputation(
    *, run_dir: Path, audit: ExternalAuditPopulation,
) -> dict[str, Any]:
    lineage = _scientific_lineage_bindings(run_dir)
    records: list[dict[str, Any]] = []
    for variant in TRAINED_VARIANTS:
        training_path = run_dir / _training_completion_relative(variant)
        training = strict_full_load_file(training_path)
        validation_path = run_dir / f"predictions/{variant}_validation.csv.gz"
        validation = pd.read_csv(validation_path, engine="c", float_precision="round_trip")
        validation = validation.loc[:, list(PREDICTION_COLUMNS)]
        validation_evidence = reporting.validate_prediction_csv_roundtrip(
            validation_path, validation,
            metrics=training["validation_metrics"], compressed=True,
        )
        scoring_path = run_dir / _scoring_completion_relative(variant)
        scoring = strict_full_load_file(scoring_path)
        external_path = run_dir / f"predictions/{variant}_external_audit.csv.gz"
        metrics_path = run_dir / f"metrics/{variant}.json"
        per_card_path = run_dir / f"metrics/{variant}_per_card.csv"
        external = pd.read_csv(external_path, engine="c", float_precision="round_trip")
        external = external.loc[:, list(PREDICTION_COLUMNS)]
        external_evidence = reporting.validate_prediction_csv_roundtrip(
            external_path, external,
            metrics=scoring["raw_candidate_metrics"], compressed=True,
        )
        raw = _binary_metrics(external.true_label, external.probability)
        weighted = _binary_metrics(
            external.true_label, external.probability, weights=external.review_weight,
        )
        if not (
            raw == scoring["raw_candidate_metrics"]
            and weighted == scoring["review_weighted_metrics"]
            and len(external) == len(audit.frame)
        ):
            raise Amendment06IntegrityError(f"Final external metric recomputation differs: {variant}")
        records.append({
            "variant_id": variant,
            "training_completion_sha256": sha256_file(training_path),
            "external_scoring_completion_sha256": sha256_file(scoring_path),
            "validation_prediction_sha256": sha256_file(validation_path),
            "validation_metric_recomputation": validation_evidence,
            "external_prediction_sha256": sha256_file(external_path),
            "external_metric_recomputation": external_evidence,
            "metrics_sha256": sha256_file(metrics_path),
            "per_card_metrics_sha256": sha256_file(per_card_path),
            "raw_metrics_exact": True, "weighted_metrics_exact": True,
        })
    return {
        "schema": "amendment06_metric_recomputation/v2",
        "run_id": run_dir.name, "run_kind": RUN_KIND,
        "phase": "SCIENTIFIC_METRIC_RECOMPUTATION_COMPLETE", "status": "PASS",
        **lineage,
        "reporting_completion_sha256": sha256_file(
            run_dir / "provenance/reporting_completion_manifest.json"
        ),
        "paired_card_bootstrap_sha256": sha256_file(
            run_dir / "statistics/paired_card_bootstrap.json"
        ),
        "external_audit_sha256": EXTERNAL_AUDIT_SHA256,
        "external_audit_rows": len(audit.frame),
        "performed_before_full_scientific_complete": True,
        "package_metric_recomputation_required": False,
        "model_retraining": False,
        "required_metrics_finite": True, "numeric_equality": "EXACT_NO_TOLERANCE",
        "csv_engine": "c", "float_precision": "round_trip",
        "variant_count": 10, "records": records,
    }


def _validate_metric_recomputation_receipt(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "provenance/metric_recomputation.json"
    lineage = _scientific_lineage_bindings(run_dir)
    payload = _require_exact_keys(
        strict_full_load_file(path),
        {
            "schema", "run_id", "run_kind", "phase", "status",
            *lineage,
            "reporting_completion_sha256", "paired_card_bootstrap_sha256",
            "external_audit_sha256", "external_audit_rows",
            "performed_before_full_scientific_complete",
            "package_metric_recomputation_required", "model_retraining",
            "required_metrics_finite", "numeric_equality", "csv_engine",
            "float_precision", "variant_count", "records",
        },
        label="Frozen metric-recomputation receipt",
    )
    reporting_path = run_dir / "provenance/reporting_completion_manifest.json"
    bootstrap_path = run_dir / "statistics/paired_card_bootstrap.json"
    records = payload["records"]
    if not (
        payload["schema"] == "amendment06_metric_recomputation/v2"
        and payload["run_id"] == run_dir.name and payload["run_kind"] == RUN_KIND
        and payload["phase"] == "SCIENTIFIC_METRIC_RECOMPUTATION_COMPLETE"
        and payload["status"] == "PASS"
        and all(payload.get(key) == value for key, value in lineage.items())
        and payload["reporting_completion_sha256"] == sha256_file(reporting_path)
        and payload["paired_card_bootstrap_sha256"] == sha256_file(bootstrap_path)
        and payload["external_audit_sha256"] == EXTERNAL_AUDIT_SHA256
        and payload["external_audit_rows"] == 10_097
        and payload["performed_before_full_scientific_complete"] is True
        and payload["package_metric_recomputation_required"] is False
        and payload["model_retraining"] is False
        and payload["required_metrics_finite"] is True
        and payload["numeric_equality"] == "EXACT_NO_TOLERANCE"
        and payload["csv_engine"] == "c"
        and payload["float_precision"] == "round_trip"
        and payload["variant_count"] == 10
        and isinstance(records, list)
        and all(isinstance(item, Mapping) for item in records)
        and [item.get("variant_id") for item in records] == list(TRAINED_VARIANTS)
    ):
        raise Amendment06IntegrityError("Frozen metric-recomputation receipt differs")
    record_keys = {
        "variant_id", "training_completion_sha256",
        "external_scoring_completion_sha256", "validation_prediction_sha256",
        "validation_metric_recomputation", "external_prediction_sha256",
        "external_metric_recomputation", "metrics_sha256",
        "per_card_metrics_sha256", "raw_metrics_exact", "weighted_metrics_exact",
    }
    metric_evidence_keys = {
        "status", "rows", "csv_sha256", "float_precision", "engine",
        "numeric_equality", "metric_recomputation",
    }
    for record in records:
        variant = str(record["variant_id"])
        paths = {
            "training_completion_sha256": run_dir / _training_completion_relative(variant),
            "external_scoring_completion_sha256": run_dir / _scoring_completion_relative(variant),
            "validation_prediction_sha256": run_dir / f"predictions/{variant}_validation.csv.gz",
            "external_prediction_sha256": run_dir / f"predictions/{variant}_external_audit.csv.gz",
            "metrics_sha256": run_dir / f"metrics/{variant}.json",
            "per_card_metrics_sha256": run_dir / f"metrics/{variant}_per_card.csv",
        }
        if not (
            isinstance(record, Mapping) and set(record) == record_keys
            and record["raw_metrics_exact"] is True
            and record["weighted_metrics_exact"] is True
            and all(_safe_regular(member, str(record[field])) for field, member in paths.items())
        ):
            raise Amendment06IntegrityError(
                f"Frozen metric-recomputation Variant binding differs: {variant}"
            )
        for field, rows, prediction_field in (
            ("validation_metric_recomputation", 56_843, "validation_prediction_sha256"),
            ("external_metric_recomputation", 10_097, "external_prediction_sha256"),
        ):
            evidence = record[field]
            if not (
                isinstance(evidence, Mapping)
                and set(evidence) == metric_evidence_keys
                and evidence.get("status") == "PASS"
                and evidence.get("rows") == rows
                and evidence.get("csv_sha256") == record[prediction_field]
                and evidence.get("engine") == "c"
                and evidence.get("float_precision") == "round_trip"
                and evidence.get("numeric_equality") == "EXACT"
                and evidence.get("metric_recomputation") == "PASS"
            ):
                raise Amendment06IntegrityError(
                    f"Frozen metric round-trip receipt differs: {variant}/{field}"
                )
    _validate_reporting_completion_receipt(run_dir)
    return dict(payload)


def _validate_amendment08_frozen_prerequisite_receipts(
    run_dir: Path,
) -> dict[str, Any]:
    """Validate the completed science using only immutable Run receipts and hashes."""

    run_dir = Path(run_dir).resolve()
    if run_dir.name != A08_TARGET_RUN_ID:
        raise Amendment06IntegrityError(
            "Frozen-receipt finalization is restricted to the Amendment 08 target"
        )
    lineage = _amendment08_frozen_lineage_bindings(run_dir)
    source_pre = run_dir / "provenance/source_input_hashes_pre.tsv"
    source_post = run_dir / "provenance/source_input_hashes_post.tsv"
    if not (
        _safe_regular(source_pre, SOURCE_INVENTORY_SHA256)
        and _safe_regular(source_post, SOURCE_INVENTORY_SHA256)
        and source_pre.read_bytes() == source_post.read_bytes()
    ):
        raise Amendment06IntegrityError("Frozen source inventory pre/post binding differs")

    training_hashes: dict[str, str] = {}
    scoring_hashes: dict[str, str] = {}
    model_hashes: dict[str, str] = {}
    metrics_hashes: dict[str, str] = {}
    for variant in TRAINED_VARIANTS:
        expected_classification = (
            "EXPLORATORY" if variant == EXPLORATORY_VARIANT else "OFFICIAL_RUNNABLE"
        )
        training_path = run_dir / _training_completion_relative(variant)
        training = strict_full_load_file(training_path)
        model_path = run_dir / f"models/{variant}/model.ubj"
        model_sha256 = sha256_file(model_path)
        if not (
            training.get("run_id") == run_dir.name
            and training.get("run_kind") == RUN_KIND
            and training.get("variant_id") == variant
            and training.get("variant_classification") == expected_classification
            and training.get("phase") == "TRAINING_AND_VALIDATION_COMPLETE"
            and training.get("status") == "PASS"
            and training.get("model_sha256") == model_sha256
            and training.get("model_size_bytes") == model_path.stat().st_size
            and training.get("feature_count") == VARIANT_FEATURE_CONTRACT[variant][0]
            and training.get("feature_list_sha256")
            == VARIANT_FEATURE_CONTRACT[variant][1]
            and training.get("full_parameter_lock_sha256")
            == FULL_MODEL_PARAMETERS_SHA256
            and training.get("frozen_population_sha256")
            == A07_FROZEN_POPULATION_SHA256
            and training.get("imputer_sha256") == A07_PORTABLE_IMPUTER_SHA256
            and training.get("split_assignment_sha256") == SPLIT_ASSIGNMENT_SHA256
            and training.get("seed") == PRIMARY_MODEL_SEED
            and training.get("decision_threshold") == DECISION_THRESHOLD
            and training.get("cpu_fallback_detected") is False
            and training.get("four_way_parity_max_abs_diff") == 0.0
            and training.get("source_code_manifest_sha256")
            == lineage["effective_scientific_execution_code_manifest_sha256"]
            and all(training.get(key) == value for key, value in lineage.items())
        ):
            raise Amendment06IntegrityError(
                f"Frozen training receipt differs: {variant}"
            )
        training_records = _validate_exact_artifact_records(
            run_dir=run_dir, value=training.get("artifact_records"),
            expected_relatives=_training_relatives(variant),
            label=f"Frozen training receipt {variant}",
        )
        for field, relative in (
            ("model_metadata_sha256", f"models/{variant}/model_metadata.json"),
            ("transform_binding_sha256", f"models/{variant}/transform_binding.json"),
            ("timing_resource_sha256", f"timing/{variant}.json"),
            ("validation_prediction_sha256", f"predictions/{variant}_validation.csv.gz"),
        ):
            if training.get(field) != training_records[relative]["sha256"]:
                raise Amendment06IntegrityError(
                    f"Frozen training subreceipt differs: {variant}/{field}"
                )
        training_hashes[variant] = sha256_file(training_path)
        model_hashes[variant] = model_sha256

        scoring_path = run_dir / _scoring_completion_relative(variant)
        scoring = strict_full_load_file(scoring_path)
        active_probe = scoring.get("active_reloaded_cuda_probe")
        independent = scoring.get("independent_load_evidence")
        if not (
            scoring.get("run_id") == run_dir.name
            and scoring.get("run_kind") == RUN_KIND
            and scoring.get("variant_id") == variant
            and scoring.get("phase") == "EXTERNAL_AUDIT_SCORING_COMPLETE"
            and scoring.get("status") == "PASS"
            and scoring.get("external_audit_rows") == 10_097
            and scoring.get("external_audit_cards") == 10
            and scoring.get("external_audit_sha256") == EXTERNAL_AUDIT_SHA256
            and scoring.get("decision_threshold") == DECISION_THRESHOLD
            and scoring.get("cpu_fallback_detected") is False
            and scoring.get("raw_booster_parity_max_abs_diff") == 0.0
            and scoring.get("compatibility_loader_parity_max_abs_diff") == 0.0
            and scoring.get("independent_raw_booster_load") is True
            and isinstance(independent, Mapping)
            and independent.get("status") == "PASS"
            and independent.get("independent_objects") is True
            and independent.get("model_sha256_before") == model_sha256
            and independent.get("model_sha256_after") == model_sha256
            and scoring.get("model_sha256_before_independent_load") == model_sha256
            and scoring.get("model_sha256_after_independent_load") == model_sha256
            and scoring.get("training_completion_sha256") == training_hashes[variant]
            and isinstance(active_probe, Mapping)
            and active_probe.get("status") == "PASS"
            and "cuda" in str(active_probe.get("device", "")).lower()
            and active_probe.get("warnings") == []
            and all(scoring.get(key) == value for key, value in lineage.items())
        ):
            raise Amendment06IntegrityError(
                f"Frozen external-scoring receipt differs: {variant}"
            )
        scoring_records = _validate_exact_artifact_records(
            run_dir=run_dir, value=scoring.get("artifact_records"),
            expected_relatives=_scoring_relatives(variant),
            label=f"Frozen external-scoring receipt {variant}",
        )
        prediction_relative = f"predictions/{variant}_external_audit.csv.gz"
        if not (
            scoring.get("prediction_sha256")
            == scoring_records[prediction_relative]["sha256"]
            and scoring.get("prediction_rows") == 10_097
            and (run_dir / f"models/{variant}/STATUS.txt").read_bytes()
            == _variant_complete_status(run_dir.name, variant)
        ):
            raise Amendment06IntegrityError(
                f"Frozen scoring prediction/status binding differs: {variant}"
            )
        scoring_hashes[variant] = sha256_file(scoring_path)
        metrics_hashes[variant] = sha256_file(run_dir / f"metrics/{variant}.json")

    frozen_path = run_dir / "provenance/full_models_frozen_manifest.json"
    frozen = strict_full_load_file(frozen_path)
    models = frozen.get("models")
    if not (
        frozen.get("run_id") == run_dir.name
        and frozen.get("run_kind") == RUN_KIND
        and frozen.get("phase") == "FULL_MODELS_FROZEN"
        and frozen.get("status") == "PASS"
        and frozen.get("trained_model_count") == 10
        and frozen.get("official_runnable_model_count") == 9
        and frozen.get("exploratory_model_count") == 1
        and frozen.get("true_no_pca_model_count") == 0
        and frozen.get("frozen_population_sha256") == A07_FROZEN_POPULATION_SHA256
        and frozen.get("imputer_sha256") == A07_PORTABLE_IMPUTER_SHA256
        and frozen.get("full_parameter_lock_sha256") == FULL_MODEL_PARAMETERS_SHA256
        and frozen.get("split_assignment_sha256") == SPLIT_ASSIGNMENT_SHA256
        and frozen.get("external_audit_opened") is False
        and all(frozen.get(key) == value for key, value in lineage.items())
        and isinstance(models, list)
        and [item.get("variant_id") for item in models] == list(TRAINED_VARIANTS)
    ):
        raise Amendment06IntegrityError("Frozen models manifest receipt differs")
    for item, variant in zip(models, TRAINED_VARIANTS, strict=True):
        if not (
            isinstance(item, Mapping)
            and set(item) == {
                "variant_id", "variant_classification", "model_sha256",
                "training_completion_sha256",
            }
            and item.get("model_sha256") == model_hashes[variant]
            and item.get("training_completion_sha256") == training_hashes[variant]
        ):
            raise Amendment06IntegrityError(
                f"Frozen model member receipt differs: {variant}"
            )

    audit_path = run_dir / "provenance/external_audit_exclusion_and_opening.json"
    audit = strict_full_load_file(audit_path)
    if not (
        audit.get("run_id") == run_dir.name and audit.get("run_kind") == RUN_KIND
        and audit.get("status") == "PASS"
        and audit.get("external_audit_path") == str(EXTERNAL_AUDIT)
        and audit.get("external_audit_sha256") == EXTERNAL_AUDIT_SHA256
        and audit.get("opened_after_full_models_frozen") is True
        and audit.get("models_frozen_manifest_sha256") == sha256_file(frozen_path)
        and audit.get("models_frozen_state_marker_sha256") == sha256_file(
            _state_marker_path(run_dir, "FULL_MODELS_FROZEN")
        )
        and audit.get("pre_freeze_audit_access")
        == "EXISTENCE_SIZE_SHA256_ONLY_NO_PARSE"
        and audit.get("external_audit_used_for_fitting") is False
        and audit.get("external_audit_used_for_imputer_fit") is False
        and audit.get("external_audit_used_for_early_stopping") is False
        and audit.get("external_audit_used_for_threshold_or_model_selection") is False
        and audit.get("post_audit_training_calls") == 0
    ):
        raise Amendment06IntegrityError("Frozen external-Audit opening receipt differs")

    reporting = _validate_reporting_completion_receipt(run_dir)
    metric = _validate_metric_recomputation_receipt(run_dir)
    if _current_run_state(run_dir) not in {
        "FULL_EXTERNAL_AUDIT_SCORED", "FULL_SCIENTIFIC_COMPLETE",
    }:
        raise Amendment06IntegrityError("Frozen prerequisite state is incomplete")
    return {
        "status": "PASS_FROZEN_RECEIPTS_ONLY", **lineage,
        "full_models_frozen_manifest_sha256": sha256_file(frozen_path),
        "external_audit_opening_sha256": sha256_file(audit_path),
        "reporting_completion_manifest_sha256": sha256_file(
            _reporting_completion_path(run_dir)
        ),
        "metric_recomputation_sha256": sha256_file(
            run_dir / "provenance/metric_recomputation.json"
        ),
        "source_input_hashes_post_sha256": sha256_file(source_post),
        "paired_card_bootstrap_sha256": sha256_file(
            run_dir / "statistics/paired_card_bootstrap.json"
        ),
        "training_completion_sha256": training_hashes,
        "external_scoring_completion_sha256": scoring_hashes,
        "metrics_sha256": metrics_hashes,
        "report_artifact_count": reporting["report_artifact_count"],
        "metric_variant_count": metric["variant_count"],
        "prediction_calls": 0, "metric_recomputation_calls": 0,
        "report_generation_calls": 0, "bootstrap_generation_calls": 0,
        "external_audit_tabular_parse_calls": 0,
    }


def _planned_amendment08_final_state_marker(
    run_dir: Path, *, recorded_at: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    run_dir = Path(run_dir).resolve()
    previous_path = _state_marker_path(run_dir, "FULL_EXTERNAL_AUDIT_SCORED")
    final_path = _state_marker_path(run_dir, "FULL_SCIENTIFIC_COMPLETE")
    if not _safe_regular(previous_path):
        raise Amendment06IntegrityError(
            "Amendment 08 final state lacks its previous immutable marker"
        )
    if final_path.exists():
        marker = _require_exact_keys(
            strict_full_load_file(final_path),
            {
                "run_id", "run_kind", "phase", "phase_index", "status",
                "previous_phase", "previous_marker_sha256", "recorded_at",
            },
            label="Amendment 08 final state marker",
        )
        marker_bytes = final_path.read_bytes()
        if marker_bytes != strict_full_json_bytes(marker):
            raise Amendment06IntegrityError(
                "Amendment 08 final state marker bytes are not canonical"
            )
        if recorded_at is not None and marker["recorded_at"] != recorded_at:
            raise Amendment06IntegrityError(
                "Reserved Amendment 08 final-state timestamp differs"
            )
    else:
        timestamp = _now() if recorded_at is None else recorded_at
        if not isinstance(timestamp, str) or not timestamp:
            raise Amendment06IntegrityError(
                "Amendment 08 final-state timestamp is empty"
            )
        marker = {
            "run_id": run_dir.name, "run_kind": RUN_KIND,
            "phase": "FULL_SCIENTIFIC_COMPLETE",
            "phase_index": RUN_PHASES.index("FULL_SCIENTIFIC_COMPLETE"),
            "status": "PASS", "previous_phase": "FULL_EXTERNAL_AUDIT_SCORED",
            "previous_marker_sha256": sha256_file(previous_path),
            "recorded_at": timestamp,
        }
        marker_bytes = strict_full_json_bytes(marker)
    if not (
        marker["run_id"] == run_dir.name
        and marker["run_kind"] == RUN_KIND
        and marker["phase"] == "FULL_SCIENTIFIC_COMPLETE"
        and marker["phase_index"] == 3
        and marker["status"] == "PASS"
        and marker["previous_phase"] == "FULL_EXTERNAL_AUDIT_SCORED"
        and marker["previous_marker_sha256"] == sha256_file(previous_path)
        and isinstance(marker["recorded_at"], str) and marker["recorded_at"]
    ):
        raise Amendment06IntegrityError("Planned Amendment 08 final state differs")
    return dict(marker), marker_bytes


def build_amendment08_raw_journal_cutoff(
    run_dir: Path, *, final_state_recorded_at: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    prerequisites = _validate_amendment08_frozen_prerequisite_receipts(run_dir)
    views = read_amendment08_journal_views(run_dir)
    raw_events = views["raw_events"]
    semantic_events = views["semantic_events"]
    genuine = [
        event for event in semantic_events
        if event.get("event") == "REPORTING_COMPLETION_COMMITTED"
    ]
    if not (
        len(genuine) == 1
        and views["raw_reporting_completion_events"] == 3
        and views["semantic_reporting_completion_events"] == 1
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 has no unique genuine reporting cutoff"
        )
    genuine_event = dict(genuine[0])
    cutoff_index = genuine_event.get("event_index")
    if isinstance(cutoff_index, bool) or not isinstance(cutoff_index, int):
        raise Amendment06IntegrityError("Genuine reporting physical index differs")
    if cutoff_index < 9 or cutoff_index >= len(raw_events):
        raise Amendment06IntegrityError("Genuine reporting cutoff is outside raw Journal")
    raw_prefix = [dict(event) for event in raw_events[:cutoff_index + 1]]
    semantic_prefix = [
        dict(event) for event in raw_prefix
        if int(event["event_index"]) not in A08_CONTAMINATION_INDICES
    ]
    raw_reporting = sum(
        event.get("event") == "REPORTING_COMPLETION_COMMITTED"
        for event in raw_prefix
    )
    semantic_reporting = sum(
        event.get("event") == "REPORTING_COMPLETION_COMMITTED"
        for event in semantic_prefix
    )
    if not (
        raw_prefix[-1] == genuine_event
        and raw_reporting == 3 and semantic_reporting == 1
        and len(semantic_prefix) == len(raw_prefix) - 2
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 reporting cutoff prefix differs"
        )
    lineage = _amendment08_frozen_lineage_bindings(run_dir)
    corrected_history = _validate_amendment08_corrected_training_history(
        run_dir=run_dir, overlay=lineage, allow_unterminated=False,
    )
    if not (
        corrected_history["completed_variants"] == list(TRAINED_VARIANTS)
        and corrected_history["next_variant"] is None
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 reporting cutoff training history is incomplete"
        )
    _validate_amendment08_cutoff_lifecycle(
        run_dir=run_dir, events=raw_prefix, lineage=lineage,
    )
    records: list[dict[str, Any]] = []
    for event in raw_prefix:
        physical_index = int(event["event_index"])
        path = _journal_event_path(run_dir, physical_index)
        raw = path.read_bytes() if _safe_regular(path) else b""
        if raw != strict_full_json_bytes(event):
            raise Amendment06IntegrityError(
                f"Amendment 08 raw Journal bytes differ at {physical_index}"
            )
        canonical = strict_full_canonical_json_bytes(event)
        records.append({
            "physical_event_index": physical_index,
            "relative_path": f"{physical_index:08d}.json",
            "raw_size_bytes": len(raw), "raw_sha256": sha256_bytes(raw),
            "canonical_sha256": sha256_bytes(canonical), "event": event,
            "raw_bytes_base64": base64.b64encode(raw).decode("ascii"),
        })
    raw_lines = b"".join(
        strict_full_canonical_json_bytes(event) + b"\n" for event in raw_prefix
    )
    semantic_lines = b"".join(
        strict_full_canonical_json_bytes(event) + b"\n" for event in semantic_prefix
    )
    marker, marker_bytes = _planned_amendment08_final_state_marker(
        run_dir, recorded_at=final_state_recorded_at,
    )
    receipt_path = run_dir / "provenance/amendment08_overlay_install_receipt.json"
    receipt = strict_full_load_file(receipt_path)
    genuine_path = _journal_event_path(run_dir, cutoff_index)
    reporting_path = _reporting_completion_path(run_dir)
    return {
        "schema": "amendment08_raw_journal_at_reporting_cutoff/v1",
        "status": "PASS", "authorization": A08_AUTHORIZATION,
        "prompt_sha256": A08_PROMPT_SHA256, "target_run_id": run_dir.name,
        "created_at": marker["recorded_at"],
        "record_count": len(records), "records": records,
        "raw_journal_event_count_at_reporting_cutoff": len(raw_prefix),
        "raw_journal_head_sha256_at_reporting_cutoff": sha256_bytes(
            strict_full_canonical_json_bytes(raw_prefix[-1])
        ),
        "raw_journal_canonical_json_lines_sha256_at_reporting_cutoff": (
            sha256_bytes(raw_lines)
        ),
        "semantic_journal_event_count_at_reporting_cutoff": len(semantic_prefix),
        "semantic_journal_canonical_json_lines_sha256_at_reporting_cutoff": (
            sha256_bytes(semantic_lines)
        ),
        "classified_test_contamination_event_count": 2,
        "classified_test_contamination_event_indices": [5, 6],
        "classified_test_contamination_raw_sha256": list(
            A08_RAW_JOURNAL_SHA256[5:7]
        ),
        "classified_test_contamination_canonical_sha256": list(
            A08_CANONICAL_JOURNAL_SHA256[5:7]
        ),
        "raw_reporting_completion_events": raw_reporting,
        "semantic_reporting_completion_events": semantic_reporting,
        "corrigendum_journal_event_index": 7,
        "corrigendum_journal_event_raw_sha256": receipt[
            "corrigendum_journal_event_raw_sha256"
        ],
        "corrigendum_journal_head_sha256": receipt[
            "corrigendum_journal_head_sha256"
        ],
        "amendment08_overlay_install_receipt_sha256": sha256_file(receipt_path),
        "genuine_reporting_completion_event_index": cutoff_index,
        "genuine_reporting_completion_event_raw_sha256": sha256_file(genuine_path),
        "genuine_reporting_completion_event_canonical_sha256": sha256_bytes(
            strict_full_canonical_json_bytes(genuine_event)
        ),
        "genuine_reporting_completion_manifest_sha256": sha256_file(reporting_path),
        **{
            key: prerequisites[key] for key in (
                "base_scientific_execution_code_manifest_sha256",
                "effective_scientific_execution_code_manifest_sha256",
                "amendment08_code_corrigendum_sha256",
                "amendment08_journal_contamination_adjudication_sha256",
                "full_models_frozen_manifest_sha256",
                "external_audit_opening_sha256",
                "reporting_completion_manifest_sha256",
                "metric_recomputation_sha256",
                "source_input_hashes_post_sha256",
            )
        },
        "previous_state_marker_sha256": marker["previous_marker_sha256"],
        "planned_full_scientific_complete_marker_recorded_at": marker[
            "recorded_at"
        ],
        "planned_full_scientific_complete_marker": marker,
        "planned_full_scientific_complete_marker_sha256": sha256_bytes(marker_bytes),
        "prerequisite_validation_status": prerequisites["status"],
    }


def validate_amendment08_raw_journal_cutoff(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    path = run_dir / A08_RAW_JOURNAL_CUTOFF_RELATIVE
    payload = _require_exact_keys(
        strict_full_load_file(path), set(A08_RAW_JOURNAL_CUTOFF_KEYS),
        label="Amendment 08 raw-Journal cutoff",
    )
    records = payload["records"]
    if not (
        payload["schema"] == "amendment08_raw_journal_at_reporting_cutoff/v1"
        and payload["status"] == "PASS"
        and payload["authorization"] == A08_AUTHORIZATION
        and payload["prompt_sha256"] == A08_PROMPT_SHA256
        and payload["target_run_id"] == run_dir.name == A08_TARGET_RUN_ID
        and isinstance(payload["created_at"], str) and payload["created_at"]
        and isinstance(records, list) and records
        and payload["record_count"] == len(records)
    ):
        raise Amendment06IntegrityError("Amendment 08 cutoff identity/count differs")
    events: list[dict[str, Any]] = []
    raw_lines = bytearray()
    previous_sha256: str | None = None
    for expected_index, record_value in enumerate(records):
        record = _require_exact_keys(
            record_value, set(A08_RAW_JOURNAL_RECORD_KEYS),
            label=f"Amendment 08 cutoff record {expected_index}",
        )
        event_value = record["event"]
        if not isinstance(event_value, Mapping):
            raise Amendment06IntegrityError(
                f"Amendment 08 cutoff event object differs at {expected_index}"
            )
        physical_index = record["physical_event_index"]
        raw_size = record["raw_size_bytes"]
        if not (
            isinstance(physical_index, int)
            and not isinstance(physical_index, bool)
            and isinstance(raw_size, int) and not isinstance(raw_size, bool)
            and raw_size >= 0
            and isinstance(record["raw_bytes_base64"], str)
            and re.fullmatch(r"[0-9a-f]{64}", str(record["raw_sha256"]))
            and re.fullmatch(r"[0-9a-f]{64}", str(record["canonical_sha256"]))
        ):
            raise Amendment06IntegrityError(
                f"Amendment 08 cutoff record types differ at {expected_index}"
            )
        try:
            raw = base64.b64decode(
                record["raw_bytes_base64"], validate=True,
            )
        except (ValueError, binascii.Error) as exc:
            raise Amendment06IntegrityError(
                f"Amendment 08 cutoff base64 differs at {expected_index}"
            ) from exc
        live_path = _journal_event_path(run_dir, expected_index)
        if not _safe_regular(live_path) or live_path.read_bytes() != raw:
            raise Amendment06IntegrityError(
                f"Live raw Journal bytes differ from the frozen cutoff at {expected_index}"
            )
        event = _require_exact_keys(
            strict_full_loads(raw), set(event_value),
            label=f"Amendment 08 cutoff event {expected_index}",
        )
        event = dict(event)
        canonical = strict_full_canonical_json_bytes(event)
        event_index = event.get("event_index")
        if not (
            isinstance(event_index, int) and not isinstance(event_index, bool)
            and event == event_value
            and raw == strict_full_json_bytes(event)
            and physical_index == expected_index
            and record["relative_path"] == f"{expected_index:08d}.json"
            and raw_size == len(raw)
            and record["raw_sha256"] == sha256_bytes(raw)
            and record["canonical_sha256"] == sha256_bytes(canonical)
            and event_index == expected_index
            and event.get("run_id") == run_dir.name
            and event.get("previous_event_sha256") == previous_sha256
        ):
            raise Amendment06IntegrityError(
                f"Amendment 08 cutoff record/chain differs at {expected_index}"
            )
        events.append(event)
        raw_lines.extend(canonical)
        raw_lines.extend(b"\n")
        previous_sha256 = sha256_bytes(canonical)
    expected_prefix = _amendment08_expected_raw_prefix_events()
    if events[:7] != list(expected_prefix):
        raise Amendment06IntegrityError("Amendment 08 cutoff immutable prefix differs")
    for index in range(7):
        if not (
            records[index]["raw_sha256"] == A08_RAW_JOURNAL_SHA256[index]
            and records[index]["canonical_sha256"]
            == A08_CANONICAL_JOURNAL_SHA256[index]
        ):
            raise Amendment06IntegrityError(
                f"Amendment 08 cutoff immutable hash pin differs at {index}"
            )
    semantic = [
        event for event in events
        if int(event["event_index"]) not in A08_CONTAMINATION_INDICES
    ]
    semantic_lines = b"".join(
        strict_full_canonical_json_bytes(event) + b"\n" for event in semantic
    )
    raw_reporting = sum(
        event.get("event") == "REPORTING_COMPLETION_COMMITTED" for event in events
    )
    semantic_reporting = sum(
        event.get("event") == "REPORTING_COMPLETION_COMMITTED" for event in semantic
    )
    genuine = [
        event for event in semantic
        if event.get("event") == "REPORTING_COMPLETION_COMMITTED"
    ]
    if len(genuine) != 1:
        raise Amendment06IntegrityError("Amendment 08 cutoff genuine reporting differs")
    genuine_event = genuine[0]
    genuine_index = int(genuine_event["event_index"])
    genuine_keys = {
        "event_index", "event", "run_id", "previous_event_sha256", "recorded_at",
        "phase", "variant_id", "attempt_id", "completion_sha256",
        "recovered_after_interruption",
    }
    reporting_path = _reporting_completion_path(run_dir)
    if not (
        set(genuine_event) == genuine_keys
        and genuine_event == events[-1]
        and genuine_event["phase"] == "REPORTING"
        and genuine_event["variant_id"] == "GLOBAL"
        and re.fullmatch(r"[0-9a-f]{32}", str(genuine_event["attempt_id"]))
        and genuine_event["attempt_id"] != "a" * 32
        and isinstance(genuine_event["recovered_after_interruption"], bool)
        and genuine_event["completion_sha256"] == sha256_file(reporting_path)
        and raw_reporting == 3 and semantic_reporting == 1
        and payload["raw_journal_event_count_at_reporting_cutoff"] == len(events)
        and payload["raw_journal_head_sha256_at_reporting_cutoff"]
        == sha256_bytes(strict_full_canonical_json_bytes(events[-1]))
        and payload["raw_journal_canonical_json_lines_sha256_at_reporting_cutoff"]
        == sha256_bytes(bytes(raw_lines))
        and payload["semantic_journal_event_count_at_reporting_cutoff"]
        == len(semantic) == len(events) - 2
        and payload[
            "semantic_journal_canonical_json_lines_sha256_at_reporting_cutoff"
        ] == sha256_bytes(semantic_lines)
        and payload["classified_test_contamination_event_count"] == 2
        and payload["classified_test_contamination_event_indices"] == [5, 6]
        and payload["classified_test_contamination_raw_sha256"]
        == list(A08_RAW_JOURNAL_SHA256[5:7])
        and payload["classified_test_contamination_canonical_sha256"]
        == list(A08_CANONICAL_JOURNAL_SHA256[5:7])
        and payload["raw_reporting_completion_events"] == 3
        and payload["semantic_reporting_completion_events"] == 1
        and payload["genuine_reporting_completion_event_index"] == genuine_index
        and payload["genuine_reporting_completion_event_raw_sha256"]
        == records[genuine_index]["raw_sha256"]
        and payload["genuine_reporting_completion_event_canonical_sha256"]
        == records[genuine_index]["canonical_sha256"]
        and payload["genuine_reporting_completion_manifest_sha256"]
        == sha256_file(reporting_path)
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 cutoff reporting/count/hash disclosure differs"
        )
    receipt_path = run_dir / "provenance/amendment08_overlay_install_receipt.json"
    receipt = strict_full_load_file(receipt_path)
    lineage = _amendment08_frozen_lineage_bindings(run_dir)
    if not (
        payload["corrigendum_journal_event_index"] == 7
        and payload["corrigendum_journal_event_raw_sha256"]
        == records[7]["raw_sha256"]
        == receipt["corrigendum_journal_event_raw_sha256"]
        and payload["corrigendum_journal_head_sha256"]
        == records[7]["canonical_sha256"]
        == receipt["corrigendum_journal_head_sha256"]
        and payload["amendment08_overlay_install_receipt_sha256"]
        == sha256_file(receipt_path)
        and all(
            payload.get(key) == lineage[key]
            for key in (
                "base_scientific_execution_code_manifest_sha256",
                "effective_scientific_execution_code_manifest_sha256",
                "amendment08_code_corrigendum_sha256",
                "amendment08_journal_contamination_adjudication_sha256",
            )
        )
    ):
        raise Amendment06IntegrityError("Amendment 08 cutoff overlay binding differs")
    prerequisites = _validate_amendment08_frozen_prerequisite_receipts(run_dir)
    for key in (
        "full_models_frozen_manifest_sha256", "external_audit_opening_sha256",
        "reporting_completion_manifest_sha256", "metric_recomputation_sha256",
        "source_input_hashes_post_sha256",
    ):
        if payload[key] != prerequisites[key]:
            raise Amendment06IntegrityError(
                f"Amendment 08 cutoff prerequisite binding differs: {key}"
            )
    marker = _require_exact_keys(
        payload["planned_full_scientific_complete_marker"],
        {
            "run_id", "run_kind", "phase", "phase_index", "status",
            "previous_phase", "previous_marker_sha256", "recorded_at",
        },
        label="Amendment 08 cutoff planned final marker",
    )
    marker_bytes = strict_full_json_bytes(marker)
    previous_path = _state_marker_path(run_dir, "FULL_EXTERNAL_AUDIT_SCORED")
    final_path = _state_marker_path(run_dir, "FULL_SCIENTIFIC_COMPLETE")
    if not (
        payload["prerequisite_validation_status"] == prerequisites["status"]
        == "PASS_FROZEN_RECEIPTS_ONLY"
        and payload["previous_state_marker_sha256"] == sha256_file(previous_path)
        and marker["previous_marker_sha256"] == sha256_file(previous_path)
        and marker["run_id"] == run_dir.name and marker["run_kind"] == RUN_KIND
        and marker["phase"] == "FULL_SCIENTIFIC_COMPLETE"
        and marker["phase_index"] == 3 and marker["status"] == "PASS"
        and marker["previous_phase"] == "FULL_EXTERNAL_AUDIT_SCORED"
        and marker["recorded_at"]
        == payload["planned_full_scientific_complete_marker_recorded_at"]
        and marker["recorded_at"] == payload["created_at"]
        and payload["planned_full_scientific_complete_marker_sha256"]
        == sha256_bytes(marker_bytes)
    ):
        raise Amendment06IntegrityError("Amendment 08 cutoff planned-state binding differs")
    if final_path.exists() and not (
        _safe_regular(final_path, sha256_bytes(marker_bytes))
        and final_path.read_bytes() == marker_bytes
    ):
        raise Amendment06IntegrityError(
            "Published Amendment 08 final state differs from cutoff plan"
        )
    corrected_history = _validate_amendment08_corrected_training_history(
        run_dir=run_dir, overlay=lineage, allow_unterminated=False,
    )
    if not (
        corrected_history["completed_variants"] == list(TRAINED_VARIANTS)
        and corrected_history["next_variant"] is None
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 frozen cutoff training history is incomplete"
        )
    _validate_amendment08_cutoff_lifecycle(
        run_dir=run_dir, events=events, lineage=lineage,
    )
    return dict(payload)


def _publish_or_validate_amendment08_raw_journal_cutoff(
    run_dir: Path, *, final_state_recorded_at: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    path = run_dir / A08_RAW_JOURNAL_CUTOFF_RELATIVE
    if path.exists():
        payload = validate_amendment08_raw_journal_cutoff(run_dir)
        if (
            final_state_recorded_at is not None
            and payload["planned_full_scientific_complete_marker_recorded_at"]
            != final_state_recorded_at
        ):
            raise Amendment06IntegrityError(
                "Existing Amendment 08 cutoff reserved timestamp differs"
            )
        return payload
    payload = build_amendment08_raw_journal_cutoff(
        run_dir, final_state_recorded_at=final_state_recorded_at,
    )
    if set(payload) != set(A08_RAW_JOURNAL_CUTOFF_KEYS):
        raise Amendment06IntegrityError("Amendment 08 cutoff builder key set differs")
    publish_strict_json_no_clobber(path, payload)
    return validate_amendment08_raw_journal_cutoff(run_dir)


def _build_legacy_execution_ledger(run_dir: Path) -> dict[str, Any]:
    all_events = _read_journal(run_dir)
    cutoffs = [
        index for index, event in enumerate(all_events)
        if event.get("event") == "REPORTING_COMPLETION_COMMITTED"
    ]
    if len(cutoffs) != 1:
        raise Amendment06IntegrityError("Execution ledger requires one reporting-completion cutoff")
    events = all_events[:cutoffs[0] + 1]
    event_counts = Counter(str(event.get("event")) for event in events)
    indexed = list(enumerate(events))
    amendment07_active = (
        Path(run_dir).resolve() == A07_TARGET_RUN.resolve()
        or (run_dir / "provenance/amendment07_overlay_install_receipt.json").exists()
    )
    lineage = _scientific_lineage_bindings(run_dir) if amendment07_active else {}
    if amendment07_active:
        overlay = validate_amendment07_overlay(run_dir)
        corrected_history = _validate_amendment07_corrected_training_history(
            run_dir=run_dir, overlay=overlay, allow_unterminated=False,
        )
        if not (
            corrected_history["completed_variants"] == list(TRAINED_VARIANTS)
            and corrected_history["next_variant"] is None
        ):
            raise Amendment06IntegrityError(
                "Amendment 07 final corrected training history is incomplete"
            )
        expected_event5 = {
            "event": "AMENDMENT07_CORRIGENDUM_INSTALLED",
            "status": "PASS", "authorization": A07_AUTHORIZATION,
            "prompt_sha256": A07_PROMPT_SHA256,
            "effective_scientific_execution_code_manifest_sha256": overlay[
                "effective_scientific_execution_code_manifest_sha256"
            ],
            "amendment07_code_corrigendum_sha256": overlay[
                "amendment07_code_corrigendum_sha256"
            ],
            "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        }
        if not (
            len(events) > 7
            and events[4].get("event") == "TRAINING_ATTEMPT_FAILED"
            and events[4].get("attempt_id") == A07_FAILED_ATTEMPT_ID
            and events[5].get("event_index") == 5
            and all(events[5].get(key) == value for key, value in expected_event5.items())
            and events[6].get("event_index") == 6
            and events[6].get("event") == "CACHE_LOAD"
            and events[6].get("status") == "PASS"
            and events[6].get("cache_path")
            == str(EXPECTED_RUNTIME_ROOT / "full_population" / run_dir.name)
            and events[7].get("event_index") == 7
            and events[7].get("event") == "TRAINING_ATTEMPT_STARTED"
            and events[7].get("variant_id") == "full_new_reference"
            and event_counts.get("CACHE_GENERATION", 0) == 1
            and event_counts.get("CACHE_LOAD", 0) == 1
            and event_counts.get("CACHE_REMATERIALIZATION", 0) == 0
            and event_counts.get("CACHE_REMATERIALIZATION_INTENT", 0) == 0
        ):
            raise Amendment06IntegrityError("Amendment 07 execution prefix/cache contract differs")
        for index, event in indexed:
            if index <= 6 or event.get("event") not in {
                "TRAINING_ATTEMPT_STARTED", "SCIENTIFIC_FIT_CALLED",
            }:
                continue
            if not (
                event.get("amendment07_authorization") == A07_AUTHORIZATION
                and event.get("amendment07_prompt_sha256") == A07_PROMPT_SHA256
                and all(event.get(key) == value for key, value in lineage.items())
            ):
                raise Amendment06IntegrityError(
                    "Amendment 07 corrected training-event lineage differs"
                )
    training_commits = [
        (index, event) for index, event in indexed
        if event.get("event") == "TRAINING_COMPLETION_COMMITTED"
    ]
    scoring_commits = [
        (index, event) for index, event in indexed
        if event.get("event") == "AUDIT_SCORING_COMPLETION_COMMITTED"
    ]
    frozen_events = [
        (index, event) for index, event in indexed if event.get("event") == "MODELS_FROZEN"
    ]
    audit_events = [
        (index, event) for index, event in indexed if event.get("event") == "EXTERNAL_AUDIT_OPENED"
    ]
    if len(frozen_events) != 1 or len(audit_events) != 1:
        raise Amendment06IntegrityError("Execution ledger model-freeze/Audit-open cardinality differs")
    frozen_index = frozen_events[0][0]
    audit_index = audit_events[0][0]
    if not (frozen_index < audit_index < cutoffs[0]):
        raise Amendment06IntegrityError("Execution ledger phase ordering differs")
    completed_training = [str(event.get("variant_id")) for _, event in training_commits]
    completed_scoring = [str(event.get("variant_id")) for _, event in scoring_commits]
    if (
        completed_training != list(TRAINED_VARIANTS)
        or completed_scoring != list(TRAINED_VARIANTS)
        or any(index >= frozen_index for index, _ in training_commits)
        or any(index <= audit_index or index >= cutoffs[0] for index, _ in scoring_commits)
    ):
        raise Amendment06IntegrityError("Execution ledger cannot close before all completions")
    for index, event in training_commits:
        variant = str(event["variant_id"])
        completion_path = run_dir / _training_completion_relative(variant)
        starts = [
            start_index for start_index, candidate in indexed[:index]
            if candidate.get("event") == "TRAINING_ATTEMPT_STARTED"
            and candidate.get("variant_id") == variant
        ]
        fits = [
            fit_index for fit_index, candidate in indexed[:index]
            if candidate.get("event") == "SCIENTIFIC_FIT_CALLED"
            and candidate.get("variant_id") == variant
        ]
        if not (
            starts and fits and starts[-1] <= fits[-1] < index
            and event.get("completion_sha256") == sha256_file(completion_path)
        ):
            raise Amendment06IntegrityError(f"Execution ledger training chain differs: {variant}")
    for index, event in scoring_commits:
        variant = str(event["variant_id"])
        completion_path = run_dir / _scoring_completion_relative(variant)
        starts = [
            start_index for start_index, candidate in indexed[:index]
            if candidate.get("event") == "AUDIT_SCORING_ATTEMPT_STARTED"
            and candidate.get("variant_id") == variant
        ]
        if not starts or starts[-1] >= index or event.get("completion_sha256") != sha256_file(completion_path):
            raise Amendment06IntegrityError(f"Execution ledger scoring chain differs: {variant}")
    fit_events = [
        (index, event) for index, event in indexed
        if event.get("event") == "SCIENTIFIC_FIT_CALLED"
    ]
    start_events = [
        (index, event) for index, event in indexed
        if event.get("event") == "TRAINING_ATTEMPT_STARTED"
    ]
    failed_events = [
        (index, event) for index, event in indexed
        if event.get("event") == "TRAINING_ATTEMPT_FAILED"
    ]
    starts_by_attempt = Counter(str(event.get("attempt_id")) for _, event in start_events)
    fits_by_attempt = Counter(str(event.get("attempt_id")) for _, event in fit_events)
    failed_by_attempt = Counter(str(event.get("attempt_id")) for _, event in failed_events)
    committed_by_attempt = Counter(
        str(event.get("attempt_id")) for _, event in training_commits
    )
    attempt_ids = set(starts_by_attempt)
    if not attempt_ids or any(
        not re.fullmatch(r"[0-9a-f]{32}", attempt_id)
        or starts_by_attempt[attempt_id] != 1
        or fits_by_attempt[attempt_id] > 1
        or failed_by_attempt[attempt_id] + committed_by_attempt[attempt_id] != 1
        for attempt_id in attempt_ids
    ):
        raise Amendment06IntegrityError("Execution ledger training-attempt lifecycle differs")
    if set(fits_by_attempt) - attempt_ids or set(failed_by_attempt) - attempt_ids:
        raise Amendment06IntegrityError("Execution ledger has an orphan fit/failure event")
    if any(
        event.get("variant_id") not in TRAINED_VARIANTS
        or event.get("seed") != PRIMARY_MODEL_SEED
        or event.get("stability_fit") is not False
        or index >= frozen_index
        or str(event.get("attempt_id")) not in attempt_ids
        for index, event in fit_events
    ):
        raise Amendment06IntegrityError("Execution ledger fit authorization/order differs")
    successful_fit_variants = [
        str(event.get("variant_id")) for _, event in fit_events
        if committed_by_attempt[str(event.get("attempt_id"))] == 1
    ]
    if successful_fit_variants != list(TRAINED_VARIANTS):
        raise Amendment06IntegrityError("Execution ledger successful-fit Variant set/order differs")
    stability_fits = sum(
        event.get("stability_fit") is not False
        or event.get("seed") != PRIMARY_MODEL_SEED
        for _, event in fit_events
    )
    post_audit_training = sum(index > audit_index for index, _ in fit_events)
    threshold_calls = sum(
        event.get("event") == "THRESHOLD_TUNING_CALLED" for event in events
    )
    bootstrap_events = [
        (index, event) for index, event in indexed
        if event.get("event") == "BOOTSTRAP_RESAMPLING"
    ]
    last_scoring_index = max(index for index, _ in scoring_commits)
    if not (
        len(bootstrap_events) == 1
        and last_scoring_index < bootstrap_events[0][0] < cutoffs[0]
        and bootstrap_events[0][1].get("phase") == "REPORTING"
        and bootstrap_events[0][1].get("status") == "PASS"
        and bootstrap_events[0][1].get("replicates") == BOOTSTRAP_REPLICATES
        and bootstrap_events[0][1].get("seed") == BOOTSTRAP_SEED
        and bootstrap_events[0][1].get("model_retraining") is False
        and re.fullmatch(
            r"[0-9a-f]{64}", str(bootstrap_events[0][1].get("bootstrap_sha256", ""))
        )
    ):
        raise Amendment06IntegrityError("Execution ledger bootstrap binding/order differs")
    if stability_fits or post_audit_training or threshold_calls:
        raise Amendment06IntegrityError("Execution ledger records forbidden scientific calls")
    failed_training = event_counts.get("TRAINING_ATTEMPT_FAILED", 0)
    failed_scoring = event_counts.get("AUDIT_SCORING_ATTEMPT_FAILED", 0)
    failed_reporting = event_counts.get("REPORTING_ATTEMPT_FAILED", 0)
    return {
        "run_id": run_dir.name, "run_kind": RUN_KIND, "status": "PASS",
        **lineage,
        "journal_schema": "ATOMIC_CONTIGUOUS_HASH_CHAINED_PER_EVENT_V1",
        "journal_phase_order_validation": "PASS",
        "completed_trained_models": len(completed_training),
        "completed_official_runnable_models": sum(
            variant in OFFICIAL_VARIANTS for variant in completed_training
        ),
        "completed_exploratory_models": sum(
            variant == EXPLORATORY_VARIANT for variant in completed_training
        ),
        "completed_true_no_pca_models": 0,
        "stability_fits": stability_fits,
        "threshold_tuning_calls": threshold_calls,
        "post_audit_training_calls": post_audit_training,
        "scientific_fit_attempts": len(fit_events),
        "failed_unpublished_training_attempts": failed_training,
        "failed_unpublished_scoring_attempts": failed_scoring,
        "failed_unpublished_reporting_attempts": failed_reporting,
        "prediction_only_reload_events": (
            event_counts.get("AUDIT_SCORING_ATTEMPT_STARTED", 0)
            + event_counts.get("COMPLETED_AUDIT_SCORING_SKIPPED", 0)
        ),
        "cache_generation_events": event_counts.get("CACHE_GENERATION", 0),
        "cache_load_events": event_counts.get("CACHE_LOAD", 0),
        "cache_rematerialization_events": event_counts.get("CACHE_REMATERIALIZATION", 0),
        "amendment07_corrigendum_installations": (
            event_counts.get("AMENDMENT07_CORRIGENDUM_INSTALLED", 0)
        ),
        "amendment07_failed_attempt_id": (
            A07_FAILED_ATTEMPT_ID if amendment07_active else None
        ),
        "external_audit_open_events": len(audit_events),
        "bootstrap_resampling_events": len(bootstrap_events),
        "package_attempts_at_scientific_freeze": 0,
        "package_attempts_tracked_adjacent_to_frozen_run": True,
        "ownership_journal_path": str(_journal_path(run_dir)),
        "ownership_journal_prefix_sha256_at_scientific_freeze": sha256_bytes(b"".join(
            strict_full_canonical_json_bytes(event) + b"\n" for event in events
        )),
        "ownership_journal_event_count_at_scientific_freeze": len(events),
    }


def _validate_amendment08_cutoff_lifecycle(
    *, run_dir: Path, events: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, str],
) -> dict[str, Any]:
    indexed = [
        (int(event["event_index"]), dict(event)) for event in events
        if int(event["event_index"]) not in A08_CONTAMINATION_INDICES
    ]
    by_physical = {index: event for index, event in indexed}
    event_counts = Counter(str(event.get("event")) for _, event in indexed)
    base_keys = {
        "event_index", "event", "run_id", "previous_event_sha256", "recorded_at",
    }
    event8 = by_physical.get(8, {})
    event9 = by_physical.get(9, {})
    expected_event8 = {
        "status": "PASS",
        "cache_path": str(EXPECTED_RUNTIME_ROOT / "full_population" / run_dir.name),
        "cache_manifest_sha256": A07_CACHE_MANIFEST_SHA256,
        "frozen_population_sha256": A07_FROZEN_POPULATION_SHA256,
        "unique_realizations": 1, "augmentation_generation_count": 1,
        "safe_smote_generation_count": 1, **lineage,
    }
    if not (
        set(event8) == base_keys | set(expected_event8)
        and event8.get("event") == "CACHE_LOAD"
        and event8.get("event_index") == 8
        and all(event8.get(key) == value for key, value in expected_event8.items())
    ):
        raise Amendment06IntegrityError("Execution ledger exact event 8 differs")

    start_keys = {*base_keys, "phase", "variant_id", "attempt_id", *lineage}
    fit_keys = {*start_keys, "seed", "stability_fit"}
    failure_base_keys = {
        *base_keys, "phase", "variant_id", "attempt_id", "error_type", "error",
    }
    completion_keys = {
        *base_keys, "phase", "variant_id", "attempt_id", "completion_sha256",
        "recovered_after_interruption",
    }
    attempts: dict[str, dict[str, Any]] = {}
    completed_training: list[str] = []
    training_starts: list[tuple[int, dict[str, Any]]] = []
    fit_events: list[tuple[int, dict[str, Any]]] = []
    training_failures: list[tuple[int, dict[str, Any]]] = []
    training_commits: list[tuple[int, dict[str, Any]]] = []
    for physical_index, event in indexed:
        name = event.get("event")
        if name not in {
            "TRAINING_ATTEMPT_STARTED", "SCIENTIFIC_FIT_CALLED",
            "TRAINING_ATTEMPT_FAILED", "TRAINING_COMPLETION_COMMITTED",
        }:
            continue
        attempt_id = str(event.get("attempt_id", ""))
        if name == "TRAINING_ATTEMPT_STARTED":
            if not re.fullmatch(r"[0-9a-f]{32}", attempt_id) or attempt_id in attempts:
                raise Amendment06IntegrityError("Execution ledger training start ID differs")
            if any(state["terminal"] is None for state in attempts.values()):
                raise Amendment06IntegrityError("Execution ledger overlaps training attempts")
            if physical_index == 2:
                if event != _amendment08_expected_raw_prefix_events()[2]:
                    raise Amendment06IntegrityError("Execution ledger original start differs")
                variant = "full_new_reference"
            else:
                next_variant = next(
                    (variant for variant in TRAINED_VARIANTS
                     if variant not in completed_training), None,
                )
                if not (
                    physical_index >= 9 and set(event) == start_keys
                    and event.get("phase") == "TRAINING"
                    and event.get("variant_id") == next_variant
                    and attempt_id not in {A07_FAILED_ATTEMPT_ID, "a" * 32}
                    and all(event.get(key) == value for key, value in lineage.items())
                ):
                    raise Amendment06IntegrityError(
                        "Execution ledger corrected training start differs"
                    )
                variant = str(next_variant)
            attempts[attempt_id] = {
                "variant": variant, "start": physical_index,
                "fit": None, "terminal": None,
            }
            training_starts.append((physical_index, event))
        elif name == "SCIENTIFIC_FIT_CALLED":
            state = attempts.get(attempt_id)
            if state is None or state["terminal"] is not None or state["fit"] is not None:
                raise Amendment06IntegrityError("Execution ledger orphan/duplicate fit differs")
            if physical_index == 3:
                valid_fit = event == _amendment08_expected_raw_prefix_events()[3]
            else:
                valid_fit = (
                    set(event) == fit_keys
                    and event.get("phase") == "TRAINING"
                    and event.get("variant_id") == state["variant"]
                    and event.get("seed") == PRIMARY_MODEL_SEED
                    and event.get("stability_fit") is False
                    and all(event.get(key) == value for key, value in lineage.items())
                )
            if not valid_fit or physical_index <= int(state["start"]):
                raise Amendment06IntegrityError("Execution ledger fit binding/order differs")
            state["fit"] = physical_index
            fit_events.append((physical_index, event))
        elif name == "TRAINING_ATTEMPT_FAILED":
            state = attempts.get(attempt_id)
            observed_keys = set(event)
            valid_schema = frozenset(observed_keys) in {
                frozenset(failure_base_keys),
                frozenset(failure_base_keys | {"failure_phase"}),
                frozenset(failure_base_keys | {
                    "failure_phase", "recovered_after_interruption",
                }),
            }
            if not (
                state is not None and state["terminal"] is None and valid_schema
                and event.get("phase") == "TRAINING"
                and event.get("variant_id") == state["variant"]
                and physical_index > int(state["start"])
                and isinstance(event.get("error_type"), str)
                and bool(event["error_type"])
                and isinstance(event.get("error"), str) and bool(event["error"])
            ):
                raise Amendment06IntegrityError("Execution ledger training failure differs")
            if "recovered_after_interruption" in event and not (
                event.get("recovered_after_interruption") is True
                and event.get("failure_phase")
                == "PROCESS_INTERRUPTION_BEFORE_COMPLETION_COMMIT"
            ):
                raise Amendment06IntegrityError(
                    "Execution ledger recovered training failure differs"
                )
            if "failure_phase" in event and "recovered_after_interruption" not in event:
                if event.get("failure_phase") != "POST_START_THROUGH_COMPLETION_COMMIT":
                    raise Amendment06IntegrityError(
                        "Execution ledger training failure phase differs"
                    )
            state["terminal"] = "FAILED"
            training_failures.append((physical_index, event))
        else:
            state = attempts.get(attempt_id)
            variant = str(event.get("variant_id", ""))
            completion_path = run_dir / _training_completion_relative(variant)
            if not (
                state is not None and state["terminal"] is None
                and state["fit"] is not None and set(event) == completion_keys
                and event.get("phase") == "TRAINING"
                and variant == state["variant"] and variant not in completed_training
                and physical_index > int(state["fit"])
                and _safe_regular(
                    completion_path, str(event.get("completion_sha256", "")),
                )
                and isinstance(event.get("recovered_after_interruption"), bool)
            ):
                raise Amendment06IntegrityError(
                    "Execution ledger training completion differs"
                )
            state["terminal"] = "COMMITTED"
            completed_training.append(variant)
            training_commits.append((physical_index, event))
    if not (
        event9 in [event for _, event in training_starts]
        and event9.get("event_index") == 9
        and event9.get("variant_id") == "full_new_reference"
        and event9.get("attempt_id") not in {A07_FAILED_ATTEMPT_ID, "a" * 32}
        and completed_training == list(TRAINED_VARIANTS)
        and attempts and all(state["terminal"] is not None for state in attempts.values())
        and len(training_commits) == 10
        and len(fit_events) >= 11
        and len(training_failures) >= 1
        and len(fit_events) - 10 <= len(training_failures)
    ):
        raise Amendment06IntegrityError("Execution ledger training lifecycle is incomplete")

    frozen_events = [
        (index, event) for index, event in indexed if event.get("event") == "MODELS_FROZEN"
    ]
    audit_events = [
        (index, event) for index, event in indexed
        if event.get("event") == "EXTERNAL_AUDIT_OPENED"
    ]
    if len(frozen_events) != 1 or len(audit_events) != 1:
        raise Amendment06IntegrityError("Execution ledger freeze/Audit cardinality differs")
    frozen_index, frozen_event = frozen_events[0]
    audit_index, audit_event = audit_events[0]
    frozen_keys = {
        *base_keys, "status", "model_count", "manifest_sha256",
        "recovered_after_interruption",
    }
    audit_keys = {
        *base_keys, "status", "evidence_sha256",
        "recovered_after_interruption",
    }
    if not (
        max(index for index, _ in training_commits) < frozen_index < audit_index
        and all(index < frozen_index for index, _ in fit_events)
        and set(frozen_event) == frozen_keys
        and frozen_event.get("status") == "PASS"
        and frozen_event.get("model_count") == 10
        and isinstance(frozen_event.get("recovered_after_interruption"), bool)
        and frozen_event.get("manifest_sha256") == sha256_file(
            run_dir / "provenance/full_models_frozen_manifest.json"
        )
        and set(audit_event) == audit_keys
        and audit_event.get("status") == "PASS"
        and isinstance(audit_event.get("recovered_after_interruption"), bool)
        and audit_event.get("evidence_sha256") == sha256_file(
            run_dir / "provenance/external_audit_exclusion_and_opening.json"
        )
    ):
        raise Amendment06IntegrityError("Execution ledger freeze/Audit order differs")

    scoring_attempts: dict[str, dict[str, Any]] = {}
    completed_scoring: list[str] = []
    scoring_completions: list[tuple[int, dict[str, Any]]] = []
    scoring_failures: list[tuple[int, dict[str, Any]]] = []
    for physical_index, event in indexed:
        name = event.get("event")
        if name not in {
            "AUDIT_SCORING_ATTEMPT_STARTED", "AUDIT_SCORING_ATTEMPT_FAILED",
            "AUDIT_SCORING_COMPLETION_COMMITTED",
        }:
            continue
        attempt_id = str(event.get("attempt_id", ""))
        if name == "AUDIT_SCORING_ATTEMPT_STARTED":
            next_variant = next(
                (variant for variant in TRAINED_VARIANTS
                 if variant not in completed_scoring), None,
            )
            if not (
                set(event) == base_keys | {"phase", "variant_id", "attempt_id"}
                and event.get("phase") == "AUDIT_SCORING"
                and event.get("variant_id") == next_variant
                and re.fullmatch(r"[0-9a-f]{32}", attempt_id)
                and attempt_id not in scoring_attempts
                and physical_index > audit_index
                and not any(
                    state["terminal"] is None for state in scoring_attempts.values()
                )
            ):
                raise Amendment06IntegrityError("Execution ledger scoring start differs")
            scoring_attempts[attempt_id] = {
                "variant": str(event["variant_id"]), "start": physical_index,
                "terminal": None,
            }
        elif name == "AUDIT_SCORING_ATTEMPT_FAILED":
            state = scoring_attempts.get(attempt_id)
            allowed = {
                frozenset(failure_base_keys),
                frozenset(failure_base_keys | {"failure_phase"}),
                frozenset(failure_base_keys | {
                    "failure_phase", "recovered_after_interruption",
                }),
            }
            if not (
                state is not None and state["terminal"] is None
                and frozenset(event) in allowed
                and event.get("phase") == "AUDIT_SCORING"
                and event.get("variant_id") == state["variant"]
                and physical_index > int(state["start"])
                and isinstance(event.get("error_type"), str)
                and bool(str(event["error_type"]).strip())
                and isinstance(event.get("error"), str)
                and bool(str(event["error"]).strip())
            ):
                raise Amendment06IntegrityError("Execution ledger scoring failure differs")
            if "recovered_after_interruption" in event and not (
                event.get("recovered_after_interruption") is True
                and event.get("failure_phase")
                == "PROCESS_INTERRUPTION_BEFORE_COMPLETION_COMMIT"
            ):
                raise Amendment06IntegrityError(
                    "Execution ledger recovered scoring failure differs"
                )
            if "failure_phase" in event and "recovered_after_interruption" not in event:
                if event.get("failure_phase") != "POST_START_THROUGH_COMPLETION_COMMIT":
                    raise Amendment06IntegrityError(
                        "Execution ledger scoring failure phase differs"
                    )
            state["terminal"] = "FAILED"
            scoring_failures.append((physical_index, event))
        else:
            state = scoring_attempts.get(attempt_id)
            variant = str(event.get("variant_id", ""))
            completion_path = run_dir / _scoring_completion_relative(variant)
            if not (
                state is not None and state["terminal"] is None
                and set(event) == completion_keys
                and event.get("phase") == "AUDIT_SCORING"
                and variant == state["variant"]
                and physical_index > int(state["start"])
                and _safe_regular(
                    completion_path, str(event.get("completion_sha256", "")),
                )
                and isinstance(event.get("recovered_after_interruption"), bool)
            ):
                raise Amendment06IntegrityError("Execution ledger scoring completion differs")
            state["terminal"] = "COMMITTED"
            completed_scoring.append(variant)
            scoring_completions.append((physical_index, event))
    if not (
        completed_scoring == list(TRAINED_VARIANTS)
        and scoring_attempts
        and all(state["terminal"] is not None for state in scoring_attempts.values())
    ):
        raise Amendment06IntegrityError("Execution ledger scoring lifecycle is incomplete")

    bootstrap_events = [
        (index, event) for index, event in indexed
        if event.get("event") == "BOOTSTRAP_RESAMPLING"
    ]
    reporting_attempts: dict[str, dict[str, Any]] = {}
    reporting_starts: list[tuple[int, dict[str, Any]]] = []
    reporting_failures: list[tuple[int, dict[str, Any]]] = []
    reporting_completions: list[tuple[int, dict[str, Any]]] = []
    for physical_index, event in indexed:
        name = event.get("event")
        if name not in {
            "REPORTING_ATTEMPT_STARTED", "REPORTING_ATTEMPT_FAILED",
            "REPORTING_COMPLETION_COMMITTED",
        }:
            continue
        attempt_id = str(event.get("attempt_id", ""))
        if name == "REPORTING_ATTEMPT_STARTED":
            if not (
                set(event) == base_keys | {"phase", "variant_id", "attempt_id"}
                and event.get("phase") == "REPORTING"
                and event.get("variant_id") == "GLOBAL"
                and re.fullmatch(r"[0-9a-f]{32}", attempt_id)
                and attempt_id != "a" * 32
                and attempt_id not in reporting_attempts
                and not any(
                    state["terminal"] is None for state in reporting_attempts.values()
                )
                and physical_index > max(index for index, _ in scoring_completions)
            ):
                raise Amendment06IntegrityError("Execution ledger reporting start differs")
            reporting_attempts[attempt_id] = {
                "start": physical_index, "terminal": None,
            }
            reporting_starts.append((physical_index, event))
        elif name == "REPORTING_ATTEMPT_FAILED":
            state = reporting_attempts.get(attempt_id)
            allowed = {
                frozenset(failure_base_keys),
                frozenset(failure_base_keys | {"failure_phase"}),
                frozenset(failure_base_keys | {
                    "failure_phase", "recovered_after_interruption",
                }),
            }
            if not (
                state is not None and state["terminal"] is None
                and frozenset(event) in allowed
                and event.get("phase") == "REPORTING"
                and event.get("variant_id") == "GLOBAL"
                and physical_index > int(state["start"])
                and isinstance(event.get("error_type"), str)
                and bool(str(event["error_type"]).strip())
                and isinstance(event.get("error"), str)
                and bool(str(event["error"]).strip())
            ):
                raise Amendment06IntegrityError("Execution ledger reporting failure differs")
            if "recovered_after_interruption" in event and not (
                event.get("recovered_after_interruption") is True
                and event.get("failure_phase")
                == "PROCESS_INTERRUPTION_BEFORE_COMPLETION_COMMIT"
            ):
                raise Amendment06IntegrityError(
                    "Execution ledger recovered reporting failure differs"
                )
            if "failure_phase" in event and "recovered_after_interruption" not in event:
                if event.get("failure_phase") != "POST_START_THROUGH_COMPLETION_COMMIT":
                    raise Amendment06IntegrityError(
                        "Execution ledger reporting failure phase differs"
                    )
            state["terminal"] = "FAILED"
            reporting_failures.append((physical_index, event))
        else:
            state = reporting_attempts.get(attempt_id)
            if not (
                state is not None and state["terminal"] is None
                and set(event) == completion_keys
                and event.get("phase") == "REPORTING"
                and event.get("variant_id") == "GLOBAL"
                and physical_index > int(state["start"])
                and _safe_regular(
                    _reporting_completion_path(run_dir),
                    str(event.get("completion_sha256", "")),
                )
                and isinstance(event.get("recovered_after_interruption"), bool)
            ):
                raise Amendment06IntegrityError(
                    "Execution ledger reporting completion differs"
                )
            state["terminal"] = "COMMITTED"
            reporting_completions.append((physical_index, event))
    bootstrap_path = run_dir / "statistics/paired_card_bootstrap.json"
    bootstrap_payload = strict_full_load_file(bootstrap_path)
    bootstrap_keys = {
        *base_keys, "phase", "status", "replicates", "seed",
        "model_retraining", "bootstrap_sha256",
    }
    if not (
        len(bootstrap_events) == 1 and len(reporting_completions) == 1
        and reporting_starts and reporting_attempts
        and all(
            state["terminal"] is not None for state in reporting_attempts.values()
        )
        and max(index for index, _ in scoring_completions)
        < reporting_starts[0][0] <= bootstrap_events[0][0]
        < reporting_completions[0][0]
        == indexed[-1][0]
        and set(bootstrap_events[0][1]) == bootstrap_keys
        and bootstrap_events[0][1].get("phase") == "REPORTING"
        and bootstrap_events[0][1].get("status") == "PASS"
        and bootstrap_events[0][1].get("replicates") == BOOTSTRAP_REPLICATES
        and bootstrap_events[0][1].get("seed") == BOOTSTRAP_SEED
        and bootstrap_events[0][1].get("model_retraining") is False
        and bootstrap_events[0][1].get("bootstrap_sha256")
        == sha256_bytes(strict_full_canonical_json_bytes(bootstrap_payload))
        and reporting_completions[0][1].get("completion_sha256") == sha256_file(
            _reporting_completion_path(run_dir)
        )
    ):
        raise Amendment06IntegrityError("Execution ledger reporting lifecycle differs")
    if not (
        event_counts.get("CACHE_GENERATION", 0) == 1
        and event_counts.get("CACHE_LOAD", 0) == 1
        and event_counts.get("CACHE_REMATERIALIZATION", 0) == 0
        and event_counts.get("CACHE_REMATERIALIZATION_INTENT", 0) == 0
        and event_counts.get("AMENDMENT08_CORRIGENDUM_INSTALLED", 0) == 1
        and event_counts.get("AMENDMENT07_CORRIGENDUM_INSTALLED", 0) == 0
        and event_counts.get("THRESHOLD_TUNING_CALLED", 0) == 0
        and all(index < audit_index for index, _ in fit_events)
    ):
        raise Amendment06IntegrityError("Execution ledger forbidden/cache calls differ")
    return {
        "event_counts": dict(event_counts),
        "completed_training": completed_training,
        "training_fit_count": len(fit_events),
        "training_failure_count": len(training_failures),
        "scoring_failure_count": len(scoring_failures),
        "reporting_failure_count": len(reporting_failures),
        "prediction_only_reload_events": (
            event_counts.get("AUDIT_SCORING_ATTEMPT_STARTED", 0)
            + event_counts.get("COMPLETED_AUDIT_SCORING_SKIPPED", 0)
        ),
        "freeze_index": frozen_index, "audit_index": audit_index,
        "bootstrap_index": bootstrap_events[0][0],
        "reporting_index": reporting_completions[0][0],
    }


def build_execution_ledger(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    if run_dir.name != A08_TARGET_RUN_ID:
        return _build_legacy_execution_ledger(run_dir)
    cutoff_path = run_dir / A08_RAW_JOURNAL_CUTOFF_RELATIVE
    cutoff = validate_amendment08_raw_journal_cutoff(run_dir)
    events = [dict(record["event"]) for record in cutoff["records"]]
    lineage = _amendment08_frozen_lineage_bindings(run_dir)
    corrected_history = _validate_amendment08_corrected_training_history(
        run_dir=run_dir, overlay=lineage, allow_unterminated=False,
    )
    if not (
        corrected_history["completed_variants"] == list(TRAINED_VARIANTS)
        and corrected_history["next_variant"] is None
    ):
        raise Amendment06IntegrityError(
            "Execution ledger corrected Amendment 08 history is incomplete"
        )
    lifecycle = _validate_amendment08_cutoff_lifecycle(
        run_dir=run_dir, events=events, lineage=lineage,
    )
    current_views = read_amendment08_journal_views(run_dir)
    current_raw = current_views["raw_events"]
    if current_raw[:len(events)] != events:
        raise Amendment06IntegrityError(
            "Current physical Journal no longer extends the frozen cutoff"
        )
    forbidden_after_cutoff = {
        "CACHE_GENERATION", "CACHE_GENERATION_INTENT", "CACHE_LOAD",
        "CACHE_REMATERIALIZATION", "CACHE_REMATERIALIZATION_INTENT",
        "TRAINING_ATTEMPT_STARTED", "SCIENTIFIC_FIT_CALLED",
        "TRAINING_ATTEMPT_FAILED", "TRAINING_PUBLICATION_INTERRUPTED",
        "TRAINING_COMPLETION_COMMITTED", "COMPLETED_VARIANT_SKIPPED",
        "MODELS_FROZEN", "EXTERNAL_AUDIT_OPENED",
        "AUDIT_SCORING_ATTEMPT_STARTED", "AUDIT_SCORING_ATTEMPT_FAILED",
        "AUDIT_SCORING_PUBLICATION_INTERRUPTED",
        "AUDIT_SCORING_COMPLETION_COMMITTED",
        "COMPLETED_AUDIT_SCORING_SKIPPED", "REPORTING_ATTEMPT_STARTED",
        "BOOTSTRAP_RESAMPLING", "REPORTING_ATTEMPT_FAILED",
        "REPORTING_PUBLICATION_INTERRUPTED", "REPORTING_COMPLETION_COMMITTED",
        "THRESHOLD_TUNING_CALLED", "ARTIFACT_PUBLICATION_INTENT",
        "ARTIFACT_PUBLISHED", "ARTIFACT_QUARANTINE_INTENT",
        "ARTIFACT_QUARANTINE_COMMITTED",
    }
    if any(
        event.get("event") in forbidden_after_cutoff
        for event in current_raw[len(events):]
    ):
        raise Amendment06IntegrityError(
            "A scientific lifecycle event exists after the reporting cutoff"
        )
    current_raw_lines = b"".join(
        strict_full_canonical_json_bytes(event) + b"\n" for event in current_raw
    )
    counts = lifecycle["event_counts"]
    cutoff_sha256 = sha256_file(cutoff_path)
    result = {
        "run_id": run_dir.name, "run_kind": RUN_KIND, "status": "PASS",
        **lineage,
        "journal_schema": "ATOMIC_CONTIGUOUS_HASH_CHAINED_PER_EVENT_V1",
        "journal_phase_order_validation": "PASS",
        "completed_trained_models": 10,
        "completed_official_runnable_models": 9,
        "completed_exploratory_models": 1,
        "completed_true_no_pca_models": 0,
        "successful_training_completions": 10,
        "stability_fits": 0, "threshold_tuning_calls": 0,
        "post_audit_training_calls": 0,
        "scientific_fit_attempts": lifecycle["training_fit_count"],
        "failed_unpublished_training_attempts": lifecycle[
            "training_failure_count"
        ],
        "failed_unpublished_scoring_attempts": lifecycle[
            "scoring_failure_count"
        ],
        "failed_unpublished_reporting_attempts": lifecycle[
            "reporting_failure_count"
        ],
        "prediction_only_reload_events": lifecycle[
            "prediction_only_reload_events"
        ],
        "cache_generation_events": counts.get("CACHE_GENERATION", 0),
        "cache_load_events": counts.get("CACHE_LOAD", 0),
        "cache_rematerialization_events": counts.get("CACHE_REMATERIALIZATION", 0),
        "amendment08_corrigendum_installations": counts.get(
            "AMENDMENT08_CORRIGENDUM_INSTALLED", 0
        ),
        "external_audit_open_events": counts.get("EXTERNAL_AUDIT_OPENED", 0),
        "bootstrap_resampling_events": counts.get("BOOTSTRAP_RESAMPLING", 0),
        "package_attempts_at_scientific_freeze": 0,
        "package_attempts_tracked_adjacent_to_frozen_run": True,
        "ownership_journal_path": str(_journal_path(run_dir)),
        "ownership_journal_prefix_sha256_at_scientific_freeze": cutoff[
            "raw_journal_canonical_json_lines_sha256_at_reporting_cutoff"
        ],
        "ownership_journal_event_count_at_scientific_freeze": cutoff[
            "raw_journal_event_count_at_reporting_cutoff"
        ],
        "amendment08_raw_journal_cutoff_sha256": cutoff_sha256,
        "amendment08_raw_journal_at_reporting_cutoff_sha256": cutoff_sha256,
        "genuine_reporting_completion_event_index": cutoff[
            "genuine_reporting_completion_event_index"
        ],
        "genuine_reporting_completion_event_raw_sha256": cutoff[
            "genuine_reporting_completion_event_raw_sha256"
        ],
        "genuine_reporting_completion_event_canonical_sha256": cutoff[
            "genuine_reporting_completion_event_canonical_sha256"
        ],
        "current_physical_journal_event_count": len(current_raw),
        "current_physical_journal_head_sha256": sha256_bytes(
            strict_full_canonical_json_bytes(current_raw[-1])
        ),
        "current_physical_journal_canonical_json_lines_sha256": sha256_bytes(
            current_raw_lines
        ),
    }
    for key in (
        "raw_journal_event_count_at_reporting_cutoff",
        "raw_journal_head_sha256_at_reporting_cutoff",
        "raw_journal_canonical_json_lines_sha256_at_reporting_cutoff",
        "semantic_journal_event_count_at_reporting_cutoff",
        "semantic_journal_canonical_json_lines_sha256_at_reporting_cutoff",
        "classified_test_contamination_event_count",
        "classified_test_contamination_event_indices",
        "classified_test_contamination_raw_sha256",
        "classified_test_contamination_canonical_sha256",
        "raw_reporting_completion_events", "semantic_reporting_completion_events",
    ):
        result[key] = cutoff[key]
    return result


MODEL_BUNDLE_GLOBALS = frozenset({
    "config/run_identity.lock.json",
    "config/training_configuration.lock.json",
    "config/variant_feature_sets.json",
    "config/feature_manifest.csv",
    "config/feature_dependency_graph.json",
    "config/locked_split_identity.json",
    "splits/row_split_manifest.csv.gz",
    "provenance/amendment06_external_review_authorization.json",
    "provenance/scientific_execution_code_manifest.tsv",
    "provenance/source_input_hashes_pre.tsv",
    "provenance/source_input_hashes_post.tsv",
    "provenance/frozen_full_imputer.json",
    "provenance/frozen_full_resampling_manifest.json",
    "provenance/full_models_frozen_manifest.json",
    "provenance/code_snapshot/amendment03_compat.py",
    "models/no_pca/STATUS.txt",
    "models/no_pca/blocked_evidence.json",
    *(
        f"provenance/state/{index:02d}_{phase}.json"
        for index, phase in enumerate(RUN_PHASES)
    ),
})


def _models_bundle_flag(relative: str) -> bool:
    if (
        relative in MODEL_BUNDLE_GLOBALS
        or relative in A07_MODEL_BUNDLE_GLOBALS
        or relative in A08_MODEL_BUNDLE_GLOBALS
    ):
        return True
    parts = Path(relative).parts
    return bool(
        len(parts) == 3 and parts[0] == "models" and parts[1] in TRAINED_VARIANTS
        and parts[2] in {
            "model.ubj", "feature_list.txt", "model_metadata.json",
            "transform_binding.json", "training_completion_manifest.json",
            "external_scoring_manifest.json", "STATUS.txt",
        }
    )


def _artifact_role(relative: str) -> str:
    if relative.endswith(".ubj"):
        return "SCIENTIFIC_MODEL_BINARY"
    if relative.startswith("predictions/"):
        return "SCIENTIFIC_PREDICTION"
    if relative.startswith("metrics/") or relative.startswith("statistics/"):
        return "SCIENTIFIC_METRIC_OR_STATISTIC"
    if relative.startswith("tables/") or relative.startswith("figures/") or relative.startswith("docs/"):
        return "SCIENTIFIC_REPORT"
    if relative.startswith("logs/") or relative.startswith("timing/"):
        return "EXECUTION_RESOURCE_EVIDENCE"
    if relative.startswith("models/"):
        return "MODEL_CONTRACT_OR_STATUS"
    if relative.startswith("config/") or relative.startswith("splits/"):
        return "LOCKED_CONFIGURATION_OR_SPLIT"
    return "PROVENANCE_AND_RUN_CONTROL"


def _walk_run_regular_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    for directory, names, filenames in os.walk(run_dir, followlinks=False):
        root = Path(directory)
        for name in names:
            child = root / name
            if child.is_symlink():
                raise Amendment06IntegrityError(f"Frozen Run contains symlink directory: {child}")
        for name in filenames:
            child = root / name
            mode = os.lstat(child).st_mode
            if not stat.S_ISREG(mode):
                raise Amendment06IntegrityError(f"Frozen Run contains unsafe file: {child}")
            files.append(child)
    return sorted(files, key=lambda path: path.relative_to(run_dir).as_posix().encode("utf-8"))


def expected_scientific_run_files() -> frozenset[str]:
    per_variant = {
        relative
        for variant in TRAINED_VARIANTS
        for relative in (
            *_training_relatives(variant), _training_completion_relative(variant),
            *_scoring_relatives(variant), _scoring_completion_relative(variant),
        )
    }
    snapshots = {
        *(
            f"provenance/code_snapshot/{Path(relative).name}"
            for relative in SCIENTIFIC_CODE_RELATIVES
        ),
        *(
            f"provenance/tests_snapshot/{Path(relative).name}"
            for relative in SCIENTIFIC_TEST_RELATIVES
        ),
    }
    expected = frozenset({
        "RUN_STATUS.txt", "BLOCKERS.md",
        "config/run_identity.lock.json",
        "config/training_configuration.lock.json",
        "config/variant_feature_sets.json",
        "config/feature_manifest.csv",
        "config/feature_dependency_graph.json",
        "config/locked_split_identity.json",
        "splits/row_split_manifest.csv.gz",
        "models/no_pca/STATUS.txt", "models/no_pca/blocked_evidence.json",
        "provenance/amendment06_prompt_snapshot.txt",
        "provenance/amendment06_external_review_authorization.json",
        "provenance/amendment06_code_lineage.json",
        "provenance/scientific_execution_code_manifest.tsv",
        "provenance/package_validator_code_manifest_initial.tsv",
        "provenance/source_input_hashes_pre.tsv",
        "provenance/source_input_hashes_post.tsv",
        "provenance/cuda_active_fit_probe.json",
        "provenance/cuda_environment.txt",
        "provenance/frozen_full_imputer.json",
        "provenance/frozen_full_resampling_manifest.json",
        "provenance/full_models_frozen_manifest.json",
        "provenance/external_audit_exclusion_and_opening.json",
        "provenance/reporting_completion_manifest.json",
        "provenance/metric_recomputation.json",
        "provenance/execution_ledger.json",
        "provenance/strict_json_validation.json",
        "docs/PROTOCOL_AMENDMENT_06.md",
        *(f"provenance/state/{index:02d}_{phase}.json" for index, phase in enumerate(RUN_PHASES)),
        *REPORTING_REQUIRED_RELATIVES,
        *per_variant, *snapshots, *A08_OVERLAY_RUN_RELATIVES,
        A08_RAW_JOURNAL_CUTOFF_RELATIVE,
    })
    if len(expected) != 236:
        raise Amendment06IntegrityError(
            f"Internal Amendment 08 final member contract differs: {len(expected)}"
        )
    return expected


def output_manifest_bytes(
    run_dir: Path, *, require_exact_contract: bool = False,
) -> bytes:
    rows: list[dict[str, Any]] = []
    files = _walk_run_regular_files(run_dir)
    relatives = {
        path.relative_to(run_dir).as_posix() for path in files
        if path.relative_to(run_dir).as_posix() != "OUTPUT_MANIFEST_FINAL.tsv"
    }
    if require_exact_contract and relatives != expected_scientific_run_files():
        raise Amendment06IntegrityError(
            "Final scientific Run member set differs: "
            f"missing={sorted(expected_scientific_run_files() - relatives)}, "
            f"extra={sorted(relatives - expected_scientific_run_files())}"
        )
    for path in files:
        relative = path.relative_to(run_dir).as_posix()
        if relative == "OUTPUT_MANIFEST_FINAL.tsv":
            continue
        rows.append({
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "artifact_role": _artifact_role(relative),
            "include_in_review_bundle": not relative.endswith(".ubj"),
            "include_in_models_bundle": _models_bundle_flag(relative),
        })
    frame = pd.DataFrame(rows, columns=OUTPUT_MANIFEST_COLUMNS)
    return frame.to_csv(index=False, sep="\t", lineterminator="\n").encode("utf-8")


def validate_output_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "OUTPUT_MANIFEST_FINAL.tsv"
    if not _safe_regular(path):
        raise Amendment06IntegrityError("Final output manifest is absent/unsafe")
    expected = output_manifest_bytes(run_dir, require_exact_contract=True)
    if path.read_bytes() != expected:
        raise Amendment06IntegrityError("Final output manifest differs from immutable Run tree")
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if list(frame.columns) != list(OUTPUT_MANIFEST_COLUMNS):
        raise Amendment06IntegrityError("Final output manifest schema differs")
    if not (
        set(frame.include_in_review_bundle) <= {"True", "False"}
        and set(frame.include_in_models_bundle) <= {"True", "False"}
        and frame.relative_path.is_unique
        and "OUTPUT_MANIFEST_FINAL.tsv" not in set(frame.relative_path)
    ):
        raise Amendment06IntegrityError("Final output manifest flags/paths differ")
    return {
        "status": "PASS", "rows": len(frame), "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _final_status(run_dir: Path) -> str:
    return (
        f"RUN_ID={run_dir.name}\nRUN_KIND={RUN_KIND}\n"
        "RUN_STATE=FULL_SCIENTIFIC_COMPLETE\n"
        f"FULL_AUTHORIZATION={AUTHORIZATION}\n"
        "FULL_MODELS_FROZEN=YES\nEXTERNAL_AUDIT_OPENED=YES\n"
        "OFFICIAL_RUNNABLE_MODELS_COMPLETED=9_OF_9\n"
        "EXPLORATORY_MODELS_COMPLETED=1_OF_1\n"
        "OFFICIAL_TRUE_NO_PCA_STATUS=NOT_RUN_BLOCKED_RAW_EMBEDDINGS_UNAVAILABLE\n"
        "PAIRED_CARD_BOOTSTRAP_REPLICATES=10000\n"
        "CPU_FALLBACK_DETECTED=NO\nSCIENTIFIC_RESULTS_CLAIMED=YES\n"
        "PACKAGE_PUBLICATION_STATE=ADJACENT_SEPARATELY_RETRYABLE\n"
    )


def _finalize_legacy_scientific_run(
    *, run_dir: Path, audit: ExternalAuditPopulation,
) -> dict[str, Any]:
    manifest_path = run_dir / "OUTPUT_MANIFEST_FINAL.tsv"
    if manifest_path.exists():
        if _current_run_state(run_dir) != "FULL_SCIENTIFIC_COMPLETE":
            raise Amendment06IntegrityError("Final output manifest exists before final scientific state")
        return validate_output_manifest(run_dir)
    metric_path = run_dir / "provenance/metric_recomputation.json"
    current = _current_run_state(run_dir)
    if current not in {"FULL_EXTERNAL_AUDIT_SCORED", "FULL_SCIENTIFIC_COMPLETE"}:
        raise Amendment06IntegrityError("Scientific finalization state differs")
    if current == "FULL_SCIENTIFIC_COMPLETE":
        _validate_reporting_completion_receipt(run_dir)
        _validate_metric_recomputation_receipt(run_dir)
    else:
        _validate_reporting_completion(run_dir)
        metric_payload = build_metric_recomputation(run_dir=run_dir, audit=audit)
        if metric_path.exists():
            if strict_full_load_file(metric_path) != metric_payload:
                raise Amendment06IntegrityError("Final metric recomputation evidence drifted")
        else:
            publish_strict_json_no_clobber(metric_path, metric_payload)
        _validate_metric_recomputation_receipt(run_dir)
    _snapshot_source_post(run_dir)
    current = _current_run_state(run_dir)
    if current == "FULL_EXTERNAL_AUDIT_SCORED":
        _set_run_state(run_dir, "FULL_SCIENTIFIC_COMPLETE")
    elif current != "FULL_SCIENTIFIC_COMPLETE":
        raise Amendment06IntegrityError("Scientific finalization state differs")
    ledger_path = run_dir / "provenance/execution_ledger.json"
    ledger = build_execution_ledger(run_dir)
    if ledger_path.exists():
        if strict_full_load_file(ledger_path) != ledger:
            raise Amendment06IntegrityError("Final execution ledger drifted")
    else:
        publish_strict_json_no_clobber(ledger_path, ledger)
    for status_path, status_bytes in (
        (run_dir / "RUN_STATUS.txt", _final_status(run_dir).encode("utf-8")),
        (run_dir / "BLOCKERS.md", b"# Blockers\n\nNone.\n"),
    ):
        if status_path.exists():
            if not _safe_regular(status_path, sha256_bytes(status_bytes)):
                raise Amendment06IntegrityError(
                    f"Final scientific status artifact drifted: {status_path}"
                )
        else:
            publish_bytes_no_clobber(status_path, status_bytes)
    strict_path = run_dir / "provenance/strict_json_validation.json"
    json_paths = sorted(
        (path for path in run_dir.rglob("*.json") if path != strict_path),
        key=lambda path: path.relative_to(run_dir).as_posix().encode("utf-8"),
    )
    strict_evidence = validate_strict_json_tree(run_dir, include=json_paths)
    strict_evidence.update({
        "run_id": run_dir.name, "self_excluded_by_non_circular_contract": True,
        "new_full_json_nonfinite_constants": 0,
    })
    if strict_path.exists():
        if strict_full_load_file(strict_path) != strict_evidence:
            raise Amendment06IntegrityError("Strict-JSON validation evidence drifted")
    else:
        publish_strict_json_no_clobber(strict_path, strict_evidence)
    strict_full_load_file(strict_path)
    publish_bytes_no_clobber(
        manifest_path, output_manifest_bytes(run_dir, require_exact_contract=True),
    )
    result = validate_output_manifest(run_dir)
    _progress("FULL_SCIENTIFIC_FINALIZATION", "COMPLETE", status="PASS", manifest_sha256=result["sha256"])
    return result


def _amendment08_crash_after(
    requested: str | None, completed_boundary: str,
) -> None:
    allowed = {
        "prerequisites", "cutoff", "state", "ledger", "status", "blockers",
        "strict_json", "output_manifest",
    }
    if requested is not None and requested not in allowed:
        raise Amendment06IntegrityError(
            f"Unknown Amendment 08 finalization crash boundary: {requested}"
        )
    if requested == completed_boundary:
        raise Amendment06ResumeError(
            f"SIMULATED_AMENDMENT08_CRASH_AFTER_{completed_boundary.upper()}"
        )


def _publish_or_replace_amendment08_final_control(
    *, run_dir: Path, relative: str, desired: bytes,
    allowed_existing: bytes | None = None,
) -> None:
    path = run_dir / _safe_run_relative(relative)
    desired_sha256 = sha256_bytes(desired)
    if _safe_regular(path, desired_sha256) and path.stat().st_size == len(desired):
        return
    if not path.exists():
        publish_bytes_no_clobber(path, desired)
        return
    if allowed_existing is None or not _safe_regular(
        path, sha256_bytes(allowed_existing)
    ):
        raise Amendment06IntegrityError(
            f"Amendment 08 final control artifact drifted: {relative}"
        )
    staging_root = EXPECTED_RUNTIME_ROOT / "finalization" / run_dir.name
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / f"{Path(relative).name}.{desired_sha256}.bin"
    if staging.exists():
        if not (
            _safe_regular(staging, desired_sha256)
            and staging.stat().st_size == len(desired)
        ):
            raise Amendment06IntegrityError(
                f"Amendment 08 final-control staging drifted: {relative}"
            )
    else:
        publish_bytes_no_clobber(staging, desired)
    if staging.stat().st_dev != path.stat().st_dev:
        raise Amendment06IntegrityError(
            f"Amendment 08 final-control replacement crosses filesystems: {relative}"
        )
    if not _safe_regular(path, sha256_bytes(allowed_existing)):
        raise Amendment06IntegrityError(
            f"Amendment 08 final-control source changed: {relative}"
        )
    os.replace(staging, path)
    _fsync_parent(path)
    if not (
        _safe_regular(path, desired_sha256) and path.stat().st_size == len(desired)
    ):
        raise Amendment06IntegrityError(
            f"Amendment 08 final-control replacement failed: {relative}"
        )


def _amendment08_finalization_handoff_ready(run_dir: Path) -> bool:
    run_dir = Path(run_dir).resolve()
    if run_dir.name != A08_TARGET_RUN_ID:
        return False
    cutoff_path = run_dir / A08_RAW_JOURNAL_CUTOFF_RELATIVE
    final_marker = _state_marker_path(run_dir, "FULL_SCIENTIFIC_COMPLETE")
    if cutoff_path.exists() or final_marker.exists():
        return True
    reporting_path = _reporting_completion_path(run_dir)
    prerequisite_paths = (
        reporting_path,
        run_dir / "provenance/metric_recomputation.json",
        run_dir / "provenance/source_input_hashes_post.tsv",
        run_dir / "provenance/full_models_frozen_manifest.json",
        run_dir / "provenance/external_audit_exclusion_and_opening.json",
    )
    if not all(path.exists() for path in prerequisite_paths):
        return False
    semantic_reporting = [
        event for event in _scientific_journal_events(run_dir)
        if event.get("event") == "REPORTING_COMPLETION_COMMITTED"
    ]
    if len(semantic_reporting) != 1:
        raise Amendment06IntegrityError(
            "Amendment 08 finalization handoff reporting cardinality differs"
        )
    _validate_amendment08_frozen_prerequisite_receipts(run_dir)
    return True


def _validate_amendment08_finalization_identity(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    if run_dir != A08_TARGET_RUN.resolve() or existing_full_runs() != [run_dir]:
        raise Amendment06IntegrityError(
            "Amendment 08 finalization target/one-Run identity differs"
        )
    identity = strict_full_load_file(run_dir / "config/run_identity.lock.json")
    receipt = strict_full_load_file(_run_receipt_path())
    identity_sha256 = sha256_file(run_dir / "config/run_identity.lock.json")
    path_matches = receipt.get("run_path") == str(run_dir)
    if not path_matches:
        path_matches = _amendment08_rehearsal_receipt_path_matches(
            run_dir=run_dir, receipt=receipt, identity_sha256=identity_sha256,
        )
    if not (
        identity.get("run_id") == run_dir.name
        and identity.get("run_kind") == RUN_KIND
        and identity.get("state") == "FULL_IN_PROGRESS"
        and identity.get("full_authorization") == AUTHORIZATION
        and identity.get("scientific_execution_code_manifest_sha256")
        == A07_BASE_SCIENTIFIC_MANIFEST_SHA256
        and receipt.get("authorization") == AUTHORIZATION
        and receipt.get("authorization_consumed") is True
        and receipt.get("run_id") == run_dir.name
        and path_matches
        and receipt.get("run_identity_sha256") == identity_sha256
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 frozen finalization identity differs"
        )
    _amendment08_frozen_lineage_bindings(run_dir)
    _current_run_state(run_dir)
    return identity


def finalize_scientific_run(
    *, run_dir: Path, audit: ExternalAuditPopulation | None = None,
    crash_after: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    if run_dir.name != A08_TARGET_RUN_ID:
        if audit is None or crash_after is not None:
            raise Amendment06IntegrityError(
                "Legacy scientific finalization requires its Audit population"
            )
        return _finalize_legacy_scientific_run(run_dir=run_dir, audit=audit)
    _validate_amendment08_finalization_identity(run_dir)
    manifest_path = run_dir / "OUTPUT_MANIFEST_FINAL.tsv"
    if manifest_path.exists():
        result = validate_final_scientific_run(run_dir)
        _amendment08_crash_after(crash_after, "output_manifest")
        return result

    current = _current_run_state(run_dir)
    if current not in {"FULL_EXTERNAL_AUDIT_SCORED", "FULL_SCIENTIFIC_COMPLETE"}:
        raise Amendment06IntegrityError("Amendment 08 finalization state differs")
    cutoff_path = run_dir / A08_RAW_JOURNAL_CUTOFF_RELATIVE
    final_marker_path = _state_marker_path(run_dir, "FULL_SCIENTIFIC_COMPLETE")
    frozen_handoff_active = cutoff_path.exists() or final_marker_path.exists()
    if cutoff_path.exists():
        validate_amendment08_raw_journal_cutoff(run_dir)
    metric_path = run_dir / "provenance/metric_recomputation.json"
    if not metric_path.exists():
        if (
            frozen_handoff_active
            or audit is None
            or current != "FULL_EXTERNAL_AUDIT_SCORED"
        ):
            raise Amendment06IntegrityError(
                "Amendment 08 cannot generate missing scientific prerequisites "
                "on the finalization-only route"
            )
        _validate_reporting_completion(run_dir)
        metric_payload = build_metric_recomputation(run_dir=run_dir, audit=audit)
        publish_strict_json_no_clobber(metric_path, metric_payload)
    _validate_metric_recomputation_receipt(run_dir)
    if not (run_dir / "provenance/source_input_hashes_post.tsv").exists():
        if frozen_handoff_active or audit is None:
            raise Amendment06IntegrityError(
                "Amendment 08 finalization-only route cannot create missing source evidence"
            )
        _snapshot_source_post(run_dir)
    prerequisites = _validate_amendment08_frozen_prerequisite_receipts(run_dir)
    if prerequisites["status"] != "PASS_FROZEN_RECEIPTS_ONLY":
        raise Amendment06IntegrityError("Amendment 08 final prerequisite gate differs")
    _amendment08_crash_after(crash_after, "prerequisites")

    cutoff = _publish_or_validate_amendment08_raw_journal_cutoff(run_dir)
    _amendment08_crash_after(crash_after, "cutoff")

    marker = cutoff["planned_full_scientific_complete_marker"]
    marker_bytes = strict_full_json_bytes(marker)
    marker_path = _state_marker_path(run_dir, "FULL_SCIENTIFIC_COMPLETE")
    if marker_path.exists():
        if not (
            _safe_regular(marker_path, sha256_bytes(marker_bytes))
            and marker_path.read_bytes() == marker_bytes
        ):
            raise Amendment06IntegrityError(
                "Amendment 08 final state marker conflicts with cutoff"
            )
    else:
        publish_bytes_no_clobber(marker_path, marker_bytes)
    if _current_run_state(run_dir) != "FULL_SCIENTIFIC_COMPLETE":
        raise Amendment06IntegrityError("Amendment 08 final state publication failed")
    validate_amendment08_raw_journal_cutoff(run_dir)
    _amendment08_crash_after(crash_after, "state")

    ledger_path = run_dir / "provenance/execution_ledger.json"
    ledger = build_execution_ledger(run_dir)
    if ledger_path.exists():
        if strict_full_load_file(ledger_path) != ledger:
            raise Amendment06IntegrityError("Amendment 08 execution ledger drifted")
    else:
        publish_strict_json_no_clobber(ledger_path, ledger)
    _amendment08_crash_after(crash_after, "ledger")

    _publish_or_replace_amendment08_final_control(
        run_dir=run_dir, relative="RUN_STATUS.txt",
        desired=_final_status(run_dir).encode("utf-8"),
        allowed_existing=_initial_status(run_dir.name).encode("utf-8"),
    )
    _amendment08_crash_after(crash_after, "status")
    _publish_or_replace_amendment08_final_control(
        run_dir=run_dir, relative="BLOCKERS.md", desired=b"# Blockers\n\nNone.\n",
    )
    _amendment08_crash_after(crash_after, "blockers")

    strict_path = run_dir / "provenance/strict_json_validation.json"
    json_paths = sorted(
        (path for path in run_dir.rglob("*.json") if path != strict_path),
        key=lambda path: path.relative_to(run_dir).as_posix().encode("utf-8"),
    )
    strict_evidence = validate_strict_json_tree(run_dir, include=json_paths)
    strict_evidence.update({
        "run_id": run_dir.name, "self_excluded_by_non_circular_contract": True,
        "new_full_json_nonfinite_constants": 0,
    })
    if strict_path.exists():
        if strict_full_load_file(strict_path) != strict_evidence:
            raise Amendment06IntegrityError(
                "Amendment 08 strict-JSON validation evidence drifted"
            )
    else:
        publish_strict_json_no_clobber(strict_path, strict_evidence)
    _amendment08_crash_after(crash_after, "strict_json")

    manifest_bytes = output_manifest_bytes(run_dir, require_exact_contract=True)
    if manifest_path.exists():
        if not _safe_regular(manifest_path, sha256_bytes(manifest_bytes)):
            raise Amendment06IntegrityError("Amendment 08 output manifest drifted")
    else:
        publish_bytes_no_clobber(manifest_path, manifest_bytes)
    result = validate_final_scientific_run(run_dir)
    _amendment08_crash_after(crash_after, "output_manifest")
    _progress(
        "FULL_SCIENTIFIC_FINALIZATION", "COMPLETE", status="PASS",
        manifest_sha256=result["sha256"],
    )
    return result


def validate_final_scientific_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    if run_dir.name == A08_TARGET_RUN_ID:
        _validate_amendment08_finalization_identity(run_dir)
        if _current_run_state(run_dir) != "FULL_SCIENTIFIC_COMPLETE":
            raise Amendment06IntegrityError(
                "Amendment 08 Run is not scientifically complete"
            )
        cutoff = validate_amendment08_raw_journal_cutoff(run_dir)
        ledger_path = run_dir / "provenance/execution_ledger.json"
        observed_ledger = strict_full_load_file(ledger_path)
        expected_ledger = build_execution_ledger(run_dir)
        cutoff_sha256 = sha256_file(
            run_dir / A08_RAW_JOURNAL_CUTOFF_RELATIVE
        )
        if not (
            observed_ledger == expected_ledger
            and observed_ledger.get("amendment08_raw_journal_cutoff_sha256")
            == cutoff_sha256
            and observed_ledger.get(
                "amendment08_raw_journal_at_reporting_cutoff_sha256"
            ) == cutoff_sha256
            and all(
                observed_ledger.get(key) == cutoff[key]
                for key in (
                    "raw_journal_event_count_at_reporting_cutoff",
                    "raw_journal_head_sha256_at_reporting_cutoff",
                    "raw_journal_canonical_json_lines_sha256_at_reporting_cutoff",
                    "semantic_journal_event_count_at_reporting_cutoff",
                    "semantic_journal_canonical_json_lines_sha256_at_reporting_cutoff",
                    "classified_test_contamination_event_count",
                    "classified_test_contamination_event_indices",
                    "classified_test_contamination_raw_sha256",
                    "classified_test_contamination_canonical_sha256",
                    "raw_reporting_completion_events",
                    "semantic_reporting_completion_events",
                )
            )
        ):
            raise Amendment06IntegrityError(
                "Amendment 08 final execution-ledger binding differs"
            )
        if not (
            _safe_regular(
                run_dir / "RUN_STATUS.txt",
                sha256_bytes(_final_status(run_dir).encode("utf-8")),
            )
            and _safe_regular(
                run_dir / "BLOCKERS.md", sha256_bytes(b"# Blockers\n\nNone.\n"),
            )
        ):
            raise Amendment06IntegrityError(
                "Amendment 08 final status/blocker bytes differ"
            )
        strict_path = run_dir / "provenance/strict_json_validation.json"
        json_paths = sorted(
            (path for path in run_dir.rglob("*.json") if path != strict_path),
            key=lambda path: path.relative_to(run_dir).as_posix().encode("utf-8"),
        )
        strict_expected = validate_strict_json_tree(run_dir, include=json_paths)
        strict_expected.update({
            "run_id": run_dir.name,
            "self_excluded_by_non_circular_contract": True,
            "new_full_json_nonfinite_constants": 0,
        })
        if strict_full_load_file(strict_path) != strict_expected:
            raise Amendment06IntegrityError(
                "Amendment 08 final strict-JSON evidence differs"
            )
        _validate_reporting_completion_receipt(run_dir)
        _validate_metric_recomputation_receipt(run_dir)
        result = validate_output_manifest(run_dir)
        if result["rows"] != 236:
            raise Amendment06IntegrityError(
                "Amendment 08 final output-manifest row count differs"
            )
        return result
    if _current_run_state(run_dir) != "FULL_SCIENTIFIC_COMPLETE":
        raise Amendment06IntegrityError("Full Run is not scientifically complete")
    status = {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in (run_dir / "RUN_STATUS.txt").read_text().splitlines()
        if "=" in line
    }
    if status.get("RUN_STATE") != "FULL_SCIENTIFIC_COMPLETE":
        raise Amendment06IntegrityError("Final RUN_STATUS differs")
    strict_full_load_file(run_dir / "provenance/strict_json_validation.json")
    _validate_reporting_completion_receipt(run_dir)
    _validate_metric_recomputation_receipt(run_dir)
    return validate_output_manifest(run_dir)


def _amendment08_run_tree_rows(run_dir: Path) -> tuple[tuple[Any, ...], ...]:
    run_dir = Path(run_dir).resolve()
    root_stat = os.lstat(run_dir)
    if not stat.S_ISDIR(root_stat.st_mode) or run_dir.is_symlink():
        raise Amendment06IntegrityError(
            "Amendment 08 rehearsal handoff Run is not a real directory"
        )
    rows: list[tuple[Any, ...]] = []
    for current, directories, filenames in os.walk(
        run_dir, topdown=True, followlinks=False,
    ):
        current_path = Path(current)
        directories.sort()
        filenames.sort()
        for name in directories:
            path = current_path / name
            info = os.lstat(path)
            relative = path.relative_to(run_dir).as_posix()
            if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
                raise Amendment06IntegrityError(
                    f"Amendment 08 rehearsal Run has an unsafe directory: {relative}"
                )
            rows.append(("D", relative, stat.S_IMODE(info.st_mode)))
        for name in filenames:
            path = current_path / name
            info = os.lstat(path)
            relative = path.relative_to(run_dir).as_posix()
            if not stat.S_ISREG(info.st_mode) or path.is_symlink():
                raise Amendment06IntegrityError(
                    f"Amendment 08 rehearsal Run has an unsafe file: {relative}"
                )
            rows.append((
                "F", relative, stat.S_IMODE(info.st_mode), int(info.st_size),
                sha256_file(path),
            ))
    return tuple(sorted(rows, key=lambda row: (row[1], row[0])))


def amendment08_run_tree_sha256(run_dir: Path) -> str:
    rows = _amendment08_run_tree_rows(run_dir)
    payload = b"".join(
        ("\t".join(map(str, row)) + "\n").encode("utf-8") for row in rows
    )
    return sha256_bytes(payload)


def _amendment08_resume_rehearsal_handoff_path() -> Path:
    _, runtime_root = _amendment08_rehearsal_context_paths()
    return runtime_root.parent / "amendment08_resume_rehearsal_handoff.json"


def validate_amendment08_resume_rehearsal_handoff(
    handoff_path: Path, *, run_dir: Path | None = None,
) -> dict[str, Any]:
    context_run, runtime_root = _amendment08_rehearsal_context_paths()
    selected = context_run if run_dir is None else Path(run_dir).resolve()
    path = Path(handoff_path)
    expected_path = _amendment08_resume_rehearsal_handoff_path()
    if not (
        selected == context_run
        and path.is_absolute()
        and path == expected_path
        and _safe_regular(path)
        and path.resolve(strict=True) == path
        and selected == runtime_root.parent / "results" / A08_TARGET_RUN_ID
        and runtime_root == runtime_root.parent / "runtime"
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 resume rehearsal handoff layout differs"
        )
    payload = _require_exact_keys(
        strict_full_load_file(path), set(A08_REHEARSAL_HANDOFF_KEYS),
        label="Amendment 08 resume rehearsal handoff",
    )
    manifest = validate_final_scientific_run(selected)
    cache = _amendment08_rehearsal_cache_clone_evidence(
        selected, rehash_source_and_clone=True,
    )
    production_inventory = amendment08_production_live_journal_inventory_bytes()
    production_inventory_sha256 = sha256_bytes(production_inventory)
    expected = {
        "schema": A08_REHEARSAL_HANDOFF_SCHEMA, "status": "PASS",
        "run_id": selected.name, "run_path": str(selected),
        "runtime_path": str(runtime_root),
        "output_manifest_sha256": manifest["sha256"],
        "output_manifest_rows": 236,
        "run_tree_sha256": amendment08_run_tree_sha256(selected),
        "cache_source_inventory_sha256": cache[
            "cache_source_inventory_sha256"
        ],
        "cache_clone_inventory_sha256": cache[
            "cache_clone_inventory_sha256"
        ],
        "live_journal_before_sha256": A08_PRELIVE_JOURNAL_SENTINEL_SHA256,
        "live_journal_after_sha256": production_inventory_sha256,
    }
    if not (
        dict(payload) == expected
        and len(_walk_run_regular_files(selected)) == 237
        and cache["cache_clone_distinct_inodes"] is True
        and cache["cache_source_inventory_sha256"]
        == cache["cache_clone_inventory_sha256"]
        and production_inventory_sha256 == A08_PRELIVE_JOURNAL_SENTINEL_SHA256
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 resume rehearsal handoff binding differs"
        )
    return {
        **expected, "handoff_path": str(path),
        "handoff_sha256": sha256_file(path),
    }


def publish_amendment08_resume_rehearsal_handoff(
    *, run_dir: Path, handoff_path: Path,
    live_journal_before_sha256: str, live_journal_after_sha256: str,
) -> dict[str, Any]:
    selected = Path(run_dir).resolve()
    context_run, runtime_root = _amendment08_rehearsal_context_paths()
    path = Path(handoff_path)
    if not (
        selected == context_run
        and path.is_absolute()
        and path == _amendment08_resume_rehearsal_handoff_path()
        and live_journal_before_sha256 == A08_PRELIVE_JOURNAL_SENTINEL_SHA256
        and live_journal_after_sha256 == A08_PRELIVE_JOURNAL_SENTINEL_SHA256
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 resume rehearsal handoff request differs"
        )
    if path.exists():
        return validate_amendment08_resume_rehearsal_handoff(
            path, run_dir=selected,
        )
    manifest = validate_final_scientific_run(selected)
    cache = _amendment08_rehearsal_cache_clone_evidence(
        selected, rehash_source_and_clone=True,
    )
    production_inventory = amendment08_production_live_journal_inventory_bytes()
    if sha256_bytes(production_inventory) != live_journal_after_sha256:
        raise Amendment06IntegrityError(
            "Production Journal changed before rehearsal handoff publication"
        )
    payload = {
        "schema": A08_REHEARSAL_HANDOFF_SCHEMA, "status": "PASS",
        "run_id": selected.name, "run_path": str(selected),
        "runtime_path": str(runtime_root),
        "output_manifest_sha256": manifest["sha256"],
        "output_manifest_rows": 236,
        "run_tree_sha256": amendment08_run_tree_sha256(selected),
        "cache_source_inventory_sha256": cache[
            "cache_source_inventory_sha256"
        ],
        "cache_clone_inventory_sha256": cache[
            "cache_clone_inventory_sha256"
        ],
        "live_journal_before_sha256": live_journal_before_sha256,
        "live_journal_after_sha256": live_journal_after_sha256,
    }
    _require_exact_keys(
        payload, set(A08_REHEARSAL_HANDOFF_KEYS),
        label="Amendment 08 resume rehearsal handoff",
    )
    publish_strict_json_no_clobber(path, payload)
    if not (
        _safe_regular(path, sha256_bytes(strict_full_json_bytes(payload)))
        and strict_full_load_file(path) == payload
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 resume rehearsal handoff publication differs"
        )
    return {
        **payload, "handoff_path": str(path),
        "handoff_sha256": sha256_file(path),
    }


def _record_amendment08_true_rehearsal_postscience(
    run_dir: Path,
) -> dict[str, Any]:
    selected, runtime_root = _amendment08_rehearsal_context_paths()
    run_dir = Path(run_dir).resolve()
    if run_dir != selected:
        raise Amendment06IntegrityError(
            "Amendment 08 true rehearsal post-science target differs"
        )
    final = validate_final_scientific_run(run_dir)
    views = read_amendment08_journal_views(run_dir)
    raw_events = list(views["raw_events"])
    semantic_events = list(views["semantic_events"])
    fit_events = [
        event for event in semantic_events
        if event.get("event") == "SCIENTIFIC_FIT_CALLED"
    ]
    failed_events = [
        event for event in semantic_events
        if event.get("event") == "TRAINING_ATTEMPT_FAILED"
    ]
    completions = [
        event for event in semantic_events
        if event.get("event") == "TRAINING_COMPLETION_COMMITTED"
    ]
    freezes = [
        event for event in semantic_events if event.get("event") == "MODELS_FROZEN"
    ]
    audit_opens = [
        event for event in semantic_events
        if event.get("event") == "EXTERNAL_AUDIT_OPENED"
    ]
    cache = _amendment08_rehearsal_cache_clone_evidence(
        run_dir, rehash_source_and_clone=False,
    )
    frozen = strict_full_load_file(
        run_dir / "provenance/full_models_frozen_manifest.json"
    )
    freeze_index = int(freezes[0]["event_index"]) if len(freezes) == 1 else -1
    audit_open_index = (
        int(audit_opens[0]["event_index"]) if len(audit_opens) == 1 else -1
    )
    if not (
        final["status"] == "PASS" and final["rows"] == 236
        and len(raw_events) > 9
        and raw_events[7].get("event") == "AMENDMENT08_CORRIGENDUM_INSTALLED"
        and raw_events[8].get("event") == "CACHE_LOAD"
        and raw_events[9].get("event") == "TRAINING_ATTEMPT_STARTED"
        and len(fit_events) == 11
        and len(failed_events) == 1
        and failed_events[0].get("attempt_id") == A07_FAILED_ATTEMPT_ID
        and len(completions) == 10
        and [event.get("variant_id") for event in completions]
        == list(TRAINED_VARIANTS)
        and len(freezes) == len(audit_opens) == 1
        and freeze_index < audit_open_index
        and all(int(event["event_index"]) < freeze_index for event in fit_events)
        and sum(
            event.get("event") == "CACHE_GENERATION_INTENT"
            for event in raw_events
        ) == 1
        and sum(
            event.get("event") == "CACHE_GENERATION" for event in raw_events
        ) == 1
        and sum(event.get("event") == "CACHE_LOAD" for event in raw_events) == 1
        and len(list(run_dir.glob("models/*/model.ubj"))) == 10
        and len(list(run_dir.glob(
            "models/*/training_completion_manifest.json"
        ))) == 10
        and frozen.get("scientific_fit_attempts") == 11
        and frozen.get("failed_unpublished_training_attempts") == 1
        and cache["cache_clone_distinct_inodes"] is True
        and cache["cache_inventory_file_count"] == 11
    ):
        raise Amendment06IntegrityError(
            "Amendment 08 true rehearsal post-science contract differs"
        )
    production_journal_sha256 = sha256_bytes(
        amendment08_production_live_journal_inventory_bytes()
    )
    log = (
        "STATUS=PASS\n"
        "CACHE_CLONE_DISTINCT_INODES=PASS_11_OF_11\n"
        "SCIENTIFIC_FIT_ATTEMPTS=11\n"
        "FAILED_UNPUBLISHED_FIT_ATTEMPTS=1\n"
        "COMPLETED_MODELS=10\n"
        "FREEZE_BEFORE_AUDIT_OPEN=PASS\n"
        "NEW_RESAMPLING_REALIZATIONS=0\n"
        f"PRODUCTION_JOURNAL_SENTINEL_SHA256={production_journal_sha256}\n"
    ).encode("utf-8")
    log_path = runtime_root.parent / "logs/amendment08_true_resume_rehearsal.log"
    if log_path.exists():
        if not _safe_regular(log_path, sha256_bytes(log)):
            raise Amendment06IntegrityError(
                "Amendment 08 true rehearsal post-science log drifted"
            )
    else:
        publish_bytes_no_clobber(log_path, log)
    return {
        "status": "PASS", "log_path": str(log_path),
        "log_sha256": sha256_file(log_path), "scientific_fit_attempts": 11,
        "failed_unpublished_fit_attempts": 1, "completed_models": 10,
        "cache_clone_distinct_inode_count": 11,
        "production_journal_sha256": production_journal_sha256,
    }


def _execute_scientific_full(
    *, run_dir: Path, preflight_run: Path, smoke_run: Path,
) -> Path:
    run_dir = Path(run_dir).resolve()
    if (
        run_dir.name == A08_TARGET_RUN_ID
        and _amendment08_finalization_handoff_ready(run_dir)
    ):
        _validate_amendment08_finalization_identity(run_dir)
        finalize_scientific_run(run_dir=run_dir, audit=None)
        return run_dir
    _validate_run_identity(run_dir)
    state = _current_run_state(run_dir)
    if state == "FULL_SCIENTIFIC_COMPLETE" and (
        run_dir / "OUTPUT_MANIFEST_FINAL.tsv"
    ).is_file():
        validate_final_scientific_run(run_dir)
        return run_dir
    initialize_run_evidence(run_dir)
    locked = load_locked_inputs(preflight_run)
    development = load_development_population(locked)
    cache_path = EXPECTED_RUNTIME_ROOT / "full_population" / run_dir.name
    if RUN_PHASES.index(state) >= RUN_PHASES.index("FULL_MODELS_FROZEN"):
        _install_post_freeze_guards()
        if not cache_path.is_dir() or cache_path.is_symlink():
            raise Amendment06IntegrityError("Frozen population cache cannot be rematerialized after model freeze")
    cache = load_or_build_population_cache(
        run_dir=run_dir, locked=locked, development=development,
    )
    ensure_blocked_no_pca(run_dir)
    state = _current_run_state(run_dir)
    if state == "FULL_IN_PROGRESS":
        for variant in TRAINED_VARIANTS:
            train_one_variant(
                run_dir=run_dir, locked=locked, cache=cache,
                development=development, variant=variant,
            )
        freeze_all_models(
            run_dir=run_dir, locked=locked, cache=cache, development=development,
        )
    else:
        validate_models_frozen_manifest(
            run_dir=run_dir, locked=locked, cache=cache, development=development,
        )
        _install_post_freeze_guards()
    audit = open_external_audit_after_freeze(
        run_dir=run_dir, locked=locked, cache=cache, development=development,
    )
    scoring_results = [
        score_one_external_variant(
            run_dir=run_dir, locked=locked, cache=cache,
            development=development, audit=audit, variant=variant,
        )
        for variant in TRAINED_VARIANTS
    ]
    complete_external_audit_phase(run_dir=run_dir, audit=audit)
    build_reports_and_bootstrap(
        run_dir=run_dir, locked=locked, cache=cache,
        audit=audit, scoring_results=scoring_results,
    )
    finalize_scientific_run(run_dir=run_dir, audit=audit)
    return run_dir


def package_dispatch(run_dir: Path, *, command_line: str = "") -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    if run_dir.name != A08_TARGET_RUN_ID:
        _validate_run_identity(run_dir)
        validate_final_scientific_run(run_dir)
    package_module = importlib.import_module("amendment06_packaging")
    return package_module.package_existing_full(
        run_dir, command_line=command_line,
        preflight_run=ACCEPTED_PREFLIGHT, smoke_run=ACCEPTED_SMOKE,
    )


def run_full(
    *, preflight_run: Path, smoke_run: Path, command_line: str = "",
) -> Path:
    with amendment06_lock():
        assert_exact_invocation(preflight_run=preflight_run, smoke_run=smoke_run)
        admission = verify_reference_admission(
            preflight_run=preflight_run, smoke_run=smoke_run,
        )
        locked = load_locked_inputs(preflight_run)
        runs = existing_full_runs()
        if runs:
            raise Amendment06ResumeError(
                "The consumed Amendment 06 Run may continue only through --resume"
            )
        else:
            run_dir = _create_authorized_run(
                command_line=command_line, admission=admission, locked=locked,
            )
        _execute_scientific_full(
            run_dir=run_dir, preflight_run=Path(preflight_run), smoke_run=Path(smoke_run),
        )
        package_dispatch(run_dir, command_line=command_line)
        return run_dir


def resume_full(
    run_dir: Path, *, preflight_run: Path, smoke_run: Path,
    command_line: str = "",
) -> Path:
    with amendment06_lock():
        assert_exact_invocation(preflight_run=preflight_run, smoke_run=smoke_run)
        target = Path(run_dir).resolve()
        rehearsal_context = _AMENDMENT08_REHEARSAL_CONTEXT.get()
        rehearsal_live_before: str | None = None
        rehearsal_handoff: Path | None = None
        if rehearsal_context is not None:
            context_run, _ = _amendment08_rehearsal_context_paths()
            handoff_text = os.environ.get(
                "AMENDMENT08_RESUME_REHEARSAL_HANDOFF", "",
            )
            if not handoff_text:
                raise Amendment06IntegrityError(
                    "Amendment 08 true rehearsal handoff path is required"
                )
            rehearsal_handoff = Path(handoff_text).expanduser()
            if not (
                target == context_run
                and rehearsal_handoff.is_absolute()
                and rehearsal_handoff
                == _amendment08_resume_rehearsal_handoff_path()
            ):
                raise Amendment06IntegrityError(
                    "Amendment 08 true rehearsal resume/handoff routing differs"
                )
            rehearsal_live_before = sha256_bytes(
                amendment08_production_live_journal_inventory_bytes()
            )
        if target.name == A08_TARGET_RUN_ID:
            if _amendment08_finalization_handoff_ready(target):
                _validate_amendment08_finalization_identity(target)
            else:
                validate_amendment08_pre_resume(
                    run_dir=target, preflight_run=preflight_run,
                    smoke_run=smoke_run,
                )
                _validate_run_identity(target)
        else:
            validate_amendment07_pre_resume(
                run_dir=target, preflight_run=preflight_run, smoke_run=smoke_run,
            )
            _validate_run_identity(target)
        _execute_scientific_full(
            run_dir=target, preflight_run=Path(preflight_run), smoke_run=Path(smoke_run),
        )
        if rehearsal_context is not None:
            postscience = _record_amendment08_true_rehearsal_postscience(target)
            rehearsal_live_after = sha256_bytes(
                amendment08_production_live_journal_inventory_bytes()
            )
            if not (
                rehearsal_handoff is not None
                and rehearsal_live_before == rehearsal_live_after
                == A08_PRELIVE_JOURNAL_SENTINEL_SHA256
                and postscience["production_journal_sha256"]
                == rehearsal_live_after
            ):
                raise Amendment06IntegrityError(
                    "Production Journal changed during the Amendment 08 true rehearsal"
                )
            publish_amendment08_resume_rehearsal_handoff(
                run_dir=target, handoff_path=rehearsal_handoff,
                live_journal_before_sha256=rehearsal_live_before,
                live_journal_after_sha256=rehearsal_live_after,
            )
        package_dispatch(target, command_line=command_line)
        return target


__all__ = [
    "AUTHORIZATION", "A08_AUTHORIZATION", "A08_PROMPT_SHA256",
    "A08_TARGET_RUN_ID", "A08_EXECUTION_LEDGER_ADDED_FIELDS",
    "A08_ISOLATED_REHEARSAL_PROJECTION_KEYS",
    "A08_ISOLATED_REHEARSAL_PROJECTION_MODE",
    "A08_ISOLATED_REHEARSAL_PROJECTION_SCHEMA",
    "A08_REHEARSAL_HANDOFF_KEYS", "A08_REHEARSAL_HANDOFF_SCHEMA",
    "A08_PRELIVE_JOURNAL_SENTINEL_SHA256",
    "A08_REHEARSAL_BASE_RUN_FILE_COUNT",
    "A08_REHEARSAL_BASE_RUN_INVENTORY_SHA256",
    "A08_REHEARSAL_BASE_RUN_SIZE_BYTES",
    "A08_LINEAGE_FIELDS", "A08_MODEL_BUNDLE_GLOBALS",
    "A08_OVERLAY_RUN_RELATIVES", "A08_RAW_JOURNAL_CUTOFF_RELATIVE",
    "A08_RAW_JOURNAL_CUTOFF_KEYS", "A08_RAW_JOURNAL_RECORD_KEYS",
    "A08_PACKAGE_LINEAGE_EVENT_KEYS", "A08_PACKAGE_LINEAGE_EVIDENCE_KEYS",
    "A08_PACKAGE_LINEAGE_INTENT_KEYS", "A08_PACKAGE_LINEAGE_RECEIPT_KEYS",
    "A08_PACKAGE_LINEAGE_RECORD_KEYS", "A08_PACKAGE_REHEARSAL_EVIDENCE_KEYS",
    "A08_REQUIRED_ZERO_SCIENCE_API_NAMES",
    "Amendment06Error", "Amendment06IntegrityError",
    "Amendment06ResumeError", "PostFreezeScientificCallError", "StrictJSONError",
    "amendment08_isolated_rehearsal_context",
    "amendment08_isolated_rehearsal_projection_path",
    "amendment08_live_journal_inventory_bytes",
    "amendment08_production_live_journal_inventory_bytes",
    "amendment08_run_tree_sha256",
    "append_amendment07_authorized_edit_event", "append_authorized_edit_event",
    "assert_required_metrics_finite",
    "build_amendment08_raw_journal_cutoff", "build_authorized_edit_event",
    "build_execution_ledger", "build_final_prerun_evidence",
    "expected_scientific_run_files", "finalize_scientific_run",
    "install_amendment07_overlay", "install_amendment08_overlay",
    "install_amendment08_isolated_rehearsal_projection",
    "install_amendment08_source_freeze", "make_test_evidence_record",
    "package_dispatch", "publish_amendment08_resume_rehearsal_handoff",
    "read_amendment08_journal_views",
    "replace_amendment07_final_prerun_evidence", "resume_full",
    "run_amendment07_cuda_producer_consumer_probe",
    "run_amendment08_cuda_producer_consumer_probe", "run_full",
    "strict_full_canonical_json_bytes", "strict_full_json_bytes", "strict_full_load_file",
    "strict_full_loads", "validate_amendment07_cuda_producer_consumer_probe",
    "validate_amendment07_overlay", "validate_amendment07_precorrection",
    "validate_amendment07_pre_resume", "validate_amendment08_overlay",
    "validate_amendment08_cuda_producer_consumer_probe",
    "validate_amendment08_isolated_rehearsal_projection",
    "validate_amendment08_prelive_package_lineage",
    "validate_amendment08_precorrection", "validate_amendment08_pre_resume",
    "validate_amendment08_raw_journal_cutoff",
    "validate_amendment08_resume_rehearsal_handoff",
    "validate_amendment08_source_freeze", "validate_final_scientific_run",
    "validate_strict_json_tree",
    "write_final_prerun_evidence",
]
