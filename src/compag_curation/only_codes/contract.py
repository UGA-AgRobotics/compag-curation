"""Frozen reference contract of the ``only-codes-compat-v1`` workflow.

Every value below was read from the supplied Only_codes source and the
effective runtime arguments of its runner (``xgb_run_sam2_profile.sh``
mode ``xgb_recall``) and notebook cells -- not from the manuscript, the SAM3
pipelines or the canonical application defaults.  The contract is persisted
in every imported project and every model bundle; reopening a project with a
different contract hash is refused.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from . import CONTRACT_SCHEMA, PROFILE_ID

# SHA-256 of the audited Only_codes sources (Only_codes.zip d95107eb...).
SOURCE_HASHES: dict[str, str] = {
    "Only_codes.zip": "d95107eb81d3933641672733a5222ae89f23c4d52036b893ed78c46df18fb94e",
    "Clean/COCO_2_features[notebook]": "106981513afc5ab0bd6040761a3ce304247fd61a12d948440f356b523d8e2865",
    "Clean/gate_core.py": "0893064f9b700c5d",  # prefix; full hash recorded in PARITY_REPORT.json
    "Clean/sam2_and_filter.py": "8b07c4cae8881c0b",
    "Clean/xgb_run_sam2_profile.sh": "e9c0711f23496692",
    "Clean/sam2_pipeline/pipeline.py": "9de9e1f5f8a45ec3",
    "Clean/sam2_pipeline/args.py": "399d5bfd9a96a5f6",
    "Clean/sam2_pipeline/xgb_utils.py": "6c9f6032941a497d",
    "Clean/sam2_pipeline/pca_utils.py": "022c6396b46b0518",
    "Clean/sam2_pipeline/light_gate.py": "36cba259e58e8599",
    "Clean/sam2_pipeline/defaults.py": "baac2d0d91808ee3",
    "Clean/full_image_review_gui.py": "77bce1efd0088c70",
}

# Notebook cells, identified by title *and* position (1-based) in the audited
# notebook.  The source SHA-256 prefixes are checked by the parity harness.
NOTEBOOK_CELLS: list[dict[str, Any]] = [
    {"pos": 1, "title": "Cell 000 - ROUND BOOTSTRAP + CONTROL PANEL", "port": "only_codes.rounds"},
    {"pos": 2, "title": "CELL 00 - PRE-CELL MERGE", "port": "only_codes.merge"},
    {"pos": 3, "title": "CELL 0 — Train-only splits (FREEZE+APPEND)", "port": "only_codes.splits"},
    {"pos": 4, "title": "JUPYTER CELL 1 (ENV-driven) — TRAIN-only pack", "port": "only_codes.pack"},
    {"pos": 5, "title": "JUPYTER CELL 2 (ENV-driven): COCO → Features", "port": "only_codes.features"},
    {"pos": 6, "title": "Cell3A (setup & data prep + OPTIONAL AUGS on TRAIN)", "port": "only_codes.training"},
    {"pos": 8, "title": "Cell3B (group-safe weighted HP search + ES refit + full-train refit)", "port": "only_codes.training"},
    {"pos": 9, "title": "Cell 1: Training completed (NO notification)", "port": "not ported (timestamp bookkeeping only)"},
    {"pos": 11, "title": "%%bash runner xgb_run_sam2_profile.sh", "port": "only_codes.runner + only_codes.legacy.pipeline"},
    {"pos": 12, "title": "patch_review_gui_to_next.py", "port": "only_codes.rounds.next_image_paths"},
    {"pos": 13, "title": "Cell 2: Last cell executed (ntfy notification)", "port": "NOT PORTED: external notification disabled"},
]

# Effective inference arguments of the reference runner (mode xgb_recall),
# after sam2_and_filter.py pre-processing.  ``None`` means "resolved from the
# imported project".
REFERENCE_INFERENCE: dict[str, Any] = {
    "runner_mode": "xgb_recall",
    "sam2_config_hydra": "configs/sam2.1/sam2.1_hiera_l",
    "sam2_checkpoint_sha256": "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
    "apply_postprocessing": False,
    "sam2_policy": "both",
    "feature_mode": "ultra",
    "core_feature_mode_for_compute_features": "rich",
    "points_per_side": 64,
    "points_per_batch": 512,
    "pred_iou": 0.80,
    "stability": 0.88,
    "crop_n_layers": 0,
    "crop_n_points_downscale_factor": 2,
    "crop_overlap_ratio": 0.4,
    "max_num_masks": 0,
    "exclude_largest": False,
    "min_true_gates": 3,
    "ms_scales": "1.0",
    "ms_merge_iou": 0.75,
    "ms_max_masks": 500,
    "export_features": True,
    "save_all_outputs": True,
    "use_xgb_inference": True,
    "xgb_features": "@<out>/_xgb_features_from_model.txt (ordered schema of the selected model)",
    "xgb_policy": "replace",
    "xgb_threshold": 0.50,
    "det_policy": "hybrid",
    "det_missing": "reject",
    "det_thr": 0.5,
    "hybrid_yolo_bias": 0.8,
    "smart_hybrid": True,
    "smart_noncj_xgb_max": 0.20,
    "smart_noncj_yolo_max": 0.70,
    "smart_cj_yolo_min": 0.85,
    "smart_cj_xgb_min": 0.30,
    "yolo_conf": 0.20,
    "yolo_iou": 0.60,
    "yolo_hint": "off",
    "yolo_imgsz": 512,
    "yolo_device": "0",
    "yolo_max_det": 300,
    "yolo_pad_frac": 0.02,
    "debug_yolo_dump": True,
    "al_topk": 50,
    "al_metric": "margin",
    "al_margin": 0.20,
    "al_only_uncertain": True,
    "al_fill_to_topk": False,
    "al_use_disagreement": True,
    "al_disagree_abs": 0.55,
    "al_disagree_xgb_hi": 0.80,
    "al_disagree_xgb_lo": 0.20,
    "al_disagree_yolo_hi": 0.85,
    "al_disagree_yolo_lo": 0.15,
    "al_unc_weight": 1.0,
    "al_disagree_weight": 1.0,
    "color_mode": "ab",
    "gate_order": "quality,light,color,shape,embed,robust,maha,border",
    "combo_gates": "quality,light,color,shape,embed,robust,maha",
    "embed_backbone": "resnet50",
    "embed_backbone_sha256": "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
    "embed_pad_frac": 0.10,
    "device": "cuda",
}

# Review GUI constants (full_image_review_gui.py).  The reviewer computes its
# own display decision with these values; they differ from the pipeline's
# hybrid decision and are preserved separately.
REFERENCE_REVIEW_GUI: dict[str, Any] = {
    "use_xgb": True,
    "use_yolo": True,
    "det_policy": "xgb",
    "det_missing": "ignore",
    "thr_xgb": 0.40,
    "thr_yolo": 0.20,
    "thr_iou": 0.60,
    "det_thr": 0.50,
    "hybrid_yolo_bias": 0.80,
    "al_margin": 0.20,
    "weight_map": {"accept": 1.0, "flip": 1.0, "sus_accept": 0.4, "sus_flip": 0.4,
                   "skip": 0.0, "auto_accept": 0.0},
    "timestamp_format": "YYYYMMDDHHMMSS (America/New_York)",
    "mosaic_format": "jpg",
    "jpg_quality": 98,
    "missing_tile_bgr": [32, 32, 32],
}

REFERENCE_MERGE: dict[str, Any] = {
    "tile_w": 512, "tile_h": 512,
    "cj_category_id": 1, "noncj_category_id": 2,
    "latest_rule": "later-or-equal timestamp replaces (ts >= prev); file order breaks ties",
    "update_key": "(image_id, meta.review_id)",
    "geometry_fallback": "detections poly JSON; else rectangle from bbox_x/y/w/h | bbox | minx..maxy",
}

REFERENCE_SPLITS: dict[str, Any] = {"random_state": 42, "freeze_old_splits": True,
                                    "policy": "append new original ids to TRAIN only"}

REFERENCE_PACK: dict[str, Any] = {
    "embedding_scales": [0.85, 1.00, 1.15],
    "order": "crop masked patch first, then resize the patch (INTER_LINEAR, int(w*s), min 1)",
    "pad_frac": 0.10,
    "pca_dims": 32,
    "thresh_method": "fixed",
    "fixed_thresh": 0.70,
    "percentile": 5.0,
    "mu_from": "POS (category name 'cj') embeddings, L2-normalized mean",
    "pca_from": "ALL embeddings, numpy SVD of float32 centered matrix",
    "run_mode": "all",
}

REFERENCE_FEATURES: dict[str, Any] = {
    "feature_mode": "ultra",
    "scales": [0.67, 0.80, 1.00, 1.25],
    "order": "resize image (INTER_LINEAR) and mask (INTER_NEAREST) first, then features and crop",
    "pad_frac": 0.10,
    "embed_sim": "clip((dot(z,mu)+1)/2, 0, 1)",
    "grid": "grid_r_norm=cy/H_scaled, grid_c_norm=cx/W_scaled, 3x3 clipped bins",
    "pred_iou": 1.0, "stability": 1.0,
    "resume_key": "(file_name, ann_id, scale '%.4f') with '*' wildcard for blank scale",
    "review_tag_weights": {"accept": 1.0, "flip": 1.0, "sus_accept": 0.4, "sus_flip": 0.4, "skip": 0.0},
}

REFERENCE_TRAINING: dict[str, Any] = {
    "test_size": 0.20, "random_state": 42, "n_splits_cv": 5, "pos_label": 1,
    "border_px": 4, "tile_size": 512, "tiny_min_wh": 32, "tiny_max_area": 1500,
    "border_weight": 0.5, "uncert_margin": 0.05, "uncert_mult": 0.7,
    "drop_tiny_in_train": False, "drop_border_from_test": True,
    "early_stop_rounds_3A": 50, "early_stop_rounds_effective": 30,
    "early_stop_val_frac": 0.30, "n_estimators_big": 2000,
    "use_review_weighting": True, "weight_correct": 0.7, "weight_wrong": 1.0,
    "auto_detect_prev_round": True, "use_review_tag_weights": True, "drop_skip_rows": True,
    "tiny_neg_weight": 0.6,
    "weight_map": {"accept": 1.0, "flip": 1.0, "sus_accept": 0.4, "sus_flip": 0.4, "skip": 0.0},
    "aug": {"apply_in_3A": True, "exclude_discrete_max_nunique": 10,
            "mixup": {"enabled": True, "alpha": 0.20, "mult": 0.50},
            "dropout": {"enabled": True, "p": 0.05, "mult": 1.00, "strategy": "median_by_class"},
            "jitter": {"enabled": True, "mult": 0.50, "sigma": 0.01, "per_feature": True,
                       "clip_q": [0.001, 0.999]},
            "shuffle_rows": True},
    "split": "pick_groups_tile_balanced(test_frac=.2, RandomState(42), tol=.01, max_tries=3000)",
    "force_hparam_search": False, "search_every_n_rounds": 500, "hparam_n_iter": 30,
    "use_weighted_search": True, "n_est_search": 300,
    "xgb_base": {"objective": "binary:logistic", "eval_metric": "aucpr", "n_estimators": 300,
                 "learning_rate": 0.05, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.9,
                 "min_child_weight": 1.0, "reg_lambda": 1.0, "scale_pos_weight": 1.0,
                 "random_state": 42, "verbosity": 0, "tree_method": "hist", "max_bin": 256},
    "cuda_detection": "import cupy succeeds -> device=cuda, grid_n_jobs=1, n_jobs=min(8,cpu)",
    "threshold_rule": "precision_recall_curve F1 argmax, threshold index max(bi-1,0)",
}

# Matched execution environment observed for the reference runs (the
# interpreter hard-coded in the original runner).  Old notebook metadata
# (Python 3.13.5) and the archived runner log (ultralytics 8.3.221, no
# pycocotools) refer to other environments; no single historical lock is
# invented.  ``pycocotools`` controls whether detections.csv carries RLE.
REFERENCE_ENVIRONMENT: dict[str, Any] = {
    "python": "3.12.7",
    "numpy": "2.0.2",
    "scipy": "1.16.2",
    "sklearn": "1.7.2",
    "imblearn": "0.14.0",
    "xgboost": "2.1.1",
    "cv2": "4.12.0",
    "torch": "2.9.0+cu128",
    "torchvision": "0.24.0+cu128",
    "ultralytics": "8.4.41",
    "pandas": "2.2.3",
    "joblib": "1.5.2",
    "PIL": "11.3.0",
    "cupy": "13.6.0",
    "pycocotools": "2.0.11",
    "torch_cuda": "12.8",
    "cudnn": 91002,
    "sam2_source_tree_sha256": "b2deab23bea9c1fb4ab578784bcb110cd7787c08653fdfc57df0f6c9ac566c1d",
    "sam2_source_files": 38,
    "gpu": "NVIDIA GeForce RTX 4090 (driver 610.60)",
}


# Legacy behaviours preserved on purpose (identified, not corrected).
LEGACY_BEHAVIORS: list[dict[str, str]] = [
    {"id": "LB-01", "what": "legacy_gate_input_zero_fill",
     "detail": "inference g_* predictor columns are 0.0 because gate booleans are never inserted into f_all; boolean gate diagnostics are written separately to detections.csv"},
    {"id": "LB-02", "what": "embed_sim_train_vs_infer",
     "detail": "training embed_sim=clip((z.mu+1)/2,0,1); inference embed_sim=raw z.mu"},
    {"id": "LB-03", "what": "grid_train_vs_infer",
     "detail": "training grid uses cy/H,cx/W with 3x3 clipped bins; inference uses detected dashed-grid lines and index/max(1,len(lines)-1)"},
    {"id": "LB-04", "what": "pred_iou_stability_export", "detail": "COCO feature export writes pred_iou=stability=1.0"},
    {"id": "LB-05", "what": "hu_topology_placeholders", "detail": "hu1/hu2/hu4=0, components_count=1, holes_ratio=0 reserved constants"},
    {"id": "LB-06", "what": "lbp_uint8_shift_overflow", "detail": "_lbp_uniform accumulates (bit<<n) in uint8; the n=8 neighbour overflows to 0"},
    {"id": "LB-07", "what": "scale_lists", "detail": "pack scales [.85,1,1.15] crop-then-resize; export scales [.67,.8,1,1.25] resize-then-crop; AMG [1.0]"},
    {"id": "LB-08", "what": "feature_cache_without_transform_hash", "detail": "features_train.csv resume keys are (file_name, ann_id, scale) with '*' wildcard; no transform identity"},
    {"id": "LB-09", "what": "augmentation_before_inner_split", "detail": "Mixup/Dropout/Jitter run on TRAIN before CV/ES splitting; Mixup parents can span cards; feature selection sees the full table"},
    {"id": "LB-10", "what": "early_stop_override", "detail": "Cell3A sets EARLY_STOP_ROUNDS=50 but Cell3B overwrites it with 30 (effective 30)"},
    {"id": "LB-11", "what": "prev_best_reuse", "detail": "hyper-parameter search runs only when round%500==0; otherwise the newest earlier best_params_used.json is reused"},
    {"id": "LB-12", "what": "threshold_index_shift", "detail": "internal-test threshold uses thr[max(argmax_f1-1,0)] (one-step shift); it is NOT applied at inference (runtime XGB threshold 0.50)"},
    {"id": "LB-13", "what": "proto_repadding", "detail": "sam2_and_filter re-pads an already padded prototype through pandas read/write (_padded_padded.csv) before parsing"},
    {"id": "LB-14", "what": "gui_decision_policy", "detail": "review GUI decides with policy xgb/ignore thr 0.40, independent of the pipeline's hybrid/reject decision"},
    {"id": "LB-15", "what": "gui_label_sort", "detail": "effective GUI labels: pandas sort_values('timestamp') (non-stable quicksort) then last row wins"},
    {"id": "LB-16", "what": "merge_label_mapping", "detail": "merge maps human_label==1 -> COCO category 1 (CJ), anything else (incl. skip/NaN) -> 2 (non-CJ); rows without matching detection are ignored"},
    {"id": "LB-17", "what": "border_test_drop", "detail": "border rows are dropped from the internal TEST report only"},
    {"id": "LB-18", "what": "al_output_format", "detail": "current source writes 12-column al_candidates.csv (with disagreement); archived r137 output used an older 4-column writer"},
]

# Environment variables read by OLD code; the compatibility workflow never
# lets them change the imported preset.  Presence is recorded as IGNORED.
IGNORED_ENVIRONMENT: list[str] = [
    "CJ_SAVE_ALL", "CJ_SAM2_POLICY", "CJ_MODE", "CJ_IMG", "CJ_OUT_ROOT", "CJ_XGB_THRESHOLD",
    "CJ_MODEL_ROOT", "CJ_MODEL_NAME", "CJ_MODEL_DIR_TAG", "CJ_FORCE_CPU", "CJ_BACKUP_DIR",
    "CJ_DO_BACKUP", "CJ_PAD_FRAC", "CJ_PCA_DIMS", "CJ_THRESH_METHOD", "CJ_FIXED_THRESH",
    "CJ_THRESH_PCTL", "CJ_CELL1_RUN_MODE", "CJ_SAMPLE_LIMIT", "CJ_FEATURE_MODE", "CJ_FEAT_SPLIT",
    "CJ_ALLOWED_IMG_IDS_JSON", "CJ_POS_NAME", "CJ_PREV_TILE_PREDS_PATH", "CJ_PREV_THRESH_JSON_PATH",
    "DET_POLICY", "DET_MISSING", "DET_THR", "HYBRID_YOLO_BIAS", "YOLO_CONF", "YOLO_IOU",
    "FORCE_XGB_P", "JASSID_FEATURE_MODE", "JASSID_EMBED_BACKBONE", "SAM2_POLICY",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS",
]


def contract() -> dict[str, Any]:
    return deepcopy({
        "schema": CONTRACT_SCHEMA,
        "profile": PROFILE_ID,
        "authority": "Only_codes/Clean source + effective runner arguments (not manuscript, not canonical defaults)",
        "source_hashes": SOURCE_HASHES,
        "notebook_cells": NOTEBOOK_CELLS,
        "inference": REFERENCE_INFERENCE,
        "review_gui": REFERENCE_REVIEW_GUI,
        "merge": REFERENCE_MERGE,
        "splits": REFERENCE_SPLITS,
        "pack": REFERENCE_PACK,
        "features": REFERENCE_FEATURES,
        "training": REFERENCE_TRAINING,
        "reference_environment": REFERENCE_ENVIRONMENT,
        "legacy_behaviors": LEGACY_BEHAVIORS,
        "ignored_environment": IGNORED_ENVIRONMENT,
    })


def contract_sha256() -> str:
    payload = json.dumps(contract(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def environment_snapshot(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Record legacy override variables that are present but ignored."""

    import os

    env = os.environ if environ is None else environ
    present = {k: env[k] for k in IGNORED_ENVIRONMENT if k in env}
    extra = {k: v for k, v in env.items() if k.startswith(("CJ_", "JASSID_", "NTFY_")) and k not in present}
    return {"ignored_present": present, "ignored_other_legacy": extra,
            "policy": "IGNORED: the imported preset is never changed by environment variables"}
