# [ONLY-CODES-COMPAT] Vendored from Only_codes/Clean/sam2_pipeline/defaults.py; PATCH DF-1 only.
# -*- coding: utf-8 -*-

from pathlib import Path

# ======= DEFAULTS =======
DEFAULT_IMAGE_DIR = None  # [ONLY-CODES-COMPAT PATCH DF-1] machine-specific Windows default removed; runner always passes an explicit value
DEFAULT_OUTPUT_DIR = None  # [ONLY-CODES-COMPAT PATCH DF-1] machine-specific Windows default removed; runner always passes an explicit value
DEFAULT_PROTO_JSON = None  # [ONLY-CODES-COMPAT PATCH DF-1] machine-specific Windows default removed; runner always passes an explicit value
DEFAULT_SAM2_CONFIG = None  # [ONLY-CODES-COMPAT PATCH DF-1] machine-specific Windows default removed; runner always passes an explicit value
DEFAULT_SAM2_CKPT = None  # [ONLY-CODES-COMPAT PATCH DF-1] machine-specific Windows default removed; runner always passes an explicit value
DEFAULT_MODE        = "auto"
DEFAULT_GATE_ORDER  = "quality,light,color,shape,embed,robust,maha,border"
DEFAULT_USE_GATES   = "light,color,shape,embed"  # used in k-of-n
DEFAULT_COMBO_GATES = "quality,light,color,shape,embed,robust,maha"

# visual params
PAD_FRAC = 0.10

DEFAULT_MS_SCALES = "1.0" #inference


# --------------------------------------
# Curated default feature lists for XGBoost
# --------------------------------------
XGB_FEATS_BALANCED_DEFAULT = [
    # color / deltas
    "delta_b","delta_a",
    # geometry & shape (scale-robust)
    "area_norm","elongation","eccentricity","solidity","aspect_ratio","circularity","extent",
    "bbox_diag_frac",        # ★
    "perim_over_sqrt_area",  # ★
    "hu1","hu2",
    # texture
    "LBP_u5","GLCM_contrast","GLCM_homogeneity","grad_mean","grad_p90",
    # color stats
    "mean_L","std_L","std_a","std_b",
    # context
    "grid_r_norm","grid_c_norm","scale_diag",  # ★
    # quality / embed
    "pred_iou","stability","embed_sim",
]

XGB_FEATS_RICH_DEFAULT = XGB_FEATS_BALANCED_DEFAULT + [
    # robust medians on ring
    "delta_b_med","delta_a_med","median_b_in","median_b_ring","median_a_in","median_a_ring",
    # compact histograms
    "histb_q1","histb_q2","histb_q3","histb_q4",
    "histab_q11","histab_q12","histab_q21","histab_q22",
]

def all_gate_names():
    return ["quality","light","color","shape","embed","robust","maha","border"]
