# [ONLY-CODES-COMPAT] Vendored verbatim from Only_codes/Clean/sam2_pipeline/light_gate.py.
# -*- coding: utf-8 -*-
from typing import Tuple
import numpy as np
import cv2

def ring_medians_scalar(channel: np.ndarray, mask01: np.ndarray, ring_px: int = 5) -> Tuple[float,float,float]:
    """Median inside mask vs. a thin dilated ring; returns (med_in, med_ring, med_in - med_ring)."""
    rp = int(ring_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rp+1, 2*rp+1))
    dil = cv2.dilate(mask01, kernel)
    ring = (dil.astype(bool) & (~mask01.astype(bool))).astype(np.uint8)
    inside = channel[mask01 > 0]
    outside = channel[ring > 0]
    if inside.size == 0:
        return 0.0, 0.0, 0.0
    med_in = float(np.median(inside))
    if outside.size == 0:
        outside = channel[mask01 == 0]
    med_ring = float(np.median(outside)) if outside.size else 0.0
    return med_in, med_ring, float(med_in - med_ring)

def evaluate_light_gate(lab_img: np.ndarray, mask_u8: np.ndarray, params: dict) -> Tuple[bool, str]:
    """Replicates the 'light' gate logic and returns (pass, fail_reason)."""
    L_ch = lab_img[:, :, 0].astype(np.float32)
    idx_mask = (mask_u8 > 0)
    if not np.any(idx_mask):
        return False, "empty"

    valsL = L_ch[idx_mask]
    low_frac = float(np.mean(valsL <= params.get("light_low_T", 5.0)))
    high_frac = float(np.mean(valsL >= params.get("light_high_T", 250.0)))
    stdL = float(valsL.std())
    med_in_L, med_ring_L, dL_med = ring_medians_scalar(L_ch, (mask_u8 > 0).astype(np.uint8), ring_px=int(params.get("light_ring_px", 5)))

    ok = (
        (low_frac  <= params.get("light_low_clip_max", 0.10)) and
        (high_frac <= params.get("light_high_clip_max", 0.08)) and
        (stdL      <= params.get("light_std_max", 40.0)) and
        (abs(dL_med) <= params.get("light_bg_deltaL_max", 25.0))
    )
    if ok:
        return True, ""
    reasons = []
    if low_frac  > params.get("light_low_clip_max", 0.10): reasons.append("low_clip")
    if high_frac > params.get("light_high_clip_max", 0.08): reasons.append("high_clip")
    if stdL      > params.get("light_std_max", 40.0): reasons.append("stdL")
    if abs(dL_med) > params.get("light_bg_deltaL_max", 25.0): reasons.append("deltaL")
    return False, "+".join(reasons)
