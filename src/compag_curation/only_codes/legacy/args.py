# [ONLY-CODES-COMPAT] Vendored from Only_codes/Clean/sam2_pipeline/args.py; PATCH AR-1 only.
# args.py
# -*- coding: utf-8 -*-
import argparse
from .defaults import (
    DEFAULT_IMAGE_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PROTO_JSON,
    DEFAULT_SAM2_CONFIG, DEFAULT_SAM2_CKPT, DEFAULT_MODE,
    DEFAULT_GATE_ORDER, DEFAULT_USE_GATES, DEFAULT_COMBO_GATES
)

def make_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--images", default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--out",    default=DEFAULT_OUTPUT_DIR)

    # Feature-set selector
    ap.add_argument(
        "--feature-mode",
        choices=["balanced","rich","ultra"],
        default="balanced",  # [ONLY-CODES-COMPAT PATCH AR-1] env JASSID_FEATURE_MODE not read
        help="Which feature set to extract in gate_core.compute_features. ULTRA builds on RICH and can add embed PCA features."
    )

    # Threshold proto (optional)
    ap.add_argument("--proto", default=DEFAULT_PROTO_JSON,
                    help="(optional) JSON thresholds; if missing, defaults used.")
    ap.add_argument("--proto-csv",
                    help="(optional) unified (key,value) CSV (embed + feature stats)")

    # Embed proto (simple CSV)
    ap.add_argument("--embed-csv",
                    help="CSV produced by embed_proto_from_coco.py (mu_*, embed_thresh)")

    # Unified feature CSV (from build_unified_proto_csv.py)
    ap.add_argument("--unified-proto-csv",
                    help="CSV produced by build_unified_proto_csv.py (includes feat_names, mu/mad/cov, embed)")

    # Optional PCA for ULTRA
    ap.add_argument("--embed-pca", default="",
                    help="Path to PCA for embed vectors (.npz/.npy/.json with 'components' and optional 'mean'). If provided and --feature-mode ultra, adds embed_pca_* features.")
    ap.add_argument("--embed-pca-preset", type=int, choices=[16,32], default=0,
                    help="If set (16 or 32) and --embed-pca is empty, tries to auto-load embed_pca_<K>.npz from <out>/, ./models/, or ./.")

    # ---------------- SAM2 config + AMG (recall-friendly) ----------------
    ap.add_argument("--mode", default=DEFAULT_MODE, choices=["auto"], help="Masking mode")
    ap.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    ap.add_argument("--sam2-ckpt",   default=DEFAULT_SAM2_CKPT)
    ap.add_argument("--sam2-policy", type=str, choices=["both","auto","prompt"],
                    default="both",  # [ONLY-CODES-COMPAT PATCH AR-1] env SAM2_POLICY not read
                    help="SAM2 mask-generation policy: both (prompt+AMG), auto (AMG only), prompt (YOLO-prompt only).")

    # NOTE: For higher recall, the defaults have been set slightly more aggressively than before
    ap.add_argument("--points-per-side", type=int, default=64,
                    help="Sampling grid density for automatic mask generation (↑ = more proposals).")
    ap.add_argument("--pred-iou", type=float, default=0.80,
                    help="Lower keeps more proposals. (Prev default was 0.88)")
    ap.add_argument("--stability", type=float, default=0.90,
                    help="Lower keeps more proposals. (Prev default was 0.95)")
    ap.add_argument("--stability-offset", type=float, default=0.0,
                    help="Optional stability offset if supported by SAM2 AMG.")
    ap.add_argument("--points-per-batch", type=int, default=1024,
                help="How many points to process per forward pass; 512 to 2048 is usually safe.")
    ap.add_argument("--crop-n-layers", type=int, default=0,
                    help="0 means no multi-crop; greatly reduces memory usage.")
    ap.add_argument("--crop-n-points-downscale-factor", type=int, default=2)
    ap.add_argument("--crop-overlap-ratio", type=float, default=0.34)
    ap.add_argument("--max-num-masks", type=int, default=0,
                    help="0 means unlimited; if you still get OOM, set it to 512 or 1024.")

    # Cropping/nms knobs (only if supported by the installed SAM2)
    #ap.add_argument("--crop-n-layers", type=int, default=1,
                    #help="Number of crop layers (0 disables, 1-2 increases recall on small objects).")
    ap.add_argument("--crop-overlap", type=float, default=0.25,
                    help="Crop overlap ratio (typical 0.25~0.34).")
    ap.add_argument("--crop-nms", type=float, default=0.90,
                    help="NMS threshold for crops (higher → keep more).")
    ap.add_argument("--box-nms", type=float, default=0.70,
                    help="Box NMS threshold within AMG proposals.")
    ap.add_argument("--min-region-area", type=int, default=0,
                    help="Minimum mask area in pixels; 0 to keep even tiny masks.")
    # AMG output
    # (the pipeline uses 'binary_mask' by default; this is only for option consistency)
    ap.add_argument("--amg-output-mode", default="binary_mask",
                    help="binary_mask | rle (used if your SAM2 supports it)")

    # -------- Multi-scale inference (to avoid misses) --------
    ap.add_argument("--ms-scales", default="1.0",
                    help="Comma-separated scales for multi-scale SAM2: e.g., '0.8,1.0,1.25'.")
    ap.add_argument("--ms-merge-iou", type=float, default=0.75,
                    help="IoU threshold to merge duplicates across scales.")
    ap.add_argument("--ms-max-masks", type=int, default=500,
                    help="Cap on number of masks kept after multi-scale merge.")

    # ---------------- decision logic (k-of-n) ----------------
    ap.add_argument("--min-true-gates", type=int, default=4)
    ap.add_argument("--use-gates", default=DEFAULT_USE_GATES,
                    help="Comma list of gates used for k-of-n (default: light,color,shape,embed)")
    ap.add_argument("--gate-order", default=DEFAULT_GATE_ORDER)
    ap.add_argument("--require-quality", action="store_true",
                    help="If set, a proposal must also pass quality gate regardless of k-of-n")

    # color gate options
    ap.add_argument("--color-mode", choices=["ab","b_only","none"], default="ab",
                    help="Color gate: 'ab' uses Δa & Δb; 'b_only' uses only Δb<=thr; 'none' disables color gate")
    ap.add_argument("--delta-b-max-bonly", type=float, default=-8.0,
                    help="If --color-mode b_only: require delta_b <= this (negative)")

    # robust / maha
    ap.add_argument("--skip-maha", action="store_true", help="Skip Mahalanobis gate even if cov is present")
    ap.add_argument("--k-mad", type=float, help="Override k-MAD for robust gate")

    # combos / visualization controls
    ap.add_argument("--combo-gates", default=DEFAULT_COMBO_GATES,
                    help="Comma list of gates to enumerate combos (default 7 gates).")
    ap.add_argument("--save-all-combos", action="store_true",
                    help="If set, save kept/rejected overlays for EVERY combo.")
    ap.add_argument("--max-combo-visuals", type=int, default=9999,
                    help="Max number of combo overlays to save per image (default: unlimited).")

    # exclude largest mask in each image
    ap.add_argument("--no-exclude-largest", dest="exclude_largest", action="store_false",
                    help="Do NOT exclude largest mask per image.")
    ap.set_defaults(exclude_largest=True)

    # === XGBoost & Active Learning ===
    ap.add_argument("--export-features", action="store_true",
                    help="Write per-mask features to <out>/features_pool.csv (or path via --export-features-path).")
    ap.add_argument("--export-features-path", default="",
                    help="Optional CSV path for features export; defaults to <out>/features_pool.csv.")
    #ap.add_argument("--xgb-model", default="",
    #                help="Path to XGBoost model (.json/.ubj Booster or .pkl/.joblib sklearn wrapper).")
    ap.add_argument("--xgb-model", default="",
                    help="Path to classifier model (.json/.ubj XGBoost Booster or .pkl/.joblib sklearn/imblearn/cuML).")

    # --- Aliases for generic classifier (backward compatibility) ---
    ap.add_argument("--clf-model", dest="xgb_model", default=argparse.SUPPRESS,
                    help="Alias for --xgb-model (generic classifier).")
    ap.add_argument("--xgb-features", default="auto",
                    help="Comma-separated feature names for XGBoost, or 'auto' to use curated set based on --feature-mode.")
    #ap.add_argument("--use-xgb-inference", action="store_true",
    #                help="(XGB ONLY) Use XGBoost as the final decision-maker (no gates/fusion).")
    ap.add_argument("--use-xgb-inference", action="store_true",
                    help="Use the classifier (XGB/RF) as the final decision-maker (no gates/fusion).")

    ap.add_argument("--use-clf-inference", dest="use_xgb_inference", action="store_true",
                    help="Alias for --use-xgb-inference (generic).")

    ap.add_argument("--xgb-policy", default="replace", type=str,
                choices=["and","or","replace"],
                help="How to combine XGB decision with main keep flag (late-fusion default).")
    ap.add_argument("--xgb-threshold", default=0.50, type=float, help="XGB decision threshold")

    # Force default to XGB-only mode
    ap.set_defaults(use_xgb_inference=True)

    # --- YOLO extra knobs ---
    ap.add_argument('--yolo-imgsz', type=int, default=640)
    ap.add_argument('--yolo-device', type=str, default=None)
    ap.add_argument('--yolo-max-det', type=int, default=300)
    ap.add_argument('--yolo-pad-frac', type=float, default=0.02)

    ap.add_argument('--yolo-prompt-logit-boost', type=float, default=0.0)
    ap.add_argument('--yolo-prompt-thr-bonus', type=float, default=0.0)

    ap.add_argument("--yolo-weights", default="", help="Ultralytics .pt (best.pt) for hinting only")
    ap.add_argument("--yolo-conf", type=float, default=0.20, help="YOLO confidence")
    ap.add_argument("--yolo-iou",  type=float, default=0.60, help="YOLO NMS IoU")
    ap.add_argument("--debug-yolo-dump", action="store_true",
                    help="If no YOLO boxes found, save <img>_yolo_input.jpg for debugging")

    ap.add_argument("--thr_yolo_raw", dest="yolo_conf", type=float, default=argparse.SUPPRESS,
                    help="Alias for --yolo-conf (compat).")
    ap.add_argument("--thr_yolo_iou", dest="yolo_iou", type=float, default=argparse.SUPPRESS,
                    help="Alias for --yolo-iou (compat).")


    ap.add_argument("--yolo-hint", choices=["off","lower_thr"], default="off",
                    help="Only bias XGB decision; no proposals are added/removed.")
    ap.add_argument("--yolo-hint-alpha",   type=float, default=0.35,
                    help="How much to lower threshold per (IoU*conf).")
    ap.add_argument("--yolo-hint-min-thr", type=float, default=0.25,
                    help="Floor for effective threshold after hint.")
    ap.add_argument("--yolo-hint-iou",     type=float, default=0.30,
                    help="Only consider YOLO boxes with IoU >= this against mask bbox.")
    
    # --- New: Detection and Hybrid Policies ---
    ap.add_argument("--det-policy", choices=["and", "or", "xgb", "yolo", "hybrid"],
                   default="and", help="Decision policy between XGB and YOLO")
    ap.add_argument("--det-missing", choices=["reject", "ignore"],
                   default="reject", help="When one of the sources is missing: reject or ignore")
    ap.add_argument("--det-thr", type=float, default=0.5,
                   help="Global threshold for the hybrid mode (e.g., 0.5)")
    ap.add_argument("--hybrid-yolo-bias", type=float, default=0.6,
                   help="When XGB>=thr but YOLO<thr, apply minimal YOLO-side weight (e.g., 0.6)")

    # --- Smart Hybrid rules (optional but ON by default) ---
    # These implement rule-based regions such as:
    #   - if (xgb <= 0.20) and (yolo <= 0.70) => force Non-CJ
    #   - if (yolo >= 0.85) and (xgb >= 0.30) => force CJ
    ap.add_argument("--smart-hybrid", dest="smart_hybrid", action="store_true",
                    help="Enable rule-based hybrid tweaks (recommended).")
    ap.add_argument("--no-smart-hybrid", dest="smart_hybrid", action="store_false",
                    help="Disable rule-based hybrid tweaks (use pure weighted fusion).")
    ap.set_defaults(smart_hybrid=True)

    ap.add_argument("--smart-noncj-xgb-max", type=float, default=0.20,
                    help="Smart hybrid: if XGB <= this AND YOLO <= smart-noncj-yolo-max => force Non-CJ.")
    ap.add_argument("--smart-noncj-yolo-max", type=float, default=0.70,
                    help="Smart hybrid: low YOLO cutoff for the Non-CJ override region.")
    ap.add_argument("--smart-cj-yolo-min", type=float, default=0.85,
                    help="Smart hybrid: if YOLO >= this AND XGB >= smart-cj-xgb-min => force CJ.")
    ap.add_argument("--smart-cj-xgb-min", type=float, default=0.30,
                    help="Smart hybrid: moderate XGB cutoff for the CJ override region.")


    # (Removed: gate-fuse / gate-lr / stack-lr arguments)

    # Active Learning
    ap.add_argument("--al-topk", type=int, default=50,
                    help="Active learning: select up to top-K samples for review.")
    ap.add_argument("--al-metric", choices=["margin","entropy"], default="margin",
                    help="Uncertainty metric for active learning (used for XGB-only uncertainty; hybrid uses |p-thr| band).")
    ap.add_argument("--al-out", default="",
                    help="CSV path for AL candidates (default: <out>/al_candidates.csv).")

    # Near-threshold band (matches GUI notion of uncertainty)
    ap.add_argument("--al-margin", type=float, default=0.20,
                    help="Active learning: consider near-threshold if |p - thr_eff| <= margin.")

    # Selection policy
    ap.add_argument("--al-only-uncertain", dest="al_only_uncertain", action="store_true",
                    help="If set (default), shortlist is limited to (near-threshold OR severe-disagreement) samples.")
    ap.add_argument("--al-all", dest="al_only_uncertain", action="store_false",
                    help="Rank ALL samples by combined AL score (uncertainty + disagreement), then take top-K.")
    ap.set_defaults(al_only_uncertain=True)

    ap.add_argument("--al-fill-to-topk", action="store_true",
                    help="If al-only-uncertain is enabled, fill remaining slots up to top-K with next best scores.")

    # Disagreement-driven AL (recommended for hybrid)
    ap.add_argument("--al-use-disagreement", dest="al_use_disagreement", action="store_true",
                    help="Include severe XGB-vs-YOLO disagreements in the AL shortlist (default ON).")
    ap.add_argument("--al-no-disagreement", dest="al_use_disagreement", action="store_false",
                    help="Disable disagreement-driven AL.")
    ap.set_defaults(al_use_disagreement=True)

    ap.add_argument("--al-disagree-abs", type=float, default=0.55,
                    help="Disagreement AL: require |xgb - yolo| >= this to count as severe (0..1).")
    ap.add_argument("--al-disagree-xgb-hi", type=float, default=0.80,
                    help="Disagreement AL: if XGB>=hi and YOLO<=yolo-lo => severe disagreement.")
    ap.add_argument("--al-disagree-xgb-lo", type=float, default=0.20,
                    help="Disagreement AL: if XGB<=lo and YOLO>=yolo-hi => severe disagreement.")
    ap.add_argument("--al-disagree-yolo-hi", type=float, default=0.85,
                    help="Disagreement AL: high YOLO cutoff for the severe disagreement rule.")
    ap.add_argument("--al-disagree-yolo-lo", type=float, default=0.15,
                    help="Disagreement AL: low YOLO cutoff for the severe disagreement rule.")

    ap.add_argument("--al-unc-weight", type=float, default=1.0,
                    help="Weight for near-threshold uncertainty in the AL ranking score.")
    ap.add_argument("--al-disagree-weight", type=float, default=1.0,
                    help="Weight for disagreement strength in the AL ranking score.")

    # === PCA Builder (subcommand) ===
    ap.add_argument("--make-embed-pca", default="",
                    help="If set, builds PCA from provided embedding data and saves to this .npz path, then exits.")
    ap.add_argument("--pca-dims", type=int, default=16, help="Number of PCA components to keep (e.g., 16 or 32).")
    ap.add_argument("--pca-from-npy", default="", help="NPY file with shape (N,D) or a directory of .npy files.")
    ap.add_argument("--pca-from-npz", default="", help="NPZ with key 'Z' or 'embeddings' (N,D) or a directory of .npz files.")
    ap.add_argument("--pca-from-csv", default="", help="CSV with columns mu_0.. or 'z' JSON arrays (stacks all rows).")
    ap.add_argument("--pca-stack-dirs", default="", help="Comma-separated dirs; stacks all .npy/.npz arrays inside.")

    return ap
