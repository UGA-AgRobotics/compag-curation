"""Reference runner argument construction and ``sam2_and_filter.py`` pre-steps.

``build_runner_argv`` reproduces the argument vector that the original
``xgb_run_sam2_profile.sh`` passes to ``sam2_and_filter.py`` (COMMON_ARGS, the
YOLO block, the mode-specific block and DECISION_FLAGS), with the numeric
strings exactly as the runner formats them.  The vector is then parsed by the
vendored original argparser, so every default not set by the runner is the
original default.  Environment variables are never consulted.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from .legacy.args import make_argparser


def build_runner_argv(
    *,
    images_dir: str,
    out_dir: str,
    sam2_config: str,
    sam2_ckpt: str,
    padded_proto: str,
    pca_npz: str,
    yolo_weights: str | None,
    xgb_model: str,
    mode: str = "xgb_recall",
    sam2_policy: str = "both",
    xgb_threshold: str = "0.50",
    det_policy: str = "hybrid",
    det_missing: str = "reject",
    det_thr: str = "0.5",
    hybrid_yolo_bias: str = "0.8",
    yolo_conf: str = "0.20",
    yolo_iou: str = "0.60",
    save_all: bool = True,
) -> list[str]:
    """Exact replica of the runner's argv (after ``$PYTHON_BIN sam2_and_filter.py``)."""

    common = [
        "--images", images_dir,
        "--out", out_dir,
        "--sam2-config", sam2_config,
        "--sam2-ckpt", sam2_ckpt,
        "--sam2-policy", sam2_policy,
        "--feature-mode", "ultra",
        "--embed-csv", padded_proto,
        "--embed-pca", pca_npz,
    ]
    if yolo_weights and Path(yolo_weights).is_file():
        common += [
            "--yolo-weights", yolo_weights,
            "--yolo-conf", yolo_conf,
            "--yolo-iou", yolo_iou,
            "--yolo-hint", "off",
            "--yolo-imgsz", "512",
            "--yolo-device", "0",
            "--yolo-max-det", "300",
            "--debug-yolo-dump",
        ]
    else:
        common += ["--yolo-hint", "off"]
    decision = [
        "--use-xgb-inference",
        "--xgb-model", xgb_model,
        "--xgb-features", "auto",
        "--xgb-policy", "replace",
        "--xgb-threshold", xgb_threshold,
        "--det-policy", det_policy,
        "--det-missing", det_missing,
        "--det-thr", det_thr,
        "--hybrid-yolo-bias", hybrid_yolo_bias,
        "--thr_yolo_raw", yolo_conf,
        "--thr_yolo_iou", yolo_iou,
    ]
    if mode == "gate":
        return common + ["--points-per-side", "64", "--pred-iou", "0.88", "--stability", "0.92"]
    if mode == "xgb":
        return common + ["--points-per-side", "64", "--pred-iou", "0.88", "--stability", "0.92"] + decision
    if mode == "xgb_recall":
        extra = ["--export-features"] if save_all else []
        return (common + [
            "--points-per-side", "64", "--points-per-batch", "512",
            "--crop-n-layers", "0", "--crop-n-points-downscale-factor", "2", "--crop-overlap-ratio", "0.4",
            "--pred-iou", "0.80", "--stability", "0.88", "--no-exclude-largest", "--min-true-gates", "3",
        ] + extra + decision)
    raise ValueError(f"unsupported legacy runner mode: {mode}")


def normalize_sam2_config(name: str, sam2_package_dir: Path) -> str:
    """Runner behaviour: YAML path -> '<parent>/<stem>'; then auto-prefix 'configs/'."""

    cfg = str(name)
    if cfg.endswith(".yaml"):
        p = Path(cfg)
        cfg = f"{p.parent.name}/{p.stem}"
    if not cfg.endswith(".yaml"):
        for cand in (sam2_package_dir / "configs" / f"{cfg}.yaml",
                     sam2_package_dir.parent / "configs" / f"{cfg}.yaml"):
            if cand.is_file():
                cfg = f"configs/{cfg}"
                break
    return cfg


def parse_legacy_args(argv: Sequence[str]):
    return make_argparser().parse_args(list(argv))


def _dst_padded_for(src: Path, work_dir: Path) -> Path:
    # original: src.with_name(f"{src.stem}_padded{src.suffix}") (beside the input);
    # compatibility: same file name, written inside the run's work directory.
    return work_dir / f"{src.stem}_padded{src.suffix}"


def pad_proto_csv(args: Any, work_dir: Path) -> dict[str, Any]:
    """Port of ``_maybe_pad_proto_csv_inplace`` (pandas read -> rename -> write)."""

    import pandas as pd

    src = str(getattr(args, "embed_csv", "") or "").strip()
    rec: dict[str, Any] = {"input": src, "padded": None}
    if not src:
        return rec
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(f"--embed-csv not found: {p}")
    df = pd.read_csv(p)
    cols = list(df.columns)
    needs_padding = any(re.match(r"^mu_embed_(\d+)$", str(c)) for c in cols)
    if not needs_padding:
        return rec
    ren = {}
    for c in cols:
        m = re.match(r"^mu_embed_(\d+)$", str(c))
        if m:
            ren[c] = f"mu_embed_{int(m.group(1)):04d}"
    if not ren:
        return rec
    work_dir.mkdir(parents=True, exist_ok=True)
    dst = _dst_padded_for(p, work_dir)
    df.rename(columns=ren).to_csv(dst, index=False)
    setattr(args, "embed_csv", str(dst))
    rec["padded"] = str(dst)
    return rec


def override_xgb_features_from_schema(args: Any, features: Sequence[str] | None) -> str | None:
    """Port of ``_maybe_override_xgb_features_from_model`` using the verified schema."""

    if not features:
        return None
    out_dir = Path(getattr(args, "out", "runs") or "runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_file = out_dir / "_xgb_features_from_model.txt"
    feat_file.write_text("\n".join(str(c) for c in features), encoding="utf-8")
    args.xgb_features = f"@{feat_file}"
    return str(feat_file)
