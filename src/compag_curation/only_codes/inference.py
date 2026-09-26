"""Legacy tiled inference (Only_codes ``xgb_recall`` runner) inside the app.

The sequence mirrors ``sam2_and_filter.py:main()`` exactly:

1. parse the runner argv with the original argparser;
2. normalize the SAM2 Hydra name (no-op for ``configs/...``);
3. set the global gate_core feature mode;
4. re-pad the prototype CSV through pandas (written to the run work dir);
5. inject the ordered feature schema of the selected model;
6. ``run_pipeline(args)`` (vendored original) with the classifier coming from
   the verified pickle-free bundle.

Outputs keep the original relative layout: ``<IMG>/run_<mode>/detections.csv``,
``features_pool.csv``, ``al_candidates.csv``, ``gate_summary.csv``,
``_xgb_features_from_model.txt`` and the overlay JPEGs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from . import PROFILE_ID
from .contract import contract_sha256, environment_snapshot


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_versions() -> dict[str, Any]:
    out: dict[str, Any] = {"python": sys.version, "platform": platform.platform()}
    for name in ("numpy", "scipy", "sklearn", "imblearn", "xgboost", "cv2", "torch",
                 "torchvision", "ultralytics", "pandas", "PIL", "sam2", "joblib"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", None) or getattr(mod, "__file__", None)
        except Exception as exc:  # pragma: no cover - environment dependent
            out[name] = f"UNAVAILABLE: {type(exc).__name__}"
    try:
        import torch

        out["torch_cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version()
        out["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import sam2

        out["sam2_package_dir"] = str(Path(sam2.__file__).resolve().parent)
    except Exception:
        pass
    return out


def _sanitize_args(args: Any) -> dict[str, Any]:
    out = {}
    for k, v in sorted(vars(args).items()):
        if k in ("xgb_proba_fn",):
            out[k] = "<verified legacy model bundle predictor>"
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = repr(v)
    return out


def _doctor_summary() -> dict[str, Any]:
    """Software/hardware readiness of this process, as separate observations."""

    try:
        from .doctor import environment_report

        rep = environment_report()
        return {"status": rep["status"], "software_environment_match": rep["software_environment_match"],
                "hardware_match": rep["hardware_match"], "hardware_observed": rep["hardware_observed"],
                "unknown_software_factors": rep["unknown_software_factors"], "blockers": rep["blockers"]}
    except Exception as exc:  # pragma: no cover - readiness reporting must not fail a run
        return {"status": "UNKNOWN", "error": f"{type(exc).__name__}: {exc}"}


def _effective_config_match(args, *, device: str, det_missing: str) -> dict[str, Any]:
    """Compare the effective runner configuration with the recorded reference preset."""

    from .contract import REFERENCE_INFERENCE

    checked, differences = [], {}
    mapping = {
        "sam2_policy": "sam2_policy", "points_per_side": "points_per_side",
        "points_per_batch": "points_per_batch", "crop_n_layers": "crop_n_layers",
        "crop_overlap_ratio": "crop_overlap_ratio", "pred_iou": "pred_iou", "stability": "stability",
        "min_true_gates": "min_true_gates", "xgb_threshold": "xgb_threshold", "det_policy": "det_policy",
        "det_missing": "det_missing", "det_thr": "det_thr", "hybrid_yolo_bias": "hybrid_yolo_bias",
        "yolo_conf": "yolo_conf", "yolo_iou": "yolo_iou", "yolo_imgsz": "yolo_imgsz",
        "yolo_max_det": "yolo_max_det", "feature_mode": "feature_mode",
    }
    for arg_name, ref_key in mapping.items():
        if ref_key not in REFERENCE_INFERENCE or not hasattr(args, arg_name):
            continue
        want, got = REFERENCE_INFERENCE[ref_key], getattr(args, arg_name)
        checked.append(arg_name)
        try:
            same = (float(want) == float(got)) if isinstance(want, (int, float)) and not isinstance(want, bool) \
                else str(want) == str(got)
        except (TypeError, ValueError):
            same = str(want) == str(got)
        if not same:
            differences[arg_name] = {"reference": want, "effective": got}
    if str(REFERENCE_INFERENCE.get("device", "cuda")) != str(device):
        differences["device"] = {"reference": REFERENCE_INFERENCE.get("device", "cuda"), "effective": device}
    checked.append("device")
    if str(REFERENCE_INFERENCE.get("det_missing", "reject")) != str(det_missing):
        differences["det_missing_argument"] = {"reference": REFERENCE_INFERENCE.get("det_missing"),
                                               "effective": det_missing}
    return {"status": "MATCH" if not differences else "DIFFERENT", "checked": sorted(set(checked)),
            "differences": differences,
            "note": "compared against contract.REFERENCE_INFERENCE; a match means the preset was "
                    "applied, not that outputs were verified"}


class InferencePreflightError(RuntimeError):
    """A dependency the requested run actually consumes is missing or does not match its pin."""

    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def _input_identity(path, pinned: dict[str, str] | None, key: str) -> dict[str, Any]:
    """Hash an input and say whether that hash was checked against an independent pin."""

    if path is None:
        return {"path": None, "sha256": None, "status": "NOT_SUPPLIED"}
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "sha256": None, "status": "MISSING"}
    got = _sha256(p)
    want = (pinned or {}).get(key)
    if want is None:
        return {"path": str(p), "sha256": got, "status": "HASH_RECORDED_NO_PINNED_REFERENCE"}
    return {"path": str(p), "sha256": got, "expected_sha256": want,
            "status": "VERIFIED_AGAINST_PINNED_REFERENCE" if got == want else "MISMATCH_WITH_PINNED_REFERENCE"}


def run_legacy_inference(
    *,
    tiles_dir: Path | str,
    run_root: Path | str,
    model_bundle_dir: Path | str,
    padded_proto: Path | str,
    pca_npz: Path | str,
    sam2_ckpt: Path | str,
    resnet50_weights: Path | str,
    yolo_weights: Path | str | None,
    sam2_config: str = "sam2.1/sam2.1_hiera_l",
    mode: str = "xgb_recall",
    xgb_threshold: str = "0.50",
    device: str = "cuda",
    det_missing: str = "reject",
    execution_variant: dict[str, Any] | None = None,
    classifier_override: Any = None,
    pinned_inputs: dict[str, str] | None = None,
    detector: str = "required",
) -> dict[str, Any]:
    """Run the reference inference; ``run_root`` corresponds to OLD ``RUNS_ROOT``.

    ``execution_variant`` may only be used for explicitly documented,
    non-reference variants (for example ``{"points_per_batch": 64}`` when the
    reference batch does not fit); such runs are labelled non-equivalent.

    ``detector`` is ``"required"`` for the reference hybrid preset: a missing or unreadable YOLO
    model is a preflight failure, never a silent fall-back to the no-detector branch.  The
    intentionally supported detector-free path is selected explicitly with ``detector="none"`` and
    is recorded as a declared variant.

    The manifest keeps the preset, the effective configuration, the input identities, the software
    environment, the observed hardware, the variant and the run outcome in separate fields.  Output
    parity is never asserted here: it starts as ``NOT_VERIFIED_IN_THIS_RUN`` and only a comparator
    bound to both runs may establish it.
    """

    from .model_bundle import BUNDLE_MANIFEST, load_safe_bundle

    # ---- preflight of what this run actually consumes -------------------------------------
    if detector not in ("required", "none"):
        raise ValueError("detector must be 'required' (reference hybrid) or 'none' (declared variant)")
    identities = {
        "model_bundle": ({"path": str(model_bundle_dir), "sha256": None,
                          "status": "NOT_USED_CLASSIFIER_OVERRIDE"} if classifier_override is not None
                         else _input_identity(Path(model_bundle_dir) / BUNDLE_MANIFEST, pinned_inputs,
                                              "model_bundle")),
        "padded_proto": _input_identity(padded_proto, pinned_inputs, "padded_proto"),
        "pca_npz": _input_identity(pca_npz, pinned_inputs, "pca_npz"),
        "sam2_ckpt": _input_identity(sam2_ckpt, pinned_inputs, "sam2_ckpt"),
        "resnet50_weights": _input_identity(resnet50_weights, pinned_inputs, "resnet50_weights"),
        "yolo_weights": _input_identity(yolo_weights, pinned_inputs, "yolo_weights"),
    }
    preflight_problems = [{"input": k, "status": v["status"]} for k, v in identities.items()
                          if k != "yolo_weights"                 # the detector is judged by the rule below
                          and (v["status"] in ("MISSING", "MISMATCH_WITH_PINNED_REFERENCE", "NOT_SUPPLIED"))]
    if identities["yolo_weights"]["status"] == "MISMATCH_WITH_PINNED_REFERENCE":
        preflight_problems.append({"input": "yolo_weights", "status": "MISMATCH_WITH_PINNED_REFERENCE"})
    if detector == "required" and identities["yolo_weights"]["status"] in ("NOT_SUPPLIED", "MISSING"):
        preflight_problems.append({"input": "yolo_weights", "status": identities["yolo_weights"]["status"],
                                   "detail": "the reference hybrid preset consumes the YOLO detector; "
                                             "select detector='none' explicitly for the detector-free variant"})
    if preflight_problems:
        raise InferencePreflightError(
            "inference preflight failed: " + ", ".join(f"{p['input']}={p['status']}" for p in preflight_problems),
            {"preflight": preflight_problems, "input_identities": identities, "detector": detector})

    from .runner import (
        build_runner_argv,
        normalize_sam2_config,
        override_xgb_features_from_schema,
        pad_proto_csv,
        parse_legacy_args,
    )

    t0 = time.time()
    run_root = Path(run_root)
    out_dir = run_root / f"run_{mode}"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite an existing legacy run directory: {out_dir}")
    work_dir = run_root / "_only_codes_compat_work"

    import sam2  # noqa: F401  (resolve the installed SAM2 package for Hydra config lookup)

    sam2_pkg = Path(sam2.__file__).resolve().parent
    sam2_cfg = normalize_sam2_config(sam2_config, sam2_pkg)
    # ``classifier_override`` is used only by the synthetic branch-coverage parity
    # stage (a table-driven predictor identical on both sides); production runs
    # always load the verified bundle.
    bundle = classifier_override if classifier_override is not None else load_safe_bundle(model_bundle_dir)

    argv = build_runner_argv(
        images_dir=str(Path(tiles_dir)),
        out_dir=str(out_dir),
        sam2_config=sam2_cfg,
        sam2_ckpt=str(sam2_ckpt),
        padded_proto=str(padded_proto),
        pca_npz=str(pca_npz),
        yolo_weights=str(yolo_weights) if yolo_weights else None,
        xgb_model=str(model_bundle_dir),
        mode=mode,
        xgb_threshold=xgb_threshold,
        det_missing=det_missing,
    )
    args = parse_legacy_args(argv)

    # ---- sam2_and_filter.py main() pre-steps, in the original order ----
    from .legacy import gate_core

    gate_core.set_feature_mode(getattr(args, "feature_mode", "ultra"))
    pad_record = pad_proto_csv(args, work_dir / "proto")
    feat_file = override_xgb_features_from_schema(args, bundle.features)

    args.xgb_proba_fn = bundle.proba_fn()
    args.device = str(device)
    args.embed_backbone_weights = str(resnet50_weights)
    args.save_all_outputs = True
    variant = dict(execution_variant or {})
    for key, value in variant.items():
        if not hasattr(args, key):
            raise ValueError(f"unknown execution-variant key: {key}")
        setattr(args, key, value)

    if detector == "none":
        args.yolo_weights = None
        variant = {**variant, "detector": "none"}
    if classifier_override is not None:
        variant = {**variant, "classifier_override": True}

    manifest: dict[str, Any] = {
        "schema": "compag-only-codes-inference-run/v1",
        "profile": PROFILE_ID,
        "contract_sha256": contract_sha256(),
        "manifest_schema": "compag-only-codes-inference-run/v2",
        # -- separate meanings; none of them is a parity claim ------------------------------
        "reference_preset_requested": bool(not variant and det_missing == "reject" and device == "cuda"
                                           and detector == "required"),
        "effective_configuration_match": _effective_config_match(args, device=device, det_missing=det_missing),
        "inputs_verified": {"identities": identities,
                            "status": ("VERIFIED_AGAINST_PINNED_REFERENCE"
                                       if pinned_inputs and all(v["status"] == "VERIFIED_AGAINST_PINNED_REFERENCE"
                                                                for k, v in identities.items()
                                                                if v["status"] != "NOT_SUPPLIED")
                                       else "HASHES_RECORDED_WITHOUT_INDEPENDENT_PIN")},
        "execution_variant": variant,
        "detector": detector,
        "run_completed": False,
        "output_parity_verification": {
            "status": "NOT_VERIFIED_IN_THIS_RUN",
            "note": "a successful run of the requested preset is not evidence that its outputs match "
                    "the reference; only a comparator result bound to both run ids establishes that"},
        # kept for readers of the v1 schema; it now states the preset, not a verified equivalence
        "equivalence": "REFERENCE_PRESET_REQUESTED" if not variant and det_missing == "reject"
        and device == "cuda" and detector == "required" else "NON_EQUIVALENT_EXECUTION_VARIANT",
        "runner_argv": argv,
        "effective_args": _sanitize_args(args),
        "prototype": pad_record,
        "feature_schema_file": feat_file,
        "model_bundle": {"dir": str(model_bundle_dir), "source_sha256": (bundle.manifest.get("source") or {}).get("sha256"),
                         "n_features": len(bundle.features or []), "threshold_in_bundle": bundle.threshold},
        "inputs": {
            "tiles_dir": str(tiles_dir),
            "tiles": {p.name: _sha256(p) for p in sorted(Path(tiles_dir).glob("*.*"))
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")},
            "padded_proto_sha256": _sha256(Path(padded_proto)),
            "pca_sha256": _sha256(Path(pca_npz)),
            "sam2_ckpt_sha256": _sha256(Path(sam2_ckpt)),
            "resnet50_sha256": _sha256(Path(resnet50_weights)),
            "yolo_sha256": (_sha256(Path(yolo_weights))
                            if yolo_weights and Path(yolo_weights).is_file() else None),
        },
        "environment": environment_snapshot(),
        "software_environment_match": _doctor_summary(),
        "versions": runtime_versions(),
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
    }

    from .legacy.pipeline import run_pipeline

    run_pipeline(args)
    manifest["run_completed"] = True
    manifest["seconds"] = time.time() - t0
    manifest["outputs"] = {p.relative_to(run_root).as_posix(): _sha256(p)
                           for p in sorted(out_dir.rglob("*")) if p.is_file()}
    (run_root / "ONLY_CODES_INFERENCE_RUN.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return manifest
