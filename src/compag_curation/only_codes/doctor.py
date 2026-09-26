"""``only-codes doctor``: compare the running environment with the reference.

The report states exactly what was checked, what differs and what is unknown, and keeps three
things apart:

* ``software_environment`` - interpreter and library versions, the SAM2 source tree hash;
* ``hardware_observed`` / ``hardware_match`` - the GPU/driver actually present (an unknown or
  differing GPU is reported as such, and the same GPU model is *not* a determinism proof);
* ``blockers`` - what cannot run at all here.

Readiness is not output parity: even a fully matched environment only means a reference run may be
attempted, never that its outputs were reproduced.  Nothing is installed or changed.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any

from .contract import REFERENCE_ENVIRONMENT


def _sam2_tree() -> tuple[str | None, int]:
    try:
        import sam2
    except Exception:
        return None, 0
    root = Path(sam2.__file__).resolve().parent
    h = hashlib.sha256()
    n = 0
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".yaml") and "__pycache__" not in p.parts:
            h.update(p.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(p.read_bytes()).hexdigest().encode() + b"\n")
            n += 1
    return h.hexdigest(), n


def _hardware_record() -> dict[str, Any]:
    """What the GPU actually is here (reported, never used as a parity proof)."""

    rec: dict[str, Any] = {"gpu_name": None, "driver": None, "cuda_runtime": None, "device_count": 0,
                           "source": "torch"}
    try:
        import torch

        rec["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            rec["device_count"] = int(torch.cuda.device_count())
            rec["gpu_name"] = torch.cuda.get_device_name(0)
            rec["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}"
    try:                        # driver version from the in-process NVML binding (no subprocess)
        import pynvml

        pynvml.nvmlInit()
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
            rec["driver"] = driver.decode() if isinstance(driver, bytes) else str(driver)
            if rec["gpu_name"] is None and pynvml.nvmlDeviceGetCount():
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle)
                rec["gpu_name"] = name.decode() if isinstance(name, bytes) else str(name)
            rec["source"] = "torch+nvml"
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        rec["driver_source_unavailable"] = type(exc).__name__
    return rec


def environment_report() -> dict[str, Any]:
    observed: dict[str, Any] = {"python": ".".join(map(str, sys.version_info[:3]))}
    for name in ("numpy", "scipy", "sklearn", "imblearn", "xgboost", "cv2", "torch", "torchvision", "ultralytics",
                 "pandas", "joblib", "PIL", "cupy", "pycocotools"):
        try:
            mod = importlib.import_module(name)
            observed[name] = getattr(mod, "__version__", None)
            if name == "pycocotools" and observed[name] is None:
                from importlib.metadata import version

                observed[name] = version("pycocotools")
        except Exception as exc:
            observed[name] = f"UNAVAILABLE ({type(exc).__name__})"
    try:
        import torch

        observed["torch_cuda"] = torch.version.cuda
        observed["cudnn"] = torch.backends.cudnn.version()
        observed["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    observed["sam2_source_tree_sha256"], observed["sam2_source_files"] = _sam2_tree()
    hardware = _hardware_record()
    checked = [k for k in REFERENCE_ENVIRONMENT if k != "gpu"]
    unknown = [k for k in checked if observed.get(k) is None or str(observed.get(k)).startswith("UNAVAILABLE")]
    diffs = {k: {"reference": v, "observed": observed.get(k)} for k, v in REFERENCE_ENVIRONMENT.items()
             if k != "gpu" and observed.get(k) != v}
    ref_gpu = str(REFERENCE_ENVIRONMENT.get("gpu", ""))
    if hardware.get("gpu_name") is None:
        hardware_match = "UNKNOWN"
    elif hardware.get("gpu_name") in ref_gpu and (hardware.get("driver") or "") in ref_gpu:
        hardware_match = "SAME_MODEL_AND_DRIVER_AS_REFERENCE"
    else:
        hardware_match = "DIFFERENT_FROM_REFERENCE"
    status = "MATCHED_REFERENCE_ENVIRONMENT" if not diffs else "EXECUTION_VARIANT_ENVIRONMENT"
    blockers = []
    if str(observed.get("ultralytics", "")).startswith("UNAVAILABLE"):
        blockers.append("ultralytics missing: the reference hybrid detector cannot run (BLOCKED_ENVIRONMENT)")
    if str(observed.get("cupy", "")).startswith("UNAVAILABLE"):
        blockers.append("cupy missing: the notebook's CUDA training criterion is not met (training device differs)")
    if not observed.get("cuda_available"):
        blockers.append("CUDA unavailable: the reference device (cuda) cannot be used")
    if observed.get("sam2_source_tree_sha256") is None:
        blockers.append("sam2 not importable")
    return {"schema": "compag-only-codes-doctor/v2", "status": status,
            "software_environment_match": "MATCH" if not diffs else "DIFFERENT",
            "checked_software_factors": checked, "unknown_software_factors": unknown,
            "observed": observed, "reference": REFERENCE_ENVIRONMENT, "differences": diffs,
            "hardware_observed": hardware, "hardware_match": hardware_match,
            "hardware_reference": REFERENCE_ENVIRONMENT.get("gpu"),
            "blockers": blockers,
            "output_parity_verification": "NOT_ESTABLISHED_BY_DOCTOR",
            "note": "software factors listed above were compared; hardware is reported separately and "
                    "the same GPU model does not by itself prove identical outputs. Readiness never "
                    "establishes output parity: only a comparator result bound to two runs does."}
