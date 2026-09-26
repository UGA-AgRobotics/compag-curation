#!/usr/bin/env python3
"""Install the laptop entry and optional native r92 CPU sample in an isolated venv.

The recorded Only_codes GPU closure is separate. The r92 sample profile installs
CPU XGBoost but never silently substitutes it for full image inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import venv
from pathlib import Path

VERSION = "1.9.4rc10"
STATE = "LAPTOP_INSTALL_STATE.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sidecar(wheel: Path) -> str:
    sidecar = Path(str(wheel) + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing wheel checksum sidecar: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1].lstrip("*") != wheel.name or len(fields[0]) != 64:
        raise ValueError("wheel sidecar must contain: SHA256  wheel_filename")
    observed = sha256(wheel)
    if observed != fields[0].lower():
        raise ValueError(f"wheel hash differs from sidecar: {observed}")
    return observed


def _write_state(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _run(command: list[str], env: dict[str, str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path, help="matching public wheel with adjacent .sha256")
    parser.add_argument("--venv", required=True, type=Path, help="new absolute environment directory")
    parser.add_argument("--profile", choices=("quick", "r92-sample"), default="r92-sample",
                        help="r92-sample adds pinned CPU XGBoost; quick keeps the model-free entry")
    parser.add_argument("--cache-dir", type=Path, help="new or existing user-owned pip download cache")
    parser.add_argument("--execute", action="store_true", help="perform the planned install; default only prints a plan")
    args = parser.parse_args(argv)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("this candidate supports Linux x86-64 only")
    if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 12):
        parser.error("invoke this script using CPython 3.12")
    root = Path(__file__).resolve().parent
    wheel = args.wheel.expanduser().resolve(strict=True)
    if not args.venv.expanduser().is_absolute():
        parser.error("--venv must be an absolute path")
    venv_root = args.venv.expanduser().resolve(strict=False)
    if not venv_root.is_absolute() or venv_root.resolve().is_relative_to(root):
        parser.error("--venv must be an absolute path outside the source tree")
    if wheel.name != f"compag_curation-{VERSION}-py3-none-any.whl":
        parser.error(f"expected the {VERSION} wheel, got {wheel.name}")
    wheel_hash = _sidecar(wheel)
    lock_names = ["bootstrap", "quick-demo", "laptop-review"]
    if args.profile == "r92-sample":
        lock_names.append("r92-cpu")
    constraints = [root / "requirements" / f"constraints-{name}-cp312-linux-x86_64.txt"
                   for name in lock_names]
    if not all(path.is_file() for path in constraints):
        parser.error("matching bootstrap, quick and review dependency locks are required beside this installer")
    cache = (args.cache_dir.expanduser().resolve(strict=False) if args.cache_dir
             else venv_root.parent / (venv_root.name + "-cache"))
    state_path = venv_root / STATE
    plan = {"schema": "compag-laptop-install-plan/v1", "profile": args.profile,
            "version": VERSION, "wheel": str(wheel), "wheel_sha256": wheel_hash,
            "venv": str(venv_root), "cache": str(cache), "interpreter": sys.executable,
            "python": platform.python_version(), "constraints": {p.name: sha256(p) for p in constraints},
            "isolation": "venv without system site packages; PYTHONPATH removed; user site disabled",
            "estimated_download_mib_upper_bound": 300, "estimated_free_disk_mib_needed": 1024,
            "scientific_gpu_reference_ready": False}
    free = shutil.disk_usage(venv_root.parent if venv_root.parent.exists() else root).free
    plan["free_disk_mib"] = free // (1024 * 1024)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    if free < 1024**3:
        parser.error("at least 1 GiB of free disk space is required before installation")
    if venv_root.exists():
        if not state_path.is_file():
            parser.error("target exists but is not managed by this installer; choose a new path")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("plan", {}).get("wheel_sha256") != wheel_hash or state.get("plan", {}).get("venv") != str(venv_root) or state.get("plan", {}).get("profile") != args.profile:
            parser.error("existing managed environment has a different wheel, profile or path")
    else:
        venv_root.mkdir(parents=True)
        state = {"status": "INSTALLING", "plan": plan, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        _write_state(state_path, state)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INPUT"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_CACHE_DIR"] = str(cache)
    cache.mkdir(parents=True, exist_ok=True)
    try:
        venv.EnvBuilder(system_site_packages=False, with_pip=True, clear=False).create(venv_root)
        python = str(venv_root / "bin" / "python")
        for lock in constraints:
            _run([python, "-m", "pip", "install", "--require-hashes", "--no-deps", "-r", str(lock)], env)
        _run([python, "-m", "pip", "install", "--no-build-isolation", "--no-deps", str(wheel)], env)
        _run([python, "-m", "pip", "check"], env)
        _run([str(venv_root / "bin" / "compag-curation"), "doctor", "--profile", "quick"], env)
        _run([python, "-c", "import pandas,cv2,numpy; assert pandas.__version__ == '2.2.3'"], env)
        _run([str(venv_root / "bin" / "compag-curation"), "only-codes", "guide", "--help"], env)
        if args.profile == "r92-sample":
            _run([python, "-c", "import xgboost,numpy,scipy; assert xgboost.__version__ == '2.1.1' and numpy.__version__ == '2.0.2'"], env)
            _run([str(venv_root / "bin" / "compag-curation"), "r92", "sample", "--help"], env)
        proof = subprocess.check_output([python, "-c", "import sys,site,json; print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'site':site.getsitepackages(),'user_site_enabled':site.ENABLE_USER_SITE}))"], env=env, text=True)
        isolation = json.loads(proof)
        if isolation["prefix"] == isolation["base_prefix"] or isolation["user_site_enabled"]:
            raise RuntimeError("environment isolation proof failed")
        state.update(status="READY_R92_CPU_SAMPLE" if args.profile == "r92-sample" else "READY_QUICK_ONLY", completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     isolation=isolation)
        _write_state(state_path, state)
        print(json.dumps({"status": state["status"], "launcher": str(venv_root / "bin" / "compag-laptop"),
                          "receipt": str(state_path)}, indent=2))
        return 0
    except Exception as exc:
        state.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}")
        _write_state(state_path, state)
        print(f"Installation incomplete: {exc}. Repair the cause and rerun the same request to resume.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
