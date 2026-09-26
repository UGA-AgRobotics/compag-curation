#!/usr/bin/env python3
"""Prepare an isolated, hash-pinned YOLO segmentation inference overlay.

The reviewed science-GPU Python provides the fixed CUDA/Torch stack. A child
virtual environment exposes that stack without changing it and installs only
the pinned Ultralytics inference dependencies that are absent from the parent.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements/yolo-segment-overlay-cp312-linux-x86_64.txt"


def _run(argv: list[str], *, env: dict[str, str], capture: bool = False) -> str:
    result = subprocess.run(argv, env=env, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()[-1200:]
        raise RuntimeError(f"YOLO setup command failed ({result.returncode}): {detail}")
    return result.stdout or ""


def _versions(python: Path, env: dict[str, str]) -> dict[str, str]:
    code = """import importlib.metadata as m
import json
import platform
import sys

names = ('compag-curation', 'ultralytics', 'ultralytics-thop',
         'torch', 'torchvision', 'numpy', 'opencv-python', 'pillow')
versions = {}
for name in names:
    try:
        versions[name] = m.version(name)
    except m.PackageNotFoundError:
        versions[name] = None
print(json.dumps({'python': list(sys.version_info[:2]),
                  'implementation': platform.python_implementation(),
                  'platform': platform.system(), 'machine': platform.machine(),
                  'versions': versions}))
"""
    return json.loads(_run([str(python), "-I", "-B", "-c", code], env=env, capture=True))


def _verify(base: Path, child: Path, env: dict[str, str]) -> dict:
    parent = _versions(base, env)
    info = _versions(child, env)
    if parent["python"] != [3, 12] or parent["implementation"] != "CPython" or \
            parent["platform"] != "Linux" or parent["machine"] != "x86_64":
        raise RuntimeError("YOLO overlay requires the verified Linux x86-64 Python 3.12 science environment")
    if info["python"] != parent["python"] or info["platform"] != parent["platform"] or \
            info["machine"] != parent["machine"]:
        raise RuntimeError("YOLO overlay interpreter differs from its science-GPU parent")
    expected = {"compag-curation": "1.9.4rc10", "ultralytics": "8.4.26",
                "ultralytics-thop": "2.0.18", "numpy": "2.0.2",
                "opencv-python": "4.12.0.88", "pillow": "12.3.0"}
    for name, version in expected.items():
        if info["versions"].get(name) != version:
            raise RuntimeError(f"YOLO overlay has the wrong {name} version")
    for name in ("torch", "torchvision"):
        if info["versions"].get(name) != parent["versions"].get(name):
            raise RuntimeError(f"YOLO overlay changed the pinned {name} version")
    if not info["versions"]["torch"].startswith("2.13.0+") or \
            not info["versions"]["torchvision"].startswith("0.28.0+"):
        raise RuntimeError("YOLO overlay requires the reviewed science-GPU Torch build")
    _run([str(child), "-m", "pip", "check"], env=env, capture=True)
    return {"schema": "compag-yolo-segment-overlay/v1", "status": "PASS",
            "python": str(child), "base_python": str(base), "versions": info["versions"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new separate virtual environment outside the source tree")
    args = parser.parse_args(argv)
    requested = args.base_python.expanduser().absolute()
    base = requested.parent.resolve(strict=True) / requested.name
    output = args.output.expanduser().absolute()
    if not base.is_file() or output.is_symlink() or not output.is_absolute() or \
            output == ROOT or ROOT in output.parents:
        parser.error("choose a regular science Python and an output outside the repository")
    if not LOCK.is_file():
        parser.error("YOLO overlay hash lock is missing")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    parent = _versions(base, env)
    if parent["versions"].get("compag-curation") != "1.9.4rc10":
        parser.error("base Python must contain the matching COMPAG Curation 1.9.4rc10 installation")
    child = output / "bin/python"
    created = False
    try:
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            # The new path belongs to this invocation even if venv fails halfway.
            created = True
            _run([str(base), "-m", "venv", "--system-site-packages", str(output)], env=env)
            _run([str(child), "-m", "pip", "install", "--require-hashes", "--no-deps",
                  "--only-binary=:all:", "-r", str(LOCK)], env=env)
        # A nested venv inherits the base interpreter site, not necessarily the
        # invoking venv's own COMPAG wheel. Bind the child to this release too.
        if _versions(child, env)["versions"].get("compag-curation") != "1.9.4rc10":
            if not created:
                raise RuntimeError("Existing YOLO environment has a different COMPAG version; choose a new output folder")
            spec = importlib.util.spec_from_file_location("release_launcher", ROOT / "start_compag.py")
            release = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(release)
            wheel = ROOT / "bundled-assets" / release.WHEEL
            if wheel.is_symlink() or not wheel.is_file() or release._sha256(wheel) != release.WHEEL_SHA256:
                raise RuntimeError("Matching bundled COMPAG wheel failed its release checksum")
            _run([str(child), "-m", "pip", "install", "--no-deps", "--ignore-installed", str(wheel)], env=env)
        result = _verify(base, child, env)
    except Exception as exc:
        if created:
            shutil.rmtree(output, ignore_errors=True)
        print(f"YOLO support setup unavailable: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
