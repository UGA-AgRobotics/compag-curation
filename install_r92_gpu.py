#!/usr/bin/env python3
"""Plan or create the pinned, separate science GPU environment for an r92 round.

The public CPU installer remains separate. This script uses only the bundled
Conda lock, pip hash locks, official SAM2 commit and the included matching wheel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

VERSION = "1.9.4rc10"
LOCK_NAME = "conda-science-gpu-cp312-linux-x86_64.lock.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True, help="new absolute Conda environment directory")
    parser.add_argument("--cache-dir", type=Path, help="verified Conda archive cache; default is beside prefix")
    parser.add_argument("--execute", action="store_true", help="perform installation; otherwise print plan")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    prefix = args.prefix.expanduser().absolute()
    wheel = args.wheel.expanduser().resolve(strict=True)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("science GPU environment requires Linux x86-64")
    if not args.prefix.expanduser().is_absolute() or prefix == source or source in prefix.parents:
        parser.error("prefix must be absolute and outside the source tree")
    if wheel.name != f"compag_curation-{VERSION}-py3-none-any.whl":
        parser.error("select the matching 1.9.4rc10 wheel from this GitHub Release")
    sidecar = Path(str(wheel) + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split() != [sha(wheel), wheel.name]:
        parser.error("wheel SHA-256 sidecar is missing or does not match")
    conda = shutil.which("conda")
    if conda is None:
        parser.error("Conda command is required for the pinned science GPU closure")
    requirements = source / "requirements"
    lock_path = requirements / LOCK_NAME
    lock = json.loads(lock_path.read_text())
    packages = lock.get("packages")
    if (lock.get("schema") != "compag-curation-conda-lock/v1"
            or lock.get("release_version") != "1.9.3"
            or not isinstance(packages, list) or len(packages) != lock.get("package_count")
            or len(packages) != 71):
        parser.error("bundled Conda lock is invalid")
    cache = args.cache_dir.expanduser().absolute() if args.cache_dir else prefix.parent / (prefix.name + "-archives")
    if cache == prefix or prefix in cache.parents:
        parser.error("archive cache must be outside the environment prefix")
    for row in packages:
        url, digest = row["url"], row["sha256"]
        if (not url.startswith(("https://conda.anaconda.org/conda-forge/",
                                "https://conda.anaconda.org/nvidia/"))
                or len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest)
                or not url.endswith("/" + row["filename"])):
            parser.error("Conda lock contains an unexpected archive source")
    gpu_pip = requirements / "constraints-science-gpu-cp312-linux-x86_64.txt"
    sam2 = requirements / "requirements-sam2-gpu-cp312-linux-x86_64.txt"
    bootstrap = requirements / "constraints-bootstrap-cp312-linux-x86_64.txt"
    if "2b90b9f5ceec907a1c18123530e92e794ad901a4" not in sam2.read_text():
        parser.error("SAM2 source commit differs from supported Full profile")
    if prefix.exists():
        parser.error("prefix already exists; choose a new environment directory")
    for ancestor in (prefix.parent, *prefix.parents):
        if ancestor.exists():
            info = ancestor.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                parser.error(f"GPU prefix parent is not safe for science-gpu attestation: {ancestor}")
    free = shutil.disk_usage(prefix.parent if prefix.parent.exists() else source).free
    plan = {"schema": "compag-r92-rc6-gpu-install-plan/v1", "version": VERSION,
            "prefix": str(prefix), "archive_cache": str(cache),
            "wheel": str(wheel), "wheel_sha256": sha(wheel),
            "conda_lock_sha256": sha(lock_path), "conda_package_count": len(packages),
            "conda_archive_bytes": lock["total_package_bytes"],
            "pip_lock_sha256": sha(gpu_pip), "sam2_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
            "free_disk_gib": round(free / 1024**3, 2), "requires_cuda_gpu": True,
            "estimated_free_disk_gib_needed": 15}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0
    if free < 15 * 1024**3:
        parser.error("at least 15 GiB free disk is required")
    # Conda and pip otherwise inherit a collaborative umask (often 0002),
    # which makes installed code unverifiable by the science-gpu doctor.
    os.umask(0o022)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PIP_NO_INPUT"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    try:
        info = json.loads(subprocess.check_output([conda, "info", "--json"], env=env, text=True))
        conda_caches = [Path(value) for value in info.get("pkgs_dirs", [])]
    except (OSError, subprocess.CalledProcessError, ValueError):
        conda_caches = []
    lines = ["@EXPLICIT"]
    for row in packages:
        target = cache / row["filename"]
        if target.is_file() and target.stat().st_size == row["size_bytes"] and sha(target) == row["sha256"]:
            pass
        else:
            copied = False
            for existing_cache in conda_caches:
                candidate = existing_cache / row["filename"]
                if candidate.is_file() and candidate.stat().st_size == row["size_bytes"] and sha(candidate) == row["sha256"]:
                    shutil.copyfile(candidate, target)
                    copied = True
                    break
            if not copied:
                partial = target.with_suffix(target.suffix + ".part")
                print(f"FETCH {row['filename']} ({row['size_bytes']} bytes)", flush=True)
                with urllib.request.urlopen(row["url"], timeout=90) as source_file, partial.open("wb") as dest_file:
                    shutil.copyfileobj(source_file, dest_file, length=1024 * 1024)
                if partial.stat().st_size != row["size_bytes"] or sha(partial) != row["sha256"]:
                    partial.unlink(missing_ok=True)
                    raise ValueError(f"Conda archive size/SHA-256 mismatch: {row['filename']}")
                os.replace(partial, target)
        if target.stat().st_size != row["size_bytes"] or sha(target) != row["sha256"]:
            raise ValueError(f"verified Conda archive changed: {row['filename']}")
        lines.append(target.as_uri())
    with tempfile.TemporaryDirectory(prefix="compag-r92-conda-spec-", dir=prefix.parent) as tmp:
        explicit = Path(tmp) / "explicit.txt"
        explicit.write_text("\n".join(lines) + "\n")
        commands = [
            [conda, "create", "--prefix", str(prefix), "--file", str(explicit), "--yes", "--no-default-packages"],
            [str(prefix / "bin/python"), "-m", "pip", "install", "--require-hashes", "--no-deps", "--no-build-isolation", "-r", str(bootstrap)],
            [str(prefix / "bin/python"), "-m", "pip", "install", "--require-hashes", "--no-deps", "--no-build-isolation", "--no-compile", "--force-reinstall", "-r", str(gpu_pip)],
            [str(prefix / "bin/python"), "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--no-compile", "-r", str(sam2)],
            [str(prefix / "bin/python"), "-m", "pip", "install", "--no-deps", "--no-compile", "--force-reinstall", str(wheel)],
            [str(prefix / "bin/python"), "-m", "pip", "check"],
            [str(prefix / "bin/python"), "-I", "-B", "-m", "compag_curation", "doctor", "--profile", "science-gpu"],
        ]
        for command in commands:
            print("RUN", " ".join(command), flush=True)
            step_env = env.copy()
            if command == commands[3]:
                step_env["SAM2_BUILD_CUDA"] = "1"
                step_env["SAM2_BUILD_ALLOW_ERRORS"] = "0"
            subprocess.run(command, env=step_env, check=True)
    (prefix / "COMPAG_R92_GPU_INSTALL_RECEIPT.json").write_text(json.dumps({
        "schema": "compag-r92-rc6-gpu-install/v1", "status": "PASS", "plan": plan}, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "command": f"{prefix / 'bin/python'} -I -B -m compag_curation"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
