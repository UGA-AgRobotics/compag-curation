#!/usr/bin/env python3
"""Friendly local setup and round launcher for the unchanged COMPAG r92 engine.

Run ``python3 start_compag.py`` after cloning the public repository. This file
uses only the Python standard library; the pinned installers and application
commands remain the sole owners of dependencies, inference and training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import threading
import uuid
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parent
BUNDLED_ASSETS = ROOT / "bundled-assets"
VERSION = "1.9.4rc10"
WHEEL = f"compag_curation-{VERSION}-py3-none-any.whl"
WHEEL_SHA256 = "b32bf6f2b48ec38904061cd956e9ecde949c0c913275c10bdd5d51b6dfcc6bc8"
MODEL = "COMPAG_CJ_R92_INFERENCE_SAMPLE_20260924T200000Z.zip"
MODEL_SHA256 = "fffef7a85192b99dcca3bcc5750215150d2ca9659121363198041244bbfe57c1"
PHOTOS = "COMPAG_R92_TEN_PHOTO_INPUTS_RC6_R3_20260924.zip"
PHOTOS_SHA256 = "344c8d767a1719403a3713e0f51587971f4ab9ea18e1d1a44034ed72c9297914"
SETTINGS_NAME = "EASY_START_SETTINGS.json"
SESSION_NAME = "EASY_SESSION.json"


class StartError(RuntimeError):
    """A recoverable setup or workflow problem with a user-facing message."""


class JobStopped(StartError):
    """The user stopped a workflow before its next operation."""


class JobCleanupError(StartError):
    """Owned work could not be confirmed stopped; conflicting work stays blocked."""


class JobControl:
    """Own one Linux session/group until every live member has terminated.

    The unreaped session leader reserves its PID/PGID throughout cleanup. Never
    poll/wait it early: EOF or parent exit is not proof that descendants exited.
    /proc start time also detects unexpected external reaping/PID reuse. Workers
    must stay in the session/group created here (the supported Linux model).
    """

    INTERRUPT_GRACE = 5.0
    TERMINATE_GRACE = 5.0
    KILL_GRACE = 2.0
    POLL_INTERVAL = .05

    def __init__(self):
        self.stopped = threading.Event()
        self.lock = threading.RLock()
        self.process = None
        self.cleanup_done = threading.Event()
        self.cleanup_done.set()
        self.cleanup_error = None
        self._cleanup_thread = None
        self._leader_start = None

    def check(self):
        if self.cleanup_error:
            raise JobCleanupError(self.cleanup_error)
        if self.stopped.is_set():
            raise JobStopped("Stopped by you. Completed analysis tiles and saved review decisions are kept. Training can be started again.")

    @staticmethod
    def _stat(pid):
        # comm may contain spaces or parentheses. Fields after its final ')' are
        # state, ppid, pgrp, session, ... starttime (Linux proc_pid_stat(5)).
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(')', 1)[1].split()
        return fields[0], int(fields[2]), int(fields[3]), int(fields[19])

    def spawn(self, command, **kwargs):
        with self.lock:
            self.check()
            if self.process is not None:
                raise StartError("The previous owned operation is still finishing.")
            if not Path('/proc/self/stat').is_file():
                raise StartError("Owned process cleanup requires the supported Linux /proc environment.")
            process = subprocess.Popen(command, start_new_session=True, **kwargs)
            self.process = process
            self.cleanup_done.clear()
            self._cleanup_thread = None
            self._leader_start = self._stat(process.pid)[3]
            return process

    def _live_group(self, process):
        # A missing/reused leader is an ownership failure, never permission to
        # signal a group that might now belong to another job.
        leader = self._stat(process.pid)
        if leader[1:] != (process.pid, process.pid, self._leader_start):
            raise JobCleanupError("Process ownership changed; cleanup cannot be confirmed. Close the guide and inspect the owned job before retrying.")
        live = False
        for entry in Path('/proc').iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                state, group, session, _ = self._stat(entry.name)
            except (FileNotFoundError, ProcessLookupError):
                continue
            if group == process.pid:
                if session != process.pid:
                    raise JobCleanupError("Process-group ownership could not be verified.")
                live |= state not in {'Z', 'X'}
        return live

    def _cleanup(self, process):
        try:
            for sig, grace in ((signal.SIGINT, self.INTERRUPT_GRACE),
                               (signal.SIGTERM, self.TERMINATE_GRACE),
                               (signal.SIGKILL, self.KILL_GRACE)):
                with self.lock:
                    if not self._live_group(process):
                        return
                    # The unreaped leader prevents PGID reuse between this
                    # check and killpg. No client PID ever reaches this path.
                    os.killpg(process.pid, sig)
                deadline = time.monotonic() + grace
                while True:
                    if not self._live_group(process):
                        return
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(self.POLL_INTERVAL)
            raise JobCleanupError("Owned work is still present after Stop escalation. Cleanup failed; new work is blocked. Close the guide and inspect the owned job before retrying.")
        except Exception as exc:
            self.cleanup_error = f"Stop cleanup could not be confirmed: {exc}"
        finally:
            self.cleanup_done.set()

    def cancel(self):
        with self.lock:
            self.stopped.set()
            if self.process is None or self._cleanup_thread is not None:
                return
            self._cleanup_thread = threading.Thread(
                target=self._cleanup, args=(self.process,),
                name="compag-stop-cleanup", daemon=False)
            self._cleanup_thread.start()

    def finished(self, process):
        """Join cleanup before reaping the leader or releasing job ownership."""
        while True:
            with self.lock:
                if self.process is not process:
                    return process.returncode
                cleanup = self._cleanup_thread
                if cleanup is None and not self._live_group(process):
                    code = process.wait()
                    self.process = None
                    self.cleanup_done.set()
                    return code
            if cleanup is not None:
                self.cleanup_done.wait()
                cleanup.join()
                with self.lock:
                    if self.cleanup_error:
                        raise JobCleanupError(self.cleanup_error)
                    code = process.wait()
                    self.process = None
                    return code
            time.sleep(self.POLL_INTERVAL)

    def settle(self, *, failed=False):
        """Also guard callers failing outside Launcher._run."""
        if failed:
            self.cancel()
        if self.process is not None:
            self.finished(self.process)


def _path(value: Path | str) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _python_path(value: Path | str) -> Path:
    """Keep a venv's bin/python symlink while making its parent absolute."""
    path = Path(value).expanduser().absolute()
    return path.parent.resolve(strict=False) / path.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StartError(f"Cannot read {path.name}. Choose a valid workspace or session.") from exc
    if not isinstance(value, dict):
        raise StartError(f"{path.name} has an unexpected format.")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise StartError(f"Refusing a linked settings file: {path}")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PIP_NO_INPUT"] = "1"
    return env


def release_page_url() -> str | None:
    """Offer the actual clone's Releases page without inventing a repo URL."""
    try:
        result = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"],
                                capture_output=True, text=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    remote = result.stdout.strip()
    match = re.fullmatch(r"(?:https://github\.com/|git@github\.com:)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?", remote)
    return f"https://github.com/{match.group(1)}/releases" if match else None


class Launcher:
    """Small adapter around existing pinned installers and r92 commands."""

    def __init__(self, release_dir: Path | str, prefix: Path | str, workspace: Path | str,
                 log: Callable[[str], None] | None = None) -> None:
        self.release_dir = _path(release_dir)
        self.prefix = _path(prefix)
        self.workspace = _path(workspace)
        self.log = log or print
        self.control = JobControl()

    @property
    def python(self) -> Path:
        return self.prefix / "bin" / "python"

    @property
    def example(self) -> Path:
        return self.workspace / "example"

    @property
    def asset_root(self) -> Path:
        return self.workspace / "official-assets"

    @property
    def settings_path(self) -> Path:
        return self.workspace / SETTINGS_NAME

    def _run(self, command: list[str], label: str, *, quiet: bool = False) -> str:
        if not quiet:
            self.log(label)
        try:
            proc = self.control.spawn(command, cwd=ROOT, env=_clean_env(),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
        except OSError as exc:
            raise StartError(f"Could not start {label.lower()}: {exc}") from exc
        output: list[str] = []
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                output.append(line)
                if not quiet:
                    shown = line.rstrip()
                    if len(shown) > 1200:
                        try:
                            data = json.loads(shown)
                        except ValueError:
                            shown = shown[:1200] + " ... (more details omitted)"
                        else:
                            if isinstance(data, dict):
                                shown = f"{data.get('schema', 'COMPAG step')}: {data.get('status', 'completed')}"
                    self.log(shown)
        except BaseException:
            self.control.cancel()
            raise
        finally:
            proc.stdout.close()
            code = self.control.finished(proc)
        self.control.check()
        if code:
            tail = "".join(output[-8:]).strip()[-1600:]
            raise StartError(f"{label} did not finish. {tail or 'See the details above.'}")
        return "".join(output)

    def _r92(self, *args: str) -> list[str]:
        return [str(self.python), "-u", "-I", "-B", "-m", "compag_curation", *args]

    def _model_set(self, folder: Path | str) -> dict:
        """Read only a model set that the installed scientific engine verifies."""
        result = self._run(self._r92("r92", "verify-model-set", "--model-set", str(folder)),
                           "Verifying the selected model set", quiet=True)
        try:
            record = json.loads(result.strip().splitlines()[-1])
        except (IndexError, ValueError) as exc:
            raise StartError("The model-set verification returned no readable result.") from exc
        if record.get("status") != "MODEL_SET_VERIFIED" or not record.get("ready_for_next_inference"):
            raise StartError("The selected model set is not ready for the next photo.")
        return record

    def _latest_model_set(self, settings: dict) -> tuple[dict, dict]:
        latest = settings.get("latest_round")
        if not isinstance(latest, dict) or not isinstance(latest.get("model_set"), str):
            raise StartError("Train an XGBoost round before selecting a YOLO mode.")
        folder = _path(latest["model_set"])
        if self.workspace not in folder.parents or not folder.is_dir() or folder.is_symlink():
            raise StartError("The latest model set is missing from this private workspace.")
        record = self._model_set(folder)
        if (record.get("snapshot") != latest.get("snapshot")
                or record.get("round_id") != latest.get("round_id")):
            raise StartError("The saved latest round differs from its verified model set.")
        return latest, record

    def yolo_locations(self, settings: dict | None = None) -> dict:
        """Find local detector inputs without loading weights or changing a round.

        Saved identity takes precedence. Nearby discovery is deliberately shallow;
        it never searches the whole home directory or chooses a different saved
        checkpoint after a folder has moved. Full checksum/runtime verification
        still takes place when the user applies the setting.
        """
        settings = settings if settings is not None else (self.load_settings() or {})
        latest = settings.get("latest_round") or {}
        saved = settings.get("yolo_defaults") or {}
        expected = latest.get("yolo_checkpoint_sha256") or saved.get("checkpoint_sha256")
        preferred = [latest.get("yolo_model_package"), saved.get("model_package")]
        if latest.get("model_set"):
            try:
                record = _read_json(_path(latest["model_set"]) / "MODEL_SET.json")
                if record.get("yolo_attestation"):
                    attestation = _read_json(_path(record["yolo_attestation"]))
                    preferred.append(attestation.get("model_package_root"))
            except (StartError, OSError, ValueError):
                pass
        python = ""
        for value in (latest.get("yolo_python"), saved.get("python"),
                      self.workspace / "yolo_segment_env" / "bin" / "python"):
            if value and _python_path(value).is_file():
                python = str(_python_path(value))
                break

        def candidate(value):
            if not value:
                return None
            package = Path(value).expanduser().absolute()
            try:
                manifest_path = package / "MODEL_MANIFEST.json"
                checkpoint = package / "weights" / "best.pt"
                if (package.is_symlink() or manifest_path.is_symlink() or checkpoint.is_symlink()
                        or not manifest_path.is_file() or not checkpoint.is_file()):
                    return None
                manifest = _read_json(manifest_path)
                sha = manifest.get("checkpoint_sha256")
                if (not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
                        or (expected and sha != expected)
                        or manifest.get("task") != "instance segmentation"
                        or manifest.get("class_mapping") != {"0": "CJ"}):
                    return None
                return {"model_package": str(package.resolve()), "checkpoint_sha256": sha}
            except (StartError, OSError, ValueError):
                return None

        for value in preferred:
            found = candidate(value)
            if found:
                return {**found, "python": python, "status": "FOUND", "source": "saved"}
        roots = [self.workspace, ROOT]
        roots += [p for p in list(ROOT.parents)[:2] if p != Path.home() and p != Path(p.anchor)]
        candidates = {}
        for root in dict.fromkeys(roots):
            locations = [root, root / "models"]
            for parent in (root, root / "models"):
                try:
                    locations.extend(p for p in parent.iterdir() if p.is_dir() and not p.is_symlink())
                except OSError:
                    pass
            for location in locations:
                found = candidate(location)
                if found:
                    candidates[found["model_package"]] = found
        identities = {item["checkpoint_sha256"] for item in candidates.values()}
        if len(identities) == 1:
            found = candidates[sorted(candidates)[0]]
            return {**found, "python": python, "status": "FOUND", "source": "nearby"}
        return {"model_package": "", "python": python,
                "status": "MULTIPLE" if candidates else "MISSING",
                "checkpoint_sha256": expected,
                "message": ("Several different YOLO models were found. Choose one package once; COMPAG will remember it."
                            if candidates else
                            "No matching YOLO package was found. Place it in the workspace models folder, or select its folder once.")}

    def _verify_yolo_boxes(self, boxes: Path, project: Path, checkpoint_sha256: str) -> bool:
        if boxes.is_symlink() or not boxes.is_file():
            return False
        code = ("from pathlib import Path; import sys; "
                "from compag_curation.r92_yolo import load_precomputed_boxes; "
                "load_precomputed_boxes(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]); print('PASS')")
        try:
            return self._run([str(self.python), "-I", "-B", "-c", code,
                              str(boxes), str(project), checkpoint_sha256],
                             "Verifying saved YOLO box predictions", quiet=True).strip() == "PASS"
        except StartError:
            return False

    def check_release(self) -> dict:
        """Check the four immutable files bundled with a GitHub clone."""
        folder = self.release_dir
        if not folder.is_dir() or folder.is_symlink():
            raise StartError("The included COMPAG files are missing. Get a complete copy of this repository.")
        expected = {WHEEL: WHEEL_SHA256, MODEL: MODEL_SHA256, PHOTOS: PHOTOS_SHA256}
        needed = {WHEEL, WHEEL + ".sha256", MODEL, PHOTOS}
        missing = [name for name in sorted(needed) if not (folder / name).is_file()]
        if missing:
            raise StartError("The included COMPAG files are incomplete. Get a complete copy of this repository. Missing: " + ", ".join(missing))
        for name, digest in expected.items():
            path = folder / name
            if path.is_symlink() or _sha256(path) != digest:
                raise StartError(f"An included COMPAG file failed its fixed checksum: {name}. Get a clean copy of this repository.")
        sidecar_path = folder / (WHEEL + ".sha256")
        if sidecar_path.is_symlink() or sidecar_path.read_text(encoding="utf-8").split() != [WHEEL_SHA256, WHEEL]:
            raise StartError("The wheel's .sha256 sidecar does not match this version.")
        index = folder / "PUBLIC_SHA256SUMS"
        verified = set(needed)
        if index.exists():
            if not index.is_file() or index.is_symlink():
                raise StartError("PUBLIC_SHA256SUMS must be a regular file if downloaded.")
            entries: dict[str, str] = {}
            for line in index.read_text(encoding="utf-8").splitlines():
                match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
                if match is None or match.group(2) in entries:
                    raise StartError("The Release checksum list is malformed or has duplicate names.")
                entries[match.group(2)] = match.group(1)
            if not needed <= set(entries) or any(entries[name] != digest for name, digest in expected.items()):
                raise StartError("The Release checksum list belongs to a different COMPAG version.")
            for name, digest in entries.items():
                path = folder / name
                if path.exists():
                    if not path.is_file() or path.is_symlink() or _sha256(path) != digest:
                        raise StartError(f"Release file failed its checksum: {name}. Download it again.")
                    verified.add(name)
            verified.add(index.name)
        return {"status": "PASS", "version": VERSION, "file_count": len(verified),
                "folder": str(folder), "wheel_sha256": WHEEL_SHA256}

    def _environment_ready(self) -> bool:
        if not self.python.is_file():
            return False
        try:
            result = self._run(self._r92("doctor", "--profile", "science-gpu"),
                               "Checking the installed GPU environment", quiet=True)
            record = json.loads(result.strip().splitlines()[-1])
            return (record.get("status") == "PASS" and record.get("profile") == "science-gpu"
                    and record.get("checks", {}).get("versions", {}).get("compag-curation") == VERSION)
        except (StartError, ValueError, IndexError):
            return False

    def _example_ready(self) -> bool:
        receipt_path = self.example / "EXAMPLE_INPUTS.json"
        if not receipt_path.is_file() or receipt_path.is_symlink():
            return False
        receipt = _read_json(receipt_path)
        if (receipt.get("schema") != "compag-r92-public-example-inputs/v2"
                or receipt.get("model_sha256") != MODEL_SHA256
                or receipt.get("photo_archive_sha256") != PHOTOS_SHA256
                or receipt.get("photo_count") != 10
                or not isinstance(receipt.get("cards"), list)
                or len(receipt["cards"]) != 10):
            return False
        model = self.example / "model" / MODEL
        if not model.is_file() or model.is_symlink() or _sha256(model) != MODEL_SHA256:
            return False
        seen: set[str] = set()
        for row in receipt["cards"]:
            if not isinstance(row, dict):
                return False
            card_id, name, digest = row.get("card_id"), row.get("filename"), row.get("sha256")
            if (not isinstance(card_id, str) or not re.fullmatch(r"IMG_[0-9]{4}", card_id)
                    or card_id in seen or name != card_id + ".jpg"
                    or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or row.get("relative_directory") != f"cards/{card_id}"):
                return False
            seen.add(card_id)
            photo = self.example / "cards" / card_id / name
            if not photo.is_file() or photo.is_symlink() or _sha256(photo) != digest:
                return False
        return True

    def load_settings(self) -> dict | None:
        if not self.settings_path.is_file():
            return None
        record = _read_json(self.settings_path)
        if (record.get("schema") != "compag-easy-start-settings/v1"
                or record.get("version") != VERSION
                or record.get("workspace") != str(self.workspace)):
            raise StartError("This workspace was prepared by a different COMPAG version. Choose another workspace.")
        return record

    def setup(self) -> dict:
        if self.workspace == ROOT or ROOT in self.workspace.parents:
            raise StartError("Choose a work folder outside the cloned COMPAG repository so private photos and review data stay separate from the public code.")
        if self.prefix == ROOT or ROOT in self.prefix.parents:
            raise StartError("Choose an installation folder outside the cloned COMPAG repository.")
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise StartError("The complete GPU workflow currently requires Linux on x86-64.")
        self.log("Checking the files included with COMPAG...")
        release = self.check_release()
        if not self._environment_ready():
            if self.prefix.exists():
                raise StartError("The chosen installation folder already exists but is not a verified COMPAG GPU install. Choose a new empty folder; existing files will not be deleted.")
            if shutil.which("conda") is None:
                raise StartError("Conda is required for the pinned GPU installation. Install Miniconda or Miniforge, then reopen this launcher.")
            self._run([sys.executable, str(ROOT / "install_r92_gpu.py"),
                       "--wheel", str(self.release_dir / WHEEL),
                       "--prefix", str(self.prefix), "--execute"],
                      "Installing COMPAG and its verified GPU dependencies (this can take a while)")
            if not self._environment_ready():
                raise StartError("Installation finished but the GPU verification did not pass. See details above.")
        else:
            self.log("Verified COMPAG GPU installation found; using it again.")
        self.workspace.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.asset_root.mkdir(mode=0o700, exist_ok=True)
        try:
            self._run(self._r92("assets", "verify", "--asset-root", str(self.asset_root),
                                "--profile", "full"), "Checking official model assets", quiet=True)
        except StartError:
            self._run(self._r92("assets", "fetch", "--asset-root", str(self.asset_root),
                                "--profile", "full"), "Downloading verified SAM2 and ResNet assets")
        self._run(self._r92("assets", "verify", "--asset-root", str(self.asset_root),
                            "--profile", "full"), "Verifying official model assets")
        if self.example.exists():
            if not self._example_ready():
                raise StartError("The example folder is incomplete or changed. Choose a new workspace; no files were deleted.")
            self.log("The ten example photos and native model are already prepared.")
        else:
            self._run([str(self.python), "-I", "-B", str(ROOT / "tools" / "prepare_r92_public_example.py"),
                       "--model-zip", str(self.release_dir / MODEL),
                       "--photos-zip", str(self.release_dir / PHOTOS),
                       "--output", str(self.example)], "Preparing the ten example photos")
            if not self._example_ready():
                raise StartError("Example preparation did not pass verification.")
        self._run(self._r92("r92", "verify", "--model", str(self.example / "model" / MODEL)),
                  "Checking the native r92 model")
        previous = self.load_settings()
        record = {"schema": "compag-easy-start-settings/v1", "version": VERSION,
                  "status": "READY", "workspace": str(self.workspace),
                  "release_dir": str(self.release_dir), "prefix": str(self.prefix),
                  "model": str(self.example / "model" / MODEL),
                  "asset_root": str(self.asset_root), "example": str(self.example),
                  "latest_round": previous.get("latest_round") if previous else None}
        if previous and previous.get("yolo_defaults"):
            record["yolo_defaults"] = previous["yolo_defaults"]
        _write_json(self.settings_path, record)
        self.log("Ready. Choose a photo and click Analyze.")
        return record

    def _require_ready(self) -> dict:
        record = self.load_settings()
        if record is None or record.get("status") != "READY":
            raise StartError("Complete Install & prepare first.")
        if (record.get("prefix") != str(self.prefix)
                or record.get("model") != str(self.example / "model" / MODEL)
                or record.get("asset_root") != str(self.asset_root)):
            raise StartError("This workspace uses another installation. Select its saved installation folder.")
        if not self.python.is_file() or not self._example_ready():
            raise StartError("The installation or example files are missing. Run Install & prepare again.")
        return record

    def list_cards(self) -> list[str]:
        if not self._example_ready():
            return []
        return sorted(p.name for p in (self.example / "cards").iterdir())

    def _sessions_dir(self) -> Path:
        return self.workspace / "sessions"

    def _session_file(self, session_path: Path | str) -> Path:
        folder = _path(session_path)
        root = self._sessions_dir().resolve(strict=False)
        if folder.parent != root or folder.is_symlink():
            raise StartError("Choose a session from this workspace.")
        return folder / SESSION_NAME

    def _session(self, session_path: Path | str) -> dict:
        path = self._session_file(session_path)
        record = _read_json(path)
        if record.get("schema") != "compag-easy-session/v1" or record.get("path") != str(path.parent):
            raise StartError("This session belongs to another workspace or has changed.")
        return record

    def list_sessions(self) -> list[dict]:
        root = self._sessions_dir()
        if not root.is_dir():
            return []
        rows: list[dict] = []
        for folder in sorted(root.iterdir(), reverse=True):
            if folder.is_dir() and not folder.is_symlink() and (folder / SESSION_NAME).is_file():
                try:
                    rows.append(self._session(folder))
                except StartError:
                    continue
        return rows

    def create_session(self, card_dir: Path | str, use_latest: bool = False) -> dict:
        settings = self._require_ready()
        card = _path(card_dir)
        if not card.is_dir() or card.is_symlink():
            raise StartError("Choose a folder containing one original photo, or one of the example cards.")
        latest = settings.get("latest_round") if use_latest else None
        if use_latest and not isinstance(latest, dict):
            raise StartError("There is no trained next-round model yet. Choose the original model.")
        selected_model = None
        if latest:
            latest, selected_model = self._latest_model_set(settings)
            if selected_model.get("yolo_mode") != "off" and not latest.get("yolo_python"):
                raise StartError("The selected YOLO model needs its saved detector Python. Choose another model or reattach YOLO.")
        root = self._sessions_dir()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        card_name = re.sub(r"[^A-Za-z0-9_-]", "_", card.name)[:32] or "photo"
        folder = root / f"{card_name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        folder.mkdir(mode=0o700)
        record = {"schema": "compag-easy-session/v1", "version": VERSION,
                  "path": str(folder), "card_id": card.name, "card_dir": str(card),
                  "status": "PREPARING", "prepared": str(folder / "prepared"),
                  "checkpoint_dir": str(folder / "checkpoints"),
                  "inference": None, "review_state": str(folder / "review"),
                  "model_set": latest["model_set"] if latest else None,
                  "yolo_mode": selected_model["yolo_mode"] if selected_model else "off",
                  "parent_snapshot": latest["snapshot"] if latest else None,
                  "parent_model_set": latest["model_set"] if latest else None,
                  "yolo_python": latest.get("yolo_python") if latest else None,
                  "yolo_boxes": None,
                  "trained_round": None}
        _write_json(folder / SESSION_NAME, record)
        return self.resume_session(folder)

    def resume_session(self, session_path: Path | str) -> dict:
        self._require_ready()
        record = self._session(session_path)
        if record.get("status") in {"SCORED", "TRAINED"}:
            self.log("This analysis is complete. Open its review instead.")
            return record
        folder = Path(record["path"])
        prepared = Path(record["prepared"])
        if prepared.exists() and not (prepared / "PROJECT_RECEIPT.json").is_file():
            number = 2
            while (folder / f"prepared-{number:03d}").exists():
                number += 1
            prepared = folder / f"prepared-{number:03d}"
            record["prepared"] = str(prepared)
            _write_json(folder / SESSION_NAME, record)
            self.log("The earlier photo preparation was interrupted; using a new preparation folder.")
        if not prepared.exists():
            self._run(self._r92("r92", "init-images", "--images", record["card_dir"],
                                "--model", str(self.example / "model" / MODEL),
                                "--output", str(prepared)), "Preparing the selected photo")
        attempt = 1
        while (folder / f"inference-{attempt:03d}").exists():
            attempt += 1
        output = folder / f"inference-{attempt:03d}"
        record["status"] = "ANALYZING"
        record["attempt"] = attempt
        _write_json(folder / SESSION_NAME, record)
        common = ["--project", str(prepared), "--asset-root", str(self.asset_root),
                  "--checkpoint-dir", record["checkpoint_dir"], "--output", str(output)]
        try:
            if record["model_set"]:
                selected = self._model_set(record["model_set"])
                if selected.get("yolo_mode") == "off":
                    command = self._r92("r92", "infer-model-set", "--model-set", record["model_set"], *common)
                else:
                    yolo_python = _python_path(record.get("yolo_python") or "")
                    if not yolo_python.is_file():
                        raise StartError("The optional YOLO Python is missing. Restore it or choose an XGBoost-only model.")
                    checkpoint = _path(selected["yolo_checkpoint"])
                    checkpoint_sha256 = selected["yolo_checkpoint_sha256"]
                    if (selected.get("yolo_status") != "YOLO_EXTERNAL_ATTESTED"
                            or not selected.get("yolo_attestation")):
                        raise StartError("The guided workflow requires an attested external YOLO segmentation package.")
                    attestation = _read_json(_path(selected["yolo_attestation"]))
                    package = attestation.get("model_package_root")
                    if not isinstance(package, str):
                        raise StartError("The YOLO package path is absent from its attestation.")
                    boxes = _path(record["yolo_boxes"]) if record.get("yolo_boxes") else None
                    if boxes is None or not self._verify_yolo_boxes(boxes, prepared, checkpoint_sha256):
                        count = 1
                        while (folder / f"yolo-boxes-{count:03d}.json").exists():
                            count += 1
                        boxes = folder / f"yolo-boxes-{count:03d}.json"
                        self._run([str(yolo_python), "-I", "-B",
                                   str(ROOT / "tools" / "yolo_segment_backend.py"), "predict-boxes",
                                   "--project", str(prepared), "--model-package", package,
                                   "--checkpoint", str(checkpoint), "--sha256", checkpoint_sha256,
                                   "--output", str(boxes)],
                                  "Predicting YOLO boxes for every image tile")
                        if not self._verify_yolo_boxes(boxes, prepared, checkpoint_sha256):
                            raise StartError("YOLO predictions did not pass the full-tile verification.")
                        record["yolo_boxes"] = str(boxes)
                        _write_json(folder / SESSION_NAME, record)
                    else:
                        self.log("Verified saved YOLO predictions found; using them again.")
                    command = self._r92("r92", "infer-model-set", "--model-set", record["model_set"],
                                        "--yolo-boxes", str(boxes), *common)
            else:
                command = self._r92("r92", "infer-full", "--model", str(self.example / "model" / MODEL), *common)
            self._run(command, "Analyzing all image tiles (completed tiles can be reused after interruption)")
            receipt = _read_json(output / "FULL_INFERENCE_RECEIPT.json")
            if receipt.get("status") != "PASS" or not (output / "scores" / "detections.csv").is_file():
                raise StartError("Analysis ended without a complete verified result.")
            if (record.get("yolo_mode", "off") != "off"
                    and (receipt.get("yolo_mode") != record["yolo_mode"]
                         or receipt.get("yolo_execution") != "COMPLETED_ALL_TILES_CPU")):
                raise StartError("The full-card inference receipt does not confirm the selected YOLO mode.")
        except Exception:
            record["status"] = "INTERRUPTED"
            _write_json(folder / SESSION_NAME, record)
            raise
        record["status"] = "SCORED"
        record["inference"] = str(output)
        _write_json(folder / SESSION_NAME, record)
        self.log("Analysis complete. Open review to inspect and decide on candidates.")
        return record

    def review_command(self, session_path: Path | str) -> list[str]:
        self._require_ready()
        record = self._session(session_path)
        if record["status"] not in {"SCORED", "TRAINED"} or not record.get("inference"):
            raise StartError("Finish analysis before opening review.")
        return self._r92("r92", "review", "--scored", str(Path(record["inference"]) / "scores" / "detections.csv"),
                         "--tiles", str(Path(record["prepared"]) / "tiles"),
                         "--state", record["review_state"])

    def export_coco(self, session_paths: list[str], scope: str = "current") -> dict:
        if scope not in {"current", "reviewed"} or not session_paths:
            raise StartError("Choose photos and a COCO export scope.")
        if not self.python.is_file():
            raise StartError("The saved COMPAG Python installation is missing.")
        command = [str(self.python), "-u", "-I", "-B", str(ROOT / "tools/export_session_coco.py")]
        for path in session_paths:
            session = self._session(path)
            if session.get("status") not in {"SCORED", "TRAINED"}:
                raise StartError("Complete or resume this analysis before exporting its masks.")
            command += ["--session", session["path"]]
        name = "coco_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6] + ".zip"
        output = self.workspace / "coco-exports" / name
        result = self._run([*command, "--scope", scope, "--output", str(output)],
                           "Exporting masks and original photos as COCO")
        receipt = json.loads(result.strip().splitlines()[-1])
        if receipt.get("status") != "PASS" or not output.is_file() or _sha256(output) != receipt.get("sha256"):
            raise StartError("COCO export did not produce a verified archive.")
        receipt["filename"] = name
        _write_json(output.with_suffix(".json"), receipt)
        return receipt

    def list_exports(self) -> list[dict]:
        result = []
        for path in sorted((self.workspace / "coco-exports").glob("coco_*.json"), reverse=True):
            try:
                item = _read_json(path)
                name = item.get("filename", "")
                if (re.fullmatch(r"coco_\d{8}_\d{6}_[a-f0-9]{6}\.zip", name)
                        and path.with_suffix(".zip").name == name
                        and path.with_suffix(".zip").is_file()):
                    result.append({k: item[k] for k in ("filename", "image_count", "annotation_count", "scope")})
            except (StartError, KeyError):
                continue
        return result[:30]

    def train(self, session_path: Path | str, policy: str = "operational") -> dict:
        if policy not in {"operational", "paper"}:
            raise StartError("Choose operational or paper training.")
        settings = self._require_ready()
        session = self._session(session_path)
        if session["status"] == "TRAINED":
            raise StartError("This review already produced a trained round. Select another session for a new round.")
        if session["status"] != "SCORED" or not Path(session["review_state"]).is_dir():
            raise StartError("Open review and save decisions before training.")
        plan_output = self._run(self._r92("r92", "plan-review", "--state", session["review_state"],
                                         "--scope", "reviewed"),
                                "Checking the reviewed decisions for training", quiet=True)
        try:
            plan = json.loads(plan_output.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise StartError("Could not read the review plan. Open review again and check saved decisions.") from exc
        self.log(f"Training review: {plan.get('eligible_labeled_count', 0)} eligible decisions; "
                 f"{plan.get('bulk_accepted_count', 0)} bulk accepted; "
                 f"{plan.get('deleted_count', 0)} deleted.")
        if plan.get("bulk_accepted_count", 0):
            self.log("Bulk accepted labels repeat model predictions; they are not independent ground truth.")
            if policy == "paper":
                raise StartError("Paper-style training needs individually reviewed labels. This review contains bulk-accepted model predictions; they cannot serve as independent labels. The existing operational model remains available.")
        inherited_yolo = None
        inherited_python = None
        inherited_package = None
        if session.get("parent_model_set"):
            parent_set = self._model_set(session["parent_model_set"])
            if parent_set.get("yolo_mode") != "off":
                if (parent_set.get("yolo_status") != "YOLO_EXTERNAL_ATTESTED"
                        or not parent_set.get("yolo_attestation")):
                    raise StartError("This guided workflow can carry forward only the verified external segmentation YOLO model.")
                inherited_python = _python_path(session.get("yolo_python") or "")
                if not inherited_python.is_file():
                    raise StartError("The prior YOLO Python is missing. Restore it before training this next round.")
                inherited_attestation = _read_json(_path(parent_set["yolo_attestation"]))
                inherited_package = inherited_attestation.get("model_package_root")
                if not isinstance(inherited_package, str):
                    raise StartError("The prior YOLO model package path is absent from its attestation.")
                inherited_yolo = parent_set
        rounds = self.workspace / "rounds"
        rounds.mkdir(mode=0o700, parents=True, exist_ok=True)
        round_id = "round_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        output = rounds / round_id
        snapshot = output / "training_snapshot.json"
        xgb = output / "xgb_model"
        model_set = output / "model_set"
        native_model = self.example / "model" / MODEL
        if policy == "operational":
            command = self._r92("r92", "complete-xgb-round",
                                "--workspace", str(self.workspace), "--round-id", round_id,
                                "--state", session["review_state"], "--inference", session["inference"],
                                "--model", str(native_model),
                                "--scope", "reviewed", "--output", str(output))
            if session.get("parent_snapshot") and session.get("parent_model_set"):
                command += ["--parent-snapshot", session["parent_snapshot"],
                            "--parent-model-set", session["parent_model_set"]]
            result = self._run(command, "Fitting an operational XGBoost classifier from saved review decisions")
            try:
                fit_result = json.loads(result.strip().splitlines()[-1])["trained"]
            except (ValueError, IndexError, KeyError) as exc:
                raise StartError("The operational trainer returned no readable fit receipt.") from exc
        else:
            finalize = self._r92("r92", "finalize-review", "--state", session["review_state"],
                                 "--inference", session["inference"], "--model", str(native_model),
                                 "--scope", "reviewed", "--review-origin", "human",
                                 "--confirm-review-complete", "--output", str(snapshot))
            if session.get("parent_snapshot"):
                finalize += ["--parent-snapshot", session["parent_snapshot"]]
            self._run(finalize, "Sealing an immutable snapshot of reviewed labels")
            if not snapshot.is_file():
                raise StartError("Review finalization produced no training snapshot.")
            helper = ROOT / "tools" / "train_paper_xgb_round.py"
            if not helper.is_file():
                raise StartError("The paper-style XGBoost trainer is missing from this copy of COMPAG.")
            fit = [str(self.python), "-I", "-B", str(helper),
                   "--snapshot", str(snapshot), "--model", str(native_model),
                   "--output", str(xgb)]
            self._run([*fit, "--preflight"], "Checking paper training groups and held-out data")
            result = self._run(fit, "Fitting and evaluating XGBoost with the paper training protocol")
            try:
                fit_result = json.loads(result.strip().splitlines()[-1])
            except (ValueError, IndexError) as exc:
                raise StartError("The paper trainer returned no readable fit receipt.") from exc
            if fit_result.get("status") != "PASS":
                raise StartError("Paper-style XGBoost fitting did not pass verification.")
            self._run(self._r92("r92", "verify-project-model", "--bundle", str(xgb)),
                      "Verifying the fitted XGBoost model")
            activate = self._r92("r92", "activate-model-set", "--workspace", str(self.workspace),
                                 "--round-id", round_id, "--snapshot", str(snapshot),
                                 "--xgb-bundle", str(xgb), "--model", str(native_model),
                                 "--output", str(model_set))
            if session.get("parent_model_set"):
                activate += ["--parent-model-set", session["parent_model_set"]]
            self._run(activate, "Activating the verified paper-style XGBoost model")
        if not snapshot.is_file() or not (model_set / "MODEL_SET.json").is_file():
            raise StartError("Training did not produce a complete model set. See the round folder for details.")
        selected = self._model_set(model_set)
        manifest = _read_json(xgb / "PROJECT_MODEL_MANIFEST.json")
        classifier = xgb / "classifier.ubj"
        classifier_sha256 = manifest.get("classifier_sha256")
        if not classifier.is_file() or classifier.is_symlink() or _sha256(classifier) != classifier_sha256:
            raise StartError("The new XGBoost classifier failed its recorded checksum.")
        recipe = _read_json(xgb / "TRAINING_RECIPE.json")
        snapshot_record = _read_json(snapshot)
        row_count = fit_result.get("row_count") or len(snapshot_record.get("rows", []))
        boosting_rounds = (fit_result.get("boosting_rounds") or
                           fit_result.get("used_trees") or
                           recipe.get("fitted_boosting_rounds") or
                           recipe.get("selected_boosting_rounds") or
                           recipe.get("method", {}).get("used_trees"))
        settings["latest_round"] = {"round_id": round_id, "snapshot": str(snapshot),
                                    "model_set": str(model_set), "policy": policy,
                                    "row_count": row_count,
                                    "boosting_rounds": boosting_rounds,
                                    "classifier_sha256": classifier_sha256,
                                    "yolo_mode": selected["yolo_mode"],
                                    "yolo_status": selected["yolo_status"],
                                    "yolo_checkpoint_sha256": selected["yolo_checkpoint_sha256"],
                                    "source_xgb_round_id": round_id}
        _write_json(self.settings_path, settings)
        session["status"] = "TRAINED"
        session["trained_round"] = round_id
        _write_json(Path(session["path"]) / SESSION_NAME, session)
        self.log(f"XGBoost fitted: {row_count} reviewed rows, {boosting_rounds} boosting rounds, "
                 f"classifier SHA-256 {classifier_sha256}. Policy: {policy}.")
        if inherited_yolo:
            combined = output / "model_set_yolo"
            carry = self._r92("r92", "activate-model-set",
                              "--workspace", str(self.workspace), "--round-id", round_id,
                              "--snapshot", str(snapshot), "--xgb-bundle", str(xgb),
                              "--model", str(native_model), "--output", str(combined),
                              "--parent-model-set", str(model_set),
                              "--yolo-mode", inherited_yolo["yolo_mode"],
                              "--yolo-external",
                              "--yolo-checkpoint", inherited_yolo["yolo_checkpoint"],
                              "--yolo-sha256", inherited_yolo["yolo_checkpoint_sha256"],
                              "--yolo-attestation", inherited_yolo["yolo_attestation"])
            try:
                self._run(carry, "Carrying the verified YOLO detector into the new XGBoost round")
                carried = self._model_set(combined)
                if (carried.get("yolo_status") != "YOLO_EXTERNAL_ATTESTED"
                        or carried.get("yolo_mode") != inherited_yolo["yolo_mode"]
                        or carried.get("yolo_checkpoint_sha256") != inherited_yolo["yolo_checkpoint_sha256"]):
                    raise StartError("The new model set did not retain the selected YOLO detector.")
                settings["latest_round"].update({
                    "model_set": str(combined), "yolo_mode": carried["yolo_mode"],
                    "yolo_status": carried["yolo_status"],
                    "yolo_checkpoint_sha256": carried["yolo_checkpoint_sha256"],
                    "yolo_python": str(inherited_python),
                    "yolo_model_package": inherited_package,
                })
                _write_json(self.settings_path, settings)
            except Exception as exc:
                raise StartError("XGBoost was fitted and saved, but YOLO could not be carried forward. "
                                 "The latest model is temporarily XGBoost-only; reattach the verified YOLO "
                                 f"package from the browser before the next photo. Detail: {exc}") from exc
            self.log(f"YOLO {carried['yolo_mode']} retained for the next photo; "
                     f"checkpoint SHA-256 {carried['yolo_checkpoint_sha256']}.")
        self.log("New model ready. Select another card and enable Use latest trained model.")
        return settings["latest_round"]

    def attach_external_yolo(self, model_package: Path | str | None,
                             yolo_python: Path | str | None, mode: str) -> dict:
        """Select the supplied segmentation detector for later full-card inference.

        This keeps the fitted XGBoost classifier and reviewed snapshot unchanged.
        The detector is attested in its separate Ultralytics environment; only
        its verified boxes and confidences will enter the existing r92 modes.
        """
        if mode not in {"off", "prompt", "fusion", "both"}:
            raise StartError("Choose YOLO off, prompt, fusion, or both.")
        settings = self._require_ready()
        previous, selected = self._latest_model_set(settings)
        rounds = self.workspace / "rounds"
        rounds.mkdir(mode=0o700, parents=True, exist_ok=True)
        round_id = "round_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        output = rounds / round_id
        model_set = output / "model_set"
        command = self._r92("r92", "activate-model-set", "--workspace", str(self.workspace),
                            "--round-id", round_id, "--snapshot", selected["snapshot"],
                            "--xgb-bundle", selected["xgb_bundle"],
                            "--model", selected["native_model"],
                            "--parent-model-set", previous["model_set"],
                            "--output", str(model_set), "--yolo-mode", mode)
        detector_python = None
        if mode != "off":
            locations = self.yolo_locations(settings)
            if model_package is None or not str(model_package).strip():
                if locations["status"] != "FOUND":
                    raise StartError(locations["message"])
                model_package = locations["model_package"]
                self.log("Found the saved or nearby YOLO model package automatically.")
            package = _path(model_package)
            if not package.is_dir() or package.is_symlink():
                raise StartError("The YOLO segmentation model package is missing or linked.")
            manifest = _read_json(package / "MODEL_MANIFEST.json")
            checkpoint = package / "weights" / "best.pt"
            checkpoint_sha256 = manifest.get("checkpoint_sha256")
            if (not checkpoint.is_file() or checkpoint.is_symlink()
                    or not isinstance(checkpoint_sha256, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256)
                    or _sha256(checkpoint) != checkpoint_sha256):
                raise StartError("The YOLO checkpoint is missing or differs from its package manifest.")
            if yolo_python is None or not str(yolo_python).strip():
                yolo_python = locations["python"] or None
            if yolo_python is None:
                setup = ROOT / "tools" / "setup_yolo_segment_env.py"
                if not setup.is_file():
                    raise StartError("The optional YOLO environment setup helper is missing.")
                setup_result = self._run([sys.executable, "-B", str(setup),
                                          "--base-python", str(self.python),
                                          "--output", str(self.workspace / "yolo_segment_env")],
                                         "Preparing a separate verified YOLO segmentation environment")
                try:
                    env_receipt = json.loads(setup_result.strip().splitlines()[-1])
                except (IndexError, ValueError) as exc:
                    raise StartError("YOLO environment setup returned no readable result.") from exc
                if env_receipt.get("status") != "PASS" or not isinstance(env_receipt.get("python"), str):
                    raise StartError("YOLO environment setup did not pass verification.")
                detector_python = _python_path(env_receipt["python"])
            else:
                detector_python = _python_path(yolo_python)
            if not detector_python.is_file():
                raise StartError("The optional YOLO Python is missing. Choose a valid interpreter with Ultralytics 8.4.26.")
            attestation = output / "yolo_attestation.json"
            self._run([str(detector_python), "-I", "-B",
                       str(ROOT / "tools" / "yolo_segment_backend.py"), "attest",
                       "--model-package", str(package), "--checkpoint", str(checkpoint),
                       "--sha256", checkpoint_sha256, "--output", str(attestation)],
                      "Verifying the external YOLO segmentation checkpoint")
            command += ["--yolo-external", "--yolo-checkpoint", str(checkpoint),
                        "--yolo-sha256", checkpoint_sha256,
                        "--yolo-attestation", str(attestation)]
        self._run(command, "Activating the selected YOLO mode with the existing XGBoost model")
        activated = self._model_set(model_set)
        latest = {**previous, "round_id": round_id, "model_set": str(model_set),
                  "source_xgb_round_id": previous.get("source_xgb_round_id", previous["round_id"]),
                  "yolo_mode": activated["yolo_mode"], "yolo_status": activated["yolo_status"],
                  "yolo_checkpoint_sha256": activated["yolo_checkpoint_sha256"],
                  "yolo_python": str(detector_python) if detector_python else None,
                  "yolo_model_package": str(package) if mode != "off" else None}
        settings["latest_round"] = latest
        if mode != "off":
            settings["yolo_defaults"] = {"model_package": str(package),
                                         "python": str(detector_python),
                                         "checkpoint_sha256": checkpoint_sha256}
        _write_json(self.settings_path, settings)
        self.log("Model set ready for the next photo: XGBoost is unchanged; "
                 + (f"YOLO {mode} is enabled." if mode != "off" else "YOLO is off."))
        return latest


def _ask_path(label: str, default: Path) -> Path:
    response = input(f"{label} [{default}]: ").strip()
    return _path(response or default)


def _choose(label: str, choices: list[str]) -> int:
    print(f"\n{label}")
    for index, value in enumerate(choices, 1):
        print(f"  {index}. {value}")
    while True:
        answer = input("Choose a number: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return int(answer) - 1
        print("Please enter a number from the list.")


def text_wizard() -> int:
    print("\nCOMPAG — setup and review assistant")
    print("The four files included with this GitHub clone are checked automatically.")
    workspace = _ask_path("Workspace for your results", Path.home() / "COMPAG_Workspace")
    existing = Launcher(BUNDLED_ASSETS, Path.home() / ".local/share/compag-r92-rc10-gpu", workspace).load_settings()
    release = BUNDLED_ASSETS
    prefix = _path(existing["prefix"]) if existing else _ask_path("Installation folder", Path.home() / ".local/share/compag-r92-rc10-gpu")
    app = Launcher(release, prefix, workspace)
    while True:
        rows = ["Check included files", "Install and prepare examples", "Analyze a photo",
                "Continue interrupted analysis", "Open or continue review",
                "Train the next XGBoost model from saved decisions", "Exit"]
        action = _choose("What would you like to do?", rows)
        if action == 6:
            return 0
        try:
            if action == 0:
                print(json.dumps(app.check_release(), indent=2))
            elif action == 1:
                app.setup()
            elif action == 2:
                cards = app.list_cards()
                options = [f"Example {name}" for name in cards] + ["My own photo folder"]
                index = _choose("Choose a photo", options)
                card = app.example / "cards" / cards[index] if index < len(cards) else _ask_path("Your photo folder", Path.home())
                settings = app.load_settings() or {}
                latest = settings.get("latest_round")
                use_latest = bool(latest) and _choose("Which model?", ["Latest trained model", "Original example model"]) == 0
                result = app.create_session(card, use_latest)
                print("Analysis saved in:", result["path"])
            elif action == 3:
                sessions = [row for row in app.list_sessions() if row["status"] in {"PREPARING", "ANALYZING", "INTERRUPTED"}]
                if not sessions:
                    print("No interrupted analysis was found.")
                    continue
                index = _choose("Choose an analysis to continue", [f"{r['card_id']} — {r['status']} — {Path(r['path']).name}" for r in sessions])
                app.resume_session(sessions[index]["path"])
            elif action in (4, 5):
                sessions = [row for row in app.list_sessions() if row["status"] in {"SCORED", "TRAINED"}]
                if not sessions:
                    print("No completed analysis was found. Analyze a photo first.")
                    continue
                index = _choose("Choose a session", [f"{r['card_id']} — {r['status']} — {Path(r['path']).name}" for r in sessions])
                selected = sessions[index]["path"]
                if action == 4:
                    print("The browser review opens now. Press Ctrl+C here when finished; decisions are saved as you work.")
                    try:
                        subprocess.run(app.review_command(selected), cwd=ROOT, env=_clean_env(), check=True)
                    except KeyboardInterrupt:
                        print("Review stopped. Open it again to continue later.")
                else:
                    app.train(selected)
        except (StartError, OSError, subprocess.CalledProcessError) as exc:
            print(f"\nCould not finish: {exc}\n")


def main(argv: list[str] | None = None) -> int:
    if sys.version_info < (3, 10):
        print("The COMPAG start guide needs Python 3.10 or newer. Run it with a newer python3.", file=sys.stderr)
        return 1
    parser = argparse.ArgumentParser(description="Browser-guided install, analysis, review, and next-round training")
    parser.add_argument("--text", action="store_true", help="use the terminal menu")
    parser.add_argument("--check", action="store_true", help="only check the four files included with this repository")
    parser.add_argument("--release-dir", type=Path, default=BUNDLED_ASSETS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.check:
        try:
            record = Launcher(args.release_dir, Path.home() / ".local/share/compag-r92-rc10-gpu",
                              Path.home() / "COMPAG_Workspace").check_release()
        except StartError as exc:
            print(f"Included-file check failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(record, indent=2))
        return 0
    if args.text:
        return text_wizard()
    try:
        from easy_start_web import launch_web
    except ImportError as exc:
        print(f"The browser guide is missing: {exc}", file=sys.stderr)
        return 1
    try:
        return launch_web(sys.modules[__name__])
    except (OSError, StartError) as exc:
        print(f"Could not open the local browser guide: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
