#!/usr/bin/env python3
"""Local browser front end for the unchanged COMPAG easy-start workflow.

Only the standard library is used. The application, installer, inference,
reviewer, and training commands are supplied by :mod:`start_compag`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
import tempfile
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


MAX_JSON = 16 * 1024
MAX_PHOTO = 50 * 1024 * 1024


class _Bridge:
    def __init__(self, backend):
        self.backend = backend
        self.lock = threading.RLock()
        self.release_dir = backend.BUNDLED_ASSETS
        self.prefix = Path.home() / ".local/share/compag-r92-rc10-gpu"
        self.workspace = Path.home() / "COMPAG_Workspace"
        self.pointer_path = Path.home() / ".local/share/compag-r92-easy-start/rc10-r54/LAST_WORKSPACE.json"
        self.job = {"status": "IDLE", "action": "", "logs": [], "error": None, "result": None, "id": 0}
        self.selected_session = None
        self.uploaded_card = None
        self.review_process = None
        self.review_session = None
        self.review_url = None
        self.review_error = None
        self.review_stopping = False
        self.review_plan = None
        self.plan_session = None
        self.cards = []
        self.job_control = None
        self.job_thread = None
        try:
            if self.pointer_path.is_file() and not self.pointer_path.is_symlink():
                pointer = backend._read_json(self.pointer_path)
                if (pointer.get("schema") == "compag-easy-start-last-workspace/v1"
                        and pointer.get("version") == backend.VERSION
                        and isinstance(pointer.get("workspace"), str)):
                    self.workspace = Path(pointer["workspace"]).expanduser().resolve(strict=False)
        except (backend.StartError, OSError, ValueError):
            self.workspace = Path.home() / "COMPAG_Workspace"
        try:
            record = self._app().load_settings()
            if record:
                self.prefix = Path(record["prefix"])
                self.cards = self._app().list_cards()
            else:
                self.workspace = Path.home() / "COMPAG_Workspace"
        except (backend.StartError, OSError, KeyError):
            self.workspace = Path.home() / "COMPAG_Workspace"

    def _app(self, log=None):
        return self.backend.Launcher(self.release_dir, self.prefix, self.workspace, log=log)

    def _log(self, message):
        line = str(message).strip()
        with self.lock:
            self.job["logs"].append(line[:1500])
            self.job["logs"] = self.job["logs"][-120:]

    def _busy(self):
        return self.job["status"] in {"RUNNING", "STOPPING", "CLEANUP_FAILED"}

    def configure(self, payload):
        with self.lock:
            if self._busy() or self.review_process is not None:
                raise self.backend.StartError("Finish the current operation or review before changing folders.")
            if set(payload) != {"prefix", "workspace"}:
                raise self.backend.StartError("Please choose the two folders shown on this page.")
            values = {}
            for key, value in payload.items():
                if not isinstance(value, str) or not value.strip() or len(value) > 4096 or "\x00" in value:
                    raise self.backend.StartError("A folder path is missing or too long.")
                values[key] = Path(value).expanduser().resolve(strict=False)
            self.prefix = values["prefix"]
            self.workspace = values["workspace"]
            self.cards = []
            try:
                self.cards = self._app().list_cards()
            except (self.backend.StartError, OSError):
                pass
        return {"status": "SAVED"}

    def start_job(self, action, fn):
        with self.lock:
            if self._busy():
                raise self.backend.StartError("A step is already running. Wait for it to finish.")
            if self.review_process is not None:
                raise self.backend.StartError("Finish the open review before starting another step.")
            app = self._app(self._log)
            self.job_control = app.control
            job_id = self.job["id"] + 1
            self.job = {"status": "RUNNING", "action": action, "logs": [], "error": None,
                        "result": None, "id": job_id}

        def work():
            try:
                result = fn(app)
                app.control.settle()
                with self.lock:
                    if action == "setup":
                        self.cards = app.list_cards()
                        try:
                            self.pointer_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                            self.backend._write_json(self.pointer_path, {
                                "schema": "compag-easy-start-last-workspace/v1",
                                "version": self.backend.VERSION,
                                "workspace": str(self.workspace)})
                        except (self.backend.StartError, OSError) as exc:
                            self._log(f"Could not remember the last workspace: {exc}")
                    if action in {"infer", "resume"}:
                        self.selected_session = result["path"]
                    if action == "plan":
                        self.review_plan = self._small_result(action, result)
                        self.plan_session = self.selected_session
                    self.job["result"] = self._small_result(action, result)
                    self.job["status"] = "DONE"
            except self.backend.JobCleanupError as exc:
                with self.lock:
                    self.job["status"] = "CLEANUP_FAILED"
                    self.job["error"] = str(exc)[:2000]
            except self.backend.JobStopped as exc:
                try:
                    app.control.settle(failed=True)
                except self.backend.JobCleanupError as cleanup_exc:
                    with self.lock:
                        self.job["status"] = "CLEANUP_FAILED"
                        self.job["error"] = str(cleanup_exc)[:2000]
                    return
                with self.lock:
                    self.job["status"] = "STOPPED"
                    self.job["error"] = None
                    self._log(str(exc))
            except Exception as exc:
                try:
                    app.control.settle(failed=True)
                except self.backend.JobCleanupError as cleanup_exc:
                    with self.lock:
                        self.job["status"] = "CLEANUP_FAILED"
                        self.job["error"] = str(cleanup_exc)[:2000]
                    return
                with self.lock:
                    if isinstance(exc, (self.backend.StartError, OSError, ValueError)):
                        self.job["error"] = str(exc)[:2000]
                    else:
                        self.job["error"] = f"{type(exc).__name__}: this step did not finish. Check the details below."
                    self.job["status"] = "ERROR"

        self.job_thread = threading.Thread(target=work, name=f"compag-{action}", daemon=True)
        self.job_thread.start()
        return {"status": "STARTED", "job_id": job_id}

    def stop_job(self, job_id=None):
        with self.lock:
            if job_id is not None and (type(job_id) is not int or job_id != self.job["id"]):
                raise self.backend.StartError("This Stop request is for an older job. Refresh the page.")
            if self.job["status"] == "CLEANUP_FAILED":
                return {"status": "CLEANUP_FAILED"}
            if not self._busy() or self.job_control is None:
                return {"status": "IDLE"}
            if self.job["action"] not in {"infer", "resume", "train", "export"}:
                raise self.backend.StartError("Wait for the current setup or check to finish.")
            if self.job["status"] != "STOPPING":
                self.job["status"] = "STOPPING"
                self._log("Stopping the running operation; keeping completed tiles and saved decisions…")
                self.job_control.cancel()
        return {"status": "STOPPING"}

    def export_coco(self, payload):
        scope = payload.get("scope", "current")
        if scope not in {"current", "reviewed"}:
            raise self.backend.StartError("Choose the COCO export scope.")
        if payload.get("session_id") == "__all__":
            paths = [s["path"] for s in self._app().list_sessions() if s.get("status") in {"SCORED", "TRAINED"}]
        else:
            paths = [self._session_path(payload.get("session_id"))]
        return self.start_job("export", lambda app: app.export_coco(paths, scope))

    @staticmethod
    def _small_result(action, result):
        if not isinstance(result, dict):
            return {"status": "DONE"}
        if action in {"infer", "resume"}:
            return {"session": Path(result["path"]).name, "status": result.get("status")}
        if action == "train":
            return {key: result.get(key) for key in
                    ("round_id", "policy", "row_count", "boosting_rounds",
                     "classifier_sha256", "yolo_mode", "yolo_status") if key in result}
        if action == "attach_yolo":
            return {key: result.get(key) for key in
                    ("round_id", "yolo_mode", "yolo_status", "yolo_checkpoint_sha256")
                    if key in result}
        if action == "plan":
            return {key: result.get(key, 0) for key in
                    ("candidate_count", "eligible_labeled_count", "skipped_count",
                     "deleted_count", "bulk_accepted_count", "unreviewed_count",
                     "source_card_count", "individually_decided_count",
                     "parent_bulk_accepted_count", "paper_ready", "paper_reason")}
        if action == "export":
            return {key: result.get(key) for key in ("filename", "image_count", "annotation_count", "scope")}
        return {key: result.get(key) for key in ("status", "version", "file_count", "folder") if key in result}

    def _session_path(self, session_id):
        if not isinstance(session_id, str) or len(session_id) > 120:
            raise self.backend.StartError("Choose a saved analysis from the list.")
        for row in self._app().list_sessions():
            if Path(row["path"]).name == session_id:
                return row["path"]
        raise self.backend.StartError("The selected analysis was not found in this workspace.")

    def infer(self, payload):
        mode = payload.get("mode")
        use_latest = payload.get("use_latest")
        if use_latest is not True and use_latest is not False:
            raise self.backend.StartError("Choose which model to use.")
        with self.lock:
            if mode == "example":
                card_id = payload.get("card_id")
                if not isinstance(card_id, str) or card_id not in self.cards:
                    raise self.backend.StartError("Choose one of the prepared example photos.")
                card = self._app().example / "cards" / card_id
            elif mode == "own":
                if self.uploaded_card is None:
                    raise self.backend.StartError("Select and upload your photo first.")
                card = self.uploaded_card
            else:
                raise self.backend.StartError("Choose an example or your own photo.")
        return self.start_job("infer", lambda app: app.create_session(card, use_latest))

    def resume(self, payload):
        path = self._session_path(payload.get("session_id"))
        return self.start_job("resume", lambda app: app.resume_session(path))

    def train(self, payload):
        path = self._session_path(payload.get("session_id"))
        policy = payload.get("policy")
        if policy not in {"operational", "paper"}:
            raise self.backend.StartError("Choose Operational or Paper protocol XGBoost training.")
        with self.lock:
            if path != self.plan_session or not self.review_plan:
                raise self.backend.StartError("Check the saved review decisions before training.")
            if self.review_plan.get("eligible_labeled_count", 0) < 1:
                raise self.backend.StartError("No labeled decisions are ready for training. Continue the review first.")
            if policy == "paper" and not self.review_plan.get("paper_ready"):
                raise self.backend.StartError(self.review_plan.get("paper_reason") or "Check paper training readiness first.")
        return self.start_job("train", lambda app: app.train(path, policy=policy))

    def attach_yolo(self, payload):
        if set(payload) != {"model_package", "yolo_python", "mode"}:
            raise self.backend.StartError("Choose the YOLO model package and how it will support inference.")
        package = payload["model_package"]
        python = payload["yolo_python"]
        mode = payload["mode"]
        if (not isinstance(package, str) or len(package) > 4096
                or "\x00" in package or not isinstance(python, str) or len(python) > 4096
                or "\x00" in python or mode not in {"off", "prompt", "fusion", "both"}):
            raise self.backend.StartError("Enter a local YOLO model location and a valid inference mode.")
        return self.start_job("attach_yolo", lambda app: app.attach_external_yolo(
            Path(package).expanduser() if package.strip() else None,
            Path(python).expanduser() if python.strip() else None,
            mode))

    def _latest_training_evidence(self, latest_round):
        """Summarize immutable on-disk fit evidence, including pre-R5 workspaces."""
        if not isinstance(latest_round, dict):
            return None
        try:
            active_round_id = latest_round["round_id"]
            round_id = latest_round.get("source_xgb_round_id", active_round_id)
            if not isinstance(round_id, str) or not round_id.startswith("round_") or Path(round_id).name != round_id:
                raise ValueError("Invalid round identity")
            root = (self.workspace / "rounds" / round_id).resolve(strict=True)
            if not root.is_relative_to(self.workspace.resolve(strict=True)):
                raise ValueError("Round is outside this private workspace")
            model = root / "xgb_model"
            snapshot = json.loads((root / "training_snapshot.json").read_text(encoding="utf-8"))
            recipe = json.loads((model / "TRAINING_RECIPE.json").read_text(encoding="utf-8"))
            manifest = json.loads((model / "PROJECT_MODEL_MANIFEST.json").read_text(encoding="utf-8"))
            model_set_dir = Path(latest_round["model_set"]).resolve(strict=True)
            if not model_set_dir.is_relative_to(self.workspace.resolve(strict=True)):
                raise ValueError("Model set is outside this private workspace")
            model_set_file = model_set_dir / "MODEL_SET.json"
            model_set = json.loads(model_set_file.read_text(encoding="utf-8"))
            pointer = json.loads((self.workspace / "LATEST_MODEL_SET.json").read_text(encoding="utf-8"))
            model_set_sha = self.backend._sha256(model_set_file)
            classifier = model / "classifier.ubj"
            actual_sha = self.backend._sha256(classifier)
            files = manifest.get("files")
            if not isinstance(files, dict) or not files:
                raise ValueError("Saved model manifest is incomplete")
            for name, expected in files.items():
                if (not isinstance(name, str) or Path(name).name != name
                        or not isinstance(expected, str)):
                    raise ValueError("Saved model manifest has an invalid file entry")
                member = model / name
                if not member.is_file() or member.is_symlink():
                    raise ValueError("Saved model file is missing or linked")
                if self.backend._sha256(member) != expected:
                    raise ValueError("Saved model file checksum differs from its manifest")
            snapshot_sha = self.backend._sha256(root / "training_snapshot.json")
            if (manifest.get("status") != "PASS" or manifest.get("classifier_sha256") != actual_sha
                    or manifest.get("snapshot_sha256") != snapshot_sha
                    or recipe.get("snapshot_sha256") != snapshot_sha
                    or model_set.get("status") != "MODEL_SET_VERIFIED"
                    or Path(pointer.get("path", "")).resolve(strict=True) != model_set_dir
                    or pointer.get("model_set_sha256") != model_set_sha
                    or Path(model_set.get("xgb_bundle", "")).resolve(strict=True) != model.resolve(strict=True)):
                raise ValueError("Saved model evidence did not match the model files")
            rows = snapshot.get("rows")
            if not isinstance(rows, list) or snapshot.get("status") != "PASS":
                raise ValueError("Saved training snapshot is incomplete")
            params = recipe.get("parameters", {})
            method = recipe.get("method", {})
            paper = recipe.get("policy") == "PAPER_SETTINGS_NEW_DATA_GROUPED"
            evidence = {
                "status": "VERIFIED", "round_id": round_id,
                "active_round_id": active_round_id, "row_count": len(rows),
                "group_count": len({row.get("group_id") for row in rows}),
                "bulk_accepted_count": sum(row.get("action") == "bulk_accept" for row in rows),
                "policy": recipe.get("policy"),
                "boosting_rounds": recipe.get("fitted_boosting_rounds", method.get("used_trees")),
                "classifier_sha256": actual_sha, "model_folder": str(model),
                "group_cv_performed": (model / "SEARCH_RESULT.json").is_file() if paper else recipe.get("group_cv_performed"),
                "heldout_test_performed": (model / "HELDOUT_RESULT.json").is_file() if paper else recipe.get("heldout_test_performed"),
                "yolo_mode": model_set.get("yolo_mode", "off"),
                "yolo_status": model_set.get("yolo_status", "YOLO_NOT_REQUESTED"),
                "yolo_checkpoint_sha256": model_set.get("yolo_checkpoint_sha256"),
                "estimators_requested": params.get("n_estimators"),
            }
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            evidence = {"status": "UNVERIFIED", "reason": str(exc)[:300]}
        return evidence

    def plan(self, payload):
        path = self._session_path(payload.get("session_id"))
        with self.lock:
            self.review_plan = None
            self.plan_session = None
            self.selected_session = path

        def read_plan(app):
            record = app._session(path)
            if record.get("status") != "SCORED" or not Path(record["review_state"]).is_dir():
                raise self.backend.StartError("Open review and save decisions before checking training.")
            output = app._run(app._r92("r92", "plan-review", "--state", record["review_state"],
                                      "--scope", "reviewed"),
                              "Checking saved review decisions", quiet=True)
            try:
                plan = json.loads(output.strip().splitlines()[-1])
                project = self.backend._read_json(Path(record["prepared"]) / "PROJECT_RECEIPT.json")
                tile_groups = {t["name"]: source["original_sha256"]
                               for source in project["images"] for t in source["tiles"]}
                groups = set()
                for key in plan.get("included_keys", []):
                    if isinstance(key, list) and key and isinstance(key[0], str):
                        groups.add(tile_groups[key[0]])
                parent = record.get("parent_snapshot")
                parent_bulk_accepted_count = 0
                if parent:
                    parent_path = Path(parent).resolve(strict=True)
                    if not parent_path.is_relative_to(self.workspace.resolve(strict=True)):
                        raise self.backend.StartError("Previous training data is outside this workspace.")
                    parent_rows = json.loads(parent_path.read_text(encoding="utf-8")).get("rows", [])
                    if not isinstance(parent_rows, list) or any(
                            not isinstance(row, dict) or not isinstance(row.get("group_id"), str)
                            for row in parent_rows):
                        raise ValueError("Previous training rows are incomplete")
                    groups.update(row["original_sha256"] for row in parent_rows)
                    parent_bulk_accepted_count = sum(row.get("action") == "bulk_accept"
                                                     for row in parent_rows)
                plan["source_card_count"] = len(groups)
                plan["individually_decided_count"] = max(0, plan["eligible_labeled_count"]
                                                         - plan["bulk_accepted_count"])
                plan["parent_bulk_accepted_count"] = parent_bulk_accepted_count
                reasons = []
                bulk = plan["bulk_accepted_count"] + parent_bulk_accepted_count
                if bulk:
                    reasons.append(f"{bulk} labels are bulk-accepted model predictions "
                                   f"({plan['bulk_accepted_count']} in this review; {parent_bulk_accepted_count} inherited). "
                                   "The current paper trainer requires individual review actions.")
                if len(groups) < 6:
                    reasons.append(f"Only {len(groups)} independent photos are available. This implementation needs "
                                   "at least 6 for five training folds and a separate held-out photo; "
                                   "the actual split and both-class checks may require more.")
                if not plan["eligible_labeled_count"]:
                    reasons.append("No saved class labels in this review.")
                plan["paper_ready"] = False
                if reasons:
                    plan["paper_reason"] = " ".join(reasons)
                else:
                    command = [str(app.python), "-I", "-B", str(self.backend.ROOT / "tools/check_paper_training.py"),
                               "--state", record["review_state"], "--inference", record["inference"],
                               "--model", str(app.example / "model" / self.backend.MODEL)]
                    if parent:
                        command += ["--parent-snapshot", parent]
                    checked = app._run(command, "Checking the unchanged paper training protocol", quiet=True)
                    plan.update(json.loads(checked.strip().splitlines()[-1]))
                return plan
            except (ValueError, IndexError) as exc:
                raise self.backend.StartError("Could not read review decisions. Open review again.") from exc

        return self.start_job("plan", read_plan)

    def _read_review(self, process):
        served = False
        tail = []
        try:
            assert process.stdout is not None
            for raw in process.stdout:
                line = raw.strip()
                try:
                    value = json.loads(line)
                except ValueError:
                    value = None
                with self.lock:
                    if self.review_process is not process:
                        continue
                    if isinstance(value, dict) and value.get("status") == "SERVING":
                        url = value.get("url")
                        if isinstance(url, str):
                            parsed = urlsplit(url)
                            try:
                                valid_port = parsed.port is not None and 0 < parsed.port < 65536
                            except ValueError:
                                valid_port = False
                            if (parsed.scheme == "http" and parsed.hostname == "127.0.0.1"
                                    and valid_port and parsed.username is None and parsed.password is None
                                    and parsed.path == "/" and parsed.query.startswith("token=")):
                                self.review_url = url
                                served = True
                    elif line:
                        tail.append(line[:500])
                        tail = tail[-3:]
                        self.job["logs"].append(line[:1000])
                        self.job["logs"] = self.job["logs"][-120:]
        finally:
            code = process.wait()
            with self.lock:
                if self.review_process is process:
                    if self.review_stopping:
                        self.review_error = None
                    elif not served:
                        detail = " ".join(tail)[-700:]
                        self.review_error = ("The review page could not start. " + detail +
                                             " Check the saved analysis, then reopen review.") if detail else (
                                             "The review page could not start. Check the saved analysis, then reopen review.")
                    elif code:
                        self.review_error = "The review page closed unexpectedly. Reopen review to continue."
                    self.review_process = None
                    self.review_url = None

    def start_review(self, payload):
        path = self._session_path(payload.get("session_id"))
        with self.lock:
            if self._busy():
                raise self.backend.StartError("Wait until the current step finishes.")
            if self.review_process is not None:
                if self.review_session == path:
                    return {"status": "ALREADY_OPEN"}
                raise self.backend.StartError("Finish the current review before opening another one.")
            app = self._app()
            command = app.review_command(path) + ["--no-open-browser"]
            try:
                process = subprocess.Popen(command, cwd=self.backend.ROOT,
                                           env=self.backend._clean_env(), stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1, start_new_session=True)
            except OSError as exc:
                raise self.backend.StartError(f"Could not open review: {exc}") from exc
            self.review_process = process
            self.review_session = path
            self.review_url = None
            self.review_error = None
            self.review_stopping = False
            self.review_plan = None
            self.plan_session = None
        threading.Thread(target=self._read_review, args=(process,), name="compag-review-log", daemon=True).start()
        return {"status": "STARTING"}

    def stop_review(self):
        with self.lock:
            process = self.review_process
            if process is None:
                return {"status": "CLOSED"}
            self.review_stopping = True
            process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        with self.lock:
            if self.review_process is process:
                self.review_process = None
                self.review_url = None
            self.review_error = None
            self.review_stopping = False
        return {"status": "CLOSED"}

    def _copy_body(self, stream, destination, length, limit):
        if length <= 0 or length > limit:
            raise self.backend.StartError("The selected file is too large or empty.")
        digest = hashlib.sha256()
        head = b""
        remaining = length
        with tempfile.NamedTemporaryFile("wb", dir=destination.parent,
                                         prefix=".upload-", delete=False) as output:
            temp = Path(output.name)
            try:
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise self.backend.StartError("The upload ended before the file was complete.")
                    if len(head) < 16:
                        head = (head + chunk)[:16]
                    output.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
                output.flush()
                os.fsync(output.fileno())
            except Exception:
                temp.unlink(missing_ok=True)
                raise
        return temp, digest.hexdigest(), head

    def upload_photo(self, name, stream, length):
        if not isinstance(name, str) or name != Path(name).name or not name or len(name) > 180:
            raise self.backend.StartError("Choose a photo with a regular file name.")
        suffix = Path(name).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            raise self.backend.StartError("Choose a JPG or PNG photo.")
        with self.lock:
            if self._busy() or self.review_process is not None:
                raise self.backend.StartError("Finish the current step before selecting a photo.")
            self._app()._require_ready()
            parent = self.workspace / "user_photos"
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            photo_id = uuid.uuid4().hex[:12]
            folder = parent / ("photo-" + photo_id)
            folder.mkdir(mode=0o700)
            # The project snapshot groups rows by the source photo filename.
            # Keep that name unique across uploads so separate cards remain
            # separate groups during group-safe paper training.
            safe_name = "photo_" + photo_id + (".jpg" if suffix == ".jpeg" else suffix)
            destination = folder / safe_name
            try:
                temp, _, head = self._copy_body(stream, destination, length, MAX_PHOTO)
                try:
                    is_jpg = suffix in {".jpg", ".jpeg"} and head.startswith(b"\xff\xd8\xff")
                    is_png = suffix == ".png" and head.startswith(b"\x89PNG\r\n\x1a\n")
                    if not (is_jpg or is_png):
                        raise self.backend.StartError("This file is not a valid JPG or PNG photo.")
                    os.replace(temp, destination)
                finally:
                    temp.unlink(missing_ok=True)
            except Exception:
                folder.rmdir()
                raise
            self.uploaded_card = folder
        return {"status": "READY", "file": name}

    def snapshot(self):
        with self.lock:
            if self.job_control is not None and self.job_control.cleanup_error and self._busy():
                self.job["status"] = "CLEANUP_FAILED"
                self.job["error"] = self.job_control.cleanup_error
            app = self._app()
            try:
                settings = app.load_settings()
            except (self.backend.StartError, OSError):
                settings = None
            try:
                sessions = app.list_sessions()
            except (self.backend.StartError, OSError):
                sessions = []
            compact = [{"id": Path(row["path"]).name, "card": row.get("card_id"),
                        "status": row.get("status"), "trained_round": row.get("trained_round"),
                        "yolo_mode": row.get("yolo_mode", "off")}
                       for row in sessions[:100]]
            latest_round = settings.get("latest_round") if settings else None
            return {"paths": {"prefix": str(self.prefix), "workspace": str(self.workspace)},
                    "ready": bool(settings and settings.get("status") == "READY"),
                    "cards": list(self.cards), "sessions": compact,
                    "latest_round": latest_round,
                    "exports": app.list_exports(),
                    "yolo_locations": app.yolo_locations(settings or {}),
                    "training_evidence": self._latest_training_evidence(latest_round),
                    "job": {"status": self.job["status"], "action": self.job["action"],
                            "logs": self.job["logs"][-80:], "error": self.job["error"],
                            "result": self.job["result"], "id": self.job["id"]},
                    "selected_session": Path(self.selected_session).name if self.selected_session else None,
                    "review_plan": self.review_plan,
                    "plan_session": Path(self.plan_session).name if self.plan_session else None,
                    "uploaded_photo": self.uploaded_card.name if self.uploaded_card else None,
                    "review": {"active": self.review_process is not None,
                               "session": Path(self.review_session).name if self.review_session else None,
                               "url": self.review_url, "error": self.review_error}}

    def close(self):
        if self.job_control is not None and self._busy():
            self.job_control.cancel()
            if self.job_thread is not None:
                self.job_thread.join(timeout=12)
        try:
            self.stop_review()
        except (OSError, subprocess.TimeoutExpired):
            pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, backend):
        self.token = secrets.token_urlsafe(32)
        self.csrf = secrets.token_urlsafe(32)
        self.bridge = _Bridge(backend)
        super().__init__(("127.0.0.1", 0), _Handler)


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, _format, *_args):
        return

    def _base(self):
        return "/t/" + self.server.token

    def _host_ok(self):
        return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        if content_type.startswith("text/html"):
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, value):
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self):
        if not self._host_ok():
            return self._json(403, {"error": "Access denied."})
        path = urlsplit(self.path).path
        base = self._base()
        if path == base + "/":
            page = HTML.replace("{{CSRF}}", self.server.csrf).replace("{{BASE}}", base)
            return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        if path == base + "/app.css":
            return self._send(200, CSS.encode("utf-8"), "text/css; charset=utf-8")
        if path == base + "/app.js":
            return self._send(200, JS.encode("utf-8"), "text/javascript; charset=utf-8")
        if path == base + "/api/state":
            return self._json(200, self.server.bridge.snapshot())
        if path.startswith(base + "/downloads/"):
            name = path.removeprefix(base + "/downloads/")
            if not re.fullmatch(r"coco_\d{8}_\d{6}_[a-f0-9]{6}\.zip", name):
                return self._json(404, {"error": "Export not found."})
            root = self.server.bridge.workspace / "coco-exports"
            file = root / name
            if (file.is_symlink() or not file.is_file()
                    or file.resolve().parent != root.resolve()
                    or name not in {r['filename'] for r in self.server.bridge._app().list_exports()}):
                return self._json(404, {"error": "Export not found."})
            with file.open("rb") as stream:
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                self.send_header("Content-Length", str(os.fstat(stream.fileno()).st_size))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                for block in iter(lambda: stream.read(1024*1024), b""):
                    self.wfile.write(block)
            return
        return self._json(404, {"error": "Page not found."})

    def _length(self, limit):
        raw = self.headers.get("Content-Length", "")
        if not raw.isascii() or not raw.isdecimal():
            raise ValueError("Missing or invalid upload size.")
        length = int(raw)
        if length < 1 or length > limit:
            raise ValueError("The selected file or request is too large or empty.")
        return length

    def _body(self):
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            raise ValueError("This action needs a browser JSON request.")
        data = json.loads(self.rfile.read(self._length(MAX_JSON)))
        if not isinstance(data, dict):
            raise ValueError("Invalid request.")
        return data

    def do_POST(self):
        origin = f"http://127.0.0.1:{self.server.server_port}"
        if (not self._host_ok() or self.headers.get("Origin") != origin
                or not secrets.compare_digest(self.headers.get("X-COMPAG-CSRF", ""), self.server.csrf)):
            return self._json(403, {"error": "Access denied."})
        path = urlsplit(self.path).path
        if not path.startswith(self._base() + "/api/"):
            return self._json(404, {"error": "Action not found."})
        action = path.removeprefix(self._base() + "/api/")
        bridge = self.server.bridge
        try:
            if action == "config":
                result = bridge.configure(self._body())
            elif action == "check":
                self._body()
                result = bridge.start_job("check", lambda app: app.check_release())
            elif action == "setup":
                self._body()
                result = bridge.start_job("setup", lambda app: app.setup())
            elif action == "infer":
                result = bridge.infer(self._body())
            elif action == "resume":
                result = bridge.resume(self._body())
            elif action == "review":
                result = bridge.start_review(self._body())
            elif action == "review-stop":
                self._body()
                result = bridge.stop_review()
            elif action == "train":
                result = bridge.train(self._body())
            elif action == "stop-job":
                payload = self._body()
                if bridge._busy() and "job_id" not in payload:
                    raise bridge.backend.StartError("Refresh the page before stopping the current job.")
                result = bridge.stop_job(payload.get("job_id"))
            elif action == "export":
                result = bridge.export_coco(self._body())
            elif action == "attach-yolo":
                result = bridge.attach_yolo(self._body())
            elif action == "plan":
                result = bridge.plan(self._body())
            elif action == "upload-photo":
                name = self.headers.get("X-File-Name", "")
                result = bridge.upload_photo(name, self.rfile, self._length(MAX_PHOTO))
            else:
                return self._json(404, {"error": "Action not found."})
            return self._json(200, result)
        except (bridge.backend.StartError, ValueError, OSError, UnicodeError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": str(exc)[:2000]})
        except Exception:
            return self._json(500, {"error": "This step could not finish. See the terminal for details."})


def launch_web(backend_module):
    """Open the guided local website; return when the user closes the launcher."""
    server = _Server(backend_module)
    url = f"http://127.0.0.1:{server.server_port}/t/{server.token}/"
    print("COMPAG guide is ready in your browser. Keep this terminal open.", flush=True)
    print(f"If the page did not open, visit this private local address:\n{url}", flush=True)
    try:
        webbrowser.open(url)
    except (OSError, webbrowser.Error):
        pass
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        server.bridge.close()
        server.server_close()
    return 0


HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="csrf-token" content="{{CSRF}}"><title>COMPAG setup and review</title><link rel="stylesheet" href="{{BASE}}/app.css"></head><body><div class="shell"><header class="top"><div class="brand"><span class="mark">C</span><div><strong>COMPAG</strong><small>Guided image review</small></div></div><div class="private">Runs on this computer</div></header><main><div class="intro"><div class="eyebrow">START HERE</div><h1>Start COMPAG</h1><p>Follow the steps below. Your photos and review decisions stay in your private work folder.</p></div><nav class="steps" aria-label="Steps"><button data-step="1"><b>1</b><span>Files</span></button><button data-step="2"><b>2</b><span>Install</span></button><button data-step="3"><b>3</b><span>Analyze</span></button><button data-step="4"><b>4</b><span>Review</span></button><button data-step="5"><b>5</b><span>Train</span></button><button data-step="6"><b>6</b><span>Next photo</span></button></nav><div id="notice" class="notice" role="status" aria-live="polite"></div><section id="step1" class="panel"><div class="number">STEP 1 OF 6</div><h2>Check the included files</h2><p>The four files needed to begin are already included with COMPAG. Their checksums are checked automatically when this page opens.</p><button id="checkRelease" class="secondary">Check included files again</button><p class="hint">There is nothing to download or select for this step.</p></section><section id="step2" class="panel" hidden><div class="number">STEP 2 OF 6</div><h2>Install and prepare</h2><p>Choose where COMPAG and your private work will live. Installation can take a while; this page shows progress and can be reopened later.</p><div class="field"><label for="prefixDir">Program installation folder</label><input id="prefixDir" type="text" autocomplete="off" spellcheck="false"></div><div class="field"><label for="workspaceDir">Private work folder</label><input id="workspaceDir" type="text" autocomplete="off" spellcheck="false"></div><button id="setup" class="primary">Install and prepare</button><p class="hint">Linux x86-64 and Conda are required for the complete GPU workflow. Official model files are downloaded and verified here.</p></section><section id="step3" class="panel" hidden><div class="number">STEP 3 OF 6</div><h2>Analyze a photo</h2><p>Start with one of the included examples or use your own JPG or PNG. The analysis checks every image tile and saves progress.</p><div class="choice"><label><input type="radio" name="cardMode" value="example" checked> Example photo</label><label><input type="radio" name="cardMode" value="own"> My photo</label></div><div id="examplePicker" class="field"><label for="cardSelect">Example</label><select id="cardSelect"></select></div><div id="ownPicker" class="field" hidden><label for="photoFile">Select one JPG or PNG</label><input id="photoFile" type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png"><small>The selected photo is copied only to your private work folder.</small><button id="uploadPhoto" class="secondary">Add this photo</button><div id="photoStatus" class="hint"></div></div><label id="latestChoice" class="check"><input id="useLatest" type="checkbox"> Use the latest model trained from my previous review</label><button id="infer" class="primary">Analyze photo</button><div class="divider">Continue previous work</div><div class="field"><label for="resumeSelect">Unfinished analysis</label><select id="resumeSelect"></select></div><button id="resume" class="secondary">Continue analysis</button></section><section id="step4" class="panel" hidden><div class="number">STEP 4 OF 6</div><h2>Review every candidate</h2><p>Open the existing COMPAG review page in another browser tab. Decisions are saved as you make them. You can return here at any time.</p><div class="field"><label for="reviewSelect">Analysis to review</label><select id="reviewSelect"></select></div><button id="startReview" class="primary">Open review page</button><a id="reviewLink" class="review-link" target="_blank" rel="noopener noreferrer" hidden>Continue review in browser ↗</a><p id="reviewStatus" class="hint"></p><button id="stopReview" class="secondary" hidden>Finish review and continue</button><p class="hint">Complete your decisions in the review tab before finishing here. You can reopen review later.</p></section><section id="step5" class="panel" hidden><div class="number">STEP 5 OF 6</div><h2>Train the next XGBoost model</h2><p>After you save the review decisions, start training here. This uses the same reviewed-round command as the advanced workflow.</p><div class="field"><label for="trainSelect">Reviewed analysis</label><select id="trainSelect"></select></div><button id="train" class="primary">Train next model</button><p class="hint">If you accepted all remaining predictions at once, those labels repeat the model's predictions; they are not independent human ground truth.</p></section><section id="step6" class="panel" hidden><div class="number">STEP 6 OF 6</div><h2>Your next round is ready</h2><p id="roundInfo">The trained model is ready for another photo.</p><button id="nextPhoto" class="primary">Analyze another photo</button><p class="hint">The latest trained model will be selected by default. You can change that choice before analysis.</p></section><aside class="progress"><div class="progress-head"><strong id="progressTitle">Ready to begin</strong><span id="progressSpin"></span></div><div id="progressMessage" class="progress-message">Checking the files included with COMPAG…</div><details><summary>Show step details</summary><pre id="log"></pre></details></aside></main><footer>COMPAG R92 · Local guided workflow</footer></div><script src="{{BASE}}/app.js"></script></body></html>"""

# The counts are read from the existing, read-only r92 review plan command.
HTML = HTML.replace('<button id="train" class="primary">',
                    '<button id="checkPlan" class="secondary">Check saved decisions</button>'
                    '<div id="planSummary" class="plan-summary"></div>'
                    '<button id="train" class="primary">')
HTML = HTML.replace(
    'After you save the review decisions, start training here. This uses the same reviewed-round command as the advanced workflow.',
    'Choose how to fit XGBoost from the saved decisions. The training record will show what actually ran.')
HTML = HTML.replace(
    '<button id="train" class="primary">Train next model</button>',
    '<div class="field"><label for="trainPolicy">XGBoost training method</label>'
    '<select id="trainPolicy">'
    '<option value="operational">Operational: fit all reviewed rows, no holdout</option>'
    '<option value="paper">Paper protocol: group-aware search and validation</option>'
    '</select></div><p id="paperStatus" class="hint"></p>'
    '<button id="train" class="primary">Train XGBoost</button>')
HTML = HTML.replace(
    '<p id="roundInfo">The trained model is ready for another photo.</p>'
    '<button id="nextPhoto" class="primary">',
    '<p id="roundInfo">The trained model is ready for another photo.</p>'
    '<div id="trainingEvidence" class="evidence" aria-live="polite"></div>'
    '<div class="divider">Optional YOLO support for inference</div>'
    '<p>Use an existing one-class segmentation YOLO model as a source of boxes and confidence scores. '
    'COMPAG keeps SAM2 masks and XGBoost features; this step does not retrain YOLO.</p>'
    '<p id="yoloLocationStatus" class="hint"></p>'
    '<div class="field"><label for="yoloPackage">YOLO model package folder (automatic)</label>'
    '<input id="yoloPackage" type="text" autocomplete="off" spellcheck="false" '
    'placeholder="Leave blank to find the model automatically"><small>Saved and nearby model folders are found automatically. You can also choose another package once.</small></div>'
    '<div class="field"><label for="yoloPython">Advanced: Python with Ultralytics (optional)</label>'
    '<input id="yoloPython" type="text" autocomplete="off" spellcheck="false" '
    'placeholder="Leave blank for automatic YOLO setup"><small>Leave blank to prepare a separate YOLO environment automatically.</small></div>'
    '<div class="field"><label for="yoloMode">How YOLO supports the next inference</label>'
    '<select id="yoloMode"><option value="off">xgb_recall: YOLO disabled (XGBoost only)</option>'
    '<option value="prompt">sam2_policy=both (AMG + YOLO prompts), det_policy=xgb</option>'
    '<option value="fusion">sam2_policy=auto (AMG), det_policy=hybrid</option>'
    '<option value="both">sam2_policy=both (AMG + YOLO prompts), det_policy=hybrid</option></select></div>'
    '<p class="hint">Paper hybrid settings: det_missing=reject; thr_xgb=det_thr=0.50; '
    'thr_yolo=0.20; thr_iou=0.60; hybrid_yolo_bias=0.80.</p>'
    '<button id="activateYolo" class="secondary">Apply YOLO setting</button>'
    '<p id="yoloStatus" class="hint"></p>'
    '<div class="divider">Continue the cycle</div>'
    '<button id="nextPhoto" class="primary">')

HTML = HTML.replace('</aside></main>', '''</aside>
<section class="panel export-panel" aria-labelledby="exportHeading">
<h2 id="exportHeading">Export saved masks</h2>
<p>Download original photos and their masks as a COCO ZIP. Export again whenever you need it; training and a completed review are not required.</p>
<div class="field"><label for="exportSelect">Saved analysis</label><select id="exportSelect"></select></div>
<div class="field"><label for="exportScope">Labels to export</label><select id="exportScope">
<option value="current">Current labels + model predictions for unreviewed masks</option>
<option value="reviewed">Saved labels only (including bulk acceptances)</option>
</select></div>
<button id="exportCoco" class="primary">Export COCO ZIP</button>
<p id="exportStatus" class="hint">Deleted and skipped masks are excluded. Close the review page before exporting. Each label records whether it came from the model, bulk acceptance, or individual review.</p>
<p class="hint">Overlapping candidates are preserved. Exports contain CJ and non-CJ masks in original-photo coordinates.</p>
<div id="exportDownloads" class="downloads" aria-live="polite"></div></section></main>''')
HTML = HTML.replace('<details><summary>Show step details</summary>',
                    '<button id="stopJob" class="secondary danger" hidden>Stop current operation</button>'
                    '<details><summary>Show step details</summary>')

CSS = r"""
:root{color-scheme:light;--ink:#17263c;--muted:#57677c;--line:#dce4eb;--blue:#176ac5;--deep:#103d74;--pale:#eef6ff;--green:#086d4b;--red:#a5363d;--bg:#f4f7fa}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,input,select{font:inherit}button{cursor:pointer}button:disabled{opacity:.48;cursor:not-allowed}.shell{max-width:1040px;margin:0 auto;min-height:100vh;background:#fff;box-shadow:0 0 50px #223f5c11}.top{display:flex;align-items:center;justify-content:space-between;padding:22px 42px;border-bottom:1px solid var(--line)}.brand{display:flex;align-items:center;gap:12px}.brand strong{display:block;font-size:18px;letter-spacing:.05em}.brand small{display:block;font-size:12px;color:var(--muted)}.mark{display:grid;place-items:center;width:42px;height:42px;border-radius:12px;background:var(--deep);color:#fff;font-weight:800;font-size:21px}.private{font-size:13px;color:var(--green);background:#e8f7ef;padding:7px 12px;border-radius:99px;font-weight:650}main{padding:36px 42px 48px}.intro{max-width:720px}.eyebrow,.number{font-size:12px;color:var(--blue);font-weight:800;letter-spacing:.13em}.intro h1{font-size:clamp(31px,5vw,46px);line-height:1.13;letter-spacing:-.035em;margin:7px 0 10px}.intro p,.panel p{color:var(--muted);margin:0 0 23px}.steps{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:7px;margin:30px 0 25px}.steps button{border:1px solid var(--line);border-radius:11px;background:#fff;color:var(--muted);padding:12px 5px;display:flex;align-items:center;justify-content:center;gap:7px;min-height:54px}.steps button b{display:grid;place-items:center;background:#edf1f5;color:#6c7b8d;border-radius:50%;width:23px;height:23px;font-size:12px}.steps button.current{background:var(--pale);border-color:#a8cbed;color:var(--deep);font-weight:750}.steps button.current b,.steps button.done b{background:var(--blue);color:#fff}.steps button.done{color:var(--deep)}.notice{border-radius:10px;padding:12px 15px;margin-bottom:18px;background:#eaf6ee;color:var(--green)}.notice:empty{display:none}.notice.error{background:#fff0f0;color:var(--red)}.panel{border:1px solid var(--line);border-radius:17px;padding:30px 32px;box-shadow:0 10px 36px #162f4b0b;max-width:790px}.panel h2{font-size:26px;letter-spacing:-.025em;margin:6px 0 7px}.panel .hint{font-size:13px;color:var(--muted);margin:15px 0 0}.field{display:flex;flex-direction:column;gap:7px;margin:18px 0}.field label{font-weight:700;font-size:14px}.field input[type=text],.field input[type=file],.field select{width:100%;background:#fff;border:1px solid #b6c4d3;border-radius:9px;padding:12px 13px;color:var(--ink);min-height:46px}.field input:focus,.field select:focus{outline:3px solid #bfddff;outline-offset:1px}.field small{color:var(--muted);font-size:12px}.primary,.secondary,.review-link{border-radius:9px;padding:11px 18px;min-height:44px;font-weight:750;display:inline-flex;align-items:center;justify-content:center;margin-right:8px}.primary{border:1px solid var(--blue);background:var(--blue);color:#fff}.primary:hover:not(:disabled){background:var(--deep)}.secondary{border:1px solid #b8c8d9;background:#fff;color:var(--deep)}.secondary:hover:not(:disabled){background:var(--pale)}.subtle-link,.review-link{color:var(--blue);text-decoration:none;font-weight:700}.subtle-link:hover,.review-link:hover{text-decoration:underline}.review-link{margin:12px 0;background:var(--pale)}.divider{display:flex;align-items:center;gap:12px;color:#79879a;font-size:12px;font-weight:750;letter-spacing:.08em;text-transform:uppercase;margin:26px 0 12px}.divider:before,.divider:after{content:"";height:1px;background:var(--line);flex:1}.choice{display:flex;gap:20px;margin:19px 0}.choice label,.check{font-weight:650}.choice input,.check input{accent-color:var(--blue);margin-right:6px}.check{display:block;margin:22px 0 14px}.progress{max-width:790px;margin-top:20px;padding:20px 22px;border:1px solid var(--line);border-radius:13px;background:#f9fbfd}.progress-head{display:flex;align-items:center;gap:10px}.progress-message{color:var(--muted);margin:7px 0 12px;white-space:pre-wrap}.progress details{border-top:1px solid var(--line);padding-top:11px}.progress summary{cursor:pointer;color:var(--deep);font-weight:700;font-size:13px}.progress pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:250px;overflow:auto;font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;color:#435165}.working #progressSpin{border:3px solid #c6def8;border-top-color:var(--blue);height:17px;width:17px;border-radius:50%;animation:spin .9s linear infinite}.plan-summary{background:var(--pale);border-radius:10px;margin:16px 0;padding:13px 16px;color:var(--deep);font-size:14px}.plan-summary:empty{display:none}footer{border-top:1px solid var(--line);padding:18px 42px;color:#7b8898;font-size:12px}@keyframes spin{to{transform:rotate(360deg)}}@media(max-width:700px){.top{padding:16px 20px}.private{display:none}main{padding:24px 18px}.steps{grid-template-columns:repeat(3,minmax(0,1fr));gap:5px}.steps button{min-height:48px}.panel{padding:24px 19px}.choice{flex-wrap:wrap}footer{padding:15px 20px}}
"""

CSS += """
.evidence{background:#f0f7f3;border:1px solid #c4dfcf;border-radius:10px;margin:14px 0 20px;padding:15px 17px;color:var(--ink);font-size:14px;white-space:pre-wrap;overflow-wrap:anywhere}
.evidence:empty{display:none}
"""

CSS += "\n.export-panel{margin-top:24px}.downloads .review-link{display:block;overflow-wrap:anywhere}.danger{color:#a32727;border-color:#d88888;margin-bottom:14px}\n"

JS = r"""
(() => {
  'use strict';
  const base = location.pathname.replace(/\/$/, '');
  const csrf = document.querySelector('meta[name="csrf-token"]').content;
  const $ = id => document.getElementById(id);
  let state = null;
  let step = 0;
  let lastJob = '';
  let localBusy = false;
  let hadLatest = false;
  let noticeText = '';
  let noticeError = false;
  let pendingReviewTab = null;
  let reviewLaunchRequested = false;
  let lastSelectedSession = null;
  let lastYoloRound = null;
  const dirtyPaths = new Set();
  const actionLabels = {
    check: 'Checking included files', setup: 'Installing and preparing COMPAG',
    infer: 'Analyzing photo', resume: 'Continuing analysis',
    plan: 'Checking saved review decisions', train: 'Training XGBoost',
    attach_yolo: 'Checking and activating YOLO for inference',
    export: 'Exporting saved masks'
  };

  async function post(action, data = {}) {
    const response = await fetch(base + '/api/' + action, {
      method: 'POST', credentials: 'same-origin', cache: 'no-store',
      headers: {'Content-Type': 'application/json', 'X-COMPAG-CSRF': csrf},
      body: JSON.stringify(data)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'This step could not finish.');
    return result;
  }

  async function upload(action, file, headerName) {
    const response = await fetch(base + '/api/' + action, {
      method: 'POST', credentials: 'same-origin', cache: 'no-store',
      headers: {'Content-Type': 'application/octet-stream', 'X-COMPAG-CSRF': csrf,
                'X-File-Name': headerName},
      body: file
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The file could not be verified.');
    return result;
  }

  function announce(message, error = false) {
    noticeText = message;
    noticeError = error;
    $('notice').textContent = message;
    $('notice').classList.toggle('error', error);
  }

  function show(target) {
    step = Math.max(1, Math.min(6, target));
    for (let i = 1; i <= 6; i++) $('step' + i).hidden = i !== step;
    for (const item of document.querySelectorAll('.steps button')) {
      const n = Number(item.dataset.step);
      item.classList.toggle('current', n === step);
      item.classList.toggle('done', n < step);
      item.setAttribute('aria-current', n === step ? 'step' : 'false');
    }
    window.scrollTo({top: 0, behavior: 'smooth'});
  }

  function options(element, rows, selected) {
    const previous = element.value;
    element.replaceChildren();
    for (const row of rows) {
      const option = document.createElement('option');
      option.value = row.value;
      option.textContent = row.label;
      element.append(option);
    }
    if (rows.some(row => row.value === previous)) element.value = previous;
    else if (selected && rows.some(row => row.value === selected)) element.value = selected;
  }

  function selected(id) {
    const value = $(id).value;
    if (!value) throw new Error('Choose an analysis from the list.');
    return value;
  }

  function syncInput(id, value) {
    if (!dirtyPaths.has(id) && document.activeElement !== $(id)) $(id).value = value || '';
  }

  function render(data) {
    state = data;
    if (!step) {
      if (data.latest_round) show(6);
      else if (data.sessions.some(row => row.status === 'SCORED')) show(4);
      else if (data.ready) show(3);
      else if (data.job.status === 'DONE' && data.job.action === 'check') show(2);
      else show(1);
    }
    syncInput('prefixDir', data.paths.prefix);
    syncInput('workspaceDir', data.paths.workspace);
    const locations = data.yolo_locations || {};
    syncInput('yoloPackage', locations.model_package);
    syncInput('yoloPython', locations.python);
    $('yoloLocationStatus').textContent = locations.status === 'FOUND' ?
      'YOLO package found. Its checksum and runtime are verified when you apply the setting; the locations are then remembered for later rounds.' :
      (locations.message || 'YOLO locations will be found automatically.');
    options($('cardSelect'), data.cards.map(name => ({value: name, label: name})));
    const interrupted = data.sessions.filter(row => ['PREPARING','ANALYZING','INTERRUPTED'].includes(row.status));
    const reviewed = data.sessions.filter(row => ['SCORED','TRAINED'].includes(row.status));
    const trainable = data.sessions.filter(row => row.status === 'SCORED');
    const label = row => row.card + ' · ' + row.status + ' · ' + row.id;
    options($('resumeSelect'), interrupted.map(row => ({value: row.id, label: label(row)})), data.selected_session);
    const selectedChanged = data.selected_session && data.selected_session !== lastSelectedSession;
    if (selectedChanged) {
      $('reviewSelect').value = '';
      $('trainSelect').value = '';
      lastSelectedSession = data.selected_session;
    }
    options($('reviewSelect'), reviewed.map(row => ({value: row.id, label: label(row)})), data.selected_session);
    options($('trainSelect'), trainable.map(row => ({value: row.id, label: label(row)})), data.selected_session);
    options($('exportSelect'), [
      ...reviewed.map(row => ({value: row.id, label: label(row)})),
      ...(reviewed.length > 1 ? [{value:'__all__',label:'All completed analyses'}] : [])
    ], data.selected_session);
    $('exportDownloads').replaceChildren();
    for (const item of data.exports || []) {
      const link = document.createElement('a');
      link.href = base + '/downloads/' + encodeURIComponent(item.filename);
      link.download = item.filename;
      link.className = 'review-link';
      link.textContent = 'Download ' + item.filename + ' · ' + item.image_count + ' photo(s), ' + item.annotation_count + ' masks';
      $('exportDownloads').append(link);
    }
    const latest = !!data.latest_round;
    $('latestChoice').hidden = !latest;
    if (latest && !hadLatest) $('useLatest').checked = true;
    hadLatest = latest;
    if (latest) $('roundInfo').textContent = 'The latest verified model set is ready for another photo.';
    const proof = data.training_evidence;
    if (latest && proof && proof.status === 'VERIFIED') {
      const profile = proof.policy === 'SMALL_DATA_OPERATIONAL_NO_HOLDOUT' ?
        'Operational fit without holdout or cross-validation' :
        proof.policy === 'PAPER_SETTINGS_NEW_DATA_GROUPED' ? 'Paper protocol group-aware fit on new reviewed data' :
        String(proof.policy || 'Recorded fit');
      const allBulk = proof.row_count > 0 && proof.bulk_accepted_count === proof.row_count;
      $('trainingEvidence').textContent =
        'Verified XGBoost training\n' +
        'XGBoost round: ' + proof.round_id + '\n' +
        'Method: ' + profile + '\n' +
        'Training rows: ' + proof.row_count + ' from ' + proof.group_count + ' original photo(s)\n' +
        'Bulk accepted predictions: ' + proof.bulk_accepted_count + '\n' +
        'Fitted boosting rounds: ' + proof.boosting_rounds + '\n' +
        'Group cross-validation: ' + (proof.group_cv_performed ? 'yes' : 'no') + '\n' +
        'Held-out card test: ' + (proof.heldout_test_performed ? 'yes' : 'no') + '\n' +
        'Classifier SHA-256: ' + proof.classifier_sha256 + '\n' +
        'Saved model: ' + proof.model_folder +
        (allBulk ? '\nAll labels in this fit repeated model predictions; this is not independent ground truth.' : '');
      const currentYoloMode = String(proof.yolo_mode || 'off');
      const currentYoloOption = Array.from($('yoloMode').options).find(option => option.value === currentYoloMode);
      const currentYoloLabel = currentYoloOption ? currentYoloOption.textContent : currentYoloMode;
      $('yoloStatus').textContent = currentYoloMode !== 'off' ?
        'Current YOLO setting: ' + currentYoloLabel + '. Checkpoint SHA-256: ' +
        (proof.yolo_checkpoint_sha256 || 'not recorded') + '. It will be used on the next photo when Use latest trained model is selected.' :
        'Current YOLO setting: ' + currentYoloLabel + '. The latest model uses XGBoost and SAM2 without YOLO.';
      const yoloKey = String(data.latest_round.model_set || data.latest_round.round_id);
      if (lastYoloRound !== yoloKey) {
        lastYoloRound = yoloKey;
        $('yoloMode').value = proof.yolo_mode || 'off';
      }
      if ($('yoloMode').value !== currentYoloMode)
        $('yoloStatus').textContent += ' Your new selection has not been applied yet.';
    } else {
      $('trainingEvidence').textContent = latest ?
        'A saved round exists, but its training evidence could not be verified here: ' +
        ((proof && proof.reason) || 'missing files') : '';
      $('yoloStatus').textContent = latest ? 'Check the saved model before enabling YOLO.' : '';
    }
    $('photoStatus').textContent = data.uploaded_photo ? 'Photo added to your private work folder.' : '';
    const review = data.review;
    $('reviewLink').hidden = !review.url;
    if (review.url) $('reviewLink').href = review.url;
    else $('reviewLink').removeAttribute('href');
    if (review.url && pendingReviewTab && !pendingReviewTab.closed) {
      pendingReviewTab.location.replace(review.url);
      pendingReviewTab = null;
    } else if (!review.active && !reviewLaunchRequested && pendingReviewTab && !pendingReviewTab.closed) {
      pendingReviewTab.close();
      pendingReviewTab = null;
    }
    $('stopReview').hidden = !review.active;
    $('reviewStatus').textContent = review.error ? review.error : review.active ?
      (review.url ? 'Review is open. Save decisions there, then return here.' : 'Starting the review page…') :
      'The review page is closed. Reopen it to continue making decisions.';
    const plan = data.review_plan;
    const planMatches = plan && data.plan_session === $('trainSelect').value;
    $('planSummary').textContent = planMatches ?
      'Saved decisions: ' + plan.eligible_labeled_count + ' labeled for training · ' +
      plan.unreviewed_count + ' unreviewed · ' + plan.deleted_count + ' deleted · ' +
      plan.skipped_count + ' skipped · ' + plan.bulk_accepted_count + ' accepted as model predictions · ' +
      plan.source_card_count + ' original photo(s) in cumulative data · ' +
      plan.parent_bulk_accepted_count + ' bulk acceptances inherited from earlier rounds.' :
      'Check saved decisions to preview what will enter training.';
    $('paperStatus').textContent = $('trainPolicy').value === 'paper' ?
      (planMatches ?
        (plan.paper_ready ? 'Ready for paper-protocol training. ' : 'Not ready for paper-protocol training. ') +
        plan.paper_reason + ' The paper specifies five group folds, 30 search settings, and group-safe early stopping; '+
        'it does not explicitly prescribe six photos. The minimum of six follows from this implementation retaining a separate held-out test. '+
        (!plan.paper_ready ? 'To build an eligible dataset, start with Original model, make individual Accept/Flip decisions on photos containing both classes, '+
        'and accumulate clean rounds with Operational training and Use latest trained model. Avoid Accept remaining in that lineage; recheck here once enough photos are available.' : '') :
        'Check saved decisions first. The paper protocol checks original-photo groups and labels before fitting.') :
      'Operational fit uses all eligible decisions with a fixed XGBoost recipe and no holdout, cross-validation, or paper-performance claim.';
    const busy = localBusy || ['RUNNING','STOPPING','CLEANUP_FAILED'].includes(data.job.status);
    document.body.classList.toggle('working', busy);
    for (const id of ['checkRelease','setup','uploadPhoto','infer','resume',
                      'startReview','stopReview','checkPlan','train','activateYolo','nextPhoto']) {
      $(id).disabled = busy;
    }
    $('infer').disabled = busy || !data.ready || !data.cards.length;
    $('resume').disabled = busy || !interrupted.length;
    $('startReview').disabled = busy || !reviewed.length || review.active;
    $('checkPlan').disabled = busy || !trainable.length || review.active;
    $('train').disabled = busy || review.active || !planMatches || plan.eligible_labeled_count < 1 ||
      ($('trainPolicy').value === 'paper' && !plan.paper_ready);
    $('exportCoco').disabled = busy || review.active || !reviewed.length;
    $('exportSelect').disabled = busy;
    $('exportScope').disabled = busy;
    $('stopJob').hidden = !['RUNNING','STOPPING'].includes(data.job.status) || !['infer','resume','train','export'].includes(data.job.action);
    $('stopJob').disabled = data.job.status === 'STOPPING';
    $('stopJob').textContent = data.job.status === 'STOPPING' ? 'Stopping…' :
      data.job.action === 'train' ? 'Stop training' : data.job.action === 'export' ? 'Stop export' : 'Stop analysis';
    $('activateYolo').disabled = busy || !latest || !proof || proof.status !== 'VERIFIED';
    $('stopReview').disabled = busy || !review.active;
    $('progressTitle').textContent = data.job.status === 'CLEANUP_FAILED' ? 'Cleanup failed — new work blocked' : data.job.status === 'STOPPING' ? 'Stopping…' :
      data.job.status === 'STOPPED' ? 'Stopped — saved work kept' : busy ? (actionLabels[data.job.action] || 'Working') :
      data.job.status === 'ERROR' ? 'A step needs attention' :
      data.job.status === 'DONE' && data.job.action === 'train' ? 'XGBoost model fitted and saved' :
      data.job.status === 'DONE' ? 'Step complete' : 'Ready';
    $('progressMessage').textContent = data.job.status === 'ERROR' ? data.job.error :
      data.job.status === 'DONE' && data.job.action === 'train' && proof && proof.status === 'VERIFIED' ?
        'XGBoost trained on ' + proof.row_count + ' rows; ' + proof.boosting_rounds +
        ' boosting rounds. See the verified training record above.' :
      data.job.logs.length ? data.job.logs[data.job.logs.length - 1] :
      busy ? 'This can take a while. Keep this page open.' : 'Follow the highlighted step to continue.';
    $('log').textContent = data.job.logs.join('\n');
    const jobKey = data.job.id + ':' + data.job.status;
    if (lastJob !== jobKey) {
      lastJob = jobKey;
      if (data.job.status === 'DONE') {
        const advance = {check:2,setup:3,infer:4,resume:4,plan:5,train:6,attach_yolo:6};
        if (advance[data.job.action]) show(advance[data.job.action]);
        announce(data.job.action === 'train' && proof && proof.status === 'VERIFIED' ?
          'XGBoost trained: ' + proof.row_count + ' rows, ' + proof.boosting_rounds + ' boosting rounds.' :
          actionLabels[data.job.action] + ' complete.');
      } else if (data.job.status === 'RUNNING') {
        const active = data.sessions.find(row => row.status === 'ANALYZING');
        const mode = active && ['infer', 'resume'].includes(data.job.action) ?
          (active.yolo_mode === 'off' ? ' · YOLO is off for this analysis.' : ' · YOLO mode: ' + active.yolo_mode) : '';
        announce((actionLabels[data.job.action] || 'Working') + mode);
      } else if (data.job.status === 'ERROR') {
        announce(data.job.error || 'This step did not finish.', true);
      } else if (data.job.status === 'STOPPED') {
        announce('Stopped. Completed tiles and saved decisions are kept. Use Continue analysis to resume, or start training again.');
      }
    }
  }

  async function refresh() {
    try {
      const response = await fetch(base + '/api/state', {cache: 'no-store', credentials: 'same-origin'});
      if (!response.ok) throw new Error('The local guide is no longer available. Restart it from the terminal.');
      render(await response.json());
    } catch (error) { announce(error.message, true); }
  }

  async function run(work) {
    if (localBusy) return;
    localBusy = true;
    announce('');
    if (state) render(state);
    try { await work(); }
    catch (error) { announce(error.message || String(error), true); }
    finally { localBusy = false; await refresh(); }
  }

  async function savePaths() {
    const result = await post('config', {prefix:$('prefixDir').value,
                                         workspace:$('workspaceDir').value});
    dirtyPaths.delete('prefixDir');
    dirtyPaths.delete('workspaceDir');
    return result;
  }

  for (const id of ['prefixDir', 'workspaceDir', 'yoloPackage', 'yoloPython']) {
    $(id).addEventListener('input', () => dirtyPaths.add(id));
  }

  $('checkRelease').addEventListener('click', () => run(async () => {
    await post('check');
  }));
  $('setup').addEventListener('click', () => run(async () => {
    await savePaths();
    await post('setup');
  }));

  function cardMode() {
    return document.querySelector('input[name="cardMode"]:checked').value;
  }
  function showCardMode() {
    const own = cardMode() === 'own';
    $('examplePicker').hidden = own;
    $('ownPicker').hidden = !own;
  }
  for (const radio of document.querySelectorAll('input[name="cardMode"]'))
    radio.addEventListener('change', showCardMode);

  $('uploadPhoto').addEventListener('click', () => run(async () => {
    const file = $('photoFile').files[0];
    if (!file) throw new Error('Choose one JPG or PNG photo first.');
    const extension = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
    if (!['.jpg','.jpeg','.png'].includes(extension)) throw new Error('Choose a JPG or PNG photo.');
    if (file.size > 50 * 1024 * 1024) throw new Error('Choose a photo smaller than 50 MB.');
    await upload('upload-photo', file, 'photo' + extension);
    announce('Photo added to your private work folder.');
  }));
  $('infer').addEventListener('click', () => run(async () => {
    if ($('useLatest').checked && state.latest_round &&
        $('yoloMode').value !== (state.latest_round.yolo_mode || 'off')) {
      show(6);
      throw new Error('Apply your selected YOLO setting before analyzing the next photo.');
    }
    await post('infer', {mode:cardMode(), card_id:$('cardSelect').value,
                         use_latest:$('useLatest').checked});
  }));
  $('resume').addEventListener('click', () => run(async () => {
    await post('resume', {session_id:selected('resumeSelect')});
  }));
  $('startReview').addEventListener('click', () => {
    reviewLaunchRequested = true;
    const tab = window.open('about:blank', '_blank');
    if (tab) {
      tab.opener = null;
      tab.document.title = 'Opening COMPAG review';
      tab.document.body.textContent = 'Opening the local COMPAG review page…';
      pendingReviewTab = tab;
    }
    run(async () => {
      try {
        await post('review', {session_id:selected('reviewSelect')});
        reviewLaunchRequested = false;
        announce(tab ? 'The review tab is opening. Return here when your decisions are saved.' :
                       'Review is starting. Use the review link below when it appears.');
      } catch (error) {
        reviewLaunchRequested = false;
        if (tab && !tab.closed) tab.close();
        pendingReviewTab = null;
        throw error;
      }
    });
  });
  $('stopReview').addEventListener('click', () => run(async () => {
    const session = state.review.session;
    await post('review-stop');
    show(5);
    if (session) await post('plan', {session_id:session});
  }));
  $('checkPlan').addEventListener('click', () => run(async () => {
    await post('plan', {session_id:selected('trainSelect')});
  }));
  $('train').addEventListener('click', () => run(async () => {
    await post('train', {session_id:selected('trainSelect'), policy:$('trainPolicy').value});
  }));
  $('stopJob').addEventListener('click', () => run(async () => {
    await post('stop-job', {job_id: state.job.id});
  }));
  $('exportCoco').addEventListener('click', () => run(async () => {
    await post('export', {session_id:selected('exportSelect'), scope:$('exportScope').value});
  }));
  $('activateYolo').addEventListener('click', () => run(async () => {
    const mode = $('yoloMode').value;
    await post('attach-yolo', {model_package:$('yoloPackage').value.trim(),
                               yolo_python:$('yoloPython').value.trim(), mode});
    dirtyPaths.delete('yoloPackage');
    dirtyPaths.delete('yoloPython');
  }));
  $('nextPhoto').addEventListener('click', () => {
    $('useLatest').checked = true;
    show(3);
  });
  for (const item of document.querySelectorAll('.steps button')) {
    item.addEventListener('click', () => show(Number(item.dataset.step)));
  }
  $('trainSelect').addEventListener('change', () => { if (state) render(state); });
  $('trainPolicy').addEventListener('change', () => { if (state) render(state); });
  $('yoloMode').addEventListener('change', () => { if (state) render(state); });
  refresh().then(() => {
    if (state && !state.ready && state.job.status === 'IDLE')
      run(async () => { await post('check'); });
  });
  setInterval(refresh, 1500);
})();
"""
