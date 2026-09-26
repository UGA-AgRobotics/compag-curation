"""Small desktop front end for the optional COMPAG easy-start launcher.

The scientific commands remain in :mod:`start_compag`; this file only gathers
paths, runs those commands off the UI thread, and presents their progress.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import webbrowser


class _EasyStartWindow:
    def __init__(self, root, backend_module, tk, ttk, filedialog, messagebox):
        self.root = root
        self.backend_module = backend_module
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.events: queue.Queue[tuple] = queue.Queue()
        self.busy = False
        self.closed = False
        self.review_process: subprocess.Popen | None = None
        self.card_paths: list[Path] = []
        self.sessions: list[dict] = []
        self.path_controls = []

        self.release_var = tk.StringVar(value=str(Path.home() / "Downloads"))
        self.prefix_var = tk.StringVar(
            value=str(Path.home() / ".local/share/compag-r92-rc10-gpu")
        )
        self.workspace_var = tk.StringVar(value=str(Path.home() / "COMPAG_Workspace"))
        self.own_card_var = tk.StringVar()
        self.card_mode = tk.StringVar(value="example")
        self.use_latest = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Choose the release files, then install.")

        self._load_saved_paths()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._drain_events)
        self._start_task("refresh", lambda backend: {
            "cards": backend.list_cards(),
            "sessions": backend.list_sessions(),
        })

    def _load_saved_paths(self) -> None:
        try:
            backend = self._backend()
            settings = backend.load_settings()
        except Exception:
            return
        if not isinstance(settings, dict):
            return
        self.use_latest.set(bool(settings.get("latest_round")))
        for var, names in (
            (self.release_var, ("release_dir", "release")),
            (self.prefix_var, ("prefix", "environment")),
            (self.workspace_var, ("workspace", "workspace_dir")),
        ):
            for name in names:
                if settings.get(name):
                    var.set(str(settings[name]))
                    break

    def _backend(self, paths: tuple[Path, Path, Path] | None = None):
        if paths is None:
            paths = (
                Path(self.release_var.get()).expanduser(),
                Path(self.prefix_var.get()).expanduser(),
                Path(self.workspace_var.get()).expanduser(),
            )
        return self.backend_module.Launcher(
            paths[0], paths[1], paths[2],
            log=lambda text: self.events.put(("log", str(text))),
        )

    def _build_ui(self) -> None:
        self.root.title("COMPAG — Set up and review")
        self.root.geometry("960x760")
        self.root.minsize(780, 620)
        outer = self.ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(4, weight=1)

        title = self.ttk.Label(
            outer, text="COMPAG", font=("TkDefaultFont", 20, "bold")
        )
        title.grid(row=0, column=0, sticky="w")
        self.ttk.Label(
            outer,
            text="Install once, choose a card, and continue your review here.",
        ).grid(row=1, column=0, sticky="w", pady=(0, 12))

        setup = self.ttk.LabelFrame(outer, text="1. Install", padding=12)
        setup.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        setup.columnconfigure(1, weight=1)
        self._path_row(setup, 0, "Downloaded release folder", self.release_var,
                       lambda: self._browse_dir(self.release_var))
        self._path_row(setup, 1, "Your work folder", self.workspace_var,
                       self._browse_workspace)
        self.ttk.Label(setup, text="App location").grid(
            row=2, column=0, sticky="w", pady=4, padx=(0, 8)
        )
        prefix_entry = self.ttk.Entry(setup, textvariable=self.prefix_var)
        prefix_entry.grid(
            row=2, column=1, sticky="ew", pady=4
        )
        self.path_controls.append(prefix_entry)
        setup_buttons = self.ttk.Frame(setup)
        setup_buttons.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.check_button = self.ttk.Button(
            setup_buttons, text="Check files", command=self._check_release
        )
        self.check_button.pack(side="left", padx=(0, 8))
        self.install_button = self.ttk.Button(
            setup_buttons, text="Install and prepare", command=self._setup
        )
        self.install_button.pack(side="left", padx=(0, 8))
        self.release_button = self.ttk.Button(
            setup_buttons, text="Get release files", command=self._open_release_page
        )
        self.release_button.pack(side="left")
        try:
            if not self.backend_module.release_page_url():
                self.release_button.state(["disabled"])
        except Exception:
            self.release_button.state(["disabled"])

        work = self.ttk.LabelFrame(outer, text="2. Choose a card", padding=12)
        work.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        work.columnconfigure(1, weight=1)
        self.ttk.Radiobutton(
            work, text="Included example", variable=self.card_mode, value="example"
        ).grid(row=0, column=0, sticky="w")
        self.card_list = self.tk.Listbox(work, height=3, exportselection=False)
        self.card_list.grid(row=0, column=1, columnspan=2, sticky="ew", pady=4)
        self.card_list.bind("<<ListboxSelect>>", self._example_selected)
        self.ttk.Radiobutton(
            work, text="My photo folder", variable=self.card_mode, value="own"
        ).grid(row=1, column=0, sticky="w", padx=(0, 8))
        self.ttk.Entry(work, textvariable=self.own_card_var).grid(
            row=1, column=1, sticky="ew", pady=4
        )
        self.ttk.Button(
            work, text="Browse…", command=self._browse_own_card
        ).grid(row=1, column=2, padx=(8, 0), pady=4)
        self.ttk.Checkbutton(
            work,
            text="Use my latest trained model, if available",
            variable=self.use_latest,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(5, 0))
        self.analyze_button = self.ttk.Button(
            work, text="Analyze this card", command=self._create_session
        )
        self.analyze_button.grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))

        saved = self.ttk.LabelFrame(outer, text="3. Continue saved work", padding=12)
        saved.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        saved.columnconfigure(0, weight=1)
        saved.rowconfigure(0, weight=1)
        self.session_list = self.tk.Listbox(saved, height=4, exportselection=False)
        self.session_list.grid(row=0, column=0, sticky="nsew")
        self.session_list.bind("<<ListboxSelect>>", lambda _event: self._refresh_buttons())
        scroll = self.ttk.Scrollbar(saved, orient="vertical", command=self.session_list.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.session_list.configure(yscrollcommand=scroll.set)
        saved_buttons = self.ttk.Frame(saved)
        saved_buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self.refresh_button = self.ttk.Button(
            saved_buttons, text="Refresh list", command=self._refresh
        )
        self.refresh_button.pack(side="left", padx=(0, 8))
        self.resume_button = self.ttk.Button(
            saved_buttons, text="Continue selected work", command=self._resume
        )
        self.resume_button.pack(side="left", padx=(0, 8))
        self.review_button = self.ttk.Button(
            saved_buttons, text="Open review", command=self._open_review
        )
        self.review_button.pack(side="left", padx=(0, 8))
        self.stop_button = self.ttk.Button(
            saved_buttons, text="Stop review", command=self._stop_review
        )
        self.stop_button.pack(side="left", padx=(0, 8))
        self.train_button = self.ttk.Button(
            saved_buttons, text="Train next model", command=self._train
        )
        self.train_button.pack(side="left")

        progress = self.ttk.LabelFrame(outer, text="Progress", padding=10)
        progress.grid(row=5, column=0, sticky="nsew")
        progress.columnconfigure(0, weight=1)
        self.ttk.Label(
            progress, textvariable=self.status_var, wraplength=850
        ).grid(row=0, column=0, sticky="w")
        log_frame = self.ttk.Frame(progress)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        self.log_text = self.tk.Text(log_frame, height=6, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = self.ttk.Scrollbar(
            log_frame, orient="vertical", command=self.log_text.yview
        )
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self._refresh_buttons()

    def _path_row(self, frame, row, label, variable, browse) -> None:
        self.ttk.Label(frame, text=label).grid(
            row=row, column=0, sticky="w", pady=4, padx=(0, 8)
        )
        entry = self.ttk.Entry(frame, textvariable=variable)
        entry.grid(
            row=row, column=1, sticky="ew", pady=4
        )
        browse_button = self.ttk.Button(frame, text="Browse…", command=browse)
        browse_button.grid(
            row=row, column=2, padx=(8, 0), pady=4
        )
        self.path_controls.extend((entry, browse_button))

    def _browse_dir(self, variable) -> bool:
        initial = Path(variable.get()).expanduser()
        selected = self.filedialog.askdirectory(
            initialdir=str(initial if initial.is_dir() else Path.home())
        )
        if selected:
            variable.set(selected)
            return True
        return False

    def _browse_workspace(self) -> None:
        if self._browse_dir(self.workspace_var):
            self.use_latest.set(False)
            self._load_saved_paths()
            self._refresh()

    def _browse_own_card(self) -> None:
        self._browse_dir(self.own_card_var)
        if self.own_card_var.get():
            self.card_mode.set("own")

    def _example_selected(self, _event) -> None:
        if self.card_list.curselection():
            self.card_mode.set("example")

    def _selected_session(self) -> Path | None:
        selection = self.session_list.curselection()
        if not selection:
            return None
        value = self.sessions[selection[0]].get("path")
        return Path(value).expanduser() if value else None

    def _start_task(self, name, action) -> None:
        if self.busy:
            return
        self.busy = True
        self.status_var.set("Working…")
        self._refresh_buttons()
        paths = (
            Path(self.release_var.get()).expanduser(),
            Path(self.prefix_var.get()).expanduser(),
            Path(self.workspace_var.get()).expanduser(),
        )

        def run() -> None:
            try:
                result = action(self._backend(paths))
                self.events.put(("task_done", name, result, None))
            except Exception as exc:
                self.events.put(("task_done", name, None, exc))

        threading.Thread(target=run, name=f"compag-{name}", daemon=True).start()

    def _check_release(self) -> None:
        self._start_task("check", lambda backend: {
            "release": backend.check_release(),
            "cards": backend.list_cards(),
            "sessions": backend.list_sessions(),
        })

    def _setup(self) -> None:
        self._start_task("setup", lambda backend: {
            "setup": backend.setup(),
            "cards": backend.list_cards(),
            "sessions": backend.list_sessions(),
        })

    def _refresh(self) -> None:
        self._start_task("refresh", lambda backend: {
            "cards": backend.list_cards(),
            "sessions": backend.list_sessions(),
        })

    def _create_session(self) -> None:
        if self.card_mode.get() == "own":
            card_text = self.own_card_var.get().strip()
            if not card_text:
                self.messagebox.showinfo("Choose photos", "Choose your photo folder first.")
                return
            card_dir = Path(card_text).expanduser()
        else:
            selection = self.card_list.curselection()
            if not selection:
                self.messagebox.showinfo("Choose a card", "Select an included card first.")
                return
            card_dir = self.card_paths[selection[0]]
        if not card_dir.is_dir():
            self.messagebox.showerror(
                "Folder missing", f"This card folder cannot be found:\n{card_dir}"
            )
            return
        use_latest = bool(self.use_latest.get())
        self._start_task("analyze", lambda backend: {
            "run": backend.create_session(card_dir, use_latest=use_latest),
            "sessions": backend.list_sessions(),
        })

    def _resume(self) -> None:
        path = self._selected_session()
        if path is None:
            self.messagebox.showinfo("Choose saved work", "Select a card run first.")
            return
        self._start_task("resume", lambda backend: {
            "run": backend.resume_session(path),
            "sessions": backend.list_sessions(),
        })

    def _open_review(self) -> None:
        if self.review_process is not None and self.review_process.poll() is None:
            self.status_var.set("The review window is already open.")
            return
        path = self._selected_session()
        if path is None:
            self.messagebox.showinfo("Choose saved work", "Select a card run first.")
            return
        self._start_task("review_command", lambda backend: backend.review_command(path))

    def _train(self) -> None:
        if self.review_process is not None and self.review_process.poll() is None:
            self.messagebox.showinfo(
                "Finish reviewing first",
                "Stop the review window before training the next model.",
            )
            return
        path = self._selected_session()
        if path is None:
            self.messagebox.showinfo("Choose saved work", "Select a card run first.")
            return
        selection = self.session_list.curselection()
        if selection and self.sessions[selection[0]].get("status") == "TRAINED":
            self.messagebox.showinfo(
                "Already trained", "This review already produced a new model. Choose another card run."
            )
            return
        self._start_task("train", lambda backend: {
            "train": backend.train(path),
            "sessions": backend.list_sessions(),
        })

    def _open_release_page(self) -> None:
        try:
            url = self.backend_module.release_page_url()
            if not url:
                raise ValueError("No release page is configured.")
            webbrowser.open(url)
        except Exception as exc:
            self.messagebox.showerror("Cannot open release page", str(exc))

    def _launch_review(self, command) -> None:
        if not isinstance(command, (list, tuple)) or not command:
            raise ValueError("The review command is empty. Select a completed card run.")
        cmd = [str(item) for item in command]
        env = self.backend_module._clean_env()
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1, env=env,
            cwd=Path(self.backend_module.__file__).resolve().parent,
            start_new_session=True,
        )
        self.review_process = process
        self.status_var.set("Review is opening in your browser. Keep this app open.")
        self._append_log("Review started. The browser page may take a few seconds to appear.")
        self._refresh_buttons()

        def read_output() -> None:
            try:
                if process.stdout is not None:
                    for line in process.stdout:
                        self.events.put(("log", line.rstrip("\n")))
            finally:
                self.events.put(("review_exit", process, process.wait()))

        threading.Thread(
            target=read_output, name="compag-review-output", daemon=True
        ).start()

    def _stop_review(self) -> None:
        process = self.review_process
        if process is None or process.poll() is not None:
            self.status_var.set("No review window is running.")
            self._refresh_buttons()
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            process.terminate()
        self.status_var.set("Stopping review…")
        self.root.after(3000, lambda: self._force_stop_review(process))

    @staticmethod
    def _force_stop_review(process) -> None:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()

    def _set_cards(self, cards) -> None:
        example_cards = self._backend().example / "cards"
        self.card_paths = [example_cards / str(value) for value in cards or []]
        self.card_list.delete(0, "end")
        for path in self.card_paths:
            self.card_list.insert("end", path.name)
        if self.card_paths:
            self.card_list.selection_set(0)

    def _set_sessions(self, sessions, select_path: str | None = None) -> None:
        previous = select_path or (
            str(self._selected_session()) if self._selected_session() else None
        )
        self.sessions = [item for item in sessions or [] if isinstance(item, dict)]
        self.session_list.delete(0, "end")
        select_index = None
        for index, item in enumerate(self.sessions):
            path = Path(str(item.get("path", "")))
            card = item.get("card_id") or path.name
            status = item.get("status") or "saved"
            self.session_list.insert("end", f"{card}  ·  {status}")
            if previous and str(path) == previous:
                select_index = index
        if select_index is None and self.sessions:
            select_index = 0
        if select_index is not None:
            self.session_list.selection_set(select_index)

    def _handle_task_result(self, name, result) -> None:
        if name == "review_command":
            self._launch_review(result)
            return
        if isinstance(result, dict):
            if "cards" in result:
                self._set_cards(result["cards"])
            if "sessions" in result:
                run = result.get("run")
                preferred = None
                if isinstance(run, dict):
                    preferred = run.get("path") or run.get("session_path")
                self._set_sessions(result["sessions"], str(preferred) if preferred else None)
            summary = result.get("release") or result.get("setup") or result.get("run") or result.get("train")
        else:
            summary = result
        if name == "check":
            self.status_var.set("Release files checked. Install when ready.")
        elif name == "setup":
            self.status_var.set("COMPAG is ready. Choose a card to analyze.")
        elif name == "analyze":
            self.status_var.set("Card analysis finished. Select its saved work and open review.")
        elif name == "resume":
            self.status_var.set("Saved work is ready. Open review to continue.")
        elif name == "train":
            self.status_var.set("Training finished. The latest model is ready for the next card.")
            self.use_latest.set(True)
        elif name == "refresh":
            self.status_var.set("Cards and saved work are up to date.")
        if summary is not None and name != "refresh":
            self._append_log(json.dumps(summary, ensure_ascii=False, default=str))

    def _drain_events(self) -> None:
        if self.closed:
            return
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "log":
                    self._append_log(event[1])
                elif kind == "task_done":
                    _kind, name, result, error = event
                    self.busy = False
                    if error is not None:
                        self.status_var.set(f"Could not finish: {error}")
                        self._append_log(f"ERROR: {error}")
                        self.messagebox.showerror(
                            "COMPAG needs your attention",
                            f"{error}\n\nCheck the selected folders and the progress log below.",
                        )
                    else:
                        try:
                            self._handle_task_result(name, result)
                        except Exception as exc:
                            self.status_var.set(f"Could not finish: {exc}")
                            self._append_log(f"ERROR: {exc}")
                            self.messagebox.showerror("Cannot open review", str(exc))
                    self._refresh_buttons()
                elif kind == "review_exit":
                    _kind, process, returncode = event
                    if process is self.review_process:
                        self.review_process = None
                        if returncode == 0 or returncode in (-signal.SIGTERM, -signal.SIGKILL):
                            self.status_var.set("Review closed. You can continue or train the next model.")
                        else:
                            self.status_var.set(
                                f"Review stopped with an error ({returncode}). See the progress log."
                            )
                        self._refresh_buttons()
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", str(message) + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _refresh_buttons(self) -> None:
        working = self.busy
        selection = self.session_list.curselection()
        selected_status = self.sessions[selection[0]].get("status") if selection else None
        review_running = self.review_process is not None and self.review_process.poll() is None
        for control in self.path_controls:
            control.state(["disabled"] if working or review_running else ["!disabled"])
        for button in (self.check_button, self.refresh_button):
            button.state(["disabled"] if working else ["!disabled"])
        for button in (self.install_button, self.analyze_button):
            button.state(["disabled"] if working or review_running else ["!disabled"])
        self.resume_button.state(
            ["disabled"] if working or review_running or selected_status not in
            {"PREPARING", "ANALYZING", "INTERRUPTED"} else ["!disabled"]
        )
        self.review_button.state(
            ["disabled"] if working or review_running or selected_status not in
            {"SCORED", "TRAINED"} else ["!disabled"]
        )
        self.train_button.state(
            ["disabled"] if working or review_running or selected_status != "SCORED"
            else ["!disabled"]
        )
        self.stop_button.state(["!disabled"] if review_running else ["disabled"])

    def _close(self) -> None:
        if self.busy:
            self.messagebox.showinfo(
                "Step in progress",
                "Please wait for the current step to finish before closing COMPAG. "
                "If it is interrupted by the computer, reopen this app and continue the saved work.",
            )
            return
        if self.review_process is not None and self.review_process.poll() is None:
            if not self.messagebox.askyesno(
                "Close COMPAG?", "The review window is still running. Stop it and close COMPAG?"
            ):
                return
            process = self.review_process
            self._stop_review()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._force_stop_review(process)
        self.closed = True
        self.root.destroy()


def launch_gui(backend_module) -> bool:
    """Open the desktop launcher; return False if Tk or a display is unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        root = tk.Tk()
    except Exception:
        return False
    _EasyStartWindow(root, backend_module, tk, ttk, filedialog, messagebox)
    root.mainloop()
    return True
