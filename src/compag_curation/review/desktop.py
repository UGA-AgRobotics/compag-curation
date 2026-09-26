"""Native Ubuntu/WSLg reviewer built on Tkinter and the shared safe session."""

from __future__ import annotations

import io
import math
from typing import Any

from compag_curation.public_io import PublicIOError
from compag_curation.review.session import ReviewSession


_CHOICE_LABELS = {
    "target": "Target · confident",
    "other": "Non-target · confident",
    "target_uncertain": "Target · uncertain",
    "other_uncertain": "Non-target · uncertain",
    "skip": "Skip · zero weight",
}


class DesktopReviewController:
    """UI-independent controller used by the Tk window and headless tests."""

    def __init__(self, session: ReviewSession) -> None:
        self.session = session
        status = session.summary()
        self.current_id = str(status["first_pending"] or session.index.order[0])
        self.state_filter = "pending"
        self.group: str | None = None
        self.auto_next = True

    def _scene(self, payload: dict[str, object]) -> dict[str, object]:
        """Attach the verified full-image scene after any session operation."""

        proposal = payload.get("proposal")
        if not isinstance(proposal, dict):
            raise PublicIOError("review operation did not return a proposal")
        proposal_id = proposal.get("proposal_id")
        if not isinstance(proposal_id, str):
            raise PublicIOError("review operation returned an invalid proposal ID")
        return self.session.scene_payload(proposal_id)

    def current(self) -> dict[str, object]:
        return self.session.scene_payload(self.current_id)

    def select(self, proposal_id: str) -> dict[str, object]:
        payload = self.session.scene_payload(proposal_id)
        self.current_id = proposal_id
        return payload

    def navigate(self, direction: int, *, force_filter: str | None = None) -> dict[str, object]:
        payload = self._scene(self.session.navigate(
            self.current_id,
            direction,
            state_filter=force_filter or self.state_filter,
            group=self.group,
        ))
        self.current_id = str(payload["proposal"]["proposal_id"])
        return payload

    def decide(self, choice: str) -> dict[str, object]:
        payload = self._scene(self.session.set_decision(self.current_id, choice))
        if self.auto_next and payload["summary"]["remaining"]:
            try:
                return self.navigate(1, force_filter="pending")
            except PublicIOError:
                pass
        return payload

    def undo(self) -> dict[str, object]:
        payload = self._scene(self.session.undo())
        self.current_id = str(payload["proposal"]["proposal_id"])
        return payload

    def confirm_model_remainder(self) -> dict[str, object]:
        """Durably confirm every pending published-model suggestion once."""

        payload = self._scene(self.session.confirm_model_remainder())
        self.current_id = str(payload["proposal"]["proposal_id"])
        return payload


class _TkReviewer:
    def __init__(
        self,
        root: Any,
        session: ReviewSession,
        tk: Any,
        ttk: Any,
        image_module: Any,
        image_tk: Any,
        *,
        target_label: str = "Target",
        non_target_label: str = "Non-target",
    ) -> None:
        self.root = root
        self.session = session
        self.controller = DesktopReviewController(session)
        self.tk = tk
        self.ttk = ttk
        self.Image = image_module
        self.ImageTk = image_tk
        self.target_label = target_label
        self.non_target_label = non_target_label
        self.assisted_mode = (
            session.summary().get("source_kind")
            == "STAGE20_PUBLISHED_MODEL_ASSIST"
        )
        self.choice_labels = {
            "target": f"{target_label} · confident",
            "other": f"{non_target_label} · confident",
            "target_uncertain": f"{target_label} · uncertain",
            "other_uncertain": f"{non_target_label} · uncertain",
            "skip": "Skip · zero weight",
        }
        self.payload: dict[str, object] | None = None
        self.scene: dict[str, object] | None = None
        self.source_image: Any = None
        self.photo: Any = None
        self.photo_origin: tuple[float, float] | None = None
        self.review_ready = False
        self.scale = 1.0
        self.offset_x = 40.0
        self.offset_y = 40.0
        self.drag_start: tuple[float, float, float, float] | None = None
        self._shortcut_keys_down: set[str] = set()
        self._build()
        self.show(self.controller.current(), reset=True)

    def _build(self) -> None:
        self.root.title("COMPAG Full-Image Review — Ubuntu")
        self.root.geometry("1240x820")
        self.root.minsize(980, 680)
        self.root.configure(bg="#07111f")
        style = self.ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except self.tk.TclError:
            pass
        style.configure("TFrame", background="#0e1a2b")
        style.configure("TLabel", background="#0e1a2b", foreground="#eef6ff")
        style.configure("Muted.TLabel", foreground="#9db0c8")
        style.configure("Title.TLabel", font=("TkDefaultFont", 15, "bold"))
        style.configure("TButton", padding=(10, 8))
        style.configure("TCombobox", fieldbackground="#132238", foreground="#07111f")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = self.ttk.Frame(self.root, padding=(18, 14))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        self.ttk.Label(header, text="COMPAG Full-Image Review", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.context_label = self.ttk.Label(
            header,
            text="Initial labeling · no model prediction",
            style="Muted.TLabel",
        )
        self.context_label.grid(row=1, column=0, sticky="w")
        self.progress_text = self.ttk.Label(header, text="Loading…", style="Muted.TLabel")
        self.progress_text.grid(row=0, column=1, padx=24, sticky="e")
        self.progress = self.ttk.Progressbar(header, maximum=100, length=260)
        self.progress.grid(row=0, column=2, padx=(0, 12))
        self.undo_button = self.ttk.Button(header, text="Undo  ↶", command=self.undo)
        self.undo_button.grid(row=0, column=3)

        body = self.ttk.Frame(self.root, padding=(14, 0, 14, 14))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        toolbar = self.ttk.Frame(body, padding=(0, 0, 0, 10))
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.ttk.Label(toolbar, text="Status").pack(side="left")
        self.filter_var = self.tk.StringVar(value="pending")
        filter_box = self.ttk.Combobox(toolbar, width=11, state="readonly", textvariable=self.filter_var, values=("pending", "all", "reviewed"))
        filter_box.pack(side="left", padx=(6, 16))
        filter_box.bind("<<ComboboxSelected>>", lambda _event: self._change_filter())
        self.ttk.Label(toolbar, text="Group").pack(side="left")
        self.group_var = self.tk.StringVar(value="All groups")
        group_box = self.ttk.Combobox(toolbar, width=24, state="readonly", textvariable=self.group_var, values=("All groups", *self.session.index.groups))
        group_box.pack(side="left", padx=(6, 16))
        group_box.bind("<<ComboboxSelected>>", lambda _event: self._change_group())
        self.outline_var = self.tk.BooleanVar(value=True)
        self.ttk.Checkbutton(
            toolbar,
            text="Mask outlines",
            variable=self.outline_var,
            command=self.redraw,
        ).pack(side="left", padx=(0, 10))
        self.bbox_var = self.tk.BooleanVar(value=False)
        self.ttk.Checkbutton(
            toolbar,
            text="Rectangles",
            variable=self.bbox_var,
            command=self.redraw,
        ).pack(side="left")
        self.context_var = self.tk.BooleanVar(value=False)
        self.context_button = self.ttk.Checkbutton(
            toolbar,
            text="Model context",
            variable=self.context_var,
            command=self.redraw,
            state="disabled",
        )
        self.context_button.pack(side="left", padx=(10, 0))
        self.follow_var = self.tk.BooleanVar(value=True)
        self.ttk.Checkbutton(
            toolbar,
            text="Follow selected",
            variable=self.follow_var,
            command=self._follow_now,
        ).pack(side="left", padx=(10, 0))
        self.search_var = self.tk.StringVar()
        search = self.ttk.Entry(toolbar, width=28, textvariable=self.search_var)
        search.pack(side="right", padx=(6, 0))
        search.bind("<Return>", lambda _event: self.search())
        self.ttk.Button(toolbar, text="Find ID", command=self.search).pack(side="right")

        viewer_frame = self.ttk.Frame(body)
        viewer_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        viewer_frame.columnconfigure(0, weight=1)
        viewer_frame.rowconfigure(0, weight=1)
        self.canvas = self.tk.Canvas(viewer_frame, bg="#050a12", highlightthickness=1, highlightbackground="#29405f", cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        controls = self.ttk.Frame(viewer_frame, padding=6)
        controls.place(x=8, y=8)
        self.ttk.Button(controls, text="＋", width=3, command=lambda: self.zoom(1.25)).pack(side="left")
        self.ttk.Button(controls, text="−", width=3, command=lambda: self.zoom(0.8)).pack(side="left", padx=4)
        self.ttk.Button(controls, text="Fit", command=self.fit).pack(side="left")
        self.ttk.Label(
            controls,
            text="Full image · click an outline",
            style="Muted.TLabel",
        ).pack(side="left", padx=(10, 4))
        self.canvas.bind("<Configure>", lambda _event: self.fit() if self.source_image is not None else None)
        self.canvas.bind("<ButtonPress-1>", self._pointer_down)
        self.canvas.bind("<B1-Motion>", self._pointer_drag)
        self.canvas.bind("<ButtonRelease-1>", self._pointer_up)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda _event: self.zoom(1.12))
        self.canvas.bind("<Button-5>", lambda _event: self.zoom(0.89))

        side = self.ttk.Frame(body, padding=14)
        side.grid(row=1, column=1, sticky="ns")
        side.configure(width=330)
        self.ttk.Label(side, text="SELECTED PROPOSAL", foreground="#42d3c8").pack(anchor="w")
        self.id_label = self.ttk.Label(side, text="—", style="Title.TLabel", wraplength=310)
        self.id_label.pack(anchor="w", pady=(5, 2))
        self.tile_label = self.ttk.Label(side, text="—", style="Muted.TLabel", wraplength=310)
        self.tile_label.pack(anchor="w", pady=(0, 12))
        self.facts_label = self.ttk.Label(side, text="—", wraplength=310, justify="left")
        self.facts_label.pack(anchor="w", pady=(0, 12))
        self.decision_label = self.ttk.Label(side, text="Pending", style="Muted.TLabel", wraplength=310)
        self.decision_label.pack(anchor="w", pady=(0, 14))

        actions = (
            (f"{self.target_label} · confident    [1 / A]", "target"),
            (f"{self.non_target_label} · confident    [2 / R]", "other"),
            (f"{self.target_label} · uncertain    [4 / W]", "target_uncertain"),
            (f"{self.non_target_label} · uncertain    [5 / U]", "other_uncertain"),
            ("Skip · zero weight    [3 / S]", "skip"),
        )
        self.decision_buttons: list[Any] = []
        for label, choice in actions:
            button = self.ttk.Button(
                side,
                text=label,
                command=lambda value=choice: self.decide(value),
                width=38,
            )
            button.pack(fill="x", pady=3)
            self.decision_buttons.append(button)
        nav = self.ttk.Frame(side, padding=(0, 10, 0, 0))
        nav.pack(fill="x")
        self.ttk.Button(nav, text="← Previous", command=lambda: self.navigate(-1)).pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.ttk.Button(nav, text="Next →", command=lambda: self.navigate(1)).pack(side="left", expand=True, fill="x", padx=(4, 0))
        self.auto_var = self.tk.BooleanVar(value=True)
        self.ttk.Checkbutton(side, text="Auto-next pending proposal", variable=self.auto_var, command=self._set_auto).pack(anchor="w", pady=(12, 8))
        self.bulk_confirm_button = self.ttk.Button(
            side,
            text="Confirm remaining XGB suggestions · weight 0.4",
            command=self.confirm_model_remainder,
        )
        if self.assisted_mode:
            self.bulk_confirm_button.pack(fill="x", pady=(5, 0))
        self.finalize_button = self.ttk.Button(side, text="Validate and create reviewed CSV", command=self.finalize)
        self.finalize_button.pack(side="bottom", fill="x", pady=(18, 0))

        key_map = {
            "1": "target", "a": "target", "2": "other", "r": "other",
            "4": "target_uncertain", "w": "target_uncertain",
            "5": "other_uncertain", "u": "other_uncertain", "3": "skip", "s": "skip",
        }
        for key, choice in key_map.items():
            self.root.bind(
                f"<KeyPress-{key}>",
                lambda event, value=choice: self._decision_shortcut(event, value),
            )
        self.root.bind("<KeyPress-Right>", lambda event: self._navigation_shortcut(event, 1))
        self.root.bind("<KeyPress-Left>", lambda event: self._navigation_shortcut(event, -1))
        self.root.bind("<KeyPress-0>", self._undo_shortcut)
        self.root.bind("<KeyPress-z>", self._undo_shortcut)
        self.root.bind("<KeyRelease>", self._shortcut_released)
        self.root.bind("<FocusOut>", lambda _event: self._shortcut_keys_down.clear())
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    @staticmethod
    def _editing_text(event: Any) -> bool:
        try:
            return event.widget.winfo_class() in {
                "Entry",
                "TEntry",
                "Text",
                "TCombobox",
                "Spinbox",
                "TSpinbox",
            }
        except (AttributeError, TypeError):
            return False

    def _unsafe_shortcut(self, event: Any) -> bool:
        """Reject modified shortcuts and a held key's auto-repeat events."""

        try:
            state = int(getattr(event, "state", 0))
        except (TypeError, ValueError):
            return True
        # X/Tk Control, Mod1 (Alt), and Mod2..Mod5 (including Super/Meta).
        # Shift and Caps Lock remain harmless so uppercase letter shortcuts work.
        if state & (0x04 | 0x08 | 0x10 | 0x20 | 0x40 | 0x80):
            return True
        key = str(getattr(event, "keysym", "")).casefold()
        keys = getattr(self, "_shortcut_keys_down", None)
        if keys is None:
            keys = set()
            self._shortcut_keys_down = keys
        if key and key in keys:
            return True
        if key:
            keys.add(key)
        return False

    def _shortcut_released(self, event: Any) -> None:
        key = str(getattr(event, "keysym", "")).casefold()
        keys = getattr(self, "_shortcut_keys_down", None)
        if key and keys is not None:
            keys.discard(key)

    def _decision_shortcut(self, event: Any, choice: str) -> str | None:
        if self._editing_text(event):
            return None
        if self._unsafe_shortcut(event):
            return "break"
        self.decide(choice)
        return "break"

    def _navigation_shortcut(self, event: Any, direction: int) -> str | None:
        if self._editing_text(event):
            return None
        if self._unsafe_shortcut(event):
            return "break"
        self.navigate(direction)
        return "break"

    def _undo_shortcut(self, event: Any) -> str | None:
        if self._editing_text(event):
            return None
        if self._unsafe_shortcut(event):
            return "break"
        self.undo()
        return "break"

    def _error(self, exc: BaseException) -> None:
        from tkinter import messagebox

        messagebox.showerror("COMPAG Human Review", str(exc), parent=self.root)

    def _set_auto(self) -> None:
        self.controller.auto_next = bool(self.auto_var.get())

    def _change_filter(self) -> None:
        self.controller.state_filter = self.filter_var.get()
        self.navigate(1)

    def _change_group(self) -> None:
        value = self.group_var.get()
        self.controller.group = None if value == "All groups" else value
        self.navigate(1)

    def search(self) -> None:
        try:
            payload = self.session.search(self.search_var.get())
            proposal = payload.get("proposal")
            if not isinstance(proposal, dict) or not isinstance(
                proposal.get("proposal_id"), str
            ):
                raise PublicIOError("review search returned an invalid proposal")
            self.show(self.controller.select(str(proposal["proposal_id"])))
        except (PublicIOError, OSError, ValueError) as exc:
            self._error(exc)

    def _clear_review_display(self) -> None:
        """Fail closed so no proposal can be labeled against a stale image."""

        self.review_ready = False
        self.payload = None
        self.scene = None
        self.source_image = None
        self.photo = None
        self.photo_origin = None
        self.drag_start = None
        self.canvas.delete("all")
        self.id_label.configure(text="IMAGE UNAVAILABLE", cursor="")
        self.tile_label.configure(text="The verified overlay could not be loaded.")
        self.facts_label.configure(text="Use Find, Previous, or Next to retry safely.")
        self.decision_label.configure(text="Review actions are disabled until an image is verified.")
        for button in self.decision_buttons:
            button.configure(state="disabled")
        self.undo_button.configure(state="disabled")
        self.bulk_confirm_button.configure(state="disabled")
        self.finalize_button.configure(state="disabled")

    @staticmethod
    def _integer_text(value: object) -> str:
        if isinstance(value, int) and not isinstance(value, bool):
            return f"{value:,}"
        return "Unavailable"

    @staticmethod
    def _metric_text(value: object) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "Unavailable"
        return f"{number:.4f}" if math.isfinite(number) else "Unavailable"

    def _stage_display(self, payload: dict[str, object]) -> dict[str, object]:
        """Build every user-visible value before a proposal becomes mutable."""

        try:
            proposal = payload["proposal"]
            tile = payload["tile"]
            scene = payload["scene"]
            summary = payload["summary"]
            if (
                not isinstance(proposal, dict)
                or not isinstance(tile, dict)
                or not isinstance(scene, dict)
                or not isinstance(summary, dict)
            ):
                raise TypeError("proposal, tile, scene, and summary must be objects")
            reviewed = summary["reviewed"]
            remaining = summary["remaining"]
            progress = float(summary["progress_percent"])
            if (
                not isinstance(reviewed, int)
                or isinstance(reviewed, bool)
                or reviewed < 0
                or not isinstance(remaining, int)
                or isinstance(remaining, bool)
                or remaining < 0
                or not math.isfinite(progress)
            ):
                raise ValueError("invalid review progress")
            width = scene.get("width")
            height = scene.get("height")
            if (
                not isinstance(width, int)
                or isinstance(width, bool)
                or width < 1
                or not isinstance(height, int)
                or isinstance(height, bool)
                or height < 1
                or scene.get("mime_type") != "image/png"
                or scene.get("selected_proposal_id") != proposal.get("proposal_id")
            ):
                raise ValueError("invalid full-image scene metadata")
            rows = scene["proposals"]
            if not isinstance(rows, list):
                raise TypeError("scene proposals must be a list")
            selected_rows = 0
            for row in rows:
                if (
                    not isinstance(row, dict)
                    or not isinstance(row.get("bbox"), list)
                    or len(row["bbox"]) != 4
                    or not isinstance(row.get("polygon"), list)
                    or len(row["polygon"]) % 2
                    or not isinstance(row.get("reviewable"), bool)
                    or not isinstance(row.get("selected"), bool)
                ):
                    raise ValueError("invalid full-image proposal geometry")
                if not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in (*row["bbox"], *row["polygon"])
                ):
                    raise ValueError("invalid full-image proposal geometry")
                if row["selected"]:
                    selected_rows += 1
                    if (
                        row.get("proposal_id") != proposal.get("proposal_id")
                        or not row["reviewable"]
                    ):
                        raise ValueError("selected scene proposal is not reviewable")
            if selected_rows != 1 or scene.get("proposal_count") != len(rows):
                raise ValueError("full-image scene proposal count is invalid")
            source_kind = summary.get("source_kind")
            is_round = source_kind == "ACTIVE_LEARNING_ROUND"
            is_assisted = source_kind == "STAGE20_PUBLISHED_MODEL_ASSIST"
            if is_assisted:
                scene_model = scene.get("model")
                if (
                    not isinstance(summary.get("can_confirm_model_remainder"), bool)
                    or not isinstance(scene_model, dict)
                    or scene_model.get("mode") != "PUBLISHED_TRANSFER_ASSIST"
                    or scene_model.get("model_suggestions_are_human_context") is not True
                ):
                    raise ValueError("invalid model-confirmation state")
                selected_model: dict[str, object] | None = None
                for row in rows:
                    model = row.get("model")
                    if not isinstance(model, dict):
                        raise ValueError("missing published-model scene metadata")
                    probability = float(model.get("xgb_p"))
                    uncertainty = float(model.get("uncertainty"))
                    if (
                        model.get("prediction") not in {0, 1}
                        or not math.isfinite(probability)
                        or not 0.0 <= probability <= 1.0
                        or not math.isfinite(uncertainty)
                        or not 0.0 <= uncertainty <= 0.5
                    ):
                        raise ValueError("invalid published-model scene metadata")
                    if row.get("selected"):
                        selected_model = model
                proposal_probability = float(proposal.get("xgb_p"))
                proposal_uncertainty = float(proposal.get("uncertainty"))
                if (
                    selected_model is None
                    or proposal.get("prediction") not in {0, 1}
                    or not math.isfinite(proposal_probability)
                    or not 0.0 <= proposal_probability <= 1.0
                    or not math.isfinite(proposal_uncertainty)
                    or not 0.0 <= proposal_uncertainty <= 0.5
                    or selected_model.get("prediction") != proposal.get("prediction")
                    or float(selected_model["xgb_p"]) != proposal_probability
                    or float(selected_model["uncertainty"]) != proposal_uncertainty
                ):
                    raise ValueError("selected published-model metadata differs")
            if is_round:
                prediction = self.target_label if proposal.get("prediction") == 1 else (
                    self.non_target_label if proposal.get("prediction") == 0 else "unavailable"
                )
                kept = "kept" if proposal.get("kept") == 1 else (
                    "not kept" if proposal.get("kept") == 0 else "unavailable"
                )
                rank = self._integer_text(proposal.get("selection_rank"))
                context = (
                    f"Active Learning round {self._integer_text(summary.get('round_number'))} "
                    "· model-guided shortlist · human decision only"
                )
                tile_text = (
                    f"{proposal.get('image_name', 'Unavailable')}\n"
                    f"Full image · rank {'#' + rank if rank != 'Unavailable' else rank} "
                    f"· model {prediction} · {kept}"
                )
                metric_text = (
                    f"XGB probability: {self._metric_text(proposal.get('xgb_p'))}\n"
                    f"Uncertainty: {self._metric_text(proposal.get('uncertainty'))}"
                )
            elif is_assisted:
                prediction = self.target_label if proposal.get("prediction") == 1 else (
                    self.non_target_label if proposal.get("prediction") == 0 else "unavailable"
                )
                rank = self._integer_text(proposal.get("selection_rank"))
                context = (
                    "Published XGBoost r92 transfer assist · verify corrections, "
                    "then explicitly confirm any remainder"
                )
                tile_text = (
                    f"{proposal.get('image_name', 'Unavailable')}\n"
                    f"Full image · uncertainty rank "
                    f"{'#' + rank if rank != 'Unavailable' else rank} "
                    f"· model {prediction}"
                )
                metric_text = (
                    f"XGB probability: {self._metric_text(proposal.get('xgb_p'))}\n"
                    f"Uncertainty: {self._metric_text(proposal.get('uncertainty'))}"
                )
            else:
                context = "Initial labeling · no model prediction"
                tile_text = (
                    f"{proposal.get('image_name', 'Unavailable')}\n"
                    f"Full image · {scene.get('proposal_count', 'Unavailable')} candidates"
                )
                metric_text = (
                    f"Predicted IoU: {self._metric_text(proposal.get('predicted_iou'))}\n"
                    f"Stability: {self._metric_text(proposal.get('stability_score'))}"
                )
            decision = proposal.get("decision")
            if decision is None:
                decision_text = "Pending — no human decision"
            elif isinstance(decision, dict):
                decision_text = (
                    f"Saved: {self.choice_labels.get(str(decision.get('choice')), 'Unavailable')} "
                    f"· label {decision.get('label', 'Unavailable')} "
                    f"· weight {decision.get('review_weight', 'Unavailable')}"
                )
            else:
                raise TypeError("decision must be an object or null")
            return {
                "proposal": proposal,
                "tile": tile,
                "scene": scene,
                "summary": summary,
                "context": context,
                "id": str(proposal.get("label_prefix", "Unavailable")),
                "proposal_id": str(proposal["proposal_id"]),
                "tile_text": tile_text,
                "facts": (
                    f"Position: {self._integer_text(proposal.get('position'))} / "
                    f"{self._integer_text(proposal.get('total'))}\n"
                    f"Group: {proposal.get('group_id', 'Unavailable')}\n{metric_text}"
                ),
                "decision": decision_text,
                "reviewed": reviewed,
                "remaining": remaining,
                "progress": progress,
                "mutable": summary.get("status") != "FINALIZED",
                "assisted": is_assisted,
                "can_confirm_model_remainder": bool(
                    summary.get("can_confirm_model_remainder", False)
                ),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise PublicIOError("review proposal display metadata is invalid") from exc

    def show(self, payload: dict[str, object], *, reset: bool = False) -> None:
        previous_id = (
            str(self.payload["proposal"]["proposal_id"])
            if self.payload is not None
            else None
        )
        previous_scene_id = (
            str(self.scene["scene_id"])
            if self.scene is not None
            else None
        )
        try:
            display = self._stage_display(payload)
            scene = display["scene"]
            if (
                previous_scene_id == str(scene["scene_id"])
                and self.source_image is not None
                and self.source_image.size
                == (int(scene["width"]), int(scene["height"]))
            ):
                next_source_image = self.source_image
            else:
                data = self.session.scene_bytes(str(scene["scene_id"]))
                image = self.Image.open(io.BytesIO(data))
                image.load()
                if image.size != (int(scene["width"]), int(scene["height"])):
                    raise PublicIOError("verified full-image scene dimensions changed")
                next_source_image = image.convert("RGB")
        except (OSError, ValueError, PublicIOError):
            if previous_id is not None:
                self.controller.current_id = previous_id
            self._clear_review_display()
            raise

        # Keep mutations disabled until all text and pixels have rendered.
        self.review_ready = False
        for button in self.decision_buttons:
            button.configure(state="disabled")
        self.undo_button.configure(state="disabled")
        self.bulk_confirm_button.configure(state="disabled")
        self.finalize_button.configure(state="disabled")
        try:
            self.payload = payload
            self.scene = display["scene"]
            self.source_image = next_source_image
            self.photo = None
            self.photo_origin = None
            self.controller.current_id = str(display["proposal_id"])
            self.context_label.configure(text=display["context"])
            self.id_label.configure(text=display["id"], cursor="hand2")
            self.tile_label.configure(text=display["tile_text"])
            self.facts_label.configure(text=display["facts"])
            self.decision_label.configure(text=display["decision"])
            self.progress_text.configure(
                text=f"{display['reviewed']:,} reviewed · {display['remaining']:,} remaining"
            )
            self.progress.configure(value=display["progress"])
            is_round = self.scene.get("kind") == "VERIFIED_STAGE60_ORIGINAL_FULL_IMAGE"
            if not is_round:
                self.context_var.set(False)
            self.context_button.configure(state=("normal" if is_round else "disabled"))
            if reset or previous_scene_id != str(self.scene["scene_id"]):
                self.fit()
            else:
                if self.follow_var.get():
                    self._ensure_selected_visible()
                self.redraw()
            self.review_ready = True
            mutable = bool(display["mutable"])
            for button in self.decision_buttons:
                button.configure(state=("normal" if mutable else "disabled"))
            self.undo_button.configure(state=("normal" if mutable else "disabled"))
            self.bulk_confirm_button.configure(
                state=(
                    "normal"
                    if display["assisted"]
                    and display["can_confirm_model_remainder"]
                    and display["remaining"] > 0
                    and mutable
                    else "disabled"
                )
            )
            self.finalize_button.configure(
                state=(
                    "normal"
                    if display["remaining"] == 0 and mutable
                    else "disabled"
                )
            )
        except Exception as exc:
            if previous_id is not None:
                self.controller.current_id = previous_id
            self._clear_review_display()
            if isinstance(exc, (PublicIOError, OSError, ValueError)):
                raise
            raise PublicIOError("review proposal display failed safely") from exc

    def fit(self) -> None:
        if self.source_image is None:
            return
        width = max(100, self.canvas.winfo_width())
        height = max(100, self.canvas.winfo_height())
        source_width, source_height = self.source_image.size
        self.scale = max(
            0.01,
            min(
                4.0,
                (width - 32) / source_width,
                (height - 32) / source_height,
            ),
        )
        self.offset_x = (width - source_width * self.scale) / 2
        self.offset_y = (height - source_height * self.scale) / 2
        self.redraw()

    def zoom(self, factor: float) -> None:
        if self.source_image is None:
            return
        before = self.scale
        self.scale = max(0.01, min(16.0, self.scale * factor))
        cx = self.canvas.winfo_width() / 2
        cy = self.canvas.winfo_height() / 2
        ratio = self.scale / before
        self.offset_x = cx - (cx - self.offset_x) * ratio
        self.offset_y = cy - (cy - self.offset_y) * ratio
        self.redraw()

    def _constrain_offsets(self) -> None:
        if self.source_image is None:
            return
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        rendered_width = self.source_image.size[0] * self.scale
        rendered_height = self.source_image.size[1] * self.scale
        margin = 24.0
        if rendered_width <= canvas_width - 2 * margin:
            self.offset_x = (canvas_width - rendered_width) / 2
        else:
            self.offset_x = min(
                margin,
                max(canvas_width - rendered_width - margin, self.offset_x),
            )
        if rendered_height <= canvas_height - 2 * margin:
            self.offset_y = (canvas_height - rendered_height) / 2
        else:
            self.offset_y = min(
                margin,
                max(canvas_height - rendered_height - margin, self.offset_y),
            )

    def _ensure_selected_visible(self) -> None:
        if self.source_image is None or self.scene is None:
            return
        selected = next(
            (row for row in self.scene["proposals"] if row.get("selected")),
            None,
        )
        if selected is None:
            raise PublicIOError("full-image scene has no selected candidate")
        x, y, width, height = (float(value) for value in selected["bbox"])
        canvas_width = max(1.0, float(self.canvas.winfo_width()))
        canvas_height = max(1.0, float(self.canvas.winfo_height()))
        padding = min(72.0, max(24.0, min(canvas_width, canvas_height) * 0.10))
        left = self.offset_x + x * self.scale
        top = self.offset_y + y * self.scale
        right = left + width * self.scale
        bottom = top + height * self.scale
        available_width = max(1.0, canvas_width - 2.0 * padding)
        available_height = max(1.0, canvas_height - 2.0 * padding)
        if right - left > available_width:
            self.offset_x = canvas_width / 2.0 - (x + width / 2.0) * self.scale
        elif left < padding:
            self.offset_x += padding - left
        elif right > canvas_width - padding:
            self.offset_x -= right - (canvas_width - padding)
        if bottom - top > available_height:
            self.offset_y = canvas_height / 2.0 - (y + height / 2.0) * self.scale
        elif top < padding:
            self.offset_y += padding - top
        elif bottom > canvas_height - padding:
            self.offset_y -= bottom - (canvas_height - padding)
        self._constrain_offsets()

    def _follow_now(self) -> None:
        if not self.follow_var.get():
            return
        try:
            self._ensure_selected_visible()
            self.redraw()
        except (PublicIOError, OSError, ValueError) as exc:
            self._error(exc)

    @staticmethod
    def _overlay_color(row: dict[str, object]) -> str:
        if row.get("selected"):
            return "#ffd166"
        choice = row.get("choice")
        if choice == "target":
            return "#35e39a"
        if choice == "other":
            return "#ff6878"
        if choice in {"target_uncertain", "other_uncertain"}:
            return "#d89cff"
        if choice == "skip":
            return "#8998aa"
        model = row.get("model")
        if isinstance(model, dict):
            if not row.get("reviewable"):
                return "#52657a"
            return "#55dfad" if model.get("prediction") == 1 else "#ff8b98"
        return "#66d4ff"

    def _display_scene_rows(self) -> list[dict[str, object]]:
        if self.scene is None:
            return []
        show_context = bool(self.context_var.get())
        return [
            row
            for row in self.scene["proposals"]
            if row.get("reviewable") or show_context
        ]

    def redraw(self) -> None:
        if self.source_image is None or self.payload is None or self.scene is None:
            return
        self._constrain_offsets()
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        image_width, image_height = self.source_image.size
        source_left = max(0, int(math.floor(-self.offset_x / self.scale)))
        source_top = max(0, int(math.floor(-self.offset_y / self.scale)))
        source_right = min(
            image_width,
            int(math.ceil((canvas_width - self.offset_x) / self.scale)),
        )
        source_bottom = min(
            image_height,
            int(math.ceil((canvas_height - self.offset_y) / self.scale)),
        )
        self.canvas.delete("all")
        if source_left >= source_right or source_top >= source_bottom:
            self.photo = None
            self.photo_origin = None
            return
        visible = self.source_image.crop(
            (source_left, source_top, source_right, source_bottom)
        )
        target_width = max(1, int(round(visible.size[0] * self.scale)))
        target_height = max(1, int(round(visible.size[1] * self.scale)))
        if visible.size != (target_width, target_height):
            visible = visible.resize(
                (target_width, target_height),
                self.Image.Resampling.LANCZOS,
            )
        self.photo = self.ImageTk.PhotoImage(visible)
        origin_x = self.offset_x + source_left * self.scale
        origin_y = self.offset_y + source_top * self.scale
        self.photo_origin = (origin_x, origin_y)
        self.canvas.create_image(origin_x, origin_y, image=self.photo, anchor="nw")

        rows = sorted(
            self._display_scene_rows(),
            key=lambda row: bool(row.get("selected")),
        )
        for row in rows:
            x, y, w, h = (float(value) for value in row["bbox"])
            if (
                x + w < source_left
                or y + h < source_top
                or x > source_right
                or y > source_bottom
            ):
                continue
            color = self._overlay_color(row)
            line_width = 4 if row.get("selected") else (2 if row.get("reviewable") else 1)
            tag = f"proposal:{row['proposal_id']}"
            polygon = row["polygon"]
            if self.outline_var.get() and len(polygon) >= 6:
                points: list[float] = []
                for index, value in enumerate(polygon):
                    offset = self.offset_x if index % 2 == 0 else self.offset_y
                    points.append(offset + float(value) * self.scale)
                self.canvas.create_polygon(
                    *points,
                    fill="",
                    outline=color,
                    width=line_width,
                    tags=(tag, "mask-outline"),
                )
            if self.bbox_var.get():
                self.canvas.create_rectangle(
                    self.offset_x + x * self.scale,
                    self.offset_y + y * self.scale,
                    self.offset_x + (x + w) * self.scale,
                    self.offset_y + (y + h) * self.scale,
                    outline=color,
                    width=line_width,
                    dash=(() if row.get("selected") else (5, 3)),
                    tags=(tag, "bbox"),
                )
            if row.get("selected"):
                self.canvas.create_text(
                    self.offset_x + x * self.scale,
                    self.offset_y + y * self.scale - 6,
                    text=str(row.get("label_prefix", "")),
                    anchor="sw",
                    fill="#ffd166",
                    font=("TkDefaultFont", 10, "bold"),
                )

    def _pointer_down(self, event: Any) -> None:
        self.drag_start = (event.x, event.y, self.offset_x, self.offset_y)

    def _pointer_drag(self, event: Any) -> None:
        if self.drag_start is None:
            return
        self.offset_x = self.drag_start[2] + event.x - self.drag_start[0]
        self.offset_y = self.drag_start[3] + event.y - self.drag_start[1]
        self.redraw()

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: list[object]) -> bool:
        if len(polygon) < 6 or len(polygon) % 2:
            return False
        vertices = [
            (float(px), float(py))
            for px, py in zip(polygon[0::2], polygon[1::2], strict=True)
        ]
        inside = False
        prior_x, prior_y = vertices[-1]
        for current_x, current_y in vertices:
            crosses = (current_y > y) != (prior_y > y)
            if crosses:
                intersection = (
                    (prior_x - current_x)
                    * (y - current_y)
                    / (prior_y - current_y)
                    + current_x
                )
                if x < intersection:
                    inside = not inside
            prior_x, prior_y = current_x, current_y
        return inside

    def _pointer_up(self, event: Any) -> None:
        if self.drag_start is None or self.payload is None or self.scene is None:
            return
        moved = math.hypot(event.x - self.drag_start[0], event.y - self.drag_start[1])
        self.drag_start = None
        if moved > 5:
            return
        x = (event.x - self.offset_x) / self.scale
        y = (event.y - self.offset_y) / self.scale
        containing: list[tuple[float, str]] = []
        nearest: tuple[float, str] | None = None
        for row in self.scene["proposals"]:
            if not row.get("reviewable"):
                continue
            bx, by, bw, bh = (float(value) for value in row["bbox"])
            polygon = row["polygon"]
            if self._point_in_polygon(x, y, polygon) or (
                len(polygon) < 6 and bx <= x <= bx + bw and by <= y <= by + bh
            ):
                containing.append((bw * bh, str(row["proposal_id"])))
            distance = math.hypot(x - (bx + bw / 2), y - (by + bh / 2))
            if nearest is None or distance < nearest[0]:
                nearest = (distance, str(row["proposal_id"]))
        proximity = max(8.0, min(45.0, 18.0 / self.scale))
        proposal_id = (
            min(containing)[1]
            if containing
            else (nearest[1] if nearest and nearest[0] <= proximity else None)
        )
        if proposal_id:
            try:
                self.show(self.controller.select(proposal_id))
            except (PublicIOError, OSError, ValueError) as exc:
                self._error(exc)

    def _wheel(self, event: Any) -> None:
        self.zoom(1.12 if event.delta > 0 else 0.89)

    def decide(self, choice: str) -> None:
        if not self.review_ready:
            return
        if choice == "skip":
            from tkinter import messagebox

            if not messagebox.askyesno(
                "Skip proposal with zero weight",
                "Skip this proposal?\n\n"
                "This explicitly saves label 0, action skip, and review weight 0.0. "
                "It will contribute zero training weight.",
                parent=self.root,
            ):
                return
        try:
            self.controller.auto_next = bool(self.auto_var.get())
            self.show(self.controller.decide(choice))
        except (PublicIOError, OSError, ValueError) as exc:
            self._error(exc)

    def navigate(self, direction: int) -> None:
        try:
            self.show(self.controller.navigate(direction))
        except (PublicIOError, OSError, ValueError) as exc:
            self._error(exc)

    def undo(self) -> None:
        if not self.review_ready:
            return
        try:
            self.show(self.controller.undo())
        except (PublicIOError, OSError, ValueError) as exc:
            self._error(exc)

    def confirm_model_remainder(self) -> None:
        """Require an explicit count/weight acknowledgement before bulk save."""

        if not self.review_ready or not self.assisted_mode:
            return
        try:
            summary = self.session.summary()
            remaining = summary.get("remaining")
            if (
                summary.get("source_kind") != "STAGE20_PUBLISHED_MODEL_ASSIST"
                or not isinstance(remaining, int)
                or isinstance(remaining, bool)
                or remaining < 1
                or summary.get("can_confirm_model_remainder") is not True
            ):
                raise PublicIOError(
                    "there are no pending published-model suggestions to confirm"
                )
            from tkinter import messagebox

            if not messagebox.askyesno(
                "Confirm remaining XGB suggestions",
                f"Confirm exactly {remaining:,} remaining XGB suggestions?\n\n"
                "Each remaining suggestion will be saved as an uncertain label "
                "with review weight 0.4. This is not the same as a proposal "
                "reviewed manually one by one.\n\n"
                "This single action is reversible with Undo.",
                parent=self.root,
            ):
                return
            self.show(self.controller.confirm_model_remainder())
        except (PublicIOError, OSError, ValueError) as exc:
            self._error(exc)

    def finalize(self) -> None:
        from tkinter import messagebox

        if not self.review_ready:
            return
        if not messagebox.askyesno(
            "Finalize human review",
            "Validate every decision and create the final reviewed CSV?\n\nThe output cannot be overwritten.",
            parent=self.root,
        ):
            return
        try:
            result = self.session.finalize()
            messagebox.showinfo("Review complete", f"Reviewed CSV created:\n{result['output']}", parent=self.root)
            self.show(self.controller.current())
        except (PublicIOError, OSError, RuntimeError, ValueError) as exc:
            self._error(exc)


def run_desktop_review(
    session: ReviewSession,
    *,
    target_label: str = "Target",
    non_target_label: str = "Non-target",
) -> dict[str, object]:
    try:
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
    except (ImportError, RuntimeError) as exc:
        raise PublicIOError(
            "Ubuntu desktop review requires Tk 8.6 and Pillow; use review-ui web on this environment"
        ) from exc
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise PublicIOError(
            "Ubuntu display is unavailable; use review-ui web or enable WSLg/X11"
        ) from exc
    _TkReviewer(
        root,
        session,
        tk,
        ttk,
        Image,
        ImageTk,
        target_label=target_label,
        non_target_label=non_target_label,
    )
    root.mainloop()
    return {"status": "CLOSED", "summary": session.summary()}


__all__ = ["DesktopReviewController", "run_desktop_review"]
