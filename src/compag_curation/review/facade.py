"""Lazy public facade for web, native Ubuntu, and status review modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from compag_curation.public_io import PublicIOError
from compag_curation.review.session import ReviewSession


def run_review_ui(
    mode: str,
    *,
    run: Path | None = None,
    selection_root: Path | None = None,
    stage60_root: Path | None = None,
    images: Path | None = None,
    output: Path,
    state: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    target_label: str = "Target",
    non_target_label: str = "Non-target",
    model_preset: str | None = None,
) -> tuple[dict[str, Any], int]:
    if mode not in {"web", "desktop", "status"}:
        raise PublicIOError("review UI mode is unsupported")
    if (run is None) == (selection_root is None):
        raise PublicIOError(
            "review UI requires exactly one Stage-20 run or active-learning selection"
        )
    if model_preset is not None and (
        target_label != "CJ" or non_target_label != "Non-CJ"
    ):
        raise PublicIOError(
            "Project 1/CJ transfer assistance fixes the labels to CJ and Non-CJ"
        )
    if run is not None:
        if stage60_root is not None or images is not None:
            raise PublicIOError(
                "Stage-60 and image inputs are only valid for round review"
            )
        opened = ReviewSession.open(
            run,
            output,
            state,
            model_preset=model_preset,
        )
    else:
        if model_preset is not None:
            raise PublicIOError(
                "published model presets are valid only for Stage-20 run review"
            )
        if stage60_root is None or images is None:
            raise PublicIOError(
                "active-learning round review requires Stage 60 and original images"
            )
        from compag_curation.review.round_session import RoundReviewSession

        assert selection_root is not None
        opened = RoundReviewSession.open(
            selection_root,
            stage60_root,
            images,
            output,
            state,
        )
    with opened as session:
        if mode == "status":
            return session.summary(), 0
        if mode == "desktop":
            from compag_curation.review.desktop import run_desktop_review

            return run_desktop_review(
                session,
                target_label=target_label,
                non_target_label=non_target_label,
            ), 0
        from compag_curation.review.web import run_web_review

        return run_web_review(
            session,
            host=host,
            port=port,
            open_browser=open_browser,
            target_label=target_label,
            non_target_label=non_target_label,
        ), 0


__all__ = ["run_review_ui"]
