
"""Retained SAM2 pipeline with lazy scientific imports."""

from __future__ import annotations

from typing import Any

__all__ = ["make_argparser", "run_pipeline"]


def __getattr__(name: str) -> Any:
    if name == "make_argparser":
        from .args import make_argparser

        return make_argparser
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(name)
