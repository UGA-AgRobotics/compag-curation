"""Symlink-safe output-directory contract for source-backed trainers."""

from __future__ import annotations

from pathlib import Path

from ..contracts import ContractError


def prepare_source_output_directory(path: Path) -> Path:
    """Create or validate one configured output directory without following links."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ContractError("source training output directory must be absolute")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise ContractError("source training output parent is unavailable") from exc
    if parent != path.parent or path.parent.is_symlink():
        raise ContractError("source training output parent must be symlink-free")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir():
            raise ContractError("source training output must be a non-symlink directory")
    else:
        try:
            path.mkdir(mode=0o700)
        except OSError as exc:
            raise ContractError("source training output directory could not be created") from exc
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ContractError("source training output directory is unavailable") from exc
    if resolved != path or path.is_symlink() or not path.is_dir():
        raise ContractError("source training output directory must remain symlink-free")
    return path


__all__ = ["prepare_source_output_directory"]
