#!/usr/bin/env python3
"""Fail-closed shim for the retired mutating post-upload finalizer.

Repository metadata must be finalized before the release commit.  Use the
read-only ``tools/release_provenance_verifier.py`` after committing and tagging
the finalized source.  No historical mutating implementation remains here.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Sequence


sys.dont_write_bytecode = True

DEPRECATION_ERROR = (
    "DEPRECATED_MUTATING_FINALIZER_DISABLED: use tools/release_provenance_verifier.py "
    "after committing and tagging the finalized tree"
)


class FinalizationError(RuntimeError):
    """Raised for every attempted use of the retired mutating operation."""


def finalize(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Refuse the retired operation before reading or writing any path."""

    raise FinalizationError(DEPRECATION_ERROR)


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    print(
        json.dumps({"error": DEPRECATION_ERROR, "status": "FAIL"}, sort_keys=True, separators=(",", ":")),
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
