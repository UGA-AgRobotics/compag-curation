"""Resolution of path-bearing input fields written by the old code (profile only-codes-compat-v1).

The old notebook stores absolute machine paths inside the COCO files (``images[].file_name``).
Two readers exist in the reference source and each keeps its own contract:

* pack (cell 3)     -- ``fn if fn.is_absolute() else images_root / fn``
* features (cell 4) -- ``images_dir / Path(fn).name`` (the directory part is ignored *deliberately*)

With **no applicable mapping rule** a reader offers exactly one candidate: the one the reference
computes.  The feature exporter therefore reads ``images_dir / Path(file_name).name`` even when
``file_name`` carries an absolute directory or a relative sub-directory, and even when that other
file exists; the pack reader reads the recorded absolute path (or ``images_root / file_name``) and
never falls back to a same-named file elsewhere.  Existence never reorders candidates, so two files
with the same name cannot change which image is read.

Relocation (``--map OLD=NEW``) stays supported under an explicit mapped-reader policy: a rule is
applied once, on whole path components, longest rule first, and only to the recorded logical path.
When a rule applies, the mapped path is the first candidate and this reader's reference candidate is
the second; both attempts are recorded with the rule, the selected path and its hash.

Windows-style names are not part of the reference contract: the reader offers the reference
candidate (whose POSIX name is the whole string, so it does not resolve) and records
``UNSUPPORTED_WINDOWS_PATH``.  Such a value is read only when an explicit rule maps it, and that read
is recorded as ``MAPPED_BY_EXPLICIT_RULE`` with ``follows_reference_contract = false``: it is an
explicitly requested relocation, never a source-identical read.

Every attempt is recorded (candidates, rule, resolved path, identity, outcome, and what the
reference reader would have read) so a run manifest explains which bytes were read and why.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .state import PathMapper

WINDOWS_ABS = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")

# candidate strategies, in the order they may be offered
MAPPED = "MAPPED_BY_EXPLICIT_RULE"
ORIGINAL_ABSOLUTE = "RECORDED_ABSOLUTE_PATH"
LEGACY_BASENAME_JOIN = "LEGACY_BASENAME_JOIN"      # reference behaviour of the feature-export cell
RELATIVE_JOIN = "RELATIVE_JOIN"                     # reference behaviour for relative file names

READ_OK = "RESOLVED"
NOT_FOUND = "NOT_FOUND"
UNSUPPORTED_WINDOWS_PATH = "UNSUPPORTED_WINDOWS_PATH"


def sha256_file(p: Path, *, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        read = 0
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
            read += len(chunk)
            if limit_bytes is not None and read >= limit_bytes:
                break
    return h.hexdigest()


@dataclass
class Resolution:
    """One resolved input field, with every attempt that was made."""

    logical: str
    kind: str                                  # RELATIVE | ABSOLUTE | WINDOWS_ABSOLUTE
    outcome: str                               # RESOLVED | NOT_FOUND | UNSUPPORTED_WINDOWS_PATH
    strategy: str | None = None
    rule: dict[str, str] | None = None
    resolved: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    identity: dict[str, Any] | None = None
    reference_candidate: str | None = None       # what the reference reader would have read
    follows_reference_contract: bool | None = None

    @property
    def ok(self) -> bool:
        return self.outcome == READ_OK

    @property
    def path(self) -> Path | None:
        return Path(self.resolved) if self.resolved else None

    def to_dict(self) -> dict[str, Any]:
        d = {"logical": self.logical, "kind": self.kind, "outcome": self.outcome, "strategy": self.strategy,
             "rule": self.rule, "resolved": self.resolved, "candidates": self.candidates,
             "reference_candidate": self.reference_candidate,
             "follows_reference_contract": self.follows_reference_contract}
        if self.identity:
            d["identity"] = self.identity
        return d


class InputResolver:
    """Resolve ``file_name`` style fields for one reader.

    ``legacy_basename_join`` selects the feature-export reader (cell 4); the pack reader (cell 3)
    leaves it off and therefore never silently picks a same-named file from another directory.
    """

    def __init__(self, base: Path | str, *, mapper: PathMapper | None = None,
                 legacy_basename_join: bool = False, record_identity: bool = True,
                 identity_limit_bytes: int | None = None) -> None:
        self.base = Path(base)
        self.mapper = mapper or PathMapper([])
        self.legacy_basename_join = bool(legacy_basename_join)
        self.record_identity = bool(record_identity)
        self.identity_limit_bytes = identity_limit_bytes
        self.records: list[Resolution] = []

    # -- rule application ------------------------------------------------
    def _rule_for(self, value: str) -> tuple[dict[str, str], str] | None:
        """Longest matching prefix rule, compared on whole path components."""

        probe = value.replace("\\", "/") if WINDOWS_ABS.match(value) else value
        mapped = self.mapper.map(probe)
        if mapped == probe:
            return None
        best = None
        for old, new in self.mapper.rules:
            o = old.rstrip("/").replace("\\", "/")
            if probe == o or probe.startswith(o + "/"):
                if best is None or len(o) > len(best[0]):
                    best = (o, new.rstrip("/"))
        if best is None:
            return None
        return {"old_prefix": best[0], "new_prefix": best[1]}, mapped

    # -- resolution ------------------------------------------------------
    def _reference_candidate(self, value: str, kind: str) -> tuple[str, Path]:
        """The single path the reference reader computes for this value."""

        if self.legacy_basename_join:                      # feature export (cell 4)
            # exactly what the reference computes on this platform, Windows-style values included
            # (their POSIX "name" is the whole string, so such a value simply does not resolve)
            return LEGACY_BASENAME_JOIN, self.base / PurePosixPath(value).name
        if kind == "ABSOLUTE":                             # pack (cell 3)
            return ORIGINAL_ABSOLUTE, Path(value)
        if kind == "WINDOWS_ABSOLUTE":
            return ORIGINAL_ABSOLUTE, Path(value)
        return RELATIVE_JOIN, self.base / value

    def resolve(self, file_name_value: str) -> Resolution:
        value = str(file_name_value)
        kind = ("WINDOWS_ABSOLUTE" if WINDOWS_ABS.match(value)
                else "ABSOLUTE" if PurePosixPath(value).is_absolute() else "RELATIVE")
        candidates: list[dict[str, Any]] = []
        rule_hit = self._rule_for(value)

        def offer(strategy: str, path: Path, rule: dict[str, str] | None = None) -> dict[str, Any]:
            rec = {"strategy": strategy, "path": str(path), "exists": path.exists()}
            if rule:
                rec["rule"] = rule
            candidates.append(rec)
            return rec

        if rule_hit:
            # explicit relocation: the mapped path is tried first, then this reader's own
            # reference candidate; nothing else is offered
            offer(MAPPED, Path(rule_hit[1]), rule_hit[0])
        ref_strategy, ref_path = self._reference_candidate(value, kind)
        offer(ref_strategy, ref_path)

        chosen = next((c for c in candidates if c["exists"]), None)
        if chosen is None:
            outcome = (UNSUPPORTED_WINDOWS_PATH if kind == "WINDOWS_ABSOLUTE" and not rule_hit else NOT_FOUND)
            res = Resolution(logical=value, kind=kind, outcome=outcome, candidates=candidates)
        else:
            res = Resolution(logical=value, kind=kind, outcome=READ_OK, strategy=chosen["strategy"],
                             rule=chosen.get("rule"), resolved=chosen["path"], candidates=candidates)
            res.reference_candidate = str(ref_path)
            res.follows_reference_contract = chosen["strategy"] == ref_strategy
            if self.record_identity:
                p = Path(chosen["path"])
                st = p.stat()
                res.identity = {"size": st.st_size,
                                "sha256": sha256_file(p, limit_bytes=self.identity_limit_bytes)}
        self.records.append(res)
        return res

    # -- reporting -------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        by_strategy: dict[str, int] = {}
        for r in self.records:
            if r.ok:
                by_strategy[r.strategy or "?"] = by_strategy.get(r.strategy or "?", 0) + 1
        failures = [r.to_dict() for r in self.records if not r.ok]
        return {"base": str(self.base), "rules": self.mapper.to_dict(),
                "reader": "features" if self.legacy_basename_join else "pack",
                "legacy_basename_join": self.legacy_basename_join,
                "reference_contract_reads": sum(1 for r in self.records if r.follows_reference_contract),
                "relocated_reads": sum(1 for r in self.records if r.ok and r.strategy == MAPPED),
                "resolved": sum(1 for r in self.records if r.ok), "failed": len(failures),
                "by_strategy": by_strategy, "failures": failures}


def resolver_for(reader: str, base: Path | str, mapper: PathMapper | None, **kw: Any) -> InputResolver:
    """``reader`` is ``"pack"`` (cell 3 join) or ``"features"`` (cell 4 basename join)."""

    if reader not in ("pack", "features"):
        raise ValueError(f"unknown reader {reader!r}")
    return InputResolver(base, mapper=mapper, legacy_basename_join=(reader == "features"), **kw)


def receipts_payload(records: Iterable[Resolution]) -> list[dict[str, Any]]:
    return [r.to_dict() for r in records]
