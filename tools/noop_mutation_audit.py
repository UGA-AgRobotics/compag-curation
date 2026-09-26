#!/usr/bin/env python3
"""Kill one isolated no-op mutant for each structurally mapped implementation."""

from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


RUNNER = r"""
import os, sys, unittest
root, node = sys.argv[1], sys.argv[2]
sys.path[:0] = [root, root + '/src']
os.environ['COMPAG_PUBLIC_ROOT'] = root
suite = unittest.defaultTestLoader.loadTestsFromName(node)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
"""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def mapped_symbols(root: Path) -> list[tuple[str, str]]:
    path = root / "manifests/OPERATIONAL_CELL_TRACEABILITY.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped = sorted({(row["implementation_symbol"], row["test_node_id"]) for row in rows})
    if len(rows) != 84 or len(grouped) != 15:
        raise RuntimeError("traceability must map 84 rows to exactly 15 symbol/test pairs")
    if any(not node.startswith("tests.test_mapped_symbol_contracts.MappedSymbolContractTests.") for _, node in grouped):
        raise RuntimeError("mapped symbol lacks a dedicated contract test")
    return grouped


def source_path(root: Path, qualified_symbol: str) -> tuple[Path, str]:
    module, symbol = qualified_symbol.split(":", 1)
    stem = root / "src" / Path(*module.split("."))
    file_path = stem.with_suffix(".py")
    if not file_path.is_file():
        file_path = stem / "__init__.py"
    if not file_path.is_file():
        raise RuntimeError(f"mapped module source is unavailable: {module}")
    return file_path, symbol


def no_op_bytes(path: Path, symbol: str) -> tuple[bytes, str]:
    original = path.read_bytes()
    source = original.decode("utf-8")
    tree = ast.parse(source, filename=str(path))
    targets = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == symbol]
    if len(targets) != 1 or not targets[0].body:
        raise RuntimeError(f"mapped symbol is not one unique top-level function: {symbol}")
    target = targets[0]
    first = target.body[0].lineno - 1
    last = target.end_lineno
    lines = source.splitlines(keepends=True)
    body_line = lines[first]
    indent = body_line[: len(body_line) - len(body_line.lstrip())]
    mutant_source = "".join([*lines[:first], f"{indent}return None\n", *lines[last:]])
    compile(mutant_source, str(path), "exec")
    mutant = mutant_source.encode("utf-8")
    diff = "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            mutant_source.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
        )
    )
    return mutant, diff


def run_node(root: Path, node: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", RUNNER, str(root), node],
        cwd=str(root),
        env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
        text=False,
        capture_output=True,
        check=False,
    )
    return {
        "argv": ["<CURRENT_PYTHON>", "-I", "-S", "-B", "-c", "<FIXED_RUNNER>", "<TREE>", node],
        "returncode": completed.returncode,
        "stdout_size": len(completed.stdout),
        "stdout_sha256": sha256_bytes(completed.stdout),
        "stderr_size": len(completed.stderr),
        "stderr_sha256": sha256_bytes(completed.stderr),
        "stderr_text": completed.stderr.decode("utf-8", errors="replace"),
    }


def assertion_excerpt(stderr_text: str) -> str:
    matches = re.findall(r"AssertionError(?:: ([^\n]+))?", stderr_text)
    if not matches:
        raise RuntimeError("mutant failure did not expose an assertion")
    return matches[0] or "AssertionError"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    pairs = mapped_symbols(root)
    records: list[dict[str, object]] = []
    canonical_before: dict[str, str] = {}
    for qualified_symbol, _test_node in pairs:
        path, _symbol = source_path(root, qualified_symbol)
        canonical_before[path.relative_to(root).as_posix()] = sha256_file(path)
    for index, (qualified_symbol, test_node) in enumerate(pairs, start=1):
        with tempfile.TemporaryDirectory(prefix=f"compag-mutant-{index:02d}-") as directory:
            candidate = Path(directory) / "repository"
            shutil.copytree(root, candidate, copy_function=shutil.copy2)
            path, symbol = source_path(candidate, qualified_symbol)
            before = path.read_bytes()
            before_sha = sha256_bytes(before)
            baseline = run_node(candidate, test_node)
            if baseline["returncode"] != 0:
                raise RuntimeError(f"baseline contract test failed for {qualified_symbol}")
            mutant, diff = no_op_bytes(path, symbol)
            path.write_bytes(mutant)
            mutant_sha = sha256_file(path)
            mutated = run_node(candidate, test_node)
            failure_text = str(mutated["stderr_text"])
            if (
                mutated["returncode"] == 0
                or "AssertionError" not in failure_text
                or "ERROR:" in failure_text
            ):
                raise RuntimeError(f"no-op mutant survived or failed incorrectly: {qualified_symbol}")
            failure_excerpt = assertion_excerpt(failure_text)
            path.write_bytes(before)
            restoration_sha = sha256_file(path)
            if restoration_sha != before_sha:
                raise RuntimeError(f"mutation restoration hash mismatch: {qualified_symbol}")
            restored = run_node(candidate, test_node)
            if restored["returncode"] != 0:
                raise RuntimeError(f"restored contract test failed: {qualified_symbol}")
            records.append(
                {
                    "ordinal": index,
                    "module_symbol": qualified_symbol,
                    "relative_source": path.relative_to(candidate).as_posix(),
                    "test_node_id": test_node,
                    "baseline_status": "PASS",
                    "source_sha256_before": before_sha,
                    "mutation": "REPLACE_FUNCTION_BODY_WITH_RETURN_NONE",
                    "mutation_diff": diff,
                    "mutation_sha256": mutant_sha,
                    "mutant_test_returncode": mutated["returncode"],
                    "failed_assertion": failure_excerpt,
                    "mutant_stdout_sha256": mutated["stdout_sha256"],
                    "mutant_stderr_sha256": mutated["stderr_sha256"],
                    "restoration_sha256": restoration_sha,
                    "restored_test_status": "PASS",
                    "process_tripwire_assertion": "PASS_BY_BASELINE_AND_RESTORED_TEST",
                }
            )
    full_suite = run_node(root, "tests.test_mapped_symbol_contracts")
    if full_suite["returncode"] != 0:
        raise RuntimeError("complete restored mapped-symbol suite failed")
    canonical_after = {
        relative: sha256_file(root / relative) for relative in sorted(canonical_before)
    }
    canonical_unchanged = canonical_after == canonical_before
    if not canonical_unchanged:
        raise RuntimeError("canonical mapped-symbol source changed during mutation audit")
    result = {
        "schema": "compag-noop-mutation-audit/v1",
        "status": "PASS",
        "implementation_symbol_contract_tests": "15/15",
        "noop_mutations_killed": "15/15",
        "noop_mutations_survived": 0,
        "mutation_strategy": "FRESH_NONSTACKED_TREE_PER_SYMBOL",
        "canonical_mapped_source_unchanged": canonical_unchanged,
        "canonical_source_sha256_before": canonical_before,
        "canonical_source_sha256_after": canonical_after,
        "complete_restored_suite_status": "PASS",
        "complete_restored_suite_stdout_sha256": full_suite["stdout_sha256"],
        "complete_restored_suite_stderr_sha256": full_suite["stderr_sha256"],
        "tripwire_evidence": "EACH_DEDICATED_TEST_ASSERTS_ZERO_INJECTED_RUNNER_EVENTS",
        "records": records,
        "tool_sha256": sha256_file(Path(__file__)),
    }
    encoded = stable_json(result)
    if args.record is not None:
        args.record.write_text(encoded, encoding="utf-8")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
