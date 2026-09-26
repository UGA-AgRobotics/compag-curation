#!/usr/bin/env python3
"""Syntax-check public Bash fences and prove selected blocks stop on failure.

The negative journeys execute the literal Markdown blocks with disposable
paths and controlled command stubs.  They never install dependencies, fetch
assets, or cross a scientific execution boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


DOCUMENTS = ("README.md", "QUICKSTART.md", "CONTRIBUTING.md")


@dataclass(frozen=True)
class BashBlock:
    relative_file: str
    heading: str
    line: int
    body: str


def _bash_blocks(root: Path) -> list[BashBlock]:
    paths = [*(root / name for name in DOCUMENTS), *sorted((root / "docs").glob("*.md"))]
    records: list[BashBlock] = []
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        heading = ""
        index = 0
        while index < len(lines):
            line = lines[index]
            if line.startswith("#"):
                heading = line.lstrip("#").strip()
            if line.strip() != "```bash":
                index += 1
                continue
            start = index + 2
            index += 1
            body: list[str] = []
            while index < len(lines) and lines[index].strip() != "```":
                body.append(lines[index])
                index += 1
            if index == len(lines):
                raise RuntimeError(f"unterminated Bash fence: {path.relative_to(root)}:{start}")
            records.append(
                BashBlock(
                    relative_file=path.relative_to(root).as_posix(),
                    heading=heading,
                    line=start,
                    body="\n".join(body) + "\n",
                )
            )
            index += 1
    return records


def _block(root: Path, relative_file: str, heading: str) -> BashBlock:
    matches = [
        block
        for block in _bash_blocks(root)
        if block.relative_file == relative_file and block.heading == heading
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one Bash block at {relative_file!r} heading {heading!r}; "
            f"observed {len(matches)}"
        )
    return matches[0]


def _meaningful_lines(block: BashBlock) -> list[str]:
    return [
        line.strip()
        for line in block.body.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _run_block(block: BashBlock, *, cwd: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/usr/bin/bash", "--noprofile", "--norc", "-c", block.body],
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )


def _write_executable(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _sidecar(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )


def _environment(home: Path, fake_bin: Path, **extra: str) -> dict[str, str]:
    return {
        "HOME": str(home),
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        **extra,
    }


def _full_gpu_venv(home: Path) -> Path:
    return home / ".local/share/compag-curation/venvs/science-gpu-full-1.9.3"


def _assert_failed_before(result: subprocess.CompletedProcess[str], condition: bool, label: str) -> None:
    if result.returncode == 0 or not condition:
        raise RuntimeError(
            f"{label} did not fail closed: rc={result.returncode}; "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}"
        )


def _syntax_and_directive_test(root: Path) -> dict[str, object]:
    blocks = _bash_blocks(root)
    complete = [block for block in blocks if len(_meaningful_lines(block)) > 1]
    missing = [
        f"{block.relative_file}:{block.line}"
        for block in complete
        if _meaningful_lines(block)[0] != "set -euo pipefail"
    ]
    if missing:
        raise RuntimeError(f"complete Bash blocks lack fail-fast directive: {missing!r}")
    environment_creation_blocks = [
        block
        for block in complete
        if " -m venv " in block.body or '"$MAMBA" create ' in block.body
    ]
    unsafe_umask = [
        f"{block.relative_file}:{block.line}"
        for block in environment_creation_blocks
        if _meaningful_lines(block)[1:2] != ["umask 0022"]
    ]
    if unsafe_umask:
        raise RuntimeError(
            "environment-creation Bash blocks must set umask 0022 immediately "
            f"after fail-fast setup: {unsafe_umask!r}"
        )
    syntax_errors: list[str] = []
    for block in blocks:
        checked = subprocess.run(
            ["/usr/bin/bash", "--noprofile", "--norc", "-n"],
            input=block.body,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
        if checked.returncode:
            syntax_errors.append(
                f"{block.relative_file}:{block.line}: {checked.stderr.strip()}"
            )
    if syntax_errors:
        raise RuntimeError(f"Bash fence syntax errors: {syntax_errors!r}")
    review_copies = [block for block in blocks if "review_request.csv" in block.body]
    if review_copies:
        raise RuntimeError(
            "documented Bash must launch the bound review UI instead of copying "
            "review_request.csv manually"
        )
    checksum_blocks = [block for block in blocks if "sha256sum --check" in block.body]
    strict_checksum_calls = sum(block.body.count("sha256sum --check --strict") for block in checksum_blocks)
    exact_target_guards = sum(block.body.count("grep -Eq '^[0-9a-f]{64}  ") for block in checksum_blocks)
    one_line_guards = sum(block.body.count('test "$(wc -l < "') for block in checksum_blocks)
    checksum_text = "\n".join(block.body for block in checksum_blocks)
    expected_release_target_guards = (
        "grep -Eq '^[0-9a-f]{64}  compag_curation-1\\.9\\.3-py3-none-any\\.whl$' \"$SIDECAR\"",
        "grep -Eq '^[0-9a-f]{64}  compag_curation-1\\.9\\.3\\.tar\\.gz$' \"$SDIST_SIDECAR\"",
        "grep -Eq '^[0-9a-f]{64}  constraints-bootstrap-cp312-linux-x86_64\\.txt$' \"$BOOTSTRAP_SIDECAR\"",
    )
    if (
        len(checksum_blocks) != 2
        or strict_checksum_calls != 3
        or exact_target_guards != strict_checksum_calls
        or one_line_guards != strict_checksum_calls
        or any(checksum_text.count(target) != 1 for target in expected_release_target_guards)
    ):
        raise RuntimeError("release checksum blocks must bind three exact one-line GNU sidecars")
    return {
        "bash_blocks": len(blocks),
        "complete_blocks": len(complete),
        "single_command_blocks": len(blocks) - len(complete),
        "syntax_template_blocks": sum(
            block.relative_file == "docs/DATA_ARTIFACT_CONTRACT.md"
            and block.heading == "Package Manifests"
            for block in blocks
        ),
        "exclusive_create_copy_blocks": len(review_copies),
        "strict_exact_sidecar_checks": strict_checksum_calls,
    }


def _checksum_failure_test(root: Path, temporary: Path) -> None:
    block = _block(root, "docs/INSTALL_CPU.md", "Base From A GitHub Release Wheel")
    home = temporary / "checksum-home"
    release = home / "Downloads/compag-curation-1.9.3"
    fake_bin = temporary / "checksum-bin"
    marker = temporary / "checksum-mutation"
    release.mkdir(parents=True)
    wheel = release / "compag_curation-1.9.3-py3-none-any.whl"
    wheel.write_bytes(b"controlled-wheel\n")
    wheel.with_name(f"{wheel.name}.sha256").write_text(
        f"{'0' * 64}  {wheel.name}\n",
        encoding="ascii",
    )
    _write_executable(
        fake_bin / "python3.12",
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" > {marker!s}\nexit 99\n",
    )
    result = _run_block(block, cwd=temporary, environment=_environment(home, fake_bin))
    venv = home / ".local/share/compag-curation/venvs/install-release-base-1.9.3"
    _assert_failed_before(result, not marker.exists() and not venv.exists(), "checksum failure")


def _sidecar_target_failure_test(root: Path, temporary: Path) -> None:
    block = _block(root, "docs/INSTALL_CPU.md", "Base From A GitHub Release Wheel")
    home = temporary / "sidecar-target-home"
    release = home / "Downloads/compag-curation-1.9.3"
    fake_bin = temporary / "sidecar-target-bin"
    marker = temporary / "sidecar-target-mutation"
    release.mkdir(parents=True)
    wheel = release / "compag_curation-1.9.3-py3-none-any.whl"
    wheel.write_bytes(b"controlled-wheel\n")
    decoy = release / "different-file.whl"
    decoy.write_bytes(b"controlled-decoy\n")
    decoy_digest = hashlib.sha256(decoy.read_bytes()).hexdigest()
    wheel.with_name(f"{wheel.name}.sha256").write_text(
        f"{decoy_digest}  {decoy.name}\n",
        encoding="ascii",
    )
    _write_executable(
        fake_bin / "python3.12",
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" > {marker!s}\nexit 99\n",
    )
    result = _run_block(block, cwd=temporary, environment=_environment(home, fake_bin))
    venv = home / ".local/share/compag-curation/venvs/install-release-base-1.9.3"
    _assert_failed_before(
        result,
        not marker.exists() and not venv.exists(),
        "sidecar target substitution",
    )


def _profile_installer_failure_test(root: Path, temporary: Path) -> None:
    blocks = [
        block
        for block in _bash_blocks(root)
        if block.relative_file == "docs/INSTALL_GPU.md"
        and block.heading == "Select A Preset And Install The One Wheel"
        and "--execute" in block.body
    ]
    if len(blocks) != 1:
        raise RuntimeError("expected one executable GPU profile-installer block")
    block = blocks[0]
    work = temporary / "profile-installer-root"
    home = temporary / "profile-installer-home"
    fake_bin = temporary / "profile-installer-bin"
    log = temporary / "profile-installer-log"
    work.mkdir()
    home.mkdir()
    (work / "pyproject.toml").write_text("[build-system]\n", encoding="ascii")
    _write_executable(
        fake_bin / "python3.12",
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" > {log!s}\nexit 41\n",
    )
    result = _run_block(block, cwd=work, environment=_environment(home, fake_bin))
    observed = log.read_text(encoding="utf-8") if log.exists() else ""
    venv = home / ".local/share/compag-curation/venvs/science-gpu-lite-1.9.3"
    _assert_failed_before(
        result,
        "tools/install_gpu_profile.py" in observed
        and "--preset lite" in observed
        and "--execute" in observed
        and not venv.exists(),
        "GPU profile-installer failure",
    )


def _install_permission_umask_test(root: Path, temporary: Path) -> None:
    """Prove the literal GPU install block restricts a collaborative umask."""

    installer_blocks = [
        selected
        for selected in _bash_blocks(root)
        if "tools/install_gpu_profile.py" in selected.body
    ]
    if not installer_blocks:
        raise RuntimeError("expected public GPU profile-installer Bash blocks")
    for selected in installer_blocks:
        if _meaningful_lines(selected)[:2] != ["set -euo pipefail", "umask 0022"]:
            raise RuntimeError(
                "public GPU install blocks must set umask 0022 immediately after "
                "fail-fast setup"
            )

    work = temporary / "install-umask-root"
    home = temporary / "install-umask-home"
    fake_bin = temporary / "install-umask-bin"
    log = temporary / "install-umask-log"
    work.mkdir()
    home.mkdir()
    (work / "pyproject.toml").write_text("[build-system]\n", encoding="ascii")
    execute_blocks = [
        selected
        for selected in installer_blocks
        if selected.relative_file == "docs/INSTALL_GPU.md"
        and selected.heading == "Select A Preset And Install The One Wheel"
        and "--execute" in selected.body
    ]
    if len(execute_blocks) != 1:
        raise RuntimeError("expected one executable GPU profile-installer block")
    _write_executable(
        fake_bin / "python3.12",
        "#!/bin/sh\n"
        "set -eu\n"
        f"umask > {str(log)!r}\n"
        "exit 53\n",
    )
    previous_umask = os.umask(0o002)
    try:
        result = _run_block(
            execute_blocks[0], cwd=work, environment=_environment(home, fake_bin)
        )
    finally:
        os.umask(previous_umask)
    observed = log.read_text(encoding="ascii").strip() if log.exists() else ""
    _assert_failed_before(
        result,
        observed == "0022"
        and not (
            home
            / ".local/share/compag-curation/venvs/science-gpu-lite-1.9.3"
        ).exists(),
        "restricted installation umask",
    )


def _preexisting_output_test(root: Path, temporary: Path) -> None:
    block = _block(root, "README.md", "Quick Demo")
    work = temporary / "preexisting-root"
    home = temporary / "preexisting-home"
    fake_bin = temporary / "preexisting-bin"
    marker = temporary / "preexisting-mutation"
    work.mkdir()
    (work / "pyproject.toml").write_text("[build-system]\n", encoding="ascii")
    (home / "compag-workspaces/readme-quick-demo-output").mkdir(parents=True)
    _write_executable(
        fake_bin / "python3.12",
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" > {marker!s}\nexit 99\n",
    )
    result = _run_block(block, cwd=work, environment=_environment(home, fake_bin))
    _assert_failed_before(
        result,
        not marker.exists() and not (work / ".venv-readme-quick").exists(),
        "pre-existing output",
    )


def _asset_failure_test(root: Path, temporary: Path) -> None:
    block = _block(root, "QUICKSTART.md", "Real GPU Demo From Source")
    work = temporary / "asset-root"
    home = temporary / "asset-home"
    fake_bin = temporary / "asset-bin"
    log = temporary / "asset-log"
    python = _full_gpu_venv(home) / "bin/python"
    work.mkdir()
    (work / "pyproject.toml").write_text("[build-system]\n", encoding="ascii")
    _write_executable(
        python,
        "#!/bin/sh\n"
        f"case \"$*\" in\n  *\"pip check\"*) echo pip-check >> {log!s}; exit 0 ;;\n"
        f"  *\"doctor --profile science-gpu\"*) echo doctor >> {log!s}; exit 0 ;;\n"
        f"  *\"assets fetch\"*) echo fetch >> {log!s}; exit 43 ;;\n"
        f"  *\"assets verify\"*) echo verify >> {log!s}; exit 0 ;;\n"
        f"  *\"demo --profile real\"*) echo demo >> {log!s}; exit 0 ;;\n"
        f"  *) echo unexpected >> {log!s}; exit 99 ;;\nesac\n",
    )
    result = _run_block(block, cwd=work, environment=_environment(home, fake_bin))
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    _assert_failed_before(
        result,
        lines == ["pip-check", "doctor", "fetch"],
        "asset-fetch failure",
    )


def _bootstrap_failure_test(root: Path, temporary: Path) -> None:
    block = _block(root, "docs/INSTALL_CPU.md", "Source Distribution")
    home = temporary / "bootstrap-home"
    release = home / "Downloads/compag-curation-1.9.3"
    fake_bin = temporary / "bootstrap-bin"
    log = temporary / "bootstrap-log"
    late_marker = temporary / "bootstrap-late-command"
    release.mkdir(parents=True)
    sdist = release / "compag_curation-1.9.3.tar.gz"
    sdist.write_bytes(b"not-opened-before-bootstrap-success\n")
    _sidecar(sdist)
    source_lock = root / "requirements/constraints-bootstrap-cp312-linux-x86_64.txt"
    lock = release / source_lock.name
    lock.write_bytes(source_lock.read_bytes())
    _sidecar(lock)
    fake_source = f"""#!{sys.executable}
import os
import subprocess
import sys
from pathlib import Path

args = sys.argv[1:]
if args and args[0] == "-c":
    raise SystemExit(0)
if args[:2] == ["-m", "venv"] and len(args) == 3:
    target = Path(args[2])
    executable = target / "bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text({("#!" + sys.executable + "\n" + "import os,sys\n" + "from pathlib import Path\n" + "args=' '.join(sys.argv[1:])\n" + "Path(os.environ['DOC_SHELL_LOG']).write_text(args + '\\\\n', encoding='utf-8')\n" + "if 'constraints-bootstrap-cp312-linux-x86_64.txt' in args: raise SystemExit(47)\n" + "Path(os.environ['DOC_LATE_MARKER']).write_text(args, encoding='utf-8')\n" + "raise SystemExit(99)\n")!r}, encoding="utf-8")
    executable.chmod(0o755)
    raise SystemExit(0)
raise SystemExit(98)
"""
    _write_executable(fake_bin / "python3.12", fake_source)
    environment = _environment(
        home,
        fake_bin,
        DOC_REAL_PYTHON=sys.executable,
        DOC_SHELL_LOG=str(log),
        DOC_LATE_MARKER=str(late_marker),
    )
    result = _run_block(block, cwd=temporary, environment=environment)
    observed = log.read_text(encoding="utf-8") if log.exists() else ""
    _assert_failed_before(
        result,
        "constraints-bootstrap-cp312-linux-x86_64.txt" in observed
        and not late_marker.exists(),
        "bootstrap installation failure",
    )


def run(root: Path) -> dict[str, object]:
    summary = _syntax_and_directive_test(root)
    with tempfile.TemporaryDirectory(prefix="compag-doc-shell-") as directory:
        temporary = Path(directory)
        _checksum_failure_test(root, temporary)
        _sidecar_target_failure_test(root, temporary)
        _install_permission_umask_test(root, temporary)
        _profile_installer_failure_test(root, temporary)
        _preexisting_output_test(root, temporary)
        _asset_failure_test(root, temporary)
        _bootstrap_failure_test(root, temporary)
    return {
        "schema": "compag-documentation-shell-regression/v1",
        "status": "PASS",
        **summary,
        "negative_journeys": {
            "asset_failure": "PASS_FAIL_CLOSED",
            "bootstrap_failure": "PASS_FAIL_CLOSED",
            "checksum_failure": "PASS_FAIL_CLOSED",
            "preexisting_output": "PASS_FAIL_CLOSED",
            "profile_installer_failure": "PASS_FAIL_CLOSED",
            "sidecar_target_substitution": "PASS_FAIL_CLOSED",
        },
        "install_permission_umask": "PASS_RESTRICTED_0022",
        "scientific_execution_calls": 0,
        "network_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--records", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    try:
        result = run(root)
        rc = 0
    except Exception as error:  # surfaced as a compact CI record
        result = {
            "schema": "compag-documentation-shell-regression/v1",
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
        }
        rc = 1
    encoded = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.records is not None:
        args.records.write_text(encoded, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
