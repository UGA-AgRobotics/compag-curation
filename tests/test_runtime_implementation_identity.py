from __future__ import annotations

import hashlib
import json
import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class _Distribution:
    version = "1.2.0"

    def __init__(self, root: Path, *, editable: bool = False, version: str = "1.2.0") -> None:
        self.root = root
        self.editable = editable
        self.version = version

    def locate_file(self, value: str) -> Path:
        if value != "compag_curation":
            raise AssertionError(value)
        return self.root

    def read_text(self, name: str) -> str | None:
        if name != "direct_url.json":
            return None
        return '{"dir_info":{"editable":' + ("true" if self.editable else "false") + '}}'


def _package(root: Path, value: bytes = b"value = 1\n") -> Path:
    package = root / "compag_curation"
    (package / "resources").mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"__version__ = '1.2.0'\n")
    (package / "module.py").write_bytes(value)
    (package / "resources" / "contract.json").write_bytes(b'{"schema":"fixture/v1"}\n')
    for path in package.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    return package


def _conda_history(prefix: Path, payload: bytes | None = None) -> Path:
    conda_meta = prefix / "conda-meta"
    conda_meta.mkdir(mode=0o755, exist_ok=True)
    conda_meta.chmod(0o755)
    history = conda_meta / "history"
    history.write_bytes(
        payload
        if payload is not None
        else (
            b"==> 2026-08-30 00:00:00 <==\n"
            b"# cmd: mamba create --prefix /fixture\n"
            b"+python-3.12.7-0\n"
        )
    )
    history.chmod(0o644)
    return history


class RuntimeImplementationIdentityTests(unittest.TestCase):
    def test_science_lock_runtime_resource_and_metadata_are_closed(self) -> None:
        from compag_curation import runtime_lock

        root = Path(__file__).resolve().parents[1]
        identity = runtime_lock.science_cpu_lock_identity()
        self.assertEqual(identity["sha256"], runtime_lock.SCIENCE_CPU_LOCK_SHA256)
        self.assertEqual(identity["versions"], runtime_lock.SCIENCE_CPU_VERSIONS)
        metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertNotIn("science-cpu", metadata["project"]["optional-dependencies"])
        self.assertNotIn("torch-2.9.0", (root / "pyproject.toml").read_text(encoding="utf-8"))

    def test_science_gpu_lock_archives_and_metadata_are_closed(self) -> None:
        from compag_curation import runtime_lock

        root = Path(__file__).resolve().parents[1]
        identity = runtime_lock.science_gpu_lock_identity()
        self.assertEqual(identity["sha256"], runtime_lock.SCIENCE_GPU_LOCK_SHA256)
        self.assertEqual(identity["schema"], "compag-curation-science-gpu-lock/v2")
        self.assertEqual(identity["versions"], runtime_lock.SCIENCE_GPU_VERSIONS)
        self.assertEqual(identity["cuda"]["runtime"], "13.2")
        self.assertEqual(identity["cuda"]["pip_cuda_toolkit"], "13.2.1")
        self.assertEqual(identity["cuda"]["sam2_build_toolchain"]["version"], "13.2.86")
        self.assertEqual(identity["cuda"]["sam2_build_toolchain"]["release"], "CUDA 13.2 Update 2")
        self.assertEqual(
            identity["dependency_owners"],
            {
                "nvidia-nccl-cu12": "xgboost 2.1.1 wheel dependency",
                "nvidia-nccl-cu13": "torch 2.13.0+cu132 wheel dependency",
            },
        )

        records: dict[str, dict[str, str]] = {}
        relative = "requirements/constraints-science-gpu-cp312-linux-x86_64.txt"
        for line in (root / relative).read_text(encoding="ascii").splitlines():
            if not line or line.startswith("#"):
                continue
            raw_name, separator, raw_url = line.partition(" @ ")
            self.assertEqual(separator, " @ ")
            url, hash_separator, sha256 = raw_url.rpartition("#sha256=")
            self.assertEqual(hash_separator, "#sha256=")
            name = raw_name.lower().replace("_", "-")
            self.assertNotIn(name, records)
            self.assertRegex(line, r"#sha256=[0-9a-f]{64}$")
            records[name] = {
                "sha256": sha256,
                "url": url,
                "version": runtime_lock.SCIENCE_GPU_VERSIONS[name],
            }
        self.assertEqual(
            set(records),
            set(runtime_lock.SCIENCE_GPU_VERSIONS) - {"compag-curation", "sam-2"},
        )
        self.assertEqual(len(records), runtime_lock.SCIENCE_GPU_ARCHIVE_COUNT)
        self.assertEqual(identity["archives"], records)
        archives_encoded = json.dumps(
            records,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        self.assertEqual(
            hashlib.sha256(
                b"compag-curation-science-gpu-archives-v1\0" + archives_encoded
            ).hexdigest(),
            runtime_lock.SCIENCE_GPU_ARCHIVES_SHA256,
        )
        for name, version in runtime_lock.SCIENCE_GPU_VERSIONS.items():
            if name in {"compag-curation", "sam-2"}:
                continue
            encoded_version = version.lower().replace("+", "%2b")
            self.assertIn(encoded_version, records[name]["url"].lower())

        sam_requirement = (
            root / "requirements/requirements-sam2-gpu-cp312-linux-x86_64.txt"
        ).read_text(encoding="ascii")
        self.assertIn(runtime_lock.SCIENCE_GPU_SAM2_URL, sam_requirement)
        self.assertIn("@" + runtime_lock.SCIENCE_GPU_SAM2_COMMIT, sam_requirement)
        self.assertIn("SAM2_BUILD_CUDA=1", sam_requirement)

        archive_install = (
            "pip install --require-hashes --no-compile --no-build-isolation "
            "--no-deps --force-reinstall -r "
            '"$REPO_ROOT/requirements/constraints-science-gpu-cp312-linux-x86_64.txt"'
        )
        install_guide = (root / "docs/INSTALL_GPU.md").read_text(encoding="utf-8")
        self.assertIn(
            archive_install,
            install_guide,
            "manual science-gpu audit path does not force exact PEP 610 archive provenance",
        )
        installer = (root / "tools/install_gpu_profile.py").read_text(encoding="utf-8")
        for marker in (
            'GPU_LOCK_RELATIVE = Path("requirements/constraints-science-gpu-cp312-linux-x86_64.txt")',
            'SAM2_REQUIREMENT_RELATIVE = Path("requirements/requirements-sam2-gpu-cp312-linux-x86_64.txt")',
            '"--require-hashes"',
            '"--force-reinstall"',
            '"SAM2_BUILD_CUDA": "1"',
            '"SAM2_BUILD_ALLOW_ERRORS": "0"',
        ):
            self.assertIn(marker, installer)
        for guide in (root / "README.md", root / "docs/INSTALL_GPU.md"):
            self.assertIn(
                "tools/install_gpu_profile.py",
                guide.read_text(encoding="utf-8"),
                f"science-gpu guide does not route through the shared profile installer: {guide}",
            )

        metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        direct = metadata["project"]["optional-dependencies"]["science-gpu"]
        direct_names = {
            requirement.split(" @ ", 1)[0].split("==", 1)[0].lower().replace("_", "-")
            for requirement in direct
        }
        self.assertEqual(
            direct_names,
            {
                "antlr4-python3-runtime", "hydra-core", "imbalanced-learn", "iopath",
                "cupy-cuda13x", "joblib", "matplotlib", "numpy", "omegaconf", "opencv-python",
                "packaging", "pandas", "pillow", "portalocker", "pyyaml", "sam-2",
                "scikit-learn", "scipy", "tabulate", "torch", "torchvision", "tqdm",
                "xgboost",
            },
        )
        for requirement in direct:
            name = requirement.split(" @ ", 1)[0].split("==", 1)[0].lower().replace("_", "-")
            if name == "sam-2":
                self.assertIn("@" + runtime_lock.SCIENCE_GPU_SAM2_COMMIT, requirement)
            else:
                encoded_version = runtime_lock.SCIENCE_GPU_VERSIONS[name].replace("+", "%2B")
                self.assertTrue(
                    f"=={encoded_version}" in requirement
                    or encoded_version in requirement,
                    f"science-gpu metadata version mismatch for {name}",
                )

    def test_content_identity_is_path_and_metadata_independent_but_byte_closed(self) -> None:
        from compag_curation import runtime_lock

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = _package(root / "left")
            right = _package(root / "right")
            (right / "module.py").chmod(0o444)
            os.utime(right / "module.py", ns=(1_000_000_000, 2_000_000_000))
            cache = right / "__pycache__"
            cache.mkdir(mode=0o755)
            (cache / "module.cpython-312.pyc").write_bytes(b"ignored-source-cache")
            with mock.patch.object(runtime_lock, "_package_root", return_value=left):
                left_identity = runtime_lock.package_implementation_identity()
            with mock.patch.object(runtime_lock, "_package_root", return_value=right):
                right_identity = runtime_lock.package_implementation_identity()
            self.assertEqual(left_identity, right_identity)
            self.assertEqual(
                set(left_identity),
                {"schema", "sha256", "file_count", "size_bytes"},
            )
            rows = []
            for path in sorted(item for item in left.rglob("*") if item.is_file()):
                payload = path.read_bytes()
                rows.append({
                    "path": path.relative_to(left).as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                })
            encoded = json.dumps(
                {
                    "schema": "compag-curation-package-implementation-manifest/v1",
                    "rows": rows,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
            self.assertEqual(left_identity["file_count"], len(rows))
            self.assertEqual(left_identity["size_bytes"], sum(row["size_bytes"] for row in rows))
            self.assertEqual(
                left_identity["sha256"],
                hashlib.sha256(
                    b"compag-curation-package-implementation-v1\0" + encoded
                ).hexdigest(),
            )

            (right / "module.py").chmod(0o644)
            (right / "module.py").write_bytes(b"value = 2\n")
            with mock.patch.object(runtime_lock, "_package_root", return_value=right):
                changed = runtime_lock.package_implementation_identity()
            self.assertNotEqual(changed["sha256"], left_identity["sha256"])
            (right / "added.py").write_bytes(b"added = True\n")
            with mock.patch.object(runtime_lock, "_package_root", return_value=right):
                added = runtime_lock.package_implementation_identity()
            self.assertNotEqual(added["sha256"], changed["sha256"])

    def test_symlink_and_hardlink_are_rejected(self) -> None:
        from compag_curation import runtime_lock

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            symlinked = _package(root / "symlinked")
            (symlinked / "alias.py").symlink_to(symlinked / "module.py")
            with mock.patch.object(runtime_lock, "_package_root", return_value=symlinked):
                with self.assertRaisesRegex(RuntimeError, "symlink"):
                    runtime_lock.package_implementation_identity()
            linked = _package(root / "linked")
            os.link(linked / "module.py", linked / "alias.py")
            with mock.patch.object(runtime_lock, "_package_root", return_value=linked):
                with self.assertRaisesRegex(RuntimeError, "hard link"):
                    runtime_lock.package_implementation_identity()

    def test_same_name_mutation_after_hashing_is_rejected(self) -> None:
        from compag_curation import runtime_lock

        with tempfile.TemporaryDirectory() as temporary:
            package = _package(Path(temporary))
            original = runtime_lock._package_paths
            calls = 0

            def racing_paths(root: Path, *, reject_bytecode: bool):
                nonlocal calls
                calls += 1
                if calls == 2:
                    (root / "module.py").write_bytes(b"value = 9\n")
                return original(root, reject_bytecode=reject_bytecode)

            with (
                mock.patch.object(runtime_lock, "_package_root", return_value=package),
                mock.patch.object(runtime_lock, "_package_paths", side_effect=racing_paths),
            ):
                with self.assertRaisesRegex(RuntimeError, "changed after hashing"):
                    runtime_lock.package_implementation_identity()

    def test_installed_attestation_rejects_group_writable_runtime_chain(self) -> None:
        from compag_curation import runtime_lock

        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "venv"
            package = _package(prefix / "lib/python3.12/site-packages")
            base_prefix = Path(temporary) / "base"
            base_prefix.mkdir(mode=0o755)

            def identity(*, conda: bool = False) -> dict[str, object]:
                with (
                    mock.patch.object(runtime_lock, "_package_root", return_value=package),
                    mock.patch.object(
                        runtime_lock.importlib.metadata,
                        "distribution",
                        return_value=_Distribution(package),
                    ),
                    mock.patch.object(runtime_lock.sys, "prefix", str(prefix)),
                    mock.patch.object(
                        runtime_lock.sys,
                        "base_prefix",
                        str(prefix if conda else base_prefix),
                    ),
                    mock.patch.object(runtime_lock.sys, "dont_write_bytecode", True),
                    mock.patch.object(
                        runtime_lock.sys,
                        "flags",
                        SimpleNamespace(isolated=1),
                    ),
                    mock.patch.object(runtime_lock.sys, "pycache_prefix", None),
                ):
                    return runtime_lock.package_implementation_identity(
                        require_installed=True,
                    )

            module = package / "module.py"
            original_mode = module.stat().st_mode & 0o7777
            module.chmod(0o664)
            try:
                with self.assertRaisesRegex(RuntimeError, "file permissions"):
                    identity()
            finally:
                module.chmod(original_mode)

            original_mode = package.stat().st_mode & 0o7777
            package.chmod(0o775)
            try:
                with self.assertRaisesRegex(RuntimeError, "directory permissions"):
                    identity()
            finally:
                package.chmod(original_mode)

            for label, directory in (
                ("prefix", prefix),
                ("parent component", package.parent),
            ):
                with self.subTest(directory=label):
                    original_mode = directory.stat().st_mode & 0o7777
                    directory.chmod(0o775)
                    try:
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "safely permissioned runtime directory chain",
                        ):
                            identity()
                    finally:
                        directory.chmod(original_mode)

            _conda_history(prefix)
            conda_meta = prefix / "conda-meta"
            original_mode = conda_meta.stat().st_mode & 0o7777
            conda_meta.chmod(0o775)
            try:
                with self.assertRaisesRegex(RuntimeError, "attested Conda/Mamba prefix"):
                    identity(conda=True)
            finally:
                conda_meta.chmod(original_mode)

    def test_installed_attestation_requires_owner_directory_access(self) -> None:
        from compag_curation import runtime_lock

        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "venv"
            package = _package(prefix / "lib/python3.12/site-packages")
            base_prefix = Path(temporary) / "base"
            base_prefix.mkdir(mode=0o755)

            def identity() -> dict[str, object]:
                with (
                    mock.patch.object(runtime_lock, "_package_root", return_value=package),
                    mock.patch.object(
                        runtime_lock.importlib.metadata,
                        "distribution",
                        return_value=_Distribution(package),
                    ),
                    mock.patch.object(runtime_lock.sys, "prefix", str(prefix)),
                    mock.patch.object(runtime_lock.sys, "base_prefix", str(base_prefix)),
                    mock.patch.object(runtime_lock.sys, "dont_write_bytecode", True),
                    mock.patch.object(
                        runtime_lock.sys,
                        "flags",
                        SimpleNamespace(isolated=1),
                    ),
                    mock.patch.object(runtime_lock.sys, "pycache_prefix", None),
                ):
                    return runtime_lock.package_implementation_identity(
                        require_installed=True,
                    )

            original_mode = prefix.stat().st_mode & 0o7777
            prefix.chmod(0o655)
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "unavailable or unsafe|safely permissioned runtime directory chain",
                ):
                    identity()
            finally:
                prefix.chmod(original_mode)

            original_mode = package.stat().st_mode & 0o7777
            package.chmod(0o355)
            try:
                with self.assertRaisesRegex(RuntimeError, "directory permissions"):
                    identity()
            finally:
                package.chmod(original_mode)

    def test_installed_attestation_accepts_venv_and_conda_but_rejects_unsafe_installs(self) -> None:
        from compag_curation import runtime_lock

        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "venv"
            package = _package(prefix / "lib/python3.12/site-packages")

            def identity(
                distribution: _Distribution,
                *,
                isolated: int = 1,
                pycache_prefix: str | None = None,
                dont_write_bytecode: bool = True,
                runtime_prefix: Path | None = None,
                runtime_base_prefix: Path | None = None,
                expected_distribution_version: str = "1.2.0",
            ) -> dict[str, object]:
                selected_prefix = runtime_prefix or prefix
                selected_base = runtime_base_prefix or (Path(temporary) / "base")
                selected_base.mkdir(parents=True, exist_ok=True)
                with (
                    mock.patch.object(runtime_lock, "_package_root", return_value=package),
                    mock.patch.object(
                        runtime_lock.importlib.metadata,
                        "distribution",
                        return_value=distribution,
                    ),
                    mock.patch.object(runtime_lock.sys, "prefix", str(selected_prefix)),
                    mock.patch.object(runtime_lock.sys, "base_prefix", str(selected_base)),
                    mock.patch.object(runtime_lock.sys, "dont_write_bytecode", dont_write_bytecode),
                    mock.patch.object(
                        runtime_lock.sys,
                        "flags",
                        SimpleNamespace(isolated=isolated),
                    ),
                    mock.patch.object(runtime_lock.sys, "pycache_prefix", pycache_prefix),
                ):
                    return runtime_lock.package_implementation_identity(
                        require_installed=True,
                        expected_distribution_version=expected_distribution_version,
                    )

            accepted = identity(_Distribution(package))
            self.assertEqual(
                accepted["installation"],
                "SYS_PREFIX_DISTRIBUTION_NON_EDITABLE_ISOLATED_NO_BYTECODE",
            )
            accepted_gpu = identity(
                _Distribution(package, version="1.3.0"),
                expected_distribution_version="1.3.0",
            )
            self.assertEqual(accepted_gpu["installation"], accepted["installation"])
            with (
                mock.patch.object(runtime_lock, "_package_root", return_value=package),
                mock.patch.object(
                    runtime_lock.importlib.metadata,
                    "distribution",
                    return_value=_Distribution(package, version="1.9.3"),
                ),
                mock.patch.object(runtime_lock.sys, "prefix", str(prefix)),
                mock.patch.object(runtime_lock.sys, "base_prefix", str(Path(temporary) / "base")),
                mock.patch.object(runtime_lock.sys, "dont_write_bytecode", True),
                mock.patch.object(runtime_lock.sys, "flags", SimpleNamespace(isolated=1)),
                mock.patch.object(runtime_lock.sys, "pycache_prefix", None),
            ):
                current_default = runtime_lock.package_implementation_identity(
                    require_installed=True
                )
            self.assertEqual(current_default["installation"], accepted["installation"])
            with mock.patch.dict(os.environ, {"CONDA_PREFIX": str(prefix)}):
                with self.assertRaisesRegex(RuntimeError, "attested Conda/Mamba prefix"):
                    identity(_Distribution(package), runtime_base_prefix=prefix)
            history = _conda_history(prefix)
            accepted_conda = identity(
                _Distribution(package),
                runtime_base_prefix=prefix,
            )
            self.assertEqual(accepted_conda["installation"], accepted["installation"])
            with self.assertRaisesRegex(RuntimeError, "editable"):
                identity(
                    _Distribution(package, editable=True),
                    runtime_base_prefix=prefix,
                )
            history.write_bytes(b"not a conda history\n")
            history.chmod(0o644)
            with self.assertRaisesRegex(RuntimeError, "attested Conda/Mamba prefix"):
                identity(_Distribution(package), runtime_base_prefix=prefix)
            history = _conda_history(prefix)
            history.chmod(0o664)
            with self.assertRaisesRegex(RuntimeError, "attested Conda/Mamba prefix"):
                identity(_Distribution(package), runtime_base_prefix=prefix)
            history.chmod(0o644)
            symlink_target = prefix / "conda-history-symlink-target"
            symlink_target.write_bytes(history.read_bytes())
            symlink_target.chmod(0o644)
            history.unlink()
            history.symlink_to(symlink_target)
            with self.assertRaisesRegex(RuntimeError, "attested Conda/Mamba prefix"):
                identity(_Distribution(package), runtime_base_prefix=prefix)
            history.unlink()
            symlink_target.unlink()
            history = _conda_history(prefix)
            hardlink_target = prefix / "conda-history-hardlink-target"
            os.link(history, hardlink_target)
            with self.assertRaisesRegex(RuntimeError, "attested Conda/Mamba prefix"):
                identity(_Distribution(package), runtime_base_prefix=prefix)
            hardlink_target.unlink()
            with self.assertRaisesRegex(RuntimeError, "editable"):
                identity(_Distribution(package, editable=True))
            with self.assertRaisesRegex(RuntimeError, "version differs"):
                identity(_Distribution(package, version="1.1.0"))
            other_prefix = Path(temporary) / "other-venv"
            other_prefix.mkdir()
            _conda_history(other_prefix)
            with self.assertRaisesRegex(RuntimeError, "match the active runtime prefix"):
                identity(
                    _Distribution(package),
                    runtime_prefix=other_prefix,
                    runtime_base_prefix=other_prefix,
                )
            mismatched = _package(prefix / "alternate-site")
            with self.assertRaisesRegex(RuntimeError, "match the active runtime prefix"):
                identity(_Distribution(mismatched))
            with self.assertRaisesRegex(RuntimeError, "isolated mode"):
                identity(_Distribution(package), isolated=0)
            with self.assertRaisesRegex(RuntimeError, "bytecode-cache prefix"):
                identity(_Distribution(package), pycache_prefix="/tmp/unbound-cache")
            with self.assertRaisesRegex(RuntimeError, "bytecode generation"):
                identity(_Distribution(package), dont_write_bytecode=False)
            cache = package / "__pycache__"
            cache.mkdir(mode=0o755)
            (cache / "module.cpython-312.pyc").write_bytes(b"unbound")
            with self.assertRaisesRegex(RuntimeError, "bytecode"):
                identity(_Distribution(package))


if __name__ == "__main__":
    unittest.main()
