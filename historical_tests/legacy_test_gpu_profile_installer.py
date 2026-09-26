from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import stat
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from tools import install_gpu_profile as installer


ARCHIVE_HASH = "a" * 64
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"


class FakeInstallRunner:
    def __init__(
        self,
        venv: Path,
        *,
        fail_command: int | None = None,
        tamper_conda_metadata: bool = False,
    ) -> None:
        self.venv = venv
        self.fail_command = fail_command
        self.tamper_conda_metadata = tamper_conda_metadata
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        argv: list[str],
        *,
        check: bool,
        env: dict[str, str],
        text: bool,
        stdout: object,
        stderr: object,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append({"argv": list(argv), "env": dict(env), "stdout": stdout, "stderr": stderr})
        index = len(self.calls) - 1
        if self.fail_command == index:
            raise subprocess.CalledProcessError(1, argv)
        if index == 0:
            for key in ("MAMBA_ROOT_PREFIX", "CONDA_PKGS_DIRS"):
                if key in env:
                    Path(env[key]).mkdir(mode=0o755, parents=True)
            (self.venv / "bin").mkdir(parents=True)
            for name in ("python", "nvcc"):
                executable = self.venv / "bin" / name
                executable.write_bytes(b"fixture executable\n")
                executable.chmod(0o755)
            metadata = self.venv / "conda-meta"
            metadata.mkdir(mode=0o755)
            lock_path = Path(installer.__file__).resolve().parents[1] / installer.CONDA_LOCK_RELATIVE
            locked = json.loads(lock_path.read_text(encoding="ascii"))["packages"]
            for row in locked:
                record = {
                    "name": row["name"],
                    "version": row["version"],
                    "build": row["build"],
                    "subdir": row["subdir"],
                    "fn": row["filename"],
                    "size": row["size_bytes"],
                    "sha256": row["sha256"],
                }
                target = metadata / (row["filename"][:-len(".conda")] + ".json")
                target.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
                target.chmod(0o644)
            if self.tamper_conda_metadata:
                first = metadata / (locked[0]["filename"][:-len(".conda")] + ".json")
                changed = json.loads(first.read_text(encoding="utf-8"))
                changed["sha256"] = "0" * 64
                first.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
            (metadata / "history").write_text("fixture\n", encoding="utf-8")
        captured = stdout is subprocess.PIPE
        output = (
            json.dumps(
                {
                    "status": "PASS",
                    "profile": "science-gpu",
                    "checks": {"cuda": {"resolved_device": "cuda:0"}},
                },
                sort_keys=True,
            )
            + "\n"
            if captured and argv[-3:] == ["doctor", "--profile", "science-gpu"]
            else ("" if captured else None)
        )
        return subprocess.CompletedProcess(argv, 0, stdout=output, stderr="" if captured else None)


class GPUProfileInstallerTests(unittest.TestCase):
    def _fixture(self, parent: Path, *, version: str = installer.RELEASE_VERSION) -> dict[str, Path]:
        source = parent / "source"
        requirements = source / "requirements"
        requirements.mkdir(parents=True)
        (source / "pyproject.toml").write_text(
            f'[project]\nname = "compag-curation"\nversion = "{version}"\n',
            encoding="utf-8",
        )
        archive = (
            "packaging @ https://files.pythonhosted.org/packages/fixture/"
            f"packaging-26.3-py3-none-any.whl#sha256={ARCHIVE_HASH}\n"
        )
        (requirements / installer.BOOTSTRAP_LOCK_RELATIVE.name).write_text(archive, encoding="ascii")
        (requirements / installer.GPU_LOCK_RELATIVE.name).write_text(archive, encoding="ascii")
        accepted_conda_lock = Path(installer.__file__).resolve().parents[1] / installer.CONDA_LOCK_RELATIVE
        (requirements / installer.CONDA_LOCK_RELATIVE.name).write_bytes(accepted_conda_lock.read_bytes())
        (requirements / installer.SAM2_REQUIREMENT_RELATIVE.name).write_text(
            "# reviewed CUDA SAM2 source\n"
            f"sam-2 @ git+{installer.SAM2_URL}@{SAM2_COMMIT}\n",
            encoding="ascii",
        )
        source.chmod(0o755)
        requirements.chmod(0o755)
        for path in source.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)

        release = parent / "release"
        release.mkdir(mode=0o755)
        wheel = release / installer.WHEEL_BASENAME
        metadata_root = f"compag_curation-{installer.RELEASE_VERSION}.dist-info"
        with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive_file:
            archive_file.writestr(
                f"{metadata_root}/METADATA",
                "Metadata-Version: 2.4\nName: compag-curation\n"
                f"Version: {installer.RELEASE_VERSION}\n",
            )
            archive_file.writestr(
                f"{metadata_root}/WHEEL",
                "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            )
            archive_file.writestr("compag_curation/__init__.py", b"__all__ = ()\n")
        wheel.chmod(0o644)
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        sidecar = wheel.with_name(installer.WHEEL_SIDECAR_BASENAME)
        sidecar.write_text(f"{digest}  {wheel.name}\n", encoding="ascii")
        sidecar.chmod(0o644)

        tools = parent / "executables"
        tools.mkdir(mode=0o755)
        mamba = tools / "mamba"
        mamba.write_bytes(b"fixture mamba\n")
        mamba.chmod(0o755)
        nvidia_smi = tools / "nvidia-smi"
        nvidia_smi.write_bytes(b"fixture nvidia-smi\n")
        nvidia_smi.chmod(0o755)

        environments = parent / "environments"
        projects = parent / "projects"
        environments.mkdir(mode=0o700)
        projects.mkdir(mode=0o700)
        return {
            "source": source,
            "wheel": wheel,
            "mamba": mamba,
            "nvidia_smi": nvidia_smi,
            "venv": environments / "gpu-profile-1.9.3",
            "project": projects / "compag-project",
        }

    def _plan(self, paths: dict[str, Path], preset: str = "lite") -> dict[str, object]:
        return installer.build_install_plan(
            preset=preset,
            venv=paths["venv"],
            wheel=paths["wheel"],
            project=paths["project"],
            source_root=paths["source"],
            mamba=paths["mamba"],
        )

    def test_conda_lock_binds_the_python_archive_used_by_binary_attestation(self) -> None:
        lock_path = Path(installer.__file__).resolve().parents[1] / installer.CONDA_LOCK_RELATIVE
        parsed = json.loads(lock_path.read_text(encoding="ascii"))
        python_row = next(row for row in parsed["packages"] if row["name"] == "python")
        self.assertEqual(python_row["build"], installer.CONDA_PYTHON_BUILD)
        self.assertEqual(python_row["sha256"], installer.CONDA_PYTHON_ARCHIVE_SHA256)
        self.assertEqual(python_row["size_bytes"], installer.CONDA_PYTHON_ARCHIVE_SIZE_BYTES)

        python_row["sha256"] = "0" * 64
        payload = installer._canonical_json(parsed)
        value = installer.VerifiedInput(
            path=lock_path,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            payload=payload,
        )
        with self.assertRaisesRegex(installer.InstallerError, "does not bind"):
            installer._validate_conda_lock(value)

    def test_planning_is_no_write_and_maps_all_explicit_presets(self) -> None:
        self.assertEqual(
            installer.PRESET_PROFILES,
            {
                "lite": "efficient-xgb-recall-gpu-v1",
                "full": "canonical-xgb-recall-gpu-v1",
                "full-image": "full-image-multiscale-xgb-recall-gpu-v1",
            },
        )
        for preset, profile in installer.PRESET_PROFILES.items():
            with self.subTest(preset=preset), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                paths = self._fixture(parent)
                before = sorted(path.relative_to(parent).as_posix() for path in parent.rglob("*"))
                plan = self._plan(paths, preset)
                after = sorted(path.relative_to(parent).as_posix() for path in parent.rglob("*"))
                self.assertEqual(before, after)
                self.assertFalse(paths["venv"].exists())
                self.assertFalse(paths["project"].exists())
                self.assertEqual(plan["status"], "PLAN_ONLY_NO_MUTATION")
                self.assertFalse(plan["mutations_performed"])
                self.assertEqual(plan["preset"], preset)
                self.assertEqual(plan["profile"], profile)
                self.assertEqual(plan["dependency_profile"], "science-gpu")
                self.assertEqual(plan["inputs"]["conda_lock"]["package_count"], 71)
                self.assertEqual(plan["conda_install"]["mode"], "locked_https")
                commands = plan["commands"]
                self.assertEqual(len(commands), 10)
                self.assertEqual(commands[0]["argv"][-2], "--file")
                self.assertEqual(commands[1]["argv"][-1], "--explicit")
                self.assertIn("--require-hashes", commands[4]["argv"])
                self.assertIn("--require-hashes", commands[5]["argv"])
                self.assertIn("--force-reinstall", commands[5]["argv"])
                self.assertEqual(
                    commands[6]["environment"],
                    {"CUDA_HOME": str(paths["venv"]), "SAM2_BUILD_ALLOW_ERRORS": "0", "SAM2_BUILD_CUDA": "1"},
                )
                self.assertEqual(commands[-1]["argv"][-3:], ["doctor", "--profile", "science-gpu"])
                next_rows = plan["next_commands"]
                self.assertEqual(next_rows[0]["argv"][-4:], ["--profile", profile, "--output", str(paths["project"])])
                self.assertEqual(next_rows[1]["argv"][-4:], ["--profile", profile, "--asset-root", str(paths["project"] / "assets")])

    def test_standard_conda_hardlinked_mamba_is_accepted_but_writable_binary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            package_cache_link = paths["mamba"].with_name("mamba-package-cache-link")
            os.link(paths["mamba"], package_cache_link)
            self.assertEqual(paths["mamba"].stat().st_nlink, 2)
            plan = self._plan(paths)
            self.assertEqual(plan["commands"][0]["argv"][0], str(paths["mamba"]))

            paths["mamba"].chmod(0o775)
            with self.assertRaisesRegex(installer.InstallerError, "metadata is unsafe"):
                self._plan(paths)

    def test_cli_plan_emits_one_compact_json_line_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = installer.main(
                    [
                        "--preset", "lite",
                        "--venv", str(paths["venv"]),
                        "--wheel", str(paths["wheel"]),
                        "--project", str(paths["project"]),
                        "--source-root", str(paths["source"]),
                        "--mamba", str(paths["mamba"]),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(output.getvalue().count("\n"), 1)
            result = json.loads(output.getvalue())
            self.assertEqual(result["schema"], installer.INSTALL_PLAN_SCHEMA)
            self.assertNotIn("_payloads", result)
            self.assertFalse(paths["venv"].exists())

    def test_execute_copies_bound_inputs_runs_doctor_and_writes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            plan = self._plan(paths, "full")
            runner = FakeInstallRunner(paths["venv"])
            result = installer.execute_install(plan, runner=runner)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["profile"], installer.PRESET_PROFILES["full"])
            self.assertEqual(len(runner.calls), 10)
            for call in runner.calls:
                environment = call["env"]
                self.assertNotIn("PYTHONPATH", environment)
                self.assertNotIn("CONDARC", environment)
                self.assertEqual(environment["MAMBA_NO_RC"], "true")
                self.assertEqual(environment["MAMBA_DOWNLOAD_THREADS"], "1")
                self.assertEqual(environment["MAMBA_REMOTE_MAX_RETRIES"], "2")
                self.assertEqual(environment["MAMBA_REMOTE_CONNECT_TIMEOUT_SECS"], "30")
                self.assertEqual(environment["PIP_CONFIG_FILE"], os.devnull)
                self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
            sam2_environment = runner.calls[6]["env"]
            self.assertEqual(sam2_environment["CUDA_HOME"], str(paths["venv"]))
            self.assertEqual(sam2_environment["SAM2_BUILD_CUDA"], "1")
            copied = paths["venv"] / "share/compag-curation/install-inputs"
            self.assertEqual(
                (copied / installer.WHEEL_BASENAME).read_bytes(),
                paths["wheel"].read_bytes(),
            )
            receipt_path = Path(result["receipt"])
            self.assertEqual(stat.S_IMODE(receipt_path.stat().st_mode), 0o644)
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            self.assertEqual(receipt["schema"], installer.INSTALL_RECEIPT_SCHEMA)
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["doctor"]["status"], "PASS")
            self.assertEqual(receipt["conda_attestation"]["package_count"], 71)
            self.assertEqual(receipt["conda_install"]["mode"], "locked_https")
            self.assertEqual(receipt["inputs"]["sam2_requirement"]["commit"], SAM2_COMMIT)
            self.assertFalse(paths["project"].exists())

    def test_failed_execution_preserves_partial_environment_without_false_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            plan = self._plan(paths)
            runner = FakeInstallRunner(paths["venv"], fail_command=5)
            with self.assertRaisesRegex(installer.InstallerError, "installation command failed"):
                installer.execute_install(plan, runner=runner)
            self.assertTrue(paths["venv"].is_dir())
            self.assertFalse((paths["venv"] / installer.RECEIPT_RELATIVE).exists())

    def test_installed_conda_metadata_must_attest_the_exact_locked_closure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            plan = self._plan(paths)
            runner = FakeInstallRunner(paths["venv"], tamper_conda_metadata=True)
            with self.assertRaisesRegex(installer.InstallerError, "differs from the accepted lock"):
                installer.execute_install(plan, runner=runner)
            self.assertFalse((paths["venv"] / installer.RECEIPT_RELATIVE).exists())

    def test_verified_conda_cache_is_exact_and_offline_explicit_uses_file_uris(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            package_dir = root / "packages/linux-64"
            package_dir.mkdir(parents=True)
            root.chmod(0o755)
            (root / "packages").chmod(0o755)
            package_dir.chmod(0o755)
            payload = b"locked-package\n"
            digest = hashlib.sha256(payload).hexdigest()
            row = {
                "name": "fixture",
                "version": "1.0",
                "build": "h123_0",
                "subdir": "linux-64",
                "filename": "fixture-1.0-h123_0.conda",
                "size_bytes": len(payload),
                "sha256": digest,
                "url": "https://conda.anaconda.org/conda-forge/linux-64/fixture-1.0-h123_0.conda",
            }
            package = package_dir / row["filename"]
            package.write_bytes(payload)
            package.chmod(0o644)
            lock_sha256 = "b" * 64
            manifest = {
                "schema": installer.CONDA_CACHE_SCHEMA,
                "description": installer.CONDA_CACHE_DESCRIPTION,
                "release_version": installer.RELEASE_VERSION,
                "lock_sha256": lock_sha256,
                "platform": "linux-64",
                "package_count": 1,
                "total_package_bytes": len(payload),
                "packages": installer._cache_manifest_rows((row,)),
            }
            manifest_path = root / installer.CONDA_CACHE_MANIFEST_BASENAME
            manifest_path.write_bytes(installer._canonical_json(manifest))
            manifest_path.chmod(0o644)

            record = installer._validate_conda_cache(
                root,
                lock_rows=(row,),
                lock_sha256=lock_sha256,
            )
            self.assertEqual(record["mode"], "verified_offline_cache")
            explicit = installer._conda_explicit_payload((row,), cache_root=root).decode("ascii")
            self.assertTrue(explicit.startswith("@EXPLICIT\n" + "file" + "://"))
            self.assertIn(f"#{digest}\n", explicit)

            extra = package_dir / "unexpected.conda"
            extra.write_bytes(b"unexpected")
            extra.chmod(0o644)
            with self.assertRaisesRegex(installer.InstallerError, "missing or extra"):
                installer._validate_conda_cache(root, lock_rows=(row,), lock_sha256=lock_sha256)

    def test_cached_execute_uses_and_removes_private_mamba_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            paths = self._fixture(parent)
            cache_root = parent / "verified-cache"
            cache_root.mkdir(mode=0o755)
            cache_record = {
                "mode": "verified_offline_cache",
                "manifest_basename": installer.CONDA_CACHE_MANIFEST_BASENAME,
                "manifest_sha256": "c" * 64,
                "lock_sha256": json.loads(
                    (paths["source"] / installer.CONDA_LOCK_RELATIVE).read_text(encoding="ascii")
                )["packages"][0]["sha256"],
                "package_count": 71,
                "total_package_bytes": 1_114_157_356,
            }
            lock_payload = (paths["source"] / installer.CONDA_LOCK_RELATIVE).read_bytes()
            cache_record["lock_sha256"] = hashlib.sha256(lock_payload).hexdigest()
            with mock.patch.object(installer, "_validate_conda_cache", return_value=cache_record):
                plan = installer.build_install_plan(
                    preset="full",
                    venv=paths["venv"],
                    wheel=paths["wheel"],
                    project=paths["project"],
                    source_root=paths["source"],
                    mamba=paths["mamba"],
                    conda_cache_root=cache_root,
                )
                workspace = paths["venv"].parent / (
                    ".{}.compag-private-mamba-cache-{}".format(
                        paths["venv"].name, installer.RELEASE_VERSION
                    )
                )
                self.assertEqual(
                    plan["commands"][0]["environment"],
                    {
                        "CONDA_PKGS_DIRS": str(workspace / "packages"),
                        "MAMBA_ROOT_PREFIX": str(workspace / "root-prefix"),
                    },
                )
                self.assertNotIn("root", plan["conda_install"]["cache"])
                result = installer.execute_install(plan, runner=FakeInstallRunner(paths["venv"]))
            self.assertEqual(result["status"], "PASS")
            self.assertFalse(workspace.exists())
            installed_explicit = (
                paths["venv"]
                / "share/compag-curation/install-inputs/conda-explicit-cp312-linux-x86_64.txt"
            ).read_text(encoding="ascii")
            self.assertTrue(installed_explicit.startswith("@EXPLICIT\nhttps://"))
            receipt = json.loads(Path(result["receipt"]).read_text(encoding="ascii"))
            self.assertEqual(
                receipt["conda_install"]["mamba_cache_policy"],
                "PRIVATE_PER_INSTALL_WORKSPACE_REMOVED_AFTER_CONDA_ATTESTATION",
            )
            self.assertNotIn("root", receipt["conda_install"]["cache"])

    def test_cached_execute_failure_also_removes_private_mamba_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            paths = self._fixture(parent)
            cache_root = parent / "verified-cache"
            cache_root.mkdir(mode=0o755)
            lock_payload = (paths["source"] / installer.CONDA_LOCK_RELATIVE).read_bytes()
            cache_record = {
                "mode": "verified_offline_cache",
                "manifest_basename": installer.CONDA_CACHE_MANIFEST_BASENAME,
                "manifest_sha256": "c" * 64,
                "lock_sha256": hashlib.sha256(lock_payload).hexdigest(),
                "package_count": 71,
                "total_package_bytes": 1_114_157_356,
            }
            with mock.patch.object(installer, "_validate_conda_cache", return_value=cache_record):
                plan = installer.build_install_plan(
                    preset="full",
                    venv=paths["venv"],
                    wheel=paths["wheel"],
                    project=paths["project"],
                    source_root=paths["source"],
                    mamba=paths["mamba"],
                    conda_cache_root=cache_root,
                )
                workspace = paths["venv"].parent / (
                    ".{}.compag-private-mamba-cache-{}".format(
                        paths["venv"].name,
                        installer.RELEASE_VERSION,
                    )
                )
                with self.assertRaisesRegex(installer.InstallerError, "installation command failed"):
                    installer.execute_install(
                        plan,
                        runner=FakeInstallRunner(paths["venv"], fail_command=0),
                    )
            self.assertFalse(workspace.exists())
            self.assertFalse(paths["venv"].exists())

    def test_paths_sidecar_version_and_plan_payload_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            paths = self._fixture(parent)
            with self.assertRaisesRegex(installer.InstallerError, "absolute path"):
                installer.build_install_plan(
                    preset="lite",
                    venv=Path("relative-venv"),
                    wheel=paths["wheel"],
                    project=paths["project"],
                    source_root=paths["source"],
                    mamba=paths["mamba"],
                )
            paths["venv"].mkdir()
            with self.assertRaisesRegex(installer.InstallerError, "must be absent"):
                self._plan(paths)
            paths["venv"].rmdir()
            sidecar = paths["wheel"].with_name(installer.WHEEL_SIDECAR_BASENAME)
            sidecar.write_text(f"{'0' * 64}  {paths['wheel'].name}\n", encoding="ascii")
            with self.assertRaisesRegex(installer.InstallerError, "strict SHA-256"):
                self._plan(paths)

        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory), version="1.3.0")
            with self.assertRaisesRegex(installer.InstallerError, "1.9.3"):
                self._plan(paths)

        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            plan = self._plan(paths)
            plan["_payloads"]["wheel"] = b"substituted wheel"
            with self.assertRaisesRegex(installer.InstallerError, "differ from their identities"):
                installer.execute_install(plan, runner=FakeInstallRunner(paths["venv"]))
            self.assertFalse(paths["venv"].exists())

    def test_read_only_gpu_advisory_reports_inventory_without_threshold_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            calls: list[list[str]] = []

            def runner(
                argv: list[str],
                *,
                check: bool,
                env: dict[str, str],
                text: bool,
                stdout: object,
                stderr: object,
            ) -> subprocess.CompletedProcess[str]:
                calls.append(list(argv))
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    stdout=(
                        "0, NVIDIA RTX 1000 Ada Generation Laptop GPU, 6141, 8.9\n"
                        "1, NVIDIA GeForce RTX 4090, 24564, 8.9\n"
                    ),
                    stderr="",
                )

            before = sorted(path.relative_to(Path(directory)).as_posix() for path in Path(directory).rglob("*"))
            result = installer.recommend_gpu_profile(nvidia_smi=paths["nvidia_smi"], runner=runner)
            after = sorted(path.relative_to(Path(directory)).as_posix() for path in Path(directory).rglob("*"))
            self.assertEqual(before, after)
            self.assertEqual(result["status"], "ADVISORY_NO_AUTOMATIC_SELECTION")
            self.assertFalse(result["automatic_selection"])
            self.assertIsNone(result["suggested_preset"])
            self.assertTrue(result["selection_required"])
            self.assertEqual(set(result["presets"]), set(installer.PRESET_PROFILES))
            self.assertEqual([row["memory_total_mib"] for row in result["devices"]], [6141, 24564])
            self.assertIn("No automatic VRAM threshold", result["claim_boundary"])
            self.assertEqual(calls[0][1:], [
                "--query-gpu=index,name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ])

    def test_cli_requires_a_mode_and_recommend_rejects_mutation(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                installer._parser().parse_args([])
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            code = installer.main(["--recommend", "--execute"])
        self.assertEqual(code, 2)
        self.assertIn("read-only", error.getvalue())


if __name__ == "__main__":
    unittest.main()
